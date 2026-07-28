"""PyTorch implementation of TimeGAN."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .timegan_utils import (
    min_max_scale,
    resolve_device,
    seed_everything,
    sequence_mask,
    to_padded_array,
    validate_module,
)


class RecurrentBlock(nn.Module):
  """Shared recurrent block used by the TimeGAN sub-networks."""

  def __init__(self, input_dim, hidden_dim, output_dim, num_layers, module_name, output_activation):
    super().__init__()
    validate_module(module_name)

    rnn_cls = nn.GRU if module_name == 'gru' else nn.LSTM
    self.rnn = rnn_cls(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=max(num_layers, 1),
        batch_first=True)
    self.layer_norm = nn.LayerNorm(hidden_dim) if module_name == 'lstmLN' else None
    self.projection = nn.Linear(hidden_dim, output_dim)
    self.output_activation = output_activation

  def forward(self, inputs, lengths):
    packed = nn.utils.rnn.pack_padded_sequence(
        inputs,
        lengths.detach().cpu(),
        batch_first=True,
        enforce_sorted=False)
    packed_outputs, _ = self.rnn(packed)
    outputs, _ = nn.utils.rnn.pad_packed_sequence(
        packed_outputs,
        batch_first=True,
        total_length=inputs.size(1))

    if self.layer_norm is not None:
      outputs = self.layer_norm(outputs)

    outputs = self.projection(outputs)
    if self.output_activation == 'sigmoid':
      outputs = torch.sigmoid(outputs)
    return outputs


class TimeGANModel(nn.Module):
  """TimeGAN network collection."""

  def __init__(self, feature_dim, hidden_dim, num_layers, module_name):
    super().__init__()
    supervisor_layers = max(num_layers - 1, 1)

    self.embedder = RecurrentBlock(feature_dim, hidden_dim, hidden_dim, num_layers, module_name, 'sigmoid')
    self.recovery = RecurrentBlock(hidden_dim, hidden_dim, feature_dim, num_layers, module_name, 'sigmoid')
    self.generator = RecurrentBlock(feature_dim, hidden_dim, hidden_dim, num_layers, module_name, 'sigmoid')
    self.supervisor = RecurrentBlock(hidden_dim, hidden_dim, hidden_dim, supervisor_layers, module_name, 'sigmoid')
    self.discriminator = RecurrentBlock(hidden_dim, hidden_dim, 1, num_layers, module_name, None)


def _masked_mean(values, mask):
  return torch.sum(values * mask) / torch.clamp(mask.sum(), min=1.0)


def _masked_mse(x, y, mask):
  return _masked_mean((x - y) ** 2, mask)


def _masked_bce(logits, targets, mask):
  loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
  return _masked_mean(loss, mask)


def _temporal_moments(data, mask):
  count = torch.clamp(mask.sum(dim=0), min=1.0)
  mean = torch.sum(data * mask, dim=0) / count
  variance = torch.sum(((data - mean.unsqueeze(0)) ** 2) * mask, dim=0) / count
  return mean, variance


def _freeze_modules(model, trainable_module_names):
  trainable_module_names = set(trainable_module_names)
  for module_name in ['embedder', 'recovery', 'generator', 'supervisor', 'discriminator']:
    module = getattr(model, module_name)
    requires_grad = module_name in trainable_module_names
    for parameter in module.parameters():
      parameter.requires_grad_(requires_grad)


def _sample_batch(dataset, lengths, batch_size):
  batch_size = min(batch_size, dataset.size(0))
  indices = torch.randperm(dataset.size(0), device=dataset.device)[:batch_size]
  return dataset[indices], lengths[indices]


def _sample_noise(lengths, max_seq_len, z_dim):
  mask = sequence_mask(lengths, max_seq_len, device=lengths.device).unsqueeze(-1)
  return torch.rand((lengths.size(0), max_seq_len, z_dim), device=lengths.device) * mask


def _to_numpy_sequences(data, lengths):
  data = data.detach().cpu().numpy()
  lengths = lengths.detach().cpu().numpy()
  return [data[idx, :length, :] for idx, length in enumerate(lengths)]


def _to_padded_numpy(data):
  """Detach a padded sequence tensor and return it as a NumPy array."""
  return data.detach().cpu().numpy()


def timegan(ori_data, parameters):
  """Train TimeGAN and generate a synthetic dataset.

  Set ``parameters['return_artifacts']`` to additionally return the real
  embeddings after autoencoder pretraining, the final real embeddings, and
  the native generated latent trajectories. The default return value remains
  unchanged for compatibility with the original command-line pipeline.
  """
  device = resolve_device(parameters.get('device', 'auto'))
  seed_everything(parameters.get('seed'))

  if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

  ori_data, ori_time = to_padded_array(ori_data)
  ori_data, min_val, max_val = min_max_scale(ori_data)

  no, max_seq_len, dim = ori_data.shape
  hidden_dim = parameters['hidden_dim']
  num_layers = parameters['num_layer']
  iterations = parameters['iterations']
  batch_size = min(parameters['batch_size'], no)
  module_name = parameters['module']
  z_dim = dim
  gamma = 1
  verbose = parameters.get('verbose', True)
  return_artifacts = parameters.get('return_artifacts', False)

  if verbose:
    print('Using device:', device)

  ori_data = torch.as_tensor(ori_data, dtype=torch.float32, device=device)
  ori_time = torch.as_tensor(ori_time, dtype=torch.long, device=device)

  model = TimeGANModel(dim, hidden_dim, num_layers, module_name).to(device)

  e_params = list(model.embedder.parameters()) + list(model.recovery.parameters())
  g_params = list(model.generator.parameters()) + list(model.supervisor.parameters())
  d_params = list(model.discriminator.parameters())

  e0_optimizer = torch.optim.Adam(e_params)
  e_optimizer = torch.optim.Adam(e_params)
  g_optimizer = torch.optim.Adam(g_params)
  gs_optimizer = torch.optim.Adam(g_params)
  d_optimizer = torch.optim.Adam(d_params)

  if verbose:
    print('Start Embedding Network Training')

  for itt in range(iterations):
    x_mb, t_mb = _sample_batch(ori_data, ori_time, batch_size)
    mask = sequence_mask(t_mb, max_seq_len, device=device).unsqueeze(-1)

    _freeze_modules(model, ['embedder', 'recovery'])
    e0_optimizer.zero_grad(set_to_none=True)

    h = model.embedder(x_mb, t_mb)
    x_tilde = model.recovery(h, t_mb)
    e_loss_t0 = _masked_mse(x_mb, x_tilde, mask)
    e_loss0 = 10.0 * torch.sqrt(e_loss_t0 + 1e-8)
    e_loss0.backward()
    e0_optimizer.step()

    if verbose and itt % 1000 == 0:
      print('step: {}/{} , e_loss: {}'.format(
          itt,
          iterations,
          np.round(torch.sqrt(e_loss_t0.detach()).item(), 4)))

  if verbose:
    print('Finish Embedding Network Training')
    print('Start Training with Supervised Loss Only')

  real_latent_pretrain = None
  if return_artifacts:
    with torch.no_grad():
      real_latent_pretrain = _to_padded_numpy(model.embedder(ori_data, ori_time))

  for itt in range(iterations):
    x_mb, t_mb = _sample_batch(ori_data, ori_time, batch_size)
    z_mb = _sample_noise(t_mb, max_seq_len, z_dim)
    sup_mask = sequence_mask(torch.clamp(t_mb - 1, min=1), max_seq_len - 1, device=device).unsqueeze(-1)

    _freeze_modules(model, ['generator', 'supervisor'])
    gs_optimizer.zero_grad(set_to_none=True)

    h = model.embedder(x_mb, t_mb).detach()
    h_hat_supervise = model.supervisor(h, t_mb)
    g_loss_s = _masked_mse(h[:, 1:, :], h_hat_supervise[:, :-1, :], sup_mask)
    g_loss_s.backward()
    gs_optimizer.step()

    if verbose and itt % 1000 == 0:
      print('step: {}/{} , s_loss: {}'.format(
          itt,
          iterations,
          np.round(torch.sqrt(g_loss_s.detach()).item(), 4)))

  if verbose:
    print('Finish Training with Supervised Loss Only')
    print('Start Joint Training')

  step_d_loss = torch.tensor(0.0, device=device)

  for itt in range(iterations):
    for _ in range(2):
      x_mb, t_mb = _sample_batch(ori_data, ori_time, batch_size)
      z_mb = _sample_noise(t_mb, max_seq_len, z_dim)
      mask = sequence_mask(t_mb, max_seq_len, device=device).unsqueeze(-1)
      sup_mask = sequence_mask(torch.clamp(t_mb - 1, min=1), max_seq_len - 1, device=device).unsqueeze(-1)

      _freeze_modules(model, ['generator', 'supervisor'])
      g_optimizer.zero_grad(set_to_none=True)

      h = model.embedder(x_mb, t_mb).detach()
      e_hat = model.generator(z_mb, t_mb)
      h_hat = model.supervisor(e_hat, t_mb)
      h_hat_supervise = model.supervisor(h, t_mb)
      x_hat = model.recovery(h_hat, t_mb)
      y_fake = model.discriminator(h_hat, t_mb)
      y_fake_e = model.discriminator(e_hat, t_mb)

      g_loss_u = _masked_bce(y_fake, torch.ones_like(y_fake), mask)
      g_loss_u_e = _masked_bce(y_fake_e, torch.ones_like(y_fake_e), mask)
      g_loss_s = _masked_mse(h[:, 1:, :], h_hat_supervise[:, :-1, :], sup_mask)

      mean_hat, var_hat = _temporal_moments(x_hat, mask)
      mean_real, var_real = _temporal_moments(x_mb, mask)
      g_loss_v1 = torch.mean(torch.abs(torch.sqrt(var_hat + 1e-6) - torch.sqrt(var_real + 1e-6)))
      g_loss_v2 = torch.mean(torch.abs(mean_hat - mean_real))
      g_loss_v = g_loss_v1 + g_loss_v2

      g_loss = g_loss_u + gamma * g_loss_u_e + 100.0 * torch.sqrt(g_loss_s + 1e-8) + 100.0 * g_loss_v
      g_loss.backward()
      g_optimizer.step()

      _freeze_modules(model, ['embedder', 'recovery'])
      e_optimizer.zero_grad(set_to_none=True)

      h = model.embedder(x_mb, t_mb)
      x_tilde = model.recovery(h, t_mb)
      h_hat_supervise = model.supervisor(h, t_mb)

      e_loss_t0 = _masked_mse(x_mb, x_tilde, mask)
      g_loss_s_embed = _masked_mse(h[:, 1:, :], h_hat_supervise[:, :-1, :], sup_mask)
      e_loss0 = 10.0 * torch.sqrt(e_loss_t0 + 1e-8)
      e_loss = e_loss0 + 0.1 * g_loss_s_embed
      e_loss.backward()
      e_optimizer.step()

    x_mb, t_mb = _sample_batch(ori_data, ori_time, batch_size)
    z_mb = _sample_noise(t_mb, max_seq_len, z_dim)
    mask = sequence_mask(t_mb, max_seq_len, device=device).unsqueeze(-1)

    _freeze_modules(model, ['discriminator'])
    d_optimizer.zero_grad(set_to_none=True)

    h = model.embedder(x_mb, t_mb).detach()
    e_hat = model.generator(z_mb, t_mb).detach()
    h_hat = model.supervisor(e_hat, t_mb).detach()
    y_real = model.discriminator(h, t_mb)
    y_fake = model.discriminator(h_hat, t_mb)
    y_fake_e = model.discriminator(e_hat, t_mb)

    d_loss_real = _masked_bce(y_real, torch.ones_like(y_real), mask)
    d_loss_fake = _masked_bce(y_fake, torch.zeros_like(y_fake), mask)
    d_loss_fake_e = _masked_bce(y_fake_e, torch.zeros_like(y_fake_e), mask)
    d_loss = d_loss_real + d_loss_fake + gamma * d_loss_fake_e

    if d_loss.item() > 0.15:
      d_loss.backward()
      d_optimizer.step()
      step_d_loss = d_loss.detach()

    if verbose and itt % 1000 == 0:
      print(
          'step: {}/{} , d_loss: {} , g_loss_u: {} , g_loss_s: {} , g_loss_v: {} , e_loss_t0: {}'.format(
              itt,
              iterations,
              np.round(step_d_loss.item(), 4),
              np.round(g_loss_u.detach().item(), 4),
              np.round(torch.sqrt(g_loss_s.detach()).item(), 4),
              np.round(g_loss_v.detach().item(), 4),
              np.round(torch.sqrt(e_loss_t0.detach()).item(), 4)))

  if verbose:
    print('Finish Joint Training')

  with torch.no_grad():
    z_mb = _sample_noise(ori_time, max_seq_len, z_dim)
    e_hat = model.generator(z_mb, ori_time)
    h_hat = model.supervisor(e_hat, ori_time)
    x_hat = model.recovery(h_hat, ori_time)

  generated_data = np.asarray(_to_numpy_sequences(x_hat, ori_time), dtype=object)
  generated_data = [sample * max_val + min_val for sample in generated_data]

  if return_artifacts:
    with torch.no_grad():
      real_latent = model.embedder(ori_data, ori_time)

    artifacts = {
        'real_latent_pretrain': real_latent_pretrain,
        'real_latent': _to_padded_numpy(real_latent),
        'synthetic_latent': _to_padded_numpy(h_hat),
        'lengths': ori_time.detach().cpu().numpy(),
        'data_min': min_val,
        'data_range': max_val,
    }
    return generated_data, artifacts

  return generated_data

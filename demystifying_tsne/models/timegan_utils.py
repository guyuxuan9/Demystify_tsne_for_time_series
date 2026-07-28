"""Utilities used by the bundled TimeGAN implementation."""

import numpy as np
import torch


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
  """Divide train and test data for both original and synthetic data."""
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no * train_rate)]
  test_idx = idx[int(no * train_rate):]

  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]

  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no * train_rate)]
  test_idx = idx[int(no * train_rate):]

  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]

  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
  """Return sequence lengths and the maximum sequence length."""
  time = []
  max_seq_len = 0
  for sample in data:
    seq_len = len(sample[:, 0])
    max_seq_len = max(max_seq_len, seq_len)
    time.append(seq_len)
  return time, max_seq_len


def validate_module(module_name):
  """Validate and normalize the supported recurrent module name."""
  supported_modules = {'gru', 'lstm', 'lstmLN'}
  if module_name not in supported_modules:
    raise ValueError('module must be one of gru, lstm, or lstmLN')
  return module_name


def to_padded_array(data, max_seq_len=None, dtype=np.float32):
  """Convert a sequence list into a padded NumPy array."""
  lengths, inferred_max_seq_len = extract_time(data)
  max_seq_len = inferred_max_seq_len if max_seq_len is None else max_seq_len
  dim = data[0].shape[1]

  padded = np.zeros((len(data), max_seq_len, dim), dtype=dtype)
  for idx, sample in enumerate(data):
    sample = np.asarray(sample, dtype=dtype)
    seq_len = min(sample.shape[0], max_seq_len)
    padded[idx, :seq_len, :] = sample[:seq_len]

  return padded, np.asarray(lengths, dtype=np.int64)


def min_max_scale(data):
  """Min-max normalize a padded 3D array feature-wise."""
  min_val = np.min(data, axis=(0, 1))
  shifted_data = data - min_val

  max_val = np.max(shifted_data, axis=(0, 1))
  norm_data = shifted_data / (max_val + 1e-7)

  return norm_data.astype(np.float32), min_val.astype(np.float32), max_val.astype(np.float32)


def random_generator(batch_size, z_dim, lengths, max_seq_len):
  """Generate padded random noise for variable-length batches."""
  lengths = np.asarray(lengths, dtype=np.int64)
  z_mb = np.random.uniform(0.0, 1.0, size=(batch_size, max_seq_len, z_dim)).astype(np.float32)
  mask = (np.arange(max_seq_len)[None, :] < lengths[:, None])[..., None]
  return z_mb * mask


def batch_generator(data, time, batch_size):
  """Sample a random mini-batch."""
  no = len(data)
  batch_size = min(batch_size, no)
  idx = np.random.permutation(no)[:batch_size]

  if isinstance(data, np.ndarray):
    x_mb = data[idx]
  else:
    x_mb = [data[i] for i in idx]
  t_mb = [time[i] for i in idx]

  return x_mb, t_mb


def resolve_device(device_name='auto', prefer_mps=False):
  """Resolve a requested device name into a torch.device."""
  if device_name in (None, 'auto'):
    if torch.cuda.is_available():
      return torch.device('cuda')
    if prefer_mps and torch.backends.mps.is_available():
      return torch.device('mps')
    return torch.device('cpu')

  device = torch.device(device_name)
  if device.type == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but no CUDA device is available.')
  if device.type == 'mps' and not torch.backends.mps.is_available():
    raise RuntimeError(
        'MPS was requested but this PyTorch build cannot access Apple Metal.')
  return device


def seed_everything(seed):
  """Seed NumPy and PyTorch for reproducible experiments."""
  if seed is None:
    return

  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def sequence_mask(lengths, max_seq_len, device=None, dtype=torch.float32):
  """Create a broadcast-friendly mask from sequence lengths."""
  if not torch.is_tensor(lengths):
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
  else:
    lengths = lengths.to(device=device)

  steps = torch.arange(max_seq_len, device=lengths.device).unsqueeze(0)
  mask = steps < lengths.unsqueeze(1)
  return mask.to(dtype=dtype)

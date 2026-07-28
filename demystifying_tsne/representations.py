"""Feature representations shared by the public-data t-SNE experiments."""

import hashlib
import random

import numpy as np
from scipy.signal import welch

from .models.timegan_utils import resolve_device


def psd_features(sequences):
  """Return log-power spectral density features for every channel."""
  sequences = np.asarray(sequences)
  if sequences.ndim != 3:
    raise ValueError('sequences must have shape (samples, timesteps, channels)')
  _, power = welch(
      sequences,
      axis=1,
      nperseg=sequences.shape[1],
  )
  return np.log1p(power).reshape(len(sequences), -1).astype(np.float32)


def catch22_features(sequences):
  """Compute the 22 canonical Catch22 features independently per channel."""
  try:
    from pycatch22 import catch22_all
  except ImportError as error:
    raise ImportError(
        'Catch22 features require pycatch22; install requirements.txt') from error

  sequences = np.asarray(sequences)
  if sequences.ndim != 3:
    raise ValueError('sequences must have shape (samples, timesteps, channels)')

  features = np.empty(
      (len(sequences), 22 * sequences.shape[2]), dtype=np.float64)
  for sample_index, sample in enumerate(sequences):
    for channel_index in range(sequences.shape[2]):
      start = 22 * channel_index
      result = catch22_all(sample[:, channel_index].astype(float).tolist())
      features[sample_index, start:start + 22] = result['values']

  # A few Catch22 statistics can be undefined for short or constant windows.
  # Median imputation preserves the usable features; all-invalid columns become 0
  # and are subsequently removed by the experiment's standardization step.
  features[~np.isfinite(features)] = np.nan
  with np.errstate(all='ignore'):
    medians = np.nanmedian(features, axis=0)
  medians = np.nan_to_num(medians, nan=0.0)
  missing = np.where(np.isnan(features))
  features[missing] = medians[missing[1]]
  return features.astype(np.float32)


def _array_fingerprint(values):
  values = np.ascontiguousarray(values)
  digest = hashlib.sha256()
  digest.update(str(values.shape).encode())
  digest.update(str(values.dtype).encode())
  digest.update(values.view(np.uint8))
  return digest.hexdigest()


def _fit_trep(sequences, args):
  from .vendor.trep import TRep
  try:
    import torch
  except ImportError as error:
    raise ImportError('T-Rep requires torch; install project dependencies') from error

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

  device = str(resolve_device(args.device, prefer_mps=True))
  print('Using device for T-Rep:', device)
  model = TRep(
      input_dims=sequences.shape[2],
      output_dims=args.trep_output_dim,
      time_embedding='t2v_sin',
      device=device,
      batch_size=args.trep_batch_size,
  )
  model.fit(
      sequences.astype(np.float32),
      n_iters=args.trep_iterations,
      verbose=1,
  )
  return model


def train_or_load_trep(sequences, args, cache_prefix):
  """Train T-Rep without labels and return full-series embeddings."""
  cache = args.cache_dir / '{}_trep_d{}_i{}_s{}.npz'.format(
      cache_prefix,
      args.trep_output_dim,
      args.trep_iterations,
      args.seed,
  )
  fingerprint = _array_fingerprint(sequences)
  if args.reuse_cache and cache.exists():
    with np.load(cache) as saved:
      if (
          'fingerprint' in saved and
          str(saved['fingerprint'].item()) == fingerprint
      ):
        print('Loading cached T-Rep representation:', cache)
        return saved['embeddings']
    print('Ignoring incompatible T-Rep cache:', cache)

  print('Training T-Rep; no reusable cache was found.')
  model = _fit_trep(sequences, args)
  embeddings = model.encode(
      sequences.astype(np.float32),
      encoding_window='full_series',
  ).astype(np.float32)

  if args.save_cache:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        fingerprint=fingerprint,
        embeddings=embeddings,
    )
    print('Saved T-Rep representation cache:', cache)
  return embeddings


def train_or_load_trep_temporal(sequences, args, cache_prefix):
  """Train T-Rep without labels and preserve timestamp embeddings."""
  cache = args.cache_dir / '{}_trep_temporal_d{}_i{}_s{}.npz'.format(
      cache_prefix,
      args.trep_output_dim,
      args.trep_iterations,
      args.seed,
  )
  fingerprint = _array_fingerprint(sequences)
  if args.reuse_cache and cache.exists():
    with np.load(cache) as saved:
      if (
          'fingerprint' in saved and
          str(saved['fingerprint'].item()) == fingerprint
      ):
        print('Loading cached T-Rep temporal representation:', cache)
        return saved['temporal_embedding']
    print('Ignoring incompatible T-Rep cache:', cache)

  print('Training T-Rep on {} unlabeled windows.'.format(len(sequences)))
  model = _fit_trep(sequences, args)
  embedding = model.encode(
      sequences.astype(np.float32),
      encoding_window=None,
  ).astype(np.float32)
  if args.save_cache:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        fingerprint=fingerprint,
        temporal_embedding=embedding,
    )
    print('Saved T-Rep temporal representation:', cache)
  return embedding


def add_representation_arguments(parser):
  """Add command-line options shared by the T-Rep experiments."""
  parser.add_argument('--trep-output-dim', type=int, default=128)
  parser.add_argument('--trep-iterations', type=int, default=200)
  parser.add_argument('--trep-batch-size', type=int, default=16)

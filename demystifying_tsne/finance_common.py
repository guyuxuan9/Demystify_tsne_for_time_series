"""Data and model helpers shared by the finance experiments."""

import hashlib
import re

import numpy as np
import pandas as pd

from .models.timegan import timegan


def ticker_slug(ticker):
  return re.sub(r'[^a-z0-9]+', '_', ticker.lower()).strip('_')


def download_history(args):
  cache = (
      args.data_dir / 'yahoo_finance' /
      '{}_yahoo_{}_{}.csv'.format(
          ticker_slug(args.ticker).upper(), args.start_date, args.end_date))
  if cache.exists() and not args.refresh_yahoo:
    return pd.read_csv(cache, parse_dates=['Date']), cache

  try:
    import yfinance as yf
  except ImportError as error:
    raise ImportError(
        'Yahoo download requires the yfinance project dependency') from error

  frame = yf.download(
      args.ticker,
      start=args.start_date,
      end=args.end_date,
      interval='1d',
      auto_adjust=False,
      progress=False,
      threads=False,
      timeout=30,
  )
  if frame.empty:
    raise RuntimeError(
        'Yahoo Finance returned no data for {}'.format(args.ticker))
  if isinstance(frame.columns, pd.MultiIndex):
    frame.columns = frame.columns.get_level_values(0)
  frame = frame.reset_index()
  cache.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(cache, index=False)
  return frame, cache


def array_fingerprint(values):
  values = np.ascontiguousarray(values)
  digest = hashlib.sha256()
  digest.update(str(values.shape).encode())
  digest.update(str(values.dtype).encode())
  digest.update(values.view(np.uint8))
  return digest.hexdigest()


def train_or_load_timegan(sequences, args, prefix):
  cache = args.cache_dir / '{}_timegan_h{}_l{}_i{}_s{}.npz'.format(
      prefix,
      args.hidden_dim,
      args.num_layers,
      args.iterations,
      args.seed,
  )
  fingerprint = array_fingerprint(sequences)
  if args.reuse_cache and cache.exists():
    with np.load(cache) as saved:
      if (
          'fingerprint' in saved and
          str(saved['fingerprint'].item()) == fingerprint
      ):
        print('Loading cached TimeGAN representation:', cache)
        return saved['temporal_embedding']
    print('Ignoring incompatible TimeGAN cache:', cache)

  print('Training TimeGAN on {} unlabeled windows.'.format(len(sequences)))
  _, artifacts = timegan(
      [sequence for sequence in sequences],
      {
          'module': 'gru',
          'hidden_dim': args.hidden_dim,
          'num_layer': args.num_layers,
          'iterations': args.iterations,
          'batch_size': args.batch_size,
          'device': args.device,
          'seed': args.seed,
          'verbose': True,
          'return_artifacts': True,
      })
  embedding = artifacts['real_latent'].astype(np.float32)
  if args.save_cache:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        fingerprint=fingerprint,
        temporal_embedding=embedding,
    )
    print('Saved TimeGAN representation:', cache)
  return embedding


def temporal_summary(embedding):
  centered = embedding - np.mean(embedding, axis=1, keepdims=True)
  lag1 = np.sum(centered[:, :-1] * centered[:, 1:], axis=1)
  lag1 /= np.sqrt(
      np.sum(centered[:, :-1] ** 2, axis=1) *
      np.sum(centered[:, 1:] ** 2, axis=1)) + 1e-8

  time = (
      np.arange(embedding.shape[1], dtype=np.float32) -
      (embedding.shape[1] - 1) / 2)
  slope = np.sum(centered * time[None, :, None], axis=1)
  slope /= np.sum(time ** 2)
  difference_rms = np.sqrt(np.mean(np.diff(embedding, axis=1) ** 2, axis=1))
  return np.concatenate([
      np.mean(embedding, axis=1),
      np.std(embedding, axis=1),
      lag1,
      slope,
      difference_rms,
  ], axis=1).astype(np.float32)

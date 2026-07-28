"""Shared t-SNE evaluation helpers."""

import numpy as np
from scipy.signal import correlate
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors


def standardize(values):
  """Scale feature columns across observations, when explicitly requested."""
  values = np.nan_to_num(np.asarray(values), copy=True)
  mean = np.mean(values, axis=0, keepdims=True)
  std = np.std(values, axis=0, keepdims=True)
  keep = std.ravel() > 1e-7
  if not np.any(keep):
    raise ValueError('representation has no non-constant dimensions')
  return ((values[:, keep] - mean[:, keep]) / std[:, keep]).astype(
      np.float32)


def normalize_over_time(sequences):
  """Z-score each recording/channel independently along time (axis 1)."""
  sequences = np.asarray(sequences)
  mean = np.mean(sequences, axis=1, keepdims=True)
  std = np.std(sequences, axis=1, keepdims=True)
  return (sequences - mean) / np.where(std > 0, std, 1)


def neighbor_purity(values, labels, k=10, metric='euclidean'):
  model = NearestNeighbors(n_neighbors=k, metric=metric).fit(values)
  indices = model.kneighbors(return_distance=False)
  return float(np.mean(labels[indices] == labels[:, None]))


def time_series_distances(values, metric):
  """Compare univariate sequences; DTW retains amplitude, SBD centers/scales."""
  values = np.ascontiguousarray(values, dtype=np.float64)
  if metric == 'dtw':
    try:
      from dtaidistance import dtw
    except ImportError as error:
      raise ImportError(
          'DTW requires dtaidistance; install requirements.txt') from error
    # Figure 3: inclusive Sakoe–Chiba band |i-j| <= 100 samples.
    # dtaidistance uses the strict bound |i-j| < window for equal lengths.
    return dtw.distance_matrix_fast(values, window=101, parallel=False)
  if metric != 'sbd':
    raise ValueError('unknown time-series distance: {}'.format(metric))

  values = values - values.mean(axis=1, keepdims=True)
  norms = np.linalg.norm(values, axis=1, keepdims=True)
  values = values / np.where(norms > 0, norms, 1)

  def sbd(left, right):
    if not np.any(left) and not np.any(right):
      return 0.0
    return float(np.clip(1 - correlate(left, right, mode='full').max(), 0, 2))

  distances = pairwise_distances(values, metric=sbd)
  np.fill_diagonal(distances, 0)
  return distances


def embed_and_score(values, labels, perplexity, seed, metric='euclidean',
                    standardize_input=True, init=None):
  values = (standardize(values)
            if metric == 'euclidean' and standardize_input else np.asarray(values))
  if len(values) < 22:
    raise ValueError('at least 22 observations are required for k=10 metrics')
  distances = (
      values if metric == 'euclidean' else time_series_distances(values, metric))
  input_metric = 'euclidean' if metric == 'euclidean' else 'precomputed'
  if init is None:
    init = 'pca' if metric == 'euclidean' else 'random'
  coordinates = TSNE(
      n_components=2,
      metric=input_metric,
      perplexity=perplexity,
      init=init,
      learning_rate='auto',
      max_iter=1500,
      random_state=seed,
  ).fit_transform(distances)
  metrics = {
      'distance': metric,
      'dimensions': int(values.shape[1]),
      'neighbor_purity_10': neighbor_purity(distances, labels, metric=input_metric),
      'silhouette': float(
          silhouette_score(distances, labels, metric=input_metric)),
      'tsne_trustworthiness_10': float(
          trustworthiness(
              distances, coordinates, n_neighbors=10, metric=input_metric)),
      'tsne_neighbor_purity_10': neighbor_purity(coordinates, labels),
      'tsne_silhouette': float(silhouette_score(coordinates, labels)),
  }
  return coordinates, metrics

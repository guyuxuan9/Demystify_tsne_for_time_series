"""Visualize daily, weekly, and monthly stock-return horizons with t-SNE.

The task is deliberately simple: each sequence contains 64 consecutive values
from SPY (an ETF that tracks the S&P 500).  A value is the market's change over
the preceding 1, 5, or 20 trading days--approximately one day, one week, or one
month.  Longer-horizon changes overlap more and therefore make smoother lines.

Each horizon is standardized separately before windows are made.  Consequently,
the representations must recognize temporal smoothness instead of relying on
the fact that longer-horizon changes are normally larger.  Class labels are used
only for plotting and evaluation, never for T-Rep or TimeGAN training.

Example:
    demystify-finance-horizons --device mps
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import embed_and_score
from .finance_common import (
    download_history,
    temporal_summary,
    ticker_slug,
    train_or_load_timegan,
)
from .representations import (
    add_representation_arguments,
    catch22_features,
    psd_features,
    train_or_load_trep_temporal,
)


DEFAULT_HORIZONS = [1, 5, 20]
DEFAULT_LABEL_NAMES = [
    '1-day change',
    '1-week change (5 days)',
    '1-month change (20 days)',
]
COLORS = ['#0072B2', '#E69F00', '#009E73']
TITLE_FONT_SIZE = 22
AXIS_LABEL_FONT_SIZE = 26
TICK_LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
PANEL_LABEL_FONT_SIZE = 22


def make_horizon_windows(frame, horizons, window_length):
  """Return aligned windows of trailing multi-day log price changes."""
  frame = (
      frame[['Date', 'Adj Close']]
      .dropna()
      .drop_duplicates(subset='Date')
      .sort_values('Date')
      .reset_index(drop=True))
  one_day_returns = np.diff(
      np.log(frame['Adj Close'].to_numpy(dtype=np.float64)))
  # A handful of bad ticks should not define the learned representation.
  lower, upper = np.quantile(one_day_returns, [0.002, 0.998])
  one_day_returns = np.clip(one_day_returns, lower, upper)

  maximum_horizon = max(horizons)
  horizon_series = []
  for horizon in horizons:
    values = np.convolve(
        one_day_returns, np.ones(horizon, dtype=np.float64), mode='valid')
    # Align every horizon at the first date for which the longest lookback is
    # available.  This gives every class the same dates and number of windows.
    values = values[maximum_horizon - horizon:]
    values = (values - np.mean(values)) / (np.std(values) + 1e-12)
    horizon_series.append(values.astype(np.float32))

  starts = range(
      0, len(horizon_series[0]) - window_length + 1, window_length)
  starts = list(starts)
  if not starts:
    raise ValueError('not enough price history for one complete window')

  windows = []
  labels = []
  start_indices = []
  for label, values in enumerate(horizon_series):
    for start in starts:
      windows.append(values[start:start + window_length, None])
      labels.append(label)
      start_indices.append(start)
  return (
      np.stack(windows).astype(np.float32),
      np.asarray(labels, dtype=np.int64),
      np.asarray(start_indices, dtype=np.int64),
  )


def label_names(horizons):
  if horizons == DEFAULT_HORIZONS:
    return DEFAULT_LABEL_NAMES
  return ['{}-day change'.format(horizon) for horizon in horizons]


def cache_prefix(args):
  horizon_text = '-'.join(str(value) for value in args.horizons)
  return '{}_return_horizons_{}_w{}'.format(
      ticker_slug(args.ticker), horizon_text, args.window_length)


def summarize_temporal_embedding(embedding):
  """Pool a learned timestamp embedding while retaining its smoothness."""
  return np.concatenate([
      temporal_summary(embedding),
      psd_features(embedding),
  ], axis=1).astype(np.float32)


def plot(windows, coordinates_by_name, labels, names, output_path):
  """Plot task examples above all t-SNE views in one figure."""
  fig = plt.figure(figsize=(22.5, 8.6))
  grid = fig.add_gridspec(
      2,
      15,
      height_ratios=[0.82, 1.0],
      left=0.04,
      right=0.99,
      top=0.90,
      bottom=0.12,
      hspace=0.52,
      wspace=0.62,
  )

  example_axes = []
  for index in range(len(names)):
    kwargs = {'sharey': example_axes[0]} if example_axes else {}
    example_axes.append(
        fig.add_subplot(grid[0, 5 * index:5 * (index + 1)], **kwargs))

  example_number = min(40, np.min(np.bincount(labels)) - 1)
  for label, (ax, name, color) in enumerate(
      zip(example_axes, names, COLORS)):
    sequence = windows[labels == label][example_number, :, 0]
    ax.plot(np.arange(1, len(sequence) + 1), sequence, color=color, lw=1.8)
    ax.axhline(0, color='#777777', lw=0.7, alpha=0.55)
    ax.set_title(name, fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_xlabel('trading day in the sequence', fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(alpha=0.16, linewidth=0.6)
    if label > 0:
      ax.tick_params(labelleft=False)
  example_axes[0].set_ylabel('Normalized Return', fontsize=AXIS_LABEL_FONT_SIZE)

  tsne_axes = [
      fig.add_subplot(grid[1, 3 * index:3 * (index + 1)])
      for index in range(len(coordinates_by_name))
  ]
  for ax, (title, coordinates) in zip(
      tsne_axes, coordinates_by_name.items()):
    for label, (name, color) in enumerate(zip(names, COLORS)):
      selected = labels == label
      ax.scatter(
          coordinates[selected, 0],
          coordinates[selected, 1],
          s=20,
          alpha=0.68,
          linewidths=0,
          color=color,
          label=name,
          rasterized=True,
      )
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('t-SNE 2', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])

  handles, legend_labels = tsne_axes[0].get_legend_handles_labels()
  fig.legend(
      handles,
      legend_labels,
      loc='lower center',
      bbox_to_anchor=(0.5, -0.02),
      ncol=len(names),
      fontsize=LEGEND_FONT_SIZE,
      markerscale=1.5,
      frameon=True,
      framealpha=0.9,
  )
  fig.text(
      0.04,
      0.91,
      '(a)',
      ha='left',
      va='bottom',
      fontsize=PANEL_LABEL_FONT_SIZE,
      fontweight='bold',
  )
  fig.text(
      0.04,
      0.47,
      '(b)',
      ha='left',
      va='bottom',
      fontsize=PANEL_LABEL_FONT_SIZE,
      fontweight='bold',
  )
  fig.savefig(output_path, bbox_inches='tight', facecolor='white')
  plt.close(fig)


def parse_args(argv=None):
  project_dir = Path(__file__).resolve().parents[1]
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--data-dir', type=Path, default=project_dir / 'data')
  parser.add_argument(
      '--figure-dir', type=Path, default=project_dir / 'outputs')
  parser.add_argument(
      '--cache-dir',
      type=Path,
      default=project_dir / '.cache' / 'representations',
  )
  parser.add_argument('--ticker', default='SPY')
  parser.add_argument('--start-date', default='2000-01-01')
  parser.add_argument('--end-date', default='2026-07-18')
  parser.add_argument('--horizons', nargs='+', type=int, default=DEFAULT_HORIZONS)
  parser.add_argument('--window-length', type=int, default=64)
  parser.add_argument('--perplexity', type=int, default=30)
  parser.add_argument(
      '--device',
      default='auto',
      help=(
          'auto selects a backend separately for TimeGAN and T-Rep; '
          'cpu, cuda, cuda:N, and mps override it'),
  )
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--hidden-dim', type=int, default=12)
  parser.add_argument('--num-layers', type=int, default=2)
  parser.add_argument('--iterations', type=int, default=700)
  parser.add_argument('--batch-size', type=int, default=128)
  add_representation_arguments(parser)
  parser.set_defaults(trep_batch_size=32)
  parser.add_argument('--refresh-yahoo', action='store_true')

  cache_mode = parser.add_mutually_exclusive_group()
  cache_mode.add_argument(
      '--reuse-cache', dest='reuse_cache', action='store_true', default=True)
  cache_mode.add_argument('--retrain', dest='reuse_cache', action='store_false')
  save_mode = parser.add_mutually_exclusive_group()
  save_mode.add_argument(
      '--save-cache', dest='save_cache', action='store_true', default=True)
  save_mode.add_argument('--no-save-cache', dest='save_cache', action='store_false')
  args = parser.parse_args(argv)

  if len(args.horizons) < 2:
    parser.error('--horizons must contain at least two positive values')
  if any(value < 1 for value in args.horizons):
    parser.error('--horizons must contain only positive values')
  if len(set(args.horizons)) != len(args.horizons):
    parser.error('--horizons must not contain duplicates')
  if args.horizons != sorted(args.horizons):
    parser.error('--horizons must be in increasing order')
  if len(args.horizons) > len(COLORS):
    parser.error('at most {} horizons are supported'.format(len(COLORS)))
  if args.window_length < max(args.horizons):
    parser.error('--window-length must be at least the largest horizon')
  return args


def main(argv=None):
  args = parse_args(argv)
  args.figure_dir.mkdir(parents=True, exist_ok=True)
  frame, source_cache = download_history(args)
  windows, labels, start_indices = make_horizon_windows(
      frame, args.horizons, args.window_length)
  names = label_names(args.horizons)
  counts = np.bincount(labels, minlength=len(names))
  if np.min(counts) < 30:
    raise ValueError(
        'too few windows per horizon; use more history or a shorter window')

  prefix = cache_prefix(args)
  # The models see every sequence but never see its horizon label.
  trep_temporal = train_or_load_trep_temporal(windows, args, prefix)
  timegan_temporal = train_or_load_timegan(windows, args, prefix)
  representations = {
      'Raw sequence': windows.reshape(len(windows), -1),
      'Power spectral density': psd_features(windows),
      'Catch22 features': catch22_features(windows),
      'T-Rep': summarize_temporal_embedding(trep_temporal),
      'TimeGAN': summarize_temporal_embedding(
          timegan_temporal),
  }

  coordinates_by_name = {}
  metrics = {}
  for name, values in representations.items():
    coordinates_by_name[name], metrics[name] = embed_and_score(
        values, labels, args.perplexity, args.seed)

  output_stem = '{}_return_horizon_tsne'.format(ticker_slug(args.ticker))
  figure_path = args.figure_dir / '{}.pdf'.format(output_stem)
  metrics_path = args.figure_dir / '{}_metrics.json'.format(output_stem)
  coordinates_path = args.figure_dir / '{}_coordinates.npz'.format(output_stem)
  plot(
      windows,
      coordinates_by_name,
      labels,
      names,
      figure_path,
  )

  with metrics_path.open('w', encoding='utf-8') as output:
    json.dump(
        {
            'task': 'Recognize the trailing return horizon: {}.'.format(
                ', '.join(str(value) for value in args.horizons)),
            'ticker': args.ticker,
            'source_cache': str(source_cache),
            'start_date': args.start_date,
            'end_date': args.end_date,
            'horizons': args.horizons,
            'window_length': args.window_length,
            'num_windows': int(len(windows)),
            'class_counts': {
                name: int(count) for name, count in zip(names, counts)
            },
            'random_class_purity_baseline': float(sum(
                (count / np.sum(counts)) ** 2 for count in counts)),
            'normalization': (
                'Each full horizon series is independently standardized to '
                'zero mean and unit variance before windowing.'),
            'temporal_embedding_pooling': (
                'Moment, lag-correlation, trend, difference, and spectral '
                'summaries of the label-free timestamp embeddings.'),
            'metrics': metrics,
        },
        output,
        indent=2,
    )
  np.savez_compressed(
      coordinates_path,
      labels=labels,
      horizons=np.asarray(args.horizons),
      start_indices=start_indices,
      **{
          name.lower().replace(' ', '_').replace('-', ''): coordinates
          for name, coordinates in coordinates_by_name.items()
      },
  )

  print('Windows per class:', dict(zip(names, counts.tolist())))
  print('Random-label purity baseline: {:.3f}'.format(
      sum((count / np.sum(counts)) ** 2 for count in counts)))
  print('Saved combined task and t-SNE figure:', figure_path)
  print('Saved metrics:', metrics_path)
  print('Saved coordinates:', coordinates_path)
  for name, values in metrics.items():
    print('{}:'.format(name), values)


if __name__ == '__main__':
  main()

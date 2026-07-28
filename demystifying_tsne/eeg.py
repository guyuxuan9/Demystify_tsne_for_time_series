"""Compare five label-free t-SNE representations on public seizure EEG.

The class labels are used only for evaluation and plot colors, never for
TimeGAN or T-Rep training.

Example:
    demystify-eeg --device mps
"""

import argparse
import json
from pathlib import Path
import urllib.request
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import embed_and_score, normalize_over_time, standardize
from .models.timegan import timegan
from .representations import (
    add_representation_arguments,
    catch22_features,
    psd_features,
    train_or_load_trep,
)


DATA_URL = (
    'https://www.timeseriesclassification.com/'
    'aeon-toolkit/EpilepticSeizures.zip')
COLORS = ['#5e8eff', '#1a1c5a']
LABEL_NAMES = ['Seizure', 'Non-Seizure']
DISTANCES = [
    ('euclidean', 'Euclidean distance'),
    ('dtw', 'DTW'),
    ('sbd', 'SBD'),
]
TITLE_FONTSIZE = 22
ANNOTATION_FONTSIZE = 22
LABEL_FONTSIZE = 24
LEGEND_FONTSIZE = 22


def download_dataset(data_dir):
  train_file = (
      data_dir / 'EpilepticSeizures' / 'EpilepticSeizures' /
      'EpilepticSeizures_TRAIN.ts')
  if train_file.exists():
    return train_file.parent

  data_dir.mkdir(parents=True, exist_ok=True)
  archive = data_dir / 'EpilepticSeizures.zip'
  urllib.request.urlretrieve(DATA_URL, archive)
  target = data_dir / 'EpilepticSeizures'
  target.mkdir(parents=True, exist_ok=True)

  with zipfile.ZipFile(archive) as zipped:
    root = target.resolve()
    for member in zipped.infolist():
      destination = (target / member.filename).resolve()
      if destination != root and root not in destination.parents:
        raise ValueError('unsafe ZIP path: {}'.format(member.filename))
    zipped.extractall(target)
  return train_file.parent


def read_ts(path):
  sequences = []
  labels = []
  in_data = False
  for line in path.read_text().splitlines():
    line = line.strip()
    if not in_data:
      in_data = line.lower() == '@data'
    elif line:
      values, label = line.rsplit(':', 1)
      sequences.append(np.fromstring(values, sep=',', dtype=np.float32))
      labels.append(int(label))
  return np.stack(sequences)[..., None], np.asarray(labels)


def load_balanced_eeg(data_dir, samples_per_class, seed, normalize=True):
  source = download_dataset(data_dir)
  train_x, train_y = read_ts(source / 'EpilepticSeizures_TRAIN.ts')
  test_x, test_y = read_ts(source / 'EpilepticSeizures_TEST.ts')
  sequences = np.concatenate([train_x, test_x])
  labels = np.concatenate([train_y, test_y])

  rng = np.random.default_rng(seed)
  selected = []
  for label in np.unique(labels):
    candidates = np.flatnonzero(labels == label)
    selected.extend(
        rng.choice(candidates, size=samples_per_class, replace=False))
  selected = np.asarray(selected)
  rng.shuffle(selected)
  sequences = sequences[selected]
  if normalize:
    sequences = normalize_over_time(sequences)
  return sequences, labels[selected]


def train_or_load(sequences, labels, args):
  cache = args.cache_dir / 'epileptic_seizures_timegan_artifacts_s{}.npz'.format(
      args.seed)
  if args.reuse_cache and cache.exists():
    with np.load(cache) as saved:
      if 'sequences' in saved and np.array_equal(saved['sequences'], sequences):
        print('Loading cached TimeGAN representation:', cache)
        return saved['latent_joint']
    print('Ignoring incompatible TimeGAN cache:', cache)

  print('Training TimeGAN; no reusable cache was found.')
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
  if args.save_cache:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        sequences=sequences,
        labels=labels,
        latent_joint=artifacts['real_latent'],
    )
    print('Saved TimeGAN representation cache:', cache)
  return artifacts['real_latent']


def plot(sequences, coordinates_by_metric, labels, output_dir,
         representative_sequences=None,
         amplitude_label='Normalized Amplitude',
         figure_filename='eeg_normalized_representation_comparison_across_time.pdf'):
  """Plot three examples from each EEG class above all t-SNE views."""
  fig = plt.figure(figsize=(22.5, 20.0))
  grid = fig.add_gridspec(
      3,
      10,
      height_ratios=[1.86, 1.0, 1.0],
      left=0.06,
      right=0.99,
      top=0.96,
      bottom=0.065,
      hspace=0.55,
      wspace=0.62,
  )
  example_grid = grid[0, :].subgridspec(3, 2, hspace=0.28, wspace=0.08)

  examples = sequences[:, :, 0]
  if representative_sequences is None:
    representative_sequences = sequences
  selection_examples = representative_sequences[:, :, 0]
  changes = np.diff(selection_examples, axis=1)
  example_features = standardize(np.column_stack([
      np.std(selection_examples, axis=1),  # Amplitude fluctuations.
      np.max(np.abs(changes), axis=1),  # Peak sharpness.
      np.std(changes, axis=1),        # Temporal variability.
  ]))
  representative_indices = []
  for label in range(len(LABEL_NAMES)):
    class_indices = np.flatnonzero(labels == label)
    features = example_features[class_indices]
    # Select the three observed waveforms closest to the class median.
    distances = np.sum((features - np.median(features, axis=0)) ** 2, axis=1)
    representative_indices.append(
        class_indices[np.argsort(distances, kind='stable')[:3]])

  example_axes = []
  for example_rank in range(3):
    pair_axes = [
        fig.add_subplot(example_grid[example_rank, 0]),
        fig.add_subplot(example_grid[example_rank, 1]),
    ]
    pair_axes[1].sharey(pair_axes[0])
    example_axes.extend(pair_axes)
    for label, (ax, name, color) in enumerate(
        zip(pair_axes, LABEL_NAMES, COLORS)):
      sequence = examples[representative_indices[label][example_rank]]
      ax.plot(
          np.arange(1, len(sequence) + 1), sequence,
          color=color, linewidth=1.8)
      if example_rank == 0:
        ax.set_title(name, fontsize=TITLE_FONTSIZE, fontweight='normal')
      if example_rank == 2:
        ax.set_xlabel('time step', fontsize=LABEL_FONTSIZE)
      ax.tick_params(axis='both', labelsize=ANNOTATION_FONTSIZE)
      ax.grid(alpha=0.16, linewidth=0.6)
      if label > 0:
        ax.tick_params(labelleft=False)
  panel_a_top = grid[0, :].get_position(fig).y1
  panel_a_bottom = grid[0, :].get_position(fig).y0
  fig.text(
      0.015,
      (panel_a_top + panel_a_bottom) / 2,
      amplitude_label,
      ha='center',
      va='center',
      rotation='vertical',
      fontsize=22,
  )

  panels = [
      ('Representation Comparison (Euclidean Distance)',
       [(name, coordinates) for name, coordinates
        in coordinates_by_metric['euclidean'].items()
        if name != 'Raw sequence']),
      ('Distance Metric Comparison (Raw Time Series)', [
          (title, coordinates_by_metric[metric]['Raw sequence'])
          for metric, title in DISTANCES]),
  ]
  tsne_axes = []
  representation_grid = grid[1, :].subgridspec(1, 4, wspace=0.15)
  raw_grid = grid[2, :].subgridspec(1, 3, wspace=0.15)
  for panel_index, (panel_title, plots) in enumerate(panels):
    grid_row = panel_index + 1
    for index, (title, coordinates) in enumerate(plots):
      slot = representation_grid[index] if panel_index == 0 else raw_grid[index]
      ax = fig.add_subplot(slot)
      tsne_axes.append(ax)
      ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='normal')
      for label, (name, color) in enumerate(zip(LABEL_NAMES, COLORS)):
        selected = labels == label
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=20,
            alpha=0.62,
            linewidths=0,
            color=color,
            label=name,
            rasterized=True,
        )
      ax.set_xlabel('t-SNE 1', fontsize=LABEL_FONTSIZE)
      ax.set_ylabel('t-SNE 2', fontsize=LABEL_FONTSIZE)
      ax.set_xticks([])
      ax.set_yticks([])
    fig.text(
        0.025, grid[grid_row, :].get_position(fig).y1 + 0.022,
        '{} {}'.format(chr(ord('b') + panel_index), panel_title),
        ha='left', va='bottom', fontsize=TITLE_FONTSIZE, fontweight='bold')

  handles, legend_labels = tsne_axes[0].get_legend_handles_labels()
  fig.legend(
      handles,
      legend_labels,
      loc='lower center',
      bbox_to_anchor=(0.5, 0.005),
      ncol=len(LABEL_NAMES),
      fontsize=LEGEND_FONTSIZE,
      markerscale=1.7,
      frameon=True,
      framealpha=0.9,
      handletextpad=0.45,
  )
  fig.text(
      0.025,
      0.97,
      'a Example EEG Time Series',
      ha='left',
      va='bottom',
      fontsize=TITLE_FONTSIZE,
      fontweight='bold',
  )
  output_path = output_dir / figure_filename
  fig.savefig(output_path, bbox_inches='tight', facecolor='white')
  plt.close(fig)
  return output_path


def parse_args(argv=None):
  project_dir = Path(__file__).resolve().parents[1]
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--data-dir',
      type=Path,
      default=project_dir / 'data' / 'public_tsne',
  )
  parser.add_argument(
      '--figure-dir',
      type=Path,
      default=project_dir / 'outputs',
      help='Directory where the PDF figure is saved.',
  )
  parser.add_argument(
      '--cache-dir',
      type=Path,
      default=project_dir / '.cache' / 'representations',
      help='Directory used for optional TimeGAN/T-Rep cache loading/saving.',
  )
  parser.add_argument(
      '--device',
      default='auto',
      help=(
          'auto selects a backend separately for TimeGAN and T-Rep; '
          'cpu, cuda, cuda:N, and mps override it'),
  )
  parser.add_argument(
      '--seed',
      type=int,
      default=7,
      help='Random seed used only by t-SNE.',
  )
  parser.add_argument(
      '--data-seed',
      type=int,
      default=7,
      help='Random seed for selecting the balanced EEG subset.',
  )
  parser.add_argument(
      '--representation-seed',
      type=int,
      default=7,
      help='Random seed for TimeGAN and T-Rep training.',
  )
  parser.add_argument('--samples-per-class', type=int, default=500)
  parser.add_argument('--perplexity', type=int, default=30)
  input_mode = parser.add_mutually_exclusive_group()
  input_mode.add_argument(
      '--normalized',
      dest='normalize_input',
      action='store_true',
      default=True,
      help='Z-score each recording/channel across time before feature extraction '
           '(default).',
  )
  input_mode.add_argument(
      '--unnormalized',
      dest='normalize_input',
      action='store_false',
      help='Use the raw EEG recordings without input normalization.',
  )
  run_mode = parser.add_mutually_exclusive_group()
  run_mode.add_argument(
      '--euclidean-only',
      action='store_true',
      help=(
          'Compute only Euclidean metrics and skip the comparison figure. '
          'Useful for repeated metric sweeps.'),
  )
  run_mode.add_argument(
      '--raw-distance-only',
      action='store_true',
      help=(
          'Compute only DTW and SBD metrics for the raw sequence and skip '
          'learned representations and the comparison figure.'),
  )
  parser.add_argument('--hidden-dim', type=int, default=12)
  parser.add_argument('--num-layers', type=int, default=2)
  parser.add_argument('--iterations', type=int, default=700)
  parser.add_argument('--batch-size', type=int, default=128)
  cache_mode = parser.add_mutually_exclusive_group()
  cache_mode.add_argument(
      '--reuse-cache',
      dest='reuse_cache',
      action='store_true',
      default=True,
      help='Reuse existing TimeGAN/T-Rep caches (default).',
  )
  cache_mode.add_argument(
      '--retrain',
      dest='reuse_cache',
      action='store_false',
      help='Ignore existing caches and retrain both learned representations.',
  )
  save_mode = parser.add_mutually_exclusive_group()
  save_mode.add_argument(
      '--save-cache',
      dest='save_cache',
      action='store_true',
      default=True,
      help='Save newly trained representation caches (default).',
  )
  save_mode.add_argument(
      '--no-save-cache',
      dest='save_cache',
      action='store_false',
      help='Do not save representations trained during this run.',
  )
  add_representation_arguments(parser)
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  # Attribute fallbacks keep programmatic callers using the older Namespace
  # interface working while command-line runs use independently controlled seeds.
  data_seed = getattr(args, 'data_seed', args.seed)
  representation_seed = getattr(args, 'representation_seed', args.seed)
  normalize_input = getattr(args, 'normalize_input', True)
  input_mode = 'normalized' if normalize_input else 'unnormalized'
  if args.samples_per_class < 11:
    raise ValueError('--samples-per-class must be at least 11 for k=10 metrics')
  args.figure_dir.mkdir(parents=True, exist_ok=True)
  print('Seeds: data={}, representations={}, t-SNE={}'.format(
      data_seed, representation_seed, args.seed))
  print('EEG input mode:', input_mode)
  # sequence: (samples=2*500, timestep=178, channel=1); labels: (2*500,)
  raw_sequences, labels = load_balanced_eeg(
      args.data_dir, args.samples_per_class, data_seed, normalize=False)
  normalized_sequences = normalize_over_time(raw_sequences)
  sequences = normalized_sequences if normalize_input else raw_sequences
  representation_args = argparse.Namespace(**vars(args))
  representation_args.seed = representation_seed
  if not normalize_input:
    representation_args.cache_dir = args.cache_dir / 'unnormalized_eeg'
  raw_distance_only = getattr(args, 'raw_distance_only', False)
  representations = {
      'Raw sequence': sequences.reshape(len(sequences), -1),
  }
  if not raw_distance_only:
    # latent: (samples=2*500, timestep=178, channel=12)
    latent = train_or_load(sequences, labels, representation_args)
    latent_summary = np.concatenate(
        [np.mean(latent, axis=1), np.std(latent, axis=1)], axis=1)
    representations.update({
        'Power spectral density': psd_features(sequences),
        'Catch22 features': catch22_features(sequences),
        'T-Rep': train_or_load_trep(
            sequences, representation_args,
            ('epileptic_seizures' if normalize_input
             else 'epileptic_seizures_unnormalized')),
        'TimeGAN': latent_summary,
    })
  # Every representation uses the selected input mode. Do not standardize
  # feature dimensions again before computing distances.
  if raw_distance_only:
    distance_specs = DISTANCES[1:]
  elif getattr(args, 'euclidean_only', False):
    distance_specs = DISTANCES[:1]
  else:
    distance_specs = DISTANCES
  coordinates_by_metric = {}
  metrics = {}
  for metric, _ in distance_specs:
    coordinates_by_metric[metric] = {}
    metrics[metric] = {}
    for name, values in representations.items():
      if metric != 'euclidean' and name != 'Raw sequence':
        continue
      print('Embedding {} with {} distance'.format(name, metric))
      coordinates_by_metric[metric][name], metrics[metric][name] = embed_and_score(
          values, labels, args.perplexity, args.seed,
          metric=metric, standardize_input=False, init='random')

  figure_path = None
  if not getattr(args, 'euclidean_only', False) and not raw_distance_only:
    figure_path = plot(
        sequences, coordinates_by_metric, labels, args.figure_dir,
        representative_sequences=normalized_sequences,
        amplitude_label=(
            'Normalized Amplitude' if normalize_input
            else 'Unnormalized Amplitude'),
        figure_filename=(
            'eeg_{}_representation_comparison_across_time.pdf'.format(
                input_mode)))
  metrics_path = args.figure_dir / 'eeg_metrics.json'
  metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')
  config_path = args.figure_dir / 'eeg_run_config.json'
  config_path.write_text(json.dumps({
      'data_seed': data_seed,
      'representation_seed': representation_seed,
      'tsne_seed': args.seed,
      'tsne_initialization': 'random',
      'input_mode': input_mode,
      'normalization_axis': 1 if normalize_input else None,
      'post_representation_standardization': False,
      'perplexity': args.perplexity,
      'samples_per_class': args.samples_per_class,
  }, indent=2) + '\n')

  if figure_path is not None:
    print('Saved figure:', figure_path)
  print('Saved metrics:', metrics_path)
  print('Saved run configuration:', config_path)
  for metric, representations_metrics in metrics.items():
    for name, values in representations_metrics.items():
      print('EEG {} ({}):'.format(name, metric), values)


if __name__ == '__main__':
  main()

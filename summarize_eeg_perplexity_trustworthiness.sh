#!/usr/bin/env bash
set -euo pipefail

# Summarize t-SNE trustworthiness@10 from existing EEG runs. This script does
# not rerun t-SNE, TimeGAN, T-Rep, DTW, or SBD.
#
# Supply perplexities as arguments, for example:
#   ./summarize_eeg_perplexity_trustworthiness.sh 30 50
# With no arguments, all perplexities from the two sweep scripts are used.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (($#)); then
  PERPLEXITIES=("$@")
else
  PERPLEXITIES=(30 50 100 200)
fi

export EEG_PROJECT_DIR="${SCRIPT_DIR}"
export EEG_PERPLEXITIES="${PERPLEXITIES[*]}"

python - <<'PY'
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev


project_dir = Path(os.environ['EEG_PROJECT_DIR'])
perplexities = os.environ['EEG_PERPLEXITIES'].split()
seeds = list(range(11))
metric = 'tsne_trustworthiness_10'


def summarize_euclidean():
  root = project_dir / 'outputs' / 'eeg_perplexity_purity'
  methods = [
      'Raw sequence',
      'Power spectral density',
      'Catch22 features',
      'T-Rep',
      'TimeGAN',
  ]
  by_seed = {}
  summary = {}
  csv_rows = []
  summary_rows = []

  for perplexity in perplexities:
    by_seed[perplexity] = {}
    summary[perplexity] = {}
    values_by_method = {method: [] for method in methods}

    for seed in seeds:
      path = (
          root / 'perplexity_{}'.format(perplexity) /
          'seed_{}'.format(seed) / 'eeg_metrics.json')
      if not path.exists():
        raise FileNotFoundError('Missing result file: {}'.format(path))
      metrics = json.loads(path.read_text())['euclidean']
      seed_results = {}

      for method in methods:
        value = float(metrics[method][metric])
        seed_results[method] = value
        values_by_method[method].append(value)
        csv_rows.append([perplexity, seed, method, value])
      by_seed[perplexity][str(seed)] = seed_results

    for method, values in values_by_method.items():
      result_mean = mean(values)
      result_std = stdev(values)
      summary[perplexity][method] = {
          'mean': result_mean,
          'std': result_std,
          'mean_plus_minus_std': '{:.4f} ± {:.4f}'.format(
              result_mean, result_std),
          'n_seeds': len(values),
      }
      summary_rows.append([
          perplexity,
          method,
          result_mean,
          result_std,
          '{:.4f} ± {:.4f}'.format(result_mean, result_std),
          len(values),
      ])

  metadata = {
      'metric': metric,
      'distance': 'euclidean',
      'varied_parameter': 'tsne_seed',
      'tsne_initialization': 'random',
      'tsne_seeds': seeds,
  }
  (root / 'trustworthiness_at_10_by_seed.json').write_text(
      json.dumps({**metadata, 'results': by_seed}, indent=2) + '\n')
  (root / 'trustworthiness_at_10_summary.json').write_text(
      json.dumps({
          **metadata,
          'std': 'sample standard deviation across t-SNE seeds (ddof=1)',
          'results': summary,
      }, indent=2) + '\n')

  with (root / 'trustworthiness_at_10_by_seed.csv').open(
      'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow([
        'perplexity', 'tsne_seed', 'method', 'trustworthiness_at_10'])
    writer.writerows(csv_rows)

  with (root / 'trustworthiness_at_10_summary.csv').open(
      'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow([
        'perplexity', 'method', 'mean', 'std',
        'mean_plus_minus_std', 'n_seeds'])
    writer.writerows(summary_rows)

  print('Saved Euclidean trustworthiness summaries under {}'.format(root))


def summarize_raw_distances():
  root = project_dir / 'outputs' / 'eeg_raw_distance_perplexity_purity'
  distances = ['dtw', 'sbd']
  by_seed = {}
  summary = {}
  csv_rows = []
  summary_rows = []

  for perplexity in perplexities:
    by_seed[perplexity] = {}
    summary[perplexity] = {}
    values_by_distance = {distance: [] for distance in distances}

    for seed in seeds:
      path = (
          root / 'perplexity_{}'.format(perplexity) /
          'seed_{}'.format(seed) / 'eeg_metrics.json')
      if not path.exists():
        raise FileNotFoundError('Missing result file: {}'.format(path))
      metrics = json.loads(path.read_text())
      seed_results = {}

      for distance in distances:
        value = float(metrics[distance]['Raw sequence'][metric])
        seed_results[distance] = value
        values_by_distance[distance].append(value)
        csv_rows.append([perplexity, seed, distance, value])
      by_seed[perplexity][str(seed)] = seed_results

    for distance, values in values_by_distance.items():
      result_mean = mean(values)
      result_std = stdev(values)
      summary[perplexity][distance] = {
          'mean': result_mean,
          'std': result_std,
          'mean_plus_minus_std': '{:.4f} ± {:.4f}'.format(
              result_mean, result_std),
          'n_seeds': len(values),
      }
      summary_rows.append([
          perplexity,
          distance,
          result_mean,
          result_std,
          '{:.4f} ± {:.4f}'.format(result_mean, result_std),
          len(values),
      ])

  metadata = {
      'metric': metric,
      'representation': 'Raw sequence',
      'distance_metrics': distances,
      'varied_parameter': 'tsne_seed',
      'tsne_initialization': 'random',
      'tsne_seeds': seeds,
  }
  (root / 'trustworthiness_at_10_by_seed.json').write_text(
      json.dumps({**metadata, 'results': by_seed}, indent=2) + '\n')
  (root / 'trustworthiness_at_10_summary.json').write_text(
      json.dumps({
          **metadata,
          'std': 'sample standard deviation across t-SNE seeds (ddof=1)',
          'results': summary,
      }, indent=2) + '\n')

  with (root / 'trustworthiness_at_10_by_seed.csv').open(
      'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow([
        'perplexity', 'tsne_seed', 'distance', 'trustworthiness_at_10'])
    writer.writerows(csv_rows)

  with (root / 'trustworthiness_at_10_summary.csv').open(
      'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow([
        'perplexity', 'distance', 'mean', 'std',
        'mean_plus_minus_std', 'n_seeds'])
    writer.writerows(summary_rows)

  print('Saved raw-distance trustworthiness summaries under {}'.format(root))


summarize_euclidean()
summarize_raw_distances()
PY

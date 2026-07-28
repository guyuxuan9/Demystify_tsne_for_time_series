#!/usr/bin/env bash
set -euo pipefail

# Run t-SNE on normalized raw EEG sequences with DTW and SBD distances.
# Supply perplexities as arguments, for example:
#   ./run_eeg_raw_distance_perplexity_purity.sh 5 10 30 50
# With no arguments, the values below are used.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (($#)); then
  PERPLEXITIES=("$@")
else
  PERPLEXITIES=(30 50 100 200)
fi

EEG_RESULTS_DIR="${EEG_RESULTS_DIR:-${SCRIPT_DIR}/outputs/eeg_raw_distance_perplexity_purity}"
EEG_DATA_DIR="${EEG_DATA_DIR:-${SCRIPT_DIR}/data/public_tsne}"
EEG_SAMPLES_PER_CLASS="${EEG_SAMPLES_PER_CLASS:-500}"
EEG_DATA_SEED="${EEG_DATA_SEED:-7}"

mkdir -p "${EEG_RESULTS_DIR}"

# The selected and normalized recordings stay fixed. Only the t-SNE seed and
# perplexity vary; no TimeGAN or T-Rep representations are trained or loaded.
for tsne_seed in {0..10}; do
  for perplexity in "${PERPLEXITIES[@]}"; do
    run_dir="${EEG_RESULTS_DIR}/perplexity_${perplexity}/seed_${tsne_seed}"
    mkdir -p "${run_dir}"
    echo "Running raw t-SNE seed=${tsne_seed}, data seed=${EEG_DATA_SEED}, perplexity=${perplexity}, distances=DTW,SBD"
    (
      cd "${SCRIPT_DIR}"
      python -m demystifying_tsne.eeg \
        --seed "${tsne_seed}" \
        --data-seed "${EEG_DATA_SEED}" \
        --perplexity "${perplexity}" \
        --samples-per-class "${EEG_SAMPLES_PER_CLASS}" \
        --data-dir "${EEG_DATA_DIR}" \
        --figure-dir "${run_dir}" \
        --raw-distance-only
    ) 2>&1 | tee "${run_dir}/run.log"
  done
done

export EEG_RESULTS_DIR
export EEG_PERPLEXITIES="${PERPLEXITIES[*]}"
export EEG_DATA_SEED
python - <<'PY'
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev

root = Path(os.environ['EEG_RESULTS_DIR'])
perplexities = os.environ['EEG_PERPLEXITIES'].split()
distances = ['dtw', 'sbd']
metric = 'tsne_neighbor_purity_10'

by_seed = {}
summary = {}
csv_rows = []
summary_rows = []
for perplexity in perplexities:
  by_seed[perplexity] = {}
  summary[perplexity] = {}
  distance_values = {distance: [] for distance in distances}
  for seed in range(11):
    path = root / f'perplexity_{perplexity}' / f'seed_{seed}' / 'eeg_metrics.json'
    metrics = json.loads(path.read_text())
    seed_results = {}
    for distance in distances:
      value = float(metrics[distance]['Raw sequence'][metric])
      seed_results[distance] = value
      distance_values[distance].append(value)
      csv_rows.append([perplexity, seed, distance, value])
    by_seed[perplexity][str(seed)] = seed_results

  for distance, values in distance_values.items():
    result_mean = mean(values)
    result_std = stdev(values)
    summary[perplexity][distance] = {
        'mean': result_mean,
        'std': result_std,
        'mean_plus_minus_std': f'{result_mean:.4f} ± {result_std:.4f}',
        'n_seeds': len(values),
    }
    summary_rows.append([
        perplexity, distance, result_mean, result_std,
        f'{result_mean:.4f} ± {result_std:.4f}', len(values),
    ])

(root / 'purity_at_10_by_seed.json').write_text(json.dumps({
    'metric': metric,
    'representation': 'Raw sequence',
    'distance_metrics': distances,
    'varied_parameter': 'tsne_seed',
    'tsne_initialization': 'random',
    'data_seed': int(os.environ['EEG_DATA_SEED']),
    'results': by_seed,
}, indent=2) + '\n')
(root / 'purity_at_10_summary.json').write_text(json.dumps({
    'metric': metric,
    'representation': 'Raw sequence',
    'distance_metrics': distances,
    'std': 'sample standard deviation across t-SNE seeds (ddof=1)',
    'varied_parameter': 'tsne_seed',
    'tsne_initialization': 'random',
    'tsne_seeds': list(range(11)),
    'data_seed': int(os.environ['EEG_DATA_SEED']),
    'results': summary,
}, indent=2) + '\n')

with (root / 'purity_at_10_by_seed.csv').open('w', newline='') as stream:
  writer = csv.writer(stream)
  writer.writerow(['perplexity', 'tsne_seed', 'distance', 'purity_at_10'])
  writer.writerows(csv_rows)

with (root / 'purity_at_10_summary.csv').open('w', newline='') as stream:
  writer = csv.writer(stream)
  writer.writerow([
      'perplexity', 'distance', 'mean', 'std', 'mean_plus_minus_std', 'n_seeds'])
  writer.writerows(summary_rows)

print(f'Saved per-seed and summary results under {root}')
PY

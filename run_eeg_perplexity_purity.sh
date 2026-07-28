#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Supply perplexities as arguments, for example:
#   ./run_eeg_perplexity_purity.sh 5 10 30 50
# With no arguments, the values below are used.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (($#)); then
  PERPLEXITIES=("$@")
else
  PERPLEXITIES=(30 50 100 200)
fi

EEG_DEVICE="${EEG_DEVICE:-auto}"
EEG_RESULTS_DIR="${EEG_RESULTS_DIR:-${SCRIPT_DIR}/outputs/eeg_perplexity_purity}"
EEG_DATA_DIR="${EEG_DATA_DIR:-${SCRIPT_DIR}/data/public_tsne}"
EEG_CACHE_DIR="${EEG_CACHE_DIR:-${SCRIPT_DIR}/.cache/representations}"
EEG_SAMPLES_PER_CLASS="${EEG_SAMPLES_PER_CLASS:-500}"
EEG_TIMEGAN_ITERATIONS="${EEG_TIMEGAN_ITERATIONS:-700}"
EEG_TREP_ITERATIONS="${EEG_TREP_ITERATIONS:-200}"
EEG_BATCH_SIZE="${EEG_BATCH_SIZE:-128}"
EEG_TREP_BATCH_SIZE="${EEG_TREP_BATCH_SIZE:-16}"
EEG_DATA_SEED="${EEG_DATA_SEED:-7}"
EEG_REPRESENTATION_SEED="${EEG_REPRESENTATION_SEED:-7}"

mkdir -p "${EEG_RESULTS_DIR}"

# Data selection and learned representations stay fixed. Only the t-SNE seed
# varies, and the same fixed representations are reused across all runs.
for tsne_seed in {0..10}; do
  for perplexity in "${PERPLEXITIES[@]}"; do
    run_dir="${EEG_RESULTS_DIR}/perplexity_${perplexity}/seed_${tsne_seed}"
    mkdir -p "${run_dir}"
    echo "Running t-SNE seed=${tsne_seed}, data seed=${EEG_DATA_SEED}, representation seed=${EEG_REPRESENTATION_SEED}, perplexity=${perplexity}"
    (
      cd "${SCRIPT_DIR}"
      python -m demystifying_tsne.eeg \
        --seed "${tsne_seed}" \
        --data-seed "${EEG_DATA_SEED}" \
        --representation-seed "${EEG_REPRESENTATION_SEED}" \
        --perplexity "${perplexity}" \
        --samples-per-class "${EEG_SAMPLES_PER_CLASS}" \
        --device "${EEG_DEVICE}" \
        --iterations "${EEG_TIMEGAN_ITERATIONS}" \
        --trep-iterations "${EEG_TREP_ITERATIONS}" \
        --batch-size "${EEG_BATCH_SIZE}" \
        --trep-batch-size "${EEG_TREP_BATCH_SIZE}" \
        --data-dir "${EEG_DATA_DIR}" \
        --cache-dir "${EEG_CACHE_DIR}" \
        --figure-dir "${run_dir}" \
        --euclidean-only
    ) 2>&1 | tee "${run_dir}/run.log"
  done
done

export EEG_RESULTS_DIR
export EEG_PERPLEXITIES="${PERPLEXITIES[*]}"
export EEG_DATA_SEED
export EEG_REPRESENTATION_SEED
python - <<'PY'
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev

root = Path(os.environ['EEG_RESULTS_DIR'])
perplexities = os.environ['EEG_PERPLEXITIES'].split()
methods = [
    'Raw sequence',
    'Power spectral density',
    'Catch22 features',
    'T-Rep',
    'TimeGAN',
]
metric = 'tsne_neighbor_purity_10'

by_seed = {}
summary = {}
csv_rows = []
summary_rows = []
for perplexity in perplexities:
  by_seed[perplexity] = {}
  summary[perplexity] = {}
  method_values = {method: [] for method in methods}
  for seed in range(11):
    path = root / f'perplexity_{perplexity}' / f'seed_{seed}' / 'eeg_metrics.json'
    with path.open() as stream:
      metrics = json.load(stream)['euclidean']
    seed_results = {}
    for method in methods:
      value = float(metrics[method][metric])
      seed_results[method] = value
      method_values[method].append(value)
      csv_rows.append([perplexity, seed, method, value])
    by_seed[perplexity][str(seed)] = seed_results

  for method, values in method_values.items():
    result_mean = mean(values)
    result_std = stdev(values)
    summary[perplexity][method] = {
        'mean': result_mean,
        'std': result_std,
        'mean_plus_minus_std': f'{result_mean:.4f} ± {result_std:.4f}',
        'n_seeds': len(values),
    }
    summary_rows.append([
        perplexity, method, result_mean, result_std,
        f'{result_mean:.4f} ± {result_std:.4f}', len(values),
    ])

(root / 'purity_at_10_by_seed.json').write_text(
    json.dumps({
        'metric': metric,
        'varied_parameter': 'tsne_seed',
        'tsne_initialization': 'random',
        'data_seed': int(os.environ['EEG_DATA_SEED']),
        'representation_seed': int(os.environ['EEG_REPRESENTATION_SEED']),
        'results': by_seed,
    }, indent=2) + '\n')
(root / 'purity_at_10_summary.json').write_text(json.dumps({
    'metric': metric,
    'std': 'sample standard deviation across t-SNE seeds (ddof=1)',
    'varied_parameter': 'tsne_seed',
    'tsne_initialization': 'random',
    'tsne_seeds': list(range(11)),
    'data_seed': int(os.environ['EEG_DATA_SEED']),
    'representation_seed': int(os.environ['EEG_REPRESENTATION_SEED']),
    'results': summary,
}, indent=2) + '\n')

with (root / 'purity_at_10_by_seed.csv').open('w', newline='') as stream:
  writer = csv.writer(stream)
  writer.writerow(['perplexity', 'tsne_seed', 'method', 'purity_at_10'])
  writer.writerows(csv_rows)

with (root / 'purity_at_10_summary.csv').open('w', newline='') as stream:
  writer = csv.writer(stream)
  writer.writerow([
      'perplexity', 'method', 'mean', 'std', 'mean_plus_minus_std', 'n_seeds'])
  writer.writerows(summary_rows)

print(f'Saved per-seed and summary results under {root}')
PY

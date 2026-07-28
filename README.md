# Demystifying t-SNE for Time Series

Companion code for the paper **“Demystifying t-SNE for Time Series.”**

t-SNE preserves neighborhoods in the space it receives. For time series,
sample-wise Euclidean distance is often a poor measure of similarity, so the
choice of input representation can matter more than the t-SNE settings. This
repository compares five label-free representations on seizure EEG:

- flattened observations;
- Welch power spectral density;
- Catch22 features;
- T-Rep embeddings; and
- TimeGAN latent representations.

Class labels are withheld from representation learning and t-SNE. They are
used only to color the final maps and compute evaluation metrics.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Automatic device selection is model-specific:

- TimeGAN selects CUDA when available and otherwise uses CPU.
- T-Rep selects CUDA, then Apple Metal (MPS), then CPU.

## Experiments

Run the seizure versus non-seizure EEG experiment with the default normalized
inputs:

```bash
python eeg_tsne_experiment.py
```

Each recording/channel is independently normalized across time before feature
extraction by default. To run the same experiment on the original EEG values,
use:

```bash
python eeg_tsne_experiment.py --unnormalized
```

The command downloads the public dataset on first use. It compares raw
sequences, power spectral density, Catch22, T-Rep, and TimeGAN using Euclidean
distance, and compares Euclidean, DTW, and SBD distances for the raw sequences.
No additional feature-wise standardization is applied after representation
extraction. Labels are used only for plot colors and evaluation.

A complete run writes the following artifacts under `outputs/`:

- `outputs/eeg_normalized_representation_comparison_across_time.pdf` or
  `outputs/eeg_unnormalized_representation_comparison_across_time.pdf`
- `outputs/eeg_metrics.json`
- `outputs/eeg_run_config.json`

To evaluate Euclidean-distance t-SNE across the five representations over 11
t-SNE seeds and the default perplexities 30, 50, 100, and 200, run:

```bash
./run_eeg_perplexity_purity.sh
```

To evaluate normalized raw EEG with DTW and SBD over the same sweep, run:

```bash
./run_eeg_raw_distance_perplexity_purity.sh
```

After both sweeps finish, summarize trustworthiness@10 from their existing
metrics with:

```bash
./summarize_eeg_perplexity_trustworthiness.sh
```

The sweep scripts save per-run logs and metrics plus CSV and JSON summaries under `outputs/eeg_perplexity_purity/` and
`outputs/eeg_raw_distance_perplexity_purity/`.

## Citation

Citation details will be added when the paper is published. Until then, please
cite the paper by title and authors listed in `CITATION.cff`.

## Acknowledgments

This project includes code adapted from
[TimeGAN](https://github.com/guyuxuan9/timeGAN) and a vendored subset of
[T-Rep](https://github.com/Let-it-Care/T-Rep). See
`THIRD_PARTY_NOTICES.md` and the included license files.

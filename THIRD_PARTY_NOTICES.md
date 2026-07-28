# Third-party notices

## TimeGAN

`demystifying_tsne/models/timegan.py` and
`demystifying_tsne/models/timegan_utils.py` are adapted from the PyTorch
TimeGAN implementation formerly maintained in this repository. TimeGAN was
introduced by Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar in
“Time-series Generative Adversarial Networks” (NeurIPS 2019). The adapted code
is distributed under the repository's Apache License 2.0.

## T-Rep

`demystifying_tsne/vendor/trep` contains the subset of
[`Let-it-Care/T-Rep`](https://github.com/Let-it-Care/T-Rep) needed for this
project, based on commit `d624998e30b34f365649b13c1cade7c54c41687f`.
Imports were made package-relative; model behavior is otherwise unchanged.
The upstream MIT license is included at
`demystifying_tsne/vendor/trep/LICENSE`.

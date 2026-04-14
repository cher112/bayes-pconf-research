# Temporal Bayes Error R*(t) for Clinical Time Series

Preliminary experiments for estimating the Bayes-optimal error rate on irregularly sampled clinical time series (PhysioNet 2012 & 2019).

## What this repo contains

- `src/`: Training and evaluation scripts for 6 encoder architectures (GRU-D, BRITS, SAITS, iTransformer, TimesNet, SeFT)
- `experiments/`: Raw metrics (AUROC, accuracy, R* via isotonic calibration, ECE) for each encoder on P12 and P19 at t=48h

## Key finding

R* is consistent across encoders (P12: mean=0.126, std=0.006; P19: mean=0.048, std=0.003), supporting the interpretation that R* measures a property of the dataset, not the model.

## Setup

Requires [pypots](https://github.com/WenjieDu/PyPOTS) and PyTorch. PhysioNet 2012 data loads automatically via pypots; PhysioNet 2019 requires downloading .psv files from [PhysioNet](https://physionet.org/content/challenge-2019/).

## Author

Chen Zhihao (czhbupt@gmail.com)

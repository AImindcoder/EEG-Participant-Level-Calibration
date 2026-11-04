# Unlabeled Participant-Level Calibration for EEG Artifact Detection

## Overview
This research evaluates unlabeled participant-level calibration methods for EEG artifact detection under inter-subject variability.

## Research Question
Can unlabeled calibration data from an unseen participant improve EEG artifact detection?

## Main Experiments

### V9-17: Locked Participant Evaluation
- Dataset: PhysioMotion.
- Six locked participants.
- No-reference participant macro F1: 0.806058.
- Geometric-median participant macro F1: 0.822585.
- Paired mean F1 improvement: +0.016527.
- Five of six participants improved.

### V9-44 and V9-45: Calibration Benchmark
- Evaluated mean-centering, geometric median, z-score normalization, robust z-score, and hybrid references.
- Highest observed participant macro F1: 0.852585 for hybrid 50/50.
- Paired comparisons did not establish statistically significant superiority.

### V9-54: Nabiyouni EEG Pilot
- Separate single-channel, six-class classification experiment.
- No-reference macro F1: 0.475585.
- Mean-centering macro F1: 0.590895.
- Geometric-median macro F1: 0.595838.
- Three held-out test participants.

## Methodology
- Feature-based classification.
- Unlabeled chronological calibration.
- Calibration/evaluation separation.
- Participant-level evaluation.
- Paired bootstrap comparisons.
- Experimental integrity audits.

## Limitations
- Main locked evaluation: six participants.
- Validation benchmark: six participants.
- Nabiyouni pilot: three test participants.
- V9 uses separate logistic-regression classifiers, not the V8 neural network.
- The Nabiyouni pilot is not strict cross-dataset validation.
- Results are exploratory and do not establish universal calibration superiority.

## Datasets
- PhysioMotion: OpenNeuro ds006386, version 1.0.1.
- Nabiyouni: https://github.com/inabiyouni/EEG_dataset_for_artifact-noise_detection

Raw EEG recordings and participant-level prediction tables are not included.

## Reproducibility
Some experiment scripts require locally stored datasets and V8 feature files. Input paths must be configured before execution. The repository provides selected scripts, aggregate results, and research reports.

## Author
Sahil Khan

GitHub: https://github.com/AImindcoder
LinkedIn: https://www.linkedin.com/in/sahil-khan18/
Portfolio: https://khan-sahil-portfolio.netlify.app/

## Research Status
Reported experiments and final evidence auditing are complete. Manuscript preparation and further reproducibility documentation remain ongoing.

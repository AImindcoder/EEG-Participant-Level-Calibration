# V9-54: Nabiyouni Independent-Participant Classification

Date: 2026-11-04T16:37:43

## Experimental protocol

- Dataset: Nabiyouni EEG Artifact Dataset
- Training participants: Subj_1 to Subj_7
- Independent test participants: Subj_8 to Subj_10
- Calibration: repetition 1, without labels
- Supervised training/evaluation: repetition 2
- Sampling rate: 1000 Hz
- Segment duration: 1 second
- Features: 15 time-domain and frequency-domain features
- Classes: six filename-derived conditions
- Classifier: logistic regression
- Feature scaling: fitted on training data only

## Participant-level results

| Method | Mean macro-F1 | SD | Mean balanced accuracy | Mean macro AP |
|---|---:|---:|---:|---:|
| no_reference | 0.4756 | 0.0815 | 0.5443 | 0.6619 |
| mean_center | 0.5909 | 0.0371 | 0.6404 | 0.6844 |
| geometric_median | 0.5958 | 0.0229 | 0.6471 | 0.6952 |

## Scientific limitations

1. This is a single-channel, six-class condition-classification pilot, not the original 34-channel artifact detector.
2. Training and test participants come from the same Nabiyouni dataset. This is participant-held-out evaluation, not a strict train-on-one-dataset/test-on-another-dataset experiment.
3. Labels are inherited from recording filenames, not independently annotated artifact boundaries.
4. Repetition 1 is an unlabeled calibration pool containing a mixture of recording conditions.
5. Repetition 2 is used for supervised training or held-out testing.
6. Only three unseen participants are tested. Results are exploratory and do not establish broad population generalization.
7. The one-second segmentation does not establish exact artifact-event boundaries.
8. No hyperparameter tuning was performed on test participants.
9. Physical signal units have not been assumed.

## Integrity

- Original EEG files modified: No
- Existing V8 model modified: No
- Existing V9 models modified: No
- Test labels used for calibration: No
- Test participants used for training: No
- Training and test participant overlap: None

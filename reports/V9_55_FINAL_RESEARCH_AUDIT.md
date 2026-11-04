# V9-55: Final Research Documentation and Evidence Audit

Date: 2026-11-04 16:51:30

## 1. Purpose

V9-55 consolidates and audits the evidence produced by V9-54.
It does not retrain models, modify EEG data, or change previous experiments.

## 2. Experimental protocol

- Dataset: Nabiyouni EEG Artifact Dataset
- Evaluation: participant-held-out evaluation within one dataset
- Training participants: Subj_1, Subj_2, Subj_3, Subj_4, Subj_5, Subj_6, Subj_7
- Test participants: Subj_8, Subj_9, Subj_10
- Calibration repetition: 1, unlabeled
- Evaluation repetition: 2
- Sampling rate: 1000 Hz
- Segment duration: 1.0 seconds
- Training segments: 524
- Test segments: 221
- Features: 15
- Classes: baseline, eye, eyebrow, head_motion, jaw_clench, jaw_movement
- Classifier: LogisticRegression
- Classifier C: 1.0
- Class weights: balanced
- Scaling: fitted on training data only

## 3. Aggregate participant-level results

| Method | Mean macro-F1 | SD | Balanced accuracy | Macro AP |
|---|---:|---:|---:|---:|
| no_reference | 0.4756 | 0.0815 | 0.5443 | 0.6619 |
| mean_center | 0.5909 | 0.0371 | 0.6404 | 0.6844 |
| geometric_median | 0.5958 | 0.0229 | 0.6471 | 0.6952 |

Highest aggregate participant macro-F1: **geometric_median** (0.5958).

The difference between geometric-median centering and mean centering is small.
The results do not establish universal superiority of either method.

## 4. Participant-level macro-F1 comparison

| Participant | No reference | Mean center | Geometric median | Geometric minus no reference |
|---|---:|---:|---:|---:|
| Subj_8 | 0.531332 | 0.620815 | 0.622799 | 0.091466 |
| Subj_9 | 0.360405 | 0.538576 | 0.566733 | 0.206328 |
| Subj_10 | 0.535017 | 0.613294 | 0.597983 | 0.062966 |

## 5. Class-wise mean F1

These values are recalculated from the original class metrics CSV by averaging participant-level F1 within each method and class.

| Class | Method | Participants | Mean F1 |
|---|---|---:|---:|
| baseline | geometric_median | 3 | 0.751084 |
| baseline | mean_center | 3 | 0.710733 |
| baseline | no_reference | 3 | 0.274892 |
| eye | geometric_median | 3 | 0.561884 |
| eye | mean_center | 3 | 0.535613 |
| eye | no_reference | 3 | 0.374133 |
| eyebrow | geometric_median | 3 | 0.765993 |
| eyebrow | mean_center | 3 | 0.772311 |
| eyebrow | no_reference | 3 | 0.699521 |
| head_motion | geometric_median | 3 | 0.436508 |
| head_motion | mean_center | 3 | 0.462653 |
| head_motion | no_reference | 3 | 0.383666 |
| jaw_clench | geometric_median | 3 | 0.680212 |
| jaw_clench | mean_center | 3 | 0.695806 |
| jaw_clench | no_reference | 3 | 0.662594 |
| jaw_movement | geometric_median | 3 | 0.379349 |
| jaw_movement | mean_center | 3 | 0.368254 |
| jaw_movement | no_reference | 3 | 0.458705 |

Class-level performance is method-dependent. A method that improves aggregate macro-F1 may reduce F1 for individual classes or participants.

## 6. Scientific interpretation

The experiment provides preliminary evidence that unlabeled participant-level feature centering can improve six-class EEG recording-condition classification on held-out participants.

Geometric-median centering achieved the highest aggregate macro-F1, balanced accuracy, and macro average precision among the three evaluated methods.

However:
- Its aggregate macro-F1 advantage over mean centering is small.
- Class-specific effects are inconsistent.
- Only three unseen participants were evaluated.
- The findings are exploratory and require larger independent cohorts.

## 7. Scientific limitations

1. This is a single-channel, six-class condition-classification pilot.
2. It is not direct validation of the original 34-channel V8 neural model or a separate 34-channel artifact detector.
3. Training and test participants come from the same Nabiyouni dataset.
4. Labels are inherited from recording filenames rather than independently annotated artifact boundaries.
5. Calibration repetition 1 contains an unlabeled mixture of recording conditions.
6. Repetition 2 supplies supervised training data and held-out participant evaluation data.
7. One-second segments do not establish precise artifact-event boundaries.
8. Only three held-out participants were evaluated.
9. Physical signal units were not assumed.
10. This experiment does not establish clinical performance or state-of-the-art results.

## 8. Integrity and evidence

- Participant overlap: 0
- Calibration labels used: False
- Test labels used for calibration: False
- Original EEG modification recorded: False
- V8 modification recorded: False
- Existing V9 model modification recorded: False
- Original V9-54 evidence files hashed: 7

The SHA-256 file is a snapshot of the V9-54 output files at audit time. It is not a historical before-and-after proof of their integrity.

## 9. Audit checks

| Check | Status | Detail |
|---|---|---|
| No participant overlap | PASS | Overlap count: 0 |
| Calibration labels unused | PASS | Calibration labels used: False |
| Test labels unused for calibration | PASS | Test labels used for calibration: False |
| Training-only scaler | PASS | Scaler fitted on training data only |
| Training-only global calibration scaling | PASS | Global calibration location and scale fitted on training data only |
| V8 protection recorded | PASS | V8 modified: False |
| V9 model protection recorded | PASS | Existing V9 models modified: False |
| Source data protection recorded | PASS | Source data modified: False |
| Expected training participants | PASS | Training participants: 7 |
| Expected test participants | PASS | Test participants: 3 |
| Six classification classes | PASS | Class count: 6 |
| Fifteen features | PASS | Feature count: 15 |
| Expected training segments | PASS | Training segments: 524 |
| Expected test segments | PASS | Test segments: 221 |
| Participant metrics row count | PASS | Rows: 9; expected 9 |
| Class metrics row count | PASS | Rows: 54; expected 54 |
| Prediction methods | PASS | Methods found: 3 |
| Prediction rows per method | PASS | Each method should contain 221 held-out predictions |
| Prediction participant identities | PASS | Observed test participants: Subj_10, Subj_8, Subj_9 |
| JSON and method CSV agreement | PASS | Aggregate metrics compared for all three methods |
| Seven original evidence files present | PASS | Files hashed: 7 |

## 10. Conclusion

V9-54 is documented as an exploratory participant-held-out EEG condition-classification experiment on the Nabiyouni dataset.

V9-55 records its protocol, aggregate metrics, participant comparisons, class-level metrics, and evidence-file hashes. The conclusions remain limited to the evaluated dataset, participants, labels, and experimental design.

---
End of report.

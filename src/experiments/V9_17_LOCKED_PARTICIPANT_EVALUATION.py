
import json
import os
import time
import hashlib
import warnings
from pathlib import Path


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# V9-17: LOCKED PARTICIPANT EVALUATION
#
# Frozen primary experiment:
#   - Training: 18 development training participants
#   - Calibration: first 50% of each recording
#   - Evaluation: final 50% of each recording
#   - Primary reference: geometric median
#   - Comparator: no reference
#   - Eye alpha: 0.8
#   - Jaw alpha: 0.4
#   - Motion alpha: 0.7
#   - Logistic Regression C=1.0
#   - Threshold=0.5
#
# LOCKED TEST LABELS ARE ACCESSED ONLY AFTER PREDICTION.
# ============================================================

START = time.time()
SEED = 42
rng = np.random.default_rng(SEED)

V7_ROOT = Path(os.environ["EEG_V7_ROOT"])

V8 = V7_ROOT / "V8_Development"

V9 = V7_ROOT / "V9_Domain_Generalization"
DEV_DATA = V8 / "Data" / "V8_NORMALIZED"

LOCKED_DATA = (
    V8 / "Results" / "V8_HELDOUT_TEST"
)

OUT = (
    V9
    / "Results"
    / "V9_17_LOCKED_PARTICIPANT_EVALUATION"
)

REPORT_DIR = (
    V9
    / "Reports"
    / "V9_17_LOCKED_PARTICIPANT_EVALUATION"
)

OUT.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEV_FEATURE_FILE = (
    DEV_DATA / "V8_0P5S_34CH_FEATURES_ROBUST.npy"
)

DEV_META_FILE = (
    DEV_DATA / "V8_0P5S_METADATA.csv"
)

TEST_FEATURE_FILE = (
    LOCKED_DATA / "V8_TEST_NORMALIZED_FEATURES.npy"
)

TEST_META_FILE = (
    LOCKED_DATA / "V8_TEST_LABELS_METADATA.csv"
)

LOCKED = {
    "sub-2",
    "sub-7",
    "sub-14",
    "sub-25",
    "sub-26",
    "sub-27",
}

EXPECTED_VAL = {
    "sub-3",
    "sub-6",
    "sub-13",
    "sub-20",
    "sub-21",
    "sub-28",
}

EXPECTED_TRAIN_COUNT = 18

CLASSES = ["eye", "jaw", "motion"]

LABEL_COLS = {
    "eye": "true_eye",
    "jaw": "true_jaw",
    "motion": "true_motion",
}

ALPHAS = {
    "eye": 0.8,
    "jaw": 0.4,
    "motion": 0.7,
}

CONDITIONS = [
    "no_reference",
    "geometric_median",
]

CALIBRATION_FRACTION = 0.50
THRESHOLD = 0.50
BOOTSTRAPS = 10000

print("=" * 75)
print("V9-17 LOCKED PARTICIPANT EVALUATION")
print("=" * 75)


# ============================================================
# 1. FILE AND HASH UTILITIES
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


for path in [
    DEV_FEATURE_FILE,
    DEV_META_FILE,
    TEST_FEATURE_FILE,
    TEST_META_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )

INPUT_HASHES_BEFORE = {
    str(path): sha256_file(path)
    for path in [
        DEV_FEATURE_FILE,
        DEV_META_FILE,
        TEST_FEATURE_FILE,
        TEST_META_FILE,
    ]
}


# ============================================================
# 2. LOAD DEVELOPMENT AND LOCKED FEATURES
# ============================================================

print("\n[1/10] Loading feature data...")

dev_memmap = np.load(
    DEV_FEATURE_FILE,
    mmap_mode="r",
)

test_memmap = np.load(
    TEST_FEATURE_FILE,
    mmap_mode="r",
)


def flatten_features(array):
    if array.ndim == 3:
        result = np.asarray(
            array,
            dtype=np.float32,
        ).reshape(len(array), -1)

    elif array.ndim == 2:
        result = np.asarray(
            array,
            dtype=np.float32,
        )

    else:
        raise ValueError(
            f"Unexpected feature dimensions: {array.shape}"
        )

    return result


X_dev = flatten_features(dev_memmap)
X_test = flatten_features(test_memmap)

dev_df = pd.read_csv(DEV_META_FILE)
test_df = pd.read_csv(TEST_META_FILE)

if len(X_dev) != len(dev_df):
    raise ValueError("Development feature/metadata mismatch")

if len(X_test) != len(test_df):
    raise ValueError("Locked feature/metadata mismatch")

if X_dev.shape[1] != 476 or X_test.shape[1] != 476:
    raise ValueError("Expected exactly 476 features per frame")

if not np.isfinite(X_dev).all():
    raise ValueError("Development features contain invalid values")

if not np.isfinite(X_test).all():
    raise ValueError("Locked features contain invalid values")

print("Development features:", X_dev.shape)
print("Locked features:", X_test.shape)
print("Development metadata:", dev_df.shape)
print("Locked metadata:", test_df.shape)


# ============================================================
# 3. METADATA AND PARTICIPANT SPLIT VALIDATION
# ============================================================

print("\n[2/10] Validating participants and metadata...")


def resolve_column(frame, candidates):
    lookup = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Missing column. Expected one of {candidates}; "
        f"available: {list(frame.columns)}"
    )


DEV_SUB = resolve_column(
    dev_df,
    ["subject", "subject_id", "participant"],
)

DEV_REC = resolve_column(
    dev_df,
    ["recording", "recording_id", "run"],
)

DEV_FRAME = resolve_column(
    dev_df,
    ["frame_index", "frame", "frame_id"],
)

SPLIT_COL = resolve_column(
    dev_df,
    ["split", "dataset_split", "partition"],
)

TEST_SUB = resolve_column(
    test_df,
    ["subject", "subject_id", "participant"],
)

TEST_REC = resolve_column(
    test_df,
    ["recording", "recording_id", "run"],
)

TEST_FRAME = resolve_column(
    test_df,
    ["frame_index", "frame", "frame_id"],
)

dev_df[DEV_SUB] = dev_df[DEV_SUB].astype(str)
dev_df[DEV_REC] = dev_df[DEV_REC].astype(str)

test_df[TEST_SUB] = test_df[TEST_SUB].astype(str)
test_df[TEST_REC] = test_df[TEST_REC].astype(str)

dev_df[DEV_FRAME] = pd.to_numeric(
    dev_df[DEV_FRAME],
    errors="coerce",
)

test_df[TEST_FRAME] = pd.to_numeric(
    test_df[TEST_FRAME],
    errors="coerce",
)

if dev_df[DEV_FRAME].isna().any():
    raise ValueError("Invalid development frame indices")

if test_df[TEST_FRAME].isna().any():
    raise ValueError("Invalid locked frame indices")

split_values = (
    dev_df[SPLIT_COL]
    .astype(str)
    .str.strip()
    .str.lower()
)

train_mask = split_values.isin(
    ["train", "training"]
).to_numpy()

val_mask = split_values.isin(
    ["val", "valid", "validation"]
).to_numpy()

if np.any(~(train_mask | val_mask)):
    raise ValueError("Unexpected development split value")

train_subjects = sorted(
    dev_df.loc[train_mask, DEV_SUB].unique().tolist()
)

val_subjects = sorted(
    dev_df.loc[val_mask, DEV_SUB].unique().tolist()
)

locked_subjects = sorted(
    test_df[TEST_SUB].unique().tolist()
)

if len(train_subjects) != EXPECTED_TRAIN_COUNT:
    raise ValueError(
        f"Expected 18 training participants, got {len(train_subjects)}"
    )

if set(val_subjects) != EXPECTED_VAL:
    raise ValueError(
        f"Unexpected validation participants: {val_subjects}"
    )

if set(locked_subjects) != LOCKED:
    raise ValueError(
        f"Locked participant mismatch: {locked_subjects}"
    )

if set(train_subjects) & set(val_subjects):
    raise RuntimeError("Train/validation participant overlap")

if set(train_subjects) & LOCKED:
    raise RuntimeError("Locked participant entered training")

if set(val_subjects) & LOCKED:
    raise RuntimeError("Locked participant entered development validation")

if set(train_subjects) & set(locked_subjects):
    raise RuntimeError("Train/test participant overlap")

if set(val_subjects) & set(locked_subjects):
    raise RuntimeError("Validation/test participant overlap")

print("Training participants:", train_subjects)
print("Development validation participants:", val_subjects)
print("Locked evaluation participants:", locked_subjects)


# ============================================================
# 4. TEMPORAL CALIBRATION AND EVALUATION INDICES
# ============================================================

print("\n[3/10] Creating temporal calibration/evaluation windows...")


def make_temporal_indices(
    frame,
    subject_column,
    recording_column,
    frame_column,
):
    calibration_parts = []
    evaluation_parts = []
    recording_count = 0

    grouped = frame.groupby(
        [subject_column, recording_column],
        sort=False,
    ).indices

    for _, positions in grouped.items():

        positions = np.asarray(
            positions,
            dtype=np.int64,
        )

        order = np.argsort(
            frame.iloc[positions][frame_column].to_numpy(),
            kind="stable",
        )

        positions = positions[order]

        n = len(positions)

        if n < 20:
            raise ValueError(
                "Recording has fewer than 20 eligible frames"
            )

        split_point = int(
            np.floor(n * CALIBRATION_FRACTION)
        )

        if split_point < 1 or split_point >= n:
            raise ValueError(
                "Invalid calibration/evaluation split"
            )

        calibration_parts.append(
            positions[:split_point]
        )

        evaluation_parts.append(
            positions[split_point:]
        )

        recording_count += 1

    calibration = np.concatenate(calibration_parts)
    evaluation = np.concatenate(evaluation_parts)

    if np.intersect1d(
        calibration,
        evaluation,
    ).size:
        raise RuntimeError(
            "Calibration/evaluation overlap detected"
        )

    if len(calibration) + len(evaluation) != len(frame):
        raise RuntimeError(
            "Temporal split did not preserve all rows"
        )

    return calibration, evaluation, recording_count


dev_cal, dev_eval, dev_recordings = make_temporal_indices(
    dev_df,
    DEV_SUB,
    DEV_REC,
    DEV_FRAME,
)

test_cal, test_eval, test_recordings = make_temporal_indices(
    test_df,
    TEST_SUB,
    TEST_REC,
    TEST_FRAME,
)

# Use only the original 18 training participants.
train_eval = dev_eval[
    train_mask[dev_eval]
]

if len(train_eval) == 0:
    raise ValueError("No development training evaluation rows")

if len(test_eval) == 0:
    raise ValueError("No locked evaluation rows")

if not set(
    dev_df.iloc[train_eval][DEV_SUB].unique()
) == set(train_subjects):
    raise RuntimeError(
        "Some training participants are missing from evaluation"
    )

print("Development recordings:", dev_recordings)
print("Locked recordings:", test_recordings)
print("Development calibration frames:", len(dev_cal))
print("Training evaluation frames:", len(train_eval))
print("Locked calibration frames:", len(test_cal))
print("Locked evaluation frames:", len(test_eval))


# ============================================================
# 5. LABELS FOR DEVELOPMENT TRAINING ONLY
# ============================================================

print("\n[4/10] Loading development training labels...")

dev_label_columns = {
    cls: resolve_column(
        dev_df,
        [f"true_{cls}", cls],
    )
    for cls in CLASSES
}

Y_train = {}

for cls, column in dev_label_columns.items():

    values = pd.to_numeric(
        dev_df[column],
        errors="coerce",
    ).fillna(0).to_numpy()

    Y_train[cls] = (
        values > 0
    ).astype(np.int8)

    if len(np.unique(Y_train[cls][train_eval])) < 2:
        raise ValueError(
            f"Training labels for {cls} contain only one class"
        )

print("Development labels ready.")
print("Locked test labels remain unused.")


# ============================================================
# 6. GEOMETRIC MEDIAN — SAME V9-14 IMPLEMENTATION
# ============================================================

def geometric_median(
    data,
    iterations=100,
    tolerance=1e-5,
):
    data = np.asarray(
        data,
        dtype=np.float64,
    )

    center0 = np.median(
        data,
        axis=0,
    )

    scale = (
        1.4826
        * np.median(
            np.abs(data - center0),
            axis=0,
        )
    )

    scale = np.maximum(
        scale,
        1e-6,
    )

    z = (data - center0) / scale

    center = np.median(
        z,
        axis=0,
    )

    for _ in range(iterations):

        distances = np.linalg.norm(
            z - center,
            axis=1,
        )

        distances = np.maximum(
            distances,
            1e-10,
        )

        weights = 1.0 / distances

        updated = (
            np.sum(
                z * weights[:, None],
                axis=0,
            )
            / weights.sum()
        )

        if np.linalg.norm(
            updated - center
        ) < tolerance:

            center = updated
            break

        center = updated

    return (
        center * scale + center0
    ).astype(np.float32)


# ============================================================
# 7. UNLABELED PARTICIPANT REFERENCES
# ============================================================

print("\n[5/10] Estimating unlabeled participant references...")


def build_references(
    feature_matrix,
    metadata,
    calibration_indices,
    subject_column,
):
    references = {}

    subjects = metadata.iloc[
        calibration_indices
    ][subject_column].to_numpy()

    for subject in np.unique(subjects):

        subject_indices = calibration_indices[
            subjects == subject
        ]

        # Feature-only estimator.
        # No class labels are accessed.
        reference = geometric_median(
            feature_matrix[subject_indices]
        )

        references[subject] = reference

    return references


train_references = build_references(
    X_dev,
    dev_df,
    dev_cal,
    DEV_SUB,
)

test_references = build_references(
    X_test,
    test_df,
    test_cal,
    TEST_SUB,
)

if set(train_references) != set(
    dev_df.iloc[dev_cal][DEV_SUB].unique()
):
    raise RuntimeError("Training reference participant mismatch")

if set(test_references) != set(locked_subjects):
    raise RuntimeError("Locked reference participant mismatch")

print("Training references:", len(train_references))
print("Locked references:", len(test_references))


# ============================================================
# 8. FEATURE TRANSFORMATION
# ============================================================

def transform_rows(
    feature_matrix,
    metadata,
    indices,
    references,
    subject_column,
    cls,
    condition,
):
    if condition == "no_reference":
        return feature_matrix[indices].copy()

    if condition != "geometric_median":
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    subjects = metadata.iloc[
        indices
    ][subject_column].to_numpy()

    result = np.empty(
        (len(indices), feature_matrix.shape[1]),
        dtype=np.float32,
    )

    alpha = ALPHAS[cls]

    for subject in np.unique(subjects):

        if subject not in references:
            raise KeyError(
                f"Missing reference for {subject}"
            )

        local_mask = subjects == subject

        result[local_mask] = (
            feature_matrix[indices[local_mask]]
            - alpha * references[subject]
        )

    return result


# ============================================================
# 9. TRAIN AND PREDICT — TEST LABELS STILL UNUSED
# ============================================================

print("\n[6/10] Training classifiers and predicting locked data...")

prediction_arrays = {}

model_audit = []

for condition in CONDITIONS:

    print("\nCondition:", condition)

    for cls in CLASSES:

        print("  Class:", cls)

        X_train = transform_rows(
            X_dev,
            dev_df,
            train_eval,
            train_references,
            DEV_SUB,
            cls,
            condition,
        )

        X_locked = transform_rows(
            X_test,
            test_df,
            test_eval,
            test_references,
            TEST_SUB,
            cls,
            condition,
        )

        y_train = Y_train[cls][train_eval]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_locked_scaled = scaler.transform(
            X_locked
        )

        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=SEED,
            solver="lbfgs",
        )

        model.fit(
            X_train_scaled,
            y_train,
        )

        probabilities = model.predict_proba(
            X_locked_scaled
        )[:, 1]

        if not np.isfinite(probabilities).all():
            raise RuntimeError(
                "Invalid locked prediction probabilities"
            )

        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise RuntimeError(
                "Probability outside [0,1]"
            )

        predictions = (
            probabilities >= THRESHOLD
        ).astype(np.int8)

        prediction_arrays[
            (condition, cls)
        ] = probabilities.astype(np.float64)

        model_audit.append({
            "condition": condition,
            "class": cls,
            "training_frames": int(len(y_train)),
            "training_positive_frames": int(y_train.sum()),
            "training_positive_rate": float(y_train.mean()),
            "locked_evaluation_frames": int(len(probabilities)),
            "predicted_positive_frames": int(predictions.sum()),
            "threshold": THRESHOLD,
            "C": 1.0,
        })

        print(
            "    Training frames:",
            len(y_train),
            "| Locked predictions:",
            len(probabilities),
        )


# ============================================================
# 10. ONLY NOW ACCESS LOCKED TEST LABELS FOR SCORING
# ============================================================

print("\n[7/10] Prediction complete. Now scoring locked labels...")

# This is the first section that reads the locked test labels.
test_label_columns = {
    cls: resolve_column(
        test_df,
        [f"true_{cls}", cls],
    )
    for cls in CLASSES
}

Y_test = {}

for cls, column in test_label_columns.items():

    values = pd.to_numeric(
        test_df.iloc[test_eval][column],
        errors="coerce",
    ).fillna(0).to_numpy()

    Y_test[cls] = (
        values > 0
    ).astype(np.int8)


def calculate_metrics(
    y_true,
    probabilities,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "frames": int(len(y_true)),
        "positive_frames": int(y_true.sum()),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "roc_auc": (
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            )
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


pooled_rows = []
participant_rows = []
prediction_rows = []

test_subject_vector = (
    test_df.iloc[test_eval][TEST_SUB]
    .to_numpy()
)

test_recording_vector = (
    test_df.iloc[test_eval][TEST_REC]
    .to_numpy()
)

test_frame_vector = (
    test_df.iloc[test_eval][TEST_FRAME]
    .to_numpy()
)

for condition in CONDITIONS:

    for cls in CLASSES:

        probabilities = prediction_arrays[
            (condition, cls)
        ]

        y_true = Y_test[cls]

        pooled_metrics = calculate_metrics(
            y_true,
            probabilities,
        )

        pooled_rows.append({
            "condition": condition,
            "class": cls,
            **pooled_metrics,
        })

        predictions = (
            probabilities >= THRESHOLD
        ).astype(np.int8)

        for subject in locked_subjects:

            subject_mask = (
                test_subject_vector == subject
            )

            if not np.any(subject_mask):
                raise RuntimeError(
                    f"No evaluation rows for {subject}"
                )

            subject_metrics = calculate_metrics(
                y_true[subject_mask],
                probabilities[subject_mask],
            )

            participant_rows.append({
                "condition": condition,
                "class": cls,
                "subject": subject,
                **subject_metrics,
            })

        for i in range(len(y_true)):

            prediction_rows.append({
                "condition": condition,
                "class": cls,
                "subject": test_subject_vector[i],
                "recording": test_recording_vector[i],
                "frame_index": int(test_frame_vector[i]),
                "true_label": int(y_true[i]),
                "probability": float(probabilities[i]),
                "predicted_label": int(predictions[i]),
            })


# ============================================================
# 11. SAVE RESULTS
# ============================================================

print("\n[8/10] Saving evaluation outputs...")

pooled_df = pd.DataFrame(pooled_rows)
participant_df = pd.DataFrame(participant_rows)
predictions_df = pd.DataFrame(prediction_rows)
model_audit_df = pd.DataFrame(model_audit)

pooled_df.to_csv(
    OUT / "V9_17_POOLED_METRICS.csv",
    index=False,
)

participant_df.to_csv(
    OUT / "V9_17_PARTICIPANT_METRICS.csv",
    index=False,
)

predictions_df.to_csv(
    OUT / "V9_17_LOCKED_PREDICTIONS.csv",
    index=False,
)

model_audit_df.to_csv(
    OUT / "V9_17_MODEL_AUDIT.csv",
    index=False,
)


# ============================================================
# 12. PARTICIPANT-CLASS MACRO SUMMARY
# ============================================================

class_summary = (
    participant_df
    .groupby(
        ["condition", "class"],
        as_index=False,
    )
    .agg(
        participant_macro_f1=("f1", "mean"),
        participant_macro_precision=("precision", "mean"),
        participant_macro_recall=("recall", "mean"),
        participant_macro_pr_auc=("pr_auc", "mean"),
        participant_macro_roc_auc=("roc_auc", "mean"),
        participant_count=("subject", "nunique"),
    )
)

class_summary.to_csv(
    OUT / "V9_17_CLASS_SUMMARY.csv",
    index=False,
)

# Each participant receives equal weight.
participant_macro = (
    participant_df
    .groupby(
        ["subject", "condition"],
        as_index=False,
    )
    .agg(
        participant_macro_f1=("f1", "mean"),
        participant_macro_pr_auc=("pr_auc", "mean"),
        participant_macro_roc_auc=("roc_auc", "mean"),
    )
)

participant_macro.to_csv(
    OUT / "V9_17_PARTICIPANT_MACRO.csv",
    index=False,
)

overall_summary = (
    participant_macro
    .groupby("condition", as_index=False)
    .agg(
        overall_participant_macro_f1=(
            "participant_macro_f1",
            "mean",
        ),
        overall_participant_macro_pr_auc=(
            "participant_macro_pr_auc",
            "mean",
        ),
        overall_participant_macro_roc_auc=(
            "participant_macro_roc_auc",
            "mean",
        ),
        participant_count=("subject", "nunique"),
    )
)

overall_summary.to_csv(
    OUT / "V9_17_OVERALL_SUMMARY.csv",
    index=False,
)


# ============================================================
# 13. PAIRED PARTICIPANT BOOTSTRAP
# ============================================================

print("\n[9/10] Calculating paired participant bootstrap...")

pivot = participant_macro.pivot(
    index="subject",
    columns="condition",
    values="participant_macro_f1",
)

if not {
    "no_reference",
    "geometric_median",
}.issubset(pivot.columns):
    raise RuntimeError(
        "Missing primary comparison conditions"
    )

paired = pivot[
    ["no_reference", "geometric_median"]
].dropna()

differences = (
    paired["geometric_median"]
    - paired["no_reference"]
).to_numpy(dtype=float)

if len(differences) != 6:
    raise RuntimeError(
        "Expected paired results for six locked participants"
    )

bootstrap_sample = rng.choice(
    differences,
    size=(BOOTSTRAPS, len(differences)),
    replace=True,
)

bootstrap_means = bootstrap_sample.mean(axis=1)

bootstrap_result = {
    "comparison": "geometric_median_minus_no_reference",
    "participants": int(len(differences)),
    "mean_f1_difference": float(differences.mean()),
    "ci_2_5": float(
        np.percentile(bootstrap_means, 2.5)
    ),
    "ci_97_5": float(
        np.percentile(bootstrap_means, 97.5)
    ),
    "participants_improved": int(
        np.sum(differences > 0)
    ),
    "participants_worsened": int(
        np.sum(differences < 0)
    ),
    "participants_unchanged": int(
        np.sum(differences == 0)
    ),
    "bootstrap_replicates": BOOTSTRAPS,
    "bootstrap_unit": "locked participant",
    "interpretation": (
        "Exploratory interval based on six participants; "
        "not definitive statistical confirmation."
    ),
}

pd.DataFrame([bootstrap_result]).to_csv(
    OUT / "V9_17_PAIRED_PARTICIPANT_BOOTSTRAP.csv",
    index=False,
)

# Save individual paired differences.
paired.reset_index().assign(
    f1_difference=differences
).to_csv(
    OUT / "V9_17_PAIRED_PARTICIPANT_DIFFERENCES.csv",
    index=False,
)


# ============================================================
# 14. POST-HOC CALIBRATION CONTAMINATION AUDIT
# ============================================================

print("\n[10/10] Post-hoc calibration contamination audit...")

# Test labels are accessed here only AFTER predictions and
# all model fitting have been completed.

contamination_rows = []

calibration_any_artifact = np.zeros(
    len(test_cal),
    dtype=bool,
)

for cls in CLASSES:

    values = pd.to_numeric(
        test_df.iloc[test_cal][test_label_columns[cls]],
        errors="coerce",
    ).fillna(0).to_numpy()

    calibration_any_artifact |= (
        values > 0
    )

cal_subject_vector = (
    test_df.iloc[test_cal][TEST_SUB]
    .to_numpy()
)

for subject in locked_subjects:

    subject_mask = (
        cal_subject_vector == subject
    )

    count = int(subject_mask.sum())

    contaminated = int(
        calibration_any_artifact[subject_mask].sum()
    )

    contamination_rows.append({
        "subject": subject,
        "calibration_frames": count,
        "artifact_contaminated_frames_posthoc": contaminated,
        "artifact_contamination_fraction_posthoc": (
            float(contaminated / count)
            if count else np.nan
        ),
    })

contamination_df = pd.DataFrame(
    contamination_rows
)

contamination_df.to_csv(
    OUT / "V9_17_POSTHOC_CALIBRATION_AUDIT.csv",
    index=False,
)


# ============================================================
# 15. INTEGRITY CHECKS AND REPORT
# ============================================================

INPUT_HASHES_AFTER = {
    str(path): sha256_file(path)
    for path in [
        DEV_FEATURE_FILE,
        DEV_META_FILE,
        TEST_FEATURE_FILE,
        TEST_META_FILE,
    ]
}

input_hashes_unchanged = (
    INPUT_HASHES_BEFORE == INPUT_HASHES_AFTER
)

if not input_hashes_unchanged:
    raise RuntimeError(
        "An input file changed during the experiment"
    )

if set(predictions_df["subject"].unique()) != LOCKED:
    raise RuntimeError(
        "Prediction participant identity mismatch"
    )

expected_prediction_rows = (
    len(test_eval)
    * len(CLASSES)
    * len(CONDITIONS)
)

if len(predictions_df) != expected_prediction_rows:
    raise RuntimeError(
        "Unexpected number of prediction rows"
    )

if predictions_df["probability"].isna().any():
    raise RuntimeError(
        "Prediction probabilities contain NaN"
    )

if not predictions_df["probability"].between(
    0, 1
).all():
    raise RuntimeError(
        "Prediction probability outside [0,1]"
    )

if predictions_df.duplicated(
    ["condition", "class", "subject", "recording", "frame_index"]
).any():
    raise RuntimeError(
        "Duplicate prediction keys detected"
    )

report = {
    "experiment": "V9-17 Locked Participant Evaluation",
    "completed": True,
    "feature_dimension": int(X_dev.shape[1]),
    "development_feature_shape": list(X_dev.shape),
    "locked_feature_shape": list(X_test.shape),
    "training_participants": train_subjects,
    "development_validation_participants_not_used": val_subjects,
    "locked_evaluation_participants": locked_subjects,
    "training_participant_count": len(train_subjects),
    "locked_participant_count": len(locked_subjects),
    "development_recordings": int(dev_recordings),
    "locked_recordings": int(test_recordings),
    "training_evaluation_frames": int(len(train_eval)),
    "locked_calibration_frames": int(len(test_cal)),
    "locked_evaluation_frames": int(len(test_eval)),
    "calibration_fraction": CALIBRATION_FRACTION,
    "evaluation_window": "final 50 percent of each recording",
    "reference_method": "Weiszfeld geometric median",
    "reference_estimation_uses_features_only": True,
    "class_adaptive_alphas": ALPHAS,
    "conditions": CONDITIONS,
    "classifier": "LogisticRegression",
    "classifier_C": 1.0,
    "classifier_max_iter": 1000,
    "scaler": "StandardScaler fitted on training evaluation rows only",
    "threshold": THRESHOLD,
    "locked_labels_used_for_calibration": False,
    "locked_labels_used_for_training": False,
    "locked_labels_used_for_threshold_selection": False,
    "locked_labels_used_for_final_scoring": True,
    "locked_labels_used_for_posthoc_contamination_audit": True,
    "validation_participants_used_for_training": False,
    "locked_participants_used_for_training": False,
    "calibration_evaluation_overlap": 0,
    "input_hashes_unchanged": input_hashes_unchanged,
    "input_hashes_before": INPUT_HASHES_BEFORE,
    "input_hashes_after": INPUT_HASHES_AFTER,
    "prediction_rows": int(len(predictions_df)),
    "participant_metric_rows": int(len(participant_df)),
    "paired_bootstrap": bootstrap_result,
    "elapsed_seconds": round(
        time.time() - START,
        2,
    ),
    "scientific_limitations": [
        "Only six locked participants are available.",
        "Bootstrap intervals are exploratory.",
        "The locked participants were previously used in V6; "
        "they are locked for V9 but are not a completely independent "
        "external dataset for the entire project.",
        "The evaluation uses participant-specific unlabeled calibration.",
        "The evaluation uses frame-level metrics and does not "
        "establish event-level clinical performance.",
    ],
}

with open(
    REPORT_DIR / "V9_17_REPORT.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        report,
        file,
        indent=4,
    )

print("\n" + "=" * 75)
print("V9-17 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nPOOLED FRAME METRICS")
print(
    pooled_df[
        [
            "condition",
            "class",
            "f1",
            "precision",
            "recall",
            "pr_auc",
            "roc_auc",
        ]
    ].to_string(index=False)
)

print("\nPARTICIPANT-LEVEL OVERALL SUMMARY")
print(overall_summary.to_string(index=False))

print("\nPAIRED PARTICIPANT BOOTSTRAP")
print(json.dumps(bootstrap_result, indent=4))

print("\nPOST-HOC CALIBRATION CONTAMINATION")
print(contamination_df.to_string(index=False))

print("\nIntegrity:")
print("Input hashes unchanged:", input_hashes_unchanged)
print("Locked participants used for training: NO")
print("Locked labels used for calibration: NO")
print("Locked labels used for threshold selection: NO")
print("Calibration/evaluation overlap: 0")
print("Prediction rows:", len(predictions_df))

print("\nResults directory:")
print(OUT)

print("\nReport:")
print(REPORT_DIR / "V9_17_REPORT.json")

print("\nElapsed seconds:", round(time.time() - START, 2))

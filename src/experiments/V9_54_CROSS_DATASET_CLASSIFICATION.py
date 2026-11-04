
from pathlib import Path
from datetime import datetime
import csv
import json
import math
import warnings

import numpy as np

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
    confusion_matrix,
)

# ============================================================
# V9-54: NABİYOUNI INDEPENDENT-PARTICIPANT CLASSIFICATION
#
# Dataset: Nabiyouni EEG Artifact Dataset
#
# Calibration:
#   Repetition 1, without calibration labels
#
# Supervised training/evaluation:
#   Repetition 2
#
# Training participants:
#   Subj_1 ... Subj_7
#
# Independent test participants:
#   Subj_8 ... Subj_10
#
# Methods:
#   1. no_reference
#   2. mean_center
#   3. geometric_median
#
# Important:
#   This is a six-class, single-channel condition-classification
#   pilot. It is NOT direct validation of the original 34-channel
#   V9 classifier or frame-level artifact event detection.
#
# Original EEG files are read-only.
# Existing V8/V9 models are not loaded or modified.
# ============================================================

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
V9_ROOT = SCRIPT_DIR.parent

DATA_ROOT = (
    V9_ROOT
    / "External_Validation"
    / "Nabiyouni_EEG"
)

RESULT_DIR = (
    V9_ROOT
    / "Results"
    / "V9_54_CROSS_DATASET_CLASSIFICATION"
)

REPORT_DIR = (
    V9_ROOT
    / "Reports"
    / "V9_54_CROSS_DATASET_CLASSIFICATION"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ------------------------------------------------------------

SAMPLE_RATE = 1000
SEGMENT_SIZE = 1000
EPS = 1e-12

TRAIN_SUBJECTS = [
    f"Subj_{i}" for i in range(1, 8)
]

TEST_SUBJECTS = [
    f"Subj_{i}" for i in range(8, 11)
]

ALL_SUBJECTS = TRAIN_SUBJECTS + TEST_SUBJECTS

CONDITIONS = {
    "blink": "eye",
    "eye_left": "eye",
    "eye_up": "eye",
    "eyebrow": "eyebrow",
    "head": "head_motion",
    "jaw_cln": "jaw_clench",
    "jaw_mov": "jaw_movement",
    "none": "baseline",
}

CLASSES = [
    "baseline",
    "eye",
    "eyebrow",
    "head_motion",
    "jaw_clench",
    "jaw_movement",
]

METHODS = [
    "no_reference",
    "mean_center",
    "geometric_median",
]

FEATURE_NAMES = [
    "log_rms",
    "log_std",
    "log_peak_to_peak",
    "log_median_abs",
    "log_p95_abs",
    "log_line_length",
    "zero_crossing_rate",
    "skewness",
    "excess_kurtosis",
    "log_delta_power",
    "log_theta_power",
    "log_alpha_power",
    "log_beta_power",
    "log_gamma_power",
    "spectral_entropy",
]


# ============================================================
# 1. READ EEG SIGNAL
# ============================================================

def read_signal(path):
    values = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as file:

        for line in file:

            for token in line.strip().split("\t"):

                token = token.strip()

                if not token:
                    continue

                try:
                    value = float(token)

                    if math.isfinite(value):
                        values.append(value)

                except ValueError:
                    continue

    if not values:
        raise ValueError(
            f"No valid numeric EEG samples found: {path}"
        )

    return np.asarray(values, dtype=np.float64)


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================

def extract_features(segment):
    """
    Extract 15 features from one complete one-second segment.
    """

    x = np.asarray(segment, dtype=np.float64)

    if len(x) != SEGMENT_SIZE:
        raise ValueError(
            f"Expected {SEGMENT_SIZE} samples, got {len(x)}"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite EEG values detected.")

    # Remove segment mean for feature calculation.
    x = x - np.mean(x)

    # Time-domain features.
    rms = np.sqrt(np.mean(x ** 2))
    std = np.std(x)
    peak_to_peak = np.ptp(x)

    abs_x = np.abs(x)

    median_abs = np.median(abs_x)
    p95_abs = np.percentile(abs_x, 95)

    line_length = np.mean(
        np.abs(np.diff(x))
    )

    zero_crossing_rate = np.mean(
        (x[:-1] * x[1:]) < 0
    )

    if std > EPS:

        standardized = x / std

        skewness = np.mean(
            standardized ** 3
        )

        excess_kurtosis = (
            np.mean(standardized ** 4) - 3.0
        )

    else:

        skewness = 0.0
        excess_kurtosis = 0.0

    # Frequency-domain features.
    spectrum = np.abs(
        np.fft.rfft(x)
    ) ** 2

    frequencies = np.fft.rfftfreq(
        len(x),
        d=1.0 / SAMPLE_RATE,
    )

    band_ranges = [
        (0.5, 4.0),
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 30.0),
        (30.0, 100.0),
    ]

    band_powers = []

    for low, high in band_ranges:

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        power = np.sum(
            spectrum[mask]
        )

        band_powers.append(power)

    total_power = float(
        np.sum(band_powers)
    )

    if total_power > EPS:

        probabilities = (
            np.asarray(band_powers)
            / total_power
        )

        spectral_entropy = -np.sum(
            probabilities
            * np.log(probabilities + EPS)
        )

    else:

        spectral_entropy = 0.0

    features = [
        np.log(rms + EPS),
        np.log(std + EPS),
        np.log(peak_to_peak + EPS),
        np.log(median_abs + EPS),
        np.log(p95_abs + EPS),
        np.log(line_length + EPS),
        zero_crossing_rate,
        skewness,
        excess_kurtosis,
        *[
            np.log(power + EPS)
            for power in band_powers
        ],
        spectral_entropy,
    ]

    result = np.asarray(
        features,
        dtype=np.float64,
    )

    if result.shape != (len(FEATURE_NAMES),):
        raise RuntimeError(
            "Unexpected feature vector dimensions."
        )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "Non-finite feature values detected."
        )

    return result


# ============================================================
# 3. GEOMETRIC MEDIAN
# ============================================================

def geometric_median(
    points,
    tolerance=1e-7,
    max_iterations=500,
):
    """
    Weiszfeld geometric median estimator.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.ndim != 2 or len(points) == 0:
        raise ValueError(
            "Geometric median requires a non-empty 2D array."
        )

    estimate = np.median(
        points,
        axis=0,
    )

    for _ in range(max_iterations):

        distances = np.linalg.norm(
            points - estimate,
            axis=1,
        )

        nearest_index = np.argmin(
            distances
        )

        if distances[nearest_index] < tolerance:

            candidate = points[nearest_index]

        else:

            weights = (
                1.0
                / np.maximum(distances, EPS)
            )

            candidate = np.sum(
                points * weights[:, None],
                axis=0,
            ) / np.sum(weights)

        change = np.linalg.norm(
            candidate - estimate
        )

        estimate = candidate

        if change < tolerance:
            break

    return estimate


# ============================================================
# 4. TRAINING-ONLY ROBUST GLOBAL SCALE
# ============================================================

def robust_global_scale(training_calibration):
    """
    Calculate feature-wise robust location and scale using
    training participants' unlabeled calibration features only.
    """

    pooled = np.vstack([
        training_calibration[subject]
        for subject in TRAIN_SUBJECTS
    ])

    location = np.median(
        pooled,
        axis=0,
    )

    mad = np.median(
        np.abs(pooled - location),
        axis=0,
    )

    scale = 1.4826 * mad

    fallback = np.std(
        pooled,
        axis=0,
    )

    scale = np.where(
        scale > EPS,
        scale,
        fallback,
    )

    scale = np.where(
        scale > EPS,
        scale,
        1.0,
    )

    return location, scale


# ============================================================
# 5. PARTICIPANT CALIBRATION
# ============================================================

def participant_center(
    calibration,
    method,
    global_location,
    global_scale,
):

    if method == "no_reference":

        return np.zeros(
            calibration.shape[1],
            dtype=np.float64,
        )

    if method == "mean_center":

        return np.mean(
            calibration,
            axis=0,
        )

    if method == "geometric_median":

        standardized = (
            calibration - global_location
        ) / global_scale

        center_standardized = geometric_median(
            standardized
        )

        center = (
            global_location
            + center_standardized * global_scale
        )

        return center

    raise ValueError(
        f"Unknown calibration method: {method}"
    )


# ============================================================
# 6. LOAD ONE PARTICIPANT
# ============================================================

def load_participant(subject):

    subject_dir = DATA_ROOT / subject

    if not subject_dir.exists():
        raise FileNotFoundError(
            f"Participant directory not found: {subject_dir}"
        )

    calibration_features = []
    evaluation_features = []
    evaluation_labels = []
    evaluation_recordings = []

    files = sorted(
        subject_dir.glob("*.txt")
    )

    if not files:
        raise FileNotFoundError(
            f"No TXT recordings found for {subject}"
        )

    for path in files:

        stem = path.stem

        if "_" not in stem:
            continue

        condition_name, repetition = stem.rsplit(
            "_",
            1,
        )

        if condition_name not in CONDITIONS:
            continue

        if repetition not in ("1", "2"):
            continue

        signal = read_signal(path)

        complete_segments = (
            len(signal) // SEGMENT_SIZE
        )

        if complete_segments == 0:
            raise ValueError(
                f"No complete one-second segments in {path}"
            )

        # Ignore only an incomplete trailing segment.
        for index in range(complete_segments):

            start = index * SEGMENT_SIZE
            end = start + SEGMENT_SIZE

            feature = extract_features(
                signal[start:end]
            )

            if repetition == "1":

                # Calibration labels are deliberately not stored.
                calibration_features.append(
                    feature
                )

            else:

                evaluation_features.append(
                    feature
                )

                evaluation_labels.append(
                    CONDITIONS[condition_name]
                )

                evaluation_recordings.append(
                    path.name
                )

    if not calibration_features:
        raise ValueError(
            f"No calibration data found for {subject}"
        )

    if not evaluation_features:
        raise ValueError(
            f"No evaluation data found for {subject}"
        )

    return {
        "calibration": np.vstack(
            calibration_features
        ),

        "X": np.vstack(
            evaluation_features
        ),

        "y": np.asarray(
            evaluation_labels,
            dtype=str,
        ),

        "recordings": np.asarray(
            evaluation_recordings,
            dtype=str,
        ),
    }


# ============================================================
# 7. CSV OUTPUT
# ============================================================

def save_csv(path, rows):

    if not rows:
        print(
            f"WARNING: No rows to save: {path.name}"
        )
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 8. CLASSIFICATION METRICS
# ============================================================

def evaluate_model(
    y_true,
    y_pred,
    probabilities,
    participant,
    method,
):

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASSES,
            zero_division=0,
        )
    )

    binary_truth = label_binarize(
        y_true,
        classes=CLASSES,
    )

    per_class_ap = average_precision_score(
        binary_truth,
        probabilities,
        average=None,
    )

    class_metric_rows = []

    for i, class_name in enumerate(CLASSES):

        class_metric_rows.append({

            "method": method,

            "participant": participant,

            "class": class_name,

            "precision": float(
                precision[i]
            ),

            "recall": float(
                recall[i]
            ),

            "f1": float(
                f1[i]
            ),

            "support": int(
                support[i]
            ),

            "average_precision": float(
                per_class_ap[i]
            ),
        })

    overall = {

        "method": method,

        "participant": participant,

        "samples": int(
            len(y_true)
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "macro_f1": float(
            np.mean(f1)
        ),

        "macro_precision": float(
            np.mean(precision)
        ),

        "macro_recall": float(
            np.mean(recall)
        ),

        "macro_average_precision": float(
            np.mean(per_class_ap)
        ),
    }

    return overall, class_metric_rows


# ============================================================
# 9. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 72)
    print("V9-54: NABİYOUNI INDEPENDENT-PARTICIPANT CLASSIFICATION")
    print("=" * 72)

    print("\nDataset:")
    print(DATA_ROOT)

    if not DATA_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist: {DATA_ROOT}"
        )

    # --------------------------------------------------------
    # LOAD ALL PARTICIPANTS
    # --------------------------------------------------------

    print(
        "\nLoading participants and extracting features..."
    )

    participant_data = {}

    for subject in ALL_SUBJECTS:

        participant_data[subject] = load_participant(
            subject
        )

        item = participant_data[subject]

        print(
            f"{subject}: "
            f"calibration={len(item['calibration'])} | "
            f"evaluation={len(item['X'])}"
        )

    # --------------------------------------------------------
    # VERIFY CLASS COVERAGE
    # --------------------------------------------------------

    print(
        "\nChecking class coverage..."
    )

    training_labels = np.concatenate([
        participant_data[subject]["y"]
        for subject in TRAIN_SUBJECTS
    ])

    for class_name in CLASSES:

        if not np.any(
            training_labels == class_name
        ):

            raise ValueError(
                f"Training group is missing class: {class_name}"
            )

    for subject in TEST_SUBJECTS:

        labels = participant_data[subject]["y"]

        for class_name in CLASSES:

            if not np.any(
                labels == class_name
            ):

                raise ValueError(
                    f"{subject} is missing class: {class_name}"
                )

    print(
        "PASS: All six classes exist in training and every test participant."
    )

    # --------------------------------------------------------
    # VERIFY PARTICIPANT SEPARATION
    # --------------------------------------------------------

    overlap = set(TRAIN_SUBJECTS).intersection(
        TEST_SUBJECTS
    )

    if overlap:

        raise RuntimeError(
            f"Training/test participant overlap: {overlap}"
        )

    print(
        "PASS: No participant overlap."
    )

    # --------------------------------------------------------
    # TRAINING CALIBRATION DATA ONLY
    # --------------------------------------------------------

    training_calibration = {

        subject: participant_data[subject]["calibration"]

        for subject in TRAIN_SUBJECTS
    }

    global_location, global_scale = (
        robust_global_scale(
            training_calibration
        )
    )

    print(
        "\nTraining-only robust feature location and scale calculated."
    )

    # --------------------------------------------------------
    # RESULT CONTAINERS
    # --------------------------------------------------------

    overall_rows = []
    class_rows = []
    prediction_rows = []
    confusion_rows = []
    calibration_rows = []

    # --------------------------------------------------------
    # RUN EACH CALIBRATION METHOD
    # --------------------------------------------------------

    for method in METHODS:

        print("\n" + "-" * 68)
        print("METHOD:", method)
        print("-" * 68)

        train_adjusted = []
        train_labels = []

        # ----------------------------------------------------
        # TRAINING PARTICIPANTS
        # ----------------------------------------------------

        for subject in TRAIN_SUBJECTS:

            data = participant_data[subject]

            center = participant_center(
                data["calibration"],
                method,
                global_location,
                global_scale,
            )

            adjusted_features = (
                data["X"] - center
            )

            train_adjusted.append(
                adjusted_features
            )

            train_labels.append(
                data["y"]
            )

            calibration_rows.append({

                "method": method,

                "participant": subject,

                "group": "training",

                "calibration_segments": int(
                    len(data["calibration"])
                ),

                "center_l2_norm": float(
                    np.linalg.norm(center)
                ),
            })

        # ----------------------------------------------------
        # BUILD TRAINING MATRIX
        # ----------------------------------------------------

        X_train = np.vstack(
            train_adjusted
        )

        y_train_method = np.concatenate(
            train_labels
        )

        # ----------------------------------------------------
        # STANDARDIZATION
        # ----------------------------------------------------

        # Fit scaler on training data only.
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        # ----------------------------------------------------
        # LOGISTIC REGRESSION
        # ----------------------------------------------------

        model = LogisticRegression(

            C=1.0,

            max_iter=3000,

            class_weight="balanced",

            solver="lbfgs",

            random_state=42,
        )

        model.fit(
            X_train_scaled,
            y_train_method,
        )

        if list(model.classes_) != CLASSES:

            raise RuntimeError(
                "Unexpected classifier class ordering: "
                f"{list(model.classes_)}"
            )

        print(
            "Training samples:",
            len(y_train_method),
        )

        print(
            "Classifier trained successfully."
        )

        # ----------------------------------------------------
        # INDEPENDENT TEST PARTICIPANTS
        # ----------------------------------------------------

        for subject in TEST_SUBJECTS:

            data = participant_data[subject]

            center = participant_center(
                data["calibration"],
                method,
                global_location,
                global_scale,
            )

            X_test_adjusted = (
                data["X"] - center
            )

            # The training scaler is reused unchanged.
            X_test_scaled = scaler.transform(
                X_test_adjusted
            )

            probabilities = model.predict_proba(
                X_test_scaled
            )

            predictions = model.predict(
                X_test_scaled
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            overall, class_metrics = evaluate_model(
                data["y"],
                predictions,
                probabilities,
                subject,
                method,
            )

            overall_rows.append(
                overall
            )

            class_rows.extend(
                class_metrics
            )

            # ------------------------------------------------
            # TEST CALIBRATION SUMMARY
            # ------------------------------------------------

            calibration_rows.append({

                "method": method,

                "participant": subject,

                "group": "test",

                "calibration_segments": int(
                    len(data["calibration"])
                ),

                "center_l2_norm": float(
                    np.linalg.norm(center)
                ),
            })

            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            matrix = confusion_matrix(
                data["y"],
                predictions,
                labels=CLASSES,
            )

            for i, actual_class in enumerate(CLASSES):

                for j, predicted_class in enumerate(CLASSES):

                    confusion_rows.append({

                        "method": method,

                        "participant": subject,

                        "actual_class": actual_class,

                        "predicted_class": predicted_class,

                        "count": int(
                            matrix[i, j]
                        ),
                    })

            # ------------------------------------------------
            # INDIVIDUAL PREDICTIONS
            # ------------------------------------------------

            for i in range(len(data["y"])):

                prediction_record = {

                    "method": method,

                    "participant": subject,

                    "recording": data["recordings"][i],

                    "segment_index_within_participant": i,

                    "true_class": data["y"][i],

                    "predicted_class": predictions[i],

                    "correct": bool(
                        data["y"][i] == predictions[i]
                    ),
                }

                for j, class_name in enumerate(CLASSES):

                    prediction_record[
                        f"prob_{class_name}"
                    ] = float(
                        probabilities[i, j]
                    )

                prediction_rows.append(
                    prediction_record
                )

            # ------------------------------------------------
            # CONSOLE OUTPUT
            # ------------------------------------------------

            print(

                f"{subject}: "

                f"Macro-F1={overall['macro_f1']:.4f} | "

                f"Balanced accuracy="
                f"{overall['balanced_accuracy']:.4f} | "

                f"Macro AP="
                f"{overall['macro_average_precision']:.4f}"
            )

    # ========================================================
    # 10. AGGREGATE PARTICIPANT-LEVEL RESULTS
    # ========================================================

    aggregate_rows = []

    for method in METHODS:

        rows = [

            row for row in overall_rows

            if row["method"] == method
        ]

        if len(rows) != len(TEST_SUBJECTS):

            raise RuntimeError(
                f"Incomplete test results for method: {method}"
            )

        aggregate_rows.append({

            "method": method,

            "test_participants": len(rows),

            "total_test_segments": int(
                sum(
                    row["samples"]
                    for row in rows
                )
            ),

            "participant_macro_f1_mean": float(
                np.mean([
                    row["macro_f1"]
                    for row in rows
                ])
            ),

            "participant_macro_f1_std": float(
                np.std([
                    row["macro_f1"]
                    for row in rows
                ])
            ),

            "participant_balanced_accuracy_mean": float(
                np.mean([
                    row["balanced_accuracy"]
                    for row in rows
                ])
            ),

            "participant_macro_ap_mean": float(
                np.mean([
                    row["macro_average_precision"]
                    for row in rows
                ])
            ),
        })

    # ========================================================
    # 11. SAVE CSV RESULTS
    # ========================================================

    print(
        "\nSaving experiment results..."
    )

    save_csv(
        RESULT_DIR / "V9_54_PARTICIPANT_METRICS.csv",
        overall_rows,
    )

    save_csv(
        RESULT_DIR / "V9_54_CLASS_METRICS.csv",
        class_rows,
    )

    save_csv(
        RESULT_DIR / "V9_54_PREDICTIONS.csv",
        prediction_rows,
    )

    save_csv(
        RESULT_DIR / "V9_54_CONFUSION_COUNTS.csv",
        confusion_rows,
    )

    save_csv(
        RESULT_DIR / "V9_54_CALIBRATION_SUMMARY.csv",
        calibration_rows,
    )

    save_csv(
        RESULT_DIR / "V9_54_METHOD_COMPARISON.csv",
        aggregate_rows,
    )

    # ========================================================
    # 12. SAVE JSON SUMMARY
    # ========================================================

    total_training_segments = int(
        len(training_labels)
    )

    total_test_segments = int(
        sum(
            len(participant_data[subject]["y"])
            for subject in TEST_SUBJECTS
        )
    )

    summary = {

        "experiment": "V9-54",

        "experiment_name": (
            "Nabiyouni independent-participant "
            "six-class condition classification"
        ),

        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),

        "dataset": str(DATA_ROOT),

        "training_participants": TRAIN_SUBJECTS,

        "test_participants": TEST_SUBJECTS,

        "participant_overlap": [],

        "calibration_repetition": 1,

        "evaluation_repetition": 2,

        "calibration_labels_used": False,

        "classes": CLASSES,

        "features": FEATURE_NAMES,

        "feature_count": len(FEATURE_NAMES),

        "sample_rate_hz": SAMPLE_RATE,

        "segment_duration_seconds": (
            SEGMENT_SIZE / SAMPLE_RATE
        ),

        "classifier": "LogisticRegression",

        "classifier_C": 1.0,

        "class_weight": "balanced",

        "scaler_fit_on_training_only": True,

        "global_calibration_location_scale_fit_on_training_only": True,

        "methods": METHODS,

        "training_segments": total_training_segments,

        "test_segments": total_test_segments,

        "results": aggregate_rows,

        "v8_modified": False,

        "existing_v9_models_modified": False,

        "source_data_modified": False,

        "test_labels_used_for_calibration": False,

        "interpretation": (

            "Exploratory six-class condition-classification pilot "
            "using an independent participant split within the "
            "Nabiyouni dataset. Repetition 1 is used as an unlabeled "
            "calibration feature pool. Repetition 2 is used for "
            "supervised training or held-out participant evaluation. "
            "Labels are inherited from recording filenames. "
            "This is not direct validation of the original "
            "34-channel V9 classifier and is not frame-level "
            "artifact event detection. Calibration pools contain "
            "a mixture of recording conditions."
        ),
    }

    summary_path = (
        RESULT_DIR / "V9_54_SUMMARY.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ========================================================
    # 13. HUMAN-READABLE REPORT
    # ========================================================

    lines = [

        "# V9-54: Nabiyouni Independent-Participant Classification",

        "",

        f"Date: {summary['timestamp']}",

        "",

        "## Experimental protocol",

        "",

        "- Dataset: Nabiyouni EEG Artifact Dataset",

        "- Training participants: Subj_1 to Subj_7",

        "- Independent test participants: Subj_8 to Subj_10",

        "- Calibration: repetition 1, without labels",

        "- Supervised training/evaluation: repetition 2",

        "- Sampling rate: 1000 Hz",

        "- Segment duration: 1 second",

        "- Features: 15 time-domain and frequency-domain features",

        "- Classes: six filename-derived conditions",

        "- Classifier: logistic regression",

        "- Feature scaling: fitted on training data only",

        "",

        "## Participant-level results",

        "",

        "| Method | Mean macro-F1 | SD | Mean balanced accuracy | Mean macro AP |",

        "|---|---:|---:|---:|---:|",
    ]

    for row in aggregate_rows:

        lines.append(

            f"| {row['method']} "

            f"| {row['participant_macro_f1_mean']:.4f} "

            f"| {row['participant_macro_f1_std']:.4f} "

            f"| {row['participant_balanced_accuracy_mean']:.4f} "

            f"| {row['participant_macro_ap_mean']:.4f} |"
        )

    lines += [

        "",

        "## Scientific limitations",

        "",

        "1. This is a single-channel, six-class condition-classification pilot, not the original 34-channel artifact detector.",

        "2. Training and test participants come from the same Nabiyouni dataset. This is participant-held-out evaluation, not a strict train-on-one-dataset/test-on-another-dataset experiment.",

        "3. Labels are inherited from recording filenames, not independently annotated artifact boundaries.",

        "4. Repetition 1 is an unlabeled calibration pool containing a mixture of recording conditions.",

        "5. Repetition 2 is used for supervised training or held-out testing.",

        "6. Only three unseen participants are tested. Results are exploratory and do not establish broad population generalization.",

        "7. The one-second segmentation does not establish exact artifact-event boundaries.",

        "8. No hyperparameter tuning was performed on test participants.",

        "9. Physical signal units have not been assumed.",

        "",

        "## Integrity",

        "",

        "- Original EEG files modified: No",

        "- Existing V8 model modified: No",

        "- Existing V9 models modified: No",

        "- Test labels used for calibration: No",

        "- Test participants used for training: No",

        "- Training and test participant overlap: None",

        "",
    ]

    report_path = (
        REPORT_DIR
        / "V9_54_CROSS_DATASET_CLASSIFICATION_REPORT.md"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # ========================================================
    # 14. FINAL CONSOLE SUMMARY
    # ========================================================

    print("\n" + "=" * 72)
    print("V9-54 EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 72)

    print(
        "\nTraining participants:",
        len(TRAIN_SUBJECTS),
    )

    print(
        "Testing participants:",
        len(TEST_SUBJECTS),
    )

    print(
        "Training segments:",
        total_training_segments,
    )

    print(
        "Testing segments:",
        total_test_segments,
    )

    print(
        "\nFINAL METHOD COMPARISON"
    )

    for row in aggregate_rows:

        print(

            f"{row['method']:20s} | "

            f"Macro-F1="
            f"{row['participant_macro_f1_mean']:.4f} | "

            f"Balanced Acc="
            f"{row['participant_balanced_accuracy_mean']:.4f} | "

            f"Macro AP="
            f"{row['participant_macro_ap_mean']:.4f}"
        )

    print(
        "\nResults directory:"
    )

    print(
        RESULT_DIR
    )

    print(
        "\nReport:"
    )

    print(
        report_path
    )

    print(
        "\nJSON summary:"
    )

    print(
        summary_path
    )

    print(
        "\nOriginal dataset and existing V8/V9 models were not modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

# ============================================================
# V9-17: LOCKED PARTICIPANT DATA PREFLIGHT
# READ-ONLY AUDIT — DOES NOT MODIFY V8 DATA
# ============================================================

ROOT = Path(os.environ["EEG_V7_ROOT"])
HELDOUT = (
    ROOT
    / "V8_Development"
    / "Results"
    / "V8_HELDOUT_TEST"
)

DEV = (
    ROOT
    / "V8_Development"
    / "Data"
    / "V8_NORMALIZED"
)

FEATURE_FILE = HELDOUT / "V8_TEST_NORMALIZED_FEATURES.npy"
RAW_FILE = HELDOUT / "V8_TEST_RAW_FEATURES.npy"
METADATA_FILE = HELDOUT / "V8_TEST_LABELS_METADATA.csv"
DEV_FEATURE_FILE = DEV / "V8_0P5S_34CH_FEATURES_ROBUST.npy"
DEV_METADATA_FILE = DEV / "V8_0P5S_METADATA.csv"

EXPECTED_LOCKED = {
    "sub-2",
    "sub-7",
    "sub-14",
    "sub-25",
    "sub-26",
    "sub-27",
}


def section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def inspect_array(path, name):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    array = np.load(path, mmap_mode="r")

    print(f"\n{name}")
    print("Path:", path)
    print("Shape:", array.shape)
    print("Dtype:", array.dtype)
    print("Dimensions:", array.ndim)

    if array.ndim >= 2:
        print("Number of frames:", array.shape[0])

    print("File size (MB):", round(path.stat().st_size / (1024 ** 2), 3))

    # Inspect finite values without modifying the source.
    sample = np.asarray(array[:min(1000, len(array))])

    print("First-sample finite:", bool(np.isfinite(sample).all()))
    print("First-sample min:", float(np.nanmin(sample)))
    print("First-sample max:", float(np.nanmax(sample)))

    return array


def main():

    section("1. LOCKED PARTICIPANT FEATURE ARRAYS")

    normalized = inspect_array(
        FEATURE_FILE,
        "LOCKED NORMALIZED FEATURES"
    )

    raw = inspect_array(
        RAW_FILE,
        "LOCKED RAW FEATURES"
    )

    section("2. LOCKED PARTICIPANT METADATA")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(METADATA_FILE)

    metadata = pd.read_csv(METADATA_FILE)

    print("Metadata path:", METADATA_FILE)
    print("Metadata rows:", len(metadata))
    print("Metadata columns:", len(metadata.columns))
    print("\nColumn names:")
    print(metadata.columns.tolist())

    print("\nFirst five rows:")
    print(metadata.head().to_string(index=False))

    subject_candidates = [
        col for col in metadata.columns
        if col.lower() in {
            "subject",
            "participant",
            "subject_id",
            "participant_id"
        }
    ]

    if not subject_candidates:
        raise ValueError(
            "Could not identify participant column. "
            "Inspect the metadata column names."
        )

    subject_col = subject_candidates[0]

    subjects = sorted(
        metadata[subject_col].astype(str).unique().tolist()
    )

    print("\nParticipant column:", subject_col)
    print("Participants:", subjects)

    print("\nFrames per participant:")
    print(
        metadata[subject_col]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_string()
    )

    section("3. DEVELOPMENT DATA COMPARISON")

    dev_features = np.load(
        DEV_FEATURE_FILE,
        mmap_mode="r"
    )

    dev_metadata = pd.read_csv(DEV_METADATA_FILE)

    print("Development feature shape:", dev_features.shape)
    print("Development metadata rows:", len(dev_metadata))

    print(
        "Locked feature shape:",
        normalized.shape
    )

    print(
        "Locked metadata rows:",
        len(metadata)
    )

    section("4. DATA INTEGRITY CHECKS")

    checks = {
        "normalized_rows_match_metadata":
            normalized.shape[0] == len(metadata),

        "raw_rows_match_metadata":
            raw.shape[0] == len(metadata),

        "raw_and_normalized_shapes_match":
            raw.shape == normalized.shape,

        "locked_participants_exact":
            set(subjects) == EXPECTED_LOCKED,

        "no_development_participants_in_locked_metadata":
            set(subjects).isdisjoint(
                set(dev_metadata[subject_col].astype(str).unique())
            ),

        "locked_features_are_3d":
            normalized.ndim == 3,

        "expected_34_channels":
            normalized.ndim == 3 and normalized.shape[1] == 34,

        "expected_14_features":
            normalized.ndim == 3 and normalized.shape[2] == 14,

        "metadata_has_unique_subject_recording_frame_keys":
            all(
                col in metadata.columns
                for col in [
                    subject_col,
                    "recording",
                    "frame_index"
                ]
            )
    }

    if checks["metadata_has_unique_subject_recording_frame_keys"]:
        checks["metadata_keys_unique"] = not metadata.duplicated(
            [subject_col, "recording", "frame_index"]
        ).any()

    for name, passed in checks.items():
        print(
            f"{'PASS' if passed else 'FAIL'}: {name}"
        )

    section("5. LABEL AND RECORDING INFORMATION")

    label_candidates = [
        col for col in metadata.columns
        if any(
            term in col.lower()
            for term in [
                "eye",
                "jaw",
                "motion",
                "clean",
                "label"
            ]
        )
    ]

    print("Potential label columns:", label_candidates)

    recording_candidates = [
        col for col in metadata.columns
        if "recording" in col.lower()
        or col.lower() == "run"
    ]

    print("Recording-related columns:", recording_candidates)

    if recording_candidates:
        for col in recording_candidates:
            print(
                f"\nUnique values in {col}:",
                metadata[col].nunique()
            )

    section("6. FINAL PREFLIGHT RESULT")

    all_passed = all(checks.values())

    print("ALL CHECKS PASSED:", all_passed)

    print("\nImportant:")
    print("- No V8 files were modified.")
    print("- No model was trained.")
    print("- No locked participant labels were used for calibration.")
    print("- This was a read-only data inspection.")

    if not all_passed:
        print("\nSTOP: Resolve failed checks before V9-17.")
    else:
        print(
            "\nREADY FOR V9-17 PROTOCOL IMPLEMENTATION."
        )


if __name__ == "__main__":
    main()

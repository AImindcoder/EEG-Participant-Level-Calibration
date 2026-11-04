$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $Root "Results\V9_54_CROSS_DATASET_CLASSIFICATION"
$ReportDir = Join-Path $Root "Reports\V9_55_FINAL_EVIDENCE_AUDIT"
$ResultDir = Join-Path $Root "Results\V9_55_FINAL_EVIDENCE_AUDIT"

New-Item -ItemType Directory -Force -Path $ReportDir, $ResultDir | Out-Null

Write-Host "`n========== V9-55 FINAL EVIDENCE AUDIT =========="

# ------------------------------------------------
# 1. Load original V9-54 evidence
# ------------------------------------------------

$Json = Get-Content (Join-Path $Source "V9_54_SUMMARY.json") -Raw |
    ConvertFrom-Json

$Participants = Import-Csv (Join-Path $Source "V9_54_PARTICIPANT_METRICS.csv")
$Classes = Import-Csv (Join-Path $Source "V9_54_CLASS_METRICS.csv")
$Predictions = Import-Csv (Join-Path $Source "V9_54_PREDICTIONS.csv")
$Methods = Import-Csv (Join-Path $Source "V9_54_METHOD_COMPARISON.csv")
$Calibration = Import-Csv (Join-Path $Source "V9_54_CALIBRATION_SUMMARY.csv")
$Confusion = Import-Csv (Join-Path $Source "V9_54_CONFUSION_COUNTS.csv")

$Checks = [System.Collections.Generic.List[object]]::new()

function Add-Check {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Detail
    )

    $Checks.Add([PSCustomObject]@{
        Check = $Name
        Status = if ($Passed) { "PASS" } else { "FAIL" }
        Detail = $Detail
    })
}

# ------------------------------------------------
# 2. Protocol integrity checks
# ------------------------------------------------

$Overlap = @(
    $Json.training_participants |
        Where-Object { $_ -in $Json.test_participants }
)

Add-Check "No participant overlap" `
    ($Overlap.Count -eq 0) `
    "Overlap count: $($Overlap.Count)"

Add-Check "Calibration labels unused" `
    ($Json.calibration_labels_used -eq $false) `
    "Calibration labels used: $($Json.calibration_labels_used)"

Add-Check "Test labels unused for calibration" `
    ($Json.test_labels_used_for_calibration -eq $false) `
    "Test labels used for calibration: $($Json.test_labels_used_for_calibration)"

Add-Check "Training-only scaler" `
    ($Json.scaler_fit_on_training_only -eq $true) `
    "Scaler fitted on training data only"

Add-Check "Training-only global calibration scaling" `
    ($Json.global_calibration_location_scale_fit_on_training_only -eq $true) `
    "Global calibration location and scale fitted on training data only"

Add-Check "V8 protection recorded" `
    ($Json.v8_modified -eq $false) `
    "V8 modified: $($Json.v8_modified)"

Add-Check "V9 model protection recorded" `
    ($Json.existing_v9_models_modified -eq $false) `
    "Existing V9 models modified: $($Json.existing_v9_models_modified)"

Add-Check "Source data protection recorded" `
    ($Json.source_data_modified -eq $false) `
    "Source data modified: $($Json.source_data_modified)"

Add-Check "Expected training participants" `
    ($Json.training_participants.Count -eq 7) `
    "Training participants: $($Json.training_participants.Count)"

Add-Check "Expected test participants" `
    ($Json.test_participants.Count -eq 3) `
    "Test participants: $($Json.test_participants.Count)"

Add-Check "Six classification classes" `
    ($Json.classes.Count -eq 6) `
    "Class count: $($Json.classes.Count)"

Add-Check "Fifteen features" `
    ($Json.feature_count -eq 15) `
    "Feature count: $($Json.feature_count)"

Add-Check "Expected training segments" `
    ($Json.training_segments -eq 524) `
    "Training segments: $($Json.training_segments)"

Add-Check "Expected test segments" `
    ($Json.test_segments -eq 221) `
    "Test segments: $($Json.test_segments)"

# ------------------------------------------------
# 3. CSV structure and row-count checks
# ------------------------------------------------

Add-Check "Participant metrics row count" `
    ($Participants.Count -eq 9) `
    "Rows: $($Participants.Count); expected 9"

Add-Check "Class metrics row count" `
    ($Classes.Count -eq 54) `
    "Rows: $($Classes.Count); expected 54"

Add-Check "Prediction methods" `
    (@($Predictions.method | Sort-Object -Unique).Count -eq 3) `
    "Methods found: $(@($Predictions.method | Sort-Object -Unique).Count)"

$PredictionCounts = @(
    $Predictions |
        Group-Object method |
        ForEach-Object {
            [PSCustomObject]@{
                Method = $_.Name
                Rows = $_.Count
                Expected = $Json.test_segments
                Passed = ($_.Count -eq $Json.test_segments)
            }
        }
)

Add-Check "Prediction rows per method" `
    (@($PredictionCounts | Where-Object { -not $_.Passed }).Count -eq 0) `
    "Each method should contain $($Json.test_segments) held-out predictions"

$ObservedTestParticipants = @(
    $Predictions.participant | Sort-Object -Unique
)

$ParticipantMatch = (
    Compare-Object `
        ($Json.test_participants | Sort-Object) `
        ($ObservedTestParticipants | Sort-Object)
).Count -eq 0

Add-Check "Prediction participant identities" `
    $ParticipantMatch `
    "Observed test participants: $($ObservedTestParticipants -join ', ')"

# ------------------------------------------------
# 4. Recalculate class-wise mean F1
# ------------------------------------------------

$ClassMeans = foreach ($Group in ($Classes | Group-Object method, class)) {

    $Rows = @($Group.Group)
    $F1Values = @($Rows | ForEach-Object { [double]$_.f1 })
    $Mean = ($F1Values | Measure-Object -Average).Average

    [PSCustomObject]@{
        Method = $Rows[0].method
        Class = $Rows[0].class
        Participants = $Rows.Count
        Mean_F1 = [math]::Round($Mean, 6)
    }
}

$ClassMeans = @($ClassMeans | Sort-Object Class, Method)

$ClassMeans |
    Export-Csv (Join-Path $ResultDir "V9_55_CLASS_WISE_MEAN_F1.csv") `
        -NoTypeInformation -Encoding UTF8

# ------------------------------------------------
# 5. Recalculate participant-wise method differences
# ------------------------------------------------

$ParticipantComparison = foreach ($Participant in $Json.test_participants) {

    $Rows = @($Participants | Where-Object { $_.participant -eq $Participant })

    $NoRef = $Rows | Where-Object { $_.method -eq "no_reference" }
    $Mean = $Rows | Where-Object { $_.method -eq "mean_center" }
    $Geo = $Rows | Where-Object { $_.method -eq "geometric_median" }

    [PSCustomObject]@{
        Participant = $Participant
        No_Reference_F1 = [math]::Round([double]$NoRef.macro_f1, 6)
        Mean_Center_F1 = [math]::Round([double]$Mean.macro_f1, 6)
        Geometric_Median_F1 = [math]::Round([double]$Geo.macro_f1, 6)
        Geometric_vs_NoReference = [math]::Round(
            ([double]$Geo.macro_f1 - [double]$NoRef.macro_f1), 6
        )
    }
}

$ParticipantComparison |
    Export-Csv (Join-Path $ResultDir "V9_55_PARTICIPANT_COMPARISON.csv") `
        -NoTypeInformation -Encoding UTF8

# ------------------------------------------------
# 6. Verify JSON aggregate metrics against method CSV
# ------------------------------------------------

$MetricChecks = foreach ($Expected in $Json.results) {

    $Observed = $Methods |
        Where-Object { $_.method -eq $Expected.method }

    $F1Difference = [math]::Abs(
        [double]$Observed.participant_macro_f1_mean -
        [double]$Expected.participant_macro_f1_mean
    )

    $BalancedDifference = [math]::Abs(
        [double]$Observed.participant_balanced_accuracy_mean -
        [double]$Expected.participant_balanced_accuracy_mean
    )

    $APDifference = [math]::Abs(
        [double]$Observed.participant_macro_ap_mean -
        [double]$Expected.participant_macro_ap_mean
    )

    [PSCustomObject]@{
        Method = $Expected.method
        F1_Match = ($F1Difference -lt 0.0000001)
        Balanced_Accuracy_Match = ($BalancedDifference -lt 0.0000001)
        Macro_AP_Match = ($APDifference -lt 0.0000001)
    }
}

Add-Check "JSON and method CSV agreement" `
    (@($MetricChecks | Where-Object {
        -not ($_.F1_Match -and $_.Balanced_Accuracy_Match -and $_.Macro_AP_Match)
    }).Count -eq 0) `
    "Aggregate metrics compared for all three methods"

# ------------------------------------------------
# 7. Hash original V9-54 evidence files
# ------------------------------------------------

$Hashes = Get-ChildItem $Source -File |
    Sort-Object Name |
    ForEach-Object {
        $Hash = Get-FileHash $_.FullName -Algorithm SHA256

        [PSCustomObject]@{
            File = $_.Name
            Bytes = $_.Length
            SHA256 = $Hash.Hash
        }
    }

$Hashes |
    Export-Csv (Join-Path $ResultDir "V9_55_V9_54_SOURCE_HASH_SNAPSHOT.csv") `
        -NoTypeInformation -Encoding UTF8

Add-Check "Seven original evidence files present" `
    ($Hashes.Count -eq 7) `
    "Files hashed: $($Hashes.Count)"

# ------------------------------------------------
# 8. Create final scientific report
# ------------------------------------------------

$Best = $Json.results |
    Sort-Object participant_macro_f1_mean -Descending |
    Select-Object -First 1

$Report = @"
# V9-55: Final Research Documentation and Evidence Audit

Date: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## 1. Purpose

V9-55 consolidates and audits the evidence produced by V9-54.
It does not retrain models, modify EEG data, or change previous experiments.

## 2. Experimental protocol

- Dataset: Nabiyouni EEG Artifact Dataset
- Evaluation: participant-held-out evaluation within one dataset
- Training participants: $($Json.training_participants -join ", ")
- Test participants: $($Json.test_participants -join ", ")
- Calibration repetition: $($Json.calibration_repetition), unlabeled
- Evaluation repetition: $($Json.evaluation_repetition)
- Sampling rate: $($Json.sample_rate_hz) Hz
- Segment duration: $($Json.segment_duration_seconds) seconds
- Training segments: $($Json.training_segments)
- Test segments: $($Json.test_segments)
- Features: $($Json.feature_count)
- Classes: $($Json.classes -join ", ")
- Classifier: $($Json.classifier)
- Classifier C: $($Json.classifier_C)
- Class weights: $($Json.class_weight)
- Scaling: fitted on training data only

## 3. Aggregate participant-level results

| Method | Mean macro-F1 | SD | Balanced accuracy | Macro AP |
|---|---:|---:|---:|---:|
$(
    ($Json.results | ForEach-Object {
        "| $($_.method) | $([math]::Round($_.participant_macro_f1_mean,4)) | $([math]::Round($_.participant_macro_f1_std,4)) | $([math]::Round($_.participant_balanced_accuracy_mean,4)) | $([math]::Round($_.participant_macro_ap_mean,4)) |"
    }) -join "`n"
)

Highest aggregate participant macro-F1: **$($Best.method)** ($([math]::Round($Best.participant_macro_f1_mean,4))).

The difference between geometric-median centering and mean centering is small.
The results do not establish universal superiority of either method.

## 4. Participant-level macro-F1 comparison

| Participant | No reference | Mean center | Geometric median | Geometric minus no reference |
|---|---:|---:|---:|---:|
$(
    ($ParticipantComparison | ForEach-Object {
        "| $($_.Participant) | $($_.No_Reference_F1) | $($_.Mean_Center_F1) | $($_.Geometric_Median_F1) | $($_.Geometric_vs_NoReference) |"
    }) -join "`n"
)

## 5. Class-wise mean F1

These values are recalculated from the original class metrics CSV by averaging participant-level F1 within each method and class.

| Class | Method | Participants | Mean F1 |
|---|---|---:|---:|
$(
    ($ClassMeans | ForEach-Object {
        "| $($_.Class) | $($_.Method) | $($_.Participants) | $($_.Mean_F1) |"
    }) -join "`n"
)

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

- Participant overlap: $($Overlap.Count)
- Calibration labels used: $($Json.calibration_labels_used)
- Test labels used for calibration: $($Json.test_labels_used_for_calibration)
- Original EEG modification recorded: $($Json.source_data_modified)
- V8 modification recorded: $($Json.v8_modified)
- Existing V9 model modification recorded: $($Json.existing_v9_models_modified)
- Original V9-54 evidence files hashed: $($Hashes.Count)

The SHA-256 file is a snapshot of the V9-54 output files at audit time. It is not a historical before-and-after proof of their integrity.

## 9. Audit checks

| Check | Status | Detail |
|---|---|---|
$(
    ($Checks | ForEach-Object {
        "| $($_.Check) | $($_.Status) | $($_.Detail) |"
    }) -join "`n"
)

## 10. Conclusion

V9-54 is documented as an exploratory participant-held-out EEG condition-classification experiment on the Nabiyouni dataset.

V9-55 records its protocol, aggregate metrics, participant comparisons, class-level metrics, and evidence-file hashes. The conclusions remain limited to the evaluated dataset, participants, labels, and experimental design.

---
End of report.
"@

$ReportPath = Join-Path $ReportDir "V9_55_FINAL_RESEARCH_AUDIT.md"

Set-Content -Path $ReportPath -Value $Report -Encoding UTF8

# ------------------------------------------------
# 9. Save audit results
# ------------------------------------------------

$Checks |
    Export-Csv (Join-Path $ResultDir "V9_55_INTEGRITY_CHECKS.csv") `
        -NoTypeInformation -Encoding UTF8

$Audit = [PSCustomObject]@{
    experiment = "V9-55"
    source_experiment = "V9-54"
    audit_timestamp = (Get-Date).ToString("o")
    source_files_hashed = $Hashes.Count
    integrity_checks_passed = @($Checks | Where-Object Status -eq "PASS").Count
    integrity_checks_failed = @($Checks | Where-Object Status -eq "FAIL").Count
    all_checks_passed = (@($Checks | Where-Object Status -eq "FAIL").Count -eq 0)
    original_v9_54_files_modified_by_audit = $false
    models_retrained = $false
    eeg_signal_processing_performed = $false
    report = $ReportPath
}

$Audit |
    ConvertTo-Json -Depth 6 |
    Set-Content (Join-Path $ResultDir "V9_55_AUDIT_SUMMARY.json") -Encoding UTF8

# ------------------------------------------------
# 10. Display completion summary
# ------------------------------------------------

Write-Host "`n========== V9-55 AUDIT CHECKS =========="
$Checks | Format-Table -AutoSize

Write-Host "`n========== PARTICIPANT COMPARISON =========="
$ParticipantComparison | Format-Table -AutoSize

Write-Host "`n========== CLASS-WISE MEAN F1 =========="
$ClassMeans | Format-Table -AutoSize

Write-Host "`n========== FINAL AUDIT STATUS =========="
$Audit | Format-List

Write-Host "`nREPORT:"
Write-Host $ReportPath

Write-Host "`nRESULTS:"
Write-Host $ResultDir

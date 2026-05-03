
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_BASE="outputs/posthoc_n5seeds"
mkdir -p "$OUT_BASE" "$OUT_BASE/heatmaps"

MERGED_DEGRADATION="outputs/tables/main_plus_noise_delay_n5_degradation.csv"

SKILL_ROLLOUTS="$OUT_BASE/skill_rollouts.csv"
SKILL_SUMMARY="$OUT_BASE/skill_summary.csv"
REPERTOIRE_SUMMARY="$OUT_BASE/repertoire_summary.csv"
REPERTOIRE_TEX="$OUT_BASE/repertoire_summary.tex"
MOVEMENT_BY_RUN="$OUT_BASE/movement_modes_by_run.csv"
MOVEMENT_SUMMARY="$OUT_BASE/movement_modes_summary.csv"
MOVEMENT_TEX="$OUT_BASE/movement_modes_summary.tex"
BENEFIT_BY_RUN="$OUT_BASE/intended_benefit_by_run.csv"
BENEFIT_SUMMARY="$OUT_BASE/intended_benefit_summary.csv"
BENEFIT_TEX="$OUT_BASE/intended_benefit_summary.tex"
HEATMAP_DIR="$OUT_BASE/heatmaps"
RUN_LOG="$OUT_BASE/run.log"

AGG_CSV="$OUT_BASE/noise_delay_extension_n5_postprocess.csv"
echo "Aggregating n=5 noise-delay extension runs..." | tee -a "$RUN_LOG"
if ! python3 code/dissertation/aggregate_robustness_results.py \
    --root code/unified_skill_discovery/logs/local/half-cheetah \
    --output "$AGG_CSV" \
    --include-run-substring noise-delay-3 \
    --include-run-substring noise-med-delay-3 2>>"$RUN_LOG"; then
  echo "  Aggregator failed; falling back to existing n=3 aggregate." | tee -a "$RUN_LOG"
  AGG_CSV="outputs/aggregates/noise_delay_extension_postprocess.csv"
fi

echo "=== Step 1/5: Skill rollouts (REQUIRES diayn conda env + MuJoCo) ===" | tee -a "$RUN_LOG"
echo "Input:  $MERGED_DEGRADATION (expecting 75 rows + header = 76 lines)" | tee -a "$RUN_LOG"
echo "Output: $SKILL_ROLLOUTS" | tee -a "$RUN_LOG"
conda run -n diayn python code/dissertation/evaluate_skill_repertoire.py \
  --degradation-csv "$MERGED_DEGRADATION" \
  --output-skill-csv "$SKILL_ROLLOUTS" \
  --output-summary-csv "$SKILL_SUMMARY" \
  --max-path-length 1000 \
  --n-rollouts 5 \
  --seed 123 \
  --deterministic 1 2>&1 | tee -a "$RUN_LOG"

echo "=== Step 2/5: Repertoire summary ===" | tee -a "$RUN_LOG"
python3 code/dissertation/summarize_skill_repertoire.py \
  --input "$SKILL_SUMMARY" \
  --output-csv "$REPERTOIRE_SUMMARY" \
  --output-tex "$REPERTOIRE_TEX" 2>&1 | tee -a "$RUN_LOG"

echo "=== Step 3/5: Movement-mode summary ===" | tee -a "$RUN_LOG"
python3 code/dissertation/summarize_skill_movement_modes.py \
  --input "$SKILL_ROLLOUTS" \
  --output-run-csv "$MOVEMENT_BY_RUN" \
  --output-summary-csv "$MOVEMENT_SUMMARY" \
  --output-tex "$MOVEMENT_TEX" \
  --stationary-threshold 1.0 2>&1 | tee -a "$RUN_LOG"

echo "=== Step 4/5: Intended-benefit summary ===" | tee -a "$RUN_LOG"
python3 code/dissertation/summarize_intended_benefit_metrics.py \
  --degradation-csv "$MERGED_DEGRADATION" \
  --aggregate-csv "$AGG_CSV" \
  --skill-summary-csv "$SKILL_SUMMARY" \
  --movement-run-csv "$MOVEMENT_BY_RUN" \
  --output-run-csv "$BENEFIT_BY_RUN" \
  --output-summary-csv "$BENEFIT_SUMMARY" \
  --output-tex "$BENEFIT_TEX" 2>&1 | tee -a "$RUN_LOG"

echo "=== Step 5/5: Heatmaps ===" | tee -a "$RUN_LOG"
python3 code/dissertation/plot_skill_heatmaps.py \
  --input "$SKILL_ROLLOUTS" \
  --output-dir "$HEATMAP_DIR" 2>&1 | tee -a "$RUN_LOG"

echo | tee -a "$RUN_LOG"
echo "=== n=5 SEEDS post-hoc pipeline complete ===" | tee -a "$RUN_LOG"
echo "All outputs under: $OUT_BASE/" | tee -a "$RUN_LOG"
echo | tee -a "$RUN_LOG"
echo "Key artifacts:" | tee -a "$RUN_LOG"
echo "  Heatmaps:        $HEATMAP_DIR/" | tee -a "$RUN_LOG"
echo "  Repertoire tex:  $REPERTOIRE_TEX" | tee -a "$RUN_LOG"
echo "  Movement tex:    $MOVEMENT_TEX" | tee -a "$RUN_LOG"
echo "  Benefit tex:     $BENEFIT_TEX" | tee -a "$RUN_LOG"
echo "  Run log:         $RUN_LOG" | tee -a "$RUN_LOG"

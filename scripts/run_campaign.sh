#!/usr/bin/env bash
# run_campaign.sh — batch holdout evaluation: N policies x M bins x R same-seed repeats.
#
# One Isaac boot per policy: async_eval runs in BATCH mode (--dataset_roots +
# --num_runs) and loops every (bin, repeat) inside that boot. Each run stays a
# fully separate eval under <out>/<policy>/eval_cylroom_async_<bin>/run<NN>/
# (results CSV + timing + eval_summary + quality_gate via --full_report).
#
# Repeats use the SAME seed on purpose: the spread across repeats isolates pure
# floating-point non-determinism instead of mixing it with seed variation.
#
# Sequential, never concurrent — the GPU is exclusive. Run it inside a tmux
# session and give it a log file; it is resume-safe: a (bin, repeat) whose
# timing_async.json already exists is skipped, so re-launching after a crash
# continues where it stopped.
#
# Required environment (no defaults on purpose -- these name YOUR data):
#   POLICY_ROOT   directory holding <policy>/checkpoints/<ckpt>/pretrained_model
#   DATASETS_ROOT directory holding the recording bins (bin_c*_r*/ with binmap.json)
#   OUT_ROOT      where campaign results and logs go
#
# Tunables (defaults in parentheses):
#   POLICIES      space-separated policy dir names (2cam_wrist_front ... _back)
#   BINS          space-separated bin names (the 5 holdout bins)
#   CKPT          checkpoint step (065000)
#   NUM_RUNS      same-seed repeats per (policy, bin) (10)
#   SEED          the one seed (42)
#   NUM_EPISODES  episodes per run (27)
#   SETTLE_STEPS  per-spawn settle steps (130). This is ALSO the RTX-denoiser
#                 ghost fix: after a cube teleport the denoiser needs ~130 steps
#                 to clear the "two cubes" ghost from BOTH the video and the
#                 policy's first observation. A video-only frame skip was
#                 rejected -- it would falsify the record of what the policy saw.
#                 Measured not to affect success (delta 0.6 sigma).
#   SCENE         scene config JSON (the shipped 6-cam cylroom config)
#   TASK          pin the env task instead of auto-select from the policy's cameras
#   KIT_ARGS      verbose Kit/carb logging, e.g.
#                 '--/log/channels/omni.syntheticdata.plugin=verbose --/log/fileLogLevel=verbose'
#                 fileLogLevel=verbose is REQUIRED for [Verbose] lines to reach
#                 the on-disk log: the carb file sink defaults to Info and
#                 silently drops them (verified A/B: 3 vs 41607 verbose lines).
#                 The env var OMNI_KIT_ARGS does NOT work on this boot path.
#   DRY_RUN=1     print the per-policy commands instead of running them
set -euo pipefail
VERSION="2.0.0"

: "${POLICY_ROOT:?POLICY_ROOT must point at the directory holding <policy>/checkpoints/...}"
: "${DATASETS_ROOT:?DATASETS_ROOT must point at the directory holding the recording bins}"
: "${OUT_ROOT:?OUT_ROOT must point at the campaign output directory}"

NUM_RUNS="${NUM_RUNS:-10}"
SEED="${SEED:-42}"
POLICIES=(${POLICIES:-2cam_wrist_front 2cam_wrist_left 2cam_wrist_right 2cam_wrist_top 2cam_wrist_back})
BINS=(${BINS:-bin_c0_r0 bin_c0_r3 bin_c2_r2 bin_c4_r0 bin_c4_r3})
CKPT="${CKPT:-065000}"
NUM_EPISODES="${NUM_EPISODES:-27}"
SETTLE_STEPS="${SETTLE_STEPS:-130}"
DRY_RUN="${DRY_RUN:-0}"

# Default scene config = the shipped package resource, resolved through Python so
# it works for editable installs and wheels alike.
if [ -z "${SCENE:-}" ]; then
  SCENE="$(python -c "import so101_mvbench, pathlib; print(pathlib.Path(so101_mvbench.__file__).parent / 'tasks' / 'scene_configs' / 'lift_cube_6cam.json')")"
fi

TASK_EXTRA=()
[ -n "${TASK:-}" ] && TASK_EXTRA=(--task "$TASK")

LOGDIR="$OUT_ROOT/logs"
TS=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "$LOGDIR"

DS_ROOTS=""
for b in "${BINS[@]}"; do DS_ROOTS+="$DATASETS_ROOT/${b},"; done
DS_ROOTS="${DS_ROOTS%,}"

echo "===== CAMPAIGN start $(date '+%F %T') | ${#POLICIES[@]} policies x ${#BINS[@]} bins x ${NUM_RUNS} runs (seed ${SEED}) | settle=${SETTLE_STEPS} | task=${TASK:-auto} ====="
for pol in "${POLICIES[@]}"; do
  POLICY_PATH="$POLICY_ROOT/${pol}/checkpoints/${CKPT}/pretrained_model"
  OUT="$OUT_ROOT/$pol"
  LOG="$LOGDIR/campaign_${pol}_${TS}.log"
  CARB_LOG="$LOGDIR/carb_${pol}_${TS}.log"
  KIT_EXTRA=()
  [ -n "${KIT_ARGS:-}" ] && KIT_EXTRA=(--kit_args "$KIT_ARGS --/log/file=$CARB_LOG")

  # Resume guard: skip the whole boot when every (bin, repeat) already has timing.
  missing=0
  for b in "${BINS[@]}"; do for r in $(seq -f "%02g" 1 "$NUM_RUNS"); do
    [ -f "$OUT/eval_cylroom_async_${b}/run${r}/timing_async.json" ] || missing=$((missing+1))
  done; done
  if [ "$missing" -eq 0 ]; then
    echo "=== [$pol] all runs present — SKIP ==="
    continue
  fi

  CMD=(python -m so101_mvbench.evaluation.async_eval
    --mode async --num_envs 10 --num_episodes "$NUM_EPISODES"
    --dataset_roots "$DS_ROOTS" --num_runs "$NUM_RUNS" --seed "$SEED"
    --full_report --record_all_envs --video_interval 15 --verify_placement
    --settle_steps "$SETTLE_STEPS"
    --policy_path "$POLICY_PATH" --scene_config "$SCENE"
    --output_dir "$OUT" --headless)

  if [ "$DRY_RUN" = "1" ]; then
    echo "=== [$pol] DRY ($missing runs to do) ==="
    printf '  %q' "${CMD[@]}" ${KIT_EXTRA[@]+"${KIT_EXTRA[@]}"} ${TASK_EXTRA[@]+"${TASK_EXTRA[@]}"}; echo
    continue
  fi

  echo "=== [$pol] start $(date '+%H:%M:%S') ($missing runs to do) -> $LOG ==="
  "${CMD[@]}" ${KIT_EXTRA[@]+"${KIT_EXTRA[@]}"} ${TASK_EXTRA[@]+"${TASK_EXTRA[@]}"} > "$LOG" 2>&1 \
    || echo "=== [$pol] FAILED (exit $?) — skipping; a later re-launch resumes the gap ==="
  echo "=== [$pol] done $(date '+%H:%M:%S') | $(grep -c '>>> RESULT' "$LOG" || true) results ==="
done
echo "===== CAMPAIGN DONE $(date '+%F %T') ====="

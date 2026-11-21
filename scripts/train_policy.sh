#!/usr/bin/env bash
# Train (or resume) a pi0.5 policy on one robot's dataset.
#
#   scripts/train_policy.sh <robot> <task> --config <train config> [options] [-- extra train.py args]
#
#   scripts/train_policy.sh spot stove_eggplant --config pi05_spot_real_lora
#   scripts/train_policy.sh spot stove_eggplant --config pi05_spot_real_lora --steps 40000   # extend
#   scripts/train_policy.sh stretch4 cube --config pi05_stretch4_lora --compute-norm-stats --detach
#   scripts/train_policy.sh spot stove_eggplant --status
#
# The dataset is data/<robot>/<task>/ (scripts/dataset_env.sh), already converted to
# LeRobot. Checkpoints go to data/<robot>/<task>/checkpoints/<config>/<exp>/<step>,
# the log to data/<robot>/<task>/logs/train_<exp>.log, and every start or resume adds a
# line to data/<robot>/<task>/train_runs.txt. See TRAINING.md.
set -euo pipefail

usage() {
    cat <<'EOF'
usage: scripts/train_policy.sh <robot> <task> --config NAME [options] [-- extra train.py args]

  --config NAME         training config (required to train), e.g. pi05_spot_real_lora,
                        pi05_spot_sim_lora, pi05_stretch4_lora, pi05_stretch4_arm_lora
  --exp NAME            checkpoint folder name                     (default: <task>_lora)
  --steps N             total training steps; more than a finished run's extends it
                                                                   (default: 30000)
  --batch N             batch size                                 (default: 8)
  --save-every N        checkpoint interval; every one is kept     (default: 5000)
  --compute-norm-stats  compute the config's norm stats first (needed once per dataset,
                        and again after any re-conversion)
  --detach              run in the background; closing the terminal doesn't stop it
  --overwrite           start over: deletes this run's checkpoints (asks first)
  --yes                 don't ask (with --overwrite)
  --allow-busy-gpu      start even if another process holds GPU memory
  --dry-run             run every check and print the command, but don't train
  --status              show progress of the run (needs --exp if not <task>_lora)
  -h, --help            this help

Anything after `--` is passed to scripts/train.py as-is (e.g. -- --no-wandb-enabled).
EOF
}

die() { echo "error: $*" >&2; exit 1; }

[ $# -ge 1 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; } && { usage; exit 0; }
[ $# -ge 2 ] || { usage >&2; exit 1; }
ROBOT_ARG="$1"; TASK_ARG="$2"; shift 2

CONFIG=""; EXP=""; STEPS=30000; BATCH=8; SAVE_EVERY=5000
NORM_STATS=0; DETACH=0; OVERWRITE=0; YES=0; BUSY_GPU_OK=0; DRY_RUN=0; STATUS=0
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="${2:?--config needs a value}"; shift 2 ;;
        --exp) EXP="${2:?--exp needs a value}"; shift 2 ;;
        --steps) STEPS="${2:?--steps needs a value}"; shift 2 ;;
        --batch) BATCH="${2:?--batch needs a value}"; shift 2 ;;
        --save-every) SAVE_EVERY="${2:?--save-every needs a value}"; shift 2 ;;
        --compute-norm-stats) NORM_STATS=1; shift ;;
        --detach) DETACH=1; shift ;;
        --overwrite) OVERWRITE=1; shift ;;
        --yes) YES=1; shift ;;
        --allow-busy-gpu) BUSY_GPU_OK=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --status) STATUS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done
for v in STEPS BATCH SAVE_EVERY; do
    [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "--$(echo "$v" | tr 'A-Z_' 'a-z-') must be a positive integer, got '${!v}'"
done

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$OPENPI_ROOT"
# shellcheck source=dataset_env.sh
source scripts/dataset_env.sh "$ROBOT_ARG" "$TASK_ARG" >/dev/null
[ -d "$DATASET_PATH" ] || die "no dataset folder $DATASET_PATH (put the .h5 files in $DATASET_PATH/raw and convert them)"

EXP="${EXP:-${TASK}_lora}"
PY="${OPENPI_PYTHON:-$HOME/miniconda3/envs/openpi/bin/python}"
LOG_DIR="$DATASET_PATH/logs"
LOG="$LOG_DIR/train_$EXP.log"
RUNS="$DATASET_PATH/train_runs.txt"

# The same environment for every JAX command: the host CUDA / ROS libraries on
# LD_LIBRARY_PATH shadow the ones the JAX wheel ships, and ROS on PYTHONPATH shadows the
# env's packages. 95% preallocation is what fits batch 8 on the 32 GB RTX 5090; on-demand
# allocation fragments and runs out of memory on the first step.
jax_env() {
    env -u LD_LIBRARY_PATH PYTHONPATH= PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 "$@"
}

trainer_pids() {
    # The trainer for this dataset + exp, never this script or the awk itself. The repo is
    # reachable as /mnt/extra/... and /home/akash/extra/..., so match the path's tail.
    ps -eo pid=,args= | awk -v ckpt="/$ROBOT/$TASK/checkpoints" -v run="$EXP" '
        index($0, "scripts/train.py") && index($0, "--exp-name " run " ") && index($0, ckpt) && !index($0, "awk") { print $1 }'
}

step_dirs() {  # step numbers present in a checkpoint folder, ascending
    [ -d "$1" ] || return 0
    find "$1" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' -printf '%f\n' | sort -n
}

# ---------------------------------------------------------------- status
if [ "$STATUS" = 1 ]; then
    echo "run       $ROBOT/$TASK  exp=$EXP"
    pids="$(trainer_pids)"
    if [ -n "$pids" ]; then echo "trainer   running (pid $(echo $pids))"; else echo "trainer   not running"; fi
    for d in "$DATASET_PATH"/checkpoints/*/"$EXP"; do
        [ -d "$d" ] && echo "ckpts     $d: $(step_dirs "$d" | tr '\n' ' ')"
    done
    if [ -f "$LOG" ]; then
        last="$(grep -ao 'Step [0-9]*: loss=[0-9.e-]*' "$LOG" | tail -1 || true)"
        # openpi logs tqdm as "Progress on: 12.3kit/30.0kit rate:1.3it/s remaining:3:41:02 elapsed:..."
        prog="$(grep -a 'Progress on:' "$LOG" | tail -1 | sed -E 's/.*Progress on: *//; s/ *postfix:.*//' || true)"
        echo "latest    ${last:-no loss logged yet}"
        [ -n "$prog" ] && echo "progress  $prog"
        echo "log       $LOG"
    else
        echo "log       none yet ($LOG)"
    fi
    exit 0
fi

[ -n "$CONFIG" ] || die "--config is required (e.g. pi05_spot_real_lora; see --help)"
CKPT="$DATASET_PATH/checkpoints/$CONFIG/$EXP"

# ---------------------------------------------------------------- checks
[ -x "$PY" ] || die "no python at $PY (set OPENPI_PYTHON)"
[ -z "$(trainer_pids)" ] || die "$EXP on $ROBOT/$TASK is already training (pid $(echo $(trainer_pids))). See --status."

if [ "$BUSY_GPU_OK" = 0 ] && command -v nvidia-smi >/dev/null; then
    busy="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
            | awk -F', ' '{ split($3, m, " "); if (m[1] + 0 > 1024) print }' || true)"
    [ -z "$busy" ] || die "the GPU is in use (a policy server or another training?):
$busy
Stop it first, or pass --allow-busy-gpu."
fi

if [ "$NORM_STATS" = 1 ]; then
    echo "== computing norm stats for $CONFIG on $LEROBOT_REPO_ID"
    if [ "$DRY_RUN" = 1 ]; then
        echo "(dry run) would run: scripts/compute_norm_stats.py --config-name $CONFIG"
    else
        jax_env "$PY" scripts/compute_norm_stats.py --config-name "$CONFIG"
    fi
fi

echo "== checking $CONFIG against $ROBOT/$TASK"
PENDING=()
[ "$DRY_RUN" = 1 ] && [ "$NORM_STATS" = 1 ] && PENDING=(--norm-stats-pending)
jax_env "$PY" scripts/train_preflight.py --config "$CONFIG" --steps "$STEPS" --batch "$BATCH" "${PENDING[@]}" \
    || die "fix the problems above, then run again."

# ---------------------------------------------------------------- resume or start
LATEST="$(step_dirs "$CKPT" | tail -1)"
if [ "$OVERWRITE" = 1 ]; then
    MODE=--overwrite
    if [ -n "$LATEST" ]; then
        echo "--overwrite deletes $CKPT ($(step_dirs "$CKPT" | tr '\n' ' '))."
        if [ "$YES" = 0 ] && [ "$DRY_RUN" = 0 ]; then
            read -r -p "Delete them and start over? [y/N] " answer
            [[ "$answer" =~ ^[Yy]$ ]] || die "not started."
        fi
    fi
elif [ -n "$LATEST" ]; then
    MODE=--resume
    [ "$LATEST" -lt "$STEPS" ] || die "$EXP already reached step $LATEST of $STEPS. Pass a larger --steps to train further."
else
    MODE=--overwrite   # nothing to lose: --overwrite only clears an empty or step-less folder
fi

# openpi keeps a checkpoint only if its step is a multiple of the keep period, plus the
# newest one, which it deletes at the next save. Two consequences:
#  1. Train to STEPS+1 so the final save is step STEPS itself (train.py always saves
#     step num_train_steps-1), a multiple that is kept, never STEPS-1.
#  2. A non-multiple step already on disk (e.g. 29999 from an older run) would be
#     deleted at the first save of a resume. Hard-link it into <exp>_kept/ first: instant,
#     no extra disk, and it survives when openpi removes the original.
TRAIN_STEPS=$((STEPS + 1))
PRESERVE=()
if [ "$MODE" = --resume ]; then
    for s in $(step_dirs "$CKPT"); do
        [ $((s % SAVE_EVERY)) -eq 0 ] || PRESERVE+=("$s")
    done
fi

CMD=("$PY" scripts/train.py "$CONFIG"
     --exp-name "$EXP"
     --checkpoint-base-dir "$DATASET_PATH/checkpoints"
     --num-workers 0
     --batch-size "$BATCH"
     --num-train-steps "$TRAIN_STEPS"
     --save-interval "$SAVE_EVERY"
     --keep-period "$SAVE_EVERY"
     --log-interval 100
     "$MODE" "${EXTRA[@]}")

echo "== $([ "$MODE" = --resume ] && echo "resuming from step $LATEST" || echo "starting fresh") -> step $STEPS  (batch $BATCH, checkpoint every $SAVE_EVERY)"
echo "   checkpoints  $CKPT"
echo "   log          $LOG"
[ ${#PRESERVE[@]} -gt 0 ] && echo "   keeping      ${PRESERVE[*]} -> ${CKPT}_kept/ (openpi would delete it at the next save)"
if [ "$DRY_RUN" = 1 ]; then
    printf '(dry run) would run:'; printf ' %q' env -u LD_LIBRARY_PATH PYTHONPATH= PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 "${CMD[@]}"; echo
    exit 0
fi

for s in "${PRESERVE[@]}"; do
    if [ -e "${CKPT}_kept/$s" ]; then
        echo "   ${CKPT}_kept/$s already exists; leaving it."
    else
        mkdir -p "${CKPT}_kept"
        cp -al "$CKPT/$s" "${CKPT}_kept/$s"
    fi
done

mkdir -p "$LOG_DIR"
{
    echo
    echo "=== $(date -Iseconds)  train_policy.sh $ROBOT $TASK --config $CONFIG  $MODE  to step $STEPS  batch $BATCH  (last checkpoint: ${LATEST:-none})"
} >>"$LOG"
printf '%s  %s/%s  config=%s exp=%s %s from=%s to=%s batch=%s save_every=%s openpi=%s%s  extra=%s\n' \
    "$(date -Iseconds)" "$ROBOT" "$TASK" "$CONFIG" "$EXP" "${MODE#--}" "${LATEST:-none}" "$STEPS" "$BATCH" "$SAVE_EVERY" \
    "$(git rev-parse --short HEAD 2>/dev/null || echo ?)" "$([ -n "$(git status --porcelain 2>/dev/null)" ] && echo -dirty)" \
    "${EXTRA[*]:-none}" >>"$RUNS"

if [ "$DETACH" = 1 ]; then
    setsid nohup env -u LD_LIBRARY_PATH PYTHONPATH= PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
        "${CMD[@]}" >>"$LOG" 2>&1 </dev/null &
    echo "== training in the background (pid $!). Follow it with:"
    echo "   tail -f $LOG"
    echo "   scripts/train_policy.sh $ROBOT $TASK --status$([ "$EXP" != "${TASK}_lora" ] && echo " --exp $EXP")"
else
    jax_env "${CMD[@]}" 2>&1 | tee -a "$LOG"
fi

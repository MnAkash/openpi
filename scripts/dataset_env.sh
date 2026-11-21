#!/usr/bin/env bash
# Point every openpi script at one robot's dataset.
#
#   source scripts/dataset_env.sh <robot> <task>
#   source scripts/dataset_env.sh spot task_cube_no_force
#
# Everything openpi writes -- converted LeRobot data, normalisation stats,
# checkpoints, rollouts -- then lands under data/<robot>/<task>/, and the two big
# download caches are shared across every robot instead of being duplicated.
#
# Adding a robot is this plus a train config; nothing here is Spot-specific.

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "usage: source scripts/dataset_env.sh <robot> <task>" >&2
    echo "  e.g. source scripts/dataset_env.sh spot task_cube_no_force" >&2
    return 1 2>/dev/null || exit 1
fi

# Repo root, however this file was sourced.
_ENV_SELF="${BASH_SOURCE[0]:-$0}"
export OPENPI_ROOT="$(cd "$(dirname "$_ENV_SELF")/.." && pwd)"
unset _ENV_SELF

export ROBOT="$1"
export TASK="$2"

export DATA_ROOT="$OPENPI_ROOT/data"
export DATASET_PATH="$DATA_ROOT/$ROBOT/$TASK"
export H5_PATH="$DATASET_PATH/raw"

# Shared across robots: HF datasets metadata and the downloaded pi base weights.
export HF_HOME="$DATA_ROOT/_cache/hf_home"
export OPENPI_DATA_HOME="$DATA_ROOT/_cache/openpi"
# Per dataset: the converted LeRobot copy.
export HF_LEROBOT_HOME="$DATASET_PATH/lerobot"

# The LeRobot repo id, which is also the norm-stats subdirectory name. Keeping it
# equal to <robot>_<task> means two datasets can never collide in one cache.
export LEROBOT_REPO_ID="${ROBOT}_${TASK}"
export SPOT_REAL_REPO_ID="$LEROBOT_REPO_ID"   # read by the legacy spot_real configs
export SPOT_SIM_REPO_ID="$LEROBOT_REPO_ID"

if [ ! -d "$DATASET_PATH" ]; then
    echo "warning: $DATASET_PATH does not exist yet" >&2
fi

echo "robot=$ROBOT task=$TASK"
echo "  DATASET_PATH   $DATASET_PATH"
echo "  raw h5         $H5_PATH"
echo "  lerobot        $HF_LEROBOT_HOME/$LEROBOT_REPO_ID"
echo "  norm stats     $DATASET_PATH/norm_stats"
echo "  checkpoints    $DATASET_PATH/checkpoints"
echo "  repo id        $LEROBOT_REPO_ID"

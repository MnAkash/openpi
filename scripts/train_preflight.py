"""Check that a training config and the dataset the shell points at belong together.

Run by scripts/train_policy.sh before it starts the GPU. It needs the environment
`source scripts/dataset_env.sh <robot> <task>` sets, and fails (exit 1) with a message
saying what to fix when:

  * the config doesn't exist, or doesn't read its dataset from LEROBOT_REPO_ID;
  * the converted LeRobot dataset, or its norm stats, are missing;
  * the norm stats were computed for a different state or action width (stale after a
    re-conversion);
  * the dataset's robot, state layout, action width or image sizes differ from what the
    config trains and tells the rollout client (policy_metadata).

The state layout is compared by name using meta/conversion.json, which the converters
write. Real and simulated Spot data are both 8-D but mean different things
(arm_q + gripper vs eef_pos + eef_quat + gripper), so a dimension check alone would let
the wrong config train silently. Datasets converted before that file existed get a
dimension-only check and a warning.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


class Preflight:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _converter_hint(robot: str) -> str:
    return {
        "spot": "examples/spot_real/convert_spot_real_to_lerobot.py (real) or "
                "examples/spot_sim/convert_spot_sim_to_lerobot.py (simulator)",
        "stretch4": "examples/stretch4/convert_stretch4_to_lerobot.py",
    }.get(robot, f"the {robot} converter")


def _image_size(metadata_size, field: str):
    """(W, H) the config expects for a LeRobot image field, from policy_metadata["image_size"]."""
    if metadata_size is None:
        return None
    if isinstance(metadata_size, dict):
        size = metadata_size.get(field)
        return tuple(int(v) for v in size) if size is not None else None
    return tuple(int(v) for v in metadata_size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--norm-stats-pending", action="store_true",
                    help="norm stats are about to be computed (dry run): don't require them yet")
    args = ap.parse_args()

    check = Preflight()
    repo_id = os.environ.get("LEROBOT_REPO_ID")
    lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    robot = os.environ.get("ROBOT", "?")
    if not repo_id or not lerobot_home or not os.environ.get("DATASET_PATH"):
        print("error: run `source scripts/dataset_env.sh <robot> <task>` first.", file=sys.stderr)
        return 1

    import openpi.training.config as _config

    try:
        config = _config.get_config(args.config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        custom = sorted(n for n in _config._CONFIGS_DICT if "spot" in n or "stretch" in n)
        print(f"       configs for this machine's robots: {', '.join(custom)}", file=sys.stderr)
        return 1

    data_config = config.data.create(config.assets_dirs, config.model)
    metadata = dict(config.policy_metadata or {})
    action_dim = getattr(config.data, "action_dim", None)

    if data_config.repo_id != repo_id:
        check.error(f"config {args.config} trains on repo_id {data_config.repo_id!r}, but the shell points at "
                    f"{repo_id!r}. It doesn't read LEROBOT_REPO_ID, so it is not a config for this dataset.")

    dataset_dir = Path(lerobot_home) / repo_id
    info_path = dataset_dir / "meta" / "info.json"
    info: dict = {}
    if not info_path.is_file():
        check.error(f"no converted dataset at {dataset_dir}. Convert raw/ first with {_converter_hint(robot)}.")
    else:
        info = json.loads(info_path.read_text())

    conversion: dict | None = None
    conversion_path = dataset_dir / "meta" / "conversion.json"
    if conversion_path.is_file():
        conversion = json.loads(conversion_path.read_text())

    features = info.get("features", {})
    state_dim = (features.get("state", {}).get("shape") or [None])[0]
    data_action_dim = (features.get("action", {}).get("shape") or [None])[0]

    # Robot.
    want_robot = metadata.get("robot")
    have_robot = (conversion or {}).get("robot") or info.get("robot_type")
    if want_robot and have_robot and not str(have_robot).startswith(want_robot):
        check.error(f"the dataset is {have_robot} data but {args.config} is a {want_robot} config.")
    elif want_robot and robot != "?" and robot != want_robot:
        check.error(f"`dataset_env.sh {robot} ...` is a {robot} dataset but {args.config} is a {want_robot} config.")

    # Actions.
    if info and action_dim is not None and data_action_dim != action_dim:
        check.error(f"the dataset has {data_action_dim}-D actions but {args.config} trains {action_dim}-D. "
                    + ("For Stretch 4 use pi05_stretch4_lora for 13-D (arm + base) and pi05_stretch4_arm_lora "
                       "for 10-D (converted with --no-include-base-actions)." if want_robot == "stretch4" else ""))
    want_actions = metadata.get("action_components")
    if conversion and want_actions and conversion.get("action_components") != want_actions:
        check.error(f"the dataset's actions are {' + '.join(conversion.get('action_components', []))} but "
                    f"{args.config} expects {' + '.join(want_actions)}.")

    # State.
    want_state = metadata.get("state_components")
    if conversion and want_state:
        if conversion.get("state_components") != want_state:
            check.error(f"the dataset's state is {' + '.join(conversion['state_components'])} but {args.config} "
                        f"(and its rollout client) use {' + '.join(want_state)}. Re-convert with "
                        f"--state-components {' '.join(want_state)}, or use the config that matches.")
    elif want_state and info:
        check.warn(f"{conversion_path.name} is missing (converted before converters wrote it), so the state "
                   f"layout can only be checked by size ({state_dim}-D). Make sure it is "
                   f"{' + '.join(want_state)}; re-converting writes the record.")
        fixed = {"eef_pos": 3, "eef_quat": 4, "gripper": 1}
        if all(c in fixed for c in want_state) and state_dim != sum(fixed[c] for c in want_state):
            check.error(f"the dataset's state is {state_dim}-D but {' + '.join(want_state)} is "
                        f"{sum(fixed[c] for c in want_state)}-D.")

    # Images: the rollout client sends policy_metadata["image_size"], so training must match it.
    for field in ("base_image", "wrist_image"):
        want = _image_size(metadata.get("image_size"), field)
        shape = features.get(field, {}).get("shape")
        if want and shape:
            have = (int(shape[1]), int(shape[0]))
            if have != want:
                check.error(f"{field} is {have[0]}x{have[1]} in the dataset but {args.config} tells the rollout "
                            f"client {want[0]}x{want[1]}. Record at that size, or change the config's "
                            "policy_metadata['image_size'].")

    # Rate: actions are per-sample deltas, so the rollout must step at the recording rate.
    fps = info.get("fps")
    if want_robot == "spot" and fps and int(fps) != 10:
        check.warn(f"the data is {fps} Hz; the Spot rollout steps at --control-hz 10 by default. Roll out with "
                   f"--control-hz {fps} (each action is one {1000 / fps:.0f} ms step), or export at 10 Hz.")

    # Norm stats.
    norm_dir = Path(config.assets_dirs) / (data_config.asset_id or repo_id)
    if data_config.norm_stats is None and args.norm_stats_pending:
        pass  # computed by --compute-norm-stats before the real run
    elif data_config.norm_stats is None:
        check.error(f"no norm stats in {norm_dir}. Compute them with --compute-norm-stats, or:\n"
                    f"    env -u LD_LIBRARY_PATH PYTHONPATH= python scripts/compute_norm_stats.py "
                    f"--config-name {args.config}")
    else:
        for key, dim in (("state", state_dim), ("actions", data_action_dim)):
            stats = data_config.norm_stats.get(key)
            if stats is not None and dim is not None and len(stats.mean) != dim:
                check.error(f"the norm stats' {key} are {len(stats.mean)}-D but the dataset's are {dim}-D: they are "
                            "stale from an earlier conversion. Recompute them with --compute-norm-stats.")

    # Summary.
    frames = info.get("total_frames")
    episodes = info.get("total_episodes")
    prompts = (conversion or {}).get("prompts")
    if prompts is None and (dataset_dir / "meta" / "tasks.jsonl").is_file():
        prompts = [json.loads(line)["task"] for line in (dataset_dir / "meta" / "tasks.jsonl").read_text().splitlines() if line]
    print(f"  config        {args.config}  ({want_robot}; state {' + '.join(want_state or ['?'])}; "
          f"{action_dim}-D actions)")
    print(f"  dataset       {dataset_dir}")
    if info:
        sources = ", ".join((conversion or {}).get("data_source") or ["?"])
        print(f"  contents      {episodes} episodes, {frames} frames at {info.get('fps')} Hz, "
              f"{state_dim}-D state, {data_action_dim}-D actions, source: {sources}")
        print(f"  images        base {features['base_image']['shape'][1]}x{features['base_image']['shape'][0]}, "
              f"wrist {features['wrist_image']['shape'][1]}x{features['wrist_image']['shape'][0]}")
    if prompts:
        for prompt in prompts:
            print(f"  prompt        {prompt!r}")
    print(f"  norm stats    {norm_dir}"
          + ("  (not computed yet)" if data_config.norm_stats is None and args.norm_stats_pending else ""))
    if frames:
        print(f"  plan          {args.steps} steps x batch {args.batch} = {args.steps * args.batch / frames:.1f} "
              "passes over the data")
    for msg in check.warnings:
        print(f"warning: {msg}")
    for msg in check.errors:
        print(f"error: {msg}")
    return 1 if check.errors else 0


if __name__ == "__main__":
    sys.exit(main())

"""Convert simulated Spot HDF5 datasets to LeRobot format for openpi training.

Reads both layouts the 3DGS world simulator exports for Spot:

  * "Legacy" (10-column actions), and the older H2R renders (README_CUSTOM.md §3).
  * "Real robot (stretch4_teleop)", the simulator's default format. It has the same
    obs/eef_*, arm_q, gripper and images, plus base/odometry keys. Its table is
    ``joint_names`` and its actions have 13 columns: the same 10, then base_dx,
    base_dy and base_dyaw. Spot's arm policy and rollout use only the first 10. The
    base columns are checked to be zero (Spot's base stands still in these exports),
    and a file whose base moved is refused rather than silently dropping that motion.

Both layouts give identical state, images and first-10 action columns for the same
trajectory (checked 2026-11-20 on spot_ee_demo).

    arm_joint_names | joint_names       (7,)  bytes
    data/demo_0/
        actions                         (N, 10|13)       delta_pos(3) + rot6d(6) + gripper(1) [+ base(3)]
        object_ids                      (2,)   bytes     NOT converted
        obs/images_agentview            (N, 240, 320, 3) uint8   iso, world mounted
        obs/images_gripper              (N, 240, 320, 3) uint8   wrist, gripper mounted
        obs/eef_pos, obs/eef_quat       (N, 3), (N, 4)   BASE frame, xyzw
        obs/arm_q                       (N, 7)
        obs/gripper, obs/t              (N, 1)
        obs/object_poses                (N, M, 7)        NOT converted
        .attrs["task_description"]      str              the training prompt

`object_poses` and `object_ids` are deliberately dropped: they are simulator ground
truth that a real robot cannot observe, so training on them would produce a policy that
cannot be deployed. The same goes for any future privileged key -- add it to
DROPPED_KEYS rather than relying on it not being listed.

The gripper camera is ``images_gripper``, the key spot_teleop and the simulator now
write. Renders made before that rename call it ``images_ego``; convert those with
``--wrist-image-key images_ego``.

The prompt is read per demo from the HDF5 attribute, so a multi-task dataset converts
without a side-car task map. `--task` overrides it for every demo.
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

# Privileged simulator state: present in the HDF5, never converted.
DROPPED_KEYS = ("object_poses", "object_ids", "object_pose")

ARM_ACTION_DIM = 10
# Largest base step (m or rad per sample) still treated as "the base stood still" in a
# 13-column file. The simulator writes exact zeros (~1e-17) for a parked base.
MAX_BASE_STEP = 1e-4

# Proprioception that may be concatenated into observation/state, with the per-element
# names LeRobot records. Order here is the order they are concatenated in.
STATE_FEATURES: dict[str, list[str]] = {
    "eef_pos": ["eef_pos_x", "eef_pos_y", "eef_pos_z"],
    "eef_quat": ["eef_quat_x", "eef_quat_y", "eef_quat_z", "eef_quat_w"],   # xyzw
    "gripper": ["gripper"],
    "arm_q": [],        # filled from arm_joint_names
}


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


@dataclasses.dataclass(frozen=True)
class Args:
    input_path: Path
    """Directory of .h5 files, one file, or a glob."""
    repo_id: str | None = None
    """Defaults to $LEROBOT_REPO_ID, else spot_task_cube_no_force."""
    task: str | None = None
    """Override the per-demo prompt stored in the HDF5."""
    robot_type: str = "spot"
    fps: int | None = None
    """Inferred from obs/t when omitted."""
    overwrite: bool = True
    push_to_hub: bool = False
    base_image_key: str = "images_agentview"
    wrist_image_key: str = "images_gripper"
    state_components: tuple[str, ...] = ("eef_pos", "eef_quat", "gripper")
    """Concatenated, in this order, into observation/state.

    The default matches the ACTION space: actions are end-effector deltas plus an
    absolute gripper, so end-effector pose + gripper is the state a policy in that
    space actually needs. `arm_q` is available and can be appended, but it is
    redundant through forward kinematics and this dataset is small.
    """
    channel_order: Literal["auto", "rgb", "bgr"] = "auto"
    """Image channel order of the input. "auto" reads each file's image_channel_order
    attribute (new recordings are RGB) and refuses files without one."""
    dataset_config: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)


def _collect_h5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*.h5") if p.is_file())
    if any(c in str(input_path) for c in "*?[]"):
        return sorted(p for p in input_path.parent.glob(input_path.name) if p.is_file())
    raise FileNotFoundError(f"Could not find input path: {input_path}")


def _iter_demo_groups(h5_file: h5py.File):
    if "data" not in h5_file:
        raise KeyError("Expected a top-level 'data' group in the HDF5 file.")
    for name in sorted(h5_file["data"].keys()):
        yield name, h5_file["data"][name]


def _decode(values) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _matrix(values, *, dtype=np.float32) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}")


def _parse_image(image) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(255 * array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    if array.ndim != 3:
        raise ValueError(f"Expected an image with 3 dimensions, got shape {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    return array


def _channel_order(h5_file: h5py.File, path: Path, override: str) -> str:
    """The file's image channel order: --channel-order, else its ``image_channel_order`` attr.

    New recordings (spot_teleop, stretch4_teleop, the 3DGS simulator) store RGB and say
    so. Older files say nothing: real-robot recordings from then are OpenCV BGR and
    simulator renders RGB, so the order cannot be guessed and must be passed.
    """
    if override != "auto":
        return override
    order = h5_file.attrs.get("image_channel_order")
    if order is None:
        raise ValueError(
            f"{path.name} does not say its image channel order (it predates RGB recording). "
            "Pass --channel-order bgr for an older real-robot recording, or rgb for an older "
            "simulator render.")
    order = order.decode("utf-8") if isinstance(order, bytes) else str(order)
    if order not in ("rgb", "bgr"):
        raise ValueError(f"{path.name}: unknown image_channel_order {order!r}")
    return order


def _to_rgb(image: np.ndarray, order: str) -> np.ndarray:
    """LeRobot / pi0.5 take RGB; flip a BGR frame."""
    return np.ascontiguousarray(image[..., ::-1]) if order == "bgr" else image


def _infer_fps(timestamps) -> int | None:
    deltas = np.diff(np.asarray(timestamps, dtype=np.float64).reshape(-1))
    deltas = deltas[deltas > 0]
    return round(1.0 / np.median(deltas)) if deltas.size else None


def _state_names(components: tuple[str, ...], arm_joint_names: list[str]) -> list[str]:
    names: list[str] = []
    for component in components:
        if component not in STATE_FEATURES:
            raise ValueError(f"Unsupported state component {component!r}; "
                             f"choose from {sorted(STATE_FEATURES)}")
        names.extend(arm_joint_names if component == "arm_q" else STATE_FEATURES[component])
    return names


def _build_state(obs: h5py.Group, components: tuple[str, ...]) -> np.ndarray:
    parts = []
    for component in components:
        if component not in obs:
            raise KeyError(f"Missing observation key 'obs/{component}' in the HDF5 file.")
        parts.append(_matrix(obs[component][:]))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def _arm_actions(demo: h5py.Group, where: str) -> np.ndarray:
    """The 10 arm action columns: as stored (10), or the first 10 of a 13-column file."""
    actions = _matrix(demo["actions"][:])
    if actions.shape[-1] == ARM_ACTION_DIM:
        return actions
    if actions.shape[-1] == ARM_ACTION_DIM + 3:
        base = np.abs(actions[:, ARM_ACTION_DIM:]).max() if len(actions) else 0.0
        if base > MAX_BASE_STEP:
            raise ValueError(
                f"{where}: the base moves (up to {base:.4f} m or rad per step), but Spot's arm policy "
                "and rollout have no base actions. Export with the base parked.")
        return actions[:, :ARM_ACTION_DIM]
    raise ValueError(f"{where}: expected 10 or 13 action columns, got {actions.shape[-1]}.")


def _check_robot(h5_file: h5py.File, path: Path) -> None:
    robot = h5_file.attrs.get("robot_type")
    robot = robot.decode("utf-8") if isinstance(robot, bytes) else robot
    if robot is not None and robot != "spot":
        raise ValueError(f"{path.name} was recorded with robot_type={robot!r}, not Spot.")


def _resolve_prompt(demo: h5py.Group, override: str | None, where: str) -> str:
    if override is not None:
        return override
    prompt = demo.attrs.get("task_description")
    if prompt is None:
        raise KeyError(
            f"{where} has no 'task_description' attribute and --task was not given. "
            "Re-render with --task-description, or pass --task here.")
    return prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)


def _write_conversion_record(output_path: Path, record: dict) -> None:
    """meta/conversion.json: what this LeRobot copy holds, checked by scripts/train_preflight.py."""
    import datetime
    import json

    record = {**record, "converted_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    (output_path / "meta").mkdir(parents=True, exist_ok=True)
    (output_path / "meta" / "conversion.json").write_text(json.dumps(record, indent=2) + "\n")


def main(args: Args):
    import os

    repo_id = args.repo_id or os.environ.get("LEROBOT_REPO_ID") or "spot_task_cube_no_force"
    if not args.state_components:
        raise ValueError("state_components must contain at least one observation key.")

    h5_files = _collect_h5_files(args.input_path)
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {args.input_path}")

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_path}")
        shutil.rmtree(output_path)

    arm_joint_names: list[str] | None = None
    inferred_fps: int | None = None
    episodes = frames = 0
    prompts: set[str] = set()
    orders: dict[str, str] = {}
    sources: dict[str, int] = {}

    for path in h5_files:
        with h5py.File(path, "r") as f:
            _check_robot(f, path)
            orders[path.name] = _channel_order(f, path, args.channel_order)
            for table in ("arm_joint_names", "joint_names"):
                if arm_joint_names is None and table in f:
                    arm_joint_names = _decode(np.asarray(f[table]))
            source = _decode([f.attrs.get("data_source", "unknown")])[0]
            sources[source] = sources.get(source, 0) + 1
            for name, demo in _iter_demo_groups(f):
                episodes += 1
                frames += _arm_actions(demo, f"{path.name}:{name}").shape[0]
                prompts.add(_resolve_prompt(demo, args.task, f"{path.name}:{name}"))
                if inferred_fps is None and "t" in demo["obs"]:
                    inferred_fps = _infer_fps(np.asarray(demo["obs"]["t"]))

    if arm_joint_names is None:
        raise KeyError("Could not find 'arm_joint_names' or 'joint_names' in the HDF5 input.")
    fps = args.fps if args.fps is not None else inferred_fps
    if fps is None:
        raise ValueError("Could not infer FPS from obs/t. Pass --fps explicitly.")

    with h5py.File(h5_files[0], "r") as f:
        _, sample = next(_iter_demo_groups(f))
        obs = sample["obs"]
        for key in (args.base_image_key, args.wrist_image_key):
            if key not in obs:
                raise KeyError(f"Missing image stream 'obs/{key}'. Present: {sorted(obs)}")
        state_dim = _build_state(obs, args.state_components).shape[-1]
        action_dim = ARM_ACTION_DIM
        base_shape = _parse_image(obs[args.base_image_key][0]).shape
        wrist_shape = _parse_image(obs[args.wrist_image_key][0]).shape

    mode = "video" if args.dataset_config.use_videos else "image"
    features = {
        "base_image": {"dtype": mode, "shape": base_shape, "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": mode, "shape": wrist_shape, "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (state_dim,),
                  "names": _state_names(args.state_components, arm_joint_names)},
        "action": {"dtype": "float32", "shape": (action_dim,),
                   "names": [f"action_{i}" for i in range(action_dim)]},
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=args.robot_type,
        fps=fps,
        features=features,
        use_videos=args.dataset_config.use_videos,
        tolerance_s=args.dataset_config.tolerance_s,
        image_writer_processes=args.dataset_config.image_writer_processes,
        image_writer_threads=args.dataset_config.image_writer_threads,
        video_backend=args.dataset_config.video_backend,
    )

    for path in h5_files:
        with h5py.File(path, "r") as f:
            for name, demo in tqdm.tqdm(list(_iter_demo_groups(f)), desc=f"Converting {path.name}"):
                prompt = _resolve_prompt(demo, args.task, f"{path.name}:{name}")
                obs = demo["obs"]
                actions = _arm_actions(demo, f"{path.name}:{name}")
                state = _build_state(obs, args.state_components)
                base_images = obs[args.base_image_key][:]
                wrist_images = obs[args.wrist_image_key][:]

                n = actions.shape[0]
                for label, array in (("state", state), (args.base_image_key, base_images),
                                     (args.wrist_image_key, wrist_images)):
                    if array.shape[0] != n:
                        raise ValueError(
                            f"{path.name}:{name}: {label} has {array.shape[0]} rows, "
                            f"actions has {n}. The dataset is not frame aligned.")

                order = orders[path.name]
                for i in range(n):
                    dataset.add_frame({
                        "base_image": _to_rgb(_parse_image(base_images[i]), order),
                        "wrist_image": _to_rgb(_parse_image(wrist_images[i]), order),
                        "state": state[i],
                        "action": actions[i],
                        "task": prompt,
                    })
                dataset.save_episode()

    _write_conversion_record(output_path, {
        "converter": "examples/spot_sim/convert_spot_sim_to_lerobot.py",
        "robot": "spot",
        "data_source": sorted(sources),
        "state_components": list(args.state_components),
        "action_components": ["arm"],
        "action_dim": ARM_ACTION_DIM,
        "base_image_key": args.base_image_key,
        "wrist_image_key": args.wrist_image_key,
        "image_size": {"base_image": [base_shape[1], base_shape[0]], "wrist_image": [wrist_shape[1], wrist_shape[0]]},
        "prompts": sorted(prompts),
        "inputs": [str(p) for p in h5_files],
    })

    if args.push_to_hub:
        dataset.push_to_hub()

    print(f"\nSaved LeRobot dataset to {output_path}")
    print(f"  repo id           {repo_id}")
    print(f"  episodes          {episodes}")
    print(f"  frames            {frames}")
    print(f"  fps               {fps}")
    print(f"  state ({state_dim}d)      {' + '.join(args.state_components)}")
    print(f"  action ({action_dim}d)     dpos + rot6d + gripper, as stored (13-column files: first 10, base parked)")
    print(f"  cameras           base={args.base_image_key}  wrist={args.wrist_image_key}")
    print(f"  channel order     {', '.join(f'{k}: {v}' for k, v in sorted(orders.items()))}  (stored RGB)")
    print(f"  prompt(s)         {sorted(prompts)}")
    print(f"  dropped           {', '.join(DROPPED_KEYS)}  (privileged simulator state)")


if __name__ == "__main__":
    main(tyro.cli(Args))

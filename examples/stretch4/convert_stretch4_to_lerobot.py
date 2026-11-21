"""Convert Stretch 4 HDF5 datasets (real or simulated) to LeRobot format for openpi training.

Source layout, written by stretch4_teleop/create_dataset.py on the real robot and by the
3DGS world simulator (same layout, field for field):

    .attrs                              robot_type="stretch4", data_source="real"|"simulator",
                                        image_channel_order="rgb", ...
    joint_names                         (5,)  bytes      lift, arm, wrist_yaw, wrist_pitch, wrist_roll
    data/demo_<n>/
        .attrs                          robot_name, robot_serial, end_of_arm_tool, recorded_at, ...
                                        task_description (simulator builds, when given)
        actions                         (N, 13)  dpos(3) + rot6d(6) + gripper(1) + base_dx, base_dy, base_dyaw
        obs/images_gripper              (N, 240, 320, 3) uint8  RGB   wrist (gripper) camera
        obs/images_head_center          (N, 320, 240, 3) uint8  RGB   head camera, portrait
        obs/eef_pos, obs/eef_quat       (N, 3), (N, 4)   base frame, xyzw
        obs/arm_q, obs/arm_dq           (N, 5)
        obs/gripper                     (N, 1)   0 closed .. 1 open
        obs/body_vel, base_in_odom, ee_pose_odom, t, ...
        obs/object_poses, object_ids    simulator only, NOT converted

Images are stored RGB (the recorders convert their OpenCV frames when they save) and the
file root says ``image_channel_order = "rgb"``. Files without that attribute predate RGB
recording: pass ``--channel-order bgr`` (real robot) or ``rgb`` (simulator) for them.

``images_head_center`` -> ``base_image`` and ``images_gripper`` -> ``wrist_image``, the same
LeRobot fields the Spot converters write, so the Spot input transforms apply unchanged.

Every episode's origin (file, demo, real or simulated, robot name and serial, tool, record
time) is written to ``meta/episode_sources.jsonl`` next to the LeRobot metadata, so a mixed
sim + real or multi-robot dataset stays traceable.
"""

import dataclasses
import json
import os
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

ROBOT_TYPE = "stretch4"

# Privileged simulator state: present in simulated HDF5s, never converted.
DROPPED_KEYS = ("object_poses", "object_ids", "object_pose")

# Proprioception that may be concatenated into observation/state, with the per-element
# names LeRobot records. Order here is the order they are concatenated in.
STATE_FEATURES: dict[str, list[str]] = {
    "eef_pos": ["eef_pos_x", "eef_pos_y", "eef_pos_z"],
    "eef_quat": ["eef_quat_x", "eef_quat_y", "eef_quat_z", "eef_quat_w"],  # xyzw
    "gripper": ["gripper"],
    "arm_q": [],  # filled from joint_names
    "arm_dq": [],  # filled from joint_names
    "body_vel": ["body_vx", "body_vy", "body_vz", "body_wx", "body_wy", "body_wz"],
    "base_in_odom": ["base_x", "base_y", "base_z", "base_qx", "base_qy", "base_qz", "base_qw"],
    "ee_pose_odom": ["ee_odom_x", "ee_odom_y", "ee_odom_z", "ee_odom_qx", "ee_odom_qy", "ee_odom_qz", "ee_odom_qw"],
}

ACTION_NAMES = [
    "dpos_x", "dpos_y", "dpos_z",
    "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
    "gripper",
    "base_dx", "base_dy", "base_dyaw",
]

# Demo-group attrs copied into meta/episode_sources.jsonl, when present.
SOURCE_ATTRS = (
    "robot_type", "robot_name", "robot_serial", "robot_model", "end_of_arm_tool",
    "recorded_at", "record_fps", "recorder", "recorder_commit", "recording_host",
    "source_file", "scene_id", "trajectory_id",
)


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
    """Directory of .h5 files, one file, or a glob. Real and simulated files can be mixed."""
    repo_id: str | None = None
    """Defaults to $LEROBOT_REPO_ID, else stretch4_dataset."""
    task: str | None = None
    """Prompt for every demo. Overrides the per-demo task_description attribute."""
    task_map_path: Path | None = None
    """JSON mapping "file.h5:demo_0", "file:demo_0" or "demo_0" to a prompt."""
    robot_type: str = ROBOT_TYPE
    fps: int | None = None
    """Inferred from obs/t when omitted."""
    overwrite: bool = True
    push_to_hub: bool = False
    base_image_key: str = "images_head_center"
    wrist_image_key: str = "images_gripper"
    state_components: tuple[str, ...] = ("eef_pos", "eef_quat", "gripper")
    """Concatenated, in this order, into observation/state.

    The default matches the arm part of the ACTION space (end-effector deltas plus an
    absolute gripper). Add `body_vel` for base motion, or `arm_q` for joint positions.
    `base_in_odom` / `ee_pose_odom` are absolute odometry poses: they depend on where
    odometry started, so a policy trained on them does not transfer between sessions.
    """
    include_base_actions: bool = True
    """Keep the three base-motion action columns (13-D). False keeps only the 10 arm columns,
    for tasks where the base never moves."""
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


def _demo_sort_key(name: str):
    tail = name[5:] if name.startswith("demo_") else name
    return (0, int(tail), name) if tail.isdigit() else (1, 0, name)


def _iter_demo_groups(h5_file: h5py.File):
    if "data" not in h5_file:
        raise KeyError("Expected a top-level 'data' group in the HDF5 file.")
    for name in sorted(h5_file["data"].keys(), key=_demo_sort_key):
        yield name, h5_file["data"][name]


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _matrix(values, *, dtype=np.float32) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}")


def _parse_image(image, order: str) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(255 * array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an (H, W, 3) image, got shape {array.shape}")
    return _to_rgb(array, order)


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


def _state_names(components: tuple[str, ...], joint_names: list[str]) -> list[str]:
    names: list[str] = []
    for component in components:
        if component not in STATE_FEATURES:
            raise ValueError(f"Unsupported state component {component!r}; choose from {sorted(STATE_FEATURES)}")
        if component == "arm_q":
            names.extend(joint_names)
        elif component == "arm_dq":
            names.extend(f"{name}_dq" for name in joint_names)
        else:
            names.extend(STATE_FEATURES[component])
    return names


def _build_state(obs: h5py.Group, components: tuple[str, ...], where: str) -> np.ndarray:
    parts = []
    for component in components:
        if component not in obs:
            raise KeyError(f"{where}: missing observation key 'obs/{component}'.")
        parts.append(_matrix(obs[component][:]))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def _build_actions(demo: h5py.Group, include_base: bool, where: str) -> np.ndarray:
    actions = _matrix(demo["actions"][:])
    if actions.shape[-1] != len(ACTION_NAMES):
        raise ValueError(f"{where}: expected {len(ACTION_NAMES)}-column Stretch 4 actions, got {actions.shape[-1]}.")
    return actions if include_base else actions[:, :10]


def _load_task_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open() as f:
        return {str(k): str(v) for k, v in json.load(f).items()}


def _resolve_prompt(demo: h5py.Group, args: Args, task_map: dict[str, str], path: Path, name: str) -> str:
    if args.task is not None:
        return args.task
    for key in (f"{path.name}:{name}", f"{path.stem}:{name}", name):
        if key in task_map:
            return task_map[key]
    prompt = demo.attrs.get("task_description")
    if prompt is not None:
        return str(_decode(prompt))
    raise KeyError(
        f"{path.name}:{name} has no prompt: pass --task, a --task-map-path entry, or build the "
        "dataset with a task_description.")


def _check_robot(h5_file: h5py.File, path: Path) -> str:
    """The file's data_source, after checking it really is Stretch 4 data."""
    robot = h5_file.attrs.get("robot_type")
    if robot is not None and str(_decode(robot)) != ROBOT_TYPE:
        raise ValueError(f"{path.name} is {_decode(robot)!r} data, not {ROBOT_TYPE!r}.")
    return str(_decode(h5_file.attrs.get("data_source", "unknown")))


def main(args: Args):
    repo_id = args.repo_id or os.environ.get("LEROBOT_REPO_ID") or "stretch4_dataset"
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

    task_map = _load_task_map(args.task_map_path)

    # Pass 1: validate every file and demo before writing anything.
    joint_names: list[str] | None = None
    inferred_fps: int | None = None
    shapes: dict[str, tuple] = {}
    episodes = frames = 0
    prompts: set[str] = set()
    sources: dict[str, int] = {}
    orders: dict[str, str] = {}
    for path in h5_files:
        with h5py.File(path, "r") as f:
            source = _check_robot(f, path)
            orders[path.name] = _channel_order(f, path, args.channel_order)
            if joint_names is None and "joint_names" in f:
                joint_names = [str(_decode(v)) for v in np.asarray(f["joint_names"])]
            for name, demo in _iter_demo_groups(f):
                where = f"{path.name}:{name}"
                obs = demo["obs"]
                for key in (args.base_image_key, args.wrist_image_key):
                    if key not in obs:
                        raise KeyError(f"{where}: missing image stream 'obs/{key}'. Present: {sorted(obs)}")
                    shape = tuple(obs[key].shape[1:])
                    if shapes.setdefault(key, shape) != shape:
                        raise ValueError(f"{where}: obs/{key} is {shape}, earlier demos are {shapes[key]}.")
                n = _build_actions(demo, args.include_base_actions, where).shape[0]
                state = _build_state(obs, args.state_components, where)
                for label, rows in (("state", state.shape[0]), (args.base_image_key, obs[args.base_image_key].shape[0]),
                                    (args.wrist_image_key, obs[args.wrist_image_key].shape[0])):
                    if rows != n:
                        raise ValueError(f"{where}: {label} has {rows} rows, actions has {n}. Not frame aligned.")
                prompts.add(_resolve_prompt(demo, args, task_map, path, name))
                if inferred_fps is None and "t" in obs:
                    inferred_fps = _infer_fps(np.asarray(obs["t"]))
                episodes += 1
                frames += n
                sources[source] = sources.get(source, 0) + 1

    if any(c in ("arm_q", "arm_dq") for c in args.state_components) and joint_names is None:
        raise KeyError("arm_q/arm_dq state needs the top-level 'joint_names' table, which is missing.")
    fps = args.fps if args.fps is not None else inferred_fps
    if fps is None:
        raise ValueError("Could not infer FPS from obs/t. Pass --fps explicitly.")

    state_names = _state_names(args.state_components, joint_names or [])
    action_names = ACTION_NAMES if args.include_base_actions else ACTION_NAMES[:10]
    mode = "video" if args.dataset_config.use_videos else "image"
    features = {
        "base_image": {"dtype": mode, "shape": shapes[args.base_image_key], "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": mode, "shape": shapes[args.wrist_image_key], "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
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

    # Pass 2: write.
    episode_sources = []
    for path in h5_files:
        with h5py.File(path, "r") as f:
            source = _check_robot(f, path)
            for name, demo in tqdm.tqdm(list(_iter_demo_groups(f)), desc=f"Converting {path.name}"):
                where = f"{path.name}:{name}"
                prompt = _resolve_prompt(demo, args, task_map, path, name)
                obs = demo["obs"]
                actions = _build_actions(demo, args.include_base_actions, where)
                state = _build_state(obs, args.state_components, where)
                base_images = obs[args.base_image_key][:]
                wrist_images = obs[args.wrist_image_key][:]
                for i in range(actions.shape[0]):
                    dataset.add_frame({
                        "base_image": _parse_image(base_images[i], orders[path.name]),
                        "wrist_image": _parse_image(wrist_images[i], orders[path.name]),
                        "state": state[i],
                        "action": actions[i],
                        "task": prompt,
                    })
                dataset.save_episode()
                record = {"episode_index": len(episode_sources), "file": path.name, "demo": name,
                          "data_source": source, "image_channel_order": orders[path.name],
                          "frames": int(actions.shape[0]), "task": prompt}
                record.update({k: _decode(demo.attrs[k]) for k in SOURCE_ATTRS if k in demo.attrs})
                episode_sources.append(record)

    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "episode_sources.jsonl").open("w") as f:
        for record in episode_sources:
            f.write(json.dumps(record) + "\n")

    record = {
        "converter": "examples/stretch4/convert_stretch4_to_lerobot.py",
        "robot": ROBOT_TYPE,
        "data_source": sorted(sources),
        "state_components": list(args.state_components),
        "action_components": ["arm", "base"] if args.include_base_actions else ["arm"],
        "action_dim": len(action_names),
        "base_image_key": args.base_image_key,
        "wrist_image_key": args.wrist_image_key,
        "image_size": {k: [shapes[key][1], shapes[key][0]]
                       for k, key in (("base_image", args.base_image_key), ("wrist_image", args.wrist_image_key))},
        "prompts": sorted(prompts),
        "inputs": [str(p) for p in h5_files],
        "converted_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (meta_dir / "conversion.json").write_text(json.dumps(record, indent=2) + "\n")

    if args.push_to_hub:
        dataset.push_to_hub()

    robots = sorted({str(r.get("robot_name", "unknown")) for r in episode_sources})
    print(f"\nSaved LeRobot dataset to {output_path}")
    print(f"  repo id           {repo_id}")
    print(f"  episodes          {episodes}  ({', '.join(f'{k}: {v}' for k, v in sorted(sources.items()))})")
    print(f"  robots            {', '.join(robots)}")
    print(f"  frames            {frames}")
    print(f"  fps               {fps}")
    print(f"  state ({len(state_names)}d)      {' + '.join(args.state_components)}")
    print(f"  action ({len(action_names)}d)    {'arm + base' if args.include_base_actions else 'arm only'}, used as-is")
    print(f"  cameras           base={args.base_image_key}  wrist={args.wrist_image_key}")
    print(f"  channel order     {', '.join(f'{k}: {v}' for k, v in sorted(orders.items()))}  (stored RGB)")
    print(f"  prompt(s)         {sorted(prompts)}")
    print(f"  dropped           {', '.join(DROPPED_KEYS)}  (privileged simulator state)")
    print(f"  sources           {meta_dir / 'episode_sources.jsonl'}")
    print(f"  record            {meta_dir / 'conversion.json'}  (checked by scripts/train_policy.sh)")


if __name__ == "__main__":
    main(tyro.cli(Args))

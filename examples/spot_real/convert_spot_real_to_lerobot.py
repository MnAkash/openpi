"""Convert the Spot real HDF5 dataset layout to LeRobot format for openpi training.

Cameras are read by name, as spot_teleop records them: ``obs/images_agentview`` (the
external RealSense) becomes ``base_image`` and ``obs/images_gripper`` (Spot's hand
camera) becomes ``wrist_image``. Datasets recorded before that rename used
``images_1`` / ``images_0``; convert those with
``--base-image-key images_1 --wrist-image-key images_0``.
"""

import dataclasses
import json
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

STATE_COMPONENTS = {
    "arm_q",
    "arm_dq",
    "body_vel",
    "ee_force",
    "eef_pos",
    "eef_quat",
    "gripper",
    "vision_in_body",
}


def _default_repo_id() -> str:
    """The dataset the shell points at (scripts/dataset_env.sh sets LEROBOT_REPO_ID = <robot>_<task>)."""
    import os

    return os.environ.get("LEROBOT_REPO_ID") or os.environ.get("SPOT_REAL_REPO_ID") or "spot_real_dataset"


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
    task: str | None = None
    repo_id: str | None = None
    robot_type: str = "spot_real"
    fps: int | None = None
    overwrite: bool = True
    push_to_hub: bool = False
    base_image_key: str = "images_agentview"
    wrist_image_key: str = "images_gripper"
    state_components: tuple[str, ...] = ("arm_q", "gripper")
    task_map_path: Path | None = None
    channel_order: Literal["auto", "rgb", "bgr"] = "auto"
    """Image channel order of the input. "auto" reads each file's image_channel_order
    attribute (new recordings are RGB) and refuses files without one."""
    dataset_config: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)


def _collect_h5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.h5") if path.is_file())
    if any(char in str(input_path) for char in "*?[]"):
        return sorted(path for path in input_path.parent.glob(input_path.name) if path.is_file())
    raise FileNotFoundError(f"Could not find input path: {input_path}")


def _iter_demo_groups(h5_file: h5py.File):
    if "data" not in h5_file:
        raise KeyError("Expected a top-level 'data' group in the HDF5 file.")
    for demo_name in sorted(h5_file["data"].keys()):
        yield demo_name, h5_file["data"][demo_name]


def _decode_strings(values: np.ndarray) -> list[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _to_feature_matrix(values: np.ndarray, *, dtype=np.float32) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}")


def _parse_image(image: np.ndarray) -> np.ndarray:
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


def _infer_fps_from_timestamps(timestamps: np.ndarray) -> int | None:
    if timestamps.size < 2:
        return None
    deltas = np.diff(np.asarray(timestamps, dtype=np.float64).reshape(-1))
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None
    return round(1.0 / np.median(deltas))


def _make_state_feature_names(components: tuple[str, ...], arm_joint_names: list[str]) -> list[str]:
    feature_names: list[str] = []
    for component in components:
        match component:
            case "arm_q":
                feature_names.extend(arm_joint_names)
            case "arm_dq":
                feature_names.extend([f"{name}_dq" for name in arm_joint_names])
            case "body_vel":
                feature_names.extend([f"body_vel_{i}" for i in range(6)])
            case "ee_force":
                feature_names.extend([f"ee_force_{i}" for i in range(3)])
            case "eef_pos":
                feature_names.extend(["eef_pos_x", "eef_pos_y", "eef_pos_z"])
            case "eef_quat":
                feature_names.extend(["eef_quat_w", "eef_quat_x", "eef_quat_y", "eef_quat_z"])
            case "gripper":
                feature_names.append("gripper")
            case "vision_in_body":
                feature_names.extend([f"vision_in_body_{i}" for i in range(7)])
            case _:
                raise ValueError(f"Unsupported state component: {component}")
    return feature_names


def _build_state_matrix(obs_group: h5py.Group, components: tuple[str, ...]) -> np.ndarray:
    state_parts = []
    for component in components:
        if component not in STATE_COMPONENTS:
            raise ValueError(f"Unsupported state component: {component}")
        if component not in obs_group:
            raise KeyError(f"Missing observation key '{component}' in the HDF5 file.")
        state_parts.append(_to_feature_matrix(obs_group[component][:]))
    return np.concatenate(state_parts, axis=-1).astype(np.float32)


def _load_task_map(task_map_path: Path | None) -> dict[str, str]:
    if task_map_path is None:
        return {}
    with task_map_path.open() as f:
        data = json.load(f)
    return {str(key): str(value) for key, value in data.items()}


def _resolve_task(task_map: dict[str, str], default_task: str | None, file_path: Path, demo_name: str) -> str:
    if not task_map:
        if default_task is None:
            raise ValueError("Either task or task_map_path must be provided.")
        return default_task

    candidates = (
        f"{file_path.name}:{demo_name}",
        f"{file_path.stem}:{demo_name}",
        demo_name,
    )
    for candidate in candidates:
        if candidate in task_map:
            return task_map[candidate]

    raise KeyError(
        "Could not find a task for demo "
        f"{file_path.name}:{demo_name}. Checked keys: {', '.join(candidates)}"
    )


def _make_feature_spec(
    sample_demo: h5py.Group,
    arm_joint_names: list[str],
    state_components: tuple[str, ...],
    base_image_key: str,
    wrist_image_key: str,
    mode: str,
) -> dict[str, dict[str, object]]:
    obs_group = sample_demo["obs"]
    state_matrix = _build_state_matrix(obs_group, state_components)
    actions = _to_feature_matrix(sample_demo["actions"][:])

    base_image = _parse_image(obs_group[base_image_key][0])
    wrist_image = _parse_image(obs_group[wrist_image_key][0])

    features: dict[str, dict[str, object]] = {
        "base_image": {
            "dtype": mode,
            "shape": base_image.shape,
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": mode,
            "shape": wrist_image.shape,
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_matrix.shape[-1],),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (actions.shape[-1],),
            "names": ["action"],
        },
    }

    raw_feature_names = {
        "arm_q": arm_joint_names,
        "arm_dq": [f"{name}_dq" for name in arm_joint_names],
        "body_vel": [f"body_vel_{i}" for i in range(6)],
        "ee_force": [f"ee_force_{i}" for i in range(3)],
        "eef_pos": ["eef_pos_x", "eef_pos_y", "eef_pos_z"],
        "eef_quat": ["eef_quat_w", "eef_quat_x", "eef_quat_y", "eef_quat_z"],
        "gripper": ["gripper"],
        "vision_in_body": [f"vision_in_body_{i}" for i in range(7)],
        "t": ["t"],
    }

    for key in raw_feature_names:
        if key not in obs_group:
            continue
        values = _to_feature_matrix(obs_group[key][:])
        features[key] = {
            "dtype": "float32",
            "shape": (values.shape[-1],),
            "names": [key],
        }

    return features


def main(args: Args):
    repo_id = args.repo_id or _default_repo_id()
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

    arm_joint_names: list[str] | None = None
    inferred_fps: int | None = None
    total_episodes = 0
    total_frames = 0

    orders: dict[str, str] = {}
    for file_path in h5_files:
        with h5py.File(file_path, "r") as h5_file:
            orders[file_path.name] = _channel_order(h5_file, file_path, args.channel_order)
            if "arm_joint_names" in h5_file and arm_joint_names is None:
                arm_joint_names = _decode_strings(np.asarray(h5_file["arm_joint_names"]))

            for _, demo_group in _iter_demo_groups(h5_file):
                total_episodes += 1
                total_frames += int(np.asarray(demo_group["actions"]).shape[0])

                if inferred_fps is None and "t" in demo_group["obs"]:
                    inferred_fps = _infer_fps_from_timestamps(np.asarray(demo_group["obs"]["t"]))

    if arm_joint_names is None:
        raise KeyError("Could not find 'arm_joint_names' in the HDF5 input.")

    if args.fps is None and inferred_fps is None:
        raise ValueError("Could not infer FPS from obs/t. Please pass --fps explicitly.")
    fps = args.fps if args.fps is not None else inferred_fps
    assert fps is not None

    with h5py.File(h5_files[0], "r") as sample_file:
        _, sample_demo = next(_iter_demo_groups(sample_file))
        features = _make_feature_spec(
            sample_demo,
            arm_joint_names,
            args.state_components,
            args.base_image_key,
            args.wrist_image_key,
            mode="video" if args.dataset_config.use_videos else "image",
        )

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

    tasks_seen: set[str] = set()
    for file_path in h5_files:
        with h5py.File(file_path, "r") as h5_file:
            for demo_name, demo_group in tqdm.tqdm(
                list(_iter_demo_groups(h5_file)),
                desc=f"Converting {file_path.name}",
            ):
                task = _resolve_task(task_map, args.task, file_path, demo_name)
                tasks_seen.add(task)
                obs_group = demo_group["obs"]
                actions = _to_feature_matrix(demo_group["actions"][:])
                state = _build_state_matrix(obs_group, args.state_components)
                base_images = obs_group[args.base_image_key][:]
                wrist_images = obs_group[args.wrist_image_key][:]

                num_frames = actions.shape[0]
                if state.shape[0] != num_frames:
                    raise ValueError(
                        f"State/actions length mismatch in {file_path.name}:{demo_name} "
                        f"({state.shape[0]} vs {num_frames})"
                    )

                raw_features = {}
                for key in ("arm_q", "arm_dq", "body_vel", "ee_force", "eef_pos", "eef_quat", "gripper", "vision_in_body", "t"):
                    if key in obs_group:
                        raw_features[key] = _to_feature_matrix(obs_group[key][:])

                order = orders[file_path.name]
                for frame_idx in range(num_frames):
                    frame = {
                        "base_image": _to_rgb(_parse_image(base_images[frame_idx]), order),
                        "wrist_image": _to_rgb(_parse_image(wrist_images[frame_idx]), order),
                        "state": state[frame_idx],
                        "action": actions[frame_idx],
                        "task": task,
                    }
                    for key, values in raw_features.items():
                        frame[key] = values[frame_idx]
                    dataset.add_frame(frame)

                dataset.save_episode()

    # meta/conversion.json: what this LeRobot copy holds, checked by scripts/train_preflight.py.
    import datetime
    import json

    shape = {k: features[k]["shape"] for k in ("base_image", "wrist_image")}
    record = {
        "converter": "examples/spot_real/convert_spot_real_to_lerobot.py",
        "robot": "spot",
        "data_source": ["real"],
        "state_components": list(args.state_components),
        "action_components": ["arm"],
        "action_dim": int(features["action"]["shape"][0]),
        "base_image_key": args.base_image_key,
        "wrist_image_key": args.wrist_image_key,
        "image_size": {k: [int(v[1]), int(v[0])] for k, v in shape.items()},
        "prompts": sorted(tasks_seen),
        "inputs": [str(p) for p in h5_files],
        "converted_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (output_path / "meta" / "conversion.json").write_text(json.dumps(record, indent=2) + "\n")

    if args.push_to_hub:
        dataset.push_to_hub()

    print(f"Saved LeRobot dataset to {output_path}")
    print(f"Episodes: {total_episodes}")
    print(f"Frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"State components: {list(args.state_components)}")
    print(f"Image channel order: {', '.join(f'{k}: {v}' for k, v in sorted(orders.items()))} (stored RGB)")
    print("Depth images were not converted; the openpi pi0.5 training path in this repo is RGB-only.")


if __name__ == "__main__":
    main(tyro.cli(Args))

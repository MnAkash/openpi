"""Inspect the custom HDF5 dataset structure before converting it to LeRobot."""

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import tyro


def _collect_h5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.h5") if path.is_file())
    if any(char in str(input_path) for char in "*?[]"):
        return sorted(path for path in input_path.parent.glob(input_path.name) if path.is_file())
    raise FileNotFoundError(f"Could not find input path: {input_path}")


def _decode_strings(values: np.ndarray) -> list[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _iter_demo_groups(h5_file: h5py.File):
    if "data" not in h5_file:
        raise KeyError("Expected a top-level 'data' group in the HDF5 file.")
    for demo_name in sorted(h5_file["data"].keys()):
        yield demo_name, h5_file["data"][demo_name]


def _infer_fps_from_timestamps(timestamps: np.ndarray) -> int | None:
    if timestamps.size < 2:
        return None
    deltas = np.diff(np.asarray(timestamps, dtype=np.float64).reshape(-1))
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None
    return round(1.0 / np.median(deltas))


@dataclasses.dataclass(frozen=True)
class Args:
    input_path: Path


def main(args: Args):
    h5_files = _collect_h5_files(args.input_path)
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {args.input_path}")

    total_demos = 0
    frame_counts: list[int] = []
    fps_candidates: list[int] = []

    first_obs_shapes: dict[str, tuple[int, ...]] | None = None
    first_action_shape: tuple[int, ...] | None = None
    arm_joint_names: list[str] | None = None

    for file_path in h5_files:
        with h5py.File(file_path, "r") as h5_file:
            if "arm_joint_names" in h5_file and arm_joint_names is None:
                arm_joint_names = _decode_strings(np.asarray(h5_file["arm_joint_names"]))

            for demo_name, demo_group in _iter_demo_groups(h5_file):
                total_demos += 1
                obs_group = demo_group["obs"]
                actions = np.asarray(demo_group["actions"])
                frame_counts.append(int(actions.shape[0]))

                if "t" in obs_group:
                    fps = _infer_fps_from_timestamps(np.asarray(obs_group["t"]))
                    if fps is not None:
                        fps_candidates.append(fps)

                if first_obs_shapes is None:
                    first_obs_shapes = {key: tuple(obs_group[key].shape) for key in sorted(obs_group.keys())}
                    first_action_shape = tuple(actions.shape)
                    print(f"First demo: {file_path.name}:{demo_name}")

    print(f"HDF5 files: {len(h5_files)}")
    print(f"Total demos: {total_demos}")
    print(
        "Frames per demo: "
        f"min={min(frame_counts)}, median={int(np.median(frame_counts))}, max={max(frame_counts)}"
    )
    if fps_candidates:
        print(f"Inferred FPS: median={int(np.median(fps_candidates))} (from obs/t)")
    else:
        print("Inferred FPS: unavailable (no valid obs/t timestamps found)")
    if arm_joint_names is not None:
        print(f"Arm joint names: {arm_joint_names}")
    if first_action_shape is not None:
        print(f"Action shape example: {first_action_shape}")
    if first_obs_shapes is not None:
        print("Observation shapes from the first demo:")
        for key, shape in first_obs_shapes.items():
            print(f"  {key}: {shape}")


if __name__ == "__main__":
    main(tyro.cli(Args))

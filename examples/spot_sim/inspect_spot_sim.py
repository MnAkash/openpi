"""Print the schema and value ranges of the H2R Spot HDF5 dataset.

Run this before converting: it is the cheapest way to catch a re-render that changed a
key name, lost a camera, or broke frame alignment.

    python examples/spot_sim/inspect_spot_sim.py --input-path "$H5_PATH"
"""

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    input_path: Path
    max_files: int = 3
    """Dump the full schema for this many files; the rest are only checked."""


def _collect(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*.h5") if p.is_file())
    return sorted(p for p in input_path.parent.glob(input_path.name) if p.is_file())


def main(args: Args):
    files = _collect(args.input_path)
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {args.input_path}")
    print(f"{len(files)} file(s) under {args.input_path}\n")

    total_frames = 0
    prompts: set[str] = set()
    fps_estimates: list[float] = []
    problems: list[str] = []

    for index, path in enumerate(files):
        with h5py.File(path, "r") as f:
            if index < args.max_files:
                print(f"=== {path.name} ===")
                if "arm_joint_names" in f:
                    print("  arm_joint_names:", [s.decode() for s in f["arm_joint_names"][:]])
            for name in sorted(f["data"].keys()):
                demo = f["data"][name]
                obs = demo["obs"]
                n = demo["actions"].shape[0]
                total_frames += n

                prompt = demo.attrs.get("task_description")
                if prompt is None:
                    problems.append(f"{path.name}:{name} has no task_description attribute")
                else:
                    prompts.add(prompt.decode() if isinstance(prompt, bytes) else str(prompt))

                for key in obs:
                    if obs[key].shape[0] != n:
                        problems.append(
                            f"{path.name}:{name} obs/{key} has {obs[key].shape[0]} rows, "
                            f"actions has {n}")

                if "t" in obs:
                    dt = np.diff(np.asarray(obs["t"]).reshape(-1))
                    dt = dt[dt > 0]
                    if dt.size:
                        fps_estimates.append(1.0 / float(np.median(dt)))

                if index < args.max_files:
                    print(f"  {name}: {n} frames")
                    for key in sorted(obs):
                        d = obs[key]
                        extra = ""
                        if d.ndim <= 2 and d.dtype.kind == "f":
                            v = d[:]
                            extra = f"  range [{v.min():+.4f}, {v.max():+.4f}]"
                        print(f"    obs/{key:18s} {str(d.shape):20s} {str(d.dtype):8s}{extra}")
                    a = demo["actions"][:]
                    print(f"    {'actions':22s} {str(a.shape):20s} {str(a.dtype):8s}"
                          f"  range [{a.min():+.4f}, {a.max():+.4f}]")
                    if "object_ids" in demo:
                        print(f"    object_ids            {[s.decode() for s in demo['object_ids'][:]]}"
                              "   (NOT converted -- privileged)")
                    print(f"    task_description      {prompt!r}")
                    print()

    print(f"total frames : {total_frames}")
    print(f"prompt(s)    : {sorted(prompts)}")
    if fps_estimates:
        print(f"fps estimate : {np.median(fps_estimates):.2f} "
              f"(min {min(fps_estimates):.2f}, max {max(fps_estimates):.2f})")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems[:20]:
            print("  -", p)
        raise SystemExit(1)
    print("\nno problems found")


if __name__ == "__main__":
    main(tyro.cli(Args))

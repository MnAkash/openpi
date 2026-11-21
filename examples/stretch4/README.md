# Stretch 4 HDF5 to LeRobot

`convert_stretch4_to_lerobot.py` converts Stretch 4 demonstrations, from the real robot
(`stretch4_teleop/create_dataset.py`) and from the 3DGS world simulator, to LeRobot
format. The two sources share one layout, so real and simulated files can be converted
together.

```bash
conda activate openpi
cd /home/akash/extra/akash/openpi
source scripts/dataset_env.sh stretch4 <task>

env -u LD_LIBRARY_PATH python examples/stretch4/convert_stretch4_to_lerobot.py \
    --input-path "$DATASET_PATH/raw" --task "pick wooden cube and place in bowl"
```

Real-robot files carry no prompt, so pass `--task` (or `--task-map-path`, a JSON
mapping `file.h5:demo_0`, `file:demo_0` or `demo_0` to a prompt). Simulator files with a
`task_description` attribute need neither.

## Mapping

| HDF5 | LeRobot | model input |
|---|---|---|
| `obs/images_head_center` (320×240 portrait, RGB) | `base_image` (RGB) | `base_0_rgb` |
| `obs/images_gripper` (240×320, RGB) | `wrist_image` (RGB) | `left_wrist_0_rgb` |
| `obs/eef_pos + eef_quat + gripper` | `state` (8-D) | `state` |
| `actions` (13-D) | `action` | `actions` |

- **Colour:** the real recorder and the simulator both store RGB and mark the file
  `image_channel_order = "rgb"`; the converter reads that attribute (`--channel-order
  auto`) and refuses a file without it. For a file recorded before RGB recording, pass
  `--channel-order bgr` (real robot, flipped to RGB) or `rgb` (simulator).
- **Actions:** `dpos(3) + rot6d(6) + gripper(1) + base_dx, base_dy, base_dyaw`.
  `--no-include-base-actions` keeps only the 10 arm columns, for tasks where the base
  never moves.
- **State:** `--state-components` also accepts `arm_q`, `arm_dq`, `body_vel`,
  `base_in_odom` and `ee_pose_odom`. The two odometry poses depend on where odometry
  started, so they don't transfer between sessions.
- **Dropped:** `obs/object_poses` and `object_ids`, the simulator's privileged ground truth.
- **Provenance:** `meta/episode_sources.jsonl` in the LeRobot output lists each episode's
  file, demo, `data_source` (real or simulator), channel order, robot name and serial,
  end-of-arm tool, and record time, copied from the HDF5 attributes.

## Train

```bash
source scripts/dataset_env.sh stretch4 <task>
# convert as above, then:
scripts/train_policy.sh stretch4 <task> --config pi05_stretch4_lora --compute-norm-stats
```

- `pi05_stretch4_lora`: 13-D actions (arm + base), the converter default.
- `pi05_stretch4_arm_lora`: 10-D, for data converted with `--no-include-base-actions`.

Both use state `eef_pos + eef_quat + gripper`, and head 240×320 / gripper 320×240
images, and reuse the Spot input transforms (`src/openpi/policies/spot_real_policy.py`),
since the converted fields are the same. The script refuses a config whose action width,
state or image sizes differ from the converted data (it reads `meta/conversion.json`).
Options, resuming and outputs: [`TRAINING.md`](../../TRAINING.md). A trained checkpoint
serves with `scripts/serve_policy.py` and returns 10×13 action chunks; there is no
Stretch 4 rollout client yet.

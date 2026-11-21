# Spot Real HDF5 to pi0.5 LoRA

This example adapts a dataset with the following layout to the `openpi` training stack:

```text
arm_joint_names
data/demo_*/actions
data/demo_*/obs/{arm_q, arm_dq, body_vel, ee_force, eef_pos, eef_quat, gripper, images_gripper, images_agentview, t, vision_in_body}
```

It assumes:

- `images_gripper` is the wrist camera (Spot's hand camera).
- `images_agentview` is the external or agent/base camera (the RealSense).
- Datasets recorded before spot_teleop's camera rename call these `images_0` and
  `images_1`; pass `--base-image-key images_1 --wrist-image-key images_0` for them.
- Images are RGB. New recordings say so in the file attribute `image_channel_order`,
  which the converter reads (`--channel-order auto`). Older recordings are OpenCV BGR
  and carry no attribute: pass `--channel-order bgr` and they are flipped to RGB.
- the model state should be built from `arm_q + gripper`.
- the raw `actions` array should be used as-is, without an extra delta-action transform.

Those assumptions are configurable in the converter and training config if your dataset semantics differ.

## 0. Environment setup

Run everything from the repo root in the `openpi` env, and point the scripts at one dataset:

```bash
conda activate openpi
cd <openpi repo>
source scripts/dataset_env.sh spot <task>     # e.g. spot stove_eggplant
```

Place your source `.h5` files in `data/spot/<task>/raw/` (`$H5_PATH`). Every script then writes under `$DATASET_PATH` (`data/spot/<task>/`): converted LeRobot data under `lerobot/`, normalization stats under `norm_stats/`, checkpoints under `checkpoints/`, and rollouts under `rollouts/`.

For this machine, keep the host CUDA toolkit on `12.8`, but run JAX commands without `LD_LIBRARY_PATH` so the JAX CUDA 12 wheel libraries are not overridden by `/usr/local/cuda` or ROS paths.

## 1. Inspect the raw HDF5

```bash
python examples/spot_real/inspect_spot_real.py --input-path "$H5_PATH"
```

If `obs/t` is present, the script will also infer an FPS estimate for you.

## 2. Convert to LeRobot

```bash
python examples/spot_real/convert_spot_real_to_lerobot.py \
    --input-path "$H5_PATH" \
    --task "pick the wooden cube and drop in the white bowl"
```

It writes to `$DATASET_PATH/lerobot/$LEROBOT_REPO_ID` (`spot_<task>`, from `scripts/dataset_env.sh`), which is where
the training config looks, along with `meta/conversion.json` recording the state layout, actions, cameras and
prompt. `--repo-id` overrides it; outside `dataset_env.sh` it falls back to `spot_real_dataset`.

Useful flags:

- `--fps 15` if your timestamps are missing or unreliable.
- `--state-components arm_q gripper` to control which proprio fields are concatenated into `observation/state`.
- `--base-image-key images_1 --wrist-image-key images_0 --channel-order bgr` for datasets recorded before the camera rename and RGB recording (e.g. `cube_rick_vr_20`, `stove_vr_20_akash`). Checkpoints trained on those before this fix saw BGR; retrain after re-converting.
- `--task-map-path /path/to/tasks.json` if you have multiple tasks. The JSON can map either `demo_0` or `file_name.h5:demo_0` to a string prompt.

Depth images are intentionally ignored here because the current `openpi` training path in this repo is RGB-only.

## 3. Train

Use the generic training script. It computes the norm stats (`--compute-norm-stats`, needed once per
conversion), checks that `pi05_spot_real_lora` matches the converted data, and resumes automatically:

```bash
scripts/train_policy.sh spot <task> --config pi05_spot_real_lora --compute-norm-stats
scripts/train_policy.sh spot <task> --status
```

See `TRAINING.md` at the repo root for every option (steps, batch, checkpoint interval, `--detach`,
extending a finished run) and for where checkpoints and logs go.

- Use `pi05_spot_real_lora` for real-robot data. Its state is `arm_q + gripper`, the converter default.
  Simulator Spot data uses `pi05_spot_sim_lora` and `examples/spot_sim/convert_spot_sim_to_lerobot.py`.
- Batch 8 fits the 32 GB RTX 5090 with the script's memory settings.
- Don't change the model's `action_dim` of 32; it pads the 10-D actions internally.

## 4. Serve the trained policy and run Spot rollout

Start the OpenPI policy server from the `openpi` environment. Select the checkpoint step by changing the final directory in `--policy.dir`.

```bash
conda activate openpi
cd <openpi repo>

env -u LD_LIBRARY_PATH \
TF_GPU_ALLOCATOR=cuda_malloc_async \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config pi05_spot_real_lora \
    --policy.dir "$DATASET_PATH/checkpoints/pi05_spot_real_lora/<exp>/<step>"
```

Then run the Spot rollout client from the `openpi` environment, which has `spot_teleop` installed (`--port` must stay before `policy:checkpoint` above, or tyro rejects it). See `ROLLOUT_SPOT.md` at the repo root for the full rollout guide. The script reads Spot connection settings from your shell environment: `SPOT_ROBOT_IP`, `BOSDYN_CLIENT_USERNAME`, and `BOSDYN_CLIENT_PASSWORD`.

```bash
conda activate openpi
cd <openpi repo>

python scripts/rollout_pi05_spot.py \
    --policy-host 127.0.0.1 \
    --policy-port 8000 \
    --prompt "pick the wooden cube and drop in the white bowl" \
    --horizon 200 \
    --output-dir "$DATASET_PATH/rollouts"
```

Rollout keys:

- `r` starts one rollout.
- `s` discards the active rollout and resets the arm to the default pose.
- `d` docks the robot and exits.
- `q` or `Esc` exits without docking.

Useful rollout args:

- `--horizon 200`
  Maximum number of rollout steps before the script auto-stops and saves the rollout.

- `--control-hz 10`
  Control frequency for policy execution. With the default horizon, this gives about 20 seconds per rollout.

- `--gripper-full-close 0.6`
  Off by default. Commands a fully closed gripper whenever the policy asks for at least 60% closure (opening <= 0.4), for a tighter grip. Higher openings pass through, so releases still work.

- `--prompt "..."`
  Pass the exact task string the dataset was converted with.

What the rollout sends is set by the served config, so it matches training without flags:

- **State:** the config's `policy_metadata["state_components"]`, which the server sends
  on connect. `pi05_spot_real_lora` is `arm_q + gripper` and `pi05_spot_sim_lora` is
  `eef_pos + eef_quat + gripper`. Both are 8-D, so the rollout refuses to start if the
  server doesn't say. `--state-components` overrides it, and must agree with the server.
  If you convert with a different `--state-components`, update the config's
  `policy_metadata` to match.
- **Images:** both cameras as RGB at the recording size, 320x240 (`policy_metadata["image_size"]`).
  The hand camera is resized from 640x480 with `INTER_AREA`, exactly as `spot_teleop`
  records it, and the server pads to 224x224 keeping the aspect ratio, as in training.
  `--image-width` / `--image-height` override it. Keep `--realsense-width/height` equal
  to the recording size.

## Important note on action semantics

The Spot real config assumes your saved `actions` tensor is already in the controller-ready action space: delta motion commands plus the absolute gripper value. The rollout script passes policy actions back through the `spot_teleop` controller logic, so no extra action conversion is needed here.

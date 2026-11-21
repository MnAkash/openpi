# openpi — local setup for Akash's robots

This file is the operating manual for training pi0.5 on this machine's own datasets.
Upstream `README.md` still describes the stock repo; everything here is local.

**To train any robot or dataset, use `scripts/train_policy.sh` — see [`TRAINING.md`](TRAINING.md).**
**To run a trained policy on Spot, see [`ROLLOUT_SPOT.md`](ROLLOUT_SPOT.md).** The sections below
describe the original simulated Spot dataset and the layout everything follows.

**Environment.** Use the `openpi` conda env. This fork raises the library versions in
`pyproject.toml` (jax 0.10.1, flax 0.12.7, orbax 0.12.0, torch 2.8.0, numpy 2.x, Python
3.11+) so training runs on an RTX 5090, but `uv.lock` is still upstream's. `uv sync` would
not reproduce this setup.

Written 2026-09-08. Current dataset: **Spot, `task_cube_no_force`**, 25 demonstrations
rendered by the Hand2Robot (H2R) 3DGS simulator.

---

## 1. Quick start

```bash
conda activate openpi
cd /home/akash/extra/akash/openpi

# point every script at one robot + task
source scripts/dataset_env.sh spot task_cube_no_force

# 1. look at the raw HDF5 (schema, ranges, frame alignment, prompt)
env -u LD_LIBRARY_PATH python examples/spot_sim/inspect_spot_sim.py --input-path "$H5_PATH"

# 2. convert to LeRobot
env -u LD_LIBRARY_PATH python examples/spot_sim/convert_spot_sim_to_lerobot.py --input-path "$H5_PATH" \
    --wrist-image-key images_ego --channel-order rgb   # predates the images_gripper rename and the colour marker (§3)

# 3. norm stats + train (TRAINING.md)
scripts/train_policy.sh spot task_cube_no_force --config pi05_spot_sim_lora --compute-norm-stats
```

`env -u LD_LIBRARY_PATH` is not optional. A host CUDA 12.8 install and ROS both put
libraries on `LD_LIBRARY_PATH` that shadow the ones the JAX CUDA wheel ships, and JAX
fails in confusing ways when they win.

---

## 2. Where everything lives

Everything openpi reads or writes is under `data/`, organised **by robot, then by task**,
so a second robot is a new directory rather than a new naming scheme:

```
data/
├── _cache/                      shared by every robot — do not duplicate per dataset
│   ├── hf_home/                 HuggingFace datasets metadata      (~8 MB)
│   └── openpi/                  downloaded pi base weights         (12 GB, pi05_base)
├── spot/
│   ├── task_cube_no_force/      <-- the current dataset
│   │   ├── SOURCE.txt           where raw/ came from, and how to re-sync
│   │   ├── raw/                 25 × demo_NNN.h5                   (580 MB)
│   │   ├── lerobot/             converted dataset                  (18 MB)
│   │   │   └── spot_task_cube_no_force/{data,videos,meta}
│   │   ├── norm_stats/          per training config
│   │   │   └── pi05_spot_sim_lora/spot_task_cube_no_force/norm_stats.json
│   │   ├── checkpoints/         written by training
│   │   └── rollouts/            written by the rollout script
│   ├── cube_rick_vr_20/         legacy, real-robot VR teleop       (20 GB)
│   └── stove_vr_20_akash/       legacy, real-robot VR teleop       (55 GB)
└── stretch4/                    same layout per task; config pi05_stretch4_lora
```

`data/` is gitignored, so none of this is in version control.

`scripts/dataset_env.sh <robot> <task>` is the only thing that knows this layout.
It exports `DATASET_PATH`, `H5_PATH`, `HF_HOME`, `HF_LEROBOT_HOME`, `OPENPI_DATA_HOME`
and `LEROBOT_REPO_ID` (= `<robot>_<task>`, so two datasets can never collide in one
cache). The training config reads `DATASET_PATH` to find norm stats, so a config never
hardcodes a path.

### Code

| path | what |
|---|---|
| `examples/spot_sim/` | this dataset: converter + inspector |
| `examples/spot_real/` | the real-robot Spot datasets (`images_agentview` / `images_gripper`) |
| `examples/stretch4/` | Stretch 4, real and simulated: converter + README |
| `src/openpi/policies/spot_real_policy.py` | the Spot input/output transforms, **shared by both** |
| `src/openpi/training/config.py` | `pi05_spot_sim_lora`, `pi05_spot_real_lora`, `pi05_stretch4_lora`, `pi05_stretch4_arm_lora` |
| `scripts/dataset_env.sh` | the per-dataset environment |
| `scripts/train_policy.sh` | train / resume / extend any dataset ([`TRAINING.md`](TRAINING.md)) |
| `scripts/train_preflight.py` | the config-vs-dataset checks `train_policy.sh` runs first |
| `scripts/rollout_pi05_spot.py` | Spot rollout client ([`ROLLOUT_SPOT.md`](ROLLOUT_SPOT.md)) |
| `scripts/sync_spot_task_cube.sh` | re-copy the H2R renders into `raw/` |

---

## 3. The dataset

Produced by the H2R pipeline, stage 04, from bare-hand iPhone demonstrations retargeted
onto Spot and re-rendered in a 3D Gaussian-splat capture of the Carl lab. Source of
truth: `/home/akash/UNH/04_corl26/hand2robot_pipeline/04_render_replay/`.

- 25 episodes, **1679 frames**, 10 Hz, 6.0–7.4 s each
- one prompt: `"pick wooden cube and place in bowl"`
- two 320×240 camera streams
- scene `carl-lab-env`, objects `wooden-cube` (manipulated) and `bowl` (target)

### Raw HDF5 (`raw/demo_NNN.h5`)

```
arm_joint_names                    (7,) bytes    arm_sh0 arm_sh1 arm_el0 arm_el1 arm_wr0 arm_wr1 arm_f1x
data/demo_0/
    actions                        (N, 10) f32
    object_ids                     (2,) bytes    [wooden-cube, bowl]
    obs/images_agentview           (N, 240, 320, 3) u8    iso camera, world mounted
    obs/images_ego                 (N, 240, 320, 3) u8    wrist camera, gripper mounted (*)
    obs/eef_pos                    (N, 3)  f32    BASE frame
    obs/eef_quat                   (N, 4)  f32    BASE frame, xyzw
    obs/arm_q                      (N, 7)  f32    joint angles, rad
    obs/gripper                    (N, 1)  f32    1 = open, 0 = closed
    obs/t                          (N, 1)  f64    dt = 0.1 s
    obs/object_poses               (N, 2, 7) f32  WORLD frame — NOT USED
    .attrs["task_description"]     str            the training prompt
```

`N` is 61–75 per demo. Every array has the same `N` (checked by the inspector).

(*) The simulator and spot_teleop now name the gripper camera `images_gripper`, and the
Spot converters default to it. This dataset was rendered before the rename, so convert
it with `--wrist-image-key images_ego`. Real Spot data recorded before the rename uses
`images_0` (gripper) / `images_1` (agentview): pass
`--base-image-key images_1 --wrist-image-key images_0` to `convert_spot_real_to_lerobot.py`.

**Colour.** Every new recording (spot_teleop, stretch4_teleop, the simulator) stores RGB
`uint8 (N, H, W, 3)` and marks the file `image_channel_order = "rgb"`. All three
converters read that attribute and refuse a file without it, so a colour mix-up fails
loudly instead of training silently. Older files carry no attribute: pass
`--channel-order rgb` for simulator renders (like this dataset) and `--channel-order bgr`
for real-robot recordings (`cube_rick_vr_20`, `stove_vr_20_akash`), which are then
flipped to RGB. The `pi05_spot_real_lora` checkpoints trained before this saw BGR while
the rollout script sends RGB; re-convert and retrain them.

**`actions` layout** — verified numerically against the observations, not assumed:

| cols | meaning |
|---|---|
| 0:3 | `delta_pos` = `eef_pos[i+1] − eef_pos[i]`, BASE frame, metres |
| 3:9 | `rot6d` of the **relative** rotation `R_iᵀ R_{i+1}` (first two columns stacked) |
| 9 | **absolute** gripper, and it is `gripper[i+1]` — the *next* step's state |

So the arm channels are deltas and the gripper channel is absolute and already one step
ahead. A policy trained on this predicts where to go next and what the gripper should be
when it arrives.

### What is deliberately dropped

`obs/object_poses` and `object_ids` are **not** converted. They are simulator ground
truth that a real Spot cannot observe, so a policy that used them could not be deployed.
The converter lists them in `DROPPED_KEYS` and prints what it dropped, so this stays a
decision rather than an oversight. Add any future privileged key to that tuple.

### Mapping into the model

| HDF5 | LeRobot | model input |
|---|---|---|
| `obs/images_agentview` | `base_image` | `base_0_rgb` (mask True) |
| `obs/images_ego` | `wrist_image` | `left_wrist_0_rgb` (mask True) |
| — | — | `right_wrist_0_rgb` zeros (mask False) |
| `obs/eef_pos + eef_quat + gripper` | `state` (8-D) | `state`, padded to 32 |
| `actions` | `action` (10-D) | `actions`, padded to 32, horizon 10 |
| `.attrs["task_description"]` | `task` | tokenized prompt (200 tokens) |

**State is `eef_pos + eef_quat + gripper` (8-D)** because that matches the action space:
the actions are end-effector deltas plus an absolute gripper, so end-effector pose plus
gripper is what a policy in that space actually needs. `obs/arm_q` is available and is
redundant through forward kinematics; with only 1679 frames the compact state is the
safer default. To include it:

```bash
python examples/spot_sim/convert_spot_sim_to_lerobot.py --input-path "$H5_PATH" \
    --state-components arm_q eef_pos eef_quat gripper      # -> 15-D
```

Re-run `compute_norm_stats.py` after any change to the state, and delete the old
checkpoint directory — the state dimension is baked into a trained model.

---

## 4. Known property of `images_ego` — read before trusting the wrist view

**The ego camera does not see the cube during the carry.** Spot's calibrated hand-eye
extrinsic puts the camera 80 mm above the jaw contact point and tilts it 7.6° up, so a
held object sits 45 mm ahead of the camera and 86 mm below it — outside a 47° vertical
field of view. Measured on `demo_001`: the cube is in the ego frame for 15 of 71 rows
and **0 of the 23 carry rows**.

This is a property of the real camera, not a rendering bug — it was checked against the
jaw collision meshes in the Spot URDF. The calibrated extrinsic was kept deliberately,
so that what the policy trains on is what the real robot will show at deployment.
`images_ego` is therefore most informative during approach and after release. If a
future experiment wants a jaws-centred ego view, it is a per-scene `mount_pose_ee`
override in the simulator, and it makes the training view no longer match the robot.

Details: `/home/akash/UNH/04_corl26/hand2robot_pipeline/04_render_replay/PLAN.md`, §Cameras.

---

## 5. Training config

`pi05_spot_sim_lora`, in `src/openpi/training/config.py`.

| | |
|---|---|
| model | pi0.5, LoRA (`gemma_2b_lora` + `gemma_300m_lora`) |
| init | `gs://openpi-assets/checkpoints/pi05_base/params` (already cached, 12 GB) |
| action horizon | 10 |
| action dim | 10 (model pads to 32 internally — **do not change the model's 32**) |
| batch size | 4 |
| steps | 30 000 (≈71 passes over 1679 frames) |
| save interval | 2 000, keeping every 10 000 |
| normalisation | quantile (pi0.5 always uses quantile, never z-score) |

It reuses `LeRobotSpotRealDataConfig` and the `SpotRealInputs`/`SpotRealOutputs`
transforms: once converted to LeRobot both Spot datasets present identical
`base_image` / `wrist_image` / `state` / `action` / `prompt` fields, so a second copy of
those transforms would only be something to keep in sync.

> One norm-stats note. `actions` channel 3 (`rot6d[0]`) has a reported std of exactly
> `0.0` — the values sit at 0.9996 ± 0.0003 and the running variance underflows in
> float32. This is harmless **because pi0.5 normalises by quantiles**, and that channel's
> q99−q01 spread is 0.0017, which normalises to a clean ±1. It would matter if anything
> were switched to z-score normalisation.

---

## 6. Train

```bash
scripts/train_policy.sh spot task_cube_no_force --config pi05_spot_sim_lora --compute-norm-stats
```

Everything else (steps, batch, checkpoint interval, resuming, extending, `--detach`,
`--status`, where checkpoints and logs go, what it checks first) is in
[`TRAINING.md`](TRAINING.md). The script defaults to batch 8 and 30 000 steps with a
checkpoint every 5 000. This config's own defaults (batch 4, save every 2 000) apply
only when `scripts/train.py` is run by hand.

## 7. Serve and roll out

```bash
env -u LD_LIBRARY_PATH TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_spot_sim_lora \
    --policy.dir "$DATASET_PATH/checkpoints/pi05_spot_sim_lora/task_cube_no_force_lora/29999"
```

Then `scripts/rollout_pi05_spot.py` (see `ROLLOUT_SPOT.md` and `examples/spot_real/README.md` §4 for the
rollout keys and Spot connection variables). The rollout client feeds policy actions
back through the `spot_teleop` controller, which expects exactly this action space —
end-effector deltas plus an absolute gripper — so no extra conversion is needed. The
client takes its state layout (here `eef_pos + eef_quat + gripper`) and image size
(320×240 RGB) from this config's `policy_metadata`, so pass only `--prompt` with the
dataset's exact task string.

Note the sim-to-real gap before trusting a rollout: this policy is trained entirely on
3DGS renders, and `images_ego` follows the real camera geometry but the pixels are
synthetic.

---

## 8. Adding another robot or task

Nothing in the code is Spot-specific except the input transforms. For a new **task on
Spot** with the same HDF5 schema:

```bash
mkdir -p data/spot/<task>/{raw,lerobot,norm_stats,checkpoints,rollouts}
cp <your h5 files> data/spot/<task>/raw/
source scripts/dataset_env.sh spot <task>
# then the same convert -> compute_norm_stats -> train, with the same config
```

`LEROBOT_REPO_ID` becomes `spot_<task>` automatically, so the new dataset gets its own
LeRobot copy and its own norm stats without touching anything else.

**Stretch 4 is set up**: converter `examples/stretch4/convert_stretch4_to_lerobot.py`,
configs `pi05_stretch4_lora` (13-D arm + base actions) and `pi05_stretch4_arm_lora`
(10-D, converted with `--no-include-base-actions`). They reuse the Spot input transforms,
since the converted fields are the same. See [`TRAINING.md`](TRAINING.md).

For another **robot**, add a converter that writes the same LeRobot fields (`base_image`,
`wrist_image`, `state`, `action`, `task`) plus `meta/conversion.json`, and a `TrainConfig`
with `repo_id=_dataset_repo_id(...)`, `assets_base_dir=_dataset_assets_base_dir()` and a
`policy_metadata` naming the robot, state components, action components and image size.
`scripts/train_policy.sh` then works for it unchanged.

---

## 9. Gotchas

- **`env -u LD_LIBRARY_PATH` on every JAX command.** See §1.
- **The editable install was repointed.** `openpi` and `openpi-client` were installed
  from `/home/akash/UNH/00_tools/openpi`, which no longer exists. The `.pth` files in
  `~/miniconda3/envs/openpi/lib/python3.11/site-packages/` now point at
  `/home/akash/extra/akash/openpi`. If imports break after a reinstall, run
  `pip install -e . --no-deps` from this directory.
- **`source` the env script, don't run it.** It only exports variables.
- **Re-running the converter deletes the existing LeRobot copy** (`--overwrite` defaults
  to true). Norm stats are not deleted — recompute them after any conversion change.
- **The legacy datasets moved** into `data/spot/`. If an old command fails, its
  `DATASET_PATH` still points at `data/cube_rick_vr_20`; use the env script instead.
- **Don't change the model's `action_dim` of 32.** It pads the 10-D actions internally.

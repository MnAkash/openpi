# Training a pi0.5 policy on this machine

One script trains every robot and dataset: `scripts/train_policy.sh`. This page covers
going from recorded `.h5` files to checkpoints. To run a trained Spot policy on the
robot, see `ROLLOUT_SPOT.md`.

```
data/<robot>/<task>/raw/*.h5   --convert-->   lerobot/   --train_policy.sh-->   checkpoints/<config>/<exp>/<step>
                                                          (norm stats, checks, train, logs)
```

Run everything from the repo root in the `openpi` env:

```bash
conda activate openpi
cd /mnt/extra/akash/openpi
```

---

## 1. Pick the config for your data

| Data | Converter | Config | State | Actions |
|---|---|---|---|---|
| Real Spot (`spot_teleop`) | `examples/spot_real/convert_spot_real_to_lerobot.py` | `pi05_spot_real_lora` | `arm_q + gripper` | 10-D arm |
| Simulated Spot (3DGS simulator, either export format) | `examples/spot_sim/convert_spot_sim_to_lerobot.py` | `pi05_spot_sim_lora` | `eef_pos + eef_quat + gripper` | 10-D arm |
| Stretch 4, real or simulated, arm + base | `examples/stretch4/convert_stretch4_to_lerobot.py` | `pi05_stretch4_lora` | `eef_pos + eef_quat + gripper` | 13-D arm + base |
| Stretch 4, arm only (convert with `--no-include-base-actions`) | same | `pi05_stretch4_arm_lora` | same | 10-D arm |

Every config uses pi0.5 LoRA from `pi05_base`, 10-step action chunks and quantile
normalization. The state and action layouts are baked into a trained model, and each
config tells the rollout client which layout and image size to send.

Real and simulated Spot data are both 8-D state but mean different things. Use the
config in the row that matches your data. The script refuses a mismatch (§4).

## 2. Put the data in place and convert it

```bash
mkdir -p data/<robot>/<task>/raw
cp /path/to/*.h5 data/<robot>/<task>/raw/
source scripts/dataset_env.sh <robot> <task>      # sets DATASET_PATH, H5_PATH, LEROBOT_REPO_ID=<robot>_<task>
```

Then run the converter for your row. Every converter writes to
`data/<robot>/<task>/lerobot/<robot>_<task>/`, where the configs look.

```bash
# Real Spot: real recordings carry no prompt, so give it
env -u LD_LIBRARY_PATH PYTHONPATH= python examples/spot_real/convert_spot_real_to_lerobot.py \
    --input-path "$H5_PATH" --task "put the pot on the stove, then put the eggplant in the pot"

# Simulated Spot: the prompt comes from each demo's task_description (set when exporting); --task overrides
env -u LD_LIBRARY_PATH PYTHONPATH= python examples/spot_sim/convert_spot_sim_to_lerobot.py --input-path "$H5_PATH"

# Stretch 4 (real needs --task; simulator exports carry task_description)
env -u LD_LIBRARY_PATH PYTHONPATH= python examples/stretch4/convert_stretch4_to_lerobot.py \
    --input-path "$H5_PATH" --task "pick wooden cube and place in bowl"
```

Each converter prints a summary: episodes, frames, fps, state, actions, cameras and
prompts. It also writes `meta/conversion.json`, which the training script checks the
config against.

**Simulator exports.** The 3DGS simulator's *Build Dataset* has two formats, and both
train:

- **Real robot (stretch4_teleop)**, the default: 13 action columns.
  - For **Stretch 4** this is the native layout. Use `pi05_stretch4_lora`.
  - For **Spot**, the Spot sim converter keeps the first 10 columns, which are the arm.
    It checks that the base never moved, because Spot's policy has no base actions.
- **Legacy**: 10 action columns, read by the Spot sim converter as they are.

For the same trajectory, the two formats give identical Spot training data (checked on
`spot_ee_demo`). A Stretch 4 export uses the robot's own cameras: `images_head_center`
(240×320 portrait) and `images_gripper` (320×240). A Spot export uses the scene's
record cameras: `images_agentview` (world-mounted) and `images_gripper` (gripper-mounted).
Export Spot cameras at 320×240, which is the size the Spot rollout sends.

**Frame rate.** Each action is one step at the recording rate. Real recordings are
10 Hz. A simulator export runs at the trajectory's own rate (e.g. `spot_ee_demo` is
30 Hz). The script warns when Spot data isn't 10 Hz: roll it out with
`--control-hz <rate>`.

## 3. Train

```bash
scripts/train_policy.sh <robot> <task> --config <config> --compute-norm-stats
```

`--compute-norm-stats` is needed the first time, and again after any re-conversion. Without
it, the script stops if the stats are missing or stale. Examples:

```bash
scripts/train_policy.sh spot stove_eggplant --config pi05_spot_real_lora --compute-norm-stats
scripts/train_policy.sh spot sim_cube --config pi05_spot_sim_lora --compute-norm-stats --detach
scripts/train_policy.sh stretch4 cube --config pi05_stretch4_lora --compute-norm-stats --batch 8 --steps 30000
```

| Option | Default | Meaning |
|---|---|---|
| `--config NAME` | required | Training config from §1 |
| `--exp NAME` | `<task>_lora` | Checkpoint folder name |
| `--steps N` | 30000 | Total steps. On a run that already has checkpoints, a larger number extends it. |
| `--batch N` | 8 | Batch size. 8 fits the 32 GB RTX 5090. |
| `--save-every N` | 5000 | Checkpoint interval. Every interval checkpoint is kept. |
| `--compute-norm-stats` | off | Compute the config's norm stats first |
| `--detach` | off | Run in the background; closing the terminal doesn't stop it |
| `--overwrite` | off | Start over. Deletes this run's checkpoints, after asking (`--yes` skips the question). |
| `--allow-busy-gpu` | off | Start even if another process holds GPU memory |
| `--dry-run` | off | Run all checks and print the command, without training |
| `--status` | | Show whether it's running, the latest loss, progress and ETA, and the checkpoints |
| `-- ARGS` | | Passed to `scripts/train.py` as-is, e.g. `-- --no-wandb-enabled` |

### Resuming, extending, watching

- **Resume after a crash or a stop:** run the same command again. If the run has
  checkpoints, it continues from the newest one.
- **Extend a finished run:** run it again with a larger `--steps`, e.g.
  `--steps 40000`. With the default, the script stops and says the run already reached
  its steps.
- **Watch:** `scripts/train_policy.sh <robot> <task> --status`, or
  `tail -f data/<robot>/<task>/logs/train_<exp>.log`. Runs also go to Weights & Biases
  (project `openpi`) unless you pass `-- --no-wandb-enabled`.
- **Plot the loss:** the log has `Step N: loss=...` every 100 steps.

### Where things go

| | |
|---|---|
| Checkpoints | `data/<robot>/<task>/checkpoints/<config>/<exp>/<step>/` (each about 9 GB) |
| Preserved checkpoints | `data/<robot>/<task>/checkpoints/<config>/<exp>_kept/<step>/` (see below) |
| Norm stats | `data/<robot>/<task>/norm_stats/<config>/<robot>_<task>/norm_stats.json`, also copied into every checkpoint's `assets/` |
| Log | `data/<robot>/<task>/logs/train_<exp>.log` (appended across resumes) |
| Run history | `data/<robot>/<task>/train_runs.txt`: one line per start or resume, with the time, config, step range, batch, openpi commit and extra args |

### Which checkpoints survive

openpi keeps a checkpoint if its step is a multiple of the save interval. It also keeps
the newest one, but deletes it at the next save. The script handles this:

- It trains to `--steps` + 1 internally. The final checkpoint is then exactly
  `--steps` (30000, not 29999), which is a multiple and is kept.
- Before resuming, it hard-links any checkpoint whose step isn't a multiple, e.g. 34999
  from an older run, into `<exp>_kept/`. That takes no time and no extra disk, and the
  copy survives when openpi deletes the original. Serve it from there.
- openpi never saves the step a run resumes at. For example, a resume from 34999 saves
  next at 40000.

## 4. What the script checks before using the GPU

It stops with a message saying what to fix if any of these is true:

- the config doesn't exist, or doesn't read its dataset from `dataset_env.sh`;
- the converted dataset is missing;
- the norm stats are missing, or stale (their sizes differ from the data's);
- the data is a different robot from the config's;
- the action width differs from the config's (e.g. 13-D Stretch data with the 10-D config);
- the state layout differs, compared by name using `meta/conversion.json`, e.g. real vs
  simulated Spot;
- the image sizes differ from what the config tells the rollout client to send;
- the same run is already training;
- the GPU is busy, e.g. a policy server is still running.

It also prints the episodes, frames, rate, cameras, prompts, and how many passes over the
data the run makes. A dataset converted before converters wrote `conversion.json` gets a
size-only state check and a warning. Re-converting writes the record.

## 5. After training

- **Spot:** serve a checkpoint and roll it out as in `ROLLOUT_SPOT.md`. Use the same
  `--policy.config` you trained with, and the prompt the converter printed.
- **Stretch 4:** checkpoints serve the same way:

  ```bash
  python scripts/serve_policy.py --port 8000 policy:checkpoint \
      --policy.config pi05_stretch4_lora --policy.dir <checkpoint>
  ```

  It returns 10×13 action chunks. There is no Stretch 4 rollout client yet. Its
  `policy_metadata` gives a future client the state layout and both image sizes.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no converted dataset at ...` | Convert first (§2). Run the converter after `source scripts/dataset_env.sh <robot> <task>` so it writes where the config looks. |
| `no norm stats` / `norm stats ... are stale` | Add `--compute-norm-stats`. |
| `the dataset's state is X but <config> uses Y` | Use the config from §1 that matches the data, or re-convert with `--state-components`. |
| `the base moves ... Spot's arm policy has no base actions` | Export Spot from the simulator with the base parked. |
| `the GPU is in use` | Stop the policy server or other job; `nvidia-smi` shows it. |
| `already reached step N of M` | Pass a larger `--steps` to extend, or `--overwrite` to start over. |
| Out of memory on the first step | Lower `--batch`. The script already preallocates 95% of the GPU, which is what batch 8 needs. |

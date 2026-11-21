# Rolling out a pi0.5 policy on Spot

Training a policy first: see [`TRAINING.md`](TRAINING.md).

How to run a trained `pi05_spot_real_lora` checkpoint on the real Spot. The commands use
the **stove / eggplant** policy; to roll out another task, change the three things marked
**(per task)**.

Two processes run on this machine (Carl5090, the one connected to Spot and the RealSense):

```
terminal 1: serve_policy.py   GPU, loads the checkpoint, listens on port 8000
terminal 2: rollout_pi05_spot.py   talks to Spot + RealSense, asks the server for actions
```

Both run in the `openpi` conda env, which also has `spot_teleop` (editable install from
`/home/akash/UNH/spot/spot_teleop`) and `pyrealsense2`. The `teleop` env does not have
`spot_teleop` and will not work.

---

## Stove / eggplant at a glance

| | |
|---|---|
| Config | `pi05_spot_real_lora` |
| Dataset | `source scripts/dataset_env.sh spot stove_eggplant` |
| Checkpoint | `data/spot/stove_eggplant/checkpoints/pi05_spot_real_lora/stove_eggplant_lora/34999` (try first), `25000` (fallback) |
| Prompt | `put the pot on the stove, then put the eggplant in the pot` (exactly this) |
| State sent | `arm_q (7) + gripper (1)`, set automatically by the server |
| Images sent | RealSense agentview + hand camera, RGB 320x240, set automatically by the server |
| Actions | 10-D: body-frame `dpos` (3), `rot6d` (6), absolute gripper 0..1; 10-step chunks |
| Demos | 25, 148–212 steps at 10 Hz (15–21 s) |

---

## 1. Set up the scene like the recordings

The policy only knows what it saw in the 25 demos, so match them:

- **RealSense:** the same mount, position and angle as during recording, at 320x240. If the
  camera has moved, the agentview image is different from training and the policy degrades.
- **Objects:** pot, eggplant and stove in roughly the places they were in the demos, with
  the same table/background and similar lighting.
- **Robot base:** every demo started with the body at about **x = 1.08 m, y = 0.03 m,
  yaw = 2°** in Spot's vision frame (about where Spot stands after undocking). The rollout
  script undocks and stands, so starting from the dock is normally enough.
  To put the base exactly where a demo started, see [Exact base pose](#exact-base-pose-optional).
- **Arm:** nothing to do. The rollout resets the hand to `0.55 0 0.55`, pitched down 60°,
  with the gripper open, which is exactly the first pose of all 25 demos.

## 2. Terminal 1: start the policy server

```bash
conda activate openpi
cd /mnt/extra/akash/openpi
source scripts/dataset_env.sh spot stove_eggplant        # (per task) sets SPOT_REAL_REPO_ID

env -u LD_LIBRARY_PATH PYTHONPATH= \
python scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config pi05_spot_real_lora \
    --policy.dir "$DATASET_PATH/checkpoints/pi05_spot_real_lora/stove_eggplant_lora/34999"   # (per task)
```

- `--port` must come **before** `policy:checkpoint`. After it, tyro rejects it with
  "Unrecognized options: --port".
- `source scripts/dataset_env.sh` is required: the config finds the checkpoint's norm stats
  (`34999/assets/spot_stove_eggplant/`) through `SPOT_REAL_REPO_ID`.
- Wait for `server listening on 0.0.0.0:8000` (about a minute). It takes most of the GPU, so
  don't train at the same time.
- To compare checkpoints, stop the server (Ctrl-C) and restart it with another step
  directory (`25000`, `20000`, ...). The rollout client can stay as it is; restart it too.

## 3. Terminal 2: run the rollout

```bash
conda activate openpi
cd /mnt/extra/akash/openpi
source scripts/dataset_env.sh spot stove_eggplant        # (per task) rollouts go to $DATASET_PATH/rollouts

export SPOT_ROBOT_IP=<spot ip>
export BOSDYN_CLIENT_USERNAME=<user>
export BOSDYN_CLIENT_PASSWORD=<password>
# export REALSENSE_SERIAL=<serial>   # only if more than one RealSense is plugged in

env -u LD_LIBRARY_PATH PYTHONPATH= \
python scripts/rollout_pi05_spot.py \
    --policy-host 127.0.0.1 --policy-port 8000 \
    --prompt "put the pot on the stove, then put the eggplant in the pot" \
    --horizon 250 \
    --max-delta-pos 0.04 \
    --output-dir "$DATASET_PATH/rollouts/ckpt_34999"
```

On start it connects to the server first, and prints the policy metadata (`state_components
['arm_q', 'gripper']`, `image_size [320, 240]`) and a config summary. Only then does it
connect to Spot, undock, stand, and move the arm to the start pose. A preview window shows
`agentview | wrist`.

**Keys** (with the preview window focused):

| Key | Action |
|---|---|
| `r` | start one rollout |
| `s` | stop the current rollout, discard it, reset the arm to the start pose |
| `d` | dock and exit |
| `q` / `Esc` | exit without docking |

A rollout runs until `--horizon` steps, then resets the arm and saves
`rollout_NNNN/{agent_view.mp4, wrist_view.mp4, trajectory.txt}` in `--output-dir`.
Rollouts stopped with `s` are not saved. Between rollouts, reset the pot and eggplant to
their start positions, then press `r` again.

**Keep the e-stop in reach.** The policy moves the arm directly. The only limits are the
per-step position clip below and the gripper clip to 0..1.

### Arguments worth knowing

| Argument | Use | Why |
|---|---|---|
| `--prompt` | the training prompt, word for word | The model was conditioned on this exact string. |
| `--horizon 250` | 25 s at 10 Hz | The longest demo is 212 steps. The default 200 would cut the slow ones short. |
| `--max-delta-pos 0.04` | per-step hand move clip (m) | The largest recorded step is 0.039 m (99th percentile 0.025). The default 0.03 clips the fastest moves. Lower it (e.g. 0.02) to make a first test run slower. |
| `--gripper-full-close 0.6` | grip harder | Off by default. When the policy commands at least 60% closed (opening <= 0.4), the gripper is commanded fully closed (0.0), so it squeezes the object as hard as Spot allows instead of stopping at the about 0.07 opening the demos recorded. Openings above 0.4 pass through unchanged, so releases still work. The client applies it, so the server doesn't need a restart. |
| `--control-hz 10` | default; leave it | The demos were recorded at 10 Hz, so the action deltas are per 0.1 s. |
| `--replan-every 10` | default; leave it | The policy predicts 10-step chunks. Smaller values re-query more often (more reactive, more GPU calls). |
| `--no-undock` | the robot is already standing off the dock | |
| `--state-components`, `--image-width/height` | don't pass | Taken from the server, and checked against it. |

---

## Checked before writing this (2026-11-20)

I served checkpoint 34999 with the commands above and sent it nine recorded frames from
demos 0, 12 and 24, using the rollout script's own client and image/state code. The
results:

- It loaded norm stats from the checkpoint and reported `{'robot': 'spot',
  'state_components': ['arm_q', 'gripper'], 'image_size': [320, 240]}`.
- It returned `(10, 10)` action chunks.
- The predicted chunks matched the recorded actions to about 1 mm per step in position
  (recorded steps average 2–11 mm).
- The gripper matched: open at about 1.0, closed at about 0.07.

These were training frames, so this shows the pipeline (images, state, norm stats, action
format) is wired correctly end to end. It does not show how well the policy generalizes on
the robot.

## Exact base pose (optional)

`/home/akash/UNH/spot/spot_teleop/replay_hdf5.py` walks Spot to the base pose a demo was
recorded from, using the `vision_in_body` field. It only works if Spot hasn't rebooted
since recording, because the vision frame resets on reboot; past 5 m it refuses to move. It
then unstows and replays that demo's arm motion. To use it only for positioning:

1. Run it:

   ```bash
   cd /home/akash/UNH/spot/spot_teleop
   python replay_hdf5.py /mnt/extra/akash/openpi/data/spot/stove_eggplant/raw/akash_stove_eggplant.h5 --demo 0
   ```

2. Press Ctrl-C as soon as it prints `Unstowing the arm`, and choose `keep`.
3. Start the rollout with `--no-undock`.

Alternatively, let it replay the whole demo with the objects cleared from the arm's path.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Unrecognized options: --port` | Put `--port 8000` before `policy:checkpoint`. |
| Server: norm stats not found / `spot_real_dataset` in the path | `source scripts/dataset_env.sh spot stove_eggplant` wasn't run in that terminal. |
| `ModuleNotFoundError: spot_teleop` | Wrong env; use `openpi`. |
| `The policy server did not say which state it was trained on` | The server is an old openpi without `policy_metadata`, or the wrong config. |
| `External RealSense camera did not start` | Camera unplugged or in use by another process (e.g. a teleop session). Set `REALSENSE_SERIAL` if several are connected. |
| `[warn] RealSense streams WxH ...` | Keep `--realsense-width/height` at 320x240, the recording size. |
| Arm drifts or hesitates at the start | The scene or camera view differs from the demos; check the RealSense pose and object placement first. Then try checkpoint 25000. |
| Moves look jerky or capped | Raise `--max-delta-pos` toward 0.04, but not above the recorded maximum. |

## Another task

Use `pi05_spot_real_lora` for any real-Spot dataset converted with
`examples/spot_real/convert_spot_real_to_lerobot.py`. Then change:

1. `source scripts/dataset_env.sh spot <task>` (both terminals).
2. `--policy.dir` to `$DATASET_PATH/checkpoints/pi05_spot_real_lora/<exp>/<step>`.
3. `--prompt` to the `--task` string that dataset was converted with. It is recorded in
   `data/spot/<task>/SOURCE.txt` and in `lerobot/<repo>/meta/tasks.jsonl`.

For a simulator-trained policy, use `--policy.config pi05_spot_sim_lora`. The rollout then
sends `eef_pos + eef_quat + gripper` automatically. See README_CUSTOM.md §7.

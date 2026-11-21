#!/usr/bin/env python3
"""Keyboard-gated pi0.5 rollouts on Spot.

Controls:
  r: start one rollout
  s: discard the active rollout and reset the arm to the default pose
  d: dock the robot and exit
  q / Esc: exit without docking


Example usage:
python scripts/rollout_pi05_spot.py \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --prompt "pick the wooden cube and drop in the white bowl" \
  --output-dir "$DATASET_PATH/rollouts"
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
import time
from typing import Any

import cv2
import msgpack
import numpy as np
import websockets.sync.client


DEFAULT_PROMPT = "pick the wooden cube and drop in the white bowl"
DEFAULT_POLICY_HOST = os.environ.get("OPENPI_HOST", "127.0.0.1")
DEFAULT_POLICY_PORT = int(os.environ.get("OPENPI_PORT", "8000"))
ARM_JOINT_ORDER = (
    "arm0.sh0",
    "arm0.sh1",
    "arm0.el0",
    "arm0.el1",
    "arm0.wr0",
    "arm0.wr1",
    "arm0.f1x",
)


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class OpenPIWebsocketClient:
    def __init__(self, host: str, port: int | None, api_key: str | None, timeout: float):
        if host.startswith(("ws://", "wss://")):
            self.uri = host
        else:
            self.uri = f"ws://{host}:{port}" if port is not None else f"ws://{host}"
        self.api_key = api_key
        self.timeout = float(timeout)
        self._packer = msgpack.Packer(default=_pack_array)
        self._ws = None
        self.metadata: dict[str, Any] = {}
        self.connect()

    def connect(self) -> None:
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        kwargs = {"compression": None, "max_size": None, "open_timeout": self.timeout}
        try:
            if headers:
                kwargs["additional_headers"] = headers
            self._ws = websockets.sync.client.connect(self.uri, **kwargs)
        except TypeError:
            kwargs.pop("additional_headers", None)
            if headers:
                kwargs["extra_headers"] = headers
            self._ws = websockets.sync.client.connect(self.uri, **kwargs)
        raw_metadata = self._ws.recv()
        self.metadata = msgpack.unpackb(raw_metadata, object_hook=_unpack_array)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("Policy websocket is not connected.")
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI policy server returned an error:\n{response}")
        return msgpack.unpackb(response, object_hook=_unpack_array)

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None


@dataclasses.dataclass
class SpotObservation:
    state: np.ndarray
    agent_rgb: np.ndarray
    wrist_rgb: np.ndarray
    eef_pos: np.ndarray
    eef_quat_xyzw: np.ndarray
    gripper: float


def _extract_arm_q(robot_state) -> np.ndarray:
    joint_by_name = {
        joint.name: float(joint.position.value)
        for joint in robot_state.kinematic_state.joint_states
        if joint.name.startswith("arm0.")
    }
    missing = [name for name in ARM_JOINT_ORDER if name not in joint_by_name]
    if missing:
        raise RuntimeError(f"Missing Spot arm joints in robot state: {missing}")
    return np.asarray([joint_by_name[name] for name in ARM_JOINT_ORDER], dtype=np.float32)


def _normalized_gripper(robot_state) -> float:
    pct = float(robot_state.manipulator_state.gripper_open_percentage)
    return float(np.clip(pct / 100.0 if pct > 1.0 else pct, 0.0, 1.0))


def _prepare_rgb(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """A camera frame exactly as spot_teleop recorded it for training.

    The recorder resizes each OpenCV BGR frame to the recording size with INTER_AREA
    (the hand camera is 640x480 -> 320x240; the RealSense already streams 320x240) and
    saves it RGB. The policy server then pads it to 224x224 keeping the aspect ratio
    (ResizeImages), so sending anything already squashed to 224x224 would show the
    model a distorted view it never trained on.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR frame, got shape {bgr.shape}")
    if bgr.shape[:2] != (height, width):
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(bgr.astype(np.uint8, copy=False), cv2.COLOR_BGR2RGB))


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# Observation pieces a training config can ask for, in the units the converters store.
STATE_COMPONENT_DIMS = {"arm_q": 7, "gripper": 1, "eef_pos": 3, "eef_quat": 4}


def _resolve_state_components(metadata: dict[str, Any], override: list[str] | None) -> list[str]:
    """The state layout the served policy trained on.

    The training config publishes it as policy_metadata["state_components"]
    (pi05_spot_real_lora: arm_q + gripper; pi05_spot_sim_lora: eef_pos + eef_quat +
    gripper). Both are 8 numbers, so a wrong guess would not raise; it would just feed
    the policy meaningless state. Hence: use the server's, or an explicit override, never
    a default.
    """
    served = metadata.get("state_components")
    served = [str(c) for c in served] if served is not None else None
    if override is not None:
        if served is not None and list(override) != served:
            raise RuntimeError(f"--state-components {override} differs from the policy's {served}")
        components = list(override)
    elif served is not None:
        components = served
    else:
        raise RuntimeError(
            "The policy server did not say which state it was trained on (no "
            "policy_metadata['state_components']). Pass --state-components, e.g. "
            "'arm_q gripper' (pi05_spot_real_lora) or 'eef_pos eef_quat gripper' (pi05_spot_sim_lora).")
    unknown = [c for c in components if c not in STATE_COMPONENT_DIMS]
    if unknown:
        raise RuntimeError(f"Unsupported state components {unknown}; known: {sorted(STATE_COMPONENT_DIMS)}")
    return components


def _resolve_image_size(metadata: dict[str, Any], width: int | None, height: int | None) -> tuple[int, int]:
    """(W, H) the training images were recorded at: --image-width/--image-height, else the server's."""
    served = metadata.get("image_size")
    if width is not None and height is not None:
        return int(width), int(height)
    if served is not None:
        return int(served[0]), int(served[1])
    return 320, 240  # spot_teleop's recording size


def _make_state_vector(components: list[str], *, arm_q: np.ndarray, gripper: float,
                       eef_pos: np.ndarray, eef_quat_xyzw: np.ndarray) -> np.ndarray:
    """Concatenate the state exactly as the converter did (same pieces, same order).

    eef_pos / eef_quat are the hand in Spot's body frame, quaternion xyzw: what the
    recorder (and the simulator's base frame) stores. gripper is the open fraction 0..1.
    """
    parts = {
        "arm_q": np.asarray(arm_q, dtype=np.float32).reshape(-1),
        "gripper": np.asarray([gripper], dtype=np.float32),
        "eef_pos": np.asarray(eef_pos, dtype=np.float32).reshape(-1),
        "eef_quat": np.asarray(eef_quat_xyzw, dtype=np.float32).reshape(-1),
    }
    return np.concatenate([parts[c] for c in components], axis=0).astype(np.float32)


def _make_policy_observation(obs: SpotObservation, prompt: str) -> dict[str, Any]:
    return {
        "observation/state": obs.state,
        "observation/base_image": obs.agent_rgb,
        "observation/wrist_image": obs.wrist_rgb,
        "prompt": prompt,
    }


def _safe_action(action: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (10,):
        raise ValueError(f"Expected action shape (10,), got {action.shape}")
    safe = action.copy()
    safe[:3] *= np.asarray(args.pos_scale, dtype=np.float32)
    safe[:3] = np.clip(safe[:3], -float(args.max_delta_pos), float(args.max_delta_pos))
    safe[9] = float(np.clip(safe[9], 0.0, 1.0))
    safe[9] = _snap_gripper(safe[9], args.gripper_full_close)
    return safe


def _snap_gripper(opening: float, full_close: float | None) -> float:
    """Command a fully closed gripper once the policy asks for at least `full_close` closure.

    The action's gripper is an opening fraction (1 = open, 0 = closed). The policy learned
    to command what the demos recorded while holding an object (about 0.07 open), which
    can grip too loosely. With full_close=0.6, any opening <= 0.4 (>= 60% closed) becomes
    0.0, so the claw squeezes as hard as Spot allows. Opening commands above the
    threshold pass through unchanged, so releases still work. None disables it.
    """
    if full_close is not None and opening <= 1.0 - float(full_close):
        return 0.0
    return float(opening)


def _closure_fraction(value: str) -> float:
    closure = float(value)
    if not 0.0 < closure <= 1.0:
        raise argparse.ArgumentTypeError("--gripper-full-close must be in (0, 1], e.g. 0.6 for 60% closed")
    return closure


class ExternalAgentCamera:
    def __init__(self, *, width: int, height: int, fps: int, device_serial: str | None):
        from spot_teleop.camera_streamer import CameraStreamer

        self.streamer = CameraStreamer(width=width, height=height, fps=fps, device_serial=device_serial)
        if not self.streamer.start():
            raise RuntimeError("External RealSense camera did not start.")

    def read_bgr(self) -> np.ndarray:
        color_bgr, _ = self.streamer.get_latest()
        if color_bgr is None:
            raise RuntimeError("External RealSense camera has not produced a color frame yet.")
        return color_bgr

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        last_err: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                self.read_bgr()
                return
            except Exception as exc:
                last_err = exc
                time.sleep(0.05)
        raise RuntimeError(f"External RealSense camera was not ready after {timeout:.1f}s: {last_err}")

    def stop(self) -> None:
        self.streamer.stop()


def _capture_observation(controller, agent_camera: ExternalAgentCamera, args: argparse.Namespace) -> SpotObservation:
    state = controller.current_state()
    arm_q = _extract_arm_q(state)
    gripper = _normalized_gripper(state)
    eef_pos, eef_quat_xyzw = controller.current_ee_pose()
    width, height = args.image_wh
    agent_rgb = _prepare_rgb(agent_camera.read_bgr(), width, height)
    wrist_rgb = _prepare_rgb(controller.get_hand_image(), width, height)
    return SpotObservation(
        state=_make_state_vector(args.state_components, arm_q=arm_q, gripper=gripper,
                                 eef_pos=eef_pos, eef_quat_xyzw=eef_quat_xyzw),
        agent_rgb=agent_rgb,
        wrist_rgb=wrist_rgb,
        eef_pos=eef_pos,
        eef_quat_xyzw=eef_quat_xyzw,
        gripper=gripper,
    )


def _apply_action(controller, action: np.ndarray, args: argparse.Namespace):
    action = _safe_action(action, args)
    controller.apply_action(action)


def _open_gripper(controller) -> None:
    """Open the gripper using SpotRobotController.send_gripper()."""
    controller.send_gripper(1.0)


def _reset_pose_with_open_gripper(controller, args: argparse.Namespace) -> None:
    controller.reset_pose(args.default_pose)
    _open_gripper(controller)
    time.sleep(float(args.reset_wait))


def _show_preview(obs: SpotObservation, status: str) -> int:
    agent = _rgb_to_bgr(obs.agent_rgb)
    wrist = _rgb_to_bgr(obs.wrist_rgb)
    view = np.concatenate([agent, wrist], axis=1)
    cv2.putText(view, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Spot pi0.5 rollout: agent | wrist", view)
    return cv2.waitKey(1) & 0xFF


def _save_video(path: Path, frames_rgb: list[np.ndarray], fps: float) -> None:
    if not frames_rgb:
        return
    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for frame in frames_rgb:
            writer.write(_rgb_to_bgr(frame))
    finally:
        writer.release()


def _write_trajectory(path: Path, rows: list[tuple[float, np.ndarray, np.ndarray, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# t x y z qx qy qz qw gripper\n")
        for t, pos, quat, gripper in rows:
            values = [t, *pos.tolist(), *quat.tolist(), gripper]
            f.write(" ".join(f"{value:.8f}" for value in values) + "\n")


def _save_rollout(output_dir: Path, rollout_idx: int, agent_frames: list[np.ndarray], wrist_frames: list[np.ndarray], trajectory, fps: float) -> Path:
    rollout_dir = output_dir / f"rollout_{rollout_idx:04d}"
    rollout_dir.mkdir(parents=True, exist_ok=False)
    _save_video(rollout_dir / "agent_view.mp4", agent_frames, fps)
    _save_video(rollout_dir / "wrist_view.mp4", wrist_frames, fps)
    _write_trajectory(rollout_dir / "trajectory.txt", trajectory)
    return rollout_dir


def _parse_pose(text: str) -> list[float]:
    values = [float(part) for part in text.replace(",", " ").split()]
    if len(values) != 7:
        raise argparse.ArgumentTypeError("Pose must contain 7 numbers: x y z qx qy qz qw")
    quat = np.asarray(values[3:], dtype=np.float32)
    if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-3):
        raise argparse.ArgumentTypeError("Default pose quaternion must be unit length.")
    return values


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default=DEFAULT_POLICY_HOST)
    parser.add_argument("--policy-port", type=int, default=DEFAULT_POLICY_PORT)
    parser.add_argument("--api-key", default=os.environ.get("OPENPI_API_KEY"))
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=Path("spot_pi05_rollouts"))
    parser.add_argument("--realsense-serial", default=os.environ.get("REALSENSE_SERIAL"))
    parser.add_argument("--realsense-width", type=int, default=320)
    parser.add_argument("--realsense-height", type=int, default=240)
    parser.add_argument("--realsense-fps", type=int, default=30)
    parser.add_argument("--camera-warmup", type=float, default=1.0)
    parser.add_argument("--image-width", type=int, default=None,
                        help="Frame width sent to the policy; default: the policy's image_size, else 320.")
    parser.add_argument("--image-height", type=int, default=None,
                        help="Frame height sent to the policy; default: the policy's image_size, else 240.")
    parser.add_argument("--state-components", nargs="+", default=None, choices=sorted(STATE_COMPONENT_DIMS),
                        help="State layout to send; default: the policy server's state_components.")
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--replan-every", type=int, default=10)
    parser.add_argument("--command-duration", type=float, default=0.1)
    parser.add_argument("--max-delta-pos", type=float, default=0.03)
    parser.add_argument("--gripper-full-close", type=_closure_fraction, default=None, metavar="CLOSURE",
                        help="Close the gripper fully whenever the policy commands at least this closure "
                             "(0..1; 0.6 = 60%% closed, i.e. opening <= 0.4). Off by default.")
    parser.add_argument("--pos-scale", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--default-pose", type=_parse_pose, default=_parse_pose("0.55 0 0.55 0 0.5 0 0.8660254"))
    parser.add_argument("--reset-wait", type=float, default=1.0)
    parser.add_argument("--no-undock", action="store_true")
    parser.add_argument("--no-stand", action="store_true")
    return parser


def _print_startup_summary(args: argparse.Namespace, robot_ip: str, username: str) -> None:
    print("\n[rollout config]", flush=True)
    print(f"  Spot robot IP        : {robot_ip}", flush=True)
    print(f"  Spot username        : {username}", flush=True)
    print(f"  Policy server        : {args.policy_host}:{args.policy_port}", flush=True)
    print(f"  Prompt               : {args.prompt}", flush=True)
    print(f"  Output dir           : {args.output_dir}", flush=True)
    print(f"  External camera      : RealSense {args.realsense_width}x{args.realsense_height}@{args.realsense_fps}", flush=True)
    print(f"  RealSense serial     : {args.realsense_serial or '<first available>'}", flush=True)
    print(f"  Policy image size    : {args.image_wh[0]}x{args.image_wh[1]} RGB (padded to 224 by the server)", flush=True)
    print(f"  Policy state         : {' + '.join(args.state_components)} "
          f"({sum(STATE_COMPONENT_DIMS[c] for c in args.state_components)}-D)", flush=True)
    print(f"  Control rate         : {args.control_hz} Hz", flush=True)
    print(f"  Command duration     : {args.command_duration} s", flush=True)
    print(f"  Horizon              : {args.horizon} steps", flush=True)
    print(f"  Replan every         : {args.replan_every} steps", flush=True)
    print("  Action application   : SpotRobotController.apply_action()", flush=True)
    print(f"  Max delta position   : {args.max_delta_pos} m", flush=True)
    print(f"  Position scale       : {tuple(args.pos_scale)}", flush=True)
    if args.gripper_full_close is None:
        print("  Gripper full close   : off (gripper follows the policy)", flush=True)
    else:
        print(f"  Gripper full close   : on, at >= {args.gripper_full_close:.0%} closed "
              f"(opening <= {1.0 - args.gripper_full_close:.2f} -> 0.0)", flush=True)
    print(f"  Default pose         : {args.default_pose}", flush=True)
    print(f"  Undock on start      : {not args.no_undock}", flush=True)
    print(f"  Stand on start       : {not args.no_stand}", flush=True)
    print("", flush=True)


def main() -> None:
    args = _build_arg_parser().parse_args()

    from spot_teleop.spot_controller import SpotRobotController

    robot_ip = os.environ["SPOT_ROBOT_IP"]
    username = os.environ["BOSDYN_CLIENT_USERNAME"]
    password = os.environ["BOSDYN_CLIENT_PASSWORD"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = OpenPIWebsocketClient(args.policy_host, args.policy_port, args.api_key, args.connect_timeout)
    print(f"[info] connected policy server: {policy.uri}", flush=True)
    if policy.metadata:
        print(f"[info] policy metadata: {policy.metadata}", flush=True)
    # Build the observation the served policy trained on, before touching the robot.
    args.state_components = _resolve_state_components(policy.metadata or {}, args.state_components)
    args.image_wh = _resolve_image_size(policy.metadata or {}, args.image_width, args.image_height)
    if (args.realsense_width, args.realsense_height) != args.image_wh:
        print(f"[warn] RealSense streams {args.realsense_width}x{args.realsense_height}; frames are resized "
              f"to {args.image_wh[0]}x{args.image_wh[1]}. Record and roll out at the same stream size.", flush=True)
    _print_startup_summary(args, robot_ip, username)

    controller = SpotRobotController(
        robot_ip,
        username,
        password,
        default_exec_time=float(args.command_duration),
    )
    agent_camera = ExternalAgentCamera(
        width=int(args.realsense_width),
        height=int(args.realsense_height),
        fps=int(args.realsense_fps),
        device_serial=args.realsense_serial,
    )
    agent_camera.wait_ready(float(args.camera_warmup))
    rollout_idx = 0

    try:
        if not args.no_undock:
            controller.undock()
            time.sleep(2.0)
        if not args.no_stand:
            controller.stand()
            time.sleep(1.0)
        _reset_pose_with_open_gripper(controller, args)

        print("[info] keys: r=start rollout, s=discard/reset, d=dock, q/esc=quit", flush=True)
        while True:
            obs = _capture_observation(controller, agent_camera, args)
            key = _show_preview(obs, "idle: r=start  d=dock  q=quit")
            if key == ord("r"):
                rollout_idx += 1
                saved = _run_one_rollout(controller, agent_camera, policy, args, rollout_idx)
                if saved:
                    print(f"[info] saved rollout to {saved}", flush=True)
            elif key == ord("d"):
                controller.dock()
                break
            elif key in (ord("q"), 27):
                break
            time.sleep(0.03)
    finally:
        agent_camera.stop()
        policy.close()
        cv2.destroyAllWindows()


def _run_one_rollout(
    controller,
    agent_camera: ExternalAgentCamera,
    policy: OpenPIWebsocketClient,
    args: argparse.Namespace,
    rollout_idx: int,
) -> Path | None:
    dt = 1.0 / float(args.control_hz)
    plan: np.ndarray | None = None
    plan_idx = 0
    agent_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    trajectory: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    start_t = time.monotonic()

    for step in range(int(args.horizon)):
        loop_t = time.monotonic()
        obs = _capture_observation(controller, agent_camera, args)
        agent_frames.append(obs.agent_rgb.copy())
        wrist_frames.append(obs.wrist_rgb.copy())
        trajectory.append((time.monotonic() - start_t, obs.eef_pos.copy(), obs.eef_quat_xyzw.copy(), float(obs.gripper)))

        key = _show_preview(obs, f"rollout {rollout_idx:04d}: s=discard/reset  d=dock")
        if key == ord("s"):
            print("[info] discarding current rollout and resetting arm", flush=True)
            _reset_pose_with_open_gripper(controller, args)
            return None
        if key == ord("d"):
            controller.dock()
            raise SystemExit(0)
        if key in (ord("q"), 27):
            raise SystemExit(0)

        if plan is None or plan_idx >= min(int(args.replan_every), plan.shape[0]):
            response = policy.infer(_make_policy_observation(obs, args.prompt))
            if "actions" not in response:
                raise RuntimeError(f"Policy response did not include 'actions'. Keys: {list(response.keys())}")
            plan = np.asarray(response["actions"], dtype=np.float32)
            if plan.ndim != 2 or plan.shape[1] != 10:
                raise RuntimeError(f"Expected policy action chunk shape (H, 10), got {plan.shape}")
            plan_idx = 0

        _apply_action(controller, plan[plan_idx], args)
        plan_idx += 1

        sleep_s = dt - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)

    _reset_pose_with_open_gripper(controller, args)
    return _save_rollout(args.output_dir, rollout_idx, agent_frames, wrist_frames, trajectory, float(args.control_hz))


if __name__ == "__main__":
    main()

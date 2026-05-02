"""
main.py
-------
Entry point for the Webcam-Based Motion Combat Controller.

Architecture (optional threading mode)
---------------------------------------
    Thread 1 (capture)  : reads frames from webcam into a shared queue
    Thread 2 (pose)     : runs MediaPipe on each frame, fills skeleton queue
    Thread 3 (main/UI)  : gesture detection + input + debug overlay + display

Single-threaded mode is also supported (set THREADED = False below).

Controls
--------
    Q  – quit
    D  – toggle debug overlay
    S  – toggle mouse control (head tracking)
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gesture_engine import GestureEngine, GestureEvent
from input_mapper import InputMapper
from pose_tracker import PoseTracker, Skeleton

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config.json"

THREADED = True          # set False for easier debugging
SHOW_DEBUG = True        # toggle with 'D' key at runtime
MOUSE_CONTROL = False    # toggle with 'S' key at runtime (off by default)


def load_config(path: Path = CONFIG_PATH) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    print(f"[WARN] config.json not found at {path}; using defaults.")
    return {}


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------

GESTURE_LABELS = {
    GestureEvent.ATTACK:        "ATTACK",
    GestureEvent.HEAVY_CHARGE:  "HEAVY CHARGE",
    GestureEvent.HEAVY_RELEASE: "HEAVY RELEASE",
    GestureEvent.COUNTER:       "COUNTER",
    GestureEvent.DODGE:         "DODGE",
    GestureEvent.THROW:         "THROW",
    GestureEvent.SPECIAL_START: "SPECIAL START",
    GestureEvent.SPECIAL_END:   "SPECIAL END",
}


def draw_debug_overlay(
    frame: np.ndarray,
    engine: "GestureEngine",
    fps: float,
    active_label: str,
    timing: str,
    mouse_active: bool,
    config: dict,
) -> np.ndarray:
    """Render FPS, active gesture, pose states, deadzone, and cooldowns onto *frame*."""
    from gesture_engine import PoseFeatures
    h, w = frame.shape[:2]
    out = frame.copy()

    def _text(text: str, pos: Tuple[int, int], scale: float = 0.55,
              color: Tuple = (0, 255, 0), thickness: int = 1) -> None:
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    # --- Deadzone rectangle ---
    dz     = config.get("deadzone", {})
    dz_cx  = float(dz.get("x",      0.5))
    dz_cy  = float(dz.get("y",      0.5))
    dz_hw  = float(dz.get("width",  0.3)) / 2.0
    dz_hh  = float(dz.get("height", 0.3)) / 2.0
    dz_x1  = int((dz_cx - dz_hw) * w)
    dz_y1  = int((dz_cy - dz_hh) * h)
    dz_x2  = int((dz_cx + dz_hw) * w)
    dz_y2  = int((dz_cy + dz_hh) * h)

    feat = engine.last_features
    head_in_dz = (feat is not None and
                  dz_cx - dz_hw <= feat.nose_x <= dz_cx + dz_hw and
                  dz_cy - dz_hh <= feat.nose_y <= dz_cy + dz_hh)
    dz_color = (0, 220, 80) if head_in_dz else (0, 140, 255)
    cv2.rectangle(out, (dz_x1, dz_y1), (dz_x2, dz_y2), dz_color, 2)
    _text("DEADZONE", (dz_x1 + 4, dz_y1 + 16), scale=0.38, color=dz_color)

    # --- FPS ---
    _text(f"FPS: {fps:.1f}", (10, 25), color=(255, 255, 255))

    # --- Active gesture label ---
    label_color = (0, 200, 255) if timing == "perfect" else (0, 255, 120)
    if active_label:
        _text(active_label, (10, 55), scale=0.9, color=label_color, thickness=2)
        if timing:
            _text(timing.upper(), (10, 85), scale=0.6, color=label_color)

    # --- State flags ---
    flags: List[str] = []
    if engine.is_special_active:
        flags.append("SPECIAL ACTIVE")
    if engine.is_heavy_charging:
        flags.append("HEAVY CHARGING")
    for i, flag in enumerate(flags):
        _text(flag, (10, 115 + i * 22), color=(0, 140, 255))

    # --- Per-limb pose state ---
    if feat is not None:
        pose_lines: List[str] = []
        if feat.right_arm_extended:
            pose_lines.append("R-ARM: EXTENDED")
        elif feat.right_arm_retracted:
            pose_lines.append("R-ARM: RETRACTED")
        else:
            pose_lines.append("R-ARM: neutral")
        if feat.left_arm_extended:
            pose_lines.append("L-ARM: EXTENDED")
        elif feat.left_arm_retracted:
            pose_lines.append("L-ARM: RETRACTED")
        else:
            pose_lines.append("L-ARM: neutral")
        if feat.right_hand_above_head:
            pose_lines.append("R-HAND: ABOVE HEAD")
        if feat.left_hand_above_head:
            pose_lines.append("L-HAND: ABOVE HEAD")
        y_pose = 165
        for line in pose_lines:
            col = (60, 220, 255) if line.split(":")[1].strip() != "neutral" else (120, 120, 120)
            _text(line, (10, y_pose), scale=0.38, color=col)
            y_pose += 17

    # --- Cooldown bars ---
    cd = engine.cooldown_display()
    y_start = h - 10 - len(cd) * 18
    for i, (name, ms) in enumerate(cd.items()):
        ratio = min(ms / 300.0, 1.0)
        bar_w = int(80 * (1.0 - ratio))
        bar_x = w - 95
        bar_y = y_start + i * 18
        cv2.rectangle(out, (bar_x, bar_y - 12), (bar_x + 80, bar_y), (50, 50, 50), -1)
        cv2.rectangle(out, (bar_x, bar_y - 12), (bar_x + bar_w, bar_y), (0, 200, 80), -1)
        _text(f"{name[:10]:<10}", (bar_x - 85, bar_y - 2), scale=0.38,
              color=(200, 200, 200))

    # --- Mouse control indicator ---
    mouse_str = "MOUSE: ON" if mouse_active else "MOUSE: OFF"
    mouse_col = (0, 255, 100) if mouse_active else (100, 100, 100)
    _text(mouse_str, (w - 120, 20), scale=0.45, color=mouse_col)

    return out


# ---------------------------------------------------------------------------
# Threaded pipeline
# ---------------------------------------------------------------------------

def threaded_main(config: dict, tracker: PoseTracker,
                   engine: GestureEngine, mapper: InputMapper) -> None:
    """Run capture, pose, and gesture/display on separate threads."""

    raw_q:      "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=2)
    pose_q:     "queue.Queue[Optional[Tuple[np.ndarray, Skeleton]]]" = queue.Queue(maxsize=2)
    stop_event  = threading.Event()

    # ---- Thread 1: capture ----
    def capture_worker() -> None:
        while not stop_event.is_set():
            frame = tracker.read_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            try:
                raw_q.put_nowait(frame)
            except queue.Full:
                pass

    # ---- Thread 2: pose ----
    def pose_worker() -> None:
        while not stop_event.is_set():
            try:
                frame = raw_q.get(timeout=0.05)
            except queue.Empty:
                continue
            skeleton = tracker.process(frame)
            try:
                pose_q.put_nowait((frame, skeleton))
            except queue.Full:
                pass

    t1 = threading.Thread(target=capture_worker, daemon=True)
    t2 = threading.Thread(target=pose_worker,    daemon=True)
    t1.start()
    t2.start()

    # ---- Thread 3 (main): gesture + input + display ----
    _run_main_loop(config, engine, mapper, tracker, pose_q, stop_event)

    stop_event.set()
    t1.join(timeout=2)
    t2.join(timeout=2)


def single_threaded_main(config: dict, tracker: PoseTracker,
                          engine: GestureEngine, mapper: InputMapper) -> None:
    """Run everything on the main thread (simpler, slightly higher latency)."""
    stop_event = threading.Event()
    pose_q: "queue.Queue[Optional[Tuple[np.ndarray, Skeleton]]]" = queue.Queue()

    def feeder() -> None:
        while not stop_event.is_set():
            frame = tracker.read_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            skeleton = tracker.process(frame)
            try:
                pose_q.put_nowait((frame, skeleton))
            except queue.Full:
                pass

    t = threading.Thread(target=feeder, daemon=True)
    t.start()
    _run_main_loop(config, engine, mapper, tracker, pose_q, stop_event)
    stop_event.set()
    t.join(timeout=2)


def _run_main_loop(
    config: dict,
    engine: GestureEngine,
    mapper: InputMapper,
    tracker: PoseTracker,
    pose_q: "queue.Queue",
    stop_event: threading.Event,
) -> None:
    global SHOW_DEBUG, MOUSE_CONTROL

    fps_counter: List[float] = []
    fps = 0.0
    active_label = ""
    label_ttl    = 0.0   # seconds to keep label visible

    LABEL_DURATION = 0.6  # seconds

    while True:
        try:
            item = pose_q.get(timeout=0.1)
        except queue.Empty:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break
            continue

        if item is None:
            break

        frame, skeleton = item
        t_now = time.monotonic()

        # FPS
        fps_counter.append(t_now)
        fps_counter = [t for t in fps_counter if t_now - t < 1.0]
        fps = float(len(fps_counter))

        # Gesture detection
        events = engine.update()

        # Input simulation
        mapper.handle_events(events)
        if MOUSE_CONTROL:
            mapper.update_mouse(skeleton)

        # Update active label
        if events:
            active_label = " + ".join(
                GESTURE_LABELS.get(e, str(e)) for e in events)
            label_ttl = t_now + LABEL_DURATION
        elif t_now > label_ttl:
            active_label = ""

        # Draw skeleton
        annotated = tracker.draw(frame, skeleton)

        # Draw debug overlay
        if SHOW_DEBUG:
            annotated = draw_debug_overlay(
                annotated, engine, fps, active_label,
                engine.last_timing, MOUSE_CONTROL, config)

        cv2.imshow("Motion Combat Controller", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break
        elif key == ord('d'):
            SHOW_DEBUG = not SHOW_DEBUG
        elif key == ord('s'):
            MOUSE_CONTROL = not MOUSE_CONTROL
            print(f"[INFO] Mouse control: {'ON' if MOUSE_CONTROL else 'OFF'}")

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Screen resolution helper
# ---------------------------------------------------------------------------

def _get_screen_size() -> Tuple[int, int]:
    """Return (width, height) of the primary display, or (1920, 1080) fallback."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except (ImportError, Exception):
        return 1920, 1080


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Motion Combat Controller ===")
    print("Press Q to quit | D to toggle debug | S to toggle mouse control")

    config  = load_config()
    tracker = PoseTracker(config)

    if not tracker.open():
        print("[ERROR] Could not open webcam. Check camera index in config.json.")
        sys.exit(1)

    screen_w, screen_h = _get_screen_size()
    engine = GestureEngine(tracker, config)
    mapper = InputMapper(config, screen_w, screen_h)

    try:
        if THREADED:
            threaded_main(config, tracker, engine, mapper)
        else:
            single_threaded_main(config, tracker, engine, mapper)
    finally:
        mapper.release_all()
        tracker.release()
        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()

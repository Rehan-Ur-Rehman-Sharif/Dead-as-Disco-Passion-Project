"""
tests.py
--------
Headless unit tests for the gesture engine and input mapper.

No webcam, no X display, and no MediaPipe inference are required;
the PoseTracker is fully mocked.

Run with:
    python tests.py
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "config.json"
with open(_CONFIG_PATH) as _f:
    CONFIG = json.load(_f)

# ---------------------------------------------------------------------------
# Helpers – skeleton factory and mock tracker
# ---------------------------------------------------------------------------

from pose_tracker import Skeleton, Joint, PoseTracker


def make_skeleton(**norm_overrides) -> Skeleton:
    """Return a valid Skeleton with all joints at the origin unless overridden."""
    s = Skeleton(timestamp=time.monotonic())
    s.shoulder_width = 0.3
    s.neck_x = 0.5
    s.neck_y = 0.5
    for attr in (
        "norm_nose", "norm_left_shoulder", "norm_right_shoulder",
        "norm_left_elbow", "norm_right_elbow",
        "norm_left_wrist", "norm_right_wrist",
    ):
        setattr(s, attr, np.zeros(2, np.float32))
    s.left_shoulder  = Joint(0.4, 0.5, 1.0)
    s.right_shoulder = Joint(0.6, 0.5, 1.0)
    for attr, val in norm_overrides.items():
        setattr(s, attr, np.array(val, dtype=np.float32))
    s.valid = True
    return s


class MockTracker:
    """Minimal PoseTracker stub backed by a deque of Skeletons."""

    def __init__(self, skeletons):
        self.buffer = collections.deque(skeletons, maxlen=5)

    def latest(self) -> Skeleton:
        return self.buffer[-1] if self.buffer else None

    def velocity(self, joint_attr: str, frames_back: int = 1) -> np.ndarray:
        buf = list(self.buffer)
        if len(buf) < frames_back + 1:
            return np.zeros(2, np.float32)
        curr = getattr(buf[-1], joint_attr)
        prev = getattr(buf[-1 - frames_back], joint_attr)
        return (curr - prev).astype(np.float32)


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from gesture_engine import GestureEngine, GestureEvent, within_beat_window
from input_mapper import (
    InputMapper,
    _MOUSE_BTN_LEFT, _MOUSE_BTN_RIGHT,
    _KEY_SPACE, _KEY_SHIFT,
)

# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

def test_normalisation():
    """Skeleton._normalise computes neck, shoulder_width, and norm coords."""
    t = PoseTracker(CONFIG)
    s = Skeleton()
    s.left_shoulder  = Joint(0.3, 0.5, 1.0)
    s.right_shoulder = Joint(0.7, 0.5, 1.0)
    s.nose           = Joint(0.5, 0.3, 1.0)
    s.left_elbow     = Joint(0.25, 0.6, 1.0)
    s.right_elbow    = Joint(0.75, 0.6, 1.0)
    s.left_wrist     = Joint(0.2, 0.7, 1.0)
    s.right_wrist    = Joint(0.8, 0.7, 1.0)
    s = t._normalise(s)

    assert abs(s.neck_x - 0.5) < 1e-5, "neck_x should be midpoint"
    assert abs(s.neck_y - 0.5) < 1e-5, "neck_y should be midpoint"
    assert abs(s.shoulder_width - 0.4) < 1e-5, "shoulder_width = 0.4"
    np.testing.assert_allclose(s.norm_right_shoulder, [0.5, 0.0], atol=1e-5)


def test_attack():
    """Right wrist forward velocity fires ATTACK (LMB click)."""
    neutral = make_skeleton(norm_right_wrist=[0.0,  0.5])
    punch   = make_skeleton(norm_right_wrist=[-0.3, 0.5])
    tracker = MockTracker([neutral])
    engine  = GestureEngine(tracker, CONFIG)
    engine._neutral = True
    tracker.buffer.append(punch)
    assert GestureEvent.ATTACK in engine.update()


def test_dodge():
    """Head lateral velocity fires DODGE (Space)."""
    s1 = make_skeleton(norm_nose=[0.0, -1.0])
    s2 = make_skeleton(norm_nose=[0.4, -1.0])
    tracker = MockTracker([s1])
    engine  = GestureEngine(tracker, CONFIG)
    engine._neutral = True
    tracker.buffer.append(s2)
    assert GestureEvent.DODGE in engine.update()


def test_counter():
    """Both elbows closing inward fires COUNTER (RMB)."""
    s1 = make_skeleton(norm_left_elbow=[-0.4, 0.0], norm_right_elbow=[0.4, 0.0])
    s2 = make_skeleton(norm_left_elbow=[-0.1, 0.0], norm_right_elbow=[0.1, 0.0])
    tracker = MockTracker([s1])
    engine  = GestureEngine(tracker, CONFIG)
    engine._neutral = True
    tracker.buffer.append(s2)
    assert GestureEvent.COUNTER in engine.update()


def test_throw():
    """Right wrist forward+downward arc fires THROW (R)."""
    s1 = make_skeleton(norm_right_wrist=[0.0,   0.0])
    s2 = make_skeleton(norm_right_wrist=[-0.25, 0.15])
    tracker = MockTracker([s1])
    engine  = GestureEngine(tracker, CONFIG)
    engine._neutral = True
    tracker.buffer.append(s2)
    assert GestureEvent.THROW in engine.update()


def test_finisher():
    """Both wrists thrust forward fires FINISHER (F)."""
    s1 = make_skeleton(norm_left_wrist=[0.0,   0.3], norm_right_wrist=[0.0,   0.3])
    s2 = make_skeleton(norm_left_wrist=[-0.25, 0.3], norm_right_wrist=[-0.25, 0.3])
    tracker = MockTracker([s1])
    engine  = GestureEngine(tracker, CONFIG)
    engine._neutral = True
    tracker.buffer.append(s2)
    assert GestureEvent.FINISHER in engine.update()


def test_execute():
    """Both wrists pull inward fires EXECUTE (E)."""
    s1 = make_skeleton(norm_left_wrist=[-0.4, 0.3], norm_right_wrist=[0.4, 0.3])
    s2 = make_skeleton(norm_left_wrist=[-0.1, 0.3], norm_right_wrist=[0.1, 0.3])
    tracker = MockTracker([s1])
    engine  = GestureEngine(tracker, CONFIG)
    engine._neutral = True
    tracker.buffer.append(s2)
    assert GestureEvent.EXECUTE in engine.update()


def test_heavy_attack():
    """Right wrist pull-back → HEAVY_CHARGE; forward spike → HEAVY_RELEASE."""
    s0 = make_skeleton(norm_right_wrist=[0.0,  0.0])
    s1 = make_skeleton(norm_right_wrist=[0.3,  0.0])   # pull back (+x)
    s2 = make_skeleton(norm_right_wrist=[0.0,  0.0])
    s3 = make_skeleton(norm_right_wrist=[-0.3, 0.0])   # punch forward (-x)

    tracker = MockTracker([s0])
    engine  = GestureEngine(tracker, CONFIG)
    engine._neutral = True

    tracker.buffer.append(s1)
    e1 = engine.update()
    assert GestureEvent.HEAVY_CHARGE in e1
    assert engine.is_heavy_charging

    tracker.buffer.append(s2)
    tracker.buffer.append(s3)
    e2 = engine.update()
    assert GestureEvent.HEAVY_RELEASE in e2
    assert not engine.is_heavy_charging


def test_special_start_and_end():
    """Left raised above head + right forward → SPECIAL_START; deactivation → SPECIAL_END."""
    s_on  = make_skeleton(norm_left_wrist=[0.0, -1.2], norm_right_wrist=[-0.2, 0.0])
    s_off = make_skeleton(norm_left_wrist=[0.0,  0.3], norm_right_wrist=[0.0,  0.3])

    tracker = MockTracker([s_on])
    engine  = GestureEngine(tracker, CONFIG)

    e1 = engine.update()
    assert GestureEvent.SPECIAL_START in e1
    assert engine.is_special_active

    tracker.buffer.append(s_off)
    e2 = engine.update()
    assert GestureEvent.SPECIAL_END in e2
    assert not engine.is_special_active


def test_special_suppresses_attack():
    """While SPECIAL is active, ATTACK must not fire."""
    s_special = make_skeleton(
        norm_left_wrist=[0.0, -1.2], norm_right_wrist=[-0.2, 0.0])
    tracker = MockTracker([s_special])
    engine  = GestureEngine(tracker, CONFIG)
    engine.update()   # activates special
    assert engine.is_special_active

    punch = make_skeleton(
        norm_right_wrist=[-0.4, 0.0], norm_left_wrist=[0.0, -1.2])
    tracker.buffer.append(punch)
    engine._neutral = True
    events = engine.update()
    assert GestureEvent.ATTACK not in events


def test_cooldown():
    """A gesture cannot retrigger until its cooldown expires."""
    s1 = make_skeleton(norm_right_wrist=[0.0,  0.5])
    s2 = make_skeleton(norm_right_wrist=[-0.3, 0.5])
    tracker = MockTracker([s1])
    engine  = GestureEngine(tracker, CONFIG)
    engine._neutral = True
    tracker.buffer.append(s2)
    e1 = engine.update()
    assert GestureEvent.ATTACK in e1

    # Immediately attempt a second attack
    engine._neutral = True
    s3 = make_skeleton(norm_right_wrist=[0.0,  0.5])
    s4 = make_skeleton(norm_right_wrist=[-0.3, 0.5])
    tracker.buffer.append(s3)
    tracker.buffer.append(s4)
    e2 = engine.update()
    assert GestureEvent.ATTACK not in e2, "attack should be on cooldown"


def test_beat_awareness():
    """within_beat_window returns True on-beat and False off-beat."""
    # On beat (t=0 is exactly on beat)
    assert within_beat_window(0.0, 128, 80)

    # Off beat (half-beat away)
    half_beat = (60.0 / 128) / 2
    assert not within_beat_window(half_beat, 128, 80)

    # BPM=0 should always return False
    assert not within_beat_window(0.0, 0, 80)

    # Boundary: just inside the window (left edge)
    beat_interval = 60.0 / 128
    half_window   = (80 / 1000) / 2
    just_inside   = half_window * 0.99
    assert within_beat_window(just_inside, 128, 80)

    # Boundary: just outside the window
    just_outside = half_window * 1.01
    assert not within_beat_window(just_outside, 128, 80)


def test_input_mapper_dispatch():
    """InputMapper routes all GestureEvents to the correct input calls."""
    mapper = InputMapper(CONFIG, 1920, 1080)
    recorded = []

    class FakeMouse:
        def click(self, btn, n): recorded.append(("click", btn))
        def press(self, btn):    recorded.append(("press_mouse", btn))
        def release(self, btn):  recorded.append(("release_mouse", btn))
        position = (0, 0)

    class FakeKbd:
        def press(self, k):   recorded.append(("press_key", k))
        def release(self, k): recorded.append(("release_key", k))

    mapper.mouse    = FakeMouse()
    mapper.keyboard = FakeKbd()

    # ATTACK → LMB click
    mapper.handle_events([GestureEvent.ATTACK])
    assert any(r == ("click", _MOUSE_BTN_LEFT) for r in recorded)

    # DODGE → Space
    recorded.clear()
    mapper.handle_events([GestureEvent.DODGE])
    assert any(r[0] == "press_key" and r[1] == _KEY_SPACE for r in recorded)

    # COUNTER → RMB click
    recorded.clear()
    mapper.handle_events([GestureEvent.COUNTER])
    assert any(r == ("click", _MOUSE_BTN_RIGHT) for r in recorded)

    # SPECIAL_START → Shift hold + LMB hold
    recorded.clear()
    mapper.handle_events([GestureEvent.SPECIAL_START])
    assert mapper._special_held
    assert any(r[0] == "press_key"   and r[1] == _KEY_SHIFT       for r in recorded)
    assert any(r[0] == "press_mouse" and r[1] == _MOUSE_BTN_LEFT  for r in recorded)

    # SPECIAL_END → LMB release + Shift release
    recorded.clear()
    mapper.handle_events([GestureEvent.SPECIAL_END])
    assert not mapper._special_held
    assert any(r[0] == "release_mouse" and r[1] == _MOUSE_BTN_LEFT for r in recorded)
    assert any(r[0] == "release_key"   and r[1] == _KEY_SHIFT      for r in recorded)

    # Tap keys: Throw→R, Finisher→F, Execute→E
    for event, char in [
        (GestureEvent.THROW,    'r'),
        (GestureEvent.FINISHER, 'f'),
        (GestureEvent.EXECUTE,  'e'),
    ]:
        recorded.clear()
        mapper.handle_events([event])
        assert any(r[0] == "press_key" and r[1] == char for r in recorded), \
            f"{event} should emit press_key('{char}'), got {recorded}"


def test_mouse_smoothing():
    """update_mouse applies exponential smoothing and deadzone."""
    mapper = InputMapper(CONFIG, 1920, 1080)
    positions = []

    class FakeMouse:
        def click(self, b, n): pass
        @property
        def position(self): return (960, 540)
        @position.setter
        def position(self, v): positions.append(v)

    mapper.mouse = FakeMouse()

    # First call: no previous smoothed value, so output = target
    s = make_skeleton(norm_nose=[0.2, 0.0])
    mapper.update_mouse(s)
    assert len(positions) == 1
    expected_x = int(960 + 0.2 * CONFIG["mouse_sensitivity"])
    assert abs(positions[0][0] - expected_x) < 2

    # Deadzone: near-zero nose displacement → no update
    positions.clear()
    s2 = make_skeleton(norm_nose=[0.01, 0.02])   # both within deadzone
    mapper.update_mouse(s2)
    assert len(positions) == 0, "deadzone should suppress movement"


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

_TESTS = [
    test_normalisation,
    test_attack,
    test_dodge,
    test_counter,
    test_throw,
    test_finisher,
    test_execute,
    test_heavy_attack,
    test_special_start_and_end,
    test_special_suppresses_attack,
    test_cooldown,
    test_beat_awareness,
    test_input_mapper_dispatch,
    test_mouse_smoothing,
]


def main() -> int:
    passed = 0
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

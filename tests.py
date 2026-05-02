"""
tests.py
--------
Headless unit tests for the pose-based gesture engine and input mapper.

No webcam, no X display, and no MediaPipe inference are required;
PoseTracker is fully mocked, and all gesture detection uses only static
skeleton geometry (no velocity or motion tracking).

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

# Ensure confirmation window is 0 so tests fire on the first matching frame
# without needing to sleep for real time to elapse.
CONFIG.setdefault("timing", {})
CONFIG["timing"]["pose_confirmation_ms"] = 0

# ---------------------------------------------------------------------------
# Helpers – skeleton factory and mock tracker
# ---------------------------------------------------------------------------

from pose_tracker import Skeleton, Joint, PoseTracker


def make_skeleton(**norm_overrides) -> Skeleton:
    """
    Return a valid Skeleton with sensible default joint positions.

    Default pose: arms hanging straight down (NOT classified as extended
    because the wrist drops more than arm_extension_max_drop below the
    shoulder).  Individual norm_* attributes can be overridden via kwargs.
    """
    s = Skeleton(timestamp=time.monotonic())
    s.shoulder_width = 0.2
    s.neck_x = 0.5
    s.neck_y = 0.5
    s.nose            = Joint(0.5, 0.5, 1.0)
    s.left_shoulder   = Joint(0.4, 0.5, 1.0)
    s.right_shoulder  = Joint(0.6, 0.5, 1.0)
    s.left_elbow      = Joint(0.35, 0.65, 1.0)
    s.right_elbow     = Joint(0.65, 0.65, 1.0)
    s.left_wrist      = Joint(0.3,  0.8,  1.0)
    s.right_wrist     = Joint(0.7,  0.8,  1.0)

    # Normalised positions (relative to neck, scaled by shoulder_width).
    # Arms hang straight down: elbow angle ≈ 180° but wrist drops > 0.5 SW
    # below shoulder → is_arm_extended returns False.
    s.norm_nose            = np.array([ 0.0, -0.3], np.float32)
    s.norm_left_shoulder   = np.array([-0.5,  0.0], np.float32)
    s.norm_right_shoulder  = np.array([ 0.5,  0.0], np.float32)
    s.norm_left_elbow      = np.array([-0.5,  0.5], np.float32)
    s.norm_right_elbow     = np.array([ 0.5,  0.5], np.float32)
    s.norm_left_wrist      = np.array([-0.5,  1.0], np.float32)
    s.norm_right_wrist     = np.array([ 0.5,  1.0], np.float32)
    s.valid = True

    for attr, val in norm_overrides.items():
        setattr(s, attr, np.array(val, dtype=np.float32))
    return s


# Extended-arm position for the right arm (angle ≈ 160°, dist ≈ 0.84, drop = 0.25)
_RIGHT_EXT = dict(
    norm_right_shoulder=[ 0.5,  0.0],
    norm_right_elbow   =[ 0.1,  0.2],
    norm_right_wrist   =[-0.3,  0.25],
)

# Extended-arm position for the left arm (mirror of right)
_LEFT_EXT = dict(
    norm_left_shoulder =[-0.5,  0.0],
    norm_left_elbow    =[-0.1,  0.2],
    norm_left_wrist    =[ 0.3,  0.25],
)

# Retracted-arm position for the right arm (angle ≈ 28°, dist ≈ 0.25)
_RIGHT_RET = dict(
    norm_right_shoulder=[ 0.5,  0.0],
    norm_right_elbow   =[ 0.4,  0.4],
    norm_right_wrist   =[ 0.35, 0.2],
)


class MockTracker:
    """Minimal PoseTracker stub backed by a deque of Skeletons."""

    def __init__(self, skeletons):
        self.buffer = collections.deque(skeletons, maxlen=5)

    def latest(self) -> Skeleton:
        return self.buffer[-1] if self.buffer else None


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from gesture_engine import (
    GestureEngine, GestureEvent, PoseFeatures,
    arm_angle, is_arm_extended, is_arm_retracted, is_above_head, distance,
    within_beat_window,
)
from input_mapper import (
    InputMapper,
    _MOUSE_BTN_LEFT, _MOUSE_BTN_RIGHT,
    _KEY_SPACE, _KEY_SHIFT,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_engine(skeletons, cfg=None):
    tracker = MockTracker(skeletons)
    return GestureEngine(tracker, cfg or CONFIG)


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

def test_normalisation():
    """PoseTracker._normalise computes neck, shoulder_width, and norm coords."""
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


def test_pose_helpers():
    """arm_angle, is_arm_extended, is_arm_retracted, is_above_head, distance."""
    # arm_angle: straight arm should give ≈ 180°
    sh = np.array([0.5, 0.0], np.float32)
    el = np.array([0.5, 0.5], np.float32)
    wr = np.array([0.5, 1.0], np.float32)
    assert abs(arm_angle(sh, el, wr) - 180.0) < 1e-3

    # is_arm_extended: punch position
    sh2 = np.array([ 0.5,  0.0], np.float32)
    el2 = np.array([ 0.1,  0.2], np.float32)
    wr2 = np.array([-0.3,  0.25], np.float32)
    assert is_arm_extended(sh2, el2, wr2, extension_ratio=0.75,
                           extension_angle=150.0, max_drop=0.5), \
        "punch position should be extended"

    # hanging arm: angle = 180° but wrist drops 1.0 SW > max_drop 0.5
    assert not is_arm_extended(sh, el, wr, extension_ratio=0.75,
                               extension_angle=150.0, max_drop=0.5), \
        "hanging arm should NOT be extended (max_drop exceeded)"

    # is_arm_retracted: wrist near shoulder with bent elbow
    sh3 = np.array([0.5, 0.0], np.float32)
    el3 = np.array([0.4, 0.4], np.float32)
    wr3 = np.array([0.35, 0.2], np.float32)
    assert is_arm_retracted(sh3, el3, wr3, retracted_ratio=0.45,
                            retracted_angle=110.0), \
        "wrist near shoulder with bent elbow should be retracted"

    # is_above_head: wrist y < nose y
    wrist_up   = np.array([0.0, -1.5], np.float32)
    nose_level = np.array([0.0, -0.3], np.float32)
    assert is_above_head(wrist_up, nose_level, threshold=0.1)
    assert not is_above_head(nose_level, wrist_up, threshold=0.1)

    # distance
    a = np.array([0.0, 0.0], np.float32)
    b = np.array([3.0, 4.0], np.float32)
    assert abs(distance(a, b) - 5.0) < 1e-5


def test_attack():
    """Right arm fully extended fires ATTACK (LMB click)."""
    s = make_skeleton(**_RIGHT_EXT)
    engine = _make_engine([s])
    events = engine.update()
    assert GestureEvent.ATTACK in events, f"Expected ATTACK, got {events}"


def test_attack_hanging_arm_no_trigger():
    """Arm hanging straight down must NOT fire ATTACK."""
    s = make_skeleton()   # default: arms hanging > max_drop below shoulder
    engine = _make_engine([s])
    events = engine.update()
    assert GestureEvent.ATTACK not in events, \
        "Hanging arm should not trigger ATTACK"


def test_dodge():
    """Head exiting the deadzone rectangle fires DODGE (Space)."""
    # First call: head at deadzone centre → sets initial state
    s_in  = make_skeleton()
    s_in.nose = Joint(0.5, 0.5, 1.0)   # centre of deadzone (default 0.5,0.5)
    engine = _make_engine([s_in])
    engine.update()   # initialise _in_deadzone_prev

    # Second call: head moves far left (outside deadzone)
    s_out = make_skeleton()
    s_out.nose = Joint(0.1, 0.5, 1.0)
    engine.tracker.buffer.append(s_out)
    events = engine.update()
    assert GestureEvent.DODGE in events, f"Expected DODGE, got {events}"


def test_dodge_no_trigger_while_inside():
    """No DODGE while head remains inside the deadzone."""
    s = make_skeleton()
    s.nose = Joint(0.5, 0.5, 1.0)
    engine = _make_engine([s])
    engine.update()
    engine.tracker.buffer.append(make_skeleton())
    events = engine.update()
    assert GestureEvent.DODGE not in events


def test_counter():
    """Both hands above head fires COUNTER (RMB)."""
    s = make_skeleton(
        norm_left_wrist  =[-0.3, -1.5],
        norm_right_wrist =[ 0.3, -1.5],
        norm_nose        =[ 0.0, -0.3],
    )
    engine = _make_engine([s])
    events = engine.update()
    assert GestureEvent.COUNTER in events, f"Expected COUNTER, got {events}"


def test_throw():
    """Both arms simultaneously extended fires THROW (E) but NOT ATTACK."""
    s = make_skeleton(**_RIGHT_EXT, **_LEFT_EXT)
    engine = _make_engine([s])
    events = engine.update()
    assert GestureEvent.THROW in events, f"Expected THROW, got {events}"
    assert GestureEvent.ATTACK not in events, "ATTACK must not fire when THROW fires"


def test_heavy_attack():
    """Arm retracted → HEAVY_CHARGE; same arm extended → HEAVY_RELEASE."""
    s_ret = make_skeleton(**_RIGHT_RET)
    engine = _make_engine([s_ret])

    e1 = engine.update()
    assert GestureEvent.HEAVY_CHARGE in e1, f"Expected HEAVY_CHARGE, got {e1}"
    assert engine.is_heavy_charging
    assert engine._heavy_charge_arm == "right"

    s_ext = make_skeleton(**_RIGHT_EXT)
    engine.tracker.buffer.append(s_ext)
    e2 = engine.update()
    assert GestureEvent.HEAVY_RELEASE in e2, f"Expected HEAVY_RELEASE, got {e2}"
    assert not engine.is_heavy_charging


def test_special_start_and_end():
    """Left hand above head + right arm extended → SPECIAL_START; pose breaks → SPECIAL_END."""
    s_on = make_skeleton(
        norm_left_wrist =[-0.3, -1.5],   # above head
        **_RIGHT_EXT,
    )
    engine = _make_engine([s_on])

    e1 = engine.update()
    assert GestureEvent.SPECIAL_START in e1, f"Expected SPECIAL_START, got {e1}"
    assert engine.is_special_active

    s_off = make_skeleton()   # default: arms hanging, not above head
    engine.tracker.buffer.append(s_off)
    e2 = engine.update()
    assert GestureEvent.SPECIAL_END in e2, f"Expected SPECIAL_END, got {e2}"
    assert not engine.is_special_active


def test_special_suppresses_attack():
    """While SPECIAL is active, ATTACK must not fire."""
    s_special = make_skeleton(
        norm_left_wrist=[-0.3, -1.5],
        **_RIGHT_EXT,
    )
    engine = _make_engine([s_special])
    engine.update()   # activates special
    assert engine.is_special_active

    # Right arm is still extended (same pose) – normally would be ATTACK
    engine.tracker.buffer.append(s_special)
    events = engine.update()
    assert GestureEvent.ATTACK not in events, \
        "ATTACK must be suppressed while SPECIAL is active"


def test_cooldown():
    """A gesture cannot re-trigger until its cooldown expires."""
    s_ext = make_skeleton(**_RIGHT_EXT)
    engine = _make_engine([s_ext])

    e1 = engine.update()
    assert GestureEvent.ATTACK in e1, "First ATTACK should fire"

    # Arm must come down for confirmation to reset
    s_rest = make_skeleton()
    engine.tracker.buffer.append(s_rest)
    engine.update()

    # Re-extend immediately (cooldown still active at 75 ms)
    engine.tracker.buffer.append(make_skeleton(**_RIGHT_EXT))
    e2 = engine.update()
    assert GestureEvent.ATTACK not in e2, "ATTACK should be on cooldown"


def test_beat_awareness():
    """within_beat_window returns True on-beat and False off-beat."""
    assert within_beat_window(0.0, 128, 80)

    half_beat = (60.0 / 128) / 2
    assert not within_beat_window(half_beat, 128, 80)

    assert not within_beat_window(0.0, 0, 80)

    beat_interval = 60.0 / 128
    half_window   = (80 / 1000) / 2
    assert within_beat_window(half_window * 0.99, 128, 80)
    assert not within_beat_window(half_window * 1.01, 128, 80)


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

    # THROW → E
    recorded.clear()
    mapper.handle_events([GestureEvent.THROW])
    assert any(r[0] == "press_key" and r[1] == 'e' for r in recorded), \
        f"THROW should emit press_key('e'), got {recorded}"

    # SPECIAL_START → Shift hold + LMB hold
    recorded.clear()
    mapper.handle_events([GestureEvent.SPECIAL_START])
    assert mapper._special_held
    assert any(r[0] == "press_key"   and r[1] == _KEY_SHIFT      for r in recorded)
    assert any(r[0] == "press_mouse" and r[1] == _MOUSE_BTN_LEFT for r in recorded)

    # SPECIAL_END → LMB release + Shift release
    recorded.clear()
    mapper.handle_events([GestureEvent.SPECIAL_END])
    assert not mapper._special_held
    assert any(r[0] == "release_mouse" and r[1] == _MOUSE_BTN_LEFT for r in recorded)
    assert any(r[0] == "release_key"   and r[1] == _KEY_SHIFT      for r in recorded)


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

    s = make_skeleton(norm_nose=[0.2, 0.0])
    mapper.update_mouse(s)
    assert len(positions) == 1
    expected_x = int(960 + 0.2 * CONFIG["mouse_sensitivity"])
    assert abs(positions[0][0] - expected_x) < 2

    # Deadzone: near-zero nose displacement → no update
    positions.clear()
    s2 = make_skeleton(norm_nose=[0.01, 0.02])
    mapper.update_mouse(s2)
    assert len(positions) == 0, "deadzone should suppress movement"


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

_TESTS = [
    test_normalisation,
    test_pose_helpers,
    test_attack,
    test_attack_hanging_arm_no_trigger,
    test_dodge,
    test_dodge_no_trigger_while_inside,
    test_counter,
    test_throw,
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
            import traceback
            print(f"  FAIL  {fn.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

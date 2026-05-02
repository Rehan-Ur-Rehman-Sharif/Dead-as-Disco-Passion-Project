"""
gesture_engine.py
-----------------
Pure pose-based gesture detection engine.

Classifies gestures using only static skeleton geometry — joint positions,
inter-joint distances, and limb angles.  No velocity or frame-to-frame motion
tracking is used anywhere in this module.

Gesture → Input mapping
-----------------------
    attack        → LMB tap      (one arm fully extended)
    heavy_attack  → LMB hold / release (arm retract → extend pose transition)
    throw         → E tap        (both arms extended simultaneously)
    counter       → RMB tap      (both hands above head level)
    dodge         → Space tap    (head exits configurable deadzone rectangle)
    special       → Shift + LMB  (one hand above head, other arm extended)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np

from pose_tracker import PoseTracker, Skeleton


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class GestureEvent(Enum):
    ATTACK         = auto()   # LMB tap
    HEAVY_CHARGE   = auto()   # begin holding LMB
    HEAVY_RELEASE  = auto()   # release LMB
    COUNTER        = auto()   # RMB tap
    DODGE          = auto()   # Space tap
    THROW          = auto()   # E tap
    SPECIAL_START  = auto()   # begin Shift + LMB hold
    SPECIAL_END    = auto()   # release Shift + LMB


# ---------------------------------------------------------------------------
# Pose geometry helpers
# ---------------------------------------------------------------------------

def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(a - b))


def arm_angle(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """
    Return the angle at the elbow joint in degrees (0–180).

    Computed as the angle between the upper-arm vector (elbow→shoulder) and
    the forearm vector (elbow→wrist).
    """
    v1 = shoulder - elbow
    v2 = wrist - elbow
    denom = float(np.linalg.norm(v1)) * float(np.linalg.norm(v2))
    if denom < 1e-8:
        return 0.0
    cos_a = float(np.dot(v1, v2) / denom)
    cos_a = max(-1.0, min(1.0, cos_a))
    return float(np.degrees(np.arccos(cos_a)))


def is_arm_extended(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    extension_ratio: float = 0.75,
    extension_angle: float = 150.0,
    max_drop: float = 0.5,
) -> bool:
    """
    Return True if the arm is fully extended (e.g. a punch or reach).

    Criteria
    --------
    * Wrist-to-shoulder distance > *extension_ratio* (normalised units)
    * Elbow angle > *extension_angle* degrees
    * Wrist is not more than *max_drop* normalised units below the shoulder
      (prevents a straight-hanging arm from being classified as extended)
    """
    return (
        distance(wrist, shoulder) > extension_ratio
        and arm_angle(shoulder, elbow, wrist) > extension_angle
        and float(wrist[1]) < float(shoulder[1]) + max_drop
    )


def is_arm_retracted(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    retracted_ratio: float = 0.45,
    retracted_angle: float = 110.0,
) -> bool:
    """
    Return True if the arm is pulled close to the torso (charge/guard pose).

    Criteria
    --------
    * Wrist-to-shoulder distance < *retracted_ratio* (normalised units)
    * Elbow angle < *retracted_angle* degrees
    """
    return (
        distance(wrist, shoulder) < retracted_ratio
        and arm_angle(shoulder, elbow, wrist) < retracted_angle
    )


def is_above_head(
    wrist: np.ndarray,
    nose: np.ndarray,
    threshold: float = 0.0,
) -> bool:
    """
    Return True if *wrist* is above *nose* by at least *threshold* normalised units.

    In the normalised skeleton y increases downward, so a smaller (more negative)
    y value means higher in the frame.
    """
    return float(wrist[1]) < float(nose[1]) - threshold


# ---------------------------------------------------------------------------
# Per-frame pose feature snapshot
# ---------------------------------------------------------------------------

@dataclass
class PoseFeatures:
    """Static pose features derived from a single skeleton frame."""

    timestamp: float = 0.0

    left_arm_extended:  bool = False
    right_arm_extended: bool = False
    left_arm_retracted:  bool = False
    right_arm_retracted: bool = False
    left_hand_above_head:  bool = False
    right_hand_above_head: bool = False

    left_elbow_angle:  float = 0.0
    right_elbow_angle: float = 0.0

    # Raw MediaPipe 0–1 nose position for deadzone comparison
    nose_x: float = 0.5
    nose_y: float = 0.5

    @property
    def both_arms_extended(self) -> bool:
        return self.left_arm_extended and self.right_arm_extended

    @property
    def both_hands_above_head(self) -> bool:
        return self.left_hand_above_head and self.right_hand_above_head

    @property
    def special_condition(self) -> bool:
        """One hand above head AND the opposite arm extended."""
        return (
            (self.left_hand_above_head  and self.right_arm_extended) or
            (self.right_hand_above_head and self.left_arm_extended)
        )


# ---------------------------------------------------------------------------
# Confirmation-window tracker
# ---------------------------------------------------------------------------

class _ConfirmationTracker:
    """
    Emits a single True once a boolean condition has been continuously met
    for *confirmation_ms* milliseconds.  Resets automatically when the
    condition becomes False.

    If *confirmation_ms* is 0 the event fires on the very first True call.
    """

    def __init__(self, confirmation_ms: float) -> None:
        self._confirmation_ms = confirmation_ms
        self._first_seen: Optional[float] = None
        self._fired: bool = False

    def update(self, condition: bool) -> bool:
        if not condition:
            self._first_seen = None
            self._fired = False
            return False
        if self._first_seen is None:
            self._first_seen = time.monotonic()
        if not self._fired:
            elapsed_ms = (time.monotonic() - self._first_seen) * 1000.0
            if elapsed_ms >= self._confirmation_ms:
                self._fired = True
                return True
        return False

    def reset(self) -> None:
        self._first_seen = None
        self._fired = False

    @property
    def active(self) -> bool:
        """True while the condition has been seen but not yet confirmed."""
        return self._first_seen is not None and not self._fired


# ---------------------------------------------------------------------------
# Per-gesture debounce state
# ---------------------------------------------------------------------------

@dataclass
class GestureState:
    """Tracks per-gesture cooldown to debounce rapid re-triggering."""

    name: str
    cooldown_ms: float = 150.0
    _last_triggered: float = field(default=0.0, repr=False)

    def can_trigger(self) -> bool:
        return (time.monotonic() - self._last_triggered) * 1000 >= self.cooldown_ms

    def mark_triggered(self) -> None:
        self._last_triggered = time.monotonic()

    def remaining_ms(self) -> float:
        elapsed = (time.monotonic() - self._last_triggered) * 1000
        return max(0.0, self.cooldown_ms - elapsed)


# ---------------------------------------------------------------------------
# Beat-awareness helper (optional, retained for rhythm annotation)
# ---------------------------------------------------------------------------

def within_beat_window(gesture_time: float, bpm: float, window_ms: float) -> bool:
    """Return True if *gesture_time* falls within a beat-window for *bpm*."""
    if bpm <= 0:
        return False
    beat_interval = 60.0 / bpm
    phase = gesture_time % beat_interval
    half_window = (window_ms / 1000.0) / 2.0
    return phase <= half_window or phase >= (beat_interval - half_window)


# ---------------------------------------------------------------------------
# GestureEngine
# ---------------------------------------------------------------------------

class GestureEngine:
    """
    Processes each frame's skeleton and emits GestureEvents using only
    static pose geometry (no velocity or motion tracking).

    Usage
    -----
    engine = GestureEngine(tracker, config)
    events = engine.update()   # call once per frame; returns list[GestureEvent]
    """

    def __init__(self, tracker: PoseTracker, config: dict) -> None:
        self.tracker = tracker
        self.cfg = config

        # Pose geometry thresholds
        pt = config.get("pose_thresholds", {})
        self._ext_ratio  = float(pt.get("arm_extension_ratio",  0.75))
        self._ret_ratio  = float(pt.get("arm_retracted_ratio",  0.45))
        self._ext_angle  = float(pt.get("arm_extension_angle",  150.0))
        self._ret_angle  = float(pt.get("arm_retracted_angle",  110.0))
        self._max_drop   = float(pt.get("arm_extension_max_drop", 0.5))
        self._head_th    = float(pt.get("head_above_threshold",  0.1))

        # Timing
        timing   = config.get("timing", {})
        conf_ms  = float(timing.get("pose_confirmation_ms", 0))
        debounce = float(timing.get("debounce_ms", 150))

        # Confirmation trackers (fire once after pose is held for conf_ms)
        self._conf_attack  = _ConfirmationTracker(conf_ms)
        self._conf_throw   = _ConfirmationTracker(conf_ms)
        self._conf_counter = _ConfirmationTracker(conf_ms)
        self._conf_heavy   = _ConfirmationTracker(conf_ms)

        # Per-gesture debounce
        cd = config.get("cooldowns", {})
        self.states: Dict[str, GestureState] = {
            "attack":  GestureState("attack",  cd.get("attack",         debounce)),
            "throw":   GestureState("throw",   cd.get("throw",          debounce)),
            "counter": GestureState("counter", cd.get("counter",        debounce)),
            "dodge":   GestureState("dodge",   cd.get("dodge",          debounce)),
            "special": GestureState("special", cd.get("special_reentry", 200)),
        }

        # Stateful flags
        self._special_active:    bool = False
        self._heavy_charging:    bool = False
        self._heavy_charge_arm:  Optional[str] = None   # "left" | "right"
        self._heavy_candidate:   Optional[str] = None   # arm being confirmed

        # Dodge: fire on the in→out transition of the deadzone
        self._in_deadzone_prev: Optional[bool] = None

        # BPM / beat awareness
        self.bpm:            float = float(config.get("bpm", 0))
        self.beat_window_ms: float = float(config.get("beat_window_ms", 80))

        # Debug state
        self.last_events:   List[GestureEvent] = []
        self.last_timing:   str = ""
        self.last_features: Optional[PoseFeatures] = None

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self) -> List[GestureEvent]:
        """Evaluate the latest skeleton pose and return any triggered events."""
        skeleton = self.tracker.latest()
        if skeleton is None or not skeleton.valid:
            return []

        features = self._evaluate_pose(skeleton)
        self.last_features = features
        events: List[GestureEvent] = []

        # Special is checked first (highest priority, stateful)
        events.extend(self._check_special(features))

        if not self._special_active:
            # Throw must be checked before Attack so that both-arm extension
            # does not also fire a single-arm Attack.
            events.extend(self._check_throw(features))
            events.extend(self._check_heavy_attack(features))
            if not features.both_arms_extended:
                events.extend(self._check_attack(features))
            events.extend(self._check_counter(features))

        events.extend(self._check_dodge(features))

        # Beat-awareness annotation
        if events and self.bpm > 0:
            if within_beat_window(skeleton.timestamp, self.bpm, self.beat_window_ms):
                self.last_timing = "perfect"
            else:
                self.last_timing = "normal"
        else:
            self.last_timing = ""

        self.last_events = events
        return events

    # ------------------------------------------------------------------
    # Pose evaluation layer
    # ------------------------------------------------------------------

    def _evaluate_pose(self, s: Skeleton) -> PoseFeatures:
        """Compute all static pose features from the current skeleton frame."""
        f = PoseFeatures(timestamp=s.timestamp)

        f.left_arm_extended = is_arm_extended(
            s.norm_left_shoulder, s.norm_left_elbow, s.norm_left_wrist,
            self._ext_ratio, self._ext_angle, self._max_drop)
        f.right_arm_extended = is_arm_extended(
            s.norm_right_shoulder, s.norm_right_elbow, s.norm_right_wrist,
            self._ext_ratio, self._ext_angle, self._max_drop)

        f.left_arm_retracted = is_arm_retracted(
            s.norm_left_shoulder, s.norm_left_elbow, s.norm_left_wrist,
            self._ret_ratio, self._ret_angle)
        f.right_arm_retracted = is_arm_retracted(
            s.norm_right_shoulder, s.norm_right_elbow, s.norm_right_wrist,
            self._ret_ratio, self._ret_angle)

        f.left_hand_above_head  = is_above_head(
            s.norm_left_wrist,  s.norm_nose, self._head_th)
        f.right_hand_above_head = is_above_head(
            s.norm_right_wrist, s.norm_nose, self._head_th)

        f.left_elbow_angle  = arm_angle(
            s.norm_left_shoulder,  s.norm_left_elbow,  s.norm_left_wrist)
        f.right_elbow_angle = arm_angle(
            s.norm_right_shoulder, s.norm_right_elbow, s.norm_right_wrist)

        # Raw MediaPipe 0-1 normalised nose position (for deadzone)
        f.nose_x = float(s.nose.x)
        f.nose_y = float(s.nose.y)

        return f

    # ------------------------------------------------------------------
    # Gesture checks
    # ------------------------------------------------------------------

    def _check_attack(self, f: PoseFeatures) -> List[GestureEvent]:
        """
        ATTACK (LMB tap): one arm fully extended.
        Fires once per extension after the confirmation window.
        """
        condition = f.left_arm_extended or f.right_arm_extended
        confirmed = self._conf_attack.update(condition)
        if confirmed and self.states["attack"].can_trigger():
            self.states["attack"].mark_triggered()
            return [GestureEvent.ATTACK]
        return []

    def _check_heavy_attack(self, f: PoseFeatures) -> List[GestureEvent]:
        """
        HEAVY_ATTACK:
          Retracted arm pose → HEAVY_CHARGE (press & hold LMB).
          Same arm then extended  → HEAVY_RELEASE (release LMB).
        The arm identity is locked when HEAVY_CHARGE fires.
        """
        events: List[GestureEvent] = []

        if not self._heavy_charging:
            # Determine which arm (if any) is being retracted
            if f.right_arm_retracted:
                candidate = "right"
            elif f.left_arm_retracted:
                candidate = "left"
            else:
                candidate = None

            # Reset confirmation if the candidate arm changes
            if candidate != self._heavy_candidate:
                self._conf_heavy.reset()
                self._heavy_candidate = candidate

            if candidate and self._conf_heavy.update(True):
                self._heavy_charging   = True
                self._heavy_charge_arm = candidate
                self._heavy_candidate  = None
                self._conf_heavy.reset()
                events.append(GestureEvent.HEAVY_CHARGE)
            elif not candidate:
                self._conf_heavy.update(False)

        else:
            # Waiting for the charged arm to become fully extended
            arm_now_extended = (
                (self._heavy_charge_arm == "left"  and f.left_arm_extended) or
                (self._heavy_charge_arm == "right" and f.right_arm_extended)
            )
            if arm_now_extended:
                self._heavy_charging   = False
                self._heavy_charge_arm = None
                events.append(GestureEvent.HEAVY_RELEASE)

        return events

    def _check_throw(self, f: PoseFeatures) -> List[GestureEvent]:
        """
        THROW (E tap): both arms fully extended simultaneously.
        """
        confirmed = self._conf_throw.update(f.both_arms_extended)
        if confirmed and self.states["throw"].can_trigger():
            self.states["throw"].mark_triggered()
            return [GestureEvent.THROW]
        return []

    def _check_counter(self, f: PoseFeatures) -> List[GestureEvent]:
        """
        COUNTER (RMB tap): both hands raised above head level.
        """
        confirmed = self._conf_counter.update(f.both_hands_above_head)
        if confirmed and self.states["counter"].can_trigger():
            self.states["counter"].mark_triggered()
            return [GestureEvent.COUNTER]
        return []

    def _check_dodge(self, f: PoseFeatures) -> List[GestureEvent]:
        """
        DODGE (Space): head exits the deadzone rectangle (edge-triggered).
        Fires on the in → out transition; does NOT require a hold duration.
        """
        state  = self.states["dodge"]
        in_dz  = self._in_deadzone(f.nose_x, f.nose_y)
        events: List[GestureEvent] = []

        # First frame: just record the initial deadzone state
        if self._in_deadzone_prev is None:
            self._in_deadzone_prev = in_dz
            return []

        # Transition: was inside, now outside
        if self._in_deadzone_prev and not in_dz and state.can_trigger():
            state.mark_triggered()
            events.append(GestureEvent.DODGE)

        self._in_deadzone_prev = in_dz
        return events

    def _in_deadzone(self, nx: float, ny: float) -> bool:
        """Return True if the nose (MediaPipe 0–1) is inside the deadzone rect."""
        dz     = self.cfg.get("deadzone", {})
        cx     = float(dz.get("x",      0.5))
        cy     = float(dz.get("y",      0.5))
        half_w = float(dz.get("width",  0.3)) / 2.0
        half_h = float(dz.get("height", 0.3)) / 2.0
        return (cx - half_w <= nx <= cx + half_w and
                cy - half_h <= ny <= cy + half_h)

    def _check_special(self, f: PoseFeatures) -> List[GestureEvent]:
        """
        SPECIAL (Shift + LMB hold): one hand above head AND other arm extended.
        Stateful: SPECIAL_START on entry, SPECIAL_END on exit.
        Sustained as long as the pose condition holds.
        """
        condition = f.special_condition
        events: List[GestureEvent] = []

        if condition and not self._special_active:
            if self.states["special"].can_trigger():
                self._special_active = True
                events.append(GestureEvent.SPECIAL_START)

        elif not condition and self._special_active:
            self._special_active = False
            self.states["special"].mark_triggered()
            events.append(GestureEvent.SPECIAL_END)

        return events

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def cooldown_display(self) -> Dict[str, float]:
        """Return remaining cooldown (ms) for each gesture (for the overlay)."""
        return {name: state.remaining_ms() for name, state in self.states.items()}

    @property
    def is_special_active(self) -> bool:
        return self._special_active

    @property
    def is_heavy_charging(self) -> bool:
        return self._heavy_charging

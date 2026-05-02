"""
gesture_engine.py
-----------------
Event-driven gesture detection engine.

Reads the latest skeleton + velocity data from a PoseTracker and fires
named gesture events when conditions are met.  Manages per-gesture cooldowns,
a neutral-state gate, and conflict-resolution priority.

Gesture → Input mapping
-----------------------
    attack        → LMB tap
    heavy_attack  → LMB hold (charge) / release
    counter       → RMB tap
    dodge         → Space
    throw         → R
    finisher      → F
    execute       → E
    special       → Shift hold + LMB hold (stateful)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

import numpy as np

from pose_tracker import PoseTracker, Skeleton


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class GestureEvent(Enum):
    ATTACK         = auto()
    HEAVY_CHARGE   = auto()   # begin holding LMB
    HEAVY_RELEASE  = auto()   # release LMB
    COUNTER        = auto()
    DODGE          = auto()
    THROW          = auto()
    FINISHER       = auto()
    EXECUTE        = auto()
    SPECIAL_START  = auto()   # begin Shift + LMB hold
    SPECIAL_END    = auto()   # release Shift + LMB


# ---------------------------------------------------------------------------
# State / priority
# ---------------------------------------------------------------------------

PRIORITY: List[GestureEvent] = [
    GestureEvent.SPECIAL_START,
    GestureEvent.FINISHER,
    GestureEvent.EXECUTE,
    GestureEvent.HEAVY_CHARGE,
    GestureEvent.ATTACK,
    GestureEvent.COUNTER,
    GestureEvent.DODGE,
]


@dataclass
class GestureState:
    """Tracks whether a gesture was just triggered and its cooldown."""
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
# Beat-awareness helper
# ---------------------------------------------------------------------------

def within_beat_window(gesture_time: float, bpm: float, window_ms: float) -> bool:
    """Return True if *gesture_time* falls within a beat-window for *bpm*."""
    if bpm <= 0:
        return False
    beat_interval = 60.0 / bpm          # seconds
    phase = gesture_time % beat_interval
    half_window = (window_ms / 1000.0) / 2.0
    return phase <= half_window or phase >= (beat_interval - half_window)


# ---------------------------------------------------------------------------
# GestureEngine
# ---------------------------------------------------------------------------

class GestureEngine:
    """
    Consumes PoseTracker output each frame and emits GestureEvents.

    Usage
    -----
    engine = GestureEngine(tracker, config)
    events = engine.update()   # call once per frame; returns list[GestureEvent]
    """

    def __init__(self, tracker: PoseTracker, config: dict) -> None:
        self.tracker = tracker
        self.cfg = config

        cd = config.get("cooldowns", {})
        self.states: Dict[str, GestureState] = {
            "attack":       GestureState("attack",       cd.get("attack", 150)),
            "heavy_attack": GestureState("heavy_attack", cd.get("heavy_attack", 300)),
            "counter":      GestureState("counter",      cd.get("counter", 200)),
            "dodge":        GestureState("dodge",        cd.get("dodge", 250)),
            "throw":        GestureState("throw",        cd.get("throw", 200)),
            "finisher":     GestureState("finisher",     cd.get("finisher", 350)),
            "execute":      GestureState("execute",      cd.get("execute", 300)),
            "special":      GestureState("special",      cd.get("special_reentry", 200)),
        }

        # Stateful flags
        self._special_active:       bool = False
        self._heavy_charging:       bool = False
        self._neutral:              bool = True

        # BPM / beat awareness
        self.bpm: float              = float(config.get("bpm", 0))
        self.beat_window_ms: float   = float(config.get("beat_window_ms", 80))

        # Most recently emitted events (for debug overlay)
        self.last_events: List[GestureEvent] = []
        self.last_timing: str = ""   # "perfect" | "normal" | ""

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self) -> List[GestureEvent]:
        """Process the latest skeleton and return any triggered events."""
        skeleton = self.tracker.latest()
        if skeleton is None or not skeleton.valid:
            return []

        events: List[GestureEvent] = []

        # --- Special (stateful, highest priority) ---
        special_events = self._check_special(skeleton)
        events.extend(special_events)

        # While special is active, suppress attack-class gestures
        if not self._special_active:
            events.extend(self._check_finisher(skeleton))
            events.extend(self._check_execute(skeleton))
            events.extend(self._check_heavy_attack(skeleton))
            events.extend(self._check_attack(skeleton))
            events.extend(self._check_counter(skeleton))

        events.extend(self._check_dodge(skeleton))
        events.extend(self._check_throw(skeleton))

        # Update neutral state AFTER gesture checks so that this frame's
        # neutral value was set by the previous frame (allowing gestures to
        # fire on the impulse frame rather than being blocked by their own
        # velocity spike).
        self._update_neutral(skeleton)

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
    # Neutral state
    # ------------------------------------------------------------------

    def _update_neutral(self, s: Skeleton) -> None:
        """
        Neutral = both wrists near torso AND low velocity.
        Neutral state is used to prevent gesture retriggering.
        """
        neutral_v   = self.cfg.get("neutral_velocity_threshold", 0.05)
        neutral_pos = self.cfg.get("neutral_hand_torso_ratio",   0.6)

        lw_vel = np.linalg.norm(self.tracker.velocity("norm_left_wrist"))
        rw_vel = np.linalg.norm(self.tracker.velocity("norm_right_wrist"))
        low_velocity = (lw_vel < neutral_v) and (rw_vel < neutral_v)

        # Hands near torso: y-component of normalised wrist < neutral_pos
        # (positive y is downward, so wrists should be below/at shoulder level)
        lw_y = s.norm_left_wrist[1]
        rw_y = s.norm_right_wrist[1]
        near_torso = (lw_y > -neutral_pos) and (rw_y > -neutral_pos)

        self._neutral = low_velocity and near_torso

    # ------------------------------------------------------------------
    # Individual gesture checks
    # ------------------------------------------------------------------

    def _check_attack(self, s: Skeleton) -> List[GestureEvent]:
        state = self.states["attack"]
        if not state.can_trigger() or not self._neutral:
            return []

        vel = self.tracker.velocity("norm_right_wrist")
        speed = np.linalg.norm(vel)
        min_v = self.cfg.get("attack_velocity_threshold", 0.15)
        max_v = self.cfg.get("attack_velocity_max", 1.5)

        # Forward = negative x (hand moving toward screen/forward)
        # We treat leftward in the normalised frame as "forward punch" since
        # the image is mirrored (right hand punches left in pixel space → negative x)
        forward = np.array([-1.0, 0.0], dtype=np.float32)
        dot_th  = self.cfg.get("attack_forward_dot_threshold", 0.5)

        if speed < 1e-6:
            return []
        direction = vel / speed

        if (min_v < speed <= max_v and
                float(np.dot(direction, forward)) > dot_th):
            state.mark_triggered()
            return [GestureEvent.ATTACK]
        return []

    def _check_heavy_attack(self, s: Skeleton) -> List[GestureEvent]:
        state = self.states["heavy_attack"]
        vel = self.tracker.velocity("norm_right_wrist")
        speed = np.linalg.norm(vel)

        back_th    = self.cfg.get("heavy_back_threshold",    0.20)
        forward_th = self.cfg.get("heavy_forward_threshold", 0.20)

        # Charging: hand moves backward (positive x in mirrored frame)
        backward = np.array([1.0, 0.0], dtype=np.float32)
        if speed > 1e-6:
            direction = vel / speed
        else:
            direction = np.zeros(2, np.float32)

        events: List[GestureEvent] = []

        if not self._heavy_charging:
            if (speed > back_th and
                    state.can_trigger() and
                    float(np.dot(direction, backward)) > 0.5):
                self._heavy_charging = True
                events.append(GestureEvent.HEAVY_CHARGE)
        else:
            # Release: forward velocity spike
            forward = np.array([-1.0, 0.0], dtype=np.float32)
            if (speed > forward_th and
                    float(np.dot(direction, forward)) > 0.5):
                self._heavy_charging = False
                state.mark_triggered()
                events.append(GestureEvent.HEAVY_RELEASE)

        return events

    def _check_counter(self, s: Skeleton) -> List[GestureEvent]:
        state = self.states["counter"]
        if not state.can_trigger() or not self._neutral:
            return []

        inward_th = self.cfg.get("counter_elbow_inward_threshold", 0.18)

        le_vel = self.tracker.velocity("norm_left_elbow")
        re_vel = self.tracker.velocity("norm_right_elbow")

        # Left elbow moves rightward (+x), right elbow moves leftward (-x)
        le_inward =  le_vel[0]
        re_inward = -re_vel[0]

        if le_inward > inward_th and re_inward > inward_th:
            state.mark_triggered()
            return [GestureEvent.COUNTER]
        return []

    def _check_dodge(self, s: Skeleton) -> List[GestureEvent]:
        state = self.states["dodge"]
        if not state.can_trigger():
            return []

        lateral_th = self.cfg.get("dodge_lateral_threshold", 0.25)
        nose_vel = self.tracker.velocity("norm_nose")
        lateral_speed = abs(nose_vel[0])  # x-axis displacement

        if lateral_speed > lateral_th:
            state.mark_triggered()
            return [GestureEvent.DODGE]
        return []

    def _check_throw(self, s: Skeleton) -> List[GestureEvent]:
        state = self.states["throw"]
        if not state.can_trigger() or not self._neutral:
            return []

        fwd_th  = self.cfg.get("throw_forward_threshold",  0.18)
        down_th = self.cfg.get("throw_downward_threshold", 0.10)

        vel = self.tracker.velocity("norm_right_wrist")
        fwd_speed  = -vel[0]   # negative x = forward
        down_speed =  vel[1]   # positive y = downward

        if fwd_speed > fwd_th and down_speed > down_th:
            state.mark_triggered()
            return [GestureEvent.THROW]
        return []

    def _check_finisher(self, s: Skeleton) -> List[GestureEvent]:
        state = self.states["finisher"]
        if not state.can_trigger() or not self._neutral:
            return []

        fwd_th = self.cfg.get("finisher_forward_threshold", 0.15)

        lw_vel = self.tracker.velocity("norm_left_wrist")
        rw_vel = self.tracker.velocity("norm_right_wrist")

        # Both hands moving forward simultaneously
        lw_fwd = -lw_vel[0]
        rw_fwd = -rw_vel[0]

        if lw_fwd > fwd_th and rw_fwd > fwd_th:
            state.mark_triggered()
            return [GestureEvent.FINISHER]
        return []

    def _check_execute(self, s: Skeleton) -> List[GestureEvent]:
        state = self.states["execute"]
        if not state.can_trigger() or not self._neutral:
            return []

        inward_th = self.cfg.get("execute_inward_threshold",    0.20)
        drop_th   = self.cfg.get("execute_hand_distance_drop",  0.15)

        buf = list(self.tracker.buffer)
        if len(buf) < 2:
            return []

        prev_dist = float(np.linalg.norm(
            buf[-2].norm_left_wrist - buf[-2].norm_right_wrist))
        curr_dist = float(np.linalg.norm(
            s.norm_left_wrist - s.norm_right_wrist))
        dist_drop = prev_dist - curr_dist

        lw_vel = self.tracker.velocity("norm_left_wrist")
        rw_vel = self.tracker.velocity("norm_right_wrist")

        # Left hand moves rightward (+x), right hand moves leftward (-x)
        lw_inward =  lw_vel[0]
        rw_inward = -rw_vel[0]

        if (lw_inward > inward_th and
                rw_inward > inward_th and
                dist_drop > drop_th):
            state.mark_triggered()
            return [GestureEvent.EXECUTE]
        return []

    def _check_special(self, s: Skeleton) -> List[GestureEvent]:
        """
        Special = Left hand above shoulder level AND Right hand extended forward.
        Stateful: emits SPECIAL_START on entry, SPECIAL_END on exit.
        """
        height_ratio = self.cfg.get("special_left_hand_height_ratio", 0.0)
        fwd_th       = self.cfg.get("special_right_hand_forward_threshold", 0.15)

        # Left hand above shoulder: norm_y < height_ratio (negative y = above)
        left_raised  = s.norm_left_wrist[1] < height_ratio
        # Right hand forward: norm_x sufficiently negative
        right_fwd    = s.norm_right_wrist[0] < -fwd_th

        condition_met = left_raised and right_fwd
        events: List[GestureEvent] = []

        if condition_met and not self._special_active:
            if self.states["special"].can_trigger():
                self._special_active = True
                events.append(GestureEvent.SPECIAL_START)

        elif not condition_met and self._special_active:
            self._special_active = False
            self.states["special"].mark_triggered()
            events.append(GestureEvent.SPECIAL_END)

        return events

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def cooldown_display(self) -> Dict[str, float]:
        """Return remaining cooldown (ms) for each gesture, for the overlay."""
        return {name: state.remaining_ms() for name, state in self.states.items()}

    @property
    def is_special_active(self) -> bool:
        return self._special_active

    @property
    def is_heavy_charging(self) -> bool:
        return self._heavy_charging

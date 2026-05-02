"""
input_mapper.py
---------------
Translates GestureEvents from the gesture engine into actual keyboard /
mouse inputs using pynput.

Mapping
-------
    ATTACK         → left-mouse click
    HEAVY_CHARGE   → press & hold left mouse button
    HEAVY_RELEASE  → release left mouse button
    COUNTER        → right-mouse click
    DODGE          → Space tap
    THROW          → E tap
    SPECIAL_START  → press & hold Shift + press & hold left mouse button
    SPECIAL_END    → release left mouse button + release Shift

Camera / Head tracking
----------------------
    Nose position → mouse movement (with deadzone + exponential smoothing)
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

# pynput is not available in headless environments.
# We attempt to import it, but fall back to None controllers gracefully so
# that all non-IO logic (dispatch routing, state tracking) remains testable.
try:
    from pynput.mouse import Button, Controller as MouseController
    from pynput.keyboard import Key, Controller as KeyboardController
    _MOUSE_BTN_LEFT  = Button.left
    _MOUSE_BTN_RIGHT = Button.right
    _KEY_SPACE       = Key.space
    _KEY_SHIFT       = Key.shift
    _PYNPUT_AVAILABLE = True
except Exception:
    Button = None          # type: ignore[assignment]
    Key    = None          # type: ignore[assignment]
    MouseController    = None  # type: ignore[assignment]
    KeyboardController = None  # type: ignore[assignment]
    _MOUSE_BTN_LEFT  = "mouse_left"
    _MOUSE_BTN_RIGHT = "mouse_right"
    _KEY_SPACE       = "space"
    _KEY_SHIFT       = "shift"
    _PYNPUT_AVAILABLE = False

from gesture_engine import GestureEvent
from pose_tracker import Skeleton


# ---------------------------------------------------------------------------
# InputMapper
# ---------------------------------------------------------------------------

class InputMapper:
    """
    Listens for GestureEvents and fires the corresponding input actions.

    Also handles head-tracking mouse movement when update_mouse() is called
    each frame with the current skeleton.

    Usage
    -----
    mapper = InputMapper(config, screen_width, screen_height)
    mapper.handle_events(events)
    mapper.update_mouse(skeleton)

    Testing (headless)
    ------------------
    Replace mapper.mouse and mapper.keyboard with mock objects after
    construction.  The dispatch logic checks ``self.mouse is not None``
    and ``self.keyboard is not None``, so mock objects will be used
    even when pynput is unavailable on the host.
    """

    def __init__(
        self,
        config: dict,
        screen_width: int  = 1920,
        screen_height: int = 1080,
    ) -> None:
        self.cfg = config
        self.screen_w = screen_width
        self.screen_h = screen_height

        # Attempt to create real controllers; stay None if unavailable.
        self.mouse    = None
        self.keyboard = None
        if _PYNPUT_AVAILABLE and MouseController is not None:
            try:
                self.mouse    = MouseController()
                self.keyboard = KeyboardController()
            except Exception:
                pass  # leave as None (e.g. no display in CI)

        # Mouse smoothing state
        self._smoothed_mx: Optional[float] = None
        self._smoothed_my: Optional[float] = None
        self._mouse_alpha = float(config.get("mouse_smoothing_alpha", 0.4))
        self._deadzone_r  = float(config.get("mouse_deadzone_radius", 0.07))
        self._sensitivity = float(config.get("mouse_sensitivity",     800))

        # Track held keys/buttons for special / heavy
        self._special_held: bool = False
        self._lmb_held:     bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_events(self, events: List[GestureEvent]) -> None:
        """Process a list of GestureEvents and simulate the corresponding inputs."""
        for event in events:
            self._dispatch(event)

    def update_mouse(self, skeleton: Optional[Skeleton]) -> None:
        """
        Move the mouse based on nose position relative to frame centre.
        Call once per frame, regardless of gesture events.
        """
        if skeleton is None or not skeleton.valid:
            return

        # Nose position in normalised skeleton space
        nx = float(skeleton.norm_nose[0])
        ny = float(skeleton.norm_nose[1])

        # Apply deadzone
        if abs(nx) < self._deadzone_r:
            nx = 0.0
        if abs(ny) < self._deadzone_r:
            ny = 0.0

        if nx == 0.0 and ny == 0.0:
            return

        # Scale to screen pixels
        target_x = (self.screen_w / 2) + nx * self._sensitivity
        target_y = (self.screen_h / 2) + ny * self._sensitivity

        # Clamp to screen bounds
        target_x = max(0.0, min(float(self.screen_w), target_x))
        target_y = max(0.0, min(float(self.screen_h), target_y))

        # Exponential smoothing: alpha controls responsiveness (higher = faster tracking).
        # Formula: smoothed = alpha * target + (1 - alpha) * previous  (per PRD §15)
        alpha = self._mouse_alpha
        if self._smoothed_mx is None:
            self._smoothed_mx = target_x
            self._smoothed_my = target_y
        else:
            self._smoothed_mx = alpha * target_x + (1 - alpha) * self._smoothed_mx
            self._smoothed_my = alpha * target_y + (1 - alpha) * self._smoothed_my

        if self.mouse is not None:
            self.mouse.position = (int(self._smoothed_mx), int(self._smoothed_my))

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, event: GestureEvent) -> None:
        if event == GestureEvent.ATTACK:
            self._tap_mouse(_MOUSE_BTN_LEFT)

        elif event == GestureEvent.HEAVY_CHARGE:
            self._press_mouse(_MOUSE_BTN_LEFT)

        elif event == GestureEvent.HEAVY_RELEASE:
            self._release_mouse(_MOUSE_BTN_LEFT)

        elif event == GestureEvent.COUNTER:
            self._tap_mouse(_MOUSE_BTN_RIGHT)

        elif event == GestureEvent.DODGE:
            self._tap_key(_KEY_SPACE)

        elif event == GestureEvent.THROW:
            self._tap_char('e')

        elif event == GestureEvent.SPECIAL_START:
            if not self._special_held:
                self._press_key(_KEY_SHIFT)
                self._press_mouse(_MOUSE_BTN_LEFT)
                self._special_held = True
                self._lmb_held     = True

        elif event == GestureEvent.SPECIAL_END:
            if self._special_held:
                self._release_mouse(_MOUSE_BTN_LEFT)
                self._release_key(_KEY_SHIFT)
                self._special_held = False
                self._lmb_held     = False

    # ------------------------------------------------------------------
    # Primitive helpers  (check self.mouse / self.keyboard, not the
    # module-level flag, so mock objects injected in tests are respected)
    # ------------------------------------------------------------------

    def _tap_mouse(self, button) -> None:
        if self.mouse is not None:
            self.mouse.click(button, 1)

    def _press_mouse(self, button) -> None:
        if self.mouse is not None:
            self.mouse.press(button)

    def _release_mouse(self, button) -> None:
        if self.mouse is not None:
            self.mouse.release(button)

    def _tap_key(self, key) -> None:
        if self.keyboard is not None:
            self.keyboard.press(key)
            self.keyboard.release(key)

    def _tap_char(self, char: str) -> None:
        if self.keyboard is not None:
            self.keyboard.press(char)
            self.keyboard.release(char)

    def _press_key(self, key) -> None:
        if self.keyboard is not None:
            self.keyboard.press(key)

    def _release_key(self, key) -> None:
        if self.keyboard is not None:
            self.keyboard.release(key)

    def release_all(self) -> None:
        """Release any held keys/buttons. Call on shutdown."""
        if self._lmb_held:
            self._release_mouse(_MOUSE_BTN_LEFT)
            self._lmb_held = False
        if self._special_held:
            self._release_key(_KEY_SHIFT)
            self._special_held = False


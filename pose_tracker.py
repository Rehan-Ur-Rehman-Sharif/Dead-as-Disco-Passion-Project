"""
pose_tracker.py
---------------
Handles webcam capture, MediaPipe Pose estimation (Tasks API, v0.10+),
skeleton normalisation, and the rolling motion buffer used for velocity-based
gesture detection.

Model download
--------------
On first run the pose landmarker model bundle (~7 MB) is downloaded
automatically from the MediaPipe CDN to ``models/pose_landmarker_lite.task``.
You can also place the file there manually if the CDN is unavailable.
"""

from __future__ import annotations

import collections
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_FILENAME = "pose_landmarker_lite.task"
_MODEL_PATH = _MODEL_DIR / _MODEL_FILENAME
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def ensure_model(model_path: Path = _MODEL_PATH,
                 model_url: str = _MODEL_URL) -> Path:
    """
    Return the path to the pose landmarker model, downloading it first if
    it is not already present.

    Raises ``FileNotFoundError`` if the download fails and the file is absent.
    """
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading pose model to {model_path} …")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print("[INFO] Model downloaded successfully.")
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not download the pose model from:\n  {model_url}\n"
            f"Error: {exc}\n\n"
            "Please download it manually and place it at:\n"
            f"  {model_path}"
        ) from exc
    return model_path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Joint:
    """A single tracked joint with a 2-D normalised position and visibility."""
    x: float = 0.0
    y: float = 0.0
    visibility: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class Skeleton:
    """
    Normalised skeleton for one frame.

    Coordinate system
    -----------------
    * Origin : neck (midpoint of shoulders)
    * Scale  : shoulder width (distance between the two shoulders)
    * +x     : rightward (screen right)
    * +y     : downward  (screen down, matching image coordinates)

    All joint positions are normalised via
        normalised_point = (pixel_point - neck) / shoulder_width
    """
    timestamp: float = field(default_factory=time.monotonic)

    # Raw image joints (pixel space, 0–1 normalised by frame dimensions)
    nose:            Joint = field(default_factory=Joint)
    left_shoulder:   Joint = field(default_factory=Joint)
    right_shoulder:  Joint = field(default_factory=Joint)
    left_elbow:      Joint = field(default_factory=Joint)
    right_elbow:     Joint = field(default_factory=Joint)
    left_wrist:      Joint = field(default_factory=Joint)
    right_wrist:     Joint = field(default_factory=Joint)

    # Derived (set during normalisation)
    neck_x: float = 0.0
    neck_y: float = 0.0
    shoulder_width: float = 1.0  # in pixel-fraction units

    # Normalised joint positions (relative to neck, scaled by shoulder_width)
    norm_nose:           np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    norm_left_shoulder:  np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    norm_right_shoulder: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    norm_left_elbow:     np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    norm_right_elbow:    np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    norm_left_wrist:     np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    norm_right_wrist:    np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))

    valid: bool = False  # False if pose was not detected or visibility too low


# ---------------------------------------------------------------------------
# MediaPipe landmark indices we care about
# ---------------------------------------------------------------------------

MP_NOSE            = 0
MP_LEFT_SHOULDER   = 11
MP_RIGHT_SHOULDER  = 12
MP_LEFT_ELBOW      = 13
MP_RIGHT_ELBOW     = 14
MP_LEFT_WRIST      = 15
MP_RIGHT_WRIST     = 16

MIN_VISIBILITY = 0.5   # joints below this are treated as undetected


# ---------------------------------------------------------------------------
# PoseTracker
# ---------------------------------------------------------------------------

class PoseTracker:
    """
    Wraps MediaPipe PoseLandmarker (Tasks API) and produces a rolling buffer
    of normalised Skeleton objects ready for the gesture engine.

    Usage
    -----
    tracker = PoseTracker(config)
    tracker.open()                        # opens webcam + loads model
    while True:
        frame    = tracker.read_frame()   # BGR numpy array
        skeleton = tracker.process(frame) # latest Skeleton
        annotated = tracker.draw(frame, skeleton)
    tracker.release()
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.buffer_size: int = config.get("motion_buffer_size", 5)

        # Rolling buffer of the last N valid skeletons
        self.buffer: Deque[Skeleton] = collections.deque(maxlen=self.buffer_size)

        # OpenCV video capture (opened lazily or via open())
        self.cap: Optional[cv2.VideoCapture] = None
        self._landmarker: Optional[mp_vision.PoseLandmarker] = None

        self._frame_width  = config.get("frame_width",  640)
        self._frame_height = config.get("frame_height", 360)
        self._model_complexity = config.get("pose_model_complexity", 0)

        # Monotonic timestamp counter for VIDEO mode (ms, must be strictly increasing)
        self._ts_ms: int = 0

    # ------------------------------------------------------------------
    # Camera / model management
    # ------------------------------------------------------------------

    # Map config string names to OpenCV backend constants.
    _BACKEND_MAP: Dict[str, int] = {
        "any":   cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf":  cv2.CAP_MSMF,
        "v4l2":  cv2.CAP_V4L2,
    }

    def open(self, index: Optional[int] = None,
             model_path: Optional[Path] = None) -> bool:
        """Open the webcam and load the pose model. Returns True on success."""
        # --- Webcam ---
        idx = index if index is not None else self.config.get("camera_index", 0)
        backend_name = self.config.get("camera_backend", "any").lower()
        backend = self._BACKEND_MAP.get(backend_name, cv2.CAP_ANY)

        self.cap = cv2.VideoCapture(idx, backend)
        if not self.cap.isOpened() and backend != cv2.CAP_ANY:
            print(
                f"[WARN] Camera backend '{backend_name}' failed to open "
                f"index {idx}; retrying with CAP_ANY."
            )
            self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)

        # --- Model ---
        if model_path is None:
            model_path = ensure_model()
        self._load_model(model_path)
        return True

    def _load_model(self, model_path: Path) -> None:
        # Choose model file based on complexity setting
        # complexity 0 → lite, 1 → full, 2 → heavy (lite used for all in v1)
        base_opts = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame from the webcam. Returns None on failure."""
        if self.cap is None or not self.cap.isOpened():
            return None
        retries = self.config.get("camera_read_retries", 3)
        retry_delay = self.config.get("camera_read_retry_delay_ms", 5) / 1000.0
        for attempt in range(max(1, retries)):
            ok, frame = self.cap.read()
            if ok:
                frame = cv2.flip(frame, 1)  # mirror so left/right feel natural
                return frame
            if attempt < retries - 1:
                time.sleep(retry_delay)
        return None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self._landmarker is not None:
            self._landmarker.close()

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Skeleton:
        """
        Run pose estimation on *frame* and return the resulting Skeleton.
        Also appends valid skeletons to the internal buffer.
        """
        skeleton = Skeleton(timestamp=time.monotonic())

        if self._landmarker is None:
            return skeleton  # model not loaded

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps in milliseconds.
        # A simple counter guarantees monotonicity regardless of system clock.
        self._ts_ms += 1
        result = self._landmarker.detect_for_video(mp_image, self._ts_ms)

        if not result.pose_landmarks:
            return skeleton  # no pose detected

        lm = result.pose_landmarks[0]  # first (only) pose

        def _joint(idx: int) -> Joint:
            p = lm[idx]
            vis = p.visibility if p.visibility is not None else 0.0
            return Joint(x=p.x, y=p.y, visibility=vis)

        skeleton.nose           = _joint(MP_NOSE)
        skeleton.left_shoulder  = _joint(MP_LEFT_SHOULDER)
        skeleton.right_shoulder = _joint(MP_RIGHT_SHOULDER)
        skeleton.left_elbow     = _joint(MP_LEFT_ELBOW)
        skeleton.right_elbow    = _joint(MP_RIGHT_ELBOW)
        skeleton.left_wrist     = _joint(MP_LEFT_WRIST)
        skeleton.right_wrist    = _joint(MP_RIGHT_WRIST)

        # Require shoulders to be visible for normalisation
        if (skeleton.left_shoulder.visibility  < MIN_VISIBILITY or
                skeleton.right_shoulder.visibility < MIN_VISIBILITY):
            return skeleton

        skeleton = self._normalise(skeleton)
        skeleton.valid = True
        self.buffer.append(skeleton)
        return skeleton

    def _normalise(self, s: Skeleton) -> Skeleton:
        """Compute neck origin and shoulder-width scale; populate norm_* fields."""
        ls = s.left_shoulder.as_array()
        rs = s.right_shoulder.as_array()

        neck = (ls + rs) / 2.0
        sw   = float(np.linalg.norm(rs - ls))
        if sw < 1e-6:
            sw = 1e-6  # guard against zero division

        s.neck_x = float(neck[0])
        s.neck_y = float(neck[1])
        s.shoulder_width = sw

        def _norm(j: Joint) -> np.ndarray:
            return (j.as_array() - neck) / sw

        s.norm_nose           = _norm(s.nose)
        s.norm_left_shoulder  = _norm(s.left_shoulder)
        s.norm_right_shoulder = _norm(s.right_shoulder)
        s.norm_left_elbow     = _norm(s.left_elbow)
        s.norm_right_elbow    = _norm(s.right_elbow)
        s.norm_left_wrist     = _norm(s.left_wrist)
        s.norm_right_wrist    = _norm(s.right_wrist)
        return s

    # ------------------------------------------------------------------
    # Velocity helpers (used by gesture engine)
    # ------------------------------------------------------------------

    def velocity(self, joint_attr: str, frames_back: int = 1) -> np.ndarray:
        """
        Return the velocity vector for *joint_attr* (e.g. 'norm_right_wrist')
        across the last *frames_back* frames in the buffer.
        Returns a zero vector if not enough frames are available.
        """
        buf = list(self.buffer)
        if len(buf) < frames_back + 1:
            return np.zeros(2, np.float32)
        curr = getattr(buf[-1], joint_attr)
        prev = getattr(buf[-1 - frames_back], joint_attr)
        return (curr - prev).astype(np.float32)

    def latest(self) -> Optional[Skeleton]:
        """Return the most recent valid skeleton, or None."""
        if self.buffer:
            return self.buffer[-1]
        return None

    # ------------------------------------------------------------------
    # Debug drawing
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray, skeleton: Optional[Skeleton] = None) -> np.ndarray:
        """Draw pose landmarks onto *frame* and return the annotated copy."""
        out = frame.copy()
        if skeleton is None or not skeleton.valid:
            return out

        h, w = out.shape[:2]

        def _px(j: Joint) -> Tuple[int, int]:
            return int(j.x * w), int(j.y * h)

        connections = [
            (skeleton.left_shoulder,  skeleton.right_shoulder),
            (skeleton.left_shoulder,  skeleton.left_elbow),
            (skeleton.left_elbow,     skeleton.left_wrist),
            (skeleton.right_shoulder, skeleton.right_elbow),
            (skeleton.right_elbow,    skeleton.right_wrist),
            (skeleton.nose,           skeleton.left_shoulder),
            (skeleton.nose,           skeleton.right_shoulder),
        ]
        joints = [
            skeleton.nose,
            skeleton.left_shoulder,
            skeleton.right_shoulder,
            skeleton.left_elbow,
            skeleton.right_elbow,
            skeleton.left_wrist,
            skeleton.right_wrist,
        ]

        for a, b in connections:
            cv2.line(out, _px(a), _px(b), (0, 255, 0), 2)
        for j in joints:
            cv2.circle(out, _px(j), 5, (0, 100, 255), -1)

        # Draw neck
        nx, ny = int(skeleton.neck_x * w), int(skeleton.neck_y * h)
        cv2.circle(out, (nx, ny), 5, (255, 255, 0), -1)

        return out


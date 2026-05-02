# Dead as Disco – Webcam Motion Combat Controller

A real-time **webcam-based motion tracking system** that converts upper-body gestures into keyboard and mouse inputs for a rhythm-based combat game.  Inspired by *Dead as Disco*.

> **Goal:** Keep your body moving to the beat — no extra hardware required, just a webcam.

---

## Features

- Upper-body pose tracking via **MediaPipe Pose** (nose, shoulders, elbows, wrists)
- **8 gesture types** mapped to game inputs (attack, counter, dodge, throw, finisher, execute, heavy attack, special)
- Head-tracking **mouse movement** with deadzone and exponential smoothing
- Per-gesture **cooldown system** to prevent false positives
- **Neutral-state gate** — gestures only fire after hands return to rest
- Optional **BPM beat-window** for timing-aware "perfect" hits
- **Debug overlay** showing skeleton, active gesture, FPS, and cooldown bars
- Optional **multi-threaded pipeline** (capture / pose / gesture+display) for lower latency
- Fully configurable via **`config.json`**

---

## Requirements

| Package        | Purpose                        |
|----------------|--------------------------------|
| Python ≥ 3.9   | Runtime                        |
| `opencv-python`| Webcam capture & display       |
| `mediapipe`    | Pose estimation                |
| `pynput`       | Keyboard / mouse simulation    |
| `numpy`        | Vector math                    |

Install dependencies:

```bash
pip install opencv-python mediapipe pynput numpy
```

---

## File Structure

```
Dead-as-Disco-Passion-Project/
├── main.py            # Entry point – threading, display loop, debug overlay
├── pose_tracker.py    # Webcam capture, MediaPipe pose, skeleton normalisation
├── gesture_engine.py  # Event-driven gesture detection & state management
├── input_mapper.py    # Keyboard / mouse input simulation via pynput
├── config.json        # All tunable thresholds, cooldowns, and settings
└── README.md
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/Rehan-Ur-Rehman-Sharif/Dead-as-Disco-Passion-Project.git
cd Dead-as-Disco-Passion-Project
pip install opencv-python mediapipe pynput numpy

# Run
python main.py
```

A window will open showing your webcam feed with a skeleton overlay.

### Runtime Controls

| Key | Action                         |
|-----|--------------------------------|
| `Q` | Quit                           |
| `D` | Toggle debug overlay on / off  |
| `S` | Toggle head-tracking mouse on / off |

---

## Gesture Reference

All gestures use **velocity impulses** detected over a 3–5 frame buffer — no long holds or poses required (except Special and Heavy Attack).

| Gesture        | Motion                                      | Input           | Cooldown   |
|----------------|---------------------------------------------|-----------------|------------|
| **Attack**     | Right hand punches forward                  | Left-click      | 150 ms     |
| **Heavy Attack** | Right hand pulls back → drives forward    | Hold LMB → Release | 300 ms |
| **Counter**    | Both hands sweep upward past shoulder level | Right-click     | 200 ms     |
| **Dodge**      | Head snaps left or right                    | Space           | 250 ms     |
| **Throw**      | Right hand forward + downward arc           | `R`             | 200 ms     |
| **Finisher**   | Both hands thrust forward simultaneously    | `F`             | 350 ms     |
| **Execute**    | Both hands pull inward toward torso         | `E`             | 300 ms     |
| **Special**    | Left hand raised above shoulder **+** right hand extended forward | Hold `Shift` + Hold LMB | 200 ms re-entry |

### Conflict Priority

When multiple gestures could fire at once, the system resolves by priority:

1. Special (blocks Attack / Counter / Heavy while active)
2. Finisher
3. Execute
4. Heavy Attack
5. Attack
6. Counter
7. Dodge / Throw

---

## Configuration (`config.json`)

```jsonc
{
  "camera_index": 0,          // webcam device index
  "camera_backend": "dshow",  // capture backend: "dshow" (Windows), "v4l2" (Linux), "msmf", "any"
  "camera_read_retries": 3,   // retry attempts per frame read (helps with virtual cameras)
  "camera_read_retry_delay_ms": 5, // milliseconds to wait between retry attempts
  "frame_width": 640,
  "frame_height": 360,
  "motion_buffer_size": 5,    // frames kept for velocity calculation

  // Per-gesture velocity thresholds (normalised shoulder-width units)
  "attack_velocity_threshold": 0.15,
  "dodge_threshold": 0.25,
  // ... (see config.json for the full list)

  "smoothing_alpha": 0.7,     // gesture smoothing
  "mouse_smoothing_alpha": 0.4,
  "mouse_deadzone_radius": 0.07,
  "mouse_sensitivity": 800,

  "cooldowns": {
    "attack": 150,            // milliseconds
    "dodge":  250,
    "counter": 200,
    "special_reentry": 200
  },

  "bpm": 128,                 // 0 = beat awareness disabled
  "beat_window_ms": 80
}
```

---

## System Architecture

```
[ Webcam ]
     ↓  Thread 1
[ Pose Estimation (MediaPipe) ]
     ↓  Thread 2
[ Skeleton Normalisation ]
     ↓
[ Motion Buffer (3–5 frames) ]
     ↓  Thread 3
[ Gesture Detection Engine ]
     ↓
[ State Manager / Cooldowns ]
     ↓
[ Input Mapper (pynput) ]
```

### Coordinate System

```
Origin  : neck (midpoint of shoulders)
Scale   : shoulder width
+x      : screen right
+y      : screen down

normalised_point = (pixel_point − neck) / shoulder_width
```

### Velocity

```
velocity = current_position − previous_position   (per frame)
```

---

## Performance

| Metric              | Target  |
|---------------------|---------|
| Frame rate          | ≥ 25 fps |
| Gesture latency     | ≤ 80 ms  |
| Recognition window  | ≤ 120 ms |

Tips for best performance:
- Ensure good lighting so MediaPipe can detect your pose reliably.
- Keep your upper body fully in frame.
- Use `pose_model_complexity: 0` in `config.json` for maximum speed (default).
- Enable threading (`THREADED = True` in `main.py`, which is the default).

---

## Tuning

If gestures fire too easily (false positives), **increase** the relevant `*_threshold` value in `config.json`.

If gestures are hard to trigger, **decrease** the threshold.

The debug overlay (press `D`) shows real-time cooldown bars so you can see when each gesture becomes available again.

---

## License

Personal passion project. Use freely for personal health and fun.

---

## Troubleshooting

### DroidCam USB — `can't grab frame` / MSMF errors

If you see repeated warnings like:

```
[ WARN] global cap_msmf.cpp:1815 CvCapture_MSMF::grabFrame videoio(MSMF): can't grab frame. Error: -1072875772
```

This means Windows defaulted to the **MSMF** backend, which is unreliable with virtual/USB cameras like DroidCam.

**Fix:** set `camera_backend` to `"dshow"` in `config.json` and set `camera_index` to the correct device index for DroidCam (typically `1`, but may vary depending on how many cameras are installed):

```json
{
  "camera_index": 1,
  "camera_backend": "dshow",
  "camera_read_retries": 3
}
```

The `camera_read_retries` setting (default `3`) makes the app automatically retry transient frame-grab failures before giving up, improving frame acquisition reliability when the camera driver emits intermittent errors.

| `camera_backend` value | Backend used         | Notes                         |
|------------------------|----------------------|-------------------------------|
| `"dshow"`              | DirectShow (Windows) | Recommended for DroidCam USB  |
| `"msmf"`               | MSMF (Windows)       | Default Windows backend; unreliable with virtual cams |
| `"v4l2"`               | V4L2 (Linux)         | Recommended on Linux          |
| `"any"`                | OpenCV auto-detect   | Cross-platform fallback       |

If the configured backend fails to open the camera, the app will automatically retry with `"any"` and print a warning.

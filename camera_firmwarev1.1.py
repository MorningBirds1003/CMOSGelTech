"""
Camera firmware scaffold for NexiGo N680E Pro / USB UVC cameras.
- NexiGo-oriented 16:9 capture defaults.
- Four-image capture sequences.
- Manual focus calibration video mode with keyboard controls.
- Persistent local settings file for save directory and focus settings.
- No lighting module; illumination is assumed to be manually adjusted.
- Placeholder device hooks for future integration with external hardware APIs.

Typical use:
    python camera_firmwarev1.1.py

At startup, choose:
    1 = calibration mode
    2 = main mode using saved calibration settings
    3 = choose/change save directory only

Optional non-interactive use:
    python camera_firmwarev1.1.py --mode calibrate
    python camera_firmwarev1.1.py --mode main
    python camera_firmwarev1.1.py --mode trigger
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

# Configuration

@dataclass
class CameraConfig:
    """OpenCV/UVC camera configuration.

    Attributes:
        camera_index: OpenCV camera index. On Windows, test indices 0-5 if unsure.
        preferred_resolutions: Ordered list of requested capture sizes. The NexiGo
            N680E Pro is a 16:9 camera; 3840x2160 is attempted first, then 1080p.
        capture_backend: OpenCV backend. cv2.CAP_DSHOW is usually most stable on
            Windows. Use None to let OpenCV auto-select.
        startup_warmup_s: Delay after opening the camera before first use.
        flush_frames_on_startup: Number of frames discarded after opening camera.
        flush_frames_before_capture: Number of frames discarded before each image.
        autofocus: If True, request camera autofocus through UVC/OpenCV if exposed.
        manual_focus_value: Optional manual focus value. Range is camera/driver
            dependent, commonly 0-255 or 0-1023. Calibration mode lets you test it.
    """

    camera_index: int = 1
    preferred_resolutions: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (3840, 2160),  # NexiGo N680E Pro 4K UHD, 16:9
        ]
    )
    capture_backend: Optional[int] = cv2.CAP_DSHOW
    startup_warmup_s: float = 1.0
    flush_frames_on_startup: int = 10
    flush_frames_before_capture: int = 5
    autofocus: bool = True
    manual_focus_value: Optional[int] = None

@dataclass
class TriggerConfig:
    """Trigger and sequence timing configuration."""

    capture_threshold: float = 10.0
    trigger_debounce_s: float = 0.25
    polling_interval_s: float = 0.01
    shutter_interval_s: float = 1.0
    sequence_shot_count: int = 4

@dataclass
class StorageConfig:
    """Image and metadata storage configuration."""

    output_dir: str = "captures"
    image_extension: str = "png"
    save_metadata_json: bool = True
    filename_prefix: str = "capture"
    settings_file: str = "camera_settings.json"

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

# Persistent settings

class SettingsManager:
    """Small JSON-backed settings store.

    JSON is used instead of a plain txt file because the settings are structured
    data: output directory, focus mode, focus value, and future device options.
    It remains human-editable, but is less fragile than parsing free-form text.
    """

    def __init__(self, settings_path: Path) -> None:
        self.settings_path = settings_path

    def load(self) -> Dict[str, Any]:
        if not self.settings_path.exists():
            return {}
        with open(self.settings_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, settings: Dict[str, Any]) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        logging.info("[SETTINGS] Saved settings to %s", self.settings_path)

    def apply_to_config(self, config: AppConfig) -> AppConfig:
        settings = self.load()

        storage = settings.get("storage", {})
        camera = settings.get("camera", {})

        if "output_dir" in storage:
            config.storage.output_dir = storage["output_dir"]
        if "autofocus" in camera:
            config.camera.autofocus = bool(camera["autofocus"])
        if "manual_focus_value" in camera:
            config.camera.manual_focus_value = camera["manual_focus_value"]

        return config

    def update_storage_output_dir(self, output_dir: str) -> None:
        settings = self.load()
        settings.setdefault("storage", {})["output_dir"] = output_dir
        self.save(settings)

    def update_focus(self, autofocus: bool, manual_focus_value: Optional[int]) -> None:
        settings = self.load()
        settings.setdefault("camera", {})["autofocus"] = autofocus
        settings["camera"]["manual_focus_value"] = manual_focus_value
        self.save(settings)

# Future external-device integration hooks

class ExternalDeviceHooks:
    """Callable placeholders for integration with the final device API.

    API contract for future scripts:
        initialize() -> None
            Open connections to the external controller, motor, carousel, DAQ,
            embedded device, or other instrument.

        before_capture_sequence(context: dict) -> None
            Called immediately before a four-shot image sequence starts. Use this
            to lock motion, move a sample into position, arm sensors, etc.

        before_each_capture(shot_index: int, context: dict) -> None
            Called before each individual image. Use this for synchronized device
            states, e.g. pause a pump or request a mechanical settle delay.

        after_each_capture(shot_index: int, image_path: Path, metadata_path: Path | None, context: dict) -> None
            Called after each image is saved. Use this for sample indexing,
            external logs, or device acknowledgements.

        after_capture_sequence(results: list, context: dict) -> None
            Called after the full sequence completes.

        shutdown() -> None
            Cleanly close the device connection.
    """

    def initialize(self) -> None:
        logging.info("[HOOK] initialize placeholder called")

    def before_capture_sequence(self, context: Dict[str, Any]) -> None:
        logging.info("[HOOK] before_capture_sequence placeholder called: %s", context)

    def before_each_capture(self, shot_index: int, context: Dict[str, Any]) -> None:
        logging.debug("[HOOK] before_each_capture placeholder called: shot=%s context=%s", shot_index, context)

    def after_each_capture(
        self,
        shot_index: int,
        image_path: Path,
        metadata_path: Optional[Path],
        context: Dict[str, Any],
    ) -> None:
        logging.debug(
            "[HOOK] after_each_capture placeholder called: shot=%s image=%s metadata=%s context=%s",
            shot_index,
            image_path,
            metadata_path,
            context,
        )

    def after_capture_sequence(self, results: List[Tuple[Path, Optional[Path], float]], context: Dict[str, Any]) -> None:
        logging.info("[HOOK] after_capture_sequence placeholder called: %d result(s)", len(results))

    def shutdown(self) -> None:
        logging.info("[HOOK] shutdown placeholder called")

# Camera capture

class OpenCVCamera:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.negotiated_resolution: Optional[Tuple[int, int]] = None

    def open(self) -> None:
        logging.info("[CV] Opening camera index %s", self.config.camera_index)
        if self.config.capture_backend is not None:
            self.cap = cv2.VideoCapture(self.config.camera_index, self.config.capture_backend)
        else:
            self.cap = cv2.VideoCapture(self.config.camera_index)

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {self.config.camera_index}")

        self._negotiate_resolution()
        self.apply_focus_settings()

        time.sleep(self.config.startup_warmup_s)
        self.flush_frames(self.config.flush_frames_on_startup)

        frame = self.read_frame()
        actual_h, actual_w = frame.shape[:2]
        self.negotiated_resolution = (actual_w, actual_h)

        logging.info("[CV] Camera opened. Final resolution: %sx%s", actual_w, actual_h)

    def _negotiate_resolution(self) -> None:
        if self.cap is None:
            raise RuntimeError("Camera not open")

        last_actual: Optional[Tuple[int, int]] = None
        for width, height in self.config.preferred_resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            time.sleep(0.2)
            self.flush_frames(3)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            actual_h, actual_w = frame.shape[:2]
            last_actual = (actual_w, actual_h)
            logging.info("[CV] Requested %sx%s; camera returned %sx%s", width, height, actual_w, actual_h)

            if abs(actual_w - width) <= 16 and abs(actual_h - height) <= 16:
                self.negotiated_resolution = (actual_w, actual_h)
                return

        if last_actual is None:
            raise RuntimeError("Camera opened but no usable frame was returned during resolution negotiation")
        self.negotiated_resolution = last_actual

    def apply_focus_settings(self) -> None:
        if self.cap is None:
            raise RuntimeError("Camera not open")

        # OpenCV focus control depends on what the UVC driver exposes.
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.config.autofocus else 0)
        if not self.config.autofocus and self.config.manual_focus_value is not None:
            self.cap.set(cv2.CAP_PROP_FOCUS, int(self.config.manual_focus_value))

        logging.info(
            "[CV] Focus requested: autofocus=%s manual_focus_value=%s",
            self.config.autofocus,
            self.config.manual_focus_value,
        )

    def close(self) -> None:
        if self.cap is not None:
            logging.info("[CV] Releasing camera")
            self.cap.release()
            self.cap = None

    def flush_frames(self, n: int) -> None:
        if self.cap is None:
            raise RuntimeError("Camera not open")
        for _ in range(n):
            self.cap.read()

    def read_frame(self):
        if self.cap is None:
            raise RuntimeError("Camera not open")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def save_frame(self, frame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), frame)
        if not ok:
            raise RuntimeError(f"Failed to write image to {output_path}")

    def set_manual_focus(self, focus_value: int) -> None:
        if self.cap is None:
            raise RuntimeError("Camera not open")
        self.config.autofocus = False
        self.config.manual_focus_value = int(focus_value)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, int(focus_value))

    def set_autofocus(self, enabled: bool) -> None:
        if self.cap is None:
            raise RuntimeError("Camera not open")
        self.config.autofocus = enabled
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if enabled else 0)

# Trigger / load-cell interface

class LoadCellInterface:
    """Placeholder for load-cell, force sensor, serial, DAQ, or ADC hardware.

    Replace read_value() with the real hardware call. Keep should_capture() stable
    so the capture manager can remain hardware-agnostic.
    """

    def initialize(self) -> None:
        logging.info("[LC] Initializing load cell interface placeholder")

    def read_value(self) -> float:
        return 12.34

    def should_capture(self, value: float, threshold: float) -> bool:
        return value >= threshold

    def shutdown(self) -> None:
        logging.info("[LC] Shutting down load cell interface placeholder")

# Calibration video mode

def resize_for_preview(frame, max_width: int = 1280):
    """Resize a frame for display while preserving the snapshot aspect ratio.

    This helper is used only for the calibration video preview. It does not
    change the raw camera frame that is saved by capture_once() or by the
    calibration-frame save command.

    Args:
        frame: OpenCV image array to display.
        max_width: Maximum preview width in pixels. Wider frames are scaled
            down proportionally; smaller frames are shown at native size.

    Returns:
        OpenCV image array with the same aspect ratio as the input frame.
    """
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / float(width)
    preview_width = int(round(width * scale))
    preview_height = int(round(height * scale))

    return cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)

class CalibrationManager:
    """Interactive focus and storage calibration UI.

    Keyboard controls in the video window:
        q / Esc : save settings and quit
        a       : toggle autofocus on/off
        [ / ]   : decrease/increase manual focus by 5
        - / +   : decrease/increase manual focus by 1
        s       : save the current full-resolution camera frame to the configured output folder

    The displayed calibration preview is resized for screen usability while
    preserving the exact aspect ratio of the camera frame/saved snapshots.
    """

    def __init__(self, camera: OpenCVCamera, settings_manager: SettingsManager, storage: StorageConfig) -> None:
        self.camera = camera
        self.settings_manager = settings_manager
        self.storage = storage

    def run(self, output_dir: Optional[str] = None, window_name: str = "Camera focus calibration") -> None:
        if output_dir is not None:
            self.storage.output_dir = output_dir
            self.settings_manager.update_storage_output_dir(output_dir)

        output_path = Path(self.storage.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        focus_value = self.camera.config.manual_focus_value
        if focus_value is None:
            raw_focus = self.camera.cap.get(cv2.CAP_PROP_FOCUS) if self.camera.cap is not None else 0
            focus_value = int(raw_focus) if raw_focus >= 0 else 0

        logging.info("[CAL] Starting calibration. Output directory: %s", output_path)
        logging.info("[CAL] Press q/Esc to save and quit")

        # WINDOW_AUTOSIZE prevents the OS/window manager from freely stretching
        # the preview. The preview frame itself is resized proportionally below.
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        while True:
            frame = self.camera.read_frame()
            overlay = frame.copy()
            status = (
                f"autofocus={self.camera.config.autofocus} | "
                f"manual_focus={focus_value} | save_dir={output_path}"
            )
            cv2.putText(overlay, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(overlay, "q/Esc quit | a autofocus | [ ] +/-5 | - + +/-1 | s save frame", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            preview = resize_for_preview(overlay, max_width=1280)
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                self.camera.set_autofocus(not self.camera.config.autofocus)
            elif key == ord("["):
                focus_value = max(0, focus_value - 5)
                self.camera.set_manual_focus(focus_value)
            elif key == ord("]"):
                focus_value += 5
                self.camera.set_manual_focus(focus_value)
            elif key == ord("-"):
                focus_value = max(0, focus_value - 1)
                self.camera.set_manual_focus(focus_value)
            elif key in (ord("+"), ord("=")):
                focus_value += 1
                self.camera.set_manual_focus(focus_value)
            elif key == ord("s"):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                calibration_image = output_path / f"calibration_{stamp}.{self.storage.image_extension}"
                self.camera.save_frame(frame, calibration_image)
                logging.info("[CAL] Saved calibration frame: %s", calibration_image)

        self.settings_manager.update_focus(
            autofocus=self.camera.config.autofocus,
            manual_focus_value=None if self.camera.config.autofocus else focus_value,
        )
        cv2.destroyWindow(window_name)
        logging.info("[CAL] Calibration finished")

# Capture / saving

class CaptureManager:
    def __init__(
        self,
        app_config: AppConfig,
        camera: OpenCVCamera,
        load_cell: LoadCellInterface,
        hooks: Optional[ExternalDeviceHooks] = None,
    ) -> None:
        self.app_config = app_config
        self.camera = camera
        self.load_cell = load_cell
        self.hooks = hooks or ExternalDeviceHooks()
        self.output_dir = Path(app_config.storage.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_capture_time: float = 0.0

    def initialize(self) -> None:
        logging.info("[APP] Initializing system")
        self.hooks.initialize()
        self.camera.open()
        self.load_cell.initialize()

    def shutdown(self) -> None:
        logging.info("[APP] Shutting down system")
        self.camera.close()
        self.load_cell.shutdown()
        self.hooks.shutdown()

    def _build_base_filename(self, prefix: str, value: float) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_prefix = prefix.replace(" ", "_")
        return f"{safe_prefix}_{timestamp}_load_{value:.3f}"

    def _build_output_path(self, base_filename: str) -> Path:
        self.output_dir = Path(self.app_config.storage.output_dir)
        return self.output_dir / f"{base_filename}.{self.app_config.storage.image_extension}"

    def _build_metadata_path(self, image_path: Path) -> Path:
        return image_path.with_suffix(".json")

    def _save_metadata(self, image_path: Path, value: float, frame_shape: Tuple[int, ...]) -> Optional[Path]:
        if not self.app_config.storage.save_metadata_json:
            return None

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "load_value": value,
            "image_path": str(image_path),
            "camera": {
                **asdict(self.app_config.camera),
                "negotiated_resolution": self.camera.negotiated_resolution,
                "model_notes": {
                    "assumed_model": "NexiGo N680E Pro",
                    "aspect_ratio": "16:9",
                    "lighting_control": "manual_external_adjustment",
                },
            },
            "trigger": asdict(self.app_config.trigger),
            "frame_shape": {
                "height": frame_shape[0],
                "width": frame_shape[1],
                "channels": frame_shape[2] if len(frame_shape) > 2 else 1,
            },
        }

        metadata_path = self._build_metadata_path(image_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logging.info("[APP] Metadata saved to %s", metadata_path)
        return metadata_path

    def _debounce_ok(self) -> bool:
        return (time.time() - self.last_capture_time) >= self.app_config.trigger.trigger_debounce_s

    def capture_once(self, prefix: str = "capture", shot_index: Optional[int] = None) -> Tuple[Path, Optional[Path], float]:
        value = self.load_cell.read_value()
        base_filename = self._build_base_filename(prefix, value)
        output_path = self._build_output_path(base_filename)

        logging.info("[APP] Capturing image%s (load value=%.4f)", f" #{shot_index}" if shot_index is not None else "", value)

        self.camera.flush_frames(self.app_config.camera.flush_frames_before_capture)
        frame = self.camera.read_frame()
        self.camera.save_frame(frame, output_path)
        metadata_path = self._save_metadata(output_path, value, frame.shape)
        self.last_capture_time = time.time()

        return output_path, metadata_path, value

    def capture_sequence(
        self,
        prefix: str = "sequence",
        shutter_interval_s: Optional[float] = None,
        shot_count: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Path, Optional[Path], float]]:
        """Capture a fixed-count image sequence.

        Default behavior is four images total, spaced by shutter_interval_s.
        This replaces the older duration-based sequence behavior.
        """

        interval = shutter_interval_s if shutter_interval_s is not None else self.app_config.trigger.shutter_interval_s
        total_shots = shot_count if shot_count is not None else self.app_config.trigger.sequence_shot_count
        context = context or {"prefix": prefix, "shot_count": total_shots, "interval_s": interval}

        if interval <= 0:
            raise ValueError("shutter_interval_s must be > 0")
        if total_shots <= 0:
            raise ValueError("shot_count must be > 0")

        logging.info("[APP] Starting capture sequence: %d shot(s), interval=%.3fs", total_shots, interval)
        self.hooks.before_capture_sequence(context)

        results: List[Tuple[Path, Optional[Path], float]] = []
        for shot_index in range(total_shots):
            if shot_index > 0:
                time.sleep(interval)

            shot_prefix = f"{prefix}_shot_{shot_index + 1:02d}_of_{total_shots:02d}"
            self.hooks.before_each_capture(shot_index, context)
            result = self.capture_once(prefix=shot_prefix, shot_index=shot_index)
            results.append(result)
            self.hooks.after_each_capture(shot_index, result[0], result[1], context)
            logging.info("[APP] Sequence shot %02d/%02d saved: %s", shot_index + 1, total_shots, result[0])

        self.hooks.after_capture_sequence(results, context)
        logging.info("[APP] Capture sequence finished: %d image(s) saved", len(results))
        return results

    def run_trigger_loop(self) -> None:
        logging.info("[APP] Entering trigger loop")
        threshold = self.app_config.trigger.capture_threshold

        while True:
            value = self.load_cell.read_value()
            if self.load_cell.should_capture(value, threshold) and self._debounce_ok():
                try:
                    prefix = f"load_{value:.2f}"
                    results = self.capture_sequence(prefix=prefix)
                    logging.info("[APP] Triggered sequence complete: %d image(s)", len(results))
                except Exception as exc:
                    logging.exception("[APP] Triggered capture sequence failed: %s", exc)

            time.sleep(self.app_config.trigger.polling_interval_s)

    def single_test_capture(self, prefix: str = "manual_test") -> None:
        image_path, metadata_path, value = self.capture_once(prefix=prefix)
        print(f"Saved image: {image_path}")
        print(f"Saved metadata: {metadata_path}")
        print(f"Load value: {value}")
        print(f"Negotiated resolution: {self.camera.negotiated_resolution}")

# Factory and CLI

def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def prompt_for_directory(current_dir: Optional[str] = None, *, allow_keep: bool = True) -> str:
    """Prompt the operator to select where images and metadata are saved.

    Args:
        current_dir: Existing directory from camera_settings.json or the active config.
        allow_keep: If True, the operator may keep the existing directory by
            pressing Enter or answering yes. If False, a new directory is required.

    Returns:
        Absolute path string for a directory that exists or was created.

    Notes for future API integration:
        This function is intentionally isolated from camera initialization. A later
        touchscreen, serial command, GUI, or parent script can replace this prompt
        while still writing the selected path through SettingsManager.
    """
    print("\n--- Save Directory Selection ---")

    if current_dir:
        current_path = Path(current_dir).expanduser()
        print(f"Current directory: {current_path}")
        if allow_keep:
            choice = input("Press Enter to keep this directory, or type a new path: ").strip()
            if not choice:
                current_path.mkdir(parents=True, exist_ok=True)
                resolved = current_path.resolve()
                print(f"Keeping directory: {resolved}")
                return str(resolved)
            candidate = choice
        else:
            candidate = input("Enter new save directory path: ").strip()
    else:
        candidate = input("Enter save directory path [captures]: ").strip() or "captures"

    while True:
        path = Path(candidate).expanduser()

        try:
            path.mkdir(parents=True, exist_ok=True)
            resolved = path.resolve()
            print(f"Directory set to: {resolved}")
            return str(resolved)
        except Exception as e:
            print(f"Invalid path: {e}")
            candidate = input("Enter a different save directory path: ").strip()
            if not candidate:
                print("Path cannot be empty.")

def build_default_config(settings_path: Path) -> AppConfig:
    config = AppConfig(storage=StorageConfig(settings_file=str(settings_path)))
    return SettingsManager(settings_path).apply_to_config(config)

def build_manager(config: AppConfig) -> CaptureManager:
    camera = OpenCVCamera(config.camera)
    load_cell = LoadCellInterface()
    hooks = ExternalDeviceHooks()
    return CaptureManager(config, camera, load_cell, hooks)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NexiGo camera firmware scaffold")
    parser.add_argument(
        "--mode",
        choices=["prompt", "calibrate", "main", "single", "sequence", "trigger", "set-output-dir"],
        default="prompt",
        help=(
            "prompt asks at startup; calibrate opens calibration video; "
            "main/sequence runs the four-image main capture sequence using saved settings; "
            "trigger enters the trigger loop; set-output-dir changes the saved image folder."
        ),
    )
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--settings", type=str, default="camera_settings.json")
    parser.add_argument("--shot-count", type=int, default=None, help="Defaults to 4")
    parser.add_argument("--interval", type=float, default=None, help="Seconds between shots")
    parser.add_argument("--prefix", type=str, default="N680E_sequence")
    return parser.parse_args()

def choose_startup_mode() -> str:
    """Ask the operator which high-level mode should run.

    Returns:
        "calibrate" when the operator enters 1.
        "main" when the operator enters 2.
        "set-output-dir" when the operator enters 3.
        "exit" when the operator enters 4.

    This function is intentionally small and callable so a future device UI can
    replace it without changing the capture or calibration logic.
    """
    while True:
        print("\nSelect camera mode:")
        print("  1) Calibration mode - live video focus tuning and save-folder setup")
        print("  2) Main mode - run capture using saved calibration/settings")
        print("  3) Choose/change save directory only")
        print("  4) Exit")
        choice = input("Enter 1, 2, 3, or 4: ").strip()

        if choice == "1":
            return "calibrate"
        if choice == "2":
            return "main"
        if choice == "3":
            return "set-output-dir"
        if choice == "4":
            return "exit"

        print("Invalid selection. Please enter 1, 2, 3, or 4.")

def run_calibration_mode(
    manager: CaptureManager,
    settings_manager: SettingsManager,
    config: AppConfig,
    output_dir: Optional[str] = None,
) -> None:
    """Run live focus calibration and persist the selected settings.

    Callable API entry point for future scripts/device UI:
        run_calibration_mode(manager, settings_manager, config, output_dir=None)

    Args:
        manager: Initialized CaptureManager containing an open camera.
        settings_manager: JSON settings persistence manager.
        config: Active app configuration.
        output_dir: Optional directory override. When supplied, it is saved to
            camera_settings.json and reused by main mode.
    """
    CalibrationManager(manager.camera, settings_manager, config.storage).run(output_dir=output_dir)

def choose_and_save_output_directory(settings_manager: SettingsManager, config: AppConfig) -> str:
    """Prompt for a save directory, persist it, and update the active config.

    Callable API entry point for future scripts/device UI:
        output_dir = choose_and_save_output_directory(settings_manager, config)

    Returns:
        Absolute path string for the selected output directory.
    """
    selected_dir = prompt_for_directory(config.storage.output_dir)
    config.storage.output_dir = selected_dir
    settings_manager.update_storage_output_dir(selected_dir)
    return selected_dir

def run_main_mode(
    manager: CaptureManager,
    prefix: str = "N680E_sequence",
    shutter_interval_s: Optional[float] = None,
    shot_count: Optional[int] = None,
) -> List[Tuple[Path, Optional[Path], float]]:
    """Run the standard main capture sequence using saved calibration settings.

    Callable API entry point for future scripts/device UI:
        results = run_main_mode(manager, prefix="sample_A")

    Main mode intentionally uses the already-loaded settings file through
    build_default_config(), including save directory, autofocus/manual focus,
    and future persisted camera options.

    Args:
        manager: Initialized CaptureManager containing an open camera.
        prefix: Filename prefix for the saved images.
        shutter_interval_s: Optional override for seconds between shots.
        shot_count: Optional override. Defaults to config.trigger.sequence_shot_count,
            currently 4.

    Returns:
        List of tuples: (image_path, metadata_path, load_value).
    """
    results = manager.capture_sequence(
        prefix=prefix,
        shutter_interval_s=shutter_interval_s,
        shot_count=shot_count,
    )
    print(f"Saved {len(results)} images in main capture sequence")
    for image_path, metadata_path, value in results:
        print(f"Image: {image_path} | Metadata: {metadata_path} | Load: {value}")
    return results

def _apply_cli_overrides(args: argparse.Namespace, settings_manager: SettingsManager, config: AppConfig) -> None:
    """Apply CLI overrides to the active config and persist output-dir overrides.

    Notes:
        This helper keeps main() smaller and allows the interactive menu loop to
        rebuild/reload config after calibration or save-directory changes.
    """
    if args.camera_index is not None:
        config.camera.camera_index = args.camera_index

    if args.output_dir is not None:
        output_path = Path(args.output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        config.storage.output_dir = str(output_path.resolve())
        settings_manager.update_storage_output_dir(config.storage.output_dir)

def _log_active_config(selected_mode: str, settings_path: Path, config: AppConfig) -> None:
    """Log the current mode, settings path, output directory, and focus settings."""
    logging.info("[APP] Selected startup mode: %s", selected_mode)
    logging.info("[APP] Settings file: %s", settings_path)
    logging.info("[APP] Active output directory: %s", config.storage.output_dir)
    logging.info(
        "[APP] Active focus settings: autofocus=%s manual_focus_value=%s",
        config.camera.autofocus,
        config.camera.manual_focus_value,
    )

def run_camera_mode_once(
    selected_mode: str,
    args: argparse.Namespace,
    settings_path: Path,
    settings_manager: SettingsManager,
) -> bool:
    """Run one selected mode.

    Args:
        selected_mode: One of calibrate, main, single, trigger, set-output-dir, or exit.
        args: Parsed CLI arguments.
        settings_path: Path to the JSON settings file.
        settings_manager: Persistent JSON settings manager.

    Returns:
        True when the caller should return to the interactive mode menu.
        False when the program should exit after this mode completes.

    Behavior:
        - Calibration returns to the mode menu after the calibration window closes.
        - Save-directory setup returns to the mode menu after saving.
        - Main/single/trigger are terminal modes by default.
    """
    if selected_mode == "sequence":
        selected_mode = "main"

    if selected_mode == "exit":
        print("Exiting camera program.")
        return False

    config = build_default_config(settings_path)
    _apply_cli_overrides(args, settings_manager, config)

    if selected_mode == "set-output-dir":
        selected_dir = choose_and_save_output_directory(settings_manager, config)
        print(f"Saved output directory: {selected_dir}")
        return True

    # In calibration mode, allow the operator to choose the save directory before
    # the camera opens. This keeps calibration screenshots and later main-mode
    # captures in the same persisted folder unless changed later.
    if selected_mode == "calibrate" and args.output_dir is None:
        choose_and_save_output_directory(settings_manager, config)

    _log_active_config(selected_mode, settings_path, config)

    manager = build_manager(config)

    try:
        manager.initialize()

        if selected_mode == "calibrate":
            run_calibration_mode(manager, settings_manager, config, output_dir=config.storage.output_dir)
            return True
        if selected_mode == "main":
            run_main_mode(
                manager,
                prefix=args.prefix,
                shutter_interval_s=args.interval,
                shot_count=args.shot_count,
            )
            return False
        if selected_mode == "single":
            manager.single_test_capture(prefix=args.prefix)
            return False
        if selected_mode == "trigger":
            manager.run_trigger_loop()
            return False

        raise ValueError(f"Unsupported mode: {selected_mode}")

    except KeyboardInterrupt:
        logging.info("[APP] Interrupted by user")
        return False
    finally:
        manager.shutdown()

def main() -> None:
    configure_logging()
    args = parse_args()

    settings_path = Path(args.settings)
    settings_manager = SettingsManager(settings_path)

    # Non-interactive modes keep the old behavior: run once, then exit.
    if args.mode != "prompt":
        run_camera_mode_once(args.mode, args, settings_path, settings_manager)
        return

    # Interactive prompt mode loops after calibration or save-directory changes.
    # Main, single, and trigger modes intentionally exit/continue as terminal modes.
    while True:
        selected_mode = choose_startup_mode()
        return_to_menu = run_camera_mode_once(selected_mode, args, settings_path, settings_manager)

        if not return_to_menu:
            break

        print("\nReturning to camera mode selection...")

if __name__ == "__main__":
    main()

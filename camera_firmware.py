import cv2
import time
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List


# Configuration

@dataclass
class CameraConfig:
    camera_index: int = 0
    preferred_resolutions: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (4640, 1456),  # Totem 180 panoramic 4.6K
            (3520, 1104),  # Totem 180 panoramic 3.5K
            (1920, 1080),  # Totem 180 full HD
        ]
    )
    capture_backend: Optional[int] = cv2.CAP_DSHOW  # Windows-friendly; set None if needed
    startup_warmup_s: float = 0.75
    flush_frames_on_startup: int = 5
    flush_frames_before_capture: int = 3
    autofocus: bool = False  # Totem 180 is fixed focus


@dataclass
class TriggerConfig:
    capture_threshold: float = 10.0
    trigger_debounce_s: float = 0.25
    polling_interval_s: float = 0.01
    shutter_interval_s: float = 1.0      # seconds between captures
    sequence_duration_s: float = 10.0    # total sequence duration sec


@dataclass
class StorageConfig:
    output_dir: str = "captures"
    image_extension: str = "png"
    save_metadata_json: bool = True
    filename_prefix: str = "capture"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


# Totem 180 has no programmable light path in this script,

class NoOpLightController:
    def initialize(self) -> None:
        logging.info("[LIGHT] No programmable light controller in use")

    def set_led(self, enabled: bool) -> None:
        logging.debug("[LIGHT] Ignoring LED request: %s", "ON" if enabled else "OFF")

    def get_led_state(self) -> bool:
        return False

    def shutdown(self) -> None:
        logging.info("[LIGHT] No programmable light controller to shut down")


# OpenCV camera capture

class OpenCVCamera:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.negotiated_resolution: Optional[Tuple[int, int]] = None
        self.negotiated_backend: Optional[int] = None

    def _open_capture(self) -> cv2.VideoCapture:
        if self.config.capture_backend is not None:
            cap = cv2.VideoCapture(self.config.camera_index, self.config.capture_backend)
        else:
            cap = cv2.VideoCapture(self.config.camera_index)

        if cap is None or not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        return cap

    def _try_resolution(self, cap: cv2.VideoCapture, width: int, height: int) -> Tuple[bool, Optional[Tuple[int, int]]]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        time.sleep(0.2)

        # Flush a few startup frames
        for _ in range(3):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None:
            return False, None

        actual_h, actual_w = frame.shape[:2]
        return True, (actual_w, actual_h)

    def open(self) -> None:
        logging.info("[CV] Opening camera")

        if self.config.capture_backend is not None:
            self.cap = cv2.VideoCapture(self.config.camera_index, self.config.capture_backend)
        else:
            self.cap = cv2.VideoCapture(self.config.camera_index)

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        # Start simple and stable
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        time.sleep(self.config.startup_warmup_s)

        # Flush a few startup frames
        for _ in range(self.config.flush_frames_on_startup):
            self.cap.read()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Camera opened but no usable frame was returned")

        actual_h, actual_w = frame.shape[:2]
        self.negotiated_resolution = (actual_w, actual_h)

        logging.info("[CV] Camera opened and warmed up")
        logging.info("[CV] Final negotiated resolution: %sx%s", actual_w, actual_h)
    
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


# Trigger / load cell interface

class LoadCellInterface:
    """
    Placeholder for your load-cell or sampling hardware.

    Replace read_value() with your real serial/DAQ/ADC code.
    """

    def initialize(self) -> None:
        logging.info("[LC] Initializing load cell interface")

    def read_value(self) -> float:
        return 12.34

    def should_capture(self, value: float, threshold: float) -> bool:
        return value >= threshold

    def shutdown(self) -> None:
        logging.info("[LC] Shutting down load cell interface")


# Capture / saving

class CaptureManager:
    def __init__(
        self,
        app_config: AppConfig,
        camera: OpenCVCamera,
        light: NoOpLightController,
        load_cell: LoadCellInterface,
    ) -> None:
        self.app_config = app_config
        self.camera = camera
        self.light = light
        self.load_cell = load_cell
        self.output_dir = Path(app_config.storage.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_capture_time: float = 0.0

    def initialize(self) -> None:
        logging.info("[APP] Initializing system")
        self.light.initialize()
        self.camera.open()
        self.load_cell.initialize()

    def shutdown(self) -> None:
        logging.info("[APP] Shutting down system")
        self.camera.close()
        self.load_cell.shutdown()
        self.light.shutdown()

    def _build_base_filename(self, prefix: str, value: float) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_prefix = prefix.replace(" ", "_")
        return f"{safe_prefix}_{timestamp}_load_{value:.3f}"

    def _build_output_path(self, base_filename: str) -> Path:
        return self.output_dir / f"{base_filename}.{self.app_config.storage.image_extension}"

    def _build_metadata_path(self, image_path: Path) -> Path:
        return image_path.with_suffix(".json")

    def _save_metadata(self, image_path: Path, value: float, frame_shape: Tuple[int, int, int]) -> Optional[Path]:
        if not self.app_config.storage.save_metadata_json:
            return None

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "load_value": value,
            "image_path": str(image_path),
            "camera_index": self.app_config.camera.camera_index,
            "requested_resolutions": self.app_config.camera.preferred_resolutions,
            "negotiated_resolution": self.camera.negotiated_resolution,
            "frame_shape": {
                "height": frame_shape[0],
                "width": frame_shape[1],
                "channels": frame_shape[2] if len(frame_shape) > 2 else 1,
            },
            "camera_model_notes": {
                "assumed_model": "IPEVO Totem 180",
                "focus_mode": "fixed_focus",
                "panoramic_camera": True,
            },
            "flush_frames_on_startup": self.app_config.camera.flush_frames_on_startup,
            "flush_frames_before_capture": self.app_config.camera.flush_frames_before_capture,
        }

        metadata_path = self._build_metadata_path(image_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logging.info("[APP] Metadata saved to %s", metadata_path)
        return metadata_path

    def _debounce_ok(self) -> bool:
        elapsed = time.time() - self.last_capture_time
        return elapsed >= self.app_config.trigger.trigger_debounce_s

    def capture_once(self, prefix: str = "capture") -> Tuple[Path, Optional[Path], float]:
        value = self.load_cell.read_value()
        base_filename = self._build_base_filename(prefix, value)
        output_path = self._build_output_path(base_filename)

        logging.info("[APP] Starting capture sequence (load value=%.4f)", value)

        self.camera.flush_frames(self.app_config.camera.flush_frames_before_capture)
        frame = self.camera.read_frame()

        self.camera.save_frame(frame, output_path)
        metadata_path = self._save_metadata(output_path, value, frame.shape)

        self.last_capture_time = time.time()

        logging.info("[APP] Capture saved to %s (load value=%.4f)", output_path, value)
        return output_path, metadata_path, value

    def capture_sequence(
        self,
        prefix: str = "sequence",
        shutter_interval_s: Optional[float] = None,
        sequence_duration_s: Optional[float] = None,
    ) -> list[Tuple[Path, Optional[Path], float]]:
        """
        Capture repeated images at a fixed shutter interval over a total duration.

        Example:
        - shutter_interval_s = 1.0
        - sequence_duration_s = 10.0

        This will attempt ~10 captures total, one every second.
        """
        interval = (
            shutter_interval_s
            if shutter_interval_s is not None
            else self.app_config.trigger.shutter_interval_s
        )
        duration = (
            sequence_duration_s
            if sequence_duration_s is not None
            else self.app_config.trigger.sequence_duration_s
        )

        if interval <= 0:
            raise ValueError("shutter_interval_s must be > 0")
        if duration <= 0:
            raise ValueError("sequence_duration_s must be > 0")

        logging.info(
            "[APP] Starting timed capture sequence: interval=%.3fs duration=%.3fs",
            interval,
            duration,
        )

        results: list[Tuple[Path, Optional[Path], float]] = []
        start_time = time.time()
        next_capture_time = start_time
        shot_index = 0

        while True:
            now = time.time()
            elapsed = now - start_time

            if elapsed >= duration:
                break

            sleep_time = next_capture_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)

            shot_prefix = f"{prefix}_shot_{shot_index:03d}"
            try:
                result = self.capture_once(prefix=shot_prefix)
                results.append(result)
                logging.info(
                    "[APP] Sequence shot %03d complete: image=%s",
                    shot_index,
                    result[0],
                )
            except Exception as exc:
                logging.exception("[APP] Sequence shot %03d failed: %s", shot_index, exc)

            shot_index += 1
            next_capture_time += interval

        logging.info(
            "[APP] Timed capture sequence finished: %d images attempted/saved",
            len(results),
        )
        return results

    def run_trigger_loop(self) -> None:
        logging.info("[APP] Entering trigger loop")
        threshold = self.app_config.trigger.capture_threshold

        while True:
            value = self.load_cell.read_value()

            if self.load_cell.should_capture(value, threshold) and self._debounce_ok():
                try:
                    prefix = f"load_{value:.2f}"
                    image_path, metadata_path, captured_value = self.capture_once(prefix=prefix)
                    logging.info(
                        "[APP] Completed capture: image=%s metadata=%s load=%.4f",
                        image_path,
                        metadata_path,
                        captured_value,
                    )
                except Exception as exc:
                    logging.exception("[APP] Capture failed: %s", exc)

            time.sleep(self.app_config.trigger.polling_interval_s)

    def single_test_capture(self, prefix: str = "manual_test") -> None:
        image_path, metadata_path, value = self.capture_once(prefix=prefix)
        print(f"Saved image: {image_path}")
        print(f"Saved metadata: {metadata_path}")
        print(f"Load value: {value}")
        print(f"Negotiated resolution: {self.camera.negotiated_resolution}")


# Main entry point

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> None:
        configure_logging()

        config = AppConfig(
            camera=CameraConfig(
                camera_index=1,
                preferred_resolutions=[(1920, 1080)],
                capture_backend=cv2.CAP_DSHOW,
                startup_warmup_s=1.0,
                flush_frames_on_startup=10,
                flush_frames_before_capture=5,
                autofocus=False,
            ),
            trigger=TriggerConfig(
                capture_threshold=10.0,
                trigger_debounce_s=0.25,
                polling_interval_s=0.01,
                shutter_interval_s=1.0,
                sequence_duration_s=10.0,
            ),
            storage=StorageConfig(
                output_dir="captures",
                image_extension="png",
                save_metadata_json=True,
                filename_prefix="capture",
            ),
        )

        light = NoOpLightController()
        camera = OpenCVCamera(config.camera)
        load_cell = LoadCellInterface()
        manager = CaptureManager(config, camera, light, load_cell)

        try:
            manager.initialize()

            # A) single test capture first
            results = manager.capture_sequence(
                prefix="totem180_sequence",
                shutter_interval_s=1.0,
                sequence_duration_s=10.0,
            )

            print(f"Saved {len(results)} images in shutter sequence")
            for image_path, metadata_path, value in results:
                print(f"Image: {image_path} | Metadata: {metadata_path} | Load: {value}")

            # B) then switch to continuous trigger loop when ready
            # manager.run_trigger_loop() - use when programming directly onto carousel

        except KeyboardInterrupt:
            logging.info("[APP] Interrupted by user")
        finally:
            manager.shutdown()


if __name__ == "__main__":
    main()
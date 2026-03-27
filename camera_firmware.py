import cv2
import time
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


# Configuration

@dataclass
class CameraConfig:
    camera_index: int = 0
    width: int = 3840
    height: int = 2160
    autofocus: bool = True
    startup_warmup_s: float = 0.5
    led_settle_s: float = 0.15
    flush_frames_after_led_on: int = 5
    capture_backend: Optional[int] = cv2.CAP_DSHOW  # Windows-friendly; set None if needed


@dataclass
class TriggerConfig:
    capture_threshold: float = 10.0
    trigger_debounce_s: float = 0.25
    polling_interval_s: float = 0.01


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


# SDK light control

class FirmwareLightController:
    """
    Placeholder controller for vendor firmware/API light control.

    Replace the methods in this class with actual vendor SDK calls
    once you know how the camera exposes LED control.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._led_state = False

    def initialize(self) -> None:
        logging.info("[FW] Initializing firmware/API controller")
        self._initialized = True

    def set_led(self, enabled: bool) -> None:
        """
        Replace this with the real firmware/API command.

        Sample future pattern:
            vendor_sdk.set_led(1 if enabled else 0)
        """
        if not self._initialized:
            raise RuntimeError("FirmwareLightController not initialized")

        self._led_state = enabled
        state = "ON" if enabled else "OFF"
        logging.info("[FW] LED set to %s", state)

    def get_led_state(self) -> bool:
        return self._led_state

    def shutdown(self) -> None:
        if self._initialized:
            try:
                self.set_led(False)
            except Exception as exc:
                logging.warning("[FW] Failed to force LED OFF during shutdown: %s", exc)

        logging.info("[FW] Shutting down firmware/API controller")
        self._initialized = False


# OpenCV camera capture

class OpenCVCamera:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        logging.info("[CV] Opening camera")

        if self.config.capture_backend is not None:
            self.cap = cv2.VideoCapture(self.config.camera_index, self.config.capture_backend)
        else:
            self.cap = cv2.VideoCapture(self.config.camera_index)

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        if self.config.autofocus:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        else:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # May be ignored silently by some drivers, which is normal
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(self.config.startup_warmup_s)
        logging.info("[CV] Camera opened and warmed up")

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
        """
        Replace this with your actual load-cell read.

        Examples:
        - serial line parse
        - NI-DAQ read
        - HX711 / ADC read
        """
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
        light: FirmwareLightController,
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
        try:
            self.light.set_led(False)
        except Exception as exc:
            logging.warning("[APP] Failed to set LED OFF during shutdown: %s", exc)

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

    def _save_metadata(self, image_path: Path, value: float) -> Optional[Path]:
        if not self.app_config.storage.save_metadata_json:
            return None

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "load_value": value,
            "image_path": str(image_path),
            "camera_index": self.app_config.camera.camera_index,
            "resolution": {
                "width": self.app_config.camera.width,
                "height": self.app_config.camera.height,
            },
            "autofocus": self.app_config.camera.autofocus,
            "led_settle_s": self.app_config.camera.led_settle_s,
            "flush_frames_after_led_on": self.app_config.camera.flush_frames_after_led_on,
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
        """
        Full sequence:
        1. Read load value
        2. Turn LED on via firmware/API
        3. Wait for lighting to settle
        4. Flush stale frames
        5. Capture frame
        6. Turn LED off
        7. Save image
        8. Save metadata JSON
        """
        value = self.load_cell.read_value()
        base_filename = self._build_base_filename(prefix, value)
        output_path = self._build_output_path(base_filename)

        logging.info("[APP] Starting capture sequence (load value=%.4f)", value)

        self.light.set_led(True)
        try:
            time.sleep(self.app_config.camera.led_settle_s)

            self.camera.flush_frames(self.app_config.camera.flush_frames_after_led_on)
            frame = self.camera.read_frame()

        finally:
            # Make sure light is turned off even if capture fails
            self.light.set_led(False)

        self.camera.save_frame(frame, output_path)
        metadata_path = self._save_metadata(output_path, value)

        self.last_capture_time = time.time()

        logging.info("[APP] Capture saved to %s (load value=%.4f)", output_path, value)
        return output_path, metadata_path, value

    def run_trigger_loop(self) -> None:
        """
        Poll the load cell and capture one image when threshold is met.
        """
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
            camera_index=0,
            width=3840,
            height=2160,
            autofocus=True,
            startup_warmup_s=0.5,
            led_settle_s=0.15,
            flush_frames_after_led_on=5,
            capture_backend=cv2.CAP_DSHOW,  # use None if this causes issues
        ),
        trigger=TriggerConfig(
            capture_threshold=10.0,
            trigger_debounce_s=0.25,
            polling_interval_s=0.01,
        ),
        storage=StorageConfig(
            output_dir="captures",
            image_extension="png",
            save_metadata_json=True,
            filename_prefix="capture",
        ),
    )

    light = FirmwareLightController()
    camera = OpenCVCamera(config.camera)
    load_cell = LoadCellInterface()
    manager = CaptureManager(config, camera, light, load_cell)

    try:
        manager.initialize()

        # A) single test capture
        # image_path, metadata_path, value = manager.capture_once(prefix="manual_test")
        # print(f"Saved image: {image_path}")
        # print(f"Saved metadata: {metadata_path}")
        # print(f"Load value: {value}")

        # B) continuous trigger loop
        manager.run_trigger_loop()

    except KeyboardInterrupt:
        logging.info("[APP] Interrupted by user")
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
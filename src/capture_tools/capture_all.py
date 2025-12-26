from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .orbbec import OrbbecHandler
from .realsense import RealsenseHandler


@dataclass(frozen=True)
class CameraKey:
    device_type: str
    serial_number: str

    def as_str(self) -> str:
        return f"{self.device_type}:{self.serial_number}"


class CaptureAll:
    def __init__(self) -> None:
        self.orbbec_handlers: List[OrbbecHandler] = []
        self.realsense_handlers: List[RealsenseHandler] = []
        self._started = False

    def _discover_orbbec_devices(self) -> List[object]:
        try:
            import pyorbbecsdk as ob
        except Exception as exc:
            print(f"[capture_all] Orbbec SDK 未就绪: {exc}")
            return []

        try:
            ctx = ob.Context()
            device_list = ctx.query_devices()
            if hasattr(device_list, "get_count"):
                count = int(device_list.get_count())
            elif hasattr(device_list, "get_device_count"):
                count = int(device_list.get_device_count())
            else:
                count = 0

            devices: List[object] = []
            for i in range(count):
                if hasattr(device_list, "get_device_by_index"):
                    device = device_list.get_device_by_index(i)
                elif hasattr(device_list, "get_device"):
                    device = device_list.get_device(i)
                else:
                    device = device_list[i]
                devices.append(device)
            return devices
        except Exception as exc:
            print(f"[capture_all] 枚举 Orbbec 设备失败: {exc}")
            return []

    def _discover_realsense_serials(self) -> List[str]:
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            print(f"[capture_all] RealSense SDK 未就绪: {exc}")
            return []

        try:
            ctx = rs.context()
            serials: List[str] = []
            for dev in ctx.query_devices():
                try:
                    serials.append(dev.get_info(rs.camera_info.serial_number))
                except Exception:
                    continue
            return serials
        except Exception as exc:
            print(f"[capture_all] 枚举 RealSense 设备失败: {exc}")
            return []

    @staticmethod
    def _load_serial_to_index_mapping() -> dict[str, int]:
        try:
            from . import camera_serial_index
        except Exception:
            return {}

        serial_to_index: dict[str, int] = {}
        for serial, camera_name in dict(camera_serial_index).items():
            if not isinstance(serial, str) or not isinstance(camera_name, str):
                continue
            if not camera_name.startswith("camera_"):
                continue
            try:
                serial_to_index[serial] = int(camera_name.split("_", 1)[1])
            except Exception:
                continue
        return serial_to_index

    def start_all(self) -> None:
        if self._started:
            return

        try:
            from . import orbbec as orbbec_mod

            orbbec_mod.keep_running = True
        except Exception:
            pass

        try:
            from . import realsense as realsense_mod

            realsense_mod.keep_running = True
        except Exception:
            pass

        self.orbbec_handlers.clear()
        self.realsense_handlers.clear()

        serial_to_index = self._load_serial_to_index_mapping()
        next_index = max(serial_to_index.values(), default=-1) + 1

        for dev in self._discover_orbbec_devices():
            try:
                serial = dev.get_device_info().get_serial_number()
            except Exception:
                serial = None

            index = serial_to_index.get(serial, next_index) if serial else next_index
            if serial not in serial_to_index:
                next_index += 1

            try:
                handler = OrbbecHandler(dev, index)
                handler.start()
                self.orbbec_handlers.append(handler)
            except Exception as exc:
                print(f"[capture_all] 启动 Orbbec 失败: {exc}")

        for sn in self._discover_realsense_serials():
            index = serial_to_index.get(sn, next_index)
            if sn not in serial_to_index:
                next_index += 1

            try:
                handler = RealsenseHandler(sn, index)
                handler.start()
                self.realsense_handlers.append(handler)
            except Exception as exc:
                print(f"[capture_all] 启动 RealSense 失败: {exc}")

        self._started = True

    def stop_all(self) -> None:
        if not self._started:
            return

        try:
            from . import orbbec as orbbec_mod

            orbbec_mod.keep_running = False
        except Exception:
            pass

        try:
            from . import realsense as realsense_mod

            realsense_mod.keep_running = False
        except Exception:
            pass

        for handler in self.orbbec_handlers:
            try:
                handler.stop()
            except Exception:
                pass

        for handler in self.realsense_handlers:
            try:
                handler.stop()
            except Exception:
                pass

        self.orbbec_handlers.clear()
        self.realsense_handlers.clear()
        self._started = False

    def get_latest_frames_all(self) -> Dict[str, Tuple[Optional[object], Optional[object]]]:
        out: Dict[str, Tuple[Optional[object], Optional[object]]] = {}

        for handler in self.orbbec_handlers:
            key = CameraKey("orbbec", getattr(handler, "serial_number", "unknown")).as_str()
            try:
                out[key] = handler.get_latest_frames()
            except Exception:
                out[key] = (None, None)

        for handler in self.realsense_handlers:
            key = CameraKey("realsense", getattr(handler, "serial_number", "unknown")).as_str()
            try:
                out[key] = handler.get_latest_data()
            except Exception:
                out[key] = (None, None)

        return out


__all__ = ["CaptureAll", "CameraKey"]


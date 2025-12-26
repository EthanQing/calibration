from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np
from pyorbbecsdk import *  # noqa: F403

keep_running = True


def frame_to_bgr_image(frame: VideoFrame) -> Optional[np.ndarray]:  # noqa: F405
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()

    try:
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    except Exception:
        data = np.ascontiguousarray(np.asanyarray(frame.get_data()), dtype=np.uint8).reshape(-1)

    image = None
    try:
        if color_format == OBFormat.RGB:  # noqa: F405
            image = data.reshape((height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_format == OBFormat.BGR:  # noqa: F405
            image = data.reshape((height, width, 3))
        elif color_format == OBFormat.YUYV:  # noqa: F405
            image = data.reshape((height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        elif color_format == OBFormat.MJPG:  # noqa: F405
            image = cv2.imdecode(np.ascontiguousarray(data), cv2.IMREAD_COLOR)
        elif color_format == OBFormat.UYVY:  # noqa: F405
            image = data.reshape((height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    except Exception as exc:
        print(f"图像解码失败: {exc}")
        return None

    if image is None:
        return None
    return np.ascontiguousarray(image)


class OrbbecHandler:
    def __init__(self, device, index: int) -> None:
        self.device = device
        self.index = int(index)
        self.serial_number = device.get_device_info().get_serial_number()
        self.pipeline = Pipeline(device)  # noqa: F405
        self.config = Config()  # noqa: F405

        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

        self.latest_color_frame: Optional[np.ndarray] = None
        self.latest_depth_raw: Optional[np.ndarray] = None
        self.latest_depth_vis: Optional[np.ndarray] = None
        self.latest_depth_scale: Optional[float] = None

        self.is_streaming = False

    def setup_pipeline(self) -> None:
        # Color stream
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)  # noqa: F405
            if profile_list is not None:
                color_profile = None

                try:
                    width = 1920
                    height = 1080
                    color_profile = profile_list.get_video_stream_profile(width, height, OBFormat.MJPG, 5)  # noqa: F405
                except OBError:  # noqa: F405
                    try:
                        color_profile = profile_list.get_profile(0).get_default_video_stream_profile()
                        width = color_profile.get_width()
                        height = color_profile.get_height()
                        fmt = color_profile.get_format()
                        color_profile = profile_list.get_video_stream_profile(width, height, fmt, 5)
                    except OBError:  # noqa: F405
                        color_profile = profile_list.get_default_video_stream_profile()

                if color_profile is not None:
                    self.config.enable_stream(color_profile)
        except OBError as exc:  # noqa: F405
            print(f"[{self.index}] 配置彩色流失败: {exc}")

        # Depth stream
        try:
            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)  # noqa: F405
            if depth_profile_list is not None:
                depth_profile = depth_profile_list.get_default_video_stream_profile()
                try:
                    w = 1280
                    h = 800
                    fmt = depth_profile.get_format()
                    depth_profile = depth_profile_list.get_video_stream_profile(w, h, fmt, 5)
                except OBError:  # noqa: F405
                    print(f"[{self.index}] 使用默认深度配置")
                self.config.enable_stream(depth_profile)
        except OBError as exc:  # noqa: F405
            print(f"[{self.index}] 配置深度流失败: {exc}")

    def start(self) -> None:
        self.setup_pipeline()
        try:
            self.pipeline.start(self.config)
            self.is_streaming = True
            self.thread = threading.Thread(target=self.process_loop, name=f"orbbec-{self.serial_number}", daemon=True)
            self.thread.start()
            print(f"[{self.index}] 流已启动: SN={self.serial_number}")
        except OBError as exc:  # noqa: F405
            print(f"[{self.index}] 启动流失败: {exc}")
            self.is_streaming = False

    def process_loop(self) -> None:
        global keep_running

        while keep_running and self.is_streaming:
            try:
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                color_image = None
                if color_frame is not None:
                    color_image = frame_to_bgr_image(color_frame)

                depth_raw = None
                depth_vis = None
                depth_scale = None
                if depth_frame is not None:
                    width = depth_frame.get_width()
                    height = depth_frame.get_height()
                    depth_scale = float(depth_frame.get_depth_scale())

                    data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    depth_raw = data.reshape((height, width)).copy()

                    depth_mm = depth_raw.astype(np.float32) * depth_scale
                    depth_mm = np.clip(depth_mm, 0, 5000.0)
                    depth_8u = (depth_mm / 5000.0 * 255.0).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

                with self.lock:
                    if color_image is not None:
                        self.latest_color_frame = color_image
                    if depth_vis is not None:
                        self.latest_depth_vis = depth_vis
                        self.latest_depth_raw = depth_raw
                        self.latest_depth_scale = depth_scale

            except OBError:  # noqa: F405
                continue
            except Exception as exc:
                print(f"[{self.index}] 线程异常: {exc}")
                break

    def stop(self) -> None:
        self.is_streaming = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
            print(f"[{self.index}] 流已停止: SN={self.serial_number}")
        except Exception:
            pass

    def get_latest_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            color = None if self.latest_color_frame is None else self.latest_color_frame.copy()
            depth_vis = None if self.latest_depth_vis is None else self.latest_depth_vis.copy()
        return color, depth_vis

    def get_latest_data(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        with self.lock:
            color = None if self.latest_color_frame is None else self.latest_color_frame.copy()
            depth_raw = None if self.latest_depth_raw is None else self.latest_depth_raw.copy()
            depth_vis = None if self.latest_depth_vis is None else self.latest_depth_vis.copy()
            scale = self.latest_depth_scale
        return color, depth_raw, depth_vis, scale


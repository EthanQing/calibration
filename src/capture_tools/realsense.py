from __future__ import annotations

import threading
import traceback
from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs

keep_running = True


class RealsenseHandler:
    def __init__(self, serial_number: str, index: int) -> None:
        self.serial_number = str(serial_number)
        self.index = int(index)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

        self.latest_color_image: Optional[np.ndarray] = None
        self.latest_depth_raw: Optional[np.ndarray] = None
        self.depth_scale: float | None = None

        self.is_streaming = False

    def setup_pipeline(self) -> None:
        self.config.enable_device(self.serial_number)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 5)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 5)

    def start(self) -> None:
        try:
            self.setup_pipeline()
            pipeline_profile = self.pipeline.start(self.config)

            depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())
            print(f"[{self.index}] RealSense 已启动: SN={self.serial_number} depth_scale={self.depth_scale}")

            self.is_streaming = True
            self.thread = threading.Thread(target=self.process_loop, name=f"realsense-{self.serial_number}", daemon=True)
            self.thread.start()

        except Exception as exc:
            print(f"[{self.index}] 启动 RealSense 失败: {exc}")
            traceback.print_exc()

    def process_loop(self) -> None:
        global keep_running

        while keep_running and self.is_streaming:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                with self.lock:
                    self.latest_depth_raw = depth_image.copy()
                    self.latest_color_image = color_image.copy()

            except RuntimeError as exc:
                # wait_for_frames timeout raises RuntimeError
                print(f"[{self.index}] 等待帧超时: {exc}")
            except Exception as exc:
                print(f"[{self.index}] 线程异常: {exc}")
                break

    def stop(self) -> None:
        self.is_streaming = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
            print(f"[{self.index}] RealSense 已停止: SN={self.serial_number}")
        except Exception:
            pass

    def get_latest_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            if self.latest_color_image is None or self.latest_depth_raw is None:
                return None, None
            return self.latest_color_image.copy(), self.latest_depth_raw.copy()


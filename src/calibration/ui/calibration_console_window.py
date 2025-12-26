"""Calibration console window recreated from the provided React dashboard layout."""

from __future__ import annotations

import math
import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from PySide6 import QtCore, QtGui, QtWidgets

from .aruco_rig_calibration import calibrate_rig_from_latest_capture
from .calibration_controller import CameraExtrinsic, Matrix4x4, RigCalibrationController, RigCalibrationResult
from .pose_view import PoseCamera, PoseView
from .sidebar.camera_rig_panel import CameraRigPanel, RigCamera
from .sidebar.camera_telemetry_panel import CameraPose, CameraTelemetryPanel


@dataclass
class CameraDescriptor:
    camera_id: str
    group: str
    x: float
    y: float
    z: float
    angle: float
    status: str
    overlap: int
    connection: str = "USB 3.0"
    resolution: str = "1280x720"
    reprojection: float = 0.42

    def to_metadata(self) -> Dict[str, str]:
        group_names = {"Top": "顶部层", "Mid": "中间层", "Bot": "底部层"}
        return {
            "id": self.camera_id,
            "group": group_names.get(self.group, self.group),
            "status": self.status,
            "connection": self.connection,
            "resolution": self.resolution,
        }

    def to_pose(self) -> CameraPose:
        return CameraPose(self.x, self.y, self.z, self.angle)

    def to_rig_camera(self) -> RigCamera:
        return RigCamera(self.camera_id, self.group, self.status)

    def to_pose_camera(self) -> PoseCamera:
        return PoseCamera(self.camera_id, self.x, self.y, self.z, self.angle, self.status)


class StreamCard(QtWidgets.QFrame):
    """Placeholder panel for RGB / depth streams."""

    def __init__(self, title: str, accent: str, placeholder: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._accent = accent
        self._placeholder = placeholder
        self._camera_id = "--"
        self._build_ui()

    def _build_ui(self) -> None:
        self.setObjectName("StreamCard")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QtWidgets.QFrame()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(12, 12, 12, 12)
        header_layout.setSpacing(6)

        indicator = QtWidgets.QLabel()
        indicator.setFixedSize(10, 10)
        indicator.setStyleSheet(f"border-radius:5px; background-color:{self._accent};")
        label = QtWidgets.QLabel()
        label.setObjectName("StreamCardTitle")
        header_layout.addWidget(indicator)
        header_layout.addWidget(label)
        header_layout.addStretch()
        layout.addWidget(header)

        body = QtWidgets.QFrame()
        body.setObjectName("StreamCardBody")
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        viewer = QtWidgets.QLabel(self._placeholder)
        viewer.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        viewer.setObjectName("StreamCardPlaceholder")
        viewer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        viewer.setMinimumHeight(240)
        body_layout.addWidget(viewer, 1)
        layout.addWidget(body, 1)

        self._title_label = label
        self._viewer = viewer
        self._pixmap: QtGui.QPixmap | None = None
        self._update_title()

        self.setStyleSheet(
            """
            QFrame#StreamCard {
                border: 1px solid rgba(15,23,42,0.12);
                border-radius: 14px;
                background-color: rgba(255,255,255,0.08);
            }
            QLabel#StreamCardTitle {
                color: #0f172a;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#StreamCardPlaceholder {
                color: rgba(15,23,42,0.45);
                font-size: 12px;
                font-family: 'JetBrains Mono', monospace;
            }
            QFrame#StreamCardBody {
                background-color: rgba(255,255,255,0.4);
                border-bottom-left-radius: 14px;
                border-bottom-right-radius: 14px;
            }
            """
        )

    def set_camera_id(self, camera_id: str) -> None:
        self._camera_id = camera_id
        self._update_title()

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        if self._pixmap is None:
            self._viewer.setText(text)

    def set_image(self, image: QtGui.QImage | None) -> None:
        if image is None:
            self._pixmap = None
            self._viewer.setPixmap(QtGui.QPixmap())
            self._viewer.setText(self._placeholder)
            return

        self._pixmap = QtGui.QPixmap.fromImage(image)
        self._viewer.setText("")
        self._render_pixmap()

    def _render_pixmap(self) -> None:
        if self._pixmap is None:
            return
        size = self._viewer.size()
        scaled = self._pixmap.scaled(
            size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._viewer.setPixmap(scaled)

    def _update_title(self) -> None:
        self._title_label.setText(f"{self._title} — {self._camera_id}")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render_pixmap()


def _make_transform(x: float, y: float, z: float, angle_deg: float) -> Matrix4x4:
    theta = -math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return (
        (cos_t, 0.0, sin_t, x),
        (0.0, 1.0, 0.0, y),
        (-sin_t, 0.0, cos_t, z),
        (0.0, 0.0, 0.0, 1.0),
    )


class LiveStreamsPage(QtWidgets.QWidget):
    """Page that shows the selected camera's RGB/depth streams."""

    FrameProvider = Callable[[str], tuple[QtGui.QImage | None, QtGui.QImage | None]]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        self._rgb_card = StreamCard("彩色视频流", "#ef4444", "[ 彩色信号传输中 ]")
        self._depth_card = StreamCard("深度映射", "#60a5fa", "[ 深度传感器工作中 ]")
        layout.addWidget(self._rgb_card, 1)
        layout.addWidget(self._depth_card, 1)

        self._selected_camera_id: str | None = None
        self._frame_provider: LiveStreamsPage.FrameProvider | None = None

        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setInterval(66)
        self._refresh_timer.timeout.connect(self._refresh_frames)
        self._refresh_timer.start()

    def set_frame_provider(self, provider: FrameProvider | None) -> None:
        self._frame_provider = provider
        self._refresh_frames()

    def update_camera(self, camera_id: str) -> None:
        self._selected_camera_id = camera_id
        self._rgb_card.set_camera_id(camera_id)
        self._depth_card.set_camera_id(camera_id)
        self._refresh_frames()

    def _refresh_frames(self) -> None:
        if not self._selected_camera_id:
            self._rgb_card.set_placeholder("[ 请选择相机 ]")
            self._depth_card.set_placeholder("[ 请选择相机 ]")
            self._rgb_card.set_image(None)
            self._depth_card.set_image(None)
            return

        if self._frame_provider is None:
            self._rgb_card.set_placeholder("[ 采集模块未启动 ]")
            self._depth_card.set_placeholder("[ 采集模块未启动 ]")
            self._rgb_card.set_image(None)
            self._depth_card.set_image(None)
            return

        try:
            rgb_image, depth_image = self._frame_provider(self._selected_camera_id)
        except Exception:
            self._rgb_card.set_placeholder("[ 获取帧失败 ]")
            self._depth_card.set_placeholder("[ 获取帧失败 ]")
            self._rgb_card.set_image(None)
            self._depth_card.set_image(None)
            return

        if rgb_image is None:
            self._rgb_card.set_placeholder("[ 等待彩色帧... ]")
        self._rgb_card.set_image(rgb_image)

        if depth_image is None:
            self._depth_card.set_placeholder("[ 等待深度帧... ]")
        self._depth_card.set_image(depth_image)


class TabbedCenterPanel(QtWidgets.QWidget):
    """Hosts the tab buttons and the stacked content."""

    tab_changed = QtCore.Signal(int)

    def __init__(self, pose_view: PoseView, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._pose_view = pose_view
        self._live_page = LiveStreamsPage()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs_bar = QtWidgets.QFrame()
        bar_layout = QtWidgets.QHBoxLayout(self._tabs_bar)
        bar_layout.setContentsMargins(16, 12, 16, 12)
        bar_layout.setSpacing(8)

        self._button_group = QtWidgets.QButtonGroup(self)
        self._button_group.setExclusive(True)

        self._live_button = self._create_tab_button("实时画面（RGB/深度）")
        self._pose_button = self._create_tab_button("3D 位姿视图")
        self._live_button.setChecked(True)
        self._button_group.addButton(self._live_button, 0)
        self._button_group.addButton(self._pose_button, 1)
        bar_layout.addWidget(self._live_button)
        bar_layout.addWidget(self._pose_button)
        bar_layout.addStretch()
        layout.addWidget(self._tabs_bar)

        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._live_page)
        self._stack.addWidget(self._pose_view)
        layout.addWidget(self._stack, 1)

        self._live_button.toggled.connect(lambda checked: checked and self._set_index(0))
        self._pose_button.toggled.connect(lambda checked: checked and self._set_index(1))
        self._set_styles()

    def _create_tab_button(self, text: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setCheckable(True)
        button.setObjectName("CenterTabButton")
        button.setStyleSheet(
            """
            QPushButton#CenterTabButton {
                border-radius: 10px;
                border: 1px solid rgba(15,23,42,0.08);
                padding: 10px 20px;
                color: #0f172a;
                font-weight: 600;
                font-size: 12px;
                background-color: rgba(255,255,255,0.7);
            }
            QPushButton#CenterTabButton:hover {
                background-color: rgba(255,255,255,0.9);
            }
            QPushButton#CenterTabButton:checked {
                background-color: #a78bfa;
                color: white;
                border-color: rgba(15,23,42,0.15);
            }
            """
        )
        return button

    def _set_index(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        self.tab_changed.emit(index)

    def _set_styles(self) -> None:
        self._tabs_bar.setStyleSheet(
            "background-color: rgba(255,255,255,0.2);"
            "border-bottom: 1px solid rgba(15,23,42,0.08);"
        )

    @property
    def live_page(self) -> LiveStreamsPage:
        return self._live_page


class CalibrationConsoleWindow(QtWidgets.QMainWindow):
    """Main window composed of left rig panel, center tabs, right telemetry."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HexaCalib 控制台")
        self.resize(1500, 860)
        self._cameras = self._generate_cameras()
        self._camera_lookup: Dict[str, CameraDescriptor] = {cam.camera_id: cam for cam in self._cameras}
        self._selected_camera_id = self._cameras[0].camera_id if self._cameras else None
        self._calibration_stage = "idle"

        self._pose_view = PoseView()
        self._pose_view.set_calibration_stage(self._calibration_stage)
        self._pose_view.set_cameras([cam.to_pose_camera() for cam in self._cameras])

        self._capture_lock = threading.Lock()
        self._capture_all = None
        self._camera_index_to_serial = self._load_camera_index_mapping()
        self._camera_params_by_index = self._load_camera_params_best()

        self._build_layout()
        if self._selected_camera_id:
            self._update_camera_selection(self._selected_camera_id)

        self._tab_panel.live_page.set_frame_provider(self._get_live_stream_images)
        self._start_capture_async()

    @staticmethod
    def _load_camera_params_best(path: Path = Path("configurations") / "camera_params_best.json") -> dict[int, dict]:
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            return {}

        try:
            data = json.loads(raw)
        except Exception:
            return {}

        if not isinstance(data, list):
            return {}

        out: dict[int, dict] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("index"))
            except Exception:
                continue
            out[idx] = entry
        return out

    def _build_layout(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._rig_panel = CameraRigPanel([cam.to_rig_camera() for cam in self._cameras])
        self._rig_panel.setFixedWidth(260)
        layout.addWidget(self._rig_panel)

        center_container = QtWidgets.QWidget()
        center_container.setObjectName("CenterContainer")
        center_layout = QtWidgets.QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        self._tab_panel = TabbedCenterPanel(self._pose_view)
        center_layout.addWidget(self._tab_panel)
        layout.addWidget(center_container, 1)

        self._telemetry_panel = CameraTelemetryPanel()
        self._telemetry_panel.setFixedWidth(300)
        layout.addWidget(self._telemetry_panel)

        self._calibration_controller = RigCalibrationController(calibrate_rig_from_latest_capture, self)

        self._rig_panel.camera_selected.connect(self._update_camera_selection)
        self._telemetry_panel.recalc_requested.connect(self._handle_recalc_requested)
        self._telemetry_panel.capture_requested.connect(self._handle_capture_requested)
        self._telemetry_panel.capture_all_requested.connect(self._handle_capture_all_requested)
        self._tab_panel.tab_changed.connect(self._handle_tab_changed)
        self._calibration_controller.calibration_started.connect(self._on_calibration_started)
        self._calibration_controller.calibration_failed.connect(self._on_calibration_failed)
        self._calibration_controller.calibration_completed.connect(self._on_calibration_completed)

        center_container.setStyleSheet(
            "#CenterContainer {"
            " background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1,"
            " stop:0 rgba(255,255,255,0.75), stop:1 rgba(236,239,244,0.9));"
            " }"
        )

    def _generate_cameras(self) -> List[CameraDescriptor]:
        cameras: List[CameraDescriptor] = []
        R = 1.5
        H_TOP, H_MID, H_BOT = 1.8, 0.0, -1.8

        def hex_position(index: int, height: float) -> tuple[float, float, float, float]:
            angle_deg = 60 * index
            angle_rad = math.radians(angle_deg)
            return (
                math.cos(angle_rad) * R,
                height,
                math.sin(angle_rad) * R,
                angle_deg,
            )

        layer_specs = [
            ("Top", H_TOP, [0, 2, 3, 5], [2, 5, 8, 11]),
            ("Mid", H_MID, [0, 1, 2, 3, 4, 5], [1, 4, 7, 10, 12, 13]),
            ("Bot", H_BOT, [0, 2, 3, 5], [0, 3, 6, 9]),
        ]

        for group, height, hex_indices, camera_indices in layer_specs:
            for hex_idx, cam_idx in zip(hex_indices, camera_indices):
                x, y, z, angle = hex_position(hex_idx, height)
                cam_label = self._format_camera_label(cam_idx)
                cameras.append(
                    CameraDescriptor(
                        cam_label,
                        group,
                        x,
                        y,
                        z,
                        angle,
                        "ok",
                        12,
                        reprojection=0.5,
                    )
                )

        return cameras

    def _update_camera_selection(self, camera_id: str) -> None:
        if camera_id not in self._camera_lookup:
            return
        self._selected_camera_id = camera_id
        self._rig_panel.select_camera(camera_id)
        camera = self._camera_lookup[camera_id]
        metadata = camera.to_metadata()
        pose = camera.to_pose()

        idx = self._parse_camera_index(camera_id)
        params = self._camera_params_by_index.get(idx, {}) if idx is not None else {}

        serial = self._camera_index_to_serial.get(idx) if idx is not None else None
        if not serial and isinstance(params, dict):
            serial = params.get("serial_number")

        metadata["serial"] = str(serial) if serial else "--"

        handler = self._find_handler(serial) if isinstance(serial, str) else None
        if handler is None and serial:
            metadata["status"] = "error"

        device_type = "未知"
        if handler is not None:
            device_type = "Orbbec" if hasattr(handler, "device") else "RealSense"
        elif isinstance(serial, str):
            if serial.upper().startswith("AY"):
                device_type = "Orbbec"
            elif serial.isdigit():
                device_type = "RealSense"
        metadata["device_type"] = device_type

        def fmt_res(res: object) -> str | None:
            if not isinstance(res, dict):
                return None
            try:
                w = int(res.get("width"))
                h = int(res.get("height"))
            except Exception:
                return None
            if w <= 0 or h <= 0:
                return None
            return f"{w}x{h}"

        rgb_res = None
        depth_res = None
        if isinstance(params, dict):
            rgb_res = fmt_res(params.get("rgb_info", {}).get("best_resolution"))
            depth_res = fmt_res(params.get("depth_info", {}).get("best_resolution"))

        if rgb_res and depth_res:
            metadata["resolution"] = f"{rgb_res} / {depth_res}"
        elif rgb_res:
            metadata["resolution"] = rgb_res
        elif depth_res:
            metadata["resolution"] = depth_res

        intrinsics = None
        if isinstance(params, dict):
            rgb_intr = params.get("rgb_info", {}).get("intrinsics")
            if isinstance(rgb_intr, dict):
                try:
                    fx = float(rgb_intr.get("fx"))
                    fy = float(rgb_intr.get("fy"))
                    cx = float(rgb_intr.get("cx"))
                    cy = float(rgb_intr.get("cy"))
                except Exception:
                    fx = fy = cx = cy = 0.0
                if fx > 0 and fy > 0:
                    intrinsics = {
                        "k_00": fx,
                        "k_02": cx,
                        "k_11": fy,
                        "k_12": cy,
                        "k_22": 1.0,
                    }

        self._telemetry_panel.update_camera(metadata, pose, intrinsics, camera.reprojection)
        self._pose_view.set_selected_camera(camera_id)
        self._tab_panel.live_page.update_camera(camera_id)

    def _handle_recalc_requested(self) -> None:
        self._calibration_controller.trigger_calibration()

    def _handle_capture_requested(self) -> None:
        camera_id = self._selected_camera_id
        if not camera_id:
            QtWidgets.QMessageBox.information(self, "采集图像", "请先选择相机。")
            return

        folder, time_suffix = self._create_capture_folder()
        saved = self._capture_camera_to_folder(camera_id, folder, time_suffix)
        if not saved:
            QtWidgets.QMessageBox.warning(self, "采集图像", "当前相机没有可用帧，请稍后重试。")
            return

        QtWidgets.QMessageBox.information(self, "采集图像", f"已保存到:\n{folder}")

    def _handle_capture_all_requested(self) -> None:
        folder, time_suffix = self._create_capture_folder()
        camera_ids = [cam.camera_id for cam in self._cameras]

        saved = 0
        skipped: List[str] = []
        for camera_id in camera_ids:
            ok = self._capture_camera_to_folder(camera_id, folder, time_suffix)
            if ok:
                saved += 1
            else:
                skipped.append(camera_id)

        if saved == 0:
            QtWidgets.QMessageBox.warning(self, "采集图像", "没有任何相机采集成功（可能还未出帧）。")
            return

        message = f"已采集 {saved}/{len(camera_ids)} 台相机。\n保存目录:\n{folder}"
        if skipped:
            message += f"\n\n未采集成功: {', '.join(skipped)}"
        QtWidgets.QMessageBox.information(self, "采集图像", message)

    def _set_calibration_stage(self, stage: str) -> None:
        self._calibration_stage = stage
        self._pose_view.set_calibration_stage(stage)

    def _handle_tab_changed(self, index: int) -> None:
        if index == 1:
            self._pose_view.update()

    @staticmethod
    def _format_camera_label(camera_index: int) -> str:
        return f"CAM-{camera_index + 1:02d}"

    def _build_fake_calibration_result(self) -> RigCalibrationResult:
        """Generate a synthetic calibration result for the UI shell."""

        extrinsics: Dict[str, CameraExtrinsic] = {}
        transforms: Dict[str, Matrix4x4] = {}

        for descriptor in self._cameras:
            marker_count = random.randint(0, 12)
            reprojection = 0.35 + random.random() * 0.9
            if marker_count < 4:
                reprojection += 1.0 + random.random() * 2.4
            if random.random() < 0.05:
                reprojection += 2.5

            extrinsics[descriptor.camera_id] = CameraExtrinsic(
                marker_count=marker_count,
                reprojection_error=reprojection,
            )

            dx = (random.random() - 0.5) * 0.06
            dy = (random.random() - 0.5) * 0.04
            dz = (random.random() - 0.5) * 0.06
            d_angle = (random.random() - 0.5) * 8.0
            transforms[descriptor.camera_id] = _make_transform(
                descriptor.x + dx,
                descriptor.y + dy,
                descriptor.z + dz,
                (descriptor.angle + d_angle) % 360.0,
            )

        return RigCalibrationResult(extrinsics=extrinsics, relative_transforms=transforms)

    def _on_calibration_started(self) -> None:
        self._set_calibration_stage("optimizing")

    def _on_calibration_failed(self, message: str) -> None:
        self._set_calibration_stage("idle")
        QtWidgets.QMessageBox.warning(self, "标定失败", message)

    def _on_calibration_completed(self, result: RigCalibrationResult) -> None:
        self._apply_calibration_result(result)
        self._set_calibration_stage("done")

    def _apply_calibration_result(self, result: RigCalibrationResult) -> None:
        for descriptor in self._cameras:
            extrinsic = result.extrinsics.get(descriptor.camera_id)
            if extrinsic is None:
                descriptor.status = "error"
                descriptor.reprojection = float("nan")
                continue

            transform = result.relative_transforms.get(descriptor.camera_id)
            if transform is not None:
                descriptor.x = float(transform[0][3])
                descriptor.y = float(transform[1][3])
                descriptor.z = float(transform[2][3])
                descriptor.angle = math.degrees(math.atan2(transform[2][0], transform[0][0]))

            descriptor.reprojection = extrinsic.reprojection_error
            descriptor.status = self._status_from_metrics(extrinsic)

        self._pose_view.set_cameras([camera.to_pose_camera() for camera in self._cameras])
        if self._selected_camera_id:
            self._update_camera_selection(self._selected_camera_id)

    def _status_from_metrics(self, extrinsic: CameraExtrinsic) -> str:
        if not math.isfinite(float(extrinsic.reprojection_error)):
            return "error"
        if extrinsic.marker_count < 4:
            return "warning"
        if extrinsic.reprojection_error >= 4.0:
            return "error"
        if extrinsic.reprojection_error >= 1.5:
            return "warning"
        return "ok"

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._stop_capture()
        super().closeEvent(event)

    @staticmethod
    def _parse_camera_index(camera_id: str) -> int | None:
        if not camera_id.startswith("CAM-"):
            return None
        try:
            # CAM-01 -> 0
            return int(camera_id.split("-", 1)[1]) - 1
        except Exception:
            return None

    @staticmethod
    def _load_camera_index_mapping() -> dict[int, str]:
        try:
            from capture_tools import camera_serial_index
        except Exception:
            return {}

        index_to_serial: dict[int, str] = {}
        for serial, camera_name in dict(camera_serial_index).items():
            if not isinstance(serial, str) or not isinstance(camera_name, str):
                continue
            if not camera_name.startswith("camera_"):
                continue
            try:
                index = int(camera_name.split("_", 1)[1])
            except Exception:
                continue
            index_to_serial[index] = serial
        return index_to_serial

    def _start_capture_async(self) -> None:
        thread = threading.Thread(target=self._start_capture, name="capture-start", daemon=True)
        thread.start()

    def _start_capture(self) -> None:
        try:
            from capture_tools.capture_all import CaptureAll
        except Exception:
            return

        capture = CaptureAll()
        try:
            capture.start_all()
        except Exception:
            return

        with self._capture_lock:
            self._capture_all = capture

    def _stop_capture(self) -> None:
        with self._capture_lock:
            capture = self._capture_all
            self._capture_all = None

        if capture is None:
            return

        try:
            capture.stop_all()
        except Exception:
            pass

    def _find_handler(self, serial: str):
        with self._capture_lock:
            capture = self._capture_all
        if capture is None:
            return None

        for handler in getattr(capture, "orbbec_handlers", []):
            if getattr(handler, "serial_number", None) == serial:
                return handler
        for handler in getattr(capture, "realsense_handlers", []):
            if getattr(handler, "serial_number", None) == serial:
                return handler
        return None

    def _get_live_stream_images(self, camera_id: str) -> tuple[QtGui.QImage | None, QtGui.QImage | None]:
        idx = self._parse_camera_index(camera_id)
        if idx is None:
            return None, None

        serial = self._camera_index_to_serial.get(idx)
        if not serial:
            return None, None

        handler = self._find_handler(serial)
        if handler is None:
            return None, None

        try:
            import numpy as np  # noqa: WPS433
        except Exception:
            return None, None

        def bgr_to_qimage(frame: object) -> QtGui.QImage | None:
            if frame is None:
                return None
            if not isinstance(frame, np.ndarray):
                return None
            if frame.ndim != 3 or frame.shape[2] != 3:
                return None
            if frame.dtype != np.uint8:
                return None
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            h, w = frame.shape[:2]
            qimg = QtGui.QImage(frame.data, w, h, int(frame.strides[0]), QtGui.QImage.Format.Format_BGR888)
            return qimg.copy()

        # Orbbec handler returns (color_bgr_uint8, depth_colormap_bgr_uint8)
        if hasattr(handler, "get_latest_frames"):
            color_bgr, depth_vis = handler.get_latest_frames()
            # RealSense handler implements get_latest_data(), so this path is for Orbbec only.
            if depth_vis is not None and isinstance(depth_vis, np.ndarray) and depth_vis.ndim == 2:
                depth_vis = None
            return bgr_to_qimage(color_bgr), bgr_to_qimage(depth_vis)

        # RealSense handler returns (color_bgr_uint8, depth_raw_uint16)
        if hasattr(handler, "get_latest_data"):
            try:
                import cv2  # noqa: WPS433
            except Exception:
                cv2 = None

            color_bgr, depth_raw = handler.get_latest_data()
            depth_qimage = None
            if depth_raw is not None and isinstance(depth_raw, np.ndarray) and cv2 is not None:
                if depth_raw.ndim == 2 and depth_raw.dtype == np.uint16:
                    depth_scale = getattr(handler, "depth_scale", None)
                    depth_mm = depth_raw.astype(np.float32)
                    if isinstance(depth_scale, (float, int)) and depth_scale > 0:
                        depth_mm *= float(depth_scale) * 1000.0
                    clip_max = 5000.0
                    depth_mm = np.clip(depth_mm, 0.0, clip_max)
                    depth_8u = (depth_mm / clip_max * 255.0).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                    depth_qimage = bgr_to_qimage(depth_vis)

            return bgr_to_qimage(color_bgr), depth_qimage

        return None, None

    @staticmethod
    def _create_capture_folder() -> tuple[Path, str]:
        now = time.localtime()
        folder_stamp = time.strftime("%Y%m%d%H%M%S", now)
        time_suffix = time.strftime("%H%M%S", now)
        folder = Path("capture_results") / folder_stamp
        folder.mkdir(parents=True, exist_ok=True)
        return folder, time_suffix

    def _capture_camera_to_folder(self, camera_id: str, folder: Path, time_suffix: str) -> bool:
        idx = self._parse_camera_index(camera_id)
        if idx is None:
            return False

        serial = self._camera_index_to_serial.get(idx)
        if not serial:
            return False

        handler = self._find_handler(serial)
        if handler is None or not hasattr(handler, "get_latest_data"):
            return False

        try:
            import cv2  # noqa: WPS433
            import numpy as np  # noqa: WPS433
        except Exception:
            return False

        try:
            data = handler.get_latest_data()
        except Exception:
            return False

        if not isinstance(data, tuple) or len(data) < 2:
            return False

        color_bgr = data[0]
        depth_raw = data[1]

        if not isinstance(color_bgr, np.ndarray):
            return False
        if color_bgr.ndim != 3 or color_bgr.shape[2] != 3 or color_bgr.dtype != np.uint8 or color_bgr.size == 0:
            return False

        if not isinstance(depth_raw, np.ndarray):
            return False
        if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16 or depth_raw.size == 0:
            return False

        color_h, color_w = color_bgr.shape[:2]
        depth_h, depth_w = depth_raw.shape[:2]

        color_path = folder / f"color_camera_{camera_id}_{color_w}x{color_h}_{time_suffix}.png"
        depth_path = folder / f"depth_camera_{camera_id}_{depth_w}x{depth_h}_{time_suffix}.raw"

        ok = cv2.imwrite(str(color_path), color_bgr)
        if not ok:
            return False

        np.ascontiguousarray(depth_raw).tofile(str(depth_path))
        return True

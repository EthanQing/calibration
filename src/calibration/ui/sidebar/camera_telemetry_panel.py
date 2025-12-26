"""Telemetry/control sidebar replicating the dashboard layout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from PySide6 import QtCore, QtWidgets


@dataclass
class CameraPose:
    x: float
    y: float
    z: float
    angle: float


class CameraTelemetryPanel(QtWidgets.QWidget):
    """Shows device info, matrices, pose, and provides recalculation action."""

    recalc_requested = QtCore.Signal()
    capture_requested = QtCore.Signal()
    capture_all_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._fields: Dict[str, QtWidgets.QLabel] = {}
        self._pose_labels: Dict[str, QtWidgets.QLabel] = {}
        self._status_chip: QtWidgets.QLabel
        self._title_label: QtWidgets.QLabel
        self._stage_label: QtWidgets.QLabel | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setObjectName("TelemetryPanel")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QtWidgets.QFrame()
        header.setObjectName("TelemetryHeader")
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(16, 16, 16, 16)
        header_layout.setSpacing(4)

        subtitle = QtWidgets.QLabel("相机参数")
        subtitle.setObjectName("TelemetrySubtitle")
        self._title_label = QtWidgets.QLabel("CAM-00")
        self._title_label.setObjectName("TelemetryTitle")
        self._status_chip = QtWidgets.QLabel("--")
        self._status_chip.setObjectName("TelemetryStatusChip")
        self._status_chip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self._title_label)
        top_row.addStretch()
        top_row.addWidget(self._status_chip)

        header_layout.addWidget(subtitle)
        header_layout.addLayout(top_row)
        layout.addWidget(header)

        body = QtWidgets.QScrollArea()
        body.setWidgetResizable(True)
        body.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout.addWidget(body, 1)

        container = QtWidgets.QWidget()
        body.setWidget(container)
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(16, 16, 16, 16)
        container_layout.setSpacing(16)

        container_layout.addWidget(self._build_device_info())
        container_layout.addWidget(self._build_intrinsics())
        container_layout.addWidget(self._build_extrinsics())
        container_layout.addWidget(self._build_reprojection_card())
        container_layout.addStretch()
        container_layout.addWidget(self._build_action_button())

        self.setStyleSheet(
            """
            QWidget#TelemetryPanel {
                background-color: rgba(255,255,255,0.35);
                border-left: 1px solid rgba(15,23,42,0.06);
            }
            #TelemetryHeader {
                background-color: transparent;
                border-left: none;
                border-bottom: 1px solid rgba(15,23,42,0.08);
            }
            #TelemetrySubtitle {
                font-size: 10px;
                letter-spacing: 2px;
                font-weight: 700;
                color: rgba(15,23,42,0.55);
                text-transform: uppercase;
            }
            #TelemetryTitle {
                font-size: 24px;
                font-family: 'JetBrains Mono', Consolas, monospace;
                color: #0f172a;
            }
            #TelemetryStatusChip {
                padding: 6px 12px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 600;
            }
            QWidget#TelemetryCard {
                background-color: rgba(255,255,255,0.8);
                border: 1px solid rgba(15,23,42,0.08);
                border-radius: 14px;
            }
            QLabel[kind='label'] {
                color: rgba(15,23,42,0.55);
                font-size: 11px;
            }
            QLabel[kind='value'] {
                color: #0f172a;
                font-size: 12px;
            }
            QPushButton#TelemetryActionButton {
                background-color: rgba(15,23,42,0.08);
                border: 1px solid rgba(15,23,42,0.12);
                border-radius: 12px;
                color: #0f172a;
                font-weight: 600;
                padding: 10px;
            }
            QPushButton#TelemetryActionButton:hover {
                background-color: rgba(15,23,42,0.15);
            }
            """
        )

    def _build_device_info(self) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("TelemetryCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        title = QtWidgets.QLabel("设备信息")
        title.setObjectName("TelemetryCardTitle")
        title.setStyleSheet("color: #94a3b8; font-size: 10px; letter-spacing: 1px; text-transform: uppercase;")
        layout.addWidget(title)

        grid = QtWidgets.QGridLayout()
        grid.setVerticalSpacing(6)
        grid.setHorizontalSpacing(12)
        layout.addLayout(grid)

        for row, (label_text, key) in enumerate((
            ("所属层级", "group"),
            ("设备类型", "device_type"),
            ("序列号", "serial"),
            ("连接方式", "connection"),
            ("分辨率", "resolution"),
        )):
            label = QtWidgets.QLabel(label_text)
            label.setProperty("kind", "label")
            value = QtWidgets.QLabel("--")
            value.setProperty("kind", "value")
            grid.addWidget(label, row, 0)
            grid.addWidget(value, row, 1)
            self._fields[key] = value
        return card

    def _build_intrinsics(self) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("TelemetryCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("内参矩阵 (K)")
        title.setStyleSheet("color: #94a3b8; font-size: 10px; letter-spacing: 1px; text-transform: uppercase;")
        layout.addWidget(title)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        layout.addLayout(grid)
        for r in range(3):
            for c in range(3):
                label = QtWidgets.QLabel("0.0")
                label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                label.setObjectName("TelemetryMatrixCell")
                label.setStyleSheet(
                    "background-color: rgba(15,23,42,0.06); border-radius: 6px;"
                    "font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #0f172a;"
                )
                grid.addWidget(label, r, c)
                self._fields[f"k_{r}{c}"] = label
        return card

    def _build_extrinsics(self) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("TelemetryCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("外参/位姿")
        title.setStyleSheet("color: #94a3b8; font-size: 10px; letter-spacing: 1px; text-transform: uppercase;")
        layout.addWidget(title)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        layout.addLayout(grid)

        for idx, (label_text, key) in enumerate(
            (
                ("位置 X", "pos_x"),
                ("位置 Y", "pos_y"),
                ("位置 Z", "pos_z"),
                ("偏航角", "yaw"),
            )
        ):
            label = QtWidgets.QLabel(label_text)
            label.setProperty("kind", "label")
            value = QtWidgets.QLabel("--")
            value.setProperty("kind", "value")
            value.setObjectName("TelemetryPoseValue")
            grid.addWidget(label, idx // 2, (idx % 2) * 2)
            grid.addWidget(value, idx // 2, (idx % 2) * 2 + 1)
            self._pose_labels[key] = value
        return card

    def _build_reprojection_card(self) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("TelemetryCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("重投影误差")
        label.setStyleSheet("color: #38bdf8; font-size: 10px; letter-spacing: 1px; text-transform: uppercase;")
        self._reproj_value = QtWidgets.QLabel("-- px")
        self._reproj_value.setStyleSheet("color: #34d399; font-weight: 700;")
        header.addWidget(label)
        header.addStretch()
        header.addWidget(self._reproj_value)
        layout.addLayout(header)

        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setFixedHeight(8)
        bar.setTextVisible(False)
        bar.setStyleSheet(
            "QProgressBar { background-color: #1f2937; border-radius: 4px; }"
            "QProgressBar::chunk { background-color: #3b82f6; border-radius: 4px; }"
        )
        self._reproj_bar = bar
        layout.addWidget(bar)
        return card

    def _build_action_button(self) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        capture_button = QtWidgets.QPushButton("采集当前相机")
        capture_button.setObjectName("TelemetryActionButton")
        capture_button.clicked.connect(self.capture_requested.emit)
        layout.addWidget(capture_button)

        capture_all_button = QtWidgets.QPushButton("一键采集所有相机")
        capture_all_button.setObjectName("TelemetryActionButton")
        capture_all_button.clicked.connect(self.capture_all_requested.emit)
        layout.addWidget(capture_all_button)

        recalc_button = QtWidgets.QPushButton("重新计算参数")
        recalc_button.setObjectName("TelemetryActionButton")
        recalc_button.clicked.connect(self.recalc_requested.emit)
        layout.addWidget(recalc_button)
        return wrapper

    def update_camera(
        self,
        metadata: Mapping[str, Any] | None,
        pose: CameraPose | None,
        intrinsics: Mapping[str, float] | None,
        reprojection: float | None,
    ) -> None:
        if not metadata:
            self._title_label.setText("CAM--")
            self._status_chip.setText("--")
            self._status_chip.setStyleSheet("background-color: rgba(148,163,184,0.2); color: #cbd5f5;")
            for label in self._fields.values():
                label.setText("--")
            for label in self._pose_labels.values():
                label.setText("--")
            self._update_intrinsics_grid({})
            self._set_reprojection(None)
            return

        camera_id = metadata.get("id", "??")
        camera_id_text = str(camera_id)
        if camera_id_text.startswith("CAM-"):
            self._title_label.setText(camera_id_text)
        else:
            self._title_label.setText(f"CAM-{camera_id_text}")
        status = str(metadata.get("status", "unknown")).lower()
        status_text = {
            "ok": "在线",
            "warning": "警告",
            "error": "故障",
        }.get(status, "未知")
        status_color = {
            "ok": ("#22c55e", "#052e16"),
            "warning": ("#facc15", "#422006"),
            "error": ("#f87171", "#3b0a0a"),
        }.get(status, ("#94a3b8", "#0f172a"))
        self._status_chip.setText(status_text)
        self._status_chip.setStyleSheet(
            f"background-color: {status_color[1]}; color: {status_color[0]};"
            "border: 1px solid rgba(255,255,255,0.1);"
        )

        self._fields["group"].setText(str(metadata.get("group", "--")))
        self._fields["device_type"].setText(str(metadata.get("device_type", "--")))
        self._fields["serial"].setText(str(metadata.get("serial", "--")))
        self._fields["connection"].setText(str(metadata.get("connection", "USB 3.0")))
        self._fields["resolution"].setText(str(metadata.get("resolution", "1280x720")))

        self._update_intrinsics_grid(intrinsics or {})

        if pose:
            self._pose_labels["pos_x"].setText(f"{pose.x:.3f} m")
            self._pose_labels["pos_y"].setText(f"{pose.y:.3f} m")
            self._pose_labels["pos_z"].setText(f"{pose.z:.3f} m")
            self._pose_labels["yaw"].setText(f"{pose.angle:.1f}°")
        else:
            for label in self._pose_labels.values():
                label.setText("--")

        self._set_reprojection(reprojection)

    def _update_intrinsics_grid(self, values: Mapping[str, float]) -> None:
        if not values:
            for key, label in self._fields.items():
                if key.startswith("k_"):
                    label.setText("--")
            return

        matrix: Dict[str, float] = {f"k_{r}{c}": 0.0 for r in range(3) for c in range(3)}
        matrix["k_22"] = 1.0
        matrix.update({k: float(v) for k, v in values.items() if k.startswith("k_")})

        for key, label in self._fields.items():
            if key.startswith("k_"):
                label.setText(f"{matrix.get(key, 0.0):.1f}")

    def _set_reprojection(self, value: float | None) -> None:
        if value is None:
            self._reproj_value.setText("-- px")
            self._reproj_bar.setValue(0)
            return
        if not math.isfinite(float(value)):
            self._reproj_value.setText("-- px")
            self._reproj_bar.setValue(0)
            return
        clamped = max(0.0, min(value, 2.0))
        pct = int((1.0 - min(clamped / 2.0, 1.0)) * 100)
        self._reproj_value.setText(f"{value:.2f} px")
        self._reproj_bar.setValue(pct)

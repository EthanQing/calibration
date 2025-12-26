"""Camera rig sidebar showing grouped sensors similar to the HexaCalib design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class RigCamera:
    """Describes one camera slot inside the rig."""

    camera_id: str
    group: str
    status: str


class _CameraListItem(QtWidgets.QFrame):
    """Interactive row for a single camera entry."""

    clicked = QtCore.Signal(str)

    def __init__(self, camera: RigCamera, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._camera = camera
        self._selected = False
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setObjectName("CameraListItem")
        self._build_ui()
        self._refresh_styles()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self._badge = QtWidgets.QLabel(self._camera.camera_id)
        self._badge.setObjectName("CameraListItemBadge")
        self._badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._badge.setFixedSize(28, 28)

        self._label = QtWidgets.QLabel(f"相机 {self._camera.camera_id}")
        self._label.setObjectName("CameraListItemLabel")

        self._status_dot = QtWidgets.QLabel()
        self._status_dot.setFixedSize(8, 8)
        self._status_dot.setObjectName("CameraListItemStatus")
        self._status_dot.setStyleSheet("border-radius: 4px;")

        layout.addWidget(self._badge)
        layout.addWidget(self._label, 1)
        layout.addWidget(self._status_dot)

    def _refresh_styles(self) -> None:
        self.setProperty("selected", self._selected)
        self.style().unpolish(self)
        self.style().polish(self)
        base_palette = """
            QFrame#CameraListItem {
                border: 1px solid rgba(15,23,42,0.08);
                border-radius: 10px;
                background-color: rgba(255,255,255,0.7);
            }
            QFrame#CameraListItem:hover {
                background-color: rgba(255,255,255,0.9);
            }
            QFrame#CameraListItem[selected='true'] {
                background-color: rgba(167,139,250,0.25);
                border: 1px solid rgba(167,139,250,0.4);
            }
            QLabel#CameraListItemBadge {
                background-color: rgba(15,23,42,0.08);
                color: #0f172a;
                border-radius: 8px;
                font-weight: 600;
            }
            QLabel#CameraListItemBadge[selected='true'] {
                background-color: #a78bfa;
                color: white;
            }
        """
        self.setStyleSheet(base_palette)
        self._badge.setProperty("selected", self._selected)
        self._badge.style().unpolish(self._badge)
        self._badge.style().polish(self._badge)

        status_color = {
            "ok": "#22c55e",
            "warning": "#fbbf24",
            "error": "#ef4444",
        }.get(self._camera.status, "#94a3b8")
        self._status_dot.setStyleSheet(f"border-radius: 4px; background-color: {status_color};")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        super().mousePressEvent(event)
        self.clicked.emit(self._camera.camera_id)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._refresh_styles()

    def set_status(self, status: str) -> None:
        self._camera.status = status
        self._refresh_styles()


class CameraRigPanel(QtWidgets.QWidget):
    """Sidebar widget listing cameras grouped by rig layers."""

    camera_selected = QtCore.Signal(str)

    def __init__(self, cameras: Sequence[RigCamera], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._cameras = list(cameras)
        self._items: Dict[str, _CameraListItem] = {}
        self.setObjectName("CameraRigPanel")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QtWidgets.QFrame()
        header.setObjectName("CameraRigHeader")
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(16, 16, 16, 16)
        header_layout.setSpacing(4)

        title = QtWidgets.QLabel("HexaCalib 控制器")
        title.setObjectName("CameraRigTitle")
        subtitle = QtWidgets.QLabel("系统: 14 台 RGB-D 相机")
        subtitle.setObjectName("CameraRigSubtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll, 1)

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(12, 12, 12, 12)
        container_layout.setSpacing(16)

        group_names = {
            "Top": "顶部层",
            "Mid": "中间层",
            "Bot": "底部层",
        }
        for group in ("Top", "Mid", "Bot"):
            group_widget = QtWidgets.QWidget()
            vbox = QtWidgets.QVBoxLayout(group_widget)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(6)

            header_row = QtWidgets.QHBoxLayout()
            header_row.setContentsMargins(0, 0, 0, 0)
            label = QtWidgets.QLabel(group_names.get(group, group))
            label.setObjectName("CameraRigGroupLabel")
            badge = QtWidgets.QLabel(str(sum(1 for cam in self._cameras if cam.group == group)))
            badge.setObjectName("CameraRigGroupBadge")
            header_row.addWidget(label)
            header_row.addStretch()
            header_row.addWidget(badge)
            vbox.addLayout(header_row)

            for camera in [c for c in self._cameras if c.group == group]:
                item = _CameraListItem(camera)
                item.clicked.connect(self.camera_selected.emit)
                self._items[camera.camera_id] = item
                vbox.addWidget(item)

            container_layout.addWidget(group_widget)

        container_layout.addStretch()
        scroll.setWidget(container)

        self.setStyleSheet(
            """
            QWidget#CameraRigPanel {
                background-color: rgba(255,255,255,0.3);
                border-right: 1px solid rgba(15,23,42,0.06);
            }
            QFrame#CameraRigHeader {
                background-color: transparent;
                border-bottom: 1px solid rgba(15,23,42,0.08);
            }
            QLabel#CameraRigTitle {
                font-size: 18px;
                font-weight: 700;
                color: #0f172a;
            }
            QLabel#CameraRigSubtitle {
                font-size: 12px;
                color: rgba(15,23,42,0.6);
            }
            QLabel#CameraRigGroupLabel {
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 1px;
                color: rgba(15,23,42,0.45);
            }
            QLabel#CameraRigGroupBadge {
                min-width: 22px;
                padding: 2px 8px;
                font-size: 10px;
                border-radius: 999px;
                background-color: rgba(15,23,42,0.08);
                color: #0f172a;
                text-align: center;
            }
            QScrollArea { background-color: transparent; }
            QScrollBar:vertical {
                width: 6px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(15,23,42,0.2);
                border-radius: 3px;
            }
            """
        )

    def select_camera(self, camera_id: str) -> None:
        for cid, item in self._items.items():
            item.set_selected(cid == camera_id)

    def update_camera_status(self, camera_id: str, status: str) -> None:
        if camera_id in self._items:
            self._items[camera_id].set_status(status)

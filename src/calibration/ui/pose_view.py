"""Custom widget that renders the 3D pose/rig view similar to the reference dashboard."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class PoseCamera:
    camera_id: str
    x: float
    y: float
    z: float
    angle: float
    status: str


class PoseView(QtWidgets.QWidget):
    """Interactive canvas showing rig layout in 3D."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._cameras: List[PoseCamera] = []
        self._selected_camera_id: str | None = None
        self._calibration_stage = "idle"
        self._pitch = 0.2
        self._yaw = math.radians(30.0)
        self._dragging = False
        self._last_pos = QtCore.QPoint()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(33)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.SplitHCursor)
        self.setMinimumHeight(360)

    def sizeHint(self) -> QtCore.QSize:  # noqa: D401
        return QtCore.QSize(640, 360)

    def set_cameras(self, cameras: Sequence[PoseCamera]) -> None:
        self._cameras = list(cameras)
        self.update()

    def set_selected_camera(self, camera_id: str | None) -> None:
        self._selected_camera_id = camera_id
        self.update()

    def set_calibration_stage(self, stage: str) -> None:
        self._calibration_stage = stage
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0.0, QtGui.QColor("#f8fafc"))
        gradient.setColorAt(1.0, QtGui.QColor("#e2e8f0"))
        painter.fillRect(rect, gradient)

        width = rect.width()
        height = rect.height()
        if width == 0 or height == 0:
            return

        # Draw prism
        R = 1.5
        H = 1.8
        hex_vertices = [
            QtCore.QPointF(
                math.cos(math.radians(60 * i)) * R,
                math.sin(math.radians(60 * i)) * R,
            )
            for i in range(6)
        ]

        def project_point(x: float, y: float, z: float) -> QtCore.QPointF | None:
            cos_yaw = math.cos(self._yaw)
            sin_yaw = math.sin(self._yaw)
            x1 = x * cos_yaw - z * sin_yaw
            z1 = x * sin_yaw + z * cos_yaw

            cos_pitch = math.cos(self._pitch)
            sin_pitch = math.sin(self._pitch)
            y2 = y * cos_pitch - z1 * sin_pitch
            z2 = y * sin_pitch + z1 * cos_pitch

            distance = 7.0
            fov = 600.0
            z_eff = z2 + distance
            if z_eff <= 0:
                return None
            factor = fov / z_eff * 0.9
            screen_x = x1 * factor + width / 2
            screen_y = -y2 * factor + height / 2
            return QtCore.QPointF(screen_x, screen_y)

        top = [project_point(v.x(), H, v.y()) for v in hex_vertices]
        bot = [project_point(v.x(), -H, v.y()) for v in hex_vertices]
        center_top = project_point(0, H, 0)
        center_bot = project_point(0, -H, 0)

        def draw_line(p1: QtCore.QPointF | None, p2: QtCore.QPointF | None,
                      color: QtGui.QColor, width_px: float = 1.0,
                      dotted: bool = False) -> None:
            if not p1 or not p2:
                return
            pen = QtGui.QPen(color, width_px)
            if dotted:
                pen.setStyle(QtCore.Qt.PenStyle.DotLine)
            painter.setPen(pen)
            painter.drawLine(p1, p2)

        draw_line(center_top, center_bot, QtGui.QColor(255, 255, 255, 40), 1, True)

        purple = QtGui.QColor("#8b5cf6")
        for i in range(6):
            next_i = (i + 1) % 6
            draw_line(top[i], top[next_i], purple, 2)
            draw_line(bot[i], bot[next_i], purple, 2)
            draw_line(top[i], bot[i], QtGui.QColor(139, 92, 246, 120), 1)

        # Cameras
        camera_points = []
        for cam in self._cameras:
            projected = project_point(cam.x, cam.y, cam.z)
            if projected is None:
                continue
            cos_yaw = math.cos(self._yaw)
            sin_yaw = math.sin(self._yaw)
            x1 = cam.x * cos_yaw - cam.z * sin_yaw
            z1 = cam.x * sin_yaw + cam.z * cos_yaw
            cos_pitch = math.cos(self._pitch)
            sin_pitch = math.sin(self._pitch)
            _ = cam.y * cos_pitch - z1 * sin_pitch
            depth = cam.y * sin_pitch + z1 * cos_pitch
            camera_points.append((cam, projected, depth))

        camera_points.sort(key=lambda item: item[2], reverse=True)
        for cam, point, depth in camera_points:
            selected = cam.camera_id == self._selected_camera_id
            if selected and center_top:
                draw_line(point, QtCore.QPointF(center_top.x(), point.y()), QtGui.QColor(255, 255, 255, 100), 1)

            radius = 6 if not selected else 8
            color = {
                "error": QtGui.QColor("#ef4444"),
                "warning": QtGui.QColor("#eab308"),
            }.get(cam.status, QtGui.QColor("#8b5cf6" if selected else "#1e293b"))
            pen_color = QtGui.QColor("#ffffff" if selected else "#94a3b8")
            painter.setBrush(color)
            painter.setPen(QtGui.QPen(pen_color, 2 if selected else 1.5))
            painter.drawEllipse(point, radius, radius)

            if selected or depth < 1:
                painter.setPen(QtGui.QPen(QtGui.QColor("white"), 1))
                font = painter.font()
                font.setBold(selected)
                font.setPointSize(8 if selected else 7)
                painter.setFont(font)
                painter.drawText(point + QtCore.QPointF(-8, -10 if selected else -8), cam.camera_id)

        self._draw_hud(painter)

    def _draw_hud(self, painter: QtGui.QPainter) -> None:
        hud_rect = QtCore.QRect(16, 16, 220, 80)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 200))
        painter.drawRoundedRect(hud_rect, 10, 10)

        painter.setPen(QtGui.QPen(QtGui.QColor("#475569")))
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(hud_rect.adjusted(12, 10, -12, 0), "标定状态")

        state_text = {
            "idle": ("待命", QtGui.QColor("#e2e8f0")),
            "optimizing": ("计算中...", QtGui.QColor("#facc15")),
            "done": ("已收敛", QtGui.QColor("#22c55e")),
        }.get(self._calibration_stage, (self._calibration_stage, QtGui.QColor("#e2e8f0")))

        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(state_text[1])
        painter.drawText(hud_rect.adjusted(12, 32, -12, 0), state_text[0])

        painter.setPen(QtGui.QPen(QtGui.QColor(71, 85, 105)))
        font.setPointSize(8)
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(hud_rect.adjusted(12, 56, -12, 0), "俯仰角已锁定")

        info_rect = QtCore.QRect(self.width() // 2 - 180, self.height() - 48, 360, 28)
        painter.setBrush(QtGui.QColor(255, 255, 255, 180))
        painter.setPen(QtGui.QPen(QtGui.QColor(15, 23, 42, 30)))
        painter.drawRoundedRect(info_rect, 14, 14)
        painter.setPen(QtGui.QPen(QtGui.QColor(71, 85, 105)))
        painter.drawText(info_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "左右拖动以旋转视角")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            delta_x = event.position().x() - self._last_pos.x()
            self._yaw += delta_x * 0.005
            self._last_pos = event.position().toPoint()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._dragging = False
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        self._dragging = False
        super().leaveEvent(event)

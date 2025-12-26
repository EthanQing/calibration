"""Qt helper that runs a (stub) calibration job on a background thread."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from PySide6 import QtCore

Matrix4x4 = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]


@dataclass(frozen=True)
class CameraExtrinsic:
    marker_count: int
    reprojection_error: float


@dataclass(frozen=True)
class RigCalibrationResult:
    """UI-facing calibration outputs (placeholder)."""

    extrinsics: Dict[str, CameraExtrinsic]
    relative_transforms: Dict[str, Matrix4x4]


class _WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(RigCalibrationResult)
    error = QtCore.Signal(str)


class _CalibrationWorker(QtCore.QRunnable):
    def __init__(self, factory: Callable[[], RigCalibrationResult]) -> None:
        super().__init__()
        self._factory = factory
        self.signals = _WorkerSignals()

    def run(self) -> None:  # noqa: D401
        try:
            result = self._factory()
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))
        else:
            self.signals.finished.emit(result)


class RigCalibrationController(QtCore.QObject):
    """Orchestrates calibration jobs and exposes Qt signals."""

    calibration_started = QtCore.Signal()
    calibration_failed = QtCore.Signal(str)
    calibration_completed = QtCore.Signal(RigCalibrationResult)

    def __init__(self, job_factory: Callable[[], RigCalibrationResult], parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._job_factory = job_factory
        self._pool = QtCore.QThreadPool(self)

    def trigger_calibration(self) -> None:
        worker = _CalibrationWorker(self._job_factory)
        worker.signals.finished.connect(self.calibration_completed.emit)
        worker.signals.error.connect(self.calibration_failed.emit)
        self.calibration_started.emit()
        self._pool.start(worker)

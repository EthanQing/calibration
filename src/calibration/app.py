"""Application entry point for the UI-only calibration shell."""

from __future__ import annotations

import sys

from PySide6 import QtWidgets

from calibration.ui.calibration_console_window import CalibrationConsoleWindow


def main() -> None:
    """Launch the PySide6 GUI shell."""
    app = QtWidgets.QApplication.instance()
    owns_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        owns_app = True

    window = CalibrationConsoleWindow()
    window.show()

    if owns_app:
        app.exec()


if __name__ == "__main__":
    main()

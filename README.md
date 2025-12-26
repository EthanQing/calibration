# Calibration (UI-only scaffold)

This repository has been stripped down to the PySide6 UI layer so you can rebuild the calibration and video-capture logic from scratch.

## Layout

```
src/calibration/
├─ app.py        # GUI entry point
└─ ui/           # PySide6 UI components (shell/demo)
```

## Run

```bash
python -m venv .venv
.\.venv\Scripts\pip install -e .[dev]
.\.venv\Scripts\python -m calibration.app
```

If you prefer, you can also run:

```bash
python main.py
```

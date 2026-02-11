import os
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")


def test_demo_interactive_headless_smoke() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "src" / "demo_interactive.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--headless-smoke",
            "--headless-steps",
            "10",
            "--N",
            "6",
            "--dt",
            "0.02",
            "--mode",
            "nolearning",
        ],
        cwd=project_root,
        env={**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(project_root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "step" in result.stdout

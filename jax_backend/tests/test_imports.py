import importlib
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


MODULES = [
    "action",
    "genmodel",
    "genprocess",
    "inference",
    "interactive",
    "learning",
    "utils",
    "demo_interactive",
    "demo_nolearning",
    "demo_withlearning",
]


def test_core_module_imports() -> None:
    for module_name in MODULES:
        importlib.import_module(module_name)

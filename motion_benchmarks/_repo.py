import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MOVING_MNIST_DIR = REPO_ROOT / "moving_mnist"

for p in (str(REPO_ROOT), str(MOVING_MNIST_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

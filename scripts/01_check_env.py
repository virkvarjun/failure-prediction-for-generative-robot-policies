#!/usr/bin/env python3
"""
01_check_env.py
───────────────
Verifies that all dependencies required for ACT training via LeRobot
are installed and importable. Run this before 02_train_act.sh.

Usage:
    python scripts/01_check_env.py
"""

import platform
import sys


def banner(text: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {text}")
    print(f"{'─' * 55}")


banner("System Info")
print(f"  Python  : {sys.version}")
print(f"  Platform: {platform.platform()}")

# ── Torch ─────────────────────────────────────────────────────────────────────
banner("PyTorch")
try:
    import torch

    print(f"  Version : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()}")
    # MPS = Apple Metal Performance Shaders (Apple Silicon / macOS GPU)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"  MPS     : {mps_available}")
except ImportError as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# ── LeRobot ───────────────────────────────────────────────────────────────────
banner("LeRobot")
try:
    import lerobot

    print(f"  Version : {lerobot.__version__}")
except ImportError as e:
    print(f"  ERROR importing lerobot: {e}")
    sys.exit(1)

# Import the specific modules that 02_train_act.sh exercises at startup.
# These will fail early and loudly if the install is broken.
modules_to_check = [
    ("lerobot.scripts.lerobot_train", "main training entrypoint"),
    ("lerobot.configs.train", "TrainPipelineConfig"),
    ("lerobot.configs.default", "DatasetConfig"),
    ("lerobot.policies.act.configuration_act", "ACTConfig"),
    ("lerobot.policies.act.modeling_act", "ACTPolicy"),
]

banner("Module Imports")
all_ok = True
for module_path, description in modules_to_check:
    try:
        __import__(module_path)
        print(f"  ✓  {module_path}  ({description})")
    except ImportError as e:
        print(f"  ✗  {module_path}  FAILED: {e}")
        all_ok = False

# ── Final result ──────────────────────────────────────────────────────────────
banner("Result")
if all_ok:
    print("  ✅  All checks passed. Environment is ready for training.\n")
else:
    print("  ❌  Some imports failed. Fix the errors above, then re-run.\n")
    sys.exit(1)

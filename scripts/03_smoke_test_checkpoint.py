#!/usr/bin/env python3
"""
03_smoke_test_checkpoint.py
───────────────────────────
Loads a trained ACT checkpoint and runs a dummy forward pass to verify
the saved model is loadable and produces valid action tensors.
No robot hardware required.

Usage:
    # After training, run:
    python scripts/03_smoke_test_checkpoint.py

    # Or point to a specific checkpoint dir:
    CHECKPOINT_DIR="outputs/train/20240101_120000/checkpoints/last"
    python scripts/03_smoke_test_checkpoint.py
"""

import os
import sys
from pathlib import Path


def find_latest_checkpoint(outputs_root: Path) -> Path | None:
    """Return the most recently modified checkpoint directory, or None."""
    checkpoints = sorted(
        outputs_root.rglob("checkpoints/last"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def main() -> None:
    # ── Locate checkpoint ─────────────────────────────────────────────────────
    repo_root = Path(__file__).resolve().parent.parent

    # Allow explicit override via env var
    checkpoint_dir_env = os.environ.get("CHECKPOINT_DIR", "")
    if checkpoint_dir_env:
        checkpoint_dir = Path(checkpoint_dir_env)
    else:
        outputs_root = repo_root / "outputs" / "train"
        checkpoint_dir = find_latest_checkpoint(outputs_root)  # type: ignore[assignment]

    if checkpoint_dir is None or not checkpoint_dir.exists():
        print("\n❌  No checkpoint found.")
        print("   Train first with:  bash scripts/02_train_act.sh")
        print("   Or set:            CHECKPOINT_DIR=outputs/train/.../checkpoints/last")
        sys.exit(1)

    print(f"\n✅  Found checkpoint: {checkpoint_dir}")

    # ── Load the policy ───────────────────────────────────────────────────────
    print("   Loading ACT policy from checkpoint …")
    try:
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy

        # from_pretrained loads config.json + model weights from the dir
        policy = ACTPolicy.from_pretrained(str(checkpoint_dir))
        policy.eval()  # switch to inference mode (disables dropout, VAE sampling)
    except Exception as e:
        print(f"\n❌  Failed to load policy: {e}")
        sys.exit(1)

    print("   Policy loaded successfully.")
    print(f"   Policy type : {policy.config.type}")
    print(f"   Device      : {policy.config.device}")
    print(f"   Chunk size  : {policy.config.chunk_size}")

    # ── Dummy forward pass ────────────────────────────────────────────────────
    # Build a minimal fake observation batch that matches the policy's
    # expected inputs (inferred from the saved config).
    print("\n   Running dummy inference …")
    try:
        device = torch.device(policy.config.device)

        # Build a batch of dummy observations.
        # ACT requires at least one image or environment_state input.
        # We check which input features the policy was trained with.
        batch = {}

        if policy.config.input_features:
            from lerobot.configs.types import FeatureType

            for key, feature in policy.config.input_features.items():
                # shape stored in feature is (C, H, W) for images, (D,) for state
                shape = feature.shape
                # Add a batch dimension of 1 for single-sample inference
                batch[key] = torch.zeros(1, *shape, device=device)
        else:
            # Fallback: dummy 1×3×480×640 image (typical camera resolution)
            print("   ⚠️  No input_features in config; using fallback dummy image.")
            batch["observation.image"] = torch.zeros(1, 3, 480, 640, device=device)

        with torch.inference_mode():  # disables gradient tracking for speed
            policy.reset()  # reset temporal ensemble state if enabled
            actions = policy.select_action(batch)

        print(f"   Action tensor shape : {actions.shape}")
        print(f"   Action dtype        : {actions.dtype}")
        print("\n✅  Smoke test passed — checkpoint is valid and produces actions.\n")

    except Exception as e:
        print(f"\n❌  Forward pass failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

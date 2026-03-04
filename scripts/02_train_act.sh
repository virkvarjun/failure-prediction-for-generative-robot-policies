#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 02_train_act.sh
# ─────────────────────────────────────────────────────────────────────────────
# Train an ACT policy using HuggingFace LeRobot.
#
# Usage:
#   # Option A: HuggingFace dataset
#   export DATASET_REPO_ID="a23v/<your-dataset-name>"
#   bash scripts/02_train_act.sh
#
#   # Option B: local dataset path
#   export DATASET_PATH="data/<your-local-dataset>"
#   bash scripts/02_train_act.sh
#
#   # Override output dir (optional)
#   export OUTPUT_DIR="outputs/train/my_experiment"
#   bash scripts/02_train_act.sh
#
# Required: one of DATASET_REPO_ID or DATASET_PATH must be set.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve the script's own directory so paths are repo-relative ─────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Activate the project venv if it exists ────────────────────────────────────
VENV="$REPO_ROOT/.venv"
if [[ -d "$VENV" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "⚠️  No .venv found at $VENV. Using system Python."
  echo "   Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi

# ── Validate dataset source ───────────────────────────────────────────────────
# We require exactly one of DATASET_REPO_ID or DATASET_PATH.
DATASET_REPO_ID="${DATASET_REPO_ID:-}"
DATASET_PATH="${DATASET_PATH:-}"

if [[ -z "$DATASET_REPO_ID" && -z "$DATASET_PATH" ]]; then
  echo ""
  echo "❌  ERROR: No dataset specified."
  echo ""
  echo "   Set one of the following before running this script:"
  echo ""
  echo "   Option A — Hugging Face dataset:"
  echo "     export DATASET_REPO_ID=\"a23v/<your-dataset-name>\""
  echo ""
  echo "   Option B — Local dataset path:"
  echo "     export DATASET_PATH=\"data/<your-local-dataset>\""
  echo ""
  exit 1
fi

# ── Device ────────────────────────────────────────────────────────────────────
# Default to CPU (safe for macOS). Override with: export DEVICE=cuda
DEVICE="${DEVICE:-cpu}"

# ── Output directory ──────────────────────────────────────────────────────────
# Defaults to outputs/train/<timestamp> if not set.
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/train/$TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║           FIPER — ACT Training (Milestone 1)         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Output dir : $OUTPUT_DIR"
echo "  Device     : $DEVICE"
if [[ -n "$DATASET_REPO_ID" ]]; then
  echo "  Dataset    : HF repo  → $DATASET_REPO_ID"
else
  echo "  Dataset    : local    → $DATASET_PATH"
fi
echo ""

# ── Build the lerobot-train command ──────────────────────────────────────────
# All flags map 1-to-1 to fields in TrainPipelineConfig / ACTConfig (draccus).
# Verified by reading lerobot source at install time.

CMD=(lerobot-train

  # ── Policy ──────────────────────────────────────────────────────────────────
  --policy.type=act              # Select ACTConfig (registered as "act")
  --policy.device="$DEVICE"     # cpu | cuda | mps
  --policy.push_to_hub=false    # Do NOT auto-push checkpoint to HF Hub

  # ── ACT architecture (matches ACTConfig defaults from the paper) ────────────
  --policy.chunk_size=100        # Actions predicted per forward pass
  --policy.n_obs_steps=1         # Observation context steps (must be 1 for ACT)
  --policy.n_action_steps=100   # Steps executed per inference call
  --policy.dim_model=512         # Transformer hidden dimension
  --policy.n_heads=8             # Multi-head attention heads
  --policy.dim_feedforward=3200  # Feed-forward expansion dimension
  --policy.n_encoder_layers=4
  --policy.n_decoder_layers=1
  --policy.use_vae=true          # Enable VAE (KL-divergence loss)
  --policy.latent_dim=32         # VAE latent space size
  --policy.n_vae_encoder_layers=4
  --policy.kl_weight=10.0        # Weight of KL loss term
  --policy.dropout=0.1
  --policy.vision_backbone=resnet18
  --policy.optimizer_lr=1e-5
  --policy.optimizer_lr_backbone=1e-5
  --policy.optimizer_weight_decay=1e-4

  # ── Training schedule ────────────────────────────────────────────────────────
  --seed=1000                    # Reproducibility seed
  --batch_size=8                 # Mini-batch size; 8 works on CPU/MPS
  --steps=100000                 # Total gradient update steps
  --log_freq=200                 # Log metrics every N steps
  --save_freq=20000              # Save checkpoint every N steps
  --save_checkpoint=true         # Enable checkpoint saving
  --num_workers=4                # DataLoader parallel workers

  # ── Output directory ─────────────────────────────────────────────────────────
  --output_dir="$OUTPUT_DIR"     # Where checkpoints + logs go

  # ── W&B (disabled by default) ────────────────────────────────────────────────
  --wandb.enable=false           # Enable with: export WANDB_API_KEY=... then set true
)

# ── Dataset: HF repo_id takes priority over local root ──────────────────────
if [[ -n "$DATASET_REPO_ID" ]]; then
  CMD+=(--dataset.repo_id="$DATASET_REPO_ID")
  # dataset.root stays unset → LeRobot will download to $HF_LEROBOT_HOME
else
  # Local dataset: repo_id is still required by DatasetConfig but will be
  # overridden by root. Use a placeholder that matches the folder name.
  DATASET_FOLDER="$(basename "$DATASET_PATH")"
  CMD+=(
    --dataset.repo_id="local/$DATASET_FOLDER"   # placeholder id (not downloaded)
    --dataset.root="$DATASET_PATH"              # actual data location
  )
fi

# ── Execute ──────────────────────────────────────────────────────────────────
echo "Running: ${CMD[*]}"
echo ""
exec "${CMD[@]}"

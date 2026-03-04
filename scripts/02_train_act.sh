#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 02_train_act.sh — Train an ACT policy via LeRobot (offline, local dataset)
#
# Usage:
#   export DATASET_PATH="/path/to/lerobot/a23v/failure-policy-implementation-training"
#   bash scripts/02_train_act.sh
#
# All flags were verified against lerobot 0.4.4 with:
#   lerobot-train --help  (output saved to docs/lerobot_train_help.txt)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Activate venv ─────────────────────────────────────────────────────────────
VENV="$REPO_ROOT/.venv"
if [[ -d "$VENV" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "❌  No .venv at $VENV — run: /usr/local/bin/python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# ── Load optional env file ────────────────────────────────────────────────────
ENV_FILE="$REPO_ROOT/configs/env"
if [[ -f "$ENV_FILE" ]]; then
  echo "📄 Loading env vars from $ENV_FILE"
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

# ── Require DATASET_PATH ──────────────────────────────────────────────────────
DATASET_PATH="${DATASET_PATH:-}"
if [[ -z "$DATASET_PATH" ]]; then
  echo ""
  echo "❌  DATASET_PATH is not set."
  echo ""
  echo "  Point it to the full path of your local LeRobot dataset, e.g.:"
  echo "    export DATASET_PATH=\"\$HOME/.cache/huggingface/lerobot/a23v/failure-policy-implementation-training\""
  echo ""
  exit 1
fi

# Resolve to absolute path
DATASET_PATH="$(realpath "$DATASET_PATH")"

if [[ ! -d "$DATASET_PATH" ]]; then
  echo "❌  DATASET_PATH does not exist: $DATASET_PATH"
  exit 1
fi

if [[ ! -f "$DATASET_PATH/meta/info.json" ]]; then
  echo "❌  $DATASET_PATH does not look like a LeRobot dataset (missing meta/info.json)"
  exit 1
fi

# ── Derive dataset.repo_id and dataset.root from the path ─────────────────────
# LeRobot expects: root / repo_id  (where repo_id = "<owner>/<name>")
# e.g. ~/.cache/huggingface/lerobot  /  a23v/failure-policy-implementation-training
DATASET_NAME="$(basename "$DATASET_PATH")"
DATASET_OWNER="$(basename "$(dirname "$DATASET_PATH")")"
DATASET_ROOT="$(dirname "$(dirname "$DATASET_PATH")")"
REPO_ID="$DATASET_OWNER/$DATASET_NAME"

# ── Optional overrides ────────────────────────────────────────────────────────
DEVICE="${DEVICE:-cpu}"           # cpu | cuda | mps
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-100000}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/train/$TIMESTAMP}"
# NOTE: do NOT mkdir OUTPUT_DIR — lerobot-train creates it and errors if it already exists

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║           FIPER — ACT Training (Milestone 1)         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Dataset repo_id : $REPO_ID"
echo "  Dataset root    : $DATASET_ROOT"
echo "  Device          : $DEVICE"
echo "  Batch size      : $BATCH_SIZE"
echo "  Steps           : $MAX_STEPS"
echo "  Output dir      : $OUTPUT_DIR"
echo ""

# ── Build command (all flags verified via lerobot-train --help) ───────────────
CMD=(lerobot-train

  # Dataset — local path split into root + repo_id (verified: --dataset.repo_id, --dataset.root)
  --dataset.repo_id="$REPO_ID"
  --dataset.root="$DATASET_ROOT"

  # Policy — select ACT (verified: --policy.type, --policy.device, --policy.push_to_hub)
  --policy.type=act
  --policy.device="$DEVICE"
  --policy.push_to_hub=false

  # ACT architecture — verified flags from --help and ACTConfig source
  --policy.chunk_size=100
  --policy.n_obs_steps=1
  --policy.n_action_steps=100
  --policy.dim_model=512
  --policy.n_heads=8
  --policy.dim_feedforward=3200
  --policy.n_encoder_layers=4
  --policy.n_decoder_layers=1
  --policy.use_vae=true
  --policy.latent_dim=32
  --policy.n_vae_encoder_layers=4
  --policy.kl_weight=10.0
  --policy.dropout=0.1
  --policy.vision_backbone=resnet18
  --policy.optimizer_lr=1e-5
  --policy.optimizer_lr_backbone=1e-5

  # Training schedule (verified: --steps, --batch_size, --log_freq, --save_freq, --save_checkpoint, --num_workers, --seed, --output_dir)
  --steps="$MAX_STEPS"
  --batch_size="$BATCH_SIZE"
  --log_freq=200
  --save_freq=20000
  --save_checkpoint=true
  --num_workers=0
  --seed=1000
  --output_dir="$OUTPUT_DIR"

  # W&B disabled (verified: --wandb.enable)
  --wandb.enable=false
)

echo "Running: ${CMD[*]}"
echo ""
exec "${CMD[@]}"

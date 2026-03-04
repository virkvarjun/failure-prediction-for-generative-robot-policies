# Failure Prediction for Generative Robot Policies

Reproducible baseline for **"Failure Prediction at Runtime for Generative Robot Policies (FIPER)"**.

> **Milestone 1 (this branch):** Train an ACT policy with HuggingFace LeRobot — offline training only, no robot hardware required.

---

## Project Structure

```
.
├── configs/
│   ├── env.example          # Template for environment variables
│   └── train_act.yaml       # Documented ACT config (maps to CLI flags)
├── scripts/
│   ├── 01_check_env.py      # Verify your environment before training
│   ├── 02_train_act.sh      # Train ACT via lerobot-train
│   └── 03_smoke_test_checkpoint.py  # Load checkpoint + dummy inference
├── data/                    # Local datasets (gitignored)
├── outputs/                 # Checkpoints and logs (gitignored)
├── requirements.txt
└── README.md
```

---

## 1. Clone & Setup

```bash
# Clone the repo
git clone https://github.com/virkvarjun/failure-prediction-for-generative-robot-policies.git
cd failure-prediction-for-generative-robot-policies

# Create a virtual environment — must be Python 3.12 (3.13/3.14 breaks draccus)
# On macOS with Homebrew: /usr/local/bin/python3.12 -m venv .venv
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install all dependencies (lerobot with ACT extras)
pip install -r requirements.txt
```

---

## 2. HuggingFace Login

LeRobot downloads datasets and optionally uploads models via HuggingFace Hub.
Log in once per machine:

```bash
huggingface-cli login
# Paste your HF token from https://huggingface.co/settings/tokens
```

---

## 3. Point to Your Dataset

### Option A — HuggingFace Dataset (preferred)

Your dataset must be uploaded to HuggingFace under your account (`a23v`):

```bash
export DATASET_REPO_ID="a23v/<your-dataset-name>"
# Example: export DATASET_REPO_ID="a23v/so101_smileyface"
```

LeRobot will download it automatically on first run.

### Option B — Local Dataset (fallback)

Place your LeRobot-format dataset inside `data/`:

```bash
# Move or symlink your dataset here:
# data/<your-dataset-name>/  ← must contain LeRobot parquet files

export DATASET_PATH="data/<your-dataset-name>"
# Example: export DATASET_PATH="data/so101_smileyface"
```

---

## 4. Check Your Environment

```bash
source .venv/bin/activate
python scripts/01_check_env.py
```

Expected output: all imports print `✓` and `✅ All checks passed.`

---

## 5. Train

```bash
source .venv/bin/activate

# Set your dataset (only one of these is needed):
export DATASET_REPO_ID="a23v/<your-dataset-name>"
# OR
export DATASET_PATH="data/<your-dataset-name>"

# Optional overrides:
export DEVICE="cpu"                          # cpu | cuda | mps (default: cpu)
export OUTPUT_DIR="outputs/train/my_run"     # default: outputs/train/<timestamp>

bash scripts/02_train_act.sh
```

Training prints metrics every 200 steps and saves a checkpoint every 20,000 steps.

---

## 6. Outputs & Resuming

All checkpoints and logs go into `$OUTPUT_DIR` (default: `outputs/train/<timestamp>/`):

```
outputs/train/20240315_143022/
├── checkpoints/
│   ├── 020000/         # checkpoint at step 20000
│   │   └── pretrained_model/
│   └── last/           # symlink to most recent checkpoint
└── train_config.json
```

### Resume a run

```bash
export OUTPUT_DIR="outputs/train/<your-previous-run-dir>"
# Then add resume flag:
# Edit 02_train_act.sh and temporarily add: --resume=true
bash scripts/02_train_act.sh
```

> **Note:** When resuming, LeRobot uses the config saved in the checkpoint directory, ignoring most CLI flags.

---

## 7. Smoke Test (after training)

```bash
source .venv/bin/activate
python scripts/03_smoke_test_checkpoint.py
```

Loads the most recent checkpoint and runs a dummy forward pass to confirm the model produces valid action tensors. No robot required.

To test a specific checkpoint:
```bash
CHECKPOINT_DIR="outputs/train/<run>/checkpoints/last" \
  python scripts/03_smoke_test_checkpoint.py
```

---

## 8. SO101 Teleoperation

This covers running the SO101 leader/follower arms before recording a dataset.

### Install hardware SDK

```bash
source .venv/bin/activate
pip install "lerobot[feetech]"   # Feetech servo SDK — not on PyPI, via extras
```

### Find your ports

```bash
ls /dev/tty.usbmodem*   # run with both arms plugged in
```

For this machine the ports are:
- **Leader** (teleop): `/dev/tty.usbmodem5AE60583121`
- **Follower** (robot): `/dev/tty.usbmodem5AE60798501`

Calibration IDs (from `~/.cache/huggingface/lerobot/calibration/`):
- follower: `so101_follower_1`
- leader: `so101_leader_1`

### Teleoperate

```bash
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5AE60798501 --robot.id=so101_follower_1 --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5AE60583121 --teleop.id=so101_leader_1
```

If you see a calibration mismatch prompt, just press ENTER to use the existing calibration file.

### If it crashes with `No status packet`

Both arms connect but the follower drops during the first read. Try:
1. Reseat the follower USB cable firmly
2. Power cycle the follower arm (unplug/replug 12V, not USB)
3. Lower the loop rate to reduce bus pressure:
   ```bash
   lerobot-teleoperate ... --fps 30
   ```

### If ports are swapped (wrong arm moves)

Flip the two port values in the command above.

---

## 9. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `DATASET_REPO_ID` | *(none)* | HF dataset repo id, e.g. `a23v/my-dataset` |
| `DATASET_PATH` | *(none)* | Path to local LeRobot dataset directory |
| `OUTPUT_DIR` | `outputs/train/<timestamp>` | Where checkpoints + logs are saved |
| `DEVICE` | `cpu` | Compute device: `cpu`, `cuda`, `mps` |
| `HF_HOME` | `~/.cache/huggingface` | HF cache directory |

Copy `configs/env.example` to `.env` and fill it in, then `source .env`.

---

## 9. Troubleshooting

### ❌ `No dataset specified`
Neither `DATASET_REPO_ID` nor `DATASET_PATH` is set. Export one of them before running the script.

### ❌ `lerobot-train: command not found`
The venv is not activated, or lerobot is not installed.
```bash
source .venv/bin/activate
pip install -r requirements.txt
which lerobot-train    # should print .venv/bin/lerobot-train
```

### ❌ `TypeError: str | None is not callable`
draccus is incompatible with Python 3.13/3.14. Rebuild your venv with Python 3.12:
```bash
rm -rf .venv
/usr/local/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "lerobot[feetech]"
```

### ❌ `ModuleNotFoundError: No module named 'scservo_sdk'`
Run: `pip install "lerobot[feetech]"`

### ❌ `Missing motor IDs` on teleoperate
Ports are swapped. Flip the `--robot.port` and `--teleop.port` values.

### ❌ `No status packet` during teleoperate loop
Both arms connected but follower dropped mid-read. Reseat USB/power on follower, or add `--fps 30`.

### ⚠️ MPS / Metal errors on macOS
Set `DEVICE=cpu` to bypass Metal:
```bash
export DEVICE=cpu
bash scripts/02_train_act.sh
```

### ⚠️ Out of memory on CPU
Reduce batch size: edit `--batch_size=4` (or lower) in `scripts/02_train_act.sh`.

### ❌ Dataset not found on HF Hub
Ensure you have run `huggingface-cli login` and your dataset is public (or you have access). Double-check the `DATASET_REPO_ID` spelling.

---

## Hardware Requirements

| | Minimum | Recommended |
|---|---|---|
| **CPU training** | 8 GB RAM | 16 GB RAM |
| **GPU (optional)** | CUDA 11.8+ | CUDA 12.x, 8 GB VRAM |
| **macOS** | MPS / CPU | Apple Silicon (M1+) |

---

## Citation

```bibtex
@article{fiper2024,
  title   = {Failure Prediction at Runtime for Generative Robot Policies},
  year    = {2024},
}
```

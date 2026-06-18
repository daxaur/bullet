#!/usr/bin/env bash
# =====================================================================================
# ARES NNUE — one-shot cloud trainer for a rented GPU box (vast.ai / RunPod / any Ubuntu+CUDA).
# Trains the REAL Ares net in a SINGLE uninterrupted run (no Kaggle 9h cap), using all CPU
# cores for feature gen. Produces checkpoints/ares-net-*/weights.bin -> download -> convert + SPRT.
#
# RECOMMENDED BOX: vast.ai INTERRUPTIBLE, RTX 3090, >=32 vCPUs, >=32GB RAM, ~50GB disk.
#   (Our bottleneck is CPU cores, not GPU — pick many vCPUs over a fancier GPU.)
#   Use an image with the CUDA *toolkit* (e.g. nvidia/cuda:12.4.1-devel-ubuntu22.04).
#
# USAGE (after SSH into the box) — hosted in the PUBLIC bullet fork so the box can fetch it:
#   curl -fsSL https://raw.githubusercontent.com/daxaur/bullet/ares/cloud/run_ares.sh -o run_ares.sh
#   chmod +x run_ares.sh
#   SUPERBATCHES=240 ./run_ares.sh            # full mature run
#   # resume after an eviction:
#   SUPERBATCHES=240 START=121 RESUME=$PWD/bullet/checkpoints/ares-net-120 ./run_ares.sh
#
# SAFETY: vast.ai bills per-second while running + storage while the box exists. When training
# finishes (or you've downloaded weights.bin), DESTROY the instance from the vast.ai console so it
# stops billing. This script does NOT auto-destroy (that needs your API key — kept off the box).
# =====================================================================================
set -euo pipefail

# ---- knobs (env-overridable) ----
SUPERBATCHES="${SUPERBATCHES:-240}"     # total target (cosine LR anneals over this)
START="${START:-1}"                     # resume start superbatch
SAVE_RATE="${SAVE_RATE:-20}"            # checkpoint every N superbatches
RESUME="${RESUME:-}"                    # checkpoint dir to resume from (empty = fresh)
BULLET_BRANCH="${BULLET_BRANCH:-ares}"
BINPACK_URL="${BINPACK_URL:-https://huggingface.co/datasets/linrock/test80-2024/resolve/main/test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst}"
WORK="${WORK:-$PWD}"

say(){ printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

say "0. environment"
nvidia-smi -L || { echo "NO GPU visible"; exit 1; }
nproc | xargs echo "CPU cores:"
# CUDA toolkit + driver lib (bullet's CUDA backend needs CUDA_PATH + to link -lcuda)
CUDA="${CUDA_PATH:-/usr/local/cuda}"
[ -d "$CUDA" ] || CUDA="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -r | head -1 || true)"
[ -n "$CUDA" ] && [ -d "$CUDA" ] || { echo "CUDA toolkit not found"; exit 1; }
export CUDA_PATH="$CUDA"; export PATH="$CUDA/bin:$PATH"
LIBDIRS=""; for d in "$CUDA/lib64/stubs" /usr/lib/x86_64-linux-gnu /usr/lib64 "$CUDA/lib64" /usr/local/nvidia/lib64; do
  if ls "$d"/libcuda.so* >/dev/null 2>&1; then LIBDIRS="$LIBDIRS -L $d"; export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; fi
done
[ -n "$LIBDIRS" ] || { echo "libcuda.so* not found"; exit 1; }
export RUSTFLAGS="${RUSTFLAGS:-} $LIBDIRS"
nvcc --version | tail -1

say "1. rust"
command -v cargo >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"; cargo --version

say "2. bullet fork (examples/ares.rs + AresThreats)"
cd "$WORK"
[ -d bullet ] || git clone --branch "$BULLET_BRANCH" --single-branch https://github.com/daxaur/bullet bullet
( cd bullet && git pull --ff-only && git log --oneline -1 )

say "3. binpack (streamed fresh)"
mkdir -p bullet/data
if [ ! -f bullet/data/ares.binpack ]; then
  command -v zstd >/dev/null 2>&1 || (apt-get -qq update && apt-get -qq install -y zstd wget)
  wget -q -O bullet/data/ares.binpack.zst "$BINPACK_URL"
  ( cd bullet/data && zstd -d --rm -f ares.binpack.zst -o ares.binpack )
fi
ls -la bullet/data/

say "4. TRAIN (single uninterrupted run; uses all cores)"
RESUME_ENV=""; [ -n "$RESUME" ] && RESUME_ENV="ARES_RESUME=$RESUME"
cd bullet
env ARES_SUPERBATCHES="$SUPERBATCHES" ARES_START_SUPERBATCH="$START" \
    ARES_SAVE_RATE="$SAVE_RATE" ARES_BINPACK="data/ares.binpack" $RESUME_ENV \
    cargo run --release --example ares --features cuda

say "5. DONE — checkpoints:"
ls -laR checkpoints/ | tail -30
echo
echo ">>> Download checkpoints/ares-net-$SUPERBATCHES/weights.bin to your Mac, then:"
echo "    ./.venv/bin/python pipeline/convert_net.py ~/Downloads/weights.bin /tmp/ares.nnue"
echo "    ./.venv/bin/python pipeline/eval_parity.py ~/Downloads/weights.bin"
echo "    ARES_ELO0=0 ARES_ELO1=4 ./.venv/bin/python pipeline/ab_net.py ours /tmp/ares.nnue   # SPRT vs v60"
echo ">>> THEN DESTROY THIS INSTANCE in the vast.ai console so it stops billing."

#!/usr/bin/env bash
# =====================================================================================
# ARES NNUE v2 — DATA-SCALE cloud trainer. Fixes v1's overfitting (-225 Elo) by training
# on 6 months of linrock test80-2024 (~6-9B unique positions) instead of 1 month.
#
# WHY: v1 trained 240 superbatches (24B position-views) over ONE month (~1B unique) =
# ~24 epochs = severe overfit (eval extremes grew net-40 std616 -> net-240 std1069,
# corr +0.89 w/ v60 but plays -225). With 6 months (~7B unique), the SAME 240 superbatches
# = ~3 epochs = healthy. Trainer config is UNCHANGED; only the data scale changes.
#
# RECOMMENDED BOX: vast.ai, RTX 3090 (or 4090), >=16 vCPUs, >=32GB RAM, **>=250GB disk**.
#   Disk matters: 6 months ~52GB compressed -> ~89GB decompressed -> ~178GB peak during
#   interleave (sources + output). After interleave the sources are deleted (~89GB for train).
#   Use a PyTorch/CUDA-*devel* image (prebuilt; avoids docker_build errors).
#
# USAGE (after SSH into the box):
#   curl -fsSL https://raw.githubusercontent.com/daxaur/bullet/ares/cloud/run_ares_v2.sh -o run_ares_v2.sh
#   chmod +x run_ares_v2.sh
#   SUPERBATCHES=240 ./run_ares_v2.sh
#   # resume after an eviction:
#   SUPERBATCHES=240 START=121 RESUME=$PWD/bullet/checkpoints/ares-net-120 ./run_ares_v2.sh
#
# SAFETY: bills per-second while running + storage while the box exists. DESTROY the instance
# from the vast.ai console when done. Does NOT auto-destroy (needs API key, kept off the box).
# =====================================================================================
set -euo pipefail

# ---- knobs (env-overridable) ----
SUPERBATCHES="${SUPERBATCHES:-240}"     # total target (cosine LR anneals over this); 240/~7B = ~3 epochs
START="${START:-1}"
SAVE_RATE="${SAVE_RATE:-20}"
RESUME="${RESUME:-}"
BULLET_BRANCH="${BULLET_BRANCH:-ares}"
WORK="${WORK:-$PWD}"
# 6 months jan-jun 2024, .min-v2.v6 (Leela T80, Syzygy 6+7p rescored, SF best-move, v6 dedup).
# All distinct months -> no overlap/double-count. Override MONTHS to change the mix.
MONTHS="${MONTHS:-01-jan 02-feb 03-mar 04-apr 05-may 06-jun}"
HF_BASE="https://huggingface.co/datasets/linrock/test80-2024/resolve/main"
INTERLEAVE_URL="https://raw.githubusercontent.com/linrock/nnue-tools/master/training-data/interleave_binpacks.py"

say(){ printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

say "0. environment"
nvidia-smi -L || { echo "NO GPU visible"; exit 1; }
nproc | xargs echo "CPU cores:"
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
df -h "$WORK" | tail -1

say "1. rust"
command -v cargo >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"; cargo --version

say "2. bullet fork (examples/ares.rs + AresThreats + multi-binpack concat)"
cd "$WORK"
[ -d bullet ] || git clone --branch "$BULLET_BRANCH" --single-branch https://github.com/daxaur/bullet bullet
( cd bullet && git pull --ff-only && git log --oneline -1 )

say "3. data: download 6 months + interleave into ONE shuffled stream"
mkdir -p bullet/data
cd bullet/data
command -v zstd >/dev/null 2>&1 || (apt-get -qq update && apt-get -qq install -y zstd wget python3)
MIX="ares-mix.binpack"
if [ ! -f "$MIX" ]; then
  FILES=""
  for M in $MONTHS; do
    BP="test80-2024-${M}-2tb7p.min-v2.v6.binpack"
    if [ ! -f "$BP" ]; then
      echo ">> fetching $M"
      wget -q --show-progress -O "$BP.zst" "$HF_BASE/${BP}.zst"
      zstd -d --rm -f "$BP.zst" -o "$BP"
    fi
    FILES="$FILES $BP"
  done
  echo ">> decompressed months:"; ls -la *.binpack; df -h . | tail -1
  echo ">> interleaving (size-weighted random alternation) -> $MIX"
  wget -q -O interleave_binpacks.py "$INTERLEAVE_URL"
  python3 interleave_binpacks.py $FILES "$MIX"
  echo ">> interleave done; freeing per-month files"
  rm -f $FILES
fi
ls -la "$MIX"; df -h . | tail -1
cd "$WORK"

say "4. TRAIN (single run; 240 superbatches over ~7B unique = ~3 epochs, no overfit)"
RESUME_ENV=""; [ -n "$RESUME" ] && RESUME_ENV="ARES_RESUME=$RESUME"
cd bullet
env ARES_SUPERBATCHES="$SUPERBATCHES" ARES_START_SUPERBATCH="$START" \
    ARES_SAVE_RATE="$SAVE_RATE" ARES_BINPACK="data/ares-mix.binpack" $RESUME_ENV \
    cargo run --release --example ares --features cuda

say "5. DONE — checkpoints:"
ls -laR checkpoints/ | tail -30
echo
echo ">>> Pull checkpoints/ares-net-*/optimiser_state/weights.bin to your Mac, then:"
echo "    ./.venv/bin/python pipeline/convert_net.py ~/Downloads/weights.bin /tmp/ares.nnue"
echo "    ./.venv/bin/python pipeline/eval_parity.py ~/Downloads/weights.bin"
echo "    ARES_ELO0=0 ARES_ELO1=4 ./.venv/bin/python pipeline/ab_net.py ours /tmp/ares.nnue   # SPRT vs v60"
echo ">>> THEN DESTROY THIS INSTANCE in the vast.ai console so it stops billing."

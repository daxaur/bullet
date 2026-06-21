#!/usr/bin/env bash
# =====================================================================================
# ARES NNUE v3 — BEAT-RECKLESS run. Battle-plan recipe (Net-Training-Battle-Plan.md):
#   DATA-SCALE (proven lever) + field-standard trainer-config corrections. No arch change
#   (FT stays 768; widening to 1024 is a separate run-2). Trains on ~12 months of clean
#   Leela-T80 binpacks (test80-2024 6mo + test80-2023 6 clean .min-v2.v6mo ~= 12-14B unique,
#   roughly DOUBLE v2's 7B) via multi-binpack CONCAT (ares.rs new_concat_multiple + 4GB
#   shuffle buffer) — no interleave step, half the disk.
#
#   Config already baked into examples/ares.rs (committed): WDL 0.25 (was 0.75), power-loss 2.5
#   (was MSE), label smoothing clip[0.01,0.99], warmup-200 on cosine LR, loader buffer 4GB/8thr.
#
# RECOMMENDED BOX: vast.ai 1xGPU (3060/3070/A4000-class), >=24 vCPU, >=24GB RAM, **>=300GB disk**.
#   On-the-fly training ran ~560K pos/s on a 3060 (v2). 300 SB = 30B views ~= 15h ~= $1.5-3.
#   Use a prebuilt PyTorch/CUDA-devel image (avoids host docker-build 'retries exceeded').
#
# USAGE: curl -fsSL https://raw.githubusercontent.com/daxaur/bullet/ares/cloud/run_ares_v3.sh -o run_ares_v3.sh
#        chmod +x run_ares_v3.sh && SUPERBATCHES=300 ./run_ares_v3.sh
#   resume: SUPERBATCHES=300 START=121 RESUME=$PWD/bullet/checkpoints/ares-net-120 ./run_ares_v3.sh
# SAFETY: destroy the instance when done (storage + GPU bill while it exists).
# =====================================================================================
set -euo pipefail

SUPERBATCHES="${SUPERBATCHES:-300}"     # ~2.4 epochs over ~12-14B unique
START="${START:-1}"
SAVE_RATE="${SAVE_RATE:-20}"
RESUME="${RESUME:-}"
BULLET_BRANCH="${BULLET_BRANCH:-ares}"
WORK="${WORK:-$PWD}"
HF24="https://huggingface.co/datasets/linrock/test80-2024/resolve/main"
HF23="https://huggingface.co/datasets/linrock/test80-2023/resolve/main"
# one variant per month, all v6-filtered (clean unique-position accounting):
FILES_2024="01-jan 02-feb 03-mar 04-apr 05-may 06-jun"             # all -2tb7p.min-v2.v6
FILES_2023="06-jun 07-jul 09-sep 10-oct 11-nov 12-dec"            # all -2tb7p.min-v2.v6 (clean)

say(){ printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

say "0. environment"
nvidia-smi -L || { echo "NO GPU"; exit 1; }
nproc | xargs echo "CPU cores:"
CUDA="${CUDA_PATH:-/usr/local/cuda}"; [ -d "$CUDA" ] || CUDA="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -r | head -1 || true)"
[ -n "$CUDA" ] && [ -d "$CUDA" ] || { echo "no CUDA toolkit"; exit 1; }
export CUDA_PATH="$CUDA"; export PATH="$CUDA/bin:$PATH"
LIBDIRS=""; for d in "$CUDA/lib64/stubs" /usr/lib/x86_64-linux-gnu /usr/lib64 "$CUDA/lib64" /usr/local/nvidia/lib64; do
  if ls "$d"/libcuda.so* >/dev/null 2>&1; then LIBDIRS="$LIBDIRS -L $d"; export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; fi
done
[ -n "$LIBDIRS" ] || { echo "no libcuda"; exit 1; }
export RUSTFLAGS="${RUSTFLAGS:-} $LIBDIRS"; nvcc --version | tail -1; df -h "$WORK" | tail -1

say "1. rust"
command -v cargo >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"; cargo --version

say "2. bullet fork (v3 config in examples/ares.rs)"
cd "$WORK"
[ -d bullet ] || git clone --branch "$BULLET_BRANCH" --single-branch https://github.com/daxaur/bullet bullet
( cd bullet && git pull --ff-only && git log --oneline -1 )

say "3. data: download 12 months (concat, no interleave)"
mkdir -p bullet/data; cd bullet/data
command -v zstd >/dev/null 2>&1 || (apt-get -qq update && apt-get -qq install -y zstd wget)
PATHS=""
fetch(){  # base, mon-tag, repo-year
  local base="$1" m="$2" yr="$3"
  local bp="test80-${yr}-${m}-2tb7p.min-v2.v6.binpack"
  if [ ! -f "$bp" ]; then
    echo ">> $yr $m"; wget -q -c -O "$bp.zst" "$base/${bp}.zst"; zstd -d --rm -f "$bp.zst" -o "$bp"
  fi
  PATHS="${PATHS:+$PATHS,}$PWD/$bp"
}
for m in $FILES_2024; do fetch "$HF24" "$m" 2024; done
for m in $FILES_2023; do fetch "$HF23" "$m" 2023; done
echo ">> decompressed:"; ls -la *.binpack; df -h . | tail -1
echo ">> ARES_BINPACK = $PATHS"
cd "$WORK"

say "4. TRAIN ($SUPERBATCHES sb, ~12-14B unique, concat + 4GB shuffle, WDL0.25 power2.5 warmup)"
RESUME_ENV=""; [ -n "$RESUME" ] && RESUME_ENV="ARES_RESUME=$RESUME"
cd bullet
env ARES_SUPERBATCHES="$SUPERBATCHES" ARES_START_SUPERBATCH="$START" \
    ARES_SAVE_RATE="$SAVE_RATE" ARES_BINPACK="$PATHS" $RESUME_ENV \
    cargo run --release --example ares --features cuda

say "5. DONE — checkpoints:"; ls -laR checkpoints/ | tail -20
echo ">>> pull checkpoints/ares-net-$SUPERBATCHES/optimiser_state/weights.bin -> convert -> SPRT vs v60. THEN DESTROY THE BOX."

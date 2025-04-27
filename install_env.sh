#!/usr/bin/env bash
###############################################################################
# fasTTalk – one-shot installation script
# This script:
#   • creates/activates a “fasttalk” conda env (Python 3.11)
#   • installs all required packages (CUDA-12.6, PyTorch, PyTorch3D, etc.)
#   • downloads & unzips the FLAME2023 assets
#   • applies the two Chumpy hot-fixes
#   • patches transformers’ processing_wav2vec2.py
#   • leaves the user inside the freshly-configured environment
###############################################################################
set -euo pipefail

# ---------- colourful logging helpers ----------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()   { echo -e "${GREEN}[fasTTalk]${NC} $*"; }
warn()  { echo -e "${YELLOW}[fasTTalk-WARN]${NC} $*"; }
error() { echo -e "${RED}[fasTTalk-ERROR]${NC} $*"; exit 1; }

# ---------- check conda ----------
command -v conda &>/dev/null || error "conda not found. Install Miniconda/Anaconda first."

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------- create & enter env ----------
if conda env list | grep -qE '^[^#]*\bfasttalk\b'; then
    warn "Conda env “fasttalk” already exists – re-using it."
else
    log  "Creating conda environment “fasttalk” (Python 3.11)…"
    conda create -y -n fasttalk python=3.11
fi

log "Activating environment…"
conda activate fasttalk

# ---------- basic Python libraries ----------
log "Installing core Python dependencies…"
pip install --upgrade pip
pip install git+https://github.com/MPI-IS/mesh.git          # MPI-IS mesh lib
pip install torch torchvision torchaudio                    # default CUDA wheel
conda install -y -c conda-forge ffmpeg                      # media support
conda install -y -c "nvidia/label/cuda-12.6" cuda-toolkit ninja
ln -sf "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"            # allow CUDA to find libs
conda env config vars set CUDA_HOME="$CONDA_PREFIX"

# ---------- (re)activate to pick up CUDA_HOME ----------
conda deactivate && conda activate fasttalk

# ---------- remaining Python deps ----------
log "Installing PyTorch3D (this may compile from source)…"
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

log "Installing miscellaneous Python packages…"
pip install tensorboardX einops scipy librosa tqdm \
            sympy==1.13.1 transformers \
            trimesh pyrender pyopengl pyglet opencv-python \
            pyyaml scikit-image wandb matplotlib \
            chumpy datasets
pip install --upgrade gradio

# ---------- FLAME2023 assets ----------
ASSETS_DIR="$PWD/assets"
FLAME_ZIP="FLAME2023.zip"
FLAME_URL="https://drive.google.com/uc?export=download&id=1xKqhqhlozyExenBs9ew7scEjPgrLe-Io"

mkdir -p "$ASSETS_DIR"
if [[ ! -d "$ASSETS_DIR/FLAME2023" ]]; then
    log "Downloading FLAME2023 (~150 MB)…"
    # requires `gdown`; install temporarily if absent
    python - <<'PY'
import importlib.util, subprocess, sys
spec = importlib.util.find_spec("gdown")
sys.exit(0 if spec is not None else 1)
PY
    [[ $? -eq 0 ]] || pip install --quiet gdown
    gdown "$FLAME_URL" -O "$ASSETS_DIR/$FLAME_ZIP"
    log "Unzipping FLAME2023 into assets/ ..."
    unzip -q "$ASSETS_DIR/$FLAME_ZIP" -d "$ASSETS_DIR"
    rm "$ASSETS_DIR/$FLAME_ZIP"
else
    warn "FLAME2023 already present – skipping download."
fi

# ---------- locate site-packages ----------
SITEPKG=$(python - <<'PY'
import sysconfig, sys; print(sysconfig.get_paths()["purelib"])
PY)
[[ -d "$SITEPKG" ]] || error "Could not locate site-packages."

# ---------- patch transformers (Wav2Vec2 tokenizer tweak) ----------
W2V_FILE=$(python - <<'PY'
from importlib.util import find_spec, spec_from_file_location
import sys, pathlib
spec = find_spec("transformers.models.wav2vec2.processing_wav2vec2")
print(pathlib.Path(spec.origin))
PY)

log "Patching transformers’ processing_wav2vec2.py…"
sed -i.bak \
    's@Wav2Vec2CTCTokenizer\.from_pretrained([^)]*@Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h"@' \
    "$W2V_FILE"

# ---------- patch Chumpy deprecated stuff ----------
CH_INIT="$SITEPKG/chumpy/__init__.py"
CH_CH="$SITEPKG/chumpy/ch.py"

log "Applying Chumpy hot-fixes…"
# comment out deprecated numpy aliases (line may move; sed is safer)
sed -i.bak '/from numpy import .*bool/s/^/# /' "$CH_INIT"

# monkey-patch inspect.getargspec if missing
if ! grep -q "getfullargspec" "$CH_CH"; then
    sed -i.bak '/^import inspect/a \
if not hasattr(inspect, "getargspec"):\n    inspect.getargspec = inspect.getfullargspec' "$CH_CH"
fi

# ---------- final message ----------
log "Installation finished! Remember to:  conda activate fasttalk"
echo -e "${GREEN}✔ All steps completed successfully.${NC}"

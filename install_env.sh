#!/usr/bin/env bash
###############################################################################
# fasTTalk – installation script (v2-emoji)
# - Creates/activates “fasttalk” conda env (Python 3.11)
# - Installs CUDA 12.6, PyTorch, PyTorch3D, etc.
# - Downloads & unpacks FLAME2023 assets
# - Patches transformers (Wav2Vec2) and Chumpy
###############################################################################
set -eo pipefail                         # strict: abort on errors + bad pipes

# ───────────────────────  pretty logging helpers  ────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()   { echo -e "${GREEN}🟢 [fasTTalk]${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠️  [fasTTalk-WARN]${NC} $*"; }
die()   { echo -e "${RED}❌ [fasTTalk-ERROR]${NC} $*"; exit 1; }

# ───────────────────────────  conda detection  ──────────────────────────────
command -v conda &>/dev/null || die "Conda not found. Install Miniconda/Anaconda."

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

# Helper: run any “conda …” command with nounset disabled
conda_safe() {
    set +u          # allow unset vars in Conda hooks
    conda "$@"
    local status=$?
    set -u          # restore (for remainder of script)
    return $status
}

# Switch on nounset for the rest of the script
set -u

###############################################################################
# 1. Conda environment
if conda env list | grep -qE '^[^#]*\bfasttalk\b' ; then
    warn "Environment “fasttalk” already exists – re-using it."
else
    log  "Creating conda environment “fasttalk” (Python 3.11)…"
    conda_safe create -y -n fasttalk python=3.11
fi

log "Activating environment…"
set +u
conda activate fasttalk
set -u

###############################################################################
# 2. Core dependencies
log "Upgrading pip…"
pip install --quiet --upgrade pip

# log "Installing MPI-IS mesh library…"
# pip install git+https://github.com/MPI-IS/mesh.git


log "Installing ffmpeg, CUDA 12.6 toolkit and ninja…"
conda_safe install -y -c conda-forge ffmpeg
conda_safe install -y -c "nvidia/label/cuda-12.6" cuda-toolkit ninja

log "Installing PyTorch (default CUDA wheel)…"
pip install torch torchvision torchaudio

ln -sf "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"          # lib → lib64
conda env config vars set CUDA_HOME="$CONDA_PREFIX"

# Re-activate so CUDA_HOME is exported
set +u
conda deactivate
conda activate fasttalk
set -u

###############################################################################
# 3. Remaining Python deps
log "Installing PyTorch3D (may compile from source)…"
# pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

log "Installing miscellaneous packages…"
pip install --no-build-isolation tensorboardX einops scipy librosa tqdm \
            sympy==1.13.1 transformers \
            trimesh pyrender pyopengl pyglet opencv-python \
            pyyaml scikit-image wandb matplotlib \
            chumpy datasets gdrive

log "Installing Gradio (for web UI)…"
pip install --upgrade gradio

log "Installing vector-quantize-pytorch (for VQGAN model)…"
pip install vector-quantize-pytorch

###############################################################################
# 4. FLAME2023 assets
FLAME_DIR="$PWD"
FLAME_ZIP="$FLAME_DIR/flame_model.zip"
FLAME_URL="1dgDWQB9hbGMrQMTVhIv32s3WZmq1CzbZ"


mkdir -p "$FLAME_DIR"
if [[ ! -d "$FLAME_DIR/assets" ]]; then
    log "Downloading FLAME assets (≈150 MB)…"

    # Check if gdown command exists, install if not
    if ! command -v gdown &> /dev/null; then
        log "gdown not found, installing..."
        # Using python -m pip is robust, just like your original script
        python -m pip install --quiet gdown
    fi

    # Proceed with download and unzip
    gdown --id "$FLAME_URL" -O "$FLAME_ZIP"
    log "Unzipping FLAME…"
    unzip -q "$FLAME_ZIP" -d "$FLAME_DIR"
    rm "$FLAME_ZIP"
else
    warn "flame already exists - skipping download."
fi
###############################################################################
# 5. Patches: transformers & Chumpy
SITEPKG=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')

# ── Wav2Vec2 tokenizer patch ────────────────────────────────────────────────
W2V_FILE=$(python -c "from importlib.util import find_spec; import pathlib; spec = find_spec('transformers.models.wav2vec2.processing_wav2vec2'); print(pathlib.Path(spec.origin))")

log "Patching transformers (processing_wav2vec2.py)…"
sed -i.bak \
 's@Wav2Vec2CTCTokenizer\.from_pretrained([^)]*@Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h"@' \
 "$W2V_FILE"

# ── Chumpy fixes ────────────────────────────────────────────────────────────
CH_INIT="$SITEPKG/chumpy/__init__.py"
CH_CH="$SITEPKG/chumpy/ch.py"

log "Patching Chumpy deprecated NumPy aliases…"
sed -i.bak '/from numpy import .*bool/s/^/# /' "$CH_INIT"

if ! grep -q "getfullargspec" "$CH_CH"; then
    log "Adding inspect.getargspec shim to Chumpy…"
    sed -i.bak '/^import inspect/a \
if not hasattr(inspect, "getargspec"):\n    inspect.getargspec = inspect.getfullargspec' "$CH_CH"
fi

###############################################################################
log "🎉  ✅ Installation complete!   ➜  Run:  conda activate fasttalk"
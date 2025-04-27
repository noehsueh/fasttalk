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

log "Installing MPI-IS mesh library…"
pip install git+https://github.com/MPI-IS/mesh.git

log "Installing PyTorch (default CUDA wheel)…"
pip install torch torchvision torchaudio

log "Installing ffmpeg, CUDA 12.6 toolkit and ninja…"
conda_safe install -y -c conda-forge ffmpeg
conda_safe install -y -c "nvidia/label/cuda-12.6" cuda-toolkit ninja

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
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

log "Installing miscellaneous packages…"
pip install tensorboardX einops scipy librosa tqdm \
            sympy==1.13.1 transformers \
            trimesh pyrender pyopengl pyglet opencv-python \
            pyyaml scikit-image wandb matplotlib \
            chumpy datasets

log "Installing Gradio…"
pip install --upgrade gradio

###############################################################################
# 4. FLAME2023 assets
ASSETS_DIR="$PWD/assets"
FLAME_ZIP="$ASSETS_DIR/FLAME2023.zip"
FLAME_URL="https://drive.google.com/uc?export=download&id=1xKqhqhlozyExenBs9ew7scEjPgrLe-Io"

mkdir -p "$ASSETS_DIR"
if [[ ! -d "$ASSETS_DIR/FLAME2023" ]]; then
    log "Downloading FLAME2023 assets (≈150 MB)…"
    python - <<'PY'
import importlib.util, subprocess, sys
if importlib.util.find_spec("gdown") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "gdown"])
PY
    gdown "$FLAME_URL" -O "$FLAME_ZIP"
    log "Unzipping FLAME2023…"
    unzip -q "$FLAME_ZIP" -d "$ASSETS_DIR"
    rm "$FLAME_ZIP"
else
    warn "FLAME2023 already present – skipping download."
fi

###############################################################################
# 5. Patches: transformers & Chumpy
SITEPKG=$(python - <<'PY'
import sysconfig; print(sysconfig.get_paths()["purelib"])
PY)

# ── Wav2Vec2 tokenizer patch ────────────────────────────────────────────────
W2V_FILE=$(python - <<'PY'
from importlib.util import find_spec
import pathlib
spec = find_spec("transformers.models.wav2vec2.processing_wav2vec2")
print(pathlib.Path(spec.origin))
PY)

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

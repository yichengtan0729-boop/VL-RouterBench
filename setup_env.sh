#!/bin/bash
# VL RouterBench environment setup script (Python dependencies only)

set -e

echo "============================================================"
echo "🚀 VL RouterBench Environment Setup"
echo "============================================================"

ENV_NAME="vl-routerbench"

# 1) Check conda and load conda.sh (so `conda activate` works inside this script)
if ! command -v conda >/dev/null 2>&1; then
  echo "✗ conda was not found. Please install Miniconda/Anaconda and ensure `conda` is on your PATH."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 2) Create the environment (if it doesn't exist)
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "✓ Found existing conda environment: ${ENV_NAME} (skip creation)"
else
  echo "📦 Creating conda environment: ${ENV_NAME} (Python 3.10)"
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

# 3) Activate the environment
echo "✓ Activating environment: ${ENV_NAME}"
conda activate "${ENV_NAME}"

# 4) Install Python dependencies (project-related Python packages only)
echo ""
echo "📥 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Core: deep learning & model related
pip install torch torchvision torchaudio transformers sentence-transformers timm tiktoken

# Core: data processing
pip install numpy pandas pyarrow openpyxl pyyaml tqdm pillow psutil scikit-learn

# Visualization (pure Python; no system font/apt setup here)
pip install matplotlib seaborn adjustText

# 5) Verify installation (Python packages only)
echo ""
echo "============================================================"
echo "✅ Verification"
echo "============================================================"

python << 'PYEOF'
import sys
print(f"Python version: {sys.version}")
print("\nInstalled packages:")

packages = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'torchaudio': 'TorchAudio',
    'transformers': 'Transformers',
    'sentence_transformers': 'Sentence-Transformers',
    'timm': 'timm',
    'sklearn': 'Scikit-learn',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'PIL': 'Pillow',
    'adjustText': 'adjustText',
    'tqdm': 'TQDM',
    'yaml': 'PyYAML',
    'psutil': 'psutil',
    'tiktoken': 'TikToken',
}

for pkg, name in packages.items():
    try:
        if pkg == 'PIL':
            import PIL
            print(f"  ✓ {name}: {PIL.__version__}")
        else:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name}: {ver}")
    except Exception:
        print(f"  ✗ {name}: import failed")
PYEOF

echo ""
echo "============================================================"
echo "✅ Environment setup complete!"
echo "============================================================"
echo ""
echo "Usage:"
echo "  conda activate ${ENV_NAME}"
echo "  cd /opt/data/private/hzhcode/vl_routerbench_v1"
echo "  # Run Steps 1-6"
echo "  bash scripts/run_all.sh"
echo ""


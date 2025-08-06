# LEP (Layer Environment Potential)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)



## Installation

### Prerequisites
- Conda package manager ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda)
- NVIDIA GPU drivers (if using GPU acceleration)
- CUDA Toolkit (compatible with your PyTorch version)

### Step 1: Create Conda Environment
```bash
conda create -n lep_env python=3.9  # Python 3.8 or 3.9 recommended
conda activate lep_env
```


### Step 2: Install PyTorch
Select the appropriate command from [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) based on your hardware configuration:

#### For CUDA 12.x (NVIDIA GPU):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


#### For CPU-only systems:
```bash
pip3 install torch torchvision torchaudio
```


### Step 3: Install LEP Package
```bash
git clone https://github.com/XZ-HaN/LEP.git
cd LEP
```

Install package and dependencies

```bash
python setup.py install
```

Or

```bash
pip install -e .
```


## Verification
Confirm successful installation with:
```python
import lep
import torch

print(f"LEP version: {lep.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

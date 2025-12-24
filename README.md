# Implementation of Split Bregman

Python implementation of the Split Bregman algorithm.

## License

Sustainable Use License from [n8n](https://github.com/n8n-io/n8n).

## Requirements

- Python >= 3.10
- NumPy
- SciPy
- PyWavelets
- **CuPy (required, not installed automatically by this package)**  
  Install the CuPy package that matches your CUDA version (e.g., `cupy-cuda12x`, `cupy-cuda11x`, etc.).
- **gradops** (required; from GitHub)
- **thresholding** (required; from GitHub)

## Installation

### 1) Install CuPy (GPU dependency)

Install the CuPy wheel matching your CUDA version.

### 2) Install required helper packages (from GitHub)

This package depends on gradops and thresholding, which are hosted on GitHub under kejang.
```bash
pip install "gradops @ git+https://github.com/kejang/gradops.git"
pip install "thresholding @ git+https://github.com/kejang/thresholding.git"
```

### 3) Install splitbregman

If you are installing from GitHub:
```bash
pip install "splitbregman @ git+https://github.com/kejang/splitbregman.git"
```

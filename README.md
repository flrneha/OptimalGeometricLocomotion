# Optimal Geometric Locomotion
**Note: This repository is under active development and accompanies the paper [Sub-Riemannian Boundary Value Problems for Optimal Geometric Locomotion](https://olligross.github.io/projects/EfficientLocomotion/EfficientLocomotion_project.html)
by *Oliver Gross, Florine Hartwig, Martin Rumpf, and Peter Schröder*. If you have any questions, feel free to reach out to [Florine Hartwig](https://ins.uni-bonn.de/staff/hartwig)**

This project provides computational tools for modeling and optimizing shape-change-induced motions of slender locomotors.
The code implements a geometric framework based on sub-Riemannian geodesics to compute energy-efficient locomotion gaits.
We can solve three different types of boundary conditions, i.e., fixing initial and target body, restricting to cyclic motion,
or solely prescribing body displacement and orientation. We provide examples as [notebooks](examples/notebooks) and [scripts](examples/scripts) for all three types.


https://github.com/user-attachments/assets/ac881dde-8471-4f46-b02c-c27a619d7847


# Project Setup Guide

This guide will help you set up the development environment for this project, which includes Python dependencies and C++ Python bindings.

## Project Structure

```
.
├── pyproject.toml          # Python dependencies and project metadata
├── environment.yml         # Environment file
├── README.md               # This file
├── <PYCURVE_DIR>/          # C++ module source code
│   ├── CMakeLists.txt
│   └── build/              # Build (created during setup)
├── src/                    # Python source code
└── examples/               # example scripts for all three boundary value problems   
```

## Prerequisites

- Python 3.8 or higher
- CMake 3.15 or higher
- A C++ compiler (GCC, Clang, or MSVC)
- Git

## Installation
We provide installation instructions for using [Conda](#using-conda) or [UV](#using-uv).

**Core Dependencies**

see [pyproject.toml](pyproject.toml) or [environment.yml](environment.yml)
- numpy - Numerical computing
- scipy - Scientific computing
- numba - JIT compilation for Python
- matplotlib - plotting
- [python bindings](pycurve) for a C++ implementation of the inner dissipation

### Using Conda

**1. Create Conda Environment**

Clone the repository and create environment
```bash
conda env create -f environment.yml
conda activate 
```

**2. Install Python project**

```bash
pip install -e .
```

**3. Build and Install C++ Python Bindings**

Navigate to the pycurve directory and build the C++ module:

Linux/macOS:
```bash
cd pycurve
mkdir -p build
cd build

# Configure CMake with conda's Python
cmake -DBUILD_PYTHON=ON \
      -DBUILD_EXAMPLES=ON \
      -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python \
      ..

# Build the module
cmake --build . --config Release
```

Windows:
```powershell
cmake -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=ON -DPython_EXECUTABLE=$env:CONDA_PREFIX\python.exe ..
cmake --build . --config Release
```
**After building, copy the compiled module to your virtual environment:**

Linux/macOS:
```bash
# Find conda site-packages
PYTHON_SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

# Copy the .so file
cp python/pycurve.*.so "$PYTHON_SITE_PACKAGES/"
```

Windows:
```powershell
$PYTHON_SITE_PACKAGES = python -c "import sysconfig; print(sysconfig.get_path('purelib'))"
Copy-Item python\pyshell.*.pyd "$PYTHON_SITE_PACKAGES\"
```

**Verify Installation**

```bash
python -c "import pyshell; import numpy; import scipy; import numba; print('✓ All imports successful')"
```


### Using UV
**1. Install uv**

`uv` is a fast Python package installer and resolver. Install it using:

Linux/macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternative (via pip):
```bash
pip install uv
```

**2. Create Virtual Environment**
Clone the repository and create environment
```bash
uv venv
```
Activate Virtual Environment

Linux/macOS:
```bash
source .venv/bin/activate
```

Windows:
```powershell
.venv\Scripts\activate
```

**3. Install Python Dependencies**

```bash
uv pip install -e ".[jupyter]"
```
Jupyter is an optional dependency. You can also install it without.
```bash
uv pip install -e .
```
**4. Build and Install C++ Python Bindings**

Navigate to the pycurve directory and build the C++ module:

```bash
cd <PYCURVE_DIR>
mkdir -p build
cd build

# Configure CMake with your virtual environment's Python
cmake -DBUILD_PYTHON=ON \
      -DBUILD_EXAMPLES=ON \
      -DPython_EXECUTABLE=$(which python) \
      ..

# Build the module
cmake --build . --config Release
```

On Windows, use:
```powershell
cmake -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=ON -DPython_EXECUTABLE=(Get-Command python).Path ..
cmake --build . --config Release
```

**After building, copy the compiled module to your virtual environment:**

Linux/macOS:
```bash
# Find your site-packages directory
PYTHON_SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

# Copy the .so file
cp python/pycurve.*.so "$PYTHON_SITE_PACKAGES/"
```

Windows:
```powershell
# Find your site-packages directory
$PYTHON_SITE_PACKAGES = python -c "import sysconfig; print(sysconfig.get_path('purelib'))"

# Copy the .pyd file
Copy-Item python\pycurve.*.pyd "$PYTHON_SITE_PACKAGES\"
```

Test that the bindings are installed correctly:

```bash
python -c "import pycurve; print('✓ All imports successful')"
```

**Usage**

Each time you work on the project, remember to activate your virtual environment:

**Conda**
```bash
conda activate 

# Deactivate when done
conda deactivate
```
**UV**

Linux/macOS:
```bash
source .venv/bin/activate
# Deactivate when done
deactivate
```

Windows:
```powershell
.venv\Scripts\activate
# Deactivate when done
deactivate
```

### Troubleshooting

**CMake Can't Find Python**

If CMake doesn't detect the correct Python interpreter, explicitly specify it:

```bash
cmake -DBUILD_PYTHON=ON \
      -DBUILD_EXAMPLES=ON \
      -DPython_EXECUTABLE=/path/to/your/venv/bin/python \
      ..
```

To find your Python path:
```bash
which python  # Linux/macOS
where python  # Windows
```

**Build Link Errors**

If you encounter linking errors during the build, you may need to use a dynamic build. Modify the `CMakeLists.txt` and change the relevant line to:

```cmake
target_link_libraries(pycurve PUBLIC curve-energy)
```

Then rebuild:
```bash
cd build
rm -rf *  # Clean the build directory
cmake -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=ON -DPython_EXECUTABLE=$(which python) ..
cmake --build . --config Release
```

**Module Not Found After Installation**

If Python can't find the `pycurve` module after installation:

1. Verify the `.so` (or `.pyd` on Windows) file is in your site-packages:
   ```bash
   python -c "import sysconfig; print(sysconfig.get_path('purelib'))"
   ls <output-from-above>/pycurve.*
   ```

**Using Anaconda Python**

If you're using Anaconda, specify the Anaconda Python executable:

```bash
cmake -DBUILD_PYTHON=ON \
      -DBUILD_EXAMPLES=ON \
      -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python \
      ..
```
**QT error in conda**
```bash
export QT_QPA_PLATFORM_PLUGIN_PATH=$CONDA_PREFIX/lib/qt6/plugins/platforms
```

## Acknowledgments

We thank Josua Sassen for the original implementation of the inner dissipation energy in pycurve, 
which served as the basis for the current version.

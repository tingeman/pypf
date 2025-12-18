# pypf - Python Permafrost Package

A Python package for permafrost analysis and borehole data processing.

## Requirements

- Python >= 3.12
- `pydatastorage` (custom package, must be installed separately)

## Installation

### Option A: Using Conda (Recommended)

#### 1. Create and activate conda environment

```bash
# Create a new conda environment with Python 3.12
conda create -n boreholes_py312 python=3.12

# Activate the environment
conda activate boreholes_py312
```

#### 2. Install dependencies via conda

```bash
# Install available dependencies from conda
conda install matplotlib numpy pandas lxml openpyxl pytables pyproj rich python-dateutil

# Install development tools (optional)
conda install ipython ipdb pytest
```

#### 3. Install pydatastorage

```bash
# If you have a local copy
pip install /path/to/pydatastorage

# Or if available via git
pip install git+https://github.com/username/pydatastorage.git
```

#### 4. Install pypf

```bash
# Navigate to the pypf directory
cd /path/to/pypf_py3

# Install in development mode
pip install -e .
```

### Option B: Using pip only

#### 1. Install pydatastorage

```bash
# If you have a local copy
pip install /path/to/pydatastorage

# Or if available via git
pip install git+https://github.com/username/pydatastorage.git
```

#### 2. Install pypf

```bash
# Clone or navigate to the pypf directory
cd /path/to/pypf_py3

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```
pip install -e ".[dev]"
```

#### Alternative: Install from requirements.txt
# Install dependencies
pip install -r requirements.txt

# Then install pypf
pip install -e .
```

## Usage

```python
import pypf
from pypf.db import boreholes

# Your code here
```

## Development

Install with development dependencies for testing and debugging:

```bash
pip install -e ".[dev]"
```

This includes:
- ipython
- ipdb
- pytest

## License

GNU General Public License (GPL)

## Author

Thomas Ingeman-Nielsen (thin@dtu.dk)

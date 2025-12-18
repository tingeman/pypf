"""
Python Permafrost (pypf) package

This package requires the custom 'pydatastorage' package to be installed separately.
"""

# Check for required custom dependencies
try:
    import pydatastorage
except ImportError:
    raise ImportError(
        "\n"
        "The 'pydatastorage' package is required but not installed.\n"
        "\n"
        "This is a custom package not available on PyPI.\n"
        "Please install it manually using one of the following methods:\n"
        "\n"
        "  1. If you have a local copy:\n"
        "     pip install /path/to/pydatastorage\n"
        "\n"
        "  2. If available via git:\n"
        "     pip install git+https://github.com/tingeman/pydatastorage.git\n"
        "\n"
        "  3. Contact the package maintainer for installation instructions.\n"
    )

__version__ = "0.2.0"

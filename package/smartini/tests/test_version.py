"""
Basic version test for the smartini package.
"""
import pytest
import smartini

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata


def test_smartini_version():
    try:
        dist_version = metadata.version("smartini")
    except metadata.PackageNotFoundError:
        pytest.skip("smartini distribution metadata not installed in this environment")

    assert smartini.__version__ == dist_version

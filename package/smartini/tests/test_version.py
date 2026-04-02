"""
Basic version test for the auto_martini package.
"""
import pytest
import smartini

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata


def test_auto_martini_version():
    try:
        dist_version = metadata.version("smartini")
    except metadata.PackageNotFoundError:
        pytest.skip("auto_martiniM3 distribution metadata not installed in this environment")

    assert smartini.__version__ == dist_version

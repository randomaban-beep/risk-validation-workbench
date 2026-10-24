"""Smoke test for repository initialization."""

import rvw


def test_package_import():
    """Verify package imports correctly and exposes expected version."""
    assert rvw.__version__ == "0.1.0"

# Package metadata and dependencies live in pyproject.toml (managed with uv).
# This shim only keeps legacy `python setup.py ...` invocations working.
from setuptools import setup

setup()

"""Test package.

Tests import helpers from ``tests.conftest`` directly, so the directory has to
be an importable package. Without this, collection passes locally (where the
working directory is on ``sys.path``) but fails in CI with
``ModuleNotFoundError: No module named 'tests'``.
"""

"""Run portable unit tests and verify all shipped GUI checkpoint paths.

Run from the repository root: python scripts/check_project.py
Tests use a writable project-local temporary folder instead of OS-specific paths.
"""
from pathlib import Path
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory(prefix='hnrs_checks_', dir=ROOT) as directory:
        previous = tempfile.tempdir
        tempfile.tempdir = directory
        try:
            suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        finally:
            tempfile.tempdir = previous
    if not result.wasSuccessful():
        return 1
    from gui_core import DEFAULT_MODELS, MODEL_NAMES, resolve_checkpoint
    for mode in DEFAULT_MODELS:
        for name in MODEL_NAMES:
            path, _ = resolve_checkpoint(ROOT, mode, name)
            print(f'Checkpoint verified: {mode} / {name}: {path.relative_to(ROOT)}')
    print('PASS: tests and all eight application checkpoints')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Builds and publishes cedanirs to PyPI.

Token resolution order:
  1. pypi_token.txt in the repo root (gitignored)
  2. TWINE_PASSWORD environment variable
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TOKEN_FILE = ROOT / "pypi_token.txt"
DIST_DIR = ROOT / "dist"


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def resolve_token() -> str:
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    token = os.environ.get("TWINE_PASSWORD", "").strip()
    if token:
        return token
    print(
        "ERROR: No PyPI token found.\n"
        "  Option 1: create pypi_token.txt in the repo root.\n"
        "  Option 2: set the TWINE_PASSWORD environment variable."
    )
    sys.exit(1)


def main():
    token = resolve_token()

    run([sys.executable, "-m", "pip", "install", "-q", "build", "twine"])

    print("\nCleaning old dist...")
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    for ei in ROOT.glob("*.egg-info"):
        shutil.rmtree(ei)
    for ei in (ROOT / "cedanirs").glob("*.egg-info"):
        shutil.rmtree(ei)

    print("\nBuilding package...")
    run([sys.executable, "-m", "build", str(ROOT)])

    print("\nUploading to PyPI...")
    dist_files = list(DIST_DIR.glob("*"))
    if not dist_files:
        print("ERROR: No dist files found after build.")
        sys.exit(1)
    run(
        [
            sys.executable, "-m", "twine", "upload",
            *[str(f) for f in dist_files],
            "-u", "__token__",
            "-p", token,
        ]
    )

    print("\nDone! Package published successfully.")


if __name__ == "__main__":
    main()

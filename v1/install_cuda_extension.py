"""Compile the bundled extension in the current Python environment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="10.0", help="B200 is 10.0")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment["TORCH_CUDA_ARCH_LIST"] = args.arch
    environment["MAX_JOBS"] = str(args.jobs)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-build-isolation",
        "--no-deps", "-v", str(root / "Point2Roof/pc_util"),
    ], check=True, env=environment)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""FedDPA-T wrapper for the shared closest-family dual-adapter trainer."""

import importlib.util
import sys
from pathlib import Path


def load_dual_trainer():
    script_dir = Path(__file__).resolve().parent
    module_path = script_dir / "52_train_dual_adapter_family.py"
    spec = importlib.util.spec_from_file_location("train_feddpa_impl", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if "--family" not in sys.argv:
        sys.argv.extend(["--family", "feddpa"])
    if "--method-name" not in sys.argv:
        sys.argv.extend(["--method-name", "FedDPA"])
    module = load_dual_trainer()
    return module.main(default_family="feddpa")


if __name__ == "__main__":
    raise SystemExit(main())

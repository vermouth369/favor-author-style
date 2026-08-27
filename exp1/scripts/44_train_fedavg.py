#!/usr/bin/env python3
"""FedAvg shared-adapter baseline.

FedAvg is the FedProx training path with proximal weight fixed to zero. This
wrapper keeps the public package small while exposing the paper-facing baseline
entry point and method name.
"""

import importlib.util
import sys
from pathlib import Path


def load_fedprox_module():
    module_path = Path(__file__).resolve().parent / "44_train_fedprox.py"
    spec = importlib.util.spec_from_file_location("train_fedprox_impl", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if "--prox-mu" not in sys.argv:
        sys.argv.extend(["--prox-mu", "0.0"])
    if "--method-name" not in sys.argv:
        sys.argv.extend(["--method-name", "FedAvg"])
    module = load_fedprox_module()
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())

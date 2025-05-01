#!/usr/bin/env python

import os
import shutil
import sys

MODEL_ROOT = "/hosted"


def model_setup(source, destination):
    """Creates a valid model folder at destination"""
    base = os.path.basename(os.path.normpath(source))
    path = os.path.join(destination, base, "1")
    os.makedirs(path, exist_ok=True)
    shutil.copy(os.path.join(source, "model.py"), path)


def main():
    if not os.path.isdir(sys.argv[1]):
        raise FileNotFoundError(f"Destination repository {sys.argv[1]} does not exist.")
    if not len(sys.argv) == 2:
        raise TypeError(f"setup_repository.py takes 1 argument for destination path.")
    move = []
    for f in os.listdir(MODEL_ROOT):
        if os.path.isdir(os.path.join(MODEL_ROOT, f)):
            if os.listdir(os.path.join(MODEL_ROOT, f)) == ["model.py"]:
                move.append(f)
    if len(move):
        print(f"Found {len(move)} models.")
        for f in move:
            print(f" moving {f}")
            model_setup(os.path.join(MODEL_ROOT, f), sys.argv[1])
        print(f"Models setup in {sys.argv[1]}.")
    else:
        print(f"No model folders found in {model_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os
import shutil
import sys


def model_setup(source, destination):
    """Creates a valid model folder at destination"""
    path = os.path.join(
        destination, 
        os.path.basename(os.path.normpath(source)),
        "1"
    )
    os.makedirs(path, exist_ok=True)
    shutil.copy(os.path.join(source, "model.py"), path)


def main():
    if not os.path.isdir(sys.argv[1]):
        raise FileNotFoundError(f"Destination repository {sys.argv[1]} does not exist.")
    if not len(sys.argv) == 2:
        raise TypeError(f"setup_repository.py takes 1 argument for destination path.")
    move = []
    model_root = os.path.join(os.path.dirname(__file__), "models")
    for f in os.listdir(model_root):
        if os.path.isdir(os.path.join(model_root, f)):
            if os.listdir(os.path.join(model_root, f)) == ["model.py"]:
                move.append(f)
    if len(move):
        print(f"Found {len(move)} models.")
        for f in move:
            print(f" moving {f}")
            model_setup(os.path.join(model_root, f), sys.argv[1])
        print(f"Models setup in {sys.argv[1]}.")
    else:
        print(f"No model folders found in {model_root}")

if __name__ == "__main__":
    main()

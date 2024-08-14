#!/usr/bin/env python

import pooch
from pathlib import Path
import os
import stat

test_data_dir="test_data"
host_model_repository=os.path.join(test_data_dir, "model_repository")
wsi_files=os.path.join(test_data_dir, "wsi")

def main():
    Path(host_model_repository).mkdir(parents=True, exist_ok=True)

    pooch.retrieve(
        fname="EfficientNetV2S.tensorflow.zip",
        url="https://drive.usercontent.google.com/download?id=1Mmm2sRGzdzCEAODjABiiPIiBdg40EPwC&export=download&confirm=t",
        known_hash="a6ed53d8343498b4ebfe7ff1a9ccbcabef23d6a164d2a521916774af49996f7e",
        path=host_model_repository
    )
    print(f"Downloaded EfficientNet to {host_model_repository}")
    
    Path(wsi_files).mkdir(parents=True, exist_ok=True)

    wsi_fname="TCGA-AN-A0G0-01Z-00-DX1.svs"
    wsi_path = pooch.retrieve(
        fname=wsi_fname,
        url="https://drive.usercontent.google.com/download?id=19agE_0cWY582szhOVxp9h3kozRfB4CvV&export=download&confirm=t",
        known_hash="d046f952759ff6987374786768fc588740eef1e54e4e295a684f3bd356c8528f",
        path=wsi_files,
        )
    print(f"Downloaded {wsi_fname} to {wsi_files}")

    mask_fname="TCGA-AN-A0G0-01Z-00-DX1.mask.png"
    mask_path = pooch.retrieve(
        fname=mask_fname,
        url="https://drive.usercontent.google.com/download?id=17GOOHbL8Bo3933rdIui82akr7stbRfta&export=download&confirm=t",
        known_hash="bb657ead9fd3b8284db6ecc1ca8a1efa57a0e9fd73d2ea63ce6053fbd3d65171",
        path=wsi_files,
        )
    print(f"Downloaded {mask_fname} to {wsi_files}")

    for filename in [wsi_path, mask_path]:
        current_permissions = os.stat(filename).st_mode
        new_permissions = current_permissions | stat.S_IRGRP | stat.S_IROTH
        os.chmod(filename, new_permissions)
        print(f"Added read privileges to {filename}")


if __name__ == "__main__":
    main()

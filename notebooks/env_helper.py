"""helps loading env variables
this is A ugly hack though LOL
I used in the notebooks to init the env variables.
    (don't know how to start the integrated pycharm jupyter server with the env variables set...)
"""
from pathlib import Path
import os

from stemseg.data import parse_generic_video_dataset
from stemseg.data.paths import KITTISTEPPaths


def load_env(file=Path(__file__).parent / ".." / "activate.sh"):
    if not file.is_file():
        print(f"warning: failed to find activate file in {file}.")
        return
    s = file.read_text()
    lines = s.split("\n")
    for line in lines:
        # search and for lines where the env is set: "export VAR=value"
        if line.startswith("export "):
            k, v = line[len("export "):].split("=")
            print(f"setting env {k} to {v}")
            os.environ[k] = v


def get_categories():
    load_env()
    _, meta_info = parse_generic_video_dataset(KITTISTEPPaths.train_images_dir(),
                                                KITTISTEPPaths.val_vds_file())

    return meta_info["category_labels"]
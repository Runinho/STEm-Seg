# read all image sequences of Kitti step and generates a hdf5 file for faster access during training time
from typing import List
import h5py
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json
from pycocotools import mask as masktools

from stemseg.data.paths import KITTISTEPPaths


def create_images_hdtf5():
    base_dir = Path(KITTISTEPPaths.train_images_dir())
    seqs: List[Path] = list(base_dir.glob("*"))
    seqs = [f for f in seqs if f.is_dir()]
    print("creating images.h5")
    hdf5_file = base_dir / "images.h5"
    bytes = 0
    with h5py.File(hdf5_file, "w") as f:
        for seq in seqs:
            name = seq.name
            print(name)
            files = list(seq.glob("*.jpg")) + list(seq.glob("*.png"))
            files.sort()
            # load images
            images = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in files]
            img_a = np.array(images)
            # save in dataset
            dset = f.create_dataset(name, img_a.shape, chunks=(1, *img_a.shape[1:]), dtype=np.uint8)
            dset[:] = img_a
            bytes += img_a.nbytes
            print(f"current bytes: {bytes} ({bytes / 1e9} GB)")
    print(f"saved images to {hdf5_file}")

def calculate_shape(seq, width, height):
    i = 0
    for seg in seq["segmentations"]:
        i += len(seg.items())
    return (i, width, height)

def create_masks_hdtf5(base_dir: Path, dataset_json: Path):
    print("converting masks")
    dataset_json = Path(dataset_json)
    dataset_json = dataset_json.with_name(dataset_json.stem.replace("_without_mask", "") + ".json")
    print(f"loading from label information from json: {dataset_json}")

    with open(dataset_json, 'rt') as file:
        dataset = json.load(file)
    h5py_filename = f"{dataset_json.stem}_masks.h5"
    h5py_file = base_dir / h5py_filename

    # save the maskfilename in the json
    dataset["meta"]["mask_filename"] = h5py_filename

    if h5py_file.exists():
        print(f"mask file already exists {h5py_file}")


    with h5py.File(h5py_file, "w") as f:
        for seq in dataset["sequences"]:
            # load image dimensions
            width, height = cv2.imread(str(base_dir / seq["image_paths"][0])).shape[:2]
            i = 0
            seq_name = seq["id"]
            dset_shape = calculate_shape(seq, width, height)
            dset = f.create_dataset(seq_name, dset_shape, dtype=bool, chunks=(1, *dset_shape[1:]),
                                    compression="lzf")
            for seg_index, seg in tqdm(enumerate(seq["segmentations"]),
                                       total=len(seq["segmentations"])):
                for k, v in seg.items():
                    seq["segmentations"][seg_index][k] = i
                    decoded_mask = np.ascontiguousarray(
                        masktools.decode({"size": (width, height), "counts": v}).astype(np.uint8))
                    dset[i] = decoded_mask
                    i += 1
    print(f"wrote masks into {h5py_file}")
    new_dataset_json = dataset_json.with_name(dataset_json.stem + "_without_mask.json")
    with open(new_dataset_json, 'w') as file:
        json.dump(dataset, file)
    print(f"wrote new label file to {new_dataset_json}")


if __name__ == "__main__":
    create_images_hdtf5()
    create_masks_hdtf5(Path(KITTISTEPPaths.train_images_dir()), KITTISTEPPaths.train_vds_file())
    create_masks_hdtf5(Path(KITTISTEPPaths.train_images_dir()), KITTISTEPPaths.val_vds_file())
    print("done.")


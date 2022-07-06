"""load and manage video datasets"""

import json
import os
from typing import List, Tuple, Dict, Iterable

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from pycocotools import mask as masktools


def parse_generic_video_dataset(base_dir, dataset_json) -> Tuple[
    List['GenericVideoSequence'], Dict]:
    """create dataset from dataset json

    Args:
        base_dir (Path): location of the data
        dataset_json (Path): location of the json file containing the labels and dataset information

    Returns:
        Tuple[List[GenericVideoSequence], Dict]:
            - seqs (list): list of GenericVideoSequence
            - meta_info (dict): information about the dataset

    Examples:
        Load KITTIMOTS dataset

        >>> parse_generic_video_dataset(KITTIMOTSPaths.train_images_dir(),
        >>>                             KITTIMOTSPaths.val_vds_file())
        ([GenericVideoSequence(id:0002, len:233),
          GenericVideoSequence(id:0006, len:270),
          GenericVideoSequence(id:0007, len:800),
          GenericVideoSequence(id:0008, len:390),
          GenericVideoSequence(id:0010, len:294),
          GenericVideoSequence(id:0013, len:340),
          GenericVideoSequence(id:0014, len:106),
          GenericVideoSequence(id:0016, len:209),
          GenericVideoSequence(id:0018, len:339)],
         {'name': 'kitti_mots_val',
          'category_labels': {1: 'car', 2: 'person', 3: 'ignore'},
          'version': '0.1'})

    Links:
        dataset jsons can be found here:
        https://omnomnom.vision.rwth-aachen.de/data/STEm-Seg/dataset_jsons/
    """
    with open(dataset_json, 'rt') as file:
        dataset = json.load(file)

    meta_info = dataset["meta"]

    # convert instance and category IDs from str to int
    meta_info["category_labels"] = {int(k): v for k, v in meta_info["category_labels"].items()}

    if "segmentations" in dataset["sequences"][0]:
        for seq in dataset["sequences"]:
            seq["categories"] = {int(iid): cat_id for iid, cat_id in seq["categories"].items()}
            seq["segmentations"] = [
                {
                    int(iid): seg
                    for iid, seg in seg_t.items()
                }
                for seg_t in seq["segmentations"]
            ]

            # sanity check: instance IDs in "segmentations" must match those in "categories"
            seg_iids = set(sum([list(seg_t.keys()) for seg_t in seq["segmentations"]], []))
            assert seg_iids == set(seq["categories"].keys()), \
                   f"Instance ID mismatch: {seg_iids} vs. {set(seq['categories'].keys())}"

    seqs = [GenericVideoSequence(base_dir, **seq) for seq in dataset["sequences"]]

    return seqs, meta_info


class GenericVideoSequence:
    """continuous image sequence
    """

    # TODO instead of dict we could use kwargs. which would make the interface much nicer (Better for the IDE) :D
    def __init__(self, base_dir, image_paths, height, width, id, segmentations=None,
                 categories=None, **kwargs):
        """

        Args:
            base_dir (Path): directory of the datset
            image_paths (List): list of image paths
            height (int): image height
            width (int): image width
            id: identifier of this sequence
            segmentations (List):
            categories (List):
            **kwargs:
        """
        self.base_dir = base_dir
        self.image_paths = image_paths
        self.image_dims = (height, width)
        self.id = id

        self.segmentations = segmentations
        self.instance_categories = categories

    @property
    def instance_ids(self) -> List:
        return list(self.instance_categories.keys())

    @property
    def category_labels(self) -> Iterable:
        """get """
        return [self.instance_categories[instance_id] for instance_id in self.instance_ids]

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return f'GenericVideoSequence(id:{self.id}, len:{len(self)})'

    def load_images(self, frame_idxes=None) -> List[np.ndarray]:
        """load images

        Args:
            frame_idxes (List[int]): image indices to load

        Returns:
            list of images as numpy arrays
        """
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        images = []
        for t in frame_idxes:
            # TODO: why do we use cv2 to read images?
            im = cv2.imread(os.path.join(self.base_dir, self.image_paths[t]), cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError(
                    f"No image found at path: {os.path.join(self.base_dir, self.image_paths[t])}")
            images.append(im)

        return images

    def load_masks(self, frame_idxes=None) -> List[List[npt.NDArray[np.uint8]]]:
        # None returns all
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        masks = []
        for t in frame_idxes:
            masks_t = []

            for instance_id in self.instance_ids:
                if instance_id in self.segmentations[t]:
                    rle_mask = {
                        "counts": self.segmentations[t][instance_id].encode('utf-8'),
                        "size": self.image_dims
                    }
                    masks_t.append(
                        np.ascontiguousarray(masktools.decode(rle_mask).astype(np.uint8)))
                else:
                    masks_t.append(np.zeros(self.image_dims, np.uint8))

            masks.append(masks_t)

        return masks

    def filter_categories(self, cat_ids_to_keep: Iterable):
        instance_ids_to_keep = sorted(
            [iid for iid, cat_id in self.instance_categories.items() if iid in cat_ids_to_keep])
        for t in range(len(self)):
            self.segmentations[t] = {iid: seg for iid, seg in self.segmentations[t].items() if
                                     iid in instance_ids_to_keep}

    def filter_zero_instance_frames(self):
        t_to_keep = [t for t in range(len(self)) if len(self.segmentations[t]) > 0]
        self.image_paths = [self.image_paths[t] for t in t_to_keep]
        self.segmentations = [self.segmentations[t] for t in t_to_keep]

    def apply_category_id_mapping(self, mapping: Dict):
        assert set(mapping.keys()) == set(self.instance_categories.keys())
        self.instance_categories = {
            iid: mapping[current_cat_id] for iid, current_cat_id in self.instance_categories.items()
        }

    def extract_subsequence(self, frame_idxes, new_id="") -> 'GenericVideoSequence':
        # check if frame_idxes are in range
        assert all([t in range(len(self)) for t in frame_idxes])

        # filter the data
        instance_ids_to_keep = set(
            sum([list(self.segmentations[t].keys()) for t in frame_idxes], []))

        subseq_dict = {
            "id": new_id if new_id else self.id,
            "height": self.image_dims[0],
            "width": self.image_dims[1],
            "image_paths": [self.image_paths[t] for t in frame_idxes],
            "categories": {iid: self.instance_categories[iid] for iid in instance_ids_to_keep},
            "segmentations": [
                {
                    iid: segmentations_t[iid]
                    for iid in segmentations_t if iid in instance_ids_to_keep
                }
                for t, segmentations_t in enumerate(self.segmentations) if t in frame_idxes
            ]
        }

        return self.__class__(subseq_dict, self.base_dir)

    def dbg_load_image(self, frame_index)-> Image:
        """load image as PIL Image

        Args:
            frame_index (int): frame index of the image to load

        Returns:
            Image:
                image
        """
        img = self.load_images([frame_index])[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return Image.fromarray(img)


def visualize_generic_dataset(base_dir, dataset_json):
    """show some images from the dataset with open cv2.imshow

    press `q` to stop the program from displaying next image.

    See Also:
        :func:`parse_generic_video_dataset` for how the data is parsed

    Args:
        base_dir (Path): location of the data
        dataset_json (Path): location of the json file containing the labels and dataset information
    """
    from stemseg.utils.vis import overlay_mask_on_image, create_color_map
    import os

    seqs, meta_info = parse_generic_video_dataset(base_dir, dataset_json)
    category_names = meta_info["category_labels"]

    cmap = create_color_map().tolist()

    print("Press `q` to end program")
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    for seq in seqs:
        # which images to load
        if len(seq) > 100:
            frame_idxes = list(range(100, 150))
        else:
            frame_idxes = None

        # load images and mask
        images = seq.load_images(frame_idxes)
        masks = seq.load_masks(frame_idxes)
        category_labels = seq.category_labels

        print("[COLOR NAME] -> [CATEGORY NAME]")
        color_key_printed = False

        for image_t, masks_t in zip(images, masks):
            for i, (mask, cat_label) in enumerate(zip(masks_t, category_labels), 1):
                image_t = overlay_mask_on_image(image_t, mask, mask_color=cmap[i])

                if not color_key_printed:
                    # print("{} -> {}".format(rgb_to_name(cmap[i][::-1]), category_names[cat_label]))
                    print("{} -> {}".format(cmap[i], category_names[cat_label]))

            color_key_printed = True

            cv2.imshow('Image', image_t)
            # wait for key `113` which is a `q`
            if cv2.waitKey(0) == 113:
                exit(0)

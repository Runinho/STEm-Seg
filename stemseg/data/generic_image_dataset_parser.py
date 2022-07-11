"""load and manage image datasets"""
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from pycocotools import mask as masktools

import cv2
import json
import os
import numpy as np


def parse_generic_image_dataset(base_dir, dataset_json) -> Tuple[List['GenericImageSample'], Dict]:
    """create image dataset from labels and a image directory

    Args:
        base_dir (Path): location of the image data
        dataset_json (Path): location of the json file containing the labels and dataset information

    Returns:
        Tuple[List[GenericImageSample], Dict]:
            - samples (list): list of GenericImageSample
            - meta_info (dict): information about the dataset

    Examples:
        Load Mapillary dataset
        >>> parse_generic_image_dataset(MapillaryPaths.images_dir(), MapillaryPaths.ids_file())

    Links:
        dataset jsons can be found here:
        https://omnomnom.vision.rwth-aachen.de/data/STEm-Seg/dataset_jsons/
    """
    # TODO (Runinho): document json label format.
    with open(dataset_json, 'r') as fh:
        dataset = json.load(fh)

    meta_info = dataset["meta"]

    # convert instance and category IDs from str to int
    meta_info["category_labels"] = {int(k): v for k, v in meta_info["category_labels"].items()}
    samples = [GenericImageSample(base_dir, **sample) for sample in dataset["images"]]

    return samples, meta_info


class GenericImageSample(object):
    """one image of the dataset"""

    def __init__(self, base_dir, height, width, image_path, categories, segmentations, ignore=None, **kwargs):
        """
        Args:
            base_dir (Path): location to load the images from
            height (int): image height
            width (int): image width
            image_path (Path): location of the image relative to ``base_dir``
            categories (List[int]): list of category labels for the masks in ``segmentations``.
                                    Same length as ``segmentations``
            segmentations (List[str]): list of RLE encoded masks. Same length as ``categories``.
            ignore (str, optional): Optional RLE encoded ignore mask. Defaults to None.
            **kwargs: not used.

        Links:
            RLE used for masks: https://github.com/ppwwyyxx/cocoapi

        See Also:
            :class:`data.generic_video_dataset_parser.GenericVideoSequence`

        """
        assert len(categories) == len(segmentations), \
            f"length of segmentation and categories is expected to be the same. " \
            f"Length of segmentations was {len(segmentations)}, " \
            f"length of categories {len(categories)}"
        assert isinstance(base_dir, Path), "expect base_dir to be pathlib path"

        self.height = height
        self.width = width
        self.path = base_dir / image_path
        self.categories = [int(cat_id) for cat_id in categories]
        self.segmentations = segmentations
        self.ignore = ignore

    def mask_areas(self):
        """number of labeled pixels per mask

        Returns:
            List[int]:
                list containing the number of pixels per mask.
        """
        rle_objs = [{
            "size": (self.height, self.width),
            "counts": seg.encode("utf-8")
        } for seg in self.segmentations ]

        return [masktools.area(obj) for obj in rle_objs]

    def load_image(self):
        """load image as numpy array

        Returns:
            np.ndarray:
                image with shape `height`, `width`
        """
        im = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("No image found at path: {}".format(self.path))
        assert im.shape == (3, self.height, self.width), \
            f"invalid image size loaded for {self.path}." \
            f"expected image size {(3, self.height, self.width)} but got {im.shape}"
        return im

    def load_ignore_mask(self) -> Optional[np.ndarray]:
        """load ignore mask from RLE encoding

        Returns:
            ignore mask
        """
        if self.ignore is None:
            return None

        return np.ascontiguousarray(masktools.decode({
            "size": (self.height, self.width),
            "counts": self.ignore.encode('utf-8')
        }).astype(np.uint8))

    def load_masks(self):
        """load mask for all categories

        Returns:
            List[np.ndarray]:
                list of masks for all categories.

        """
        return [np.ascontiguousarray(masktools.decode({
            "size": (self.height, self.width),
            "counts": seg.encode('utf-8')
        }).astype(np.uint8)) for seg in self.segmentations]

    def filter_categories(self, cat_ids_to_keep):
        """ remove instances from the dataset

        removes all instances from the dataset that are not in `cat_ids_to_keep`.

        Args:
            cat_ids_to_keep (Iterable): category ids to keep.
        """
        self.categories, self.segmentations = zip(*[
            (cat_id, seg) for cat_id, seg in zip(self.categories, self.segmentations) if cat_id in cat_ids_to_keep
        ])

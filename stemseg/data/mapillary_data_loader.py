"""Load images and labels from the Mapillary Vistas Dataset

Note:
    resized images are expected. Check Readme for more information.

Links: https://www.mapillary.com/dataset/vistas
"""
from functools import partial
from typing import Tuple, List

from torch.utils.data import Dataset

from stemseg.config import cfg
from stemseg.data.generic_image_dataset_parser import parse_generic_image_dataset, GenericImageSample
from stemseg.data.image_to_seq_augmenter import ImageToSeqAugmenter
from stemseg.structures import BinaryMask, BinaryMaskSequenceList
from stemseg.data.common import compute_resize_params, scale_and_normalize_images
from stemseg.utils import RepoPaths, transforms

import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import yaml


class MapillaryDataLoader(Dataset):
    def __init__(self, base_dir, ids_json_file, min_instance_size=30, max_nbr_instances=30,
                 metainfo_file="mapillary.yaml", new_id="id_kittimots"):
        """

        Args:
            base_dir (Path): location of the images
            ids_json_file: json file with datasetinformation and containing the labels
            min_instance_size (int, optional): Minimal number of pixels for a mask.
                                              All masks with a smaller area will be ignored.
                                              Defaults to 30.
            max_nbr_instances (int, optional): Defaults to 30.
        """
        super().__init__()

        samples, meta_info = parse_generic_image_dataset(base_dir, ids_json_file)

        with open(os.path.join(RepoPaths.dataset_meta_info_dir(), metainfo_file), 'r') as fh:
            #TODO (Runinho): we do not check yaml version here
            #                whereas we do in config.load_from_file
            category_details = yaml.load(fh, Loader=yaml.SafeLoader)
            category_details = {cat['id']: cat for cat in category_details}

        self.cat_ids_to_keep = [cat_id for cat_id, attribs in category_details.items() if attribs['keep']]
        self.cat_ids_to_ignore = [cat_id for cat_id, attribs in category_details.items() if attribs['ignore_mask']]

        self.category_id_mapping = {
            cat_id: category_details[cat_id][new_id] for cat_id in self.cat_ids_to_keep
        }
        self.category_labels = {
            cat_id: category_details[cat_id]['label'] for cat_id in self.cat_ids_to_keep
        }

        # remove category labels that have less than ``min_instance_size`` pixels.
        def filter_by_mask_area(sample: GenericImageSample) -> GenericImageSample:
            """remove all masks that have less than ``min_instance_size`` pixels labeled"""
            mask_areas = sample.mask_areas()
            instance_idxes_to_keep = [
                i for i in range(len(sample.segmentations)) if mask_areas[i] >= min_instance_size
            ]

            sample.segmentations = [sample.segmentations[i] for i in instance_idxes_to_keep]
            sample.categories = [sample.categories[i] for i in instance_idxes_to_keep]

            return sample

        samples = map(filter_by_mask_area, samples)

        # remove instances that do not contain relevant labels.
        self.samples = []
        for s in samples:
            if sum([1 for cat in s.categories if cat in self.cat_ids_to_keep]) == 0:
                continue  # no relevant instances present in image

            instance_idxes_to_keep = [
                i for i in range(len(s.segmentations)) if s.categories[i] in self.cat_ids_to_keep + self.cat_ids_to_ignore
            ]

            s.segmentations = [s.segmentations[i] for i in instance_idxes_to_keep]
            s.categories = [s.categories[i] for i in instance_idxes_to_keep]

            self.samples.append(s)

        self.max_nbr_instances = max_nbr_instances

        self.augmenter = ImageToSeqAugmenter(perspective=True, affine=True, motion_blur=True,
                                             rotation_range=(-10, 10), perspective_magnitude=0.08,
                                             hue_saturation_range=(-5, 5), brightness_range=(-40, 40),
                                             motion_blur_prob=0.0, translate_range=(-0.1, 0.1))
        self.num_frames = cfg.INPUT.NUM_FRAMES

        self.np_to_tensor = transforms.BatchImageTransform(transforms.ToTorchTensor(format='CHW'))

    def filter_instance_masks(self, instance_masks, category_labels, instance_areas):
        """separates ignore mask and masks we want to use

        We limit the number of instances to ``self.max_nbr_instances`` and ignore
        categories in ``self.cat_ids_to_ignore``.

        Args:
            instance_masks (List[np.ndarray]): list of label masks
            category_labels (List[int]): list of category ids for ``instance_masks``.
                Same length as `instance_masks`
            instance_areas (List[int]): list of the area of ``instance_masks``

        Returns:
            Tuple[List[np.ndarray], List[int], np.ndarray]:
                tuple containing the filtered ``instance_mask``, the category labels for the
                returned instance masks and the ignore mask.

        """
        # reorder instances in descending order of their mask areas
        reorder_idxes, instance_areas = zip(*sorted(
            [(i, area) for i, area in enumerate(instance_areas)], key=lambda x: x[1], reverse=True))

        instance_masks = [instance_masks[idx] for idx in reorder_idxes]
        category_labels = [category_labels[idx] for idx in reorder_idxes]

        filtered_instance_masks = []
        filtered_category_labels = []
        ignore_instance_masks = []

        # create ignore mask
        # we igonre all masks with categories in self.cat_ids_to_ignore
        # and all instances that are not in the set of the largest ``self.max_nbr_instances``
        for i, (mask, label) in enumerate(zip(instance_masks, category_labels)):
            # TODO (Runinho): why do we use i to limit the number of instances and not
            #  len(filtered_instance_mask)? We can also just break the loop if we detect we
            #  reached self.max_nbr_instances.
            if i < self.max_nbr_instances:
                if label in self.cat_ids_to_ignore:
                    ignore_instance_masks.append(mask)
                else:
                    filtered_instance_masks.append(mask)
                    filtered_category_labels.append(label)
            else:
                ignore_instance_masks.append(mask)

        if ignore_instance_masks:
            # TODO (Runinho): instead of stacking a list we could just do the or in the loop
            #  starting with zeros.
            ignore_mask = np.any(np.stack(ignore_instance_masks), 0).astype(np.uint8)
        else:
            ignore_mask = np.zeros_like(instance_masks[0])

        return filtered_instance_masks, filtered_category_labels, ignore_mask

    def __len__(self):
        """number of label/image paris in the dataset
        """
        return len(self.samples)

    def __getitem__(self, index):
        """loads image and label and create synthetic image sequence

        Uses :class:`data.ImageToSeqAugmenter` to create the image sequence.

        randomly flips the image.


        """
        sample = self.samples[index]

        image = sample.load_image()
        image_height, image_width = sample.height, sample.width

        instance_masks = sample.load_masks()

        # separate instance masks of categories which have to be evaluated from the ones which have to be ignored.
        instance_masks, category_ids, ignore_mask = self.filter_instance_masks(
            instance_masks, sample.categories, sample.mask_areas())

        # apply random horizontal flip
        image, instance_masks, ignore_mask = self.apply_random_flip(image, instance_masks, ignore_mask)

        # convert masks to BinaryMask type
        instance_masks = [BinaryMask(mask) for mask in instance_masks]
        ignore_mask = BinaryMask(ignore_mask)

        # Convert everything to a single element list so that it becomes a one-image 'sequence'
        seq_images, seq_instance_masks, seq_ignore_masks, seq_invalid_pts_masks = \
            [image], [instance_masks], [ignore_mask], [np.zeros((image_height, image_width), np.uint8)]

        # add remaining sequence images by augmenting the original image
        for t in range(self.num_frames - 1):
            # get transformed image, instance mask and point validity mask
            masks = instance_masks + [ignore_mask]
            im_trafo, masks_trafo, invalid_pts = self.augmenter(image, masks)

            instance_masks_trafo, ignore_mask_trafo = masks_trafo[:-1], masks_trafo[-1]

            # add everything to the sequence lists
            seq_images.append(im_trafo)
            seq_instance_masks.append(instance_masks_trafo)
            seq_ignore_masks.append(ignore_mask_trafo)
            seq_invalid_pts_masks.append(invalid_pts)

        # shuffle the elements of the sequence
        seq_images, seq_instance_masks, seq_ignore_masks, seq_invalid_pts_masks = self.apply_random_sequence_shuffle(
            seq_images, seq_instance_masks, seq_ignore_masks, seq_invalid_pts_masks)

        # normalize/scale/offset the image colors as needed (automatically converts from uint8 to float32)
        seq_images = torch.stack(self.np_to_tensor(*seq_images), 0).float()

        # scale and normalize images
        seq_images = scale_and_normalize_images(seq_images, cfg.INPUT.IMAGE_MEAN, cfg.INPUT.IMAGE_STD,
                                                not cfg.INPUT.BGR_INPUT, cfg.INPUT.NORMALIZE_TO_UNIT_SCALE)

        seq_invalid_pts_masks = [torch.from_numpy(mask).float() for mask in seq_invalid_pts_masks]

        for i in range(len(seq_images)):
            invalid_pts = 1. - seq_invalid_pts_masks[i][None, :, :]
            seq_images[i] = seq_images[i] * invalid_pts

        # resize images to the required input size
        new_width, new_height, scale_factor = compute_resize_params(image, cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM)
        seq_images = F.interpolate(seq_images, (new_height, new_width), mode='bilinear', align_corners=False)

        # resize masks to the required input size
        seq_instance_masks = BinaryMaskSequenceList(seq_instance_masks)
        seq_instance_masks = seq_instance_masks.resize((new_width, new_height))
        seq_ignore_masks = [mask.resize((new_height, new_width)) for mask in seq_ignore_masks]

        # convert masks to torch tensors
        seq_instance_masks = seq_instance_masks.tensor().permute(1, 0, 2, 3)  # [N, T, H, W]
        seq_ignore_masks = torch.stack([mask.tensor() for mask in seq_ignore_masks], 0)  # [T, H, W]

        category_labels = [self.category_labels[label] for label in category_ids]
        category_ids = [self.category_id_mapping[label] for label in category_ids]
        category_ids = torch.tensor(category_ids, dtype=torch.long)

        # combine everything into a dictionary
        targets = {"masks": seq_instance_masks,
                   "category_ids": category_ids,
                   'ignore_masks': seq_ignore_masks}

        return seq_images, targets, (image_width, image_height), {"category_labels": category_labels}

    def apply_random_flip(self, image, instance_masks, ignore_mask):
        """with a chance of 50% flip the image

        Args:
            image (np.ndarray): image
            instance_masks (List[np.ndarray]): masks
            ignore_mask (np.ndarray):

        Returns:
            Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
                with a 50% chance atuple of the flipped inputs otherwise a tuple with
                the unaltered inputs
        """
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            instance_masks = [np.flip(instance_mask, axis=1) for instance_mask in instance_masks]
            ignore_mask = np.flip(ignore_mask, axis=1)

        return image, instance_masks, ignore_mask

    def apply_random_sequence_shuffle(self, images, instance_masks, ignore_masks, invalid_pts_masks):
        """shuffle image and masks

        Args:
            images (List): list of images. Expect a length of ``self.num_frames``.
            instance_masks (List): list of masks. Expect a length of ``self.num_frames``.
            ignore_masks (List): list of ignore masks. Expect a length of ``self.num_frames``.
            invalid_pts_masks (List): list of invalid points masks. Expect a length of ``self.num_frames``.

        Returns:
            Tuple[List, List, List, List]:
                shuffled input lists.

        """
        perm = list(range(self.num_frames))
        random.shuffle(perm)
        images = [images[i] for i in perm]
        instance_masks = [instance_masks[i] for i in perm]
        ignore_masks = [ignore_masks[i] for i in perm]
        invalid_pts_masks = [invalid_pts_masks[i] for i in perm]

        return images, instance_masks, ignore_masks, invalid_pts_masks

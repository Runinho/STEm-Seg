from collections import defaultdict
from stemseg.structures import ImageList
from stemseg.utils.vis import overlay_mask_on_image, create_color_map
from torch.nn import functional as F
from typing import List

import cv2
import math
import numpy as np
import torch


def scale_and_normalize_images(images, means, scales, invert_channels, normalize_to_unit_scale):
    """ Scales and normalizes images

    Args:
        images (tensor(T, C, H, W)):
        means (List[float]):
        scales (List[float]):
        invert_channels (bool):

    Returns:
        tensor(T, C, H, W):
    """
    means = torch.tensor(means, dtype=torch.float32)[None, :, None, None]  # [1, 3, 1, 1]
    scales = torch.tensor(scales, dtype=torch.float32)[None, :, None, None]  # [1. 3. 1. 1]
    if normalize_to_unit_scale:
        images = images / 255.

    images = (images - means) / scales
    if invert_channels:
        return images.flip(dims=[1])
    else:
        return images


def compute_padding(width, height):
    pad_right = (int(math.ceil(width / 32)) * 32) - width
    pad_bottom = (int(math.ceil(height / 32)) * 32) - height
    return pad_right, pad_bottom


def resize_and_pad_images(images, min_dim, max_dim):
    """
    Resizes and pads images for input to network
    :param images: tensor(T, C, H, W)
    :param min_dim: int
    :param max_dim: int
    :return: tensor(T, C, H, W)
    """
    height, width = images.shape[-2:]
    resize_width, resize_height, _ = compute_resize_params_2((width, height), min_dim, max_dim)

    # make width and height a multiple of 32
    pad_right = (int(math.ceil(resize_width / 32)) * 32) - resize_width
    pad_bottom = (int(math.ceil(resize_height / 32)) * 32) - resize_height

    images = F.interpolate(images, (resize_width, resize_height), mode="bilinear", align_corners=False)
    return F.pad(images, (0, pad_right, 0, pad_bottom))


def pad_masks_to_image(image_seqs, targets: List['MaskTarget']):
    padded_h, padded_w = image_seqs.max_size

    for targets_per_seq in targets:
        instance_masks = targets_per_seq.masks  # [N, T, H, W]
        ignore_masks = targets_per_seq.ignore_masks  # [T, H, W]

        mask_h, mask_w = instance_masks.shape[-2:]
        pad_bottom, pad_right = padded_h - mask_h, padded_w - mask_w

        instance_masks = F.pad(instance_masks, (0, pad_right, 0, pad_bottom))
        ignore_masks = F.pad(ignore_masks.unsqueeze(0), (0, pad_right, 0, pad_bottom)).squeeze(0)

        targets_per_seq.masks = instance_masks
        targets_per_seq.ignore_masks = ignore_masks

    return targets

# contains all the information to calculate the loss
class MaskTarget:
    def __init__(self, masks, category_ids, ignore_masks, foreground_categories):
        """Contains the label information for one sequence and is used to calcualte the losses.

        Args:
            masks (Torch.Tensor): masks for each id and timestep. [N, T, H, W]
            category_ids (Torch.Tensor): category_ids (class labels) for each mask [N]
            ignore_masks (Torch.Tensor): ignore mask of pixel that should
                                         be not included in the loss calcualtion[T, H, W]
            foreground_categories (list[int[): category_ids that are consider to be foreground
        """
        self.masks = masks
        self.category_ids = category_ids
        self.ignore_masks = ignore_masks
        self.foreground_categories = torch.tensor(foreground_categories)

        # lazy computed
        self.semseg_mask = None


    def pin_memory(self):
        # return self with the pinned memory
        self.masks = self.masks.pin_memory()
        self.category_ids = self.category_ids.pin_memory()
        self.ignore_masks = self.ignore_masks.pin_memory()
        self.foreground_categories = self.foreground_categories.pin_memory()
        return self

    def to(self, *args, **kwargs):
        # like torch.Tensor.to but works inplace. Fore compatibility reasons we return just self
        self.masks = self.masks.to(*args, **kwargs)
        self.category_ids = self.category_ids.to(*args, **kwargs)
        self.ignore_masks = self.ignore_masks.to(*args, **kwargs)
        self.foreground_categories = self.foreground_categories.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def get_semseg_masks(self):
        # lazy compute the semseg mask an array containing the label at the pixel position.
        if self.semseg_mask is None:
            self.semseg_mask = instance_masks_to_semseg_mask(self.masks, self.category_ids)
        return self.semseg_mask

    def foreground_masks(self):
        is_foreground = list([True in self.foreground_categories for  c in self.category_ids])
        return self.masks[is_foreground]
    @torch.no_grad()
    def get_foreground_mask(self):
        return torch.isin(self.get_semseg_masks(), self.foreground_categories).float()

class Batch:
    def __init__(self, image_seqs, targets: List[MaskTarget], meta_info):
        self.image_seqs = image_seqs
        self.targets = targets
        self.meta_info = meta_info

    def pin_memory(self):
        self.image_seqs = self.image_seqs.pin_memory()
        self.targets = list([target.pin_memory() for target in self.targets])
        return self

def collate_fn(samples):
    image_seqs, targets, original_dims, meta_info = zip(*samples)
    image_seqs = ImageList.from_image_sequence_list(image_seqs, original_dims)
    targets = pad_masks_to_image(image_seqs, targets)
    return Batch(image_seqs, targets, meta_info)


def compute_resize_params_2(image_dims, min_resize_dim, max_resize_dim):
    """
    :param image_dims: as tuple of (width, height)
    :param min_resize_dim:
    :param max_resize_dim:
    :return:
    """
    lower_size = float(min(image_dims))
    higher_size = float(max(image_dims))

    scale_factor = min_resize_dim / lower_size
    if (higher_size * scale_factor) > max_resize_dim:
        scale_factor = max_resize_dim / higher_size

    width, height = image_dims
    new_height, new_width = round(scale_factor * height), round(scale_factor * width)

    return new_width, new_height, scale_factor


def compute_resize_params(image, min_dim, max_dim):
    lower_size = float(min(image.shape[:2]))
    higher_size = float(max(image.shape[:2]))

    scale_factor = min_dim / lower_size
    if (higher_size * scale_factor) > max_dim:
        scale_factor = max_dim / higher_size

    height, width = image.shape[:2]
    new_height, new_width = round(scale_factor * height), round(scale_factor * width)

    return new_width, new_height, scale_factor


def compute_mask_gradients(masks, dilation_kernel_size=5):
    """
    :param masks: tensor(N, T, H, W)
    :return:
    """
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size))
    mask_gradients = masks.to(torch.float32).numpy()
    mask_gradients = torch.stack([
        torch.stack([
            torch.from_numpy(cv2.dilate(cv2.Laplacian(mask_gradients[n, t], cv2.CV_32F), kernel))
            for t in range(mask_gradients.shape[1])
        ])
        for n in range(mask_gradients.shape[0])
    ]) > 0
    mask_gradients = mask_gradients.to(torch.uint8)
    return torch.any(mask_gradients, dim=0)


@torch.no_grad()
def instance_masks_to_semseg_mask(instance_masks, category_labels):
    """
    Converts a tensor containing instance masks to a semantic segmentation mask.
    :param instance_masks: tensor(N, T, H, W)  (N = number of instances)
    :param category_labels: tensor(N) containing semantic category label for each instance.
    :return: semantic mask as tensor(T, H, W] with pixel values containing class labels
    """
    assert len(category_labels) == instance_masks.shape[0], \
        "Number of instances do not match: {}, {}".format(len(category_labels), len(instance_masks))
    semseg_masks = instance_masks.long()

    #TODO (RUNINHO): one could fuse the max into this for loop
    for i, label in enumerate(category_labels):
        semseg_masks[i] = torch.where(instance_masks[i], label, semseg_masks[i])

    # for pixels with differing labels, assign to the category with higher ID number (arbitrary criterion)
    return semseg_masks.max(dim=0)[0]  # [T, H, W]


def visualize_semseg_masks(image, semseg_mask):
    category_labels = set(np.unique(semseg_mask).tolist()) - {0}
    if not category_labels:
        return image
    assert max(category_labels) < 256

    image = np.copy(image)
    cmap = create_color_map()

    for label in category_labels:
        image = overlay_mask_on_image(image, semseg_mask == label, mask_color=cmap[label])

    return image

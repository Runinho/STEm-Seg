"""load and manage video datasets"""

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

import cv2
import h5py
import numpy as np
import numpy.typing as npt
from PIL import Image
from pycocotools import mask as masktools

dataset = {}
def get_hdtf(base_dir, name):
    hdf5_file = Path(base_dir) / (name + ".h5")
    if hdf5_file in dataset:
        return dataset[hdf5_file]
    else:
        if not hdf5_file.is_file():
            print(f"couldn't load hdtf5 file for {hdf5_file}. please run `python -m scripts.create_hdtf5`")
            sys.exit()
        d = h5py.File(hdf5_file)
        dataset[hdf5_file] = d
        return d
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
    mask_filename = meta_info["mask_filename"]

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

    seqs = [GenericVideoSequence(base_dir, mask_filename, **seq) for seq in dataset["sequences"]]

    return seqs, meta_info


class GenericVideoSequence:
    """continuous image sequence"""

    def __init__(self, base_dir, mask_filename, image_paths, height, width, id, segmentations=None,
                 categories=None, **kwargs):
        """Stores all information to generated label and input data for a image sequence.

        We use the following wording:

            * **instance**: One instance of an labeled object. E.g. a specific car.
              Can occur in multiple images.
            * **category**: label of a segmentation mask: E.g Car, Person, ..,
              Can have multiple instances of the same category in the same frame.

        Input images are loaded from the provided ``image_paths`` in ``base_dir``.
        Masks are encoded in ``segmentations`` and ``categories``.
        We can access the segmentation mask for timestamp ``x`` (int) and
        instance ``y`` (int) in the following way: ``segmentations[x][y]``.
        Observe that we do not have a mask for each instance at every
        timestamp. The instances are not dense, so we use a dict for the instance dimension.
        Therefor ``segmentations`` is a list of dicts.
        ``categories`` is a dict mapping instance ids to category ids.

        Examples:
            Construct sample masks

            >>> import numpy as np
            >>> masks = []
            >>> for i in range(3):
            >>>     a = np.zeros((2,4), order="F", dtype=np.uint8)
            >>>     a[:,i] = 1
            >>>     masks.append(a)
            >>> masks
            [array([[1, 0, 0, 0],
                    [1, 0, 0, 0]], dtype=uint8),
             array([[0, 1, 0, 0],
                    [0, 1, 0, 0]], dtype=uint8),
             array([[0, 0, 1, 0],
                    [0, 0, 1, 0]], dtype=uint8)]

            Masks are in the segementation format as the cocodataset_ is using.
            We use pycocotools_ for encoding.

            >>> from pycocotools import  mask
            >>> encoded = [mask.encode(a)["counts"].decode("utf8") for a in masks]
            >>> seq = GenericVideoSequence(base_dir="some/dir",
            >>>                            image_paths=["1.png", "2.png"],
            >>>                            height=2,
            >>>                            width=4,
            >>>                            id="some_id",
            >>>                            categories={1: "Person", 2: "Car", 3: "Car"}
            >>>                            segmentations=[{1:encoded[0]},
            >>>                                           {1:encoded[1], 2:encoded[2]}])
            >>> seq.load_masks([1])
            [[array([[0, 1, 0, 0],
                     [0, 1, 0, 0]], dtype=uint8),
              array([[0, 0, 1, 0],
                     [0, 0, 1, 0]], dtype=uint8),
              array([[0, 0, 0, 0],
                     [0, 0, 0, 0]], dtype=uint8)]]
            >>> seq.load_masks([0, 1])
            [[array([[1, 0, 0, 0],
                     [1, 0, 0, 0]], dtype=uint8),
              array([[0, 0, 0, 0],
                     [0, 0, 0, 0]], dtype=uint8),
              array([[0, 0, 0, 0],
                     [0, 0, 0, 0]], dtype=uint8)],
             [array([[0, 1, 0, 0],
                     [0, 1, 0, 0]], dtype=uint8),
              array([[0, 0, 1, 0],
                     [0, 0, 1, 0]], dtype=uint8),
              array([[0, 0, 0, 0],
                     [0, 0, 0, 0]], dtype=uint8)]]

        Args:
            base_dir (Path): Directory of the datset
            mask_filename (str): name of the hdtf5 file containing the masks.
                                 (see scripts/create_hdtf5.py)
            image_paths (List): List of image paths
            height (int): Image height
            width (int): Image width
            id: Identifier of this sequence
            segmentations (List[Dict[InstanceID, str]], optional ): List of dicts containing
                the RLE encoded masks. The length of the list expected
                to be the same as ``image_paths``.
            categories (Dict[InstanceID, Category], optional): Dict mapping each instance to a
                category. The keys of this dict are used to access the masks
                in the dicts of `segmentation`.
            **kwargs: not used. Here for ease of use in ``parse_generic_video_dataset``.

        Links:
            RLE used for masks: https://github.com/ppwwyyxx/cocoapi

        .. _pycocotools:
            https://github.com/ppwwyyxx/cocoapi
        .. _cocodataset:
            https://cocodataset.org/
        """
        self.base_dir = base_dir
        self.image_paths = list([Path(f) for f in image_paths])
        self.image_dims = (height, width)
        self.id = id
        self.mask_filename = mask_filename

        self.segmentations = segmentations
        self.instance_categories = categories

    @property
    def instance_ids(self) -> List:
        """list of the instance ids"""
        return list(self.instance_categories.keys())

    @property
    def category_labels(self) -> Iterable:
        """category label for every instance"""
        return [self.instance_categories[instance_id] for instance_id in self.instance_ids]

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return f'GenericVideoSequence(id:{self.id}, len:{len(self)})'

    def load_images(self, frame_idxes=None) -> np.ndarray:
        """load images for a list of frames

        Args:
            frame_idxes (List[int], optional): image indices to load. If None returns all images.
                                     Defaults to None.

        Returns:
            list of images as numpy arrays
        """
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))


        f = get_hdtf(self.base_dir, "images")
        dataset = f[self.image_paths[0].parts[0]]
        indices = list([int(f.stem) for f in self.image_paths])
        # if the indices are consequtive we acess them as such (faster)
        if indices == list(range(indices[0], indices[-1] + 1)):
            images = dataset[indices[0]:indices[-1] + 1]
        else:
            images = dataset[indices]

        return images

    def load_masks(self, frame_idxes=None) -> List[List[npt.NDArray[np.uint8]]]:
        """ load masks for a list of frames

        Args:
            frame_idxes (int, optional): Indices of masks to load. If None returns all masks for all images.
                         Defaults to None.

        Returns:
            List[List[npt.NDArray[np.uint8]]]:
                list of list containing the image masks as 2D numpy array for
                every instance of categories.
                The shape of the image masks is (height, width).
                The first dimension is the frame indices, the second one the instance dimension.
                The length of the returned list is the same as ``frame_idxes``.
                The length of the lists contained in the list is the same as ``self.instance_categories``.
        """
        # None returns all
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        f = get_hdtf(self.base_dir, self.mask_filename[:-3])
        dataset = f[self.image_paths[0].parts[0]]
        masks = []
        # iterate over the time
        for t in frame_idxes:
            masks_t = []

            for instance_id in self.instance_ids:
                if instance_id in self.segmentations[t]: # self.segmentations is the mask id for the sequence
                    masks_t.append(np.ascontiguousarray(dataset[self.segmentations[t][instance_id]]))
                else:
                    # TODO: would be awsome if we could return just a 0.
                    # TODO: a more dirty trick would be to pass just a reference to a 0 array :D
                    # would greatly reduce the memory required
                    masks_t.append(np.zeros(self.image_dims, np.bool))

            masks.append(masks_t)
        return masks

    def filter_categories(self, cat_ids_to_keep: Iterable):
        """ remove instances from the dataset

        removes all instances from the dataset that are not in `cat_ids_to_keep`.

        Args:
            cat_ids_to_keep (Iterable): instance ids to keep.
        """
        instance_ids_to_keep = sorted(
            [iid for iid, cat_id in self.instance_categories.items() if iid in cat_ids_to_keep])
        for t in range(len(self)):
            self.segmentations[t] = {iid: seg for iid, seg in self.segmentations[t].items() if
                                     iid in instance_ids_to_keep}

    def filter_zero_instance_frames(self):
        """ remove all frames that contain no label"""
        t_to_keep = [t for t in range(len(self)) if len(self.segmentations[t]) > 0]
        self.image_paths = [self.image_paths[t] for t in t_to_keep]
        self.segmentations = [self.segmentations[t] for t in t_to_keep]

    def apply_category_id_mapping(self, mapping: Dict):
        """rename categories

        Args:
            mapping (Dict): Renaming mapping from old category to the new category ids.

        Examples:
            Rename categories from describtive names to ints

            >>> seq  = GenericVideoSequence(base_dir="some/dir",
            >>>                             image_paths=["1.png", "2.png"],
            >>>                             height=128,
            >>>                             width=256,
            >>>                             id="some_id",
            >>>                             categories={1: "Person", 2: "Car", 3: "Car"})
            >>> (seq.instance_ids, seq.category_labels)
            ([1, 2, 3], ['Person', 'Car', 'Car'])
            >>> seq.apply_category_id_mapping({"Person":11, "Car":22})
            >>> (seq.instance_ids, seq.category_labels)
            ([1, 2, 3], [11, 22, 22])
        """
        assert set(mapping.keys()) == set(self.instance_categories.values())
        self.instance_categories = {
            iid: mapping[current_cat_id] for iid, current_cat_id in self.instance_categories.items()
        }

    def extract_subsequence(self, frame_idxes, new_id="") -> 'GenericVideoSequence':
        """extract subsequence for a list of frames

        Args:
            frame_idxes: frames to be included in the returned Sequence
            new_id: id of the extracted Subsequence

        Returns:
            GenericVideoSequence:
                sequence only containing the images from indices ``frame_idxes`` with id ``new_id``
        """
        # check if frame_idxes are in range
        assert all([t in range(len(self)) for t in frame_idxes])

        # filter the data
        instance_ids_to_keep = set(
            sum([list(self.segmentations[t].keys()) for t in frame_idxes], []))

        new_categories = {iid: self.instance_categories[iid] for iid in instance_ids_to_keep}
        new_segmentations = [
            {
                iid: segmentations_t[iid]
                for iid in segmentations_t if iid in instance_ids_to_keep
            }
            for t, segmentations_t in enumerate(self.segmentations) if t in frame_idxes
        ]
        return self.__class__(self.base_dir,
                              self.mask_filename,
                              id=new_id if new_id else self.id,
                              height=self.image_dims[0],
                              width=self.image_dims[1],
                              image_paths=[self.image_paths[t] for t in frame_idxes],
                              categories=new_categories,
                              segmentations=new_segmentations)

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
            # wait for q
            if cv2.waitKey(0) == ord('q'):
                exit(0)

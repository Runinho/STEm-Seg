# load the inference data
import abc
from collections import defaultdict
from typing import Union, Dict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from enum import Enum
import torch.nn.functional as F

from stemseg.config import cfg
from stemseg.data import parse_generic_video_dataset
from stemseg.data.generic_video_dataset_parser import GenericVideoSequence
from stemseg.data.paths import KITTISTEPPaths
from stemseg.inference.main import TrackGenerator
from stemseg.inference.online_chainer import OnlineChainer
from stemseg.inference.output_utils import KittiMOTSOutputGenerator


class InferenceSplitType(Enum):
    TRAIN = "train"
    VAL = "val"

# custom output_generator to create the data we need for visualization
class VisualizationKittiMOTSOutputGenerator(KittiMOTSOutputGenerator):
    def save_sequence_visualizations(self, seq, instances):
        pass
        # save it so we can see it LOL

# provides all the data that is required for the visualization
# provides only data for one sequence
class InferenceDataProvider:
    def __init__(self, split_type:InferenceSplitType):
        vds_file_map = {
            InferenceSplitType.TRAIN: KITTISTEPPaths.train_vds_file,
            InferenceSplitType.VAL: KITTISTEPPaths.val_vds_file
        }
        # load the label filename
        vds_file = vds_file_map[split_type]()
        self.sequences, self.meta_info = parse_generic_video_dataset(KITTISTEPPaths.train_images_dir(),
                                                                     vds_file)
        self.categories = self.meta_info["category_labels"]
        self.init_providers()

    @abc.abstractmethod
    def init_providers(self):
        raise NotImplementedError()

    def get_sequence_ids(self):
        return [s.id for s in self.sequences]

    def get_sequence_by_id(self, id: Union[int, str]):
        converter = lambda x: x # identity function (do nothing)
        if type(id) is int:
            converter = int
        id_2_seq = {converter(s.id): p for s, p in zip(self.sequences, self.sequence_providers)}
        if id in id_2_seq:
            return id_2_seq[id]
        else:
            print(f"id {id} not in available sequences: {', '.join(id_2_seq.keys())}")

class OfflineInferenceDataProvider(InferenceDataProvider):

    def __init__(self, split_type:InferenceSplitType,  pred_location:Path):
        self.pred_location = pred_location
        super().__init__(split_type)
    def init_providers(self):
        self.sequence_providers = list([OfflineInferenceDataProviderSequence(s, self.pred_location, self.categories) for s in self.sequences])

class OnlineInferenceDataProvider(InferenceDataProvider):

    def __init__(self, model_path, split_type=InferenceSplitType.VAL):
        self.model_path = model_path
        super().__init__(split_type=split_type)

    def init_providers(self):
        dataset= "kittistep"
        # TODO: make this customizable
        seediness_thresh = 0.25
        output_resize_scale = 1
        semseg_averaging_on_gpu = False
        clustering_device = "cuda:0"
        frame_overlap = -1

        cluster_full_scale = False
        save_vis = True

        output_generator = VisualizationKittiMOTSOutputGenerator("tmp",
                                                                 OnlineChainer.OUTLIER_LABEL,
                                                                 save_vis,
                                                                 upscaled_inputs=cluster_full_scale,
                                                                 category_label=self.categories)

        max_tracks = cfg.DATA.KITTI_MOTS.MAX_INFERENCE_TRACKS
        preload_images = False

        # move model loading to the inference part and not to the init part...
        self.track_generator = TrackGenerator(
            self.sequences, dataset, output_generator, "tmp", self.model_path,
            save_vis=save_vis,
            seediness_thresh=seediness_thresh,
            frame_overlap=frame_overlap,
            max_tracks=max_tracks,
            preload_images=preload_images,
            resize_scale=output_resize_scale,
            semseg_averaging_on_gpu=semseg_averaging_on_gpu,
            clustering_device=clustering_device
        )
        self.sequences = self.track_generator.sequences
        self.sequence_providers = list([OnlineInferenceDataProviderSequence(self.track_generator, s, self.categories) for s in self.track_generator.sequences])

class InferenceDataProviderSequence:
    def __init__(self, sequence:GenericVideoSequence, categories: Dict[int, str]):
        self.sequence = sequence
        self.categories = categories

    def __len__(self) -> int:
        return len(self.sequence)

    @abc.abstractmethod
    def get_images(self) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_output(self) -> np.ndarray:
        raise NotImplementedError()

class OnlineInferenceDataProviderSequence(InferenceDataProviderSequence ):
    def __init__(self, track_generator, sequence, categories):
        """calculate the model output with the model and input data

        Args:
            track_generator (TrackGenerator): used to create the output of the model
            sequence (GenericVideoSequence): input datat
            categories (Dict[int, str]): dict of the category names
        """
        super().__init__(sequence, categories)
        self.track_generator = track_generator
        self.categories = categories
        self.images = None
        self.output = None

    def run_inference(self):
        print(self.sequence.id)
        # multiclass and mask and (result of the head)
        embeddings, fg_masks, multiclass_masks = self.track_generator.do_inference(self.sequence)

        # stitch sequence together
        subseq_dicts = []
        all_embeddings = embeddings


        for i, (subseq_frames, embeddings, bandwidths, seediness) in tqdm(enumerate(all_embeddings), total=len(all_embeddings)):
            subseq_dicts.append({
                "frames": subseq_frames,
                "embeddings": embeddings,
                "bandwidths": bandwidths,
                "seediness": seediness,
            })

        (track_labels, instance_pt_counts, instance_lifetimes), framewise_mask_idxes, subseq_labels_list, \
                    fg_embeddings, subseq_clustering_meta_info = self.track_generator.chainer.process(
            fg_masks, subseq_dicts)

        #clustering
        instances_to_keep, instance_id_mapping, instance_rle_masks, instance_semantic_label_votes = \
            self.track_generator.output_generator.process_sequence(
                self.sequence, framewise_mask_idxes, track_labels, instance_pt_counts, instance_lifetimes, multiclass_masks,
                fg_masks.shape[-2:], 4.0, cfg.DATA.KITTI_MOTS.MAX_INFERENCE_TRACKS, device=self.track_generator.clustering_device
            )

        #output formating
        image_size = self.sequence.load_images(0).shape[1:3]

        # scale up multiclass
        # torch wants [N, C, w, h] so we add one channel dimension and remove it afterwards
        upsampled_multiclass_masks = F.interpolate(multiclass_masks.byte()[:, None], image_size, mode='nearest').long()[:,0]

        # shift the category id
        #upsampled_multiclass_masks = upsampled_multiclass_masks * max_instances_per_category
        # set the instance ids

        # arrange the masks according to frame rather than by instance.
        instances_by_frame = defaultdict(list)
        for instance in instance_rle_masks.values():
            for frame_instance in instance:
                instances_by_frame[frame_instance["frame_id"]].append(frame_instance)

        self.images = self.sequence.load_images()
        self.output = np.zeros_like(self.images)
        # set the instance ids
        # so we get format as the STEP labels (categories id shifted to the left, etc)
        for t, (out_img, image_t) in enumerate(zip(self.output, self.images)):
            out_img[:,:,0] = upsampled_multiclass_masks[t]
            for instance in instances_by_frame[t]:
                category_label = instance["category_id"]
                # NOTE: instance_id should be 1 based so the instance id 0 is reserved as the "crowd case"
                instance_id = instance["instance_id"]
                mask = instance["raw_mask"].cpu()
                out_img[mask, 2] = instance_id % 256
                out_img[mask, 1] = instance_id / 256
                # set the category_label
                #upsampled_multiclass_masks[t][mask] = category_label * max_instances_per_category + instance_id

            # # save the image
            # pred_filename = pred_out_folder / sequence.image_paths[t]
            # if not pred_filename.parent.exists():
            #     print(f" creating {pred_filename.parent}")
            #     pred_filename.parent.mkdir(parents=True, exist_ok=True)
            # Image.fromarray(out_img).save(pred_filename)

    def get_images(self):
        if self.images is None:
            self.run_inference()
        return self.images

    def get_outputs(self):
        if self.output is None:
            self.run_inference()
        return self.output


class OfflineInferenceDataProviderSequence(InferenceDataProviderSequence):
    def __init__(self, sequence, result_folder, categories):
        """reads results from disk

        Args:
            sequence (GenericVideoSequence): input data
            result_folder (Path): folder that contains the output of the model in the STEP image format
        """
        super().__init__(sequence, categories)
        self.result_folder = result_folder

    def get_images(self):
        return self.sequence.load_images()

    def get_outputs(self):
        imgs = [np.array(Image.open(self.result_folder / f)) for f in self.sequence.image_paths]
        return np.array(imgs)


#TODO: create common type
InferenceDataProviderSequence = Union[OnlineInferenceDataProviderSequence, OfflineInferenceDataProviderSequence]

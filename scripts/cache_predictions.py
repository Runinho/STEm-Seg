# cache predictions to visualize later
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from stemseg.config.config import load_global, cfg
from stemseg.data import parse_generic_video_dataset
from stemseg.data.paths import KITTISTEPPaths
from stemseg.inference.main import TrackGenerator
from stemseg.inference.online_chainer import OnlineChainer
from stemseg.inference.output_utils import KittiMOTSOutputGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='cache_predictions',
        description='runs inference for a given model and split so it can be used in the visualization')
    parser.add_argument("model")
    parser.add_argument("-t", "--train", default=False, action='store_true', help="runs for training datasplit")
    args = parser.parse_args()
    print(args)

    vds_file = KITTISTEPPaths.val_vds_file()
    split = "validation"
    if args.train:
        vds_file = KITTISTEPPaths.train_vds_file()
        split = "training"

    print(f"running on split {split} ({vds_file})")

    # TODO: maybe reuse the code in the inference helper of the visualization
    load_global("kitti_step_2.yaml")

    # load sequences
    sequences, meta_info = parse_generic_video_dataset(KITTISTEPPaths.train_images_dir(), vds_file)

    categories = meta_info["category_labels"]

    output_dir = "tmp"
    save_vis = False
    cluster_full_scale = False

    output_generator = KittiMOTSOutputGenerator(output_dir, OnlineChainer.OUTLIER_LABEL, save_vis,
                                                upscaled_inputs=cluster_full_scale,
                                                category_label=categories)
    max_tracks = cfg.DATA.KITTI_MOTS.MAX_INFERENCE_TRACKS
    preload_images = False

    # load model
    dataset= "kittistep"
    model_path = Path(args.model)
    seediness_thresh = 0.25
    output_resize_scale = 1
    semseg_averaging_on_gpu = False
    clustering_device = "cuda:0"
    frame_overlap = -1

    print(f"loading model from {model_path}")

    track_generator = TrackGenerator(
        sequences, dataset, output_generator, output_dir, str(model_path),
        save_vis=save_vis,
        seediness_thresh=seediness_thresh,
        frame_overlap=frame_overlap,
        max_tracks=max_tracks,
        preload_images=preload_images,
        resize_scale=output_resize_scale,
        semseg_averaging_on_gpu=semseg_averaging_on_gpu,
        clustering_device=clustering_device
    )

    # output to file
    max_instances_per_category = 256

    pred_out_folder = model_path.parent /f"STEP_pred_{split}"
    print(f"saving predictions into {pred_out_folder}")

    # count the number of inference results we write out
    num_images = 0
    for sequence in track_generator.sequences:
        print(sequence.id)
        # multiclass and mask and (result of the head)
        embeddings, fg_masks, multiclass_masks = track_generator.do_inference(sequence)

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
        fg_embeddings, subseq_clustering_meta_info = track_generator.chainer.process(
            fg_masks, subseq_dicts)

        #clustering
        instances_to_keep, instance_id_mapping, instance_rle_masks, instance_semantic_label_votes = \
            track_generator.output_generator.process_sequence(
                sequence, framewise_mask_idxes, track_labels, instance_pt_counts, instance_lifetimes, multiclass_masks,
                fg_masks.shape[-2:], 4.0, max_tracks, device=track_generator.clustering_device
            )

        #output formating
        image_size = sequence.load_images(0).shape[1:3]

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

        # TODO: we only need the length why also load the images LOL
        images = sequence.load_images()
        # set the instance ids
        # so we get format as the STEP labels (categories id shifted to the left, etc)
        for t, image_t in enumerate(images):
            out_img = np.zeros(((*image_size, 3)), dtype=np.uint8)
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

            # save the image
            pred_filename = pred_out_folder / sequence.image_paths[t]
            if not pred_filename.parent.exists():
                print(f" creating {pred_filename.parent}")
                pred_filename.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(out_img).save(pred_filename)
            num_images += 1

    print(f"done. saved {num_images} prediction results.")

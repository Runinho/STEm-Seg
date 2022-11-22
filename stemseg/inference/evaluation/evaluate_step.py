import glob
import os
import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pickle
from pathlib import Path

from stemseg.inference.evaluation.stq import STQuality

LABEL_BIT_SHIFT = 1
#THINGS_CLASSES = np.setdiff1d(np.arange(0, 19), [12, 14])
# TODO: check if we also want to add 20 to this because that class is the ignore class??
THINGS_CLASSES = [11, 13] #np.setdiff1d(np.arange(0, 19), [11, 13])
MAX_INSTANCES_PER_CATEGORY = 128
stq = STQuality(19, THINGS_CLASSES, 255, max_instances_per_category=MAX_INSTANCES_PER_CATEGORY, offset=100000)

def main(args):
  pred_path = args.pred_path
  gt_path = args.gt_path
  pickle_path = Path(pred_path).parent / f"{Path(pred_path).stem}_stq_obj_per_t_train.pkl"

  sequence_paths = glob.glob(os.path.join(gt_path, '*'))
  sequence_ids = [_id.split("/")[-1] for _id in sequence_paths]

  pred_sequence_paths = [os.path.join(pred_path, '{:04d}'.format(int(_id))) for _id in sequence_ids]

  # with mp.Pool(5) as p:
  #   p.starmap(process_seq, zip(sequence_ids, pred_sequence_paths, sequence_paths))
  for seq_id, pred_seq_path, gt_seq_path in tqdm(zip(sequence_ids, pred_sequence_paths, sequence_paths)):
    process_seq(seq_id, pred_seq_path, gt_seq_path)

  print("---KITTI-STEP VALIDATIOn STQ: {}---".format(stq.result()))
  pickle.dump(stq, open(pickle_path, "wb"))


def process_seq(seq_id, pred_seq_path, gt_seq_path):
    pred_seq_files = glob.glob(pred_seq_path + "/*.png")
    gt_seq_files = glob.glob(gt_seq_path + "/*.png")
    pred_seq_files.sort()
    gt_seq_files.sort()
    preds = np.stack([np.asarray(Image.open(_p)) for _p in pred_seq_files])
    gts = np.stack([np.asarray(Image.open(_g)) for _g in gt_seq_files])
    # for pred,gt in zip(preds, gts):
    # stq needs the predictions and gt to be encoded as semantic_map * max_instances_per_category + instance_map
    # TODO: rewrite stq to take seperate arrays for instance id and category id
    pred_instance = preds[..., 1] * 256 + preds[..., 2]
    assert pred_instance.max() < MAX_INSTANCES_PER_CATEGORY,\
        f"instance ids of all prediction must be smaller than the MAX_INSTNACES_PER_CATEGORY " \
        f"{MAX_INSTANCES_PER_CATEGORY} is {max(pred_instance)}"
    pred_stq = (preds[..., 0].astype(np.int64) * MAX_INSTANCES_PER_CATEGORY) + pred_instance
    gt_instance = gts[..., 1] * 256 + gts[..., 2]
    assert gt_instance.max() < MAX_INSTANCES_PER_CATEGORY, \
        f"instance ids of all label data must be smaller than the MAX_INSTNACES_PER_CATEGORY" \
        f" {MAX_INSTANCES_PER_CATEGORY} is {max(gt_instance)}"
    gt_stq = (gts[..., 0].astype(np.int64) * MAX_INSTANCES_PER_CATEGORY) + gt_instance
    stq.update_state(torch.tensor(gt_stq.astype(np.int)).int(), torch.tensor(pred_stq.astype(np.int)).int(),
                     sequence_id=seq_id)
    # stq.update_state(gt_stq, pred_stq)
    print("Processed Sequence {:04d} with unique labels {}. \nUpdated STQ {}".
          format(int(seq_id), np.unique(preds[..., 0]), stq.result()))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--pred_path", required=True)

    main(parser.parse_args())

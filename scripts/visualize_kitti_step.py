from stemseg.data.generic_video_dataset_parser import visualize_generic_dataset
from stemseg.data.paths import KITTIMOTSPaths


base_dir = KITTIMOTSPaths.train_images_dir()
dataset_json = "notebooks/kittistep_train.json"

if __name__ == "__main__":
    visualize_generic_dataset(base_dir, dataset_json)
import json
import cv2
from pathlib import Path
#%%
json_path = Path("..") / "stemseg" / "data" / "metainfo" / "mapillary_image_dims.json"
org_dir = Path("/work/frohn/data/mapillary/all_images_org")
resized_dir = Path("/work/frohn/data/mapillary/all_images_resized")
#%%
size_dict = json.load(open(json_path, "r"))

for i, (image_name, new_size) in enumerate(size_dict.items()):
        img = cv2.imread(str(org_dir / image_name))
        img = cv2.resize(img, new_size)
        cv2.imwrite(str(resized_dir / image_name), img)
        print(i)

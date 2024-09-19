import json
import os
import glob

dataset_content = []

images = []
base_image_dir = "./dataset"
split = "train"
images_split = glob.glob(
                os.path.join(
                    base_image_dir, "ReasonSeg", split, "*.jpg"
                )
            )
images.extend(images_split)

for image in images:
    path_seg = image.split("/")
    image_path = os.path.join(*(p for p in path_seg[2:]))
    id = path_seg[-1].split(".")[0]
    meta = image_path.replace(".jpg", ".json")
    dataset_content.append({
        "id": id,
        "image": image_path,
        'meta': meta,
    })

with open('dataset/ReasonSeg/index.json', 'w') as json_file:
    json.dump(dataset_content, json_file)

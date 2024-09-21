import os
from tqdm import tqdm
import xmltodict

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

STATE_IDX = {
    "0": "background",
    "1": "left hand",
    "2": "right hand",
    "3": "object contact by left hand",
    "4": "object contact by right hand",
    "5": "object contact by both left and right hands"
}
DEFAULT_COLOR = [
    (0, 0, 0), # Black
    (255, 0, 0), # Red
    (0, 255, 0), # Green
    (0, 0, 255), # Blue
    (255, 255, 0), # Yellow
    (0, 255, 255), # Cyan
]

dataset_dir = "/media/automan/6E94666294662CB1/A_Content/Datasets/100DOH_pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007"
save_dir = "/media/automan/6E94666294662CB1/A_Content/Datasets/100DOH"
with open(os.path.join(dataset_dir, "ImageSets/Main", "train.txt"), "r") as f:
    train_lines = f.read().splitlines()

enable_resume = True
with open("./checkpoint.txt", "r") as f:
    train_lines_buffer = f.read().splitlines()
top_k = 0
resume_buffer = []
if top_k > 0:
    samples = train_lines[:top_k]
else:
    samples = train_lines

sam_checkpoint = "/home/automan/MobileLISA_ws/LISA/pretrained/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
for sample in tqdm(samples):
    if enable_resume and sample in train_lines_buffer:
        print(f"{sample} repeat, skip!")
        continue
    # 读取 XML 文件并转换为字典
    with open(os.path.join(dataset_dir, "Annotations", f"{sample}.xml"), 'r') as xml_file:
        xml_content = xml_file.read()
        xml_dict = xmltodict.parse(xml_content)
    image = cv2.imread(os.path.join(dataset_dir, "JPEGImages", f"{sample}.jpg"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotation = np.zeros(image.shape[:2], dtype=np.uint8)
    masked_image = np.zeros_like(image)
    if not isinstance(xml_dict['annotation']['object'], list):
        objs = [xml_dict['annotation']['object']]
    else:
        objs = xml_dict['annotation']['object']
    for obj in objs:
        if obj['name'] == 'hand':
            seg_idx = int(obj['handside']) + 1
        elif obj['name'] == 'targetobject':
            contact_left = bool(int(obj['contactleft']))
            contact_right = bool(int(obj['contactright']))
            if contact_left and not contact_right:
                seg_idx = 3
            elif contact_right and not contact_left:
                seg_idx = 4
            elif contact_left and contact_right:
                seg_idx = 5
            else:
                raise NotImplementedError(f"Not implement neither left hand nor right hand")
        else:
            raise NotImplementedError(f"Not implement {obj['name']}")
        bbox = np.array([obj['bndbox']['xmin'], obj['bndbox']['ymin'], obj['bndbox']['xmax'], obj['bndbox']['ymax']])
        # SAM
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False,
        )
        mask = masks[0].astype(bool)
        annotation[mask] = seg_idx
        masked_image[mask, :] = DEFAULT_COLOR[seg_idx]
    masked_image = (image*0.5 + masked_image*0.5).astype(np.uint8)
    resume_buffer.append(sample+"\n")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, "images", "train", f"{sample}.jpg"), image)
    cv2.imwrite(os.path.join(save_dir, "annotations", "train", f"{sample}.png"), annotation)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, "visualizations", "train", f"{sample}.jpg"), masked_image)

if enable_resume:
    with open("./checkpoint.txt", "a") as f:
        f.writelines(resume_buffer)
else:
    with open("./checkpoint.txt", "w") as f:
        f.writelines(resume_buffer)

# 输出转换后的字典
# print(xml_dict)

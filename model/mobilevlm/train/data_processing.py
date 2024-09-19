import glob
import json
import os
import random
from typing import Dict, Union

import cv2
from PIL import Image
import numpy as np

from mobilevlm.constants import SHORT_QUESTION_LIST, LONG_QUESTION_LIST, ANSWER_LIST, ADE20K_CLASSES




def get_mask_from_json(json_path: str,
                       img: Union[Image.Image, np.ndarray],
                       ) -> np.ndarray:
    if isinstance(img, Image.Image):
        img = np.array(img)
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]

    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask


def meta_info_retrieve(sample: Dict,
                       json_path: str,
                       num_classes_per_sample: int):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())
    sents, is_sentence = anno["text"], anno["is_sentence"]
    if len(sents) >= num_classes_per_sample:
        sampled_indices = np.random.choice(
            list(range(len(sents))), size=num_classes_per_sample, replace=False
        )
    else:
        sampled_indices = list(range(len(sents)))
    sampled_sents = np.vectorize(sents.__getitem__)(sampled_indices).tolist()
    questions = []
    answers = []
    for text in sampled_sents:
        if is_sentence:
            question_template = random.choice(LONG_QUESTION_LIST)
            q = {'from': 'human', 'value': question_template.format(sent=text)}
        else:
            question_template = random.choice(SHORT_QUESTION_LIST)
            q = {'from': 'human', 'value': question_template.format(class_name=text.lower())}
        questions.append(q)
        a = {'from': 'gpt', 'value': random.choice(ANSWER_LIST)}
        answers.append(a)
    conversations = []
    for question, answer in zip(questions, answers):
        conversations.extend([question, answer])
    sample['conversations'] = conversations
    sample['sampled_indices'] = sampled_indices

def seg_add_ref(sample: Dict,
                num_classes_per_sample: int = 1,
                data_type = "ADE"):
    image_path, label_path = sample['image'], sample['annotation']
    label = cv2.imread(label_path, 0)
    # Process annotation
    if data_type == "ADE":
        label[label == 0] = 255
        label -= 1
        label[label == 254] = 255
    unique_label = np.unique(label).tolist()
    if 255 in unique_label:
        unique_label.remove(255)
    # Random select label idx
    if len(unique_label) > num_classes_per_sample:
        unique_label = random.sample(unique_label, num_classes_per_sample)

    questions = []
    answers = []
    # Generate grounding sentence
    for label_idx in unique_label:
        if data_type == "ADE":
            class_name = ADE20K_CLASSES[label_idx]
        else:
            raise NotImplementedError(f"data_type {data_type} not implemented")

        question_template = random.choice(SHORT_QUESTION_LIST)
        q = {'from': 'human', 'value': question_template.format(class_name=class_name.lower())}
        questions.append(q)
        a = {'from': 'gpt', 'value': random.choice(ANSWER_LIST)}
        answers.append(a)

    # Combine Q&A
    conversations = []
    for question, answer in zip(questions, answers):
        conversations.extend([question, answer])

    sample['conversations'] = conversations
    sample['sampled_indices'] = unique_label

def ADE_preprocess(base_image_dir):
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "images", "training"))
    )
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "images",
                "training",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))

    data_list = []
    for image, label in zip(ade20k_images, ade20k_labels):
        data_list.append({"image": image, "annotation": label})
    with open(os.path.join(base_image_dir, "training.json"), "w") as f:
        json.dump(data_list, f)


if __name__ == "__main__":
    ADE_preprocess("/media/armine/6E94666294662CB1/A_Content/Datasets/ADEChallengeData2016")

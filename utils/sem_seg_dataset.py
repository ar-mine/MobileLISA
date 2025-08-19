import glob
import json
import os
import random
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
import matplotlib.pyplot as plt

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST, AFFORD_QUESTION_LIST, DEFAULT_IMAGE_TOKEN


def init_mapillary(base_image_dir):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir, split="train"):
    with open("utils/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    if split == "train":
        split = "training"
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ade20k/images", split))
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
                "ade20k",
                "images",
                split,
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_100DOH(base_image_dir, split="train"):
    with open("utils/100DOH_classes.json", "r") as f:
        _100DOH_classes = json.load(f)
    _100DOH_classes = [_100DOH_classes[str(i)] for i in range(len(_100DOH_classes))]
    _100DOH_classes = _100DOH_classes[1:]
    _100DOH_classes = np.array(_100DOH_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "100DOH/images", split))
    )
    _100DOH_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            _100DOH_image_ids.append(x[:-4])
    _100DOH_images = []
    for image_id in _100DOH_image_ids:  # self.descriptions:
        _100DOH_images.append(
            os.path.join(
                base_image_dir,
                "100DOH",
                "images",
                split,
                "{}.jpg".format(image_id),
            )
        )
    _100DOH_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in _100DOH_images
    ]
    print("100DOH: ", len(_100DOH_images))
    return _100DOH_classes, _100DOH_images, _100DOH_labels


def init_agd20k(base_image_dir):
    _agd20k_images = []
    _agd20k_classes = []

    root_path = os.path.join(base_image_dir, "agd20k")
    for split in os.listdir(root_path):
        # Seen / Unseen
        if split not in ["Seen", "Unseen"]:
            raise ValueError("split must be 'Seen' or 'Unseen'")
        split_path = os.path.join(root_path, split, "testset/egocentric")
        for split_action in os.listdir(split_path):
            action_path = os.path.join(split_path, split_action)
            for split_category in os.listdir(action_path):
                _agd20k_image_ids = os.listdir(
                    os.path.join(action_path, split_category))
                for image_id in _agd20k_image_ids:
                    if 'json' not in image_id:
                        _agd20k_images.append(
                            os.path.join(
                                action_path,
                                split_category,
                                image_id
                            )
                        )
                        _agd20k_classes.append([split_action, split_category])
    _agd20k_labels = [
        x.replace(".jpg", ".png").replace("egocentric", "GT")
        for x in _agd20k_images if os.path.exists(x.replace(".jpg", ".png").replace("egocentric", "GT"))
    ]
    _agd20k_images = [
        x.replace(".png", ".jpg").replace("GT", "egocentric")
        for x in _agd20k_labels
    ]
    print("agd20k: ", len(_agd20k_images))
    return _agd20k_classes, _agd20k_images, _agd20k_labels


def init_cocostuff(base_image_dir):
    cocostuff_classes = []
    with open("utils/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "cocostuff", "train2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_paco_lvis(base_image_dir):
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part


def combine_images_2x3(image_paths, gray:bool=False):
    """
    Combine six images into a 2x3 grid and resize to half size.
    Returns the combined and resized image as a numpy array.

    Args:
    - image_paths (list of str): List of six image file paths.

    Returns:
    - np.ndarray: The combined and resized image array.
    """
    if len(image_paths) != 6:
        raise ValueError("Exactly six image paths must be provided.")

    # Find the first non-None image to get shape and dtype
    first_image = None
    for path in image_paths:
        if path is not None:
            if gray:
                first_image = np.array( Image.open(path).convert('L'))
                h, w = first_image.shape
            else:
                first_image = plt.imread(path)
                h, w, c = first_image.shape
            dtype = first_image.dtype
            break

    if first_image is None:
        raise ValueError("At least one image path must be provided.")

    # Load all images or create zeros, check consistency
    images = []
    for path in image_paths:
        if path is not None:
            if gray:
                img = np.array(Image.open(path).convert('L'))
                if img.shape != (h, w) or img.dtype != dtype:
                    raise ValueError("All images must have the same dimensions and dtype.")
            else:
                img = plt.imread(path)
                if img.shape != (h, w, c) or img.dtype != dtype:
                    raise ValueError("All images must have the same dimensions and dtype.")
            images.append(img)
        else:
            if gray:
                zero_img = np.zeros((h, w), dtype=dtype)
            else:
                zero_img = np.zeros((h, w, c), dtype=dtype)
            images.append(zero_img)

    # Create combined image array
    if gray:
        combined = np.zeros((2 * h, 3 * w), dtype=dtype)
    else:
        combined = np.zeros((2 * h, 3 * w, c), dtype=dtype)

    # Place images in 2x3 grid
    for i in range(2):
        for j in range(3):
            idx = i * 3 + j
            combined[i * h:(i + 1) * h, j * w:(j + 1) * w] = images[idx]

    # If the images are in float format, convert to uint8
    if combined.dtype == np.float32 or combined.dtype == np.float64:
        combined = (combined * 255).astype(np.uint8)

    return combined


CAM_ORDER = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT')
def init_drivelm(base_image_dir):
    file_path = os.path.join(base_image_dir, "drivelm", "v1_0_train_nus_grounding.json")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    key_frames = []
    for k, v in data.items():
        key_frames.extend([value for value in v['key_frames'].values()])

    for frame in key_frames:
        frame['multimodal'] = {'perception': []}
        frame['pure_text'] = {'perception': []}
        # frame['QA']['perception'] = [p for p in frame['QA']['perception'] if p['Q'].count('<c') == 0]
        for i, p in enumerate(frame['QA']['perception']):
            if p['Q'].count('<c') == 0 and 4 >= p['A'].count('<c') >= 1:
                frame['multimodal']['perception'].append(i)
            if p['Q'].count('<c') == 0 and p['A'].count('<c') == 0:
                frame['pure_text']['perception'].append(i)

    print("DriveLm: ", len(key_frames))
    return None, None, key_frames

class SemSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary||100DOH||agd20k||drivelm",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            classes, images, labels = eval("init_{}".format(ds))(base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

        if "cocostuff" in self.sem_seg_datas:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }

        self.count = 0
        self.multimodal_count = 0

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]

        if ds in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if len(anns) == 0:
                return self.__getitem__(0)
            if len(anns) >= self.num_classes_per_sample:
                sampled_anns = np.random.choice(
                    anns, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_anns = anns
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    if random.random() < 0.5:
                        name = obj + " " + part
                    else:
                        name = "the {} of the {}".format(part, obj)
                else:
                    name = sampled_cls
                sampled_classes.append(name)

        elif ds in ["ade20k", "cocostuff", "mapillary", "100DOH"]:
            image, labels = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            label_path = labels[idx]
            label = Image.open(label_path)
            label = np.array(label)
            if ds in ["ade20k", "100DOH", "agd20k"]:
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]
            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)
            if len(unique_label) == 0:
                return self.__getitem__(0)

            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            if len(classes) >= self.num_classes_per_sample:
                sampled_classes = np.random.choice(
                    classes, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = classes

        elif ds == "agd20k":
            image, labels = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            label_path = labels[idx]
            label = Image.open(label_path)
            label = np.array(label)>128

            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]

            sampled_classes = self.data2classes[ds][idx]

        elif ds == "drivelm":
            _, info_list = self.data2list[ds]
            idx = random.randint(0, len(info_list) - 1)
            info = info_list[idx]
            image_path_list = [os.path.join(self.base_image_dir, "drivelm", info['image_paths'][idx][3:]) for idx in CAM_ORDER]
            image = combine_images_2x3(image_path_list)

            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]

            image_path = None
            sampled_classes = None

        else:
            raise NotImplementedError

        questions = []
        answers = []
        class_ids = []
        if ds == "agd20k":
            question_template = random.choice(AFFORD_QUESTION_LIST)
            questions.append(
                question_template.format(object_name=sampled_classes[0].lower(),
                                         action_name=sampled_classes[1].lower())
            )
            answers.append(random.choice(self.answer_list))
        elif ds == "drivelm":
            candidates = info['QA']['perception']
            if len(info['multimodal']['perception']) == 0:
                multimodal = False
                candidate_idx = random.choice(info['pure_text']['perception'])
            else:
                if random.random() < 0.5:
                    multimodal = False
                    candidate_idx = random.choice(info['pure_text']['perception'])
                else:
                    multimodal = True
                    candidate_idx = random.choice(info['multimodal']['perception'])
            candidate = candidates[candidate_idx]
            question = candidate['Q']
            answer = candidate['A']
            # for candidate in candidates:
            #     question = candidate['Q']
            #     answer = candidate['A']
            #     if question.count('<c') == 0 and 4 >= answer.count('<c') >= 1:
            #         multimodal = True
            #         break
            questions.append(DEFAULT_IMAGE_TOKEN + "\n" + question)
            # Now only support one
            if multimodal:
                tag = re.findall(r'<(.*?)>', answer)
                for t in tag:
                    full_tag = f"<{t}>"
                    full_tag_processed = f"<{t}>".replace(' ', '')
                    category = info['key_object_infos'][full_tag_processed]['Visual_description'].lower().replace('.', '')
                    answer = answer.replace(full_tag, f'{category} <SEG>')
            answers.append(answer)
        else:
            for sampled_cls in sampled_classes:
                text = sampled_cls

                assert len(text.split("||")) == 1
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

                answers.append(random.choice(self.answer_list))

                if ds in ["paco_lvis", "pascal_part"]:
                    continue

                class_id = self.data2classes[ds].tolist().index(sampled_cls)
                class_ids.append(class_id)

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        if ds == "drivelm":
            conv.system = ("As an AI assistant specialized in analyzing 2x3 grid collages of vehicle driving scenes "
                           "from six perspectives (top row: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT; bottom row: "
                           "CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT), carefully examine the provided image, interpret "
                           "all views, and provide accurate, detailed, context-aware, helpful responses to user "
                           "questions based on visible elements.")

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if ds in ["paco_lvis", "pascal_part"]:
            masks = []
            for ann in sampled_anns:
                try:
                    masks.append(coco_api.annToMask(ann))
                except Exception as e:
                    print(e)
                    return self.__getitem__(0)

            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        elif ds == "agd20k":
            label = torch.from_numpy(label).long()
            masks = torch.stack([label], dim=0)
        elif ds == "drivelm":
            if multimodal:
                masks = []
                for t in tag:
                    index = f"<{t}>".replace(" ", "")
                    try:
                        cam_id = index.split(",")[1]
                        order = CAM_ORDER.index(cam_id)
                    except Exception as e:
                        print(e)
                    mask_path_list = [None]*6
                    mask_path = os.path.join(self.base_image_dir, "drivelm", info['key_object_infos'][index]['mask_path'])
                    mask_path_list[order] = mask_path
                    mask = combine_images_2x3(mask_path_list, gray=True)
                    masks.append((torch.from_numpy(mask)/255.0).long())
                masks = torch.stack(masks, dim=0)
                label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
            else:
                masks = torch.Tensor(0)
                label = torch.Tensor(0)
        else:
            label = torch.from_numpy(label).long()
            masks = []
            for class_id in class_ids:
                masks.append(label == class_id)
            masks = torch.stack(masks, dim=0)

        # self.count += 1
        # if multimodal:
        #     self.multimodal_count += 1
        # if self.count % 100 == 0:
        #     print(self.multimodal_count/self.count)

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )

import glob
import json
import os
import random
import re

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import DEFAULT_IMAGE_TOKEN, ANSWER_LIST


def init_meccano(base_image_dir, split="train"):
    file_path = os.path.join(base_image_dir, "meccano", "visualizations", "annotations.json")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    ann_file = os.path.join(base_image_dir, "meccano", "MECCANO_active_objects_annotations", f"instances_meccano_{split}_seg.json")
    meccano = COCO(ann_file)
    filename_to_id = {img_info['file_name']: img_id for img_id, img_info in meccano.imgs.items()}

    image_key = [k for k in data.keys()]
    image_names = [k+".jpg" for k in image_key]
    image_ids = [filename_to_id[i] for i in image_names]
    image_dir = os.path.join(base_image_dir, "meccano", "RGB_frames", f"{split.capitalize()}")
    image_paths = []
    for i in image_names:
        prefix = "00" + i[:2]
        image_paths.append(os.path.join(image_dir, prefix, i[3:]))

    ann_ids = meccano.getAnnIds(imgIds=image_ids)
    anns = meccano.loadAnns(ann_ids)
    labels = [maskUtils.decode(ann["segmentation"]) for ann in anns]

    captions = [data[i] for i in image_key]

    print("MECCANO: ", len(image_names))
    return captions, image_paths, labels


class CustomSegDataset(torch.utils.data.Dataset):
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
    ):
        self.samples_per_epoch = samples_per_epoch

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        captions, images, labels = init_meccano(base_image_dir)
        self.data2list = (captions, images, labels)

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
        captions, images, labels = self.data2list
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        label = labels[idx]
        caption_list = captions[idx]

        image = Image.open(image_path)
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        image = self.transform.apply_image(np.array(image))  # preprocess image for sam
        resize = image.shape[:2]

        questions = []
        answers = []

        questions.append(DEFAULT_IMAGE_TOKEN + "\n" + random.choice(caption_list))
        answers.append(random.choice(ANSWER_LIST))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        label = torch.from_numpy(label).long()
        masks = torch.stack([label], dim=0)

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            None,
        )
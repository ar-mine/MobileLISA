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
from transformers import CLIPImageProcessor
import matplotlib.pyplot as plt

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import DEFAULT_IMAGE_TOKEN


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
QA_KEYS = ('perception', 'prediction', 'planning', 'behavior')
def init_drivelm(base_image_dir):
    file_path = os.path.join(base_image_dir, "drivelm", "v1_0_train_nus_grounding.json")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    key_frames = []
    for k, v in data.items():
        key_frames.extend([value for value in v['key_frames'].values()])


    for frame in key_frames:
        frame['multimodal'] = {k: [] for k in QA_KEYS}
        frame['pure_text'] = {k: [] for k in QA_KEYS}
        # frame['QA']['perception'] = [p for p in frame['QA']['perception'] if p['Q'].count('<c') == 0]
        for k in QA_KEYS:
            for i, p in enumerate(frame['QA'][k]):
                if p['Q'].count('<c') == 0 and 4 >= p['A'].count('<c') >= 1:
                    frame['multimodal'][k].append(i)
                if p['Q'].count('<c') == 0 and p['A'].count('<c') == 0:
                    frame['pure_text'][k].append(i)

    print("DriveLm: ", len(key_frames))
    return None, None, key_frames


class DriveLMSegDataset(torch.utils.data.Dataset):
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

        _, _, self.info_list = init_drivelm(base_image_dir)

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
        idx = random.randint(0, len(self.info_list) - 1)
        info = self.info_list[idx]
        image_path_list = [os.path.join(self.base_image_dir, "drivelm", info['image_paths'][cam][3:]) for cam in CAM_ORDER]
        image = combine_images_2x3(image_path_list)

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        image_path = None
        sampled_classes = None

        questions = []
        answers = []

        qa_key = random.choice(QA_KEYS)
        candidates = info['QA'][qa_key]
        if len(info['multimodal'][qa_key]) == 0:
            multimodal = False
            candidate_idx = random.choice(info['pure_text'][qa_key])
        elif len(info['pure_text'][qa_key]) == 0:
            multimodal = True
            candidate_idx = random.choice(info['multimodal'][qa_key])
        else:
            if random.random() < 0.5:
                multimodal = False
                candidate_idx = random.choice(info['pure_text'][qa_key])
            else:
                multimodal = True
                candidate_idx = random.choice(info['multimodal'][qa_key])
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

        conversations = []
        conv = conversation_lib.default_conversation.copy()

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
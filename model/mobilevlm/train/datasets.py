import random
import os
import numpy as np
import copy
import json
import cv2
import tqdm
import transformers
from typing import Dict, Sequence
from dataclasses import dataclass
from PIL import Image
# from shortuuid import random
import torch
from torch.utils.data import Dataset

from mobilevlm.model.segment_anything.utils.transforms import ResizeLongestSide
from mobilevlm.constants import IGNORE_INDEX
from mobilevlm.train.preprocess import (DataArguments, preprocess, preprocess_multimodal,
                                        rank0_print, preprocess_sam)
from mobilevlm.train.data_processing import get_mask_from_json, meta_info_retrieve, seg_add_ref
from mobilevlm.utils import expand2square


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


class ReasonSegDataset(LazySupervisedDataset):
    """Dataset for supervised fine-tuning focuing on segmentation."""

    ignore_label = 255
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 image_processor = None,
                 image_size: int = 1024,):
        super(ReasonSegDataset, self).__init__(data_path, tokenizer, data_args)

        self.transform = ResizeLongestSide(image_size)
        self.num_classes_per_sample = 1
        if image_processor is not None:
            self.image_processor = image_processor
        else:
            self.image_processor = self.data_args.image_processor

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Data to be trained
        data_dict = {}
        # Get source data according to the idx
        sources = self.list_data_dict[i]
        # TODO: What conditions to skip it?
        # if isinstance(i, int):
        #     sources = [sources]
        # assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        # Multimodal input
        if 'image' in sources.keys():
            # Open image file
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            # TODO: Do not transfer instance in this way
            image = cv2.imread(os.path.join(image_folder, image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # If supervised segmentation is avaliable
            if 'meta' in sources.keys():
                json_path = os.path.join(image_folder, sources['meta'])
                masks = get_mask_from_json(json_path, image)
                sampled_indices = sources['sampled_indices']
                sampled_masks = [
                    (masks == 1).astype(np.float32) for _ in range(len(sampled_indices))
                ]
                # TODO: Unify PIL and Opencv
                # Two kinds of preprocess for CLIP and SAM
                image_sam = self.transform.apply_image(image)  # preprocess image for sam
                resize = image_sam.shape[:2]
                image_sam = preprocess_sam(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())

                masks = np.stack(sampled_masks, axis=0)
                masks = torch.from_numpy(masks)
                # TODO: What 'label' is used to do
                label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

                data_dict['image_sam'] = image_sam
                data_dict['label'] = label
                data_dict['masks'] = masks
                data_dict['resize'] = resize

            if self.data_args.image_aspect_ratio == 'pad':
                image_clip = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                image_clip = self.image_processor.preprocess(image_clip, return_tensors='pt')['pixel_values'][0]
            else:
                image_clip = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([sources["conversations"]]), self.data_args)
        else:
            sources = copy.deepcopy([sources["conversations"]])

        ########################
        # if random.random() < 0.2:
        #     sources = [[
        #         {'from': 'human', 'value': random.choice(SHORT_QUESTION_LIST).format(class_name="the object")},
        #         {'from': 'gpt', 'value': random.choice(ANSWER_LIST)}
        #     ]]
        ########################
        data_dict.update({"conversations": [source[0]['value']+source[1]['value'] for source in sources]
                          })
        data_dict.update(preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        )
        if isinstance(i, int):
            data_dict.update(dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0]
            ))

        # image exist in the data
        if 'image' in self.list_data_dict[i].keys():
            data_dict['image_clip'] = image_clip
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.image_processor.crop_size
            data_dict['image_clip'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        # Force to change the raw data to test finetune results

        return data_dict

    # TODO: Modify to match both modalities
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            # cur_len = 1
            length_list.append(cur_len)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # Preprocess segmentation data
            if 'meta' in sample.keys():
                json_path = os.path.join(self.data_args.image_folder, sample['meta'])
                meta_info_retrieve(sample, json_path, self.num_classes_per_sample)
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list


class SegDataset(Dataset):
    """Dataset for pure semantic segmentation."""

    ignore_label = 255
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 image_processor = None,
                 image_size: int = 1024):
        super(SegDataset, self).__init__()
        # Load dataset
        self.data_type = data_args.data_type
        self.data_args = data_args
        if self.data_args.mini_batch:
            data_path = os.path.join(data_args.data_path, "sample.json")
        else:
            data_path = os.path.join(data_args.data_path, "training2.json")
        self.list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")

        # Load process function
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        if image_processor is not None:
            self.image_processor = image_processor
        else:
            self.image_processor = self.data_args.image_processor

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Data to be trained
        data_dict = {}
        # Get source data according to the idx
        sources = self.list_data_dict[i]

        # Multimodal input
        # Open image file
        image = cv2.imread(sources['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get sampled masks
        gt = cv2.imread(sources['annotation'], 0)
        if self.data_type == "ADE":
            gt[gt == 0] = 255
            gt -= 1
            gt[gt == 254] = 255
        sampled_indices = sources['sampled_indices']
        sampled_masks = [
            (gt == label_idx).astype(np.float32) for label_idx in range(len(sampled_indices))
        ]
        masks = np.stack(sampled_masks, axis=0)
        masks = torch.from_numpy(masks)
        sample_length = len(sampled_indices)
        assert sample_length == len(sources["conversations"])//2

        # Preprocess for SAM
        image_sam = self.transform.apply_image(image)  # preprocess image for sam
        resize = image_sam.shape[:2]
        image_sam = preprocess_sam(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
        # Preprocess for CLIP
        if self.data_args.image_aspect_ratio == 'pad':
            image_clip = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
            image_clip = self.image_processor.preprocess(image_clip, return_tensors='pt')['pixel_values'][0]
        else:
            image_clip = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # TODO: What 'label' is used to do
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        conversations = [[sources["conversations"][2*i], sources["conversations"][2*i+1]] for i in range(sample_length)]
        sources = preprocess_multimodal(conversations, self.data_args)
        data_dict.update(preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        )
        if isinstance(i, int):
            data_dict.update(dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0]
            ))
        # repeated_images = image.unsqueeze(0).repeat(5, 1, 1, 1)
        data_dict['conversations'] = conversations
        data_dict['image_sam'] = image_sam
        data_dict['label'] = label
        data_dict['masks'] = masks
        data_dict['resize'] = resize
        data_dict['image_clip'] = image_clip
        return data_dict

    # TODO: Modify to match both modalities
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            # cur_len = 1
            length_list.append(cur_len)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in tqdm.tqdm(self.list_data_dict):
            # sample: {image:, annotation:, }
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            length_list.append(cur_len)
        print("Dataset initialization finish!")
        return length_list


    def __len__(self):
        return len(self.list_data_dict)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, conversations = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "conversations"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        offset = [0]
        cnt = 0
        for instance in instances:
            cnt += len(instance["conversations"])
            offset.append(cnt)
        # if inferences[0] == False:
        # if True:
        #     truncate_len = self.tokenizer.model_max_length - 255
        #     if input_ids.shape[1] > truncate_len:
        #         input_ids = input_ids[:, :truncate_len]
        #         attention_masks = attention_masks[:, :truncate_len]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            offset=offset,
            attention_masks=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'label' in instances[0].keys():
            batch['resize_list'] = [instance['resize'] for instance in instances]
            batch['label_list'] = [instance['label'] for instance in instances]
            batch['masks_list'] = [instance['masks'].float() for instance in instances]
        if 'image_clip' in instances[0].keys():
            images_clip = [instance['image_clip'] for instance in instances]
            if all(x is not None and x.shape == images_clip[0].shape for x in images_clip):
                batch['images_clip'] = torch.stack(images_clip)
            else:
                batch['images_clip'] = images_clip
        if 'image_sam' in instances[0].keys():
            images_sam = [instance['image_sam'] for instance in instances]
            if all(x is not None and x.shape == images_sam[0].shape for x in images_sam):
                batch['images_sam'] = torch.stack(images_sam)
            else:
                batch['images_sam'] = images_sam
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.segment_label:
        train_dataset = SegDataset(tokenizer=tokenizer,
                                   data_args=data_args)
    else:
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

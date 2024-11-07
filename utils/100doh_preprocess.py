# VOC TO ADE20K Format
import os
import sys
import time

from tqdm import tqdm
import xmltodict
import argparse
from multiprocessing import Lock, Manager

import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# 有一个问题，这里这个state index是在哪里用到的，为什么只找到了编号，没找到可以对应到的state index
STATE_IDX = { # 这是一些编号，对这些编号描述语句进行一定的处理，不然那个class的定义过于狭隘
    "0": "background",
    "1": "left hand",
    "2": "right hand",
    "3": "objects touched by the left hand",
    "4": "objects touched by right hand",
    "5": "objects touched by both left and right hands"
}

"""
touched by the left hand：被左手接触
touched by the right hand：被右手接触
held by the left hand：被左手拿着
held by the right hand：被右手拿着
grasped by the left hand：被左手抓住
grasped by the right hand：被右手抓住
supported by the left hand：被左手托住
supported by the right hand：被右手托住
lifted by the left hand：被左手抬起
led by the right hand：被右手抬起
carriifted by the left hand：被左手携带
carried by the right hand：被右手携带
handled by the left hand：被左手处理
handled by the right hand：被右手处理
manipulated by the left hand：被左手操控
manipulated by the right hand：被右手操控
passed to the left hand：传递给左手
passed to the right hand：传递给右手

突然发现如果训练的时候通过介词来表述，好像泛用性更好一些呀
in the left hand
on the left hand
with the left hand
between the fingers
by the right hand
at hand
"""

DEFAULT_COLOR = [
    (0, 0, 0), # Black
    (255, 0, 0), # Red
    (0, 255, 0), # Green
    (0, 0, 255), # Blue
    (255, 255, 0), # Yellow
    (0, 255, 255), # Cyan
]


def parse_args(args):
    parser = argparse.ArgumentParser(description="100DOH dataset format processing.")
    parser.add_argument("--source_dir", type=str, help="path of original dataset")
    parser.add_argument("--target_dir", type=str, help="path to be saved")
    parser.add_argument(
        "--subset",
        default="train",
        type=str,
        choices=["train", "val", "test"],
        help="subset to be processed",
    )
    parser.add_argument("--enable_resume", action="store_true")
    parser.add_argument("--top_k", default=0, type=int)
    parser.add_argument("--sam_checkpoint",
                        default="./pretrained/sam_vit_h_4b8939.pth", type=str)

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    source_dir = args.source_dir
    subset = args.subset

    with open(os.path.join(source_dir, "ImageSets/Main", f"{subset}.txt"), "r") as f:
        subset_lines = f.read().splitlines()

    enable_resume = args.enable_resume
    if enable_resume:
        with open("./checkpoint.txt", "r") as f:
            args.train_lines_buffer = f.read().splitlines()

    top_k = args.top_k
    if 0 < top_k < len(subset_lines)-1:
        samples = subset_lines[:top_k]
    else:
        samples = subset_lines

    sam_checkpoint = args.sam_checkpoint
    model_type = "vit_h"
    available_cuda = range(torch.cuda.device_count())
    # 创建 GPU 锁，每个 GPU 一把锁
    manager = Manager()
    locks = [manager.Lock() for _ in available_cuda]

    os.makedirs(os.path.join(args.target_dir, "images", f"{args.subset}"), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, "annotations", f"{args.subset}"), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, "visualizations", f"{args.subset}"), exist_ok=True)

    results = []
    # 使用线程池并行执行任务
    with (ProcessPoolExecutor(max_workers=2*len(available_cuda)-1) as executor):
        futures = []
        predictors = []
        for cuda_idx in available_cuda:
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=f"cuda:{cuda_idx}")
            predictors.append(SamPredictor(sam))
        # 为每张图像分配一个 GPU 进行处理
        for idx, sample in enumerate(samples):
            gpu_id = available_cuda[idx % len(available_cuda)]  # 轮流分配 GPU
            futures.append(executor.submit(process, locks, predictors, gpu_id, sample, args))

        # tqdm 进度条，监控任务完成进度
        for future in tqdm(as_completed(futures), total=len(samples), desc="Processing images"):
            results.append(future.result())  # 获取结果并收集

    if enable_resume:
        with open("./checkpoint.txt", "a") as f:
            f.writelines(results)
    else:
        with open("./checkpoint.txt", "w") as f:
            f.writelines(results)


def process(locks, predictors, gpu_id, sample, args):
    with locks[gpu_id]:
        predictor = predictors[gpu_id]
        if args.enable_resume and sample in args.train_lines_buffer:
            print(f"{sample} repeat, skip!")
            return
        # 读取 XML 文件并转换为字典
        with open(os.path.join(args.source_dir, "Annotations", f"{sample}.xml"), 'r') as xml_file:
            xml_content = xml_file.read()
            xml_dict = xmltodict.parse(xml_content)
        image = cv2.imread(os.path.join(args.source_dir, "JPEGImages", f"{sample}.jpg"))
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
            torch.cuda.empty_cache()
            mask = masks[0].astype(bool)
            # 这里筛选了一下需要mask的部分，如果annotation已经被打上了手的标签，那么就不会被再次打上mask
            mask_to_update = mask & ((annotation != 1) & (annotation != 2))
            annotation[mask_to_update] = seg_idx
            masked_image[mask_to_update, :] = DEFAULT_COLOR[seg_idx]
        masked_image = (image*0.5 + masked_image*0.5).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.target_dir, "images", f"{args.subset}", f"{sample}.jpg"), image)
        cv2.imwrite(os.path.join(args.target_dir, "annotations", f"{args.subset}", f"{sample}.png"), annotation)
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.target_dir, "visualizations", f"{args.subset}", f"{sample}.jpg"), masked_image)
        return sample+"\n"


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main(sys.argv[1:])

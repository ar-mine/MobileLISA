import os
import logging
import pathlib
from pdb import find_function

import transformers
from typing import Dict
import torch

from mobilevlm.train.preprocess import (ModelArguments, DataArguments, TrainingArguments, rank0_print)
from mobilevlm.train.trainer import VLMTrainer
from mobilevlm.train.datasets import make_supervised_data_module
from mobilevlm import conversation as conversation_lib
from mobilevlm.vis_utils import save_model_structure
from mobilevlm.model.mobilelisa import MobileLisaForCasualLM

global local_rank


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def find_linear_layers(model, lora_target_modules):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
                isinstance(module, cls)
                and all(
            [
                x not in name
                for x in [
                "visual_model",
                "vision_tower",
                "mm_projector",
                "text_hidden_fcs",
            ]
            ]
        )
                and any([x in name for x in lora_target_modules])
        ):
            lora_module_names.add(name)
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return sorted(list(lora_module_names))


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        # weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.state_dict().items(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # RANK for deepspeed
    local_rank = training_args.local_rank
    # Whether using wandb to visualize
    wandb_enable = training_args.wandb_enable
    if wandb_enable:
        import wandb
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="mobileLisa",
            # track hyperparameters and run metadata
            config={
                # "learning_rate": 0.02,
                # "architecture": "CNN",
                # "dataset": "CIFAR-100",
                "epochs": training_args.num_train_epochs,
            }
        )

    # 0. Initialization
    # Add parameters specified for training
    bnb_model_from_pretrained_args = {
        "train_mask_decoder": training_args.train_mask_decoder,
        "cache_dir": training_args.cache_dir,
        "ce_loss_weight": training_args.ce_loss_weight,
        "bce_loss_weight": training_args.bce_loss_weight,
        "dice_loss_weight": training_args.dice_loss_weight,

        "segment_encoder_path": model_args.segment_encoder_path,
        "segment_output_dim": model_args.segment_output_dim,
    }
    # Set model precision
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # 1. Init the whole model
    assert model_args.vision_tower is not None, "Please provide vision_tower used in the model!"
    model = MobileLisaForCasualLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype = compute_dtype,
        **bnb_model_from_pretrained_args
    )
    # Disable mm_projector training
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False
    # Disable cache while training
    model.config.use_cache = False
    # TODO: Why enable these two properties
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # 2. (Optional) Add lora to base model
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        if training_args.train_mask_decoder:
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_linear_layers(model, "q_proj,v_proj".split(",")),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
        else:
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )

        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # 3. Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # Modify according to different version
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 4. Load vision tower
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.vision_tower_type = training_args.vision_tower_type = model_args.vision_tower_type

        if not training_args.lora_enable:
            model.requires_grad_(True)

        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False


        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # 5. Modify [text_hidden_fcs, mask_decoder, lm_head, embed_tokens] trainable
    non_lora_trainable_parameters = []
    for n, p in model.named_parameters():
        if any(
                [
                    x in n
                    for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
                ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
            non_lora_trainable_parameters.append(n)
    print(f"num_non_lora_trainable_parameters: {len(non_lora_trainable_parameters)}")

    # 6. (Optional) Save architecture to ensure model can be trained
    save_model_structure(model)

    # 7. Prepare dataset
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # 8. Train the model
    trainer = VLMTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    # 9. Save model state_dict
    if training_args.lora_enable:
        '''state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        non_lora_state_dict_names = [k for k in non_lora_state_dict.keys()]
        print(f"Save {len(non_lora_state_dict)} non-lora parameters!")'''
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model = model.merge_and_unload()
            model.config.save_pretrained(training_args.output_dir)
            #model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            model.save_pretrained(training_args.output_dir, state_dict=model.state_dict())
            #print('non_lora_trainable...', non_lora_state_dict.keys())
            # torch.save(model.state_dict(), os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            tokenizer.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
  --version="liuhaotian/llava-v1.5-7b" \
  --weight="runs/lisa-7b/pytorch_model.bin" \
  --save_path="../LISA-7B-new"
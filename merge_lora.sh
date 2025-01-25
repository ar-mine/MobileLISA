CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
  --version="mtgv/MobileVLM_V2-1.7B" \
  --weight="runs/mobilelisa-1.7b/pytorch_model.bin" \
  --vision-tower="openai/clip-vit-large-patch14-336" \
  --save_path="./pretrained/MobileLISA-1.7B-ori"
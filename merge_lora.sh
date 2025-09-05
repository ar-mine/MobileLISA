CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
  --version="pretrained/MobileLISA-1.7B-Base" \
  --weight="runs/mobilelisa_deictic/pytorch_model.bin" \
  --vision-tower="openai/clip-vit-large-patch14-336" \
  --save_path="./pretrained/Deictic-1.7B" \
  --lora_r=16
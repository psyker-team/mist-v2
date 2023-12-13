export MODEL_NAME="stable-diffusion/stable-diffusion-1-5"
export INSTANCE_DIR="output/mist/"
export OUTPUT_DIR="output/lora/"
export CLASS_DIR="data/lora_class"

accelerate launch eval/train_dreambooth_lora_15.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR\
  --instance_prompt "an animated painting of a sks person, high quality, masterpiece" \
  --class_prompt "an animated painting of a person, high quality, masterpiece" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --scale_lr \
  --max_train_steps=3000 \
  --mixed_precision=bf16 \
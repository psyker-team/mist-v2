export MODEL_NAME="stable-diffusion/sdxl-base-1.0"
export INSTANCE_DIR="data/training/painting"
export OUTPUT_DIR="output/lora/sdxl/"
export CLASS_DIR="data/class"

accelerate launch eval/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR\
  --instance_prompt "a painting of in the style of sks, high quality, masterpiece" \
  --class_prompt "a painting, high quality, masterpiece" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --scale_lr \
  --max_train_steps=2000 \
  --validation_prompt="a painting of a sks person, high quality, masterpiece" \
  --validation_epochs=25 \
  --checkpointing_steps=500 \
  --mixed_precision=bf16 \
  --gradient_checkpointing
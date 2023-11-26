export MODEL_NAME="stable-diffusion/stable-diffusion-1-5"
export INSTANCE_DIR="output/mist/sd1-5"
export OUTPUT_DIR="output/lora/sd1-5_adv"
export CLASS_DIR="data/class"

accelerate launch eval/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR\
  --instance_prompt "a photo of a sks person, high quality, masterpiece" \
  --class_prompt "a photo of a person, high quality, masterpiece" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --scale_lr \
  --max_train_steps=2000 \
  --validation_prompt="a photo of a sks person, high quality, masterpiece" \
  --validation_epochs=25 \
  --checkpointing_steps=500 \
  --pre_compute_text_embeddings \
  --tokenizer_max_length=77 \
  --text_encoder_use_attention_mask \
  --mixed_precision=bf16
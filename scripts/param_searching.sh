export MODEL_NAME="stable-diffusion/stable-diffusion-1-5"
export INSTANCE_DIR="data/training/kent/"
export OUTPUT_DIR="output/mist/sd1-5/param/5_lunet"
export CLASS_DIR="data/class"
export LORA_MODEL_NAME="stable-diffusion/novelai"
export LORA_OUTPUT_DIR="output/lora/sd1-5_adv/param/5_lunet"
export LORA_CLASS_DIR="data/lora_class"
 

accelerate launch attacks/mist.py \
 --cuda --low_vram_mode --resize \
 --pretrained_model_name_or_path=$MODEL_NAME  \
 --instance_data_dir=$INSTANCE_DIR \
 --output_dir="output/mist/sd1-5/param/5_30_3_antidb" \
 --class_data_dir=$CLASS_DIR\
 --instance_prompt "an animated girl" \
 --class_prompt "an animated girl" \
 --mixed_precision bf16 \
 --max_train_steps 5 \
 --checkpointing_iterations 1 \
 --prior_loss_weight 0.1 \
 --pgd_alpha 0.005 \
 --pgd_eps 0.04 \
 --max_adv_train_steps 30 \
 --max_f_train_steps 3 \
 --mode anti-db

accelerate launch attacks/mist.py \
 --cuda --low_vram_mode --resize \
 --pretrained_model_name_or_path=$MODEL_NAME  \
 --instance_data_dir=$INSTANCE_DIR \
 --output_dir="output/mist/sd1-5/param/5_30_5_antidb" \
 --class_data_dir=$CLASS_DIR\
 --instance_prompt "an animated girl" \
 --class_prompt "an animated girl" \
 --mixed_precision bf16 \
 --max_train_steps 5 \
 --checkpointing_iterations 1 \
 --prior_loss_weight 0.1 \
 --pgd_alpha 0.005 \
 --pgd_eps 0.04 \
 --max_adv_train_steps 30 \
 --max_f_train_steps 5 \
 --mode anti-db


accelerate launch attacks/mist.py \
 --cuda --low_vram_mode --resize \
 --pretrained_model_name_or_path=$MODEL_NAME  \
 --instance_data_dir=$INSTANCE_DIR \
 --output_dir="output/mist/sd1-5/param/5_30_10_antidb" \
 --class_data_dir=$CLASS_DIR\
 --instance_prompt "an animated girl" \
 --class_prompt "an animated girl" \
 --mixed_precision bf16 \
 --max_train_steps 5 \
 --checkpointing_iterations 1 \
 --prior_loss_weight 0.1 \
 --pgd_alpha 0.005 \
 --pgd_eps 0.04 \
 --max_adv_train_steps 30 \
 --max_f_train_steps 10 \
 --mode anti-db

accelerate launch attacks/mist.py \
 --cuda --low_vram_mode --resize \
 --pretrained_model_name_or_path=$MODEL_NAME  \
 --instance_data_dir=$INSTANCE_DIR \
 --output_dir="output/mist/sd1-5/param/5_30_20_antidb" \
 --class_data_dir=$CLASS_DIR\
 --instance_prompt "an animated girl" \
 --class_prompt "an animated girl" \
 --mixed_precision bf16 \
 --max_train_steps 5 \
 --checkpointing_iterations 1 \
 --prior_loss_weight 0.1 \
 --pgd_alpha 0.005 \
 --pgd_eps 0.04 \
 --max_adv_train_steps 30 \
 --max_f_train_steps 20 \
 --mode anti-db

accelerate launch attacks/mist.py \
 --cuda --low_vram_mode --resize \
 --pretrained_model_name_or_path=$MODEL_NAME  \
 --instance_data_dir=$INSTANCE_DIR \
 --output_dir="output/mist/sd1-5/param/5_30_30_antidb" \
 --class_data_dir=$CLASS_DIR\
 --instance_prompt "an animated girl" \
 --class_prompt "an animated girl" \
 --mixed_precision bf16 \
 --max_train_steps 5 \
 --checkpointing_iterations 1 \
 --prior_loss_weight 0.1 \
 --pgd_alpha 0.005 \
 --pgd_eps 0.04 \
 --max_adv_train_steps 30 \
 --max_f_train_steps 30 \
 --mode anti-db

accelerate launch eval/train_dreambooth_lora_15.py \
  --pretrained_model_name_or_path=$LORA_MODEL_NAME  \
  --instance_data_dir="output/mist/sd1-5/param/5_30_5_antidb" \
  --output_dir="output/lora/sd1-5_adv/param/5_30_5_antidb" \
  --class_data_dir=$LORA_CLASS_DIR\
  --instance_prompt "an animated painting of a sks person, high quality, masterpiece" \
  --class_prompt "an animated painting of a person, high quality, masterpiece" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --scale_lr \
  --max_train_steps=2000 \
  --mixed_precision=bf16 \

accelerate launch eval/train_dreambooth_lora_15.py \
  --pretrained_model_name_or_path=$LORA_MODEL_NAME  \
  --instance_data_dir="output/mist/sd1-5/param/5_30_10_antidb" \
  --output_dir="output/lora/sd1-5_adv/param/5_30_10_antidb" \
  --class_data_dir=$LORA_CLASS_DIR\
  --instance_prompt "an animated painting of a sks person, high quality, masterpiece" \
  --class_prompt "an animated painting of a person, high quality, masterpiece" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --scale_lr \
  --max_train_steps=2000 \
  --mixed_precision=bf16 \

accelerate launch eval/train_dreambooth_lora_15.py \
  --pretrained_model_name_or_path=$LORA_MODEL_NAME  \
  --instance_data_dir="output/mist/sd1-5/param/5_30_20_antidb" \
  --output_dir="output/lora/sd1-5_adv/param/5_30_20_antidb" \
  --class_data_dir=$LORA_CLASS_DIR\
  --instance_prompt "an animated painting of a sks person, high quality, masterpiece" \
  --class_prompt "an animated painting of a person, high quality, masterpiece" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --scale_lr \
  --max_train_steps=2000 \
  --mixed_precision=bf16 \

accelerate launch eval/train_dreambooth_lora_15.py \
  --pretrained_model_name_or_path=$LORA_MODEL_NAME  \
  --instance_data_dir="output/mist/sd1-5/param/5_30_30_antidb" \
  --output_dir="output/lora/sd1-5_adv/param/5_30_30_antidb" \
  --class_data_dir=$LORA_CLASS_DIR\
  --instance_prompt "an animated painting of a sks person, high quality, masterpiece" \
  --class_prompt "an animated painting of a person, high quality, masterpiece" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --scale_lr \
  --max_train_steps=2000 \
  --mixed_precision=bf16 \


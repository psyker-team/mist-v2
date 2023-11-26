# mist-v2

This is an updated version of Mist. See our new paper [Understanding and Improving Adversarial Attacks on Latent Diffusion Model](https://arxiv.org/abs/2310.04687) for details and visualization.


## Installation

This repository is based on [PyTorch](https://pytorch.org/). Specifically, please make sure your environment satisfies `python==3.10` and `cuda >= 11.8`. We do not guarantee the code's performance on other versions.

If you're a conda user, simply run:
```bash
conda create -n mist-v2 python=3.10
conda activate mist-v2
```


Then, install the other requirements:
```bash
pip install -r requirements.txt
```

## Usage

To use Mist-v2, directly run:

```bash
python mist-webui.py
```

We also detail the default command in the WebUI:

GPU:
```bash
accelerate launch attacks/ita_dev.py --cuda --low_vram_mode --instance_data_dir_for_adversarial data/training --output_dir output/ --class_data_dir data/class --instance_prompt "a photo of a sks person, high quality, masterpiece" --class_prompt "a painting, high quality, masterpiece" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
```

CPU:
```bash
accelerate launch --cpu attacks/ita.py --instance_data_dir_for_adversarial data/training --output_dir output/ --class_data_dir data/class --instance_prompt "a photo of a sks person, high quality, masterpiece" --class_prompt "a painting, high quality, masterpiece" --mixed_precision bf16 --max_train_steps 3 --checkpointing_iterations 1
```

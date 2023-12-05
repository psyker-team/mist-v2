import argparse
import copy
import hashlib
import itertools
import logging
import os
import sys
import time, datetime, gc
from pathlib import Path
import warnings
import math, shutil
from packaging import version
from colorama import Fore, Style, init,Back
'''some system level settings'''
init(autoreset=True)
sys.path.insert(0, sys.path[0]+"/../")

import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder


from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from torch import autograd
import pynvml
pynvml.nvmlInit()

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionXLPipeline
)
from diffusers.loaders import (
    LoraLoaderMixin,
    text_encoder_lora_state_dict,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import unet_lora_state_dict
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from attacks.utils import LatentAttack

logger = get_logger(__name__)

class DreamBoothDatasetFromTensor(Dataset):
    """Just like DreamBoothDataset, but take instance_images_tensor instead of path"""

    def __init__(
            self,
        instance_images_tensor,
        instance_prompt,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        self.center_crop = center_crop
        self.encoder_hidden_states = encoder_hidden_states

        self.instance_images_tensor = instance_images_tensor
        self.num_instance_images = len(self.instance_images_tensor)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images


        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images_tensor[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["instance_prompt"] = self.instance_prompt

        return example
    
def collate_fn(examples):
    has_attention_mask = "instance_attention_mask" in examples[0]

    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]
        batch["attention_mask"] = attention_mask

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    tokens = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    text_inputs, attn_mask = tokens["input_ids"], tokens["attention_mask"]
    return text_inputs, attn_mask


def encode_prompt_sd(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

def encode_prompt_sdxl(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, 
        revision: str,
        subfolder: str = "text_encoder"
    ):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['sdxl', 'sd'],
        default='sd',
        help="sd or sdxl?",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use gpu for attack",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='stable-diffusion/stable-diffusion-1-5/',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )

    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=1, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    # following are the training args of adv attacks
    parser.add_argument(
        "--max_f_train_steps",
        type=int,
        default=5,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--low_vram_mode",
        action="store_true",
        help="Whether or not to use low vram mode.",
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=5e-3,
        help="The step size for pgd.",
    )
    parser.add_argument(
        "--pgd_eps",
        type=float,
        default=float(8.0/255.0),
        help="The noise budget for pgd.",
    )
    parser.add_argument(
        "--fused_weight",
        type=float,
        default=1e-5,
        help="The decay of alpha and eps when applying pre_attack",
    )
    parser.add_argument(
        "--target_image_path",
        default="data/MIST.png",
        help="target image for attacking",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=['lunet','fused'],
        default='lunet',
        help="The mode of attack",
    )
    parser.add_argument(
        "--max_adv_train_steps",
        type=int,
        default=50,
        help="Total number of sub-steps to train adversarial noise.",
    )
    parser.add_argument(
        "--pre_attack_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train adversarial noise.",
    )

    # for resizing mode
    parser.add_argument(
        "--original_resolution",
        action="store_true",
        help="Whether or not to use original resolution resizing.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    if args.train_text_encoder and args.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")

    return args


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def load_data(data_dir, size=512, center_crop=True) -> torch.Tensor:
    image_transforms = transforms.Compose(
        [
            transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    images = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            file_path = os.path.join(data_dir, filename)
            images.append(Image.open(file_path).convert("RGB"))

    sizes = [img.size for img in images]
    images = [image_transforms(img) for img in images]
    images = torch.stack(images)
    return images, sizes

def prepare_dataset(
        args,
        data_tensor,
    ):

    train_dataset = DreamBoothDatasetFromTensor(
        instance_images_tensor=data_tensor,
        instance_prompt=args.instance_prompt,
        size=args.resolution,
        center_crop=args.center_crop,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    ) 

    return train_dataloader

def prepare_lora(
        args,
        accelerator,
        models, 
        weight_dtype
    ):
    
    if args.model_type == 'sd':
        unet, text_encoder = models
        unet.to(accelerator.device, dtype=weight_dtype)
        if not args.pre_compute_text_embeddings and args.train_text_encoder:
            text_encoder.to(accelerator.device, dtype=weight_dtype)
    else:
        unet, text_encoder_one, text_encoder_two = models
        unet.to(accelerator.device, dtype=weight_dtype)
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            if args.model_type == 'sd':
                text_encoder.gradient_checkpointing_enable()
            else:
                text_encoder_one.gradient_checkpointing_enable()
                text_encoder_two.gradient_checkpointing_enable()
    
    unet_lora_parameters = []
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_q.in_features, out_features=attn_module.to_q.out_features, rank=args.rank
            )
        )
        attn_module.to_k.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_k.in_features, out_features=attn_module.to_k.out_features, rank=args.rank
            )
        )
        attn_module.to_v.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_v.in_features, out_features=attn_module.to_v.out_features, rank=args.rank
            )
        )
        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                rank=args.rank,
            )
        )

        # Accumulate the LoRA params to optimize.
        unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            attn_module.add_k_proj.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.add_k_proj.in_features,
                    out_features=attn_module.add_k_proj.out_features,
                    rank=args.rank,
                )
            )
            attn_module.add_v_proj.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.add_v_proj.in_features,
                    out_features=attn_module.add_v_proj.out_features,
                    rank=args.rank,
                )
            )
            unet_lora_parameters.extend(attn_module.add_k_proj.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.add_v_proj.lora_layer.parameters())

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    text_lora_parameters, text_lora_parameters_one, text_lora_parameters_two= [], [], []
    if args.train_text_encoder:
        if args.model_type == 'sd':
            # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
            text_lora_parameters = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=torch.float32, rank=args.rank)
        else:
            text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(
                text_encoder_one, dtype=torch.float32, rank=args.rank
            )
            text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(
                text_encoder_two, dtype=torch.float32, rank=args.rank
            )

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None

            if args.model_type == 'sd':
                text_encoder_lora_layers_to_save = None

                for model in models:
                    if isinstance(model, type(accelerator.unwrap_model(unet))):
                        unet_lora_layers_to_save = unet_lora_state_dict(model)
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                        text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(model)
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

                LoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_lora_layers_to_save,
                )
            else:
                text_encoder_one_lora_layers_to_save = None
                text_encoder_two_lora_layers_to_save = None

                for model in models:
                    if isinstance(model, type(accelerator.unwrap_model(unet))):
                        unet_lora_layers_to_save = unet_lora_state_dict(model)
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                        text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(model)
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                        text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(model)
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

                StableDiffusionXLPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                    text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
                )

    def load_model_hook(models, input_dir):
        unet_ = None
        if args.model_type == 'sd':
            text_encoder_ = None

            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_ = model
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                    text_encoder_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
            LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)
            LoraLoaderMixin.load_lora_into_text_encoder(
                lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_
            )
        else:
            text_encoder_one_ = None
            text_encoder_two_ = None

            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_ = model
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                    text_encoder_one_ = model
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                    text_encoder_two_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
            LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)

            text_encoder_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder." in k}
            LoraLoaderMixin.load_lora_into_text_encoder(
                text_encoder_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_one_
            )

            text_encoder_2_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k}
            LoraLoaderMixin.load_lora_into_text_encoder(
                text_encoder_2_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_two_
            )


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.model_type == 'sd':
        return (unet_lora_parameters, text_lora_parameters) 
    else:
        return (unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two)


def prepare_optimizer(
        args,
        accelerator,
        unet_lora_parameters,
        text_lora_parameters_one,
        text_lora_parameters_two=None,
    ):
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet_lora_parameters, text_lora_parameters_one)
        if args.train_text_encoder
        else unet_lora_parameters
    )
    unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": args.learning_rate}
    if args.train_text_encoder:
        if args.model_type == 'sd':
            text_lora_parameters_with_lr = {
                "params": text_lora_parameters_one,
                "weight_decay": args.adam_weight_decay_text_encoder,
                "lr":  args.learning_rate,
            }
            params_to_optimize = [
                unet_lora_parameters_with_lr,
                text_lora_parameters_with_lr,
            ]
        else:
            # different learning rate for text encoder and unet
            text_lora_parameters_one_with_lr = {
                "params": text_lora_parameters_one,
                "weight_decay": args.adam_weight_decay_text_encoder,
                "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
            }
            text_lora_parameters_two_with_lr = {
                "params": text_lora_parameters_two,
                "weight_decay": args.adam_weight_decay_text_encoder,
                "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
            }
            params_to_optimize = [
                unet_lora_parameters_with_lr,
                text_lora_parameters_one_with_lr,
                text_lora_parameters_two_with_lr,
            ]
    else:
        params_to_optimize = [unet_lora_parameters_with_lr]
    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    return optimizer, lr_scheduler

def prepare_prompts(
        args,
        accelerator,
        text_encoders,
        tokenizers,
        weight_dtype,
    ):
    tokens_one = None
    tokens_two = None
    add_time_ids = None
    prompt_embeds = None
    unet_add_text_embeds = None
    if args.model_type == 'sd':
        text_encoder = text_encoders
        tokenizer = tokenizers

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs, attn_mask = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
                prompt_embeds = encode_prompt_sd(
                    text_encoder,
                    text_inputs,
                    attn_mask,
                    text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                )
            return prompt_embeds
        
        prompt_embeds = compute_text_embeddings(args.instance_prompt)
    else:       
        def compute_time_ids():
            # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
            original_size = (args.resolution, args.resolution)
            target_size = (args.resolution, args.resolution)
            crops_coords_top_left = (0, 0)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
            return add_time_ids
        
        add_time_ids = compute_time_ids()
    
        tokenizer_one, tokenizer_two = tokenizers
        if not args.train_text_encoder:
            def compute_text_embeddings(prompt, text_encoders, tokenizers):
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = encode_prompt_sdxl(text_encoders, tokenizers, prompt)
                    prompt_embeds = prompt_embeds.to(accelerator.device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                return prompt_embeds, pooled_prompt_embeds

            prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                args.instance_prompt, text_encoders, tokenizers
            )
            del tokenizers, text_encoders
        else:
            tokens_one, _ = tokenize_prompt(tokenizer_one, args.instance_prompt)
            tokens_two, _ = tokenize_prompt(tokenizer_two, args.instance_prompt)

    gc.collect()
    torch.cuda.empty_cache()
    
    return (tokens_one, tokens_two, add_time_ids, prompt_embeds, unet_add_text_embeds)


def train_one_epoch(
        args,
        accelerator,
        models,
        noise_scheduler,
        vae,
        data_tensor: torch.Tensor,
        prompts,
        weight_dtype=torch.float16,
    ):

    # prepare training data
    tokens_one, tokens_two, add_time_ids, prompt_embeds, unet_add_text_embeds = prompts
    train_dataloader = prepare_dataset(args, data_tensor)
    
    if args.model_type == 'sd':
        # prepare lora
        lora_parameters = prepare_lora(args, accelerator, models, weight_dtype)
        unet_lora_parameters, text_lora_parameters = lora_parameters
        # prepare optimizer
        optimizer, lr_scheduler = prepare_optimizer(args, accelerator, unet_lora_parameters, text_lora_parameters)
    else:
        # prepare lora
        lora_parameters = prepare_lora(args, accelerator, models, weight_dtype)
        unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two = lora_parameters
        # prepare optimizer
        optimizer, lr_scheduler = prepare_optimizer(args, accelerator, unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        if args.model_type == 'sd':
            unet, text_encoder = models
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, text_encoder_one, text_encoder_two = models
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
    else:
        if args.model_type == 'sd':
            unet, text_encoder = models
        else:
            unet, text_encoder_one, text_encoder_two = models
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers("dreambooth-lora", config=tracker_config)

    # Train!
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            if args.model_type == 'sd':
                text_encoder.train()
            else:
                text_encoder_one.train()
                text_encoder_two.train()
                text_encoder_one.text_model.embeddings.requires_grad_(True)
                text_encoder_two.text_model.embeddings.requires_grad_(True)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                if vae is not None:
                    # Convert images to latent space
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                else:
                    model_input = pixel_values

                if args.model_type == 'sd':
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz, channels, height, width = model_input.shape
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = prompt_embeds.to(device=model_input.device, dtype=weight_dtype)

                    if accelerator.unwrap_model(unet).config.in_channels == channels * 2:
                        noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                    if args.class_labels_conditioning == "timesteps":
                        class_labels = timesteps
                    else:
                        class_labels = None
                    # Predict the noise residual
                    model_pred = unet(
                        noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels
                    ).sample

                    # if model predicts variance, throw away the prediction. we will only train on the
                    # simplified training objective. This means that all schedulers using the fine tuned
                    # model must be configured to use one of the fixed variance variance types.
                    if model_pred.shape[1] == 6:
                        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                else:
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                    elems_to_repeat_text_embeds =  bsz
                    elems_to_repeat_time_ids = bsz
                    add_time_ids, unet_add_text_embeds, prompt_embeds = \
                        add_time_ids.to(dtype=weight_dtype), unet_add_text_embeds.to(dtype=weight_dtype), prompt_embeds.to(dtype=weight_dtype)

                    # Predict the noise residual
                    if not args.train_text_encoder:
                        unet_added_conditions = {
                            "time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1),
                            "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
                        }
                        prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                        model_pred = unet(
                            noisy_model_input,
                            timesteps,
                            prompt_embeds_input,
                            added_cond_kwargs=unet_added_conditions,
                        ).sample
                    else:
                        unet_added_conditions = {"time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1)}
                        prompt_embeds, pooled_prompt_embeds = encode_prompt_sdxl(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=None,
                            prompt=None,
                            text_input_ids_list=[tokens_one, tokens_two],
                        )
                        unet_added_conditions.update(
                            {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                        )
                        prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                        model_pred = unet(
                            noisy_model_input, timesteps, prompt_embeds_input, added_cond_kwargs=unet_added_conditions
                        ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet_lora_parameters, text_lora_parameters)
                        if args.train_text_encoder
                        else unet_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    
    return [unet, text_encoder] if args.model_type == 'sd' else [unet, text_encoder_one, text_encoder_two]



def pgd_attack(
    args,
    models,
    noise_scheduler:DPMSolverMultistepScheduler,
    vae:AutoencoderKL,
    data_tensor: torch.Tensor,
    original_images: torch.Tensor,
    target_tensor: torch.Tensor,
    prompts,
    device: Accelerator.device,
    mode: str = 'fused',
    weight_dtype: torch.dtype = torch.float16
):
    """Return new perturbed data"""
    num_steps = args.max_adv_train_steps
    target_tensor = copy.deepcopy(target_tensor)
    target_tensor = target_tensor.to(device, dtype=weight_dtype)
    tokens_one, tokens_two, add_time_ids, prompt_embeds, unet_add_text_embeds = prompts
    
    # prepare models
    vae.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    if args.model_type == 'sd':
        unet, text_encoder = models
        text_encoder.to(device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
    else:
        unet, text_encoder_one, text_encoder_two = models
        text_encoder_one.to(device, dtype=weight_dtype)
        text_encoder_two.to(device, dtype=weight_dtype)
    
    unet.to(device, dtype=weight_dtype)
    if args.enable_xformers_memory_efficient_attention:
        unet.set_use_memory_efficient_attention_xformers(True)
    unet.requires_grad_(False)
    
    num_image = len(data_tensor)
    image_list = []
    tbar = tqdm(range(num_image))
    tbar.set_description("PGD attack")
    for id in range(num_image):
        tbar.update(1)
        perturbed_image = data_tensor[id, :].unsqueeze(0)
        perturbed_image.requires_grad = True
        original_image = original_images[id, :].unsqueeze(0)
        for step in range(num_steps):
            perturbed_image.requires_grad = False
            with torch.no_grad():
                latents = vae.encode(perturbed_image.to(device, dtype=weight_dtype)).latent_dist.mean
            #offload vae
            latents = latents.detach().clone()
            latents.requires_grad = True
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents, dtype=weight_dtype)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if args.model_type == 'sd':
                # Get the text embedding for conditioning
                encoder_hidden_states = prompt_embeds.to(device, dtype=weight_dtype)
              
                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                text_encoder.zero_grad()
            else:
                elems_to_repeat_text_embeds =  bsz
                elems_to_repeat_time_ids = bsz
                add_time_ids, unet_add_text_embeds, prompt_embeds = \
                        add_time_ids.to(dtype=weight_dtype), unet_add_text_embeds.to(dtype=weight_dtype), prompt_embeds.to(dtype=weight_dtype)

                # Predict the noise residual
                if not args.train_text_encoder:
                    unet_added_conditions = {
                        "time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1),
                        "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
                    }
                    # print(add_time_ids.device, add_time_ids.dtype, unet_add_text_embeds.device, unet_add_text_embeds.dtype, prompt_embeds.device, prompt_embeds.dtype)
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                    ).sample
                else:
                    unet_added_conditions = {"time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1)}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt_sdxl(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two],
                    )
                    unet_added_conditions.update(
                        {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                    )
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(
                        noisy_latents, timesteps, prompt_embeds_input, added_cond_kwargs=unet_added_conditions
                    ).sample
                    text_encoder_one.zero_grad()
                    text_encoder_two.zero_grad()

            unet.zero_grad()
            loss = None
            # target-shift loss
            if target_tensor is not None:
                loss = - F.mse_loss(model_pred.float(), target_tensor.float())
                # fused mode
                if mode == 'fused':
                    latent_attack = LatentAttack()
                    loss = loss - 1e2 * latent_attack(latents.float(), target_tensor=target_tensor.float())
            loss = loss / args.gradient_accumulation_steps
            # print(model_pred.dtype, loss.dtype, latents.dtype, unet.dtype)
            grads = autograd.grad(loss, latents)[0].detach().clone()
            model_pred.detach_()
            torch.cuda.empty_cache()
            gc.collect()
            # now loss is backproped to latents

            #do forward on vae again
            perturbed_image.requires_grad = True
            gc_latents = vae.encode(perturbed_image.to(device, dtype=weight_dtype)).latent_dist.mean
            gc_latents.backward(gradient=grads)
            assert perturbed_image.grad != None
            alpha = args.pgd_alpha
            eps = args.pgd_eps
            if step % args.gradient_accumulation_steps == args.gradient_accumulation_steps - 1:
                adv_images = perturbed_image + alpha * perturbed_image.grad.sign()
                eta = torch.clamp(adv_images - original_image, min=-eps, max=+eps)
                perturbed_image = torch.clamp(original_image + eta, min=-1, max=+1).detach_()
                perturbed_image.requires_grad = True
            #print(f"PGD loss - step {step}, loss: {loss.detach().item()}")

        image_list.append(perturbed_image.detach().clone().squeeze(0))
    outputs = torch.stack(image_list)

    return outputs
    
def main(args):
    start_time = time.time()
    # check computational resources        
    if args.cuda:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_free = mem_info.free  / float(1073741824)
            if mem_free < 12.0 and not args.low_vram_mode:
                raise NotImplementedError("Your GPU memory is not enough for normal mode. Please try low VRAM mode.")
            if mem_free < 5.5:
                raise NotImplementedError("Your GPU memory is not enough for running Mist on GPU. Please try CPU mode.")
        except:
            raise NotImplementedError("No GPU found in GPU mode. Please try CPU mode.")
    elif args.low_vram_mode:
        raise NotImplementedError("Low VRAM mode needs to run on GPUs. No GPU found!")


    logging_dir = Path(args.output_dir, args.logging_dir)

    if not args.cuda:
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_dir=logging_dir,
            cpu=True
        )
    else:
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_dir=logging_dir
        )

    device = accelerator.device
    print("Mist will run on {}".format(device.type))

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (sayakpaul): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.model_type == 'sd':
        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
        elif args.pretrained_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=args.revision,
                use_fast=False,
            )

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

        # Load scheduler and models
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
    else:
        tokenizer_one = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
            use_fast=False,
        )

        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
        )
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
        )
        # print(type(text_encoder_one), type(text_encoder_two)) 

    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
    except OSError:
        # IF does not have a VAE so let's just set it to None
        # We don't have to error out here
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # We only train the additional adapter LoRA layers
    if vae is not None:
        vae.requires_grad_(False)
        vae.encoder.training = True
        vae.encoder.gradient_checkpointing = True
    if args.model_type == 'sd':
        text_encoder.requires_grad_(False)
    else:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.cuda:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    print("==precision: {}==".format(weight_dtype))

    # 1.26
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)

    # load data 
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    perturbed_data, data_sizes = load_data(
        args.instance_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    original_data = perturbed_data.clone()
    original_data.requires_grad_(False)

    target_latent_tensor = None
    if args.target_image_path is not None and args.target_image_path != "":
        print(Style.BRIGHT+Back.BLUE+Fore.GREEN+'load target image from {}'.format(args.target_image_path))
        target_image_path = Path(args.target_image_path)
        assert target_image_path.is_file(), f"Target image path {target_image_path} does not exist"

        target_image = Image.open(target_image_path).convert("RGB").resize((args.resolution, args.resolution))
        target_image = np.array(target_image)[None].transpose(0, 3, 1, 2)

        target_image_tensor = torch.from_numpy(target_image).to(device=device, dtype=weight_dtype) / 127.5 - 1.0
        target_latent_tensor = (
            vae.encode(target_image_tensor).latent_dist.sample().to(weight_dtype) * vae.config.scaling_factor
        ).detach().clone()
        target_image_tensor = target_image_tensor.to('cpu')
        del target_image_tensor
        gc.collect()
        torch.cuda.empty_cache()
    if args.model_type == 'sd':
        prompts = prepare_prompts(args, accelerator, text_encoder, tokenizer, weight_dtype)
        f = [unet, text_encoder]
    else:
        prompts = prepare_prompts(args, accelerator, [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two], weight_dtype)
        f = [unet, text_encoder_one, text_encoder_two]
    print("==Pre computed prompts: {}".format(args.instance_prompt))
    print("==Device: vae: {}, unet: {}, data: {}".format(vae.device, unet.device, perturbed_data.device, ))
    
    
    for i in range(args.max_f_train_steps):       
        f_sur = copy.deepcopy(f)
        perturbed_data = perturbed_data.detach().clone()
        perturbed_data = pgd_attack(
            args,
            f_sur,
            noise_scheduler,
            vae,
            perturbed_data,
            original_data,
            target_latent_tensor,
            prompts,
            device,
            args.mode,
            weight_dtype
        )
        del f_sur
        gc.collect()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print("=======mem after pgd: {}=======".format(mem_info.used / float(1073741824)))
        f = train_one_epoch(
            args,
            accelerator,
            f,
            noise_scheduler,
            vae,
            perturbed_data,
            prompts,
            weight_dtype=weight_dtype,
        )
        gc.collect()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print("=======mem after lora: {}======".format(mem_info.used / float(1073741824)))
        

        for model in f:
            if model != None:
                model.to('cpu')

        if i + 1 == args.max_f_train_steps:
            save_folder = f"{args.output_dir}"
            os.makedirs(save_folder, exist_ok=True)
            noised_imgs = perturbed_data.detach().cpu()
            origin_imgs = original_data.detach().cpu()
            img_names = []
            for filename in os.listdir(args.instance_data_dir):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    img_names.append(str(filename))
            for img_pixel, ori_img_pixel, img_name, img_size in zip(noised_imgs, origin_imgs, img_names, data_sizes):
                save_path = os.path.join(save_folder, f"{i+1}_noise_{img_name}")
                if not args.original_resolution:
                    Image.fromarray(
                        (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    ).resize(img_size).save(save_path)
                else:
                    ori_img_path = os.path.join(args.instance_data_dir, img_name)
                    ori_img = np.array(Image.open(ori_img_path).convert("RGB"))

                    ori_img_duzzy = np.array(Image.fromarray(
                        (ori_img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    ).resize(img_size), dtype=np.int32)
                    perturbed_img_duzzy = np.array(Image.fromarray(
                        (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    ).resize(img_size), dtype=np.int32)
                    
                    perturbation = perturbed_img_duzzy - ori_img_duzzy
                    assert perturbation.shape == ori_img.shape

                    perturbed_img =  (ori_img + perturbation).clip(0, 255).astype(np.uint8)
                    # print("perturbation: {}, ori: {}, res: {}".format(
                    #     perturbed_img_duzzy[:2, :2, :], ori_img_duzzy[:2, :2, :], perturbed_img_duzzy[:2, :2, :]))
                    Image.fromarray(perturbed_img).save(save_path)


                print(f"==Saved misted image to {save_path}, size: {img_size}==")
            # print(f"Saved noise at step {i+1} to {save_folder}")
            del noised_imgs
        gc.collect()
    end_time = time.time()
    running_time = str(datetime.timedelta(seconds = end_time - start_time))
    print("Finished! Running time: {}".format(running_time))


def update_args_with_config(args, config):
    '''
        Update the default augments in args with config assigned by users
        args list:
            eps: 
            max train epoch:
            data path:
            class path:
            output path:
            device: 
                gpu normal,
                gpu low vram,
                cpu,
            mode:
                lunet, full
    '''

    args = parse_args()
    eps, device, mode, model_type, original_resolution, data_path, output_path, model_path, \
              prompt, max_f_train_steps, max_train_steps, max_adv_train_steps, lora_lr, pgd_lr, rank = config
    args.pgd_eps = float(eps)/255.0
    if device == 'cpu':
        args.cuda, args.low_vram_mode = False, False
    else:
        args.cuda, args.low_vram_mode = True, True
    # if precision == 'bfloat16':
    #     args.mixed_precision = 'bf16'
    # else:
    #     args.mixed_precision = 'fp16'
    if mode == 'Mode 1':
        args.mode = 'lunet'
    else:
        args.mode = 'fused'
    if model_type == 'Stable Diffusion':
        args.model_type = 'sd'
        args.resolution = 512
    else:
        args.model_type = 'sdxl'
        args.resolution = 1024
    if original_resolution:
        args.original_resolution = True

    assert os.path.exists(data_path) and os.path.exists(output_path)
    args.instance_data_dir = data_path
    args.output_dir = output_path
    args.pretrained_model_name_or_path = model_path
    args.instance_prompt = prompt
    args.max_f_train_steps = max_f_train_steps
    args.max_train_steps = max_train_steps
    args.max_adv_train_steps = max_adv_train_steps
    args.learning_rate = lora_lr
    args.pgd_alpha = pgd_lr
    args.rank = rank

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

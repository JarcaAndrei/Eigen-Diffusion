# Adapted from PixArt-alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/master/train_scripts/train_pixart_lora_hf.py
import argparse
import io
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Union

import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk, DatasetDict
from packaging import version
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model
from torchvision import transforms
from tqdm.auto import tqdm
import gc
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    Transformer2DModel,
    DiffusionPipeline
)
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version
from get_strategy_pixart import get_strategy_indexes

# If the PixArt weights require authentication, log in via the HF CLI
# (`huggingface-cli login`) or the HF_TOKEN environment variable rather than
# hardcoding a token here.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# Pick-a-Pic columns: (preferred_img, unpreferred_img, label, caption)
DATASET_NAME_MAPPING = {
    "yuvalkirstain/pickapic_v1": ("jpg_0", "jpg_1", "label_0", "caption"),
    "yuvalkirstain/pickapic_v2": ("jpg_0", "jpg_1", "label_0", "caption"),
    "pickapic-anonymous/pickapic_v1": ("jpg_0", "jpg_1", "label_0", "caption"),
    "rapidata": ("jpg_0", "jpg_1", "label_0", "caption"),
    "hpdv2": ("jpg_0", "jpg_1", "label_0", "caption"),
    "roco": ("jpg_0", "jpg_1", "label_0", "caption"),
}



def parse_args():
    parser = argparse.ArgumentParser(description="PixArt-Alpha DPO/SFT fine-tuning with LoRA.")

    # Model & data paths
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="HuggingFace hub ID or local path to PixArt-Alpha model.")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None,
                        help="Model variant, e.g. 'fp16'.")
    parser.add_argument("--transformer_init", type=str, default="",
                        help="Optional: init transformer from a different checkpoint than pretrained_model_name_or_path.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="HuggingFace dataset name (e.g. yuvalkirstain/pickapic_v2) or local path.")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--image_column", type=str, default="jpg_0",
                        help="Column name for images (SFT non-pickapic only).")
    parser.add_argument("--caption_column", type=str, default="caption")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="pixart-dpo-lora")
    parser.add_argument("--cache_dir", type=str, default=None)

    # Training hyper-parameters
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--random_crop", default=False, action="store_true")
    parser.add_argument("--no_hflip", action="store_true", help="Suppress random horizontal flip.")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Per-device batch size. Each sample is one preference *pair* in DPO mode.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=2000,
                        help="Overrides num_train_epochs if set.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--snr_gamma", type=float, default=None,
                        help="SNR weighting gamma (Section 3.4 of arxiv:2303.09556). Only used for SFT.")
    parser.add_argument("--noise_offset", type=float, default=0)
    parser.add_argument("--input_perturbation", type=float, default=0,
                        help="Scale of input perturbation noise. Recommended 0.1.")
    parser.add_argument("--prediction_type", type=str, default=None,
                        help="Override scheduler prediction type: 'epsilon' or 'v_prediction'.")

    # Optimizer
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")

    # LoRA
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--use_dora", action="store_true", default=False)
    parser.add_argument("--use_rslora", action="store_true", default=False)

    # DPO-specific
    parser.add_argument("--sft", action="store_true",
                        help="Run Supervised Fine-Tuning (MSE) instead of DPO.")
    parser.add_argument("--beta_dpo", type=float, default=5000,
                        help="DPO temperature β controlling KL-penalty strength. "
                             "Higher = less deviation from reference policy.")
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.2,
                        help="Fraction of prompts replaced with empty string (classifier-free guidance support).")
    parser.add_argument("--hard_skip_resume", action="store_true",
                        help="Load checkpoint weights but skip iterating through loader to resume offset.")
    parser.add_argument("--choice_model", type=str, default="",
                        help="AI feedback model to rank pairs instead of dataset labels. "
                             "Choices: hps, clip, pickscore, aes.")
    parser.add_argument("--dreamlike_pairs_only", action="store_true",
                        help="Only train on Pick-a-Pic pairs where both images are from dreamlike model.")

    # Infrastructure
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="pixart-dpo")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default="latest")

    parser.add_argument("--validation_epochs", type=int, default=100,
                        help="Run inference samples every N optimizer steps (0 to disable).")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="Number of denoising steps for validation inference.")
    parser.add_argument("--validation_prompt", type=str, default=" Computerized tomography scan prior to wound washout. The scan reveals mild extra and intracranial fluid collection along the craniotomy with diffuse intracranial leptomeningeal enhancement\n",
                        help="Number of denoising steps for validation inference.")
    parser.add_argument("--validation_images", type=int, default=5,
                        help="Number of denoising steps for validation inference.")
    parser.add_argument("--first_eigen", type=int, default=1024,
                        help="Stage-1 anchor budget (n_1).")
    parser.add_argument("--bin_count", type=int, default=4,
                        help="Number of quantile bands in Stage 2.")
    parser.add_argument("--ratio", type=float, default=6,
                        help="Stage-2/Stage-1 budget ratio (n_2 = ratio * n_1).")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="tau1: cumulative-variance end of the Stage-1 band.")
    parser.add_argument("--thresh_stage2", type=float, default=0.8,
                        help="tau2: cumulative-variance end of the Stage-2 band.")
    parser.add_argument(
        "--no_var_weight",
        action="store_false",
        dest="variance_weighted",
        help="Disable variance-weighted PC allocation."
    )
    parser.set_defaults(variance_weighted=True)

    parser.add_argument(
        "--no_stratify",
        action="store_false",
        dest="stratify",
        help="Disable stratification in stage 2."
    )
    parser.set_defaults(stratify=True)
    parser.add_argument(
        "--random", action="store_true",
        help="Random-selection ablation: select the same number of samples "
             "(first_eigen * (ratio + 1)) uniformly at random instead of the "
             "eigen selection."
    )
    parser.add_argument(
        "--strategy", type=str, default="eigen",
        choices=["eigen", "kcenter", "facility_location", "kmeans++", "semdedup"],
        help="Coreset baseline that replaces the eigen selection, with the same budget."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full-scale baseline: train on the entire (tie-filtered) set."
    )
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either --dataset_name or --train_data_dir.")
    if not 0 <= args.proportion_empty_prompts <= 1:
        raise ValueError("--proportion_empty_prompts must be in [0, 1].")

    args.train_method = "sft" if args.sft else "dpo"
    if args.choice_model:
        raise NotImplementedError(
            "--choice_model (AI-feedback relabeling) requires a utils/ Selector package "
            "that is not included in this release.")
    return args
 

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    max_length = 120

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",
    )
    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device)
    #text_encoder.to("cpu")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    vae.requires_grad_(False)
    vae.to(accelerator.device)

    transformer_path = args.transformer_init or args.pretrained_model_name_or_path
    transformer_subfolder = None if args.transformer_init else "transformer"

    if args.train_method == "dpo":
        ref_transformer = Transformer2DModel.from_pretrained(
            transformer_path,
            subfolder=transformer_subfolder,
            torch_dtype=weight_dtype,
        )
        ref_transformer.requires_grad_(False)
        ref_transformer.to(accelerator.device)
        ref_transformer.eval()


    transformer = Transformer2DModel.from_pretrained(
        transformer_path,
        subfolder=transformer_subfolder,
        torch_dtype=weight_dtype,
    )
    transformer.requires_grad_(False)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,  # scale = alpha/r = 1.0, matches previous SD1.5 config
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.01,
        bias="none",
        use_dora=args.use_dora,
        use_rslora=args.use_rslora,
    )
    transformer.to(accelerator.device)
    transformer = get_peft_model(transformer, lora_config)

    def cast_training_params(
        model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32
    ):
        if not isinstance(model, list):
            model = [model]
        for m in model:
            for param in m.parameters():
                if param.requires_grad:
                    param.data = param.to(dtype)

    if accelerator.mixed_precision == "fp16":
        cast_training_params(transformer, dtype=torch.float32)

    transformer.print_trainable_parameters()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                transformer_ = accelerator.unwrap_model(transformer)
                lora_state_dict = get_peft_model_state_dict(transformer_, adapter_name="default")
                StableDiffusionPipeline.save_lora_weights(
                    os.path.join(output_dir, "transformer_lora"), lora_state_dict
                )
                transformer_.save_pretrained(output_dir)
                for _ in range(len(models)):
                    weights.pop()

        def load_model_hook(models, input_dir):
            transformer_ = accelerator.unwrap_model(transformer)
            transformer_.load_adapter(input_dir, "default", is_trainable=True)
            for _ in range(len(models)):
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    lora_params = [p for p in transformer.parameters() if p.requires_grad]

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Install bitsandbytes to use 8-bit Adam: pip install bitsandbytes")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    
    if args.dataset_name is not None:
        data_root = args.train_data_dir or "."

        if args.dataset_name == "hpdv2":
            print("hpdv2")
            dataset = load_from_disk(os.path.join(data_root, "hpdv2_pickapic_format"))
            print(len(dataset))

            dataset = DatasetDict({
                "train": dataset
            })
            indices_for_selection = get_strategy_indexes(
                dataset_name = "hpdv2",
                fixed_seed = args.seed,
                threshold = args.threshold,
                thresh_stage2 = args.thresh_stage2,
                n_strat_first = args.first_eigen,
                ratio = args.ratio,
                n_bins = args.bin_count,
                n_components = 1000,
                variance_weighted = args.variance_weighted,
                stratify = args.stratify,
                sft = args.sft,
                strategy = args.strategy
                )
        elif args.dataset_name == "pickapic-anonymous/pickapic_v1":
            print("PickApic")
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                data_dir=args.train_data_dir,
            )
            indices_for_selection = get_strategy_indexes(
                dataset_name = "pick", 
                fixed_seed = args.seed,
                threshold = args.threshold,
                thresh_stage2 = args.thresh_stage2,
                n_strat_first = args.first_eigen,
                ratio = args.ratio,
                n_bins = args.bin_count,
                n_components = 1000,
                variance_weighted = args.variance_weighted,
                stratify = args.stratify,
                sft = args.sft,
                strategy = args.strategy
                )

        elif args.dataset_name == "rapidata":
            print("rapidata")
            dataset = load_from_disk(os.path.join(data_root, "processed_rapidata_dataset_bytes"))
            print(len(dataset))

            dataset = DatasetDict({
                "train": dataset
            })
            indices_for_selection = get_strategy_indexes(
                dataset_name = "rapidata", 
                fixed_seed = args.seed,
                threshold = args.threshold,
                thresh_stage2 = args.thresh_stage2,
                n_strat_first = args.first_eigen,
                ratio = args.ratio,
                n_bins = args.bin_count,
                n_components = 1000,
                variance_weighted = args.variance_weighted,
                stratify = args.stratify,
                sft = args.sft,
                strategy = args.strategy
                )
        elif args.dataset_name == "roco":
            # ROCO Radiology is SFT-only, DPO on ROCO is not supported.
            if not args.sft:
                raise ValueError("ROCO Radiology is SFT-only, run with --sft.")
            print("Roco")
            dataset = load_from_disk(os.path.join(data_root, "roco_dataset"))
            dataset = DatasetDict({
                "train": dataset
            })
            print(len(dataset))
            indices_for_selection = get_strategy_indexes(
                dataset_name = "roco",
                fixed_seed = args.seed,
                threshold = args.threshold,
                thresh_stage2 = args.thresh_stage2,
                n_strat_first = args.first_eigen,
                ratio = args.ratio,
                n_bins = args.bin_count,
                n_components = 1000,
                variance_weighted = args.variance_weighted,
                stratify = args.stratify,
                sft = args.sft,
                strategy = args.strategy
                )

        not_split_idx = [i for i, label_0 in enumerate(dataset['train']['label_0'])
                         if label_0 in (0, 1)]
        dataset['train'] = dataset['train'].select(not_split_idx)

        if args.dataset_name in ["roco", "rapidata"]:
            dataset['train'] = dataset['train'].shuffle(seed=args.seed)
        else:
            dataset['train'] = dataset['train'].shuffle(seed=args.seed).select(range(100_000))

        n_rows = dataset['train'].num_rows
        if args.full:
            pass
        elif args.random:
            # Random baseline: same subset size as the eigen selection.
            n_select = min(len(indices_for_selection), n_rows)
            dataset['train'] = dataset['train'].select(range(n_select))
        else:
            if max(indices_for_selection) >= n_rows:
                raise ValueError(
                    f"selection index {max(indices_for_selection)} >= working-set size "
                    f"{n_rows}; embeddings and training set are misaligned (check --seed).")
            dataset['train'] = dataset['train'].select(indices_for_selection)
        print("train subset size:", dataset['train'].num_rows)
    else:
        data_files = {args.split: os.path.join(args.train_data_dir, "**")}
        dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=args.cache_dir)

    column_names = dataset[args.split].column_names
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    is_pickapic = args.dataset_name and ("pickapic" in args.dataset_name)

    if args.train_method == "sft" and not is_pickapic:
        image_column = args.image_column or (
            dataset_columns[0] if dataset_columns else column_names[0]
        )
        if image_column not in column_names:
            raise ValueError(f"--image_column '{image_column}' not in dataset columns: {column_names}")
    else:
        image_column = None

    caption_column = args.caption_column or (
        dataset_columns[1] if dataset_columns else column_names[1]
    )
    if caption_column not in column_names:
        raise ValueError(f"--caption_column '{caption_column}' not in dataset columns: {column_names}")

    def tokenize_captions(captions_list, proportion_empty_prompts=0.0):
        processed = []
        for caption in captions_list:
            if random.random() < proportion_empty_prompts:
                processed.append("")
            elif isinstance(caption, str):
                processed.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                processed.append(random.choice(caption))
            else:
                raise ValueError("Captions must be strings or lists of strings.")
        inputs = tokenizer(
            processed,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids, inputs.attention_mask

    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution),
        transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    if args.train_method == "dpo":
        def preprocess_train(examples):
            all_pixel_values = []
            for col in ["jpg_0", "jpg_1"]:
                imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in examples[col]]
                all_pixel_values.append([train_transforms(img) for img in imgs])

            combined_pixel_values = []
            for (im0, im1), label_0 in zip(zip(*all_pixel_values), examples["label_0"]):
                if label_0 == 0 and not args.choice_model:
                    im0, im1 = im1, im0
                combined_pixel_values.append(torch.cat([im0, im1], dim=0))  # [2C, H, W]
            examples["pixel_values"] = combined_pixel_values
            examples["input_ids"], examples["prompt_attention_mask"] = tokenize_captions(
                examples[caption_column],
                proportion_empty_prompts=args.proportion_empty_prompts,
            )
            return examples
        def collate_fn(examples):
            pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            ret = {
                "pixel_values": pixel_values,
                "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
                "prompt_attention_mask": torch.stack([ex["prompt_attention_mask"] for ex in examples]),
            }
            if args.choice_model:
                for k in ["jpg_0", "jpg_1"]:
                    ret[k] = [Image.open(io.BytesIO(ex[k])).convert("RGB") for ex in examples]
                ret["caption"] = [ex[caption_column] for ex in examples]
            return ret

        if args.choice_model:
            if args.choice_model == "hps":
                from utils.hps_utils import Selector
            elif args.choice_model == "clip":
                from utils.clip_utils import Selector
            elif args.choice_model == "pickscore":
                from utils.pickscore_utils import Selector
            elif args.choice_model == "aes":
                from utils.aes_utils import Selector
            else:
                raise ValueError(f"Unknown choice_model: {args.choice_model}")
            selector = Selector(accelerator.device)

            def choice_model_says_flip(batch):
                assert len(batch["caption"]) == 1
                scores = selector.score([batch["jpg_0"][0], batch["jpg_1"][0]], batch["caption"][0])
                return scores[1] > scores[0]

    else:  # SFT
        def preprocess_train(examples):
            has_jpg1 = "jpg_1" in examples
            images = []
            for i, (b0, label) in enumerate(zip(examples["jpg_0"], examples["label_0"])):
                assert label in (0, 1)
                b = b0 if (label == 1 or not has_jpg1) else examples["jpg_1"][i]
                images.append(Image.open(io.BytesIO(b)).convert("RGB"))
            examples["pixel_values"] = [train_transforms(img) for img in images]
            examples["input_ids"], examples["prompt_attention_mask"] = tokenize_captions(
                examples[caption_column],
                proportion_empty_prompts=args.proportion_empty_prompts,
            )
            return examples

        def collate_fn(examples):
            return {
                "pixel_values": torch.stack([ex["pixel_values"] for ex in examples]).to(
                    memory_format=torch.contiguous_format).float(),
                "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
                "prompt_attention_mask": torch.stack([ex["prompt_attention_mask"] for ex in examples]),
            }

    with accelerator.main_process_first():
        if is_pickapic:
            # Drop ambiguous (50/50) label examples
            """
            orig_len = dataset[args.split].num_rows
            keep_idx = [i for i, lbl in enumerate(dataset[args.split]["label_0"]) if lbl in (0, 1)]
            dataset[args.split] = dataset[args.split].select(keep_idx)
            print(f"Dropped {orig_len - len(keep_idx)}/{orig_len} ambiguous Pick-a-Pic pairs.")
            """
            if args.dreamlike_pairs_only:
                orig_len = dataset[args.split].num_rows
                dream_idx = [
                    i for i, (m0, m1) in enumerate(
                        zip(dataset[args.split]["model_0"], dataset[args.split]["model_1"])
                    )
                    if ("dream" in m0) and ("dream" in m1)
                ]
                dataset[args.split] = dataset[args.split].select(dream_idx)
                print(f"Dropped {orig_len - len(dream_idx)}/{orig_len} non-dreamlike pairs.")

        if args.max_train_samples is not None:
            dataset[args.split] = (
                dataset[args.split].shuffle(seed=args.seed).select(range(args.max_train_samples))
            )
        train_dataset = dataset[args.split].with_transform(preprocess_train)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(args.split == "train"),
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=False,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Method             = {args.train_method.upper()}")
    logger.info(f"  Num examples       = {len(train_dataset)}")
    logger.info(f"  Num epochs         = {args.num_train_epochs}")
    logger.info(f"  Batch size/device  = {args.train_batch_size}")
    logger.info(f"  Total batch size   = {total_batch_size}")
    logger.info(f"  Grad accum steps   = {args.gradient_accumulation_steps}")
    logger.info(f"  Total opt steps    = {args.max_train_steps}")
    if args.train_method == "dpo":
        logger.info(f"  Beta DPO           = {args.beta_dpo}")

    global_step = 0
    first_epoch = 0
    resume_step = 0  # how many micro-steps to skip in the first epoch

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1]),
            )
            path = dirs[-1] if dirs else None

        if path is None:
            accelerator.print("No checkpoint found — starting fresh.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = (
                global_step * args.gradient_accumulation_steps
                % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip steps to reach resume position (unless hard_skip_resume)
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
                and not args.hard_skip_resume
            ):
                if step % args.gradient_accumulation_steps == 0:
                    print(f"Skipping step {step} (resume to {resume_step})")
                continue

            with accelerator.accumulate(transformer):
                # -------- Prepare input images -------------------------------- #
                if args.train_method == "dpo":
                    # pixel_values: [B, 2C, H, W]
                    # chunk(2, dim=1) → two [B, C, H, W] tensors (preferred first)
                    # cat along batch → [2B, C, H, W]  (first B = preferred, last B = dispreferred)
                    feed_pixel_values = torch.cat(batch["pixel_values"].chunk(2, dim=1))
                    if args.choice_model and choice_model_says_flip(batch):
                        feed_pixel_values = feed_pixel_values.flip(0)
                else:
                    feed_pixel_values = batch["pixel_values"]

                with torch.no_grad():
                    latents = vae.encode(
                        feed_pixel_values.to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)

                bsz = latents.shape[0]  # 2B in DPO mode, B in SFT mode
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                if args.train_method == "dpo":
                    timesteps = timesteps.chunk(2)[0].repeat(2)
                    noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)
                    if args.input_perturbation:
                        new_noise = new_noise.chunk(2)[0].repeat(2, 1, 1, 1)

                noisy_latents = noise_scheduler.add_noise(
                    latents,
                    new_noise if args.input_perturbation else noise,
                    timesteps,
                )
                with torch.no_grad():
                    prompt_embeds = text_encoder(
                        batch["input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                    )[0]

                if args.train_method == "dpo":
                    prompt_embeds = prompt_embeds.repeat(2, 1, 1)
                    prompt_attention_mask = batch["prompt_attention_mask"].repeat(2, 1)
                else:
                    prompt_attention_mask = batch["prompt_attention_mask"]

                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise  # always use unperturbed noise as target
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
                unwrapped = getattr(transformer, "module", transformer)
                if unwrapped.config.sample_size == 128:
                    res_tensor = torch.tensor(
                        [args.resolution, args.resolution], dtype=weight_dtype, device=latents.device
                    ).repeat(bsz, 1)
                    ar_tensor = torch.tensor(
                        [1.0], dtype=weight_dtype, device=latents.device  # square images → AR=1
                    ).repeat(bsz, 1)
                    added_cond_kwargs = {"resolution": res_tensor, "aspect_ratio": ar_tensor}

                raw_model_out = transformer(
                    noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timesteps,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                model_pred = raw_model_out.chunk(2, 1)[0]

                if args.train_method == "sft":
                    if args.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        snr = compute_snr(noise_scheduler, timesteps)
                        if noise_scheduler.config.prediction_type == "v_prediction":
                            snr = snr + 1
                        mse_weights = (
                            torch.stack(
                                [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                            ).min(dim=1)[0]
                            / snr
                        )
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = (loss.mean(dim=list(range(1, loss.ndim))) * mse_weights).mean()

                elif args.train_method == "dpo":
                    model_losses = (model_pred.float() - target.float()).pow(2).mean(dim=[1, 2, 3])
                    model_losses_w, model_losses_l = model_losses.chunk(2)
                    raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                    model_diff = model_losses_w - model_losses_l

                    with torch.no_grad():
                        with torch.autocast(
                            device_type="cuda",
                            dtype=weight_dtype,
                            enabled=weight_dtype != torch.float32,
                        ):
                            ref_out = ref_transformer(
                                noisy_latents,
                                encoder_hidden_states=prompt_embeds,
                                encoder_attention_mask=prompt_attention_mask,
                                timestep=timesteps,
                                added_cond_kwargs=added_cond_kwargs,
                            ).sample.chunk(2, 1)[0].detach()
                        ref_losses = (ref_out.float() - target.float()).pow(2).mean(dim=[1, 2, 3])
                        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                        ref_diff = ref_losses_w - ref_losses_l
                        raw_ref_loss = ref_losses.mean()

                    scale_term = -0.5 * args.beta_dpo
                    inside_term = scale_term * (model_diff - ref_diff)
                    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                    loss = -1 * F.logsigmoid(inside_term).mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                if args.train_method == "dpo":
                    avg_model_mse = (
                        accelerator.gather(raw_model_loss.repeat(args.train_batch_size)).mean().item()
                    )
                    avg_ref_mse = (
                        accelerator.gather(raw_ref_loss.repeat(args.train_batch_size)).mean().item()
                    )
                    avg_acc = accelerator.gather(implicit_acc).mean().item()
                    implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                log_dict = {"train_loss": train_loss}
                if args.train_method == "dpo":
                    log_dict.update({
                        "model_mse": avg_model_mse,
                        "ref_mse": avg_ref_mse,
                        "implicit_acc": implicit_acc_accumulated,
                    })
                accelerator.log(log_dict, step=global_step)
                train_loss = 0.0
                implicit_acc_accumulated = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            ckpts = sorted(
                                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]),
                            )
                            if len(ckpts) >= args.checkpoints_total_limit:
                                for old in ckpts[: len(ckpts) - args.checkpoints_total_limit + 1]:
                                    shutil.rmtree(os.path.join(args.output_dir, old))
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                if accelerator.is_main_process:
                    if args.validation_prompt is not None and global_step % args.validation_epochs == 0:
                        # text_encoder=text_encoder.to('cpu')
                        # vae = vae.to('cpu')
                        logger.info(
                            f"Running validation... \n Generating {args.validation_images} images with prompt:"
                            f" {args.validation_prompt}."
                        )

                        pipe_eval = DiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            transformer=accelerator.unwrap_model(transformer, keep_fp32_wrapper=False),
                            text_encoder=accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=False),
                            vae=accelerator.unwrap_model(vae, keep_fp32_wrapper=False),
                            torch_dtype=weight_dtype,
                            safety_checker=None,
                        )#.to(accelerator.device)
                        # pipe_eval.enable_model_cpu_offload()
                        pipe_eval.set_progress_bar_config(disable=True)

                        generator = torch.Generator(device=accelerator.device)
                        if args.seed is not None:
                            generator = generator.manual_seed(0)
                        images = []
                        with torch.inference_mode():
                            for _ in range(args.validation_images):
                                images.append(pipe_eval(args.validation_prompt, num_inference_steps=20, generator=generator).images[0])

                        for index, temp_image in enumerate(images):

                            output_dir_step = os.path.join(args.output_dir, f"output_{global_step}")
                            os.makedirs(output_dir_step, exist_ok=True)

                            image_path = os.path.join(output_dir_step, f"output_{index}.png")
                            temp_image.save(image_path)
                        pipe_eval.to('cpu')
                        del pipe_eval.scheduler
                        del images
                        del generator
                        del pipe_eval
                        gc.collect()
                        torch.cuda.empty_cache()

                        remove_hook_from_module(text_encoder, recurse=True)
                        remove_hook_from_module(vae, recurse=True)
                        remove_hook_from_module(transformer, recurse=True)

                        text_encoder=text_encoder.to(accelerator.device)
                        vae = vae.to(accelerator.device)
                        transformer = transformer.to(accelerator.device)


            pbar_logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.train_method == "dpo":
                pbar_logs["implicit_acc"] = avg_acc
            progress_bar.set_postfix(**pbar_logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer_unwrapped = accelerator.unwrap_model(transformer, keep_fp32_wrapper=False)
        transformer_unwrapped.save_pretrained(args.output_dir)
        lora_state_dict = get_peft_model_state_dict(transformer_unwrapped)
        StableDiffusionPipeline.save_lora_weights(
            os.path.join(args.output_dir, "transformer_lora"), lora_state_dict
        )
        logger.info(f"Saved final LoRA weights to {args.output_dir}/transformer_lora")

    accelerator.end_training()


if __name__ == "__main__":
    main()
import argparse
import os
import os.path as osp
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from dataset import PairedGUAVADataset
from model import Difix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Difix inference on the GUAVA dataset split.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where predictions will be saved.")
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint path used to initialize the model.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to evaluate (default: train).")
    parser.add_argument("--timestep", type=int, default=199, help="Diffusion timestep used during inference.")
    parser.add_argument("--lora_rank_vae", type=int, default=4, help="LoRA rank used when initializing the VAE.")
    parser.add_argument("--mv_unet", action="store_true", help="Use the multi-view UNet variant.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--allow_tf32", action="store_true", help="Enable TF32 matrix multiplications on Ampere GPUs.")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable xFormers memory efficient attention.",
    )
    return parser.parse_args()


def prepare_model(args: argparse.Namespace) -> Difix:
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    model = Difix(
        pretrained_path=args.model_path,
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )
    model.set_eval()

    if args.enable_xformers_memory_efficient_attention:
        model.unet.enable_xformers_memory_efficient_attention()

    return model


def save_prediction(image_tensor: torch.Tensor, output_path: str) -> None:
    image_tensor = image_tensor.detach().float().clamp(-1.0, 1.0)
    image_tensor = image_tensor * 0.5 + 0.5
    image = to_pil_image(image_tensor.cpu())
    image.save(output_path)


def move_to_device(tensor: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.to(device, non_blocking=True)


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Difix currently requires CUDA for inference.")

    os.makedirs(args.output_dir, exist_ok=True)

    model = prepare_model(args)
    model.to(device)
    model.timesteps = model.timesteps.to(device)

    dataset = PairedGUAVADataset(
        dataset_path=args.dataset_path,
        split=args.split,
        tokenizer=model.tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    to_process = len(dataset)
    progress = tqdm(total=to_process, desc="Generating predictions")

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            conditioning = move_to_device(batch["conditioning_pixel_values"], device)
            prompt_tokens = None

            if "input_ids" in batch:
                prompt_tokens = batch["input_ids"].squeeze(1)
                prompt_tokens = move_to_device(prompt_tokens, device)
            elif "caption" in batch:
                prompt_tokens = model.tokenizer(
                    batch["caption"],
                    max_length=model.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                prompt_tokens = move_to_device(prompt_tokens, device)
            else:
                raise ValueError("Dataset must provide either tokenized input_ids or captions for prompting.")

            predictions = model(conditioning, prompt_tokens=prompt_tokens)
            predictions = predictions[:, 0]  # take the first (target) view

            batch_start = batch_idx * args.batch_size
            for local_idx in range(predictions.shape[0]):
                dataset_idx = batch_start + local_idx
                if dataset_idx >= to_process:
                    continue

                img_id = dataset.img_ids[dataset_idx]
                video_id, src_frm_key, tgt_frm_key = img_id.split(".")
                out_dir = osp.join(args.output_dir, video_id)
                os.makedirs(out_dir, exist_ok=True)
                output_path = osp.join(out_dir, f"{src_frm_key}.{tgt_frm_key}.png")

                save_prediction(predictions[local_idx], output_path)

            progress.update(predictions.shape[0])


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

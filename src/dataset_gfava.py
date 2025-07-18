import os.path as osp
import json
import torch
from PIL import Image
import torchvision.transforms.functional as F


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=512, width=512, tokenizer=None, root_dir=None):

        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer

        self.root_dir = root_dir

    def __len__(self):

        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        
        input_img = self.data[img_id]["render_image"]
        output_img = self.data[img_id]["gt_image"]
        output_mask = self.data[img_id]["gt_mask"]
        ref_img = self.data[img_id]["ref_image"] if "ref_image" in self.data[img_id] else None
        caption = self.data[img_id]["prompt"] if "prompt" in self.data[img_id] else "high quality human image"

        if self.root_dir is not None:
            input_img = osp.join(self.root_dir, input_img)
            output_img = osp.join(self.root_dir, output_img)
            output_mask = osp.join(self.root_dir, output_mask)
            if ref_img is not None:
                ref_img = osp.join(self.root_dir, ref_img)
        
        try:
            img_t = Image.open(input_img)
            output_t = Image.open(output_img)
            mask_t = Image.open(output_mask)
        except:
            print("Error loading image:", input_img, output_img)
            return self.__getitem__(idx + 1)

        img_t = F.to_tensor(img_t)
        if img_t.shape[0] == 4:
            img_t = img_t[:3, ...] * img_t[3:, ...] + (1.0 - img_t[3:, ...])
        img_t = F.resize(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        output_t = F.to_tensor(output_t)
        if output_t.shape[0] == 4:
            output_t = output_t[:3, ...] * output_t[3:, ...] + (1.0 - output_t[3:, ...])
        output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        mask_t = F.to_tensor(mask_t) == 1.0
        mask_t = F.resize(mask_t, self.image_size)

        if ref_img is not None:
            ref_t = Image.open(ref_img)
            ref_t = F.to_tensor(ref_t)
            if ref_t.shape[0] == 4:
                ref_t = ref_t[:3, ...] * ref_t[3:, ...] + (1.0 - ref_t[3:, ...])
            ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
        
            img_t = torch.stack([img_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0)            
        else:
            img_t = img_t.unsqueeze(0)
            output_t = output_t.unsqueeze(0)

        out = {
            "output_pixel_values": output_t,
            "output_mask_values": mask_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
        }
        
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out

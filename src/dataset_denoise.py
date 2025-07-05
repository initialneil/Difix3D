import json
import torch
from PIL import Image
import glob
import os
from torchvision import transforms
import torchvision.transforms.functional as F


class DenoiseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, noise_ival=[5, 30], height=512, width=512, input_ids=None):

        super().__init__()
        self.dataset_path = dataset_path
        self.noise_ival = torch.tensor(noise_ival) / 255.0
        self.image_size = (height, width)
        self.input_ids = input_ids

        img_fns = glob.glob(os.path.join(dataset_path, '*.jpg')) + glob.glob(os.path.join(dataset_path, '*.png'))
        self.data = {}
        for img_fn in img_fns:
            img_id = os.path.splitext(os.path.basename(img_fn))[0]
            self.data[img_id] = {
                'image_path': img_fn,
            }
        self.img_ids = list(self.data.keys())

        self.split = split
        if split == 'train':
            self.T_gt = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, interpolation=Image.LANCZOS),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
            self.T_input = transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.01)
                ], p=0.8),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.01, 0.2))
                ], p=0.5),
                transforms.Lambda(
                    self.add_random_noise
                ),
            ])
        else:
            self.T_gt = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=Image.LANCZOS),
            ])
            self.T_input = transforms.Compose([
                # transforms.Resize(self.image_size, interpolation=Image.LANCZOS),
            ])


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        input_img = self.data[img_id]['image_path']
        
        try:
            input_img = Image.open(input_img)
        except:
            print('Error loading image:', input_img)
            return self.__getitem__(idx + 1)

        output_t = self.T_gt(input_img)
        img_t = F.to_tensor(output_t)
        output_t = F.to_tensor(output_t)

        # output images scaled to -1,1
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        # input images scaled to 0,1
        img_t = self.T_input(img_t)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        out = {
            'output_pixel_values': output_t,
            'conditioning_pixel_values': img_t,
            'input_ids': self.input_ids,
        }

        return out

    def add_random_noise(self, img):
        """
        Adds Gaussian noise to a tensor image.
        Args:
            img (Tensor): Image tensor.
            noise_ival (list of 2 values): Noise training interval.
        Returns:
            Tensor: Noisy image tensor.
        """
        # std dev of each sequence
        stdn = torch.empty((1, 1, 1)).to(img).uniform_(self.noise_ival[0], to=self.noise_ival[1])
        # draw noise samples from std dev tensor
        if torch.rand(1).item() > 0.5:
            noise = torch.zeros_like(img)
        else:
            noise = torch.zeros_like(img[:1, ...])
        noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
        return img + noise

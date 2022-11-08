from turtle import color
import numbers
import torch
import math
import torchvision.transforms.functional as F
import torchvision.transforms as T
from typing import Dict, List, Optional, Tuple
from torch import Tensor
import numpy as np
import random

TYPES = [
    "RandomResizedCrop",
    "RandomHorizontalFlip",
    "RandomApply",
    "RandomGrayscale",
]
REQUIRED_FEAUTES = {
    "RandomGrayscale": 0,
    "RandomApply": 4,
    "RandomHorizontalFlip": 0,
    "RandomResizedCrop": 2,
}


class RandAugmentFixedNumAugs(torch.nn.Module):
    def __init__(self, size, color_jitter, num_augs) -> None:
        super().__init__()
        self.size = (size, size)
        self.color_jitter = color_jitter

        self.resize_scale = (0.08, 1.0)
        self.resize_ratio = (3.0 / 4.0, 4.0 / 3.0)
        self.interpolation = T.InterpolationMode.BILINEAR

        self.grayscale_prop = 0.2
        self.flip_prop = 0.5

        self.color_jitter_prop = 0.8

        self.brightness = self._check_input(color_jitter[0], "brightness")
        self.contrast = self._check_input(color_jitter[1], "contrast")
        self.saturation = self._check_input(color_jitter[2], "saturation")
        self.hue = self._check_input(
            color_jitter[3],
            "hue",
            center=0,
            bound=(-0.5, 0.5),
            clip_first_on_zero=False,
        )
        self.hash_per_image = {}
        self.num_augs = num_augs

    @torch.jit.unused
    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f"If {name} is a single number, it must be non negative."
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(
                f"{name} should be a single number or a list/tuple with length 2."
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    # @staticmethod
    def get_params(
        self, img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        height, width = F.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    # @staticmethod
    def get_jitter_params(
        self,
        brightness: Optional[List[float]],
        contrast: Optional[List[float]],
        saturation: Optional[List[float]],
        hue: Optional[List[float]],
    ) -> Tuple[
        Tensor, Optional[float], Optional[float], Optional[float], Optional[float]
    ]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = (
            None
            if brightness is None
            else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        )
        c = (
            None
            if contrast is None
            else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        )
        s = (
            None
            if saturation is None
            else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        )
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    # @staticmethod
    def get_image_hash(self, img):
        pixel_data = list(img.getdata())
        pixel_data = np.array(pixel_data).reshape(-1)
        return sum(pixel_data)

    def forward(self, img: Tensor) -> Tensor:
        # update hash per image
        img_hash = self.get_image_hash(img)

        if img_hash not in self.hash_per_image:
            self.hash_per_image[img_hash] = int(img_hash)

        bash_hash = self.hash_per_image[img_hash]
        aug_num = np.random.randint(0, self.num_augs)
        seed = bash_hash + aug_num

        # overkill ...
        # np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)

        # augmentations = []

        i, j, h, w = self.get_params(img, self.resize_scale, self.resize_ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        # augmentations.append([True, h / self.size[0], w / self.size[1]])

        if torch.rand(1) < self.flip_prop:
            img = F.hflip(img)
        #     augmentations.append([True])
        # else:
        #     augmentations.append([False])

        # color jitter
        if torch.rand(1) < self.color_jitter_prop:
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = self.get_jitter_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)
        #     augmentations.append(
        #         [
        #             True,
        #             brightness_factor,
        #             contrast_factor,
        #             saturation_factor,
        #             hue_factor,
        #         ]
        #     )
        # else:
        #     augmentations.append([False, 0, 0, 0, 0])

        if torch.rand(1) < self.grayscale_prop:
            num_output_channels = F.get_image_num_channels(img)
            img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        #     augmentations.append([True])
        # else:
        #     augmentations.append([False])

        return F.to_tensor(img), aug_num

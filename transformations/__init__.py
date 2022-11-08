import torchvision
from .randaugment import MyRandAugment
from .fullfixed import RandAugmentFixedNumAugs
from .custom_augmentations import AddGaussianNoise

mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


class Transforms:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, augmentation="full"):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        if augmentation == "none":
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=size),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif augmentation.startswith("fullfixed_"):

            num_augs = int(augmentation[len("fullfixed_") :])

            self.train_transform = RandAugmentFixedNumAugs(
                size, [0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s], num_augs
            )

        elif augmentation == "most_difficult":
            gaussian_blur = torchvision.transforms.GaussianBlur(
                kernel_size=(3, 3), sigma=(0.1, 2)
            )
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomAffine(
                        degrees=(0, 180), translate=(0.2, 0.2)
                    ),
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomApply([gaussian_blur], p=0.3),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.RandomErasing(p=0.5),
                ]
            )
        elif augmentation == "full_rotations":
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomAffine(
                        degrees=(0, 180), translate=(0.2, 0.2)
                    ),
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif augmentation == "full":
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif augmentation == "isometry":
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomAffine(
                        degrees=(0, 180), translate=(0.2, 0.2)
                    ),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif augmentation == "myrandaugment":
            self.train_transform = torchvision.transforms.Compose(
                [
                    MyRandAugment(2, 14),
                    torchvision.transforms.RandomCrop(size, padding=4),
                    torchvision.transforms.Resize(size),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            # augmentation should be a number between 0 and 1
            augmentation = float(augmentation)
            assert augmentation >= 0 and augmentation <= 1

            num_ops = int(augmentation * 10 // 3 + 1)

            num_bins = 100
            magnitude = min(int(augmentation * num_bins), num_bins - 1)

            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandAugment(
                        num_ops=num_ops,
                        magnitude=magnitude,
                        num_magnitude_bins=num_bins,
                    ),
                    torchvision.transforms.ToTensor(),
                ]
            )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

import os
import numpy as np
import json
import torch
import torchvision
import argparse

from tqdm import tqdm

from torch.nn.parallel import DataParallel

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from models import get_model, WholeModel
from modules.sync_batchnorm import convert_model
from transformations import Transforms

from utils.training_utils import load_optimizer, save_model
from sklearn.neighbors import KNeighborsClassifier
from utils.da import CutMix, MixUp
from utils.criterion import get_criterion
from utils.tinyimagenet import TinyImageNet
from functools import partial

DATASETS = {
    "CIFAR10": {
        "train": torchvision.datasets.CIFAR10,
        "test": partial(torchvision.datasets.CIFAR10, train=False),
    },
    "CIFAR100": {
        "train": torchvision.datasets.CIFAR100,
        "test": partial(torchvision.datasets.CIFAR100, train=False),
    },
    "TinyImageNet": {
        "train": TinyImageNet,
        "test": partial(TinyImageNet, split="val"),
    },
}


def filter_data(dataset, classes_to_keep, percentage_per_class_to_keep):
    # Keep a subset of a given dataset by filtering percentages per class
    # check if dataset has attribute classes and data and targets
    if (
        not hasattr(dataset, "classes")
        or not hasattr(dataset, "data")
        or not hasattr(dataset, "targets")
    ):
        print("Cannot filter data for the specified dataset")
        return

    if classes_to_keep is None:
        classes_to_keep = np.arange(len(dataset.classes))

    indices_per_class = {k: [] for k in classes_to_keep}

    for i, t in enumerate(dataset.targets):
        if t in classes_to_keep:
            indices_per_class[t].append(i)

    for i in classes_to_keep:
        np.random.shuffle(indices_per_class[i])

        indices_per_class[i] = indices_per_class[i][
            : int(len(indices_per_class[i]) * percentage_per_class_to_keep)
        ]

    indices = np.concatenate(list(indices_per_class.values()))
    np.random.shuffle(indices)

    dataset.data = np.array([dataset.data[i] for i in indices])
    dataset.targets = np.array([dataset.targets[i] for i in indices])


class RandomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        dataset,
        transform=None,
        classes_to_keep=None,
        percentage_per_class_to_keep=1,
        subsample_n=-1,
        augm_iid=False,
    ) -> None:
        super().__init__()
        if augm_iid and (subsample_n <= 0 or subsample_n >= len(dataset)):
            raise ValueError("Wrong value for subsample_n when augm_iid is true")

        self.dataset = dataset
        self.augm_iid = augm_iid
        self.num_augs = args.num_augs

        np.random.seed(args.randomdataset_seed)

        if subsample_n > 0:
            new_data, self.rest_data = (
                self.dataset.data[:subsample_n],
                self.dataset.data[subsample_n:],
            )
            new_targets, rest_targets = (
                self.dataset.targets[:subsample_n],
                self.dataset.targets[subsample_n:],
            )
            if augm_iid:
                labels = np.unique(self.dataset.targets)
                indices_per_class = {k: [] for k in labels}
                self.augmentation_dict = {k: [] for k, _ in enumerate(new_data)}
                curr_index_dict = {k: 0 for k in labels}
                for i, _ in enumerate(self.rest_data):
                    indices_per_class[rest_targets[i]].append(i)
                try:
                    for i, _ in enumerate(new_data):
                        label = new_targets[i]
                        self.augmentation_dict[i] = indices_per_class[label][
                            curr_index_dict[label]
                            * self.num_augs : self.num_augs
                            * (curr_index_dict[label] + 1)
                        ]
                        curr_index_dict[label] += 1
                except IndexError:
                    raise ValueError(
                        "too many augmentations per sample, not enough data"
                    )

            dataset.data = new_data

            self.rest_data = torch.tensor(self.rest_data).permute(0, 3, 1, 2).float()

            dataset.targets = new_targets
        else:
            self.rest_data, self.augmentation_dict = None, None

        self.size = len(dataset)

        filter_data(self.dataset, classes_to_keep, percentage_per_class_to_keep)

        self.size = len(dataset)

        self.num_classes = len(self.dataset.classes)

        if args.label_noise is not None:
            original_labels = self.dataset.targets

        labels = np.zeros((self.size, self.num_classes))
        indices = np.random.randint(low=0, high=self.num_classes, size=(self.size, 1))

        labels[
            np.arange(self.size).repeat(1),
            indices.reshape(-1),
        ] = 1

        if args.label_noise is not None and int(self.size * (1 - args.label_noise)) > 0:
            # sample some labels to be right
            indices_to_keep = np.random.choice(
                np.arange(self.size),
                size=int(self.size * (1 - args.label_noise)),
                replace=False,
            )
            labels[indices_to_keep, :] = 0
            labels[indices_to_keep, np.array(original_labels)[indices_to_keep]] = 1

        if args.fullfixed_labels == "random":
            num_augs = int(args.augmentation[len("fullfixed_") :])

            labels_for_augments = np.zeros((self.size * num_augs, self.num_classes))

            indices = np.random.randint(
                low=0, high=self.num_classes, size=(self.size * num_augs, 1)
            )
            labels_for_augments[
                np.arange(self.size * num_augs).repeat(1),
                indices.reshape(-1),
            ] = 1
            self.labels_for_augments = labels_for_augments.reshape(
                (self.size, num_augs, self.num_classes)
            )

        self.labels = labels
        self.transform = transform
        self.fullfixed_labels = args.fullfixed_labels

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        x, _ = self.dataset[idx]
        y = self.labels[idx]

        if self.fullfixed_labels is not None:
            # fixed number of augmentations per sample
            x, aug_seed = self.transform(x)

            if self.fullfixed_labels == "random":
                return x, self.labels_for_augments[idx, aug_seed]
            elif self.fullfixed_labels == "same":
                return x, self.labels[idx]
            else:
                raise NotImplementedError

        if self.augm_iid:
            return (
                self.rest_data[
                    np.random.choice(self.augmentation_dict[idx], size=1)
                ].squeeze(),
                y,
            )
        else:
            return self.transform(x), y


def train(args, train_loader, model, criterion, optimizer, writer):
    model.train()
    loss_epoch = 0

    if args.cutmix:
        cutmix_op = CutMix(args.image_size, beta=1.0)
    if args.mixup:
        mixup_op = MixUp(alpha=1.0, mixup_lims=args.mixup_lims)

    for step, batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()

        x, y = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        if args.cutmix or args.mixup:
            if args.cutmix:
                x, y, rand_label, lambda_ = cutmix_op((x, y))
            elif args.mixup:
                if np.random.rand() <= 0.8:
                    x, y, rand_label, lambda_ = mixup_op((x, y))
                else:
                    x, y, rand_label, lambda_ = x, y, torch.zeros_like(y), 1.0

        h_i, z_i = model(x)

        loss = criterion(z_i, y)

        y = torch.argmax(y, dim=-1)

        writer.add_scalar(
            "acc/train",
            torch.argmax(z_i.detach(), dim=-1).eq(y).float().mean(),
            args.global_step,
        )

        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)

        args.global_step += 1

        loss_epoch += loss.item()

    return loss_epoch / len(train_loader)


def knn_eval(model, train_loader, test_loader, n_neighbors=20):
    model.eval()

    features_train = []
    labels_train = []

    for step, (x, y) in enumerate(train_loader):
        x = x.cuda(non_blocking=True)

        with torch.no_grad():
            # take the embedding
            features = model(x)[0].detach().cpu().numpy()

        features_train.append(features)
        labels_train.append(y.detach().cpu().numpy())

    features_test = []
    labels_test = []

    for step, (x, y) in enumerate(test_loader):
        x = x.cuda(non_blocking=True)

        with torch.no_grad():
            features = model(x)[0].detach().cpu().numpy()

        features_test.append(features)
        labels_test.append(y.detach().cpu().numpy())

    features_train = np.concatenate(features_train, axis=0)
    features_test = np.concatenate(features_test, axis=0)

    labels_train = np.concatenate(labels_train, axis=0)
    labels_test = np.concatenate(labels_test, axis=0)

    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(
        features_train / np.linalg.norm(features_train, axis=-1, keepdims=True),
        labels_train,
    )

    preds = clf.predict(
        features_test / np.linalg.norm(features_test, axis=-1, keepdims=True)
    )

    return preds, labels_test


def main(gpu, args):
    if args.continue_run is not None:
        stored_args = json.load(open(args.continue_run + "/args.txt", "r"))
        for k, v in stored_args.items():
            setattr(args, k, v)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataset = RandomDataset(
        args,
        DATASETS[args.dataset]["train"](args.dataset_dir, download=True),
        transform=Transforms(
            size=args.image_size,
            augmentation=args.augmentation,
        ).train_transform,
        classes_to_keep=args.classes_to_keep,
        percentage_per_class_to_keep=args.percentage_per_class_to_keep,
        augm_iid=args.augm_iid,
        subsample_n=args.subsample_n,
    )

    train_dataset_eval_mode = DATASETS[args.dataset]["train"](
        args.dataset_dir,
        download=True,
        transform=Transforms(size=args.image_size).test_transform,
    )

    test_dataset = DATASETS[args.dataset]["test"](
        args.dataset_dir,
        download=True,
        transform=Transforms(size=args.image_size).test_transform,
    )

    filter_data(
        train_dataset_eval_mode,
        args.classes_to_keep_evaluation,
        args.percentage_per_class_to_keep_evaluation,
    )
    filter_data(
        test_dataset,
        args.classes_to_keep_evaluation,
        args.percentage_per_class_to_keep_evaluation,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    train_loader_eval_mode = torch.utils.data.DataLoader(
        train_dataset_eval_mode,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # initialize
    print("Getting model", args.model)
    encoder = get_model(
        args.model, pretrained=False, normalization=args.normalization_encoder
    )

    # initialize model
    model = WholeModel(
        encoder,
        projection_str=args.projection_str,
        projection_dim=train_dataset.num_classes,
        normalization=args.normalization_projector,
        last_normalization=args.last_normalization,
        final_affine=args.final_normalization_affine,
        bottleneck_dim=args.bottleneck_dim,
    )

    print("Model", model)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)

    criterion = get_criterion(args)
    print("Criterion", criterion)

    args.start_epoch = 0
    args.global_step = 0

    if args.continue_run:
        checkpoints = os.listdir(os.path.join(args.continue_run, "saves"))
        largest_checkpoint = max(
            [int(checkpoint.split("_")[1].split(".")[0]) for checkpoint in checkpoints]
        )

        args.start_epoch = int(largest_checkpoint) + 1
        args.global_step = args.start_epoch * len(train_loader)

        model_path = os.path.join(
            args.continue_run,
            "saves",
            "checkpoint_{}.tar".format(largest_checkpoint),
        )
        args.model_path = os.path.join(args.continue_run, "saves")

        print(
            "Setting start epoch to {}, global step to {} and loading path from {}".format(
                args.start_epoch,
                args.global_step,
                model_path,
            ),
        )
        model.load_state_dict(torch.load(model_path, map_location=args.device))

        writer = SummaryWriter(args.continue_run)
    else:
        writer = SummaryWriter(comment=args.comment)
        args.model_path = os.path.join(writer.log_dir, "saves")
        os.makedirs(args.model_path)

        with open(os.path.join(writer.log_dir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)
            writer.add_text("args", json.dumps(args.__dict__))

    # DDP / DP
    if args.num_gpus > 1:
        model = convert_model(model)
        model = DataParallel(model, device_ids=list(range(args.num_gpus)))

    model = model.to(args.device)

    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch = epoch

        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if epoch % args.save_every == 0 and epoch > 0:
            save_model(args, model, optimizer)

        writer.add_scalar("Loss/train", loss_epoch, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}\t")

        if epoch % args.eval_every == 0:
            preds, labels_test = knn_eval(
                model, train_loader_eval_mode, test_loader, n_neighbors=20
            )
            writer.add_scalar("Misc/knn_acc", (preds == labels_test).mean(), epoch)

        if scheduler:
            scheduler.step()

    ## end training
    save_model(args, model, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Random Labels")

    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=list(DATASETS.keys()),
    )
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--dataparallel", default=0, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--dataset-dir", default="./datasets", type=str)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--epochs", default=5000, type=int)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)

    parser.add_argument("--comment", default="test", type=str)
    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument("--continue-run", default=None, type=str)

    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--weight-decay", default=1e-6, type=float)
    parser.add_argument("--warmup-steps", default=None, type=int)

    parser.add_argument("--final-normalization-affine", default="false", type=str)
    parser.add_argument("--normalization-projector", default="none", type=str)
    parser.add_argument("--last-normalization", default=None, type=str)
    parser.add_argument("--normalization-encoder", default="bn2d", type=str)
    parser.add_argument("--bottleneck-dim", type=int, default=None)

    parser.add_argument("--cutmix", action="store_true")
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--mixup-lims", type=str, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--augmentation", type=str, default="full")
    parser.add_argument("--num-augs", type=int, default=None)
    parser.add_argument(
        "--fullfixed-labels", type=str, default=None, choices=[None, "random", "same"]
    )
    parser.add_argument("--classes-to-keep", type=str, default=None)
    parser.add_argument("--percentage-per-class-to-keep", type=float, default=1.0)
    parser.add_argument("--classes-to-keep-evaluation", type=str, default=None)
    parser.add_argument(
        "--percentage-per-class-to-keep-evaluation", type=float, default=1.0
    )
    parser.add_argument("--projection-str", type=str, default=None)

    parser.add_argument("--label-noise", type=float, default=None)
    parser.add_argument("--use-mse-loss", type=str, default="true")
    parser.add_argument("--randomdataset-seed", type=int, default=17)
    parser.add_argument("--augm-iid", type=str, default="false")
    parser.add_argument("--subsample-n", type=int, default=-1)

    args = parser.parse_args()

    args.classes_to_keep = (
        [int(x) for x in args.classes_to_keep.split(",")]
        if args.classes_to_keep is not None
        else None
    )
    args.classes_to_keep_evaluation = (
        [int(x) for x in args.classes_to_keep_evaluation.split(",")]
        if args.classes_to_keep_evaluation is not None
        else None
    )
    args.lr = float(args.lr)
    args.final_normalization_affine = args.final_normalization_affine.lower() == "true"
    args.use_mse_loss = args.use_mse_loss.lower() == "true"
    args.augm_iid = args.augm_iid.lower() == "true"

    if args.projection_str is not None:
        print("projection_str", args.projection_str)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.num_gpus = torch.cuda.device_count()

    main(0, args)

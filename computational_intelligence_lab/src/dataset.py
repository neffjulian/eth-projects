import os

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from src.utils import object_from_dict


def get_train_transform(augmentations, use_small, output_stride):
    """Returns the transformations used for training images

    Args:
        augmentations: A list of dicts containing the configuration for training augmentations
        use_small: A boolean indicating if small images (200x200) are used
    """
    size = 200 if use_small else 400
    new_size = (size // output_stride + 1) * output_stride if size % output_stride != 0 else size

    if (size != new_size):
        f"Input shape ({size}, {size}) not compatible with encoder. Expected image size "
        f"divisible by {output_stride}. Padding images to ({new_size}, {new_size})"

    train_transform = [A.PadIfNeeded(new_size, new_size)]

    if augmentations is not None:
        for a in augmentations:
            train_transform.append(object_from_dict(a))
    else:
        train_transform += [
            A.OneOf(
                [
                    A.HorizontalFlip(p=1),
                    A.VerticalFlip(p=1),
                    A.RandomRotate90(p=1),
                ],
                p=0.75,
            )
        ]

    return A.Compose(train_transform)


def get_val_transform(use_small, output_stride):
    """Returns the transformations used for validation and test images

    Args:
        size: Optional dimension for image resizing
    """
    size = 200 if use_small else 400
    new_size = (size // output_stride + 1) * output_stride if size % output_stride != 0 else size
    val_transform = [A.PadIfNeeded(new_size, new_size)]

    return A.Compose(val_transform)


class RoadSegDataset(Dataset):
    def __init__(self, img_path, mask_path, transform=None, preprocessing=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.preprocessing = preprocessing
        self.img_names = os.listdir(img_path)

    def _preprocess_mask(self, mask):
        mask = mask.astype(np.float32)
        mask[mask == 255] = 1.0
        mask = mask[..., np.newaxis]
        return mask

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]

        image = cv2.imread(os.path.join(self.img_path, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(
            self.mask_path, img_name), cv2.IMREAD_UNCHANGED)
        mask = self._preprocess_mask(mask)

        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask


class RoadSegTestDataset(Dataset):
    def __init__(self, img_path, transform=None, preprocessing=None):
        self.img_path = img_path
        self.transform = transform
        self.preprocessing = preprocessing
        self.img_names = os.listdir(img_path)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]

        image = cv2.imread(os.path.join(self.img_path, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            sample = self.transform(image=image)
            image = sample["image"]

        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        return img_name, image


class RoadSegDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir="../../data",
                 batch_size=8,
                 val_split=0.1,
                 shuffle=True,
                 random_seed=42,
                 num_workers=4,
                 preprocessing_fn=None,
                 massachusetts=False,
                 use_small_images=False,
                 augmentations=None,
                 output_stride=32):

        self.batch_size = batch_size
        self.val_split = val_split
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.preprocessing_fn = preprocessing_fn
        self.prepare_data_per_node = False
        self._log_hyperparams = False

        subfolder = "training"
        if massachusetts:
            subfolder = "massachusetts"

        self.train_img_path = os.path.join(
            data_dir, "processed", subfolder, "images" + ("_small" if use_small_images else ""))
        self.train_mask_path = os.path.join(
            data_dir, "processed", subfolder, "groundtruth" + ("_small" if use_small_images else ""))

        self.test_img_path = os.path.join(
            data_dir, "processed", "test", "images")

        self.train_transform = get_train_transform(
            augmentations=augmentations, use_small=use_small_images, output_stride=output_stride)
        self.val_transform = get_val_transform(use_small=use_small_images, output_stride=output_stride)

    def setup(self, stage=None):
        preprocessing_ = []
        if self.preprocessing_fn:
            preprocessing_.append(A.Lambda(image=self.preprocessing_fn))
        preprocessing_.append(ToTensorV2(transpose_mask=True))
        preprocessing = A.Compose(preprocessing_)

        self.train_dataset = RoadSegDataset(
            img_path=self.train_img_path, mask_path=self.train_mask_path, transform=self.train_transform, preprocessing=preprocessing)
        self.val_dataset = RoadSegDataset(
            img_path=self.train_img_path, mask_path=self.train_mask_path, transform=self.val_transform, preprocessing=preprocessing)
        self.test_dataset = RoadSegTestDataset(
            img_path=self.test_img_path, transform=self.val_transform, preprocessing=preprocessing)

        train_len = len(self.train_dataset)
        idx = list(range(train_len))
        split = int(np.floor(self.val_split * train_len))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(idx)

        train_idx, val_idx = idx[split:], idx[:split]
        self.train_sampler = SequentialSampler(train_idx)
        self.val_sampler = SequentialSampler(val_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=self.val_sampler, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

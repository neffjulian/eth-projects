import os
from importlib import import_module

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from tqdm import tqdm


def get_stats(pr_masks: torch.Tensor, gt_masks: torch.Tensor, patch_size=16, foreground_threshold=0.25):
    """Compute true positive, false positive, false negative, true negative 'patches'
    for each image.

    Args:
        pr_masks: predicted mask of shape (batch_size, 1, height, width) containing values in [0,1]
        gt_masks: groundtruth mask of shape (batch_size, 1, height, width)
        patch_size: size of a patch
        foreground_threshold: percentage of pixels of val 1 required to assign a foreground label to a patch
    """
    batch_size, num_classes, *dims = gt_masks.shape

    gt_masks = F.avg_pool2d(gt_masks, kernel_size=patch_size)
    pr_masks = F.avg_pool2d(pr_masks, kernel_size=patch_size)

    gt_masks = (gt_masks > foreground_threshold).float()
    pr_masks = (pr_masks > foreground_threshold).float()

    gt_masks = gt_masks.view(batch_size, num_classes, -1)
    pr_masks = pr_masks.view(batch_size, num_classes, -1)

    tp = (gt_masks * pr_masks).sum(2)
    fp = pr_masks.sum(2) - tp
    fn = gt_masks.sum(2) - tp
    tn = torch.prod(torch.tensor(dims)) - (tp + fp + fn)

    return tp, fp, fn, tn


def object_from_dict(d, **default_kwargs):
    """Create an object from a config dict

    Args:
        d: a dictionary containing the description of the object to be created. At least 'type' must be present.
    """
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)
    try:
        module_path, class_name = object_type.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % object_type
        raise ImportError(msg)

    module = import_module(module_path)
    try:
        return getattr(module, class_name)(**kwargs)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (
            module_path, class_name)
        raise ImportError(msg)

def get_dataset_statistics():
    """Plots the distribution of the number of street pixels 
    in the massachusetts dataset
    """

    data_path = "data/processed/massachusetts/groundtruth"
    size = len(os.listdir(data_path))
    percentages = np.zeros(size)


    for index, name in tqdm(enumerate(os.listdir(data_path))):
        path = os.path.join(data_path, name)
        im = Image.open(path)
        im = np.asarray(im)
        percentages[index] = np.count_nonzero(im) / im.size

    print(np.percentile(percentages, 10))
    print(np.percentile(percentages, 25))
    print(np.percentile(percentages, 50))
    print(np.percentile(percentages, 75))
    print(np.percentile(percentages, 90))

    plt.hist(percentages, bins=100) 
    plt.title("histogram") 
    plt.show()


class LogPredictionsCallback(Callback):
    """Callback to log example images with predictions and grountruth mask to wandb

    Args:
        wandb_logger: an instance of a wandb logger
        num_samples: number of samples to log per validation step
    """
    def __init__(self, wandb_logger: WandbLogger, num_samples=4):
        self.logger = wandb_logger
        self.num_samples = num_samples

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        def to_mask_image(image, pr_mask, gt_mask):
            return wandb.Image(image, masks={
                "predictions": {
                    "mask_data": pr_mask
                },
                "groundtruth": {
                    "mask_data": gt_mask
                }})

        if batch_idx == 0:
            n = 10
            images, gt_masks = batch

            with torch.no_grad():
                pl_module.eval()
                pr_masks = pl_module(images)
                pl_module.train()
            pr_masks = (pr_masks.sigmoid() > 0.5).float()

            images = [img.numpy().transpose(1, 2, 0) for img in images[:n]]
            gt_masks = [mask.numpy().squeeze() for mask in gt_masks[:n]]
            pr_masks = [mask.numpy().squeeze() for mask in pr_masks[:n]]

            data = [to_mask_image(img, pr, gt)
                    for img, pr, gt in zip(images, gt_masks, pr_masks)]
            self.logger.log_metrics({"examples": data})

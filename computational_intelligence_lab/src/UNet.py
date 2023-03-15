import torch
from pytorch_lightning import LightningModule

import segmentation_models_pytorch as smp

from src.utils import get_stats, object_from_dict

class SMPModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()
        self.model = object_from_dict(hparams["model"])

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(hparams["model"]["encoder_name"])
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.loss_fn = object_from_dict(hparams["loss_fn"])


    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch
        # Shape of the image should be (batch_size, num_channels, height, width)
        # Shape of the mask should be (batch_size, 1, height, width)

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = get_stats(pred_mask, mask)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])


        f1 = smp.metrics.f1_score(tp, fp, fn, tn)

        metrics = {
            f"{stage}_f1": f1,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def predict_step(self, batch, batch_idx):
        img_name, image = batch
        logits_mask = self.forward(image)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        return (img_name, pred_mask)

    def configure_optimizers(self):
        optimizer = object_from_dict(self.hparams["optimizer"],
            params=self.parameters())
        scheduler = object_from_dict(self.hparams["scheduler"],
            optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1"
            }
        }
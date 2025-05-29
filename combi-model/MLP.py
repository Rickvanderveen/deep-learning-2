import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd


RINE_EMBEDDING_SIZE = 1024
PATCHCRAFT_EMBEDDING_SIZE = 32
SPAI_EMBEDDING_SIZE = 1096


class ProjectionBlock(nn.Module):
    def __init__(self, input_size, projection_size, activation=nn.ReLU(), norm=True):
        super().__init__()

        self.block = nn.Sequential(nn.Linear(input_size, projection_size), activation)
        if norm:
            self.block.append(nn.LayerNorm(projection_size))

    def forward(self, x):
        return self.block(x)


class MLPClassifier(pl.LightningModule):
    def __init__(
        self,
        projection_size,
        num_classes,
        mlp_ratio=4,
        lr=1e-3,
        weight_decay=1e-4,
        dropout=0.5,
    ):
        super().__init__()
        self.projection_size = projection_size
        self.concat_embedding_dim = projection_size * 3
        self.hidden_dim = self.concat_embedding_dim * mlp_ratio

        self.save_hyperparameters()

        self.rine_projection = ProjectionBlock(
            RINE_EMBEDDING_SIZE, self.projection_size
        )
        self.patchcraft_projection = ProjectionBlock(
            PATCHCRAFT_EMBEDDING_SIZE, self.projection_size
        )
        self.spai_projection = ProjectionBlock(
            SPAI_EMBEDDING_SIZE, self.projection_size
        )

        self.model = nn.Sequential(
            nn.Linear(self.concat_embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_classes),
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.val_outputs = []
        self.eval_df = None

    def forward(self, x):
        rine_embedding, patchcraft_embedding, spai_embedding = torch.split(
            x,
            [RINE_EMBEDDING_SIZE, PATCHCRAFT_EMBEDDING_SIZE, SPAI_EMBEDDING_SIZE],
            dim=-1,
        )

        projected_rine = self.rine_projection(rine_embedding)
        projected_patchcraft = self.patchcraft_projection(patchcraft_embedding)
        projected_spai = self.spai_projection(spai_embedding)

        embeddings = torch.concat(
            [projected_rine, projected_patchcraft, projected_spai], dim=1
        )
        return self.model(embeddings).squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = y.detach().cpu().numpy()

        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        self.val_outputs.append(
            {
                "spai": probs.flatten(),
                "class": labels.flatten(),
                "val_loss": loss,
                "val_acc": acc,
            }
        )

        return loss

    def on_validation_epoch_end(self):
        all_spai = []
        all_class = []
        all_validation_loss = []

        # Get all batches their outputs
        for output in self.val_outputs:
            all_spai.extend(output["spai"])
            all_class.extend(output["class"])
            all_validation_loss.append(output["val_loss"])

        df = pd.DataFrame({"spai": all_spai, "class": all_class})
        self.eval_df = df

        avg_val_loss = torch.stack(all_validation_loss).mean()
        self.log("hp_metric", avg_val_loss)

        self.val_outputs = []  # Clear for next epoch

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

import argparse
import pickle
from utils import EmbeddingDataset
import torch
from transformers import set_seed
from transformers import Trainer, TrainingArguments
from MLP import MLPClassifier
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier for combi model")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--train_data_path", type=str, default="data/training_data/train_embeddings.pkl"
    )
    parser.add_argument(
        "--val_data_path", type=str, default="data/val_data/val_embeddings.pkl"
    )
    parser.add_argument("--model_name", type=str, default="Multi-Model-MLP-Classifier")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--projection_size", type=int, default=256)
    parser.add_argument("--mlp_ratio", type=float, default=2)

    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    params = parse_args()
    set_seed(params.seed)

    # --------
    # Dataset loading, make sure your data is preprocessed (for examples of the right format see preprocessing.ipynb)
    # --------
    print("loading dataset...")
    with open(params.train_data_path, "rb") as f:
        data = pickle.load(f)

    train_dataset = EmbeddingDataset(data)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True
    )

    with open(params.val_data_path, "rb") as f:
        data = pickle.load(f)

    val_dataset = EmbeddingDataset(data)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=params.batch_size, shuffle=False
    )

    # ---
    # Setup training
    # ---
    # Get input dimensions from the first sample
    x_sample, y_sample = train_dataset[0]
    input_dim = x_sample.shape[0]
    num_classes = 1

    # Initialize model
    model = MLPClassifier(
        projection_size=params.projection_size,
        mlp_ratio=params.mlp_ratio,
        num_classes=num_classes,
        lr=params.lr,
        weight_decay=params.weight_decay,
    )
    logger = TensorBoardLogger("tensorboard_multimodel_logs", name=params.model_name)

    custom_hyperparams = {"batch_size": params.batch_size}
    hyperparameters = {**model.hparams, **custom_hyperparams}
    logger.log_hyperparams(hyperparameters)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="checkpoints",
        filename=params.model_name,
        verbose=True,
    )
    # ---
    # Train
    # ---
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    device_stats = DeviceStatsMonitor()

    trainer = pl.Trainer(
        max_epochs=params.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor, device_stats],
        logger=logger,
    )

    trainer.fit(model, train_loader, val_loader)

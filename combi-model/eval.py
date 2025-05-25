import argparse
import pickle
import torch
from utils import EmbeddingDataset
from MLP import MLPClassifier
import pytorch_lightning as pl
from transformers import set_seed
from torch.utils.data import DataLoader
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MLP classifier")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    params = parse_args()
    set_seed(params.seed)

    print("Loading evaluation dataset...")
    with open(params.eval_data_path, "rb") as f:
        data = pickle.load(f)

    eval_dataset = EmbeddingDataset(data)
    eval_loader = DataLoader(eval_dataset, batch_size=params.batch_size, shuffle=False)

    x_sample, y_sample = eval_dataset[0]
    input_dim = x_sample.shape[0]
    num_classes = len(set([y for _, y in eval_dataset])) # Kinda unneeded cuz i expect we always got label 0-1 but whatever

    model = MLPClassifier.load_from_checkpoint(params.checkpoint_path, input_dim=input_dim, num_classes=num_classes)

    trainer = pl.Trainer(accelerator="auto", devices=1)

    print("Running evaluation...")
    results = trainer.validate(model, dataloaders=eval_loader)

    print(f"Evaluation results: {results}")

    os.makedirs(params.output_dir, exist_ok=True)
    prefix = params.eval_data_path.split("/")[-1]
    results_path = os.path.join(params.output_dir, f"{prefix}eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

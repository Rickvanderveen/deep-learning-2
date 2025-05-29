import argparse
import csv
from pathlib import Path

import click
from torch import Tensor, LongTensor
import torch
from torchmetrics.functional.classification import binary_auroc
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy


@click.group()
def cli():
    pass


def csv_to_tensor(csv_path: Path) -> tuple[Tensor, Tensor]:
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        labels = []
        predictions = []
        for row in reader:
            labels.append(float(row["class"]))
            predictions.append(float(row["spai"]))

    predictions = Tensor(predictions)
    labels = LongTensor(labels)
    return predictions, labels


@cli.command(name="aggregate")
@click.argument("csv_files", type=click.Path(exists=True, path_type=Path), nargs=-1)
def aggregate_metrics(csv_files: list[Path]) -> None:
    """
    Read one or more CSV files containing 'class' labels and 'spai' predictions,
    and compute the combined AUC and accuracy across all of them.

    Arguments:
      csv_files  One or more paths to CSV files. Each file must have columns:
                   - class: Ground-truth labels (0 or 1)
                   - spai:  Model prediction scores.
    """
    auroc = BinaryAUROC()
    accuracy = BinaryAccuracy()

    for csv_file in csv_files:
        predictions, labels = csv_to_tensor(csv_file)
        auroc.update(predictions, labels)
        accuracy.update(predictions, labels)

    auc = auroc.compute().item()
    accuracy = accuracy.compute().item()

    print("AUC:", auc)
    print("Accuracy:", accuracy)


@cli.command(name="avg")
@click.argument(
    "fake",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "real",
    type=click.Path(exists=True, path_type=Path),
    nargs=-1,
)
def compute_average(fake: Path, real: list[Path]):
    """
    Calculate the average AUC between the fake and multiple sets of real images.

    Arguments:
      fake  The path to the CSV with the predictions of the fake images.
      real  One or more paths to CSV file(s) with the predictions of the real images.
    """
    print(f"Average AUC of {fake.name} over " + ", ".join(map(lambda x: x.name, real)))

    longest_name = max(map(lambda path: len(path.stem), real))

    auc_scores = []

    fake_predictions, fake_labels = csv_to_tensor(fake)

    for real_csv in real:
        predictions, labels = csv_to_tensor(real_csv)

        auc = binary_auroc(
            torch.cat((fake_predictions, predictions)),
            torch.cat((fake_labels, labels)),
        ).item()
        auc_scores.append(auc)

        print(f"{real_csv.stem.rjust(longest_name)}: {auc}")

    avg_auc = sum(auc_scores) / len(auc_scores)
    print("Average AUC".rjust(longest_name) + f": {avg_auc}")


def compute_auc(fake_dir: Path, real_dir: Path):
    fake_dataset_paths = list(fake_dir.glob("*.csv"))
    real_dataset_paths = list(real_dir.glob("*.csv"))

    auc_scores = []

    for fake_dataset_path in fake_dataset_paths:
        fake_predictions, fake_labels = csv_to_tensor(fake_dataset_path)

        dataset_auc_scores = []
        for real_dataset_path in real_dataset_paths:
            real_predictions, real_labels = csv_to_tensor(real_dataset_path)

            auc = binary_auroc(
                torch.cat((fake_predictions, real_predictions)),
                torch.cat((fake_labels, real_labels)),
            ).item()
            dataset_auc_scores.append(auc)

        avg_dataset_auc = sum(dataset_auc_scores) / len(dataset_auc_scores)
        auc_scores.append(avg_dataset_auc)

    longest_name = max(map(lambda path: len(path.stem), fake_dataset_paths))

    print("Fake dataset | AUC")
    for fake_dataset_path, auc_score in zip(fake_dataset_paths, auc_scores):
        print(f"{fake_dataset_path.stem.rjust(longest_name)}: {auc_score}")
    print()
    average_auc = sum(auc_scores) / len(auc_scores)
    print("Average AUC".rjust(longest_name) + f": {average_auc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--fake_dir", type=Path)
    parser.add_argument("--real_dir", type=Path)

    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    params = parse_args()
    compute_auc(params.fake_dir, params.real_dir)

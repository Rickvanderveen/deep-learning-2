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


if __name__ == "__main__":
    cli()

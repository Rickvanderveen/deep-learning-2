import copy
import csv
from pathlib import Path
from multiprocessing import Process, Manager
from typing import Optional

import click
from torch import Tensor, LongTensor
import torch
from torchmetrics.functional.classification import binary_auroc
from tqdm import tqdm

from spai.metrics import Metrics


@click.group()
def cli(): ...


def update_metric_from_csv(
    csv_path: Path, metric_manager: Metrics, pbar: Optional[tqdm] = None
) -> None:
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        labels = []
        predictions = []
        for row in reader:
            labels.append(float(row["class"]))
            predictions.append(float(row["spai"]))
            if pbar is not None:
                pbar.update(1)

        predictions = Tensor(predictions)
        labels = Tensor(labels)
        metric_manager.update(predictions, labels)


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


def count_csv_rows(file_path: Path) -> int:
    with file_path.open("r") as file:
        return sum(1 for _ in file) - 1


def setup_metrics() -> Metrics:
    return Metrics(metrics=("auc", "ap", "accuracy"))


def create_pbar(
    csv_file_paths: list[Path],
    desc: str = "Update metrics with row",
    position: Optional[int] = None,
    leave_pbar: Optional[bool] = None,
):
    total_rows = sum(map(count_csv_rows, csv_file_paths))
    return tqdm(total=total_rows, desc=desc, position=position, leave=leave_pbar)


@cli.command(name="all")
@click.argument("csv_files", type=click.Path(exists=True, path_type=Path), nargs=-1)
def compute_all(csv_files: list[Path]) -> None:
    metric_manager = setup_metrics()

    for csv_file in csv_files:
        update_metric_from_csv(csv_file, metric_manager)

    metrics = metric_manager.compute()
    auc = metrics.get("auc")
    ap = metrics.get("ap")
    accuracy = metrics.get("accuracy")

    print("AUC:", auc)
    print("AP:", ap)
    print("Accuracy:", accuracy)


def compute_real_fake_metric(
    real_path: Path,
    metric_manager: Metrics,
    position: int,
    results: list[Optional[float]],
):
    update_metric_from_csv(real_path, metric_manager)
    metrics = metric_manager.compute()
    auc = metrics["auc"]
    assert auc.ndim == 0, "Expected metric is a single value"
    results[position] = float(metrics["auc"])


@cli.command(name="avg")
@click.argument("fake", type=click.Path(exists=True, path_type=Path))
@click.argument("real", type=click.Path(exists=True, path_type=Path), nargs=-1)
def compute_average(fake: Path, real: list[Path]):
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

import copy
import csv
from pathlib import Path
from typing import Optional

import click
from torch import Tensor
from tqdm import tqdm

from spai.metrics import Metrics


@click.group()
def cli(): ...


def update_metric_from_csv(
    csv_path: Path, metric_manager: Metrics, pbar: Optional[tqdm]
) -> None:
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = Tensor([float(row["class"])])
            prediction = Tensor([float(row["spai"])])

            metric_manager.update(prediction, label)
            if pbar is not None:
                pbar.update(1)


def count_csv_rows(file_path: Path) -> int:
    with file_path.open("r") as file:
        return sum(1 for _ in file) - 1


def setup_metrics() -> Metrics:
    return Metrics(metrics=("auc", "ap", "accuracy"))


def create_pbar(csv_file_paths: list[Path]):
    total_rows = sum(map(count_csv_rows, csv_file_paths))
    return tqdm(total=total_rows, desc="Update metrics with row")


@cli.command(name="all")
@click.argument("csv_files", type=click.Path(exists=True, path_type=Path), nargs=-1)
def compute_all(csv_files: list[Path]) -> None:
    metric_manager = setup_metrics()

    pbar = create_pbar(csv_files)

    for csv_file in csv_files:
        update_metric_from_csv(csv_file, metric_manager, pbar)
    pbar.close()

    metrics = metric_manager.compute()
    auc = metrics.get("auc")
    ap = metrics.get("ap")
    accuracy = metrics.get("accuracy")

    print("AUC:", auc)
    print("AP:", ap)
    print("Accuracy:", accuracy)


@cli.command(name="avg")
@click.argument("fake", type=click.Path(exists=True, path_type=Path))
@click.argument("real", type=click.Path(exists=True, path_type=Path), nargs=-1)
def compute_average(fake: Path, real: list[Path]):
    pbar = create_pbar([*real, fake])

    fake_metric_manager = setup_metrics()
    update_metric_from_csv(fake, fake_metric_manager, pbar)

    auc_scores = []
    for real_csv_path in real:
        metric_manager = copy.deepcopy(fake_metric_manager)

        update_metric_from_csv(real_csv_path, metric_manager, pbar)
        metrics = metric_manager.compute()
        auc_scores.append(metrics["auc"])

    for real_csv_path, auc in zip(real, auc_scores):
        print(f"{real_csv_path.stem} - {fake} -> {auc}")


if __name__ == "__main__":
    cli()

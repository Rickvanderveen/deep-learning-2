import argparse
import csv
from enum import StrEnum
from pathlib import Path

from torch import Tensor, LongTensor
import torch
from torchmetrics.functional.classification import binary_auroc, binary_accuracy, binary_average_precision


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


def compute_auc(fake_dataset_paths: list[Path], real_dataset_paths: list[Path]):
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

    print("Fake dataset".rjust(longest_name) + ": AUC")
    for fake_dataset_path, auc_score in zip(fake_dataset_paths, auc_scores):
        print(f"{fake_dataset_path.stem.rjust(longest_name)}: {auc_score}")
    print()
    average_auc = sum(auc_scores) / len(auc_scores)
    print("Average AUC".rjust(longest_name) + f": {average_auc}")


def compute_acc_over_datasets(fake_dataset_paths: list[Path], real_dataset_paths: list[Path]):
    acc_scores = []

    for fake_dataset_path in fake_dataset_paths:
        fake_predictions, fake_labels = csv_to_tensor(fake_dataset_path)

        dataset_acc_scores = []
        for real_dataset_path in real_dataset_paths:
            real_predictions, real_labels = csv_to_tensor(real_dataset_path)

            accuracy = binary_accuracy(
                torch.cat((fake_predictions, real_predictions)),
                torch.cat((fake_labels, real_labels)),
            ).item()
            dataset_acc_scores.append(accuracy)

        avg_dataset_acc = sum(dataset_acc_scores) / len(dataset_acc_scores)
        acc_scores.append(avg_dataset_acc)

    longest_name = max(map(lambda path: len(path.stem), fake_dataset_paths))

    print("Fake dataset".rjust(longest_name) + ": Accuracy")
    for fake_dataset_path, acc_score in zip(fake_dataset_paths, acc_scores):
        print(f"{fake_dataset_path.stem.rjust(longest_name)}: {acc_score}")
    print()
    average_acc = sum(acc_scores) / len(acc_scores)
    print("Average Accuracy".rjust(longest_name) + f": {average_acc}")


def compute_acc(paths: list[Path]):
    acc_scores = []

    for path in paths:
        predictions, labels = csv_to_tensor(path)

        acc = binary_accuracy(
            predictions,
            labels,
            threshold=0.5
        ).item()
        acc_scores.append(acc)

    longest_name = max(map(lambda path: len(path.stem), paths))

    print("Fake dataset".rjust(longest_name) + ": Accuracy")
    for path, acc_score in zip(paths, acc_scores):
        print(f"{path.stem.rjust(longest_name)}: {acc_score}")
    print()
    average_acc = sum(acc_scores) / len(acc_scores)
    print("Average Accuracy".rjust(longest_name) + f": {average_acc}")


def compute_ap(fake_dataset_paths: list[Path], real_dataset_paths: list[Path]):
    ap_scores = []

    for fake_dataset_path in fake_dataset_paths:
        fake_predictions, fake_labels = csv_to_tensor(fake_dataset_path)

        dataset_ap_scores = []
        for real_dataset_path in real_dataset_paths:
            real_predictions, real_labels = csv_to_tensor(real_dataset_path)

            ap = binary_average_precision(
                torch.cat((fake_predictions, real_predictions)),
                torch.cat((fake_labels, real_labels)),
            ).item()
            dataset_ap_scores.append(ap)

        avg_dataset_ap = sum(dataset_ap_scores) / len(dataset_ap_scores)
        ap_scores.append(avg_dataset_ap)

    longest_name = max(map(lambda path: len(path.stem), fake_dataset_paths))

    print("Fake dataset".rjust(longest_name) + ": AP")
    for fake_dataset_path, ap_score in zip(fake_dataset_paths, ap_scores):
        print(f"{fake_dataset_path.stem.rjust(longest_name)}: {ap_score}")
    print()
    average_ap = sum(ap_scores) / len(ap_scores)
    print("Average AP".rjust(longest_name) + f": {average_ap}")


class Metric(StrEnum):
    AUC = "auc"
    ACC = "acc"
    SINGLE_ACC = "single_acc"
    AP = "ap"


def compute_metric(fake_paths: list[Path], real_paths: list[Path], metric: Metric) -> None:
    match metric:
        case Metric.AUC:
            compute_auc(fake_paths, real_paths)
        case Metric.AP:
            compute_ap(fake_paths, real_paths)
        case Metric.ACC:
            compute_acc_over_datasets(fake_paths, real_paths)
        case Metric.SINGLE_ACC:
            compute_acc(fake_paths + real_paths)


def collect_csv_paths(path: Path) -> tuple[list[Path], list[Path]]:
    """
    Collect CSV file paths from a given path.

    If the path is a directory, this function returns all `.csv` files within it,
    categorized into 'fake' and 'real' based on their filenames.
    If the path is a file, it categorizes the file accordingly.

    Args:
        path (Path): The file or directory path to collect CSV files from.

    Returns:
        Tuple[List[Path], List[Path]]: Two lists containing paths to 'fake' and 'real'
        CSV files respectively.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a directory.
    """
    if not path.exists():
        raise FileNotFoundError(f"Can not find {path}")
    if path.is_dir():
        fake_paths = list(path.glob("fake*.csv"))
        real_paths = list(path.glob("real*.csv"))
        return fake_paths, real_paths
    else:
        raise ValueError(f"Path is not a directory: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--metric", type=Metric, choices=list(Metric), required=True)
    parser.add_argument("--input_dir", "-i", type=Path, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    params = parse_args()

    fake_dataset_paths, real_dataset_paths = collect_csv_paths(params.input_dir)

    print(f"Fake datasets found: {len(fake_dataset_paths)}")
    print(f"Real datasets found: {len(real_dataset_paths)}")

    compute_metric(fake_dataset_paths, real_dataset_paths, params.metric)

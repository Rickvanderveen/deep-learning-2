import argparse
import logging
from pathlib import Path
import pickle

from sidbench.dataset.dataset import RecursiveImageDataset
from sidbench.dataset.process import clip_processing, rptc_processing
from sidbench.models.RPTC import Net as RPTCNet
from sidbench.models.Rine import RineModel
from spai.config import get_config
from spai.data.data_finetune import build_loader_test as build_loader_test_spai
from spai.models import build_cls_model
from tqdm import tqdm
import torch

from spai.utils import (
    find_pretrained_checkpoints,
    load_pretrained as load_pretrained_spai,
)


def forward_model(img, model, device):
    if isinstance(img, list):
        img = [i.to(device) if isinstance(i, torch.Tensor) else i for i in img]
        embeddings = model.get_embedding(*img)
    else:
        img = img.to(device)
        embeddings = model.get_embedding(img)
    return embeddings


def forward_spai(images, target, model, dataset_idx, config):
    if isinstance(images, list):
        # In case of arbitrary resolution models the batch is provided as a list of tensors.
        images = [img.cuda(non_blocking=True) for img in images]
        # Remove views dimension. Always 1 during inference.
        images = [img.squeeze(dim=1) for img in images]
    else:
        images = images.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    # Compute output.
    if isinstance(images, list) and config.TEST.EXPORT_IMAGE_PATCHES:
        export_dirs: list[Path] = [
            Path(config.OUTPUT) / "images" / f"{dataset_idx.detach().cpu().tolist()[i]}"
            for i in range(len(dataset_idx))
        ]
        output, attention_masks = model.get_embedding(
            images, config.MODEL.FEATURE_EXTRACTION_BATCH, export_dirs
        )
    elif isinstance(images, list):
        output = model.get_embedding(images, config.MODEL.FEATURE_EXTRACTION_BATCH)
        _attention_masks = [None] * len(images)
    else:
        if images.size(dim=1) > 1:
            predictions: list[torch.Tensor] = [
                model.get_embedding(images[:, i]) for i in range(images.size(dim=1))
            ]
            predictions: torch.Tensor = torch.stack(predictions, dim=1)
            if config.TEST.VIEWS_REDUCTION_APPROACH == "max":
                raise NotImplementedError
                # output: torch.Tensor = predictions.max(dim=1).values
            elif config.TEST.VIEWS_REDUCTION_APPROACH == "mean":
                raise NotImplementedError
                # output: torch.Tensor = predictions.mean(dim=1)
            else:
                raise TypeError(
                    f"{config.TEST.VIEWS_REDUCTION_APPROACH} is not a "
                    f"supported views reduction approach"
                )
        else:
            images = images.squeeze(dim=1)  # Remove views dimension.
            output = model.get_embedding(images)
        _attention_masks = [None] * images.size(0)

    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Triple model training")
    parser.add_argument("--dataPath", type=str, help="data folder path")
    parser.add_argument("--data_split", type=str)
    parser.add_argument(
        "--embedding_file",
        "-o",
        type=str,
        default="embeddings.pkl",
        help="The path of the output embeddings file",
    )

    parser.add_argument(
        "--root_dir", type=str, default=None, help="root dir of the data"
    )
    parser.add_argument("--output_dir", type=str, default="./triple_model_results")
    parser.add_argument("--numThreads", type=int, default=1)
    parser.add_argument("--batchSize", type=int, default=2)

    parser.add_argument(
        "--loadSize", type=int, default=None, help="scale images to this size"
    )
    parser.add_argument("--cropSize", type=int, default=224, help="crop to this size")

    parser.add_argument(
        "--isTrain",
        default=False,
        type=bool,
        help="train or test for rine and patchcraft",
    )

    # Specific to RPTC
    parser.add_argument("--patchNum", type=int, default=3)

    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    opt = parse_args()

    # Check the data split input argument
    valid_data_splits = ["train", "val", "test"]
    if opt.data_split not in valid_data_splits:
        raise ValueError(
            f"Invalid data_split: '{opt.data_split}'. Expected one of {valid_data_splits}."
        )

    # Check if the output embedding file is a pickle file
    if Path(opt.embedding_file).suffix != ".pkl":
        raise ValueError(
            f"Invalid file extension for embedding_file: '{opt.embedding_file}'."
            "Expected a '.pkl' file."
        )

    device = torch.device("cuda")
    seed = 10

    if opt.root_dir is None:
        root_dir = Path(opt.dataPath).parent
    else:
        root_dir = Path(opt.root_dir)

    # Setup dataset
    rine_dataset = RecursiveImageDataset(
        data_path=opt.dataPath,
        opt=opt,
        process_fn=clip_processing,
        root_dir=str(root_dir),
        data_split=opt.data_split,
    )
    patchcraft_dataset = RecursiveImageDataset(
        data_path=opt.dataPath,
        opt=opt,
        process_fn=rptc_processing,
        root_dir=str(root_dir),
        data_split=opt.data_split,
    )

    # Setup dataloader
    rine_loader = torch.utils.data.DataLoader(
        rine_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.numThreads,
        generator=torch.Generator().manual_seed(10),
    )
    patchcraft_loader = torch.utils.data.DataLoader(
        patchcraft_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.numThreads,
        generator=torch.Generator().manual_seed(10),
    )

    # Checkpoint paths
    patchcraft_checkpoint = Path("sidbench/weights/rptc/RPTC.pth")
    rine_checkpoint = Path("sidbench/weights/rine/model_1class_trainable.pth")
    spai_checkpoint = Path("weights/spai.pth")

    # Load patchcraft model
    patchcraft = RPTCNet()
    patchcraft.load_weights(ckpt=patchcraft_checkpoint)
    patchcraft.eval()
    patchcraft = patchcraft.to(device)

    # Load rine model
    # Hyperparameter based on the '1class' RineModel
    rine = RineModel(backbone=("ViT-L/14", 1024), nproj=4, proj_dim=1024)
    rine.load_weights(ckpt=rine_checkpoint)
    rine.eval()
    rine = rine.to(device)

    # Init spai config values
    # Temp empty
    spai_config = get_config(
        {
            "cfg": str("configs/spai.yaml"),
            "batch_size": opt.batchSize,
            "test_csv": [str(opt.dataPath)],
            "test_csv_root": [str(root_dir)],
            "lmdb_path": None,
            "output": str(opt.output_dir),
            "tag": "triple model",
            "pretrained": str(spai_checkpoint),
            "resize_to": None,
            "opts": (),
        }
    )

    # Setup spai dataset and dataloader
    _test_datasets_names, spai_test_datasets, spai_test_loaders = (
        build_loader_test_spai(
            spai_config,
            logger,
            split=opt.data_split,
            dummy_csv_dir=Path(opt.output_dir),
            data_loader_generator=torch.Generator().manual_seed(10),
            shuffle_data_loader=True,
            alternative_data_split="test",
        )
    )
    spai_dataset = spai_test_datasets[0]
    spai_loader = spai_test_loaders[0]

    # Check if all datasets loaded the same amount of images
    rine_len = len(rine_dataset)
    patchcraft_len = len(patchcraft_dataset)
    spai_len = len(spai_dataset)
    assert rine_len == patchcraft_len == spai_len, (
        f"Dataset size mismatch:\n"
        f"RINE: {rine_len}, PatchCraft: {patchcraft_len}, SPAI: {spai_len}"
    )

    # Load spai model
    spai = build_cls_model(spai_config)
    spai.eval()
    spai = spai.to(device)
    model_ckpt = find_pretrained_checkpoints(spai_config)[0]
    logger.info(f"Spai model checkpoint path: {model_ckpt}")
    load_pretrained_spai(
        spai_config, spai, logger, checkpoint_path=model_ckpt, verbose=False
    )

    data = []

    with tqdm(total=len(rine_dataset)) as pbar:
        for rine_batch, patchcraft_batch, spai_batch in zip(
            rine_loader, patchcraft_loader, spai_loader
        ):
            rine_img, rine_label, rine_img_path = rine_batch
            patchcraft_img, patchcraft_label, patchcraft_img_path = patchcraft_batch
            spai_img, spai_target, spai_dataset_idx = spai_batch

            assert rine_img_path == patchcraft_img_path, "Paths should be the same"

            spai_target = spai_target.float().to(device)

            with torch.no_grad():
                rine_embeddings = forward_model(rine_img, rine, device)
                patchcraft_embeddings = forward_model(
                    patchcraft_img, patchcraft, device
                )
                spai_embeddings = forward_spai(
                    spai_img, spai_target, spai, spai_dataset_idx, spai_config
                )

                concat_embedding = torch.concat(
                    (rine_embeddings, patchcraft_embeddings, spai_embeddings), dim=1
                ).cpu()

            for embedding, label, path in zip(
                concat_embedding, rine_label, rine_img_path
            ):
                data.append({"embedding": embedding, "label": label, "path": path})

            pbar.update(rine_img.shape[0])

    with open(opt.embedding_file, "wb") as f:
        pickle.dump(data, f)

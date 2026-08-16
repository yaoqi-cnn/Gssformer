import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import available_variants, get_experiment
from dataset import load_attributes, load_raw_scene, make_features, scene_files


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _axis_centers(lower, upper, block_size, stride):

    if block_size <= 0 or stride <= 0:
        raise ValueError("block_size and stride must be positive")
    extent = float(upper - lower)
    midpoint = (float(lower) + float(upper)) * 0.5
    if extent <= block_size:
        return np.asarray([midpoint], dtype=np.float32)
    half = block_size * 0.5
    first = float(lower) + half
    last = float(upper) - half
    values = np.arange(first, last + 0.5 * stride, stride, dtype=np.float64)
    if values.size == 0:
        values = np.asarray([first], dtype=np.float64)
    if values[-1] < last - 1e-6:
        values = np.append(values, last)
    else:
        values[-1] = min(values[-1], last)
    return values.astype(np.float32)


def centers(xyz, block_size, stride):
    x = _axis_centers(xyz[:, 0].min(), xyz[:, 0].max(), block_size, stride)
    y = _axis_centers(xyz[:, 1].min(), xyz[:, 1].max(), block_size, stride)
    return np.asarray([(a, b) for a in x for b in y], dtype=np.float32)


def confusion(prediction, target, classes):
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    valid = (
        (target >= 0)
        & (target < classes)
        & (prediction >= 0)
        & (prediction < classes)
    )
    encoded = target[valid] * classes + prediction[valid]
    return np.bincount(
        encoded, minlength=classes * classes
    ).reshape(classes, classes)


def metrics(matrix, names):
    true_positive = np.diag(matrix).astype(np.float64)
    denominator = matrix.sum(1) + matrix.sum(0) - true_positive
    iou = np.divide(
        true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return {
        "mIoU": float(iou.mean()),
        "OA": float(true_positive.sum() / max(int(matrix.sum()), 1)),
        "IoU_per_class": {
            name: float(iou[index]) for index, name in enumerate(names)
        },
        "confusion_matrix": matrix.tolist(),
    }


def infer(
    model,
    xyz,
    rgb,
    normals,
    color_gradient,
    block_size,
    stride,
    device,
    classes,
):
    score_sum = np.zeros((len(xyz), classes), dtype=np.float64)
    votes = np.zeros(len(xyz), dtype=np.int32)
    half = block_size / 2.0
    model.eval()
    with torch.inference_mode():
        for center in tqdm(
            centers(xyz, block_size, stride), desc="Inference"
        ):
            mask = (
                (xyz[:, 0] >= center[0] - half)
                & (xyz[:, 0] <= center[0] + half)
                & (xyz[:, 1] >= center[1] - half)
                & (xyz[:, 1] <= center[1] + half)
            )
            indices = np.flatnonzero(mask)
            if len(indices) == 0:
                continue
            features, coordinates = make_features(
                xyz[indices],
                rgb[indices],
                normals[indices],
                color_gradient[indices],
                center,
                block_size,
            )
            features = torch.from_numpy(features).unsqueeze(0).to(device)
            coordinates = torch.from_numpy(coordinates).unsqueeze(0).to(device)
            logits, _, _, _, _ = model(features, coordinates)
            score_sum[indices] += logits.squeeze(0).float().cpu().numpy()
            votes[indices] += 1

    uncovered = np.flatnonzero(votes == 0)
    if len(uncovered):
        raise RuntimeError(f"Uncovered points: {len(uncovered)}")
    return score_sum.argmax(1), votes


def average(results, names):
    output = {}
    for key in ("mIoU", "OA"):
        values = np.asarray([result[key] for result in results], dtype=np.float64)
        output[key] = float(values.mean())
        output[key + "_std"] = float(values.std(ddof=0))
    output["IoU_per_class"] = {}
    output["IoU_per_class_std"] = {}
    for name in names:
        values = np.asarray(
            [result["IoU_per_class"][name] for result in results],
            dtype=np.float64,
        )
        output["IoU_per_class"][name] = float(values.mean())
        output["IoU_per_class_std"][name] = float(values.std(ddof=0))
    return output


def _resolve_experiment(args, checkpoint):
    current = get_experiment(
        args.dataset,
        args.data_root,
        args.variant or "gssformer",
    )
    saved = checkpoint.get("experiment")
    if saved is None:
        if args.variant is None:
            raise ValueError(
                "This checkpoint has no embedded experiment configuration; "
                "specify --variant explicitly."
            )
        return current

    saved_dataset = saved.get("name")
    if saved_dataset is not None and saved_dataset != args.dataset:
        raise ValueError(
            f"Checkpoint dataset={saved_dataset}, requested dataset={args.dataset}"
        )
    if saved.get("class_names") not in (None, current["class_names"]):
        raise ValueError("Checkpoint class order does not match the dataset config")
    for split_key in ("train_areas", "val_areas", "test_areas"):
        if saved.get(split_key) not in (None, current[split_key]):
            raise ValueError(
                f"Checkpoint {split_key} does not match the current fixed split"
            )

    experiment = deepcopy(current)
    if args.variant is None:
        experiment["model"] = deepcopy(saved["model"])
        experiment["variant"] = saved.get("variant", "checkpoint")
    elif saved.get("model") != current["model"]:
        raise ValueError(
            "The requested variant does not match the model configuration "
            "stored in the checkpoint."
        )
    return experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("village", "sensaturban"), required=True)
    parser.add_argument("--variant", choices=available_variants(), default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    prediction_group = parser.add_mutually_exclusive_group()
    prediction_group.add_argument("--save-predictions", action="store_true")
    prediction_group.add_argument("--no-save-predictions", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for the paper configuration")

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    experiment = _resolve_experiment(args, checkpoint)
    if "epoch" not in checkpoint:
        raise ValueError(
            "Checkpoint epoch metadata is missing. Paper-aligned evaluation "
            "requires the checkpoint selected by validation mIoU."
        )
    if checkpoint.get("selection_metric") != "validation_mIoU":
        raise ValueError(
            "Paper-aligned evaluation requires a checkpoint selected by "
            "validation mIoU."
        )
    if checkpoint.get("checkpoint_role") != "best_validation":
        raise ValueError(
            "Use best_model.pth for test evaluation; last_model.pth and "
            "final_model.pth are not validation-selected checkpoints."
        )
    best_epoch = int(checkpoint.get("best_epoch", -1))
    checkpoint_epoch = int(checkpoint["epoch"])
    if best_epoch != checkpoint_epoch:
        raise ValueError(
            f"Checkpoint epoch={checkpoint_epoch}, but best_epoch={best_epoch}."
        )
    if "best_validation_mIoU" not in checkpoint:
        raise ValueError(
            "The checkpoint does not contain best_validation_mIoU metadata."
        )
    evaluation = experiment["evaluation"]
    runs = int(evaluation["num_runs"])
    if runs < 1:
        raise ValueError("runs must be at least 1")
    save_predictions = bool(evaluation["save_predictions"])
    if args.save_predictions:
        save_predictions = True
    if args.no_save_predictions:
        save_predictions = False


    base_seed = int(args.seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from models import create_model

    model = create_model(
        num_classes=len(experiment["class_names"]),
        **experiment["model"],
    ).to(args.device)
    model.load_state_dict(
        checkpoint.get("model_state_dict", checkpoint), strict=True
    )
    files = scene_files(experiment["data_root"], experiment["test_areas"])

    run_results = []
    run_seeds = []
    for run in range(runs):
        run_seed = int(base_seed + run)
        run_seeds.append(run_seed)
        set_seed(run_seed)
        matrix = np.zeros(
            (len(experiment["class_names"]), len(experiment["class_names"])),
            dtype=np.int64,
        )
        scenes = []
        for path in files:
            xyz, rgb, labels = load_raw_scene(path)
            normals, color_gradient, _ = load_attributes(
                path, xyz, rgb, None
            )
            prediction, votes = infer(
                model,
                xyz,
                rgb,
                normals,
                color_gradient,
                experiment["block_size"],
                experiment["stride"],
                args.device,
                len(experiment["class_names"]),
            )
            scene_matrix = confusion(
                prediction, labels, len(experiment["class_names"])
            )
            matrix += scene_matrix
            result = metrics(scene_matrix, experiment["class_names"])
            scenes.append(
                {
                    "scene": str(path),
                    "metrics": result,
                    "average_votes": float(votes.mean()),
                }
            )
            if save_predictions:
                np.save(
                    output / f"run_{run + 1}_{path.parent.name}_{path.stem}_pred.npy",
                    prediction,
                )
        result = metrics(matrix, experiment["class_names"])
        run_results.append(result)
        with open(output / f"run_{run + 1}.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "run": run + 1,
                    "seed": run_seed,
                    "metrics": result,
                    "scenes": scenes,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Run {run + 1}: {result}")

    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": checkpoint_epoch,
        "best_validation_mIoU": float(checkpoint["best_validation_mIoU"]),
        "selection_metric": checkpoint["selection_metric"],
        "variant": experiment.get("variant"),
        "model": experiment["model"],
        "train_areas": experiment["train_areas"],
        "val_areas": experiment["val_areas"],
        "test_areas": experiment["test_areas"],
        "run_seeds": run_seeds,
        "mean": average(run_results, experiment["class_names"]),
        "runs": run_results,
    }
    with open(output / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(summary["mean"])


if __name__ == "__main__":
    main()

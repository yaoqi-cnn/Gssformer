import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import available_variants, get_experiment
from dataset import (
    SpatialBlockDataset,
    load_attributes,
    load_raw_scene,
    make_loader,
    scene_files,
)
from evaluate import confusion, infer, metrics


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment(features, xyz, scale_min, scale_max, jitter_std):

    batch_size = xyz.shape[0]
    device = xyz.device
    theta = torch.rand(batch_size, device=device) * (2.0 * math.pi)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    zeros, ones = torch.zeros_like(cos_t), torch.ones_like(cos_t)
    rotation = torch.stack(
        (
            cos_t, -sin_t, zeros,
            sin_t, cos_t, zeros,
            zeros, zeros, ones,
        ),
        dim=-1,
    ).reshape(batch_size, 3, 3)

    xyz = torch.bmm(xyz, rotation.transpose(1, 2))
    feature_xyz = torch.bmm(features[:, :, :3], rotation.transpose(1, 2))
    feature_normals = torch.bmm(
        features[:, :, 6:9], rotation.transpose(1, 2)
    )
    features = torch.cat(
        (
            feature_xyz,
            features[:, :, 3:6],
            feature_normals,
            features[:, :, 9:],
        ),
        dim=-1,
    )

    flip_x = (torch.rand(batch_size, 1, 1, device=device) > 0.5).float() * 2 - 1
    flip_y = (torch.rand(batch_size, 1, 1, device=device) > 0.5).float() * 2 - 1
    xyz = xyz.clone()
    xyz[:, :, 0:1] *= flip_x
    xyz[:, :, 1:2] *= flip_y
    feature_xyz = features[:, :, :3].clone()
    feature_xyz[:, :, 0:1] *= flip_x
    feature_xyz[:, :, 1:2] *= flip_y
    feature_normals = features[:, :, 6:9].clone()
    feature_normals[:, :, 0:1] *= flip_x
    feature_normals[:, :, 1:2] *= flip_y
    features = torch.cat(
        (
            feature_xyz,
            features[:, :, 3:6],
            feature_normals,
            features[:, :, 9:],
        ),
        dim=-1,
    )

    scale = scale_min + (scale_max - scale_min) * torch.rand(
        batch_size, 1, 1, device=device
    )
    xyz = xyz * scale
    features = torch.cat(
        (features[:, :, :3] * scale, features[:, :, 3:]), dim=-1
    )

    noise = torch.randn_like(xyz) * jitter_std
    xyz = xyz + noise
    features = torch.cat(
        (features[:, :, :3] + noise, features[:, :, 3:]), dim=-1
    )
    return features, xyz


def edge_loss(edge_logits, targets, maximum):
    if len(edge_logits) != len(targets):
        raise RuntimeError(
            f"Boundary prediction/target count mismatch: "
            f"{len(edge_logits)} vs {len(targets)}"
        )
    losses = []
    for prediction, target in zip(edge_logits, targets):
        target = target.float().flatten()
        prediction = prediction.flatten()
        if prediction.numel() != target.numel():
            raise RuntimeError(
                f"Boundary prediction/target length mismatch: "
                f"{prediction.numel()} vs {target.numel()}"
            )
        positive = target.sum().clamp(min=1)
        negative = (1.0 - target).sum().clamp(min=1)
        weight = (negative / positive).clamp(max=maximum)
        losses.append(
            F.binary_cross_entropy_with_logits(
                prediction, target, pos_weight=weight
            )
        )
    return torch.stack(losses).mean() if losses else None


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    experiment,
    validation_metrics,
    best_validation_miou,
    best_epoch,
    checkpoint_role,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "experiment": experiment,
            "validation_metrics": validation_metrics,
            "best_validation_mIoU": float(best_validation_miou),
            "best_epoch": int(best_epoch),
            "selection_metric": "validation_mIoU",
            "checkpoint_role": str(checkpoint_role),
            "rng_state": capture_rng_state(),
        },
        path,
    )


def restore_rng_state(checkpoint_or_state):
    state = checkpoint_or_state.get("rng_state", checkpoint_or_state)
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def validate(
    model,
    paths,
    block_size,
    stride,
    device,
    class_names,
):
    classes = len(class_names)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for path in paths:
        xyz, rgb, labels = load_raw_scene(path)
        normals, color_gradient, _ = load_attributes(path, xyz, rgb, None)
        prediction, _ = infer(
            model,
            xyz,
            rgb,
            normals,
            color_gradient,
            block_size,
            stride,
            device,
            classes,
        )
        matrix += confusion(prediction, labels, classes)
    return metrics(matrix, class_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("village", "sensaturban"), required=True)
    parser.add_argument("--variant", choices=available_variants(), default="gssformer")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for the paper configuration")

    experiment = get_experiment(args.dataset, args.data_root, args.variant)
    training = experiment["training"]
    if args.num_workers is not None:
        training["num_workers"] = args.num_workers
    experiment["seed"] = args.seed
    experiment["checkpoint_selection"] = {
        "metric": "validation_mIoU",
        "mode": "max",
        "validation_seed": int(training["validation_seed"]),
    }
    set_seed(args.seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "experiment.json", "w", encoding="utf-8") as file:
        json.dump(experiment, file, ensure_ascii=False, indent=2)

    dataset = SpatialBlockDataset(
        experiment["data_root"],
        experiment["train_areas"],
        experiment["class_names"],
        experiment["block_size"],
        training["num_points"],
        training["minimum_block_points"],
        training["block_sampling_attempts"],
    )
    loader = make_loader(
        dataset,
        training["num_workers"],
        True,
        args.seed,
    )
    validation_paths = scene_files(
        experiment["data_root"],
        experiment["val_areas"],
    )

    from models import create_model

    model = create_model(
        num_classes=len(experiment["class_names"]),
        **experiment["model"],
    ).to(args.device)
    print(f"Variant={args.variant}")
    print(f"Parameters={sum(parameter.numel() for parameter in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    accumulation = max(1, int(training["gradient_accumulation"]))
    epochs = int(training["epochs"])
    steps_per_epoch = math.ceil(len(loader) / accumulation)

    def factor(step):
        current_epoch = step / max(steps_per_epoch, 1)
        if current_epoch < training["warmup_epochs"]:
            return max(
                training["minimum_lr_factor"],
                current_epoch / max(training["warmup_epochs"], 1),
            )
        progress = (
            current_epoch - training["warmup_epochs"]
        ) / max(1, epochs - training["warmup_epochs"])
        progress = min(max(progress, 0.0), 1.0)
        return training["minimum_lr_factor"] + (
            1.0 - training["minimum_lr_factor"]
        ) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    start_epoch = 1
    history = []
    best_validation_miou = float("-inf")
    best_epoch = 0
    validation_metrics = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        saved_experiment = checkpoint.get("experiment", {})
        saved_model = saved_experiment.get("model")
        if saved_model is not None and saved_model != experiment["model"]:
            raise ValueError(
                "The resume checkpoint model configuration does not match "
                "the requested variant."
            )
        for split_key in ("train_areas", "val_areas", "test_areas"):
            saved_split = saved_experiment.get(split_key)
            if saved_split is not None and saved_split != experiment[split_key]:
                raise ValueError(
                    f"The resume checkpoint {split_key} does not match the "
                    "current fixed data split."
                )
        saved_training = saved_experiment.get("training")
        if saved_training is not None and int(saved_training.get("epochs", epochs)) != epochs:
            raise ValueError(
                "The resume checkpoint was not created under the 300-epoch "
                "paper protocol. Restart training from scratch."
            )
        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        if checkpoint_epoch >= epochs:
            raise ValueError(
                "The resume checkpoint has already reached the 300-epoch "
                "paper protocol."
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_rng_state(checkpoint)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_miou = float(
            checkpoint.get("best_validation_mIoU", float("-inf"))
        )
        best_epoch = int(checkpoint.get("best_epoch", 0))
        validation_metrics = checkpoint.get("validation_metrics")
        history_path = output / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        segmentation_total = 0.0
        boundary_total = 0.0
        batches = 0
        remainder = len(loader) % accumulation
        last_group_start = len(loader) - remainder if remainder else len(loader)

        for step, (features, xyz, labels, boundary) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        ):
            features = features.to(args.device, non_blocking=True)
            xyz = xyz.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            boundary = boundary.to(args.device, non_blocking=True)
            features, xyz = augment(
                features,
                xyz,
                training["scale_min"],
                training["scale_max"],
                training["jitter_std"],
            )
            logits, edge_logits, boundary_targets, _, _ = model(
                features, xyz, labels, boundary=boundary
            )
            segmentation = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )
            boundary_value = edge_loss(
                edge_logits,
                boundary_targets,
                training["boundary_pos_weight_max"],
            )
            loss = segmentation
            if boundary_value is not None:
                loss = loss + training["boundary_loss_weight"] * boundary_value
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, step {step}"
                )


            divisor = (
                remainder
                if remainder and step >= last_group_start
                else accumulation
            )
            (loss / divisor).backward()
            total += float(loss.item())
            segmentation_total += float(segmentation.item())
            boundary_total += (
                0.0 if boundary_value is None else float(boundary_value.item())
            )
            batches += 1

            is_group_end = (step + 1) % accumulation == 0
            is_last_batch = step + 1 == len(loader)
            if is_group_end or is_last_batch:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training["gradient_clip"]
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        if any(
            not torch.isfinite(parameter).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError(
                f"Non-finite model parameter after epoch {epoch}"
            )

        rng_state = capture_rng_state()
        try:
            set_seed(int(training["validation_seed"]))
            validation_metrics = validate(
                model,
                validation_paths,
                experiment["block_size"],
                experiment["stride"],
                args.device,
                experiment["class_names"],
            )
        finally:
            restore_rng_state(rng_state)
        model.train()

        validation_miou = float(validation_metrics["mIoU"])
        if not np.isfinite(validation_miou):
            raise FloatingPointError(
                f"Non-finite validation mIoU after epoch {epoch}"
            )
        is_best = validation_miou > best_validation_miou
        if is_best:
            best_validation_miou = validation_miou
            best_epoch = epoch

        record = {
            "epoch": epoch,
            "loss": total / max(batches, 1),
            "segmentation_loss": segmentation_total / max(batches, 1),
            "boundary_loss": boundary_total / max(batches, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_mIoU": validation_miou,
            "validation_OA": float(validation_metrics["OA"]),
            "best_validation_mIoU": best_validation_miou,
            "best_epoch": best_epoch,
        }
        history.append(record)
        with open(output / "history.json", "w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)
        if is_best:
            save_checkpoint(
                output / "best_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                experiment,
                validation_metrics,
                best_validation_miou,
                best_epoch,
                "best_validation",
            )
        save_checkpoint(
            output / "last_model.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            experiment,
            validation_metrics,
            best_validation_miou,
            best_epoch,
            "last",
        )
        print(record)

    save_checkpoint(
        output / "final_model.pth",
        model,
        optimizer,
        scheduler,
        epochs,
        experiment,
        validation_metrics,
        best_validation_miou,
        best_epoch,
        "final",
    )
    print(
        {
            "best_epoch": best_epoch,
            "best_validation_mIoU": best_validation_miou,
            "checkpoint": str(output / "best_model.pth"),
        }
    )


if __name__ == "__main__":
    main()

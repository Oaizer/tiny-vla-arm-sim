"""Train and score the first small B601 behavior-cloning policy."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from tinyvla.b601_bc import (VOCAB, TinyB601Policy, load_samples, regression_metrics, tensors,
                             normalize_cartesian_actions, denormalize_cartesian_actions,
                             joint_targets_to_cartesian, encode_visual_targets)  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts" / "b601-demos-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "b601-bc-v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--spatial-size", type=int, choices=(4, 8, 12), default=8,
                        help="vision feature grid width/height; use 12 with 384-pixel images")
    parser.add_argument("--action-mode", choices=("joint", "cartesian"), default="joint",
                        help="predict joint targets or TCP xyz, wrist yaw, and gripper width")
    parser.add_argument("--successful-only", action="store_true",
                        help="train only on episodes where the teacher completed all three placements")
    parser.add_argument("--extra-training-dataset", type=Path,
                        help="append this dataset's training split and keep validation from the primary dataset")
    parser.add_argument("--visual-target-features", action="store_true",
                        help="add target cube and pad centers read from camera color pixels")
    parser.add_argument("--grasp-phase-weight", type=float, default=1.0,
                        help="relative loss weight for descend, close, and lift phases")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.grasp_phase_weight <= 0:
        parser.error("epochs, batch size, and grasp-phase weight must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    dataset_manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    dataset = load_samples(args.dataset, successful_only=args.successful_only, action_mode=args.action_mode)
    image_size = int(dataset_manifest.get("image_size", 96))
    retained = [episode for episode in dataset_manifest["episodes"]
                if not args.successful_only or episode["success"]]
    training_arrays = dataset["train"]
    extra_training_episodes = 0
    if args.extra_training_dataset:
        extra_manifest = json.loads((args.extra_training_dataset / "manifest.json").read_text(encoding="utf-8"))
        if int(extra_manifest.get("image_size", image_size)) != image_size:
            parser.error("extra training dataset image size must match the primary dataset")
        extra = load_samples(args.extra_training_dataset, action_mode=args.action_mode)
        training_arrays = tuple(np.concatenate((training_arrays[index], extra["train"][index]), axis=0)
                                for index in range(len(training_arrays)))
        extra_training_episodes = sum(episode["split"] == "train" for episode in extra_manifest["episodes"])
    train = tensors(training_arrays, device)
    validation = tensors(dataset["validation"], device)
    train_visual = (torch.from_numpy(encode_visual_targets(training_arrays[0], training_arrays[2])).to(device)
                    if args.visual_target_features else None)
    validation_visual = (torch.from_numpy(encode_visual_targets(dataset["validation"][0],
                                                                 dataset["validation"][2])).to(device)
                        if args.visual_target_features else None)
    action_dim = 6 if args.action_mode == "cartesian" else 8
    visual_feature_dim = 8 if args.visual_target_features else 0
    model = TinyB601Policy(spatial_size=args.spatial_size, action_dim=action_dim,
                           visual_feature_dim=visual_feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss(reduction="none")
    phase_weights = torch.ones(8, dtype=torch.float32, device=device)
    phase_weights[1:4] = args.grasp_phase_weight
    phase_weights /= phase_weights.mean()
    best_validation = float("inf")
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(train[0].shape[0])
        train_losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size].to(device)
            prediction = model(train[0][indices].float() / 255,
                               train[1][indices], train[2][indices], train[3][indices],
                               train_visual[indices] if train_visual is not None else None)
            per_sample = loss_function(prediction, train[4][indices]).mean(dim=1)
            loss = (per_sample * phase_weights[train[3][indices]]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        validation_predictions = []
        validation_losses = []
        with torch.no_grad():
            for start in range(0, validation[0].shape[0], args.batch_size):
                stop = start + args.batch_size
                prediction = model(validation[0][start:stop].float() / 255,
                                   validation[1][start:stop], validation[2][start:stop],
                                   validation[3][start:stop],
                                   validation_visual[start:stop] if validation_visual is not None else None)
                validation_predictions.append(prediction.cpu().numpy())
                per_sample = loss_function(prediction, validation[4][start:stop]).mean(dim=1)
                weights = phase_weights[validation[3][start:stop]]
                validation_losses.append((float(per_sample.mean()),
                                          float((per_sample * weights).mean()), prediction.shape[0]))
        validation_loss = sum(loss * count for loss, _, count in validation_losses) / sum(
            count for _, _, count in validation_losses)
        selection_loss = sum(loss * count for _, loss, count in validation_losses) / sum(
            count for _, _, count in validation_losses)
        if args.action_mode == "cartesian":
            predicted = denormalize_cartesian_actions(np.concatenate(validation_predictions))
            expected = denormalize_cartesian_actions(validation[4].cpu().numpy())
            error = np.abs(predicted - expected)
            error[:, 3] = np.abs((predicted[:, 3] - expected[:, 3] + np.pi) % (2 * np.pi) - np.pi)
            metrics = {"xyz_mae_m": error[:, :3].mean(axis=0).tolist(),
                       "yaw_mae_rad": float(error[:, 3].mean()),
                       "gripper_width_mae_m": float(error[:, 4].mean())}
        else:
            metrics = regression_metrics(np.concatenate(validation_predictions), validation[4].cpu().numpy())
        row = {"epoch": epoch + 1, "train_weighted_smooth_l1": float(np.mean(train_losses)),
               "validation_smooth_l1": validation_loss,
               "validation_weighted_smooth_l1": selection_loss, **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if selection_loss < best_validation:
            best_validation = selection_loss
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1,
                        "spatial_size": args.spatial_size,
                        "phase_conditioned": True,
                        "action_mode": args.action_mode,
                        "visual_feature_dim": visual_feature_dim,
                        "visual_yaw_convention": "world" if visual_feature_dim else None,
                        "grasp_phase_weight": args.grasp_phase_weight,
                        "camera_view": dataset_manifest.get("camera_view", "oblique"),
                        "image_size": image_size,
                        "vocabulary": {"<pad>": 0, **VOCAB}},
                       args.output / "policy.pt")
    checkpoint = torch.load(args.output / "policy.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(validation[0][:1].float() / 255, validation[1][:1], validation[2][:1], validation[3][:1],
                  validation_visual[:1] if validation_visual is not None else None)
        inference_start = time.perf_counter()
        for _ in range(100):
            model(validation[0][:1].float() / 255, validation[1][:1], validation[2][:1], validation[3][:1],
                  validation_visual[:1] if validation_visual is not None else None)
        latency_ms = (time.perf_counter() - inference_start) * 10
        validation_prediction_parts = []
        for start in range(0, validation[0].shape[0], args.batch_size):
            stop = start + args.batch_size
            validation_prediction_parts.append(model(
                validation[0][start:stop].float() / 255,
                validation[1][start:stop], validation[2][start:stop],
                validation[3][start:stop],
                validation_visual[start:stop] if validation_visual is not None else None).cpu().numpy())
        validation_prediction = np.concatenate(validation_prediction_parts)
    if args.action_mode == "cartesian":
        predicted = denormalize_cartesian_actions(validation_prediction)
        expected = denormalize_cartesian_actions(validation[4].cpu().numpy())
        error = np.abs(predicted - expected)
        error[:, 3] = np.abs((predicted[:, 3] - expected[:, 3] + np.pi) % (2 * np.pi) - np.pi)
        final_metrics = {"xyz_mae_m": error[:, :3].mean(axis=0).tolist(),
                         "yaw_mae_rad": float(error[:, 3].mean()),
                         "gripper_width_mae_m": float(error[:, 4].mean())}
    else:
        final_metrics = regression_metrics(validation_prediction, validation[4].cpu().numpy())
    report = {
        "model": "TinyB601Policy image CNN + instruction embeddings + joint/gripper-state MLP",
        "vision_spatial_size": args.spatial_size,
        "action_mode": args.action_mode,
        "visual_target_features": args.visual_target_features,
        "visual_yaw_convention": "world" if args.visual_target_features else None,
        "grasp_phase_weight": args.grasp_phase_weight,
        "checkpoint_selection_metric": "phase-weighted validation Smooth L1; phases 1-3 are descend, close, and lift",
        "task": "three-cube open-table color sorting waypoint imitation",
        "dataset": str(args.dataset),
        "training_samples": int(train[0].shape[0]),
        "validation_samples": int(validation[0].shape[0]),
        "training_episodes": sum(episode["split"] == "train" for episode in retained) + extra_training_episodes,
        "extra_training_dataset": str(args.extra_training_dataset) if args.extra_training_dataset else None,
        "validation_episodes": sum(episode["split"] == "validation" for episode in retained),
        "dataset_teacher_successful_episodes": dataset_manifest["successful_episodes"],
        "dataset_teacher_success_rate": dataset_manifest["successful_episodes"] / dataset_manifest["episode_count"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes": (args.output / "policy.pt").stat().st_size,
        "best_epoch": checkpoint["epoch"],
        "validation_waypoint_metrics": final_metrics,
        "inference_latency_ms_cpu_batch1": round(latency_ms, 3),
        "task_success_evaluated": False,
        "training_episodes_successful_only": args.successful_only,
        "training_seconds": round(time.perf_counter() - started, 2),
        "history": history,
    }
    (args.output / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()

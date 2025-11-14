import os
import json
import torch
import torch.nn as nn
import numpy as np
import logging
from datetime import datetime
from train import train_epoch, evaluate
from torch.utils.data import ConcatDataset
from transformers import ViTForImageClassification
from torchvision import models


def run_lodo(model_fn, CFG, logger, dataset_key, domains, loaders, optimizer_fn, device, ckpt_root, log_dir, epochs):

    lodo_results = {}

    for target_domain in domains:
        source_domains = [d for d in domains if d != target_domain]

        model = model_fn(CFG, dataset_key).to(device)
        optimizer = optimizer_fn(model)

        source_datasets = [loaders[d]["train"].dataset for d in source_domains]
        combined_train_dataset = ConcatDataset(source_datasets)
        combined_train_loader = torch.utils.data.DataLoader(
            combined_train_dataset,
            batch_size=CFG["train"]["batch_size"],
            shuffle=True,
            num_workers=CFG["system"]["num_workers"]
        )

        val_loader = loaders[target_domain]["val"]

        best_acc = 0.0
        best_ckpt_path = os.path.join(ckpt_root, f"best_{target_domain}.ckpt")

        print(f"\n=== LODO: Leaving out domain '{target_domain}' ===")
        logger.info(f"=== LODO: Leaving out domain '{target_domain}' ===")

        for epoch in range(1, epochs + 1):
            train_loss, train_cls, train_grqo, train_acc = train_epoch(model, combined_train_loader, optimizer, device)
            val_loss, val_cls, val_grqo, val_acc = evaluate(model, val_loader, device)

            print(
                f"[{target_domain}] Epoch {epoch}/{epochs} | "
                f"Train - Loss: {train_loss:.4f}, Cls: {train_cls:.4f}, GRQO: {train_grqo:.4f}, Acc: {train_acc:.4f} | "
                f"Val - Loss: {val_loss:.4f}, Cls: {val_cls:.4f}, GRQO: {val_grqo:.4f}, Acc: {val_acc:.4f}"
            )
            logger.info(
                f"[{target_domain}] Epoch {epoch}/{epochs} | "
                f"Train - Loss: {train_loss:.4f}, Cls: {train_cls:.4f}, GRQO: {train_grqo:.4f}, Acc: {train_acc:.4f} | "
                f"Val - Loss: {val_loss:.4f}, Cls: {val_cls:.4f}, GRQO: {val_grqo:.4f}, Acc: {val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"[{target_domain}] New best val acc: {best_acc:.4f}")
                logger.info(f"[{target_domain}] New best val acc: {best_acc:.4f}")

        lodo_results[target_domain] = float(best_acc)
        print(f"[{target_domain}] Best Acc: {best_acc:.4f}")
        logger.info(f"[{target_domain}] Best Acc: {best_acc:.4f}")
        print("-" * 60)
        logger.info("-" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(log_dir, f"lodo_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump({"lodo_results": lodo_results, "timestamp": timestamp}, f, indent=2)

    mean_acc = float(np.mean(list(lodo_results.values())))
    print(f"LODO finished | Mean Acc: {mean_acc:.4f}")
    print(f"Summary saved to {summary_path}")
    logger.info(f"LODO finished | Mean Acc: {mean_acc:.4f} | Summary saved to {summary_path}")
    
    return lodo_results, mean_acc, summary_path


def run_baseline(model_name, CFG, logger, dataset_key, domains, loaders, optimizer_fn, device, ckpt_root=None, log_dir=None, epochs=10):
    lodo_results = {}

    for target_domain in domains:
        source_domains = [d for d in domains if d != target_domain]

        if "vit" in model_name.lower():
            print(f"Initializing ViT baseline: {model_name}")
            logger.info(f"Initializing ViT baseline: {model_name}")
            model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=CFG["datasets"][dataset_key]["num_classes"],
                ignore_mismatched_sizes=True
            ).to(device)
        elif "resnet" in model_name.lower():
            print(f"Initializing ResNet baseline: {model_name}")
            logger.info(f"Initializing ResNet baseline: {model_name}")
            if "resnet18" in model_name.lower():
                backbone = getattr(models, model_name)(
                    weights=models.ResNet18_Weights.IMAGENET1K_V1 if "18" in model_name else None
                )
            elif "resnet34" in model_name.lower():
                backbone = getattr(models, model_name)(
                    weights=models.ResNet34_Weights.IMAGENET1K_V1 if "34" in model_name else None
                )
            num_features = backbone.fc.in_features
            backbone.fc = nn.Linear(num_features, CFG["datasets"][dataset_key]["num_classes"])
            model = backbone.to(device)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        optimizer = optimizer_fn(model)
        criterion = nn.CrossEntropyLoss()

        source_datasets = [loaders[d]["train"].dataset for d in source_domains]
        combined_train_dataset = ConcatDataset(source_datasets)
        combined_train_loader = torch.utils.data.DataLoader(
            combined_train_dataset,
            batch_size=CFG["train"]["batch_size"],
            shuffle=True,
            num_workers=CFG["system"]["num_workers"]
        )

        val_loader = loaders[target_domain]["val"]
        best_val_acc = 0.0

        print(f"\n=== Baseline LODO: Leaving out domain '{target_domain}' ===")
        logger.info(f"=== Baseline LODO: Leaving out domain '{target_domain}' ===")

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss, running_corrects, running_samples = 0.0, 0, 0

            for images, labels, _ in combined_train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                running_loss += loss.item() * images.size(0)
                running_corrects += (preds == labels).sum().item()
                running_samples += labels.size(0)

            train_loss = running_loss / running_samples
            train_acc = running_corrects / running_samples

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total if total > 0 else 0.0

            print(
                f"[{target_domain}] Epoch {epoch}/{epochs} | "
                f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )
            logger.info(
                f"[{target_domain}] Epoch {epoch}/{epochs} | "
                f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            best_val_acc = max(best_val_acc, val_acc)

        lodo_results[target_domain] = float(best_val_acc)
        print(f"[{target_domain}] Best Val Acc: {best_val_acc:.4f}")
        print("-" * 60)
        logger.info(f"[{target_domain}] Best Val Acc: {best_val_acc:.4f}")
        logger.info("-" * 60)

    mean_acc = float(np.mean(list(lodo_results.values())))
    print(f"Baseline LODO ({model_name}) finished | Mean Acc: {mean_acc:.4f}")
    logger.info(f"Baseline LODO ({model_name}) finished | Mean Acc: {mean_acc:.4f}")

    return lodo_results, mean_acc

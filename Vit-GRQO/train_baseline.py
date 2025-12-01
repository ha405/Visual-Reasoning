import argparse
import yaml
import os
import json
import torch
import torch.nn as nn
import gc
from datetime import datetime
from models import get_baseline_model
from dataset import get_dataset_class
from transform import get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Training")
    parser.add_argument("--model", type=str, required=True,
                       choices=["vit-tiny", "vit-small", "resnet18", "resnet34", "resnet50"])
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["PACS", "VLCS", "OfficeHome", "RMNIST", "CMNIST", "TerraIncognita"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def run_lodo_baseline(args, cfg):
    device = torch.device(cfg["system"]["device"] if torch.cuda.is_available() else "cpu")
    
    dataset_cls = get_dataset_class(args.dataset)
    transforms_dict = get_transforms(cfg["datasets"][args.dataset])
    dataset = dataset_cls(
        cfg["datasets"][args.dataset]["root"],
        transforms_dict["train"],
        cfg["train"]["batch_size"]
    )
    
    if cfg["datasets"][args.dataset]["domains"]:
        domains = cfg["datasets"][args.dataset]["domains"]
    else:
        domains = dataset.domains
    
    loaders = {}
    for d in domains:
        loaders[d] = {
            "train": dataset.get_dataloader(d, train=True),
            "val": dataset.get_dataloader(d, train=False)
        }
    
    lodo_results = {}
    
    for target_domain in domains:
        source_domains = [d for d in domains if d != target_domain]
        
        print(f"\n{'='*60}")
        print(f"Baseline LODO: Leaving out domain '{target_domain}'")
        print(f"{'='*60}")
        
        model = get_baseline_model(args.model, args.dataset, cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss()
        
        source_datasets = [loaders[d]["train"].dataset for d in source_domains]
        combined_train_dataset = torch.utils.data.ConcatDataset(source_datasets)
        combined_train_loader = torch.utils.data.DataLoader(
            combined_train_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=cfg["system"]["num_workers"]
        )
        
        val_loader = loaders[target_domain]["val"]
        best_val_acc = 0.0
        
        for epoch in range(1, cfg["train"]["epochs"] + 1):
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
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
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
            
            print(f"[{target_domain}] Epoch {epoch}/{cfg['train']['epochs']} | "
                  f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
            
            best_val_acc = max(best_val_acc, val_acc)
        
        lodo_results[target_domain] = float(best_val_acc)
        print(f"[{target_domain}] Best Val Acc: {best_val_acc:.4f}")
        print("-" * 60)
        
        del model, optimizer, combined_train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    summary_path = os.path.join(args.output_dir, "baseline_results.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "dataset": args.dataset,
            "lodo_results": lodo_results
        }, f, indent=2)
    
    mean_acc = sum(lodo_results.values()) / len(lodo_results)
    print(f"\n{'='*60}")
    print(f"BASELINE LODO RESULTS SUMMARY")
    print(f"{'='*60}")
    for domain, acc in lodo_results.items():
        print(f"{domain:20}: {acc:.4f}")
    print(f"{'Mean Accuracy':20}: {mean_acc:.4f}")
    print(f"{'='*60}")
    print(f"\nResults saved to {summary_path}")
    
    return lodo_results, mean_acc


def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["train"]["lr"] = args.lr
    
    if args.output_dir is None:
        args.output_dir = os.path.join(
            cfg["system"]["results_dir"],
            f"{args.model}_{args.dataset}_baseline"
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    
    print(f"\n{'='*60}")
    print(f"Baseline Training")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {cfg['train']['epochs']}")
    print(f"Batch Size: {cfg['train']['batch_size']}")
    print(f"Learning Rate: {cfg['train']['lr']}")
    print(f"Output Directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    run_lodo_baseline(args, cfg)


if __name__ == "__main__":
    main()

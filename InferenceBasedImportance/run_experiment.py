# run_experiment.py
import os
import json
import torch
import numpy as np
import logging
from datetime import datetime

from configs import pacs_config as config
from data_handling.pacs_dataset import get_pacs_dataloaders
from models.pacs_net import PACSNet
from models.pruning import rebuild_pruned_model
from engine import train_model, evaluate
from analysis.importance import compute_domain_ig, save_importance_scores
from analysis.selection import select_neurons, save_selected_neurons
from utils.helpers import set_seed, setup_logging, save_json, load_json


logger = logging.getLogger(__name__)


def run_lodo_split(target_domain: str, seed: int, config):
    """
    Run a complete Leave-One-Domain-Out experiment for a specific target domain and seed.
    
    Args:
        target_domain: Domain to use as target (test set)
        seed: Random seed for reproducibility
        config: Configuration object
        
    Returns:
        results_dict: Dictionary containing all experimental results
    """
    # Setup
    experiment_dir = os.path.join(config.ARTIFACTS_DIR, "results", target_domain, f"seed_{seed}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    log_dir = experiment_dir
    logger = setup_logging(log_dir)
    
    logger.info("="*60)
    logger.info(f"Starting LODO experiment")
    logger.info(f"Target Domain: {target_domain}")
    logger.info(f"Seed: {seed}")
    logger.info("="*60)
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Load class mapping
    class_map_path = os.path.join(config.ARTIFACTS_DIR, "class_map.json")
    if os.path.exists(class_map_path):
        class_map = load_json(class_map_path)
    else:
        # Default PACS class mapping
        class_map = {
            'dog': 0,
            'elephant': 1,
            'giraffe': 2,
            'guitar': 3,
            'horse': 4,
            'house': 5,
            'person': 6
        }
        save_json(class_map, class_map_path)
    
    # Prepare domains
    source_domains = [d for d in config.DOMAINS if d != target_domain]
    logger.info(f"Source domains: {source_domains}")
    
    # Get dataloaders
    logger.info("Loading datasets...")
    train_loader, target_loader, ig_loaders = get_pacs_dataloaders(
        config, source_domains, target_domain, class_map
    )
    
    # Initialize results dictionary
    results = {
        'target_domain': target_domain,
        'seed': seed,
        'source_domains': source_domains,
        'timestamp': datetime.now().isoformat()
    }
    
    # ============================================================================
    # 1. ERM BASELINE
    # ============================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: ERM Baseline Training")
    logger.info("="*60)
    
    # Create model
    erm_model = PACSNet(num_classes=config.NUM_CLASSES, hidden_sizes=config.HIDDEN_SIZES)
    
    # Train ERM model
    phase1_model_path = os.path.join(experiment_dir, "phase1_model.pth")
    erm_val_acc = train_model(
        erm_model,
        train_loader,
        target_loader,
        config,
        num_epochs=config.PHASE1_EPOCHS,
        learning_rate=config.PHASE1_LR,
        save_path=phase1_model_path,
        phase_name="Phase 1"
    )
    
    # Evaluate ERM on target domain
    loss_fn = torch.nn.CrossEntropyLoss()
    erm_test_loss, erm_test_acc = evaluate(
        erm_model, target_loader, loss_fn, config.DEVICE, config.USE_AMP
    )
    
    logger.info(f"ERM Test Accuracy: {erm_test_acc:.4f}")
    results['erm'] = {
        'test_acc': erm_test_acc,
        'test_loss': erm_test_loss,
        'val_acc': erm_val_acc
    }
    
    # ============================================================================
    # 2. RANDOM PRUNING BASELINE
    # ============================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 2A: Random Pruning Baseline")
    logger.info("="*60)
    
    # Load trained model
    erm_model.load_state_dict(torch.load(phase1_model_path))
    
    # Random selection
    total_neurons = erm_model.get_total_mlp_neurons()
    n_keep = int(total_neurons * config.PRUNING_TARGET_PROPORTION)
    
    np.random.seed(seed)  # Ensure reproducibility
    random_indices = sorted(np.random.choice(total_neurons, n_keep, replace=False).tolist())
    
    logger.info(f"Randomly selected {len(random_indices)}/{total_neurons} neurons")
    
    # Rebuild model with random pruning
    random_pruned_model = rebuild_pruned_model(erm_model, random_indices, config)
    
    # Fine-tune random pruned model
    random_val_acc = train_model(
        random_pruned_model,
        train_loader,
        target_loader,
        config,
        num_epochs=config.PHASE2_EPOCHS,
        learning_rate=config.PHASE2_LR,
        phase_name="Phase 2 (Random)"
    )
    
    # Evaluate random pruned model
    random_test_loss, random_test_acc = evaluate(
        random_pruned_model, target_loader, loss_fn, config.DEVICE, config.USE_AMP
    )
    
    logger.info(f"Random Pruning Test Accuracy: {random_test_acc:.4f}")
    results['random_pruning'] = {
        'test_acc': random_test_acc,
        'test_loss': random_test_loss,
        'val_acc': random_val_acc,
        'n_neurons_kept': len(random_indices)
    }
    
    # ============================================================================
    # 3. IG-BASED PRUNING
    # ============================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 2B: IG-based Pruning")
    logger.info("="*60)
    
    # Reload trained model
    erm_model.load_state_dict(torch.load(phase1_model_path))
    
    # Compute importance scores for each source domain
    logger.info("\nComputing Integrated Gradients...")
    domain_importances = {}
    
    for domain in source_domains:
        logger.info(f"\nProcessing domain: {domain}")
        importance_scores = compute_domain_ig(
            erm_model,
            ig_loaders[domain],
            domain,
            config.DEVICE,
            config
        )
        domain_importances[domain] = importance_scores
    
    # Save importance scores
    importance_path = os.path.join(experiment_dir, "domain_importances.npz")
    save_importance_scores(domain_importances, importance_path)
    
    # Select neurons based on importance
    logger.info("\nSelecting neurons...")
    selected_indices = select_neurons(domain_importances, total_neurons, config)
    
    # Save selected neurons
    selected_path = os.path.join(experiment_dir, "selected_neurons.json")
    save_selected_neurons(selected_indices, selected_path)
    
    logger.info(f"Selected {len(selected_indices)}/{total_neurons} neurons using {config.SELECTION_METHOD}")
    
    # Rebuild pruned model
    ig_pruned_model = rebuild_pruned_model(erm_model, selected_indices, config)
    
    # Save pruned model before fine-tuning
    pruned_model_path = os.path.join(experiment_dir, "pruned_model.pth")
    torch.save(ig_pruned_model.state_dict(), pruned_model_path)
    
    # Fine-tune IG pruned model
    ig_val_acc = train_model(
        ig_pruned_model,
        train_loader,
        target_loader,
        config,
        num_epochs=config.PHASE2_EPOCHS,
        learning_rate=config.PHASE2_LR,
        phase_name="Phase 2 (IG)"
    )
    
    # Evaluate IG pruned model
    ig_test_loss, ig_test_acc = evaluate(
        ig_pruned_model, target_loader, loss_fn, config.DEVICE, config.USE_AMP
    )
    
    logger.info(f"IG Pruning Test Accuracy: {ig_test_acc:.4f}")
    results['ig_pruning'] = {
        'test_acc': ig_test_acc,
        'test_loss': ig_test_loss,
        'val_acc': ig_val_acc,
        'n_neurons_kept': len(selected_indices),
        'selection_method': config.SELECTION_METHOD
    }
    
    # ============================================================================
    # 4. SUMMARY
    # ============================================================================
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    logger.info(f"Target Domain: {target_domain}")
    logger.info(f"Seed: {seed}")
    logger.info(f"ERM Accuracy: {erm_test_acc:.4f}")
    logger.info(f"Random Pruning Accuracy: {random_test_acc:.4f}")
    logger.info(f"IG Pruning Accuracy: {ig_test_acc:.4f}")
    logger.info(f"IG vs ERM: {ig_test_acc - erm_test_acc:+.4f}")
    logger.info(f"IG vs Random: {ig_test_acc - random_test_acc:+.4f}")
    logger.info("="*60)
    
    # Save all results
    summary_path = os.path.join(experiment_dir, "run_summary.json")
    save_json(results, summary_path)
    
    return results


if __name__ == "__main__":
    # Example: run single experiment
    import sys
    
    if len(sys.argv) > 2:
        target_domain = sys.argv[1]
        seed = int(sys.argv[2])
    else:
        target_domain = "sketch"
        seed = 42
    
    results = run_lodo_split(target_domain, seed, config)
    print(f"\nExperiment completed. Results saved to artifacts/results/{target_domain}/seed_{seed}/")

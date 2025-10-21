"""
Main training script for Domain Generalization with Importance-Based Weight Masking
Run this script to execute the complete two-phase training procedure on PACS dataset

Usage:
    python main.py
"""

import os
import torch
import numpy as np
from config_file import *
from dataset_file import PACSDataset, get_transform
from models_file import DomainGeneralizationModel
from importance_file import HessianComputer, compute_obd_saliency, aggregate_obd_saliencies, compute_mask_statistics
from trainer_phase1_file import run_phase1_all_domains
from trainer_phase2_file import run_phase2_lodo
from visualization_file import (
    visualize_phase1_domain_obd, 
    plot_training_curves, 
    plot_lodo_results,
    plot_phase2_all_training_curves,
    create_summary_report,
    visualize_weight_matrices_grid,
    visualize_phase1_domain
)


def setup_directories():
    """Create necessary directories for saving results"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(IMPORTANCE_DIR, exist_ok=True)
    print(f"Created output directories in {RESULTS_DIR}")


def main():
    """Main training pipeline"""
    
    print("\n" + "="*100)
    print("DOMAIN GENERALIZATION WITH IMPORTANCE-BASED WEIGHT MASKING")
    print("="*100)
    print(f"\nUsing device: {DEVICE}")
    print(f"Random seed: {SEED}")
    
    # Setup directories
    setup_directories()
    
    # Initialize dataset handler
    print(f"\nInitializing dataset from: {DATA_ROOT}")
    transform = get_transform()
    dataset_handler = PACSDataset(DATA_ROOT, DOMAINS, transform)
    print(f"Domains: {DOMAINS}")
    print(f"Number of classes: {NUM_CLASSES}")
    
    # Storage for all results
    all_phase1_results = {}  # {test_domain: {train_domain: results}}
    all_phase2_results = {}  # {test_domain: results}
    
    # ============================================================================
    # Run all LODO configurations
    # ============================================================================
    for test_domain in DOMAINS:
        print("\n" + "="*100)
        print(f"LODO CONFIGURATION: Test Domain = {test_domain}")
        print("="*100)
        
        # Get training domains (all except test domain)
        train_domains = [d for d in DOMAINS if d != test_domain]
        
        # ========================================================================
        # PHASE 1: Domain-Specific Training
        # ========================================================================
        print(f"\n>>> PHASE 1: Training on domains {train_domains}")
        
        phase1_results = run_phase1_all_domains(
            dataset_handler=dataset_handler,
            train_domains=train_domains,
            save_results=True
        )
        
        all_phase1_results[test_domain] = phase1_results
        
        # Visualize Phase 1 results for each domain
        print("\n>>> Generating Phase 1 visualizations...")
        phase1_plot_dir = f"{PLOTS_DIR}/phase1_{test_domain}_holdout"
        os.makedirs(phase1_plot_dir, exist_ok=True)
        
        for domain_name, domain_results in phase1_results.items():
            visualize_phase1_domain(
                domain_results=domain_results,
                domain_name=domain_name,
                save_dir=phase1_plot_dir
            )
        
        # Plot Phase 1 training curves
        for domain_name, domain_results in phase1_results.items():
            if domain_name != 'aggregated_saliencies':  # Skip the aggregated entry
                plot_training_curves(
                    train_losses=domain_results['train_losses'],
                    train_accs=domain_results['train_accuracies'],
                    val_accs=None,
                    title=f'Phase 1 Training: {domain_name}',
                    save_path=f"{phase1_plot_dir}/training_{domain_name}.png"
                )
        
        # ========================================================================
        # Aggregate OBD Saliencies
        # ========================================================================
        print("\n>>> Aggregating OBD saliencies by summing across domains...")
        
        # Get the saliencies dictionary from Phase 1 results
        saliencies_dict = phase1_results['aggregated_saliencies']
        
        # Aggregate using element-wise sum across domains
        summed_saliencies, normalized_saliencies, aggregated_masks = aggregate_obd_saliencies(
            saliencies_dict
        )
        
        # Visualize the aggregated OBD saliencies
        print("\n>>> Generating aggregated OBD saliency visualizations...")
        aggregated_plot_dir = f"{PLOTS_DIR}/aggregated_{test_domain}_holdout"
        os.makedirs(aggregated_plot_dir, exist_ok=True)
        
        # Visualize summed saliencies (raw values)
        from visualization_file import visualize_weight_matrices_grid
        visualize_weight_matrices_grid(
            weight_matrices=summed_saliencies,
            title=f'Aggregated OBD Saliencies (Summed) Across {train_domains}',
            save_path=f"{aggregated_plot_dir}/summed_obd_saliencies.png"
        )
        
        # Visualize normalized saliencies
        visualize_weight_matrices_grid(
            weight_matrices=normalized_saliencies,
            title=f'Normalized OBD Saliencies (0-1)',
            save_path=f"{aggregated_plot_dir}/normalized_obd_saliencies.png",
            value_range=(0, 1)
        )
        
        # Visualize binary masks
        visualize_weight_matrices_grid(
            weight_matrices=aggregated_masks,
            title=f'Binary Importance Masks from OBD (Top 60%) for Test Domain: {test_domain}',
            save_path=f"{aggregated_plot_dir}/binary_masks_obd.png",
            value_range=(0, 1)
        )
        
        # Compute and print mask statistics
        mask_stats = compute_mask_statistics(aggregated_masks)
        print("\nAggregated Mask Statistics (from OBD saliencies):")
        for layer, stats in mask_stats.items():
            if layer != 'overall':
                print(f"  {layer}: {stats['important_weights']}/{stats['total_weights']} "
                      f"({stats['percentage_important']:.1f}%) important")
        print(f"  Overall: {mask_stats['overall']['important_weights']}/"
              f"{mask_stats['overall']['total_weights']} "
              f"({mask_stats['overall']['percentage_important']:.1f}%) important")
        
        # ========================================================================
        # PHASE 2: Cross-Domain Generalization with Pruning
        # ========================================================================
        print(f"\n>>> PHASE 2: Training pruned network using OBD masks (test on {test_domain})")
        print(f"Pruning bottom {100-IMPORTANCE_THRESHOLD_PERCENTILE}% of weights, "
              f"training top {IMPORTANCE_THRESHOLD_PERCENTILE}%")
        
        phase2_results = run_phase2_lodo(
            dataset_handler=dataset_handler,
            test_domain=test_domain,
            importance_masks=aggregated_masks
        )
        
        all_phase2_results[test_domain] = phase2_results
        
        # Plot Phase 2 training curves
        plot_training_curves(
            train_losses=phase2_results['train_losses'],
            train_accs=phase2_results['train_accuracies'],
            val_accs=phase2_results['val_accuracies'],
            title=f'Phase 2 Training (Test Domain: {test_domain})',
            save_path=f"{PLOTS_DIR}/phase2_training_{test_domain}.png"
        )
        
        print(f"\n>>> Completed LODO configuration for test domain: {test_domain}")
    
    # ============================================================================
    # Visualize Final Results
    # ============================================================================
    print("\n" + "="*100)
    print("FINAL RESULTS: LEAVE-ONE-DOMAIN-OUT TEST ACCURACIES")
    print("="*100)
    
    plot_lodo_results(
        results_dict=all_phase2_results,
        save_path=f"{PLOTS_DIR}/lodo_test_accuracies.png"
    )
    
    # ============================================================================
    # Generate Summary Report
    # ============================================================================
    print("\n>>> Generating summary report...")
    
    # Flatten Phase 1 results for the report
    flattened_phase1 = {}
    for test_domain, train_results in all_phase1_results.items():
        for train_domain, results in train_results.items():
            flattened_phase1[f"{train_domain} (for {test_domain} holdout)"] = results
    
    summary = create_summary_report(
        phase1_results=flattened_phase1,
        phase2_results=all_phase2_results,
        save_path=f"{RESULTS_DIR}/summary_report.txt"
    )
    
    # ============================================================================
    # Print Detailed Results
    # ============================================================================
    print("\n" + "="*100)
    print("DETAILED PHASE 2 RESULTS")
    print("="*100)
    
    for test_domain, results in all_phase2_results.items():
        print(f"\nTest Domain: {test_domain}")
        print("-" * 60)
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  Final Train Accuracy: {results['train_accuracies'][-1]:.4f}")
        print(f"  Final Val Accuracy: {results['val_accuracies'][-1]:.4f}")
        print(f"  Final Train Loss: {results['train_losses'][-1]:.4f}")
    
    # Compute average and std
    test_accs = [results['test_accuracy'] for results in all_phase2_results.values()]
    avg_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    
    print("\n" + "="*100)
    print(f"AVERAGE TEST ACCURACY: {avg_acc:.4f} ± {std_acc:.4f}")
    print("="*100)
    
    # ============================================================================
    # Save All Results
    # ============================================================================
    print("\n>>> Saving final results...")
    
    results_save_path = f"{RESULTS_DIR}/all_results.npz"
    
    # Convert results to saveable format
    save_dict = {
        'phase2_test_accuracies': {k: v['test_accuracy'] for k, v in all_phase2_results.items()},
        'average_test_accuracy': avg_acc,
        'std_test_accuracy': std_acc
    }
    
    np.savez(results_save_path, **save_dict)
    print(f"Saved all results to {results_save_path}")
    
    print("\n" + "="*100)
    print("TRAINING COMPLETE! All results saved to ./results/")
    print("="*100)
    print("\nGenerated files:")
    print(f"  - Plots: {PLOTS_DIR}/")
    print(f"  - Importance matrices: {IMPORTANCE_DIR}/")
    print(f"  - Summary report: {RESULTS_DIR}/summary_report.txt")
    print(f"  - Results archive: {RESULTS_DIR}/all_results.npz")
    print("\n")


if __name__ == "__main__":
    main()
# main.py
import os
import json
import pandas as pd
from datetime import datetime
import logging

from configs import pacs_config as config
from data_handling.pacs_dataset import compute_and_cache_baselines
from run_experiment import run_lodo_split
from utils.helpers import save_json, load_json


def main():
    """
    Main entry point for running all LODO experiments across seeds and domains.
    """
    print("\n" + "="*80)
    print("PACS DOMAIN GENERALIZATION WITH NEURON PRUNING")
    print("="*80)
    print(f"Starting experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Domains: {config.DOMAINS}")
    print(f"  - Seeds: {config.RANDOM_SEEDS}")
    print(f"  - Pruning Target: {config.PRUNING_TARGET_PROPORTION:.1%}")
    print(f"  - Selection Method: {config.SELECTION_METHOD}")
    print(f"  - Phase 1 Epochs: {config.PHASE1_EPOCHS}")
    print(f"  - Phase 2 Epochs: {config.PHASE2_EPOCHS}")
    print("="*80)
    
    # Create base artifact directories
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.ARTIFACTS_DIR, "baselines"), exist_ok=True)
    os.makedirs(os.path.join(config.ARTIFACTS_DIR, "results"), exist_ok=True)
    
    # Load or create class mapping
    class_map_path = os.path.join(config.ARTIFACTS_DIR, "class_map.json")
    if not os.path.exists(class_map_path):
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
        print(f"Created class mapping: {class_map_path}")
    else:
        class_map = load_json(class_map_path)
        print(f"Loaded existing class mapping from {class_map_path}")
    
    # Compute and cache baselines for IG
    print("\nComputing baseline tensors for Integrated Gradients...")
    compute_and_cache_baselines(config, class_map)
    
    # Initialize results collector
    all_results = []
    
    # Run experiments
    total_experiments = len(config.RANDOM_SEEDS) * len(config.DOMAINS)
    experiment_count = 0
    
    for seed in config.RANDOM_SEEDS:
        for target_domain in config.DOMAINS:
            experiment_count += 1
            
            print(f"\n" + "-"*80)
            print(f"Experiment {experiment_count}/{total_experiments}")
            print(f"Target Domain: {target_domain} | Seed: {seed}")
            print("-"*80)
            
            try:
                # Run LODO split
                results = run_lodo_split(target_domain, seed, config)
                
                # Collect results for aggregation
                for method in ['erm', 'random_pruning', 'ig_pruning']:
                    if method in results:
                        all_results.append({
                            'target_domain': target_domain,
                            'seed': seed,
                            'method': method,
                            'test_acc': results[method]['test_acc'],
                            'test_loss': results[method]['test_loss']
                        })
                
                print(f"✓ Completed: {target_domain} (seed {seed})")
                
            except Exception as e:
                print(f"✗ Failed: {target_domain} (seed {seed})")
                print(f"  Error: {str(e)}")
                logging.error(f"Experiment failed for {target_domain} seed {seed}: {str(e)}", exc_info=True)
                continue
    
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)
    
    if all_results:
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Calculate statistics per domain and method
        stats = df.groupby(['target_domain', 'method'])['test_acc'].agg(['mean', 'std'])
        stats = stats.round(4)
        
        print("\nPer-Domain Results (Test Accuracy):")
        print(stats)
        
        # Calculate overall statistics
        overall_stats = df.groupby('method')['test_acc'].agg(['mean', 'std'])
        overall_stats = overall_stats.round(4)
        
        print("\nOverall Results (Test Accuracy):")
        print(overall_stats)
        
        # Calculate improvements
        if 'erm' in df['method'].values and 'ig_pruning' in df['method'].values:
            erm_mean = df[df['method'] == 'erm']['test_acc'].mean()
            ig_mean = df[df['method'] == 'ig_pruning']['test_acc'].mean()
            random_mean = df[df['method'] == 'random_pruning']['test_acc'].mean()
            
            print("\nComparison:")
            print(f"  IG vs ERM: {ig_mean - erm_mean:+.4f}")
            print(f"  IG vs Random: {ig_mean - random_mean:+.4f}")
        
        # Save results to CSV
        results_csv_path = os.path.join(config.ARTIFACTS_DIR, "final_results.csv")
        df.to_csv(results_csv_path, index=False)
        print(f"\nDetailed results saved to: {results_csv_path}")
        
        # Save statistics
        stats_csv_path = os.path.join(config.ARTIFACTS_DIR, "final_statistics.csv")
        stats.to_csv(stats_csv_path)
        print(f"Statistics saved to: {stats_csv_path}")
        
    else:
        print("No results collected. Please check the logs for errors.")
    
    print("\n" + "="*80)
    print(f"All experiments completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()

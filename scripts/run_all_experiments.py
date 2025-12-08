#!/usr/bin/env python
import os
import sys
import glob
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Run multiple pruning experiments')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--config_dir', type=str, default='configs/experiments', help='Directory containing config files')
    parser.add_argument('--pattern', type=str, default='*.yaml', help='Pattern to match config files')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    return parser.parse_args()


def main():
    args = parse_args()
    
    config_pattern = os.path.join(args.config_dir, args.pattern)
    config_files = glob.glob(config_pattern)
    
    if not config_files:
        print(f"No config files found matching: {config_pattern}")
        return
    
    print(f"Found {len(config_files)} config files:")
    for cfg in config_files:
        print(f"  - {cfg}")
    print()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    for i, config_file in enumerate(config_files, 1):
        print("="*80)
        print(f"Running experiment {i}/{len(config_files)}: {os.path.basename(config_file)}")
        print("="*80)
        
        cmd = [
            sys.executable, 'main.py',
            '--config', config_file,
            '--data_dir', args.data_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"\n✓ Experiment {i} completed successfully\n")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Experiment {i} failed with error: {e}\n")
            continue
    
    print("="*80)
    print("All experiments completed!")
    print("="*80)


if __name__ == '__main__':
    main()

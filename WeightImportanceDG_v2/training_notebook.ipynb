{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Domain Generalization with Importance-Based Weight Masking\n",
    "## Training Pipeline for PACS Dataset\n",
    "\n",
    "This notebook runs the complete two-phase training procedure:\n",
    "- **Phase 1**: Train domain-specific networks to identify important weights\n",
    "- **Phase 2**: Train cross-domain generalization with importance-based learning rates"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setup and Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import torch\n",
    "import numpy as np\n",
    "from config import *\n",
    "from dataset import PACSDataset, get_transform\n",
    "from models import DomainGeneralizationModel\n",
    "from importance import ImportanceCalculator, aggregate_importance_masks, compute_mask_statistics\n",
    "from trainer_phase1 import run_phase1_all_domains
from trainer_phase2 import run_phase2_lodo
from visualization import (
    visualize_phase1_domain, 
    plot_training_curves, 
    plot_lodo_results,
    plot_phase2_all_training_curves,
    create_summary_report
)

# Create directories for saving results
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(IMPORTANCE_DIR, exist_ok=True)

print(f\"Using device: {DEVICE}\")
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Initialize Dataset Handler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create dataset handler with transforms\n",
    "transform = get_transform()\n",
    "dataset_handler = PACSDataset(DATA_ROOT, DOMAINS, transform)\n",
    "\n",
    "print(f\"Dataset root: {DATA_ROOT}\")\n",
    "print(f\"Domains: {DOMAINS}\")\n",
    "print(f\"Number of classes: {NUM_CLASSES}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Run All LODO Configurations\n",
    "\n",
    "We'll iterate through all 4 possible LODO configurations:\n",
    "- Test on art_painting, train on [cartoon, photo, sketch]\n",
    "- Test on cartoon, train on [art_painting, photo, sketch]\n",
    "- Test on photo, train on [art_painting, cartoon, sketch]\n",
    "- Test on sketch, train on [art_painting, cartoon, photo]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Storage for all results\n",
    "all_phase1_results = {}  # {test_domain: {train_domain: results}}\n",
    "all_phase2_results = {}  # {test_domain: results}\n",
    "\n",
    "# Run for each LODO configuration\n",
    "for test_domain in DOMAINS:\n",
    "    print(\"\\n\" + \"=\"*100)\n",
    "    print(f\"LODO CONFIGURATION: Test Domain = {test_domain}\")\n",
    "    print(\"=\"*100)\n",
    "    \n",
    "    # Get training domains (all except test domain)\n",
    "    train_domains = [d for d in DOMAINS if d != test_domain]\n",
    "    \n",
    "    # ========================================================================\n",
    "    # PHASE 1: Domain-Specific Training\n",
    "    # ========================================================================\n",
    "    print(f\"\\n>>> PHASE 1: Training on domains {train_domains}\")\n",
    "    \n",
    "    phase1_results = run_phase1_all_domains(\n",
    "        dataset_handler=dataset_handler,\n",
    "        train_domains=train_domains,\n",
    "        save_results=True\n",
    "    )\n",
    "    \n",
    "    all_phase1_results[test_domain] = phase1_results\n",
    "    \n",
    "    # Visualize Phase 1 results for each domain\n",
    "    print(\"\\n>>> Generating Phase 1 visualizations...\")\n",
    "    for domain_name, domain_results in phase1_results.items():\n",
    "        visualize_phase1_domain(\n",
    "            domain_results=domain_results,\n",
    "            domain_name=domain_name,\n",
    "            save_dir=f\"{PLOTS_DIR}/phase1_{test_domain}_holdout\"\n",
    "        )\n",
    "    \n",
    "    # Plot Phase 1 training curves\n",
    "    for domain_name, domain_results in phase1_results.items():\n",
    "        plot_training_curves(\n",
    "            train_losses=domain_results['train_losses'],\n",
    "            train_accs=domain_results['train_accuracies'],\n",
    "            val_accs=None,\n",
    "            title=f'Phase 1 Training: {domain_name}',\n",
    "            save_path=f\"{PLOTS_DIR}/phase1_{test_domain}_holdout/training_{domain_name}.png\"\n",
    "        )\n",
    "    \n",
    "    # ========================================================================\n",
    "    # Aggregate Importance Masks\n",
    "    # ========================================================================\n",
    "    print(\"\\n>>> Aggregating importance masks from training domains...\")\n",
    "    \n",
    "    # Collect binary masks from all training domains\n",
    "    mask_list = [phase1_results[d]['binary_masks'] for d in train_domains]\n",
    "    \n",
    "    # Aggregate masks (union strategy: if any domain finds it important, mark as important)\n",
    "    aggregated_masks = aggregate_importance_masks(mask_list)\n",
    "    \n",
    "    # Compute and print mask statistics\n",
    "    mask_stats = compute_mask_statistics(aggregated_masks)\n",
    "    print(\"\\nAggregated Mask Statistics:\")\n",
    "    for layer, stats in mask_stats.items():\n",
    "        if layer != 'overall':\n",
    "            print(f\"  {layer}: {stats['important_weights']}/{stats['total_weights']} \"\n",
    "                  f\"({stats['percentage_important']:.1f}%) important\")\n",
    "    print(f\"  Overall: {mask_stats['overall']['important_weights']}/\"\n",
    "          f\"{mask_stats['overall']['total_weights']} \"\n",
    "          f\"({mask_stats['overall']['percentage_important']:.1f}%) important\")\n",
    "    \n",
    "    # ========================================================================\n",
    "    # PHASE 2: Cross-Domain Generalization\n",
    "    # ========================================================================\n",
    "    print(f\"\\n>>> PHASE 2: Training with importance masks (test on {test_domain})\")\n",
    "    \n",
    "    phase2_results = run_phase2_lodo(\n",
    "        dataset_handler=dataset_handler,\n",
    "        test_domain=test_domain,\n",
    "        importance_masks=aggregated_masks\n",
    "    )\n",
    "    \n",
    "    all_phase2_results[test_domain] = phase2_results\n",
    "    \n",
    "    # Plot Phase 2 training curves\n",
    "    plot_training_curves(\n",
    "        train_losses=phase2_results['train_losses'],\n",
    "        train_accs=phase2_results['train_accuracies'],\n",
    "        val_accs=phase2_results['val_accuracies'],\n",
    "        title=f'Phase 2 Training (Test Domain: {test_domain})',\n",
    "        save_path=f\"{PLOTS_DIR}/phase2_training_{test_domain}.png\"\n",
    "    )\n",
    "    \n",
    "    print(f\"\\n>>> Completed LODO configuration for test domain: {test_domain}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualize Final Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot LODO test accuracies comparison\n",
    "print(\"\\n\" + \"=\"*100)\n",
    "print(\"FINAL RESULTS: LEAVE-ONE-DOMAIN-OUT TEST ACCURACIES\")\n",
    "print(\"=\"*100)\n",
    "\n",
    "plot_lodo_results(\n",
    "    results_dict=all_phase2_results,\n",
    "    save_path=f\"{PLOTS_DIR}/lodo_test_accuracies.png\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Generate Summary Report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create summary report combining Phase 1 and Phase 2 results\n",
    "# Note: Phase 1 results are nested by test_domain, we'll flatten for the report\n",
    "flattened_phase1 = {}\n",
    "for test_domain, train_results in all_phase1_results.items():\n",
    "    for train_domain, results in train_results.items():\n",
    "        flattened_phase1[f\"{train_domain} (for {test_domain} holdout)\"] = results\n",
    "\n",
    "summary = create_summary_report(\n",
    "    phase1_results=flattened_phase1,\n",
    "    phase2_results=all_phase2_results,\n",
    "    save_path=f\"{RESULTS_DIR}/summary_report.txt\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Print Detailed Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"\\n\" + \"=\"*100)\n",
    "print(\"DETAILED PHASE 2 RESULTS\")\n",
    "print(\"=\"*100)\n",
    "\n",
    "for test_domain, results in all_phase2_results.items():\n",
    "    print(f\"\\nTest Domain: {test_domain}\")\n",
    "    print(\"-\" * 60)\n",
    "    print(f\"  Test Accuracy: {results['test_accuracy']:.4f}\")\n",
    "    print(f\"  Final Train Accuracy: {results['train_accuracies'][-1]:.4f}\")\n",
    "    print(f\"  Final Val Accuracy: {results['val_accuracies'][-1]:.4f}\")\n",
    "    print(f\"  Final Train Loss: {results['train_losses'][-1]:.4f}\")\n",
    "\n",
    "# Compute average and std\n",
    "test_accs = [results['test_accuracy'] for results in all_phase2_results.values()]\n",
    "avg_acc = np.mean(test_accs)\n",
    "std_acc = np.std(test_accs)\n",
    "\n",
    "print(\"\\n\" + \"=\"*100)\n",
    "print(f\"AVERAGE TEST ACCURACY: {avg_acc:.4f} ± {std_acc:.4f}\")\n",
    "print(\"=\"*100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Save All Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save results as numpy archive for later analysis\n",
    "results_save_path = f\"{RESULTS_DIR}/all_results.npz\"\n",
    "\n",
    "# Convert results to saveable format\n",
    "save_dict = {\n",
    "    'phase2_test_accuracies': {k: v['test_accuracy'] for k, v in all_phase2_results.items()},\n",
    "    'average_test_accuracy': avg_acc,\n",
    "    'std_test_accuracy': std_acc\n",
    "}\n",
    "\n",
    "np.savez(results_save_path, **save_dict)\n",
    "print(f\"\\nSaved all results to {results_save_path}\")\n",
    "\n",
    "print(\"\\n\" + \"=\"*100)\n",
    "print(\"TRAINING COMPLETE! All results saved to ./results/\")\n",
    "print(\"=\"*100)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"
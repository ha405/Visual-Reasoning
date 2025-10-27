"""
Visualization utilities for weight matrices and training results
Creates circular grid visualizations for weight importance
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent pop-ups
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from config_file import *


def visualize_weight_matrices_grid(
    weight_matrices, title, save_path=None, value_range=None
):
    """
    Visualize 4 weight matrices (10x10 each) as grids of circles
    Each circle's color intensity represents the weight value

    Args:
        weight_matrices (list): List of 4 weight tensors/arrays [10, 10]
        title (str): Overall title for the figure
        save_path (str): Path to save figure (if None, display only)
        value_range (tuple): (vmin, vmax) for color normalization, or None for auto
    """
    import torch

    # Convert tensors to numpy if needed
    matrices = []
    for w in weight_matrices:
        if torch.is_tensor(w):
            matrices.append(w.cpu().numpy())
        else:
            matrices.append(np.array(w))

    # Determine value range for consistent coloring
    if value_range is None:
        all_values = np.concatenate([m.flatten() for m in matrices])
        vmin, vmax = all_values.min(), all_values.max()
    else:
        vmin, vmax = value_range

    # Create figure with 4 subplots (one per layer)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for layer_idx, (ax, matrix) in enumerate(zip(axes, matrices)):
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=14)
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-0.5, 9.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Put row 0 at top

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add grid
        for i in range(11):
            ax.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)

        # Draw circles for each weight
        for i in range(10):  # rows
            for j in range(10):  # columns
                value = matrix[i, j]

                # Normalize value to [0, 1] for color mapping
                if vmax - vmin > 1e-10:
                    normalized_value = (value - vmin) / (vmax - vmin)
                else:
                    normalized_value = 0.5

                # Color from colormap
                color = plt.cm.RdYlGn(normalized_value)

                # Draw circle
                circle = patches.Circle(
                    (j, i),
                    radius=0.4,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.add_patch(circle)

                # Add text value in circle
                text_color = "white" if normalized_value < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                    fontweight="bold",
                )

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Weight Value", fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=VIZ_DPI, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.close()  # Close figure instead of showing it


def visualize_phase1_domain_obd(domain_results, domain_name, save_dir=None):
    """
    Create Phase 1 visualizations for a single domain using OBD saliencies:
    1. Final trained weights
    2. Diagonal Hessian values
    3. OBD Saliencies (computed from weights and Hessian)

    Args:
        domain_results (dict): Results dictionary from Phase 1 for this domain
        domain_name (str): Name of the domain
        save_dir (str): Directory to save visualizations (if None, display only)
    """
    print(f"\nGenerating Phase 1 OBD visualizations for domain: {domain_name}")

    # 1. Visualize final trained weights
    visualize_weight_matrices_grid(
        weight_matrices=domain_results["final_weights"],
        title=f"{domain_name} - Final Trained Weights",
        save_path=f"{save_dir}/{domain_name}_final_weights.png" if save_dir else None,
    )

    # 2. Visualize diagonal Hessian (h_kk)
    visualize_weight_matrices_grid(
        weight_matrices=domain_results["hessian_diagonal"],
        title=f"{domain_name} - Diagonal Hessian (h_kk)",
        save_path=(
            f"{save_dir}/{domain_name}_hessian_diagonal.png" if save_dir else None
        ),
    )

    # 3. Visualize OBD saliencies
    visualize_weight_matrices_grid(
        weight_matrices=domain_results["obd_saliencies"],
        title=f"{domain_name} - OBD Saliencies (0.5 * h_kk * w²)",
        save_path=f"{save_dir}/{domain_name}_obd_saliencies.png" if save_dir else None,
    )

    print(f"Completed OBD visualizations for {domain_name}")


def visualize_phase1_domain(domain_results, domain_name, save_dir=None):
    """
    LEGACY: Create all Phase 1 visualizations for a single domain (old approach)
    This is kept for backward compatibility but uses the old weight-change method

    Args:
        domain_results (dict): Results dictionary from Phase 1 for this domain
        domain_name (str): Name of the domain
        save_dir (str): Directory to save visualizations (if None, display only)
    """
    print(f"\nGenerating Phase 1 visualizations for domain: {domain_name}")

    # 1. Visualize initial weights
    visualize_weight_matrices_grid(
        weight_matrices=domain_results["initial_weights"],
        title=f"{domain_name} - Initial Weights (Before Training)",
        save_path=f"{save_dir}/{domain_name}_initial_weights.png" if save_dir else None,
    )

    # 2. Visualize final trained weights
    visualize_weight_matrices_grid(
        weight_matrices=domain_results["final_weights"],
        title=f"{domain_name} - Final Weights (After Training)",
        save_path=f"{save_dir}/{domain_name}_final_weights.png" if save_dir else None,
    )

    # 3. Visualize continuous importance (0-1)
    visualize_weight_matrices_grid(
        weight_matrices=domain_results["normalized_importance"],
        title=f"{domain_name} - Continuous Importance (0-1)",
        save_path=(
            f"{save_dir}/{domain_name}_continuous_importance.png" if save_dir else None
        ),
        value_range=(0, 1),
    )

    # 4. Visualize binary masks (0 or 1)
    visualize_weight_matrices_grid(
        weight_matrices=domain_results["binary_masks"],
        title=f"{domain_name} - Binary Importance Masks (Top 60%)",
        save_path=f"{save_dir}/{domain_name}_binary_masks.png" if save_dir else None,
        value_range=(0, 1),
    )

    print(f"Completed visualizations for {domain_name}")


def plot_training_curves(
    train_losses, train_accs, val_accs=None, title="Training Curves", save_path=None
):
    """
    Plot training loss and accuracy curves

    Args:
        train_losses (list): Training losses per epoch
        train_accs (list): Training accuracies per epoch
        val_accs (list): Validation accuracies per epoch (optional)
        title (str): Plot title
        save_path (str): Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    epochs = range(1, len(train_losses) + 1)

    # Plot loss
    ax1.plot(epochs, train_losses, "b-", marker="o", label="Train Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, train_accs, "g-", marker="o", label="Train Accuracy")
    if val_accs is not None:
        ax2.plot(epochs, val_accs, "r-", marker="s", label="Val Accuracy")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=VIZ_DPI, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")

    plt.close()  # Close figure instead of showing it


def plot_lodo_results(results_dict, save_path=None):
    """
    Plot bar chart comparing LODO test accuracies across all test domains

    Args:
        results_dict (dict): Dictionary mapping test_domain -> results
        save_path (str): Path to save figure
    """
    domains = list(results_dict.keys())
    test_accs = [results_dict[d]["test_accuracy"] for d in domains]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        domains, test_accs, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    )

    # Add value labels on bars
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Test Domain (Held Out)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Test Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("Leave-One-Domain-Out Test Accuracies", fontsize=16, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, axis="y", alpha=0.3)

    # Add average line
    avg_acc = np.mean(test_accs)
    ax.axhline(
        avg_acc,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Average: {avg_acc:.4f}",
    )
    ax.legend(fontsize=12)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=VIZ_DPI, bbox_inches="tight")
        print(f"Saved LODO results to {save_path}")

    plt.close()  # Close figure instead of showing it


def plot_phase2_all_training_curves(all_results, save_dir=None):
    """
    Plot training curves for all LODO configurations

    Args:
        all_results (dict): Dictionary mapping test_domain -> Phase 2 results
        save_dir (str): Directory to save plots
    """
    for test_domain, results in all_results.items():
        plot_training_curves(
            train_losses=results["train_losses"],
            train_accs=results["train_accuracies"],
            val_accs=results["val_accuracies"],
            title=f"Phase 2 Training (Test Domain: {test_domain})",
            save_path=(
                f"{save_dir}/phase2_training_{test_domain}.png" if save_dir else None
            ),
        )


def create_summary_report(phase1_results, phase2_results, save_path=None):
    """
    Create a text summary report of all results

    Args:
        phase1_results (dict): Results from Phase 1
        phase2_results (dict): Results from Phase 2
        save_path (str): Path to save report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DOMAIN GENERALIZATION WITH IMPORTANCE-BASED WEIGHT MASKING")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Phase 1 Summary
    report_lines.append("PHASE 1: DOMAIN-SPECIFIC TRAINING")
    report_lines.append("-" * 80)
    for domain, results in phase1_results.items():
        final_loss = results["train_losses"][-1]
        final_acc = results["train_accuracies"][-1]
        report_lines.append(
            f"Domain: {domain:20s} | Final Loss: {final_loss:.4f} | Final Acc: {final_acc:.4f}"
        )
    report_lines.append("")

    # Phase 2 Summary
    report_lines.append("PHASE 2: LEAVE-ONE-DOMAIN-OUT GENERALIZATION")
    report_lines.append("-" * 80)
    test_accs = []
    for test_domain, results in phase2_results.items():
        test_acc = results["test_accuracy"]
        test_accs.append(test_acc)
        report_lines.append(
            f"Test Domain: {test_domain:20s} | Test Accuracy: {test_acc:.4f}"
        )

    avg_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)
    report_lines.append("")
    report_lines.append(
        f"Average Test Accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}"
    )
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)
    print(report_text)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report_text)
        print(f"\nSaved summary report to {save_path}")

    return report_text

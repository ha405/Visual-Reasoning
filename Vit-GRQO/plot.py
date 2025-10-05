import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_domainwise_accuracy(domains, baseline, grqo, 
                             title="Domain-wise Accuracy Comparison: Baseline vs GRQO",
                             save_path=None):
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.figsize": (8, 5),
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2
    })

    x = np.arange(len(domains))
    width = 0.35
    colors = sns.color_palette("mako", 2)

    # === Figure ===
    fig, ax = plt.subplots()

    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline',
                   color=colors[0], edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, grqo, width, label='GRQO',
                   color=colors[1], edgecolor='black', linewidth=0.8)

    # === Labels & Titles ===
    ax.set_ylabel('Accuracy', labelpad=8)
    ax.set_xlabel('Domain', labelpad=8)
    ax.set_title(title, pad=12, fontweight='semibold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 1.05)

    # === Annotate bars ===
    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.2f}",
                    ha='center', va='bottom', fontsize=10, fontweight='medium')

    annotate_bars(bars1)
    annotate_bars(bars2)

    # === Legend ===
    ax.legend(frameon=True, fancybox=True, shadow=False, facecolor='white',
              edgecolor='gray', loc='upper left')

    sns.despine(left=False, bottom=False)
    plt.tight_layout()

    # === Save or Show ===
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()


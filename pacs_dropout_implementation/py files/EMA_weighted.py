# %%
# =================================================================================
# SECTION 1: PROJECT SCAFFOLDING & CONFIGURATION
# =================================================================================

# ---------------------------------------------------------------------------------
# 1.1: IMPORTS
# ---------------------------------------------------------------------------------
%matplotlib inline
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import copy

# Set a seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# =================================================================================
# SECTION 1.2: CONFIGURATION CLASS (FOR EMA WEIGHT TRANSFER)
# =================================================================================
class Config:
    # --- Data Paths and Domains ---
    DATA_DIR = r"C:\Users\Haseeb\OneDrive - Higher Education Commission\sproj\PACS\pacs_data"
    DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
    
    # --- Model & Architecture ---
    MODEL_NAME = "WinKawaks/vit-tiny-patch16-224"
    NUM_CLASSES = 7
    NUM_HEADS = 4
    DROPOUT_OPTIONS = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    ### NEW: EMA Hyperparameter ###
    # Controls how much of the old student weights to keep. 0.999 is a common value.
    EMA_DECAY = 0.999
    
    # --- Training Hyperparameters ---
    BATCH_SIZE = 128
    NUM_EPOCHS = 20 # Train longer to let the EMA stabilize
    LEARNING_RATE = 1e-4
    OPTIMIZER = "AdamW"
    
    # --- Hardware & Reproducibility ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

# Instantiate the config
config = Config()

print("--- Project Configuration (EMA Weight Transfer) ---")
for key, value in config.__class__.__dict__.items():
    if not key.startswith('__'):
        print(f"{key}: {value}")
print("---------------------------")
print(f"Device: {config.DEVICE}")

# ---------------------------------------------------------------------------------
# 1.3: RESULTS TRACKER
# ---------------------------------------------------------------------------------
experiment_results = []

print("\nProject scaffolding is complete. Ready for Section 2.")

# %%
# =================================================================================
# SECTION 2: DATA LOADING & PREPROCESSING
# =================================================================================

# ---------------------------------------------------------------------------------
# 2.1: IMAGE TRANSFORMATIONS
# Define the transformations for training (with augmentation) and validation/testing.
# ---------------------------------------------------------------------------------

# The ViT model was pre-trained on images of size 224x224
IMG_SIZE = 224

# The normalization values are standard for many pre-trained models
# but it's good practice to use the ones specified by the model's authors if available.
# For ViT, a simple (0.5, 0.5, 0.5) normalization is common.
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), # A simple data augmentation technique
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
}


# %%
# ---------------------------------------------------------------------------------
# 2.2: CUSTOM PACS DATASET CLASS
# This class will read the images and labels from our specific folder structure.
# ---------------------------------------------------------------------------------
class PACSDataset(Dataset):
    def __init__(self, root_dir, domains, transform=None):
        """
        Args:
            root_dir (string): Directory with all the domain folders.
            domains (list of string): List of domains to include in this dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.domains = domains
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Discover all classes (dog, elephant, etc.) and map them to integers
        self.classes = sorted(os.listdir(os.path.join(root_dir, domains[0])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load image paths and labels from the specified domains
        for domain in self.domains:
            domain_path = os.path.join(self.root_dir, domain)
            for class_name in self.classes:
                class_path = os.path.join(domain_path, class_name)
                for img_name in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# %%
# =================================================================================
# SECTION 2.3: DATALOADER HELPER FUNCTION (NEW 80/20 SPLIT VERSION)
# =================================================================================
# Replace the old get_dataloaders function in Section 2 of BOTH notebooks with this.

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def get_dataloaders(root_dir, target_domain, all_domains, batch_size, seed):
    """
    Creates dataloaders for a LODO split using an 80/20 split on the source domains.
    """
    source_domains = [d for d in all_domains if d != target_domain]
    
    print(f"--- Creating DataLoaders (80/20 Split Strategy) ---")
    print(f"Target (Test) Domain: {target_domain}")
    print(f"Source Domains for Train/Val: {source_domains}")
    
    # 1. Create a single, large dataset by combining all source domains
    source_dataset = PACSDataset(
        root_dir=root_dir, 
        domains=source_domains, 
        transform=data_transforms['train'] # Use training transforms for the whole source
    )
    
    # We need to perform a stratified split to ensure the train and val sets
    # have a similar distribution of classes.
    indices = list(range(len(source_dataset)))
    labels = source_dataset.labels
    
    # Use sklearn's train_test_split to get indices for an 80% train / 20% val split
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=labels, 
        random_state=seed
    )
    
    # 2. Create the training and validation subsets
    train_subset = Subset(source_dataset, train_idx)
    val_subset = Subset(source_dataset, val_idx)
    
    # Important: The validation subset should not use training augmentations (like RandomFlip).
    # We create a new dataset object for validation with the correct transforms.
    # This is a cleaner way to handle transforms for subsets.
    val_dataset_clean = PACSDataset(root_dir=root_dir, domains=source_domains, transform=data_transforms['val'])
    val_subset_final = Subset(val_dataset_clean, val_idx)
    
    # 3. Create the test dataset from the full target domain
    test_dataset = PACSDataset(
        root_dir=root_dir, 
        domains=[target_domain], 
        transform=data_transforms['val']
    )

    # 4. Create the DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset_final, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Source data size: {len(source_dataset)}")
    print(f"  -> Training on: {len(train_subset)} images (80%)")
    print(f"  -> Validating on: {len(val_subset_final)} images (20%)")
    print(f"Testing on full '{target_domain}' domain: {len(test_dataset)} images")
    print("----------------------------------------------------")
    
    return train_loader, val_loader, test_loader

# %%
# =================================================================================
# SECTION 3: THE MODEL ARCHITECTURE (EMA WEIGHT TRANSFER)
# =================================================================================

class DistillationViT(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rates: list, dummy_dropout_rate: float):
        super(DistillationViT, self).__init__()
        
        self.vit_backbone = ViTModel.from_pretrained(model_name)
        hidden_dim = self.vit_backbone.config.hidden_size
        
        # The 4 "teacher" heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=rate),
                nn.Linear(hidden_dim, num_classes)
            ) for rate in dropout_rates
        ])
        
        ### CRITICAL CHANGE ###
        # The "student" head MUST have the same architecture as the teachers
        # so that their weights can be averaged.
        self.dummy_head = nn.Sequential(
            nn.Dropout(p=dummy_dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        # We initialize the dummy head with the same weights as the first teacher head
        self.dummy_head.load_state_dict(self.heads[0].state_dict())


    def update_dropout_rates(self, new_rates: list):
        for i, head in enumerate(self.heads):
            head[0].p = new_rates[i]
            
    def update_dummy_dropout_rate(self, new_rate: float):
        self.dummy_head[0].p = new_rate
        
    def forward(self, images):
        z = self.vit_backbone(pixel_values=images).last_hidden_state[:, 0, :]
        return z

print("EMA-ready DistillationViT class defined.")

# %%
# =================================================================================
# SECTION 4: TRAINING & EVALUATION LOGIC (WEIGHTED EMA + FULL DIAGNOSTICS)
# =================================================================================

classification_criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    
    # Trackers for ALL losses (used for plotting)
    batch_backbone_losses = []
    batch_dummy_losses = []
    batch_head_losses = [[] for _ in range(len(model.heads))]
    
    head_correct_preds = defaultdict(int)
    total_samples = 0

    # Temporary holder for weighted average teacher
    with torch.no_grad():
        weighted_avg_teacher = copy.deepcopy(model.dummy_head)

    progress_bar = tqdm(train_loader, desc="Training Epoch", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        total_samples += len(labels)
        
        z = model(images)
        z_detached = z.detach()

        # --- PHASE 1: DIAGNOSTICS (Calculate all losses for plotting) ---
        with torch.no_grad():
            # 1. Dummy Head Diagnostic Loss
            dummy_logits = model.dummy_head(z)
            batch_dummy_losses.append(classification_criterion(dummy_logits, labels).item())
            
            # 2. All Teacher Heads Diagnostic Losses (on live z for fair comparison)
            for i, head in enumerate(model.heads):
                head_logits_diag = head(z)
                batch_head_losses[i].append(classification_criterion(head_logits_diag, labels).item())

        # --- PHASE 2: TRAINING (The actual work) ---
        head_logits = {f'head_{i+1}': head(z) for i, head in enumerate(model.heads)}

        # Find the winner
        batch_accuracies = {}
        for head_name, logits in head_logits.items():
            _, preds = torch.max(logits, 1)
            correct = torch.sum(preds == labels).item()
            batch_accuracies[head_name] = correct / len(labels)
            head_correct_preds[head_name] += correct
        winner_head_name = max(batch_accuracies, key=batch_accuracies.get)

        # Calculate Training Losses
        winner_loss = classification_criterion(head_logits[winner_head_name], labels)
        batch_backbone_losses.append(winner_loss.item()) # Winner drives backbone loss

        losers_loss = 0
        for i, head in enumerate(model.heads):
            if i != (int(winner_head_name[-1])-1):
                loser_logits = head(z_detached)
                losers_loss += classification_criterion(loser_logits, labels)
        
        final_loss = winner_loss + losers_loss
        
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        
        # --- PHASE 3: EMA UPDATE ---
        with torch.no_grad():
            accuracies_tensor = torch.tensor(list(batch_accuracies.values()), device=device)
            accuracy_weights = torch.nn.functional.softmax(accuracies_tensor, dim=0)

            for param in weighted_avg_teacher.parameters():
                param.data.zero_()
            
            for i, teacher_head in enumerate(model.heads):
                weight = accuracy_weights[i]
                for avg_param, teacher_param in zip(weighted_avg_teacher.parameters(), teacher_head.parameters()):
                    avg_param.data.add_(teacher_param.data, alpha=weight)

            for student_param, avg_teacher_param in zip(model.dummy_head.parameters(), weighted_avg_teacher.parameters()):
                student_param.data.mul_(config.EMA_DECAY).add_(avg_teacher_param.data, alpha=1 - config.EMA_DECAY)

    # Collate all metrics for return
    epoch_metrics = {
        "avg_backbone_loss": np.mean(batch_backbone_losses),
        "avg_dummy_loss": np.mean(batch_dummy_losses), # Now correctly populated
        "head_accuracies": {name: correct / total_samples for name, correct in head_correct_preds.items()}
    }
    for i in range(len(model.heads)):
        epoch_metrics[f"avg_head_{i+1}_loss"] = np.mean(batch_head_losses[i])
        
    return epoch_metrics

def evaluate(model, data_loader, device):
    model.eval()
    total_loss, correct_preds, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            total_samples += len(labels)
            z = model(images)
            dummy_logits = model.dummy_head(z)
            loss = classification_criterion(dummy_logits, labels)
            total_loss += loss.item()
            _, preds = torch.max(dummy_logits, 1)
            correct_preds += torch.sum(preds == labels).item()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_preds / total_samples
    return {"avg_loss": avg_loss, "accuracy": accuracy}

print("Diagnostic-ready train_one_epoch and evaluate functions defined.")

# %%
# =================================================================================
# SECTION 5: THE MAIN EXPERIMENT LOOP (WEIGHTED EMA + DIAGNOSTICS)
# =================================================================================

config = Config()
lodo_histories = {} 

for target_domain in config.DOMAINS:
    print(f"==============================================================")
    print(f"  STARTING LODO EXPERIMENT: Target Domain = {target_domain.upper()}")
    print(f"==============================================================")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=config.DATA_DIR, target_domain=target_domain,
        all_domains=config.DOMAINS, batch_size=config.BATCH_SIZE, seed=config.SEED
    )
    
    current_dropout_rates = list(np.random.choice(
        config.DROPOUT_OPTIONS, config.NUM_HEADS, replace=False
    ))
    current_dummy_rate = np.random.choice(config.DROPOUT_OPTIONS)
    
    model = DistillationViT(
        model_name=config.MODEL_NAME, num_classes=config.NUM_CLASSES,
        dropout_rates=current_dropout_rates,
        dummy_dropout_rate=current_dummy_rate
    ).to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(
        list(model.vit_backbone.parameters()) + list(model.heads.parameters()), 
        lr=config.LEARNING_RATE
    )
    
    best_val_accuracy = 0.0
    best_model_state = None
    history = defaultdict(list)
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        print(f"Current Teacher Rates: { {f'head_{i+1}': rate for i, rate in enumerate(current_dropout_rates)} }")
        print(f"Current Dummy Head Rate: {current_dummy_rate}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, config.DEVICE)
        val_metrics = evaluate(model, val_loader, config.DEVICE)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Backbone Loss: {train_metrics['avg_backbone_loss']:.4f}")
        print(f"  Dummy Diagnostic Loss: {train_metrics['avg_dummy_loss']:.4f}") # NEW PRINT
        print(f"  Validation Loss: {val_metrics['avg_loss']:.4f}")
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.4f}")
        
        print(f"  Epoch Training Accuracies (Teachers):")
        epoch_winner_head_name = max(train_metrics['head_accuracies'], key=train_metrics['head_accuracies'].get)
        for name, acc in sorted(train_metrics['head_accuracies'].items()):
            marker = "<- EPOCH WINNER" if name == epoch_winner_head_name else ""
            print(f"    {name}: {acc:.4f} {marker}")
        
        # --- Store ALL metrics for plotting ---
        history["backbone_loss"].append(train_metrics['avg_backbone_loss'])
        history["dummy_loss"].append(train_metrics['avg_dummy_loss']) # NEW STORAGE
        for i in range(config.NUM_HEADS):
             history[f"head_{i+1}_loss"].append(train_metrics[f"avg_head_{i+1}_loss"]) # NEW STORAGE
        history["val_loss"].append(val_metrics['avg_loss'])
        history["val_accuracy"].append(val_metrics['accuracy'])

        if val_metrics['accuracy'] > best_val_accuracy:
            print(f"  New best validation accuracy! Saving model state.")
            best_val_accuracy = val_metrics['accuracy']
            best_model_state = copy.deepcopy(model.state_dict())
            
        print("  Updating all dropout rates for next epoch...")
        current_dropout_rates = list(np.random.choice(config.DROPOUT_OPTIONS, config.NUM_HEADS, replace=False))
        model.update_dropout_rates(current_dropout_rates)
        
        current_dummy_rate = np.random.choice(config.DROPOUT_OPTIONS)
        model.update_dummy_dropout_rate(current_dummy_rate)

    lodo_histories[target_domain] = history

    print("\nTraining complete. Loading best model for test evaluation...")
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, config.DEVICE)
    
    print(f"\n--- RESULTS FOR TARGET DOMAIN: {target_domain.upper()} ---")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    experiment_results.append({
        "target_domain": target_domain,
        "test_accuracy": test_metrics['accuracy'],
        "best_val_accuracy": best_val_accuracy
    })

print("\n\nALL DIAGNOSTIC EMA LODO EXPERIMENTS COMPLETE")

# %%
# =================================================================================
# SECTION 7: VISUALIZE COMPONENT LOSS CURVES (SUBPLOTS VERSION)
# =================================================================================
print("\n" + "="*70)
print("--- Visualizing Component Loss Curves (Subplots) ---")
print("="*70)

# Create a figure with a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
# Flatten the 2x2 array of axes to make it easy to loop over
axes = axes.flatten()

# Define the colors and line styles to be consistent across all subplots
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'] 

# Loop through each domain's history and its corresponding subplot axis
for i, (domain, history) in enumerate(lodo_histories.items()):
    ax = axes[i]
    epochs = range(1, config.NUM_EPOCHS + 1)
    
    # 1. Backbone Loss (Red Solid) - This is the winner's loss
    ax.plot(epochs, history['backbone_loss'], 'r-', linewidth=3, label='Backbone Loss (Winner)')
    
    # 2. Dummy Head Diagnostic Loss (Blue Solid)
    ax.plot(epochs, history['dummy_loss'], 'b-', linewidth=3, label='Dummy Head Loss')
    
    # 3. Validation Loss (Black Dotted)
    ax.plot(epochs, history['val_loss'], 'k:', linewidth=2, label='Validation Loss')
    
    # 4-7. Individual Teacher Head Losses (Various Colored Dashed Lines)
    for j in range(config.NUM_HEADS):
        ax.plot(epochs, history[f'head_{j+1}_loss'], linestyle='--', color=colors[j], alpha=0.8, label=f'Head {j+1} Loss')

    # --- Subplot Styling ---
    ax.set_title(f'Target Domain: {domain.upper()}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0 for fair comparison

# --- Overall Figure Styling ---
# Create a single, shared legend for the entire figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=7, fontsize=14, fancybox=True)

fig.suptitle('Full Training Loss Dynamics Across All LODO Experiments', fontsize=24, fontweight='bold', y=1.05)

# Adjust layout to prevent titles and labels from overlapping
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
# =================================================================================
# SECTION 6: ANALYSIS & VISUALIZATION (with Dictionary Output)
# =================================================================================
# Now that all experiments are complete, we'll process the results
# and create visualizations to understand the performance of our method.
# ---------------------------------------------------------------------------------

# Add this magic command to ensure plots are displayed in the notebook
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------------
# 6.1: DISPLAY RESULTS IN A TABLE
# ---------------------------------------------------------------------------------
print("--- Final Experiment Results ---")

results_df = pd.DataFrame(experiment_results)
column_order = [
    "target_domain", "test_accuracy", "best_val_accuracy", "num_epochs",
    "batch_size", "learning_rate", "model_name"
]
existing_columns = [col for col in column_order if col in results_df.columns]
results_df = results_df[existing_columns]
average_accuracy = results_df['test_accuracy'].mean()

print(results_df.to_string())
print("\n" + "="*50)
print(f"Average Test Accuracy Across All Domains: {average_accuracy:.4f}")
print("="*50)


# ---------------------------------------------------------------------------------
# 6.2: VISUALIZE THE RESULTS
# ---------------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

sns.barplot(
    data=results_df, x='target_domain', y='test_accuracy', ax=ax, palette='viridis'
)

for index, row in results_df.iterrows():
    ax.text(index, row['test_accuracy'] + 0.01, f"{row['test_accuracy']:.2%}",
            color='black', ha="center", fontsize=12)
    
ax.axhline(average_accuracy, ls='--', color='red', label=f'Average Accuracy ({average_accuracy:.2%})')

ax.set_title('Model Performance on Unseen Target Domains (LODO)', fontsize=16, pad=20)
ax.set_xlabel('Target (Unseen) Domain', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_ylim(0, 1.0)
ax.legend()

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------
### NEW SECTION ###
# 6.3: GENERATE COPY-PASTE DICTIONARY FOR FINAL PLOTTING
# ---------------------------------------------------------------------------------
print("\n" + "="*70)
print("--- Dictionary for Final Plotting ---")
print("# Copy the dictionary below and paste it into your final analysis notebook.")

# Determine the variable name based on the notebook (you can adjust this)
# For the baseline notebook, you'd want 'baseline_results'.
# For the evolutionary notebook, you'd want 'evolutionary_results'.
method_name = "my_method_results" # Generic name
if "baseline" in os.getcwd(): # Simple check if 'baseline' is in the notebook path
    method_name = "baseline_results"
elif "drop-out" in os.getcwd():
    method_name = "evolutionary_results"
    
# Extract the lists from the DataFrame
domain_list = results_df['target_domain'].tolist()
accuracy_list = [round(acc, 4) for acc in results_df['test_accuracy'].tolist()]

# Print in the desired format
print(f"{method_name} = {{")
print(f"    'target_domain': {domain_list},")
print(f"    'test_accuracy': {accuracy_list}")
print(f"}}")
print("="*70)


print("\n--- Experiment Complete ---")

# %%
# =================================================================================
# SECTION 7: COMPARATIVE ANALYSIS & VISUALIZATION (ACADEMIC STYLE - FINAL FIX)
# =================================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 7.1: COMBINE EXPERIMENT RESULTS ---
baseline_results = {
    'target_domain': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'test_accuracy': [0.8213, 0.7082, 0.9060, 0.5887]
}
# Using the results from your successful Option 4 run
evolutionary_results = {
    'target_domain': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'test_accuracy': [0.7993, 0.7381, 0.9587, 0.6149]
}
baseline_df = pd.DataFrame(baseline_results)
baseline_df['method_name'] = 'Baseline'
evolutionary_df = pd.DataFrame(evolutionary_results)
evolutionary_df['method_name'] = 'Train All'
combined_df = pd.concat([baseline_df, evolutionary_df])

# --- 7.2: CREATE THE GROUPED BAR CHART (ROBUST VERSION) ---

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(14, 8))

custom_palette = {'Baseline': '#4B6A9A', 'Train All': '#66C2A5'}

barplot = sns.barplot(
    data=combined_df,
    x='target_domain',
    y='test_accuracy',
    hue='method_name',
    ax=ax,
    palette=custom_palette,
    edgecolor='black'
)

### THE FIX IS HERE ###
# Use the robust 'containers' method to apply patterns correctly.

# ax.containers[0] is the container for the first hue category (Baseline)
# ax.containers[1] is the container for the second hue category (Evolutionary Dropout)

# We want to add a pattern to the second container's bars.
for bar in ax.containers[1]:
    bar.set_hatch('..')

# We also need to apply the pattern to the corresponding legend handle.
# The legend handles are created in the same order.
ax.legend_.legend_handles[1].set_hatch('..')

# --- Add annotations (text on bars) ---
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(
            f"{p.get_height():.2%}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 10),
            textcoords='offset points',
            fontsize=14,
            fontweight='bold',
            color='black'
        )

# --- Final plot styling ---
ax.set_title('Model Comparison on Unseen Target Domains (LODO)', fontsize=22, fontweight='bold', pad=20)
ax.set_xlabel('Target (Unseen) Domain', fontsize=18, fontweight='bold')
ax.set_ylabel('Top-1 Test Accuracy (%)', fontsize=18, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

legend = ax.get_legend()
plt.setp(legend.get_title(), fontweight='bold')

sns.despine()
plt.tight_layout()
plt.show()

# --- Print the final summary table ---
avg_baseline = baseline_df['test_accuracy'].mean()
avg_evolutionary = evolutionary_df['test_accuracy'].mean()
print("\n--- Average Performance Summary ---")
print(f"Average Baseline Accuracy: {avg_baseline:.2%}")
print(f"Average Evolutionary Dropout Accuracy: {avg_evolutionary:.2%}")



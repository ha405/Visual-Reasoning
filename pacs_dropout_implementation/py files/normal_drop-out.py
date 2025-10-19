# %%
# =================================================================================
# SECTION 1: PROJECT SCAFFOLDING & CONFIGURATION
# =================================================================================

# ---------------------------------------------------------------------------------
# 1.1: IMPORTS
# All necessary libraries for the project.
# ---------------------------------------------------------------------------------
%matplotlib inline 
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTConfig
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold # We might use this later for robustness
from collections import defaultdict
import copy

# Set a seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%

# ---------------------------------------------------------------------------------
# 1.2: CONFIGURATION CLASS
# This class holds all hyperparameters and settings in one place.
# ---------------------------------------------------------------------------------
class Config:
    # --- Data Paths and Domains ---
    # In your Config class
    DATA_DIR = r"D:\Haseeb\Datasets\pacs_data"
    DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
    
    # --- Model & Architecture ---
    # In your Config class
    MODEL_NAME = "WinKawaks/vit-tiny-patch16-224"
    NUM_CLASSES = 7
    NUM_HEADS = 4
    DROPOUT_OPTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    # --- Training Hyperparameters ---
    BATCH_SIZE = 128
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4  # A good starting point for fine-tuning transformers
    OPTIMIZER = "AdamW"   # AdamW is generally preferred for transformers
    
    # --- Hardware & Reproducibility ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

# Instantiate the config
config = Config()

# Print out the configuration to verify
print("--- Project Configuration ---")
for key, value in config.__class__.__dict__.items():
    if not key.startswith('__'):
        print(f"{key}: {value}")
print("---------------------------")
print(f"Device: {config.DEVICE}")

# ---------------------------------------------------------------------------------
# 1.3: RESULTS TRACKER
# A list to store the final results from each LODO experiment run.
# This will be converted to a DataFrame at the end for analysis.
# ---------------------------------------------------------------------------------
experiment_results = []

print("\nProject scaffolding is complete. Ready for Section 2: Data Loading.")

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
# SECTION 3: THE MODEL ARCHITECTURE
# =================================================================================

# ---------------------------------------------------------------------------------
# 3.1: CUSTOM ViT MODEL WITH EVOLUTIONARY HEADS
# We define a class that wraps the ViT backbone and adds our 4 competing heads.
# ---------------------------------------------------------------------------------

class EvolutionaryViT(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rates: list):
        """
        Args:
            model_name (string): The name of the pre-trained ViT model from Hugging Face.
            num_classes (int): The number of output classes.
            num_heads (int): The number of parallel classification heads.
            dropout_rate (float): The dropout probability.
        """
        super(EvolutionaryViT, self).__init__()
        
        # 1. Load the pre-trained ViT backbone
        # We use ViTModel, which gives us the feature extractor without the final classification layer.
        self.vit_backbone = ViTModel.from_pretrained(model_name)
        
        # Get the hidden size (feature dimension) from the model's config
        hidden_dim = self.vit_backbone.config.hidden_size
        
        # 2. Create the list of competing heads
        # We use nn.ModuleList, which is the proper way to hold a list of PyTorch modules.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=rate), # Use the specific rate for this head
                nn.Linear(hidden_dim, num_classes)
            ) for rate in dropout_rates
        ])

    def update_dropout_rates(self, new_rates: list):
        """
        Updates the dropout probability for each head.
        This allows us to change the rates between epochs without re-creating the model.
        """
        for i, head in enumerate(self.heads):
            # nn.Sequential has layers indexed. 0 is Dropout, 1 is Linear.
            head[0].p = new_rates[i]
        
    def forward(self, images):
        """
        Defines the forward pass of the model.
        """
        # 1. Get features from the backbone
        # The output is a dictionary-like object. We want the 'last_hidden_state'.
        outputs = self.vit_backbone(pixel_values=images)
        
        # For ViT, the feature representation for the entire image is the output
        # corresponding to the special [CLS] token, which is the first one.
        # Shape: (batch_size, sequence_length, hidden_dim) -> (batch_size, hidden_dim)
        feature_vector_z = outputs.last_hidden_state[:, 0, :]
        
        # 2. Pass the feature vector through all heads
        head_outputs = {}
        for i, head in enumerate(self.heads):
            head_outputs[f'head_{i+1}'] = head(feature_vector_z)
            
        return head_outputs

# --- Let's test it to make sure it works ---
print("\nModel architecture seems correct. Ready for Section 4: Training and Evaluation Logic.")

# %%
# =================================================================================
# SECTION 4: TRAINING & EVALUATION LOGIC
# =================================================================================

# ---------------------------------------------------------------------------------
# 4.1: TRAIN_ONE_EPOCH FUNCTION
# This function handles the custom "survival of the fittest" training loop for one epoch.
# ---------------------------------------------------------------------------------
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    
    total_winner_loss = 0.0
    # We still track accuracy over the epoch to see general trends
    head_correct_preds = defaultdict(int)
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training Epoch", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # --- Per-Batch Tournament ---
        # 1. Forward pass
        head_outputs = model(images)
        
        # 2. Find the winner FOR THIS BATCH
        batch_accuracies = {}
        batch_losses = {}
        for head_name, logits in head_outputs.items():
            loss = criterion(logits, labels)
            batch_losses[head_name] = loss
            
            _, preds = torch.max(logits, 1)
            correct = torch.sum(preds == labels).item()
            batch_accuracies[head_name] = correct / labels.size(0)

            # Update epoch-level stats for logging
            head_correct_preds[head_name] += correct
            
        total_samples += labels.size(0)
            
        winner_head_name = max(batch_accuracies, key=batch_accuracies.get)
        
        # 3. Backpropagate from the batch winner's loss
        winner_loss = batch_losses[winner_head_name]
        
        optimizer.zero_grad()
        winner_loss.backward()
        optimizer.step()
        
        total_winner_loss += winner_loss.item()

    # --- End of Epoch ---
    final_head_accuracies = {name: (correct / total_samples) for name, correct in head_correct_preds.items()}
    
    return {
        "avg_winner_loss": total_winner_loss / len(train_loader),
        "head_accuracies": final_head_accuracies
    }

# %%
# =================================================================================
# SECTION 4: TRAINING & EVALUATION LOGIC (ADVANCED STRATEGY)
# =================================================================================

# ---------------------------------------------------------------------------------
# 4.1: TRAIN_ONE_EPOCH FUNCTION
# (This function is correct and does not need to change)
# ---------------------------------------------------------------------------------
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_winner_loss = 0.0
    head_correct_preds = defaultdict(int)
    total_samples = 0
    progress_bar = tqdm(train_loader, desc="Training Epoch", leave=False)
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        head_outputs = model(images)
        batch_accuracies = {}
        batch_losses = {}
        for head_name, logits in head_outputs.items():
            loss = criterion(logits, labels)
            batch_losses[head_name] = loss
            _, preds = torch.max(logits, 1)
            correct = torch.sum(preds == labels).item()
            batch_accuracies[head_name] = correct / labels.size(0)
            head_correct_preds[head_name] += correct
        total_samples += labels.size(0)
        winner_head_name = max(batch_accuracies, key=batch_accuracies.get)
        winner_loss = batch_losses[winner_head_name]
        optimizer.zero_grad()
        winner_loss.backward()
        optimizer.step()
        total_winner_loss += winner_loss.item()
    final_head_accuracies = {name: (correct / total_samples) for name, correct in head_correct_preds.items()}
    return {
        "avg_winner_loss": total_winner_loss / len(train_loader),
        "head_accuracies": final_head_accuracies
    }

# ---------------------------------------------------------------------------------
# 4.2: EVALUATE FUNCTION
# (This function is correct and does not need to change)
# ---------------------------------------------------------------------------------
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            total_samples += labels.size(0)
            head_outputs = model(images)
            all_logits = torch.stack(list(head_outputs.values()))
            ensembled_logits = torch.mean(all_logits, dim=0)
            loss = criterion(ensembled_logits, labels)
            total_loss += loss.item()
            _, preds = torch.max(ensembled_logits, 1)
            correct_preds += torch.sum(preds == labels).item()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_preds / total_samples
    return {
        "avg_loss": avg_loss,
        "accuracy": accuracy
    }

# --- Quick Test of the Functions (Optional but Recommended) ---
### CHANGE ###
# The test block is now updated to work with the new model __init__
# and the new 80/20 get_dataloaders function.
# ---------------------------------------------------------------------------------
print("Running a quick test of the training and evaluation functions...")

# We need some dataloaders for the test
target_domain_test = config.DOMAINS[3] # "sketch"
train_loader_test, val_loader_test, _ = get_dataloaders(
    root_dir=config.DATA_DIR,
    target_domain=target_domain_test,
    all_domains=config.DOMAINS,
    batch_size=config.BATCH_SIZE,
    seed=config.SEED # Pass the seed
)

# 1. Generate an initial list of dropout rates for the test
initial_dropout_rates = list(np.random.choice(
    config.DROPOUT_OPTIONS, 
    config.NUM_HEADS, 
    replace=False
))
print(f"Test model initial dropout rates: {initial_dropout_rates}")

# 2. Instantiate the model using the LIST of rates
test_model = EvolutionaryViT(
    model_name=config.MODEL_NAME,
    num_classes=config.NUM_CLASSES,
    dropout_rates=initial_dropout_rates # Pass the list here
).to(config.DEVICE)

test_optimizer = torch.optim.AdamW(test_model.parameters(), lr=config.LEARNING_RATE)
test_criterion = nn.CrossEntropyLoss()

# Run one training epoch
train_metrics = train_one_epoch(test_model, train_loader_test, test_optimizer, test_criterion, config.DEVICE)
print("\n--- One Training Epoch Test ---")
print(f"Average Winner Loss: {train_metrics['avg_winner_loss']:.4f}") 
print("Final Head Accuracies for the Epoch:")
for name, acc in train_metrics['head_accuracies'].items():
    print(f"  {name}: {acc:.4f}")

# Run one evaluation pass
eval_metrics = evaluate(test_model, val_loader_test, test_criterion, config.DEVICE)
print("\n--- One Evaluation Pass Test ---")
print(f"Ensembled Validation Loss: {eval_metrics['avg_loss']:.4f}")
print(f"Ensembled Validation Accuracy: {eval_metrics['accuracy']:.4f}")

print("\nTraining and evaluation logic seems correct. Ready for Section 5: The Main Experiment Loop.")

# %%
# =================================================================================
# SECTION 5: THE MAIN EXPERIMENT LOOP (ADVANCED STRATEGY)
# =================================================================================
# This version implements the "Winner-Stays, Losers-Re-roll" dropout strategy.
# The dropout rates are now dynamic and adapt based on epoch performance.
# ---------------------------------------------------------------------------------

# A fresh copy of the config to ensure we start clean
config = Config()

# Loop over each domain to set it as the target domain once
for target_domain in config.DOMAINS:
    print(f"==============================================================")
    print(f"  STARTING LODO EXPERIMENT: Target Domain = {target_domain.upper()}")
    print(f"==============================================================")
    
    # --- 1. Setup for this specific LODO run ---
    # Get the specific data loaders for this train/test split
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=config.DATA_DIR,
        target_domain=target_domain,
        all_domains=config.DOMAINS,
        batch_size=config.BATCH_SIZE, seed=config.SEED
    )
    
    # Initialize the first set of random dropout rates for the competing heads.
    # np.random.choice ensures we get unique rates if possible.
    current_dropout_rates = list(np.random.choice(
        config.DROPOUT_OPTIONS, 
        config.NUM_HEADS, 
        replace=False # Tries to pick unique rates
    ))
    
    # Initialize a fresh model with the starting dropout rates
    model = EvolutionaryViT(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        dropout_rates=current_dropout_rates # Pass the list of rates
    ).to(config.DEVICE)
    
    # Initialize a fresh optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # --- 2. Training Loop for this LODO run ---
    best_val_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        print(f"Current Dropout Rates: { {f'head_{i+1}': rate for i, rate in enumerate(current_dropout_rates)} }")
        
        # Train for one epoch using the per-batch winner selection
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        
        # Evaluate on the validation set to check progress and save the best model
        val_metrics = evaluate(model, val_loader, criterion, config.DEVICE)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Avg Winner Loss: {train_metrics['avg_winner_loss']:.4f}")
        print(f"  Validation Loss: {val_metrics['avg_loss']:.4f}")
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Check if this is the best model so far based on validation performance
        if val_metrics['accuracy'] > best_val_accuracy:
            print(f"  New best validation accuracy! Saving model state.")
            best_val_accuracy = val_metrics['accuracy']
            best_model_state = copy.deepcopy(model.state_dict())

        # --- Adaptive Dropout Logic for the NEXT epoch ---
        # Find the head that had the best OVERALL accuracy during this epoch
        epoch_winner_head_name = max(train_metrics['head_accuracies'], key=train_metrics['head_accuracies'].get)
        epoch_winner_index = int(epoch_winner_head_name.split('_')[-1]) - 1
        
        print(f"  Epoch Training Accuracies:")
        for name, acc in sorted(train_metrics['head_accuracies'].items()):
            marker = "<- WINNER" if name == epoch_winner_head_name else ""
            print(f"    {name}: {acc:.4f} {marker}")
        
        # Keep the winner's dropout rate for the next epoch
        winner_rate = current_dropout_rates[epoch_winner_index]
        
        # Generate a new set of random rates for all heads
        new_random_rates = list(np.random.choice(config.DROPOUT_OPTIONS, config.NUM_HEADS, replace=False))
        
        # "Exploitation": Overwrite the winner's slot with its successful rate
        new_random_rates[epoch_winner_index] = winner_rate
        
        # "Exploration": The other heads get new random rates
        current_dropout_rates = new_random_rates
        
        # Update the model in-place with the new dropout rates
        model.update_dropout_rates(current_dropout_rates)
            
    # --- 3. Final Evaluation for this LODO run ---
    print("\nTraining complete for this LODO split.")
    print("Loading best model state and evaluating on the TEST set...")
    
    # Load the best performing model based on validation accuracy
    model.load_state_dict(best_model_state)
    
    # Evaluate on the unseen target domain (the test set)
    test_metrics = evaluate(model, test_loader, criterion, config.DEVICE)
    
    print(f"\n--- RESULTS FOR TARGET DOMAIN: {target_domain.upper()} ---")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"--------------------------------------------------")
    
    # --- 4. Store the final results ---
    experiment_results.append({
        "target_domain": target_domain,
        "source_domains": [d for d in config.DOMAINS if d != target_domain],
        "test_accuracy": test_metrics['accuracy'],
        "best_val_accuracy": best_val_accuracy,
        "model_name": config.MODEL_NAME,
        "num_epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE
    })

print("\n\n==============================================================")
print("          ALL ADAPTIVE DROPOUT LODO EXPERIMENTS COMPLETE")
print("==============================================================")

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
    'test_accuracy': [0.7339, 0.6877, 0.8287, 0.5312]
}
baseline_df = pd.DataFrame(baseline_results)
baseline_df['method_name'] = 'Baseline'
evolutionary_df = pd.DataFrame(evolutionary_results)
evolutionary_df['method_name'] = 'Evolutionary Dropout'
combined_df = pd.concat([baseline_df, evolutionary_df])

# --- 7.2: CREATE THE GROUPED BAR CHART (ROBUST VERSION) ---

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(14, 8))

custom_palette = {'Baseline': '#4B6A9A', 'Evolutionary Dropout': '#66C2A5'}

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

# %%
# =================================================================================
# SECTION 8: GRAND COMPARATIVE ANALYSIS & VISUALIZATION (FINAL FIXED LEGEND)
# =================================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 8.1: COMBINE ALL EXPERIMENT RESULTS ---
baseline_results = {
    'target_domain': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'test_accuracy': [0.8208, 0.7082, 0.906, 0.589]
}
wta_dropout_results = {
    'target_domain': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'test_accuracy': [0.7339, 0.6877, 0.8287, 0.5312]
}
train_all_results = {
    'target_domain': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'test_accuracy': [0.7993, 0.7381, 0.9587, 0.6149]
}
shared_head_results = {
    'target_domain': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'test_accuracy': [0.7979, 0.7543, 0.9587, 0.6546]
}

baseline_df = pd.DataFrame(baseline_results); baseline_df['Method'] = 'A: Baseline'
wta_df = pd.DataFrame(wta_dropout_results); wta_df['Method'] = 'B: Winner-Take-All'
train_all_df = pd.DataFrame(train_all_results); train_all_df['Method'] = 'C: Train All Heads'
shared_head_df = pd.DataFrame(shared_head_results); shared_head_df['Method'] = 'D: Shared Head'
combined_df = pd.concat([baseline_df, wta_df, train_all_df, shared_head_df])

# --- 8.2: CREATE THE GRAND GROUPED BAR CHART ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(18, 9))

custom_palette = {
    'A: Baseline': '#4B6A9A',
    'B: Winner-Take-All': '#DB845B',
    'C: Train All Heads': '#92B56F',
    'D: Shared Head': '#66C2A5'
}

barplot = sns.barplot(
    data=combined_df, x='target_domain', y='test_accuracy', hue='Method',
    ax=ax, palette=custom_palette, edgecolor='black'
)

# Apply hatching patterns
patterns = ['', '..', '//', 'xx']
for i, container in enumerate(ax.containers):
    for bar in container:
        bar.set_hatch(patterns[i])

# Add annotations on bars
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.2%}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9),
                    textcoords='offset points',
                    fontsize=12, fontweight='bold', color='black')

# --- Fix legend properly ---
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels,
                   title='Method',
                   loc='upper center',
                   bbox_to_anchor=(0.5, 1),
                   ncol=4,
                   frameon=False,
                   fontsize=14,
                   title_fontsize=16)

# Apply hatching to legend handles
for i, handle in enumerate(legend.legend_handles):
    handle.set_hatch(patterns[i])

plt.setp(legend.get_title(), fontweight='bold')

# --- Final styling ---
ax.set_title('Comparison of All Methods on Unseen Target Domains (LODO)',
             fontsize=24, fontweight='bold', pad=20)
ax.set_xlabel('Target (Unseen) Domain', fontsize=18, fontweight='bold', labelpad=15)
ax.set_ylabel('Top-1 Test Accuracy (%)', fontsize=18, fontweight='bold', labelpad=15)
ax.set_ylim(0, 1.15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

sns.despine()
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# --- Summary table ---
print("\n" + "="*50)
print("--- Average Performance Summary ---")
print(f"A: Average Baseline Accuracy: {baseline_df['test_accuracy'].mean():.2%}")
print(f"B: Average Winner-Take-All Accuracy: {wta_df['test_accuracy'].mean():.2%}")
print(f"C: Average Train All Heads Accuracy: {train_all_df['test_accuracy'].mean():.2%}")
print(f"D: Average Shared Head Accuracy: {shared_head_df['test_accuracy'].mean():.2%}")
print("="*50)




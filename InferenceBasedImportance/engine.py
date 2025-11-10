# engine.py
import torch
from tqdm import tqdm
from torchmetrics import Accuracy
import logging


logger = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, optimizer, loss_fn, scaler, scheduler, device, use_amp):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training dataloader
        optimizer: Optimizer
        loss_fn: Loss function
        scaler: GradScaler for mixed precision
        scheduler: Learning rate scheduler
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        avg_loss: Average training loss
        avg_acc: Average training accuracy
    """
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass with mixed precision
        with torch.autocast(device_type=str(device).split(':')[0], enabled=use_amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate metrics
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            
            total_loss += loss.item() * images.size(0)
            total_correct += correct
            total_samples += images.size(0)
            
            # Update progress bar
            current_loss = total_loss / total_samples
            current_acc = total_correct / total_samples
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })
    
    # Step the scheduler
    scheduler.step()
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc


def evaluate(model, dataloader, loss_fn, device, use_amp):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Neural network model
        dataloader: Evaluation dataloader
        loss_fn: Loss function
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        avg_loss: Average evaluation loss
        avg_acc: Average evaluation accuracy
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for images, labels in progress_bar:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass with mixed precision
            with torch.autocast(device_type=str(device).split(':')[0], enabled=use_amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            
            # Calculate metrics
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            
            total_loss += loss.item() * images.size(0)
            total_correct += correct
            total_samples += images.size(0)
            
            # Update progress bar
            current_loss = total_loss / total_samples
            current_acc = total_correct / total_samples
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc


def train_model(model, train_loader, val_loader, config, num_epochs, learning_rate, 
                save_path=None, phase_name="Training"):
    """
    Full training loop for a model.
    
    Args:
        model: Neural network model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Configuration object
        num_epochs: Number of epochs to train
        learning_rate: Initial learning rate
        save_path: Optional path to save the best model
        phase_name: Name of the training phase for logging
        
    Returns:
        best_val_acc: Best validation accuracy achieved
    """
    device = config.DEVICE
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.PHASE1_WEIGHT_DECAY if phase_name == "Phase 1" else config.PHASE2_WEIGHT_DECAY
    )
    
    # Setup scheduler based on config
    if config.PHASE1_LR_SCHEDULE == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=num_epochs//3, gamma=0.1
        )
    
    # Setup loss and scaler
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
    
    best_val_acc = 0.0
    best_model_state = None
    
    logger.info(f"Starting {phase_name} training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        logger.info(f"\n{phase_name} - Epoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, scheduler, 
            device, config.USE_AMP
        )
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, config.USE_AMP)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
        # Save model if path provided
        if save_path:
            torch.save(best_model_state, save_path)
            logger.info(f"Saved best model to {save_path}")
    
    return best_val_acc

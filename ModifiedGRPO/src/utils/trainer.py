import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.configuration import DEVICE

class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, dataloader):
        self.model.train()
        total_loss = 0
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        
        # Change 1: Manually set the dropout layers in our custom head to train mode
        # This keeps them active during evaluation for Monte Carlo Dropout
        for module in self.model.classifier_head.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        total_correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # Change 2: Perform N=20 forward passes to get an ensemble of predictions
                ensemble_preds = []
                for _ in range(20): # N=20 passes
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    ensemble_preds.append(preds.unsqueeze(0))
                
                # Change 3: Calculate the majority vote for the final prediction
                stacked_preds = torch.cat(ensemble_preds, dim=0)
                final_preds, _ = torch.mode(stacked_preds, dim=0)

                total_correct += (final_preds == labels).sum().item()
                total += labels.size(0)
        return total_correct / total
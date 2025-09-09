from configuration import *

class ViTModel(nn.Module):
    def __init__(self, num_classes, model_size="base"):
        super(ViTModel, self).__init__()
        # Change 1: Load the base ViTModel using the alias 'ViTModelBase'
        self.model = ViTModelBase.from_pretrained(
            MODELS[model_size]
        )
        
        # Change 2: Define our custom "thinking" head
        hidden_size = self.model.config.hidden_size # This is 768 for the base model
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # Change 3: Update the forward pass logic
        # Pass input through the base model
        outputs = self.model(x)
        # Get the feature vector for the [CLS] token
        cls_token_features = outputs.last_hidden_state[:, 0, :]
        # Pass the features through our custom head
        logits = self.classifier_head(cls_token_features)
        return logits
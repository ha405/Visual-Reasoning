import torch

CFG = {
    "system": {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "seed": 42,
        "num_workers": 4,
        "log_dir": "./logs",
        "ckpt_dir": "./checkpoints",
        "save_freq": 1,       # Save every N epochs
        "print_freq": 50,     # Print logs every N iterations
    },

    "train": {
        "batch_size": 128,
        "epochs": 5,
        "lr": 1e-4,
    },

    "grqo": {
        "topk": 24,
        "alpha": 2.0,
        "beta": 0.5,
        "tau": 1e-3,
        "temperature": 0.1,
        "hidden_dim": 192,
        "num_heads": 8,
        "num_tokens": 24,
        "num_layers": 2,
        "dropout": 0.1,
        "ddropout": 0.1,
        "lambda_grqo": 1.0,
        "teacher_ema": 0.99,
        "reward_proxy": "taylor",  # or "gradnorm"
    },

    "datasets": {
        "PACS": {
            "root": r"D:\Haseeb\Datasets\pacs_data",
            "domains": ["art_painting", "cartoon", "photo", "sketch"],
            "classes": ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
            "num_classes": 7,
        },

        "OfficeHome": {
            "root": r"D:\Haseeb\Datasets\OfficeHomeDataset_10072016",
            "domains": ["Art", "Clipart", "Product", "Real World"],
            "num_classes": 65,
        },

        "VLCS": {
            "root": r"D:\Haseeb\Datasets\VLCS",
            "domains": ["VOC2007", "LabelMe", "Caltech101", "SUN09"],
            "classes": ["bird", "car", "chair", "dog", "person"],
            "num_classes": 5,
        },
    },

    "da": {
        "source_domains": ["photo", "art_painting", "cartoon"],
        "target_domain": "sketch",
    },
}

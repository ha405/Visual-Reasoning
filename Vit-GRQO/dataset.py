import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split

class PACSDataset:
    def __init__(self, data_root, domains, transform,batch_size):
        self.data_root = data_root
        self.domains = domains
        self.transform = transform
        self.batch_size=batch_size

    def get_dataloader(self, domain, train=True):
        dataset = datasets.ImageFolder(os.path.join(self.data_root, domain), transform=self.transform)
        
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=[dataset.targets[i] for i in indices], random_state=SEED)
        selected_idx = train_idx if train else val_idx
        
        subset = Subset(dataset, selected_idx)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=train)
        return loader

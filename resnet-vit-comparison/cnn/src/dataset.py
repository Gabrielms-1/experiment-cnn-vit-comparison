from torch.utils.data import Dataset, DataLoader    
from PIL import Image
import glob
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from importlib.machinery import SourceFileLoader

utils_seed_path = "./resnet-vit-comparison/utils/__init__.py"
seed_module = SourceFileLoader("utils_seed", utils_seed_path).load_module()

class FolderBasedDataset(Dataset):
    def __init__(self, root_dir, resize=None):
        self.root_dir = root_dir
        self.images, self.labels = self._get_images_paths()
        self.transform = self.get_transform(resize)
        
        self.label_map_to_int = {label: i for i, label in enumerate(sorted(set(label for label in self.labels)))}
        self.int_to_label_map = {i: label for label, i in self.label_map_to_int.items()}

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        label = (image_path).split("/")[-2]
        label = self.label_map_to_int[label]

        image = self.transform(image=image)['image']

        return image, label, image_path
    
    def _get_images_paths(self):
        images = glob.glob(os.path.join(self.root_dir, '**', '*.jpg'), recursive=True)
        
        labels = [item.split("/")[-2] for item in images]
        
        return images, labels
    
    def get_transform(self, resize):
        if 'train' in self.root_dir:
            return A.Compose([
                A.Resize(resize, resize, interpolation=cv2.INTER_LANCZOS4),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomBrightnessContrast(),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(resize, resize, interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        
def create_dataloader(train_dataset, val_dataset, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_module.seed_worker,
        pin_memory=True,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_module.seed_worker,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_dataset = FolderBasedDataset(root_dir="cnn/data/train", resize=400)
    valid_dataset = FolderBasedDataset(root_dir="cnn/data/val", resize=400)

    train_loader, valid_loader = create_dataloader(train_dataset, valid_dataset, batch_size=12, seed=42)

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(valid_loader)}")
    


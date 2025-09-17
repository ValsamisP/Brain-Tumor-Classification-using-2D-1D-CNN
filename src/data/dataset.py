""" 
Dataset class and data loading utilities for brain tumor classification
"""

import os 
from collections import Counter 
import torch 
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

class BrainTumorDataset(Dataset):
    """Custom dataset for brain tumor images"""

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of paths to images
            labels(list): List of labels
            tranform (torchvision.transforms): Optional transform
        """

        self.image_paths = image_paths
        self.labels = labels 
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224,224),(0,0,0))

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
    
def load_data(data_dir):
    """
    Load image paths and labels from directory structure
    """

    image_paths = [] 
    labels = [] 
    class_names = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

    print("Class Distributioon:")
    for class_name in class_names:
        class_dir = os.path.join(data_dir,class_name)
        if os.path.exists(class_dir):
            class_count = 0
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png','.jpg','.jpeg')):
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(class_name)
                    class_count += 1
            print(f" {class_name}: {class_count} images")
        else:
            print(f" Warning: {class_dir} not found")
    return image_paths, labels, class_names
    

def get_transforms(image_size=224):
    """
    Get training and validation/test transforms
    
    Input image_size
    
    Returns -> tuple(train_transform,val_test_transform)"""


    #Transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomResizedCrop(image_size, scale=(0.8,1.0)),
        transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,hue=0.1),
        transforms.RandomAffine(degrees=10,translate=(0.1,0.1),scale=(0.9,1.1)),
        transforms.GaussianBlur(kernel_size=3,sigma=(0.1,2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02,0.33),ratio=(0.3,3.3))
    ])

    # Standard validation/test transforms
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    return train_transform, val_test_transform


def create_weighted_sampler(labels):
    """
    Create weighted sampler to handle class imbalance
    """

    class_counts = Counter(labels)
    total_samples = len(labels)

    #Calculating weights proportional to class frequency
    class_weights = {cls: total_samples/ count for cls, count in class_counts.items()}

    # Assign weight to each sample 
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True)


def analyze_dataset(image_paths,labels,class_names):
    """
    Analyze and print dataset statistics
    """

    print(f"\n Dataset Analysis:")
    print(f"Total images: {len(image_paths)}")
    print(f"Number of classes: {len(class_names)}")

    label_counts = Counter(labels)
    print(f"\n Class distribution:")
    for class_name in class_names:
        count = label_counts.get(class_name,0)
        percentage = (count/len(labels))*100
        print(f" {class_name}: {count} ({percentage:.1f}%)")

    min_count = min(label_counts.values())
    max_count =max(label_counts.values())
    imbalance_ratio = max_count/min_count

    print(f"\nImbalance analysis:")
    print(f"  Min class size: {min_count}")
    print(f"  Max class size: {max_count}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2:
        print("    Significant class imbalance detected!")
        print("   Weighted sampling will be used during training.")
    else:
        print("   Classes are relatively balanced.")       
            
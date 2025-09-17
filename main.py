"""
Brain Tumor Classification using Enhanced 2D+1D CNN
Main training script with transfer learning and attention mechanisms
"""

"""
To run it -> python main.py --data_dir data_brain_tumor  (where data_brain_tumor is the name of the folder that includes the data)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import argparse
import os

from src.models.cnn import Enhanced_CNN2D1D
from src.data.dataset import BrainTumorDataset, load_data, get_transforms, create_weighted_sampler
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model, evaluate_model_with_tta
from src.utils.visualization import plot_training_history, plot_confusion_matrix
from src.utils.helpers import set_seed, setup_device, save_results

def parse_args():
    parser = argparse.ArgumentParser(description='Brain Tumor Classification Training')
    parser.add_argument('--data_dir', type=str, default='data_brain_tumor',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_tta', action='store_true',
                        help='Use Test Time Augmentation for evaluation')
    parser.add_argument('--model_save_path', type=str, default='models/enhanced_brain_tumor_model.pth',
                        help='Path to save the trained model')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = setup_device()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("Enhanced Brain Tumor Classification Training")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    
    # Load and analyze data
    print("\nLoading and analyzing data...")
    image_paths, labels, class_names = load_data(args.data_dir)
    print(f"Total images: {len(image_paths)}")
    print(f"Classes: {class_names}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(encoded_labels), y=encoded_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, encoded_labels, test_size=0.3, random_state=args.seed, stratify=encoded_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(args.image_size)
    
    # Create datasets
    train_dataset = BrainTumorDataset(X_train, y_train, train_transform)
    val_dataset = BrainTumorDataset(X_val, y_val, val_test_transform)
    test_dataset = BrainTumorDataset(X_test, y_test, val_test_transform)
    
    # Create data loaders
    train_sampler = create_weighted_sampler(y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = Enhanced_CNN2D1D(num_classes=len(class_names)).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training components
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, verbose=True, min_lr=1e-7
    )
    
    # Train model
    print("\nStarting training...")
    print("=" * 50)
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, args.num_epochs, device)
    
    # Load best model
    model.load_state_dict(history['best_model_state'])
    
    # Save model
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    torch.save({
        'model_state_dict': history['best_model_state'],
        'class_names': class_names,
        'label_encoder': label_encoder,
        'best_val_acc': history['best_val_acc'],
        'class_weights': class_weights,
        'args': vars(args)
    }, args.model_save_path)
    
    print(f"Model saved to {args.model_save_path}")
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
    
    # Evaluate model
    print("\nEvaluating model...")
    if args.use_tta:
        print("Using Test Time Augmentation...")
        y_pred, y_true = evaluate_model_with_tta(model, test_loader, class_names, device)
        eval_type = "TTA"
    else:
        print("Standard evaluation...")
        y_pred, y_true = evaluate_model(model, test_loader, device)
        eval_type = "Standard"
    
    # Save and display results
    save_results(y_true, y_pred, class_names, history, eval_type, args)
    
    # Create visualizations
    plot_training_history(history, save_path='plots/training_history.png')
    plot_confusion_matrix(y_true, y_pred, class_names, save_path='plots/confusion_matrix.png')
    
    print("\nTraining completed successfully!")
    print("Check the 'results' and 'plots' directories for detailed results.")

if __name__ == "__main__":
    main()
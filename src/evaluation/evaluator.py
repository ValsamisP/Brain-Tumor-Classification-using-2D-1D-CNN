"""
Model evaluation utilities including standard evaluation and Test Time Augmentation
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_loader, device):
    """
    Standard model evaluation without augmentation
    
    Input:
        model: PyTorch model
        test_loader: Test data loader
        device: Device (CPU/GPU)
        
    Returns:
        tuple: (predictions, true_labels)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_predictions, all_labels

def evaluate_model_with_tta(model, test_loader, class_names, device, num_augmentations=3):
    """
    Evaluate model with Test Time Augmentation (TTA) 
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    print(f"Applying Test Time Augmentation with {num_augmentations} augmentations...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            # Initialize prediction accumulator
            batch_predictions = torch.zeros(batch_size, len(class_names)).to(device)
            
            # Original prediction
            outputs = model(images)
            batch_predictions += F.softmax(outputs, dim=1)
            
            # Augmented predictions - work with normalized tensors correctly
            augmentations_applied = 0
            
            if num_augmentations >= 1:
                # Horizontal flip (safe with any normalization)
                aug_images = torch.flip(images, dims=[3])
                aug_outputs = model(aug_images)
                batch_predictions += F.softmax(aug_outputs, dim=1)
                augmentations_applied += 1
            
            if num_augmentations >= 2:
                # Small noise in normalized space
                noise = torch.randn_like(images) * 0.02  # Very small noise
                aug_images = images + noise  # No clamping needed
                aug_outputs = model(aug_images)
                batch_predictions += F.softmax(aug_outputs, dim=1)
                augmentations_applied += 1
            
            if num_augmentations >= 3:
                # Vertical flip
                aug_images = torch.flip(images, dims=[2])
                aug_outputs = model(aug_images)
                batch_predictions += F.softmax(aug_outputs, dim=1)
                augmentations_applied += 1
            
            # Average predictions (original + augmentations)
            batch_predictions = batch_predictions / (augmentations_applied + 1)
            _, predicted = torch.max(batch_predictions, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    return all_predictions, all_labels

def get_detailed_metrics(y_true, y_pred, class_names):
    """
    Calculate detailed evaluation metrics
    
    Input:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        dict: Dictionary containing various metrics
    """
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    cm = metrics['confusion_matrix']
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        if cm[i].sum() > 0:
            per_class_acc[class_name] = cm[i, i] / cm[i].sum()
        else:
            per_class_acc[class_name] = 0.0
    
    metrics['per_class_accuracy'] = per_class_acc
    
    return metrics

def print_evaluation_results(y_true, y_pred, class_names, eval_type="Standard"):
    """
    Print comprehensive evaluation results
    
    Input:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        eval_type: Type of evaluation (e.g., "Standard", "TTA")
    """
    metrics = get_detailed_metrics(y_true, y_pred, class_names)
    
    print(f"\n{eval_type} Evaluation Results:")
    print("=" * 60)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    print(f"\nPer-Class Results:")
    print("-" * 50)
    for class_name in class_names:
        class_metrics = metrics['classification_report'][class_name]
        print(f"{class_name:17}: "
              f"Precision: {class_metrics['precision']:.3f} "
              f"Recall: {class_metrics['recall']:.3f} "
              f"F1: {class_metrics['f1-score']:.3f}")
    
    print(f"\nMacro Average:")
    print("-" * 30)
    macro_avg = metrics['classification_report']['macro avg']
    print(f"Precision: {macro_avg['precision']:.3f}")
    print(f"Recall: {macro_avg['recall']:.3f}")
    print(f"F1-Score: {macro_avg['f1-score']:.3f}")
    
    print(f"\nWeighted Average:")
    print("-" * 30)
    weighted_avg = metrics['classification_report']['weighted avg']
    print(f"Precision: {weighted_avg['precision']:.3f}")
    print(f"Recall: {weighted_avg['recall']:.3f}")
    print(f"F1-Score: {weighted_avg['f1-score']:.3f}")


def analyze_errors(y_true, y_pred, class_names):
    """
    Analyze prediction errors to identify patterns
    
    Input:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nError Analysis:")
    print("=" * 40)
    
    total_errors = len(y_true) - np.trace(cm)
    print(f"Total errors: {total_errors}/{len(y_true)} ({100*total_errors/len(y_true):.1f}%)")
    
    print(f"\nMost common misclassifications:")
    print("-" * 40)
    
    error_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                error_pairs.append((cm[i, j], class_names[i], class_names[j]))
    
    # Sort by error count
    error_pairs.sort(reverse=True)
    
    for count, true_class, pred_class in error_pairs[:5]:  # Top 5 errors
        if count > 0:
            print(f"{true_class} → {pred_class}: {count} cases")
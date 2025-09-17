"""
Visualization utilities for training history and results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_training_history(history, save_path=None):
    """
    Plot comprehensive training history
    
    Input
        history (dict): Training history containing losses and accuracies
        save_path (str): Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot loss difference (overfitting indicator)
    if len(history['val_losses']) == len(history['train_losses']):
        loss_diff = np.array(history['val_losses']) - np.array(history['train_losses'])
        ax3.plot(epochs, loss_diff, 'purple', linewidth=2)
        ax3.set_title('Overfitting Indicator (Val Loss - Train Loss)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Difference')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
    
    # Plot accuracy difference
    if len(history['val_accuracies']) == len(history['train_accuracies']):
        acc_diff = np.array(history['train_accuracies']) - np.array(history['val_accuracies'])
        ax4.plot(epochs, acc_diff, 'orange', linewidth=2)
        ax4.set_title('Accuracy Gap (Train Acc - Val Acc)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy Difference (%)')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot enhanced confusion matrix with both counts and percentages
    
    Input:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Confusion matrix with counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1,
                cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Confusion matrix with percentages
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names, ax=ax2,
                cbar_kws={'label': 'Percentage'})
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()

def plot_class_distribution(labels, class_names, save_path=None):
    """
    Plot class distribution in the dataset
    
    Input:
        labels: List of labels
        class_names: List of class names
        save_path (str): Path to save the plot
    """
    from collections import Counter
    
    label_counts = Counter(labels)
    counts = [label_counts.get(class_name, 0) for class_name in class_names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    total = sum(counts)
    percentages = [f"{(c/total)*100:.1f}%" for c in counts]
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                pct, ha='center', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()

def plot_model_comparison(results_dict, save_path=None):
    """
    Plot comparison between different models or evaluation methods
    
    Input:
        results_dict: Dictionary with model names as keys and accuracies as values
        save_path (str): Path to save the plot
    """
    models = list(results_dict.keys())
    accuracies = list(results_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['lightblue', 'lightcoral', 'lightgreen', 'gold'])
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Models/Methods')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.show()

def plot_learning_curves(train_sizes, train_scores, val_scores, save_path=None):
    """
    Plot learning curves to analyze model performance vs dataset size
    
    Input:
        train_sizes: Array of training sizes
        train_scores: Training scores for each size
        val_scores: Validation scores for each size
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, train_scores, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, val_scores, 'o-', color='red', label='Validation Score')
    
    plt.title('Learning Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves plot saved to {save_path}")
    
    plt.show()
"""
Helper utilities for the brain tumor classification project
"""

import random
import numpy as np
import torch
import os
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report

def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    Helps to have the same split/when retraining
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_device():
    """
    Setup and returning the appropriate device (CUDA/CPU)
    
    Returns:
        torch.device: Device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    return device

def save_results(y_true, y_pred, class_names, history, eval_type, args):
    """
    Save  results to files
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        history: Training history
        eval_type: Evaluation type (e.g., "Standard", "TTA")
        args: Training arguments
    """
    # Calculate metrics
    test_accuracy = accuracy_score(y_true, y_pred) * 100
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/experiment_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    predictions_list = [int(pred) for pred in y_pred]
    true_labels_list = [int(label) for label in y_true]
    
    # Convert history values to Python native types
    clean_history = {}
    for key, value in history.items():
        if key == 'best_model_state':
            clean_history[key] = None  # Don't save model weights in JSON
        elif isinstance(value, list):
            clean_history[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
        elif isinstance(value, (np.floating, np.integer)):
            clean_history[key] = float(value)
        else:
            clean_history[key] = value
    
    # Save detailed results
    results = {
        'experiment_info': {
            'timestamp': timestamp,
            'eval_type': eval_type,
            'test_accuracy': float(test_accuracy),
            'best_val_accuracy': float(history['best_val_acc']),
            'training_args': vars(args) if hasattr(args, '__dict__') else args
        },
        'class_names': class_names,
        'predictions': predictions_list,
        'true_labels': true_labels_list,
        'training_history': clean_history
    }
    
    # Save as JSON
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    with open(f"{results_dir}/classification_report.txt", 'w', encoding='utf-8') as f:
        f.write("Brain Tumor Classification - Detailed Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment Timestamp: {timestamp}\n")
        f.write(f"Evaluation Type: {eval_type}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        f.write(f"Best Validation Accuracy: {history['best_val_acc']:.2f}%\n\n")
        
        f.write("Training Configuration:\n")
        f.write("-" * 30 + "\n")
        if hasattr(args, '__dict__'):
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("Class Distribution in Test Set:\n")
        f.write("-" * 30 + "\n")
        for i, class_name in enumerate(class_names):
            count = sum(1 for label in y_true if label == i)
            f.write(f"{class_name}: {count} samples\n")
        
        f.write("\nClassification Report:\n")
        f.write("-" * 30 + "\n")
        f.write(report)
    
    # Print summary
    print(f"\nResults Summary:")
    print("=" * 50)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Validation Accuracy: {history['best_val_acc']:.2f}%")
    
    
    print(f"\nDetailed results saved to: {results_dir}/")
    
    return results_dir

def load_model_checkpoint(checkpoint_path, model, device):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model: PyTorch model to load weights into
        device: Device to load model on
        
    Returns:
        dict: Checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    return checkpoint

def create_project_structure():
    """
    Create the complete project directory structure
    """
    directories = [
        'src',
        'src/models',
        'src/data',
        'src/training',
        'src/evaluation',
        'src/utils',
        'models',
        'results',
        'plots',
        'data',
        'notebooks',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Project structure created successfully!")

def format_time(seconds):
    """
    Format time in seconds to  readable format
    
    Args:
        seconds : Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"

def count_parameters(model):
    """
    Count the number of parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def get_model_size(model):
    """
    Get model size in MB
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def validate_dataset_structure(data_dir):
    """
    Validate that the dataset has the correct structure
    
    Args:
        data_dir : Path to dataset directory
        
    Returns:
        bool: True if structure is valid
    """
    expected_classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory {data_dir} does not exist")
        return False
    
    missing_classes = []
    for class_name in expected_classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            missing_classes.append(class_name)
    
    if missing_classes:
        print(f"Error: Missing class directories: {missing_classes}")
        print(f"Expected structure:")
        print(f"{data_dir}/")
        for class_name in expected_classes:
            print(f"  ├── {class_name}/")
        return False
    
    # Check if directories contain images
    empty_classes = []
    for class_name in expected_classes:
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            empty_classes.append(class_name)
    
    if empty_classes:
        print(f"Warning: Empty class directories: {empty_classes}")
    
    print("Dataset structure validation passed!")
    return True

def get_system_info():
    """
    Get system information for logging
    
    Returns:
        dict: System information
    """
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'torch_version': torch.__version__,
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    
    return info

def log_experiment_start(args):
    """
    Log experiment start with configuration
    
    Args:
        args: Experiment arguments
    """
    print("Starting Brain Tumor Classification Experiment")
    print("=" * 60)
    
    # System info
    sys_info = get_system_info()
    print("System Information:")
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    
    print(f"\nExperiment Configuration:")
    if hasattr(args, '__dict__'):
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
    
    print("=" * 60)

def cleanup_old_results(results_dir='results', keep_last=5):
    """
    Clean up old result directories, keeping only the most recent ones
    
    Args:
        results_dir (str): Results directory path
        keep_last (int): Number of recent experiments to keep
    """
    if not os.path.exists(results_dir):
        return
    
    # Get all experiment directories
    experiment_dirs = [d for d in os.listdir(results_dir) 
                      if d.startswith('experiment_') and os.path.isdir(os.path.join(results_dir, d))]
    
    if len(experiment_dirs) <= keep_last:
        return
    
    # Sort by modification time
    experiment_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
    
    # Remove old directories
    for old_dir in experiment_dirs[:-keep_last]:
        old_path = os.path.join(results_dir, old_dir)
        try:
            import shutil
            shutil.rmtree(old_path)
            print(f"Removed old experiment: {old_dir}")
        except Exception as e:
            print(f"Failed to remove {old_dir}: {e}")
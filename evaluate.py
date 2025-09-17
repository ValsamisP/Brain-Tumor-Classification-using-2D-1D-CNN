"""
Evaluation script for pre-trained brain tumor classification model
"""

"""
This file helps me to evaluate models that trained before without retraining from the beginning
Evaluate/compare two models
Test the model in different datasets and can also change some of the setting of the model(like batch)

python evaluate.py --model_path models/my_model.pth --data_dir new_test_data -> Test on different datasets (in my case data_brain_tumor)

python evaluate.py --model_path models/my_model.pth --use_tta -> Testi with TTA 

python evaluate.py --model_path models/my_model.pth --batch_size 64 -> Test in different batches
"""
import torch
import argparse
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.models.cnn import Enhanced_CNN2D1D
from src.data.dataset import BrainTumorDataset, load_data, get_transforms
from src.evaluation.evaluator import evaluate_model, evaluate_model_with_tta, print_evaluation_results
from src.utils.visualization import plot_confusion_matrix
from src.utils.helpers import setup_device, load_model_checkpoint, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Brain Tumor Classification Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data_brain_tumor',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--use_tta', action='store_true',
                        help='Use Test Time Augmentation')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save confusion matrix plot')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = setup_device()
    
    print("Brain Tumor Classification - Model Evaluation")
    print("=" * 60)
    
    # Load and prepare data
    print("Loading data...")
    image_paths, labels, class_names = load_data(args.data_dir)
    
    # Encode labels and create same splits as training
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, encoded_labels, test_size=0.3, random_state=args.seed, stratify=encoded_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )
    
    # Get transforms
    _, test_transform = get_transforms()
    
    # Create test dataset
    test_dataset = BrainTumorDataset(X_test, y_test, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {class_names}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = Enhanced_CNN2D1D(num_classes=len(class_names)).to(device)
    checkpoint = load_model_checkpoint(args.model_path, model, device)
    
    
    print("\nEvaluating model...")
    if args.use_tta:
        print("Using Test Time Augmentation...")
        y_pred, y_true = evaluate_model_with_tta(model, test_loader, class_names, device)
        eval_type = "TTA"
    else:
        print("Standard evaluation...")
        y_pred, y_true = evaluate_model(model, test_loader, device)
        eval_type = "Standard"
    
    from sklearn.metrics import accuracy_score
    test_accuracy = accuracy_score(y_true, y_pred) * 100
    
    print_evaluation_results(y_true, y_pred, class_names, eval_type)
    
    # Save confusion matrix plot
    if args.save_plots:
        save_path = f'plots/confusion_matrix_{eval_type.lower()}.png'
        os.makedirs('plots', exist_ok=True)
        plot_confusion_matrix(y_true, y_pred, class_names, save_path)
    else:
        plot_confusion_matrix(y_true, y_pred, class_names)
    
    print(f"\nEvaluation completed!")
    print(f"Final {eval_type} Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
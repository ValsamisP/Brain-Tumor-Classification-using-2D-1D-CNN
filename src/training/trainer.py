"""Training utilities and training loop"""

import torch
import torch.nn as nn

def train_model(model,train_loader,val_loader,criterion,optimizer,scheduler,num_epochs,device):
    """
    Train the model including early stopping
    
    
    Input:
    model -> pytorch model
    train_loader -> Training data loader
    val_loader -> Validation data loader
    criterion -> Loss function
    Optimizer -> Optimizer
    scheduler -> learning rate scheduler
    num_epochs -> maximum number of epochs
    device -> Training device -> CPU/GPU
    """

    # Training History
    history ={
        'train_losses':[],
        'train_accuracies':[],
        'val_losses':[],
        'val_accuracies':[],
        'best_model_state':None,
        'best_val_acc':0.0
    }

    # Early Stopping parameters
    patience = 15
    patience_counter = 0

    print(f"Training for maximum {num_epochs} epochs with early stopping (patience={patience})")
    print("=" * 70)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images,labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs,labels)

                val_loss += loss.item()
                _ , predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            
        train_acc = 100 * train_correct/train_total
        val_acc = 100 * val_correct/val_total
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)

        # Store metrics
        history['train_losses'].append(avg_train_loss)
        history['train_accuracies'].append(train_acc)
        history['val_losses'].append(avg_val_loss)
        history['val_accuracies'].append(val_acc)

        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_model_state'] = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step(avg_val_loss)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            overfitting_gap = train_acc - val_acc
            print(f'Epoch [{epoch+1:3d}/{num_epochs}] '
                  f'Train: {avg_train_loss:.4f}/{train_acc:5.2f}% '
                  f'Val: {avg_val_loss:.4f}/{val_acc:5.2f}% '
                  f'Gap: {overfitting_gap:+5.2f}% '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}') 
            
            # Warning for high overfitting
            if overfitting_gap > 10:
                print(f'         Warning: High overfitting detected (gap: {overfitting_gap:.1f}%)')
            
            # Warning for suspiciously high early accuracy
            if val_acc > 90 and epoch < 20:
                print(f'         Note: Very high validation accuracy early in training')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            print(f'Best validation accuracy: {history["best_val_acc"]:.2f}%')
            break
    
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {history["best_val_acc"]:.2f}%')
    
    return history


def validate_model(model, val_loader, criterion, device):
    """
    Validate the model on validation set
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device (CPU/GPU)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy
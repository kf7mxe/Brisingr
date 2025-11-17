import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TinyWakeWordCNN(nn.Module):
    """Ultra-lightweight CNN for wake word detection"""
    def __init__(self, input_dim=13, sequence_length=101):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        # Tiny CNN layers
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Global average pooling instead of fixed size
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32, 2)
        
    def forward(self, x):
        # x shape: (batch, time, features) -> (batch, features, time)
        x = x.transpose(1, 2)
        
        # Convolution layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

class WakeWordDataset(Dataset):
    """Optimized dataset with data augmentation"""
    def __init__(self, x, y, augment=False):
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)
        self.augment = augment
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        # Simple data augmentation
        if self.augment and torch.rand(1) < 0.3:
            # Add small amount of noise
            noise = torch.randn_like(x) * 0.01
            x = x + noise
            
        return x, y

def create_balanced_dataloader(x, y, batch_size=32, shuffle=True, augment=False):
    """Create dataloader with balanced sampling"""
    dataset = WakeWordDataset(x, y, augment=augment)
    
    if shuffle:
        # Create weighted sampler for balanced training
        class_counts = np.bincount(y.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y.astype(int)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def train_model(data_file='optimized_wake_word_data.npz', 
                model_save_path='tiny_wake_word_model.pth',
                epochs=50, batch_size=32, learning_rate=0.001):
    
    # Load data
    print("Loading data...")
    data = np.load(data_file)
    
    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    y_val = data['y_val']
    x_test = data['x_test']
    y_test = data['y_test']
    
    print(f"Data shapes:")
    print(f"Train: {x_train.shape}, {y_train.shape}")
    print(f"Val: {x_val.shape}, {y_val.shape}")
    print(f"Test: {x_test.shape}, {y_test.shape}")
    
    # Create model
    input_dim = x_train.shape[2]  # Number of MFCC features
    sequence_length = x_train.shape[1]  # Time frames
    
    model = TinyWakeWordCNN(input_dim=input_dim, sequence_length=sequence_length).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create data loaders
    train_loader = create_balanced_dataloader(x_train, y_train, batch_size, augment=True)
    val_loader = create_balanced_dataloader(x_val, y_val, batch_size, shuffle=False)
    test_loader = create_balanced_dataloader(x_test, y_test, batch_size, shuffle=False)
    
    # Loss and optimizer with class weighting
    class_counts = np.bincount(y_train.astype(int))
    class_weights = torch.FloatTensor([class_counts[1]/class_counts[0], 1.0]).to(device)
    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        val_acc = 100. * val_correct / val_total
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss / len(val_loader))
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'input_dim': input_dim,
                'sequence_length': sequence_length
            }, model_save_path)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=['Non-Wake', 'Wake Word']))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    return model, test_acc

def export_for_inference(model_path='tiny_wake_word_model.pth', 
                        export_path='tiny_wake_word_optimized.pt'):
    """Export model for optimized inference"""
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = TinyWakeWordCNN(
        input_dim=checkpoint['input_dim'],
        sequence_length=checkpoint['sequence_length']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, checkpoint['sequence_length'], checkpoint['input_dim'])
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for inference
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    # Save
    traced_model.save(export_path)
    print(f"Optimized model saved to {export_path}")
    
    return export_path

if __name__ == "__main__":
    # Train the model
    model, test_acc = train_model(epochs=30)
    
    # Export for inference
    export_for_inference()
    
    print(f"Training complete! Final test accuracy: {test_acc:.2f}%")
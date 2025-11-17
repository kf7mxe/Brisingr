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
    """Original model from your code for comparison"""
    def __init__(self, input_dim=13, sequence_length=101):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class UltraTinyWakeWordCNN(nn.Module):
    """Ultra-optimized model with minimal parameters"""
    def __init__(self, input_dim=13, sequence_length=101):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        # Reduce channels dramatically
        self.conv1 = nn.Conv1d(input_dim, 16, kernel_size=5, stride=2, padding=2)  # Stride reduces computation
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1)         # Even fewer channels
        
        # Use GroupNorm instead of BatchNorm (better for small batches/inference)
        self.gn1 = nn.GroupNorm(4, 16)  # 4 groups of 4 channels
        self.gn2 = nn.GroupNorm(2, 8)   # 2 groups of 4 channels
        
        # Calculate output size after convolutions
        # After conv1: (101 + 2*2 - 5) // 2 + 1 = 51
        # After conv2: (51 + 2*1 - 3) // 2 + 1 = 25
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Single small FC layer
        self.fc = nn.Linear(8, 2)
        
        # No dropout for inference speed
        
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, time)
        
        # Use ReLU6 for better quantization
        x = F.relu6(self.gn1(self.conv1(x)))
        x = F.relu6(self.gn2(self.conv2(x)))
        
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class DepthwiseSeparableConv1d(nn.Module):
    """Memory and computation efficient convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                  stride=stride, padding=padding, groups=in_channels)
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetStyleWakeWord(nn.Module):
    """MobileNet-inspired ultra-efficient model"""
    def __init__(self, input_dim=13, sequence_length=101):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        # Initial standard conv
        self.conv1 = nn.Conv1d(input_dim, 16, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 16)
        
        # Depthwise separable convolutions
        self.dw_conv1 = DepthwiseSeparableConv1d(16, 16, kernel_size=3, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(4, 16)
        
        self.dw_conv2 = DepthwiseSeparableConv1d(16, 8, kernel_size=3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(2, 8)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, 2)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = F.relu6(self.gn1(self.conv1(x)))
        x = F.relu6(self.gn2(self.dw_conv1(x)))
        x = F.relu6(self.gn3(self.dw_conv2(x)))
        
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class StreamingWakeWordCNN(nn.Module):
    """Optimized for streaming/real-time inference"""
    def __init__(self, input_dim=13, chunk_size=32):  # Process smaller chunks
        super().__init__()
        self.input_dim = input_dim
        self.chunk_size = chunk_size
        
        # Very simple architecture for streaming
        self.conv1 = nn.Conv1d(input_dim, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(12, 8, kernel_size=3, stride=2, padding=1)
        
        # Use running statistics for normalization in streaming
        self.bn1 = nn.BatchNorm1d(12, track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(8, track_running_stats=True)
        
        # Adaptive pooling handles variable input sizes
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, 2)
        
    def forward(self, x):
        # x shape: (batch, time, features) or (batch, chunk_size, features)
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class QuantizedFriendlyWakeWord(nn.Module):
    """Designed specifically for optimal quantization"""
    def __init__(self, input_dim=13, sequence_length=101):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        # Use power-of-2 channel sizes for better quantization
        self.conv1 = nn.Conv1d(input_dim, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        
        # Reduce spatial dimension early
        self.conv3 = nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(8)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, 2)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        # Use ReLU6 for better int8 quantization
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        x = F.relu6(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# Update the dataset to handle variable sequence lengths
class OptimizedWakeWordDataset(Dataset):
    def __init__(self, x, y, target_length=None, augment=False):
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.target_length = target_length
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        # Truncate or pad to target length for consistent processing
        if self.target_length and x.shape[0] != self.target_length:
            if x.shape[0] > self.target_length:
                # Random crop during training, center crop during inference
                if self.augment:
                    start = torch.randint(0, x.shape[0] - self.target_length + 1, (1,))
                    x = x[start:start + self.target_length]
                else:
                    start = (x.shape[0] - self.target_length) // 2
                    x = x[start:start + self.target_length]
            else:
                # Pad with zeros
                pad_size = self.target_length - x.shape[0]
                x = F.pad(x, (0, 0, 0, pad_size))
        
        # Lightweight augmentation
        if self.augment and torch.rand(1) < 0.2:
            # Add minimal noise
            noise = torch.randn_like(x) * 0.005
            x = x + noise
            
        return x, y


def compare_model_efficiency():
    """Compare different model architectures for efficiency"""
    
    # Model configurations
    try:
        models = {
            'Original': TinyWakeWordCNN(input_dim=13, sequence_length=101),
            'UltraTiny': UltraTinyWakeWordCNN(input_dim=13, sequence_length=101),
            'MobileNet': MobileNetStyleWakeWord(input_dim=13, sequence_length=101),
            'Streaming': StreamingWakeWordCNN(input_dim=13, chunk_size=32),
            'QuantFriendly': QuantizedFriendlyWakeWord(input_dim=13, sequence_length=101)
        }
    except Exception as e:
        print(f"Error creating models: {e}")
        return
    
    # Count parameters and FLOPs
    print("Model Efficiency Comparison:")
    print("-" * 60)
    print(f"{'Model':<15} {'Parameters':<12} {'Size (KB)':<12} {'Relative':<10}")
    print("-" * 60)
    
    original_params = sum(p.numel() for p in models['Original'].parameters())
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        size_kb = total_params * 4 / 1024  # Assuming float32
        relative_size = total_params / original_params
        
        print(f"{name:<15} {total_params:<12,} {size_kb:<12.1f} {relative_size:<10.2f}x")
    
    # Test inference speed
    print("\nInference Speed Test (CPU):")
    print("-" * 40)
    
    batch_size = 1
    sequence_length = 101
    input_dim = 13
    
    test_input = torch.randn(batch_size, sequence_length, input_dim)
    test_input_streaming = torch.randn(batch_size, 32, input_dim)  # Smaller chunk for streaming
    
    import time
    
    for name, model in models.items():
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                if name == 'Streaming':
                    _ = model(test_input_streaming)
                else:
                    _ = model(test_input)
        
        # Timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                if name == 'Streaming':
                    _ = model(test_input_streaming)
                else:
                    _ = model(test_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
        print(f"{name:<15}: {avg_time:.2f} ms per inference")


def export_ultra_optimized_model(model, model_path, export_path):
    """Export model with maximum optimization for inference"""
    
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 101, 13)
    if isinstance(model, StreamingWakeWordCNN):
        example_input = torch.randn(1, 32, 13)
    
    print("Starting model export optimization...")
    
    try:
        # Method 1: Try standard TorchScript tracing first (most compatible)
        print("Attempting standard TorchScript export...")
        
        # First, trace the original float model
        traced_model = torch.jit.trace(model, example_input)
        optimized_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save the standard optimized model
        standard_export_path = export_path.replace('.pt', '_standard.pt')
        optimized_model.save(standard_export_path)
        print(f"Standard optimized model saved to {standard_export_path}")
        
        # Test the standard model
        loaded_standard = torch.jit.load(standard_export_path)
        with torch.no_grad():
            output_standard = loaded_standard(example_input)
            print(f"Standard model test - Output shape: {output_standard.shape}")
        
    except Exception as e:
        print(f"Standard export failed: {e}")
        standard_export_path = None
    
    try:
        # Method 2: Try quantization with proper backend setup
        print("\nAttempting quantized export...")
        
        # Clone the model for quantization to avoid modifying the original
        model_for_quant = type(model)(model.input_dim, 
                                     getattr(model, 'sequence_length', 101) if hasattr(model, 'sequence_length') 
                                     else getattr(model, 'chunk_size', 32))
        model_for_quant.load_state_dict(model.state_dict())
        model_for_quant.eval()
        
        # Set quantization backend
        torch.backends.quantized.engine = 'qnnpack'
        
        # Configure quantization
        model_for_quant.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
        
        # Prepare for quantization
        prepared_model = torch.ao.quantization.prepare(model_for_quant)
        
        # Calibration with multiple samples
        print("Calibrating quantized model...")
        with torch.no_grad():
            for i in range(50):  # More calibration samples
                if isinstance(model, StreamingWakeWordCNN):
                    calib_input = torch.randn(1, 32, 13) * 0.1 + torch.randn(1, 32, 13) * 0.01
                else:
                    calib_input = torch.randn(1, 101, 13) * 0.1 + torch.randn(1, 101, 13) * 0.01
                prepared_model(calib_input)
        
        # Convert to quantized model
        quantized_model = torch.ao.quantization.convert(prepared_model)
        
        # Test quantized model before tracing
        with torch.no_grad():
            quant_output = quantized_model(example_input)
            print(f"Quantized model test - Output shape: {quant_output.shape}")
        
        # Try to trace the quantized model with error handling
        try:
            quantized_traced = torch.jit.trace(quantized_model, example_input)
            quantized_optimized = torch.jit.optimize_for_inference(quantized_traced)
            
            quantized_export_path = export_path.replace('.pt', '_quantized.pt')
            quantized_optimized.save(quantized_export_path)
            print(f"Quantized model saved to {quantized_export_path}")
            
        except Exception as trace_error:
            print(f"Quantized model tracing failed: {trace_error}")
            # Save the quantized model without tracing
            quantized_export_path = export_path.replace('.pt', '_quantized_state.pth')
            torch.save({
                'model_state_dict': quantized_model.state_dict(),
                'model_class': type(model).__name__,
                'input_dim': model.input_dim,
                'sequence_length': getattr(model, 'sequence_length', 101) if hasattr(model, 'sequence_length') 
                                 else getattr(model, 'chunk_size', 32),
                'quantized': True
            }, quantized_export_path)
            print(f"Quantized model state dict saved to {quantized_export_path}")
            
    except Exception as e:
        print(f"Quantization export failed: {e}")
    
    try:
        # Method 3: Mobile optimization (if available)
        print("\nAttempting mobile optimization...")
        
        if 'traced_model' in locals():
            # Try mobile optimization on the standard traced model
            mobile_optimized = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
            mobile_export_path = export_path.replace('.pt', '_mobile.ptl')
            mobile_optimized._save_for_lite_interpreter(mobile_export_path)
            print(f"Mobile optimized model saved to {mobile_export_path}")
            
    except Exception as e:
        print(f"Mobile optimization not available or failed: {e}")
    
    # Method 4: Save comprehensive model info for manual optimization
    try:
        print("\nSaving comprehensive model information...")
        
        # Calculate model statistics
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # Get model architecture info
        model_info = {
            'model_state_dict': model.state_dict(),
            'model_class': type(model).__name__,
            'input_dim': model.input_dim,
            'sequence_length': getattr(model, 'sequence_length', 101) if hasattr(model, 'sequence_length') 
                             else getattr(model, 'chunk_size', 32),
            'total_parameters': total_params,
            'model_size_mb': model_size_mb,
            'architecture_summary': str(model),
            'export_timestamp': torch.tensor([1.0])  # Placeholder for timestamp
        }
        
        comprehensive_path = export_path.replace('.pt', '_comprehensive.pth')
        torch.save(model_info, comprehensive_path)
        print(f"Comprehensive model info saved to {comprehensive_path}")
        
        print(f"\nModel Export Summary:")
        print(f"- Parameters: {total_params:,}")
        print(f"- Size: {model_size_mb:.2f} MB")
        print(f"- Architecture: {type(model).__name__}")
        
    except Exception as e:
        print(f"Comprehensive export failed: {e}")
    
    print("\nModel export process completed!")



def train_optimized_model(model_class=UltraTinyWakeWordCNN, 
                         data_file='optimized_wake_word_data.npz',
                         model_save_path='ultra_tiny_wake_word.pth',
                         epochs=50, batch_size=64, learning_rate=0.002):
    """Train one of the optimized models"""
    
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
    input_dim = x_train.shape[2]
    sequence_length = x_train.shape[1]
    
    if model_class == StreamingWakeWordCNN:
        model = model_class(input_dim=input_dim, chunk_size=32).to(device)
        target_length = 32  # Use smaller chunks
    else:
        model = model_class(input_dim=input_dim, sequence_length=sequence_length).to(device)
        target_length = sequence_length
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create optimized datasets
    train_dataset = OptimizedWakeWordDataset(x_train, y_train, target_length=target_length, augment=True)
    val_dataset = OptimizedWakeWordDataset(x_val, y_val, target_length=target_length, augment=False)
    test_dataset = OptimizedWakeWordDataset(x_test, y_test, target_length=target_length, augment=False)
    
    # Data loaders with larger batch size for efficiency
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Optimized training setup
    class_counts = np.bincount(y_train.astype(int))
    class_weights = torch.FloatTensor([class_counts[1]/class_counts[0], 1.0]).to(device)
    criterion = nn.NLLLoss(weight=class_weights)
    
    # Use AdamW with higher learning rate for smaller models
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, 
                                            steps_per_epoch=len(train_loader), epochs=epochs)
    
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        val_acc = 100. * val_correct / val_total
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'input_dim': input_dim,
                'sequence_length': sequence_length,
                'model_class': model_class.__name__
            }, model_save_path)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)
    
    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    return model, test_acc


if __name__ == "__main__":
    # Compare model efficiency
    compare_model_efficiency()
    
    print("\n" + "="*60)
    print("Training ultra-optimized model...")
    
    # Train the most efficient model
    model, test_acc = train_optimized_model(
        model_class=UltraTinyWakeWordCNN,
        epochs=30,
        batch_size=64
    )
    
    # Export for maximum efficiency
    export_ultra_optimized_model(
        model, 
        'ultra_tiny_wake_word.pth',
        'ultra_tiny_wake_word_mobile.pt'
    )
    
    print(f"\nOptimization complete! Final test accuracy: {test_acc:.2f}%")
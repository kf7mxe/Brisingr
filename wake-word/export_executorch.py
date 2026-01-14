
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
from executorch.exir import to_edge, EdgeCompileConfig
import os

# --- Model Definitions (Copy from training-v3.py) ---

class TinyWakeWordCNN(nn.Module):
    """Original model"""
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
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# --- Export Logic ---

def export_model(model_class=TinyWakeWordCNN, model_path=None, output_path="wake_word.pte"):
    print(f"Exporting {model_class.__name__} to {output_path}...")
    
    # Initialize model
    model = model_class()
    model.eval()
    
    # Load weights if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            print("Using random initialization.")
    else:
        print("No weights found or provided. Using random initialization.")

    # Create example input
    example_input = (torch.randn(1, 101, 13),)
    
    try:
        # 1. torch.export (Capture information)
        print("Tracing model...")
        exported_model = export(model, example_input)
        
        # 2. to_edge (Lower to Edge IR)
        print("Lowering to Edge IR...")
        edge_program = to_edge(exported_model)
        
        # 3. to_executorch (Serialize)
        print("Serializing to ExecuTorch...")
        executorch_program = edge_program.to_executorch()
        
        # Save
        with open(output_path, "wb") as f:
            f.write(executorch_program.buffer)
            
        print(f"Successfully exported to {output_path}")
        
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for likely model paths
    possible_paths = [
        "ultra_tiny_wake_word.pth",
        "training/ultra_tiny_wake_word.pth",
        "model-convolution-13-101-android_lite_model.pt1" # Note: likely state dict won't match direct load for ScriptModule
    ]
    
    selected_path = None
    for p in possible_paths:
        if os.path.exists(p):
            selected_path = p
            break
            
    export_model(model_path=selected_path)

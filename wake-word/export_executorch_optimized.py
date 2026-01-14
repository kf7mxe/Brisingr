#!/usr/bin/env python3
"""
Optimized ExecuTorch export with mobile backend support (XNNPACK for CPU/GPU acceleration)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
from executorch.exir import to_edge, EdgeCompileConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
import os

# --- Model Definition ---

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

def export_model_optimized(model_path=None, output_path="wake_word_xnnpack.pte"):
    """Export model with XNNPACK backend for mobile optimization"""
    print(f"Exporting optimized model to {output_path}...")

    # Initialize model
    model = TinyWakeWordCNN()
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
        print("No weights found. Using random initialization.")

    # Create example input - IMPORTANT: This must match what Android sends!
    example_input = (torch.randn(1, 101, 13),)

    try:
        # Test the model first
        print("Testing model...")
        with torch.no_grad():
            test_output = model(*example_input)
            print(f"Model output shape: {test_output.shape}")
            print(f"Model output sample: {test_output[0]}")

        # 1. torch.export (Capture)
        print("\n1. Exporting model...")
        exported_model = export(model, example_input)

        # 2. to_edge (Lower to Edge IR with XNNPACK backend)
        print("2. Lowering to Edge IR with XNNPACK optimization...")
        edge_program = to_edge(
            exported_model,
            compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )

        # 3. Partition for XNNPACK (enables CPU/GPU acceleration)
        print("3. Partitioning for XNNPACK backend...")
        edge_program_xnnpack = edge_program.to_backend(XnnpackPartitioner())

        # 4. to_executorch (Serialize)
        print("4. Serializing to ExecuTorch...")
        executorch_program = edge_program_xnnpack.to_executorch()

        # Save
        with open(output_path, "wb") as f:
            f.write(executorch_program.buffer)

        print(f"\n✓ Successfully exported XNNPACK-optimized model to {output_path}")
        print(f"  This model will use CPU SIMD instructions and can leverage NPU/DSP on supported devices")

        return True

    except Exception as e:
        print(f"XNNPACK export failed: {e}")
        print("\nFalling back to basic export without XNNPACK...")

        try:
            # Fallback: Basic export without backend optimization
            exported_model = export(model, example_input)
            edge_program = to_edge(exported_model)
            executorch_program = edge_program.to_executorch()

            fallback_path = output_path.replace('.pte', '_basic.pte')
            with open(fallback_path, "wb") as f:
                f.write(executorch_program.buffer)

            print(f"✓ Basic model exported to {fallback_path}")
            return False

        except Exception as e2:
            print(f"Basic export also failed: {e2}")
            import traceback
            traceback.print_exc()
            return False

def export_model_quantized(model_path=None, output_path="wake_word_quantized.pte"):
    """Export model with dynamic quantization for even better efficiency"""
    print(f"\nExporting quantized model to {output_path}...")

    # Initialize model
    model = TinyWakeWordCNN()
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
            return False
    else:
        print("No weights found. Quantization requires trained weights.")
        return False

    try:
        # Apply dynamic quantization
        print("Applying dynamic quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv1d},  # Quantize these layer types
            dtype=torch.qint8
        )

        # Test quantized model
        example_input = (torch.randn(1, 101, 13),)
        with torch.no_grad():
            test_output = quantized_model(*example_input)
            print(f"Quantized model output: {test_output[0]}")

        # Export
        print("Exporting quantized model...")
        exported_model = export(quantized_model, example_input)
        edge_program = to_edge(exported_model)
        executorch_program = edge_program.to_executorch()

        with open(output_path, "wb") as f:
            f.write(executorch_program.buffer)

        print(f"✓ Quantized model exported to {output_path}")
        print(f"  Expected benefits: 2-4x smaller, 2-3x faster inference")
        return True

    except Exception as e:
        print(f"Quantization export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Check for model paths
    possible_paths = [
        "ultra_tiny_wake_word.pth",
        "training/ultra_tiny_wake_word.pth",
        "tiny_wake_word_optimized.pt"
    ]

    selected_path = None
    for p in possible_paths:
        if os.path.exists(p):
            selected_path = p
            print(f"Found model at: {p}")
            break

    if not selected_path:
        print("WARNING: No trained model found. Exporting with random weights.")

    # Export with XNNPACK optimization
    print("="*60)
    print("EXPORTING XNNPACK-OPTIMIZED MODEL")
    print("="*60)
    success = export_model_optimized(model_path=selected_path, output_path="wake_word_xnnpack.pte")

    # Also try quantized export if we have trained weights
    if selected_path:
        print("\n" + "="*60)
        print("EXPORTING QUANTIZED MODEL (SMALLER & FASTER)")
        print("="*60)
        export_model_quantized(model_path=selected_path, output_path="wake_word_quantized.pte")

    if success:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Choose which model to use:")
        print("   - wake_word_xnnpack.pte: Full precision, XNNPACK acceleration")
        print("   - wake_word_quantized.pte: Quantized (smaller, faster, slight accuracy loss)")
        print("")
        print("2. Copy your chosen model:")
        print("   cp wake_word_xnnpack.pte ../app/apps/src/androidMain/assets/wake_word.pte")
        print("   OR")
        print("   cp wake_word_quantized.pte ../app/apps/src/androidMain/assets/wake_word.pte")
        print("")
        print("3. Rebuild and run the Android app")
        print("="*60)

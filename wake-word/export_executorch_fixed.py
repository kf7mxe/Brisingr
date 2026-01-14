#!/usr/bin/env python3
"""
Fixed ExecuTorch export that properly extracts weights from optimized TorchScript models.

The key insight: When a model is exported with torch.jit.optimize_for_inference(),
BatchNorm layers are fused into Conv layers, and weights are stored as inline
graph constants rather than named parameters. This script extracts those constants
and creates a matching model architecture.

Usage:
    python export_executorch_fixed.py [model_path]

    If model_path is not provided, looks for tiny_wake_word_optimized.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
from executorch.exir import to_edge, EdgeCompileConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
import os
import sys


class TinyWakeWordCNN_NoBN(nn.Module):
    """
    Model without BatchNorm - for loading weights from optimized TorchScript
    where BatchNorm has been fused into Conv layers.
    """
    def __init__(self, input_dim=13):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def extract_weights_from_torchscript(model_path):
    """
    Extract weights from an optimized TorchScript model.

    When torch.jit.optimize_for_inference() is used, weights become
    prim::Constant nodes in the graph rather than named parameters.
    """
    print(f"Loading TorchScript model: {model_path}")
    scripted = torch.jit.load(model_path, map_location='cpu')
    scripted.eval()

    # Extract tensor constants from graph
    graph = scripted.graph
    constants = {}

    for node in graph.nodes():
        if node.kind() == 'prim::Constant':
            output = node.output()
            if output.type().str().startswith('Tensor'):
                try:
                    if node.hasAttribute('value'):
                        tensor_val = node.t('value')
                        name = output.debugName()
                        constants[name] = tensor_val
                        print(f"  Found constant {name}: {tensor_val.shape}")
                except Exception as e:
                    pass

    if len(constants) < 6:
        print(f"WARNING: Expected 6 tensor constants, found {len(constants)}")
        print("This model may have a different architecture.")
        return None, None

    # Map constants to layer weights based on graph order
    # The graph shows: conv1(7,8) -> conv2(17,18) -> fc(33,35)
    # We identify by shape and position in graph

    # Find weights by shape
    conv1_weight = None
    conv1_bias = None
    conv2_weight = None
    conv2_bias = None
    fc_weight = None
    fc_bias = None

    # Analyze graph operations to understand the order
    conv_inputs = []
    for node in graph.nodes():
        if node.kind() == 'aten::conv1d':
            inputs = [i.debugName() for i in node.inputs()]
            conv_inputs.append(inputs)

    if len(conv_inputs) >= 2:
        # First conv uses inputs[1] as weight, inputs[2] as bias
        conv1_weight_name = conv_inputs[0][1]
        conv1_bias_name = conv_inputs[0][2]
        conv2_weight_name = conv_inputs[1][1]
        conv2_bias_name = conv_inputs[1][2]

        conv1_weight = constants.get(conv1_weight_name)
        conv1_bias = constants.get(conv1_bias_name)
        conv2_weight = constants.get(conv2_weight_name)
        conv2_bias = constants.get(conv2_bias_name)

    # Find fc weights (remaining tensors)
    for name, tensor in constants.items():
        shape = list(tensor.shape)
        if len(shape) == 2 and 2 in shape and 32 in shape:
            fc_weight = tensor
        elif shape == [2]:
            fc_bias = tensor

    if any(x is None for x in [conv1_weight, conv1_bias, conv2_weight, conv2_bias, fc_weight, fc_bias]):
        print("ERROR: Could not extract all required weights!")
        print(f"  conv1_weight: {conv1_weight is not None}")
        print(f"  conv1_bias: {conv1_bias is not None}")
        print(f"  conv2_weight: {conv2_weight is not None}")
        print(f"  conv2_bias: {conv2_bias is not None}")
        print(f"  fc_weight: {fc_weight is not None}")
        print(f"  fc_bias: {fc_bias is not None}")
        return None, None

    # Build state dict
    # Note: fc_weight may need transpose depending on how matmul is used
    # Original graph uses matmul(input, weight) so weight is [in, out]
    # Linear expects [out, in], so transpose
    state_dict = {
        'conv1.weight': conv1_weight,
        'conv1.bias': conv1_bias,
        'conv2.weight': conv2_weight,
        'conv2.bias': conv2_bias,
        'fc.weight': fc_weight.t() if fc_weight.shape[0] == 32 else fc_weight,
        'fc.bias': fc_bias,
    }

    print("\nExtracted state dict:")
    for k, v in state_dict.items():
        print(f"  {k}: {v.shape}")

    return state_dict, scripted


def verify_extraction(state_dict, original_model):
    """Verify extracted weights produce same output as original model."""
    model = TinyWakeWordCNN_NoBN()
    model.load_state_dict(state_dict)
    model.eval()

    # Test with random inputs
    for i in range(3):
        test_input = torch.randn(1, 101, 13)
        with torch.no_grad():
            out_original = original_model(test_input.clone())
            out_new = model(test_input.clone())

        if not torch.allclose(out_original, out_new, atol=1e-5):
            print(f"ERROR: Outputs don't match on test {i+1}!")
            print(f"  Original: {out_original}")
            print(f"  New: {out_new}")
            return False

    print("✓ Verification passed: outputs match original model")
    return True


def export_to_executorch(state_dict, output_path="wake_word_xnnpack.pte"):
    """Export model with extracted weights to ExecuTorch."""

    # Create and load model
    model = TinyWakeWordCNN_NoBN()
    model.load_state_dict(state_dict)
    model.eval()

    example_input = (torch.randn(1, 101, 13),)

    print("\nExporting to ExecuTorch...")

    # 1. Export
    print("  1. Exporting model...")
    exported_model = export(model, example_input)

    # 2. To Edge
    print("  2. Converting to Edge IR...")
    edge_program = to_edge(
        exported_model,
        compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )

    # 3. XNNPACK
    print("  3. Applying XNNPACK optimization...")
    edge_program_xnnpack = edge_program.to_backend(XnnpackPartitioner())

    # 4. Serialize
    print("  4. Serializing...")
    executorch_program = edge_program_xnnpack.to_executorch()

    # Save
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    file_size = os.path.getsize(output_path) / 1024
    print(f"\n✓ Successfully exported to: {output_path}")
    print(f"  File size: {file_size:.1f} KB")

    return True


def main():
    print("="*60)
    print("FIXED EXECUTORCH EXPORT")
    print("Extracts weights from optimized TorchScript models")
    print("="*60 + "\n")

    # Find model path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "tiny_wake_word_optimized.pt"),
            "tiny_wake_word_optimized.pt",
        ]
        model_path = None
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break

    if not model_path or not os.path.exists(model_path):
        print("ERROR: Could not find TorchScript model!")
        print("Usage: python export_executorch_fixed.py [model_path]")
        return 1

    # Extract weights
    state_dict, original_model = extract_weights_from_torchscript(model_path)
    if state_dict is None:
        return 1

    # Save extracted weights for future use
    weights_path = os.path.join(os.path.dirname(model_path), "extracted_weights.pth")
    torch.save(state_dict, weights_path)
    print(f"\nSaved extracted weights to: {weights_path}")

    # Verify extraction
    print("\nVerifying extracted weights...")
    if not verify_extraction(state_dict, original_model):
        return 1

    # Export to ExecuTorch
    output_path = os.path.join(os.path.dirname(model_path), "wake_word_xnnpack.pte")
    if not export_to_executorch(state_dict, output_path):
        return 1

    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print("""
The model outputs LOG SOFTMAX probabilities.

Your Android WakeWordDetector.kt should use:
    val probabilities = logProbs.map { kotlin.math.exp(it.toDouble()).toFloat() }

Copy the model to Android assets:
    cp wake_word_xnnpack.pte ../app/apps/src/androidMain/assets/
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())

# testing/test_pruning.py
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pacs_net import PACSNet
from models.pruning import rebuild_pruned_model, create_masked_model
import logging


logger = logging.getLogger(__name__)


def test_pruning_correctness():
    """
    Test that pruning produces identical outputs to masking.
    This verifies that the weight copying logic is correct.
    """
    print("\n" + "="*50)
    print("Testing Pruning Correctness")
    print("="*50)
    
    # Configuration
    num_classes = 7
    hidden_sizes = [512, 256]
    batch_size = 4
    
    # Create a dummy config object
    class Config:
        NUM_CLASSES = num_classes
        HIDDEN_SIZES = hidden_sizes
    
    config = Config()
    
    # Create original model
    print("\n1. Creating original model...")
    original_model = PACSNet(num_classes=num_classes, hidden_sizes=hidden_sizes)
    original_model.eval()
    
    # Get total neurons
    total_neurons = original_model.get_total_mlp_neurons()
    print(f"   Total MLP neurons: {total_neurons}")
    
    # Select random subset of neurons to keep (50%)
    import numpy as np
    np.random.seed(42)
    n_keep = total_neurons // 2
    selected_indices = sorted(np.random.choice(total_neurons, n_keep, replace=False).tolist())
    print(f"   Selected {n_keep} neurons to keep")
    
    # Create pruned model
    print("\n2. Creating pruned model...")
    pruned_model = rebuild_pruned_model(original_model, selected_indices, config)
    pruned_model.eval()
    print(f"   Pruned model created with {pruned_model.get_total_mlp_neurons()} neurons")
    
    # Create masked model
    print("\n3. Creating masked model...")
    masked_model = create_masked_model(original_model, selected_indices)
    masked_model.eval()
    
    # Test with random input
    print("\n4. Testing forward pass...")
    test_input = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        pruned_output = pruned_model(test_input)
        masked_output = masked_model(test_input)
    
    print(f"   Pruned output shape: {pruned_output.shape}")
    print(f"   Masked output shape: {masked_output.shape}")
    
    # Check if outputs are close
    max_diff = torch.max(torch.abs(pruned_output - masked_output)).item()
    mean_diff = torch.mean(torch.abs(pruned_output - masked_output)).item()
    
    print(f"\n5. Comparing outputs:")
    print(f"   Max difference: {max_diff:.8f}")
    print(f"   Mean difference: {mean_diff:.8f}")
    
    # Assert outputs are very close
    tolerance = 1e-5
    assert torch.allclose(pruned_output, masked_output, atol=tolerance), \
        f"Outputs differ by more than {tolerance}! Max diff: {max_diff}"
    
    print(f"\n✓ Test PASSED! Pruning is correct (within tolerance {tolerance})")
    print("="*50 + "\n")
    
    return True


def test_layer_wise_pruning():
    """
    Test that layer-wise pruning preserves the correct neurons.
    """
    print("\n" + "="*50)
    print("Testing Layer-wise Pruning")
    print("="*50)
    
    # Configuration
    num_classes = 7
    hidden_sizes = [100, 50]  # Smaller for easier testing
    
    class Config:
        NUM_CLASSES = num_classes
        HIDDEN_SIZES = hidden_sizes
    
    config = Config()
    
    # Create model
    print("\n1. Creating model with layers:", hidden_sizes)
    model = PACSNet(num_classes=num_classes, hidden_sizes=hidden_sizes)
    
    # Select specific neurons from each layer
    # Layer 0: keep neurons [0, 10, 20, 30, 40]
    # Layer 1: keep neurons [100, 105, 110]  (offset by 100)
    selected_indices = [0, 10, 20, 30, 40, 100, 105, 110]
    print(f"2. Selected neurons: {selected_indices}")
    
    # Create pruned model
    pruned_model = rebuild_pruned_model(model, selected_indices, config)
    
    # Check dimensions
    pruned_sizes = pruned_model.get_num_neurons_per_layer()
    print(f"3. Pruned layer sizes: {pruned_sizes}")
    
    assert pruned_sizes[0] == 5, f"Layer 0 should have 5 neurons, got {pruned_sizes[0]}"
    assert pruned_sizes[1] == 3, f"Layer 1 should have 3 neurons, got {pruned_sizes[1]}"
    
    print("\n✓ Layer-wise pruning test PASSED!")
    print("="*50 + "\n")
    
    return True


if __name__ == "__main__":
    # Run tests
    try:
        test_pruning_correctness()
        test_layer_wise_pruning()
        print("\nAll tests completed successfully! ✨")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise

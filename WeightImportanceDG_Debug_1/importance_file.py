"""
Importance Calculation using Optimal Brain Damage (OBD)
Computes weight importance based on saliency using diagonal Hessian
Based on Le Cun et al. 1990 - "Optimal Brain Damage"
"""

import torch
import torch.nn as nn
import numpy as np
from config_file import *


class HessianComputer:
    """
    Computes diagonal Hessian (second derivatives) using OBD backpropagation algorithm
    Based on Le Cun et al. 1990
    """

    def __init__(self, use_lm_approximation=True):
        """
        Initialize Hessian computer

        Args:
            use_lm_approximation (bool): If True, use Levenberg-Marquardt approximation
                                        (ignore f'' terms, guarantees positive values)
        """
        self.use_lm_approximation = use_lm_approximation

    def compute_diagonal_hessian(self, model, dataloader, device=DEVICE):
        """
        Compute diagonal Hessian h_kk for all weights in hidden layers

        Algorithm:
        1. Forward pass: compute all activations
        2. Backward pass for second derivatives: compute ∂²E/∂a² recursively
        3. Compute ∂²E/∂w² = ∂²E/∂a² * z²
        4. Average over entire dataset

        Args:
            model: DomainGeneralizationModel with trained weights
            dataloader: DataLoader for computing Hessian (use full domain data)
            device: Device to compute on

        Returns:
            List of 4 tensors [10, 10] containing h_kk values for each hidden layer
        """
        model.eval()
        model.to(device)

        # Initialize accumulators for Hessian diagonal
        # We only track the 4 hidden layers (10x10 each)
        num_samples = 0
        hessian_accum = [torch.zeros(10, 10).to(device) for _ in range(4)]

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.size(0)

                # Compute Hessian for this batch
                batch_hessian = self._compute_batch_hessian(model, inputs, targets)

                # Accumulate
                for layer_idx in range(4):
                    hessian_accum[layer_idx] += batch_hessian[layer_idx] * batch_size

                num_samples += batch_size

        # Average over dataset
        hessian_diagonal = [h / num_samples for h in hessian_accum]

        print(f"Computed diagonal Hessian over {num_samples} samples")
        print(f"Using Levenberg-Marquardt approximation: {self.use_lm_approximation}")

        return hessian_diagonal

    def _compute_batch_hessian(self, model, inputs, targets):
        """
        Compute diagonal Hessian for a single batch

        Args:
            model: The model
            inputs: Input images [batch_size, 3, 224, 224]
            targets: Target labels [batch_size]

        Returns:
            List of 4 tensors [10, 10] with batch Hessian diagonal
        """
        batch_size = inputs.size(0)

        # Storage for activations and pre-activations during forward pass
        activations = []  # z_i = f(a_i)
        pre_activations = []  # a_i = Σ w_ij * z_j

        # Forward pass through ViT (frozen)
        with torch.no_grad():
            features = model.feature_extractor(inputs)  # [batch_size, 768]

        # Forward pass through classifier, storing activations
        x = features

        # Input layer (768 -> 10)
        x = model.classifier.input_layer(x)
        x = model.classifier.relu(x)
        activations.append(x.clone())

        # Hidden layers (10 -> 10) - we track these
        for layer in model.classifier.hidden_layers:
            a = layer(x)  # pre-activation
            z = model.classifier.relu(a)  # activation
            pre_activations.append(a.clone())
            activations.append(z.clone())
            x = z

        # Output layer (10 -> 7)
        outputs = model.classifier.output_layer(x)  # logits

        # Convert targets to one-hot for MSE loss computation
        targets_one_hot = torch.zeros(batch_size, NUM_CLASSES).to(inputs.device)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Compute first derivatives (gradients) for the backward pass
        # ∂E/∂z_i for each activation
        with torch.enable_grad():
            outputs_var = outputs.clone().requires_grad_(True)
            loss = torch.mean((outputs_var - targets_one_hot) ** 2)
            loss.backward()
            grad_output = outputs_var.grad  # ∂E/∂output

        # Initialize second derivative at output layer
        # For MSE loss: ∂²E/∂a² = 2*f'(a)² at output
        # But output layer has no activation, so we work backwards from there

        # Backward pass for second derivatives
        second_derivs = self._backward_second_derivatives(
            model, activations, pre_activations, grad_output, targets_one_hot, outputs
        )

        # Compute ∂²E/∂w² for each hidden layer weight
        batch_hessian = []
        for layer_idx in range(4):
            # For weight w_ij connecting layer (layer_idx) to layer (layer_idx+1)
            # ∂²E/∂w_ij² = ∂²E/∂a_i² * z_j²

            second_deriv_a = second_derivs[layer_idx]  # [batch_size, 10]
            z_prev = activations[layer_idx]  # [batch_size, 10] - input to this layer

            # Compute outer product and average over batch
            # h_ij = mean over batch of (∂²E/∂a_i² * z_j²)
            hessian_layer = torch.zeros(10, 10).to(inputs.device)

            for b in range(batch_size):
                # Outer product: second_deriv[i] * z[j]²
                hess_b = second_deriv_a[b].unsqueeze(1) * (z_prev[b].unsqueeze(0) ** 2)
                hessian_layer += hess_b

            hessian_layer /= batch_size
            batch_hessian.append(hessian_layer)

        return batch_hessian

    def _backward_second_derivatives(
        self, model, activations, pre_activations, grad_output, targets_one_hot, outputs
    ):
        """
        Backward pass to compute ∂²E/∂a² for each layer

        Algorithm from paper:
        - Output layer: ∂²E/∂a² = 2*f'(a)² - 2*(d-z)*f''(a)
        - Hidden layers: ∂²E/∂a² = f'(a)² * Σ(w_l² * ∂²E/∂a_l²) - f''(a)*∂E/∂z

        With Levenberg-Marquardt approximation, we ignore f''(a) terms.
        """
        # We work backwards from output to input
        # Layers: input(768->10), hidden[0](10->10), hidden[1](10->10),
        #         hidden[2](10->10), hidden[3](10->10), output(10->7)

        second_derivs = []

        # Start from output layer and work backwards
        # Output layer boundary condition
        # For linear output (no activation): ∂²E/∂a² = 2 (constant for MSE)
        # The paper assumes sigmoid at output, but we have linear output

        # For our case with linear output and MSE:
        # E = Σ(target - output)²
        # ∂²E/∂output² = 2
        batch_size = outputs.size(0)
        second_deriv_output = (
            torch.ones(batch_size, NUM_CLASSES).to(outputs.device) * 2.0
        )

        # Now backpropagate through layers
        # Output layer (10 -> 7): we need ∂²E/∂a² for the input to output layer
        # which is the output of hidden layer 3

        current_second_deriv = second_deriv_output

        # Backprop through output layer to get second deriv at hidden layer 3 output
        # ∂²E/∂z² = Σ(w² * ∂²E/∂a_next²)
        weights_output = model.classifier.output_layer.weight  # [7, 10]
        second_deriv_h3_output = torch.zeros(batch_size, 10).to(outputs.device)

        for i in range(10):  # For each unit in hidden layer 3 output
            # Sum over all output units
            second_deriv_h3_output[:, i] = torch.sum(
                (weights_output[:, i] ** 2).unsqueeze(0) * current_second_deriv, dim=1
            )

        # Now go through hidden layers in reverse (3, 2, 1, 0)
        for layer_idx in range(3, -1, -1):
            # Get activation function derivatives
            a = pre_activations[layer_idx]  # pre-activation at this layer
            f_prime = self._relu_derivative(a)  # f'(a)
            f_double_prime = (
                self._relu_second_derivative(a) if not self.use_lm_approximation else 0
            )

            # ∂²E/∂a² = f'(a)² * (second_deriv from next layer)
            second_deriv_a = (f_prime**2) * second_deriv_h3_output

            # Store for this layer
            second_derivs.insert(0, second_deriv_a)  # Insert at beginning

            # Backpropagate to previous layer (if not at layer 0)
            if layer_idx > 0:
                weights = model.classifier.hidden_layers[layer_idx].weight  # [10, 10]
                second_deriv_prev = torch.zeros(batch_size, 10).to(outputs.device)

                for i in range(10):
                    second_deriv_prev[:, i] = torch.sum(
                        (weights[:, i] ** 2).unsqueeze(0) * second_deriv_a, dim=1
                    )

                second_deriv_h3_output = second_deriv_prev

        return second_derivs  # List of 4 tensors [batch_size, 10]

    def _relu_derivative(self, x):
        """First derivative of ReLU: 1 if x > 0, else 0"""
        return (x > 0).float()

    def _relu_second_derivative(self, x):
        """Second derivative of ReLU: 0 everywhere (except undefined at 0)"""
        return torch.zeros_like(x)


def compute_obd_saliency(weights, hessian_diagonal):
    """
    Compute OBD saliency for each weight using the formula:
    s_k = (1/2) * h_kk * u_k²

    Args:
        weights (list): List of 4 weight tensors [10, 10] - final trained weights
        hessian_diagonal (list): List of 4 h_kk tensors [10, 10]

    Returns:
        list: List of 4 saliency tensors [10, 10]
    """
    saliencies = []
    for w, h in zip(weights, hessian_diagonal):
        # Saliency = 0.5 * h_kk * weight²
        s = 0.5 * h * (w**2)
        saliencies.append(s)

    return saliencies


def aggregate_obd_saliencies(saliencies_dict):
    """
    Aggregate OBD saliencies from multiple domains by element-wise sum

    Args:
        saliencies_dict (dict): Dictionary mapping domain_name -> list of 4 saliency tensors [10, 10]

    Returns:
        tuple: (summed_saliencies, normalized_saliencies, binary_masks)
    """
    domains = list(saliencies_dict.keys())
    num_layers = 4

    print(f"\nAggregating OBD saliencies from {len(domains)} domains: {domains}")

    # Element-wise sum across all domains for each layer
    summed_saliencies = []
    for layer_idx in range(num_layers):
        # Stack all domains' saliencies for this layer
        layer_saliency_stack = torch.stack(
            [saliencies_dict[domain][layer_idx] for domain in domains]
        )  # Shape: [num_domains, 10, 10]

        # Sum across domains (element-wise)
        summed = torch.sum(layer_saliency_stack, dim=0)  # Shape: [10, 10]
        summed_saliencies.append(summed)

    print(f"Computed element-wise sum of OBD saliencies across {len(domains)} domains")

    # Normalize globally to [0, 1]
    all_summed = torch.stack(summed_saliencies)
    global_min = all_summed.min()
    global_max = all_summed.max()

    if global_max - global_min < 1e-10:
        normalized_saliencies = [torch.ones_like(s) * 0.5 for s in summed_saliencies]
    else:
        normalized_saliencies = [
            (s - global_min) / (global_max - global_min) for s in summed_saliencies
        ]

    print(f"Normalized summed saliencies to [0, 1] range")

    # Create binary masks: top IMPORTANCE_THRESHOLD_PERCENTILE% = important
    all_normalized = torch.stack(normalized_saliencies).flatten()
    threshold_value = torch.quantile(
        all_normalized, q=(100 - IMPORTANCE_THRESHOLD_PERCENTILE) / 100.0
    )

    binary_masks = []
    for norm_sal in normalized_saliencies:
        mask = (norm_sal >= threshold_value).float()
        binary_masks.append(mask)

    print(
        f"Created binary masks (top {IMPORTANCE_THRESHOLD_PERCENTILE}% marked as important)"
    )

    return summed_saliencies, normalized_saliencies, binary_masks


class ImportanceCalculator:
    """
    Legacy class for backward compatibility
    Now wraps OBD-based importance computation
    """

    def __init__(self):
        """Initialize importance calculator"""
        self.hessian_computer = HessianComputer(use_lm_approximation=True)

    def compute_importance(self, initial_weights, final_weights):
        """
        Legacy method - now deprecated
        Use compute_obd_saliency with Hessian instead
        """
        print("WARNING: Using legacy importance computation (absolute weight change)")
        print("Consider using OBD saliency computation instead")

        importance_matrices = []
        for init_w, final_w in zip(initial_weights, final_weights):
            importance = torch.abs(final_w - init_w)
            importance_matrices.append(importance)

        return importance_matrices


def compute_mask_statistics(binary_masks):
    """
    Compute statistics about binary masks

    Args:
        binary_masks (list): List of 4 binary mask tensors [10, 10]

    Returns:
        dict: Statistics including percentage of important weights per layer
    """
    stats = {}

    for i, mask in enumerate(binary_masks):
        total_weights = mask.numel()
        important_weights = mask.sum().item()
        percentage_important = (important_weights / total_weights) * 100

        stats[f"layer_{i+1}"] = {
            "total_weights": total_weights,
            "important_weights": int(important_weights),
            "percentage_important": percentage_important,
        }

    # Overall statistics
    total_all = sum([mask.numel() for mask in binary_masks])
    important_all = sum([mask.sum().item() for mask in binary_masks])
    stats["overall"] = {
        "total_weights": total_all,
        "important_weights": int(important_all),
        "percentage_important": (important_all / total_all) * 100,
    }

    return stats

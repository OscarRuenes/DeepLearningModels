# Image Classification Models in PyTorch

A collection of fully connected neural network architectures for multi-class image classification, implemented with **PyTorch**. This implementation explores how model complexity and residual connections change the structure of a classifier, from a single linear layer to deeper multilayer perceptrons (MLPs).

## Model Implementations

All models accept image batches shaped `(batch_size, 3, height, width)` and return raw class logits shaped `(batch_size, num_classes)`. The default input dimensions are `64 × 64` RGB images, and the default number of classes is `6`.

| Model | Architecture | Key idea |
|---|---|---|
| `LinearClassifier` | Flatten → Linear | A linear baseline with no hidden layers |
| `MLPClassifier` | Flatten → Linear → ReLU → Linear → ReLU → Linear | A fully connected network with two 128-unit hidden layers |
| `MLPClassifierDeep` | Flatten → Linear → ReLU × 4 hidden layers → Linear | A deeper MLP with four 128-unit hidden layers |
| `MLPClassifierDeepResidual` | Flatten → Linear → four residual updates → Linear | Adds activated hidden-layer outputs to the current representation |
| `ClassificationLoss` | Cross-entropy | Multi-class classification loss for logits and integer class labels |

### Architecture details

- **Linear baseline:** Flattens each RGB image into a vector of `3 × height × width` features and maps it directly to the output classes.
- **MLP:** Learns nonlinear feature combinations using fully connected layers and ReLU activations.
- **Deep MLP:** Extends the MLP to four hidden layers, allowing the model to learn more complex nonlinear mappings.
- **Residual MLP:** Projects the flattened image into a 128-dimensional representation, then applies four updates of the form `x = x + ReLU(Linear(x))` before producing class logits. The skip additions preserve the existing representation while adding learned transformations.

## Design Highlights

- **Consistent interface:** Each classifier accepts an image tensor and returns logits, making the models easy to swap during experiments.
- **Factory-based selection:** `model_factory` maps model names (`linear`, `mlp`, `mlp_deep`, and `mlp_deep_residual`) to their corresponding classes.
- **Checkpoint support:** `save_model()` saves a model's `state_dict`, and `load_model()` reconstructs a model and can optionally load its saved weights.
- **Model-size guardrail:** `calculate_model_size_mb()` estimates parameter storage using four bytes per parameter; `load_model()` rejects models larger than 10 MB.
- **Configurable dimensions:** Constructors allow the image height, width, and class count to be specified, while retaining default arguments for compatibility.

## Example Usage

```python
import torch
from models import model_factory, ClassificationLoss

# Create a model
model = model_factory["mlp"]()

# Example batch: 8 RGB images, each 64 × 64
images = torch.randn(8, 3, 64, 64)
logits = model(images)

print(logits.shape)  # torch.Size([8, 6])

# Calculate classification loss using integer class labels
targets = torch.randint(0, 6, (8,))
loss_fn = ClassificationLoss()
loss = loss_fn(logits, targets)

print(loss.item())
```

## Saving and Loading Weights

The module provides helper functions to save model parameters and load them later.

```python
from models import model_factory, save_model, load_model

model = model_factory["mlp_deep"]()

# Save the model's state dictionary to mlp_deep.th
save_model(model)

# Recreate the model and load the saved weights
loaded_model = load_model("mlp_deep", with_weights=True)
loaded_model.eval()
```

`save_model()` writes the checkpoint beside `models.py`. Make sure the model architecture and constructor arguments match when loading a checkpoint. The default `load_model()` behavior creates a model without loading weights.

## Requirements

- Python 3
- PyTorch

## Scope

This file contains the **model architectures and model-loading utilities**. It does not include a dataset pipeline, training loop, evaluation results, or benchmark metrics. 

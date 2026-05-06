# TTNN _experimental Package

This directory contains experimental modules that will be dynamically added to `ttnn.experimental`.

## How it works

1. The `experimental_loader` creates `ttnn.experimental` as a module containing deprecated operations
2. After that, the code in `ttnn/__init__.py` discovers all submodules in this `_experimental` directory
3. Each submodule is appended to `ttnn.experimental`, making it accessible via `ttnn.experimental.submodule_name`

## Adding new experimental modules

To add a new experimental module:

1. Create a new directory under `ttnn/ttnn/_experimental/`
2. Add your module files and an `__init__.py`
3. The module will automatically be available as `ttnn.experimental.your_module`

## Example

The `moe` module in this directory can be imported as:
```python
from ttnn.experimental.moe_compute_utils import cluster_distance
# or
import ttnn.experimental.moe_compute_utils
```

This approach avoids conflicts with the `experimental_loader` while allowing us to extend the experimental namespace.

## MoE compute helpers

``moe_compute_utils`` provides reference implementations for packing expert
weights and optional biases into the DRAM-sharded tensors expected by
``ttnn.experimental.moe_compute``. See the module docstring for full details:

- Two packed weight tensor arguments (W0+W1 bundle, W2) containing three logical matrices
- Bias tensor formats: PyTorch ``(L, E, N)`` or ``(L, E, K)`` expanded to kernel tile layout
- Constants that must stay in sync with ``moe_ring_common.h``
- DRAM sharding requirements for the packed tensors

Example usage:
```python
from ttnn.experimental.moe_compute_utils import (
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w2_tensor_for_moe_compute,
    prepare_w0_w1_tensor_with_bias,
    prepare_w2_tensor_with_bias,
)
```

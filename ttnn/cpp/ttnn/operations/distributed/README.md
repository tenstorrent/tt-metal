# Distributed Operations for TTNN

This directory contains distributed operations that support the unified programming model for multi-device execution.

## Overview

Distributed operations are designed to work seamlessly across single and multiple devices, abstracting away device boundaries and enabling scalable tensor operations across mesh devices.

## Operations

### `ttnn.distributed.matmul`

Distributed matrix multiplication operation that supports multi-device tensors with arbitrary sharding/replication layouts.

**Current Implementation:**
- Prototype implementation that delegates to the standard `ttnn.matmul` operation
- Supports replicated tensors across mesh devices
- Provides the API structure for future enhancements

**Future Enhancements (Per UNIFIED_PM.README):**
- Automatic handling of different input tensor topologies (sharded/replicated)
- Optional output tensor topology specification
- Cross-device communication as needed
- Support for arbitrary input/output global layouts
- Virtualization of device boundaries

**Usage:**
```python
import ttnn
import torch

# Open mesh device (e.g., 2x4 mesh)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

# Create replicated tensors
a = torch.randn([4, 1, 64, 64], dtype=torch.bfloat16)
b = torch.randn([1, 1, 64, 128], dtype=torch.bfloat16)

tt_a = ttnn.from_torch(a, device=mesh_device, layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
tt_b = ttnn.from_torch(b, device=mesh_device, layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))

# Call distributed matmul
output = ttnn.distributed.matmul(tt_a, tt_b)
```

## File Structure

```
distributed/
├── README.md                          # This file
├── CMakeLists.txt                     # Build configuration
├── distributed_pybind.hpp             # Python bindings header
├── distributed_pybind.cpp             # Python bindings implementation
└── matmul/
    ├── CMakeLists.txt                 # Matmul build configuration
    ├── distributed_matmul.hpp         # Operation header
    ├── distributed_matmul.cpp         # Operation implementation
    ├── distributed_matmul_pybind.hpp  # Python bindings header
    └── distributed_matmul_pybind.cpp  # Python bindings implementation
```

## Testing

Tests are located in `tests/ttnn/unit_tests/operations/test_distributed_matmul.py`

Run tests with:
```bash
pytest tests/ttnn/unit_tests/operations/test_distributed_matmul.py
```

## Design Goals

Following the unified programming model outlined in UNIFIED_PM.README:

1. **Ease of Use**: Operations should "just work" for users without manual collective insertion
2. **Scalable by Default**: Written once, runs correctly on 2, 20, or 200 devices
3. **Optimizable When Needed**: Clean path to incremental optimization without rewriting

## Implementation Status

- [x] Basic scaffolding and API structure
- [x] Python bindings through ttnn
- [x] Integration with build system
- [x] Basic tests with replicated tensors on mesh device
- [ ] TensorTopology awareness for input/output
- [ ] Cross-device communication support
- [ ] Virtual coordinate system integration
- [ ] Support for various sharding strategies
- [ ] Performance optimizations

## Related Documentation

- Main design document: `/localdev/bliu/tt-metal/UNIFIED_PM.README`
- Unified DM interface: https://docs.google.com/document/d/1DltQArjcl6cCBl_nghk5ihm7u1av6YJPIJal9jFEM6o/edit?tab=t.0

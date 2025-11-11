# DeepSeek B1 Experimental Operations

This directory contains experimental operations optimized for DeepSeek models, specifically a full implementation of the MCast 1D matmul variant.

## Operations

### `matmul_1d` - MCast 1D Matrix Multiplication

A complete, self-contained implementation of the 1D multicast matrix multiplication operation, copied from the main ttnn matmul operation.

**Namespace**: `ttnn::experimental::deepseek_b1`

**Python API**:
```python
ttnn.experimental.deepseek_b1.matmul_1d(
    input_a, input_b,
    core_grid,        # CoreGrid(x=8, y=7)
    in0_block_w,      # Tiles per core in K dimension
    out_subblock_h,   # Output subblock height
    out_subblock_w,   # Output subblock width
    per_core_M,       # M tiles per core
    per_core_N,       # N tiles per core
    fuse_batch=True,
    mcast_in0=True,
    memory_config=None,
    dtype=None,
    compute_kernel_config=None
)
```

## Structure

Follows the same pattern as `experimental/ccl` with each operation in its own subfolder:

```
deepseek_b1/
├── deepseek_b1_pybind.hpp                      # Module binding header
├── deepseek_b1_pybind.cpp                      # Module binding implementation
├── CMakeLists.txt                              # Build configuration
├── README.md                                   # This file
└── matmul_1d/                                  # MCast 1D matmul operation
    ├── matmul_1d.hpp                           # Operation header
    ├── matmul_1d.cpp                           # Operation implementation
    ├── matmul_1d_pybind.hpp                    # Python binding header
    ├── matmul_1d_pybind.cpp                    # Python binding implementation
    └── device/
        ├── matmul_1d_device_operation.hpp      # Device operation interface
        ├── matmul_1d_device_operation.cpp      # Device operation implementation
        ├── matmul_1d_program_factory.hpp       # Program factory header
        ├── matmul_1d_program_factory.cpp       # Full program factory (1257 lines)
        └── kernels/                            # Kernel suite (6 files)
            ├── bmm_large_block_zm_fused_bias_activation.cpp
            ├── pad_tile.hpp
            ├── reader_bmm_tile_layout_in0_receiver.cpp
            ├── reader_bmm_tile_layout_in0_sender_padding.cpp
            ├── reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp
            └── reader_bmm_tile_layout_in1_sender_writer_padding.cpp
```

This structure mirrors `experimental/ccl` where each operation (like `all_gather_async/`, `matmul_reduce_scatter_async/`) lives in its own subfolder under the parent module directory.

## Implementation Details

### Specialized MCast 1D Program Factory

This operation includes a specialized matmul mcast 1d implementation:

1. **Program Factory**: Custom implementation based on `matmul_op_multi_core_reuse_mcast_1d_program_factory.cpp` (1257 lines)
   - MCast in0 variant optimized for DeepSeek patterns
   - Runtime argument override callbacks
   - Sharded and interleaved memory support
   - Global circular buffer support

2. **Kernels**: Focused kernel suite (6 files)
   - 1 compute kernel: `bmm_large_block_zm_fused_bias_activation.cpp`
   - 4 dataflow kernels for in0/in1 sender/receiver operations
   - 1 header file: `pad_tile.hpp` for tile padding utilities

3. **Device Operation**: Custom device operation wrapper
   - Simplified config structure (`Matmul1DProgramConfig`)
   - Validation and output spec computation
   - Direct program factory invocation

### Benefits of Full Copy

- **Independence**: Can modify and optimize without affecting main matmul
- **Simplification**: Cleaner API tailored for DeepSeek use cases
- **Experimentation**: Safe sandbox for testing optimizations
- **Maintenance**: Self-contained for easier reasoning

### Use Case

Optimized for DeepSeek model patterns:
- Width-sharded inputs across many cores
- 1D multicast communication patterns
- Specific tile sizes and block configurations
- Batch fusion for inference

## Example

See `/models/demos/deepseek_v3_b1/tests/test_matmul.py` for a complete working example.

## Build Integration

Integrated into the ttnn build system via:
- `/ttnn/CMakeLists.txt` - Added subdirectory
- `/ttnn/cpp/ttnn/operations/experimental/experimental_pybind.cpp` - Registered module
- Python module: `ttnn.experimental.deepseek_b1`

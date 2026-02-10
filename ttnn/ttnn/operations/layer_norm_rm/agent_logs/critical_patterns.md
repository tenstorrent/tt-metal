# Critical Implementation Patterns - Layer Norm RM

## Generic Op Call Pattern
```python
# OUTPUT MUST BE LAST in io_tensors list
io_tensors = [input_tensor]
if gamma is not None:
    io_tensors.append(gamma)
if beta is not None:
    io_tensors.append(beta)
io_tensors.append(output_tensor)  # LAST

return ttnn.generic_op(io_tensors, program_descriptor)
```

## Tensor Allocation Pattern
```python
# CRITICAL: Use positional args, not keyword args
output_tensor = ttnn.allocate_tensor_on_device(
    ttnn.Shape(output_shape),  # arg 0: shape
    input_tensor.dtype,         # arg 1: dtype
    input_tensor.layout,        # arg 2: layout
    target_device,              # arg 3: device
    target_memory_config        # arg 4: memory_config
)
```

## Compute Kernel Config Pattern
```python
# CRITICAL: Use ComputeConfigDescriptor, NOT ComputeConfig
config=ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=fp32_dest_acc_en,  # True for float32 or high precision
    math_approx_mode=False,
)
```

## Reduce Scaler Pattern
```python
# CRITICAL: CB 6 (reduce scaler) MUST ALWAYS be bfloat16 format
# This is a hardware requirement, regardless of input dtype

# Pack reduce scaler as (bf16 << 16 | bf16)
def _pack_bf16_pair(value: float) -> int:
    bf16 = _bf16_to_uint32(value)
    return (bf16 << 16) | bf16

reduce_scaler_packed = _pack_bf16_pair(1.0 / W)

# CB descriptor for reduce scaler
ttnn.CBDescriptor(
    total_size=bf16_page_size,
    core_ranges=core_range_set,
    format_descriptors=[
        ttnn.CBFormatDescriptor(
            buffer_index=6,
            data_format=ttnn.DataFormat.Float16_b,  # ALWAYS bfloat16
            page_size=bf16_page_size,
            num_pages=1
        )
    ]
)
```

## Kernel Include Pattern
```cpp
// Reader/Writer kernels - CRITICAL: Use full include path
#include "api/dataflow/dataflow_api.h"  // CORRECT

// NOT this:
// #include "dataflow_api.h"  // WRONG - will not compile

// Compute kernels
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
```

## TensorAccessor Pattern
```python
# Add to compile-time args for reader/writer kernels
reader_compile_args.extend(
    ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
)

# For optional tensors (gamma/beta), use dummy args if None
if gamma is not None:
    reader_compile_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
    )
else:
    reader_compile_args.extend([0] * 8)  # Dummy accessor args
```

## Runtime Args Pattern
```python
# CRITICAL: Must set runtime args for ALL cores in grid
# Even if single-core, must explicitly set args for core (0,0)

rt_args = ttnn.RuntimeArgs()
rt_args[0][0] = [  # Core (x=0, y=0)
    input_buffer_addr,
    gamma_addr,
    # ... more args
]

# If multi-core, would set args for all cores:
# for x in range(grid_width):
#     for y in range(grid_height):
#         if (x, y) in active_cores:
#             rt_args[x][y] = [actual, args]
#         else:
#             rt_args[x][y] = []  # Empty list for idle cores
```

## Data Format Mapping
```python
# Input/output data formats
if dtype == ttnn.bfloat16:
    data_format = ttnn.DataFormat.Float16_b
elif dtype == ttnn.float32:
    data_format = ttnn.DataFormat.Float32

# For intermediate precision (always use float32)
intermed_data_format = ttnn.DataFormat.Float32
```

## Test Pattern
```python
# CRITICAL: torch imports MUST be inside functions, not global
@pytest.mark.parametrize("shape", [...])
@pytest.mark.parametrize("dtype_str", ["bfloat16", "float32"])
def test_layer_norm_rm(shape, dtype_str, device):  # Use device fixture
    import torch  # Import inside function
    import ttnn   # Import inside function

    # Use PCC > 0.99 for correctness, not torch.allclose
    # pcc = compute_pcc(expected, actual)
    # assert pcc > 0.99
```

## Shape Indexing Pattern
```python
# CRITICAL: ttnn.Shape cannot be sliced directly
shape = tensor.shape  # This is a ttnn.Shape object

# WRONG:
# shape[1:]  # Will fail

# CORRECT:
output_shape = [shape[i] for i in range(len(shape))]
```

## Intermediate Precision Pattern
```python
# For bfloat16 input, use float32 intermediates for better accuracy
if dtype == ttnn.float32:
    intermed_dtype = ttnn.float32
    fp32_dest_acc_en = True
else:  # bfloat16
    intermed_dtype = ttnn.float32  # Still use float32 for intermediates
    fp32_dest_acc_en = True

# Configure intermediate CBs with float32 format
ttnn.CBDescriptor(
    total_size=Wt * intermed_page_size,
    core_ranges=core_range_set,
    format_descriptors=[
        ttnn.CBFormatDescriptor(
            buffer_index=25,
            data_format=ttnn.DataFormat.Float32,  # Intermediate precision
            page_size=intermed_page_size,
            num_pages=Wt
        )
    ]
)
```

## Buffer Address Pattern
```python
# Get buffer addresses for runtime args
input_buffer_addr = input_tensor.buffer().address()
output_buffer_addr = output_tensor.buffer().address()

# For optional tensors
gamma_addr = gamma.buffer().address() if gamma is not None else 0
beta_addr = beta.buffer().address() if beta is not None else 0
```

## Page Size Calculation Pattern
```python
# For TILE_LAYOUT: use get_tile_size()
page_size = tensor.tile.get_tile_size(tensor.dtype)

# For ROW_MAJOR_LAYOUT (sticks): use element_size()
stick_size = W * tensor.element_size()

# For different dtype than tensor
bf16_page_size = tensor.tile.get_tile_size(ttnn.bfloat16)
```

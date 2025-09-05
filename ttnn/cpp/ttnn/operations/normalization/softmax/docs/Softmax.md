# TTNN Softmax Operations

This document provides comprehensive documentation for the softmax operation. The softmax implementation provides both standard and optimized variants for different use cases, particularly targeting transformer attention mechanisms.

## Overview

The TTNN softmax implementation includes multiple variants designed for different performance requirements and memory constraints:

1. **Standard Softmax** - General-purpose softmax with arbitrary dimension support
2. **Scale-Mask-Softmax** - Fused operation combining scaling, masking, and softmax
3. **In-Place Operations** - Memory-efficient variants that modify input tensors directly
4. **Specialized Variants** - Optimized implementations for specific attention patterns

All operations compute the mathematical softmax function:

```
softmax(x_i) = exp(x_i) / Σ(exp(x_i))
```

With optional numerical stability improvements that subtract the maximum value before exponentiation.

## Quick Decision Guide

| Use Case | Recommended Operation | Reason |
|----------|----------------------|--------|
| Transformer Attention | `ttnn::scale_mask_softmax()` | Optimized for attention patterns with scale/mask fusion |
| Standard Softmax | `ttnn::softmax()` | General-purpose with automatic optimization |
| Memory-Constrained | `ttnn::softmax_in_place()` | Modifies input tensor directly |
| Attention with Masks | `ttnn::scale_mask_softmax_in_place()` | Memory-efficient fused operations |
| Causal Attention (Specialized) | `ttnn::scale_causal_mask_hw_dims_softmax_in_place()` | Optimized for specific transformer patterns |
| Numerical Stability | Any operation with `numeric_stable=True` | Prevents overflow for large input values |

## Available Operations

### 1. ttnn.softmax()

**Purpose**: Standard softmax operation along a specified dimension.

**Function Signature**:
```python
ttnn.softmax(
    input_tensor: ttnn.Tensor,
    dim: int = -1,
    *,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    compute_kernel_config: Optional[DeviceComputeKernelConfig] = None,
    numeric_stable: bool = False,
    queue_id: int = 0
) -> ttnn.Tensor
```

**Parameters**:
- `input_tensor`: Input tensor to apply softmax to
- `dim`: Dimension along which to compute softmax (default: -1, last dimension)
- `memory_config`: Memory configuration for output tensor (inherits from input if not specified)
- `compute_kernel_config`: Compute kernel configuration for the operation
- `numeric_stable`: Whether to use numerically stable computation (subtracts max before exp)
- `queue_id`: Queue ID for asynchronous execution

**Supported Data Types**: BFLOAT16, FLOAT32, BFLOAT8_B
**Supported Layouts**: TILE

### 2. ttnn.scale_mask_softmax()

**Purpose**: Fused scale-mask-softmax operation for attention mechanisms.

**Function Signature**:
```python
ttnn.scale_mask_softmax(
    input_tensor: ttnn.Tensor,
    scale: Optional[float] = None,
    mask: Optional[ttnn.Tensor] = None,
    *,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    is_causal_mask: bool = False,
    compute_kernel_config: Optional[DeviceComputeKernelConfig] = None,
    numeric_stable: bool = False,
    queue_id: int = 0
) -> ttnn.Tensor
```

**Parameters**:
- `input_tensor`: Input tensor to process
- `scale`: Scaling factor to multiply with input tensor (e.g., 1/√d_k for attention)
- `mask`: Attention mask tensor to add to scaled input (supports broadcasting)
- `is_causal_mask`: Whether the mask is a causal mask
- Other parameters same as standard softmax

**Operation Sequence**:
1. `scaled = input_tensor * scale` (if scale provided)
2. `masked = scaled + mask` (if mask provided, with broadcasting)
3. `output = softmax(masked)`

**Mask Requirements**:
- Input tensor: BFLOAT16, FLOAT32, BFLOAT8_B with TILE layout
- Mask tensor: BFLOAT16, BFLOAT8_B with TILE or ROW_MAJOR layout
- For ROW_MAJOR masks: intermediate dimensions must be 1, last dimension must equal TILE_WIDTH

### 3. ttnn.softmax_in_place()

**Purpose**: Memory-efficient in-place softmax operation.

**Function Signature**:
```python
ttnn.softmax_in_place(
    input_tensor: ttnn.Tensor,
    dim: int = -1,
    *,
    program_config: SoftmaxProgramConfig = SoftmaxDefaultProgramConfig(),
    compute_kernel_config: Optional[DeviceComputeKernelConfig] = None,
    numeric_stable: bool = False,
    queue_id: int = 0
) -> ttnn.Tensor
```

**Key Features**:
- Modifies input tensor directly (no additional memory allocation)
- Supports both default and sharded multi-core program configurations
- Automatic fallback to standard softmax for very wide tensors when L1 memory usage exceeds 90%

### 4. ttnn.scale_mask_softmax_in_place()

**Purpose**: Memory-efficient fused scale-mask-softmax operation.

**Function Signature**:
```python
ttnn.scale_mask_softmax_in_place(
    input_tensor: ttnn.Tensor,
    scale: Optional[float] = None,
    mask: Optional[ttnn.Tensor] = None,
    *,
    program_config: SoftmaxProgramConfig = SoftmaxDefaultProgramConfig(),
    is_causal_mask: bool = False,
    compute_kernel_config: Optional[DeviceComputeKernelConfig] = None,
    numeric_stable: bool = False,
    queue_id: int = 0
) -> ttnn.Tensor
```

**Operation Sequence (In-Place)**:
1. `input_tensor *= scale` (if scale provided)
2. `input_tensor += mask` (if mask provided, with broadcasting)
3. `input_tensor = softmax(input_tensor)`

### 5. ttnn.scale_causal_mask_hw_dims_softmax_in_place()

**Purpose**: Specialized in-place operation for causal masked softmax with height-width constraints.

**Function Signature**:
```python
ttnn.scale_causal_mask_hw_dims_softmax_in_place(
    input_tensor: ttnn.Tensor,
    scale: Optional[float] = None,
    mask: Optional[ttnn.Tensor] = None,
    *,
    program_config: SoftmaxProgramConfig = SoftmaxDefaultProgramConfig(),
    compute_kernel_config: Optional[DeviceComputeKernelConfig] = None,
    numeric_stable: bool = False,
    queue_id: int = 0
) -> ttnn.Tensor
```

**Special Requirements**:
- Input tensor must be sharded with ROW_MAJOR orientation
- Mask must be interleaved with shape [1, 1, H, W]
- Scale parameter must be provided
- Optimized for transformer attention patterns

## Program Configurations

### SoftmaxDefaultProgramConfig

Default configuration that automatically selects appropriate parameters based on input tensor characteristics.

```python
config = ttnn.SoftmaxDefaultProgramConfig()
```

### SoftmaxShardedMultiCoreProgramConfig

Multi-core sharded configuration for fine-grained control over computation parameters.

```python
config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size: CoreCoord,
    subblock_w: int,
    block_h: int,
    block_w: int
)
```

**Parameters**:
- `compute_with_storage_grid_size`: Grid size for compute cores with storage capability
- `subblock_w`: Width of sub-blocks for computation
- `block_h`: Height of blocks for processing
- `block_w`: Width of blocks for processing (modifiable after creation)

**Requirements**:
- Designed specifically for sharded tensors
- Block dimensions must be compatible with tensor's shard specification
- `block_w` must be divisible by `subblock_w`
- Proper block sizing significantly impacts performance

## Kernel Implementation Details

### Program Factory Selection

The implementation automatically selects the optimal program factory based on multiple criteria:

#### Selection Logic:

1. **Operation Type Check**:
   - **In-Place Operations** (`SoftmaxInPlace`, `ScaleMaskSoftmaxInPlace`, `ScaleCausalMaskHWSoftmaxInPlace`):
     - Uses attention-optimized factories
     - Selects sharded vs. interleaved based on `program_config` type

   - **Fused Scale-Mask Operations** (`ScaleMaskSoftmax`):
     - Uses attention-optimized factories for 4D tensors with last-dimension softmax

   - **Standard Softmax** (`Softmax`):
     - For 4D tensors with last-dimension softmax: Uses attention-optimized factories
     - For other cases: Uses general-purpose moreh factories

2. **Dimension-Based Selection** (for general-purpose operations):
   - **Last Dimension** (`rank - 1 == dim`):
     - Small tensors (< 512KB L1): `SoftmaxProgramFactoryGeneralWSmall`
     - Large tensors: `SoftmaxProgramFactoryGeneralWLarge`

   - **Second-to-Last Dimension** (`rank - 2 == dim`):
     - Small tensors (< 512KB L1): `SoftmaxProgramFactoryGeneralHSmall`
     - Large tensors: `SoftmaxProgramFactoryGeneralHLarge`

   - **Other Dimensions**:
     - Always uses: `SoftmaxProgramFactoryGeneralCLarge`

3. **Memory Layout Check**:
   - **Sharded Tensors**: Use sharded program factories when `SoftmaxShardedMultiCoreProgramConfig` is specified
   - **Interleaved Tensors**: Use standard program factories with `SoftmaxDefaultProgramConfig`

4. **Memory Size Analysis**:
   - **L1 Memory Calculation**: Considers circular buffer requirements for input, output, intermediate values, and masks
   - **512KB Threshold**: Determines small vs. large tensor handling strategy
   - **Data Format Impact**: FP32 destination accumulation affects memory requirements

### Available Program Factories

The implementation uses different program factories that automatically select appropriate kernels:

#### 1. General-Purpose Factories (Using Moreh Kernels)
These factories use the moreh kernel implementations for comprehensive dimension support:

- **SoftmaxProgramFactoryGeneralWSmall**:
  - For width-dimension softmax with small tensors (< 512KB L1)
  - Uses: `moreh_softmax_w.cpp`, `reader_moreh_softmax_w.cpp`, `writer_moreh_softmax_w.cpp`

- **SoftmaxProgramFactoryGeneralWLarge**:
  - For width-dimension softmax with large tensors
  - Uses: `moreh_softmax_w_large.cpp`, `reader_moreh_softmax_w_large.cpp`, `writer_moreh_softmax_w_large.cpp`

- **SoftmaxProgramFactoryGeneralHSmall**:
  - For height-dimension softmax with small tensors (< 512KB L1)
  - Uses: `moreh_softmax_h.cpp`, `reader_moreh_softmax_h.cpp`, `writer_moreh_softmax_h.cpp`

- **SoftmaxProgramFactoryGeneralHLarge**:
  - For height-dimension softmax with large tensors
  - Uses: `moreh_softmax_h_large.cpp`, `reader_moreh_softmax_h_large.cpp`, `writer_moreh_softmax_h_large.cpp`

- **SoftmaxProgramFactoryGeneralCLarge**:
  - For channel-dimension softmax
  - Uses: `moreh_softmax_c_large.cpp`, `reader_moreh_softmax_c_large.cpp`, `writer_moreh_softmax_c_large.cpp`

#### 2. Attention-Optimized Factories (Using Attention Kernels)
These factories use specialized attention kernels for transformer patterns:

- **SoftmaxProgramFactoryAttentionOptimized**:
  - Interleaved memory layout for attention patterns
  - Uses attention-specific kernels from `/device/kernels/attention/`

- **SoftmaxShardedProgramFactoryAttentionOptimized**:
  - Sharded memory layout for attention patterns
  - Uses sharded attention kernels for multi-core execution

### Kernel Components

The softmax implementation uses two main kernel groups for different optimization strategies:

#### 1. General-Purpose Kernels (Moreh)
Located in `/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/`:

These kernels provide comprehensive softmax support for arbitrary dimensions and tensor sizes:

**Compute Kernels**:
- `moreh_softmax_w.cpp`: Width-dimension softmax for small tensors
- `moreh_softmax_w_large.cpp`: Width-dimension softmax for large tensors
- `moreh_softmax_h.cpp`: Height-dimension softmax for small tensors
- `moreh_softmax_h_large.cpp`: Height-dimension softmax for large tensors
- `moreh_softmax_c_large.cpp`: Channel-dimension softmax for large tensors

**Reader Kernels**:
- `reader_moreh_softmax_w.cpp`: Reads data for width-dimension operations
- `reader_moreh_softmax_w_large.cpp`: Reads data for large width-dimension tensors
- `reader_moreh_softmax_h.cpp`: Reads data for height-dimension operations
- `reader_moreh_softmax_h_large.cpp`: Reads data for large height-dimension tensors
- `reader_moreh_softmax_c_large.cpp`: Reads data for channel-dimension operations

**Writer Kernels**:
- `writer_moreh_softmax_w.cpp`: Writes results for width-dimension operations
- `writer_moreh_softmax_w_large.cpp`: Writes results for large width-dimension tensors
- `writer_moreh_softmax_h.cpp`: Writes results for height-dimension operations
- `writer_moreh_softmax_h_large.cpp`: Writes results for large height-dimension tensors
- `writer_moreh_softmax_c_large.cpp`: Writes results for channel-dimension operations

**Key Features of Moreh Kernels**:
- Support for arbitrary reduction dimensions (width, height, channel)
- Automatic max finding for numerical stability across all tiles in the reduction dimension
- Efficient tile-based processing with proper masking for irregular tensor shapes
- Support for both softmax and log-softmax operations (controlled by `SOFTMAX` compile-time flag)
- Optimized memory access patterns for different tensor sizes
- Three-step algorithm: 1) Find max, 2) Compute exp(x-max), 3) Normalize by sum
- Uses `moreh_common.hpp` utilities for standardized tile operations

**Key Features of Attention Kernels**:
- Specialized for transformer attention patterns (typically last-dimension softmax on 4D tensors)
- Fused scale-mask-softmax operations in single kernels
- Support for both causal and non-causal attention masks
- Optimized for common attention tensor shapes and access patterns
- Hardware-accelerated approximate exponentiation (`EXP_APPROX`)
- Specialized handling for different mask types (interleaved, sharded, ROW_MAJOR)
- Memory-efficient streaming for large attention matrices

#### Kernel Selection Strategy

The system chooses between moreh and attention kernels based on:

1. **Operation Type**: In-place and fused operations prefer attention kernels
2. **Tensor Rank and Dimension**: 4D tensors with last-dimension softmax can use attention kernels
3. **Optimization Requirements**: Attention kernels provide better performance for transformer patterns
4. **Memory Layout**: Sharded tensors typically use attention kernels for better parallelization

#### 2. Attention-Optimized Kernels
Located in `/device/kernels/attention/`:

These kernels are specifically optimized for transformer attention patterns:

**Compute Kernels** (`compute/`):
- `softmax.cpp`: Main compute kernel for attention-optimized softmax
  - Supports fused scale-mask operations
  - Handles both causal and non-causal masks
  - Implements numerical stability optimizations
  - Uses approximate exponentiation (`EXP_APPROX`) for performance

- `softmax_sharded.cpp`: Specialized compute kernel for sharded tensors
  - Optimized for multi-core sharded execution
  - Efficient memory access patterns for sharded data

- `softmax_large_tensor.cpp`: Compute kernel for large tensors
  - Handles tensors that don't fit entirely in L1 memory
  - Implements streaming data access patterns

**Dataflow Kernels** (`dataflow/`):

*Reader Kernels*:
- `reader_unary_interleaved_sm.cpp`: Reads interleaved tensors
- `reader_unary_sharded_sm.cpp`: Reads sharded tensors
- `reader_unary_sharded_sm_causal_mask_hw_dims.cpp`: Specialized for causal masks with HW constraints
- `reader_unary_sharded_sm_rm_mask.cpp`: Handles ROW_MAJOR mask layout
- `reader_unary_interleaved_sm_large_tensor.cpp`: For large tensor streaming

*Writer Kernels*:
- `writer_unary_interleaved_start_id_blocked_sm.cpp`: Writes results back to memory

### Memory Management

#### Circular Buffers (CB)
The implementation uses multiple circular buffers for efficient data flow:

- **CB_in0**: Input tensor data
- **CB_out0**: Output tensor data
- **CB_exps**: Exponential values storage
- **CB_max**: Maximum values for numerical stability
- **CB_recipsumexps**: Reciprocal of sum of exponentials
- **CB_mask**: Attention mask data
- **CB_scale**: Scaling factors

#### Memory Optimization
- **L1 Memory Threshold**: 512KB threshold determines small vs. large tensor handling
- **Small Tensor Optimization**: Keeps all data in L1 for maximum performance
- **Large Tensor Streaming**: Processes data in chunks when L1 capacity is exceeded

## Performance Optimizations

### 1. Numerical Stability
When `numeric_stable=True`:
- Computes `max(x)` along the reduction dimension
- Subtracts max from input: `x_stable = x - max(x)`
- Computes `softmax(x_stable)` to prevent overflow

**Performance Impact**: Small computational overhead for improved numerical accuracy
**Recommendation**: Enable for large input values or when encountering NaN/inf outputs

### 2. Fused Operations
Scale-mask-softmax operations fuse multiple steps into single kernels:
- Reduces memory bandwidth requirements
- Eliminates intermediate tensor allocations
- Optimizes for attention mechanism patterns

**Typical Transformer Attention Pattern**:
```python
# Fused: scale + mask + softmax in one kernel
attention_weights = ttnn.scale_mask_softmax(
    attention_scores,
    scale=1.0 / math.sqrt(head_dim),  # 1/√d_k scaling
    mask=attention_mask,              # attention mask
    is_causal_mask=True,             # causal masking
    numeric_stable=True              # numerical stability
)
```

### 3. Multi-Core Execution
- Distributes computation across available cores
- Work-splitting strategies for different tensor sizes
- Sharded execution for maximum parallelism

### 4. Memory Layout Optimizations
- **Interleaved Layout**: Standard layout for general use
- **Sharded Layout**: Distributes tensor data across cores for parallel processing
- **ROW_MAJOR Masks**: Efficient storage for attention masks

### 5. Compute Kernel Configurations

#### High Precision Configuration:
```python
from ttnn import MathFidelity
config = ttnn.DeviceComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi4,
    fp32_dest_acc_en=True  # FP32 accumulation for FLOAT32 inputs
)
```

#### Fast Approximation Configuration:
```python
config = ttnn.DeviceComputeKernelConfig(
    math_fidelity=MathFidelity.LoFi,
    math_approx_mode=True  # Hardware-accelerated approximations
)
```

## Usage Examples

### Basic Softmax
```python
import ttnn

# Create input tensor
input_tensor = ttnn.rand((1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Apply softmax along last dimension
result = ttnn.softmax(input_tensor, dim=-1)
```

### Attention with Scale and Mask
```python
import math

# Attention scaling factor (1/√d_k)
scale = 1.0 / math.sqrt(64)

# Create attention mask
mask = ttnn.rand((1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Fused scale-mask-softmax
attention_weights = ttnn.scale_mask_softmax(
    input_tensor=attention_scores,
    scale=scale,
    mask=mask,
    is_causal_mask=False
)
```

### Complete Transformer Attention Implementation
```python
def transformer_attention(query, key, value, attention_mask=None, head_dim=64):
    """Complete transformer attention implementation using TTNN softmax"""

    # Step 1: Compute attention scores (Q @ K^T)
    key_transposed = ttnn.transpose(key, -2, -1)
    attention_scores = ttnn.matmul(query, key_transposed)

    # Step 2: Apply fused scale + mask + softmax
    scale_factor = 1.0 / math.sqrt(head_dim)
    attention_weights = ttnn.scale_mask_softmax(
        input_tensor=attention_scores,
        scale=scale_factor,
        mask=attention_mask,
        is_causal_mask=True,  # For autoregressive models
        numeric_stable=True   # For numerical stability
    )

    # Step 3: Apply attention to values (Attention @ V)
    output = ttnn.matmul(attention_weights, value)

    return output, attention_weights

# Usage example
batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64
query = ttnn.rand((batch_size, num_heads, seq_len, head_dim), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
key = ttnn.rand((batch_size, num_heads, seq_len, head_dim), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
value = ttnn.rand((batch_size, num_heads, seq_len, head_dim), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

output, weights = transformer_attention(query, key, value)
```

### Memory-Efficient In-Place Operations
```python
# In-place softmax (modifies input tensor)
ttnn.softmax_in_place(input_tensor)

# In-place scale-mask-softmax
ttnn.scale_mask_softmax_in_place(
    input_tensor=input_tensor,
    scale=1.0,
    mask=attention_mask
)
```

### Memory-Efficient Large Attention Matrices
```python
def efficient_large_attention(query, key, value, head_dim=64):
    """Memory-efficient attention for large sequence lengths"""

    # Use sharded configuration for large tensors
    compute_grid = device.compute_with_storage_grid_size()

    # Create sharded memory configuration
    shard_shape = [query.shape[-2], query.shape[-1]]  # [seq_len, head_dim]
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                          ttnn.CoreCoord(compute_grid.x-1, compute_grid.y-1))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR
    )
    sharded_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec
    )

    # Shard the tensors
    query_sharded = ttnn.to_memory_config(query, sharded_config)
    key_sharded = ttnn.to_memory_config(key, sharded_config)

    # Compute attention scores
    key_t = ttnn.transpose(key_sharded, -2, -1)
    scores = ttnn.matmul(query_sharded, key_t)

    # Create program configuration for sharded softmax
    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=compute_grid,
        subblock_w=8,
        block_h=32,
        block_w=24
    )

    # Apply in-place softmax to save memory
    attention_weights = ttnn.softmax_in_place(
        scores,
        program_config=program_config,
        numeric_stable=True
    )

    return attention_weights

# Example usage for large sequences
large_query = ttnn.rand((1, 8, 2048, 64), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
large_key = ttnn.rand((1, 8, 2048, 64), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
large_value = ttnn.rand((1, 8, 2048, 64), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

large_attention = efficient_large_attention(large_query, large_key, large_value)
```

### Classification with Temperature Scaling
```python
def classification_head(logits, temperature=1.0):
    """Classification head with temperature scaling"""

    # Apply temperature scaling
    if temperature != 1.0:
        scaled_logits = ttnn.multiply(logits, 1.0/temperature)
    else:
        scaled_logits = logits

    # Apply softmax to get probabilities
    probabilities = ttnn.softmax(
        scaled_logits,
        dim=-1,  # Along class dimension
        numeric_stable=True  # Important for numerical stability
    )

    return probabilities

# Example usage
num_classes = 1000
batch_size = 32
logits = ttnn.rand((batch_size, num_classes), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

# Standard classification
probs = classification_head(logits)

# With temperature scaling for calibration
calibrated_probs = classification_head(logits, temperature=1.5)
```
    scale=1.0,
    mask=attention_mask
)
```

### Sharded Multi-Core Configuration
```python
# Get device compute grid
compute_grid = device.compute_with_storage_grid_size()

# Create sharded program configuration
program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=compute_grid,
    subblock_w=8,
    block_h=32,
    block_w=24
)

# Apply softmax with sharded configuration
result = ttnn.softmax_in_place(
    sharded_tensor,
    program_config=program_config
)
```

### Specialized Causal Mask with HW Dimensions
```python
# Input must be sharded with ROW_MAJOR orientation
grid_coord = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
shard_spec = ttnn.ShardSpec(shard_grid, [384, 768], ttnn.ShardOrientation.ROW_MAJOR)
sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

input_sharded = ttnn.to_memory_config(input_tensor, sharded_mem_config)

# Mask must be interleaved with shape [1, 1, H, W]
causal_mask = ttnn.rand((1, 1, 384, 768), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

# Specialized causal mask operation
result = ttnn.scale_causal_mask_hw_dims_softmax_in_place(
    input_tensor=input_sharded,
    scale=1.0,
    mask=causal_mask,
    program_config=program_config
)
```

## Implementation Architecture

### Core Components

#### 1. SoftmaxDeviceOperation
Main device operation class that orchestrates the softmax computation:
- **Program Factory Selection**: Automatically chooses optimal implementation
- **Validation**: Ensures tensor compatibility and memory requirements
- **Memory Management**: Handles output tensor creation and memory configuration

#### 2. Program Factories
Specialized factories for different execution patterns:
- **General Factories**: Support arbitrary dimensions with automatic optimization
- **Attention Factories**: Optimized for transformer attention patterns
- **Sharded Factories**: Designed for multi-core sharded execution

#### 3. Compute Kernels
Low-level compute implementations:
- **Attention-Optimized**: Specialized for attention mechanism patterns
- **Large Tensor Support**: Streaming implementations for memory-constrained scenarios
- **Sharded Execution**: Multi-core parallel processing

#### 4. Dataflow Kernels
Memory access and data movement:
- **Reader Kernels**: Efficient data loading with support for different layouts
- **Writer Kernels**: Optimized result storage
- **Mask Handling**: Specialized readers for different mask formats

### Automatic Optimization Selection

The implementation automatically selects the optimal execution path based on:

1. **Tensor Size Analysis**:
   - Small tensors (< 512KB L1): Use L1-optimized kernels
   - Large tensors: Use streaming implementations

2. **Dimension Analysis**:
   - Width dimension (dim=-1): Use width-optimized kernels
   - Height dimension (dim=-2): Use height-optimized kernels
   - Other dimensions: Use general-purpose kernels

3. **Memory Layout**:
   - Interleaved tensors: Use standard dataflow kernels
   - Sharded tensors: Use sharded-optimized kernels

4. **Operation Type**:
   - Standard softmax: Select based on dimension and size
   - Fused operations: Use attention-optimized implementations
   - In-place operations: Use memory-efficient variants

## Performance Considerations

### Memory Efficiency
- **In-Place Operations**: Eliminate intermediate tensor allocations
- **Sharded Execution**: Maximize memory bandwidth utilization
- **L1 Optimization**: Keep frequently accessed data in fastest memory

### Computational Efficiency
- **Fused Operations**: Reduce kernel launch overhead
- **Multi-Core Scaling**: Distribute work across available cores
- **Approximate Math**: Use hardware-accelerated approximate functions where appropriate

### Numerical Stability
- **Stable Softmax**: Optional max subtraction prevents overflow
- **Mixed Precision**: Support for different data types with appropriate accumulation
- **FP32 Accumulation**: Automatic enabling for FLOAT32 inputs

## Memory Requirements and Constraints

### L1 Memory Constraints
- Small tensor threshold: 512KB
- Automatic fallback for memory-constrained scenarios
- Circular buffer sizing based on tensor dimensions

### Tensor Layout Requirements
- All tensors must use TILE layout
- Input tensors must be on-device
- Mask tensors support both TILE and ROW_MAJOR layouts

### Sharding Requirements
- Sharded tensors require compatible program configurations
- Block dimensions must align with shard specifications
- ROW_MAJOR orientation required for certain specialized operations

## Error Handling and Validation

The implementation includes comprehensive validation:

1. **Device Placement**: All tensors must be on-device
2. **Layout Compatibility**: TILE layout required for most operations
3. **Data Type Support**: Validates supported data types (BFLOAT16, FLOAT32, BFLOAT8_B)
4. **Dimension Bounds**: Ensures dimension parameters are within tensor rank
5. **Memory Configuration**: Validates sharding and memory configuration compatibility
6. **Mask Compatibility**: Ensures mask tensors are compatible with input tensors

## Troubleshooting

### Common Issues and Solutions

#### Memory Errors
**Problem**: Out of L1 memory errors during execution
**Solutions**:
- Use `numeric_stable=False` if precision requirements allow
- Switch to in-place operations: `ttnn.softmax_in_place()` or `ttnn.scale_mask_softmax_in_place()`
- Use sharded program configuration for large tensors:
  ```python
  config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
      compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
      subblock_w=8, block_h=32, block_w=24
  )
  ```

#### Performance Issues

#### Numerical Instability
**Problem**: NaN or infinity values in output
**Solutions**:
- Enable numerical stability: `numeric_stable=True`
- Use FP32 accumulation for FLOAT32 inputs:
  ```python
  config = ttnn.DeviceComputeKernelConfig(fp32_dest_acc_en=True)
  ```
- Reduce input magnitude or normalize inputs before softmax
- Verify input tensor doesn't contain extremely large values

#### Tensor Shape Issues
**Problem**: Dimension errors or incompatible shapes
**Solutions**:
- Verify dimension parameter is within tensor rank: `-tensor.rank() <= dim < tensor.rank()`
- For mask operations, ensure mask tensor shapes are compatible:
  - Sharded masks: identical padded shape to input
  - Interleaved masks: batch dimension must match, intermediate dimensions must be 1
- Check that all tensors use TILE layout for computation

#### Sharding Configuration Errors
**Problem**: Invalid sharding configuration
**Solutions**:
- Ensure `block_w % subblock_w == 0`
- Verify `block_w * tile_width == tensor.shape[3]`
- Check that `num_cores == (M * K) / (block_w * block_h * tile_hw)`
- Use ROW_MAJOR shard orientation for specialized causal mask operations

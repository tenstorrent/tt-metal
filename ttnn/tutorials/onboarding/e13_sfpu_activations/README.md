# E13: SFPU Activations

Implement element-wise activation functions using the SFPU.

## Goal

Learn SFPU (Special Function Processing Unit) operations:
- Implement gelu, relu, silu activations
- Understand SFPU init and compute patterns
- Prepare for kernel fusion in e14

## Prerequisites

- **e04**: Basic kernel structure (reader/compute/writer)
- **e12**: Familiarity with compute kernels

## Reference

- `ttnn/cpp/ttnn/operations/eltwise/unary/`
- `tt_metal/include/tt_metal/compute_kernels/compute/`

## Key Concepts

### SFPU Overview
- Separate unit from FPU (matrix math)
- Handles transcendentals, activations, special functions
- Operates on tiles in dst registers

### Activation Functions
- **ReLU**: `max(0, x)` - simplest, fast
- **GELU**: `x * Φ(x)` - used in transformers
- **SiLU**: `x * sigmoid(x)` - smooth ReLU alternative

### Compute Pattern
```cpp
// Init (once at kernel start)
unary_op_init_common(cb_in, cb_out);
gelu_tile_init();  // or relu_tile_init(), etc.

// Per-tile loop
tile_regs_acquire();
cb_wait_front(cb_in, 1);
copy_tile(cb_in, 0, dst_idx);    // Load to dst register
gelu_tile(dst_idx);               // Apply activation
tile_regs_commit();

tile_regs_wait();
pack_tile(dst_idx, cb_out);
tile_regs_release();
cb_pop_front(cb_in, 1);
cb_push_back(cb_out, 1);
```

### Init Functions
Each SFPU op needs its own init:
- `relu_tile_init()`, `gelu_tile_init()`, `silu_tile_init()`
- Must re-init when switching between ops

## Common Pitfalls

1. **Missing init** - SFPU ops fail silently without init
2. **Wrong data format** - Some ops require specific formats
3. **Approximation modes** - GELU has fast vs accurate variants
4. **Register state** - SFPU modifies dst registers in-place

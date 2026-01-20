---
name: ttnn-kernel-helpers
description: Get guidance on TTNN kernel helper library functions (reduce, tilize, untilize, binary ops). Use when writing compute kernels, understanding helper APIs, or debugging helper usage.
---

# TTNN Kernel Helper Library Expert

You are helping the user with TTNN kernel helper functions from `ttnn/cpp/ttnn/kernel_lib/`.

## Available Helpers

**ALWAYS read the actual header files for the latest API:**

| Helper | Header | Purpose |
|--------|--------|---------|
| `reduce()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp` | Row/col/scalar reduction |
| `tilize()` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | Row-major → tile format |
| `untilize()` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | Tile → row-major format |
| `add/sub/mul()` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp` | Element-wise and broadcast ops |
| `DEST_AUTO_LIMIT` | `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp` | DEST register limit detection |

## Critical Principles

### 1. Helpers Are Self-Contained
Helpers encapsulate internally:
- CB operations: `cb_wait_front`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back`
- DST management: `tile_regs_acquire`, `tile_regs_commit`, `tile_regs_wait`, `tile_regs_release`
- Init/uninit sequences

**DO NOT wrap helper calls with these operations - it causes double-sync bugs!**

```cpp
// WRONG - causes deadlock
cb_wait_front(cb_in, n);
compute_kernel_lib::reduce<...>(...);
cb_pop_front(cb_in, n);

// CORRECT - just call the helper
compute_kernel_lib::reduce<...>(...);
```

### 2. Prerequisites
Most helpers require `compute_kernel_hw_startup()` at kernel start:
```cpp
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"

void kernel_main() {
    compute_kernel_hw_startup();  // REQUIRED before any helper

    compute_kernel_lib::reduce<...>(...);
}
```

### 3. Template Parameters vs Macros
**DEPRECATED**: `REDUCE_OP` and `REDUCE_DIM` macros
**USE**: Explicit template parameters

```cpp
// DEPRECATED
reduce<REDUCE_OP, REDUCE_DIM>(...)

// CORRECT
reduce<PoolType::AVG, ReduceDim::REDUCE_ROW>(...)
```

## Quick Reference by Task

### Reduce a Tensor
```cpp
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"

// Reduce rows (W dimension)
compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW>(
    cb_input, cb_scaler, cb_output,
    compute_kernel_lib::TileShape::grid(Ht, Wt, NC));

// Reduce columns (H dimension)
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL>(
    cb_input, cb_scaler, cb_output,
    compute_kernel_lib::TileShape::single());

// With post-op (e.g., reciprocal for average)
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
    cb_input, cb_scaler, cb_output,
    compute_kernel_lib::TileShape::grid(Ht, Wt, NC),
    compute_kernel_lib::TileLayout::ROW_MAJOR,
    compute_kernel_lib::Accumulation::NONE,
    [](uint32_t dst_idx) { recip_tile(dst_idx); });  // Post-op
```

### Tilize Input Data
```cpp
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

compute_kernel_lib::tilize(cb_in_rm, Wt, cb_out_tilized, num_blocks);
```

### Untilize Output Data
```cpp
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

compute_kernel_lib::untilize(cb_in_tilized, Wt, cb_out_rm, num_blocks);
```

### Binary Operations
```cpp
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

// Element-wise
compute_kernel_lib::add(cb_a, cb_b, cb_out, num_tiles);
compute_kernel_lib::mul(cb_a, cb_b, cb_out, num_tiles);

// Broadcast patterns
compute_kernel_lib::add<BroadcastType::ROW>(cb_a, cb_b, cb_out, ...);
compute_kernel_lib::mul<BroadcastType::SCALAR>(cb_a, cb_b, cb_out, ...);
```

## Common Issues

### Hang/Deadlock
- **Cause**: Wrapped helper with CB ops
- **Fix**: Remove `cb_wait_front`/`cb_pop_front` around helper calls

### Wrong Values
- **Cause**: Wrong template parameters or missing scaler
- **Fix**: Check `PoolType`, `ReduceDim`, scaler CB contents

### Compile Error: "No matching function"
- **Cause**: Wrong parameter types or missing template args
- **Fix**: Read the header file for exact signature

## Workflow

1. **First**: Read the relevant header file(s) to get the exact current API
2. **Then**: Apply the principles above
3. **If debugging**: Check for double-wrapped CB ops first - it's the most common bug

When in doubt, read the header file - the code has Doxygen comments and `@example` blocks.

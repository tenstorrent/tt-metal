---
name: ttnn-cb-patterns
description: Get guidance on circular buffer (CB) configuration, synchronization, and common patterns in TTNN operations. Use when designing CB flow, debugging hangs, or understanding producer-consumer patterns.
---

# TTNN Circular Buffer Patterns Expert

You are helping the user with circular buffer (CB) patterns in TTNN operations.

## CB Fundamentals

Circular buffers are SRAM-based page queues for producer-consumer synchronization between kernels.

### API Reference

| Function | Role | Description |
|----------|------|-------------|
| `cb_reserve_back(cb, n)` | Producer | Block until space for n pages |
| `cb_push_back(cb, n)` | Producer | Publish n written pages |
| `cb_wait_front(cb, n)` | Consumer | Block until n pages available |
| `cb_pop_front(cb, n)` | Consumer | Free n processed pages |

### The Golden Rule
**Total pushes MUST equal total pops for each CB across all kernels.**

## Common CB IDs

| CB ID | Typical Use | Producer | Consumer |
|-------|-------------|----------|----------|
| `cb::c_in0` (0) | Input data | Reader | Compute |
| `cb::c_in1` (1) | Secondary input/scaler | Reader | Compute |
| `cb::c_intermed0` (24) | Intermediate | Compute | Compute |
| `cb::c_intermed1` (25) | Intermediate | Compute | Compute |
| `cb::c_out0` (16) | Output | Compute | Writer |

## Buffering Strategies

**Define buffering factor as constexpr at top of factory:**
```cpp
constexpr uint32_t buffering_factor = 2;  // Double buffering
```

### Single Buffering (Simple)
```cpp
#include "ttnn/operations/cb_utils.hpp"

tt::tt_metal::create_cb(cb_id, program, all_cores, tile_size, 1, data_format);
```
- 1 page capacity
- No overlap between producer and consumer
- Use for: Simple ops, debugging, scaler CBs

### Double Buffering (Recommended)
```cpp
constexpr uint32_t buffering_factor = 2;

tt::tt_metal::create_cb(cb_id, program, all_cores, tile_size, buffering_factor, data_format);
```
- 2 page capacity
- Producer fills page 2 while consumer processes page 1
- Use for: Most production operations

### N-Buffering (Advanced)
```cpp
// For operations that need to hold multiple tiles (e.g., full row for reduction)
const uint32_t tiles_to_hold = Wt;
tt::tt_metal::create_cb(cb_id, program, all_cores, tile_size, tiles_to_hold, data_format);
```
- Use for: Reduce ops holding partial results, matmul accumulation

## Common Patterns

### Pattern 1: Simple Pipeline (Reader → Compute → Writer)
```
Reader                    Compute                   Writer
────────────────────      ────────────────────      ────────────────────
cb_reserve_back(c_in0)    cb_wait_front(c_in0)      cb_wait_front(c_out0)
// read data              // process                // write data
cb_push_back(c_in0)       cb_pop_front(c_in0)       cb_pop_front(c_out0)
                          cb_reserve_back(c_out0)
                          // pack result
                          cb_push_back(c_out0)
```

### Pattern 2: Compute with Intermediate CB
```cpp
// Phase 1: Read from input, write to intermediate
cb_wait_front(cb_in, 1);
// ... compute ...
cb_pop_front(cb_in, 1);
cb_reserve_back(cb_intermed, 1);
// ... pack ...
cb_push_back(cb_intermed, 1);

// Phase 2: Read from intermediate, write to output
cb_wait_front(cb_intermed, 1);
// ... compute ...
cb_pop_front(cb_intermed, 1);
cb_reserve_back(cb_out, 1);
// ... pack ...
cb_push_back(cb_out, 1);
```

### Pattern 3: Scaler CB (Read Once, Use Many)
```cpp
// Reader: Push scaler once
cb_reserve_back(cb_scaler, 1);
// ... write scaler tile ...
cb_push_back(cb_scaler, 1);

// Compute: Use scaler multiple times without popping
for (...) {
    cb_wait_front(cb_scaler, 1);  // Always sees same scaler
    // ... use scaler in computation ...
    // NO cb_pop_front here!
}
cb_pop_front(cb_scaler, 1);  // Pop once at the end
```

### Pattern 4: Block Processing
```cpp
// Process Wt tiles per row, Ht rows per batch
for (uint32_t h = 0; h < Ht; ++h) {
    for (uint32_t w = 0; w < Wt; ++w) {
        cb_wait_front(cb_in, 1);
        // ... process tile ...
        cb_pop_front(cb_in, 1);
    }
    // Row complete - push output
    cb_reserve_back(cb_out, 1);
    // ... pack result ...
    cb_push_back(cb_out, 1);
}
```

## CB Configuration in Program Factory

```cpp
#include "ttnn/operations/cb_utils.hpp"

constexpr uint32_t buffering_factor = 2;
const uint32_t tile_size = tt::tile_size(data_format);

// Modern API - simple and clean
tt::tt_metal::create_cb(cb_id, program, all_cores, tile_size, buffering_factor, data_format);

// For sharded (globally allocated) - pass buffer, num_pages from shard spec
const uint32_t shard_pages = input.shard_spec().value().shape[0];
tt::tt_metal::create_cb(cb_id, program, all_cores, page_size, shard_pages, data_format,
    input.buffer());
```

## Debugging Hangs

### Symptom: Kernel hangs
**Common cause**: CB sync mismatch

### Debugging Checklist:
1. **Count push/pop pairs** for each CB across all kernels
2. **Check page counts** - producer and consumer must agree on N
3. **Look for missing pops** - especially in error paths or early returns
4. **Check helper usage** - helpers handle CB ops internally, don't duplicate

### Verification Pattern:
```
CB c_in0:
  Reader pushes: N tiles
  Compute pops: N tiles  ✓ Match

CB c_out0:
  Compute pushes: M tiles
  Writer pops: M tiles   ✓ Match
```

## IMPORTANT: Helpers Handle CB Ops

When using kernel helpers (`compute_kernel_lib::reduce()`, etc.), the helpers handle CB operations internally.

```cpp
// WRONG - helper already does CB ops
cb_wait_front(cb_in, 1);
compute_kernel_lib::reduce<...>(...);  // This also does cb_wait/pop!
cb_pop_front(cb_in, 1);  // Double pop = hang

// CORRECT
compute_kernel_lib::reduce<...>(...);  // Just call the helper
```

## Workflow

1. **Design CB flow first** - draw the producer/consumer relationships
2. **Count pages** - verify push = pop for each CB
3. **Choose buffering** - single for debug, double for production
4. **Implement** - follow patterns above
5. **Debug** - if hang, check CB sync first

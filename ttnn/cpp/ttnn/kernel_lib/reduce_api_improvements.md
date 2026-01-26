# Reduce Helpers API Improvements

Analysis and recommendations for making the reduce helpers API cleaner and easier to use.

## Current API

```cpp
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t ocb,
    typename InputPolicy = policies::StreamingPolicy,
    typename ReconfigPolicy = policies::ReconfigBothPolicy,
    typename AccumT = NoAccumulation,
    typename PostReduceOp = NoOp>
ALWI void reduce(
    TileGrid grid,
    InputMemoryLayout layout = InputMemoryLayout::contiguous(),
    AccumT accum = AccumT{},
    PostReduceOp post_reduce_op = PostReduceOp{});
```

---

## Issue 1: Bool template parameters (`init`, `uninit`)

**RESOLVED:** Removed init policy entirely. We always call `reduce_init` at start and `reduce_uninit` at end. No call sites needed custom init/uninit behavior, so the configurability was unnecessary complexity.

---

## Issue 2: Avoiding `{}, {}` for accum and post_reduce_op

### Problem

Current signature forces you to pass `layout` and `accum` just to specify `post_reduce_op`:
```cpp
reduce<...>(cb_in, cb_scaler, cb_out, shape, {}, {}, my_lambda);  // What are {}, {}?
```

### Solution

Make default constructors `explicit` to enforce named types at call sites:

```cpp
struct NoAccumulation {
    explicit NoAccumulation() = default;
};

struct NoOp {
    explicit NoOp() = default;
    ALWI void operator()(uint32_t = 0) const {}
};

struct InputMemoryLayout {
    uint32_t row_stride = 0;
    uint32_t batch_stride = 0;

    explicit InputMemoryLayout() = default;

    static constexpr InputMemoryLayout contiguous() { return InputMemoryLayout(); }
    static constexpr InputMemoryLayout with_row_stride(uint32_t s) { return {s, 0}; }
};
```

With `explicit` default constructors:
- `func({})` fails (copy-list-initialization blocked)
- `func(NoAccumulation{})` works (direct-list-initialization allowed)

Call sites become self-documenting:
```cpp
reduce<...>(cb_in, cb_scaler, cb_out, shape,
    InputMemoryLayout::contiguous(),
    NoAccumulation{},
    [](uint32_t dst) { ... });
```

Apply this to `NoAccumulation`, `NoOp`, and `InputMemoryLayout`.

---

## Issue 3: TileShape naming

### Problem

"TileShape" sounds like "shape of one tile" (32x32 elements), but it's actually "grid dimensions in tiles".

### Better names

- **`TileGrid`** - Clear that it's a grid of tiles (rows × cols × batches)
- **`GridExtent`** - Emphasizes it's about extent/dimensions
- **`InputGrid`** - Specific to input dimensions

### Recommendation

Rename to **`TileGrid`** - matches the factory method `TileGrid::grid(r, c, b)` nicely.

Also rename the factory methods for consistency:
```cpp
struct TileGrid {
    uint32_t rows;
    uint32_t cols;
    uint32_t batches;

    static constexpr TileGrid of(uint32_t r, uint32_t c, uint32_t b = 1);
    static constexpr TileGrid single();
    static constexpr TileGrid row(uint32_t c, uint32_t b = 1);
    static constexpr TileGrid col(uint32_t r, uint32_t b = 1);
};
```

---

## Issue 4: Replace `ReduceInputMode` enum with policy structs

### Problem

The `ReduceInputMode` enum conflates multiple orthogonal behaviors:
```cpp
enum class ReduceInputMode { STREAMING, STREAMING_BATCHED, PRELOADED, PERSISTENT };
```

Each mode implicitly encodes: when to wait, whether to pop, and access pattern. This makes it hard to understand what each mode does and prevents custom combinations.

| Mode | Wait | Pop | Access |
|------|------|-----|--------|
| STREAMING | per-tile | yes | sequential |
| STREAMING_BATCHED | per-batch | yes | indexed |
| PRELOADED | none | no | indexed |
| PERSISTENT | upfront | no | indexed |

### Solution

Use policy structs with enum members (inspired by `normalization/kernel_util/compute/policies.h`):

```cpp
namespace compute_kernel_lib::policies {

/**
 * @brief When to synchronize on input tiles
 */
enum class WaitMode {
    PER_TILE,   // wait/process/pop one tile at a time
    PER_BATCH,  // wait for batch, process all, pop batch
    UPFRONT,    // wait for everything upfront
    NONE        // caller manages synchronization
};

/**
 * @brief Whether to pop tiles after processing
 */
enum class PopMode {
    POP,        // pop tiles after processing
    NO_POP      // leave tiles in CB (for reuse)
};

// =============================================================================
// Input synchronization policies
// =============================================================================

struct StreamingPolicy {
    static constexpr WaitMode wait = WaitMode::PER_TILE;
    static constexpr PopMode pop = PopMode::POP;
};

struct StreamingBatchedPolicy {
    static constexpr WaitMode wait = WaitMode::PER_BATCH;
    static constexpr PopMode pop = PopMode::POP;
};

struct PreloadedPolicy {
    static constexpr WaitMode wait = WaitMode::NONE;
    static constexpr PopMode pop = PopMode::NO_POP;
};

struct PersistentPolicy {
    static constexpr WaitMode wait = WaitMode::UPFRONT;
    static constexpr PopMode pop = PopMode::NO_POP;
};

}  // namespace compute_kernel_lib::policies
```

Function signature changes from enum to policy type:
```cpp
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    typename InputPolicy = policies::StreamingPolicy,  // policy struct
    ...>
ALWI void reduce(...);
```

Implementation uses policy members:
```cpp
// Wait logic
if constexpr (InputPolicy::wait == WaitMode::UPFRONT) {
    cb_wait_front(icb, total_tiles);
}
if constexpr (InputPolicy::wait == WaitMode::PER_BATCH) {
    cb_wait_front(icb, tiles_per_batch);
}
if constexpr (InputPolicy::wait == WaitMode::PER_TILE) {
    cb_wait_front(icb, onetile);
}

// Access - indexed unless per-tile streaming
constexpr bool indexed = (InputPolicy::wait != WaitMode::PER_TILE);

// Pop logic
if constexpr (InputPolicy::pop == PopMode::POP) {
    // pop based on wait granularity
}
```

**Benefits:**
- Self-documenting: reading `PersistentPolicy` shows exactly what it does
- No invalid states: enums prevent conflicting bool combinations
- Extensible: users can define custom policy structs
- Same compile-time elimination via `if constexpr`

---

## Additional Suggestions

### A. Group the CBs

```cpp
struct ReduceCBs {
    uint32_t input;
    uint32_t scaler;
    uint32_t output;
};
```

Call site becomes:
```cpp
reduce<SUM, REDUCE_SCALAR>(
    ReduceCBs{cb_in, cb_scaler, cb_out},
    TileGrid::of(Ht, Wt, NC));
```

This prevents argument swapping bugs and is self-documenting.

### B. Consider flipping template param order

Currently the rarely-changed params (`reconfig`, `init`, `uninit`) come before the often-changed type params. If grouped into a struct, the signature becomes:
```cpp
template <PoolType reduce_type, ReduceDim reduce_dim,
          ReduceInputMode input_mode = STREAMING,
          typename AccumT = NoAccumulation,
          typename PostReduceOp = NoOp>
void reduce(ReduceCBs cbs, TileGrid grid, ReduceConfig cfg = {}, ...);
```

---

## Summary

| Issue | Recommendation | Status |
|-------|----------------|--------|
| Bool template args | Removed entirely - always init/uninit | ✅ Removed |
| `{}, {}` for optional params | `explicit` default constructors to enforce `NoAccumulation{}` over `{}` | ✅ Implemented |
| TileShape naming | Rename to `TileGrid` | ✅ Implemented |
| ReduceInputMode enum | Policy structs with `WaitMode`/`PopMode` enums | ✅ Implemented |
| CB arguments | Moved to constexpr template params | ✅ Implemented |

---

## Implementation History

### Issue 1: Init policy (Removed)

**Date:** 2026-01-26

**Changes made:**
1. Removed `InitUninitPolicy`, `InitOnlyPolicy`, `UninitOnlyPolicy`, `NoInitPolicy` structs from `reduce_helper_policies.hpp`
2. Removed `InitPolicy` template parameter from `reduce()` function
3. Made `reduce_init` and `reduce_uninit` calls unconditional

**Rationale:** No call sites needed custom init/uninit behavior - all used the default (both init and uninit). The configurability added complexity without benefit.

**Files modified:**
- `ttnn/cpp/ttnn/kernel_lib/reduce_helper_policies.hpp`
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`

### Issue 2: Explicit default constructors (Implemented)

**Date:** 2026-01-22

**Changes made:**
1. Added `explicit` to default constructors of `InputMemoryLayout`, `NoAccumulation`, and `NoOp`
2. Added two-argument constructor to `InputMemoryLayout` for factory method usage
3. Updated function signature default parameters from `{}` to explicit constructions:
   - `InputMemoryLayout layout = InputMemoryLayout::contiguous()`
   - `AccumT accum = AccumT{}`
   - `PostReduceOp post_reduce_op = PostReduceOp{}`
4. Updated all call sites using `{}` to use explicit type names

**Files modified:**
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/compute_common.hpp`
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax_large_tensor.cpp`
- `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_h_large.cpp`
- `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`
- `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp`
- `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_h.cpp`
- `ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/moreh_bias_backward_multi_core_h.cpp`
- `ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/moreh_bias_backward_single_core_hw.cpp`
- `ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/moreh_sum_h.cpp`
- `ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_h.cpp`

**API change:**

Before:
```cpp
reduce<SUM, REDUCE_ROW>(..., shape, {}, {}, my_lambda);  // What are {}, {}?
```

After:
```cpp
reduce<SUM, REDUCE_ROW>(..., shape,
    InputMemoryLayout::contiguous(),
    NoAccumulation{},
    my_lambda);  // Self-documenting
```

**Migration guide:**
- `{}` for layout → `InputMemoryLayout::contiguous()` or `compute_kernel_lib::InputMemoryLayout::contiguous()`
- `{}` for accum → `NoAccumulation{}` or `compute_kernel_lib::NoAccumulation{}`
- `{}` for post_reduce_op → `NoOp{}` or omit (uses default)

**Testing:**
- Ran `pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py` - all tests pass
- Ran `pytest tests/ttnn/unit_tests/operations/fused/test_softmax.py` - all tests pass

### Issue 3: Rename TileShape to TileGrid (Implemented)

**Date:** 2026-01-22

**Changes made:**
1. Renamed struct `TileShape` to `TileGrid` in `reduce_helpers_compute.hpp`
2. Renamed factory method `grid()` to `of()` for clarity (`TileGrid::of(r, c, b)`)
3. Added backward compatibility alias: `using TileShape = TileGrid;`
4. Updated function signature parameter from `TileShape shape` to `TileGrid grid`
5. Migrated all call sites from `TileShape::*` to `TileGrid::*`
6. Updated all documentation examples

**Files modified:**
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
- All 40+ kernel files using TileShape (migrated to TileGrid)

**API change:**

Before:
```cpp
TileShape::grid(Ht, Wt, NC)  // confusing - sounds like "shape of grid" not "grid of tiles"
TileShape::single()
TileShape::row(Wt)
TileShape::col(Ht)
```

After:
```cpp
TileGrid::of(Ht, Wt, NC)    // clearer - "a TileGrid of these dimensions"
TileGrid::single()
TileGrid::row(Wt)
TileGrid::col(Ht)
```

**Migration guide:**
- `TileShape::grid(r, c, b)` → `TileGrid::of(r, c, b)`
- `TileShape::single()` → `TileGrid::single()`
- `TileShape::row(c)` → `TileGrid::row(c)`
- `TileShape::col(r)` → `TileGrid::col(r)`
- Old code using `TileShape` still works via alias (backward compatible)

**Testing:**
- Ran `pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py` - all tests pass
- Ran `pytest tests/ttnn/unit_tests/operations/fused/test_softmax.py` - all tests pass

### Issue 4: Input Policy Structs (Implemented)

**Date:** 2026-01-22

**Changes made:**
1. Added `policies` namespace with `WaitMode` and `PopMode` enums
2. Added policy structs: `StreamingPolicy`, `StreamingBatchedPolicy`, `PreloadedPolicy`, `PersistentPolicy`
3. Added type traits to detect policy structs
4. Added `InputModeToPolicy` helper for backward compatibility with `ReduceInputMode` enum
5. Existing code using `ReduceInputMode` enum continues to work (backward compatible)

**Files modified:**
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`

**New API (policy structs):**

```cpp
namespace compute_kernel_lib::policies {

enum class WaitMode { PER_TILE, PER_BATCH, UPFRONT, NONE };
enum class PopMode { POP, NO_POP };

struct StreamingPolicy {       // wait: PER_TILE, pop: POP
    static constexpr WaitMode wait = WaitMode::PER_TILE;
    static constexpr PopMode pop = PopMode::POP;
};

struct StreamingBatchedPolicy { // wait: PER_BATCH, pop: POP
    static constexpr WaitMode wait = WaitMode::PER_BATCH;
    static constexpr PopMode pop = PopMode::POP;
};

struct PreloadedPolicy {        // wait: NONE, pop: NO_POP
    static constexpr WaitMode wait = WaitMode::NONE;
    static constexpr PopMode pop = PopMode::NO_POP;
};

struct PersistentPolicy {       // wait: UPFRONT, pop: NO_POP
    static constexpr WaitMode wait = WaitMode::UPFRONT;
    static constexpr PopMode pop = PopMode::NO_POP;
};

}  // namespace compute_kernel_lib::policies
```

**Benefits:**
- Self-documenting: reading `PersistentPolicy` shows exactly what it does
- No invalid states: enum members prevent conflicting bool combinations
- Extensible: users can define custom policy structs with their own wait/pop behavior
- Same compile-time elimination via `if constexpr`

**Migration guide (optional):**
- Existing `ReduceInputMode::STREAMING` → `policies::StreamingPolicy` (or keep using enum)
- Existing `ReduceInputMode::STREAMING_BATCHED` → `policies::StreamingBatchedPolicy`
- Existing `ReduceInputMode::PRELOADED` → `policies::PreloadedPolicy`
- Existing `ReduceInputMode::PERSISTENT` → `policies::PersistentPolicy`

**Testing:**
- Ran `pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py` - all tests pass
- Backward compatibility preserved - existing code unchanged

### Additional A: ReduceCBs struct (Implemented)

**Date:** 2026-01-22

**Changes made:**
1. Added `ReduceCBs` struct to group input, scaler, and output CB arguments
2. Added `ReduceCBs::of(in, scaler, out)` factory method for inline construction
3. Added function overload accepting `ReduceCBs` instead of three separate arguments
4. Original three-argument signature preserved for backward compatibility

**Files modified:**
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`

**New API:**

```cpp
struct ReduceCBs {
    uint32_t input;   // Input CB containing tiles to reduce
    uint32_t scaler;  // CB containing scaler tile
    uint32_t output;  // Output CB for reduced tiles

    static constexpr ReduceCBs of(uint32_t in, uint32_t scaler, uint32_t out) { return {in, scaler, out}; }
};

// New overload accepting ReduceCBs
template <PoolType reduce_type, ReduceDim reduce_dim, ...>
void reduce(ReduceCBs cbs, TileGrid grid, ...);
```

**Usage example:**

```cpp
// Before (still works):
compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
    cb_in, cb_scaler, cb_out,
    TileGrid::of(Ht, Wt, NC));

// After (new option):
compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
    ReduceCBs::of(cb_in, cb_scaler, cb_out),
    TileGrid::of(Ht, Wt, NC));
```

**Benefits:**
- Prevents argument-swapping bugs (clear which CB is input vs scaler vs output)
- Self-documenting at call sites
- Backward compatible - original three-argument signature still works

**Testing:**
- Ran `pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py` - all tests pass
- Backward compatibility preserved - existing code unchanged

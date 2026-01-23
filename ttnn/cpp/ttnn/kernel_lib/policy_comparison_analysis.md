# Policy Implementation Comparison Analysis

## Overview

This document compares two policy implementations for TTNN kernel helpers:

1. **Reduce Helpers** (branch: `sjovic/llk-helper-reduce`) - `reduce_helper_policies.hpp`
2. **Binary Op Helpers** (branch: `mstaletovic/binary_helpers_policy_rework`) - `cb_policies.hpp`

Both originated from similar requirements: controlling circular buffer synchronization patterns (wait/pop timing) for compute kernels.

---

## Architecture Comparison

### Reduce Helpers: Enum + Flat Structs

```cpp
// Single enum for wait timing
enum class WaitMode {
    PER_TILE,   // wait/process/pop one tile at a time
    PER_BATCH,  // wait for batch, process all, pop batch
    UPFRONT,    // wait for everything upfront
    NONE        // caller manages synchronization
};

// Flat policy structs
struct StreamingPolicy {
    static constexpr WaitMode wait = WaitMode::PER_TILE;
    static constexpr bool pop = true;
};

struct PersistentPolicy {
    static constexpr WaitMode wait = WaitMode::UPFRONT;
    static constexpr bool pop = false;
};
```

**Characteristics:**
- 2 fields per input policy: `wait` (enum) + `pop` (bool)
- Separate policy families for different concerns (Input, Init, Reconfig)
- Policy selection via `if constexpr` on enum values
- Simple, easy to understand at a glance

### Binary Op Helpers: Composable Type Templates

```cpp
// Separate Wait policy types (each with multiple bool flags)
struct WaitPerTile {
    static constexpr bool per_tile = true;
    static constexpr bool per_chunk = false;
    static constexpr bool upfront = false;
    static constexpr bool caller_managed = false;
};

// Separate Pop policy types
struct PopPerTile {
    static constexpr bool per_tile = true;
    static constexpr bool per_chunk = false;
    static constexpr bool at_end = false;
    static constexpr bool never = false;
    static constexpr bool caller_managed = false;
};

// Combined via template
template <typename WaitPolicy, typename PopPolicy>
struct InputPolicy {
    using wait = WaitPolicy;
    using pop = PopPolicy;

    // Convenience accessors flatten the nested bools
    static constexpr bool waits_per_tile = WaitPolicy::per_tile;
    static constexpr bool pops_per_tile = PopPolicy::per_tile;
    // ... etc
};

// Predefined combinations
using Streaming = InputPolicy<WaitPerTile, PopPerTile>;
using Persistent = InputPolicy<WaitUpfront, PopNever>;
```

**Characteristics:**
- Many bool fields per primitive policy type
- Composable via templates: `InputPolicy<WaitPolicy, PopPolicy>`
- Type traits for runtime policy detection
- More flexible but more complex

---

## Detailed Comparison

### 1. Input Policy Design

| Aspect | Reduce Helpers | Binary Op Helpers |
|--------|----------------|-------------------|
| **Wait representation** | Single `WaitMode` enum | Separate struct types with 4 bool flags each |
| **Pop representation** | Single `bool pop` | Separate struct types with 5 bool flags each |
| **Composition** | Flat struct with 2 fields | Template `InputPolicy<Wait, Pop>` |
| **Predefined policies** | 4 named structs | 4 type aliases |
| **Custom combinations** | Not supported | Fully supported via template |

### 2. Policy Consumption in Code

**Reduce Helpers:**
```cpp
// Uses enum comparison
if constexpr (InputPolicy::wait == policies::WaitMode::PER_TILE) {
    cb_wait_front(icb, onetile);
    // process
    cb_pop_front(icb, onetile);
} else if constexpr (InputPolicy::wait == policies::WaitMode::PER_BATCH) {
    // batched logic
}
```

**Binary Op Helpers:**
```cpp
// Uses bool flags
if constexpr (InputAPolicy::waits_per_tile) {
    cb_wait_front(icb_a, onetile);
}
if constexpr (InputAPolicy::pops_per_tile) {
    cb_pop_front(icb_a, onetile);
}
```

### 3. Separate Concern Policies

| Concern | Reduce Helpers | Binary Op Helpers |
|---------|----------------|-------------------|
| **Init/Uninit lifecycle** | `InitBothPolicy`, `InitOnlyPolicy`, `UninitOnlyPolicy`, `NoInitPolicy` | Single `bool init` template param |
| **Data format reconfig** | `ReconfigNonePolicy`, `ReconfigInputPolicy`, `ReconfigOutputPolicy`, `ReconfigBothPolicy` | `BinaryDataFormatReconfig` enum (NONE, INPUT, OUTPUT, BOTH) |
| **Output CB handling** | Derived from input policy | Separate `OutputPerTile`, `OutputPerChunk`, `OutputBulk` policies |

### 4. Type Safety & Extensibility

| Aspect | Reduce Helpers | Binary Op Helpers |
|--------|----------------|-------------------|
| **Type safety** | Medium - enum prevents invalid combinations | High - template enforces valid Wait+Pop pairs |
| **Custom policies** | Requires new struct definition | Can compose existing primitives |
| **Type traits** | None | `is_streaming_policy_v`, `is_input_policy_v`, etc. |
| **Compile-time validation** | Limited | Better via `static_assert` on traits |

---

## Critique

### Reduce Helpers - Strengths

1. **Simplicity**: Two fields (`wait` enum + `pop` bool) are easy to understand
2. **Readability**: `WaitMode::PER_TILE` is more descriptive than `waits_per_tile = true`
3. **Compact**: Policy structs are small (2 fields vs 4-5 bools)
4. **Clear separation**: Input, Init, Reconfig policies are distinct types
5. **Self-documenting enums**: `WaitMode::UPFRONT` immediately conveys intent

### Reduce Helpers - Weaknesses

1. **No custom combinations**: Can't mix `WaitMode::UPFRONT` with `pop = true` without new struct
2. **Enum branching**: Multiple `if constexpr` chains on enum values
3. **No type traits**: Can't easily detect policy type at compile time
4. **Pop is binary**: Only yes/no, no `PopAtEnd` vs `PopNever` distinction

### Binary Op Helpers - Strengths

1. **Full composability**: Any Wait + Pop combination is valid
2. **Fine-grained pop control**: `PopAtEnd`, `PopNever`, `PopCallerManaged` are distinct
3. **Type traits**: Enable compile-time policy detection and validation
4. **Separate output policy**: Input and output CB handling are independent concerns
5. **Future-proof**: New wait/pop modes can be added without changing existing policies

### Binary Op Helpers - Weaknesses

1. **Complexity**: 4-5 bool fields per primitive type is verbose
2. **Redundancy**: Each policy type has N-1 false bools (only one is true)
3. **Indirection**: `InputPolicy::waits_per_tile` accesses `WaitPolicy::per_tile`
4. **Learning curve**: Template composition requires understanding the primitives
5. **No init policies**: Init handling is a bare `bool` template param, not a policy

---

## Recommendations for Unified Design

### What to Take from Reduce Helpers

1. **Enum for mutually-exclusive modes**: `WaitMode` enum is cleaner than N bool flags
2. **Separate policy families**: Keep Input, Init, Reconfig as distinct concerns
3. **Self-documenting factory methods**: `TileGrid::col(Ht)`, `ReduceCBs::of(...)` pattern

### What to Take from Binary Op Helpers

1. **Composable input policy template**: `InputPolicy<WaitPolicy, PopPolicy>`
2. **Fine-grained pop modes**: Distinguish `PopAtEnd` vs `PopNever` vs `PopCallerManaged`
3. **Separate output policy**: Output CB handling should be independent
4. **Type traits**: Enable compile-time policy validation

### Proposed Unified Design

```cpp
namespace cb_policies {

// === Wait Modes (enum, not bool flags) ===
enum class WaitMode {
    PER_TILE,       // wait 1 tile at a time
    PER_CHUNK,      // wait for DEST_LIMIT tiles
    UPFRONT,        // wait for all tiles at start
    CALLER_MANAGED  // caller handles wait
};

// === Pop Modes (enum, not bool flags) ===
enum class PopMode {
    PER_TILE,       // pop immediately after processing
    PER_CHUNK,      // pop after chunk processed
    AT_END,         // pop all at end of operation
    NEVER,          // tiles persist (no pop)
    CALLER_MANAGED  // caller handles pop
};

// === Output Modes (enum) ===
enum class OutputMode {
    PER_TILE,   // reserve/push 1 tile at a time
    PER_CHUNK,  // reserve/push chunks
    BULK        // reserve all upfront, push all at end
};

// === Composable Input Policy ===
template <WaitMode W, PopMode P>
struct InputPolicy {
    static constexpr WaitMode wait = W;
    static constexpr PopMode pop = P;
};

// === Predefined Combinations ===
using Streaming       = InputPolicy<WaitMode::PER_TILE, PopMode::PER_TILE>;
using StreamingBatched = InputPolicy<WaitMode::PER_CHUNK, PopMode::PER_CHUNK>;
using Preloaded       = InputPolicy<WaitMode::CALLER_MANAGED, PopMode::CALLER_MANAGED>;
using Persistent      = InputPolicy<WaitMode::UPFRONT, PopMode::NEVER>;

// Custom combinations are now possible:
using WaitAllPopAtEnd = InputPolicy<WaitMode::UPFRONT, PopMode::AT_END>;

// === Init Policy (keep from reduce helpers) ===
struct InitBothPolicy {
    static constexpr bool init = true;
    static constexpr bool uninit = true;
};
// ... InitOnlyPolicy, UninitOnlyPolicy, NoInitPolicy

// === Reconfig Policy (keep from reduce helpers) ===
struct ReconfigBothPolicy {
    static constexpr bool reconfig_input = true;
    static constexpr bool reconfig_output = true;
};
// ... ReconfigNonePolicy, ReconfigInputPolicy, ReconfigOutputPolicy

// === Type Traits ===
template <typename T>
struct is_input_policy : std::false_type {};

template <WaitMode W, PopMode P>
struct is_input_policy<InputPolicy<W, P>> : std::true_type {};

template <typename T>
inline constexpr bool is_input_policy_v = is_input_policy<T>::value;

}  // namespace cb_policies
```

### Benefits of Unified Design

1. **Enums for clarity**: `WaitMode::UPFRONT` is clearer than 4 bool flags
2. **Template for flexibility**: Custom Wait+Pop combinations without new structs
3. **Type traits for safety**: Compile-time validation of policy types
4. **Separate concerns**: Input, Output, Init, Reconfig are all distinct
5. **Backward compatible**: Predefined aliases match existing usage patterns
6. **Minimal redundancy**: No N-1 false bools per policy type

---

## Migration Path

### Phase 1: Unify Input Policies
- Create shared `cb_policies.hpp` with enum-based `InputPolicy<WaitMode, PopMode>`
- Update both reduce and binary helpers to use it
- Keep predefined aliases for backward compatibility

### Phase 2: Unify Output Policies
- Add `OutputMode` enum and output policy types
- Binary helpers already have this; reduce helpers can adopt

### Phase 3: Standardize Init/Reconfig
- Move `InitPolicy` and `ReconfigPolicy` to shared header
- Both helpers use the same policy types

### Phase 4: Add Type Traits
- Implement `is_input_policy_v`, `is_init_policy_v`, etc.
- Add `static_assert` validation in helper functions

---

## Conclusion

Both implementations have merit. The reduce helpers prioritize **simplicity and readability**, while the binary helpers prioritize **composability and type safety**.

The recommended unified design takes:
- **Enums** from reduce helpers (cleaner than bool flags)
- **Template composition** from binary helpers (custom combinations)
- **Separate policy families** from reduce helpers (clear separation of concerns)
- **Type traits** from binary helpers (compile-time validation)

This gives the best of both worlds: simple to use for common cases, flexible for advanced cases, and type-safe throughout.

---

## TileGrid vs BinaryTileShape Comparison

Both implementations need to specify the dimensions of the tile grid being processed. They take different approaches.

### Reduce Helpers: TileGrid

```cpp
struct TileGrid {
    uint32_t rows;
    uint32_t cols;
    uint32_t batches;

    // Factory methods
    static constexpr TileGrid of(uint32_t r, uint32_t c, uint32_t b = 1) { return {r, c, b}; }
    static constexpr TileGrid single() { return {1, 1, 1}; }
    static constexpr TileGrid row(uint32_t c, uint32_t b = 1) { return {1, c, b}; }
    static constexpr TileGrid col(uint32_t r, uint32_t b = 1) { return {r, 1, b}; }
};
```

**Usage examples:**
```cpp
// Full grid with batches
reduce<SUM, REDUCE_SCALAR>(cbs, TileGrid::of(Ht, Wt, NC));

// Single column of tiles (common for REDUCE_COL)
reduce<MAX, REDUCE_COL>(cbs, TileGrid::col(Ht));

// Single tile
reduce<SUM, REDUCE_SCALAR>(cbs, TileGrid::single());
```

### Binary Op Helpers: BinaryTileShape

```cpp
struct BinaryTileShape {
    uint32_t rows, cols;

    // Factory methods
    static constexpr BinaryTileShape single() { return {1, 1}; }
    static constexpr BinaryTileShape row(uint32_t cols) { return {1, cols}; }
    static constexpr BinaryTileShape col(uint32_t rows) { return {rows, 1}; }
    static constexpr BinaryTileShape grid(uint32_t rows, uint32_t cols) { return {rows, cols}; }
};
```

**Usage examples:**
```cpp
// Full grid (no batches)
binary_op<ADD>(icb_a, icb_b, ocb, BinaryTileShape::grid(Ht, Wt));

// Single row of tiles
add(icb_a, icb_b, ocb, BinaryTileShape::row(Wt));

// Single tile
mul(icb_a, icb_b, ocb, BinaryTileShape::single());
```

### Comparison Table

| Aspect | TileGrid (Reduce) | BinaryTileShape (Binary) |
|--------|-------------------|--------------------------|
| **Fields** | `rows`, `cols`, `batches` | `rows`, `cols` |
| **Batch support** | Built-in (3rd dimension) | Not supported |
| **Primary factory** | `of(r, c, b)` | `grid(r, c)` |
| **Naming** | "Grid" (emphasizes 2D/3D structure) | "Shape" (emphasizes dimensions) |
| **Default batch** | `b = 1` in factory | N/A |

### Critique

#### TileGrid (Reduce) - Strengths

1. **Batch support**: Native `batches` field handles NC dimension
2. **Semantic naming**: `TileGrid` clearly implies a grid of tiles, not tile dimensions
3. **Consistent factory**: `of()` pattern matches `ReduceCBs::of()`
4. **Complete**: Handles all reduce patterns (scalar, row, col) with batches

#### TileGrid (Reduce) - Weaknesses

1. **Verbosity**: Always carries `batches` even when not needed
2. **Potential confusion**: `rows` means "number of tile rows", not "number of elements"

#### BinaryTileShape (Binary) - Strengths

1. **Simplicity**: Only 2 fields, no batch overhead
2. **Familiar naming**: `grid()` is intuitive for 2D specification

#### BinaryTileShape (Binary) - Weaknesses

1. **No batch support**: Can't express NC batches natively
2. **Name ambiguity**: "Shape" could be confused with tensor shape (elements, not tiles)
3. **Inconsistent factory**: Uses `grid()` vs reduce's `of()`

---

## TileLayout Comparison

Both implementations also have a layout struct for non-contiguous tile access patterns.

### Reduce Helpers: TileLayout

```cpp
struct TileLayout {
    uint32_t row_stride = 0;    // 0 = auto-detect from Wt
    uint32_t batch_stride = 0;  // 0 = auto-detect from Ht * row_stride

    explicit constexpr TileLayout() = default;
    constexpr TileLayout(uint32_t row, uint32_t batch) : row_stride(row), batch_stride(batch) {}

    // Factory methods
    static constexpr TileLayout contiguous() { return TileLayout(); }
    static constexpr TileLayout with_row_stride(uint32_t s) { return TileLayout(s, 0); }
    static constexpr TileLayout with_strides(uint32_t row, uint32_t batch) { return TileLayout(row, batch); }
};
```

**Usage:**
```cpp
// Contiguous layout (auto-detect strides)
reduce<SUM, REDUCE_ROW>(cbs, grid, TileLayout::contiguous());

// Custom row stride (e.g., tiles are spaced apart)
reduce<SUM, REDUCE_ROW>(cbs, grid, TileLayout::with_row_stride(input_stride));
```

### Binary Op Helpers: BinaryTileLayout

```cpp
struct BinaryTileLayout {
    uint32_t row_major_a = 1, row_major_b = 1;
};
```

**Usage:**
```cpp
// Default layout
binary_op<ADD>(icb_a, icb_b, ocb, shape, BinaryTileLayout{});
```

### Comparison Table

| Aspect | TileLayout (Reduce) | BinaryTileLayout (Binary) |
|--------|---------------------|---------------------------|
| **Fields** | `row_stride`, `batch_stride` | `row_major_a`, `row_major_b` |
| **Purpose** | Tile access pattern in CB | Per-operand layout flag |
| **Auto-detect** | Yes (0 = auto) | No |
| **Factory methods** | `contiguous()`, `with_row_stride()`, `with_strides()` | None |
| **Semantic richness** | High (describes memory layout) | Low (just a flag) |

### Critique

#### TileLayout (Reduce) - Strengths

1. **Rich semantics**: Describes actual memory layout (stride between rows/batches)
2. **Auto-detection**: `0` means "compute from grid dimensions"
3. **Factory methods**: `contiguous()` is self-documenting
4. **Explicit default**: Prevents accidental `{}` initialization

#### TileLayout (Reduce) - Weaknesses

1. **Complexity**: Two stride fields may be overkill for simple cases
2. **batch_stride unused**: Comment says "reserved for future use"

#### BinaryTileLayout (Binary) - Strengths

1. **Simplicity**: Just two flags
2. **Per-operand**: Can specify different layouts for A and B

#### BinaryTileLayout (Binary) - Weaknesses

1. **Unclear semantics**: What does `row_major_a = 1` mean exactly?
2. **No factory methods**: Less self-documenting
3. **No stride support**: Can't handle non-contiguous memory layouts
4. **Minimal functionality**: Essentially just two bools

---

## Recommendations for Unified TileGrid/TileShape Design

### Proposed Unified TileGrid

```cpp
struct TileGrid {
    uint32_t rows;
    uint32_t cols;
    uint32_t batches = 1;  // Default to 1 for binary ops that don't need batches

    // Primary factory - consistent naming
    static constexpr TileGrid of(uint32_t r, uint32_t c, uint32_t b = 1) { return {r, c, b}; }

    // Convenience factories
    static constexpr TileGrid single() { return {1, 1, 1}; }
    static constexpr TileGrid row(uint32_t c, uint32_t b = 1) { return {1, c, b}; }
    static constexpr TileGrid col(uint32_t r, uint32_t b = 1) { return {r, 1, b}; }

    // Total tile count helpers
    constexpr uint32_t total() const { return rows * cols * batches; }
    constexpr uint32_t per_batch() const { return rows * cols; }
};
```

**Benefits:**
1. Keeps batch support from reduce helpers
2. Uses `of()` pattern consistently
3. Adds helper methods for common calculations
4. Works for both reduce and binary ops (batches defaults to 1)

### Proposed Unified TileLayout

```cpp
struct TileLayout {
    uint32_t row_stride = 0;    // 0 = auto (contiguous)
    uint32_t batch_stride = 0;  // 0 = auto

    // Explicit default prevents accidental {} use
    explicit constexpr TileLayout() = default;

    // Factory methods
    static constexpr TileLayout contiguous() { return TileLayout{}; }
    static constexpr TileLayout with_row_stride(uint32_t s) {
        TileLayout l; l.row_stride = s; return l;
    }
    static constexpr TileLayout with_strides(uint32_t row, uint32_t batch) {
        TileLayout l; l.row_stride = row; l.batch_stride = batch; return l;
    }

    // Compute effective stride given grid dimensions
    constexpr uint32_t effective_row_stride(uint32_t Wt) const {
        return row_stride > 0 ? row_stride : Wt;
    }
};
```

**Benefits:**
1. Keeps rich semantics from reduce helpers
2. Auto-detection with 0 sentinel value
3. Self-documenting factory methods
4. Helper for computing effective stride

### Summary

| Component | Recommendation |
|-----------|----------------|
| **Grid struct** | Use `TileGrid` with batches (from reduce) |
| **Primary factory** | Use `of()` pattern (from reduce) |
| **Layout struct** | Use `TileLayout` with strides (from reduce) |
| **Factory methods** | Keep `contiguous()`, `with_row_stride()` (from reduce) |

The reduce helpers have a more complete and semantic design for both TileGrid and TileLayout. The binary helpers' versions are simpler but less capable. The unified design should adopt the reduce helpers' approach while ensuring it works seamlessly for binary operations (where batches may be unused).

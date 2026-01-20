# Kernel Helper Library API Analysis

> **Analysis Date:** 2025-01-20
> **Scope:** All files in `ttnn/cpp/ttnn/kernel_lib/`
> **Purpose:** Evaluate API clarity, consistency, and agent usability

---

## Executive Summary

The kernel helper library provides a powerful abstraction layer for compute kernel operations. After analyzing all 6 files and examining 50+ real-world usages across the codebase, this document provides detailed feedback on API clarity, consistency, and usability from an agent's perspective.

**Key Findings:**
- The library significantly reduces code complexity (reduce kernels: ~30 lines → ~10 lines)
- Input mode naming is mostly clear, with some room for improvement
- CB page production/consumption is implicit and requires implementation tracing
- Cross-helper inconsistencies exist (CB passing style, Shape types)
- Reader/writer kernel coordination is not documented

---

## 1. File-by-File Analysis

### 1.1 `common_types.hpp` ✅ Excellent

**Purpose:** Shared types for zero-overhead abstractions

**Contents:**
```cpp
struct NoAccumulation {};  // Tag type - accumulation code eliminated at compile-time
struct NoOp {              // No-op functor - compiles away completely
    ALWI void operator()(uint32_t = 0) const {}
};
```

**API Clarity:** Trivially usable. Documentation explains compile-time elimination behavior.

**Agent Assessment:** No issues. Clear purpose and zero cognitive overhead.

---

### 1.2 `dest_helpers.hpp` ✅ Excellent

**Purpose:** Auto-detect DEST register capacity based on JIT-generated headers

**Key APIs:**
| API | Description |
|-----|-------------|
| `get_dest_limit()` | Returns capacity based on sync/accum mode |
| `DEST_AUTO_LIMIT` | Compile-time constant for DEST capacity |
| `get_fp32_dest_acc_enabled()` | Detect FP32 accumulation mode |
| `get_dst_full_sync_enabled()` | Detect full-sync vs half-sync mode |

**Capacity Table (well-documented in header):**
| Sync Mode | Accum Mode | Capacity |
|-----------|------------|----------|
| SyncFull | 16-bit | 16 tiles |
| SyncFull | 32-bit | 8 tiles |
| SyncHalf | 16-bit | 8 tiles |
| SyncHalf | 32-bit | 4 tiles |

**Agent Assessment:** An agent can easily use `DEST_AUTO_LIMIT` without understanding JIT internals. The capacity table is invaluable for understanding constraints.

---

### 1.3 `tilize_helpers.hpp` ✅ Good with Minor Issues

**Purpose:** Unified tilize function for all patterns (simple, activation, fast, DT)

**Function Signature:**
```cpp
template <bool init = true, bool uninit = true, bool use_fast = false,
          bool use_dt = false, bool skip_wait = false>
ALWI void tilize(
    uint32_t icb,           // Input CB
    uint32_t block_w,       // Block width in tiles
    uint32_t ocb,           // Output CB
    uint32_t num_blocks,    // Number of blocks to process
    uint32_t subblock_h = 1,    // Height of each subblock
    uint32_t old_icb = 0,       // Previous CB for DT tracking
    uint32_t input_count = 0,   // Override for asymmetric patterns
    uint32_t total_rows = 0);   // For variable row alignment
```

**Template Parameters:**
| Parameter | Purpose | Clarity |
|-----------|---------|---------|
| `init` | Call tilize_init before processing | ✅ Clear |
| `uninit` | Call tilize_uninit after processing | ✅ Clear |
| `use_fast` | Use fast_tilize_* functions | ✅ Clear |
| `use_dt` | Use DT-aware init/uninit | ⚠️ "DT" not expanded |
| `skip_wait` | Skip cb_wait_front in loop | ✅ Clear |

**CB Pages Produced/Consumed:**
```
Standard Pattern (total_rows = 0, input_count = 0):
  Per iteration (num_blocks * subblock_h total):
    Input (icb):  Wait block_w tiles, Pop block_w tiles
    Output (ocb): Reserve block_w tiles, Push block_w tiles

Asymmetric Pattern (input_count > 0):
  Per iteration:
    Input (icb):  Wait input_count tiles, Pop input_count tiles
    Output (ocb): Reserve block_w tiles, Push block_w tiles

Variable Row Pattern (total_rows > 0):
  Per iteration:
    Input (icb):  Wait min(rows_left, 32) tiles, Pop same
    Output (ocb): Reserve block_w tiles, Push block_w tiles
```

**Clarity Issues:**
| Issue | Severity | Description |
|-------|----------|-------------|
| `total_rows` purpose | ⚠️ Medium | "Variable row alignment" not explained clearly |
| `input_count` purpose | ⚠️ Medium | "Asymmetric patterns" needs more context |
| `use_dt` abbreviation | ⚠️ Low | "DT" should be expanded to "Data Type" |

**Real Usage Examples from Codebase:**

```cpp
// Simple - very clear (tilize.cpp)
compute_kernel_lib::tilize(cb_id_in0, per_core_block_tile_cnt, cb_id_out0, per_core_block_cnt);

// With skip_wait - moderately clear (groupnorm.cpp)
compute_kernel_lib::tilize<true, true, false, false, true>(cb_in_rm, per_core_N, cb_in, per_core_M);

// Activation pattern (rotary_embedding.cpp)
compute_kernel_lib::tilize(in0_cb, num_tiles, out_cb, 1);
```

**Agent Usability:** An agent would easily use simple cases but struggle to determine when to use `subblock_h > 1` vs `total_rows > 0`. The examples cover basic cases well but don't explain advanced parameters.

---

### 1.4 `untilize_helpers.hpp` ✅ Good with Minor Issues

**Purpose:** Unified untilize with automatic dispatch (pack_untilize vs standard)

**Function Signature:**
```cpp
template <uint32_t tile_width,      // Width in tiles (compile-time!)
          uint32_t icb_id,          // Input CB (compile-time!)
          uint32_t ocb_id,          // Output CB (compile-time!)
          bool init = true,
          bool uninit = true,
          bool wait_upfront = false>
ALWI void untilize(uint32_t num_rows, uint32_t block_rt_dim = 1, uint32_t total_tiles = 0);
```

**Automatic Dispatch Logic:**
```
If wait_upfront=true OR (tile_width > DEST AND non-integer):
  → STANDARD UNTILIZE PATH (fallback for floats)

If tile_width > DEST AND integer type:
  → BLOCK-BASED PACK UNTILIZE (hardware-accelerated, multi-pass)

If tile_width ≤ DEST:
  → PACK UNTILIZE (hardware-accelerated, single-pass, preferred)
```

**CB Pages Produced/Consumed:**

| Path | Input CB | Output CB |
|------|----------|-----------|
| **Pack (single-pass)** | Wait/pop `tile_width * block_rt_dim` per row | Reserve/push same per row |
| **Pack (block-based)** | Wait/pop `block_width` per block within row | Reserve `tile_width` once, push after all blocks |
| **Standard** | Wait/pop `tile_width` per row | Reserve/push `tile_width` per row |
| **Wait-upfront** | Wait all upfront, pop per row | Reserve/push per row |

**Critical Inconsistency:**
```cpp
// untilize: CB IDs are TEMPLATE parameters (compile-time required)
compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(num_rows);

// tilize: CB IDs are RUNTIME parameters
compute_kernel_lib::tilize(cb_in, block_w, cb_out, num_blocks);
```

**Reason for Template CBs:** The `is_integer_format<icb_id>()` function requires compile-time CB ID to check JIT-generated `unpack_dst_format` array.

**Agent Usability:** Moderately clear. The dispatch logic is well-documented, but the compile-time CB requirement is buried in implementation details and not explained in the header comment.

---

### 1.5 `reduce_helpers.hpp` ⚠️ Good but Complex

**Purpose:** Unified reduce for ROW, COL, and SCALAR reductions

**Function Signature:**
```cpp
template <PoolType reduce_type,           // SUM, AVG, MAX (required)
          ReduceDim reduce_dim,           // REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR (required)
          ReduceInputMode input_mode = ReduceInputMode::STREAMING,
          ReduceDataFormatReconfig reconfig = ReduceDataFormatReconfig::BOTH,
          bool init = true,
          bool uninit = true,
          typename AccumT = NoAccumulation,
          typename PostReduceOp = NoOp>
ALWI void reduce(
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t ocb,
    TileShape shape,
    TileLayout layout = {},
    AccumT accum = {},
    PostReduceOp post_reduce_op = {});
```

**Input Mode Semantics:**

| Mode | Wait Behavior | Pop Behavior | Use Case |
|------|---------------|--------------|----------|
| `STREAMING` | Wait 1 tile at a time | Pop 1 tile at a time | Safe default, any CB size |
| `STREAMING_BATCHED` | Wait all tiles in row/batch | Pop all after processing | Optimal when tiles pre-loaded |
| `PRELOADED` | Caller manages wait | Caller manages pop | Asymmetric wait/pop patterns |
| `PERSISTENT` | Wait all upfront | NO pop (tiles persist) | Softmax pattern, tile reuse |

**CB Pages Produced/Consumed by Reduce Dimension:**

| Dim | Input Tiles Consumed | Output Tiles Produced |
|-----|---------------------|----------------------|
| `REDUCE_SCALAR` | Ht × Wt × batches | 1 per batch |
| `REDUCE_ROW` | Ht × Wt × batches | Ht per batch |
| `REDUCE_COL` | Ht × Wt × batches | Wt per batch |

**Mode-Specific CB Behavior:**

```
STREAMING:
  Input:  cb_wait_front(icb, 1) per tile, cb_pop_front(icb, 1) per tile
  Output: cb_reserve_back(ocb, 1), cb_push_back(ocb, 1) per output tile

STREAMING_BATCHED:
  REDUCE_ROW: Wait Wt tiles, process row, pop Wt tiles
  REDUCE_COL: Wait Ht*chunk_size tiles, process chunk, pop same
  REDUCE_SCALAR: Wait Ht*Wt tiles, process batch, pop same
  Output: Reserve/push 1 per output tile

PRELOADED:
  Input:  No wait/pop (caller manages externally)
  Output: Bulk reserve upfront, bulk push at end

PERSISTENT:
  Input:  Wait all upfront, NO pop (tiles remain for subsequent ops)
  Output: Reserve/push 1 per output tile
```

**TileShape Factory Methods:**
```cpp
TileShape::grid(rows, cols, batches = 1)  // Full grid
TileShape::single()                        // 1×1×1
TileShape::row(cols, batches = 1)          // 1×cols×batches
TileShape::col(rows, batches = 1)          // rows×1×batches
```

**Mode Naming Clarity:**
| Mode Name | Clarity | Potential Confusion |
|-----------|---------|---------------------|
| `STREAMING` | ✅ Clear | None |
| `STREAMING_BATCHED` | ✅ Clear | None |
| `PRELOADED` | ⚠️ Medium | Doesn't convey "caller manages CB" |
| `PERSISTENT` | ⚠️ Medium | Sounds like cross-call persistence, actually means "no pop" |

**Real Usage Examples:**

```cpp
// Simple scalar reduction (reduce_hw.cpp)
compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM,
    compute_kernel_lib::ReduceInputMode::STREAMING,
    compute_kernel_lib::ReduceDataFormatReconfig::NONE>(
    tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3,
    compute_kernel_lib::TileShape::grid(Ht, Wt, NC));

// PERSISTENT with PostOp for softmax (softmax.cpp)
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputMode::PERSISTENT,
    compute_kernel_lib::ReduceDataFormatReconfig::INPUT>(
    cb_in, cb_bcast_scaler, cb_max, compute_kernel_lib::TileShape::row(Wt));

// PERSISTENT with lambda PostOp (softmax.cpp)
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputMode::PERSISTENT>(
    cb_exps, cb_bcast_scaler, cb_recipsumexps,
    compute_kernel_lib::TileShape::row(Wt),
    {}, {},
    [](uint32_t) { recip_tile_init(); recip_tile(0); });

// PRELOADED for partial reduction (groupnorm.cpp)
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR,
    compute_kernel_lib::ReduceInputMode::PRELOADED,
    compute_kernel_lib::ReduceDataFormatReconfig::NONE>(
    cb_x, cb_scaler, cb_ex_partial,
    compute_kernel_lib::TileShape::grid(out_block_h_actual, block_w));
```

**Agent Usability:** An agent can use basic cases effectively. The main challenges are:
1. Understanding when to use PRELOADED vs PERSISTENT
2. Coordinating with reader kernels for non-STREAMING modes
3. Understanding REDUCE_COL chunking behavior (auto but undocumented for reader)

---

### 1.6 `binary_op_helpers.hpp` ⚠️ Good but Verbose

**Purpose:** Unified add/sub/mul/square with broadcast support

**Main Functions:**
```cpp
// Generic binary operation
template <BinaryOpType op_type, BroadcastDim bcast_dim = NONE,
          BinaryInputMode input_mode = STREAMING, ...>
void binary_op(icb_a, icb_b, ocb, shape, layout, accum, post_op);

// Convenience aliases
void add(icb_a, icb_b, ocb, shape, ...);   // A + B
void sub(icb_a, icb_b, ocb, shape, ...);   // A - B
void mul(icb_a, icb_b, ocb, shape, ...);   // A × B
void square(icb_a, ocb, shape, ...);       // A × A (single input)
```

**Broadcast Dimension Semantics:**

| BroadcastDim | B Shape | Operation | B Tiles Needed |
|--------------|---------|-----------|----------------|
| `NONE` | Same as A | C[h,w] = A[h,w] op B[h,w] | Ht × Wt |
| `ROW` | 1 × Wt | C[h,w] = A[h,w] op B[w] | Wt (persists) |
| `COL` | Ht × 1 | C[h,w] = A[h,w] op B[h] | Ht (1 per row) |
| `SCALAR` | 1 × 1 | C[h,w] = A[h,w] op B[0,0] | 1 (persists) |

**Input Mode Semantics (same as reduce):**

| Mode | A Tiles | B Tiles | Output |
|------|---------|---------|--------|
| `STREAMING` | Wait/pop 1 at a time | Varies by broadcast | Reserve/push 1 at a time |
| `STREAMING_BATCHED` | Wait chunk, pop chunk | Varies | Reserve/push chunk |
| `PRELOADED` | Caller manages | Caller manages | Bulk reserve/push |
| `PERSISTENT` | Wait all, no pop | Varies | Reserve/push per tile |

**Critical B-Input Persistence Rules:**

```
BroadcastDim::ROW:
  - B tiles are waited once at start (Wt tiles)
  - B tiles are NOT popped - persist for all rows
  - Caller may need B for subsequent operations

BroadcastDim::COL:
  - B tiles are waited 1 per row
  - B tiles ARE popped after each row

BroadcastDim::SCALAR:
  - B tile is waited once at start (1 tile)
  - B tile is NOT popped - persists for entire operation

BroadcastDim::NONE:
  - B tiles follow same pattern as A tiles
```

**CB Pages Produced/Consumed (STREAMING + NONE):**
```
Per tile (Ht × Wt total):
  Input A:  Wait 1, Pop 1
  Input B:  Wait 1, Pop 1
  Output:   Reserve 1, Push 1
```

**Real Usage Examples:**

```cpp
// Simple streaming add (rotary_embedding.cpp)
compute_kernel_lib::add(
    cos_interm_cb, sin_interm_cb, out_cb,
    compute_kernel_lib::BinaryTileShape::single());

// Scalar broadcast multiply (rotary_embedding.cpp)
compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::SCALAR,
    compute_kernel_lib::BinaryInputMode::STREAMING>(
    rotated_in_cb, scalar_cb, rotated_in_interm_cb,
    compute_kernel_lib::BinaryTileShape::single());

// Simple mul/add wrappers (ssm_prefix_scan.cpp)
FORCE_INLINE void mul(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    compute_kernel_lib::mul(cb_a, cb_b, cb_out,
        compute_kernel_lib::BinaryTileShape::single());
}
```

**Agent Usability:** The API is verbose but clear. An agent would find it easy to use for simple cases. The broadcast CB persistence rules need to be memorized or looked up.

---

## 2. Consistency Analysis

### 2.1 Internal Consistency (within each helper)

| Helper | Consistency | Notes |
|--------|-------------|-------|
| tilize | ⚠️ Medium | `total_rows` and `input_count` add complexity without clear examples |
| untilize | ✅ High | Three paths (pack, block-pack, standard) are clearly delineated |
| reduce | ✅ High | All modes follow same pattern with clear differences |
| binary | ✅ High | All broadcast dims have consistent init/exec/reload pattern |

### 2.2 Cross-Helper Consistency

| Aspect | tilize | untilize | reduce | binary |
|--------|--------|----------|--------|--------|
| Namespace | ✅ `compute_kernel_lib` | ✅ `compute_kernel_lib` | ✅ `compute_kernel_lib` | ✅ `compute_kernel_lib` |
| CB passing | Runtime | **Template** | Runtime | Runtime |
| Input modes | N/A | N/A | 4 modes | 4 modes (same names) |
| Shape struct | N/A | N/A | `TileShape` | `BinaryTileShape` |
| Layout struct | N/A | N/A | `TileLayout` | `BinaryTileLayout` |
| init/uninit flags | ✅ bool templates | ✅ bool templates | ✅ bool templates | ✅ bool templates |
| Accum support | N/A | N/A | `Accumulate` | `BinaryAccumulate` |
| PostOp support | N/A | N/A | ✅ Lambda | ✅ Lambda |

**Key Inconsistencies:**

1. **CB ID Passing Style:**
   - `untilize` requires compile-time CB IDs (for data format detection)
   - All others accept runtime CB IDs
   - **Impact:** Agent must know this exception

2. **Shape Structs Are Duplicated:**
   - `TileShape` (reduce) and `BinaryTileShape` (binary) are semantically identical
   - Both have: `grid()`, `single()`, `row()`, `col()`
   - **Impact:** Unnecessary cognitive load

3. **Accumulate Types Are Duplicated:**
   - `Accumulate` and `BinaryAccumulate` are nearly identical
   - Both have: `cb_accumulator`, `dst_index`, `iteration`, `is_first()`
   - **Impact:** Code duplication, potential for divergence

---

## 3. Agent Usability Assessment

### 3.1 Key Questions an Agent Might Ask

| Question | Answer Availability |
|----------|---------------------|
| "How many tiles does tilize consume from input CB?" | ⚠️ Partial - depends on `input_count` parameter |
| "How many tiles does reduce produce?" | ✅ Clear - documented by reduce_dim |
| "When should I use PRELOADED vs STREAMING?" | ⚠️ Implicit - need to understand wait/pop ownership |
| "What CB synchronization does the reader need?" | ❌ Not documented |
| "Does the B input get popped in row broadcast?" | ⚠️ Buried in implementation |

### 3.2 Decision Trees for Agent Usage

**Choosing Tilize Parameters:**
```
Is input already aligned to tiles?
  └─ Yes: tilize(icb, block_w, ocb, num_blocks)
  └─ No, rows vary per iteration: Use total_rows parameter
  └─ No, input count differs from output: Use input_count parameter

Need fast tilize?
  └─ Yes: tilize<true, true, true>(...)

Switching data types mid-kernel?
  └─ Yes: tilize<true, true, false, true>(..., old_icb)
```

**Choosing Reduce Input Mode:**
```
Are tiles streamed one-at-a-time from reader?
  └─ Yes: STREAMING (safest, default)

Are all tiles pre-loaded in CB before compute starts?
  └─ Yes, and I need to pop after: PRELOADED
  └─ Yes, and I need to keep for reuse: PERSISTENT

Is reader sending batches?
  └─ Yes: STREAMING_BATCHED (optimal performance)
```

**Choosing Binary Broadcast Dimension:**
```
Is second operand a single tile applied to all?
  └─ Yes: BroadcastDim::SCALAR

Is second operand a single row (Wt tiles) applied to all rows?
  └─ Yes: BroadcastDim::ROW

Is second operand a single column (Ht tiles) applied to all columns?
  └─ Yes: BroadcastDim::COL

Both operands have same shape?
  └─ Yes: BroadcastDim::NONE (default)
```

### 3.3 Information Gaps for Agent

1. **Reader/Writer Coordination:**
   - No documentation on what reader/writer kernels must do for each mode
   - Example: REDUCE_COL in STREAMING mode requires column-major tile order from reader

2. **CB Sizing Requirements:**
   - PRELOADED mode requires CB to hold all tiles - not explicitly stated
   - PERSISTENT mode requires input CB to remain valid after reduce completes

3. **DEST Chunking for REDUCE_COL:**
   - Automatic in compute kernel
   - Reader must know chunk size to send tiles in correct order
   - `DEST_AUTO_LIMIT` should be documented as shared constant

4. **B-Input Persistence Rules:**
   - Critical for correct operation
   - Buried in implementation, not in header documentation

---

## 4. CB Contract Summary

### 4.1 Tilize CB Contract

```
Inputs Required:
  - icb must have tiles available before tilize() is called (unless skip_wait=true)

Wait/Pop Per Iteration:
  - Standard: block_w tiles
  - Asymmetric: input_count tiles
  - Variable: min(rows_left, 32) tiles

Reserve/Push Per Iteration:
  - Always: block_w tiles

Total Iterations:
  - num_blocks × subblock_h
```

### 4.2 Untilize CB Contract

```
Inputs Required:
  - icb must have tiles available before untilize() is called
  - Wait-upfront mode: all tiles at once
  - Standard mode: tile_width per row

Wait/Pop Per Row:
  - Pack path: tile_width × block_rt_dim
  - Block-based: block_width per block, tile_width total
  - Standard: tile_width

Reserve/Push Per Row:
  - All paths: tile_width × block_rt_dim (or tile_width for standard)
```

### 4.3 Reduce CB Contract

```
Scaler CB:
  - Must have 1 tile available before reduce() is called
  - Library waits for it internally when init=true
  - NOT popped - remains available after reduce()

Input CB by Mode:
  - STREAMING: 1 tile waited/popped per reduce_tile call
  - STREAMING_BATCHED: Row/batch waited at once, popped after
  - PRELOADED: No wait/pop (caller manages externally)
  - PERSISTENT: All waited upfront, NO pop

Output CB:
  - STREAMING/STREAMING_BATCHED/PERSISTENT: 1 tile reserved/pushed per output
  - PRELOADED: Bulk reserve at start, bulk push at end

Output Tile Count:
  - REDUCE_SCALAR: 1 per batch
  - REDUCE_ROW: Ht per batch
  - REDUCE_COL: Wt per batch
```

### 4.4 Binary CB Contract

```
Input A CB by Mode:
  - STREAMING: 1 tile waited/popped per tile
  - STREAMING_BATCHED: Chunk waited, chunk popped
  - PRELOADED: No wait/pop (caller manages)
  - PERSISTENT: All waited upfront, NO pop

Input B CB by Broadcast:
  - NONE: Same as A
  - ROW: Wt tiles waited once, NOT popped (persists for reuse)
  - COL: 1 tile waited/popped per row
  - SCALAR: 1 tile waited once, NOT popped (persists for reuse)

Output CB:
  - STREAMING: 1 tile reserved/pushed per tile
  - STREAMING_BATCHED: Chunk reserved/pushed
  - PRELOADED: Bulk reserve at start, bulk push at end
  - PERSISTENT: 1 tile reserved/pushed per tile
```

---

## 5. Recommendations

### 5.1 High Priority

1. **Document Reader/Writer Coordination:**
   Add a section or companion document showing what dataflow kernels must do for each mode.

   ```cpp
   // Example: REDUCE_COL + STREAMING mode
   // Reader must send tiles in column-major chunks:
   //   For wt in [0, Wt) step chunk_size:
   //     For ht in [0, Ht):
   //       For i in [wt, min(wt+chunk_size, Wt)):
   //         Send tile[ht][i]
   ```

2. **Unify or Document CB Passing:**
   Either make all helpers use runtime CBs, or prominently document why untilize needs compile-time CBs.

   ```cpp
   // Add to untilize_helpers.hpp header:
   // NOTE: CB IDs must be compile-time constants because is_integer_format<icb_id>()
   // reads from JIT-generated unpack_dst_format array at compile time.
   ```

3. **Add CB Contract Comments:**
   ```cpp
   // @cb_contract icb: STREAMING waits/pops 1 per tile, PRELOADED expects all tiles present
   // @cb_contract ocb: Always reserves/pushes 1 per output tile
   // @cb_contract icb_scaler: Must have 1 tile, NOT popped
   ```

4. **Merge Shape/Layout Structs:**
   `TileShape` and `BinaryTileShape` should be unified (or one should alias the other).

### 5.2 Medium Priority

1. **Rename PERSISTENT to More Descriptive Name:**
   ```cpp
   // Current: ambiguous
   ReduceInputMode::PERSISTENT

   // Better: clearly describes behavior
   ReduceInputMode::NO_POP
   // or
   ReduceInputMode::REUSABLE
   ```

2. **Add Mode Decision Helper Function:**
   ```cpp
   constexpr ReduceInputMode recommended_mode(
       bool caller_manages_wait,
       bool caller_manages_pop,
       bool tiles_reused_after) {
       if (!caller_manages_wait && !caller_manages_pop) return STREAMING;
       if (!caller_manages_wait && caller_manages_pop) return STREAMING_BATCHED;
       if (caller_manages_wait && caller_manages_pop) return PRELOADED;
       if (caller_manages_wait && tiles_reused_after) return PERSISTENT;
       // ...
   }
   ```

3. **Document B-Input Persistence in Header:**
   ```cpp
   // In binary_op_helpers.hpp header comment:
   // IMPORTANT: B-input CB behavior varies by broadcast dimension:
   // - ROW: B tiles waited once, NOT popped (caller may reuse)
   // - COL: B tiles popped after each row
   // - SCALAR: B tile waited once, NOT popped (caller may reuse)
   // - NONE: B tiles follow same wait/pop as A tiles
   ```

### 5.3 Low Priority

1. **Expand "DT" Abbreviation:**
   Change `use_dt` to `use_datatype_reconfig` or add comment explaining DT = Data Type.

2. **Consider Deprecating Obscure Parameters:**
   If `total_rows` and `input_count` in tilize are rarely used, consider making them a separate advanced API.

3. **Add Compile-Time Validation:**
   Use `static_assert` for impossible combinations (e.g., STREAMING + layout.row_stride).

---

## 6. Summary Ratings

| Helper | API Clarity | Examples Quality | CB Documentation | Agent Usability |
|--------|-------------|------------------|------------------|-----------------|
| common_types | ⭐⭐⭐⭐⭐ | N/A | N/A | Excellent |
| dest_helpers | ⭐⭐⭐⭐⭐ | N/A | N/A | Excellent |
| tilize | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐☆ | ⭐⭐☆☆☆ | Good for simple cases |
| untilize | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | ⭐⭐⭐☆☆ | Good |
| reduce | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐☆☆ | Good after learning modes |
| binary | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | Good after learning broadcast |

**Overall Assessment:**

The kernel helper library is well-designed and significantly reduces code complexity. Real-world evidence shows:
- `reduce_hw.cpp`: ~30 lines → ~10 lines
- `tilize.cpp`: ~20 lines → ~4 lines
- Softmax kernel: Complex manual reduce → clean library call with PostOp lambda

An agent can use the library effectively for standard patterns after understanding the input mode semantics. The main gaps are:

1. **Reader/writer coordination** is implicit
2. **CB page counts** require tracing through implementation
3. **Cross-helper inconsistencies** add cognitive load
4. **Advanced parameters** lack clear use-case documentation

The library would benefit most from a companion "coordination guide" showing complete reader-compute-writer patterns for each mode combination.

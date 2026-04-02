# Kernel Helper Library — Coding Conventions

Reference for writing `compute_kernel_lib` helpers. All helpers live in `ttnn/cpp/ttnn/kernel_lib/`, use the `compute_kernel_lib` namespace, and follow a header-only `.hpp` (declarations) + `.inl` (implementation) split.

## Existing Helpers

| Helper | What it unifies | Key patterns |
|---|---|---|
| `sfpu_helpers.hpp` | SFPU unary/binary/ternary ops + chains | CRTP op bases, CompactLoad, chain transformation, pipeline batching |
| `tilize_helpers.hpp` | tilize/fast_tilize init/block/uninit | Config enums, CB indices as template params, auto fast_tilize dispatch |
| `untilize_helpers.hpp` | untilize/pack_untilize | Auto-dispatches pack_untilize vs standard based on width/datatype |
| `reduce_helpers_compute.hpp` | reduce ROW/COL/SCALAR with SUM/AVG/MAX | InputPolicy enum, Accumulate type trait, PostReduceOp callback, auto DEST limit detection |
| `dest_helpers.hpp` | DEST register capacity detection | constexpr functions, JIT header integration, sync/accum mode auto-detect |
| `binary_op_helpers.hpp` | add/sub/mul with broadcast | BroadcastDim, input/output policies, DEST chunking, post-op callback |
| `common_types.hpp` | Shared types | `NoOp`, `NoAccumulation` |
| `cb_helpers.hpp` | CB query utilities | tile size, page count, validation |

## 1. File structure

```
ttnn/cpp/ttnn/kernel_lib/
  {name}_helpers.hpp      <- declarations, enums, structs, doc comments, examples
  {name}_helpers.inl      <- implementation, #include'd at bottom of .hpp
```

The `.hpp` contains ALL documentation. The `.inl` contains only implementation code.

## 2. Namespace and includes

```cpp
// .hpp
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"

namespace compute_kernel_lib {
// declarations
}
#include "{name}_helpers.inl"

// .inl
#pragma once
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
// ... operation-specific LLK includes
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {
// implementations
}
```

## 3. Policy enums

Each helper defines its own policy enums:

- **InputPolicy**: Controls when to wait for input tiles and whether to pop them
  - `WaitAndPopPerTile` (default), `WaitUpfrontNoPop`, `NoWaitNoPop`, etc.
- **OutputPolicy**: Controls when to reserve/push output tiles
  - `PerTile` (default), `Bulk`, `PerChunk`
- **DataFormatReconfig**: Controls unpacker/packer reconfiguration
  - `NONE`, `INPUT`, `OUTPUT`, `INPUT_AND_OUTPUT` (default)

Only add variants the use cases require.

## 4. DEST register management

Use `DEST_AUTO_LIMIT` from `dest_helpers.hpp`:

```cpp
constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
for (uint32_t base = 0; base < total; base += dest_limit) {
    uint32_t chunk = min(dest_limit, total - base);
    tile_regs_acquire();
    // process chunk tiles into DEST[0..chunk-1]
    tile_regs_commit();
    tile_regs_wait();
    // pack chunk tiles from DEST
    tile_regs_release();
}
```

## 5. Op-type structs (CRTP base + init/call)

Use lightweight structs inheriting from a CRTP base. The base provides `exec()` (offset arithmetic), `apply()` (init+exec), slot constants, and static assertions. The derived struct only defines `init()` and `call()`.

```cpp
// Base (provided by infrastructure):
template <typename Derived, Dst Slot>
struct UnaryOp {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;
    static_assert(dst_idx < 8, "DEST slot exceeds capacity");
    ALWI void exec(uint32_t offset = 0) const {
        static_cast<const Derived*>(this)->call(dst_idx + offset);
    }
    ALWI void apply(uint32_t offset = 0) const {
        static_cast<const Derived*>(this)->init(); exec(offset);
    }
};

// Derived (what the op author writes):
template <Dst Slot = Dst::D0>
struct Sin : UnaryOp<Sin<Slot>, Slot> {
    ALWI void init() const;           // *_tile_init()
    ALWI void call(uint32_t d0) const; // *_tile(d0) -- offset already applied
};
```

Similarly `BinaryOp<Derived, In0, In1, Out>` and `TernaryOp<Derived, In0, In1, In2, Out>`.

## 6. Compile-time chain transformation

When a helper supports composable chains, the chain factory function performs compile-time transformation:

1. **Compact adjacent same-resource operations**: Multiple operations consuming from the same source are merged into a single compound element whose `exec()` handles the shared resource lifecycle (wait before first, pop after last).

2. **Annotate resource lifecycle**: Determine at compile time whether each compound element should wait/acquire and release/pop, based on whether the same resource appears elsewhere in the chain.

3. **Validate constraints**: `static_assert` on illegal patterns (e.g., same resource in multiple non-adjacent groups).

All chain elements expose the same `init()`/`exec()`/`apply()` interface.

## 7. PostOp callbacks

```cpp
template <typename PostOp = NoOp>
void main_func(..., PostOp post_op = {});
// Called as: post_op(dst_idx);
```

`NoOp` from `common_types.hpp` compiles away entirely.

## 8. Function signatures

- Use `ALWI` (always inline) on all functions
- Template params: operation config first, then policies, then PostOp/Accumulate
- Function params: input CBs, output CB, shape/count, then optional PostOp/Accumulate

## 9. LLK Sequence Rules

**Critical**: Each helper internally calls LLK init and exec functions. These sequences MUST match patterns found in existing kernels.

- Each `*_tile_init()` must immediately precede its corresponding `*_tile()` calls
- Some inits are mutually exclusive (reconfigure the same hardware)
- After a disruptive init, subsequent operations may need re-initialization
- When in doubt, verify against an existing kernel with the same call pattern

## 10. Performance testing methodology

Every helper MUST have a performance comparison against raw LLK code:

1. **Always benchmark against hand-written raw LLK baseline**
2. **Use min of trimmed runs**, not average
3. **Test across a range of tile counts** (powers of 2, 8 to 32K+)
4. **Test the full complexity spectrum**: single ops, multi-op chains, multi-slot loads
5. **Warmup runs mandatory** (3+ before measurement)
6. **When numbers are ambiguous, disassemble the ELF** (`riscv-tt-elf-objdump` on trisc2)
7. **One parametric test, one table**: all workloads x all tile counts
8. **Thresholds**: <2% OK, 2-5% REVIEW, >5% BLOCKER

## Known Operation Categories

| Category | LLK Prefix(es) | Compute API Dir | Notes |
|---|---|---|---|
| Elementwise unary | `llk_math_eltwise_unary_sfpu` | `hw/inc/api/compute/eltwise_unary/` | Largest category |
| Binary eltwise | `llk_math_eltwise_binary` | `hw/inc/api/compute/eltwise_binary/` | |
| Ternary SFPU | `llk_math_eltwise_ternary_sfpu` | `hw/inc/api/compute/eltwise_unary/` | Filed under unary dir |
| Matmul | `llk_math_matmul` | `hw/inc/api/compute/matmul/` | |
| Reduce (FPU) | `llk_math_reduce` | `hw/inc/api/compute/reduce/` | |
| Tilize | `llk_math_fast_tilize` | `hw/inc/api/compute/tilize/` | |

## Migration Strategy

| Tier | Criteria | Action |
|------|----------|--------|
| **Tier 1** (Easy) | Direct 1-for-1 swap | Migrate first, proof-of-concept |
| **Tier 2** (Medium) | Minor restructuring needed | After Tier 1 verified |
| **Tier 3** (Hard) | Conditional dispatch, multi-CB, interleaved FPU+SFPU | Case-by-case, often better left manual |

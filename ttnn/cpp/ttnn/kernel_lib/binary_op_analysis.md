# Binary Operation Kernel Analysis

This document inventories raw `add_tiles` / `sub_tiles` / `mul_tiles` (and their
broadcast variants) usage across `ttnn/cpp/ttnn/operations/`, groups usages
into pattern categories, and evaluates migration feasibility against the
current `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp` API.

**Reference sources**

- API header: `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`
- Common types / `NoOp` / `NoAccumulation`: `ttnn/cpp/ttnn/kernel_lib/common_types.hpp`
- DEST chunk limit: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp` (`DEST_AUTO_LIMIT`)
- Overlapping legacy helper: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`

---

## Executive Summary

Ground-truth counts from the current tree (grep scope
`ttnn/cpp/ttnn/operations/`):

| Raw API                        | Files |
|--------------------------------|-------|
| `add_tiles(...)`               | 70    |
| `mul_tiles(...)`               | 46    |
| `sub_tiles(...)`               |  9    |
| `*_tiles_bcast_rows` / `<ROW>` | 40    |
| `*_tiles_bcast_cols`           | 34    |
| `*_tiles_bcast_scalar`         | 24    |
| `reconfig_data_format_srcb/a`  | 53 files (172 call sites) |
| `*_tiles_to_cb` (moreh_common) | 13    |

**Adoption state:** Zero production kernels currently call
`compute_kernel_lib::{add,sub,mul,square,binary_op}` from
`binary_op_helpers.hpp`. Verified via
`rg 'compute_kernel_lib::(add|sub|mul|square|binary_op)\s*[<(]'` under
`ttnn/cpp/ttnn/operations/` — the only hit is a feasibility markdown at
`normalization/layernorm/device/kernels/compute/layernorm_binary_helpers_feasibility.md`.
The previous version of this document claimed `rotary_embedding_llama.cpp`,
`softmax.cpp`, and `groupnorm.cpp` were already using the helper; that was
wrong.

**Key library gaps that block wider adoption**

1. No partial data-format reconfig (`_srcb` / `_srca`). 53 kernels / 172
   call sites rely on it; today the helper only supports
   `NONE | INPUT | OUTPUT | INPUT_AND_OUTPUT`.
2. `moreh_common.hpp` is a parallel abstraction covering ~13 moreh kernels;
   overlap needs to be addressed explicitly.
3. `PostOp` is strictly per-tile; no batch-level hook for vectorised
   multi-tile SFPU follow-ups.
4. `sub_bcast_rows` has no upstream `*_init_short` LLK helper (confirmed — no
   such symbol exists anywhere in the repo). Any `SUB<ROW>` caller must
   tolerate whatever init the helper emits.

**Migration feasibility estimate (re-verified against current tree):**

- ~40% of the 70 unique files map cleanly to the helper today (Tier 1).
- ~35% require the partial-reconfig extension OR acceptance of
  `reconfig = NONE` with manual reconfigs around the call (Tier 2).
- ~25% have conditional dispatch, multi-CB interleaving, or fused
  matmul/reduce patterns that make abstraction of low value (Tier 3).

---

## 1. Helper Library Capabilities

Canonical reference for the symbols used in every replacement snippet below.
All symbols live in `namespace compute_kernel_lib`.

### 1.1 Enums

| Enum                           | Values                                                              |
|--------------------------------|---------------------------------------------------------------------|
| `BinaryOpType`                 | `ADD`, `SUB`, `MUL`, `SQUARE`                                       |
| `BroadcastDim`                 | `NONE`, `ROW`, `COL`, `SCALAR`                                      |
| `BinaryDataFormatReconfig`     | `NONE`, `INPUT`, `OUTPUT`, `INPUT_AND_OUTPUT`                       |
| `BinaryInputPolicy`            | `WaitAndPopPerTile`, `WaitAndPopPerChunk`, `WaitUpfrontNoPop`, `WaitUpfrontPopAtEnd`, `NoWaitNoPop`, `NoWaitPopAtEnd` |
| `BinaryOutputPolicy`           | `PerTile`, `PerChunk`, `Bulk`                                       |

`BroadcastDim` semantics (directly from the header):

| `BroadcastDim` | B shape  | B tiles | What gets broadcast              |
|----------------|----------|---------|----------------------------------|
| `NONE`         | `[Ht,Wt]`| `Ht*Wt` | Full tensor                      |
| `ROW`          | `[1,Wt]` | `Wt`    | Single row replicated down rows  |
| `COL`          | `[Ht,1]` | `Ht`    | Single column replicated right   |
| `SCALAR`       | `[1,1]`  | `1`     | Single tile replicated everywhere|

### 1.2 Shapes and policies

```cpp
// Tile grid dimensions — pick one factory
BinaryInputBlockShape::of(r, c);   // [r, c] tiles
BinaryInputBlockShape::single();   // 1 x 1
BinaryInputBlockShape::row(c);     // 1 x c   (row vector)
BinaryInputBlockShape::col(r);     // r x 1   (column vector)

// Optional accumulation (reloads prior partial sum/product from CB)
BinaryAccumulate{cb_accumulator, /*dst_index=*/0};
NoAccumulation{};  // default (no reload)
```

### 1.3 Entry points

```cpp
// Must be called once before any helper op
binary_op_init_common(icb_a, icb_b, ocb);

// Low-level (explicit op_type)
binary_op<op_type, bcast_dim, input_a_policy, input_b_policy,
          output_policy, reconfig, init, PostOp, AccumT>(
    icb_a, icb_b, ocb, shape, post_op, accum);

// Aliases (op_type omitted; template param list is identical otherwise)
add<...>(icb_a, icb_b, ocb, shape, post_op, accum);
sub<...>(icb_a, icb_b, ocb, shape, post_op, accum);
mul<...>(icb_a, icb_b, ocb, shape, post_op, accum);

// square takes a single-CB input; no bcast_dim; no input_b_policy
square<input_policy, output_policy, reconfig, init, PostOp, AccumT>(
    icb, ocb, shape, post_op, accum);
```

Defaults (as of current header):

```
bcast_dim         = BroadcastDim::NONE
input_a_policy    = BinaryInputPolicy::WaitAndPopPerTile
input_b_policy    = input_a_policy                      // tracks A unless overridden
output_policy     = BinaryOutputPolicy::PerTile
reconfig          = BinaryDataFormatReconfig::INPUT_AND_OUTPUT
init              = true
PostOp            = NoOp      // from common_types.hpp, compiles away
AccumT            = NoAccumulation
```

DEST chunking is automatic via `DEST_AUTO_LIMIT`
(`ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`); the caller does not pick a
subblock size.

---

## 2. Pattern Categories

Eight patterns cover essentially every observed call site. Each entry below
shows the raw pattern and the verified replacement against the current API.

### Pattern 1 — Simple streaming (one tile at a time)

**Difficulty: EASY. Helper support: FULL.**

```cpp
for (uint32_t i = 0; i < N; ++i) {
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    add_tiles(cb_in0, cb_in1, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
    cb_push_back(cb_out, 1);
}
```

Replacement:

```cpp
using namespace compute_kernel_lib;

binary_op_init_common(cb_in0, cb_in1, cb_out);
add(cb_in0, cb_in1, cb_out, BinaryInputBlockShape::row(N));
// Defaults: BroadcastDim::NONE, WaitAndPopPerTile, PerTile, INPUT_AND_OUTPUT.
```

Representative files (ground-truthed):

- `ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/*.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_{h,w,hw}.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/{line,ring,dim_zero_line,dim_zero_ring}_reduction.cpp`

### Pattern 2 — Batched DEST accumulation

**Difficulty: EASY. Helper support: FULL (via `WaitAndPopPerChunk` or `BinaryAccumulate`).**

```cpp
cb_wait_front(cb_in0, N);
cb_wait_front(cb_in1, 1);
tile_regs_acquire();
for (uint32_t i = 0; i < N; ++i) {
    add_tiles(cb_in0, cb_in1, i, 0, /*dst=*/0);
}
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_out);
tile_regs_release();
```

Replacement (scalar B reused; partial result reloaded from accumulator CB):

```cpp
add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerChunk,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    cb_in0, cb_in1, cb_out, BinaryInputBlockShape::row(N),
    NoOp{}, BinaryAccumulate{cb_accum, /*dst_index=*/0});
```

Representative files:

- `ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/accumulation_compute.cpp`
- `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/compute/reduction.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/reduction.cpp`

### Pattern 3 — Grid traversal with manual subblock indexing

**Difficulty: MEDIUM. Helper support: FULL for the indexing; PARTIAL for reconfig.**

```cpp
for (uint32_t h = 0; h < Ht; ++h) {
    for (uint32_t w = 0; w < Wt; w += subblock_w) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < subblock_w; ++i) {
            uint32_t idx = h * Wt + w + i;
            mul_tiles(cb_a, cb_b, idx, idx, i);
        }
        tile_regs_commit();
        // pack subblock_w tiles...
    }
}
```

Replacement:

```cpp
mul<BroadcastDim::NONE,
    BinaryInputPolicy::WaitUpfrontPopAtEnd>(
    cb_a, cb_b, cb_out, BinaryInputBlockShape::of(Ht, Wt));
```

`DEST_AUTO_LIMIT` handles the subblock split internally. The `subblock_w`
concept disappears from caller code.

**Caveat:** many files in this pattern interleave `reconfig_data_format_srcb`
mid-chain — see `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp`
(6 `_srcb` sites) and `welford_groupnorm.cpp` (7 sites). Those kernels need
Gap #1 (Section 4) or have to use `reconfig = BinaryDataFormatReconfig::NONE`
with caller-side manual reconfigs.

### Pattern 4 — Broadcast with persisted B (no pop)

**Difficulty: EASY. Helper support: FULL.**

```cpp
cb_wait_front(cb_scale, 1);
for (uint32_t h = 0; h < Ht; ++h) {
    for (uint32_t w = 0; w < Wt; ++w) {
        cb_wait_front(cb_data, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_data, cb_scale, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_pop_front(cb_data, 1);
        cb_push_back(cb_out, 1);
    }
}
// cb_scale stays in the CB
```

Replacement:

```cpp
mul<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    cb_data, cb_scale, cb_out, BinaryInputBlockShape::of(Ht, Wt));
```

Representative files:

- `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp`
- `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp`
- `ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather.cpp`

### Pattern 5 — Binary op with fused post-op (rsqrt / sfpu)

**Difficulty: MEDIUM. Helper support: FULL for per-tile PostOp; missing for batch-level PostOp.**

```cpp
tile_regs_acquire();
add_tiles(cb_var, cb_eps, 0, 0, /*dst=*/0);
rsqrt_tile_init<true>();
rsqrt_tile<true>(0);
tile_regs_commit();
// pack...
```

Replacement:

```cpp
add(cb_var, cb_eps, cb_out, BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) {
        rsqrt_tile_init<true>();
        rsqrt_tile<true>(dst_idx);
    });
```

Representative files:

- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm*.cpp`
- `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather*.cpp`
- `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`

### Pattern 6 — Reconfig-heavy chains (`_srcb` / `_srca`)

**Difficulty: MEDIUM. Helper support: PARTIAL — this is Gap #1.**

```cpp
reconfig_data_format(cb_a, cb_b);
pack_reconfig_data_format(cb_out);
add_tiles_init(cb_a, cb_b);
// op 1

reconfig_data_format_srcb(cb_b, cb_c);   // swap only srcb unpacker
mul_tiles_init(cb_a, cb_c);
// op 2
```

Current best-effort replacement (pass `reconfig = NONE` and handle reconfig
outside):

```cpp
reconfig_data_format(cb_a, cb_b);
pack_reconfig_data_format(cb_out);
add<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::NONE>(
    cb_a, cb_b, cb_out, BinaryInputBlockShape::of(Ht, Wt));

reconfig_data_format_srcb(cb_b, cb_c);
mul<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::NONE>(
    cb_a, cb_c, cb_out, BinaryInputBlockShape::of(Ht, Wt));
```

Files with the heaviest `_srcb/_srca` concentration (top of the 53-file list):

- `layernorm_large_tensor_welford.cpp` (9)
- `groupnorm_sharded_v2.cpp` (6), `groupnorm.cpp` (6), `welford_groupnorm.cpp` (7)
- `layernorm.cpp` (6), `layernorm_large_tensor.cpp` (6), `layernorm_welford.cpp` (6)
- `softmax_large_tensor.cpp` (5), `layernorm_sharded.cpp` (5)

### Pattern 7 — Moreh-style single-tile chains (`*_tiles_to_cb`)

**Difficulty: N/A — uses `moreh_common.hpp`, not raw tile APIs.**

```cpp
mul_tiles_to_cb(cb_a,   cb_b,   cb_out, 0, 0, /*pop_a=*/0, /*pop_b=*/0);
add_tiles_to_cb(cb_c,   cb_out, cb_out, 0, 0, /*pop_a=*/0, /*pop_b=*/1);
sub_tiles_to_cb(cb_one, cb_d,   cb_tmp, 0, 0, /*pop_a=*/0, /*pop_b=*/0);
```

13 production kernels rely on this API (see Section 5). Most can be expressed
as chained `binary_op_helpers.hpp` calls with `BinaryInputBlockShape::single()`
and per-input `BinaryInputPolicy` choices, but each replacement is a multi-line
rewrite — different cost profile from Tiers 1–2.

### Pattern 8 — Complex multi-CB / conditional dispatch

**Difficulty: HARD. Helper support: NONE intended.**

```cpp
for (uint32_t w = 0; w < block_w; ++w) {
    if (copy_or_add) {
        copy_tile_init(cb_xmm);
        copy_tile(cb_xmm, w, /*dst=*/0);
    } else {
        add_tiles_init(cb_reread_out, cb_xmm);
        add_tiles(cb_reread_out, cb_xmm, w_reread, w, /*dst=*/0);
    }
    // conditional pack...
}
```

Kernels: `groupnorm.cpp`, `welford_groupnorm.cpp`, `welford_groupnorm_sharded_v2.cpp`,
`conv_bmm_tilize.cpp`, `bmm_large_block_zm_fused_bias_activation.cpp`,
`transformer_group_attn_matmul.cpp`. These are not Tier-1/2/3 migration
candidates; they should stay on raw APIs or be surgically refactored.

---

## 3. Kernel Tiering

### Tier 1 — Direct replacement (EASY)

Verified against current paths. Each entry lists the dominant pattern.

| File                                                                                                     | Pattern | Primary op |
|----------------------------------------------------------------------------------------------------------|---------|------------|
| `eltwise/binary/device/kernels/compute/bcast_h.cpp`                                                      | 1       | `add<ROW>` / templated |
| `eltwise/binary/device/kernels/compute/bcast_w.cpp`                                                      | 1       | `add<COL>` / templated |
| `eltwise/binary/device/kernels/compute/bcast_hw.cpp`                                                     | 1       | `add<SCALAR>` / templated |
| `data_movement/bcast/device/kernels/*.cpp`                                                               | 1       | any-op, bcast templated |
| `experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduction.cpp`                        | 2       | `add` with accumulator |
| `experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduction.cpp`                        | 2       | `add` with accumulator |
| `experimental/ccl/reduce_scatter_minimal_async/device/kernels/dim_zero_line_reduction.cpp`               | 2       | `add` with accumulator |
| `experimental/ccl/reduce_scatter_minimal_async/device/kernels/dim_zero_ring_reduction.cpp`               | 2       | `add` with accumulator |
| `experimental/ccl/strided_reduce_scatter_async/device/kernels/minimal_ring_reduction.cpp`                | 2       | `add` with accumulator |
| `experimental/ccl/llama_reduce_scatter/device/kernels/compute/reduction.cpp`                             | 2       | `add` with accumulator |
| `experimental/ccl/llama_reduce_scatter_create_heads/device/kernels/compute/reduction.cpp`                | 2       | `add` with accumulator |
| `experimental/ccl/all_reduce_async/device/kernels/compute/reduction.cpp`                                 | 2       | `add` with accumulator |
| `experimental/transformer/all_reduce_create_qkv_heads/device/kernels/compute/reduction.cpp`              | 2       | `add` with accumulator |
| `experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/deepseek_moe_reduce_scatter_reduction.cpp`  | 2       | `add` with accumulator |
| `experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp`                                     | 2       | `add` with accumulator |
| `experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/deepseek_moe_fast_reduce_nc_reduce.cpp` | 2     | `add` with accumulator |
| `reduction/accumulation/device/kernels/compute/accumulation_compute.cpp`                                 | 2       | `add` / `mul` |
| `experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp`      | 4       | `add<SCALAR>` or `mul<SCALAR>` |
| `experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded_row_major.cpp` | 4 | similar |

### Tier 2 — Requires partial-reconfig support OR caller-managed reconfig

Needs Gap #1 resolved or the migration swallows extra reconfig lines in
caller code. Grid + persisted-B + PostOp stack is the common shape.

| File                                                                                               | Notes |
|----------------------------------------------------------------------------------------------------|-------|
| `normalization/layernorm/device/kernels/compute/layernorm.cpp`                                     | 6 `_srcb`, grid + PostOp rsqrt |
| `normalization/layernorm/device/kernels/compute/layernorm_large_tensor.cpp`                        | 6 `_srcb` |
| `normalization/layernorm/device/kernels/compute/layernorm_welford.cpp`                             | 6 `_srcb` |
| `normalization/layernorm/device/kernels/compute/layernorm_large_tensor_welford.cpp`                | 9 `_srcb` |
| `normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp`                             | 5 `_srcb` |
| `normalization/layernorm/device/kernels/compute/layernorm_sharded_welford.cpp`                     | 4 `_srcb` |
| `normalization/layernorm/device/kernels/compute/layernorm_sharded_post_allgather.cpp`              | 2 `_srcb` |
| `normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather*.cpp`         | PostOp chain |
| `normalization/softmax/device/kernels/attention/compute/softmax*.cpp`                              | persistent max/sum |
| `normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_*.cpp`                           | persistent scale |
| `normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`                            | ADD/SUB/MUL chain + PostOp |
| `experimental/ccl/rms_allgather/device/kernels/compute/rms_compute.cpp`                            | 2 `_srcb` |
| `experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather.cpp`| persistent scale |
| `experimental/transformer/dit_layernorm_post_all_gather/device/kernels/compute/layernorm_post_allgather_welford.cpp` | welford |
| `transformer/sdpa/device/kernels/compute/compute_common.hpp`                                       | persistent K/V lifetime |
| `conv/conv2d/device/kernels/compute_depthwise_conv1d.cpp`                                          | grid, per-channel bcast |

### Tier 3 — Not recommended without refactor

Conditional dispatch, multi-CB reread, or fused matmul/reduce. Migration
yields marginal readability win at best.

| File                                                                                      | Reason |
|-------------------------------------------------------------------------------------------|--------|
| `normalization/groupnorm/device/kernels/compute/groupnorm.cpp`                            | Pattern 8 (copy-or-add branching) |
| `normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp`                 | Pattern 8 |
| `normalization/groupnorm/device/kernels/compute/welford_groupnorm.cpp`                    | Welford state machine |
| `normalization/groupnorm/device/kernels/compute/welford_groupnorm_sharded_v2.cpp`         | Welford state machine |
| `conv/conv2d/device/kernels/conv_bmm_tilize.cpp`                                          | Fused with matmul |
| `matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`              | Fused with matmul |
| `experimental/matmul/group_attn_matmul/device/kernels/compute/transformer_group_attn_matmul.cpp` | Fused with matmul |
| `moreh/moreh_adam/device/kernels/moreh_adam.cpp`                                          | Pattern 7 (moreh_common) |
| `moreh/moreh_adamw/device/kernels/moreh_adamw.cpp`                                        | Pattern 7 |
| `moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_*_kernel.cpp`   | Pattern 7 + `_srcb` |

---

## 4. Library Gaps

### Gap 1 — Partial data-format reconfig (BLOCKER for Tier 2)

**Symptom:** `BinaryDataFormatReconfig` has no `SRCB_ONLY` / `SRCA_ONLY`
variant, while 53 kernels (172 call sites) use `reconfig_data_format_srcb` /
`_srca` to switch only one unpacker side between ops.

**Affected kernels:** layernorm / groupnorm / softmax / rmsnorm families,
layernorm-welford variants, matmul-fused activation kernels, several moreh
kernels.

**Proposed extension:**

```cpp
enum class BinaryDataFormatReconfig {
    NONE              = 0,
    INPUT             = 1,
    OUTPUT            = 2,
    INPUT_AND_OUTPUT  = 3,
    SRCA_ONLY         = 4,  // reconfig_data_format_srca(old_cb, new_cb)
    SRCB_ONLY         = 5,  // reconfig_data_format_srcb(old_cb, new_cb)
};
```

The helper also needs to learn the previous-CB identity for a one-sided
reconfig (currently inferred from the `binary_op_init_common` state). Two
realistic API shapes: either thread explicit `prev_icb_a` / `prev_icb_b`
parameters, or have the helper track per-side last-configured CB in a thread-
local state object. The header already keeps enough state to do the latter.

**Classification:** BLOCKER for Tier 2. Without it, Tier 2 migrations must
sprinkle `reconfig = NONE` + manual reconfigs, defeating most of the
readability win.

### Gap 2 — `moreh_common.hpp` overlap

**Symptom:** `moreh_common.hpp` exposes `add_tiles_to_cb`, `mul_tiles_to_cb`,
`sub_tiles_to_cb`, plus `*_bcast_*_to_cb` and `*_with_dt` variants. 13
production kernels use these APIs; they provide a narrower feature set
(single-tile only, explicit per-input pop flag) but identical responsibility
(wait/acquire/op/commit/pack/push).

**Recommendation — explicit:**

- **Coexistence, not deprecation, in the short term.** The `*_to_cb` APIs
  are per-tile-granularity and already pervasive in moreh. Forcing migration
  buys little because moreh kernels chain many small ops.
- **New code: prohibit `moreh_common.hpp` binary ops.** Require
  `binary_op_helpers.hpp` for any new compute kernel under `operations/`.
- **Long term:** reimplement the `*_to_cb` helpers as thin shims over
  `binary_op_helpers.hpp` (single-tile shape + `WaitAndPopPerTile` or
  equivalent) so the two abstractions collapse into one. That requires Gap #1
  (many moreh kernels interleave `_srcb` reconfigs) and a per-side pop flag
  analogous to the `WaitAndPop*` / `WaitUpfrontNoPop` distinction.

**Classification:** Policy decision, not a code blocker.

### Gap 3 — Batch-level PostOp hook

**Symptom:** `PostOp` is invoked per DEST slot. Some follow-up SFPU passes
(e.g. vectorised rsqrt over a subblock) are cheaper when issued once across
a chunk.

**Proposed extension:**

```cpp
template <..., typename PostOp = NoOp, typename BatchPostOp = NoOp, ...>
ALWI void binary_op(..., PostOp post_op = {}, BatchPostOp batch_post_op = {});
// batch_post_op(base_dst, chunk_size) called once per DEST chunk.
```

**Classification:** MINOR. Per-tile PostOp works everywhere; batch hook is
optimisation-only.

### Gap 4 — `sub_bcast_rows` LLK `_init_short`

**Symptom:** No `sub_bcast_rows_init_short` symbol exists in the repo
(verified — six hits all for `sub_bcast_rows` as part of
`sub_tiles_bcast_rows`/`sub_bcast_rows_init_short` were only found as
substrings in `sub_bcast_cols`-related names; no true match). Any
`sub<BroadcastDim::ROW>` invocation has to emit a generic
`init_bcast<...>` or construct the equivalent from the other primitives.

**Classification:** MINOR. Few kernels need sub-with-row-broadcast; the
helper can silently fall back to the generic form.

---

## 5. `moreh_common.hpp` relationship — explicit recommendation

See Gap #2. Concrete position:

1. `binary_op_helpers.hpp` is the canonical abstraction; new kernels under
   `ttnn/cpp/ttnn/operations/` must use it.
2. `moreh_common.hpp` remains supported for existing moreh kernels. Do not
   migrate moreh kernels eagerly — they chain 10+ ops per step; a
   large-surface rewrite is high-risk.
3. When Gap #1 (partial reconfig) lands, plan a follow-up to rewrite
   `*_tiles_to_cb` as thin shims. Until then, keep the two abstractions
   parallel.

---

## 6. Migration Strategy

### Phase 0 — Prove the abstraction (no library changes)

**Tier 1 POC set:**

- `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_h.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_w.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_hw.cpp`
- `ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/*.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduction.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduction.cpp`
- `ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/accumulation_compute.cpp`

Exit criteria: line-count reduction ≥ 40 % per file; perf parity on the
covering CI jobs.

### Phase 1 — Land Gap #1 (partial reconfig)

Ship `BinaryDataFormatReconfig::{SRCA_ONLY, SRCB_ONLY}`. Add unit coverage
using a contrived two-op chain with differently-typed CBs. No kernel migration
in this phase.

### Phase 2 — Tier 2 migration (normalization family)

Once Gap #1 ships, migrate (in order):

1. `rmsnorm_distributed/device/kernels/compute/rmsnorm_*.cpp`
2. `normalization/softmax/device/kernels/attention/compute/softmax*.cpp`
3. `normalization/layernorm/device/kernels/compute/layernorm*.cpp`
4. `normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`
5. `experimental/ccl/rms_allgather/device/kernels/compute/rms_compute.cpp`
6. `experimental/transformer/*/compute/rmsnorm_post_allgather.cpp` variants

### Phase 3 — Tier 3 case-by-case

Evaluate per file. Default is **do not migrate**; keep raw APIs with a
one-line comment pointing to this document.

---

## 7. Summary Statistics

| Metric                                               | Count |
|------------------------------------------------------|-------|
| Files using `add_tiles`                              | 70    |
| Files using `mul_tiles`                              | 46    |
| Files using `sub_tiles`                              |  9    |
| Files using row-broadcast variants                   | 40    |
| Files using column-broadcast variants                | 34    |
| Files using scalar-broadcast variants                | 24    |
| Files using `reconfig_data_format_srcb/_srca`        | 53    |
| Total `_srcb/_srca` call sites                       | 172   |
| Files using `moreh_common` `*_tiles_to_cb`           | 13    |
| Files using `compute_kernel_lib::{add,sub,mul,...}`  |  0 (production) |
| **Tier 1** (direct migration ready today)            | ~28   |
| **Tier 2** (blocked on partial reconfig)             | ~25   |
| **Tier 3** (refactor required or not worth it)       | ~17   |

Counts intentionally round; a handful of files span multiple tiers and are
categorised by dominant pattern.

---

## 8. Confidence Notes

- All file paths, counts, and `reconfig_data_format_*` totals verified via
  `rg` against the current tree at the time of writing.
- Adoption claim ("zero production callers of `compute_kernel_lib::*` binary
  APIs") verified via regex `compute_kernel_lib::(add|sub|mul|square|binary_op)\s*[<(]`
  under `ttnn/cpp/ttnn/operations/` — only hit is a markdown file.
- Moreh kernel count for `*_tiles_to_cb` is 13 **production** kernels
  (excludes `moreh_common.hpp` header itself).
- Line-reduction estimates in the migration strategy are speculative
  (UNCERTAIN) — they are based on the Tier-1 bcast kernels' structure but
  have not been measured.
- Pattern 7's claim that "most moreh chains can be expressed via chained
  `binary_op_helpers.hpp` calls" is directionally correct but
  **UNCERTAIN** until Gap #1 lands, because many moreh kernels interleave
  `_srcb` reconfigs.

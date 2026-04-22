# FPU Binary Patterns — What The Helper Does Not Cover, And How To Cover It

Ground truth (2026-04-21): `binary_op_helpers.hpp` covers ~65% of the 70
`add_tiles` / 46 `mul_tiles` / 9 `sub_tiles` call-site files. This document
enumerates the remaining patterns, with the current raw kernel code on the left
and the **proposed helper API + migrated kernel** on the right.

Legend:

- **FIX (small)** — a targeted API addition, typically one struct or one enum
  value + a handful of `.inl` lines.
- **FIX (medium)** — new helper surface (50–150 lines) but no semantic change
  to existing callers.
- **SHIM** — thin wrappers, no new semantic capability.
- **HARD** — structurally incompatible, document as stays-raw.

Status matrix (anchored to specific files — grep-verified):

| # | Pattern | Files | Fixability |
|---|---------|-------|------------|
| A | Standalone `binary_dest_reuse_tiles` as primary op | 16 | FIX (medium) |
| B | Runtime-indexed DEST slot for dest-reuse | 3 | FIX (small) |
| C | Multi-tile one-to-one dest-reuse after bcast-add | 2 | FIX (small, depends on A+B) |
| D | Cumulative CB wait + non-zero B tile offset | 3–4 | FIX (small) or call-site refactor |
| E | `moreh_common.hpp` `*_tiles_to_cb` | 13 | SHIM |
| F | Runtime copy-or-add dispatch / random-access pack | 6 | HARD |
| G | Fused matmul + activation | 5 | HARD (out of scope) |
| H | `power_tile` / DEST-scalar SFPU between FPU stages | 2 | HARD (moreh_adam/adamw) |
| I | Same-CB non-contiguous index pairs | 2 | FIX (small) or stays-raw |

---

## A — Standalone `binary_dest_reuse_tiles` as primary op

**Current helper state.** `DestReuseOp` is a *chain element*, valid only as a
PostOp to `binary_op()`. Its contract is "DEST already holds a binary result
— fuse another CB read on top of it." These kernels instead seed DEST with
`fill_tile`, `copy_tile`, `matmul_block`, or a bcast-add and then use
`binary_dest_reuse_tiles` as the *only* FPU binary operation.

**Affected files (16):**

```
eltwise/unary_ng/.../kernels/compute/hardshrink_kernel.cpp
eltwise/unary_ng/.../kernels/compute/hardswish_kernel.cpp
eltwise/unary_ng/.../kernels/compute/mish_kernel.cpp
eltwise/unary_ng/.../kernels/compute/tanhshrink_kernel.cpp
eltwise/unary/.../kernels/compute/hardshrink_kernel.cpp
eltwise/unary/.../kernels/compute/hardswish_kernel.cpp
eltwise/unary/.../kernels/compute/mish_kernel.cpp
eltwise/unary/.../kernels/compute/tanhshrink_kernel.cpp
eltwise/ternary/.../ternary_addc_ops_fpu.cpp
eltwise/ternary/.../ternary_addc_ops_fpu_bcast.cpp
experimental/deepseek/mla/matmul_wo/.../compute_collector.cpp
experimental/deepseek/moe/moe_gate_mm/.../compute.cpp
experimental/topk_router_gpt/.../compute.cpp
normalization/softmax/.../compute/softmax_large_tensor.cpp
normalization/layernorm/.../compute/layernorm_large_tensor.cpp
normalization/layernorm/.../compute/layernorm_large_tensor_welford.cpp
```

### Raw kernel today (deepseek/mla compute_collector — accumulator)

```cpp
binary_op_init_common(cb_s2c_in2, cb_s2c_in2, cb_s2c_out);
binary_dest_reuse_tiles_init<ELWADD, DEST_TO_SRCA>(cb_s2c_in2);

for (uint32_t iter_id = 0; iter_id < num_iters; ++iter_id) {
    cb_wait_front(cb_s2c_in2, num_cores);
    tile_regs_acquire();
    for (uint32_t k = 0; k < num_cores; ++k) {
        binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>(cb_s2c_in2, k, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_s2c_out);
    tile_regs_release();
    cb_pop_front(cb_s2c_in2, num_cores);
}
```

### Raw kernel today (unary_ng hardshrink — DEST seeded by fill_tile)

```cpp
fill_tile(0, *lambd);                                                      // DEST[0] = scalar
binary_dest_reuse_tiles_init<ELWADD, DEST_TO_SRCA>(cb_input);
binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>(cb_input, 0, 0);             // DEST[0] += cb_input[0]
ltz_tile(0);                                                               // DEST[0] = (DEST[0] < 0)
binary_dest_reuse_tiles_init<ELWMUL, DEST_TO_SRCA>(cb_input);
binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_input, 0, 0);             // DEST[0] *= cb_input[0]
```

### Why the helper can't express this today

- `binary_op()` *must* start the pipeline with a two-CB FPU op. Here the
  first FPU operand is already in DEST (from `fill_tile` / `matmul_block`).
- `DestReuseOp` as a PostOp requires a prior `binary_op()` call.

### Proposal — promote dest-reuse to a first-class primary op

Add a standalone entry point, and re-express the existing `DestReuseOp`
PostOp as a thin wrapper over it.

```cpp
// New header surface in binary_op_helpers.hpp
namespace compute_kernel_lib {

enum class DestReuseInputPolicy {
    WaitAndPopPerTile,
    WaitUpfrontNoPop,
    WaitUpfrontPopAtEnd,
    NoWaitNoPop,
    NoWaitPopAtEnd
};

/**
 * @brief Accumulate/apply CB tiles onto DEST that was seeded by another op.
 *
 * DEST is already populated by one of: fill_tile, copy_tile, matmul_block,
 * a prior binary_op, a bcast_add, etc. This helper performs
 *     DEST[dst_idx + i] = DEST[dst_idx + i] OP cb[cb_tile_offset + i]   (DEST_TO_SRCA)
 * or  DEST[dst_idx + i] = cb[cb_tile_offset + i] OP DEST[dst_idx + i]   (DEST_TO_SRCB)
 * for i in [0, count).
 *
 * @tparam OpType     ELWADD / ELWSUB / ELWMUL
 * @tparam ReuseType  DEST_TO_SRCA (default) or DEST_TO_SRCB
 * @tparam Policy     CB wait/pop lifecycle
 * @tparam Reconfig   CB-side data-format reconfig mode
 */
template <
    EltwiseBinaryType OpType,
    EltwiseBinaryReuseDestType ReuseType = EltwiseBinaryReuseDestType::DEST_TO_SRCA,
    DestReuseInputPolicy Policy = DestReuseInputPolicy::WaitAndPopPerTile,
    DestReuseReconfig Reconfig = DestReuseReconfig::None>
ALWI void dest_reuse(
    uint32_t cb,
    uint32_t count,
    uint32_t dst_idx = 0,
    uint32_t cb_tile_offset = 0);

// Convenience aliases mirroring the existing add/sub/mul style
template <EltwiseBinaryReuseDestType R = EltwiseBinaryReuseDestType::DEST_TO_SRCA, ...>
ALWI void dest_reuse_add(uint32_t cb, uint32_t count, uint32_t dst_idx = 0, uint32_t cb_off = 0);
template <EltwiseBinaryReuseDestType R = EltwiseBinaryReuseDestType::DEST_TO_SRCA, ...>
ALWI void dest_reuse_mul(uint32_t cb, uint32_t count, uint32_t dst_idx = 0, uint32_t cb_off = 0);
template <EltwiseBinaryReuseDestType R = EltwiseBinaryReuseDestType::DEST_TO_SRCA, ...>
ALWI void dest_reuse_sub(uint32_t cb, uint32_t count, uint32_t dst_idx = 0, uint32_t cb_off = 0);

}  // namespace compute_kernel_lib
```

Implementation (~40 lines in `.inl`):

```cpp
template <EltwiseBinaryType Op, EltwiseBinaryReuseDestType R, DestReuseInputPolicy P, DestReuseReconfig Rc>
ALWI void dest_reuse(uint32_t cb, uint32_t count, uint32_t dst_idx, uint32_t cb_off) {
    if constexpr (Rc == DestReuseReconfig::Input) {
        if constexpr (R == EltwiseBinaryReuseDestType::DEST_TO_SRCA) reconfig_data_format_srcb(cb);
        else                                                         reconfig_data_format_srca(cb);
    }
    binary_dest_reuse_tiles_init<Op, R>(cb);
    if constexpr (does_wait_upfront(P)) cb_wait_front(cb, count);
    for (uint32_t i = 0; i < count; ++i) {
        if constexpr (P == DestReuseInputPolicy::WaitAndPopPerTile) cb_wait_front(cb, 1);
        binary_dest_reuse_tiles<Op, R>(cb, cb_off + i, dst_idx + i);
        if constexpr (P == DestReuseInputPolicy::WaitAndPopPerTile) cb_pop_front(cb, 1);
    }
    if constexpr (does_pop_at_end(P)) cb_pop_front(cb, count);
}
```

### Migrated kernel (deepseek compute_collector)

```cpp
binary_op_init_common(cb_s2c_in2, cb_s2c_in2, cb_s2c_out);

cb_reserve_back(cb_s2c_out, num_iters);
for (uint32_t iter_id = 0; iter_id < num_iters; ++iter_id) {
    tile_regs_acquire();
    dest_reuse_add<DEST_TO_SRCA, DestReuseInputPolicy::WaitUpfrontPopAtEnd>(
        cb_s2c_in2, num_cores);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_s2c_out);
    tile_regs_release();
}
cb_push_back(cb_s2c_out, num_iters);
```

### Migrated kernel (unary_ng hardshrink, one branch)

```cpp
fill_tile(0, *lambd);
dest_reuse_add<DEST_TO_SRCA, DestReuseInputPolicy::NoWaitNoPop>(cb_input, /*count=*/1);
ltz_tile(0);
dest_reuse_mul<DEST_TO_SRCA, DestReuseInputPolicy::NoWaitNoPop>(cb_input, /*count=*/1);
```

---

## B — Runtime-indexed DEST slot for dest-reuse

**Current helper state.** `DestReuseOp` takes `Dst Slot` as a compile-time
template parameter. These kernels need it to be a runtime value.

**Affected files (3):**

```
normalization/softmax/.../compute/softmax_large_tensor.cpp
normalization/layernorm/.../compute/layernorm_large_tensor.cpp
normalization/layernorm/.../compute/layernorm_large_tensor_welford.cpp
```

### Raw kernel today (softmax_large_tensor)

```cpp
if (do_mask && cur_blk == cb_length_t - blk) {
    reconfig_data_format_srca(cb_mask_padded);
    binary_dest_reuse_tiles_init<ELWADD, DEST_TO_SRCB>(cb_mask_padded);
    cb_wait_front(cb_mask_padded, 1);
    binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCB>(cb_mask_padded, 0, blk - 1);  // blk is runtime
}
```

### Why the helper can't express this today

`DestReuseOp<CB, OpType, ReuseType, Slot>` static_asserts `Slot < 8`;
`blk - 1` cannot be a template arg.

### Proposal — runtime `dst_idx` parameter

Either extend `DestReuseOp` with a runtime `dst_idx` instance member, or
require A (the standalone primary-op form above), which already takes
`dst_idx` as a runtime arg. Same two-liner change, either way:

```cpp
// Existing:
template <uint32_t CB, ..., Dst Slot = Dst::D0, ...>
struct DestReuseOp { static constexpr uint32_t dst_idx = (uint32_t)Slot; ... };

// Proposed: remove Slot from the template, keep it as a runtime field
template <uint32_t CB, ..., DestReuseReconfig Reconfig = DestReuseReconfig::None>
struct DestReuseOp {
    uint32_t dst_idx = 0;
    uint32_t cb_tile_idx = 0;
    ...
};
```

### Migrated kernel (softmax_large_tensor fix-up branch)

```cpp
if (do_mask && cur_blk == cb_length_t - blk) {
    dest_reuse_add<DEST_TO_SRCB,
                   DestReuseInputPolicy::WaitUpfrontNoPop,
                   DestReuseReconfig::Input>(
        cb_mask_padded, /*count=*/1, /*dst_idx=*/blk - 1);
}
```

---

## C — Multi-tile one-to-one dest-reuse fused after bcast-add

Same kernel file, same block. `i` pairs DEST[i] with `cb_inb[i]`. Needs A+B.

**Affected files (2):**

```
normalization/layernorm/.../compute/layernorm_large_tensor.cpp
normalization/layernorm/.../compute/layernorm_large_tensor_welford.cpp
```

### Raw kernel today (layernorm_large_tensor)

```cpp
// ... prior loop packed N tiles into DEST[0..N-1] via some bcast ...
binary_dest_reuse_tiles_init<ELWADD, DEST_TO_SRCB>(cb_inb);
for (uint32_t i = 0; i < blk; ++i) {
    binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCB>(cb_inb, i, i);
}
```

### Migrated (A + B together)

```cpp
dest_reuse_add<DEST_TO_SRCB, DestReuseInputPolicy::WaitUpfrontNoPop>(
    cb_inb, /*count=*/blk, /*dst_idx=*/0, /*cb_off=*/0);
```

---

## D — Cumulative CB wait + non-zero B tile start offset

**Current helper state.** `WaitUpfrontNoPop` waits for N tiles once.
`WaitAndPopPerChunk` waits `blk` each chunk but pops them. Neither covers
"wait for `wt + blk` tiles cumulatively, do not pop, then read from
non-zero offset."

**Affected files:**

```
normalization/rmsnorm_distributed/.../compute/rmsnorm_pre_allgather.cpp
normalization/rmsnorm_distributed/.../compute/rmsnorm_pre_allgather_2d.cpp
experimental/transformer/fused_distributed_rmsnorm/.../compute/rmsnorm_pre_allgather.cpp
experimental/transformer/rotary_embedding_llama/.../compute/rotary_embedding_llama.cpp (partial)
```

### Raw kernel today (rmsnorm_pre_allgather)

```cpp
mul_tiles_init(cb_inp, cb_inp);
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    cb_wait_front(cb_inp, wt + blk);                         // cumulative
    cb_reserve_back(cb_x2, blk);
    tile_regs_acquire();
    for (uint32_t wtr = 0; wtr < blk; ++wtr) {
        mul_tiles(cb_inp, cb_inp, wt + wtr, wt + wtr, wtr);  // non-zero input indices
        pack_tile(wtr, cb_x2, wt + wtr);                     // random-access output (sep. issue)
    }
    tile_regs_commit(); tile_regs_wait(); tile_regs_release();
    cb_push_back(cb_x2, blk);
}
```

### Fix option 1 — call-site refactor (no helper change)

Works if the CB can hold all `Wt` tiles at once (UNCERTAIN for this kernel
— needs CB-size verification):

```cpp
cb_wait_front(cb_inp, Wt);
square<BinaryInputPolicy::NoWaitNoPop, BinaryOutputPolicy::Bulk>(
    cb_inp, cb_x2, BinaryInputBlockShape::row(Wt));
cb_pop_front(cb_inp, Wt);
```

### Fix option 2 — policy + runtime offset parameters

```cpp
enum class BinaryInputPolicy {
    ...
    WaitCumulativeNoPop,   // wait for (current_total + this_chunk) each iteration
};

template <...>
ALWI void binary_op(
    uint32_t icb_a, uint32_t icb_b, uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op = {}, AccumT accum = {},
    uint32_t a_tile_offset = 0,       // added
    uint32_t b_tile_offset = 0);      // added
```

`.inl` tracks a running total for `WaitCumulativeNoPop`; `a_tile_offset` /
`b_tile_offset` are applied to index math. ~30 lines.

Caller:

```cpp
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    square<BinaryInputPolicy::WaitCumulativeNoPop,
           BinaryOutputPolicy::PerChunk>(
        cb_inp, cb_x2, BinaryInputBlockShape::row(blk),
        NoOp{}, NoAccumulation{}, /*a_off=*/wt);
}
```

---

## E — `moreh_common.hpp` `*_tiles_to_cb` — SHIM

**Current helper state.** `moreh_common.hpp` exposes
`add_tiles_to_cb`, `mul_tiles_to_cb`, `sub_tiles_to_cb`, `copy_tile_to_cb`,
plus `*_bcast_*_to_cb` and `*_with_dt` variants. Each wraps
wait → acquire → init → exec → commit → pack → push with per-input pop flags
and uses `pack_tile_with_dt`.

**Affected files (13):**

```
moreh/moreh_sgd/device/kernels/moreh_sgd.cpp
moreh/moreh_softmax/device/kernels/moreh_softmax_{w_large,c_large,h_large}.cpp
moreh/moreh_norm_backward/device/kernels/moreh_norm_backward_kernel.cpp
moreh/moreh_adam/device/kernels/moreh_adam.cpp
moreh/moreh_adamw/device/kernels/moreh_adamw.cpp
moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_{w,w_large,h,h_large,c_large}.cpp
```

### Raw pattern

```cpp
mul_tiles_to_cb(cb_a, cb_b, cb_tmp, /*a=*/0, /*b=*/0, /*pop_a=*/0, /*pop_b=*/0);
add_tiles_to_cb(cb_c, cb_tmp, cb_out, /*a=*/0, /*b=*/0, /*pop_a=*/0, /*pop_b=*/1);
```

### Proposal — shim layer in a new header

`moreh_common_shim.hpp` (opt-in replacement):

```cpp
ALWI void add_tiles_to_cb(uint32_t icb_a, uint32_t icb_b, uint32_t ocb,
                          uint32_t tile_a, uint32_t tile_b,
                          uint32_t pop_a, uint32_t pop_b) {
    // non-zero tile_a / tile_b require the A-gap `a_tile_offset` / `b_tile_offset`
    compute_kernel_lib::add<BroadcastDim::NONE,
                            BinaryInputPolicy::NoWaitNoPop,
                            BinaryInputPolicy::NoWaitNoPop,
                            BinaryOutputPolicy::PerTile>(
        icb_a, icb_b, ocb, BinaryInputBlockShape::single(), NoOp{}, NoAccumulation{},
        tile_a, tile_b);
    if (pop_a) cb_pop_front(icb_a, 1);
    if (pop_b) cb_pop_front(icb_b, 1);
}
// analogous mul_tiles_to_cb, sub_tiles_to_cb, copy_tile_to_cb
```

**Prerequisites:**
1. Ship D (runtime `a_tile_offset` / `b_tile_offset`) — moreh chains use
   non-zero tile indices.
2. Verify `pack_tile_with_dt` vs `pack_tile` equivalence for BF16/FP32 (open
   question in the existing gap analysis).

**Migration policy (unchanged):** do not migrate eagerly; collapse the two
abstractions once shims are stable.

---

## F — Runtime copy-or-add dispatch / random-access `pack_tile<true>` — HARD

**Affected files (6):**

```
normalization/groupnorm/.../compute/groupnorm.cpp
normalization/groupnorm/.../compute/groupnorm_sharded_v2.cpp
normalization/groupnorm/.../compute/welford_groupnorm.cpp
normalization/groupnorm/.../compute/welford_groupnorm_sharded_v2.cpp
conv/conv2d/device/kernels/compute_depthwise_conv1d.cpp
eltwise/binary/device/kernels/compute/bcast_h_sharded_optimised.cpp  // random-access pack only
```

### Raw kernel today (groupnorm)

```cpp
if (copy_or_add == true) {                        // runtime boolean
    copy_tile_init(cb_xmm_id);
    copy_tile(cb_xmm_id, index_xmm, dst0);
} else {
    add_tiles_init(cb_reread_out_id, cb_xmm_id);
    add_tiles(cb_reread_out_id, cb_xmm_id, index_reread_out, index_xmm, dst0);
}
pack_tile<true>(dst0, cb_reread_write_out_id, index_reread_out);  // random-access
```

### Why there's no reasonable helper form

- `binary_op<op_type>` requires `op_type` to be a compile-time constant.
  Bridging a runtime bool to a compile-time `BinaryOpType` would require
  duplicating the whole FPU-binary template instantiation, compiled twice and
  selected at runtime — worse code size and IR for marginal readability win.
- `pack_tile<true>(dst, cb, explicit_index)` writes at an arbitrary output
  slot. The helper's output policies all assume sequential packing. Exposing
  "packer output index" as a per-tile callback destroys the chunk-level
  optimisations.

**Recommendation:** keep raw. Add a one-line comment in each file pointing
to this document. (Already planned in migration plan Phase 3.)

---

## G — Fused matmul + activation — HARD (out of scope)

**Affected files (5):**

```
conv/conv2d/device/kernels/conv_bmm_tilize.cpp
matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp
matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp
experimental/matmul/group_attn_matmul/.../transformer_group_attn_matmul.cpp
transformer/sdpa/device/kernels/compute/compute_common.hpp
```

These kernels interleave `matmul_block` with bias-add and optional SFPU
activation on the same DEST tiles. Abstracting that chain would require
modelling matmul inside `binary_op_helpers.hpp`, which is a category error —
`binary_op_helpers.hpp` owns FPU eltwise, not matmul.

**Recommendation:** stays raw, document as such.

---

## H — `power_tile` / DEST-scalar SFPU between FPU stages — HARD

**Affected files (2):**

```
moreh/moreh_adam/device/kernels/moreh_adam.cpp   (AMSGRAD path mostly cleared by SfpuMax)
moreh/moreh_adamw/device/kernels/moreh_adamw.cpp
```

### Raw pattern

```cpp
mul_tiles_to_cb(...);                // stage N
power_tile_init();
power_tile(dst0, step);              // DEST[0] = DEST[0]^step (integer power of DEST scalar)
mul_tiles_to_cb(...);                // stage N+1
```

### Why the helper can't express this today

`power_tile(dst, integer_exponent)` consumes a DEST slot and a runtime
integer — it is not an SFPU struct in `sfpu_chain` (unlike `SfpuRsqrt`,
`SfpuMax`, etc., which take DEST slots only). Wrapping it as a chain element
would require the chain to thread a runtime scalar argument through
`apply(dst_idx, extra_arg)` — a generalisation that no other chain element
needs and that breaks the zero-overhead CRTP model.

**Recommendation:** keep the enclosing loop raw. `SfpuMax`/`SfpuMin` already
cover `binary_max_tile` / `binary_min_tile`, which was the other moreh_adam
holdout.

---

## I — Same-CB non-contiguous index pairs

**Current helper state.** `add(cb, cb, cb_out, shape)` (same-CB same-index)
works. But `add_tiles(cb, cb, i, j, dst)` with `i != j` computed per
iteration does not map cleanly to the fixed-stride iteration pattern of
`binary_op()`.

**Affected files (2):**

```
experimental/ccl/llama_reduce_scatter/.../compute/reduction.cpp
experimental/transformer/all_reduce_create_qkv_heads/.../compute/reduction.cpp
```

### Raw kernel today (llama_reduce_scatter)

```cpp
for (uint32_t i = 0; i < num_pairs; ++i) {
    uint32_t first_index  = pair_i_first(i);   // non-contiguous
    uint32_t second_index = pair_i_second(i);
    tile_regs_acquire();
    add_tiles(cb, cb, first_index, second_index, 0);
    tile_regs_commit(); tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
}
```

### Fix option 1 — leverage D's runtime offsets

If the loop runs `num_pairs` iterations with one helper call per pair:

```cpp
for (uint32_t i = 0; i < num_pairs; ++i) {
    add<BroadcastDim::NONE,
        BinaryInputPolicy::NoWaitNoPop,
        BinaryInputPolicy::NoWaitNoPop,
        BinaryOutputPolicy::PerTile>(
        cb, cb, cb_out, BinaryInputBlockShape::single(),
        NoOp{}, NoAccumulation{},
        /*a_off=*/pair_i_first(i),
        /*b_off=*/pair_i_second(i));
}
```

Still one call per pair, but the boilerplate (acquire/commit/wait/pack/release)
collapses. Depends on D.

### Fix option 2 — stays raw

The per-pair boilerplate is already ~7 lines. If the team prefers not to
depend on D for this case, document and keep raw.

---

## Summary — recommended order of operations

1. **Ship A (primary-op `dest_reuse`) + B (runtime `dst_idx`) together.**
   Unblocks 16 unary-ng / unary / ternary / deepseek / softmax_large_tensor
   / layernorm_large_tensor files. These kernels are all recent, none depend
   on each other, low migration risk.
2. **Ship D (runtime `a_tile_offset` / `b_tile_offset` + `WaitCumulativeNoPop`).**
   Unblocks rmsnorm_pre_allgather variants AND is a prerequisite for the
   moreh shim (E).
3. **Ship E as shims over 1 + 2.** 13 moreh files. Policy: do not migrate,
   keep as parallel ABI; new code forbidden from using `moreh_common.hpp`.
4. **Leave F, G, H as documented raw.** ~13 files; migration value is
   negative.
5. **Re-express the existing PostOp `DestReuseOp` on top of the primary-op
   form from step 1.** No caller-visible change; removes duplicated logic
   between PostOp and primary-op paths.

Rough migratable-file counts after each step (cumulative, of the ~95 files
with any FPU binary activity): step 0 today = 60, step 1 = 76, step 2 = 80,
step 3 = 93. Steps 4/5 don't change the count.

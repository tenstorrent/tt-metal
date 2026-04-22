# Feature Gap Map

Maps each missing or broken helper feature to the kernels it blocks.
Derived from `kernel_eltwise_coverage.md` + `binary_migration_log.md`.

Each gap has: description, kernels blocked, estimated count, and fix complexity.

---

## binary_op_helpers gaps

### GAP-1: Absolute tile index on A or B operand
`mul_tiles(A, B, abs_i, abs_j, dst)` where indices aren't 0-based sequential.
Helpers only support: sequential (0,1,2,...) or compile-time-stride (ht*Wt+wt).

**Blocked kernels** (~40):
- normalization/layernorm/* (layernorm.cpp, layernorm_sharded*.cpp, layernorm_welford*.cpp, layernorm_large_tensor*.cpp) — 8 files
- normalization/layernorm_distributed/layernorm_post_allgather_welford.cpp (tile 1 on A for rsqrt)
- normalization/layernorm_distributed/layernorm_pre_allgather*.cpp
- normalization/groupnorm/* (groupnorm*.cpp, welford_groupnorm*.cpp) — 4 files
- normalization/rmsnorm_distributed/rmsnorm_pre_allgather*.cpp
- experimental/ccl/rms_allgather/rms_compute.cpp
- experimental/transformer/rotary_embedding_llama*/rotary_embedding_llama.cpp (RELOAD_IMPL=0 `j + sin_cos_row_cnt*Wt`)
- experimental/transformer/rotary_embedding_llama_fused_qk/rotary_embedding_llama_sharded_row_major.cpp
- normalization/layernorm_distributed/layernorm_post_allgather.cpp (chain_llk stages with fixed_CB_B_index=1)
- normalization/layernorm_distributed/layernorm_post_allgather_welford.cpp
- experimental/transformer/dit_layernorm_post_all_gather/layernorm_post_allgather_welford.cpp (rsqrt tile 1)
- moreh/moreh_layer_norm*.cpp — 5 files
- moreh/moreh_softmax*.cpp — 8 files

**Fix**: Add `cb_tile_idx` parameter on B side of `binary_op` (like `DestReuseOp` already has).
For A side: expose absolute tile index as a separate template or runtime field.
**Complexity**: Medium — `DestReuseOp` precedent exists; binary_op tile index tracking is internal.

---

### GAP-2: Cumulative wait policy
`cb_wait_front(CB, base + iter * step)` — count grows each chunk iteration.
No `BinaryInputPolicy` value waits cumulatively; only "upfront all" or "per tile/chunk fixed count".

**Blocked kernels** (~15):
- normalization/rmsnorm_distributed/rmsnorm_pre_allgather.cpp
- normalization/rmsnorm_distributed/rmsnorm_pre_allgather_2d.cpp
- normalization/layernorm_distributed/layernorm_pre_allgather.cpp
- normalization/layernorm_distributed/layernorm_pre_allgather_2d.cpp
- experimental/transformer/fused_distributed_rmsnorm/rmsnorm_post_allgather.cpp (weight fusion)
- normalization/softmax/attention/softmax.cpp (fused_attn cb)
- normalization/layernorm/layernorm_sharded_pre_allgather.cpp

**Fix**: Add `BinaryInputPolicy::WaitCumulative` — wait for `base + iter * step` tiles,
pop at end. Requires the helper to track an internal running wait count.
**Complexity**: High — breaks the helper's "wait count == pop count" invariant.

---

### GAP-3: Non-sequential output pack
`pack_tile(i, cb_out, wt + i)` — packs to an absolute output tile index, not 0-based.
Helper output is always sequential from 0 (PerTile/PerChunk/Bulk).

**Blocked kernels** (~10):
- normalization/rmsnorm_distributed/rmsnorm_pre_allgather.cpp
- normalization/rmsnorm_distributed/rmsnorm_pre_allgather_2d.cpp
- normalization/layernorm_distributed/layernorm_pre_allgather.cpp
- normalization/layernorm_distributed/layernorm_pre_allgather_2d.cpp
- normalization/layernorm/layernorm_sharded_pre_allgather.cpp
- eltwise/binary/bcast_h_sharded_optimised.cpp (pack_tile<true> with absolute index + L1 acc)

**Fix**: New `BinaryOutputPolicy::BulkAbsolute` that accepts a base-offset, or a caller-supplied
output tile index per pack. Hard: breaks the CB FIFO assumption (can you write to slot N while
slots 0..N-1 are still "not pushed"?).
**Complexity**: Very high — likely requires CB API changes or a new output CB model.

---

### GAP-4: Asymmetric wait/process/pop — `wait N, process M≤N, pop N`
Compute pops `tile_granularity` tiles per iteration even when only `M < tile_granularity` are processed.
Helper symmetric contract: wait(shape) == process(shape) == pop(shape).

**Blocked kernels** (9):
- experimental/ccl/reduce_scatter_minimal_async/line_reduction.cpp
- experimental/ccl/reduce_scatter_minimal_async/ring_reduction.cpp
- experimental/ccl/reduce_scatter_minimal_async/dim_zero_line_reduction.cpp
- experimental/ccl/reduce_scatter_minimal_async/dim_zero_ring_reduction.cpp
- experimental/ccl/all_reduce_async/reduction.cpp
- experimental/ccl/llama_reduce_scatter/reduction.cpp
- experimental/ccl/llama_reduce_scatter_create_heads/reduction.cpp
- experimental/ccl/deepseek_moe_reduce_scatter/deepseek_moe_reduce_scatter_reduction.cpp
- experimental/transformer/all_reduce_create_qkv_heads/reduction.cpp

**Fix**: New `BinaryInputPolicy::WaitAndPopPadded` — wait `N`, process `min(M, N)`, pop `N`.
Requires passing `N` (the pad stride) separately from shape's Wt (the process count).
**Complexity**: Medium — fits the policy enum model; requires a new shape parameter.

---

### GAP-5: In-place output with CB capacity = 1 (pop-before-reserve ordering)
Raw code: `cb_pop_front(A, 1); cb_reserve_back(A, 1)` — pops input before reserving output
on the SAME CB with capacity 1. Helper always reserves before pops → deadlocks.

**Blocked kernels** (1):
- experimental/transformer/fused_distributed_rmsnorm/rmsnorm_post_allgather.cpp (rsqrt stage 2)

**Fix**: Either increase CB capacity to 2 in the program factory (program_factory change, not helper),
or add `BinaryOutputPolicy::InplacePopFirst` that pops input before reserving output.
**Complexity**: Low if done in program factory; Medium if done as helper policy.

---

### GAP-6: `EltwiseBinaryType` → `BinaryOpType` mapping
`BCAST_LLKOP` is a preprocessor macro set to `EltwiseBinaryType::ELWMUL`, but `binary_op`
requires `BinaryOpType::MUL`. No automatic mapping; different enum types from different layers.

**Blocked kernels** (4):
- eltwise/binary/bcast_h.cpp
- eltwise/binary/bcast_w.cpp
- eltwise/binary/bcast_hw.cpp
- eltwise/binary/bcast_h_sharded_optimised.cpp

**Fix**: Add a `constexpr BinaryOpType from_eltwise_type(EltwiseBinaryType)` mapping helper,
OR add a `binary_op_with_bcast_macro<BCAST_LLKOP, BCAST_DIM>(...)` variant that accepts
`EltwiseBinaryType` directly.
**Complexity**: Low — purely mechanical. These kernels are already 10-15 lines; migration value
may not justify the effort.

---

### GAP-7: Self-feeding accumulator
Output is packed to `cb_acc`, then `cb_wait_front(cb_acc, 1) + copy_tile + pack_tile(cb_out)`
inside the SAME acquire window. Alternately, operand B switches between `cb_start` (first
iteration) and `cb_acc` (subsequent iterations) via a runtime conditional.

**Blocked kernels** (3):
- reduction/accumulation/accumulation_compute.cpp (cumprod/cumsum)
- reduction/prod/prod_all.cpp
- reduction/prod/prod_nc.cpp

**Fix**: Requires a new "accumulator reload" output model — helper packs result, then
immediately re-reads it for the next iteration. The `BinaryAccumulate` struct in the helper
already handles this for a subset of cases; needs extension for the "output-then-reload-as-B"
pattern.
**Complexity**: Medium — `BinaryAccumulate` stub exists; generalization needed.

---

### GAP-8: L1 accumulation pack
`pack_tile<true>(i, cb, 0)` + `llk_pack_reconfig_l1_acc(1)` — repeatedly packs to the same
output tile slot, accumulating in L1 (not replacing). Helper packs are always sequential writes.

**Blocked kernels** (2):
- experimental/transformer/fused_distributed_rmsnorm/rmsnorm_pre_allgather.cpp
- eltwise/binary/bcast_h_sharded_optimised.cpp

**Fix**: New `BinaryOutputPolicy::L1Accumulate` — calls `llk_pack_reconfig_l1_acc(1)` before
pack and packs to the same slot repeatedly. Very hardware-specific; not a general streaming pattern.
**Complexity**: High — fundamentally different from streaming; close to impossible to abstract cleanly.

---

## sfpu_chain / sfpu_pipeline gaps

### ~~GAP-9: Runtime constant fill_tile in chain~~ — FIXED (commit `31f3de5460e`)
`FillScalar<Dst>` stores a runtime `float value` field; `FillConst<Bits, Dst>` handles
compile-time constants via `fill_tile_bitcast`. Both in `sfpu_math.hpp`.

**Remaining blockers for hardshrink** (even with FillScalar available):
- 3 `DestReuseOp` calls in one DEST window still require multiple FPU reinits per window;
  the current `sfpu_chain` calling convention re-inits only after the full chain. Need
  either per-element reinit support or manually managed windows.

---

### GAP-10: Multi-DST SFPU ops (3+ DST-slot arguments)
`logsigmoid_tile(dst0, dst1, dst_out)` and similar ops operate on 3 separate DEST slots.
`sfpu_chain` BinaryOp base handles 2-slot (In0, In1 → Out); TernaryOp handles 3-input.
But `logsigmoid_tile` takes two *input* DST slots plus an *output* DST slot with its own
API that doesn't match the CRTP base signature.

**Blocked kernels** (4):
- eltwise/unary/logsigmoid_kernel.cpp
- eltwise/unary_ng/logsigmoid_kernel.cpp
- eltwise/unary_ng/logit_kernel.cpp
- experimental/unary_backward/gelu_backward/eltwise_bw_gelu_poly.cpp

**Fix**: Add dedicated `Logsigmoid<DataSlot, AuxSlot>` and `Logit` chain elements wrapping
the 3-arg LLK calls directly (can't use `BinaryOp` CRTP base).
**Complexity**: Low per element — just wrapper structs with custom `call(a, b, out)`.

---

### GAP-11: Indexed Load (tile i from a pre-waited block)
`sfpu_pipeline` always uses `Load::cb_tile_idx = 0`. When the caller pre-waits N tiles
and wants to access tiles 0..N-1 across a batched acquire window, `cb_tile_idx` stays at 0
for every call.

**Blocked kernels** (4):
- eltwise/unary_backward/tanh_bw/eltwise_bw_tanh_deriv.cpp
- reduction/generic/reduce_h_neg.cpp
- moreh/moreh_norm*/moreh_norm_h/w_kernel.cpp (multiple)

**Fix**: `sfpu_pipeline` with `WaitUpfrontNoPop` Loads and batching support that increments
`cb_tile_idx` per batch step. Currently the batching loop calls `chain.apply(k * stride)` but
all `Load::exec()` calls use the same `cb_tile_idx = 0`. Needs the pipeline to pass the batch
index to Load so it can compute `cb_tile_idx = base + k`.
**Complexity**: Medium — requires `Load` to accept an external batch counter or a new `policy`
mode that auto-increments.

---

### ~~GAP-12: Missing `TanhDerivative` sfpu_chain element~~ — FIXED (commit `31f3de5460e`)
`TanhDerivative<Approx, Dst>` added to `sfpu_math.hpp`, wrapping
`tanh_derivative_tile_init<fast>()` and `tanh_derivative_tile<fast>(d0)`.

**Remaining blocker for tanh_bw**: GAP-11 (indexed Load) — the kernel pre-waits
`per_core_block_size` tiles from two CBs and accesses them at index i; `sfpu_pipeline`
currently always uses `cb_tile_idx=0` inside batched loops.

---

### GAP-13: `copy_dest_values` — copy one DEST slot to another
`copy_dest_values(src_slot, dst_slot)` moves a DST register value to another slot in-place.
No chain element wraps this. Needed in gelu_backward to save tanh before squaring it.

**Blocked kernels** (2):
- experimental/unary_backward/gelu_backward/eltwise_bw_gelu_approx_tanh.cpp
- experimental/unary_backward/gelu_backward/eltwise_bw_gelu_poly.cpp

**Fix**: Add `CopyDest<SrcSlot, DstSlot>` element (10 lines). Wraps `copy_dest_values(src, dst)`.
**Complexity**: Very low.

---

### GAP-14: Mid-chain-reinit after DestReuseOp followed by Load in same iteration
`chain_has_non_load_fpu_clash_v` triggers `copy_tile_to_dst_init_short` reinit at the START
of each sfpu_pipeline iteration. But if a DestReuseOp clobbers the MOP state and a Load
follows within the SAME `chain.apply()`, the Load's `copy_tile` runs under the wrong state.

**Blocked kernels** (2):
- eltwise/unary/hardshrink_kernel.cpp
- eltwise/unary_ng/hardshrink_kernel.cpp

**Fix**: In `SfpuChain::apply()`, detect when a non-Load FPU-clashing element precedes a Load
element and call `copy_tile_to_dst_init_short(CB)` inline between them. This requires each
Load to call its own `copy_tile_to_dst_init_short(CB)` inside `exec()` when the preceding
element is known to have clobbered the state — i.e., `Load::init()` should be non-trivial
when the chain contains a preceding DestReuseOp.
**Complexity**: Medium — needs compile-time chain analysis to identify the "DestReuseOp precedes
Load" pattern and conditionally emit the reinit.

---

## Structural — not fixable via helpers alone

### STRUCT-1: Matmul fused with eltwise
`matmul_tiles(...)` interleaved with `add_tiles`/`mul_tiles` in the same compute sequence.
Helpers are eltwise-only; matmul is a separate operation class.

**Blocked kernels** (8):
- matmul/bmm.cpp, bmm_large_block_zm.cpp, bmm_large_block_zm_fused_bias_activation.cpp
- experimental/matmul/attn_matmul/transformer_attn_matmul.cpp
- experimental/matmul/group_attn_matmul/transformer_group_attn_matmul.cpp
- conv/conv2d/compute_depthwise_conv1d.cpp
- conv/conv2d/conv_bmm_tilize.cpp
- experimental/transformer/fused_distributed_rmsnorm/rmsnorm_post_allgather.cpp (ROPE matmul)

**Resolution**: Not applicable — matmul requires dedicated helper or kernel restructuring.

---

### STRUCT-2: moreh *_to_cb helper layer
Moreh kernels already use `moreh_common.hpp` `*_to_cb` abstractions (add_tiles_to_cb,
mul_tiles_to_cb, etc.). Replacing one helper with another is churn with no value.

**Blocked kernels** (~15):
- moreh/moreh_adam.cpp, moreh_adamw.cpp, moreh_sgd.cpp
- moreh/moreh_matmul.cpp, moreh_dot.cpp, moreh_dot_backward.cpp
- moreh/moreh_mean*.cpp, moreh_sum*.cpp, moreh_mean_backward.cpp
- moreh/moreh_nll_loss_backward.cpp, moreh_sum_backward.cpp

**Resolution**: Leave as-is. The moreh helper layer serves the same purpose.

---

### STRUCT-3: Runtime bcast dispatch
`if (ht_need_bcast && wt_need_bcast) mul_tiles_bcast_scalar(...)` — selects between SCALAR,
ROW, COL, NONE broadcast at runtime. `binary_op<bcast_dim>` requires compile-time `BcastDim`.

**Blocked kernels** (3):
- moreh/moreh_norm_backward/moreh_norm_backward_kernel.cpp
- (some SSM/integral image kernels with similar dispatch)

**Resolution**: Write two separate helper calls in `if constexpr` branches, OR keep raw.
If the dispatch condition is a runtime argument, it cannot be expressed as a single `binary_op` call.

---

### STRUCT-4: Macro-parameterized SFPU ops (SFPU_OP_CHAIN_0, BINARY_SFPU_OP, etc.)
Kernel ops are parameterized via preprocessor macros, not C++ types. No C++ type is available
to instantiate a chain element.

**Blocked kernels** (~20):
- eltwise/binary/*.cpp (bcast_h/w/hw.cpp use BCAST_LLKOP)
- eltwise/binary_ng/eltwise_binary_sfpu*.cpp (BINARY_SFPU_OP)
- eltwise/ternary/ternary_sfpu*.cpp, ternary_addc_ops_sfpu*.cpp (TERNARY_SFPU_OP_FUNC)
- eltwise/unary/eltwise_sfpu.cpp, unary_ng/eltwise_sfpu.cpp (SFPU_OP_CHAIN_0)
- copy/typecast/eltwise_typecast.cpp (TYPECAST_LLK)

**Resolution**: Either change the parameterization from macros to C++ templates (significant
refactor of the op framework), or accept these as permanently macro-driven.

---

## Summary — kernels per gap

| Gap | Description | Kernels blocked |
|-----|-------------|----------------|
| GAP-1 | Absolute tile index on A/B | ~40 |
| GAP-2 | Cumulative wait policy | ~15 |
| GAP-3 | Non-sequential output pack | ~10 |
| GAP-4 | Asymmetric wait/process/pop | 9 |
| GAP-5 | In-place CB capacity=1 | 1 |
| GAP-6 | EltwiseBinaryType → BinaryOpType | 4 |
| GAP-7 | Self-feeding accumulator | 3 |
| GAP-8 | L1 accumulation pack | 2 |
| ~~GAP-9~~ | ~~Runtime fill_tile in chain~~ | ~~4~~ | **FIXED**: FillScalar + FillConst |
| GAP-10 | Multi-DST SFPU (3+ slots) | 4 |
| GAP-11 | Indexed Load in pipeline | 4 |
| ~~GAP-12~~ | ~~Missing TanhDerivative element~~ | ~~1~~ | **FIXED**: TanhDerivative |
| GAP-13 | Missing `CopyDest` element | 2 |
| GAP-14 | Mid-chain reinit after DestReuseOp+Load | 2 |
| STRUCT-1 | Matmul fused | 8 |
| STRUCT-2 | moreh *_to_cb layer | ~15 |
| STRUCT-3 | Runtime bcast dispatch | 3 |
| STRUCT-4 | Macro-parameterized ops | ~20 |

**High ROI gaps** (fix once, unblock many): GAP-1 (~40 kernels), GAP-2 (~15), STRUCT-4 (~20 if framework changes).
**Low-hanging fruit** (small fix, immediate payoff): GAP-12 (10 lines), GAP-10 (per-element wrappers).

# Kernel Migration Log — compute_kernel_lib

Running log of kernels evaluated for migration to `compute_kernel_lib` helpers
(`binary_op_helpers`, `sfpu_chain`, `reduce_helpers_compute`).

Format: one entry per kernel. Status ∈ { **MIGRATED**, **NOT-MIGRATED**,
**PARTIAL**, **DEFERRED** }. Reason always cites the specific pattern / LLK
feature that does or does not fit the helper API.

---

## MIGRATED

### `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`
- **Commit**: `a77de3b3793`
- **Stages migrated**: all 4 — add+rsqrt (PostOp), sub+DestReuseMul (PostOp), mul, add
- **Helper features used**: `add`, `sub`, `mul` with `BinaryInputBlockShape::single()`,
  `sfpu_chain(Rsqrt<>{})`, `sfpu_chain(DestReuseMul<cb, D0, NoWaitNoPop>{})`,
  `NoWaitNoPop` for persistent operands, `INPUT_AND_OUTPUT` reconfig
- **Required refactor**: templatized `batchnorm_bcast_tiles` on CB ids because
  `DestReuseMul<cb_den>` needs `cb_den` at compile time
- **Tests**: ~1,080 passed / 0 failed across 4 test functions (bf16 +
  fp32_dest_acc_en={False,True})

### `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp`
- **Commit**: `3c9cf9890ac`
- **Stages migrated**: stage 2 (add+rsqrt PostOp), stage 3 (mul COL bcast),
  stage 4 (mul ROW bcast), stage 5 (add ROW bcast). Stage 1 reduce was
  already helper-based.
- **Helper features used**: `add`/`mul` with `BroadcastDim::{NONE, COL, ROW}`,
  `WaitUpfrontPopAtEnd` for single-iteration persistent operand, `NoWaitNoPop`
  for across-iteration persistent operands, `sfpu_chain(Rsqrt<Legacy::On|Off>{})`
  for compile-time toggleable rsqrt
- **Tests**: 37 passed / 0 failed via `test_distributed_layernorm_post_allgather.py
  -k rmsnorm` (bf16 / bfloat8_b / mixed dtypes, 4/8 simulated devices)
- **Path verification**: `ttnn.rms_norm_post_all_gather` →
  `layernorm_post_all_gather_program_factory.cpp` (is_rmsnorm=true) →
  this kernel path (confirmed by grep of exact path string)

---

### `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp`
- **Commit**: `61fe380f934`
- **Stages migrated**: Stage 4 (final add) only
- **Helper features used**: `add` NONE bcast, `WaitUpfrontPopAtEnd` × 2, `BinaryOutputPolicy::Bulk`
- **Blocker for stages 2/3**: `mul_tiles(…, j + sin_cos_row_cnt * Wt, …)` in RELOAD_IMPL=0 mode
  — B-tile index advances by `Wt` per seq_tile iteration. No helper policy supports
  runtime-varying absolute B-tile offsets.
- **Tests**: 13 prefill passed (head_dim 64/128/96/256, n_heads variants)

### `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama_sharded.cpp`
- **Commit**: `64dac8f3798`
- **Stages migrated**: Stages 2-4 (sin_interim, cos_interim, final add)
- **Helper features used**: `mul` ROW bcast `WaitUpfrontPopAtEnd/NoWaitNoPop/Bulk` × 2,
  `add` NONE bcast `WaitUpfrontPopAtEnd` × 2 `Bulk`. sin/cos CBs pre-waited globally
  and reused across all `ht` iterations (`NoWaitNoPop`).
- **Tests**: 18 decode tests passed (head_dim 64/128/96/256, batch 1/15/32, with program cache)

### `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded.cpp`
- **Commit**: `443b5e61d40`
- **Stages migrated**: Stages 2-4 (sin_interim, cos_interim, final add)
- **Helper features used**: same as regular sharded above; sin/cos CBs provided by reader
  (NoWaitNoPop), runtime `in_cb`/`out_cb` from q/k selection — helpers take `uint32_t` OK
- **Tests**: 16 decode tests passed (head_dim 128/64/256, batch 1/8/16/32, with program cache)

### `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/compute/hardswish_kernel.cpp`
### `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/compute/tanhshrink_kernel.cpp`
### `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/compute/mish_kernel.cpp`
- **Commit**: `218105480b0`
- **Pattern**: same sfpu_chain + DestReuseOp/SfpuSub/SfpuMul as the `unary/` counterparts,
  but per-tile loop (output CB capacity=2 requires `SfpuBatching::Disabled`).
- **INP_FLOAT32**: `sfpu_chain(Load<D0,WaitNoPop>, [Load<D1,NoWaitPop>,] SfpuOp, SfpuMul/SfpuSub)`
- **INP_FLOAT**: `sfpu_chain(Load<D0,WaitNoPop>, SfpuOps, DestReuseOp<WaitAndPop>)`
- **Required fix from initial attempt**: `SfpuBatching::Auto` + `SfpuOutputPolicy::PerTile`
  deadlocked — Auto picked batch_size > CB capacity, held DEST while waiting for output
  space that the writer couldn't free. Fixed with `SfpuBatching::Disabled`.
- **Discovered by**: fresh independent investigation — `unary_ng` was not in the original plan
- **Tests**: 39 test_unary.py + 5 test_activation.py passed (bf16, fp32, approx modes)

### `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardswish_kernel.cpp`
- **Commit**: `d02a44f0c89`
- **Pattern**: `sfpu_chain(Load<WaitNoPop>, SfpuOp, DestReuseOp<WaitAndPop>)` — single CB
  read twice (once for SFPU, once via FPU dest-reuse to preserve BFloat16 precision path)
- **Helper features used**: `sfpu_chain` with `Load<cb_input, D0, WaitNoPop>`,
  `Hardsigmoid<D0>`, `DestReuseOp<cb_input, ELWMUL, DEST_TO_SRCA, D0, WaitAndPop>`.
  `sfpu_pipeline` handles CB lifecycle + output packing.
- **Required helper fix**: `sfpu_pipeline` called `copy_tile_to_dst_init_short` once before
  the loop; `DestReuseOp::exec()` runs `binary_dest_reuse_tiles_init` which clobbers that
  state for subsequent tiles. Added `chain_has_non_load_fpu_clash_v` trait to `sfpu_chain.hpp`
  and updated `sfpu_chain.inl` to re-call `copy_tile_to_dst_init_short` per tile when the
  trait is true. Zero behavior change for SFPU-only chains.
- **Discovered by**: fresh investigation outside the original plan — plan only looked at
  binary_op_helpers (two-CB streaming); these kernels are unary but use a terminal FPU
  dest-reuse step after SFPU ops, a different helper axis never explored in the plan.
- **Tests**: 14 hardswish tests passed (bf16, bfloat8_b, fp32, shapes)

### `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
- **Commit**: `d02a44f0c89`
- **Pattern**: same sfpu_chain + DestReuseOp pattern as hardswish
- **Helper features used**: `Load<WaitNoPop>`, `Tanh<Approx::Exact, D0>`,
  `DestReuseOp<cb_input, ELWSUB, DEST_TO_SRCB, D0, WaitAndPop>`.
  `DEST_TO_SRCB`: loads x into SRCB, computes SRCA(tanh(x)) - SRCB(x) → x - tanh(x).
- **Tests**: 7 tanhshrink tests passed (bf16, bfloat8_b, fp32)

### `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/mish_kernel.cpp` (INP_FLOAT path)
- **Commit**: `d02a44f0c89`
- **Pattern**: same sfpu_chain + DestReuseOp, longer SFPU chain (exp+log1p+tanh)
- **Helper features used**: `Load<WaitNoPop>`, `Exp<Fast|Exact>`, `Log1p<Fast|Exact>`,
  `Tanh<Exact>`, `DestReuseOp<cb_input, ELWMUL, DEST_TO_SRCA, D0, WaitAndPop>`.
  `use_approx` runtime flag dispatches to two statically-typed chains.
  INP_FLOAT32 path was already migrated separately (uses `SfpuMul` instead of DestReuseOp
  to preserve FP32-specific precision semantics).
- **Tests**: 25 mish tests passed (bf16+fp32, approx=True/False, multiple shapes)

### `ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/device/kernels/compute/layernorm_post_allgather_welford.cpp`
- **Commit**: `3b46be49921`
- **Stages migrated**: Stage 2 (x_minus_mean: sub COL bcast) only
- **Helper features used**: `sub` COL bcast, `WaitUpfrontPopAtEnd/NoWaitNoPop/Bulk`,
  `BinaryInputBlockShape::of(1, block_size)`. cb_stats_reduced tile 0 = mean,
  pre-waited by outer tile_row scope → NoWaitNoPop.
- **Blockers for remaining stages**:
  1. Stage 1 (rsqrt): `add_tiles(cb_stats_reduced, cb_eps, 1, 0, 0)` reads tile 1 (variance) — non-zero A tile index
  2. Stage 3 (normalize mul): in-place output when `do_gamma||do_beta` (output = cb_intermediate = input); pop-before-reserve requires CB capacity ≥ 2×block_size (unverified)
  3. Stages 4-5 (gamma/beta): `cb_wait_front(gamma, col_tile + block_size)` cumulative wait + non-zero B tile index `col_tile + i`
- **Tests**: 12 tp1 passed (tile/row_major, no_affine/with_affine, dim 2048/2432/3072/5120)
- **Path verification**: `dit_layernorm_post_all_gather_welford_program_factory.cpp` → this kernel

---

## NOT-MIGRATED

### `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather.cpp`
- **Status**: PARTIAL only (stages 3 and 4 migratable in isolation, not worth
  the churn without the harder stages)
- **Blockers**:
  1. Stage 2 (`E[x]²`): `mul_tiles(cb_stats_reduced, cb_stats_reduced, 1, 1, 0)`
     reads tile index **1** from the CB. `binary_op` under COL/NONE with
     same-CB (icb_a == icb_b) always reads tile 0. No `cb_tile_idx` param
     on B side of binary_op.
  2. Stages 5–8 use `chain_llk` — a DIFFERENT helper abstraction that
     composes LLK nodes. Replacing with `binary_op` chain-PostOps would
     require handling `fixed_CB_B_index = 1` on the x_minus_mean node (same
     non-zero tile index issue) and the multi-stage fused bcast pipeline
     that `chain_llk` already abstracts.
- **Required helper additions to unblock**: runtime `cb_tile_idx` for the
  CB-side B operand in `binary_op` (similar to what `DestReuseOp` has for
  its CB), OR a dedicated `mul_tile_at<index>` primitive.

### `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp`
- **Blockers**:
  1. **Cumulative wait** — `cb_wait_front(cb_inp, wt + blk)` grows each chunk
     iteration. No `BinaryInputPolicy` value waits cumulatively; options
     either wait upfront (all Wt) or per-tile/per-chunk (fixed count).
  2. **Non-zero tile index access** — `mul_tiles(cb_inp, cb_inp, wt+wtr, wt+wtr, wtr)`
     reads from absolute tile positions in cb_inp (not just tile 0).
  3. **Non-sequential output pack** — `pack_tile(wtr, cb_x2, wt + wtr)` with
     an absolute output tile index. The helper's output packing is
     sequential (per-tile at relative offset, per-chunk with `tiles_processed`
     counter). No per-tile caller-supplied absolute output index.
- **Workaround considered**: WaitUpfrontNoPop on A side + NoWaitNoPop on B
  side + absolute tile_a indexing — the helper does support absolute
  `tile_a = ht*Wt + wt_base + wt` for WaitUpfront policies. But the
  non-sequential pack remains unfixable, AND the raw code's output pattern
  (reserve `blk` per iter, pack to `wt + wtr`) looks semantically off; I
  don't want to rewrite it without understanding why it works in production.

### `ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather.cpp`
- **Blockers**:
  1. **ROPE fusion** — matmul + rotation ops interleaved with the binary
     sequence. Helpers are eltwise-only; `matmul_tiles` is out of scope.
  2. **In-place CB writes with capacity=1** — the rsqrt stage (stage 2) reads
     `reduce_result_cb`, then pops it BEFORE reserving the same CB as output.
     `reduce_result_cb_num_tiles = 1` in the program factory. The helper's
     reserve-before-pop order would deadlock on a capacity-1 CB.
  3. **In-place CB writes (ROPE)** — `mul_tiles(intermediate_cb, rope_cos_cb, i, idx, i)`
     followed by `cb_pop_front(intermediate_cb, block_size); cb_reserve_back(intermediate_cb, block_size)`.
  4. **Variable B-tile index** — `mul_tiles(intermediate_cb, rope_cos_cb, i, rope_cos_tile_in_head, i)`
     where `rope_cos_tile_in_head` is a runtime-varying index that resets
     per head.
  5. **Weight fusion cumulative wait** — `cb_wait_front(weight_cb, col_tile + block_size)`
     same cumulative-wait pattern as rmsnorm_pre_allgather.

### `ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_pre_allgather.cpp`
- **Blockers**:
  1. **L1 accumulation pack** — `pack_tile<true>(i, intermediate_cb, 0)` plus
     `llk_pack_reconfig_l1_acc(1)` repeatedly packs DST tiles onto the
     **same** output tile slot, accumulating in L1. Helper's pack path is
     purely sequential, never re-packs to the same output tile.
  2. The entire accumulate-into-single-tile pattern is fundamentally
     different from eltwise streaming; it's a write-back fused with pack
     reconfig.
- **Conclusion**: not a standard eltwise pattern; helper won't fit without
  a new "packed accumulator" output policy.

### `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduction.cpp`
(Representative of the entire **Batch B** CCL reduction family — 13 files
share this pattern.)
- **Blockers**:
  1. **Asymmetric wait/pop vs processed count** — `cb_wait_front(..., tile_granularity)`
     and `cb_pop_front(..., tile_granularity)` every iteration, but the
     middle `for` loop only processes `min(tiles_remaining, tile_granularity)`
     tiles. On the last iteration, the helper's symmetric
     `wait(shape) == pop(shape)` contract would over-process or
     under-pop.
  2. Contract between reader and compute kernel: reader appears to pad to
     multiples of `tile_granularity`; compute pops the padding without
     processing. Baking the helper into this requires the helper to
     support "wait N, process M, pop N" where M ≤ N is runtime-varying.
- **Workaround considered**: always pass `shape::of(1, tile_granularity)`
  and let the helper wait/process/pop `tile_granularity` unconditionally.
  This would force the reader to only push meaningful tiles and change
  the producer-consumer contract. Scope would bleed beyond just the
  compute kernel.

### `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded_row_major.cpp`
- **Blockers**:
  1. Processes only 1 tile per loop iteration but does `cb_push_back(out_cb, Wt)`,
     `cb_pop_front(sin_interm_cb, Wt)`. The "wait Wt, process 1, pop Wt" pattern
     (Wt=1 in practice) doesn't map cleanly to any helper shape + policy combo.
  2. Low value: Wt=1 is likely the only real config, making this kernel trivial.

### `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp`
- **Blockers**:
  1. **Cumulative wait** — `cb_fused_attn_obj.wait_front(wt + ndst)` inside
     the streaming loop (same pattern as rmsnorm_pre_allgather).
  2. **Non-zero B tile index** — `add_tiles_bcast_rows(cb_scale_mask, cb_fused_attn, wt8, wt + wt8, wt8)`
     reads B at absolute `wt + wt8` across iterations.
  3. **Mixed SFPU+FPU in one DEST window** — `add_tiles(...)` then
     `exp_tile(...)` then `pack_tile(...)` interleaved with
     `cb_scale_mask_obj.pop_front(ndst)` mid-window. `reduce(...)` with an
     sfpu_chain lambda-style PostOp already handles the reduce-with-recip
     piece; rest of softmax is not cleanly expressible.
  4. Multiple `#ifdef` branches (FUSED_SCALE_MASK / CAUSAL_MASK /
     NUMERIC_STABLE / mask_padded_data) with subtly different flow.
- **Partial migration possible** only for the reduce (already migrated
  upstream) and maybe the final `mul_bcast_cols`, but the non-zero B-tile
  index blocker applies there too.

### `ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/accumulation_compute.cpp`
- **Blockers**:
  1. **Runtime op selection** — `if (accumulation_op == CUMPROD) mul_tiles(...); else add_tiles(...);`
     dispatches at runtime. Helper's binary_op takes compile-time `BinaryOpType`.
     Per HQ migration guidance this could be handled with two helper calls
     in branches, BUT:
  2. **Self-feeding accumulator** — each iteration's output is packed to
     cb_acc, then IMMEDIATELY `cb_wait_front(cb_acc, 1)` + `copy_tile` +
     `pack_tile(..., cb_out)` inside the same acquire window. The enable_reload
     ternary switches the B operand between cb_start (first iter) and cb_acc
     (rest). The pack-wait-copy-pack pattern doesn't fit binary_op's
     single-output contract.
  3. **CUMSUM_USE_INT32 `#ifdef` path** — int32-specific `add_int_tile(INT32_TILE_DEST, INT32_TILE_ACC, INT32_TILE_DEST)`
     is a DEST-to-DEST SFPU op operating on pre-loaded DEST slots, not a
     CB-based binary_op.

### `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded_post_allgather.cpp`
- **Blockers**:
  1. **Absolute tile index on A operand** — `sub_tiles_bcast_cols(cb_in0, cb_ex_global, index, 0, w)`
     where `index = w + index_subblock_w_offset` reads arbitrary tile
     positions in cb_in0. Same as the "runtime cb_tile_idx on A side"
     that no current policy supports; the helper's `waits_upfront` gives
     `tile_a = ht*Wt + wt_base + wt` which is sequential, not arbitrary.
  2. **Variable chunk/subblock iteration** — nested `(block_h, num_subblocks_w, subblock_w)`
     loops pack absolute indices; each inner iteration has its own
     acquire/release around a `subblock_w`-tile DEST window.
  3. Stage 1 rsqrt (the one stage that matches batch_norm's pattern)
     is gated behind `is_allgather_worker && enable_sqrt` — partial
     migration of just that stage is possible but low value.
  4. Pre-existing `layernorm_binary_helpers_feasibility.md` in this dir
     already concluded these same issues for the sister `layernorm.cpp`.

---

### `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/compute/rms_compute.cpp`
- **Blockers**:
  1. FUSE_PRE_ADD stage: `add_tiles(cb_in0, cb_in1, index, index, w)` with `index = w + offset + h_offset` — absolute non-sequential tile indexing for both A and B.
  2. X^2 stage: same absolute indexing pattern.
  3. mul_bcast_cols/rows gamma stages: absolute tile indices.
  Only stage 5 (rsqrt: add+rsqrt inside `is_allgather_worker && enable_sqrt`) is migratable in isolation but low value (4 raw lines inside a runtime-gated branch).

### `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather_2d.cpp`
- **Blockers**:
  1. X^2 stage: cumulative wait + non-sequential pack (same as rmsnorm_pre_allgather.cpp).
  2. merge_core section: `add_tiles(cb_x2_merge, cb_zero, i, 0, dst0)` accumulates tiles 0..num_cores_y-1 into single DST slot — not a CB streaming pattern.
  Stage 2 (reduce): already helper-based.

### `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather.cpp`
- **Blockers**:
  1. X^2 stage: cumulative wait + non-sequential pack (same as rmsnorm_pre_allgather.cpp).
  Reduce stages already helper-based. No other stages.

### `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather_welford.cpp`
- **Blockers**:
  1. rsqrt: `add_tiles(cb_stats_reduced, cb_eps, 1, 0, 0)` — reads tile 1 (variance) from A; non-zero A tile index.
  2. chain_llk stages: `fixed_CB_B_index = 1` on x_minus_mean node (same non-zero B tile index).
  No migratable stages beyond what's already done in the non-welford version.

### `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_h.cpp` (and `bcast_w.cpp`, `bcast_hw.cpp`)
- **Blockers**:
  1. `BCAST_LLKOP` is an `EltwiseBinaryType` preprocessor macro; helper uses `BinaryOpType` enum. No reverse mapping exists without helper API change or `#if` chains.
  The kernels are compiled once per op type with different defines — migration requires either a `BinaryOpType` macro or a map from `EltwiseBinaryType` to `BinaryOpType`.
  Low priority: kernels are already 10-15 lines and simple.

### `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_h_sharded_optimised.cpp`
- **Blockers**:
  1. Same `BCAST_LLKOP EltwiseBinaryType` → `BinaryOpType` mismatch.
  2. `pack_tile<true>(htr, cb_out, current_index)` — L1 accumulation pack with absolute output tile index — same L1-acc blocker as rmsnorm_pre_allgather.cpp.

### moreh optimizer/softmax kernels (`moreh_adam.cpp`, `moreh_adamw.cpp`, `moreh_sgd.cpp`, `moreh_norm_backward_kernel.cpp`, etc.)
- **Blockers**:
  1. Already use moreh-specific `*_to_cb` helpers from `moreh_common.hpp`. Replacing one helper layer with another is not a migration.
  2. `moreh_norm_backward_kernel.cpp` has runtime bcast dispatch (`if (ht_need_bcast && wt_need_bcast)` etc.) — cannot express in a single helper template call.

---

### Layernorm sharded / full family (bulk entry)
**Files** (all share the same blockers):
- `layernorm_sharded.cpp` — 5 stages
- `layernorm_sharded_pre_allgather.cpp` — 3 stages
- `layernorm_sharded_welford.cpp` — 4 stages
- `layernorm.cpp` — 6 stages
- `layernorm_large_tensor.cpp` — 6 stages
- `layernorm_welford.cpp` — 6 stages
- `layernorm_large_tensor_welford.cpp` — 9 stages
- `layernorm_distributed/layernorm_pre_allgather_welford.cpp`
- `layernorm_distributed/layernorm_pre_allgather_2d.cpp`
- `groupnorm.cpp` / `groupnorm_sharded_v2.cpp` / Welford variants

**Shared blockers**:
1. **Absolute tile indexing in subblock inner loops** — `index = w + index_subblock_w_offset + index_h_offset` (or similar) everywhere in normalization stages. Affects both A and B access.
2. **Cumulative wait** in pre_allgather variants — `cb_wait_front(cb_in, wt + blk)` pattern.
3. **Non-sequential output pack** — `pack_tile(i, cb_out)` with `i` not equal to 0 in per-chunk pattern.

These are structural to how the blocked sub-tile computation works across all non-trivial normalization kernels. The helper needs runtime `cb_tile_idx` support on both A and B sides (Helper gap #1 from recurring blockers section) to address them.

### `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/compute_depthwise_conv1d.cpp`
- **Blockers**:
  1. `eltwise_mul_and_add_block_v2`: runtime conditional `if (idx == 0)` dispatches between copy-tile-init and add-tiles-init — different ops in same loop. Cannot express in single helper template.
  2. `out_cb` is read AND written (accumulate-then-copy pattern) — self-feeding accumulator blocker.
  3. Fused with matmul and transpose_wh (FUSE_BIAS variant).

---

## DEFERRED (evaluated, migration possible, deprioritized)

*(none yet)*

---

## Patterns that unblock many kernels

Recurring blockers that, if addressed in the helper, would unlock multiple
kernels at once. Ordered by estimated impact:

1. **Runtime `cb_tile_idx` for `binary_op`'s B operand** (and maybe A) —
   unblocks `layernorm_post_allgather` stage 2, possibly `rmsnorm_pre_allgather`
   B-side access patterns. `DestReuseOp` already has this field; extending
   to `binary_op`'s regular path is mechanical.

2. **Cumulative-wait policy** (`WaitCumulativeNoPop` or similar) —
   unblocks `rmsnorm_pre_allgather` cb_inp access and `fused_distributed_rmsnorm_post_allgather`
   weight fusion. Raw pattern: wait for increasing tile counts each
   iteration (caller pops once at the end).

3. **Non-sequential output pack / caller-supplied output index** —
   unblocks `rmsnorm_pre_allgather` output, `groupnorm.cpp`, `compute_depthwise_conv1d.cpp`.
   Hard because it breaks the helper's "ocb = sequential FIFO" assumption.

4. **Asymmetric wait/process/pop** — unblocks all 13 Batch B CCL
   reductions with one helper extension. Pattern: `wait N, process M≤N, pop N`.

# Normalization kernel migration status

Tracking the per-kernel migration to `compute_kernel_lib::eltwise_chain` on this
branch (`astancov/eltwise_vx_migration`). Source-of-truth catalog is
`pack_patterns.tsv` (commit `0888981c2285da8e9e29e876bca65dd5513cd26b` on
`astancov/eltwise_run7_refined_rebase_v2`); this doc only tracks the 30
kernels under `ttnn/cpp/ttnn/operations/normalization/`.

## Status overview

| Status | Count |
|---|---|
| Migrated | 4 |
| Easy (single FPU pattern matches batch_norm) | 0 |
| Medium (chunked-output or held-bcast operands) | ~9 |
| Hard (mixed SFPU DEST-DEST + manual `last_srca_cb` threading) | ~5 |
| Out-of-scope under current taxonomy (welford / transpose / raw reduce_tile) | ~8 |
| Structural anomaly (semantic ambiguity) | 1 |
| Shared headers / not actual kernels | 3 |

## Per-kernel status

### Migrated

| Kernel | Sites | Commit | Notes |
|---|---|---|---|
| `batch_norm/.../batch_norm_kernel.cpp` | 4 | `17056fe08cb` | Template on `<WeightHas, BiasHas>`. Stage 1 = 1-tile BinaryFpu<Add>+Rsqrt+PackTile. Stages 2-4 fused via DestReuseBinary; cb_tmp_1 staging eliminated. 20/20 PASS on `test_batch_norm`. |
| `layernorm_distributed/.../layernorm_pre_allgather.cpp` | 1 | `189ef03f223` | Squaring chain over `EltwiseShape::of(Wt/blk, blk)`. cb_inp = HeldCumulative on Block (same-CB BinaryFpu<Mul>); cb_x2 = OutChunked on Block. Verified bit-exact (max diff 0.000000) against original via single-device probe; full pytest needs 4 PCIe devices. |
| `rmsnorm_distributed/.../rmsnorm_pre_allgather.cpp` | 1 | `c41aca20b88` | Same squaring chain as `layernorm_pre_allgather`. Only difference: explicit `cb_pop_front(cb_inp, Wt)` post-reduce vs. layernorm's second reduce. Verified bit-exact against original; full pytest needs 4 PCIe devices. |
| `rmsnorm_distributed/.../rmsnorm_post_allgather.cpp` | 4 | _pending_ | Full migration. Stage 1 (add(var,eps)+rsqrt) onetile chain (Streaming A / CallerManaged B / Rsqrt / OutStreaming). Stages 2-4 each one `eltwise_chain<blk>(Wt, ...)` call — chain owns A-side Bulk wait/pop-at-end, B-side CallerManaged (caller waits cb_recip_sqrt_var per ncht, cb_gamma/cb_beta once), pack-side OutBulk reserve/push-at-end. B index = Scalar for cb_recip_sqrt_var (col-bcast pinned at 0), Block for cb_gamma/cb_beta (row-bcast walks 0..Wt-1). ACQ/REL helpers + dead `onetile`/`dst0` locals removed. Verified 36/36 PASS on `test_distributed_layernorm_post_allgather.py::test_layernorm_part_2_with_program_cache` (rmsnorm subset — full bf16/bf8/mixed × 3 shapes × {4,8} devices × {fp32 on,off}). |

### Bit-exact-probe verification pattern

For kernels whose pytest needs hardware we don't have locally (multi-chip /
multi-PCIe), use this two-step probe protocol:

1. Before migration: probe op end-to-end on a single device, save the kernel's output via `torch.save`.
2. After migration: re-probe with the same seed/shape, load saved output, compare with `(out - orig).abs().max()`.

A bit-exact zero-diff result is strong evidence the chain's CB/DEST/dtype-reconfig emission matches the pre-migration LLK sequence. Used to validate the two `pre_allgather` migrations above without 4-PCIe hardware.


### Medium difficulty (next migration targets)

| Kernel | Sites | Pattern | Blocker(s) |
|---|---|---|---|
| `layernorm_distributed/.../layernorm_post_allgather.cpp` | 3 + chain_llk | Stage A: E[x]² = mul(stats_reduced[1], stats_reduced[1]) → cb_mean_squared. Stage B: var = sub(stats_reduced[0], mean_squared) → cb_var. Stage C: rsqrt(var + eps) → cb_recip_sqrt_var. Then chain_llk template for the 4 final stages. | The 3 1-tile FPU stages map cleanly to BinaryFpu+Rsqrt+PackTile (same as batch_norm Stage 1). The final 4 stages use the `chain_llk` template, which is itself a hand-rolled mini-chain — migrating those means replacing chain_llk callers with `eltwise_chain` directly. Practical PARTIAL migration: stages A/B/C, leave chain_llk. |
| `layernorm/.../layernorm_sharded_pre_allgather.cpp` | 2 | Subblock pattern: process `subblock_w` tiles per tile_regs window. Pre-add FPU add + squaring FPU mul (cb_in × cb_in). | Chain element BlockSize / subblock support exists; needs `EltwiseShape::of(block_h, block_w)` with subblock dispatch. Both pack sites are inside the per-block loop. |
| `layernorm_distributed/.../layernorm_pre_allgather_2d.cpp` | 2 | Squaring (mul cb_inp × cb_inp) with cumulative wait + chunked pack. + merge-core final-sum accumulate. | Squaring: chain `BinaryFpu<Mul, same-cb>` with HeldCumulative + OutChunked. Merge-core: DEST-accumulating add_tiles (acc_to_dest=true) — chain elements don't support DEST-accumulating binary, would need a new helper. |
| `rmsnorm_distributed/.../rmsnorm_pre_allgather_2d.cpp` | 2 | Same shape as layernorm_pre_allgather_2d (squaring + merge-core). | Same blockers as layernorm_pre_allgather_2d. |
| `layernorm/.../layernorm.cpp` | 7 | Multi-stage layernorm: variance via E[(x-mean)²], rsqrt(var+eps), x*recip, [gamma], [beta]. Uses raw ACQ/REL macros. | Each stage maps to known chain patterns. ~3 distinct stage types repeated across the loop. |
| `layernorm/.../layernorm_large_tensor.cpp` | 9 | Same as layernorm.cpp but with multi-block tiling. | Same patterns as layernorm.cpp at larger scale. |
| `layernorm/.../layernorm_sharded.cpp` | 7 | Sharded variant of layernorm.cpp. | Same stage patterns; sharded memory layout doesn't affect compute kernel. |
| `layernorm/.../layernorm_sharded_post_allgather.cpp` | 7 | Same shape as layernorm_post_allgather but sharded. | Same blockers as layernorm_post_allgather. |
| `softmax/.../softmax_sharded.cpp` | 5 | Sharded softmax. | Same patterns as softmax.cpp but with subblock dispatch. |

### Hard difficulty (SFPU DEST-DEST without DestReuseBinary equivalent)

| Kernel | Sites | Pattern | Blocker |
|---|---|---|---|
| `batch_norm/.../batch_norm_sfpu_kernel.cpp` | 5 | Structurally identical to batch_norm_kernel.cpp, but SFPU DEST-DEST ops (`add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`) + output typecast + manual `last_srca_cb` threading. | Chain has `AddBinary`/`SubBinary`/`MulBinary` SFPU elements (in `eltwise_binary_sfpu.hpp`) but **no SFPU equivalent of `DestReuseBinary`** — i.e. no element shaped like "load CB → D1, run SFPU binary against running D0". Stage 2 (sub→mul fusion) cannot be expressed as one chain — would need separate chains, defeating the fusion benefit. Helper gap: `DestReuseBinarySfpu` (load CB into one DEST slot, SFPU-binary-op against another DEST slot in same chain). |
| `batch_norm/.../running_statistics_sfpu_kernel.cpp` | 11 | Same SFPU DEST-DEST pattern as batch_norm_sfpu, plus `maybe_typecast_stat` helper that runs a separate compute+pack chain per iteration. Conditional mean / var paths each with 4 inner stages. | Same helper gap. |
| `groupnorm/.../groupnorm.cpp` | 10 | Group-wise statistics + normalize + gamma/beta. | Mixed FPU/SFPU + sharded across cores; structural complexity beyond a single migration session. |
| `groupnorm/.../groupnorm_sharded_v2.cpp` | 11 | V2 sharded groupnorm. | Same as above. |
| `softmax/.../softmax.cpp` | 6 | Multi-path softmax: FUSED_SCALE_MASK / NUMERIC_STABLE / CAUSAL_MASK conditional. `ndst`-wide subblocks. Cumulative wait on cb_fused_attn. | Each ifdef path is its own audit. Chain supports the patterns individually but the per-path wiring is non-trivial. |
| `softmax/.../softmax_large_tensor.cpp` | 6 | Same shape as softmax.cpp at large-tensor scale. | Same as softmax.cpp. |

### Out of scope under current taxonomy

Per `eltwise_taxonomy.md` §"Deliberate exclusions", these stages cannot be
expressed in the chain and must stay on raw LLK (partial-migrate the
surrounding eltwise stages, document blocker):

| Kernel | Sites | Why OOS |
|---|---|---|
| `layernorm/.../layernorm_welford.cpp` | 10 | Welford accumulator: in-DEST persistent accumulator across iterations. Orthogonal to chain's per-iter dst-sync window. |
| `layernorm/.../layernorm_large_tensor_welford.cpp` | 15 | Welford + transpose stages. Both OOS. |
| `layernorm/.../layernorm_sharded_welford.cpp` | 8 | Welford + sharded. |
| `layernorm_distributed/.../layernorm_post_allgather_welford.cpp` | 1 | Welford. |
| `layernorm_distributed/.../layernorm_pre_allgather_welford.cpp` | 4 | Welford. |
| `groupnorm/.../welford_groupnorm.cpp` | (not in TSV) | Welford accumulator + transpose stage are OOS. Per-group rsqrt and trailing single-tile stages (group-accumulate, gamma, beta, final cb_x -> cb_out) migrated in `92a19132dd7` + `20ce6da381c`. |
| `groupnorm/.../welford_groupnorm_sharded_v2.cpp` | (not in TSV) | Same scope as non-sharded welford_groupnorm. Per-group rsqrt + trailing stages migrated in `66907980ee9` + `20ce6da381c`. |
| `kernel_util/.../combine_welford.h` | 2 | Welford fused/shared header. |

### Structural anomaly

| Kernel | Sites | Issue |
|---|---|---|
| `batch_norm/.../running_statistics_kernel.cpp` | 1 | Outer `tile_regs_acquire` + `pack_tile(0, cb_out0)` window contains no compute targeting DST[0] — inner moreh helpers (`sub_tiles_to_cb`, `mul_tiles_to_cb`, `add_tiles_to_cb`) each open their own `tile_regs` window. The outer pack writes whatever DST[0] holds after the last inner macro's release. Either intentional (last-computed running stat acts as cb_out0 output) or stale code from a refactor. Migration requires semantic clarification: is cb_out0 supposed to mirror the last updated stat, or should the outer window be removed? |

### Shared headers / not migration targets

| File | Notes |
|---|---|
| `layernorm_distributed/.../chain_llk.hpp` | Hand-rolled mini-chain template used by `layernorm_post_allgather.cpp` and `layernorm_post_allgather_welford.cpp`. Migrating its callers to `eltwise_chain` would let this file be deleted. |
| `kernel_util/.../numeric.h` | Numeric helper inline functions, not a compute kernel. |
| `kernel_util/.../pre_add.h` | `pre_add::one_row` helper used by pre_allgather kernels. Internally uses `add_tiles_to_cb` style. Migrating this helper would also migrate every caller. |

## Cross-op migration attempts

| Kernel | Attempt | Outcome |
|---|---|---|
| `ttnn/.../eltwise/unary/.../tanhshrink_kernel.cpp` | Attempted full migration of both INP_FLOAT (FPU DestReuse) and INP_FLOAT32 (SFPU fan-out + SubBinary) paths. | **Reverted** — INP_FLOAT32 path failed `test_unary_tanhshrink_ttnn` with Max ATOL Delta ~197. Suspected issues with the fan-out CopyTile<D1,HeldStream> + CopyTile<D0,NoWaitPop> dedup + same-tile-twice access pattern, or SubBinary signature mismatch. Needs deeper triage (likely a chain-emission bug or wrong lifecycle/index combo for the SFPU-fan-out + SubBinary pattern). Bfloat16 / bfloat8_b paths (INP_FLOAT) did NOT get tested before revert. |

## Helper gaps surfaced during this cycle

1. **`DestReuseBinarySfpu`**. Chain `DestReuseBinary` is FPU-only. SFPU
   kernels that fuse "load CB into D1, run SFPU DEST-DEST binary against
   running D0" (i.e. `add_binary_tile` / `sub_binary_tile` /
   `mul_binary_tile` with one operand sourced from D0 left by a prior
   element) have no chain element today. Five+ catalog kernels affected
   (batch_norm_sfpu_kernel, running_statistics_sfpu_kernel, anything else
   using the moreh SFPU pattern).

2. **DEST-accumulating BinaryFpu**. The merge-core stage of
   `*_pre_allgather_2d.cpp` uses `add_tiles_init(a, b, acc_to_dest=true)`
   to accumulate into DEST across a runtime-bounded loop. Chain elements
   are non-accumulating; would need a new variant.

3. **chain_llk substitution playbook**. `layernorm_distributed/chain_llk.hpp`
   composes 1-4 LLK ops into a fused chain at compile time via `LLK_Node`
   structs. The chain helper expresses this differently (per-element
   structs in a parameter pack). Replacing chain_llk callers needs a
   one-to-one node→element mapping that documents which `fixed_CB_B_index`
   semantics map to which chain InputLifecycle (`0xFFFF` = stream
   per-tile pop; `0xDDDD` = held; literal = pinned at that index).

## Suggested migration order (resume from here)

1. ~~`layernorm_distributed/layernorm_pre_allgather.cpp`~~ — DONE (`189ef03f223`).
2. ~~`rmsnorm_distributed/rmsnorm_pre_allgather.cpp`~~ — DONE (`c41aca20b88`).
3. ~~`rmsnorm_distributed/rmsnorm_post_allgather.cpp`~~ — DONE. Bulk + CallerManaged + OutBulk pattern with B index walking 0..Wt-1 (Block) or pinned at 0 (Scalar). Full migration (4 sites) — no PARTIAL needed.
4. `layernorm_distributed/layernorm_post_allgather.cpp` (3 sites +
   chain_llk callers) — PARTIAL migration; chain_llk callers blocked
   pending gap #3 resolution.
5. `layernorm/layernorm.cpp` (7 sites) — repeating per-stage patterns,
   once the post_allgather wiring is validated this generalizes.
6. `softmax/softmax.cpp` — biggest open-ended cleanup.
7. SFPU kernels gated on helper gap #1 resolution.

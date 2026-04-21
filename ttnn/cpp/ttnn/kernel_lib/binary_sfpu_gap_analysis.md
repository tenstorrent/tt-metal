# Binary + SFPU Chain: Missing Features Gap Analysis

## TL;DR

Phase 1 is already fully implemented — `SRCA_ONLY`, `SRCB_ONLY`, `SRCA_ONLY_AND_OUTPUT`, and `SRCB_ONLY_AND_OUTPUT` are declared in `binary_op_helpers.hpp` (lines 128–141) and handled in `binary_op_helpers.inl` (lines 31–45, 257–263). The migration plan's stated Phase 1 blocker does not exist. The real blockers for the ~25 Tier 2 normalization files are three distinct helper-fixable gaps: (1) the `reconfig_data_format_srca`/`_srcb` signatures used in the `.inl` are single-argument but the raw kernels use two-argument forms that supply a "previous" CB identity; (2) several normalization kernels use non-zero B tile offsets and cumulative CB waits that have no policy equivalent; and (3) `batch_norm_kernel.cpp` uses `binary_dest_reuse_tiles` — a dest-aliasing fused multiply that the helper cannot model. The moreh kernels (Adam, SGD) have a structural incompatibility with `moreh_common.hpp`'s stateful `*_tiles_to_cb` wrappers that requires shim work, not a helper feature addition.

---

## Phase 1 Status

**CONFIRMED IMPLEMENTED end-to-end.**

Evidence:

- `binary_op_helpers.hpp` lines 128–141: all eight `BinaryDataFormatReconfig` enum values including `SRCA_ONLY = 4`, `SRCB_ONLY = 5`, `SRCA_ONLY_AND_OUTPUT = 6`, `SRCB_ONLY_AND_OUTPUT = 7` are declared.
- `binary_op_helpers.inl` lines 31–45: helper functions `reconfig_srca()`, `reconfig_srcb()`, `reconfig_output()` correctly include all four partial variants in their predicates.
- `binary_op_helpers.inl` lines 257–263: the `.inl` body calls `reconfig_data_format_srca(icb_a)` when `reconfig_srca(reconfig)` is true but `reconfig_srcb(reconfig)` is false, and `reconfig_data_format_srcb(icb_b)` symmetrically.

**Phase 1 exit criteria are all met.** The migration plan's Phase 1 section should be marked DONE.

---

## Summary Table

| # | Gap | Files unblocked | Complexity | Priority |
|---|-----|-----------------|------------|----------|
| 1 | `reconfig_data_format_srca/srcb` single-arg form | NOT A GAP — CONFIRMED RESOLVED | — | — |
| 2 | No cumulative-wait policy; non-zero B tile start offset | `rmsnorm_pre_allgather*.cpp` (3–4 files) + rotary embedding partial | Medium | MEDIUM |
| 3 | `binary_dest_reuse_tiles` — dest-aliased fused multiply | `batch_norm_kernel.cpp` Stage 2 only | Low | RESOLVED |
| 4 | `moreh_common.hpp` `*_tiles_to_cb` wrapper incompatibility | 12 moreh files (Adam, SGD, softmax, etc.) | Policy decision + shim work | MEDIUM |
| 5 | Copy-or-add conditional dispatch + random-access pack | `groupnorm*.cpp` (4 files), `compute_depthwise_conv1d.cpp` (unfixable) | N/A | N/A |
| 6 | `SfpuMax` SFPU struct missing | `moreh_adam.cpp` AMSGRAD path | Low (~15 lines) | RESOLVED |

---

## Gap 1: Single-argument `reconfig_data_format_srca/srcb` — CONFIRMED NOT A GAP

### Status: CONFIRMED RESOLVED

`tt_metal/hw/inc/api/compute/reconfig_data_format.h` exports both overloads:

- `reconfig_data_format_srca(uint32_t srca_new_operand)` — single-arg, unconditional reconfig (line 52)
- `reconfig_data_format_srca(uint32_t srca_old_operand, uint32_t srca_new_operand)` — two-arg, skips if format unchanged (line 68)
- Same for `reconfig_data_format_srcb` (lines 84 and 100).

The `.inl` correctly uses the single-arg form (`binary_op_helpers.inl` lines 260, 262). The single-arg form unconditionally reconfigures, which is slightly less optimal than the two-arg skip-if-same form used in raw kernels — but it is correct and compiles cleanly.

The raw kernels' two-arg calls (e.g., `reconfig_data_format_srca(cb_stats_reduced, cb_ex2)` at `layernorm_sharded_post_allgather.cpp:174`) are a performance hint, not a correctness requirement. The helper's single-arg form is safe for all Tier 2 migrations.

**No library change required. Gap 1 is eliminated.**

---

## Gap 2: Cumulative CB wait and non-zero B tile start offset

### What's missing

A `BinaryInputPolicy` variant that waits for `wt + blk` tiles cumulatively (not just `blk` per chunk), plus the ability to address B tiles at a non-zero offset.

### Kernels blocked

- `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp`
- `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather_2d.cpp`
- `ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_pre_allgather.cpp`
- (Rotary embedding non-sharded variant partially — B index `j + sin_cos_row_cnt * Wt`)

### Evidence from kernel code

`rmsnorm_pre_allgather*.cpp` (pattern quoted in migration plan, lines 811–824):
```cpp
mul_tiles_init(cb_inp, cb_inp);
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    cb_wait_front(cb_inp, wt + blk);   // CUMULATIVE: grows each iteration
    cb_reserve_back(cb_x2, blk);
    ACQ();
    for (uint32_t wtr = 0; wtr < blk; wtr++) {
        mul_tiles(cb_inp, cb_inp, wt + wtr, wt + wtr, wtr);  // NON-ZERO START INDEX
        pack_tile(wtr, cb_x2, wt + wtr);                     // NON-ZERO OUTPUT INDEX
    }
    REL();
    cb_push_back(cb_x2, blk);
}
```

The helper's `WaitUpfrontNoPop` or `WaitAndPopPerChunk` cannot express cumulative waiting (the pattern is: wait for `blk` more each time, keeping all prior tiles accessible).

### Proposed addition

Option A (minimal): add `WaitCumulativeNoPop` to `BinaryInputPolicy`. In `.inl`, track a running count and call `cb_wait_front(icb, running_total)`. ~30 lines.

Option B (equivalent): the caller uses `WaitUpfrontNoPop` with a single upfront wait for `Wt` total tiles, then a `NoWaitNoPop` pass. This is a call-site refactoring, not a library change, and is the correct approach:

```cpp
cb_wait_front(cb_inp, Wt);   // upfront, single wait
square<BinaryInputPolicy::NoWaitNoPop,
       BinaryOutputPolicy::Bulk>(
    cb_inp, cb_x2, BinaryInputBlockShape::row(Wt));
cb_pop_front(cb_inp, Wt);
```

**UNCERTAIN**: whether the cumulative-wait pattern exists because the CB is too small to hold all `Wt` tiles at once (in which case upfront-wait is impossible). If so, Option A is required; otherwise Option B is a call-site workaround with no library change needed.

The non-zero B tile offset (for rotary embeddings) is a separate issue: the helper always computes `tile_b` from the start of the CB. This is **not** fixable without passing an explicit `b_tile_offset` parameter.

### Proposed addition (B tile offset)

Add optional `b_tile_offset` parameter:

```cpp
template <BinaryOpType op_type, BroadcastDim bcast_dim = BroadcastDim::NONE, ...>
ALWI void binary_op(
    uint32_t icb_a, uint32_t icb_b, uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op = {},
    AccumT accum = {},
    uint32_t b_tile_offset = 0);   // added
```

In the tile-index computation, apply `b_tile_offset` to all B index calculations. ~10 lines.

### Complexity

`b_tile_offset`: Low (~10 lines in `.inl`, backward-compatible default). `WaitCumulativeNoPop` policy: Medium (~30 lines). Risk: low if CB is large enough for upfront wait.

---

## Gap 3: `binary_dest_reuse_tiles` — dest-aliased fused multiply — RESOLVED

### Status: RESOLVED

`DestReuseOp<CB, OpType, ReuseType, Slot>` is now available as a `PostOp` to `binary_op()`. The DEST slot and which SRC gets DEST (`DEST_TO_SRCA` vs `DEST_TO_SRCB`) are both template parameters. The CB argument (`CB`) is the source CB whose tile[0] is read as the second operand; the caller is responsible for waiting on it upfront — `DestReuseMul` always reads `CB[0]`. A `DestReuseMul<CB, Slot>` alias covers the common multiply case.

### Kernels unblocked

`ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp` — Stage 2 (previously raw):

```cpp
// Before (raw):
sub_tiles(cb_other, cb_bcast, 0, 0, 0);
binary_dest_reuse_tiles_init<ELWMUL, DEST_TO_SRCA>(cb_den);
binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_den, 0, 0);

// After (helper):
sub(cb_other, cb_bcast, cb_affine_or_out, BinaryInputBlockShape::single(),
    DestReuseMul<cb_den>{});
// Note: cb_den must be waited upfront before the loop; DestReuseMul always reads CB[0]
```

The sub and the dest-reuse multiply are fused into a single `binary_op` call via the PostOp chain.

---

## Gap 4: `moreh_common.hpp` `*_tiles_to_cb` wrapper incompatibility

### What's missing

`moreh_adam.cpp` and `moreh_sgd.cpp` use `mul_tiles_to_cb`, `add_tiles_to_cb`, `sub_tiles_to_cb`, `copy_tile_to_cb` from `ttnn/kernel/compute/moreh_common.hpp`. These are stateful wrappers that own their own DEST acquire/release cycle per call and use a CB-backed intermediate for chaining. They are structurally incompatible with `binary_op()` because:

1. They interleave init+exec+pack in a single call per tile — no way to batch.
2. They use `pack_tile_with_dt` (data-type-aware packing), which is not exposed by the helper.
3. `moreh_adam.cpp` uses `power_tile` / `binary_max_tile` (DEST-level ops) inline between binary stages — these are outside the SFPU chain model entirely.

### Evidence from kernel code

`moreh_adam.cpp` lines 78–113 (representative excerpt):
```cpp
mul_tiles_to_cb(cb_param_in, cb_scalar_args, cb_tmp1, first_tile, weight_decay_tile, 0, 0);
add_tiles_to_cb(cb_grad_in, cb_tmp1, tmp_cb_grad, first_tile, first_tile, 0, 1);
sub_tiles_to_cb(cb_one, cb_scalar_args, cb_tmp1, first_tile, beta1_tile, 0, 0);
mul_tiles_to_cb(tmp_cb_grad, cb_tmp1, cb_tmp1, first_tile, first_tile, 0, 1);
// ... then:
power_tile_init();
power_tile(dst0, step);   // integer power of scalar args — no CB involved
// ... then:
binary_max_tile_init();
binary_max_tile(dst0, dst1, dst0);   // element-wise max between two DEST slots
```

`moreh_sgd.cpp` lines 38–87: similar `*_tiles_to_cb` chain, but simpler (no `power_tile` / `binary_max_tile`).

### Proposed addition

Per migration plan Phase 4: implement thin shims over `binary_op_helpers.hpp` rather than migrating directly:

```cpp
// moreh_common_shim.hpp (replaces moreh_common.hpp *_tiles_to_cb functions)
ALWI void add_tiles_to_cb(uint32_t icb_a, uint32_t icb_b, uint32_t ocb,
                           uint32_t tile_a, uint32_t tile_b,
                           uint32_t pop_a, uint32_t pop_b) {
    // Wrap binary_op with NoWaitNoPop policies and single-tile shape
    // Then manually handle pop based on pop_a/pop_b flags
}
```

`moreh_adam.cpp`'s `power_tile` and `binary_max_tile` sections cannot be wrapped — those stay raw.

### Complexity

Medium (~100 lines of shim code). The shim work is blocked on verifying no behavioral differences between `add_tiles_to_cb` and the helper with `BinaryInputBlockShape::single()`.

---

## Gap 5: Copy-or-add conditional dispatch (groupnorm, compute_depthwise_conv1d)

### What's missing

Both `groupnorm.cpp` and `compute_depthwise_conv1d.cpp` dispatch at runtime between `copy_tile` and `add_tiles` based on a loop-iteration index or state variable:

`groupnorm.cpp` lines 645–659:
```cpp
if (copy_or_add == true) {
    copy_tile_init(cb_xmm_id);
    // ...
    copy_tile(cb_xmm_id, index_xmm, dst0);
} else {
    add_tiles_init(cb_reread_out_id, cb_xmm_id);
    // ...
    add_tiles(cb_reread_out_id, cb_xmm_id, index_reread_out, index_xmm, dst0);
}
pack_tile<true>(dst0, cb_reread_write_out_id, index_reread_out);  // random-access pack
```

`compute_depthwise_conv1d.cpp` lines 49–79: same pattern (`idx == 0` → copy; `idx > 0` → add+copy through temp CB).

`binary_op()` requires a compile-time fixed op type. Runtime dispatch between two different LLK operations is not expressible.

Additionally, both kernels use `pack_tile<true>(dst0, cb, explicit_index)` — random-access output packing — which the helper does not support (`pack_tile` calls are always sequential or per-chunk with helper-managed indices).

### Classification: Fundamental incompatibility. These patterns cannot be expressed in the helper model.

`groupnorm.cpp` and `compute_depthwise_conv1d.cpp` should stay raw. The migration plan already correctly classifies them as no-migrate (Tier 3) and partial-migrate respectively.

### Complexity

N/A (unfixable).

---

## Gap 6: `binary_max_tile` / `SfpuMax` SFPU struct missing — RESOLVED

### Status: RESOLVED

`SfpuMax` and `SfpuMin` are now implemented in `sfpu_binary.hpp`, wrapping `binary_max_tile` / `binary_min_tile` respectively. Both follow the same `BinaryOp<..., In0, In1, Out>` CRTP pattern as the existing `SfpuAdd`, `SfpuMul`, etc.

### Kernels unblocked

- `ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/moreh_adam.cpp` (AMSGRAD path) — `binary_max_tile` between stages can now be expressed as `SfpuMax` in a PostOp chain. See the updated Hard Blockers section for the remaining constraint on moreh_adam's overall structure.

### Complexity

Delivered (~15 lines). Risk: none.

---

## Hard Blockers (unfixable by helper additions)

### 1. `binary_dest_reuse_tiles` in `batch_norm_kernel.cpp` Stage 2 — RESOLVED

`DestReuseMul<CB, Slot>` PostOp now covers this. See Gap 3 above. All four stages of batch_norm are now migratable.

### 2. Copy-or-add runtime dispatch in `groupnorm.cpp` and `compute_depthwise_conv1d.cpp`

Both kernels branch at runtime between `copy_tile` and `add_tiles`. `binary_op<op_type>` requires `op_type` to be a compile-time constant. No template machinery can bridge a runtime boolean to a compile-time `BinaryOpType`. These files stay raw.

### 3. Same-CB adds in CCL reduction kernels — partially fixed

The double-wait/pop bug for same-CB inputs is now fixed: `add(cb, cb, cb_out, ...)` no longer double-waits or double-pops. For the simple same-index case (both operands at the same tile position in the same CB), `add(cb, cb, cb_out, shape)` now works correctly.

The non-contiguous index problem remains. `llama_reduce_scatter`, `all_reduce_create_qkv_heads` — both add pairs of tiles from the **same** CB with non-contiguous index pairs (`first_index`, `second_index` computed per iteration). The helper iterates tile indices sequentially and cannot express variable non-contiguous index pairs. For these kernels, use `NoWaitNoPop` + external wait/pop + two separate helper calls, or keep the inner loop raw.

### 4. Non-sequential random-access `pack_tile<true>` with explicit output index

`pack_tile<true>(dst_idx, cb, N)` writes to absolute position N in the output CB rather than the next sequential slot. This is distinct from non-contiguous *input* tile access. `bcast_h_sharded_optimised.cpp`, `groupnorm.cpp`, `compute_depthwise_conv1d.cpp` all use this form. The helper always packs sequentially; adding non-contiguous output would require a per-tile output-index callback that breaks the sequential-output assumption and complicates the API significantly. These kernels (`groupnorm.cpp`, `bcast_h_sharded_optimised.cpp`, `compute_depthwise_conv1d.cpp`) remain raw for this reason.

### 5. Stateful CB lifecycle in moreh_adam's `power_tile`/`binary_max_tile` DEST stages — partially improved

Load ops are now valid in PostOp chains: a `Load` PostOp calls `copy_tile_to_dst_init_short` for its CB on the first tile and handles its own CB lifecycle (wait/pop/copy). `SfpuMax` is now available (see Gap 6 above). This means a `binary_max_tile` between binary stages can be expressed as `SfpuMax` in a PostOp chain — resolving the `moreh_adam` AMSGRAD case at the PostOp level.

The remaining blocker for moreh_adam's overall loop structure is `power_tile` (integer power of a scalar DEST value), which is still not expressible as a PostOp callback. The `moreh_common.hpp` `*_tiles_to_cb` wrappers also interleave init+exec+pack per tile in a way that is structurally incompatible with the helper's batched model. Until those wrappers are shimmed, the entire moreh_adam loop body stays raw.

---

## Open Questions

1. **UNCERTAIN (Gap 2)**: In `rmsnorm_pre_allgather.cpp`, is the cumulative-wait pattern (`cb_wait_front(cb_inp, wt + blk)`) required because `cb_inp` is a single-tile-capacity CB that fills incrementally, or is it an artifact of the old coding style that can be replaced by a single upfront `cb_wait_front(cb_inp, Wt)` before the loop? If the CB can hold all `Wt` tiles, then `NoWaitNoPop` with an upfront wait is a valid call-site refactoring with no library change needed.

2. **UNCERTAIN (Gap 4 — moreh shim)**: Does `pack_tile_with_dt` produce different results from `pack_tile` for the data formats used by moreh kernels (typically BF16/FP32)? If they are identical for those formats, the shim is safe. If not, the shim must also expose a `use_dt` flag.

3. ~~**UNCERTAIN (Gap 6)**~~: RESOLVED — `SfpuMax` and `SfpuMin` are now implemented and ship in `sfpu_binary.hpp`. The `binary_max_tile` / `binary_min_tile` LLKs were confirmed reachable via the existing include chain.

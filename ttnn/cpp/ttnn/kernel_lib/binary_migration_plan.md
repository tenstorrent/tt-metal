# Binary Helper Migration Plan

Reference: `binary_op_helpers.hpp`, `binary_op_analysis.md`

---

## Phase 0 — Prove the abstraction (no library changes)

**Goal:** Establish a merged example before asking the team to accept wider
migration. These are the cleanest cases — no partial reconfig, no moreh, no
conditional dispatch. Line count reduction target ≥ 40% per file; CI must hold.

### Batch A — bcast kernels (Pattern 1, 4)

| File | Pattern | Primary op |
|------|---------|-----------|
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_h.cpp` | 1 | `add<BroadcastDim::ROW>` (templated on op) |
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_w.cpp` | 1 | `add<BroadcastDim::COL>` |
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_hw.cpp` | 1 | `add<BroadcastDim::SCALAR>` |
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_h_sharded_optimised.cpp` | 1/4 | `add<BroadcastDim::ROW, ..., WaitUpfrontNoPop>` |

### Batch B — CCL reduction kernels (Pattern 2, accumulator)

All use the same structure: wait N tiles upfront, add into dst=0, repeat.
`BinaryAccumulate{cb_accum}` replaces the manual reload.

| File | Notes |
|------|-------|
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduction.cpp` | |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduction.cpp` | |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/dim_zero_line_reduction.cpp` | |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/dim_zero_ring_reduction.cpp` | |
| `ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/kernels/minimal_ring_reduction.cpp` | |
| `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/reduction.cpp` | |
| `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/compute/reduction.cpp` | |
| `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/kernels/compute/reduction.cpp` | |
| `ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/deepseek_moe_reduce_scatter_reduction.cpp` | |
| `ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/compute/reduction.cpp` | |
| `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp` | |
| `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/deepseek_moe_fast_reduce_nc_reduce.cpp` | |
| `ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/accumulation_compute.cpp` | |

### Batch C — Rotary embedding (Pattern 4, persisted B)

| File | Notes |
|------|-------|
| `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp` | `mul<SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop>` |
| `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama_sharded.cpp` | same |
| `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded.cpp` | same |
| `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded_row_major.cpp` | same |

**Phase 0 exit criteria:**
- All migrated files compile cleanly
- CI passes on the covering test jobs for bcast, CCL, and rotary_embedding_llama
- At least one file reviewed and merged to main

---

## Phase 1 — Land Gap #1: partial data-format reconfig — DONE

**This library change has landed.** It unblocks ~25 files (the entire
normalization family).

### Change (already implemented)

`BinaryDataFormatReconfig` in `binary_op_helpers.hpp` / `.inl` includes:

```cpp
enum class BinaryDataFormatReconfig {
    NONE                   = 0,
    INPUT                  = 1,
    OUTPUT                 = 2,
    INPUT_AND_OUTPUT       = 3,
    SRCA_ONLY              = 4,   // reconfig_data_format_srca(prev_icb_a, icb_a)
    SRCB_ONLY              = 5,   // reconfig_data_format_srcb(prev_icb_b, icb_b)
    SRCA_ONLY_AND_OUTPUT   = 6,
    SRCB_ONLY_AND_OUTPUT   = 7,
};
```

The `.inl` must track the previous CB identity per side. Simplest approach:
pass `prev_icb_a` / `prev_icb_b` as optional parameters defaulting to
`icb_a` / `icb_b` (same-as-current = no-op reconfig, identical to existing
`INPUT` behaviour). The header comment in `binary_op_helpers.hpp` already notes
this design option.

**Phase 1 exit criteria: MET.**
- All eight `BinaryDataFormatReconfig` enum values confirmed in `binary_op_helpers.hpp` lines 128–141.
- `.inl` correctly emits `reconfig_data_format_srca/b` calls for all partial variants.
- No kernel migrations were required in this phase.

---

## Phase 2 — Normalization family (Tier 2)

Migrate in dependency order (simpler post-allgather kernels first, then the
full welford variants). Each kernel chain uses `SRCB_ONLY` or `SRCA_ONLY`
for its mid-chain reconfigs.

### 2a — rmsnorm (fewest reconfig sites, ~2 each)

| File | Reconfig sites | Notes |
|------|---------------|-------|
| `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp` | 2 | mul<SCALAR, WaitUpfrontNoPop> + PostOp rsqrt |
| `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp` | ~2 | |
| `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather_2d.cpp` | ~2 | |
| `ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather.cpp` | ~2 | |
| `ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_pre_allgather.cpp` | ~2 | |
| `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/compute/rms_compute.cpp` | 2 | |

### 2b — softmax (~3–5 reconfig sites each)

| File | Reconfig sites | Notes |
|------|---------------|-------|
| `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp` | ~3 | persistent max/sum; WaitUpfrontNoPop for B |
| `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax_sharded.cpp` | ~4 | |
| `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax_large_tensor.cpp` | 5 | |

### 2c — layernorm (4–9 reconfig sites each)

| File | Reconfig sites |
|------|---------------|
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded_post_allgather.cpp` | 2 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather.cpp` | ~3 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather_welford.cpp` | ~3 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather.cpp` | ~2 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_welford.cpp` | ~2 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_2d.cpp` | ~2 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp` | 5 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded_pre_allgather.cpp` | ~3 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded_welford.cpp` | 4 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp` | 6 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_large_tensor.cpp` | 6 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_welford.cpp` | 6 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded_welford.cpp` | 4 |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_large_tensor_welford.cpp` | 9 |
| `ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/device/kernels/compute/layernorm_post_allgather_welford.cpp` | ~4 |

### 2d — batch_norm and depthwise conv

| File | Notes |
|------|-------|
| `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp` | ADD/SUB/MUL chain + PostOp rsqrt; Stage 2 now uses `DestReuseMul<cb_den>` PostOp (all stages migratable) |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/compute_depthwise_conv1d.cpp` | per-channel bcast + grid traversal |

**Phase 2 exit criteria:**
- All normalization CI jobs pass
- At least two welford variants migrated (highest reconfig density, best proof)

---

## Phase 3 — Tier 3: case-by-case, default no-migrate

These are not migration targets unless there is a specific driver (e.g., a
planned refactor of the file for other reasons). Document in-place with a
comment pointing to `binary_op_analysis.md` §Pattern 8.

| File | Reason |
|------|--------|
| `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp` | copy-or-add conditional dispatch |
| `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp` | same |
| `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/welford_groupnorm.cpp` | Welford state machine |
| `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/welford_groupnorm_sharded_v2.cpp` | same |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp` | fused with matmul |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | fused with matmul |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | fused with matmul |
| `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/compute/transformer_group_attn_matmul.cpp` | fused with matmul |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp` | persistent K/V lifetime, fused with SDPA |

---

## Phase 4 — Moreh: policy decision, not code

**Current state:** 12 moreh kernels use `moreh_common.hpp` `*_tiles_to_cb` APIs.
These should not be migrated eagerly.

**Policy:**
1. No new code under `ttnn/cpp/ttnn/operations/` may include `moreh_common.hpp`
   binary ops (`add_tiles_to_cb` etc.). New kernels must use `binary_op_helpers.hpp`.
2. Existing moreh kernels stay as-is until Gap #1 lands AND a specific feature
   work on the file justifies the rewrite.
3. When Gap #1 is stable, plan thin shims: reimplement `*_tiles_to_cb` as
   wrappers over `binary_op_helpers.hpp` with `BinaryInputBlockShape::single()`.

Moreh files to revisit after Gap #1:
| File |
|------|
| `ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/moreh_sgd.cpp` |
| `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp` |
| `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_c_large.cpp` |
| `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_h_large.cpp` |
| `ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/kernels/moreh_norm_backward_kernel.cpp` |
| `ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/moreh_adam.cpp` |
| `ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/moreh_adamw.cpp` |
| `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_w.cpp` |
| `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_c_large.cpp` |
| `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_h.cpp` |
| `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_w_large.cpp` |
| `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_h_large.cpp` |

---

## Summary

| Phase | Action | Files | Blocker |
|-------|--------|-------|---------|
| 0 | Migrate Tier 1 (bcast, CCL reductions, rotary) | ~21 | None — start now |
| 1 | Ship SRCA_ONLY / SRCB_ONLY reconfig enum + .inl | library only | DONE |
| 2 | Migrate Tier 2 (normalization family) | ~25 | Phase 1 |
| 3 | No-migrate Tier 3; add comments | 9 | N/A |
| 4 | Moreh policy + shims | 12 | Phase 1 |

---

## Appendix — Migratable Code Per File

For each file: raw tile API section to be replaced, and the helper call that replaces it.

---

### Phase 0 / Batch A — bcast kernels

#### `eltwise/binary/device/kernels/compute/bcast_h.cpp`
Pattern 1 — streaming add/op with ROW broadcast.
```cpp
// RAW — entire kernel loop
for (uint32_t b = 0; b < B; b++) {
    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            cb_wait_front(tt::CBIndex::c_1, onetile);
            cb_reserve_back(tt::CBIndex::c_2, onetile);
            acquire_dst();
            cb_wait_front(tt::CBIndex::c_0, onetile);
            BCAST_OP<BroadcastType::ROW>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
            pack_tile(0, tt::CBIndex::c_2);
            cb_pop_front(tt::CBIndex::c_0, onetile);
            release_dst();
            cb_push_back(tt::CBIndex::c_2, onetile);
            cb_pop_front(tt::CBIndex::c_1, onetile);
        }
    }
}
```
```cpp
// REPLACEMENT
binary_op_init_common(c_0, c_1, c_2);
BCAST_HELPER<BroadcastDim::ROW>(c_0, c_1, c_2, BinaryInputBlockShape::of(B * Ht, Wt));
// c_1 is per-row (Ht tiles): use WaitAndPopPerTile for both sides (default)
```

---

#### `eltwise/binary/device/kernels/compute/bcast_w.cpp`
Pattern 1 — streaming op with COL broadcast; B tile persists per row.
```cpp
// RAW
for (uint32_t b = 0; b < B; b++) {
    for (uint32_t h = 0; h < Ht; h++) {
        cb_wait_front(tt::CBIndex::c_1, onetile);   // B tile: one per row
        for (uint32_t w = 0; w < Wt; w++) {
            cb_reserve_back(tt::CBIndex::c_2, onetile);
            acquire_dst();
            cb_wait_front(tt::CBIndex::c_0, onetile);
            BCAST_OP<BroadcastType::COL>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
            pack_tile(0, tt::CBIndex::c_2);
            cb_pop_front(tt::CBIndex::c_0, onetile);
            release_dst();
            cb_push_back(tt::CBIndex::c_2, onetile);
        }
        cb_pop_front(tt::CBIndex::c_1, onetile);
    }
}
```
```cpp
// REPLACEMENT — B persists for each row of Wt A tiles
binary_op_init_common(c_0, c_1, c_2);
BCAST_HELPER<BroadcastDim::COL,
             BinaryInputPolicy::WaitAndPopPerTile,
             BinaryInputPolicy::WaitUpfrontPopAtEnd>(
    c_0, c_1, c_2, BinaryInputBlockShape::of(B * Ht, Wt));
```

---

#### `eltwise/binary/device/kernels/compute/bcast_hw.cpp`
Pattern 1 — streaming op with SCALAR broadcast; B may be pre-loaded (`BCAST_SCALAR` define).
```cpp
// RAW
for (uint32_t b = 0; b < B; b++) {
    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
#ifndef BCAST_SCALAR
            cb_wait_front(tt::CBIndex::c_1, onetile);
#endif
            cb_reserve_back(tt::CBIndex::c_2, onetile);
            acquire_dst();
            cb_wait_front(tt::CBIndex::c_0, onetile);
            BCAST_OP<BroadcastType::SCALAR>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
            pack_tile(0, tt::CBIndex::c_2);
            cb_pop_front(tt::CBIndex::c_0, onetile);
#ifndef BCAST_SCALAR
            cb_pop_front(tt::CBIndex::c_1, onetile);
#endif
            release_dst();
            cb_push_back(tt::CBIndex::c_2, onetile);
        }
    }
}
```
```cpp
// REPLACEMENT
binary_op_init_common(c_0, c_1, c_2);
#ifdef BCAST_SCALAR
// B is pre-loaded (sharded), caller already waited
BCAST_HELPER<BroadcastDim::SCALAR,
             BinaryInputPolicy::WaitAndPopPerTile,
             BinaryInputPolicy::NoWaitNoPop>(
    c_0, c_1, c_2, BinaryInputBlockShape::of(B * Ht, Wt));
#else
BCAST_HELPER<BroadcastDim::SCALAR>(c_0, c_1, c_2, BinaryInputBlockShape::of(B * Ht, Wt));
#endif
```

---

#### `eltwise/binary/device/kernels/compute/bcast_h_sharded_optimised.cpp`
Pattern 4 — bulk pre-loaded A and B; B persists per W column; packed with random-access `pack_tile<true>`.
```cpp
// RAW
cb_wait_front(tt::CBIndex::c_0, Wt * Ht);      // A: all tiles upfront
cb_reserve_back(tt::CBIndex::c_2, Wt * Ht);    // out: bulk reserve
uint32_t b_offset = 0;
for (uint32_t bn = 0; bn < batch_b; bn++) {
    for (uint32_t wt = 0; wt < Wt; wt++) {
        cb_wait_front(tt::CBIndex::c_1, onetile); // B: one tile per W column
        for (uint32_t ht = 0; ht < Ht_per_batch_b; ht += h_blk) {
            acquire_dst();
            for (uint32_t htr = 0; htr < h_blk; htr++) {
                uint32_t current_index = b_offset + (ht + htr) * Wt + wt;
                BCAST_OP<BroadcastType::ROW>(tt::CBIndex::c_0, tt::CBIndex::c_1, current_index, 0, htr);
                pack_tile<true>(htr, tt::CBIndex::c_2, current_index);
            }
            release_dst();
        }
        cb_pop_front(tt::CBIndex::c_1, onetile);
    }
    b_offset += Ht_per_batch_b * Wt;
}
cb_pop_front(tt::CBIndex::c_0, Wt * Ht);
cb_push_back(tt::CBIndex::c_2, Wt * Ht);
```
Note: uses random-access `pack_tile<true>` with explicit output index — this breaks the helper's sequential pack assumption. **Requires manual tile loop or helper extension for random-access packing. Partial migration only.**

---

### Phase 0 / Batch B — CCL/reduction kernels

Files 5–8 (`line_reduction.cpp`, `ring_reduction.cpp`, `dim_zero_line_reduction.cpp`, `dim_zero_ring_reduction.cpp`) are structurally identical. All use the same granular streaming add:

```cpp
// RAW (representative — same in all four)
while (tiles_read < tiles_to_read) {
    uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);
    cb_wait_front(input_cb_id, tile_granularity);
    cb_wait_front(intermediate_cb, tile_granularity);
    cb_reserve_back(output_cb, tile_granularity);
    acquire_dst();
    for (uint32_t tile_id = 0; tile_id < num_pages_to_read; tile_id++) {
        add_tiles(input_cb_id, intermediate_cb, tile_id, tile_id, tile_id);
        pack_tile(tile_id, output_cb);
    }
    release_dst();
    cb_pop_front(input_cb_id, tile_granularity);
    cb_pop_front(intermediate_cb, tile_granularity);
    cb_push_back(output_cb, tile_granularity);
    tiles_read += num_pages_to_read;
}
```
```cpp
// REPLACEMENT (per outer loop iteration)
binary_op_init_common(input_cb_id, intermediate_cb, output_cb);
add(input_cb_id, intermediate_cb, output_cb,
    BinaryInputBlockShape::row(num_pages_to_read));
// WaitAndPopPerTile default handles the while-loop internally;
// wrap in the outer for/while as needed for multi-step reductions
```

---

#### `strided_reduce_scatter_async/.../minimal_ring_reduction.cpp`
Has two sections. First section (normal ring step) is identical to the four files above.
Second section (final step) is a fused addcmul: add → mul (with optional ROW bcast) → mul_unary scalar → add. The three binary stages have different CB pairs and interleaved `reconfig_data_format` calls.

```cpp
// RAW — fused addcmul final step (simplified)
// Stage 1: add
reconfig_data_format(input_cb, intermediate_cb);
pack_reconfig_data_format(addcmul_temp_cb);
add_tiles_init(input_cb, intermediate_cb, false);
acquire_dst();
for (uint32_t tile_id = 0; tile_id < N; tile_id++)
    add_tiles(input_cb, intermediate_cb, tile_id, tile_id, tile_id);
// ...pack, pop...

// Stage 2: mul (with optional ROW bcast) + scalar
reconfig_data_format(addcmul_temp_cb, addcmul_b_cb);
pack_reconfig_data_format(addcmul_temp_cb);
#ifdef ADDCMUL_B_BROADCAST
mul_bcast_rows_init_short(addcmul_temp_cb, addcmul_b_cb);
#else
mul_tiles_init(addcmul_temp_cb, addcmul_b_cb, false);
#endif
for (uint32_t tile_id = 0; tile_id < N; tile_id++) {
    tile_regs_acquire();
    #ifdef ADDCMUL_B_BROADCAST
    mul_tiles_bcast<BroadcastType::ROW>(addcmul_temp_cb, addcmul_b_cb, tile_id, tile_id, 0);
    #else
    mul_tiles(addcmul_temp_cb, addcmul_b_cb, tile_id, tile_id, 0);
    #endif
    mul_unary_tile(0, fused_ternary_scalar_uint);
    tile_regs_commit(); tile_regs_wait(); pack_tile(0, addcmul_temp_cb); tile_regs_release();
}

// Stage 3: add a + (scalar*acc*b)
reconfig_data_format(addcmul_temp_cb, addcmul_a_cb);
pack_reconfig_data_format(output_cb);
add_tiles_init(addcmul_temp_cb, addcmul_a_cb, false);
acquire_dst();
for (uint32_t tile_id = 0; tile_id < N; tile_id++)
    add_tiles(addcmul_temp_cb, addcmul_a_cb, tile_id, tile_id, tile_id);
// ...pack, pop...
```
```cpp
// REPLACEMENT — normal ring step
add(input_cb, intermediate_cb, output_cb, BinaryInputBlockShape::row(N));

// REPLACEMENT — addcmul final step (requires SRCB_ONLY reconfig — Gap #1)
// Stage 1: add (same as above)
add<BroadcastDim::NONE, BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitAndPopPerTile, BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    input_cb, intermediate_cb, addcmul_temp_cb, BinaryInputBlockShape::row(N));

// Stage 2: mul + scalar PostOp (bcast variant conditional)
#ifdef ADDCMUL_B_BROADCAST
mul<BroadcastDim::ROW,
#else
mul<BroadcastDim::NONE,
#endif
    BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::PerTile, BinaryDataFormatReconfig::SRCB_ONLY>(  // Gap #1
    addcmul_temp_cb, addcmul_b_cb, addcmul_temp_cb, BinaryInputBlockShape::row(N),
    [](uint32_t dst_idx) { mul_unary_tile(dst_idx, fused_ternary_scalar_uint); });

// Stage 3: add
add<BroadcastDim::NONE, BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitAndPopPerTile, BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::SRCB_ONLY>(  // Gap #1
    addcmul_temp_cb, addcmul_a_cb, output_cb, BinaryInputBlockShape::row(N));
```
Stages 2 and 3 require **Gap #1** (`SRCB_ONLY`). Stage 1 of the normal ring path has no dependency.

---

#### `all_reduce_async/.../reduction.cpp`, `deepseek_moe_fast_reduce_nc/.../reduce.cpp`
Pattern 2 — paired-block accumulation. Both are identical in structure.
```cpp
// RAW
uint32_t block_num_tiles_cnt = 0;
for (uint32_t p = 0; p < num_pack_iters; ++p) {
    uint32_t num_tiles_to_pack = std::min(max_dst_tiles, block_num_tiles - block_num_tiles_cnt);
    tile_regs_acquire();
    for (uint32_t block = 0; block < num_blocks; block += 2) {
        for (uint32_t i = 0; i < num_tiles_to_pack; ++i) {
            add_tiles(cb_in0, cb_in1,
                      block * block_num_tiles + p * max_dst_tiles + i,
                      (block + 1) * block_num_tiles + p * max_dst_tiles + i,
                      i);
        }
    }
    tile_regs_commit(); tile_regs_wait();
    for (uint32_t i = 0; i < num_tiles_to_pack; ++i)
        pack_tile(i, cb_out0, p * max_dst_tiles + i);
    tile_regs_release();
    block_num_tiles_cnt += num_tiles_to_pack;
}
```
Note: uses non-sequential input indices (stride by `block_num_tiles`) and random-access output packing (`pack_tile(i, cb_out0, p * max_dst_tiles + i)`). The helper's sequential index assumption doesn't hold here. **Partial migration only — keep the DEST management, replace only the trivially-indexable pairs.**

---

#### `llama_reduce_scatter/.../reduction.cpp`, `llama_reduce_scatter_create_heads/.../reduction.cpp`, `all_reduce_create_qkv_heads/.../reduction.cpp`
Pattern 2 — device-pair accumulation with explicit first/second indices.
```cpp
// RAW
tile_regs_acquire();
for (uint32_t page_group = 0; page_group < num_pages_per_packet; page_group++) {
    for (uint32_t device_pair = 0; device_pair < num_device_pairs; device_pair++) {
        add_tiles(fabric_receiver_cb_id, fabric_receiver_cb_id,
                  first_index, second_index, page_group);
    }
}
tile_regs_commit(); tile_regs_wait();
for (uint32_t page_group = 0; page_group < num_pages_per_packet; page_group++)
    pack_tile(page_group, accumulator_cb_id, page_group);
tile_regs_release();
```
Note: A and B are the **same CB** (`fabric_receiver_cb_id`) with non-contiguous index pairs (`first_index`, `second_index` computed per iteration). The same-CB double-wait/pop bug is now fixed — `add(cb, cb, ...)` is correct for the same-index case. However, the non-contiguous index pairs (different `first_index` and `second_index` each iteration) still cannot be expressed in the helper's sequential tile model. **Keep raw for the non-contiguous index case.** For reduction trees that always use the same tile position for both operands, `add(cb, cb, cb_out, shape)` is now valid.

---

#### `deepseek_moe_reduce_scatter/.../reduction.cpp`
Pattern 1 — streaming add across per-ring-step CB pairs.
```cpp
// RAW (per ring step)
binary_op_init_common(input_slice_cb_id, intermediate_slice_cb_id, compute_cb_id);
add_tiles_init(input_slice_cb_id, intermediate_slice_cb_id, false);
while (tiles_read < tiles_to_read) {
    cb_wait_front(input_slice_cb_id, tile_granularity);
    cb_wait_front(intermediate_slice_cb_id, tile_granularity);
    cb_reserve_back(compute_cb_id, tile_granularity);
    acquire_dst();
    for (uint32_t tile_id = 0; tile_id < tile_granularity; ++tile_id)
        add_tiles(input_slice_cb_id, intermediate_slice_cb_id, tile_id, tile_id, tile_id);
    release_dst();
    cb_pop_front(input_slice_cb_id, tile_granularity);
    cb_pop_front(intermediate_slice_cb_id, tile_granularity);
    cb_push_back(compute_cb_id, tile_granularity);
    tiles_read += tile_granularity;
}
```
```cpp
// REPLACEMENT (per ring step — CB IDs change each iteration, init needed each time)
add<BroadcastDim::NONE, BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitAndPopPerTile, BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT, /*init=*/true>(
    input_slice_cb_id, intermediate_slice_cb_id, compute_cb_id,
    BinaryInputBlockShape::row(tiles_per_step));
```

---

#### `fast_reduce_nc/.../reduce_nc.cpp`
Pattern 2 — accumulator add across input rows into a single output tile.
```cpp
// RAW
for (uint32_t i = 0; i < num_output_tiles; i++) {
    add_tiles_init(cb_in0, cb_in1, true);
    reconfig_data_format(cb_in0, cb_in1);
    tile_regs_acquire();
    for (uint32_t j = 0; j < num_input_tiles_iter; ++j) {
        cb_in0_obj.wait_front(input_granularity);
        for (uint32_t k = 0; k < input_granularity; k++)
            add_tiles(cb_in0, cb_in1, k, first_tile, dst0);   // dst0=0, accumulates
        cb_in0_obj.pop_front(input_granularity);
    }
    tile_regs_commit();
    cb_out0_obj.reserve_back(onetile);
    pack_reconfig_data_format(cb_out0);
    tile_regs_wait();
    pack_tile(dst0, cb_out0);
    tile_regs_release();
    cb_out0_obj.push_back(onetile);
}
```
Note: all `add_tiles` accumulate into `dst0=0` (classic reduce-into-register). `cb_in1` is the scalar identity (persisted, `first_tile` index). Maps to `BinaryAccumulate` pattern but requires the helper to emit `add` into a fixed dst without pack-per-tile. **Partial migration — DEST management can be replaced; CB orchestration is already clean.**

---

#### `reduction/accumulation/.../accumulation_compute.cpp`
Pattern — cumulative sum/product; dispatches to `add_tiles` or `mul_tiles` based on `accumulation_op` enum.
```cpp
// RAW
binary_op_init_common(cb_in, cb_op, cb_out);
if (accumulation_op == AccumulationOp::CUMPROD) {
    mul_tiles_init(cb_in, cb_op);
    mul_tiles(cb_in, cb_op, FIRST_TILE, FIRST_TILE, WORKING_REG);
} else if (accumulation_op == AccumulationOp::CUMSUM) {
    add_tiles_init(cb_in, cb_op);
    add_tiles(cb_in, cb_op, FIRST_TILE, FIRST_TILE, WORKING_REG);
}
```
Note: this is a single-tile operation inside an outer row×col loop that reloads the accumulator CB (`cb_acc`) each iteration. The helper's loop management doesn't map to this state-machine pattern. **Keep raw; structure is already minimal.**

---

### Phase 0 / Batch C — Rotary embedding kernels

#### `rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp`
The migratable binary segments are the two `mul_tiles` loops (rotated×sin, in×cos) and the final `add_tiles` (cos_result + sin_result). The `matmul_tiles` section stays raw.
```cpp
// RAW — mul: rotated * sin
mul_tiles_init(rotated_in_interm_cb, sin_cb);
ACQ();
for (uint32_t j = 0; j < Wt; ++j) {
    mul_tiles(rotated_in_interm_cb, sin_cb, j, j + (sin_cos_row_cnt * Wt), j);
    pack_tile(j, sin_interm_cb, j);
}
REL();

// RAW — mul: in * cos
ACQ();
for (uint32_t j = 0; j < Wt; ++j) {
    mul_tiles(in_cb, cos_cb, j, j + (sin_cos_row_cnt * Wt), j);
    pack_tile(j, cos_interm_cb, j);
}
REL();

// RAW — add: cos_result + sin_result
cb_wait_front(sin_interm_cb, Wt);
cb_wait_front(cos_interm_cb, Wt);
add_tiles_init(cos_interm_cb, sin_interm_cb);
ACQ();
for (uint32_t j = 0; j < Wt; ++j) {
    add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
    pack_tile(j, out_cb, j);
}
REL();
```
Note: sin/cos CB index uses `j + (sin_cos_row_cnt * Wt)` — non-zero B tile offset, not 0-based. The helper iterates B from 0. **Not directly migratable without offset support in the helper.** The `add` at the end (both CBs 0-based) is directly migratable:
```cpp
// Replacement for the final add only
add<BroadcastDim::NONE,
    BinaryInputPolicy::WaitUpfrontNoPop,   // already waited before this section
    BinaryInputPolicy::WaitUpfrontNoPop>(
    cos_interm_cb, sin_interm_cb, out_cb, BinaryInputBlockShape::row(Wt));
```

---

#### `rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama_sharded.cpp`
#### `rotary_embedding_llama_fused_qk/.../rotary_embedding_llama_sharded.cpp`
Both identical. Two `mul_tiles_bcast<ROW>` loops + one `add_tiles` loop. Same offset issue for B index.
```cpp
// RAW — mul_bcast_rows: rotated * sin (row bcast)
mul_bcast_rows_init_short(rotated_in_interm_cb, sin_cb);
ACQ();
for (uint32_t j = 0; j < Wt; ++j) {
    mul_tiles_bcast<BroadcastType::ROW>(rotated_in_interm_cb, sin_cb, j, j, j);
    pack_tile(j, sin_interm_cb, j);
}
REL();

// RAW — mul_bcast_rows: in * cos (row bcast)
ACQ();
for (uint32_t j = 0; j < Wt; ++j) {
    mul_tiles_bcast<BroadcastType::ROW>(in_cb, cos_cb, j, j, j);
    pack_tile(j, cos_interm_cb, j);
}
REL();

// RAW — add: cos_result + sin_result
cb_wait_front(sin_interm_cb, Wt);
cb_wait_front(cos_interm_cb, Wt);
add_tiles_init(cos_interm_cb, sin_interm_cb);
ACQ();
for (uint32_t j = 0; j < Wt; ++j) {
    add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
    pack_tile(j, out_cb, j);
}
REL();
```
The `mul_bcast<ROW>` loops use 0-based B index — migratable:
```cpp
// Replacement for mul_bcast_rows: rotated * sin
mul<BroadcastDim::ROW,
    BinaryInputPolicy::WaitUpfrontNoPop,   // rotated_in_interm_cb already in CB
    BinaryInputPolicy::WaitUpfrontNoPop>(  // sin_cb persists
    rotated_in_interm_cb, sin_cb, sin_interm_cb, BinaryInputBlockShape::row(Wt));

// Replacement for mul_bcast_rows: in * cos
mul<BroadcastDim::ROW,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    in_cb, cos_cb, cos_interm_cb, BinaryInputBlockShape::row(Wt));

// Replacement for final add
add<BroadcastDim::NONE,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    cos_interm_cb, sin_interm_cb, out_cb, BinaryInputBlockShape::row(Wt));
```

---

#### `rotary_embedding_llama_fused_qk/.../rotary_embedding_llama_sharded_row_major.cpp`
Single-tile version. Same structure as sharded but with `Wt=1`.
```cpp
// RAW — identical pattern, one tile each
mul_tiles_init(rotated_in_interm_cb, sin_cb);
ACQ(); mul_tiles(rotated_in_interm_cb, sin_cb, 0, 0, 0); pack_tile(0, sin_interm_cb, 0); REL();

mul_tiles_init(in_cb, cos_cb);
ACQ(); mul_tiles(in_cb, cos_cb, 0, 0, 0); pack_tile(0, cos_interm_cb, 0); REL();

add_tiles_init(cos_interm_cb, sin_interm_cb);
ACQ(); add_tiles(cos_interm_cb, sin_interm_cb, 0, 0, 0); pack_tile(0, out_cb, 0); REL();
```
All three sections migratable (no offset):
```cpp
mul(rotated_in_interm_cb, sin_cb, sin_interm_cb, BinaryInputBlockShape::single());
mul(in_cb, cos_cb, cos_interm_cb, BinaryInputBlockShape::single());
add(cos_interm_cb, sin_interm_cb, out_cb, BinaryInputBlockShape::single());
```

---

### Phase 2a — rmsnorm

#### `rmsnorm_distributed/.../rmsnorm_post_allgather.cpp`
Full pipeline: add→rsqrt PostOp + mul_bcast_cols + mul_bcast_rows + add_bcast_rows.
```cpp
// RAW — add(var, eps) + rsqrt PostOp
reconfig_data_format(cb_var, cb_eps);
pack_reconfig_data_format(cb_recip_sqrt_var);
add_tiles_init(cb_var, cb_eps);
ACQ();
add_tiles(cb_var, cb_eps, 0, 0, 0);
rsqrt_tile_init<LEGACY_RSQRT>(); rsqrt_tile<LEGACY_RSQRT>(0);
pack_tile(0, cb_recip_sqrt_var);
REL();

// RAW — mul_bcast_cols: x * recip_sqrt_var
reconfig_data_format(cb_norm_x_input, cb_recip_sqrt_var);
pack_reconfig_data_format(normed_output_cb);
mul_bcast_cols_init_short(cb_norm_x_input, cb_recip_sqrt_var);
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    cb_wait_front(cb_norm_x_input, blk);
    cb_reserve_back(normed_output_cb, blk);
    ACQ();
    for (uint32_t wtr = 0; wtr < blk; wtr++) {
        mul_tiles_bcast_cols(cb_norm_x_input, cb_recip_sqrt_var, wtr, 0, wtr);
        pack_tile(wtr, normed_output_cb);
    }
    REL();
}

// RAW — mul_bcast_rows: x_normed * gamma
reconfig_data_format(cb_x_normed, cb_gamma);
pack_reconfig_data_format(cb_times_gamma_out);
mul_bcast_rows_init_short(cb_x_normed, cb_gamma);
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    cb_wait_front(cb_x_normed, blk);
    cb_reserve_back(cb_times_gamma_out, blk);
    ACQ();
    for (uint32_t wtr = 0; wtr < blk; wtr++) {
        mul_tiles_bcast_rows(cb_x_normed, cb_gamma, wtr, wt + wtr, wtr);
        pack_tile(wtr, cb_times_gamma_out);
    }
    REL();
}

// RAW — add_bcast_rows: x_normed_scaled + beta
reconfig_data_format(cb_times_gamma_out, cb_beta);
pack_reconfig_data_format(cb_out);
add_bcast_rows_init_short(cb_times_gamma_out, cb_beta);
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    cb_wait_front(cb_times_gamma_out, blk);
    cb_reserve_back(cb_out, blk);
    ACQ();
    for (uint32_t wtr = 0; wtr < blk; wtr++) {
        add_tiles_bcast_rows(cb_times_gamma_out, cb_beta, wtr, wt + wtr, wtr);
        pack_tile(wtr, cb_out);
    }
    REL();
}
```
```cpp
// REPLACEMENT (requires Gap #1 for SRCB_ONLY between stages)
// Stage 1: add + rsqrt
add(cb_var, cb_eps, cb_recip_sqrt_var, BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) {
        rsqrt_tile_init<LEGACY_RSQRT>(); rsqrt_tile<LEGACY_RSQRT>(dst_idx);
    });

// Stage 2: mul_bcast_cols (SRCB_ONLY reconfig from prior stage — Gap #1)
mul<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::SRCB_ONLY>(      // Gap #1
    cb_norm_x_input, cb_recip_sqrt_var, normed_output_cb,
    BinaryInputBlockShape::of(NCHt, Wt));

// Stage 3: mul_bcast_rows (SRCB_ONLY reconfig — Gap #1)
mul<BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::SRCB_ONLY>(      // Gap #1
    cb_x_normed, cb_gamma, cb_times_gamma_out,
    BinaryInputBlockShape::of(NCHt, Wt));

// Stage 4: add_bcast_rows (SRCB_ONLY reconfig — Gap #1)
add<BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::SRCB_ONLY>(      // Gap #1
    cb_times_gamma_out, cb_beta, cb_out,
    BinaryInputBlockShape::of(NCHt, Wt));
```

---

#### `rmsnorm_distributed/.../rmsnorm_pre_allgather.cpp`
#### `rmsnorm_distributed/.../rmsnorm_pre_allgather_2d.cpp`
#### `fused_distributed_rmsnorm/.../rmsnorm_pre_allgather.cpp`
Pattern — `mul_tiles(cb_inp, cb_inp, ...)` self-multiply for x². Cumulative wait (`cb_wait_front(cb_inp, wt + blk)`), non-zero start index.
```cpp
// RAW
mul_tiles_init(cb_inp, cb_inp);
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    cb_wait_front(cb_inp, wt + blk);   // cumulative: holds all tiles seen so far
    cb_reserve_back(cb_x2, blk);
    ACQ();
    for (uint32_t wtr = 0; wtr < blk; wtr++) {
        mul_tiles(cb_inp, cb_inp, wt + wtr, wt + wtr, wtr);
        pack_tile(wtr, cb_x2, wt + wtr);
    }
    REL();
    cb_push_back(cb_x2, blk);
}
```
Note: cumulative wait (waits for `wt + blk` tiles cumulatively, not just `blk`) and non-zero A/B start index `wt + wtr`. The helper always waits for `blk` tiles per chunk. **Not directly migratable without a `WaitCumulative` policy or upfront wait variant.**

The `_2d` variant additionally has a merge-core add:
```cpp
// RAW — merge core accumulation
add_tiles_init(cb_x2_merge, cb_zero, true);
ACQ();
for (uint32_t i = 0; i < num_cores_y; i++)
    add_tiles(cb_x2_merge, cb_zero, i, 0, dst0);   // accumulate into dst0
tile_regs_commit(); tile_regs_wait();
pack_tile(dst0, cb_out_final);
tile_regs_release();
```
This maps to `BinaryAccumulate` but B is a zero-tile identity rather than a real accumulator CB.

---

#### `fused_distributed_rmsnorm/.../rmsnorm_post_allgather.cpp`
#### `ccl/rms_allgather/.../rms_compute.cpp`
Same pipeline as `rmsnorm_post_allgather.cpp` above. Identical replacement applies.

---

### Phase 2b — softmax

#### `softmax/device/kernels/attention/compute/softmax.cpp`
Four distinct binary stages. Each stage changes the CB pair, requiring `reconfig_data_format*` between them.
```cpp
// RAW — Stage 1: mul_bcast_scalar (scale * input)
mul_tiles_bcast_scalar_init_short(cb_in0, cb_fused_scale);
for (uint32_t wt = 0; wt < Wt; wt += ndst) {
    tile_regs_acquire();
    cb_in0_obj.wait_front(ndst);
    cb_scale_mask_obj.reserve_back(ndst);
    for (uint32_t wt8 = 0; wt8 < ndst; wt8++)
        mul_tiles_bcast_scalar(cb_in0, cb_fused_scale, wt8, 0, wt8);
    tile_regs_commit(); tile_regs_wait();
    for (uint32_t wt8 = 0; wt8 < ndst; wt8++) pack_tile(wt8, cb_scale_mask);
    tile_regs_release();
    cb_scale_mask_obj.push_back(ndst); cb_in0_obj.pop_front(ndst);
}

// RAW — Stage 2: add or add_bcast_rows (scaled + mask), conditional on CAUSAL_MASK
reconfig_data_format(cb_scale_mask, cb_fused_attn);
for (uint32_t wt = 0; wt < Wt; wt += ndst) {
    tile_regs_acquire();
#ifdef CAUSAL_MASK
    add_tiles(cb_scale_mask, cb_fused_attn, wt8, wt + wt8, wt8);
#else
    add_tiles_bcast_rows(cb_scale_mask, cb_fused_attn, wt8, wt + wt8, wt8);
#endif
    tile_regs_commit(); tile_regs_wait(); /* pack... */ tile_regs_release();
}

// RAW — Stage 3: sub_bcast_cols (x - max), fused exp PostOp
sub_bcast_cols_init_short(cb_in, cb_max);
for (uint32_t wt = 0; wt < Wt; wt += ndst) {
    tile_regs_acquire();
    for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
        sub_tiles_bcast_cols(cb_in, cb_max, wt + wt8, 0, wt8);
    }
    cb_out_obj.reserve_back(ndst);
    for (uint32_t wt8 = 0; wt8 < ndst; wt8++) exp_tile<EXP_APPROX>(wt8);
    tile_regs_commit(); tile_regs_wait(); /* pack... */ tile_regs_release();
}

// RAW — Stage 4: mul_bcast_cols (exp * 1/sum_exp)
mul_bcast_cols_init_short(cb_exps, cb_recipsumexps);
for (uint32_t wt = 0; wt < Wt; wt += ndst) {
    tile_regs_acquire();
    for (uint32_t wt8 = 0; wt8 < ndst; wt8++)
        mul_tiles_bcast<BroadcastType::COL>(cb_exps, cb_recipsumexps, wt + wt8, 0, wt8);
    tile_regs_commit(); tile_regs_wait(); /* pack... */ tile_regs_release();
}
```
```cpp
// REPLACEMENT (all stages require Gap #1 for cross-stage SRCB_ONLY reconfig)
// Stage 1: mul_bcast_scalar
mul<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    cb_in0, cb_fused_scale, cb_scale_mask, BinaryInputBlockShape::of(Ht, Wt));

// Stage 2: add or add_bcast_rows (conditional)
#ifdef CAUSAL_MASK
add<BroadcastDim::NONE,
#else
add<BroadcastDim::ROW,
#endif
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::SRCA_ONLY>(      // Gap #1
    cb_scale_mask, cb_fused_attn, cb_x, BinaryInputBlockShape::of(Ht, Wt));

// Stage 3: sub_bcast_cols + exp PostOp
sub<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::SRCB_ONLY>(      // Gap #1
    cb_in, cb_max, cb_out, BinaryInputBlockShape::of(Ht, Wt),
    [](uint32_t dst_idx) { exp_tile<EXP_APPROX>(dst_idx); });

// Stage 4: mul_bcast_cols
mul<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::SRCA_ONLY>(      // Gap #1
    cb_exps, cb_recipsumexps, cb_out0, BinaryInputBlockShape::of(Ht, Wt));
```

`softmax_sharded.cpp` and `softmax_large_tensor.cpp` follow the same four-stage pattern with minor blocking/sharding differences; the same replacement template applies.

---

### Phase 2c — layernorm

All layernorm variants share the same five-stage pipeline. The variations are: welford vs non-welford stats (affects pre-compute, not binary stages), sharded vs non-sharded (affects CB names), and post-allgather vs full (affects which stages are present).

Core migratable pipeline (representative from `layernorm.cpp`):
```cpp
// RAW — Stage 1: add(var, eps) + rsqrt PostOp (same as rmsnorm)
reconfig_data_format(cb_var, cb_eps);
add_tiles_init(cb_var, cb_eps);
ACQ();
add_tiles(cb_var, cb_eps, 0, 0, 0);
rsqrt_tile_init<LEGACY_RSQRT>(); rsqrt_tile<LEGACY_RSQRT>(0);
pack_tile(0, cb_recip_sqrt_var);
REL();

// RAW — Stage 2: sub_bcast_cols (x - mean)
reconfig_data_format_srcb(cb_stats_reduced, cb_recip_sqrt_var);  // SRCB switch
sub_bcast_cols_init_short(cb_inp, cb_stats_reduced);
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    cb_wait_front(cb_inp, blk);
    cb_reserve_back(cb_x_minus_mean, blk);
    ACQ();
    for (uint32_t wtr = 0; wtr < blk; wtr++) {
        sub_tiles_bcast_cols(cb_inp, cb_stats_reduced, wtr, 1, wtr);  // index=1: mean
        pack_tile(wtr, cb_x_minus_mean);
    }
    REL();
}

// RAW — Stage 3: mul_bcast_cols ((x-mean) * recip_sqrt_var)
reconfig_data_format_srcb(cb_stats_reduced, cb_recip_sqrt_var);  // SRCB switch (again)
mul_bcast_cols_init_short(cb_norm_x_input, cb_recip_sqrt_var);
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    // ... same blk loop
}

// RAW — Stage 4: mul_bcast_rows (normed * gamma)
reconfig_data_format(cb_norm_x_input, cb_gamma);
mul_bcast_rows_init_short(cb_x_normed, cb_gamma);
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    // ...
}

// RAW — Stage 5: add_bcast_rows (result + beta)
reconfig_data_format_srcb(cb_gamma, cb_beta);  // SRCB switch
add_bcast_rows_init_short(cb_times_gamma_out, cb_beta);
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    // ...
}
```
```cpp
// REPLACEMENT — requires Gap #1 for all SRCB_ONLY transitions
add(cb_var, cb_eps, cb_recip_sqrt_var, BinaryInputBlockShape::single(),
    [](uint32_t d) { rsqrt_tile_init<LEGACY_RSQRT>(); rsqrt_tile<LEGACY_RSQRT>(d); });

sub<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::SRCB_ONLY>(     // Gap #1
    cb_inp, cb_stats_reduced, cb_x_minus_mean, BinaryInputBlockShape::of(NCHt, Wt));

mul<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::SRCB_ONLY>(     // Gap #1
    cb_x_minus_mean, cb_recip_sqrt_var, normed_output_cb, BinaryInputBlockShape::of(NCHt, Wt));

mul<BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    cb_x_normed, cb_gamma, cb_times_gamma_out, BinaryInputBlockShape::of(NCHt, Wt));

add<BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::SRCB_ONLY>(     // Gap #1
    cb_times_gamma_out, cb_beta, cb_out, BinaryInputBlockShape::of(NCHt, Wt));
```
This replacement template applies to all 14 layernorm variants. Differences per file:
- `layernorm_large_tensor_welford.cpp` (9 `_srcb` sites): welford stats add an extra `mul_bcast_cols` + `add_bcast_cols` pair before Stage 2
- `layernorm_sharded*.cpp`: CB names differ; `blk` may vary; same binary stages
- `layernorm_distributed/*.cpp`: post-allgather variants only contain stages 3–5

---

### Phase 2d — batch_norm and depthwise conv

#### `batch_norm/device/kernels/compute/batch_norm_kernel.cpp`
```cpp
// RAW — Stage 1: add(var, eps) + rsqrt
add_tiles_init_with_dt(cb_batch_var, cb_eps);
ACQ();
add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);
rsqrt_tile_init(); rsqrt_tile(dst0);
pack_tile_with_dt(dst0, cb_den);
REL();

// RAW — Stage 2: sub(input, mean)
sub_tiles_init(cb_other, cb_bcast);
ACQ();
sub_tiles(cb_other, cb_bcast, 0, 0, 0);
// ... dest-reuse mul into dst0 (mul by recip from cb_den)
pack_tile_with_dt(0, cb_affine_or_out);
REL();

// RAW — Stage 3: mul(result, weight)
mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
ACQ();
mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);
pack_tile_with_dt(dst0, cb_scaled_output);
REL();

// RAW — Stage 4: add(result, bias)
add_tiles_init_with_dt(cb_tmp_1, cb_bias);
ACQ();
add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0);
pack_tile_with_dt(dst0, cb_output_0);
REL();
```
All four stages are now migratable. Stage 2 previously required `binary_dest_reuse_tiles` (dest-aliased fused multiply with no helper equivalent). `DestReuseMul<CB, Slot>` PostOp now covers this case.

```cpp
// REPLACEMENT — all four stages
// Stage 1: add(var, eps) + rsqrt
add(cb_batch_var, cb_eps, cb_den, BinaryInputBlockShape::single(),
    [](uint32_t d) { rsqrt_tile_init(); rsqrt_tile(d); });

// Stage 2: sub(input, mean) + DestReuseMul (previously raw)
// cb_den must be waited upfront before the loop; DestReuseMul always reads CB[0]
sub(cb_other, cb_bcast, cb_affine_or_out, BinaryInputBlockShape::single(),
    DestReuseMul<cb_den>{});

// Stage 3: mul(result, weight)
mul(cb_affine_or_out, cb_weight, cb_scaled_output, BinaryInputBlockShape::single());

// Stage 4: add(result, bias)
add(cb_tmp_1, cb_bias, cb_output_0, BinaryInputBlockShape::single());
```

---

#### `conv/conv2d/device/kernels/compute_depthwise_conv1d.cpp`
```cpp
// RAW — per input tile: mul then conditional add-accumulate
for (uint32_t i = 0; i < block_num_tiles; i++) {
    cb_wait_front(in1_cb_id, 1); cb_wait_front(in0_cb_id, 1);
    cb_reserve_back(eltwise_mul_partials_cb_cb_id, 1);
    mul_tiles_init(in0_cb_id, in1_cb_id);
    ACQ();
    mul_tiles(in0_cb_id, in1_cb_id, 0, 0, 0);
    pack_tile(0, eltwise_mul_partials_cb_cb_id);
    REL();
    cb_push_back(eltwise_mul_partials_cb_cb_id, 1);
    cb_pop_front(in0_cb_id, 1); cb_pop_front(in1_cb_id, 1);

    if (idx == 0) {
        // first partial: copy
        copy_tile_to_dst_init_short(eltwise_mul_partials_cb_cb_id);
        ACQ(); copy_tile(eltwise_mul_partials_cb_cb_id, 0, 0); pack_tile(0, out_cb_id); REL();
    } else {
        // subsequent: accumulate
        add_tiles_init(eltwise_mul_partials_cb_cb_id, out_cb_id);
        ACQ();
        add_tiles(eltwise_mul_partials_cb_cb_id, out_cb_id, 0, 0, 0);
        pack_tile(0, temp_sum_cb);
        REL();
    }
}
```
The `mul` step is directly migratable; the add-accumulate is conditional on `idx` (copy vs add first iteration). **Partial migration — mul step:**
```cpp
mul(in0_cb_id, in1_cb_id, eltwise_mul_partials_cb_cb_id,
    BinaryInputBlockShape::single());
// idx==0 copy branch and accumulation loop stay raw
```

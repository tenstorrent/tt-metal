# Fused Matmul Helpers — Investigation

**Commit**: a62a03c2181e083484fb6ba0496610b2d66c0ba7
**Branch**: wransom/fused2

This document analyzes the Tier 1 and Tier 2 target kernels in depth. Every factual claim
has a unique ID (C-001...) with file:line anchor. Phase 2 verifies these claims.

---

## Group C — Core Fused Matmul (bmm_large_block_zm_fused_bias_activation)

**File**: `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`
**Abbreviation**: C1

### C1 Structural Overview

The kernel has a 4-level loop nest: batch → bh → bw → K-blocks. Inside the K-block loop,
a 2-level subblock loop (in0_subblocks × in1_subblocks) performs the matmul_block LLK calls.
After the K-loop, optional post-processing phases run: bias add, then untilize.

### Claim Ledger — C1

| ID | Claim | File:Line | Excerpt |
|----|-------|-----------|---------|
| C-001 | Outer loop is `batch × num_blocks_h_dim × num_blocks_w_dim` | C1:219,231,232 | `for (uint32_t b = 0; b < batch; b++)` / `for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh)` / `for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw)` |
| C-002 | PACK_RELU is disabled at the start of each bh×bw output block (when multi-batch/block) | C1:236-241 | `PACK((llk_pack_relu_config(ReluType::NO_RELU)));` inside `#ifdef PACK_RELU` |
| C-003 | PACK_RELU is enabled on last K-block only when FUSE_BIAS is NOT defined | C1:250-255 | `#if not defined FUSE_BIAS and defined PACK_RELU` → `llk_pack_relu_config(ReluType::ZERO_RELU)` |
| C-004 | in0_transpose path calls transpose_tile_block then mm_block_init_short_with_dt to reconfigure | C1:257-274 | `transpose_tile_block<in0_block_num_tiles>(in0_transpose_cb_id, in0_cb_id)` at line 264 |
| C-005 | PACKER_L1_ACC is explicitly disabled (set to 0) before the transpose phase | C1:261-263 | `PACK((llk_pack_reconfig_l1_acc(0)));` inside `#ifdef PACKER_L1_ACC` |
| C-006 | K-block loop iterates `num_blocks_inner_dim` times, `last_out` is true on final iteration | C1:247-248 | `for (uint32_t block = 0; block < num_blocks_inner_dim; block++)` |
| C-007 | Partial results are reloaded from interm CB via `reload_from_cb_to_dst` when `enable_reload` is true | C1:284-294 | `reload_from_cb_to_dst(in0_cb_id, in1_cb_id, mm_partials_cb_id, ...)` |
| C-008 | The `reload_from_cb_to_dst` function calls `copy_tile_to_dst_init_short_with_dt`, `copy_block_matmul_partials`, then `mm_block_init_short_with_dt` to restore matmul state | C1:84-106 | Function defined at lines 84-106 |
| C-009 | SKIP_COMPUTE guard wraps the entire matmul_block inner loop | C1:296-323 | `#ifndef SKIP_COMPUTE` ... `#endif` |
| C-010 | SFPU activation on last K-block fires only when FUSE_BIAS is NOT defined | C1:327-331 | `#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION` |
| C-011 | On last_out, pack_reconfig_data_format is called under FP32_DEST_ACC_EN or PACKER_L1_ACC | C1:337-339 | `PACK((pack_reconfig_data_format(mm_out_cb_id)));` |
| C-012 | On last_out with PACKER_L1_ACC+FUSE_BIAS: L1_ACC=0 for block 0, L1_ACC=1 for subsequent blocks | C1:341-347 | `if (block == 0) { PACK((llk_pack_reconfig_l1_acc(0))); } else { PACK((llk_pack_reconfig_l1_acc(1))); }` |
| C-013 | On last_out with PACKER_L1_ACC but no FUSE_BIAS: L1_ACC is always disabled | C1:348-349 | `PACK((llk_pack_reconfig_l1_acc(0)));` |
| C-014 | Non-last-out spill: block==0 reserves out_cb to prevent interm overwrite (shared memory) | C1:363-366 | `out_cb.reserve_back(out_num_tiles_to_wait); out_num_tiles_to_wait += out_subblock_num_tiles;` |
| C-015 | Non-last-out L1_ACC spill has 3 cases: block==0 disable, block==1 enable, in0_transpose re-enable | C1:371-381 | Three-way if/else chain |
| C-016 | PACKER_L1_ACC+FUSE_BIAS post-loop: wait/pop interm for all non-last blocks, never reload | C1:396-404 | `enable_reload = false` at line 404 |
| C-017 | PACKER_L1_ACC without FUSE_BIAS post-loop: wait/pop all but last 2 blocks, reload on K-2 | C1:406-413 | `enable_reload = true` at line 412 |
| C-018 | Without PACKER_L1_ACC: always enable reload when spill | C1:415-419 | `enable_reload = true` inside `if constexpr (spill)` |
| C-019 | FUSE_BIAS section: PACK_RELU enabled, L1_ACC disabled, data format reconfigured, then bias add loop | C1:425-484 | `add_bcast_rows_init_short(mm_partials_cb_id, bias_cb_id)` at line 438 |
| C-020 | FUSE_BIAS bias loop waits for bias_cb upfront, does NOT pop bias_cb (except when num_blocks_w_dim > 1) | C1:440,481-483 | `bias_cb.wait_front(in1_block_w)` / `if constexpr (num_blocks_w_dim > 1) { bias_cb.pop_front(in1_block_w); }` |
| C-021 | SFPU activation inside FUSE_BIAS fires after add_tiles_bcast_rows, before tile_regs_commit | C1:462-466 | `SFPU_OP_FUNC_ACTIVATION` inside `#ifdef SFPU_OP_INIT_ACTIVATION` |
| C-022 | Untilize path uses `reblock_and_untilize` template function that copies tiles then calls pack_untilize_dest | C1:108-142,485-505 | `reblock_and_untilize<out_subblock_w, out_block_w>(...)` |
| C-023 | `reblock_and_untilize` gathers sub-blocks column-wise: copies tiles from interm, packs with `pack_untilize_dest<out_subblock_w, out_block_w>` row by row | C1:118-141 | Inner loop over h×n subblocks |
| C-024 | After untilize, PACK_RELU is disabled and pack_untilize_uninit is called | C1:486-488,504 | `llk_pack_relu_config(ReluType::NO_RELU)` / `pack_untilize_uninit(mm_partials_cb_id)` |
| C-025 | Multi-batch/block end: reconfig_data_format and mm_block_init_short reconfigure for next iteration | C1:506-517 | `mm_block_init_short(in0_cb_id, in1_cb_id, ...)` |
| C-026 | Untilize path, when no FUSE_BIAS: reconfig_data_format_srca from in1_cb_id to mm_partials_cb_id | C1:489-497 | `reconfig_data_format_srca(in1_cb_id, mm_partials_cb_id)` + pack_reconfig + L1_ACC disable |
| C-027 | mm_block_init is called once before outer loops with mm_partials_cb_id as the third arg | C1:217-218 | `mm_block_init(in0_cb_id, in1_cb_id, mm_partials_cb_id, ...)` |
| C-028 | The SFPU_OP_INIT_ACTIVATION macro is called once at top of kernel_main (not per-subblock) | C1:205-207 | `#ifdef SFPU_OP_INIT_ACTIVATION` / `SFPU_OP_INIT_ACTIVATION` |
| C-029 | FUSE_BIAS bias add uses `add_tiles_bcast_rows(mm_partials_cb_id, bias_cb_id, i, bcast_tile_idx, i)` — src tile index == dst tile index | C1:450 | Tile-by-tile bcast-rows add in the bias loop |
| C-030 | `mm_out_cb_id` equals `mm_partials_cb_id` when FUSE_BIAS, equals `untilize_mode_out_cb_id` otherwise | C1:198-202 | `#ifdef FUSE_BIAS` → `mm_out_cb_id = mm_partials_cb_id` / `#else` → `mm_out_cb_id = untilize_mode_out_cb_id` |
| C-031 | `untilize_mode_out_cb_id` equals `mm_partials_cb_id` when untilize_out, equals `out_cb_id` otherwise | C1:184 | `constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;` |
| C-032 | `get_batch_from_reader` uses mailbox_read to check batch validity from BRISC | C1:220-229 | `UNPACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)` |
| C-033 | pack_reconfig_data_format(mm_partials_cb_id) is called at start of each bh/bw block when multi-batch/block | C1:243-245 | `PACK((pack_reconfig_data_format(mm_partials_cb_id)));` |

---

## Group C2/C3 — Gathered Variants

**C2**: `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`
**C3**: `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`

### Claim Ledger — C2/C3

| ID | Claim | Evidence |
|----|-------|----------|
| C-034 | C2 and C3 are NOT identical — they differ in CB API style (experimental::CircularBuffer vs raw), CB ID resolution (named args vs positional), ring traversal direction (forward vs reverse), and per-shard width handling (dynamic vs static) | Detailed diff in investigation agents |
| C-035 | Neither C2 nor C3 supports FUSE_BIAS — the entire bias path is absent | Neither file contains `FUSE_BIAS`, `add_bcast_rows`, or `bias_cb_id` |
| C-036 | Neither C2 nor C3 supports in0 transpose — `transpose_tile_block` and `in0_transpose_tile` are absent | Neither file contains `transpose_wh` or `in0_transpose` |
| C-037 | Both C2 and C3 have ring-based all-gather logic (ring_idx, sync CBs, per-batch CB arrays) that has no equivalent in C1 | New infrastructure: ring_idx, sync_cb, sync_cb2, CORE_TYPE enum, CB pointer management helpers |
| C-038 | C2/C3 handle untilize inline per-subblock in the last_out branch, not as a post-loop step | C2: `pack_untilize_dest<out_subblock_num_tiles>(mm_out_cb_id)` inside last_out |
| C-039 | C2/C3 collapse the bh/bw loops into a single `block` loop over ring_size | C2/C3 have `for (uint32_t block = 0; block < num_blocks; block++)` |
| C-040 | C2/C3 toggle between in0_cb_id and in2_cb_id based on block index (block 0 uses in0, others use in2) | C2 lines 295-296 |

**Migration assessment for C2/C3**: These are fundamentally different kernels sharing only the core matmul_block LLK loop with C1. The ring-based all-gather infrastructure, per-batch CB arrays, and sync mechanism make them **Tier 3** (out of scope for generic fused helper). The matmul_block inner loop could potentially use the existing simple helper, but the surrounding orchestration is too specialized.

---

## Group D — Conv Fused Matmul (conv_bmm_tilize)

**File**: `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp`
**Abbreviation**: D1

### D1 Structural Overview

Loop nest: `in1_num_blocks_w → in0_num_blocks_h → in0_num_blocks_w (K-blocks)`. Before each
K-block (or group of K-blocks), input is tilized from raw buffer to tilized CB. The matmul
core uses the same `ckernel::matmul_block` LLK as C1. After K-blocks: optional bias add,
then optional untilize. Unlike C1, uses compile-time bools (not #ifdef) for most feature flags.

### Claim Ledger — D1

| ID | Claim | File:Line | Excerpt |
|----|-------|-----------|---------|
| C-041 | Outer loop is `in1_num_blocks_w × in0_num_blocks_h`, K-loop is `in0_num_blocks_w` | D1:267-268,280 | `for (uint32_t in1_block_w_i = 0; ...)` / `for (uint32_t in0_block_h_i = 0; ...)` / `for (uint32_t in0_block_w_i = 0; ...)` |
| C-042 | Tilize is called before K-blocks, conditioned on `in0_block_w_i % in0_nblocks_w_tilize == 0` (non-height_sharded) or every K-iteration (height_sharded) | D1:283,315 | `if (in0_block_w_i % in0_nblocks_w_tilize == 0)` |
| C-043 | Uses `compute_kernel_lib::tilize` helper (already migrated) and `fast_tilize_block` for activation reuse path | D1:46-52,57-59 | `compute_kernel_lib::tilize<...>(in_num_subblocks)` / `fast_tilize_block(in_cb_id, in_block_w, out_cb_id)` |
| C-044 | `mm_block_init_short_with_both_dt` is called after tilize to reconfigure both src datapaths | D1:305-313,353-361 | Different args for height_sharded vs non-height_sharded path |
| C-045 | skip_compute is a runtime arg that bypasses matmul but allows tilize to proceed | D1:256-258,367-372 | `skip_compute = (bool)get_arg_val<uint32_t>(0)` / `cb_pop_front(mm_in0_cb_id, ...); continue;` |
| C-046 | L1_ACC toggle uses `pack_reconfig_l1_acc()` (high-level wrapper) not `llk_pack_reconfig_l1_acc()` | D1:456-466 | `pack_reconfig_l1_acc(0)` / `pack_reconfig_l1_acc(1)` / `pack_reconfig_l1_acc(fuse_bias ? 1 : 0)` |
| C-047 | L1_ACC has 3 states: block==0 disable, last_block enable-if-bias/disable-if-not, middle enable | D1:459-465 | Three-way if/else chain |
| C-048 | `curr_matmul_out_cb` switches from `matmul_partials_cb` to `mm_out_cb_id` on last K-block when !fuse_bias | D1:376-383 | `curr_matmul_out_cb = mm_out_cb_id;` |
| C-049 | Extensive partials CB FIFO pointer manipulation (fifo_rd_ptr/fifo_wr_ptr) at ~8 locations throughout the kernel | D1:264-265,276-277,480-481,492-493,504-505,518-519,523-526,536,592-593 | Direct `get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr` manipulation |
| C-050 | bias_block_offset tracks absolute bias tile position across in1_num_blocks_w iterations | D1:230,636-638 | `bias_block_offset += in1_block_w` at end of outer loop |
| C-051 | Untilize has two paths: `packer_untilize` uses reblock_and_untilize (same as C1), non-packer_untilize uses `compute_kernel_lib::untilize` helper | D1:608-626 | `if constexpr (packer_untilize)` branch |
| C-052 | The `reblock_and_untilize` function in D1 (lines 150-178) is identical to C1 (lines 108-142) except D1 uses raw CB API vs C1 uses experimental::CircularBuffer wrapper | D1:150-178, C1:108-142 | Same algorithm: gather subblocks by row, copy_tile, pack_untilize_dest |
| C-053 | D1 uses compile-time bools (constexpr bool pack_relu, packer_l1_acc, fuse_bias, etc.) not #ifdef for most feature flags | D1:211-216 | `constexpr bool pack_relu = get_compile_time_arg_val(29)` etc. |
| C-054 | D1 has split_reader support with two tilize paths (in0_cb_id and in0_cb_second_reader_id) | D1:300-303,332-335 | Two `tilize_in` calls with different CB sources |
| C-055 | D1 activation_reuse path directly manipulates CB fifo_wr_ptr and uses tilize_in_reuse_split_reader | D1:337-349 | Complex template with 11 parameters for activation window reuse |
| C-056 | D1 matmul core (subblock loop + matmul_block LLK calls) is structurally identical to C1 | D1:389-477, C1:279-388 | Same pattern: in0_subblock × in1_subblock → acquire → [reload] → matmul_block inner_dim loop → [sfpu] → commit → pack → release |
| C-057 | D1 does NOT have tile_regs_acquire inside enable_reload path — it calls acquire first, then copy_block_matmul_partials inside the same acquire block | D1:392-416 | `tile_regs_acquire()` before reload check, vs C1 which acquires THEN reloads |
| C-058 | D1 SFPU activation (inside K-loop, !fuse_bias) fires after matmul_block but before tile_regs_commit, same as C1 | D1:443-451 | `SFPU_OP_FUNC_ACTIVATION` inside `#ifdef SFPU_OP_INIT_ACTIVATION` with `!fuse_bias` guard |
| C-059 | D1 bias section after K-loop has same structure as C1: PACK_RELU enable, pack_reconfig, L1_ACC disable, reconfig_data_format, add_bcast_rows loop | D1:543-594 | Nearly identical to C1:425-484 |
| C-060 | D1 bias loop uses `bias_block_offset + in1_index_subblock_offset` for bias tile indexing — unlike C1 which uses just `in1_index_subblock_offset` | D1:563, C1:448-449 | D1 has global offset for multi-block-w support |

---

## Shared Patterns — Helpers Needed

### reblock_and_untilize (C-052)

Identical function in C1 and D1. Algorithm:
1. Wait for a full row of subblocks in interm_cb
2. For each tile row h in subblock:
   a. Reserve out_block_w in out_cb
   b. For each subblock n in row:
      - tile_regs_acquire
      - copy_tile from interm_cb to DST (subblock_w tiles)
      - tile_regs_commit/wait
      - pack_untilize_dest<subblock_w, block_w>(out_cb, 1, n)
      - tile_regs_release
   c. Push out_block_w to out_cb
3. Pop the row from interm_cb

This should be extracted as a standalone helper.

### transpose_tile_block (C-004)

Only used in C1. Algorithm:
1. Process in0_block_num_tiles in chunks of block_size (default 4)
2. Each chunk: wait_front → acquire → transpose_wh_tile per tile → commit → pop_front → reserve_back → wait → pack_tile per tile → release → push_back
3. Handle remainder tiles in a second pass

### Matmul Core (C-056)

The inner matmul computation (subblock loop + matmul_block LLK calls + partial reload +
K-blocking + spill/accumulation) is structurally identical between C1 and D1. The existing
`matmul_block` helper already handles this, but C1 and D1 add features beyond it:
- SFPU activation on last_out (C-010, C-058)
- PACK_RELU toggle (C-002, C-003)
- Variable pack target (mm_out_cb_id vs mm_partials_cb_id vs untilize_mode_out_cb_id) (C-030, C-031, C-048)
- in0_transpose per-K-block (C-004)
- Outer bh/bw/batch loops with reconfiguration between iterations (C-001, C-025, C-033)

---

## Encapsulation Boundary Analysis

What should a fused helper OWN vs EXPOSE to the caller:

**Helper should own:**
- K-blocking loop with spill/reload/L1_ACC management (already in matmul_block helper)
- Bias addition phase (already in bias_add helper)
- reblock_and_untilize phase (new helper)
- PACK_RELU enable/disable lifecycle
- Pack target selection (interm vs out, depending on bias/untilize)
- Data format reconfiguration between phases

**Helper should NOT own (caller's responsibility):**
- Outer batch/bh/bw loops (too many kernel-specific concerns: batch validity check, bias_block_offset, CB pointer resets)
- Tilize input (conv-specific, already has its own helper)
- in0_transpose (could be a PreKBlockFn or separate helper)
- SFPU_OP_INIT_ACTIVATION macro call (once at top, kernel-specific)
- mm_block_init call (once before loops)
- SKIP_COMPUTE check
- CB FIFO pointer manipulation (conv-specific)
- Gathered/ring infrastructure (too specialized)

**Key insight**: The fused helper should handle one "output block" iteration: the K-blocking loop + optional bias + optional untilize + PACK_RELU lifecycle. The caller handles the outer loops and any per-iteration setup (tilize, transpose, batch checks).

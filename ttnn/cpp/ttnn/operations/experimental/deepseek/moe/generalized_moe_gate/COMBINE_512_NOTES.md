# Generalized MoE Gate — 512-expert combine (A2): design, journey, pitfalls & solutions

Status: **✅ DONE.** `generalized_moe_gate` computes the true **global top-8** over **512 experts** in **one op**
on **Wormhole B0**. `test_generalized_moe_gate_512_global` passes (all params); the 256 path is not regressed.
(Chinese version: `COMBINE_512_NOTES.zh.md`.)

---

## 1. Goal & constraints

- Generalize the fused single-op gate from 256 → **256 / 384 / 512** experts in **ONE op**, computing the true
  **global top-8** (k=8). 256 stays the fast single op (~2.48 µs). Kimi = 384, Qwen = 512.
- **Single op is required** — softmax will be fused in later, so it cannot be split into two ops.
- Target arch: **Wormhole B0**. `fp32_dest_acc_en = false` (16-bit DEST); `dst_full_sync_en = true`.

## 2. Architecture — per-block "run" + combine

- **Input layout (slice):** each 256-expert block → face 0 of its own 32×32 tile; logits/bias sharded
  `num_blocks` tiles/core. `num_blocks = ceil(N/256)` (512 → 2).
- **Per block (`produce_run`):** run the proven 256 ungrouped pipeline up to a **re-mergeable top-8 RUN**
  (`merge16_to_run`, skipping normalize/step2). A "run" = `(bias, idx, score)` for 8 experts:
  - `bias` = the rank key (sigmoid_score + bias term),
  - `idx`  = the global expert id (uint16),
  - `score`= the sigmoid value (the output weight).
- **Combine:** get the two block-runs co-resident in DEST at SFPU offsets `{0,2}` (block1) and `{4,6}` (block0),
  then run the proven `combine_finalize` (`merge16_core` reads the 16 candidates at `{0,2,4,6}`, full bitonic
  sort → global top-8, normalize, step2 → output).

**The hard part** is getting *both* runs co-resident in DEST in the SFPU "math" layout that `merge16_core`
reads — block0's run has to survive while block1 is produced, and land at `{4,6}`. The final, working answer is
the **merge-only acquire** (§3).

## 3. The working recipe

```
num_blocks == 2 (combine path):

  # --- stash BOTH blocks to L1 (each via the proven round-trip) ---
  process_block_to_run<0>()    # block0 -> L1 run CBs page 0
  process_block_to_run<1>()    # block1 -> L1 run CBs page 1
    # process_block_to_run<b>:
    #   copy_tile(input_indices, b, 1)                 # per-block GLOBAL indices (tile b = arange + b*256)
    #   produce_run<...,0,2,idx_offset=0>              # run at math {0,2}
    #   relocate_run<0,2,0,4>                          # {0,2} -> {0,4}  (step2 expects the {0,4} finalize layout)
    #   step2_only<false>                              # math -> standard (now transposes 3 tiles: score+idx+BIAS)
    #   per field: pack_untilize_dest(tile_dst_rt_offset = 0/1/2)  # standard DEST -> row-major L1

  # --- tilize all fields BEFORE the merge acquire (tilize self-manages DEST) ---
  hw_startup(run_scores_cb, cb_tilize);  tilize run_scores x num_blocks -> cb_tilize p0,p1   (bf16)
  (reuse)                                tilize run_bias   x num_blocks -> cb_tilize p2,p3   (bf16)
  hw_startup(run_idx_cb, cb_tilize_idx); tilize run_idx    x num_blocks -> cb_tilize_idx p0,p1 (uint16)

  # --- merge-only acquire: NO produce_run inside it ---
  tile_regs_acquire()
  for (run, dst) in [(block1, {0,2}), (block0, {4,6})]:
     for field in [score(HI16), idx(LO16), bias(mode0)]:
        reconfig_data_format_srca(cb_tilize / cb_tilize_idx)
        transpose_wh_init_short(...)
        transpose_wh_tile(cb_tilize[page], 0, 3)                 # standard tiled -> interm (DEST tile 3), math {0,4}
        place_field_from_interm<field, dst_lo, dst_hi, src=0,4>()# SFPU row/col-selective copy interm{0,4} -> home{dst}
  combine_init()
  UNPACK(set_srcb_dummy_valid())                                 # AFTER the transposes, before step2
  combine_finalize()                                             # merge16 {0,2}+{4,6} -> top8 + normalize + step2
  tile_regs_commit(); pack tile0->scores_out, tile1->idx_out
```

Why **merge-only acquire**: `produce_run` leaves SFPU/SrcB/addrmod state that *poisons a same-acquire
`transpose_wh`*. By stashing both blocks first and doing the restore+merge in a fresh acquire with no
`produce_run`, the restore runs in a clean state (the same state the proven 256 stash-isolation runs in).

## 4. DEST layout facts (WH B0, verified)

- Regions: `scores @ off 0`, `indices @ 64`, `bias @ 128`, `interm @ 192` (`dst_tile_offset = 64`).
  `copy_tile`/`pack_tile`/`transpose_wh_tile` tile index k ↔ SFPU offset k*64 (tile 0/1/2/3).
- A run lives at **2 SFPU offset-pairs** `{store_lo, store_hi}` (4 candidates each). `merge16_core` reads
  `{0,2,4,6}` (two runs: `{0,2}` and `{4,6}`).
- **`merge16_core`'s `{0,2,4,6}` offsets are ROWS** of the face (each SFPLOAD reads a row's lanes), not columns.
- **Concatted field encoding:** a candidate is `idx (LO16) | score (HI16)` in one 32-bit SFPU LREG; stored SPLIT
  as `score → scores region (HI16)`, `idx → indices region (LO16)`, `bias → bias region (mode 0 / full)`.
  `merge16_core` re-loads `idx (LO16)` + `score (HI16)` and concats; it **sorts by `bias`** and index-tracks the
  `idx|score` along the swaps.
- `fp32_dest_acc_en = false` (16-bit dest). `dst_full_sync_en = true`.
- **bf16 ↔ raw 16-bit:** the DEST datatype is determined by the **CB compile-time format metadata**
  (`datatype_to_dataformat_converter`), not page size. `UInt16` is the integer path; `RawUInt16` maps to the
  float16 pack path (a corruption trap — don't use it for ids).

## 5. Pitfalls & solutions (in the order they were hit)

| # | Symptom | Root cause → fix |
|---|---------|------------------|
| 1 | `tilize_block` hangs immediately | The MATH↔PACK DST semaphore is uninitialized → call **`compute_kernel_hw_startup(icb, ocb)` before the first tilize**. |
| 2 | L1 stash dumps all-zero | PACK only reads the **standard** tile layout, but the run is in the SFPU **"math"/transposed** layout → pack reads empty cells. Insert **`step2_only` (math→standard) before `pack_untilize`** (and `transpose_wh` standard→math on restore). |
| 3 | scores/idx/bias all pack as **scores** | `pack_untilize_dest` selects the DEST tile via the **runtime `tile_dst_rt_offset` (last arg)**, NOT the 3rd positional arg (that's `block_c_index`). Use `pack_untilize_dest<1,1>(cb, 1, 0, 16, 4, 0/1/2)`. |
| 4 | idx (uint16) comes back as garbage | The bf16 scores path left the **runtime unpack format at bf16**, and `tilize_uninit` doesn't fully restore it on WH, so the idx `tilize_block` decoded raw uint16 as bf16. → give the idx tilize its **own `compute_kernel_hw_startup(run_idx_cb, cb_tilize_idx)`** (UInt16 CBs). The CB compile-time format was always correct; it was the *runtime* format. **bf16 as a raw-bit carrier is unsafe** (denormal flush — ids 0-255 = 0x00xx are subnormal). |
| 5 | restored run is **`[a,b,a,b]` 2-period duplicated** | `step2` is built for the FINALIZE layout (`store8_even_cols` at offsets `{0,4}`); applied to `produce_run`'s `{0,2}` run it mis-strides and 2-period-collapses the second half. → **`relocate_run<0,2,0,4>` before `step2`** (align to `{0,4}`). |
| 6 | place / transpose restore is all-zero | **`llk_unpack_set_srcb_dummy_valid()` placed BEFORE a `transpose_wh`** makes its TRNSPSRCB read the dummy SrcB → all-0. transpose_wh itself needs NO srcb-dummy-valid; it goes **AFTER all transposes**, right before the `step2` (in `combine_finalize`) that needs it. (Also: transpose_wh's `idst` is NOT tile-limited — it writes any DEST tile; the earlier "can't write tile 2/3" was this same srcb bug.) |
| 7 | combine: garbage + hang, then **wrong half selected** | `produce_run` + restore in the SAME acquire — `produce_run`'s SFPU/SrcB state poisons the same-acquire transpose_wh. → **merge-only acquire** (stash BOTH blocks, restore+merge with no produce_run inside). |
| 8 | combine: merge picks the **wrong 8** (dev max key == gold min key) | The **bias sort-key was 2-period corrupted** while scores+idx were fine: `step2` used `num_tiles=2`, so it transposed only scores(tile0)+idx(tile1) math→standard, **NOT bias(tile2)** → bias packed in math layout → corrupt round-trip. The 256 output path never reads bias (normalize reads only scores), so this was invisible until the merge sorted by bias. → **`step2_configure_mop<3>`** (transpose tiles 0,1,2). Harmless for the 256/finalize output (they pack only tiles 0,1). |

Bonus: in-kernel `idx += b*256` was a **no-op** (both `sfpi l_reg[]` SSA write-back and `TTI_SFPIADD ARG_IMM`
showed no effect) → sidestepped with **per-block `input_indices` tiles** (tile b = `arange + b*256`).

## 6. Key files & functions

- **`device/unified_kernels/generalized_moe_gate.hpp`** — TRISC op. `process_block_to_run<b>()` (stash one
  block to L1), the multi-block combine path (merge-only acquire), the 256 path.
- **`.../compute_kernel_api/generalized_moe_gate.h`** — ALWI wrappers: `generalized_moe_gate<...,produce_run,...>`,
  `relocate_run`, `step2_only`, `combine_init`, `combine_finalize`, `place_field_from_interm<field,dst_lo,dst_hi,
  src_lo,src_hi>`.
- **`.../tt_llk/.../ckernel_sfpu_generalized_moe_gate_topk_single_face.h`** — `merge16_to_run`, `merge16_core`,
  `copy_topk_run` (relocate), `normalize_run`, `place_field_from_interm`.
- **`.../tt_llk/.../llk_math_generalized_moe_gate_transpose_dest_single_face.h`** — step0/step1/step2 transposes.
  **`step2_init` now uses `step2_configure_mop<3>`** (pitfall 8).
- **`device/generalized_moe_gate_program_descriptor_builder.cpp`** — CBs: `run_scores/idx/bias_cb` (5/6/7,
  L1 stash, `num_blocks` tiles), `cb_tilize` (8, bf16, `2*num_blocks` tiles), `cb_tilize_idx` (9, uint16,
  `num_blocks` tiles).

## 7. Test / debug macros (`generalized_moe_gate_kernel.cpp`) & tests

Kept macros: `GMG_UNGROUPED_TOP8` (default ON), `GMG_DIAG_BLOCK` (per-block A1 validation in the combine path),
`GMG_DUMP_AFTER_SUM_TOP2/STEP0/STEP1` and `GMG_DIAG_TOPA/TOPB` (stage probes). The combine-bring-up isolation macros
(`GMG_TEST_STASH/PARK/PARK2/PRODUCE_RUN`, `GMG_DUMP_OCCUPANCY`, `GMG_COMBINE_DIAG`) and the dead helper functions
(`place_run_at`, `unpack_run_to_regions[_transpose]`) were removed after the combine landed. Tests (in
`models/demos/deepseek_v3/tests/test_generalized_moe_gate.py`): `test_generalized_moe_gate` (256/128),
`test_generalized_moe_gate_512_global` (the combine), `test_generalized_moe_gate_512_per_block`,
`test_dump_stash_run` / `test_dump_combine_run` (debug-only full-16×16 region dumps — `test_dump_combine_run` is the
tool that finally localized the bias bug by reading the whole 32×32 face).

## 8. Remaining work (none block 512)

1. **Kimi 384:** `num_blocks=2` with block1 = 256-383 (+128 padding). Likely just an op.py/test concern — set the
   padding experts' keys very low so they're never selected; the kernel combine should be unchanged.
2. **Top-n:** ✅ k = 4/6/8 done (256 + 512), see `TOPN_NOTES.md`. k>8 (top-10) and k<4 still TODO.
3. **Softmax / sqrt-softplus** normalization variants (see `SOFTMAX_NOTES.md`).
4. **>512:** needs a combine **tree** (the current path is a single 2-run merge).
5. **Perf.**

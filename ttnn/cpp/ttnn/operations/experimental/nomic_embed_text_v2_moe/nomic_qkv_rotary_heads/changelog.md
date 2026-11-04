# Changelog: nomic_qkv_rotary_heads

## Phase 0 — Core Implementation
- **Date**: 2026-11-03
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime
  `mcast2d_fused_rotary`:
  - 2D multicast matmul on 81–99 cores. x is gathered from the norm's ND shards and multicast down columns. W (bias
    as K tile 24, q pre-scaled) is multicast along rows and resident after M block 0.
  - 5 K blocks with packer L1 accumulation.
  - q/k rotary as packer-thread fp32 SFPU, through a custom-functor slot added to `matmul_block`'s Activation
    parameter (`is_custom_pack_activation`, `sfpu_activation_helpers.hpp`).
  - One bf16 rounding.
  - Writer scatters every tile to its q/k/v page and zeroes padding rows by selection.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], placement=[ln_layout,
  op_layout (≡ ln_layout)], output_buffer=[l1, dram], output_mode=[new, preallocated], seq_alignment=[tile_aligned,
  seq_padded].
- **Accuracy achieved** (HiFi2 default, measured on 4 shapes via `test_nomic_qkv_rotary_heads_precision_baseline.py`):
  - q/k: PCC=0.999994, max_abs_err=4.6e-3 (q) / 3.2e-2 (k), rel_rms_err=0.00477.
  - v: PCC=0.999994, max_abs_err=3.2e-2, rel_rms_err=0.00457.
  - bf16 ULP mean 1.1.
  - The got/true median of 0.9967 is HiFi2's srcB truncation (HiFi4: 0.9995, rel RMS 0.0018), not a kernel scale bug.
- **Golden suite at Phase 0**: 68 / 180 cells passing (per `verifier_report.json`):
  - supported_pass 68.
  - xfail_expected 56 (`model_layout`).
  - supported_skipped 56 (`op_layout` ≡ `ln_layout`).
  - supported_fail / xpass_drift / xfail_wrong_mode all 0.
- **Perf at Phase 0** (device kernel ns, 1350 MHz, ln_layout, preallocated L1): 2x512 28,960; 4x512 50,259;
  8x512 93,070. All are above the accept ceilings 26,000 / 45,000 / 85,000.
- **Issues encountered** (verifier fixes, no behaviour change):
  - Compute restated `head_tiles = 2`; it is now the host's `HEAD_TILES` CT arg.
  - The C/S table block size was recomputed in host, reader and compute. It now comes from one host constant
    (`TABLE_TILES_PER_ROW`), passed as `table_block_tiles`.
  - `_placement()` had duplicate exception arms; they are collapsed into one.
  - `l1_ledger.md` currency fixes plus a verifier audit section. The `cb_heads_out` row-window finding is folded into
    Refinement 4.
- **Tests added**: `test_nomic_qkv_rotary_heads.py` (acceptance, 35), `test_nomic_qkv_rotary_heads_perf.py`
  (profiler harness), `test_nomic_qkv_rotary_heads_precision_baseline.py` (8: 4 shapes x HiFi2/HiFi4).

## Refinement 1 — Model split layout placement (`model_layout`)
- **Date**: 2026-11-03
- **What was done**: added `"model_layout"` to `SUPPORTED["placement"]`. This is a host-only knob-turn; kernels are
  unchanged.
  - Reused: the reader's gather (any `x_part_tiles` / `x_num_parts`), the x source table format and its self-check,
    and the whole program descriptor.
  - Added: a legacy `BLOCK_SHARDED` branch in `_x_source_table`. Tile row m (= b·S_t + s_t) and K part j map to shard
    (m // bh, j), on grid (x, y) = (r, j) under COL_MAJOR or (j, r) under ROW_MAJOR. bh, kp, the orientation and the
    grid origin are all read off the tensor's `shard_spec`. Here kp = 3 and there are 8 parts, so a 5-tile K block is
    at most 3 runs a row.
  - Added: `model_layout_memory_config()`, a rebuild of the model's `sharded_norm_split` / `sharded_norm_configs`
    without importing the model package. `_placement()` tags `model_layout` by exact `MemoryConfig` equality with
    it; anything else stays `"other"` and is refused.
- **Accuracy achieved**: same kernels and precision as Phase 0 (PCC ≥ 0.99998, rel RMS ≤ 0.0050 per output, gate
  passed on every model_layout cell). Shapes: 2x512, 1x1024, 3x352, 3x400, 4x512, 8x256, 5x512, 8x384, 7x512, 8x500,
  8x512 and the rest of INPUTS.
- **Perf** (device kernel ns, 1350 MHz, preallocated L1), ln_layout vs model_layout:
  - 2x512: 29,004 vs 29,174.
  - 4x512: 50,193 vs 50,027.
  - 8x512: 93,171 vs 93,211.
  - This answers op_design.md's "op_layout decision": in-column gather reads do not move the op, because the gather
    is off the critical path. 8x512 ln_layout is unchanged against Phase 0's 93,070 (within noise).
- **Golden test progress**:
  - Targeted slice: 56/56 `model_layout` cells pass, plus the 8x512 ln_layout cell and its LOOSE case.
  - Expected full suite: 124/180 pass (68 + 56). The remaining 56 are `op_layout` ≡ `ln_layout` skips.
- **Issues encountered**: None.
- **Tests added**:
  - `test_nomic_qkv_rotary_heads_model_layout.py`:
    - The layout rebuild equals the model's builder (9 shapes).
    - model_layout correctness at 2x512, 3x400 (padded, 10 columns), 8x320 (M = 80, ragged) and 8x512.
    - A ROW_MAJOR block shard is refused.
  - `test_nomic_qkv_rotary_heads_perf.py` gains a `placement` parameter (ln_layout / model_layout).

## Refinement 1b — Model split layout placement (`model_layout`) (debug: fix gate violations)
- **Date**: 2026-11-03
- **What was done**: the harness gate failed Refinement 1 on bullet 3 (responsible cells 124/180 < 75%).
  - The 56 unpassed cells were all `placement=op_layout`. The golden helper skips them because
    `input_memory_config()` is the norm's layout. Nothing failed or hung.
  - `"op_layout"` was a dead SUPPORTED value: `_placement()` tags such a tensor `"ln_layout"`, so validate() never
    yields `"op_layout"`. It is removed from `SUPPORTED["placement"]` (now `[ln_layout, model_layout]`), which
    aligns SUPPORTED with what validate() actually produces.
  - The op_layout decision itself is unchanged (op_design.md: the norm keeps its own layout). Those cells are served
    under `ln_layout`.
  - Reused: everything. Added: nothing (registry-only change; no kernel or host-path change).
- **Accuracy achieved**: unchanged from Refinement 1 (PCC ≥ 0.99998, rel RMS ≤ 0.0050 on every golden cell).
- **Golden test progress**: full suite run.
  - test_golden.py: 115 passed, 56 skipped (op_layout → `xfail_other`, not a loud category), 0 failed.
  - test_regression.py: 9/9.
  - Responsible cells: 124/124. Zero hangs.
- **Issues encountered**: none in the kernels. The gate failure was a registry declaration problem.
- **Tests added**: none. The unit directory runs 57 passed / 6 skipped.

## Refinement 2 — q/k rotary on the last K block (perf focus 8x512)
- **Date**: 2026-11-03
- **What was done**: three levers on the q/k cores' last K block. Each was measured on device and kept.
  1. **Static SFPU addressing in `rotate_head_pair`**:
     - Before: `dst_reg[i]` with a loop-variant `i`. The SFPI compiler had to synthesize every SFPLOAD / SFPSTORE word
       at run time and push it through the instruction-buffer MMIO (`add` + `sw` per access; seen in the trisc2
       disassembly). The issuing packer TRISC, not the SFPU, bounded the rotate.
     - Now: constant per-tile offsets (`dst_reg[0 / 32 / 64 / 96]`) plus `dst_reg++`, so every SFPU instruction is
       static.
     - Same math, same fp32 DEST, one bf16 rounding.
     - 8x512: 93.17 → 85.08 µs.
  2. **Lamp L1, a per-phase q/k subblock**:
     - The q/k K loop now runs as two `matmul_block` calls per M block:
       - K blocks 0..3 at 1x4 (`LastBlockTarget::Interm`, packer L1 accumulation; the last spill is left pushed).
       - The last K block at 1x2 (one head), which reloads those partials first.
     - TileRowMajor partials are row-grouped at absolute column offsets, so the two widths compose.
     - Helper extension (inert by default): `MatmulBlockShape::reload_on_first_k_block`.
     - Knob: `WIDE_SUBBLOCK_W` (one source for the v cores' subblock and the q/k accumulation subblock; it replaces the
       descriptor's `_v_subblock_w()`). Setting `QKV_QK_ACCUM_SUBBLOCK_W=2` restores the single call.
     - 8x512: 85.08 → 84.24 µs.
  3. **C/S copy on the reload's datacopy config**:
     - The C/S `copy_tile` pair moves from after the FMAs (its own `reconfig_data_format_srca` + `copy_init`) into a
       new opt-in `post_reload` hook. The hook runs right after the helper's fp32 partials reload, with the same
       datacopy config (tables are fp32 like the partials).
     - Helper extension (inert by default): the `has_post_reload_hook` trait on PostComputeFn.
     - Knob: `CS_ON_RELOAD` (0 = the old placement).
     - 8x512: 84.24 → 83.41 µs.
  - Reused: `matmul_block` (two calls instead of one), both rotary hooks, all CBs, the host plan.
  - Added: the two helper extensions above, two knobs, and no new CB or kernel file.
- **Perf** (device kernel ns, 1350 MHz, preallocated L1), before → after:
  - 8x512: 93,173 → 83,407–83,675 (ln_layout), 83,637 (model_layout).
  - 4x512: 50,273 → 45,713–45,876 (ln), 45,504 (model).
  - 2x512: 28,868 → 28,189–28,298 (ln), 28,156 (model).
  - 8x512 is now under the 85 µs accept.
  - Ablation at 8x512 after the change:
    - rotary SFPU off: 82,703 (the SFPU costs 0.7 µs, down from 9.4).
    - C/S copy also off: 80,730 (the copy costs 2.0 µs, down from 2.7).
- **Accuracy achieved**: precision baseline (8 cases: 4 shapes x HiFi2 / HiFi4) passes. The soft gate (PCC ≥ 0.99998,
  rel RMS ≤ 0.0050) holds on q, k and v. The rotary is unchanged arithmetic on fp32 DEST.
- **Golden test progress**: targeted slice 30/30 pass. It covers:
  - Every non-op_layout 8x512 and 8x500 (`seq_padded`) cell, in l1 / dram x new / preallocated x ln / model_layout.
  - The 2x512 dram ln_layout cells.
  - test_regression.py.
  - Full-suite expectation unchanged: 124 / 124 responsible cells.
- **Issues encountered**: none. Remaining headroom is a finding, not a follow-up:
  - The rest of the rotary region is the fp32 C/S unpack per head (2.0 µs).
  - Keeping C/S resident in DEST across a row's 4 heads would need a release that does not clear the whole half.
- **Tests added**: none. The existing precision baseline, acceptance, model_layout and perf files cover the changed
  paths. 49/49 acceptance + model_layout pass.

## Refinement 3 — M-block-0 weight stream and multicast fill (perf focus 8x512; binds the 2x512 / 4x512 guards)
- **Date**: 2026-11-03
- **What was done**: the in1 (weight) stream of M block 0. Every lever named by the refinement was built, measured on
  device and left in the tree as a knob (one source of truth in `nomic_qkv_rotary_heads.py`; `QKV_<KNOB>` env overrides
  for measurement).
  1. **16 KB lane pages** (catalog `read_coalescing`), kept:
     - `prepare_qkv_weights` lays out one page per (K row, `WEIGHT_PAGE_TILES` N tiles). The default is a whole lane:
       one 16 KB read per K row instead of four 4 KB head pages.
     - Layout tag `n_pages_8_v1`; validate() checks the derived tag and shape. `WEIGHT_PAGE_TILES = HEAD_TILES` gives
       back the old layout.
     - −0.5 to −1.1 µs.
  2. **Read-ahead with transaction ids** (lamp L4, catalog `trid_double_issue`):
     - `WEIGHT_READ_AHEAD_CHUNKS` (1..14). Each multicast chunk's reads retire on their own trid
       (`set_read_trid` / `async_read_barrier_with_trid`).
     - Default: one K block, the old behaviour. Deeper is slower: 2 blocks in flight cost +2–3 µs, because the stream
       is DRAM-bound grid-wide and a second block delays block 0.
  3. **In1 sender placement** (lamp L5, catalog `mcast_topology` / `noc_placement`):
     - `W_SENDER` = right (default) / diagonal (`McastFixedSenderConfig` Staggered) / rotating
       (`McastRotatingSenderConfig`, per-row round offset `W_SENDER_STAGGER`).
     - Rotating: flat at one block of read-ahead, up to +7 µs deeper. Diagonal: +0.3–1.3 µs.
     - The resident-ring contract and equal CB addresses are unchanged in all three.
  4. **Chunked multicast of a K block** (`W_MCAST_CHUNK_K_ROWS`):
     - The sender multicasts each K row as soon as its reads land; the block's last chunk's `send()` publishes the
       block.
     - Helper extension, inert unless called: `SenderPipeImpl::send_payload()` (`kernel_lib/mcast/kernel/pipes`), an
       unsignalled payload multicast with no readiness wait. It is legal here because the ring is still empty in
       M block 0.
     - Measured at 1 K row (2x512 zones): block 0 lands ~0.5 µs earlier and compute ends ~1.1 µs earlier. With output
       writes stubbed it wins ~0.6 µs at every shape; with writes on it ties, because the output tail absorbs it.
       Parked at `K_BLOCK_TILES` (one multicast per block).
  - Reused: the writer's M-block-0 loop, the resident ring, the w_mcast channel, `TensorAccessor` reads.
  - Added: the knobs above, one mcast-helper method, a measurement-only `ABLATE_WMCAST` stub. No new CB or kernel file.
- **Perf** (device kernel ns, 3-run medians, 1350 MHz, ln_layout, preallocated L1), before → after:
  - 2x512: 28,024 → 26,935.
  - 4x512: 45,775 → 45,227.
  - 8x512: 83,487 → 82,702.
  - model_layout: 27,007 / 44,792 / 82,531 (was 28,156 / 45,504 / 83,637).
  - Noise is about ±0.5 µs.
- **Findings** (2x512 device zones):
  - Weight block 0 reaches compute at ~5.5 µs, against ~1.7 µs for x. That is a ~3.1 µs DRAM read (720 KB grid-wide at
    ~240 GB/s; placement-independent) plus a ~1.5 µs row multicast.
  - Later blocks arrive every ~3.0 µs, faster than compute's ~3.3–3.7 µs per K block. Only the fill is exposed.
  - At 2x512 the op's end is the output scatter. The v rows (bottom of the grid) finish compute first and write last,
    3–4.5 µs after compute, against ~1 µs on q/k rows.
  - Next levers (outside this refinement): the drain tail (lamps L6/L7), after which `W_MCAST_CHUNK_K_ROWS = 1` pays;
    and a smaller first K block (compute).
- **Accuracy achieved**: unchanged arithmetic. The precision baseline (PCC ≥ 0.99998, rel RMS ≤ 0.0050 on q/k/v)
  passes. Only the weight page layout and the stream order changed.
- **Golden test progress**: targeted slice 52 passed, 20 skipped (op_layout), 0 failed. It covers 8x512, 2x512,
  4x512, 8x500 (`seq_padded`) and 3x400 (M = 39, 10 columns) across l1 / dram, new / preallocated and ln / model_layout,
  plus test_regression.py.
- **Issues encountered**: none in correctness. The parked knob settings pass the 35-case acceptance file: diagonal and
  rotating senders with 1-row chunks, and a read-ahead of 14 chunks.
- **Tests added**: none. The unit directory runs 57 passed / 6 skipped (the profiler-only perf file).

## Refinement 4 — K-loop efficiency and M/K block co-tune (perf focus 8x512, toward stretch)
- **Date**: 2026-11-04
- **What was done**: re-measured the binding region first (fidelity sweep, compute device zones per phase, block
  sweeps), then built and measured the block-surface levers. No CB was added, resized or removed; no SUPPORTED change.
  1. **`TABLE_LOAD_K_BLOCK`** (new knob, default `NUM_K_BLOCKS - 2`). The q/k reader asked for an M block's C / S
     right after in0 K block 0. `cb_rotary_table` is single-slot, so that request waited for the previous block's
     finish, and then the table read sat in front of K block 1's gather + multicast on every reader of the column.
     The tables are needed only by the last K block, so the request now rides K block n-2. Measured
     (one run each, same build): 8x512 83.07 → 82.41, 2x512 27.37 → 26.84, 4x512 45.59 → 45.28 µs. Kept.
  2. **`OUT_DRAIN_TILES`** (new knob, default `HEAD_TILES`; perf lamp L7, the last-block tail).
     - The writer drains `cb_heads_out` one head (2 pages) at a time instead of a whole row.
     - The q/k finish call pushes per head: `OutputCBLayout::SubblockMajor`, which gives the same tile layout because
       h = 1, and is a single K block, so there is no whole-block reserve.
     - The median compute-to-writer tail halves (0.98 → 0.52 µs). The wall ties (8x512 82.6–82.8 vs 82.6–83.3 at
       `LANE_N_TILES`), because the q/k compute, not the drain, ends the op.
     - Paired with `W_MCAST_CHUNK_K_ROWS = 1` (Refinement 3's parked lever, waiting on this tail) it is still a tie
       (82.0–83.1 µs). Both stay as knobs.
  3. **Block sweep (measurement only, 8x512, all values measured on device)**:
     - `K_BLOCK_TILES` = 1: 162 µs (25 multicast rounds per M block). = 5: 83.4 µs. = 25: hangs in the old in1
       sender (one 400 KB block); superseded by the whole-K prototype below.
     - `BLOCK_M_TILES_MAX` = 1: 100.2, = 2: 87.6, = 3: 83.4. With the L1 budget lifted, = 4: 81.7 µs (−0.9). It needs
       704 KiB > 640.
     - `X_BLOCK_DEPTH` = 3 (L1 budget lifted): 83.0 µs, flat.
  4. **Whole-K row prototype (built, measured, not kept)**:
     - After M block 0, one x multicast per row (25 tiles), weights resident (`InputPolicy::NoWaitNoPop`), and one
       DEST pass per subblock over all of K. No partials spill or reload.
     - v rows: 5.1 µs a row (91% of HiFi2 math). q/k rows: 6.2 µs a row. The per-head fp32 C/S copy, without the
       reload's datacopy config, needs its own srcA reconfig, copy_init and matmul re-init: ~0.27 µs a head.
     - That is no better than the current q/k 6.2 µs a row, and q/k sets the op's end. Not worth the restructure
       (new x ring layout, M-block-0 special case) for at most ~1 µs.
  - Reused: the reader's per-K-block loop, `matmul_block` (one template argument changes on the finish call), and
    the writer's scatter (now per drain step).
  - Added: two knobs and their asserts. No new CB, kernel file or helper change.
- **Findings** (8x512, 1350 MHz):
  - **The op is q/k-compute-bound.** Op time ≈ math + 27 µs at every fidelity ≥ HiFi2 (LoFi 66.5, HiFi2 83.0,
    HiFi3 110.7, HiFi4 139.5 µs; ~28 µs a fidelity phase = 2400 tile products a core × ~15.6 cycles).
  - **Device zones per M block (3 rows)**:
    - q/k: the 1x4 accumulation over K blocks 0..3 takes 13.1 µs for 11.1 µs of math (85%).
    - q/k: the 1x2 finish (reload, C/S, 5 K, rotary) takes 5.5 µs for 2.8 µs of math. That is ~0.23 µs fixed a head:
      fp32 reload + C/S datacopy and matmul re-init.
    - v: 17.5–18.8 µs for 14.2 µs of math.
    - M block 0 waits ~5 µs on the weight fill.
  - The matmul LLK unpacks srcA once per tile product whatever the subblock shape (`reuse_a`, rt-streaming), so
    1x2 vs 1x4 and K block size barely move the per-product cost. Refinement 2 already saw this: halving the calls
    saved 0.8 µs.
  - Remaining headroom is the q/k finish's per-head fixed cost (48 heads × ~0.23 µs ≈ 11 µs). Moving it needs
    C/S resident in DEST across a row's heads, which means a release that does not ZEROACC the whole half (LLK-level),
    or a cheaper fp32 C/S path into DEST. `BLOCK_M_TILES_MAX = 4` (−0.9 µs) needs ~64 KiB of reclaim: a 1-row
    `cb_heads_out` plus a per-row C/S ring interleaved into the reader's x schedule.
- **Perf** (device kernel ns, 1350 MHz, preallocated L1, defaults), before (Refinement 3 medians / this session's
  first run) → after:
  - 8x512 ln: 82,702 / 83,439 → 82,650–83,276 (median ~82.75 µs).
  - 4x512 ln: 45,227 / 44,743 → 44,756–45,408.
  - 2x512 ln: 26,935 / 27,261 → 26,875–27,324.
  - model_layout: 26,758 / 45,100 / 82,899.
  - Within ±0.5 µs noise of Refinement 3 at every guard shape. No regression; the stretch (76 µs) is not reached.
- **Accuracy achieved**: unchanged arithmetic. Only the push / drain granularity and the table request point moved.
  The precision baseline passes (PCC ≥ 0.99998, rel RMS ≤ 0.0050 on q/k/v; 4 shapes × HiFi2 / HiFi4).
- **Golden test progress**: targeted slice 59/59 pass, 0 fail. It covers 8x512, 8x500 (`seq_padded`), 7x512 / 5x512
  (M = 112 / 80, ragged), 3x400 (M = 39), 2x512 across l1 / dram, new / preallocated and ln / model_layout, plus
  test_regression.py. Full-suite expectation unchanged: 124/124 responsible.
- **Issues encountered**: two hangs, both in measurement-only configurations that were never committed:
  - The K_BLOCK = 25 sender path.
  - The whole-K prototype before the C/S hook restored the matmul unpack config. Without a reload, nothing re-inits
    after the copy.
- **Tests added**: none. The unit directory runs 57 passed / 6 skipped (the profiler-only perf file).

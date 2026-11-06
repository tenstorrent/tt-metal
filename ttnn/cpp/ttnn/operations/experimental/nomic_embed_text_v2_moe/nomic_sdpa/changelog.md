# Changelog: nomic_sdpa

## Phase 0 — Core Implementation
- **Date**: 2026-11-05
- **What was done**: initial implementation through the incremental pipeline (planner → implementer → verifier).
  - Regime: row-split, per-core K/V read; up to 110 cores.
  - Algorithm: exact two-pass softmax over the resident key range, with the row sum taken through a ones column of
    V.
  - One `generic_op` per call; cached descriptor, with buffer addresses rebound per call.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], buffer=[l1, dram], mask_mode=[none], alignment=[tile_aligned]
- **Accuracy achieved** (default HiFi2, fp32 DEST, approx exp; `test_nomic_sdpa_precision_baseline.py`; 4 shapes ×
  3 profiles):
  - real: PCC 0.99984–0.99988, rel-RMS 0.0220–0.0255, max_abs ≤ 0.064;
  - flat: PCC ≥ 0.99992, rel-RMS ≤ 0.0142;
  - peaky: PCC ≥ 0.99985, rel-RMS ≤ 0.0192, max_abs ≤ 0.178;
  - got/true median 0.981 on real. The approximate exp plus HiFi2 make outputs uniformly ~1.9% small (the
    incumbent's precision class); with the exact exp it is 0.994.
- **Golden suite at Phase 0**: 45 / 90 passing: 45 supported_pass, 40 xfail_expected, 5 regression self-xfail; 0
  supported_fail / xpass_drift / xfail_wrong_mode (`verifier_results/verifier_report.json`).
- **Perf at Phase 0** (device µs, L1): 8x512 ≈ 130, 4x512 ≈ 79, 2x512 ≈ 54. The incumbent runs 72.1 / 38.2 / 24.5;
  the 8x512 target is 58.
- **Issues encountered**:
  - Implementer: the design's bf16-DEST default missed the bar at S = 512, so the default became fp32 DEST.
  - Verifier fix: `validate()` now refuses bf16 DEST + approx exp at S > 384, which was measured to miss the bar at
    S = 512.
  - Verifier fix (DRY): the common-RT table offset is now a named CT arg (`table_arg`) from the host constants,
    replacing the kernel literals 3 / 1.
  - Verifier: the perf harness's refused config was replaced by bf16 DEST + exact exp, and a full-sync config was
    added.
  - Verifier: the L1 ledger audit found the multicast deferral reason falsified, so it is filed as Refinement 2.
- **Tests added**: `test_nomic_sdpa.py` (acceptance, 39), `test_nomic_sdpa_perf.py` (profiling harness),
  `test_nomic_sdpa_precision_baseline.py` (12 baseline cells, the bf16-DEST + approx-exp refusal, and bf16 DEST +
  exact exp at 8x512 × 3 profiles).

## Refinement 2 — Read K/V once per pair and multicast it to the cores sharing that pair
- **Date**: 2026-11-05
- **What was done**: the deferred regime **row-split + pair multicast**, built on the existing row split.
  - Host (`kv_multicast_topology`): a pair's group is every core holding it at a segment index `< kv_depth`, which is
    a consecutive core run. Its first core reads K/V once and multicasts them to the run. The other members receive.
    Cores holding the pair at a later segment read their own copy. Roles and up to 3 rectangles (virtual NoC
    coordinates) per sending segment go to the reader as per-core RT args; there are 2 Flag semaphores per ring slot.
  - Reader: ring slots start `p0 % kv_depth` data-less phantom slots in (compute pops them), so pair p lands at ring
    slot `p % kv_depth` on every core. Every member's slot is fresh, so the multicast needs no handshake.
  - Reader: senders pre-send. K goes at t = 0; V goes after the core's own first Q. K moves as one block; V moves as
    N row payloads of HD tiles, so the ones column is never sent.
  - Reader: blocks now read Q before waiting on a received K.
  - Transport is kernel_lib `SenderPipe` / `ReceiverPipe` (`mcast_pipe.hpp`). The host `ttnn.Mcast` /
    `McastArgs` channel cannot express irregular, overlapping, table-sized groups.
  - `kv_depth` cap is per placement (`KV_DEPTH_L1 = 1`, `KV_DEPTH_DRAM = 2`). The entry point falls back to 1 when the
    deeper ring does not fit below live L1.
  - Path gate `KV_MCAST_MIN_SEQ_TILES = 8`. At S_t = 4 (1x128) the store-and-forward hop cost 4.7 → 5.3 us.
  - `KV_MCAST` is kept as the A/B knob.
  - Reused: the reader / compute / descriptor, the CB inventory, the block schedule and the ones-column writer.
  - Added: the topology function, the per-core reader RT args, the semaphores, the phantom pops in compute, the
    pre-send phase, and the `ABLATE_ZONES` profiling scaffold (compute input-wait zones).
- **Perf** (device us, median of 3, Phase 0 → R2):
  - L1: 2x512 59.1 → 30.5, 4x512 76.0 → 53.8, 8x512 unchanged (~120-127 vs ~122, 7-sample medians overlap), 1x512
    49.5 → 17.8, 8x384 82.0 → 74.3;
  - DRAM: 2x512 60.7 → 34.4, 4x512 86.2 → 63.6, 8x512 134.7 → 117.8;
  - no shape slower: 16x256, 32x128 and 1x128 are within noise.
- **What binds 8x512 now** (ablation + device zones):
  - DM floor 50 us (was 70-80), compute floor 74 us.
  - Compute's first Q + K wait ends at 38.5 us median / 54 us max: the t = 0 first-segment burst on NOC0. The
    multicast removes only second-segment bytes.
  - Next levers (not taken here): key-chunked K delivery (R4 L1), splitting the t = 0 reads onto NOC1, and the
    compute floor (R3).
- **Accuracy achieved**: unchanged arithmetic, so outputs are the same as Phase 0. The real-profile golden bar holds
  at S = 512 on L1 and DRAM (acceptance `test_nomic_sdpa_real_profile`, rel-RMS < 0.0265, PCC ≥ 0.9997).
- **Golden test progress**: slice `loose or (none and tile_aligned)` 31/31 passing (the 3 loose perf cases plus
  every supported `mask_mode = none` cell); the full suite is the harness's run.
- **Issues encountered**:
  - First build regressed 8x512 126 → 175 us. Its sender was the first core whose first segment is the pair, which
    sits east of its receiver, so the multicast wrapped the NOC0 torus row. Fixed by sending from the run's first
    core, with a t = 0 pre-send.
  - Intermittent 16x256 corruption (max_abs 1.47). `write_ones_column`'s local L1 copies were retired only by the
    following V read barrier, which RECEIVE / SEND segments no longer issue. Fixed with a read barrier inside it.
  - The verifier's headroom estimate (1055 KiB free at the perf case) was wrong: 914 KiB were measured, and the 2-deep
    ring (994 KiB) did not fit. Hence the L1 cap of 1 and the fit fallback.
- **Tests added**:
  - `test_nomic_sdpa.py::test_nomic_sdpa_kv_depth_falls_back_under_l1_pressure`;
  - `test_nomic_sdpa_debug.py`: per-knob bad-row map;
  - perf harness: `NOMIC_SDPA_KNOBS` knob overrides, `NOMIC_SDPA_REPS` repeats, and more shapes.

## Refinement 3 — Overlap the SFPU exp with the FPU matmuls (row-group interleave)
- **Date**: 2026-11-05
- **What was done**: perf lamp L2, a scheme change inside the per-core block (the reader, the writer and the whole-block
  Q / out CBs are unchanged).
  - **Row-group schedule** (`EXP_OVERLAP`, default on under the approximate exp, path-gated to `q_block_rows ≥ 3`):
    the block's rows run in groups of `ROW_GROUP_ROWS` (default 1). The exp moves to the **pack thread**
    (`exp_packthread_tile`, the incumbent's unclamped fast exp + pack ReLU, scale folded into its init). The math
    thread only issues each probs window's `s - m` and moves on. Each probs DEST window is followed by one long FPU
    window: P·V of the previous group, then Q·K^T of the next group. This is a 3-stage software pipeline: t+1's
    scores, t's P and t-1's [O | l] are in flight together.
  - **CBs**: scores and P are now separate 2-group rings (`2*G*N` each, new `cb_probs`). P can no longer be computed
    in place, because the rings are FIFOs read from their front. `cb_row_stats` and `cb_pv_out` are 1-group rings.
    The per-core footprint drops from 834 to 410 KiB at 8x512 (the `Q*N` term is gone).
  - **`probs_window`** is a custom block operation (raw LLK: `sub_tiles_bcast_cols` on the FPU, then
    `exp_packthread_tile` between the pack thread's `tile_regs_wait` and its pack). Kernel_lib gap: `eltwise_chain` has
    no pack-thread SFPU element and `PackTile` no pre-pack hook. The Q·K^T and P·V windows are `matmul_block` calls
    with an `In1BaseOffset` column functor; row max and normalize stay on the helpers.
  - **`pack_sfpu_fence`** (two empty DEST windows) holds normalize's MATH-thread recip until the pack thread has
    released every window, so its `recip_tile_init` cannot reprogram the shared SFPU mid-exp.
  - Knobs: `EXP_OVERLAP` (A/B), `ROW_GROUP_ROWS`, `PV_WINDOW_COLS` (0 = the whole [O | l] row; 1 = one column per
    window) and `EXP_OVERLAP_MIN_ROWS = 3`.
  - Reused: the reader, the writer, the CB indices, the helper calls (`matmul_block`, `reduce<MAX>`, the normalize
    chains, now one shared `normalize` lambda for both schedules) and the ring-pad discipline.
  - Added: `cb_probs`, `probs_window`, `pack_sfpu_fence`, the window lambdas, the Plan fields `exp_overlap`,
    `row_group_rows`, `pv_window_cols` and `exp_overlap_for(compute)`, and an optional `compute_kernel_config` on
    `circular_buffer_bytes`.
- **Perf** (device us, Blackhole p300c, default config, A/B on the `EXP_OVERLAP` knob):
  - **8x512 L1 (perf focus): 128.2 → 111.7, median of 7** (−13%). Compute floor (reads + writes stubbed) 74.0 → 72.3;
    the exp went from 13.6 us visible to ~6.7 us. Most of the wall gain is the output now streaming to the writer
    row by row instead of at block end.
  - L1, median of 3: 8x384 79.3 → 69.9, 16x256 66.0 → 56.7, 32x128 38.2 → 34.1; 4x512 54.6 → 55.1, 2x512 31.8 → 30.7
    and 1x128 4.9 → 4.8 are noise. 1x512 (Q = 2) was 17.9 → 19.4 with the schedule, so it is path-gated off.
  - DRAM, median of 3: 8x512 118.2 → 114.4, 16x256 84.5 → 77.9, 8x384 77.6 → 76.3; 4x512, 2x512, 1x512 and 32x128
    are within noise.
  - Configs, 8x512 L1: fp32 full-sync 134.2 → 119.9, HiFi3 129.2 → 125.7. The exact exp (bf16 DEST) keeps the
    block-sequential schedule.
- **What binds 8x512 now** (ablation, device us): wall 111.7; compute floor 72.3; DM floor (compute stubbed) 68–83;
  all payloads stubbed 4.6. The wall is still well above both floors: the t = 0 first-segment burst (Refinement 2)
  still delays the start of compute. Inside the compute floor, the window structure costs ~5 us with the exp stubbed
  (65.6 vs 60.5 for the block-sequential schedule). That is the per-window `matmul_block_init` /
  `sub_bcast_cols_init` re-inits and the per-window pack-ReLU toggles (~1 us). One-column P·V windows hide more of the
  exp (residual 1.7 us) but pay ~7 us in 3x matmul calls.
- **Accuracy achieved**: unchanged arithmetic class (approx exp + pack ReLU, bf16 P, row sum from the P·V ones
  column). `test_nomic_sdpa_precision_baseline.py`, S = 512 (8x512 / 2x512):
  - real PCC 0.99984, rel-RMS 0.0255 / 0.0251 (bar 0.0265);
  - flat PCC 0.99994, rel-RMS 0.0126 / 0.0127 (bar 0.0175);
  - peaky PCC 0.99985, rel-RMS 0.0191 / 0.0192 (bar 0.0375).
  These match Phase 0 to the printed digits.
- **Golden test progress**: slice `loose or (none and tile_aligned)` 31/31 passing (the 3 loose perf cases plus every
  supported `mask_mode = none` cell). The full suite is the harness's run.
- **Issues encountered**:
  - First pairing (probs windows with Q·K^T windows only) was correct but flat: compute floor 74.0 → 73.5. The pack
    thread bound it, since each pair packs 8 tiles plus the exp, and window re-inits added 5.8 us.
  - Pairing with the P·V windows instead hung at 2x512, where a row has 4 probs windows against 3 P·V columns. The
    MATH-thread `recip_tile_init` in normalize is gated by no operand, so it ran while the pack thread's last exp was
    still on the SFPU. Fixed with `pack_sfpu_fence`.
- **Tests added**: `test_nomic_sdpa.py::test_nomic_sdpa_row_group_knobs` (G = 4 ragged groups, one-column P·V windows,
  both together, and the `EXP_OVERLAP = False` A/B path at 8x384). Perf harness: `NOMIC_SDPA_ABLATE=EXP` (exp
  stubbed) and `RELU` (pack-ReLU toggles stubbed).

## Refinement 4 — Block, buffer-depth and cheap-lever co-tune on the restructured kernel
- **Date**: 2026-11-05
- **What was done**: measured each lever on the R2 + R3 kernel, kept the winners as defaults and parked the rest as
  live knobs. No SUPPORTED change.
  - **Diagnosis** (device zones, 8x512 L1, 176 KB reserved): the wall was the startup burst and compute in series.
    Compute's first K landed at 31.5 us median and its first V at 46 us (a new `wait-v` zone in the row-group path).
    Every reader first moved its whole Q block (~56 KiB a core), which compute consumes over ~70 us. Ablation floors:
    compute 72.2, DM ~77, wall 110.7. `Q_DEPTH = 1` also kept the reader from reaching the next block's K / V until
    compute had finished the current block.
  - **Q lead rows** (`Q_LEAD_GROUPS = 2`, reader): a block's first 2 row groups of Q go ahead of the segment's K / V,
    and the rest follow V. All pushes are cumulative inside one reserve, still the nominal `Q*HD`. Gated to the
    row-group schedule; the block-sequential one needs its whole Q block before Q·K^T (2x512 exact exp was 53.4 → 57.2
    without the gate). First K 31.5 → 18 us, first V 46 → 35 us.
  - **`Q_DEPTH` per placement** (`Q_DEPTH_L1 = 2`, `Q_DEPTH_DRAM = 1`), with an L1-fit fallback chain
    (kv_depth → 1, then q_depth → 1). `q_depth` is now in the call cache key.
  - **L7 pair-aligned `core_row_start`** (`PAIR_ALIGNED_ROWS`, `_core_row_starts`): each pair is split over its share
    of the grid, one segment a core. It is taken only when the largest per-core range does not grow (2x512, 1x512).
    `q_block_rows` now derives from the table.
  - **L8 column-0 recip** (`RECIP_COL_ONLY`, a `RecipSum` chain element calling `recip_tile(.., VectorMode::C)`).
  - **L9 bank stagger** (`KV_BANK_STAGGER`, reader key-row rotation `core % N`): parked at False.
  - **L6 `kv_depth = 2` at L1**: parked at `KV_DEPTH_L1 = 1`.
  - **Not built**: L1 K chunking, because V, not K, gates compute's start. L5 normalize in the P·V `PostComputeFn`,
    because a math-thread SFPU recip / mul there would contend with R3's pack-thread exp, which runs in exactly those
    windows.
  - Reused: the reader / compute / writer, every CB and helper call, the multicast topology (unchanged),
    `Plan` / `circular_buffer_specs`.
  - Added: the `q_lead_rows` / `kv_bank_stagger` / `recip_col_only` CT args, `RecipSum`, `_core_row_starts`, the
    `wait-v` zone. Perf harness: `NOMIC_SDPA_RESERVE_KB` (the perf case's L1 reservation), the `NOMIC_SDPA_SHAPES`,
    `NOMIC_SDPA_CONFIGS` and `NOMIC_SDPA_BUFFERS` filters (`--profile` cannot forward a multi-word `-k`), and no
    finiteness assert under ablation.
- **Perf** (device us, Blackhole p300c, A/B with all R4 knobs off → on in one session):
  - L1, 176 KB reserved, median of 9: **8x512 115.1 → 99.8**, 4x512 53.0 → 49.3, 2x512 30.1 → 27.8.
  - Guard sweep, 8 shapes × 4 configs × 2 placements, median of 3-5:
    - DRAM: 2x512 34.4 → 32.3, 4x512 63.3 → 58.7, 8x512 114.9 → 111.2;
    - L1: 32x128 34.1 → 33.0, 1x512 19.7 → 18.5;
    - full-sync: 2x512 32.5 → 30.2, 8x512 127.1 → 110.9;
    - exact exp: 2x512 53.5 → 51.6.
    No case is slower beyond the ~2-3% noise band. 1x512 DRAM is 20.7 → 21.0 (9-sample medians).
  - Per-lever measurements are in the knob comments in `nomic_sdpa.py`.
- **What binds 8x512 now**: compute runs at full rate only from the first-segment K + V landing (~35 us; ~12.6 MiB
  chip-wide on NOC0, the minimum for this regime), then the ~72 us compute floor runs. Next levers (outside this
  heading): split the t = 0 reads across NOC0 / NOC1; a deeper row-group prologue, so more Q·K^T + exp runs before the
  first P·V; the per-window matmul re-init.
- **Accuracy achieved**: arithmetic is unchanged except the recip's untouched columns 1-31, which nothing reads.
  `test_nomic_sdpa_precision_baseline.py`:
  - S = 512 (8x512 / 2x512): real PCC 0.99984, rel-RMS 0.0255 / 0.0251 (bar 0.0265); flat 0.0126 / 0.0127 (bar
    0.0175); peaky 0.0191 / 0.0192 (bar 0.0375);
  - bf16 DEST + exact exp at 8x512: real 0.0211, flat 0.0165, peaky 0.0321.
- **Golden test progress**:
  - slice `loose or (none and tile_aligned)`: 31/31 passing;
  - `test_regression.py`: 14 passed / 5 xfailed (as before);
  - unit directory: 133/133.
  The full suite is the harness's run.
- **Issues encountered**:
  - `kv_depth = 2` measured slower again. The reason is now located: a store-and-forward hop on the startup's critical
    path.
  - The lead order regressed the block-sequential schedule until it was gated.
  - `Q_DEPTH = 2` regressed 16x256 DRAM, hence the per-placement cap.
- **Tests added**: perf harness env knobs only (above); the existing acceptance / precision / debug tests cover the
  changed paths.

## Perf 1 — perf tournament round 1 (measured breakdown → 6-idea portfolio → 4 graduations)
- **Date**: 2026-11-05. Blackhole p300c, AICLK 1350 MHz (profiler CHIP_FREQ). No SUPPORTED change.
- **Focus** (`feature_spec.py` LOOSE_CASES, the `attention:` perf case): 8x512, bf16, L1, mask none, the op's default
  compute config (HiFi2, fp32 DEST, approx exp, half-sync — the fixed precision contract), 176 KB of L1 reserved.
  Guards: 4x512 / 2x512 L1 (the other loose cases) plus the guard set below.

### Measured breakdown (before; permanent zones + cumulative ablation, 8x512 L1)
- Whole op: 101.2 (one run), median of 5 109.2 us (spread 98-120: the DM half was congestion-noisy).
- Peel: full 101.2 → reads stubbed 75.7 → + writes stubbed **75.4 (compute floor)** → + exp stubbed 68.1 → + ReLU
  toggles 68.0. Compute stubbed: **63.6-76.5 (DM floor)**. Everything stubbed: 5.0. The two halves were balanced and
  only partly overlapped: neither floor alone explained the wall.
- Zones (per core p50, us; kernel spans balanced, max/p50 1.05; markers ≤ 100 / RISC, 100% span coverage):
  - unpack `compute_wait_qk` 15.3: compute starved ~15 us at t = 0 for its first Q + K;
  - math `compute_exp_pv_qk` 45.3, `compute_row_max` **17.6**, `compute_normalize` 8.8;
  - pack `compute_exp_pv_qk` 68.8 (the pack thread carried every exp and every pack);
  - reader `reader_kv_issue` 4.8 for 128 pages (cheap), `reader_kv_barrier` **28-37** (waiting on the NoC).
- DM ceiling (`perf_tools/link_ceiling.py`, route-load per perf-ceiling-dm Step 4b; the NPE CLI is not built in this
  tree): every read on NOC0 puts 1456 KiB on the busiest link → 17.3 us; V on NOC1 → 10.3 us. The DM floor sat ~4x
  above it: NOC0-congestion bound, not issue- or bandwidth-of-a-core bound.
- Ranked: (1) DM NOC0 congestion (~65-76 us floor vs 17 us ceiling); (2) row max on the math thread (17.6 us);
  (3) window / handoff overhead in the exp-P V-Q K^T loop; (4) normalize (~9-12 us); (5) the pack-thread exp (~7 us
  visible); (6) the t = 0 startup.

### Portfolio and verdicts (6 perf-part-optimizer subagents in parallel; experiment dirs under `perf_experiments/`)
| idea | verdict | measured (us) | domain / notes |
|---|---|---|---|
| `dual_noc_reads`: Q + V reads to the writer RISC (BRISC, NOC1) | **WIN, graduated** | 8x512 full ~108.5 → 92.6, DM floor ~69 → 44 (on the then-current op) | everywhere except a measured regression: 1-deep cb_query (DRAM) with S_t ≤ 8 (DRAM 32x128 57.4 → 61.8, 16x256 76.9 → 81.4, 1x128 6.6 → 7.6). Variants with one RISC on both NoCs BLOCKED: `noc.h:767` static_asserts `DM_DYNAMIC_NOC` for any kernel including it. Cooperative K split across RISCs: within noise, +0.7 us at 1x128 — dropped |
| `kv_neighbor_pull`: 2nd-segment K/V from the neighbour core's resident slot (reuse-shared standing idea) | push form WIN-small, pull NULL; **not graduated** | push: 8x512 108.6 → 106.8 (p = 0.046), DM floor 67.7 → 50.6; pull flat at L1, DRAM 117.7 → 121.4 | needs a slack gate (16x256 / 32x128 L1 regress 2x without it). Measured on the pre-split dataflow and built on the writer RISC that split reads now uses for Q / V: the combination is unmeasured. Round-2 lead (its gain came from short-first-segment cores' startup) |
| `rowmax_block_llk`: `reduce_block_max_row` | **WIN, graduated (inside `window_reinit`'s kernel)** | floor 75.2 → 62.6; math row max 17.57 → 1.95 per core; bit-exact (756 runs) | everywhere. `llk_s2` (last probs window after the FPU windows, row max before normalize, new `cb_row_max`): floor −2 us more on the old kernel; not graduated (its reorder targeted the removed normalize; re-measure in round 2) |
| `normalize_in_dest`: O / l in DEST on the pack thread | **WIN, graduated** | floor 75.2 → 71.6 (old kernel), 56.8 → 54.6 (in sequence); real rel-RMS 0.0255 → 0.0216 | everywhere, both schedules. Inexpressible: one-column P V windows (`PV_WINDOW_COLS = 1`) — knob deleted. Options measured: library `div_binary_tile` on pack 97.5 (regression), math-thread + fence 76.2 (regression), helper + custom packer Activation 73.1, math-thread semaphore form 70.8 (worse at N = 4 / 8), Newton steps nr0 / nr1 −0.4 us at +0.0001 rel-RMS (nr2 kept) |
| `window_reinit`: per-window init / handoff overhead | **WIN, graduated** | floor 75.3 → 56.7, 8x512 full 108.1 → 89.5 (raw windows + block row max); bit-exact | everywhere. Found: `ckl::matmul_block` issues `pack_reconfig_data_format` per output subblock under fp32 DEST even with reconfig NONE (4.8 us). The asked-for inits were small (≤ 1.8 us); 2:1 / 4:1 window reorders slower; `ROW_GROUP_ROWS = 2` floor −2.6 but whole-op regression at 32x128 / 16x256 L1 (+128 KB L1) — not taken |
| `packthread_exp_impl`: faster pack-thread exp | batched: WIN, **graduated**; magic: **rejected** | micro 83.2 → 76.0 ns / tile batched, 51.9 magic; floor 75.2 → 73.9 / 69.7 (old kernel); in sequence 54.6 → 53.3 | batched bit-exact everywhere. Magic (1 SFPLOADMACRO / vector) is INCORRECT for logit gaps scale·(max − s) > 264 (17-bit window aliasing; PCC 0.89-0.98 on spread inputs) — inside the op's input domain, no cheap guard: recorded as an option, never graduated. Clamped forms 119.5 ns / tile (slower). Pack thread at the floor: ~24 us busy (exp 13.4, packs 10.3) of a 64 us zone, the rest waiting on the math thread |

### What graduated (in order, each golden-green and measured end-to-end; 8x512 L1 median of 5)
| step | commit | 8x512 | 4x512 | 2x512 | compute floor |
|---|---|---|---|---|---|
| before (zones only) | `69dd549c913` | 109.2 | 52.3 | 28.7 | 75.4 |
| A raw mm windows + `reduce_block_max_row` | `932923fd060` | 90.5 | 44.9 | 26.1 | 56.8 |
| B split reads (Q + V on BRISC / NOC1) | `4c2213afc90` | 74.9 | 43.1 | 24.0 | DM floor 47.4 |
| C normalize in DEST (pack thread) | `b0831247873` | 72.0 | 42.5 | 23.6 | 54.6 |
| D batched pack-thread exp | `c3aa8513377` | **70.3** | — | — | 53.3 |
- Every graduation is the op's one path, and the code it replaced is deleted:
  - the `ckl::matmul_block` window calls and `ColumnOffset`;
  - `ckl::reduce<MAX>` on both schedules;
  - both normalize chains, `RecipSum` / `RECIP_COL_ONLY`, `cb_pv_out` (CB 5) and the row-group `pack_sfpu_fence`;
  - `PV_WINDOW_COLS`;
  - the per-tile `exp_packthread_tile` loop.
- **One carve-out**, earned by a measured regression: split reads keep every read on the reader when cb_query is 1
  deep (DRAM) and S_t ≤ 8 (`SPLIT_READS_LEGACY_MAX_SEQ_TILES`; `Plan.split_reads`, written `legacy if (cannot) else
  new`). It is the only dual path.
- The block-sequential schedule (bf16-DEST exact exp, blocks < 3 rows) takes A's row max and C's normalize too.
- The compute input waits in `compute_wait_qk` / `compute_wait_v` are now functional: the raw windows rely on them.
- **Instrumentation** (permanent, `MaybeDeviceZoneScope`):
  - reader: `q_reserve`, `q_lead`, `q_rest`, `kv_issue` / `kv_barrier`, `k`, `v`, `v_ones`, `presend_k`, `presend_v`;
  - writer: `q_reserve`, `q_lead`, `q_rest`, `v_reserve`, `v_ones`, `v`, `rows`, `barrier`;
  - compute: `wait_qk`, `wait_v`, `prologue`, `exp_pv_qk`, `row_max`, `qk` / `pv` (block-sequential).
  - The normalize now lives inside the P V window: it is timed by `compute_exp_pv_qk` / `compute_pv`.
  - Tools: `tests/ttnn/unit_tests/operations/nomic_sdpa/perf_tools/` (`zone_report.py`, `ablate.sh`, `sweep.sh`,
    `link_ceiling.py`).

### Guard set: pre-Perf-1 op (`e55220a99d9`) vs Perf 1, same session, median of 3, device us
| case | before | after | | case | before | after |
|---|---|---|---|---|---|---|
| 8x512 L1 | 101.1 | **70.2** | | 8x512 DRAM | 111.0 | 91.1 |
| 4x512 L1 | 49.6 | 44.0 | | 4x512 DRAM | 58.9 | 48.2 |
| 2x512 L1 | 27.9 | 23.2 | | 2x512 DRAM | 32.1 | 27.0 |
| 1x512 L1 | 18.6 | 17.2 | | 1x512 DRAM | 21.1 | 19.0 |
| 8x384 L1 | 71.3 | 48.3 | | 8x384 DRAM | 74.2 | 60.8 |
| 16x256 L1 | 55.6 | 40.4 | | 16x256 DRAM (carve-out) | 75.4 | 71.5 |
| 32x128 L1 | 31.6 | 24.8 | | 32x128 DRAM (carve-out) | 58.0 | 56.2 |
| 1x128 L1 | 4.8 | 4.5 | | 1x128 DRAM (carve-out) | 6.2 | 6.5 |
| 8x512 bf16 DEST exact exp | 201.9 | 159.3 | | 2x512 / 1x512 exact exp | 51.5 / 27.8 | 45.2 / 25.1 |
| 8x512 HiFi3 | 111.8 | 79.7 | | 2x512 / 1x512 HiFi3 | 30.4 / 19.2 | 25.6 / 17.7 |
| 8x512 full-sync | 110.9 | 84.0 | | 2x512 / 1x512 full-sync | 30.1 / 18.9 | 25.3 / 17.3 |
- No graduation regresses a cell. 1x128 DRAM (+0.3 us) was bisected over the commits, median of 5 each: zones-only
  6.5, A 6.5, B 6.5, C 6.4. The whole +0.3 us is the profiler cost of the permanent zones on a ~6 us op. They compile
  to nothing without the device profiler, and they are mandatory, so this is not a graduation regression.
- Against `feature_spec` (1350 MHz):
  - 8x512 70.2 vs incumbent 72.1 / acceptance 58 (now faster than the incumbent, above acceptance);
  - 4x512 44.0 vs 38.2 / 30;
  - 2x512 23.2 vs 24.5 / 18.

### What binds now (8x512 L1, after)
Full 69.3; compute floor 53.3; DM floor 55.8; sync floor 4.8. Compute still waits ~15.5 us at t = 0 for its first Q
+ K (`compute_wait_qk`; the reader's first K takes ~16.6 us on NOC0), then runs ~53 us. Round-2 leads: the startup K
burst (split K across NoCs for the first segment, or key-chunked K delivery so Q·K^T starts on the first chunk); the
neighbour push re-measured on the split-reads dataflow; `llk_s2`-style reorders; `ROW_GROUP_ROWS = 2` once L1 allows.

### Accuracy
Steps A, B and D are bit-identical. C changes the arithmetic: the divide happens in fp32 DEST before the bf16 pack.
`test_nomic_sdpa_precision_baseline.py` (16 / 16):
- 8x512: real PCC 0.999843, rel-RMS 0.02158 (was 0.0255; bar 0.0265); flat 0.00994 (0.0126); peaky 0.01699 (0.0191).
- 2x512: real rel-RMS 0.02123.
- bf16 DEST + exact exp at 8x512: real 0.02074, flat 0.01617, peaky 0.03214 (bar 0.0375).

### Tests
- `eval/golden_tests/nomic_sdpa/`: 45 passed / 45 xfailed, run after every step.
- `test_regression.py` + `test_nomic_sdpa_debug.py`: 23 passed / 5 xfailed.
- Acceptance `test_nomic_sdpa.py`: 43 passed. `test_nomic_sdpa_row_group_knobs` dropped its two `PV_WINDOW_COLS = 1`
  cases (knob removed) and added `ROW_GROUP_ROWS = 2`.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `ckl::matmul_block` (Q K^T / P V windows) | capability | issues `PACK(pack_reconfig_data_format(out))` on every output subblock whenever fp32 DEST is on, even with `DataFormatReconfig::NONE`; no switch suppresses it. Every CB is bf16, so it is a no-op reconfig that drains the packer. Also re-waits in0 / in1 and re-inits every call | 61700 (8x512 compute floor, with the block row max) | 56900 | `nomic_sdpa_compute.cpp:212` (`mm_window`) |
| `ckl::reduce<MAX, REDUCE_ROW>` | capability | no block row-max path: it unpacks A + scaler, pools and transposes per tile. `reduce_block_max_row` unpacks the row's N tiles in one macro-op with the scaler once and transposes once a row | 75300 (floor) / 1250 per 16-tile row | 63200 / 140 | `nomic_sdpa_compute.cpp:399` (`block_row_max`) |
| `ckl::matmul_block` (P V + normalize) | capability | always packs every DEST tile of its subblock to one CB, so it cannot pack a subset ([O0 O1] without l). The packer `Activation` hook works but packs l too, which needs a 3-wide cb_out and an extra pack on the binding pack thread | 73100 (floor, old kernel) | 72000 | `nomic_sdpa_compute.cpp:355` (`pv_norm_row`) |
| `ckl::eltwise_chain` + `ckl::Recip` / `BinaryFpu<Mul>` Col bcast (normalize) | capability | chains run SFPU work on the MATH thread only, which races the pack-thread exp and needs a fence; there is no pack-thread SFPU element and no fused "recip one DEST tile, multiply two others" element. The library `div_binary_tile` fp32 path is heavy and its init clobbers the exp constants | 75200 (floor, old kernel) | 71600 | `nomic_sdpa_compute.cpp:287` (`normalize_row_sfpu`) |
| `exp_packthread_tile` | capability | no multi-tile form: every tile pays an SFPU start / done + drain, although the tiles of a window are contiguous in DEST and the recorded replay body chains | 83.2 a tile; floor 54600 | 76.0 a tile; floor 53300 | `nomic_sdpa_compute.cpp:90` (`nomic_exp::exp_window`) |
| no helper (dataflow: same-core RISC-to-RISC signal) | capability | kernel_lib has no same-core NCRISC → BRISC flag. The `SenderPipe` Flag cell is the multicast flag's source on a sender core, so a same-core `receive()` resetting it would race the NoC read; a raw volatile L1 semaphore is used | n/a (no helper path) | n/a | `nomic_sdpa_reader.cpp:248` / `nomic_sdpa_writer.cpp:171` |
- The kernel-head comments at each site carry the measured justification. `probs_window`'s raw sub + pack-thread exp
  is unchanged from Refinement 3 and already documented.

## Perf 2 — perf tournament round 2 (re-measured breakdown → 6 ideas in two waves → 5 graduations)
- **Date**: 2026-11-05. Blackhole p300c, AICLK 1350 MHz. No SUPPORTED change.
- **Resumed**: the first round-2 coordinator was interrupted (~21:35 UTC) after recording the breakdown and four
  experiments. This run reused the finished experiments, finished `ones_column_l`, graduated, re-measured and then
  floated a second wave aimed at what was still hot.
- **Focus**: the `feature_spec.py` LOOSE_CASES `attention:` case, unchanged from Perf 1:
  - shape and placement: 8x512, bf16, L1, mask none, 176 KB reserved;
  - precision contract (fixed): HiFi2, fp32 DEST, approx exp, half-sync;
  - targets: incumbent 72.1, acceptance 58.0, stretch 48.0 us;
  - guards with the same weight: the 4x512 and 2x512 loose cases.

### Measured breakdown (start of round; 8x512 L1, after Perf 1; cumulative peel + permanent zones)
- Peel (us):
  - full 70.8;
  - reads stubbed 53.6;
  - + writes stubbed **53.3** (compute floor);
  - + exp stubbed 49.3;
  - compute stubbed **42.8** (DM floor);
  - reads only 35.7;
  - everything stubbed 4.8.
- Zones (per core p50, us):
  - unpack `compute_wait_qk` **16.9**: compute idles until its first Q + K. `reader_k` 18.2; the startup K burst's
    link ceiling is 3.7;
  - `compute_exp_pv_qk`: unpack 40.7, math 44.0, pack 46.9;
  - `compute_row_max`: unpack 5.8, math 3.5. Its unpack waited on the step's last Q K^T pack.
- Ranked:
  1. the t = 0 startup: ~17 us before compute starts, against a 3.7 us link ceiling;
  2. the steady-state compute floor of 53.3 us. Math and unpack are ~85% busy. The ones column is 20% of the P V
     tile-matmuls, and the row max sits between steps;
  3. the pack-thread exp (~4 us visible).
- Not floated: `kv_neighbor` (the second segment's K / V waits were ~0, off the critical path).

### Portfolio and verdicts
Experiment dirs are under `perf_experiments/`. Wave 1 is from the interrupted run; wave 2 was floated after
re-measuring the op with wave 1 graduated.

| idea | verdict | measured (us) | domain / notes |
|---|---|---|---|
| `rowmax_interleave` (w1): step = Q K^T(t+1, k) / probs(t, k) pairs, then P V(t-1), then row_max(t+1) | **WIN, graduated** | floor 53.3 → 48.8; 8x512 whole 70.1-71.2 → 64.7-65.4; unpack row-max zone 5.8 → 1.3 | everywhere, bit-exact (186 / 186 + 87 / 87). `RI_TRIG` (semaphore-gated chunked row-max unpack) −0.2 us more, as raw LLK: not taken. A generic pattern interpreter cost ~2 us of TRISC control: rejected |
| `ones_column_l` (w1, finished this run): the ones tile of [V \| 1] runs only the fidelity phases where bf16 1.0 has bits | **WIN, graduated** | floor 48.8 → 46.2; whole 8x512 66.6 → 63.8, 4x512 41.1 → 39.6, 2x512 22.5 → 21.7, 16x256 38.4 → 37.0, 8x512 DRAM 88.6 → 85.9, HiFi3 77.4 → 74.3, HiFi4 89.8 → 85.2, full-sync 82.3 → 79.6, exact exp 164.2 → 159.6 | everywhere, bit-exact in every config. HiFi2 runs phase {0}; HiFi3 / HiFi4 run phases {0, 2}. Odd phases multiply SrcA's low mantissa bits, which are zero for 1.0; phase 2 multiplies P's low bits. Phase 0 alone at HiFi3 / 4 is NOT exact (rel-RMS 0.01785 → 0.01656, ~440 k mismatches), so it was rejected as a precision change. `ones8b` (8 MVMULs, left half) 0.2-0.4 us slower with more raw LLK: dropped |
| `pv_lag_v_defer` (w1): P V lagged D groups; V deferred behind K | **NULL** (lag superseded), V defer REGRESSION | lag D = 2: floor 53.3 → 52.5 alone; on top of `rowmax_interleave` 48.72 → 48.90. V defer 8x512 70.5 → 85.4 (D1) / 77.1 (D4) | K is NOC0-bound with V on NOC1, so deferring V does not speed up K. D = 3 regresses DRAM 16x256 / 32x128 |
| `startup_k_first` (w1; ported clean by the coordinator): segment 0's K and V each read half by the reader (NOC0) and half by the writer (NOC1); reduce scaler prepared under the writer's first reads | **WIN, graduated** | same session (us), HEAD → new: 8x512 L1 63.7 → ~57.8 (pooled 10), 4x512 L1 38.5 → 30.7, 8x384 L1 42.0 → 35.0, 16x256 L1 36.8 → 32.5, 8x512 DRAM 85.7 → 83.4, full-sync 8x512 77.1 → 72.5, HiFi3 74.2 → 68.6 | split V + late scaler: everywhere under split reads. **K split carve-out** (measured): off when the t = 0 K burst is ≤ 768 tiles. K+V split vs V split alone: 2x512 DRAM 28.2 vs 25.4, 1x512 DRAM 20.0 vs 18.5, 1x512 L1 16.9 vs 16.3, 2x512 L1 21.0 vs 20.7, 1x128 L1 4.6 vs 4.2. 32x128 L1 (880 tiles) flat 24.7 vs 24.9, so it keeps the K split. Tried null / worse: `kfirstw`, `kqmerge`, `knomcast`, `mcastpipe`, `kw14`, chip-wide phase barriers, time-delayed V (bimodal) |
| `startup_first_qk` (w2): throttle the segment-0 burst (read barrier every 6 pages on both RISCs) + reader takes 11/16 of segment 0's key rows | **WIN, graduated** | 8x512 L1 60.9 → 53.6 (subagent, median of 15); spread 55.1-67.1 → 53.1-57.2. Coordinator, combined with w2's compute change: 58.3 → 52.7 (11 reps) | Root cause: ~200 RISCs with 16-32 reads each in flight collapse the NoC; first V landed anywhere from 9 to 25 us. Carve-outs (both measured): (a) burst ≤ 1536 tiles, where throttling costs 4x512 L1 31.0 → 31.5, 2x512 21.2 → 22.3, 1x512 16.8 → 18.5; (b) **block-sequential schedule (bf16 DEST exact exp) with a 1-deep cb_query (DRAM)**, found by the coordinator's guard sweep: throttled vs not, 8x512 DRAM 172.3 vs 169.7, 8x384 109.8 vs 107.3, 16x256 114.4 vs 113.3. In L1 the same schedule gains 155.8 → 145.3, 104.0 → 91.2 and 78.8 → 73.6. Mechanism of (b) not isolated. Options measured: raw NIU sliding window of 8, 55.1 (the only ungated one, but +0.9 DRAM, +0.3 1x512); trid batches of 4, 54.6; chunks of 4 / 8 / 12, 54.9 / 54.7 / 55.2; K shares of 9-13 sixteenths. Compute consuming K per half: ≤ 0.5 us upside, not built |
| `compute_floor_r2` (w2): pack each DEST window's contiguous tiles with one pack MOP; pack-thread exp init once a kernel | **WIN, graduated** | floor 46.6 → 45.6; whole 8x512 58.4 → 56.9, 4x512 30.6 → 30.3, full-sync 72.2 → 70.8, HiFi3 69.1 → 67.9 | bit-exact 78 / 78. Carve-out (measured): one window per score row (`N == qk_w`, e.g. 32x128) keeps per-tile packs; ungated, its floor went 18.51 → 19.02 (the MOP is re-programmed twice a row). Findings: P V is FPU-bound under the fixed fidelity (the ones-tile MVMULs fill a SrcA gap, 0.56 us total); the Q K^T / probs pairs are pack-bound (a 200-cycle probe on pack costs 7.0 us, on math 0.7). Null / slower: a MATH_PACK-semaphore exp start, math commit without the SFPU wait, raw sub with m unpacked once, ones face-0 unpack, per-tile exp / pack interleave (48.2), `ROW_GROUP_ROWS = 2` (46.6, needs +128 KB) |

### What graduated (in order; each golden-green, committed separately)
| step | commit | 8x512 L1 whole (us) | compute floor |
|---|---|---|---|
| before (Perf 1 end) | `16907f6d418` | 70.2-71.3 | 53.3 |
| A `rowmax_interleave` | `485833c075b` | ~65 | 48.8 |
| B `ones_column_l` | `49ac9d0ce57` | 63.7-63.8 | 46.2 |
| C split startup (`startup_k_first`) | `f312b9a3c80` | ~57.8 (bimodal 55-61.5) | 46.8 |
| D window pack MOP + exp init once (`compute_floor_r2`) | `dca20bbeb51` | — | 45.6 |
| E throttled startup (`startup_first_qk`) | `ce715f515b4` | **52.3** (tight, 52.2-53.2) | 45.8 |
- Each graduation is the op's one path, and the code it replaced is deleted:
  - the probs-then-FPU-partner step order;
  - the full-MOP ones-tile product;
  - the writer's up-front scaler fill;
  - the per-tile `pack_tile` window packs, except under carve-out D;
  - the per-row-group exp init.
- **Carve-outs** (each earned by a measured regression and written `legacy if (cannot) else new`):
  - C: no K split (V still split) when the t = 0 K burst is ≤ `SPLIT_K_LEGACY_MAX_BURST_TILES` = 768
    (`Plan.k_split_rows`).
  - D: per-tile packs with one window per score row (`pack_runs`, constexpr).
  - E: unthrottled when the burst is ≤ `READ_CHUNK_MIN_BURST_TILES` = 1536, or on the block-sequential schedule with a
    1-deep cb_query (`Plan.read_chunk`).
- New host quantity: `Plan.k_burst_tiles`, the segment-0 K tiles the cores read themselves at t = 0. Multicast
  receivers read none. Both startup gates key on it.
- **Instrumentation** (permanent, `MaybeDeviceZoneScope`). New zones:
  - reader: `reader_k_wait_half`, `reader_v_half`, `reader_v_wait_half`, `reader_read_chunk`;
  - writer: `writer_scaler`, `writer_k_half`, `writer_v_wait_half`, `writer_read_chunk`.
  - 39 zones, 0 profiler zone-hash collisions. Busiest RISC 72 markers (cap ~250). Zones cover 99-100% of every
    RISC's span.
  - `writer_scaler` nests under the K-half reads: it times their barrier (occupancy), not the fill.

### Guard set: Perf 1 end (`16907f6d418`) vs Perf 2 end (`ce715f515b4`), same session, device us
Medians of 5; the focus is a median of 11; the configs are medians of 3.

| case | before | after | | case | before | after |
|---|---|---|---|---|---|---|
| **8x512 L1** | 70.2 / 71.3 | **52.3 / 52.5** | | 8x512 DRAM | 90.6 | 82.5 |
| 4x512 L1 | 42.6 | 30.4 | | 4x512 DRAM | 48.0 | 44.7 |
| 2x512 L1 | 23.0 | 20.7 | | 2x512 DRAM | 27.0 | 25.4 |
| 1x512 L1 | 17.2 | 16.4 | | 1x512 DRAM | 19.0 | 18.5 |
| 8x384 L1 | 46.0 | 34.4 | | 8x384 DRAM | 61.4 | 56.7 |
| 16x256 L1 | 41.0 | 31.6 | | 16x256 DRAM | 71.7 | 67.4 |
| 32x128 L1 | 24.2 * | 24.7 * | | 32x128 DRAM | 55.6 * | 56.3 * |
| 1x128 L1 | 4.5 | 4.2 | | 1x128 DRAM | 6.5 | 6.4 |
| 8x512 HiFi3 L1 / DRAM | 79.5 / 99.9 | 65.0 / 94.2 | | 8x512 full-sync L1 / DRAM | 83.4 / 100.7 | 65.5 / 95.2 |
| 8x512 exact exp L1 / DRAM | 165.5 / 174.1 | 145.8 / 169.3 | | 2x512 HiFi3 / full-sync / exact (L1) | 25.8 / 25.4 / 45.0 | 23.5 / 23.3 / 44.1 |
| 1x512 HiFi3 / full-sync / exact (L1) | 17.7 / 17.4 / 25.2 | 16.9 / 16.6 / 24.4 | | same, DRAM | 19.7 / 19.3 / 27.0 | 19.1 / 18.7 / 26.4 |

- **No regression.** \* 32x128 was re-measured interleaved (before / after / before / after, 9 reps each):
  - L1 25.3 / 24.7 / 24.6 / 24.2;
  - DRAM 56.0 / 56.0 / 55.8 / 55.9.
  - The first sweep's +0.5 / +0.7 was session noise. Every step-wise comparison in the round also showed 32x128 flat.
- Against `feature_spec` (1350 MHz):

  | case | now | incumbent | acceptance | stretch | Perf 1 |
  |---|---|---|---|---|---|
  | 8x512 | **52.3** | 72.1 | 58.0 | 48.0 | 70.2 |
  | 4x512 | 30.4 | 38.2 | 30.0 | | 42.6 |
  | 2x512 | 20.7 | 24.5 | 18.0 | | 23.0 |

  8x512 now beats acceptance, 1.38x faster than the incumbent.

### What binds now (8x512 L1, end of round)
- Peel: full 52.2; compute floor 45.8; DM floor 39.2 (noisy, 38.6-48.9); sync floor 6.4.
- Compute waits only 4.5 us p50 / 6.7 max at t = 0 (it was 16.9), then runs ~45 us.
- The op is compute-bound: three TRISCs each ~39-40 us busy in `compute_exp_pv_qk`.
  - P V is FPU-bound at the fixed fidelity.
  - The Q K^T / probs pairs are pack-thread-bound: the exp plus packs.
- Leads for a round 3:
  - fewer pack-thread instructions a probs window. Magic exp stays out: it is incorrect for logit gaps > 264;
  - the 2x512 / 4x512 guards (20.7 / 30.4 vs 18.0 / 30.0), whose startup is unthrottled and K-split-less by the
    carve-outs, so their startup is the next DM lever.
- Follow-up flagged by `compute_floor_r2`, not acted on:
  - the pack-thread exp waits for its DEST section through `STALLWAIT(STALL_SFPU, MATH)` (an idle FPU), because
    `tile_regs_wait`'s SEMWAIT does not block SFPU instructions;
  - deleting that stall gave wrong results;
  - the explicit MATH_PACK-semaphore form is bit-exact and 0.1 us faster. Worth a review: can HEAD race if the FPU
    idles mid-window?
  - Golden and all bit-exact checks pass.

### Accuracy
- Every round-2 graduation is bit-identical to the op before it:
  - A: 186 + 87 cases;
  - B: 69 + 51 checks across HiFi2 / 3 / 4, full-sync and bf16 DEST;
  - C: 23 cases;
  - D: 78;
  - E: 99 cases including DRAM and every config.
- The Perf 1 precision numbers stand: 8x512 real rel-RMS 0.02158, PCC 0.999843.

### Tests
- Run after every step: `eval/golden_tests/nomic_sdpa/` (45 passed / 45 xfailed, includes `test_regression.py`),
  `test_nomic_sdpa.py`, `test_nomic_sdpa_debug.py`, `test_nomic_sdpa_precision_baseline.py`.
- Final: 113 passed / 45 xfailed.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `ckl::matmul_block` / `ckernel::matmul_block` (P V k-step, ones tile of [V \| 1]) | capability | runs every in1 tile of a block through one MOP at the configured fidelity: no per-tile phase subset. It cannot express "this operand is exact at phases {0} / {0, 2}". The raw `pv_ones::k_step` replays the MOP's recorded 16-MVMUL tile body (mirrors `matmul_block_math_dynamic_throttle`, throttle branches included). HiFi3 / 4 use ADDR_MOD_3, which the Blackhole matmul init does not program, so it is re-set after every matmul init | 48800 (8x512 L1 floor) | 46200 | `nomic_sdpa_compute.cpp:147` (`pv_ones::k_step`, line 168) |
| `pack_tile` / `pack_block` / `ckl::PackTile` (Q K^T, probs and P V window packs) | capability | every API packs one MOP per tile. On Blackhole `pack_block` is a `pack_tile` loop. Each pack re-programs the packer's L1 address with a stall and closes the tile (a packer drain); nothing packs a contiguous DEST run in one MOP. Raw `llk_pack_init<Default, false, true, true>(cb, n)` (re-programs only the MOP's num_tiles) + `llk_pack` | 46620 (floor) | 45600 | `nomic_sdpa_compute.cpp:217` (`nomic_pack::pack_window`, gate `pack_runs` line 458) |
| no helper (dataflow: same-core RISC-to-RISC signal) — new sites of the Perf 1 row | capability | kernel_lib still has no same-core NCRISC ↔ BRISC flag. The split startup adds two raw volatile L1 semaphores: `k_half` (writer → reader: its segment-0 K part landed) and `v_half` (writer → reader: its half of a SEND slot's V). It also reuses the V hand-off flag for READ slots | n/a (no helper path) | n/a | `nomic_sdpa_reader.cpp:141` / `nomic_sdpa_writer.cpp:91, 235, 261` |
- Not bypasses:
  - the read throttle is plain `noc_async_read_barrier()`;
  - the split reads use the op's existing `noc_async_read_page`;
  - the reordered windows reuse Perf 1's raw `mm_window` (already recorded).
- The kernel-head / site comments carry the measured justification at every bypass.

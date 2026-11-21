# Perf 18 — shared context for every perf-part-optimizer (read this first)

## Ground rules
- Repo: `/localdev/pmilojevic/workspaces/opgen` (a git worktree). Before ANY device run:
  `export TT_METAL_HOME=$PWD PYTHONPATH=$PWD; source python_env/bin/activate`
  (otherwise the run silently tests another checkout).
- Device runs only through `scripts/run_safe_pytest.sh` / `scripts/tt-probe.sh`, in the foreground.
  Never `run_in_background`, never `python3` with device code. Device runs serialize on a flock with
  the other 5 optimizers; that is expected. Keep each run small (only the cases you need).
- Do NOT edit the real op (`ttnn/ttnn/operations/conv2d_3x3/`). Copy it into your own dir
  `ttnn/ttnn/operations/conv2d_3x3/perf_experiments/<your_slug>/` as `op_base/` (untouched baseline)
  and `op_copy/` (your candidate), the way `perf_experiments/nsplit_resident/` and
  `perf_experiments/bias_epilogue/` did (a pytest that imports both, signposts each call
  `cmp|<case>|<variant>|<i>` with `tracy.signpost`, a `run.sh` that runs `--profile --run-all` and parses
  the ops CSV with `perf_experiments/asym_padding/cmp_windows.py`; kernels resolve via the copy's
  `KERNEL_DIR`). Copy the WORKING TREE, not HEAD: it carries uncommitted user changes in
  `conv2d_3x3.py` / `conv2d_3x3_program_descriptor.py` (the `l1_small_bytes` L1-budget term,
  `prepare_conv2d_weights_host`) that the measurements depend on. The device is opened with
  `l1_small_size=65536` (FLUX.2's VAE does that), which shrinks the CB budget.
- Do not commit. Do not spawn other optimizer agents. Under `ttnn/ttnn/` import torch only inside
  functions (the pre-commit hook `check-torch-imports-in-ttnn` walks untracked files too).
- Precision contract is fixed: bf16 activations + weights, HiFi4, `fp32_dest_acc_en=True`,
  `math_approx_mode=False`. Never change those for speed. Preferred bar: output bit-identical to
  `op_base` (sha1 of the output bytes). If not bit-identical, report PCC vs torch (current
  0.9999985 on every focus case) and how many outputs differ and by how many bf16 ulp.
- Correctness is the only pass/fail. Perf is measured (DEVICE KERNEL DURATION), never asserted.
  Device kernel time has no warm-up transient; the harness already times 3 calls per case (median).
  Run-to-run noise is ~0.5-1.5%.
- Guards owned by the coordinator: Panoptic-DeepLab (L1-sharded, LoFi bf8_b, 22 cases) and the
  single-device FLUX.2 TILE-input convs. State explicitly whether your change can reach sharded or
  TILE-input programs. If it reaches TILE input, measure 2-3 TILE cases too
  (`FLUX_CMP_BH_LAYOUT=TILE` on the bh2x4 idx, or the `1x1` group idx 0-30). If it reaches sharded
  programs, run a few Panoptic compare cases (`CONV_CMP_CASES=...` with
  `tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_panoptic_conv3x3_compare.py`).
- Code comments: present-tense intent only (why), no history / no narration of what you tried.
  Kernel zones (`MaybeDeviceZoneScope` from `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`) are
  permanent: keep the existing ones and add zones to any new stage you create.

## The focus: 17 FLUX.2 bh2x4 per-device decoder convs
bf16, HiFi4, fp32 DEST, bias, padding (0, 0, 1, 1) (= pad_top 0, pad_bottom 0, pad_left 1,
pad_right 1: the input carries the halo rows, H_out = H - 2, W_out = W), DRAM-interleaved ROW_MAJOR
input, output DRAM-interleaved TILE (C_out padded to 32 lanes), Blackhole p150b, 110 cores (11x10),
8 DRAM banks, 1350 MHz. Whole-op harness (the real op):
`FLUX_CMP_CASES=<idx,...> FLUX_CMP_SKIP_TTNN=1 scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_flux_conv3x3_compare.py`
then `python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py <PROFILER CSV>`
(`perf_experiments/ablation_r18/ab.sh <label> <ablate-tokens> <idx,...>` wraps both).
See that test for how the op is called (weights/bias prep, ComputeConfigDescriptor).

Every case runs the interleaved K-blocked **tap-direct** walk with the **ROW_MAJOR ring slab**
(`kernels/conv2d_3x3_rm_ring.hpp`, reader `RM_RING_PX`), **resident weights** (`WEIGHTS_RESIDENT`),
and the **bias pass inside the last K-block** (`kernels/conv2d_3x3_kblock_walk.hpp`) — except the
one-M-block cases idx 31 / 32 / 34 (per-block slab, `BANK_RUN_FILL`, streamed weights, 3 K-blocks).
K-block = one tap (kh, kw): bk = C_in/32 tiles (tw), 9 K-blocks per M-block; per K-block compute
tilizes bm units (one unit = 32 consecutive output pixels x one tap, read straight from the ring by
`fifo_tilize` from a 16 B entry of `cb_src_table` the reader wrote, `TapDirectFifo` in
`kernels/conv2d_3x3_gather_dataflow.hpp`), then multiplies bm x bn x bk and packs fp32 partials
(packer L1 accumulation) into a [bm, bn] caller-owned region; the last K-block runs the bias pass
2 row groups behind its matmul into `cb_output_tiles` (depth 2 row groups), which the writer
(BRISC/NoC1) stores to DRAM. W_out % 32 == 0 on every case: a tile-row of 32 output pixels lies in one
image row, and the kh offset is W/32 whole tile-rows.

| idx | prio | case (H x W, C_in->C_out) | calls | now us | target us | DRAM floor us* | matmul ideal us | blocking (groups MxN, core_m, bm, bk, bn, nkb, M-blocks/core, subblock, ring_px) |
|---|---|---|---|---|---|---|---|---|
| 41 | 1 | 1026x2048 32->128 | 5 | 1821.2 | 1453 | 1490 | 1011 | 110x1, 596, 46, 1, 4, 9, 13, 1x4, 5570 |
| 42 | 2 | 514x1024 64->256 | 6 | 1373.4 | 1453 | 745 | 1011 | 110x1, 149, 15, 2, 8, 9, 10, 1x4, 2530 |
| 35 | 3 | 1026x2048 128->3 | 1 | 2398.1 | 136 | 1490 | 1011 | 110x1, 596, 7, 4, 1, 9, 86, 1x1, 4322 |
| 36 | 4 | 514x1024 32->128 | 5 | 595.1 | 363 | 373 | 253 | 110x1, 149, 50, 1, 4, 9, 3, 1x4, 3650 |
| 44 | 5 | 1026x2048 64->128 | 1 | 2920.3 | 2905 | 1790 | 2022 | 110x1, 596, 24, 2, 4, 9, 25, 1x4, 4866 |
| 46 | 6 | 1026x2048 64->256 | 1 | 5316.6 | 5810 | 2980 | 4045 | 110x1, 596, 10, 2, 8, 9, 60, 1x4, 4418 |
| 43 | 7 | 258x512 128->512 | 8 | 1309.4 | 1453 | 373 | 1011 | 55x2, 75, 7, 4, 8, 9, 11, 1x4, 1250 |
| 37 | 8 | 258x512 64->256 | 5 | 448.4 | 363 | 187 | 253 | 108x1, 38, 19, 2, 8, 9, 2, 1x4, 1634 |
| 33 | 9 | 514x1024 128->3 | 1 | 800.8 | 34 | 373 | 253 | 110x1, 149, 30, 4, 1, 9, 5, 3x1, 3010 |
| 38 | 10 | 130x256 128->512 | 17 | 387.5 | 363 | ~95 | 253 | 54x2, 19, 10, 4, 8, 9, 2, 1x4, 834 |
| 39 | 11 | 514x1024 64->128 | 1 | 799.4 | 726 | 447 | 505 | 110x1, 149, 30, 2, 4, 9, 5, 1x4, 3010 |
| 45 | 12 | 514x1024 128->256 | 1 | 2794.6 | 2905 | 895 | 2022 | 110x1, 149, 2, 4, 8, 9, 75, 1x4, 2114 |
| 47 | 13 | 514x1024 128->512 | 1 | 5389.9 | 5810 | 1490 | 4045 | 55x2, 298, 2, 4, 8, 9, 149, 1x4, 2114 |
| 40 | 14 | 258x512 128->256 | 1 | 745.2 | 726 | 223 | 505 | 108x1, 38, 7, 4, 8, 9, 6, 1x4, 1250 |
| 34 | 15 | 66x128 128->512 | 10 | 139.4 | 91 | — | 63 | 86x1, 3, 3, 12, 16, 3, 1 (streamed weights), 1x4, — |
| 32 | 16 | 130x256 32->128 | 1 | 70.6 | 23 | — | 16 | 103x1, 10, 10, 3, 4, 3, 1, 1x4, — |
| 31 | 17 | 66x128 32->128 | 1 | 26.1 | 6 | — | 4 | 86x1, 3, 3, 3, 4, 3, 1, 1x4, — |

Call-weighted over the 17: **62.28 ms** (110-core HiFi4 ideal 40.18 ms). *DRAM floor = (input bytes
read once + output bytes incl. the 32-lane padding) / 450 GB/s. P1, P4, P3, P9 are DRAM-bound at their
roofline; P5, P2, P11 are near-balanced; the rest are compute-bound.

## Measured breakdown, this round's start (us; `perf_experiments/ablation_r18/`, `apply_hooks.py <op_dir>`
adds `CONV2D_3X3_ABLATE=fill,weights,store,tilize,bias,matmul` stubs to any op copy: payload stubbed,
CB / trid sync kept. Raw: `ablation_r18/ab_results.txt`)

| stubbed (cumulative) | P1 | P3 | P6 | P2 | P4 | P7 | P10 | P8 | P15 |
|---|---|---|---|---|---|---|---|---|---|
| none | 1821 | 2398 | 5317 | 1373 | 595 | 1309 | 388 | 448 | 139 |
| fill | 1764 | 1944 | 5206 | 1357 | 569 | 1286 | 374 | 430 | 138 |
| + weights | 1766 | 1943 | 5198 | 1354 | 565 | 1282 | 369 | 424 | 126 |
| + store | 1547 | 1933 | 5110 | 1264 | 392 | 1241 | 324 | 323 | 103 |
| + tilize | 1462 | 1574 | 4889 | 1208 | 372 | 1194 | 313 | 310 | 103 |
| + bias | 1401 | 1558 | 4771 | 1177 | 357 | 1179 | 309 | 302 | 102 |
| + matmul (everything) | 400 | 662 | 531 | 128 | 108 | 95 | 29 | 38 | 17 |
| solo: store | 1615 | 2355 | 5265 | 1309 | 432 | 1306 | 368 | 354 | 118 |
| solo: tilize | **2001** | 2405 | 5115 | 1318 | 607 | 1262 | 374 | 435 | 139 |

Readings:
- **Matmul is at its roofline**: removing it saves ~ its 64-cycle/tile-MAC HiFi4 ideal on every case.
  Everything above the ideal (62.3 vs 40.2 ms) is the rest.
- **The everything-stubbed floor is large and NOT overlapped with the matmul**: P1 matmul-only 1401 =
  1000 matmul + 400 floor. It scales with tap-direct units (tile-rows x 9 taps per core: 5364 on
  P1 / P3 / P6, 1341 on P2 / P4, 675 on P7): ~75-140 ns (100-190 cycles) per unit. P3 adds 86 M-blocks x
  9 K-blocks of per-K-block fixed cost (bm 7: a K-block's matmul is only 28 tile-MACs = 1.3 us). Zones
  agree: P4's `compute_tilize` zone is 144 us vs ~20 us of tilize payload; P2's 190 vs 55. Whether the
  per-unit cost is the compute handshake (table wait / pop, `invalidate_l1_cache`, per-unit DEST
  sections) or the reader's per-unit planning (`TapDirectFifo::unit`, ring cursors, fill steps on the
  same NCRISC) is NOT settled. A prior round on Panoptic measured a tap-direct reader's per-unit
  planning at ~200 cycles and found it pacing the tilize.
- **The store is serialized into the last K-block**: the whole M-block's output (P1: 46 x 4 tiles =
  368 KB per core, ~40 MB chip-wide per M-block) drains during the last of 9 K-blocks through a 2-row-
  group output CB, so compute stalls in the bias pass (`compute_last_kblock` P4 p50 139 / max 231 us for
  3 executions; a normal K-block is ~10 us). Solo store stub: P1 -206, P4 -163, P8 -95, P2 -65 us.
- **P1 is DRAM-coupled**: stubbing only the tilize made P1 180 us SLOWER (the store burst then collides
  with the fill). P1 moves 671 MB in 1.82 ms (368 GB/s); its DRAM floor is ~1.49 ms.
- **conv_out (P3, P9)**: N = 1 tile for 3 channels (29 of 32 matmul lanes wasted); bm 7 (ring of 2 image
  rows of 256 B pixels) gives 1x1 subblocks and 774 K-blocks per core; the fill is exposed (-454 us when
  stubbed) with a per-grid-row imbalance (`reader_band_wait` p50 254, max 474 us; kernel span max/p50
  1.10).
- Tilize payload ~21 cycles per tile (P3 358, P6 220, P1 84 us). Bias pass 15-118 us. Weights resident:
  ~0.
- Zones: `perf_experiments/zone_report.py <profile_log_device.csv>`; the big cases (P1, P3, P5, P6,
  P12, P13) exhaust the ~250-marker budget per RISC, so read zones on P2 / P4 / P7 / P8 / P10.

## Prior rounds (do not re-float their nulls unless you name what is different)
Read `ttnn/ttnn/operations/conv2d_3x3/changelog.md` sections "Perf 16" and "Perf 17" (FLUX.2 rounds)
and the 2026-11-20 entries at the end of `.claude/references/op-gen-lessons-learned.md`. Notably:
store from the reader (NoC0) regressed (+4.5% to +14.7%: DRAM writes on NoC0 cost ~2x NoC1);
background-refilling a re-read slab saturated DRAM and starved stores; bias seeded at K-block 0 was
slower on the K-blocked walk; K-block coarsening on a LoFi Panoptic walk barely moved it because the
L1-acc packs hid under the matmul.

## What to return
verdict (WIN / NULL / REGRESSION / BLOCKED); base vs candidate device ns on your primary cases and
on a sweep of every focus case your change reaches (all 17 if it reaches them); bit-identity / PCC;
L1 impact (does the blocking solve change?); the domain (where it applies; list only exceptions that
are incorrect, inexpressible or measured slower — flat and untested are not exceptions); any raw-LLK /
raw-NoC helper bypass (helper name, capability vs ergonomics, what was missing / hard); an
`integration.patch` against the real op (`diff -ru` of op_base -> op_copy) plus a short README in your
dir; a menu of variants with their ns and precision if you tried more than one.

## Code map (line numbers at this round's start)
- Host: `conv2d_3x3.py` (call path, `select_blocking` call ~L705); `conv2d_3x3_program_descriptor.py`:
  `l1_cb_budget_bytes` L533, `Conv3x3Blocking` L665, tap-direct sizing `_tap_direct*` L1051-1139,
  `weights_resident_capable` L1327, `rm_ring_capable` / `rm_ring_plan` L1342-1432 (the ring solve),
  `select_blocking` L1439, `kw_fold_plan` L2280, `create_program_descriptor` L2778 (CBs, CT args,
  defines ~L4000-4110).
- Reader (NCRISC/NoC0) `kernels/conv2d_3x3_reader.cpp`: ring fill entry ~L1242 (`RM_RING_PX`), the
  K-blocked tap-direct loop ~L1809-1980 (ring release / band wait, `fifo.reserve(..., rm_idle)`,
  `fifo.unit(...)`). `kernels/conv2d_3x3_rm_ring.hpp` (`RmBankRing`: issue / process / step / need).
  `kernels/conv2d_3x3_gather_dataflow.hpp` `TapDirectFifo` L675 (reserve L798-830, unit L872).
- Compute `kernels/conv2d_3x3_compute.cpp`: `fifo_tilize` L627-733 (per-unit table wait, sources,
  fast tilize, pops), `TilizeIm2colKBlock` L763-840 (the PreKBlockFn), `ResidentWeightsPreKBlock`
  L863, the walk dispatch L2200-2250. `kernels/conv2d_3x3_kblock_walk.hpp` (raw-LLK K-block walk,
  partials, interleaved bias pass). `kernels/conv2d_3x3_lean_tilize.hpp` (BH fast tilize pack side).
- Writer (BRISC/NoC1) `kernels/conv2d_3x3_writer.cpp`: weight rounds `issue_weight_k_block` ~L1395-
  1500, output store ~L2630-2660 (per row group: `noc_async_write_page` per tile, flushed, pop).

## Harness traps
- `run_safe_pytest.sh` precompiles by running each test body once in a collect pass before the real
  pass: guard expensive host work and result writes on that pass (see `_in_collect_pass()` in
  `test_flux_conv3x3_compare.py`); `cmp_windows.py` already drops its empty signpost windows.
- The big cases' device-profile CSVs are 0.5-1 GB; delete or gzip them in your `runs/` dir, and keep
  nothing over 500 KB that you want committed (the commit hook rejects large files).

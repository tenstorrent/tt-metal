# Perf 19 — shared context for every perf-part-optimizer (read this first)

## Ground rules
- Repo: `/localdev/pmilojevic/workspaces/opgen` (a git worktree). Before ANY device run:
  `export TT_METAL_HOME=$PWD PYTHONPATH=$PWD; source python_env/bin/activate`
  (otherwise the run silently tests another checkout).
- Device runs only through `scripts/run_safe_pytest.sh` / `scripts/tt-probe.sh`, in the foreground.
  Never `run_in_background`, never `python3` with device code (a bare `import ttnn` also opens the device
  outside the lock). Device runs serialize on a flock with the other 5 optimizers; that is expected. Keep
  each run small (only the cases you need), and keep every single tool call well under 10 minutes: a
  call that hits the tool time limit kills a `--profile` run before the profiler CSV is written.
- Do NOT edit the real op (`ttnn/ttnn/operations/conv2d_3x3/`). Copy it into your own dir
  `ttnn/ttnn/operations/conv2d_3x3/perf_experiments/<your_slug>/` as `op_base/` (untouched baseline)
  and `op_copy/` (your candidate), the way `perf_experiments/ring_fill_noc_split/` and
  `perf_experiments/whole_k_rowstream/` did (a pytest that imports both, signposts each call
  `cmp|<case>|<variant>|<i>` with `tracy.signpost`, a `run.sh` that runs `--profile --run-all` and parses
  the ops CSV with `perf_experiments/asym_padding/cmp_windows.py`; kernels resolve via the copy's
  `KERNEL_DIR`). Copy the WORKING TREE, not HEAD: it carries uncommitted user changes in
  `conv2d_3x3.py` / `conv2d_3x3_program_descriptor.py` / `__init__.py` (the `l1_small_bytes`
  L1-budget term, `prepare_conv2d_weights_host`) that the measurements depend on. The device is opened
  with `l1_small_size=65536` (FLUX.2's VAE does that), which shrinks the CB budget.
- Do not commit. Do not spawn other optimizer agents. Under `ttnn/ttnn/` import torch only inside
  functions (the pre-commit hook `check-torch-imports-in-ttnn` walks untracked files too).
- Precision contract is fixed: bf16 activations + weights, HiFi4, `fp32_dest_acc_en=True`,
  `math_approx_mode=False`. Never change those for speed. Preferred bar: output bit-identical to
  `op_base` (sha1 of the output bytes). If not bit-identical, report PCC vs torch (current
  0.9999985 on every focus case) and how many outputs differ and by how many bf16 ulp.
- Correctness is the only pass/fail. Perf is measured (DEVICE KERNEL DURATION), never asserted.
  Device kernel time has no warm-up transient; the harness already times 3 calls per case (median).
  Run-to-run noise is ~0.5-1.5%. A claimed win under ~2% needs an A/B repeat (alternate the variant order).
- Guards owned by the coordinator: Panoptic-DeepLab (L1-sharded, LoFi bf8_b, 22 cases) and the
  single-device FLUX.2 TILE-input convs. State explicitly whether your change can reach sharded or
  TILE-input programs. If it reaches TILE input, measure 2-3 TILE cases too
  (`FLUX_CMP_BH_LAYOUT=TILE` on the bh2x4 idx, or the `1x1` group idx 0-30). If it reaches sharded
  programs, run a few Panoptic compare cases (`CONV_CMP_CASES=...` with
  `tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_panoptic_conv3x3_compare.py`).
- Code comments: present-tense intent only (why), no history / no narration of what you tried.
  Kernel zones (`MaybeDeviceZoneScope` from `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`) are
  permanent: keep the existing ones and add zones to any new stage you create (mind the ~250-marker
  budget per RISC: no per-tile or per-chunk zones).

## The focus: 17 FLUX.2 bh2x4 per-device decoder convs
bf16, HiFi4, fp32 DEST, bias, padding (0, 0, 1, 1) (= pad_top 0, pad_bottom 0, pad_left 1,
pad_right 1: the input carries the halo rows, H_out = H - 2, W_out = W), DRAM-interleaved ROW_MAJOR
input, output DRAM-interleaved TILE (C_out padded to 32 lanes), Blackhole p150b, 110 cores (11x10),
8 DRAM banks, 1350 MHz. Whole-op harness (the real op):
`FLUX_CMP_CASES=<idx,...> FLUX_CMP_SKIP_TTNN=1 scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_flux_conv3x3_compare.py`
then `python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py <PROFILER CSV>`
(`perf_experiments/ablation_r19/ab.sh <label> <ablate-tokens> <idx,...>` wraps both for the real op).
See that test for how the op is called (weights/bias prep, ComputeConfigDescriptor).

Paths (Perf 16-18 built them; read `changelog.md` "Perf 16", "Perf 17", "Perf 18" for each):
- Every case except idx 31 / 32 / 34 runs the interleaved K-blocked **tap-direct** walk on the **ROW_MAJOR
  ring slab** (`kernels/conv2d_3x3_rm_ring.hpp`, reader `RM_RING_PX`: each core reads its contiguous
  pixel range once, as 8-bank bank runs into 2 x 8 KB staging chunks, de-interleaved one pixel per
  self-aimed NoC read into the ring) with **resident weights** (`WEIGHTS_RESIDENT`) and the **bias pass
  inside the last K-block** (`kernels/conv2d_3x3_kblock_walk.hpp`). K-block = one tap (kh, kw), bk =
  C_in/32 tiles, 9 K-blocks per M-block, fp32 partials by packer L1 accumulation, output through
  `cb_output_tiles` (2 row groups) to the writer (BRISC/NoC1), which stores whole 2 KB tiles to DRAM.
- **kh tile reuse** (`KH_REUSE`, compute `kh_reuse::` ~L880): C_in >= 64 where the tile ring fits
  (P2, P8, P10, P11): each input tile-row tilized once into a tile ring, K-block (kh, kw) reads ring rows
  at an offset. It re-picked bm (P2's M-blocks are ~26 us now).
- **C_out fold** (`CONV2D_3X3_COUT_FOLD`, `kernels/conv2d_3x3_cout_fold.hpp`): C_out <= 4 (P3, P9):
  kw folded into the N lanes, 3 K-blocks, plane packs, final pass with bias.
- **Ring startup split** (`RM_SPLIT`, rm_ring.hpp `RmSplitWords` / writer `serve()` ~L538-700): the
  first ring window (>= 512 KB, `RM_SPLIT_MIN_WINDOW_BYTES`) split reader-NoC0 front / writer-NoC1 back
  with Dekker claims; the writer serves only while it waits for its first output.
- **N split** (P7, P10, P13: 2 N groups, each [K, N/2] resident).
- idx 31 / 32 / 34 (one M-block per core): per-block slab, `BANK_RUN_FILL`, streamed weights.

| idx | prio | case (H x W, C_in->C_out) | calls | now us | target us | 110-core ideal us | DRAM floor us* | per-core BRISC span max / p50 / min |
|---|---|---|---|---|---|---|---|---|
| 41 | 1 | 1026x2048 32->128 | 5 | 1819 | 1453 | 1017 | 1490 | 1819 / 1702 / 1608 (1.07) |
| 42 | 2 | 514x1024 64->256 | 6 | 1385 | 1453 | 1017 | 745 | 1386 / 1376 / 1317 (1.007) |
| 35 | 3 | 1026x2048 128->3 | 1 | 2048 | 136 | 95 | 1490 (1.74 with the halo re-read) | 2047 / 1539 / 1076 (**1.33**) |
| 36 | 4 | 514x1024 32->128 | 5 | 597 | 363 | 254 | 373 | 595 / 504 / 427 (**1.18**) |
| 44 | 5 | 1026x2048 64->128 | 1 | 2892 | 2905 | 2034 | 1790 | |
| 46 | 6 | 1026x2048 64->256 | 1 | 5282 | 5810 | 4067 | 2980 | |
| 43 | 7 | 258x512 128->512 | 8 | 1309 | 1453 | 1017 | 373 | 1309 / 1270 / 814 (1.03) |
| 37 | 8 | 258x512 64->256 | 5 | 398 | 363 | 254 | 187 | 398 / 369 / 278 (1.08) |
| 33 | 9 | 514x1024 128->3 | 1 | 587 | 34 | 24 | 373 (450 with halo) | 587 / 466 / 356 (**1.26**) |
| 38 | 10 | 130x256 128->512 | 17 | 374 | 363 | 254 | ~95 | 375 / 366 / 315 (1.02) |
| 39 | 11 | 514x1024 64->128 | 1 | 758 | 726 | 508 | 447 | 758 / 715 / 663 (1.06) |
| 45 | 12 | 514x1024 128->256 | 1 | 2757 | 2905 | 2034 | 895 | 1.007 |
| 47 | 13 | 514x1024 128->512 | 1 | 5337 | 5810 | 4067 | 1490 | 1.002 |
| 40 | 14 | 258x512 128->256 | 1 | 745 | 726 | 508 | 223 | 745 / 706 / 584 (1.06) |
| 34 | 15 | 66x128 128->512 | 10 | 139.5 | 91 | 64 | — | 139 / 129 / 118 (1.08) |
| 32 | 16 | 130x256 32->128 | 1 | 70.8 | 23 | 16 | — | 1.19 |
| 31 | 17 | 66x128 32->128 | 1 | 25.3 | 6 | 4 | — | 1.10 |

Call-weighted over the 17: **61.13 ms** (110-core HiFi4 ideal 40.18 ms). *DRAM floor = (input bytes
read once + output bytes incl. the 32-lane padding) / 450 GB/s. Span grids (per-core BRISC/NCRISC kernel
spans, us): `python3 perf_experiments/ring_fill_noc_split/span_grid.py <ops csv> <device log csv>`.
Zones: `perf_experiments/zone_report.py <device log> --run <id>`, timelines
`perf_experiments/ring_fill_noc_split/zone_timeline.py`. The big cases exhaust the ~250-marker budget per
RISC (zone sums truncate silently: compare executions per core with the expected count).

## Measured breakdown at this round's start (us; cumulative peel, everything left of a row stubbed;
`ablation_r19/apply_hooks.py <op_dir>` adds `CONV2D_3X3_ABLATE=fill,weights,store,tilize,bias,matmul`
stubs to any op copy: payload stubbed, CB / trid / semaphore sync kept. Raw: `ablation_r19/ab_results.txt`)

| stubbed | P1 (41) | P3 (35) | P2 (42) | P4 (36) | P7 (43) | P10 (38) | P8 (37) | P15 (34) |
|---|---|---|---|---|---|---|---|---|
| none | 1819 | 2048 | 1385 | 597 | 1309 | 374 | 398 | 139.5 |
| fill | 1763 | **787** | 1310 | 565 | 1287 | 361 | 385 | 138.5 |
| + weights | 1748 | 786 | 1305 | 565 | 1282 | 355 | 382 | 126 |
| + store | 1547 | 785 | 1289 | 393 | 1241 | 342 | 328 | 103 |
| + tilize | 1480 | 660 | 1231 | 377 | 1194 | 332 | 321 | 103 |
| + bias | 1401 | 659 | 1206 | 357 | 1179 | 328 | 312 | 101.5 |
| + matmul (everything) | 394 | 311 | 135 | 108 | 96 | 31 | 28 | 16.5 |
| solo: store | 1616 | 1932 | 1378 | 433 | 1306 | 373 | 346 | 117 |

Readings:
- **Matmul is at its per-core roofline** on every case (removing it saves ~ its 64-cycle/tile-MAC HiFi4 time).
- **Position-dependent NoC imbalance is the largest single theme** (~2.5 ms call-weighted if every core
  finished at the p50 span):
  - **P3 / P9 (C_out fold) are fill-bound**: fill stubbed, P3 2048 -> 787. The fill (NoC0 bank-run reads,
    reader only after the startup window) has a **grid-row gradient**: P3 NCRISC spans y2-y3 ~2040 us, y5
    ~1760, y7 ~1430, y11 ~1100 (rows, not columns). Bytes: P3 reads 538 MB + 21% halo re-read and writes
    134 MB (32-lane padded) = ~784 MB in 2.05 ms = 383 GB/s; P9 +43% halo.
  - **P1 / P4 (C_in 32) are store-bound**: store stubbed, P4 565 -> 393, P1 1748 -> 1547. The writer's
    NoC1 DRAM writes have an **x and y gradient**: P4 BRISC spans x1 ~440-480 us, x14 ~525-595 us, y2
    fastest, y9 slowest, y11 fast (P1 the same shape: 1608 -> 1819). Perf 18 measured that evenly spread
    stores starve the far cores (grid x 12-15, y 4-10) on NoC1; bursty stores (the whole M-block's
    output drains in its last K-block) leave gaps in which they catch up.
- **Startup bubble on the ring**: P2's first M-block takes ~110 us against ~26 us steady (compute_tilize
  zones at block 0's kh = 1 and kh = 2: 37 us each, waiting for the first window: 2 image rows per core,
  all 110 cores at once, DRAM-bound). Fill-stub cost: P2 75, P1 56, P4 32, P7 22, P10 13, P8 13 us.
- **The store** (after fill stubbed): P1 -201, P4 -172, P8 -54, P7 -41, P15 -23, P2 -16 us.
- Tilize payload: P3 125, P1 67, P2 58, P7 47 us. Bias pass: P1 79, P2 25, P4 20, P7 15 us. Weights
  (resident): ~0-7 us, P15 (streamed) 12.5 us.
- The everything-stubbed floor (P1 394, P3 311) is NOT on the critical path of the real op (Perf 18
  tapdirect_unit_cost, lessons file): do not spend effort on it without a full-op zone timeline.

## Prior rounds (do not re-float their nulls unless you name what is different)
Read `ttnn/ttnn/operations/conv2d_3x3/changelog.md` sections "Perf 16", "Perf 17", "Perf 18" (FLUX.2
rounds) and the 2026-11-20 / 2026-11-21 entries at the end of `.claude/references/op-gen-lessons-learned.md`.
Notably: Perf 16 store_offload (ALL stores from the reader on NoC0: +4.5% to +14.7%, back when the fill
was ~6.3 M per-page NoC0 reads; DRAM writes on NoC0 cost ~2x NoC1 then); Perf 17 brf_noc_split (dynamic
front/back chunk claims over both NoCs flattened P3's row imbalance on the old per-block fill: row max
5186 -> 4105); Perf 18 ring_fill_noc_split menu ("writer in every epoch": P3 flat, P1 +13%, measured
BEFORE the fold made P3 fill-bound); Perf 18 kblock_store_decouple (deep output CB: P1 +66% because bm
shrank; free-L1-only deeper CB flat); Perf 18 whole_k_rowstream (WIN with a tuned store pacer, not
graduated); Perf 18 tapdirect_unit_cost (NULL).

## What to return
verdict (WIN / NULL / REGRESSION / BLOCKED); base vs candidate device ns on your primary cases and
on a sweep of every focus case your change reaches (all 17 if it reaches them); per-core span max/p50
where your idea is about balance; bit-identity / PCC; L1 impact (does the blocking solve change?); the
domain (where it applies; list only exceptions that are incorrect, inexpressible or measured slower —
flat and untested are not exceptions); any raw-LLK / raw-NoC helper bypass (helper name, capability vs
ergonomics, what was missing / hard); an `integration.patch` against the real op (`diff -ru` of
op_base -> op_copy) plus a short README in your dir; a menu of variants with their ns and precision if
you tried more than one. Keep nothing over 500 KB in your dir that should be committed (gzip or delete
the device-profile CSVs; the commit hook rejects large files).

## Code map (line numbers at this round's start)
- Host: `conv2d_3x3.py` (call path, `select_blocking` ~L706); `conv2d_3x3_program_descriptor.py`:
  `RM_SPLIT_MIN_WINDOW_BYTES` L140, `NSPLIT_GROUPS` L408, `l1_cb_budget_bytes` L566, `Conv3x3Blocking`
  L709, `_tap_direct*` L1101-1140, `weights_resident_capable` L1387, `rm_ring_capable` / `rm_ring_plan`
  L1402-1494, `cout_fold_capable` L1494, `kh_reuse_capable` / `kh_reuse_blocking` L1544-1640,
  `select_blocking` L1642 (N split ~L1782), `create_program_descriptor` L2984 (`rm_split` ~L3784, kernel
  defines ~L4240-4390).
- Reader (NCRISC/NoC0) `kernels/conv2d_3x3_reader.cpp`: ring constants ~L165, ring fill / K-blocked
  tap-direct loop from ~L1364 (`RM_RING_PX`), kh reuse stream ~L786. `kernels/conv2d_3x3_rm_ring.hpp`:
  `RmSplitWords` L59, `RmBankRing` L175 (`issue_claimed` L328, `issue` L391, `need` L500), the writer's
  split part L538-700 (`issue` L601, `process_bank` L623, `serve` L660).
  `kernels/conv2d_3x3_gather_dataflow.hpp` `TapDirectFifo`.
- Compute `kernels/conv2d_3x3_compute.cpp`: `fifo_tilize`, kh reuse ~L880-1010, walk dispatch ~L2370;
  `kernels/conv2d_3x3_kblock_walk.hpp` (raw-LLK K-block walk, partials, interleaved bias pass);
  `kernels/conv2d_3x3_cout_fold.hpp` (fold walk L195, final pass ~L254).
- Writer (BRISC/NoC1) `kernels/conv2d_3x3_writer.cpp`: fold store ~L2494-2565, K-blocked store_block
  ~L2727-2760 (per row group: `noc_async_write_page` per tile, flushed, pop; RM_SPLIT serve in its wait).

## Harness traps
- `run_safe_pytest.sh` precompiles by running each test body once in a collect pass before the real
  pass: guard expensive host work and result writes on that pass (see `_in_collect_pass()` in
  `test_flux_conv3x3_compare.py`); `cmp_windows.py` already drops its empty signpost windows.
- The big cases' device-profile CSVs are 0.3-1 GB; delete or gzip them in your `runs/` dir.

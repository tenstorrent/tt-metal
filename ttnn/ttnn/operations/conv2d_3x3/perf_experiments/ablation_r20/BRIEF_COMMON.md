# Perf 20 — shared context for every perf-part-optimizer (read this first)

## Ground rules
- Repo: `/localdev/pmilojevic/workspaces/opgen` (a git worktree). Before ANY device run:
  `export TT_METAL_HOME=$PWD PYTHONPATH=$PWD; source python_env/bin/activate`
  (otherwise the run silently tests another checkout).
- Device runs only through `scripts/run_safe_pytest.sh` / `scripts/tt-probe.sh`, in the foreground.
  Never `run_in_background`, never `python3` with device code (a bare `import ttnn` also opens the device
  outside the lock; host-only tools must stub ttnn, see `ablation_r20/plan.py`). Device runs serialize on a
  flock with the other optimizers; that is expected. Keep each run small (only the cases you need), and keep
  every single tool call well under 10 minutes: a call that hits the tool time limit kills a `--profile` run
  before the profiler CSV is written.
- Do NOT edit the real op (`ttnn/ttnn/operations/conv2d_3x3/`). Copy it into your own dir
  `ttnn/ttnn/operations/conv2d_3x3/perf_experiments/<your_slug>/` as `op_base/` (untouched baseline) and
  `op_copy/` (your candidate), the way `perf_experiments/graduation_r19/` and `whole_k_rowstream_v2/` did
  (a pytest that imports both, signposts each call `cmp|<case>|<variant>|<i>` with `tracy.signpost`, a
  `run.sh` that runs `--profile --run-all` and parses the ops CSV with
  `perf_experiments/asym_padding/cmp_windows.py`; kernels resolve via the copy's `KERNEL_DIR`).
  `graduation_r19/test_final.py` is the most recent such harness (bit-identity by sha1, PCC, alternating
  variants): start from it. Copy the WORKING TREE, not HEAD: it carries uncommitted user changes in
  `conv2d_3x3.py` / `conv2d_3x3_program_descriptor.py` / `__init__.py` (the `l1_small_bytes` L1-budget term,
  `prepare_conv2d_weights_host`) that the measurements depend on. The device is opened with
  `l1_small_size=65536` (FLUX.2's VAE does that), which shrinks the CB budget to 1,334,368 B.
- Profiler zone hashes are 16-bit over "name,path,line": two op copies compiled in one session collide
  across copies too. Run `python3 -I perf_experiments/ablation_r19/zone_hash_check.py op_base op_copy`
  before each profile run; fix a collision by re-wrapping a comment, not by renaming a zone.
- Do not commit. Do not spawn other optimizer agents. Under `ttnn/ttnn/` import torch only inside
  functions (the pre-commit hook `check-torch-imports-in-ttnn` walks untracked files too).
- Precision contract is fixed: bf16 activations + weights, HiFi4, `fp32_dest_acc_en=True`,
  `math_approx_mode=False`. Never change those for speed. Preferred bar: output bit-identical to `op_base`
  (sha1 of the output bytes). If not bit-identical, report PCC vs torch (currently 0.9999985 on every focus
  case) and how many outputs differ and by how many bf16 ulp.
- Correctness is the only pass/fail. Perf is measured (DEVICE KERNEL DURATION), never asserted. Device kernel
  time has no warm-up transient; the harness already times 3 calls per case (median).
- Guards owned by the coordinator: Panoptic-DeepLab (L1-sharded, LoFi bf8_b, 22 cases) and the single-device
  FLUX.2 TILE-input convs. State explicitly whether your change can reach sharded or TILE-input programs. If
  it reaches TILE input, measure 2-3 TILE cases too (`FLUX_CMP_BH_LAYOUT=TILE` on the bh2x4 idx, or the
  `1x1` group idx 0-30). If it reaches sharded programs, run a few Panoptic compare cases
  (`CONV_CMP_CASES=...` with `tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_panoptic_conv3x3_compare.py`).
- Code comments: present-tense intent only (why), no history / no narration of what you tried. Kernel zones
  (`MaybeDeviceZoneScope` from `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`) are permanent: keep the
  existing ones and add zones to any new stage you create (mind the ~250-marker budget per RISC: no per-tile
  or per-chunk zones; the big cases already hit it, so zone sums truncate silently — compare executions per
  core with the expected count).

## The focus: 17 FLUX.2 bh2x4 per-device decoder convs
bf16, HiFi4, fp32 DEST, bias, padding (0, 0, 1, 1) (= pad_top 0, pad_bottom 0, pad_left 1, pad_right 1:
the input carries the halo rows, H_out = H - 2, W_out = W), DRAM-interleaved ROW_MAJOR input, output
DRAM-interleaved TILE (C_out padded to 32 lanes), Blackhole p150b, 110 cores (11x10), 8 DRAM banks, 1350 MHz.
Whole-op harness (the real op):
`FLUX_CMP_CASES=<idx,...> FLUX_CMP_SKIP_TTNN=1 scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_flux_conv3x3_compare.py`
then `python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py <PROFILER CSV>`.
See that test for how the op is called (weights/bias prep, ComputeConfigDescriptor).

Paths (Perf 16-19 built them; read `changelog.md` "Perf 16" .. "Perf 19"):
- Every case except idx 31 / 32 / 34 runs the interleaved K-blocked **tap-direct** walk on the **ROW_MAJOR ring
  slab** (`kernels/conv2d_3x3_rm_ring.hpp`, reader `RM_RING_PX`: each core reads its contiguous pixel range
  once, as 8-bank bank runs into staging chunks, de-interleaved one pixel per self-aimed NoC read into the ring)
  with **resident weights** (`WEIGHTS_RESIDENT`) and the **bias pass inside the last K-block**
  (`kernels/conv2d_3x3_kblock_walk.hpp`). K-block = one tap (kh, kw), bk = C_in/32 tiles, 9 K-blocks per
  M-block, fp32 partials by packer L1 accumulation, output through `cb_output_tiles` (2 row groups) to the
  writer (BRISC), which stores whole 2 KB tiles to DRAM.
- **kh tile reuse** (`KH_REUSE`, compute `namespace kh_reuse` ~L903): C_in >= 64 where the tile ring fits
  (P2, P8, P10, P11): each input tile-row tilized once into a tile ring (ring row = [kw][c] = 3 * tw tiles),
  K-block (kh, kw) reads ring rows at an offset; band-first stream order (`conv2d_3x3_khr_band_order.hpp`,
  reader `RmBandRing`).
- **C_out fold** (`CONV2D_3X3_COUT_FOLD`, `kernels/conv2d_3x3_cout_fold.hpp`): C_out <= 4 (P3, P9).
- **Whole-stream fill split** (`RM_SPLIT`, rm_ring.hpp `RmSplitWords` L67, `RmBankRing` L194, writer `serve()`
  ~L1061): reader (NoC0, front claims) and writer (NoC1, back claims) share the ring fill for the whole
  stream; the writer claims only while the reader is blocked. Off on kh reuse and on `output_stream_bound`.
- **Routed output store** (`STORE_ROUTE`, `kernels/conv2d_3x3_store_split.hpp`): each output tile on the NoC
  with fewer hops to its DRAM bank (BRISC writes NoC0 on command buffer 2); only where
  `output_stream_bound(g, b, fidelity)` (descriptor ~L2538) holds = C_in 32 at bf16 HiFi4 (P1, P4).
- **N split** (P7, P10, P13: 2 N groups, each [K, N/2] resident); one-M-block N split (P15).
- idx 31 / 32 / 34 (one M-block per core): per-block slab, `BANK_RUN_FILL`, streamed weights.

Blocking at this round's start (`python3 -I perf_experiments/ablation_r20/plan.py [op dir]`, host-only,
ttnn stubbed): core_m = the busiest core's output tile-rows, mb = M-blocks per core.

| idx | prio | case (H x W, C_in->C_out) | calls | now us | 110-core ideal us | blocking |
|---|---|---|---|---|---|---|
| 41 | 1 | 1026x2048 32->128 | 5 | 1746.9 | 1017 | bm 46 bk 1 bn 4 mb 13 core_m 596, routed store |
| 42 | 2 | 514x1024 64->256 | 6 | 1268.1 | 1017 | bm 3 bk 2 bn 8 mb 50 core_m 149, kh reuse (ring 64 rows) |
| 35 | 3 | 1026x2048 128->3 | 1 | 1837.9 | 95 | C_out fold, bm 8 mb 75 |
| 36 | 4 | 514x1024 32->128 | 5 | 545.4 | 254 | bm 50 bk 1 bn 4 mb 3 core_m 149, routed store |
| 44 | 5 | 1026x2048 64->128 | 1 | 2893.4 | 2034 | bm 24 bk 2 bn 4 mb 25 core_m 596 |
| 46 | 6 | 1026x2048 64->256 | 1 | 5287.3 | 4067 | bm 10 bk 2 bn 8 mb 60 core_m 596 |
| 43 | 7 | 258x512 128->512 | 8 | 1298.2 | 1017 | bm 7 bk 4 bn 8 mb 11 core_m 75, N split 2 |
| 37 | 8 | 258x512 64->256 | 5 | 396.8 | 254 | bm 10 bk 2 bn 8 mb 4 core_m 38, kh reuse (32) |
| 33 | 9 | 514x1024 128->3 | 1 | 520.0 | 24 | C_out fold, bm 30 mb 5 |
| 38 | 10 | 130x256 128->512 | 17 | 358.4 | 254 | bm 3 bk 4 bn 8 mb 7 core_m 19, N split 2, kh reuse (16) |
| 39 | 11 | 514x1024 64->128 | 1 | 720.9 | 508 | bm 12 bk 2 bn 4 mb 13 core_m 149, kh reuse (64) |
| 45 | 12 | 514x1024 128->256 | 1 | 2746.6 | 2034 | bm 2 bk 4 bn 8 mb 75 core_m 149 |
| 47 | 13 | 514x1024 128->512 | 1 | 5334.2 | 4067 | bm 2 bk 4 bn 8 mb 149 core_m 298, N split 2 |
| 40 | 14 | 258x512 128->256 | 1 | 734.7 | 508 | bm 7 bk 4 bn 8 mb 6 core_m 38 |
| 34 | 15 | 66x128 128->512 | 10 | 119.4 | 64 | one M-block, bm 5 bk 12 nkb 3, streamed weights, N split 2 |
| 32 | 16 | 130x256 32->128 | 1 | 70.3 | 16 | one M-block, bm 10 bk 3 nkb 3 |
| 31 | 17 | 66x128 32->128 | 1 | 25.8 | 4 | one M-block, bm 3 bk 3 nkb 3 |

Call-weighted over the 17 (calls x us): **58.90 ms** (110-core HiFi4 ideal 40.18 ms).

## Measured breakdown at this round's start (device kernel us; cumulative peel, everything left of a row
stubbed; `ablation_r20/apply_hooks.py <op_dir>` adds `CONV2D_3X3_ABLATE=fill,weights,store,tilize,bias,matmul`
stubs to any op copy (payload stubbed, CB / trid / semaphore sync kept); `ablation_r20/ab.sh`; raw:
`ablation_r20/ab_results.txt`)

| stubbed | P1 (41) | P4 (36) | P2 (42) | P7 (43) | P10 (38) | P13 (47) | P15 (34) | P3 (35) |
|---|---|---|---|---|---|---|---|---|
| none | 1747 | 545 | 1268 | 1298 | 358.4 | 5334 | 119.4 | 1838 |
| fill | 1661 | 506 | 1267 | 1287 | 355.4 | 5243 | 112.9 | 788 |
| + weights | 1658 | 505 | 1263 | 1283 | 349.4 | 5239 | 104.3 | 788 |
| + store | 1547 | 393 | 1245 | 1242 | 335.4 | 5224 | 84.8 | 785 |
| + tilize | 1463 | 377 | 1206 | 1194 | 326.7 | 5022 | 83.5 | 660 |
| + bias | 1401 | 357 | 1182 | 1179 | 323.6 | 4969 | 82.4 | 659 |
| + matmul (everything) | 400 | 109 | 147 | 107 | 33.2 | 853 | 11.9 | 346 |
| solo: store | 1615 | 433 | 1249 | 1283 | 344.1 | 5332 | 113.7 | 1351 |

Readings:
- **The ragged last M-block computes its nominal bm rows.** The K-blocked walk's row-group count is the
  compile-time `bm / sb_h` for every M-block (`kblock_walk.hpp` L187, compute L1318), so a core whose rows are
  not a multiple of bm tilizes and multiplies padding rows: P10 21 rows for 19 (+10.5%), P14 42 for 38 (+10.5%),
  P8 40 for 38, P11 156 for 149, P7 77 for 75. P10's matmul-only time (291 us) is exactly its 21-row ideal
  (287 us), not its 19-row one (259 us).
- **Matmul is otherwise at its per-core roofline** (P1 1001 vs 1017, P2 1035 vs 1017, P13 4116 vs 4067).
- **The per-K-block / per-unit floor on small-bm walks**: everything stubbed leaves P13 853 us (bm 2, 149
  M-blocks x 9 K-blocks), P2 147 us (bm 3, 450 K-blocks), P1 400 us. On P1 it is off the critical path
  (Perf 18 tapdirect_unit_cost); on P13 / P12 / P6 / P5 the whole-K row stream (Perf 19 whole_k_rowstream_v2)
  measured -7% to -10%, consistent with it being on the path there.
- **Store serialization on C_in 32 (P1, P4; 10 calls).** Output only exists after an M-block's last K-block;
  the 2-row-group output CB makes compute's last K-block wait on the store, so 8/9 of compute never overlaps
  any store. P4: compute_last_kblock 39 us per M-block p50 (59 max) vs ~12 us for the other K-blocks; BRISC
  span max / p50 545 / 485. Solo store stub: P1 -132, P4 -113 us. DRAM floor at 450 GB/s: P1 1490, P4 373 us.
- **conv_out (P3, P9)** is at its DRAM roofline (P3 784 MB in 1.84 ms = 427 GB/s: input + 21% halo re-read +
  the 32-lane-padded output). Solo store stub: P3 -487 us.
- Tilize payload: P13 202, P3 125, P1 84, P7 47 (x8 calls), P2 39 us. It runs near the fast-tilize rate
  (~24 cycles per tile on P7); only tilizing fewer tiles (kh reuse) reduces it.
- P7 per core: block 0 160 us, blocks 1-10 112 us each (reader_band_wait 69 us in block 0).

## Prior rounds (do not re-float their nulls unless you name what is different)
Read `ttnn/ttnn/operations/conv2d_3x3/changelog.md` sections "Perf 16" .. "Perf 19" (FLUX.2 rounds) and the
2026-11-20 / 2026-11-21 entries at the end of `.claude/references/op-gen-lessons-learned.md`. Notably:
Perf 18 kblock_store_decouple (deep output CB: P1 +66% because bm shrank, P4 +21%; cause: NoC1 write bandwidth
by core position, the far columns starved), Perf 18 whole_k_rowstream (WIN only with a tuned store pacer),
Perf 19 whole_k_rowstream_v2 (WIN on C_in >= 64, not graduated: incompatible with the whole-stream fill split
as built, loses to band-first kh reuse on P2 / P11, P14 +2.0%), Perf 19 store_noc_split (the routed store,
graduated on C_in 32), Perf 18 tapdirect_unit_cost (NULL).

## What to return
verdict (WIN / NULL / REGRESSION / BLOCKED); base vs candidate device ns on your primary cases and on a sweep
of every focus case your change reaches (all 17 if it reaches them); per-core span max/p50 where your idea is
about balance; bit-identity / PCC; L1 impact (does the blocking solve change?); the domain (where it applies;
list only exceptions that are incorrect, inexpressible or measured slower — flat and untested are not
exceptions); any raw-LLK / raw-NoC helper bypass (helper name, capability vs ergonomics, what was missing /
hard); an `integration.patch` against the real op (`diff -ru` of op_base -> op_copy, kernels + host) plus a short
README in your dir; a menu of variants with their ns and precision if you tried more than one. Keep nothing
over 500 KB in your dir that should be committed (gzip or delete the device-profile CSVs; the commit hook
rejects large files).

## Harness traps
- `run_safe_pytest.sh` precompiles by running each test body once in a collect pass before the real pass:
  guard expensive host work and result writes on that pass (see `_in_collect_pass()` in
  `test_flux_conv3x3_compare.py`); `cmp_windows.py` already drops its empty signpost windows.
- The big cases' device-profile CSVs are 0.3-1 GB; delete or gzip them in your `runs/` dir.
- Per-core zone tools: `perf_experiments/zone_report.py <device log> --run <id>`,
  `perf_experiments/ring_fill_noc_split/zone_timeline.py`, `span_grid.py`.

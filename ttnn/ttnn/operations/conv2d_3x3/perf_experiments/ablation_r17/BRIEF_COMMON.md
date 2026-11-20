# Perf 17 — shared context for every perf-part-optimizer (read this first)

## Ground rules
- Repo: `/localdev/pmilojevic/workspaces/opgen` (a git worktree). Before ANY device run:
  `export TT_METAL_HOME=$PWD PYTHONPATH=$PWD; source python_env/bin/activate`
  (otherwise the run silently tests another checkout).
- Device runs only through `scripts/run_safe_pytest.sh` / `scripts/tt-probe.sh`, in the foreground.
  Never `run_in_background`, never `python3` with device code. Device runs serialize on a flock with
  the other 5 optimizers; that is expected.
- Do NOT edit the real op (`ttnn/ttnn/operations/conv2d_3x3/`). Copy it into your own dir
  `ttnn/ttnn/operations/conv2d_3x3/perf_experiments/<your_slug>/` as `op_base/` (untouched baseline)
  and `op_copy/` (your candidate), the way `perf_experiments/weight_stream_decouple/` and
  `perf_experiments/rm_ring_slab/` did (a pytest that imports both, signposts each call
  `cmp|<case>|<variant>|<i>` with `tracy.signpost`, `run.sh` that runs `--profile --run-all` and parses
  the ops CSV with `perf_experiments/asym_padding/cmp_windows.py`; kernels resolve via the copy's
  `KERNEL_DIR`). Copy the WORKING TREE, not HEAD: it carries uncommitted user changes in
  `conv2d_3x3.py` / `conv2d_3x3_program_descriptor.py` (the `l1_small_bytes` L1-budget term) that
  the measurements depend on.
- Do not commit. Do not spawn other optimizer agents. Under `ttnn/ttnn/` import torch only inside
  functions (the pre-commit hook `check-torch-imports-in-ttnn` walks untracked files too).
- Precision contract is fixed: bf16 activations + weights, HiFi4, `fp32_dest_acc_en=True`,
  `math_approx_mode=False`. Never change those for speed. Preferred bar: output bit-identical to
  `op_base` (sha1 of the output bytes). If not bit-identical, report PCC vs torch (current
  0.9999985 on every focus case) and how many outputs differ and by how many bf16 ulp.
- Correctness is the only pass/fail. Perf is measured (DEVICE KERNEL DURATION), never asserted.
- Guards owned by the coordinator: Panoptic-DeepLab (L1-sharded, LoFi bf8_b) and the single-device
  FLUX.2 TILE-input convs. State explicitly whether your change can reach sharded or TILE-input
  programs. If it reaches TILE input, measure 2-3 TILE cases too (`test_flux_conv3x3_compare.py`
  with `FLUX_CMP_BH_LAYOUT=TILE` on the bh2x4 idx, or the `1x1` group idx 0-30).

## The focus: 17 FLUX.2 bh2x4 per-device decoder convs
bf16, HiFi4, fp32 DEST, bias, padding (0, 0, 1, 1), DRAM-interleaved ROW_MAJOR input, 110 cores
(11x10), Blackhole p150b, device opened with `l1_small_size=65536`. Whole-op harness (the real op):
`FLUX_CMP_CASES=<idx,...> FLUX_CMP_SKIP_TTNN=1 scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_flux_conv3x3_compare.py`
then `python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py <PROFILER CSV>`.
See that test for how the op is called (weights/bias prep, ComputeConfigDescriptor).

| idx | case (H x W, C_in->C_out) | calls | now us | target us | blocking now (bm, bk, K-blocks, M-blocks/core, resident) |
|---|---|---|---|---|---|
| 41 | P1 1026x2048 32->128 | 5 | 2606 | 1453 | 46, 1, 9, 13, resident |
| 35 | P3 1026x2048 128->3 (conv_out) | 1 | 5212 | 136 | 32, 4, 9, 19, resident |
| 43 | P7 258x512 128->512 | 8 | 1681 | 1453 | 8, 4, 9, 5, STREAMED (weights 1.18 MB) |
| 36 | P4 514x1024 32->128 | 5 | 727 | 363 | 50, 1, 9, 3, resident |
| 38 | P10 130x256 128->512 | 17 | 446 | 363 | 5, 4, 9, 2, STREAMED, 103 cores |
| 33 | P9 514x1024 128->3 (conv_out) | 1 | 1409 | 34 | 30, 4, 9, 5, resident |
| 42 | P2 514x1024 64->256 | 6 | 1618 | 1453 | 19, 2, 9, 8, resident |
| 44 | P5 1026x2048 64->128 | 1 | 3927 | 2905 | 34, 2, 9, 18, resident |
| 37 | P8 258x512 64->256 | 5 | 513 | 363 | 19, 2, 9, 2, resident, 108 cores |
| 47 | P13 514x1024 128->512 | 1 | 6390 | 5810 | 9, 4, 9, 17, STREAMED |
| 34 | P15 66x128 128->512 | 10 | 140 | 91 | 3, 12, 3, 1, STREAMED, 86 cores |
| 46 | P6 1026x2048 64->256 | 1 | 5882 | 5810 | 19, 2, 9, 32, resident |
| 39 | P11 514x1024 64->128 | 1 | 1077 | 726 | 30, 2, 9, 5, resident |
| 45 | P12 514x1024 128->256 | 1 | 2775 | 2905 | 10, 4, 9, 15, resident |
| 40 | P14 258x512 128->256 | 1 | 836 | 726 | 10, 4, 9, 4, resident |
| 32 | P16 130x256 32->128 | 1 | 74 | 23 | 10, 3, 3, 1 |
| 31 | P17 66x128 32->128 | 1 | 25 | 6 | 3, 3, 3, 1 |

## Measured breakdown this round (us; cumulative peel, everything left of a column stubbed)
Hooks: `perf_experiments/ablation_r17/apply_hooks.py <op_dir>` adds `CONV2D_3X3_ABLATE=fill,weights,store,tilize,bias,matmul`
stubs to any op copy (payload stubbed, CB/trid sync kept). Raw: `ablation_r17/ab_results.txt`.

| stubbed | P1 | P3 | P4 | P9 | P10 | P7 | P2 | P5 |
|---|---|---|---|---|---|---|---|---|
| none | 2606 | 5212 | 727 | 1409 | 446 | 1681 | 1618 | 3927 |
| fill | 1805 | 1726 | 576 | 449 | 430 | 1643 | 1420 | 2925 |
| +weights | 1801 | 1725 | 571 | 448 | 376 | 1334 | 1416 | 2919 |
| +store | 1556 | 1683 | 395 | 420 | 323 | 1235 | 1272 | 2742 |
| +tilize | 1467 | 1376 | 373 | 345 | 317 | 1210 | 1215 | 2572 |
| +bias | 1385 | 1349 | 354 | 339 | 313 | 1193 | 1181 | 2500 |
| +matmul (all) | 292 | 300 | 76 | 79 | 25 | 46 | 88 | 305 |
| only fill left | 1439 | 4445 | 317 | 1125 | 82 | 303 | 627 | 2426 |
| solo store | 2504 | 5043 | 596 | 1383 | 363 | 1361 | 1491 | 3764 |

Readings:
- The ROW_MAJOR bank-run fill (reader `#ifdef BANK_RUN_FILL`, ~`conv2d_3x3_reader.cpp:1186`) is
  SERIALIZED with compute: the reader fills all 3 kh-bands of M-block j+1 only after compute popped
  every tap-direct entry of block j (`conv2d_3x3_reader.cpp:~1868`), because the bank-run staging
  aliases cb_im2col_rm and the slab is single-buffered. P1: full 2606 ~= floor 292 + compute ~1260 +
  fill ~1050.
- Bands are disjoint whenever bm*32 < W (P1, P3, P5, P6, P2, P9...): each input pixel is read ~3x per core.
  DRAM bytes now vs 1x-read: P1 940 vs 671 MB, P3 1748 vs 672 MB, P5 1344 vs 806 MB, P6 1881 vs 1343 MB.
- conv_out (P3, P9, 256 B pixels) is fill/DRAM-bound with a strong per-grid-row imbalance: NCRISC
  kernel span 5.2 ms on grid rows y=2-3, 3.1 ms on y=10-11 (P9: 1.39 vs 0.77 ms). P1/P5 are balanced.
- P1's fill: ~57k pixels de-interleaved per core at ~24 NCRISC cycles each (~1.0 ms of issue) — about
  equal to its DRAM time; both bound it.
- Non-resident weight stream (128->512): P7 -309 us, P10 -54 us when stubbed.
- Matmul runs at 93-107% of the 64-cycle/tile-MAC HiFi4 ideal: no compute headroom in the matmul.
- Zones (`MaybeDeviceZoneScope`) are permanent; keep and extend them in any new code path.
  `perf_experiments/zone_report.py <profile_log_device.csv>` summarizes them. Profiles of the big
  cases (P1, P3) exhaust the ~250-marker budget per RISC — use smaller cases for zone reads.

## What to return
verdict (WIN / NULL / REGRESSION / BLOCKED); base vs candidate device ns on your primary cases and
on a sweep of every focus case your change reaches; bit-identity / PCC; L1 impact (does the blocking
solve change?); the domain (where it applies; list only exceptions that are incorrect, inexpressible
or measured slower — flat and untested are not exceptions); any raw-LLK / raw-NoC helper bypass
(helper name, capability vs ergonomics, what was missing); an `integration.patch` against the real op
(`diff -ru` of op_base -> op_copy) plus a short README in your dir; a menu of variants with their ns if
you tried more than one.

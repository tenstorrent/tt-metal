# rms_norm Perf 2 — shared brief for perf-part-optimizer subagents

Repo root: `/localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal`
(`source python_env/bin/activate` first). Board: Blackhole p150b, 13×10 = 130 Tensix cores, AICLK 1350 MHz.
This is round 2. Round 1 is in `ttnn/ttnn/operations/rms_norm/changelog.md` ("## Perf 1"); its brief
(`perf_experiments/BRIEF_perf1.md`) describes the round-1 op — read the changelog for what changed since.

## The op (frozen; correctness done; HEAD = commit ebe9c9d1ebc + Perf 2 ablation hooks)

- `ttnn/ttnn/operations/rms_norm/rms_norm.py` (entry + SUPPORTED), `rms_norm_program_descriptor.py` (blocking:
  `select_blocking` interleaved / `select_sharded_blocking` sharded; CB plan `cb_plan`; kernels; RT args),
  `kernels/rms_norm_{reader,compute,writer}.cpp`, `kernels/rms_norm_common.hpp`.
- Design + Blocking Model: `op_design.md`; L1 accounting: `l1_ledger.md`.
- Stages per block: reader (TILE interleaved: DRAM → cb_x; sharded TILE: cb_x IS the shard; RM interleaved: DRAM
  sticks → cb_x_rm; RM sharded: local stick copies shard → cb_x_rm staging) → compute: [RM tilize] → pass 1
  `sum_squares_block` (raw x·x DEST-accumulated per tile-row → cb_sumsq) → `collapse_sumsq_block` (FPU reduce_tile ROW
  with a 1.0 scaler; k == 1: + column-0 sfpi finalize) → k > 1: writer exchanges column-0 partials (symmetric all-gather
  for k ≤ 8 interleaved; owner combine for k > 8 interleaved and every sharded k > 1: row r owned by member r % k, faces
  0/2 shipped, owner sums k partials + finalize, returns inv_rms — one `ttnn.Mcast` for single-owner blocks, unicast for
  multi-owner blocks; TILE sharded may run the "combine pipeline" over row sub-blocks) → pass 2 `normalize_window`
  (x · inv_rms bcast Col → raw `MulGammaRowBcast` · gamma bcast Row) → cb_out → writer (TILE interleaved DRAM writes;
  sharded TILE: cb_out IS the output shard; RM: untilize → sticks; RM sharded: local stick write-back).
- Gamma: cb_gamma pages are faces 0/1 of each gamma tile (row-broadcast operand, never expanded). Loaded by the writer
  (multicast for interleaved / sharded k = 1). Fallback when the resident slice does not fit L1: STREAMED per block
  in G-tile chunks (`RMS_GAMMA_CHUNK`), re-read from DRAM. An RM gamma is ONE DRAM page (every reader hits one bank).
- Stage zones (`MaybeDeviceZoneScope`, `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`) are PERMANENT — keep them
  in any variant you author and extend them to new paths. Ablation defines (probe-only, set via `PERF_ABLATE`):
  `RMS_ABLATE_EMPTY`, `RMS_ABLATE_DM` (interleaved TILE x/out payload AND, since Perf 2, the RM-sharded stick staging +
  write-back payload), `CKL_ELTWISE_CHAIN_SKIP_COMPUTE`.

## Precision contract (fixed — never a lever)

`fp32_dest_acc_en`, `math_fidelity`, `math_approx_mode`, dtypes are the user's. Change HOW the same math is computed.
Golden tolerances: bf16 PCC ≥ 0.995 (perf cells: soft pcc_threshold 0.9995), fp32 ≥ 0.999, bf8b ≥ 0.99. Report
per-option PCC / max-abs error vs torch AND vs the op's current output on the cells you measure.

## Measuring

- Probe (cells built exactly like the golden harness): `tests/ttnn/unit_tests/operations/rms_norm/perf_p2_rms_norm.py`.
  Select with `P2_CELLS=a,b,...`; ablations with `P2_ABLATE=` / `P2_NOGAMMA=1`. **CSV rows come out in the CELLS dict
  order of that file, not your P2_CELLS order.** Copy it into your dir and add cells if you need more.
- Device time: `scripts/run_safe_pytest.sh --profile <test>` → prints `SAFE_PYTEST: PROFILER CSV: <path>`; column
  `DEVICE KERNEL DURATION [ns]` of the `GenericOpDeviceOperation` rows. Zones: `generated/profiler/.logs/profile_log_device.csv`;
  `perf_experiments/tools/zones.py <labels> <ops.csv> <profile_log_device.csv>` summarizes per zone (max / p50 over
  Tensix cores, per RISC-V).
- ONE fresh run per variant (no trial loops); a difference within ~2–3 % is noise. Only cells > 32 tiles count.
- Foreground device runs only (`run_safe_pytest.sh` / `tt-probe.sh`), never `python3` with device code directly.
- Bench shape: clone the op (descriptor + kernels) into YOUR dir `ttnn/ttnn/operations/rms_norm/perf_experiments/<idea_slug>/`,
  apply your change there, and A/B the clone against the unmodified op on the same cells in the same pytest session.
  Never edit the real op files (other agents measure them as their baseline). Do not commit.
- Several agents share the device and `generated/profiler/`. In the SAME bash command as your profiled run, copy the CSV
  and `profile_log_device.csv` into your dir:

      out=$(scripts/run_safe_pytest.sh --profile <your_test.py> 2>&1); csv=$(echo "$out" | sed -n 's/^SAFE_PYTEST: PROFILER CSV: //p'); \
        cp "$csv" $MYDIR/ops_run1.csv; cp generated/profiler/.logs/profile_log_device.csv $MYDIR/dev_run1.csv

- Correctness: your candidate must pass the golden cells it touches. A quick correctness sweep: run the matching subset
  of `eval/golden_tests/rms_norm/test_golden.py` against your clone (monkeypatch the op's descriptor module, or a
  small test of your own that calls `run_rms_norm` from `eval/golden_tests/rms_norm/helpers.py` with your clone).

## Baseline scoreboard (HEAD, profiler DEVICE KERNEL DURATION, ns; empty-program floor 0.29–0.48 µs)

| cell id (probe) | config | ns | breakdown (cumulative ablation: noDM / +chains stubbed) |
|---|---|---|---|
| int_prefill_7168 | (1,1,8192,7168) bf16 TILE interleaved, HiFi2, fp32 DEST off | 598 004 | DRAM: noDM 82.6 µs; rd_read max 517 / p50 203 µs (2.5× Tensix-core imbalance); golden R5 snapshot measured 477.7 µs |
| int_prefill_5120 | (1,1,8192,5120) same | 421 456 | DRAM: noDM 59.4 µs; R5 snapshot 333.8 µs |
| int_prefill_1024 | (1,1,8192,1024) | 82 284 | DRAM; rd_read max 69 / p50 32 µs |
| wid_tile_128x4096 | (1,1,128,4096) WIDTH auto: 128 Tensix cores × [128,32], K = 128, b = 4 tile-rows; HiFi4 fp32 DEST | 26 621 | exchange: chains stubbed 25.7 µs; wr_wait_inv 23.2, wr_xchg_send 12.1 (max), owners' cmp_wait_gathered 12.4 |
| wid_tile_224x3072 | (1,1,224,3072) WIDTH, 96 cores × [224,32], K = 96, b = 7; bf16 fp32 DEST off | 21 450 | exchange (chains stubbed 19.4) |
| wid_tile_160x11008 | (1,1,160,11008) WIDTH, 115 cores, K = 115, b = 5 | 23 172 | exchange |
| wid_tile_13x777x1023 | (13,777,1023) WIDTH, 32 cores × [10400,32], K = 32, b = 325 | 386 828 | chains stubbed 322 µs; exchange + per-row collapse |
| wid_perf_32x7168 | WIDTH [32,256] 7×4, K = 28, 1 tile-row (perf cfg) | 4 987 | beats achievable 5 481 — at-risk cell |
| wid_rm_5x3x928x544 | RM WIDTH: 68 cores × [13920, 8] (16 B sticks), K = 68, 435 tile-rows | 850 204 | noDM 469.9 µs; +chains 384.6 µs |
| wid_rm_13x777x1023 | RM WIDTH: 128 cores × [10101, 8] (last member 7 valid), K = 128 | 677 812 | noDM 398.2; +chains 336.5 |
| wid_rm_1024x1024 | RM WIDTH 128 × [1024, 8], K = 128, fp32 DEST, fp32 RM gamma | 117 160 | noDM 89.1; +chains 82.5 (exchange) |
| wid_rm_f32_4x1x512x512 | fp32 RM WIDTH 128 × [2048, 4] (16 B sticks) | 227 110 | noDM 177.3; +chains 161.0 |
| hgt_rm_f32_128x4096 | fp32 RM HEIGHT: 128 cores × [1, 4096] (ONE stick per core), fp32 RM gamma | 93 794 | gamma STREAMED (wr_gamma_stream 48.5 µs); tilize/pass/untilize on 32×-padded tiles; chains stubbed 86.9 |
| hgt_rm_224x11008 | RM HEIGHT 112 cores × [2, 11008], bf16 RM gamma | 121 889 | pass 2 87 µs (incl. untilize + writer back-pressure), wr_gamma_stream 22 µs |
| hgt_tile_128x8192 | (128,8192) TILE HEIGHT auto (4 cores × 256 tiles), HiFi4 fp32 DEST | 68 741 | compute |
| blk_perf_7168x1024 | BLOCK [736,128] 8×10 perf cfg | 22 476 | (achievable 18 871) |
| blk_perf_7168x2048 | BLOCK [736,192] 11×10 perf cfg | 30 726 | (achievable 25 947) |
| blk_rm_99991x64 | RM BLOCK: 80 cores × [10000, 8] (16 B sticks), k = 8 | 513 234 | noDM 236.6; +chains 174.1 |
| blk_rm_128x4096 | RM BLOCK 130 cores | 13 138 | |
| int_rm_99991x64 | RM interleaved | 236 636 | |
| int_tile_f32_128x8192 / int_tile_bf8_128x8192 | interleaved fp32 / bf8b | 23 839 / 11 998 | |
| wid_tile_bf8_128x4096_nog | bf8b WIDTH, no gamma | 26 087 | exchange |
| hgt_tile_bf8_2048x256 | bf8b HEIGHT | 3 821 | |
| blk_tile_128x4096 | BLOCK TILE 52 cores | 10 049 | |

Golden harness sharding (`eval/sharding.py auto_shard_config`): ROW_MAJOR granule = 16 B of width (8 bf16 / 4 fp32
elements) × 1 stick; WIDTH splits W over up to 130 Tensix cores, so every WIDTH RM cell with W ≤ 1040 bf16 (520 fp32)
gets 16 B-wide shards; BLOCK splits W over ≤ 13 columns; HEIGHT splits rows over up to 130 cores (1 stick per core
when rows ≤ 130).

## What to return

`verdict` (WIN / NULL / REGRESSION / BLOCKED); a MENU of options, each with baseline-vs-candidate device ns on your target
cells AND on the at-risk cells named in your assignment, plus precision (PCC / max-abs vs torch and vs the op's current
output); the DOMAIN (applies everywhere, minus explicitly tagged exceptions: incorrect / inexpressible /
measured-regression — flat or untested is NOT an exception), with any measured crossover stated in portable units
(sticks or tiles per Tensix core, K, bytes per transfer); any raw-LLK path with the helper it bypasses and WHY
(capability vs ergonomics, and what exactly was missing/hard); the exact change you would graduate (file + function, and
a graduation-ready version of your kernels / host plan in your dir that mirrors the op's conventions so the coordinator
can port it mechanically), and your artifact dir.

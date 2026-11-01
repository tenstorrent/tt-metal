# rms_norm Perf 1 — shared brief for perf-part-optimizer subagents

Repo root: `/localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal`
(`source python_env/bin/activate` first). Board: Blackhole p150b, 13×10 = 130 Tensix cores, AICLK 1350 MHz.

## The op (frozen; correctness done)

- `ttnn/ttnn/operations/rms_norm/rms_norm.py` (entry + SUPPORTED), `rms_norm_program_descriptor.py` (blocking,
  CB plan, kernels, RT args), `kernels/rms_norm_{reader,compute,writer}.cpp`, `kernels/rms_norm_common.hpp`.
- Design + Blocking Model: `ttnn/ttnn/operations/rms_norm/op_design.md`; history: `changelog.md` (Refinements 1–5).
- Stages per block: reader (TILE interleaved: DRAM → cb_x; sharded TILE: cb_x IS the shard; RM: sticks → cb_x_rm)
  → compute: [RM tilize] → pass 1 `sum_squares_block` (eltwise_chain x*x DEST-accumulated per tile-row → cb_sumsq)
  → `collapse_sumsq_block` (reduce AccumulateViaAdd + Collapse: copy_tile seed + SFPU `sfpu_reduce` per tile-row;
  k == 1: + finalize post-op mul(1/W), add(eps), rsqrt on the FULL tile) → k > 1: writer exchanges column-0 partials
  (symmetric all-gather k ≤ 8 interleaved; owner combine for k > 8 interleaved and EVERY sharded k > 1: row r owned
  by member r % k; owner combines k partials + finalize, returns inv_rms by k unicasts) → pass 2 `normalize_window`
  (x · inv_rms bcast Col → DestReuse(DEST_TO_SRCA) · expanded gamma) → cb_out → writer (TILE interleaved DRAM
  writes; sharded TILE: cb_out IS the output shard; RM: untilize → sticks).
- Gamma ("expanded"): the writer loads this core's gamma slice into cb_gamma so EVERY row of tile j = gamma[32j..32j+31]
  (TILE gamma: 2 × 64 B DRAM reads per tile; RM gamma: one contiguous read + RISC-V scatter; then 4 doubling stages of
  local NoC copies + a face 0/1 → 2/3 copy, one barrier per stage). Interleaved / sharded k = 1: read + expanded once per
  W slice and multicast (`ttnn.Mcast`). Sharded k > 1: per-core load (`gamma_after_send` finishes it after the
  exchange when blocks ≤ 8 tile-rows). Fallback when the resident slice does not fit L1 (e.g. HEIGHT wide W): gamma
  STREAMED per block in G-tile chunks through a 2-chunk ring (`RMS_GAMMA_CHUNK`), re-read from DRAM + re-expanded.
- Stage zones (`MaybeDeviceZoneScope`, header `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`) are PERMANENT —
  keep them in any variant you author, extend them to new paths.

## Precision contract (fixed — never a lever)

`fp32_dest_acc_en`, `math_fidelity`, `math_approx_mode`, dtypes are the user's. Change HOW the same math is computed.
Golden tolerances: bf16 PCC ≥ 0.995 (perf cells: soft pcc_threshold 0.9995), fp32 ≥ 0.999, bf8b ≥ 0.99. Report per-option
PCC / max-abs error vs the op's current output on the cells you measure.

## Measuring

- Probe (cells built exactly like the golden harness): `tests/ttnn/unit_tests/operations/rms_norm/perf_p1_rms_norm.py`.
  Select with `P1_CELLS=a,b,...`; ablations with `P1_ABLATE=` / `P1_NOGAMMA=1` (see its docstring).
  **CSV rows come out in the CELLS dict order of that file, not your P1_CELLS order.**
- Device time: `scripts/run_safe_pytest.sh --profile <test>` → prints `SAFE_PYTEST: PROFILER CSV: <path>`; column
  `DEVICE KERNEL DURATION [ns]` of the `GenericOpDeviceOperation` rows. Zones: `generated/profiler/.logs/profile_log_device.csv`;
  `python3 ttnn/ttnn/operations/rms_norm/perf_experiments/tools/zones.py <labels>` summarizes the latest run per zone (max / p50 over Tensix cores, per RISC-V).
  (Copy it into your own dir if you need to modify it.)
- ONE fresh run per variant (no trial loops); a difference within ~2–3 % is noise. Only cells > 32 tiles count.
- Foreground device runs only (`run_safe_pytest.sh` / `tt-probe.sh`), never `python3` with device code directly.
- Recommended bench shape: clone the op (descriptor + kernels) into YOUR dir
  `ttnn/ttnn/operations/rms_norm/perf_experiments/<idea_slug>/`, apply your change there, and A/B the clone against the
  unmodified op on the same cells in the same pytest session. Never edit the real op files (other agents measure them
  as their baseline). Do not commit.

## Baseline scoreboard (grader real-time timer, current HEAD + zones, ns)

| cell id (probe) | config | ns | bottleneck (measured) |
|---|---|---|---|
| int_prefill_7168 | (1,1,8192,7168) bf16 TILE interleaved, HiFi2, fp32 DEST off | 633 563 | DRAM read+write; Tensix-core imbalance max/p50 1.47× (rd_read 551 vs 317 µs) |
| int_prefill_1024 | (1,1,8192,1024) same | 89 351 | DRAM |
| int_mid_7168 | (1,1,640,7168) same, k = 6 | 52 724 | DRAM reads (max/p50 2×) + combine |
| int_decode_32768 | (1,1,32,32768) HiFi4 fp32 DEST, k = 49 owner | 22 769 | combine + reads |
| blk_perf_7168x1024 | BLOCK [736,128] 8×10, perf cfg (23 tile-rows × 4 tiles / core, k = 8) | 30 309 | gamma 9 µs; collapse/combine/exchange; pass 2 |
| blk_perf_8192x1024 | BLOCK [1024,128] 8×8 | 40 949 | same |
| blk_perf_7168x2048 | BLOCK [736,192] 11×10 | 39 470 | same |
| blk_perf_640x6144 | BLOCK [64,704] 9×10 | 19 165 | CLOSE TO CEILING (achievable 19 039) |
| wid_perf_32x7168 | WIDTH [32,256] 7×4, k = 28, 1 tile-row | 8 893 | owner combine 2.2 µs + inv return 3.7 µs; gamma 2.5 µs |
| wid_perf_32x5120 | WIDTH [32,160] 8×4, k = 32 | 8 201 | same |
| hgt_tile_g | (128,8192) HEIGHT auto (4 Tensix cores × 256 tiles), HiFi4 fp32 DEST, bf16 TILE gamma | 79 267 | streamed gamma (wr_gamma_stream 75.6 µs); no-gamma 36.3 µs |
| hgt_tile_grm / hgt_tile_gbf8 | same, gamma bf16 RM / bf8b TILE | 119 594 / 126 089 | same |
| hgt_tile_4096_g | (1,1,128,4096) HEIGHT | 57 319 | streamed / resident gamma |
| hgt_rm_grm | (128,8192) RM HEIGHT, RM gamma | 183 659 | gamma (no-gamma 53 µs) |
| wid_tile_2048x256 | (1,1,2048,256) WIDTH auto (8 cores × 64 tile-rows × 1 tile) | 54 965 | per-tile-row collapse (≈220 ns/row), owner combine, exchange; gamma 8 µs |
| wid_rm_2048x256 | same, RM (32 cores, 8-datum = 16 B wide shards) | 245 666 | reader stick staging 125 µs (2048 × 16 B local NoC reads), writer 55 µs |
| blk_rm_2048x256 | RM BLOCK auto | 25 086 | tilize / staging |
| int_rm_128x4096_gf32 | (1,1,128,4096) RM interleaved, fp32 RM gamma | 18 894 | gamma 6.5 µs, tilize/untilize |
| int_tile_f32_128x8192 | fp32 TILE interleaved, fp32 TILE gamma | 27 752 | DRAM |
| int_tile_bf8_128x8192 | bf8b TILE interleaved | 13 972 | combine / fixed |
| reg_f32_2x1x128x256 | fp32 no gamma, defaults | 5 548 | fixed / combine |

Empty program (all kernels return at once) on the same cores: 0.18–0.39 µs.
Achieved DRAM read+write at full grid on BH p150b (catalog): ~415–430 GB/s.

## What to return

`verdict` (WIN / NULL / REGRESSION / BLOCKED); a MENU of options, each with baseline-vs-candidate device ns on your target
cells AND on the at-risk cells named in your assignment, plus precision (PCC / max-abs vs the op's current output);
the DOMAIN (applies everywhere, minus explicitly tagged exceptions: incorrect / inexpressible / measured-regression —
flat or untested is NOT an exception); any raw-LLK path with the helper it bypasses and WHY (capability vs ergonomics, and
what exactly was missing/hard); the exact diff you would graduate (file + function), and your artifact dir.

## Shared-device hygiene (several agents measure concurrently)

`generated/profiler/` is shared and overwritten by the next run. In the SAME bash command as your profiled run, copy the
printed `SAFE_PYTEST: PROFILER CSV:` file and `generated/profiler/.logs/profile_log_device.csv` into your own dir, then
parse those copies: `tools/zones.py <labels> <ops.csv> <profile_log_device.csv>`. Example:

    out=$(scripts/run_safe_pytest.sh --profile <your_test.py> 2>&1); csv=$(echo "$out" | sed -n 's/^SAFE_PYTEST: PROFILER CSV: //p'); \
      cp "$csv" $MYDIR/ops_run1.csv; cp generated/profiler/.logs/profile_log_device.csv $MYDIR/dev_run1.csv

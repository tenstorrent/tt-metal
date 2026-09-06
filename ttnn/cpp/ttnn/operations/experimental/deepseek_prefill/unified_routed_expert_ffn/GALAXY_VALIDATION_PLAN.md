# Grouped `unified_routed_expert_ffn`: Galaxy (32 x P150) validation hand-off

Branch `zbaczewski/moe-ffn-grouped` (8 commits on top of `main` 28238f903b3). Everything below was
measured on ONE P100 (11x10 grid, 7 DRAM channels, 448 GB/s). Read `GROUPED_FFN_PERF_P100.md` first; this
file is the to-do list for the first Galaxy session. Nothing here has run on a P150 yet.

## What changes on a P150 / Galaxy and why it matters

| P100 (measured) | P150 (production) | consequence |
|---|---|---|
| 7 DRAM channels, 448 GB/s | 8 channels, 512 GB/s | DRAM-bound cases (bf8, big experts) should scale ~8/7; bf4 Kimi is compute co-limited and will not |
| 11x10 compute grid | 13x10 | op still uses 11 columns (`grid_cols` default); 2 columns idle. `grid_cols=13` is NOT tested: per_core_N and the column->hidden mapping assume `N_tiles` padded to a multiple of 11 |
| bank = page % 7 | bank = page % 8 | band mode (`col_strided=1, grid_cols=8`) becomes usable: needs `N_gate_tiles % 8 == 0` and `N_down_tiles % 8 == 0` (true for Kimi 64/224 and M3 96/192). Implemented, compile-checked, never validated |
| single device | mesh of 32, fabric routers + CCL on some cores | rows 8-9 and the shared-expert sub-device may not be free during prefill; `grid_rows=10` may have to fall back to 8 |
| no other program in flight | dispatch/routing ops before and after | program-cache hits with changing counts every layer (tested on P100 with the cache-hit test) |

## Step 0: build and smoke (30 min)

```bash
git fetch && git checkout zbaczewski/moe-ffn-grouped
./build_metal.sh --enable-ccache          # or an incremental cmake --build build_Release --target install
source python_env/bin/activate; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD; unset TT_METAL_DPRINT_CORES
tt-smi -r && sleep 5
# single-chip smoke on one P150 of the mesh (same tests that pass on P100):
pytest tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_grouped_routed_expert.py \
  --timeout=0 -q -k "x_rm and (G5r10 or G10r10 or G4r8) and bf4"
pytest tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_grouped_routed_expert.py \
  --timeout=0 -q -k "cache_hit or count_clamp or all_empty or legacy_path"
```
Expected: all pass (72 + 5 on P100). A failure here is a P150-specific difference (grid, banks); check the
factory's `log_info` line `unified_routed_expert_ffn GROUPED: ...` for the chosen geometry first.

## Step 1: single-chip A/B on a P150 (20 min)

The drivers live in `perf_data/` (copies of the session scratchpad; they write `results/*.jsonl` next to
themselves and need `moe_bench_common.py` and `test_grouped.py` from the same directory):
```bash
cd ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/perf_data
python -u bench_dram_ceiling.py           # ceilings with 8 banks: expect ~460 GB/s all-core dual-NoC
python -u bench_ab.py --dists kimi_u,kimi_zipf,kimi_e24,m3_u4,m3_u8,m3_u16,m3_skew8 --dtypes bf4,bf8 \
   --configs "legacy;G5r10;G10r10;G4r8;G5r10m8" --iters 3 --check
python make_report.py > report_p150.md
```
Then add band mode to the sweep: `--configs "G5r10;G5r10c8s1;G10r10c8s1"` (c8 = 8 columns, s1 = strided).
Gate: band mode must be PCC-clean (>= 0.97 bf4, >= 0.99 bf8); keep it only if it beats G5r10 by > 5%.

## Step 2: does the model tolerate 10 rows? (30 min)

Kimi/M3 prefill on Galaxy runs CCL (all-to-all dispatch/combine over fabric) and possibly the shared
expert on a sub-device. Check `models/demos/deepseek_v3_d_p/tt/moe/tt_shared_expert.py` (`subdevice_cores`)
and the fabric router core reservations on the P150 mesh: if any worker rows 8-9 are reserved,
`ffn_grid_rows=10` will fail at program creation (`TT_FATAL grid too small`) or hang. Use
`TT_MOE_FFN_ROW_GROUPS=4 TT_MOE_FFN_GRID_ROWS=8` in that case (2-3% slower on P100).

## Step 3: full-model gate (1-2 h, needs the high-power 8x4 galaxy)

```bash
# baseline (legacy path, unchanged code): confirm the gate still holds
pytest models/demos/deepseek_v3_d_p/tests/perf/test_kimi_moe_perf.py -k galaxy
# grouped, report-only run (set expected_ns=None for the variant you run, or read the logged total):
TT_MOE_FFN_ROW_GROUPS=5 TT_MOE_FFN_GRID_ROWS=10 pytest models/demos/deepseek_v3_d_p/tests/perf/test_kimi_moe_perf.py -k galaxy
TT_MOE_FFN_ROW_GROUPS=10 TT_MOE_FFN_GRID_ROWS=10 pytest ... (many small uniform experts: best on P100 for Kimi 12/24)
```
The legacy expert FFN is ~2.4 ms of the 5.41 ms Kimi layer on Galaxy (12 x ~200 us); P100 numbers predict
~1.0-1.1 ms grouped, i.e. the layer should drop to ~4.0-4.1 ms. If the whole-layer gain is much smaller
than the op-level gain, the routing ops around it are now on the critical path, not this op.
Also run the M3 equivalent (`test_moe_perf.py`) with bf8 weights: that is where the P100 gain was 3.2-3.6x.

Correctness at model level: the existing Kimi/M3 MoE accuracy tests with the env overrides set.

## Step 4: profile on Galaxy (optional, 1 h)

Realtime program profiler works without env vars (`ttnn.device.RegisterProgramRealtimeProfilerCallback`);
`tests/ttnn/profiling/realtime_profiler_utils.py` filters by kernel path `/unified_routed_expert_ffn/`.
For per-RISC attribution: `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1` on one chip.
Realised GB/s = bytes streamed / op time (`bench_ab.py` prints it); on P150 peak is 512 GB/s.

## Known limits and things to decide

- Defaults in the model are still LEGACY (`ffn_num_row_groups=0`). Promote `ffn_num_row_groups=5,
  ffn_grid_rows=10` in `tt_moe.py` / `tt_minimax_moe.py` only after Step 3 passes on Galaxy.
- Rows per group R <= 7 (semaphore budget 9 + R <= 16). `experts_per_chip <= 32`, `num_row_groups <= 16`.
- `grid_cols=13` unsupported (see table). Band mode requires `grid_cols=8`.
- Per-core M cap: default 4 tile-rows per row (chunk = 4 x R tokens x 32). `per_core_m_max=8` helps only
  giant/skewed experts; the L1 guard may lower it (it logs the pick).
- Three NoC races were found and fixed in the GROUPED kernels only (report, section "Races"). The legacy
  reader has the same latent patterns: (1) `act_valid` multicast sourced from its own L1 word that the
  receiver reset can overtake; (2) the act sender pushes its own loopback copy before it landed. Port both
  (about 10 lines: flush before the reset, relay the valid from a constant word, make the sender wait on
  act_valid too) if any unexplained Galaxy hang or one-core-block PCC drop appears in the legacy path.
- Debugging on Galaxy: light watcher `TT_METAL_WATCHER=2 TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1
  TT_METAL_WATCHER_DISABLE_ASSERT=1` keeps timing close enough to reproduce races (the full watcher hides
  them); the grouped kernels push ring-buffer traces that `tools/parse_ring_trace.py` decodes; the
  waypoint columns are BRISC = writer, NCRISC = reader, then TRISC0-2.
- Compute is co-dominant at bf4 (about 24 cycles per tile-matmul, `out_subblock_h == 1`); the next op-level
  step is taller output subblocks, which also helps the legacy path.

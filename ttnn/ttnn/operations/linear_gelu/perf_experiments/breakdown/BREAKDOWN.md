# Perf 1 measured breakdown: linear_gelu 8x512 (focus), L1 x / L1 out, bf16, fp32_dest_acc_en=True, HiFi2, approx=False

Device: blackhole p100a, 110 cores used (grid 11x10, m over rows, n over cols, NOT transposed at 8x512),
AICLK 1350 MHz. Busiest core: 13 m tiles x 9 n tiles = 117 output tiles; block_h 3 (row-blocks 3,3,3,3,1),
block_w 1, subblock 3x1 (tail 1x1), K_ext = 25 (24 K tiles + 1 bias tile against a ones tile in x).

## Dataflow
- reader (NCRISC): x row-block (block_h x 25 tiles incl. ones tile) read by a rotating sender per grid row and
  multicast along the row; W slice chunk j (25 tiles: 24 W + bias) read from DRAM by a rotating owner per grid
  column and multicast down the column into cb_weight_resident (never popped, resident).
- compute: one matmul_block call per (row-block, n-block): K=25 accumulated in fp32 DEST, GELU applied on the
  PACK thread (TRISC2) per subblock via GeluTanhActivation<1> (calculate_gelu_tanh_fast in
  tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h), packed subblock-major to the
  output CB (bf16).
- writer (BRISC): drains output subblocks, one page per tile, flush per subblock.

## Per-stage zones (us, per-core totals, max/p50 over 110 cores) - balanced, max/p50 ~1.0
BRISC  writer_drain 135.6 (occupancy: waits on packer)     NCRISC kernel span 94.2 (reader is NOT critical)
TRISC0 compute_fill 7.0 (waiting for x row-block 0 + W chunk 0)
TRISC2 kernel 135.6, compute_matmul_gelu 135.2 (pack thread, GELU+pack)
Whole-op DEVICE KERNEL DURATION: 136.8 us

## Cumulative ablation (DEVICE KERNEL DURATION, us)
full                          136.8
no GELU (matmul+pack+DM)       92.3   -> GELU costs ~44.5 us on the wall; matmul path alone ~84 us + ~8 us fill
GELU only (SKIP_COMPUTE mm)   127.6   -> packer GELU+pack ~120 us / 117 tiles = ~1.02 us (~1380 cycles) per tile
no compute (DM only)           47.2
  + no W reads                 37.8
  + no x reads                 37.8
sync floor (+ no writes)       23.7
no writes only                136.6   (writer off the critical path)

## Ranking
1. Packer-thread SFPU GELU: binds (1.02 us/tile x 117). Target 90.8 us needs it under ~0.7 us/tile.
2. Matmul (math thread, HiFi2, fp32 DEST): ~84 us + fill; overlap partner of GELU. nogelu (92.3) is already
   above the 90.8 target, so it binds next once GELU drops. Spec target T_compute ~63.6 us.
3. Fill (~7 us before first compute). Writer/reader: not critical (ablation).

## Precision contract (fixed)
fp32_dest_acc_en=True, HiFi2, math_approx_mode=False, bf16 in/out. Current precision (precision baseline test):
PCC 0.99997, rel RMS 1.17e-2..1.18e-2 (dominated by bf16 matmul input rounding), GELU_FAST within 1 bf16 ULP
of torch gelu(approximate="tanh") where |GELU| >= 1e-3.

## Measurement mechanics in THIS checkout
- Profile: `LINEAR_GELU_PERF_PROBES=1 R4_VARIANTS=full scripts/run_safe_pytest.sh --profile <test.py>`
- TT_METAL_HOME=/localdev/kgrujcic/tt-metal (a non-checkout dir): the profiler CSVs land in
  /localdev/kgrujcic/tt-metal/generated/profiler/reports/<ts>/ops_perf_results_*.csv (col 19 = DEVICE KERNEL
  DURATION ns) and the zone log in /localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv.
  run_safe_pytest prints "no ops_perf_results CSV" - ignore it, read the path above
  (`ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1`).
- Zone summary: `python ttnn/ttnn/operations/linear_gelu/perf_experiments/breakdown/zones.py`.
- Whole-op kernel swap without touching the op: copy ttnn/ttnn/operations/linear_gelu/kernels/*.cpp into your own
  dir, edit, and monkeypatch `linear_gelu_program_descriptor.KERNEL_DIR` to your dir (see
  tests/ttnn/unit_tests/operations/linear_gelu/probes/test_r3_ablate.py / test_r4_ablate.py for the pattern;
  call device.disable_and_clear_program_cache()). Quoted #include "x.h" resolves relative to the kernel file,
  so put a modified SFPU function in a header inside your dir - NEVER edit shared tt_metal / kernel_lib headers
  (siblings run in parallel in the same tree).

# Perf 1 — common brief for all part-optimizers (wq_b_linear)

Op: `ttnn/ttnn/operations/wq_b_linear/` (program descriptor `wq_b_linear_program_descriptor.py`, kernels in `kernels/`,
design `op_design.md`, history `changelog.md`). DO NOT edit any file of the real op — other agents run in parallel.
Work only in your own `perf_experiments/<idea_slug>/` dir. To A/B against the whole op, copy the descriptor + kernels
into your dir and point KERNEL_DIR there (bench: `tests/ttnn/unit_tests/operations/wq_b_linear/_op_bench.py` shows
how to dispatch; copy it into your dir too). Don't commit.

Focus case (the ONLY cell): x [1,1,160,1536] bf16 TILE DRAM-interleaved, W [1536,4096] bfloat8_b TILE DRAM-interleaved,
y [1,1,160,4096] bf16 TILE L1-interleaved. Compute config FIXED: HiFi2, fp32_dest_acc_en=False, math_approx_mode=False.
Precision gate: PCC >= 0.9998 vs float matmul of the device-quantized operands (bench prints it); report PCC per option.
Tiles: M=5, K=48, N=128.

Board: Blackhole p150b, 11x10 = 110 workers, 8 DRAM banks (512 GB/s), AICLK 1.35 GHz. Whole-op SOL 14.0 us (DRAM).
Matmul SOL per core: 32 cycles per tile-matmul at HiFi2.

Current op (Refinement 2): 64 workers (8x8 rect at logical (2,2)-(9,9)) + 1 x-sender at (10,9).
Sender NCRISC reads x in 12 rounds of 5x4 tiles (trid window 2); sender BRISC multicasts each round linked on NoC1
into a fully-resident cb_x_resident on every worker, publishing an ordered round-count write (no handshake).
Worker NCRISC streams its 2 W columns (bank-spread, coalesced 2,176 B per k-row), cb_w_block depth 4 K-steps.
Worker compute: one kernel_lib `matmul_block` (packer L1 accumulation into cb_interm_acc, subblock 5x1,
in1_num_subblocks 2, block_k_tiles 4, 12 K-steps, SubblockMajor output on the last K-step).
Worker BRISC: receives x rounds (Counter) and pushes them to compute; stores the 5x2 output (per subblock, NoC split).

Measured breakdown this round (device ns, profiler, median dispatch; zones from `verify/stages.py`):
| run | span ns | note |
|---|---|---|
| full | 19,727 | x round 0 lands 2.77 us; compute matmul zone 2.8 -> 18.3 us (TRISC_0 15.4 us); y store done 19.4 us |
| compute alone (all DM stubbed) | 16,587 | matmul zone 15.3 us vs 11.38 us SOL (480 tile-matmuls/core) |
| sync floor (all payloads stubbed, matmul LLK skipped — unpack/pack/L1-acc still run) | 6,047 | TRISC_2 5.6 us, TRISC_0 5.1 us with no math |
| DM alone (compute stubbed) | 17,114 | W end max 15.8 us; x delivery ends 13.1 us |
| W read alone | 15,561 | W end max 14.25 us vs 13.06 us SOL |
| x read + mcast alone | 13,375 | x delivered by 12.6 us (one 64-receiver tree, ~39 GB/s); estimator 13.8 us |
| x read alone (1 core) | 9,836 | 8.6 us |
| BLOCK_K_TILES 6 / 8 (current schedule) | 20,844 / 21,786 | slower: x round 0 lands 3.45 / 5.1 us |
Ranking: compute is 3.9 us above its 64-core SOL and is balanced with DM (both ~16.6-17.1 us alone);
overlap loss 2.6 us (= fill: x round 0 at 2.77 us + tail ~1.1 us from last math to last y write).

Measurement: `scripts/run_safe_pytest.sh --profile <your test>` then parse
`generated/profiler/.logs/profile_log_device.csv` (see `ttnn/ttnn/operations/wq_b_linear/verify/stages.py` / `span.py`).
Device runs serialize on a flock — expected. Foreground runs only; never run python3 with device code directly.
Kernel zones: `#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"` -> `MaybeDeviceZoneScope("name")`.
Ablation defines in the real op: ABLATE knob (`WQB_KNOBS="ABLATE='compute+w_read'"`) maps to SKIP_COMPUTE /
ABLATE_W_READ / ABLATE_X_READ / ABLATE_X_MCAST / ABLATE_Y_STORE.

Report back (concise): verdict WIN/NULL/REGRESSION/BLOCKED; a menu of options each with device ns (part-alone AND
whole-op where you could measure it) and PCC; the domain (only incorrect / inexpressible / measured-regression
exceptions); any raw-LLK path with its kernel-head justification text and which helper it bypasses + why
(capability vs ergonomics, specific); and your artifact dir + the exact files a coordinator should port.

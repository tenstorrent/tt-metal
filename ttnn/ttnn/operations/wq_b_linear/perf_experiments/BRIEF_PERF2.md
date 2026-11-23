# Perf 2 — common brief for all part-optimizers (wq_b_linear)

Op: `ttnn/ttnn/operations/wq_b_linear/` (descriptor `wq_b_linear_program_descriptor.py`, kernels in `kernels/`,
design `op_design.md`, history `changelog.md` — read the Perf 1 section). DO NOT edit any file of the real op — other
agents run in parallel. Work only in your own `perf_experiments/<idea_slug>/` dir: copy the descriptor + kernels there,
point KERNEL_DIR at your copy, and dispatch like `tests/ttnn/unit_tests/operations/wq_b_linear/_op_bench.py` (copy it;
it supports `WQB_KNOBS="NAME=value,..."` and prints PCC). Don't commit. Don't write `dm_estimates.json`.

Focus case (the ONLY cell): x [1,1,160,1536] bf16 TILE DRAM-interleaved, W [1536,4096] bfloat8_b TILE DRAM-interleaved,
y [1,1,160,4096] bf16 TILE L1-interleaved. Compute config FIXED: HiFi2, fp32_dest_acc_en=False, math_approx_mode=False.
Precision gate: PCC >= 0.9998 vs the bench's reference (current op: 0.9999734). Tiles: M=5, K=48, N=128.
Board: Blackhole p150b, 11x10 workers, 8 DRAM banks, 1.35 GHz. Whole-op SOL 14.0 us (DRAM).

Current op (after Perf 1): 64 workers (8x8 rect, logical (2,2)-(9,9)) + 1 x-sender at (10,9). BLOCK_K_TILES=2.
- Sender NCRISC (`wq_b_linear_x_reader.cpp`, NoC0): reads x in 24 rounds of 5x2 tiles (20 KB), trid window 2,
  in-order retire + push.
- Sender BRISC (`wq_b_linear_sender.cpp`, NoC1, X_ORDERED_SIGNAL path): per round, linked multicast of the 20 KB
  (16 KB + 4 KB bursts, all linked) then an UNLINKED 4-byte count multicast into every worker's Counter cell. So the
  path is reserved once per round (24 reservations).
- Worker NCRISC: W stream, coalesced 2,176 B per k-row, depth 4 K-steps. Worker BRISC: receives x rounds, pushes them
  to compute, then stores the 5x2 output block (whole block, every other tile on NoC0), one barrier.
- Worker compute: kernel_lib `matmul_block`, caller_owns pack target = cb_out_block, packer L1-acc every K-step,
  subblock 5x1, 24 K-steps, one push at the end.

Measured breakdown (device span ns, median of 5 dispatches; zone end times in us after op start):
| run | span | key zone ends |
|---|---|---|
| full | 18,665 | x round 0 lands 1.74; sender x reads done 12.86; x mcast issue 13.5 us long, done 16.39; worker x_wait_rest 16.51; TRISC_0 matmul 16.71; TRISC_2 17.20; y_issue 0.69 us -> 17.96; y_barrier 0.30 -> 18.40 |
| compute stubbed | 17,616 | x mcast done 16.06 (with W traffic) |
| compute + W stubbed | 17,236 | x mcast done 15.57; x reads done 9.94 |
| x read+mcast alone (compute, W, y stubbed) | 15,887 | mcast issue 12.9 us (491 KB -> ~38 GB/s); x read alone 9.5-9.9 us (~50 GB/s) |
| W alone (compute, x, y stubbed) | 14,197 | last W read 13.7 med / 14.1 max (SOL 13.06) |
| x stubbed (W + compute + y real) | 15,656 | W end 13.6/14.0; TRISC_2 end 14.3/14.7 |
| compute alone (W, x, y stubbed) | 13,686 | matmul zone 12.4 us (SOL 11.38); pack-only path (SKIP_COMPUTE) alone 10.7 us |
| X_READ_TRIDS=8 (knob) | 20,543 | reads still end 9.8; round 0 lands 3.6 (in-order retire) — worse |

Ranking: x multicast pacing is the wall (2.7x its part SOL of 5.69 us = sender NoC egress). The estimator binds the
multicast on per-burst path reservation (~352 cycles); the op model's 16.6 us prediction assumes a linked multicast pays
its reservation ONCE for the stream. Next: x read (one core, 9.9 us, ~50 GB/s), tail (~1.2-1.4 us from last pack to
y done), compute (12.4 vs 11.38), W (near SOL).

Measurement: `scripts/run_safe_pytest.sh --profile <your test>`, then `python3 ttnn/ttnn/operations/wq_b_linear/verify/stages.py`
(zones) / `verify/span.py`. Foreground only; never python3 with device code directly. Device runs serialize on a flock.
Zones: `#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"` -> `MaybeDeviceZoneScope("name")`; keep the existing
zones in anything you port. ABLATE knob: `WQB_KNOBS="ABLATE='compute+w_read'"` (members w_read, x_read, x_mcast, compute, y_store).

Report back (concise, <= 40 lines): verdict WIN/NULL/REGRESSION/BLOCKED; a menu of options, each with whole-op span ns
(and part-alone ns where relevant) and PCC; the domain (only incorrect / inexpressible / measured-regression exceptions);
any raw-LLK path with its kernel-head justification and which helper it bypasses + why (capability vs ergonomics,
specific); your artifact dir + the exact files/knobs a coordinator should port.

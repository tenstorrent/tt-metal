# scatter_offload: move the bank_coalesced NoC-loopback scatter off NCRISC

This experiment is isolated. The real op is untouched, nothing is committed, and every number is from a WH B0 n150 (64 Tensix cores, AICLK 1000 MHz), measured as `DEVICE KERNEL DURATION [ns]`.

## What it tests
In the bank_coalesced `load_block` (`read_bank_coalesced`), the reader RISC-V (NCRISC) handles each unit (2 tile-rows = 64 sticks) serially: about 12 per-bank DRAM reads, a barrier, then 64 NoC-loopback one-packet reads that scatter the bank-major staged sticks into their tilize positions. Meanwhile the writer RISC-V (BRISC) waits in `cb_wait_front`. This experiment hands part of that work to BRISC:

| SO_MODE | who does what |
|---|---|
| 0 (`base`) | the op's own `read_bank_coalesced` (the baseline, byte-identical apart from the ABL_W guard) |
| 1 (`half`, `half_q`, `half_e`) | NCRISC does every DRAM read. The scatter is split by rotated bank ordinal: NCRISC takes ordinals `[0, NC_NUM/DEN * banks)` and BRISC the rest, as NoC1 loopback reads |
| 2 (`all`) | BRISC does the whole scatter. NCRISC only reads DRAM and pushes |
| 3 (`split*`) | BRISC also does the DRAM reads for its bank ordinals (NoC1), into its own staging ring, and scatters them |
| `_s3` | staging depth 3 instead of 2 (NCRISC issues unit k+2 before it scatters unit k) |
| `_p` | BRISC's poll loop stores output before it scatters (the scatter only fills BRISC's idle time) |

Handshake: two program semaphores (local L1 words holding monotonic unit counters). `staged` goes NCRISC to BRISC (slot reserved, plus staging landed in modes 1/2). `done` goes BRISC to NCRISC. NCRISC stays the only producer of cb_input_sticks: it defers its pushes and polls for them, so it never blocks while holding a reserved slot it hasn't pushed. BRISC never makes a blocking CB call: it polls `staged` and `cb_pages_available_at_front(cb_output_tiles)`. That keeps it deadlock-free. Mode 3 needs its own BRISC staging ring (the staging CB is doubled). A bank's byte range inside a staging slot depends on the unit's run structure (one 64-stick run vs two 32-stick runs when the rotated walk wraps). With a shared slot, one RISC-V's reads for unit k+2 overwrote the other's unscattered data for unit k. An earlier run caught this: 143–462 bad sticks, all on odd-rotation cores.

## Files
- `kernels_so/`: a copy of the op's kernels. `so_coalesced.hpp` holds the shared geometry (bank-ordinal shares for issue and scatter). The reader has `read_bank_coalesced_so`, and the writer has the `#if SO_MODE` poll loop. `tilize_stick_reads.hpp` adds an `#ifndef ABL_W` guard around the tile write (the writes-stubbed ablation).
- `tests/ttnn/unit_tests/operations/tilize/test_tilize_perf1_scatter_offload.py`: the harness. It monkeypatches `KERNEL_DIR` and `BANK_COALESCE_STAGE_DEPTH`, and wraps `create_program_descriptor` to inject the SO_* defines and 2 semaphores **only when reader CT arg 26 (coalesce_depth) != 0**. It is `torch.equal`-gated. Environment variables: `SO_VARIANTS`, `SO_SHAPES`, `SO_GUARDS=1` (adds low_l1).
- `ops_ns.py`: maps the ops-CSV rows to variant names by collection order.

Run: `SO_VARIANTS=base,half,base_W,half_W scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/tilize/test_tilize_perf1_scatter_offload.py`

## Results: focus shape [1,1,16384,64] bf16, DRAM interleaved (medians; n = runs)
| variant | full op ns | writes-stubbed ns |
|---|---|---|
| base | 24308 (n=3; session range 23527–24521) | 15973 (n=3) |
| half (1/2 scatter on BRISC) | 24508 (n=3; 7-run spread 23266–26890) | **14700 (-8.0%)** |
| half_q (1/4) / half_e (1/6) | 25067 / 25580 (n=1) | 15437 / 16018 (n=1) |
| all (BRISC scatters all) | 24311 (n=3) | 16628 (+4.1%) |
| split (1/2 banks read on NoC1 + scatter) | 32050 (n=1, +32%) | 20599 (+29%) |
| split_q3 (1/4) | 25980 (n=3, +6.9%) | **14782 (-7.5%)** |
| split_e (1/6) | 24653 (n=1) | 15557 (n=1) |
| base_s3 (reorder, depth 3) | 24050 vs 24063 (n=3) | 18542 vs 16038 (+16%) |
| half_s3 / all_s3 / split_s3 | 24692 / 25124 / 32205 (n=1) | 16611 / 18683 / 21485 |
| half_p / half_q_p / all_p (store priority) | 26751 / 25600 / 24324 vs 23940 (n=3) | — |

## Domain sweep of the best reader-chain candidate `half` (full op, then writes-stubbed; medians)
| shape | base -> half, full | base -> half, writes-stubbed |
|---|---|---|
| [1,1,32768,64] | 46488 -> 50683 (**+9.0%**, n=3; +5.3% in a separate n=2) | 27138 -> 24692 (-9%) |
| [1,1,16384,32] | 14380 -> 14481 (flat) | 13319 -> 11211 (-16%) |
| [1,1,4096,64] | 8322 -> 8353 (flat) | 5417 -> 5242 (-3%) |
| [4,3,256,96] (takes bank_coalesced) | 9231 -> 9080 (flat) | 6145 -> 5945 (-3%) |
| [1,1,16384,128] | 46690 -> 46025 (flat) | 27824 -> 26105 (-6%) |
| [1,1,16384,64] low_l1 | 23972 -> 26154 (+9%, n=2, noisy) | 16055 -> 14662 (-9%) |

## Mechanism
The reader chain does get shorter. The zones show NCRISC `reader_scatter` falling from 1622 to 980 cycles per unit, and the writes-stubbed op running 3–16% faster. In the full op, though, the critical path is the writes on the straggler Tensix cores: `writer_issue` / `writer_flush` sum-max is 13.0k / 11.8k cycles against a p50 of 4.5k / 3.5k. Any scatter work placed on BRISC, which adds about 1000 cycles per unit (`so_brisc_scatter`), plus its NoC1 loopback traffic, lands on that path. The result is flat on the focus shape, with a heavy +8..11% tail in some runs, and +9% on [1,1,32768,64]. BRISC DRAM reads on NoC1 lose badly at a 1/2 bank share, which confirms the earlier NoC1-DRAM-read finding. Depth-3 staging is flat in the full op and hurts the isolated read chain (+12..16%), because more DRAM requests in flight means more bank congestion.

Verdict: NULL on the focus shape and REGRESSION across the domain. Do not graduate. If the writes ever stop being the critical path, `half` (-8% reader chain) and `split_q3` (-7.5%) are the options to re-measure.

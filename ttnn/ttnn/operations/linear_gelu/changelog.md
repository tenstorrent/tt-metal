# Changelog: linear_gelu

## Phase 0 - Core Implementation
- **Date**: 2026-11-02
- **What was done**: Initial implementation via incremental pipeline (planner -> implementer -> verifier). Regime R1 (2D grid, W slice resident per core via column multicast, full K + bias tile in fp32 DEST, accurate tanh GELU on the packer) plus R2 (x row-block multicast along grid rows, built early on measurement: 8x512 393 -> 195 us).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned], fp32_dest_acc_en=[True], memory_layout=[INTERLEAVED], buffer_type=[L1], out_buffer_type=[L1]
- **Accuracy achieved**: PCC=0.99997, max_abs_err=5.5e-2, rel_rms_err=1.17e-2..1.18e-2 (measured on 4 shapes via test_linear_gelu_precision_baseline.py; stock 1.21e-2..1.30e-2)
- **Perf at Phase 0 (device us)**: 8x512 195.6, 16x256 195.7, 32x128 195.7, 8x384 155.0, 4x512 114.1, 8x256 114.2, 2x512 73.1, 4x256 73.0 (beats stock on every row; target 90.8 at 8x512)
- **Golden suite at Phase 0**: 23 / 23 supported cells passing, 73 xfail_expected, 0 loud (per `golden_results/verifier_report.json`)
- **Issues encountered**: verifier code-review fixes: deduplicated config/placement resolution (`_resolve`), moved the output memory_layout refusal ahead of SUPPORTED, hoisted the loop-invariant subblock derivation. No drift fixes.
- **Tests added**: test_linear_gelu.py, test_linear_gelu_perf.py, test_linear_gelu_precision_baseline.py

## Refinement 1 - Non-aligned S + DRAM placement for x and output
- Date: 2026-11-02
- What was done: writer stage W3. For S % 32 != 0, rows `S % 32`..31 of every tile in a tail tile row (`m % seq_tiles == seq_tiles - 1`) are zeroed in the `cb_output_tiles` page via `noc.async_write_zeros` before the NoC write, with one L1 barrier per tail-row subblock row. `seq_tiles` and `last_valid_rows` are writer CT args, so aligned S compiles the path out. DRAM x / output needed no kernel change (TensorAccessor, R2 x multicast reads x once). SUPPORTED widened: alignment += h_non_aligned, buffer_type += DRAM, out_buffer_type += DRAM.
- Accuracy achieved: PCC >= 0.9999 on all golden cells (unchanged path); padded rows exactly 0 with NaN / Inf padding in x.
- Golden test progress: 96/96 (73 newly supported cells pass, 23 prior pass), regression file 6/6.
- Perf (device us, 8x512): L1/L1 195.4 (no regression vs 195.6), DRAM x 199.4, DRAM out 196.2, 8x500 non-aligned 195.5.
- Issues encountered: None.
- Tests added: `test_linear_gelu_perf.py::test_linear_gelu_perf_guard` (8x512 DRAM x, 8x512 DRAM out, 8x500 non-aligned).

## Refinement 2 - Speed up the flagged 8x512 profile: per-tile compute cost
- Date: 2026-11-02
- What was done: ablated the per-tile compute at 8x512: no activation 90.6 us, RELU6 on the packer 97.0 us, accurate tanh GELU 195.4 us, so the packer-thread SFPU GELU binds. Adopted `GELU_FAST = 1` (x / (1 + exp(-2u)), the same tanh GELU by exact identity, about 22 vs 38 SFPU ops) after the lamp L1 precision gate held. Tried splitting the GELU between the MATH thread (PostCompute hook) and the packer: reverted. The SFPU is a single shared unit, so concurrent streams race (NaN), and the split cannot add SFPU throughput anyway (157 us at 2 tiles on MATH). Subblock / block_w / block_h not turned: they only shorten the matmul, which is off the critical path.
- Perf (device us, before -> after): 8x512 195.4 -> 143.8, 16x256 195.7 -> 143.6, 32x128 195.7 -> 143.5, 8x384 155.0 -> 114.7, 4x512 114.1 -> 85.9, 8x256 114.2 -> 85.9, 2x512 73.1 -> 57.0, 4x256 73.0 -> 56.9; guards 8x512 DRAM x 199.4 -> 147.7, 8x512 DRAM out 196.2 -> 144.3, 8x500 non-aligned 195.5 -> 143.4.
- Accuracy achieved: PCC=0.99997, rel RMS 1.1748e-2..1.1839e-2, max abs 3.5e-2..5.5e-2 on shapes [1x32, 2x512, 8x384, 8x512], identical to the accurate path to 4 significant digits.
- Golden test progress: slice 8x512 / 1x32 / 8x500 13/13, regression 6/6; unit tests 35/35.
- Issues encountered: the MATH/PACK GELU split races on the shared SFPU (see above). Remaining headroom to the 90.8 us target is SFPU GELU time: the next lever is fewer dependent SFPU ops per iteration, which lowers internal precision and needs a measured-identical gate first.
- Tests added: None (existing perf + precision tests cover the change).

## Refinement 3 - Speed up the flagged small-token rows (1024 / 2048 tokens)
- Date: 2026-11-02
- What was done: host-side blocking and W-delivery changes on the existing R1/R2 topology, plus one reader change. (1) `BLOCK_W` 3 -> 1: the first W chunk is 25 tiles instead of 75, which cuts the fill latency in front of all compute. It also gives the finest n split. Measured faster on every shape (1x32 24.0 -> 21.6, 8x512 143.4 -> 138.7). (2) `select_blocking` scores the transposed orientation (m on grid columns, n on grid rows) and keeps it when the busiest core gets fewer tiles: 1024 tokens 36 -> 30 tiles/core on 110 cores (was 88), 2048 tokens 63 -> 60. The 3072/4096-token rows tie or lose, so they keep the old orientation. (3) The transposed grid put every W sender on one grid line and measured +20 us. The W channel is now a rotating sender inside each column, with round offset `col + W_ROUND_SHIFT`, so senders sit on a diagonal. (4) `W_ROUND_SHIFT = 1` moves the chunk-0 owner off the row's x round-0 sender, and `W_PREFETCH_ROUNDS = 1` issues chunk 0's DRAM read at kernel start, before x row-block 0. Together these overlap the two first reads. (5) `OUT_DEPTH` 2 -> 6 keeps `cb_output_tiles` at 24 pages, the same as at block_w 3. At depth 2 the 8-page ring stalled the packer on DRAM output (143..146 vs 138.7 us).
- Latent bug fixed: `cb_output_tiles` drains a subblock as one contiguous span, so a tail subblock must not straddle the ring end. The host now limits the tail subblock area to divisors of `block_h * block_w`. At block_w 1 / 2 the old rule wrapped (PCC 0.72); at block_w 3 it never triggered.
- Levers measured and parked: W prefetch beyond the first round (rolling lookahead 1/2/4, or depth 2/3 at start) is slower at every depth. The W reads are DRAM-bandwidth-bound (all 10 column streams together run near the ~11 us floor for 4.7 MB), so more reads in flight only delay the chunk compute needs first. `W_PREFETCH_ROUNDS` stays a live knob at 1. Rotating W without prefetch measured identical to the fixed staggered sender.
- Perf (device us, L1/L1, before -> after): 2x512 56.8 -> 42.4, 4x256 56.9 -> 42.4, 4x512 85.8 -> 74.8, 8x256 85.8 -> 74.7, 8x384 114.8 -> 108.4, 8x512 143.5 -> 137.1, 16x256 143.7 -> 137.1, 32x128 143.5 -> 137.2. Guards: 8x512 DRAM x 147.6 -> 143.4, 8x512 DRAM out 144.5 -> 137.5, 8x500 non-aligned 143.5 -> 137.3; 1x32 24.0 -> 21.5.
- Bottleneck now (2x512 ablation): full 43.9, no GELU 34.6, no compute (data movement only) 32.4, sync floor 15.1 us. The FILL zone (compute waiting for x block 0 + W chunk 0) was 8.1 us before the prefetch. Reader zones: x round 0 (72-tile L1 read + multicast) 4.6 us, then W chunk 0 about 3 us. The packer-thread GELU (about 1.1 us/tile x 30 tiles) and data movement now co-bind.
- Accuracy achieved: PCC=0.99997, rel RMS 1.1743e-2..1.1839e-2, max abs 3.5e-2..5.5e-2 on shapes [1x32, 2x512, 8x384, 8x512], unchanged from Refinement 2.
- Golden test progress: slice (all 8 loose cells, every placement of 2x512 / 4x256 / 4x512 / 8x256 / 13x300 / 7x511 / 2x37, regression file) 42/42; unit tests 35/35 + precision 4/4 + perf 11/11.
- Issues encountered: the ring-wrap bug above; the rotating Mcast refuses a one-core group, so a one-row column (1x32) keeps the fixed sender.
- Tests added: perf/ablation probes `tests/ttnn/unit_tests/operations/linear_gelu/probes/test_r3_ablate.py` (stage ablation + FILL / reader zones), `test_r3_blocking.py` (forced block_w x orientation x stagger x OUT_DEPTH), `test_r3_prefetch.py` (prefetch depth x round shift).

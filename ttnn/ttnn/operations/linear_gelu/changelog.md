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

## Refinement 4 - Speed up the flagged 4096-token profile: fill, drain and data-movement residue
- Date: 2026-11-02
- What was done: profiled 8x512 first. Compute still binds: full 137.2, no GELU 88.2, GELU only 126.1, data movement only 45.4, no writes 136.7, sync floor 21.8 us. LoFi gives 135.7, so the gap above GELU-only is not FPU work. The exposed data-movement residue is the fill: FILL zone 8.1 us = x round 0 (96-tile read + multicast, 6.2 us) then the W chunk-0 multicast. Drain is 0.4 us. Two levers landed. (1) `BLOCK_H_MAX = 3`: a shorter x round 0 (150 vs 200 KB), with the steady state still GELU-bound (block_h 4 / 3 / 2 / 1 = 137.2 / 135.7 / 140.6 / 147.9 us). The `cb_output_tiles` ring is now sized in pages (`OUT_RING_PAGES = 24`, depth derived per block_h): at block_h 3, depth 6 gave 18 pages and DRAM out 138.6 vs 137.5. (2) Reader split of the activation load into issue-reads / finish-round, plus `W_CHUNKS_BEFORE_X` (CT arg): W chunk 0 is multicast while the x round-0 sender's reads are in flight. It wins only with DRAM x (143.4 -> 141.0 at block_h 4, 140.9 -> 139.7 at block_h 3) and is flat or +0.6 us with L1 x, so it is gated by x's buffer type. Either order still serializes the two first multicasts, because each channel's consumer-ready handshake waits for the other.
- Levers not built: write batching (removing every write saves 0.5 us, so the writer is off the critical path) and x bank stagger (x reads are hidden after round 0). `ACT_DEPTH = 3` does not fit at block_h 3 with the 24-page ring.
- Perf (device us, R3 -> R4): 8x512 137.2 -> 135.8, 16x256 137.2 -> 135.7, 32x128 137.1 -> 135.8, 8x384 108.4 -> 106.7, 4x512 74.6 -> 74.7, 8x256 74.7 -> 74.6, 2x512 42.5 -> 42.5, 4x256 42.5 -> 42.5; guards 8x512 DRAM x 143.4 -> 138.4, 8x512 DRAM out 137.5 -> 137.6 (3 runs: 137.5..137.7, noise), 8x500 137.3 -> 135.9.
- Bottleneck now: the packer-thread SFPU GELU (117 tiles x about 1.0 us on the busiest core) plus about 8 us of fill. Remaining levers: fewer SFPU ops per GELU tile (precision-gated, see Refinement 2), and overlapping the first W and x multicasts. The second needs the W channel without a handshake, which in turn needs a per-group rotation offset in `ttnn.Mcast` (Counter data-ready waits on an absolute round).
- Accuracy achieved: PCC >= 0.9999 on every golden cell run; precision baseline test unchanged (rel RMS 1.17e-2..1.18e-2 on [1x32, 2x512, 8x384, 8x512]).
- Golden test progress: slices 55/55 (every DRAM-x cell, 8x512, 1x32, regression) and 54/54 (all placements of 8x512, 16x256, 32x128, 8x384, 128x32, 64x64, 8x500, 7x511, 13x300, 1x32, 16x8, plus regression); unit dir 50 passed.
- Issues encountered: the profiler CSV lands in the sibling checkout's `generated/` (PYTHONPATH resolves `tracy` there).
- Tests added: perf probes `probes/test_r4_ablate.py` (8x512 stage ablation / zones / fidelity), `probes/test_r4_block_h.py` (block_h cap x output ring), `probes/test_r4_fill_order.py` (W_CHUNKS_BEFORE_X x placement); `gelu_only` variant added to `test_r3_ablate.py`; `test_r3_blocking.py` moved to `OUT_RING_PAGES`.

## Perf 1
- Date: 2026-11-02. Device: Blackhole p100a, 110 cores, AICLK 1350 MHz. Focus: LOOSE_CASES 8x512 (x / out L1, W / bias DRAM, bf16, default config: HiFi2, fp32_dest_acc_en=True, math_approx_mode=False). Every number is DEVICE KERNEL DURATION. Precision knobs were never changed.
- Instrumentation (permanent `MaybeDeviceZoneScope`): `compute_fill`, `compute_matmul_gelu` (per row-block), `reader_x_issue` / `reader_x_mcast` (per row-block), `reader_w_prefetch`, `reader_w_first`, `reader_w_deliver`, `writer_drain` (per row-block), `writer_barrier`. At most 30 markers per RISC at 8x512, so well inside the budget.

### Measured breakdown (8x512, before)
- Zones, per-core totals, max / p50 over 110 cores (balanced): TRISC2 (pack) span 135.6 us. compute_fill 7.0 us on TRISC0. NCRISC span 94.2 us, so the reader is not critical. writer_drain 135.6 us is occupancy waiting on the packer.
- Cumulative ablation (us):

| variant | us |
|---|---|
| full | 136.8 |
| no GELU | 92.3 |
| GELU only (SKIP_COMPUTE matmul) | 127.6 |
| no compute | 47.2 |
| + no W reads | 37.8 |
| + no x reads | 37.8 |
| sync floor (+ no writes) | 23.7 |
| no writes only | 136.6 |

- Ranked bottleneck:
  1. Pack-thread SFPU GELU: about 1.02 us/tile x 117 tiles, binds.
  2. Matmul: about 84 us, the overlap partner on the math thread.
  3. Fill: 7 us.
  Data movement alone is 47 us, so the DM ceiling is not the gate.

### Portfolio and verdicts (one perf-part-optimizer each; artifacts in `perf_experiments/<slug>/`)

**`gelu_loadmacro`: WIN, graduated.**
- Raw-LLK tanh GELU: two DEST rows in lockstep (covers the SFPMAD latency), constants kept in LRegs, log2 split on SFPSTOCHRND, and the per-row reload / multiply / store tail issued as SFPLOADMACRO. The reciprocal is refined by `r(1 + e + e^2 + e^3)`.
- Bench: 1031 -> 602 ns/tile. Whole op: 8x512 136.3 -> 98.1 us.
- Precision: bf16 output within 1 ULP of the old body, no bias. PCC 0.99997 and rel RMS 1.1748e-2, identical to before.
- Options not taken:
  - V5 (series cut at e^2): 96.0 us, but its fp32 result is about 1 fp32 ULP low (one-sided) for x > 4. It is invisible after bf16 rounding but breaks the unbiased contract.
  - V2 to V4: 709 / 638 / 590 ns/tile.

**`gelu_sfpu_schedule`: WIN, superseded (same stage).**
- `magic4x2w` (magic-number exp2, minimax degree-4, cubic reciprocal, two rows per pass): 674 ns/tile; whole op 136.3 -> 104.8 us; 1 ULP.
- `magic3x2w`: 101.7 us, but 1.25% of values off by 1 ULP.
- `nr1` variants: one-sided bias, not graduatable.
- Hardware approx exp: wrong results.

**`gelu_alt_approx`: WIN, superseded (same stage).**
- `t21s` (degree-3 2^f, one bias-corrected Newton step, SFPSTOCHRND split): 850 ns/tile; whole op 136.2 -> 115.2 us; 1 ULP.
- Hardware `approx_exp`: inexpressible (valid only for |x| < 1.4).

**`gelu_call_overhead`: small WIN, not graduated.**
- Per-tile SFPU call scaffolding is 21 cycles/tile (1.3%). Exposed pack time is about 26 cycles/tile.
- Removing the per-subblock `pack_reconfig_data_format` saves 1.2 us, which is in the noise band and would need a change to the shared `matmul_block` helper.
- Hoisting one constant: superseded by the load-macro body.
- Unroll 4 / 2 / 1 is slower: the body is already replayed, so I-cache is not a factor.

**`matmul_throughput`: conditional WIN, graduated as `BLOCK_H_MAX = 4` with a DRAM-out carve-out.**
- In steady state the matmul runs at 93% of HiFi2 peak. Row-block 0 is paced by W delivery (about 2.7 us per chunk).
- No-GELU path: 92.1 -> 88.7 us with block_h 4.
- Other options: subblock / block_w 2 / 3 / 4 flat or slower; raw loop without per-call overhead flat (92.2); bias seeded into DEST 125.2 (regression: about 1000 cycles per subblock of copy / matmul re-init); x row-block 1 reads during W delivery flat; deeper W prefetch slower.

**`fill_first_block`: NULL.**
- The fill is a serial chain: x0 reads 2.4 us, x0 multicast 2.2 us, W0 multicast 1.15 us.
- Results (us): ragged first block 151.3; every core reads x0 itself 188.0; every core reads W0 itself 154.0; split x0 senders 137.0 to 138.7; W first 136.8; W channel without handshake 139.3; NoC1 x reads hang (dedicated-NoC mode).
- What would help is a `ttnn.Mcast` receiver that signals ready without waiting for data, so the two first multicasts can overlap (about 1.15 us at best).

**Reuse broadcast (standing idea):** already in place. x is multicast along rows and W along columns, so it was not floated.

### What graduated
1. **Load-macro GELU** (`kernels/linear_gelu_gelu_sfpu.h`, V6).
   - It is the op's only GELU path. `GELU_FAST` and the compute CT arg are deleted.
   - It plugs into `matmul_block`'s pack-thread Activation slot through explicit specializations of `ActivationInit/ApplyHelper<GELU_TANH, kGeluLoadMacro>`.
   - Domain: every supported cell, no carve-out. The only constraint is fp32 DEST (static_assert), which is the only supported config.
2. **`BLOCK_H_MAX` 3 -> 4.**
   - Medians of 3: 8x512 98.5 -> 95.1, 8x384 79.9 -> 76.1, 8x500 98.6 -> 94.5, 8x512 DRAM x 105.0 -> 101.9 us.
   - Flat at 2x512, 4x256 and 1x32, where block_h does not change.
   - Carve-out `BLOCK_H_MAX_DRAM_OUT = 3`, earned by a measured regression: 8x512 with DRAM output 143.8 -> 153.7 us (medians of 3; the writer binds there).

### Whole op, before (R4 kernels) -> after (us, one run each unless noted)

| case | before | after |
|---|---|---|
| 8x512 (focus) | 136.7 | 94.7 (1.44x) |
| 16x256 | 136.7 | 95.3 |
| 32x128 | 136.8 | 94.8 |
| 8x384 | 107.6 | 76.5 |
| 4x512 | 75.5 | 59.6 |
| 8x256 | 75.4 | 60.5 |
| 2x512 | 43.3 | 36.2 |
| 4x256 | 43.3 | 37.3 |
| 8x512 DRAM x | 140.3 | 104.3 |
| 8x500 | 137.0 | 94.7 |
| 1x32 | 22.2 | 21.6 (medians of 3, flat) |
| 8x512 DRAM out | 138.6 | 141.4 (medians of 3) |

- Every flagged row now beats stock by 1.2x to 2.6x.
- 8x512 is still 4% above the 90.8 us target.
- DRAM out is +2.0%. That is inside the noise band, but all 3 pairs moved the same way, so it is recorded here. No carve-out: it is not a material regression.

### Guard set and correctness
- Golden `eval/golden_tests/linear_gelu/`: 102 / 102 passed (after each graduation).
- Unit, precision and perf tests: 50 / 50.
- Precision baseline unchanged: 8x512 PCC 0.999970, rel RMS 1.1748e-2; 1x32 1.1839e-2.
- Guard sweep: `perf_experiments/breakdown/guard.sh` (probe `tests/ttnn/unit_tests/operations/linear_gelu/probes/test_p1_guard.py`) covers every loose shape, DRAM x, DRAM out, non-aligned and 1x32.

### Bottleneck now (8x512)
- Ablation (us): full 94.9, no GELU 89.1, GELU only 80.4, DM only 45.9.
- The matmul path binds, with the W-paced row-block 0 and the fill exposed. GELU is mostly hidden.
- DRAM output is writer-bound at about 141 us and is the next target there.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `GeluTanhActivation` / `ActivationInitHelper` + `ActivationApplyHelper` (`sfpu_activation_helpers.hpp`) -> `calculate_gelu_tanh_fast` | capability | The pack-thread activation slot of `matmul_block` takes a closed `KernelActivation` enum: there is no way to supply a custom SFPU body / init, so the op had to explicitly specialize the library templates on an unused param value (`kGeluLoadMacro = 2`). The library's sfpi GELU compiles to one serial chain per row (34 instr + ~10 MAD stall cycles, 1031 ns/tile); there is no 2-row-interleaved or SFPLOADMACRO variant, and the load-macro state (LReg11-14, macro templates / sequences, ADDR_MOD_6) has no ownership contract in the helper. | 136719 | 94660 (98270 with block_h 3) | `kernels/linear_gelu_gelu_sfpu.h:150-165`, used at `kernels/linear_gelu_compute.cpp:47` |

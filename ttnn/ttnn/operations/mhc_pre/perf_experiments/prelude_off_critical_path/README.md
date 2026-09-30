# prelude_off_critical_path — Perf 1 round 1 (BH p150, card 1, in-process DEVICE KERNEL DURATION)

A self-contained copy of mhc_pre (package `ttnn.operations.mhc_pre.perf_experiments.prelude_off_critical_path`),
baseline = the unmodified copy of the working tree (incl. the permanent zones), variants picked by defines
(`mhc_pre_program_descriptor.PRELUDE_DEFINES`). Same precision contract for every variant (fp32_dest_acc_en=True,
default fidelity / approx). Every variant's outputs are **bit-identical** to the baseline.

## Variants
| name | defines | what |
|---|---|---|
| base | – | the op as is |
| a | PRELUDE_A | reader: non-blocking poll of the W share's trid 15 (`ncrisc_noc_read_with_transaction_id_flushed`) after every X page issue; token pushed the moment it landed |
| f | PRELUDE_FASTFILL | writer: bias fill as 16 contiguous `[b[2j], b[2j+1]] x 32` blocks (1024 straight word stores; no NoC zero fill, no slot_index scatter): 5.5 -> 1.07 us on BRISC |
| b1 | PRELUDE_B_EARLY | writer: bias DRAM reads issued at kernel start |
| b | PRELUDE_B | b1 + fill moved after block 0's partial send |
| rf | PRELUDE_B_READER + f | reader issues the bias reads behind its share on trid 15 |
| sf | PRELUDE_B_AFTER_SHARE + f | writer issues the bias reads after its own share landed |
| ab1f / abf / arf / asf | combinations | |
| +fo / +skN / +bx | descriptor knobs | share on the writer for non-flipped rows / no share on the N most congested rows / parked W_SHARE_BEFORE_X |

## Focus shapes, bf16 X / fp32 W (medians us; 5-15 interleaved calls per run, several runs)
| variant | 640x7168 | 640x1792 | 1280x4096 |
|---|---|---|---|
| base | 144.9 / 145.3 / 146.1 / 144.9 | 43.7-44.1 | 147.4-149.2 (bimodal, ~150-162 outliers) |
| a | 144.1-145.2 | 43.5-44.0 | **155.9-160.9 (regression)** |
| f (graduation.patch) | 145.3-146.1 | 43.6-43.8 | 147.5-148.4 |
| b1f | 144.7-146.3 | 43.8-43.9 | 148.3-148.6 |
| bf | 145.0 | 43.9 | 149.0 |
| **ab1f (graduation_ab1f.patch)** | **140.9-142.3** | 43.6-44.0 | 147.4-148.6 |
| abf | 139.9-142.4 | 43.6-44.1 | 147.5-150.4 |
| arf / asf | 144.5 / 145.0 | – | 147.8 / 153.5 |
| ab1f+bx / abf+fo / abf+sk2 | 144.9 / 147.5 / 139.9 | – | 160.0 / 150.8 / 160.2 |

## Domain sweep (medians us, R=5..11)
| shape | dtype | base | f | ab1f |
|---|---|---|---|---|
| 640x7168 | bf16X/fp32W | 144.9 | 145.4 | 141.5 |
| 640x1792 | bf16X/fp32W | 43.9 | 43.6 | 43.8 |
| 1280x4096 | bf16X/fp32W | 148.1 | 147.5 | 147.9 |
| 4096x1792 | bf16X/fp32W | 237.7 | 237.5 | 238.3 |
| 2048x5120 | bf16X/fp32W | 314.7 | 316.3 | 314.2 |
| 1x7168 | bf16X/fp32W | 46.2 | 46.2 | 46.1 |
| 64x4096 | bf16X/fp32W | 43.0 | 42.9 | **44.1** |
| 32x128 | bf16X/fp32W | 18.9 | **17.3** | 17.4 |
| 640x7168 | fp32X/fp32W | 414.2 | 412.7 | 407.1 |
| 640x1792 | fp32X/fp32W | 122.6 | 122.0 | **129.3** |
| 1280x4096 | fp32X/fp32W | 382.6 | 381.4 | 379.3 |
| 4096x1792 | fp32X/fp32W | 548.3 | 547.8 | **556.1** |
| 2048x5120 | fp32X/fp32W | 716.8 | 716.6 | 708.3 |
| 1x7168 | fp32X/fp32W | 121.1 | 120.3 | 120.7 |
| 64x4096 | fp32X/fp32W | 92.9 | 93.1 | 93.1 |
| 32x128 | fp32X/fp32W | 24.2 | 24.2 | 24.2 |
| 640x7168 | fp32X/bf16W | 352.7 | 353.7 | 352.7 |
| 640x1792 | fp32X/bf16W | 115.9 | 116.1 | **119.2** |
| 1280x4096 | fp32X/bf16W | 362.0 (bimodal) | 360.4 | 348.1 |
| 4096x1792 | fp32X/bf16W | 518.5 (bimodal 518/536) | 535.1 (same two modes) | 539.0 |
| 2048x5120 | fp32X/bf16W | 647.3 | 647.3 | 649.0 |
| 1x7168 | fp32X/bf16W | 80.8 | 81.3 | 80.1 |
| 64x4096 | fp32X/bf16W | 64.6 | 64.6 | 64.4 |
| 32x128 | fp32X/bf16W | 20.1 | 20.0 | 19.9 |
| 640x7168 | bf16X/bf16W | 148.1 | 147.8 | 148.4 |
| 640x1792 | bf16X/bf16W | 44.1 | 44.0 | 44.0 |
| 1280x4096 | bf16X/bf16W | 182.8 | 183.2 | 183.9 |
| 4096x1792 | bf16X/bf16W | 260.1 | 260.6 | 260.7 |
| 2048x5120 | bf16X/bf16W | 354.6 | 354.0 | 351.4 |
| 1x7168 | bf16X/bf16W | 40.9 | **40.0** | 40.0 |
| 64x4096 | bf16X/bf16W | 38.4 | **35.7** | 35.8 |
| 32x128 | bf16X/bf16W | 19.4 | **14.6** | 14.3 |

The patched real kernels (kernels_grad/ = graduation.patch, kernels_grad_ab1f/ = graduation_ab1f.patch) were
re-measured through the same bench (`+grad` / `+grad2`) and reproduce the f / ab1f numbers.

## Mechanism findings (zones in zones/)
- The coordinator's premise "the share had landed, the token was late" is only partly true. With the poll
  (r_w_land_poll zone) the share's TRUE landing time is 50-72 us on the first NoC0 rows below the flip (y6-y8) at
  640x7168 (27-63 us at 1280x4096): their DRAM responses share the most-loaded column links with the X burst.
  Token lateness was ~10 us; the W all-gather end moved 90 -> ~80 us only.
- (a) alone regresses 1280x4096 because the W exchange then finishes during the X peak and the writer's
  (baseline-position) bias read queues behind the burst (w_bias 18.6 us on the critical core -> late partial ->
  late root S).
- The bias fill itself cost 5.5 us of BRISC time (scattered slot_index stores + NoC zero fill); the block-store
  fill is 1.07 us and is on the critical path on small shapes (bf16 W 32x128 -25%).
- Issuing the bias reads at kernel start (b1) hurts fp32 X: all cores' bias reads hit the ONE bank holding the
  bias tile at t=0 and delay every core's W share read from that bank (fp32X 640x1792: r_w_land max 9 -> 20 us,
  w_w_recv max 9.5 -> 19.9 us). Moving them behind the share (reader trid 15 / after the share token) removes
  the fp32 regression partly but also the 640x7168 gain.

## Files
- kernels/ (bench kernels with all defines), mhc_pre_program_descriptor.py (PRELUDE_* knobs), test_prelude.py
  (import-safe pytest shim) -> bench/prelude_bench.py (test_correctness: golden + bit-identical-to-base,
  alternating seeds; test_perf: interleaved A/B), bench/perf.sh, bench/zone_cores.py (per-core zone end grid).
- graduation.patch (recommended, fast fill), graduation_ab1f.patch (option: poll + early bias + fast fill).

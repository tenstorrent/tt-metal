# q_stem changelog

## Phase 0 (implementer)

Built per op_design.md R1/R2: one generic_op dispatch per call; x injector + Mcast to a column-split stage-1 rect;
q stays in L1 (cb_q_slice); port cores (row 0, 2 x num_links) pull remote head-rows and send them over FABRIC_2D
(2D node routes, ready fence, fused arrival increments) into the L1 landing scratch; own quarter delivered over the
NoC; stage 3 per head (matmul_block + eltwise copy), TILE output.

Deviations from the design (advisory items, recorded):
- Port placement: the centred row-0 layout of fabric_ring_exchange, not the Ethernet probe (the probe is an extra
  dispatch on first call). Lamp for later.
- Arrival counters count tiles; one fused increment per (destination core, row) group of segments (the design's
  `inc_every_packets` knob collapses to "per row" — the exchange measured hidden: ablating its payload saves 5 us).
- Raw fabric API on the port connection (FabricStream cannot express node-addressed 2D routes; one connection carries
  both 1-hop and 2-hop destinations).
- Injector split into a DRAM-read RISC (RISCV_0/NoC1) and a multicast RISC (RISCV_1/NoC0).
- Stage-1 rect sized to 2 columns per core (lamp L7) and load-weighted head placement (lamp L8); remote heads are
  consumed row-major across heads (all remote heads land with the same row profile), local heads head-major.
- s3_block_m_tiles is pinned to 1 (host assert): the landing row layout interleaves nope | rope per row.

Perf (QuietBox 2, 1x4 ring, Kimi K3.1 Phase-0 config: bf16, HiFi4, fp32 DEST; eval perf_measure.py, 5 experiments;
roofline 90.6 us, target 120.8 us):

| Version | fastest chip | slowest chip |
|---------|--------------|--------------|
| first passing build (99-core stage 1, head-major) | 223.7 us | 232.5 us |
| L7 + L8 + row-major remote heads + split injector + x depth 3 + flushed output pops | 190.9 us | 206.2 us |

Ablation breakdown of the first build (device kernel time, probe test, QSTEM_ABLATE defines): output writes -45 us,
x read+multicast -45 us, q_b read -38 us, wkv read -32 us, stage-1 matmul -18 us, fabric payload -5 us; every payload
stubbed at once: 131 us; plus q_b/wkv/fabric: 61 us; plus x multicast: 42 us (sync/launch floor).

Current breakdown (device zones S1/S3 on compute): q_b prologue ~31 us, stage 1 ends ~139 us (2-column cores are
compute-bound at HiFi4 + x supply), stage-3 tail to ~180 us = the stage-1 cores' heads after stage 1, bound by DRAM
output writes (ablating them: ~161 us). Next levers (not built): interleave stage-3 rows into the stage-1 compute loop
so the stage-1 cores' output writes overlap stage 1; K-split stage 1 (R6); a cheaper stage-3 row (per-row matmul
init + fp32 pack overheads).

## Phase 0 — Verification (incremental-verifier)
- **Date**: 2026-11-24
- **What was done**: code review, acceptance + golden suite + verifier CLI, precision baseline, per-stage perf
  ablation, refinement queue (`op_requirements.md`), `verification_report.md`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], q_b_proj_dtype=[bfloat16], wkv_b1_dtype=[bfloat16], layout=[TILE],
  output_layout=[TILE], fp32_dest_acc_en=[True], rope_mode=[nope], alignment=[tile_aligned], input_buffer=[DRAM],
  memory_layout=[INTERLEAVED], cluster_axis=[1], topology=[Ring], num_links=[2]
- **Accuracy achieved**: PCC=0.999997, max_abs_err=0.0197, mean_abs_err=0.0018, rel_rms_err=0.0023 (Kimi K3.1
  shape; 4 shapes × 4 devices measured via test_q_stem_precision_baseline.py, all within 0.999997–0.999998 /
  rel RMS 0.0021–0.0023; got/true ratio median 1.0000)
- **Golden suite at Phase 0**: 15 supported_pass / 0 supported_fail / 1077 xfail_expected / 0 xpass_drift /
  0 xfail_wrong_mode (1103 collected; `verifier_report.json`)
- **Perf**: Phase-0 Kimi loose case 190.5 µs fastest / 207.3 µs slowest chip (roofline 90.6, target 120.8);
  fabric payload fully hidden; output writes −22.7 µs; stage 3 + out + wkv −53.1 µs (ablation)
- **Issues encountered / fixes**: added the missing host `fits_l1` check (`check_fits_l1`, RuntimeError naming the
  term); `QB_READ_BATCH` and `MAX_HEADERS` routed host → CT (were duplicated kernel literals); reader CT list built
  per core set instead of a magic-index patch; `_crs` merges core ranges into rectangles; latent buffer type added
  to `custom_program_hash` (pre-empts aliasing once input_buffer=L1 lands); `QSTEM_ABLATE` documented as the
  perf-breakdown hook; `l1_ledger.md` totals re-evaluated from the predicate. `--dev` mode cannot open the fabric
  mesh on this board (ACTIVE_ETH kernel config overflow) — tests ran in default mode.
- **Tests added**: test_q_stem_precision_baseline.py

## Refinement 1 — Numerical configurability + L1 latent (weights bfloat8_b, 16-bit DEST, input_buffer L1)
- Date: 2026-11-24
- What was done: `SUPPORTED` widened — `q_b_proj_dtype` / `wkv_b1_dtype` += bfloat8_b (independently),
  `fp32_dest_acc_en` += False, `input_buffer` += L1. No kernel change: the weight CBs already took the weight's
  page format (`Geometry.qb_tile_bytes` / `wkv_tile_bytes`), the matmul helpers reconfigure unpack/pack formats
  between stage 1 (x @ bfp8 q_b) and stage 3 (landing @ wkv), and the injector's `TensorAccessor` handles an L1
  interleaved latent from its accessor args. The one descriptor change: `cb_q_partial` (stage-1 K-block partials)
  now follows DEST width (`Geometry.partial_dtype`: Float32 under fp32 DEST, bf16 under 16-bit DEST) in both the
  CB and `l1_terms`. `cb_q_slice` stays bf16 under both (exchanged payload). `custom_program_hash` already keyed on
  weight dtypes, `fp32_dest_acc_en`, and the latent buffer type — nothing to add. DEST limit is derived
  (`_dest_limit`: 8 tiles at 16-bit) so stage-3 `sbw3 = largest divisor of s3n <= 8` widens to 8 (s3n = 16 at
  N=512) and stage-1 `sbw1` widens likewise. Grid fill / batching identical to Phase 0.
  Reused: every CB, kernel, helper call and the existing format derivations. Added: `partial_dtype` (one field).
- Accuracy achieved (HiFi2, bfp8 q_b, L1 latent, vs a reference on the device-rounded weights):
  Kimi (1,640,1536) 16-bit DEST: PCC=0.99977, rel RMS=0.028; fp32 DEST: PCC=0.99999, rel RMS=0.0070.
  K-blocked (1,128,4096) 16-bit DEST: PCC=0.99920, rel RMS=0.062; fp32 DEST: PCC=0.99999, rel RMS=0.0072.
  wkv_b1 bfp8 changes nothing beyond the 4th digit.
- Golden test progress: targeted slice 36/36 passing (16 new-axis combos x {1x1x128x64, Kimi 640x1536} +
  1x1x2048x1536 L1-latent 16-bit-DEST x 4 weight-dtype combos); unit suite test_q_stem.py 9/9.
- Issues encountered: under 16-bit DEST the median got/true ratio sits above 1 and grows with K (1.018 at K=1536,
  1.046 at K=4096) with a broad p5..p95 band (0.80..1.29). Stock `ttnn.matmul` with the identical config shows
  the same (1.019 / 1.049, rel RMS 0.028 / 0.062 — `probes/probe_003.py`), so it is the 16-bit DEST accumulation,
  not an op scale bug. No EXCLUSIONS needed.
- Tests added: `test_q_stem_model_config_precision` in test_q_stem_precision_baseline.py (bfp8 q_b ± bfp8 wkv,
  HiFi2, 16-/32-bit DEST, L1 latent; Kimi + K-blocked shapes; ratio-band scale-bug check for 16-bit DEST).

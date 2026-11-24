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

## Refinement 2 — Row-major query output (in-kernel untilize + stick writer)
- Date: 2026-11-24
- What was done: `SUPPORTED["output_layout"]` += ROW_MAJOR. The output tensor is allocated ROW_MAJOR
  (`run_q_stem(output_layout=…)`). Stage 3 gains `untilize_row_block`: after each prefix N-block's matmul,
  `untilize<s3n, cb_prefix_out, cb_prefix_rm>(1)`; after the rope producer (NoPE copy),
  `untilize<Dr_t, cb_rope_out, cb_rope_rm>(1)` — a separate block op behind the producers, so a rotated rope tile
  (Refinement 5) and realigned rows (Refinement 4) slot in front of it unchanged. The writer's `store_row_block`
  gains a ROW_MAJOR branch: each untilized block's 32 sticks go straight to DRAM as segments at their byte offset
  in the (N+Dr)·2 B output page (`TensorAccessor` over the RM output, `get_noc_addr(stick, col_byte)`): prefix
  segment(s) of `s3n`·64 B at `nb·s3n·64`, rope segment of `Dr_t`·64 B at `N·2` — 32 writes per block, one
  `noc_async_writes_flushed` per block (not per stick), one write barrier at the end. No `ttnn.to_layout` /
  untilize op anywhere: one dispatch. Under ROW_MAJOR the tiled `cb_prefix_out` / `cb_rope_out` are
  compute-internal and shrink to one block (`Geometry.out_tile_depth`); the RM pair takes `OUT_DEPTH`.
  `output_layout` added to `custom_program_hash`; `l1_terms` / `l1_ledger.md` carry the RM CBs.
  Reused: the stage-3 matmul/copy producers, `store_row_block` loop order, `OUT_DEPTH`, the output accessor.
  Added: `cb_prefix_rm` / `cb_rope_rm`, two `untilize` helper calls, `store_sticks` (writer), 3 CT args each side.
- Accuracy achieved: unchanged from the TILE path (untilize and the stick writes are exact copies); every
  ROW_MAJOR cell run passes the golden `check_output` at its tolerance (incl. PERF FOCUS loose case, HiFi2 / bfp8
  q_b / 16-bit DEST / L1 latent).
- Perf (PERF FOCUS, `perf_measure.py`, 5 experiments): fastest chip **167.4 µs** (sem 0.08), slowest 177.2 µs,
  spread 9.8 µs, not dispatch-bound (gap ~0.58 µs). Saved as `agent_logs/perf_r2_focus_rowmajor.json` — the
  `ccl_perf compare` base for Refinement 3. (Phase-0 TILE / HiFi4 / fp32 / DRAM-latent Kimi was 190.5 µs; not the
  same config, so not a like-for-like delta.)
- Golden test progress: 25/25 targeted ROW_MAJOR cells passing (11: Kimi / one-tile / B=2 / K-blocked Q=4096 /
  streamed-wkv N=4096 × fp32 DEST on/off + the PERF FOCUS loose case; 14: every shape at bfp8 q_b + bfp8 wkv +
  16-bit DEST + L1 latent). Unit suite test_q_stem.py 9/9 (TILE path regression).
- Issues encountered: None.
- Tests added: none (the golden ROW_MAJOR cells cover the path; the TILE unit suite is the regression net).

## Refinement 3 — OVERLAP: hide the stage-3 tail of the PERF FOCUS behind stage 1
- Date: 2026-11-24
- **Breakdown re-taken on the PERF FOCUS** (`perf_measure.py` protocol; ablations via the new
  `tests/ttnn/unit_tests/operations/q_stem/perf_ablate.py`, same protocol with the golden check skipped under
  `QSTEM_ABLATE`; 3 experiments, fastest chip): full **167.5**; OUT_WRITE stubbed 130.4 (−37); S3,OUT_WRITE,WKV_READ
  120.6 (−47); S3 compute alone 164.9 (−2.6); SEND_PAYLOAD 152.7 (−15, **the exchange is no longer fully hidden in
  this config**). Per-core timeline (`zone_probe.py` + `zone_summary.py`, device zones, 1 call): stage 1 ends
  ~99–107 µs, the call ~165–170; the cores *outside* the stage-1 rect finished last.
- **Calculator / prediction (written before measuring)**: the op model does not express stage 3 and `estimate.py`
  refuses the port pull (`l1_read`, recorded in `dm_estimates.json`, phase "Refinement 3"). Hand estimate from the
  zones: the busiest port moves 24 packets per block → ~5.6 µs/block at ~235 ns/packet (fabric_all_gather report)
  vs ~8.5 µs measured, so batching its pull barrier should move the last landing ~130 → ~95–100 µs. **Device: flat**
  (the fabric link, not the pull, binds the busiest port — PSB per block stayed ~8–9 µs). Prediction wrong; recorded.
- What was done (all knobs, single source in `q_stem_program_descriptor.py`):
  1. `RM_FULL_STICK` (default True; ROW_MAJOR + one N-block per row): the rope copy packs behind the prefix in
     `cb_prefix_out`, one `untilize<s3n + Dr_t>` per row, and the writer writes each 1152 B output stick in ONE NoC
     write (was 1024 B + 128 B segments). Measured 167.4 → 164.1 (5 exp); off vs on in the final tree 158.7 vs 154.4.
  2. Port pull batching: the port reader issues up to `PORT_BATCH_SEGMENTS` segments of reads per barrier (the knob
     already sized `cb_port_stage` to 2 batches but only the sender used it); batches never wrap past the CB end.
     Measured neutral (batch 1: 154.0 vs 4: 154.4) — the busiest fabric link binds the port. Kept (correct, live).
  3. `FREE_WRITER_NOC` per region: the stage-3-only cores' output writer moves to NoC0 (reader to NoC1) for the
     columns right of the stage-1 rect; the rows below stay on NoC1. Zones showed a far-east free core's 32 stick
     writes + flush taking ~45 µs while stage 1 / the stage-1 cores' tail held NoC1 (compute back-pressured, its
     rows long landed). all-NoC1 163.9 / all-NoC0 152.1 / right-0 + below-1 **147.7** / right-1 + below-0 153.6.
  4. `S3_LATE_REMOTE_ROWS` (shared `q_stem::RowOrder` consumption order = landing layout, host twin
     `landing_slot` with a per-row jump): consume the last k remote rows after the local heads. **Parked at 0**
     (byte-identical order): k=1 measured 154.4 vs 151.9 at k=0, k=2 worse. The reader / compute / writer loops now
     all walk `RowOrder` (one definition instead of three hand-written two-phase loops).
  5. `S1_WRITER_NOC` (stage-1 rect writer on NoC0, reader + x-multicast receive on NoC1): correct, 227.7 µs —
     parked at 1 (default).
  Placement knobs re-swept on this config: `S3_COST_FACTOR` 2.0 / 2.5 / 3.5 tie (151.8–151.9), 1.3 158.7 (slowest
  217.9), `FREE_CORE_START` 0.6 168.0, `S1_COLS_PER_CORE` 3 165.6 — kept at 2.5 / 0.35 / 2.
- **Result** (`perf_measure.py`, 5 experiments; `agent_logs/perf_r3_focus.json`): fastest chip **167.43 → 147.66 µs**
  (−19.8, noise 0.38), slowest 177.22 → 161.28 (−15.9); `ccl_perf compare` verdict **win**. Spread 13.6 µs; not
  dispatch-bound (max gap 0.58 µs).
- Measurement pitfall found: with ~8 temporary per-row/per-block zones on every RISC, the harness reported 133.9 µs
  for a tree that measures 154.4 without them; no single zone set reproduced it — the per-RISC marker buffer
  overflows over 500 traced calls and the duration markers go bad. Headline numbers here are zone-free.
- Accuracy achieved: unchanged paths are exact copies (full-stick untilize + whole-page writes); every golden cell
  run passes `check_output` at its tolerance, incl. the PERF FOCUS (HiFi2 / bfp8 q_b / 16-bit DEST / L1 / RM).
- Golden test progress: targeted slice 30/30 (every tile-aligned shape × {RM + bfp8 + 16-bit DEST + L1,
  TILE + bf16 + fp32 + DRAM} incl. R2 streamed wkv N=4096 and K-blocked Q=4096, + the 4 loose cases);
  test_regression.py 8 passed / 3 xfailed (unchanged); unit test_q_stem.py 9/9.
- Issues encountered: `estimate.py` refuses L1→L1 reads (the port pull). The S3_LATE_REMOTE_ROWS > 0 path was
  checked against golden on the PERF FOCUS only (it is parked at 0).
- Remaining headroom (finding, not a queue item): stage 1 still ends ~100–130 µs per chip and the stage-1 cores'
  one-head tail runs ~30 µs past it on NoC1; the busiest fabric link (the antipodal quarter routed one way) lands
  the last remote rows ~25–35 µs after stage 1 — that is TRANSPORT (R4 relay / direction balance of the antipodal
  quarter) and now visible in the SEND_PAYLOAD ablation (−15 µs). Next OVERLAP lever: interleave landed stage-3
  rows into the stage-1 loop on the stage-1 cores (L2), not built here.
- Tests added: `perf_ablate.py`, `zone_probe.py`, `zone_summary.py` (tools, no `test_` prefix → not collected).

## Refinement 4 — Rows per device that are not whole tiles (m_non_aligned)
- Date: 2026-11-24
- What was done: `SUPPORTED["alignment"]` += `m_non_aligned` (regime R9). Every stage now works in **quarter-aligned row
  tiles**: `Geometry.qrt = ceil(Sq/32)` (Sq = S/G), row tile r of quarter d = latent rows `d*Sq + 32r .. +31` of the image.
  The x injector's DRAM half realigns them (`x_realign_block` in `q_stem_injector_reader.cpp`): when `g0 = d*Sq + 32r` is
  not a tile boundary it reads the two latent tiles the rows straddle (Ta, Tb; Tb past the image -> Ta, padding rows only)
  into the new injector-only scratch `cb_x_stage` (2·s1_kb pages, allocated only when non-aligned) and assembles each
  slot tile from face-row runs (<= 4 runs per column half, local L1->L1 NoC reads — the DRAM read alignment forbids the
  shift in flight). Aligned row tiles (delta = 0, incl. every tile-aligned shape) keep the direct read. The RM writer writes
  only the `min(32, Sq - 32r)` valid sticks of a quarter's last row tile at stick `(b*H + h)*Sq + 32r`; TILE writes the
  padded tile as before. Nothing else changed: stage 1, exchange, landing layout, arrival thresholds and stage 3 all
  run on `qrt` as before.
  **Deviation from the design** (recorded in op_design.md R9): the plan put a 0/1 row-shift matmul at the destination,
  before stage 3 (ceil-covering 2 tile rows on the fabric, landing / thresholds / output mapping reshaped). Applying the same
  row permutation to x before stage 1 is legal by the same algebra (`(P x) @ q_b = P (x @ q_b)`), keeps q bit-identical
  (exact copies, unchanged per-tile K accumulation), sends `ceil(Sq/32)` tile rows per quarter (<= the plan's), and leaves
  every downstream stage — and Refinement 5's cos/sin rows, which follow the output rows — untouched. Cost: stage 1
  computes `B*G*ceil(Sq/32)` row tiles instead of `B*ceil(S/32)` (Kimi S=160: 8 vs 5) and shifted x tiles are read twice.
  None of the m_non_aligned shapes is a perf case.
  Reused: the whole pipeline, `qrt`/`for_each_block`, the injector's read loop, `store_sticks`. Added: `Geometry.Sq/x_St/aligned`,
  `cb_x_stage`, the realign branch (injector), the valid-stick count (writer), 4 CT args.
- Accuracy achieved: the realignment is an exact copy, so accuracy is that of the aligned path at the same config; every
  m_non_aligned cell passes the golden `check_output` at its tolerance (S=48 / 96 / 160; bf16 + bfp8 weights, fp32 / 16-bit
  DEST, TILE / ROW_MAJOR, DRAM / L1 latent).
- Golden test progress: 96/96 m_non_aligned NoPE cells (3 shapes x 32 configs; the rope cells stay xfail until Refinement 5).
  Regression: test_q_stem.py 9/9, test_regression.py 8 passed / 3 xfailed (unchanged), 58 aligned NoPE cells incl. the
  NoPE loose cases (PERF FOCUS + Phase-0 Kimi), R2 streamed wkv and B=2 bfp8. PERF FOCUS fastest chip 147.9 us
  (R3: 147.7, noise).
- Issues encountered: None.
- Tests added: none (the 96 golden m_non_aligned cells pin the path).

## Refinement 5 — Rotary embedding (rope_mode = rope)
- Date: 2026-11-24
- What was done: `"rope"` added to `SUPPORTED["rope_mode"]`; `cos` / `sin` now reach the device program (they were
  validated but dropped before). Scheme addition inside the existing stage-3 nest, as designed:
  - Reader (`rope_tables_block`): on every core holding heads, generates the 32x32 rotate_half_meta trans tile in L1
    (NoC zero-fill via `dataflow_kernel_lib::zero_tile` + its 32 ±1 entries — no host tensor), and reads the
    quarter's cos / sin rows (`qrt × Dr_t` tiles each, one barrier) at the first landed row, so free cores do not
    compete with the stage-1 q_b prologue. Resident for the whole stage 3.
  - Compute (`rope_block`): `matmul_block` rope (`Dr_t × 1`) @ trans (`NoWaitNoPop` in1, rope retained in
    `cb_landing`) → `cb_rope_tmp`, then ONE fused `eltwise_chain` per row: `BinaryFpu Mul(landing, cos)`→D0,
    `BinaryFpu Mul(rope_tmp, sin)`→D1, `AddBinary` D0+=D1, pack → `cb_rope_out` (or behind the prefix in
    `cb_prefix_out` under the ROW_MAJOR full stick, so Refinement 2/3's untilize runs unchanged). cos/sin tile =
    `(t mod qrt)·Dr_t + i` via `TileAddressing::Offset` (caller-managed wait/pop around the loop), so batches reuse
    the device's rows and m_non_aligned rows need no shift (Refinement 4 made every stage quarter-aligned).
  - Host: `CB_COS/SIN/TRANS/ROPE_TMP` (12–15) on the compute rect only under RoPE; rope flags via named CT args
    (`rope`, `cb_*`, `qrt`, `sbh_rot`); cos/sin accessor args appended to the reader's CT args (wkv's as a stand-in
    under NoPE); cos/sin addresses as reader RT args; `l1_terms` counts the new CBs; rope in `custom_program_hash`;
    cos/sin passed to `generic_op`. `_check_caller` requires bf16 TILE interleaved tables.
  - Deviation from op_design: `* sin` / `+ rope*cos` as one chain (two FPU muls into D0/D1 + SFPU add) instead of
    separate `mul` / `add` convenience passes — same helpers, one DEST window, no product CB.
  - Reused: stage-3 loop nest, landing layout, `cb_rope_out` / full-stick path, untilize, writer (unchanged), ports,
    injector. Added: one reader block op, one compute block op branch, 4 CBs.
- Accuracy achieved (test_q_stem_precision_baseline.py::test_q_stem_rope_precision, HiFi4, base_position 4096, per
  part): fp32 DEST — rope PCC 0.999997, rel RMS 0.0024 on all 4 shapes [(1,128,256,Hl4,64,64,128),
  (1,160,512,4,128,64,512) m_non_aligned, (1,640,1536,32,128,64,512) DSV3.2, (1,128,512,4,128,128,512) wide rope];
  same as the nope part (the floor is q's bf16 rounding). 16-bit DEST — rope PCC 0.99973–0.99996, rel RMS
  0.009–0.032 (tracks the nope part; dominated by stage-1 16-bit accumulation). `test_rope_pairing` is exact.
- Golden test progress: all 546 `rope_mode=rope` golden cells pass (TILE/RM out × fp32/16-bit DEST × weight dtypes ×
  latent DRAM/L1 × aligned/non-aligned, incl. LOOSE_CASES[2] DSV3.2 and [3] GLM-5.3); 0 failed, 0 XPASS drift.
  Regression: `test_rope_pairing`, `test_rope_positions[pos0, pos65632]` (previously xfail) pass; all 11 regression
  tests and the 9 acceptance tests pass.
- Perf (not a perf refinement; recorded as the guard-set baseline, eval `ccl_perf`, 5 experiments): DSV3.2
  (LOOSE_CASES[2]) fastest chip 227.8 µs / slowest 232.7 (roofline 98.5, target 131.3); GLM-5.3 (LOOSE_CASES[3])
  fastest 154.0 / slowest 171.3 (roofline 65.1, target 86.8). Not dispatch-bound. Known lamp: cos/sin are read
  once per head-owning core (~4 MB DRAM per chip on DSV3.2, as designed); a one-reader multicast would make it 1×.
- Issues encountered: None (first device run correct).
- Tests added: `test_q_stem_precision_baseline.py::test_q_stem_rope_precision` (4 shapes × 2 DEST widths, nope /
  rope reported separately).

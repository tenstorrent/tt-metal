# Verification Report: all_gather_rms_norm

Box: Blackhole QuietBox 2 (4 chips, 2×2 system mesh; perf / ring cases on the 1×4 snake ring under
FABRIC_2D_TORUS_XY, 14 KiB payload). Results dir: `generated/agv/merged/` (not committed); the CLI report is copied to
`verifier_report.json` next to this file.

## Code Review

Fixed:
- **FPU normalize precision (scale bias).** The ratio check found a tight per-row scale of 0.9938 ± 0.0012 on every bf16
  layer_norm cell with fp32 DEST and 0.9971 on bf16 rms. The cells passed, but this is the scale-bug signature
  (PCC 0.99999, rel RMS 0.007 — half the tolerance spent on a bias). Localized by swapping only the normalize
  stage to the SFPU chain (bias gone), then to the cause: `DestReuseBcast` (the base op's `MulGammaRowBcast`
  generalization) moves DEST into srcA in the *current* srcA format — the chain head's cb_x (bf16 / bf8b) — so every
  dest-reuse step truncated the fp32 intermediate. Fix: the first dest-reuse step of a chain switches srcA (unpack +
  math — a math-only switch was measured ineffective) to the DEST-width format, the last one restores cb_x's
  (`SrcaIn` / `SrcaOut` template flags; 2 reconfigs per tile, not per step). Result: LN bf16 rel RMS 0.0072 →
  0.0019, rms bf16 0.0042 → 0.0019, row scale 0.9996. Cost (same-session `eval.ccl_perf compare`): indexer LN
  case 16.16 → 16.33 µs (+1 %), focus 21.24 → 21.29 µs (at the noise edge; a numerically inert variant showed the
  same +0.05). Kept: it fixes a correctness-quality defect and one borderline golden failure.
- **EXCLUSIONS re-measured** (prompt: carry a base-op exclusion only where this op measures it necessary):
  `{bfloat8_b, h_non_aligned}` dropped (241 / 241 sampled cells pass with it lifted — the norm is row-local and the H
  tail is never touched); `{bfloat8_b, w_non_aligned}` kept (238 / 238 `numerical-bug`, inf — element-addressed tail
  zero-fill / dense assembly on block-float tiles).
- **Failure classification.** `select_split`'s refusal said "no member split fits L1", which the classifier binned as
  `other`; it now carries `OOM` so these cells route as L1-capacity failures.
- **L1 / DRY** (l1_ledger.md "Verifier deltas"): dead `cb_moment_a` dropped on the LN FPU path, `cb_moment_b` sized
  `rows` there, `cb_out` capped at the member block, writer's ring-size CT arg derived from `cb_plan()` instead of a
  restated `OUT_DEPTH·unit`; duplicate `MulScalarPost` / `MulScalar` structs merged.

Reviewed, no change needed:
- One `ttnn.generic_op` per call; `validate()` is the first line; TensorAccessor everywhere; `void kernel_main()`;
  `api/dataflow/dataflow_api.h` includes; CB push/wait parity (single block per member; port stage batches
  reserve/push and wait/pop the same packet counts); global semaphores re-armed by decrement; program hash covers
  the plan; CoreRangeSets merged into rectangles (not dispatch-bound).
- Raw-API bypasses (port reader / sender, partial exchange, LN SFPU loops, dense matmul) are documented in-kernel with
  the helper mismatch; the partial exchange is the base op's symmetric unicast round (catalog `tensix_all_reduce`:
  it beats a multicast for k ≤ 8), so no `McastArgs` rewrite.

Deferred to refinements:
- k > 8 owner combine (design said built; implementer capped k at 8) → Refinement 1 (moves the OOM cells).
- 16-bit DEST layer_norm precision → Refinement 2.

Advisories:
- On the first call per mesh, a one-time Ethernet-core placement probe program runs (cached per mesh, like the global
  semaphores). Every call's semantics run in exactly one `generic_op` dispatch.
- With `run_safe_pytest.sh --dev` the watcher run passes (loose perf cases on the ring + a 14-cell 2×2 matrix sample,
  watcher-clean), but only one fabric-mesh open per process works: re-opening after any fabric traffic times out in
  the Ethernet teardown. Reproduced with `examples/fabric_all_gather` in place of this op (and a bare open/close/open
  without traffic passes), so this is environmental. Bulk suites therefore ran without `--dev`.
- `eval/golden_tests/all_gather_rms_norm/perf_measure.py` cannot write its result (`OSError: File name too long`: the
  JSON is named after the full case id); `tests/.../perf_agrms.py` (same `ccl_perf.run` protocol, short names) is used.

## Registry Conformance
- INPUT_TAGGERS (`alignment`, `rank`, both `(inputs, axes)`), SUPPORTED (every gated axis incl. num_links /
  topology / cluster_axis / norm_type / beta_mode), EXCLUSIONS (cell dicts), `validate()` (SUPPORTED per axis, then
  EXCLUSIONS; `UnsupportedAxisValue` / `ExcludedCell`) — present and wired; the entry point calls `validate()` first.
  The op file does not declare INVALID.
- SUPPORTED == TARGET on every axis; no XPASS evidence to fold in. One EXCLUSION removed on measurement (above).
- INVALID audit (`feature_spec.py`): only the gamma "none" canonicalization (both directions), `rms_norm ∧ beta`, and
  `no_gamma ∧ beta` (beta shares gamma's format axes) — all single-tensor / canonicalization entries, no cross-tensor
  coupling, nothing that is "not built yet". No bf8b+ROW_MAJOR entry is needed (input is TILE-only; gamma has no
  bf8b). Well-formed.

## Design Conformance
- Algorithm, pipeline topology and work distribution match op_design.md R1: raw slices land in compute-core L1 at the
  mirror address; ports placed by probe; balanced ring schedule; relay out of L1; waves; landing notifications; base-op
  combine; LN pivot + Chan combine; dense assembly. The focus case fills 80 of ~94 non-port cores (20 row-groups ×
  k = 4); the indexer case 20 cores (k = 1, 20 tile-rows — the whole row work).
- Blocking fidelity: block knobs (k, rows, waves, `PORT_BATCH_CHUNKS`, `INC_EVERY`, `RELAY_LAG_WAVES`, `OUT_DEPTH`,
  `OUT_UNIT_CAP`) are host constants / CT / RT args with one source (`cb_plan()` drives both the fit solve and the CB
  declarations). The one duplicated quantity (writer ring size) is fixed. `cb_x = rows·pitch` is the
  predicate-guarded resident block (`fits()` with a refusal, not an unbounded op-sized CB). Each block operation is
  block-scoped: one landing wait / one DRAM barrier per block, one exchange round per block; writer batches
  `out_unit` tiles per flush.
- Deviations: k > 8 owner combine deferred (→ Refinement 1); sharded input read through a `TensorAccessor` instead
  of zero-copy. The design gives a positive reason (`R-zero-copy own shard`: the caller's shard grid need not coincide
  with the mirror layout's own-member cores, so the read is a genuine cross-core gather, not a re-read of the core's
  own shard). It matters for L1 only on the OOM cells → Refinement 1 item 3.

## L1 Ledger Audit
- Ledger currency: the as-built table matches `cb_plan()` (all 23 CB indices); the planned table and the focus
  footprint were stale (104 KB claimed, 98,048 B actual) — annotated / corrected. Verifier changes recorded in a
  "Verifier deltas" table.
- Capacity vs live set: `cb_out` over the block on narrow members (8 pages for a 4-tile block) — fixed; dead
  `cb_moment_a` and over-sized `cb_moment_b` on the LN FPU path — fixed; `cb_gamma` / `cb_beta` FULL-tile pages for
  row-0 data (24 KB on the focus case) — reason stated (element-shift loader); folded into Refinement 1 (compact
  pages). Port cores' `cb_x` is allocated but unused there — required for address identity (stated).
- Page format vs DEST width: T_acc CBs are Float32 iff `fp32_dest_acc_en` (cb_cent, stats, partials); bf16 pivots by
  design (must be bf16-exact FPU operands); no Float32 page with 16-bit DEST, no 16-bit accumulation page with fp32
  DEST.
- Disjoint lifetime: `cb_moment_a` / `cb_moment_b` on the LN SFPU path ("kept separate for clarity") — not a mechanism
  reason; small (column-0 tiles), folded into Refinement 1.
- Bounds: `rows`, `pitch`, `k`, `n_stat`, foreign counts, `chunk_pages` bounded in the symbol table; total closed-form
  (`plan_bytes(cb_plan(...))`) and guarded by `fits()` against `cb_l1_budget(device)`.
- Data-movement budget: present and consistent with R1 (x read once from DRAM, output written once, fabric at the
  balanced-ring minimum, partials intra-chip). Cheapest-traffic split = the implemented one; alternatives are
  `rejected` / `deferred` regime rows with positive reasons.
- Per-core footprint: focus 98,048 B (cb_x 24.6 K + gamma 24.6 K + gathered 16.4 K + out 16.4 K + ≤ 4 K stats), port
  139,264 B; indexer 89,216 B. Scales with `rows·pitch` (cb_x, gamma pages, cb_cent on LN FPU) and `rows·k`
  (gathered).

## Precision Baseline

`tests/ttnn/unit_tests/operations/all_gather_rms_norm/test_all_gather_rms_norm_precision_baseline.py` — 1×4 ring,
Ring, 2 links, gamma (+ beta for LN) ROW_MAJOR, LN inputs with per-row offsets (N(0, 2²)); worst device reported.

| Config | Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio median [p5, p95] | per-row scale |
|---|---|---|---|---|---|---|---|
| bf16 rms | (1,1,32,384) | 0.999998 | 0.040 | 0.00101 | 0.00188 | 0.9993 [0.9965, 1.0021] | 0.9993 ± 0.0003 |
| bf16 rms | (1,1,640,384) | 0.999999 | 0.045 | 0.00099 | 0.00186 | 0.9993 [0.9965, 1.0021] | 0.9993 ± 0.0003 |
| bf16 rms | (1,1,128,1792) | 0.999999 | 0.041 | 0.00099 | 0.00188 | 0.9993 [0.9965, 1.0020] | 0.9993 ± 0.0003 |
| bf16 layer_norm | (1,1,640,32) | 0.999998 | 0.026 | 0.00183 | 0.00187 | 1.0000 [0.9961, 1.0038] | 0.9997 ± 0.0003 |
| bf16 layer_norm | (1,1,640,384) | 0.999998 | 0.040 | 0.00180 | 0.00191 | 0.9999 [0.9958, 1.0039] | 0.9996 ± 0.0002 |
| bf16 layer_norm | (1,1,128,1792) | 0.999998 | 0.043 | 0.00179 | 0.00192 | 0.9999 [0.9958, 1.0039] | 0.9996 ± 0.0002 |
| fp32 layer_norm | (1,1,640,384) | 1.000000 | 0.011 | 0.00056 | 0.00058 | 0.9996 [0.9986, 1.0004] | 0.9995 ± 0.0001 |
| bf8b rms | (1,1,640,384) | 0.999955 | 0.087 | 0.00745 | 0.00960 | 1.0025 [0.881, 1.135] | 1.0012 ± 0.0004 |
| bf16 layer_norm, fp32_dest_acc_en=False | (1,1,32,384) | 0.999931 | 0.167 | 0.01287 | 0.01416 | 1.0040 [0.958, 1.056] | 1.0078 ± 0.0019 |
| bf16 layer_norm, fp32_dest_acc_en=False | (1,1,128,1792) | 0.999934 | 0.275 | 0.01282 | 0.01421 | 1.0042 [0.959, 1.057] | 1.0082 ± 0.0017 |

(20 rows measured; the remaining shapes match their family. Before the verifier fix, bf16 layer_norm read rel RMS
0.0072 with row scale 0.9938 ± 0.0012 — the scale bias described above.)

**Assessment**: with fp32 DEST every family is at output-rounding level and unbiased (row scale within 0.05 % of 1
for bf16 / fp32; bf8b's +0.1 % is block-float output quantization). The 16-bit-DEST layer_norm path is the one gap:
its error decomposes into a per-row offset (μ rounded to bf16, 0.012–0.020 rms on the golden data) plus a +1 %
per-row scale (16-bit variance accumulation), with a 0.003 residual — genuine precision, not a scale bug.
16-bit-DEST rms meets its tolerance (rel RMS 0.0047).

**Recommended tolerances**: defaults hold (`helpers.TOLERANCES`); the (bf16, False) layer_norm cells need
Refinement 2, not a looser tolerance.

## Verifier CLI Summary

Golden coverage: the full matrix is 39,312 run cells (≈ 0.6 s each, ≈ 6.5 h), so the verifier ran six disjoint
CRC32-hash slices (≈ 1/80 of the matrix each; every shape and axis value represented), plus every ring-mock,
regression and loose test (162, all pass), plus targeted re-runs (the bf8b exclusions, the borderline LN 1×384 family:
108 / 108 pass after the fix).

- supported_pass: 2357
- xfail_expected: 644 (the two EXCLUSIONS)
- invalid_skipped: 13999
- infeasible_skipped: 55 (harness shard geometry does not fit L1)
- supported_fail: 174 — 143 `numerical-precision` (all bf16 ∧ fp32_dest_acc_en=False ∧ layer_norm → Refinement 2),
  31 `OOM` (`select_split` finds no k ≤ 8 that fits: L1-sharded wide / decode shapes and 128×8192 LN interleaved →
  Refinement 1)
- xpass_drift: 0
- xfail_wrong_mode: 0
- One harness artifact seen once and not reproduced on re-run: `L1ProfilingError peak_total=18446744073709541120`
  (graph-capture underflow) on a bf8b LN cell.

## Recommendations
- Queue (op_requirements.md): R1 L1 fit (base-op owner combine k > 8 + compact gamma) → R2 16-bit LN precision →
  R3 TRANSPORT focus (link at ≈ 62 %) → R4 OVERLAP focus (5.8 µs tail, 5 µs start) → R5 indexer LN local tail +
  own-slice overlap → R6 indexer TRANSPORT (1-tile packets). The perf-flagged configs are already fully SUPPORTED and
  passing, so no unlock precedes the perf slots beyond the from-base port.
- Documented omissions (TARGET cells under EXCLUSIONS, not queued): `{float32, fp32_dest_acc_en=False}` — precision
  convention (fp32 input requires fp32 DEST; `references/precision_convention.md`, TARGET comment);
  `{bfloat8_b, w_non_aligned}` — block-float tiles cannot take the element-addressed slice-tail zero fill / dense
  assembly (base op excludes it too; measured 238 / 238 inf). Candidate if the queue empties: a bf8b-aware tail
  mask (zero the mantissa bytes past v in the data section, exponents untouched) + bf8b dense assembly output.
- L1 pressure: the gathered output of an L1-sharded call is G× the input shard; with both resident the CB budget on
  wide shapes drops to tens of KB (measured 19–68 KB). Refinement 1's larger k is the first lever; zero-copy shard
  backing the second.

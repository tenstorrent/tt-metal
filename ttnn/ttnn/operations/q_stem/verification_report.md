# Verification Report: q_stem

Phase 0 as built: one `ttnn.generic_op` dispatch per call. Per chip, 4 port cores and an x injector sit on row 0.
Stage 1 is a column-split `x @ q_b[p]` (x multicast) with q held in L1. Port cores pull the remote quarters and send
them over FABRIC_2D into an L1 landing scratch (2D node routes, ready fence, fused arrival increments). The own
quarter is delivered over the NoC. Stage 3 runs per head (`matmul_block` with wkv_b1 + rope copy) into a TILE
output. Board: QuietBox 2 (Blackhole), 1x4 snake ring, FABRIC_2D_TORUS_XY, 2 links.

**Test-mode note.** The task asks for `run_safe_pytest.sh --dev`. Under `--dev` every mesh open fails:
`TT_FATAL: Program size (26864) too large for kernel config buffer (26624) on ACTIVE_ETH`. The watcher-instrumented
fabric router binary does not fit the Ethernet core, so no fabric test can run in dev mode on this board. All
results below come from default mode (`run_safe_pytest.sh` / `tt-probe.sh` / `eval_test_runner.sh`).

## Code Review

Fixed in place:

| # | Finding | Fix |
|---|---------|-----|
| 1 | **`fits_l1` missing.** The design ("host predicate `fits_l1`; RuntimeError naming the term") and `l1_ledger.md` ("checked on host before building") both claim an L1-fit check; the code had none. An over-budget shape would fail at program build with an allocator error instead of a named term. | Added `cb_l1_budget(mesh)` (lowest allocated L1 address, so it accounts for the landing scratch and an L1 latent), `l1_terms(geo, wkv_depth)` per core role (stage-1 / stage-3-only / port), and `check_fits_l1`, which raises RuntimeError naming the role and the largest term. It is called before any program is built. |
| 2 | **DRY: the q_b read batch was a kernel literal** (`qb_batch = 8` in `q_stem_reader.cpp`), a dm-shortlist knob (lamp L10) with no host source. | New host knob `QB_READ_BATCH`, passed as a reader compile-time arg. |
| 3 | **DRY: the packet-header pool cap was restated** (`MAX_HEADERS = 12` on host for `hdr_per_dest` sizing, `max_headers = 12` in `q_stem_port_sender.cpp`). | The sender reads it as a compile-time arg from `MAX_HEADERS`. |
| 4 | **Fragile CT patch.** The reader CT list was built, then index 16 overwritten per core set with a magic index. | `reader_ct_for(wkv_units)` builds each core set's list explicitly. |
| 5 | **Unmerged core sets.** `_crs()` built one `CoreRange` per core (the port set); the dispatch note in the task flags this pattern. | `_crs()` now merges into rectangles (and replaces the removed `_crs_list`). Measured: not dispatch-bound before or after (max launch gap 0.59 µs); perf neutral (`ccl_perf compare`: null). |
| 6 | **Program-cache aliasing ahead of Refinement 1.** `custom_program_hash` did not include the latent's buffer type, but the injector's `TensorAccessorArgs` differ for an L1 latent. Once `input_buffer=L1` is supported, a DRAM-latent and an L1-latent call of the same shape would share a cached program. | Added `x.memory_config().buffer_type` to the hash key. |
| 7 | The `QSTEM_ABLATE` stage stubs were labelled "TEMP". | Kept on purpose and documented as the perf-breakdown hook: it is in the hash, off by default, and every perf refinement in the queue uses it for the per-stage breakdown. |

Reviewed with no change needed:
- **Helpers.** Stage-1 and stage-3 matmuls use `matmul_block`; the NoPE copy uses `copy<>`; the x multicast uses the
  `ttnn.Mcast` / `McastArgs` helper. The raw fabric API on the port sender is justified: `FabricStream` builds 1-D
  hop-count routes only, and the antipodal destination on the snake ring needs a 2D node route (design § Helpers
  considered and rejected). TensorAccessor everywhere, `void kernel_main()`, `api/dataflow/...` include paths.
- **CB sync.** Push and wait counts match per CB: `cb_q_slice` grows per block and is popped once by the writer;
  `cb_landing` gets credit pushes per row and two pops (Dn_t + Dr_t) per row; `cb_wkv` is R1 head-resident or R2
  per-unit; `cb_port_stage` pushes per segment and pops per batch.
- **Broadcast.** Phase 0 has no binary op.

Design conformance:
- **Algorithm, pipeline topology, inter-core and inter-chip communication** match `op_design.md` (R1/R2 built,
  regime-pinned tests in the acceptance file). Implementer deviations are recorded in `changelog.md`: centred port
  placement instead of the Ethernet probe, arrival increments per row instead of `inc_every_packets`, and a split
  injector. All are TRANSPORT lamps on an exchange that measures as fully hidden (below).
- **Parallelization.** It fills the machine: row-0 ports and injector, a 72-core stage-1 rect (Kimi), and heads
  over all 99 compute cores. Both dataflow halves are batched: q_b reads batched per `QB_READ_BATCH`, a barrier per
  x round, output writes flushed per row block.
- **Blocking-model fidelity.** `S1_BLOCK_M_TILES`, `X_ROUND_*`, `S3_BLOCK_N_CAP`, `WKV_*`, `OUT_DEPTH` and
  `PORT_BATCH_SEGMENTS` are each single-source host knobs flowing through `Geometry` into CB sizes and CT args.
  **One collapsed knob remains:** `S3_BLOCK_M_TILES` is pinned to 1 by a host assert, because the landing row
  layout interleaves `[nope | rope]` per row. That forecloses the design's lamp L2 (M > 1 stage-3 blocks). Fixing
  it is a landing-layout rework that touches the exchange's slot math, so it is filed as a lever of Refinement 3
  (stage-3 OVERLAP), not done in this pass.
- **Expression.** Every scheduling boundary matches the planned extent (stage-1 block = 1 row tile, segment = one
  head-row, stage-3 block = 1 row). No reader or writer handshakes below its block.
- **Whole-op-dimension CBs.** `cb_q_slice` (all row blocks) and `cb_landing` (every row a core receives) are
  resident by design: there is no ack path over the fabric, and q never touches DRAM. They are bounded by the new
  `check_fits_l1`. The streaming fallback is deferred regime R3, with a positive reason: every INPUTS shape fits
  with ≥ 307 KiB headroom.

## Prompt Rules (eval/prompts/q_stem.txt `## Rules`)

| Rule | Status |
|------|--------|
| ONE fused native implementation, no composite ops (MUST) | ✓ one `generic_op`; no matmul / CCL / rotary / concat ops called |
| Trace-capturable, no state from a previous call, deterministic (MUST) | ✓ perf harness runs traced; `test_back_to_back`, `test_deterministic` and the acceptance back-to-back test pass. Advisory: the persistent resources (global semaphores, landing scratch) are created on the first call with a `synchronize_device`, and the scratch is **grown** (deallocated and reallocated) when a later call needs more landing rows. Inside a trace capture that regrowth would be an in-call allocation. Today a model calls one shape per mesh and the first call is outside the trace, so this is not a violation, but sizing the scratch for the mesh's largest call up front would remove the hazard. |
| Fabric: caller's config, honour num_links / cluster_axis, packet size from the active max payload (MUST) | ✓ `pkt_tiles` from `get_tt_fabric_max_payload_size_bytes()`; `test_small_packet_size` (4160 B) passes |
| ROW_MAJOR written by the op's own kernels (MUST) | not yet applicable (output_layout=ROW_MAJOR is Refinement 2; its entry forbids the wrapper) |

## Registry Conformance

- `INPUT_TAGGERS = {"alignment": tag_alignment}` with signature `(inputs, axes)`. It returns `m_non_aligned` when
  `S % 32`, `S % G` or `(S/G) % 32` is non-zero; the `S % G` term is a caller error caught earlier and is harmless.
- `SUPPORTED` lists all 13 axes the prompt names. `EXCLUSIONS = []`. `validate()` checks caller errors (ValueError),
  then SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`). `q_stem()` calls `validate()`
  first. `default_compute_kernel_config()` is the single source of the default (the golden tagger reads it).
- The op file declares no `INVALID`. Link-count / wrap-link checks (caller errors) are raised from `exchange_plan`
  as ValueError.
- **INVALID audit** (`feature_spec.py`): `INVALID = []`, which is correct. Latent and weights are TILE-only in
  TARGET, so no bf8b × ROW_MAJOR activation cell exists; every weight dtype × DEST width × rope mode × buffer is
  meaningful. Not a norm-like op, so no weight-canonicalization entries are needed. No cross-tensor couplings.

## L1 Ledger Audit

- **Currency.** Every declared CB has a row: `cb_q_slice`, `cb_x_round`, `cb_qb_slice`, `cb_q_partial` (K-blocked
  only), `cb_landing` (scratch-backed), `cb_wkv` (two descriptors), `cb_prefix_out`, `cb_rope_out`, `cb_port_stage`.
  The size expressions match the allocations. The *evaluated totals* were stale (planner figures from before the
  implementer's L7 / L8 changes), so the verifier replaced them with the predicate's numbers. Kimi: stage-1 core
  760 KiB, stage-3-only 328, port 176. The maximum is the deep-Q stage-1 core at 1120 KiB, against a live budget of
  1427 KiB.
- **Capacity vs live set.** No over-sized buffer. `cb_wkv` (R1) spans every remote head of a core because those
  heads are consumed row-major, so this is a real live set. The port cores' `cb_q_slice` copy is unused but required
  for address identity, and justified in the ledger. Under-sizing: none.
- **Page format vs DEST.** `cb_q_slice` is bf16 under fp32 DEST by decision (the exchanged payload, bit-exact as
  stored, bf16 on the links; justified in the ledger). `cb_q_partial` is Float32 unconditionally: correct under
  Phase 0's fp32 DEST, but it becomes an "over" finding once `fp32_dest_acc_en=False` lands. **Folded into
  Refinement 1.**
- **Disjoint lifetimes.** Stage-1-only CBs (`cb_x_round`, `cb_qb_slice`) and stage-3 output CBs are not aliased.
  The justification is that the wkv prefetch overlaps stage 1 and there is no fit pressure (≥ 307 KiB headroom).
  Accepted.
- **Bounds / closed form.** All symbols are bounded. `fits_l1` is now implemented and checked (Code Review #1).
- **Block-size defaults.** Interleaved stage 1 spreads columns over a rect sized to `S1_COLS_PER_CORE = 2`
  columns per core (Kimi 72 cores). That departs from "full grid, coarsest block" and was justified by the
  implementer's measurement (lamp L7: the other 27 cores start stage 3 at t ≈ 0; 223.7 → 190.9 µs together with
  L8).
- **Data-movement budget.** Present and consistent with the code. x, q_b, wkv_b1 and the output each cross DRAM
  once (wkv_b1 re-read factor 1 in R2 for the only INPUTS shape); q never touches DRAM. The cheapest-traffic split
  is the one implemented (R1). K-split R6 and 2-band R5 are deferred rows with positive reasons.
- **Per-core footprint** (stage-1 core):
  `tb·tensor_row_tiles·cols_max + tb·X_ROUND_DEPTH·m·s1_kb + tb_qb·Q_t·cols_max + [nkb>1]·4096·m·cols_max +
  tb_wkv·wkv_units·Dn_t·s3n + tb·OUT_DEPTH·(s3n + Dr_t)`, plus the landing scratch `tb·hpc·B·qrt·hw`. The
  `cb_q_slice` and landing terms scale with the rows (S); `cb_qb_slice` scales with Q and `S1_COLS_PER_CORE`;
  `cb_wkv` with `WKV_RESIDENT_BYTES` / `hpc`.

## Precision Baseline

`tests/ttnn/unit_tests/operations/q_stem/test_q_stem_precision_baseline.py`: Phase 0 (bf16, HiFi4, fp32 DEST,
NoPE), torch.randn inputs (seed 42), fp32 reference. Device 0 shown (all 4 devices agree to the 4th digit).

| Shape (B,S,Q,Hl,Dn,Dr,N) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | bf16 ULP median / p99 | got/true ratio p5 / median / p95 |
|---|---|---|---|---|---|---|
| (1,128,64,2,32,32,32) | 0.999998 | 0.0128 | 0.00155 | 0.00209 | 0.32 / 11.2 | 0.9938 / 1.0000 / 1.0058 |
| (1,256,256,4,64,32,128) | 0.999997 | 0.0179 | 0.00172 | 0.00226 | 0.37 / 17.0 | 0.9915 / 1.0001 / 1.0086 |
| (2,128,512,8,128,64,512) | 0.999997 | 0.0195 | 0.00178 | 0.00233 | 0.40 / 18.3 | 0.9907 / 1.0001 / 1.0095 |
| (1,640,1536,24,128,64,512) Kimi | 0.999997 | 0.0197 | 0.00177 | 0.00232 | 0.40 / 18.0 | 0.9907 / 1.0001 / 1.0094 |

The rope part alone (a pass-through of the bf16 q) has rel RMS 0.0017 and ULP p99 ≈ 3.7: that is the single bf16
rounding of q, so it arrives bit-exact as stored. The nope part has rel RMS 0.0024 (q rounded to bf16, then a K=Dn
matmul).

**Assessment**: rounding-limited, no structure. The ratio is centred on 1.0000 with a narrow symmetric spread (no
scale bug), and the error is flat across shapes, from K = 64 to 1536.
**Recommended tolerances**: the defaults hold. `(bfloat16, fp32_dest_acc_en=True)`: PCC ≥ 0.995, rel RMS ≤ 0.04;
measured margins are ~200× on PCC deficit and ~17× on RMS. `helpers.TOLERANCES` stays empty.

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/q_stem/` + `python3 -m eval.verify_supported … ttnn.operations.q_stem`
(artifact: `ttnn/ttnn/operations/q_stem/verifier_report.json`):

- supported_pass: 15 (14 tile-aligned INPUTS shapes × the Phase-0 cell + the Phase-0 Kimi loose case)
- xfail_expected: 1077
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 11 (`test_regression.py`): 8 pass (provenance ×2, distribution ×3, back-to-back, determinism,
  plus small-packet under `run_safe_pytest.sh`) and 3 RoPE tests xfail (Refinement 5).
  `test_small_packet_size` *fails in the graded runner only*, with
  `eval.l1_profiling.L1ProfilingError: … the test has no device or mesh_device fixture`. The test takes
  `_fabric_mesh_cache` instead of `mesh_device`, so this is a harness/test-file issue in the read-only golden suite,
  not the op. It passes when run directly (`run_safe_pytest.sh …::test_small_packet_size`: 1 passed). Please fix it
  in the golden suite (give the test a `mesh_device`-named fixture or exempt it from L1 profiling).

**TARGET − SUPPORTED coverage** (every missing value has a refinement; no INVALID covers any):

| Axis | Missing value | Refinement |
|------|---------------|------------|
| q_b_proj_dtype | bfloat8_b | 1 |
| wkv_b1_dtype | bfloat8_b | 1 |
| fp32_dest_acc_en | False | 1 |
| input_buffer | L1 | 1 |
| output_layout | ROW_MAJOR | 2 |
| alignment | m_non_aligned | 4 |
| rope_mode | rope | 5 |

## Performance (device time, eval `ccl_perf` protocol, Phase-0 Kimi loose case, 7232 B packets)

| Measurement | fastest chip | slowest chip |
|---|---|---|
| Phase 0 as handed over | 190.5 µs | 207.3 µs |
| after verifier fixes | 190.9 µs | 207.7 µs (`compare`: null, within noise) |

Roofline 90.6 µs (DRAM), target 120.8 µs. Not dispatch-bound (max launch gap 0.59 µs).

Per-stage ablation (QSTEM_ABLATE, same protocol, 3 experiments, fastest chip):

| Stubbed | fastest | Δ |
|---|---|---|
| nothing | 190.8 | — |
| fabric payload (**TRANSPORT**) | 193.1 | ≈ 0 (hidden) |
| output writes | 168.1 | −22.7 |
| stage 3 + output + wkv reads | 137.7 | −53.1 |

Reading: the exchange is fully hidden, so a TRANSPORT lever is a no-op in this regime and none is filed (Refinement
6's Done-when re-checks it and spawns `6b` if that changes). The local work dominates with 2× headroom against the
roofline. The first perf slot (Refinement 3) is the **OVERLAP** tail: stage 3 of the stage-1 cores' heads runs
after stage 1. The second (Refinement 6) is stage-1 throughput, ~138 µs against the op model's 57 µs.

## Recommendations

- **Order.** Refinements 1 and 2 land the PERF FOCUS contract (bfp8 q_b, 16-bit DEST, L1 latent, ROW_MAJOR).
  Refinement 3 tunes that exact config, so the Phase-0 numbers above are only indicative for it: HiFi2 and bfp8 cut
  stage-1 compute and q_b bytes. Re-take the ablation first.
- **TRANSPORT levers not filed** (no-op while the payload is hidden): R4 relay of the antipodal quarter, the L4
  increment interval, the L5 port batch, and Ethernet-probe port placement (the implementer used centred row-0
  ports). They become refinement candidates only when the payload ablation shows exposure.
- **Ruled-out levers.** 2-band stage 1 (R5) moves +7.5 MB of q_b DRAM traffic on a DRAM-bound op. The op model's
  `--designs` run puts one band at 57.1 µs vs 62.2 µs for two bands (non-designs mode), so it is behind.
- **L1 pressure.** The deep-Q stage-1 core is at 1120 / 1427 KiB. A deeper `X_ROUND_DEPTH` or larger
  `S1_BLOCK_M_TILES` (Refinement 6) at Q = 4096 will reach the budget; `check_fits_l1` will name the term.
- **Scratch regrowth inside a trace** (advisory under Prompt Rules): size the landing scratch for the mesh's
  largest call, or allocate it per program-cache entry.
- **Dev mode.** Fabric tests cannot run under `--dev` on this board (ACTIVE_ETH kernel-config overflow with the
  watcher). Hang triage on this op has to use default mode plus `tt-triage`.

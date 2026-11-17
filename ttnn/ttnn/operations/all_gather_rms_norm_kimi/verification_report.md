# Verification Report: all_gather_rms_norm_kimi

Phase 0 verification on a Blackhole QuietBox 2. The 4 chips are opened as a 1×4 ring under `FABRIC_2D_TORUS_XY`, with a 14 KiB router payload.
The suite has one cell: TARGET = SUPPORTED, `INPUTS = []`, one PERF FOCUS `LOOSE_CASES` entry (640 × 384 → 640 × 1536, G = 4, 2 links).

## Code Review

### Fixed in place

1. **L1 ledger vs code: three planned CB aliases were not built** (`cb_gamma`/`cb_out`, `cb_xsq_sum`/`cb_rstd`, `cb_partial`/`cb_rstd_full`).
   - The implementer's only reason was "not needed to fit". That is not a positive reason.
   - Each pair is now one `CBDescriptor` with two `CBFormatDescriptor`s. The capacity of `cb_gamma`/`cb_out` is rounded up to a multiple of `chunk_tiles`, so `cb_out` never wraps mid-push when `block_w > 2·chunk_tiles`.
   - The ledger records why each alias is safe in the built schedule (unpacker/packer ordering, and the hub-multicast happens-before for `cb_partial`/`cb_rstd_full`).
   - Compute-core footprint: 164 KiB → 132 KiB at Phase 0, and 378 → 346 KiB in the tall regime.
2. **Writer stored each chunk with a full `noc_async_write_barrier`.**
   - It now does `noc_async_writes_flushed()` before `cb_pop_front`, which is enough to reuse the L1 source, plus one barrier at kernel end.
   - Measured with `eval.ccl_perf compare` together with fix 1: fastest chip 29.924 → 29.820 µs (−0.10, noise 0.05); slowest 32.707 → 32.291 µs (−0.42, noise 0.11). Verdict: **win**.
3. **DRY: the compute kernel hard-coded the 10 CB indices** that the host defines as `CB_*`. They are now passed as CT args from the host's single definition.
4. **DRY: `fp32_tile_bytes = 4096` was defined separately in the reader and the writer, and the stats-row byte count (`128`) was a literal in the host and both kernels.**
   - Both now live once: `FP32_TILE_BYTES` / `STATS_ROW_BYTES` in `kernels/all_gather_rms_norm_kimi_common.hpp`, and `STATS_ROW_BYTES` on the host, which `slot_bytes` and the stats-cap check use.
   - The γ stick read length now uses the `gamma_page` CT arg instead of a literal `64`.

All four fixes are bit-identical in output: the precision numbers are unchanged to the last digit.

### Reviewed, no change

- **Kernel API hygiene.**
  - `void kernel_main()` is used everywhere.
  - Includes use `api/dataflow/dataflow_api.h`.
  - Every DRAM access goes through `TensorAccessor`; there is no `InterleavedAddrGen`.
- **CB push/wait balance** (checked for every compute-core CB, including the short-last-block path):
  - `cb_x_block`: the own path pushes `h·block_w` once; the remote path pushes per chunk; compute waits cumulatively and pops `h·block_w`.
  - `cb_partials`: push G·h, wait/pop G·h.
  - `cb_partial`: push h, writer pops h.
  - `cb_out`: chunk quanta on both sides.
  - `cb_gamma` / `cb_rstd` / `cb_rstd_full` / `cb_scale`: popped by their helpers' `PopPolicy::AtEnd`.
  - `cb_scaler`: held for the whole call.
- **Helper usage.** Every compute phase is a kernel-lib helper:
  - `eltwise_chain` x·x with `DestAccumulation::PerRow`;
  - `reduce<…, ReduceFp32Mode::Accurate>`;
  - `eltwise_chain` combine with SFPU fp32 adds, `MulUnary`, `AddUnary`, `Rsqrt`;
  - `unary_bcast<Col>`;
  - two `mul<>` calls.

  The raw dataflow pieces (γ stick → tile row 0, stats column ↔ tile column 0) have no helper. The design's "helpers considered and rejected" table explains why.
- **Hub multicast is raw `noc_async_write_multicast` + `noc_semaphore_set_multicast`, not `McastArgs`.** The compute cores form two unequal rectangles (7 full rows of 11 and a partial row of 3), which the helper's equal-group pipe does not describe. The kernel header documents this. Accepted.
- **Broadcast.**
  - γ is a row broadcast on srcB, with only row 0 written.
  - rstd is column-broadcast once into `cb_rstd_full`, then multiplied with `InputTileMapping::Col`.
  - No CB is filled with repeated data.
- **Fence and counter discipline.**
  - Every global counter has exactly one ordered sender and is re-armed by decrementing what it consumed.
  - The ready fence is safe across back-to-back calls: a chip's program N+1 launches only after its program N has fully finished, hub included.
  - The regression suite's bit-identical-repeat test passes.

### Prompt rules (`eval/prompts/all_gather_rms_norm_kimi.txt` § Rules)

- **MUST be one fused native implementation: met.**
  - Each invocation is exactly one `ttnn.generic_op` dispatch (one `MeshProgramDescriptor`, one program per chip).
  - The op calls no norm op, no CCL op and no example as an op.
  - `allocate_tensor_on_device`, plus the one-time scratch / global-semaphore creation and `synchronize_device` on a geometry's first call, are allocation and setup, not op dispatches.
- **MUST return the full (1,1,M,W) on every device: met.** `check_output` scores every device and every slice.
- **Advisory (not a violation): the host rebuilds the whole `MeshProgramDescriptor` on every call**, including `setup_fabric_connection`. The device program is cached through `custom_program_hash`, so device time is unaffected, and the harness shows the op is not dispatch-bound (max gap 0.54 µs). It is still host overhead in an untraced model. A cached descriptor that patches only the three tensor addresses would remove it. This is recorded as a recommendation, not a refinement, because no device-ns gate moves.

## Design Conformance

| Dimension | Status |
|-----------|--------|
| Algorithm | ✓ R1: L1 landing, fp32 partials ahead of the bulk, a fixed-order SFPU fp32 combine, `S = rstd ⊗ γ` precomputed before landing, one FPU multiply per landed tile. No fp32 value is truncated: the partial reduce is `Accurate` and the combine is CopyTile from `UnpackToDestFp32` CBs + SFPU adds. |
| Pipeline topology / RISC ownership | ✓ Port reader NCRISC/NoC0 and sender BRISC/NoC1. Compute reader NCRISC/NoC0 and writer BRISC/NoC1. Receivers and hub are communication-only. |
| Work distribution | ✓ 80 compute cores (G·nb = 4·20) in two merged rectangles, 4 ports, 4 receivers, 1 hub; 21 cores idle (the design's grid-sync lamp). Compute kernels take common RT args only. Not dispatch-bound. |
| Inter-core communication | ✓ Landed counters, relay_ready, own_loaded / own_partials fan-in waited for totals, hub multicast. |
| **Port placement** | ✗ **Deviation.** The design asks for the example's probe placement: each port is the worker nearest its link's Ethernet core by NoC1 hops. The build centres the 4 ports on logical row 0, because the probe is a separate device program and the one-dispatch rule forbids it. The deviation is justified, but it leaves a TRANSPORT lever unused. **Filed as part of perf Refinement 1** (place the ports by link without a probe dispatch). |
| Blocking-model fidelity | ✓ `block_h` (derived from the post-port grid), `block_w`, `chunk_tiles` (derived from the router payload and the block), `PORT_BATCH_CHUNKS`, `INC_EVERY`, `OUT_DEPTH_CHUNKS` and `num_links` are each defined once on the host. CB sizes, trip counts and RT args derive from them. The tall regime (`block_h = 3`, ragged last block) is pinned by an acceptance test. `block_w` is host-gated to `Wl_t`, which is its design Phase 0 value; narrower blocks need same-slice partial combining (a knob-turn the design lists). |
| Expression | ✓ The remote reader handshakes once per landed chunk, which is the design's landing quantum and not a per-tile handshake. The writer flushes once per chunk. The port reader / sender batch `port_batch_chunks` per barrier / flush. Compute runs each phase on the whole block in one helper call. |

## Registry Conformance

- `INPUT_TAGGERS` holds exactly `alignment` and `rank`, both with the `(inputs, axes)` signature.
- `SUPPORTED` = TARGET on all 11 axes.
- `EXCLUSIONS = []`.
- `validate()` checks SUPPORTED per axis, then EXCLUSIONS, and raises `UnsupportedAxisValue` / `ExcludedCell`. The public entry calls it first.
- `default_compute_kernel_config()` is the single factory that resolves `None`.
- The op file does **not** declare `INVALID`.
- Caller errors (gamma format, link count on every hop including the wrap, G < 3, non-DRAM-interleaved input) raise `ValueError`.
- No XPASS evidence, so no auto-fixes to SUPPORTED.
- **INVALID audit** (`feature_spec.py`): `INVALID = []`. That is correct for a one-cell universe.
  - There is no ROW_MAJOR activation and no bf8b in TARGET, so the canonical bf8b+ROW_MAJOR entry would be vacuous.
  - gamma is required, so there are no no-weight canonicalization cells.
  - There are no cross-tensor entries.

## L1 Ledger Audit

- **Currency.** Every declared CB has a row and every row maps to a live CB. Before this pass, three rows claimed shares the code did not build; the aliases are now built (Code Review fix 1) and the ledger's implementation-notes section was rewritten to match. Size expressions match `create_mesh_program_descriptor`.
- **Capacity vs live set, over.**
  - `cb_scale` (`block_h·block_w`, fp32) and `cb_x_block` (`block_h·block_w`) span both block axes, and their live sets do span both. `cb_scale` is precomputed before landing; `cb_x_block` is resident for ports and compute.
  - `cb_out` is a 2-chunk streaming window.
  - No buffer is sized to a block while its live set is one tile.
- **Capacity vs live set, under.** No collapsed extent. `cb_partials` spans slices × rows (`G·block_h`) and scales with both.
- **Page format vs DEST width.** Every Float32 page (`cb_partials`, `cb_xsq_sum`/`cb_rstd`, `cb_partial`/`cb_rstd_full`, `cb_scale`) sits under `fp32_dest_acc_en=True`, the only supported value. No bf16 page carries a DEST accumulation. No finding.
- **Disjoint lifetimes.** The three disjoint pairs are now shared. `cb_partials` vs `cb_xsq_sum` is disjoint on own cores but not shared, and the ledger gives a reason (different page count and UnpackToDest role; sharing would serialize the partial phase against the stats receive). Accepted.
- **Bounds.** `stats_scratch` (`G·Mt·128 B`) is the only op-dimension term. It is bounded by the 64 KiB predicate, and the host raises a `ValueError` outside it. The total is closed-form.
- **Data-movement budget.** Present and consistent with the code:
  - DRAM: own slice once, output once, γ re-read per core.
  - Fabric: 1.5 slices per busiest link direction.
  - The cheapest-traffic split (R1) is the implemented one.
  - The γ broadcast (R6) is a `deferred` row with a positive reason: 61 KiB against 2.4 MiB is about 0.1 µs, below the harness noise. It is **not** filed as a refinement because it has no headroom.
- **Block-size defaults.** Interleaved: the row split takes the whole post-port grid (`block_h` derived upward) and each core's block is its whole assignment (the coarsest block). There is no measurement-driven departure.
- **Per-core footprint, compute core.** `2048·bh·bw + 2048·⌈max(bw, 2·chunk)/chunk⌉·chunk + 4096·G·bh + 2·4096·bh + 4096·bh·bw + 2048 + G·Mt·128`.
  - That is 132 KiB at Phase 0 and 346 KiB in the tall regime, against about 1.5 MB.
  - The `bh·bw` terms scale with the block knobs; `G·Mt·128` scales with M under the predicate.
- **Port core.** `2·port_batch_chunks·chunk_tiles·2048 + G·Mt·128` = 106 KiB.

## Precision Baseline

From `test_all_gather_rms_norm_kimi_precision_baseline.py`: bf16, `fp32_dest_acc_en=True`, HiFi4, randn x, γ ∈ [0.5, 1.5). Each row is the worst of the 4 devices; every device is scored on its full gathered row. Max/mean abs and ULP are measured against the **fp32 (un-rounded)** reference. Relative RMS is measured against the bf16-rounded reference, using the helpers' definition: RMS error / reference RMS.

| Shape (per device in) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP (bf16) | Mean ULP | got/true ratio p5 / median / p95 |
|---|---|---|---|---|---|---|---|
| (1,1,32,32) | 0.999998 | 0.0194 | 0.00123 | 0.00225 | 0.90 | 0.288 | 0.9964 / 0.99906 / 1.0018 |
| (1,1,128,256) | 0.999998 | 0.0220 | 0.00133 | 0.00246 | 1.09 | 0.301 | 0.9961 / 0.99892 / 1.0017 |
| (1,1,640,384) [perf focus] | 0.999998 | 0.0251 | 0.00134 | 0.00244 | 1.18 | 0.300 | 0.9961 / 0.99894 / 1.0017 |
| (1,1,2048,384) [block_h = 3] | 0.999998 | 0.0289 | 0.00133 | 0.00245 | 1.19 | 0.300 | 0.9962 / 0.99894 / 1.0017 |

**Assessment:** the output is close to the precision of a single bf16 rounding, and accuracy does not change with shape.
- The ideal for round-to-nearest is a mean of 0.25 ULP and a max of 0.5 ULP; this op measures a mean of 0.30 and a max of ≤ 1.2 ULP.
- The got/true ratio spread is broad and centred near 1, so this is **not** a scale bug.
- There is a small systematic bias: the median ratio is 0.9989 (−0.1%, about a quarter of a bf16 ULP). It matches truncation, specifically the fp32 `S` entering srcB as tf32 (≤ 2⁻¹⁰ relative) in `normalize_block`, which the design accepted (associativity row of its stall-shadow table).
- The bias is 25× inside tolerance. It has no failing cell and no concrete lever in scope: removing it would mean an SFPU multiply, or two FPU multiplies on the tail, both slower. So it is not a refinement.

**Recommended tolerances:** the defaults hold. `helpers.TOLERANCES[(bf16, True)] = (PCC ≥ 0.999, rel-RMS ≤ 0.01)`, and the measured margin is about 4× on rel-RMS.

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/all_gather_rms_norm_kimi/` reported 7/7 passed. `verifier_report.json` is in this directory.

- supported_pass: 1 (the single LOOSE_CASES cell)
- xfail_expected: 0
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 6 (the `test_regression.py` checks: rank constant, γ ramp, 3 input distributions, determinism). All passed.

The acceptance suite is 8/8 under `--dev` and in production mode. The precision baseline is 4/4.

**Infra note.** Running the whole unit-test directory in one pytest session can fail the **second** module at mesh open: "Timed out while waiting for active ethernet core … to become active again". This happens because the acceptance module and the precision module each open and close their own fabric mesh. It is an environment artifact of re-opening the fabric mesh in one process, not an op failure. Each file passes when run on its own.

## Performance and Per-Chip Breakdown

Measured with the eval harness (`perf_measure.py`, 5 experiments × 500 traced calls). The target is the fastest chip's mean.

| | fastest chip | slowest chip | spread |
|---|---|---|---|
| Phase 0 as handed over | 29.92 µs | 32.71 µs | 2.78 µs |
| After this pass (fixes 1–4) | **29.82 µs** | **32.29 µs** | 2.47 µs |
| Roofline / target | 7.6 / 10.1 µs | | |

The op is not dispatch-bound: back to back, the max gap is 0.54 µs.

**Breakdown by stubbed-payload ablation.** The hooks are in `perf/ablation_hooks.patch`. `perf/perf_ablate_nocheck.py.txt` is the harness copy with the golden check removed. Each variant's JSON is in `perf/ablate_*.json`. The stubs are:

- `nobulk`: ports send no bulk chunks, and remote cores do not wait for landing. The fence and the stats still run.
- `nocompute`: compute does the CB handshakes only.
- `nostore`: the writer pops without writing DRAM.

The ablations were measured on the code as handed over; fixes 1–4 moved the full op by only 0.1 µs. The patch has been regenerated against the current code.

| Variant (fastest chip) | µs | Reading |
|---|---|---|
| skeleton: `nobulk,nocompute,nostore` | **8.84** | ready fence (≈ 5.0 µs per `fabric_ring_exchange`) + fp32 stats round (2 hops) + own x load + γ + hub multicast |
| `nobulk,nocompute` (+ store) | 11.74 | DRAM store of 960 tiles adds 2.9 µs when it runs alone |
| `nobulk,nostore` (+ compute) | 17.81 | **per-core compute chain adds 9.0 µs** |
| `nobulk` (+ compute + store) | **23.98** | local work alone; the store is serialized behind compute per chunk (+6.2) |
| `nocompute,nostore` (+ bulk transport) | **26.32** | **TRANSPORT alone: the bulk adds 17.5 µs** over the skeleton |
| `nocompute` (transport + store) | 27.33 | |
| `nostore` (transport + compute) | 26.49 | compute is almost fully hidden behind transport |
| full op | 29.92 (29.82 after fixes 1–4) | **OVERLAP tail: 3.6 µs** exposed after transport (full − transport-only) |

What the breakdown says:

1. **TRANSPORT is the largest single term.** The L1→L1 bulk takes 26.3 µs with no compute at all.
   - The `fabric_all_gather` example (`examples/fabric_all_gather/report.md`, quoted in the eval prompt) measured a balanced ring at 256 tiles per chip on 2 links at about 18 µs per call on this ring. That is DRAM→DRAM, and it includes its ≈ 8.4 µs fixed cost.
   - This op moves 240 tiles per chip from L1. Its transport is therefore about 8 µs slower than the example's.
   - Candidate causes:
     - **unprobed port placement** (ports centred on row 0, not next to their link's Ethernet core);
     - 12 KiB packets (6 tiles) instead of 14 KiB;
     - relay latency (`inc_every = 8` notifications);
     - link-0 stats interleaved with the bulk.
2. **OVERLAP leaves 3.6 µs exposed after the last landing.** That is the normalize of the last chunk plus its DRAM store, waited and flushed per chunk, plus the S precompute whenever the stats arrive late. The stats chain (fence → own partial → export → hub → 2-hop relay → multicast) sits inside the 8.8 µs skeleton.
3. **The local chain is the next floor.** With no bulk at all the op still takes 24.0 µs, because per-core compute costs 9.0 µs and the store is serialized behind it.
   - The own-slice cores have no landing to wait for, so they finish at about this time in the full op.
   - Fixing transport alone cannot take the op below about 24 µs. Fixing the local chain alone cannot take it below about 26 µs.
   - Both are filed. By the CCL ordering rule, the cross-device items come first, because transport-only (26.3) is above local-only (24.0).

## Recommendations

- **Queue.** See `op_requirements.md`. TARGET − SUPPORTED is empty (single-cell suite), so there are **no generality refinements** and the queue is all perf, in this order:
  1. TRANSPORT;
  2. OVERLAP;
  3. local compute chain.
- **Re-run the breakdown after every perf refinement.** Apply `perf/ablation_hooks.patch`, run `perf/perf_ablate_nocheck.py.txt` from `eval/golden_tests/all_gather_rms_norm_kimi/`, then revert. Once Refinement 1 lands, `nobulk` (local-only) is expected to become the binding term. Refinement 3 is filed for that reason and should not slip behind further transport tuning.
- **Not filed, and why:**
  - R6 γ broadcast: about 0.1 µs, below noise.
  - R7 direct 2-hop send: `fabric_ring_exchange` measured relay as 1.1–1.6 µs *cheaper*.
  - Host-side descriptor caching: no device-ns change; see the advisory above.
  - Removing the −0.1% tf32 bias: no lever that does not cost tail time.
- **L1 headroom.** The compute core uses 132 KiB of about 1.5 MB, so every block-size or depth knob-turn in the perf refinements has ample room. One limit to watch: `stats_scratch` caps M at 4096 for G = 4, and the host raises a `ValueError` beyond it.

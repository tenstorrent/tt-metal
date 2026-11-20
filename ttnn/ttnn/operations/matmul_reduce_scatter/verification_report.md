# Verification Report: matmul_reduce_scatter

Verified on a Blackhole QuietBox 2 (4 chips). The system mesh is 2x2 (FABRIC_2D lines, G=2 per axis). The perf placement is the 4-chip snake ring opened as 1x4 under FABRIC_2D_TORUS_XY (G=4, Ring). The router payload is 14 KiB + 64 B throughout.

## Code Review

Fixed in place:
- **DRY: mailbox stride restated as a kernel literal.** `matmul_reduce_scatter_port_reader.cpp` hard-coded `MAILBOX_STRIDE = 16`, a second copy of the host constant that lays out the port mailbox (`MAILBOX_STRIDE` in the program descriptor). It is now passed as port CT arg 6, so the host constant is the only source.
- **Port reader handoff table sized by a magic bound.** `uint32_t handoffs[MAX_CORES = 256]` (1 KiB of NCRISC stack, no host check). It is now sized by CT arg 7, `max_port_cores = ceil(n_cores / L)`, which is exactly the number of matmul cores one port serves (`link(k) = k mod L`). `ASSERT(num_cores <= max_cores)` guards it under `--dev`.
- **L1 ledger currency.** The worked example still described the planner's grid (column strip, P=12, kb=32, 100 cores). It now gives the implemented plan: row strip, focus case 10x8 = 80 cores, P=14, kb=16, 893 KiB. It also gains the MiMo stream-regime example (1284 KiB, and why resident W misses the budget by ~125 KiB).

Tried and reverted (measured, no gain):
- **`UnpackToDestFp32` on `cb_partial_hold`.** It would give the finish chains a lossless fp32 reload of the matmul partial instead of TF32 through srcA. It is legal here, because the hold CB is unpacked only by `CopyTile` and `matmul_block` reaches it through packer L1 accumulation. Measured on the 4 precision-baseline shapes: rel-RMS 0.00467 → 0.00469 and mean-abs 0.00726 → 0.00729, i.e. no improvement. The error is dominated by HiFi2 operand truncation and the bf16 output rounding (see the triage below), so the change was reverted rather than shipped as an unmeasured lever.

Reviewed, no change needed:
- Helpers. `matmul_block` (caller-owned pack target, packer L1 accumulation, TileRowMajor + Interm) for the matmul; `copy` / `binary_sfpu<AddBinary>` / `eltwise_chain(CopyTile, BinaryFpu Add→D1, AddBinary, PackTile)` for the three finish roles; `Mcast1D` + `McastArgs` sender/receiver pipes for both operand families. The raw-API sites (arrival landing, handoff/gather with per-producer acks, fabric sender) are the ones the design justified in "Helpers considered and rejected". No `InterleavedAddrGen`, `kernel_main()` syntax, `api/dataflow/dataflow_api.h` includes, and CB push/pop counts are nominal and matched per CB.
- Blocking-model fidelity. Every block knob is a module constant or a `choose_grid` / `choose_k_block` output flowing into CT args from one place: `DEPTH_IN0/IN1/HOLD/HANDOFF`, `PORT_GROUP`, `KB_MAX`, `KB_MIN_RESIDENT`, `PREFETCH_STEPS`, `chunk_tiles`, `core_m/core_n/kb`. CB capacities derive from them, and the reader's window counts are derived (`a_pages // (core_m*kb)`), not restated. The only CBs that span an op dimension (`Kt`) are the resident-xblock operand windows, guarded by the `F(kb, True) ≤ L1_budget` predicate with the stream regime as fallback. That is the sanctioned residency fast path, not a collapsed knob.
- Expression. Each block operation's scheduling boundary is a block:
  - one DRAM-read batch plus one trid barrier per K-block step, and one multicast per K-block part;
  - one `matmul_block` per entry over `core_m × core_n × Kt`;
  - one arrival wait plus one CB credit per (entry, direction);
  - one finish chain per entry over `P` tiles;
  - one mailbox increment per (core, entry);
  - the fin write as `P` page writes with one barrier;
  - port gathers in batches of `PORT_GROUP` chunks with one barrier.

  No per-tile handshake on any of the three kernels.
- Prompt **MUST / MUST NOT** rules (`eval/prompts/matmul_reduce_scatter.txt` → Rules), every one applicable at Phase 0:
  - one fused `generic_op`: no `ttnn.matmul` / CCL / composite fallback. Per-mesh setup (global semaphores, mailbox, staging tensor) is created once and cached; it is not an op dispatch.
  - transport of a block starts as soon as its partial is finished (start roles hand off right after the copy finish), with no full partial in DRAM. Scatter -2 computes blocks farthest-destination-first (`build_schedule` descending key).
  - every cross-device add runs in fp32 DEST (fp32 DEST forced on the matmul cores). Partials travel in bf16, which the spec allows.
  - never below the requested precision: fidelity is the caller's and fp32 DEST is a floor.
  - every partial exactly once: verified by the acceptance `test_every_partial_exactly_once` and golden `test_regression.py` exact-integer tests.
  - `num_links` is honoured (one port per (direction, link); above the discovered links is a ValueError, never clamped).
  - transport cores are carved from the grid.
  - repeated calls need no host synchronization after per-mesh creation, and outputs are deterministic (acceptance `test_repeated_calls_deterministic`, golden `test_deterministic`).
  - Ring uses both directions; scatter -1 uses column stripes; `Topology` is imported from `ttnn.operations.ccl`.

  **No violations.**

Known issues (documented, not fixed in this pass):
- **Perf-experiment env switches** (`MRS_STRIP`, `MRS_ORIENT`, `MRS_KB`, `MRS_SB_TALL`, `MRS_PORT_PLACEMENT`, `MRS_SENDERS`, `MRS_PREFETCH`, `MRS_SENDER_PLACEMENT`, `MRS_LAG`, `MRS_ZONES`, `MRS_ABLATE`) are read from the environment on every call. They are the implementer's measurement scaffolding, and the perf refinements below will use them. `MRS_ABLATE=compute|dram` deliberately produces wrong output. If any of them leaked into a production environment it would silently change the kernel, so the perf refinements should fold the winning settings into the module constants and delete the switches they no longer need.
- `ROLE_*` (compute) and `KIND_*` (writer) protocol enums are restated in the kernels and the host. They are wire-protocol constants, not block knobs, so they are left as is.

## Registry Conformance

- `INPUT_TAGGERS = {"alignment": tag_alignment}` with signature `(inputs, axes)`: exactly one tagger, as the prompt requires. `SUPPORTED` lists all 8 axes (dtype, weight_dtype, layout, alignment, cluster_axis, scatter_dim, topology, num_links). `EXCLUSIONS = []`. `validate()` checks SUPPORTED per axis and then EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`. It is the first statement of `matmul_reduce_scatter()`. A non-negative `scatter_dim` is canonicalized before the check. `num_links=None` is not a registry value and is resolved after validation.
- The op file does **not** declare `INVALID`.
- Caller errors (K mismatch, unsplittable scattered dim, num_links above the discovered links, Ring without a wrapping config, bad rank / layout / memory) raise `ValueError`, as the prompt requires. A sharded `memory_config` raises `NotImplementedError` (TARGET has no placement axis).
- No XPASS evidence, so no SUPPORTED auto-fix.
- **INVALID audit** (`eval/golden_tests/matmul_reduce_scatter/feature_spec.py`): `INVALID = []`, matching the prompt ("INVALID: none") and the design ("Structural impossibilities: none"). TARGET's only layout is TILE, so no bf8b + ROW_MAJOR cell exists. There are no weight/no-weight canonicalizations and no cross-tensor couplings. Infeasible topology / group-size cells are pruned at collection time (`topology_feasible`, `scatter_feasible`) instead of being declared INVALID, which is correct because they depend on the live cluster. Well-formed.

## L1 Ledger Audit

- **Ledger currency.** All 7 CBs (6 on matmul cores, 1 on port cores) have rows, and the size expressions match `create_mesh_program_descriptor`:
  - `a_pages` / `w_pages` with depth 3;
  - `DEPTH_HOLD·P` Float32;
  - staging-backed `cb_finished` / `cb_arrivals_fwd` / `cb_arrivals_bwd` at offsets `0`, `DEPTH_HANDOFF·P·2048`, `+slots_fwd·P·2048`;
  - `2·PORT_GROUP·chunk_tiles` on ports.

  The stale worked example was fixed in place (see Code Review).
- **Capacity vs live set.**
  - Over: none. Every streaming window is depth 3 with a measured reason (prefetch). The hold and handoff are 2 slots for the lag-1 pipeline. Arrival zones are one slot per arrival because the fabric writes each slot once per call, with no credits.
  - Under: none. The resident operand spans `k` with capacity `core_m·Kt` / `Kt·core_n`, and the zones span `g/j` with `slots_d·P`.
  - One departure, folded into Refinement 3 rather than fixed here: zone slots are sized to the max over all chips of the group (line G=4: 3+3 instead of a per-chip maximum of G+1), recorded as an implementation delta. It costs `≤ 1·P` pages on line interiors and buys one staging layout for every chip.
- **Page format vs DEST.**
  - `cb_partial_hold` Float32 with fp32 DEST on: correct (it is the packer-L1-accumulation target).
  - `cb_finished` / `cb_arrivals_*` bf16 with fp32 DEST on would be the "Under" case (truncation at a phase boundary). Here it is explicitly sanctioned: the prompt says partials "MAY travel in bfloat16", and fin is the bf16 output. Every add re-accumulates in fp32 DEST.
  - Operand windows carry the input formats (Float16_b / Bfp8_b).
  - **No finding.**
- **Disjoint lifetimes.** Every row has a non-blank `Shares with / why not`. The three staging-backed CBs share one allocation at disjoint offsets and are simultaneously live. Hold vs finished differ in format, are concurrent, and have different consumer core sets.
- **Bounds and closed form.** `Kt` (the only op dimension in a capacity) appears only under the resident-xblock predicate. `G ≤ 8`, `slots_fwd + slots_bwd ≤ G+1`, `chunk_tiles ≤ 7`, `port_group ≤ 8` and `L ≤ strip/2` are all in the symbol table with their establishing predicates. The total `F(kb, resident)` is closed-form.
- **Data-movement budget.** Present and consistent with the code: A once (resident) or G× (stream ∧ scatter -1); W once or G× (stream ∧ scatter -2); output once; 0 partial bytes in DRAM. The closing line names the cheapest split (2D m,n + time-walk over j + resident cross-block operand + L1 landing) as implemented, with the stream fallback reached only by the L1 predicate. The regime table carries k-split and DRAM-landing as `deferred` with positive reasons.
- **Block-size defaults.**
  - `kb` departs from the coarsest fit (`KB_MAX = 16`), justified by measurement in the ledger deltas: focus case kb 64 → 212 µs, 32 → 160, 16 → 159, 8 → 164.
  - The grid default ("spread across the full grid") holds only partially: the focus case uses 80 of the 99 cores of the 11x9 rectangle (block 20x56 tiles splits into 10 m-lines x 8 n-lines at P=14), GLM 80, MiMo 88. Folded into Refinement 2 (occupancy lever) and Refinement 3.
- **Per-core footprint**: `F = tA·(rA ? core_m·Kt : 3·core_m·kb) + tW·(rW ? Kt·core_n : 3·kb·core_n) + 4096·2·P + 2048·max(cached, (2 + slots_fwd + slots_bwd)·P)`:
  - the A term scales with core_m and kb (or Kt when resident);
  - the W term with core_n and kb (or Kt);
  - the hold term with P;
  - the staging term with P and G.

  Focus 893 KiB; MiMo (stream) 1284 KiB of a ~1.33 MiB budget. Ports: 112 KiB.

## Precision Baseline

`tests/ttnn/unit_tests/operations/matmul_reduce_scatter/test_matmul_reduce_scatter_precision_baseline.py`: 1x4 TORUS_XY ring, Ring, all links, HiFi2. Reference: fp32 torch on the device-quantized inputs. Metrics over all 4 devices' outputs. ULP is bf16 ULP against the fp32 reference rounded to bf16; "nonsmall" restricts it to `|ref| ≥ 0.1·rms(ref)`, because cancellation near zero inflates ULP without being a large error.

| Shape (A × W, scatter_dim, W dtype, fp32 DEST floor) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP median / p99 (nonsmall p99) | got/true ratio p5 / median / p95 |
|---|---|---|---|---|---|---|
| (1,1,128,256)×(256,512), -1, bf16, True | 0.999993 | 0.0519 | 0.00726 | 0.00467 | 1 / 38 (7) | 0.977 / 0.9973 / 1.019 |
| (1,1,640,1024)×(1024,2048), -2, bf16, True | 1.000000 | 0.0704 | 0.00721 | 0.00467 | 1 / 39 (7) | 0.977 / 0.9973 / 1.018 |
| (1,1,640,2048)×(2048,7168), -1, bf8b, False (PERF FOCUS) | 0.999988 | 0.0709 | 0.00720 | 0.00466 | 1 / 39 (7) | 0.977 / 0.9973 / 1.018 |
| (1,1,2048,2048)×(2048,4096), -2, bf8b, True (MiMo) | 0.999885 | 0.0848 | 0.00721 | 0.00467 | 1 / 38 (7) | 0.977 / 0.9973 / 1.018 |

**Scale-vs-precision triage.** PCC is high and rel-RMS is low, but the ratio median sits at 0.9973 on every shape, a uniform small shrink, so I checked whether it is a scale bug. A fidelity sweep (probe, 640x1024x2048, both topologies) gives a ratio median of 0.978 at LoFi, 0.9973 at HiFi2 and 1.0001 at HiFi4, with rel-RMS 0.026 / 0.0047 / 0.0024 against a bf16-output-rounding floor of 0.0017. The bias tracks math fidelity (operand-mantissa truncation), not the transport or the adds, and it disappears at HiFi4. Ring and Linear agree to within 6% of rel-RMS. **Not a scale or structural bug.**

**Assessment**: the error is the HiFi2 matmul plus one bf16 output rounding (median 1 bf16 ULP). The cross-device sum adds no visible error, and rel-RMS is identical for G-way ring adds and line adds. The PERF FOCUS case is 7.5× under its soft gate (0.0047 vs `rel_rms_threshold = 0.035`) and 5× under the unfused path's measured 0.0235.
**Recommended tolerances**: bf16 W: PCC ≥ 0.999, rel-RMS ≤ 0.02 (the golden TOLERANCES hold with ~4× margin); bf8b W: PCC ≥ 0.998, rel-RMS ≤ 0.03.

## Verifier CLI Summary

Golden suite = `test_golden.py` (registry matrix + LOOSE_CASES), `test_ring_mock.py`, `test_fabric_configs.py`, `test_regression.py`. Report: `verifier_report.json` next to this file (a compact copy: summary, xfail breakdown, per-category node ids; the full 575 KB CLI output exceeds the repo large-file hook).

**Graded run with `EVAL_CAPTURE_L1=0`** (417 passed / 0 failed / 0 errors / 0 hangs of 1041):
- supported_pass: 411
- xfail_expected: 624, exactly `dtype=bfloat8_b` ∪ `weight_dtype=bfloat4_b` (156 cells each for bf16/bf4b, bf8b/bf16, bf8b/bf8b, bf8b/bf4b), all covered by Refinement 1
- invalid_skipped: 0 (INVALID is empty)
- no_axes_found: 6, the `test_regression.py` exactness / placement / determinism tests, all passed
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

### Golden-harness L1 capture (escalation — framework, not op)

With the runner's default `EVAL_CAPTURE_L1=1`, the same suite reports 31 supported_pass and **380 supported_fail**. Every failure is `L1ProfilingError: L1 profiling returned peak_total=0`, raised by `eval/l1_profiling.py` after a body that passed numerically. Root cause, confirmed with a graph-capture probe:
- This op is a **heterogeneous MeshWorkload**: one distinct program per chip, because schedules, roles, port placement and fabric connections are per chip, and `MeshProgramDescriptor` has no per-device runtime args that would let one program serve all chips.
- `MeshWorkloadImpl::compile` compiles and allocates the CBs of a multi-program workload **on thread-pool workers** (`tt_metal/distributed/mesh_workload.cpp:150-157`).
- `GraphTracker` processors are **`thread_local`** (`tt_metal/impl/graph/graph_tracking.cpp:23`), so `track_allocate_cb` fires where no capture processor is registered.
- The probe shows `ttnn.add` on the same mesh logging 12 `circular_buffer_allocate` events, while this op logs 0.
- The 31 "passing" cells are each the first call on a freshly opened mesh, which allocates the cached staging/mailbox L1 tensors and so shows a positive buffer peak.

No op-side change can surface CB allocations made on another thread. Making the workload homogeneous would need per-device runtime args and identical port cores on every chip, giving up the measured Ethernet-nearest port placement. The fix belongs in the framework (propagate graph processors to the workers, or compile on the calling thread while a capture is active) or in the harness (an opt-out for heterogeneous mesh ops). The numbers above use the runner's documented `EVAL_CAPTURE_L1=0` switch. **Owners of `eval/l1_profiling.py` / tt-metal graph tracking should pick this up**; the same limitation applies to every per-chip CCL op graded under L1 capture.

## Acceptance

`tests/ttnn/unit_tests/operations/matmul_reduce_scatter/test_matmul_reduce_scatter.py` passes 62/62 with `--dev` (38 line placement + 24 ring placement), before and after the fixes. It needs two invocations (`-k "not test_ring"` / `-k test_ring`) because the module-scoped line and ring mesh fixtures cannot both be open. On this box a `--dev` fabric run leaves an Ethernet core whose heartbeat makes the next mesh open fail. `touch /tmp/tt-device.dirty` before each invocation lets the wrapper reset the board under its lock.

## Perf baseline (for the perf refinements)

`test_matmul_reduce_scatter_perf.py` under `--profile`: LOOSE_CASES on the 1x4 TORUS_XY ring, 2 links, 2 warm-up + 3 measured calls, **untraced** device-kernel duration. The op's latency is the max over the 4 chips.

| Case | max over chips (µs) | chips 1-3 (µs) | chip 0 (µs) | target (µs) | roofline (µs) | unfused today (µs) |
|---|---|---|---|---|---|---|
| Kimi K2.7 o_proj [FOCUS] 640x2048x7168, -1 | 142-144 | 132-135 | 142-144 | 124.8 | 93.6 | 212.5 |
| GLM 5.x o_proj 640x4096x6144, -1 | 206-207 | 206-207 | 206 | 207.3 | 155.5 | 256.7 |
| MiMo-V2 o_proj 2048x2048x4096, -2 | 255-261 | 243-255 | 255-261 | 220.6 | 165.4 | 400.7 |

GLM already meets its target and stays a no-regression guard. The focus case is at 1.15× its target, and MiMo at 1.17×. Chip 0 (the MMIO chip) runs 8-10 µs longer than the others untraced, which looks like launch skew absorbed by the ready fence. Refinement 2 must confirm or refute that traced before treating it as kernel time. The baseline unfused numbers were traced, so compare like for like.

## Recommendations

- Queue (`op_requirements.md`):
  - R1: numeric formats (bf8b activation + bf4b weight). This is the whole TARGET − SUPPORTED gap.
  - R2: perf on the PERF FOCUS profile (occupancy / grid, fused-increment cadence via P, kb / depth co-tune, chip-0 skew).
  - R3: perf on MiMo (bring resident-W inside L1, with the ledger findings folded in).
  - R4: perf on the overlap fill term (`ENTRIES_PER_BLOCK = 2` wave split). This is a T3 schedule change.

  Only one generality refinement exists, so after R1 the queue is all perf, per the cadence rule.
- Coverage limits of this box: no physical wrap links, so Ring along each Galaxy axis is exercised only through the 1x4 / 4x1 snake-ring mocks (`test_ring_mock.py`, G=4). G=8 (a Galaxy axis 0, or a LoudBox mock) and Galaxy link rates (~25 GB/s) are untested here. That is a hardware-coverage finding, not a refinement.
- L1 pressure: MiMo's stream regime runs at 1284 KiB of a ~1.33 MiB budget. Larger P (fewer cores) or larger G on scatter -2 shapes will push the stream regime toward the `NotImplementedError` (DRAM-landing, deferred) bound. Refinement 3's L1 work should keep headroom there.
- Numerical: no precision lever is needed at the golden tolerances. HiFi4 halves rel-RMS if a tighter consumer ever needs it, and it is already supported through `compute_kernel_config`.

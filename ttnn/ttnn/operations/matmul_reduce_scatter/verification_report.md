# Verification Report: matmul_reduce_scatter

Verifier pass, Phase 0. Board: Blackhole LoudBox 2×4 (8 chips), golden suite under FABRIC_2D (+ the six fabric
configs of `test_fabric_configs.py` and the 1×8 / 8×1 ring stand-ins of `test_ring_mock.py`).

## Code Review

### Fixed

1. **L1 leak → CB/L1 clash across plans (correctness, supported_fail).** `_get_plan` allocated the L1
   HEIGHT-sharded `handoff_l1` tensor (`handoff_depth·core_m_tiles·core_n_tiles` bf16 tiles per compute core) and
   cached it **per plan key** for the life of the mesh. Every distinct shape / dtype / config kept its shard
   resident. Over one session the shards piled up from the top of L1 until a later plan's static CB region overlapped
   them. In the first golden run this produced ~600 failures: `TT_THROW: Statically allocated circular buffers in
   program N clash with L1 buffers ... L1 buffer allocated at 568512 and static circular buffer region ends at
   660480`. In a model it would also permanently take L1 away from every other op.
   **Fix:** `handoff_l1` is now allocated **per call** in the entry point (`_allocate_handoff`) and freed when the
   call returns. The generic_op program cache re-points the globally-allocated CB on a hit
   (`UpdateDynamicCircularBufferAddress`, `program_descriptors.cpp:232`), and the transport readers get the address
   as a runtime arg that is rebuilt on every call.
2. **Global-semaphore leak (same mechanism).** Seven `create_global_semaphore` L1 buffers were created per plan
   key. **Fix:** `_get_sems(mesh, cluster_axis, num_links)` makes one set per *transport identity*, over the whole
   worker grid, so a mesh holds at most 4 sets. Sharing within that key is safe: port/final placement depends only on
   `num_links` and the neighbours only on `cluster_axis`, so each ready-fence / arrival counter is always fed by the
   same neighbour. That is the same semantics as reusing one plan. Every kernel re-arms what it consumed, so the
   counters are 0 between calls. Sharing across `cluster_axis` would *not* be safe: a port could consume a ready
   sent by a neighbour on the other axis.
3. **L1 budget over-estimated by 70.6 KB (correctness, supported_fail).** The planner budgeted CBs against
   `ttnn.get_max_worker_l1_unreserved_size()` (1,531,904 B). That is `l1_end − KERNEL_CONFIG`, so it also counts the
   kernel-config ring that sits *below* the allocator base. The static CBs start at the allocator base, and the
   allocator's L1 bank is 1,461,248 B. R1 plans with `scatter_dim=-2` (W resident, e.g. `640×2048×6144`,
   `640×1536×7168`, `2048×2048×4096` on G=2) therefore chose a K-block that overflowed into the hand-off shard even
   in isolation. Those cells were the remaining failures in the second golden run.
   **Fix:** `_l1_cb_capacity()` = `ttnn.get_memory_view(mesh, L1).total_bytes_per_bank`. Some of these shapes now
   correctly fall back to R2, as the design's regime predicate intends.
4. **DRY: duplicated block / format literals.** The accumulator tile size (`4096 if fp32 else 2048`) was in
   `_plan_blocking`, and its inverse (`ttnn.float32 if acc_tile_bytes == 4096`) was in the descriptor. The DEST
   capacity was written as `4 if fp32 else 8` in the subblock solver and again as `XPORT_ADD_BLOCK_MAX = 4`.
   **Fix:** `Blocking.acc_dtype` is the single source (the tile size comes from `_tile_bytes(acc_dtype)`), and
   `DEST_TILES_16B = 8` is the single DEST constant (fp32 = half; the transport add block derives from it).
   `_tile_bytes`'s fallback table now covers float32 and bfloat4_b.
5. Dead code: removed `blk = None` and the duplicated `_plan_blocking(...)` argument lists in `_get_plan`.

### Checked, no change needed

- **Kernel hygiene:** `void kernel_main()` everywhere, `api/dataflow/dataflow_api.h` includes, `TensorAccessor`
  (no `InterleavedAddrGen`). CB push/wait counts match: the hand-off pushes the nominal
  `core_m_tiles·core_n_tiles` on every core, and the transport CBs push/pop whole segment groups and flush at the
  ring wrap. Operand multicast uses `mcast_pipe` `SenderPipe`/`ReceiverPipe` (`McastArgs`). The matmul uses
  `compute_kernel_lib::matmul_block` (packer-L1 K accumulation, TileRowMajor hand-off). The transport adds use
  `eltwise_chain` (BinaryFpu + DestReuseBinary for the 3-input form).
- **Raw-API hand-off semaphores and fabric sender** are justified in the design's "helpers considered and rejected"
  table: many-producer/few-consumer cumulative counters, and the FABRIC_2D unicast routing that the CCL helper lacks.
  Not flagged (scope boundary: mechanism is the implementer's choice).
- **Broadcast:** no broadcast operands (all adds are full-tile, `None` broadcast). Nothing to simplify.

### Deferred (needs architectural rework; recorded, not queued)

- `_PLAN_CACHE` / `_SEM_CACHE` hold a reference to every mesh they have seen (MeshDevice has no liveness API to
  purge closed meshes). The cost is benign: DRAM scratch handles of a closed mesh. `relay_scratch` is per plan key
  in DRAM (≈ `(G+1)·B` each), so DRAM use grows with the number of distinct shapes. Fine for a model's handful of
  shapes; a long sweep of distinct shapes accumulates.
- The L1 budget keeps a fixed 64 KiB reserve for other L1 allocations. A caller holding large L1-resident tensors
  across this op can still make the plan's CB region clash. The plan is cached and deterministic by design, so it
  does not read the transient free list.

## Prompt-rule check (`eval/prompts/matmul_reduce_scatter.txt` § Rules)

| Rule | Status |
|------|--------|
| MUST be one fused native implementation (single generic_op, own kernels; no ttnn.matmul / CCL ops / host round-trip) | ✓ one `ttnn.generic_op` per call; plan creation allocates buffers and global semaphores (host-side, not dispatches) |
| MUST start transport of a block as soon as its partial is complete; MUST NOT materialize the whole partial in DRAM; `-2` farthest-first compute order | ✓ hand-off from compute L1, `compute_order` = fwd/bwd farthest-first interleave, own block last |
| Partials MAY travel bf16; every cross-device add MUST accumulate in fp32 | ✓ transport add kernels hard-wired `fp32_dest_acc_en=True` |
| `compute_kernel_config` is a precision FLOOR; MUST NOT go below it | ✓ matmul honours the user config exactly (L7, using fp32 where it is free, is an open lamp; see Precision) |
| MUST include every partial exactly once | ✓ `test_rank_identity`, `test_block_placement`, and the acceptance `test_every_partial_exactly_once` are bit-exact |
| `num_links` given → MUST spread over that many links, not clamp/ignore | ✓ explicit value checked against every hop (`ValueError` if unavailable); `None` resolves to min(usable, 2) |
| Transport cores carved from the compute grid; size the split by measurement | ✓ `4L` transport cores in row 0; compute = the rest (FOCUS: 80 compute + 8 transport) |
| Repeated calls: no host sync inside the op; persistent semaphores/scratch; second call only patches RT args; deterministic | ✓ deterministic (golden + acceptance). **Advisory:** `ttnn.synchronize_device` runs once when a new `(mesh, cluster_axis, num_links)` semaphore set is created, i.e. on a first call. That is trace-unsafe if the first call happens inside a trace capture. The Python layer also rebuilds the full `MeshProgramDescriptor` (incl. `setup_fabric_connection`) on every call, which is a host-time cost (device-side, generic_op only patches). |
| Ring MUST use both directions | n/a (Ring not yet supported → Refinement 1 carries the rule) |
| `scatter_dim=-1`: produce N-blocks in transport order | ✓ same `compute_order` for both dims |
| `from ttnn.operations.ccl import Topology` at import time | ✓ |

## Registry Conformance

- `INPUT_TAGGERS` = `{"alignment": tag_alignment(inputs, axes)}`: correct two-arg signature.
- `SUPPORTED` covers every axis the kernel gates on (`dtype`, `weight_dtype`, `layout`, `alignment`,
  `cluster_axis`, `scatter_dim`, `topology`, `num_links`). `validate()` checks SUPPORTED per axis (layout also checks
  W's layout; `num_links=None` = default, skipped), then EXCLUSIONS (`[]`). It raises `UnsupportedAxisValue` /
  `ExcludedCell` and is the first line of the entry point.
- The op file does **not** declare `INVALID`.
- No SUPPORTED auto-fixes were needed (no `xpass_drift`).
- **INVALID audit** (`feature_spec.py`): `INVALID = []`, which is correct. TARGET `layout` is `[TILE]` only, so the
  canonical bf8b+ROW_MAJOR entry does not apply. The op is not norm-like, so no weight canonicalization applies. No
  cross-tensor couplings. Topology / scatter feasibility is pruned at collection time by `helpers.topology_feasible`
  / `scatter_feasible` (device-derived), not by INVALID, which is the right place for it.

## Design Conformance

- **Algorithm / pipeline / RISC ownership**: as designed. The compute rectangle runs a 2D-multicast matmul (A
  injector per m-line on NCRISC/NoC0, W injector per n-line on BRISC/NoC1). The hand-off is parked in L1. The
  transport row runs ports (gather + relay add + fabric send) and finals (gather + 3-way add + store).
- **Parallelization (fills the machine)**: FOCUS uses 80 compute cores (orientation B, per-core block 2×7, 10×8
  lines) + 8 transport cores = 88 of the 110-core grid, and the profiler's CORE COUNT confirms 86–88 per chip. Both
  dataflow halves are batched: injectors read a whole K-block then do one barrier and one multicast; transport
  readers use one barrier per `xport_group` segments; the sender flushes once per group; the final writer uses one
  barrier per group.
- **Blocking-model fidelity**: `OPERAND_DEPTH`, `HANDOFF_DEPTH`, `BLOCKS_IN_FLIGHT`, `STREAM_BUDGET`, `seg_tiles`,
  `xport_group`, `core_m_tiles`/`core_n_tiles`, `k_block_tiles` are host parameters. Each is defined once, and the CB
  sizes, CT args and RT args derive from them. No CB scales unconditionally with a whole-op dimension: the only
  `Kt`-sized CB is the R1 resident operand, guarded by the RESIDENT predicate with the R2 streaming fallback.
  Expression check (reader / compute / writer): no per-unit acquire/sync/retire loop at any block-operation
  boundary. The final writer issues per-tile writes inside one segment, but those are distinct interleaved output
  pages behind one barrier per group, so this is not a per-unit completion boundary.
- **Deviation recorded:** the design caches `handoff_l1` per plan key; it is now per call (fix 1), and the ledger
  records why.
- **Perf lamps still open** (design §Perf lamps): L5 transport placement, L6 W-injector DRAM reads on NoC1, L7 fp32
  accumulation under the floor, L3 `blocks_in_flight`, L4 port issue rate.

## L1 Ledger Audit

- **Ledger currency**: every declared CB has a row, and the size expressions match the code (`a_pages`, `w_pages`,
  `block_tiles·acc_tile`, the sharded hand-off, `xport_pages = 2·xport_group·seg_tiles`). The implementation's CB
  indices (4/5/6/16 on transport cores) are recorded in its notes. Updated: the `L1_CB_BUDGET` symbol (allocator bank
  size, fix 3) and a new "Persistent vs per-call L1" table (fixes 1–2).
- **Capacity vs live set**: no over-capacity buffer. The operand CBs' extra capacity is exactly the
  `operand_depth` prefetch window. `cb_partial_accum` is the packer-L1 accumulator and is live for the whole K loop.
  The transport CBs are 2× for reader/compute overlap. No under-capacity: the axis accounting is right (the resident
  operand spans K; streamed operands span `k_block_tiles·operand_depth`; the hand-off spans `handoff_depth` blocks).
- **Page format vs DEST width**: `cb_partial_accum` follows `fp32_dest_acc_en` (Float32 iff on), now from one
  source (`Blocking.acc_dtype`). The transport CBs are bf16 while the transport DEST is fp32. That is answered by a
  mechanism reason: the add inputs are bf16 wire payloads, and the sum is the bf16 wire/output ABI the prompt allows,
  not an intermediate accumulation.
- **Disjoint lifetimes**: every pair is justified in the ledger (operand CBs concurrent; accum vs hand-off required
  separate by TileRowMajor and the overlap of block i draining with block i+1 accumulating; sum vs partial on
  transport concurrent across the fabric flush).
- **Bounds / closed form**: `core_m_tiles·core_n_tiles ≤ CORE_BLOCK_MAX`, `Kt` only under RESIDENT,
  `k_block_tiles` divisor of Kt within `STREAM_BUDGET`, `seg_tiles ≤ payload/2048`. The total is closed form
  (ledger §Total).
- **Per-core footprint**: `F_compute = A_pages·a_tile + W_pages·w_tile + core_m·core_n·(acc_tile + handoff_depth·2048)`.
  The R1 resident term scales with `Kt·core_x_tiles`, the streamed terms with
  `operand_depth·k_block_tiles·core_y_tiles`, and accum/hand-off with `core_m·core_n`. FOCUS = 591,872 B of
  1,461,248.
- **Data-movement budget**: present and consistent with R1/R2 as built (A once in R1 / G× in R2, W once, relay
  scratch 2×(G−1) blocks, output once). The cheapest-traffic split (R5, L1 relay landing) is a `deferred` regime
  row with a positive reason. The measured ablation (transport reads stubbed → no change) supports that reason at
  the live 4352 B payload.
- **Block-size defaults**: held. Interleaved spreads the per-core block over the full compute rectangle, then takes
  the coarsest K-block that fits.
- **Filing**: all ledger findings were fixed in place (fixes 1–4). None were folded into a refinement.

## Precision Baseline

`tests/ttnn/unit_tests/operations/matmul_reduce_scatter/test_matmul_reduce_scatter_precision_baseline.py`
(worst device of 8; G = 4 except `large_k_bf16` on axis 0, G = 2). ULP = |err| in bf16 spacings at |expected|.

| Case | Config | PCC | Max Abs Err | Mean Abs Err | Rel. RMS Err | ULP mean / p99 | got/true median (p5–p95) |
|------|--------|-----|-------------|--------------|--------------|----------------|--------------------------|
| 256×512×1024, `-2`, W bf16 | HiFi4, fp32 acc | 0.999996 | 0.0508 | 0.00505 | 0.0035 | 2.3 / 21.6 | 1.0017 (0.990–1.013) |
| 640×512×7168, `-1`, W bf8b | HiFi2, fp32 acc | 0.999991 | 0.1011 | 0.00799 | 0.0052 | 5.3 / 40.0 | 0.9970 (0.975–1.018) |
| **FOCUS** 640×2048×7168, `-1`, W bf8b | **HiFi2, bf16 acc** (production) | 0.999859 | 0.2948 | 0.03803 | **0.0241** | 39.5 / 200.5 | **1.0163** (0.913–1.121) |
| FOCUS 640×2048×7168, `-1`, W bf8b | HiFi2, fp32 acc | 0.999991 | 0.0870 | 0.00823 | 0.0054 | 6.3 / 41.3 | 0.9967 (0.975–1.018) |
| MiMo 2048×2048×4096, `-2`, W bf8b | HiFi2, fp32 acc | 0.999991 | 0.0832 | 0.00814 | 0.0053 | 20.6 / 41.5 | 0.9968 (0.975–1.018) |
| 640×8448×7168, `-1`, W bf16 (axis 0) | HiFi2, fp32 acc | 0.999993 | 0.0446 | 0.00483 | 0.0044 | 5.4 / 37.6 | 0.9976 (0.978–1.017) |

**Assessment**: with fp32 DEST accumulation the op is near its operand formats' limits (rel-RMS ≈ 0.4–0.5 %; the
~0.3 % low median ratio is HiFi2's operand truncation). With the production floor (`fp32_dest_acc_en=False`) the FOCUS
case gives rel-RMS 0.024. That passes its 0.055 gate and the default bf8b 0.03 tolerance, and beats the unfused
production path's 0.036.

**Scale-vs-precision triage of the FOCUS bf16-acc row**: the median ratio of 1.016 was checked against a structural
bug with a probe (least-squares slope, large-|e| ratio, error by sign, two K values):
- Slope is 1.007 at K=512 and 1.016 at K=2048 with bf16 DEST.
- With fp32 DEST on the same data and the same transport, slope is 0.999 and rel-RMS 0.0044.

The bias grows with K and appears only with bf16 accumulation, so it is a rounding bias in the matmul's bf16
accumulation (DEST + packer L1 accumulation in Float16_b), not a doubled or missing K-block or partial. That would be
a shift of ≥ 1/num_k_blocks (25 %) or 1/G, independent of the accumulation width. **Not a bug; no refinement
filed.** The lever, if the precision is ever wanted, is design lamp L7: the floor lets the op use fp32 accumulation
where that costs no measurable perf. FOCUS is link-bound (matmul ablation: no change), so it may well be free there.
GLM (compute-bound) would pay for the halved subblocks. It has to be measured per shape. Recorded under
Recommendations, not queued (no failing cell).

**Recommended tolerances**: fp32 acc: PCC ≥ 0.9999, rel-RMS ≤ 0.01. bf16 acc (production): PCC ≥ 0.9995,
rel-RMS ≤ 0.03 at K ≤ 2048 (≤ 0.055 per the loose-case gate for larger K). atol is not meaningful (the outputs are
sums at unit scale); use rel-RMS.

## Performance baseline (Phase 0, measured)

`test_matmul_reduce_scatter_perf.py` under `run_safe_pytest.sh --profile`, 5 back-to-back calls. DEVICE KERNEL
DURATION, steady state (calls 2–5), max over the 8 chips (median chip in parentheses):

| Case | Phase 0 | Unfused baseline | Target | Roofline | Note |
|------|---------|------------------|--------|----------|------|
| **FOCUS** 640×2048×7168 `-1`, bf16 acc | **209.5 us** (199.7) | 394 | 118 | 88.3 | 1.88× over unfused; link-bound at the live 4352 B payload (realistic floor ≈ 139 us) |
| GLM 640×4096×6144 `-1`, bf16 acc | 241.9 us (217.8) | 409 | 198 | 148.6 | 1.69× over unfused |
| MiMo 2048×2048×4096 `-2`, fp32 acc | 439.3 us (421.6) | 385 | 215 | 161.4 | **slower than unfused** (link-bound: 6.3 MB per link at ~20 GB/s ≈ 315 us + fill) |

The first call carries the cross-chip launch skew at the ready fence (FOCUS 259 us).

## Verifier CLI Summary

Golden run `eval/eval_test_runner.sh eval/golden_tests/matmul_reduce_scatter/` (after fixes 1–4), then
`python3 -m eval.verify_supported`. Summary committed as `verifier_report.json` in this directory (the full per-cell report, 575 KB, exceeds the repo large-file limit and stays in the results dir).

- supported_pass: **375** (test_golden 312 + loose 3, fabric configs 24 (Linear under FABRIC_1D / 1D_RING / 2D /
  2D_TORUS_X / _Y / _XY), ring-mock Linear at G = 8: 36)
- xfail_expected: **660** = exactly TARGET − SUPPORTED: `dtype=bfloat8_b` 312, `weight_dtype=bfloat4_b` 156, both
  156, `topology=Ring` 36 (ring mock). Every one maps to a refinement (Refinements 1–2).
- invalid_skipped: 0 (INVALID is empty)
- no_axes_found: 6. These are the registry-free regression tests (`test_rank_identity`, `test_block_placement`,
  `test_deterministic` × 2 axes), all passing.
- supported_fail: **0**
- xpass_drift: **0**
- xfail_wrong_mode: **0**

Before the fixes, the first two golden runs had ~600 and then ~10 `supported_fail` (all L1 clashes, fixes 1–3). The
acceptance suite (53 passed, 1 skipped) did not catch it, because it runs too few distinct plans per session and its
shapes stay small.

## Recommendations

- **Queue** (`op_requirements.md`): R1 Ring (scheme-change), R2 bfloat8_b activations + bfloat4_b weights
  (knob-turn, `/numeric-formats-metal`), R3 perf: FOCUS transport send rate (L5 placement first), R4 perf: FOCUS
  pipeline fill (R4 sub-block waves). The FOCUS contract is already fully inside SUPPORTED, so R1/R2 are ordered by
  difficulty only.
- **Real per-axis Ring cells** (`test_golden.py` Ring along a wrapped axis) only exist on a Galaxy. On this LoudBox
  they are pruned at collection, and Ring correctness here comes from the 1×8 / 8×1 snake-ring mocks. Re-checking
  on a Galaxy is a finding for whoever owns that hardware, not a refinement.
- **Perf targets vs. live payload**: the LOOSE_CASES targets assume 14336 B packets at 48.5 GB/s. The harness's
  FABRIC_2D router config gives 4352 B (`seg_tiles = 2`), which puts FOCUS's realistic link floor near 139 us. Either
  derive the targets from the harness payload, or have the harness configure a larger router payload.
- **MiMo (`-2`, M=2048) is slower than its unfused baseline** (439 vs 385 us). It is link-bound like FOCUS, so
  Refinement 3's transport levers should carry over. It is in R3's guard set, but MiMo is not the flagged shape.
- **L7 precision lever**: fp32 accumulation under `fp32_dest_acc_en=False` cuts FOCUS rel-RMS 0.024 → 0.0054. The
  floor rule allows it; measure it on FOCUS (link-bound) before adopting, and do not adopt it on compute-bound shapes
  (GLM) without a no-regression check.
- **Subblock quality on awkward per-core blocks**: the subblock must divide the per-core block, so a prime or odd
  `core_n_tiles` under a 4-tile fp32 DEST degrades it. Examples: `640×8448×7168 -1` fp32 → core 2×7, subblock 2×1;
  `-2` R2 shapes → 1×21, subblock 1×3; G=2 `640×8448×7168 -2` → 1×25, subblock 1×1. The grid-factorization
  tie-break could take subblock size into account (`matmul_output_subblock`, 1.40–1.46×). This only matters on
  compute-bound shapes; no flagged case hits it.
- **Host overhead**: the descriptor (incl. `setup_fabric_connection` for every port) is rebuilt in Python on every
  call. Cache the descriptor per plan and patch addresses if host time shows up in model traces.
- **Unit-test status on the final code**: acceptance 53 passed / 1 skipped; precision baseline 6/6; debug tests
  3/3 (each file / function run on its own). Running `test_matmul_reduce_scatter_debug.py` as one file errors at
  the third mesh **open** (`Could not query the system mesh descriptor`, `llrt.cpp:663` Ethernet-training
  timeout) when the root `mesh_device` fixture re-opens the fabric mesh between test functions. That is board
  infra, before any op call; the same tests pass when invoked per function.
- **Board hygiene note for future passes**: killing a golden run mid-flight left the Ethernet cores untrained
  (`Timed out while waiting for active ethernet core`). The run after a post-run reset can hit the same thing.
  `touch /tmp/tt-device.dirty` before a device run forces a pre-run reset and recovers.

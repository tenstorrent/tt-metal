# Operation Design: all_gather_rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused CCL (Fabric all-gather of the hidden-dim slices + RMSNorm / LayerNorm over the full gathered row, one `ttnn.generic_op` per call over a `MeshProgramDescriptor`) |
| Goal | Every device of a `cluster_axis` group starts with its `(…, M, W_local)` slice and returns the same normalized `(…, M, W = G·W_local)` tensor. The raw slices cross the fabric straight into the **L1 of the compute Tensix cores** of every chip of the group (never through DRAM), in one identical **mirror layout** on every chip. Each chip then runs the base op's local norm (its R2 W-split combine) on the whole row, overlapped with the exchange in row-group **waves**. |
| Math (rms_norm) | `out[..., m, w] = x[..., m, w] · rsqrt( (1/W)·Σ_{w'<W} x[..., m, w']² + eps ) · gamma[w]`, `x = concat_p x[p]` along the last dim, `W` the true gathered width (padding never enters Σ or 1/W) |
| Math (layer_norm) | `out = (x − μ) · rsqrt(σ² + eps) · gamma + beta`, `μ = (1/W)Σx`, `σ² = (1/W)Σ(x − μ)²` over the full gathered row, padding excluded from both; `σ²` built from **centered** member moments (pivot-shifted, Chan-combined; see Block schedule) |
| Mode | Fused: compute = kernel-lib helpers ported from the base op (`eltwise_chain`, `reduce`) plus the base op's raw `MulGammaRowBcast` dest-reuse; Tensix-to-Tensix = the base op's partial exchange (NoC unicast + semaphores); chip-to-chip = the `fabric_all_gather` port reader / port sender pattern (relay + arrival counters + ready fence) retargeted from DRAM to compute-core L1. |
| References | Base op `ttnn/ttnn/operations/base_ops/rms_norm/` (`op_design.md`, `changelog.md`, `rms_norm_program_descriptor.py`, `kernels/`); `ttnn/ttnn/operations/examples/fabric_all_gather/` (`README.md`, `report.md`, `program_descriptor_with_inline_kernels.py`: `plan()`, `_schedule()`, `probe_ethernet_cores()`, `_worker_nearest_noc1()`, `_needs_ready()`); `examples/fabric_ring_exchange/README.md`; `examples/fabric_reduce_scatter/README.md`; `examples/master.md`; `.claude/references/{blocking-model.md, l1-footprint-discipline.md, blackhole-fabric.md, blackhole-ccl-kernels.md, precision_convention.md}`; golden suite `eval/golden_tests/all_gather_rms_norm/` (`feature_spec.py`, `roofline.py`, `helpers.py`, `perf_measure.py`) |

### Catalog / example measurements → knobs they set

| Source | Measured fact used | Knob / decision it set |
|---|---|---|
| `fabric_all_gather` README + report | balanced ring beats line 1.25–1.83×; balance the far shard from ~64 tiles (1.10–1.28×); 2 links 0.63× at 256 tiles; fixed cost ≈ 8.4 µs + bytes / 77.2 GB/s; increment every 8 chunks; 4 chunks per batch (112 KiB at 14,336 B); port core next to its Ethernet core, sender on NoC1 | **TRANSPORT**: balanced ring schedule (`_schedule(…, balance=True)`); `INC_EVERY = 8` packets per entry sets the wave size; `port_batch_chunks = 4`; port placement and NoC assignment copied verbatim |
| `fabric_ring_exchange` README | the ready fence costs ≈ 5.0 µs of a traced call; direct 2-hop sends 1.1–1.6 µs slower than relay (the fence waits on the 2-hop chip); a second link does not help ≤ 8 KiB payloads | **TRANSPORT**: relay (one-hop store-and-forward), neighbour-only ready fence; direct / chip-multicast is a `deferred` regime |
| `blackhole-fabric.md` rules 1, 3, 4, 6, 7, 10 | full packets; flush per batch not per packet; fused write+inc stalls the receiving router (one per 8 packets); sender below its ERISC; senders on NoC1; one producer per (link, direction) | packet = run of ≤ `chunk_pages` tiles; one fused inc per (port, wave, shard) entry ≈ 8 packets; one port core per (direction, link) |
| `blackhole-ccl-kernels.md` §5–§7, §10 | cross-chip counters must be global semaphores, re-armed by decrement; landing in a fixed L1 buffer needs credits **unless every slot is used once per call** | arrival / ready counters are cached global semaphores; landing slots (cb_x) are written once per call (resident regime) → no credits; the bounded-slot variant with credits is a `deferred` regime |
| base op `width_split`, `tensix_all_reduce`, `row_reduce_accumulate`, `eltwise_l1_vs_dest_accumulate`, `sfpu_tile_scope`, `compute_block_size` (via its op_design / changelog) | W-split + combine pays at few tile-rows; symmetric all-gather for k ≤ 8, owner combine beyond; DEST accumulation of x²; column-0 SFPU finalize; whole assignment as one block | the local half: k selection rule (`_choose_k` / `_occ_cost`, re-run on the fused grid), combine topology, pass-1 DEST accumulation, finalize scope, block = whole assignment |
| `tensix_all_reduce` (L1 axis) | push-style gather buffers must be allocated identically on every core for `get_write_ptr()` to resolve on the peer | `cb_x` is the **first** CB of an **identical CB list** declared over the identical core set on every chip → one landing address on every core of every chip |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor (mesh) | yes | TILE; rank ≥ 2; identical per-device shape / dtype / placement; dtype ∈ {float32, bfloat16, bfloat8_b}; DRAM interleaved or L1 HEIGHT / WIDTH sharded | — | buffer address RT (reader, own members) |
| cluster_axis | int (kw-only) | yes | 0 or 1, axis size ≥ 2 | — | host only (groups, positions) |
| gamma | ttnn.Tensor \| None (kw-only) | no | full-W weight, identical on every device; TILE `(1,1,1,W)` or ROW_MAJOR sticks `(1,1,ceil(W/32),32)`; dtype float32 / bfloat16 | None | `has_gamma`, `gamma_is_sticks` CT; address RT |
| beta | ttnn.Tensor \| None (kw-only) | no | layer_norm only, only with gamma; same dtype, layout and format-shape as gamma | None | `has_beta` CT; address RT |
| norm_type | str (kw-only) | no | "rms_norm" \| "layer_norm" | "rms_norm" | `is_layer_norm` CT |
| epsilon | float (kw-only) | no | finite, ≥ 0 | 1e-6 | RT (fp32 bits) |
| topology | Topology (kw-only) | no | Linear \| Ring (imported as `from ttnn.operations.ccl import Topology`) | Linear | host (schedule) |
| num_links | int \| None (kw-only) | no | 1 ≤ n ≤ min over the group's hops of `len(ttnn.get_forwarding_link_indices(src, dst))` | None → that minimum | host (ports) |
| compute_kernel_config | ttnn.ComputeConfigDescriptor (kw-only) | no | `fp32_dest_acc_en` gated (SUPPORTED + EXCLUSION float32 ∧ False); fidelity / approx: any, honored | `default_compute_kernel_config()` → HiFi4, fp32_dest_acc_en=True, math_approx_mode=False | compute KernelDescriptor `config` |
| memory_config | ttnn.MemoryConfig \| None (kw-only) | no | output placement: DRAM interleaved, or L1 HEIGHT / WIDTH sharded for the gathered shape | DRAM interleaved | host (output allocation) |
| output_tensor | ttnn.Tensor \| None (kw-only) | no | per-device shape `(…, M, W)`, input dtype, TILE | None | output address RT |

Entry-point contract:

| Item | Decision |
|------|----------|
| `default_compute_kernel_config()` | exported from `ttnn/operations/all_gather_rms_norm/__init__.py`, a factory (fresh descriptor per call); `None` resolves through it and nowhere else (`axes.py` imports it). |
| First line | `validate(...)` (support gate), then the caller-error checks below. |
| Caller errors (ValueError / RuntimeError) | rank < 2; gamma shape ≠ `(1,1,1,W)` (TILE) / `(1,1,ceil(W/32),32)` (ROW_MAJOR); beta without gamma; beta with `rms_norm`; norm_type ∉ {rms_norm, layer_norm}; beta shape mismatch, or beta dtype / layout ≠ gamma's; `output_tensor` per-device shape / dtype / layout ≠ expected; `num_links` > discovered on any hop or ≤ 0; Ring on an axis with no wrap route (`get_forwarding_link_indices(last, first)` empty or the fabric config does not wrap `cluster_axis`: TORUS_X wraps axis 1, TORUS_Y axis 0, TORUS_XY both); `cluster_axis` of size 1. |
| Support gate | `validate()` raises `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`. Axes: dtype, fp32_dest_acc_en, layout, alignment, rank, gamma_mode, gamma_dtype, gamma_layout, norm_type, beta_mode, memory_layout, cluster_axis, topology, num_links. `gamma_dtype` / `gamma_layout` read `"none"` when gamma is absent and never refuse it. |
| INPUT_TAGGERS (exactly two) | `alignment(inputs, axes)` on `inputs[0]` (per-device input): `"w_non_aligned"` if `W_local % 32`, else `"h_non_aligned"` if `H % 32`, else `"tile_aligned"`; `rank` → `len(inputs[0])`. |
| Recommended Phase 0 SUPPORTED (implementer owns it) | the from-base list for the base-op axes (dtype {fp32, bf16, bf8b}; fp32_dest_acc_en {True, False}; layout {TILE}; alignment {all three}; rank {2,3,4}; gamma_mode both; gamma_dtype {fp32, bf16, "none"}; gamma_layout {TILE, ROW_MAJOR, "none"}; memory_layout {INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED}); norm_type {rms_norm, layer_norm}; beta_mode {beta, no_beta}; cluster_axis {0, 1}; topology {Linear, Ring}; num_links {1, 2}. EXCLUSIONS carried from the base op: `{dtype: float32, fp32_dest_acc_en: False}` (precision convention), `{dtype: bfloat8_b, alignment: w_non_aligned}` and `{dtype: bfloat8_b, alignment: h_non_aligned}` (block-float tile padding; the dense assembly below would re-quantize shifted face rows). |
| Single dispatch | host: validate → derive geometry → `plan()` (groups, schedules, ports, compute grid, k, waves) → allocate output (or take `output_tensor`) → build one `MeshProgramDescriptor` (one `ProgramDescriptor` per chip) → one `ttnn.generic_op([input, (gamma), (beta), output], mesh_desc)`. No other ttnn op runs on the tensors; no host synchronization per call. |
| Persistent state | global semaphores `sem_arrived`, `sem_ready` (one each per port core), created once per (mesh, port-core set) with `ttnn._ttnn.ccl_host.make_ccl_semaphore` / `create_global_semaphore` + one `synchronize_device` at creation, cached in a module dict and kept alive (generic_op keeps only addresses). Every call leaves them at 0. The probe-derived port placement is cached per mesh (the example's `_PROBE_CACHE`). |
| Program cache | `custom_program_hash` covers everything that changes the per-core RT-arg layout (G, topology, L, k, waves, regime flags) — the `fabric_ring_exchange` gotcha (two layouts sharing one cached program hang). |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per device `(…, M, W_local)`, rank ≥ 2, identical on every device; leading dims fold into the row axis |
| Dtype | float32, bfloat16, bfloat8_b (identical on every device) |
| Layout | TILE |
| Memory | DRAM interleaved, or L1 HEIGHT_SHARDED / WIDTH_SHARDED (identical on every device) |
| gamma / beta | full-W, replicated; TILE `(1,1,1,W)` (row 0 of each tile valid) or ROW_MAJOR sticks `(1,1,ceil(W/32),32)` (stick j = `gamma[32j … 32j+31]`, tail zero-padded); dtype float32 / bfloat16, may differ from the input's |

### Output

| Property | Value |
|----------|-------|
| Shape | per device `(…, M, W)`, `W = G·W_local`; group position p's slice at columns `[p·W_local, (p+1)·W_local)`, dense (no padding between slices) |
| Dtype | input dtype |
| Layout | TILE |
| Memory | `output_tensor`'s, else `memory_config`, else DRAM interleaved; HEIGHT / WIDTH sharded for the gathered shape accepted |
| Group consistency | bit-identical on every device of a group (every chip performs the same arithmetic in the same order on the same mirror layout) and run to run (fixed combine order, no data-dependent scheduling of arithmetic) |

## Blocking Model

The op is the base op's local norm over a row whose W columns are **pre-split across the G devices by the caller**: the gather axis is the base op's dependent W axis, already cut G ways. The design undoes that cut by moving each slice — raw — into the L1 of the compute cores that own those columns on every chip, and then runs the base op's R2 regime (row-groups × W-split k members, NoC combine) on the whole row, chip-locally and redundantly on all G chips.

Geometry (single source: host `derive_geometry()`; `ceil`, per image):

| Symbol | Formula | Unit |
|---|---|---|
| `G`, `p` | group size `mesh_shape[cluster_axis]`; group position (`c` for axis 1, `r` for axis 0; the example's sorted-coord rank) | chips |
| `W_local`, `W` | `shape[-1]`, `G·W_local` | elements |
| `num_images` | `prod(shape[:-2])` | images |
| `tensor_row_tiles` (trt) | `num_images · ceil(H/32)` | tile-rows |
| `slice_w_tiles` (sw) | `ceil(W_local/32)` | tiles per slice per tile-row |
| `slice_valid_last` (v) | `W_local − 32·(sw − 1)` ∈ [1, 32] | elements |
| `padded_w_tiles` | `G · sw` — the **slice-tile space**: G slices side by side, each with its own tile padding (this is how cb_x stores a row) | tiles |
| `out_w_tiles` | `ceil(W/32)` — the dense output tile columns | tiles |
| `slice_shift[q]` (s_q) | `(q·W_local) mod 32` (all 0 when tile-aligned) | elements |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| group (mesh axis other than `cluster_axis`) | **independent** — groups never exchange data | `groups` (mesh rows / columns) | every group, concurrently | `plan()` → per-chip program | one program per chip; each chip belongs to exactly one group | — (fixed by the mesh) |
| row (tile-rows, incl. folded leading dims) | **independent** — each output row depends only on its own input row | `core_row_tiles` (per row-group); block extent along rows = whole `core_row_tiles` | the whole per-group assignment `ceil(trt / num_row_groups)` (or floor; 2 classes) — one block | host `select_split()` → RT `group_row_start`, `core_row_tiles` | row-groups over the fused compute grid (`split_work_to_cores` over `num_row_groups = min(trt, floor(N_compute / k))`) | knob-turn (grid, k) |
| source slice (the gather axis, `q ∈ [0,G)`) | **dependent** — the statistic spans all G slices; **pre-split across chips by the caller** | `G` (fixed by the mesh) and its mapping onto members: `members_per_slice` (k_s) or `slices_per_member` (d) | `k = G·k_s` or `k = G/d` from the base op's `_choose_k` restricted to slice-aligned candidates (focus: k_s = 1, k = 4) | host `select_split()` → CT `w_split_cores` (k), RT member column range | each slice's columns are owned by fixed member indices in **every** row-group on **every** chip (mirror layout); remote slices arrive over the fabric, own slice from DRAM / shard | scheme-change already built (fabric gather); member mapping is a knob-turn |
| W within a slice (tiles) | **dependent** — Σ spans it | `member_w_tiles` = `ceil(sw / k_s)` (k_s ≥ 1) or `d·sw` (d slices per member) | per `_choose_k`; resident (whole member width, one block) | host `select_split()` | k members of a row-group | knob-turn |
| wave (row-groups in transport order) | **independent** (an ordering of the row axis, not a data axis) — groups row-groups into fabric entries | `wave_row_groups` | `min(num_row_groups, 2L · ceil(INC_EVERY / (2 · member_packets)))`, `member_packets = ceil(member_w_tiles·core_row_tiles / chunk_pages)` | host `plan_waves()` → port RT entry lists | waves are processed by every port in ascending order; all members of a wave's row-groups land together | knob-turn (OVERLAP lamp) |
| link (`l ∈ [0,L)`) | **independent** — links are separate routing planes | `num_links` (L) | requested / all usable | entry point → `plan()` | row-group r is carried by link `r mod L` | knob-turn |
| direction (fwd p→p+1, bwd p→p−1) | **independent** | — (2 on a ring and interior line chips, 1 at line ends) | the example's balanced schedule | `_schedule(p, G, ring, balance=True)` | far slice (even ring) split by row-group: row-group r's far block goes fwd iff `(r div L) mod 2 == 0` | — |
| dense output tile column (`w_non_aligned` only) | **independent** (after normalization) | — (one producer member per output tile: the member owning output column 32j) | — | host `plan_assembly()` | home member of each output tile | built with the dense regime |

Operand-reuse check per (operand, chosen split):

| Operand | Split across cores / chips | Varies along it? | Consequence |
|---|---|---|---|
| x slice of chip q | chips (the gather) | — it **is** the split axis | every chip needs every slice: the all-gather; raw slices cross each link the schedule assigns, once |
| x | row-groups (cores) | yes | none |
| x | members (cores) | yes | none |
| gamma / beta | row-groups | **no** → reuse-shared | each member reads its own columns once per call into a resident CB (row-0 form, ≤ 1/32 of its x bytes); the broadcast (one reader + `ttnn.Mcast` down a member column) is a `deferred` row, as in the base op |
| gamma / beta | chips | **no** (replicated input) | each chip reads its own replica from its own DRAM — no fabric traffic |
| row statistics | members | **no** | the base op's combine (symmetric gather k ≤ 8, owner k > 8) |

Step-1 table — one row per axis of every data object, no blanks:

| Data | Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|---|---|---|---|---|---|---|
| x (own slice, from DRAM / shard) | row | independent | `core_row_tiles` (whole assignment) | own-slice members of every row-group | 1 (resident) | read once; also the source the ports send from |
| x (own slice) | W | dependent | `member_w_tiles` | own-slice member(s) | 1 | resident across all passes |
| x (remote slices, landed) | row | independent | `core_row_tiles` | remote-slice members | 1 (written once per call) | landing slot = the compute block → no credits, no copy |
| x (remote slices) | W / source slice | dependent | `member_w_tiles` | the member owning those columns | 1 | the mirror layout makes the slot address chip-invariant |
| outgoing chunks (port) | wave × row-group × member run | independent | packet = run of ≤ `chunk_pages` contiguous tiles of one member block | port (direction, link) | `2·port_batch_chunks` packets | the example's batching |
| member moments (pass outputs) | row | independent | `core_row_tiles` | every member | 1 block | per tile-row column-0 tiles |
| member moments | W | reduced away | n/a | n/a | n/a | DEST accumulation folds W |
| gathered partials | row × member | row independent, member dependent (summed) | `core_row_tiles × k` (× 2 tiles for LN) | every member combines redundantly (k ≤ 8) / row owner (k > 8) | 1 block | base op mechanism; every member needs the result |
| row stats (inv / pivot / shift) | row | independent | `core_row_tiles` | every member | 1 block | consumed by normalize |
| normalized out | row, W | as x | the member block | same core | streams (`out_depth = 2`) aligned; whole block when assembling | written once |
| dense output tiles (`w_non_aligned`) | row × output column | independent | `core_row_tiles × home_out_tiles` | home member of each output tile | 1 block | assembled from the member's own + ≤ `max_foreign` foreign contribution tiles per row |
| foreign contributions (`w_non_aligned`) | row | independent | `core_row_tiles × max_foreign` | sent by the contributing member to the home member | 1 block | a boundary output tile straddles two members |
| gamma / beta | W | reuse-shared over row | member columns, resident | each member its own columns | 1 (program) | ≤ 1/32 of x bytes |
| stage designation: port cores | — | communication role, not work | — | port reader / sender only | — | ports carry no normalize work; the normalize of every row is spread over k members of its row-group — no stage runs on one core while the group waits, except the combine owner for k > 8 (base op R3, measured) |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x` | none (single resident block) | 1 block = whole assignment | it is the landing slot: written once per call by the fabric or the DRAM reader |
| `cb_port_stage` | `port_batch_chunks` (batch) × 2 | 4 × 2 packets | the reader fills batch b+1 while the sender drains batch b (example: 4 per batch best or within 10 %) |
| `cb_out` (aligned) | `out_depth` × `out_unit_tiles` | 2 × `min(member_w_tiles, 16)` | writer drains while normalize packs |
| `cb_norm` (`w_non_aligned`) | none | whole member block | the assembly reads source tiles i−1 and i per output tile |
| `cb_gathered` | `gathered_depth` | 1 (one block per core) | — (base op needed 2 only for ≥ 2 blocks per core in lockstep) |
| all per-row stat CBs | 1 block | 1 | producer/consumer hand-off within the block |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fabric max payload (`ttnn.get_tt_fabric_max_payload_size_bytes()`, 14,336 B in the suite) | packet run length | `chunk_pages = max_payload // tile_bytes` (bf16 7, fp32 3, bf8b 13); a packet never spans two destination cores or a non-contiguous run | the router rejects / truncates; a run across cores lands at the wrong core |
| Mirror layout: the landing address is the sender's own `cb_x` address | `cb_x` address on every core of every chip | `cb_x` is the **first** CB descriptor; every chip's program declares the **same CB list, same sizes, same core sets** (`cb_x` over compute ∪ ports_union; `cb_port_stage` over ports_union) | a peer lands bytes at a different offset → corrupt neighbour CBs, silent |
| Resident landing (one write per slot per call) | `core_row_tiles × member_w_tiles` | regime predicate `resident_fits` (whole assignment + all other CBs ≤ `cb_l1_budget(device)`) | without the predicate the slot would need reuse within a call → credits (the deferred R-stream regime) |
| Arrival counter: one ordered sender per counter | increments on `sem_arrived` of a port | exactly one upstream sender (same direction, same link) per receiving port; one fused inc on the **last packet of every (wave, shard) entry**; threshold for incoming entry e = e + 1 | several senders on one counter → a threshold no longer implies which entry landed → relay / notify early → wrong data |
| Fused inc covers earlier packets of the channel | entry granularity | rely on the same router semantics the example relies on (inc issued after the write lands, packets of one channel in order) | — (the example is bit-exact under these semantics) |
| Header pool: 12 headers per data-movement RISC | `port_batch_chunks` | ≤ 8 data headers + 1 ready header per sender | pool exhaustion hang |
| Global semaphores return to 0 every call | `sem_ready`, `sem_arrived` | consumers decrement exactly what they consumed (`noc_semaphore_inc(addr, 0u − n)` + atomic barrier); never set to 0 | next traced call hangs or relays early |
| 16 semaphore ids per core; fabric setup takes the lowest free ids | program-local semaphores on ports | declare `sem_own_loaded` (and any other own semaphore) **before** `setup_fabric_connection` | "Semaphore id N already in use" at build |
| Member must not partially cover two slices | `k` candidates | `k ∈ {G·k_s : 1 ≤ k_s ≤ sw}` ∪ `{G/d : d | G}`; within a slice, columns split balanced (`ceil(sw/k_s)`, ragged last) | runs per tile-row become fragments of two slices → more, smaller packets; the dense assembly's home rule breaks |
| Base-op combine caps | k | symmetric all-gather for `k ≤ K_SYM_MAX_INTERLEAVED = 8`; owner combine for k > 8 up to `K_CAP` (RT-arg pairs) | as in the base op |
| Identical member layout on every chip | compute core set | `compute_cores = worker grid − ports_union` (union over **all chips of the mesh**), row-major; groups = k consecutive compute cores | a chip whose ports occupy a different core would hold a block at a different place → the mirror breaks |
| DEST capacity (fp32 DEST: 4 tiles half-sync) | none on the block; chain / reduce clamp their own subblocks | — | — |
| Padding never enters Σ | last tile of each slice (`v < 32`), H tail rows | the member reader zero-fills columns ≥ v of every slice-last tile it holds (own or landed — its own copy); LN centered pass multiplies `c' = x − K` by `cb_tail_mask` on those tiles; rows are row-local | golden poisons padding; a padded zero becomes −K after centering |
| Precision: FPU operands are tf32 (srcA / srcB 19 bit) | LN centering of fp32 inputs; any large-magnitude value fed to an FPU reduce | LN: pivot-shifted moments (below); fp32 input LN centering on SFPU (`CopyTile` UnpackToDestFp32 + broadcast bf16 pivot + `SubBinary`); never feed un-centered squares of a large mean to an FPU reduce | big_offset (mean 50, std 1) fails by > 1 % |
| BH DRAM read alignment 64 B | gamma / beta stick reads | read whole sticks (64 B bf16, 128 B fp32) into staging, scatter face rows by RISC-V; TILE gamma: 2 × 64 B row-0 reads per tile (base op) | wrong bytes |

### Regimes

Named memory boundaries: **DRAM** (minimum: x read once = S per chip, output written once = G·S per chip; gamma/beta ≤ 1/32 of a row per member) and **Fabric** (minimum: every chip receives the G−1 remote slices; busiest link direction carries `(G−1)/2·S` on a balanced ring, `(G−1)·S` on a line, over L links — `roofline.bounds`).

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 mirror_resident_gather** (raw slices relayed into compute-core L1, chip-local base-op norm, row-group waves) | **built** | `resident_fits(k)` holds for the chosen k (every INPUTS shape on BH with G ≤ 8) | `core_row_tiles × member_w_tiles` per compute core; transport entry = (wave, slice, port) | DRAM: **minimum** (S read once by own members, G·S written once). Fabric: **minimum** link load (balanced ring / line schedule of the example) — the far slice crosses 2 hops, relayed by the middle chip out of its compute L1, no DRAM. Cross-core above minimum: ports read every outgoing chunk from compute L1 (own + relayed = the port's link load, local NoC), the k-member partial exchange (k²·core_row_tiles column-0 tiles, faces 0/2), landing notifications (one local inc per member per entry) | per call: one ready fence per port; per entry (≈ 8 packets): one fused inc + one counter wait + ≤ wave-size notifications; per block (once per call): one DRAM barrier / one landing wait, one chain init per pass, one reduce per pass, one combine round, one normalize init |
| R1 · k-variants (sub-regimes of R1, the base op's) | built | k = 1: no combine; 2 ≤ k ≤ 8 symmetric all-gather; k > 8 owner combine (base op R3) | as R1 | as base op R1 / R2 / R3 | — |
| R1 · layer_norm (pivot-shifted moments + Chan combine) | built | `norm_type == "layer_norm"` | as R1 | +1 local pass over resident x (rough mean) and the LN normalize chain is 5 ops instead of 2; combine payload 2 column-0 tiles per member per tile-row instead of 1 | one more chain init + reduce per block |
| R1 · dense assembly (`w_non_aligned`) | built | `W_local % 32 != 0` | as R1, plus `core_row_tiles × home_out_tiles` dense tiles | same DRAM / fabric bytes (raw padded slices cross: `sw·32` columns per slice instead of W_local, ≤ 31 columns per slice per row extra); cross-core: ≤ `max_foreign` 1-tile contributions per tile-row to the home member | assembly matmul init once per block; shift matrices built once per call |
| R-stream credited landing (cb_x as a ring of landing slots; the sender waits for a fabric credit from the receiving core before reusing a slot) | **deferred** — positive reason: no INPUTS shape fails `resident_fits` on the target boxes (worst: (1,1,2048,1280) fp32 at G = 4 → 2048·5120·4 B = 42 MB per chip over ≥ 94 compute cores ≈ 450 KB/core; at G = 8 ≈ 900 KB/core, inside BH's budget); and credits are an unmeasured cross-chip round trip per slot (`blackhole-ccl-kernels.md` §7). Reachable: entries, waves and the per-core landing slot are unchanged; only cb_x becomes `landing_depth` slots and the port reader gains a credit wait | `¬resident_fits` | `landing_rows × member_w_tiles` per slot | same bytes; adds one credit packet per slot per hop | — |
| R-multicast transport (own slice chip-multicast to p+1, p+2 in one send; no relay read) | **deferred** — the fence must then wait on the 2-hop chip: measured +1.1–1.6 µs per call for direct sends (`fabric_ring_exchange`), multicast throughput unmeasured (`blackhole-ccl-kernels.md` §4); relay already reaches the minimum link load. Reachable: entries and landing addresses are chip-invariant (mirror), so only the port sender's route + the ready fan-in change | Ring, G ≥ 3 | as R1 | same link bytes; removes the relay's L1 read and its arrival-wait hop | — |
| R-zero-copy own shard (sharded input: own member = the shard core, cb_x backed on the shard) | **deferred** — the shard grid is caller-chosen and need not coincide with the mirror layout's own-member cores; the Phase 0 own-member read through `TensorAccessor` moves S over the local NoC (not DRAM) inside the ready-fence window (≈ 5 µs measured), off the critical path. Reachable: own members already read through one accessor call per block; co-locating them on the shard cores is a `select_split()` constraint | input sharded and shard block == member block | shard | removes S of local NoC traffic | — |
| gamma_broadcast (one reader per member column + `ttnn.Mcast`) | **deferred** — base op: catalog `shared_input_reuse` measured null for window ratio ≤ 0.125; here ≤ 1/32. Reachable: cb_gamma is a resident single-producer CB | — | — | gamma read once per member column instead of once per member | — |
| normalize-then-gather (exchange per-row partials over the fabric, combine, normalize the own slice, then all-gather the normalized slices into the output DRAM) | **rejected** — superseded by R1: equal DRAM and link bytes (± M·4 B), but (a) two serial fabric rounds (partials, then bulk) instead of one; (b) landing a W-gather in interleaved DRAM pages gives runs of only `ceil(sw/num_banks)` pages per bank → 1–2-tile packets (packet-rate-bound, `blackhole-fabric.md` rule 1); (c) a non-tile-aligned slice cannot land page-wise in a dense output (two chips own one output tile); (d) LN would need a second partial round or a cross-chip Chan payload | — | — | +1 fabric round; packets 3–7× smaller | — |
| gather-to-DRAM then norm | **rejected** — dead end: +2·G·S DRAM (write gathered, read back) on top of the minimum; nothing in R1 is built on it | — | — | DRAM 13·S vs 5·S per chip at G = 4 | — |
| single_core / one row-group per chip | **rejected** — Phase 0 is multi-core: R1 spreads row-groups × k members over the fused grid | — | — | — | — |

Regime selection (exact, host-side, `select_split()` — re-run of the base op's rule on the fused grid):

```
compute_cores = row-major list of (worker grid − ports_union)        # identical on every chip of the mesh
N             = len(compute_cores)
cands         = sorted({G*ks for ks in 1..sw} ∪ {G//d for d | G})    # slice-aligned member counts
member_w(k)   = ceil(sw / (k // G)) if k >= G else (G // k) * sw
groups(k)     = min(trt, N // k)
rows(k)       = ceil(trt / groups(k))
fits(k)       = footprint(k, rows(k)) <= cb_l1_budget(device)          # l1_ledger.md, whole assignment resident
cost(k)       = rows(k) * (member_w(k) + COMBINE_TILE_COST*[k > 1])  + [k > 8] * ceil(rows(k)/k) * k * OWNER_MEMBER_COST
k             = argmin cost over fitting cands with member_w(k) >= OCC_TILES_PER_MEMBER (always admit the
                smallest fitting cand); ties -> smaller k; base-op guard: padded_w_tiles < 2*OCC_TILES_PER_MEMBER -> smallest fitting
if no cand fits: RuntimeError (R-stream is deferred)
wave_row_groups = min(groups(k), 2*L*ceil(INC_EVERY / (2*member_packets)))
```

Constants `COMBINE_TILE_COST = 3`, `OWNER_MEMBER_COST = 0.4`, `OCC_TILES_PER_MEMBER = 4`, `K_SYM_MAX_INTERLEAVED = 8` are the base op's (measured there). Result on the perf cases (QuietBox 2, 11×10 grid, ports_union ≤ 16 → N ≥ 94):

| Case | trt | sw | k (members) | groups × rows | member block | Notes |
|---|---|---|---|---|---|---|
| 640×384 → 640×1536, rms (FOCUS) | 20 | 12 | 4 (k_s = 1) | 20 × 1 | 1 × 12 tiles (24 KB bf16) | 80 compute cores; cost 15 vs k = 8: 18 (only 11–13 groups fit → 2 rows) |
| 640×32 → 640×128, layer_norm | 20 | 1 | 1 (d = 4: one member holds all 4 slices) | 20 × 1 | 1 × 4 tiles | base-op narrow guard (`padded_w_tiles = 4 < 8`); runs are 1 tile per (row, slice) → 1-tile packets (TRANSPORT lamp) |
| 640×1792 → 640×7168 | 20 | 56 | 4 | 20 × 1 | 1 × 56 tiles (112 KB) | k = 8 costs 62 vs 59 |

Regime-pinned tests: the acceptance test pins k = 1 (narrow, 640×32), k = 4 symmetric (640×384), the dense assembly (`W_local = 50`, `17`), and layer_norm with a large row offset. The k > 8 owner path is pinned by the golden decode shapes (`(1,1,32,2048)`: trt = 1 → k ≈ 24). The predicates depend only on N ≥ 2·G, true on every BH / WH grid.

### Traffic ranking

Counted as crossings per tensor, most expensive tier first (fabric link, DRAM, cross-core NoC, core-local).

| Candidate split | Fabric (busiest link dir) | DRAM per chip (x / out) | Cross-core NoC per chip | Verdict |
|---|---|---|---|---|
| **R1**: raw slices relayed into compute L1 (mirror), local norm | (G−1)/2·S ring, (G−1)·S line — minimum | S / G·S — minimum | ports read outgoing chunks from L1 (each port its link load); k-member partial gather; notifications | **chosen**: minimum on both expensive tiers, one fabric round, full packets whenever a member covers one slice |
| normalize-then-gather into output DRAM | same + M·4 B partials | S / G·S | partial exchange only | rejected: two serial fabric rounds; 1–2-tile packets for a W-gather into interleaved pages; non-aligned slices cannot land |
| gather raw into DRAM, then norm | same | S + (G−1)·S landed / +G·S read / G·S | — | rejected: +2·G·S DRAM, serial |
| R1 with multicast transport | same bytes | same | no relay read | deferred: costs a 2-hop fence |
| R1 with landing at one core per row-group (k = 1 everywhere) | same | same | none | not a split choice: k is the base op's occupancy knob; k = 1 is chosen where it wins (narrow slices) |

Cheapest-traffic split = the implemented one. Occupancy (the k / row-group split) is decided separately by the base op's cost rule and never changes the fabric or DRAM crossings.

### Block schedule

Logical schedule per chip. Roles: **port** cores (per direction × link: reader on NCRISC / NoC0, sender on BRISC / NoC1) and **compute** cores (reader, compute, writer — the base op's three kernels, extended). The schedule is written per compute core (one block) and per port (a list of entries).

```cpp
// ---- compute core (row-group g, member m) — one block per call ----
load_affine_rows();                 // reader: gamma / beta columns of this member -> cb_gamma / cb_beta (once)
acquire_x_block();                  // reader: own member  -> read x block from input (DRAM / shard), zero tails, signal ports
                                    //         remote member -> wait sem_landed >= expected_landings, zero tails
member_moments_block();             // compute: rms -> sumsq; ln -> rough mean, pivot, centered moments  -> cb_partial
exchange_partials_block();          // writer: base-op all-gather (k <= 8) / owner combine (k > 8) -> cb_gathered
combine_finalize_block();           // compute: rms -> inv_rms; ln -> Chan combine -> inv_std, pivot K_g, shift
normalize_block();                  // compute: x -> out tiles (aligned: cb_out stream; non-aligned: cb_norm block)
assemble_dense_block();             // compute + writer, w_non_aligned only: shift-matmul + foreign contributions
store_block();                      // writer: output tiles -> output (TensorAccessor: DRAM or L1 shards)

// ---- port (direction d, link l) ----
ready_fence();                      // sender: one ready to the upstream port, wait own ready, decrement
for (entry_idx = 0; entry_idx < num_out_entries; ++entry_idx)       // (wave, shard) in send order, own leads relay by relay_lag_waves
    send_entry(entry_idx);          // reader: wait own_loaded (own) / sem_arrived >= in_entry+1 (relay); read runs from compute L1
                                    // sender: packets to the same core / offset on the neighbour; fused inc on the entry's last packet
notify_landed(in_entry_idx);        // reader, serviced whenever an incoming entry's arrival threshold is met (never starved by
                                    //   CB back-pressure or a relay wait): one local inc of sem_landed per member in the entry
drain_arrivals();                   // reader: wait total incoming, decrement sem_arrived by it
```

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `load_affine_rows` | member columns × 1 row of gamma (and beta) | — | once per call: ROW_MAJOR sticks: one 64/128 B read per tile column + RISC-V face-row scatter (non-aligned: element shift by `s_q`); TILE: 2 × 64 B row-0 reads per tile (base op) |
| `acquire_x_block` (own) | `core_row_tiles × member_w_tiles` | gamma | one `cb_reserve_back`, reads of a whole tile-row per barrier, one push; one local `noc_semaphore_inc` to each of the ≤ 2 ports sending this block |
| `acquire_x_block` (remote) | same | gamma | one reserve, one semaphore wait (`expected_landings` = number of entries covering this member: 1 when the member covers ≤ 1 slice, G−1 when it holds every remote slice), one zero-fill of slice tails, one push |
| `member_moments_block` (rms) | `grid(core_row_tiles, member_w_tiles)` | cb_x (not popped) | base op: one chain call (DEST-accumulated x², pack per tile-row) + one `reduce` (collapse) per block |
| `member_moments_block` (ln) | same, 2 passes | cb_x | pass A: Σx (chain DEST accumulation of `x·1` or `reduce` SUM ROW over the block) → rough mean → pivot `K_m = bf16(mean_m)` (pack to a bf16 column-0 tile = the rounding); pass B: `c' = (x − K_m)` (bf16/bf8b input: FPU sub bcast col, exact; fp32 input: SFPU) masked on slice-last tiles, accumulate Σc' and Σc'²; finalize per row: `mean_m = K_m + D1/n_m`, `M2_m = D2 − D1²/n_m` (column-0 SFPU, fp32) → 2 partial tiles per tile-row |
| `exchange_partials_block` | `core_row_tiles` (× 2 for ln) column-0 tiles → `k ×` that | cb_x | one round per call: base op symmetric gather (k ≤ 8: k·rows writes, one flush, k incs, one wait) / owner combine (k > 8); k = 1: skipped |
| `combine_finalize_block` | `ReduceInputBlockShape::of(core_row_tiles, k)` | cb_x | one `reduce` (Skip) per block + finalize post-op per tile-row; ln: Chan `μ = Σ n_m mean_m / W`, `M2 = Σ M2_m + Σ n_m (mean_m − μ)²`, `inv_std = rsqrt(M2/W + eps)`, `K_g = bf16(μ)`, `shift = −(μ − K_g)·inv_std` (column-0 SFPU, per-member scalars `n_m` from RT args) |
| `normalize_block` | `grid(core_row_tiles, member_w_tiles)` | cb_x (popped at end), row stats (popped at end), gamma / beta | one chain call per block. rms: `x · inv_rms(bcast col) · gamma(row)` (base op). ln: `(x − K_g)(bcast col) · inv_std(bcast col) + shift(bcast col)` then `· gamma(row) + beta(row)`; fp32 input: the first subtraction on SFPU |
| `assemble_dense_block` (`w_non_aligned`) | per home output tile j of each tile-row: ≤ 2 `matmul_tiles` of member source tiles by 0/1 shift matrices + ≤ `max_foreign` foreign tiles added | cb_norm (member block), shift matrices | one matmul init per block; one foreign-contribution exchange round per block (writer → home member's `cb_foreign`, one semaphore) |
| `store_block` | aligned: member block tiles at output tile ids `t·out_w_tiles + (q·sw + local col)`; dense: home tiles | — | batched writes `out_unit_tiles` per barrier (base op) |
| `ready_fence` (port) | — | — | once per call per port (TRANSPORT) |
| `send_entry` (port) | one (wave, shard) entry: the runs of the members of the wave's row-groups this port carries, packetized ≤ `chunk_pages` | `cb_port_stage` ring | per batch of `port_batch_chunks` packets: one read barrier + push (reader), one flush + pop (sender); per entry: one fused inc |
| `notify_landed` (port) | one incoming entry | — | per entry: one counter observation + one local inc per member in the entry |

**Stall-shadow check.** Stages that wait on data they do not produce: (1) the remote members' `acquire_x_block` (fabric landing), (2) `exchange_partials_block` (peers' partials), (3) the ports' `send_entry` for relays (upstream arrival) and for every send (the ready fence). Work independent of those waits, and where it runs:

| Wait | Independent work scheduled into it | Legal because | Status |
|---|---|---|---|
| ready fence (≈ 5 µs measured, every port) | own members: DRAM read of x, zero-fill, the whole `member_moments_block` (rms and ln); every member: `load_affine_rows`; port readers: stage the first own batch | the own slice needs nothing remote; moments are per member and commute with the combine | built |
| remote landing (per member) | the own member and every member whose wave already landed run their moments (each member starts the instant its block lands; waves make early row-groups land first) | moments are member-local | built |
| landing of later waves | the whole chain (moments → combine → normalize → store) of earlier waves' row-groups: the output DRAM write overlaps the remaining transfer | row-groups are independent | built (wave ordering) |
| partial exchange (per row-group) | nothing: every member of the row-group needs the combined statistic before normalize; the x·gamma pre-multiply commutes but needs a block-sized fp32 buffer (base op: rejected on L1 and precision) | — | stated: the tail after the last landing is acquire → moments → exchange → combine → normalize → store of one member block |
| relay arrival (port) | the port's next own entries (own leads relay by `relay_lag_waves = 1`) | own entries depend only on the ready fence | built |

### Perf lamps

| Lamp | Kind | Why the default may be wrong here | Nearby alternative to measure |
|------|------|-----------------------------------|-------------------------------|
| Wave size (`wave_row_groups`, from `INC_EVERY = 8` packets per entry) | OVERLAP | bigger waves = fewer fused incs but a longer exposed last wave and coarser relay lag; the example's 8-packet interval was measured for DRAM landing, not L1 | half and double the wave (4 / 16 packets per entry), and one row-group per wave |
| Relay lag (`relay_lag_waves = 1`) | OVERLAP | 0 stalls the sender on the first relay; 2 delays the far slice and the last wave | 0 and 2 |
| Tail chain block size (member block = whole slice, k_s = 1) | OVERLAP | the exposed tail after the last landing is one member block's moments + exchange + normalize + store; a narrower member halves the compute part but doubles combine members (grid-sync lamp) | k_s = 2 (k = 8) on the focus case; per-wave sub-blocking of the last wave only |
| Landing notification through the port reader's event loop | OVERLAP | adds a local hop and depends on the reader not being starved | fused inc straight to the member's `sem_landed` on the entry's last packet to that member (more router stalls, one fewer hop) |
| Own-loaded wait-all | OVERLAP | a port waits for every own member before its first own read; harmless while the fence is longer than the own load | per-wave own-loaded counters |
| Grid synchronization (k from the base rule) | OVERLAP | combine round per row-group (~0.5–1.5 µs, base op catalog) sits on the tail; for k = 4 every member waits on the slowest landing | k = 1 with 4× the member block (no combine) on the focus case |
| Narrow-slice packetization (`sw` small, k = 1 member holding all slices) | TRANSPORT | runs are 1 tile per (row, slice) → 1-tile packets, packet-rate-bound (12.8 GB/s per link, rule 1) | slice-major cb_x layout (`[q][t][j]`) with per-slice chain calls; or k = G members (one slice each) so a member's block is one run; or `scatter_write` (unmeasured) |
| Links at small payloads | TRANSPORT | the example: 2 links do not help ≤ 8 KiB per link; the indexer case is 41 KB per slice | `num_links = 1` vs 2 on the 640×32 case (the caller's choice; report both) |
| Batch depth (`port_batch_chunks = 4`) | TRANSPORT | entries here are shorter than the example's shards | 2 and 8 |
| Ports read outgoing chunks from compute L1 | TRANSPORT | L1→L1 reads from up to 20 cores per entry vs the example's bank-run DRAM reads | read the own slice from DRAM directly (+S DRAM, no own-loaded handshake) |
| Per-core runtime args (dispatch) | — | ~3 µs flat per call for per-core args (prompt); harmless while device time ≫ 3 µs, flagged DISPATCH-BOUND otherwise | common args + core index derivation; CoreRangeSet merged into rectangles |
| Finalize SFPU scope / partial faces-only / signal-after-flush | local | ported from base op Perf 1 (already measured there) | — (port as is) |

**Refinement 4 measurements (focus case, eval `ccl_perf`, fastest / slowest chip; baseline 20.74 / 21.51 µs):**
own-loaded wait-all → per-entry bit masks (`OWN_LOADED_MASK`, member i of a port adds `1 << i` to `sem_own_loaded`, an
own entry waits for exactly its members' bits; wait-all beyond 31 members) **20.64 / 21.35 (kept)**; relay lag 0 / 2
25.72 / 21.93 (1 kept, `AGRMS_RELAY_LAG`); wave 12 row-groups 22.58 (8 kept; 4 / 16 measured in Refinement 3);
last-wave relay lag 0 (`TAIL_RELAY_LAG`) 22.41; tail chain block size = the last wave's row-groups at the next wider k
(`TAIL_K_STEPS = 1`: k 4 → 8, 6-tile members, two kernel sets, merged CB plan) 21.94 / 23.24 (parked at 0: the wave-1
relays land as late as the last wave, so the narrow groups do not hold the end, and the last 8-member group's partial
exchange measured ~4 µs late — unexplained, zone markers unreliable on those cores); x·γ computed before the
combine (`XG_FIRST`) 21.00 (parked: on the last-landing member the pass is longer than the exchange it should hide);
gamma finished before the exchange (`AFFINE_FIRST`) null (kept). Landing notification fused to the member: not built —
one fused increment per member per entry multiplies the receiving router's write-ack stalls (`INC_EVERY` 4 measured
+2.0 µs in Refinement 3). Breakdown after the refinement: launch → first packet ≈ 2 µs (connection open ≈ 1.1, fence
≈ 0.6–0.9), fence end → last landing ≈ 14 µs (link), last landing → end ≈ 5.3 µs (moments 1.1, exchange 0.5, combine
0.2–0.3, normalize 2.5 — compute-bound: two HiFi4 FPU multiplies per tile, unchanged with the stores stubbed or the
srcA format switches removed — store 0.4).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| gamma / beta DRAM → `cb_gamma` / `cb_beta` | gamma dtype, row-0 compact pages (base op Perf 2 layout: row 0 of faces 0 / 1) | member reader, `TensorAccessor`; sticks: stick id `j` = dense output tile column, read whole stick, scatter elements 0–15 / 16–31 into face 0 / face 1 row 0; TILE: base op's 2 × 64 B row-0 reads | non-aligned: member's slice-local tile i needs dense columns `[q·W_local + 32i, +32)` → read sticks `⌊(q·W_local + 32i)/32⌋` and the next, copy element-wise with offset `s_q` (RISC-V, ≤ 32·member_w elements); columns past the slice's valid width zero |
| own x → `cb_x` | input dtype tiles | own member reader, `TensorAccessor` over the input (DRAM interleaved or L1 shards — one code path), tile id `t·sw + (local col)`; zero-fill columns ≥ v of the slice-last tile; push; `noc_semaphore_inc` each sending port's `sem_own_loaded` | the shard is not this core's block in general (R-zero-copy deferred) |
| own `cb_x` → neighbour `cb_x` (**TRANSPORT**) | input dtype tiles, raw | port reader `noc_async_read` from `(compute core, cb_x_addr + offset)` into `cb_port_stage`; port sender: one packet per run piece ≤ `chunk_pages`, `to_noc_unicast_write` to `get_noc_addr(core.x, core.y, cb_x_addr + offset)` **on the neighbour** (same logical core → same NoC coords, same address), fused inc to the neighbour port's `sem_arrived` on the entry's last packet | `cb_x_addr` = the port's own `get_write_ptr(cb_x)` (identical everywhere by construction) |
| relay (**TRANSPORT**) | same | port reader waits `sem_arrived ≥ in_entry + 1`, reads the landed runs from this chip's compute cores, sender forwards | never touches DRAM |
| landing → compute (**OVERLAP**) | — | port reader `noc_semaphore_inc(member, sem_landed, 1)` per member of each landed incoming entry; member reader waits `sem_landed ≥ expected_landings`, zero-fills slice tails in its copy, pushes cb_x | the zero-fill can race a relay read of the same block: it only rewrites padding, which every receiver zero-fills again |
| compute passes | T_acc (Float32 when `fp32_dest_acc_en`, Float16_b otherwise) column-0 stat tiles | helpers + base-op raw post-ops | — |
| partial exchange | T_acc tiles (faces 0/2 payload) | writer, base op `exchange_partials_block` (symmetric k ≤ 8; owner k > 8) | intra-chip only; partials never cross the fabric |
| out → output | output dtype tiles | writer `TensorAccessor` over the output (DRAM interleaved or L1 shards); aligned: tile id `t·out_w_tiles + q·sw + j` | the output is placed per `memory_config` / `output_tensor` |
| dense assembly (`w_non_aligned`) | output dtype tiles | compute shift-matmul; foreign contributions: writer `noc_async_write` into the home member's `cb_foreign` slot (lockstep slot address = own `get_write_ptr`) + semaphore | home member of output tile j = member owning dense column 32j |

Placement axis (`memory_layout`) as the physical realization of the logical shards:

| `TensorMemoryLayout` (input / output) | Axis it cuts | Character | Unlock class | Realization |
|---|---|---|---|---|
| INTERLEAVED | — | — | built | as above |
| HEIGHT_SHARDED | row | independent | **knob-turn** | input: own members read through the sharded `TensorAccessor` (local NoC, not DRAM); output: writer through the output's sharded accessor. The fabric part is unchanged (it never touches the input or output tensors) |
| WIDTH_SHARDED | W_local (input) / W (output) | dependent | **knob-turn here** (not a scheme-change): the dependent-axis combine is already the gather + member combine; a width shard only changes which core the own member reads from / the writer writes to | same accessor path |

Structural impossibilities: none beyond `feature_spec.py`'s INVALID.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a member block: `core_row_tiles` tile-rows × `member_w_tiles` slice-tile columns; for the fabric, a (wave, shard) entry |
| Grid | `device.compute_with_storage_grid_size()` (runtime); port cores from the example's `plan()` (probe-based, `_worker_nearest_noc1`); `ports_union` = union over every chip of the mesh; `compute_cores` = the rest, row-major |
| Row-groups | `num_row_groups = min(trt, N // k)`; `ttnn.split_work_to_cores` semantics over row-groups (2 classes, ceil / floor); group g = compute cores `[g·k, g·k + k)` of the list; member m of every group owns the same slice columns |
| Per-core work | tile-rows `[group_row_start, + core_row_tiles)`, slice-tile columns of member m; one block |
| Remainder | rows: two classes, the CB capacity is the class max, pushes/waits are the actual count (single block → ring-wrap trivially closed); W: ragged last member within a slice (`sw − (k_s−1)·ceil(sw/k_s)`); geometry per image with `ceil` |
| Ports | ring: 2 directions × L per chip; line: same core set on every chip, end chips run one direction (empty entry lists on the other). Kernels on `ports_union` everywhere; a port core idle on this chip gets zero entries |
| Fabric schedule | `_schedule(p, G, ring, balance=True)` per chip (the example's); its shard list becomes, per (direction d, link l), the entry list `[(wave w, shard q, part)]` over the row-groups r with `r mod L == l` (and, for the far half on an even ring, `(r div L) mod 2 == (0 if d == fwd else 1)`), ordered wave-major with own entries leading relay entries by `relay_lag_waves` |
| Incoming entries per port | the upstream port's outgoing list, in its order: thresholds `e + 1`, the members to notify, and whether the entry is relayed onward (host-computed RT args) |
| Idle cores | compute cores beyond `num_row_groups·k` and unused port cores: not in any kernel's range (except the ports_union kernels with zero entries) |
| Kernel descriptors | compute cores: reader / compute / writer over `compute_cores_used` (merged rectangles); ports: port_reader (NCRISC, NoC0) and port_sender (BRISC, NoC1) over `ports_union`; one `ProgramDescriptor` per chip with identical CB lists and core sets, differing only in RT args and fabric connections |

## Circular Buffers

T_in / T_out / T_g = tile bytes of input / output / gamma dtype; T_acc = Float32 4096 B (`fp32_dest_acc_en`) or Float16_b 2048 B; `rows = core_row_tiles_max`, `mw = member_w_tiles_max`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x` | 0 (first descriptor) | T_in | `rows · mw` | spans row × member W: the landing slot and the resident block | input dtype | reader (DRAM read or landing wait; the fabric writes the bytes) | compute | call (one block) |
| `cb_port_stage` (ports) | 1 | T_in | `2 · port_batch_chunks · chunk_pages` | streams over entries | input dtype | port_reader | port_sender | per batch |
| `cb_gamma` (has_gamma) | 2 | compact row-0 page (base op) | `mw` | spans member W | gamma dtype | reader | compute | call |
| `cb_beta` (has_beta) | 3 | same | `mw` | spans member W | gamma dtype | reader | compute | call |
| `cb_ones` | 4 | bf16 tile | 1 | constant (reduce-tile ROW scaler 1.0, base op `CB_ONES`) | Float16_b | reader | compute | call |
| `cb_tail_mask` (ln ∧ v < 32) | 5 | T_acc | 1 | constant 0/1 column mask of a slice-last tile | T_acc | reader | compute | call |
| `cb_moment_a` | 6 | T_acc | `rows` | spans row (rms: Σx² tiles; ln: Σx tiles) | T_acc | compute | compute | pass → collapse |
| `cb_moment_b` (ln) | 7 | T_acc | `2 · rows` | spans row (Σc', Σc'²) | T_acc | compute | compute | pass B → member finalize |
| `cb_pivot` (ln) | 8 | bf16 tile | `rows` | spans row (K_m, then reused for K_g — disjoint lifetimes) | Float16_b | compute | compute | per pass |
| `cb_partial` | 9 | T_acc | `n_stat · rows` (`n_stat` = 1 rms, 2 ln) | spans row | T_acc | compute | writer | moments → exchange |
| `cb_gathered` (k > 1) | 10 | T_acc | `n_stat · k · rows` (owner: base op's owner sizing) | spans row × member | T_acc | writer | compute | exchange → combine |
| `cb_rowstat` | 11 | T_acc | `n_rowstat · rows` (rms: inv_rms; ln: inv_std, shift) | spans row | T_acc | compute | compute | finalize → end of normalize |
| `cb_out` | 12 | T_out | aligned: `out_depth · out_unit_tiles`; dense: `rows · mw` (as `cb_norm`) | aligned: streams; dense: spans the member block | output dtype | compute | writer (aligned) / compute (dense) | per unit / block |
| `cb_shift_mats` (dense) | 13 | bf16 tile | `n_shift_mats` (≤ 4 per member) | constant per call | Float16_b | reader | compute | call |
| `cb_foreign` (dense) | 14 | T_out | `rows · max_foreign_in` | spans row | output dtype | writer (of the contributing peer member; lockstep slot) | compute | assembly |
| `cb_contrib` (dense) | 15 | T_out | `rows · max_foreign_out` | spans row | output dtype | compute | writer | assembly |
| `cb_out_dense` (dense) | 16 | T_out | `out_depth · out_unit_tiles` | streams | output dtype | compute | writer | per unit |

Every CB descriptor is declared identically (same index order, sizes, core sets) on every chip; compute CBs over `compute_cores_used ∪ ports_union` is **required only for `cb_x`** (address identity for the landing); the others may be declared over their own core sets as long as `cb_x` stays first. Semaphores: compute — `sem_landed`, `sem_partials_arrived` (base op), `sem_inv_arrived` (owner combine), `sem_foreign_arrived` (dense) (program-local); ports — `sem_own_loaded` (program-local, declared before the fabric setup calls), `sem_arrived`, `sem_ready` (global, cached). Push / wait parity: cb_x one push of `rows_this·member_w_this` per call, compute waits the block; cb_port_stage reader pushes / sender pops batch counts; stat CBs `rows_this · n` per block; cb_gathered `n_stat·k·rows` (base op).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `compute_kernel_hw_startup` | — | API | cb_x, cb_x, cb_out | — | first statement |
| 1 | `member_moments_block` rms | `grid(rows_this, mw_this)` then `ReduceInputBlockShape::of(rows_this, 1)` | yes (base op) | cb_x (None/None, not popped) | cb_moment_a → cb_partial (k > 1) / cb_rowstat (k = 1, finalize applied) | cb_x resident |
| 2 | `member_moments_block` ln, pass A | `grid(rows_this, mw_this)` | yes | cb_x | cb_moment_a (Σx) → reduce → mean_m → cb_pivot (bf16 K_m) | cb_x resident |
| 3 | `member_moments_block` ln, pass B | `grid(rows_this, mw_this)` | yes (chain; fp32 input: SFPU sub) | cb_x, cb_pivot (bcast col), cb_tail_mask (slice-last tiles) | cb_moment_b (Σc', Σc'²) → reduce → member finalize → cb_partial (mean_m, M2_m) | cb_pivot popped |
| 4 | `exchange_partials_block` | `n_stat·rows` → `n_stat·k·rows` | raw dataflow (base op) | cb_partial | cb_gathered | — |
| 5 | `combine_finalize_block` | `ReduceInputBlockShape::of(rows_this, k)` (rms) / per-member column-0 SFPU (ln Chan) | yes (reduce) + raw post-op | cb_gathered (k > 1) or cb_partial (k = 1) | cb_rowstat (+ cb_pivot K_g for ln) | — |
| 6 | `normalize_block` | `grid(rows_this, mw_this)` | yes (chain) + base op raw `MulGammaRowBcast`-style dest-reuse row broadcasts for gamma / beta | cb_x, cb_rowstat (bcast col), cb_pivot (ln, bcast col), cb_gamma, cb_beta | cb_out | caller pops cb_x, cb_rowstat, cb_pivot |
| 7 | `assemble_dense_block` (dense) | per tile-row: `home_out_tiles` outputs, ≤ 2 source tiles each + foreign | raw (matmul_tiles) | cb_out (member block), cb_shift_mats, cb_foreign | cb_contrib (to peers), cb_out_dense | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| startup | raw_api (boot) | `compute_kernel_hw_startup` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | (cb_x, cb_x, cb_out) | — | — | — |
| moments (rms) pass | helper | `eltwise_chain` + `BinaryFpu<Mul, A, B, Dst::D0, DestAccumulation::PerRow>` + `PackTile` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:533`, `:485`, `:503`; `DestAccumulation` `:283` | A = B = `input(cb_x, None, None, Block)` (base op) | cb_x | cb_moment_a | shape `grid(rows_this, mw_this)` |
| collapse / combine | helper | `compute_kernel_lib::reduce` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:632` (`ReduceInputBlockShape` `:290`, `ReduceWithinTile` `:191`, `ReduceAlgorithm` `:156`) | base op's: `<SUM, REDUCE_ROW, …, BulkWaitBulkPop, …, ReduceTile (collapse, `cb_ones`) / AccumulateViaAdd + Skip (combine)>` | cb_moment_* / cb_gathered | cb_partial / cb_rowstat | `rows_this`, k |
| ln pass B centering (bf16 / bf8b) | helper | `BinaryFpu<Sub, input(cb_x), input(cb_pivot, BroadcastDim::Col, …, Col)>` then `DestReuseBinary` mul by `cb_tail_mask` (slice-last tiles) | `chain.hpp:485`, `:499`; `BroadcastDim` `:305` | — | cb_x, cb_pivot, cb_tail_mask | DEST → pack | shape |
| ln centering (fp32 input) | helper | `CopyTile<input(cb_x)>` (UnpackToDestFp32) + broadcast pivot tile + `SubBinary` | `chain.hpp:482`; `ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp:26` | fp32-exact `x − K` | cb_x, cb_pivot | DEST | shape |
| squares | helper | `Square` (SFPU) or FPU self-mul | `ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp:35` | — | DEST | DEST | — |
| finalize post-ops | raw_api in reduce callback | `binop_with_scalar_tile_init`, `mul_unary_tile`, `add_unary_tile`, `sub_unary_tile`, `rsqrt_tile_init`, `rsqrt_tile` | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:156`, `:60`, `:35`, `:47`; `eltwise_unary/rsqrt.h:21`, `:44` | `1/W`, `eps`, `1/n_m`, `n_m` as fp32 bits RT args | DST | DST | — |
| normalize | helper + base-op raw | `eltwise_chain` with `BinaryFpu` (bcast Col) and base op `MulGammaRowBcast` (`base_ops/rms_norm/kernels/rms_norm_compute.cpp:323`) for row broadcasts (gamma; beta add analogous) | `chain.hpp:485`, `:499` | — | cb_x, cb_rowstat, cb_pivot, cb_gamma, cb_beta | cb_out | shape |
| dense assembly | raw_api | `mm_init` / `matmul_tiles` | `tt_metal/hw/inc/api/compute/matmul.h:166` | per output tile: 2 source tiles × shift mats | cb_out, cb_shift_mats, cb_foreign | cb_out_dense, cb_contrib | `home_out_tiles` |
| tail zero-fill / mask build | helper | `fill_l1_range` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:90` | byte ranges of columns ≥ v | cb_x / cb_tail_mask | same | — |
| reduce scaler | helper | `prepare_reduce_scaler<PoolType::SUM, ReduceDim::REDUCE_ROW>` (pool-type-aware) | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:59` | 1.0, bf16 packed | — | cb_ones | — |
| x / gamma / out access | raw_api (dataflow) | `TensorAccessor`, `noc_async_read/write`, barriers | `tech_reports/tensor_accessor/tensor_accessor.md` | — | DRAM / shards | CBs | — |
| partial exchange | raw_api (dataflow) | base op writer (`noc_async_write`, `noc_semaphore_inc`, `noc_semaphore_wait_min`) | `base_ops/rms_norm/kernels/rms_norm_writer.cpp` | — | cb_partial | cb_gathered | k, rows |
| fabric port sender | raw_api (fabric) | `WorkerToFabricEdmSender::build_from_args`, `PacketHeaderPool::allocate_header`, `fabric_set_unicast_route`, `to_noc_unicast_write` / `to_noc_fused_unicast_write_atomic_inc` / `to_noc_unicast_atomic_inc`, `send_current_slot_non_blocking`, `noc_async_writes_flushed` | `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`, `tt_metal/fabric/hw/inc/packet_header_pool.h`, `tt_metal/fabric/fabric_edm_packet_header.hpp`; pattern: `examples/fabric_all_gather/program_descriptor_with_inline_kernels.py:172` (`_SENDER_SOURCE`) | — | cb_port_stage | peer L1 | `port_batch_chunks`, `chunk_pages` |
| fabric port reader | raw_api (dataflow) | `noc_async_read` from compute L1, `noc_semaphore_wait_min` / non-blocking poll (`invalidate_l1_cache`), `noc_semaphore_inc` (notify, decrement) | `tt_metal/hw/inc/api/dataflow/dataflow_api.h`; pattern: `fabric_all_gather/…:89` (`_READER_SOURCE`) | — | compute cb_x | cb_port_stage | entries, waves |
| host fabric setup | host | `ttnn.setup_fabric_connection`, `ttnn.get_forwarding_link_indices`, `mesh_device.get_fabric_node_id`; placement `plan()` | `fabric_all_gather/…:717`; `_schedule` `:384`; `probe_ethernet_cores` `:436` | appended last in the sender's RT args | — | — | L |

Helpers considered and rejected (raw_api entries):

| raw_api entry | Candidate helper | Mismatch (file:line) | Reason |
|---|---|---|---|
| finalize / Chan post-ops | `eltwise_chain` SFPU elements (`Rsqrt`, `eltwise/unary/math.hpp:38`) | `reduce()` exposes its post-reduce hook as a `uint32_t dst_idx` callable (`reduce_helpers_compute.hpp:632`) | a chain would need the reduced tile packed out and re-read; the hook is the helper's extension point (base op) |
| gamma / beta row broadcast | `DestReuseBinary` | `chain.hpp:499`: `DestReuseBinary` takes an `InputSpec` without `BroadcastDim` | base op's `MulGammaRowBcast` (`rms_norm_compute.cpp:314`) closes the gap; port it and its add twin |
| dense assembly | `matmul_block` (`matmul_block_helpers.hpp`) | dense K: an output tile would multiply all `mw` source tiles, of which 2 are nonzero | banded 0/1 shift: 2 `matmul_tiles` per output tile; the block op is the custom realization |
| partial exchange | `ttnn.Mcast` / CCL helpers | base op: k ≤ 8 one symmetric unicast round wins (catalog `tensix_all_reduce`); `kernel_lib/ccl` is chip-to-chip | as base op |
| port sender | `kernel_lib/ccl/ccl_helpers_dataflow.hpp` `FabricStreamSender` | unmeasured (`blackhole-ccl-kernels.md` §12); the measured path is the example's raw sender | copy the measured raw pattern; the typed helper is a refactor candidate |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| rms moments | Mul (acc to DEST) | cb_x: All | cb_x: All | None |
| ln pass B | Sub | cb_x: All | cb_pivot: Col0 (bf16) | Col |
| ln pass B mask | DestReuse Mul | DEST: All | cb_tail_mask: All | None |
| rms normalize | Mul | cb_x: All | cb_rowstat (inv_rms): Col0 | Col |
| ln normalize | Sub, Mul, Add | cb_x: All | cb_pivot (K_g): Col0; cb_rowstat (inv_std, shift): Col0 | Col |
| gamma / beta | DestReuse Mul / Add | DEST: All | cb_gamma / cb_beta: Row0 | Row |
| combine (rms) | AccumulateViaAdd + Skip | cb_gathered: Col0 | — | — |
| dense assembly | matmul | cb_out: All | cb_shift_mats: All (0/1) | — |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Landing address differs between chips / cores | the sender writes at its own `cb_x` address on the peer | `cb_x` first in an identical CB list over identical core sets on every chip; bring-up check: run a probe build where each member writes `get_write_ptr(cb_x)` to a debug output and compare across cores and chips |
| Port cores differ per chip | Ethernet channels per direction differ chip to chip | `ports_union` over the whole mesh is excluded from compute on every chip; port kernels declared on `ports_union` everywhere |
| Landing before the receiver's program runs | another op's L1 would be overwritten | the ready fence (one per port, neighbour-only, the example's): nothing is sent before the downstream port's ready, sent only after its program started; every chip receives only from neighbours (relay) |
| Notification starvation | the port reader blocks on CB space / a relay arrival while a final-hop entry already landed | the reader's loop polls (non-blocking `cb_pages_reservable_at_back` / semaphore reads with `invalidate_l1_cache`) and services notifications before blocking |
| Counter left non-zero | traced repeats hang or relay early | decrement exactly what was consumed; `drain_arrivals` waits the host-computed total; ready consumed once per port per call |
| Program-cache collision | same kernels and core ranges, different RT-arg layout (ring vs line, wave count) | `custom_program_hash` over the plan |
| LN precision at large mean | FPU truncates operands to tf32; `E[x²] − μ²` cancels | pivot-shifted moments (bf16 pivot → exact FPU subtraction for bf16 / bf8b, SFPU for fp32), Chan combine of centered member moments, normalize as `(x − K_g)·a + b`; acceptance test has a mean-50 / std-1 case |
| Padding in LN after centering | a padded zero becomes `−K` | `cb_tail_mask` multiply on slice-last tiles in pass B; n_m counts valid columns only; `1/W` true W |
| Partials precision clause | prompt: per-row partial sums stay fp32 on the wire | no partial crosses the fabric (raw slices do, in the input dtype — at least the input's precision); with `fp32_dest_acc_en=True` every stat CB is Float32; with False the base op's 16-bit intermediates (Refinement 1) |
| Dense assembly on fp32 output | matmul truncates the normalized values to tf32 | relative 2⁻¹¹ on unit-scale values, inside the fp32 tolerance (0.9999 / 0.005); applied after normalization only (never on raw x) |
| Narrow slices → tiny packets | 1-tile runs at k = 1 | correct; TRANSPORT lamp |
| Line ends | end chips send in one direction only | `_schedule` returns empty lists; their idle ports run zero entries but still sit in `ports_union` |
| Ring on G = 2 | a 2-chip ring is the line | `plan()` treats it as a line (example) |
| num_links=None | must be the usable minimum over every hop | `min(len(get_forwarding_link_indices(a, b)))` over the group's hops |
| Hang triage | many parties | `blackhole-ccl-kernels.md` §14 signatures; port sender in `wait_for_empty_write_slot` = route / receiver stall; member reader on `sem_landed` = a missing notification |

## Implementation notes (Phase 0, as built) — deviations and realizations

Binding elements implemented as designed: R1 mirror_resident_gather (raw slices relayed by port cores straight into
the compute cores' `cb_x`, same core / offset on every chip; `cb_x` first CB over compute_used ∪ ports_union),
balanced ring / line schedule of `fabric_all_gather`, one port (reader NCRISC/NoC0 + sender BRISC/NoC1) per
(direction, link) placed by the probe next to its Ethernet core, ready fence, one fused arrival increment per
(wave, shard) entry, relay with `relay_lag_waves = 1`, landing notifications serviced by the port reader's polling
loop, row-groups × k members over `worker grid − ports_union`, base-op symmetric partial exchange, LN pivot-shifted
member moments + Chan combine, dense assembly by home member + foreign contributions for `w_non_aligned`.

| Item | Design | As built | Why |
|---|---|---|---|
| k > 8 (owner combine, base op R3) | built | **built in Refinement 1**: row-form partials + unicast return (base op Perf 1 interleaved return; no root multicast / owner spread yet); LN owner Chan combine pivot-centered (not in the base op) | the decode shapes now pick k ≈ 16–32 |
| `out_unit_tiles` | `min(mw, 16)` | cap 4, halved down to 1 when the CB plan does not fit L1 | measured: a 12-tile unit kept the writer from overlapping the normalize of the same row (focus tail); L1 pressure from L1-sharded inputs/outputs in the golden matrix |
| gamma / beta load | reader, compact 192 B pages | writer, full-tile pages; tile-aligned slices: base-op direct row-0 reads issued at start and finished after the partial exchange; non-aligned: staged sticks + RISC-V element shift | the first cut's element scatter cost ~11 µs ahead of the exchange (zones); compact pages are an L1-only refinement |
| LN passes | chain / reduce helpers | aligned bf16/bf8b (`AGRMS_LN_FPU`): pivot = one block `reduce` (row stride = pitch); centering on the FPU into `cb_cent`; D1 / D2 by `reduce` + DEST-accumulated squares; normalize = FPU sub + `DestReuseBcast` COL/ROW. fp32 or non-aligned: raw SFPU loops (copy_tile UnpackToDestFp32 → exact x − K; slice tails masked by a row-0 mask) | the all-SFPU path costs ~10 element inits per tile (indexer 22.7 µs, zones: ~14 µs of LN compute on a 1×4-tile block); FPU subtraction is exact only for bf16/bf8b operands |
| LN combine (k = 1) | Chan over k | shortcut μ = mean_1, M2 = M2_1 | saves ~8 full-tile SFPU ops per row |
| `cb_pivot` | one CB (K_m then K_g) | `cb_km` + `cb_kg` | both live in a multi-row block's combine window |
| dense assembly | ≤ 4 shift matrices per member | generic `(e0, e1, delta)` 0/1 matrices per member (host-enumerated, incl. identity for foreign tiles), output-dtype format | W_local < 32 makes an output tile span > 2 slices; one generic matrix form covers every case |
| sharded input / output | accessor path (zero-copy deferred) | as designed | — |
| program cache | `custom_program_hash` | sha1 of every structural input (shapes, dtypes, memory configs, cfg, mesh, ports_union, k, waves, compute config) + coord | — |

Measured (eval `ccl_perf` protocol via `tests/ttnn/unit_tests/operations/all_gather_rms_norm/perf_agrms.py`,
5 experiments, fastest chip, QuietBox 2 1x4 ring, 2 links): focus 640×384 rms **21.25 µs** (roofline 7.6,
target 10.1); indexer 640×32 layer_norm **16.16 µs** (roofline 0.6). Not dispatch-bound.
Zone breakdown of the focus case (profiler on; one call): kernel start ≈ 2.8 µs, port connection open done ≈
5.0 µs, fabric send ≈ 12.3 µs per port (≈ 30 GB/s = 62 % of a link: TRANSPORT), last wave lands ≈ 18.7 µs, tail
after the last landing ≈ 5.8 µs (moments 1 µs, exchange + combine ≈ 1.1 µs, normalize ≈ 3 µs, last store
≈ 0.5 µs: OVERLAP).

## Verification notes (Phase 0 verifier)

| Item | Finding | Disposition |
|---|---|---|
| FPU normalize chain (`DestReuseBcast`, rms gamma and the LN FPU path) | DEST → srcA moved the fp32 intermediate in the chain head's srcA format (cb_x: bf16 / bf8b), truncating it at every dest-reuse step: per-row scale bias −0.6 % (LN, 4 steps), −0.3 % (rms bf16); the SFPU LN path was unbiased | fixed: the first dest-reuse step of a chain switches srcA (unpack + math) to the DEST width, the last restores cb_x's (a math-only switch was measured ineffective). LN bf16 rel RMS 0.0072 → 0.0019, rms bf16 0.0042 → 0.0019; cost +0.17 µs on the indexer LN case (16.16 → 16.33), +0.05 µs on the focus case |
| EXCLUSIONS carried from the base op | `{bfloat8_b, h_non_aligned}` measured unnecessary (row-local norm, the H tail is never zero-filled: 241 / 241 sampled cells pass); `{bfloat8_b, w_non_aligned}` measured necessary (238 / 238 inf) | the first dropped, the second kept |
| k > 8 owner combine (base op R3) | deferred by the implementer (candidates capped at 8) | the decode / wide-hidden shapes with L1-sharded I/O (and 128×8192 interleaved LN) OOM in `select_split`: op_requirements.md Refinement 1 |
| LN with `fp32_dest_acc_en=False` | per-row offset rms 0.012–0.020 (μ rounded to bf16 before x − μ) + per-row scale +1 % (16-bit variance accumulation); residual after both 0.003 | op_requirements.md Refinement 2 (two-word mean through the Chan combine; variance accumulation order) |
| Zone breakdown, indexer LN case (single un-traced call, least-skewed chip) | last landing ≈ 14.8 µs after the first zone; local LN tail after it ≈ 8.5 µs (moments 1.5–3, k = 1 Chan combine ≈ 3.4 µs of full-tile SFPU on column-0 data, normalize ≈ 1.7, store) | op_requirements.md perf refinements |

## Refinement 1 deviations (as built)

| Item | Base op / plan | Built | Why |
|---|---|---|---|
| Owner return | root multicast (sharded) / single-owner multicast + owner spread (Perf 1/2) | unicast faces 0/2 to every member, one `SEM_INV` inc per row; no owner spread | correctness port first; the multicast / spread are perf levers for the decode tail (OVERLAP refinements) |
| LN owner combine | — (base op has no LN) | row form: pivot K = bf16(mean_0), centered weighted terms (SFPU fp32), member sums by a ones-row matmul, transpose_dest to column form | every FPU operand is centered (big row offsets stay exact); the pivot needs no extra exchange round |
| L1 budget | `cb_l1_budget` = distance to the lowest L1 buffer | + persistent global semaphores relocated when free L1 opens above them | they were the lowest buffer after the first sharded call's tensors were freed (measured) |
| LN moments path | FPU fast path for aligned bf16 / bf8b | FPU unless no split fits and DEST is fp32 → SFPU path | `cb_cent` is block-sized in the accumulator format |

## Refinement 2 deviations (layer_norm with 16-bit DEST)

| Item | Phase 0 / plan | Built | Why |
|---|---|---|---|
| Member moments (16-bit DEST) | pass A pivot + pass B Σc', Σc'² DEST-accumulated (FPU / SFPU), member finalize on SFPU | `moments16_block`: each x tile transposed into DEST (unpack), loaded 4 columns at a time with the Welford LLK's SFPTRANSP pattern (one LREG lane per row), shifted sums S1 = Σ(x − K), S2 = Σ(x − K)² with K = the row's first sample kept in fp32 LREGs across all member tiles (slice-last tiles: only their valid columns); finalize mean_m = K + S1/n, M2_m = S2 − S1²/n in fp32, stored once with RNE | every 16-bit DEST write rounds and an SFPU store TRUNCATES: tile-by-tile accumulation under-estimated σ² by 1–3 % (per-row scale +0.3…1.3 %), growing with member width |
| Chan combine (16-bit DEST) | per-member SFPU op chain in DEST (truncating at every step) | `combine16_block` (symmetric, k = 1) / `owner16_ln` (k > 8): members' (mean_m, M2_m) as row-form LREGs, one-pass pivoted Chan in fp32 (P = member 0's mean, A = Σ n_m(mean_m − P), B = Σ M2_m + n_m(mean_m − P)²), finalize mu, inv_std (fp32 rsqrt body), K_g = RNE bf16(mu), shift = (K_g − mu)·inv_std, each stored once with RNE | same truncation bias; the fp32 mu also keeps the sub-bf16 part of the mean in `shift` (the planned two-word mean, without a second payload word) |
| Row → column form | `transpose_dest` | pack to `cb_mu`, `transpose_tile` back (unpack transpose) | the generic Welford reduction's pattern; keeps `transpose_dest`'s replay / ALU-format programming out of the per-row SFPU config |
| Normalize (16-bit DEST, w_non_aligned) | SFPU chain when a slice tail is masked | FPU chain for any bf16 / bfloat8_b input with 16-bit DEST | the SFPU chain truncates at every store (−0.6 % per-row scale measured); normalize needs no tail mask (padding columns are dropped by the dense assembly) |
| Partial payload | (mean_m, M2_m), 2 tiles | unchanged | a single RNE bf16 mean_m costs ≤ 0.001 output offset rms (measured with row offsets N(0, 4)); the exchange / owner payload stay as built |

Raw TTI SFPU (documented helper bypass): no helper keeps a per-row accumulator in LREGs across tiles, and the
row-per-lane load needs SFPTRANSP. fp32 DEST keeps the Phase 0 / Refinement 1 path byte-identical.

## Refinement 3 deviations (TRANSPORT, as built)

| Item | Phase 0 / plan | Built | Why (measured, focus case0 640×384 rms, 1×4 ring, 2 links; eval `ccl_perf`, fastest chip) |
|---|---|---|---|
| Port reader | batches of `port_batch_chunks` packets: issue the batch's L1 reads, one read barrier, one push | **stream** (`PORT_READ_STREAM = 1`): every stage slot's read carries its own NoC transaction id (1..slots); up to the whole ring in flight; each slot is pushed, in order, the moment its own read has landed (`ncrisc_noc_read_with_transaction_id_flushed`), polled between notifications | 21.29 → 20.98 µs. A per-packet barrier instead (`PORT_PUSH_PACKETS = 1`, batch mode) measured 24.94: one 14 KB L1→L1 read in flight is latency-bound, so reads must overlap. Batch mode stays as the `PORT_READ_STREAM = 0` path |
| `port_batch_chunks` (sender flush + pop granularity) | 4 (the example's DRAM-sourced optimum) | **2** | 20.98 → 20.75 µs; case1 (1-tile packets) 16.24 → 15.68. 1: 22.62, 3: 20.77, 6: 21.69. The ring depth itself is flat (`STAGE_BATCHES` 2 / 3 / 4 at batch 2: 20.73–20.75): the cost is the sender stalling on `noc_async_writes_flushed` for a whole batch |
| `INC_EVERY` (packets per entry) | 8 | 8 (unchanged) | 4: 22.75, 16: 23.69 µs (entry size also sets the wave: OVERLAP couples in) |
| Packet fill (`PORT_PACK`) | one packet per ≤ `chunk_pages` piece of one member run (12-tile run → 7 + 5) | knob built, **default 0** (= the original packets); 1 = fabric scatter write packing up to 4 run pieces of one entry into a full packet (fused scatter + inc for 2-piece entry-last packets, a separate flushed inc after a > 2-piece one), shared packer `all_gather_rms_norm_port_pack.hpp` | with reads and relay waits stubbed the send takes the same time packed (28 packets) or not (30): ≈ 10.3–10.45 µs per port, so the link is byte-bound here, not packet-rate-bound; in the real run packing costs +0.4 µs (two L1 reads per slot). Expected to matter for 1-tile runs (case1, Refinement 6) |

Transport breakdown after the refinement (zones, `tests/ttnn/unit_tests/operations/all_gather_rms_norm/zones_agrms.py`,
probe-only ablation bits `AGRMS_ABL`: 1 = no port L1 reads, 2 = relay entries do not wait): per port 368 KB in
≈ 11.2–11.6 µs real vs ≈ 10.4–10.6 µs with reads and relay waits both stubbed (≈ 35 GB/s per link direction: the
link floor of this fabric at 2 links, cf. `fabric_all_gather` ≈ 37 GB/s per link at 2 links); the remaining ≈ 1 µs is
the relay dependency (neighbour launch skew) and the ready fence ≈ 0.7–1.3 µs before the first send.

## Refinement 5 deviations (indexer layer_norm tail, as built)

Measured on `LOOSE_CASES[1]` (640×32 bf16 LN, gamma + beta RM, Ring 1×4, 2 links, fp32 DEST: k = 1, rows 1, pitch 4;
eval `ccl_perf`, 5 experiments, fastest chip): **15.64 → 12.07 µs** (slowest 16.39 → 12.82).

| Item | Before | Built | Why (measured) |
|---|---|---|---|
| k = 1 member finalize + Chan combine (fp32 DEST, FPU and SFPU moments paths) | `member_finalize_block` (copy ×3 + 5 full-tile SFPU init/op pairs) + `combine_block` (two DEST windows, ~10 more full-tile SFPU pairs, `cb_mu` round trip) | `ln_finalize_k1_block`: ONE window, column-0 SFPU scope (the base op's `finalize_col0_body` pattern: faces 0/2, even dst rows): t = D1/n, M2 = D2 − D1·t, inv_std = rsqrt(M2/W + eps), shift; block pivot: K_g = K_m (the pivot reduce packs straight to `cb_kg`), shift = −t·inv_std | 15.64 → 12.59 µs (the largest single lever: ≈ 2.8 µs of full-tile SFPU on column-0 data). Precision identical (precision baseline rows unchanged to 5 digits) |
| Member pivot (LN FPU, fp32 DEST, non-owner) | bf16 mean of the whole member block, reduced after the last landing | `AGRMS_OWN_PIVOT` (default 1): bf16 mean of the member's OWN slice, reduced from `cb_own` (a second reader copy of the own tiles, pushed with the own read) BEFORE the landing; members without an own slice keep the block pivot. k = 1: K_m is staged into fp32 `cb_mstat` ahead of (D1, D2) before the landing and the finalize forms K_g = bf16_rne(mu), shift = (K_g − mu)·inv_std, so the normalize keeps its row-mean pivot | removes the 0.6 µs pivot reduce from the tail but adds the K copy / K_g pack to the finalize: 12.53 (block pivot, direct K_g) vs **12.42** (own pivot) — −0.1 µs, inside the noise band: kept as a live knob (correct, no regression). A bf16 `cb_km` copy inside the finalize cost +0.27 µs (format switch among UnpackToDestFp32 copies) → staged in `cb_mstat`. Precision: any bf16 value near the member mean is a valid pivot (M2 = D2 − D1²/n exact algebra); an own-slice pivot bounds the D2 cancellation by n_m / n_own (≤ G); bf16_ln 640×32 rel RMS 0.00187 → 0.00185 |
| Own-slice centered moments before the landing | — | **not built** | k = 1: the own slice is 1 of the member's 4 tiles; the remote columns are not contiguous around `own_off` (two segments on the middle chips), the reduce helper has no tile offset, and the post-landing moments cost is per-phase fixed overhead (centering 0.35 µs, the three reduce phases 0.9 µs for 4 tiles), not per tile: moving one tile saves ≤ ~0.2 µs and adds a segment's phases on two of four chips |
| LN FPU normalize (all DEST widths) | `eltwise_chain` per tile: 5 FPU steps, each re-initialized per tile, plus the srcA DEST-width switch pair per tile | blocked chain: `IterationShape::block_size(norm_block)`, output `PerBlockSize`; `DestReuseBcast(blk, Wt)` switches srcA on the block's first tile and restores after its last (default blk = 1 = per tile, the rms chains are unchanged) | 12.42 → **12.07 µs**; precision identical. `norm_block = gcd(writer unit, cb_out ring pages, every member's core_w)` (compute CT [6], cap `AGRMS_NORM_BLOCK`): a block is packed at contiguous offsets, so it must never straddle the ring end — the first cut (block = unit) wrote past `cb_out` on 2×4×128×512 (pitch 6, members 6 / 4, rows 2: 4 + 2 per row into an 8-page ring, PCC 0.85), found by the golden slice |
| Writer unit (`OUT_UNIT_CAP`) | 4 | 4, now an env knob `AGRMS_OUT_UNIT` | 2: 12.11, 1: 12.32 µs with the blocked normalize (before it: 2: 12.36, 1: 12.32 vs 12.42) — 4 kept (Refinement 4 tuned it on case0) |

Breakdown after (zones, chip 0, median of 20 traced calls): last landing 7.70 µs; moments 7.78 → 9.05; finalize → 10.14;
normalize 10.18 → 11.80; last store 12.36. The post-landing chain (≈ 4.6 µs for four tiles) is still made of fixed
per-phase costs (six phases: centering, D1 reduce, squares, D2 reduce, finalize, normalize), and the 7.7 µs up to the
last landing (launch, fence, 1-tile packets) is now the larger share — Refinement 6 (TRANSPORT).

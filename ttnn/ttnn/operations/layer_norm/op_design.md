# Operation Design: layer_norm

The specification is `eval/specs/layer_norm/spec.py`; where this design and it disagree, the
specification governs. The L1 ledger and the data-movement budget are in `l1_ledger.md` (same
directory).

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (fused: two row reductions + normalise + affine, one program) |
| Goal | Layer normalisation over the last dimension with gamma/beta, as one `ttnn.generic_op` dispatch of one Metal 2.0 `ttnn.ProgramSpec` on Quasar. Phase 0 = PHASE0 of the spec: bfloat16, TILE, rank 4, tile-aligned, DRAM interleaved, bfloat16 TILE weight/bias, `fp32_dest_acc_en` ∈ {True, False}. |
| Math | `mean_r = (1/W)·Σ_k x[r,k]`; `c[r,k] = x[r,k] − mean_r`; `var_r = (1/W)·Σ_k c[r,k]²` (biased); `out[r,j] = c[r,j] · rsqrt(var_r + ε) · γ[j] + β[j]`, with `γ = weight[0,0,:]`, `β = bias[0,0,:]` (spec R5, R7, R8) |
| References | spec `eval/specs/layer_norm/spec.py`; target `.claude/references/targets/quasar.md`; backend `.claude/references/backends/ttsim.md`; host model `.claude/references/quasar_op_examples/` (`toy_row_passes` is the template for rows dealt to NEOs, rounds, compute-owned rings, `pack_init`); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md`; `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; perf catalog `ttnn/ttnn/operations/examples/master.md` (entries `row_reduce_accumulate`, `eltwise_l1_vs_dest_accumulate`, `compute_fusion`, `compute_block_size`, `width_split`, `double_buffer`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (probe: `compute_with_storage_grid_size()` = 8-4) | target page; queried at run time |
| L1 per core | 4,194,304 B per cluster, shared by its 4 NEOs and 8 DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES` = 3,963,328 − 32,768 = **3,930,560 B** per cluster on this target (query value measured by probe on ttsim) | probe, 2026-11-21 |
| Ring cap | 1,048,560 B per DataflowBuffer ring (TRISC extent, `trisc_dfb_extent_bytes`). Note: at this tt-metal base the host check (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:1600-1666`) and the TRISC slot (`dataflow_buffer_interface.h:37-47`, `uint32_t ring_size`) allow up to unreserved L1; the design honours the page's smaller cap (a clamp that costs nothing on INPUTS, see Mechanism caps) | target page `family_facts` |
| Host program | `ttnn.ProgramSpec` (Metal 2.0) via `ttnn.experimental` (`ProgramSpec`, `KernelSpec`, `DataflowBufferSpec`, `WorkUnitSpec`, `ProgramRunArgs`) and `ttnn.generic_op` — reference `.claude/references/quasar_op_examples/` | target page |
| Threads per core | 4 NEOs (C ∈ {1, 2, 4}), 6 user DM harts (R + W ≤ 6), on STRIDED rings R \| C and W \| C; 32 DFBs per cluster, ≤ 24 active per NEO; 16 semaphores; kernel binary ≤ 24,576 B compute, ≤ 49,152 B DM | target page *Cluster model* / *Thread plan* |
| Unavailable here | (1) `ttnn.ProgramDescriptor` / legacy program-factory device op — refused at kernel creation (target page). (2) `bfloat8_b`, `bfloat4_b` — no path on this part (target page). (3) `ReduceFp32Mode::Accurate` — static_assert on Quasar (`reduce_helpers_compute.inl:66-90`). (4) `ReduceAlgorithm::AccumulateViaAdd` — not used. (5) reduce helper indexed policies (`WaitUpfrontNoPop`, `BulkWaitBulkPop`) under C > 1 — not used (`WaitAndPopPerTile` only). (6) reduce `post_reduce_op` — not used. (7) `add_unary_tile` / chain `AddUnary` — compiled out on Quasar (`tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:34-58`; only `mul_unary_tile` has a Quasar path). (8) FPU dest-reuse combined with a broadcast — static_assert (`tt_metal/hw/ckernels/quasar/metal/llk_api/llk_math_binary_api.h:56-59`, `:97-99`). (9) `bcast_row_idx ≠ 0` on row broadcasts (`tt_metal/hw/inc/api/compute/bcast.h:194-196`) — row 0 only. (10) ALL consumer access pattern with more consumer threads than producer threads — not used: no ring is shared by the NEOs of a cluster. (11) `pack_init(<ring>)` required when the output ring changes. (12) `UnpackToDest` of a bf16 page into 32-bit DEST — not used: default unpack mode only. (13) `fpu_math_fidelity` is set explicitly on every compute kernel. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes (positional) | Phase 0: bf16, TILE, rank 4, H and W multiples of 32, DRAM interleaved | — | tensor binding |
| `epsilon` | float (keyword-only) | no | any finite float ≥ 0 | `1e-12` | written by the reader into `cb_epsilon` (RT arg: fp32 bit pattern) |
| `weight` | `ttnn.Tensor` (keyword-only) | yes | `[1, 32, W]` (padded height exactly one tile, last dim = input W — X1, X2); Phase 0: bf16 TILE; any placement (R11) | — | tensor binding |
| `bias` | `ttnn.Tensor` (keyword-only) | yes | same shape, dtype and layout as `weight` (R9) | — | tensor binding |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` or device compute kernel config (R4) | no | any `math_fidelity`, any `math_approx_mode` (R3); `fp32_dest_acc_en` ∈ {True, False} for bf16; False refused for a float32 input (R1) | `default_compute_kernel_config()` = HiFi4, approx off, `fp32_dest_acc_en=True` (R2) | mapped to `ComputeHardwareConfig(fpu_math_fidelity=…, sfpu_precision_mode=…, enable_32_bit_dest=…)`; `fp32_dest_acc_en` also selects the intermediate page format (host) |

Entry-point flow (Python): `validate()` → resolve the config (`None` → `default_compute_kernel_config()`, read the three fields as attributes from either config type, R4) → allocate output (`ttnn.allocate_tensor_on_device(input.spec, device)`: input's shape/dtype/layout and buffer type, interleaved — R12) → build the ProgramSpec → one `ttnn.generic_op`. There is no `dim` parameter, so no index-axis canonicalisation.

`validate()` gates, in order: the AXES taggers (`INPUT_TAGGERS = AXES` imported from the spec), `dtype`, `layout`, `memory_config`/`sharding` from `eval.memory.memory_axes(input_tensor)` (R13), `affine_dtype`/`affine_layout` read off `weight` (R9), `fp32_dest_acc_en` (R1), against `SUPPORTED` → `NotImplementedError`. Malformed calls (X1: weight/bias last dim ≠ W; X2: weight/bias padded height ≠ 32, i.e. more than one tile high; bias dtype/layout ≠ weight's, R9) → `ValueError`. The resident-fit predicate `fits(C=1, bh=1)` (Blocking Model) failing → `NotImplementedError` (the streaming regime is deferred). `validate()` reads neither `math_fidelity` nor `math_approx_mode` (R3).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[N, Ch, H, W]` (Phase 0 rank 4; ranks 2/3 are a later knob-turn: leading dims fold into `num_tile_rows`) |
| Dtype | bfloat16 (Phase 0); float32 in TARGET |
| Layout | TILE |
| Memory | DRAM interleaved (Phase 0); L1 interleaved in TARGET |
| weight / bias | `[1, 32, W]`, TILE, bfloat16 (Phase 0), DRAM interleaved where the caller keeps them (read where they are, R11); only row 0 of each tile is used (R8) |

### Output

| Property | Value |
|----------|-------|
| Shape | same as input |
| Dtype | same as input (whatever weight/bias dtype, R9) |
| Layout | TILE |
| Memory | input's buffer type, interleaved (R12) |

## Blocking Model

Tile index space of the op: `(tile_row, w_tile)` where `tile_row ∈ [0, num_tile_rows)`,
`num_tile_rows = (Π leading dims) · ceil(H/32)` (per image, `ceil`), and `w_tile ∈ [0, tensor_w_tiles)`,
`tensor_w_tiles = ceil(W/32)`. The leading dims (N, Ch) and the H-tiles are treated identically and
are contiguous in page order (`page = tile_row · tensor_w_tiles + w_tile`), so they fold into the one
`tile_row` axis honestly. Inside a tile, the 32 rows are independent and the 32 columns belong to the
reduced axis; the block is never cut inside a tile.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_row` (N × Ch × H/32 folded) | **independent** — every output row depends on its own input row only (spec DEFINITION, "rows do not interact") | `block_tile_rows` (bh): tile-rows one NEO holds and drives through one pass of the phase sequence | `min(rounds_this_cluster_max, bh_fit(C))` — the coarsest that fits; = 1 for every Phase 0 INPUTS shape on Quasar (each NEO owns one row) and > 1 only when a NEO owns several rounds | host constant computed once in `create_program_artifacts` (`select_blocking()`), passed as CT arg `block_tile_rows`; ring sizes derived from it | **spread across clusters × NEOs**: rows dealt round-robin to the C NEOs of a cluster (NEO t owns local rows t, t+C, …), whole rounds of C rows dealt to clusters by `ttnn.split_work_to_cores(grid, num_rounds)`; grid is `device.compute_with_storage_grid_size()` (RT, queried) | knob-turn (bh, C, cluster count) |
| `w_tile` (W/32) | **dependent** — mean and variance span the whole row | `tensor_w_tiles` (Wt): whole extent, **not split** | `ceil(W/32)` (≤ 128 in INPUTS) | host, CT arg `w_tiles` | not assigned across cores or NEOs: a row's statistics stay on one NEO (no combine) | scheme-change (W-split + cross-core combine: regime `w_split_combine`, deferred) |
| γ / β vector (W) | **reuse-shared** — the same `[W]` vectors feed every `tile_row` (the operand-reuse check: they do not vary along the split axis) | resident `tensor_w_tiles` tiles per NEO, loaded once per kernel | whole vector resident | derived from Wt | every engaged NEO holds its own copy (no ALL ring here) | reuse scheme-change (fan-out: regime `affine_fanout`, deferred successor) |
| per-row statistics (row_partial, mean, var, rstd — data produced by the scheme) | **independent** along `tile_row`, one tile per tile-row | follows `block_tile_rows` | bh tiles | derived | computed on the NEO that owns the row; no stage is left on a designated core (there is no collector role in this regime) | — |

**Interleaved default, two ordered steps.** (1) Occupancy: deal rows over clusters first and over
the NEOs of a cluster second (see *Thread plan selection*), so every engaged NEO owns
`rounds = ceil(rows_this_cluster / C)` rows and no grid capacity is left idle while rows remain.
(2) Coarsest block: `block_tile_rows = min(rounds_max, bh_fit(C))`, i.e. a NEO's whole share is one
block whenever it fits; otherwise it is cut into `blocks_per_neo = ceil(rounds / bh)` blocks with a
ragged last block of `rounds − (blocks_per_neo − 1)·bh` rows (same block operations, shorter runtime
extent). Every knob is a parameter: the loop is
`for (block_idx = 0; block_idx < num_blocks_this_core; ++block_idx)` with the extent from CT/RT args,
never a literal.

**Thread plan selection (single source: `THREAD_PLAN = (4, 4, 2)` host constant + `select_blocking()`).**
`C_eff` = the smallest c ∈ {1, 2, 4}, c ≤ `THREAD_PLAN.C`, with `ceil(num_tile_rows / c) ≤ num_clusters`
(else `THREAD_PLAN.C`); then while `not fits(c, 1)`: c //= 2. `R_eff = min(THREAD_PLAN.R, C_eff)`,
`W_eff = min(THREAD_PLAN.W, C_eff)` (powers of two, so R | C and W | C hold).
`num_rounds = ceil(num_tile_rows / C_eff)`, dealt by `split_work_to_cores`. Rationale: with fewer
rows than NEOs in the machine, one row per NEO on more clusters is the same NEO occupancy as packing
NEOs, with no padded rounds, more DM harts and less L1 per cluster; with more rows than clusters the
NEOs of each cluster are engaged before any NEO takes a second row.

**Resident-fit predicate** (host, exact): `fits(c, bh)` ⇔
`c · F(bh) ≤ l1_budget` **and** every ring `c · entries_per_neo · page_bytes ≤ DFB_RING_CAP_BYTES`,
where `F(bh)` is the per-NEO footprint of `l1_ledger.md`, `l1_budget =
ttnn.get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES` (`L1_RESERVE_BYTES = 32768`, host knob),
`DFB_RING_CAP_BYTES = 1_048_560` (target page). `bh_fit(c)` = the largest bh ≥ 1 with `fits(c, bh)`.
Page bytes come from `tensor.buffer_page_size()` / `ttnn.tile_size(fmt)`, never literals.

Worked Phase 0 values (Quasar, 32 clusters, budget 3,930,560 B):

| Shape | num_tile_rows | Wt | C_eff | clusters | rounds | bh | per-NEO F (fp32 acc / bf16 acc) |
|-------|---------------|----|-------|----------|--------|----|------------------------------|
| (1,1,128,1024) (model) | 4 | 32 | 1 | 4 | 1 | 1 | 356,352 / 282,624 B |
| (1,1,64,4096) (model) | 2 | 128 | 1 | 2 | 1 | 1 | 1,339,392 / 1,069,056 B |
| (1,1,2048,256) | 64 | 8 | 2 | 32 | 1 | 1 | 110,592 / 86,016 B |
| (4,8,32,256) | 32 | 8 | 1 | 32 | 1 | 1 | 110,592 / 86,016 B |
| (1,1,128,4096) | 4 | 128 | 1 | 4 | 1 | 1 | 1,339,392 / 1,069,056 B |
| model's untraced (1,1,12288,1024) | 384 | 32 | 4 | 32 | 3 | 1 (fp32 acc: ring cap on `cb_centred`, 4·bh·32·4096 ≤ 1,048,560 ⇒ bh = 1, 3 blocks per NEO) / 3 (bf16 acc: one block) | 356,352 per NEO at bh = 1 / 557,056 per NEO at bh = 3 |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_block` | `input_block_depth` (blocks in flight per NEO) | 1 block = bh·Wt pages per NEO | residency of the block across passes 1–2; because the centre pass pops it, the reader refills block k+1 while compute runs passes 3–6 of block k (overlap without a second block) |
| `cb_centred` | 1 block (compute-owned) | bh·Wt per NEO | residency across sq-sum and affine passes |
| `cb_output_tiles` | `output_depth` | 2 pages per NEO | double buffering packer ↔ writer |
| `cb_beta_stream` | `affine_stream_depth` | 2 pages per NEO | double buffering reader ↔ β expansion |
| `cb_row_partial`, `cb_row_stat`, `cb_rstd_rows` | 1 block | bh per NEO | full block between sequential helpers on one NEO |
| `cb_gamma`, `cb_beta_rows` | resident for the kernel | Wt per NEO | γ/β fetched/expanded once per NEO, reused by every block |
| `cb_mean_scaler`, `cb_var_scaler`, `cb_epsilon` | constant | 1 per NEO | — |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DFB ring extent (target page, 1,048,560 B) | `C · bh · Wt · page` for `cb_input_block`, `cb_centred`; `C · Wt · page` for `cb_gamma`, `cb_beta_rows` | inside `fits()`: lower bh, then C; refuse if `fits(1,1)` fails | ring creation failure on a stricter base |
| Indexed window must not straddle the ring end | resident windows of `cb_input_block`, `cb_centred`, `cb_gamma`, `cb_beta_rows` | ring capacity **exactly** one block (bh·Wt per NEO, never rounded up) and every non-last block consumes exactly that many entries, so each block's window starts at the ring base | — |
| STRIDED strict rotation: every thread of a kernel takes the same number of entries | rows per NEO | every NEO of a cluster runs `rounds = ceil(rows_this_cluster / C)`; a NEO without a real row in the last round recomputes row `round·C` (NEO 0's row of that round); the writer stores it again with the same bytes; whole rounds are dealt to clusters so only the last cluster can pad | hang (a reader hart waits on a NEO that never pops) |
| Reduce helper indexed access under C > 1: not used | reduce input window | the reduce helper only ever sees **one tile per row** with `WaitAndPopPerTile` (index 0) | — |
| Thread-plan limits | C ∈ {1,2,4}, R+W ≤ 6, R\|C, W\|C | `select_blocking()` clamps R, W to C | program build error |
| DFB count (32 / cluster, 24 / NEO) | number of rings | 12 rings | build error |
| `num_entries` uint16 | `C · bh · Wt` | ≤ 65,535, implied by the L1 predicate | — |
| DEST: one accumulator live per row | none on block extent — chains run block_size 1, the row accumulator lives in D0 only (`DestAccumulation` requires `Dst::D0`, `chain.inl:1273-1275`) | — | — |
| Compute kernel binary ≤ 24,576 B per TRISC | number of distinct chain instantiations | one call site per block operation; the ragged last block reuses the same call with a runtime extent | link failure |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (boundary: DRAM; minimum = x once, out once, γ once, β once) | What a bigger block buys |
|--------|--------|-----------|-------|-----------------------------------------------------|--------------------------|
| `row_resident` | **built** | `fits(1, 1)` — on Quasar: W ≤ 8,160 with fp32 accumulation (ring cap on `cb_centred`: Wt·4096 ≤ 1,048,560), W ≤ 15,264 (Wt ≤ 477, footprint-bound) with the 16-bit accumulator; covers every INPUTS/TARGET shape (W ≤ 4096) | `bh × Wt` tiles per NEO, `bh = min(rounds, bh_fit(C))` | x: **1** crossing (minimum); out: 1 (minimum, + ≤ C−1 duplicate row writes per op from the padded round); γ and β: **one crossing per engaged NEO** (above minimum by ×engaged NEOs — the reuse-shared operand per NEO, stepping stone for `affine_fanout`) | per block: 7 helper inits + 7 pack-target switches + 1 scaler/ε wait, the pipeline fill/drain of each pass; and per extra row in a block the reduce/rsqrt chains amortise one init over bh tiles (`compute_block_size` catalog entry: 1.27–1.65× from fewer passes on WH) |
| `row_streaming` (three reads of x per row, window rings) | **deferred** — positive reason: no shape of TARGET/INPUTS reaches it (W ≤ 4096 fits `row_resident` at C = 1 with ≥ 2× margin); it is the fallback for W beyond `fits(1,1)`, which `validate()` refuses with `NotImplementedError` until then. Reachable: the same block operations with streaming input policies, the reader re-issuing the row once per pass (the `toy_row_passes` pattern) | `not fits(1,1)` | `1 × Wt` streamed, window 2 | x: 3 crossings (sum, centre/sq-sum recomputed, affine) | — |
| `w_split_combine` (split Wt across clusters, per-slice partial statistics combined in one round with Chan's formula `M2 = Σ M2_i + Σ n_i (mean_i − mean)²`) | **deferred** — positive reason: `row_resident` is needed in any design (it is the cheaper split whenever num_tile_rows ≥ num_clusters·C, and it is the per-slice kernel of this regime); this regime is built **on top of** it — the per-row partial tiles in `cb_row_partial` are exactly the combine payload — and its cross-cluster combine needs multicast + semaphores in a Metal 2.0 ProgramSpec, a mechanism with no worked example on this part at this base (`quasar_op_examples/README.md`, *Not covered*: "multicast between clusters"). Selection when built: `num_tile_rows < num_clusters` (few-row wide shapes, incl. both model calls) | `num_tile_rows < num_clusters` (proposed) | `bh × (Wt / S)` per cluster, S slices | x, out: 1; γ/β: **1 crossing total** (each slice reads only its columns); added cross-core: per (row, slice) 2 tiles up (mean_i, M2_i) + 2 tiles down (mean, rstd) | parallelism over W for few-row shapes (model call (1,1,64,4096): 2 NEOs busy today, up to 64 with the split) |
| `w_split_in_cluster` (split Wt over the NEOs of one cluster, combine in L1) | **rejected** — superseded by `w_split_combine`: no ring lets a NEO read another NEO's slice (STRIDED slices are private; ALL with C > P is not used), so the combine has to route through a DM kernel exactly as the cross-cluster combine does; building it separately would be a dead end | — | — | — | — |
| `affine_fanout` (γ/β read once per cluster or once per op, fanned out) | **deferred** successor of the per-NEO γ/β reads — positive reason: intra-cluster fan-out needs an ALL ring with C consumers on one producer, not used at this base; cross-cluster fan-out is the multicast mechanism of `w_split_combine`; it is a reader-only change (the compute kernel and ring layout are unchanged), so the built structure keeps it reachable | — | — | γ/β crossings from ×engaged NEOs to ×1 | — |
| `single_pass_moments` (Σx and Σx² in one pass, var = E[x²] − E[x]²) | **rejected** — superseded by `row_resident`'s two-pass centred variance, which costs no extra DRAM traffic because the row is resident; the single-pass form cancels catastrophically in a 16-bit accumulator (spec GUIDANCE "The variance") | — | — | — | — |

### Traffic ranking

Counted per tensor at the DRAM tier (bytes = tiles × 2048 for bf16), then cross-core.

| Rank | Candidate split | DRAM crossings | Cross-core | Notes |
|------|-----------------|----------------|------------|-------|
| 1 (for num_tile_rows < clusters, e.g. both model calls) | `w_split_combine` | x 1, out 1, γ 1, β 1 | 4 tiles per (row, slice) | cheapest bytes; adds a rendezvous; deferred (above) |
| 1 (for num_tile_rows ≥ clusters·C) / 2 otherwise | `row_resident` (built) | x 1, out 1, γ ×E, β ×E (E = engaged NEOs, ≤ num_tile_rows) | 0 | e.g. (1,1,64,4096): x 512 KiB, out 512 KiB, γ+β 2 × 2 × 256 KiB = 1 MiB → 2 MiB vs 1.5 MiB minimum |
| 3 | `row_streaming` | x 3, out 1, γ ×E, β ×E | 0 | only beyond the resident fit |
| — | `single_pass_moments` | x 1 | 0 | rejected for precision, not bytes |

Chosen: `row_resident` — it is the cheapest-traffic split for every shape with at least as many rows
as clusters, and the only one with no combine. For few-row shapes `w_split_combine` saves
`(E − 1)·(|γ| + |β|)` DRAM bytes and adds the combine; it is the deferred row above, and the
structure built here (per-row partial tiles, per-NEO statistics) is its per-slice kernel.

**Stall-shadow.** (1) Compute waits for the first input block: the row-sum pass uses a **Cumulative**
wait, so it accumulates each tile as it lands (addition order equals arrival order, so this is legal
by associativity and changes no rounding order) — pass 1 runs inside the read. (2) The β expansion
is independent of x: the reader sends β before block 0 and compute expands β as its prologue, so
it runs under the first input read. (3) The reader waits on `cb_input_block` space: the centre pass
pops block k, so block k+1 streams in under passes 3–6 of block k; γ is read in that window too.
(4) Nothing independent remains during the two one-tile reduces (they depend on the pass before).

### Block schedule

```cpp
// per NEO; reader/compute/writer are separate kernels realising their parts
expand_beta();                                            // once per kernel (prologue)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);       // reader: bh x Wt input tiles
    row_sum_block(block_idx);    // per row: Σ_w (x_w + x_w) in DEST  -> cb_row_partial
    mean_block(block_idx);       // per row: reduce_row(partial) · 1/(2W) -> cb_row_stat
    centre_block(block_idx);     // c = x − mean (bcast col)            -> cb_centred (pops input)
    sq_sum_block(block_idx);     // per row: Σ_w c_w·c_w in DEST        -> cb_row_partial
    var_block(block_idx);        // per row: reduce_row(partial) · 1/W  -> cb_row_stat
    rstd_block(block_idx);       // rsqrt(bcast_col(var) + ε)           -> cb_rstd_rows
    affine_block(block_idx);     // (c · γ_row) · rstd + β               -> cb_output_tiles (pops centred)
    store_block(block_idx);      // writer
}
```

| Block operation | Block shape | Resident across it | Intended frequency of inits / barriers |
|-----------------|-------------|--------------------|----------------------------------------|
| `expand_beta` | Wt tiles | `cb_beta_rows` filled for the kernel | 1 init + 1 `pack_init` per kernel |
| `load_block` | bh·Wt tiles per NEO | — | one tagged NoC read per tile, no barrier (implicit sync); constants once per kernel |
| `row_sum_block` | grid(bh, Wt) | `cb_input_block` (not popped) | 1 chain init + 1 `pack_init` per block; one DEST acquire/pack per row |
| `mean_block` | bh tiles (one per row) | `cb_input_block` | 1 reduce init per block (helper does its own `pack_init`) |
| `centre_block` | grid(bh, Wt) | `cb_row_stat` (mean, one per row, popped per row) | 1 chain init + 1 `pack_init` per block |
| `sq_sum_block` | grid(bh, Wt) | `cb_centred` (not popped) | 1 chain init + 1 `pack_init` per block |
| `var_block` | bh tiles | `cb_centred` | 1 reduce init per block |
| `rstd_block` | bh tiles | `cb_centred`, `cb_epsilon` | 1 chain init + 1 `pack_init` per block |
| `affine_block` | grid(bh, Wt) | `cb_gamma`, `cb_beta_rows` (kernel-resident), `cb_rstd_rows` (one per row) | 1 chain init + 1 `pack_init` per block |
| `store_block` | bh·Wt tiles | — | one tagged NoC write per tile; trid reset once at kernel end |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Thread-plan C selection (spread rows over clusters before NEOs) | slow-dispatch launch cost per cluster and per-cluster γ/β load are fixed per engaged cluster | pack NEOs first (C = 4 always, fewer clusters) at the model shapes; measure model cycles on `ttsim-perf` |
| Thread plan (R, W) at C = 4 | (4,4,2) is the page's plan for ops that read more than they write; with γ/β resident the steady-state reads ≈ writes | (2,4,2) and (1,4,1) at C = 4 |
| Overlap (`input_block_depth` = 1 block) | with bh = rounds (one block per NEO) the whole read is exposed before passes 3–6 can overlap anything | `input_block_depth` = 2 blocks of bh/2 (L1 permitting) |
| Grid-synchronization-like fixed cost (full occupancy) | at small Wt (e.g. (4,8,32,256): one row of 8 tiles per cluster) the per-cluster γ/β load (16 tiles) exceeds the row itself | fewer clusters with several rows each (bh > 1) so γ/β is amortised |
| Row-sum doubling (`x_w + x_w`, Wt FPU adds per row) | the pair-add (`x[i] + x[i + Wt/2]`, Strided operands) halves the FPU steps and the bf16-DEST rounding steps (`eltwise_l1_vs_dest_accumulate`, `row_reduce_accumulate`: 2.9× and 1.4 vs 2.4 ULP on WH) but needs caller-managed waits (loses the Cumulative overlap) and parity handling | pair-add with a copy-seeded odd tile |
| Fused affine via two FPU dest-reuse ops | on WH dest-reuse lost to an L1 round trip (`compute_fusion`: 0.82× for the combine step) | separate chains with a bh·Wt intermediate (costs L1) |
| β (and rstd) expansion on compute | DM harts are idle; compute is the bound | reader-side expansion (replicate row 0 into 32 rows while filling an explicit-sync ring) |
| bf16 reduce scaler `1/(2W)`, `1/W` | exact for power-of-two W (all Phase 0 shapes) but rounded to 8 bits otherwise | scaler 1.0 and `mul_unary_tile(1/W)` (fp32, Quasar path exists) applied to the one-tile reduce result |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → `cb_input_block` | input dtype pages (bf16 2048 B) | reader, implicit sync: `noc.async_read<NocOptions::TXN_ID>(acc_x, cb_input_block, {.page_id = row·Wt + w}, {})` one per tile | reader hart p feeds NEOs p, p+R, … in strict rotation (k-th read lands at NEO p + (k mod C/R)·R) |
| DRAM → `cb_gamma` | weight dtype pages | reader, implicit sync, Wt tiles per NEO (page ids 0..Wt−1 of `weight`), once per kernel | read where it lives (R11) |
| DRAM → `cb_beta_stream` | bias dtype pages | reader, implicit sync, Wt tiles per NEO, once per kernel, **before** input block 0 | stall-shadow (2) |
| reader → `cb_mean_scaler`, `cb_var_scaler` | bf16 | explicit sync, written by the hart: `dataflow_kernel_lib::prepare_reduce_scaler<id, PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb, 1/(2W))` and `(dfb, 1/W)`, one entry per NEO served | listed in the reader's `disable_dfb_implicit_sync_for`; no NoC transfer touches them |
| reader → `cb_epsilon` | intermediate format (Float32 when `fp32_dest_acc_en`, else Float16_b) | explicit sync: reserve, fill all 1024 elements with ε (fp32 bits from an RT arg, or bf16-rounded), push; one per NEO served | same opt-out |
| compute rings | see Circular Buffers | packer → unpacker of the same NEO | `pack_init(<ring>)` before every helper whose output ring differs from the previous call's |
| `cb_output_tiles` → DRAM | output dtype | writer, implicit sync: `noc.async_write<NocOptions::TXN_ID>(cb_output_tiles, acc_out, {}, {.page_id = …})` | writer hart w drains NEOs w, w+W, …; padded rows rewritten with identical bytes |
| kernel end | — | reader: `noc_async_read_set_trid(0, noc.get_noc_id())`; writer: `noc_async_write_set_trid(0, noc.get_noc_id())` | hands the sticky tag back (target page ring protocols) |

TARGET axes seen through the scheme:

| TARGET value beyond Phase 0 | Class | Why |
|------|-------|-----|
| `memory_config = L1` | knob-turn | interleaved L1 pages through the same `TensorAccessor`; output in the input's buffer type (R12) |
| `sharding` (only `interleaved` in TARGET) | — | no physical shard in TARGET; a height shard would be the `tile_row` split pinned by the caller (knob-turn, consumed in place through a ring borrowed from the shard), a width shard would cut the dependent axis (scheme-change = `w_split_combine`) |
| `dtype = float32` (requires `fp32_dest_acc_en=True`, R1) | knob-turn | page formats only (input/output pages 4096 B, `fits()` re-evaluates) |
| `affine_dtype = float32` | knob-turn + precision check | γ/β page format; note the β expansion and the row-broadcast unpack go through srcB (tf32) on Quasar (`bcast.h` Quasar path) |
| `alignment = w_non_aligned` | knob-turn in structure, adds one mask | R6: padding enters neither sum nor count — the scaler stays 1/W (true count), the last w-tile's padded columns are masked out of `row_sum` (zero in the sum only if the padding is zero, so mask) and of `centre` (padded c must be 0 before `sq_sum`): one constant mask tile + a masked op on tile Wt−1 |
| `alignment = h_non_aligned` | knob-turn | `num_tile_rows` already per-image `ceil(H/32)`; padded rows are independent and their outputs are not read back |
| `rank ∈ {2, 3}` | knob-turn | leading dims fold into `num_tile_rows` |

Inter-Tensix contract reserved for `w_split_combine` (not built): per (row, slice) the slice owner
sends its `(mean_i, M2_i)` tiles (the `row_sum` and `sq_sum` partials after the one-tile reduce) to the
row's combining cluster over the NoC with a semaphore increment; the combiner computes the global
`mean` and `rstd` and multicasts the two tiles back to the S slice owners, which then run
`centre`/`affine` on their columns. Chan's formula makes it one round (no second round for the variance).

### Structural impossibilities

None to add: the specification's `INVALID = []` is right for this rectangle.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one tile-row (Wt tiles) dealt to a NEO; a NEO drives its rows in blocks of `block_tile_rows` |
| Grid | `device.compute_with_storage_grid_size()` (8 × 4 on Quasar); test-only override `max_clusters` |
| Per-core work | `num_rounds = ceil(num_tile_rows / C_eff)` split by `ttnn.split_work_to_cores(grid, num_rounds, row_wise=True)`; cluster gets `rounds_c` rounds = `rows_c = min(rounds_c · C_eff, num_tile_rows − row_start_c)` rows starting at `row_start_c` (RT args `num_rows`, `row_start`) |
| Remainder | `num_tile_rows = Π(leading dims) · ceil(H/32)` (per image, `ceil`), `Wt = ceil(W/32)`; only the last cluster's last round can be short; its missing NEOs recompute row `round·C` of that round; ragged last block per NEO = `rounds_c − (blocks − 1)·bh` |
| Bound | **compute** — per input tile the NEO runs 6 FPU tile ops (row-sum add, centre sub, sq-sum mul, affine mul + 2 dest-reuse) and 2 packs, plus per-row reduce/SFPU work, against 1 tile read and 1 tile written (γ/β read once per NEO) |
| Thread plan | `THREAD_PLAN = (4, 4, 2)` (R, C, W) — the page's plan for a compute-bound op with several inputs; effective `(R_eff, C_eff, W_eff)` per call from `select_blocking()` (clamped to occupancy and fit, see Blocking Model); passed as CT args `reader_threads`, `compute_threads`, `writer_threads`; `KernelSpec(num_threads=…)` on each kernel |
| Unit dealt to compute threads | a tile-row (larger than a ring entry): NEO t owns local rows t, t+C, …; every NEO runs the same `rounds_c`; a NEO without a real row in the last round recomputes NEO 0's row of that round; whole rounds of C rows are dealt to clusters so padding happens at most once per op (≤ C−1 rows) |
| Tests the plan needs | (a) **threads ran**: build the program with `max_clusters=1`, `THREAD_PLAN=(4,4,2)` forced (`create_program_artifacts(..., thread_plan=..., max_clusters=1)`), shape `(1,1,256,64)` (8 rows → 2 rounds, bh = 2) and `(1,1,192,64)` (6 rows → padded last round): a NEO that did not run leaves its rows unwritten, so a correct result over every row shows each NEO, each reader and each writer ran; assert `C_eff == 4` was selected. (b) **regime-pinned C values**: C_eff = 1 (`(1,1,128,1024)`), 2 (`(1,1,2048,256)`), 4 (`(1,1,4096,64)` on the full grid) each asserted from `select_blocking()` and checked for values; one case with `blocks_per_neo > 1` (force `max_block_tile_rows=1` with 2 rounds). (c) **session tests**: in one device session, the op after another program (e.g. `ttnn.add`) and the op before `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)`; both at C = 1 and at C = 4. |

## Circular Buffers

> **As built** (implementer + verifier): the implemented ring set is `l1_ledger.md`'s, 11 rings — `cb_gamma`/`cb_beta_stream` became one `cb_affine_stream` feeding compute-expanded `cb_gamma_rows`/`cb_beta_rows`, and rstd is a column-0 tile carried as the third FIFO use of `cb_row_partial` (no `cb_rstd_rows`). The table below is the planner's original inventory.

All rings are `DataflowBufferSpec(unique_id=<name>, entry_size=<page>, num_entries=<per NEO> · C_eff,
data_format_metadata=<dtype>)`, STRIDED on both sides (default). `acc_fmt` = `ttnn.float32` when
`fp32_dest_acc_en` else `ttnn.bfloat16`. Page sizes are queried, never literal.

| Semantic Name | Index | Page Size | Num Pages (per NEO × C_eff) | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|------------------------------|------------------|--------|----------|----------|------|----------|
| `cb_input_block` | 0 | input page | `bh·Wt` × C | spans tile_row (bh) and w_tile (Wt): held through `row_sum` and `mean`, popped in `centre` | input dtype | reader | compute | reader: implicit | per block |
| `cb_gamma` | 1 | weight page | `Wt` × C | spans w_tile; streams nothing; resident for the kernel | weight dtype | reader | compute | reader: implicit | kernel |
| `cb_beta_stream` | 2 | bias page | `2` × C | window; streams w_tile once per kernel | bias dtype | reader | compute | reader: implicit | prologue |
| `cb_beta_rows` | 3 | bias page | `Wt` × C | spans w_tile; resident (β row 0 broadcast to 32 rows) | bias dtype (exact copy of bf16 values) | compute | compute | — | kernel |
| `cb_centred` | 4 | acc page | `bh·Wt` × C | spans tile_row and w_tile: held through `sq_sum`, popped in `affine` | `acc_fmt` | compute | compute | — | per block |
| `cb_row_partial` | 5 | acc page | `bh` × C | spans tile_row (one tile per row); reused: 2Σx partial, then Σc² partial | `acc_fmt` | compute | compute | — | per block, two uses |
| `cb_row_stat` | 6 | acc page | `bh` × C | spans tile_row; reused: mean, then var (column 0 valid) | `acc_fmt` | compute | compute | — | per block, two uses |
| `cb_rstd_rows` | 7 | acc page | `bh` × C | spans tile_row; rstd broadcast across columns (full tile) | `acc_fmt` | compute | compute | — | per block |
| `cb_output_tiles` | 8 | output page | `2` × C | window; streams both axes | input dtype | compute | writer | writer: implicit | per block |
| `cb_mean_scaler` | 9 | bf16 tile | `1` × C | constant `1/(2W)` in the reduce row-0 layout | bfloat16 | reader | compute | reader: explicit (opted out) | kernel |
| `cb_var_scaler` | 10 | bf16 tile | `1` × C | constant `1/W` | bfloat16 | reader | compute | reader: explicit (opted out) | kernel |
| `cb_epsilon` | 11 | acc page | `1` × C | constant ε in every element | `acc_fmt` | reader | compute | reader: explicit (opted out) | kernel |

CB sync (push = wait, per NEO per block, Rb = rows in this block): `cb_input_block` reader pushes
Rb·Wt / compute waits Rb·Wt (Cumulative) and pops Rb·Wt (centre, per tile). `cb_row_partial`: row_sum
pushes Rb (PerOuter), mean reduce waits/pops Rb; sq_sum pushes Rb, var reduce waits/pops Rb.
`cb_row_stat`: mean reduce pushes Rb, centre waits/pops Rb (Col, PerTile per row); var reduce pushes
Rb, rstd waits/pops Rb. `cb_centred`: centre pushes Rb·Wt, sq_sum waits Rb·Wt (no pop), affine pops
Rb·Wt. `cb_rstd_rows`: rstd pushes Rb, affine waits/pops Rb. `cb_output_tiles`: affine pushes Rb·Wt,
writer drains Rb·Wt. Kernel-lifetime rings (`cb_gamma`, `cb_beta_rows`, scalers, ε) are pushed once
(Wt, Wt, 1, 1, 1) and popped once by compute after the last block — every ring must be empty at kernel
exit (TRISC drain waits for zero occupancy). `cb_beta_stream`: reader pushes Wt, `expand_beta` pops Wt.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `expand_beta` | tiles(Wt) | eltwise chain `UnaryBcast<Row>` → `PackTile` | `cb_beta_stream`, 1 at a time (PerTile/PerTile) | `cb_beta_rows`, Wt (PerTile) | `cb_beta_rows` holds Wt, kept to kernel end |
| 1 | `load_block` | Rb·Wt | reader | DRAM | `cb_input_block` | — |
| 2 | `row_sum_block` | grid(Rb, Wt) | eltwise chain `BinaryFpu<Add, x, x, D0, PerRow>` → `PackTile` (PerOuter, PerRow) | `cb_input_block` Rb·Wt (Cumulative, None, Block) | `cb_row_partial`, Rb | input block still resident |
| 3 | `mean_block` | of(Rb, 1) | `reduce<SUM, REDUCE_ROW>` | `cb_row_partial` Rb (WaitAndPopPerTile), `cb_mean_scaler` (waited, not popped) | `cb_row_stat`, Rb | — |
| 4 | `centre_block` | grid(Rb, Wt) | eltwise chain `BinaryFpu<Sub, x, mean(bcast Col)>` → `PackTile` | `cb_input_block` (PerTile/PerTile, Scalar), `cb_row_stat` (bcast Col, PerTile/PerTile, Col: one per row) | `cb_centred`, Rb·Wt | input block freed (reader refills); centred resident |
| 5 | `sq_sum_block` | grid(Rb, Wt) | eltwise chain `BinaryFpu<Mul, c, c, D0, PerRow>` → `PackTile` (PerOuter, PerRow) | `cb_centred` Rb·Wt (Upfront, None, Block) | `cb_row_partial`, Rb | centred still resident |
| 6 | `var_block` | of(Rb, 1) | `reduce<SUM, REDUCE_ROW>` | `cb_row_partial` Rb, `cb_var_scaler` | `cb_row_stat`, Rb | — |
| 7 | `rstd_block` | tiles(Rb) | eltwise chain `UnaryBcast<Col, var>` → `DestReuseBinary<Add, ε>` → `Rsqrt<>` → `PackTile` | `cb_row_stat` (PerTile/PerTile), `cb_epsilon` (Upfront, None, Scalar) | `cb_rstd_rows`, Rb | — |
| 8 | `affine_block` | grid(Rb, Wt) | eltwise chain `BinaryFpu<Mul, c, γ(bcast Row)>` → `DestReuseBinary<Mul, rstd_rows>` → `DestReuseBinary<Add, β_rows>` → `PackTile` | `cb_centred` (PerTile/PerTile, Scalar), `cb_gamma` (bcast Row, Upfront, None, Row), `cb_rstd_rows` (PerTile/PerTile, Col), `cb_beta_rows` (Upfront, None, Row) | `cb_output_tiles` (PerTile) | centred freed |
| 9 | `store_block` | Rb·Wt | writer | `cb_output_tiles` | DRAM | — |
| end | drain | — | raw `pop_front` | `cb_gamma` Wt, `cb_beta_rows` Wt, `cb_mean_scaler` 1, `cb_var_scaler` 1, `cb_epsilon` 1 | — | all rings empty |

`compute_kernel_hw_startup(cb_input_block, cb_gamma, cb_output_tiles)` is the compute kernel's first
statement (once). `pack_init(<output ring>)` precedes ops 0, 2, 4, 5, 7, 8 (the reduce helper issues its
own `pack_init`, `reduce_helpers_compute.inl:962-964`). Inits run once per block per operation (chain
`InitReconfigOwner::Chain`).

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all compute | helper | `compute_kernel_hw_startup` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | `(cb_input_block, cb_gamma, cb_output_tiles)` | — | — | — |
| ops 0,2,4,5,7,8 | helper | `eltwise_chain` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:620` | `IterationShape::grid(Rb, Wt)` / `tiles(n)` (`chain.hpp:146-173`) | — | — | `Rb` (≤ `block_tile_rows`), `Wt` |
| 0 | helper | `UnaryBcast<BroadcastDim::Row, input(cb_beta_stream)>` + `PackTile<output(cb_beta_rows)>` | `ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp:26`; Quasar path `tt_metal/hw/inc/api/compute/bcast.h:83` (B2D via srcB, row 0) | — | `cb_beta_stream` | `cb_beta_rows` | Wt |
| 2 | helper | `BinaryFpu<BinaryFpuOp::Add, input(cb_input_block, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block), input(cb_input_block, Cumulative, None, Block), Dst::D0, DestAccumulation::PerRow>` + `PackTile<output(cb_row_partial, ReservePolicy::PerOuter, PushPolicy::PerOuter, DataFormatReconfig::Enabled, TileAddressing::Direct, DestAccumulation::PerRow)>` | `chain.hpp:568` (BinaryFpu), `chain.hpp:308` (DestAccumulation), `chain.hpp:588` (PackTile); same-buffer dedup `chain.inl:1284-1291`; legality `chain.inl:53-63` (Cumulative+Block+None); PerRow pack rule `chain.inl:1141-1149` | sum 2·x per element column, exact doubling | `cb_input_block` | `cb_row_partial` | Rb, Wt |
| 3, 6 | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_row_partial, cb_{mean,var}_scaler, cb_row_stat, ReduceInputPolicy::WaitAndPopPerTile>(…, ReduceInputBlockShape::of(Rb, 1))` (ReduceFp32Mode::Fast, Algorithm Auto→ReduceTile) | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:654-681`; `ReduceInputBlockShape::of` `:305` | — | `cb_row_partial` | `cb_row_stat` | Rb |
| reader consts | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_id, PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb, value)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:70` | value `1/(2W)` and `1/W` (non-standard SUM scaler: the documented use, `:30-34`) | — | `cb_mean_scaler`, `cb_var_scaler` | — |
| 4 | helper | `BinaryFpu<Sub, input(cb_input_block, PerTile, PerTile, Scalar), input(cb_row_stat, BroadcastDim::Col, PerTile, PerTile, InputTileMapping::Col)>` + `PackTile<output(cb_centred)>` | `chain.hpp:568`, `chain.hpp:411-418` (srcB broadcast spec); Quasar bcast path `llk_math_binary_api.h:55-61`, `llk_unpack_AB_api.h:59-119` (strided index) ; Col+PerTile streams one tile per row `chain.hpp:221-226` | — | `cb_input_block`, `cb_row_stat` | `cb_centred` | Rb, Wt |
| 5 | helper | `BinaryFpu<Mul, input(cb_centred, Upfront, None, Block), input(cb_centred, Upfront, None, Block), D0, PerRow>` + `PackTile<output(cb_row_partial, PerOuter, PerOuter, …, PerRow)>` | as op 2 | Mul at `math_fidelity` (HiFi4) | `cb_centred` | `cb_row_partial` | Rb, Wt |
| 7 | helper | `UnaryBcast<Col, input(cb_row_stat)>` → `DestReuseBinary<Add, input(cb_epsilon, Upfront, None, Scalar), DestReuseType::DEST_TO_SRCA>` → `Rsqrt<>` → `PackTile<output(cb_rstd_rows)>` | `bcast.hpp:26`; `chain.hpp:580` (DestReuseBinary), impl `chain.inl:1484`; Quasar dest-reuse `tt_metal/hw/inc/api/compute/eltwise_binary.h:430-445`; `Rsqrt` `eltwise/unary/math.hpp:38`, `math.inl:57-65` (no Quasar guard) | — | `cb_row_stat`, `cb_epsilon` | `cb_rstd_rows` | Rb |
| 8 | helper | `BinaryFpu<Mul, input(cb_centred, PerTile, PerTile, Scalar), input(cb_gamma, BroadcastDim::Row, Upfront, None, InputTileMapping::Row)>` → `DestReuseBinary<Mul, input(cb_rstd_rows, PerTile, PerTile, Col), DEST_TO_SRCA>` → `DestReuseBinary<Add, input(cb_beta_rows, Upfront, None, Row), DEST_TO_SRCA>` → `PackTile<output(cb_output_tiles)>` | `chain.hpp:568`, `:580`, `:588`; Row mapping `chain.hpp:247` | — | `cb_centred`, `cb_gamma`, `cb_rstd_rows`, `cb_beta_rows` | `cb_output_tiles` | Rb, Wt |
| output-ring switches | raw_api | `pack_init(ring)` | `tt_metal/hw/inc/api/compute/pack.h:32` | before ops 0,2,4,5,7,8 | — | — | — |
| end drain | raw_api | `DataflowBuffer::pop_front(n)` | `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A | n = Wt, Wt, 1, 1, 1 | kernel-lifetime rings | — | — |
| reader/writer | raw_api | `noc.async_read<NocOptions::TXN_ID>` / `noc.async_write<NocOptions::TXN_ID>` (implicit sync), `reserve_back`/`push_back` (explicit rings), `noc_async_{read,write}_set_trid(0, noc.get_noc_id())` | `.claude/references/quasar_op_examples/toy_row_passes/kernels/reader.cpp`, `writer.cpp`; `toy_thread_plan/kernels/reader_mixed.cpp` | — | — | — | — |

Helpers considered and rejected (for every raw_api entry and every helper mode not taken):

| Candidate | Where it fails | Reason |
|-----------|----------------|--------|
| end-of-kernel `pop_front` (raw) — chain `PopPolicy::AtEnd` on the last block's call | `chain.hpp:236` (AtEnd pops after that call) | the same call site serves every block; popping kernel-lifetime rings inside it would need a second instantiation of the affine chain (binary-size cap); one raw pop after the loop is the block-level realisation |
| reduce over the whole row (`reduce<SUM, REDUCE_ROW>` on Rb·Wt tiles) for the mean/variance | `WaitAndPopPerTile` would pop the resident row; FPU reduce accumulates via tf32 across tiles | replaced by DEST accumulation (ops 2, 5) + one-tile reduce (ops 3, 6), `row_reduce_accumulate`'s faster path |
| `AddUnary` (chain, scalar add of ε) | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:34-58` (`add_unary_tile` under `#ifndef ARCH_QUASAR`) | ε added with `DestReuseBinary<Add>` from a constant ring instead |
| reduce `post_reduce_op` for `+ε, rsqrt` | needs an SFPU scalar add (above) | op 7 is a separate one-tile chain |
| broadcast in `DestReuseBinary` (γ, β, rstd all as broadcasts) | `llk_math_binary_api.h:56-59` static_assert | one broadcast per chain (γ as srcB Row); β and rstd are materialised as full tiles (ops 0, 7) |
| ALL ring for γ/β/constants | — | per-NEO STRIDED copies |
| `matmul_block`, `bias_add_helpers` | not applicable (no matmul) | — |
| `tilize` / `untilize` | input and output are TILE | — |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| row_sum | Add | `cb_input_block` All | `cb_input_block` All | None |
| centre | Sub | `cb_input_block` All | `cb_row_stat` (mean, REDUCE_ROW out) Col0 | Col |
| sq_sum | Mul | `cb_centred` All | `cb_centred` All | None |
| expand_beta | UnaryBcast | `cb_beta_stream` Row0 (β = row 0, R8) | — | Row |
| rstd | UnaryBcast | `cb_row_stat` (var, REDUCE_ROW out) Col0 | — | Col |
| rstd | DestReuse Add | DEST (var broadcast) All | `cb_epsilon` All | None |
| affine | Mul | `cb_centred` All | `cb_gamma` Row0 (γ = row 0, R8) | Row |
| affine | DestReuse Mul | DEST All | `cb_rstd_rows` All | None |
| affine | DestReuse Add | DEST All | `cb_beta_rows` All | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Pack target | 6 eltwise chains with 6 different output rings in one kernel | `pack_init(<ring>)` before each |
| Resident window straddling the ring end | indexed (Block/Row) access to `cb_input_block`, `cb_centred`, `cb_gamma`, `cb_beta_rows` | capacities exactly one block; every non-last block consumes exactly the capacity |
| Strict rotation hang | a NEO with fewer rows than its siblings | every NEO runs `rounds_c` rows, padded rows recompute row `round·C`; reader/writer use the same `row_of(neo, round)` as compute's count |
| Using weight rows other than row 0 (R8) | the suite's weights hold 32 different rows | γ is only consumed through a Row broadcast (srcB row 0); β only through `UnaryBcast<Row>` |
| Doubled row sum | `row_sum` produces 2·Σx | mean scaler is `1/(2W)`, variance scaler `1/W` — two scaler rings, never swapped |
| 16-bit page under fp32 DEST | `cb_centred`, `cb_row_partial`, `cb_row_stat`, `cb_rstd_rows`, `cb_epsilon` must follow `acc_fmt` | host derives every intermediate format from `fp32_dest_acc_en` in one place |
| Kernel-exit drain | kernel-lifetime rings still hold tiles | compute pops them after the last block (table above) |
| Sticky NoC transaction tag | the op followed by `ttnn.to_layout` (untagged untilize) | trid reset as the last statement of reader and writer; session tests required |
| Reduce helper under C > 1 | indexed reduce: not used | reduce only ever sees one tile per row with `WaitAndPopPerTile` |
| Fidelity | set on every compute kernel | `ComputeHardwareConfig(fpu_math_fidelity=config.math_fidelity, …)` always set |
| Explicit-sync rings in the reader | constants are written by the hart | `cb_mean_scaler`, `cb_var_scaler`, `cb_epsilon` listed in the reader's `disable_dfb_implicit_sync_for`; no NoC transfer of the reader is untagged |
| 16-bit accumulator at W = 4096 | Σ over 128 tiles rounds per add in a bf16 DEST (spec DECISIONS) | two-pass centred variance; graded at PCC 0.999; pair-add lamp halves the steps |
| DestReuse/UnaryBcast through srcB (tf32) on Quasar | var/rstd and β pass through a 19-bit source register | acceptable for bf16 output; flagged for the float32 refinements |
| Compute binary size (24 KiB per TRISC) | 6 chain instantiations + 2 reduces in one kernel | one call site per block operation; ragged last block uses the same call with a runtime extent |

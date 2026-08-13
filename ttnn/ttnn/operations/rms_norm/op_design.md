# Operation Design: rms_norm

> **Provenance.** No prior generation of this op exists in this tree: `ttnn/ttnn/operations/rms_norm/`
> did not exist before this document was written, `ttnn/cpp/ttnn/operations/normalization/` has been
> nuked of the non-distributed `rmsnorm`/`layernorm` ops, and no `eval/investigations/` or parked
> archive for this op was read. The design below is derived from the requirements, `feature_spec.py`,
> the kernel-lib headers, and the measured pattern catalog (`ttnn/ttnn/operations/examples/master.md`)
> only. Other ops (`toy_*`, the `examples/` catalog, `rmsnorm_distributed`, `dit_fused_distributed_rmsnorm`)
> were consulted for *idiom and proven mechanism*, never for this op's design.

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (row-wise normalization with a cross-core reduction combine) |
| Goal | Normalize each row of the input by its root-mean-square over the last dimension, optionally scaled by a per-column `gamma`. Fills the grid in both the many-rows and the wide-hidden regimes. |
| Math | `output[..., r, c] = input[..., r, c] * rsqrt( (1/W) * Σ_{c'=0}^{W-1} input[..., r, c']² + epsilon ) * gamma[c]` |
| Mode | Derivative (native `ProgramDescriptor` op, Python-side entry point) |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/ttnn-cb-memory-fundamentals.md`, `.claude/references/precision_convention.md`, `.claude/references/generic_op_template/`, `tech_reports/tensor_accessor/tensor_accessor.md`, `tech_reports/tensor_layouts/tensor_layouts.md`, `ttnn/ttnn/operations/examples/master.md`, `eval/golden_tests/rms_norm/feature_spec.py` |

### Measured catalog entries that set the knob defaults

| Knob | Catalog entry | Measured finding used |
|------|---------------|-----------------------|
| Σx² algorithm = DEST accumulate, not one wide `reduce` | `examples/row_reduce_accumulate/report.md:51-56` | `dest_accum_pairs` beats `reduce_fold` from **W ≥ 4 tiles**, 2.94× at 32 tiles (bf16-in/fp32-acc). Below 2 tiles `reduce_fold` wins. |
| Stat CBs in fp32, never bf16 | `examples/row_reduce_accumulate/README.md:102-109` | bf16 *accumulation* error grows with width (13 ULP @32t for `reduce_fold`); fp32 accumulation is ~exact. bf16 *input* is nearly free. |
| `hidden_tiles_per_core_floor = 4` | `feature_spec.py:343-346` measured-fastest sharded geometries (`[32,128]`→8 cores, `[32,256]`→9, `[32,160]`→32, `[32,256]`→28) | The winning width-shard geometries all land at **4–8 hidden tiles per core**, not 1. |
| `out_cb_depth = 2`, block ≥ 4 tiles per NoC barrier | `examples/double_buffer/report.md:30-42` | block 4–8 tiles × depth 2 = 17.9 GB/s vs 6.5 GB/s at block 1 / depth 1 (2.78×). Bigger just wastes L1; block 256 OOMs. |
| Combine topology = gather-to-root + mcast | `examples/tensix_all_reduce/report.md:15-38, :85-99` | For a 1-D group tree collapses to reduce-root; at **1 tile/core (our payload)** tree/root is the latency floor winner (1271–1377 ns vs reduce-scatter 1981), and root is contention-insensitive. |
| One block per core (coarsest) | `examples/compute_block_size/report.md:10-13` + `report_reconfig_ablation.md:8-15` | 1 block beats 8 blocks by 1.64×; combined with reconfig hoisting 1.72×. ≈1.6 µs fixed cost per extra pass. |
| Reads on NoC0 / writes on NoC1, `row_wise=True` core order | `examples/noc_placement/report.md:20-37` | A column line of cores is 2.91× slower than a row line when bandwidth-bound. |
| rsqrt on a column-0-valid tile is 8× over-computed | `examples/sfpu_tile_scope/report.md:12-21` | `rc` 748 ns vs `c_skip` 195 ns (1.94× over `c`); **not reachable through the helper API** → perf lamp, not a Phase 0 mechanism. |
| Gamma broadcast is a lamp, not Phase 0 | `examples/mcast_topology/report.md:8-10`, `examples/shared_input_reuse/report.md:13-16` | Device win (1.71–1.91×) is far below the DRAM-read-count reduction (8–11×) because the injector reads serially — so a 3–11 % traffic term is not worth a second mcast family yet. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2; `dtype ∈ {bfloat16, float32}`; `layout ∈ {TILE, ROW_MAJOR}`; `INTERLEAVED` | — | — |
| `gamma` | `Optional[ttnn.Tensor]` | no (kw-only) | shape `(1,1,1,W)` with `W == input.shape[-1]`; `layout ∈ {TILE, ROW_MAJOR}`; `dtype ∈ {bfloat16, float32}` | `None` | — |
| `epsilon` | `float` | no (kw-only) | any finite `> 0` | `1e-6` | RT (packed f32 bits, reader→compute is not needed: compute RT arg) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no (kw-only) | `fp32_dest_acc_en == True` in Phase 0; `math_fidelity` / `math_approx_mode` ungated (any value accepted and honored) | `default_compute_kernel_config()` | passed as `config=compute_kernel_config` |
| `num_row_groups` | derived host knob | — | divisor of `grid_y`, `≤ max(1, tensor_row_tiles)` | selection function (§Work Distribution) | CT (per-kernel) |
| `num_hidden_slices` | derived host knob | — | `1 … min(tensor_hidden_tiles, grid_x * grid_y / num_row_groups)` | selection function | CT |
| `block_rows` | block extent knob | — | `1 … core_row_tiles`, clamped by L1 fit | coarsest that fits (default = `core_row_tiles`) | CT |
| `block_hidden_tiles` | block extent knob | — | `= slice_hidden_tiles` (whole slice) | `slice_hidden_tiles` | CT |
| `hidden_tiles_per_core_floor` | tuning knob | — | `≥ 1` | `4` | host constant |
| `out_cb_depth` | buffer-depth knob | — | `≥ 1` | `2` | host constant |
| `l1_working_budget` | host constant | — | `device.l1_size_per_core() - L1_RESERVE` | — | host |

`default_compute_kernel_config()` is exported from `ttnn/ttnn/operations/rms_norm/rms_norm.py` and is the
*only* definition of the Phase 0 default (`math_fidelity=HiFi4`, `fp32_dest_acc_en=True`,
`math_approx_mode=False`) — see `.claude/references/precision_convention.md:61-85`. `None` resolves
through it; the golden axis tagger reads the same factory.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2, `(..., H, W)`; `H`, `W` need **not** be multiples of 32 |
| Dtype | `bfloat16` or `float32` (Phase 0); `bfloat8_b` is a later refinement |
| Layout | `TILE_LAYOUT` or `ROW_MAJOR_LAYOUT` — both native, no host-side transform |
| Memory | `INTERLEAVED` (Phase 0). `HEIGHT_/WIDTH_/BLOCK_SHARDED` are placement refinements over the *same* logical scheme (§Dataflow Strategy) |

`gamma`: shape `(1,1,1,W)`, `TILE` or `ROW_MAJOR`, `bfloat16` or `float32`, `INTERLEAVED`. Absent ⇒ the
golden `gamma_dtype` / `gamma_layout` axes canonicalize to the string sentinel `"none"`, which
`SUPPORTED` must accept and `validate()` must never refuse.

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | identical to input |
| Layout | identical to input |
| Memory | `INTERLEAVED`, same memory config as input |

---

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Everything below this section is its realization.

### Axes

The op's data has exactly two physical axes after the honest flattening below, plus one operand axis
and two intermediate axes. **Flattening claim:** every dimension except the last is treated
identically by the math (each row is normalized in isolation) and rows are contiguous in both
layouts, so all leading dims fold into one **row axis**. The reduce runs along the last dim only, so
folding rows across image boundaries is legal — an important consequence for `ROW_MAJOR` tile-row
counting below.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| **row** — flattened leading dims × H, in tile-rows (`tensor_row_tiles`) | **independent** — each row's RMS depends only on that row, so blocks along it compute in isolation | `block_rows` | `core_row_tiles` (the *whole* per-core assignment ⇒ `num_blocks_this_core == 1`), clamped down only by the L1 fit predicate | one host CT arg `block_rows`, derived from the L1 solve; every CB page count and loop bound derives from it | spread across the grid as `num_row_groups` **rectangles** of cores (rect height `grid_y / num_row_groups`); each rect owns a contiguous range of tile-rows via `ttnn.split_work_to_cores` | knob-turn (raise `num_row_groups`, change per-core share) |
| **hidden** — the reduced last dim, in tiles (`tensor_hidden_tiles`) | **dependent** — the RMS denominator is a sum spanning the whole axis, so a cross-block result requires a combine | `block_hidden_tiles` | `slice_hidden_tiles = ceil_div(tensor_hidden_tiles, num_hidden_slices)` — the **whole** slice, one block, never sub-chunked | one host CT arg `block_hidden_tiles`; `slice_hidden_tiles` is computed once on host from `num_hidden_slices` | split across the `num_hidden_slices` cores **inside** each row-group rect (row-major within the rect); partials combined over the rect. **Built in Phase 0** | knob-turn on `num_hidden_slices`; scheme-change to sub-chunking the slice (`TwoPassStreaming`, lamped) |
| **gamma hidden** — gamma's only axis, `tensor_hidden_tiles` tiles | **reuse-shared** — gamma does not vary along the row axis, so every row-group re-reads the same bytes (surfaced by the operand-reuse check against the chosen split, not by the op's math) | `block_hidden_tiles` (same knob — gamma is sliced exactly like the hidden axis) | `slice_hidden_tiles`, resident for the whole kernel | same CT arg as the hidden axis | **not split across cores independently**: a core's gamma slice is exactly its hidden slice, loaded once at boot. Cores in *different* row-groups holding the same hidden slice each read it from DRAM | scheme-change: mcast gamma along the row-group direction (**lamped**, quantified below) |
| **stat row** (intermediate: `cb_sq_partials`, `cb_slice_stat`, `cb_rms_recip`) — one tile per tile-row | **independent** — inherited from the row axis; a stat tile belongs to exactly one row | `block_rows` (same knob) | `core_row_tiles` | same CT arg `block_rows` | same assignment as the row axis — one stat tile per owned tile-row, on the core that owns it. Fully spread; nothing is centralized here | knob-turn |
| **contributor** (intermediate: `cb_gathered_partials`) — the `num_hidden_slices` partial-Σx² tiles per tile-row | **dependent** — this axis *is* the combine; it exists only to be summed away | `num_hidden_slices` (not an independent knob — fixed by the core-assignment of the hidden axis) | `num_hidden_slices` | derived from `num_hidden_slices`; no second literal | **not spread across cores**: gathered onto the row-group's root core, which sums `block_rows × (num_hidden_slices − 1)` tiles and then mcasts. This is a **communication role that also carries a small stage of computation**, and the rest of the rect waits for it. The stage is deliberately left on the root because its payload is `block_rows` tiles (1–8) — at 1 tile/core the measured latency floor favours root/tree over reduce-scatter (`examples/tensix_all_reduce/report.md:85-99`). The root is *not* an idle coordinator: it owns a hidden slice and computes its own partial **before** it begins waiting | scheme-change: reduce-scatter the contributor axis so all `num_hidden_slices` cores share the sum (**lamped**) |

**No blank cells.** Every axis named above carries a decision and its reason. Two axes are
deliberately *not* split across cores (`gamma hidden`, `contributor`) and both rows say why.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `in_cb_depth` | **1** (one block resident) | Nothing at Phase 0: the default block *is* the whole per-core assignment, so `num_blocks_this_core == 1` and there is no next block to prefetch. Depth 2 becomes meaningful only when the L1 solve forces `num_blocks_this_core > 1` — see the overlap perf lamp. |
| `cb_output_tiles` | `out_cb_depth` | **2** (two tile-rows in flight) | Overlaps compute's per-row pack against the writer's NoC drain. `examples/double_buffer/report.md:30-42`: depth 2 at block 4–8 tiles is 2.78× over depth 1 / block 1. |
| `cb_rm_stage_in` (ROW_MAJOR only) | `rm_in_depth` | **2** tile-rows of sticks | Overlaps the reader's 32 stick reads for tile-row `i+1` against `tilize` of tile-row `i`. |
| `cb_rm_stage_out` (ROW_MAJOR only) | `rm_out_depth` | **2** tile-rows of sticks | Overlaps `untilize` against the writer's stick writes. |
| `cb_gamma_tiles` | — | 1 (`slice_hidden_tiles` pages, no depth) | Resident constant; depth would buy nothing. |
| `cb_scaler`, `cb_w_mask` | — | 1 page each | Constants pushed once per kernel and never popped (`reduce_helpers_compute.inl:955` waits, never pops). |
| `cb_gathered_partials` | — | 1 (`num_hidden_slices × block_rows` pages) | The combine is one round per block; a second round is never in flight. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `Mcast2D` needs a **rectangle** of cores (`host/mcast_host.hpp:448-516` takes the bounding box as THE rect) | the core-assignment of the hidden axis | Row-groups are `num_row_groups` equal-height horizontal rectangles ⇒ `num_row_groups` must **divide `grid_y`**, and `num_hidden_slices ≤ grid_x × (grid_y / num_row_groups)`. Host clamps `num_row_groups` to a divisor of `grid_y`. | A non-rectangular group makes the mcast bounding box cover cores that are not in the group: they receive a stat tile for rows they do not own, silently corrupting their normalize pass. |
| `prepare_reduce_mask` fills only `valid_elems < tile_dim` positions (`reduce_helpers_dataflow.inl:339-345` asserts `0 < partial_positions < full_dim`) | the W-mask, and therefore `block_hidden_tiles` on the core owning the **global last** hidden tile | Mask CB is created and the mask phase enabled **only** when `W % 32 != 0`; `valid_w_in_last_tile = W - 32*(tensor_hidden_tiles - 1) ∈ [1,31]`. | Passing `valid_elems == 32` trips the assert; skipping the mask when `W % 32 != 0` folds tile padding into Σx² — a *uniform scale error* of `sqrt(W_padded/W) − 1`, which PCC is largely blind to (this is exactly the class `feature_spec.py`'s `pad_poison` group exists to catch). |
| Scaler / mask CBs must be `Float16_b` or `Float32` (`reduce_helpers_dataflow.inl:185-187` `static_assert`) | `cb_scaler`, `cb_w_mask` format | Both declared `bfloat16` regardless of input dtype. | Compile failure (loud, not silent). |
| `DEST_AUTO_LIMIT` = **4** tiles under `fp32_dest_acc_en=True` + half-sync (`dest_helpers.hpp:22-26`, `:103`) | the number of accumulators that must stay live at once | `sum_of_squares` uses `DestAccumulation::PerRow` in **D0 only** (1 live accumulator, `convenience.inl:38-46`). The `eltwise_chain` runtime clamp on `block_size` (`chain.hpp:445-449`) and `untilize`'s internal sub-blocking (`untilize_helpers.inl:80-86`) handle the rest. **No block extent is bounded by DEST** — subblocking is the helpers' own concern. | An over-wide DEST window silently aliases lanes. Both helpers clamp, so this is a stated non-issue rather than an assumed one. |
| `tilize<block_width_tiles, …>` / `untilize<block_width_tiles, …>` take the block width as a **template** param | `block_hidden_tiles` | Must be a compile-time arg ⇒ `slice_hidden_tiles` is a CT arg, and a ragged last hidden slice needs its own CT specialization or a padded-to-uniform slice. Host makes hidden slices **uniform** (`slice_hidden_tiles` identical on every core; the last core's tail tiles are simply outside `tensor_hidden_tiles` and are not read/written). | A runtime value will not compile; a per-core-varying value forces N kernel variants. |
| Tile page size is `32×32×elem_size`; a `ROW_MAJOR` stick is `W×elem_size` | `cb_rm_stage_in` stick pitch | Staging stick pitch is `align_up(slice_hidden_tiles*32*elem_size, l1_alignment)`, **not** `W*elem_size`, so `tilize` sees a contiguous `32 × (slice_hidden_tiles*32)` element region; the reader zero-fills the `slice_hidden_tiles*32 − valid_w_this_core` tail of every stick. | `tilize` reads the next stick's leading elements as this stick's trailing columns — a shear that produces wrong values with no error. |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **RowParallel** (`num_hidden_slices == 1`) | **Phase 0** | selection function yields `num_hidden_slices == 1` — i.e. `tensor_row_tiles ≥ grid_y` **and** `grid_x == 1`-equivalent partition, or `tensor_hidden_tiles < 2 * hidden_tiles_per_core_floor` | `(block_rows, block_hidden_tiles) = (core_row_tiles, tensor_hidden_tiles)` | Minimum at the DRAM boundary = *input once + output once + gamma once* = `(2·Rt·Wt + Wt)·T`. This regime: input **1×**, output **1×**, gamma **`num_row_groups`×** ⇒ above minimum by `(num_row_groups − 1)·Wt·T`. Zero cross-core traffic. Input crosses once because x stays L1-resident across the Σx² pass and the normalize pass. | Fewer `reduce`/`tilize`/`untilize` inits and format reconfigs (≈110–150 ns per reconfig, `compute_block_size/report_reconfig_ablation.md:8-15`), fewer `cb_output_tiles` reserve/push handshakes, one pipeline fill/drain instead of `num_blocks_this_core`. **Intended frequency: 1 init set + 1 fill/drain per core** (because Phase 0's default block is the whole assignment). |
| **BlockParallel + cross-core combine** (`num_hidden_slices > 1`) | **Phase 0** | selection function yields `num_hidden_slices > 1` — driven by `tensor_row_tiles < grid_y` (grid under-filled by rows) **or** by `min_slices_for_L1 > 1` (a whole `tensor_hidden_tiles` does not fit the working budget) | `(block_rows, block_hidden_tiles) = (core_row_tiles, ceil_div(tensor_hidden_tiles, num_hidden_slices))` | input **1×**, output **1×**, gamma **`num_row_groups`×** ⇒ same DRAM excess `(num_row_groups − 1)·Wt·T`, and `num_row_groups` is *smaller* here, so this regime is **closer** to the DRAM minimum than RowParallel. **Adds** cross-core: gather `Rt·(num_hidden_slices − 1)` tiles + mcast `Rt` tiles with fan-out `num_hidden_slices − 1`. The DRAM minimum is reachable only at `num_row_groups == 1`. | Everything RowParallel buys, **plus** fewer combine round-trips: the gather + mcast + semaphore handshake is paid once per block, so `num_blocks_this_core == 1` pays exactly one round-trip per core. This is the dominant regime-added fixed cost. **Intended frequency: 1 gather + 1 mcast + 1 handshake per core per block ⇒ 1 per core at the Phase 0 default.** |
| **TwoPassStreaming** (hidden slice sub-chunked; x re-read for the normalize pass) | **lamped** | `!fits_L1(block_rows=1, slice_hidden_tiles=ceil_div(Wt, min(Wt, grid_x·grid_y)))` | `(1, chunk_hidden_tiles)` with `Accumulate::at(cb_acc, chunk)` across chunks (`reduce_helpers_compute.hpp:367-374`) | input **2×** (re-read for normalize), output 1×, gamma `num_row_groups`× ⇒ above the minimum by a whole `Rt·Wt·T`. Structurally cannot reach the minimum: nothing can be resident. | Amortizes the second read's fill/drain and the per-chunk `Accumulate` reload. |
| **SingleCore** | **rejected** | — | — | — | Rejected outright: the grid is a runtime parameter and both Phase 0 regimes degenerate correctly at `num_row_groups = num_hidden_slices = 1`, so a single-core variant would be a strictly-dominated duplicate. |
| **GammaBroadcast** (mcast gamma across row-groups) | **lamped** | `num_row_groups > 1 && gamma is not None` | unchanged | Removes the entire `(num_row_groups − 1)·Wt·T` DRAM excess ⇒ reaches the stated minimum. Adds a second `Mcast1D(PerColumn)` family on disjoint semaphore ids. | unchanged |

**Why TwoPassStreaming is lamped for a positive reason, not because it is hard.** With
`l1_working_budget ≈ 1.2 MB` and `float32` tiles (4 KB), the `block_rows = 1` working set is
`≈ (1 + 1 + out_cb_depth)·slice_hidden_tiles + 5` tiles, so `slice_hidden_tiles ≤ ~73`. With
`min(Wt, grid_x·grid_y)` slices on a 110-core Blackhole part that admits `tensor_hidden_tiles ≤ 8030`,
i.e. `W ≤ 256 960`. The widest shape anywhere in `feature_spec.py` is `W = 32768`
(`LOOSE_CASES`, `feature_spec.py:225`) — a factor of ~8 below the bound. The regime is **unreachable
for this op's shape universe**, which is the positive reason. It is costed above so a future
refinement can move into it without rediscovering it.

**Why GammaBroadcast is lamped rather than built.** Its saving is quantified per regime below; it
peaks at ~11 % of DRAM bytes in the row-heavy prefill regime and is *zero* in the wide-decode regime.
The measured multicast entries (`mcast_topology/report.md:8-10`,
`shared_input_reuse/report.md:13-16`) both show the device-time win landing at 1.71–1.91× against an
8–11× read-count reduction, because the injector reads its slice serially — so a second mcast family
buys well under its read-count arithmetic. Phase 0's structure keeps it reachable: the gamma load is
already a *separate, once-per-kernel* named block operation reading a per-core hidden slice, so
swapping its DRAM read for an `Mcast1D(PerColumn)` receive changes one operation and no loop nest.

**Regime-selection function** (exact, host-checkable — restated concretely in Work Distribution) and
**regime-pinned tests** are mandatory: the `num_hidden_slices > 1` regime only triggers when rows
under-fill the grid, which is grid-size dependent and would otherwise pass on one device and fail on
another. `feature_spec.py`'s `_WIDE` `LOOSE_CASES` (`feature_spec.py:222-228`) pin it.

### Traffic ranking

Qualitative, on paper. `Rt = tensor_row_tiles`, `Wt = tensor_hidden_tiles`, `T = tile bytes`,
`G = grid_x · grid_y`, `s = num_hidden_slices`, `g = num_row_groups`. Named memory boundary = **DRAM**.

| Rank | Candidate split | DRAM bytes | Cross-core bytes / fan-out | Cores engaged | Verdict |
|------|-----------------|------------|----------------------------|---------------|---------|
| 1 | **hidden across cores, rows in one group** (`g=1`, `s=min(Wt,G)`) — the dependent-axis split with a combine | `2·Rt·Wt·T + Wt·T` = **the minimum**: gamma is *partitioned*, so it crosses DRAM exactly once in total | gather `Rt·(s−1)` tiles; mcast `Rt` tiles, fan-out `s−1` | `min(Wt, G)` | **Cheapest traffic.** Chosen whenever rows under-fill the grid. |
| 2 | **2D: `g` row-groups × `s` hidden slices** | `2·Rt·Wt·T + g·Wt·T` | `g` independent combines: gather `Rt·(s−1)`, mcast `Rt` fan-out `s−1` | `g·s ≤ G` | **Chosen as the general form.** Rank 1 and rank 3 are its `g=1` and `s=1` corners. |
| 3 | **rows across cores only** (`s=1`, `g=min(Rt,G)`) — the independent-axis split, no combine | `2·Rt·Wt·T + min(Rt,G)·Wt·T` — gamma re-read on **every** participating core | 0 | `min(Rt, G)` | Chosen when rows over-fill the grid (safer: no combine latency, no synchronization). **Not the cheapest** — the gamma excess is the lamp line below. |
| 4 | **rows across cores, hidden resident, x re-read** (two-pass, no residency) | `3·Rt·Wt·T + min(Rt,G)·Wt·T` — input crosses **twice** | 0 | `min(Rt, G)` | **Rejected as primary.** Strictly dominated by 1–3 whenever residency is achievable, which §Regimes proves it always is for this op's shapes. Retained as the `TwoPassStreaming` lamp. |

Two structural facts this ranking turns on, both of which the naive reading misses:

1. **Splitting the *dependent* axis is the cheaper-traffic option here, not the more expensive one.**
   Gamma is invariant along the row axis, so cutting rows across cores replicates gamma's DRAM read
   `g` times, whereas cutting the hidden axis *partitions* gamma. The combine's NoC bytes
   (`≈ Rt·s·T`) are paid on the moderate tier; the gamma replication (`(g−1)·Wt·T`) is paid on the
   most expensive one.
2. **The dependent-axis split is also a residency mechanism.** Cutting the hidden axis shrinks each
   core's reduced extent, which is what lets `cb_input_tiles` hold x across *both* passes and keeps
   input at **one** DRAM crossing. Without it, wide-W shapes fall into rank 4 and pay a whole extra
   `Rt·Wt·T`. This is why the hidden split is on the table even at `Rt ≫ G`.

Worked deltas for the two decisive `feature_spec.py` perf shapes (bf16, `T = 2 KB`):

| Shape | Regime chosen | DRAM bytes | Cheapest alternative | Delta |
|-------|---------------|------------|----------------------|-------|
| decode `(1,1,32,7168)` — `Rt=1`, `Wt=224` | rank 1: `g=1`, `s=56` | `2·224·2K + 224·2K` = 1.34 MB | already rank 1 | **0 — the implemented split *is* the cheapest.** |
| prefill `(1,1,8192,1024)` — `Rt=256`, `Wt=32` | rank 2: `g=8`, `s=8` (64 cores) | `2·256·32·2K + 8·32·2K` = 32.0 MB + 0.5 MB | rank 1 (`g=1`, `s=64`): 32.0 MB + 0.06 MB, but `Rt=256` rows then serialize on 64 cores in one group with a 64-way combine per tile-row (`Rt·s = 16384` gather tiles) | −0.44 MB DRAM for +32 MB NoC ⇒ rank 2 is chosen; the combine traffic swamps the gamma saving. |
| prefill `(1,1,8192,1024)` at `s=1` (the `g=min(Rt,G)=64` corner) | — | 32.0 MB + **4.0 MB** gamma | rank 2 above | **+3.5 MB DRAM (≈11 %)** — this is why the selection function prefers a 2D partition (`g` = divisors of `grid_y`, `s = grid_x·grid_y/g`) over pure row-parallelism even when rows over-fill the grid. |

**Occupancy is deliberately absent from this ranking.** "Rows under-fill the grid" is what puts the
dependent-axis split *on* the list; how many cores ultimately participate is settled by the extent
ranking and the grid-synchronization perf lamp, per `blocking-model.md` §4.

### Block schedule

Logical schedule — reader, compute and writer are separate asynchronous kernels; adjacent blocks may
pipeline. Names are the contract; realization (direct helper call / thin wrapper / custom block
helper / raw LLK) is the implementer's choice.

```cpp
// ---- once per kernel (boot) ----
compute_kernel_hw_startup(...);        // exactly once, before any helper (chain.hpp:26-40)
prepare_stat_constants();              // cb_scaler (1 page, SUM ⇒ 1.0), cb_w_mask (1 page, gated)
load_gamma_once();                     // cb_gamma_tiles: slice_hidden_tiles tiles, row-0-valid, resident

for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);                 //  x slice  -> cb_input_tiles
    tilize_block(block_idx);               //  ROW_MAJOR only: cb_rm_stage_in -> cb_input_tiles
    mask_tail_block(block_idx);            //  gated: zero the W-pad lanes, IN PLACE in cb_input_tiles
    square_accumulate_block(block_idx);    //  sum_of_squares -> cb_sq_partials
    collapse_partial_block(block_idx);     //  within-tile REDUCE_ROW -> cb_slice_stat | cb_rms_recip
    combine_block(block_idx);              //  s>1: gather -> root sum + finalize -> mcast -> cb_rms_recip
    scale_block(block_idx);                //  x *= rsqrt(...)  (bcast Col); in place, or -> sink
    apply_gamma_block(block_idx);          //  gated on gamma: x * gamma (bcast Row) -> sink
    untilize_block(block_idx);             //  ROW_MAJOR only: cb_input_tiles -> cb_rm_stage_out
    store_block(block_idx);                //  -> DRAM
}
```

The **sink** of the last normalize phase is layout-dependent, and this is a CB-ownership consequence,
not a tuning choice:

| Layout | gamma | `scale_block` writes | `apply_gamma_block` writes | Reason |
|--------|-------|----------------------|----------------------------|--------|
| TILE | present | `cb_input_tiles` (in place) | `cb_output_tiles` | The writer is a *different processor*, so `cb_output_tiles` streams at depth `out_cb_depth`. It cannot be `cb_input_tiles`: that would make the writer a second consumer of a compute-owned CB (silent UB). |
| TILE | absent | `cb_output_tiles` | *elided* | A `copy` pass just to reach the output CB would be pure waste. |
| ROW_MAJOR | present | `cb_input_tiles` (in place) | `cb_input_tiles` (in place) | The consumer is `untilize`, another **compute** helper. Two sequential compute helpers cannot overlap, so any CB between them must hold the whole `B*S` block — writing in place instead removes that buffer entirely. `untilize` then pops the `B*S` window. |
| ROW_MAJOR | absent | `cb_input_tiles` (in place) | *elided* | same |

Per-operation contract:

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------|-------------|--------------------|-------------------------------|
| `prepare_stat_constants` | 1 tile ×2 | `cb_scaler`, `cb_w_mask` for the whole kernel | **once per kernel**; the scaler is waited but never popped (`reduce_helpers_compute.inl:955`) |
| `load_gamma_once` | `(1, slice_hidden_tiles)` | `cb_gamma_tiles` for the whole kernel | **once per kernel** — one DRAM read of this core's gamma slice |
| `load_block` | `(block_rows, slice_hidden_tiles)` | — | one NoC read burst per `min(4…8, slice_hidden_tiles)`-tile chunk, **one barrier per chunk**, reads on **NoC0** (`noc_placement/report.md:20-37`) |
| `tilize_block` | `(block_rows, slice_hidden_tiles)`, one tile-row per LLK block | `cb_rm_stage_in` window | `InitOnly` / `Neither` / `UninitOnly` across the block's tile-rows so LLK init is paid **once per block**, not per tile-row (`tilize_helpers.hpp:180-185`) |
| `mask_tail_block` | `(block_rows, 1)` — the last hidden tile of each row | `cb_input_tiles` block (held, not popped) | **once per block**, and only on the one core per row-group that owns the global last hidden tile, and only when `W % 32 != 0` |
| `square_accumulate_block` | `(block_rows, slice_hidden_tiles)` | `cb_input_tiles` block — x is **not** consumed here; it is needed again by `scale_block` | one DEST acquire/pack **per tile-row** (`DestAccumulation::PerRow`), one init + one reconfig **per block** |
| `collapse_partial_block` | `(block_rows, 1)` | — | one `reduce_init` + one format reconfig **per block** |
| `combine_block` | `(num_hidden_slices, block_rows)` gathered → `(block_rows, 1)` mcast | `cb_gathered_partials` (root only) | **one gather + one mcast + one semaphore handshake per block** ⇒ one per core at the Phase 0 default. Contributors write `block_rows` single-tile NoC writes at stride `num_hidden_slices` |
| `scale_block` | `(block_rows, slice_hidden_tiles)` | `cb_input_tiles` block, rewritten **in place** | one init + one reconfig **per block**; no reserve/push on `cb_input_tiles` |
| `apply_gamma_block` | `(block_rows, slice_hidden_tiles)` | `cb_gamma_tiles` (whole kernel) | one init + one reconfig **per block**; `cb_output_tiles` reserve/push **per tile-row** (`PerOuter`) to keep the writer overlapped |
| `untilize_block` | `(block_rows, slice_hidden_tiles)` | `cb_rm_stage_out` window | init amortized across the block's tile-rows, same `InitOnly`/`Neither`/`UninitOnly` idiom |
| `store_block` | `(block_rows, slice_hidden_tiles)` | — | one barrier per 4–8-tile chunk (or per 32 sticks in RM), writes on **NoC1** |

**Architecturally important state.** `cb_input_tiles` must hold the block across **three** phases
(`square_accumulate_block` → `scale_block` → `apply_gamma_block`) and is rewritten in place twice.
The implementer owns its `cb_wait_front(block_rows·slice_hidden_tiles)` / `cb_pop_front(...)` window
explicitly: every chain that touches it uses `WaitPolicy::None` / `PopPolicy::None` /
`ReservePolicy::None` / `PushPolicy::None` so the helper never issues a competing handshake
(`chain.hpp:225-234`, and the in-place hazard note in
`ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/lifecycle/inplace_chain.cpp:5-21`).

**Stall-shadow analysis.** `combine_block` is the only stage that blocks on data it does not produce:
non-root cores wait for the mcast, and the root waits for `num_hidden_slices − 1` remote partials.

- **The root's wait is already shadowed by construction:** the root computes its own
  `square_accumulate_block` + `collapse_partial_block` *before* it starts waiting, so its slice's work
  is never inside the window.
- **The non-root wait is shadowed by the coarsest-block default.** Because `block_rows` defaults to
  the whole per-core assignment, `num_blocks_this_core == 1` and the stall is paid exactly **once**,
  after all local Σx² work and before all normalize work. There is genuinely no other independent
  work on that core in that window: `scale_block` and everything after it depend on the awaited value,
  and `load_gamma_once` is already hoisted out of the loop into boot.
- **When the L1 solve forces `num_blocks_this_core > 1`**, block `b+1`'s `load_block` *is* independent
  of block `b`'s combine and can be scheduled into the window — but only by raising `in_cb_depth` to
  2, which doubles the largest CB. Recorded as the **overlap perf lamp**, not assumed free.
- **No floating-point reorder is introduced.** The per-row summation order is fixed
  (`sum_of_squares` walks each row left-to-right in DEST; the root sums contributors in core order;
  the `×1/W` is applied exactly once, after the combine). Nothing above depends on
  re-associating the sum, so no precision baseline is at risk.

### Lamps

**Scheme lamps** — scheme-changes Phase 0 deliberately leaves room for:

| Lamp | Scheme-change it leaves room for | How the structure keeps it reachable |
|------|----------------------------------|--------------------------------------|
| **Gamma broadcast** (from the operand-reuse check: gamma does not vary along the row axis, so a row-split re-reads it on every row-group) | Read gamma once per hidden slice on an injector core and `Mcast1D(PerColumn)` it down the row-group direction, instead of `num_row_groups` DRAM reads. Positive reason not to build: worth ~11 % of DRAM bytes in the worst realistic regime and 0 % in the decode regime, against a measured 1.71–1.91× device win per 8–11× read-count reduction. | `load_gamma_once` is already a separate, once-per-kernel named operation over a per-core hidden slice with its own CB. Swapping its DRAM read for an mcast receive touches one operation. `Mcast2D`'s `next_base_sem_id()` (`host/mcast_host.hpp:607`) is designed to chain a second family on the same grid. |
| **Reduce-scatter the contributor axis** | Replace gather-to-root + mcast with reduce-scatter + mcast, so all `num_hidden_slices` cores share the partial-sum work instead of the root carrying it. Positive reason not to build: our payload is `block_rows` tiles (1–8) and at **1 tile/core** the measured latency floor puts tree/root ahead of reduce-scatter (1271–1377 ns vs 1981 ns, `tensix_all_reduce/report.md:85-99`); reduce-scatter's advantage appears at 6 tiles/core and is contention-sensitive (2287→6443 ns, noise 1 %→15-28 %). | The combine is one named operation (`combine_block`) with the contributor layout stated (`page = row·num_hidden_slices + contributor`). Reduce-scatter changes which core sums which page range and adds an all-gather — no other phase or CB moves. |
| **Physical shard placement** (`HEIGHT_/WIDTH_/BLOCK_SHARDED`) | Consume the caller's shard in place. Classified against *this* op's axis characters, not the flavor's name: `HEIGHT_SHARDED` cuts the **independent** row axis ⇒ **knob-turn** (the reduce stays core-local); `WIDTH_SHARDED` cuts the **dependent** hidden axis ⇒ the **already-built** cross-core combine, so it is also only a placement change; `BLOCK_SHARDED` cuts both ⇒ exactly the Phase 0 2D scheme with the geometry pinned by the caller. Positive reason not to build in Phase 0: this is the *support surface*, which is additive — `feature_spec.py:210-218` explicitly frames these as growth of `SUPPORTED["memory_layout"]`. | The logical scheme is already the shard geometry. Phase 0 keeps `num_row_groups` / `num_hidden_slices` / `core_row_tiles` / `slice_hidden_tiles` as host-derived parameters, so a sharded call simply *reads them off the shard spec* instead of computing them. The one implementation difference is stated natively in §Dataflow Strategy: a physical shard's block is **already in the core's L1** and must be consumed through a CB backed on the sharded buffer (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy) — **no NoC re-read** — which is why `load_block` is a named operation with a placement-dependent mechanism rather than a hardcoded `TensorAccessor` read. |
| **TwoPassStreaming** (sub-chunk the hidden slice, re-read x) | `Accumulate::at(cb_acc, chunk)` across hidden chunks (`reduce_helpers_compute.hpp:367-374`) with x re-read for the normalize pass. Positive reason not to build: proven unreachable for this op's shape universe by a factor of ~8 (§Regimes). | `block_hidden_tiles` is a parameter, and the Σx² path is already a `reduce` call whose `Accumulate` template slot is exactly the cross-chunk mechanism. |

**Perf lamps** — defaults that may be wrong *here*:

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Grid synchronization** | The default fills the grid subject to `hidden_tiles_per_core_floor = 4`. The measured fastest width-shard geometries land at 4–8 hidden tiles per core (`feature_spec.py:343-346`), and at `Wt=32` the winner used **8** cores, not the 32 a floor of 1 would pick. Too many slices makes the gather + mcast + handshake dominate a 1-tile payload. | Sweep `hidden_tiles_per_core_floor ∈ {1, 4, 8, 16}` on the four interleaved decode perf cases. Also compare `num_row_groups` at every divisor of `grid_y` for the prefill cases (the 2D vs 1D partition trade in the traffic table). |
| **Overlap** | `block_rows = core_row_tiles` with `in_cb_depth = 1` means `cb_input_tiles` is filled completely before `square_accumulate_block` starts and drained completely before the next block — data movement serializes against compute for the whole block, and the `combine_block` stall has nothing to shadow. | Compare `block_rows = core_row_tiles` / depth 1 against `block_rows = core_row_tiles/2` / `in_cb_depth = 2` (same L1) on the prefill perf cases, where `core_row_tiles` is largest. `compute_block_size/report.md:10-13` says the coarse block wins by 1.64× on fixed cost; `double_buffer/report.md:30-42` says depth 2 wins by 2.78× on overlap — which dominates here is unmeasured. |
| **rsqrt over-computation** | `cb_rms_recip` tiles are **column-0-valid**: 1 of 32 SFPU vectors carries data, and `Rsqrt<>` through the chain / `rsqrt_tile` in a `post_reduce_op` runs all 32 (`rc` = 748 ns/tile, `sfpu_tile_scope/report.md:12-21`). The `c_skip` even-parity address stride would cut it 1.94×, but `rsqrt_tile` **hardcodes its vector mode and exposes neither `ITERATIONS` nor a parity stride** (`sfpu_tile_scope/README.md:95-104`), so it is not reachable through the helper API. | Measure the finalize phase's share with a device zone before reaching for raw sfpi. Cost scales with `block_rows` per core, not with `Wt`, so it is a decode-regime concern, not a prefill one. |
| **`sum_of_squares` vs a wide `reduce` at narrow W** | The DEST-accumulate path is the default because it wins from **W ≥ 4 tiles** — but at **W ≤ 2 tiles** `reduce_fold` is measurably *faster* (466 vs 821 ns @1t, `row_reduce_accumulate/report.md:51-56`). `feature_spec.py` contains many such shapes (`(1,1,32,64)`, `(1,1,3232,96)`, `W=40/72`). | Add a `slice_hidden_tiles <= 2` fast path calling `reduce<SUM, REDUCE_ROW, …>` over a squared block, and measure against the `sum_of_squares` default on the narrow-W INPUTS. Do **not** add it speculatively — it is a second regime with a grid-independent predicate, so it must be regime-pinned if adopted. |

---

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x: DRAM → `cb_input_tiles` (TILE, interleaved) | tiles | `TensorAccessor` reads on NoC0, 4–8 tiles per barrier | Page = tile. `page_id = (global_row_tile)·tensor_hidden_tiles + (hidden_slice_base + j)`. Barrier granularity from `double_buffer/report.md:30-42`. |
| x: DRAM → `cb_rm_stage_in` → `cb_input_tiles` (ROW_MAJOR, interleaved) | sticks → tiles | `TensorAccessor` stick reads on NoC0, then `tilize` | Reader writes each `W`-element stick at stick pitch `align_up(slice_hidden_tiles·32·elem, l1_align)` and **zero-fills the tail** — required both for `tilize` correctness (mechanism cap) and so masked-out lanes are never NaN. A ragged last tile-row (`total_sticks % 32 != 0`) has its missing sticks fully zero-filled. |
| x: L1 shard → `cb_input_tiles` (`*_SHARDED`, lamped) | tiles | **CB backed on the sharded buffer** (`ttnn.cb_descriptor_from_sharded_tensor`), zero-copy, **no NoC read** | Stated here even though Phase 0 does not implement it, because it is the *implementation* of the sharded scheme, not a tuning of it: re-reading a local shard through a `TensorAccessor` would re-fetch over the NoC data already resident in L1. |
| gamma: DRAM → `cb_gamma_tiles` | tiles, **row-0-valid** | TILE gamma: direct tile reads of this core's slice. ROW_MAJOR gamma: one stick read, scattered into row 0 of each of `slice_hidden_tiles` tiles (2 face-writes of 16 elements per tile), remaining rows zeroed once. | Both layouts land the *same* CB contract, so compute is layout-agnostic. `BroadcastDim::Row` reads only row 0 (`chain.hpp:311-313`). Done **once per kernel**. |
| Σx² partial: `cb_input_tiles` → `cb_sq_partials` → `cb_slice_stat` | fp32 tiles | compute-internal CBs (`sum_of_squares`, then within-tile `reduce<REDUCE_ROW>`) | fp32 for accumulation accuracy (`row_reduce_accumulate/README.md:102-109`). `cb_slice_stat` is column-0-valid (`reduce_helpers_compute.hpp:167-171`). |
| **Tensix→Tensix, gather** (`num_hidden_slices > 1`) | fp32 tiles | Contributor `c`'s **writer** kernel reads `cb_slice_stat` and issues `block_rows` single-tile `noc_async_write`s into the root's `cb_gathered_partials` at page `row·num_hidden_slices + c`, then one `noc_semaphore_inc` on the root's progress semaphore. Root's **reader** kernel `cb_reserve_back(num_hidden_slices·block_rows)`, waits the semaphore for `num_hidden_slices − 1` arrivals, copies its **own** slice's `block_rows` tiles into its slots, then `cb_push_back`. | Contract: `cb_slice_stat` has exactly one consumer (the writer kernel) and `cb_gathered_partials` exactly one producer (the root's reader kernel) — the cross-kernel handoff gets its own CB per `ttnn-cb-memory-fundamentals.md:96-104`. Contributor-page stride `num_hidden_slices` is chosen so the root's combine is the **documented** `ReduceWithinTile::Skip` case: `ReduceInputBlockShape::of(block_rows, num_hidden_slices)` with `REDUCE_ROW` over column-0-valid partials. |
| **Tensix→Tensix, broadcast** (`num_hidden_slices > 1`) | fp32 tiles | Root compute → `cb_rms_bcast`; root's **reader** kernel is a `SenderPipe` (`mcast_pipe.hpp:178-228`) over the row-group rectangle emitted by `Mcast2D` (`host/mcast_host.hpp:448-516`); every core (root included, via `src_l1 != dst_l1` **loopback** mode) lands `block_rows` tiles in `cb_rms_recip`. Non-root cores run `ReceiverPipe::receive()` (`mcast_pipe.hpp:256-287`). | `cb_rms_bcast` (root, compute→reader) and `cb_rms_recip` (reader→compute) are **distinct CBs** so neither grows a second consumer. `mcast_pipe` touches no CB itself — the kernel does `reserve_back` / `get_write_ptr` / `push_back` around it, and `dst_l1` is identical on all receivers because `cb_rms_recip` has the same index and geometry on every core. `consumer_ready` must be host-initialized to 0 (`mcast_pipe.hpp:21-45`). |
| Normalize | tiles | `cb_input_tiles` rewritten **in place** by `scale_block`, then `apply_gamma_block` reads it and packs to `cb_output_tiles` | Two FPU broadcast multiplies with an L1 round-trip between them, deliberately **not** fused through DEST: `compute_fusion/README.md:82-105` measures DEST-reuse for an FPU consumer at **0.94×** (the isolated combine step at 0.82×), i.e. the L1 round-trip is 1.22× *faster*, and the penalty is per-tile. `DestReuseBinary` also carries a plain `InputSpec` (`chain.hpp:518-520`) with no `BroadcastDim`, so gamma's row broadcast could not be expressed through it. |
| Output: `cb_output_tiles` → DRAM (TILE) | tiles | `TensorAccessor` writes on NoC1, 4–8 tiles per barrier | Same page mapping as the input read. |
| Output: `cb_output_tiles` → `cb_rm_stage_out` → DRAM (ROW_MAJOR) | tiles → sticks | `untilize`, then stick writes on NoC1 | Writer emits only the `W` valid elements of each stick and only the valid sticks of a ragged last tile-row. |

---

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | **a block** = `(block_rows, slice_hidden_tiles)` tiles of x |
| Grid | `grid_x, grid_y = device.compute_with_storage_grid_size()` — a **runtime query**, never a literal. Cores are enumerated `row_wise=True` (`ttnn.grid_to_cores`) so a hidden line lies along a grid row (`noc_placement/report.md:20-37`: a column line is 2.91× slower when bandwidth-bound). |
| Per-core work | `core_row_tiles` tile-rows of one hidden slice; `num_blocks_this_core = ceil_div(core_row_tiles, block_rows)`; Phase 0 default `block_rows = core_row_tiles` ⇒ **1 block per core** |
| Remainder | alignment-aware throughout: `ceil`, per-image where the layout pads per image (below). Ragged last block uses a shorter **runtime** `last_block_rows` passed into the *same* block operations; a ragged core pushes/waits its **actual** page count. Ragged hidden slice: the last slice's tiles beyond `tensor_hidden_tiles` are simply not read or written (`slice_hidden_tiles` stays uniform for the `tilize`/`untilize` template cap). |

### Tile geometry (alignment-aware, per-image)

```python
Wt = ttnn.div_up(W, 32)                                  # ceil, never W // 32

if layout is TILE_LAYOUT:
    # TILE pads H per image, so tile-rows must be counted per image and summed.
    num_images = prod(input.padded_shape[:-2])           # 1 for rank 2
    Rt = num_images * ttnn.div_up(H, 32)                 # NOT floor(num_images*H/32)
    assert Rt * Wt == input.buffer_num_pages()           # cross-check against the real buffer
else:  # ROW_MAJOR
    # No tile padding in the buffer. Rows may be folded across image boundaries because
    # the reduce runs along W only — rows sharing a tile never interact.
    total_sticks = input.buffer_num_pages()              # prod(shape[:-1])
    Rt = ttnn.div_up(total_sticks, 32)                   # ceil; last tile-row may be ragged
```

### Regime-selection function (exact, host-checkable)

```python
gx, gy = device.compute_with_storage_grid_size()
HIDDEN_TILES_PER_CORE_FLOOR = 4          # single source of truth (measured; see Overview)

def working_tiles(block_rows, S, *, layout, has_gamma, s, is_root):
    """Peak per-core CB tiles. Mirrors l1_ledger.md exactly — one definition, host-side."""
    t  = block_rows * S                          # cb_input_tiles
    t += S if has_gamma else 0                   # cb_gamma_tiles
    t += block_rows                              # cb_sq_partials
    t += block_rows if s > 1 else 0              # cb_slice_stat
    t += (s * block_rows + block_rows) if (is_root and s > 1) else 0   # gathered + rms_bcast
    t += block_rows                              # cb_rms_recip
    if layout is ROW_MAJOR:
        # No cb_output_tiles: the last normalize phase packs in place into
        # cb_input_tiles and untilize reads it from there (two sequential compute
        # helpers cannot pipeline through a CB, so a buffer here would need B*S).
        t += RM_IN_DEPTH * S                     # cb_rm_stage_in  (stick pages, S tile-equivalents)
        t += RM_OUT_DEPTH * S                    # cb_rm_stage_out (tile pages)
    else:
        t += OUT_CB_DEPTH * S                    # cb_output_tiles
    t += 2                                       # cb_scaler + cb_w_mask
    return t

def fits(block_rows, S, **kw):
    return working_tiles(block_rows, S, **kw) * tile_bytes <= l1_working_budget

# 1. OCCUPANCY FIRST: pick the 2D partition. Prefer MORE row groups (fewer combines),
#    subject to the hidden-granularity floor and to L1 residency.
for g in sorted((d for d in divisors(gy) if d <= max(1, Rt)), reverse=True):
    s = min(gx * (gy // g), Wt, max(1, ttnn.div_up(Wt, HIDDEN_TILES_PER_CORE_FLOOR)))
    S = ttnn.div_up(Wt, s)
    if fits(1, S, s=s, is_root=True, ...):
        num_row_groups, num_hidden_slices, slice_hidden_tiles = g, s, S
        break
else:
    raise RuntimeError("no partition fits L1 — TwoPassStreaming regime required (lamped)")

# 2. THEN the coarsest block that fits within a core's assignment.
rect_h        = gy // num_row_groups
core_row_tiles = <this core's share from ttnn.split_work_to_cores(row_group_extent)>
block_rows = max(1, min(core_row_tiles,
                        largest B <= core_row_tiles with fits(B, slice_hidden_tiles, ...)))
num_blocks_this_core = ttnn.div_up(core_row_tiles, block_rows)
```

Regime = `num_hidden_slices == 1` (RowParallel) vs `> 1` (BlockParallel + combine). **Regime-pinned
tests are mandatory** — the predicate reads `grid_y` and `Rt`, so a regime reachable on an 8×8 part
may be unreachable on another. `feature_spec.py`'s `_WIDE` `LOOSE_CASES` (`feature_spec.py:222-228`)
pin `num_hidden_slices > 1`; the acceptance test pins both via shape choice.

Core-assignment layout, per row-group rect `y ∈ [r·rect_h, (r+1)·rect_h)`:

| Quantity | Value |
|----------|-------|
| Cores in the rect | `grid_x · rect_h` = `num_hidden_slices` |
| Hidden slice of core `(x, y)` | `slice_index = (y − r·rect_h)·grid_x + x`; base tile `slice_index · slice_hidden_tiles` |
| Root of the rect | `(0, r·rect_h)`; `Mcast2D(device, rect, root, McastConfig(...))` |
| Tile-rows of the rect | `ttnn.split_work_to_cores`-style even split of `Rt` over `num_row_groups`, remainder to the low-indexed rects |
| Owns the global last hidden tile (⇒ `mask_tail_block` enabled) | `slice_index == num_hidden_slices − 1` (or the last slice that actually contains tiles when the split is ragged) **and** `W % 32 != 0` **and** `layout is TILE` |

`ROW_MAJOR` does not need `mask_tail_block`: the reader's mandatory tail zero-fill already makes the
pad lanes exactly `0`, and `0² = 0`. The compute mask is the TILE-layout mechanism, where the pad
lanes are real device bytes that may be poisoned.

---

## Circular Buffers

`B = block_rows`, `S = slice_hidden_tiles`, `s = num_hidden_slices`. Every page count is a function of
these knobs; none is a literal standing in for one, and none is an unbounded op dimension. Full
ledger (capacity vs. live set, axis accounting, sharing) is `l1_ledger.md`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | `tile_size(input.dtype)` | `B * S` | Live set **spans both** block axes: x must be simultaneously resident across `square_accumulate_block`, `scale_block` and `apply_gamma_block` — that residency is what holds the input to **one** DRAM crossing. Streams over no axis. | input dtype | reader (TILE) / compute-`tilize` (RM) | compute | whole block |
| `cb_gamma_tiles` | 1 | `tile_size(gamma.dtype)` | `S` | Spans the hidden axis, streams over the row axis (the same `S` tiles serve every row). Constant for the kernel. Omitted entirely when `gamma is None`. | gamma dtype | reader | compute | whole kernel |
| `cb_sq_partials` | 2 | `tile_size(float32)` | `B` | Spans the row axis (one tile per tile-row), streams over the hidden axis (`S` tiles are folded into each page inside DEST). fp32 for accumulation accuracy. | fp32 | compute | compute | `square_accumulate_block` → `collapse_partial_block` |
| `cb_slice_stat` | 3 | `tile_size(float32)` | `B` | Spans the row axis, streams over the hidden axis. Exists only when `s > 1`; when `s == 1` the finalize is fused into `collapse_partial_block`'s `post_reduce_op` and this CB is not created. | fp32 | compute | writer | `collapse_partial_block` → gather |
| `cb_gathered_partials` | 4 | `tile_size(float32)` | `s * B` | Spans the row axis **and** the contributor axis; streams over the hidden axis. **Root cores only** (`s > 1`). Page layout `row·s + contributor` is what makes the root combine the documented `ReduceWithinTile::Skip` case. | fp32 | reader | compute | `combine_block` |
| `cb_rms_bcast` | 5 | `tile_size(float32)` | `B` | Spans the row axis, streams over the hidden axis. **Root cores only** (`s > 1`). Distinct from `cb_rms_recip` so the mcast source has exactly one consumer. | fp32 | compute | reader | `combine_block` |
| `cb_rms_recip` | 6 | `tile_size(float32)` | `B` | Spans the row axis, streams over the hidden axis. Holds `rsqrt(mean+eps)`, column-0-valid, consumed as `OperandKind::Col` + `BroadcastDim::Col`. | fp32 | reader (`s>1`, mcast landing) / compute (`s==1`) | compute | `combine_block` → `scale_block` |
| `cb_scaler` | 7 | `tile_size(bfloat16)` | `1` | Constant — spans no block axis, streams over none. `PoolType::SUM` ⇒ value `1.0`. bf16 is a hard mechanism requirement (`reduce_helpers_dataflow.inl:185-187`). | bf16 | reader | compute | whole kernel |
| `cb_w_mask` | 8 | `tile_size(bfloat16)` | `1` | Constant — spans no block axis. Row-0-valid `1.0`/`0.0` mask over the last hidden tile's `valid_w_in_last_tile` columns. Created only when `W % 32 != 0`. | bf16 | reader | compute | whole kernel |
| `cb_output_tiles` | 9 | `tile_size(output.dtype)` | `out_cb_depth * S` | **TILE layout only.** Spans the hidden axis; **streams over** the row axis with a window of `out_cb_depth` tile-rows — the *writer* (a different processor) drains it as compute packs, so pipelining works and the live set is the window, not `B*S`. **Does not exist in ROW_MAJOR**: there the consumer would be `untilize`, another compute helper, and two sequential compute helpers own all three TRISCs and cannot overlap, so the CB would have to hold the producer's entire `B*S` output (`ttnn-cb-memory-fundamentals.md:122-154`). Instead the ROW_MAJOR path packs the final result **in place** into `cb_input_tiles` and `untilize` reads it from there — Rule 3.2, and it removes a whole block-sized buffer from the heaviest layout. | output dtype | compute | writer | per tile-row, whole block |
| `cb_rm_stage_in` | 10 | `align_up(S*32*elem, l1_align)` (stick pitch) | `rm_in_depth * 32` | ROW_MAJOR only. Spans the hidden axis (a whole padded stick); streams over the row axis with a `rm_in_depth`-tile-row window. | input dtype | reader | compute | per tile-row |
| `cb_rm_stage_out` | 11 | `tile_size(output.dtype)` | `rm_out_depth * S` | ROW_MAJOR only. Spans the hidden axis; streams over the row axis with a `rm_out_depth`-tile-row window (`untilize`'s producer is compute, its consumer the *writer* — different processors, so pipelining works). Tile-sized pages because `untilize` has no asymmetric-page mode (`untilize_helpers.hpp:109-110`); the writer extracts 32 sticks per tile page. | output dtype | compute | writer | per tile-row |

Every Producer / Consumer cell names exactly one kernel. The two in-place phases
(`mask_tail_block`, `scale_block`) do **not** add a producer to `cb_input_tiles`: they use
`ReservePolicy::None` / `PushPolicy::None` and write into pages compute already holds via its own
`cb_wait_front` window, which is the `blocking-model`/`l1-footprint-discipline` "transform in place"
pattern (Rule 3.2) and not a second producer.

**CB sync (push == wait), per CB:**

| CB | Producer pushes | Consumer waits |
|----|-----------------|----------------|
| `cb_input_tiles` | `B*S` per block (reader: one `push_back(B*S)`; RM: `tilize` pushes `S` per tile-row ⇒ `B*S`) | `B*S` per block — one `cb_wait_front(B*S)`, held across every normalize phase. Popped once: by the compute kernel after `apply_gamma_block`/`scale_block` (TILE), or by `untilize_block` at `S` per tile-row ⇒ `B*S` (RM). Exactly one pop path per layout |
| `cb_gamma_tiles` | `S` once | `S` once (never popped) |
| `cb_sq_partials` | `1` per tile-row ⇒ `B` per block (`PushPolicy::PerOuter`) | `B` per block (`reduce` with `ReduceInputBlockShape::of(B,1)` + `BulkWaitBulkPop`) |
| `cb_slice_stat` | `B` per block | `B` per block (writer) |
| `cb_gathered_partials` | `s*B` per block (root reader, after the semaphore) | `s*B` per block (root compute) |
| `cb_rms_bcast` | `B` per block | `B` per block (root reader) |
| `cb_rms_recip` | `B` per block | `B` per block (`WaitPolicy::Upfront` / `PopPolicy::AtEnd` on an `OperandKind::Col` operand ⇒ waits `Ht = B`) |
| `cb_scaler` | `1` once | `1` once, never popped |
| `cb_w_mask` | `1` once | `1` once (`PopPolicy::None`), never popped |
| `cb_output_tiles` (TILE only) | `S` per tile-row ⇒ `B*S` per block | `S` per tile-row ⇒ `B*S` per block |
| `cb_rm_stage_in` | `32` sticks per tile-row ⇒ `32*B` per block | `min(32, remaining)` per `tilize` block ⇒ `32*B` per block (`total_input_pages = 32*B`) |
| `cb_rm_stage_out` | `S` tile pages per tile-row ⇒ `B*S` per block | `S` per tile-row ⇒ `B*S` per block |

> `cb_rm_stage_out` is declared with **tile-sized pages**, because `untilize_helpers` has no
> asymmetric-page mode (`untilize_helpers.hpp:109-110` — unlike `tilize`, which does) and therefore
> pushes `block_width_tiles` tile pages per block. The writer then extracts 32 sticks per tile page,
> which is the documented untilize writer pattern (`ttnn-cb-memory-fundamentals.md:198-218`), emitting
> only the `W` valid elements of each stick.

---

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0a | `prepare_stat_constants` | `1` tile ×2 | yes | — | `cb_scaler` (1), `cb_w_mask` (1) | both pushed once, never popped for the kernel's life |
| 0b | `load_gamma_once` | `(1, S)` | no (dataflow) | — | `cb_gamma_tiles` (`S`) | pushed once; compute waits `S` once and never pops — resident |
| 1 | `load_block` | `(B, S)` | no (dataflow) | — | `cb_input_tiles` (`B*S`) (TILE) or `cb_rm_stage_in` (`32` per tile-row) (RM) | TILE: one `push_back(B*S)`. RM: `32` stick pages per tile-row |
| 1b | `tilize_block` (RM only) | `(B, S)`, one tile-row per LLK block | yes | `cb_rm_stage_in` (`32` per block, asymmetric pages) | `cb_input_tiles` (`S` per tile-row ⇒ `B*S`) | `cb_rm_stage_in` popped per tile-row; `cb_input_tiles` fully pushed |
| 2 | `mask_tail_block` (gated) | `(B, 1)` — tile `(S−1)` of each row | yes | `cb_input_tiles` (**held**, `WaitPolicy::None`/`PopPolicy::None`, `TileOffset::Strided{base=S−1, row_stride=S}`, `OperandKind::Col`), `cb_w_mask` (1, `BroadcastDim::Row`, `PopPolicy::None`) | `cb_input_tiles` **in place** (`ReservePolicy::None`/`PushPolicy::None`, same strided offset) | `cb_input_tiles` unchanged in page accounting; W-pad lanes now exactly `0` |
| 3 | `square_accumulate_block` | `(B, S)` | yes | `cb_input_tiles` (**held**, `WaitPolicy::None`/`PopPolicy::None`, `OperandKind::Block`) | `cb_sq_partials` (`1` per tile-row ⇒ `B`) | x still resident; `cb_sq_partials` holds `B` raw per-row Σx² tiles (all 32 columns are per-column sums) |
| 4 | `collapse_partial_block` | `(B, 1)` | yes | `cb_sq_partials` (`B`, `BulkWaitBulkPop`), `cb_scaler` (1, waited not popped) | `s>1`: `cb_slice_stat` (`B`); `s==1`: `cb_rms_recip` (`B`) with the finalize fused as `post_reduce_op` | `cb_sq_partials` popped. Output is **column-0-valid** |
| 5 | `combine_block` (`s>1` only) | gather `(B, s)` → mcast `(B, 1)` | mixed | root compute: `cb_gathered_partials` (`s*B`, `BulkWaitBulkPop`), `cb_scaler` (1) | root compute → `cb_rms_bcast` (`B`); root reader mcasts → every core's `cb_rms_recip` (`B`) | `cb_gathered_partials` popped; `cb_rms_recip` holds `r` on every core in the rect |
| 6 | `scale_block` | `(B, S)` | yes | `cb_input_tiles` (**held**), `cb_rms_recip` (`WaitPolicy::Upfront`/`PopPolicy::AtEnd`, `OperandKind::Col`, `BroadcastDim::Col`) | per the sink table above: `cb_input_tiles` **in place**, or `cb_output_tiles` (`S` per tile-row) when TILE + no gamma | `cb_rms_recip` popped (`B`). In the in-place case x is replaced by `x·r` |
| 7 | `apply_gamma_block` (gamma only) | `(B, S)` | yes | `cb_input_tiles` (**held**), `cb_gamma_tiles` (`S`, `OperandKind::Row`, `BroadcastDim::Row`, `PopPolicy::None`) | TILE: `cb_output_tiles` (`S` per tile-row, `ReservePolicy::PerOuter`/`PushPolicy::PerOuter`) — and the compute kernel pops `cb_input_tiles`'s `B*S` window here. RM: `cb_input_tiles` **in place**, window popped later by `untilize_block` | `cb_gamma_tiles` still resident |
| 8 | `untilize_block` (RM only) | `(B, S)`, one tile-row per LLK block | yes | `cb_input_tiles` (`WaitMode::NoWait` — compute already holds the window; `untilize` still pops `S` per tile-row ⇒ `B*S`) | `cb_rm_stage_out` (`S` per tile-row) | `cb_input_tiles` window fully popped and released for the next block |
| 9 | `store_block` | `(B, S)` | no (dataflow) | `cb_output_tiles` (TILE) or `cb_rm_stage_out` (RM) | — | popped per tile-row |

---

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `prepare_stat_constants` (scaler) | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:98` | `<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()` — the **pool-type-aware** overload; `SUM` ⇒ value `1.0`, `reduce_factor` ignored | — | `cb_scaler` (1 page) | none (constant) |
| `prepare_stat_constants` (W mask) | helper | `dataflow_kernel_lib::prepare_reduce_mask` | `reduce_helpers_dataflow.hpp:74` | `<cb_w_mask, ReduceDim::REDUCE_ROW>(valid_w_in_last_tile)` → row-0-valid mask for `mul_tiles_bcast_rows` | — | `cb_w_mask` (1 page) | none; `valid_w_in_last_tile` is derived once on host from `W` |
| `tilize_block` | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:197` | `<block_hidden_tiles, cb_rm_stage_in, cb_input_tiles, InitUninitMode::{InitOnly\|Neither\|UninitOnly}, WaitMode::WaitBlock>(num_blocks = block_rows, total_input_pages = 32*block_rows)` (asymmetric-page mode) | `cb_rm_stage_in` | `cb_input_tiles` | **`block_hidden_tiles`** (template = tiles per LLK block) and **`block_rows`** (runtime `num_blocks`) |
| `mask_tail_block` | helper | `compute_kernel_lib::mul` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:53` | `mul<input(cb_input_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Col, TileOffset::Strided), input(cb_w_mask, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None), output(cb_input_tiles, ReservePolicy::None, PushPolicy::None, TileOffset::Strided)>(IterationShape::grid(block_rows, 1))` with `StridedTileRange{base = block_hidden_tiles − 1, row_stride = block_hidden_tiles}` (`chain.hpp:277-282`) | `cb_input_tiles`, `cb_w_mask` | `cb_input_tiles` (in place) | **`block_rows`** (grid H), **`block_hidden_tiles`** (the stride and the base) |
| `square_accumulate_block` | helper | `compute_kernel_lib::sum_of_squares` | `eltwise/api/convenience.hpp:86` (expansion `convenience.inl:36-47`) | `sum_of_squares<input(cb_input_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Block), row_output(cb_sq_partials)>(IterationShape::grid(block_rows, block_hidden_tiles))` — accumulates `x·x` in `D0` per row via `DestAccumulation::PerRow`, packs one tile per row (`ReservePolicy::PerOuter`/`PushPolicy::PerOuter`) | `cb_input_tiles` | `cb_sq_partials` | **`block_rows` × `block_hidden_tiles`** (the whole `IterationShape::grid`) |
| `collapse_partial_block` | helper | `compute_kernel_lib::reduce` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:606` | `<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sq_partials, cb_scaler, out_cb, ReduceInputPolicy::BulkWaitBulkPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::AccumulateViaAdd, NoAccumulation, PostReduceOp, ReduceWithinTile::Collapse>(ReduceInputBlockShape::of(block_rows, 1))`. `out_cb` = `cb_slice_stat` (`s>1`, `PostReduceOp = NoOp`) or `cb_rms_recip` (`s==1`, `PostReduceOp` = the finalize below) | `cb_sq_partials`, `cb_scaler` | `cb_slice_stat` \| `cb_rms_recip` | **`block_rows`** (`ReduceInputBlockShape::rows`) |
| `combine_block` — root sum + finalize | helper | `compute_kernel_lib::reduce` | `reduce_helpers_compute.hpp:606` | `<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_gathered_partials, cb_scaler, cb_rms_bcast, BulkWaitBulkPop, …, ReduceAlgorithm::AccumulateViaAdd, NoAccumulation, PostReduceOp, ReduceWithinTile::Skip>(ReduceInputBlockShape::of(block_rows, num_hidden_slices))`. **`Skip` is the documented case**: summing per-core partials that each came out of a `REDUCE_ROW` and are therefore column-0-valid (`reduce_helpers_compute.hpp:167-171`) | `cb_gathered_partials`, `cb_scaler` | `cb_rms_bcast` | **`block_rows`** (rows) and **`num_hidden_slices`** (cols) |
| finalize (`post_reduce_op` slot, both reduces) | helper extension point + raw SFPU | `post_reduce_op(uint32_t dst_idx)` lambda | contract at `reduce_helpers_compute.hpp:491-495`; worked example `reduce_helpers_compute.hpp:580-591` | `mul_unary_tile(dst_idx, bits(1.0f/W))` → `add_unary_tile(dst_idx, bits(epsilon))` → `rsqrt_tile_init(); rsqrt_tile(dst_idx)`. **`1/W` uses the true element count `W`, not `32*Wt`, and is applied exactly once — after the cross-core combine** | (DEST) | (DEST) | none; `W` and `epsilon` are host-derived RT args |
| `combine_block` — gather | raw_api | `noc_async_write` + `noc_semaphore_inc` | `tt_metal/hw/inc/dataflow_api.h` (`api/dataflow/dataflow_api.h` include path) | contributor `c` writes `block_rows` single-tile pages into the root's `cb_gathered_partials` at page `row·num_hidden_slices + c`; then one semaphore increment | `cb_slice_stat` | root's `cb_gathered_partials` | **`block_rows`** (write count), **`num_hidden_slices`** (page stride) |
| | | **Helpers considered and rejected** | | `dataflow_kernel_lib::SenderPipe::send` (`mcast_pipe.hpp:197`) is a **multicast** to a rectangle with a single `dst_l1` identical on all receivers (`mcast_pipe.hpp:44-45`). The gather is the opposite shape: `num_hidden_slices` *different* sources writing to `num_hidden_slices` *different* destination pages on one core. Using `SenderPipe` per contributor with a 1-core rect would work mechanically but adds a per-contributor handshake pair for a single unicast, and `mcast_pipe` deliberately touches no CB (`mcast_pipe.inl` has no `cb_*` calls), so the CB reserve/push would still be hand-written. No gather/scatter helper exists in `kernel_lib`. | | | |
| `combine_block` — broadcast | helper | `dataflow_kernel_lib::McastArgs` → `SenderPipe::send` / `ReceiverPipe::receive` | `mcast_pipe.hpp:328` (`McastArgs`), `:197` (`send`), `:274` (`receive`) | `McastArgs<CT_BASE, RT_BASE>` → `.sender(noc)` on the root, `.receiver(noc)` elsewhere; `send(src_l1 = get_read_ptr(cb_rms_bcast), dst_l1 = get_write_ptr(cb_rms_recip), size = block_rows*tile_bytes)`. Root is in the rect and `src_l1 != dst_l1` ⇒ **loopback** mode delivers to the root's own `cb_rms_recip` too (`mcast_pipe.inl:84-90`) | `cb_rms_bcast` | `cb_rms_recip` (all cores in the rect) | **`block_rows`** (payload size) |
| `combine_block` — host wire | helper | `ttnn.Mcast2D` / `ttnn.McastConfig` | `ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp:448` (`Mcast2D`), `:87` (`McastConfig`); Python binding `ttnn/cpp/ttnn-nanobind/mcast_host.cpp` | one `Mcast2D(device, row_group_rect, root, McastConfig(noc=NOC_0, handshake=True, sem_ids=[...]))` per row-group rect; `compile_time_args()` / `runtime_args(core)` / `owned_semaphores()` | — | — | **`num_row_groups`** (how many rects), rect geometry |
| `scale_block` | helper | `compute_kernel_lib::mul` | `eltwise/api/convenience.hpp:53` | `mul<input(cb_input_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Block), input(cb_rms_recip, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Col, TileOffset::Unset), output(...)>(IterationShape::grid(block_rows, block_hidden_tiles))`. `BroadcastDim::Col` because a `REDUCE_ROW` result is column-shaped and broadcasts back **across columns** (`chain.hpp:311-313`). Output = `cb_input_tiles` in place (gamma present) or `cb_output_tiles` (gamma absent) | `cb_input_tiles`, `cb_rms_recip` | `cb_input_tiles` \| `cb_output_tiles` | **`block_rows` × `block_hidden_tiles`** |
| `apply_gamma_block` | helper | `compute_kernel_lib::mul` | `eltwise/api/convenience.hpp:53` | `mul<input(cb_input_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Block), input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None, OperandKind::Row, TileOffset::Unset), output(cb_output_tiles, ReservePolicy::PerOuter, PushPolicy::PerOuter)>(IterationShape::grid(block_rows, block_hidden_tiles))` | `cb_input_tiles`, `cb_gamma_tiles` | `cb_output_tiles` | **`block_rows` × `block_hidden_tiles`** |
| `untilize_block` | helper | `compute_kernel_lib::untilize` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:154` | `<block_hidden_tiles, cb_input_tiles, cb_rm_stage_out, InitUninitMode::{InitOnly\|Neither\|UninitOnly}, WaitMode::NoWait>(num_blocks = block_rows)`. `NoWait` because compute already holds the `B*S` window; `untilize` still pops `S` per tile-row, which *is* the window's release. Internally sub-blocks when `block_hidden_tiles > DEST_AUTO_LIMIT` (`untilize_helpers.inl:80-86`) — no caller-side clamp needed. Note the `ReconfigureRegisterDatatypeMode` caveat at `untilize_helpers.hpp:54-61` when pairing with standalone `untilize_init`/`untilize_uninit` | `cb_input_tiles` | `cb_rm_stage_out` | **`block_hidden_tiles`** (template), **`block_rows`** (runtime) |
| `load_block` / `store_block` | raw_api | `TensorAccessor` + `noc_async_read`/`noc_async_write` | `tech_reports/tensor_accessor/tensor_accessor.md`; kernel pattern in `.claude/references/ttnn-cb-memory-fundamentals.md:244-268` | `TensorAccessorArgs<N>()`, `accessor.get_noc_addr(page_id)`; one barrier per 4–8-tile chunk | — | `cb_input_tiles` / — | chunk size = `min(dm_chunk_tiles, block_hidden_tiles)`, `dm_chunk_tiles` default 8 |
| | | **Helpers considered and rejected** | | Interleaved DRAM page addressing has no kernel-lib helper — `TensorAccessor` *is* the sanctioned mechanism for it (`ttnn-cb-memory-fundamentals.md:244-268`), and `tilize_helpers_dataflow.hpp` covers only the stick→tile staging arithmetic, not the DRAM read itself. | | | |
| `load_gamma_once` (ROW_MAJOR gamma) | raw_api | `noc_async_read` into row 0 of each tile page | `api/dataflow/dataflow_api.h` | 2 writes of 16 elements per tile (face 0 row 0, face 1 row 0), plus a one-time `noc_async_write_zeros` over the remaining rows | — | `cb_gamma_tiles` | **`block_hidden_tiles`** (tile count) |
| | | **Helpers considered and rejected** | | `compute_kernel_lib::tilize` (`tilize_helpers.hpp:197`) tilizes **32** sticks into a tile-row; gamma is **one** stick and only row 0 is ever read by `BroadcastDim::Row` (`chain.hpp:311-313`), so a tilize would fabricate 31 dummy sticks and cost a full LLK pass to produce data the FPU discards. `dataflow_kernel_lib` has no single-stick scatter helper. | | | |
| all compute kernels | helper | `compute_kernel_hw_startup` | contract at `eltwise/core/chain.hpp:26-40`; also `reduce_helpers_compute.hpp:31-35` | called **exactly once**, as the first statement of `MAIN()`, never in a loop | — | — | — |
| DEST budget | helper | `compute_kernel_lib::DEST_AUTO_LIMIT` | `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:103` (capacity table `:22-26`) | resolves to **4** tiles under `fp32_dest_acc_en=True` + half-sync. Never spell a literal `8` (`chain.hpp:406-408`) | — | — | — |

---

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `mask_tail_block` | `mul` | `cb_input_tiles` (last hidden tile of each row) — All `[32,32]` | `cb_w_mask` — Row0 (`prepare_reduce_mask<…, REDUCE_ROW>` fills row-0 layout for `mul_tiles_bcast_rows`) | `BroadcastDim::Row` |
| `scale_block` | `mul` | `cb_input_tiles` — All `[32,32]` | `cb_rms_recip` — **Col0** (a `REDUCE_ROW` output is column-0-valid, `reduce_helpers_compute.hpp:167-171`) | `BroadcastDim::Col` — the dim names the axis being *broadcast*, not the axis reduced (`chain.hpp:311-313`) |
| `apply_gamma_block` | `mul` | `cb_input_tiles` (holding `x·r`) — All `[32,32]` | `cb_gamma_tiles` — **Row0** (1D `[W]` operand) | `BroadcastDim::Row` |
| `square_accumulate_block` | `mul` (x·x via `square`) | `cb_input_tiles` — All `[32,32]` | same CB, same tile — All `[32,32]` | `BroadcastDim::None` |
| `combine_block` root sum | `reduce<SUM, REDUCE_ROW, …, Skip>` | `cb_gathered_partials` — Col0 per page (each page is a column-0-valid partial) | `cb_scaler` — Row0, value `1.0` | n/a (FPU reduce; `Skip` suppresses the within-tile collapse so the sum of Col0-valid tiles stays Col0-valid) |

---

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| **Tile padding folded into the RMS denominator** | For `W % 32 != 0` the last hidden tile carries device padding. Folding it in is a near-*uniform scale error* of `sqrt(W_padded/W) − 1` — **PCC is largely blind to it**, so it passes the accuracy gate and ships. `feature_spec.py`'s `pad_poison` group (`feature_spec.py:471-500`) exists precisely to catch it, with `W=40` giving a 26.5 % error. | Two independent guards: (1) `mask_tail_block` zeroes the pad lanes of the last hidden tile in place before Σx² (TILE), and the reader's mandatory tail zero-fill does the same (ROW_MAJOR); (2) the `1/W` factor uses the **true** element count `W`, applied exactly once in the finalize `post_reduce_op` — never `PoolType::AVG`, whose scaler would divide by the padded tile width. |
| **`bfloat16` *accumulation*, not bf16 input, is what loses precision** | The natural mistake is to make the stat CBs match the input dtype. Measured: `reduce_fold` in bf16 DEST is off by **13 ULP** at 32 tiles, while bf16 *input* error actually *averages down* with width (`row_reduce_accumulate/README.md:102-109`). With `fp32_dest_acc_en=True` the DEST side is fp32, but a bf16 `cb_sq_partials` would re-truncate at every pack. | `cb_sq_partials`, `cb_slice_stat`, `cb_gathered_partials`, `cb_rms_bcast`, `cb_rms_recip` are **all `float32`** regardless of input dtype. Format reconfig stays **enabled** on every chain boundary where the format genuinely changes — disabling it there is silent corruption (`compute_block_size/README.md:118-149`). |
| **x must survive three phases in one CB, written in place twice** | `square_accumulate_block` needs raw x, `scale_block` overwrites it with `x·r`, `apply_gamma_block` reads that. If any chain issues its own `cb_pop_front` or `cb_reserve_back` on `cb_input_tiles`, the reader refills pages that are still live — a silent data race that is PCC-invisible when it lands as a scale error. | Every chain touching `cb_input_tiles` uses `WaitPolicy::None` / `PopPolicy::None` / `ReservePolicy::None` / `PushPolicy::None`; the compute kernel owns exactly one `cb_wait_front(B*S)` … `cb_pop_front(B*S)` window per block. The in-place hazard analysis is `kernel_lib/tests/eltwise/chain/lifecycle/inplace_chain.cpp:5-21`. |
| **The mcast source and the mcast landing must be different CBs** | The root's compute produces `r` and the root's reader multicasts it. If both used one CB, the reader would be a second consumer of a compute-owned CB — silent UB, not a hang (`ttnn-cb-memory-fundamentals.md:96-118`). The same applies to `cb_slice_stat` (compute→writer) vs `cb_sq_partials` (compute→compute). | `cb_rms_bcast` (root, compute→reader) and `cb_rms_recip` (reader→compute) are distinct; `cb_slice_stat` is distinct from `cb_sq_partials`. The `s == 1` variant simply does not create `cb_slice_stat` / `cb_gathered_partials` / `cb_rms_bcast`. |
| **`reduce` forbids `input_dfb == output_dfb`** | The obvious L1 saving — collapsing `cb_sq_partials` into `cb_slice_stat` via an in-place reduce — trips a runtime `ASSERT` (`reduce_helpers_compute.inl:895-897`), and under a non-`--dev` build that assert is compiled out, so the failure mode is corrupt data rather than a halt. | Two `B`-page CBs are kept, and the ledger's `Shares with / why not` cell cites this assert as the concrete reason they cannot alias. |
| **`Mcast2D` takes the *bounding box* of the core set as the rect** | Any row-group whose cores are not an exact rectangle multicasts into cores outside the group, handing them a stat tile for rows they do not own (`host/mcast_host.hpp:460-465`). | `num_row_groups` is clamped to a **divisor of `grid_y`**, so every row-group is exactly `grid_x × (grid_y/num_row_groups)` cores. Declared as a mechanism cap. |
| **`consumer_ready` must be host-initialized to 0, not by the sender** | `SenderPipe` deliberately initializes neither semaphore; a ctor `set(0)` would clobber an early remote ack and **hang**, and the hang would appear as `noc_semaphore_wait` on the root only under contention (`mcast_pipe.hpp:21-45`). | Semaphores come from `Mcast2D::owned_semaphores()` (`host/mcast_host.hpp:522`) with `initial_value=0`, declared on the participating core set via `ttnn.SemaphoreDescriptor`. |
| **Two compute regimes selected by a grid-dependent predicate** | `num_hidden_slices > 1` triggers only when `tensor_row_tiles < grid_y`. A kernel that is only ever exercised at `s == 1` on an 8×8 part will fail on a part with a different `grid_y` — the classic passes-on-one-device failure. | The selection function is pinned in §Work Distribution, and **regime-pinned tests are mandatory**: the acceptance test includes a `(1, 1, 32, 8192)`-class wide/short shape (forces `s > 1`) alongside tall shapes (force `s == 1`), and `feature_spec.py`'s `_WIDE` `LOOSE_CASES` pin the wide regime independently. |
| **`ROW_MAJOR` stick pitch in the tilize staging CB is *not* `W·elem_size`** | `tilize` reads a contiguous `32 × (block_hidden_tiles·32)` element region. Writing sticks back-to-back at `W·elem_size` shears the data by `32·Wt − W` elements per row — wrong values, no error. | Stick pitch is `align_up(block_hidden_tiles·32·elem, l1_align)` and the reader zero-fills the tail of every stick. Declared as a mechanism cap. |
| **A ragged last `ROW_MAJOR` tile-row** | `total_sticks % 32 != 0` leaves the last tile-row partly unwritten; whatever L1 held before becomes rows of the tile and gets squared. Uninitialized L1 can be `NaN`. | The reader zero-fills every missing stick of a ragged last tile-row. Those rows are outside the logical shape, so a zero row's degenerate RMS (`rsqrt(eps)`) is harmless. |
| **H padding rows are *not* masked, and must not be** | It is tempting to add an H mask by symmetry with the W mask. It would be wrong work: the reduce runs along W only, so an H-pad row's garbage affects **only its own row's** output, which lies outside the logical shape. Masking H would cost a pass and buy nothing. | Explicitly no H mask. `feature_spec.py`'s `h_non_aligned` and `(1,1,40,40)` poisoned cases are correct under this design because poison in an H-pad row stays in that row (and `1000² × W` is far inside `float32` range, so no `Inf`/`NaN` escapes into the logical rows). |
| **`float32` + `fp32_dest_acc_en=False` must be refused, not silently upgraded** | Accepting it would compute at reduced accumulation width while the caller believes otherwise. | `validate()` rejects the combination (op-side `EXCLUSIONS`, xfail-strict in the golden suite — `.claude/references/precision_convention.md:31-39`), and the config is passed through as `config=compute_kernel_config` after validation. `math_fidelity` / `math_approx_mode` are accepted at any value and honored. |
| **The `"none"` gamma sentinel is legal, always** | Omitting `"none"` from `SUPPORTED["gamma_dtype"]` / `["gamma_layout"]` makes the canonical `no_gamma` cell fall outside `SUPPORTED` and **xpass-fail**, which reads as an op bug rather than a registry bug. | `SUPPORTED` includes `"none"` for both axes; `tag_gamma_dtype` / `tag_gamma_layout` return `"none"` when `gamma is None`; `validate()` never refuses it. The `gamma is None` path elides `cb_gamma_tiles` and `apply_gamma_block` entirely — `scale_block` writes straight to the layout's sink (see the sink table under §Block schedule) rather than running a wasteful `copy` pass. |

### Structural impossibilities (candidates for a future `/golden-tests` pass)

Not edited into `feature_spec.py` — noted here for the user to fold in:

| Candidate `INVALID` cell | Why it is structurally impossible |
|---|---|
| `{"gamma_layout": ROW_MAJOR, "gamma_dtype": float32, "dtype": bfloat16}` — *not* impossible; **do not** add | Listed only to record that mixed input/gamma precision is legitimate (bf16 activations + fp32 weights) and is handled by per-CB formats plus chain format reconfig. No new `INVALID` proposed. |
| `{"memory_layout": WIDTH_SHARDED, "rank": 2}` with `W < 32 · num_cores` | A width shard cannot be formed when the hidden axis has fewer tiles than the shard grid has columns; `eval.sharding.auto_shard_config` should already refuse to synthesize a spec here, in which case no `INVALID` entry is needed. Flagging it so the golden author can confirm which side handles it. |

---

## Phase 0 SUPPORTED rectangle (recommendation to the implementer)

The implementer owns the `SUPPORTED` block; this is the recommended Phase 0 rectangle, narrow on
**placement only**, because that is the axis whose growth is purely additive.

| Axis | Phase 0 recommendation | Refinement candidates (`TARGET − SUPPORTED`) |
|------|------------------------|----------------------------------------------|
| `dtype` | `bfloat16`, `float32` | `bfloat8_b` |
| `fp32_dest_acc_en` | `True` | `False` (with `bfloat16`; `float32 + False` stays an `EXCLUSION`) |
| `layout` | `TILE_LAYOUT`, `ROW_MAJOR_LAYOUT` | — |
| `alignment` | `tile_aligned`, `w_non_aligned`, `h_non_aligned` | — |
| `rank` | `2`, `3`, `4` | — |
| `gamma_mode` | `gamma`, `no_gamma` | — |
| `gamma_dtype` | `bfloat16`, `float32`, `"none"` | `bfloat8_b` |
| `gamma_layout` | `TILE_LAYOUT`, `ROW_MAJOR_LAYOUT`, `"none"` | — |
| `memory_layout` | `INTERLEAVED` | `HEIGHT_SHARDED` (knob-turn: cuts the independent axis), `WIDTH_SHARDED` (placement over the already-built combine), `BLOCK_SHARDED` (placement over the Phase 0 2D scheme) |

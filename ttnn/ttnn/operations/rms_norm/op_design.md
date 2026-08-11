# Operation Design: rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (row-wise normalization with a cross-core reduction combine) |
| Goal | Normalize the last dimension by its root-mean-square, with an optional per-column scale `gamma`. Multi-core from Phase 0, with the hidden dimension splittable across cores and partials combined over the NoC. |
| Math | `output[..., h, w] = input[..., h, w] * rsqrt( (1/W) * Σ_{w'=0}^{W-1} input[..., h, w']² + eps ) * gamma[w]` |
| Mode | Derivative (built on `kernel_lib` block helpers + `mcast_pipe`) |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/precision_convention.md`, `ttnn/ttnn/operations/examples/master.md` (entries: `reduce_block`, `row_reduce_accumulate`, `eltwise_l1_vs_dest_accumulate`, `compute_block_size`, `compute_fusion`, `double_buffer`, `width_split`, `tensix_all_reduce`, `tensix_all_reduce_ring_transport`, `noc_placement`), `ttnn/ttnn/operations/toy_variance/`, `ttnn/ttnn/operations/toy_tilize_untilize/`, `ttnn/ttnn/operations/examples/tensix_all_reduce/program_descriptor_with_inline_kernels.py` |
| Prior-attempt check | No prior attempt at `rms_norm` exists in this tree: no `eval/investigations/`, no `ttnn/ttnn/operations/rms_norm/` before this document, and no `op_design.md` anywhere in the repo. Nothing was mined. |
| Feature spec | `eval/golden_tests/rms_norm/feature_spec.py` — read as authoritative, not modified. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2; TILE or ROW_MAJOR; bfloat16 or float32; INTERLEAVED (Phase 0) | — | tensor |
| `gamma` | `Optional[ttnn.Tensor]` | no | shape `(..., W)` with `gamma.shape[-1] == input.shape[-1]`; TILE or ROW_MAJOR; bfloat16 / float32 | `None` | tensor |
| `epsilon` | `float` | no | > 0 | `1e-6` | CT (fp32 bits) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | keyword-only | `fp32_dest_acc_en == True` in Phase 0; any `math_fidelity` / `math_approx_mode` | `default_compute_kernel_config()` | descriptor |
| `w_group_cols`, `w_group_rows` | host int | derived | divisors of the compute grid extents | selection function below | CT |
| `block_row_tiles` | host int | derived | `1 .. core_row_tiles`, clamped by L1 + `MAX_GATHER_TILES` | L1 solve below | RT |
| `input_cb_depth` | host int | knob | ≥ 1 | `2` | CT |
| `output_cb_depth` | host int | knob | ≥ 1 | `2` | CT |
| `L1_RESERVE_BYTES` | host int | knob | — | `131072` | host constant |
| `MAX_GATHER_TILES` | host int | knob | — | `64` | host constant |

`default_compute_kernel_config()` is exported from `ttnn/ttnn/operations/rms_norm/rms_norm.py` and is the single
source for the `None` case (`math_fidelity=HiFi4`, `fp32_dest_acc_en=True`, `math_approx_mode=False`), per
`.claude/references/precision_convention.md`. After `validate()`, the caller's descriptor is passed through
unchanged as `config=compute_kernel_config`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2, arbitrary `(..., H, W)`; H and W need not be multiples of 32 |
| Dtype | `bfloat16`, `float32` (Phase 0); `bfloat8_b` is a later refinement |
| Layout | `TILE_LAYOUT` or `ROW_MAJOR_LAYOUT`, both handled natively (no host-side `to_layout` / `pad` / `slice`) |
| Memory | INTERLEAVED (Phase 0). `HEIGHT_SHARDED` / `WIDTH_SHARDED` / `BLOCK_SHARDED` are placement unlocks of the *same* scheme — see Dataflow Strategy. |

### gamma (optional)

| Property | Requirement |
|----------|-------------|
| Shape | last dim `== input.shape[-1]`; leading dims are 1 |
| Dtype | `bfloat16`, `float32`; may differ from the input dtype |
| Layout | `TILE_LAYOUT` or `ROW_MAJOR_LAYOUT` |
| Absent | `gamma_mode = "no_gamma"`; `gamma_dtype` / `gamma_layout` canonicalize to the `"none"` sentinel, which is always legal |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to the input |
| Dtype | input dtype |
| Layout | input layout |
| Memory | INTERLEAVED, same buffer type as the input |

---

## Blocking Model

### Axes

The op has exactly two axes after the leading dims are folded: a **row** axis (all dims but the last, which
the math treats identically and which are contiguous in both layouts) and the **hidden** axis (the last dim,
the one the RMS reduces over).

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `row` — flattened leading dims, in 32-row tile-rows (`tensor_row_tiles`) | **independent**: each row's RMS is a function of that row alone; no result spans this axis | `block_row_tiles` (extent of one block along `row`) | coarsest that fits: `clamp(floor((l1_cb_budget − fixed_bytes) / per_row_bytes), 1, core_row_tiles)`, further clamped by `MAX_GATHER_TILES / w_group_size` | one host int, RT arg `block_row_tiles` + `num_blocks_this_core` + `last_block_row_tiles`; every CB page count and loop bound derives from it | Spread across the grid in Phase 0: `num_row_groups = (grid_x/w_group_cols)·(grid_y/w_group_rows)` disjoint row-groups | knob-turn |
| `hidden` — the reduced last dim, in 32-column tiles (`tensor_w_tiles`) | **dependent**: `Σ x²` spans the whole axis, so a result depends on every block along it. Within a core it is a sequential DEST accumulation; across cores it needs a NoC combine. | `core_w_tiles` — the block always spans the core's **entire** hidden slice (`block_w_tiles == core_w_tiles`), so one block ⇒ one combine round | `ceil(tensor_w_tiles / w_group_size)` | one host int, CT arg `core_w_tiles` (compile-time because `tilize`/`untilize` take `width_in_tiles` as a template param) | **Split across cores in Phase 0** — `w_group_size = w_group_cols · w_group_rows` cores per reduction group; partials combined by gather-to-root + mcast-back | knob-turn (the combine exists; only `w_group_size` changes) |
| `gamma` operand along `hidden` | **reuse-shared by construction of the split**: `gamma` does not vary along `row`, so every row-group re-reads the identical bytes. It *is* partitioned along `hidden`, so the W-split already divides its per-core cost. | (no separate extent — follows `core_w_tiles`) | `core_w_tiles` tiles, read **once per core for the whole kernel** and held resident across every block | CT arg `core_w_tiles` | one full copy per row-group | scheme-change (broadcast `gamma` from an injector core to the row-groups via `mcast_pipe` — scheme lamp G1) |

**Naming.** `tensor_row_tiles` / `tensor_w_tiles` (tensor scope), `core_row_tiles` / `core_w_tiles` (this core's
share), `block_row_tiles` / `block_w_tiles` (block scope), `num_blocks_this_core` / `block_idx` /
`last_block_row_tiles`.

**Alignment-aware tile geometry (used everywhere, not only at the boundary):**

```
tensor_w_tiles = ceil(W / 32);   partial_w = W % 32          # 0 => tile-aligned
TILE input:       tensor_row_tiles = prod(shape[:-2]) * ceil(shape[-2] / 32)   # per-image tile padding
ROW_MAJOR input:  tensor_row_tiles = ceil(prod(shape[:-1]) / 32)               # sticks are contiguous, no per-image pad
```

The two formulas genuinely differ — a TILE tensor pads **each image's** H to 32 independently, a ROW_MAJOR one
does not. `floor` / `//` appears nowhere.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `input_cb_depth` | **2** | The reader prefetches block `b+1` while compute runs block `b`. Without it the schedule fully serializes DRAM read against compute, because `sumsq_block` waits upfront for the whole block and nothing pops until `scale_block`. `double_buffer` measures 2.78× for exactly this (depth 1 → depth 2 on a DRAM-fed stream). |
| `cb_output_tiles` | `output_cb_depth` | **2** | Lets the writer drain tile-row `r` while compute produces `r+1`; also gives the writer ≥ 4–8 tiles to batch behind one `noc_async_write_barrier` (`double_buffer`'s measured plateau). |
| `cb_input_rm`, `cb_output_rm` | `rm_cb_depth` | **2** | Overlaps stick reads/writes with `tilize` / `untilize` on the ROW_MAJOR path. |
| `cb_normed` | (1) | 1 | Intermediate between two sequential compute helpers — both own all three TRISCs, so depth > 1 buys no overlap (`ttnn-cb-memory-fundamentals.md`). |
| `cb_scaler`, `cb_wmask`, `cb_zero_tile` | (1) | 1 | Constants, pushed once at kernel start, never popped. |
| all `cb_stat_*`, `cb_rstd*` | (1) | 1 | One combine round per block; the round is a group-wide barrier, so a deeper stat pipeline cannot overlap anything. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 residency of the block working set | `block_row_tiles` | `block_row_tiles = clamp(floor((l1_cb_budget − fixed_bytes) / per_row_bytes), 1, core_row_tiles)`; if the result would be 0, `w_group_size` is raised until it is ≥ 1 | CB creation exceeds L1 → allocation failure at program build |
| `cb_stat_gather` capacity (`MAX_GATHER_TILES = 64` fp32 tiles = 256 KiB) | `block_row_tiles × w_group_size` | `block_row_tiles ≤ MAX_GATHER_TILES / w_group_size` | a wide reduction group silently eats a sixth of L1 in stat storage and the block CBs no longer fit |
| `tilize` / `untilize` take `width_in_tiles` as a **template** parameter (`tilize_helpers.hpp:197`, `untilize_helpers.hpp:154`) | `core_w_tiles` must be a compile-time constant per core | `core_w_tiles` is a CT arg; the ragged remainder of `tensor_w_tiles % w_group_size` is handled by **two kernel core-ranges** (`core_group_1` with `ceil`, `core_group_2` with `floor`), each with its own CT block; CBs are sized to `ceil` on both | a runtime width silently tilizes the wrong number of tiles → wrong data, no error |
| `read_sticks_for_tilize` in TILE granularity asserts `width_in_tiles <= cb_capacity` (`tilize_helpers_dataflow.inl:105-108`) | `cb_input_rm` page count | `rm_cb_depth × core_w_tiles ≥ core_w_tiles` (holds for depth ≥ 1) | reader deadlocks in `cb_reserve_back` |
| Multicast rectangle: a `w`-group must be an axis-aligned rectangle on the compute grid (`Mcast2D`, `host/mcast_host.hpp:450`) | `w_group_size` | `w_group_cols` divides `grid_x`, `w_group_rows` divides `grid_y` | the mcast reaches cores outside the reduction group and overwrites their `cb_rstd` → wrong results on unrelated rows |
| `w_group_size ≤ tensor_w_tiles` | `w_group_size` | clamp in the selection function | a core owning zero hidden tiles never pushes its partial; the root spins forever in `wait_min` |
| Reduce scaler CB format must be `Float16_b` or `Float32` (`reduce_helpers_dataflow.inl:185-187`) | `cb_scaler` page format | declared bfloat16 | compile-time `static_assert` |
| `ReducePartialScaler` is rejected for `REDUCE_SCALAR` (`reduce_helpers_compute.hpp:241-243`) | reduce dim choice | we use `REDUCE_ROW` throughout | runtime assert |
| DEST capacity with `fp32_dest_acc_en=True` → `DEST_AUTO_LIMIT` = 4 (half-sync) / 8 (full-sync), `dest_helpers.hpp:103` | eltwise chain `block_size`; reduce COL chunking | the helpers clamp automatically; the design never hardcodes 8 | over-subscribed DEST → corrupted tiles |

Note the last row explicitly: **DEST does not bound a block here.** It bounds the helpers' internal walk, which
they size from `DEST_AUTO_LIMIT` themselves.

### Regimes

| Regime | Status | Predicate (exact, host-checkable) | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------------------------------|-------|---------------------------|--------------------------|
| **R1 `row_parallel_resident`** | **Phase 0** | `w_group_size == 1` (chosen when `tensor_row_tiles ≥ num_cores` **and** one core's whole `tensor_w_tiles` block fits L1) | `(block_row_tiles, tensor_w_tiles)` | Named boundary = DRAM. Minimum is *input once + output once + gamma once*. Actual: input **1×**, output **1×**, gamma **num_cores×** (every core needs the whole vector; the split does not partition it). Above minimum only by `(num_cores−1)·W·gamma_elem_bytes`. | Amortizes, per block: 4 LLK init sequences (accumulate-square, reduce, scale-mul, gamma-mul) + 3 dtype reconfigs (bf16 in → fp32 stat → bf16 out) + the pipeline fill/drain of `cb_input_tiles`. Intended frequency: **once per block**, i.e. `ceil(core_row_tiles / block_row_tiles)` times per core. No cross-core term. |
| **R2 `w_split_combine`** | **Phase 0** | `w_group_size > 1` (chosen when `tensor_row_tiles < num_cores`, i.e. the independent axis under-fills the grid, **or** when `tensor_w_tiles` does not fit one core's L1) | `(block_row_tiles, core_w_tiles)` with `core_w_tiles = ceil(tensor_w_tiles / w_group_size)` | Input **1×** DRAM, output **1×**, gamma **num_row_groups×** (each group's members read disjoint slices, so one group reads gamma exactly once — *cheaper* than R1's `num_cores×`). Adds cross-core: per block, `(w_group_size−1)·block_row_tiles` fp32 tiles unicast to the root, plus `block_row_tiles` fp32 tiles multicast to `w_group_size` receivers. Minimum is **unreachable** for gamma: with `num_row_groups > 1` the vector is structurally replicated. | Everything R1 amortizes, **plus one whole cross-core combine round per block** — a gather-barrier + a root reduce + a multicast, measured at ~1.3–1.5 µs for a 1-tile payload on a 16-core group (`tensix_all_reduce`, Blackhole, 1 tile/core). Intended frequency: **once per block**. This is the dominant reason to take a coarse `block_row_tiles`. |
| **R3 `w_stream_two_pass`** | **lamped** | `ceil(tensor_w_tiles / num_cores) · tile_bytes` exceeds the residency budget even at `w_group_size = num_cores` — i.e. `W > 32 · num_cores · max_core_w_tiles` (≈ `W > 400 000` for bf16 on a 64-core grid) | `(1, w_chunk_tiles)` streaming, `cb_input_tiles` constant-sized | Input **2×** DRAM (a second read for the apply pass), output 1×. Strictly above the minimum by one whole tensor. | Same fixed costs as R2 plus a per-chunk `Accumulate` reload (`reduce_helpers_compute.hpp:328`). |
| **R4 `row_parallel_only`** (never split the hidden axis) | **rejected** | — | `(block_row_tiles, tensor_w_tiles)` always | Input 1× only while it fits; otherwise degenerates to R3's 2×. | **Rejected:** strands every decode shape on a single core (`tensor_row_tiles == 1` for `(1,1,32,W)`, which is 12 of the 13 perf cases and 3 of the `_WIDE` loose cases), and cannot hold `W ≥ 16384` resident on one core. The feature spec's `_WIDE` block exists specifically to fail this regime. |
| **R5 `all_gather_combine`** (rotating mcast, every core reduces) | **rejected** | — | same as R2 | Cross-core traffic `w_group_size ×` R2's, in `w_group_size` rounds. | **Rejected:** O(group) rounds. Measured 8938 ns vs 6179 ns for gather-to-root on an 8-core group (`tensix_all_reduce`), and the gap grows linearly with `w_group_size`, which reaches 64 in the decode regime. |
| **R6 `ring_combine`** | **rejected** | — | same as R2 | `w_group_size − 1` hops. | **Rejected:** 4.6–6.5× slower than root-gather, and a 2-row serpentine costs ~47 µs on *either* NoC because the ring order fights NoC routing (`tensix_all_reduce_ring_transport`). |
| **R7 `two_phase_reduce_mcast`** | **rejected for Phase 0** | — | same as R2 | 2 rounds, tile-index-parallel workers. | **Rejected:** it degenerates to a single worker at `min(num_tiles, group−1)` and our partial payload is `block_row_tiles` tiles, frequently 1. Measured *worst* of the three at 1 tile/core (1981 ns vs 1377 ns) and 15–28% noise under grid contention. |

**Regime-selection function** (exact, host, deterministic — regime-pinned tests are required because R1 vs. R2
depends on the device grid):

```
grid_x, grid_y  = device.compute_with_storage_grid_size()
candidates = [ (gc, gr) for gc in divisors(grid_x) for gr in divisors(grid_y) ]
for (gc, gr) in candidates:
    G           = gc * gr                                  # w_group_size
    num_groups  = (grid_x // gc) * (grid_y // gr)
    if G > tensor_w_tiles:            continue             # mechanism cap
    C           = ceil(tensor_w_tiles / G)                 # core_w_tiles (max over the group)
    R           = max_block_row_tiles(C, G)                # L1 solve below; 0 == does not fit
    if R == 0:                        continue
    active      = min(tensor_row_tiles, num_groups) * G
    score       = (active, -G, R)                          # occupancy, then fewest combine partners, then coarsest block
pick argmax(score);  regime = R1 if G == 1 else R2
if no candidate survives: regime = R3  (lamped -> validate() raises today)
```

`max_block_row_tiles(C, G)` is the closed-form L1 solve in `l1_ledger.md`; it is a *single* expression over the
declared symbols, not a search.

### Traffic ranking

Qualitative, per memory tier, for a tensor of `N` bytes with a `W`-element gamma. `num_cores` is the grid size,
`num_row_groups = num_cores / w_group_size`.

| Rank | Candidate split | DRAM bytes | Cross-core bytes | Verdict |
|------|-----------------|-----------|------------------|---------|
| 1 | **Hybrid: `row` across groups × `hidden` within a group (R1/R2)** | `2N + num_row_groups · W · gamma_bytes`. Input crosses **once** because the block stays resident between the reduce and the apply pass. | `(G−1)·R·4096` gather + `R·4096` mcast per block | **Chosen.** In the R1 corner (`G = 1`) it *is* the pure-row split; in the R2 corner it is strictly cheaper in DRAM bytes than pure-row, because the hidden split both partitions gamma and is what makes the input resident. |
| 2 | Pure `row` split, hidden never split (R4) | `2N + num_cores · W · gamma_bytes` while `W` fits resident; `3N + …` once it does not | 0 | Rejected. Same or more DRAM bytes, and it cannot fill the grid when `tensor_row_tiles < num_cores`. |
| 3 | Pure `hidden` split, every core takes all rows (`num_row_groups = 1`) | `2N + W · gamma_bytes` — the **cheapest possible gamma traffic** | `(num_cores−1)·R·4096` gather + mcast, paid `tensor_row_tiles / R` times | Rejected as a *default*: it pays a combine round for every tile-row of the whole tensor, and the gamma saving (`(num_row_groups−1)·W·2` bytes ≈ 0.9 MB on the widest prefill case, against 234 MB of tensor traffic) is ~0.4% of DRAM traffic. It is reachable by forcing `w_group_size = num_cores`; kept as perf lamp P3. |
| 4 | Two-pass streaming (R3) | `3N + …` — one extra full read of the input | same as 1 | Rejected except as the out-of-budget fallback. The extra `N` bytes is by far the largest single term in the ranking. |

**Operand-reuse check, run mechanically over every (operand, chosen-split) pair:**

| Operand | Varies along `row` (the cross-group split)? | Varies along `hidden` (the in-group split)? | Consequence |
|---------|---------------------------------------------|---------------------------------------------|-------------|
| `input_tensor` | yes | yes | fully partitioned, no reuse |
| `gamma` | **no** | yes | Every row-group re-reads the identical `W` bytes ⇒ reuse-shared **by construction of the split**. Recorded as scheme lamp **G1** (broadcast gamma from one injector per grid row via `mcast_pipe`, the `shared_input_reuse` pattern, measured 1.71× there). Not built in Phase 0 because the traffic it saves is ~0.4% of the total (row 3 above) — a *measured* proportion, not a difficulty dodge. |
| `epsilon`, `1/W` | no | no | scalars in compile-time args |

The traffic ranking and the occupancy question are kept separate: occupancy is why R2 gets *considered*
(the independent axis under-fills the grid at `tensor_row_tiles = 1`), but the choice between R1 and R2 is
settled by the table above plus the residency constraint — never by core count alone.

### Block schedule

The logical schedule. Reader (NCRISC/NoC0), compute (3 TRISCs) and writer (BRISC/NoC1) are separate
asynchronous kernels; adjacent blocks pipeline through `cb_input_tiles` (depth 2) and `cb_output_tiles`.

```cpp
// once per kernel, before the block loop
prepare_constants();          // cb_scaler, cb_wmask, cb_zero_tile
load_gamma_slice();           // cb_gamma_tiles: core_w_tiles tiles, resident for every block

for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);           // reader
    sumsq_block(block_idx);          // compute
    reduce_stat_block(block_idx);    // compute
    combine_stat_block(block_idx);   // writer + compute, cross-core
    scale_block(block_idx);          // compute
    gamma_block(block_idx);          // compute   (elided when gamma is absent)
    store_block(block_idx);          // writer
}
```

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------|-------------|--------------------|-------------------------------|
| `prepare_constants` | — | `cb_scaler` (1 bf16 tile, value 1.0), `cb_wmask` (1 bf16 tile, 1.0 in columns `[0, partial_w)`), `cb_zero_tile` (1 fp32 zero tile) | **once per kernel.** All three are waited but never popped. |
| `load_gamma_slice` | `(1, core_w_tiles)` | `cb_gamma_tiles`, for the entire kernel | **once per kernel.** For ROW_MAJOR gamma this includes one `tilize<core_w_tiles>` of a zero-padded single stick, producing the row-0-valid tile form that TILE gamma already has. |
| `load_block` | `(block_row_tiles, core_w_tiles)` | — | Per block. Reads whole tiles (TILE) or sticks (ROW_MAJOR) for this core's `(row-block × hidden-slice)` rectangle. Intended: **one `noc_async_read_barrier` per 4–8 tiles**, not per tile (`double_buffer`: 2.78× for exactly this). |
| `sumsq_block` | `(block_row_tiles, core_w_tiles)` | `cb_input_tiles` — waited **upfront and not popped**, because `scale_block` reads it again | Per block: one init + one reconfig. Computes `Σ_c x[r,c]²` **elementwise across tiles** into a persistent fp32 DEST accumulator, packing one tile per tile-row. This is the measured-fastest reduction shape (`eltwise_l1_vs_dest_accumulate`: DEST accumulation 10.59× over an L1 read-modify-write; `row_reduce_accumulate`: 2.93× over one wide reduce at 32 tiles) **and** it removes a block-sized `x²` scratch buffer entirely. When this core owns the tensor's last hidden tile and `partial_w != 0`, that tile is excluded from the bulk and folded in by `mask_tail_block` (below). |
| `mask_tail_block` | `(block_row_tiles, 1)` | `cb_input_tiles` | Per block, **only when `owns_last_w_tile && partial_w != 0`**. Two tiny passes: `x_last · wmask` (row-broadcast 0/1 mask) into `cb_tail_masked`, then `cb_tail_masked²` packed into `cb_stat_sq` with `L1Accumulation::Enabled`. Because `mask ∈ {0,1}`, `(x·mask)² == x²·mask` exactly. **This is what makes the RMS denominator count only valid elements** — the `pad_poison` cases test precisely this. |
| `reduce_stat_block` | `(block_row_tiles, 1)` | `cb_stat_sq` | Per block. `reduce<SUM, REDUCE_ROW>` folds the 32 within-tile columns of each accumulated tile into a column-0-valid tile → `cb_stat_partial` (fp32). Scaler is a plain 1.0; **no partial scaler is needed here**, because the hidden padding was already zeroed by `mask_tail_block`. |
| `combine_stat_block` | `(block_row_tiles, w_group_size)` | `cb_rstd` (the mcast landing buffer, at an identical L1 address on every group member) | Per block, **one cross-core round**: (a) every member's writer `noc_async_write`s its `block_row_tiles` partial tiles into the root's `cb_stat_gather` at tile index `r · w_group_size + my_slot`, barriers, and bumps the gather semaphore; (b) the root's writer `wait_min((block_idx+1)·(w_group_size−1))` then pushes `block_row_tiles · w_group_size` pages; (c) the root's compute sums across slots per tile-row and finalizes `rsqrt(sum/W + eps)` (the ×`1/W` uses the **true, unpadded** `W`); (d) the root's writer multicasts `block_row_tiles` fp32 tiles to the group rectangle with `src != dst` so the root loops back into its own `cb_rstd`; every member's writer pushes `cb_rstd` after `receive()`. At `w_group_size == 1` the rectangle is degenerate and `SenderPipe` performs a local copy (`mcast_pipe.inl:71-76`) — the same code path, no branch in the schedule. |
| `scale_block` | `(block_row_tiles, core_w_tiles)` | — (pops `cb_input_tiles` here, releasing the block) | Per block. `x · rstd` with `BroadcastDim::Col` and `OperandKind::Col` on the stat operand → `cb_normed`. |
| `gamma_block` | `(block_row_tiles, core_w_tiles)` | `cb_gamma_tiles` | Per block. `normed · gamma` with `BroadcastDim::Row` and `OperandKind::Row` → `cb_output_tiles`. **Elided at compile time when gamma is absent**, in which case `scale_block` writes `cb_output_tiles` directly and `cb_normed` is not allocated. |
| `untilize_out_block` | `(block_row_tiles, core_w_tiles)` | — | Per block, **ROW_MAJOR output only**. `untilize<core_w_tiles>(block_row_tiles)` → `cb_output_rm`. |
| `store_block` | `(block_row_tiles, core_w_tiles)` | — | Per block. Whole tiles, or `write_sticks_after_untilize` with `byte_offset_within_page = w_start_tile·32·elem_bytes` for this core's hidden slice. Intended: one write barrier per 4–8 tiles. |

The ragged last block passes a smaller **runtime** `block_row_tiles` (`last_block_row_tiles`) into the same
operations; there is no separate tail code path and no separate accounting.

### Lamps

**Scheme lamps** — scheme-changes Phase 0 deliberately leaves room for:

| Lamp | Scheme-change it leaves room for | How the structure keeps it reachable |
|------|----------------------------------|--------------------------------------|
| **G1 — gamma broadcast** | `gamma` is reuse-shared across row-groups (operand-reuse check above). Read the slice once on one injector core per grid row and `mcast_pipe`-broadcast it to that row's groups instead of `num_row_groups` DRAM reads. | `cb_gamma_tiles` is already a separate CB filled by a single `load_gamma_slice` operation outside the block loop, and the program already carries `Mcast2D` wiring and semaphores. Turning G1 on replaces the body of `load_gamma_slice` and adds one more `Mcast*` object; nothing in the block schedule moves. **Positive reason not to build it now:** measured at ~0.4% of total DRAM traffic (traffic ranking row 3) — the win is bounded and quantified, not merely awkward. |
| **S1 — physical shard placement** | `HEIGHT_SHARDED` / `WIDTH_SHARDED` / `BLOCK_SHARDED` inputs. | The logical shard is already the Phase 0 scheme (see Dataflow Strategy). Adding the physical placement is a CB-construction change only: `ttnn.cb_descriptor_from_sharded_tensor(cb_input_tiles, input_tensor)` pins `cb_input_tiles` zero-copy over the resident L1 shard and the reader's DRAM path is dropped. **This is a support-rectangle growth (placement), not a new algorithm** — which is exactly why the cross-core combine had to be in Phase 0. |
| **R3 — streaming two-pass** | Hidden dims beyond the residency budget of the whole grid. | `reduce_helpers_compute.hpp`'s cross-call `Accumulate` path (`hpp:328`) and the reader's `byte_offset_within_page` chunk hook (`tilize_helpers_dataflow.hpp:80-85`) are the two mechanisms; `core_w_tiles` is already the only extent the CBs are sized against, so a `w_chunk_tiles` sub-knob slots under it. **Positive reason not to build it now:** unreachable in the declared universe — the widest shape anywhere in `feature_spec.py` is `W = 32768` (`tensor_w_tiles = 1024`), against a grid capacity of `64 × ~110 = 7040` hidden tiles. `validate()` raises with an explicit message beyond that. |

**Perf lamps** — defaults that may be wrong *here*:

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **P1 — overlap** | `input_cb_depth = 2` at a coarse `block_row_tiles` means two whole blocks of `x` are resident. On the widest prefill shapes that is most of the L1 budget, and it may be better spent on a deeper, finer pipeline than on one fat block. | `block_row_tiles / 2` with `input_cb_depth = 3`, and `block_row_tiles = 1` with `input_cb_depth = 4`. The trade is combine rounds (`~1.3 µs` each) against read/compute overlap. |
| **P2 — grid synchronization** | Maximum occupancy is the default first step, but at `tensor_row_tiles = 1` the selection function pushes `w_group_size` to the whole grid, leaving as little as 3–4 hidden tiles of real work per core against a full gather+multicast round. The measured-fastest geometries in `feature_spec.py` use **28–32 cores, not 64**, for exactly these shapes. | Cap `w_group_size` at 8 / 16 / 32 and compare against the full-grid choice on the four decode perf cases. The selection function's `score` tuple is the single place to change. |
| **P3 — one row-group** | The selection function prefers the smallest `w_group_size` that fills the grid; forcing `num_row_groups = 1` instead would cut gamma DRAM traffic to its true minimum. | Force `w_group_size = num_cores` on the wide prefill cases and compare; ranking row 3 predicts ~0.4% and a large increase in combine rounds, so this is expected to lose — worth one measurement to close it. |
| **P4 — fuse `scale_block` + `gamma_block`** | The two broadcast multiplies cost a pack + an unpack of the whole block through `cb_normed` (≈ 2 of the ~6 engine operations per input tile) and a block-sized CB. `DestReuseBinary` (`eltwise_chain.hpp:519`) would fuse them into one DEST window. | Measure `BinaryFpu<x, rstd, Mul, Col> → DestReuseBinary<gamma, Mul> → PackTile`. **Not the default** because dest-reuse routes DEST through a Src register (bf16 on Wormhole), which is a real precision loss for `float32` input, and `compute_fusion` measured FPU-consumer dest-reuse at 0.94× / 0.82× isolated. Verify PCC on the fp32 cells before adopting. |
| **P5 — reduce library vs. fast path at tiny widths** | `reduce_stat_block` reduces exactly **one** tile per tile-row, which is below every crossover in `reduce_block` (row crossover ≥ 4 reduced tiles) — the library path is the right one there, and it is what `reduce<>` picks. But if a later refinement merges `sumsq_block` and `reduce_stat_block`, the crossover moves. | Only if the two operations are merged: compare `reduce<>` against the inline pairwise fast path at the merged width. |
| **P6 — `cb_stat_*` page format** | The stat path is fp32 throughout (`row_reduce_accumulate`: bf16 *accumulation* error grows with width and `x²` is the all-positive, monotonic-swamping case). fp32 doubles `cb_stat_gather`, which is the largest stat buffer at wide `w_group_size`. | bf16 `cb_stat_gather` only (partials still summed in fp32 DEST), measured against PCC 0.9995 on the perf cells. |

---

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → `cb_input_rm` (RM input) | row-major sticks, page = tile-sized | `dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm, TilizeGranularity::TILE>(acc, rows, slice_bytes, start_page, byte_offset_within_page)` | `start_page` = this core's first stick; `byte_offset_within_page = w_start_tile · 32 · elem_bytes` selects this core's hidden slice. The reader zero-fills the stale rows of a ragged last 32-row block so no uninitialized L1 enters the FPU. |
| `cb_input_rm` → `cb_input_tiles` | tiles | `compute_kernel_lib::tilize<core_w_tiles, cb_input_rm, cb_input_tiles>(block_row_tiles)` | RM path only. |
| DRAM → `cb_input_tiles` (TILE input) | tiles | `TensorAccessor` + `noc_async_read_tile`, batched 4–8 per barrier | Tile id = `(row_tile_global · tensor_w_tiles) + w_tile_global`. |
| DRAM → `cb_gamma_tiles` | tiles, row-0 valid | TILE gamma: direct tile read of `core_w_tiles` tiles. RM gamma: one stick into a zeroed `cb_gamma_rm`, then `tilize<core_w_tiles>(1)`. | Both layouts converge on the same CB contract: `core_w_tiles` tiles with the scale factors in row 0, which is exactly what `BroadcastDim::Row` consumes. Once per kernel. |
| Tensix → Tensix, partials | fp32 tiles | `noc_async_write` into `cb_stat_gather` at byte offset `(r · w_group_size + my_slot) · stat_tile_bytes` on the root, then `noc_semaphore_inc` on the gather semaphore; root does `wait_min((block_idx+1)·(w_group_size−1))` | The tile-row-major slot layout makes each output tile's `w_group_size` contributions **contiguous**, which is what lets the combine be a single `eltwise_chain` over `grid(block_row_tiles, w_group_size)` instead of a strided hand-rolled walk. Costs `block_row_tiles` separate unicasts instead of one — the payload is ≤ 64 tiles, and `tensix_all_reduce_ring_transport` measures semaphore/handshake cost at 215–410 ns, i.e. the route, not the count, dominates. |
| Tensix → Tensix, `rstd` | fp32 tiles | `mcast_pipe` `SenderPipe::send(cb_rstd_send_addr, cb_rstd_addr, bytes)` on the root; `ReceiverPipe::receive()` on the members. Host wiring: one `ttnn.Mcast2D(device, group_rect, group_root, ttnn.McastConfig(sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED]))` per group, contributing only per-core runtime args; the three `ttnn.SemaphoreDescriptor`s are declared once over the full active grid. | `src != dst` with the sender inside the rectangle selects INCLUDE_SRC loopback (`mcast_pipe.inl:84`), so the root receives its own copy and every member's `cb_rstd` is filled by the **writer** — one producer, one consumer, no root special case in the CB table. |
| `cb_output_tiles` → DRAM (TILE output) | tiles | `TensorAccessor` + `noc_async_write_tile`, batched | NoC1 (writer default), reads stay on NoC0 (`noc_placement`: reads·NoC0 / writes·NoC1 is 4.3–4.8× better than the reverse). |
| `cb_output_tiles` → `cb_output_rm` → DRAM (RM output) | tiles → sticks | `compute_kernel_lib::untilize<core_w_tiles, cb_output_tiles, cb_output_rm>(block_row_tiles)` then `dataflow_kernel_lib::write_sticks_after_untilize<cb_output_rm>(acc, rows, slice_bytes, start_page, byte_offset_within_page)` | The helper skips L1 row padding and writes only the valid rows of a ragged last block. |

**Sharded placements (lamp S1), classified against *this* op's axis characters — not against the flavor's name:**

| `memory_layout` | Which axis the shard cuts | Character of that axis here | Unlock class | What changes |
|-----------------|---------------------------|-----------------------------|--------------|--------------|
| `HEIGHT_SHARDED` | the `row` axis | **independent** | knob-turn | `cb_input_tiles` becomes `ttnn.cb_descriptor_from_sharded_tensor(...)`, zero-copy over the core's own L1 shard — **no NoC read**. `w_group_size = 1`, the reduction stays local, the combine degenerates. `block_row_tiles` defaults to the **whole resident shard** (one block), sub-chunked only if the shard exceeds the working-set budget. |
| `WIDTH_SHARDED` | the `hidden` axis | **dependent** | knob-turn (**because Phase 0 already built the combine**) | The shard spec *is* the `w_group` geometry: `w_group_size` = the shard grid's core count, `core_w_tiles` = the shard width in tiles, both read off the spec instead of chosen. `cb_input_tiles` is zero-copy over the shard; the gather + mcast is unchanged. |
| `BLOCK_SHARDED` | both | independent × dependent | knob-turn | Both extents come from the shard spec; the `w`-group is one grid row of the shard grid, which is already the rectangle `Mcast2D` wants. |

This table is the whole payoff of building R2 in Phase 0: because the cross-core combine exists, *every*
sharded placement is additive support growth rather than a scheme-change. A sharded input is consumed **in
place through a CB backed on the sharded buffer** — re-reading it through a `TensorAccessor` would fetch over
the NoC data the core already holds, and is explicitly not the design.

---

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one **block** = a `(block_row_tiles × core_w_tiles)` tile rectangle |
| Grid | `device.compute_with_storage_grid_size()`, partitioned into `num_row_groups` axis-aligned rectangles of `w_group_cols × w_group_rows` cores by the selection function above. Cores laid out **row-wise** (`noc_placement`: `row_wise=False` is the column trap, 2.91×). The grid extent is a program parameter, never an inlined constant. |
| Per-core work | `core_row_tiles = ceil(tensor_row_tiles / min(tensor_row_tiles, num_row_groups))` tile-rows × `core_w_tiles` hidden tiles, processed as `num_blocks_this_core = ceil(core_row_tiles / block_row_tiles)` blocks |
| Remainder — rows | `ttnn.split_work_to_cores(row_group_corerangeset, tensor_row_tiles, row_wise=True)` over the *row-groups*; every core of a group receives the same row range. The ragged final block passes `last_block_row_tiles` as a runtime arg into the same block operations. |
| Remainder — hidden | `tensor_w_tiles % w_group_size` cores take one extra hidden tile. Because `core_w_tiles` is a **compile-time** template argument of `tilize`/`untilize`, this is expressed as two kernel core-ranges (`core_group_1` = `ceil`, `core_group_2` = `floor`) with separate CT blocks; all CBs are sized to `ceil` on both ranges so L1 addresses stay identical across the group (required for the multicast destination). |
| Remainder — sub-tile | `partial_w = W % 32` is handled numerically by `mask_tail_block`, not by padding. `H % 32` needs no masking: the RMS reduces only along the hidden axis, so a padded row produces a padded output row that is never read back. |

**Regime-pinned tests are required.** R1 vs. R2 is selected from the device grid, so a shape that lands in R2
on a 64-core Wormhole part can land in R1 on a 110-core Blackhole part (or vice versa). The acceptance test
therefore includes shapes that force each regime by construction: `tensor_row_tiles == 1` with a wide hidden
dim can only be R2 on any grid, and a tall/narrow shape (`tensor_row_tiles ≫ num_cores`, `tensor_w_tiles`
small) can only be R1.

---

## Circular Buffers

Page format `in` = input dtype, `γ` = gamma dtype, `f32` = float32, `bf16` = bfloat16.
`R` = `block_row_tiles`, `C` = `core_w_tiles`, `G` = `w_group_size`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_rm` | 0 | `tile_size(in)` | `rm_cb_depth · C` | Live set **spans** `hidden` (one 32-row block needs all `C` tiles before `tilize` can run); **streams over** `row` one tile-row at a time. Depth 2 overlaps stick reads with `tilize`. ROW_MAJOR input only. | in | reader | compute | whole kernel (RM input) |
| `cb_input_tiles` | 1 | `tile_size(in)` | `input_cb_depth · R · C` | Live set **spans both** axes — the whole block must stay resident from `sumsq_block` through `scale_block`, which is what makes the input cross DRAM once. Depth 2 lets the reader prefetch block `b+1`. | in | reader (TILE) / compute-`tilize` (RM) | compute | one block, ×2 in flight |
| `cb_scaler` | 2 | `tile_size(bf16)` | `1` | Constant. `calculate_and_prepare_reduce_scaler<…, SUM, REDUCE_ROW>()` fills 1.0. Spans no axis. bfloat16 is mandated by `reduce_helpers_dataflow.inl:185-187`. | bf16 | reader | compute | whole kernel |
| `cb_wmask` | 3 | `tile_size(bf16)` | `1` | Constant 0/1 column mask for the ragged hidden tile, `prepare_reduce_mask<…, REDUCE_ROW>(partial_w)`. Allocated only when `partial_w != 0 && owns_last_w_tile`. Spans no axis. | bf16 | reader | compute | whole kernel |
| `cb_zero_tile` | 4 | `tile_size(f32)` | `1` | Constant zero tile, the identity operand `B` for the two `DestAccumulation` sums (`BinaryFpu` needs two inputs; `A + 0` accumulates `A`). Spans no axis. | f32 | reader | compute | whole kernel |
| `cb_stat_sq` | 5 | `tile_size(f32)` | `R` | Live set **spans** `row` (one accumulated tile per tile-row of the block); **streams over** `hidden` — the whole hidden slice passes through the DEST accumulator, never through this buffer. | f32 | compute | compute | one block |
| `cb_tail_masked` | 6 | `tile_size(f32)` | `R` | Live set **spans** `row`; **streams over** `hidden` with a window of exactly the one ragged tile. Allocated only when `partial_w != 0 && owns_last_w_tile`. | f32 | compute | compute | one block, `mask_tail_block` only |
| `cb_stat_partial` | 7 | `tile_size(f32)` | `R` | This core's `Σ x²` column tiles. **Spans** `row`, **streams over** `hidden` (fully folded). | f32 | compute | writer | one block |
| `cb_stat_gather` | 8 | `tile_size(f32)` | `R · G` | **Spans** `row` and the *cross-core* extent of `hidden` — the whole point is that all `G` slices of a tile-row are simultaneously live. Slot layout is `r · G + slot`. Capped by `MAX_GATHER_TILES`. | f32 | writer | compute | one block, combine only |
| `cb_stat_sum` | 9 | `tile_size(f32)` | `R` | Combined `Σ x²` before the finalize chain. **Spans** `row`, **streams over** `hidden`. | f32 | compute | compute | one block, combine only |
| `cb_rstd_send` | 10 | `tile_size(f32)` | `R` | Multicast source on the root. **Spans** `row`, **streams over** `hidden`. | f32 | compute | writer | one block, combine only |
| `cb_rstd` | 11 | `tile_size(f32)` | `R` | Multicast destination — must sit at an identical L1 address on every group member, which is why every CB is allocated on the full active core set. **Spans** `row`, **streams over** `hidden`. | f32 | writer | compute | one block, from combine through `scale_block` |
| `cb_gamma_rm` | 12 | `tile_size(γ)` | `C` | ROW_MAJOR gamma only: one zero-padded stick block, tilized once. **Spans** `hidden`, **streams over** `row` with a window of 0 (gamma has no row extent). | γ | reader | compute | until `load_gamma_slice` completes |
| `cb_gamma_tiles` | 13 | `tile_size(γ)` | `C` | **Spans** `hidden`; **streams over** `row` with a window of 0 — one copy serves every block. Waited upfront, never popped. Not allocated when gamma is absent. | γ | reader (TILE) / compute-`tilize` (RM) | compute | whole kernel |
| `cb_normed` | 14 | `tile_size(in)` | `R · C` | Intermediate between two sequential FPU broadcast multiplies. Live set **spans both** axes: the two helpers own all three TRISCs, so the first must complete the whole block before the second starts. Not allocated when gamma is absent (`scale_block` then writes `cb_output_tiles` directly). | in | compute | compute | one block, `scale_block` → `gamma_block` |
| `cb_output_tiles` | 16 | `tile_size(in)` | `output_cb_depth · C` | Live set **spans** `hidden` (a whole tile-row, so `untilize` and the writer's 4–8-tile batching both have a full quantum); **streams over** `row`. Depth 2 overlaps compute with the drain. | in | compute | writer (TILE out) / compute-`untilize` (RM out) | one block |
| `cb_output_rm` | 17 | `tile_size(in)` | `rm_cb_depth · C` | ROW_MAJOR output only. **Spans** `hidden`, **streams over** `row`. `untilize` always emits tile-sized pages. | in | compute | writer | one block (RM output) |

Every `Num Pages` cell is an expression in `R`, `C`, `G` and the depth knobs. No cell is a literal standing in
for a block extent, and no cell is an unbounded op dimension: `C` is bounded by the residency solve and `R·G`
by `MAX_GATHER_TILES`. Full capacity-vs-live-set accounting, sharing decisions and the symbol table are in
`l1_ledger.md`.

### CB synchronization contract (producer push count == consumer wait count)

| CB | Pushed per block by | Waited / popped per block by | Balance |
|----|---------------------|------------------------------|---------|
| `cb_input_rm` | reader, `C` per 32-row block × `R` | `tilize`, `C` per block × `R` | `R·C` = `R·C` |
| `cb_input_tiles` | reader / `tilize`, `R·C` | `sumsq_block` waits `R·C` **without popping**; `mask_tail_block` is caller-managed (`None, None`) at a strided offset; `scale_block` waits `R·C` and pops `R·C` **once, at the end** | pushed `R·C`, popped `R·C` — the two waits are re-waits of the same resident pages, which is legal precisely because only one pop exists |
| `cb_scaler`, `cb_wmask`, `cb_zero_tile` | reader, once at kernel start | waited every use, **never popped** (matches `reduce()`'s scaler contract, `reduce_helpers_compute.inl:906`) | push 1, pop 0 — intentional and permanent |
| `cb_stat_sq` | `sumsq_block` `R` (one per tile-row, `PushPolicy::PerOuter`), plus `mask_tail_block`'s in-place `L1Accumulation` pack which pushes nothing extra | `reduce_stat_block` waits `R`, pops `R` | `R` = `R` |
| `cb_tail_masked` | `mask_tail_block` pass 1, `R` | `mask_tail_block` pass 2, `R` | `R` = `R` |
| `cb_stat_partial` | `reduce_stat_block`, `R` | writer (gather write), `R` | `R` = `R` |
| `cb_stat_gather` | writer on the root, `R·G` **after** `wait_min((block_idx+1)·(G−1))` | compute (combine sum), `R·G` | `R·G` = `R·G`. Non-root cores neither push nor wait this CB. |
| `cb_stat_sum` | combine sum, `R` | finalize chain, `R` | `R` = `R` |
| `cb_rstd_send` | finalize chain, `R` | writer (multicast source), `R` | `R` = `R` |
| `cb_rstd` | writer, `R` — on the root after the multicast barrier, on members after `receive()` | `scale_block`, `R` | `R` = `R` on **every** member, root included (INCLUDE_SRC loopback) |
| `cb_gamma_tiles` | reader / `tilize`, `C` **once at kernel start** | `gamma_block` waits `C` every block, **never pops** | push `C`, pop 0 — one copy serves `num_blocks_this_core` blocks |
| `cb_normed` | `scale_block`, `R·C` | `gamma_block`, `R·C` | `R·C` = `R·C` |
| `cb_output_tiles` | `scale_block` (no gamma) or `gamma_block`, `R·C` | writer or `untilize`, `R·C` | `R·C` = `R·C` |
| `cb_output_rm` | `untilize`, `C` per tile-row × `R` | writer, same | `R·C` = `R·C` |

The one asymmetry to be deliberate about: `cb_input_tiles` is **waited twice and popped once**. That is the
mechanism that keeps the input to one DRAM crossing, and it is why `sumsq_block` must use
`PopPolicy::None` and `scale_block` `PopPolicy::AtEnd`.

Every Producer/Consumer cell names exactly one kernel. The two places where that constraint actually bit and
shaped the layout are worth naming: (1) `cb_rstd_send` and `cb_rstd` are two CBs rather than one in-place
buffer, because a single `rstd` CB would be produced by compute, read by the writer for the multicast, **and**
read back by compute for `scale_block` — three parties; (2) `gamma_block` writes a separate
`cb_output_tiles` rather than transforming `cb_normed` in place into the writer's buffer, because an in-place
output CB would give the writer and compute two concurrent consumers and a race on the intermediate pages.

---

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0a | `prepare_constants` | — | yes | — | `cb_scaler` (1), `cb_wmask` (1), `cb_zero_tile` (1) | all pushed once, never popped for the kernel's lifetime |
| 0b | `load_gamma_slice` | `(1, C)` | yes (`tilize` on the RM branch) | `cb_gamma_rm` (C, RM branch only) | `cb_gamma_tiles` (C) | pushed once; waited upfront by `gamma_block` every block, never popped |
| 1 | `load_block` | `(R, C)` | yes on RM (`read_sticks_for_tilize`), raw `noc_async_read_tile` on TILE | DRAM | `cb_input_rm` (C per tile-row) / `cb_input_tiles` (R·C) | `cb_input_tiles` holds the full block |
| 1b | `tilize_in_block` | `(R, C)` | yes | `cb_input_rm` (C per 32-row block) | `cb_input_tiles` (R·C) | RM input only |
| 2 | `sumsq_block` | `(R, C_full)` where `C_full = C − (owns_last_w_tile && partial_w != 0)` | yes | `cb_input_tiles` (R·C, waited **upfront**, **not popped**) | `cb_stat_sq` (R) | `cb_input_tiles` still full — required by `scale_block` |
| 3 | `mask_tail_block` | `(R, 1)` | yes | `cb_input_tiles` (strided at tile `C−1 + r·C`, caller-managed), `cb_wmask` (1) | `cb_tail_masked` (R) → accumulated into `cb_stat_sq` (R) | only when `owns_last_w_tile && partial_w != 0` |
| 4 | `reduce_stat_block` | `(R, 1)` | yes | `cb_stat_sq` (R), `cb_scaler` (1, not popped) | `cb_stat_partial` (R) | `cb_stat_sq` popped |
| 5 | `combine_stat_block` | `(R, G)` | partly (see API Mapping) | `cb_stat_partial` (R, writer-consumed) → `cb_stat_gather` (R·G) → `cb_stat_sum` (R) | `cb_rstd_send` (R) → `cb_rstd` (R) | `cb_rstd` filled on every group member at an identical L1 address |
| 6 | `scale_block` | `(R, C)` | yes | `cb_input_tiles` (R·C, popped at end), `cb_rstd` (R, `OperandKind::Col`, popped at end) | `cb_normed` (R·C), or `cb_output_tiles` when gamma is absent | block released; the reader may now fill the next block |
| 7 | `gamma_block` | `(R, C)` | yes | `cb_normed` (R·C), `cb_gamma_tiles` (C, `OperandKind::Row`, not popped) | `cb_output_tiles` (R·C streamed through `output_cb_depth · C` pages) | elided at compile time when gamma is absent |
| 8 | `untilize_out_block` | `(R, C)` | yes | `cb_output_tiles` (C per tile-row) | `cb_output_rm` (C per tile-row) | RM output only |
| 9 | `store_block` | `(R, C)` | yes on RM (`write_sticks_after_untilize`), raw `noc_async_write_tile` on TILE | `cb_output_tiles` / `cb_output_rm` | DRAM | — |

---

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `prepare_constants` (scaler) | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:98` | `<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()` — the **pool-type-aware overload**, as required | — | `cb_scaler` (1 page) | none |
| `prepare_constants` (mask) | helper | `dataflow_kernel_lib::prepare_reduce_mask` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:74` | `<cb_wmask, ReduceDim::REDUCE_ROW>(partial_w)` | — | `cb_wmask` (1 page) | `partial_w` (shape-derived, not a block knob) |
| `prepare_constants` (zero) | helper | `compute_kernel_lib::` fill via `eltwise_fill.hpp`, or a dataflow memset | `ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp` | one zero fp32 tile | — | `cb_zero_tile` (1 page) | none |
| `load_gamma_slice` (RM) | helper | `dataflow_kernel_lib::read_sticks_for_tilize` + `compute_kernel_lib::tilize` | `tilize_helpers_dataflow.hpp:88`, `tilize_helpers.hpp:197` | `tilize<core_w_tiles, cb_gamma_rm, cb_gamma_tiles>(1)` | `cb_gamma_rm` | `cb_gamma_tiles` | **`core_w_tiles`** (CT template param) |
| `load_block` (RM) | helper | `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:88` | `<cb_input_rm, TilizeGranularity::TILE>(acc, rows_this_block, slice_bytes, start_page, w_start_tile·32·elem_bytes)` | DRAM | `cb_input_rm` | `block_row_tiles` (→ `rows_this_block`), `core_w_tiles` (→ `slice_bytes`) |
| `tilize_in_block` | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:197` | `<core_w_tiles, cb_input_rm, cb_input_tiles>(block_row_tiles)` | `cb_input_rm` | `cb_input_tiles` | **`core_w_tiles`** (CT), **`block_row_tiles`** (RT `num_blocks`) |
| `sumsq_block` | helper | `compute_kernel_lib::eltwise_chain` | `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp:552` | `eltwise_chain(EltwiseShape::grid(block_row_tiles, C_full), BinaryFpu<input(cb_input_tiles, Upfront, None, Block), input(cb_input_tiles, Upfront, None, Block), Mul, None, D0, DestAccumulation::PerRow>{}, PackTile<output(cb_stat_sq, PerOuter, PerOuter, …, DestAccumulation::PerRow)>{})` — `x·x` accumulated in fp32 DEST; see note ‡ | `cb_input_tiles` | `cb_stat_sq` | **`block_row_tiles`**, **`core_w_tiles`** (the `grid(H, W)` shape *is* the block) |
| `mask_tail_block` (mask) | helper | `compute_kernel_lib::mul` | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp:64` | `mul<input(cb_input_tiles, None, None, Col, TileOffset::Strided{base=C−1, row_stride=C}), input(cb_wmask, Upfront, None, Scalar), output(cb_tail_masked), BroadcastDim::Row>(EltwiseShape::grid(block_row_tiles, 1))` | `cb_input_tiles`, `cb_wmask` | `cb_tail_masked` | **`block_row_tiles`**, `core_w_tiles` (via `row_stride`) |
| `mask_tail_block` (accumulate) | helper | `compute_kernel_lib::eltwise_chain` | `eltwise_chain.hpp:552`, `L1Accumulation` at `eltwise_chain.hpp:322` | `eltwise_chain(EltwiseShape::grid(block_row_tiles, 1), BinaryFpu<input(cb_tail_masked), input(cb_tail_masked), Mul>{}, PackTile<output(cb_stat_sq, …, L1Accumulation::Enabled)>{})` — `Enabled` when `C_full > 0`, `Disabled` when `C_full == 0` | `cb_tail_masked` | `cb_stat_sq` | **`block_row_tiles`** |
| `reduce_stat_block` | helper | `compute_kernel_lib::reduce` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:525` | `reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_stat_sq, cb_scaler, cb_stat_partial, ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(block_row_tiles, 1))`; `ReducePartialScaler::none()` — the mask was already applied | `cb_stat_sq`, `cb_scaler` | `cb_stat_partial` | **`block_row_tiles`** (the `rows` field of `ReduceInputBlockShape`) |
| `combine_stat_block` (gather) | raw_api | `noc_async_write` + `noc_async_write_barrier` + `noc_semaphore_inc` / `noc_semaphore_wait_min` | modelled on `ttnn/ttnn/operations/examples/tensix_all_reduce/program_descriptor_with_inline_kernels.py:487-497` | write `block_row_tiles` fp32 tiles into the root's `cb_stat_gather` at slot `r·G + my_slot` | `cb_stat_partial` | `cb_stat_gather` | **`block_row_tiles`**, **`w_group_size`** |
| `combine_stat_block` (sum) | helper | `compute_kernel_lib::eltwise_chain` | `eltwise_chain.hpp:552`, `DestAccumulation` at `eltwise_chain.hpp:333`, reference usage `ttnn/cpp/ttnn/kernel_lib/tests/accumulation.cpp:32-46` | `eltwise_chain(EltwiseShape::grid(block_row_tiles, w_group_size), BinaryFpu<input(cb_stat_gather, Upfront, AtEnd, Block), input(cb_zero_tile, Upfront, None, Scalar), Add, None, D0, DestAccumulation::PerRow>{}, PackTile<output(cb_stat_sum, PerOuter, PerOuter, …, DestAccumulation::PerRow)>{})` | `cb_stat_gather`, `cb_zero_tile` | `cb_stat_sum` | **`block_row_tiles`**, **`w_group_size`** |
| `combine_stat_block` (finalize) | helper | `compute_kernel_lib::eltwise_chain` with `MulUnary` / `AddUnary` / `Rsqrt` | `eltwise_scalar.hpp:32`, `eltwise_scalar.hpp:26`, `eltwise_math.hpp:38`, chain at `eltwise_chain.hpp:552` | `eltwise_chain(EltwiseShape::tiles(block_row_tiles), CopyTile<input(cb_stat_sum)>{}, MulUnary<>{bitcast(1.0f/W_true)}, AddUnary<>{bitcast(epsilon)}, Rsqrt<>{}, PackTile<output(cb_rstd_send)>{})` | `cb_stat_sum` | `cb_rstd_send` | **`block_row_tiles`** |
| `combine_stat_block` (mcast) | helper | `SenderPipe::send` / `ReceiverPipe::receive`, wired by `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp:197`, `:274`, `:327`; host `ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp:450` (`Mcast2D`) | root: `sender.send(cb_rstd_send_addr, cb_rstd_addr, block_row_tiles·stat_tile_bytes)`; members: `receiver.receive()` | `cb_rstd_send` | `cb_rstd` | **`block_row_tiles`** (payload size), **`w_group_size`** (fan-out) |
| `scale_block` | helper | `compute_kernel_lib::mul` | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp:64` | `mul<input(cb_input_tiles, Upfront, AtEnd, Block), input(cb_rstd, Upfront, AtEnd, Col), output(cb_normed), BroadcastDim::Col>(EltwiseShape::grid(block_row_tiles, core_w_tiles))` | `cb_input_tiles`, `cb_rstd` | `cb_normed` | **`block_row_tiles`**, **`core_w_tiles`** |
| `gamma_block` | helper | `compute_kernel_lib::mul` | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp:64` | `mul<input(cb_normed, Upfront, AtEnd, Block), input(cb_gamma_tiles, Upfront, None, Row), output(cb_output_tiles), BroadcastDim::Row>(EltwiseShape::grid(block_row_tiles, core_w_tiles))` | `cb_normed`, `cb_gamma_tiles` | `cb_output_tiles` | **`block_row_tiles`**, **`core_w_tiles`** |
| `untilize_out_block` | helper | `compute_kernel_lib::untilize` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:154` | `<core_w_tiles, cb_output_tiles, cb_output_rm>(block_row_tiles)` | `cb_output_tiles` | `cb_output_rm` | **`core_w_tiles`** (CT), **`block_row_tiles`** |
| `store_block` (RM) | helper | `dataflow_kernel_lib::write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:130` | `<cb_output_rm>(acc, rows_this_block, slice_bytes, start_page, w_start_tile·32·elem_bytes)` | `cb_output_rm` | DRAM | **`block_row_tiles`**, **`core_w_tiles`** |
| `load_block` / `store_block` (TILE) | raw_api | `noc_async_read_tile` / `noc_async_write_tile` over `TensorAccessor` | `tech_reports/tensor_accessor/tensor_accessor.md` | batched 4–8 per barrier | DRAM | `cb_input_tiles` / DRAM | **`block_row_tiles`**, **`core_w_tiles`** (transfer counts) |
| `compute_kernel_hw_startup` | helper | required first statement of `MAIN()` | `eltwise_chain.hpp:25-41` | `compute_kernel_hw_startup(cb_input_tiles, cb_zero_tile, cb_stat_sq)` | — | — | — |

‡ **Note on `sumsq_block`.** The intent is `Σ_c x[r,c]²` accumulated **elementwise across tiles inside a
persistent fp32 DEST accumulator**, one packed tile per tile-row. `BinaryFpu` takes two CB operands, so the
square is expressed as `x · x` — both `InputSpec`s naming `cb_input_tiles`, exactly as
`eltwise_convenience.inl:33-38` expands `square<>`, which the header documents as waiting and popping that CB
once. If the chain's duplicate-upfront-CB `static_assert` nevertheless rejects two `Upfront` specs on one CB
id, two equivalent realizations preserve the same contract: (a) `square<…>` into a one-tile-row window
followed by `reduce<SUM, REDUCE_ROW, …, Accumulate::at(cb_stat_sq, c)>` per hidden chunk
(`reduce_helpers_compute.hpp:328`), or (b) a custom block operation built on
`add_tiles(…, acc_to_dest=true)` pairs, the pattern `row_reduce_accumulate` measures at 2.93×. The implementer
picks whichever the helper actually admits and records the choice; **all three preserve the planned residency,
phase boundaries and per-block init frequency.** `cb_zero_tile` is the identity operand for the *`Add`-based
combine only* (`combine_stat_block`), where a genuine second operand is needed — it plays no part in the
square.

### Helpers considered and rejected (every `raw_api` entry)

| `raw_api` entry | Helper considered | File:Line of the mismatch | Concrete reason it cannot express the shape |
|-----------------|-------------------|---------------------------|---------------------------------------------|
| `combine_stat_block` (gather) | `mcast_pipe` `SenderPipe`/`ReceiverPipe` | `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp:197` | `SenderPipe::send` is a **one-to-many broadcast** of one buffer to a rectangle; the gather is **many-to-one into disjoint slots** of a single destination CB. `mcast_pipe.hpp:44-45` states its precondition as "one sender per receiver … `dst_l1` identical on all receivers", which is the opposite direction. The helper *is* used for the return multicast in the same block operation. |
| `combine_stat_block` (gather) | `combine_welford_partials` | `ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/combine_welford.h:47-53` | Consumes a CB of interleaved `[mean, variance]` tile pairs and merges Welford sets; rms_norm's partial is a single raw `Σ x²` tile with no mean and no count-weighted merge, so the tile layout and the merge arithmetic both mismatch. |
| `load_block` / `store_block` (TILE) | `read_sticks_for_tilize` / `write_sticks_after_untilize` | `tilize_helpers_dataflow.inl:82-85` | These are RM-only: the implementation asserts `tile_size % tile_hw == 0` and derives a *stick* stride, so a TILE-layout tensor (whose DRAM pages are already tiles) has no sticks to read. The helpers are used on the RM branch. |

---

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `sumsq_block` | `Mul` (into a `PerRow` DEST accumulator) | `cb_input_tiles` — All `[32,32]` | `cb_input_tiles` — All `[32,32]` | `None` |
| `mask_tail_block` (mask) | `Mul` | `cb_input_tiles` (tile `C−1 + r·C`) — All `[32,32]` | `cb_wmask` — **Row0** (1.0 in columns `[0, partial_w)`, 0 elsewhere) | `Row` |
| `mask_tail_block` (accumulate) | `Mul` | `cb_tail_masked` — All | `cb_tail_masked` — All | `None` |
| `combine_stat_block` (sum) | `Add` (into a `PerRow` DEST accumulator) | `cb_stat_gather` — **Col0** (REDUCE_ROW output) | `cb_zero_tile` — All (zeros) | `None` |
| `scale_block` | `Mul` | `cb_input_tiles` — All `[32,32]` | `cb_rstd` — **Col0** (REDUCE_ROW output, `OperandKind::Col`) | `Col` |
| `gamma_block` | `Mul` | `cb_normed` — All `[32,32]` | `cb_gamma_tiles` — **Row0** (`[1,W]` operand, `OperandKind::Row`) | `Row` |

The two rules that matter here and are easy to invert: a `REDUCE_ROW` result is **column**-shaped and is
broadcast back with `BroadcastDim::Col` (`eltwise_chain.hpp:458-460`), while a `[1, W]` operand such as gamma
is **row**-shaped and uses `BroadcastDim::Row`. `OperandKind` (which tile to fetch) and `BroadcastDim` (how the
FPU expands it) are independent and both must be set.

---

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| The RMS denominator silently includes tile padding | `W % 32 != 0` is a first-class TARGET value, and the error is a near-uniform *scale* error that PCC is largely blind to at large `W`. The `_PAD_POISON_SHAPES` block exists solely to expose it: at `W = 40` the padding is 37.5% of the row and is filled with `1000.0`. | Two independent guards. (1) `mask_tail_block` zeroes the invalid **columns** of the ragged hidden tile *before* they reach the accumulator, using a 0/1 mask and the identity `(x·mask)² == x²·mask`. (2) the finalize multiplies by `1/W_true` computed from the **logical** W, never `tensor_w_tiles·32`. Both are needed: masking values without fixing the divisor still fails on the pad fraction alone. |
| Poisoned padding becoming NaN | `toy_variance.py:45-52` documents the trap: a garbage padding value `g` masked by multiplying by 0 gives `inf · 0 = nan`. | The mask is applied to `x` (not to `x²`) *before* squaring, so a finite poison value is annihilated before it can overflow. Non-finite padding would still poison the row — the same caveat the production reduce path carries — and the RM reader additionally zero-fills stale rows of a ragged last 32-row block. |
| `w_group_size` chosen so that a core owns zero hidden tiles | The root then waits forever on a semaphore that will never reach `w_group_size − 1`. Only reachable on narrow tensors (`tensor_w_tiles < num_cores`) — which `_RESILIENCE_SHAPES` has plenty of, e.g. `(1,1,3232,96)` with `Wt = 3`. | Declared mechanism cap: the selection function discards every candidate with `w_group_size > tensor_w_tiles`. |
| `core_w_tiles` differing across a reduction group | `tilize`/`untilize` take the width as a **template** parameter, so a ragged hidden remainder cannot be a runtime value; and CB total sizes must be identical across the group or the multicast destination address diverges. `tensor_w_tiles % w_group_size != 0` is the common case (`Wt = 89, 127, 37, 23 …` in `_RESILIENCE_SHAPES`). | Two kernel core-ranges with separate CT blocks (`ceil` and `floor` widths); **all CBs sized to `ceil` on both**, so L1 addresses stay identical group-wide. |
| Multicast destination address mismatch | `mcast_pipe.hpp:44-45` requires `dst_l1` identical on every receiver. Restricting `cb_stat_gather` to root cores only (a tempting L1 saving) would shift every subsequent CB on the roots and break this. | Every CB is allocated on the **full active core set**, including the ones only the root uses. The cost is accounted explicitly in `l1_ledger.md`. |
| `cb_input_tiles` must survive two consumers-in-sequence | `sumsq_block` and `scale_block` both read the same block; if `sumsq_block` popped, the input would have to be re-read from DRAM (regime R3, +1 whole tensor of DRAM traffic — the largest term in the traffic ranking). | `sumsq_block` uses `WaitPolicy::Upfront, PopPolicy::None`; `scale_block` uses `PopPolicy::AtEnd`. Push count (`R·C` by the reader) equals total wait count, and the pop happens exactly once. |
| Regime selected from the device grid | R1 vs. R2 depends on `grid_x × grid_y`; a 64-core part and a 110-core part can disagree for the same shape, so a bug in one regime can pass CI on one board. | Selection function stated exactly above; the acceptance test pins both regimes by construction (`tensor_row_tiles == 1` forces R2 on any grid; a tall narrow shape forces R1). |
| fp32 DEST halves capacity | Phase 0 is the maxed-out corner (`fp32_dest_acc_en=True`), so `DEST_AUTO_LIMIT` is 4 in half-sync — half of what a design tuned at `False` would assume. `DestAccumulation::PerRow` also holds one slot for the whole row walk. | Nothing hardcodes 8: the eltwise chain clamps `block_size` to `DEST_AUTO_LIMIT` (`eltwise_chain.hpp:430-442`) and the reduce derives its chunk from the same constant (`reduce_helpers_compute.inl:1120`). Recorded as a mechanism cap, not as a block bound. |
| Reduce scaler CB dtype | `reduce_helpers_dataflow.inl:185-187` `static_assert`s on anything but `Float16_b` / `Float32`; a `bfloat8_b` scaler is a compile error, and a wrong pool-type/reduce-dim overload silently fills the wrong tile pattern. | `cb_scaler` is declared **bfloat16** and filled with the pool-type-aware `calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()`. |
| `gamma` at a different dtype from the input | A realistic mixed-precision case (`gamma_dtype` is its own TARGET axis) and a silent-corruption source if the format reconfig is skipped. | `cb_gamma_tiles` carries the gamma dtype and `gamma_block`'s `InputSpec` leaves `DataFormatReconfig::Enabled`. The reconfig-off optimization from `compute_block_size` is deliberately **not** applied at any boundary where the dtype changes (bf16 in → fp32 stat → gamma dtype → bf16 out). |
| `gamma` shape validation | The requirement is an explicit `ValueError` on a last-dim mismatch, and `rank < 2` on the input. | `validate()` checks `input.shape` rank ≥ 2 and `gamma.shape[-1] == input.shape[-1]` before any device work, and never refuses the `"none"` sentinel for `gamma_dtype` / `gamma_layout`. **Error-text contract** (pinned by the acceptance test): the rank rejection must contain `rank`, the gamma rejection `gamma`, and the precision rejection `fp32_dest_acc_en`, so the three are distinguishable by a caller and by CI log triage. |

## Structural impossibilities (candidates for a future `/golden-tests` pass)

`feature_spec.py` already covers the ones this design can see. Two further candidates the author may wish to
fold in — noted here only, **not** edited into `feature_spec.py`:

| Candidate cell | Why it is structurally impossible |
|----------------|-----------------------------------|
| `{gamma_layout: ROW_MAJOR, gamma_dtype: bfloat8_b}` | Already present at `feature_spec.py:69`. Listed only to confirm it was checked. |
| `rank < 2` | Not a cartesian cell (TARGET's `rank` starts at 2), and it is a `validate()` rejection rather than an INVALID skip — correctly modelled as-is. |

Everything else in `TARGET − Phase-0-SUPPORTED` (`bfloat8_b`, `fp32_dest_acc_en=False`, the three `*_SHARDED`
placements) is a genuine refinement candidate, not an impossibility. `{float32, fp32_dest_acc_en=False}`
belongs in the op's `EXCLUSIONS`, per `.claude/references/precision_convention.md`.

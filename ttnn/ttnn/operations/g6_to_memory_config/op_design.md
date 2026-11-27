# Operation Design: g6_to_memory_config

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Return the input's tile pages, bit-identical, in the placement `memory_config` names. One device program per call; every page is read once from its place under the input's placement and written once to its place under the target placement, through the L1 of the core that moves it. |
| Math | `y[i] = x[i]`. Per tile page: `out_page[p] = in_page[p]` for `p ∈ [0, P)`, `p = tile_row · tensor_cols_tiles + tile_col` over the padded 2D tile grid both placements share. Positions past the logical H/W inside a tile are copied as they are (unspecified by the contract). |
| References | spec (this task); `eval/golden_tests/g6_to_memory_config/{feature_spec.py,helpers.py}`; `.claude/references/{blocking-model.md,l1-footprint-discipline.md,noc-alignment-in-tt-metal.md}`; `.claude/skills/memory-layouts/SKILL.md` §3.4; `tech_reports/tensor_accessor/{tensor_accessor.md,tensor_accessor_iterator.md}`; catalog `ttnn/ttnn/operations/examples/master.md` entries `noc_placement`, `double_buffer`, `trid_double_issue`, `tile_reorder`, `bank_stagger`, `read_coalescing`, `split_reader` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | on device, `TILE_LAYOUT`, rank 2–4, any placement with a row-major shard spec if sharded | — | shape / dtype / placement → plan + CT args; buffer address → common RT arg |
| `memory_config` | `ttnn.MemoryConfig` | yes, positional | DRAM or L1; interleaved, or height / width / block sharded with a row-major shard spec whose shard shape is a multiple of 32 and whose grid holds every shard | — | plan + CT args (both TensorAccessor arg blocks are CT); output buffer address → common RT arg |

Second public function (BB-08 / BB-21): `plan_of(input_tensor, memory_config) -> dict` returns the plan the call selects — the same function and the same cache the op uses (see Host plan). No dtype, layout or compute-config parameter exists.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(..., H, W)`, rank 2–4; H any positive value; W a multiple of 32 in every production case (any W is still page-exact) |
| Dtype | `bfloat16`, `bfloat8_b`, `bfloat4_b` (block formats since Refinement 1 — same kernels, page size is a parameter) |
| Layout | TILE |
| Memory | any: DRAM / L1 × interleaved / height / width / block sharded |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `input_tensor.dtype` |
| Layout | TILE |
| Memory | exactly `memory_config` (buffer type, memory layout, shard grid with its range order, shard shape, orientation): allocated with `ttnn.allocate_tensor_on_device(shape, dtype, TILE_LAYOUT, device, memory_config)` |

## Registry contract (op file)

Axis names are TARGET's in `feature_spec.py`. Taggers read only the scenario dict `inputs[0]`; `validate()` builds the same scenario dict from the tensor and `memory_config` and runs the same taggers (single source of truth).

| Tagger (declaration order) | Returns |
|---|---|
| `tag_dtype` | `scenario["dtype"]` |
| `tag_layout` | `scenario["layout"]` |
| `tag_rank` | `int(len(scenario["shape"]))` |
| `tag_alignment` | `"tile_aligned"` if `shape[-2] % 32 == 0 and shape[-1] % 32 == 0` else `"h_non_aligned"` |
| `tag_in_scheme` / `tag_out_scheme` | `scenario["in_memory"|"out_memory"]["scheme"]` |
| `tag_in_buffer` / `tag_out_buffer` | `scenario["in_memory"|"out_memory"]["buffer"]` |

| Declaration | Value (Refinement 1; Phase 0 had `dtype: [bfloat16]` and the DRAM width-sharded target excluded) |
|---|---|
| `SUPPORTED` | `dtype: [bfloat16, bfloat8_b, bfloat4_b]`, `layout: [TILE]`, `rank: [2,3,4]`, `alignment: ["tile_aligned","h_non_aligned"]`, `in_scheme`/`out_scheme`: all four, `in_buffer`/`out_buffer`: `["dram","l1"]` |
| `EXCLUSIONS` | `[]` (SUPPORTED equals TARGET) |
| INVALID | not declared, not checked (lives in feature_spec.py; empty) |

`validate(input_tensor, memory_config)` order: (1) `ValueError` if the input is not a device tensor (the axes cannot be read otherwise); (2) build scenario, tag; (3) SUPPORTED per axis → `UnsupportedAxisValue`; (4) EXCLUSIONS → `ExcludedCell`; (5) structural checks below → `ValueError`. `UnsupportedAxisValue`/`ExcludedCell` subclass `NotImplementedError` (`_op_contract.py`), itself a `RuntimeError`, so a ROW_MAJOR input or a rank-5 input is refused with a `RuntimeError` subclass, as the spec requires.

| Structural check (`ValueError`) | Rule |
|---|---|
| shard orientation | input's and `memory_config`'s shard spec must be `ROW_MAJOR` |
| shard shape | both dims multiples of 32 |
| height sharded | `shard_w == padded W` |
| width sharded | `shard_h == tensor_rows_tiles · 32` |
| shard capacity (1D: height/width) | `num_shards ≤ grid.num_cores()`; more cores than shards is valid |
| shard capacity (2D: block) | grid is one `CoreRange`; `n_shard_cols ≤ grid width`, `n_shard_rows ≤ grid height` (`buffer_distribution_spec.cpp:176-193` would `TT_FATAL` otherwise) |
| BB-05 budget | per-core op L1 (output share on the most loaded core + CB bytes, Ledger) `≤ L1_OP_BUDGET_BYTES = 1347776`; refuse on the host otherwise. Every golden shape passes (max 192512 B, Qwen 2078). |

No host-side tensor transformation of any kind (BB-02): no `to_layout`, `pad`, `slice`, `typecast`, `to_memory_config`, `clone`.

## Blocking Model

The op is a permutation of whole tile pages between two placements. Every page is independent; the design question is entirely **who moves each page and through how many NoC transfers**. The minimum is one NoC transfer per page whenever one endpoint is the mover's own L1 (a NoC read lands only in the issuer's L1 and a NoC write sources only from it), and two when neither endpoint is a core's L1 (DRAM→DRAM). So the mover of a page is chosen to *own* one side of it: the core that holds the page's target slot pulls it, or the core that holds the page's source slot pushes it. That makes the core-assignment a property of the placements, and the block of a mover is the whole set of pages it owns.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `leading` (dims before H) | independent — each output page depends on the one input page at the same index; folds into `tile_row` exactly, since TILE pads each image to whole tile rows and both TensorAccessors index the same flattened padded grid (legacy shard specs fold leading dims too: `squeeze_shape_ranks`, `buffer_distribution_spec.cpp:398-447`) | none (folded): `tensor_rows_tiles = prod(leading) · ceil(H/32)` | folded | host `tile_grid(input)` | carried by `tile_row` | none needed — folding is exact for every placement |
| `tile_row` | independent — no output page reads another row | `block_extent[tile_row]` | shard regimes: the owned shard's tile rows, `shard_h_tiles`, ragged last shard `tensor_rows_tiles − (n_shard_rows − 1)·shard_h_tiles` (one block = the whole resident shard); bank / bounce regimes: not cut on its own — the `page` walk below owns it | the owned side's `ShardSpec` (target's for pull, input's for push), read once by the host plan | shard row → its owner core, exactly as the buffer places it | knob-turn (`movers_per_owner` sub-cuts an owner's block across its second RISC) |
| `tile_col` | independent — no output page reads another column | `block_extent[tile_col]` | shard regimes: `shard_w_tiles`, ragged last shard (e.g. 40 of 52 for Qwen 2071) | owned side's `ShardSpec` | shard column → owner core | knob-turn |
| `page` (`tile_row · tensor_cols_tiles + tile_col`; the view interleaved placements and both accessors use) | independent — same argument, flattened | `bank_block_pages` (bank regimes); `bounce_block_pages` (bounce) | bank: the whole stripe of bank `b`, `ceil((P − b)/nb)` pages, one block; bounce: `min(ceil(core_pages_max / bounce_depth), floor(L1_OP_BUDGET_BYTES / (bounce_depth · page_size)))` | host plan from `P`, `nb`, `bounce_depth`, `page_size`, `L1_OP_BUDGET_BYTES` | bank: page `p` → the core holding L1 bank `p mod nb`; bounce: `ttnn.split_work_to_cores(grid, P, row_wise=True)` contiguous page ranges (or, with `BOUNCE_TARGET_ORDER`, contiguous ranges of a sharded target's storage order) | knob-turn; bounce write unit `BOUNCE_WRITE_RUN_PAGES` (0 = whole run contiguous in the target) |

Phase 0 values at a glance: every owner moves its whole owned set as **one block under one barrier**; the bounce mover cuts its range into `bounce_depth` blocks so the read of one overlaps the write of the other at no extra L1 (the CB holds the whole assignment either way). Bounce is the only regime with a block below the coarsest; the reason is overlap at zero L1 cost, measured by lamp L6.

Additional core-assignment knob: `movers_per_owner` ∈ {1, 2} — how many data-movement RISCs share one owner's block. Phase 0 = 1 (NCRISC/NoC0 for pull, BRISC/NoC1 for push). Source: host constant → CT arg. It is a knob, not a regime: the block, the NoC path and the bytes are unchanged; lamp L1 measures 2.

Single sources: `page_size = input_tensor.buffer_aligned_page_size()` (host asserts equal to the output's); `P = tensor_rows_tiles · tensor_cols_tiles` from `tile_grid(input)` with `ceil` per image (never `buffer_num_pages()` of a sharded buffer, which can count padded shard slots); `nb = G` (one L1 bank per compute core, mechanism cap 4); `owner_cores` and `owned_shard_id` from the host plan; `bounce_depth = 2`, `L1_OP_BUDGET_BYTES = 1347776` as named host constants. Every CT/RT value and the CB capacity derive from these.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_bounce_pages` (bounce only) | `bounce_depth` = `BOUNCE_DEPTH` (per-page reads) or `BOUNCE_STAGED_DEPTH` (staged reads) | 2 / 6 (Refinement 3) | the reader fills later blocks while the writer drains earlier ones. A staged block lands in one burst after its L1 reorder, so 6 finer blocks let the writer start early (−6% on 2183 / 2179); per-page blocks stream in, and more of them only add handshakes (bf16 16×128×1536: 30.5 → 33.9 µs at 6) |
| `cb_bounce_stage` (bounce, staged reads only) | one bounce block (`BOUNCE_STAGE_PAGES = 0`) | — | reader-private landing zone of the per-bank runs, reordered to page order in L1 before the block is pushed |
| none (pull / push regimes) | in-flight transactions = whole owned block | `|owned set|` NoC transactions before the one barrier | every transfer of the block is on the wire at once; there is no staging buffer to deepen |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Blackhole DRAM-read alignment: local L1 address ≡ DRAM address (mod 64) (`noc-alignment-in-tt-metal.md` §1) | any DRAM→L1 transfer (pull from a DRAM source, bounce reads) | holds by construction: L1 allocations are 64 B aligned (`bank_manager.cpp:122-131,176` — the FreeListOpt is built with `dram_alignment`), the CB region starts 64 B aligned (`l1_banking_allocator.cpp:249-250`), and every tile page (2048 / 1088 / 576 B) is a multiple of 64. Host asserts `page_size % ttnn.get_dram_alignment() == 0` and, for each L1 side paired with a DRAM side, `buffer_address() % ttnn.get_dram_alignment() == 0` per call. | silent data corruption (watcher stops only under `--dev`) |
| Legacy shard specs hold at most one shard per core (`ROUND_ROBIN_1D` with `num_shards ≤ num_cores`; `GRID_2D` trims the grid to exactly the shard grid: `buffer_distribution_spec.cpp:162-201`) | owned shards per mover | one `owned_shard_id` per core; validation refuses `num_shards > capacity` | a second shard on a core would be silently unmoved |
| Owner order is the buffer's, not the grid's written order: 1D → `corerange_to_cores(grid, row_wise=True)[:num_shards]`; 2D block → the trimmed rectangle from the grid start, row-wise | `owned_shard_id` ↔ core | the plan derives `owner_cores` with that rule and, once per cache key, asserts it equals `ttnn.get_optimal_worker_cores_for_sharded_tensor(t)` (`tensor_utils.cpp:191-212`, = `cores_with_data()`) for the real tensors; mismatch → `RuntimeError` | a pull core would compute a non-local "local" address and read into the wrong L1 |
| One L1 bank per compute core, banks only on compute cores (`l1_banking_allocator.cpp:96-111`, ComputeAndStore = logical compute cores `:285-289`); bank ids are shuffled (`:80-86`) and no Python binding exposes the map | bank stripe ownership | bank regimes launch on the full `compute_with_storage_grid`; each core finds its bank in-kernel (scan `b < min(NUM_L1_BANKS, P)` for the `b` whose `get_noc_addr(b)` node bits equal its own); host predicate `bank_cores_cover_grid = (ttnn.reports.get_device_info(device).l1_num_banks == G)`, cached per device; false → the bounce regime (reported by `plan_of`) | a bank on an unlaunched core keeps its pages unmoved: wrong output |
| `shard_pages()` requires a static-rank accessor (`tensor_accessor.h:336`) | sharded owned-side walk | both accessors' args are fully compile-time (`ttnn.TensorAccessorArgs(t).get_compile_time_args()`, default config) | compile error |
| Same page id on both sides | every transfer | interleaved and legacy-sharded accessors both index the flattened padded tile grid (`squeeze_shape_ranks`); host asserts input/output padded shapes and page sizes are equal | a page would land at another page's slot |
| Bounce CB L1 budget (BB-05) | `bounce_block_pages` | `bounce_depth · bounce_block_pages · page_size ≤ L1_OP_BUDGET_BYTES` (bounce targets DRAM, so no output share on the core) | CB/L1 clash at launch inside the model |

### Regimes

Named boundary for "minimum": each input page crosses its source tier once and each output page its target tier once; in NoC terms **one transfer per page when the mover owns one endpoint, two when neither endpoint is a core's L1**.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `pull_shard` | built | `dst_l1 ∧ dst_sharded ∧ ¬(src_l1 ∧ src_sharded ∧ n_src_owners > n_dst_owners)` — 24 bf16 golden cases (e.g. Qwen 2058, 2067, 2154, FLUX 313) | the owned target shard, `block_extent[tile_row] × block_extent[tile_col]` (ragged last), one per owner | minimum: one NoC read per page, landing at its final address; source read once from wherever it lives | the whole shard is one block, so one `noc_async_read_barrier` and one kernel launch per owner per call; per-page cost is only the issue |
| `pull_bank` | built | `dst_l1 ∧ ¬dst_sharded ∧ ¬(src_l1 ∧ src_sharded) ∧ bank_cores_cover_grid` — 27 bf16 golden cases, all DRAM interleaved → L1 interleaved, up to P = 10240 (Qwen 2078) | the stripe of the core's L1 bank `b`: pages `b, b+nb, …`, `ceil((P−b)/nb)` of them, one block | minimum: one NoC read per page from the source straight into its final bank slot (the bounce alternative adds an L1→L1 write per page) | one bank scan + one read barrier per core per call |
| `push_shard` | built | `src_l1 ∧ src_sharded ∧ (¬(dst_l1 ∧ dst_sharded) ∨ n_src_owners > n_dst_owners)` — 35 bf16 golden cases (e.g. Qwen 2071, 2065, 2073, FLUX 314) | the owned source shard (ragged last), one per owner | minimum: one NoC write per page from its resident source address; the resident shard is never re-read over the NoC | one `noc_async_write_barrier` and one launch per owner per call |
| `push_bank` | built | `src_l1 ∧ ¬src_sharded ∧ ¬dst_l1 ∧ bank_cores_cover_grid` — no golden case; a SUPPORTED cell (L1 interleaved → DRAM), pinned by the acceptance test | the stripe of the core's L1 bank on the source side | minimum: one NoC write per page | one scan + one write barrier per core |
| `bounce` | built | `¬src_l1 ∧ ¬dst_l1` (DRAM → DRAM), or a bank regime when `¬bank_cores_cover_grid` — no bf16 golden case (DRAM→DRAM SUPPORTED cells, pinned by the acceptance test); the 7 bf4b weight cases land here with the refinement | `bounce_block_pages` contiguous pages; `bounce_blocks_this_core = ceil(core_pages / bounce_block_pages)` (Phase 0: `bounce_depth`) | DRAM minimum (read once, write once); two NoC transfers per page, which **is** the minimum for DRAM→DRAM. In the fallback case (an L1 side exists) it is one transfer above minimum, stated and reported by `plan_of` | fewer CB handshakes and read barriers per core; bounded below by `bounce_depth` blocks so read and write overlap |
| `bounce_dram_shard` | built in Refinement 2 as bounce knobs, not a regime. Shipped: the writer's transfer unit, one NoC write per run of pages contiguous in the target (`BOUNCE_WRITE_RUN_PAGES = 0`, default on; −9 to −12% on the bf4b cases). Parked, default off, measured a loss: assignment over the target's storage order (`BOUNCE_TARGET_ORDER`, 2183: 375 µs vs 295 µs) and `bank_stagger` (`BOUNCE_BANK_STAGGER`, noise). | `¬src_l1 ∧ dst_dram ∧ dst_sharded` | page order (default): `bounce_block_pages` contiguous page ids, written as runs of up to a shard row (64 × 576 B = 36 KB on 2183). Target order (knob): `bounce_block_pages` pages of the target's storage order from a host-given storage slot (`shard_pages`), movers interleaved over shards | DRAM minimum; NoC writes drop from one per page to one per run (`num_contiguous_pages` on the target accessor; adjacent runs merge by address, never across a block); reads stay per page (interleaved source) | one write per run instead of one per page; past ~4.6 KB per write the size stops mattering (run caps of 8 and 28 pages measure as noise) |
| `bounce_staged_read` | built in Refinement 3 as bounce knobs, not a regime. Shipped: `BOUNCE_READ_COALESCE` with `BOUNCE_STAGE_BELOW_PAGE_BYTES = 1024` (stage strided reads of bf4b pages only) and `BOUNCE_STAGED_DEPTH = 6`. Parked: `BOUNCE_STAGE_PAGES` (stage chunk below a block: noise), `BOUNCE_NOC_SWAP` (loss), staging 1088 B / 2048 B pages (loss: bf8b 114.8 → 119.5 µs, bf16 31.8 → 33.4 µs) | `¬src_l1 ∧ ¬dst_l1`, strided source, `page_size < 1024` | `bounce_block_pages` pages per block, `BOUNCE_STAGED_DEPTH` blocks per core; one DRAM read per bank run of the block into `cb_bounce_stage`, then one self-aimed L1 read per page into its page-order CB slot | DRAM minimum unchanged; adds one L1→L1 pass of the block on the mover (no DRAM or cross-core bytes) | 2183: 294.8 → 260.2 µs; 2184 233.0 → 203.2; 2179 157.7 → 141.3; 2182 58.8 → 52.1 |
| `bounce_all` | rejected — superseded by the four owner regimes wherever an L1 side exists (two transfers per page → one). A dead end: the owner regimes are built instead of it, not on top of it. | — | contiguous page ranges on third-party cores | one transfer per page above minimum for every L1-side case | — |
| `two_step` (stage through an interleaved tensor in a third placement, or call an existing data-movement op) | rejected — forbidden by the spec and BB-01; pays every crossing twice | — | — | 2× the DRAM/L1 crossings and two programs | — |
| `fan_out` (spread one owner's pages over extra third-party cores) | rejected — adds a NoC hop per page; superseded by `movers_per_owner = 2` (the owner's second RISC, no hop). Every golden case above 8 pages already has ≥ 8 owners; the largest per-owner load is 131072 B (Qwen 2072 push), far under a single core's measured NoC ceiling (~122 GB/s, `trid_double_issue`) | — | — | +1 transfer per page | — |
| `single_core` | rejected — same bytes, no parallelism; Phase 0 is multi-core | — | — | minimum bytes, serialized issue | — |

No dependent axis exists, so there is no "split the dependent axis and combine" row. Operand-reuse check: the single operand varies along every axis any regime splits, so no broadcast row.

Selection function (exact, host-checkable; also what `plan_of` reports):

```
src_l1, dst_l1 = in.buffer_type == L1, out.buffer_type == L1
src_sh, dst_sh = in.is_sharded(), out.is_sharded()
n_src_owners / n_dst_owners = len(owner_cores(side))           # cores holding a shard
if dst_l1 and dst_sh:  regime = "push_shard" if (src_l1 and src_sh and n_src_owners > n_dst_owners) else "pull_shard"
elif dst_l1:           regime = "push_shard" if (src_l1 and src_sh) else "pull_bank"
elif src_l1:           regime = "push_shard" if src_sh else "push_bank"
else:                  regime = "bounce"
if regime in ("pull_bank", "push_bank") and not bank_cores_cover_grid: regime = "bounce"
```

### Traffic ranking

| Candidate split | DRAM crossings (input / output) | NoC transfers per page | Fan-out | Verdict |
|---|---|---|---|---|
| by target ownership (pull) | 1 if in DRAM / 1 if in DRAM — 0 for an L1 side | 1 | none — each page goes to one core | rank 1 (tie) |
| by source ownership (push) | same | 1 | none | rank 1 (tie) |
| third-party contiguous ranges (bounce) | same | 2 | none | rank 2 — chosen only when neither side is any core's L1 (DRAM→DRAM), where it is the minimum |
| two-step through a third placement | 2 / 2 | 2–4 | — | rejected |
| split a dependent axis + combine | n/a — none exists | — | — | — |

Pull/push tiebreak (equal bytes): the side with more owner cores moves (more injection points); exact tie → pull. An L1-interleaved side is never the mover when the other side is L1-sharded (the shard side has the owners and needs no bank scan). Occupancy is kept out of the ranking; separately: bank and bounce regimes use the whole grid, shard regimes use every core that holds a shard of the owned side.

Stall-shadow check: the only waits are the final barrier of each owner (everything is already issued — nothing independent remains to schedule into it), the bank scan (must precede any issue: the bank id names the pages), and the bounce CB waits (covered by `bounce_depth = 2`). No reorder is available or needed; there is no arithmetic, so nothing changes floating-point order.

### Block schedule

Logical schedule per regime; reader / writer are separate kernels and adjacent bounce blocks pipeline.

```cpp
// pull_shard / pull_bank — one mover kernel per owner core (NCRISC, NoC0)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {  // Phase 0: 1 for an owner, 0 for an idle core
    own_block(block_idx);     // shard: owned_shard_id (RT arg); bank: scan once for this core's bank b
    pull_block(block_idx);    // one NoC read per owned page: source page p -> its final local target slot
    drain_block(block_idx);   // one noc_async_read_barrier
}

// push_shard / push_bank — one mover kernel per owner core (BRISC, NoC1)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    own_block(block_idx);     // shard: owned_shard_id; bank: scan once
    push_block(block_idx);    // one NoC write per owned page: resident source slot -> target page p
    drain_block(block_idx);   // one noc_async_write_barrier
}

// bounce — reader (NCRISC, NoC0) and writer (BRISC, NoC1) over cb_bounce_pages
for (uint32_t block_idx = 0; block_idx < bounce_blocks_this_core; ++block_idx) {
    load_block(block_idx);    // reader: reserve, one read per page, one read barrier, push bounce_block_pages
    store_block(block_idx);   // writer: wait, one write per page, writes_flushed, pop bounce_block_pages
}
// writer ends with noc_async_write_barrier (BB-03)
```

| Operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `own_block` | — | — | shard: none (host-given id); bank: one scan of ≤ `min(NUM_L1_BANKS, P)` table entries per core per call |
| `pull_block` | owned target shard `block_extent[tile_row] × block_extent[tile_col]`, or the bank stripe | the target buffer (it *is* the destination) | 0 barriers inside; reads issued back to back |
| `push_block` | owned source shard or source bank stripe | the source buffer (read in place, never over the NoC) | 0 barriers inside |
| `drain_block` | the block | — | 1 barrier per core per call (BB-03 is met by this barrier) |
| `load_block` | `bounce_block_pages` pages | `cb_bounce_pages` slot | 1 `cb_reserve_back`, 1 read barrier, 1 `cb_push_back` per block; Phase 0 `bounce_depth` blocks per core |
| `store_block` | `bounce_block_pages` pages | `cb_bounce_pages` slot | 1 `cb_wait_front`, 1 `noc_async_writes_flushed`, 1 `cb_pop_front` per block; 1 final write barrier per core |

Ragged tails: the last owner shard and the last bounce block are shorter. Push/pop counts stay at the nominal `bounce_block_pages` (two pushes of `bounce_block_pages` close the ring exactly: capacity = `bounce_depth · bounce_block_pages`); only the NoC work narrows. Owner regimes have no CB, so the ragged shard only shortens the page walk (`shard_pages()` skips slots beyond the tensor).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 RISC split (`split_reader`) | an owner with tens of pages issues them all from one RISC on one NoC; if issue-bound, half the core's issue capacity and one NoC sit idle | `movers_per_owner = 2`: halves of the block on NCRISC/NoC0 and BRISC/NoC1 (reads on NoC1 can lose for spread placements — `noc_placement`; measure pull and push separately) |
| L2 transfer coalescing (`read_coalescing`, `tile_reorder`) | shard↔shard moves have runs contiguous on both sides (Qwen 2067: each 4-page target shard lies inside one source shard), moved one 2 KB page per transaction | transfer = maximal run contiguous on both sides (`num_contiguous_pages` on both accessors when their `contiguous_page_stride()` agree), vs. one page per transaction |
| L3 bank stagger (`bank_stagger`) | owners of height-sharded targets read row-aligned DRAM pages: every owner starts on DRAM bank 0 (Qwen 2058: 16 owners × pages `8k..8k+7`) | start each owner's walk at offset `owner_idx mod NUM_DRAM_BANKS` and wrap |
| L4 pull vs push on L1-shard↔L1-shard | the "more owners moves" tiebreak assumes injection points dominate; posted writes vs. round-trip reads may reverse it at 8 vs 40 owners | force the other direction on Qwen 2067 (8→40) and 2073 (40→8) |
| L5 bank scan on small P / grid-synchronization | `pull_bank` launches all G cores and every core scans even when `P < nb` (Qwen 2089: 96 pages on 110 banks; most cores move ≤ 1 page) | `bounce` over `min(P, G)` movers (no scan, +1 hop), and the scan written against the firmware bank table directly instead of `get_noc_addr` |
| L6 overlap (bounce) | `bounce_depth` blocks of half the assignment trade one extra handshake for read/write overlap; at DRAM-bound grid scale overlap may buy nothing (`double_buffer`: no gain once bandwidth-bound) | one block (no overlap), and 8-page blocks with `trid` pipelining (`trid_double_issue`, `local_copy_helpers_dataflow.hpp:147,158`) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| source page | tile page, bytes untouched (`page_size` B) | source TensorAccessor (`get_noc_addr(p)`, or the owned side's `shard_pages()` / strided `Pages` iterator) | interleaved or sharded, DRAM or L1 — one accessor type covers all |
| pull: source → final target slot | same bytes | one `noc_async_read` per page into the local L1 address of target page `p` (taken from the target accessor; local by construction) | no CB, no staging; a same-core source is a NoC self-read |
| push: resident source slot → target | same bytes | one `noc_async_write` per page from the local L1 address of source page `p` | the resident shard / bank stripe is consumed in place — never re-read over the NoC |
| bounce: DRAM → `cb_bounce_pages` → DRAM | same bytes | reader `noc_async_read` per page (or, for pages under `BOUNCE_STAGE_BELOW_PAGE_BYTES` = 1024 B from an interleaved source, one read per bank run into `cb_bounce_stage`, then one self-aimed L1 read per page into its page-order slot); writer `noc_async_write` per run contiguous in the target | a stride-1 source coalesces in place, with no stage |
| tile padding (rows `H..`, cols `W..` inside a tile) | copied as stored | whole-page moves | content unspecified by the contract; block formats keep every shared exponent because no page is cut |

No regime needs inter-Tensix synchronization: every page has exactly one mover, sources are immutable during the call, and targets are written by exactly one core. No semaphores, no multicast.

TARGET placement axes as schemes (no axis is dependent, so every placement value is a knob-turn of the same structure):

| source \ target | L1 sharded | L1 interleaved | DRAM interleaved | DRAM sharded |
|---|---|---|---|---|
| L1 sharded | `pull_shard` / `push_shard` (more owners moves) | `push_shard` | `push_shard` | `push_shard` |
| L1 interleaved | `pull_shard` | `pull_bank` (same bank both sides: local copies) | `push_bank` | `push_bank` |
| DRAM interleaved | `pull_shard` | `pull_bank` | `bounce` | `bounce` (width: supported since Refinement 1; `bounce_dram_shard` is Refinement 2) |
| DRAM sharded | `pull_shard` | `pull_bank` | `bounce` | `bounce` |

Refinement classes: `dtype` bfloat8_b / bfloat4_b — knob-turn (`page_size` is the only dtype-dependent quantity; kernels move bytes). `out_scheme=width_sharded ∧ out_buffer=dram` — knob-turn on `bounce` (remove the EXCLUSION) plus the deferred `bounce_dram_shard` regime for performance.

Structural impossibilities: none beyond `feature_spec.py`'s (INVALID is empty, and every axis is projected from production cases).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one mover's block (Blocking Model): an owned shard, an owned bank stripe, or a bounce range |
| Grid | `G = device.compute_with_storage_grid_size()` (x·y); bank and bounce regimes use all of it |
| Per-core work | `pull_shard`/`push_shard`: owner `k` of `owner_cores(side)` moves shard `k` (`owned_shard_id = k`); `pull_bank`/`push_bank`: every core moves its bank's stripe; `bounce`: `split_work_to_cores(G, P, row_wise=True)` → `(start_page, core_pages)` per core |
| Remainder | shard regimes: the ragged last shard is the owner's smaller block (`shard_pages()` skips slots past the tensor); bank: stripe lengths differ by ≤ 1; bounce: group-2 cores get one page fewer, the last block of a core is shorter. Tile counts: `tensor_rows_tiles = prod(padded_shape[:-2]) · ceil(H/32)`, `tensor_cols_tiles = ceil(W/32)` — per image, never `floor`. |
| Kernel core range (BB-17) | the bounding rectangle of the regime's movers — one `CoreRange`. Shard regimes: bounding box of `owner_cores` (Qwen 2058: 16 single-core ranges → one 11×2 box, 6 idle cores); bank / bounce: the full grid. Non-mover cores in the box get `owned_shard_id = NO_SHARD` / zero pages and return at once. |
| NoC / RISC | pull and bounce-reader: NCRISC, NoC0 (`ReaderConfigDescriptor`); push and bounce-writer: BRISC, NoC1 (`WriterConfigDescriptor`) — the default `noc_placement` measured fastest. `BOUNCE_NOC_SWAP` (default `"none"`) swaps the bounce NoCs on a grid subset as separate kernel groups; every subset measured slower on 2183 (all 385, right half 289, odd columns 287, bottom half 283, odd rows 279, checker 278 vs 268 µs) |
| Regime selection | the function above; regime-pinned tests: one acceptance test per built regime asserts `plan_of(...)["regime"]` |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_bounce_pages` | 0 | `page_size` (= `input_tensor.buffer_aligned_page_size()`) | `bounce_depth · bounce_block_pages` | spans `page` → `bounce_block_pages` per slot × `bounce_depth` slots; `tile_row`/`tile_col` folded into `page`; `leading` folded | input dtype (dataflow-only payload, never in DEST) | reader | writer | bounce kernels, whole call |
| `cb_bounce_stage` | 1 | `page_size` | `bounce_stage_pages` (= `bounce_block_pages`; 0 = not allocated) | one block of per-bank runs; allocated only for a staged read (interleaved or one-page-wide sharded source, page under 1024 B) | input dtype | reader | reader (L1 reorder) | bounce kernels, whole call |

Pull and push regimes allocate no CB: the target (pull) or source (push) buffer is the mover's own L1 and is addressed directly. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `own_block` | — | no (raw: accessor + firmware bank table / RT arg) | — | — | — |
| 2 | `pull_block` | owned target shard or bank stripe | no (raw dataflow) | — | — (writes land in the output buffer) | — |
| 3 | `push_block` | owned source shard or bank stripe | no (raw dataflow) | — (reads the resident input buffer) | — | — |
| 4 | `drain_block` | the block | no | — | — | all transactions complete (BB-03) |
| 5 | `load_block` | `bounce_block_pages` pages | no (raw dataflow) | — | `cb_bounce_pages`, `bounce_block_pages` | +`bounce_block_pages` filled |
| 6 | `store_block` | `bounce_block_pages` pages | no (raw dataflow) | `cb_bounce_pages`, `bounce_block_pages`, filled | — | slot freed after `noc_async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all | raw_api | `TensorAccessor` ctor + `TensorAccessorArgs` (two arg blocks chained by `next_compile_time_args_offset()`) | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:97-114` (sharded), `:516-530` (interleaved) | `(args, base_addr, page_size)`; base addresses are common RT args | — | — | `page_size` |
| `own_block` (shard) | raw_api | `TensorAccessor::shard_pages(shard_id)` | `tensor_accessor.h:331-339`; skips slots beyond the tensor: `shard_pages_address_iterator.h:38-50` | `owned_shard_id` | — | — | the shard is the block |
| `own_block` (bank) | raw_api | interleaved `get_noc_addr(b)` vs own node bits (`get_noc_addr(0)`), `NOC_UNICAST_ADDR_X/Y` | `tensor_accessor.h:554-556`; `noc_parameters.h:221,314` (BH) | scan bound `min(NUM_L1_BANKS, P)` | — | — | — |
| `pull_block`/`push_block` (bank walk) | raw_api | `tensor_accessor::Pages(accessor, b, P, nb)` strided iterator | `pages_address_iterator.h:284-300` | `start=b`, `stride=nb` | — | — | stripe length |
| `pull_block`, `load_block` | raw_api | `noc_async_read` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `(src_noc_addr, local_l1_addr, page_size)` | — | — | — |
| `push_block`, `store_block` | raw_api | `noc_async_write` | `dataflow_api.h:876` | `(local_l1_addr, dst_noc_addr, page_size)` | — | — | — |
| `drain_block`, `load_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | — | — | — | — |
| `drain_block`, `store_block` | raw_api | `noc_async_write_barrier`, `noc_async_writes_flushed` | `dataflow_api.h:1951`, `:1980` | — | — | — | — |
| `load_block`/`store_block` | raw_api | `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`, `get_write_ptr`, `get_read_ptr` | `dataflow_api.h:406`, `:208`, `:479`, `:260`, `:325`, `:346` | count = `bounce_block_pages` | `cb_bounce_pages` | `cb_bounce_pages` | `bounce_block_pages` |
| host | raw_api | `ttnn.get_optimal_worker_cores_for_sharded_tensor` (owner cross-check), `ttnn.split_work_to_cores`, `ttnn.reports.get_device_info`, `ttnn.TensorAccessorArgs` | `ttnn/cpp/ttnn-nanobind/pytensor.cpp:1919`, `operations/core.cpp:471`, `reports.cpp:92`, `tensor_accessor_args.cpp:15` | — | — | — | — |

Helpers considered and rejected (every row above is `raw_api`):

| Candidate | File:line of the mismatch | Why it cannot express the planned shape |
|---|---|---|
| compute helpers (`tilize`, `untilize`, `reduce`, `eltwise_*`, `matmul_block`, …) | `ttnn/cpp/ttnn/kernel_lib/` (whole compute family) | the op has no compute phase: a page that went through unpack/pack would no longer be bit-exact for block formats (spec), and a byte copy needs no TRISC at all |
| `local_copy_helpers_dataflow.hpp` `local_addr` / `set_read_state` / `read_with_state` | `local_copy_helpers_dataflow.hpp:57,75,95` | self-aimed reads only (the source is this core's L1); our sources are remote DRAM / remote L1 in every regime but the same-core special case |
| `set_read_trid` / `async_read_barrier_with_trid` | `local_copy_helpers_dataflow.hpp:147,158` | not needed by the Phase 0 bounce (two large blocks); it is the measured alternative of lamp L6 |
| `mcast` (`McastArgs`, `ttnn.Mcast`) | `ttnn/cpp/ttnn/kernel_lib/mcast/` | every page has exactly one destination; there is no reuse-shared operand to broadcast |
| `tilize_helpers_dataflow.hpp` (`read_sticks_for_tilize`) | `tilize_helpers_dataflow.hpp` | row-major sticks; both sides here are TILE pages |

No compute kernel exists, so `compute_kernel_hw_startup()` does not apply. No missing helper blocks the plan: each block operation is a loop of single NoC calls inside the block it owns, with one barrier per block.

## Host plan, cache and reporting

| Concern | Decision |
|---|---|
| Entry point | `g6_to_memory_config(input_tensor, memory_config)`: `validate()` first line → allocate the output → plan (cached) → patch addresses → `ttnn.generic_op([input_tensor, output], descriptor)` (output last). Exactly one program. |
| Plan | `{regime, owner_cores, kernel_core_range, roles: {(x, y): "pull_owner k" / "push_owner k" / "bank_walker" / "bounce [start, start+n)" / "idle"}, bounce_block_pages, bounce_depth, page_size, P}` — derived from configs and shapes only, no allocation, so `plan_of` and the op share it |
| Cache key (BB-07) | `(device id, padded shape, dtype, input MemoryConfig incl. shard spec + range order, output MemoryConfig incl. shard spec + range order)`; value: plan + `ProgramDescriptor` (CT args, per-core RT args, CB) + `custom_program_hash` set from the key. A repeat call rebuilds nothing; it patches only the two base addresses (common RT args of every kernel). The owner cross-check against `get_optimal_worker_cores_for_sharded_tensor` runs once per key, at build. |
| What is CT | `page_size`, `P`, regime/walk flags, `movers_per_owner`, both `TensorAccessorArgs` blocks (placement-dependent, so part of the key) |
| What is RT | per core: `owned_shard_id` (shard regimes) or `(start_page, core_pages)` (bounce); common: `src_base_addr`, `dst_base_addr` |
| Trace (BB-16) | the only allocation is the output tensor; no kernel build after the first (eager) call for a key |
| Plan reporting (BB-08) | `plan_of(input_tensor, memory_config)` returns the plan dict above (keys `regime`, `movers`, `kernel_cores`, `roles`, …); `movers` lists `owner_cores` in owned-shard order for shard regimes and every grid core for bank/bounce regimes |
| Zones (BB-21) | one named zone per piece of work and per wait, under a define that is off in the plain build: `G6_OWN_SCAN`, `G6_PULL_ISSUE`, `G6_PULL_DRAIN`, `G6_PUSH_ISSUE`, `G6_PUSH_DRAIN`, `G6_BOUNCE_RD_WAIT_FREE`, `G6_BOUNCE_RD_ISSUE`, `G6_BOUNCE_RD_DRAIN`, `G6_BOUNCE_WR_WAIT_DATA`, `G6_BOUNCE_WR_ISSUE`, `G6_BOUNCE_WR_FLUSH`, `G6_BOUNCE_WR_DRAIN`; the role dump is `plan_of(...)["roles"]` |
| Kernel size (BB-04/BB-18) | four small dataflow kernels (pull, push, bounce reader, bounce writer) or two with a CT walk flag; no size-optimising build flag; per-core RT args ≤ 2 words |

## Acceptance test

`tests/ttnn/unit_tests/operations/g6_to_memory_config/test_g6_to_memory_config.py` (immutable spec). Exact equality (`helpers.TOLERANCES` is `("exact", None)` for every dtype). Production cases only (BB-22), plus the cells the rules require:

| Test | Cases | Asserts |
|---|---|---|
| `test_production_case` | Qwen 2058, 2154, 2067, 2064, 2157, FLUX 313 (`pull_shard`); Qwen 2078, 2105, 2089 (`pull_bank`); Qwen 2071, 2065, 2073, 2061, 2155, 2062, FLUX 314 (`push_shard`) | regime pinned, owner count, shape/dtype/layout, exact placement (grid range order included), bit-exact values |
| `test_supported_cell_regimes` | L1 interleaved → DRAM interleaved on Qwen 2064's input (`push_bank`); DRAM interleaved → DRAM interleaved on Qwen 2077's shape (`bounce`) | regime pinned, placement, bit-exact values |
| `test_repeat_call_cache_and_determinism` | Qwen 2104 shape (`pull_bank`), 2071 (`push_shard`), 2067 (`pull_shard`) | second call on a new input at a new address adds no program-cache entry and is exact; a repeat on the first input is bit-identical |
| `test_block_format_case` | Panoptic 11 bf8b (`push_shard`), Qwen 2178 bf4b (`bounce`, DRAM width-sharded target) | regime pinned, page size from the buffer, placement, bit-exact values |
| `test_refused_cells` | Panoptic 11 placements in float32 (outside TARGET) | `UnsupportedAxisValue` (no `ExcludedCell` remains: EXCLUSIONS is empty) |
| `test_validation_errors` | host tensor, ROW_MAJOR, col-major target shard spec, col-major input shard spec, shard shape not ×32, too few grid cores, rank 5 | `ValueError` / `RuntimeError` |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Owner order ≠ the grid as written | block sharding trims the grid to the shard grid from its start (`buffer_distribution_spec.cpp:195-200`); a grid with more cores than shards leaves the tail empty (Qwen 2065: 36 cores, 34 shards) | `owner_cores` replicates `compute_core_list` and is cross-checked once per key against `get_optimal_worker_cores_for_sharded_tensor` |
| Pull into a non-local slot | `noc_async_read` can only land in the issuer's L1; a wrong `owned_shard_id` or bank id silently reads into another tensor's bytes | shard ids from the cross-checked owner list; bank id found by matching the page's own NoC node bits; `--dev` ASSERT that the derived target address is local |
| Stale addresses on a program-cache hit | the descriptor is cached; only base addresses change per call | addresses are common RT args patched on every call; acceptance test calls twice with a different input and checks exactness and the cache-entry count |
| `buffer_num_pages()` on a sharded buffer | can count padded shard slots, not `P` | `P` from the padded shape, per image (`ceil(H/32)`) |
| Bank regimes on a device whose L1 banks do not match the compute grid | a bank on an unlaunched core keeps its pages unmoved | `bank_cores_cover_grid` predicate (cached per device) → `bounce`, reported by `plan_of` |
| A write left in flight | the next program starts right away (BB-03) | every push kernel and the bounce writer end with `noc_async_write_barrier`; every pull kernel with `noc_async_read_barrier` |
| H not a multiple of 32 (67 of 105 cases) | the last tile row of each image is partial | nothing to do: whole pages move, padding is copied as stored; `P` uses `ceil` per image |
| Block formats (refinement) | any unpack/repack or partial-page move changes shared exponents | pages move whole as bytes; `page_size` (576 / 1088 B) is the only dtype-dependent quantity and is a 64 B multiple |

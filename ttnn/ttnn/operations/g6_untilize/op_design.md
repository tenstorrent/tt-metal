# Operation Design: g6_untilize

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (pure NoC byte move; no compute kernel, no arithmetic) |
| Goal | Write one DRAM-interleaved uint32 TILE tensor into a newly allocated DRAM-interleaved uint32 ROW_MAJOR tensor of the same shape, every logical element bit-identical. This runs as one device program in which every byte travels DRAM → L1 → DRAM through the NoC and never enters an engine register. |
| Math | `out[b, r, w] = x[b, r, w]` bit for bit (math.md M1–M4). In storage terms (math.md F1): output page `ρ = b·H + r` is the concatenation, over tile columns `c = 0..Wt-1`, of face row `r mod 16` of faces `2·((r mod 32) div 16)` and `2·((r mod 32) div 16) + 1` of input tile `(b·Ht + r div 32)·Wt + c`, cut to the first `W` values. |
| References | `docs/explorations/g6_untilize/` (DOSSIER.md, math.md, tensix.md, risks.md, lessons.md), `eval/golden_tests/g6_untilize/{feature_spec.py, helpers.py}`, `.claude/references/{blocking-model.md, l1-footprint-discipline.md}`, catalog `ttnn/ttnn/operations/examples/master.md` (entries listed under *Catalog evidence*) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | device tensor, rank 4, uint32, TILE, DRAM interleaved | — | buffer address → common RT arg |
| `use_multicore` | `bool`, keyword-only | no | `True` only; any other value → `ValueError` | `True` | host only; decides nothing (not the core count) |
| `sub_core_grids` | `ttnn.CoreRangeSet`, keyword-only | no | `None` only; any other value → `ValueError` | `None` | host only; decides nothing |

There is no dtype, memory_config, output-tensor or compute-config keyword.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 4 `(d0, d1, H, W)`; production `(1, 1, 32, 128)`. No shape is refused (no size/alignment axis may exist, spec Rules). |
| Dtype | uint32 |
| Layout | TILE (32×32 tiles of four 16×16 faces, face order TL, TR, BL, BR, row-major inside a face) |
| Memory | DRAM interleaved (sharded or L1 → `ValueError`) |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `input_tensor.dtype` (uint32) |
| Layout | ROW_MAJOR: one page per row `ρ = b·H + r`, `W·4` bytes |
| Memory | `input_tensor.memory_config()` (DRAM interleaved), new buffer allocated by the op at call time (at capture under trace), never shared with the input |

## Registry Contract

| Axis | TARGET (feature_spec.py) | Phase 0 SUPPORTED | Tagger (`INPUT_TAGGERS`) |
|------|--------------------------|-------------------|--------------------------|
| `dtype` | `[ttnn.uint32]` | `[ttnn.uint32]` | `tag_dtype(inputs, axes) -> inputs[0]["dtype"]` |
| `layout` | `[ttnn.TILE_LAYOUT]` | `[ttnn.TILE_LAYOUT]` | `tag_layout(inputs, axes) -> inputs[0]["layout"]` |

`inputs` is `(scenario,)` with `scenario = {"input_shape": tuple(input_tensor.shape), "dtype": input_tensor.dtype, "layout": input_tensor.layout}`. These are the only two taggers. EXCLUSIONS: none. INVALID: none (feature_spec.py). Structural impossibilities to add: none. TARGET has no `memory_layout` axis, so the logical row split is never exposed as a physical shard.

`validate(input_tensor, *, use_multicore, sub_core_grids)` is the first line of `g6_untilize` and runs these checks in order:

| # | Check | Raises | Message must contain (acceptance test matches it) |
|---|-------|--------|---------------------------------------------------|
| 1 | `use_multicore is True` | `ValueError` | `use_multicore` |
| 2 | `sub_core_grids is None` | `ValueError` | `sub_core_grids` |
| 3 | `ttnn.is_tensor_storage_on_device(input_tensor)` | `ValueError` | `device` (any case) |
| 4 | SUPPORTED per axis over the tagged scenario | `UnsupportedAxisValue` | free |
| 5 | EXCLUSIONS (empty) | `ExcludedCell` | free |
| 6 | `len(input_tensor.shape) == 4` | `ValueError` | `rank` (any case) |
| 7 | `memory_config().memory_layout == INTERLEAVED` and `buffer_type == DRAM` | `ValueError` | `DRAM interleaved` (any case) |

## Blocking Model

The op is one tile row of four tiles (production): 16384 B in, 16384 B out, 0 arithmetic. Per-call fixed cost dominates (risks.md R4), so the blocking decides how many cores share the 256 face-row moves and how many serial round trips each core pays. Bytes per tier are identical for every legal split.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `row`: output page index `ρ = b·H + r`, extent `total_rows = B·H` (leading dims B folded in) | **independent**: each `out[ρ, :]` reads only its own face-row pieces (math.md, *Axes and dependence*). B folds into it because rows of different images are distinct pages, addressed by one per-image formula (`tile_row = (ρ div H)·Ht + (ρ mod H) div 32`). | `block_rows` | **1**. This is the core's whole assignment: 32 rows over 32 cores. One row is the coarsest block that keeps one whole output page per core *and* fills 32 cores; a larger block means fewer cores (grid-sync lamp). | `plan()` on host → CT arg | Balanced contiguous runs over the `cx × cy` rectangle: core `i` owns rows `[i·q + min(i, rem), … + q + (i < rem))` with `q = total_rows div C`, `rem = total_rows mod C`. **Phase 0: 8×4 rectangle, 1 row per core.** | knob-turn (rows per core, `block_rows`) |
| `tile_col`: W in tile columns, extent `Wt = ceil(W/32)` | **independent**: no result spans W | `block_tile_cols` | **Wt = 4** (the whole row, so one output page per write) | `plan()` → CT arg | **Not split across cores.** Each output page is written whole by one core; a cut fragments every page into `Wt / block_tile_cols` partial writes (traffic ranking). Inside a core it is cut only when one staged row exceeds the L1 clamp (mechanism caps). | knob-turn (half-row blocks on 64 cores: perf lamp) |

The intra-tile structure (face band, face column, face row of 16 values) is **not a block axis**. It is the transport unit: one face row (`face_row_bytes = 16·elem = 64 B`) is the longest run contiguous in both x and out (math.md F1, mechanism caps). Every block is cut along `row` and `tile_col` only.

Re-running the table on the scheme's one intermediate (the staged rows in `cb_row_staging`) gives the same axes with the same assignment, because each core stages exactly the rows it writes. No core collects, coordinates or hands out work: every core of the rectangle has the same single role.

Operand-reuse check (primary split = `row` across cores): `x` varies along `row` (each row reads disjoint face-row pieces), so no core re-reads another core's bytes and nothing is reuse-shared. There is no broadcast row.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_row_staging` | `staging_depth` | **1** | At `num_blocks_this_core > 1`, depth 2 lets `gather_block(k+1)` land in the other slot while `store_block(k)`'s writes are still in flight. Phase 0 has one block per core, so a second slot would hold nothing. `plan()` sets `staging_depth = 1 if max_blocks_per_core == 1 else 2`. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| BH DRAM read alignment: `NOC_DRAM_READ_ALIGNMENT_BYTES = 64` (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`); the L1 address must equal the NoC address mod alignment (`tt_metal/hw/inc/internal/debug/sanitize.h:746`) | L1 landing address of every face-row piece | Staging base is 64 B-aligned because CBs are allocated at DRAM alignment (`tt_metal/impl/program/program.cpp:2504-2508`). `staged_row_bytes = block_tile_cols·tile_w·elem` is a multiple of 128, and piece offsets `(c−c0)·tile_w·elem + f·face_row_bytes` are multiples of 64, matching the source offsets `(2·band+f)·face_bytes + fr·face_row_bytes` (multiples of 64). | A row stride of `W·elem` (non-aligned W) or a hand-placed buffer lands pieces off-alignment: corrupted bytes, or a watcher NoC-alignment fault under `--dev`. |
| Contiguous-run cap (math.md F1; tensix.md *Mechanism caps*) | size of one read | Every read is exactly `face_row_bytes` (64 B). Pieces wholly beyond W (`32·c + 16·f ≥ W`) are skipped. A piece straddling W is read whole (its tail is tile padding inside the tile page) and never written out. | A longer read pulls face row `fr+1` (or the next face) into row `r`'s slot, giving wrong values in a fixed pattern (risks.md R2). |
| Single-packet read: `NOC_MAX_BURST_SIZE` (`noc_parameters.h:290`) | read size | 64 B ≤ burst always, so reads use `noc_async_read_one_packet` | n/a (cap never reached) |
| Store size vs burst | write size per row | `page_bytes = W·elem`: one write per row via `noc_async_write`, which splits above the burst | n/a |
| Model L1 bound, BB-05: 1347776 B per core (`L1_BUDGET`) | `staging_depth × block_rows × staged_row_bytes` | `plan()`: `block_rows ← min(rows_this_core_max, floor(L1_BUDGET / (staging_depth·Wt·tile_w·elem)))`. If that is 0, then `block_rows = 1` and `block_tile_cols ← floor(L1_BUDGET / (staging_depth·tile_w·elem))`. Phase 0 uses 512 B. | A shape with a very wide W fails to launch on CB allocation. |
| Kernel-group rectangle, BB-17 | `cx`, `cy` | `cx ≤ grid_x`, `cy ≤ grid_y` of `device.compute_with_storage_grid_size()` | a non-rectangular or out-of-grid core set |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 `gather_rows`**: face-row pieces read from DRAM straight to their row-major position in L1, then whole pages written | **built** (the only built regime, every shape) | always; the regime is selected for every input, so no selection predicate exists | `block_rows × block_tile_cols` (Phase 0: `1 × 4`) | **Minimum on bytes.** Boundary is DRAM: x is read once (16384 B), out is written once (16384 B), 0 cross-core, 0 re-reads. Transfer shape: x as `2·Wt` reads of 64 B per row (256 device-wide), out as one write of `page_bytes` per row (32 device-wide). The 64 B read granularity is structural: no run longer than one face row is contiguous in both layouts (F1). Serial round trips per core: 2 (DRAM read, DRAM write). | Per kernel, paid once regardless of block: launch, two `TensorAccessor` constructions, core-index derivation from logical coordinates. Per block, intended once: one read barrier (one DRAM round trip), `block_tile_cols` tile-page address lookups shared by all rows of the block in one tile row, one `writes_flushed` before slot reuse. A larger `block_rows` spreads the read round trip over more rows, but Phase 0's block already equals the core's whole assignment, so a bigger block needs fewer cores (grid-sync lamp). |
| **R2 `scatter_pieces`**: input-contiguous runs (`block_rows` face rows of one face) read whole, each 64 B piece written to its row's DRAM address | **deferred**: R1 already serves every shape R2 would. | — | `block_rows` rows × one face column of one tile (input-contiguous `block_rows·64 B`) | Same bytes per tier (delta 0 B DRAM, 0 B cross-core) and the same device-wide transaction count (32 reads of 512 B + 256 writes of 64 B, against R1's 256 + 32). The 256 small transfers become writes spread over all DRAM banks of the output pages, where R1 concentrates them as reads on the Wt banks holding the tile row. Every output page is assembled from `2·Wt` writes by `2·Wt` cores. | same terms as R1, plus per-page write acks (`2·Wt` per page) |
| **R3 `tile_read_local_reorder`**: whole tiles (or face pairs) read contiguous, reordered L1→L1 by self-aimed NoC reads or RISC-V copies, whole pages written | **rejected**: superseded by R1 | — | whole tile(s) per core | Same bytes. DRAM transactions fall to `B·Ht·Wt` reads only when cores own whole tiles (at most 4 cores here), which serializes 64 local moves per tile on each core. It also adds a third serial hop (local round trip or word copies) to reach the order R1 gets in transit. | — |
| **R4 `engine_untilize`**: compute kernel, unpack tiles to DEST, `pack_untilize` to rows | **rejected**: superseded by R1 | — | Wt tiles | Same DRAM traffic as R1 plus a compute kernel (3 TRISC binaries, HW startup and untilize init every call), added fixed cost on a call whose cost is fixed cost. The helper path unpacks through SrcA (`untilize_helpers.inl:179`), where UInt32 is illegal (`cunpack_common.h:472-473`, UInt32 only valid unpacked to DEST). A DEST-direct path needs fp32 DEST, and whether the packer keeps all 32 bits is unverified (risks.md R1). The spec forbids any path that holds fewer than 32 bits. | — |

Single built regime, its domain and why no other applies: R1 covers every rank-4 uint32 TILE DRAM-interleaved input. R2 ties it on paper and needs nothing R1 lacks. R3 and R4 cost strictly more fixed cost or round trips for the same bytes.

### Traffic ranking

All candidate splits move the same bytes: each tensor crosses DRAM exactly once, nothing crosses between cores, and nothing is re-read. The ranking therefore falls to transfer shape: transactions on each core's critical path and serial round trips.

| Candidate split | DRAM crossings x / out | Cross-core | Device-wide transfers | Per core | Serial round trips | Verdict |
|-----------------|------------------------|------------|-----------------------|----------|--------------------|---------|
| `row` across cores, gather-on-read (R1), 32 cores × 1 row | 1 / 1 | 0 | 256 R ×64 B + 32 W ×512 B | 8 R + 1 W | 2 | **chosen** |
| input-contiguous runs across cores, scatter-on-write (R2), 32 cores | 1 / 1 | 0 | 32 R ×512 B + 256 W ×64 B | 1 R + 8 W | 2 | ties R1; deferred (R2 row) |
| `row` × `tile_col` halves (R1, `block_tile_cols = 2`), 64 cores | 1 / 1 | 0 | 256 R + 64 W ×256 B | 4 R + 1 W | 2 | fragments every output page into 2 writes; grid-sync lamp |
| `tile_col` across cores (whole tile per core), 4 cores | 1 / 1 | 0 | 4 R ×4096 B + 256 W ×64 B (or 256 local + 128 W ×128 B) | 1 R + 64 W | 2–3 | rejected: serializes 64 piece moves per core on 4 cores |
| everything on one core | 1 / 1 | 0 | 256 R + 32 W | 256 R + 32 W | 2 | rejected: single-core |
| split a dependent axis + combine | n/a | n/a | n/a | n/a | n/a | no dependent axis exists (math.md: no result spans any axis) |
| broadcast a shared operand | n/a | n/a | n/a | n/a | n/a | no operand is reuse-shared under any split |

Chosen: R1 on `row`. The occupancy step comes second: within whole-page blocks the work units are the `total_rows = 32` rows, so occupancy is 32 cores at one row each. That puts the coarsest block at the core's whole assignment.

### Block schedule

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    gather_block(block_idx);   // DRAM tile pages -> block's rows, row-major, in staging slot (block_idx % staging_depth)
    store_block(block_idx);    // staging slot rows -> output pages, one write per row
}
drain();                       // every issued write acked before return (BB-03)
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `gather_block` | `block_rows × block_tile_cols` (Phase 0: 1 × 4) | staging slot `block_idx % staging_depth` (`block_rows` staged rows) | `2·block_tile_cols` 64 B single-packet reads per row (Phase 0: 8). Tile-page NoC address computed once per (block, tile column, tile row touched). **One `noc_async_read_barrier` per block** (Phase 0: one per kernel). Before issuing into a slot whose previous store may be in flight: one `noc_async_writes_flushed` (only when `num_blocks_this_core > staging_depth`). Tile order rotated per core: start at tile column `core_linear_index mod block_tile_cols` (bank-stagger lamp). |
| `store_block` | same block | same slot, read by the NoC write engine | one `noc_async_write` of `min(block_tile_cols·tile_w·elem, page_bytes − c0·tile_w·elem)` bytes per row, at page `ρ`, byte offset `c0·tile_w·elem` (Phase 0: 1 write of 512 B at offset 0). The output-page NoC address is computed **before** the gather's read barrier (stall shadow). No barrier per block. |
| `drain` | — | — | **one `noc_async_write_barrier` per kernel** (non-posted writes: BB-03 requires every write acked before the kernel returns) |

Kernel-scope inits, once per kernel: input and output `TensorAccessor` construction (common RT args 0 and 1), and the core linear index `i = (get_absolute_logical_x() − x0) + (get_absolute_logical_y() − y0)·cx`. No compute init exists.

Stall-shadow check: the waits are the gather's read barrier (a DRAM round trip on data the core does not produce) and the final write barrier.

| Wait | Work independent of it, scheduled into it |
|------|-------------------------------------------|
| read barrier, block `k` | Output-page NoC address(es) of block `k` (`TensorAccessor::get_noc_addr(ρ)`), computed before the barrier. At `num_blocks_this_core > 1` (depth 2): `store_block(k)`'s writes drain in the shadow of block `k+1`'s read barrier, because the issue order is gather(k) → wait(k) → store(k) → gather(k+1) → wait(k+1). Phase 0: nothing else is independent: the single store needs all 8 pieces. |
| final write barrier | nothing: it is the last work of the kernel. |

No reorder changes floating-point order (there is no arithmetic).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Grid synchronization**: 32 cores × 1 row (default) | Each core does 8 reads + 1 write, so per-core launch, prologue and two DRAM round trips dominate. Fewer cores may end sooner overall, and more cores shave read issues. | 16 cores × 2 rows (`block_rows = 2`, 8×2) and 8 cores × 4 rows. 64 cores × half row (`block_tile_cols = 2`, 8×8: 4 R + 1 W ×256 B). Judge on unprofiled host time of blocking trace replays plus the profiled kernel duration (lessons.md RB-053: cores ending at different times bias the profiler). |
| **Overlap**: one block per core with the store after the whole gather (default) | Phase 0 cannot overlap read and write inside a core (1 block). Overlap exists only when a core owns more than one block. | Together with the 16-core point: `block_rows = 1`, 2 blocks per core, `staging_depth = 2`, against `block_rows = 2`, 1 block. |
| **Bank stagger**: each core's tile order starts at `core_linear_index mod block_tile_cols` (default) | All 256 reads hit the `Wt = 4` banks of one tile row. Without rotation every core's first reads hit bank 0 at once (catalog `bank_stagger`). With so few reads per bank the rotation may buy nothing. | Unrotated order (`c = 0..Wt-1` on every core), a compile-time switch. |
| **Read issue mechanism**: one `noc_async_read_one_packet` per piece (default) | Two pieces per tile share one DRAM endpoint and size, so the per-read register writes may be measurable at 8 reads per core. | `noc_async_read_one_packet_set_state` once per tile column + `noc_async_read_one_packet_with_state` per piece (`dataflow_api.h:575`, `:599`). |
| **RISC / NoC placement**: one kernel on RISCV_0 (BRISC), reads and writes on NOC_0 (default) | BRISC needs no subordinate launch. Writes on NOC_0 go against the catalog's reads-NoC0 / writes-NoC1 preference, which was measured on bandwidth-heavy lines. | (a) same kernel on RISCV_1/NOC_0; (b) `noc_mode = DM_DYNAMIC_NOC` with writes on NOC_1; (c) split the 8 reads across NCRISC and BRISC (catalog `split_reader`), 4 each, with a local handshake before the write. |
| **Rectangle placement**: 8 wide × 4 tall at logical (0, 0) (default; row-wise per catalog `noc_placement`) | DRAM endpoints sit at the grid edges, so distance to the 4 input banks differs by placement. | 4×8 at (0, 0); an 8×4 origin moved toward the DRAM column holding banks 0–3. |

### Catalog evidence

| Entry | Informed |
|-------|----------|
| `noc_placement` | Row-wise rectangle (`cx ≥ cy`); reads on NOC_0. The writes-on-NOC_1 alternative is a lamp. |
| `width_split` | The one-tile-row trap: splitting by *tile rows* strands the op on one core. The split is by *rows inside the tile row* (sub-tile granularity), which only a pure byte move allows. |
| `double_buffer` | `staging_depth` knob (Phase 0 = 1: one block per core, nothing to overlap). Reads issued as a batch with one barrier per block, never read-one/barrier. |
| `bank_stagger` | Per-core rotated tile order (default) and its lamp. |
| `tile_reorder` | Its "move whole pages on a DRAM-bound move" is why output pages are written whole. The 64 B reads are forced by F1, and the op is latency-bound, not bandwidth-bound. |
| `split_reader`, `trid_double_issue` | Lamps only: one block per core leaves no barrier to hide and no issue-bound RISC is demonstrated. |
| lessons.md RB-036 / RB-053 | One rectangular core range for the single kernel group; measure core-count lamps on trace replay host time. |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x in DRAM | TILE pages of `tile_bytes = 1024·elem` (4096 B), page `t` in bank `t mod num_banks` | — | production: 4 tiles in banks 0..3 |
| gather | face-row pieces of `face_row_bytes` (64 B) | `noc_async_read_one_packet` on NOC_0 from `in.get_noc_addr(tile, (2·band+f)·face_bytes + fr·face_row_bytes)` to `slot + i·staged_row_bytes + (c−c0)·tile_w·elem + f·face_row_bytes` | The tile→row reorder happens in transit. Tiles are never staged in L1. |
| L1 staging | row-major rows at stride `staged_row_bytes = block_tile_cols·tile_w·elem` (512 B) | `cb_row_staging`, scratch of the single kernel | The tail of each staged row past `W·elem` holds tile padding (non-aligned W only) and is never written out. |
| store | ROW_MAJOR pages of `page_bytes = W·elem` (512 B) | `noc_async_write` on NOC_0 to `out.get_noc_addr(ρ, c0·tile_w·elem)` | One write per row; each page has exactly one writer. |
| out in DRAM | ROW_MAJOR pages, page `ρ` in bank `ρ mod num_banks` | — | new buffer |

No built or deferred regime communicates between Tensix cores: there are no semaphores, multicast or rings. The only placement in TARGET is DRAM interleaved, so there is no physical-shard path. Each bank address is device-local, so on the (1, 4) mesh the same program run on each device reads that device's input and writes that device's output (SPMD `generic_op`).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_rows` consecutive output rows × `block_tile_cols` tile columns |
| Grid | One rectangle `CoreRange((x0, y0), (x0+cx−1, y0+cy−1))`, `(x0, y0) = (0, 0)`. `C_target = min(total_rows, grid_x·grid_y)`. Choose `(cx, cy)` with `cx ≤ grid_x`, `cy ≤ grid_y`, `cx·cy ≥ C_target`, minimal `cx·cy`; ties go to larger `cx`. **Production: 8×4 = 32 cores, 1 core range.** |
| Per-core work | Core `i = (lx − x0) + (ly − y0)·cx` owns rows `[start_i, start_i + n_i)` with `q = total_rows div (cx·cy)`, `rem = total_rows mod (cx·cy)`, `start_i = i·q + min(i, rem)`, `n_i = q + (i < rem)`. `num_blocks_this_core = ceil(n_i / block_rows)`; the last block has runtime extent `n_i − (num_blocks_this_core − 1)·block_rows`. A core with `n_i = 0` returns at once. Production: `n_i = 1`, one block. |
| Index derivation | No per-core RT args. The kernel derives `i` from `get_absolute_logical_x/y()` and CT args `x0, y0, cx`. Both buffer addresses are **common** RT args (multicast once per kernel group, patched in O(1) on cache hits). |
| Geometry (alignment-aware, per image) | `Ht = ceil(H/32)`, `Wt = ceil(W/32)`, `total_rows = B·H` (not `B·Ht·32`). Row `ρ`: `b = ρ div H`, `r = ρ mod H`, `tile_row = b·Ht + r div 32`, `band = (r mod 32) div 16`, `fr = r mod 16`. Tile id `= tile_row·Wt + c`. `page_bytes = W·elem`; pieces with `32·c + 16·f ≥ W` are skipped. |
| Remainder | balanced `q`/`rem` split above. Ragged block extents are runtime values; no CB handshake exists, so there are no nominal push/pop counts to keep. |

There is one regime, so there is no regime selection. `plan()` is the single source of every knob. Its production output is pinned by the acceptance test through bit-exact values and by `describe_plan()` (below).

## Host Entry Point and Program Cache

| Concern | Decision |
|---------|----------|
| Dispatch | Exactly one `ttnn.generic_op([input_tensor, output_tensor], program_descriptor)` per call, output last. No other ttnn op is called: no layout or memory conversion, no host readback (BB-01, BB-02). |
| Output allocation | `ttnn.allocate_tensor_on_device(ttnn.Shape(shape), input.dtype, ttnn.ROW_MAJOR_LAYOUT, input.device(), input.memory_config())` inside the call. It is the only allocation; under trace it happens at capture (BB-16). |
| Descriptor cache (BB-07) | Module-level dict keyed by `(tuple(shape), dtype, layout, memory_config, device grid size, mesh shape)`. On a hit, the cached `ProgramDescriptor` is reused and only the kernel's `common_runtime_args = [input.buffer_address(), output.buffer_address()]` is replaced. `generic_op`'s hash excludes RT-arg values (`generic_op_device_operation.cpp:83-102`, sizes only), so a hit replays the cached program. Every CT arg derives from the key. |
| `plan()` | Pure host function of `(shape, elem, grid)`. It returns `x0, y0, cx, cy, block_rows, block_tile_cols, staging_depth, rows per core` and is the single source of every knob. |
| Plan report (BB-08) and role dump (BB-21) | `describe_plan(input_tensor) -> dict`: regime name (`"gather_rows"`), rectangle, knob values, and per core `{(x, y): ("gather_rows", start_row, num_rows)}`. Exported from the op module. |
| Kernel | One data-movement kernel `untilize_rows` on `DataMovementConfigDescriptor(processor=RISCV_0, noc=NOC_0)` over the rectangle. No compute kernel, no semaphores, one CB. |
| Profiler zones (BB-21) | `DeviceZoneScopedN` around each of: gather issue, gather wait (read barrier), store issue, drain wait (write barrier). The macro compiles out of the plain build (`tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1136`, the non-profiler branch). |
| Mesh | `generic_op` with a single `ProgramDescriptor` is SPMD (`ttnn/cpp/ttnn/operations/generic/generic_op.hpp:33`): every device runs the program on its own buffers, which share one address because mesh buffers are allocated in lockstep. |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_row_staging` | 0 | `staged_row_bytes = block_tile_cols · tile_w · elem` (Phase 0: 4·32·4 = 512 B) | `staging_depth · block_rows` (Phase 0: 1) | Spans `row` (`block_rows` rows simultaneously staged per slot) and `tile_col` (page width = `block_tile_cols` tile columns); B is folded into `row`. Streams over nothing: the block is the core's whole assignment. | `ttnn.uint32` (raw bytes; never enters DEST) | `untilize_rows` (NoC reads land in it) | `untilize_rows` (NoC write reads from it) | whole kernel |

`cb_row_staging` is L1 scratch owned by one kernel on one RISC. It uses **no** `cb_reserve_back / cb_push_back / cb_wait_front / cb_pop_front`: the address comes from `get_write_ptr(cb_row_staging)` and ordering comes from NoC barriers (read barrier before the store reads a slot, `writes_flushed` before a gather overwrites a slot). Push count = wait count = 0. One RISC touches it, so the single-producer/single-consumer invariant holds trivially. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `gather_block` | `block_rows × block_tile_cols` | no (raw NoC; rejected helpers below) | — (source: DRAM tile pages of x) | `cb_row_staging`, slot `block_idx % staging_depth`, `block_rows` pages | Slot holds the block's rows in row-major order after the read barrier. |
| 2 | `store_block` | `block_rows × block_tile_cols` | no | `cb_row_staging`, same slot, `block_rows` pages, complete | — (destination: DRAM pages of out) | Slot is reusable once `writes_flushed` returns. |
| 3 | `drain` | — | no | — | — | Every write acked: kernel may return (BB-03). |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (kernel scope) | raw_api | `get_absolute_logical_x()`, `get_absolute_logical_y()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:44`, `:57` | core linear index with CT `x0, y0, cx` | — | — | `cx` (rectangle) |
| (kernel scope) | raw_api | `TensorAccessor(args, addr)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71` | input (common RT 0), output (common RT 1) | — | — | — |
| `gather_block` | raw_api | `get_write_ptr` | `dataflow_api.h:325` | `cb_row_staging` base | — | `cb_row_staging` | slot = `block_idx % staging_depth` |
| `gather_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tensor_accessor.h:165` | `page_id = tile_row·Wt + c`, `offset = (2·band+f)·face_bytes + fr·face_row_bytes` | — | — | — |
| `gather_block` | raw_api | `noc_async_read_one_packet` | `dataflow_api.h:515` | `size = face_row_bytes` | — | `cb_row_staging` | piece count `2·block_tile_cols·block_rows` |
| `gather_block` | raw_api | `noc_async_writes_flushed` | `dataflow_api.h:1980` | only when `num_blocks_this_core > staging_depth` | — | — | `staging_depth` |
| `gather_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per block | — | `cb_row_staging` | — |
| `store_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tensor_accessor.h:165` | `page_id = ρ`, `offset = c0·tile_w·elem` | — | — | — |
| `store_block` | raw_api | `noc_async_write` (non-posted) | `dataflow_api.h:876` | size = row bytes of the block | `cb_row_staging` | — | `block_rows` writes |
| `drain` | raw_api | `noc_async_write_barrier` | `dataflow_api.h:1951` | once per kernel | — | — | — |
| (all) | raw_api | `DeviceZoneScopedN` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` | zone per issue/wait | — | — | — |

**Helpers considered and rejected** (applies to every raw_api row above. The op has no compute phase; the gather/store phases are dataflow):

| Candidate | File:Line of the mismatch | Why it cannot express the planned shape |
|-----------|---------------------------|------------------------------------------|
| `compute_kernel_lib::untilize` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:163`; `untilize_helpers.inl:179` (`reconfig_data_format_srca(input_dfb)`), `:119` (`pack_untilize_init` datacopy route), `:28-30` (even Float32 narrows through SrcA); `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:472-473` (UInt32 valid only unpacked to DEST) | It routes tiles through SrcA, where a uint32 operand is not representable. This is the rejected R4 and would break bit-exactness (spec rule 1). |
| `dataflow_kernel_lib::write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:106` ("Always operates at TILE granularity"); `tilize_helpers_dataflow.inl:219` (`ASSERT(width_in_tiles <= cb_capacity)`), `:229` (`cb_wait_front(cb_id, width_in_tiles)`), `:238` (`noc_async_write_barrier()` per 32-row block) | Its consume quantum is Wt tile pages: a 32-row tile row of 16384 B at Wt = 4. The planned block is `block_rows` staged rows (512 B), so it would need 32× capacity over the live set (ledger audit 1). It also needs a separate pushing producer, and its per-block full barrier rules out the depth-2 store/gather overlap. |
| `dataflow_kernel_lib::read_sticks_for_tilize` | `tilize_helpers_dataflow.hpp:88` | It reads row-major sticks into tile pages, the opposite direction; x is TILE. |
| `dataflow_kernel_lib::set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57`, `:60-75` | Self-aimed L1 source only (`local_addr`), so it cannot address a DRAM bank. It would only serve the rejected R3. |
| `index_tile_dataflow`, `tile_column_dataflow` | `index_tile_dataflow.hpp:42-61`; `tile_column_dataflow.hpp:7-25` | They generate index tiles or set a tile column with RISC-V stores. Neither moves face rows from DRAM. |

Gap record (for the helper library): no dataflow helper gathers face-row pieces of TILE pages into row-major sticks (an "untilize-on-read" reader). `gather_block` is that missing block operation.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Wrong face order (risks.md R2) | A row takes faces `2·band` and `2·band+1` of each tile at row `fr`. Treating a tile as 32 contiguous rows, or swapping faces 1 and 2, misplaces half or three quarters of the values. | Piece formula in Work Distribution; acceptance test `face_order` encodes each element's own index. |
| Any path that holds fewer than 32 bits (R1) | Small integers survive lossy paths. | No engine anywhere: NoC reads and writes only. Tests draw full-range uint32 patterns and compare bits. |
| DRAM read alignment | BH requires the L1 landing address ≡ the source address mod 64. | Staging is a CB (64 B-aligned base); row stride and piece offsets are multiples of 64 (Mechanism caps). Run once under `--dev` (watcher NoC sanitizer). |
| Logical-coordinate index derivation | Rows are mapped to cores from `get_absolute_logical_x/y()`, not RT args. A coordinate-space mismatch writes rows to the wrong pages. | CT `x0, y0, cx` are taken from the same CoreRange the kernel is placed on; the acceptance test's bit-exact check catches any mismatch. |
| Cached descriptor patch not reaching dispatch | Python accessors on `ProgramDescriptor.kernels` may return copies, so a patch can be silently dropped and stale addresses replayed (wrong output on the second call). | After patching, verify on the cached descriptor that `common_runtime_args` hold the new addresses; acceptance test `repeat` uses two different output allocations. |
| BB-03 write completion | Non-posted writes are complete only when acked; the next program may read `out`. | `drain` = `noc_async_write_barrier()` at kernel end. No posted writes. |
| Output aliasing / input mutation (R3) | — | Fresh output allocation every call. The kernel only reads x. Acceptance test `new_storage`. |
| Mesh (R6) | A single-device suite cannot see a first-device-only write. | SPMD `generic_op` with one `ProgramDescriptor`, device-local bank addresses; no per-device RT args. |

## Acceptance Test

`tests/ttnn/unit_tests/operations/g6_untilize/test_g6_untilize.py`. Scope follows BB-22, the spec's own rule: only the production shape `(1, 1, 32, 128)` plus the cases the rules require. The template default of four shapes and `torch.randn` is overridden. Inputs are full-range uint32 bit patterns (`torch.manual_seed`), the reference is `x.clone()`, and every comparison is bit-exact (`helpers.TOLERANCES[uint32] = ("exact", None)`).

| Test | Checks |
|------|--------|
| `test_production_case[model_call, defaults]` | both call patterns; output contract (device, shape, uint32, ROW_MAJOR, DRAM interleaved); bits |
| `test_face_order` | element value = its row-major index (+2^31 on odd elements) |
| `test_extremes` | 0, 2^32−1, 2^31, 2^24+1 in every row, the rest random |
| `test_new_storage_and_input_unchanged` | output buffer address ≠ input's; input read back unchanged |
| `test_repeat_is_bit_identical_and_cached` | second call bit-identical, into a new output allocation; no new program-cache entry |
| `test_rejects_keyword_values` | `use_multicore=False`, `sub_core_grids=<set>` → `ValueError` |
| `test_rejects_host_tensor`, `test_rejects_rank_other_than_4`, `test_rejects_l1_interleaved`, `test_rejects_sharded` | host tensor, rank 3, L1 interleaved, height-sharded → `ValueError`/`RuntimeError` |

The mesh and trace replay are not covered here: the module device is a single chip without a trace region. The verifier and perf pass carry BB-16 and R6.

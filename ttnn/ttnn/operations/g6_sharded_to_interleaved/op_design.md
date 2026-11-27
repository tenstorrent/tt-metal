# Operation Design: g6_sharded_to_interleaved

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Write every tile of a height- or width-sharded L1 TILE tensor, once, from the L1 of the core that holds its shard to its page of a newly allocated interleaved TILE tensor (DRAM or L1, as `memory_config` names). One `ttnn.generic_op` dispatch per call. |
| Math | `out[i] = input_tensor[i]` for every logical index i, bit for bit; shape, dtype, TILE kept; `out` is new storage, `input_tensor` unchanged. In tile terms: tile `(a, b)` of the padded 2-D view goes to output page `p = a * C + b`. |
| References | `docs/explorations/g6_sharded_to_interleaved/` (`DOSSIER.md`, `math.md` F1, `tensix.md`, `risks.md`, `lessons.md`); oracle `docs/explorations/g6_sharded_to_interleaved/oracle/pytorch_g6_sharded_to_interleaved.py`; golden suite `eval/golden_tests/g6_sharded_to_interleaved/` (`feature_spec.py`, `helpers.py`); `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`; catalog `ttnn/ttnn/operations/examples/master.md` |

## Parameters

The op has two positional arguments and no options. Every row below the first two is a parameter derived on the host from those two, with its single source.

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes (positional) | on device, TILE, HEIGHT_SHARDED or WIDTH_SHARDED in L1, shard shape whole tiles, rank 2 or 4 | — | — |
| `memory_config` | `ttnn.MemoryConfig` | yes (positional) | `memory_layout == INTERLEAVED`, buffer DRAM or L1 | — | — |
| `tensor_tile_rows` (R) | int | derived | `volume(padded_shape[:-1]) / 32` = `prod(shape[:-2]) * ceil(shape[-2] / 32)`; p2: 1, p1: 1024 | — | CT |
| `tensor_tile_cols` (C) | int | derived | `padded_shape[-1] / 32` = `ceil(shape[-1] / 32)`; p2: 160, p1: 1 | — | CT |
| `shard_tiles_h`, `shard_tiles_w` | int | derived | `shard_spec.shape[0] / 32`, `shard_spec.shape[1] / 32`; p2: 1 x 5, p1: 10 x 1 | — | CT |
| `shard_row_step`, `shard_col_step` | int | derived | height: `(shard_tiles_h, 0)`; width: `(0, shard_tiles_w)` — the one place the strategy enters the page map | — | CT |
| `num_shards` | int | derived | height: `ceil(R / shard_tiles_h)`; width: `ceil(C / shard_tiles_w)`; ≤ `shard_spec.grid.num_cores()`; p2: 32, p1: 103 | — | CT |
| `shard_core_ranges` | list of `(x0, y0, x1, y1)` | derived | `shard_spec.grid.ranges()` in stored order (the order the buffer walked, `tt_metal/impl/buffers/buffer.cpp:439-440`); p2: 1 range, p1: 2 | — | CT (count + table) |
| `orientation_row_major` | bool | derived | `shard_spec.orientation == ROW_MAJOR` | — | CT |
| `page_bytes` | int | derived | `input_tensor.buffer_aligned_page_size()`: 2048 (bfloat16), 1088 (bfloat8_b); must equal the output's | — | CT (via the output `TensorAccessorArgs` page size and the shard CB page size) |
| `num_writer_lanes` (L) | int | host constant `NUM_WRITER_LANES` | {1, 2}; Phase 0 = 2 | 2 | CT |
| `writer_lane` | int | per kernel descriptor | `[0, L)` | — | CT |
| output buffer address | uint32 | per call | `output_tensor.buffer_address()` | — | common RT (the only runtime value; refreshed every call) |

Index axes: none (no `dim` argument).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, ..., d_{r-1})`, r = 2 or 4, every d ≥ 1 |
| Dtype | bfloat16 (Phase 0), bfloat8_b (refinement) |
| Layout | TILE |
| Memory | L1, HEIGHT_SHARDED or WIDTH_SHARDED; shard shape a whole number of 32x32 tiles; height: shard width = padded W; width: shard height = padded H' (= 32·R); shard k on the k-th core of `shard_spec.grid` walked range by range in stored order, each range row-wise (ROW_MAJOR) or column-wise (COL_MAJOR); last shard may be shorter |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` (same padded shape, same R x C tiles) |
| Dtype | `input_tensor.dtype` |
| Layout | TILE |
| Memory | INTERLEAVED in `memory_config.buffer_type` (DRAM or L1); allocated by the call with `ttnn.allocate_tensor_on_device(shape, dtype, TILE, device, memory_config)` (positional) on the input's device — a MeshDevice allocates it on every chip |

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_rows` (a; leading dims folded into the 2-D view's rows — contiguous and treated identically) | independent — output tile (a, b) depends on input tile (a, b) alone (`math.md` axes table) | `block_tile_rows` | `shard_tiles_h` (whole shard; ragged last shard: runtime `rows_k = min(shard_tiles_h, R - row0_k)`) | host: `shard_spec.shape[0] / 32` → CT `shard_tiles_h` | height: shard k (rows `[k·shard_tiles_h, …)`) on its holding core; width: not split — every core holds all R rows, by the producer's shard spec | knob-turn (none needed: the block is already the whole shard) |
| `tile_cols` (b) | independent — same reason | `block_tile_cols` | `shard_tiles_w` (ragged last shard: `cols_k = min(shard_tiles_w, C - col0_k)`) | host: `shard_spec.shape[1] / 32` → CT `shard_tiles_w` | width: shard k (cols `[k·shard_tiles_w, …)`) on its holding core; height: not split — every core holds all C columns | knob-turn (none needed) |
| `lane` (the block's tiles in local row-major order, split between the core's two data-movement RISCs) | independent — tiles of one block are independent | `num_writer_lanes` (L); lane sub-block = `ceil(rows_k·cols_k / L)` tiles | L = 2: lane j owns local tiles `[j·ceil(n_k/L), min((j+1)·ceil(n_k/L), n_k))`, `n_k = rows_k·cols_k` | host constant `NUM_WRITER_LANES` → CT `num_writer_lanes` (also sets the number of writer kernel descriptors and shard CBs) | lane 0 on RISCV_0 / NOC_1, lane 1 on RISCV_1 / NOC_0, on every shard-holding core | knob-turn (L ∈ {1, 2}) |

No other axis exists: the op has no dependent axis and no reuse-shared operand (one input, which varies along every axis it is cut on). Intra-tile positions are not an axis of the block — a tile is the atomic transfer unit (mechanism caps).

**Operand-reuse check.** The chosen split is the input's own shard split. The input varies along both axes it is cut on, and there is no second operand: no reuse-shared operand, no broadcast row.

**Stall-shadow check.** The only wait in the schedule is each lane's final write barrier (waiting for the acks of writes it issued). Every unit of work (all page writes of the lane's sub-block) is issued before it, so the in-flight writes already overlap their issue. Nothing independent of the wait is left to schedule into it. No stage waits on another kernel or core.

**Intermediate data.** The scheme has one stage (store). It produces no intermediate data and designates no collector core, so there is no further table to re-run.

**Committed scheme (Phase 0).** Owner writes from the resident shard. Every core that holds a shard writes its own tiles, read in place from its L1 through a CB backed on the sharded buffer (zero-copy), as whole pages to the interleaved output. Both of the core's data-movement RISCs issue writes, one contiguous half of the shard each. Block = the whole shard, one block per core. The core grid is the input's shard grid, read from the tensor. Every quantity above is a parameter with one host source. Nothing is inlined: not the grid, the shard shape, the orientation, the page size or the lane count.

**Why the core count is the blocking decision's choice and not a shape limit.** Every tile already sits in the L1 of the core that holds its shard. A core that does not hold a tile could only write it after fetching it over the NoC — a second crossing, which spec rule 4 forbids. So the participating set is exactly the data-holding cores (p2: 32, p1: 103), each using both of its writer RISCs (64 and 206 issuers). This is the minimum-traffic decomposition, and it is also the widest one that keeps one crossing per tile.

**Shard index source.** Each core computes its shard index k on device from `(get_absolute_logical_x(), get_absolute_logical_y())` against the CT range table. Within range i, k = (cores of the ranges before i) + local index, where local index = `(y−y0)·(x1−x0+1) + (x−x0)` when row-major and `(x−x0)·(y1−y0+1) + (y−y0)` when column-major. A core with `k ≥ num_shards` (or outside every range) has `num_blocks_this_core = 0`. The host passes **no per-core runtime args**. The only runtime value is the common output address, so the dispatcher writes no per-core payload on any call or replay (BB-17).

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_shard_lane0`, `cb_input_shard_lane1` | none — zero-copy view of the resident shard; capacity is the shard (`shard_tiles_h·shard_tiles_w` pages), fixed by the producer | whole shard | Nothing to buy. The whole block is resident before the call, so no streaming exists that a depth could overlap. |

There are no streaming CBs: the store reads the shard in place and writes straight to output pages. No staging buffer exists.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Interleaved page placement — page p lives in bank `p mod N_banks` at bank row `p / N_banks` (`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:20-42`) | bytes per NoC write | one write per page (`page_bytes`); never coalesce consecutive pages | a multi-page write puts pages 2.. into the wrong bank |
| bfloat8_b block format — 1024 datum bytes + 64 shared-exponent bytes per tile (`tt_metal/api/tt-metalium/tt_backend_api_types.hpp:129`) | sub-tile cuts and transfer size | whole tiles only, size = `buffer_aligned_page_size()` (1088), never `32·32` or a face | exponents dropped or a neighbour page overwritten; values change |
| `NOC_MAX_BURST_SIZE` = 256 words (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:289-290`) | bytes per write | `page_bytes` (≤ 2048) is below it, so every page is one packet | n/a for TARGET dtypes (`noc_async_write_impl` would split a larger page) |
| Shard local layout — slot of local tile (r, c) is `r·shard_tiles_w + c` with the **allocated** shard width, also on the ragged last shard (`tt_metal/impl/buffers/buffer.cpp:460-469`; `tt_metal/impl/buffers/buffer_distribution_spec.cpp:41-52`, strides from the full shard shape) | source address per tile | source = `shard_base + (r·shard_tiles_w + c)·page_bytes` | stepping by `cols_k` on a ragged last width shard with > 1 tile row reads the wrong tiles |
| Shard → core walk (`tt_metal/impl/buffers/buffer.cpp:439-440`: `corerange_to_cores(grid, num_cores, row_major)`) | which core holds shard k | device lookup uses `grid.ranges()` in stored order and the orientation flag. Host-side enumeration (role dump) passes `row_wise = orientation == ROW_MAJOR` explicitly: the Python default is `row_wise=False` (`tt_metal/api/tt-metalium/core_coord.hpp:196-197`) | column-major specs map to the wrong pages |
| Logical core coordinates are `uint8_t` (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:44-60`) | range-table entries | grid coordinates < 256 (always true) | n/a |
| Lane split | lane sub-block | `ceil(n_k/L)` per lane; a lane may own 0 tiles (p2 other spec's last core: 1 tile → 1 + 0) | a 0-tile lane must issue nothing; its barrier returns at once |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `owner_writes_resident` | **built** | always (every input that passes validation: HEIGHT/WIDTH_SHARDED L1, whole-tile shard) | `block_tile_rows x block_tile_cols` = the whole shard (`rows_k x cols_k`), one block per core; L lane sub-blocks of `ceil(n_k/L)` tiles | **Minimum.** Boundary: NoC/DRAM. The input crosses nothing (read in place from the core's own L1). Each output page is written once: P NoC writes, the DRAM crossing itself for a DRAM output, an L1-bank write for an L1 output. p2: 160 writes, 327680 B to DRAM. p1: 1024 writes, 1114112 B to L1 banks | Already maximal: the block is the core's whole assignment. Paid once per lane per call: accessor setup, the shard-index lookup, one write barrier. Paid once per call: launch and one binary write per kernel group per core range (p2: 1 range, p1: 2). A smaller block would only add barriers |
| `owner_writes_subchunked` (several blocks per shard, a barrier each) | rejected — superseded by `owner_writes_resident` everywhere | — | `< rows_k x cols_k` | same bytes | Nothing: the shard is zero-copy, so there is no L1 to save, and each extra block adds a barrier round trip |
| `gather_then_write` (fewer cores read remote shards, then write pages) | rejected — two NoC crossings per tile (spec rule 4); superseded by `owner_writes_resident` | — | n/a | +P·`page_bytes` NoC reads over the minimum | n/a |
| `redistribute_to_more_cores` (spread tiles over cores beyond the shard holders to raise occupancy) | rejected — same second crossing; the owner set already uses every data-holding core and both of its RISCs | — | n/a | +P·`page_bytes` NoC reads | n/a |
| `staged_through_dram` | rejected — forbidden (extra full-size DRAM buffer); +2 DRAM crossings of the tensor | — | n/a | +2·P·`page_bytes` DRAM | n/a |
| `compute_engine_repack` (unpack/pack through a compute kernel) | rejected — no arithmetic exists; a repack recomputes bfloat8_b exponents (spec rule 7); superseded by `owner_writes_resident` | — | n/a | same NoC bytes + engine passes | n/a |
| `coalesced_bank_runs` (pages p, p+N_banks, … contiguous in one bank sent as one write) | rejected — structurally unavailable: those pages' source tiles sit N_banks slots apart in the shard, never adjacent, so a multi-page transfer needs a local gather first | — | n/a | n/a | n/a |

There is one built regime, so there is no selection predicate. No regime is `deferred`: every alternative either adds a NoC or DRAM crossing to a regime already at the minimum, or does not exist for interleaved pages.

### Traffic ranking

Counted per tensor and tier. P = R·C pages of `page_bytes`.

| Rank | Candidate split | Input crossings | Output crossings | Cross-core traffic | Combine | Verdict |
|------|-----------------|-----------------|------------------|--------------------|---------|---------|
| 1 | Input's own shard split (owner writes) | 0 (core-local L1 read) | 1 per page (DRAM for p2, L1 bank for p1) | P writes | none | **chosen — the minimum** |
| 2 | Re-split tile_rows / tile_cols over any other core set (more, fewer, one) | 1 NoC read per moved tile | 1 per page | 2·P transfers | none | rejected (rule 4) |
| 3 | Split a dependent axis and combine | — | — | — | — | not applicable: no dependent axis exists |
| 4 | Stage through DRAM | 1 write + 1 read DRAM | 1 per page | 3·P | none | rejected (forbidden) |

Within rank 1 the extent is fixed by the shard. Bytes are constant in the extent (no overlapping footprint, no shared operand), so the extent ranking reduces to "one block per core" (fewest barriers).

### Block schedule

```cpp
// per shard-holding core (shard k), per writer lane j in [0, num_writer_lanes)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // 1 if k < num_shards, else 0
    locate_shard_block(block_idx);   // k, (row0_k, col0_k), (rows_k, cols_k), lane j's sub-range
    store_shard_block(block_idx);    // one page write per tile of lane j's sub-block, from the resident shard
}
drain_writes();                      // one write barrier per lane
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `locate_shard_block` | scalars of shard k: `row0_k = k·shard_row_step`, `col0_k = k·shard_col_step`, `rows_k`, `cols_k`, lane range | the shard (never moved) | once per lane per call: logical coordinates read once, range-table walk once, shard base read once from the lane's CB (`get_read_ptr`), output accessor built once |
| `store_shard_block` | lane sub-block: `ceil(rows_k·cols_k / L)` tiles in local row-major order. Tile (r, c) → source slot `r·shard_tiles_w + c`, output page `(row0_k + r)·C + col0_k + c` | the shard | one NoC write per tile (mechanism cap). `(r, c)` comes from one division at the lane's start, then incremental stepping. Writes issue back to back with no barrier or flush between them. No CB handshake, no init |
| `drain_writes` | — | — | once per lane per call: `noc_async_write_barrier` on the lane's NoC, so no transaction is in flight when the kernel returns (BB-03) |

**Profiling zones (BB-21).** `locate_shard_block` + `store_shard_block` run inside one named zone per lane (`S2I-STORE`), and `drain_writes` in its own zone (`S2I-DRAIN`). Zones compile out of the plain build. The host exposes a role dump: core → (shard k, lane tile ranges, first/last page), computed with `ttnn.corerange_to_cores(grid, None, row_wise=orientation_row_major)` (`ttnn/cpp/ttnn-nanobind/tensor.cpp:659-665`).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Writer lanes: L = 2 (both DM RISCs) | p2 has 5 tiles per core (split 3 + 2). The second kernel binary written per core range, and the second RISC's launch, may cost more than the halved serial issue chain saves. Catalog `split_reader` measured ~1.2x at tile-size transfers only when one RISC was issue-bound | L = 1 (lane 0 only, one shard CB). Judge on unprofiled traced-replay host time as well as device kernel time (lesson RB-053: cores ending at different times skew profiled time) |
| Kernel group core ranges = the input's shard grid (p1: 2 ranges, 11x9 + 4x1) | Binaries and launch messages are written once per core range (lesson RB-036). One rectangle may start sooner than two | Kernel ranges = the grid's bounding rectangle (p1: 11x10, one range). The 7 cores without a shard compute `num_blocks_this_core = 0` and return. Shard CBs stay on the shard grid; idle cores must not touch the CB |
| Per-page output address through `TensorAccessor::get_noc_addr` | An L1-interleaved output has a non-power-of-two `NUM_L1_BANKS`: each page pays a constant-divisor division `udivsi3_const_divisor` (`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:20-34`) plus a bank-table lookup | Compute (bank, bank row) once at the lane's first page, then step bank += 1 (wrap → bank row += 1) along its consecutive pages |
| Shard index derived on device from logical coordinates | Adds a short CT-table walk per lane at kernel start | One per-core runtime arg (k) written by the dispatcher. Costs one unicast payload per core per launch, which a traced replay also pays |

Overlap lamp: not attached. No compute exists and no CB streams, so a large block cannot serialize data movement against anything. Grid-synchronization lamp: not attached. No collective, multicast or semaphore exists, and the participating cores are fixed by where the data lies.

**Catalog entries that informed the knobs** (`ttnn/ttnn/operations/examples/master.md`):

| Entry | Knob it informed |
|-------|------------------|
| `double_buffer` | Issue every write of the lane's sub-block back to back, then one barrier. Past ~4 in flight the cost is per transaction (~115 ns per tile transaction on one RISC) → motivates L = 2 |
| `split_reader` | Splitting NoC issue across BRISC + NCRISC: 1.2x at 1–2 KB transfers, only when one RISC is issue-bound → the L lamp |
| `noc_placement` | NoC choice matters only when bandwidth-bound. This op is latency-bound (≤ 11 writes per lane), so the lanes keep the default NoC per RISC |
| `tile_reorder` | Whole-page writes beat sub-page (face) writes → one write per page |
| `bank_stagger` | Consecutive pages already start each core at a different bank (shard first pages differ). No stagger needed |
| `trid_double_issue` | Not applicable: one block per lane leaves nothing to pipeline behind a per-trid barrier |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input shard (resident) | TILE pages of `page_bytes`, input dtype, slot stride `page_bytes` | `ttnn.cb_descriptor_from_sharded_tensor(cb_index, input_tensor)` for each lane's CB: globally allocated on the input buffer, core ranges = shard grid, page size = buffer aligned page size (`ttnn/core/tensor/tensor_utils.cpp:48-72`; binding `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:517-556`) | Zero-copy, consumed in place. Never read over the NoC, never through a sharded `TensorAccessor` |
| Shard base address | L1 byte address | `get_read_ptr(cb_input_shard_laneJ)` once at lane start (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:346`) | No reserve/push/wait/pop. The shard was produced by the previous program, and no kernel in this program writes it |
| Shard L1 → output page | same bytes, unchanged | `noc_async_write_page(page, out_accessor, src)` (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:1418`) with an interleaved `TensorAccessor` built from `TensorAccessorArgs(output_tensor)` + the common-RT output address + `page_bytes` | One crossing per tile. DRAM banks (p2) or L1 banks (p1, NoC loopback where the bank is the source core) |
| Completion | — | `noc_async_write_barrier()` per lane (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:1951`) | BB-03 |
| Mesh | — | `ttnn.generic_op` wraps a `ProgramDescriptor` into an SPMD mesh program over `MeshCoordinateRange(mesh_device->shape())` (`ttnn/cpp/ttnn/operations/generic/generic_op.cpp:15-22`) | Same program on every chip. The output's mesh buffer has one address on every chip (rule 5) |

**TARGET axes as placement, not algorithm.** Every TARGET value is served by the same kernel and the same page map from Phase 0 on. Refinements widen `SUPPORTED` and pin the cell's tests. They do not change the scheme.

| TARGET axis value | Cost class | What changes |
|-------------------|------------|--------------|
| `dtype: bfloat8_b` | knob-turn | `page_bytes` = 1088 from the tensor (already the single source); the shard CB data format follows the tensor |
| `in_scheme: height_sharded` | knob-turn | `(shard_row_step, shard_col_step)` = `(shard_tiles_h, 0)` instead of `(0, shard_tiles_w)`. Both axes are independent, so neither cut needs a combine |
| `buffer: l1` | knob-turn | `TensorAccessorArgs(output_tensor)` encodes the L1 bank set. The output's L1 share per bank joins the per-core budget (`l1_ledger.md`) |
| `layout: TILE_LAYOUT` | — | only value |

**Structural impossibilities.** None to add. `feature_spec.py` INVALID stays empty: TARGET holds TILE only, and both dtypes tile.

**Program cache (BB-07).** `generic_op` hashes kernel sources, CT args, core ranges, CB sizes/formats and runtime-arg **counts**, never tensors (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:79-133`, `:307-322`). On a hit it re-applies runtime args, common runtime args and sharded-CB addresses from the descriptor passed in (`tt_metal/impl/program/program_descriptors.cpp:188-243`). The Python layer therefore:

| Rule | Detail |
|------|--------|
| Cache the descriptor | A module-level dict keyed on: device/mesh id, `shape`, `padded_shape`, dtype, layout, input memory layout, shard shape, shard ranges (stored order), orientation, output buffer type, `NUM_WRITER_LANES`. A hit rebuilds nothing |
| Refresh before every dispatch | Point both shard CBs at the current input buffer (`CBDescriptor.set_buffer_from_cb`, `program_descriptors.cpp:488-499`, or re-create the two CB descriptors). Set `common_runtime_args[0]` of both writer kernels to the current output address (`KernelDescriptor.common_runtime_args`, `program_descriptors.cpp:851-854`) |
| Hash | Set `ProgramDescriptor.custom_program_hash` (`program_descriptors.cpp:1141`) from the full key. generic_op skips its collision check when the hash is custom, so the key must hold everything any CT arg, core range or CB size is derived from |
| Trace (BB-16) | The only device allocation is the output tensor. Building the descriptor allocates no device memory, and a replay does not run the host path |

**Entry point order.** `validate(input_tensor, memory_config)` is the first line, then: allocate the output (the call's first device allocation, so a freed output-sized range is reused — the dirty-destination case), look up or build the descriptor, refresh it, then `ttnn.generic_op([input_tensor, output_tensor], descriptor)` once.

**validate().**

1. Structural checks raise `ValueError` (never a support refusal): input not on device; not TILE; not sharded; not L1; BLOCK_SHARDED; shard shape not a whole number of tiles; height shard width ≠ padded W, or width shard height ≠ padded H'; `num_shards > grid.num_cores()`; rank ∉ {2, 4}; `memory_config.memory_layout ≠ INTERLEAVED`.
2. Build the scenario dict (`shape, dtype, layout, in_scheme, shard_shape, grid, orientation, buffer`), as `eval/golden_tests/g6_sharded_to_interleaved/axes.py:39-62` does, then apply `INPUT_TAGGERS` (`tag_dtype`, `tag_layout`, `tag_in_scheme`, `tag_buffer`).
3. Check SUPPORTED → `UnsupportedAxisValue`, then EXCLUSIONS (empty) → `ExcludedCell`.

No size, shape or shard-geometry axis exists, and no other tagger.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = one shard: `rows_k x cols_k` tiles, split into L lane sub-blocks of `ceil(rows_k·cols_k / L)` |
| Grid | the input's shard grid `shard_spec.grid`, read from the tensor (p2: (0,0)-(7,3), one range; p1: (0,0)-(10,8) + (0,9)-(3,9), two ranges). Two writer kernel descriptors (lane 0: `DataMovementConfigDescriptor(processor=RISCV_0, noc=NOC_1)`; lane 1: `(RISCV_1, NOC_0)`, `program_descriptors.cpp:599-622`) over the same ranges |
| Per-core work | shard k = the core's position in the walk. `row0_k = k·shard_row_step`, `col0_k = k·shard_col_step`, `rows_k = min(shard_tiles_h, R − row0_k)`, `cols_k = min(shard_tiles_w, C − col0_k)`. p2: 5 tiles (lanes 3 + 2) on 32 cores; p1: 10 tiles (5 + 5) on 102 cores, 4 (2 + 2) on (3,9) |
| Remainder | ragged last shard: runtime extents from the `min` above (p2 other spec: last 1 tile; p1 other spec: last 1 tile row). Cores with `k ≥ num_shards`: 0 tiles. Tile counts are per image: R from `padded_shape` (`prod(shape[:-2])·ceil(shape[-2]/32)`), C = `ceil(shape[-1]/32)`, never `floor(volume/1024)` |

Single regime: no selection predicate. Regime-pinned tests are not needed. The other-spec tests pin the parts of the page map that only some specs exercise: column-major walk, two ranges, a ragged last shard, a multi-row shard (p1 other: 11 rows).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_shard_lane0` | 0 | `page_bytes` (buffer aligned page size) | `shard_tiles_h·shard_tiles_w` (= `aligned_size_per_bank / page_bytes`; set by `cb_descriptor_from_sharded_tensor`) | spans tile_rows (`shard_tiles_h`) and tile_cols (`shard_tiles_w`); streams nothing. A zero-copy view of the resident shard (0 B of op L1) | input dtype (Float16_b / Bfp8_b), dataflow-only | none — resident shard written by the producing program; globally allocated, never pushed | writer lane 0 | whole call |
| `cb_input_shard_lane1` | 1 | same | same | same; the same bytes as lane 0 (both CBs are backed on the input buffer at offset 0) | same | none — resident shard | writer lane 1 | whole call |

Both CBs view one buffer, so each lane kernel has its own CB as the one-consumer rule requires. That costs no L1. With L = 1 only `cb_input_shard_lane0` exists. CB sync: push count = wait count = 0 on both (no handshake: resident data, single reader per CB).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `locate_shard_block` | scalars of shard k + lane range | no (raw) | `cb_input_shard_laneJ`, base pointer only, resident | — | unchanged |
| 2 | `store_shard_block` | `ceil(rows_k·cols_k / L)` tiles, local row-major | no (raw) | `cb_input_shard_laneJ`, read in place, never popped | none — output tensor pages via `TensorAccessor` | shard unchanged (input intact) |
| 3 | `drain_writes` | — | no (raw) | — | — | all of the lane's writes acked |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `locate_shard_block` | raw_api | `get_absolute_logical_x()`, `get_absolute_logical_y()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:44-60` | — | — | — | none (k derived; range table and orientation are CT) |
| `locate_shard_block` | raw_api | `get_read_ptr(cb)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:346` | `cb_input_shard_laneJ` | `cb_input_shard_laneJ` | — | none |
| `store_shard_block` | raw_api | `TensorAccessor` (interleaved specialization) + `TensorAccessorArgs` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:493-562` | output accessor args (CT), output address (common RT), `page_bytes` | — | output pages | none |
| `store_shard_block` | raw_api | `noc_async_write_page<AddrGen>(page_id, accessor, src_l1_addr)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1418` | size defaults to the accessor's aligned page size | `cb_input_shard_laneJ` (address arithmetic) | output pages | sub-block extent `ceil(rows_k·cols_k/L)` sets the trip count; `shard_tiles_w` sets the slot stride |
| `drain_writes` | raw_api | `noc_async_write_barrier()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1951` | lane's NoC | — | — | none |
| host | raw_api | `ttnn.cb_descriptor_from_sharded_tensor` | `ttnn/core/tensor/tensor_utils.cpp:48-72` | `(cb_index, input_tensor)` | — | — | capacity = shard (`shard_tiles_h·shard_tiles_w`) |
| host | raw_api | `ttnn.generic_op` (single dispatch) | `ttnn/cpp/ttnn/operations/generic/generic_op.cpp:34-36` | `[input_tensor, output_tensor]`, descriptor | — | — | — |

**Helpers considered and rejected** (each `raw_api` row above; no compute phase exists, so the compute helpers — tilize, untilize, reduce, eltwise, matmul — have nothing to cover):

| Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|--------------------------------|
| `local_copy_helpers_dataflow.hpp` (`local_addr`, `set_read_state`/`read_with_state`) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:9-22`, `:57`, `:74-75` | an L1 → L1 copy built on self-aimed **reads** into a local typed buffer. Here the destination is a remote interleaved page (DRAM or another core's L1 bank), which needs a NoC write keyed by page id |
| `mcast/kernel/mcast_args.hpp` (`McastArgs`) | `ttnn/cpp/ttnn/kernel_lib/mcast/` | multicasts one payload to many receivers. Every page here has exactly one destination and no operand is shared |
| `tilize_helpers_dataflow.hpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp` | moves row-major sticks for tilize. Input and output here are whole tiles |
| `reduce_helpers_dataflow.hpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | prepares reduce scalers; no reduction exists |

No helper exists for "write a block of resident tiles to interleaved pages". The block operation is small enough to realize directly. Gap recorded: a `write_resident_block_to_pages` dataflow helper would cover it.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Tile on the wrong page, or a page unwritten | The op is nothing but the page map. Orientation, two core ranges, a ragged last shard and the strategy each change it, and a fresh buffer can hold the right bytes by chance | One formula for both strategies (`row0_k`, `col0_k` from the step pair). Device walk = buffer's walk (`buffer.cpp:439-440`). Tests: per-tile-scaled inputs, exact gate, dirty destination, both other specs (column-major, two ranges, ragged, 11-row shards) |
| Host walk default | `ttnn.corerange_to_cores` defaults to `row_wise=False` (`core_coord.hpp:196-197`) | Always pass `row_wise=orientation_row_major` (role dump, num-core checks) |
| Ragged last shard slot stride | The last width shard keeps the allocated width as its stride (`buffer.cpp:460-469`) | Source slot uses `shard_tiles_w`, never `cols_k` |
| bfloat8_b page = 1088 B | A 1024-B or face-sized move drops exponents or spills (risks R5) | All sizes from `buffer_aligned_page_size()`; whole-tile moves only; no compute kernel |
| Stale `Buffer*` in a cached descriptor | generic_op re-applies CB addresses from the passed descriptor's `buffer` (`program_descriptors.cpp:230-242`). A cached CB still pointing at a freed input dereferences a dangling buffer | Refresh both shard CBs and the output address on **every** call before dispatch, hit or miss |
| Incomplete cache key | `custom_program_hash` disables generic_op's collision check (`generic_op_device_operation.cpp:79-81`) | Key holds every field a CT arg / range / CB size derives from; the cache-order test alternates specs, including orientation alone |
| Writes in flight at return (BB-03) | Pure write kernel; the next program starts at once | Barrier on each lane after its last write; non-posted writes only |
| Two globally allocated CBs on one buffer | Each lane needs its own CB (ownership rule), so two CBs alias the shard | If the runtime refuses the alias, lane 1 takes the shard base from a second common runtime arg (`input_tensor.buffer_address()`, refreshed per call like the output address). Same address on every core of an L1-sharded buffer |
| Logical coordinate space | Shard specs use device logical worker coordinates. `get_absolute_logical_*` is device-absolute, `get_relative_logical_*` sub-device-relative (`dataflow_api.h:36-90`) | Use the absolute pair. A mismatch shows as wrong pages in the other-spec tests |
| Output must be the first device allocation | The dirty-destination test needs the output to land on the freed range | No device allocation before `allocate_tensor_on_device`; descriptor building is host-only |
| p1 output in L1 | Output pages share the L1 of the shard-holding cores. They come out of the 1347776 B budget (BB-05, spec D16) | Op buffers: 0 B. Output share ≤ `ceil(1024 / N_L1_banks)·1088` B per core (`l1_ledger.md`) |
| Acceptance-test scope | The planner template asks for ≥ 4 generic shapes; the spec limits tests to production + rule-required cases (BB-22) | Test file covers the two production specs, the other spec of each, destination-from-memory_config, dirty destination, repeat / input intact, cache order, special values, validation. Cells outside SUPPORTED must raise `SupportRefusal` and report xfail |

# Operation Design: g6_reshard

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (L1 → L1, cross-core; no compute kernel, no DRAM) |
| Goal | Move every tile of a BLOCK_SHARDED L1 tensor, as stored, into a new tensor BLOCK_SHARDED in L1 on the placement `memory_config` names, in one device program, for every legal placement pair. |
| Math | `out[i] = in0[i]` for every logical index `i`; out held in the placement `memory_config` names, in0 unchanged (`docs/explorations/g6_reshard/math.md` M1, M2) |
| Tile form | For every tile index `(tile_row, tile_col)` of the padded 2-D view: the 1088 B tile (bfloat8_b, shared exponents included) at that index in in0's shard is copied byte-for-byte into the slot of out's shard that holds the same index (`math.md` F2) |
| Production call | x `(1, 1, 2048, 1024)` bfloat8_b TILE on (0,0)-(10,9), shard (224, 96), ROW_MAJOR → out on (0,0)-(7,7), shard (256, 128), ROW_MAJOR |
| References | `docs/explorations/g6_reshard/{DOSSIER,math,tensix,risks,golden_brief}.md`; `eval/golden_tests/g6_reshard/{feature_spec,helpers}.py`; `tech_reports/tensor_accessor/tensor_accessor.md`; catalog `ttnn/ttnn/operations/examples/master.md` (entries below) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes (positional) | rank 4, bfloat8_b, TILE, BLOCK_SHARDED in L1, on device; padded 2-D view H, W multiples of 32 | — | its placement → accessor CT args; its address → common RT arg |
| `memory_config` | `ttnn.MemoryConfig` | yes (positional) | BLOCK_SHARDED, L1, with a shard spec: one core rectangle inside the device worker grid, shard dims multiples of 32, enough grid rows/cols for the block rows/cols in its orientation | — | its placement → kernel core range, CT args, per-core RT args |

No output dtype, no output tensor argument, no `compute_kernel_config`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 4 `(d0, d1, d2, d3)`; logical 2-D view `H = d0·d1·d2`, `W = d3`, both multiples of 32 |
| Dtype | bfloat8_b (Phase 0 SUPPORTED) |
| Layout | TILE |
| Memory | BLOCK_SHARDED in L1, any legal placement (any core rectangle, either orientation, shard dims multiples of 32, cut-short last block row/column); read from the tensor, never assumed |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` |
| Memory | exactly `memory_config` (BLOCK_SHARDED, L1, its grid, shard shape, orientation); a new allocation, never a view of in0 |

## Blocking Model

The op computes nothing and no element depends on another, so the blocking decides only **which core issues which transfer and how coarse each transfer is**. Where each tile starts and ends is fixed by the two placements.

**Committed scheme (Phase 0): PULL.** Each core that holds a block of out reads, into its own out shard, the pieces of in0's blocks that overlap it. The block is the out shard (one per core, honored as handed by `memory_config`). Inside a core the block's tile rows are split between both data-movement RISC-Vs (`issuers_per_core = 2`, one per NoC). Each issuer enumerates its pieces **on device** from the two placements' geometry and moves every piece as maximal runs that are contiguous on both sides, one NoC read per run, then waits once. Every knob below is a parameter with one source; none is an inlined constant.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| tile rows of the padded 2-D view (`tensor_height_tiles` = d0·d1·⌈d2/32⌉; P: 64) | independent: `out[h, w]` is `in0[h, w]` alone, no result spans it | `block_height_tiles` (out shard height / 32); runtime extent of the last block row = `tensor_height_tiles − (num_block_rows − 1)·block_height_tiles` | whole out shard (P: 8; O1: 8) | `memory_config.shard_spec.shape[0] // 32`, read once on host | out block row `r` → its holder core (ROW_MAJOR: `(x0 + c, y0 + r)`, COL_MAJOR: `(x0 + r, y0 + c)`); inside the core the block's rows split over `issuers_per_core` issuers as contiguous ranges of `⌈rows/2⌉` and `⌊rows/2⌋` | knob-turn (`issuers_per_core`) |
| tile columns (`tensor_width_tiles` = d3/32; P: 32) | independent: same reason | `block_width_tiles` (out shard width / 32); last block column = `tensor_width_tiles − (num_block_cols − 1)·block_width_tiles` | whole out shard (P: 4; O1: 5) | `memory_config.shard_spec.shape[1] // 32`, read once on host | out block column `c` → its holder core (same map); not split inside a core: each issuer covers the block's full width, so full-width rows can merge into one run | knob-turn |
| piece boundaries along both axes (not a knob: in0's placement) | — set by in0's block grid; a piece is (in block) ∩ (issuer sub-block) | `in_block_height_tiles`, `in_block_width_tiles` | P: 7 × 3; I0/I3/I5: 8 × 4 | `input_tensor.memory_config().shard_spec.shape // 32`, read once on host | in block `(r, c)` lives on the core in0's placement names; resolved on device by in0's `TensorAccessor` (never a hand-built table) | — |
| transfer (run) extent | — | maximal: the piece's width in tiles, or `piece_rows × piece_width` when `piece_width == in_block_width_tiles == block_width_tiles` (rows contiguous on both sides) | P: 1–3 tiles; I0/I3/I5: 16 tiles (one run per issuer) | derived on device from the above; never a literal | the issuer that owns the piece | — |

Operand-reuse check (the split is by out block): in0 varies along both split axes; each tile of in0 feeds exactly one out tile, so no operand is reuse-shared and no broadcast row exists.

Stall-shadow check: the only wait is the final read barrier of each issuer. Nothing remains to schedule into it: the op has one phase and all of its reads are already in flight when the barrier starts. Enumerating piece `n+1` already overlaps the in-flight reads of piece `n`.

`num_blocks_this_core = 1` (the out shard is the coarsest block and it needs no scratch L1, so there is no L1-pressure reason to cut it).

### Buffer-depth knobs

The op allocates **no circular buffer**: reads land directly in out's shard by address (pack-into-destination, `l1-footprint-discipline.md` Rule 3.1). The only depth-like knob is NoC reads in flight.

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| — (NoC reads in flight per issuer; no CB) | reads issued before the barrier | all runs of the issuer's sub-block (P: ≤ 8; O1: ≤ 12; I0/I3/I5: 1) | one barrier per block: the NoC never idles between batches and the round-trip latency is paid once (catalog `double_buffer`, `trid_double_issue`: batch the barrier; trids add nothing when there is one batch) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| NoC packet size `NOC_MAX_BURST_SIZE` (16384 B on BH, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:290`) | run bytes | none with `noc_async_read` (it splits a run into packets itself, `dataflow_api.h:547`). If a realization uses `noc_async_read_one_packet` (`:515`) or stateful single-packet reads: clamp each transfer to `⌊NOC_MAX_BURST_SIZE / aligned_page_bytes⌋` tiles (15 for bfloat8_b on BH) | I0/I3/I5 runs are 16 tiles = 17408 B > 16384 B: a one-packet read truncates or corrupts the run |
| Shard slot stride: a cut-short shard keeps the full shard row stride (slot of in-shard tile `(i, j)` = `i·in_block_width_tiles + j`; out: `i·block_width_tiles + j`) | run merging across rows | merge rows into one run only when `piece_width == in_block_width_tiles == block_width_tiles` (compare against the **shard** widths, never the valid widths of a tail block) | P's last input block column is 2 tiles valid in a 3-wide shard: merging by valid width reads the wrong slots of every tail shard (wrong results, R1) |
| L1 / NoC address alignment (16 B, `tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h:43`) | run start offsets | slot stride = `aligned_page_bytes` (`buffer_aligned_page_size()`; 1088 = 68 × 16 for bfloat8_b); host asserts in0 and out report the same aligned page size | misaligned NoC transfers |
| Kernel-config region (BB-04) | per-core args | per-core RT args are O(1) (the issuer's rectangle); in0's accessor args scale with in0's core count ≤ worker grid (≤ 140 on BH) | a host-built run list would grow with the run count (one per tile in the worst placement) and overflow the region — see the rejected HOST-RUN-LIST row |
| L1 budget 1347776 B per core (BB-05, `sources/local/contract_r3.md:68`) | out shard bytes = `block_height_tiles·block_width_tiles·aligned_page_bytes` | host refuses (RuntimeError) a `memory_config` whose out shard exceeds 1347776 B; golden INPUTS max is 43520 B (O1) | allocation succeeds alone and fails inside the model |
| Empty issuer range | issuer sub-block height | a block with 1 tile row gives issuer 1 an empty range: it issues no read and still runs its barrier (a no-op) and returns | a kernel that assumes ≥ 1 row reads garbage geometry |

### Regimes

Named boundary for "minimum": **each tile whose source core differs from its target core crosses the NoC once; each tile already on its target core is copied once inside that core; DRAM is never crossed.**

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| PULL — out holders read their pieces | **built** | every legal placement pair (any rectangles, orientations, shard shapes, tails; identity placement included). The only built row; no selection function. | the out shard `block_height_tiles × block_width_tiles`, clipped at the tensor edge; rows split over `issuers_per_core`; pieces = (in block) ∩ (issuer sub-block) | **minimum**. P: 1880 tiles (2045440 B) cross once, 168 tiles copied in place by a self-aimed read; I0: 0 cross / 2048 local; I3: 1792 / 256; I5: 2048 / 0; O1: 2005 / 43. DRAM 0 | The block is fixed by `memory_config` (one per core), so the fixed costs are paid once per block = once per issuer kernel: RT-arg load, accessor construction, one read barrier (the round-trip latency of the last read). Inside it, the run is the extent that amortizes: one NoC command per run, so maximal runs give P 16 commands per block (2 per tile row) and I0/I3/I5 2 commands per block (16 tiles each). Per piece: one accessor address resolution, amortized over the piece's rows. |
| PUSH — in holders write their pieces | deferred: identical bytes on every tier and the same per-destination ingress (P: 34816 B into each out core), and PULL already covers every placement pair it would serve; it differs only in which end issues (P: 110 issuers with ≤ 14 runs each, 9.3 on average, instead of 64 with 16) and needs a write-ack barrier on every source. Reachable: the piece enumeration is (my block) ∩ (other placement's block grid) and is symmetric, so PUSH is the same kernel with the roles swapped (kernel on in0's block-holder rectangle, out-side addressing through an out accessor, `noc_async_write` + `noc_async_write_barrier`); no semaphore is needed because nothing in the program consumes out | same as PULL | in0's shard | minimum (same bytes) | same terms, with the in shard as the block |
| BALANCED — each piece issued by the less-loaded of its two holders | deferred: same bytes and same per-destination ingress as PULL, and PULL covers every placement pair; it adds a per-piece owner decision and a kernel role on in-only holders, which is a second core-range group (P: 46 in-only cores form two rectangles, BB-17). Reachable: the issuer of a piece is a host-side assignment over the same piece enumeration | same as PULL | out shard or in shard per piece | minimum | same terms |
| PER-TILE — one transfer per tile (accessor page walk) | rejected: superseded by PULL's maximal runs (every run is ≥ 1 tile; P 2048 commands instead of 1024, I0 2048 instead of 128) | — | one tile | minimum bytes, maximum command count | — |
| HOST-RUN-LIST — host enumerates every run and ships it as per-core RT args | rejected: RT args grow with the run count, unbounded by the placement (one entry per tile of the out block in the worst legal pair), so BB-04 fails on legal placements; superseded by on-device enumeration with O(1) per-core args | — | out shard | minimum | — |
| DRAM / INTERLEAVED STAGING | rejected: forbidden by rule 1; +1 DRAM write and +1 DRAM read of 2228224 B and a second NoC crossing | — | — | 2× NoC, 2 DRAM crossings | — |
| DELEGATED — any existing placement-changing TTNN op | rejected: forbidden by rule 1 and BB-01 (a second program, or the move handed to the library) | — | — | — | — |
| COMPUTE COPY — unpack → pack through a compute kernel | rejected: forbidden by rule 1 (no unpack/repack); the packer can regroup bfloat8_b exponents; an extra engine pass | — | — | minimum bytes + 2 L1 passes per tile | — |
| THIRD-CORE ISSUER — a core holding neither end stages tiles | rejected: forbidden by rule 1; every tile crosses the NoC twice | — | — | 2× NoC | — |

### Traffic ranking

| Candidate split (who issues) | DRAM crossings | NoC crossings per non-local tile | Extra traffic | Rank |
|------------------------------|----------------|----------------------------------|---------------|------|
| PULL (split by out block; out holder issues reads) | 0 | 1 | none (reads of an immutable, resident input need no handshake) | 1 (chosen) |
| PUSH (split by in block; in holder issues writes) | 0 | 1 | write acks back to every source | 1 (tie) |
| BALANCED (split by piece) | 0 | 1 | write acks for the pushed pieces | 1 (tie) |
| third-core issuer | 0 | 2 | a staging copy | 4 (forbidden) |
| DRAM / interleaved staging | 2 per tile | 2 | 4456448 B through DRAM | 5 (forbidden) |
| split a dependent axis + cross-core combine | — | — | no axis is dependent (no result spans an axis), so no combine exists | n/a |

PULL, PUSH and BALANCED tie on bytes per tier; occupancy is not part of this ranking. Tiebreak to PULL: its completion is local (a read barrier on the core that owns the block, no remote acks), the out block is written only by the core that holds it (each tile written once falls out of the row partition of each block), and its issuer group is one rectangle (out's block holders).

### Block schedule

Logical schedule of each issuer kernel (one source, two instances per out-holding core: issuer 0 on RISCV_1/NoC0, issuer 1 on RISCV_0/NoC1):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {  // == 1: the out shard
    locate_pieces(block_idx);   // in0 blocks overlapping [issuer_row_begin, issuer_row_end) x [block_col_begin, block_col_end)
    move_pieces(block_idx);     // per piece: resolve the source once, issue its maximal runs into the out shard
    complete_block(block_idx);  // one read barrier on this issuer's NoC
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `locate_pieces` | the issuer's sub-block: `issuer_rows × block_width_tiles` (P: 4 × 4) | in0's shards on their holders (model's tensor, read-only); out's shard on this core | once per block: load the issuer rectangle, construct in0's accessor, find the first overlapping in block row and column (one division per axis); the walk over in block rows and columns is additions only |
| `move_pieces` | each piece: `piece_rows × piece_width` with `piece_rows ≤ in_block_height_tiles`, `piece_width ≤ in_block_width_tiles` (P: ≤ 4 × 3) | same | per piece: one source-address resolution (`get_noc_addr` of the piece's first tile); per maximal run: one NoC read; no barrier, no semaphore, no multicast inside the block |
| `complete_block` | — | out's sub-block complete in L1 | one `noc_async_read_barrier` per block per issuer (BB-03: nothing left in flight at return) |

Piece walk, stated once so both issuers and every placement use the same rule:

| Quantity | Value (tiles of the padded 2-D view) |
|----------|---------------------------------------|
| in block rows overlapping the sub-block | `issuer_row_begin / in_block_height_tiles` … `(issuer_row_end − 1) / in_block_height_tiles` |
| in block columns overlapping the sub-block | `block_col_begin / in_block_width_tiles` … `(block_col_end − 1) / in_block_width_tiles` |
| piece rows of in block row `ri` | `[max(issuer_row_begin, ri·ih), min(issuer_row_end, (ri+1)·ih))` |
| piece columns of in block column `ci` | `[max(block_col_begin, ci·iw), min(block_col_end, (ci+1)·iw))` |
| source of the piece's first tile | in0 accessor `get_noc_addr(page_id = piece_row0·tensor_width_tiles + piece_col0)` on this issuer's NoC; each next row adds `in_block_width_tiles·aligned_page_bytes` |
| destination of the piece's first tile | `out_shard_base + ((piece_row0 − block_row_begin)·block_width_tiles + (piece_col0 − block_col_begin))·aligned_page_bytes`; each next row adds `block_width_tiles·aligned_page_bytes` |
| runs of the piece | one run of `piece_rows·piece_width` tiles if `piece_width == in_block_width_tiles == block_width_tiles`, else `piece_rows` runs of `piece_width` tiles |

`ih`, `iw` above are `in_block_height_tiles`, `in_block_width_tiles`. `block_row_begin` / `block_col_begin` are the out block's origin; `block_col_end` and the issuer row range are already clipped to the tensor, so no piece reaches beyond the tensor or into the unused slots of a tail shard.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Issuers per core (default 2: both DM RISC-Vs, one per NoC) | Each core moves only 32 tiles (P); the second instance's launch and RT-arg load may cost as much as halving the per-RISC issue (8 instead of 16 runs) and adding NoC1 ingress saves (catalog `split_reader`: pays only when one RISC is issue-bound) | `issuers_per_core = 1` (issuer 0 only, NoC0) on P and O1 |
| Issuer row split shape (default: contiguous halves) | Halves put each issuer's reads on one source block row at a time; interleaving rows may balance the two NoCs' link load better | even/odd tile rows per issuer (runs stay per row; full-width merging is lost) on P |
| Per-run command cost (default: plain `noc_async_read` per run) | P issues ≤ 4 rows per piece per issuer with the same size; programming the source and size once per piece may cut command-register writes | `noc_async_read_one_packet_set_state` + `_with_state` per piece (run ≤ 15 tiles: mechanism cap above) |
| Start-piece order (default: every core starts at its top-left piece) | In P each in block feeds up to 4 out blocks; cores that start on the same source core at the same moment contend on its egress (catalog `bank_stagger`) | rotate each core's piece walk by its block index |
| NoC assignment (default: issuer 0 NoC0, issuer 1 NoC1) | NoC0 and NoC1 route in opposite directions; for P sources lie up to 3 columns east/1–2 rows south of their targets (catalog `noc_placement`) | swap the NoCs of the two issuers |
| Overlap lamp | does not apply: no compute phase and no CB; all reads of a block are in flight together | — |
| Grid-synchronization lamp | does not apply: no collective, barrier or semaphore across cores; every core completes on its own reads | — |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| in0's shards on their holder cores (L1) | bfloat8_b tiles as stored, `aligned_page_bytes` slots, row-major inside the shard with the full shard row stride | resident before the program (the model's tensor) | never written; read only over the NoC (self-aimed for tiles already on their target core) |
| source → destination | raw bytes, whole tiles | `noc_async_read` issued by the out holder, source address from in0's `TensorAccessor`, destination = out's shard by address | no CB, no compute kernel, no conversion; one NoC crossing per non-local tile |
| out's shard on its holder core (L1) | bfloat8_b tiles as stored, slot `(i, j)` at `i·block_width_tiles + j` | written once per tile by the issuer whose row range holds it | complete when both issuers' read barriers return; consumed by the next program |
| deferred PUSH contract | same bytes | in holder `noc_async_write` into the out holder's shard address + `noc_async_write_barrier`; no semaphore (no in-program consumer) | out side addressed by an out `TensorAccessor`; kept reachable by the symmetric piece walk |

TARGET connection: TARGET is `{dtype: [bfloat8_b], layout: [TILE_LAYOUT]}` (`eval/golden_tests/g6_reshard/feature_spec.py`). Placements are case arguments, not a TARGET axis; PULL is the single scheme for every legal placement pair, so no placement value is a knob-turn or scheme-change unlock. Structural impossibilities beyond `INVALID = []`: none.

Catalog entries that set knobs: `double_buffer`, `trid_double_issue` (all runs in flight, one barrier); `read_coalescing`, `tile_reorder` (whole tiles, maximal contiguous runs, merged full-width rows); `tensix_all_reduce` (pull from an immutable resident shard needs no handshake; "prefer pull"); `split_reader`, `noc_placement`, `bank_stagger` (the lamps above).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one out block: out's shard on its holder core, clipped at the tensor edge |
| Grid | the block-holding cores of out's placement — one rectangle: ROW_MAJOR `(x0 … x0+num_block_cols−1, y0 … y0+num_block_rows−1)`, COL_MAJOR `(x0 … x0+num_block_rows−1, y0 … y0+num_block_cols−1)`. P: (0,0)-(7,7), 64 cores; O1: (1,0)-(8,6), 56 cores. Cores of `memory_config`'s grid that hold no block run no kernel |
| Per-core work | 1 block; issuer 0 takes rows `[block_row_begin, block_row_begin + ⌈rows/2⌉)`, issuer 1 the rest; both take columns `[block_col_begin, block_col_end)`. P: 16 tiles, ≤ 8 runs per issuer; O1: ≤ 20 tiles, ≤ 12 runs |
| Remainder | `tensor_height_tiles = d0·d1·⌈d2/32⌉` (each image tile-padded on its own), `tensor_width_tiles = ⌈d3/32⌉`; `num_block_rows = ⌈tensor_height_tiles / block_height_tiles⌉`, `num_block_cols = ⌈tensor_width_tiles / block_width_tiles⌉`; last block extents are the remainders. The same `ceil` rule gives in0's block grid. `split_work_to_cores` is not used: the placement fixes which core owns which block |
| Regime selection | one built regime (PULL); no predicate, no regime-pinned tests needed |

## Circular Buffers

None. Reads land directly in out's shard (by its buffer address); the source is read where it lives. No reserve/push/wait/pop exists, so no CB sync or ownership question arises. The two issuers write disjoint row ranges of the same out shard.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| — | — | — | — | no CB: pack-into-destination (`l1-footprint-discipline.md` Rule 3.1); see `l1_ledger.md` | — | — | — | — |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `locate_pieces` | issuer sub-block `issuer_rows × block_width_tiles` | no (integer walk over in0's block grid) | none | none | — |
| 2 | `move_pieces` | each piece `piece_rows × piece_width`, as maximal runs | `TensorAccessor` for the source address; raw `noc_async_read` per run | none (source: in0's shard on its holder, by NoC address) | none (destination: out's shard, by local address) | — |
| 3 | `complete_block` | — | raw `noc_async_read_barrier` | none | none | out's sub-block complete |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `locate_pieces` | raw_api | `get_arg_val<uint32_t>` / `get_common_arg_val<uint32_t>` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:148`, `:168` | the issuer rectangle (per core); in0 / out buffer addresses (common) | — | — | the issuer rectangle carries `issuers_per_core`'s split |
| `locate_pieces` | helper | `TensorAccessor` constructed from `TensorAccessorArgs` (sharded, all compile-time) | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` (`get_noc_addr`), host `ttnn.TensorAccessorArgs(input_tensor)` `ttnn/cpp/ttnn-nanobind/tensor_accessor_args.cpp:26-28`, `.get_compile_time_args()` `:35` | in0's address, `aligned_page_bytes` | — | — | none (in0's placement) |
| `move_pieces` | helper | `TensorAccessor::get_noc_addr(page_id, offset, noc)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | `page_id` of the piece's first tile; `noc` = this issuer's NoC (default `noc_index`) | — | — | — |
| `move_pieces` | raw_api | `noc_async_read(src_noc_addr, dst_l1_addr, bytes)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `bytes = run_tiles·aligned_page_bytes` | — | — | run extent (derived) |
| `complete_block` | raw_api | `noc_async_read_barrier()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1921` | this issuer's NoC | — | — | — |
| profiling (BB-21) | raw_api | `DeviceZoneScopedN(name)` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` (compiles out in the plain build, `:1047`) | one zone around `locate_pieces` + `move_pieces` (issue), one around `complete_block` (wait) | — | — | — |
| host: issuer placement | host | `ttnn.DataMovementConfigDescriptor(processor, noc)` or `ReaderConfigDescriptor` (RISCV_1/NoC0) and `WriterConfigDescriptor` (RISCV_0/NoC1) | `tt_metal/api/tt-metalium/program_descriptors.hpp:96-102`; defaults `tt_metal/impl/kernels/kernel_types.cpp:18-19`, `:36-37` | issuer 0: RISCV_1 + NOC_0; issuer 1: RISCV_0 + NOC_1 | — | — | `issuers_per_core` |
| host: dispatch | host | `ttnn.generic_op([input_tensor, out], program_descriptor)`; `ProgramDescriptor.custom_program_hash` | `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:83`; `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1141` | output last in the tensor list | — | — | — |

Helpers considered and rejected (for every `raw_api` row above):

| Candidate | File:Line | Why it cannot express the planned operation |
|-----------|-----------|---------------------------------------------|
| `dataflow_kernel_lib::local_addr` / `set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57`, `:75`, `:95` | self-aimed only (source at `my_x/my_y` of the issuing core) and single-packet only (`:66-68`); PULL reads from a remote core for every non-local tile (P: 1880 of 2048) and I0/I3/I5 runs exceed one packet |
| `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147`, `:158` | scoped waits for a pipelined multi-batch loop; the block issues one batch and waits once |
| `McastArgs` / `ttnn.Mcast` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:18` | one source to many receivers; every tile of a reshard has exactly one destination |
| `noc_async_read_shard` (whole-shard read through the accessor) | `tech_reports/tensor_accessor/tensor_accessor.md:141` | reads a whole in0 shard into a contiguous buffer; a piece is a sub-rectangle of a shard landing at the out shard's stride |
| `TensorAccessor::num_contiguous_pages` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:192` | stops at one shard row and knows nothing of the destination stride; run length is already known from the piece geometry without its divisions |
| compute helpers (`eltwise` `copy`, tilize/untilize) | `ttnn/cpp/ttnn/kernel_lib/eltwise/`, `tilize_helpers.hpp`, `untilize_helpers.hpp` | unpack → pack is forbidden by rule 1 and would regroup bfloat8_b |

## Host Contract

### Registry model (op file)

| Declaration | Value |
|-------------|-------|
| `INPUT_TAGGERS` | `{}` — no shape-derived axis; placements are arguments |
| `SUPPORTED` | `{"dtype": [ttnn.bfloat8_b], "layout": [ttnn.TILE_LAYOUT]}` |
| `EXCLUSIONS` | `[]` |
| `validate(input_tensor, memory_config)` | builds `{dtype, layout}` from `input_tensor`; raises `UnsupportedAxisValue` / `ExcludedCell` (`ttnn.operations._op_contract`); the first line of `g6_reshard` |

### Input validation (after `validate()`, before any device work)

| Condition | Error |
|-----------|-------|
| `input_tensor` not on device, not BLOCK_SHARDED, not L1, or without a shard spec | `ValueError` |
| `memory_config` not BLOCK_SHARDED, not L1, or without a shard spec | `ValueError` |
| `memory_config` shard dims not multiples of 32 | `ValueError` |
| `memory_config` grid not one rectangle, or fewer grid columns/rows than `num_block_cols`/`num_block_rows` (ROW_MAJOR) — rows/cols swapped for COL_MAJOR | `ValueError` |
| `memory_config` grid outside `device.compute_with_storage_grid_size()` | `ValueError` |
| rank ≠ 4, or logical H or W not a multiple of 32 | `ValueError` |
| out shard bytes > 1347776 (BB-05) | `RuntimeError` |

No other refusal: every other legal placement on either side is served (rule 2).

### Program build and cache (BB-01, BB-07, BB-08, BB-17, BB-21)

| Item | Decision |
|------|----------|
| Programs per call | one `ttnn.generic_op` dispatch; out allocated by `ttnn.allocate_tensor_on_device(shape, dtype, layout, device, memory_config)` (positional) before it; nothing else touches the device |
| Kernels | one dataflow source, two `KernelDescriptor`s on the same core range (out's block-holder rectangle): issuer 0 and issuer 1; no compute kernel, no CBs, no semaphores |
| Per-call values | in0 and out buffer addresses (identical on every core: both are lockstep-allocated sharded L1 buffers), passed as common RT args |
| Placement values | in0's placement → accessor CT args; `in_block_*_tiles`, `block_width_tiles`, `tensor_width_tiles`, `aligned_page_bytes` → CT args; each issuer's rectangle → per-core RT args |
| Host plan cache | key = `(padded shape, dtype, layout, in0 placement (grid start/end, shard shape, orientation), out placement (same), device id)`; value = the built `ProgramDescriptor`. A hit rewrites only the two address common RT args and dispatches; the descriptor is never rebuilt (BB-07). The same key, hashed, is set as `custom_program_hash`, so generic_op's program cache key holds both placements (I0 and I3 differ only in orientation, P and O1 only in out's placement) |
| Plan report (BB-08) | one plan; the op exposes the plan of the last call (regime `PULL`, issuer rectangles) |
| Role dump (BB-21) | host function returning, per core of the kernel range, its out block `(r, c)`, each issuer's row range and piece count |
| Core ranges (BB-17) | one rectangle per kernel (both kernels share it) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Tail shards read at the wrong stride (R1) | P's input has a 1-tile-high last block row and a 2-wide last block column inside 7 × 3 shards; O1's output a 2-wide last column in 8 × 5 shards | slot stride is always the **shard** width (`in_block_width_tiles`, `block_width_tiles`); runs merge only when the piece width equals both shard widths; the source address of each piece comes from in0's accessor. P and O1 are the cases that catch it |
| Orientation or grid origin wrong (R1) | I3 (COL_MAJOR input), O1 (COL_MAJOR output at (1,0)), I5 (input at (2,1)) | input side: the accessor carries in0's own core order (no hand-built table); output side: the host maps out block `(r, c)` to its core by `memory_config`'s orientation and origin, and the kernel range is exactly the block-holder rectangle |
| `TensorAccessorArgs` page model differs from the 2-D view | page ids must be `tile_row·tensor_width_tiles + tile_col` of the padded 2-D view | first device probe: run P and compare exactly (the golden P cell); if the accessor's page model disagrees, the source base per piece is computed from a host-built in-block → NoC-coordinate table in common RT args (≤ in0's block count words) instead of the accessor |
| NoC coordinate space of issuer 1 | NoC0 and NoC1 address cores differently (`local_copy_helpers_dataflow.hpp` warning on `my_x/my_y`) | every accessor call passes the issuer's own NoC (`noc_index` default); no coordinate is hard-coded to NoC0 |
| Cache replays a stale program or address (R6) | out is a fresh allocation every call (the repeat loose case keeps three alive); I0 vs I3 and P vs O1 share shape and dtype | the plan-cache key and `custom_program_hash` hold both placements; addresses are rewritten on every call. Verify on the first build that generic_op's cache hit re-applies common RT args (the `placements_one_process` and `repeat_determinism` loose cases fail otherwise); if the Python binding returns copies of `common_runtime_args`, assign the patched list back through the descriptor |
| Transactions in flight at return (BB-03) | another program starts right after | each issuer ends with `noc_async_read_barrier` on its own NoC; PULL issues no writes, atomics or semaphore increments |
| One-packet API on long runs | I0/I3/I5 runs are 16 tiles = 17408 B | plain `noc_async_read`; the one-packet lamp carries the 15-tile clamp |
| Aliasing (R8) | I0 has identical placements on both sides | out is always a new allocation; in-place tiles are copied by a self-aimed read into out's own buffer; in0 is never written |
| A conversion sneaks in (R2) | an implementer reaching for a compute copy | no compute kernel exists in the program; the move is bytes over the NoC |

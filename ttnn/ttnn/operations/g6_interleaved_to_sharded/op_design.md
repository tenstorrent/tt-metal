# Operation Design: g6_interleaved_to_sharded

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement |
| Goal | Move every tile page of a DRAM-interleaved bfloat16 TILE tensor, bit for bit, straight into the L1 shard `memory_config` gives it, in one device program: each shard's owner core reads its own pages from DRAM directly into its shard. No staging buffer, no compute engine, no core-to-core traffic. |
| Math | `y[i] = x[i]`. Folded tile grid: `tensor_tile_rows = num_images * ceil(H/32)`, `tensor_tile_cols = ceil(W/32)`. Tile `T(r, c)` is input page `r * tensor_tile_cols + c`; it belongs to shard `s = c div shard_tile_cols` (width) or `s = r div shard_tile_rows` (height), lives on core `G[s]`, at shard position `(r - row_begin(s)) * shard_tile_cols + (c - col_begin(s))`. `G` = cores of the shard grid in the orientation's order. |
| References | Spec `eval/prompts/g6_interleaved_to_sharded.txt`; `docs/explorations/g6_interleaved_to_sharded/{DOSSIER,math,tensix,risks,lessons}.md`; `eval/golden_tests/g6_interleaved_to_sharded/{feature_spec,helpers,test_regression}.py`; `.claude/references/{blocking-model,l1-footprint-discipline,noc-alignment-in-tt-metal}.md`; sharded page map `tt_metal/impl/buffers/buffer.cpp:424-470`; catalog entries listed under Perf lamps |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | rank 2 (Phase 0) or 4 (refinement); bfloat16; TILE 32x32; DRAM interleaved; on device | — | buffer address: common RT; geometry: CT |
| `memory_config` | `ttnn.MemoryConfig` | yes, positional | buffer L1; `WIDTH_SHARDED` (Phase 0) or `HEIGHT_SHARDED` (refinement); `shard_spec` with shard dims multiples of 32 that cover the padded tile grid; `ROW_MAJOR` or `COL_MAJOR` | — | grid → kernel core ranges; orientation → core order `G` (host); shard shape → CT row stride + per-core RT rectangles |
| `issuers_per_core` (module knob, `KNOBS` dict in the program-descriptor module, not a public argument) | int | no | {1, 2} | 2 (Refinement 2; was 1) | host: number of KernelDescriptors (which RISC/NoC each runs on: `issuer_placement`; how the block is cut: `issuer_split`) |
| `issuer_placement` (module knob) | str | no | keys of `ISSUER_PLACEMENTS` | `"ncrisc_noc0+brisc_noc1"` | host: (RISC, NoC) of issuer 0, 1; issuer 0 takes the larger piece of an uneven split |
| `issuer_split` (module knob) | str | no | {`"even"`, `"noc_balanced"`} | `"noc_balanced"` | host: NOC0 issuer takes `C/(R+C)` of the block (R/C = most owners in one grid column/row), `"even"` = near-equal halves |
| `bank_stagger` (module knob) | bool | no | {False, True} | False | CT switch + per-issuer RT `col_rotate`: rotate each tile row's column walk so the first page sits on bank `shard mod n_banks` |
| `partial_tile_read` (module knob, same dict) | str | no | {`"valid_rows"`, `"face_rows"` (lamp L2, two spans), `"whole_tile"`} | `"valid_rows"` | CT: `partial_span_bytes` and the second span's offset/bytes |

The public signature is exactly `g6_interleaved_to_sharded(input_tensor, memory_config)`. Knobs are module constants, never call arguments; every knob value is part of the descriptor cache key.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(..., H, W)`, rank 2 (Phase 0) or 4; leading dims fold into tile rows; H not a multiple of 32 (`h_non_aligned`, H = 1 in production); W multiple of 32 |
| Dtype | bfloat16 |
| Layout | TILE, 32x32 tiles with 16x16 faces |
| Memory | DRAM interleaved (pages round-robin over the DRAM banks) |

### Output

| Property | Value |
|----------|-------|
| Shape | input's logical shape |
| Dtype | input's (bfloat16) |
| Layout | TILE |
| Memory | L1, exactly `memory_config` (memory layout, shard grid, shard shape, orientation). Allocated by the entry point: `ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, device, memory_config)` |

Padding positions of output tiles (rows `>= h_rem` of the last tile row of each image) are not part of the result and are left as the read spans leave them.

## Blocking Model

Index space: the folded 2-D tile grid `tensor_tile_rows x tensor_tile_cols` (per-image padding, so `image_tile_rows = ceil(H/32)` and `tensor_tile_rows = num_images * image_tile_rows`), plus the sub-tile row axis that the partial-row read trims.

Scope names (blocking-model.md §3): `tensor_*` = whole tensor, `shard_*` = the placement's shard shape, `block_*` = one issuer's rectangle on one core. Production values:

| Symbol | p1: (1, 5120), width (32,160) on (0,0)-(7,3) | p2: (1,1,1,128), height (32,128) on (0,0) |
|---|---|---|
| `num_images`, `image_tile_rows`, `h_rem = H mod 32` | 1, 1, 1 | 1, 1, 1 |
| `tensor_tile_rows x tensor_tile_cols` | 1 x 160 | 1 x 4 |
| `shard_tile_rows x shard_tile_cols` | 1 x 5 | 1 x 4 |
| `num_shards` | 32 (= grid cores) | 1 |
| block per owner core | 1 x 5 tiles | 1 x 4 tiles |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_row` (folded: images x tile rows per image) | independent — every output page is one input page; no result spans it | `block_tile_rows` | the core's whole assignment: width → `tensor_tile_rows`; height → `min(shard_tile_rows, tensor_tile_rows - row_begin)`. Equals 1 in production because the tensor has one tile row (H = 1), not because the knob is collapsed | per-core RT value, computed once on host by the work-split function from (shape, shard spec) | width: every owner core takes all rows; height: shard `s` = rows `[s*shard_tile_rows, ...)` on `G[s]` — fixed by `memory_config` | height scheme: knob-turn (host rectangle + SUPPORTED; same kernel) |
| `tile_col` | independent — same reason | `block_tile_cols` | width → `min(shard_tile_cols, tensor_tile_cols - col_begin)` (5 for p1); height → `tensor_tile_cols` (4 for p2) | per-core RT, same host function | width: shard `s` = cols `[s*shard_tile_cols, ...)` on `G[s]` (p1: 32 cores) — fixed by `memory_config` | built |
| `image` (leading dims) | independent — same reason; images are padded independently | folded into `tile_row`; the per-image partial row is found by `r mod image_tile_rows` | `num_images` = 1 (rank 2) | CT `image_tile_rows`, host | via `tile_row` | rank 4: knob-turn (host folding; predicate already per image) |
| `row_in_tile` (rows of a tile that hold logical values) | independent — rows `>= h_rem` of an image's last tile row are padding (math.md M5), not part of `y` | `partial_span_bytes` = bytes read from a tile of a partial tile row; `full_span_bytes` = `tile_bytes` for full rows | `face_bytes + h_rem * face_row_bytes` = 512 + 32 = **544 B** at `h_rem = 1`; full rows 2048 B | CT, computed once on host from (`H mod 32`, tile and face shape, element size) and the `partial_tile_read` knob | not split: a tile's span is one NoC request | lamp L2 (two face-row spans) |
| `col_in_tile` | independent — W is a multiple of 32 in TARGET, so every column of a tile is logical | none: always the full face-row width (a span covers whole face rows) | 32 | — (fixed by TARGET: `w_non_aligned` is outside TARGET) | not split | none |

No axis is dependent and no operand is reuse-shared (operand-reuse check: the only operand is `x`, which varies along the split axis; each page is read by exactly one core).

**Block.** One block = one issuer's whole rectangle on its core, `block_tile_rows x block_tile_cols`; `num_blocks_this_core = 1`. Nothing bounds the block below the shard: it needs no L1 of its own (every read lands at its final address), and nothing inside this program consumes the data, so a smaller block would only add barriers. "A shard hands you the block size" — the block is the shard.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| — (no CB exists) | in-flight reads per barrier (not a CB depth) | every read of the block outstanding, one barrier per block | All reads overlap each other on the NoC/DRAM. No RISC hands data to another RISC, so no CB and no double buffer. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| NoC DRAM-read address congruence: local L1 and NoC address equal mod `NOC_DRAM_READ_ALIGNMENT_BYTES` (64 B on Blackhole, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`; rule in `noc-alignment-in-tt-metal.md` §1) | span start offset within a tile | every span starts at tile byte 0 of both the DRAM page and the shard slot. The shard base is DRAM-aligned because the allocator initializes L1 banks with the DRAM alignment (`tt_metal/impl/allocator/bank_manager.cpp:175-176`). The host asserts `output.buffer_address() % ttnn.get_dram_alignment() == 0`. | silent corruption of the landed bytes (watcher stops it in `--dev`) |
| `NOC_MAX_BURST_SIZE` = 256 words x 64 B = 16 KiB (`noc_parameters.h:289-290`) | bytes per read | span ≤ `tile_bytes` = 2048 ≤ 16 KiB, so every read is one packet. Issue with `noc_async_read<tile_bytes>` (one-packet path); the kernel `static_assert`s `tile_bytes <= NOC_MAX_BURST_SIZE`. | a one-packet issue of a longer span truncates it |
| Face layout of a 32x32 tile (faces 0..3 of 16x16, row-major inside a face; row `k < 16` of the tile is bytes `[k*32, k*32+32)` of face 0 and the same offsets of face 1) | `partial_span_bytes` | the host derives `face_row_bytes`, `face_bytes`, `tile_bytes` from the tensor's tile (`tile.tile_shape`, `tile.face_shape`, element size). `validate()` refuses any tile other than 32x32 (ValueError). | a span that cuts through the wrong bytes: wrong values |
| Shard geometry: partial last shard and surplus grid cores | `block_tile_rows`, `block_tile_cols` | `num_shards = ceil(tensor_tile_cols / shard_tile_cols)` (width) or `ceil(tensor_tile_rows / shard_tile_rows)` (height); extents clamp with `min(...)`; cores `G[s >= num_shards]` get an empty rectangle and return at once | reads of pages past the tensor; writes past the shard |
| Shard count vs grid cores | `num_shards` | `num_shards <= len(G)`, else ValueError (malformed) | shards with no owner: missing data |
| L1 bound BB-05: 1347776 B per core, output shard included | shard bytes | `shard_tile_rows * shard_tile_cols * tile_bytes + 0 <= L1_BUDGET_BYTES`, else ValueError (the op adds 0 B) | refused launch next to the model's tensors |
| Kernel-config region (BB-04/BB-18) | runtime-arg count | constant count per kernel (one rectangle per core + 2 common addresses), independent of tile count | config overflow on large shards |

### Regimes

| Regime | Status | Predicate | Block | Data movement (vs. minimum) | What a bigger block buys |
|--------|--------|-----------|-------|------------------------------|--------------------------|
| **A. owner-fetch, valid-row span** | built (default for partial tile rows) | per tile row `r`: `h_rem != 0 and r % image_tile_rows == image_tile_rows - 1` and `partial_tile_read == "valid_rows"`. Every tile row of both production shapes (H = 1). | `block_tile_rows x block_tile_cols`; one read of `partial_span_bytes` per tile | Minimum boundary: each input page crosses DRAM once, each output page never. Met: 1 crossing per page, 0 core-to-core bytes. Bytes are **below** the tiled tensor: 544 B/page (p1: 87,040 B against 327,680 tiled). The logical floor (64 B/page = 10,240 B) is not reached by one request: face-1 row 0 lies 512 B after face-0 row 0, so a single contiguous span must cross face 0. Reaching the floor costs a second request per tile (lamp L2). | Fixed costs per block: one kernel launch per core, one TensorAccessor setup, one read barrier. A whole-shard block pays each once per issuer per call (the intended frequency). Interleaved neighbours sit in different banks, so one request per page is already the coarsest legal transfer. |
| **B. owner-fetch, whole-tile read** | built | per tile row `r`: `h_rem == 0 or r % image_tile_rows < image_tile_rows - 1` (fully valid row). Also every row when `partial_tile_read == "whole_tile"` (A/B knob only, never the default). No production tile row; reached by H > 32, e.g. the regime-pin case (33, 5120) and `height_multi_shard` (2,1,33,128) | same block; one read of `full_span_bytes = tile_bytes` per tile | Minimum: 2048 B per fully valid page, once; 0 core-to-core bytes | same as A |
| C. reader-and-forward (non-owner cores read pages, write them into the owner's shard) | rejected — superseded by A/B plus lamp L1 | — | k staged pages per reader | Same DRAM bytes, plus one core-to-core write per forwarded page and a second serialized hop (read latency + write latency), plus staging L1 on readers. Its only gain, more issuers, is available on the owner's idle BRISC (lamp L1) at zero core-to-core bytes. Ingestion into the owner's L1 bounds both regimes equally. Applies to p2 (4 pages, 1 owner) too: 4 back-to-back requests on the owner finish in one DRAM round trip. | — |
| D. single-core gather (one core reads all pages, writes the other shards) | rejected — dead end, superseded by A | — | whole tensor | Same DRAM bytes, plus (S-1)/S of the pages over the NoC a second time; one RISC serializes every request (p1: 160) | — |
| E. staged two-step (interleaved L1 copy, second sharded tensor, host round trip, or calling an existing ttnn data-movement op) | rejected — forbidden by spec rule 1 and BB-01 | — | — | every byte crosses twice; a second dispatch | — |

**Regime selection function** (host-checkable; the kernel evaluates it per tile row from CT args):

```
span_bytes(r) = partial_span_bytes   if h_rem != 0 and r % image_tile_rows == image_tile_rows - 1
              = full_span_bytes      otherwise                                   (full_span_bytes = tile_bytes)
partial_span_bytes = tile_bytes                                        if partial_tile_read == "whole_tile" or h_rem == 0
                   = face_bytes + h_rem * face_row_bytes               if 0 < h_rem <= face_h          (face_h = 16)
                   = 3 * face_bytes + (h_rem - face_h) * face_row_bytes  if face_h < h_rem < 32
```

At bf16 32x32: `face_row_bytes = 32`, `face_bytes = 512`, `tile_bytes = 2048`. Check values: `h_rem=1` → 544, `h_rem=16` → 1024, `h_rem=17` → 1568, `h_rem=31` → 2016.

**Regime-pinned tests** (acceptance test): A alone — p1 (`flux58_width_8x4_row_major`), plus `special_values_bit_exact` and `poisoned_padding` on p1. A and B in one block — `regime_pin_33x5120_width` ((33, 5120), shard (64,160) on (0,0)-(7,3): tile row 0 full, tile row 1 partial; Phase 0). A and B across images and cores — `height_multi_shard_2x1x33x128` (refinement; xfail-strict until height is SUPPORTED).

### Traffic ranking

Count crossings per tier for each candidate split. Bytes are per tensor at p1, with the valid-row span.

| Rank | Candidate split | DRAM crossings of `x` | Core-to-core bytes | Fan-out | Verdict |
|------|-----------------|-----------------------|--------------------|---------|---------|
| 1 | cut along the **target's shard axis**, owner-fetch (width: tile columns; height: tile rows) | 1 per page (87,040 B) | 0 | 1 reader per page; the response terminates on its owner | **chosen**: minimum on every tier |
| 2 | cut finer than the shard, or along the other independent axis (pages read by non-owners) = regime C | 1 per page | +1 write per forwarded page (≤ 87,040 B) | 2 hops | strictly more traffic |
| 3 | no cut (one core) = regime D | 1 per page | (S-1)/S of the pages | 1 → S | strictly more traffic, serialized issue |
| — | split a dependent axis + combine | n/a | n/a | n/a | no axis is dependent: no result spans any axis |
| — | broadcast a reuse-shared operand | n/a | n/a | n/a | no operand is read by more than one core |

Occupancy is not used to rank. The core count follows from split 1: one owner per shard (32 cores for p1). This is the blocking decision, not a limit imposed by the tensor's shape. More issuers come from inside each owner (lamp L1), not from more cores.

**Stall shadow.** The only wait is each issuer's final read barrier. Nothing else in the program is independent of it (no compute, no write, no other output), so nothing can be scheduled into it. All reads of the block are issued before the barrier, so they already overlap each other.

### Block schedule

```cpp
// Per issuer on each owner core. Phase 0: one issuer (NCRISC, NOC0); num_blocks_this_core = 1
// (the issuer's whole shard rectangle, block_tile_rows x block_tile_cols).
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    fetch_block_into_shard(block_idx);  // one NoC read per tile; span chosen per tile row; lands at the tile's shard slot
    retire_block(block_idx);            // one read barrier: every read of the block has landed (BB-03)
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `fetch_block_into_shard` | `block_tile_rows x block_tile_cols` tiles, walked row-major; `span_bytes(r)` once per tile row | the output shard (caller's buffer) is the destination; nothing staged | TensorAccessor construction once per kernel; span selection once per tile row; one read request per tile (the transfer itself: interleaved neighbours are never adjacent in one bank, `tensor_accessor.h:559-560`); no CB handshake, no init, no multicast; zone `I2S-FETCH` once |
| `retire_block` | the same block | the landed shard | one `noc_async_read_barrier` on the issuer's own NoC per block = once per call per issuer; zone `I2S-RETIRE` once |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 issuers per core: default 1 (NCRISC on NOC0 issues the whole block) | The op has no compute and no writer, so BRISC idles. With 5 requests per core (p1), issue time is comparable to one DRAM round trip. Catalog `split_reader`: wins only when issue-bound. `noc_placement`: reads on NOC1 can be slower for spread placements. RB-013: a second issuer does not help bank-bound traffic. | `issuers_per_core = 2`: NCRISC/NOC0 takes block columns `[0, ceil(n/2))`, BRISC/NOC1 the rest (rows halved when the block has one column). Host-only change (second KernelDescriptor, same source). Measure p1 and p2 on unprofiled trace replays (RB-053). **Refinement 2: lit, default 2 with the `noc_balanced` split** (p1 804 → 758 ns; see changelog). |
| L2 partial-tile read shape: default one span (544 B, 1 request per tile) | RB-017: small reads are bound by request count, so the default keeps one request per tile. Aggregate DRAM bank time still scales with bytes: p1 moves 87,040 B over the banks (about 10.9 KB per bank) against 10,240 B with two spans. | Two face-row spans per partial tile: `[0, h_rem*face_row_bytes)` and `[face_bytes, face_bytes + h_rem*face_row_bytes)` (2 requests, 64 B/tile at `h_rem = 1`, the logical floor). Both starts are tile-relative multiples of 32, so DRAM congruence holds (both offsets ≡ 0 mod 64 at `h_rem = 1`: 0 and 512). |
| L3 issue order / bank stagger: default shard order (block columns ascending) | Core `s` starts on bank `(s * shard_tile_cols) mod n_banks`. For p1 (`shard_tile_cols = 5`, coprime with 8 banks) the 8 cores of a grid row start on 8 distinct banks. For placements whose `shard_tile_cols` shares a factor with the bank count (e.g. (32, 640) on 8 cores: 20 tiles per core, starts on banks 0, 4, 0, 4, ...) the cores pile onto the same bank first (catalog `bank_stagger`, RB-015). | Rotate each issuer's walk to start at tile `shard_index mod block_tiles` and wrap (CT switch); measure on p1 and the (32, 640) placement. **Refinement 2: built as `bank_stagger`, measured flat/worse on (32, 640) (979 → 1023 ns), parked at False.** |

Overlap lamp: not applicable — there is no compute to overlap; every read of the block is in flight at once. Grid-synchronization lamp: not applicable — there is no collective, multicast or barrier across cores. The only per-core fixed cost is its own launch, and fewer cores would require regime C.

Catalog entries behind these knobs (`ttnn/ttnn/operations/examples/`):

| Entry | Informed | How |
|-------|----------|-----|
| `width_split` | core-assignment | A one-tile-row tensor must be split by tile columns to fill cores. The width placement already does so (p1: 32 cores). |
| `read_coalescing` | transfer size | Coalescing needs physical contiguity. Consecutive interleaved page ids sit in different banks, and pages adjacent in one bank (ids `p`, `p + n_banks`) are never adjacent in a shard. One request per page is the coarsest legal read. |
| `tile_reorder` | transfer shape | Move whole contiguous spans, not scattered sub-tile pieces, by default → one-span default (L2 measures the alternative) |
| `double_buffer`, `trid_double_issue` | in-flight depth | Issue a block of reads, then one barrier (never read-one/barrier). Trids pay only when a consumer drains blocks; none exists here. |
| `split_reader`, `noc_placement` | lamp L1 | second DM RISC only if issue-bound; reads on NOC0 by default |
| `bank_stagger` | lamp L3 | rotate the start when cores line up on banks |

### Statement of the scheme

Phase 0 realizes owner-fetch on the shard grid of `memory_config` (multi-core from day 1: 32 cores for p1). Each owner reads its whole shard rectangle from DRAM into its own L1 with one request per tile, then issues one barrier. Every knob is a parameter:
- block extents: per-core RT values from one host work-split function;
- spans: CT values from one host geometry function;
- issuers and read shape: the `KNOBS` dict.

No count, extent or grid is inlined in a kernel.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM input page `p = r * tensor_tile_cols + c` | bf16 tile, faces 0..3, `tile_bytes` = 2048 B page in bank `p mod n_banks` | `TensorAccessor` (interleaved, DRAM) `get_noc_addr(p)` | base address from a common RT arg |
| DRAM → owner L1 | `span_bytes(r)` bytes from tile byte 0 (544 B partial, 2048 B full) | `noc_async_read<tile_bytes>(src, dst, span_bytes(r))` issued by the owner's issuer (NCRISC/NOC0 by default) | `dst = out_shard_base + shard_pos(r, c) * tile_bytes`; the read response lands at the final address — no CB, no staging, no second hop |
| Owner L1 shard (the output tensor) | bf16 tile at shard position `(r - row_begin) * shard_tile_cols + (c - col_begin)` | final placement | the next program of the model reads it in place |
| Completion | — | `noc_async_read_barrier(noc)` once per block per issuer | no writes, no semaphores, no atomics: nothing else can be in flight at return (BB-03) |

No regime needs core-to-core communication, so there is no Tensix-to-Tensix contract.

**TARGET axes seen as placements** (each value against this op's axis characters):

| TARGET axis value | Cut | Class | Status |
|-------------------|-----|-------|--------|
| `out_scheme = "width_sharded"` | tile columns (independent) | knob-turn | built (Phase 0) |
| `out_scheme = "height_sharded"` | folded tile rows (independent; may straddle images) | knob-turn: host rectangle per shard + SUPPORTED. Same kernel, same predicate. | refinement |
| `rank = 4` | leading dims fold into `tile_row` with per-image padding | knob-turn: host folding | refinement (with height) |
| `alignment = "h_non_aligned"` | sub-tile rows trimmed by `partial_span_bytes` | built | Phase 0 |

Structural impossibilities: none beyond `feature_spec.py` (INVALID is empty, correctly).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = one issuer's rectangle of its core's shard |
| Grid | the shard grid `memory_config.shard_spec.grid` as given (one rectangle in both production cases; BB-17) |
| Core order `G` | `ttnn.corerange_to_cores(grid, None, row_wise=(orientation == ttnn.ShardOrientation.ROW_MAJOR))` — the same call the sharded buffer's page map uses (`tt_metal/impl/buffers/buffer.cpp:440`). Never assume ROW_MAJOR. |
| Shard → core | shard `s` on `G[s]` for `s < num_shards` |
| Per-core work (width) | rows `[0, tensor_tile_rows)`, cols `[s*shard_tile_cols, min((s+1)*shard_tile_cols, tensor_tile_cols))` |
| Per-core work (height) | rows `[s*shard_tile_rows, min((s+1)*shard_tile_rows, tensor_tile_rows))`, cols `[0, tensor_tile_cols)` |
| Shard slot of tile (r, c) | `(r - row_begin) * shard_tile_cols + (c - col_begin)` (row stride is the **shard** width, never `tensor_tile_cols`) |
| Geometry (alignment-aware) | `tensor_tile_cols = ceil(W/32)`; `image_tile_rows = ceil(H/32)`; `num_images = prod(shape[:-2])` (1 for rank 2); `tensor_tile_rows = num_images * image_tile_rows`; `h_rem = H mod 32`. Per image, never `floor(total_rows/32)`. |
| Remainder | `num_shards` by `ceil`; last shard's extent clamped by `min`; cores `G[s >= num_shards]` get an empty rectangle and return at once |
| Issuers | `issuers_per_core = 1`: issuer 0 of `issuer_placement` (NCRISC/NOC0 by default) takes the rectangle. `= 2` (default): block columns are cut between the two issuers (rows when `n = 1`; an issuer with nothing gets an empty rectangle): `"even"` gives issuer 0 `ceil(n/2)`; `"noc_balanced"` (default) gives the NOC0 issuer `round(n * C/(R+C))`, because read responses on NOC0 share their last leg with the owners of one grid column (R of them) and on NOC1 with the owners of one grid row (C). |
| Kernel groups | one reader kernel (two with `issuers_per_core = 2`) on the whole shard grid CoreRangeSet; no compute kernel, no writer kernel |
| Production | p1: 32 owner cores x 1 block of 1x5 tiles; p2: 1 owner core x 1 block of 1x4 tiles |

Regime selection is per tile row and does not depend on the grid (see Regimes). The regime-pinned tests are listed there.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| — none — | — | — | — | Every read lands at its final address in the output shard (pack-into-destination, l1-footprint-discipline Rule 3 pattern 1). No RISC hands data to another RISC, and no compute kernel exists. Inventory = 0 op-added buffers. | — | — | — | — |

The output shard address reaches the kernel as a **common runtime arg**, not as a CB aliased on the sharded tensor. A cached descriptor would hold the first call's output buffer pointer in such a CB, while a common arg is patched per call (RB-041). The full ledger is in `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `fetch_block_into_shard` | `block_tile_rows x block_tile_cols` tiles; `span_bytes(r)` per tile row | no — raw NoC read (see API Mapping) | none: DRAM interleaved pages via `TensorAccessor` | none: output shard L1 at `out_shard_base + slot * tile_bytes` | reads outstanding |
| 2 | `retire_block` | same block | no — raw barrier | — | — | every read landed; the shard rectangle holds its tiles |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `fetch_block_into_shard` | raw_api | `TensorAccessor` (interleaved DRAM specialization), ctor + `get_noc_addr(page_id, offset=0, noc)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:493` (struct), `:515` (ctor), `:554` (`get_noc_addr`) | `TensorAccessorArgs<cta_offset, crta_offset>`, base address (common RT), `tile_bytes`; host side `ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()` (`.claude/references/generic_op_template/template_op_program_descriptor.py:114`) | — (DRAM) | — | none (addressing only) |
| `fetch_block_into_shard` | raw_api | `noc_async_read<max_page_size>(src_noc_addr, dst_local_l1_addr, size, noc)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `max_page_size = tile_bytes` (≤ `NOC_MAX_BURST_SIZE`, `noc_parameters.h:290`: one-packet path); `size = span_bytes(r)`; `noc` = the issuer's NoC | — | — | `span_bytes` (CT from `partial_tile_read`); walk extents `block_tile_rows`, `block_tile_cols` (RT) |
| `retire_block` | raw_api | `noc_async_read_barrier(noc)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1921` | issuer's NoC | — | — | none |
| both (profiling, BB-21) | raw_api | `DeviceZoneScopedN(name)` | `tt_metal/hw/inc/internal/profiler/kernel_profiler_streaming.hpp:513` (compiles to nothing outside the zone build, `:569`) | `"I2S-FETCH"`, `"I2S-RETIRE"` | — | — | none |

**Helpers considered and rejected** (applies to every raw_api row above):

| Candidate | File:Line | Mismatch |
|-----------|-----------|----------|
| eltwise `copy` (and every compute helper: tilize, untilize, reduce, matmul, eltwise chain) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:113` | The op has no compute phase. Routing tiles through unpack → DEST → pack is a value path that spec rule 4 forbids (bit-exact, NaN payloads, denormals; risks.md R6). `compute_kernel_hw_startup` is therefore not called: no compute kernel exists. |
| `set_read_state` / `read_with_state` (stateful single-packet reads) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:74-75`, `:94-95` | Self-aimed L1 → L1 only: the source is built by `local_addr()` (`:57`) on this core's own L1. Our source is a DRAM bank, and every tile of a block sits in a different bank, so a programmed source would be reprogrammed per tile. |
| `set_read_trid` / `async_read_barrier_with_trid` | `local_copy_helpers_dataflow.hpp:147`, `:158` | A trid ring retires the oldest block early so a consumer can drain it. This program has one block per issuer and no consumer; the single barrier at the end is the minimum. |
| `McastArgs` sender/receiver | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` | No operand is shared across cores (operand-reuse check); multicast has nothing to carry. |

No kernel_lib helper reads interleaved DRAM pages into an arbitrary L1 address. The raw read/barrier pair is the block operation's realization; nothing is missing that a helper would close.

## Host Contract

### Registry (op file `g6_interleaved_to_sharded.py`)

| Item | Value |
|------|-------|
| `INPUT_TAGGERS` | `tag_dtype` → `scenario["dtype"]`; `tag_layout` → `scenario["layout"]`; `tag_rank` → `int(len(scenario["shape"]))`; `tag_alignment` → with `H = shape[-2]`, `W = shape[-1]`: both multiples of 32 → `"tile_aligned"`, W not → `"w_non_aligned"`, else `"h_non_aligned"`; `tag_out_scheme` → `scenario["out_memory"]["scheme"]`. Each reads `inputs[0]`, the scenario dict. |
| `SUPPORTED` (Phase 0) | `dtype: [ttnn.bfloat16]`, `layout: [ttnn.TILE_LAYOUT]`, `rank: [2]`, `alignment: ["h_non_aligned"]`, `out_scheme: ["width_sharded"]` |
| `EXCLUSIONS` | `[]` |
| `validate(input_tensor, memory_config)` | Step 1: the malformed checks below, each raising `ValueError`. Step 2: build the scenario dict of the live call (`shape`, `dtype`, `layout`, `in_memory {buffer, scheme}`, `out_memory {buffer, scheme, grid, shard_shape, orientation}`) and run `INPUT_TAGGERS` on `(scenario,)`, so tagging has one source. Step 3: SUPPORTED per axis → `UnsupportedAxisValue`. Step 4: EXCLUSIONS → `ExcludedCell`. INVALID is never checked. |
| Entry point | `validate()` on its first line, then plan (cached), allocate output, patch addresses, one `ttnn.generic_op([input_tensor, output], descriptor)`, return output. No other ttnn op on input or output (BB-01, BB-02, spec rule 1). |
| Plan report (BB-08, BB-21) | `describe_plan(input_tensor, memory_config) -> dict`, exported from the package and built by the same planning function the entry point uses. Keys: `"regime"` (`"owner_fetch"`), `"num_shards"`, `"issuers_per_core"`, `"partial_tile_read"`, `"issuer_placement"`, `"issuer_split"`, `"issuer_weights"`, `"bank_stagger"`, `"full_span_bytes"`, `"partial_span_bytes"`, `"cores"`: a list in `G` order of `{"core": (x, y), "role": "owner" or "idle", "shard": s or None, "block_tile_rows", "block_tile_cols"}`. It runs `validate()` first. |

Malformed requests (ValueError, before the support check):

| Check | Spec item |
|-------|-----------|
| input not on device | input not on a device |
| `input.layout != TILE_LAYOUT`, or tile not 32x32 | not in TILE_LAYOUT (tile shape: mechanism cap) |
| input `memory_layout != INTERLEAVED` or `buffer_type != DRAM` | not interleaved / not in DRAM (L1 interleaved refused) |
| `dtype != bfloat16` | dtype other than bfloat16 |
| `rank not in (2, 4)` | rank other than 2 or 4 |
| `memory_config.buffer_type != L1` | not L1 |
| `memory_layout not in (HEIGHT_SHARDED, WIDTH_SHARDED)` or `shard_spec is None` | not height or width sharded |
| `shard_shape[0] % 32 or shard_shape[1] % 32` | shard shape not a multiple of 32 |
| width: `shard_tile_rows != tensor_tile_rows` | width shard shorter than the tensor (no owner for some rows); taller cannot be allocated (TensorSpec requires equality, `tt_metal/impl/tensor/spec/tensor_spec.cpp:62-68`) |
| height: `shard_tile_cols != tensor_tile_cols` | height shard narrower than the tensor; wider cannot be allocated (`tensor_spec.cpp:52-56`) |
| `num_shards > grid.num_cores()` | grid with fewer cores than shards |
| `shard_tile_rows * shard_tile_cols * tile_bytes > L1_BUDGET_BYTES (1347776)` | BB-05 host refusal (op buffers add 0) |

Support refusals in Phase 0 (NotImplementedError subclasses): `rank = 4`, `out_scheme = "height_sharded"`, `alignment` in {`"tile_aligned"`, `"w_non_aligned"`}.

### Descriptor cache (BB-07, BB-16; lesson RB-041)

| Item | Value |
|------|-------|
| Key | device; input `shape`, `padded_shape`, `dtype`, `layout`, memory layout and buffer type; output `memory_layout`, `buffer_type`, shard grid ranges, shard shape, orientation; every `KNOBS` value |
| Cached value | the KernelDescriptor object(s) with CT args and per-core RT args (rectangles), plus the plan dict |
| Per call (hit or miss) | Set each cached kernel's `common_runtime_args = [input.buffer_address(), output.buffer_address()]`, then build `ttnn.ProgramDescriptor(kernels=cached_kernels, semaphores=[], cbs=[])` from those objects (no per-core loop). Addresses are the only per-call values. On a hit, `generic_op` re-applies all runtime args of the passed descriptor (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:345-348`). |
| Trace | Kernels compile on the first (warm-up) call. Under capture, the output allocation is part of the captured call, as for every ttnn op. A replay builds and allocates nothing. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Shard core order (R3, spec rule 8) | p1's 32 shards are ROW_MAJOR in production, so a hard-wired order passes every production cell | `G` from `ttnn.corerange_to_cores(..., row_wise=orientation == ROW_MAJOR)`, the buffer's own call (`buffer.cpp:440`); orientation in the cache key; acceptance cases `flux58_width_8x4_col_major` and `test_plan_report_core_order` |
| Stale address on a cache hit (R5) | every call has a new output tensor and possibly a new input address | addresses only in common RT args, patched every call; tests `test_program_cache_and_addresses` |
| Patch lost through binding copies | nanobind may return copies of `ProgramDescriptor.kernels` | patch the cached KernelDescriptor objects themselves and construct the ProgramDescriptor from them on every call |
| Transfers in flight at return (R1, BB-03) | the op is nothing but reads, and another program starts right after | every issuer ends with `noc_async_read_barrier` on its own NoC; the op issues no writes, semaphores or atomics; test `test_repeat_then_other_program` |
| Value-changing path (R6) | graded bit-exact, NaN payloads included | no compute kernel; bytes move DRAM → L1 by NoC only; test `test_special_values_bit_exact` |
| Partial-row span geometry (R9) | the span assumes 16x16 faces; row 0 of a tile is bytes `[0,32)` of face 0 and `[512,544)` of face 1 | host derives face sizes from the tensor's tile; validate refuses other tiles; test `test_poisoned_padding` (NaN in rows 1-31 never reaches the logical row) |
| Row stride in the shard | for width shards the shard row stride (`shard_tile_cols`) differs from the input's (`tensor_tile_cols`) | shard slot uses `shard_tile_cols`; a taller/wider shard than the tensor is refused as malformed (TensorSpec cannot allocate it) |
| Surplus cores / partial last shard | shards beyond the tile grid; cores of G after the last shard | `ceil` for `num_shards`, `min` clamps, empty rectangle for surplus cores |
| Full-row path untested by production | no production tile row is fully valid | regime-pin case (33, 5120) in Phase 0; `height_multi_shard` after the refinement |
| Kernel text and core ranges (BB-17, BB-18) | the op runs in the FLUX trace before/after large programs | one small reader kernel; kernel group = the grid's own ranges (one rectangle in each production case); no TRISC binaries at all |
| Profiled time misleads lamp decisions (RB-053) | the lamps change which RISCs run and when cores end | judge L1-L3 on unprofiled trace-replay host time and the model step, not profiled kernel duration |
| Second issuer on NOC1 (lamp L1) | NOC0 and NOC1 have different coordinate spaces | the issuer's NoC id is a CT arg of its kernel and is passed to `get_noc_addr(page, 0, noc)`, `noc_async_read(..., noc)` and `noc_async_read_barrier(noc)`; never a hard-coded `noc_index` mix |

## Acceptance Test

`tests/ttnn/unit_tests/operations/g6_interleaved_to_sharded/test_g6_interleaved_to_sharded.py` (immutable). Cases are the production cases plus those the spec's rules or this design's regime-pin requirement demand (BB-22). The generic "single-tile / multi-tile / non-square / multi-batch" buckets map to these:

| Bucket | Case | Why it is allowed |
|--------|------|-------------------|
| multi-tile, many cores | `flux58_width_8x4_row_major` (1, 5120) | production p1 |
| small single image (4 tiles, one core) | `flux57_height_1core` (1,1,1,128) | production p2 (refinement: xfail-strict until SUPPORTED) |
| non-square second placement | `flux58_width_8cores_640` (32, 640) on (0,0)-(7,0) | BB-07 cache key (golden `test_cache_key_shard_spec`) |
| orientation | `flux58_width_8x4_col_major` | spec rule 8 |
| two span classes in one block | `regime_pin_33x5120_width` | regime-pin requirement (A + B) |
| multi-batch | `height_multi_shard_2x1x33x128` | spec rule 9 (refinement) |

Grading: exact (`helpers.TOLERANCES[bfloat16] == ("exact", None)`), i.e. bit equality of the bf16 patterns, plus equality of the result's memory configuration with the requested one.

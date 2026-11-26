# Operation Design: g6_reshape

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no arithmetic, no compute kernel) |
| Goal | Return `input_tensor` with a new shape: every logical value moved bit for bit to the position of the same row-major linear index, laid out again in the input's own layout (TILE in Phase 0), in ONE device program with two dataflow kernels. |
| Math | `out[j] = x[i]` whenever `lin(j; e) = lin(i; d)`; on the page view (leading dims folded into rows): `Y[g, c] = X[(g·W_out + c) div W_in, (g·W_out + c) mod W_in]` (F1/F2 of `docs/explorations/g6_reshape/math.md`) |
| References | `docs/explorations/g6_reshape/{DOSSIER,math,tensix,risks,lessons}.md`; `eval/golden_tests/g6_reshape/{feature_spec,helpers}.py`; `eval/op_template.py`; `.claude/references/{blocking-model,l1-footprint-discipline,op-design-template}.md`; `ttnn/ttnn/operations/examples/master.md` (entries `noc_placement`, `tile_reorder`, `double_buffer`, `trid_double_issue`, `bank_stagger`, `read_coalescing`, `width_split`) |

The op is a **gather**: each output face-row segment (16 consecutive logical elements of one output tile row) is a bijective image of 16 consecutive logical input elements. Phase 0 moves exactly the valid segments — input tile padding is never read as a value, output tile padding is never written.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 1..4, on device, DRAM interleaved; Phase 0: bfloat16 + TILE_LAYOUT | — | buffer address = RT |
| `shape` | `list[int]` or `tuple[int]` (positional) | yes | length 1..4; entries ≥ 1, at most one `-1`; product (after resolving `-1`) = `input_tensor` element count N | — | resolved shape → CT (via geometry symbols below) |

Host-side resolution: if `shape` holds one `-1`, it becomes `N // prod(other entries)` (ValueError if not exact). Entry `0`, entry `< -1`, two `-1`, a non-int entry, rank outside 1..4 (input or output), or a product ≠ N → `ValueError`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `d = (d_0..d_{n-1})`, n ∈ 1..4, every d_i ≥ 1 |
| Dtype | Phase 0: `ttnn.bfloat16`; TARGET adds `int32`, `uint32` (ROW_MAJOR) and `float32` (TILE) |
| Layout | Phase 0: `TILE_LAYOUT` (32×32, four 16×16 faces) |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `e` = `shape` with `-1` resolved |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` (DRAM interleaved); allocated with `ttnn.allocate_tensor_on_device(ttnn.Shape(e), dtype, layout, device, input_tensor.memory_config())` — the ONLY device allocation of the op |

### Geometry symbols (host-derived, single source: the program-descriptor builder)

| Symbol | Definition (input side `_in`, output side `_out`, same rule) |
|--------|---------------------------------------------------------------|
| `W` | last logical dim (`shape[-1]`) |
| `H` | second-to-last logical dim; `1` for a rank-1 tensor |
| `IMG` | product of the dims before the last two (1 for rank ≤ 2) |
| `Wt` | `ceil(W / 32)` |
| `Ht` | `ceil(H / 32)` — per image; tile padding is per image |
| `num_tiles` | `IMG · Ht · Wt` (implementer asserts it equals `tensor.buffer_num_pages()`) |
| tile id | `t = (img · Ht + tr) · Wt + tc` |
| `v(tr)` | valid rows of tile row `tr`: `min(32, H − 32·tr)` |
| `vc(tc)` | valid columns of tile column `tc`: `min(32, W − 32·tc)` |
| `esize` | `input_tensor.element_size()` (2 for bf16) |
| `face_row_bytes` | `16 · esize` (32 B bf16) |
| `face_bytes` | `256 · esize`; `tile_bytes = 1024 · esize` |
| byte offset of logical (row-in-tile `r`, col-in-tile `c`) | `((r / 16)·2 + c / 16) · face_bytes + (r % 16) · face_row_bytes + (c % 16) · esize` |

## Blocking Model

### Work unit: the output strip

An **output strip** is one valid row `r < v(tr)` of one output tile `t`: up to 32 logical elements = up to two face-row **segments** (`fh = 0`: columns 0..15 → face `2·(r/16)`; `fh = 1`: columns 16..31 → face `2·(r/16)+1`, present only when `vc(tc) > 16`). Strips are numbered in **tile-major order**: tile id `t` ascending, then `r` ascending. Strip count:

`U = IMG_out · H_out · Wt_out`   (each image contributes `H_out` valid rows to each of its `Wt_out` tile columns)

The input side is not partitioned: each segment's source is computed from the index map and read on demand by the core that owns the segment (gather).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `t_out` — output tile sequence (`img_out`, `tr_out`, `tc_out` folded; they are contiguous and treated identically) | independent — no output value depends on another output tile | `block_tiles` (output tiles one block touches) | `min(max_core_tiles, BLOCK_SLOT_TILES_CAP)` — the whole per-core assignment in one block on every golden shape | host constant `BLOCK_SLOT_BYTES_CAP = 256 KiB` → `BLOCK_SLOT_TILES_CAP = BLOCK_SLOT_BYTES_CAP // tile_bytes`; `block_tiles` computed once in the descriptor builder, passed as CT arg | contiguous strip ranges across `n_cores` (below) — a core owns a run of tiles, the first/last possibly partial in rows | knob-turn |
| `r` — valid row within an output tile | independent — each row reads only its own linear range | `block_rows` = rows of a tile the block covers | all valid rows `v(tr)` of every tile the block touches (fewer only at the core's first/last tile) | derived from the core's strip range (RT args `start_tile`, `start_row`, `num_strips`) | split across cores only where a core boundary falls inside a tile (strip granularity) | knob-turn |
| `fh` — face-column half of a strip | independent — two disjoint 16-column segments | `segments_per_strip` | `1 + (vc(tc) > 16)` — whole (never split) | derived from `W_out` (CT) | not split | none needed — it is the transfer atom |
| `e` — element within a segment | independent | `SEGMENT_ELEMS = 16` | 16 (whole face row) | fixed by the tile format (face row) | not split | none — the face row is the DRAM-contiguous atom of the tile format |
| `g_in`, `c_in` — input rows / columns (source side) | independent — every input element feeds exactly one output element (bijection); **not reuse-shared**: no operand is read by more than one output position | none — the source footprint of a block is exactly the segments its strips name | n/a | n/a | follows output ownership (owner-computes on output) | scatter regime (input-driven) is a `deferred` regime row |

`n_cores = min(grid_cores, ceil(U / MIN_STRIPS_PER_CORE))`, `MIN_STRIPS_PER_CORE = 16` (host constant, single source). `grid_cores = grid.x · grid.y` of `device.compute_with_storage_grid_size()` (runtime query, never a literal). Strips are dealt in contiguous runs: the first `U mod n_cores` cores get `ceil(U / n_cores)` strips, the rest `floor(U / n_cores)`. `max_core_tiles` = the largest number of distinct output tiles any core's run touches (host computes it from the runs).

Production consequence (grid ≥ 100 cores): `[1,16,6,256]→[1,1,16,1536]` U = 768 → 48 cores; `[16,1,5120]→[1,1,16,5120]` U = 2560 → full grid; `[1,1,16,4608]→[16,1,4608]` U = 2304 → full grid; `[16,1,64]→[1,1,16,64]` U = 32 → 2 cores; `[2,1,5120]→[1,1,2,5120]` U = 320 → 20 cores; `[1,1,6,256]→[1,1,1536]` U = 48 → 3 cores. The core count is `MIN_STRIPS_PER_CORE`'s choice, not the tile count's: `[16,1,64]→[1,1,16,64]` has 2 output tiles and would get 4 cores at `MIN_STRIPS_PER_CORE = 8`.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_output_tiles` | `OUTPUT_DEPTH` (host constant) | 2 | the reader gathers block `b+1` while the writer's DRAM writes of block `b` drain (only active when a core has > 1 block) |
| `cb_granule_staging` | `STAGING_GRANULES` (host constant) | 512 granules (32 KiB at 64 B) | number of staged granule reads in flight before a forced flush (barrier + assemble). Every golden shape stages ≤ 2·ceil(U/n_cores) ≤ ~64 granules per block → one flush per block |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| NoC DRAM read congruence: `(l1_dst ^ dram_src) & (NOC_DRAM_READ_ALIGNMENT_BYTES − 1) == 0` (64 B on Blackhole — this machine is a p150b — 32 B on Wormhole; `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`, checked at `tt_metal/hw/inc/internal/debug/sanitize.h:745`) | a **direct** segment read (DRAM face row → output-slot face row) | per transfer, at run time: direct only if the piece is a whole source face row at element offset 0, lands at element 0 of the destination segment, and the congruence holds; otherwise **staged** (read the aligned granule into `cb_granule_staging`, then copy) | NoC alignment error under the watcher, or silently shifted rows (risk R4) |
| NoC DRAM write congruence: `(l1_src ^ dram_dst) & 15 == 0` (`noc_parameters.h:380`) | output span writes | spans start at face-row boundaries (multiples of 32 B) in both the slot and DRAM; the output slot base is aligned to `NOC_DRAM_READ_ALIGNMENT_BYTES` (descriptor: page size a multiple of 64 B; kernel asserts `slot_base % 64 == 0` under `--dev`) | misaligned write → hang / NoC error |
| Granule = `max(NOC_DRAM_READ_ALIGNMENT_BYTES, face_row_bytes)` bytes, aligned down from the source address | staging slot size | staging base aligned up to 64 B inside a CB allocated `STAGING_GRANULES · granule_bytes + granule_bytes` | a granule straddling the staging end, or a misaligned staged read |
| L1 self-copy congruence `(src ^ dst) & 15 == 0` (`NOC_L1_READ_ALIGNMENT_BYTES`, `noc_parameters.h:375`) | staged → slot copy by NoC (`dataflow_kernel_lib::local_addr`, `local_copy_helpers_dataflow.hpp:57`) | a NoC self-copy is used only for whole-face-row pieces (offsets multiples of 32 B on both sides); partial pieces are copied by the reader RISC-V with 2-byte (or 4-byte when both offsets are 4-B aligned) loads/stores | misaligned L1 copy |
| `cb_output_tiles` ring wrap | push/pop quantum | every push and pop is the nominal `block_tiles` (a ragged last block narrows the transfers, never the count); capacity `OUTPUT_DEPTH · block_tiles` closes the ring | CB pointer drift → hang |
| Kernel L1 budget 1,347,776 B (BB-05) | `block_tiles`, `STAGING_GRANULES` | `BLOCK_SLOT_BYTES_CAP = 256 KiB` → total ≤ 2·256 KiB + 32 KiB + 64 B = 557,120 B on every shape, any N | an unbounded block would OOM on large N |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (boundary: DRAM; minimum = every logical input byte read once, every logical output byte written once) | What a bigger block buys |
|--------|--------|-----------|-------|-----------------------------------------------------------------|--------------------------|
| **R1 `segment_gather_aligned`** | **built** (Phase 0) | TILE ∧ `W_in % 16 == 0` ∧ `W_out % 16 == 0` (every production bf16 case: W ∈ {64, 256, 1536, 4608, 5120}) | `block_tiles` output tiles × their valid rows × whole segments | Input: each valid segment read once (32 B). Segments failing the 64-B congruence are read as 64-B granules → ≤ 2× the logical bytes on those segments (about half of them on the production shapes); padding rows never read. Output: valid face rows only, one write per (tile, face) row run → exactly the logical bytes rounded to 16 columns. vs. whole tiles: `[16,1,5120]` reads 160 KiB–240 KiB instead of 5.24 MB; `[16,1,4608]` writes 147 KiB instead of 4.7 MB | one CB handshake, one read barrier, one assemble pass, one write flush per block instead of per tile; Phase 0 pays each once per core on every golden shape |
| **R2 `segment_gather_general`** | **built** (Phase 0; same pipeline, the general piece iterator compiled in by define `G6_GENERAL_SEGMENTS`) | TILE ∧ ¬R1 predicate (a `W` not a multiple of 16: then one destination segment's 16 elements span up to `min(16, ceil(16/W_in)+1)` source pieces) | same as R1 | Input: each distinct source granule read once per block (last-granule memo, below) → ≤ logical bytes rounded up to 64-B granules; Output: as R1 | same as R1, plus granule reuse inside a block: `[1,4096]→[4096,1]`-like geometries (`W_out = 1`) read 16 consecutive strips from one granule |
| R3 `tile_page_copy` | deferred — no golden or production shape satisfies it (every production TILE tensor has a partial tile row); R1 is correct on it at minimum bytes | TILE ∧ `W_in == W_out` ∧ `H_in % 32 == 0` ∧ `H_out % 32 == 0` (then output tile `t` ≡ input tile `t`) | `block_tiles` whole tiles | minimum, in whole-tile transfers (1 read + 1 write of 2 KiB per tile instead of 64 + 4) | reachable: same slot + span writer, the gather emits one whole-tile direct read per tile and the writer one whole-tile span |
| R4 `segment_scatter` (input-driven) | deferred — R1/R2 already move minimum bytes on every shape it would serve; it changes only the read **transaction count** on shapes whose input tiles hold more valid rows than output tiles (`[1,1,H,4608]→[H,1,4608]`, `[1,H,64]→[H,1,64]`: 2 face reads of 16 rows per input tile instead of 32 segment reads), bounded by ≤ 2·ceil(U/n_cores) reads per core (≤ ~46 at full grid); it is the perf pass's first experiment on those two shapes | TILE ∧ R1 predicate ∧ `H_in_valid_rows_per_tile > H_out_valid_rows_per_tile` | input tiles × valid rows | same bytes as R1; reads coalesced per input face, writes 32-B per segment | reachable: the index map is a bijection; the scatter walk is the same piece iterator run from the input side, and the writer's span list already handles per-segment spans |
| R5 `rm_page_gather` | deferred — dtype refinement (int32 / uint32 enter SUPPORTED together with ROW_MAJOR, rule 7); Phase 0 SUPPORTED is bf16 TILE only | ROW_MAJOR | `block_pages` output pages (unit = output page = one output row) | input: one granule-aligned read per input page run; output: one write per output page (4 B into a 64-B slot for `[1,W]→[W,1]`: W writes) | same pipeline; the address function changes (page `g`, byte `c·esize`; no faces); the output slot holds each page at its 64-B aligned DRAM slot image so the span writer is unchanged |
| R6 `rm_bank_run_writes` | deferred — successor of R5 for `aligned_page_size > W_out·esize` (the flagged p1–p4 `[1,W]→[W,1]` int32); not reachable before R5 | ROW_MAJOR ∧ `W_out · esize < NOC_DRAM_READ_ALIGNMENT_BYTES` | per core: its pages grouped by DRAM bank (page p → bank `p mod num_banks`, slot `p div num_banks`) | output: one write per (bank, run of consecutive slots), alignment gaps written as part of the output buffer's own slots → ~`num_banks · ceil(run_bytes / 16 KiB)` writes per core-range instead of W (16 instead of 4096 for p1 on one core-range) | the slot image already holds pages at their 64-B slot stride; R6 only reorders the slot by bank and merges spans — layered on R5, nothing rewritten |
| R7 `fp32_segment_gather` | deferred — dtype refinement (float32 enters with TILE, rule 7) | TILE ∧ float32 | as R1/R2 with `esize = 4` (face row 64 B = granule; always congruent for whole face rows) | as R1/R2; class D `[1,W]→[1,1,W,1]` (W_out = 1) reads ceil(W/16) granules via the last-granule memo, writes 2 face spans per output tile | knob-turn: `esize` is a CT arg, every offset derives from it |
| R8 `whole_tiles_in_out` | rejected — superseded by R1/R2 everywhere: reads every padding tile (2560 tiles = 5.24 MB for `[16,1,5120]`, 160 KiB valid) and writes every padding row (4.7 MB for `[16,1,4608]`) | — | — | 16–32× above minimum on the dominant production shapes | — (dead end: a better regime is built instead of it, not on it) |
| R9 `untilize_regroup_tilize` (compute engine) | rejected — two engine format conversions on a bit-exact contract (risk R2), reads padding tiles anyway, adds a compute kernel and 2 CBs; superseded by R1/R2 | — | — | ≥ whole-tile traffic plus two unpack/pack passes | — |
| R10 `staged_through_dram_or_host` | rejected — forbidden by the contract (doubles DRAM traffic; host read of values) | — | — | 2× minimum | — |
| R11 `cross_core_segment_routing` | rejected — moves each segment DRAM→core A→core B→DRAM; R1 reads each segment directly on its owner, so the extra NoC hop saves no DRAM byte | — | — | minimum DRAM + one extra NoC crossing per segment | — |

**Regime selection function** (host, exact): `regime = "R1" if (layout == TILE and W_in % 16 == 0 and W_out % 16 == 0) else "R2"` (Phase 0 SUPPORTED is TILE-only). R2 is compiled by adding the define `G6_GENERAL_SEGMENTS`; R1 compiles the single-piece iterator only (BB-18: dead paths compiled out per shape). The selection depends only on shapes, never on the grid. **Regime-pinned tests**: R1 = the 14 production bf16 cases; R2 = the bf16 TILE cases with the production geometries `[1,185]→[1,1,185,1]`, `[1,207]→[1,1,207,1]`, `[1,77]→[77,1]`, `[1,4096]→[4096,1]`, `[8]→[8,1]`, `[8,1]→[8]` (all in the acceptance test).

### Traffic ranking

Candidate splits, ranked by DRAM crossings per tensor, then cross-core traffic (qualitative, no ns):

| Rank | Split | x DRAM crossings | out DRAM crossings | Cross-core | Verdict |
|------|-------|------------------|--------------------|------------|---------|
| 1 | **contiguous output-strip runs, gather on owner (R1/R2)** | 1 × valid bytes (granule rounding ≤ 2× on staged segments) | 1 × valid bytes | none | **chosen** — minimum on both tensors, no combine, fills the grid at strip granularity |
| 1 (tie) | contiguous input-strip runs, scatter on owner (R4) | 1 × valid bytes | 1 × valid bytes | none | equal bytes; differs only in transaction counts → `deferred` R4 |
| 3 | whole output tiles per core | 1 × valid bytes | 1 × valid bytes | none | equal bytes but the core count is capped by the output tile count (2 cores for `[16,1,64]→[1,1,16,64]`) — the shape would decide the core count; strip runs subsume it |
| 4 | whole tiles in, whole tiles out | 1 × **all** input tiles (padding incl.) | 1 × **all** output tiles | none | 16–32× bytes → rejected R8 |
| 5 | split + cross-core routing | 1 × valid | 1 × valid | +1 NoC crossing per segment | rejected R11 |
| — | dependent-axis split + combine | — | — | — | not applicable: the op has no dependent axis (no value spans an axis; math.md "Axes and dependence") |

Operand-reuse check: the op has one input; on the chosen split no input byte is needed by two cores except a source granule shared by the two segments at a core boundary in R2 (≤ 1 granule per core boundary). No operand is reuse-shared → no broadcast/multicast regime.

Stall-shadow check: the reader blocks on the per-block read barrier. Independent work in that window = issuing the next block's reads (they target the other output slot); that is the `trid` lamp below and only exists when a core has > 1 block — Phase 0 golden shapes have one block per core, so nothing is independent of the wait there. The writer blocks on `cb_wait_front`; it has no other work. No reorder changes values (pure moves), so no precision gate applies.

### Block schedule

```cpp
// reader (NCRISC, NoC0)                         // writer (BRISC, NoC1)
for (block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    gather_block(block_idx);     // reader
    assemble_block(block_idx);   // reader
    store_block(block_idx);      // writer
}
```

| Block operation | Kernel | Block shape | What it does | Resident across it | Intended frequency of fixed costs |
|-----------------|--------|-------------|--------------|--------------------|-----------------------------------|
| `gather_block` | reader | the block's strips: ≤ `block_tiles` tiles × their covered rows × `segments_per_strip` segments | `cb_reserve_back(cb_output_tiles, block_tiles)` once; walk the block's segments in strip order with the **piece iterator** (below); per piece: **direct** NoC read DRAM→slot face row when the congruence holds, else a **staged** granule read into `cb_granule_staging` (skipped when the granule equals the previously read granule — last-granule memo); when the staging ring fills: `noc_async_read_barrier` + `assemble_block` on the staged pieces so far, then continue | the reserved output slot (block_tiles tiles) | 1 reserve per block; 1 read barrier per block (+1 per staging flush, none on golden shapes); TensorAccessor built once per kernel |
| `assemble_block` | reader | the block's staged pieces | after the barrier, copy each staged piece from its granule into its destination bytes in the slot: whole-face-row pieces by self-aimed NoC read (`local_addr`) or 8 word copies; partial pieces (R2) by RISC-V 2-/4-byte copies; then (if NoC self-copies were issued) one `noc_async_read_barrier`; `cb_push_back(cb_output_tiles, block_tiles)` | the slot and the staging ring | 1 extra barrier + 1 push per block. The piece list is NOT stored: the iterator is re-run over the block and only staged pieces are acted on (deterministic) — or the implementer keeps a compact per-staged-piece descriptor array in the staging CB's tail; either is legal |
| `store_block` | writer | the block's tiles × faces | `cb_wait_front(cb_output_tiles, block_tiles)`; per touched tile, per face `f ∈ {0,1,2,3}` with valid columns in that half (`f odd` only if `vc(tc) > 16`), write the contiguous run of the block's rows in that face (`rows ∩ [16·(f/2), 16·(f/2)+16)`) as ONE `noc_async_write` of `run_rows · face_row_bytes` bytes from `slot + (tile − first_tile)·tile_bytes + f·face_bytes + (row0 % 16)·face_row_bytes` to the same offset of output page `t`; then `noc_async_writes_flushed()` and `cb_pop_front(cb_output_tiles, block_tiles)` | the slot being drained | one write per valid face of each touched tile (≤ 4; 2 for a 1-row tile, 1 when `vc(tc) ≤ 16`); 1 flush + 1 pop per block; `noc_async_write_barrier()` once at kernel end (BB-03) |

**Piece iterator** (the index map; one function shared by both phases of the reader):
for destination segment (output tile `t = (img·Ht_out + tr)·Wt_out + tc`, row `r`, half `fh`): global output row `g = img·H_out + 32·tr + r`, columns `c0 = 32·tc + 16·fh`, count `n = min(16, W_out − c0)`; linear start `L = g·W_out + c0`. Walk `p = L … L+n−1`: `gi = p / W_in`, `ci = p % W_in`, `img_in = gi / H_in`, `ri = gi % H_in`; source tile `(img_in·Ht_in + ri/32)·Wt_in + ci/32`, byte offset inside it per the geometry table; a piece ends at the end of the source face row (`ci % 16 == 15`), at the end of the source row (`ci == W_in − 1`) or at `n`. R1 guarantees exactly one piece per segment starting at `ci % 16 == 0`; R2 handles up to `min(16, ceil(16/W_in)+1)` pieces. Divisors are CT args (cheap constant division) or the implementer walks incrementally.

A partial destination segment (`n < 16`) may be filled by a direct whole-face-row read when its single piece starts at source element 0: the extra `16 − n` elements land in output **padding** columns of the same face row, which the contract leaves unspecified. Never extend a read into another destination segment's bytes.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (`MIN_STRIPS_PER_CORE = 16`) | production shapes are 32–2560 strips; at full grid a core moves ~20 strips (≈40 reads + ≤40 writes) and the per-core launch/RT-arg/barrier cost may dominate; fewer cores also means fewer start-gap core ranges | `MIN_STRIPS_PER_CORE ∈ {8, 32, 64}`; judge on unprofiled trace-replay host time (lesson RB-053), not profiled kernel time |
| Overlap (`block_tiles` = whole per-core run) | with one block per core the writer cannot start before every read of the core has landed | `block_tiles = ceil(core_tiles / 2)` or `/4` (2–4 blocks, `OUTPUT_DEPTH = 2`) on `[16,1,5120]→[1,1,16,5120]` and `[1,1,16,4608]→[16,1,4608]` |
| Read barrier → trid ring (catalog `trid_double_issue`) | once a core has > 1 block, the drain-all barrier idles the NoC | `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` (`local_copy_helpers_dataflow.hpp:147,158`), one trid per output slot |
| Face-row writes vs whole-tile writes (catalog `tile_reorder`) | a tile with ≥ ~24 valid rows is written as up to 4 face runs; one 2-KiB write may be faster than 4 face writes even though padding rows ride along | write the whole tile when `v ≥ WHOLE_TILE_WRITE_ROWS` (try 16, 24, 32); bytes rise by ≤ 2× on those tiles only |
| Direct-read congruence on Blackhole | about half of the R1 segments on the production shapes fail the 64-B parity and are staged (64-B read + L1 copy) | (a) stage all segments of a block uniformly as granule reads (simpler code, more bytes); (b) choose the strip walk order so consecutive segments share a granule |
| Read coalescing (catalog `read_coalescing`) | consecutive face rows of one source face that land in consecutive face rows of one destination face (identity-like segments) are issued as separate 32-B reads | merge runs of contiguous source+destination face rows into one read |
| NoC placement / bank stagger (catalog `noc_placement`, `bank_stagger`) | every core starts its walk on bank 0-aligned tiles; the default row-wise core placement may still contend | rotate each core's first tile by `core_idx mod num_banks`; verify `row_wise=True` placement |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → output slot (direct) | bf16 face rows (32 B) | `noc_async_read` via `TensorAccessor(input).get_noc_addr(tile_id, byte_offset)` (`tensor_accessor.h:165`) into `cb_output_tiles` write slot | only when `(dst ^ src) & (NOC_DRAM_READ_ALIGNMENT_BYTES−1) == 0` and the piece is a whole source face row at element 0 landing at segment element 0 |
| DRAM → staging (staged) | granule (`max(64, face_row_bytes)` B on BH) | `noc_async_read` of the granule containing the piece, aligned down, into the next 64-B-aligned staging slot | last-granule memo: a piece whose granule equals the previous staged granule reuses it |
| staging → output slot | bytes | self-aimed NoC read (`dataflow_kernel_lib::local_addr`, `local_copy_helpers_dataflow.hpp:57`) for whole face rows; RISC-V loads/stores for partial pieces | inside the reader, after the block's read barrier |
| output slot → DRAM | contiguous face-row runs | `noc_async_write` via `TensorAccessor(output).get_noc_addr(tile_id, byte_offset)` | one write per (tile, face, row run); `noc_async_writes_flushed` before pop; `noc_async_write_barrier` at end |
| (no compute kernel) | — | — | no value enters an unpacker, DEST or a packer → bit-exact by construction (rule 1, risk R2) |

Placement: reader on NCRISC/NoC0 (`ttnn.ReaderConfigDescriptor()`), writer on BRISC/NoC1 (`ttnn.WriterConfigDescriptor()`) — catalog `noc_placement`. No inter-Tensix communication in any built or deferred regime (no semaphores, no multicast).

TARGET axes as regimes: `dtype ∈ {int32, uint32}` with `layout = ROW_MAJOR` → R5 (+R6) on the same pipeline (a new address function: regime addition); `dtype = float32` with `TILE` → R7 (knob-turn: `esize`). There is no placement axis in TARGET (every case is DRAM interleaved), so no sharded realization is designed.

Structural impossibilities: none (agrees with `feature_spec.INVALID = []`).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | output strip (one valid row of one output tile); a block = the core's run of strips, ≤ `block_tiles` tiles |
| Grid | `device.compute_with_storage_grid_size()`; used cores = `ttnn.num_cores_to_corerangeset(n_cores, grid, row_wise=True)` → ≤ 2 rectangular ranges (BB-17); one reader + one writer kernel over that CoreRangeSet |
| Per-core work | RT args per core: input/output buffer addresses, `start_tile`, `start_row`, `num_strips` (host walks the strip order once to compute them: per tile, `v(tr)` strips) |
| Blocks per core | `num_blocks_this_core = ceil(core_tiles / block_tiles)`; = 1 on every golden shape |
| Remainder | strip-level balance (counts differ by ≤ 1 across cores); tile geometry uses `ceil` and per-image `Ht` (`Ht = ceil(H/32)` per image, `num_tiles = IMG·Ht·Wt`), never `floor(IMG·H/32)` |
| Plan reporting (BB-08, BB-21) | the op module exposes `g6_reshape_plan(input_tensor, shape) -> dict` with `regime`, `n_cores`, `core_ranges`, `block_tiles`, `strips_per_core` (min/max), `blocks_per_core`, and per-core role (`reader+writer`, all cores identical). Kernels carry `DeviceZoneScopedN` zones `G6_GATHER`, `G6_READ_WAIT`, `G6_ASSEMBLE`, `G6_RESERVE_WAIT` (reader) and `G6_WAIT_FRONT`, `G6_STORE`, `G6_FLUSH_WAIT` (writer); zones compile to nothing in the plain build |
| Program cache (BB-07, BB-16) | geometry (`W/H/IMG/Wt/Ht` both sides, `esize`, `block_tiles`, `STAGING_GRANULES`, CB ids) are **compile-time args** → covered by the generic-op program hash; buffer addresses and the per-core strip ranges are runtime args. The Python side memoizes the built `ProgramDescriptor` in a module dict keyed by `(tuple(input.shape), tuple(input.padded_shape), tuple(e), dtype, layout, memory_config, device id, grid)` and on a hit only patches the two buffer-address RT args per core. No host read of tensor values, no allocation besides the output |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_output_tiles` | 0 | `tile_bytes` (2048 B bf16) | `OUTPUT_DEPTH · block_tiles` | spans `t_out` (block_tiles), `r` (all 32 rows of a tile image), `fh`, `e`; streams over blocks; source axes not resident | input dtype (raw bytes; never unpacked) | reader | writer | whole kernel, ring of `OUTPUT_DEPTH` block slots |
| `cb_granule_staging` | 1 | `granule_bytes` (64 B on BH) | `STAGING_GRANULES + 1` (one extra page to align the base up to 64 B) | live set = staged granules between two flushes; streams over strips and blocks | raw bytes | reader (private scratch: `get_write_ptr` once, never pushed) | reader | whole kernel |

Full ledger, axis accounting and the data-movement budget: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `gather_block` | ≤ `block_tiles` tiles × covered rows × segments | no compute helper applies (dataflow only); TensorAccessor + `noc_async_read` | input tensor in DRAM; `cb_granule_staging` (≤ `STAGING_GRANULES` staged granules) | `cb_output_tiles`, `block_tiles` pages reserved | slot reserved, direct segments landed after the barrier |
| 2 | `assemble_block` | the block's staged pieces | `dataflow_kernel_lib::local_addr` for whole-face-row self-copies | `cb_granule_staging` (staged granules) | `cb_output_tiles`, `block_tiles` pages | `block_tiles` pages pushed; staging free |
| 3 | `store_block` | touched tiles × valid faces × row runs | TensorAccessor + `noc_async_write` | `cb_output_tiles`, wait `block_tiles` | output tensor in DRAM | writes flushed, `block_tiles` popped |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `gather_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | page = source tile id, offset = face-row / granule offset | — | — | — |
| `gather_block` | raw_api | `noc_async_read` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `(src_noc_addr, dst_l1, face_row_bytes or granule_bytes)` | — | `cb_output_tiles` / `cb_granule_staging` | — |
| `gather_block` / `assemble_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per block (+ per staging flush) | — | — | frequency set by `block_tiles`, `STAGING_GRANULES` |
| `assemble_block` | helper | `dataflow_kernel_lib::local_addr` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57` | `(staging_addr)` as src of a self-aimed read of `face_row_bytes` | `cb_granule_staging` | `cb_output_tiles` | — |
| `assemble_block` (R2 partial pieces) | raw_api | RISC-V L1 loads/stores (`volatile tt_l1_ptr uint16_t*` / `uint32_t*`) | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30` (`addr_to_l1_ptr`) | element copies | `cb_granule_staging` | `cb_output_tiles` | — |
| `store_block` | raw_api | `noc_async_write`, `noc_async_writes_flushed`, `noc_async_write_barrier` | `dataflow_api.h:876`, `:1980`, `:1951` | one write per (tile, face, row run) | `cb_output_tiles` | — | `block_tiles` sets flush frequency |
| all | raw_api | `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` | `dataflow_api.h` (`cb_push_back` :208, `cb_pop_front` :260, `cb_wait_front` :479, `cb_reserve_back` :406) | nominal `block_tiles` | `cb_output_tiles` | | `block_tiles` |
| (perf lamp) | helper | `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` | `local_copy_helpers_dataflow.hpp:147`, `:158` | one trid per output slot | | | `OUTPUT_DEPTH` |

**Helpers considered and rejected** (for every `raw_api` row above):

| Candidate | File:Line | Why it cannot express this op's block operation |
|-----------|-----------|-----------------------------------------------|
| `compute_kernel_lib::untilize` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:163` | converts whole tiles through unpack → DEST → pack: (1) it consumes whole input tiles, so padding tiles are read (`[16,1,5120]`: 2560 tiles for 160 KiB of values, regime R8/R9); (2) it routes values through the compute engine on a bit-exact contract (risk R2: denormal/NaN handling of the unpacker is unverified); (3) the regroup between untilize and tilize has no helper either, so it still needs the same index-map copies this design does directly |
| `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:206` | the reverse of the row above, same three reasons |
| `read_sticks_for_tilize` / `write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:88`, `:130` | move whole row-major sticks of a ROW_MAJOR tensor; the Phase 0 tensors are TILE in DRAM and the move unit is a face-row segment inside a tile page — no stick exists on either side |
| `dataflow_kernel_lib::set_read_state` / `read_with_state` | `local_copy_helpers_dataflow.hpp:75`, `:95` | single fixed-size, single-source self-copy loops; the staged copies have per-piece sources and (R2) per-piece sizes. Usable for the R1 32-B staged copies as an optimization (the implementer may) |
| mcast (`ttnn.Mcast`, `mcast/kernel/mcast_args.hpp`) | — | no reuse-shared operand and no cross-core step in any built regime |
| eltwise / reduce / matmul helpers | — | the op has no arithmetic |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Input padding reaching a logical output (R3) | the golden suite poisons the input padding with NaN | the piece iterator only addresses logical positions (`p < N`); direct whole-face-row reads may carry source-row padding only into **output padding** columns (partial destination segment, `n < 16`); staged copies copy exactly the piece's elements |
| DRAM read congruence on Blackhole (R4) | face rows are 32 B, the read congruence is 64 B; odd/even row parity decides | per-transfer runtime check `((dst ^ src) & (NOC_DRAM_READ_ALIGNMENT_BYTES−1)) == 0`; else staged granule. Run new kernels with `--dev` (NoC sanitizer) first |
| Rank-1 TILE tensors | `H` of a rank-1 tensor is 1; ttnn's padded shape for rank 1 may differ from `[32, Wt·32]` | derive `H, W, IMG` from the logical shape exactly as the geometry table says and assert `IMG·Ht·Wt == buffer_num_pages()` on both tensors |
| Several cores writing one output tile | a core boundary inside a tile (strip granularity) | each core writes only its own rows' face-row runs; byte ranges are disjoint, no read-modify-write |
| Program cache key (R9, BB-07) | `[16,1,64]→[1,1,16,64]` then `→[1,1,1,1024]` share the input | output geometry is CT → different program hash; the Python memo key includes the resolved output shape |
| Writes in flight at kernel exit (BB-03) | another program starts right after | writer ends with `noc_async_write_barrier()`; reader ends with `noc_async_read_barrier()` (no reads outstanding after the last push anyway) |
| Staging overflow | R2 with tiny `W_in` stages up to 16 pieces per segment (fewer with the last-granule memo) | ring of `STAGING_GRANULES`; full ring → barrier + assemble + reset, inside the same block (no CB handshake change) |
| Uneven / tiny work (R8) | 32-strip shapes, 1-strip shapes (`[8]→[8,1]` has U = 8 → 1 core), prime widths | `n_cores ≤ U` always → no core without work; ragged blocks keep nominal CB counts |
| Kernel text size (BB-04, BB-18) | two regimes in one source | R1/R2 selected by a define; no compute kernel; no `-Os` needed |

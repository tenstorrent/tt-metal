# Operation Design: g6_expand

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Materialize `x.expand(*sizes).contiguous()` for `x` of shape `(d_0, d_1, 1, W)` as a new DRAM-interleaved TILE tensor `(d_0, d_1, H_out, W)`, bit-identical, in one device program. Production: `(1, 1, 1, W) -> (1, 1, W, W)`, W in {512, 256, 196, 185, 207}. |
| Math | `out[n, c, h, w] = x[n, c, 0, w]` for all `h < H_out`; every other dim keeps its index. In tile terms: output tile `(image, r, col)` = input tile `(image, col)` with its row 0 copied into all 32 rows. |
| References | Contract: `eval/prompts/g6_expand.txt`. Golden suite: `eval/golden_tests/g6_expand/{feature_spec.py, helpers.py}`. Semantics of blocks: `.claude/references/blocking-model.md`. Ledger: `ttnn/ttnn/operations/g6_expand/l1_ledger.md`. Catalog entries used: `examples/{mcast_topology, tile_reorder, noc_placement, double_buffer, bank_stagger, width_split}` (see "Catalog evidence"). |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 4, `d_2 == 1`, bf16, TILE, DRAM interleaved, on device | — | buffer address RT (reader) |
| `sizes` | list or tuple of 4 ints, positional | yes | each `-1` or `>= 1`; `sizes[d] in {-1, d_d}` for d in {0, 1, 3}; `sizes[2] in {-1} ∪ [1, ∞)` | — | host only (derives the output shape and the plan) |

There is no `memory_config`, output dtype or `compute_kernel_config`: `out` takes the dtype, layout and memory config of `input_tensor`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, d_1, 1, W)`, rank 4; production `(1, 1, 1, W)` |
| Dtype | bfloat16 |
| Layout | TILE (32×32, four 16×16 faces) |
| Memory | DRAM interleaved, on device |
| Bytes read | only row 0 of each tile (two 32-B segments: face 0 row 0 at page offset 0, face 1 row 0 at page offset 512). Rows 1–31 and nothing else of `x` is ever read — padding-independence by construction. |

### Output

| Property | Value |
|----------|-------|
| Shape | `[d if s == -1 else s for d, s in zip(x.shape, sizes)]` = `(d_0, d_1, H_out, W)` |
| Dtype | bfloat16 (= input) |
| Layout | TILE (= input) |
| Memory | DRAM interleaved (= input's memory config); a new allocation, never a view |
| Padding | Tile padding of `out` (rows ≥ H_out of the last tile row, columns ≥ W of the last tile column) holds copies of `x`'s row-0 tile bytes, including `x`'s own column padding. The contract leaves it unconstrained; no logical element depends on it. |

## Blocking Model

The first design section. Everything below it is the realization of these decisions.

### Axes

Index space: output tiles `(image, replica, col)` with `image = n·d_1 + c`, `replica` = tile row within an image, `col` = tile column. Input tiles `(image, col)`; the input has exactly one tile row per image. Leading `image` and `col` fold into one axis, **pattern** = the input page index `p = image·tensor_pattern_wt + col`: they are contiguous in `x`, and the op treats them identically (each `p` owns one distinct output pattern). Tensor-scope extents (all `ceil`, per image):

- `tensor_pattern_wt = ceil(W / 32)` (production 16, 8, 7, 7, 6)
- `tensor_pattern_tiles = d_0 · d_1 · tensor_pattern_wt` (production = `tensor_pattern_wt`)
- `tensor_replica_rows = ceil(H_out / 32)` (production 16, 8, 7, 7, 6)

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `pattern` (x page `p`; folds `image` × `col`) | **independent** — output tile `(p, r)` depends on input tile `p` alone; no result spans `p`. The input varies along it. | `block_pattern_tiles` | `min(core_pattern_tiles, max_block_pattern_tiles)` = the core's whole pattern assignment (production: 2 at W=512, 1 otherwise) | host plan (`select_plan`), passed as reader/writer CT arg; capacity of `cb_pattern_tiles` derives from it | `pattern_groups` contiguous ranges of `pattern_tiles_per_group` pages, one per **grid column** of the rectangle; `pattern_tiles_per_group = ceil(P / min(P, max_pattern_groups))`, `pattern_groups = ceil(P / pattern_tiles_per_group)`; `max_pattern_groups` defaults to `grid_x` | knob-turn |
| `replica` (out tile row within an image) | **independent, and the input is reuse-shared along it** — no output depends on another, and every replica row of a pattern needs the identical input tile, so splitting it makes every core of a grid column need the same `x` pages. | `block_replica_rows` | `core_replica_rows` — the core's whole replica assignment. The replica axis costs no L1: writes stream over it from the resident pattern pages, so the block spans it fully at any size. | host plan; RT arg (`core_replica_rows`, `replica_start`) | `replica_groups` contiguous ranges of `replica_rows_per_group` rows, one per **grid row** of the rectangle; `replica_rows_per_group = ceil(R / min(R, max_replica_groups))`, `replica_groups = ceil(R / replica_rows_per_group)`; `max_replica_groups` defaults to `grid_y`. The input pages of a grid column are read from DRAM **once** by the column's top core and multicast down the column. | knob-turn (raising `max_replica_groups` adds multicast receivers, not DRAM reads) |
| `image` (`n·d_1 + c`) | independent — folded into `pattern` (see above). | folded: `block_pattern_tiles` | folded | folded | folded: a pattern range may span images; each `p` carries its image as `p / tensor_pattern_wt` | knob-turn |
| in-tile rows (32 rows of one output tile) | not a block axis — sub-tile. All 32 rows of an output tile are the same two 32-B segments of input row 0. | none: the whole tile is the atomic unit (one 2 KB page per DRAM write, catalog `tile_reorder`) | 32 rows, one page | tile format | n/a | n/a |

**Phase 0 realizes the full 2D split on day 1**: the production shapes use 64, 64, 49, 49, 36 cores (Work Distribution). No knob is an inlined constant: `max_pattern_groups`, `max_replica_groups`, `block_pattern_tiles`, `pattern_cb_depth_blocks` and the grid are host parameters; every page count, trip count, runtime arg and the core rectangle derive from `select_plan`.

**The intermediate (the pattern pages) re-run through the table.** A pattern page has the `pattern` axis only; it is built once per grid column by that column's top core (the injector) and multicast to the column. Building it on the injector while the column's other cores wait is a **work-assignment decision**, recorded here rather than left implicit: the build of one page is not divisible across cores without an all-gather of the 2 KB result, and replicating on every core instead (multicast only the 64 B of row 0, each core replicates) leaves the chain `fetch → replicate → write` exactly as long — the replicate sits on every core's critical path either way. The alternative payload is the "mcast payload" perf lamp, not a different occupancy.

**Stall-shadow check.** Two stages wait: the injector on the DRAM fetch of row 0, the receivers on the column multicast. No work in this op is independent of `x`'s row 0 — every output byte is a copy of it, and there is no arithmetic whose algebra would allow a reorder. The only work that fits the windows is setup: building the `TensorAccessor`s, programming the writer's stateful write size (2048 B), and computing the first output page ids. The kernels do that setup before their first blocking wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_pattern_tiles` | `pattern_cb_depth_blocks` (blocks of `block_pattern_tiles` pages) | 1 | Depth 1 holds one block resident while the writer streams it to every replica row. Depth 2 overlaps block `k+1`'s fetch/replicate/multicast with block `k`'s writes, which only exists when `num_blocks_this_core > 1`. Every production core has exactly one block, so depth 2 buys nothing there. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Blackhole DRAM read alignment: source and L1 destination must agree modulo 64 B (`NOC_DRAM_READ_ALIGNMENT_BYTES 64`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`; check `tt_metal/hw/inc/internal/debug/sanitize.h:746`) | the landing offset of the row-0 fetch | The fetch lands each 32-B segment at the **same in-page offset** it has in `x` (0 and 512) in the pattern page itself. CB bases are allocated at DRAM alignment (`tt_metal/impl/program/program.cpp:2505-2509`) and the page size is 2048, so both offsets agree modulo 64. Never fetch into a staging slot at a different offset. | watcher alignment hang (`--dev`), or silently shifted bytes on a lenient path |
| Per-core L1 budget 1,347,776 B (spec, Model environment / BB-05) | `block_pattern_tiles` | `max_block_pattern_tiles = (L1_OP_BUDGET_BYTES − L1_RESERVED_BYTES) // (pattern_cb_depth_blocks · tile_bytes)` = (1,347,776 − 4,096) // 2,048 = **656** at depth 1. `L1_RESERVED_BYTES = 4096` covers the multicast semaphores and alignment. A core whose `core_pattern_tiles` exceeds it walks `num_blocks_this_core > 1` blocks; nothing is refused. | CB allocation overlaps the model's own L1 tensors and fails at launch |
| Stateful self-read size ≤ `NOC_MAX_BURST_SIZE` (`ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.inl:66`) | replicate transfer size | The replicate copies 32-B face-row segments (or larger power-of-two spans within a page if the implementer doubles), always ≤ one burst | compile-time `static_assert` |
| Multicast destination address is the **same L1 address on every core of a grid column** (one `send(src, dst, size)` per block, `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/pipes/mcast_pipe.hpp:92`) | per-core `cb_pattern_tiles` push/pop history | Every core of a column runs the same `num_blocks_this_core`, and every block reserves/pushes/pops the **nominal** `block_pattern_tiles` pages. The ragged last block narrows only the fetch, replicate, send size and writes. This also closes the CB ring each block (depth 1: one push = capacity). | multicast lands on a slot the receiver has not reserved, so the receiver writes stale pages: wrong output |
| NoC L1 alignment 16 B (`noc_parameters.h:375-376`) | replicate segment offsets | Segments are 32 B at 32-B offsets inside a 64-B-aligned page | sanitizer alignment hang |

### Regimes

The named-boundary minimum: **each input byte the output depends on crosses DRAM once, each output page crosses once.** For `x` that is the 64 B of row 0 per input tile, which is below "the input tensor crosses once": rows 1–31 never cross.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 `column_mcast_2d`**: pattern axis over grid columns, replica axis over grid rows; each column's top core fetches row 0 of its pattern range, replicates each page, multicasts the block down the column; every core writes its replica rows × its pattern block | **built** | every valid call (rank 4, `d_2 == 1`, bf16, TILE, DRAM interleaved, any `P ≥ 1`, `R ≥ 1`). The single built regime. A ragged split, `replica_groups == 1` or `num_blocks_this_core > 1` are runtime extents of the same regime, not separate algorithms. | `block_pattern_tiles × core_replica_rows` output pages from `block_pattern_tiles` resident pattern pages | **Minimum on DRAM**: `x` row-0 bytes once per call (P × 64 B), `out` once (P × R × 2 KB). Above it: cross-core, one multicast per block per column, `block_pattern_tiles × 2 KB` injected once and delivered to `replica_groups − 1` receivers, plus the handshake semaphores. | Per block: one fetch round trip and barrier, one replicate barrier, one multicast handshake (receivers' ready → data → flag), one CB reserve/push on the reader, one wait/flush/pop on the writer. A larger `core_replica_rows` spreads that fixed chain over more 2 KB writes. A larger `block_pattern_tiles` puts more pages under one handshake and barrier pair. |
| R0 `per_core_fetch_2d`: the same split, every core fetches and replicates its own pattern block from DRAM, no multicast | **rejected**: superseded by R1 | — | same as R1 | `x` row-0 bytes cross DRAM `replica_groups` times (W=512: 8×; W ≤ 256: once per **output tile**, since `core_replica_rows = 1`). This multiplies 32-B requests on the few banks holding `x`, which then queue behind `out`'s writes. RB-010's dispreferred form. | as R1, minus the handshake |
| R2 `pattern_split_only` (`replica_groups = 1`) | **rejected as the Phase-0 value**; it is R1 at `max_replica_groups = 1`, reachable by a knob turn with no code change | — | `block_pattern_tiles × tensor_replica_rows` | Minimum on DRAM and zero cross-core traffic, the cheapest traffic of all. But the core count is `pattern_groups ≤ tensor_pattern_wt` (W=185: 6 cores), fixed by the tensor's shape, which the spec forbids for multi-tile outputs. | a core writes `R` pages per pattern from one fetch |
| R3 `replica_split_only` (`pattern_groups = 1`) | **rejected**: superseded by R1 | — | `tensor_pattern_tiles × core_replica_rows` | Every core needs every pattern. Delivered by DRAM re-reads that is `x` × all cores; by multicast it needs a 2D broadcast to the whole rectangle (catalog `mcast_topology`: a 1D work split needs a 2D mcast). Every core also holds P pages and replicates all P. | — |
| R4 `fpu_row_broadcast`: build the pattern by unpacking `x` with a row broadcast into DEST and packing | **rejected**: superseded by R1's byte copies | — | — | Same traffic as R1, but values pass through srcB/DEST. The contract forbids any representation that can change bits (denormal flush, NaN payload, signed zero), and R1's NoC byte copies are bit-exact by construction. | — |
| R5 `dram_scatter_rows`: fill the 64 row segments of each pattern page with 32-B DRAM reads directly | **rejected**: superseded by R1's one fetch + local replicate | — | — | 32× more 32-B DRAM requests on `x`'s banks than R1 (RB-010) | — |
| R6 `segment_writes`: write each output page as 64 × 32-B DRAM writes straight from row 0, no pattern page | **rejected**: superseded by R1's whole-page writes | — | — | 64× the write commands, sub-page DRAM writes. Catalog `tile_reorder`: whole 2 KB page writes are at least as fast as face-sized pieces. | — |

There is one built regime, so there is no regime-selection function and no regime-pinned test. `select_plan` still reports its plan (BB-08).

### Traffic ranking

There is no dependent axis, so no "split the dependent axis and combine" candidate exists. Byte counts are for the production shapes (bf16, `P = R`).

| Candidate split | `x` crossings of DRAM (row-0 bytes) | `out` crossings | Cross-core traffic | Parallelism | Verdict |
|-----------------|------------------------------------|-----------------|--------------------|-------------|---------|
| pattern only (R2) | 1 × (P × 64 B) | 1 | none | `≤ P` cores (6–16), set by the shape | cheapest traffic; core count forbidden by the spec as a default |
| 2D, column multicast (R1) | 1 × (P × 64 B) | 1 | per column: `block_pattern_tiles × 2 KB` multicast to `replica_groups − 1` cores | up to `grid_x × grid_y` | **chosen**: DRAM minimum, multicast of a small payload, full occupancy |
| 2D, per-core fetch (R0) | `replica_groups` × (P × 64 B) | 1 | none | up to `grid_x × grid_y` | rejected (RB-010) |
| replica only (R3) | `cores` × (P × 64 B), or one 2D multicast | 1 | every core receives all P pages | `≤ R` cores | rejected |

Production totals for R1 on the 11×10 grid: W=512 reads 1,024 B of `x`, writes 524,288 B, multicasts 8 × 4 KB. W=256: 512 B, 131,072 B, 8 × 2 KB. W=207/196: 448 B, 100,352 B, 7 × 2 KB. W=185: 384 B, 73,728 B, 6 × 2 KB. `out` dominates every case by more than 99%: the op is bound by DRAM writes (feature_spec LOOSE_CASES p1), so the plan's job is to start every core's writes as early as possible and keep every bank busy.

### Block schedule

Logical schedule, realized by two asynchronous kernels on every core of the rectangle: **reader** on NCRISC, NoC0 (role `injector` on the top row, `receiver` below it); **writer** on BRISC, NoC1. No compute kernel.

```cpp
// reader, injector role (top core of each grid column)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    reserve_pattern_block(block_idx);    // cb_pattern_tiles: nominal block_pattern_tiles pages
    fetch_row0_block(block_idx);         // row 0 of each x page in the block -> page offsets 0 / 512
    replicate_block(block_idx);          // rows 1..31 of every page from its own row 0
    mcast_block(block_idx);              // whole block -> same address on the column's receivers
    publish_pattern_block(block_idx);    // push nominal block_pattern_tiles
}
// reader, receiver role (every other core of the column)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    reserve_pattern_block(block_idx);
    receive_block(block_idx);            // ready signal to the injector, wait for the data flag
    publish_pattern_block(block_idx);
}
// writer, every core
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    wait_pattern_block(block_idx);       // nominal block_pattern_tiles
    store_block(block_idx);              // core_replica_rows x block_pattern_tiles whole-page writes
    release_pattern_block(block_idx);    // writes flushed, then pop nominal block_pattern_tiles
}
// both kernels end with every NoC transaction completed (BB-03)
```

| Block operation | Kernel | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------------|--------|------------------------|--------------------|----------------------------------|
| `reserve_pattern_block` | reader | `block_pattern_tiles` pages | — | one CB reserve per block |
| `fetch_row0_block` | reader (injector) | `this_block_pattern_tiles` × 2 segments of 32 B | the reserved block | two 32-B DRAM reads per input page, all issued back to back, **one read barrier per block** |
| `replicate_block` | reader (injector) | `this_block_pattern_tiles` pages × 62 row segments | the block | one stateful-read setup per (page, face half), **one read barrier per block** (not per page) |
| `mcast_block` | reader (injector) | `this_block_pattern_tiles × 2048` B contiguous | the block | **one multicast send per block**: one handshake (wait for `replica_groups − 1` ready signals), one data multicast, one flag |
| `receive_block` | reader (receiver) | same bytes, landing at the receiver's reserved block | the reserved block | one ready signal + one flag wait per block |
| `publish_pattern_block` | reader | `block_pattern_tiles` pages (nominal) | — | one push per block |
| `wait_pattern_block` | writer | `block_pattern_tiles` pages (nominal) | — | one wait per block |
| `store_block` | writer | `core_replica_rows × this_block_pattern_tiles` output pages, replica-row outer, pattern inner (consecutive writes hit different banks when the block has more than one pattern) | the block, read-only, for every replica row | one 2048-B write per output page, issued back to back with **no barrier inside the block**; stateful write size programmed once per kernel |
| `release_pattern_block` | writer | `block_pattern_tiles` pages | — | one writes-flushed wait + one pop per block; one write barrier at kernel end |

Ragged extents (`this_block_pattern_tiles = last_block_pattern_tiles` on the last block of a ragged column, ragged `core_replica_rows` on the last grid row) narrow only the fetch/replicate/send/write work. CB reserve/push/wait/pop counts stay nominal.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Grid synchronization** (core count) | Full participation leaves **one output page per core** at W ≤ 256 (36–64 cores). Every receiver then pays launch, handshake and flag wait for a single 2 KB write, and the DRAM write bound is reached with far fewer writers. | Cap `max_replica_groups` (e.g. 4 or 2) so `core_replica_rows ≥ 2` on W = 185–256; measure W=207, 196, 256. R2 (`max_replica_groups = 1`) is the extreme. |
| **Overlap** (multicast granularity) | One send per block means W=512 receivers wait for both pages of the block to be replicated before writing anything. | One send per pattern page (`block_pattern_tiles` sends per block, receivers push per page) so receivers start writing page 0 while the injector replicates page 1. Measure W=512. |
| **Multicast payload** | The built pattern (2 KB per page) crosses the column. Multicasting only row 0 (2 × 32 B, or the 544-B span `[0, 544)`) right after the fetch moves 32× fewer bytes, and every core replicates its own copy in parallel. | Row-0 multicast + per-core `replicate_block`. Measure W=512 (largest payload). |
| **Replicate mechanism** | 62 stateful 32-B self-reads per page may be issue-bound on NCRISC. | (a) doubling copies (row 0→1, 0–1→2–3, 0–3→4–7, 0–7→8–15 per face half, then faces 0–1 → 2–3: 9 reads, 5 barriers); (b) RISC-V word stores (row 0's 16 words loaded once, 496 stores); (c) on the injector, split the replicate between NCRISC and BRISC (catalog `split_reader`, only if NCRISC is issue-bound). |
| **Bank alignment** | When `tensor_pattern_wt % 8 == 0` (W = 256, 512) every page of output column `c` lies in bank `c % 8`, so every core of a grid column writes the same one or two banks in the same order (catalog `bank_stagger`). | Rotate each core's `store_block` walk by its grid row: start at pattern index `gr % this_block_pattern_tiles` and replica row `gr % core_replica_rows`, wrapping. Measure W=512. |
| **Grid orientation** | Patterns ride grid columns and replicas ride grid rows. On the asymmetric 11×10 Blackhole grid the line placement changes NoC/DRAM contention (catalog `noc_placement`). | Transposed rectangle: patterns on grid rows, `McastCoreOrder.RowMajor` groups. Measure W=512. |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `x` row 0 → injector L1 | raw bf16 bits, two 32-B segments per input page | NoC0 DRAM read through `TensorAccessor(x)` at `get_noc_addr(p, 0)` and `get_noc_addr(p, 512)` into the pattern page at offsets 0 and 512 | landing offset = source offset (DRAM alignment cap) |
| replicate in place | raw bits | NoC0 self-aimed L1→L1 reads (`local_copy` helper): face 0 row 0 → face 0 rows 1–15 and face 2 rows 0–15. Face 1 row 0 → face 1 rows 1–15 and face 3 rows 0–15. | in the destination page itself. No staging buffer, no second CB. |
| injector → column | raw bits, `this_block_pattern_tiles × 2048` B | `ttnn.Mcast`, column-major groups of `replica_groups` cores, fixed sender index 0 (top core), NoC0, `handshake=True` | the handshake is required: it guarantees each receiver has reserved its slot (multi-block back-pressure) and that its semaphores are initialized before data lands |
| reader → writer | pages of `cb_pattern_tiles` | CB push / wait | single producer (reader), single consumer (writer) |
| pattern page → `out` | raw bits, 2048-B page | NoC1 DRAM write through `TensorAccessor(out)`, page id `out_page(p, r) = (p / tensor_pattern_wt) · tensor_replica_rows · tensor_pattern_wt + r · tensor_pattern_wt + p % tensor_pattern_wt` | each output page written exactly once |

No value is ever interpreted: there is no compute kernel, no unpack/pack and no format conversion, so every bit (NaN payloads, ±0, denormals, ±inf) arrives unchanged.

**Placement axis.** TARGET has no `memory_layout` axis: production is DRAM interleaved only, so there are no sharded schemes to classify. If a sharded placement were added, a height cut of `out` (replica axis) and a width cut (pattern axis) are both cuts of independent axes, so each would be a knob-turn that pins this regime's assignment.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_pattern_tiles × core_replica_rows` output pages, sourced from `block_pattern_tiles` pattern pages |
| Grid | one rectangle `CoreRange((0, 0), (pattern_groups − 1, replica_groups − 1))` inside `device.compute_with_storage_grid_size()` (`grid_x × grid_y`, 11 × 10 on the target Blackhole). Grid column `gc` = pattern group, grid row `gr` = replica group. Reader and writer each occupy that one range (BB-17). If the reader is compile-time-specialized per role, the injector row and the receiver rows are two rectangles. |
| Per-core work | core `(gc, gr)`: patterns `[gc · pattern_tiles_per_group, min(P, (gc + 1) · pattern_tiles_per_group))`, replica rows `[gr · replica_rows_per_group, min(R, (gr + 1) · replica_rows_per_group))`. Writes `core_pattern_tiles × core_replica_rows` pages. Core `(gc, 0)` is its column's injector and also writes its own share. |
| Remainder | `pattern_groups = ceil(P / pattern_tiles_per_group)` and `replica_groups = ceil(R / replica_rows_per_group)`, so the last group on each axis is non-empty and ragged. All tile counts use `ceil` per image: `tensor_pattern_wt = ceil(W / 32)`, `tensor_replica_rows = ceil(H_out / 32)`. |
| Why not `split_work_to_cores` | It yields a linear core list in two work groups. R1 needs each pattern group to be one grid column (the multicast line) and each replica group one grid row, so it uses per-axis `ceil` math on the rectangle. |

Production plans on the 11 × 10 grid (`max_pattern_groups = 11`, `max_replica_groups = 10`):

| W | P = R | `pattern_tiles_per_group` | `pattern_groups` | `replica_rows_per_group` | `replica_groups` | cores | pages per core | multicast per column |
|---|-------|---------------------------|------------------|--------------------------|------------------|-------|----------------|----------------------|
| 512 | 16 | 2 | 8 | 2 | 8 | 64 | 4 | 4 KB → 7 receivers |
| 256 | 8 | 1 | 8 | 1 | 8 | 64 | 1 | 2 KB → 7 receivers |
| 207 | 7 | 1 | 7 | 1 | 7 | 49 | 1 | 2 KB → 6 receivers |
| 196 | 7 | 1 | 7 | 1 | 7 | 49 | 1 | 2 KB → 6 receivers |
| 185 | 6 | 1 | 6 | 1 | 6 | 36 | 1 | 2 KB → 5 receivers |

The evenized `ceil` rule uses the **fewest cores that reach the minimum per-core maximum**. At W=512, 16 patterns over 11 columns cannot beat 2 per core, so 8 columns are used, not 11.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_pattern_tiles` | 0 | `tile_bytes` = 2048 (bf16 32×32) | `pattern_cb_depth_blocks × block_pattern_tiles` | spans `pattern` (`block_pattern_tiles` pages live for the whole block); streams over `replica` (the writer re-reads the resident pages for every replica row, no pages per row) | bfloat16 (bits only; never unpacked, no DEST) | reader | writer | one block per CB cycle, every block of the kernel |

Semaphores: allocated by `ttnn.Mcast.attach` from `first_semaphore_id = 0` (ready + data), on the rectangle. There is no other buffer. The full ledger is in `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `reserve_pattern_block` | `block_pattern_tiles` | raw CB API | — | `cb_pattern_tiles`, nominal pages reserved | reserved, unpublished |
| 2 | `fetch_row0_block` (injector) | `this_block_pattern_tiles` × 2 × 32 B | raw `Noc::async_read` + `TensorAccessor` | DRAM `x` | `cb_pattern_tiles` reserved pages, row 0 of every page | row 0 valid, rows 1–31 stale |
| 3 | `replicate_block` (injector) | `this_block_pattern_tiles` × 62 segments | **helper** `dataflow_kernel_lib::set_read_state` / `read_with_state` | `cb_pattern_tiles` row 0 (same pages) | `cb_pattern_tiles` same pages, in place | full pattern pages, still reserved |
| 4 | `mcast_block` (injector) | `this_block_pattern_tiles × 2048` B | **helper** `McastArgs::sender(noc).send` | `cb_pattern_tiles` reserved block (source) | same address on the column's receivers | source guarded (read by the NoC) before `send` returns |
| 4' | `receive_block` (receiver) | same | **helper** `McastArgs::receiver(noc).receive` | — | `cb_pattern_tiles` reserved block | full pattern pages, reserved |
| 5 | `publish_pattern_block` | `block_pattern_tiles` (nominal) | raw CB API | — | `cb_pattern_tiles` push | published to the writer |
| 6 | `wait_pattern_block` | `block_pattern_tiles` (nominal) | raw CB API | `cb_pattern_tiles` published | — | readable |
| 7 | `store_block` | `core_replica_rows × this_block_pattern_tiles` pages | raw `Noc::async_write` (stateful size) + `TensorAccessor` | `cb_pattern_tiles` (read-only, every replica row) | DRAM `out` | — |
| 8 | `release_pattern_block` | `block_pattern_tiles` | raw CB API | `cb_pattern_tiles` | — | popped after `async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `fetch_row0_block` | raw_api | `Noc::async_read`; `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/dataflow/noc.h:252`; `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165`; barrier `noc.h:739` | size 32 B, offsets 0 and 512, page id `p` | DRAM `x` | `cb_pattern_tiles` | trip count `this_block_pattern_tiles` |
| `replicate_block` | helper | `dataflow_kernel_lib::set_read_state<32>(noc, row0_addr)`, `read_with_state(noc, dst_addr, row0_addr)` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:74-75`, `:94-95` (example "fill a destination region with N copies of one stick", `local_copy_helpers_dataflow.inl:41-50`); barrier `noc.h:739` | `transfer_size = 32` | `cb_pattern_tiles` | `cb_pattern_tiles` (in place) | trip count `this_block_pattern_tiles` |
| `mcast_block` | helper | `McastArgs<CT, RT>().sender(noc).send(src_l1, dst_l1, size_bytes)` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:37-42`, `:217`; `kernel/pipes/mcast_pipe.hpp:92`; source guard `kernel/pipes/pipe_common.hpp:13` | `src_l1 = dst_l1` = the block's write pointer, `size = this_block_pattern_tiles · 2048` | `cb_pattern_tiles` | receivers' `cb_pattern_tiles` | `size` from `this_block_pattern_tiles` |
| `receive_block` | helper | `McastArgs<CT, RT>().receiver(noc).receive()` | `mcast_args.hpp:44-47`, `:219`; `mcast_pipe.hpp:163` | round 0..`num_blocks_this_core − 1` | — | `cb_pattern_tiles` | — |
| host channel | helper | `ttnn.Mcast(device, ttnn.McastConfig(noc=NOC_0, handshake=True), rectangle, replica_groups, ttnn.McastFixedSenderConfig(), ttnn.McastCoreOrder.ColumnMajor)`; `.attach(descriptor, "pattern_mcast", [reader_kernel], 0)` | `ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast.hpp:96-110`, `:133-139`, `:149-153`; Python binding `ttnn/cpp/ttnn-nanobind/mcast_host.cpp:59-79`; usage `ttnn/ttnn/operations/examples/mcast_topology/mcast_topology.py:373-380` | receiver group size `replica_groups` | — | — | `replica_groups` |
| `store_block` | raw_api | `Noc::set_async_write_state` + `async_write_with_state` (or `async_write`); `TensorAccessor::get_noc_addr(page_id)` | `noc.h:562`, `:610` (`:411`); `tensor_accessor.h:165`; `async_writes_flushed` `noc.h:783`; `async_write_barrier` `noc.h:760` | 2048 B per page | `cb_pattern_tiles` | DRAM `out` | trip counts `core_replica_rows`, `this_block_pattern_tiles` |
| kernel exit (BB-03) | raw_api | `Noc::async_write_barrier`, `Noc::async_atomic_barrier` (reader: multicast data/flag writes and ready-signal increments) | `noc.h:760`, `:805` | — | — | — | — |

**Helpers considered and rejected for the raw entries:**

| Raw entry | Candidate | File:line of the mismatch | Reason |
|-----------|-----------|---------------------------|--------|
| `fetch_row0_block` | `local_copy_helpers_dataflow.hpp` | `:57` (`local_addr` builds a source aimed at this core's own L1) | self-aimed L1 only; cannot address a DRAM page through a `TensorAccessor` |
| `fetch_row0_block` | `l1_helpers.hpp` | `:100-101` (`fill_l1_range` writes one constant value) | fills a constant, does not read |
| `fetch_row0_block`, `store_block` | `tile_column_dataflow.hpp` | `:9-24` (writes one tile column with RISC-V stores) | column of a tile in L1, not a DRAM page transfer |
| `store_block` | kernel_lib (all dataflow headers: `dfb_helpers_dataflow.hpp:14-18`, `reduce_helpers_dataflow.hpp`, `tilize_helpers_dataflow.hpp`, `mcast/`) | — | kernel_lib has no DRAM page writer. The block operation is the raw `Noc` write issued per page inside `store_block`. |
| all | compute helpers (`eltwise_convenience.hpp`, `untilize_helpers.hpp`, …) | — | regime R4 rejected: a compute path passes values through srcB/DEST, which the bit-identity contract forbids |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Combining the 32 rows of an input tile | A reduce/bcast-style read of `x` touches rows 1–31, which hold the model's garbage. The golden suite poisons them with 1000.0. | Only bytes `[0, 32)` and `[512, 544)` of each `x` page are read. Rows 1–31 never leave DRAM. |
| Bits changed in transit | NaN payloads, −0, denormals (golden `special_values` case) | Pure NoC byte copies, no compute kernel |
| Fetch alignment on Blackhole | DRAM reads require src ≡ dst (mod 64) | Land at the same in-page offset (Mechanism caps) |
| Multicast address divergence | Receivers must have the block reserved at the injector's address | Nominal push/pop counts; identical block sequence per column; `handshake=True` |
| Ragged last pattern group | Its column has fewer pages than nominal | Send size and writes use `this_block_pattern_tiles`; CB counts stay nominal |
| `replica_groups == 1` (`R == 1`: `H_out ≤ 32`) | A group of one core has no receivers. The helper's zero-fan-out `send` is a degenerate local copy (`mcast_args.hpp:21-23`). | Either keep the channel (send onto itself: same bytes) or attach an absent channel (`ttnn.attach_absent_mcast`) and skip `mcast_block`. Same kernels either way. |
| Transactions left in flight (BB-03) | Ready-signal increments and multicast flags are NoC transactions; the next program starts right after | Reader ends with `async_write_barrier` + `async_atomic_barrier`; writer with `async_write_barrier` |
| Hand-over before data lands (BB-06) | A receiver publishing before the flag | `receive()` returns only after delivery (`mcast_pipe.hpp:140-141`); publish follows it |
| Out-of-memory as a support statement (BB-05) | Large W | Clamp `block_pattern_tiles ≤ 656`; extra pages become extra blocks; nothing is refused |
| `d_2 ≠ 1` or a non-dim-2 expansion | Outside the Phase-0 contract | `ValueError` in `validate()` (Host contract) |
| Program rebuilt per call (BB-07) | Python building a descriptor and attaching an `Mcast` every call | Cache the descriptor per key; patch only the two buffer addresses per call (pattern: `tests/ttnn/unit_tests/base_functionality/test_kernel_defines_hash.py:45`) |

## Host contract

### Registry (`ttnn/ttnn/operations/g6_expand/g6_expand.py`)

| Item | Decision |
|------|----------|
| `INPUT_TAGGERS` | `{"alignment": tag_alignment}`. `tag_alignment(inputs, axes)`: `scenario = inputs[0]`; `out = [d if s == -1 else s for d, s in zip(scenario["shape"], scenario["sizes"])]`; `"w_non_aligned"` if `out[-1] % 32` (H is then not read), else `"h_non_aligned"` if `out[-2] % 32`, else `"tile_aligned"`. |
| `SUPPORTED` | `dtype: [ttnn.bfloat16]`, `layout: [ttnn.TILE_LAYOUT]`, `alignment: ["tile_aligned", "w_non_aligned"]` (= TARGET) |
| `EXCLUSIONS` | `[]` |
| `validate(input_tensor, sizes)` order | (1) structural checks raising `ValueError`: on device; DRAM interleaved; rank 4; `len(sizes) == 4`; every size `-1` or `≥ 1`; a size `≠ -1` differing from a non-1 input dim; an expansion of dim 0, 1 or 3; `d_2 != 1`. (2) SUPPORTED per axis → `UnsupportedAxisValue`. (3) EXCLUSIONS → `ExcludedCell`. Structural checks come first because the tagger needs a well-formed output shape. |
| Entry point | `g6_expand(input_tensor, sizes)`: `validate` first; output shape; `ttnn.allocate_tensor_on_device` in the input's memory config; cached descriptor; exactly one `ttnn.generic_op([input_tensor, output], descriptor)`. No `pad`/`slice`/`to_layout`/reshape on input or output. |
| Plan reporting (BB-08) | `select_plan(shape, sizes, grid) -> Plan` (frozen: rectangle, `pattern_tiles_per_group`, `pattern_groups`, `replica_rows_per_group`, `replica_groups`, `block_pattern_tiles`, `num_blocks` per column, role per core). Exported, and the plan of the last call kept as `g6_expand.last_plan`. |
| Descriptor cache (BB-07, BB-16) | Key: input shape, output shape, dtype, layout, memory config, device id and grid. A hit patches only the `x` and `out` buffer-address runtime args. No allocation besides the output tensor, and no kernel build after the first call per key. |
| Profiling zones (BB-21) | Zone build only (no change to the plain build): reader `G6_FETCH`, `G6_REPLICATE`, `G6_MCAST_SEND`, `G6_MCAST_WAIT`, `G6_CB_RESERVE`; writer `G6_CB_WAIT`, `G6_STORE`, `G6_DRAIN`. Role per core comes from `Plan` (`injector` / `receiver`). |

### Structural impossibilities

None beyond `feature_spec.py`'s `INVALID = []`.

## Catalog evidence

| Entry | Knob it informed |
|-------|------------------|
| `mcast_topology` | Broadcast along the axis the operand does not vary with. Column-major `ttnn.Mcast` groups on a 2D split. A 1D replica split would need a 2D mcast (R3 rejected). |
| `tile_reorder` | Whole 2 KB page per DRAM write, batched barriers (R6 rejected; `store_block` has no per-page barrier) |
| `noc_placement` | Reads on NoC0, writes on NoC1 (reader/writer defaults); grid-orientation lamp |
| `double_buffer` | No buffering gain once DRAM-bound with enough cores. Depth 1 is the Phase-0 value; depth 2 only for multi-block cores. |
| `bank_stagger` | Bank-alignment lamp (`tensor_pattern_wt % 8 == 0`) |
| `width_split` | A short, wide output must not be split by the axis that has few units. Here both axes are spread. |

## Acceptance test

`tests/ttnn/unit_tests/operations/g6_expand/test_g6_expand.py`, limited by BB-22 to the five production cases and the cases the spec's rules require. It deliberately does not sweep single-tile, non-square or multi-batch shapes: production never calls them, and the spec overrides the generic sweep. Every case is graded bit-exactly (`TOLERANCES[bf16] = ("exact", None)`): production W ∈ {512, 256, 196, 185, 207}; sizes as a tuple; sizes with −1; special bf16 bit patterns; poisoned input padding; output is a new DRAM-interleaved TILE bf16 tensor and the input is left unchanged; repeat determinism with no new program-cache entry on a repeat call; the mixed-width sequence under the program cache; and the spec's validation errors.

# Operation Design: g6_embedding

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (index-driven row gather + row-major → TILE conversion; no arithmetic) |
| Goal | Gather one table row per uint32 index on the device and write the rows as a bfloat16 TILE tensor `idx.shape + [D]`, every value bit-identical to the table, in one device program. |
| Math | `out[i, d] = W[idx[i], d]` for every position `i` of `idx`, `d ∈ [0, D)`; `W = weight` (`[V, D]`) or `weight[0]` (`[1, V, D]`) |
| References | `docs/explorations/g6_embedding/DOSSIER.md`, `math.md`, `tensix.md`, `risks.md`, `lessons.md`; `eval/golden_tests/g6_embedding/feature_spec.py`, `helpers.py`; `.claude/references/blocking-model.md`, `l1-footprint-discipline.md`, `noc-alignment-in-tt-metal.md`; perf catalog `ttnn/ttnn/operations/examples/master.md`; device probes `tests/ttnn/unit_tests/operations/g6_embedding/probes/probe_00{1,2,3}.py` |
| Companion artifact | `l1_ledger.md` (CB ledger + data-movement budget), same directory |

### Design-deciding evidence (measured on this Blackhole, 11×10 grid)

| Probe | What ran | Result | Consequence |
|-------|----------|--------|-------------|
| `probe_001.py` | `ttnn.tilize` of a bf16 ROW_MAJOR tensor holding -0.0, subnormals, ±inf, NaN patterns (engine tilize, Float16_b CBs, `fast_tilize`) | 717/2048 (W=64) and 14336/40960 (W=1280) values changed: `0x8000→0x0000`, `0x0001/0x8008/0x007F→0x0000`, `0x7FC0/0x7F81→0x7F80`, `0xFFFF→0xFF80` | Float16_b-typed tilize is **forbidden** (spec rule 2) |
| `probe_002.py` | same bit patterns carried as `ttnn.uint16`, `ttnn.tilize` (standard tilize path, 16-bit DEST, UInt16 CBs) | 0 mismatches at W=64 and W=1280 | the engine tilize is bit-exact when the CBs are typed **UInt16** |
| `probe_003.py` | all 65,536 16-bit patterns, shapes `(32, 2048)` and `(1024, 64)`, uint16 tilize | 0 mismatches | UInt16 tilize is a pure bit permutation for every bf16 pattern |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes (positional) | uint32, rank 1 `[N]` or rank 2 `[B, S]`, DRAM interleaved; Phase 0 ROW_MAJOR | — | buffer address: common RT arg |
| `weight` | `ttnn.Tensor` | yes (positional) | bfloat16, ROW_MAJOR, `[V, D]` or `[1, V, D]`, `D % 32 == 0`, DRAM interleaved | — | buffer address: common RT arg |
| `layout` | `ttnn.Layout` | no (kw-only) | `ttnn.TILE_LAYOUT` only (axis `output_layout`) | `ttnn.TILE_LAYOUT` | host only |
| `memory_config` | `ttnn.MemoryConfig \| None` | no (kw-only) | `None` or DRAM interleaved | `None` (= DRAM interleaved) | host only |
| `runtime_rows` | `bool` | no (kw-only) | `False` only (axis `runtime_rows`) | `False` | host only |

Not parameters (refused by `TypeError` from the signature): `padding_idx`, `max_norm`, `norm_type`, `scale_grad_by_freq`, `sparse`, `compute_kernel_config`.

## Tensors

### Input — `input_tensor` (idx)

| Property | Requirement |
|----------|-------------|
| Shape | `[B, S]` or `[N]` (rank-1 is read as `B = 1, S = N`) |
| Dtype | `ttnn.uint32` |
| Layout | Phase 0: ROW_MAJOR — one page per row of `B`, `4·S` bytes (rank 1: one page of `4·N` bytes). Refinement: TILE (positions in face order; padding positions are not indices) |
| Memory | DRAM interleaved |

### Input — `weight` (W)

| Property | Requirement |
|----------|-------------|
| Shape | `[V, D]`, or `[1, V, D]` read **as it arrives** (page id = row id; no reshape, no copy — BB-02) |
| Dtype | `ttnn.bfloat16` (axis `dtype`) |
| Layout | ROW_MAJOR — one page per table row, `2·D` bytes; `D % 32 == 0` ⇒ page size is a multiple of 64 B ⇒ `aligned_page_size == page_size` on Blackhole |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `[B, S, D]` (rank-2 idx) or `[N, D]` (rank-1 idx); padded `[B, ceil(S/32)·32, D]` / `[ceil(N/32)·32, D]` |
| Dtype | `ttnn.bfloat16` (= `weight.dtype`) |
| Layout | TILE; tile page id of `(b, tr, tc)` = `(b·St + tr)·Dt + tc` with `St = ceil(S/32)`, `Dt = D/32` |
| Memory | DRAM interleaved (`memory_config=None` or explicit DRAM interleaved); allocated per call with `ttnn.allocate_tensor_on_device` |
| Padding rows | rows `≥ valid_rows` of each image's last tile row hold whatever the staging buffer held (unspecified by contract) |

## Blocking Model

The op has **no dependent axis**: every output value is a copy of one input value (`math.md#axes-and-dependence`). The work is a 2-D grid of **units**: one unit = one output tile row (32 tokens of one image) × one column piece of `col_block_tiles` tile columns. Units are numbered **column-piece-major** (`unit = p·T + t`, piece `p` outer, tile row `t` inner), so every core's contiguous unit range is a run of tile rows inside one column piece and every block is a rectangle.

Symbols used below (host-derived once per call signature):

| Symbol | Definition |
|--------|------------|
| `Dt` (`tensor_col_tiles`) | `D / 32` |
| `St` (`image_tile_rows`) | `ceil(S / 32)` (`S = N` for rank 1) — per image, never `floor(B·S/32)` |
| `T` (`tensor_tile_rows`) | `B · St` (`B = 1` for rank 1) |
| `C` (`grid_cores`) | `grid.x · grid.y` of `device.compute_with_storage_grid_size()` (110 on this Blackhole) |
| `P` (`col_pieces`) | `Dt / col_block_tiles` |
| `U` (`total_units`) | `T · P` |
| `valid_rows(t)` | `min(32, S − (t mod St)·32)` |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| tokens — tile rows of the result (`S` or `N`, tiled per image) | independent — each output row reads only its own index and the row it names | `block_tile_rows` (tile rows per reader block; tilize consumes 1 tile row at a time inside the block) | `token_split`: `max(1, min(bt_l1, ceil(core_units_max / min_blocks_per_core)))` with `min_blocks_per_core = 4`; `token_column_split`: same formula, which yields 1 for every INPUTS shape (≤ 2 units per core). Values on C = 110: p1 `[16,4096]` → 5, p2 `[8,4096]` → 3, p3 `[16,1024]` → 2, p4 `[1,3844]` → 1 | host plan function, one value per call signature → reader CT arg; CB pages derived from it | spread: contiguous unit ranges via `ttnn.split_work_to_cores(grid, U, row_wise=True)` over the full grid | knob-turn |
| images — `B` | independent — rows of two images never share a tile (`St` tile rows per image) | none of its own: folded into the tile-row index `t = b·St + tr`; a block may span images, each tile row carries its own `(b, tr, valid_rows)` | folded | — | spread with the tokens (same flattened unit index) | knob-turn |
| columns — `D` (tile columns `Dt`) | independent — column `d` of the result reads column `d` of the table only | `col_block_tiles` (tile columns per unit = tilize block width) | if `T ≥ C`: `ct_cap` (whole rows, `P = 1`); else the largest divisor `d` of `Dt` with `d ≤ ct_cap` and `T·(Dt/d) ≥ C`, or `1` if none. `ct_cap` = largest divisor of `Dt` with `≤ MAX_COL_BLOCK_TILES (64)` and `footprint(1, d) ≤ L1_BUDGET`. Values on C = 110: D=1280 & `T ≥ 110` → 40; `[1,1000]`/`[1,1024]`/`[8,128]` D=1280 → 10; FLUX `[1,512]` D=2560 → 10; one-tile-row calls → 1 | host plan function → compute CT arg (tilize template param) + reader/writer CT arg | spread: column pieces are units; `P > 1` only when the tile rows alone cannot fill the grid | knob-turn |
| table rows — `V` | not an output axis — data-dependent addressing (`page = idx[i]`); nothing to combine | none — **not blocked, not split**: a row is read whole-slice by the token that names it; the table stays in DRAM | — | — | not assigned: any core reads any row | — |
| tile-internal rows/faces (32 rows × 4 faces) | fixed by the 32×32 tile | none (tile geometry) | 32 rows per tile row | — | — | — |

Operand-reuse check (every operand × the chosen split):

| Operand | Varies along the split axes? | Consequence |
|---------|------------------------------|-------------|
| `idx` | varies along tokens; **does not vary along columns** | with `P > 1` the 128-B index chunk of a tile row is read by the `P` cores holding its column pieces → reuse-shared by construction of the column split → `index_broadcast` regime row (deferred) |
| `W` | rows chosen by data; a row is shared only when indices repeat | not statically exploitable → `repeat_dedup` regime row (rejected) |
| `out` | written once per tile | none |

Stall-shadow check:

| Waiting stage | Waits on | Independent work scheduled into the stall | Legal because |
|---------------|----------|-------------------------------------------|---------------|
| reader, `gather_rows_block(k)` barrier | the block's table-row reads landing | `fetch_index_block(k+1)` is issued **before** waiting on block `k`'s barrier (2-slot index ring) | index values of block `k+1` do not depend on table data; no floating point is involved (pure data independence, no algebraic property needed) |
| reader, first block | the index read gating every row read | nothing — it is issued first, before any bulk read (lesson RB-012) | — |
| compute, `tilize_block` | a unit's rows | nothing independent (tilize needs the rows) | — |
| writer, `store_tiles_block` | a unit's tiles | nothing; per-unit `noc_async_writes_flushed()` instead of a write barrier so the writer never waits for DRAM write completion mid-kernel | — |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_index` | `index_depth` | 2 (blocks) | prefetch of block `k+1`'s indices while block `k`'s row reads are in flight (stall-shadow fill) |
| `cb_rows` | `rows_depth` | 2 (blocks) | reader gathers block `k+1` while compute tilizes block `k` |
| `cb_tiles` | `tiles_depth` | 2 (units) | compute tilizes unit `j+1` while the writer drains unit `j` |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Blackhole DRAM read rule: local L1 addr ≡ DRAM addr (mod 64) (`noc-alignment-in-tt-metal.md` §1; `core_config.h:42` `LOG_BASE_2_OF_DRAM_ALIGNMENT 6`) | column-slice offset and staging-row stride | slices are whole tile columns: source offset `p·col_block_tiles·64` B ≡ 0, staging row stride `col_block_tiles·64` B ≡ 0, CB bases are DRAM-aligned (`program.cpp:2505-2509` allocates CBs at `get_alignment(BufferType::DRAM)`) | sub-tile-column slices (32-B face halves) land at offset 32 mod 64 → silent corruption, NoC error under watcher |
| same rule for index chunks | index chunk offset/slot stride | chunk offset `tr·128` B ≡ 0; `cb_index` slot stride 128 B | misaligned index values → wrong rows read |
| Tilize helper path: UInt16 is not a fast-tilize format (`tilize_helpers.inl:31-35, 66-69`), so the standard `tilize_block` path runs. Bit-exactness is a property of the format path, not the width (probe_003 ran all 65,536 patterns through this helper at 2 tiles wide); the width only sets the unpacker's row stride, which no probe here pinned above 2 tiles through this helper. Production max is 40 | `col_block_tiles` | `MAX_COL_BLOCK_TILES = 64` host constant; the first `token_split` run on p1 (`col_block_tiles = 40`) is the width check | a stride the unpacker tilize does not support → wrong tiles |
| Tilize helper: `block_width_tiles` is a template parameter (`tilize_helpers.hpp:194-207`) | `col_block_tiles` | must be a compile-time arg of the compute kernel | — (compile error) |
| L1 bound BB-05: `L1_BUDGET = 1,347,776` B for all op buffers on a core | `block_tile_rows`, `col_block_tiles` | `footprint(bt, ct) = 2048·ct·(rows_depth·bt + tiles_depth) + 128·index_depth·bt ≤ L1_BUDGET`; `bt_l1 = floor((L1_BUDGET − 2048·ct·tiles_depth) / (2048·ct·rows_depth + 128·index_depth))` | refused launch inside the model |
| Image boundary: a tile row never spans two images | tile rows | each tile row decoded to `(b, tr, valid_rows)`; reader reads exactly `valid_rows` index values and rows | rows of image `b+1` shifted into image `b`'s padding |
| Page id width: uint32 index, `V ≤ 262,144` in INPUTS | index value as page id | reader clamps `page = min(idx, V − 1)` (no effect on in-contract indices; keeps an out-of-contract value inside the table allocation instead of a wild DRAM address) | out-of-contract garbage index → read past the table, possible NoC fault |
| CB ring-wrap (pointers reset only on an exact `fifo_limit` hit) | `cb_rows` push quanta | every non-final block pushes the nominal `block_tile_rows·col_block_tiles` pages; only a core's **last** block may be short (single pass per launch, pointers reset per launch) | pointer overshoot into neighbouring L1 |

### Regimes

Selection (host, once per call signature; exact, depends only on shapes and grid):

```
Dt = D/32; St = ceil(S/32); T = B*St; C = grid.x*grid.y
ct_cap = max{ d | d divides Dt, d <= MAX_COL_BLOCK_TILES, footprint(1, d) <= L1_BUDGET }
ct     = ct_cap                                                        if T >= C
       = max{ d | d divides Dt, d <= ct_cap, T*(Dt/d) >= C }, else 1   if T <  C
regime = "token_split" if Dt/ct == 1 else "token_column_split"
```

| Regime | Status | Predicate | Block | Data movement vs. minimum (boundary: DRAM; minimum = idx read once, each named row read once per token (`2·N·D` B), each output tile written once) | What a bigger block buys |
|--------|--------|-----------|-------|-----------------------------------------------------------|--------------------------|
| `token_split` | built | `Dt / ct == 1` (on C = 110: every INPUTS case with `T ≥ 110` and D ∈ {64, 1280}: `[16,4096]`, `[8,4096]`, `[16,1024]`, `[8,1024]`, `[1,3844]`, `[1,4096]`, both tables) | `block_tile_rows × Dt` tiles (32·`block_tile_rows` tokens × whole rows) | **minimum**: idx 1×, W `N` reads of `2D` B (one per token; repeats re-read — the minimum for a gather without dedup), out 1× in `T·Dt` whole-tile writes | reader: one read-barrier drain (one DRAM round trip with nothing in flight on this core) and one `cb_rows` reserve/push per block, intended once per block; index fetch once per block (prefetched). Compute init/uninit and writer barrier are once per kernel independent of the block |
| `token_column_split` | built | `Dt / ct > 1` (on C = 110: every INPUTS case with `T < 110`, plus any shape where `ct_cap < Dt`) | `block_tile_rows × col_block_tiles` (Phase 0: `1 × ct` for every INPUTS shape) | above minimum by occupancy choice: idx chunk read `P` times (≤ `P·128` B per tile row), W `N·P` row-slice reads of `2·D/P` B (same bytes, `P×` transactions); out 1× | as above, plus fewer column pieces ⇒ fewer duplicate index reads and fewer, larger row-slice transactions (the grid-synchronization lamp measures this against occupancy) |
| `tile_index_fetch` | deferred — the TILE index layout is outside Phase 0 SUPPORTED by contract (2 of 49 cases, FLUX 42/45) | `input_tensor.layout == TILE` | unchanged | idx: two 64-B face-row reads per tile row (`face (b%32≥16)·2 + {0,1}`, offset `(b%16)·64` in index tile `(b/32)·ceil(S/32) + tr`), only `valid_rows` positions used; else identical | identical |
| `index_broadcast` | deferred — `token_split` already reads every index byte exactly once (no duplicate bytes); duplicates exist only under `token_column_split`, ≤ `P·128` B per tile row against `2·32·D` B of row reads per tile row, so the remaining gain is request queueing on the single DRAM bank of a `B = 1` index page (RB-010, rule 8), which a measurement on p4 `[1,3844]` / `[1,4096]` settles. Reachable: one injector per column-piece group reads the chunk(s) and multicasts into the receivers' `cb_index` slot (`ttnn.Mcast`, `mcast/host/mcast.hpp`; `McastArgs`, `mcast/kernel/mcast_args.hpp:117`); `cb_index` keeps its single consumer (reader), gather/tilize/store unchanged | `B == 1` or `P > 1` | unchanged | idx reads: `cores·chunk` → one read per group + one multicast; W, out unchanged | adds a semaphore round trip per block on receivers |
| `bf16_engine_tilize` (Float16_b CBs, `fast_tilize`) | rejected — not bit-exact (probe_001: -0.0, subnormals, NaN payloads destroyed); superseded by the UInt16-typed tilize (probe_002/003: 0 mismatches over all 65,536 patterns) | — | — | same as `token_split` | — |
| `dm_face_permute` (rows staged in L1, then 32-B NoC self-reads or RISC-V stores place face rows) | rejected — 64 local NoC transactions per output tile (≈ 47.7K per core at p1) on the DM RISCs that also issue the DRAM traffic; superseded by the UInt16 engine tilize (one tilize per unit on otherwise-idle TRISCs) | — | — | same bytes, +2·Dt L1→L1 transactions per token | — |
| `face_direct_gather` (DRAM reads straight into face rows) | rejected — `2·Dt` DRAM reads of 32 B per token (5.2M at p1) and the odd halves start at 32 mod 64, violating the Blackhole DRAM read rule; superseded by `token_split` (one read per token) | — | — | `2·Dt×` row transactions | — |
| `repeat_dedup` (fetch each distinct index of a block once) | rejected — for uniform indices a block of `32·bt` tokens holds ≈ `(32·bt)²/(2V)` repeated rows (0.05 per 160-token block at p1, 0.03% of reads) and needs a per-block compare on the reader; superseded by `token_split` | — | — | saves ≈ 0.03% of W reads at p1 | — |
| `table_pass` (tilize or copy the whole table, then gather tiles) | rejected — forbidden by spec rule 1 (671 MB vs ≤ 168 MB per call, and a replayed trace) | — | — | `V/N×` W traffic | — |

No "split a dependent axis + combine" regime exists: the op has no dependent axis.

Regime-pinned tests (required; the predicate depends on `C`, so `[1,3844]` (T=121) and `[1,4096]` (T=128) flip to `token_column_split` on a 130- or 140-core part): `[16,4096]`×`[248320,1280]` must report `token_split` on any grid `C ≤ 2048`; `[1,1]`×`[248320,1280]` must report `token_column_split` on any grid `C ≥ 2`. Both are in the acceptance test through `describe_plan`.

### Traffic ranking

| Candidate split | Bytes / transactions crossing DRAM | Cross-core traffic | Rank |
|-----------------|-------------------------------------|--------------------|------|
| tokens only (`P = 1`) | idx 1× (`4·N` B in `T` chunk reads); W `2·N·D` B in `N` reads; out 1× in `T·Dt` tile writes | none | 1 — the minimum |
| tokens × columns (`P > 1`) | idx `P×` (`≤ P·128` B per tile row); W same bytes in `N·P` reads; out 1× | none | 2 — same bytes, `P×` row transactions, `P×` index reads |
| tokens × columns + index broadcast | as rank 2 but idx 1× from DRAM | `(P−1)` multicast deliveries of 128 B per tile row + semaphores | between 1 and 2 (deferred row) |
| columns only | idx `Dt/ct ×` of the whole index, W `N·P` reads | none | 3 — strictly dominated by tokens × columns |
| any table-side split (`V`) | a table pass (rule 1) | — | excluded |

Chosen: rank 1 whenever the tile rows fill the grid (`T ≥ C`); rank 2 when they do not — an explicit trade of `P×` row transactions for occupancy (spec rule 6: "the number of cores used MUST be the blocking decision's choice"). Occupancy is not used to override rank 1.

### Block schedule

Per core: units `[unit_begin, unit_end)` (column-piece-major), cut into blocks of at most `block_tile_rows` tile rows that never straddle a column piece.

```cpp
fetch_index_block(0);
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    fetch_index_block(block_idx + 1);   // reader: prefetch (only if block_idx + 1 exists)
    gather_rows_block(block_idx);       // reader: one row-slice read per valid token -> cb_rows; one barrier; push
    tilize_block(block_idx);            // compute: UInt16 tilize, one tile row x col_block_tiles per step
    store_tiles_block(block_idx);       // writer: col_block_tiles whole-tile writes per tile row
}
// writer: one noc_async_write_barrier() before return (BB-03); reader: last barrier covers every read
```

| Block operation | Kernel | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------------|--------|------------------------|--------------------|-----------------------------------|
| `fetch_index_block` | reader | `block_tile_rows` chunks of `valid_rows(t)` uint32 (≤ 128 B each), DRAM page `b`, offset `tr·128`; one read per maximal run of tile rows inside one image (contiguous bytes of page `b`) | the other `cb_index` slot (block being gathered) | TensorAccessor for idx built once per kernel; no barrier of its own — retired by the next `gather_rows_block` barrier (block 0: own barrier, issued before any bulk read) |
| `gather_rows_block` | reader | `Σ valid_rows` row-slice reads of `col_block_tiles·64` B: source page `min(idx, V−1)`, offset `p·col_block_tiles·64`; destination `cb_rows` slot row `r` of tile row `j` at `(j·32 + r)·col_block_tiles·64` | `cb_index` slot of this block (addresses) | `cb_rows` reserve/push once per block (`units_in_block·col_block_tiles` pages); one `noc_async_read_barrier` per block; zero per-row barriers |
| `tilize_block` | compute | `units_in_block` steps of 1 tile row × `col_block_tiles` tiles (UInt16 → UInt16) | nothing beyond the CB windows | `compute_kernel_hw_startup` once; tilize init/uninit **once per kernel** (one helper call with `num_blocks = units_this_core`, or per-unit calls with `InitUninitMode::InitOnly/Neither/UninitOnly`); CB wait/pop of `col_block_tiles` per unit |
| `store_tiles_block` | writer | per unit `col_block_tiles` writes of one 2048-B tile to page `t·Dt + p·col_block_tiles + j` | nothing | `cb_tiles` wait/pop per unit; `noc_async_writes_flushed()` per unit before the pop; `noc_async_write_barrier()` **once per kernel** at the end |

Profiling zones (BB-21; compiled only into the zone build): reader `idx-fetch`, `rows-reserve`, `rows-issue`, `rows-wait`; compute `tilize-wait`, `tilize`; writer `tiles-wait`, `tiles-write`, `write-barrier`. Every core holds the same three roles; `describe_plan()` reports the core ranges (role dump).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| overlap | `block_tile_rows = ceil(units/4)` makes the first block's gather (p1: 5 × 80 KB) the pipeline fill before any tile is written, and the last block's tilize+write the drain | p1/p2: `min_blocks_per_core` 2 and 8 (bt 10/3 at p1), and `block_tile_rows = 1` |
| grid synchronization | one-tile-row calls (`[1,1]`, `[1,8]`, `[1,16]`, `[1,31]` — 60–20 traced calls each) run 40–160 cores with one 64-B slice and one tile each; launch and duplicate index reads may dominate | `col_block_tiles` floor of 2, 4, 8 tiles (10–40 cores) on `[1,1]`×D=1280 and `[1,1]`×D=5120 |
| load balance | `T` just above `C` (`[1,4096]`: 128 units on 110 cores, `[1,3844]`: 121) gives a makespan of 2 units against a mean of 1.16 | force `P = 4` (`col_block_tiles = 10`) on `[1,4096]` and `[1,3844]` |
| reader barrier drain | one `noc_async_read_barrier` per block drains this core's read stream; D=64 rows are 128 B, so a block is only `32·bt·128` B in flight | trid-pipelined gather (`set_read_trid` / `async_read_barrier_with_trid`, `local_copy_helpers_dataflow.hpp:147,158`; catalog `trid_double_issue`) with `rows_depth` 3–4 on `[16,4096]`×D=64 and p1 |
| write bank alignment | `Dt = 40` and 8 DRAM banks: every core's `j`-th tile write of a tile row lands on bank `j mod 8` (`t·40 ≡ 0 mod 8`) | start each core's tile-row write order at `core_index mod num_dram_banks` (catalog `bank_stagger`) |
| tilize engine rate | UInt16 runs the standard (not fast) tilize path; at p1 ≈ 745 tiles per core must hide under ≈ 730 µs of DRAM time | measure the `tilize` zone against `rows-wait` on p1; if exposed, raise `tiles_depth` to 3 |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| idx DRAM → `cb_index` | uint32, natural order, 128-B slot per tile row | `noc_async_read` via `TensorAccessor(idx)` page `b` + byte offset `tr·128`, size `valid_rows·4` | reader-private scratch; read only logical positions |
| W DRAM → `cb_rows` | bf16 bytes carried as UInt16, row-major `32·bt` rows × `col_block_tiles·32` values | one `noc_async_read` per valid token: `TensorAccessor(W).get_noc_addr(min(idx, V−1), p·ct·64)`, size `ct·64` | `[1, V, D]` weight addressed as is (pages = rows); never a table pass |
| `cb_rows` → `cb_tiles` | UInt16 tiles (32×32, four 16×16 faces) | `compute_kernel_lib::tilize<ct, cb_rows, cb_tiles>` on 16-bit DEST | the only conversion on the path; bit-exact as UInt16 (probes) |
| `cb_tiles` → out DRAM | bf16 TILE pages (2048 B) | `noc_async_write` via `TensorAccessor(out)`, whole tiles (catalog `tile_reorder`) | NoC1 (writer default) |
| inter-core | — | none built | `index_broadcast` (deferred) would multicast `cb_index` chunks from one injector per column-piece group |

TARGET mapping (axes of `feature_spec.py`): `dtype` [bfloat16] — Phase 0. `layout` (index) — ROW_MAJOR Phase 0; TILE = `tile_index_fetch` deferred row (a reader-only change: different index-fetch block operation, same CBs, compute, writer — knob-turn class). `alignment` — both values Phase 0 (`valid_rows(t)` per tile row; no separate path). `output_layout` [TILE], `runtime_rows` [False] — Phase 0. TARGET has no `memory_layout` axis: no tensor of this contract is sharded, so no physical-shard scheme exists.

Structural impossibilities: none beyond `INVALID = []`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one unit = 1 tile row (32 tokens of one image) × `col_block_tiles` tile columns |
| Grid | full `compute_with_storage_grid_size()` (`C` cores); used cores = `min(C, U)` |
| Per-core work | contiguous unit range from `ttnn.split_work_to_cores(grid, U, row_wise=True)` (row-wise placement per catalog `noc_placement`); runtime args `unit_begin`, `num_units` |
| Kernel groups | one reader, one compute, one writer kernel descriptor, each over `all_cores` (≤ 2 rectangles: full rows + one partial row — BB-17); per-core counts are runtime args, never a second kernel group |
| Remainder | units: `split_work_to_cores` group 1/2; tiles: `St = ceil(S/32)` per image, `valid_rows(t) = min(32, S − (t mod St)·32)`; tile counts never use `floor`/`//` on `B·S` |
| Per-unit decode (in kernel) | `p = unit / T`, `t = unit mod T`, `b = t / St`, `tr = t mod St`, out tile base `t·Dt + p·ct`, slice offset `p·ct·64` B; divide once at kernel start, then advance incrementally |

### Host contract (registry, cache, plan)

| Item | Decision |
|------|----------|
| `INPUT_TAGGERS` | `alignment`: `"tile_aligned"` if `scenario["index_shape"][-1] % 32 == 0` else `"h_non_aligned"`; `layout`: `scenario["index_layout"]` `"ROW_MAJOR"→ttnn.ROW_MAJOR_LAYOUT`, `"TILE"→ttnn.TILE_LAYOUT`. `inputs = (scenario,)` with keys `index_shape`, `weight_shape`, `index_layout`, `memory_config` (`"dram_interleaved"` or `None`), built by `validate()` from the live tensors exactly as `eval/golden_tests/g6_embedding/axes.py:classify_call` does |
| Derived axes | `dtype = weight.dtype`, `output_layout = layout` kwarg, `runtime_rows = bool(runtime_rows)` |
| `validate()` order | (1) axes → SUPPORTED (`UnsupportedAxisValue`, message names the axis, e.g. `runtime_rows`), EXCLUSIONS (`ExcludedCell`, empty in Phase 0); (2) input requirements → `ValueError` (the index-dtype message contains `uint32`, the placement message contains `DRAM`): idx dtype ≠ uint32 or rank ∉ {1,2}; weight rank ∉ {2,3}, rank 3 with leading extent ≠ 1, or layout ≠ ROW_MAJOR; `D % 32 ≠ 0`; any tensor not on device or not DRAM interleaved; `memory_config` not `None`/DRAM interleaved; plan footprint > `L1_BUDGET` (message names 1347776) |
| Program-description cache (BB-07) | module-level dict keyed by `(idx logical shape, idx layout, weight shape, weight dtype, output layout, normalized memory_config, device id, grid size)`; value = CB descriptors, kernel descriptors (CT args, per-core RT args), plan. Per call only the three buffer addresses change: they are **common runtime args** of the reader (idx, W) and writer (out), set on the cached kernel descriptors before `ttnn.generic_op`. `generic_op` hashes arg counts, not values (`generic_op_device_operation.cpp:78-100`), and re-applies runtime args on a hit (`generic_op_spec_factory.cpp:44-49`). Nanobind vector accessors copy elements (`program_descriptors.cpp:627`): keep the Python `KernelDescriptor` objects themselves in the cache and assign `common_runtime_args` on them |
| Trace (BB-16) | the call allocates only its output (captured at trace time); no index value reaches host, program or args; replay reads the overwritten index buffer in place |
| Plan report (BB-08) | `describe_plan(input_tensor, weight, *, memory_config=None) -> dict` exported from `ttnn.operations.g6_embedding`, keys: `regime`, `col_block_tiles`, `col_pieces`, `block_tile_rows`, `tensor_tile_rows`, `total_units`, `num_cores`, `core_ranges`, `l1_bytes_per_core`; the entry point uses the same function, so bench and model get the same plan for the same shapes |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_index` | 0 | 128 B (`32 · 4`) | `index_depth · block_tile_rows` | spans tokens (`block_tile_rows`), streams columns, none on V; 2 blocks for prefetch | UInt32 (dataflow-only) | reader | reader (private scratch: write pointer used as a 2-slot ring, no push/pop) | whole kernel |
| `cb_rows` | 1 | 2048 B (`ttnn.tile_size(ttnn.uint16)`; 32 rows × 32 cols of row-major bytes) | `rows_depth · block_tile_rows · col_block_tiles` | spans tokens (`32·block_tile_rows` rows) and columns (`col_block_tiles`); streams V | **UInt16** (bit carrier for bf16 — Float16_b is forbidden, probe_001) | reader | compute | whole kernel |
| `cb_tiles` | 2 | 2048 B | `tiles_depth · col_block_tiles` | spans columns (`col_block_tiles`), streams tokens (one tile row per unit) | **UInt16** | compute | writer | whole kernel |

Compute config: `ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False)` (16-bit DEST, matches the probed `ttnn.tilize` uint16 configuration: no `UnpackToDest`). Every page count is derived from `block_tile_rows`, `col_block_tiles` and the depth knobs; none uses `T`, `N`, `V` or `Dt` unclamped.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `fetch_index_block` | `block_tile_rows` × ≤ 32 indices | no (raw NoC reads, DRAM source) | — (DRAM idx) | `cb_index`, slot `block_idx mod index_depth` (`block_tile_rows` pages) | slot holds the block's indices until `gather_rows_block(block_idx)` has issued its reads |
| 2 | `gather_rows_block` | `units_in_block` tile rows × `col_block_tiles·32` values | no (raw NoC reads by data-dependent page id) | `cb_index` slot (read via L1 pointer) | `cb_rows`, `units_in_block·col_block_tiles` pages pushed after one barrier | rows of padding positions (`r ≥ valid_rows`) are stale |
| 3 | `tilize_block` | `units_in_block` × (1 tile row × `col_block_tiles`) | yes — `compute_kernel_lib::tilize` | `cb_rows`, waits/pops `col_block_tiles` per unit | `cb_tiles`, `col_block_tiles` per unit | — |
| 4 | `store_tiles_block` | `units_in_block` × `col_block_tiles` tiles | no (raw NoC writes; dataflow) | `cb_tiles`, waits/pops `col_block_tiles` per unit | — (DRAM out) | all writes complete at the kernel-end barrier |

Push/wait agreement: `cb_rows` pushes `Σ units·ct` = compute pops `units_this_core·ct`; capacity `rows_depth·bt·ct` is a multiple of both quanta. `cb_tiles` pushes/pops `ct` per unit, capacity `tiles_depth·ct`.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `tilize_block` | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:194-207` (impl `tilize_helpers.inl:76-240`) | `<block_width_tiles = col_block_tiles, cb_rows, cb_tiles, InitUninitMode (InitAndUninit for one call per core), WaitMode::WaitBlock (or NoWait + explicit zoned wait), ReconfigureRegisterDatatypeMode::NoReconfigure>`, `num_blocks = units_this_core` (symmetric pages, no `total_input_pages`) | `cb_rows` | `cb_tiles` | `block_width_tiles` = `col_block_tiles`; `num_blocks` = this core's units |
| `tilize_block` (prereq) | raw_api | `compute_kernel_hw_startup(icb, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | `(cb_rows, cb_tiles)` | — | — | — |
| `fetch_index_block`, `gather_rows_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | idx: `(b, tr·128)`; W: `(min(idx, V−1), p·ct·64)` | — | — | offset uses `col_block_tiles` |
| `fetch_index_block`, `gather_rows_block` | raw_api | `noc_async_read` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | idx: `valid_rows·4` B; W: `col_block_tiles·64` B | — | `cb_index`, `cb_rows` | size = `col_block_tiles·64` |
| `gather_rows_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per block | — | — | `block_tile_rows` sets its frequency |
| `store_tiles_block` | raw_api | `noc_async_write`, `noc_async_writes_flushed`, `noc_async_write_barrier` | `dataflow_api.h:876`, `:1980`, `:1951` | page `t·Dt + p·ct + j`, 2048 B; flushed per unit; barrier once at end | `cb_tiles` | — | `col_block_tiles` writes per unit |

Helpers considered and rejected for the raw_api rows:

| Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|----------------------------------|
| `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-93` | reads sticks at consecutive page ids `start_page + k`; the gather needs page id `idx[k]` read from L1 per stick, which its `(accessor, total_num_rows, row_bytes, start_page, byte_offset_within_page)` signature cannot carry |
| `dataflow_kernel_lib::local_addr` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57-133` | self-aimed L1→L1 copies only; the sources here are DRAM pages (it would serve only the rejected `dm_face_permute` route) |
| `dataflow_kernel_lib::write_sticks_after_untilize` | `tilize_helpers_dataflow.hpp:129-135` | writes row-major sticks; the output is TILE pages |
| `ttnn.Mcast` / `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast.hpp`, `mcast/kernel/mcast_args.hpp:117` | no cross-core transfer in the built regimes; reserved for the deferred `index_broadcast` row |
| `set_read_trid` / `async_read_barrier_with_trid` | `local_copy_helpers_dataflow.hpp:147,158` | not needed by the Phase 0 one-barrier-per-block gather; named by the reader-barrier perf lamp |

No compute phase other than tilize exists; no helper covers an index-driven gather (`tensix.md#gaps`).

### Catalog entries that set knobs

| Entry | Knob it informed |
|-------|------------------|
| `double_buffer` | `rows_depth = tiles_depth = 2`; reads batched per block, one barrier per block |
| `width_split` | `col_block_tiles < Dt` for one-tile-row / few-tile-row results (`token_column_split`) |
| `noc_placement` | `split_work_to_cores(..., row_wise=True)`; reads on NoC0, writes on NoC1 (default configs) |
| `tile_reorder` | whole 2-KB tile writes, never face writes |
| `compute_block_size` | tilize init/uninit once per kernel, not per block |
| `trid_double_issue` | reader-barrier lamp |
| `bank_stagger` | write bank alignment lamp |
| `shared_input_reuse`, `mcast_topology` | `index_broadcast` deferred row |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Special values destroyed by the engine | the only engine on the path is tilize; Float16_b CBs flush subnormals, drop -0.0's sign, turn NaN into inf (probe_001); random test data never shows it | `cb_rows`/`cb_tiles` typed **UInt16** with 16-bit DEST; acceptance test grades special rows bit for bit on both Qwen tables |
| Index chunk/row slice misaligned (Blackhole mod-64 DRAM read rule) | 32-B face halves or 16-column slices land at 32 mod 64 | slices are whole tile columns (`ct·64` B), chunks 128 B; CB bases are DRAM-aligned |
| Regime flips with grid size | `T = 121, 128` (p4, `[1,4096]`) are `token_split` at C = 110 and `token_column_split` at C ≥ 130 | regime-pinned tests use `T = 2048` and `T = 1`; `describe_plan` reports the regime (BB-08) |
| Image boundary with `S % 32 ≠ 0` and `B > 1` | a block's tile rows may belong to different images; the index page and `valid_rows` change per tile row | per-tile-row decode `(b, tr, valid_rows)`; never treat `B·S` as one token stream |
| Write completion before return (BB-03) | per-unit `writes_flushed` does not wait for DRAM completion | one `noc_async_write_barrier()` at the end of the writer; reader never returns with a prefetch outstanding (the last block issues no prefetch) |
| Stale descriptor replays a wrong program (BB-07) | the descriptor is cached per signature; addresses differ per call | addresses are common RT args rewritten every call; the cache key holds every shape/dtype/layout/memory/grid attribute, never an address |
| Out-of-contract index value | a garbage index (e.g. TILE padding in the refinement) would address memory far past the table | Phase 0 reads only logical positions; the reader clamps the page id to `V − 1` |
| p1 perf target (< 765 µs, ≤ 1.051× of the 728 µs DRAM roofline) | the call moves 335.8 MB; any per-core bubble or bank hot spot is visible | minimum-traffic `token_split`, full grid, double-buffered CBs, flushed (not barriered) writes; lamps name the next levers (trid, bank stagger, overlap) |
| `B = 1` index page on one DRAM bank (RB-010, rule 8) | every core's first read hits that bank before any bulk read starts | Phase 0 reads each byte once (no duplicates under `token_split`); `index_broadcast` row records the multicast with its trigger |

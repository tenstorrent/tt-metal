# Operation Design: g6_reshard

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (L1 → L1, whole tiles, no compute kernel) |
| Goal | Return a new BLOCK_SHARDED L1 tensor in the placement `memory_config` names whose every tile is the tile of `input_tensor` at the same tile index, moved as stored (bfloat8_b shared exponents included), in one device program with no staging buffer. |
| Math | `out[i] = in0[i]` for every logical index `i`; `out` held in placement `P_out`, `in0` unchanged (`docs/explorations/g6_reshard/math.md` M1, M2) |
| References | `docs/explorations/g6_reshard/{DOSSIER,math,tensix,risks}.md`; `eval/golden_tests/g6_reshard/{feature_spec,helpers}.py`; `tt_metal/impl/buffers/buffer_distribution_spec.cpp` (shard page layout and block → core order); `ttnn/ttnn/operations/examples/master.md` (catalog entries listed under Catalog Evidence) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | on device; BLOCK_SHARDED in L1; rank 4; 2-D view `H`, `W` multiples of 32; SUPPORTED: bfloat8_b, TILE_LAYOUT | — | placement → CT args; buffer address → common RT arg |
| `memory_config` | `ttnn.MemoryConfig` | yes, positional | BLOCK_SHARDED in L1 with a shard spec; shard dims multiples of 32; grid one rectangle inside the device worker grid with enough core rows/columns for out's block rows/columns in its orientation | — | placement → CT args; out buffer address → common RT arg |

No dtype, output-tensor or compute-config argument exists (the op allocates `out`, takes `in0`'s dtype, does no arithmetic).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 4 `(d0, d1, d2, d3)`; production `(1, 1, 2048, 1024)`; the 2-D view folds all dims but the last into `H` |
| Dtype | bfloat8_b (Phase 0 SUPPORTED and TARGET) |
| Layout | TILE |
| Memory | BLOCK_SHARDED, L1; placement `P_in` (grid rectangle, shard shape, orientation) read from `input_tensor.memory_config().shard_spec`; production grid (0,0)-(10,9), shard (224, 96), ROW_MAJOR (7 × 3 tiles; last block row 1 tile high, last block column 2 tiles wide) |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` (padded shape identical) |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` |
| Memory | exactly `memory_config` (BLOCK_SHARDED, L1, its grid / shard shape / orientation); a fresh allocation (`ttnn.allocate_tensor_on_device`), never a view of `in0`; production grid (0,0)-(7,7), shard (256, 128), ROW_MAJOR (8 × 4 tiles) |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Both tensors are **physical** shards: `P_in`
pins where every source tile is, `P_out` pins where every target tile must land. The free
decisions are (a) which cores/RISCs **issue** each move, (b) how an issuer's share is cut into
blocks, and (c) how a block is cut into NoC transfers. All three are fixed here.

Vocabulary (scopes per `blocking-model.md` §3):

| Name | Scope | Definition |
|------|-------|------------|
| `tensor_rows_tiles`, `tensor_cols_tiles` | tensor | tile rows / columns of the padded 2-D view: `prod(padded_shape[:-2]) * padded_shape[-2] / 32`, `padded_shape[-1] / 32` (per-image tile padding is already in `padded_shape`) |
| `in_shard_rows_tiles`, `in_shard_cols_tiles` | in0 shard (allocation) | `P_in` shard shape / 32 — also the **page strides** inside every in0 shard, tail shards included |
| `out_shard_rows_tiles`, `out_shard_cols_tiles` | out shard (allocation) | `P_out` shard shape / 32 — page strides inside every out shard |
| `in_num_block_rows/cols`, `out_num_block_rows/cols` | tensor | `ceil(tensor_rows_tiles / *_shard_rows_tiles)`, `ceil(tensor_cols_tiles / *_shard_cols_tiles)` |
| `core_rows_tiles`, `core_cols_tiles` | core | the tiles of the out block this core holds: `min(out_shard_rows_tiles, tensor_rows_tiles - r_out*out_shard_rows_tiles)`, same for columns (cut short only on the last block row/column) |
| `issuers_per_core` | core | data-movement RISCs on a target core that issue reads |
| `block_rows_tiles`, `block_cols_tiles` | block | extent knobs: rows / columns of one issuer block |
| `num_blocks_this_issuer`, `block_idx` | issuer | `ceil(issuer_rows_tiles / block_rows_tiles)`; 1 or 0 in Phase 0 |
| piece | inside a block | the rectangle where a block meets one in0 block (one source core) |
| run | inside a piece | one NoC read: tiles contiguous in the source shard **and** in the target shard |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_row` (folded `d0·d1·d2` / 32; 64 in production) | independent — `out[h, w] = in0[h, w]` alone, no result spans rows | `block_rows_tiles` | `ceil(out_shard_rows_tiles / issuers_per_core)` = the issuer's whole share of the out shard (4 in P), so `num_blocks_this_issuer` = 1 | host, derived once from `out_shard_rows_tiles` and `ISSUERS_PER_CORE`; passed as one CT arg | out block rows pinned by `P_out` to target cores; inside a target core the rows are cut into `issuers_per_core` contiguous ranges, issuer `k` owns local rows `[k*block_rows_tiles, min((k+1)*block_rows_tiles, core_rows_tiles))` | knob-turn |
| `tile_col` (`d3` / 32; 32 in production) | independent — same clause | `block_cols_tiles` | `out_shard_cols_tiles` (whole shard width; 4 in P); clamped per core to `core_cols_tiles` | host, = `out_shard_cols_tiles`; one CT arg | out block columns pinned by `P_out`; **not** split across issuers: a run is a row of a piece, so a column split would cut runs and add transfers, a row split never does | knob-turn |

No axis is dependent (no combine exists anywhere) and none is reuse-shared: operand-reuse check
for the chosen split (out blocks across target cores) — in0 varies along it (every target needs
different tiles; no tile goes to two cores), so no broadcast row exists.

Intermediate data: none. The scheme introduces no stage between source shard and target shard,
so there is no second table to re-run. No core collects, coordinates or hands out work.

Stall-shadow check: the only wait is each issuer's final read barrier. Every read of the issuer is
issued before it and nothing else remains to schedule into it; the wait is the program's last act.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| — (no CB exists) | NoC reads in flight per issuer: `runs_in_flight` | every run of the block (no software cap; command-buffer back-pressure only) | all transfers of a block overlap one another; one barrier per issuer instead of one per run (catalog `double_buffer`, `trid_double_issue`: batch reads, barrier once) |

There is no CB: reads land directly in out's shard (ledger Rule 3, pattern 1 taken to its limit —
the destination **is** the tensor). A CB would be a staging copy, forbidden by rule 1.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Shard page layout: a cut-short tail shard keeps the **full** shard row stride (`buffer_distribution_spec.cpp:41-51` uses `shard_strides` of the full shard shape; `:80-87` only narrows the iterated size) | row stride of every source and target address | source stride = `in_shard_cols_tiles * page_bytes`, target stride = `out_shard_cols_tiles * page_bytes`, never the block's actual width | tail blocks (P: last in0 block row and column) read or write wrong tiles; wrong values, no error |
| Run contiguity | `run_tiles` | a piece is **one** run only if `piece_rows == 1` or (`piece_cols == in_shard_cols_tiles` and `piece_cols == out_shard_cols_tiles`); otherwise one run per piece row of `piece_cols` tiles | coalescing across rows of a narrower piece reads tiles from the next shard row: wrong values |
| Tensor edge | `core_rows_tiles`, `core_cols_tiles`, piece extents | clamp every range to `[0, tensor_rows_tiles)` × `[0, tensor_cols_tiles)` | reads of unused tail slots and writes past the out block's valid tiles (harmless for values but breaks "every tile written once" accounting and can read beyond the source shard) |
| `noc_async_read` multi-packet split (`dataflow_api.h:547`, default `max_page_size = NOC_MAX_BURST_SIZE + 1`) | `run_bytes` | none needed: the call splits into packets of `NOC_MAX_BURST_SIZE` (16384 B BH, `noc_parameters.h:287-290`) | — ; only the stateful one-packet perf lamp must clamp runs to `NOC_MAX_BURST_SIZE` (I0/I3/I5 runs are 16 tiles = 17408 B) |
| L1 NoC alignment: source and destination offsets equal mod 16 (`.claude/references/noc-alignment-in-tt-metal.md:43`) | `page_bytes` | host asserts `page_bytes % ttnn.get_l1_alignment() == 0` and in0/out page sizes equal; every address is `base + k*page_bytes` | silent corruption (watcher stops the core) |
| Virtual NoC coordinates are separable (x from logical x, y from logical y) and identical on NOC0/NOC1 (`risc_common.h:23`, BH `noc_nonblocking_api.h:46`: `NOC_0_X` is the identity) | NoC coordinate tables | host asserts, for every in0 block core, `worker_core_from_logical_core(core) == (table_x, table_y)` | reads from wrong cores |
| Common RT arg count | NoC tables: `in_num_block_rows + in_num_block_cols` words | bounded by the device worker grid (`≤ grid_x + grid_y`, 21 in P) | none at any legal placement — the in-kernel piece walk keeps arg count independent of the number of runs (BB-04) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **PULL** — every core holding an out block reads its tiles from the cores holding them, straight into its own out shard, on `issuers_per_core` RISCs | **built** | every legal pair of block placements (the only built regime: selection is not shape- or L1-dependent, so no plan switch exists) | `block_rows_tiles × block_cols_tiles` = the issuer's whole share of its out shard (P: 4 × 4 tiles, 8 runs) | **minimum**: boundary = source L1 → target L1; every tile crosses it exactly once (P: 1880 tiles / 2045440 B cross-core, 168 tiles / 182784 B core-local loopback), 0 DRAM bytes, no staging | the block's fixed cost is one `noc_async_read_barrier` (a full drain of the issuer's NoC) plus per-piece address setup; one block per issuer pays the barrier once per issuer per program (intended frequency: 1); a finer block only adds barriers — there is no consumer to overlap with |
| **PUSH** — every core holding an in0 block writes its tiles into the target cores' out shards | deferred — the same tiles land on the same 64 target cores, so the per-target landing volume (32 tiles, 34816 B in P) that bounds both regimes is unchanged; push's only gain is more issuers (110 vs 64 cores in P), and PULL already halves per-RISC issue to 8 runs; push also needs write-ack completion on 110 cores. Reachable: the piece walk is symmetric (walk the out blocks overlapping my in0 block) — same kernel source, role CT flag, kernel group = in0's trimmed rectangle | — | the in0 shard of a source core | minimum (same crossings); adds write acks | one write barrier per issuer |
| **BOTH-ENDS** — each run issued by its source or its target core, whichever has fewer runs | deferred — balances issue over the union of both grids (≤ 4.7 runs per RISC in P vs 8), landing bound unchanged; costs ≥ 3 kernel core ranges in P (the out rectangle plus the L-shaped rest of in0's grid, BB-17) and mixed read/write completion. Reachable: PULL's issuer kernel plus PUSH's kernel, each walking a host-assigned subset of pieces | — | issuer's assigned pieces | minimum | as PULL / PUSH |
| **CB-STAGED** — read into a scratch CB on the target, then copy into out | rejected — superseded by PULL (reads land in out directly); a staging copy is forbidden by rule 1 and costs a second local pass of 2228224 B | — | — | 2× (every tile crosses into L1 twice) | — |
| **DRAM / interleaved intermediate** | rejected — rule 1 (no DRAM staging), BB-01 (two programs); dead end | — | — | in0 crosses DRAM twice (4456448 B) + a second program | — |
| **THIRD-CORE RELAY** | rejected — rule 1 (no buffer on a core holding neither end of a tile); adds a hop | — | — | 2× cross-core | — |
| **COMPUTE PATH** (unpack → pack) | rejected — rule 1 (no unpack/convert/repack; bfloat8_b regrouping risk R2); dead end | — | — | + an engine pass per tile | — |
| **SINGLE ISSUER CORE** | rejected — X3; one core issues all 1024 runs of P; superseded by PULL on every target core | — | — | minimum bytes, all issue serialized | — |

There is no dependent axis, so "split the dependent axis and combine" does not exist for this op.

### Traffic ranking

| Candidate split (who issues) | DRAM crossings | Cross-core bytes (P) | Core-local bytes (P) | Fan-out / fan-in | Rank |
|------------------------------|----------------|----------------------|----------------------|------------------|------|
| PULL (target cores) | 0 | 2045440 | 182784 (loopback) | each target reads ≤ 4 sources (P), each source read by ≤ 4 targets | 1 (tie on bytes) — chosen |
| PUSH (source cores) | 0 | 2045440 | 182784 | mirror | 1 (tie) |
| BOTH-ENDS | 0 | 2045440 | 182784 | mixed | 1 (tie) |
| CB-STAGED / THIRD-CORE | 0 | ≥ 2045440 + a second pass | 2228224 extra | — | 4 |
| DRAM intermediate | 2 (write + read of 2228224 B) | 4456448 via DRAM | — | — | 5 |

The three issuer splits tie on every tier. Tiebreaker for PULL: each core writes only its own L1
and reads an immutable input, so completion is a local read barrier with no handshake (catalog
`tensix_all_reduce`: "Prefer pull", push adds a `SEMAPHORE_INC`/ack census), and the kernel group is
out's trimmed rectangle — one core range (BB-17).

Run counts for the suite (PULL, `issuers_per_core` = 2): P 1024 runs of ≤ 3 tiles (8 per issuer);
I0 / I3 / I5 128 runs of 16 tiles (1 per issuer, shard widths match so pieces coalesce); O1 960 runs
of ≤ 3 tiles over 56 target cores.

### Block schedule

Each issuer kernel (one per data-movement RISC of a target core) runs:

```cpp
locate_block();                                    // my out block (r_out, c_out) and my row range
for (uint32_t block_idx = 0; block_idx < num_blocks_this_issuer; ++block_idx) {   // Phase 0: 0 or 1
    gather_block(block_idx);                       // every run of every piece → out shard, async
}
land_blocks();                                     // one noc_async_read_barrier on this issuer's NoC
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `locate_block` | — | — | once per issuer kernel: derive `(r_out, c_out)` from `get_absolute_logical_x/y()` and `P_out` (ROW_MAJOR: `c_out = x - out_x0`, `r_out = y - out_y0`; COL_MAJOR: `r_out = x - out_x0`, `c_out = y - out_y0`); `core_rows_tiles`, `core_cols_tiles`; the issuer's row range; one integer division per axis to find the first overlapping in0 block |
| `gather_block` | `block_rows_tiles × block_cols_tiles` (clamped to the core's valid tiles) | in0 shards on their source cores (the model's), out shard on this core | per piece (in0 block met): one source-core lookup from the NoC tables and one source/target base address; per run: one `noc_async_read`; no inits, no barriers, no semaphores |
| `land_blocks` | — | out shard | once per issuer kernel: one `noc_async_read_barrier()`; it is also the BB-03 completion point (reads only — no writes, atomics or semaphores are ever issued) |

Piece walk inside `gather_block` (the contract the implementer realizes):

| Step | Rule |
|------|------|
| Block rows (tensor coords) | `row_lo = r_out*out_shard_rows_tiles + k*block_rows_tiles`, `row_hi = min(row_lo + block_rows_tiles, r_out*out_shard_rows_tiles + core_rows_tiles)` |
| Block cols | `col_lo = c_out*out_shard_cols_tiles`, `col_hi = col_lo + core_cols_tiles` |
| In0 block rows met | from `br = row_lo / in_shard_rows_tiles` while `br*in_shard_rows_tiles < row_hi`; piece rows `[max(row_lo, br*in_shard_rows_tiles), min(row_hi, (br+1)*in_shard_rows_tiles))` — `(br+1)*in_shard_rows_tiles` may exceed `tensor_rows_tiles` only past `row_hi`, which is already clamped |
| In0 block cols met | same with `bc`, `in_shard_cols_tiles`, `[col_lo, col_hi)` |
| Source core | ROW_MAJOR in0: `noc_x = in_block_col_noc[bc]`, `noc_y = in_block_row_noc[br]`; COL_MAJOR in0: `noc_x = in_block_row_noc[br]`, `noc_y = in_block_col_noc[bc]` |
| Source address of piece tile `(tr, tc)` | `in_addr + ((tr - br*in_shard_rows_tiles)*in_shard_cols_tiles + (tc - bc*in_shard_cols_tiles))*page_bytes` |
| Target address of tile `(tr, tc)` | `out_addr + ((tr - r_out*out_shard_rows_tiles)*out_shard_cols_tiles + (tc - c_out*out_shard_cols_tiles))*page_bytes` |
| Runs | one run of `piece_rows*piece_cols` tiles when the contiguity cap allows; else `piece_rows` runs of `piece_cols` tiles, source += `in_shard_cols_tiles*page_bytes`, target += `out_shard_cols_tiles*page_bytes` per run |

Every tile of the out block is covered exactly once: the in0 blocks partition the tile grid and
the issuers' row ranges are disjoint, so "every tile of out written once" holds by construction.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Issuers per core (default 2, NCRISC on NOC0 + BRISC on NOC1) | catalog `split_reader`: a second issuer pays only when one RISC is issue-bound; P's 8 runs of 3264 B per issuer may already be landing-bound | `ISSUERS_PER_CORE = 1` (one kernel, NOC0) on P and I0 |
| NoC per issuer (default: issuer 0 NOC0, issuer 1 NOC1) | two NoCs give two landing paths, but NOC1's routing direction may contend differently on this source→target pattern (catalog `noc_placement`) | both issuers on NOC0 / both on NOC1 |
| Per-run issue path (default plain `noc_async_read`, `dataflow_api.h:547`) | P's runs are ≤ 3264 B and each piece's runs share source core and size, so per-run register programming may dominate (risks.md R3) | per piece `noc_async_read_one_packet_set_state` once + `noc_async_read_one_packet_with_state` per run (`dataflow_api.h:575`, `:599`), runs clamped to `NOC_MAX_BURST_SIZE` |
| Issue order (default: every issuer walks its pieces top-left first) | up to 8 issuers hit the same source core at once at program start (each in0 block feeds ≤ 4 targets × 2 issuers) | start each issuer at piece `(r_out + c_out) % pieces` and wrap (catalog `bank_stagger`, applied to source cores) |

Overlap lamp: does not apply — there is no consumer of out inside the program, so a large block
serializes nothing. Grid-synchronization lamp: does not apply — no multicast, semaphore, cross-core
barrier or combine exists.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| in0 shard on source core | bfloat8_b tiles, 1088 B each, stored as written by the producer op | resident in L1 before the program (the model's tensor) | never modified; read only |
| source L1 → target L1 | the same bytes | `noc_async_read` issued by the target core's issuer RISC (NCRISC/NOC0, BRISC/NOC1); source NoC address `get_noc_addr(noc_x, noc_y, src_l1)` (`dataflow_api_addrgen.h:209`) | whole tiles, runs of contiguous tiles; tiles whose source core is the target core use the same call (NoC loopback) — the copy into a new tensor is required |
| out shard on target core | identical bytes at the out page slot | written once per tile; completion by `noc_async_read_barrier` | no CB, no compute kernel, no unpack/pack: no conversion is possible on this path |

Placement as a TARGET axis: TARGET carries no `memory_layout` axis — both placements are arguments
of every cell. Each physical shard side is consumed in place: out's shard is the landing buffer
itself; in0's shard is the source of the move (reading it over the NoC is the move, not a re-read
of data already resident on the reading core). The deferred PUSH / BOTH-ENDS regimes need no
inter-Tensix handshake beyond write barriers (no consumer inside the program waits on out).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one out block (a target core's whole shard), cut by tile rows into `issuers_per_core` issuer blocks |
| Grid | out's **trimmed** rectangle: from `(out_x0, out_y0)` with `out_num_block_cols` × `out_num_block_rows` cores (ROW_MAJOR) or `out_num_block_rows` × `out_num_block_cols` (COL_MAJOR) — one `CoreRange`; cores of `memory_config`'s grid beyond it hold no block and run nothing. P: (0,0)-(7,7), 64 cores; O1: (1,0)-(8,6), 56 cores |
| Per-core work | `core_rows_tiles × core_cols_tiles` tiles (P: 8 × 4 = 32, 16 runs) |
| Per-issuer work | rows `[k*block_rows_tiles, min((k+1)*block_rows_tiles, core_rows_tiles))` × all `core_cols_tiles`; an empty range issues nothing and its barrier returns at once |
| Remainder | tail out blocks: `core_rows_tiles = min(out_shard_rows_tiles, tensor_rows_tiles - r_out*out_shard_rows_tiles)` (and columns) from `tensor_rows_tiles` computed per image from `padded_shape` (`ceil`), never `floor(H/32)` of a flattened logical shape |
| Kernels | two `KernelDescriptor`s, same source `kernels/g6_reshard_gather.cpp`, both on the trimmed rectangle: issuer 0 `ReaderConfigDescriptor()` (RISCV_1, NOC0), issuer 1 `WriterConfigDescriptor()` (RISCV_0, NOC1); with `ISSUERS_PER_CORE = 1` only issuer 0 exists. No compute kernel, no CBs, no semaphores |
| Per-core identity | derived in-kernel from `get_absolute_logical_x()/y()` (`dataflow_api.h:44`, `:56`); **no unique runtime args** — everything per core is a function of CT args |

Single built regime: no regime-selection function exists; the regime is the same for every
placement pair. The plan dump (Host Contract) still reports it.

### Arguments the implementer threads (single sources)

| Value | Source | Delivered as |
|-------|--------|--------------|
| `ISSUERS_PER_CORE` | host module constant (Phase 0: 2) | number of issuer kernels; CT `issuers_per_core`, CT `issuer_index` per kernel |
| `block_rows_tiles` | host: `ceil(out_shard_rows_tiles / ISSUERS_PER_CORE)` | CT |
| `block_cols_tiles` | host: `out_shard_cols_tiles` | CT |
| `page_bytes` | `input_tensor.buffer_aligned_page_size()` (asserted equal to out's) | CT |
| `tensor_rows_tiles`, `tensor_cols_tiles` | host from `padded_shape` | CT |
| `P_in`: `in_shard_rows_tiles`, `in_shard_cols_tiles`, `in_col_major`, `in_num_block_rows`, `in_num_block_cols` | `input_tensor.memory_config().shard_spec` | CT |
| `P_out`: `out_shard_rows_tiles`, `out_shard_cols_tiles`, `out_col_major`, `out_x0`, `out_y0` | `memory_config.shard_spec` | CT |
| `in_addr`, `out_addr` | `input_tensor.buffer_address()`, `out.buffer_address()` (lockstep: same on every core of each shard grid) | common RT args 0, 1 — the only values patched per call |
| `in_block_row_noc[in_num_block_rows]`, `in_block_col_noc[in_num_block_cols]` | host via `device.worker_core_from_logical_core` (separability asserted) | common RT args after the addresses |

Placement values are CT args on purpose: they define the program, so they enter generic_op's
program hash (`generic_op_device_operation.cpp:78-100` hashes CT args, core ranges and RT-arg
counts, not RT values) and the kernel's loop bounds fold to constants.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| — none — | — | — | — | Reads land in out's shard directly; no phase boundary exists between source and target, so no buffer exists between them (ledger Rule 3 pattern 1). The op's only L1 allocation is out's shard, accounted in `l1_ledger.md`. | — | — | — | — |

CB sync / ownership rules are vacuous (zero CBs). out's shard is written by the issuer RISCs of
its own core on **disjoint** page ranges and read by no kernel of this program.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input (semantic name, pages, state) | Output (semantic name, pages) | State after |
|---|-----------------|-------------|---------|-------------------------------------|-------------------------------|-------------|
| 1 | `locate_block` | — | no (integer arithmetic on CT args) | CT placement args, logical coords | issuer row range, first in0 block `(br, bc)` | — |
| 2 | `gather_block` | `block_rows_tiles × block_cols_tiles` | no (raw NoC reads; see API Mapping) | `in0_shard` on source cores, by NoC address; all pages the block meets; untouched | `out_shard` on this core, the block's pages | reads in flight |
| 3 | `land_blocks` | — | no | — | `out_shard` | every page of the issuer's rows landed; no NoC transaction outstanding (BB-03) |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input | Output | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|-------|--------|------------------------------|
| `locate_block` | raw_api | `get_absolute_logical_x()`, `get_absolute_logical_y()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:44`, `:56` | — | — | — | — |
| all | raw_api | `get_compile_time_arg_val(i)` | `tt_metal/hw/inc/api/compile_time_args.h:64` | placement + knob CT args | — | — | `block_rows_tiles`, `block_cols_tiles`, `issuers_per_core` |
| all | raw_api | `get_common_arg_val<uint32_t>(i)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:168` | `in_addr`, `out_addr`, NoC tables (runtime-indexed) | — | — | — |
| `gather_block` | raw_api | `get_noc_addr(noc_x, noc_y, addr)` | `tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:209` | source virtual coords + source L1 address | `in0_shard` | — | — |
| `gather_block` | raw_api | `noc_async_read(src_noc_addr, dst_l1, run_bytes)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | default `max_page_size` (multi-packet capable) | `in0_shard` | `out_shard` | `run_bytes` derives from the piece, bounded by `block_*_tiles` |
| `land_blocks` | raw_api | `noc_async_read_barrier()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1921` | issuer's own NoC | — | `out_shard` | — |
| zones (BB-21) | raw_api | `DeviceZoneScopedN(name)` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` (compiles to nothing in the plain build, `:1047`) | `"G6-GATHER"` around `locate_block` + `gather_block`, `"G6-LAND"` around `land_blocks` | — | — | — |

Helpers considered and rejected (every raw entry above):

| Candidate | File:Line of the mismatch | Why it cannot express the shape |
|-----------|---------------------------|---------------------------------|
| `dataflow_kernel_lib::local_addr` / `set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.inl:30-34` | the source endpoint is hard-wired to `my_x[noc_id], my_y[noc_id]` (self-aimed); 1880 of P's 2048 tiles come from other cores |
| `McastArgs` sender/receiver | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:18-42` | moves one source to the same address on a group of cores; no tile of a reshard goes to two cores (operand-reuse check above) |
| `TensorAccessor` over in0 (`TensorAccessorArgs(input_tensor)`) | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` (`get_noc_addr(page_id)` → `get_bank_and_offset`, `:381`), `:184-200` (`num_contiguous_pages` stops at the shard edge along the contiguous dim) | address resolution per run instead of per piece, and runs never span rows, so I0/I3/I5 would issue 8 reads per issuer instead of 1; the piece walk needs the in0 block boundaries anyway |
| compute-kernel helpers (`copy`, `tilize`, `untilize`, `eltwise_chain`) | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp` (unpack → DEST → pack) | rule 1 forbids unpacking/repacking; there is no compute phase |

`compute_kernel_hw_startup()` is not applicable: the program has no compute kernel.

## Host Contract

| Item | Decision |
|------|----------|
| Registry | `INPUT_TAGGERS = {}` (no shape-derived axis; the placements are arguments); `SUPPORTED = {"dtype": [ttnn.bfloat8_b], "layout": [ttnn.TILE_LAYOUT]}`; `EXCLUSIONS = []`; no INVALID |
| Entry point order | `validate(input_tensor, memory_config)` as the first line (SUPPORTED → `UnsupportedAxisValue`, EXCLUSIONS → `ExcludedCell`), then argument checks (below), then allocate out, then descriptor (cached), then one `ttnn.generic_op([input_tensor, out], descriptor)` |
| Argument checks → `ValueError` | input not on device (`ttnn.is_tensor_storage_on_device`); input not BLOCK_SHARDED, not L1, or no shard spec; rank ≠ 4; logical 2-D view `H = prod(shape[:-1])` or `W = shape[-1]` not a multiple of 32 (tile geometry itself is taken from `padded_shape`); memory_config not BLOCK_SHARDED, not L1, or no shard spec; memory_config shard dims not multiples of 32; memory_config grid not one rectangle (bounding-box area ≠ `num_cores()`); grid beyond `device.compute_with_storage_grid_size()`; grid with fewer core rows / columns than out's block rows / columns in its orientation |
| L1 bound (BB-05) → `RuntimeError` | `out_shard_rows_tiles * out_shard_cols_tiles * page_bytes > 1347776` (the op adds no other L1); never reached by any suite placement (34816 B P, 43520 B O1) |
| No other refusal | every legal block placement on either side is accepted (rule 2): any origin, either orientation, any multiple-of-32 shard, cut-short tails |
| Descriptor cache (BB-07) | module dict keyed by `(device id, padded_shape, dtype, layout, P_in, P_out)` with `P = (grid bbox, shard shape, orientation)`; buffer addresses are **not** in the key. On a hit only common RT args 0, 1 (`in_addr`, `out_addr`) of each issuer kernel are rewritten (`program_descriptors.cpp:188-213` copies common RT args into the cached program) — the descriptor is never rebuilt |
| Plan / role dump (BB-08, BB-21) | `describe_plan(input_tensor, memory_config) -> dict` in the op module: `regime` (`"PULL"`), `issuers_per_core`, kernel core ranges, per target core its block `(r_out, c_out)`, per issuer its row range, NoC and run count, and every in0 core that holds no out block tagged `"source-only"`. Pure host arithmetic, no dispatch |
| Profiling (BB-21) | the two zones of API Mapping in every issuer kernel; nothing else differs between plain and zone builds |
| Program count (BB-01) | exactly one `ttnn.generic_op` per call; no host read of tensor values; no other ttnn op on in0 or out |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Tail shard stride (R1) | P's in0 has a 1-tile-high last block row and 2-tile-wide last block column; the shard keeps its full 3-tile row stride | mechanism cap 1: strides always from the shard shape; suite case P covers both tails, O1 covers an out tail column |
| Orientation / origin read wrongly (R1) | I0 and I3 differ only in in0's orientation; I5 has in0 off the origin; O1 has out off the origin and COL_MAJOR | `locate_block` and the source-core rule carry both orientations; NoC tables built from the tensor's own shard spec and asserted separable; all five placements graded bit-exact |
| Absolute logical coords ≠ shard-spec coords | a sub-device offset would shift every block | the suite's off-origin cases (I5, O1) catch it; the op runs on the default sub-device |
| Cache key misses a placement (R6) | P and O1 differ only in memory_config; I0 and I3 only in in0's orientation | `P_in` and `P_out` are in the Python key **and** in CT args (generic_op hash); `test_program_cache_placements` runs them in one process |
| Coalescing a narrower piece | a piece of a tail block narrower than the shard looks contiguous if the actual width is compared | contiguity cap compares against **shard** widths only |
| out aliasing in0 (R8) | I0 has identical placements | out always freshly allocated; acceptance test asserts distinct buffer and in0 unchanged |
| Transactions in flight at return (R7, BB-03) | another program starts right after | reads only; each issuer ends with `noc_async_read_barrier` on its own NoC; no writes, semaphores or atomics exist |
| Issuer split leaves one RISC idle | a 1-tile-high out shard gives issuer 1 no rows | correct by construction (empty range); production shards are 8 tiles high |

## Catalog Evidence

| Catalog entry | Knob it informed |
|---------------|------------------|
| `split_reader` | `issuers_per_core` = 2 default, with its perf lamp (pays only when issue-bound) |
| `double_buffer`, `trid_double_issue` | `runs_in_flight` = all runs of the block, one barrier per issuer (never read/barrier per run) |
| `read_coalescing`, `tile_reorder` | run = the widest contiguous span of whole tiles (whole piece when shard widths match) |
| `tensix_all_reduce` (push vs pull paragraph) | PULL chosen over PUSH on the tie (no handshake, reads an immutable input) |
| `noc_placement`, `bank_stagger` | NoC-per-issuer and issue-order perf lamps |

## Test Plan

Acceptance test: `tests/ttnn/unit_tests/operations/g6_reshard/test_g6_reshard.py` (immutable).
Per BB-22 it holds the production shape `(1, 1, 2048, 1024)` alone and varies **placements**, not
shapes: P, I0, I3, I5, O1 of `feature_spec.py`. Gate: bit-exact (`helpers.TOLERANCES` is
`("exact", None)`), comparing int16 views of the readback against in0 read back and against the
`torch.clone` reference; plus out's placement, in0 unchanged and not aliased, all placements in one
process with the program cache on, three repeat calls bit-identical, and the `## Validation` errors
(`expect_error((ValueError, RuntimeError), "")`, so a support refusal there fails the test). The
`H`/`W` multiple-of-32 refusal is specified in Host Contract but not exercised: the test builds no
tensor whose 2-D view is not tile-aligned.

Structural impossibilities: none beyond `feature_spec.py` (INVALID is empty and correct).

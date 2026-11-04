# Operation Design: nomic_moe_dispatch

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (gather + tilize; no arithmetic) |
| Goal | Build Xg = the dispatch table's rows picked by `gather_index`, tilized, for grouped_matmul_w1. One device program, index read on device, bit-exact. Replaces routed_combine's gather mode (8.97 / 13.03 / 21.85 us at R = 2272 / 4320 / 8416). The perf focus is R = 8416: at most 15.0 us to accept, 11.0 us stretch. |
| Math | `out[0, 0, r, :] = table[0, 0, gather_index[0, r], :]` for r in 0 .. R-1 |
| Mode | Derivative (incumbent: `ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/routed_combine/`, gather mode) |
| References | incumbent `device/routed_combine_program_factory.cpp`, `device/kernels/dataflow/combine_reader.cpp`, `read_batches.hpp`, `device/kernels/compute/combine.cpp:41-53`; `models/experimental/nomic_embed_text_v2_moe/docs/MOE_GLUE.md` §3-5, 13, 15; `ROUTED_COMBINE.md` §10-11; `BLACKHOLE.md` §1, 4, 6; `tt/experts.py` (`dispatch`, `routed_memory_config`); grouped_matmul_w1 `device/grouped_matmul_w1_device_operation.cpp:81-101` (any input memory layout) and `grouped_matmul_w1_program_factory.cpp:195-199` (an ND-sharded input is read in contiguous runs of `shard_shape[-1]/32` tiles); catalog `ttnn/ttnn/operations/examples/master.md` entries `split_reader`, `trid_double_issue`, `double_buffer`, `read_coalescing`, `noc_placement`, `bank_stagger` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| table | ttnn.Tensor | yes | (1, 1, T, 768) bf16 ROW_MAJOR, interleaved in L1, 1000 ≤ T ≤ 4096 | — | RT: buffer address (common RT arg) |
| gather_index | ttnn.Tensor | yes | (1, R) uint32 ROW_MAJOR, DRAM interleaved (one page), R % 32 == 0, 2240 ≤ R ≤ 8512, entries in [0, T) | — | RT: buffer address (common RT arg). **Values are never CT args or cached state.** |
| memory_config | ttnn.MemoryConfig \| None | no | `buffer_type` L1 (supported); DRAM is refused (see Regimes) | `ttnn.L1_MEMORY_CONFIG` | host only: picks `output_memory_config(R, device, buffer_type)` |
| compute_kernel_config | ttnn.ComputeConfigDescriptor \| None | no | any; values are unchanged by it. The fast path needs `dst_full_sync_en=False` | `default_compute_kernel_config()` | compute kernel config |

Module exports: `nomic_moe_dispatch`, `output_memory_config(rows, device, buffer_type=ttnn.BufferType.L1)`, `default_compute_kernel_config()`, `circular_buffer_bytes(rows)`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`.

| Registry item | Value |
|---|---|
| `INPUT_TAGGERS` | `{"window": tag_window}`. `tag_window(inputs, axes)` returns `"aligned"` if `inputs[0][-2] % 32 == 0`, else `"ragged"`. |
| `SUPPORTED` | `dtype: [ttnn.bfloat16]`, `table_buffer: ["l1"]`, `output_buffer: ["l1"]`, `window: ["aligned", "ragged"]` |
| `EXCLUSIONS` | `[]` |
| `validate()` | First line of `nomic_moe_dispatch`. It raises `SupportRefusal` (`UnsupportedAxisValue` for an axis value) unless every row of the Validation table holds. |
| `default_compute_kernel_config()` | `ttnn.ComputeConfigDescriptor(math_fidelity=LoFi, fp32_dest_acc_en=False, dst_full_sync_en=False, math_approx_mode=False)`. LoFi is irrelevant here: there is no math. |
| `circular_buffer_bytes(rows)` | `row_slots·w·2048 + 2·(max_blocks_per_core·64 + 64)`. The output-backed CB is the output tensor, so it is not counted. The value is ≤ 123,648 B at every R. |

| Validation (all must hold) | Refusal reason |
|---|---|
| table: rank 4, shape (1, 1, T, 768), bfloat16, ROW_MAJOR, `memory_layout == INTERLEAVED`, `buffer_type == L1`, 1000 ≤ T ≤ 4096 | outside SUPPORTED (another width, dtype or T range; DRAM table, see Regimes R4) |
| gather_index: shape (1, R), uint32, ROW_MAJOR, INTERLEAVED, R % 32 == 0, 2240 ≤ R ≤ 8512 | outside the op's range |
| `memory_config.buffer_type == L1` | output DRAM is not built (R3) |
| `device.compute_with_storage_grid_size() == (11, 10)` and the L1 bank count == 110 | the bank divisor, the round robin and the plan are compiled for 110 |
| `circular_buffer_bytes(R) ≤ 256 KiB` | the plan cannot fit. This never triggers in range; the check stays. |

## Tensors

### Input

| Property | table | gather_index |
|----------|-------|--------------|
| Shape | (1, 1, T, 768) | (1, R) |
| Dtype | bfloat16 | uint32 |
| Layout | ROW_MAJOR (page t = token t's 1536 B row) | ROW_MAJOR (one page of R·4 B) |
| Memory | L1 interleaved, page t on bank t mod 110 | DRAM interleaved, one page, so one DRAM bank |

### Output

| Property | Value |
|----------|-------|
| Shape | (1, 1, R, 768) |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | `output_memory_config(R, device, L1)`, defined below |

**`output_memory_config(rows, device, buffer_type)`**

- **L1:** `ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(shard_shape=[32, 32·w], grid=CoreRangeSet{(0,0)-(10,9)}, orientation=ROW_MAJOR, shard_distribution_strategy=ROUND_ROBIN_1D))`.
  - `w = block_width_tiles(rows)`, defined in the Blocking Model.
  - Shard s = tile_row·k + piece goes to core `s mod 110`, in local slot `s div 110`.
  - Tiles are row-major inside a shard (`tt_metal/impl/buffers/buffer_distribution_spec.cpp:54-87`, `:63-64`, `:207-214`).
- **DRAM:** `ttnn.DRAM_MEMORY_CONFIG`. That buffer type is refused by `validate()`.
- **Canonical form:** return the memory config that a tensor allocated from this spec actually reports, i.e. `TensorSpec(...).memory_config()`. A tensor allocated from an ND config can be rewritten to a legacy layout, so it may compare unequal to the config it was made from (Risks). Memoize per `(rows, buffer_type)`.

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. The op is a pure gather plus tilize. Every axis is independent or reuse-shared, and none is dependent: nothing combines across blocks.

The core assignment is **set by the output layout**: a core tilizes exactly the shards it holds, so every output byte is packed into the core's own L1 and no output write crosses the NoC. The output layout is in turn bounded by the model's L1 plan. A core's share may exceed the interleaved share `ceil(24·tile_rows/110)` tiles by at most 4 tiles (8 KB). That bound selects the column extent per R.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_row` (output rows, `tensor_tile_rows = R/32` blocks of 32 rows) | independent: each output row is a copy that depends on no other row | `block_tile_rows` (the shard height in tile rows) | **1.** This is the minimum on purpose: the per-core share is `ceil(shards/110)·shard_volume`. 2-tile-row shards break the +8 KB budget, e.g. 72 > 62 tiles at R = 8416. Coarseness on this axis comes from residency instead: all of a core's blocks are resident at once (`row_slots`). | host plan constant `SHARD_TILE_ROWS = 1` → `shard_shape[0]` | round robin: shard s → core `s mod num_cores`. Core c owns `s = c + num_cores·j` for j = 0 .. `num_blocks_this_core − 1`. Phase 0 spreads over all 110 cores. | knob-turn (fixed by the L1 budget) |
| `column` (hidden, `tensor_col_tiles = 24`) | independent: tile columns of a row copy independently | `block_width_tiles` w = 24 / `pieces_per_row` (k) | Selection rule (below): w = 8 at R = 2272 and 4320, w = 12 at R = 8416; w ∈ {4, 6, 8, 12, 24} over the range | host `plan(R).pieces_per_row` → CT arg `block_width_tiles`. Derived from it: `piece_bytes = 64·w`, `piece_offset = piece·piece_bytes`, `shard_shape[1] = 32·w` | a tile row's k pieces sit on k consecutive cores in the round robin | knob-turn |
| `row_in_block` (the 32 rows of one block) | independent: rows land independently; tilize needs all 32 | `rows_noc0` (rows [0, rows_noc0) read by `reader_noc0`, the rest by `reader_noc1`) | 16 / 16: the reads split evenly over both RISCs and both NoCs | host CT arg `rows_noc0`; `rows_noc1 = 32 − rows_noc0` | not across cores: across the core's two data-movement RISCs | knob-turn (perf lamp L1) |
| `token` (table rows, T) | reuse-shared: each token's row feeds about 2 output rows (top 2) plus filler rows, on unrelated cores | none of ours: the table's placement is the caller's (page t on bank t mod 110). A read moves `piece_bytes` of one row. | read pieces = `piece_bytes` | derived from `block_width_tiles` | not assigned: every core reads the rows it needs from wherever they live | none (owner push is rejected, R6) |
| `index` (gather_index, R ids) | reuse-shared across `column`: the k piece cores of a tile row read the same 32 ids | ids per reader per block = `rows_noc0` / `rows_noc1` | 16 ids (64 B) per reader per block | derived from `rows_noc0` | follows `tile_row`. Phase 0: each reader read its own 16 ids per block (64 B DRAM reads). **Refinement 3:** one read per grid row per round (the row's span), multicast along the row (R10). | built: R10 `ids_row_mcast` (Refinement 3) |

**Intermediate data re-tabled.**

| Stage output | Spans / streams | Assignment |
|---|---|---|
| Gathered RM rows of a block | `tile_row` streams, `column` spans w, `row_in_block` spans 32 (two RISC halves) | on the core that owns the shard |
| Tiles | packed straight into the output shard | — |
| ids scratch | spans the core's blocks | per RISC |

No stage collects, coordinates or hands out work on one core: there is no designating core and no stage that idles the rest of the grid.

**Selection rule for `pieces_per_row` (the pinned regime-selection function of the column extent):**

```
tile_rows = R / 32
budget    = ceil(24 * tile_rows / 110) + 4                 # tiles a core may hold
k(R)      = min { k in (1, 2, 3, 4, 6, 8, 12, 24) : ceil(tile_rows * k / 110) * (24 / k) <= budget }
w(R)      = 24 / k(R);  max_blocks_per_core = ceil(tile_rows * k / 110)
```

This is the fewest reads per row that fit the budget. Over the op's range:

| R | k | w | read bytes | blocks/core (max) | `circular_buffer_bytes` |
|---|---|---|---|---|---|
| 2240-2336 | 3 | 8 | 512 | 2 | 33,152 |
| 2368-2624 | 4 | 6 | 384 | 3 | 37,376 |
| 2656-2784 | 6 | 4 | 256 | 5 | 41,728 |
| 2816-3520 | 1 | 24 | 1536 | 1 (cores c ≥ tile_rows idle) | 49,408 |
| 3552-3648 | 6 | 4 | 256 | 7 | 58,368 |
| 3680-3936 | 4 | 6 | 384 | 5 | 62,208 |
| 3968-4544 | 3 | 8 | 512 | 4 | 66,176 |
| 4576-5280 | 2 | 12 | 768 | 3 | 74,240 |
| 5312-5856 | 3 | 8 | 512 | 5 | 82,688 |
| 5888-6144 | 4 | 6 | 384 | 7 | 87,040 |
| 6176-6304 | 6 | 4 | 256 | 11 | 91,648 |
| 6336-7040 | 1 | 24 | 1536 | 2 | 98,688 |
| 7072-7168 | 6 | 4 | 256 | 13 | 108,288 |
| 7200-7456 | 4 | 6 | 384 | 9 | 111,872 |
| 7488-8064 | 3 | 8 | 512 | 7 | 115,712 |
| 8096-8512 | 2 | 12 | 768 | 5 | 123,648 |

> **Verifier amendment (2026-11-04, measured):** k = 1 is removed from the candidate list
> (`PIECES_PER_ROW_CANDIDATES = (2, 3, 4, 6, 8, 12, 24)`), which is perf lamp L4. k = 2 fits the budget wherever k = 1
> does, because ceil(2x)·12 ≤ ceil(x)·24. Whole-row shards leave a core 1 or 2 blocks, and the last 24-tile tilize
> is not overlapped. Measured: R = 3296 went 6.59 → 4.86 us, and R = 6368 went 14.97 → 7.47 us. So the two "k = 1" rows
> in the table above now read k = 2, w = 12 (2 and 4 blocks a core). Every other row is unchanged.

Perf shapes:

| R | k | w | Reads | Per-core load |
|---|---|---|---|---|
| 2272 | 3 | 8 | 6,816 (512 B) | 103 cores × 2 blocks + 7 × 1 |
| 4320 | 3 | 8 | 12,960 (512 B) | 75 × 4 + 35 × 3 |
| 8416 | 2 | 12 | **16,832 (768 B)**, against 25,248 × 512 B today | 86 × 5 + 24 × 4 (max / mean = 1.046) |

**Per-core work, derived on the device.** Core index `c = logical_y·11 + logical_x`, the round-robin order of the ND spec. The shard count `num_shards = tile_rows·k` is a CT arg, and `num_blocks_this_core = c < num_shards ? ceil((num_shards − c) / 110) : 0`. Per-core work then needs no per-core runtime args, and the only RT args are the 2 common addresses (`table_addr`, `index_addr`). Fallback: if the implementer cannot read the logical coordinate in-kernel, pass `(c, num_blocks_this_core)` as 2 per-core RT args, the same values.

**Every knob is a parameter.**

| Knob | Kind | Defined once in |
|---|---|---|
| `block_width_tiles`, `rows_noc0`, `row_slots`, `reads_in_flight`, `num_cores`, `num_shards` | CT args | the host plan |
| `num_blocks_this_core` | derived on device | from the CT args and the core coordinate |

Loop forms: `for (block_idx = 0; block_idx < num_blocks_this_core; ++block_idx)` and `for (row = row_begin; row < row_end; ++row)`. No stride-over-count loops.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_rows_head` / `cb_rows_tail` (aliased) | `row_slots` (blocks resident) | `max_blocks_per_core`: the whole per-core assignment (1-13 blocks, ≤ 120 KiB) | The readers never wait for compute. They reserve once, issue every read under the in-flight bound, and push per block, so reads overlap tilize across all blocks with no ring wrap. Measured: 2 slots drained the incumbent's pipeline at 2-3 units a core (MOE_GLUE §15, ROUTED_COMBINE §11). |
| NoC read window (not a CB) | `reads_in_flight` per reader RISC | 8 (16 a core, over both NoCs) | Bytes in flight against latency. Above about 16 a core the grid's reads congest (MOE_GLUE §3). |
| `cb_gather_ids_noc0` / `_noc1` | none: all of a reader's ids are resident | `max_blocks_per_core` pages | One fetch at start (the brief's rule: each core's slice of gather_index read once, at the start). |
| `cb_output_tiles` | none: it is the output shard region | `max_blocks_per_core·w` pages | — |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Model L1 plan: a core's output share ≤ interleaved share + 8 KB | `block_width_tiles` (k) | the selection rule above | Xg pushes the model's next tensors out of L1, or the output OOMs |
| fast tilize: `block_width_tiles < 256`, 32×32 tiles, Float16_b in, `!dst_full_sync_en` (`tilize_helpers.inl:66-69`) | `block_width_tiles` ≤ 24 | w ≤ 24 always. With `dst_full_sync_en=True` the helper picks standard tilize, which is still exact. | slow path only |
| One-packet NoC read ≤ `NOC_MAX_BURST_SIZE` (`noc_parameters.h:290`) | `piece_bytes` ≤ 1536 | always within | the read is split or truncated |
| L1 NoC alignment 16 B; DRAM read alignment 64 B (`BLACKHOLE.md` §4) | row pieces, ids reads | Pieces are 256-1536 B at multiples of 256. Ids are read as 64 B at offsets `128·tile_row + 64·half` into a scratch whose base is `align_up(base, 64)`, with 64 B of slack allocated. | misaligned reads corrupt data |
| Transaction ids 0..15 on BH (`NOC_MAX_TRANSACTION_ID 0xF`, `noc_parameters.h:15`) | blocks in flight per reader | Ids: trids 1 and 2. Row blocks: a ring of `ROW_TRIDS = 4` trids, 3..6. A block's trid is reused only after that block was pushed, which the in-flight bound already guarantees (8 reads < 16 rows a block). | two in-flight blocks share an id; a push lands before its rows |
| trid barrier race: a `TXN_ID` barrier can pass before the latest read under that id has left the command buffer (MOE_GLUE §4) | push timing | Poll a block's trid only after a later read has been issued on that NoC, or after `noc_cmd_buf_ready` (`noc_nonblocking_api.h:200`). The last block of each reader uses the full read barrier. | the first 5 to 8 rows of a block are stale (seen on the incumbent) |
| In-flight count from `noc_reads_num_issued[noc] − NIU_MST_RD_RESP_RECEIVED` (`noc_nonblocking_api.h:87`, `:599`) | `reads_in_flight` | Valid only because each NoC has exactly one issuing RISC (dedicated mode; compute issues no NoC traffic) | the bound is miscounted |
| `NUM_L1_BANKS == 110` compile-time divisor (`dataflow_api_addrgen.h:29`) | — | `validate()` checks the grid and bank count | wrong bank or offset per read |
| CB budget ≤ 256 KiB a core | `row_slots` | the maximum is 123,648 B in range; `validate()` re-checks | — |

### Regimes

The named memory boundary is the **NoC**. The minimum is that every gathered row byte crosses the NoC once, in R·1536 B, and Xg crosses it zero times, because it is produced in its home L1.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 `local_tilize_two_noc`**: output ND-sharded as above; each core gathers its shards' row pieces on two reader RISCs (rows [0,16) on NOC0, [16,32) on NOC1) and tilizes into its own shard | **built** | table L1-interleaved (110 banks) ∧ output L1 ∧ 2240 ≤ R ≤ 8512 ∧ R%32 == 0 ∧ 1000 ≤ T ≤ 4096 ∧ grid 11×10 | 32 rows × `block_width_tiles` (= one ND shard) | **Minimum on rows and output:** R·1536 B of table reads over the NoC, 0 B of output writes. **Above minimum:** ids R·4·k B from one DRAM bank, against R·4 B. That is ≤ 1.3% of row bytes and is R7's subject. | Larger w means fewer reads per row (k), and per-read issue is the dominant per-row fixed cost. Per block: 1 trid switch per reader, 1 push per reader, 1 wait + pop of 2 CBs, and 1 output reserve/push on compute. Tilize init and uninit happen once per kernel. Fill/drain is block 0's ids + rows wait and the last block's tilize. |
| R2 `interleaved_output_rw_split`: output L1-interleaved; reads on NOC0, tile writes on NOC1 (the incumbent's topology, with whole-row or half-row reads) | **rejected**: superseded by R1 at every R in range | — | 32 rows × w | +R·1536 B of 2 KB tile writes to other banks; NOC1 lost to reads | — |
| R3 `output_dram`: compute packs into a ring CB; a writer on RISCV_1/NOC1 writes DRAM-interleaved tiles; one reader on RISCV_0/NOC0 reads all 32 rows | **deferred**. The model asks for DRAM Xg only when Xg does not fit beside the layer's L1 tensors (`experts.py` `routed_memory_config`), and the incumbent routed_combine already serves every such call. `validate()` refuses DRAM. Reachable: the block chain is unchanged; `cb_output_tiles` becomes a 2-block ring and `reader_noc1` is replaced by a writer. | `memory_config.buffer_type == DRAM` | 32 rows × w | +R·1536 B of DRAM writes; reads lose NOC1 | — |
| R4 `table_dram`: R1 with the reader's bank flavor switched to DRAM (8 banks, `dram_bank_to_noc_xy`, 64 B alignment) | **deferred**. In the op's T range the model puts the table in DRAM only on its L1-pressure fallback; multi-window layers, whose tables are always DRAM, have T > 4096, outside the op's range. The incumbent serves both. Reachable: the reader's per-read address is `bank_xy[bank] / base + (id div NUM_BANKS)·page + piece_offset`, so this is a CT switch of `NUM_BANKS` and the xy table. All piece offsets are multiples of 64. | table `buffer_type == DRAM` | same as R1 | rows cross DRAM once (the minimum for a DRAM table) | — |
| R5 `whole_row_pair_forward`: read each row once (1536 B) on one core of the pair that holds its two halves; forward the other half 1 hop | **rejected**: superseded by R1. It halves read issues at k = 2, but adds R·768 B of cross-core writes and a per-block handshake. With R1's stateful read path, read issue no longer bounds the op. | — | 32 × 24 | +R·768 B NoC writes | — |
| R6 `owner_push_scatter`: each table bank's owner reads its rows locally and writes them to the gathering cores | **rejected**: superseded by R1. Same NoC bytes, but it needs the inverse of gather_index (not an input; building it is a cross-core pass) and a per-block cross-core completion count. | — | — | R·1536 B of NoC writes instead of reads, plus the inverse-index pass | — |
| R7 `ids_sibling_forward`: the piece-0 core of each tile row reads its 32 ids once and forwards them 1 hop to the k−1 sibling cores | **deferred**. The k-fold ids read is ≤ 1.3% of row bytes, and R1 hides its latency behind block 0 (the stall-shadow fill below). Reachable: ids are per-RISC scratch filled before a block's rows issue, so a forward replaces the DRAM read with a semaphore wait into the same scratch. | — | 32 ids per tile row | ids DRAM bytes R·4·k → R·4; adds R·4·(k−1) B of 1-hop writes and one semaphore per tile row | — |
| **R10 `ids_row_mcast`** (Refinement 3): each grid row's x = 0 core reads its row's round-j span of gather_index (the 11 consecutive shards' tile rows, ≤ 896 B) from DRAM and multicasts it along the row behind a count semaphore; every reader reads its ids from a local full-size image | **built** (replaces R1's per-reader fetch; R7 not built) | same as R1 | same as R1 | ids DRAM bytes R·4·k → ≈ R·4, in 10 requests a round instead of 220; adds ≈ R·4 B of row multicasts | measured, see the amendment below |
| R8 `height_sharded_table` | **rejected**: the table's placement is the caller's, and it measured 48 us (a block's 32 rows on 2 or 3 cores; MOE_GLUE §13) | — | — | — | — |
| R9 `one_reader_dynamic_noc`: one RISC in `DM_DYNAMIC_NOC` issuing on both NoCs | **rejected**: superseded by R1's two dedicated readers. It halves issue throughput, and dynamic mode rewrites `NOC_CTRL` and bumps an L1 counter on every read (`noc_nonblocking_api.h:573-577`). | — | — | same bytes | — |

> **Refinement 3 amendment (2026-11-04, measured): the gather_index fetch topology and the issue stagger.** R7 and the
> refinement's grid-broadcast form were both weighed against the Phase 0 per-reader fetch (device us, mean of 5 calls,
> R = 2272 / 4320 / 8416; Phase 0 4.61 / 7.02 / 9.81):
> - whole index multicast to the grid by one sender (chunks of one round): 4.16 / 6.97 / 14.44. A 7 KB multicast takes
>   ~0.56 us idle and ~1-1.25 us behind the gather's read traffic, so the sender (publishing 4 more chunks) was the
>   11.5 us tail; non-sender cores finished at 8.4 us. Five senders: 5.73 / 9.53 / 15.01 (grid multicasts serialize).
> - first round multicast to the grid, later blocks fetched per reader as before: 4.33 / 6.77 / 9.84, and R = 3296 / 6368
>   slower (the 64 B per-reader fetches stretch the NOC0 reader's block from ~1.13 to ~1.45 us).
> - **R10 `ids_row_mcast` (built):** a grid row holds 11 consecutive shards each round, so its ids are one contiguous span.
>   10 senders multicast on disjoint row paths, ~0.17 us a publish. With every reader's ids local the whole grid starts
>   within ~0.3 us and the gather goes bimodal at k = 2: half the calls 14-17 us (a NoC congestion collapse on one
>   NoC in the last blocks; zones and dummy DRAM reads perturb it away). Two levers fix it, both needed: an explicit
>   per-core issue stagger (core c walks its half's rows from (s·c mod 16) in steps of 2·(c mod d)+1; (s, d) = (3, 4) at
>   k = 2, (5, 2) at k ≥ 3, `ROW_STAGGER_BY_K` / `ROW_STAGGER_DEFAULT`), and a bytes-in-flight cap
>   (`READ_BYTES_IN_FLIGHT = 6·768`: 6 reads at k = 2, 8 at k ≥ 3). Per-call probes (16 calls each at R = 8416 / 8224 /
>   6368 / 3296) then showed no outlier.
> Built: 3.73-3.75 / 6.94-7.03 / 9.13-9.22 us (R = 2240 −16%, 2656 −4%, 7392 −6%, 8064 −8%, 8224 −3%; R = 4320 and
> 6368 at parity). The start (first block at ~1.3 us) is now one DRAM round trip on the gather_index bank plus a row
> multicast; at k = 3 the NOC0 reader's block (~1.06 us) is ~40% longer than the NOC1 reader's (~0.75 us), which is
> Refinement 4's lamp L1.
>
> **Refinement 4 amendment (2026-11-04, measured per call): the readers are issue-bound; contention levers retuned.**
> Ablation (R = 8416 / 7392, mean of 5): every row read stubbed (address math kept) 8.62 / 13.39 us vs 9.17 / 13.99
> real; tilize also stubbed 8.11 / 13.11. Zones: a block body with no NoC reads still took ~0.85 us (~72 cycles a row;
> ~26 of them the two NIU register reads). So the per-row issue cost, not NoC or bank contention, held the wall:
> - **Issue path (built):** compile-time `noc_id` (register addresses fold), credit-batched in-flight accounting (the
>   response counter is read only when the credits from its last reading are spent), ids read non-volatile after the
>   ready wait. Faster issue moved the congestion-collapse cliff (R3 amendment) to k = 3 as well, so:
> - **L2 retuned:** one ~4 KB bytes-in-flight rule (`READ_BYTES_IN_FLIGHT = 4096`, `READS_IN_FLIGHT = 12`): 5 / 8 / 10 /
>   12 reads at k = 2 / 3 / 4 / 6, each the last stable step below that k's cliff (12-40 calls a point).
> - **L6 (built):** issue orders re-measured, `(5, 1)` at k = 2 and `(7, 1)` otherwise; R3's `(5, 2)` collapses at
>   k = 3 / 8 in flight in 40-50% of calls.
> - **L1, sender-column rebalance (built):** the column-0 cores (the ids row senders) finished ~0.45 us after the rest;
>   their sender reader now takes 14 rows, the other reader 18 (`SENDER_ROW_SHED_*` = 2; column 0 gets its own reader
>   kernels). L1 path-aware split (built, **parked** `ROW_SPLIT_PATH = False`): its per-block owner pass sits on the
>   issue path. At equal RISC cost the assignment was within −0.6 / +0.14 us of positional, and the pass cost
>   +1.4-3 us. Static NOC0 share (`ROWS_NOC0` 13 / 19): mixed, 16 kept.
> - **L3 / L7 not built:** `row_slots` already holds the whole assignment. The tail tilize is ~0.31 us at R = 8416, but
>   a column-split last block doubles that block's reads on an issue-bound RISC (~45 cycles a row), costing more than
>   the ≤ 0.15 us it could hide.
> Per call (median, Tracy): R = 8416 8.65 → 7.97 us, 4320 6.45 → 5.30, 7392 13.40 → 9.56, 8064 10.76 → 8.21, 2656
> 8.02 → 5.72; 2240 / 2272 / 3296 at parity.

Single built regime: R1 is the only regime that meets all three constraints at every R in range: L1 budget, local writes, and both NoCs for reads. No other regime applies inside SUPPORTED. **Regime-pinned tests:** the acceptance test runs one R from each `pieces_per_row` value {1, 2, 3, 4, 6}. A k that is only hit at some R otherwise goes untested.

### Traffic ranking

Per call at R = 8416 (12.9 MB of row bytes). Bytes are per tier; no timings.

| Candidate split | Table bytes over NoC | Output bytes over NoC | ids from the DRAM bank | Cross-core extra | Rank |
|---|---|---|---|---|---|
| **R1: output shards local, rows split across 2 RISCs / 2 NoCs** | 12.9 MB, half on each NoC | 0 | 67 KB (k = 2) | 0 | **1, chosen** |
| R5: whole-row read + pair forward | 12.9 MB | 0 | 34 KB | 6.5 MB of 1-hop writes + handshakes | 2 |
| R6: owner-push scatter | 0 reads (local), 12.9 MB writes | 0 | + an inverse-index pass | completion counts | 3 (needs data that is not an input) |
| R2: interleaved output (incumbent topology) | 12.9 MB, NOC0 only | 12.9 MB of tile writes, NOC1 | 34-67 KB | 0 | 4 |
| Intra-core: blocks alternated between the two RISCs instead of rows | same as R1 | 0 | same | 0 | rejected: 1-13 blocks a core leaves one RISC a block ahead (3/2 at 5 blocks) |
| Dependent-axis split + combine | — | — | — | — | not applicable: no axis is dependent |

The occupancy is the same for every row: all 110 cores, except the k = 1 ranges, where cores c ≥ tile_rows have no shard. The chosen split is also the cheapest.

**Operand-reuse check** for the chosen split (output shards):

| Operand | Varies along the split? | Consequence |
|---|---|---|
| table | yes: each output row reads its own row piece | no broadcast |
| gather_index | does **not** vary along `column`: the k piece cores of a tile row read the same ids | broadcast row R7 (deferred) |

**Stall-shadow check.**

| Stage that waits | What fills the window |
|---|---|
| The ids fetch from the one DRAM bank, about 220 RISCs at once | Each reader issues block 0's ids first under trid 1 and the rest under trid 2. It waits only for trid 1 before issuing block 0's row reads; the other blocks' ids land under block 0's reads. This is legal because a block's row reads depend only on its own ids. Floating-point order is untouched (no arithmetic). |
| Compute waiting on a block's rows | Nothing: tilize is its only work, and blocks are tilized in order. |
| A reader waiting on the in-flight bound | Nothing algebraic. Address precomputation for the next read is the implementer's. |

### Block schedule

```cpp
// per core; num_blocks_this_core and shard s(block_idx) = c + num_cores * block_idx derived on device
fetch_ids_block();                                   // each reader: its 16 ids of every block, block 0 first
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    gather_rows_block(block_idx);                    // reader_noc0: rows [0,rows_noc0); reader_noc1: the rest
    tilize_block(block_idx);                         // compute: 32 x w RM -> w tiles into output shard slot block_idx
}
// no store_block: tilize packs straight into the output shard (zero output NoC bytes)
```

**`fetch_ids_block` (both readers, once a kernel)**

| Item | Value |
|---|---|
| Block shape | all `num_blocks_this_core` × `rows_per_reader` ids |
| Reads | one 64 B DRAM read per block from offset `128·tile_row + 64·half`; block 0 under trid 1, the rest under trid 2 |
| Resident | ids stay for the whole kernel |
| Frequency | one wait on trid 1 before block 0's rows (with the cmd-buf rule), one wait on trid 2 before block 1's rows; then the stateful read state is programmed once for table reads |

**`gather_rows_block` (each reader, per block)**

| Item | Value |
|---|---|
| Block shape | its `rows_per_reader` rows × `piece_bytes` |
| Reads | one read per row: source `l1_bank_to_noc_xy[noc][id mod 110]`, address `table_addr + (id div 110)·1536 + piece·piece_bytes (+ bank_to_l1_offset[bank])`; destination `slot(block_idx) + row·piece_bytes` |
| Resident | all slots |
| Frequency | trid tag set once a block. Reserve once a kernel (`row_slots·w` pages). Push `w` pages once a block, when its trid is flushed (polled between reads, non-blocking, under the race rule). No per-read barrier. One full read barrier at kernel end. The in-flight bound is checked per read against `reads_in_flight`. |

**`tilize_block` (compute, per block)**

| Item | Value |
|---|---|
| Block shape | 32 rows × `block_width_tiles` |
| CB sync per block | `wait_front(w)` on `cb_rows_tail`, then the tilize helper (waits/pops `cb_rows_head`, reserves/pushes `w` on `cb_output_tiles`), then `pop_front(w)` on `cb_rows_tail` |
| Resident | the output shard (written once per tile) |
| Frequency | `compute_kernel_hw_startup` once; tilize init once (`InitOnly`, 0 blocks) before the loop and uninit once (`UninitOnly`, 0 blocks) after it |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 row-to-NoC split | Positional halves (rows 0-15 on NOC0, 16-31 on NOC1) balance the RISCs exactly but ignore path length. On the NoC torus the shorter of the two paths averages roughly a third fewer hops than a fixed one. | Per row, the RISC whose NoC has the shorter response path from the row's bank. The host builds a 110-bit mask per core (4 common/per-core RT words). Both readers then read all 32 ids of a block and skip the other's rows. Costs: per-block imbalance and ids read twice. |
| L2 `reads_in_flight` | 8 per RISC follows the incumbent's 512 B pieces. 768-1536 B pieces put more bytes in flight per read. | 4, 12, 16 per RISC at R = 8416 and 2272 |
| L3 `row_slots` (overlap lamp) | Full residency costs ≤ 120 KiB, which counts against the model's L1 plan through `circular_buffer_bytes`. | `row_slots = 3` (ring; slot = base + (block_idx mod row_slots)·slot_bytes) |
| L4 `pieces_per_row` (overlap lamp) | Minimal k gives the fewest reads, but in the k = 1 ranges a core has 1 block, so its tilize cannot overlap any read. | The next legal k: at R = 3296, k = 2 gives 2 blocks a core at the same 24-tile max share |
| L5 grid synchronization | Not applicable: no cross-core collective, semaphore or multicast | — |
| L6 block issue order | Cores read their blocks in increasing slot order, and the banks are token-scattered, so no stagger is planned (`bank_stagger`). | A start at block `c mod num_blocks` if the profile shows bank hot spots |
| L7 tilize width split | One `fast_tilize_block` of w tiles per block. | Two half-width calls (no CB change), if the last block's tilize shows on the tail |

Catalog entries that informed the knobs:

| Entry | Knob |
|---|---|
| `split_reader` | two reader RISCs, because the incumbent is issue-bound |
| `trid_double_issue` | trid per block, a ring of 4 |
| `double_buffer` | in-flight window, full residency |
| `read_coalescing` | one read per row piece, never 64 B face rows |
| `noc_placement` | both NoCs for reads, because there are no writes |
| `bank_stagger` | L6 |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| gather_index → ids scratch | uint32, 64 B per block per reader | plain NoC read from the DRAM page (TensorAccessor or bank-0 address), trids 1/2 | each reader its own half |
| table rows → `cb_rows_head` / `cb_rows_tail` slot | bf16 RM, `piece_bytes` per row, 32 rows per slot | stateful one-packet read: size programmed once; per read only the source coordinate, source address and destination are written | `reader_noc0` = RISCV_0 on NOC0 (rows 0-15), `reader_noc1` = RISCV_1 on NOC1 (rows 16-31). Aliased CBs give one slot, two producers of disjoint halves, two credit counters. |
| slot → output shard | bf16 TILE, w tiles | fast tilize (kernel_lib helper), packed into `cb_output_tiles`, which is backed on the output buffer | Pack Float16_b unchanged. Bit-exact for finite values (the incumbent's `fast_tilize_block`, `combine.cpp:41-53`). |
| output | — | none: the data is already home | no writer kernel |

**TARGET axes against the Blocking Model.**

| Axis | Built | Not built |
|---|---|---|
| `table_buffer` | `l1` (R1) | `dram` = R4: a knob-turn of the reader's bank flavor, excluded from SUPPORTED |
| `output_buffer` | `l1` (R1) | `dram` = R3: a scheme-change, because writes take one NoC and a RISC; excluded |
| `window` | both (`ragged` changes nothing: ids < T; the kernels never see T) | — |

TARGET has no `*_SHARDED` values: the output's ND layout is the op's own (`output_memory_config`).

**Structural impossibilities:** none to add to INVALID.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one ND shard = one block of 32 rows × w tiles (tile_row = s div k, piece = s mod k) |
| Grid | all 110 cores, CoreRangeSet (0,0)-(10,9). Round robin in ROW_MAJOR order: c = y·11 + x, matching the ND spec's core order |
| Per-core work | `num_blocks_this_core = c < num_shards ? ceil((num_shards − c)/110) : 0`; blocks s = c + 110·j |
| Remainder | `num_shards mod 110` cores take one more block. R is a multiple of 32, so `tile_rows = R/32` exactly. The output is padded to whole tiles by construction. T is never tiled. Cores with 0 blocks (the k = 1 ranges) run kernels that return at once, and still fetch no ids. |
| Plan caching | `plan(R, compute_kernel_config)` is computed once and reused: k, w, max_blocks, CB descriptors, CT args, kernel descriptors. Per call only `table_addr` and `index_addr` (common RT args) and the output CB's buffer change. generic_op re-points a tensor-backed CB on a cache hit (`tt_metal/impl/program/program_descriptors.cpp:239-241`). |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_rows_head` | 0 | 2048 (one tile's bytes) | `row_slots·w` | Spans `row_in_block` (32 rows) × `column` (w) per slot and `row_slots` blocks of `tile_row`. One `CBDescriptor` carries both format descriptors (alias group with `cb_rows_tail`). Head and tail push and pop in lockstep. | Float16_b | reader_noc0 | compute | whole kernel |
| `cb_rows_tail` | 1 | 2048 | `row_slots·w` | Same allocation as `cb_rows_head`, with its own credit counter, so each reader has its own push edge for its 16-row half | Float16_b | reader_noc1 | compute | whole kernel |
| `cb_output_tiles` | 2 | 2048 | `max_blocks_per_core·w` | Backed on the output tensor (`ttnn.cb_descriptor_from_sharded_tensor(2, output, core_ranges=grid)`), i.e. the core's shard slots in order. No wrap: pushes ≤ capacity. | Float16_b | compute | none (it is the output tensor) | whole kernel |
| `cb_gather_ids_noc0` | 3 | 64 | `max_blocks_per_core + 1` (+1 page = 64 B of alignment slack) | Spans this reader's ids of all blocks | UInt32 | reader_noc0 (self-owned scratch: reserve once, never pushed) | reader_noc0 | whole kernel |
| `cb_gather_ids_noc1` | 4 | 64 | `max_blocks_per_core + 1` | as above | UInt32 | reader_noc1 | reader_noc1 | whole kernel |

The full ledger is `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `fetch_ids_block` | `num_blocks_this_core` × `rows_per_reader` ids | raw read + kernel_lib trid helpers | gather_index (DRAM) | `cb_gather_ids_noc0` / `_noc1` (reserved, never pushed) | ids resident to kernel end |
| 2 | `gather_rows_block` | `rows_per_reader` × `piece_bytes` | additive kernel_lib gather-read helper (see API Mapping) | table (L1 interleaved) | `cb_rows_head` (reader_noc0) / `cb_rows_tail` (reader_noc1): push `w` per block | the slot is visible to compute once both pushes land |
| 3 | `tilize_block` | 32 × `block_width_tiles` | `compute_kernel_lib::tilize` | `cb_rows_tail` (wait `w`, caller), `cb_rows_head` (`w`, helper waits / pops) | `cb_output_tiles` (`w`, helper reserves / pushes) | tail popped by the caller after the helper |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| tilize_block | helper | `compute_kernel_lib::tilize<w, cb_rows_head, cb_output_tiles, InitUninitMode, WaitMode::NoWait>` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:194-207` (WaitMode `:43-47`, InitUninitMode `:35-40`; loop `tilize_helpers.inl:217-238`, fast path `:66-69`, `:228`) | `InitOnly`(0 blocks) once; `Neither`(1 block) per block with `WaitMode::NoWait`, the caller having waited on head and tail; `UninitOnly`(0 blocks) once | `cb_rows_head` | `cb_output_tiles` | `block_width_tiles` (template arg = CT `w`) |
| tilize_block (startup) | helper | `compute_kernel_hw_startup(cb_rows_head, cb_output_tiles)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | once | — | — | — |
| tilize_block (tail credit) | raw_api | `DataflowBuffer::wait_front` / `pop_front` on `cb_rows_tail` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h` | `w` pages | `cb_rows_tail` | — | `w` |
| gather_rows_block (read state) | raw_api | `ncrisc_noc_read_set_state<noc_mode, one_packet=true>` once per reader (size `piece_bytes`) | `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:1436` | `piece_bytes` | — | — | `piece_bytes` (from w) |
| gather_rows_block (per read) | raw_api | `noc_cmd_buf_set_targ_addr_coordinate(noc, cmd_buf, l1_bank_to_noc_xy[noc][bank])` + `ncrisc_noc_read_with_state<noc_mode, true, true>(noc, cmd_buf, src_addr, dst_addr)` | `noc_nonblocking_api.h:400`, `:1494`; `firmware_common.h:42-43` (bank tables); bank = `id − (id div 110)·110` via `udivsi3_const_divisor<NUM_L1_BANKS>` (`dataflow_api_addrgen.h:29`) | src, dst only | table | `cb_rows_*` slot | — |
| gather_rows_block (trid) | helper | `dataflow_kernel_lib::set_read_trid(noc, trid)` once per block; push when `noc.is_read_trid_flushed(trid)` (`api/dataflow/noc.h:700`), under the cmd-buf rule (`noc_cmd_buf_ready`, `noc_nonblocking_api.h:200`) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.inl:110-114` | trid ring of 4 | — | `cb_rows_*` | — |
| gather_rows_block (window) | raw_api | in-flight = `noc_reads_num_issued[noc] − NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED)` ≤ `reads_in_flight` | `noc_nonblocking_api.h:87`, `:599` | `reads_in_flight` | — | — | `reads_in_flight` |
| gather_rows_block (end) | raw_api | `noc_async_read_barrier()` once, then the remaining pushes | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1919` | — | — | `cb_rows_*` | — |
| fetch_ids_block | raw_api | `noc.async_read` from the gather_index accessor (DRAM, one page, `offset_bytes = 128·tile_row + 64·half`) under trids 1/2 | `tt_metal/hw/inc/api/dataflow/noc.h:243`; `api/tensor/tensor_accessor.h:493` | 64 B | gather_index | `cb_gather_ids_*` | `rows_per_reader` |

**Helpers considered and rejected** (raw_api rows):

| Candidate | File:line of the mismatch | Why it cannot express the shape |
|---|---|---|
| `dataflow_kernel_lib::set_read_state` / `read_with_state` | `local_copy_helpers_dataflow.hpp:57`, `:74-95` | The source is fixed to the reading core (`local_addr`), so it cannot change the source coordinate per read. |
| `Noc::set_async_read_state` / `async_read_with_state` | `api/dataflow/noc.h:295-379`; requirement at `:322` | The source must stay at the set-state NoC location; it keeps only the low 32 bits of the address. |
| `noc.async_read(TensorAccessor, …)` | `api/dataflow/noc.h:243` | Generic per-read address generation and cmd-buf setup: the incumbent's 60-70 cycles a read (MOE_GLUE §4). |

**What closes the gap:** an **additive** kernel_lib dataflow helper, e.g. `gather_read_helpers_dataflow.hpp`: `set_gather_read_state<piece_bytes>(noc)` plus `gather_read(noc, bank_xy, src_addr, dst_addr)`. Existing users stay unchanged, so this is allowed by the brief; the implementer chooses between it and inline raw calls. Tile-row tilize: no dataflow helper applies (compute does it). No `ttnn.Mcast` is used: no operand is multicast in R1.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| ND MemoryConfig equality | A tensor allocated from an ND spec may report a rewritten legacy config (`tensor_spec.cpp:337-446`), and the golden test compares `output.memory_config()` with `output_memory_config(...)` | `output_memory_config` returns the config of the `TensorSpec` built from the ND spec; the op allocates from that same spec. The acceptance test checks equality. |
| `cb_descriptor_from_sharded_tensor` on a pure ND tensor | Its default `core_ranges` dereferences the legacy shard spec, which is empty (`ttnn/core/tensor/tensor_utils.cpp:62`) | Pass `core_ranges=` the full grid explicitly |
| Aliased CB lockstep | Two producers fill disjoint halves of one slot; the pointers drift if the quanta differ | Both readers push exactly `w` pages per block in block order. Compute waits on both and pops both per block. Same page size (2048) on both indices. |
| Slot addressing ahead of the write pointer | A reader issues several blocks before pushing; the CB write pointer lags | Reserve `row_slots·w` once; `slot(j) = base + (j mod row_slots)·w·2048`, with `base` from `get_write_ptr()` after that reserve |
| trid race | Polling the most recent read's trid passes early, leaving the first rows of a block stale | The cmd-buf rule above; the last block uses the full barrier. `test_index_patterns_copy_exactly` and the acceptance test catch it. |
| Outstanding-count bound | Out-of-order completion: a count never proves a given read landed | The count is used only for the window bound. Completion is by trid per block. |
| `bank_to_l1_offset` | `get_noc_addr_from_bank_id<false>` does not add it (`dataflow_api_addrgen.h:516`) | Add `bank_to_l1_offset[bank]` per read, or assert it is 0 for all 110 banks on host and drop it |
| ids alignment | DRAM reads need 64 B agreement | Scratch base `align_up(write_ptr, 64)` with one spare page |
| Idle cores at k = 1 | `num_blocks_this_core = 0` for c ≥ tile_rows | Kernels return before any reserve, read or init |
| Kernel exit with reads outstanding | A preloaded next program can hang (`BLACKHOLE.md` §6) | Final `noc_async_read_barrier()` on each reader |
| Many cached programs | w is a CT arg; each distinct k changes the program | ≤ 5 distinct kernel binaries (k ∈ {1,2,3,4,6}) × `num_shards` CT values; the model uses a few R per run |
| Index values in CT/cached state | Traced model; the index changes every call | Only buffer addresses are RT args; the ids are read on device each call. `test_new_tensors_between_calls` and the acceptance test's re-call check guard this. |
| Hot rows (all ids equal) | One bank serves all reads | Correct, slow; not a perf case |

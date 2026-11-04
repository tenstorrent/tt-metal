# Operation Design: nomic_moe_combine

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (indexed gather + weighted fp32 reduction over top-k + bias + tilize), one `ttnn.generic_op` program |
| Goal | The combine of nomic-embed-text-v2-moe's MoE layers on one Blackhole p300c (11x10 grid): each token's two expert rows weighted, summed and biased in fp32, rounded to bf16 once, written straight into the layer norm's ND-sharded layout. Perf target at 8x512 (T = 4096, R = 8416): <= 20.0 us accept, 13.0 us stretch (incumbent 32.22 us). |
| Math | `out[t, :] = round_bf16( b + W[0,0,t,0] * Y[0,0,I[0,t],:] + W[0,1,t,0] * Y[0,0,I[1,t],:] )`, t in 0..T'-1 (padding tokens t >= T have I = 0, W = 0, so their rows are exactly `b`) |
| Mode | Derivative (incumbent: `ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/routed_combine/`), re-blocked |
| References | incumbent `device/routed_combine_program_factory.cpp`, `device/kernels/{compute/combine.cpp, dataflow/combine_reader.cpp, dataflow/read_batches.hpp, dataflow/combine_writer.cpp}`; `models/experimental/nomic_embed_text_v2_moe/docs/{ROUTED_COMBINE.md, MOE_GLUE.md §2-5,13,15, ROUTING_METADATA.md §2-3, BLACKHOLE.md, LAYER_NORM.md}`; `ttnn/ttnn/operations/nomic_residual_layer_norm/nomic_residual_layer_norm.py` (`_plan`, `input_memory_config`); `eval/golden_tests/nomic_moe_combine/{feature_spec.py, helpers.py}`; catalog `ttnn/ttnn/operations/examples/master.md` (split_reader, trid_double_issue, compute_block_size, compute_fusion, double_buffer) |

Catalog entries and the knob each informed:

| Entry | Knob it informed |
|-------|------------------|
| `split_reader` | the row reads are issue-bound on one RISC in the incumbent (MOE_GLUE §4: 63 cycles a read); this design splits them across both DM RISCs, one choice k each, because the output writes are local and no RISC is needed for writing |
| `trid_double_issue` | bounded reads in flight per RISC under rotating transaction ids (`batch_reads` x `batch_depth`), with the CB-slot-by-ring-position rule |
| `compute_block_size` | inits once per core / per DEST group, never per small unit; no format reconfig in the main loop (bias and weights both fp32 in srcB) |
| `compute_fusion` | keep the bias + two products in DEST (FPU products accumulate in DEST, no L1 intermediate between them) |
| `double_buffer` | not the lever here: the rows CB holds the whole shard, so the reader never waits on compute |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `expert_rows` | ttnn.Tensor | yes | (1, 1, R, 768) bf16 ROW_MAJOR interleaved, 2240 <= R <= 8512; Phase 0: L1 | - | address RT |
| `combine_index` | ttnn.Tensor | yes | (2, T') uint32 ROW_MAJOR interleaved DRAM | - | address RT |
| `combine_weights` | ttnn.Tensor | yes | (1, 2, T', 1) float32 TILE interleaved DRAM | - | address RT |
| `bias` | `PreparedCombineBias` | yes | exactly what `prepare_combine_bias` returned | - | address RT |
| `num_tokens` (T) | int | yes | 1000..4096, T % batch == 0 | - | host (plan) |
| `batch` (B) | int | no | 1, or B > 1 with (T / B) % 32 == 0 | 1 | host (plan) |
| `memory_config` | ttnn.MemoryConfig | no | Phase 0: ND-sharded L1 per "Output" below | `ttnn.L1_MEMORY_CONFIG` (outside Phase 0 SUPPORTED -> SupportRefusal; the model always passes norm_layout) | host (plan) |
| `compute_kernel_config` | ttnn.ComputeConfigDescriptor | no | `fp32_dest_acc_en=True`, `math_fidelity=HiFi4` (precision contract) | `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False | CT |

Module exports: `nomic_moe_combine`, `prepare_combine_bias(bias, device)`, `default_compute_kernel_config()`, `circular_buffer_bytes(num_tokens, batch, memory_config)`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate`.

`prepare_combine_bias(bias, device)` (load time, host code allowed): reads the (1,1,1,768) bf16 TILE bias to host, widens it to fp32 (exact), replicates it into `BIAS_COPIES = number of DRAM banks (8)` rows and uploads `(1, 1, BIAS_COPIES, 768) float32 ROW_MAJOR` interleaved in **DRAM** (row p = page p on DRAM bank p). Returns a frozen dataclass `PreparedCombineBias(rows: ttnn.Tensor)`. One copy per bank spreads the cores' bias reads over all banks (core i reads copy `i % BIAS_COPIES`); fp32 keeps srcB's format identical for the bias and the weights in the compute's main loop (no srcB reconfig).

## Tensors

### Input

| Property | expert_rows | combine_index | combine_weights | bias (prepared) |
|----------|-------------|---------------|-----------------|-----------------|
| Shape | (1,1,R,768) | (2,T') | (1,2,T',1) | (1,1,8,768) |
| Dtype | bf16 | uint32 | float32 | float32 |
| Layout | ROW_MAJOR (page = row, 1536 B) | ROW_MAJOR (page k = choice k, 4T' B) | TILE (page = k*(T'/32) + tb, 4 KB; only col 0 of faces 0, 2 used) | ROW_MAJOR (page = one copy, 3072 B) |
| Memory | interleaved L1 (Phase 0); DRAM deferred | interleaved DRAM | interleaved DRAM | interleaved DRAM |

### Output

| Property | Value |
|----------|-------|
| Shape | (B, 1, T/B, 768) — page for page (1, 1, T, 768) since B > 1 implies S % 32 == 0 |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | Phase 0: `memory_config` ND-sharded L1, shard (1, 1, 32·r, 32·W), ROW_MAJOR orientation, ROUND_ROBIN_1D distribution, one shard per core — the norm's `input_memory_config((B,1,S,768), device)`. All shard geometry is read off `memory_config.nd_shard_spec`, never from a table of shapes. |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Scheme: **owner-computes** — each core of the output's ND-shard grid computes exactly its own shard and packs it straight into that shard (the output CB is the shard). Its two data-movement RISCs each gather one choice's row segments, on separate NoCs; no RISC writes anything.

### Axes

Names: `tensor_token_tiles = T'/32` (flat tile rows, B folded in), `shard_row_tiles = r` (shard height in tiles, from nd_shard_spec), `shard_col_tiles = W` (shard width in tiles), `core_row_tiles` = tile rows of this core's shard (≤ r; < r only for the last row group when St % r ≠ 0).

| Axis | Character (+ reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|---------------------|-------------|---------------|-------------|-----------------|--------------|
| token tile rows (flat t / 32; batch folded in) | independent — each output row depends only on its own token's index, weights and rows | `block_extent[token_rows]` = `core_row_tiles` (whole shard) ; streaming quantum `rows_per_push` = 1 tile row | whole shard (r ≤ 4 in every norm plan); push quantum 1 tile row (the tilize unit: 32 tokens — a tile cannot be computed before its 32 rows land) | `r` = `nd_shard_spec.shard_shape[-2] / 32` (host, CT `shard_row_tiles`); `core_row_tiles` RT per core | by the output's shard row group: shard i -> core i (row-major, round robin) | knob-turn (re-shard = new layout; the op reads it off nd_shard_spec) |
| hidden column tiles (768 / 32 = 24) | independent — no cross-column dependency; the bias and weights are per column / per token only | `block_extent[hidden]` = `W` (whole shard width) | W = 768 / G: 8 (G=3) or 12 (G=2) | `nd_shard_spec.shard_shape[-1] / 32` (host, CT `shard_col_tiles`) | by the shard's hidden slice g = i % G | knob-turn |
| choice k (top_k = 2) | dependent — summed into one output; fixed order k = 0 then 1 (deterministic fp order) | `top_k` = 2, whole extent, never split across cores | 2 (whole) | CT `top_k` (validated == 2) | not split across cores: summed in DEST on the owner. Its **reads** are split across the core's two DM RISCs (k = 0 on RISCV_0/NOC0, k = 1 on RISCV_1/NOC1) | scheme-change (cross-core split needs an fp32 partial exchange: rejected regime) |
| batch b | independent — folded into the flat token axis (B > 1 ⇒ S % 32 == 0, so tile row tr = b·St + s/32 is flat) | none separate: folded | folded | `batch` (host) only for the output shape | via the token-row axis | n/a |
| expert-row source axis R (gather source) | independent — rows are gathered, never reduced; each used row's segment is read exactly once per hidden slice | not blocked (indexed) | n/a — the reader reads the 32·k-th slot rows of each tile row by index | `combine_index` values (device) | follows the token-row owner | n/a |
| intra-tile token rows (32) / columns (32) | independent — inside the tile; the tilize unit | fixed by the tile | 32 x 32 | hardware | n/a | n/a |

Per-stage data (re-running the table on what each stage produces):

| Stage output | Axes it spans | Assigned to | Note |
|--------------|---------------|-------------|------|
| raw weight tiles (faces 0, 2 of tiles (k, tb)) | token rows x choice | each owner reads its own; the G cores of a row group read the same tiles (reuse-shared across the hidden split — see deferred regime `group_shared_index_weights`) | |
| index slice (32 ids a tile row a choice) | token rows x choice | each owner reads its own, one read per choice per core | |
| expanded weight tiles (column-broadcast fp32) | token rows x choice | every owner expands its own (2·core_row_tiles tile ops, G-fold redundant across a group, negligible) | no stage computes on one core while others wait |
| bias row-0 tiles | hidden | every owner builds its own W tiles from one read | |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_rows_choice0`, `cb_rows_choice1` | `rows_depth_tile_rows` | `r` (whole shard resident) | the reader never waits on compute: the row reads are the bound, so their issue must not stall on CB space; compute starts per tile row as each lands |
| `cb_weights_raw_choice0/1` | none (one per tile row of the shard) | `r` | all weights prefetched at kernel start, ahead of the rows they gate |
| `cb_weights_bcast` | none | `2·r` | expansion once per core; resident for the main loop |
| `cb_bias_tiles` | none | `W` | resident for the whole kernel |
| `cb_output_shard` | n/a (the output shard) | `r·W` | the destination itself |
| reads in flight (per DM RISC) | `batch_reads` x `batch_depth` | 4 x 2 = 8 a RISC, 16 a core | bounded issue: MOE_GLUE §3 measured congestion above ~16 a core |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| bias staging inside `cb_bias_tiles` tile 0, bytes 2048..4095 (faces 2-3, fp32) | W ≤ 16 | validate: `shard_col_tiles <= 16` and `24 % shard_col_tiles == 0` | the staged bias row overruns into tile 1's row-0 region before it is scattered: wrong bias columns |
| output CB packs the shard's pages sequentially from page 0 | shards per core == 1 | validate: `num_shards == num_cores` (true for every norm plan) | a partial shard followed by another shard misaligns every later page |
| DEST accumulators live across the seed + 2 multiply passes | DEST group `dest_group_tiles` ≤ `DEST_AUTO_LIMIT` (4 at fp32 half-sync, 8 full-sync) | `n = min(DEST_AUTO_LIMIT, W - j0)` per group | packing tiles that were never accumulated / overwriting live accumulators |
| `unpack_tilizeA_B` row stride = `W·32` datums (CT `ct_dim`) | the rows CB block width == W | CT arg from the same `shard_col_tiles` | tilize reads the wrong row of the row-major block |
| transaction ids are 4-bit (1..15) | `batch_depth + 3` (rows ids + index / bias / weights ids) ≤ 15 | `static_assert(batch_depth + 3 <= 15)`; `static_assert(batch_depth >= 2)` (MOE_GLUE §4 race rule) | aliased ids retire the wrong batch: stale rows consumed |
| one-packet stateful NoC read | segment bytes `W·64` ≤ `NOC_MAX_BURST_SIZE` | W ≤ 24 ⇒ ≤ 1536 B | n/a at these sizes; assert |
| L1 read alignment (16 B) / DRAM (64 B) | segment offset `g·W·64`, bias offset `g·W·128` | both multiples of 64 by construction | misaligned NoC reads (Watcher) |
| CB L1 budget | `r`, `W` | `circular_buffer_bytes(...) <= 384 KB` in validate | OOM / clobbering the model's L1 tensors |

### Regimes

Minimum named: every *used* expert-row byte crosses the NoC once (2T row segments x 1536 B = 12.6 MB at T = 4096), every output byte is written once and never crosses the NoC, index and weights cross DRAM once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `owner_computes` | **built** | rows_buffer L1 ∧ output ND-sharded L1, ROW_MAJOR, ROUND_ROBIN_1D, one shard per core, shard (1,1,32r,32W) with 24 % W == 0, W ≤ 16 ∧ `circular_buffer_bytes ≤ 384 KB` | the whole shard: `core_row_tiles x W` tiles x 2 choices, streamed per tile row | rows: **minimum** (each used row segment read once, in W·64 B pieces, 512 B or 768 B); output: **minimum** (packed in place, 0 NoC bytes); index read G times (3x, 96 KB at 4k) and weight faces 0+2 read G times (1.5 MB at 4k, vs. 512 KB distinct) — above minimum, see `group_shared_index_weights`; bias W·128 B a core | already the whole shard: one pass per core; the fixed costs it amortizes once per core are the index DRAM round trip (incumbent: once per token block), the weight prefetch and expansion (incumbent: per unit), the bias read, every write barrier (none: no writes), the pack-format reconfig (once). Per DEST group: 2 inits (bias seed, tilize-A x B multiply) |
| `owner_computes_dram_rows` | deferred — positive reason: the perf cases and the model's L1-placed windows are all `rows_buffer = l1`; where w2's output lands in DRAM the model keeps the incumbent (it adopts the op per shape). Reachable: only the row-address function changes (8 DRAM banks via TensorAccessor instead of the 110-bank L1 arithmetic); CBs, compute and split unchanged | rows_buffer DRAM | same | rows from DRAM in 512 B scattered pieces (request-bound, incumbent 83 us at 4k) | same |
| `interleaved_output` | deferred — positive reason: the model passes only norm_layout; l1/dram_interleaved are optional TARGET values served today by the incumbent. Reachable: the plan takes a virtual shard spec (the norm's plan for the shape), `cb_output_shard` becomes a streaming `cb_output_tiles` (2 tile rows deep), and RISCV_1 gains a write role (tiles by page id through a TensorAccessor) after its row reads are issued; compute unchanged | output INTERLEAVED (L1 or DRAM) | same | + 6.3 MB of tile writes over the NoC | same |
| `whole_row_remote_write` | deferred — positive reason: the built regime already moves the minimum NoC bytes (12.6 MB) and keeps both NoCs free of writes; this one moves 12.6 MB of reads in 1536 B pieces **plus 4.2 MB (2/3 of the output) of remote 2 KB tile writes**, and its only advantage — 3x fewer, larger reads — is a per-piece efficiency hypothesis. Reachable: W = 24 is the same block knob, plus the `interleaved_output` write role aimed at the owners. One experiment settles it: 8x512 with W = 24 on 110 cores vs. the built regime | any output | `core_row_tiles x 24` | +4.2 MB cross-core writes; 3x fewer read issues | same |
| `owner_plus_helpers` | deferred — positive reason: once both DM RISCs issue reads, the read phase is bound by the grid's aggregate NoC bandwidth (the stated speed of light: 12.6 MB ≈ 10 us at 0.65 TB/s a NoC), which idle cores (14 at 2k/4k, 46 at 1k) do not add; a helper's tiles also cost remote writes and a completion signal per owner. Reachable: extend the core list; a helper takes a tail of an owner's tile rows and writes them with the `interleaved_output` write role into the owner's shard | any | sub-shard | + helper share of the output as remote tile writes | n/a |
| `group_shared_index_weights` (operand-reuse broadcast) | **built (Refinement 2)**, as a layer on `owner_computes` (knobs `SHARE_INDEX` / `SHARE_WEIGHTS`, leader = member 0 per choice; 8x512 14.9 → ~13.0 us). Original deferral, falsified by measurement: index and weights do not vary along the hidden split, so the G owners of a row group read identical bytes (reuse-shared by construction of the split); sharing saves 1.0 MB DRAM weights + 64 KB index at 4k but every byte saved is off the critical path in the built regime (prefetched at kernel start, consumed by compute that lags the row reads) while sharing adds a group rendezvous ahead of each peer's first row read. Reachable: member 0 of a group reads index + weight faces and unicasts them (2-3 consecutive row-major cores, not a rectangle) into the peers' `cb_index_*`/`cb_weights_raw_*` + one semaphore; the readers' "index/weights landed" step becomes a semaphore wait | G > 1 | same | -1.0 MB DRAM, +1.0 MB cross-core | n/a |
| `choice_split_across_cores` | rejected — superseded by `owner_computes`: splitting the dependent k axis needs an fp32 partial tile exchange (+12.6 MB of 4 KB partials at 4k) and a combine, with no residency gain (the reduced extent is 2) | - | - | +12.6 MB cross-core | - |
| `incumbent_units` (32 tokens x 8 tiles over 110 cores, contiguous runs) | rejected — superseded by `owner_computes`: same 12.6 MB reads but + 6.3 MB of remote tile writes on NOC1, an index DRAM round trip + full read barrier per token block, a write barrier per unit, and the tilize round trip through a CB | - | 1 x 8 tiles | +6.3 MB writes | - |

Single built regime ⇒ no runtime selection; the predicate is enforced by `validate()` (anything else raises SupportRefusal).

### Traffic ranking

| Rank | Candidate split | NoC bytes (4k) | DRAM bytes (4k) | Fan-out / pieces | Verdict |
|------|-----------------|----------------|-----------------|------------------|---------|
| 1 | token rows x hidden slice on the shard owner (`owner_computes`) | 12.6 MB reads, 0 writes | index 96 KB, weights 1.5 MB, bias 96 KB | 256 reads of 512 B a core (128 a RISC) at 4k; 128 at 2k; 64 of 768 B at 1k | **built** — fewest NoC bytes, both NoCs read-only |
| 2 | same + group-shared index/weights | 12.6 + 1.0 MB | 0.6 MB | + one unicast per peer | deferred (bytes off critical path) |
| 3 | whole rows (24 tiles) on any core + remote writes | 12.6 + 4.2..6.3 MB | 0.6 MB | 3x fewer, 1536 B reads | deferred (measured hypothesis) |
| 4 | owners + helpers | 12.6 + helper share | same as 1 | more issuers, same NoC | deferred |
| 5 | incumbent units | 12.6 + 6.3 MB | 3 MB weights (full 4 KB tiles per unit) + per-block index | 66 reads a unit | rejected |
| 6 | split k across cores | 25.2 MB | same | partial exchange | rejected |

The chosen split is the cheapest-traffic split. Occupancy (64 / 96 / 96 of 110 cores at 1k / 2k / 4k) is the norm layout's, accepted because the read phase is NoC-aggregate-bound (row 4).

Stall-shadow check:

| Stage that waits | What it waits for | Work scheduled into the stall |
|------------------|-------------------|-------------------------------|
| reader k, start | the index slice (DRAM, all owners hit the 2 index pages at once) | the weight-face reads (and, on reader 0, the bias read) are issued **behind** the index read, before waiting on it — they do not depend on it |
| reader 0, bias scatter | the bias read | waited only after the first row batches are issued (rows in flight cover it) |
| reader k, weights push | weight faces | waited between row batches, pushed once before the first tile row is pushed |
| compute, each tile row | that tile row's 64 rows | nothing independent: the k order is fixed (k = 0 then 1) because reordering the fp32 adds by arrival would make the result non-deterministic; expansion of all weights is done before the first tile row |

### Block schedule

```cpp
// per core: one block = the whole output shard (core_row_tiles x W tiles), both choices
load_index_block();            // readers: index slice of choice k (core_row_tiles x 32 ids), 1 read / reader
prefetch_weights_block();      // readers: faces 0, 2 of tiles (k, tb) for the shard; reader 0 also the bias row
build_bias_tiles();            // reader 0: scatter W·32 fp32 bias values into W row-0 tiles (once)
expand_weights_block();        // compute: 2·core_row_tiles column-broadcast fp32 tiles (once)
for (uint32_t row_idx = 0; row_idx < core_row_tiles; ++row_idx) {   // streaming quantum: 1 tile row
    gather_rows_block(row_idx);        // readers: 32 segments of W·64 B per choice -> cb_rows_choice{k}
    combine_tile_row_block(row_idx);   // compute: per DEST group: seed bias, += r0·w0, += r1·w1, pack to shard
}
```

| Block operation | Block shape | Resident across it | Intended frequency of inits / barriers / mcast |
|-----------------|-------------|--------------------|------------------------------------------------|
| `load_index_block` | core_row_tiles x 32 uint32 per choice (contiguous `core_row_tiles·128` B of page k) | `cb_index_choice{k}` for the whole kernel | 1 DRAM read per reader per core; waited by its own transaction id with the weight reads issued behind it (no full barrier) |
| `prefetch_weights_block` | core_row_tiles tiles per choice, 2 x 1 KB reads a tile (faces 0, 2) | `cb_weights_raw_choice{k}` until expansion | 2·core_row_tiles reads per reader per core, under one transaction id; one push of core_row_tiles pages |
| `build_bias_tiles` | W tiles (row 0 of faces 0, 1) from one read of W·128 B | `cb_bias_tiles` whole kernel | 1 DRAM read + 2·W word-group copies by reader 0, once per core; one push |
| `expand_weights_block` | 2 x core_row_tiles fp32 tiles | `cb_weights_bcast` whole kernel | one init (unary column broadcast) + one pack reconfig to fp32, once per core; then one pack reconfig to bf16 for the main loop |
| `gather_rows_block` | 32 tokens x W tiles per choice (32 segments of W·64 B) | the whole shard stays in `cb_rows_choice{k}` | no per-read barrier; reads in batches of `batch_reads` under `batch_depth` rotating ids; one push of W pages per tile row; no index reload, no full barrier inside the loop; one final full barrier per reader |
| `combine_tile_row_block` | 1 tile row x W tiles, in DEST groups of n ≤ DEST_AUTO_LIMIT | `cb_bias_tiles`, `cb_weights_bcast` | per DEST group: 1 init of the bias row-broadcast seed, 1 init of the tilize-A x B multiply (shared by k = 0 and 1), no data-format reconfig, one pack of n tiles; one pop of W pages per choice per tile row; no multicast, no semaphores anywhere |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 reads in flight | 8 a RISC (16 a core) is the incumbent's single-RISC optimum; two RISCs on two NoCs with no write traffic may sustain more | `batch_reads x batch_depth` = 8x2 and 4x4 per RISC; 2x4 |
| L2 read issue path | stateful issue (src coordinate + address registers per read, size fixed) is the default; the TensorAccessor path costs ~63 cycles a read | `noc.async_read` through TensorAccessor (incumbent path) — expected slower; and a per-batch sticky trid (MOE_GLUE §15: 0.42 -> 0.34 us a batch) |
| L3 DEST sync | half-sync (4 fp32 tiles a group, pack overlaps math) costs 2 inits per 4 tiles | `dst_full_sync_en=True`: 8-tile groups (one group per tile row at W = 8), no math/pack overlap |
| L4 bias seed | the row-broadcast seed needs its own init per DEST group | bias as a third tilize-A x B multiply (a 32-row replicated row-major bias block x an fp32 ones tile): one init per core, +1 HiFi4 multiply a tile |
| L5 weight expansion granularity (overlap) | expanding all weights before the first tile row makes the first tile row wait on every weight read | expand per tile row (2 tiles) just before `combine_tile_row_block` |
| L6 rows CB depth | whole shard resident (≤ 96 KB a choice) is the default; L1 is not the constraint | `rows_depth_tile_rows = 2` (double-buffered tile rows) — expected equal |
| L7 bias tile format | fp32 bias tiles avoid a srcB reconfig per DEST group at +W·2 KB L1 | bf16 bias tiles (from a bf16 prepared row) with the srcB reconfig |

Grid-synchronization lamp: not applicable (no collective, no multicast, no barrier across cores). Overlap lamp: the whole-shard block is streamed per tile row, so compute overlaps the reads (L5 covers the one serialization point).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| index slice: DRAM -> `cb_index_choice{k}` | uint32 words | one `core_row_tiles·128` B read from `combine_index` page k at byte offset `tr0·128` (TensorAccessor, DRAM interleaved), own transaction id | reader k only; tr0 = first flat tile row of the shard |
| weights: DRAM -> `cb_weights_raw_choice{k}` | fp32 tile (faces 0, 2 written; faces 1, 3 untouched) | 2 reads of 1 KB per tile (bytes 0..1023 -> slot +0, bytes 2048..3071 -> slot +2048) from page `k·(T'/32) + tb` | the column-broadcast unpack reads only faces 0 and 2 |
| bias: DRAM -> staging -> `cb_bias_tiles` | fp32 | reader 0: one read of `W·128` B from prepared page `core_index % 8` at offset `g·W·128` into tile 0 bytes 2048.., then per tile j copy cols 32j..+15 (64 B) to tile j bytes 0..63 and cols 32j+16..+31 to bytes 1024..1087 | the row-broadcast unpack reads only row 0 of faces 0 and 1 |
| rows: L1 (110 banks) -> `cb_rows_choice{k}` | bf16 row-major block: 32 rows x W·32 values a tile row (row i at offset i·W·64) | reader k: per token i of tile row tr: `row = idx[tr·32+i]`; bank `row % 110`, bank offset `(row / 110)·1536 + g·W·64`; one-packet stateful read of `W·64` B; batches of `batch_reads` under rotating trids, push W pages when the tile row's last batch retires | reader 0 on RISCV_0 / NOC0 (choice 0), reader 1 on RISCV_1 / NOC1 (choice 1); bank count 110 and page stride are compile-time |
| weights expansion: `cb_weights_raw_choice{k}` -> `cb_weights_bcast` | fp32 tiles, every column = column 0 (TF32-valued) | unary column broadcast into DEST (fp32), pack fp32 | weights at TF32 exactly as in the incumbent's srcB; lossless afterwards |
| combine: rows + bias + weights -> DEST -> `cb_output_shard` | DEST fp32 -> bf16 | per DEST group: bias row-broadcast seed; `unpack_tilizeA_B` (rows block, tile j of the row; bcast weight tile (tb,k)) + ELWMUL (NONE) HiFi4 accumulate-to-DEST for k = 0 then k = 1; pack n tiles bf16 (the one rounding) | output CB backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor`): zero-copy, no writer, no NoC write |

Placement as a TARGET axis (`output_layout`): `norm_layout` is the physical realization of the built logical split (token rows x hidden slices, both independent) — a **knob-turn** placement, no new algorithm. `l1_interleaved` / `dram_interleaved` are the `interleaved_output` regime (deferred): a write stage, not a different algorithm. `rows_buffer = dram` is a row-address change (`owner_computes_dram_rows`).

Inter-Tensix contract of the deferred `group_shared_index_weights`: member 0 of a row group (cores i, i+1, .., i+G-1 in row-major order; not a rectangle) unicasts the index slice and the weight faces into the same CB offsets on each peer (CBs sit at the same L1 address on all cores) and increments one semaphore per peer per choice; a peer's reader k waits for the semaphore instead of its DRAM reads.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output shard (the block): `core_row_tiles x W` tiles, both choices |
| Grid | exactly the cores of `memory_config.nd_shard_spec.grid`, row-major order (`ttnn.corerange_to_cores(grid, None, True)`); 64 / 96 / 96 cores at 2x512 / 4x512 / 8x512 |
| Per-core work | shard i on core i: `G = 24 / W`, `g = i % G`, `rg = i // G`, `nSB = ceil(St / r)` with `St = ceil(S / 32)`, `b = rg // nSB`, `sb = rg % nSB`; flat tile rows `tr0 = b·St + sb·r`, `core_row_tiles = min(r, St - sb·r)`; hidden tiles `g·W .. g·W + W - 1` |
| Remainder | ragged window (B = 1, S % 32 ≠ 0): the last tile row holds padding tokens (I = 0, W = 0) and is computed like any other (its padding rows come out as exactly the bias); partial last shard (St % r ≠ 0, e.g. 1x3904, 1x4000): `core_row_tiles < r`, same kernels, fewer pushes. `tensor_token_tiles = ceil(T / 32)` per image, never floor. |

Host plan (single source of truth, computed once per (T, B, R, memory_config, rows buffer, compute config) and cached; a call only rebinds addresses): `shard_row_tiles`, `shard_col_tiles`, core list, per-core (`tr0`, `core_row_tiles`, `g`, bias copy), CB descriptors, CT args. RT args carry the four input addresses; the output binds through its shard-backed CB. As built (Refinement 4): the whole ProgramDescriptor is cached per that key, and there are no per-core RT args (generic_op re-sends each one on every call, ~0.3 µs of host time a core per kernel). The readers' common RT args are [4 addresses, N core records, N NoC coordinates]. The compute's are [N core records]. Core i = y·11 + x of the norm layout's row-major grid reads record i (`tr0 | core_row_tiles | g | bias copy`, 8 bits each). A call patches only the readers' common args 0..3 and the output CB's buffer.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_rows_choice0` | 0 | 2048 (one tile's worth of row-major values) | `rows_depth_tile_rows · W` (Phase 0: r·W) | spans hidden (W) and token rows (whole shard); streams per tile row | Float16_b | reader (choice 0) | compute | kernel |
| `cb_rows_choice1` | 1 | 2048 | `rows_depth_tile_rows · W` | as above | Float16_b | reader_noc1 (choice 1) | compute | kernel |
| `cb_weights_raw_choice0` | 2 | 4096 | r | spans token rows; 1 tile per tile row (faces 0, 2) | Float32 | reader (choice 0) | compute | until expansion |
| `cb_weights_raw_choice1` | 3 | 4096 | r | as above | Float32 | reader_noc1 | compute | until expansion |
| `cb_weights_bcast` | 4 | 4096 | 2·r | spans token rows x choice | Float32 | compute (pack) | compute (unpack) | kernel |
| `cb_bias_tiles` | 5 | 4096 | W | spans hidden; constant over token rows | Float32 | reader (choice 0) | compute | kernel |
| `cb_index_choice0` | 6 | 128 | r | spans token rows; reader-private scratch (no handshake) | UInt32 | reader (choice 0) | reader (choice 0) | kernel |
| `cb_index_choice1` | 7 | 128 | r | as above | UInt32 | reader_noc1 | reader_noc1 | kernel |
| `cb_output_shard` | 16 | 2048 | r·W (the shard; backed by the output buffer) | spans token rows x hidden: it is the output | Float16_b | compute | — (the output tensor) | kernel |

"reader" and "reader_noc1" are two instances of `kernels/nomic_moe_combine_reader.cpp` (CT `choice` = 0 / 1; RISCV_0 + NOC_0 / RISCV_1 + NOC_1). No writer kernel. Full ledger: `l1_ledger.md`.

Fallback realization R2 only (see Key Risks): `cb_rows_tilized` (index 8, 2048 B, `2·W` pages, Float16_b, compute -> compute) holds one tile row of both choices tilized; absent under R1.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_index_block` | core_row_tiles x 32 ids | no (dataflow) | DRAM `combine_index` page k | `cb_index_choice{k}`, core_row_tiles | resident, read by the same reader |
| 2 | `prefetch_weights_block` | core_row_tiles tiles x 2 faces | no (dataflow) | DRAM `combine_weights` | `cb_weights_raw_choice{k}`, core_row_tiles pushed once | waited by compute in #4 |
| 3 | `build_bias_tiles` | W tiles | no (dataflow) | DRAM prepared bias | `cb_bias_tiles`, W pushed once | resident, popped at kernel end |
| 4 | `expand_weights_block` | 2·core_row_tiles tiles | yes (`unary_bcast` / UnaryBcast<Col>) | `cb_weights_raw_choice0/1`, core_row_tiles each, waited then popped | `cb_weights_bcast`, 2·core_row_tiles (tile 2·tb + k) | raw CBs empty; bcast resident until kernel end |
| 5 | `gather_rows_block` | 32 x W·32 values per choice | no (dataflow) | L1 `expert_rows` via ids in `cb_index_choice{k}` | `cb_rows_choice{k}`, W per tile row | one tile row pushed per choice |
| 6 | `combine_tile_row_block` | 1 x W tiles, DEST groups of n | custom block op (see API Mapping) | `cb_rows_choice0/1` W each (waited, popped per tile row); `cb_bias_tiles` W (resident); `cb_weights_bcast` (resident) | `cb_output_shard`, W per tile row (reserve/push per DEST group) | shard filled tile row by tile row; nobody pops it |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | raw_api | `compute_kernel_hw_startup(cb_weights_raw_choice0, cb_weights_bcast)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h` | first statement of the compute kernel | - | - | - |
| `expand_weights_block` | helper | `compute_kernel_lib::unary_bcast<BroadcastDim::Col, input(cb_weights_raw_choice{k}), output(cb_weights_bcast)>(IterationShape::tiles(core_row_tiles), in, out)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp:24-34`; `BroadcastDim` `eltwise/api/chain.hpp:330` | one call per choice; output pack format Float32 | `cb_weights_raw_choice{k}` | `cb_weights_bcast` | `core_row_tiles` (tile count) |
| `combine_tile_row_block` — bias seed | raw_api | `unary_bcast_init<BroadcastType::ROW>(cb_bias_tiles)`, `unary_bcast<ROW>(cb_bias_tiles, j0 + d, d)` | `tt_metal/hw/inc/api/compute/bcast.h:38, :83` | per DEST group: init once, n tiles | `cb_bias_tiles` | DEST slots 0..n-1 | n = `dest_group_tiles` |
| `combine_tile_row_block` — weighted accumulate (R1) | raw_api | unpack: `llk_unpack_tilizeA_B<false, false /*reload_srcB*/, false, false>(cb_rows_choice{k}, cb_weights_bcast, j0 + d, 2·tb + k, W)` after `llk_unpack_tilizeA_B_init<...>(cb_rows_choice0, cb_weights_bcast, W)`; math: `llk_math_eltwise_binary_init<ELWMUL, BroadcastType::NONE, MATH_FIDELITY>(..., acc_to_dest = 1)` + per tile the ELWMUL NONE execute into DEST slot d | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_tilize_api.h:181, :226, :307`; `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_tilize.h:475, :529` (srcB one face per srcA face, `:587`); `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h:232` | `ct_dim = W` (rows-block width), operand A = tile j of the row-major block, operand B = expanded weight tile | `cb_rows_choice{k}`, `cb_weights_bcast` | DEST slots 0..n-1 (accumulated) | `W` (ct_dim), n |
| `combine_tile_row_block` — pack | raw_api | `pack_tile(d, cb_output_shard)` x n between `tile_regs_commit/wait/release`; `cb_reserve_back/push_back(n)` | `tt_metal/hw/inc/api/compute/pack.h` | Float16_b pack (the one rounding; packer ties away from zero) | DEST | `cb_output_shard` | n |
| DEST capacity | helper | `DEST_AUTO_LIMIT` | `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:103` | n = min(DEST_AUTO_LIMIT, W - j0) | - | - | `dest_group_tiles` |
| `gather_rows_block` | helper + raw_api | `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid`; per read a one-packet stateful issue (`noc_async_read_one_packet_set_state` once a RISC for size / VC, then per read the source coordinate + address and the destination) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147, :158`; `tt_metal/hw/inc/api/dataflow/dataflow_api.h:575, :599, :2494, :2518` | `batch_reads`, `batch_depth`; segment `W·64` B | L1 `expert_rows` | `cb_rows_choice{k}` | `batch_reads`, `batch_depth`, W |
| `load_index_block`, `prefetch_weights_block`, `build_bias_tiles` | helper | `TensorAccessor` + `Noc::async_read` (DRAM interleaved), trid-tagged | `tech_reports/tensor_accessor/tensor_accessor.md`; incumbent usage `combine_reader.cpp:116-125` | page ids / offsets above | DRAM | `cb_index_*`, `cb_weights_raw_*`, `cb_bias_tiles` | `core_row_tiles` |
| R2 fallback tilize | helper | `compute_kernel_lib::tilize<W, cb_rows_choice{k}, cb_rows_tilized>(in, out, 1)` + `mul_tiles_bcast_cols` with acc_to_dest | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:194-208`; `tt_metal/hw/inc/api/compute/bcast.h:178` | as the incumbent `combine.cpp:61-101` | `cb_rows_choice{k}` | `cb_rows_tilized` | W |

Helpers considered and rejected:

| raw_api entry | Candidate helper | Mismatch (file:line) | Reason |
|---------------|------------------|----------------------|--------|
| bias seed + weighted accumulate (the DEST group) | `eltwise_chain` with `UnaryBcast<Row>` + `BinaryFpu<Mul, ..., Col, Dst, DestAccumulation>` + `PackTile` | `eltwise/api/chain.hpp:302-312` (DestAccumulation = accumulate **across the walk into one tile**, PerRow / WholeShape); `eltwise/core/chain.inl:1391-1421` (`init()` per element, acc_to_dest derived from that reduction mode, dst = DstSlot without slot offset when accumulating) | the needed shape is n DEST slots, each seeded and then accumulated twice in place, with the inits hoisted over the group: the chain expresses it only as a per-tile chain with three per-element inits per tile — the fixed cost this design removes |
| weighted accumulate with tilize in the unpack | `tilize_helpers.hpp:194` `tilize` | `tilize_helpers.hpp:194-208` writes tiles to an output CB | it is the separate tilize round trip (unpack, pack, unpack again) that principle 1 removes; kept only as the R2 fallback |
| weighted accumulate | `mul_tiles_bcast_cols` / `BinaryFpu<..., BroadcastDim::Col>` with an unpack-tilized A | `tt_llk_blackhole/llk_lib/llk_unpack_AB.h:88-106` (COL: srcB loaded once per face row) vs `llk_unpack_tilize.h:587` (tilizeA_B loads one srcB face per srcA face); `llk_math_eltwise_binary.h:280-290` (COL ELWMUL clears srcB once per face row) | the COL-broadcast math expects srcB once per face row, the tilize-A unpacker supplies one per face: the pairing needs NONE broadcast against a pre-expanded weight tile, hence `expand_weights_block` |
| `gather_rows_block` | `Noc::async_read` through `TensorAccessor` (as the incumbent) | `routed_combine/.../read_batches.hpp:79-87` | ~63 cycles a read (MOE_GLUE §4), the issue bound this op exists to cut; stateful issue keeps the per-read work to the source coordinate + address and the destination |

Gap record: no kernel-library helper pairs `unpack_tilizeA_B` with an eltwise-binary math stage (only with reduce: `tt_metal/hw/inc/api/compute/tilize.h:100` `tilizeA_B_reduce_init`). The implementer builds the block operation in the op's compute kernel (or as an additive kernel_lib helper, existing users unchanged).

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|----|-------------------|-------------------|---------------|
| expand weights | unary broadcast | `cb_weights_raw_choice{k}`: Col0 (faces 0, 2) | - | Col |
| bias seed | unary broadcast | `cb_bias_tiles`: Row0 (row 0 of faces 0, 1) | - | Row |
| weighted accumulate | ELWMUL acc_to_dest | `cb_rows_choice{k}` (tilized in unpack): All | `cb_weights_bcast`: All (every column = col 0) | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| R1 pairing (`unpack_tilizeA_B` + ELWMUL NONE acc_to_dest, HiFi4, fp32 DEST, bf16 srcA x fp32/TF32 srcB) has no existing user | only reduce math is paired with it today; the per-face dvalid handshake and the mixed formats are unproven | first implementation step is a one-tile probe checked bit for bit with one-hot weights; if it fails, realization R2 (tilize into `cb_rows_tilized` + `mul_tiles_bcast_cols` on the raw weights, the incumbent's compute) — same blocking, +`2·W` tiles of L1, no expansion needed |
| transaction-id barrier race | `noc_async_read_barrier<TXN_ID>` can pass before the most recent read under that id left the command buffer (MOE_GLUE §4) | wait for a batch by id only with a later read behind it; the last batch and any id waited immediately after issue use the full barrier (or a command-buffer-ready wait) |
| landing slot vs CB write pointer | with reads in flight across tile rows the CB write pointer lags | address a tile row's slot by its position in the shard (`row_idx·W·2048`), not by `get_write_ptr()`; push in order |
| weight tiles' faces 1, 3 and bias rows 1..31 are never written | the broadcast unpacks must not read them | the column broadcast reads faces 0, 2; the row broadcast reads row 0 of faces 0, 1 (the incumbent's probes, MOE_GLUE §2); verified by the one-hot exact test |
| NaN rows | rows no index points at hold NaN in the tests | only indexed rows are read; padding tokens index row 0, a real slot's row, with weight 0 ⇒ 0·finite = 0, padding rows = bias exactly |
| ND-shard page order | a partial last shard (1x3904, 1x4000) still occupies a full shard slot | pack sequentially from page 0 of the core's single shard; validate one shard per core |
| index DRAM hot spot | all owners read their slices from the same 2 DRAM pages at kernel start | one read per reader (not per token block); weights and bias issued behind it |
| precision | a bf16 intermediate or weights rounded to bf16 fails the 0.0019 relative-RMS bar | everything stays in fp32 DEST until the one pack; weights pass through srcB at TF32 exactly as in the incumbent; HiFi4 + fp32_dest_acc_en enforced by validate |
| default memory_config | `L1_MEMORY_CONFIG` is outside Phase 0 SUPPORTED | SupportRefusal with a clear message; the model always passes norm_layout |

## Registry

| Declaration | Value |
|-------------|-------|
| `INPUT_TAGGERS` | `tag_window(inputs, axes)`: T = B·S from inputs[0]; "aligned" if T % 32 == 0 else "ragged". `tag_batch(inputs, axes)`: "single" if B == 1 else "multi" |
| `SUPPORTED` (Phase 0) | dtype [bfloat16]; rows_buffer ["l1"]; output_layout ["norm_layout"]; window ["aligned", "ragged"]; batch ["single", "multi"] |
| `EXCLUSIONS` | none |
| Not supported (TARGET − SUPPORTED) | rows_buffer "dram" (regime `owner_computes_dram_rows`, deferred); output_layout "l1_interleaved", "dram_interleaved" (regime `interleaved_output`, deferred) |
| `validate()` (first line of the entry point) | per the requirements' list: shapes / dtypes / layouts / memory of all inputs, 2240 ≤ R ≤ 8512, T' = ceil(T/32)·32, 1000 ≤ T ≤ 4096, T % B == 0, B == 1 or S % 32 == 0, bias is a `PreparedCombineBias` of the expected tensor, compute grid 11x10, fp32_dest_acc_en and HiFi4, the `owner_computes` predicate on memory_config, `circular_buffer_bytes ≤ 384 KB`. Every refusal message starts with `nomic_moe_combine:` |

Structural impossibilities: none beyond feature_spec.py's (INVALID is empty and stays empty).

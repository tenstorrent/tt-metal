# Operation Design: g6_to_memory_config

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Return a tensor holding the input's stored tile pages bit for bit, in the input's shape, dtype and TILE layout, placed exactly as `memory_config` names. One device program per call; every page read once from its source placement and written once to its target placement, through the L1 of the core that moves it. |
| Math | `y[p] = x[p]` for every tile page `p` (bit copy of the whole page, padding rows included); `shape(y)=shape(x)`, `dtype(y)=dtype(x)`, `layout(y)=TILE`, `placement(y)=memory_config` |
| References | `eval/golden_tests/g6_to_memory_config/feature_spec.py` (105 production cases, TARGET); `eval/golden_tests/g6_to_memory_config/helpers.py` (exact grading, placement check); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `tech_reports/tensor_accessor/tensor_accessor.md` (+ `tensor_accessor_iterator.md`); `tt_metal/hw/inc/api/tensor/tensor_accessor.h`; `tt_metal/impl/buffers/buffer_distribution_spec.cpp` (shard→core rule); `tt_metal/impl/allocator/l1_banking_allocator.cpp:68-111` (L1 bank→core rule); perf catalog `ttnn/ttnn/operations/examples/{double_buffer,trid_double_issue,read_coalescing,split_reader,noc_placement,bank_stagger,tensix_all_reduce}` |

Device facts this design was checked against (probe on this Blackhole board): `compute_with_storage_grid_size = 11×10 = 110` cores; **110 L1 banks, one per worker core, bank offset 0, bank ids a seeded shuffle of the worker grid** (`l1_banking_allocator.cpp:80-85`); 8 DRAM banks; L1 alignment 16 B, DRAM alignment 64 B; `NOC_MAX_BURST_SIZE = 16384 B` on Blackhole (`noc_parameters.h:290`), 8192 B on Wormhole. Page sizes: bfloat16 2048 B, bfloat8_b 1088 B (17×64), bfloat4_b 576 B (9×64) — every page is a multiple of both alignments, so `aligned_page_size == page_size` in every dtype.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes (positional) | on device, TILE, rank 2–4, any placement with a ROW_MAJOR shard spec when sharded | — | buffer address → common RT arg; geometry/placement → CT args (TensorAccessorArgs) + per-core RT args |
| `memory_config` | `ttnn.MemoryConfig` | yes (positional) | DRAM or L1; INTERLEAVED, or HEIGHT/WIDTH/BLOCK sharded with a ROW_MAJOR shard spec whose shard shape is a multiple of 32 and whose grid has ≥ as many cores as shards (BLOCK: one CoreRange, shard grid fits the range) | — | output buffer address → common RT arg; placement → CT args (TensorAccessorArgs) + per-core RT args |

No dtype, layout or compute-config parameter: the op never casts and does no arithmetic.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(..., H, W)`, rank 2–4; H ≥ 1 any value; W any value (production: multiple of 32) |
| Dtype | Phase 0: bfloat16. TARGET adds bfloat8_b, bfloat4_b (same code path, page size from the accessor) |
| Layout | TILE |
| Memory | any: DRAM/L1 × INTERLEAVED/HEIGHT/WIDTH/BLOCK sharded (ROW_MAJOR shard spec) |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` (allocated with `ttnn.allocate_tensor_on_device(ttnn.Shape(input.shape), input.dtype, TILE, device, memory_config)`, positional args) |
| Dtype | `input_tensor.dtype` |
| Layout | TILE |
| Memory | exactly `memory_config` (buffer type, memory layout, shard grid with its ranges in the given order, shard shape, orientation) |

Tile geometry (single source, host, alignment-aware, per image): `tensor_tile_rows = prod(shape[:-2]) * ceil(H/32)`, `tensor_tile_cols = ceil(W/32)`, `tensor_pages = tensor_tile_rows * tensor_tile_cols`, page id `p = r * tensor_tile_cols + c`. This is the 2D page view every placement uses: `BufferDistributionSpec::from_shard_spec` folds every leading dim into the outer shard dim (`squeeze_shape_ranks`, `buffer_distribution_spec.cpp:399-444`), so shards may span image boundaries and the shard grid is `ceil(tensor_tile_rows/shard_tile_rows) × ceil(tensor_tile_cols/shard_tile_cols)` for all three schemes. A partial last tile row of an image (`h_non_aligned`) is still one whole page; nothing in the op depends on H beyond `ceil(H/32)`.

---

## Blocking Model

The first design section. Everything below it is downstream of these decisions.

**The scheme in one sentence:** every page is moved by a core that holds it in its own L1 — the
target's owner pulls it (`noc_async_read` remote→local), or the source's owner pushes it
(`noc_async_write` local→remote) — so each page crosses the NoC exactly once and no staging buffer
exists; only when neither end is in L1 (DRAM→DRAM) does a full-grid staged mover exist, because a
NoC transaction cannot go DRAM→DRAM.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `page_row` — leading dims folded with tile rows, extent `tensor_tile_rows` | independent — each output page is a copy of exactly one input page; no result spans rows | `core_rows` (rows of the owner's shard inside the tensor) | the whole owner shard: `min(shard_tile_rows, tensor_tile_rows − r0)` — one block | owner side's shard spec (host derives per core) → per-core RT args `r0`, implied by `seq_end` | **shard_pull / shard_push:** by the owner side's shard spec (the shard *is* the core assignment). **bank_* / staged:** this axis is flattened into `page_linear` | knob-turn |
| `page_col` — extent `tensor_tile_cols` | independent — same reason | `core_cols` | the whole owner shard: `min(shard_tile_cols, tensor_tile_cols − c0)` — one block | same → per-core RT args `c0`, `core_cols` | as `page_row` | knob-turn |
| `page_linear` — row-major flattening `p`, extent `tensor_pages`; the native axis of an interleaved placement | independent — same reason | **bank_*:** `core_pages`; **staged:** `staged_block_pages` | **bank_*:** the whole bank, `ceil((tensor_pages − b)/NB)` pages, one block. **staged:** `min(core_pages, 16)` (overlap lamp, see below) | **bank_*:** derived on device from CT arg `tensor_pages` and the discovered bank id `b`. **staged:** host constant `STAGED_BLOCK_PAGES = 16` → CT arg | **bank_*:** cyclic `p mod NB` — fixed by the L1-interleaved placement, every worker core owns one bank. **staged:** `ttnn.split_work_to_cores(full grid, tensor_pages, row_wise=True)` contiguous page ranges | knob-turn |
| `intra_tile` — 32×32 datums, 4 faces, 16-datum shared-exponent groups (block formats) | independent but **atomic** — a block-format page is only bit-exact when moved whole | `page` | 1 whole tile — the *only* axis at extent 1, required by the bit-exact contract: cutting a face row or a 16-datum group changes bfloat8_b/bfloat4_b bits | mechanism cap (not a knob) | never split | none |
| `mover` — second-level split of one core's page sequence across its two data-movement RISCs (not a data axis) | — | `movers_per_core` | **2**: mover 0 on NCRISC/NoC0, mover 1 on BRISC/NoC1; mover `m` takes sequence slice `[m·half, min(n,(m+1)·half))`, `half = ceil(n / movers_per_core)` | host constant `MOVERS_PER_CORE = 2` → number of KernelDescriptors + per-mover RT slice + CT arg `mover_index` | contiguous halves of the core's page sequence (keeps runs contiguous) | knob-turn |

No axis is dependent and no operand is reuse-shared: there is one input, and under every split each
core touches a disjoint set of its pages (operand-reuse check: the input varies along every split
axis — no broadcast row exists).

Intermediate data: the owner regimes have none (the page lands in its final place). The staged
regime's only intermediate is the page block in `cb_staged_pages`, which has the same `page_linear`
axis and inherits its assignment. No core collects, coordinates or hands out work; there is no
idle-while-one-core-works stage.

**Interleaved default, applied:** step 1 (occupancy) — bank_* uses every worker core (all 110
banks), staged uses the full grid; step 2 (coarsest block) — each mover's whole assignment is one
block with one barrier. **Sharded default, applied:** one block = the whole resident shard,
split only across the two movers; never re-chunked (there is no L1 pressure: the owner regimes
allocate no op buffer at all).

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| (owner regimes: no CB) | in-flight depth = every transfer of the mover is issued before its single barrier | all of the mover's runs | the full assignment is in flight at once; the destination memory itself is the landing buffer, so there is nothing to double-buffer (catalog `double_buffer` report: barrier-per-read 6.5 GB/s vs ≥4 in flight 17.9 GB/s; `trid_double_issue`: more blocks per barrier is the biggest lever) |
| `cb_staged_pages` (staged only) | `staged_buffer_depth` | 2 (host constant `STAGED_BUFFER_DEPTH`) | reader fills block k+1 while writer drains block k (catalog `double_buffer`: NoC0-reader/NoC1-writer overlap 13 → 17.9 GB/s per core) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Block-format page atomicity (shared exponent per 16 datums) | `intra_tile` | every transfer is a whole number of whole pages (`n · aligned_page_size` bytes from a page boundary) | bfloat8_b/bfloat4_b bits change silently (passes a loose compare, fails the exact one) |
| Transfer-run contiguity | run length along the owner's walk | a run is the maximal page sequence contiguous on **both** ends: on a sharded end it stops at the shard row edge (`TensorAccessor::num_contiguous_pages`, `tensor_accessor.h:187-192`); on an interleaved end, or a sharded end with `contiguous_page_stride() != 1` (1-tile-wide shard), it is 1 page (`tensor_accessor.h:562`: stride = NB); Phase 0 runs never cross a row of the owner's shard | bytes land at wrong addresses — silent wrong placement, values read back wrong |
| Local shard addressing | local offset inside the owner's shard | offset of `(r, c)` = `((r − r0) · shard_tile_cols + (c − c0)) · aligned_page_size` with the **nominal** `shard_tile_cols`, also for the uneven last shard | pages of the uneven last shard (e.g. 40 of 52 columns, Qwen 2071) misplaced |
| Shards per core | owner block = exactly one shard | host refuses (ValueError) a grid with fewer cores than shards; more cores than shards are legal, empty cores get no kernel | a second shard on a core would never be moved |
| GRID_2D (BLOCK_SHARDED) core rule | shard (i, j) → core | grid must be one CoreRange with `ceil(tc/sc) ≤ width`, `ceil(tr/sr) ≤ height` (`buffer_distribution_spec.cpp:174-196`); host refuses otherwise (ValueError) before allocation | allocation `TT_FATAL` (RuntimeError) — or, if the host derived cores differently from the buffer, wrong placement |
| L1 bank ownership | bank_* core set | launch on the full `compute_with_storage_grid`; host checks `l1_num_banks == grid.x·grid.y` once per device (`ttnn._ttnn.reports.get_device_info(device).l1_num_banks`, cached) and refuses (RuntimeError) otherwise; kernel asserts (`--dev`) that it found its bank | pages of a bank on an unlaunched core are never written |
| NoC transfer size | run bytes | use the multi-packet entry points (`noc_async_read`/`noc_async_write` with default `max_page_size`, `dataflow_api.h:546-555`, `:876`), which split above `NOC_MAX_BURST_SIZE`; never the `one_packet`/stateful variants for runs. Largest Phase 0 run: one shard row, ≤ 64 pages × 2048 B = 128 KB < the 1 MB valid range | a one-packet call above 16 KB (8 KB on Wormhole) truncates the transfer |
| CB ring wrap (staged) | `cb_staged_pages` push/pop count | reader pushes and writer pops the **nominal** `staged_block_pages` on every block including the ragged last one; only the NoC transfer carries the actual count | ring pointers drift → hang or reordered pages |
| Kernel-config region (BB-04) | per-core RT args | ≤ 5 per-core RT words (shard_*), 0 (bank_*), 2 (staged); 2 common RT words; no per-page lists | launch refused in profiler / NoC-trace builds |
| L1 budget (BB-05) | output L1 per core + op buffers | host refuses (ValueError) when `output_l1_bytes_per_core + cb_bytes > 1347776` (see `l1_ledger.md`) | launch-time OOM inside the model only |

### Regimes

Minimum, against the NoC/DRAM boundary: each page is read once from its source placement and
written once to its target placement; when at least one end is in L1, the mover is that L1's core,
so one of the two accesses is core-local and the page crosses the NoC **once**; when both ends are
DRAM the page must pass through some core's L1 (a NoC read lands only in local L1, a NoC write
departs only from local L1), so **two** NoC crossings is the structural minimum. DRAM ends are
crossed exactly once each in every built regime.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `shard_pull` | built | target is L1-sharded AND (source is not L1-sharded OR `num_shards(target) ≥ num_shards(source)`) | per target-shard core: the whole shard rectangle `core_rows × core_cols`, halved across movers; transfer unit = run | minimum: 1 NoC crossing per page (remote source → own L1, already at the target address) | block is already the whole assignment; its fixed cost — launch, RT-arg/accessor setup, one read-barrier round trip per mover — is paid once per call. Larger runs (not larger blocks) cut per-transaction issue cost |
| `shard_push` | built | source is L1-sharded AND (target is not L1-sharded OR `num_shards(source) > num_shards(target)`) | per source-shard core: the whole shard rectangle, halved across movers; transfer unit = run | minimum: 1 NoC crossing per page (own L1 → remote target) | as `shard_pull`, with one write barrier (acks) per mover per call |
| `bank_pull` | built | target is L1-interleaved AND source is not L1-sharded | per worker core: all pages of its L1 bank `b`: `p = b + k·NB`, `k < ceil((tensor_pages − b)/NB)`, halved across movers; transfer unit = 1 page (interleaved end) | minimum: 1 NoC crossing per page | as above plus the one-time bank discovery scan (≤ NB table compares) per mover per call |
| `bank_push` | built | source is L1-interleaved AND target is not L1 (DRAM, any scheme) | as `bank_pull`, source side | minimum | as `bank_pull` |
| `staged` | built | neither end is in L1 (DRAM → DRAM, any schemes) | per core: contiguous range of `core_pages` from `split_work_to_cores(full grid, tensor_pages, row_wise=True)`, cut into blocks of `staged_block_pages` | minimum for DRAM→DRAM: 2 NoC crossings per page (DRAM → staging L1 → DRAM), each DRAM end once | per block: one CB reserve/push/wait/pop handshake, one read-barrier round trip, one writes-flushed wait. Bounded by the overlap lamp: a whole-assignment block serializes all reads before all writes |
| `staged_coalesced` | deferred — successor of `staged` for DRAM width-sharded targets (bfloat4_b weights, PERF FOCUS Qwen 2183: 81920 pages of 576 B). Positive reason: no Phase 0 cell reaches it — `{out_scheme: width_sharded, out_buffer: dram}` is in EXCLUSIONS and bfloat4_b is outside SUPPORTED. Reachable: `staged` already moves these pages correctly; the successor only reshapes the block | `staged` ∧ target DRAM-sharded | block = `staged_block_rows × shard_tile_cols` inside one target DRAM shard (a DRAM shard is contiguous in its bank, so one write per block); cores assigned blocks of target bank j nearest that bank (`ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment`); reader on trids (`trid_double_issue`) | same bytes as `staged`; writes drop from one per 576 B page to one per block (64× fewer for shard width 64) | write-transaction count ÷ `shard_tile_cols·staged_block_rows`; the reads stay per page (interleaved source) |
| `helper_fanout` | deferred — positive reason: it doubles NoC bytes for the pages it forwards (owner L1 → helper → target) to buy issue parallelism, and the largest Phase 0 owner load is 64 pages per core = 32 per mover (Qwen 2072/2080/2135/2171), already halved by the two-mover split. Reachable: a helper is a `staged` mover whose source accessor is the owner's sharded tensor — same kernels, a different page range | regime ∈ {`shard_pull`, `shard_push`} ∧ `num_shards(owner) ≤ 16` ∧ `ceil(core_pages / movers_per_core) ≥ 32` (candidate thresholds; the deciding experiment is Qwen 2072, 8 owners × 64 pages, against Qwen 2074, 40 owners × 4 pages) | owner shard split between the owner and k helpers, each a contiguous slice of shard rows/columns | +1 NoC crossing per forwarded page | n/a — it shrinks each core's block to add cores; the trade is bytes for issue parallelism |
| `staged_everywhere` (any-core staging for placements with an L1 end) | rejected — superseded by `shard_*`/`bank_*` on every cell with an L1 end: 2 NoC crossings + a staging CB instead of 1 crossing and no buffer | — | — | +1 NoC crossing per page, + `2·block·page` L1 per core | — |
| `two_step_via_interleaved` | rejected — forbidden by the contract (third placement / second dispatch); superseded by every built regime | — | — | 2× every tier | — |
| `compute_unpack_repack` (pages through unpacker/packer) | rejected — changes bfloat8_b/bfloat4_b bits; no arithmetic exists to justify a compute kernel | — | — | — | — |
| `dependent_split_combine` | rejected — not applicable: no axis is dependent (`y[p] = x[p]`), there is nothing to combine | — | — | — | — |

**Regime-selection function (exact; host; also exported as `select_plan`):**

```
src_l1_sh = in.buffer == L1  and in.scheme  != INTERLEAVED
dst_l1_sh = out.buffer == L1 and out.scheme != INTERLEAVED
if dst_l1_sh and src_l1_sh: regime = shard_pull if num_shards(out) >= num_shards(in) else shard_push
elif dst_l1_sh:             regime = shard_pull
elif src_l1_sh:             regime = shard_push
elif out == L1 interleaved: regime = bank_pull       # source DRAM (any scheme) or L1 interleaved
elif in  == L1 interleaved: regime = bank_push       # target DRAM (any scheme)
else:                       regime = staged          # DRAM -> DRAM
num_shards(m) = ceil(tensor_tile_rows/shard_tile_rows(m)) * ceil(tensor_tile_cols/shard_tile_cols(m))
```

Production map (86 bfloat16 cases): `shard_pull` 24 (DRAM→L1 height 4, DRAM→L1 width 12, L1-interleaved→L1 width 3, width 8→40 cores 3, height→block 1→1 core 2); `shard_push` 35 (L1 height→DRAM 4, L1 width→DRAM 15, L1 width→L1 interleaved 7, L1 block→L1 interleaved 2, width 40→10/8 cores 7); `bank_pull` 27 (DRAM→L1 interleaved). Refinement dtypes: bfloat8_b 12 cases → `shard_push` 9, `bank_pull` 3; bfloat4_b 7 cases → `staged` (then `staged_coalesced`).

### Traffic ranking

| Candidate split | NoC crossings / page | DRAM crossings / tensor | L1 op buffers | Fan-out | Verdict |
|-----------------|----------------------|-------------------------|---------------|---------|---------|
| Owner moves (`shard_*` / `bank_*`): the core holding the page in L1 moves it | 1 | input 1 (if DRAM), output 1 (if DRAM) | none | unicast, 1 page → 1 place | **chosen** wherever an L1 end exists — cheapest possible |
| Staged over the full grid (`staged`) | 2 | 1 each end | `2 × block` pages | unicast | chosen only for DRAM→DRAM, where 2 is the minimum |
| Staged over the full grid for cells with an L1 end | 2 | 1 each end | `2 × block` pages | unicast | rejected (dominated) |
| Helper fan-out from few owner cores | 1 + (forwarded fraction) | 1 each end | helper staging | unicast | deferred (`helper_fanout`) |
| For L1↔L1 sharded pairs: pull by target owners vs push by source owners | 1 either way | 0 | none | unicast | equal bytes; decided by occupancy (more cores with data), tie → pull |
| Split a dependent axis + combine | — | — | — | — | not applicable: no dependent axis |
| Two-step through an interleaved intermediate | 2 (+ a second program) | 2× | intermediate tensor | — | rejected (contract) |

The cheapest-traffic split is the one implemented. Occupancy enters only as the tiebreak between
equal-byte owner sides (it does not override bytes). Stall-shadow check: the only waits are each
mover's single end barrier (every transfer of the mover is already issued before it — nothing
independent remains to schedule into it) and, in `staged`, the per-block read barrier and CB
handshakes; the next block's reads are independent of the current block's barrier and can be
issued into it with transaction ids — recorded as the `staged` overlap lamp, legal with no
algebraic condition because the blocks are disjoint pages. No reorder changes values (pure copy).

### Block schedule

Owner regimes (`shard_pull`, `shard_push`, `bank_pull`, `bank_push`) — per mover (2 per core),
`num_blocks_this_mover = 1`:

```cpp
locate_owned_block();                       // shard_*: decode RT slice; bank_*: discover own L1 bank b, derive k-range
for (uint32_t block_idx = 0; block_idx < num_blocks_this_mover; ++block_idx) {
    move_block(block_idx);                  // issue one NoC transfer per run: pull = read remote->own L1, push = write own L1->remote
    drain_block(block_idx);                 // one barrier: read barrier (pull) / write barrier with acks (push)
}
```

`staged` — per core:

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);                  // reader (NCRISC/NoC0): reserve nominal block, read pages, read barrier, push nominal
    store_block(block_idx);                 // writer (BRISC/NoC1): wait nominal block, write pages, writes flushed, pop nominal
}
drain_writes();                             // writer: one write barrier (acks) before return
```

| Block operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------------|-------------|--------------------|-------------------------------|
| `locate_owned_block` | — | — | once per mover per call. `bank_*`: one scan of ≤ NB (=110) bank-coordinate entries comparing against this core's own NoC coordinates (`my_x[noc_index]`, `my_y[noc_index]`) |
| `move_block` | `shard_*`: the mover's slice of `core_rows × core_cols` walked in shard row-major order; `bank_*`: the mover's slice of `k ∈ [0, core_pages)` | the source pages (input tensor, never written) and the destination region (output tensor) | one NoC command per run; one remote address computation (`TensorAccessor::get_noc_addr`) per run; local addresses stepped incrementally, never recomputed per page; no barrier, no CB handshake inside |
| `drain_block` | the whole mover slice | — | exactly one barrier per mover per call: `noc_async_read_barrier` (pull) or `noc_async_write_barrier` (push). `noc_async_writes_flushed` is **not** sufficient (BB-03) |
| `load_block` | `staged_block_pages` pages of the core's contiguous range | previous block in the other CB half | one `cb_reserve_back`/`cb_push_back` of the nominal block, one read barrier per block |
| `store_block` | same | next block filling | one `cb_wait_front`/`cb_pop_front` of the nominal block, one `noc_async_writes_flushed` per block |
| `drain_writes` | — | — | one `noc_async_write_barrier` per call |

Profiling zones (BB-21, compiled out of the plain build by `DeviceZoneScopedN`, `api/debug/kernel_profiler.hpp:1012-1047`): `G6-LOCATE`, `G6-ISSUE`, `G6-DRAIN` (owner movers); `G6-STAGED-RESERVE`, `G6-STAGED-READ`, `G6-STAGED-READ-BARRIER` (staged reader); `G6-STAGED-WAIT`, `G6-STAGED-WRITE`, `G6-STAGED-DRAIN` (staged writer). Every blocking wait is inside a zone of its own.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Two movers per core (`movers_per_core = 2`) | for decode shards of 4–8 pages the second RISC's launch + RT-arg fetch may cost more than the halved issue time; catalog `split_reader` saw 1.20–1.21× at 1–2 KiB only when issue-bound | `movers_per_core = 1` on Qwen 2074 (40 cores × 4 pages) and Qwen 2072 (8 × 64 pages) |
| Owner side for L1↔L1 sharded pairs chosen by core count (grid-synchronization lamp) | equal bytes either way; catalog `tensix_all_reduce`: pull beat push at 8 tiles/core (2721 vs 3080 ns) but became erratic above ~24 tiles/core | always-pull vs always-push on Qwen 2070 (40→10) and 2067 (8→40) |
| Bank discovery scan in `bank_*` | ≤ 110 compares ≈ a few hundred cycles — visible next to the smallest `bank_pull` cases (96 pages, Qwen 2089) | scan via direct bank-table compare vs. `TensorAccessor::get_noc_addr(b)` per entry; and, for `tensor_pages < NB`, `staged`-style any-core movers (2 crossings, no scan) |
| Run coalescing (sharded↔sharded) | runs up to one shard row (8 KB–32 KB in production); catalog `read_coalescing`: 3.24× merging small reads, only 1.11–1.16× at 1024 B, nothing measured above 4 KiB | page-at-a-time transfers vs. runs on Qwen 2070/2073; runs crossing shard rows where both ends are full-width contiguous |
| `staged` block 16 × depth 2 (overlap lamp) | catalog `double_buffer` flat for blocks 4–64; `trid_double_issue` best at depth 3–4 and ~122 GB/s/core with large CBs; a whole-assignment block would serialize reads before writes | 64 × 3, and 16 × 2 with transaction-id double issue, on the DRAM→DRAM regime pin |
| Full-grid occupancy for `bank_pull` (grid-synchronization lamp) | all 110 cores launch even when `tensor_pages < 110`; idle cores only scan and exit | launch on the first `tensor_pages` banks' cores is impossible without a host bank map — measure the idle-core launch cost instead |

Catalog evidence used per knob: `movers_per_core` ← `noc_placement`, `split_reader`, `double_buffer`; one barrier per mover ← `double_buffer`, `trid_double_issue`; run coalescing ← `read_coalescing` (invalid across interleaved pages — hence runs of 1 there); `staged` block/depth ← `double_buffer`, `trid_double_issue`; `staged` row-wise split ← `noc_placement`; bank stagger not adopted (`bank_stagger`: 1.006–1.07× on interleaved sources; the owner walks already start on different DRAM banks).

---

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Source page | whole tile page (bf16 2048 B; bf8b 1088 B; bf4b 576 B) | where the input placement keeps it: DRAM bank / L1 bank / L1 shard | never modified; read once |
| Pull (`shard_pull`, `bank_pull`) | same bytes | `noc_async_read(src_accessor.get_noc_addr(p), local_dst_addr, run_bytes)` | lands directly at the output's address in this core's L1 — the output shard / bank region is the landing buffer (zero-copy, no CB, no NoC re-read) |
| Push (`shard_push`, `bank_push`) | same bytes | `noc_async_write(local_src_addr, dst_accessor.get_noc_addr(p), run_bytes)` | reads the core's own input shard / bank region in place (zero-copy) |
| Staged (`staged`) | same bytes | reader: DRAM → `cb_staged_pages`; writer: `cb_staged_pages` → DRAM | the only path with an L1 buffer; DRAM→DRAM needs one |
| Completion | — | one read barrier (pull) / one write barrier (push, staged writer) per mover before return | BB-03: nothing in flight when the kernel returns |

Local address rules: `shard_*` — `local_base + ((q / core_cols) · shard_tile_cols + q % core_cols) · aligned_page_size` for sequence index `q` (nominal stride); `bank_*` — `local_base + k · aligned_page_size` (L1 bank offset is 0 on every bank); `local_base` is the owner-side tensor's buffer address (the same on every core). The implementer cross-checks both rules under `--dev` against the local-address part of the owner-side accessor's `get_noc_addr(p)`.

Remote addresses always come from the remote-side `TensorAccessor` (CT `TensorAccessorArgs`), which carries the buffer's own distribution spec — the device-side authority for every interleaved or sharded, DRAM or L1 placement, uneven last shard and multi-range grid included.

### TARGET placement values seen as regimes

| TARGET value (`in_*` / `out_*`) | Served by | Unlock class |
|---------------------------------|-----------|--------------|
| DRAM interleaved (in or out) | `shard_*` / `bank_*` / `staged` (remote end via accessor) | built |
| L1 interleaved (in or out) | `bank_*`, or `shard_*` when the other end is L1-sharded | built |
| L1 height / width / block sharded (in or out) | `shard_pull` / `shard_push` (the shard is the core assignment) | built |
| DRAM height / width / block sharded source | remote end via accessor in every regime (no production case) | built |
| DRAM height / block sharded target | `shard_push` / `bank_push` / `staged` (no production case) | built |
| DRAM width-sharded target | `staged` moves it correctly today; EXCLUSIONS keeps it out of Phase 0; speed comes from `staged_coalesced` | knob-turn (lift the exclusion) + deferred regime for perf |
| bfloat8_b, bfloat4_b | every regime unchanged — page size comes from the accessor; whole pages keep every exponent group | knob-turn (SUPPORTED) |

No TARGET value is a scheme-change: no axis is dependent, so no placement needs a combine.

**Structural impossibilities:** none found beyond `INVALID = []`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | `shard_*`: one owner shard (whole); `bank_*`: one L1 bank's pages (whole); `staged`: a block of `staged_block_pages` pages |
| Grid | `shard_*`: the owner side's cores with data, coalesced into maximal rectangles for the kernel's `core_ranges` (BB-17; e.g. Qwen 2058's 16 single-core ranges → 3 rectangles). `bank_*`: the full `compute_with_storage_grid` (one rectangle). `staged`: `split_work_to_cores(compute_with_storage_grid, tensor_pages, row_wise=True).all_cores` |
| Per-core work | `shard_*`: `core_rows · core_cols` pages; `bank_*`: `ceil((tensor_pages − b)/NB)` pages (0 when `b ≥ tensor_pages`); `staged`: `per_core_1` / `per_core_2` pages |
| Per-mover work | contiguous halves of the core's sequence: `half = ceil(n / movers_per_core)` |
| Remainder | `ceil` per image everywhere (`tensor_tile_rows = prod(leading)·ceil(H/32)`); uneven last shard clipped by `min(...)`; banks `b ≥ tensor_pages` idle; staged last block ragged with nominal CB counts |

Shard → core rule (host; must equal the buffer's, `buffer_distribution_spec.cpp:163-197`):

| Scheme | Core list | Shard `s` (row-major over the shard grid, `s = i·nsh_c + j`) |
|--------|-----------|------------------------------------------------------------|
| HEIGHT / WIDTH (ROUND_ROBIN_1D) | `ttnn.corerange_to_cores(grid, row_wise=True)` — ranges in the order the CoreRangeSet holds them | `cores[s]` |
| BLOCK (GRID_2D) | row-major cores of `CoreRange(start, start + (nsh_c − 1, nsh_r − 1))` of the single range | `cores[s]` |

Shard `s` rectangle: `r0 = i·shard_tile_rows`, `c0 = j·shard_tile_cols`, `core_rows = min(shard_tile_rows, tensor_tile_rows − r0)`, `core_cols = min(shard_tile_cols, tensor_tile_cols − c0)`. Before the first device run the implementer cross-checks this rule against `ttnn._ttnn.bfp_utils.compute_shard_page_mapping` (the host twin of the buffer mapping) on Qwen 2058, 2065, 2071 and a block-sharded case; it is not called per op call.

Per-core runtime arguments (contract, not layout): `shard_*` per mover: `r0`, `c0`, `core_cols`, `seq_begin`, `seq_end`; `bank_*`: none (everything derives from CT `tensor_pages` and the discovered bank); `staged`: `start_page`, `num_pages`. Common runtime args of every kernel: `src_base_addr`, `dst_base_addr` — the only values that change between calls with the same configuration.

### Host contract (validation, registry, caching, observability)

| Item | Decision |
|------|----------|
| Entry point | `g6_to_memory_config(input_tensor, memory_config)`; first line `validate(input_tensor, memory_config)` |
| `validate()` order | (1) structural refusals, `ValueError`: input not on device; layout not TILE; rank ∉ {2,3,4}; a shard spec (input's or `memory_config`'s) not ROW_MAJOR; a target shard shape not a multiple of 32; `num_shards > cores` (HEIGHT/WIDTH) or BLOCK grid not one range / shard grid wider or taller than the range; (2) SUPPORTED per axis → `UnsupportedAxisValue`; (3) EXCLUSIONS → `ExcludedCell`. The structural checks precede (2) because the spec demands `ValueError`/`RuntimeError` for them |
| Axis derivation in `validate()` | `scenario = {"shape": list(input.shape), "dtype": input.dtype, "layout": input.layout, "in_memory": {buffer, scheme} of input.memory_config(), "out_memory": {buffer, scheme} of memory_config}` run through `INPUT_TAGGERS` (the same taggers the golden suite projects) |
| INPUT_TAGGERS | `dtype`, `layout`, `rank` (`len(shape)`), `alignment` (`"tile_aligned"` iff H%32==0 and W%32==0, else `"h_non_aligned"`), `in_scheme`, `out_scheme`, `in_buffer`, `out_buffer` — names/values exactly as TARGET |
| SUPPORTED (Phase 0) / EXCLUSIONS | as the spec lists: dtype `[bfloat16]`, layout `[TILE]`, rank `[2,3,4]`, alignment both, schemes all four (in and out), buffers both (in and out); EXCLUSIONS = `[{"out_scheme": "width_sharded", "out_buffer": "dram"}]` only |
| L1 budget refusal (BB-05) | after `validate()`, before allocating the output: `ValueError` if `output_l1_bytes_per_core + cb_bytes > 1347776` (expressions in `l1_ledger.md`) |
| Descriptor cache (BB-07) | module-level dict keyed by `(device id, tuple(shape), dtype, input placement key, output placement key)`, placement key = `(buffer_type, memory_layout, shard grid ranges in order, shard shape, orientation)`; on a hit only the common RT args `src_base_addr`, `dst_base_addr` are rewritten; `custom_program_hash` set from the key (op-name-salted) so the C++ side does not rehash bank-coordinate CT args |
| Plan report (BB-08) | `select_plan(input_tensor, memory_config) -> Plan(regime, owner_side, num_cores, movers_per_core)`; `regime ∈ {"shard_pull","shard_push","bank_pull","bank_push","staged"}`; exported from the package; the entry point uses the same function |
| Role dump (BB-21) | `dump_core_roles(input_tensor, memory_config) -> list[(core_xy, role, pages)]`, role = `"<regime>/mover<m>"` or `"staged/reader"`, `"staged/writer"`; exported from the package |
| Output | `ttnn.allocate_tensor_on_device(ttnn.Shape(input.shape), input.dtype, ttnn.TILE_LAYOUT, device, memory_config)`; `ttnn.generic_op([input_tensor, output_tensor], descriptor)`; never returns the input, also when the placements are equal |

Regime-pinned tests (required, in the acceptance test): every built regime is pinned by `select_plan(...).regime` on a named case — `shard_pull`, `shard_push`, `bank_pull` by production cases; `bank_push` (L1 interleaved → DRAM) and `staged` (DRAM → DRAM interleaved) by one placement each on a production shape, because no Phase 0 production case selects them. The pin is exact where the placements leave one owner side (DRAM ↔ L1-sharded, DRAM ↔ L1-interleaved, DRAM ↔ DRAM); for L1 ↔ L1 pairs the test accepts any owner regime, because which side owns there carries a perf lamp and a tuning pass may move it. Selection depends on placement only, never on the device grid size, so a regime cannot appear on one board and vanish on another; the shard grids of the cases fit the 11×10 grid.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_staged_pages` | 0 | `aligned_page_size` (2048 / 1088 / 576 B) | `staged_buffer_depth · staged_block_pages` (Phase 0: 2 · 16 = 32) | live set spans one block on `page_linear` per depth slot; streams over the core's range; `page_row`/`page_col` flattened into `page_linear`; `intra_tile` whole | input dtype (dataflow-only payload, never in DEST) | reader (staged) | writer (staged) | `staged` regime only, whole kernel |

The owner regimes (`shard_pull`, `shard_push`, `bank_pull`, `bank_push`) allocate **no** circular buffer: the destination memory is the landing buffer and the source memory is read in place. Their descriptors list `cbs=[]`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `locate_owned_block` (owner regimes) | — | no (raw: RT args / bank-table scan) | — | — | — |
| 2 | `move_block` (owner regimes) | mover slice of `core_rows × core_cols` (`shard_*`) or of `core_pages` (`bank_*`), as runs | no (raw NoC + TensorAccessor) | — (source tensor in place) | — (destination tensor in place) | output pages in flight |
| 3 | `drain_block` (owner regimes) | whole mover slice | no (raw barrier) | — | — | output pages landed / acked |
| 4 | `load_block` (staged) | `staged_block_pages` pages | no (raw NoC + TensorAccessor) | — | `cb_staged_pages`, reserve + push nominal `staged_block_pages` | one block visible to writer |
| 5 | `store_block` (staged) | `staged_block_pages` pages | no (raw NoC + TensorAccessor) | `cb_staged_pages`, wait + pop nominal `staged_block_pages` | — | slot freed |
| 6 | `drain_writes` (staged) | — | no (raw barrier) | — | — | all writes acked |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all (accessors) | raw_api | `TensorAccessorArgs<cta, crta>()`, `TensorAccessor(args, base_addr)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h` (ctors `:85-96` sharded, `:515-545` interleaved), doc `tech_reports/tensor_accessor/tensor_accessor.md` | src and dst args, base from common RT args | — | — | — |
| `move_block`, `load_block`, `store_block` | raw_api | `TensorAccessor::get_noc_addr(page_id)` | `tensor_accessor.h:165` (sharded), `:554` (interleaved) | `page_id` = first page of the run | — | — | — |
| `move_block` | raw_api | `TensorAccessor::num_contiguous_pages(page_id, end)`, `contiguous_page_stride()` | `tensor_accessor.h:181-192` (sharded), `:562-575` (interleaved) | run only when stride == 1 | — | — | run length (derived, capped by shard row) |
| `move_block` (pull), `load_block` | raw_api | `noc_async_read(src_noc_addr, dst_l1, bytes)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:546-555` | default `max_page_size` (multi-packet) | — | — | `bytes = run_pages · aligned_page_size` |
| `move_block` (push), `store_block` | raw_api | `noc_async_write(src_l1, dst_noc_addr, bytes)` | `dataflow_api.h:876` | default template (multi-packet, non-posted) | — | — | as above |
| `drain_block` (pull), `load_block` | raw_api | `noc_async_read_barrier()` | `dataflow_api.h:1921` | — | — | — | frequency: 1 per mover / 1 per staged block |
| `drain_block` (push), `drain_writes` | raw_api | `noc_async_write_barrier()` | `dataflow_api.h:1951` | — | — | — | 1 per mover per call |
| `store_block` | raw_api | `noc_async_writes_flushed()` | `dataflow_api.h:1980` | — | — | — | 1 per staged block |
| `load_block` / `store_block` | raw_api | `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`, `get_write_ptr`, `get_read_ptr` | `dataflow_api.h:406, 208, 479, 260, 325, 346` | nominal `staged_block_pages` | `cb_staged_pages` | `cb_staged_pages` | `staged_block_pages` |
| `locate_owned_block` (bank_*) | raw_api | local core NoC address `get_noc_addr(addr)` vs bank NoC coordinates (`l1_bank_to_noc_xy[noc][b]`, or the L1 accessor's `get_noc_addr(b)`) | `tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:272`; `internal/dataflow/dataflow_api_common.h:28` | scan `b ∈ [0, NUM_L1_BANKS)` on this kernel's `noc_index` | — | — | — |
| all (zones) | raw_api | `DeviceZoneScopedN(name)` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012-1047` | zone names in the Block schedule | — | — | — |

**Helpers considered and rejected** (every entry above is `raw_api`; the op has no compute kernel, so no compute helper and no `compute_kernel_hw_startup()` applies):

| Candidate | File:Line of the mismatch | Why it cannot express this op's block operations |
|-----------|---------------------------|--------------------------------------------------|
| `dataflow_kernel_lib::set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:74-95` | programs a **fixed self-aimed source** and a single-packet size once (`static_assert ≤ NOC_MAX_BURST_SIZE`); every pull here reads a different remote core/bank per run, runs exceed 16 KB, and pushes are writes |
| `dataflow_kernel_lib::local_addr` | `local_copy_helpers_dataflow.hpp:57` | builds a self-aimed source endpoint only — covers the one self-copy case (height→block on one core) but not remote pages; adopting it for one case would split the kernel's path |
| `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` | `local_copy_helpers_dataflow.hpp:147, 158` | trid pipelining matters only inside `staged` (the overlap lamp); Phase 0 `staged` uses CB double buffering, the trid variant is the lamp's alternative |
| `McastArgs` / `ttnn.Mcast` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:18-32` | no operand is shared across cores; every page has exactly one destination |
| `tilize` / `untilize` / `reduce` / `eltwise` / `matmul_block` helpers | `ttnn/cpp/ttnn/kernel_lib/*.hpp` | compute helpers; the op must not route pages through the unpacker/packer (block-format bits) |

Gap record: no `kernel_lib` helper moves whole pages between two `TensorAccessor`s by contiguous runs. The `move_block` operation (owner walk + run formation + one barrier) is the missing block helper; the implementer realizes it as a kernel-local function shared by the four owner regimes.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Host shard→core rule differs from the buffer's | values read back right only if every page lands where `ttnn.to_torch` looks; Qwen 2058's 16 single-core ranges are not sorted by (y, x); Qwen 2065 has 36 cores for 34 shards | derive cores exactly as `BufferDistributionSpec::compute_core_list` (ROUND_ROBIN_1D: CoreRangeSet order; GRID_2D: trimmed single range); one-time probe against `compute_shard_page_mapping`; golden cases 2058/2065/2071 + block cases |
| Uneven last shard addressed with the clipped width | Qwen 2071: 404 columns over 8 shards of 52 → last shard 40 valid columns stored with stride 52 | local offset uses nominal `shard_tile_cols` (mechanism cap); `--dev` cross-check against the accessor |
| L1-interleaved bank → core mapping guessed on host | bank ids are a seeded shuffle of the worker grid (`l1_banking_allocator.cpp:80-85`; probe: page 0 on core (0,3), page 1 on (8,7)) | discover the own bank on device; launch `bank_*` on the full grid; host checks bank count == grid size |
| NoC coordinate space mixed between the two movers | mover 0 runs NoC0, mover 1 NoC1; `my_x/my_y` and accessor addresses differ per NoC | every coordinate compare and address uses the kernel's own `noc_index` |
| `noc_async_writes_flushed` used as the final wait | data left L1 but acks outstanding → next program hangs (BB-03) | push movers and the staged writer end with `noc_async_write_barrier`; pull movers with `noc_async_read_barrier` |
| One-packet / stateful NoC calls for runs | runs reach 128 KB; one-packet APIs cap at 16 KB (8 KB WH) | multi-packet `noc_async_read`/`noc_async_write` only |
| Program replay with stale config (BB-07) | the Python cache key must cover every CT input: shape, dtype, both placements incl. ordered grid ranges, shard shape, orientation | key definition above; addresses are the only patched values |
| Self-aimed transfers | height→block on one core (Qwen 2157/2158) pulls from its own L1 | NoC self-reads are legal; no special path |
| Movers with zero pages | `core_pages = 1` gives mover 1 nothing; banks `b ≥ tensor_pages` own nothing | kernels return immediately (still inside `G6-LOCATE`), no barrier issued without transfers is harmless |
| Same placement in and out | tempting to return the input | always allocate and copy (contract: never return the input) |
| Partial tile rows | 67 of 105 cases are `h_non_aligned` | whole pages are copied; the op never inspects H beyond `ceil(H/32)`; no host padding/slicing |

# Operation Design: g6_expand

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement — two dataflow kernels per core, **no compute kernel, no arithmetic** |
| Goal | In one device program, materialize `x` `(d0, d1, 1, W)` broadcast along dim 2 to `(d0, d1, H, W)` as a new TILE tensor. Production: `(1,1,1,W) -> (1,1,W,W)`, `W ∈ {185,196,207,256,512}`. Every logical element bit-identical, every out page written once. |
| Math | `out[i0,i1,h,w] = x[i0,i1,0,w]`. In tile terms: out page `(n, r, c) = T(q)`, with `q = n·Wt + c` and `T(q)` = x page `q` with its row 0 copied into all 32 rows. |
| Precision | No value passes through any engine that interprets it: x bytes move by NoC transfers and RISC-V word copies only. Bit identity (NaN payloads, ±0, denormals, ±inf) holds by construction. No `fp32_dest_acc_en`, fidelity or approx mode applies. |
| References | task spec (contract); `eval/golden_tests/g6_expand/{feature_spec.py,helpers.py}`; `.claude/references/{blocking-model.md,l1-footprint-discipline.md,op-design-template.md}`; `ttnn/cpp/ttnn/kernel_lib/mcast/{host/mcast.hpp,kernel/mcast_args.hpp,kernel/pipes/mcast_pipe.hpp}`; `ttnn/cpp/ttnn/kernel_lib/{local_copy_helpers_dataflow.hpp,l1_helpers.hpp,tile_column_dataflow.hpp}`; `tt_metal/hw/inc/api/dataflow/{noc.h,circular_buffer.h}`; catalog `ttnn/ttnn/operations/examples/master.md` (see Catalog evidence) |

### Tile-level restatement

| Quantity | Definition |
|----------|------------|
| `num_images` (N) | `d0·d1` (production 1) |
| `tensor_cols_tiles` (Wt) | `ceil(W/32)` = `x.padded_shape[-1] // 32` |
| `tensor_rows_tiles` (Ht) | `ceil(H/32)`, per image (H = out dim 2) |
| `tensor_pages` (Q) | `N·Wt` = `x.buffer_num_pages()` — one x page per `(n, c)` because `d2 = 1` |
| out pages | `N·Ht·Wt` = `out.buffer_num_pages()` |
| `q -> (n, c)` | `n = q // Wt`, `c = q % Wt` |
| out page id of `(q, r)` | `(n·Ht + r)·Wt + c` |
| Row-broadcast page `T(q)` | bf16 tile = 4 faces × 512 B, face `f` at byte `512·f`, 16 rows × 32 B row-major inside a face. Row 0 of a tile = bytes `[0,32)` (face 0) ++ `[512,544)` (face 1). `T(q)`: rows 0–15 of faces 0 **and 2** = x page `q` bytes `[0,32)`; rows 0–15 of faces 1 **and 3** = bytes `[512,544)`. Equivalently: build bytes `[0,1024)` from the two 32 B segments, then bytes `[1024,2048)` = bytes `[0,1024)`. |
| Input padding | x rows 1–31 are never a source (they are overwritten in place). x row-0 columns `≥ W` land only in out's padding columns. No logical out element depends on padding. |
| Output padding | rows `≥ H mod 32` of the last tile row and columns `≥ W` of the last tile column hold row copies — legal ("may hold anything"). Non-aligned W and H need no separate path. |

| W | Wt = Ht | Q | out pages | `alignment` |
|---|---------|---|-----------|-------------|
| 512 | 16 | 16 | 256 | tile_aligned |
| 256 | 8 | 8 | 64 | tile_aligned |
| 207 | 7 | 7 | 49 | w_non_aligned |
| 196 | 7 | 7 | 49 | w_non_aligned |
| 185 | 6 | 6 | 36 | w_non_aligned |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | see Input | — | buffer address → common RT arg of the reader (patched per call) |
| `sizes` | list or tuple of 4 `int`, positional | yes | each `-1` or `≥ 1`; see Validation | — | host only: gives out shape → Ht → plan → CT/RT |

No `memory_config`, no output dtype, no `compute_kernel_config`. `-1` in `sizes` is replaced by the size of x along that dim before any check (canonicalization); there is no index-axis parameter.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 4, `(d0, d1, d2, d3)` with `d2 == 1` (dim 2 is the only expanded dim); production `(1,1,1,W)` |
| Dtype | bfloat16 (SUPPORTED) |
| Layout | TILE |
| Memory | DRAM interleaved, on device |

### Output

| Property | Value |
|----------|-------|
| Shape | `sizes` with each `-1` replaced by x's size: `(d0, d1, H, d3)`, `H = sizes[2]` (`-1` → 1) |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` (TILE) |
| Memory | `input_tensor.memory_config()` (DRAM interleaved). A new allocation, never a view. |

### Validation and registry (`validate()`, first line of every entry point)

| # | Check | Raises |
|---|-------|--------|
| 1 | `x` not on device | `ValueError` |
| 2 | `rank(x) != 4`, or `len(sizes) != rank(x)`, or a size that is not an `int` | `ValueError` |
| 3 | a size `s != -1` with `s <= 0` | `ValueError` |
| 4 | a dim with `x.shape[d] != 1` and `s ∉ {-1, x.shape[d]}` | `ValueError` |
| 5 | a dim `d ∈ {0,1,3}` with `x.shape[d] == 1` and `s ∉ {-1, 1}` (expansion outside dim 2) | `ValueError` |
| 6 | `x.shape[2] != 1` (the contract has one row per image) | `ValueError` |
| 7 | `x` not DRAM interleaved | `ValueError` |
| 8 | axes `{dtype, layout, alignment}` outside `SUPPORTED` | `UnsupportedAxisValue` |
| 9 | cell in `EXCLUSIONS` | `ExcludedCell` |

Every message raised by `validate()` (rows 1–9) starts with `g6_expand`. The acceptance test matches on that prefix.

| Registry item | Value |
|---------------|-------|
| `INPUT_TAGGERS` | `{"alignment": tag_alignment}`. `tag_alignment(inputs, axes)`: `scenario = inputs[0]`; out shape = `scenario["sizes"]` with each `-1` replaced by `scenario["shape"][d]`; `"w_non_aligned"` if `W_out % 32` (regardless of H), elif `H_out % 32` → `"h_non_aligned"`, else `"tile_aligned"`. `validate()` builds the same scenario dict from the call. |
| `SUPPORTED` (Phase 0) | `dtype: [bfloat16]`, `layout: [TILE_LAYOUT]`, `alignment: ["tile_aligned", "w_non_aligned"]` — equals TARGET. The kernels are alignment-agnostic; `"h_non_aligned"` is refused only because it is outside TARGET. |
| `EXCLUSIONS` | `[]` |
| INVALID | none (lives in `feature_spec.py`, not checked by the op) |
| Structural impossibilities | none to add |

## Blocking Model

### Axes

| Axis | Character (+ reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|----------------------|-------------|---------------|-------------|-----------------|--------------|
| `n` — image (`d0·d1`) | independent — out image `n` reads only x image `n` | folded with `c` into the page axis `q = n·Wt + c` (the op treats `n` and `c` identically and x pages are contiguous in `q`); extent = `block_pages` | whole group assignment | `ExpandPlan.block_pages` | groups own contiguous `q` ranges, which may straddle images | knob-turn |
| `c` — out tile column = x page within an image | independent — out column `c` reads only x page `c` | `block_pages` (along `q`) | `group_pages` of the group = **1 in every production case**: that is the group's whole assignment in one block, not a collapsed knob | `ExpandPlan.block_pages`, set once in `select_plan` | `num_groups` groups, contiguous balanced `q` ranges (first `Q mod G` groups get one more page) | knob-turn |
| `r` — out tile row (expanded axis, tile granularity) | independent. **x does not vary along `r`, so splitting `r` makes x reuse-shared by construction** → each x page is broadcast within its group | `block_rows` | `core_rows_tiles` (the core's whole share). The block's pages stay resident while every copy is written; cutting `r` has no L1 term, so there is nothing to buy. | derived from `ExpandPlan.row_splits` | `row_splits` cores per group split `Ht` balanced; the injector (split 0) takes the floor share, the last `Ht mod row_splits` splits take the ceiling | knob-turn |
| `i` — row inside a tile (32) | independent — every row sources row 0 | not a blocking axis: the 2048 B page is the indivisible transfer and write quantum | 32 | tile shape | none (inside one page) | — |
| stage data: built block `T` (`q` × 32 rows) | produced by `replicate_rows_block` | spans `q` → `block_pages` | `block_pages` | as above | **Computed on the injector only.** The injector is a communication role, and this is the one stage it also computes while its receivers wait. It has nothing to spread: every core writes whole pages, so every core needs all of `T`, and splitting one page's replication across cores would cost an all-gather larger than the work. | perf lamp L3 (split the replication across the injector's two RISC-Vs) |

### Core-assignment knobs

| Knob | Meaning | Phase 0 value | Source |
|------|---------|---------------|--------|
| `grid` `(gx, gy)` | `device.compute_with_storage_grid_size()` | runtime, never inlined | `ExpandPlan.grid` |
| `strips_per_column` (b) | groups stacked in one grid column | Phase 0 rule below | `ExpandPlan.strips_per_column` |
| `row_splits` (rs) | cores per group, a vertical strip | `min(Ht, gy // b)` | `ExpandPlan.row_splits` |
| `num_groups` (G) | groups; group `g` sits in grid column `g // b`, rows `[(g mod b)·rs, (g mod b)·rs + rs)`; its injector is the strip's top core | `min(Q, gx·b)` | `ExpandPlan.num_groups` |

**Phase 0 rule (`select_plan`):** pick `b ∈ [1, gy]` that minimizes `key(b) = (ceil(Q/G_b) · ceil(Ht/rs_b), ceil(Q/G_b), b)`, with `rs_b = min(Ht, gy // b)` and `G_b = min(Q, gx·b)`. The first term is the most out pages any core writes (occupancy). The second is the pages each injector replicates, which is the serial term on every group's critical path. The third prefers fewer strips. Columns fill first, so the used set is one rectangle (plus one partial column when `G mod b ≠ 0`).

| W | b | rs | G | `group_pages` | `core_rows_tiles` | cores | used set |
|---|---|----|---|---------------|-------------------|-------|----------|
| 512 | 2 | 5 | 16 | 1 | 3 (splits 0–3, injector included), 4 (split 4) | 80 | 8 cols × 10 rows |
| 256 | 1 | 8 | 8 | 1 | 1 | 64 | 8 × 8 |
| 207, 196 | 1 | 7 | 7 | 1 | 1 | 49 | 7 × 7 |
| 185 | 1 | 6 | 6 | 1 | 1 | 36 | 6 × 6 |

(The plans are identical on 13×10 and 11×10 grids. For W=512, `b = 1` ties on the first key term, 4 pages per core, but loses on the second: 2 pages per injector.)

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_row_bcast_tiles` | `t_block_depth` | **1** (every production group has one block). Derived: `1` if `ceil(group_pages_max / block_pages) == 1`, else `2`. | With more than one block per group, a second slot lets the injector load, replicate and publish block `k+1` while the group writes block `k`. With one block there is no next block. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 left to the op (BB-05: 1,347,776 B/core) | `block_pages` | `block_pages ≤ floor(1347776 / (t_block_depth · page_bytes))` (658 at depth 1, 329 at depth 2, for 2 KiB pages), applied in `select_plan`. A shape is never refused. | launch refused / L1 clash for very wide W |
| Multicast landing address | slot sequence | one CB descriptor over all participating cores. Every core of a group reserves and pushes the **nominal** `block_pages` on every block, including a ragged last block (the actual count goes only into the NoC transfers and writes). | injector and receiver slots drift apart → the multicast lands in a slot the receiver is not reading (silent corruption) |
| Multicast receiver set | `row_splits` | a group is a 1-D line of `rs` cores in one grid column, with `b·rs ≤ gy` | a strip crossing into the next group → foreign pages |
| BH DRAM read alignment (64 B, relative) | none on extent | whole-page reads into slot offsets `k·page_bytes`, which keep the DRAM page offset modulo 64 | silent corruption (watcher stops it) |
| `NOC_MAX_BURST_SIZE` (16 KiB on BH, noc_parameters.h:290) | write and multicast size | page writes are 2 KiB single-packet. Multicast payload `pages·page_bytes` may exceed 16 KiB, and `Noc::async_write_multicast`'s default `max_page_size` takes the multi-packet path (noc.h:498). | — |
| Ready-ack counter | `rs − 1` increments per block | ≤ `gy − 1` | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (boundary: DRAM — each x page read once, each out page written once) | What a bigger block buys |
|--------|--------|-----------|-------|------------------------------------------------------------|--------------------------|
| **R1 `column_group_bcast`** — 2-D split `q × r`; each group's injector reads its x pages once, replicates row 0 in place, multicasts the built pages down its strip; every core writes its rows | **built** | always (every call that passes `validate`). `rs = 1` is its no-multicast setting of the same kernels (reached when `Ht = 1` or forced), not a separate algorithm. | `block_pages` (q) × `core_rows_tiles` (r) | **At the DRAM minimum for out.** x: each page read once, but as a whole 2048 B page, of which 64 B is logical (perf lamp L5). Above minimum only on the NoC: `(rs − 1)·page_bytes` multicast per x page, plus one data-ready flag multicast and `rs − 1` ready-acks per block. Domain: all shapes. | Per block: one DRAM read barrier, one replicate pass (its fences), one ready-ack round + one multicast + one flag per group, one CB reserve/push/wait/pop, one writes-flushed. Phase 0: each **once per kernel** (one block per group in every production case). |
| R2 `per_core_read` — same 2-D split, every core reads its own x pages from DRAM and replicates them itself | **rejected** — superseded by R1 | — | as R1 | x crosses DRAM `rs` times (Q·rs page reads; all `rs` reads of a page hit the same bank). For the 1-page-per-core shapes that is one x read per out page — the duplicated small requests RB-010 forbids. | — |
| R3 `row_only_split` — split only `r`, every core writes all `q` | **rejected** — superseded by R1 | — | Q × `core_rows_tiles` | x needs a whole-grid 2-D multicast of all Q pages (or Q·cores DRAM reads). Every core replicates all Q pages serially, and L1 grows as `Q·page_bytes`, unbounded in W. | — |
| R4 `compute_row_bcast` — TRISC unpack with row broadcast → DEST → pack | **rejected** — superseded by R1's byte copies | — | — | Same DRAM traffic as R1, but every value passes through unpacker / FPU / DEST / packer, so bit identity of denormals and signalling-NaN payloads is not guaranteed (the contract forbids a representation that can change a value). It also adds three TRISC binaries per core (BB-18). | — |
| R5 `sub_page_split` — more cores than out pages, each writing faces of a page | **rejected** — superseded by R1 | — | — | 512 B DRAM writes (4× the transactions). Catalog `tile_reorder`: whole-page writes win. No production shape has fewer out pages than the grid can use. | — |
| Dependent-axis split + combine | not applicable | — | — | The op has no dependent axis (no reduction, no scan). | — |

The op has one built regime, so there is no regime-selection function. The knob values that vary by shape and grid (`b`, `rs`, `block_pages`, `t_block_depth`) are pinned by the forced-plan tests (`test_pinned_plans`).

### Traffic ranking

| Candidate split | x DRAM crossings | out DRAM crossings | Cross-core bytes | Serial work on the slowest core | Verdict |
|-----------------|------------------|--------------------|------------------|---------------------------------|---------|
| `q` only (R1 at `rs = 1`) | 1 | 1 | 0 | replicate `ceil(Q/G)` pages, write `Ht·ceil(Q/G)` pages | **cheapest traffic**; writer count capped at `Q` (6–16 in production), i.e. by the tensor's shape |
| `q × r` + group broadcast (R1) | 1 | 1 | `Q·(rs−1)·page_bytes` (1-D column lines, ≤ `gy−1` hops) | replicate `ceil(Q/G)`, write `ceil(Q/G)·ceil(Ht/rs)` | **chosen** |
| `q × r`, per-core reads (R2) | `rs` | 1 | 0 | as R1 plus a DRAM read and a replicate per core | rejected (RB-010) |
| `r` only + whole-grid broadcast (R3) | 1 | 1 | `Q·(cores−1)·page_bytes` | replicate **Q** pages on every core | rejected |
| dependent axis + combine | — | — | — | — | does not exist |

**Chosen: R1.** It ties the cheapest split on DRAM crossings (x once, out once), the most expensive tier. Against `q`-only it adds only multicast bytes on the cheapest non-local tier. This is an explicit trade (blocking-model §4): those NoC bytes buy write-issue parallelism. Each core issues at most `ceil(Ht/rs)` page writes per page instead of `Ht`, and the core count becomes the plan's choice instead of `Q`, as the spec's Work-geometry rule requires. The `q`-only point stays one knob value away (`row_splits = 1`), built and test-pinned. The grid-synchronization lamp L1 measures the trade.

### Block schedule

Logical schedule (reader and writer are separate asynchronous kernels):

```cpp
// injector core of group g (row split 0)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);            // reader
    replicate_rows_block(block_idx);  // reader, in place
    publish_block(block_idx);         // reader: hand to local writer, multicast to the strip
    store_block(block_idx);           // writer
}
// receiver core of group g (row splits 1 .. rs-1)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    receive_block(block_idx);         // reader
    store_block(block_idx);           // writer
}
```

| Operation | Runs on | Block shape | Resident across it | Intended frequency of fixed-cost work |
|-----------|---------|-------------|--------------------|---------------------------------------|
| `load_block` | injector reader (NCRISC, NoC0) | `block_pages` x pages (`group_q_start + block_idx·block_pages + k`) | the reserved slot | one `reserve_back(block_pages)` per block; all page reads issued back-to-back, **one read barrier per block**. Each whole page lands at slot offset `k·page_bytes`, so row 0 is already where `T` needs it (no staging). |
| `replicate_rows_block` | injector reader | `block_pages` pages, in place | slot | **one pass per block.** Every page goes through each copy round together: one ordering fence per round per block, never per page. Byte copies only (RISC-V 32-bit word copies and/or self-aimed NoC L1→L1 copies). It writes rows 1–15 of faces 0/1 from row 0, then faces 2/3 from faces 0/1, and never reads x rows ≥ 1. |
| `publish_block` | injector reader | the slot's actual `pages·page_bytes` | slot until `send()` returns | `push_back(block_pages)` to the local writer first, then **one** `send()` per block (waits `rs−1` ready-acks, multicasts the pages to the strip below, multicasts the data-ready flag). `send()` with the default source guard returns once the NoC has read the slot, before the next `reserve_back`. Skipped when `rs = 1`. |
| `receive_block` | receiver reader | `block_pages` pages | slot | `reserve_back(block_pages)`, then **one** `receive(block_idx)` (one ready-ack to the injector, wait for the data-ready flag, clear it), then `push_back(block_pages)`, per block |
| `store_block` | writer (BRISC, NoC1) on every core | pages × `core_rows_tiles` | the block's pages, until the last copy has left L1 | one `wait_front(block_pages)` per block. `pages × core_rows_tiles` page writes issued back-to-back (row-major over `(r, k)`), **one `async_writes_flushed` per block** before `pop_front(block_pages)`, **one `async_write_barrier` per kernel**. |

Kernel ends (BB-03): the injector reader ends with `async_write_barrier` (multicast + flag acked). The receiver reader ends with `async_atomic_barrier` (its ready-ack increments). The writer ends with `async_write_barrier`.

**Stall-shadow check.** Receivers stall on the data-ready flag, and the injector stalls on the DRAM read. No logical out element is independent of x, so no data work exists to schedule into either wait. Only per-kernel setup is hoisted in front of the waits: accessors, the core's out page ids, one-packet write state. No reorder touches floating-point order, because there is no floating-point work.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 grid synchronization (`row_splits`) | For 185–256 every core writes one page. The multicast hop (ready-ack, multicast, flag poll) is a fixed cost paid to save ≤ `Ht−1` page writes per writer, and DRAM bandwidth may already saturate with `Q` writers. | `row_splits = 1` (no multicast, `Q` cores each writing `Ht` pages) and `row_splits ≈ Ht/2`, on all five shapes |
| L2 strip packing (`strips_per_column`) | For W=512 the rule picks `b=2` (80 cores, 1 page per injector). `b=1` (13 or 11 groups, ≤ 2 pages per injector, up to 130 cores) has the same pages per core. | `strips_per_column = 1` on W=512 |
| L3 replicate realization | The replicate is serial on every group's critical path: about 500 word stores per page (rows 1–15 of faces 0/1, then faces 2/3), or 5 self-aimed NoC rounds. | RISC-V word copies vs NoC doubling `32→64→128→256 B` then `1 KiB → 1 KiB` vs a mix. Also split the replicate across the injector's NCRISC and BRISC (BRISC is idle until `publish_block`). |
| L4 half-page body | Faces 2/3 duplicate faces 0/1, so only 1 KiB of each page is unique. | Replicate faces 0/1 only, multicast 1 KiB per page, write each out page as two 1 KiB writes. This halves replicate and multicast bytes and doubles write transactions (catalog `tile_reorder` predicts whole pages win). |
| L5 x read shape | The whole-page read moves 2048 B to use 64 B (32 KiB vs 1 KiB at W=512). | Two 32 B reads per page (offsets 0 and 512) into the same slot offsets |
| L6 bank alignment of writes | When `Wt` is a multiple of the DRAM bank count (W=512: 16 columns), every page of column `c` sits on bank `c mod banks`, so all cores of a group write to one bank at once. | Stagger each core's first row (`bank_stagger`), and horizontal strips (`McastCoreOrder.RowMajor`) |
| L7 overlap (multi-block groups only; none in production) | With more than one block, depth 2 may still serialize replicate against writes. | `t_block_depth = 3` against 2 at `block_pages` < `group_pages` |

### Catalog evidence

| Entry | What it set |
|-------|-------------|
| `mcast_topology`, `shared_input_reuse` | Broadcast x along the axis it does not vary with (`r` → 1-D column lines, one injector per line). The measured win is smaller than the read-count reduction, which motivates lamp L1. |
| `noc_placement` | Reader on NoC0, writer on NoC1 (the default config descriptors). Column-shaped lines are slower for DRAM traffic → lamp L6 (orientation). |
| `tile_reorder` | Whole-page 2 KiB writes, not faces (R5 rejected; L4 measured, not assumed). |
| `double_buffer` | Batch the writes: one flush per block, not per page. Depth matters only with more than one block (L7). |
| `bank_stagger` | Lamp L6. |
| `width_split` | The page axis `q` (width) is the primary cut, because Ht alone strands most of the grid when x is one tile row tall. |
| compute entries (`compute_block_size`, `compute_fusion`, …) | Not applicable: there is no compute kernel. |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x page → injector L1 | bf16 tile page (2048 B); row 0 logical, rows 1–31 padding | NoC0 read via `TensorAccessor` (DRAM interleaved) into `cb_row_bcast_tiles` slot `k` | lands in place; padding is overwritten next |
| replicate | same page | byte copies on the injector's reader | in place; bit-exact by construction |
| injector → receivers | built pages | `ttnn.Mcast` channel on NoC0, one 1-D multicast down the strip + data-ready flag, ready-ack handshake | lands at the same slot address on every receiver |
| reader → writer | CB pages | `cb_row_bcast_tiles` push / wait | one producer (reader), one consumer (writer) per core |
| L1 → out DRAM | 2048 B pages | NoC1 writes via `TensorAccessor`; each built page is written `core_rows_tiles` times | each out page written once, by exactly one core |

Placement: TARGET has no `memory_layout` axis. x and out are DRAM interleaved only, so no physical shard is consumed or produced. A later sharded out would be a knob-turn, because the op has no dependent axis: a height shard (cut along `r`) or a width shard (cut along `q`) keeps the group multicast and redirects only the writes into the local shard.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_pages` x pages × `core_rows_tiles` out tile rows |
| Grid | `device.compute_with_storage_grid_size()`; used set = `G·rs` cores, columns `0 .. ceil(G/b)−1`, rows `0 .. b·rs−1` (last column partial when `G mod b ≠ 0`) |
| Per-core work | `group_q_start, group_pages` (shared by the strip), `core_r_start, core_rows_tiles` (own), role ∈ {injector, receiver} |
| Remainder | `Q` over `G`: first `Q mod G` groups take `ceil`. `Ht` over `rs`: last `Ht mod rs` splits take `ceil`, injector floor. All counts come from `ceil`, per image: `Wt = ceil(W/32)`, `Ht = ceil(H/32)`, out page id `(n·Ht + r)·Wt + c`. |
| Blocks per core | `num_blocks_this_core = ceil(group_pages / block_pages)`, `last_block_pages = group_pages − (num_blocks_this_core−1)·block_pages`. Every core of a strip has the same block sequence. |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_row_bcast_tiles` | 0 | `page_bytes` (2048) | `t_block_depth · block_pages` | spans `q` (`block_pages` pages of one block); streams `r` (each resident page is re-written `core_rows_tiles` times, no capacity); `i` inside the page | `x.dtype` (bf16; bytes moved verbatim) | reader | writer | whole kernel, one slot per block |

It is the only CB. The x read lands in it, the replicate is in place, the multicast sends from it and lands in it, and the writer writes out pages straight from it. Ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_pages` pages | raw NoC + accessor | — (x in DRAM) | `cb_row_bcast_tiles`, `block_pages` reserved | reserved, raw pages |
| 2 | `replicate_rows_block` | `block_pages` pages | custom block op (no covering helper) | `cb_row_bcast_tiles` (reserved, raw) | same slot | reserved, built |
| 3 | `publish_block` | actual pages of the block | `Mcast` sender pipe | `cb_row_bcast_tiles` (reserved, built) | `cb_row_bcast_tiles` on receivers; local `push_back(block_pages)` | pushed to local writer; receivers' slots filled |
| 4 | `receive_block` | `block_pages` pages | `Mcast` receiver pipe | — | `cb_row_bcast_tiles`, `push_back(block_pages)` | pushed |
| 5 | `store_block` | pages × `core_rows_tiles` | raw NoC + accessor | `cb_row_bcast_tiles`, `wait_front(block_pages)` | — (out in DRAM) | popped after flush |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `CircularBuffer::reserve_back`; `Noc::async_read(TensorAccessor, CircularBuffer, page_bytes, {.page_id}, {.offset_bytes = k·page_bytes})`; `Noc::async_read_barrier` | circular_buffer.h:31; noc.h:252; noc.h:739 | x accessor from `TensorAccessorArgs(x)` | — | `cb_row_bcast_tiles` | `block_pages` (reserve count, read count) |
| `replicate_rows_block` | raw_api | RISC-V word copies via `addr_to_l1_ptr`; and/or self-aimed copies `local_addr` + `set_read_state<size>` + `read_with_state`, ordered by `Noc::async_read_barrier` | l1_helpers.hpp:30; local_copy_helpers_dataflow.hpp:57, :75, :95; noc.h:739 | segment offsets 0 / 512, face size 512 | `cb_row_bcast_tiles` | same | `block_pages` (pages per pass) |
| `publish_block` | helper | host `ttnn.Mcast(device, McastConfig(handshake=True), used_cores, row_splits, McastFixedSenderConfig(), McastCoreOrder.ColumnMajor)` + `attach`; device `McastArgs<ct_off, rt_off>().sender(noc).send(src_l1, dst_l1, pages·page_bytes)`; `CircularBuffer::push_back` | mcast.hpp:131, :149; mcast_args.hpp:217; mcast_pipe.hpp:92; circular_buffer.h:43 | `src_l1 == dst_l1` = slot write pointer; group size = `row_splits`; `rs = 1` → `ttnn.attach_absent_mcast` | `cb_row_bcast_tiles` | receivers' `cb_row_bcast_tiles` | payload = actual pages of the block |
| `receive_block` | helper | `CircularBuffer::reserve_back`; `McastArgs<…>().receiver(noc).receive(block_idx)`; `CircularBuffer::push_back`; `Noc::async_atomic_barrier` at kernel end | circular_buffer.h:31; mcast_args.hpp:219; mcast_pipe.hpp:163; circular_buffer.h:43; noc.h:805 | PRE_HANDSHAKE on (host `handshake=True`) | — | `cb_row_bcast_tiles` | `block_pages` |
| `store_block` | raw_api | `CircularBuffer::wait_front`; `Noc::async_write(slot + k·page_bytes → out page)`; `Noc::async_writes_flushed`; `CircularBuffer::pop_front`; `Noc::async_write_barrier` at end | circular_buffer.h:55; noc.h:411; noc.h:783; circular_buffer.h:63; noc.h:760 | out accessor from `TensorAccessorArgs(out)`; one-packet write state reuse is an allowed realization (2 KiB ≤ 16 KiB burst) | `cb_row_bcast_tiles` | — | `block_pages`, `core_rows_tiles` |

**Helpers considered and rejected:**

| Raw entry | Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|-----------|---------------------------------|
| `load_block`, `store_block` | kernel_lib dataflow helpers (tilize / reduce-scaler / mcast / local-copy / tile-column) | tilize_helpers_dataflow.hpp; reduce_helpers_dataflow.hpp; local_copy_helpers_dataflow.hpp:57–158 | None of them moves interleaved DRAM pages into or out of a CB block. They cover RM-stick tilize, scaler tiles, multicast and L1→L1 copies. The catalog's own readers (`mcast_topology`) use `Noc::async_read` directly. |
| `replicate_rows_block` | `fill_tile_column`, `copy_tile_column` | tile_column_dataflow.hpp:69, :91 | They move one **column** (16 elements at stride `FACE_WIDTH` in two faces). The op needs one 32 B **row segment** per face copied into 15 rows and two faces. |
| `replicate_rows_block` | `fill_l1_range<val_size>` | l1_helpers.hpp:101 | It replicates one scalar (≤ 4 B), not a 32 B pattern. |
| `replicate_rows_block` | compute `copy` with `BroadcastDim::Row` | eltwise/api/convenience.hpp:113; eltwise/api/chain.hpp:330 | Values pass through unpacker / FPU / DEST / packer, so bit identity is not guaranteed for denormals and signalling-NaN payloads (R4 rejected). It also needs a compute kernel. |
| **Gap to close** | a dataflow block helper `replicate_tile_row0(addr, pages)` (row 0 → all 32 rows, N pages, byte copies) | — | It would make `replicate_rows_block` a one-line helper call. Build it in-kernel now, as a block function over `block_pages`. |

## Host Program Contract

| Item | Contract |
|------|----------|
| Dispatch (BB-01) | Exactly one `ttnn.generic_op([x, out], descriptor)` per call. `out = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), x.dtype, x.layout, device, x.memory_config())`. No other ttnn op, no host readback, no pad/slice/to_layout. |
| Plan API (BB-08) | `select_plan(input_tensor, sizes, *, strips_per_column=None, row_splits=None, block_pages=None) -> ExpandPlan`: host-only, pure, deterministic, validates first. `ExpandPlan` is a frozen dataclass with fields `regime` (`"column_group_bcast"`), `grid`, `tensor_pages`, `tensor_rows_tiles`, `tensor_cols_tiles`, `strips_per_column`, `row_splits`, `num_groups`, `group_pages_max`, `block_pages`, `t_block_depth`, `num_cores`, `l1_bytes_per_core`, `core_roles` (dict `(x, y) → "injector" \| "receiver"`) and `core_work` (dict `(x, y) → (group_q_start, group_pages, core_r_start, core_rows_tiles)`). The plan a call used is `select_plan(x, sizes)`. |
| Override semantics | `strips_per_column` fixes `b` (`1 ≤ b ≤ gy`, else `ValueError`). `row_splits` caps `rs = min(row_splits, Ht, gy // b)` (≥ 1). `block_pages` sets `min(block_pages, group_pages_max, L1 clamp)` (≥ 1), then `t_block_depth` is re-derived and the L1 clamp re-checked. Un-overridden knobs follow the Phase 0 rule. |
| Forced plan | `g6_expand_with_plan(input_tensor, sizes, plan)`: `validate()` first. `plan` must match `(Q, Ht, grid)` of the call, else `ValueError`. Same build and dispatch path as `g6_expand`. Used only by the regime-pinned tests. |
| Exports | `ttnn.operations.g6_expand` exports `g6_expand`, `select_plan`, `g6_expand_with_plan`, `ExpandPlan`, `validate`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`. |
| Program cache (BB-07, BB-16) | A Python-side dict maps `(device id, x.shape, x.padded_shape, out shape, dtype, layout, x.memory_config(), grid, plan knobs)` to the built `ProgramDescriptor`. On a hit the descriptor is **not rebuilt**: only the two buffer addresses change. They live in **common runtime args** (reader: x address; writer: out address) and are patched in place. Do not set `custom_program_hash`: generic_op's structural hash (kernel sources, CT args, core ranges, CB sizes) keys the device cache and cannot collide with another generic op. |
| Knob sources | Every knob is set once in `ExpandPlan`. CB `total_size = t_block_depth·block_pages·page_bytes`. `block_pages` is a kernel CT arg (the CB quantum). Per-core `group_q_start, group_pages, core_r_start, core_rows_tiles` are per-core RT args. `Wt, Ht` are kernel args. No literal restates a knob. |
| Core ranges (BB-17) | Injector-reader kernel: the `b` injector rows over the used columns. Receiver-reader kernel: the remaining rows. Writer kernel: the whole used set. Each is ≤ 2 rectangles per band (W=512: 2 + 2 + 1 ranges). Roles are chosen by core placement; the data path never changes. |
| Zones and role dump (BB-21) | Every block operation and every blocking wait sits in its own `DeviceZoneScopedN` zone (`G6E-load`, `G6E-replicate`, `G6E-publish`, `G6E-receive-wait`, `G6E-store`, `G6E-drain`), compiled out of the plain build. `plan.core_roles` is the role dump. |
| L1 (BB-05) | `l1_bytes_per_core = t_block_depth·block_pages·page_bytes ≤ 1,347,776` is guaranteed by the `block_pages` clamp, never by refusing or tagging a shape. Production: 2048 B/core. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Poisoned x padding leaks | The whole-page read brings rows 1–31 (golden poison 1000.0) into the slot. Any row of faces 0–3 left unreplicated ships padding into a logical out element. | The replicate overwrites all of rows 1–15 of faces 0/1 and all of faces 2/3, sourcing only bytes `[0,32)` and `[512,544)`. The poison loose cases and `test_input_padding_does_not_leak` pin it. |
| Multicast lands in the wrong slot | The landing address is the injector's write pointer, so receivers must sit on the same slot. | Lockstep nominal `block_pages` reserve/push on every core of a strip, including a ragged last block; one CB descriptor over all used cores. |
| RISC-V stores not yet visible to a NoC read | If the replicate uses word stores, the following multicast, self-aimed copy or DRAM write reads that L1 through the NoC. | Order the stores before the first NoC read of the page with the data-movement store-ordering idiom of the target arch (see tile_column_dataflow.hpp header note). The repeat-determinism test (BB-06) catches races. |
| Lingering NoC transactions (BB-03) | The receiver's ready-ack is a remote atomic increment that `receive()` does not drain. The injector's `send()` returns before the multicast is acked. | Receiver reader ends with `async_atomic_barrier`; injector reader and writer end with `async_write_barrier`. |
| Injector hand-off order | `push_back` before `send()` lets the local writer start early. Overwriting the slot is safe only after the NoC has read it. | `send()` with the default `SourceL1Guard::Guard` returns only after the slot is read, and the next `reserve_back` comes after it. At depth 1 the reserve also waits for the writer's pop. |
| `rs = 1` (no receivers) | Happens for `Ht = 1` or a forced plan. A multicast channel with no receivers must not wait for acks. | Host attaches the channel absent (`ttnn.attach_absent_mcast`). The injector skips `send()` and still writes all rows. Pinned by `test_pinned_plans`. |
| Same-bank writes at W=512 | `Wt = 16` is a multiple of the bank count, so each group's pages all sit on one bank. | Correct as is; lamp L6 measures staggering. |
| Grid-dependent plans | `b`, `rs` and `G` come from the device grid; a path that only some grids reach can pass here and fail elsewhere. | Production plans are identical on 13×10 and 11×10. `test_pinned_plans` forces `rs = 1`, `b = 1` (ragged 2-page groups) and `block_pages = 1` (multi-block, depth 2) on any grid. |
| Stale program on a cache hit | A Python descriptor cache with an incomplete key replays the wrong program. | The key holds every input of `select_plan` and the descriptor builder. Addresses are the only per-call values, in common RT args. `test_repeat_and_program_cache` checks zero new device-cache entries and bit-identical repeats. |

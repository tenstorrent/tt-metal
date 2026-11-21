# Operation Design: nlp_create_qkv_heads

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel: every output element is one input element, unchanged, rule R1) |
| Goal | Cut a fused QKV row tensor `[B, 1, S, (nh + 2·nkv)·hd]` into three head tensors q `[B, nh, S, hd]`, k `[B, nkv, S, hd]`, v `[B, nkv, S, hd]`, in one `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec`. |
| Math | `q[b,h,s,d] = x[b,0,s,h·hd+d]`; `k[b,h,s,d] = x[b,0,s,(nh+h)·hd+d]`; `v[b,h,s,d] = x[b,0,s,(nh+nkv+h)·hd+d]`. With `hd % 32 == 0` this is a **tile permutation**: output tile `(b, h, r, c)` of section `sec` = input tile `(b, r, col = (first_head(sec)+h)·hd_t + c)`, bytes unchanged. |
| References | `eval/specs/nlp_create_qkv_heads/spec.py` (authoritative: DEFINITION, RULES R1–R10, REJECTIONS X1–X6, PHASE0, GUIDANCE); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/toy_thread_plan/` (host program shape, implicit-sync reader/writer: `kernels/reader_implicit.cpp`, `kernels/writer_implicit.cpp`, `toy_thread_plan_program_artifacts.py`); `tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/test_tensor_accessor_addrgen.cpp:325-365` (tt-metal's own DM→DM tensor copy through a STRIDED "staging" ring, implicit sync on both sides, no compute kernel); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A; catalog `ttnn/ttnn/operations/examples/master.md` entries `tile_reorder`, `double_buffer`, `noc_placement`, `width_split`, `bank_stagger`, `split_reader` |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (fixed on this part; confirmed by probe `probes/probe_002.py`: `compute_with_storage_grid_size() = 8-4`). Op queries `device.compute_with_storage_grid_size()`. | target page + probe |
| L1 per core | 4,194,304 B (4 MiB) per cluster | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` = 3,963,328 − 0 = 3,963,328 B on this target (probe `probes/probe_002.py`). The op needs ≤ 32 KiB of it (see `l1_ledger.md`), so no budget predicate is evaluated; the query is named for any future one. | probe |
| Alignment | L1 16 B, DRAM 64 B (`ttnn.get_l1_alignment()`, `ttnn.get_dram_alignment()`); irrelevant here: every transfer is one whole tile page | target page + probe |
| DRAM channels | 2 | target page |
| Host program | `ttnn.ProgramSpec` + `ttnn.experimental.ProgramRunArgs` through `ttnn.generic_op(io_tensors, spec, run_args, tensor_args)` — reference `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/toy_thread_plan_program_artifacts.py`) | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}); 6 user DM harts (R + W ≤ 6); STRIDED rings: the producer count and consumer count of a ring divide one another (`DataflowBuffer.md:253-254`). This op has **no compute kernel**, so its only ring is DM→DM (reader harts → writer harts). | target page *Thread plan*, `DataflowBuffer.md` C2 |
| Unavailable here | (1) `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ program factory — refused at kernel creation (target page). (2) `bfloat8_b`, `bfloat4_b` — no path on this part (target page *Family facts*); not in this op's TARGET anyway. (3) Rules that touch this design: every transfer here is implicit-sync tagged and each kernel hands its tag back; **`UnpackToDest` of a bfloat16 page into 32-bit DEST** is not used — binds the deferred k-transpose regime (bf16 must run with 16-bit DEST there); **ttsim: set `fpu_math_fidelity` on every compute kernel** — binds the deferred k-transpose regime; **uint16/uint32 are not buffer formats** — not used (TARGET dtypes are bfloat16, float32). | target page |

All worked numbers below are for this target (32 clusters, 3,963,328 B unreserved). Every predicate the op evaluates queries the device.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input` | `ttnn.Tensor` | yes | rank 4, `shape[1] == 1`, TILE layout, interleaved DRAM/L1 | — | tensor parameter `input` |
| `num_heads` | int (keyword-only) | yes | ≥ 1 (X1); not bool | — | host only: folded into per-core walk state and the writer's section sizes |
| `num_kv_heads` | int or None (keyword-only) | no | ≥ 1 (X2); None → `num_heads` (R4) | None | host only (same) |
| `transpose_k_heads` | bool (keyword-only) | no | {False, True}; PHASE0 = False only | **True** (tt-metal's default; a call that omits it is refused in Phase 0 by the support gate, R10) | host only (selects regime; Phase 0 regime has no transpose) |
| `memory_config` | `ttnn.MemoryConfig` or None (keyword-only) | no | interleaved DRAM / L1; a non-MemoryConfig (X5) → `TypeError` via the spec's `canonical_memory`; a sharded config is refused by name (R6) | None → the input's buffer, interleaved (R5) | host only (output allocation) |

Derived host quantities (single source: the op's host function, computed once per call):

| Symbol | Definition |
|--------|------------|
| `kv` | `num_heads if num_kv_heads is None else num_kv_heads` |
| `sections` | `num_heads + 2·kv` |
| `head_dim` | `W // sections` (`W = input.shape[3]`; X3 rejects `W % sections != 0`) |
| `tensor_head_dim_tiles` (`hd_t`) | `ceil(head_dim / 32)` |
| `tensor_row_tiles` (`St`) | `ceil(S / 32)`, per image (each `(b,0)` image is tile-padded independently) |
| `tensor_width_tiles` (`Wt`) | `ceil(W / 32)`; in the built regime `Wt == sections·hd_t` (holds because `hd % 32 == 0`) |
| `num_pages` | `input.buffer_num_pages()` = `B·St·Wt` |
| `page_bytes` | `input.buffer_page_size()` (2048 B bf16, 4096 B fp32); equal to every output's page size (same dtype, same 32×32 tile) |
| `section_heads` | `(num_heads, kv, kv)` for (q, k, v) |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[B, 1, S, sections·head_dim]` (X4 rejects rank ≠ 4 or `shape[1] ≠ 1`); Phase 0: `B = 1`, `S % 32 == 0`, `head_dim % 32 == 0` |
| Dtype | Phase 0: bfloat16. TARGET: bfloat16, float32 |
| Layout | TILE (row-major refused by name, X6 / R9 — through the SUPPORTED gate, `layout` axis) |
| Memory | Phase 0: DRAM interleaved. TARGET: DRAM or L1, interleaved (R7). Axes read with `eval.memory.memory_axes(input)` (R8) |

### Output

| Property | Value |
|----------|-------|
| Shape | q `[B, nh, S, hd]`; k `[B, kv, S, hd]` (transposed regime, deferred: `[B, kv, hd, S]`); v `[B, kv, S, hd]` — exactly the spec's `head_shapes` |
| Dtype | input's dtype, each |
| Layout | TILE, each |
| Memory | each a tensor of its own (R5), allocated before dispatch with `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=<buffer>), device)` (`ttnn/cpp/ttnn-nanobind/tensor.cpp:330-349`), `<buffer>` = the `memory_config`'s buffer type, or the input's buffer type when `memory_config is None`. Returned as the Python list `[q, k, v]` (R2). |

**Entry point and validation order** (host; `validate()` raises `NotImplementedError` for unsupported cells, R10):

| Step | Check | Error |
|------|-------|-------|
| 1 | rank 4 and `shape[1] == 1` (X4) | `ValueError` |
| 2 | `num_heads` int ≥ 1, not bool (X1); `num_kv_heads` None or int ≥ 1 (X2) | `ValueError` |
| 3 | `W % sections == 0` (X3) | `ValueError` |
| 4 | `memory_config` → `output_memory` via the spec's `canonical_memory` (raises `TypeError` on a non-config, X5; a sharded config reads `"L1/height"` etc., a value outside SUPPORTED → refused by name, R6) | `TypeError` / `NotImplementedError` |
| 5 | axes: `dtype`, `layout`, `memory_config` + `sharding` from `eval.memory.memory_axes(input)` (R8), `transpose_k_heads`, `output_memory`, and `INPUT_TAGGERS = AXES` (imported from `eval.specs.nlp_create_qkv_heads.spec`, never redefined) applied to `(tuple(input.shape), num_heads, num_kv_heads)` | — |
| 6 | every axis value in SUPPORTED (initially = PHASE0), no EXCLUSIONS match | `NotImplementedError` |

No index axis (`dim`) exists in this op, so no canonicalization step is needed.

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

### Axes

The op's tile index space has four axes. Because the op does the *same* thing (one whole-page copy) to every input tile and the four axes are contiguous in the input's page order, they **fold exactly into one axis, the input page id** `p = ((b·St + r)·sections + head)·hd_t + c` (= `(b·St + r)·Wt + col`, `col = head·hd_t + c`) — `blocking-model.md` §1 "leading axes fold into one when the op treats them identically and they are contiguous". The block is a 1-D rectangle on that folded axis: a contiguous range of input pages. Each original axis keeps its row and its decision.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (b, `B`) | independent — no output element depends on another sequence | folded into `block_pages` | whole core range (fold) | host: `num_pages` → `split_work_to_cores` | spread: part of the folded page range dealt across the grid | knob-turn (`batch` = batched: the walk already carries `b`; only SUPPORTED changes) |
| `token_row` (r, `St` tile rows) | independent — rows are copied, never combined | folded into `block_pages` | whole core range (fold) | same | spread (folded) | knob-turn (`h_non_aligned`: `St = ceil(S/32)`, the padding rows travel inside their tiles) |
| `head` (all `sections` heads, q then k then v) | independent — each head is its own output slab; no head reads another | folded into `block_pages` | whole core range (fold) | same | spread (folded); a core range may cross head and section boundaries — the writer's walk handles that | knob-turn |
| `head_dim_tile` (c, `hd_t`) | independent — tile columns of a head are copied separately | folded into `block_pages` | whole core range (fold) | same | spread (folded) — a range may start mid-head, which is what fills the grid when `B·St·sections < 32` (e.g. `[1,1,32,4096]`, 8+4+4 heads of 8 tiles: 16 heads, 128 pages → 4 pages on each of 32 clusters) | knob-turn; **scheme-change** when `hd % 32 ≠ 0` (an output tile is assembled from parts of two input tiles — regime `subtile_assemble`) |
| folded `page` (p) | independent (all four above are) | `block_pages` = pages per block | `core_pages` (the core's whole assignment: `num_blocks_this_core = 1`) | host constant `BLOCKS_PER_CORE = 1`; `block_pages = ceil(core_pages / BLOCKS_PER_CORE)` | `ttnn.split_work_to_cores(grid, num_pages, row_wise=True)`: contiguous ranges, sizes differ by ≤ 1 page; `min(num_pages, 32)` clusters engaged | knob-turn |

There is **no dependent axis** (nothing is reduced or scanned) and **no reuse-shared operand** (one input; each input page feeds exactly one output page).

**Operand-reuse check** (every operand × the chosen split): the only operand, `input`, varies along the split axis (each core reads a disjoint page range) → no core re-reads another's bytes → no broadcast regime exists.

**Stall-shadow check**: the only waits are the writer waiting for the reader's ring credits and the reader waiting for ring space. Nothing is independent of those waits except the next transfer, which the ring depth already overlaps (reads of entries `i+1 … i+depth−1` run while entry `i` is written). No algebraic reorder applies — the op computes nothing. Nothing to schedule into the stall.

**Intermediate data**: the scheme creates none (no stage produces data another stage reduces); the ring holds input pages in transit only.

**The second scheme reading of the block**: bytes moved are constant in `block_pages` (`blocking-model.md` §4: no overlapping footprint, no shared operand re-fetched per block), so the extent decision is made on fixed per-block cost alone: one block per core pays the per-block costs (runtime-arg fetch of the walk start state, the walk-state setup, the end-of-kernel credit flush and tag reset) once per core. More blocks per core buy nothing — no residency is at stake (the ring streams), so `BLOCKS_PER_CORE = 1` is the coarsest value and the default.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `RING_DEPTH_TILES` (host constant) → `ring_entries = align_up(RING_DEPTH_TILES, 4·max(R, W))` | 8 tiles (16 KiB bf16) | Tagged NoC reads in flight ahead of the writer: the runtime splits the ring into 2 (up to 4) transaction-id slices (`DataflowBuffer.md` "Implicit sync" / C4), so with 8 entries the reader keeps one 4-tile slice landing while the DM0 ISR credits the other and the writer drains it — read/write overlap. Catalog `double_buffer`: ~4–8 outstanding transfers per batch, double-buffered (`2·block`), is the sweet spot; deeper wastes L1 for no gain once DRAM-bound. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Implicit-sync ISR batching (`DataflowBuffer.md:410-430`): `num_entries / num_txn_ids` must divide over the side's harts and their tile counters; the runtime picks `num_txn_ids ∈ {1,2,4}` | `ring_entries` | `ring_entries = align_up(RING_DEPTH_TILES, 4·max(R, W))` (divisible by 4 txn ids × each side's harts) | host rejects the DFB (no txn-id count works) |
| DFB ring size ≤ 1,048,560 B (target page *Cluster model*) | `ring_entries · page_bytes` | 8 · 4096 = 32 KiB at most in the built regime — 32× under; no clamp needed while `RING_DEPTH_TILES ≤ 255` | build-time rejection |
| One NoC transaction per implicit-sync STRIDED entry (`DataflowBuffer.md` "Each such call issues one NOC transaction") and `NOC_MAX_BURST_SIZE = 256 words × 256 B` on Quasar (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:395-396`) | entry size = one tile page | entry = exactly `page_bytes` (2 KiB / 4 KiB ≤ 64 KiB) | — |
| STRIDED ring P/C divisibility (`DataflowBuffer.md:253-254`), R + W ≤ 6 (target page) | thread plan (R, W) | host `validate_thread_plan`: `R ≥ 1, W ≥ 1, R + W ≤ 6, (R % W == 0 or W % R == 0)` | ring construction fails |
| Every thread of a STRIDED side takes its own entries `t, t+N, …` | per-core page count vs threads | none needed: the unit dealt is the ring entry itself (a page), so a thread's share may stop anywhere (target page *Thread plan*, third bullet) | — |
| 32-bit runtime args / page ids | `num_pages` | `num_pages < 2^32` (75 MB = 36,864 bf16 pages for the largest model call — far below) | wrap-around addressing |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `tile_relocate` | **built** (Phase 0; also serves the later `h_non_aligned`, `batched`, `float32`, L1-input and L1/None-output cells, each a knob-turn of SUPPORTED + `page_bytes`/dtype only) | `head_dim % 32 == 0 and not transpose_k_heads` | `block_pages` contiguous input pages = the core's whole range (`BLOCKS_PER_CORE = 1`) | **minimum** against the DRAM boundary: every input page crosses DRAM once (one read), every output page once (one write); 0 cross-core bytes. Traced vision call `[1,1,128,3072]`: 384 pages, 786,432 B read + 786,432 B written | Per-block fixed cost here = runtime-arg fetch + writer walk-state decomposition + DM0's end-of-kernel partial-batch credit and the tag reset: intended once per core. No LLK inits, no reconfigs, no explicit handshakes exist in this regime (implicit sync: the ISR credits per txn-id slice, i.e. every `ring_entries/num_txn_ids` pages, independent of the block). |
| `tile_relocate_k_transpose` | **deferred** — PHASE0 pins `transpose_k_heads = False` (both traced callers pass False); a refinement adds it. Reachable: the writer's walk already classifies every page by section, so the k-section pages are re-routed without touching the q/v path. | `head_dim % 32 == 0 and transpose_k_heads` | same page range | minimum (same crossings); adds one L1 round trip per k tile through a compute kernel (core-local) | same, plus the compute kernel's `transpose_init` once per kernel (not per block or per tile) |
|  | Topology of the deferred row: reader routes k-section pages to a second ring `cb_k_tiles` (reader → compute), the compute kernel transposes each tile in DEST (`transpose_init` / `transpose_tile`, `tt_metal/hw/inc/api/compute/transpose.h:37,120`; on Quasar the tile is unpacked to DEST and math transposes it within and across faces, `transpose.h:76-80`) and packs to `cb_k_transposed` (compute → writer), which the writer stores at k page `((b·kv + h)·hd_t + c)·St + r` (the tile's (r, c) exchange). q and v pages keep the direct reader→writer ring. Exactness (R1): float32 needs `fp32_dest_acc_en = True`; bfloat16 must keep 16-bit DEST; `fpu_math_fidelity` set explicitly. The reader then has two output rings (one per destination kernel) — each ring still one producer, one consumer. | | | | |
| `subtile_assemble` | **deferred** — PHASE0 pins `alignment = tile_aligned`; `hd % 32 ≠ 0` is in TARGET (`w_non_aligned`). A scheme-change: the unit becomes one input tile-row (`Wt` tiles) resident in L1, from which the writer assembles output tiles by copying 16-element face-row segments (a head starts and ends inside a tile; each head's own padding columns are new and must be zero-filled), then writes whole output pages. Reachable: it is the same folded page axis split at tile-row granularity (`split_work_to_cores` over `B·St` rows instead of pages; a coarsening of today's split), the reader is unchanged (contiguous input pages), and only the writer's store changes from "write the entry" to "assemble from the resident row, write". | `head_dim % 32 != 0` | `rows_per_block` input tile-rows (`Wt` pages each) | minimum (each input page read once; the row stays resident while all its heads' tiles are assembled) | one resident row amortizes reads across `sections·hd_t` assembled tiles; bigger `rows_per_block` amortizes the explicit-sync handshakes on the assembly ring |
| `row_unit_split` (GUIDANCE's unit: one tile-row per work unit) | **rejected** — superseded by `tile_relocate`'s page split: identical bytes, but only `B·St` units (4 for both traced calls) → 4 of 32 clusters busy (catalog `width_split`: a short, wide tensor split by rows strands the grid) | — | — | minimum | — |
| `compute_copy` (reader → compute `copy_tile` → writer) | **rejected** — a dead end: adds an unpack + pack per tile and a second ring for no gain on a move bound by DRAM, and a 16-bit DEST would round float32 (R1). The k-transpose regime adds compute for k pages only, on top of `tile_relocate`, not instead of it. | — | — | minimum DRAM + 2 extra L1 passes per tile | — |
| `section_pass_full_reread` (three passes, each reading whole input rows and keeping one section) | **rejected** — a dead end: 3 DRAM reads of the input instead of 1; superseded everywhere by `tile_relocate` | — | — | input crosses DRAM 3× | — |
| `three_dispatch` (one program per output) | **rejected** — forbidden: one native dispatch per invocation (task rule) | — | — | — | — |

**Regime selection** (host, after validation): `"tile_relocate" if (head_dim % 32 == 0 and not transpose_k_heads) else <deferred regime>`; in Phase 0 the deferred regimes are unreachable because the SUPPORTED gate (`alignment == tile_aligned`, `transpose_k_heads == False`) refuses their cells first. When a second regime is built, regime-pinned tests are required (one cell per regime, asserted by the regime name the host function returns).

### Traffic ranking

All candidate splits move the **same bytes**: there is no dependent axis to combine and no shared operand to re-read, so every split reads each input page from DRAM once and writes each output page to DRAM once, with zero cross-core traffic. The candidates, ranked by bytes per tier then transfer shape:

| Split | DRAM crossings (input / each output) | Cross-core | Transfer shape | Rank |
|-------|--------------------------------------|------------|----------------|------|
| folded input-page range (chosen) | 1 / 1 | 0 | one 2 KiB page per NoC transaction, reads contiguous in page id, writes in runs of `hd_t` contiguous output pages | 1 (tie) |
| output-page range (walk the outputs' concatenated page spaces) | 1 / 1 | 0 | writes contiguous, reads in runs of `hd_t` | 1 (tie) — perf lamp below |
| (row, head) units | 1 / 1 | 0 | runs of `hd_t` both sides | 1 (tie) — coarser unit, worse balance (`[1,1,32,4096]`: 16 units → 16 clusters) |
| tile rows (`row_unit_split`) | 1 / 1 | 0 | same | 1 (tie) — strands the grid (rejected) |
| "split a dependent axis + combine" | — | — | — | does not exist (no dependent axis) |

Chosen: the folded input-page range. Tiebreaker: finest balance (±1 page per cluster, engages `min(num_pages, 32)` clusters) and the simplest reader (contiguous page ids, the tt-metal addrgen copy's sequential pattern). Occupancy was not used to rank; it decided only between equal-traffic splits.

### Block schedule

```cpp
// per cluster; num_blocks_this_core == BLOCKS_PER_CORE == 1 in Phase 0
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);       // reader
    relocate_block(block_idx);   // writer (store)
}
```

| Operation | Kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|--------|-------------|--------------------|-----------------------------------|
| `load_block` | reader | `block_pages` contiguous input pages `[block_start, block_start + block_pages)`; this reader thread takes pages `t, t+R, …` of the block | nothing beyond `ring_entries` pages in flight | `TensorAccessor(tensor::input)` built once per kernel; one tagged `noc.async_read<NocOptions::TXN_ID>` per page (one NoC transaction per ring entry — the transfer quantum, not a unit-at-a-time loop of phases); no barrier, no reserve/push (implicit sync: DM0's ISR posts credits once per txn-id slice); tag handed back once, as the kernel's last statement (`noc_async_read_set_trid(0, noc.get_noc_id())`) |
| `relocate_block` | writer | the same `block_pages` pages, in the same order; this writer thread takes entries `w, w+W, …` | the walk state `(sec, h, b, r, c)` of the current page (5 counters), the three output accessors | three `TensorAccessor`s (q, k, v) built once per kernel; walk state decomposed **on the host** from `block_start` and passed as runtime args once per core; per page: one tagged `noc.async_write<NocOptions::TXN_ID>` to `out[sec]` at page `((b·section_heads[sec] + h)·St + r)·hd_t + c`, then the walk advances `W` pages by increments (c → h → sec → r → b carries; no division on device); no barrier, no wait/pop (implicit sync; firmware waits for the tagged writes at kernel exit); tag handed back once at the end (`noc_async_write_set_trid(0, noc.get_noc_id())`) |

The reader's and writer's walks are over the same page sequence, so the ring order is the page order and no metadata travels with an entry.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (`blocking-model.md` §4) | All 32 clusters are engaged even when each gets only 4–12 pages (most Phase 0 entries); with 2 DRAM channels the move is DRAM-bound long before 32 clusters, so per-cluster launch + DM0 exit flush may dominate small calls | cap engaged clusters at `ceil(num_pages / MIN_PAGES_PER_CORE)` for `MIN_PAGES_PER_CORE ∈ {8, 16, 32}` (one host constant; the split is unchanged otherwise) |
| Overlap / ring depth | 8 entries = 2 txn-id slices of 4; the catalog's sweet spot was measured on Wormhole/Blackhole with explicit barriers, not on Quasar implicit sync | `RING_DEPTH_TILES ∈ {4, 16, 32}` |
| Thread plan | (1, –, 1) is the plan the target page gives a DRAM-bound op; the writer here also does the walk arithmetic per page | (2, –, 2) and (1, –, 2) on `ttsim-perf` (model cycles per call) — the kernels already stride by their thread counts |
| Walk order (read-contiguous vs write-contiguous) | the writer's destination jumps between q/k/v and between heads every `hd_t` pages; tt-metal's hardware AddrGen serves sequential page walks and sends jumps to software (`test_tensor_accessor_addrgen.cpp:385-400`) | output-order walk: split the concatenated output page space instead, reader takes runs of `hd_t` pages at stride `Wt` |
| Bank stagger (catalog `bank_stagger`) | every cluster starts at a different page, but consecutive clusters' ranges begin on predictable banks of the 2 DRAM channels | rotate each cluster's start within its range (`core % 2`) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM (or L1) interleaved input → cluster L1 | input tile pages (bf16 / fp32 bytes, untouched) | reader: `noc.async_read<NocOptions::TXN_ID>(acc_input, cb_input_tiles, {.page_id = p}, {})`, STRIDED ring, implicit sync | reads on the reader's default NoC (`ttnn.experimental.create_reader_datamovement_config()`) |
| reader → writer | same pages | `cb_input_tiles`, DM→DM STRIDED ring, implicit sync on both sides (tt-metal's own pattern: `test_tensor_accessor_addrgen.cpp:325-365`, ring "staging") | no compute kernel in the program |
| cluster L1 → DRAM (or L1) interleaved q / k / v | same pages | writer: `noc.async_write<NocOptions::TXN_ID>(cb_input_tiles, acc_out[sec], {}, {.page_id = dst})` | writes on the writer's default NoC (`create_writer_datamovement_config()`); catalog `tile_reorder`/`noc_placement`: whole-page transfers, reads and writes on different NoCs |

Placement axis (TARGET `memory_config`, `sharding`, `output_memory`): TARGET holds interleaved placements only. DRAM ↔ L1 for the input or the outputs is a **knob-turn** — the `TensorAccessor`s resolve either buffer; nothing in the scheme changes. No sharded value is in TARGET, so no physical-shard scheme is designed; a future sharded input would cut the same independent folded axis (a knob-turn in character), consumed in place through a ring backed on the shard.

Inter-Tensix communication: none, in any regime (no combine, no multicast).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one input tile page (the ring entry); a core's block = its contiguous page range |
| Grid | `device.compute_with_storage_grid_size()` (8×4 = 32 on this target) |
| Per-core work | `ttnn.split_work_to_cores(grid, num_pages, row_wise=True)` → group 1 gets `per_core_1` pages, group 2 `per_core_2 = per_core_1 − 1`; start pages assigned in row-major core order, cumulatively (as `toy_thread_plan_program_artifacts.py:split_tiles`). `row_wise=True` per catalog `noc_placement`. Worked (32 clusters): `[1,1,128,3072]` 16/16 heads → 384 pages, 12 each; `[1,1,128,6144]` 32/8 → 768, 24 each; `[1,1,32,96]` → 3 pages, 3 clusters; `[1,1,32,4096]` 8/4 → 128, 4 each; `[1,1,4096,96]` → 384, 12 each |
| Remainder | handled by the two groups of `split_work_to_cores` (≤ 1 page difference); `num_pages` = `input.buffer_num_pages()` = `B·ceil(S/32)·ceil(W/32)` — per image, `ceil`, never `floor(B·S/32)` |
| Per-core runtime args | reader: `start_page`, `num_pages`; writer: `num_pages`, walk start `(start_sec, start_h, start_b, start_r, start_c)` decomposed on the host from `start_page` (`b = p // (St·Wt)`, `r = (p // Wt) % St`, `col = p % Wt`, `head = col // hd_t`, `c = col % hd_t`, `sec`/`h` from `head` against `section_heads`). Compile-time: `reader_threads`, `writer_threads`, `St`, `hd_t`, `num_heads`, `kv` (writer) |
| Bound | **data movement**: zero compute per page against one 2 KiB DRAM read + one 2 KiB DRAM write per page |
| Thread plan | **(R, C, W) = (1, –, 1)**: one reader hart, **no compute kernel**, one writer hart; host constant `THREAD_PLAN = (1, 1)` (readers, writers). The op is bound by DRAM (2 channels shared by 32 clusters); the target page's *Choosing the plan* gives a DM-bound op one thread of each kind with implicit sync. No compute thread exists because there is no compute: a NEO copy would add an unpack/pack per tile and risk R1. Kernels take `reader_threads` / `writer_threads` as compile-time args and stride their page loops by them from `get_my_thread_id()`, so (2, –, 2) is a one-constant change (lamp). |
| Unit dealt to compute threads | n/a (no compute kernel). Unit dealt to DM threads of the ring: the ring entry itself (one page) — a thread's share may stop anywhere, no padding rounds |
| Tests the plan needs | (a) threads ran: with one thread of each kind, a correct result over a multi-cluster cell shows every reader and writer hart ran (each owns a disjoint page range; a missing one leaves its pages unwritten — the cell's exact check catches it); the implementer adds a test that a cell with more pages than clusters (`[1,1,128,3072]`) and one with fewer (`[1,1,32,96]`, 3 clusters) both pass exactly. (b) Device-session tests (target page *Ring protocols*; backend *On Quasar*): in one device session, run the op **after** another program (e.g. `ttnn.experimental.quasar.softmax` or a second call of this op on a different shape) and run `ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)` (and on k and v) **after** the op; both must complete and match. |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_input_tiles` | DFB `unique_id="input_tiles"` | `page_bytes` (= `input.buffer_page_size()`) | `ring_entries = align_up(RING_DEPTH_TILES, 4·max(R, W))` = 8 | streams over the folded page axis (and so over b, r, head, c); spans none — live set is the in-flight window | input dtype (`data_format_metadata = input.dtype`: bfloat16 / float32); a dataflow-only payload, never through DEST | reader | writer | implicit on both sides (tagged reads / tagged writes; no `disable_dfb_implicit_sync_for`); both kernels reset their tag at exit | whole program |

Access pattern STRIDED on both bindings (the default). No compute kernel, so the capacity is not multiplied by a C.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_pages` contiguous input pages | no (dataflow; see API Mapping) | input tensor (DRAM/L1, via `TensorAccessor`) | `cb_input_tiles`, one entry per page, credited by the DM0 ISR per txn-id slice | entries posted; the last partial slice is credited by DM0 when the reader returns |
| 2 | `relocate_block` | the same `block_pages` pages | no (dataflow; see API Mapping) | `cb_input_tiles`, one entry per page | q / k / v tensors (via three `TensorAccessor`s) | ring drained; firmware waits for the tagged writes at kernel exit |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| host: output allocation | raw_api (host) | `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(...), device)` | `ttnn/cpp/ttnn-nanobind/tensor.cpp:330-349` (TensorSpec with `memory_layout`, `buffer_type`) | shape from `head_shapes`, input dtype, `TILE_LAYOUT`, `INTERLEAVED`, buffer type per R5 | — | — | — |
| host: work split | raw_api (host) | `ttnn.split_work_to_cores(grid, num_pages, row_wise=True)`; `ttnn.corerange_to_cores` | used as in `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` (`split_tiles`) | — | — | — | `num_pages` (folded axis) |
| host: program | raw_api (host) | `ttnn.experimental.ProgramSpec`, `KernelSpec`, `DataflowBufferSpec`, `TensorParameter`, `TensorBinding`, `WorkUnitSpec`, `producer_of` / `consumer_of`, `create_reader_datamovement_config` / `create_writer_datamovement_config`, `ProgramRunArgs`, `KernelRunArgs` | `toy_thread_plan_program_artifacts.py` (whole file) | 2 kernels (`reader`, `writer`), 1 DFB, 4 tensor parameters (`input`, `q`, `k`, `v`) | — | — | `RING_DEPTH_TILES`, `THREAD_PLAN`, `BLOCKS_PER_CORE` |
| host: dispatch | raw_api (host) | `ttnn.generic_op([input, q, k, v], spec, run_args, {"input": 0, "q": 1, "k": 2, "v": 3})` — one dispatch; the op returns `[q, k, v]` itself (generic_op returns only the last tensor) | `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:33-48,72-80` | — | — | — | — |
| `load_block` | raw_api (dataflow) | `noc.async_read<NocOptions::TXN_ID>(acc_input, cb_input_tiles, {.page_id = p}, {})`; `noc_async_read_set_trid(0, noc.get_noc_id())` at exit | `tt_metal/hw/inc/api/dataflow/noc.h:252` (`async_read`), `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558`; pattern `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader_implicit.cpp` | page id per entry | — | `cb_input_tiles` | `num_pages` (RT), `reader_threads` (CT) |
| `relocate_block` | raw_api (dataflow) | `noc.async_write<NocOptions::TXN_ID>(cb_input_tiles, acc_out[sec], {}, {.page_id = dst})`; `noc_async_write_set_trid(0, noc.get_noc_id())` at exit | `tt_metal/hw/inc/api/dataflow/noc.h:411` (`async_write`), `dataflow_api.h:2579`; pattern `toy_thread_plan/kernels/writer_implicit.cpp` | destination page from the walk state | `cb_input_tiles` | — | `num_pages` (RT), `writer_threads` (CT) |

**Helpers considered and rejected** (both dataflow operations):

| Candidate | File:Line | Why it cannot express this operation |
|-----------|-----------|--------------------------------------|
| compute helpers (`eltwise_convenience.hpp` `copy`, `eltwise_chain` `CopyTile`) | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp` | there is no compute phase: the op only relocates pages. Routing pages through a NEO is the rejected `compute_copy` regime (extra unpack/pack, 16-bit DEST rounds float32 against R1). |
| `tilize` / `untilize` helpers | `tilize_helpers.hpp`, `untilize_helpers.hpp` | input and outputs are already tiled; no layout change |
| `local_copy_helpers_dataflow.hpp` (`local_addr`, `set_read_state`/`read_with_state`, `set_read_trid`) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:9-34` | L1→L1 self-aimed copies; this op's transfers are DRAM→ring and ring→DRAM through `TensorAccessor`s with implicit ring sync, which the plain `Noc` calls already are |
| `tile_column_dataflow.hpp` (`copy_tile_column`) | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:81-92` | moves one column within tiles; the built regime moves whole tiles. **Candidate building block for the deferred `subtile_assemble` regime** (face-row segment copies), to be re-read then. |
| `mcast` helpers | `ttnn/cpp/ttnn/kernel_lib/mcast/` | no reuse-shared operand, no cross-core traffic |
| `l1_helpers.hpp` `fill_l1_range` / `zero_tile` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:47-101` | nothing to fill in the built regime; **needed by `subtile_assemble`** to zero each head's new padding columns |

No compute kernel exists, so `compute_kernel_hw_startup()` does not apply. (The deferred k-transpose regime's compute kernel calls it before `transpose_init`.)

## Structural impossibilities

None found beyond the specification's `INVALID = []`.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Untagged transfer after an implicit-sync program; sticky tag | both kernels use implicit sync; a following `ttnn.to_layout(q, ROW_MAJOR_LAYOUT)` issues untagged transfers | every transfer `NocOptions::TXN_ID`; reader ends with `noc_async_read_set_trid(0, noc.get_noc_id())`, writer with `noc_async_write_set_trid(0, noc.get_noc_id())`; the device-session tests in Work Distribution |
| `generic_op` returns only the last io tensor | three outputs | io list `[input, q, k, v]`; the op returns its own `[q, k, v]` list (R2), not the return value |
| Default `transpose_k_heads=True` | the harness's default-call test and any caller that omits the argument | Phase 0 refuses it through the SUPPORTED gate with `NotImplementedError` (R10); never silently returns an untransposed k |
| `memory_config=None` | R5: outputs go to the input's buffer | read the input's buffer with `memory_axes(input)` and allocate there, interleaved |
| A core range starting mid-head / mid-section / mid-row | the folded split cuts anywhere | walk start state decomposed on the host from `start_page`; device advances only by carries |
| `head_dim` vs input width alignment | a row of whole tiles can hold heads that are not (`hd = 16` in a 96-wide row) | regime predicate is on `head_dim % 32`, not on `W % 32`; the spec's `alignment` tagger (imported) gates it identically |
| Output page formula per image | output tensors are tile-padded per `(b, h)` image | `St = ceil(S/32)` per image; `dst = ((b·section_heads[sec] + h)·St + r)·hd_t + c` — same `St` as the input because S is shared |
| Ring entries not divisible for implicit sync | runtime picks up to 4 txn ids | `ring_entries = align_up(RING_DEPTH_TILES, 4·max(R, W))` |
| X1–X4 before the taggers | the spec's `head_dim_of` returns None for malformed entries | validation order above: structural rejections first, then axes |

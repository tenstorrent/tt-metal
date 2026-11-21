# Operation Design: zeros_like

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (write-only: no input read, no compute) |
| Goal | Return a new tensor with the input's shape, dtype, layout and memory config, every element `+0.0`, in exactly one `ttnn.generic_op` dispatch of a `ttnn.ProgramSpec`. Every core that owns output pages zeroes one page-sized scratchpad in its own L1 once, then writes that page to each output page it owns. |
| Math | `output[i] = 0.0` for every element `i`; the input's values are never read (spec R1) |
| References | `eval/specs/zeros_like/spec.py` (authoritative: DEFINITION, RULES R1–R6, GUIDANCE, PHASE0); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/toy_thread_plan/` (host program shape, `writer_implicit.cpp` / `writer.cpp` for the tagged write and the tag hand-back); `ttnn/ttnn/operations/examples/master.md` (`noc_placement`, `double_buffer`, `trid_double_issue`, `tile_reorder`, `split_reader`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (fixed on this part); the op queries `device.compute_with_storage_grid_size()` | target page |
| L1 per core | 4,194,304 B (4 MiB) per cluster, shared by its NEOs and DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` (reserve knob = 0 here); value on this target: not listed on the page, ≤ 4,194,304 B. This op's whole footprint is ≤ 64 KiB per core (ledger), so the budget appears only as a host assert | `unknown` (value); query named on the target page |
| Host program | `ttnn.ProgramSpec` (Metal 2.0) + `ttnn.generic_op(io_tensors, spec, run_args, tensor_args)`. Reference: `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/toy_thread_plan_program_artifacts.py`) | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}); 6 user DM cores (R + W ≤ 6; on STRIDED rings R \| C and W \| C) | target page, *Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ program factory (refused at kernel creation); `bfloat8_b` / `bfloat4_b` (no path on this part; not in TARGET anyway); uint16 / uint32 as `DataflowBufferSpec` / `ScratchpadSpec` format metadata (`validation/resource/scratchpad.cpp:36`; this op's scratchpad carries no format metadata); untagged NoC transfers: not used, every write is tagged with user id 0 — this is why the design does **not** use `Noc::async_write_zeros(TensorAccessor, …, scratch)` (`tt_metal/hw/inc/internal/noc_zero_dram.inl:45`, untagged `noc_async_write_impl`) | target page *Not available* |

All worked numbers below are for this target (32 clusters). Every run-time predicate uses the query, not the number.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `tensor` | `ttnn.Tensor` | yes | any tensor on device inside SUPPORTED; only its `spec` (shape, dtype, layout, memory config) and device are read | — | host only |

The signature is exactly the spec stub `zeros_like(tensor: ttnn.Tensor) -> ttnn.Tensor`. No `dtype`, `layout`, `device`, `memory_config`, `optional_tensor` (spec `TT_METAL["drop"]`). No index axes, nothing to canonicalize.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 2, 3 or 4 (`rank` axis, `AXES["rank"]`); any H, W (`alignment` axis: `tile_aligned`, `w_non_aligned`, `h_non_aligned`); volume > 0 |
| Dtype | Phase 0: `bfloat16`. TARGET adds `float32` |
| Layout | Phase 0: `TILE_LAYOUT`. TARGET adds `ROW_MAJOR_LAYOUT` |
| Memory | Phase 0: DRAM, interleaved. TARGET adds L1 interleaved. Derived with `eval.memory.memory_axes(tensor)` → `{"memory_config", "sharding"}` (R5); any sharded input is refused by name |
| Values | never read (R1). No kernel binds the input buffer |

### Output

| Property | Value |
|----------|-------|
| Shape | `tensor.shape` |
| Dtype | `tensor.dtype` |
| Layout | `tensor.layout` |
| Memory | `tensor.memory_config()` — allocated with `ttnn.allocate_tensor_on_device(tensor.spec, tensor.device())`, a new buffer (R2), same buffer type and interleaving (R3), same shape/dtype/layout (R4) |
| Padding | in TILE layout the padded region of the last tiles is written with zeros too (whole pages); nothing reads it (GUIDANCE) |

### Entry point (host, in order)

| Step | What |
|------|------|
| 1 | `validate(tensor)`: axes = `{dtype, layout, **memory_axes(tensor), **apply AXES}`; every value must be in `SUPPORTED`; else `NotImplementedError` (R6). A zero-volume tensor raises `NotImplementedError` (keeps the one-dispatch rule: there is no program to dispatch for it) |
| 2 | `out = ttnn.allocate_tensor_on_device(tensor.spec, tensor.device())` |
| 3 | `spec, run_args, tensor_args = create_program_artifacts(out)` — takes **only the output**, so a test can hand it a pre-filled tensor (see Work Distribution, tests) |
| 4 | `return ttnn.generic_op([out], spec, run_args, {"out": 0})` — one dispatch; a single-element `io_tensors` list is legal (`generic_op_nanobind.cpp:35-38`) |

`INPUT_TAGGERS = AXES` imported from the spec; never redefined. `SUPPORTED` = `PHASE0`.

## Blocking Model

The op's data is one output tensor and one constant (a page of zero bits). There is no reduction, no scan and no operand that varies along any axis, so every output axis is **independent**, and the only shared operand — the zero page — is generated **locally** on every core (iDMA zero mode, no bytes over the NoC), never read or multicast.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (all leading dims folded: rank-4 `N·C`, rank-3 `B`, rank-2 `1`) | independent — each image's pages are written in isolation | folded into `output_page` | whole extent, folded | — (folding rule below) | via `output_page` | — |
| `tile_row` (`ceil(H/32)` per image) | independent — no value depends on another row | folded into `output_page` | whole extent, folded | — | via `output_page` | — |
| `tile_col` (`ceil(W/32)` per image) | independent — no value depends on another column | folded into `output_page` | whole extent, folded | — | via `output_page` | — |
| `output_page` (flat page index `0 … num_pages-1`, the buffer's own page order) | independent — every page is the same constant, written once | `pages_per_barrier` — pages issued between two write barriers (the block) | `min(core_pages, PAGES_PER_BARRIER)` with `PAGES_PER_BARRIER = 256` → for every `INPUTS` shape the whole per-core assignment (≤ 32 pages) is **one block** | host constant `PAGES_PER_BARRIER`, passed as CT arg `pages_per_barrier`; kernel derives `num_blocks_this_core = ceil(num_pages_this_core / pages_per_barrier)` and `last_block_extent` | contiguous ranges of `output_page` over the full grid: `ttnn.split_work_to_cores(grid, num_pages, row_wise=True)` → RT args `start_page`, `num_pages_this_core` | knob-turn (raise / lower `PAGES_PER_BARRIER`; it is clamped by the trid cap below) |
| `page_chunk` (bytes within one page) | independent — byte ranges of a constant | `zero_src_bytes` — bytes per NoC write = scratchpad size | `align_up(page_bytes, l1_alignment)`, clamped to `ZERO_SRC_MAX_BYTES = 65536` → TILE bf16: 2048 B = **whole page in one write**; `chunks_per_page = ceil(page_bytes / zero_src_bytes) = 1` | host computes `zero_src_bytes` from `out.buffer_page_size()` and `ttnn.get_l1_alignment()`; it sizes the `ScratchpadSpec` and is the CT arg `zero_src_bytes`; kernel derives `chunks_per_page` from CT `page_bytes` | not split across cores (a page is never shared between cores) — whole page owned by the page's core | knob-turn (only matters for ROW_MAJOR pages wider than 64 KiB) |

**Folding rule.** `batch`, `tile_row`, `tile_col` are contiguous in the interleaved buffer's page order and the op treats them identically, so they fold into the one flat `output_page` axis. The page count is the buffer's: `num_pages = out.buffer_num_pages()`, which in TILE layout equals `prod(leading dims) · ceil(H/32) · ceil(W/32)` — **per image, with `ceil`**, never `floor(batch·H/32)` (e.g. `(2,1,100,47)` → `2·4·2 = 16`, `(70,512)` → `3·16 = 48`). In ROW_MAJOR (refinement) the same call gives `prod(shape[:-1])` rows. One source: the buffer.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `zero_page` (scratchpad, not a ring) | none — fixed at 1 | 1 | Nothing to buy: the scratchpad is written once (zeroed) and then only **read** by every NoC write; any number of writes may read it while in flight, so a second copy overlaps nothing. Write overlap comes from `pages_per_barrier` (writes in flight), not from depth. |

There are no dataflow buffer rings in this op (no reader, no compute).

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Per-trid outstanding-write counter (`NOC_MAX_TRANSACTION_ID_COUNT = 0xFFFF`, `tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:36`) | `pages_per_barrier × chunks_per_page × packets_per_chunk` (packets ≤ 8 KiB each, `noc_cmd_buf_common.h:74`) | `pages_per_barrier = min(num_pages_this_core, PAGES_PER_BARRIER)`, `PAGES_PER_BARRIER = 256`: at ≤ 64 KiB per chunk that is ≤ 256 × 8 = 2048 packets per barrier ≪ 0xFFFF | — |
| Scratchpad as a NoC write source: one write reads `[0, zero_src_bytes)` of it (`internal/local_mem/noc_traits.h:85-89`, offset asserted `< size_in_bytes()`) | `zero_src_bytes` (bytes per write) | `zero_src_bytes = min(align_up(page_bytes, l1_alignment), ZERO_SRC_MAX_BYTES)`; a page larger than that is written in `chunks_per_page` writes at `offset_bytes = k·zero_src_bytes` (a multiple of 64 → DRAM-aligned) | a write longer than the scratchpad streams uninitialized L1 into the output (wrong values) |
| iDMA zero mode borrows overlay cmd buffer 0 until `write_zeros_l1_barrier()` (`noc.h:833-838`; `tt-2xx/noc_zero_l1.inl:110-128`) | ordering, not extent: no NoC write may be issued between `async_write_zeros` and its barrier | `zero_source` completes (barrier) before the first `store_block` write | writes issued in zero mode carry zeros to wrong places / corrupt their data |
| One writer thread owns the scratchpad (`scratchpad_spec.hpp`: no synchronization across threads) | `W` (writer threads) | Phase 0 `W = 1`; a `W > 1` plan gives each thread its own `zero_src_bytes` slice (`size_per_node = W·zero_src_bytes`) so no thread reads bytes another is still zeroing | a thread writes un-zeroed L1 to DRAM |
| ProgramSpec work unit: DM kernels' threads ≤ 6 user DM cores (`validation/placement/work_unit.cpp:82-87`) | `W` | `W ≤ 6` | program rejected at validation |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `fill_from_local_zero_page` | **built** | every SUPPORTED cell (interleaved output, any layout/dtype/buffer in TARGET). Domain = all of TARGET × INPUTS. Single built row: no axis is dependent and no operand varies, so no combine or reuse regime can apply | `pages_per_barrier` pages × `zero_src_bytes` (one page in TILE) | named boundary DRAM (L1 for an L1 output): **output crosses once (written), input crosses 0 times** — below the generic "each input once" minimum because R1 forbids reading it; this *is* the minimum. Cross-core: 0 B. Core-local: one iDMA zero of `zero_src_bytes` per core | the per-block fixed cost is **one trid write barrier** (a pipeline drain: the NoC goes idle until the block's last write acks). Intended frequency: once per block = once per core for every INPUTS shape. The zero-source init (iDMA zero + its barrier) is once **per kernel**, independent of the block. No inits, reconfigs or ring handshakes exist |
| `fill_bank_runs` (each core writes runs of pages that are contiguous within one DRAM bank as one multi-page transfer from a multi-page zero source) | **deferred** — the only addressing contract the op uses, `TensorAccessor`, exposes per-page addresses; bank-run contiguity (page `p` and `p + num_banks` adjacent in one bank at the aligned page stride) is an allocator layout the interleaved buffer contract does not state, and a whole page per write already is the coalesced unit (catalog `tile_reorder`); the library records the same open idea (`noc_zero_dram.inl:10-11`) | would be: interleaved DRAM output and `aligned_page_size == page_bytes` | `run_pages × page_bytes` per write | same bytes as built; fewer transactions (÷ run length). Adds: a core-assignment by bank instead of by contiguous range | fewer issue/ack round trips per byte. Reachable: the `store_block` operation's contract (write pages `[start, start+n)` of `out` with zeros) is unchanged; only its realization and the host's per-core ranges change |
| `broadcast_zero_page` (one core zeroes a page and multicasts it to the others) | **rejected** — superseded by `fill_from_local_zero_page`: each core makes the page locally with zero NoC bytes; a multicast adds `31 × zero_src_bytes` of NoC traffic plus a semaphore rendezvous to deliver bytes every core can generate for free | — | — | +31 pages cross-core, +1 rendezvous; DRAM unchanged | — |
| `compute_packs_zeros` (a NEO fills DEST with zero and packs a zero tile into a ring the writer drains, once per output page or once and held) | **rejected** — superseded by `fill_from_local_zero_page`: identical DRAM bytes, but adds a compute kernel, a ring with its credit protocol, and (per page) one pack per output page; no FPU/SFPU work is needed for zero bits (GUIDANCE) | — | — | DRAM unchanged; +1 ring, +num_pages packs | — |
| `read_and_multiply_by_zero` (read the input, `x · 0`) | **rejected** — dead end: reads the input (an extra full DRAM crossing), and is **wrong** for NaN/Inf inputs (`0·NaN = NaN`), violating R1. Superseded everywhere by not reading | — | — | input +1 DRAM crossing | — |

Only one regime is built, so there is no selection predicate and no regime-pinned test beyond the Phase 0 cells.

### Traffic ranking

| Candidate split | DRAM bytes | Cross-core bytes | Transfers | Verdict |
|-----------------|-----------|------------------|-----------|---------|
| `output_page` in contiguous ranges over the full grid (`row_wise=True`) | `num_pages · page_bytes` written, 0 read | 0 | `num_pages · chunks_per_page` writes | **chosen** — the minimum on every tier; contiguous ranges keep one RT pair per core |
| `output_page` dealt by DRAM bank (`fill_bank_runs`) | same | 0 | fewer (runs) | deferred row above — same bytes, fewer transactions, relies on an unstated layout |
| split by `batch` / `tile_row` only | same | 0 | same | dominated: same bytes, but under-fills the grid for small-batch shapes (e.g. `(1,1,32,8192)` has 1 image, 1 tile-row) — the flat page split fills it |
| "split a dependent axis + combine" | n/a | n/a | n/a | **no dependent axis exists** (no reduction, no scan): nothing to combine |
| Operand-reuse check: the input tensor | 0 | 0 | — | not read at all (R1) — no reuse question |
| Operand-reuse check: the zero page | 0 | 0 (local iDMA) | 1 local zero per core | does not vary along `output_page` → reuse-shared by construction of the split; the "broadcast it" row is `broadcast_zero_page`, rejected: local generation is cheaper than any broadcast |

Occupancy is kept out of this ranking: every candidate moves the same DRAM bytes, so the flat split wins on the tiebreaker (no combine, full grid).

**Stall-shadow check.** The writer waits in two places: `write_zeros_l1_barrier()` after zeroing the source, and the per-block trid write barrier. Nothing can fill either window: every write reads the zero page (so none may start before the first barrier, and cmd buffer 0 is borrowed until then), and the final barrier is the kernel's last act. The page-address computation is folded into each write by the accessor. Recorded: no independent work exists.

### Block schedule

```cpp
zero_source();                                   // once per kernel
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    store_block(block_idx);                      // pages [start_page + block_idx*pages_per_barrier, +extent)
}
release_tag();                                   // once per kernel
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `zero_source` | `zero_src_bytes` of the `zero_page` scratchpad | — (creates the resident page) | one `noc.async_write_zeros(zero_page, zero_src_bytes)` + one `noc.write_zeros_l1_barrier()` **per kernel** |
| `store_block` | `extent = (block_idx < num_blocks_this_core-1) ? pages_per_barrier : last_block_extent` pages × `chunks_per_page` chunks of `zero_src_bytes` | `zero_page` (read-only, all of it) | `extent · chunks_per_page` tagged writes (`trid = 0`), then **one** `async_write_barrier<TXN_ID>({.trid = 0})` per block — no barrier per page |
| `release_tag` | — | — | `noc_async_write_set_trid(0, noc.get_noc_id())` once, last statement (target page, ring protocols: hand the tag back) |

There is no `load_block` and no compute phase: the op reads nothing and computes nothing.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (participation) | Small outputs give each core 1–2 pages (e.g. `(70,512)`: 48 pages → 16 cores × 2 + 16 × 1; `(32,17)`: 1 page on 1 core). Per-core launch cost may dominate one 2 KiB write | `MIN_PAGES_PER_CORE` host knob (Phase 0 = 1): engage `min(grid, ceil(num_pages / MIN_PAGES_PER_CORE))` cores; compare 1 vs 4 vs 8 |
| Overlap (barrier drain) | One barrier per block drains the NoC; with blocks > `PAGES_PER_BARRIER` (shapes beyond INPUTS: > 8192 pages) every block boundary idles the writer's NoC port | two alternating trids (ping-pong): retire block `k-1`'s id while block `k` is in flight (catalog `trid_double_issue`); or raise `PAGES_PER_BARRIER` within the trid cap |
| Writer threads | One DM hart issues every write; catalog `split_reader` shows a single DM RISC can saturate on **issuing** transactions before DRAM saturates | `W = 2` (pages dealt round-robin to the two harts, scratchpad `2·zero_src_bytes`, one slice per thread); measure on `ttsim-perf` as model cycles |
| Core placement | `row_wise=True` spreads a short line of cores along the DRAM-facing axis (catalog `noc_placement`, measured on WH); Quasar's 2 DRAM channels and single NoC are a different topology | `row_wise=False` |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Host: output allocation | — | `ttnn.allocate_tensor_on_device(tensor.spec, device)` | R2/R3/R4: new buffer, input's memory config |
| Core L1: zero source | raw bytes (zero bits are the same in bf16, fp32, TILE and ROW_MAJOR) | `ScratchpadSpec("zero_page", size_per_node = W·zero_src_bytes)`, no `data_format_metadata`; kernel: `Scratchpad<uint32_t> zero_page(scratch::zero_page)`; filled by `noc.async_write_zeros` (Quasar iDMA zero mode, bypasses the DM cache) | zeroed once per kernel |
| Core L1 → output buffer | page bytes | `noc.async_write<NocOptions::TXN_ID>(zero_page, out_accessor, zero_src_bytes, {.offset_bytes = 0}, {.page_id = p, .offset_bytes = k·zero_src_bytes}, {.trid = 0})` per chunk; `noc.async_write_barrier<NocOptions::TXN_ID>({.trid = 0})` per block | `TensorAccessor(tensor::out)` resolves DRAM **or L1** interleaved banks; explicit protocol (no rings), every transfer tagged with user id 0 |
| Input buffer | — | not bound to any kernel | R1 |

**TARGET axes seen as this scheme** (each is a knob-turn on the built regime: no new algorithm):

| TARGET value beyond PHASE0 | Cost class | What changes |
|----------------------------|------------|--------------|
| `dtype: float32` | knob-turn | `page_bytes` 4096 → `zero_src_bytes` 4096; nothing else |
| `layout: ROW_MAJOR_LAYOUT` | knob-turn | page = one row (`W·elem_size` B, `num_pages = prod(shape[:-1])`); `chunks_per_page > 1` only for rows > 64 KiB; the accessor's aligned page size governs `offset_bytes` |
| `memory_config: L1` (interleaved) | knob-turn | the same accessor addresses L1 banks; output allocated in L1 by step 2. (`async_write_zeros`' DRAM overload would have `static_assert`ed here, `noc_zero_dram.inl:18-19` — another reason it is not used) |
| `sharding`: only `interleaved` in TARGET | — | a sharded input is refused by name (R5). Were sharding ever added: the shard is the core's block, written in place through a buffer backed on the shard (zero-copy, no NoC write), one block per core |

No deferred regime needs a Tensix-to-Tensix contract (the only cross-core row is rejected).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output page (TILE: one 32×32 tile; ROW_MAJOR: one row); a block is `pages_per_barrier` consecutive pages of a core's range |
| Grid | `device.compute_with_storage_grid_size()` (32 clusters on this target) |
| Per-core work | `_, core_set, g1, g2, n1, n2 = ttnn.split_work_to_cores(grid, num_pages, row_wise=True)`; cores of `g1` get `n1` pages, `g2` get `n2`; `start_page` is the running sum in core order (same as `split_tiles` in `toy_thread_plan_program_artifacts.py`). RT args per core: `start_page`, `num_pages_this_core`. Worked (32 clusters): `(70,512)` 48 pages → 16×2 + 16×1; `(4,8,32,256)` 256 → 8 each; `(1,1,2048,256)` 512 → 16; `(2,512,1024)`, `(1024,1024)` 1024 → 32; `(1,1,32,50)` 2 → 2 cores × 1; `(32,17)` 1 → 1 core |
| Remainder | `num_pages = out.buffer_num_pages()` (= per-image `ceil` tile geometry); `split_work_to_cores` puts the remainder in `g1`; inside a core the last block has `last_block_extent = num_pages_this_core - (num_blocks_this_core-1)·pages_per_barrier` |
| Bound | **data movement** — zero compute per unit against one page written per unit; the op is bound by DRAM write bandwidth and by write issue |
| Thread plan | **(R, C, W) = (0, 0, 1)**, host constant `THREAD_PLAN = {"reader_threads": 0, "compute_threads": 0, "writer_threads": 1}`; the writer's count goes in as CT arg `writer_threads`. No reader (nothing is read, R1) and no compute kernel (nothing is computed, GUIDANCE), so the work unit holds the writer kernel only. W = 1: target page *Choosing the plan* — one thread of each kind for a data-movement-bound op; W = 2 is the perf lamp |
| Unit dealt to compute threads | none (no compute kernel). The writer thread `t` of `W` takes pages `t, t+W, …` of each block (with W = 1: all of them); the kernel loops from `get_my_thread_id()` and `writer_threads` so a missing thread would leave pages unwritten |
| Tests the plan needs | (1) **every page written**: build `out` pre-filled with non-zero data (e.g. `ttnn.from_torch(torch.full(shape, 7.0), …)` in the cell's layout/dtype/memory config), run `create_program_artifacts(out)` + `ttnn.generic_op([out], …)` and require all zeros — a freshly allocated buffer may already hold zeros on the simulator, so the golden cells alone do not prove a core or thread ran; use a shape that engages all 32 cores and has a ragged split (e.g. `(70,512)`, and `(1,1,2048,256)` for full blocks). (2) **one device session**: an implicit-sync tt-metal Quasar op (e.g. `ttnn.experimental.quasar.softmax`, or the `toy_thread_plan` example) → `zeros_like` → `ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT)`, all in one test, result checked |

## Circular Buffers

This op has **no circular buffers / dataflow buffer rings**. Its one L1 allocation is a kernel scratchpad, listed here so the inventory is complete:

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `zero_page` (`ScratchpadSpec` / accessor `scratch::zero_page`) | — (scratchpad, not a ring) | `zero_src_bytes` | `W` (1 in Phase 0) — one slice per writer thread | spans `page_chunk` (one chunk of a page); streams over `output_page` (reused for every page), and over the folded `batch`/`tile_row`/`tile_col` | none (raw bytes; no format metadata, DM-only) | writer (`zero_source`, iDMA zero) | writer (`store_block`, NoC source) | — (no ring credits: ordered by `write_zeros_l1_barrier()` before the first write) | whole kernel |

A scratchpad is private to its one kernel, so "producer and consumer" are the same kernel by construction; there is no inter-kernel ownership question.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input (semantic name, size, state) | Output (semantic name, size) | State after |
|---|-----------------|-------------|---------|------------------------------------|------------------------------|-------------|
| 1 | `zero_source` | `zero_src_bytes` | library NoC zero API (raw) | — | `zero_page`, `zero_src_bytes`, all zero bits | `zero_page` holds zeros for the rest of the kernel; cmd buffer 0 restored to write mode |
| 2 | `store_block` | `extent × chunks_per_page × zero_src_bytes` | raw tagged NoC write | `zero_page`, `zero_src_bytes`, read-only | `out` pages `[block_start, block_start + extent)` | pages written and acked (trid 0 barrier) |
| 3 | `release_tag` | — | raw | — | — | the hart's write tag register holds user id 0 |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input | Output | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|-------|--------|------------------------------|
| host: program | host API | `ttnn.experimental.ProgramSpec`, `KernelSpec`, `ScratchpadSpec`, `ScratchpadBinding`, `TensorParameter`, `TensorBinding`, `WorkUnitSpec`, `ProgramRunArgs`, `KernelRunArgs`, `create_writer_datamovement_config` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:365-398` (ScratchpadSpec), `:704-718` (ScratchpadBinding), `:749-787` (KernelSpec `scratchpad_bindings`), `:875-899` (ProgramSpec `scratchpads`), `:476` | ScratchpadSpec(`unique_id="zero_page"`, `size_per_node=W·zero_src_bytes`); KernelSpec(`unique_id="writer"`, `num_threads=W`, `scratchpad_bindings=[ScratchpadBinding("zero_page","zero_page")]`, `tensor_bindings=[TensorBinding("out","out")]`, CT `{writer_threads, pages_per_barrier, page_bytes, zero_src_bytes}`, RT schema `["start_page","num_pages_this_core"]`); `dataflow_buffers=[]`; WorkUnitSpec(`kernels=["writer"]`, `target_nodes=core_set`) | — | — | `pages_per_barrier`, `zero_src_bytes`, `writer_threads` (CT); `start_page`, `num_pages_this_core` (RT) |
| host: dispatch | host API | `ttnn.generic_op([out], spec, run_args, {"out": 0})` | `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:35-38,71-83` | single-element `io_tensors` | — | `out` | — |
| host: split | host API | `ttnn.split_work_to_cores(grid, num_pages, row_wise=True)` | `ttnn/ttnn/core.py:19` | — | — | — | core count (participation lamp) |
| `zero_source` | raw_api | `Scratchpad<uint32_t>(scratch::zero_page)`; `Noc::async_write_zeros(dst, size_bytes)` (overload 1); `Noc::write_zeros_l1_barrier()` | `tt_metal/hw/inc/api/scratchpad.h:55-59`; `tt_metal/hw/inc/api/dataflow/noc.h:849,904`; impl `tt_metal/hw/inc/internal/tt-2xx/noc_zero_l1.inl:26-105,110-128` | `async_write_zeros(zero_page, zero_src_bytes, {.offset_bytes = t·zero_src_bytes})` for thread `t` | — | `zero_page` | `zero_src_bytes` |
| `store_block` | raw_api | `Noc::async_write<NocOptions::TXN_ID>(src, dst, size_bytes, src_args, dst_args, noc_opts)`; `Noc::async_write_barrier<NocOptions::TXN_ID>(noc_opts)`; `TensorAccessor(tensor::out)` | `noc.h:405-417` (write), `noc.h:760-767` (barrier); Scratchpad as source `internal/local_mem/noc_traits.h:75-89` | `src = zero_page`, `size = zero_src_bytes`, `src_args = {.offset_bytes = t·zero_src_bytes}`, `dst_args = {.page_id = p, .offset_bytes = k·zero_src_bytes}`, `noc_opts = {.trid = 0}` | `zero_page` | `out` | `pages_per_barrier` (writes per barrier), `zero_src_bytes` (bytes per write) |
| `release_tag` | raw_api | `noc_async_write_set_trid(0, noc.get_noc_id())` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2579` | trid 0 | — | — | — |

**Helpers considered and rejected**

| Raw entry | Candidate helper | File:line of the mismatch | Why it cannot express this |
|-----------|------------------|---------------------------|----------------------------|
| `zero_source` | `dataflow_kernel_lib::zero_tile(dfb)` / `prepare_zero_tile(dfb)` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:53-56` | It sizes the zero from `dfb.get_tile_size()`, which a `Scratchpad` does not have (`api/scratchpad.h:42-139`); passing a ring instead needs a second kernel as the ring's other endpoint (target page: one producer kernel and one consumer kernel per ring) — a kernel this op does not otherwise need. The op calls the same two primitives the helper wraps (`l1_helpers.hpp:55-56`) on the scratchpad |
| `zero_source` (fallback only) | `dataflow_kernel_lib::fill_l1_range<4>(addr, n, 0)` | `l1_helpers.hpp:99-130` | CPU stores through the DM core's cache; the NoC reads L1 directly, so the stores need a `zero_page.scoped_lock(0, n)` scope to be visible (`api/core_local_mem.h:145-151`). Kept as the fallback if the iDMA zero mode misbehaves on the simulator (Risks) |
| `store_block` | `Noc::async_write_zeros(TensorAccessor, size, args, scratch)` (overload 2) + `write_zeros_dram_barrier()` | `tt_metal/hw/inc/internal/noc_zero_dram.inl:45,50` | Issues **untagged** `noc_async_write_impl` and barriers untagged; this op tags every write. It also `static_assert`s `DSpecT::is_dram` (`noc_zero_dram.inl:18-19`), so the L1 TARGET value would need a second path |
| `store_block` | compute-side `eltwise_convenience` / `eltwise_chain` (`copy`, `CopyTile`/`PackTile`) | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp` (compute kernels only) | Compute helpers pack into rings; this op has nothing to compute and no compute kernel (regime `compute_packs_zeros`, rejected) |
| — | `dataflow_kernel_lib` mcast (`McastArgs`) | `ttnn/cpp/ttnn/kernel_lib/mcast/` | No cross-core transfer exists (regime `broadcast_zero_page`, rejected) |

`compute_kernel_hw_startup()` does not apply: there is no compute kernel.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A passing golden cell proves nothing about coverage | A new DRAM buffer on the simulator may already be zero, so an op that writes nothing (or only some cores' pages) still passes R1 | Required test (1): pre-filled non-zero output through `create_program_artifacts(out)`; ragged split on all 32 cores |
| Tag left on the hart / untagged transfer | Implicit-sync programs (tt-metal's own Quasar ops) leave a ring's tag | Every write tagged `trid = 0`, barrier on trid 0; `release_tag` as the kernel's last statement; required test (2) in one session (before and after) |
| Writes issued while cmd buffer 0 is in iDMA zero mode | `async_write_zeros` borrows cmd buffer 0 until `write_zeros_l1_barrier()` | `zero_source` ends with the barrier before any `store_block` write |
| iDMA zero mode on ttsim | The zero device path is new; the inl notes a simulator-specific deviation (`noc_zero_l1.inl:43-48`) | If it hangs or leaves non-zero bytes: fall back to `fill_l1_range<4>(…, 0)` inside `zero_page.scoped_lock(0, zero_src_bytes / 4)` (cache flush on release); the design is otherwise unchanged |
| Writing past the scratchpad | A write of `page_bytes` from a smaller scratchpad streams garbage | `zero_src_bytes ≥` each write's size by construction; `chunks_per_page` for larger pages |
| Zero-volume tensor | `split_work_to_cores(…, 0)` has no cores, and dispatching nothing breaks the one-dispatch rule | `validate()` refuses it with `NotImplementedError` (not in INPUTS) |
| Input returned as output | R2: the result must be a new buffer | always `allocate_tensor_on_device`; never return `tensor` |
| `-0.0` vs `+0.0` | spec DECISIONS: tt-metal writes `+0.0` | all-zero bits = `+0.0` in bf16 and fp32 |

### Structural impossibilities

None to add: `INVALID = []` is right for this TARGET.

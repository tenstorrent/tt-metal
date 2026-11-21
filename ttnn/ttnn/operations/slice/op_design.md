# Operation Design: slice

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel: every output element is an input element, unchanged — spec R1) |
| Goal | Copy the contiguous block `input[s0:e0, …, s(r-1):e(r-1)]` of an interleaved tensor into a new interleaved tensor of the same dtype and layout, placed in the input's buffer (R7), in **one** `ttnn.generic_op` dispatch that relays whole pages through one DM→DM ring per core, across the full grid. |
| Math | `output[i_0, …, i_{r-1}] = input[slice_start[0] + i_0, …, slice_start[r-1] + i_{r-1}]`, `output.shape[d] = slice_end[d] - slice_start[d]` (negative indices normalized first, R2) |
| References | Specification `eval/specs/slice/spec.py` (governs); target `.claude/references/targets/quasar.md`; backend `.claude/references/backends/ttsim.md`; ring semantics `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` (Part A, C4 txn batching); host-program pattern `.claude/references/quasar_op_examples/` (`toy_thread_plan`: implicit-sync reader/writer kernels and host split) and `ttnn/ttnn/operations/toy_spec_single_tensor/toy_spec_single_tensor.py` (`create_fill_artifacts`: a ProgramSpec whose ring has two data-movement endpoints and no compute kernel); perf catalog `ttnn/ttnn/operations/examples/master.md` (entries `double_buffer`, `tile_reorder`, `noc_placement`, `split_reader`, `trid_double_issue`, `read_coalescing`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (queried at run time with `device.compute_with_storage_grid_size()`; probe on this backend returned `8-4`) | target page + probe |
| L1 per core | 4,194,304 B | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` = 3,963,328 − 0 = 3,963,328 B on this target (query value read by probe on ttsim, 2026-11-21) | probe |
| Alignment | L1 16 B (`ttnn.get_l1_alignment()`), DRAM 64 B (`ttnn.get_dram_alignment()`) — probe confirmed | target page + probe |
| Host program | `ttnn.ProgramSpec` (`ttnn.experimental.*`) dispatched by one `ttnn.generic_op` — `.claude/references/quasar_op_examples/` | target page |
| Threads per core | 4 NEO compute engines, 6 user DM harts; C ∈ {1,2,4}, R + W ≤ 6, R \| C and W \| C on STRIDED rings with a compute endpoint. **This op has no compute kernel**: its only ring is DM→DM STRIDED, for which the DFB guide requires `R % W == 0` or `W % R == 0` and `num_entries % (num_txn_ids × max(R, W)) == 0` (`DataflowBuffer.md` lines 253-254, 430-436) | target page + DFB guide |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ program factory (refused at kernel creation, target page); `bfloat8_b` / `bfloat4_b` (no path on the family — not in TARGET anyway); **uint32/uint16 as a `DataflowBufferSpec.data_format_metadata`** (refused by `tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp:143-148`) — the refinement that adds uint32 carries it as int32 or leaves the optional metadata unset; untagged NoC transfers: not used — every transfer here is implicit-sync tagged and each kernel hands its tag back. | target page |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device, interleaved (R5, R8), dtype/layout/rank per SUPPORTED | — | tensor parameter |
| `slice_start` | list or tuple of int (R3) | yes | one entry per dim (X1); negative → `+ size` (R2); after normalization `0 ≤ start ≤ size` | — | host only (feeds the page map, RT) |
| `slice_end` | list or tuple of int (R3) | yes | one entry per dim (X1); negative → `+ size` (R2); after normalization `start ≤ end ≤ size` (X2, X3) | — | host only (feeds the page map, RT) |

Entry-point order (all host side, before anything touches the device):

| # | Step | Raises |
|---|------|--------|
| 1 | `slice_start`, `slice_end` must be `list` or `tuple` of `int`; `len == input.rank` | `ValueError` (X1) |
| 2 | Normalize: `i + size if i < 0 else i`, per dim (R2) — before any support check | — |
| 3 | `end > size` → X2; `end < start` (also `start < 0` or `start > size` after normalization) → X3 | `ValueError` |
| 4 | Any dim with `end == start` (empty block: unspecified by the spec) | `NotImplementedError` |
| 5 | Axis values: `dtype`, `layout`, `eval.memory.memory_axes(input_tensor)` (R8: `memory_config`, `sharding`), and the four `AXES` functions on `(tuple(input.shape), start_norm, end_norm)`; each checked against `SUPPORTED`, then `EXCLUSIONS` (R6) | `NotImplementedError` |
| 6 | Mechanism caps (below): `entry_bytes ≤ NOC_MAX_BURST_BYTES`; `RING_DEPTH_BLOCKS × entry_bytes ≤ ring_cap_bytes` | `NotImplementedError` |

The entry point is named `slice`, shadowing Python's builtin in that module: the module must not use the builtin `slice` (use `builtins.slice` if ever needed).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank r ∈ SUPPORTED["rank"] (Phase 0: 3; Refinement 10: 3, 4); TARGET ranks 2–4 all go through the same 4-D page map (left-padded with 1s) |
| Dtype | Phase 0: bfloat16 (TARGET adds float32, uint32 — knob-turns, see Dataflow Strategy) |
| Layout | TILE or ROW_MAJOR |
| Memory | interleaved, DRAM or L1 (R5) |

### Output

| Property | Value |
|----------|-------|
| Shape | `[end_d - start_d for each d]` (spec `OUTPUT["shape"]`) |
| Dtype | input's |
| Layout | input's (TILE output is tile-padded for its own shape; padding content unspecified, R4) |
| Memory | interleaved, input's buffer type (R7): `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(ttnn.Shape(out_shape), input.dtype, input.layout, buffer_type=input.memory_config().buffer_type), device)` (constructor at `ttnn/cpp/ttnn-nanobind/tensor.cpp:306-322`) |

A block that is the whole tensor is **copied by the same program** (one dispatch). Returning the input without a dispatch is rejected (regime table): the task requires exactly one device-program dispatch per invocation.

## Blocking Model

The first design section. Everything below is downstream of it.

### The op's index space

The unit of everything in this op is the **page** of an interleaved buffer: a row of W elements in ROW_MAJOR, a 32×32 tile in TILE. Each output page is exactly one input page, unchanged and whole, within the built regime's domain (below). Define, for rank r left-padded to 4 page axes `(a0, a1, a2=h, a3=w)`:

| Quantity | ROW_MAJOR | TILE |
|----------|-----------|------|
| `in_page_shape` | `[1]*(4-r) + shape[:-2] + [H, 1]` | `[1]*(4-r) + shape[:-2] + [ceil(H/32), ceil(W/32)]` |
| `out_page_shape` | `[1]*(4-r) + out_shape[:-2] + [H_out, 1]` | `[1]*(4-r) + out_shape[:-2] + [ceil(H_out/32), ceil(W_out/32)]` |
| `page_start` | `[0]*(4-r) + start[:-2] + [h_start, 0]` | `[0]*(4-r) + start[:-2] + [h_start/32, w_start/32]` |
| `in_page_stride[d]` | `prod(in_page_shape[d+1:])` | same |
| source page of output page with multi-index `m` (row-major over `out_page_shape`) | `Σ_d (page_start[d] + m[d]) · in_page_stride[d]` | same |

Every tile count uses `ceil` per image (each `(…, H, W)` image is tile-padded independently); `num_out_pages = prod(out_page_shape)` and the host asserts it equals `output.buffer_num_pages()`. Correctness of the TILE map needs `h_start % 32 == 0 and w_start % 32 == 0` (then `page_start[d] + out_page_shape[d] ≤ in_page_shape[d]`, since `32a + H_out ≤ H ⇒ a + ceil(H_out/32) ≤ ceil(H/32)`); of the ROW_MAJOR map, `W_out == W`. When the block ends inside a tile, the whole edge tile is copied and the input rows/columns past the end become output padding, which nothing reads (spec GUIDANCE, R4).

**Honest fold.** Output pages are dense and contiguous in output page order, and every page is treated identically (one whole-page relay); the axes differ only in the source-address arithmetic, which is the page map above, not a different treatment. So the op's block space folds honestly to **one** axis, `out_page`, in row-major output page order, and a block is a 1-D range of it. A block that straddles an outer-index or tile-row boundary is legal: the page map walks through the boundary.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `out_page` (the fold of the four below, output row-major page order) | independent — each output page is one input page, no combine | `block_pages` | `max(1, min(ceil(max_core_pages / RING_DEPTH_BLOCKS), block_pages_cap))`, rounded up to a multiple of `max(R, W)` — worked values below | host: computed once in `create_program_artifacts` from `RING_DEPTH_BLOCKS`, the split's `max_core_pages` and `ring_cap_bytes`; only consumer is the ring's `num_entries` | contiguous ranges of `out_page` over the full grid, `ttnn.split_work_to_cores(grid, num_out_pages, row_wise=True)` | knob-turn |
| `a0`, `a1` (leading dims: N for rank 3; N, C for rank 4; absent for rank 2) | independent — a slice along them selects planes, no cross-plane dependency | folded: `block_pages` (no separate knob — fold is honest, above) | folded | page map RT args (`out_page_shape[0..1]`, `page_start[0..1]`) | through `out_page` ranges | knob-turn (rank 2/4 is a validate change; the page map is 4-D from day 1) |
| `h` (rows in RM, tile-rows in TILE) | independent — rows/tile-rows are copied independently | folded: `block_pages` | folded | page map RT args (`out_page_shape[2]`, `page_start[2]`) | through `out_page` ranges | knob-turn |
| `w` (inside the page in RM — one page spans W; tile-columns in TILE) | independent — columns of whole tiles are copied independently; in RM the whole row is one page | folded: `block_pages` (RM: extent 1 page by construction — a row is the NoC unit of an interleaved RM buffer, not a collapsed knob) | folded | page map RT args (`out_page_shape[3]`, `page_start[3]`) | through `out_page` ranges | TILE `w` with tile-aligned start: knob-turn (already in the page map). RM `w`: scheme-change (`rm_column_cut`) |

There is no dependent axis and no reuse-shared operand: one input, and every input page feeds at most one output page on one core.

**Operand-reuse check** (chosen split: contiguous `out_page` ranges): the only operand is `input_tensor`, and it *does* vary along the split axis (each core's source pages are disjoint) → no operand is reuse-shared by construction of the split; no broadcast row is needed.

**Stall-shadow check.** The only stage that waits on something it does not produce is the writer (waits for the reader's block to land) and the reader (waits for the writer to free ring space). The independent work that fills each stall is the other half of the ring: with `RING_DEPTH_BLOCKS = 2` the reader fills block k+1 while the writer drains block k. Nothing else exists to schedule (no compute, no algebra to reorder); no floating-point order is involved.

**Phase 0 worked values** (Quasar, 8 × 4 grid, unreserved budget 3,963,328 B, `ring_cap_bytes = min(1,048,560, 3,963,328) = 1,048,560`):

| Call | Layout / buffer | `entry_bytes` | `num_out_pages` | cores engaged × pages | `block_pages` | ring bytes |
|------|-----------------|---------------|-----------------|------------------------|---------------|------------|
| `[1,144,8400]` rows 0–64 (YOLO) | RM / DRAM | 16,832 (16,800 → 64-B aligned) | 64 | 32 × 2 | 1 | 33,664 |
| `[1,144,8400]` rows 64–144 (YOLO) | RM / DRAM | 16,832 | 80 | 16 × 3 + 16 × 2 | 2 | 67,328 |
| `[1,144,400]` rows 0–64 (YOLO) | TILE / L1 | 2,048 | 2 × 13 = 26 | 26 × 1 | 1 | 4,096 |
| `[1,144,400]` rows 64–144 (YOLO) | TILE / L1 | 2,048 | 3 × 13 = 39 | 7 × 2 + 25 × 1 | 1 | 4,096 |
| `[2,512,1024]` rows 256–512 | TILE / DRAM | 2,048 | 2 × 8 × 32 = 512 | 32 × 16 | 8 | 32,768 |
| `[4,128,47]` rows 64–100 | RM / DRAM | 128 (94 → 64-B aligned) | 4 × 36 = 144 | 16 × 5 + 16 × 4 | 3 | 768 |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_slice_pages` | `RING_DEPTH_BLOCKS` (host constant) | 2 | Read/write overlap: the reader fills one block while the writer drains the other. With an even `num_entries` the runtime assigns two transaction ids and the ISR credits one half-ring (= one block) per batch (`DataflowBuffer.md` C4, "first eligible n ≥ 2") |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Implicit-sync ring transfer = one NoC transaction of exactly `entry_size` bytes per call (`tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:775-812`, `821-870`) | `entry_bytes` (one page) | `entry_bytes = input.buffer_aligned_page_size()` (== output's; host asserts) — the whole aligned page slot moves, so source and destination page starts are both aligned and the alignment pad of an RM row is moved with it | A transfer of the unaligned `buffer_page_size()` is not expressible on this path (no size argument); a smaller entry would truncate rows |
| NoC burst: `NOC_MAX_BURST_SIZE = 256 words × 256 B = 65,536 B` on Quasar (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:394-396`) | `entry_bytes` | validate refuses `entry_bytes > NOC_MAX_BURST_BYTES` (host constant 65,536, cited) with `NotImplementedError` — RM bf16 rows wider than 32,768 elements; no INPUTS entry comes near (max 16,800 B) | The ISR threshold is counted in transactions (`DataflowBuffer.md` lines 420-425) |
| Interleaved page = one bank: consecutive page ids live in different banks | the transfer quantum | one NoC transaction per page; no coalescing across pages (catalog `read_coalescing`: "invalid across interleaved pages") | Reads span the wrong bank → wrong data |
| DFB ring size ≤ 1,048,560 B (target page cluster model) and the L1 budget | `block_pages` | `block_pages_cap = floor(ring_cap_bytes / (RING_DEPTH_BLOCKS × entry_bytes))`, `ring_cap_bytes = min(DFB_RING_MAX_BYTES, ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES)`; validate refuses when `block_pages_cap < 1` (needs `entry_bytes > 524,280`, already excluded by the burst cap) | Ring creation fails (`dfb.cpp:85-93`) |
| Implicit-sync txn batching on a DM→DM STRIDED ring: `num_entries % (num_txn_ids × max(R, W)) == 0` (`DataflowBuffer.md` lines 430-436) | `block_pages` | `block_pages` rounded up to a multiple of `max(R, W)`; `num_entries = RING_DEPTH_BLOCKS × block_pages` is then even and divides | Host rejects the DFB, or falls back to one txn id (no overlap) |
| Implicit-sync entry ≤ one NoC overlay packet, 8 KiB (`NOC_OVERLAY_MAX_BYTES_IN_PACKET`, `tt_metal/hw/inc/internal/tt-2xx/quasar/noc_cmd_buf_common.h`) — *added by the implementer, Phase 0; recorded here by the verifier* | `entry_bytes` on the implicit-sync protocol | `IMPLICIT_SYNC_MAX_ENTRY_BYTES = 8 KiB`: larger pages (RM bf16 rows wider than 4,096) take the ring's **explicit** protocol — the ring is listed in `disable_dfb_implicit_sync_for`, each hart reserves/waits a block of `thread_block_pages = block_pages / max(R, W)` entries, tags every transfer with user trid 0 and barriers once per block; needs `R == W` | — |
| Ragged per-core tail: implicit-sync partial batches are credited by DM0 at kernel exit (`DataflowBuffer.md` C6, lines 571-575) | `last_block_extent` | none needed — a core's page count need not divide `block_pages` | — |

**Verifier note (Phase 0): the depth knob under implicit sync.** The runtime gives an implicit-sync ring the smallest txn-id count `n ∈ [2, 4]` that divides `num_entries / max(R, W)` (`DataflowBuffer.md`, "How many txn ids"), and the ISR credits `num_entries / n` entries per batch. With `num_entries = RING_DEPTH_BLOCKS × block_pages` and an even ring, `n = 2` whatever `RING_DEPTH_BLOCKS` is: the credited block is always **half the ring**. So on the implicit path `block_pages` is the only effective granularity knob, and turning `RING_DEPTH_BLOCKS` to 3–4 does not give 3–4 in-flight blocks unless the ring is sized so that `n` comes out equal to it. On the explicit path the kernels own the batching and the depth knob works as written.

### Regimes

Named memory boundary for every row: **the input tensor's buffer tier** (DRAM banks, or L1 banks for an L1-interleaved input — the output lives in the same tier, R7). Minimum = every output page's source crosses that tier once into a core, and every output page crosses it once out.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `page_relay` — reader relays whole source pages through one DM→DM ring to the writer, page map on the reader, contiguous output pages on the writer | **built** (Phase 0) | `(layout == TILE and h_start % 32 == 0 and w_start % 32 == 0) or (layout == ROW_MAJOR and w_start == 0 and w_end == W)`, and `entry_bytes ≤ NOC_MAX_BURST_BYTES`. Phase 0's SUPPORTED gates it further to rank 3, bf16, `outer`/`h`, `tile_aligned` start | `block_pages` consecutive `out_page`s (1-D fold), `RING_DEPTH_BLOCKS` blocks in the ring | **Minimum**, plus two structural surpluses: (1) RM rows move their alignment pad (`entry_bytes - W·elem`, < 64 B DRAM / < 16 B L1 per row); (2) TILE edge tiles move whole when the block ends inside a tile (the output tile must be written whole anyway). Pages outside the block are never read | Per block: one ISR post on the producer side and one ISR ack on the consumer side (intended frequency: once per block per side), and the pipeline fill/drain of the two-block ring. No inits, no reconfigs, no barriers in the kernels (firmware drains at exit) |
| `rm_column_cut` — RM with W narrowed: each output row is the byte range `[w_start·e, w_end·e)` of its input row | **built** (Refinement 1). Same page map (W page axis extent 1), same full-grid `out_page` split, same ring, on its **explicit** protocol (opted out via `disable_dfb_implicit_sync_for`, every transfer `TXN_ID {.trid = 0}`, block-batched reserve/barrier/push and wait/barrier/pop, one barrier per block per hart — the Phase 0 large-page path, reused). Ring entry = the output's aligned row slot. The reader moves exactly `transfer_bytes = W_out·e` from byte `src_offset_bytes = w_start·e` of its source row (TensorAccessor page read with `offset_bytes`); the writer moves the same `transfer_bytes` from the entry start to the output row. **No aligned superset and no writer realign**: the Quasar NoC shifts data between differently aligned source and destination (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:482-491`, `NOC_{L1,DRAM}_{READ,WRITE}_ALIGNMENT_BYTES = 1`), measured bit-exact on ttsim at byte offsets 6, 14, 34 B and row lengths 64, 66, 2,000, 16,794 B in DRAM and L1 (`test_slice_plan.py::test_rm_column_cut_any_byte_offset`). Silicon should confirm the shift (a 16-B-aligned superset read plus a writer-hart realign is the fallback this row first planned) | `layout == ROW_MAJOR and (w_start, w_end) != (0, W)` (`regime_for` in `slice_program_artifacts.py`) | `block_pages` rows | **Minimum**: `W_out·e` bytes read and written per row (the alignment pad of the output slot is not even written) | Same as `page_relay`'s explicit path: one reserve/push and one barrier per block per hart |
| `tile_retile` — TILE with an H or W start inside a tile: each output tile gathers rows/columns from 2 (one axis) or 4 (both) source tiles with an intra-tile shift across faces | **deferred** — positive reason: PHASE0 has `start_alignment = tile_aligned` only, and spec GUIDANCE names it "another kernel". Reachable: same split over `out_page`, same writer; the reader reads the 2–4 source tiles of an output tile (the page map gives the top-left one), and the shift is done either by the writer hart on L1 bytes (face-aware row/col copy) or by a compute stage (untilize→shift→tilize) — which one is a later measurement; the second makes the op compute-bound and engages the NEOs. Scheme-change | `layout == TILE and (h_start % 32 != 0 or w_start % 32 != 0)` | `block_pages` output tiles (each with its 2–4 source tiles resident) | Each source tile is read by up to 2 (or 4) output tiles: up to 2× (4×) the minimum reads, unless a core keeps the shared source tile-row resident across its block — a residency knob on `block_pages` | Fewer re-reads of shared source tiles (overlapping footprints, `blocking-model.md` §4) |
| `rm_unaligned_h_start` — RM with H start inside a tile | covered by `page_relay` (a row is the page; nothing changes) — not a separate algorithm; listed so the refinement is seen as a validate-only knob-turn | `layout == ROW_MAJOR and h_start % 32 != 0` | — | minimum | — |
| `alias_input` — return the input itself for a whole-tensor block (tt-metal's behaviour) | **rejected** — superseded by `page_relay`, which copies at the minimum; returning without a dispatch violates the task's "exactly one native dispatch per invocation" rule | all `start == 0`, `end == size` | — | zero traffic | — |
| `single_hart_copy` — one DM kernel reads a block into L1 and writes it out (no ring, barrier between) | **rejected** — superseded by `page_relay`: identical bytes, but read and write serialize on one hart; no good scheme passes through it (dead end) | — | — | minimum, no overlap | — |
| `coalesced_run_read` — one NoC transaction per run of consecutive source pages | **rejected** — impossible on an interleaved buffer: consecutive pages are in different banks (catalog `read_coalescing`). Its home is a sharded source, which is not in TARGET (`sharding: ["interleaved"]`) | — | — | — | — |
| `split_dependent_and_combine` | **n/a** — the op has no dependent axis (no output element depends on more than one input element) | — | — | — | — |
| `broadcast_shared_operand` | **n/a** — no operand is read by more than one core (operand-reuse check above) | — | — | — | — |

Two regimes are built (Refinement 1): `regime_for(input, start, out_shape)` is the selector — `rm_column_cut` when the predicate above holds, `page_relay` otherwise — and `uses_explicit_sync(entry_bytes, regime)` puts `rm_column_cut` (and pages over 8 KiB) on the explicit protocol. Regime-pinned tests: `test_slice_plan.py::test_w_narrowed_regime` (TILE-`w` → `page_relay`, RM-`w` → `rm_column_cut`, DRAM and L1) and `test_rm_column_cut_between_tt_metal_programs` (session order). When `tile_retile` is built, its predicate joins `regime_for`.

### Traffic ranking

| Candidate split | Bytes crossing the input's tier | Cross-core traffic | Fills the 32-core grid for every INPUTS shape? |
|-----------------|-------------------------------|--------------------|-----------------------------------------------|
| **Contiguous `out_page` ranges over the grid (chosen)** | read `num_out_pages × entry_bytes` once, write the same once (minimum) | none | yes, down to one page per core (26 of 32 cores at the smallest YOLO call) |
| By leading-dim plane (`a0`/`a1`) | same | none | no — YOLO has N = 1: one core |
| By `h` page rows | same | none | partly — fails for few rows (e.g. `[1,64,8192]` TILE rows 0–32: 1 tile-row) |
| By `w` tile-columns (TILE only) | same | none | no for narrow W (Wt = 2) and meaningless in RM (one page per row) |
| Split a dependent axis + combine | n/a (no dependent axis) | — | — |

Every split moves the same bytes (no operand is shared, no footprint overlaps), so the traffic ranking ties; the tiebreaker is occupancy and transfer shape: the page-granular fold is the only split that fills the grid on every shape, and it keeps each core's writes a contiguous range of output pages, which the interleaved buffer spreads round-robin over the banks. `row_wise=True` follows catalog `noc_placement` (row placement over the DRAM-facing axis).

### Block schedule

Logical schedule per core (reader and writer are separate asynchronous kernels on two DM harts; the ring pipelines them):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // num_blocks_this_core = ceil(num_pages_this_core / block_pages)
    load_block(block_idx);    // reader: the block's block_pages source pages → cb_slice_pages
    store_block(block_idx);   // writer: cb_slice_pages → the block's block_pages consecutive output pages
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_pages` consecutive output pages (`last_block_extent` on the ragged last block), source pages via the page map | the other block in the ring (being drained) | one ISR post per block (the ring's txn batch = half the ring); no barrier, no reserve/push (implicit sync); one tag hand-back (`noc_async_read_set_trid(0, noc.get_noc_id())`) per kernel, at the end |
| `store_block` | the same `block_pages` pages, output page ids `first_out_page + i` | the other block (being filled) | one ISR ack per block; no barrier, no wait/pop; one tag hand-back (`noc_async_write_set_trid(0, noc.get_noc_id())`) per kernel, at the end |

**Realization contract.** On this ring the block boundary is enforced by the ring itself: with `num_entries = RING_DEPTH_BLOCKS × block_pages` and two txn ids, the DM0 ISR credits exactly one block per batch, so `block_pages` lives only on the host (the ring's size) and the kernels never see it. Inside each block operation the kernel issues one tagged NoC call per page — that is the mechanism (one interleaved page = one NoC transaction, cap above), not a per-unit handshake: no credit, barrier or init is paid per page. The reader advances the page-map multi-index as an odometer (increment the innermost page index with carry, `R` steps per thread iteration), not a divide/mod per page. Ragged tails need nothing: DM0 credits the partial last batch at kernel exit.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (block size) | `block_pages = ceil(max_core_pages / 2)` makes each block half the core's work: on large per-core shares (16 pages at `[2,512,1024]`) the writer waits for a whole 8-page batch before its first write | Fixed 4–8-page blocks with `RING_DEPTH_BLOCKS` 3–4 (catalog `double_buffer`: 4–8 in flight is the sweet spot; `trid_double_issue`: depth 3–4) — host-only change |
| Grid synchronization | Small outputs (26–80 pages) engage 26–32 clusters with 1–3 pages each; launch and per-cluster fixed cost may exceed the transfer time | Cap engaged cores so each gets ≥ k pages (`MIN_PAGES_PER_CORE` knob on the split; Phase 0 = 1) |
| Thread plan for small RM pages | One reader hart may be issue-bound on 128-B entries (`[4,128,47]`, `[32,17]`): per-transaction cost, not bandwidth, dominates | `DM_THREAD_PLAN` (2, 2) — the kernels already stride by R and W (catalog `split_reader`); host constant change plus the ring-divisibility rounding above |
| Placement | `row_wise=True` is the catalog's Wormhole result; Quasar's 8 × 4 grid with 2 DRAM channels is unmeasured | `row_wise=False` |

On this backend (ttsim, untimed) each lamp is judged by a structural count (pages per block, cores engaged, DM harts at work), not by time; a timed backend settles the ranking.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input buffer → ring | the tensor's page (RM row of `entry_bytes`, or a 2,048-B bf16 tile), raw bytes | reader DM hart: `noc.async_read<NocOptions::TXN_ID>(input_accessor, cb_slice_pages, {.page_id = src_page}, {})` per page | Implicit sync (ring default): no reserve/push/barrier; `src_page` from the page map |
| Ring → output buffer | same bytes | writer DM hart: `noc.async_write<NocOptions::TXN_ID>(cb_slice_pages, output_accessor, {}, {.page_id = first_out_page + i})` per page | Implicit sync: no wait/pop/barrier; firmware waits for the writes to land at kernel exit |
| Compute | — | none | Bytes never pass through a NEO, DEST or another dtype (R1) |

**Page-map contract** (host computes, reader consumes; one set of values for all cores): `out_page_shape[4]`, `page_start[4]`, `in_page_stride[4]`; per core `first_out_page`, `num_pages_this_core`. Writer per core: `first_out_page`, `num_pages_this_core`. Plan counts `reader_threads` (R) and `writer_threads` (W) are compile-time arguments of the kernel that uses them. Everything shape-dependent is a runtime argument, so a new shape does not recompile kernels.

**TARGET axes against this scheme** (each `TARGET − PHASE0` value, classified by this op's axis characters):

| Axis value outside PHASE0 | Class | What changes |
|---------------------------|-------|--------------|
| dtype `float32` | knob-turn | `entry_bytes` doubles (4,096-B tiles; 4-B elements per row); validate only |
| dtype `uint32` | knob-turn | as float32, plus the ring's `data_format_metadata` set to `ttnn.int32` (same width) or left unset — uint32 is refused as a buffer format (Unavailable row) |
| rank 2, rank 4 | knob-turn | validate only: the page map is 4-D, left-padded with 1s (rank 4 landed in Refinement 10, probe_007: 60/60 rank-4 cases exact with no kernel or host-program change) |
| `sliced_region = w`, TILE, tile-aligned start | knob-turn | validate only: the page map already carries `page_start[3]`, `out_page_shape[3] = ceil(W_out/32)` (landed, Refinement 1) |
| `start_alignment = non_tile_aligned`, ROW_MAJOR, W whole | knob-turn | validate only (`rm_unaligned_h_start` row) |
| `sliced_region = w`, ROW_MAJOR | scheme-change | `rm_column_cut` (built, Refinement 1) |
| `start_alignment = non_tile_aligned`, TILE | scheme-change | `tile_retile` (deferred row) |
| sharded placements | not in TARGET (`sharding: ["interleaved"]`) | refused by name through `memory_axes` (R8). A future sharded source would be consumed in place through a ring backed on the shard, never re-read over the NoC |

No deferred regime needs inter-cluster communication (no mcast, no semaphore): both are per-core changes to the reader/writer and the ring protocol.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output page (`out_page`); a core's share is cut into blocks of `block_pages` |
| Grid | `device.compute_with_storage_grid_size()` (8 × 4 here); `ttnn.split_work_to_cores(grid, num_out_pages, row_wise=True)` → core groups 1 and 2 with `per_core_1 ≥ per_core_2` pages; `max_core_pages = per_core_1` |
| Per-core work | contiguous output pages `[first_out_page, first_out_page + num_pages_this_core)`, `first_out_page` accumulated over the cores in the split's row-major order (as in `toy_thread_plan_program_artifacts.split_tiles`) |
| Remainder | handled by the split's two groups; `num_out_pages = prod(out_page_shape)` with `ceil` per image for TILE (`ceil(H_out/32)`, `ceil(W_out/32)`), asserted equal to `output.buffer_num_pages()`; cores with 0 pages are not in the work unit's core set |
| Bound | **data movement** — the op computes nothing: zero compute per page against one page read and one page written |
| Thread plan | `DM_THREAD_PLAN = (R, W) = (1, 1)` with **no compute kernel** (C = 0: the four NEOs stay idle because there is no work for them). One named host constant; R and W passed as CT args; the reader strides its pages by R and the writer by W from `get_my_thread_id()` (`tt_metal/hw/inc/api/kernel_thread_globals.h:155`) |
| Unit dealt to compute threads | none (no compute kernel). Unit dealt to DM threads: the ring entry itself (one page), round-robin — any count stops cleanly, no padding |
| Tests the plan needs | (1) **every thread ran**: the op called twice in one session on two different inputs of the same shape into fresh outputs; the second result must be exact (a missing reader/writer thread leaves pages stale or unwritten) — repeat with the plan overridden to (2, 2) through a test-only keyword so the strided dealing is exercised; (2) **session order**: in one device session, a tt-metal program (`ttnn.to_layout`) then `slice`, and `slice` then `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` on a TILE output, both compared to the reference |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|----------|
| `cb_slice_pages` (DFB `unique_id="slice_pages"`) | — (DFBs are named) | `entry_bytes = input.buffer_aligned_page_size()` | `RING_DEPTH_BLOCKS × block_pages` | spans `out_page` by `block_pages × RING_DEPTH_BLOCKS`; streams over `a0`, `a1`, `h`, `w` (folded into `out_page`) | input dtype as metadata (bf16 in Phase 0; raw bytes — never unpacked) | reader | writer | implicit both sides (STRIDED → STRIDED, DM → DM) | whole program |

This is the whole inventory: one buffer, the minimum for a relay (a core can only read into its own L1 and write out of it, so a staging buffer is unavoidable; a second buffer would hold the same bytes). Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_pages` output pages (`last_block_extent` on the last) | no compute helper applies (data movement); DFB implicit-sync NoC overload | input tensor via `TensorAccessor` | `cb_slice_pages`, `block_pages` entries posted by the ISR | one block posted to the writer |
| 2 | `store_block` | the same `block_pages` pages | DFB implicit-sync NoC overload | `cb_slice_pages`, `block_pages` entries | output tensor via `TensorAccessor` | one block acked back to the reader |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(src, DataflowBuffer&, src_args, {})` (implicit-sync overload) | decl `tt_metal/hw/inc/api/dataflow/noc.h:923-927`; def `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:773-812` | `src = TensorAccessor(tensor::input)`, `{.page_id = src_page}` | input tensor | `cb_slice_pages` | none in the call; `block_pages` is the ring's size on the host |
| `store_block` | raw_api | `Noc::async_write<NocOptions::TXN_ID>(DataflowBuffer&, dst, {}, dst_args)` | decl `noc.h:936-941`; def `dataflow_buffer.inl:819-870` | `dst = TensorAccessor(tensor::output)`, `{.page_id = first_out_page + i}` | `cb_slice_pages` | output tensor | same |
| end of each DM kernel | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())` (reader), `noc_async_write_set_trid(0, noc.get_noc_id())` (writer) | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558`, `:2579` | — | — | — | — |
| thread dealing | raw_api | `get_my_thread_id()` | `tt_metal/hw/inc/api/kernel_thread_globals.h:155` | — | — | — | R, W (CT args) |
| host: split | host | `ttnn.split_work_to_cores(grid, num_out_pages, row_wise=True)` | `ttnn/ttnn/core.py:19` | — | — | — | core-assignment |
| host: program | host | `ttnn.experimental.ProgramSpec`, `DataflowBufferSpec(unique_id, entry_size, num_entries, data_format_metadata)`, `KernelSpec(num_threads=…, dfb_bindings=[producer_of / consumer_of], tensor_bindings, compile_time_args, runtime_arg_schema)`, `create_reader_datamovement_config()` / `create_writer_datamovement_config()`, `WorkUnitSpec(kernels=["reader","writer"], target_nodes=core_set)`, `ttnn.generic_op([input, output], spec, run_args, tensor_indices)` | pattern: `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py:72-158`; DM-only ring: `ttnn/ttnn/operations/toy_spec_single_tensor/toy_spec_single_tensor.py:91-131`; bindings `ttnn/cpp/ttnn-nanobind/program_spec.cpp:470-477` | `num_entries = RING_DEPTH_BLOCKS × block_pages` | — | — | `block_pages`, `RING_DEPTH_BLOCKS`, `DM_THREAD_PLAN` |

**Helpers considered and rejected** (for every `raw_api` row; there is no compute phase):

| Candidate | Why it cannot express this op | Citation |
|-----------|-------------------------------|----------|
| `copy` / `eltwise_chain(CopyTile, PackTile)` | A copy through a NEO unpacks to DEST and packs back: it adds a compute kernel and two rings to a byte relay, moves no fewer bytes, and passes every element through DEST — a 16-bit DEST would round float32 and corrupt uint32 (R1). Also tile-only: it cannot carry an RM row | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:109-113` (`copy` = `CopyTile(D0) -> PackTile(D0)`: unpack to DEST, pack back, tile operands); spec R1 |
| `tilize` / `untilize` helpers | The layout does not change | spec `OUTPUT["layout"] = same_as input` |
| `dataflow_kernel_lib::McastArgs` / `ttnn.Mcast` | No operand is read by more than one core (operand-reuse check) | — |
| A dataflow helper for interleaved page relays | None exists in `ttnn/cpp/ttnn/kernel_lib/` for DFB implicit-sync page copies; the implicit-sync NoC overload is itself the one-call-per-page primitive the toy examples use | `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader_implicit.cpp:26-29` |

## Structural impossibilities

None to add: TARGET has no block-float dtype, and every (dtype, layout, placement) combination in TARGET is buildable on this part.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Tag register | Both kernels use implicit sync; the hart's NoC tag register is sticky | Each kernel's last statement hands the tag back (reader `noc_async_read_set_trid(0, …)`, writer `noc_async_write_set_trid(0, …)`); the session-order test (Work Distribution) proves it |
| RM row moved short or misplaced | `buffer_page_size()` (e.g. 94 B) differs from the bank slot `buffer_aligned_page_size()` (128 B DRAM / 96 B L1); the implicit-sync transfer always moves `entry_size` | `entry_size = buffer_aligned_page_size()` and the host asserts input and output aligned page sizes are equal (true in the built regime: same W, dtype and buffer type) |
| TILE page count off for non-aligned shapes | Tile padding is per image | `ceil` per image in the page map; `prod(out_page_shape) == output.buffer_num_pages()` asserted on the host |
| Ring rejected or overlap lost | Implicit-sync batching needs `num_entries % (num_txn_ids × max(R, W)) == 0` | `num_entries = 2 × block_pages`, `block_pages` a multiple of `max(R, W)` |
| Builtin shadowing | The entry point is named `slice` | The module never uses the builtin `slice` |
| Ring base alignment vs DRAM reads of 128-B RM entries | Tile entries (2,048 B) are proven by the toy ops; 64-B-multiple RM entries into a ring are not yet seen on this backend | If RM/DRAM cells fail while TILE passes, check the ring base alignment first; the acceptance test covers RM in DRAM with `W = 47`-class rows through the golden suite |
| Whole-tensor block | Tempting to return the input (tt-metal does) | Rejected regime: always one dispatch of `page_relay` |
| Negative indices / list bounds | R2, R3 | Normalize before any support check; accept `list` and `tuple` |

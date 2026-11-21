# Operation Design: upsample

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel: every output element is an input element, unchanged — spec DEFINITION, rule R1) |
| Goal | Nearest-neighbour enlargement of a channels-last image `[N, H, W, C]` by whole factors `(sh, sw)`: one `ttnn.generic_op` dispatch whose reader reads every input stick once and whose writer writes it `sh·sw` times. |
| Math | `output[n, i, j, c] = input[n, i // sh, j // sw, c]`; output shape `[N, H·sh, W·sw, C]`, input's dtype, layout and buffer |
| References | Specification `eval/specs/upsample/spec.py` (authoritative); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/` (host-program pattern, thread plan, ring protocols); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` (Part A, C2); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; perf catalog `ttnn/ttnn/operations/examples/master.md` (entries `double_buffer`, `noc_placement`, `tile_reorder`, `trid_double_issue`); structural precedent for a reader→ring→writer `ProgramSpec` with no compute kernel: tt-metal's Quasar row-major slice factory `ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/slice_program_factory_rm.cpp:255-325` (a different op) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (probe on this backend: `compute_with_storage_grid_size()` = 8-4); queried at run time with `device.compute_with_storage_grid_size()` | target page; probe 2026-11-21 |
| L1 per core | 4,194,304 B (4 MiB) per cluster, shared by its NEOs and DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` = 3,963,328 − 0 = 3,963,328 B on this target (query value read by probe on ttsim, 2026-11-21) | probe; target page |
| Alignment | L1 16 B (`ttnn.get_l1_alignment()`), DRAM 64 B (`ttnn.get_dram_alignment()`) — both confirmed by probe | target page; probe |
| Host program | `ttnn.ProgramSpec` (`ttnn.experimental.ProgramSpec` / `KernelSpec` / `DataflowBufferSpec` / `WorkUnitSpec`, dispatched by one `ttnn.generic_op`) — reference `.claude/references/quasar_op_examples/` (README first; `toy_thread_plan/toy_thread_plan_program_artifacts.py` is the Python pattern) | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1, 2, 4}); 6 user data-movement harts (R + W ≤ 6); on STRIDED rings the smaller thread count divides the larger (`DataflowBuffer.md` C2: "when C ≥ P, C must be divisible by P"). DFB ring ≤ 1,048,560 B; 32 DFBs per cluster; DM kernel binary ≤ 48 KiB | target page *Cluster model*, *Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal-1 kernels and the legacy C++ program factory (refused at kernel creation, `tt_metal/impl/kernels/kernel.hpp`); `bfloat8_b` / `bfloat4_b` (no path on this part: refused in `validate()`, no refinement queued); untagged NoC transfers: not used (every transfer here carries `trid 0`); uint16/uint32 are not ring formats (irrelevant: ring metadata is the tensor dtype, bfloat16 / float32); self-looped DM ring (DM kernel bound as producer and consumer of one DFB) is refused on Gen2 (`tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp`, "Self-loop DFBs are not supported for data-movement kernels") — so reader and writer are two kernels | target page |

Every worked number below is for **quasar, 32 clusters, 3,963,328 B unreserved L1**. Every run-time
predicate uses the query, never the value.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 4 `[N, H, W, C]`, on device | — | tensor binding `input` |
| `scale_factor` | `int` or pair `(sh, sw)` | yes | whole numbers ≥ 1 (a `bool` is not a whole number here); normalized to `(scale_h, scale_w)` before anything else (rule R2) | — | `scale_h`, `scale_w`: CT args of the writer |
| `mode` | `str` | no | `"nearest"` | `"nearest"` | host only (gate) |

### Entry point and validation (host, before any allocation)

| Step | Rule | Raises |
|------|------|--------|
| 1. Rank | `len(input_tensor.shape) != 4` | `ValueError` (X2) |
| 2. Factor | `scale_factor` is not an `int ≥ 1` (bool excluded) nor a 2-sequence of them → malformed; else `(scale_h, scale_w) = (s, s)` or `tuple(s)` | `ValueError` (X1) |
| 3. Mode | `mode != "nearest"` | `ValueError` (X3) |
| 4. Axes | `dtype`, `layout`, `mode`; `memory_config` and `sharding` from `eval.memory.memory_axes(input_tensor)` (R6); `channel_alignment`, `scale` from `INPUT_TAGGERS = AXES` (imported from the spec, never redefined) called with `((tuple(shape), scale_factor), axes)` | — |
| 5. Support | any axis value outside `SUPPORTED` (= `PHASE0`) → refusal by name | `NotImplementedError` (R4) |
| 6. Regime | the regime predicate of the Blocking Model (`stick_stream` needs `RING_ENTRIES × stick_stride ≤ ring_bytes_cap`); a shape failing it is the deferred `channel_chunk` regime | `NotImplementedError` |

`scale_factor` form is never gated (R2): the int `2` and the pair `(2, 2)` normalize to the same
`(2, 2)` and land in the same cell.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[N, H, W, C]`, rank 4; Phase 0: `C % 32 == 0` (`channel_alignment = tile_aligned`) |
| Dtype | Phase 0: `bfloat16` (TARGET adds `float32`; `bfloat8_b` unbuildable on Quasar) |
| Layout | Phase 0: `ROW_MAJOR_LAYOUT` — one page = one stick = the `C` values of one position `[n, h, w]`, `stick_bytes = C · element_size` |
| Memory | interleaved, DRAM or L1 (R3); sharded refused by name (R6) |

### Output

| Property | Value |
|----------|-------|
| Shape | `[N, H·scale_h, W·scale_w, C]` |
| Dtype | input's |
| Layout | input's (ROW_MAJOR in Phase 0) |
| Memory | input's buffer type, interleaved (R5): `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(out_shape, input.dtype, input.layout, input.memory_config()), device)` (`ttnn/cpp/ttnn-nanobind/tensor.cpp:290-305`). Allocation is not a dispatch. |

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

The op's data has six axes: the input positions `batch` (N), `in_row` (H), `in_col` (W); the
channel axis `channel` (C), which is the inside of one page; and the two replication axes of the
output, `rep_h` (sh) and `rep_w` (sw). `batch`, `in_row`, `in_col` are treated identically by the op
and are contiguous in row-major page order, so they **fold into one flattened axis `stick`** with
index `s = (n·H + h)·W + w` (`blocking-model.md` §1, "leading axes fold into one when the op treats
them identically and they are contiguous"). Page `s` of the input is that stick.

Output page of input stick `s` and replica `(a, b)`: with `q = s / W` (the global input row
`n·H + h`) and `w = s % W`,
`out_page(s, a, b) = (q·scale_h + a) · (W·scale_w) + w·scale_w + b`, `0 ≤ a < scale_h`, `0 ≤ b < scale_w`.
(Batch folds in because `n·H·scale_h + h·scale_h = q·scale_h`.)

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `stick` = `batch`×`in_row`×`in_col` flattened (N·H·W input pages) | independent — each output element depends on exactly one input stick; no result spans sticks | `block_sticks` (sticks per block = sticks per ring entry) | `max(1, min(ceil(max_core_sticks / (WRITER_THREADS · BLOCKS_PER_WRITER_THREAD)), floor(ring_bytes_cap / (RING_ENTRIES · stick_stride))))` — e.g. 7 for `[1,40,40,512]`, 2 for `[1,20,20,512]` | host, computed once in the program-artifacts function from `BLOCKS_PER_WRITER_THREAD`, `THREAD_PLAN`, `RING_ENTRIES_PER_THREAD`, the caps; passed as one CT arg to reader and writer and used for the ring's `entry_size` | **split across the full grid**: contiguous ranges of input sticks per cluster, `ttnn.split_work_to_cores(grid, N·H·W, row_wise=True)`; inside a cluster, blocks dealt round-robin to reader threads (block `b` → reader `b mod R`) and to writer threads (block `b` → writer `b mod W_t`) | knob-turn (`block_sticks`, `BLOCKS_PER_WRITER_THREAD`, grid) |
| `channel` (C, inside one page) | independent — output channel `c` reads input channel `c` only | `channel_chunk` (bytes of one stick moved per transfer) | whole stick: `stick_bytes = C · elem_size` (one transfer per page) | host: `stick_bytes = input.buffer_page_size()`, `stick_stride = align_up(stick_bytes, max(l1_alignment, dram_alignment))`; CT args | not split across cores — splitting C fragments every page into several transfers for no traffic gain; whole-C blocks fill the grid for every INPUTS entry but `[1,4,4,8192]` (16 sticks → 16 clusters) | scheme-change: `channel_chunk` regime (deferred, see Regimes) |
| `rep_h` (scale_h output rows per input row) | independent — every replica is the same bytes | `rep_h_extent` | whole `scale_h` (every replica of a resident stick written from that one L1 copy) | CT arg `scale_h` (from the normalized factor) | not split across cores — a replica on another cluster would re-read the stick from DRAM (cost: one extra input crossing per split) | not an unlock: splitting it is rejected (`output_split` regime) |
| `rep_w` (scale_w output columns per input column) | independent — same as `rep_h` | `rep_w_extent` | whole `scale_w` | CT arg `scale_w` | not split across cores — same reason as `rep_h` | not an unlock (same) |

No axis is **dependent** (nothing spans an axis: no reduction, no scan) and no operand is
**reuse-shared** across clusters.

**Operand-reuse check** (chosen split: input sticks across clusters). The only operand is the input,
and it **varies** along the split axis: each input stick is read by exactly one cluster. Not
reuse-shared; no broadcast row. The `sh·sw`-fold reuse of a stick is **core-local** (one L1 copy feeds
all its replicas) and is what keeps the input at one DRAM crossing.

**Stall-shadow check.** Two stages wait: a writer thread waits for its block to land in the ring
(`wait_front`), and a reader thread waits for ring space (`reserve_back`) and for its block's reads
(barrier). Work independent of each wait: the other writer threads' blocks (the thread plan places
`W_t = 4` writers so one's barrier is shadowed by the others' writes) and the next blocks' reads
(ring depth 2 per thread). The op's math makes the order of the `block_extent·sh·sw` writes of a block
free (a pure permutation of bytes; no floating-point order exists to preserve), so no algebraic
reorder is needed and no precision gate applies.

**Intermediate data.** The scheme introduces one stage boundary, the ring holding resident input
sticks; its axes are the `stick` axis (block of `block_sticks`) and `channel` (whole). It is dealt
exactly as the input is. No core collects or coordinates; no stage computes on one core while others
wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_sticks` | `RING_ENTRIES_PER_THREAD` (entries per thread slice; `RING_ENTRIES = RING_ENTRIES_PER_THREAD · max(R, W_t)`) | 2 → `RING_ENTRIES = 8` | double buffering per writer thread: reader threads fill block `b + W_t` while writer thread `b mod W_t` writes block `b`; with `BLOCKS_PER_WRITER_THREAD = 2` a core's whole share is resident at once in every Phase 0 INPUTS entry, so readers never stall on `reserve_back` there |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DFB ring size on Quasar: ≤ 1,048,560 B (target page *Cluster model*; no Python query, host constant `RING_MAX_BYTES`) and ≤ the unreserved L1 budget | `block_sticks · stick_stride · RING_ENTRIES` | `ring_bytes_cap = min(RING_MAX_BYTES, ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES)`; `block_sticks ≤ floor(ring_bytes_cap / (RING_ENTRIES · stick_stride))` | program build fails or rings overlap L1 tensor banks |
| Page is the indivisible transfer unit of `stick_stream` (one stick = one interleaved page, its own bank address) | `block_sticks ≥ 1` → `RING_ENTRIES · stick_stride ≤ ring_bytes_cap` needed | predicate refused in `validate()` with `NotImplementedError` → deferred `channel_chunk`; holds for C ≤ 65,535 bf16 / 32,767 fp32 (all of TARGET×INPUTS: max stick 32 KiB fp32 → ring 256 KiB) | ring larger than the cap: build failure |
| Ring entry alignment: entries are L1 destinations of DRAM/L1 page reads | per-stick stride inside an entry | `stick_stride = align_up(stick_bytes, max(ttnn.get_l1_alignment(), ttnn.get_dram_alignment()))` (= `stick_bytes` in Phase 0: `C % 32 == 0` → `2C` is a multiple of 64) | misaligned NoC transfers: corrupted sticks once `channel_alignment = non_tile_aligned` is unlocked |
| STRIDED ring thread dealing (`DataflowBuffer.md` C2) | `RING_ENTRIES` and the plan | `W_t % R == 0`; `RING_ENTRIES % max(R, W_t) == 0` (8 % 4) | host validation error (`dfb.cpp`) |
| `reserve_back` / `wait_front` count is `uint16_t` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-246`) | entries per call | always 1 entry per call | n/a |
| NoC burst: a stick larger than `NOC_MAX_BURST_SIZE` | none at op level | the NoC API splits it itself (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc_nonblocking_api_v1.h:680`) | n/a |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `stick_stream` — input sticks split across clusters; per cluster, blocks of `block_sticks` sticks read once into the ring and each stick written `scale_h·scale_w` times from L1 | **built** (Phase 0) | `layout == ROW_MAJOR and RING_ENTRIES · stick_stride ≤ ring_bytes_cap` (host). Domain: every row-major interleaved cell of TARGET (every PHASE0 cell; float32, non-aligned C and per-axis factors are knob-turns of this regime) | `block_sticks` sticks × whole C × whole `(scale_h, scale_w)` replication | **= minimum** at the DRAM (or remote-L1) boundary: input crosses once (`N·H·W` page reads), output crosses once (`N·H·sh·W·sw` page writes); zero cross-core traffic. The transaction count is fixed by the layout (one NoC transfer per interleaved page) and does not depend on the block | per block, paid once: 1 `reserve_back` + 1 `push_back` + 1 tagged read barrier (reader), 1 `wait_front` + 1 `pop_front` + 1 tagged write barrier (writer). Intended frequency: once per `block_sticks` reads / `block_sticks·sh·sw` writes. Kernel-wide (once): accessor setup, trid reset at exit |
| `channel_chunk` — a stick cut into `channel_chunk`-byte pieces (`offset_bytes` into the page), the (stick, chunk) pair the unit dealt to clusters | **deferred** — positive reason: `stick_stream` serves every TARGET × INPUTS cell (largest ring 256 KiB < 1 MiB cap), and its occupancy gain is bounded to the one INPUTS entry with fewer input sticks than clusters (`[1,4,4,8192]`: 16 of 32 clusters, 1 MiB of output). Reachable: the reader/writer already address a page by `page_id` plus `offset_bytes`; the split becomes `N·H·W·ceil(C/chunk)` units with the same kernels | `stick_bytes ... > ring cap` (correctness), or `N·H·W < num_clusters` (occupancy) | `block_sticks` × `channel_chunk` × whole replication | same bytes as minimum; `ceil(stick_bytes/channel_chunk)`× more NoC transactions | same as `stick_stream` |
| `output_split` — split output sticks/rows across clusters; each cluster reads the input sticks its outputs need | **rejected** — superseded by `stick_stream`: every input stick yields exactly `sh·sw` writes, so the input-stick split is already write-balanced; the output split re-reads an input stick on every cluster that owns one of its replicas (up to `sh·sw` DRAM crossings of the input at cluster boundaries, `sh` for a whole-row split) | — | — | input crosses up to `sh·sw` times | — |
| `l1_row_replicate` — reader assembles the replicated output row in L1 (`sw` local copies per stick) so the writer writes contiguous output rows | **rejected** — superseded by `stick_stream`: an interleaved row-major page is one stick at its own bank address, so the output needs one NoC write per page either way; this adds `sw·` local L1 copies per stick and an output-row-sized ring for zero DRAM saving | — | — | adds core-local traffic `(sw−1)·input bytes` | — |
| `tile_layout` — TILE-layout cells (`[N, H, W, C]` tiles over `(W, C)` per `(n, h)`): H replication writes whole output tile-rows `sh` times; W replication repacks 32-element tile rows (face rows) of each tile into `sw`-dilated output tiles | **deferred** — positive reason: `layout = TILE_LAYOUT` is outside PHASE0 (no caller in the trace); it is a second kernel pair, not a change of this one (spec DECISIONS 2026-11-21). Reachable: the split (contiguous ranges of input tile-rows `(n, h, w-tile)` across clusters) and the ring structure are the same; the W-repack is either a DM face-row shuffle into an output-tile ring or `untilize` → `stick_stream` → `tilize` helpers in one compute kernel. Tile geometry is per image and `ceil`: `tensor_c_tiles = ceil(C/32)`, `tensor_w_tiles = ceil(W/32)` per `(n, h)` | `layout == TILE_LAYOUT` | tile-row block of `block_tiles` × `scale_h` | minimum (input once, output once) for the H replication; W repack core-local | per-block handshakes as `stick_stream`, plus tilize/untilize inits once per block |
| `sharded_in_place` — a height-sharded L1 input consumed through a ring borrowed from its shard (`borrowed_from`), output shards written locally | **deferred** — positive reason: TARGET's `sharding` axis is `["interleaved"]` only and the signature has no `memory_config` argument (spec DECISIONS 2026-11-21); a sharded input is refused by name (R6). Reachable: per-cluster work is already a contiguous stick range; a height shard is exactly such a range, so the reader becomes a no-op on a borrowed ring | `sharding != interleaved` | the whole shard (one block) | zero input NoC traffic | one block per shard |

Regime selection: one built regime. Its predicate is checked in `validate()` (step 6); there is no
run-time selection between built regimes, so no regime-pinned test beyond the predicate's refusal
test (a shape with `RING_ENTRIES · stick_stride > ring_bytes_cap` raises `NotImplementedError`).

### Traffic ranking

Bytes per tier for the candidate splits (B_in = `N·H·W·C·elem` input bytes; output = `sh·sw·B_in`):

| Rank | Split | DRAM crossings (input / output) | Cross-core traffic | Notes |
|------|-------|-------------------------------|--------------------|-------|
| 1 (chosen) | input sticks (`stick` axis, flattened N·H·W) across clusters | 1 / 1 | none | minimum; write-balanced (each stick → `sh·sw` writes) |
| 1 (tie, deferred) | (stick, channel chunk) across clusters (`channel_chunk`) | 1 / 1 | none | same bytes, more and smaller transactions; occupancy for tiny N·H·W only |
| 3 | input rows (`batch`×`in_row`) across clusters | 1 / 1 | none | same bytes but coarser units: `[1,20,20,512]` has 20 rows for 32 clusters (under-fills); the stick split subsumes it |
| 4 (rejected) | output sticks / output rows across clusters | up to `sh·sw` / 1 | none | re-reads input across cluster boundaries |
| 5 (rejected) | replicas `(a, b)` across clusters | `sh·sw` / 1 | none | every replica cluster re-reads the stick |

No dependent axis exists, so "split a dependent axis + combine" has no candidate. Chosen: the input-stick
split — the cheapest-traffic split, and also the one with the most units (finest balance).

### Block schedule

Logical schedule per cluster (reader and writer are separate kernels; blocks pipeline through the ring):

```cpp
// cluster owns input sticks [start_stick, start_stick + core_sticks)
// num_blocks_this_core = ceil(core_sticks / block_sticks); last_block_extent = core_sticks - (num_blocks_this_core-1)*block_sticks
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);              // reader thread block_idx % R
    replicate_store_block(block_idx);   // writer thread block_idx % W_t
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_extent` sticks (= `block_sticks`, or `last_block_extent` for the last block) × `stick_bytes`, packed at `stick_stride` into ONE ring entry | the ring entry being filled | per block: 1 `reserve_back(1)`, `block_extent` tagged page reads (`trid 0`), 1 read barrier on `trid 0`, 1 `push_back(1)`. Per kernel: accessor built once; tag register reset once at exit |
| `replicate_store_block` | the same entry; `block_extent · scale_h · scale_w` output pages. Write order: `for a < scale_h: for j < block_extent: for b < scale_w: write stick j → out_page(s_j, a, b)` (each output row segment in increasing page id) | the entry, until its last write is barriered | per block: 1 `wait_front(1)`, `block_extent·scale_h·scale_w` tagged page writes (`trid 0`), 1 write barrier on `trid 0`, 1 `pop_front(1)`. Page index arithmetic: one division `s / W` per block, then stepped incrementally per stick. Per kernel: tag register reset once at exit |

A ragged last block keeps the nominal push/pop count (1 entry); only its transfer count shrinks
(`blocking-model.md` §2). A thread with no block (fewer blocks than threads on a cluster) loops zero
times and stays balanced.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Thread plan (default `(R, C, W) = (2, 0, 4)`) | The target page's DM-bound plan is (1, ·, 1) for implicit sync with one transfer per entry; this op's writer must be explicit (an entry feeds `sh·sw` writes) and issues `sh·sw`× the reader's transactions. Whether 4 writer harts on one cluster's NoC beat 2 is unmeasured | `(1,0,1)`, `(1,0,2)`, `(2,0,2)`, `(1,0,4)` on `ttsim-perf` (`python3 -m eval.model_cycles`) — one constant (`THREAD_PLAN`) |
| Overlap (default `BLOCKS_PER_WRITER_THREAD = 2`, below the coarsest "one block per writer thread") | The pipeline fill (first block's read before any write) is ~1/(2·W_t) of a cluster's reads; one block per writer thread doubles it, four halves it, at one barrier more per block | `BLOCKS_PER_WRITER_THREAD ∈ {1, 4}` |
| Grid synchronization (default: all clusters, split by sticks) | Small inputs give 1–2 sticks per cluster (`[1,8,8,4096]`: 2; `[1,4,4,8192]`: 1), so 2–3 of 4 writer threads idle and per-cluster launch cost dominates | split in units of `W_t` sticks (fewer clusters, every writer thread busy) |
| Implicit-sync reader | With `block_sticks = 1` an entry is one NoC read, so the reader could use implicit sync | entry = one stick, reader implicit (`noc.async_read<NocOptions::TXN_ID>(acc, ring, {.page_id}, {})`), writer explicit; tag handed back at exit |
| Ring vs. self-contained copier threads | Without compute, a ring exists only to overlap reads with writes across harts; six harts each reading a block into a private scratchpad and writing it out would drop the ring handshakes | 6 copier threads on `ScratchpadSpec` regions, overlap only across threads |
| Small sticks | `C = 32` bf16 gives 64 B pages (`[1,256,4,32]`, `[1,64,64,32]`): per-transaction cost dominates and no block size changes the transaction count | `trid_double_issue` (barrier on the oldest of several trids instead of per block) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input (DRAM or L1, interleaved, row-major) → ring | raw bytes of `block_extent` sticks, input dtype unchanged | reader threads (NoC0 default): `TensorAccessor(tensor::input)` page reads, explicit sync, every transfer `NocOptions::TXN_ID` with `{.trid = 0}`, barrier on the same id | one read per input page; the L1 placement is read through the same accessor |
| Ring `cb_input_sticks` | `entry_size = block_sticks · stick_stride` bytes; `data_format_metadata = input.dtype` (metadata only; the bytes never pass through an unpacker, so no format conversion — R1) | STRIDED DM→DM DFB, R producers, W_t consumers; listed in both kernels' `disable_dfb_implicit_sync_for_all` | one producer kernel, one consumer kernel |
| Ring → output (input's buffer type, interleaved, row-major) | raw bytes, unchanged | writer threads (NoC1 default): `TensorAccessor(tensor::output)` page writes from `read_ptr + j·stick_stride`, explicit sync, `trid 0` | `sh·sw` writes per stick |

No inter-cluster communication in any built or deferred regime.

**TARGET axes as unlocks of this scheme** (each `TARGET − PHASE0` value):

| Axis value | Class | What changes |
|------------|-------|--------------|
| `dtype = float32` | knob-turn | `stick_bytes` doubles (from `buffer_page_size()`); ring metadata float32; kernels unchanged (byte movers — R1 holds by construction) |
| `dtype = bfloat8_b` | not buildable on Quasar | refused in `validate()`; no refinement (target page *Family facts*); INVALID with ROW_MAJOR anyway |
| `layout = TILE_LAYOUT` | scheme-change | deferred `tile_layout` regime |
| `memory_config = L1` | in Phase 0 | output allocated in L1 (R5); same kernels |
| `sharding` | only `interleaved` in TARGET | — (`sharded_in_place` deferred row) |
| `channel_alignment = non_tile_aligned` | knob-turn | `stick_stride = align_up(stick_bytes, max(l1_align, dram_align))` already in the formula; gate only |
| `scale = per_axis` | knob-turn | kernels take `scale_h`, `scale_w` separately from Phase 0; gate only |

**Structural impossibilities**: none beyond the specification's `INVALID`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block of `block_sticks` contiguous input sticks (one ring entry) |
| Grid | `device.compute_with_storage_grid_size()` (32 clusters on this target); `ttnn.split_work_to_cores(grid, N·H·W, row_wise=True)` (catalog `noc_placement`: row-wise spreads a line over the DRAM-facing axis) |
| Per-core work | `core_sticks` ∈ {`per_core_1`, `per_core_2`} from the split, `start_stick` = running sum in core order; `num_blocks_this_core = ceil(core_sticks / block_sticks)`. Runtime args `start_stick`, `core_sticks` per cluster; clusters with 0 sticks are not in the work unit's `target_nodes` |
| Remainder | two core groups from `split_work_to_cores` (sizes differ by 1 stick); ragged last block per cluster (`last_block_extent`). No tiles in Phase 0; the deferred `tile_layout` regime uses per-image `ceil` tile counts |
| Bound | **data movement** — zero compute per stick against `(1 + sh·sw)·stick_bytes` moved per input stick |
| Thread plan | `(R, C, W) = (2, 0, 4)`, host constant `THREAD_PLAN = (READER_THREADS, WRITER_THREADS) = (2, 4)`; C = 0: no compute kernel (the op computes nothing). Reason: the writer issues `sh·sw`× the reader's transactions under explicit sync (forced: one entry feeds `sh·sw` writes) and barriers once per block, so four writer harts keep the NoC busy across one another's barriers; two readers cover the read side. R + W = 6 = all user DM harts; `4 % 2 == 0` (STRIDED rule) |
| Unit dealt to threads | the ring entry itself (one block): block `b` → reader thread `b mod R`, writer thread `b mod W_t`; threads may stop anywhere, no padded rounds (target page: "unless the unit dealt is the ring entry itself"). Kernels size their loops from the CT counts `reader_threads` / `writer_threads` and `get_my_thread_id()`, never `get_num_threads()` |
| Tests the plan needs | (1) `test_each_thread_moves_its_blocks`: a probe-only CT mask (`writer_thread_mask`, default all ones; not reachable from the public signature) makes writer thread `w` skip its writes (still wait/pop); with an output pre-filled with a sentinel, exactly the output pages of blocks `b ≡ w (mod W_t)` (predicted from the host split) keep the sentinel and all others match the reference; same for a reader mask (reader skips the reads, still reserve/push: those blocks' pages mismatch). Run on a shape where every thread of some cluster has a block (`[1,40,40,512]`: 8 blocks per cluster). (2) `test_session_order`: in one device session, a tt-metal program first (e.g. `ttnn.to_layout` of a TILE tensor to ROW_MAJOR), then upsample twice (two shapes), then tt-metal's `ttnn.to_layout(output, ttnn.TILE_LAYOUT)` on the result, compared with the reference. (3) `validate()` refusal of the `channel_chunk` predicate |

Worked Phase 0 numbers (quasar, 32 clusters, bf16, `BLOCKS_PER_WRITER_THREAD = 2`, `RING_ENTRIES = 8`):

| INPUTS entry | sticks N·H·W | sticks/cluster | `stick_bytes` | `block_sticks` | blocks/cluster | ring bytes | writes/cluster |
|--------------|-------------|----------------|---------------|----------------|----------------|------------|----------------|
| `[1,20,20,512]` (2,2) | 400 | 13 / 12 | 1,024 | 2 | 7 / 6 | 16,384 | 52 / 48 |
| `[1,40,40,512]` (2,2) | 1,600 | 50 | 1,024 | 7 | 8 (last 1) | 57,344 | 200 |
| `[1,40,40,256]` (2,2) | 1,600 | 50 | 512 | 7 | 8 | 28,672 | 200 |
| `[1,64,64,32]` 2 | 4,096 | 128 | 64 | 16 | 8 | 8,192 | 512 |
| `[1,8,8,4096]` 2 | 64 | 2 | 8,192 | 1 | 2 | 65,536 | 8 |
| `[1,4,4,8192]` 2 | 16 | 1 (16 clusters) | 16,384 | 1 | 1 | 131,072 | 4 |
| `[1,8,8,64]` 4 | 64 | 2 | 128 | 1 | 2 | 1,024 | 32 |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_input_sticks` (DFB `unique_id = "cb_input_sticks"`) | DFB 0 | `block_sticks · stick_stride` B (one entry = one block) | `RING_ENTRIES = RING_ENTRIES_PER_THREAD · max(READER_THREADS, WRITER_THREADS)` = 2 · 4 = 8 (2 per writer thread slice) | live set spans the `stick` axis by `block_sticks` per entry and `channel` whole; streams over blocks; `rep_h`/`rep_w` never stored (replicas are re-writes of the same bytes) | input dtype (bfloat16 in Phase 0), metadata only | reader (R = 2 threads, STRIDED) | writer (W_t = 4 threads, STRIDED) | explicit on both sides (`create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True)`, same for the writer); every NoC transfer tagged `trid 0`, barrier on `trid 0` before `push_back` / `pop_front` | whole program |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_extent` sticks × `stick_bytes`, at `stick_stride` | no (raw NoC + DFB API) | input tensor pages `[start_stick + block_idx·block_sticks, +block_extent)` | `cb_input_sticks`, 1 entry reserved and pushed | 1 entry posted to writer thread `block_idx mod W_t` |
| 2 | `replicate_store_block` | `block_extent · scale_h · scale_w` output pages | no (raw NoC + DFB API) | `cb_input_sticks`, 1 entry waited | output tensor pages `out_page(s, a, b)` | entry popped after the write barrier |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `DataflowBuffer::reserve_back(1)`, `push_back(1)`, `get_write_ptr()` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-244, 407` | 1 entry per block | — | `cb_input_sticks` | entry size = `block_sticks · stick_stride` |
| `load_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(accessor, dfb, size, {.page_id}, {.offset_bytes}, {.trid = 0})` (explicit form, as in `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader.cpp`) | `tt_metal/hw/inc/api/dataflow/noc.h:252` | `size = stick_bytes`, `offset_bytes = j · stick_stride` | input tensor | `cb_input_sticks` | `block_extent` reads per call site loop |
| `load_block` | raw_api | `Noc::async_read_barrier<NocOptions::TXN_ID>({.trid = 0})` | `tt_metal/hw/inc/api/dataflow/noc.h:739` | once per block | — | — | — |
| `replicate_store_block` | raw_api | `DataflowBuffer::wait_front(1)`, `pop_front(1)`, `get_read_ptr()` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:245-246, 408` | 1 entry per block | `cb_input_sticks` | — | — |
| `replicate_store_block` | raw_api | `Noc::async_write<NocOptions::TXN_ID>(dfb, accessor, size, {.offset_bytes}, {.page_id}, {.trid = 0})` (explicit form, as in `toy_thread_plan/kernels/writer.cpp`) | `tt_metal/hw/inc/api/dataflow/noc.h:411` | `size = stick_bytes`, `offset_bytes = j · stick_stride`, `page_id = out_page(s_j, a, b)` | `cb_input_sticks` | output tensor | `scale_h`, `scale_w`, `block_extent` |
| `replicate_store_block` | raw_api | `Noc::async_write_barrier<NocOptions::TXN_ID>({.trid = 0})` | `tt_metal/hw/inc/api/dataflow/noc.h:760` | once per block | — | — | — |
| both (exit) | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())` (reader) / `noc_async_write_set_trid(0, noc.get_noc_id())` (writer) as last statement | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558, 2579` | user id 0 | — | — | — |
| both | raw_api | `get_my_thread_id()` | `tt_metal/hw/inc/api/kernel_thread_globals.h:155` | thread index for the block dealing | — | — | `reader_threads`, `writer_threads` (CT) |
| host | host API | `ttnn.split_work_to_cores(grid, N·H·W, row_wise=True)` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:471` | — | — | — | grid |
| host | host API | `ttnn.experimental.DataflowBufferSpec`, `KernelSpec(num_threads=...)`, `producer_of` / `consumer_of`, `WorkUnitSpec`, `ProgramSpec`, `create_reader_datamovement_config` / `create_writer_datamovement_config(disable_dfb_implicit_sync_for_all=True)` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:302, 567, 772, 798, 811, 841, 864, 470-479` | pattern: `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` (reader + writer only; no compute `KernelSpec`) | — | — | `num_threads`, `entry_size`, `num_entries` |
| host | host API | `ttnn.generic_op([input, output], spec, run_args, {"input": 0, "output": 1})` — the one dispatch | as used in `toy_thread_plan.py` | — | — | — | — |

**Helpers considered and rejected** (applies to every `raw_api` row above):

| Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|---------------------------------|
| Compute helpers (`eltwise_convenience.hpp` `copy`, `tilize_helpers.hpp`, `untilize_helpers.hpp`) | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp`, `tilize_helpers.hpp`, `untilize_helpers.hpp` | they run on a compute kernel over tiles; this op has no compute stage (spec DEFINITION: "moves values and computes nothing"; GUIDANCE: "no compute kernel is needed") and row-major sticks are not tiles. Routing bytes through unpack/DEST/pack would add a stage and, for float32, a format path R1 forbids risking. They return in the deferred `tile_layout` regime |
| `local_copy_helpers_dataflow.hpp` (`local_addr`, `set_read_state` / `read_with_state`, `set_read_trid`, `async_read_barrier_with_trid`) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57, 74-95, 147, 158` | `local_addr` aims at this core's own L1 (`:57`, "pointing at (my_x, my_y, addr)"), and the stateful read takes a compile-time `transfer_size` with fixed source coordinates (`:74`, `:94`); every page here is a different bank address resolved by the `TensorAccessor`. The trid pair is the `trid_double_issue` perf lamp, not a Phase 0 need |
| `mcast/kernel/mcast_args.hpp` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` | no reuse-shared operand: each input stick is read by one cluster |
| `tilize_helpers_dataflow.hpp`, `tile_column_dataflow.hpp`, `index_tile_dataflow.hpp` | `ttnn/cpp/ttnn/kernel_lib/` | tile-oriented dataflow helpers; the Phase 0 unit is a row-major page |

No kernel-library helper moves interleaved row-major pages through a DFB with replication;
the block operations are realized with the raw Device-2.0 NoC/DFB API as above.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Untagged NoC transfer | Both kernels are explicit-sync; a plain `noc.async_read` / `async_write` reuses the hart's last ring tag | every transfer `NocOptions::TXN_ID` + `{.trid = 0}`, barriers on `trid 0`; both kernels list the ring out of implicit sync (`disable_dfb_implicit_sync_for_all=True`); last statement resets the tag; `test_session_order` |
| STRIDED dealing out of step between reader and writer | Producer `p`'s k-th push lands in slot `p + k·R`, consumer `w`'s j-th pop reads slot `w + j·W_t` | both kernels deal block `b` to thread `b mod (their count)` in increasing `b`; never `get_num_threads()` |
| Ragged last block | `core_sticks` not a multiple of `block_sticks` | push/pop stay 1 entry; only the transfer loop narrows to `last_block_extent` |
| Popping before the writes left L1 | The same entry feeds `sh·sw` writes | write barrier on `trid 0` before `pop_front(1)` |
| Output page arithmetic (batch fold, per-axis factors) | `out_page = (q·scale_h + a)·(W·scale_w) + w·scale_w + b` with `q = s / W`; a block may straddle input rows and images | compute `q, w` once per block, step per stick (wrap `w` at `W`); `scale_h`, `scale_w` are separate args from Phase 0, so `per_axis` cells exercise the same code |
| Precision (R1: exact) | float32 later: any trip through a compute engine or a bf16 ring would round | the ring holds raw bytes of the input dtype; no compute kernel; `data_format_metadata` = tensor dtype |
| L1-placed tensors and the ring share each cluster's L1 | the `[1,40,40,512]` × L1 cell puts ~256 KiB of tensor banks per cluster beside a 56 KiB ring | ring capped at `RING_MAX_BYTES` (Phase 0 max 128 KiB); the program build's DFB/L1-buffer clash check reports any overlap as an error |
| `scale_factor` form | int and pair must land in the same cell (R2); `True` is an `int` in Python | normalize first; reject `bool`, non-int, `< 1`, wrong length with `ValueError` |
| Wrong output placement (R5) | `allocate_tensor_on_device` with a DRAM-only spec would put an L1 input's result in DRAM | build the output `TensorSpec` from `input.memory_config()` |
| One dispatch | a convenience `ttnn.to_layout` / `ttnn.repeat_interleave` composition would violate the one-dispatch rule | exactly one `ttnn.generic_op` per call |
| Thread plan uses all six user DM harts | R + W ≤ 6 | no further DM kernel can be added without lowering the plan |

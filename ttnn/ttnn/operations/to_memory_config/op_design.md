# Operation Design: to_memory_config

The specification `eval/specs/to_memory_config/spec.py` governs. This document says how to build what it asks for.

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel) |
| Goal | Return a copy of `tensor` (same shape, dtype, layout, bytes) that lives where `memory_config` says: DRAM interleaved, L1 interleaved, or L1 height-sharded with that config's shard shape, cores and orientation. The input may be DRAM interleaved, L1 interleaved or L1 height-sharded. |
| Math | `output.page[p] = input.page[p]` for every page id `p` of the tensor's page space. A page is a tile in TILE layout and a row in ROW_MAJOR layout. Layout and dtype are unchanged, so input and output have the same page space and page size. Only where each page is stored changes. |
| References | spec `eval/specs/to_memory_config/spec.py` (RULES R1–R8, GUIDANCE); `eval/memory.py` (`memory_axes`, `height_shards`, `place`); `.claude/references/quasar_op_examples/toy_thread_plan/` (host `ProgramSpec` pattern, `reader_implicit.cpp`, `writer_implicit.cpp`); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A and C4; catalog `ttnn/ttnn/operations/examples/master.md` entries `double_buffer`, `read_coalescing`, `noc_placement`, `split_reader` |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters. The op queries `device.compute_with_storage_grid_size()` at run time | target page; probe 2026-11-21 returned 8 × 4 |
| L1 per core | 4,194,304 B | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`, with `l1_reserve_bytes = 0` = 3,963,328 B on this target. The ring spends at most `RING_BYTES_CAP = 262,144 B` of it (see Mechanism caps) | probe 2026-11-21 |
| Alignment | L1 16 B, DRAM 64 B. The op queries `ttnn.get_l1_alignment()` / `ttnn.get_dram_alignment()`; it reads per-tensor values from `tensor.buffer_aligned_page_size()` | target page; probe: a row-major `[1,1,17,50]` bf16 page is 100 B, aligned 112 B in L1 and 128 B in DRAM |
| Host program | `ttnn.experimental.ProgramSpec` + `ttnn.generic_op` — reference `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/toy_thread_plan_program_artifacts.py`) | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}); 6 user data-movement harts (R + W ≤ 6). On STRIDED rings R and W each divide C | target page, *Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal 1 kernels: refused at kernel creation. Legacy C++ device op with a program factory: same refusal. `bfloat8_b` / `bfloat4_b`: no path on this part (they are not in TARGET). Kernel-library compute helpers: none are used. Both kernels hand the transaction tag back as their last statement. Python `DFBAdvancedOptions` exposes only `alias_with` (`ttnn/cpp/ttnn-nanobind/program_spec.cpp:283-297`). No Python binding maps an L1 bank to its core (`ttnn/cpp/ttnn-nanobind/device.cpp:207-213` exposes per-bank sizes only) | target page |

Every worked number below is for `quasar` (8 × 4 grid, 4 MiB L1). Every run-time predicate uses the queries above.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `tensor` | `ttnn.Tensor` | yes | on device; axes inside `SUPPORTED` (= spec `PHASE0`) | — | host |
| `memory_config` | `ttnn.MemoryConfig` | yes (no default; X2: `None` is refused) | read into the axis `to` by the spec's `canonical_to` (R3): `"DRAM"`, `"L1"`, `"L1/height"`, `"L1/width"` (Refinement 0). Anything else is refused by name (`L1/block`, `DRAM/height`, …). A non-config (`"L1"`, `None`) raises `TypeError` from `canonical_to` (X1, X2) | — | host |

The signature is exactly the stub `to_memory_config(tensor, memory_config)`. No other parameters, no keyword knobs. The op has no index axes, so there is nothing to canonicalize.

**`validate()`** (R5, R8). It runs on the Python side and raises `NotImplementedError` outside SUPPORTED. It derives the axes as follows:

| Axis | Derived with |
|------|--------------|
| `dtype`, `layout` | `tensor.dtype`, `tensor.layout` |
| `memory_config`, `sharding` | `eval.memory.memory_axes(tensor)` |
| `to` | `canonical_to(memory_config, …)` from the spec |
| `alignment`, `rank` | `INPUT_TAGGERS = AXES`, imported from the spec and never redefined |

It gates every axis on `SUPPORTED`. It also refuses two mechanism-cap violations (see Mechanism caps). There are no `EXCLUSIONS` in Phase 0.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 2, 3 or 4, any H and W (tile-aligned or not) |
| Dtype | bfloat16 (Phase 0). float32 is a TARGET refinement |
| Layout | TILE or ROW_MAJOR |
| Memory | DRAM interleaved, L1 interleaved, or L1 height- or width-sharded (any grid, either orientation; a row-major width shard must divide W) |

### Output

| Property | Value |
|----------|-------|
| Shape | `tensor.shape` |
| Dtype | `tensor.dtype` |
| Layout | `tensor.layout` |
| Memory | exactly `memory_config` (R2). Allocated with `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(tensor.shape, tensor.dtype, tensor.layout, memory_config), device)`, which carries the shard spec through (`ttnn/cpp/ttnn-nanobind/tensor.cpp:290-305`) |

**R7 decision: always copy, one dispatch.** When the input already has the requested config the op still copies. The run rule demands exactly one device dispatch per invocation, and returning the input would be zero dispatches. Under the owner assignment below, an equal-config sharded copy is entirely core-local: each core moves its own shard to its own output shard, and no page crosses the mesh.

## Blocking Model

### Page space (the single index space of the op)

| Quantity | Formula (alignment-aware, per image) | Source |
|----------|--------------------------------------|--------|
| `tensor_rows_pages` | TILE: `prod(shape[:-2]) * ceil(H / tile_h)`. ROW_MAJOR: `prod(shape[:-1])` | host, from `tensor.shape` and `tensor.tile.tile_shape` |
| `tensor_cols_pages` | TILE: `ceil(W / tile_w)`. ROW_MAJOR: 1 (a page is a whole row) | host |
| `num_pages` | `tensor_rows_pages * tensor_cols_pages`, the page ids the accessors map. Global row-major page order, the same for input and output | host |
| `page_bytes` | `tensor.buffer_page_size()`: 2048 B for a bf16 tile, `W * 2` B for a bf16 row | host |

For a rank-2 shape, `prod(shape[:-2])` is 1. Each image is tile-padded on its own: `[2,1,100,47]` TILE has `2 * 4 * 2 = 16` pages, as the probe confirmed.

### Entry space (Refinement 0: width sharding)

Width sharding breaks "one page space for both sides" in ROW_MAJOR only. The accessor's page shape of a
width-sharded tensor is `(1, shard_w)` (`buffer_distribution_spec.cpp`, `convert_shape_to_pages`): a page is a
`shard_w`-wide run of one row, and page `r * (W / shard_w) + c` is run `c` of row `r`. In TILE layout a width
shard is whole tiles wide and its page is still a tile, so the Phase 0 page walk served width shards on both sides
unchanged (probe 2026-11-21, gate bypassed: 15/15 TILE cases exact, 15/15 ROW_MAJOR cases wrong).

The ring therefore carries **entries**, not pages:

| Quantity | TILE | ROW_MAJOR |
|----------|------|-----------|
| `Pi`, `Po` (elements of a row per page, each side) | — | `shard_w` when that side is width-sharded, else `W` |
| entry width `L` | one tile | `lcm(Pi, Po)` (divides `W`, since both do) |
| `in_pieces`, `out_pieces` | 1, 1 | `L / Pi`, `L / Po` |
| entry `g` holds | input page `g`, output page `g` | input pages `g * in_pieces + i`, output pages `g * out_pieces + j` |
| `num_entries` | `num_pages` | `prod(shape[:-1]) * W / L` |
| `entry_bytes` | `min(in_aligned_page, out_aligned_page)` (Phase 0) | same when both sides have one piece; else `L * elem` rounded up to `ttnn.get_l1_alignment()` |

No page straddles an entry, so no transfer needs an offset into a tensor page. A side with one piece per entry
keeps implicit sync (one tagged transfer per entry, `entry_bytes` from/to the page slot, which holds it). A side
with several pieces assembles (reader) or splits (writer) the entry with sized transfers of `Pi * elem` /
`Po * elem` bytes at entry offset `i * piece_bytes`; it keeps the ring's credits itself (`reserve_back` /
`push_back` or `wait_front` / `pop_front`, one entry at a time, one barrier per entry), lists the ring in
`disable_dfb_implicit_sync_for`, and tags every transfer with the user id 0. **Quasar's NoC needs no address
alignment** (`noc_parameters.h`: `NOC_L1_*_ALIGNMENT_BYTES = NOC_DRAM_*_ALIGNMENT_BYTES = 1`, "the Quasar NOC
shifts the data"), so a 2-byte piece at an odd 2-byte offset is one legal transfer; on ttsim 84/84 probe cases
were exact, including `[32, 17]` RM (17 shards of one column, 2-byte pages).

Examples (bf16, 8 × 4 grid, the runner's width shards): `[1,1,32,256]` RM DRAM → L1/width (32 shards of 8
columns): `L = 256`, pieces (1, 32), R-bal. L1/width → L1/width with equal configs: (1, 1), R-in, core-local.
Two width grids of 64 and 96 columns of `W = 192`: `L = 192`, pieces (3, 2), both sides explicit.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| leading (batch) dims `shape[:-2]` | independent: every output element is the input element at the same index, so no result spans any axis | folded into `page` | whole extent, folded | `num_pages` formula | through `page` | — (folded) |
| H (tile-rows in TILE, rows in RM) | independent: same reason | folded into `page` | whole extent, folded. In row-major page order, H and the leading dims are contiguous and the copy treats them identically | `num_pages` formula | through `page` | — (folded) |
| W (tile-columns in TILE; bytes of a row in RM) | independent: same reason | TILE: folded into `page`. RM: inside one page | TILE: folded. RM: the whole row in one page, not split. A row-major page is a whole row, and a height shard's width is the full width, so splitting W would mean sub-page transfers that serve nothing in TARGET (width/block sharding is not in TARGET) | `num_pages` formula / `page_bytes` | through `page` | scheme-change only if width sharding enters TARGET |
| **page** (flattened page id, the axis the op is blocked on) | independent: page `p` of the output depends only on page `p` of the input | `block_pages` (pages per block) | coarsest that fits, at most the whole per-core assignment: `block_pages = core_pages_max` when that fits one block, else `min(TXN_ENTRIES_CAP, ring_budget // (BUFFER_DEPTH * entry_bytes))` (see derivation below) | one host value, `block_pages`, computed once in the program builder. The DFB `num_entries` and the kernels' block trip counts derive from it | contiguous page ranges per core. Owner-aligned when a side is sharded (regimes R-in / R-out); otherwise balanced over the full grid with `ttnn.split_work_to_cores(grid, num_pages, row_wise=True)` (regime R-bal). Per core: RT args `start_page`, `num_pages_this_core`, `page_stride` (= 1 in Phase 0) | knob-turn (block, depth, stride, core list) |

There is no dependent axis and no reuse-shared axis. **Operand-reuse check:** the op has one operand, and it varies along the split axis (each core reads only its own pages), so the split re-reads nothing. **No stage computes on a designated core while others wait.** Every engaged core does the same two block operations on its own pages, and no core collects, coordinates or hands out work. The op has no intermediate data beyond the staged pages themselves, which have the same page axis.

**`block_pages` derivation (single source).** This is one host function. It runs after the work split, because it needs the largest per-core count:

```
entry_bytes        = min(input.buffer_aligned_page_size(), output.buffer_aligned_page_size())
ring_budget        = min(RING_BYTES_CAP, ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes)
core_pages_max     = max(num_pages_this_core over engaged cores)
single_block_cap   = min(TXN_ENTRIES_CAP, ring_budget // entry_bytes)
if core_pages_max <= single_block_cap:  block_pages = core_pages_max;  depth = 1
else:  block_pages = min(TXN_ENTRIES_CAP, ring_budget // (BUFFER_DEPTH * entry_bytes));  depth = BUFFER_DEPTH
ring_entries       = block_pages * depth
```

Named host constants: `RING_BYTES_CAP = 262144`, `BUFFER_DEPTH = 2`, `TXN_ENTRIES_CAP = 255`, `l1_reserve_bytes = 0`, `THREAD_PLAN = (1, 0, 1)`. Tests monkeypatch them; they are not parameters of the public signature.

When every core's whole assignment fits one block, the ring is exactly that block: depth 1, no unused half. With depth 1 the runtime still slices the ring into 2–4 transaction ids whenever `block_pages` has a divisor in {2, 3, 4} (`DataflowBuffer.md` C4, "How many txn ids"), so reads and writes still overlap at slice granularity.

**Worked Phase 0 values (quasar, 32 cores):**

| Case | Regime | Engaged cores × pages | entry_bytes | block_pages × depth | Ring bytes |
|------|--------|-----------------------|-------------|---------------------|-----------|
| `[1,1,1600,256]` TILE, DRAM → L1 | R-bal | 16 × 13 + 16 × 12 (400 pages) | 2048 | 13 × 1 | 26,624 |
| `[1,1,1600,256]` RM, L1 → DRAM | R-bal | 32 × 50 | 512 | 50 × 1 | 25,600 |
| `[1,1,64,512]` TILE, L1/height (2 cores) → L1 | R-in | 2 × 16 | 2048 | 16 × 1 | 32,768 |
| `[1,1,32,4096]` TILE, L1/height (1 core) → DRAM | R-in | 1 × 128 | 2048 | 128 × 1 | 262,144 |
| `[1,1,17,50]` RM, DRAM → L1 | R-bal | 17 × 1 | 112 | 1 × 1 | 112 |
| reshard `[1,1,64,512]` TILE, 1 core → 2 cores | R-out | 2 × 16 | 2048 | 16 × 1 | 32,768 |
| a caller's 1-core shard of `[1,1,1600,256]` TILE → DRAM | R-in | 1 × 400 | 2048 | 64 × 2 | 262,144 |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `dfb_pages` | `BUFFER_DEPTH` (used only when a core's assignment exceeds one block) | 1 when `core_pages_max <= single_block_cap`, else 2 | Depth 2 lets the reader fill block `b+1` (its own transaction id) while the writer drains block `b`. That is the DM0-ISR double buffering of `DataflowBuffer.md` "Implicit sync" / C4. Depth 1 means the whole assignment is resident at once, and overlap comes from the runtime's 2–4 transaction-id slices |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Implicit-sync entries per transaction id fit in a `uint8` (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:1323-1400`, 1446-1474) | `block_pages` (with depth 2 the runtime picks 2 txn ids, so one id covers one block; with depth 1, an id covers `block_pages / n`) | `block_pages <= TXN_ENTRIES_CAP = 255` | the host rejects the DFB |
| Implicit-sync transfer size = the DFB's `entry_size`, not the page size (`tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:775-882`; `noc.h:913-944` takes no size) | `entry_bytes` | `entry_bytes = min(in_aligned_page, out_aligned_page)`. This is ≥ `page_bytes` (both aligned sizes are) and ≤ each side's page slot | `entry_bytes` larger than the destination's aligned slot (e.g. 128 B DRAM slot into a 112 B L1 slot) overwrites the first bytes of the next page: silent corruption of ragged row-major cells |
| One NoC packet per implicit STRIDED transfer, max 64 KiB (`NOC_MAX_BURST_SIZE`, `tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:393-396`) | `entry_bytes` | `validate()` refuses `entry_bytes > 65536` with `NotImplementedError` (a row-major row wider than 32,768 bf16 elements; nothing in INPUTS comes close) | a transfer split across packets under one implicit credit; behaviour not established |
| Ring budget: the ring shares each core's L1 with L1-resident inputs and outputs, which may be an entire tensor's shard on one core | `block_pages * depth * entry_bytes` | `<= ring_budget = min(RING_BYTES_CAP, unreserved query - l1_reserve_bytes)`. `validate()` refuses a page that does not fit `BUFFER_DEPTH` entries (`ring_budget // (BUFFER_DEPTH * entry_bytes) < 1`, i.e. `entry_bytes > 131072`), which is already excluded by the 64 KiB cap | the ring and an L1 tensor overlap: program launch fails |
| Row-major width shard that does not divide `W` (Refinement 0) | entry width `L` | `entry_space()` raises `NotImplementedError` when `W % shard_w != 0` in ROW_MAJOR (the runner's width shards always divide; a caller's could not) | the last page of every row is short, so an entry would straddle a page |
| Shard geometry | per-core page range in R-in / R-out | a core's range is its shard's page range: `[k * pages_per_shard, min((k+1) * pages_per_shard, num_pages))`. Shards past the data get 0 pages and are left out of the work unit | out-of-range page ids would make the accessor address padding past the tensor |

### Regimes

The data-movement minimum is stated against the source and destination tiers. Each page leaves the input's storage once and enters the output's storage once. A Tensix core can only read into and write out of its own L1, so a DRAM → DRAM page always stages through some core's L1. That staging is the minimum for that pair.

| Regime | Status | Predicate (host, exact) | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-------------------------|-------|---------------------------|--------------------------|
| **R-in** (input-owner): each core of the input's shard grid moves its own shard's pages | built | `in_sharded and (not out_sharded or in_shards >= out_shards)`, where `x_shards = ceil(num_pages / pages_per_shard(x))` | `block_pages` contiguous pages of the core's shard | minimum on the mesh: the read is core-local (input accessor → own L1, a loopback that never enters the mesh), the write is one hop to the destination. Above minimum by one core-local L1→L1 copy per page (staging ring), which regime R-borrow removes | fewer ISR post/ack batches (one per block per side), fewer ring wraps; the regime adds no fixed cost |
| **R-out** (output-owner): each core of the output's shard grid pulls its own output shard's pages | built | `out_sharded and not R-in` | `block_pages` contiguous pages of the core's output shard | minimum on the mesh: one hop (or a DRAM read) in, a core-local write out. Above minimum by one core-local copy per page (staging ring → own shard) | same as R-in |
| **R-bal** (balanced staging): pages split evenly over the whole grid | built | `not in_sharded and not out_sharded` | `block_pages` contiguous pages of the core's range | DRAM → DRAM: minimum. Any L1-interleaved side: each page takes two hops (source → staging core → destination bank) instead of one. The extra hop is the staging core ≠ bank owner, closed by R-bank | same as R-in |
| **R-bank** (L1-interleaved bank-owner): the core holding a page's L1 bank moves it | deferred: the host needs the L1 bank → core map of the allocator, and no Python binding exposes it (`ttnn/cpp/ttnn-nanobind/device.cpp:207-213` exposes per-bank sizes only). Reachable without kernel change: kernels already take `page_stride`, so a bank-owner split is `start_page = bank`, `page_stride = num_l1_banks` per core | `not in_sharded and not out_sharded and (in_L1 or out_L1)` | as R-bal | saves one mesh hop per page for L1-interleaved cells (traffic delta: `num_pages * page_bytes` mesh bytes); DRAM crossings unchanged | same |
| **R-borrow** (zero-copy shard side): the owner side's DFB is backed on the shard itself (`DataflowBufferSpec.borrowed_from` / `ttnn.experimental.dfb_spec_from_sharded_tensor`, `ttnn/cpp/ttnn-nanobind/program_specs.cpp:24-81`). R-in: the reader only marks entries ready and the writer drains the shard. R-out: the reader lands pages directly in the output shard | deferred. DRAM and mesh crossings are identical to R-in / R-out; the only delta is one core-local L1 copy per page. It also needs a second protocol: a borrowed ring's entry count is the shard's page count (up to 400 in INPUTS-derived configs), which exceeds the 255-entries-per-txn-id implicit-sync cap with no guaranteed divisor, so that side must use explicit sync with tagged transfers. Reachable: the per-core page ranges are already shard-aligned, and the ring is one named DFB whose spec swaps | R-in / R-out predicates | whole shard (one block) | removes `num_pages * page_bytes` of core-local L1 copies; mesh/DRAM unchanged | one block = whole shard: one mark/drain handshake |
| **R-split-W** (split tile-columns across cores) | rejected: superseded by R-bal / R-in / R-out. It moves the same bytes, breaks the shard-contiguous page ranges the owner regimes rely on, and there is no dependent axis that would motivate it | — | — | same bytes as R-bal, more fragmented | — |
| **R-alias** (return the input when configs are equal) | rejected: zero device dispatches violates the one-dispatch rule. Superseded by R-in, which makes an equal-config sharded copy entirely core-local | — | — | — | — |
| **R-bal-for-sharded** (ignore shard ownership, balance over the grid) | rejected: superseded by R-in / R-out. It adds a second mesh hop per page, and it does not relieve the shard cores' NoC ports, which carry every byte of the sharded side either way. Kept reachable as a perf lamp, since it is only a different core list | — | — | +1 mesh hop per page | — |

**Width sharding (Refinement 0).** R-in / R-out also take a width-sharded side, when its shards own whole
entries (`shard_w % L == 0`; always in TILE, and in ROW_MAJOR when the other side cuts rows no wider than the
shard). A width shard `k` owns columns `[k * sc, (k+1) * sc)` of every entry row (`sc = shard_w / L`), so its core
row is `(first = k * sc, run_len = min(sc, entry_cols - k * sc), run_stride = entry_cols, count = run_len *
entry_rows)`, and the kernels walk entry `first + (i / run_len) * run_stride + (i % run_len) * entry_stride`. A
height shard or an R-bal range is one run (`run_len = count`). A row-major width side whose shards are narrower
than an entry (interleaved or height on the other side: `L = W`) cannot own entries and falls to R-bal or to the
other side's owner split. `x_shards` counts entry columns for a width grid and entries for a height grid.

**Regime selection function** (host, before allocation of the program):

```
in_sharded  = the input is height- or width-sharded and its shards own whole entries
out_sharded = the output is height- or width-sharded and its shards own whole entries
if in_sharded and (not out_sharded or in_shards >= out_shards): regime = R-in   (cores = input shard grid, orientation order)
elif out_sharded:                                               regime = R-out  (cores = output shard grid, orientation order)
else:                                                           regime = R-bal  (cores = split_work_to_cores(grid, num_pages, row_wise=True))
```

Core order for a shard grid is `ttnn.corerange_to_cores(shard_spec.grid, None, row_wise=(shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR))`. Shard `k` goes to the `k`-th core. `pages_per_shard` is `(shard_h / tile_h) * (shard_w / tile_w)` in TILE and `shard_h` in ROW_MAJOR.

**Correctness does not depend on the regime.** Every page `p` is read through the input accessor and written through the output accessor with the same `p`, so a wrong core-assignment costs traffic, never values. The three regimes are one kernel pair with different `(core, start_page, num_pages_this_core)` tables. Regime-pinned tests are required (see Work Distribution).

### Traffic ranking

| Candidate split | DRAM crossings per tensor | Mesh hops per page | Core-local copies per page | Verdict |
|-----------------|---------------------------|--------------------|----------------------------|---------|
| Owner-aligned page ranges (R-in / R-out) for any sharded side | input 1 (if in DRAM), output 1 (if in DRAM) | 1, or 0 for equal-config sharded | 1 (staging ring) | **chosen** wherever a side is sharded |
| Owner-aligned + borrowed shard ring (R-borrow) | same | same | 0 | cheapest; deferred (row above) |
| L1-bank-owner page ranges (R-bank), interleaved L1 sides | same | 1 | 1 | cheaper than R-bal on L1-interleaved cells; deferred (bank map unavailable) |
| Balanced page ranges over the grid (R-bal) | same | 2 if any side is L1, else 0 (DRAM → core → DRAM) | 0 | **chosen** for interleaved ↔ interleaved |
| Balanced, ignoring shard ownership | same | 2 | 0 | rejected (+1 hop, same port bottleneck) |
| Split tile-columns (W) | same | as R-bal | 0 | rejected (no gain, fragments shard runs) |
| Split a dependent axis + combine | — | — | — | none exists: no output element depends on more than one input element |

DRAM crossings are identical across all splits: every tensor crosses DRAM at most once each way. The ranking is decided on mesh hops, then core-local copies. Occupancy is not part of this ranking. Engaging all 32 clusters in R-bal is the occupancy step taken within the chosen split.

**Stall-shadow check.** The writer waits on read credits and the reader waits on ring space. The only work independent of a pending read is the next block's reads. Depth 2, or depth 1 with 2–4 txn-id slices, already schedules those into the wait. No algebraic reorder exists or is needed (no arithmetic, no floating-point order to preserve).

### Block schedule

```cpp
// per core, logical schedule; the reader and writer realize their halves asynchronously
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // num_blocks_this_core = ceil(num_pages_this_core / block_pages)
    load_block(block_idx);    // reader: pages [start_page + (block_idx*block_pages + j)*page_stride] -> dfb_pages
    store_block(block_idx);   // writer: dfb_pages -> the same page ids through the output accessor
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|----------------------------------|
| `load_block` | `block_pages` pages (`last_block_extent = num_pages_this_core - (num_blocks_this_core-1)*block_pages` on the last block) | the ring slice being filled; the input `TensorAccessor` (built once per kernel) | One NoC read per page. That is necessary, not a unit loop left over by accident: interleaved pages sit in different banks and are never contiguous, so one transaction per entry is what implicit sync issues (`DataflowBuffer.md` "Implicit sync": one entry per STRIDED transaction). No barrier, reserve or push in the kernel. One DM0-ISR credit post per txn-id slice. The transaction tag is handed back once, at kernel exit |
| `store_block` | same | the ring slice being drained; the output `TensorAccessor` | One NoC write per page. No barrier, wait or pop in the kernel. One DM0-ISR ack per txn-id slice. Firmware waits for the writes to land at kernel exit. Tag handed back once at exit |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap** | When a core's whole assignment is one block (depth 1), a `block_pages` with no divisor in {2,3,4} gets one txn id. The writer then starts only after every read of the core has landed: reads and writes serialize | `block_pages = ceil(core_pages_max / 2)` with depth 2, and the catalog `double_buffer` sweet spot (4–8 pages, depth 2) |
| **Grid synchronization / occupancy** | R-in / R-out engage only the shard grid, as few as 1 core (a TILE tensor of 32 padded rows is one shard). The other 31 clusters idle | R-bal-for-sharded (the full grid, +1 mesh hop per page). It is the same kernels with a different core table. Expect no gain, because the shard cores' ports carry every byte either way |
| **Thread plan** | One reader and one writer hart may be issue-bound on many small transfers (row-major rows of 100 B, 1 page per core) rather than bandwidth-bound | (2, 0, 2) against (1, 0, 1) on the timed backend: STRIDED over 2 harts per side (catalog `split_reader`) |
| **Coalesced shard runs** | Inside one shard, consecutive pages are contiguous in L1, but each page is a separate transaction. Catalog `read_coalescing` measured 1.11–3.66× from one wide call per run | one transfer per contiguous run inside a shard (R-out reading a remote input shard during a reshard; R-in writing a remote output shard). It needs explicit sync on that side, or BLOCKED implicit with `block_pages * entry_bytes <= 64 KiB` |
| **NoC placement** | Reads on NoC0, writes on NoC1 (the `create_reader/writer_datamovement_config` defaults) and `row_wise=True` placement (catalog `noc_placement`) were measured on another part | the reverse NoC assignment; `row_wise=False` |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input storage → `dfb_pages` | pages as stored (tile or row), `entry_bytes` per entry | reader: `noc.async_read<NocOptions::TXN_ID>(input_accessor, dfb_pages, {.page_id = p}, {})`, implicit sync | The input `TensorAccessor` maps `p` to a DRAM bank, an L1 bank, or the core holding its shard (`tensor_accessor.h:253-262`, 440-455). For R-in this is a local read |
| `dfb_pages` → output storage | unchanged bytes | writer: `noc.async_write<NocOptions::TXN_ID>(dfb_pages, output_accessor, {}, {.page_id = p})`, implicit sync | For R-out this is a local write |
| kernel exit | — | reader: `noc_async_read_set_trid(0, noc.get_noc_id())`. Writer: `noc_async_write_set_trid(0, noc.get_noc_id())` | hands the sticky transaction tag back to tt-metal's untagged programs (target page, *Ring protocols*) |

There is no compute, no format conversion and no tilize or untilize. R1 holds by construction: bytes are moved, never unpacked.

**Placement axes as regimes (TARGET `sharding` / `to`).**

| Placement | Cut axis | Cost class |
|-----------|----------|------------|
| `L1/height` on either side | the H/page axis (independent) | **knob-turn**: a different core table (R-in / R-out), same kernels |
| `L1` interleaved and `DRAM` | none | R-bal |
| width / block / nd sharding | not in TARGET; refused by name (R3, R5) | if added: width/block cut W, which is still independent here (each page is copied alone). It would need sub-row pages in ROW_MAJOR (a different page definition), so it is a scheme-change in page geometry, not in the topology |

No regime has a cross-core combine, multicast or semaphore.

**float32 (the remaining TARGET dtype).** This is a knob-turn: `page_bytes`, and therefore `entry_bytes`, doubles, and the DFB `data_format_metadata` follows the dtype. Nothing else changes, and bytes are still moved exactly.

**Structural impossibilities.** None to add beyond the spec's INVALID.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block of `block_pages` contiguous page ids (stride `page_stride` = 1) of a core's range |
| Grid | `device.compute_with_storage_grid_size()` (8 × 4 here). R-bal spreads over all of it. R-in / R-out use the caller's shard grid, which may be any subset |
| Per-core work | R-bal: `ttnn.split_work_to_cores(grid, num_pages, row_wise=True)` gives two groups (per_core_1, per_core_2), assigned contiguous ranges in core order, as `toy_thread_plan_program_artifacts.split_tiles` does. R-in / R-out: shard `k`'s page range goes to the `k`-th core of that shard grid. Cores with 0 pages are left out of `WorkUnitSpec.target_nodes` |
| Remainder | R-bal: split_work_to_cores' second group carries one page fewer. R-in / R-out: the last shard may be partial (`min(...)` above). Per-core ragged last block: `last_block_extent`, same block operation. Page counts are `ceil` per image (Page space table) |
| Bound | **data movement**. Per page the op does one NoC read and one NoC write and zero compute, so bytes are the only limit |
| Thread plan | `THREAD_PLAN = (R, C, W) = (1, 0, 1)`: one reader hart, no compute kernel (the op computes nothing), one writer hart. The target page's rule for an op bound by data movement is one thread of each kind with implicit sync. R and W are passed to the kernels as compile-time args `reader_threads` / `writer_threads`, and the loops stride by them from `get_my_thread_id()` (`tt_metal/hw/inc/api/kernel_thread_globals.h:155`), so turning the plan is one constant. A DM→DM STRIDED ring with R ≠ 1 or W ≠ 1 is unproven; see the perf lamp |
| Unit dealt to compute threads | none (no compute kernel). The unit dealt to the reader and writer harts is the ring entry itself (one page), so no round padding exists, and ragged per-core counts may stop anywhere |
| Tests the plan needs | (1) A coverage test that shows every engaged core and both kernels ran: every page lands. Use a tensor with more pages than cores in R-bal (`[1,1,1600,256]` TILE) and with ragged groups, so a missing core or thread leaves pages unwritten. Compare against `torch.clone` of a nonzero random input; an unwritten page shows as a mismatch. (2) In **one device session**: run another program first (e.g. `ttnn.add`, or a previous to_memory_config call), then this op, then `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` on a TILE output, checking values each time (target page, *Ring protocols*) |

**Regime-pinned tests the implementer's own test file must carry** (the golden suite sizes input and destination alike, so it cannot reach some of these):

| Test | Pins |
|------|------|
| interleaved DRAM ↔ L1, TILE and RM, ragged shapes (`[1,1,17,50]` RM: 112 B vs 128 B slots) | R-bal; the `entry_bytes = min(...)` cap |
| L1/height → DRAM and → L1 | R-in |
| DRAM / L1 → L1/height | R-out |
| reshard 2 cores → 1 and 1 → 2, TILE `[1,1,64,512]` and RM `[1,4,4,64]` (R6) | R-in (2→1) and R-out (1→2) |
| a COL_MAJOR-orientation shard config, and a height shard whose last shard is partial (e.g. `[1,1,96,64]` TILE, shard `[64,64]` on 2 cores) | shard order; partial-shard ranges |
| equal-config L1/height → L1/height | R-in, all-local path (R7: a copy, one dispatch) |
| `RING_BYTES_CAP` monkeypatched to 16384 with `[1,1,1600,256]` TILE in R-bal (13 pages per core, above the single-block cap of 8) | the depth-2, multi-block path (`block_pages = 4`, 4 blocks, ragged last) |
| `memory_config="L1"`, `None`, a width-sharded config, a width-sharded input | X1, X2 (TypeError), refusals by name (R3, R5) |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `dfb_pages` | 0 (`unique_id="dfb_pages"`) | `entry_bytes = min(input.buffer_aligned_page_size(), output.buffer_aligned_page_size())` | `block_pages * depth` | Spans `page` (`block_pages * depth`); streams over the core's range. Leading dims, H and W are folded into `page`. RM W lives inside an entry. Per compute thread: n/a (no compute kernel, C = 0) | `data_format_metadata = tensor.dtype` (bf16; optional for a DM-only ring per `dfb.cpp:272-281`, set for float32 readiness) | reader | writer | reader side: implicit (TXN_ID tagged reads). Writer side: implicit (TXN_ID tagged writes). No `disable_dfb_implicit_sync_for` on either kernel | the whole program |

It is the only buffer. **Inventory check:** a pull straight into the destination, with no ring, is possible only where the destination is the core's own L1, which is R-borrow (deferred above). A DM self-loop DFB, one kernel producing and consuming, is forbidden on this part (`tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp:646-652`), so the reader and writer are two kernels sharing one ring. There is no L1 budget solve beyond the ring's single cap.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_pages` pages (last: `last_block_extent`) | no (dataflow) | input tensor via its accessor | `dfb_pages`, `block_pages` entries posted by the DM0 ISR | entries posted, awaiting the writer |
| 2 | `store_block` | same | no (dataflow) | `dfb_pages`, `block_pages` entries | output tensor via its accessor | entries acked by the DM0 ISR, free for the reader |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(src, DataflowBuffer&, src_args, dst_args)` | `tt_metal/hw/inc/api/dataflow/noc.h:913-926`; transfer size = entry size `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:775-817` | `src = TensorAccessor(tensor::input)`, `{.page_id = start_page + (block_idx*block_pages + j)*page_stride}`, `{}` | input tensor | `dfb_pages` | `block_pages` (host → ring size), `page_stride`, R |
| `store_block` | raw_api | `Noc::async_write<NocOptions::TXN_ID>(DataflowBuffer&, dst, src_args, dst_args)` | `noc.h:928-944`; `dataflow_buffer.inl:821-882` | `dst = TensorAccessor(tensor::output)`, `{}`, `{.page_id = ...}` | `dfb_pages` | output tensor | `block_pages`, `page_stride`, W |
| accessor | raw_api | `TensorAccessor(tensor::x)`: interleaved and sharded distributions deduced from the host's `TensorParameter` args | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:686-718`, 253-262; host args `tt_metal/impl/metal2_host_api/program_spec/construction/resource/tensor.cpp:57-207` | `TensorBinding(tensor_parameter_name=..., accessor_name="input"/"output")` | — | — | — |
| kernel exit | raw_api | `noc_async_read_set_trid(0, noc)` / `noc_async_write_set_trid(0, noc)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558`, `:2579` | trid 0 | — | — | — |
| host program | raw_api | `ttnn.experimental.DataflowBufferSpec`, `KernelSpec(num_threads=…)`, `create_reader_datamovement_config` / `create_writer_datamovement_config`, `WorkUnitSpec`, `ProgramSpec`, `ProgramRunArgs`, `ttnn.generic_op` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:300` (DFB), `:470`, `:476`, `:567` (KernelSpec), `:841`, `:864`; pattern in `toy_thread_plan_program_artifacts.py` | DM-only work unit: allowed (`validation/placement/work_unit.cpp:195-207`); DM producer + DM consumer: allowed (`validation/resource/dfb.cpp:515-526`, 584-620) | — | — | `block_pages`, depth, core table |
| work split | raw_api | `ttnn.split_work_to_cores(grid, num_pages, row_wise=True)`; `ttnn.corerange_to_cores(grid, None, row_wise)` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:471`; `ttnn/cpp/ttnn-nanobind/tensor.cpp:660` | — | — | — | core-assignment |

**Helpers considered and rejected:**

| Candidate | Why it cannot serve | Citation |
|-----------|---------------------|----------|
| compute helpers (`eltwise_convenience.hpp` `copy`, `tilize_helpers.hpp`, `untilize_helpers.hpp`) | The op has no compute phase. A compute copy would unpack and repack every page through DEST, which adds a NEO pass and, for float32, a precision hazard (R1). Layout never changes, so tilize and untilize do not apply | spec RULES R1, GUIDANCE "No compute" |
| `mcast/kernel/mcast_args.hpp` (`McastArgs`) | No operand is shared across cores; every page goes to exactly one destination | Operand-reuse check above |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A DM→DM ring with implicit sync on **both** ends has no precedent among tt-metal's Quasar ops (all DM-only ones use explicit sync) | The host validation allows it (`dfb.cpp:298-333`, per-side opt-out), but it is unexercised | Phase 0 uses it, as the target page's default protocol. If it hangs, the fallback is explicit sync on the writer side only: list `dfb_pages` in the writer's `disable_dfb_implicit_sync_for`, use `wait_front(1)`, then a write tagged `NocOptions::TXN_ID` `{.trid = 0}`, then a barrier on trid 0, then `pop_front(1)`. Every transfer of that kernel is tagged (target page, *Ring protocols*). Opt-out is per side, so the reader stays implicit |
| `entry_bytes` larger than one side's page slot | For ragged row-major cells the DRAM slot (128 B) exceeds the L1 slot (112 B). A 128 B transfer into L1 tramples the next row | `entry_bytes = min(aligned sizes)` (Mechanism caps). The test `[1,1,17,50]` RM DRAM ↔ L1 pins it |
| Sticky transaction tag | An implicit-sync kernel leaves a ring's transaction id in the hart's tag register | Both kernels end with the `set_trid(0, …)` hand-back. The one-session test runs `ttnn.to_layout` afterwards |
| Partial last batch credit | The reader's last batch, if short of a txn-id threshold, is posted only when the reader returns (`DataflowBuffer.md` "Thresholds") | Harmless: the reader returns right after issuing. Do not add a barrier or wait to the reader that would delay its return |
| Sharded input not on the device worker grid, or a caller config with shard width ≠ padded W | `ttnn.TensorSpec` / allocation raises | Let the allocation error propagate (a malformed config); there is no silent fallback |
| Owner assignment that does not match the accessor's bank order (e.g. orientation misread) | It costs traffic only, never correctness (page ids are global) | A COL_MAJOR test; conformance reads the core table off the host program |
| L1 tensors and the ring compete for the same L1 | A 1-core shard of a large tensor, plus its output, plus the ring | `RING_BYTES_CAP` = 256 KiB bounds the ring; INPUTS' largest shard is 256 KiB |

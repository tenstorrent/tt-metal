# Operation Design: slice_write

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel: the op moves values and computes nothing, spec DEFINITION) |
| Goal | Write a row-major source into the block `start..end` of a caller-owned row-major destination in DRAM, in place, and return that destination. Every element outside the block keeps its value. |
| Math | `output[start[0]:end[0], start[1]:end[1], start[2]:end[2], start[3]:end[3]] = input` (rank 4, step 1) |
| References | spec `eval/specs/slice_write/spec.py` (authoritative); target `.claude/references/targets/quasar.md`; backend `.claude/references/backends/ttsim.md`; host-program examples `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/`); ring semantics `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A; sharded-source Metal 2.0 precedent (API pattern only): `ttnn/cpp/ttnn/operations/experimental/quasar/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp` (borrowed DFB + fake push + `TensorAccessor` writer); `ttnn/cpp/ttnn-nanobind/program_specs.cpp:28-80` (`dfb_spec_from_sharded_tensor`); perf catalog `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `trid_double_issue`, `split_reader`, `read_coalescing`, `bank_stagger`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (fixed on this part); the op queries `device.compute_with_storage_grid_size()` only for the deferred interleaved regime. The sharded regime runs on the source's shard grid. | target page |
| L1 per core | 4,194,304 B (4 MiB) per cluster | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`. The target page does not record the value it returns (`unknown` here). Phase 0 allocates **0 B** of it: its only buffer is borrowed onto the resident shard. | target page (query column) |
| Host program | `ttnn.ProgramSpec` (Metal 2.0) dispatched by `ttnn.generic_op(io_tensors, spec, run_args, tensor_args)`. Reference: `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4} when a compute kernel exists), 6 user DM harts (R + W ≤ 6), and R and W must each divide C on STRIDED rings that have compute. 32 DFBs per cluster. Ring capacity counter is 16 bits (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:1589-1595`). User NoC transaction ids `[0, 7]` (`tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h:17`). | target page + cited lines |
| Unavailable here | (1) `ttnn.ProgramDescriptor` / Metal 1 kernels: refused at kernel creation (target page). (2) The legacy C++ device operation with a program factory: same refusal (target page). (3) `bfloat8_b`, `bfloat4_b`: no path on this part (not in this op's TARGET anyway). (4) A second NoC / dynamic NoC: Quasar has one NoC (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc_nonblocking_api_v1.h:556`), so the catalog's "reads on NoC0 / writes on NoC1" lever does not exist. (5) `NocOptions::TXN_ID` on stateful writes (`set_async_write_state` / `async_write_with_state`): static_assert (`tt_metal/hw/inc/api/dataflow/noc.h:545-568`). (6) Untagged NoC transfers in a kernel whose ring is explicit: not used (every transfer is tagged trid 0). | target page, cited lines |

The worked numbers below are for this target (32 clusters, 4 MiB). Every run-time predicate queries the device or the tensor.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 4, ROW_MAJOR, shape `end - start`. Phase 0: bfloat16, L1 height-sharded | — | tensor parameter `src` (borrowed DFB) |
| `output_tensor` | `ttnn.Tensor` | yes | rank 4, ROW_MAJOR, DRAM interleaved, same dtype and layout as the source (R8) | — | tensor parameter `dst` (TensorAccessor) |
| `start` | list or tuple of int | yes | `len == 4`; `0 <= start[d] <= end[d]` | — | host-derived RT args (dest page map) |
| `end` | list or tuple of int | yes | `len == 4`; `end[d] <= output.shape[d]`. Phase 0: `start[2:] == (0, 0)` and `end[2:] == output.shape[2:]` (`outer`) | — | host-derived RT args |
| `step` | list or tuple of int | yes | all ones (axis `step` = `"ones"` via the spec's `canonical_step`) | — | not passed to kernels |

Argument handling in the entry point, in this order:
1. Normalize `start` / `end` / `step` from list or tuple to tuples of int (R5).
2. Raise the REJECTIONS with `ValueError`: X1 (length ≠ destination rank), X2 (any `start[d] < 0`, `end[d] > output.shape[d]`, or `end[d] < start[d]`), X3 (`input.shape[d] != end[d] - start[d]`).
3. Call `validate()`. It raises `NotImplementedError` outside SUPPORTED (R9). The axes it gates on:

| Axis | How it is read |
|---|---|
| `step` | `canonical_step(step, …)` from the spec |
| `memory_config`, `sharding` | `eval.memory.memory_axes(input_tensor)` (R7) |
| `dtype`, `layout` | the source's own |
| `written_region`, `alignment`, `rank` | `AXES` (`INPUT_TAGGERS = AXES`) on `(input.shape, output.shape, start, end)` |

   It also checks the destination (R8): `memory_axes(output_tensor) == {"memory_config": "DRAM", "sharding": "interleaved"}`, the same dtype and layout as the source, else `NotImplementedError`. A zero-volume block cannot be height-sharded, so `validate()` refuses it (not in any cell). Every call that passes validate makes exactly one dispatch.

There are no index axes (`dim`), so there is no sign canonicalization.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[n_s, h_s, w_s, c]` = `end - start`. Phase 0: `w_s == W`, `c == C` (whole `[W, C]` planes). |
| Dtype | Phase 0 bfloat16. TARGET adds float32 and uint32. |
| Layout | ROW_MAJOR. A page is one row of `c` elements. |
| Memory | Phase 0: L1, HEIGHT_SHARDED (any shard grid, either shard orientation, last shard possibly partial). TARGET adds DRAM / L1 interleaved. Read only (R4). |

### Output

| Property | Value |
|----------|-------|
| Shape | the destination's own `[N, H, W, C]`, unchanged |
| Dtype | the destination's own (equal to the source's) |
| Layout | ROW_MAJOR (the destination's own) |
| Memory | DRAM interleaved: the caller's buffer. No tensor is allocated (R3). The entry point returns the `output_tensor` object it was given, not a new handle. |

## Blocking Model

The op is a **page map**. A row-major page is one row of the last dimension, and the block always covers the destination's whole last dimension (`written_region` ∈ {outer, h}; `w` is not in TARGET). So source row `r` is moved, unchanged, to exactly one destination page `dst_page(r)`:

```
(n, h, w)   = (r / (h_s·w_s), (r / w_s) mod h_s, r mod w_s)
dst_page(r) = ((n + start[0])·H + (h + start[1]))·W + (w + start[2])
```

`dst_page` is a bijection onto the block's pages. In Phase 0 (`outer`: `w_s = W`, `start[2] = 0`) it produces `n_s` runs of `h_s·W` consecutive destination pages; under `h` it produces `n_s·h_s` runs of `w_s` pages. The store operation walks this map incrementally: one division at the core's first row, then additions. The same kernel serves both values of `written_region`; only the host-side numbers differ.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `src_row` (source dims 0–2 folded: n, h, w; one entry per page) | **independent**: each row goes to its own destination page; no result spans rows | `rows_per_block` | **the whole resident shard**: `rows_this_core` valid rows (allocation `shard_rows` = shard spec height) | host: `shard_spec.shape[0]` (allocation) and per-core `rows_this_core = clamp(total_rows - shard_idx·shard_rows, 0, shard_rows)`. Per-core RT arg `num_rows`; DFB `entry_size` derived from it | **caller-pinned**: shard `i` lives on `corerange_to_cores(shard_grid, None, row_wise = orientation == ROW_MAJOR)[i]`, and that core writes it. Cores whose shard holds no valid rows get no kernels. On the harness's shard configs (`eval/sharding.auto_shard_config`, HEIGHT) this is up to all 32 clusters. | knob-turn: interleaved source → `split_work_to_cores` over the full grid (regime `interleaved_stream`). Scheme-change: spreading beyond the shard grid (regime `spread_beyond_shard_grid`). |
| `row` (dim 3, `c` elements = one page) | **independent**, and it is the transfer unit: a destination page is one row, and the block never narrows it (`w` ∉ TARGET) | `row_bytes` (bytes written per page) | whole row: `c · element_size` (64 B for the traced `c = 32` bf16 rows) | host: `input.logical_shape[-1] * element_size`. The source stride in L1 is a separate quantity, `src_row_stride_bytes` = the DFB entry stride from `dfb_spec_from_sharded_tensor` (shard width × element size; equals `row_bytes` in Phase 0) | not assigned across cores. Splitting a page would multiply transactions and produce sub-page writes, for no gain (regime `split_row` rejected). | none |
| `write_batch` (cadence inside the store walk: rows per write barrier) | not a data axis: the transactions-in-flight window of the store | `rows_per_write_batch` | `min(rows_this_core, WRITE_BATCH_ROWS)`, `WRITE_BATCH_ROWS = 32` | one host constant `WRITE_BATCH_ROWS`, passed as a CT arg | per core | knob-turn (perf lamp) |

There is no **dependent** axis, because the op has no reduction, scan or combine. So the split "cut the dependent axis and combine" does not exist for this op. There is no **reuse-shared** axis either. The operand-reuse check over the one input: the source does vary along the split axis (each core owns disjoint rows), so no core reads bytes another core reads. The destination is write-only, and its out-of-block pages are never addressed (R2).

Every stage's data has been re-tabled: the op has a single stage, `store_block`, and its data is the source rows themselves. No intermediate is produced. No core collects or coordinates: there is no designated-core role and no idle-while-one-works stage.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `dfb_src_shard` | `src_shard_entries` | **1** entry = the whole shard (`entry_size = shard_rows · src_row_stride_bytes`) | The ring is borrowed onto the resident shard, so the data is already there. Depth buys nothing, and one entry is the coarsest block and one credit handshake per core. It also keeps the 16-bit ring capacity cap irrelevant (see Mechanism caps). |
| (store window) | `rows_per_write_batch` | 32 rows | NoC writes in flight between barriers on user transaction id 0. Larger values keep more writes outstanding (catalog `double_buffer`: never write-one/barrier). Bounded by the per-id outstanding counter (Mechanism caps). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DFB tile-counter capacity is 16 bits (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:1589-1595`; `num_entries` is narrowed to `uint16_t` at `:882`) | entries of `dfb_src_shard`: one entry per row would cap a shard at 65,535 rows | one entry per shard (`num_entries = 1`, `entry_size = shard_rows · src_row_stride_bytes`), so the cap never binds | `TT_FATAL` at program build for shards over 65,535 rows if the ring were one-entry-per-row |
| Borrowed DFB must fit in the shard (`program_specs.cpp:69`) | `entry_size · num_entries` | `entry_size` is built from the shard's own allocation height `shard_rows` and stride, never from the valid-row count or a rounded value | build-time error |
| Per-transaction-id outstanding-write counter (`tt_metal/hw/inc/api/dataflow/noc.h:422`); the counter width is not documented in tree | `rows_per_write_batch` (writes outstanding on trid 0) | `rows_per_write_batch = min(rows_this_core − done, WRITE_BATCH_ROWS)` with `WRITE_BATCH_ROWS = 32`, one barrier per batch | — |
| One NoC packet per row: `NOC_MAX_BURST_SIZE = 256 words × 256 B = 64 KiB` (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:394-396`) | `row_bytes` | holds for every TARGET row: the largest is 8192 × 4 B = 32 KiB. `validate()` needs no clamp; `async_write` splits longer rows by itself (`ncrisc_noc_fast_write_any_len`, `noc.h:434`) | a row spans several packets, so the batch counts several outstanding transactions per row (the cap above shrinks proportionally) |
| User transaction ids are `[0, USER_TXN_ID_MAX = 7]` (`dataflow_buffer_config.h:17-18`) | trids the writer may use | the writer uses trid 0 only (Phase 0). The double-issue lamp may use 0 and 1 | an id ≥ 8 collides with the runtime's DFB ids |
| Explicit-sync ring kernels must tag every transfer, and an implicit-sync predecessor leaves a ring's tag sticky (target page "Ring protocols") | every NoC write of the writer | `async_write<NocOptions::TXN_ID>(…, {.trid = 0})` and `async_write_barrier<NocOptions::TXN_ID>({.trid = 0})`; last statement `noc_async_write_set_trid(0, noc.get_noc_id())` | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `shard_owner_writes` | **built** (Phase 0) | `memory_axes(input) == {L1, height}` (and SUPPORTED: bf16, `outer`, aligned W) | `rows_per_block = rows_this_core` (the whole resident shard) × `row_bytes` | **Minimum** against the L1→DRAM boundary: each block byte crosses the NoC once (L1 shard → DRAM page); 0 source DRAM reads; 0 destination reads; 0 out-of-block destination bytes; 0 cross-core traffic. Write transactions = one per source row, which is the minimum for an interleaved row-major destination: consecutive pages sit in different banks, so no two rows of a run are address-contiguous. | Fixed per-block cost paid **once per core**: one `push_back(1)` / `wait_front(1)` / `pop_front(1)` handshake, one tag set, one final barrier, kernel launch. Per batch: one trid barrier per `rows_per_write_batch` rows. No inits (no compute). Covers every Phase-0 shape. No other regime applies there: the source is resident, so any other regime adds bytes (rows below). |
| `interleaved_stream` | **deferred**: the spec's PHASE0 excludes interleaved sources (`memory_config` L1 only, `sharding` height only); TARGET has them (R7) | `memory_axes(input).sharding == "interleaved"` (DRAM or L1) | `rows_per_block` = coarsest that fits `2 · rows_per_block · src_row_stride_bytes ≤ budget` and ≤ the core's share; Phase 0 of that regime: whole share if it fits | Minimum for a non-resident source: each source byte crosses DRAM (or the L1-bank NoC) once on read; each block byte is written once. 0 cross-core traffic. | Reader: `reserve_back` / one barrier / `push_back` per block. Writer: `wait_front` / `pop_front` per block, as in the built regime. **Reachable**: the writer is unchanged (it already consumes "a block of rows at a ring entry, starting at source row `first_row`"). Add a reader kernel that fills `dfb_src_rows` (allocated, depth 2) through `TensorAccessor(tensor::src)`, explicit sync, tagged trid 0. The host split is `ttnn.split_work_to_cores(grid, total_rows, row_wise=True)` over `device.compute_with_storage_grid_size()`. |
| `spread_beyond_shard_grid` | **deferred**: the traffic ranking is binding, and this regime adds one NoC crossing of every relocated byte to a built regime that already moves the minimum and covers every shape. Occupancy alone does not override that (`blocking-model.md` §4); the harness's shard configs already spread over up to 32 clusters. | would select when `cores(shard grid with rows) < grid cores` and a measurement shows per-core write issue bounding the op (e.g. the traced single-shard ResNet call: 6785 rows on 1 cluster) | helper core `k` takes a contiguous row range `[a_k, b_k)` of an owner's shard | +1 L1→L1 NoC crossing of relocated bytes (as **one coalesced read per range**: shard rows are contiguous in L1, catalog `read_coalescing`), +1 staging ring per helper. Saves no bytes; spreads write issue. | **Reachable**: the helper's writer is the built writer unchanged (source = a staging ring instead of the borrowed one; `first_row` / `num_rows` RT args). The helper's reader uses `dataflow_kernel_lib::local_addr` / `async_read` from the owner (`ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp`). |
| `bank_coalesced_writes` | **deferred**: the built regime covers every shape it would serve. This regime only trades local bytes for fewer DRAM transactions, and it relies on the interleave function (page → bank, bank offset) that `TensorAccessor`'s per-page contract does not expose | would select when `row_bytes == align(row_bytes, dram_alignment)` and the run length ≫ number of banks | gather the rows of one bank (pages `p, p+B, p+2B…`, contiguous inside bank `p mod B`) into a staging entry; one write per gather | +1 core-local copy of every byte, +1 staging ring; DRAM write transactions ÷ ≈ run length / B | Reachable: it replaces only the inner per-row write of `store_block` |
| `accessor_reread_of_resident_shard` | **rejected**: superseded by `shard_owner_writes` (zero-copy borrowed DFB). A dead end for a sharded source: the same code as `interleaved_stream`'s reader, but it re-reads bytes that are already resident | — | — | every byte crosses the NoC twice (loopback read + write) plus a staging ring | — |
| `split_row` (cut a page across cores or into sub-writes) | **rejected**: the page is the destination's unit and the block never narrows it. Splitting multiplies transactions and adds sub-page DRAM writes | — | — | same bytes, more and smaller transactions | — |
| `dest_read_modify_write` (read the whole destination, merge, write it back) | **rejected**: superseded by the page map, which never touches out-of-block pages (R2) | — | — | +2 DRAM crossings of the whole destination | — |

Regime selection (host, exact): `sharding == "height" and memory_config == "L1"` → `shard_owner_writes`. Anything else is refused by `validate()` in Phase 0. When `interleaved_stream` is built, `sharding == "interleaved"` selects it. Its tests pin both regimes with the same shapes, so each regime's cells exist on every grid.

### Traffic ranking

Units: bytes crossing each tier, per call. `B` = block bytes = `n_s·h_s·w_s·row_bytes`.

| Rank | Candidate split | DRAM | Cross-core NoC | Core-local | Verdict |
|---|---|---|---|---|---|
| 1 | Cut `src_row` along the source's shards (owner writes its own rows) | `B` written once; 0 read | 0 | 0 (the borrowed ring is the shard) | **chosen**: the minimum on every tier |
| 2 | Cut `src_row` evenly over the full grid, helpers pull from owners | `B` written once | up to `B` (relocated share) | staging | deferred (`spread_beyond_shard_grid`): more bytes, better occupancy only when the shard grid is narrow |
| 3 | Owner re-reads its shard through a `TensorAccessor` | `B` | `B` (loopback) | staging | rejected |
| 4 | Cut `row` across cores | `B` | 0 | 0 | rejected: transaction count × k, sub-page writes |
| — | Cut a dependent axis + combine | — | — | — | does not exist: no dependent axis |

The independent axis is cut exactly where the caller's shard cuts it. Doing that is also the cheapest-traffic split, so here occupancy and traffic do not conflict.

### Block schedule

```cpp
// per core (one block: the whole resident shard)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // num_blocks_this_core = 1 (sharded)
    load_block(block_idx);    // reader: publish the resident shard as one ring entry (no data moves)
    store_block(block_idx);   // writer: write every valid row of the entry to dst_page(first_row + i)
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `load_block` | 1 entry = `shard_rows × src_row_stride_bytes` (the shard allocation) | the shard (borrowed; it is the input tensor's own L1) | one `push_back(1)` per core per call. No NoC transfer, no barrier. |
| `store_block` | `rows_this_core` rows × `row_bytes` | the shard entry (held between one `wait_front(1)` and one `pop_front(1)`) | one `wait_front(1)` per core. One division to get `(n, h, w)` of `first_row`, then an incremental walk with no per-row division. One `async_write<TXN_ID>` per row (trid 0), one `async_write_barrier<TXN_ID>` per `rows_per_write_batch` rows (and one at the end). One `pop_front(1)`. Trid hand-back once at kernel exit. |

A core with `rows_this_core == 0` is not in the work unit. A ragged last shard keeps the nominal push/wait/pop of 1 entry and writes only `rows_this_core` rows.

**Stall-shadow check.** The only stage that waits is the write barrier. Everything independent of it is the rest of the store: every row goes to a disjoint destination page, so writes commute and any order is legal (the math grants this; there is no floating-point order to preserve). The design fills the barrier window by issuing the next batch before retiring the previous one (perf lamp `trid_double_issue`). Phase 0 keeps one trid with a barrier per batch: the simple form, which the lamp measures against.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Write batch depth (`WRITE_BATCH_ROWS = 32`) | 64 B rows are latency-bound transactions; 32 in flight is 2 KiB, which may not cover NoC→DRAM round-trip latency | 8 / 32 / 128 rows per barrier (128 only once the trid counter width is confirmed ≥ 8 bits), and the two-trid double issue (batch k on trid `k mod 2`, retire `k-1` before issuing `k+1`; catalog `trid_double_issue`). Structural count on ttsim: barriers per core. |
| Writer thread count (W = 1) | With 64 B rows the cost per row is the hart's command-issue work, not bytes. One hart may be issue-bound (catalog `split_reader`), and 4 user harts are idle | W = 2: `dfb_src_shard` cut into W entries of `shard_rows / W` rows (requires `W | shard_rows`, else W = 1), STRIDED consumer with W writer threads, thread `t` storing entry `t`. Structural count: rows issued per hart. |
| Stateful writes | The per-row write reprograms the whole command buffer, but only the destination address changes between rows | `set_async_write_state` once per batch + `async_write_with_state` per row, after `noc_async_write_set_trid(0)` so untagged state writes carry user id 0 (TXN_ID is not allowed on stateful writes, `noc.h:545-568`). Barrier with the untagged `async_write_barrier`. |
| Bank stagger | All cores start their walk at their first row, and with 2 DRAM channels neighbouring cores may hit the same bank at once | Start each core's walk at row offset `core_idx mod num_banks` and wrap (catalog `bank_stagger`). |
| Grid synchronization | The core count is the caller's shard grid. Small shards (e.g. 32 rows per core) leave launch cost dominant | Nothing collective to cut. The alternative is the `spread_beyond_shard_grid` regime in reverse (fewer cores), which the caller controls through its shard spec. Recorded, not actionable inside the op. |
| Overlap | Not applicable: no compute to overlap with. The only overlap is write issue vs. barrier, covered by the first lamp | — |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Source in L1 | row-major rows, stride `src_row_stride_bytes` in the shard | resident (caller placed it) | never re-read over the NoC |
| Publish | one DFB entry over the shard | `dfb_src_shard` borrowed from tensor parameter `src` (`DataflowBufferSpec(entry_size = shard_rows·src_row_stride_bytes, num_entries = 1, data_format_metadata = input.dtype, borrowed_from = "src")`); reader `push_back(1)` | zero-copy: the DFB *is* the shard (`ttnn.experimental.dfb_spec_from_sharded_tensor` gives `src_row_stride_bytes` and `shard_rows` as its `entry_size` / `num_entries`; the op then builds the one-entry spec from them) |
| Store | row `i` of the entry → destination page `dst_page(first_row + i)` | writer: `noc.async_write<NocOptions::TXN_ID>(dfb_src_shard, dst_accessor, row_bytes, {.offset_bytes = i·src_row_stride_bytes}, {.page_id = dst_page}, {.trid = 0})` | `dst_accessor = TensorAccessor(tensor::dst)`, DRAM interleaved row-major. Only block pages are addressed. |
| Destination | the caller's DRAM buffer | in place | returned as the same `output_tensor` object |

Placement axis as regimes (TARGET `memory_config` × `sharding`):

| Placement | Cost class | Why |
|---|---|---|
| L1 height-sharded | built | the scheme's own split. Shard grid = core assignment, shard = block |
| L1 interleaved | knob-turn (new reader kernel + host split; writer and page map unchanged) | rows are independent, so no combine. Regime `interleaved_stream` |
| DRAM interleaved | knob-turn (same as L1 interleaved; the accessor resolves the bank) | regime `interleaved_stream` |
| DRAM height-sharded | INVALID in the spec | — |

Other TARGET axes:

| Axis value | Cost class | Why |
|---|---|---|
| float32, uint32 | knob-turn (validate-only) | `row_bytes` and the stride scale with element size. No value passes through another format (no compute, no DEST), so R1 holds by construction. |
| `written_region = h` | knob-turn (validate-only) | the page map already handles `w_s < W` and `start[2] > 0` |
| `alignment = w_non_aligned` | knob-turn, with a risk to retire | `row_bytes = c·elem` is not a multiple of 16 B (e.g. 34 B), while the source stride is the 16 B-aligned shard width. A sub-16 B-multiple NoC write into a 64 B-aligned DRAM page must be confirmed on ttsim (Key Risks). |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block = one core's whole resident shard (`rows_this_core` rows) |
| Grid | the source's shard grid: `input.memory_config().shard_spec.grid`, ordered by `ttnn.corerange_to_cores(grid, None, row_wise = (orientation == ttnn.ShardOrientation.ROW_MAJOR))`. Shard `i` → `cores[i]`. Work unit `target_nodes` = the cores with `rows_this_core > 0`. |
| Per-core work | `first_row = i·shard_rows`, `rows_this_core = min(shard_rows, total_rows − first_row)`, `total_rows = n_s·h_s·w_s` (logical rows; a row-major shard has height granule 1, so there are no padding rows inside a shard, only past the last valid row of the last shard) |
| Remainder | the last shard may be partial: handled by `rows_this_core`. Push/wait/pop stay at 1 entry. Shards past `total_rows` (a provisioned grid wider than the data) are excluded from the work unit. |
| Worked example (this target) | Traced ResNet call 1: `total_rows = 59·115 = 6785` (434,240 B bf16). Golden-suite shard config (`auto_shard_config`, HEIGHT, 32 clusters): `shard_rows = ceil(6785/32) = 213`, 32 cores, last core 182 rows → 213 writes and 7 barriers per core. Caller's own config (one shard on one core): 6785 writes and 213 barriers on one cluster (see `spread_beyond_shard_grid`). Traced call 2: 1024 rows → 32 rows on each of 32 cores. |
| Bound | **data movement**: zero compute per row against one NoC write of `row_bytes` per row. The op computes nothing (spec DEFINITION). |
| Thread plan | **(R, C, W) = (1, 0, 1)**, one host constant `THREAD_PLAN = (1, 0, 1)`. R = 1: the reader publishes the shard (one `push_back`). C = 0: no compute kernel exists (nothing to compute, no compute config per the spec). W = 1: one writer hart issues the row writes. This is the target page's plan for an op bound by data movement. R + W = 2 ≤ 6. The counts go to the kernels as CT args `reader_threads`, `writer_threads`; the writer's row loop strides by `writer_threads` from `get_my_thread_id()`, so a missing thread leaves rows unwritten. |
| Unit dealt to compute threads | none (C = 0). Dealt to writer threads: the ring entry. With W = 1 that is the whole shard; the W = 2 lamp cuts it into W entries. |
| Tests the plan needs | (1) *Every thread ran*: the golden check covers every destination page of the block against a random (non-zero) destination, so a missing writer thread leaves rows stale and fails. A test also reads the plan off the built `ProgramSpec` (kernels `reader` / `writer` with `num_threads` 1 / 1, no compute kernel). (2) *Session*: in one device session, run another program first and then `slice_write`. Then run `slice_write` followed by a tt-metal program that issues NoC transfers on the same cores, e.g. `ttnn.to_layout(output, ttnn.TILE_LAYOUT)` (the destination is already row-major, so a to-row-major is a no-op), and check both results. |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `dfb_src_shard` | 0 | `shard_rows · src_row_stride_bytes` (one entry = the shard allocation) | `src_shard_entries` = 1 | spans `src_row` (whole shard) and `row` (stride); streams over `write_batch`. Borrowed: 0 B allocated | `input.dtype` (metadata only: dataflow-only payload, never through DEST) | `reader` | `writer` | explicit on both sides: `push_back(1)` (reader) / `wait_front(1)` … `pop_front(1)` (writer). Listed in `disable_dfb_implicit_sync_for` on both kernels. Every writer transfer is tagged trid 0. | the whole program |

Host construction:
- **Reader**: `create_reader_datamovement_config()` with `config_2xx = DataMovement2XXConfig(disable_dfb_implicit_sync_for=["dfb_src_shard"])`, `num_threads = R`, `dfb_bindings = [producer_of("dfb_src_shard", "src_shard")]`.
- **Writer**: `create_writer_datamovement_config()` with the same opt-out, `num_threads = W`, `dfb_bindings = [consumer_of("dfb_src_shard", "src_shard")]`, `tensor_bindings = [TensorBinding("dst", "dst")]`.
- **Tensor parameters** `src` and `dst`; `tensor_args = {"src": 0, "dst": 1}`; `io_tensors = [input_tensor, output_tensor]`.
- **Run-time arguments**: shape-dependent values (`first_row`, `num_rows`, `row_bytes`, `src_row_stride_bytes`, page-map parameters) are run-time arguments, so one compiled program serves every shape. Plan counts and `WRITE_BATCH_ROWS` are compile-time arguments.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | 1 entry (whole shard) | no (raw DFB API; no data moves) | — (the shard is already resident) | `dfb_src_shard`, 1 entry pushed | 1 entry visible to the writer |
| 2 | `store_block` | `rows_this_core × row_bytes` | no (raw NoC API) | `dfb_src_shard`, wait 1, read at `offset = i·src_row_stride_bytes` | destination tensor pages via `TensorAccessor(tensor::dst)` | entry popped after the final barrier; ring balanced (required at kernel exit, `DataflowBuffer.md` "When the kernel returns") |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `DataflowBuffer::push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:244` | `push_back(src_shard_entries)` | — | `dfb_src_shard` | `src_shard_entries` |
| `store_block` (credit) | raw_api | `DataflowBuffer::wait_front`, `pop_front` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:245-246` | `(src_shard_entries)` | `dfb_src_shard` | — | `src_shard_entries` |
| `store_block` (row write) | raw_api | `Noc::async_write<NocOptions::TXN_ID>` | `tt_metal/hw/inc/api/dataflow/noc.h:405-418` (DFB as source: `noc_traits_t<DataflowBuffer>::src_addr`, `dataflow_buffer.h:581-597`) | `(dfb_src_shard, dst_accessor, row_bytes, {.offset_bytes}, {.page_id}, {.trid = 0})` | `dfb_src_shard` | `dst` tensor | `row_bytes` (row axis) |
| `store_block` (batch barrier) | raw_api | `Noc::async_write_barrier<NocOptions::TXN_ID>` | `tt_metal/hw/inc/api/dataflow/noc.h:760-765` | `({.trid = 0})` every `rows_per_write_batch` rows and once at the end | — | — | `rows_per_write_batch` |
| `store_block` (exit) | raw_api | `noc_async_write_set_trid` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2579` | `(0, noc.get_noc_id())` as the writer's last statement | — | — | — |
| host: shard stride | host API | `ttnn.experimental.dfb_spec_from_sharded_tensor` | `ttnn/cpp/ttnn-nanobind/program_specs.cpp:28-80` | `("dfb_src_shard", input_tensor, borrowed_from="src")` → read `entry_size` (stride) and `num_entries` (shard rows), then build the one-entry `DataflowBufferSpec` | — | — | `src_shard_entries` |

**Helpers considered and rejected** (they apply to every `raw_api` row above, since the op has one data path):
- `compute_kernel_lib::*` (tilize, untilize, reduce, eltwise, matmul): the op has no compute kernel. The spec says it computes nothing and has no compute config.
- `dataflow_kernel_lib` local copy (`ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:11-23`): it covers **L1→L1 self-aimed reads** only. Its header says a DFB cannot be a write destination, and it offers no L1→DRAM write. The store here writes DFB→`TensorAccessor`. It is the helper for the deferred `spread_beyond_shard_grid` and `bank_coalesced_writes` gathers.
- `dataflow_kernel_lib` mcast (`mcast/kernel/mcast_args.hpp`): there is no operand to broadcast.
- `reduce_helpers_dataflow.hpp`, `tilize_helpers_dataflow.hpp`: no scaler and no tilize.

The block operation is "write N rows of a resident entry to N accessor pages", and that is one raw call per row by construction (each row is its own destination page in its own bank).

## Structural impossibilities

None found beyond the spec's INVALID.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| In-place output (R3) | `generic_op` returns a handle, and allocating an output would violate R3 | Pass `[input_tensor, output_tensor]` as `io_tensors`, allocate nothing, and return the caller's `output_tensor` object itself |
| Untagged writes after an implicit-sync program | The writer's ring is explicit, so the hart's sticky tag may belong to a previous program's ring | Tag every write and barrier with trid 0; hand back trid 0 at exit; run the session tests |
| Unbalanced ring at kernel exit | Firmware waits for every DFB to drain; a ragged shard tempts popping `rows_this_core` | Push/wait/pop are always `src_shard_entries` (= 1). Only the write loop uses `rows_this_core`. |
| Source stride ≠ row bytes | The shard width is padded to the L1 granule (`auto_shard_config`: 8 bf16 / 4 fp32 elements), so for `c` not a multiple of that the stride exceeds the row | Two separate values: offset by `src_row_stride_bytes`, write `row_bytes`. They are equal in Phase 0, but the kernel keeps both. |
| Shard order vs. orientation | A COL_MAJOR shard grid orders cores column-first; reading the cores in the wrong order writes rows to the wrong pages | `corerange_to_cores(grid, None, row_wise = orientation == ROW_MAJOR)`, shard `i` → `cores[i]` (the s2i precedent) |
| Per-trid outstanding counter | The width is undocumented | `rows_per_write_batch ≤ 32`; raising it is a lamp gated on confirming the width |
| Sub-16 B-multiple rows (`w_non_aligned`, not Phase 0) | A 34 B write into a 64 B-aligned DRAM page | When that refinement lands, a probe of one 34 B-row cell on ttsim confirms the bytes past the row are untouched. If not, refuse the cell (EXCLUSIONS). |
| Destination pages are tiny (64 B) | Per-transaction cost dominates and per-core issue may bound the op | Batched barriers (Phase 0); W = 2 / stateful / double-issue lamps; `spread_beyond_shard_grid` regime |
| Source unchanged (R4) | A zero-copy ring over the source must never be written | Only the reader publishes (no write); the writer only reads the entry as a NoC source |

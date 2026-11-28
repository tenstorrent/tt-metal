# Operation Design: g6_zeros

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement. The op only writes: it reads nothing and does no arithmetic. |
| Goal | Allocate a new DRAM-interleaved TILE tensor of `shape`/`dtype` on every device of `device`. Write the all-zero bit pattern into every page, in one device program per call, at the DRAM write bound. |
| Math | `out[i0,i1,i2,i3] = +0.0` for every logical index. In bytes, every byte of every page is `0x00`. That pattern is +0.0 in float32 and bfloat16. In bfloat8_b, the zero exponent bytes and the zero sign/mantissa bytes read back as +0.0. One byte pattern serves all three dtypes. |
| Byte-level restatement | Output pages are interleaved over N DRAM banks. Bank `b` holds page `p = b + s·N` at `base + s·aligned_page_size`, for slot `s ∈ [0, slots_b)`. The op therefore writes zero bytes over `slots_b · aligned_page_size` contiguous bytes in each bank. |
| References | `eval/golden_tests/g6_zeros/{feature_spec.py, helpers.py}`.<br>Device zero API: `tt_metal/hw/inc/api/dataflow/noc.h:817-904`, with `internal/tt-1xx/noc_zero_l1.inl` and `internal/noc_zero_dram.inl`.<br>Interleaved contiguity API: `tt_metal/hw/inc/api/tensor/tensor_accessor.h:559-575`.<br>Perf catalog (`ttnn/ttnn/operations/examples/master.md`) entries `double_buffer` (transaction size sets per-core bandwidth), `noc_placement` (DRAM writes on NoC1), `split_reader` (a second RISC pays only when issue-bound), `bank_stagger` (all cores starting on one bank), `read_coalescing` (same-bank coalescing, bank affinity).<br>The exploration bundle `docs/explorations/g6_zeros/` is absent; the design is derived from the spec. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `shape` | 4 ints (list or tuple), positional | yes | every `d ≥ 1`; `d2 % 32 == 0`, `d3 % 32 == 0` | none | Sets `num_pages`, which goes to the host plan: per-core runtime args (`first_page_id`, `core_slots`). Part of the host plan-cache key. |
| `dtype` | `ttnn.DataType`, keyword-only | yes | TARGET: `float32`, `bfloat16`, `bfloat8_b`. Phase 0 SUPPORTED: `float32`. | none | CT: `aligned_page_size` and the CB format tag. |
| `layout` | `ttnn.Layout`, keyword-only | yes | `ttnn.TILE_LAYOUT` | none | Validated only. |
| `device` | `ttnn.MeshDevice` (a single device in tests), keyword-only | yes | open device or mesh device | none | Host: grid size and DRAM bank count go into the plan. |
| `memory_config` | `ttnn.MemoryConfig`, keyword-only | yes | DRAM + INTERLEAVED | none | Validated. CT: `TensorAccessorArgs(output)`. |

### Entry-point contract

| Step | Rule |
|------|------|
| 1 | The first line of `g6_zeros(...)` is `validate(shape, dtype=..., layout=..., device=..., memory_config=...)`. |
| 2a: argument checks, inside `validate()`, before the support gate | Raise `ValueError` when `shape` is not a list/tuple of exactly 4 `int` (not `bool`), when any size is `< 1`, or when `d2 % 32 != 0` or `d3 % 32 != 0`.<br>Raise `ValueError` when `layout != ttnn.TILE_LAYOUT`.<br>Raise `ValueError` when `memory_config.buffer_type != DRAM` or `memory_config.memory_layout != INTERLEAVED`.<br>Raise `ValueError` when `device` is not a ttnn device or mesh device, and `RuntimeError` when it is closed.<br>Every check is host-only and runs before any allocation. |
| 2b: support gate | Build `scenario = {"shape": tuple(int(d) for d in shape), "dtype": dtype, "layout": layout}`. Compute `axes = {name: tagger((scenario,), axes) ...}` over `INPUT_TAGGERS` in declaration order. Check SUPPORTED per axis (raise `UnsupportedAxisValue`), then EXCLUSIONS (raise `ExcludedCell`). Both come from `ttnn.operations._op_contract`. Never check INVALID. |
| 3: registry | `INPUT_TAGGERS = {"dtype": tag_dtype, "layout": tag_layout}`. `tag_dtype` returns `inputs[0]["dtype"]`; `tag_layout` returns `inputs[0]["layout"]`.<br>Phase 0: `SUPPORTED = {"dtype": [ttnn.float32], "layout": [ttnn.TILE_LAYOUT]}` and `EXCLUSIONS = []`.<br>No size, shape or rank axis and no other tagger. |
| 4: output | `ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, memory_config)`. This gives new storage on every call and on every device of a mesh. |
| 5: dispatch | Exactly one `ttnn.generic_op([output], program_descriptor)`. No other ttnn op runs, and no host-made values are transferred. |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | None. The op takes no input tensor. `shape` is an argument. |
| Dtype | — |
| Layout | — |
| Memory | Nothing is read from DRAM. The only reads are core-local L1 loopback reads of the firmware zero region (`MEM_ZEROS_BASE`). |

### Output

| Property | Value |
|----------|-------|
| Shape | `shape`, per device. |
| Dtype | `dtype`. The value is made in `dtype`; there is no cast. |
| Layout | TILE_LAYOUT. One page is one 32×32 tile: 4096 B (fp32), 2048 B (bf16) or 1088 B (bfp8_b). On Blackhole the DRAM alignment is 64 B, so `aligned_page_size == page_size` for all three. |
| Memory | DRAM interleaved, newly allocated per call on every device of `device`. |
| Pages | `num_pages = d0·d1·ceil(d2/32)·ceil(d3/32)`: per image, alignment-aware. The source of truth is `output.buffer_num_pages()`. |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`.

**Derivation.**
- Every output element is +0.0 whatever its index, and the op has no input. Every axis is therefore **independent**: there is no dependent axis to combine and no input operand that could be reuse-shared.
- The four logical tile axes enumerate the interleaved page id `p` contiguously and are treated identically, so they fold into the page axis (`blocking-model.md` §1).
- The DRAM interleave re-indexes `p ↦ (bank b = p mod N, slot s = p div N)`. Slots of one bank are byte-contiguous (`tensor_accessor.h:559-575`), so the blocking is done on `(bank, slot)`.
- A block is one core's contiguous slot run inside one bank.

**Intermediate data.** The scheme adds one stage that produces its own data: the per-core zero source. It has no op axis and is not assignable across cores, because each core's own NoC writes read from its own L1 copy. No core collects, coordinates or hands out work.

**Stall shadow.**
- The scheme has two waits: the L1 read barrier that ends the zero fill, and the final write barrier.
- Nothing is independent of either. Every DRAM write sources the filled region, and the final barrier follows the last write.
- No reorder is designed.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `device` (mesh coordinate; 4 chips in production) | independent: each chip's tensor is written by that chip alone | `device_block` = one whole device tensor | whole | mesh shape (`generic_op` SPMD, `ttnn/cpp/ttnn/operations/generic/generic_op.cpp:13-23`) | one chip per device tensor, the same program on every chip | none: already fully split |
| `d0` | independent: the value is +0.0 for every index | folded into `page` (whole extent) | whole | `output.buffer_num_pages()` | not assigned separately; reaches cores through `bank`/`slot` | none. An extent on `d0` would only divide the page axis, which the interleave scatters over all banks. Splitting `d0` alone caps cores at `d0` (1 for p4): the shape-imposed limit the spec forbids. |
| `d1` | independent: same reason as `d0` | folded into `page` (whole) | whole | `output.buffer_num_pages()` | as `d0` | none, same reason (`d1 = 1` for p1) |
| `tile_row` (`ceil(d2/32)`) | independent: same reason | folded into `page` (whole) | whole | `output.buffer_num_pages()` | as `d0` | none, same reason |
| `tile_col` (`ceil(d3/32)`) | independent: same reason | folded into `page` (whole) | whole | `output.buffer_num_pages()` | as `d0` | none, same reason |
| `page` (`p ∈ [0, num_pages)`) | independent: same reason | re-indexed to `(bank, slot)`, not blocked directly | — | `output.buffer_num_pages()` | through `bank` and `slot` | — |
| `bank` (`b ∈ [0, N)`, N = 8 on BH) | independent: no page of bank `b` depends on another bank | `block_banks` = 1 | 1 | Mechanism cap. One NoC write targets one DRAM endpoint, and a run is byte-contiguous only within one bank. Extent 1 is the cap, not a collapsed knob. | `writers_per_bank` cores per bank. Writer `w` (row-wise rank in the core set) serves bank `w mod N`. | knob-turn: a core owning runs in several banks means `num_blocks_this_core > 1` with the same loop |
| `slot` (`s ∈ [0, slots_b)`) | independent: same reason | `core_slots`: the whole per-core share | the whole assignment, `slots_b` split evenly over `writers_per_bank` | host plan (`_plan()`), the single source; goes to the runtime arg `core_slots` | contiguous slot ranges, balanced base/extra, across bank `b`'s writers | knob-turn (`writers_per_bank`) |

The core count is not an axis knob. `writers_per_bank` is set once in the host plan:
- Phase 0 rule: `writers_per_bank = max(1, min(floor(grid_cores / N), ceil(num_pages / N)))`.
- On this Blackhole: `floor(110 / 8) = 13`, giving **104 writer cores** for every production shape.
- The cores used are this blocking choice. No shape axis limits them.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_zero_source` | Depth: `zero_source_regions` = 1.<br>Capacity: `zero_source_bytes` (host constant, the single source). | 1 region of 16384 B, one Blackhole max NoC packet (`NOC_MAX_BURST_SIZE = 256·64`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:287-290`) | Depth 1 is complete. The source never changes, so a second region buys no overlap: a write never waits on a refill.<br>Capacity sets the largest single write (the per-packet amortization). |
| Writes in flight (not a CB) | Barrier frequency. | One `noc_async_write_barrier` per core per call; every write of the run in flight at once. | No stall between packets. The NIU back-pressures issue on its own (`noc_async_write` waits for the command buffer). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `noc_async_write` source span (`dataflow_api.h:876-884`): a call larger than `NOC_MAX_BURST_SIZE` is split into packets with the **source address advancing** | bytes per write call | `chunk_bytes = min(zero_source_bytes, run_bytes_remaining)`. Each call covers at most the zero region. | Stale L1 bytes past the zero region stream into the output. The dirty-destination case catches it only when those bytes are non-zero. |
| Bank contiguity (`tensor_accessor.h:565-575`): slots are `aligned_page_size` apart only when the accessor's page size equals the interleaved aligned page size | slot run | Construct the output `TensorAccessor` with `output.buffer_aligned_page_size()`. A run never leaves bank `b`. | Writes land at wrong offsets. Some pages stay stale; other bytes of the bank get zeroed. |
| Bank slot count | end of the last run in bank `b` | `slots_b = num_pages div N + (1 if b < num_pages mod N else 0)`. The runs of bank `b`'s writers tile `[0, slots_b)` exactly. | The neighbouring DRAM allocation in that bank is overwritten: silent corruption of another tensor. |
| Interleave modulus | `b = p mod N` | Host `N = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks` (`ttnn/ttnn/device.py:232`). Passed as a CT arg. The kernel `ASSERT`s `N == accessor.contiguous_page_stride()` (`= NUM_DRAM_BANKS`, `tensor_accessor.h:562`). | The host plans runs for the wrong banks. Pages are left unwritten or written twice. |
| DRAM write alignment (64 B on BH, `ttnn.get_dram_alignment()`) | chunk start and size | Every TARGET page size (4096, 2048, 1088) is a multiple of 64. The host asserts `zero_source_bytes % dram_alignment == 0`. | Misaligned DRAM write. |
| L1 zero fill (`noc.h:825-831`): destination 16 B aligned on WH/BH | `fill_bytes` | CB base is aligned. `fill_bytes = min(zero_source_bytes, core_run_bytes)` is a multiple of 64. | `ASSERT` under `--dev`. |
| Model L1 share: 1,347,776 B per core (BB-05) | `zero_source_bytes` | The host asserts `zero_source_bytes ≤ 1,347,776`. Phase 0 is 16384. | Launch failure inside the model only. |
| Grid ≥ banks | `writers_per_bank ≥ 1` | `grid_cores ≥ N` holds on every target device (BH: 110 ≥ 8). | A core would need runs in more than one bank (`num_blocks_this_core > 1`), a knob-turn. |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `bank_run` | built | Every call: single regime.<br>Domain: every TARGET dtype and every shape. All TARGET page sizes are DRAM-aligned, so same-bank slots are always byte-contiguous and a multi-page run is always legal. Production spans 128 to 262,648 pages. | `[bank: 1] × [slot: core_slots]` = `core_slots · aligned_page_size` contiguous bytes in bank `b`. It is written as `ceil(run_bytes / zero_source_bytes)` NoC writes from one unchanging zero source. | **Minimum.**<br>Boundary: each output page crosses DRAM once, as a write. Nothing is read from DRAM.<br>Cross-core traffic: 0.<br>Core-local: one fill of ≤ 16 KiB per core (L1 loopback reads of `MEM_ZEROS_BASE`). | Once per core per call: kernel launch, the zero fill and its L1 read barrier, one accessor address computation per run, and the final write barrier. Per ≤ 16 KiB chunk: NoC command programming, one header flit (0.4 % at 16 KiB against 5.9 % at a 1088 B page), one ack.<br>Per-chunk cost stops falling at `NOC_MAX_BURST_SIZE`.<br>Per-core cost falls with fewer writers per bank, at the price of fewer concurrent streams per bank (the grid-sync lamp). |
| `page_at_a_time` | rejected: superseded by `bank_run` on every shape | — | One page per NoC write. Each core owns a contiguous page-id range, which round-robins over banks. The library's DRAM zero overload `Noc::async_write_zeros(accessor, …)` is this regime: `noc_zero_dram.inl:24` limits a call to one page. | Same DRAM bytes, but packets = `num_pages` (262,648 on p1 against 17,472). Each bank receives one-page writes from every core at scattered offsets. | Nothing. Consecutive page ids sit in different banks, so the block cannot exceed one page. |
| `mem_zeros_direct` | rejected: superseded by `bank_run` | — | Writes sourced straight from the firmware zero region (`MEM_ZEROS_SIZE = 512`, `blackhole/dev_mem_map.h:88`), with no CB. | Same DRAM bytes, but 512 B packets: 32 times the packets of a 16 KiB source. | It saves only 16 KiB of the 1,347,776 B L1 share. |
| `compute_fill` | rejected: superseded by `bank_run` | — | A compute kernel packs zero tiles into a CB and a writer streams pages. | Same DRAM bytes. Adds a compute kernel, a per-tile CB handshake and packer work, all to produce bytes that are dtype-independent zeros. One page per packet. | — |
| `broadcast_zero_source` | rejected: superseded by the per-core local fill | — | One injector fills the zero source and multicasts it to every writer. | Adds a 16 KiB multicast to 103 receivers plus a semaphore handshake (cross-core tier). That replaces a core-local fill of 32 loopback reads. A local fill is never more expensive than any transfer. | — |
| `dependent_split_combine` | rejected: no dependent axis exists | — | — | No combine exists to split. | — |

### Traffic ranking

Figures are per device. Every candidate writes the minimum DRAM bytes (each page once, 0 read), so the ranking turns on transfer shape: packet count and size, and streams per bank.

| Rank | Candidate split | DRAM bytes | Cross-core | Transfer shape | Verdict |
|------|-----------------|------------|------------|----------------|---------|
| 1 | **Bank-major**: `bank` across writer groups, `slot` across a bank's writers | minimum | 0 | Packets ≤ 16 KiB, contiguous: 17,472 (p1), 832 (p2), 416 (p3), 104 (p4). Each bank serves exactly `writers_per_bank` sequential streams for the whole call. | **chosen** |
| 2 | Page-range split plus per-core coalescing into N same-bank runs | minimum | 0 | p1–p3 get about the same packet counts. Every core walks the banks in the same order unless staggered (catalog `bank_stagger`), and p4 has about 1 page per core, which cannot coalesce. | dominated by 1 |
| 3 | Page-range split, one page per packet | minimum | 0 | Packets = `num_pages`; 1088 B packets for bfp8_b; scattered per-bank arrivals. | dominated by 1 |
| 4 | Split `d0`, `d1`, `tile_row` or `tile_col` alone | minimum | 0 | A sub-case of 3, with the core count capped by one axis extent (`d0 = 1` in p4, `d1 = 1` in p1). | rejected: a shape-imposed core limit |
| 5 | Split a dependent axis and combine | — | — | No dependent axis. | not applicable |

Tie-breaker: all candidates split only independent axes. Rank 1 wins on transaction count and on bank-stream regularity, not on bytes.

### Block schedule

```cpp
// The writer kernel is the only kernel of the program. This is the logical schedule of every writer core.
fill_zero_source();                                                              // once per core per call
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // Phase 0: 1 for every writer
    store_bank_run(block_idx);                                                   // no barrier, no CB handshake inside
}
drain_writes();                                                                  // once per core per call
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `fill_zero_source` | `fill_bytes = min(zero_source_bytes, core_run_bytes)` bytes of `cb_zero_source` | nothing before it. Afterwards the zero region stays resident until kernel exit. | 1 zero-fill call (`fill_bytes / 512` loopback reads on BH) and 1 L1 read barrier, per core per call |
| `store_bank_run` | `[bank: 1] × [slot: core_slots]`, giving `run_bytes = core_slots · aligned_page_size` contiguous bytes from `first_page_id`'s address | the zero source, unchanged, as the source of every write | `ceil(run_bytes / zero_source_bytes)` NoC writes. 1 accessor address computation per run. 0 barriers, 0 CB reserve/push/wait/pop, 0 inits. |
| `drain_writes` | every write this core issued | — | 1 `noc_async_write_barrier` per core per call (BB-03) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Grid synchronization**: `writers_per_bank = 13` | p4 leaves 1–2 pages per core, so launch, fill and barrier dominate. On p1, 13 concurrent streams per bank may cost DRAM row locality, and 1–4 writers may already saturate a bank. | `writers_per_bank ∈ {1, 2, 4, 8, 13}` on p1 (perf focus, ≤ 837.19 µs) and on p4 |
| Core↔bank placement | Phase 0 interleaves banks over the row-wise core order, so each bank's 13 writers are spread over the whole grid. Their NoC1 traffic converges on one DRAM endpoint from every row. | Cluster each bank's writers nearest its DRAM endpoint. Seed from `device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_1)`, then grow by distance. |
| Write issue mechanism | Every 16 KiB `noc_async_write` reprograms the full command, though within a run the source, size and destination endpoint never change. | Stateful `noc_async_write_one_packet_set_state` once per run, then `noc_async_write_one_packet_with_state` per chunk (`dataflow_api.h:982, 1008`), with the chunk clamped to `NOC_MAX_BURST_SIZE`. |
| One RISC per core | BRISC alone, on NoC1, issues every packet. | Split each run in halves across BRISC/NoC1 and NCRISC/NoC0. Catalog `split_reader` says this pays only when issue-bound, which 16 KiB packets likely are not, so measure on p1. |
| `zero_source_bytes = 16384` | One BH packet per call. A larger region does not reduce packet count but does reduce call count. | 8192 and 32768 |
| Host dispatch: `custom_program_hash` unset | p4 is the most-called case. Untraced calls hash the descriptor on every call; trace replay skips host work. | Set `custom_program_hash` from the plan key and measure untraced host time on p4. |

The **overlap lamp** does not apply. There is no compute stage. The only serialization is fill before the first write: at most 32 loopback reads of 512 B (2–8 reads for p4's 2–4 KiB runs), against a run of up to 2.7 MB.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Zero source | raw zero bytes in L1 (`cb_zero_source`, `fill_bytes`) | `Noc::async_write_zeros(cb, fill_bytes)` (overload 1) then `write_zeros_l1_barrier()`. On BH these are NoC loopback reads of `MEM_ZEROS_BASE` in 512 B pieces (`noc_zero_l1.inl:13-46`). | Core-local. The region is filled on every call, because other programs reuse that L1 between calls. |
| Store | page bytes of `dtype`, all zero, as one contiguous bank run | `noc_async_write(zero_src, dst, chunk_bytes)`, where `dst = accessor.get_noc_addr(first_page_id)` plus the running byte offset within the bank | NoC1, the preferred NoC for DRAM writes (`tt_metal/api/tt-metalium/kernel_types.hpp:148-153`), via `ttnn.WriterConfigDescriptor()` on BRISC. Non-posted writes. |
| Drain | — | `noc_async_write_barrier()` | Every transaction completes before return (BB-03). |
| Mesh | same program on every chip | `generic_op` with a `ProgramDescriptor` builds an SPMD `MeshProgramDescriptor` over the whole mesh (`generic_op.cpp:13-23`). | The output from `allocate_tensor_on_device` on the mesh has the same address on every chip, so one common runtime arg serves all chips. |

TARGET axes as support growth:
- `dtype`: bf16 and bfp8_b are **knob-turn** unlocks. Only `aligned_page_size` (a CT arg) and the CB format tag change. The bank runs, the plan and the kernel are unchanged, and no new algorithm is needed.
- `layout`: TILE only.
- TARGET has no `memory_layout` axis: the output is DRAM interleaved only, so no physical shard exists to consume in place.

Structural impossibilities: none beyond `feature_spec.py`'s empty INVALID.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block: a bank run of `core_slots` slots in one bank |
| Grid | `device.compute_with_storage_grid_size()`: 11×10 = 110 on this Blackhole |
| Banks | `N = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks`: 8 |
| Writers per bank | `writers_per_bank = max(1, min(floor(grid_cores / N), ceil(num_pages / N)))`: 13 |
| Writer cores | `W = writers_per_bank · N` = 104.<br>Core set: `ttnn.num_cores_to_corerangeset(W, grid, row_wise=True)`, which gives {(0,0)–(10,8), (0,9)–(4,9)}: 2 rectangles (BB-17). |
| Writer → bank | `w` = row-wise rank in the core set. `b = w mod N`, `k = w div N`. |
| Slots per bank | `slots_b = num_pages div N + (1 if b < num_pages mod N else 0)` |
| Per-core run | `base = slots_b div writers_per_bank` and `extra = slots_b mod writers_per_bank`.<br>`core_slots = base + (1 if k < extra else 0)`.<br>`first_slot = k·base + min(k, extra)`.<br>`first_page_id = b + first_slot·N`. |
| Remainder | `num_pages mod N` gives the first banks one extra slot. `slots_b mod writers_per_bank` gives the first writers of a bank one extra slot. A writer with `core_slots = 0` launches and returns; that happens only when `slots_b < writers_per_bank`, which no production case reaches. |
| Blocks per core | `num_blocks_this_core = 1` for every writer (Phase 0), passed as a runtime arg so the loop bound is a parameter |
| Per-call runtime value | Output buffer address: one common runtime arg, the same on every core and every chip. |

| Case | dtype | `num_pages` | `aligned_page_size` | `slots_b` | `core_slots` (writers per bank) | max `run_bytes` | writes/core (max) | `fill_bytes` |
|------|-------|-------------|---------------------|-----------|----------------------------------|-----------------|-------------------|--------------|
| p2 (16,12,128,128) | float32 | 3,072 | 4096 | 384 | 30 (×7), 29 (×6) | 122,880 | 8 | 16,384 |
| p3 (8,12,128,128) | float32 | 1,536 | 4096 | 192 | 15 (×10), 14 (×3) | 61,440 | 4 | 16,384 |
| p1 (32831,1,32,256) | bfloat8_b | 262,648 | 1088 | 32,831 | 2526 (×6), 2525 (×7) | 2,748,288 | 168 | 16,384 |
| p4 (1,4,256,128) | bfloat16 | 128 | 2048 | 16 | 2 (×3), 1 (×10) | 4,096 | 1 | 2,048–4,096 |

**Host plan and caching (BB-07, BB-08, BB-21).**
- `_plan(shape, dtype, device)` is computed once per key `(device identity, tuple(shape), dtype)` and cached in a module dict together with its `ProgramDescriptor`.
- A repeat call does a dictionary lookup, sets the writer kernel's common runtime arg to `output.buffer_address()`, and calls `generic_op`. It never rebuilds the descriptor.
- The plan is reportable through a module function `describe_plan(shape, *, dtype, device) -> dict`. It returns `regime="bank_run"`, `N`, `writers_per_bank`, `zero_source_bytes`, the core ranges, and a per-core role map: `writer(bank, first_page_id, core_slots)` or `idle` for the 6 unused grid cores.
- Program-cache key: `generic_op` hashes structure, not runtime-arg values (`generic_op_device_operation.cpp:78-146`). p2 and p3 therefore share one program and differ only in runtime values, which the cache-hit path re-applies. The cache-order case p3 → p2 → p3 pins this.
- A single regime means no selection predicate.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_zero_source` | 0 | `zero_source_bytes` (16384) | 1 (`zero_source_regions`) | Spans no block axis: it is the NoC source of every chunk of every run, and streams over `slot` in `zero_source_bytes` steps. Capacity is the largest single write. | output `dtype`, as a tag only: no compute kernel ever unpacks or packs it | writer (`fill_zero_source`) | writer (NoC source of `store_bank_run`) | whole kernel invocation |

`cb_zero_source` is a raw scratch region, not a FIFO.
- There is no `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`.
- In a fresh CB the read pointer equals the write pointer. The fill targets that address and every write sources it.
- One kernel owns it end to end.

The full ledger is in `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `fill_zero_source` | `fill_bytes` | helper (device API): `Noc::async_write_zeros` overload 1 and `write_zeros_l1_barrier` | — (source: firmware `MEM_ZEROS_BASE`) | `cb_zero_source`, first `fill_bytes` of its 1 page | zero bytes resident until kernel exit, never popped |
| 2 | `store_bank_run` | `[bank: 1] × [slot: core_slots]` | custom block operation over raw `noc_async_write` (helper gap recorded in API Mapping) | `cb_zero_source`, 1 page, read in place and never popped | DRAM output pages `first_page_id + j·N`, `j < core_slots` | unchanged |
| 3 | `drain_writes` | all writes of the core | raw API | — | — | no NoC transaction outstanding |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `fill_zero_source` | helper | `Noc::async_write_zeros(const Dst&, uint32_t size_bytes, const dst_args_t<Dst>& = {})` (overload 1), then `Noc::write_zeros_l1_barrier()` | decl `tt_metal/hw/inc/api/dataflow/noc.h:848-849, 904`; BH/WH impl `tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl:13-46`; `CircularBuffer` is a legal endpoint at `tt_metal/hw/inc/api/dataflow/circular_buffer.h:259` | `dst` = `CircularBuffer(cb_zero_source)`, `size_bytes = fill_bytes` | — | `cb_zero_source` | `fill_bytes` (from `zero_source_bytes` and `core_slots`) |
| `store_bank_run`: address | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` (interleaved specialization); the contiguity check is `num_contiguous_pages(page_id, end_page_id)` and `contiguous_page_stride()` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:554, 562, 568-575` | `TensorAccessor(TensorAccessorArgs<…>(), output_addr, aligned_page_size)`, `page_id = first_page_id`. Under `--dev`, `ASSERT(num_contiguous_pages(first_page_id, first_page_id + (core_slots-1)·N + 1) == core_slots)`. | — | — | `core_slots`, `first_page_id` |
| `store_bank_run`: writes | raw_api | `noc_async_write(src_l1, dst_noc_addr, size)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:876-884` | `src_l1` = `cb_zero_source` address; `size = min(zero_source_bytes, remaining)`; `dst` advanced by `size` within the bank, the same pattern as `noc_zero_dram.inl:43-48` | `cb_zero_source` | DRAM output | `zero_source_bytes` (chunk), `core_slots` (run) |
| `drain_writes` | raw_api | `noc_async_write_barrier()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1951` | — | — | — | — |

**Helpers considered and rejected.**

| Raw-API entry | Candidate helper | File:Line of the mismatch | Reason it cannot express the planned shape |
|---------------|------------------|---------------------------|---------------------------------------------|
| `store_bank_run` | `Noc::async_write_zeros(accessor, size_bytes, {.page_id, .offset_bytes}, scratch)` (overload 2, DRAM) | `tt_metal/hw/inc/internal/noc_zero_dram.inl:24` `ASSERT(args.offset_bytes + size_bytes <= accessor.get_aligned_page_size())` | A call covers at most one page. The planned block is a multi-page bank run of `core_slots · aligned_page_size` bytes, up to 2,748,288 B on p1. The helper would force the `page_at_a_time` regime: 262,648 packets on p1 against 17,472. The missing overload is named by the TODO at `noc_zero_dram.inl:10-11` ("A page-range overload could zero a bank at a time"). `store_bank_run` is that operation, reusing the helper's burst loop (`:43-48`) over a run instead of a page. |
| `store_bank_run` | `noc_async_write` with the run's full size in one call | `dataflow_api.h:876-884` (the any-length path advances the source) | Its source must be `run_bytes` of zeros: up to 2.7 MB, over the 1,347,776 B L1 share. Chunking against a fixed 16 KiB source is the block operation. |
| `fill_zero_source` (alternative) | `dataflow_kernel_lib::fill_l1_range<4>` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:101` | It would be used rather than rejected if overload 1 were unavailable. It is a RISC-V word-store loop: 4096 stores for 16 KiB, against 32 loopback reads of 512 B. |
| `fill_zero_source` (alternative) | `dataflow_kernel_lib::zero_tile` / `prepare_zero_tile` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:53, 67` | Zeroes exactly `dfb.get_tile_size()` and does reserve/push. The source is 16 KiB, not one tile, and it never cycles, so the handshake would have no consumer. |
| (compute) | tilize / untilize / reduce / eltwise / matmul helpers | — | There is no compute phase. The op does no arithmetic, and the zero bytes are identical in every dtype. |
| (cross-core) | `ttnn.Mcast` / `McastArgs` | — | There is no operand to broadcast (regime `broadcast_zero_source`, rejected). |

**Profiler zones (BB-21).** All zones live in the writer kernel and use `DeviceZoneScopedN` (`tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012`). They compile out of the plain build.

| Zone | Contents |
|------|----------|
| `G6Z-FILL` | issuing the zero fill |
| `G6Z-FILL-WAIT` | the L1 read barrier |
| `G6Z-STORE` | the write-issue loop |
| `G6Z-DRAIN` | the final write barrier |

The role map comes from `describe_plan`.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A run overruns a bank's slot range | A write past `slots_b` zeroes another tensor's DRAM bytes. No golden cell sees it. | `slots_b` is computed per bank with the remainder rule, and bank `b`'s runs tile `[0, slots_b)` exactly. The dev `ASSERT` checks contiguity via `num_contiguous_pages`. |
| Wrong run stride | Bank slots are `aligned_page_size` apart. An accessor built with an unaligned size returns 1-page runs (`tensor_accessor.h:565-571`). | Build the accessor with `output.buffer_aligned_page_size()`, and pass that as a CT arg. |
| Writes issued before the zero fill lands | Stale L1 bytes reach DRAM. On a fresh buffer this can pass by chance. | `write_zeros_l1_barrier()` ends `fill_zero_source` before the first write. Dirty-destination cases poison the range first. |
| Writes larger than the zero region | `noc_async_write` advances the source for multi-packet sizes. | Every call is `≤ zero_source_bytes`. |
| Relying on a zeroed buffer | The model's buffers hold stale data. | Every page is written on every call. Fill is per call. The dirty-destination cases (p2, p3, p4, p1) require the output to land on a poisoned, freed range. |
| -0.0 | The gate compares bit patterns. | Only `0x00` bytes are ever written. No fill value goes through a packer or SFPU. |
| Posted writes or an early return | A write still in flight hangs or corrupts the next program (BB-03). | Non-posted writes. One `noc_async_write_barrier` before return. |
| Using the CB as a FIFO | `push`/`pop` would move the read pointer off the zeroed region. | No CB handshake at all. The source address is the fresh CB's write pointer, which equals its read pointer. |
| Host bank count differs from the device interleave | Pages map to the wrong banks: some unwritten, some written twice. | `N` comes from the allocator's DRAM `MemoryView`. The kernel `ASSERT`s it against `contiguous_page_stride()`. |
| p2 and p3 share one program | Runtime args must be refreshed on a cache hit. | Every shape-dependent value is a runtime arg and every dtype-dependent value is a CT arg. The cache-order test p3 → p2 → p3 pins this. |
| Mesh: writing one chip only | The single-device tests cannot see the other chips. | Output comes from `allocate_tensor_on_device(…, device=mesh)`, dispatched through `generic_op`'s SPMD path over the whole mesh shape. No per-chip loop, no device-0 shortcut. |
| bfp8_b p1 is the perf target, but outside Phase 0 SUPPORTED | p1 (262,648 pages, ≤ 837.19 µs) is where `writers_per_bank` and placement get measured. | The design is identical for all dtypes. The bfp8_b refinement adds `ttnn.bfloat8_b` to SUPPORTED, and the lamps are measured there. |

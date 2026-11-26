# Operation Design: g6_zeros

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (creation: writes only, no input tensor, no arithmetic) |
| Goal | Allocate a new DRAM-interleaved TILE tensor of the given per-device shape and dtype on every device of `device`, and write the all-zero bit pattern into every page, as exactly one device program per call. Nothing is read from DRAM; the zero bytes are made in each core's own L1. |
| Math | `out[i] = +0.0` (all-zero bit pattern of `dtype`) for every logical index `i`; `out` is new storage on every call |
| References | `docs/explorations/g6_zeros/DOSSIER.md`, `tensix.md`, `risks.md`, `lessons.md`; `eval/golden_tests/g6_zeros/feature_spec.py`, `helpers.py`; `.claude/references/blocking-model.md`, `l1-footprint-discipline.md`; `ttnn/ttnn/operations/examples/master.md` (entries below); `tt_metal/hw/inc/api/dataflow/noc.h`, `tt_metal/hw/inc/internal/noc_zero_dram.inl`, `tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl`, `tt_metal/hw/inc/api/tensor/tensor_accessor.h`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h` |

Device facts this design was sized on (probed on this checkout's chip, Blackhole): worker grid 11 x 10 = 110 cores, DRAM grid 8 x 1 = 8 banks, DRAM alignment 64 B, L1 alignment 16 B, `NOC_MAX_BURST_SIZE` = 256 x 64 = 16384 B (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:287-290`). Page size = aligned page size for all three dtypes: 4096 (float32), 2048 (bfloat16), 1088 (bfloat8_b). None of these numbers is a literal in the op: each is queried (see Knob source columns).

### Catalog entries that informed the knobs

| Entry (`ttnn/ttnn/operations/examples/`) | What it measured | Knob it set here |
|---|---|---|
| `read_coalescing` | Coalescing physically contiguous same-bank rows into one wide call: 1.11-3.66x; bank-affine work assignment (each core one bank) after coalescing: 0.91-1.28x | Block = one **bank run** written as wide transfers; core-assignment = bank-affine (Phase 0 default), with the alternative placement as a perf lamp |
| `bank_stagger` | All cores walking banks in the same order contend on one bank (1.04-1.32x from staggering); writes into bank-aligned outputs have the same pattern | Bank-affine assignment gives every bank a fixed, equal set of concurrent writers by construction, so no stagger loop is needed |
| `noc_placement` | Writes on NoC1 / reads on NoC0 is 2.5-4.8x faster than the reverse for spread placements; row-wise core enumeration beats column-wise | Single writer kernel on the `WriterConfigDescriptor` default (RISCV_0, NoC1); cores enumerated row-major |
| `double_buffer` | Batching and buffer depth stop paying once the grid is DRAM-bandwidth-bound (64 cores hit ~DRAM peak untuned) | Source depth 1 (constant contents); all writes issued back-to-back, no per-block barrier |
| `trid_double_issue`, `tile_reorder` | Batch barriers; whole-page / coalesced transfers beat fragmented ones on a DRAM-bound move | One write barrier per kernel; transfers of `zero_chunk_bytes` (up to the burst cap), not one per page |
| `split_reader` | A second data-movement RISC only helps when one RISC is issue-bound | Recorded as a perf lamp, not built |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `shape` (positional) | list or tuple of 4 `int` | yes | each `d_k >= 1`; `d_2 % 32 == 0`, `d_3 % 32 == 0`; `bool` is not an int here | none | host only: sets `P = output.buffer_num_pages()` (common RT) and the per-core ranges (per-core RT) |
| `dtype` (keyword-only) | `ttnn.DataType` | yes | TARGET: `float32`, `bfloat16`, `bfloat8_b`; Phase 0 SUPPORTED: `float32` | none | sets the page size: CT (`page_bytes`, TensorAccessor args) and the `zero_chunk_bytes` CT clamp |
| `layout` (keyword-only) | `ttnn.Layout` | yes | `ttnn.TILE_LAYOUT` | none | host only (a page is one 32x32 tile) |
| `device` (keyword-only) | `ttnn.MeshDevice` (single device = 1x1 mesh) | yes | open device or mesh device | none | host only: output allocated on every device of it; `generic_op` runs the same program on every device (SPMD) |
| `memory_config` (keyword-only) | `ttnn.MemoryConfig` | yes | DRAM interleaved only | none | host only: output placement; TensorAccessor CT args |

No compute configuration exists (no arithmetic). `pin_memory`, `names`, `requires_grad` are not parameters.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | none: the op has no input tensor and reads no tensor data |
| Dtype | n/a |
| Layout | n/a |
| Memory | n/a |

### Output

| Property | Value |
|----------|-------|
| Shape | `shape` (per device), `(d_0, d_1, d_2, d_3)` |
| Dtype | `dtype`; every element stored as the all-zero bit pattern (float32 / bfloat16: +0.0; bfloat8_b: shared exponent 0 and every datum 0x00) |
| Layout | `TILE_LAYOUT`; `P = d_0 * d_1 * ceil(d_2/32) * ceil(d_3/32)` pages, taken from `output.buffer_num_pages()` |
| Memory | DRAM interleaved (`memory_config`), newly allocated by this call on every device of `device` via `ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, device, memory_config)` (positional) |

## Blocking Model

The op has no input, no reduction and no arithmetic. Every page receives the same bytes, so the whole design is: **where each core's pages lie in DRAM, and how few, how large and how well-spread the transfers that write them are.**

### Axes

The four logical axes are folded into the page index (row-major tile order, which is exactly how the interleaved buffer numbers its pages; the op treats them identically). The DRAM interleave then factors the page index into the two axes the blocking acts on: `page = bank_row * B + bank`, with `B` = number of DRAM banks. Within one bank, consecutive `bank_row`s are consecutive addresses (`tensor_accessor.h:560-575`: "within a bank every num_banks'th page id is contiguous"), so a block that is 1 bank wide and `n` rows tall is **one contiguous DRAM region**.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `d_0` | independent: every element is the constant +0.0 and reads nothing | folded into `page` (no knob of its own: identical treatment, contiguous in page order) | folded | `output.buffer_num_pages()` (host) | through `bank` / `bank_row` | none needed: a fold, never split on its own |
| `d_1` | independent: as `d_0` | folded into `page` | folded | as `d_0` | through `bank` / `bank_row` | none needed |
| `tensor_ht_tiles` = `ceil(d_2/32)` | independent: as `d_0` | folded into `page` | folded | as `d_0` | through `bank` / `bank_row` | none needed |
| `tensor_wt_tiles` = `ceil(d_3/32)` | independent: as `d_0` | folded into `page` | folded | as `d_0` | through `bank` / `bank_row` | none needed |
| `page` (the fold, extent `P`) | independent | none of its own: the block is defined in its factored form (`bank`, `bank_row`) | `P` = 1536 (p3) to 262648 (p1) | `output.buffer_num_pages()` (host) -> common RT arg `num_pages` | through `bank` / `bank_row` | none needed |
| `bank` = `page mod B` (extent `B` = 8) | independent: pages of different banks share nothing | `block_extent[bank]` | **1**. Not a collapse: one NoC write addresses one bank endpoint and two banks are not address-contiguous, so a block 2 banks wide is 2 runs that share no fixed cost (mechanism cap, below) | scheme constant: one bank run per block | **bank-affine**: core `c` (row-major index over the worker grid) owns bank `c mod B` | knob-turn (placement is host-only: the kernel takes a bank-major page range, see Work Distribution) |
| `bank_row` = `page div B` (extent `rows_in_bank(b)`) | independent: as `bank` | `block_extent[bank_row]` = `core_bank_rows` | **the core's whole row share of its bank** (coarsest: bounded only by the core's assignment; L1 does not bound it because the L1 source is a constant region re-read for every transfer, independent of the block extent) | host per-core RT arg `range_count` | bank `b`'s rows split contiguously and evenly (+-1 row) over the `W_b` cores with `c mod B == b` | knob-turn |

Re-run on the data the scheme creates: the only produced data is the zero source in L1. It does not vary along any axis (it is **reuse-shared by construction** of every split), it is made once per core, and it stays resident for the whole kernel. It has no axis to block or spread; no stage of the scheme runs on one core while others wait (no collector, no coordinator).

Core-count knob:

| Knob | Phase 0 value | Knob source | Bound |
|---|---|---|---|
| `num_worker_cores` (`W`) | full worker grid, `grid.x * grid.y` (110 here), one rectangle from `(0,0)` | host: `device.compute_with_storage_grid_size()` | `B <= W <= grid.x * grid.y` (mechanism cap below) |

Derived (never restated): `B` from `device.dram_grid_size()` (x * y), asserted equal to the kernel's `NUM_DRAM_BANKS`; `W_b = (W - 1 - b) div B + 1` cores for bank `b`; `rows_in_bank(b) = P div B + (b < P mod B)`.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_zero_source` | none (fixed 1) | 1 page of `zero_chunk_bytes` | Nothing beyond 1: the contents are constant and only read after the one zeroing barrier, so there is no producer/consumer rotation to overlap. Every transfer of the run reads the same bytes. |
| (NoC writes in flight) | not a CB: writes are issued back-to-back with no per-block barrier | unbounded in software; throttled only by NoC command-buffer back-pressure | Maximum transactions in flight per core for the whole run; the single barrier is at kernel end (BB-03). |

Transfer quantum (inside a block, the mechanism's unit, not a block extent):

| Knob | Phase 0 value | Knob source | Effect |
|---|---|---|---|
| `zero_chunk_bytes` (`Z`) | `min(NOC_MAX_BURST_SIZE, max over cores of core_bank_rows * page_bytes)`: 16384 for p1, p2, p3; 4096 for p4 | host, CT arg; also the size of `cb_zero_source`. `NOC_MAX_BURST_SIZE` on the host comes from a per-arch table keyed on `device.arch()` (Blackhole 16384) and the kernel `static_assert`s it | One NoC write per `Z` bytes of a run; zeroing cost `Z / 512` loopback reads once per core. Clamping to the largest run keeps the one-time zeroing proportional to what a small call writes (p4: 8 loopback reads instead of 32). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Single-packet NoC write: at most `NOC_MAX_BURST_SIZE` bytes (`noc_parameters.h:290`; `noc_async_write_one_packet`, `dataflow_api.h:841`) | `zero_chunk_bytes` | host `min(...)`; kernel `static_assert(zero_chunk_bytes <= NOC_MAX_BURST_SIZE)` | a one-packet write past the burst size is malformed: corrupt or missing bytes |
| One write addresses one NoC endpoint (one DRAM bank) | `block_extent[bank]` = 1 | the block op writes one bank run; a core range crossing banks becomes several blocks | bytes of one bank's run land in the wrong bank or wrap |
| Bank-run contiguity holds only when the accessor's page size equals the interleaved aligned page size (`tensor_accessor.h:570-572` returns 1 otherwise) | `block_extent[bank_row]` | construct the accessor with `output.buffer_aligned_page_size()`; host asserts `buffer_page_size() == buffer_aligned_page_size()` (true for 4096 / 2048 / 1088 at 64 B alignment) | a multi-page run written with the wrong stride overwrites neighbouring pages and leaves others stale (R1) |
| Host bank count must equal the device interleave modulus `NUM_DRAM_BANKS` = `accessor.contiguous_page_stride()` (`tensor_accessor.h:562`) | the `bank` / `bank_row` factoring | host passes `B` as a CT arg; kernel `static_assert(dram_bank_count == NUM_DRAM_BANKS)` | ranges computed with the wrong `B` leave pages unwritten (R1) and write others twice |
| Every bank needs at least one core | `num_worker_cores` | host clamps `W` to `[B, grid.x * grid.y]` | the pages of a bank with no core are never written (R1) |
| L1 zeroing is a NoC loopback read: destination 16 B aligned (`noc_zero_l1.inl:22`, `NOC_L1_READ_ALIGNMENT_BYTES` = 16, `noc_parameters.h:375`) | `cb_zero_source` base, `zero_chunk_bytes` | CB base is L1-aligned; `zero_chunk_bytes` is a multiple of 64 (page sizes are) - host asserts `Z % 64 == 0` | ASSERT in `async_write_zeros`; with asserts off, misaligned zeroing |
| Zero source bytes are undefined until `write_zeros_l1_barrier()` returns (`noc.h:825-826`) | ordering, not extent | barrier before the first DRAM write | stale L1 bytes written into the output, different run to run (R2, BB-06) |
| DRAM write alignment 16 B (`noc_parameters.h:380`) | chunk starts | run starts are page-aligned (64 B) and `Z` is a multiple of 64, so every chunk start is 64 B aligned | misaligned DRAM writes |
| Non-posted writes are the only kind the final barrier can wait on | write kind | `posted = false` (the default) on every write | writes still in flight when the kernel returns hang the next program (BB-03) |

### Regimes

The minimum, against the DRAM boundary: **`out` crosses DRAM once, as a write of every page; zero bytes are read from DRAM; no cross-core traffic.** Core-local L1 is the source of every transfer (the NoC engine reads `Z` bytes from `cb_zero_source` per transfer); that is the floor for any DRAM write.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **bank-run zero write**: each core zeroes `Z` bytes of L1 once, then writes its whole share of one bank (`core_bank_rows` contiguous rows) as `ceil(run_bytes / Z)` single-packet writes from that one source, then one write barrier | **built** | every call: any `P >= 1`, all three dtypes, TILE, DRAM interleaved. The block's L1 footprint is `Z` bytes whatever its extent, so there is no fits-in-L1 boundary and no resident/streaming split; no other regime applies anywhere | 1 bank x `core_bank_rows` rows | **minimum**: `P * page_bytes` written once, 0 B read, 0 B cross-core. Transactions: p1 17472, p2 796, p3 440, p4 110 (vs one per page: 262648 / 3072 / 1536 / 128) | Per block (once per core at the Phase 0 extent): one `get_noc_addr` for the run's first page, one partial last transfer. Per kernel (once per core whatever the extent): the `Z / 512` loopback reads + one L1 barrier, one final write barrier. Per transfer (set by `Z`, not by the block): one NoC command issue. A block smaller than the whole share would add address lookups and partial transfers with no offsetting gain. |
| page-at-a-time zero write (`Noc::async_write_zeros` overload 2, one call per page, `noc.h:893`) | **rejected**: superseded by bank-run, which reduces to exactly this when a run is one page | n/a | 1 page | same bytes; `P` transactions of one page each (p1: 262648 writes of 1088 B, ~94M-142M core-cycles of issue across the grid per `tensix.md`) | n/a |
| page-order contiguous split: each core owns `[start, start + n)` in page order and writes it as `B` bank runs | **rejected**: same bytes; superseded by bank-affine, which writes the same pages with 1 run per core instead of `B`, and gives each bank a fixed set of concurrent writers instead of needing a per-core bank stagger to avoid convoys (`bank_stagger`) | n/a | 1 bank x ~`n/B` rows, `B` blocks per core | same bytes; `+ (B - 1)` partial transfers and address lookups per core | n/a |
| hardware zero region as the direct write source (`MEM_ZEROS_BASE`, 512 B, `dev_mem_map.h:88,115`) | **rejected**: superseded by bank-run with `Z` up to 16384 B; it saves only the one-time `Z`-byte zeroing | n/a | as bank-run, transfers capped at 512 B | same bytes; 32x the transactions of `Z` = 16384 | n/a |
| compute-engine fill and pack per page (DEST zero, pack to a CB, writer drains the CB) | **rejected** (dead end): per-page engine work and CB handshakes for bytes a single L1 zeroing makes once; the bfloat8_b pack computes shared exponents (R5) | n/a | 1 page | same DRAM bytes; adds per-page unpack/pack/CB traffic in L1 | n/a |
| zero source multicast from one injector core | **rejected**: adds `(W - 1) * Z` bytes of NoC traffic and a semaphore handshake to replace a core-local loopback of `Z` bytes | n/a | as bank-run | above minimum by `(W - 1) * Z` cross-core bytes | n/a |
| zeros read from a DRAM buffer | **rejected**: forbidden by spec rule 4 (no DRAM read), doubles DRAM traffic | n/a | n/a | `2x` the minimum | n/a |
| split a dependent axis + cross-core combine | **not applicable**: the op has no dependent axis; nothing spans an axis, nothing is combined | n/a | n/a | n/a | n/a |

Single built regime: no selection predicate and no regime-pinned tests are needed. The only shape-dependent quantities (`Z`, the per-core ranges) are values of one regime's knobs.

### Traffic ranking

Every candidate split moves the same bytes across every tier (the op has no input and no dependent axis), so the split ranking is a tie at the minimum, and the decision falls to the fixed cost per core and how evenly the 8 banks are loaded at every instant.

| Candidate split | DRAM writes | DRAM reads | Cross-core bytes | Runs (address lookups) per core | Bank concurrency |
|---|---|---|---|---|---|
| **bank-affine: core `c` -> bank `c mod B`, contiguous row share** (chosen) | `P * page_bytes`, once | 0 | 0 | 1 | fixed: every bank has `W_b` (13-14) writers for the whole call, each bank's writers spread over the grid |
| bank-major contiguous: split bank-major page order evenly over cores | same | 0 | 0 | <= 2 | fixed: ~13.75 writers per bank; each bank's writers clustered in ~1.3 adjacent grid rows (concentrated NoC paths) |
| page-order contiguous (every core spans all banks) | same | 0 | 0 | `B` | depends on walk order; convoys on one bank unless staggered |
| one page per transaction (any of the above) | same | 0 | 0 | `P / W` | as its split |
| zero source broadcast from one core | same | 0 | `(W - 1) * Z` | as its split | as its split |
| dependent-axis split + combine | none exists | | | | |

Chosen: **bank-affine.** It is tied for cheapest traffic. It has the fewest address lookups and partial transfers, and every bank has a fixed set of concurrent writers spread over the grid. Its one cost: cores of the 13-core banks (6 and 7) each write ~7.7% more bytes than cores of the 14-core banks. While the banks, not the cores, are the bottleneck that does not change the finish time. It is a perf lamp below, against the bank-major alternative that balances bytes per core exactly.

Idle-core occupancy is not in this table. No stage of the scheme leaves cores waiting on another core: there is no communication at all.

### Block schedule

Per core (one kernel, on the writer RISC). `num_blocks_this_core` is the number of banks the core's bank-major range touches: 1 for every core in every production shape at the Phase 0 placement, 0 for a core whose range is empty.

```cpp
make_zero_source();                       // once per core: Z bytes of zeros in cb_zero_source
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    write_zero_run(block_idx);            // one bank run: ceil(run_bytes / Z) single-packet writes
}
drain_writes();                           // one write barrier
```

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `make_zero_source` | `Z` bytes of L1 (`cb_zero_source`), not a block of the output | nothing before; `cb_zero_source` afterwards, for the whole kernel | once per core per call: `Z / 512` NoC loopback reads from the hardware zero region, then one `write_zeros_l1_barrier()`. **Stall shadow:** the kernel derives its first run's geometry (bank, first page id, run bytes, NoC address) *after* issuing the loopback reads and *before* the barrier; that work does not touch the source. No floating point is involved, so the reorder cannot change any result. |
| `write_zero_run` | 1 bank x `core_bank_rows` rows = one contiguous DRAM region of `core_bank_rows * page_bytes` bytes | `cb_zero_source` (read-only) | per block: one NoC address computation for the run's first page; per `Z` bytes: one single-packet non-posted write, every one sourced from the same `Z` bytes; no barrier, no CB handshake, no init |
| `drain_writes` | all writes of the core | nothing | once per core per call: one `noc_async_write_barrier()`. Nothing independent of it remains to schedule in its shadow: the writes are the kernel's only work. |

No CB reserve/push/wait/pop, no LLK init, no multicast, no semaphore anywhere in the schedule.

### Perf lamps

The overlap lamp does not apply: the op has no compute or read stage for the writes to overlap with. The only serialization is the one-time zeroing before the first write, and its shadow is filled (Block schedule).

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (occupancy) | p1 is DRAM-bound (`feature_spec.py`: roofline 620.14 us, target <= 837.19 us). 13-14 writers per bank may only add NoC contention at the DRAM endpoints. p4 (128 pages, the most called) is per-call overhead, so more cores is more launch work for 1-2 pages each. | `num_worker_cores` = `B * k` for k in {2, 4, 8} vs the full grid, on p1 and p4; judge on trace-replay host time, not the profiler's kernel duration alone (lesson RB-053: cores that end early read slower when profiled) |
| Bank-to-core placement | Default round-robin `c mod B` spreads each bank's writers over the grid but leaves banks 6-7 with 13 writers and ~7.7% more bytes each. | (a) bank-major contiguous placement (equal bytes per core, <= 2 runs per core, writers clustered); (b) locality placement: each bank's cores nearest its DRAM endpoint, seeded from `device.get_optimal_dram_bank_to_logical_worker_assignment(noc)`. Host-only changes: the kernel takes a bank-major range either way. Keep each kernel group a few rectangles (BB-17). |
| Transfer quantum `Z` | 16 KB transfers from ~14 writers per bank may interleave less evenly at the DRAM controller than smaller ones | `zero_chunk_bytes` in {16384, 8192, 4096} on p1 |
| Per-core RT args vs in-kernel derivation | Phase 0 sends 2 per-core RT words to each of 110 cores. On p4 that dispatch traffic is not negligible next to a ~1-2 page write per core. | All-common RT args (`num_pages`, buffer address, `W`, grid width), with each core deriving its range from `get_absolute_logical_x/y()` (`dataflow_api.h:44,57`). The host keeps the same formula only for the plan report, checked against the kernel by the dirty-destination test. |
| Stateful single-packet writes | Each write reprograms the full destination although a run never changes endpoint | `noc_async_write_one_packet_set_state` once per run (`dataflow_api.h:982`) + `noc_async_write_one_packet_with_state` per transfer (`dataflow_api.h:1008`); p1 only (168 transfers per core) |
| Second writer RISC | One RISC issues all of a core's transfers; this matters only if the RISC, not DRAM, is the bottleneck (`split_reader`) | Split each core's run in two halves: BRISC on NoC1 + NCRISC on NoC0. Measure only if profiling shows the writer issue-bound. |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Make the zero source | raw bytes in L1 (`cb_zero_source`, `Z` bytes) | `Noc::async_write_zeros(cb, Z)` (NoC loopback reads from `MEM_ZEROS_BASE`, 512 B each) + `Noc::write_zeros_l1_barrier()` | once per core; the bytes are undefined until the barrier returns |
| Write the bank run | destination: TILE pages of `dtype`. All-zero bytes are +0.0 in float32 / bfloat16 and exponent 0 + datum 0x00 (+0.0) in bfloat8_b, so the same bytes serve every dtype | non-posted single-packet NoC writes, L1 -> one DRAM bank, NoC1 (`WriterConfigDescriptor` default, RISCV_0), each `min(Z, remaining)` bytes from the same source address | `ceil(run_bytes / Z)` writes per block; chunk boundaries need not fall on page boundaries because the run is one contiguous byte range of the output buffer |
| Completion | n/a | `noc_async_write_barrier()` | every write acked before the kernel returns (BB-03) |
| Mesh | n/a | `ttnn.generic_op([output], descriptor)` with a `ProgramDescriptor` runs the same program on every device of the output's mesh (`generic_op.cpp:13-23`); a single-tensor io list is the documented generator form (`generic_op_nanobind.cpp:35-38`) | the output is allocated with the per-device shape on every device; the buffer address is the same on every device of a mesh buffer |

No Tensix-to-Tensix communication exists in any built or deferred regime.

TARGET axes and the blocking scheme:

| TARGET axis value | Phase 0 | Cost to add | Why |
|---|---|---|---|
| `dtype: float32` | SUPPORTED | n/a | Phase 0 |
| `dtype: bfloat16` | refused (`UnsupportedAxisValue`) | **knob-turn: SUPPORTED-only change** | The kernel writes bytes, and the page size enters only through `page_bytes` / the accessor's aligned page size and the host's run arithmetic. No kernel, CB or schedule change. |
| `dtype: bfloat8_b` | refused (`UnsupportedAxisValue`) | **knob-turn: SUPPORTED-only change** | As bfloat16. 1088 B pages are a multiple of the 64 B DRAM alignment, so bank runs stay contiguous (mechanism cap). The all-zero page is exponent 0 + datum 0 = +0.0 (`blockfloat_common.cpp:200-234`). |
| `layout: TILE_LAYOUT` | SUPPORTED | n/a | the only TARGET value |

There is no `memory_layout` axis in TARGET: DRAM interleaved is the only placement. Structural impossibilities: none (INVALID stays empty).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one bank run (block): 1 bank x `core_bank_rows` rows |
| Grid | `W` = `grid.x * grid.y` worker cores (110 here), one rectangle `(0,0)-(grid.x-1, grid.y-1)` from `device.compute_with_storage_grid_size()`; one kernel group, one core range (BB-17) |
| Per-core work | core `c` = row-major index (`c = y * grid.x + x`). Bank `b = c mod B`; slot `s = c div B` among the `W_b = (W - 1 - b) div B + 1` cores of bank `b`. `rows = rows_in_bank(b) = P div B + (b < P mod B)`; `base = rows div W_b`, `rem = rows mod W_b`; `start_row = s * base + min(s, rem)`; `core_bank_rows = base + (s < rem)` |
| Kernel-facing form | per-core RT args `[range_start, range_count]` in **bank-major page order** (`q = prefix(b) + row`, `prefix(b) = b * (P div B) + min(b, P mod B)`): `range_start = prefix(b) + start_row`, `range_count = core_bank_rows`. The kernel splits its range into bank runs (`block_idx`) and turns `(bank b, row r)` into page id `r * B + b` and its NoC address with the accessor. Every placement (Phase 0, the lamps, `W < grid`) is a host-only change. |
| Remainder | +-1 row per core within a bank (`rem`); +-1 row per bank (`P mod B`); the host asserts that the ranges tile `[0, P)` exactly once before the descriptor is cached. `P` comes from `output.buffer_num_pages()` (per-image `ceil` tile padding included). Cores with `range_count == 0` (only when `rows_in_bank(b) < W_b`, i.e. `P < ~B * 14`; no production shape) return at once without zeroing. |

Per-shape realization (`W` = 110, `B` = 8; banks 0-5 have 14 cores, banks 6-7 have 13):

| Case | dtype, page | `P` | rows per bank | rows per core (14-core banks / 13-core banks) | largest run | `Z` | transfers per core (max) | total transfers |
|---|---|---|---|---|---|---|---|---|
| p2 (16,12,128,128) | float32, 4096 B | 3072 | 384 | 27-28 / 29-30 | 122880 B | 16384 | 8 | 796 |
| p3 (8,12,128,128) | float32, 4096 B | 1536 | 192 | 13-14 / 14-15 | 61440 B | 16384 | 4 | 440 |
| p4 (1,4,256,128) | bfloat16, 2048 B | 128 | 16 | 1-2 / 1-2 | 4096 B | 4096 | 1 | 110 |
| p1 (32831,1,32,256) | bfloat8_b, 1088 B | 262648 | 32831 | 2345-2346 / 2525-2526 | 2748288 B | 16384 | 168 | 17472 |

Every core has at least one row in every production shape: the full grid is busy in all four cases.

Single regime: no regime-selection function.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_zero_source` | 0 | `zero_chunk_bytes` | 1 | Constant L1 source of every write. Spans neither block axis (`bank`: streams, `bank_row`: streams): it is re-read for every transfer, so its size is the transfer quantum, not the block. Used as raw L1 through its write pointer; no reserve/push/wait/pop. Read pointer == write pointer on a fresh CB. | raw bytes; declared `ttnn.bfloat16` (a non-block format, so `Z` need not be a multiple of a block-float tile; no unpacker or packer ever touches it) | writer (NoC loopback zeroing) | writer (source of the DRAM writes) | whole kernel |

The full ledger is `l1_ledger.md` beside this file.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `make_zero_source` | `Z` bytes of L1 | raw dataflow API (`Noc::async_write_zeros` overload 1) | none (hardware zero region `MEM_ZEROS_BASE`) | `cb_zero_source`, 1 page, written through its write pointer | holds `Z` zero bytes once `write_zeros_l1_barrier()` returns; never written again |
| 2 | `write_zero_run` | 1 bank x `core_bank_rows` rows | raw dataflow API (single-packet NoC writes) | `cb_zero_source`, 1 page, read-only | none (DRAM: output buffer, one bank) | unchanged |
| 3 | `drain_writes` | all of the core's writes | raw dataflow API (`noc_async_write_barrier`) | none | none | unchanged |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `make_zero_source` | raw_api | `Noc::async_write_zeros(dst, size_bytes, args)` (overload 1, local L1) | `tt_metal/hw/inc/api/dataflow/noc.h:849`; impl `tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl:13-44` | `dst` = the `cb_zero_source` handle, `size_bytes` = `zero_chunk_bytes` | none | `cb_zero_source` | `size_bytes` = `zero_chunk_bytes` (transfer quantum) |
| `make_zero_source` | raw_api | `Noc::write_zeros_l1_barrier()` | `noc.h:904`; impl `noc_zero_l1.inl:46` | none | none | `cb_zero_source` | none |
| `write_zero_run` | raw_api | `TensorAccessor::get_noc_addr(page_id)` (interleaved) | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:553` | `page_id = first_row * B + bank`; accessor built with `TensorAccessorArgs` CT args, the buffer address (common RT) and `page_bytes` = aligned page size | none | none | none |
| `write_zero_run` | raw_api | `TensorAccessor::contiguous_page_stride()` / `num_contiguous_pages(page_id, end_page_id)` (run contiguity contract; stride check) | `tensor_accessor.h:562`, `tensor_accessor.h:568-575` | stride == `B`; a run of `n` rows is `n * get_aligned_page_size()` contiguous bytes (`tensor_accessor.h:545`) | none | none | `core_bank_rows` (`n`) |
| `write_zero_run` | raw_api | `noc_async_write_one_packet<true, false>(src, dst_noc_addr, size)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:841` | `src` = `cb_zero_source` address (same for every transfer), `size` = `min(Z, remaining)`, non-posted | `cb_zero_source` | none | `Z` |
| `drain_writes` | raw_api | `noc_async_write_barrier()` | `dataflow_api.h:1951` | default NoC | none | none | none |

Helpers considered and rejected:

| Block operation | Candidate | File:line of the mismatch | Concrete reason |
|---|---|---|---|
| `make_zero_source` | `dataflow_kernel_lib::zero_tile(dfb)` / `prepare_zero_tile` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:53-57`, `:67-71` | Zeroes exactly `dfb.get_tile_size()` bytes (one tile, `l1_helpers.hpp:56`). `Z` is not a whole number of tiles: 16384 B is 15.06 bfloat8_b tiles. The `prepare_` form also pushes a page this kernel never pops. Internally it is the same `async_write_zeros` + `write_zeros_l1_barrier` pair used here (`l1_helpers.hpp:55-56`), with a different size. |
| `make_zero_source` | `dataflow_kernel_lib::fill_l1_range<4>(addr, Z, 0)` | `l1_helpers.hpp:101`, bulk loop `l1_helpers.hpp:136-141` | Writes 4 B per RISC store: 4096 serial stores for 16 KB on the writer RISC, vs 32 asynchronous 512 B NoC loopback reads (`noc_zero_l1.inl:31-38`). The loopback reads also leave the RISC free to compute the run geometry in the barrier's shadow. |
| `write_zero_run` | `Noc::async_write_zeros(accessor, size, {.page_id}, scratch)` (overload 2) | `noc.h:893`; `noc_zero_dram.inl:24` asserts `args.offset_bytes + size_bytes <= accessor.get_aligned_page_size()`; TODO `noc_zero_dram.inl:10-11` | One call per page: it cannot express a multi-page bank run. On p1 it would issue 262648 transactions of 1088 B instead of 17472 of up to 16 KB. Missing block operation: a page-range overload (`noc_zero_dram.inl:10-11` names exactly this). Implement `write_zero_run` as a small kernel function over `noc_async_write_one_packet`. It may be promoted to that overload later. |
| `write_zero_run` | `noc_async_write(src, dst, size)` (any length) | `dataflow_api.h:876`; it splits at `NOC_MAX_BURST_SIZE` and advances the source with the destination | A run is up to 2748288 B (p1), so the source would have to be as long as the run. That is more than the 1347776 B L1 budget, and it makes the L1 footprint scale with the block. The fixed `Z`-byte source needs a one-packet-per-`Z` loop. |
| all | compute-kernel helpers (`eltwise_*`, `tilize`, ...) | n/a | No compute phase exists: the op does no arithmetic and the zero bytes are already in final page format for every dtype. No compute kernel is created. |

## Host Entry Point, Registry and Program Cache

| Concern | Decision |
|---|---|
| Registry (`g6_zeros.py`) | `INPUT_TAGGERS = {"dtype": tag_dtype, "layout": tag_layout}`: `tag_dtype` returns `inputs[0]["dtype"]`, `tag_layout` returns `inputs[0]["layout"]`; `inputs = (scenario,)`, `scenario = {"shape": tuple, "dtype": ..., "layout": ...}`. Phase 0 `SUPPORTED = {"dtype": [ttnn.float32], "layout": [ttnn.TILE_LAYOUT]}`; `EXCLUSIONS = []`. No size, shape or rank axis, no other tagger. INVALID is not declared or checked by the op. |
| `validate(shape, *, dtype, layout, device, memory_config)` | Called as the entry point's first line. (1) Structural checks, raising `ValueError`: `shape` is a list or tuple of 4 `int` (not `bool`), each `>= 1`, `d_2 % 32 == 0`, `d_3 % 32 == 0`; `layout == TILE_LAYOUT`; `memory_config.memory_layout == INTERLEAVED and memory_config.buffer_type == DRAM`; `isinstance(device, ttnn.MeshDevice)` (`ttnn.Device` is the same class). A closed device raises `RuntimeError` from its first query. (2) Build `(scenario,)` from the arguments, apply `INPUT_TAGGERS`, refuse values outside `SUPPORTED` with `UnsupportedAxisValue`, then `EXCLUSIONS` with `ExcludedCell` (`ttnn/ttnn/operations/_op_contract.py`; both subclass `NotImplementedError`, itself a `RuntimeError`). |
| Allocation | `ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, memory_config)` on every call: new storage every call (spec rule 2), allocated at trace capture in the traced models and never at replay (BB-16) |
| Dispatch | exactly one `ttnn.generic_op([output], descriptor)` per call (BB-01). No other ttnn op, no host-made values, no conversion (BB-02). |
| Program cache (BB-07) | The `ProgramDescriptor` is built once per key and memoized in a module dict. Key: device identity (mesh id + mesh shape), worker grid, `P`, `dtype` (page size), `layout`, `memory_config`. On every call the cached descriptor gets only the new output buffer address, a **common** RT arg, so it is one value for all cores. `generic_op` hashes the descriptor structurally (CT args, core ranges, CB sizes, kernel source; not RT values or addresses: `generic_op_device_operation.cpp:78-146`) and copies the RT args on a cache hit. Everything that changes the program binary (`Z`, `page_bytes`, `B`, accessor args, core range) is in CT args or descriptor structure, so it is in the key. p2 and p3 share one binary and differ only in RT args. |
| Plan report (BB-08, BB-21) | The module exposes the plan a call selects: regime `"bank_run"`, `W`, `Z`, `B`, and per core `(x, y, role, bank, start_row, core_bank_rows)`. Role is `zero_writer`, or `idle` when the range is empty. It is computed by the same host function that fills the RT args, never by a second formula. |
| Profiling zones (BB-21) | `DeviceZoneScopedN` zones around `make_zero_source`, the run loop, and `drain_writes`; they compile to nothing in the plain build |
| Kernel set | one data-movement kernel (`WriterConfigDescriptor`: RISCV_0, NoC1) on the whole worker rectangle; no reader kernel, no compute kernel; one CB; no semaphores. Kernel text is a few dozen instructions (BB-04, BB-18). |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Pages left unwritten (R1) | A fresh test buffer often holds zeros, so an op that misses pages passes | Ranges tile `[0, P)` exactly (host assert); bank count checked against `NUM_DRAM_BANKS` (`static_assert`); `W >= B`; the acceptance and golden dirty-destination procedures poison the exact output range first |
| Source written before its zeroing lands (R2) | `async_write_zeros` to L1 is an asynchronous loopback read (`noc.h:825-826`) | `write_zeros_l1_barrier()` before the first DRAM write; only geometry arithmetic runs between the two |
| Writes in flight at kernel return (R3, BB-03) | the op is nothing but writes (up to 168 per core) | non-posted writes, one `noc_async_write_barrier()` at kernel end on every path, including a core that returns early: it issued nothing |
| Run written with the wrong stride | Bank-run contiguity needs page size == aligned page size and the right `B` | mechanism caps above (host asserts + `static_assert`) |
| bfloat8_b page of 1088 B (R5) | not a power of two; `Z` = 16384 is not a whole number of pages | Chunks are byte ranges of a contiguous run. A chunk may end mid-page and the next chunk continues the same page, so no page is split across a gap. All bytes are zero, so there is no sign bit to set. |
| One-packet write larger than the burst | `Z` is host-computed | `static_assert(zero_chunk_bytes <= NOC_MAX_BURST_SIZE)` |
| Incomplete cache key (R6, BB-07) | p2/p3 differ only in `d_0`; p1/p4 in dtype | Shape enters only through RT args, which are re-applied on every hit. Dtype changes `page_bytes`/`Z` CT args. The Python memo key includes `P` and `dtype`. Tested by the cache-order procedure. |
| Mesh: only one device written (R9) | the suite runs on one chip | `generic_op` with a `ProgramDescriptor` is SPMD over the output's mesh (`generic_op.cpp:13-23`); the output is allocated on the mesh device, never on a sub-device |
| CB used as raw scratch | no handshake orders anything | single kernel, single thread: zeroing, barrier, then reads, all in program order on one RISC; no other kernel touches `cb_zero_source` |

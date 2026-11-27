# Operation Design: g6_zeros

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (creation: writes only, no input, no arithmetic) |
| Goal | Allocate a new DRAM-interleaved TILE tensor of the per-device `shape` and `dtype` on every device of `device`, and write the all-zero bit pattern into every page, as one device program that reads nothing from DRAM and writes each page exactly once. |
| Math | `out[i] = +0.0` (all-zero bits of `dtype`) for every logical index `i`; `out` is new storage on every call |
| References | `docs/explorations/g6_zeros/{DOSSIER,math,tensix,risks}.md`; `eval/golden_tests/g6_zeros/{feature_spec,helpers}.py`; `.claude/references/{blocking-model,l1-footprint-discipline,noc-alignment-in-tt-metal}.md`; catalog `ttnn/ttnn/operations/examples/{read_coalescing,bank_stagger,noc_placement,double_buffer,split_reader}/` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `shape` | list/tuple of 4 `int` (positional) | yes | every `d_k >= 1`; `d_2 % 32 == 0`, `d_3 % 32 == 0` | none | sets `P` (common RT) and per-core ranges (RT) |
| `dtype` | `ttnn.DataType` (keyword-only) | yes | TARGET: `float32`, `bfloat16`, `bfloat8_b`; Phase 0 SUPPORTED: `float32` | none | sets `page_bytes` (CT) |
| `layout` | `ttnn.Layout` (keyword-only) | yes | `ttnn.TILE_LAYOUT` | none | validation only |
| `device` | `ttnn.MeshDevice` (keyword-only) | yes | an open device or mesh device | none | grid (`compute_with_storage_grid_size()`), mesh extent |
| `memory_config` | `ttnn.MemoryConfig` (keyword-only) | yes | DRAM, INTERLEAVED | none | `TensorAccessorArgs` (CT) |

### Validation order (`validate()` is the entry point's first line)

| # | Check | Failure |
|---|-------|---------|
| 1 | `shape` is a list/tuple of exactly 4 Python ints (not `bool`), each `>= 1` | `ValueError` |
| 2 | `shape[2] % 32 == 0` and `shape[3] % 32 == 0` | `ValueError` |
| 3 | `layout == ttnn.TILE_LAYOUT` | `ValueError` (outside TARGET — a contract error, not a support refusal) |
| 4 | `memory_config.buffer_type == DRAM` and `memory_config.memory_layout == INTERLEAVED` | `ValueError` |
| 5 | `device` is a `ttnn.MeshDevice` (single device = 1x1 mesh) and is open | `ValueError` |
| 6 | Build `scenario = {"shape": tuple(shape), "dtype": dtype, "layout": layout}`; run `INPUT_TAGGERS` on `(scenario,)` | — |
| 7 | SUPPORTED per axis | `UnsupportedAxisValue` |
| 8 | EXCLUSIONS (empty) | `ExcludedCell` |

Registry declarations (op file): `INPUT_TAGGERS = {"dtype": tag_dtype, "layout": tag_layout}`, where `tag_dtype` returns `inputs[0]["dtype"]` and `tag_layout` returns `inputs[0]["layout"]`. `SUPPORTED = {"dtype": [ttnn.float32], "layout": [ttnn.TILE_LAYOUT]}`. `EXCLUSIONS = []`. No size, shape or rank axis and no other tagger. INVALID is not declared. Structural impossibilities: none (TARGET holds TILE only, and all three dtypes store in tiles).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | none — the op takes no input tensor and reads no tensor data |
| Dtype | — |
| Layout | — |
| Memory | — |

### Output

| Property | Value |
|----------|-------|
| Shape | `shape` (per device), e.g. `(d_0, d_1, d_2, d_3)` |
| Dtype | `dtype` (made in `dtype`, no cast) |
| Layout | TILE; one page = one 32x32 tile: `page_bytes` = 4096 (float32), 2048 (bfloat16), 1088 (bfloat8_b: 64 B exponents + 1024 B data) |
| Memory | DRAM interleaved, allocated by this call with `ttnn.allocate_tensor_on_device` on `device` (every device of the mesh, same address on each) |
| Pages | `P = d_0 * d_1 * ceil(d_2/32) * ceil(d_3/32)` (alignment-aware; equals the exact product for every legal shape) |

## Blocking Model

The op is a write of `P` identical pages. Nothing is read and no element depends on another, so the only structure that matters is **where the pages sit in DRAM**. Page `p` of a DRAM-interleaved tensor sits in bank `b = p mod NB` at bank slot `k = p div NB`. Slots `k, k+1, …` of one bank are contiguous bytes, `page_bytes` apart (`tensor_accessor.h:560-576`). The blocking therefore re-indexes the page space as `(bank, bank_slot)` and blocks on the slots. A block is one **run**: consecutive slots of one bank, written as one contiguous DRAM range from a zeroed L1 source that every packet reuses.

Symbols: `NB` = DRAM bank count (`NUM_DRAM_BANKS`, 8 on Blackhole); `S = ceil(P / NB)`; `r = P - NB*(S-1)` (`1 <= r <= NB`); `n_b = S` for `b < r`, else `S - 1` (slots that exist in bank `b`). In every production case `P % 8 == 0`, so `r = NB` and all banks hold `S` slots. The general `n_b` stays in the design because a 12-bank device has unequal banks.

**Bank-major position space.** Position `q` in `[0, P)` enumerates bank 0's slots, then bank 1's, and so on: `start(b) = b*S` for `b <= r`, `start(b) = r*S + (b-r)*(S-1)` for `b > r`. The work split is a balanced contiguous split of this position space.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `d0` (1..32831) | independent: every element is the constant +0.0, so no result spans it | folded into `page`; it has no extent knob of its own | whole extent, folded | TILE page order `p = ((i0*d1+i1)*Ht+ht)*Wt+wt` | through (`bank`, `bank_slot`) | n/a. Folding is exact because the output is constant; no logical axis survives the bank re-indexing with any property |
| `d1` (1..12) | independent: as `d0` | folded into `page` | whole extent, folded | as `d0` | through (`bank`, `bank_slot`) | n/a |
| `ht` = `d2/32` (1..8) | independent: as `d0` | folded into `page` | whole extent, folded | as `d0` | through (`bank`, `bank_slot`) | n/a |
| `wt` = `d3/32` (4 or 8) | independent: as `d0` | folded into `page` | whole extent, folded | as `d0` | through (`bank`, `bank_slot`) | n/a |
| `bank` (`NB`) | independent: a page's bank is placement, not data | `banks_per_run` | **1**. Mechanism cap: contiguity exists only inside one bank, and one NoC write targets one bank endpoint | kernel, from `NUM_DRAM_BANKS` (`TensorAccessor::contiguous_page_stride()`, `tensor_accessor.h:562`); never a host literal | bank-major split: bank `b`'s `n_b` slots are covered by the about `C/NB` cores whose position ranges intersect `[start(b), start(b)+n_b)` | n/a (cap) |
| `bank_slot` (`n_b` per bank) | independent: as `d0` | `run_slots` = size of (core position range ∩ bank `b`) | **whole intersection**: one run per bank touched, never sub-cut. The core's entire assignment in that bank is one block | per-core RT range `[q_begin, q_begin+q_count)` from `ttnn.split_work_to_cores(grid, P, row_wise=True)` | contiguous position ranges over the full grid; ≤ 2 runs per core while `C >= NB` | knob-turn (`grid`, core order) |
| `packet` (bytes inside a run) | independent: every packet copies the same zero bytes to a disjoint DRAM range | `packet_bytes` = `min(zero_src_bytes, NOC_MAX_BURST_SIZE)` | `zero_src_bytes = min(ZERO_SRC_BYTES_MAX, core_positions_max * page_bytes)`, `ZERO_SRC_BYTES_MAX = 16384` (one Blackhole burst, `noc_parameters.h:290`) | host constant `ZERO_SRC_BYTES_MAX` → `zero_src_bytes` (CT); kernel clamps to `NOC_MAX_BURST_SIZE` | inside one core | knob-turn |

`core_positions_max = ceil(P / cores_used)`; `cores_used = min(C, P)`; `C = grid.x * grid.y`; `grid = device.compute_with_storage_grid_size()`. All are derived once on the host from the single source `grid` (a parameter of the descriptor builder, default the full compute grid).

**Intermediate data.** The scheme produces one datum besides the output: the zero source. It is per-core, identical on every core and has no axis to spread. No stage computes on one core while others wait: every core makes its own source and writes its own runs.

**Operand-reuse check.** The only operand is the zero source. It does not vary along any split axis, so it is reuse-shared by construction. Each core makes it **once**, in its own L1, and sources every packet of every run from it. It is never re-made per page or per run, and never broadcast: a core-local NoC loopback from the hardware zero region (`MEM_ZEROS_BASE`, 512 B, `dev_mem_map.h:88,115`) costs no cross-core traffic, which any multicast would add.

**Stall-shadow check.** There are two waits. (1) The zero-fill barrier before the first write. Every write depends on the zero bytes, but the mapping of the core's position range to runs (bank, first slot, slot count, destination address) does not. Issue the fill, compute the run list, then wait on the barrier. (2) The final write barrier (BB-03). No work remains, so nothing fills it.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_zero_src` | `zero_src_depth` | 1 | Nothing more: the source is written once per kernel and only read afterwards, so there is no refill to overlap. A second page would be dead L1. The NoC keeps packets in flight from the same source, bounded only by command-buffer flow control (catalog `double_buffer`: no gain once DRAM-bound). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| One-packet NoC write (stateful issue, `dataflow_api.h:982,1008`) | `packet_bytes` | `min(zero_src_bytes, NOC_MAX_BURST_SIZE)` (16384 BH, 8192 WH) | an oversize single-packet write is illegal; the sanitizer stops it, or it corrupts |
| Zero source covers every packet | `packet_bytes <= zero_src_bytes` | same clamp | the packet reads past the zeroed bytes and streams stale L1 into the output (the same hazard `noc.h:855-857` states for overload 2) |
| Run contiguity (one bank) | `banks_per_run = 1`; run end `<= n_b` | the kernel cuts a run at the bank's last existing slot `n_b`, derived from `P` and `NUM_DRAM_BANKS` | a run into bank `b`'s slot `>= n_b` writes allocation slack that is not an output page; past the allocation it would write another buffer |
| Slot stride = aligned page size | address of slot `k0 + j` = `addr(k0) + j * aligned_page_bytes` | the accessor is built with the aligned page size (the default); `tensor_accessor.h:570-572` collapses runs to 1 page otherwise | wrong addresses for every slot after the first |
| NoC/DRAM alignment (`noc-alignment-in-tt-metal.md:43`; BH DRAM writes modulo 16) | `zero_src_bytes`, source base address | `zero_src_bytes % 16 == 0` (all page sizes are multiples of 64; 16384 is too); the CB base is 16-B aligned (asserted at `noc_zero_l1.inl:21`) | silent data corruption without the watcher |
| Kernel-side bank count | `NB` | the kernel uses `NUM_DRAM_BANKS` only; the host never sends a bank count | runs land in the wrong banks on another arch |
| L1 budget (BB-05) | `zero_src_bytes` | `<= 16384` ≪ 1347776 | — |

### Regimes

Named boundary: **DRAM**. The minimum is that the output crosses DRAM once (`P * page_bytes` written), with zero bytes read and no cross-core traffic.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `bank_run_write`: full-grid bank-major split; each core zero-fills one L1 source once, then writes its ≤ 2 bank runs as contiguous DRAM ranges in `packet_bytes` packets | **built** | always: every shape and dtype of TARGET (the write is identical, only `page_bytes` differs) | (`banks_per_run`=1) x (`run_slots` = core range ∩ bank) | **minimum**: out written once, 0 B read, 0 B cross-core. Core-local extra: `zero_src_bytes` of NoC loopback per core, once | Per-run fixed cost: run-to-address mapping, destination-state setup, and the run's tail packet. Intended frequency: ≤ 2 per core per call. Per-kernel fixed cost: the zero fill and its barrier, and one final write barrier, each once per core. Packet count is set by `packet_bytes`, not by the block. The block is already the whole core assignment per bank, so the only bigger block is fewer cores (grid-sync lamp). |
| `page_write`: page-id split, one NoC write of one page per page | **rejected**, superseded by `bank_run_write` | — | 1 page | bytes are minimum, but transactions equal `P` (262648 writes of 1088 B on p1; `tensix.md` puts that at 94M–142M core-cycles of issue work) | The dead end: no run-based scheme passes through it, and it loses everywhere except p4, where both write one page per core |
| `page_id_split_bank_coalesced`: contiguous page-id range per core, regrouped into one run per bank | **rejected**, superseded by `bank_run_write` | — | 1 bank x about `core_pages/NB` slots | same bytes | NB runs per core instead of ≤ 2, and every core starts on the same bank at the same moment unless staggered (catalog `bank_stagger`). Bank-major gets the coalescing with each bank fed by its own core set at all times. |
| `compute_fill_pack`: compute engine fills DEST with 0 and packs each tile into a CB that the writer drains | **rejected**, superseded by the core-local NoC zero fill | — | 1 tile | same DRAM bytes; adds a compute kernel, a CB handshake per tile and per-tile pack work | The pack of bfloat8_b computes shared exponents (`risks.md` R5), and every page passes through a CB handshake that `bank_run_write` never pays |
| `zero_region_direct`: packets sourced straight from `MEM_ZEROS_BASE` (512 B), no L1 source | **rejected**, superseded by `bank_run_write` | — | runs, in 512 B packets | same bytes; 32x the packets of a 16 KiB source | Saves 16 KiB of L1 that the budget does not need |
| `dram_zero_source`: read zero pages from a DRAM buffer | **rejected**: spec rule 4 forbids it | — | — | adds one DRAM read per write | — |
| `mcast_zero_source`: one core makes zeros and multicasts them to the grid | **rejected**, superseded by per-core loopback fill | — | — | adds `C * zero_src_bytes` of cross-core traffic and a semaphore round | — |
| dependent-axis split + combine | **n/a**: the op has no dependent axis, so no combine regime exists | — | — | — | — |
| resident fast-path | **n/a**: nothing is read, so residency saves no traffic | — | — | — | — |

There is one built regime, so there is no selection predicate and no regime-pinned test. `bank_run_write` is the whole domain: every shape is the same page write, and no shape class changes the transfer structure. The grid size changes only the per-core extent.

### Traffic ranking

| Candidate split | DRAM bytes | Cross-core bytes | Transfers (shape) | Rank |
|-----------------|-----------|------------------|-------------------|------|
| bank-major positions (`bank_run_write`) | `P*page_bytes` written, 0 read | 0 | ≤ 2 contiguous runs per core, `ceil(run_bytes/packet_bytes)` packets each; each bank fed continuously by its own about `C/NB` cores | **1 (chosen)** |
| page-id ranges, per-bank coalescing | same | 0 | NB runs per core, all cores on the same bank at once | 2 |
| logical-axis splits (`d0`/`ht` rows, `wt` columns) | same | 0 | equivalent to page-id ranges; a column split fragments further | 3 |
| page-id ranges, one write per page | same | 0 | `P` writes of `page_bytes` | 4 (dead end) |

Every split moves the minimum bytes, so the ranking falls to transfer shape: fewer, longer, bank-local transfers, which catalog `read_coalescing` measured at 1.11–3.66x. Occupancy is not a ranking term. All splits fill the grid equally.

### Block schedule

```cpp
// per core; num_runs_this_core <= 2 while C >= NB, derived in-kernel from [q_begin, q_begin + q_count)
if (q_count == 0) return;                       // idle core: no NoC traffic, nothing to barrier
make_zero_source();                             // once per kernel: zero_src_bytes of cb_zero_src from MEM_ZEROS
map_range_to_runs();                            // in the fill's shadow: (bank, first_slot, run_slots, dst addr) per run
wait_zero_source();                             // write_zeros_l1_barrier: no packet before the zero bytes landed (R2)
for (uint32_t run_idx = 0; run_idx < num_runs_this_core; ++run_idx) {
    write_zero_run(run_idx);                    // one block: run_slots * page_bytes contiguous bytes in one bank
}
drain_writes();                                 // once per kernel: noc_async_write_barrier (BB-03)
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `make_zero_source` | `zero_src_bytes` bytes (1 page of `cb_zero_src`) | — | once per kernel: `ceil(zero_src_bytes/512)` loopback reads |
| `map_range_to_runs` | the core's position range | — | once per kernel (≤ 2 divisions per run) |
| `wait_zero_source` | — | `cb_zero_src` | once per kernel |
| `write_zero_run` | (1 bank) x (`run_slots` slots) = `run_slots*page_bytes` contiguous DRAM bytes | `cb_zero_src` (read-only, every packet) | per run: one destination-state setup (the bank's NoC endpoint) and the length state; per packet: one issue (source address constant, destination advancing by `packet_bytes`); the tail packet re-sets the length once. **No barrier, flush or CB handshake inside the run or between runs.** |
| `drain_writes` | — | — | once per kernel |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (p4, and p3) | p4 is 128 pages: with the full grid each core writes 1–2 pages (2–4 KiB, one packet), and the per-core fixed cost (launch, fill, ack round trip) dominates. p4 is the most-called case (240x per FLUX pass) | the same descriptor with `grid` cut to `NB`, `2*NB`, `4*NB` cores (longer runs per core). Judge on unprofiled blocking trace-replay host time, not profiled kernel time (`lessons.md` RB-053) |
| Second NoC (`writer_riscs`) | one writer per core on BRISC/NoC1 means all of bank `b`'s traffic shares the NoC1 links into its endpoint. At the p1 target (≤ 837 µs for 285.76 MB, about 341 GB/s, about 43 GB/s per bank) that is a large fraction of one link | `writer_riscs = 2`: the same kernel source also on NCRISC/NoC0, each RISC taking half of the core's position range, with a second `cb_zero_src` per RISC (no cross-RISC handshake on one CB). Descriptor-only. Catalog `split_reader` and `noc_placement` |
| Bank-affine placement | Phase 0 feeds bank `b` from the cores at consecutive row-major indices (about 1.25 grid rows), regardless of where bank `b`'s NoC endpoint sits | assign each bank's positions to the cores physically nearest its endpoint (catalog `read_coalescing` affinity: 0.91–1.28x after coalescing), and a diagonal core order (catalog `noc_placement`). RT-arg and core-order only |
| Overlap (zero-source fill vs. first write) | the 16 KiB fill (32 loopback reads) precedes the first packet. On p2/p3 (4–7 packets per core) the fill is a visible share of the core's time | `ZERO_SRC_BYTES_MAX` = 4096 / 8192 (more packets, shorter fill); or start the first run's writes as soon as each 512 B-aligned prefix lands |

Catalog entries that set the knobs: `read_coalescing` (one wide call over physically contiguous bytes → block = whole run), `bank_stagger` (synchronized bank walks → the bank-major split avoids them by construction, so no stagger knob), `noc_placement` (writes on NoC1, `row_wise=True` core order), `double_buffer` (depth buys nothing once DRAM-bound → `zero_src_depth = 1`), `split_reader` (second RISC only if issue-bound → lamp), `tile_reorder` (whole coalesced transfers beat fragments).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Hardware zero region → `cb_zero_src` | raw zero bytes | `Noc::async_write_zeros` overload 1 (NoC loopback read from `MEM_ZEROS_BASE`, 512 B chunks) + `Noc::write_zeros_l1_barrier` | core-local; the writer kernel on BRISC/NoC1 is both producer and consumer; no push/pop (a fresh CB has read_ptr == write_ptr, `noc.h:861-862`) |
| `cb_zero_src` → DRAM bank `b` | raw zero bytes = all-zero pages of `dtype` (+0.0; bfloat8_b exponents and data all zero) | NoC unicast writes, `packet_bytes` each, constant source, destination advancing along the run | packet boundaries need not align to pages (16384 is not a multiple of 1088); the run is one contiguous range |
| Compute | — | none: no compute kernel in the program | the op does no arithmetic |
| Mesh | — | `ttnn.generic_op([output], descriptor)` runs the descriptor SPMD on every device of the output's mesh (`generic_op.cpp:13-22`) | the same address on every device (lockstep mesh allocation); every chip writes its own copy |

TARGET as placement: TARGET has no `memory_layout` axis (DRAM interleaved only), so no physical-shard scheme exists. The `dtype` values `bfloat16` and `bfloat8_b` are **knob-turns**: they change only `page_bytes` (CT), and through it `zero_src_bytes` and the run byte counts. No algorithm, CB or kernel changes. A refinement adds them by editing SUPPORTED and running their two cases.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one bank-major position (= one page / bank slot); a core's units form ≤ 2 runs |
| Grid | `grid = device.compute_with_storage_grid_size()` (parameter, default full grid); kernel and CB core range = the **full grid rectangle**, one `CoreRange` (BB-17), including cores with no work |
| Per-core work | `(num_cores, all_cores, g1, g2, n1, n2) = ttnn.split_work_to_cores(grid, P, row_wise=True)`. Enumerate the grid row-major (x fastest). The first `|g1|` cores get `n1` positions, the next `|g2|` get `n2`, the rest get 0. `q_begin` is cumulative. Per-core RT: `(q_begin, q_count)`. Common RT: output buffer address, `P` |
| Remainder | `split_work_to_cores` balances to ±1 position. Ragged runs at bank boundaries are handled in-kernel by `n_b`. `P` is alignment-aware (`ceil` per image). `cores_used = min(C, P)`: a page is the finest work unit, by decision, because a sub-page split adds runs and fights the grid-sync lamp |
| Runs per core | ≤ 2 whenever `C >= NB` (then `q_count <= S`). The kernel walks runs until the range is exhausted, so a `grid` knob below `NB` stays correct |

Illustrative extents (NB = 8, C = 110; the formulas above are normative):

| Case | P | page_bytes | S (slots/bank) | positions/core (max) | bytes/core (max) | packets/core | zero_src_bytes |
|------|---|-----------|----------------|----------------------|------------------|--------------|----------------|
| p1 (32831,1,32,256) bf8b | 262648 | 1088 | 32831 | 2388 | 2,598,144 | ≤ 160 | 16384 |
| p2 (16,12,128,128) fp32 | 3072 | 4096 | 384 | 28 | 114,688 | ≤ 8 | 16384 |
| p3 (8,12,128,128) fp32 | 1536 | 4096 | 192 | 14 | 57,344 | ≤ 5 | 16384 |
| p4 (1,4,256,128) bf16 | 128 | 2048 | 16 | 2 | 4096 | ≤ 2 | 4096 |

### Host-side realization

| Requirement | Design |
|-------------|--------|
| One dispatch (BB-01) | allocate the output, then make one `ttnn.generic_op([output], descriptor)` call. No other device op, no host-made values, no `from_torch` |
| New storage (spec rule 2) | `ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, device, memory_config)` on every call; never cached or reused |
| Program cache (BB-07) | host-side descriptor cache keyed by `(tuple(shape), dtype, grid)`. A hit patches only the output address (one **common** RT arg) and reuses the cached descriptor. The generic_op hash covers structure (kernels, CT args, core ranges, RT-arg counts, CBs; `generic_op_device_operation.cpp:78-146`); RT values are re-applied from the passed descriptor on every hit (`mesh_device_operation_adapter.hpp:958-962`). So p2 and p3 may share one compiled program, and each call still carries its own `P` and ranges |
| Plan report (BB-08, BB-21) | the op module exposes the plan of a call: regime `bank_run_write`, `grid`, `cores_used`, per-core `(q_begin, q_count, runs)`, `zero_src_bytes`, `packet_bytes`. The only role is "zero writer"; idle cores are listed |
| Zones (BB-21) | profiler-build-only named zones: `make_zero_source` (fill + barrier), `write_zero_run` (issue), `drain_writes` (final barrier). The plain build is unchanged |
| Trace (BB-16) | allocation happens at capture (inside the call). Replay re-runs the captured program with the captured address and RT args, and allocates nothing |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_zero_src` | 0 | `zero_src_bytes` | `zero_src_depth` = 1 | Spans the `packet` axis (it must cover one packet). Streams over `bank` and `bank_slot`: every packet of every run re-reads the same bytes. Constant live set ≤ 16 KiB | output `dtype` (metadata only: no compute kernel reads it) | writer | writer | the whole kernel: filled once, then read-only |

No other CB exists. There is no output CB: the output is written to DRAM straight from the zero source, with no staging.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `make_zero_source` | `zero_src_bytes` | device dataflow API (kernel_lib helper rejected, see API Mapping) | — (hardware zero region) | `cb_zero_src`, 1 page (no reserve/push) | zero bytes in flight |
| 2 | `map_range_to_runs` | core position range | — (integer math) | — | — | — |
| 3 | `wait_zero_source` | — | device dataflow API | `cb_zero_src` | — | zeroed; read-only from here |
| 4 | `write_zero_run` | 1 bank x `run_slots` | raw NoC write (no page-range helper exists) | `cb_zero_src`, 1 page, read without wait/pop | — (DRAM) | unchanged |
| 5 | `drain_writes` | — | raw NoC barrier | — | — | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `make_zero_source` | raw_api | `Noc::async_write_zeros(dst, size_bytes)` (overload 1) | `tt_metal/hw/inc/api/dataflow/noc.h:849`; impl `tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl:13-41` | `dst` = `cb_zero_src` handle, `size_bytes = zero_src_bytes` | — | `cb_zero_src` | `zero_src_bytes` |
| `wait_zero_source` | raw_api | `Noc::write_zeros_l1_barrier()` | `noc.h:904`; impl `noc_zero_l1.inl:46` | — | `cb_zero_src` | — | — |
| `map_range_to_runs` | raw_api | `TensorAccessor::get_noc_addr(page_id)` (interleaved) | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:554`; stride `:562`; contiguity contract `:560-576` | `page_id = k0*NB + b` of the run's first slot; accessor from `TensorAccessorArgs` (CT) + output address (common RT), aligned page size | — | — | — |
| `write_zero_run` | raw_api | `noc_async_write_one_packet_set_state` / `noc_async_write_one_packet_with_state` (or `noc_async_write`) | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:982`, `:1008` (`:876`) | dst = run base + `j*packet_bytes`, size = `packet_bytes` (tail: remainder), src = `cb_zero_src` base; **non-posted** | `cb_zero_src` | — | `packet_bytes`, `run_slots` |
| `drain_writes` | raw_api | `noc_async_write_barrier()` | `dataflow_api.h:1951` | — | — | — | — |

Helpers considered and rejected:

| Raw-API entry | Candidate helper | File:Line of the mismatch | Why it cannot express the shape |
|---------------|------------------|---------------------------|----------------------------------|
| `make_zero_source` | `dataflow_kernel_lib::zero_tile` / `prepare_zero_tile` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:53-57`, `:67-71`; `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:276-282` | Zeroes exactly `dfb.get_tile_size()` bytes: one tile of the buffer's data format, read from the JIT tile-size descriptor. The source here is `zero_src_bytes` (16384 = 4 float32 tiles = 15.06 bfloat8_b tiles), not a tile multiple for bfloat8_b. `prepare_zero_tile` also pushes, which a same-kernel source must not do. The helper is a thin wrapper over the same two calls used here |
| `make_zero_source` | `dataflow_kernel_lib::fill_l1_range<4>(addr, n, 0)` | `l1_helpers.hpp:101-140` | RISC stores, 4 B each: 4096 stores for 16 KiB, which block NoC issue, against 32 background loopback reads |
| `write_zero_run` | `Noc::async_write_zeros(accessor, size, {page_id}, scratch)` (overload 2) | `noc.h:893-897`; single-page limit asserted at `tt_metal/hw/inc/internal/noc_zero_dram.inl:24`; the gap is named at `noc_zero_dram.inl:10-11` | One call stays inside one page (`offset + size <= aligned page size`), so a run would become `run_slots` calls with packets ≤ `page_bytes` (1088 B on bfloat8_b): the rejected `page_write` regime. **Gap to close:** a page-range overload "zero a bank at a time" (the TODO at `noc_zero_dram.inl:10-11`) would make `write_zero_run` a direct helper call |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A page left unwritten (R1) | the op's whole result is what it writes; fresh test buffers often hold zeros | Σ `q_count` = `P` exactly; runs cut at `n_b`; the kernel walks runs until its range is exhausted; dirty-destination tests (acceptance + golden loose 4–7) |
| Zero source read before its fill landed (R2) | `async_write_zeros` is asynchronous; bytes are undefined until the barrier (`noc.h:825-826`) | `wait_zero_source` precedes the first packet; BB-06 bit-identity is checked in the dirty-destination test |
| Writes in flight at kernel exit (R3, BB-03) | up to about 160 packets per core in flight with no intermediate barrier | `drain_writes` (`noc_async_write_barrier`) on every core that issued a write. **Posted writes are not allowed**: they carry no ack, so completion cannot be proven before the next program |
| Wrong bank count or holes | `NB` differs by arch (8 BH, 12 WH); `P mod NB != 0` makes banks unequal | the kernel derives `NB` from `NUM_DRAM_BANKS` and `n_b` from `P`; the host never sends `NB`; mapping math guards `S == 1` (no division by `S-1`) |
| bfloat8_b page size (R5) | 1088 B is not a power of two; 16384-B packets straddle pages | runs are byte ranges `run_slots * aligned_page_bytes`; packet boundaries are free; nothing assumes `page_bytes` divides `packet_bytes` |
| Stale per-shape args on a cache hit (R6, BB-07) | p2 and p3 can share one compiled program | `P` and ranges are RT values, which generic_op re-applies from the descriptor passed on every call. The host cache is keyed per shape, so one shape's descriptor is never patched for another |
| Cache-hit fast path freezes `P` and ranges | if the descriptor declares RT-arg buffer bindings (`emplace_runtime_args` with a `Buffer*`), a hit takes the binding fast path. That path patches only addresses (`mesh_device_operation_adapter.hpp:936-941`, `program_descriptor_patching.hpp:28-37`), so p3 would replay p2's `P` and ranges | pass the address as a plain common RT value (full re-apply path, `mesh_device_operation_adapter.hpp:958-962`). If bindings are ever used, `P` and the per-core ranges must enter the hash (`custom_program_hash` over shape, dtype, grid). The acceptance cache test (p3, p2, p3) checks every result |
| One device of the mesh only (R9) | the suite runs on a 1x1 mesh | allocate on `device` (the mesh) and dispatch SPMD over `MeshCoordinateRange(mesh_device->shape())` (`generic_op.cpp:21`); never build a single-coordinate `MeshProgramDescriptor` |
| Raw-address write annotation | `TensorAccessor::get_noc_addr` notes the tensor as read+write for op-to-op inference (`tensor_accessor.h:555`) | no DRAM read is issued; the annotation is static metadata only |
| Idle cores | `cores_used < C` when `P < C` (p4 on a 130-core grid) | `q_count == 0` returns before any NoC call; the kernel range stays the full rectangle (BB-17) |

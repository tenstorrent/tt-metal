# Operation Design: g6_zeros_like

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (creation: write-only, no arithmetic, no read) |
| Goal | Write +0.0 into every page of one DRAM-interleaved TILE tensor shaped like `input_tensor` (a new allocation, or the caller's `optional_tensor`) as one device program, without reading `input_tensor`'s data. |
| Math | `out[i] = +0.0` for every logical `i` (float32: all 32 bits 0; bfloat8_b: decodes to 0); `out = optional_tensor` if given, else a new tensor; `out` independent of `x`'s values |
| References | `docs/explorations/g6_zeros_like/{DOSSIER.md, math.md, tensix.md, risks.md, lessons.md}`; `eval/golden_tests/g6_zeros_like/{feature_spec.py, helpers.py, test_regression.py}`; `.claude/references/{blocking-model.md, l1-footprint-discipline.md}`; `ttnn/ttnn/operations/examples/master.md`; L1 ledger and traffic budget: `ttnn/ttnn/operations/g6_zeros_like/l1_ledger.md` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | device tensor; rank 4; TILE; `d_2 % 32 == 0`, `d_3 % 32 == 0`; volume > 0; DRAM interleaved; dtype in TARGET (`float32`, `bfloat8_b`) | — | metadata only (host): sets page count, page size, data format. Its buffer is never bound to the program. |
| `optional_tensor` (kw-only) | `ttnn.Tensor` or `None` | no | device tensor with `shape`, `dtype`, `layout`, `memory_config()` equal to `input_tensor`'s; may be `input_tensor` itself | `None` | its buffer address is the per-call RT value (see Program cache) |

No compute configuration and no other option exists (spec). Host-side knobs that are not public parameters are listed in the Blocking Model with their single source.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, d_1, d_2, d_3)`, rank 4, `d_2`, `d_3` multiples of 32, volume > 0 |
| Dtype | `float32` (Phase 0 SUPPORTED); `bfloat8_b` (SUPPORTED since Refinement 1) |
| Layout | TILE |
| Memory | DRAM interleaved |
| Data | never read — not by a kernel, not into L1, not on the host. The input's buffer is not in `io_tensors` and no TensorAccessor is built for it. |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `input_tensor.dtype` |
| Layout | TILE |
| Memory | `input_tensor.memory_config()` (DRAM interleaved) |
| Storage | `optional_tensor` itself when given (nothing allocated); else `ttnn.allocate_tensor_on_device(input_tensor.spec, device)` on the host before `generic_op` is enqueued (at trace capture this allocation happens at capture; replay allocates nothing) |
| Pages | `P = output_tensor.buffer_num_pages() = d_0 · d_1 · ceil(d_2/32) · ceil(d_3/32)`; page = one tile, `page_size = output_tensor.buffer_aligned_page_size()` (4096 B float32, 1088 B bfloat8_b) |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Production geometry: float32 `(1,12,128,128)` → P = 192 pages of 4096 B (48 tile rows × 4); bfloat8_b `(5,1,32,256)` → P = 40 pages of 1088 B (5 × 8). Worker grid on the target Blackhole board: 11 × 10 = 110 cores (`device.compute_with_storage_grid_size()`, probed); 8 DRAM banks; page `p` lives in bank `p mod 8`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `d_0` (batch) | independent — `out[i]` is the constant +0.0, no result spans it | folded into `page` | whole extent, inside the folded page range | `output_tensor.buffer_num_pages()` (host) | through `page` | none needed |
| `d_1` | independent — same | folded into `page` | whole extent, inside the folded page range | same | through `page` | none needed |
| `h_tiles` = `ceil(d_2/32)` | independent — same | folded into `page` | whole extent, inside the folded page range | same | through `page` | none needed |
| `w_tiles` = `ceil(d_3/32)` | independent — same | folded into `page` | whole extent, inside the folded page range | same | through `page` | none needed |
| `page` (folded, `tensor_pages = P`) | independent — every page is written from the same constant and needs nothing from any other page | `num_pages_this_core` — the block is the core's **whole** contiguous page range (`num_blocks_this_core = 1`) | float32: 2 pages on 82 cores, 1 page on 28 cores (110 cores); bfloat8_b: 1 page on 40 cores. This is the whole per-core assignment, i.e. the block at its bound: the block's L1 footprint is constant in its extent (one zero page serves every write), so no L1 limit ever argues for cutting it | host `_plan()` → per-core RT args `start_page`, `num_pages_this_core` | contiguous page ranges over `num_cores = min(P, grid_cores, NUM_CORES_CAP)` cores, column-major (`ttnn.split_work_to_cores(core_range_set, P, row_wise=ROW_WISE)`) | knob-turn (`NUM_CORES_CAP`, `ROW_WISE`, page order) |
| zero page (the only operand) | reuse-shared — it does not vary along any axis | `zero_scratch_bytes` | `page_size` (4096 / 1088 B) | CT arg `zero_scratch_bytes` = `ZERO_SCRATCH_PAGES · page_size` (host, one source for the CB size and the kernel's fill size); `page_size` = `output_tensor.buffer_aligned_page_size()` | made once per core, locally (no cross-core move; see Traffic ranking) | grows to `min(run_bytes, 16 KiB)` only in the deferred `bank_run_coalesced` regime |

The four tensor axes fold honestly into one: the op treats them identically and TILE interleaved pages are linear in `(d_0, d_1, h_tile, w_tile)` order. Splitting on the folded page index is what makes the core count the blocking decision's choice rather than the tensor's tile-row count (48 / 5).

Core-assignment knobs (host constants in the program-descriptor module, one source each):

| Knob | Phase 0 value | Effect |
|------|---------------|--------|
| `NUM_CORES_CAP` | `80` (Refinement 2; Phase 0 had `None` → 110) | upper bound on participating cores. Column-major 80 = one rectangle `(0,0)-(7,9)`: the whole segment east of DRAM column x=0 plus the two worker columns nearest DRAM column x=9. Float32 production measured flat over 70–90 cores (~2.05–2.12 µs) and ~8 % slower at 110: the far east columns start ~200–380 ns late and hold pages that drain last. bfloat8_b (P = 40) is unaffected (C = min(P, cap)). |
| `ROW_WISE` | `False` (column-major fill) | float32 uses all 110 cores = one rectangle `(0,0)-(10,9)`; bfloat8_b uses 40 cores = one rectangle `(0,0)-(3,9)` (BB-17: one core range per production shape) |
| `REGIME` | `"bank_run_coalesced"` (Refinement 2) | `bank_run_coalesced` or `page_split_local_zero`; the bank-run regime falls back to the page split when a bank in use would get no core or every run would be one page (`P <= C`, e.g. bfloat8_b 40 pages on 40 cores) |
| `BANK_ASSIGNMENT` | `"segment_rows"` (Refinement 2) | which core writes which bank's runs: `segment_rows` (default, see Regimes), `bank_major`, `stagger`, `nearest`; `segment_rows` falls back to `bank_major` when a band gets no core or more cores than its bank has pages |

Every block-dependent quantity (core ranges, per-core RT args, cache key) is derived from these and from `P`, `page_size` by `_plan()`; nothing restates them as literals.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_zero_page` | `zero_scratch_pages` = `max(1, min(ZERO_SCRATCH_PAGES, pages one write may cover))` (1 in the page split, the longest run in `bank_run_coalesced`) | 1 | Nothing to pipeline: the scratch is written once (zero fill) and afterwards only read, as the constant source of every NoC write of the core. In `bank_run_coalesced` it also sets the transfer size: a run is written in `zero_scratch_bytes` chunks. Measured (float32, 80 and 110 cores): 2 pages (8 KiB chunks) is ~90–150 ns slower than 1 page — the larger loopback fill on every core's critical path costs more than the fewer write commands save — so the knob stays at 1. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `Noc::async_write_zeros` overload 2 (DRAM, `noc.h:893`, `noc_zero_dram.inl:13-48`) | one call writes within **one** page: `offset_bytes + size_bytes <= aligned_page_size` (`noc_zero_dram.inl:24`) | one call per page, `size_bytes = page_size`, `offset_bytes = 0` | `--dev` ASSERT; in a plain build the bytes past the page land at the next in-bank offset — a page of another index (wrong pages zeroed, some never) |
| Same overload: source = pre-zeroed scratch **at its read pointer**, covering `min(size_bytes, NOC_MAX_BURST_SIZE)` bytes; the source address is not advanced between bursts (`noc_zero_dram.inl:26-47`) | `zero_scratch_bytes >= min(page_size, 16384)` (BH `NOC_MAX_BURST_SIZE = 256 · 64 B`, `noc_parameters.h:289-290`) | scratch = one full page (4096 / 1088 B ≤ 16 KiB) | garbage past the zeroed prefix streamed into DRAM |
| Zeroed bytes must sit at `cb_zero_page`'s read pointer, which overload 1 fills at the write pointer (the same-kernel scratch contract: a fresh CB has read ptr == write ptr, `noc.h:859-869`) | no credit moves on `cb_zero_page` | the kernel never calls `reserve_back` / `push_back` / `wait_front` / `pop_front` on it, so the pointers stay equal by construction (no consumer exists for a credit) | once the capacity grows past one page (deferred regime), stray credit moves the read pointer off the zeroed bytes and overload 2 streams unzeroed L1 |
| `Noc::async_write_zeros` overload 1 (L1 loopback from `MEM_ZEROS_BASE`, `MEM_ZEROS_SIZE = 512 B`; `noc_zero_l1.inl:13-44`, `dev_mem_map.h:88,115`) | destination 16 B aligned (`NOC_L1_READ_ALIGNMENT_BYTES`, `noc_parameters.h:375`); chunked into 512 B reads + one remainder read | CB base address (L1-aligned, 16 B); float32 = 8 reads of 512 B, bfloat8_b = 2 × 512 B + 1 × 64 B | misaligned loopback read faults / corrupts |
| Zero-fill completion before use (`write_zeros_l1_barrier`, `noc_zero_l1.inl:46`) | the first page write may not be issued before the barrier returns | one barrier after the fill, before `store_zero_block` | a write streams half-zeroed scratch to DRAM (nondeterministic garbage, BB-06) |
| BB-03: every NoC transaction completed at kernel return | write acks | one `write_zeros_dram_barrier()` (= `noc_async_write_barrier`, `noc_zero_dram.inl:51`) as the kernel's last act | the next program in the model hangs or sees in-flight writes |
| BB-05: L1 per core ≤ 1347776 B | `zero_scratch_bytes` | `page_size ≤ 4096 B` for every TARGET dtype: always fits, no host refusal needed | — |
| Per-dtype page bytes | `page_size` | read from the output buffer (`buffer_aligned_page_size()`), never `ttnn.tile_size(float32)` or 1024 | bfloat8_b written with 1024 B or 2048 B pages: exponent section unwritten / neighbouring page overrun (risk R1) |

### Regimes

Named boundary for "minimum": **DRAM — the input crosses 0 times (it is not an operand), the output crosses once as a write.** Cross-core minimum: 0 bytes.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `page_split_local_zero` | **built** | `REGIME == "page_split_local_zero"`, or the fallback of `bank_run_coalesced` (a bank in use gets no core, or `P <= C`): every TARGET cell can take it. bfloat8_b production (P = 40 on 40 cores) runs it. | the core's whole page range `[start_page, start_page + num_pages_this_core)`; source = one zero page in `cb_zero_page` | **minimum** at DRAM: input 0 B; output 1 write (786432 B float32, 43520 B bfloat8_b), each page written exactly once. Cross-core 0 B. Core-local: one `page_size` loopback fill per participating core (8 / 3 reads of the 512 B zero region). | The per-core fixed cost, paid once per core per call: launch message + RT-arg read, TensorAccessor construction, zero fill (8 or 3 loopback reads + 1 read barrier), and the final write barrier (one ack round trip from DRAM). More pages per core (fewer cores) amortize these over more pages but reduce concurrent writers — the grid-synchronization lamp. The per-page cost (one NoC write command) is not amortized by block size in this regime (cap: one page per overload-2 call). Domain: every TARGET cell; no other regime applies there because no axis is dependent (nothing to combine), the input is not an operand (no read regime), and every alternative below either moves more bytes on a costlier tier or is a transfer-shape refinement of this one. |
| `bank_run_coalesced` | **built** (Refinement 2, default) | `REGIME == "bank_run_coalesced"`, at least one core per bank in use, and `P > C` (some run ≥ 2 pages). Interleaved DRAM, `aligned_page_size` == in-bank page stride (TILE: true; the kernel ASSERTs `num_contiguous_pages` under `--dev`). | per core: one bank `b` and `r` consecutive in-bank pages `{b + B·k : k0 ≤ k < k0 + r}` (B = bank count; contiguous bytes `[addr(b + B·k0), + r·page)`), written in `zero_scratch_bytes` chunks from the one zero scratch | DRAM minimum (same bytes); writes per core = `ceil(r·page / zero_scratch_bytes)`; core-local fill `zero_scratch_bytes` per core | its real lever is **bank affinity, not fewer commands**: each core streams to one bank, and `segment_rows` places each bank's writers so that no two banks share a NoC link (below). Float32 production: 3.59 → 2.08 µs. |
| `injector_broadcast_zero` | **rejected** — superseded by `page_split_local_zero`: the zero page is a constant every core makes with core-local loopback reads (cheapest tier); receiving it instead adds `(C−1)·page_size` NoC payload + a semaphore per receiver (cross-core tier) and serializes every writer behind one sender | — | one zero page made on an injector, multicast to `C−1` cores | DRAM minimum; cross-core `(C−1)·page_size` B + `C` semaphore round trips above minimum | — |
| `compute_packed_zero` | **rejected** — superseded by `page_split_local_zero`: a compute kernel (zero/fill DEST + pack into a CB, then a writer) adds three TRISC binaries per core (BB-18 kernel-config pressure), a CB handshake and LLK inits per core, for a constant the NoC zero API produces with 8 local loopback reads; bfloat8_b exponent bytes would then depend on packer behaviour instead of being zero bytes by construction (risk R1) | — | one packed zero page per core | DRAM minimum; adds compute init + pack per core | — |
| `sub_page_writes_from_mem_zeros` | **rejected** — superseded by `page_split_local_zero`: writing each page straight from the 512 B `MEM_ZEROS` region removes the local fill but multiplies DRAM write commands by 8 (float32) / 3 (bfloat8_b) with sub-page transfers; catalog `tile_reorder`: whole-page writes are at least as fast as face-sized writes on a DRAM-bound move | — | 512 B pieces of a page | DRAM bytes minimum; 8× / 3× write transactions | — |
| `tile_row_split` | **rejected** — superseded by the folded `page` split: assigning whole tile rows (`w_tiles` pages each) caps the core count at the tensor's tile-row count (48 float32, 5 bfloat8_b), the shape-imposed limit the spec's cores rule forbids; same bytes | — | `w_tiles` pages per unit | DRAM minimum | — |
| `value_path_from_input` | **rejected (dead end)** — reads `x` (+1 DRAM crossing: 786432 B float32) and forms `x·0` / `x−x`: NaN where `x` is NaN/inf, −0.0 on sign-carrying paths; violates the contract and no good scheme passes through it | — | — | +1 read crossing of the input above minimum | — |

`segment_rows` (the bank-run placement). The writer is BRISC on NOC_1, whose packets go north (−y) and then west (−x). `get_optimal_dram_bank_to_logical_worker_assignment(device, NOC_1)` gives one anchor worker per bank: on this board, banks 0–3 → column 0 (east of DRAM column x=0), banks 4–7 → column 6 (east of DRAM column x=9; physical worker columns are 1–6 and 11–15). The anchor columns cut the grid into segments (`[0,6)` and `[6,11)`): a segment's cores write only that DRAM column's banks, so no westward write crosses another DRAM column. Inside a segment the anchor rows (NOC_1: 0, 2, 5, 8) cut it into bands: a bank takes the rows from its anchor row down to the next anchor row, so its packets climb only inside its band and turn west on its own endpoint row. At the shipped 80 cores: banks 0/1 get 12 cores (2 pages each), banks 2/3 get 18 (1–2 pages), banks 4/5 get 4 (6 pages), banks 6/7 get 6 (4 pages).

Measured placements (float32 production, 110 cores, median of 12): page split 3.65 µs; bank runs with `bank_major` (column bands) 2.44–2.53 µs; `stagger` 3.51 µs; `nearest` (Manhattan to the anchor, ignores routing direction) 3.16 µs; `segment_rows` with even row chunks 2.30 µs; `segment_rows` with endpoint-row bands 2.25–2.27 µs; the same at 70–90 cores 2.05–2.12 µs.

Regime-pinned tests: `test_plan_writes_every_page_once_on_many_cores` checks coverage for both production shapes; the float32 shape runs `bank_run_coalesced`, bfloat8_b runs `page_split_local_zero`.

### Traffic ranking

Counted per tensor and tier for the production shapes (C = participating cores). No nanosecond estimates.

| Rank | Candidate split | `x` DRAM crossings | `out` DRAM crossings | Cross-core | Core-local | Verdict |
|------|-----------------|--------------------|----------------------|------------|------------|---------|
| 1 | folded `page` split, one local zero page per core, full grid | 0 | 1 write | 0 | `C · page_size`: float32 110 · 4096 = 450560 B; bfloat8_b 40 · 1088 = 43520 B | **built** |
| 1 (tie at DRAM and cross-core) | bank-affine page runs | 0 | 1 write | 0 | `C · min(r·page, 16 KiB)` | built in Refinement 2 (`bank_run_coalesced`): ties on bytes, wins on bank affinity |
| 3 | tile-row split | 0 | 1 write | 0 | `min(rows, cores) · page_size` | rejected — core count set by the shape |
| 4 | injector + broadcast of the zero page | 0 | 1 write | `(C−1)·page_size` + `C` semaphores | 1 fill | rejected |
| — | split a dependent axis + combine | — | — | — | — | not applicable: no result spans any axis, there is nothing to combine |
| 6 | value path through `x` | 1 read | 1 write | 0 | — | rejected (contract) |

Chosen: rank 1. It is the cheapest at the DRAM and cross-core tiers. Its only above-minimum term is core-local (one fill per core), which shrinks with fewer cores — that trade is the grid-synchronization lamp, accounted separately from this ranking.

**Operand-reuse check** (every operand × the chosen split): the zero page does not vary along `page` → reuse-shared by construction. Within a core: one fill, reused by every page write (realized). Across cores: the broadcast alternative (rank 4) costs a cross-core payload to save a core-local fill — rejected. `input_tensor` is not an operand.

**Stall-shadow check**: (a) `write_zeros_l1_barrier` waits on the loopback reads — in its shadow go the RT-arg reads, the output TensorAccessor construction and the NoC address of the core's first page (independent of the scratch's contents); no page write may enter it. (b) The final write barrier waits on DRAM acks — nothing remains independent of it (every write is already issued; it is the kernel's last act). No reorder changes floating-point order (there is no arithmetic).

### Block schedule

Logical schedule of the single writer kernel (BRISC, NoC1) on every core of `all_cores`:

```cpp
zero_page_fill();                                         // once per core per call
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // num_blocks_this_core = 1
    store_zero_block(block_idx);                          // page split: one write per page; bank run: store_zero_bank_run, chunked writes of one in-bank run
}
drain_writes();                                           // one write barrier, the kernel's last act
```

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs | Profiler zone (BB-21) |
|-----------------|------------------------|--------------------|-----------------------------------|-----------------------|
| `zero_page_fill` | one page of `page_size` bytes in `cb_zero_page` | — (produces the resident zero page) | once per core per call: `ceil(page_size / 512)` loopback reads (8 float32, 3 bfloat8_b) + one read barrier | `g6zl-zero-page` (fill issue, the core's RT-arg reads and output accessor construction in its shadow, then its barrier) |
| `store_zero_block` | `num_pages_this_core` pages `[start_page, start_page + num_pages_this_core)` of the output | the zero page (read-only source for every write) | one NoC write command per page (mechanism cap); no barrier, no CB handshake, no init inside the block | `g6zl-issue-writes` |
| `store_zero_bank_run` (`bank_run_coalesced`) | `num_pages_this_core` in-bank pages `start_page + B·k` = one contiguous byte range of one bank | the zero scratch (read-only source of every chunk) | one NoC write per `zero_scratch_bytes` chunk; no barrier, no CB handshake, no init inside the block | `g6zl-issue-writes` |
| `drain_writes` | all writes of the core | — | once per core per call: one write barrier | `g6zl-write-barrier` |

There is no compute kernel, no reader kernel and no CB handshake. Zones use `DeviceZoneScopedN` and compile out of the plain build. The block operation names above are how the implementer structures the kernel; how each is realized (helper call vs. raw API) is in API Mapping.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (`NUM_CORES_CAP`) | Full occupancy leaves ≤ 2 pages per core (float32) and 1 (bfloat8_b); the per-core fixed cost (launch, zero fill, final barrier) is of the order of the per-core payload, and the writes are DRAM-bound (`double_buffer`: no gain once DRAM-bound, ~64 cores reach peak). Fewer cores also cut the core-local fill traffic (`C · page_size`). | `NUM_CORES_CAP` ∈ {110, 96 (2 pages each), 48 (4), 24 (8), 8 (one per DRAM bank)} for float32; {40, 20, 8} for bfloat8_b. Judge on trace-replay host time and model step, not on profiled kernel duration alone (lesson RB-053: cores that end at different times read slower under the profiler). |
| Zero-fill mechanism | The fill sits on each core's critical path before its first write. NoC loopback (8 × 512 B reads + barrier) is the default; RISC stores may be shorter for one page. | `dataflow_kernel_lib::fill_l1_range<4>(addr, page_size, 0)` (`l1_helpers.hpp:101`) — 1024 / 272 word stores, no barrier. |
| Page order (bank stagger) | Contiguous 2-page ranges make every two-page core's first write land on an even bank (start pages 0, 2, 4, … → banks 0, 2, 4, 6) and its second on an odd bank. | Strided assignment (core `i` writes pages `i, i + C, …`) so the first wave covers all 8 banks (`bank_stagger`: 1.04–1.32× on short workloads). Host-only change: per-core page list. |
| Placement / NoC | Writer on BRISC/NoC1 is the catalog default for writes (`noc_placement`). For `P < grid_cores` (bfloat8_b: 40 cores) the column-major block `(0,0)-(3,9)` concentrates the writers in four columns. | `ROW_WISE = True` (40 cores → 3 full rows + 7 cores = 2 rectangles; BB-17 cost of the second range included in the judgement); writer on NoC0. |
| Second writer RISC | Only matters if a core's writer becomes issue-bound, i.e. after the grid-synchronization lamp moves to many pages per core (`split_reader`: no effect unless the DM RISC is the bottleneck). | BRISC/NoC1 + NCRISC/NoC0 each taking half the core's pages from the same zero page (second RISC waits on a one-page CB handshake for the fill, or fills its own page). |
| Stateful page writes | One full command setup per page write; negligible at ≤ 2 pages per core. | With many pages per core: `noc_async_write_one_packet_set_state` / `_with_state` (`dataflow_api.h:982,1008`) — constant source and size, only the destination changes. |
| Host program-hash walk | Untraced calls hash the cached descriptor every call (`generic_op_device_operation.cpp:78-140`). | Set `custom_program_hash` on the cached descriptor from the full cache key (key must include the op name and every program-changing value — BB-07). |

### Catalog evidence

| Entry | Informed which knob |
|-------|---------------------|
| `width_split` | split on the folded page index, never on tile rows (core count not set by the shape) |
| `double_buffer` | no buffer depth: write-only and DRAM-bound at grid scale; issue every write of a core, then one barrier |
| `tile_reorder` | whole-page writes (reject sub-page writes from `MEM_ZEROS`) |
| `noc_placement` | writer on BRISC/NoC1; `ROW_WISE` lamp |
| `bank_stagger` | page-order lamp |
| `split_reader` | second-writer-RISC lamp is conditional on an issue-bound writer |
| `trid_double_issue` | not applicable (no read stream; a single end-of-kernel write barrier) |
| lessons RB-036, RB-053 | one rectangle per production shape (column-major fill); judge the core-count lamp on unprofiled trace replay |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `MEM_ZEROS` (firmware zero region, 512 B, every core's L1) → `cb_zero_page` | raw zero bytes | NoC loopback reads, `Noc::async_write_zeros` overload 1 via `dataflow_kernel_lib::zero_tile` | all-zero bytes are +0.0 in float32 and decode to 0 in bfloat8_b (exponents 0, signs 0, mantissas 0) — one mechanism, no per-dtype code |
| `cb_zero_page` → output pages in DRAM | one page = one tile, `page_size` bytes (Float32 4096 B / Bfp8_b 1088 B, all zero) | `Noc::async_write_zeros` overload 2 per page through `TensorAccessor(output)` | BRISC on NoC1 (`ttnn.WriterConfigDescriptor()`); each page written once |
| `input_tensor` | — | not bound to the program | metadata (shape, dtype, layout, memory config) is read on the host only |

Placement per TARGET scheme: TARGET has no `memory_layout` axis; DRAM interleaved is the only placement (both production cases). Sharded and L1 placements are refused by `validate()` with `ValueError` (spec, Validation). No physical-shard regime exists; none is designed.

Regimes needing inter-Tensix communication: none (`bank_run_coalesced` is core-local).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output page (tile); the block is the core's whole contiguous page range |
| Grid | `device.compute_with_storage_grid_size()` (11 × 10 = 110 on the target board), restricted to the first `min(P, grid_cores, NUM_CORES_CAP)` cores column-major (`ttnn.num_cores_to_corerangeset(n, grid, ROW_WISE)`) |
| Split | `ttnn.split_work_to_cores(core_range_set, P, row_wise=ROW_WISE)` → `(num_cores, all_cores, group_1, group_2, pages_g1, pages_g2)`; group 1 cores get `pages_g1 = ceil`, group 2 `pages_g2 = floor` |
| Per-core work | `start_page` = prefix sum over cores in the split's fill order (`ttnn.corerange_to_cores(group, row_wise=ROW_WISE)`, group 1 then group 2); `num_pages_this_core` ∈ {`pages_g1`, `pages_g2`} |
| float32 `(1,12,128,128)` | P = 192 → `bank_run_coalesced` on 80 cores, one rectangle `(0,0)-(7,9)`; per bank (24 pages) the `segment_rows` bands listed under Regimes; runs of 1–6 pages |
| bfloat8_b `(5,1,32,256)` | P = 40 → 40 cores × 1 page; kernel on one rectangle `(0,0)-(3,9)`; 70 cores carry no kernel |
| Remainder | the ragged split is the `group_1` / `group_2` page counts; P itself is per-image alignment-aware by construction (`buffer_num_pages()` = `d_0·d_1·ceil(d_2/32)·ceil(d_3/32)`), never `floor(volume / 1024)` |
| Idle cores | cores outside `all_cores` get no kernel (no empty launches) |
| Regime selection | `REGIME` knob + `_bank_run_split` predicate (see Core-assignment knobs); `describe_plan` reports the selected regime per entry |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_zero_page` | 0 | `page_size` (= `output_tensor.buffer_aligned_page_size()`) | `zero_scratch_pages` = 1 | live set = 1 page: spans no block axis; streams over `page` (every page write of the core reads the same page). Constant in `num_pages_this_core` and in the core count. | output dtype (`Float32` / `Bfp8_b`); a dataflow-only payload that never passes through DEST | writer (zero fill) | writer (NoC write source) | whole kernel |

`cb_zero_page` is a private L1 scratch of the writer kernel: one kernel both fills and reads it, and no credit ever moves on it (see Mechanism caps). No other kernel binds it. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `zero_page_fill` | 1 page, `page_size` B (`zero_scratch_bytes`) | the two calls of `dataflow_kernel_lib::zero_tile(cb)`, issued apart so the stall-shadow work (a) fits between them | — (source: `MEM_ZEROS`) | `cb_zero_page`, 1 page at its write pointer (= read pointer) | holds `page_size` zero bytes; not pushed; read ptr == write ptr |
| 2 | `store_zero_block` | `num_pages_this_core` pages from `start_page` | device API (no kernel_lib helper covers it; see API Mapping) | `cb_zero_page`, 1 page, read at its read pointer, never popped | — (output tensor in DRAM) | unchanged |
| 3 | `drain_writes` | all writes of the core | device API | — | — | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `zero_page_fill` | raw_api (the body of `zero_tile`, split) | `Noc::async_write_zeros(cb, zero_scratch_bytes)`, then RT-arg reads + `TensorAccessor` construction, then `Noc::write_zeros_l1_barrier()` — `zero_tile` fuses issue and barrier, which leaves no room for the stall-shadow check (a) | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:52-57`; `tt_metal/hw/inc/api/dataflow/noc.h:848-849,904`; `tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl:13-46` | `CircularBuffer cb(cb_zero_page)` (`circular_buffer.h:23`; accepted endpoint `circular_buffer.h:259`); byte count = `cb.get_tile_size()` (`circular_buffer.h:113` → `dataflow_api.h:281`) = the CB's tile size for its data format = `page_size` (4096 / 1088, probed) — the kernel ASSERTs `cb.get_tile_size() == page_size` under `--dev` | — | `cb_zero_page` | none (size = one page) |
| `store_zero_block` | raw_api | `Noc::async_write_zeros(accessor, page_size, {.page_id = p}, cb)` for `p` in `[start_page, start_page + num_pages_this_core)` | `tt_metal/hw/inc/api/dataflow/noc.h:851-897`; `tt_metal/hw/inc/internal/noc_zero_dram.inl:13-48` | `TensorAccessor` over the output: `TensorAccessorArgs<cta_offset>()` (`tensor_accessor_args.h:21`), `TensorAccessor(args, out_addr, page_size)` (`tensor_accessor.h:71-96`); dst args `{page_id, offset_bytes = 0}` (`tt_metal/hw/inc/api/tensor/noc_traits.h:141-142`); scratch `cb` read at its read pointer | `cb_zero_page` | — | `num_pages_this_core`, `start_page` (RT) |
| `store_zero_bank_run` | raw_api | `Noc::async_write(cb_zero_page, output, chunk, {}, {.page_id = start_page, .offset_bytes = off})` for `off` in steps of `zero_scratch_bytes` over `num_pages · page_size` | `tt_metal/hw/inc/api/dataflow/noc.h:411`; contiguity: `TensorAccessor::contiguous_page_stride` / `num_contiguous_pages` (`tensor_accessor.h:560-575`) | CT `bank_run_stride` (host bank count; 0 = page split), ASSERTed equal to `contiguous_page_stride()`; the zeros overload is not used because it refuses a write that crosses a page (`noc_zero_dram.inl:24`) | `cb_zero_page` | — | `num_pages_this_core`, `start_page` (RT); `zero_scratch_bytes` (CT) |
| `drain_writes` | raw_api | `Noc::write_zeros_dram_barrier()` (= `noc_async_write_barrier`) | `noc.h:911`; `noc_zero_dram.inl:51`; `dataflow_api.h:1951` | — | — | — | none |

**Helpers considered and rejected — `store_zero_block`:**

| Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|---------------------------------|
| `dataflow_kernel_lib::prepare_zero_tile(dfb)` | `l1_helpers.hpp:66-71` (reserve at :68, push at :70) | writes nothing to DRAM — it is a `zero_page_fill` candidate, not a store. For the fill it adds a reserve and a push whose credit no kernel ever consumes (the scratch has no second thread), so `zero_tile` (`l1_helpers.hpp:52-57`), the same fill without credit, is used instead |
| `dataflow_kernel_lib::write_sticks_after_untilize<cb_id>(accessor, rows, row_bytes, start_page)` | `tilize_helpers_dataflow.hpp:96-135` | consumes a CB of untilized row-major sticks by wait/pop (one stick per page of a stick-indexed accessor); this op writes whole TILE pages from a constant, never-popped scratch |
| `dataflow_kernel_lib::read_with_state` / `set_read_state` | `local_copy_helpers_dataflow.hpp:60-117` | local L1 → L1 reads; no DRAM write |
| (gap) | — | no kernel_lib helper writes a constant page to a range of interleaved DRAM pages. The device API `Noc::async_write_zeros` overload 2 is the dedicated mechanism (its own TODO, `noc_zero_dram.inl:10-11`, names the page-range overload that the deferred `bank_run_coalesced` regime would need). |

**`drain_writes`** is a single barrier; no helper wraps it.

## Entry Point, Registry and Program Cache

Registry (op file `ttnn/ttnn/operations/g6_zeros_like/g6_zeros_like.py`; names match `feature_spec.py` TARGET):

| Declaration | Value |
|-------------|-------|
| `INPUT_TAGGERS` | `{"dtype": tag_dtype}`; `tag_dtype(inputs, axes) -> inputs[0]["dtype"]` (`inputs = (scenario,)`, `scenario = {"input_shape", "dtype", "with_optional_tensor"}`) |
| `SUPPORTED` | `{"dtype": [ttnn.float32, ttnn.bfloat8_b]}` (bfloat8_b since Refinement 1) — the implementer owns this block; bfloat8_b needed no kernel or host change beyond adding it here (same zero bytes, page size read from the buffer) |
| `EXCLUSIONS` | `[]` |
| Forbidden | any tagger beyond `tag_dtype`; a size, rank, layout or call-form axis; an EXCLUSIONS entry; refusing a call because `optional_tensor` is given |

`validate(input_tensor, *, optional_tensor=None)` — the entry point's first line — checks in this order:

| # | Check | Raises |
|---|-------|--------|
| 1 | `input_tensor` is a `ttnn.Tensor` on device (`ttnn.is_tensor_storage_on_device`) | `ValueError` |
| 2 | layout TILE; rank 4; `shape[-1] % 32 == 0` and `shape[-2] % 32 == 0`; volume > 0 | `ValueError` |
| 3 | `memory_config()` is INTERLEAVED in DRAM (not sharded, not L1) | `ValueError` |
| 4 | if `optional_tensor` is not None: it is a device tensor and its `shape`, `dtype`, `layout`, `memory_config()` equal `input_tensor`'s | `ValueError` |
| 5 | build `scenario` from `input_tensor` / `optional_tensor`, apply `INPUT_TAGGERS`; each axis value in `SUPPORTED` | `UnsupportedAxisValue` |
| 6 | `EXCLUSIONS` (empty) | `ExcludedCell` |

Entry point `g6_zeros_like(input_tensor, *, optional_tensor=None)`:

| Step | Action |
|------|--------|
| 1 | `validate(...)` |
| 2 | `output = optional_tensor` if given, else `ttnn.allocate_tensor_on_device(input_tensor.spec, input_tensor.device())` |
| 3 | fetch the cached `ProgramDescriptor` for the cache key (build on miss via `_plan()` + descriptor builder); write `output.buffer_address()` into the writer's **common** runtime arg 0 |
| 4 | `ttnn.generic_op([output], descriptor)` — `output` is the only io tensor (a generator; `make_tensor_args` accepts a single tensor). The input is never in the list, so `optional_tensor is input_tensor` creates no aliased binding. |
| 5 | return `output` (the caller's `optional_tensor` object when given) |

Program cache (BB-07, BB-16, spec rule 4):

| Item | Decision |
|------|----------|
| Per-call value | output buffer address only — a common runtime arg (same for every core). Never a CT arg, define or part of the kernel source. On a framework cache hit `apply_descriptor_runtime_args` copies the call's common RT args into the cached program (`tt_metal/impl/program/program_descriptors.cpp:188-212`; reached from `mesh_device_operation_adapter.hpp:958-962`), so each call writes the buffer it names. |
| Host cache | a module-level dict: key → built `ProgramDescriptor`. Key = `(P, page_size, dtype, grid.x, grid.y, num DRAM banks, NUM_CORES_CAP, ROW_WISE, REGIME, BANK_ASSIGNMENT, WRITER_PROCESSOR, WRITER_NOC, ZERO_SCRATCH_PAGES)` — every value the descriptor depends on. A repeat call does not re-run `_plan()`, `split_work_to_cores` or descriptor construction; it patches the address word and dispatches. If the binding returns copies of `kernels`, hold the `KernelDescriptor` and reassign `descriptor.kernels = [writer]` after patching. |
| Program-fixed values | CT args: `cb_zero_page` index, `page_size`, `TensorAccessorArgs(output)`; per-core RT args: `start_page`, `num_pages_this_core`; CB descriptor on `all_cores`; kernel on `all_cores`. |
| Plan report (BB-08, BB-21) | `describe_plan(input_tensor) -> list[dict]` exported from the op package: one entry per participating core `{"core": (x, y), "role": "writer", "regime": r, "start_page": s, "num_pages": n, "page_stride": d, "bank": b}` (the core writes pages `s + k·d`, `k < n`; `bank` is `None` in the page split); cores not listed carry no kernel. Structural checks only (steps 1-3 of `validate`), no SUPPORTED gate, so it reports the plan for every TARGET dtype. The descriptor builder consumes the same `_plan()` result. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Stale output buffer on a cache hit / trace replay | the program is cached by shape and dtype; the model calls with other buffers | address is a common RT arg patched every call; acceptance test writes two different `optional_tensor`s and a fresh output in one process |
| Half-zeroed or non-zero scratch streamed to DRAM | the fill is an async loopback; overload 2 reads at the CB's read pointer | `write_zeros_l1_barrier` returns before the first page write is issued; no credit ever moves on `cb_zero_page` (read ptr == write ptr) |
| Pages left unwritten | a fresh buffer may already hold zeros; `optional_tensor` holds NaN | `_plan()` covers `[0, P)` exactly once (acceptance test checks `describe_plan` coverage); tests prefill `optional_tensor` with NaN and poison a freed block |
| −0.0 or a nonzero bfloat8_b element | any arithmetic or pack path can carry a sign / exponent | the only bytes written are copies of the firmware zero region — no arithmetic, no pack |
| Wrong bfloat8_b page size | bfloat8_b tile = 1024 B mantissas + 64 B exponents = 1088 B | `page_size` from `output.buffer_aligned_page_size()`; CB format = output dtype; ASSERT `cb.get_tile_size() == page_size` |
| Input read by accident | a template reader or `io_tensors = [x, out]` makes reading easy | no reader kernel, no input accessor, `io_tensors = [output]` |
| Writes in flight at kernel return | the next model program starts right after (BB-03) | one write barrier as the last act of the kernel |
| Shape-limited core count | tile-row split gives 48 / 5 cores | split on the folded page index |
| Scattered core ranges (BB-17) | ragged splits can produce several ranges per group | one kernel on `all_cores`; column-major fill gives one rectangle for both production shapes; per-core counts differ only in RT args |
| `optional_tensor is input_tensor` | aliasing an input and an output binding | input is never bound; writing zeros into `x` is the specified result |
| Zero-volume tensor | no page to write; an empty program cannot satisfy "one dispatch" | `validate()` raises `ValueError` (outside the rank-4 tile-aligned production domain) |

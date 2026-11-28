# Operation Design: g6_zeros

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (generator: no input tensor, no compute kernel) |
| Goal | Allocate a new DRAM-interleaved TILE tensor of `shape`/`dtype` on every device of `device` and write the all-zero bit pattern into every page in **one** device program, writing each output page exactly once and reading no DRAM. |
| Math | `out[i] = +0.0` (all-zero bit pattern of `dtype`) for every logical index `i` of `shape` |
| References | `eval/golden_tests/g6_zeros/feature_spec.py` (TARGET/INPUTS/LOOSE_CASES), `eval/golden_tests/g6_zeros/helpers.py` (exact bit-pattern gate), `tt_metal/hw/inc/api/dataflow/noc.h:849,893` (Noc zero APIs), `tt_metal/hw/inc/internal/noc_zero_dram.inl:10-11` (TODO: "a page-range overload could zero a bank at a time" — this design is that page-range path), `tt_metal/hw/inc/api/tensor/tensor_accessor.h:560-575` (interleaved bank contiguity API), catalog `ttnn/ttnn/operations/examples/master.md` entries `noc_placement`, `double_buffer`, `tile_reorder`, `read_coalescing`, `bank_stagger`, `split_reader` |

**Why the all-zero byte pattern is correct for every TARGET dtype.** float32 / bfloat16: all bytes 0 is +0.0. bfloat8_b: a tile is 64 B of shared exponents followed by 1024 B of sign+mantissa bytes; all bytes 0 gives exponent 0 and sign/mantissa 0, i.e. +0.0 for every datum. So the op is **dtype-agnostic in its kernel**: the dtype enters only through the aligned page size (4096 / 2048 / 1088 B) and therefore through the burst/run byte counts.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `shape` | 4 ints (list or tuple), positional | yes | rank 4, every `d_i >= 1`, `d_2 % 32 == 0`, `d_3 % 32 == 0` | — | host only (folds into `tensor_num_pages`) |
| `dtype` | `ttnn.DataType` (kw-only) | yes | TARGET: float32, bfloat16, bfloat8_b. Phase 0 SUPPORTED: float32 | — | host only (sets `aligned_page_size`) |
| `layout` | `ttnn.Layout` (kw-only) | yes | `ttnn.TILE_LAYOUT` | — | host only |
| `device` | `ttnn.MeshDevice` or single device (kw-only) | yes | open device / mesh device | — | host only |
| `memory_config` | `ttnn.MemoryConfig` (kw-only) | yes | DRAM interleaved | — | host only (accessor CT args) |

No compute config, no `pin_memory`/`names`/`requires_grad`.

### Registry model (op file)

| Item | Value |
|------|-------|
| `INPUT_TAGGERS` | exactly `{"dtype": tag_dtype, "layout": tag_layout}`; each reads `inputs[0]["dtype"]` / `inputs[0]["layout"]` of the scenario dict `{"shape", "dtype", "layout"}` |
| `SUPPORTED` (Phase 0) | `{"dtype": [ttnn.float32], "layout": [ttnn.TILE_LAYOUT]}` — written by the implementer |
| `EXCLUSIONS` | `[]` — and it stays empty: no size/shape/rank axis, no other tagger, no entry refusing a production case |
| `validate(shape, *, dtype, layout, device, memory_config)` | first line of `g6_zeros`. Order: (1) structural checks raising `ValueError` — shape is 4 positive ints with `d_2, d_3` multiples of 32; layout is TILE; memory_config is DRAM + INTERLEAVED; device is an open `ttnn.MeshDevice`/device (raise `RuntimeError`/`ValueError` otherwise); (2) build `scenario = {"shape": tuple(shape), "dtype": dtype, "layout": layout}`, run INPUT_TAGGERS on `(scenario,)`; (3) SUPPORTED per axis → `UnsupportedAxisValue`; (4) EXCLUSIONS → `ExcludedCell` |
| Structural impossibilities | none (INVALID empty is correct: every TARGET dtype is storable in tiles) |

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
| Shape | `shape` (per device), `(d_0, d_1, d_2, d_3)` |
| Dtype | `dtype` (made in `dtype`; no cast) |
| Layout | TILE_LAYOUT (32×32 tiles; one page = one tile) |
| Memory | `memory_config` (DRAM interleaved), **newly allocated in every call** with `ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, device, memory_config)` (positional args) on every device of `device` |

`tensor_num_pages = d_0 · d_1 · ceil(d_2/32) · ceil(d_3/32)` (per-image `ceil`, never `floor` of a flattened product; equals `output.buffer_num_pages()`, which is the value the host uses).

| Case | shape | dtype | pages | aligned page | bytes / device |
|------|-------|-------|-------|--------------|----------------|
| p2 | (16,12,128,128) | float32 | 3072 | 4096 | 12 582 912 |
| p3 | (8,12,128,128) | float32 | 1536 | 4096 | 6 291 456 |
| p1 | (32831,1,32,256) | bfloat8_b | 262 648 | 1088 | 285 761 024 |
| p4 | (1,4,256,128) | bfloat16 | 128 | 2048 | 262 144 |

## Blocking Model

The op is one write of `tensor_num_pages` zero pages. Every page is independent of every other page and of any input, so the only real decisions are (a) which core owns which pages, (b) how many bytes one NoC write carries, and (c) how often a core synchronizes. The page index is re-expressed in the coordinates that DRAM interleaving actually stores it in: page `p` lives in bank `b = p mod B` at bank-row `r = p div B`, and bank-rows `r, r+1, …` of one bank are **contiguous bytes** (`aligned_page_size` apart; `tensor_accessor.h:560-575`). That re-expression is what lets a core's whole assignment be one contiguous DRAM byte range.

Symbols: `B` = DRAM bank count of the interleaved allocator (host: `ttnn.GetMemoryView(device, ttnn.BufferType.DRAM).num_banks`; device: `accessor.contiguous_page_stride()` == `NUM_DRAM_BANKS`); `G` = cores of `device.compute_with_storage_grid_size()`; `aps` = `output.buffer_aligned_page_size()`; `N` = `tensor_num_pages`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `d_0`, `d_1`, tile-rows `ceil(d_2/32)`, tile-cols `ceil(d_3/32)` | independent — no element depends on another or on any input | folded into the page index `p` (the op treats them identically and interleaved pages are numbered contiguously in this order) | whole extent, folded | `output.buffer_num_pages()` | via the bank / bank-row axes below | — (folding is exact; never needs un-folding) |
| bank `b` (extent `B`) | independent — banks hold disjoint pages | `B` (not a tunable: a property of the allocator) | `B` (all banks) | host `num_banks` query, checked against `accessor.contiguous_page_stride()` in-kernel (`ASSERT`) | each core owns **exactly one bank**: core ordinal `k` → bank `k % B` (adjacent cores in row-major order hit different banks — a built-in bank stagger) | — |
| bank-row `r` within bank `b` (extent `n_b = ceil((N-b)/B)`) | independent — rows are disjoint pages | `run_pages` = this core's contiguous bank-rows; `cores_per_bank` = how many cores split one bank's rows | `cores_per_bank = min(floor(G/B), ceil(N/B))`; `run_pages` = even split of `n_b` over `cores_per_bank` (first `n_b % cores_per_bank` slots get one more); **block = the whole run** (coarsest: one pipeline pass per core) | host: `cores_per_bank` is one host variable; per-core `(first_page, run_pages)` derived from it as RT args | core ordinal `k` → slot `k // B` of bank `k % B` | `cores_per_bank` is a knob-turn (grid-sync lamp) |
| burst (bytes per NoC write inside a run) | independent | `burst_pages` | `max(1, floor(MAX_WRITE_BURST_BYTES / aps))` = 4 (fp32), 8 (bf16), 15 (bfp8) — the widest write one zero source of ≤16 KiB can feed | host constant `MAX_WRITE_BURST_BYTES = 16384` (Blackhole `NOC_MAX_BURST_SIZE`, `noc_parameters.h:290`); `burst_pages` derived; `zero_source_bytes = burst_pages·aps` derived from it; passed as CT arg | n/a (inside a run) | knob-turn (`burst_pages = 1` is the per-page fallback; lamp) |
| mesh device (extent = devices of `device`) | independent — each device holds its own full per-device tensor | whole mesh | every device runs the same program (SPMD via one `ProgramDescriptor` given to `ttnn.generic_op`) | `device` | all devices | — |

All knobs are parameters; no page count, core count, bank count or burst size is a literal in a kernel loop bound.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_zero_source` | none — it is a constant source, filled once and never consumed | 1 entry of `zero_source_bytes` | Nothing to overlap: the source never changes, so every write of the run reads the same bytes and the NoC keeps all writes of the run in flight with no CB handshake. A depth >1 would buy nothing (catalog `double_buffer`: "Skip all of it if you're already bandwidth-bound"; here there is no producer to overlap with). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Zero source size vs. one NoC write | `burst_pages · aps` bytes per write | `burst_pages = max(1, floor(16384 / aps))`; zero source sized `burst_pages · aps` | a raw `noc_async_write` advances its **source** with the destination: a write longer than the zeroed source streams uninitialised L1 (garbage, not zeros) into DRAM — wrong results, silently |
| Bank contiguity of a run | `run_pages` contiguous only within one bank and only when the accessor's page size equals the allocator's aligned page size | each core's run is bank-rows of a **single** bank; host passes `aps = buffer_aligned_page_size()` as the accessor page size; kernel `ASSERT(accessor.num_contiguous_pages(first_page, first_page + (run_pages-1)·B + 1) == run_pages)` (`tensor_accessor.h:568-575`) | crossing into another bank or using the unaligned size writes zeros into the wrong pages and leaves the intended ones stale (dirty-destination case catches it) |
| Host `B` vs device bank count | the bank axis | kernel `ASSERT(B == accessor.contiguous_page_stride())` | wrong page coverage (some pages never written) |
| L1 local zero-fill (`Noc::async_write_zeros` overload 1) | zero-source alignment | CB base is 16-B aligned by the allocator (`noc_zero_l1.inl:21` asserts it) | assert / misaligned loopback read |
| Model L1 bound (BB-05) | total per-core L1 | `zero_source_bytes ≤ 16384 ≪ 1 347 776` | — (cannot be reached) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 bank-affine coalesced runs** | **built** | every call (`N ≥ 1`) | one run of `run_pages` contiguous bank-rows of one bank per core, written in `ceil(run_pages / burst_pages)` NoC writes of `≤ burst_pages·aps` bytes from one 16 KiB zero source | **minimum**: output crosses DRAM exactly once as a write of every page; 0 DRAM reads; 0 cross-core traffic; local L1 traffic = one loopback zero-fill of `zero_source_bytes` per core | Fixed costs per core: kernel launch, one L1 zero-fill + `write_zeros_l1_barrier`, one final `noc_async_write_barrier`, RT-arg fetch — all **once per core per call** because the block is the whole run. Fixed cost per NoC write (header/issue, DRAM row-open) paid `ceil(run_pages/burst_pages)` times — a bigger burst pays it fewer times, up to the 16 KiB packet cap |
| R0 page-range split, one write per page (`split_work_to_cores` over `N`, `noc.async_write_zeros` overload 2 per page) | rejected — superseded by R1 on every shape | — | contiguous page-id range per core; one NoC write per page | same DRAM bytes (minimum) but `N` NoC transactions instead of `N / burst_pages` (4× more for fp32, 15× more for bfp8) and every core sprays all `B` banks | It is the same kernel as R1 with `burst_pages = 1`, `B`-strided runs replaced by stride-1 ranges; R1 contains it as a knob setting (`burst_pages = 1`), so writing it separately adds code and no capability |
| R2 zeros straight from `MEM_ZEROS_BASE` (no CB) | rejected — superseded by R1 | — | as R1 | minimum DRAM bytes, but `MEM_ZEROS_SIZE = 512` B per write (`dev_mem_map.h:88`): 32× more transactions than a 16 KiB burst | — saving 16 KiB of L1 out of a 1.35 MB budget does not pay for 32× the write count |
| R3 host-made zeros transferred / read from a DRAM zero buffer | rejected — forbidden by the spec (reads DRAM / host transfer) and doubles DRAM crossings | — | — | 2× minimum (read + write) | — |
| Dependent-axis split | n/a — the op has no dependent axis and no reuse-shared operand (no input) | — | — | — | — |

Single built regime: R1 covers every legal call (any rank-4 tile-aligned shape, any TARGET dtype) because the only shape-dependent quantities (`N`, `n_b`, `run_pages`) are runtime extents of the same block operation; no shape needs a different loop nest.

### Traffic ranking

| Candidate split | DRAM crossings (output) | DRAM reads | Cross-core | NoC transactions | Rank |
|-----------------|-------------------------|------------|------------|------------------|------|
| R1: bank × bank-row, coalesced runs | 1 (write) | 0 | 0 | `≈ N / burst_pages` (+≤1 per core for the ragged tail) | **1 — chosen** |
| R0: page-id range per core | 1 | 0 | 0 | `N` | 2 (same bytes, more/smaller transactions, every core touches every bank) |
| R2: write from 512 B hardware zeros | 1 | 0 | 0 | `N · aps / 512` | 3 |
| R3: copy from a zero buffer | 1 write + 1 read | 1 | 0 | `2N` | rejected (forbidden) |

All legal splits move the minimum bytes; R1 wins on transfer shape (fewest, largest, bank-local transactions — catalog `tile_reorder`, `read_coalescing`: larger coalesced transfers reach higher achieved DRAM bandwidth). No operand exists, so the operand-reuse check is vacuous. **Stall-shadow check:** the only waits are the one-time L1 zero-fill barrier (nothing to schedule into it — every write needs the zeroed source) and the final write barrier (nothing remains after it; it is required by BB-03).

### Block schedule

Per core (logical schedule; one data-movement kernel, no compute kernel, no reader):

```cpp
// block = this core's whole run: run_pages contiguous bank-rows of bank (core % B)
fill_zero_source();                       // once per core per call
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // num_blocks_this_core = 1 (0 if run_pages == 0)
    store_zero_run(block_idx);            // ceil(run_pages / burst_pages) writes, no barrier between them
}
drain_writes();                           // once per core per call
```

| Block operation | Acts on | Resident across it | Intended frequency of fixed costs |
|-----------------|---------|--------------------|-----------------------------------|
| `fill_zero_source` | `cb_zero_source`, `zero_source_bytes` | — | 1 local loopback zero (`zero_source_bytes / 512` reads from `MEM_ZEROS_BASE`) + 1 `write_zeros_l1_barrier` per core per call |
| `store_zero_run` | `run_pages · aps` contiguous DRAM bytes in bank `b`, starting at `accessor.get_noc_addr(first_page)` | `cb_zero_source` (read-only) | 1 address computation per run; 1 `noc_async_write` per burst; **no** barrier / flush inside the run (the source never changes, so no write-flushed wait is needed) |
| `drain_writes` | all writes this core issued | — | 1 `noc_async_write_barrier` per core per call (BB-03: nothing in flight at kernel exit) |

`num_blocks_this_core` is 1 by design (block = whole per-core assignment; the zero source never limits how much of the run can be in flight). Writing it as a loop keeps the block extent a parameter: a later lamp that splits a run (e.g. across two RISCs) changes only the per-core RT args.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (cores used) | p4 (128 pages) and p3 (1536 pages) give 1 and 12 pages per core at `cores_per_bank = 16`; launch + RT-arg + barrier fixed cost may dominate the 2–48 KiB a core writes | `cores_per_bank ∈ {1, 2, 4, 8, floor(G/B)}` per case (knob-turn: host only) |
| Burst size | 16 KiB bursts are the packet cap; DRAM controller page/row behaviour may favour a smaller multiple of the page | `burst_pages ∈ {1, 2, floor(16384/aps)}` |
| Writer count per core | one RISC (BRISC, NoC1) issues every write; for p1 (~2050 bfp8 pages/core at 16 cores/bank) issue rate could bind (catalog `split_reader`: helps only if issue-bound) | split each core's run in two halves: second kernel on NCRISC/NoC0 (catalog `noc_placement`: writes on NoC0 can be 2.5–4.8× slower for spread placements — measure, do not assume) |
| Core placement | ordinal `k → bank k % B` in row-major grid order ignores which DRAM column each bank sits on (catalog `noc_placement`, `bank_stagger`) | bank-near placement (cores of bank `b` chosen nearest that bank's DRAM endpoint) vs row-major interleaved. **Refinement 2 (measured): `placement="bank_near"` + `cores_per_bank=1` is the default** (p1 1 092 860 → 559 495 ns, ≈511 GB/s, at the DRAM write ceiling); see `l1_ledger.md` |
| Overlap | n/a — no compute and no CB handshake; nothing to overlap. Declared empty for that reason | — |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| zeros made in L1 | raw bytes (all 0) | `Noc::async_write_zeros(cb_zero_source, zero_source_bytes)` (local NoC loopback from `MEM_ZEROS_BASE`) + `write_zeros_l1_barrier()` | no DRAM read, no host data |
| L1 → DRAM | raw bytes = `burst_pages` zero pages of `dtype` | `noc_async_write(zero_src_l1_addr, run_noc_addr + offset, bytes)` per burst on the writer's NoC (NoC1, `WriterConfigDescriptor` default) | each output page written exactly once; destination run computed once via `TensorAccessor::get_noc_addr(first_page)` |
| completion | — | `noc_async_write_barrier()` | before kernel return |

TARGET has no `memory_layout` axis: placement is fixed to DRAM interleaved, so no physical-shard scheme exists. The other TARGET values (bfloat16, bfloat8_b) are **knob-turns**: the identical kernel and descriptor with a different `aps` (and hence `burst_pages`); the refinement adds them to SUPPORTED only.

Mesh: the single `ttnn.ProgramDescriptor` passed to `ttnn.generic_op([output], descriptor)` runs SPMD on every device of the mesh; the mesh output buffer has the same address on every device, so the same RT/common args are correct everywhere.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one run = `run_pages` contiguous bank-rows of one bank (one block) |
| Grid | `device.compute_with_storage_grid_size()` (`G` cores); cores used `= B · cores_per_bank`, `cores_per_bank = min(floor(G / B), ceil(N / B))`; they are the first `B·cores_per_bank` cores in **row-major** order (full grid rows + at most one partial row → ≤ 2 rectangular `CoreRange`s, BB-17; row-wise spreading per catalog `noc_placement`) |
| Per-core work | core ordinal `k`: bank `b = k % B`, slot `s = k // B`; `n_b = (N - b + B - 1) // B` (0 if `b ≥ N`); `q, rem = divmod(n_b, cores_per_bank)`; `run_pages = q + (s < rem)`; `row_start = s·q + min(s, rem)`; `first_page = row_start · B + b` |
| Remainder | ragged across banks (`n_b` differ by ≤1) and across slots (`run_pages` differ by ≤1), handled by the formulas above; a core with `run_pages == 0` (only possible when `N < B·cores_per_bank`, i.e. `n_b < cores_per_bank` for the last banks) fills nothing and writes nothing, then returns. Last burst of a run is `run_pages − (bursts−1)·burst_pages` pages |
| RT args (per core) | `first_page`, `run_pages` (derived on host from `cores_per_bank`) |
| Common RT args | output buffer address (the **only** value that changes between calls of the same shape) |
| CT args | `B`, `burst_pages`, `aps`, `TensorAccessorArgs(output)` (at the end) |
| Plan reporting (BB-08) | the op exposes the selected plan (`cores_per_bank`, cores used, `burst_pages`, `B`) e.g. via a `g6_zeros_plan(shape, dtype, device)` helper; there is one regime, so this is informational |

Host caching (BB-07, BB-16): the host builds the `ProgramDescriptor` once per key `(device identity, N, dtype, aps, B, G, cores_per_bank, burst_pages)` and stores it in a module-level dict; on a repeat call it only (1) allocates the new output tensor and (2) rewrites the writer kernel's `common_runtime_args = [output.buffer_address()]`, then calls `ttnn.generic_op`. Set `descriptor.custom_program_hash` to a stable 64-bit digest of `("g6_zeros", key)` so the C++ side skips the descriptor walk; the digest must include every element of the key (the BB-07 cache-order case runs p3, p2, p3 and requires no rebuild on the repeat and no wrong replay).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_zero_source` | 0 | `zero_source_bytes = burst_pages · aps` | 1 | live set spans the burst (one write's source); streams over bank-row (re-read for every burst) and bank (one per core); bounded by the 16 KiB cap | output `dtype` (never unpacked — dataflow-only payload) | writer (zero-fill) | writer (NoC write source) | whole kernel |

Only one CB. The writer is both filler and reader of its own scratch in the same thread; it never does `reserve/push/wait/pop` on it — it is a raw L1 scratch obtained via `get_write_ptr` (fresh CB: read ptr == write ptr). No compute kernel and no reader kernel exist.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `fill_zero_source` | `zero_source_bytes` | yes (Noc zero API) | — | `cb_zero_source`, 1 page, zeroed | holds zeros for the rest of the kernel; no push |
| 2 | `store_zero_run` | `run_pages · aps` bytes = `ceil(run_pages/burst_pages)` bursts | raw (see API Mapping) | `cb_zero_source`, read-only | DRAM output | unchanged |
| 3 | `drain_writes` | — | raw barrier | — | — | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `fill_zero_source` | helper (device API) | `Noc::async_write_zeros(dst, size_bytes)` overload 1 + `Noc::write_zeros_l1_barrier()` | `tt_metal/hw/inc/api/dataflow/noc.h:849`, `:904`; impl `tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl:13-44` | `dst = CircularBuffer(cb_zero_source)` (`noc_zero_l1_endpoint_v<CircularBuffer>`, `circular_buffer.h:259`), `size_bytes = zero_source_bytes` | — | `cb_zero_source` | `zero_source_bytes` (derived from `burst_pages`) |
| `store_zero_run` | raw_api | `TensorAccessor::get_noc_addr(first_page)` once, then `noc_async_write(src_l1, dst_noc, bytes)` per burst | `tensor_accessor.h:554` (interleaved `get_noc_addr`), `dataflow_api.h:876` | `src_l1 = cb_zero_source` base; `dst_noc = run_addr + i·burst_pages·aps`; `bytes = min(burst_pages, remaining)·aps` | `cb_zero_source` | DRAM | `burst_pages`, `run_pages` |
| contiguity guard | raw_api (dev-only `ASSERT`) | `TensorAccessor::contiguous_page_stride()`, `num_contiguous_pages(page_id, end)` | `tensor_accessor.h:562`, `:568` | `ASSERT(stride == B)`, `ASSERT(num_contiguous_pages(first_page, first_page+(run_pages-1)·B+1) == run_pages)` | — | — | — |
| `drain_writes` | raw_api | `noc_async_write_barrier()` | `dataflow_api.h:1951` | — | — | — | — |

**Helpers considered and rejected — `store_zero_run`:**

| Candidate | Mismatch (file:line) | Why it cannot express the block |
|-----------|----------------------|---------------------------------|
| `Noc::async_write_zeros(accessor, size, {page_id}, scratch)` overload 2 | `tt_metal/hw/inc/internal/noc_zero_dram.inl:24` `ASSERT(args.offset_bytes + size_bytes <= accessor.get_aligned_page_size())`; TODO at `:10-11` | one call is limited to one page, so the run would degrade to one transaction per page (regime R0). The planned block is a multi-page bank-contiguous run; the missing helper is exactly the "page-range overload" the TODO names. With `burst_pages = 1` the implementer MAY use overload 2 per page (it is then equivalent). |
| `dataflow_kernel_lib::zero_tile` / `prepare_zero_tile` (for `fill_zero_source`) | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:53`, `:67` | takes a `DataflowBuffer` and zeros `get_tile_size()` — usable if the implementer wraps `cb_zero_source` in a `DataflowBuffer` whose entry size is `zero_source_bytes`; it calls the same `noc.async_write_zeros` + `write_zeros_l1_barrier`. Either form is accepted; `prepare_zero_tile` is NOT (its `push_back` would hand the entry to a consumer that does not exist). |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A freshly allocated buffer already holds zeros | a do-nothing or partial-coverage op passes a clean-buffer test | acceptance test and golden `dirty_destination` cases poison the exact range first; the coverage formulas (`n_b`, `run_pages`) must sum to `N` — host asserts `Σ run_pages == N` when building the plan |
| Source shorter than a write | raw `noc_async_write` advances the source with the destination | `bytes ≤ burst_pages · aps = zero_source_bytes` for every write, enforced by construction (`min(burst_pages, remaining)`) |
| Wrong contiguity assumption | runs are only contiguous within one bank and with the aligned page size | single-bank runs; accessor built with `aps`; dev `ASSERT`s on `contiguous_page_stride()` and `num_contiguous_pages` |
| Writes in flight at exit (BB-03) | the model starts another program immediately | one `noc_async_write_barrier()` at the end of every kernel that issued a write |
| Rebuilding the descriptor every call (BB-07) / incomplete key | cache-order loose case (p3, p2, p3) | module-level descriptor cache keyed as in Work Distribution; only `common_runtime_args[0]` is patched per call; `custom_program_hash` covers the whole key |
| Reusing a cached output (spec rule 2) | caching the descriptor tempts caching the tensor | the output is allocated fresh in every call; only the descriptor is cached |
| Mesh: writing only device 0 | single-device tests cannot see it | one SPMD descriptor through `ttnn.generic_op` on the mesh tensor; never index a single device |
| Negative zero | bit-exact gate rejects -0.0 | bytes come from `MEM_ZEROS_BASE` (all 0x00); nothing is computed |
| Profiling (BB-21) | time outside zones reads as waiting | `DeviceZoneScopedN` zones `fill_zero_source`, `store_zero_run`, `drain_writes` in the writer; no effect on the plain build |
| Idle cores with `run_pages == 0` | only for `N < B·floor(G/B)` | `cores_per_bank = min(floor(G/B), ceil(N/B))` keeps empty cores out of the core ranges except the ≤1-page-per-bank tail |

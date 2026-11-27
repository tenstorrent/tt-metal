# Operation Design: g6_zeros

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (creation: write-only, no input, no compute kernel) |
| Goal | Allocate a new DRAM-interleaved TILE tensor of the given per-device shape and dtype on every device of `device`, and write the all-zero bit pattern into every page of it, as one device program that reads nothing from DRAM. |
| Math | `out[i] = +0.0` (all-zero bits of `dtype`) for every logical index `i`; `out` is new storage on every call |
| References | `docs/explorations/g6_zeros/DOSSIER.md` (index), `math.md`, `tensix.md`, `risks.md`, `lessons.md`; `eval/golden_tests/g6_zeros/feature_spec.py`, `eval/golden_tests/g6_zeros/helpers.py`; `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`; catalog entries `noc_placement`, `double_buffer`, `tile_reorder`, `split_reader`, `bank_stagger`, `read_coalescing` in `ttnn/ttnn/operations/examples/master.md` |

Catalog entries and the knob each one informed:

| Entry | Knob it informed | What it says that applies here |
|-------|------------------|--------------------------------|
| `read_coalescing` | block extent on the bank-slot axis | pages contiguous in one bank go out as one wide call; the API splits the bursts |
| `tile_reorder`, `double_buffer` | transfer size (`zero_source_bytes`) | larger coalesced transactions reach higher DRAM bandwidth; with enough cores the op is DRAM-bound, so buffer depth buys nothing |
| `noc_placement` | core order, writer NoC | writes on NoC1 (the writer default); spread cores over rows (`row_wise`) |
| `split_reader` | lamp L3 (second DM RISC) | a second DM RISC helps only when one RISC is issue-bound |
| `bank_stagger` | lamp L4 | only matters when every core walks every bank in the same order. Phase 0 gives each core one bank, so it does not apply |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `shape` (positional) | list or tuple of 4 `int` | yes | each `d_k >= 1`; `d_2 % 32 == 0`, `d_3 % 32 == 0` | none | host only: sets `tensor_pages` and the split (per-core RT args, common RT arg `tensor_pages`) |
| `dtype` (keyword-only) | `ttnn.DataType` | yes | Phase 0 `ttnn.float32`; TARGET adds `ttnn.bfloat16`, `ttnn.bfloat8_b` | none | CT: `page_bytes` (4096 / 2048 / 1088) |
| `layout` (keyword-only) | `ttnn.Layout` | yes | `ttnn.TILE_LAYOUT` | none | host only |
| `device` (keyword-only) | `ttnn.MeshDevice` (or a single device in a test) | yes | open | none | host only: grid, DRAM bank count, SPMD target |
| `memory_config` (keyword-only) | `ttnn.MemoryConfig` | yes | DRAM + INTERLEAVED (`ttnn.DRAM_MEMORY_CONFIG`) | none | CT: `TensorAccessorArgs(output)` |

No compute configuration exists. `pin_memory`, `names`, `requires_grad` are not parameters.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | none. The op takes no input tensor and reads no tensor data |
| Dtype | n/a |
| Layout | n/a |
| Memory | n/a |

### Output

| Property | Value |
|----------|-------|
| Shape | `shape` (per device), on every device of `device` |
| Dtype | `dtype` (made in `dtype`; there is no cast) |
| Layout | TILE (a page is one 32x32 tile: 4096 B float32, 2048 B bfloat16, 1088 B bfloat8_b) |
| Memory | DRAM interleaved, newly allocated in every call by `ttnn.allocate_tensor_on_device` (one mesh buffer, same address on every device) |
| Pages | `tensor_pages = d_0 * d_1 * ceil(d_2/32) * ceil(d_3/32)`, read as `output.buffer_num_pages()` |
| Tile padding | none in any production shape. Its content is not part of the result |

## Blocking Model

The op has no arithmetic, no input and no dependency between elements. Its whole cost is the
transfer shape of one write of every output page. In an interleaved DRAM tensor, page `p` lies in
bank `p mod num_dram_banks` at bank slot `p div num_dram_banks`, and consecutive slots of one bank
are adjacent in memory, `aligned_page_size` apart (`tensor_accessor.h:560-574`). So the blocking
works in the **(bank, bank_slot)** space. A block is one contiguous DRAM region: one bank, a run of
consecutive slots.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `d_0`, `d_1`, `d_2/32` (tile rows), `d_3/32` (tile columns) | independent: every element is the constant +0.0, so no result spans any of them | folded into the page index `p` (row-major tile order). The op treats the four identically and they are contiguous, so separate extents would change no transfer | whole extent, folded | `output.buffer_num_pages()` (host) | through the bank axes below | none needed |
| `bank` (DRAM bank of a page, extent `num_dram_banks`) | independent: pages of different banks share nothing | `block_banks` (bank extent of one block) | **1**, a mechanism cap: one contiguous transfer cannot cross a bank (Mechanism caps, row 1). How many banks one core visits is `banks_per_core`, a core-assignment knob | `block_banks`: constant 1 (cap). `banks_per_core`: host constant, Phase 0 = 1 | **bank-affine**: core `c` of the writer set owns bank `c mod num_dram_banks` | knob-turn (`banks_per_core`, lamp L4) |
| `bank_slot` (slot of a page in its bank, extent `bank_pages(b) = ceil((tensor_pages - b) / num_dram_banks)`) | independent: same reason | `block_slots` | **whole per-core assignment** (`core_slots`): the block is not L1-resident, so nothing bounds it below the assignment, and a larger block removes one address setup and one partial-burst tail per extra block | host split → per-core RT arg | the cores of bank `b`'s group split `bank_pages(b)` into balanced contiguous slot ranges | knob-turn |
| `device` (chip of the mesh) | independent: each device holds its own per-device tensor | whole per-device tensor | one program per device | `ttnn.ProgramDescriptor` (SPMD over the mesh) | every device runs the same program on its own copy of the mesh buffer | none |

Derived per-core quantities, each computed on the host from the knobs above and never restated:

| Quantity | Definition |
|----------|------------|
| `grid_cores` | `device.compute_with_storage_grid_size()`: `x * y` |
| `max_writer_cores` | host constant, Phase 0 = `grid_cores` (lamp L1) |
| `num_writer_cores` | `min(max_writer_cores, tensor_pages)`: the first `num_writer_cores` cores of the grid in **row-major** order (`row_wise`) |
| `core_bank(c)` | `c mod num_dram_banks` (lamp L2 replaces this map, RT args only) |
| `core_rank(c)` | `c div num_dram_banks` |
| `bank_group_size(b)` | `ceil((num_writer_cores - b) / num_dram_banks)` |
| `core_slots(c)` | `bank_pages(b) div G + (rank < bank_pages(b) mod G ? 1 : 0)`, with `b = core_bank(c)`, `G = bank_group_size(b)` |
| `core_slot_begin(c)` | `rank * (bank_pages(b) div G) + min(rank, bank_pages(b) mod G)` |
| `num_blocks_this_core` | `banks_per_core` (= 1 in Phase 0) |
| `run_pages` (per block) | `min(block_slots, accessor.num_contiguous_pages(first_page, tensor_pages))`, `first_page = slot_begin * num_dram_banks + bank` (the kernel clips to the tensor end through the contiguity API) |
| `run_bytes` | `run_pages * page_bytes` |
| `zero_source_bytes` | `min(ZERO_SOURCE_MAX_BYTES, max over writer cores of run_bytes)`; `ZERO_SOURCE_MAX_BYTES = 16384` (one host constant, = `NOC_MAX_BURST_SIZE` on Blackhole, `noc_parameters.h:287-290`) |
| `writes_per_block` | `ceil(run_bytes / zero_source_bytes)` |

`num_writer_cores <= tensor_pages` and `bank_group_size(b) <= bank_pages(b)` for every bank (both are
monotone in the same formula), so no writer core is assigned zero pages.

Per production case (illustration: 130-core grid, 8 banks. The real values come from the device at
run time):

| Case | `tensor_pages` | `page_bytes` | `bank_pages` | `num_writer_cores` | `core_slots` (G = 16 / G = 17) | `run_bytes` (max) | `zero_source_bytes` | writes per core (max) |
|------|---------------:|-------------:|-------------:|-------------------:|-------------------------------|------------------:|--------------------:|----------------------:|
| p1 (32831,1,32,256) bfloat8_b | 262648 | 1088 | 32831 | 130 | 2051-2052 / 1931-1932 | 2232576 | 16384 | 137 |
| p2 (16,12,128,128) float32 | 3072 | 4096 | 384 | 130 | 24 / 22-23 | 98304 | 16384 | 6 |
| p3 (8,12,128,128) float32 | 1536 | 4096 | 192 | 130 | 12 / 11-12 | 49152 | 16384 | 3 |
| p4 (1,4,256,128) bfloat16 | 128 | 2048 | 16 | 128 | 1 / n/a | 2048 | 2048 | 1 |

Every block knob is a parameter. No loop bound in any kernel is a literal: the kernel loops
`block_idx < num_blocks_this_core` and `written < run_bytes` in steps of `zero_source_bytes`.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_zero_source` | none: a read-only scratch, not a FIFO | 1 page of `zero_source_bytes` | nothing. Every in-flight write reads the same zero bytes, so the NoC keeps all writes of a run in flight off one page with no CB handshake. Overlap comes from issuing writes back to back with one barrier at the end, not from depth |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Interleaved placement: one bank's slots are contiguous, different banks are different DRAM endpoints | `block_banks` | `block_banks = 1`. A block is one bank's run | a write longer than the bank-local run lands on the next pages of the **same** bank (`p + num_dram_banks`, ...): it overwrites pages owned by another core, past the tensor end, or both |
| Zero source size: each NoC write reads its whole size from `cb_zero_source` | bytes per write call | `chunk = min(zero_source_bytes, run_bytes - written)` | the write streams L1 bytes past the zeroed region (stale data) into DRAM: wrong values |
| Tensor end (last slot row is partial: bank `b` holds `bank_pages(b)` pages) | `run_pages` | `min(block_slots, accessor.num_contiguous_pages(first_page, tensor_pages))` (`tensor_accessor.h:568-574`) | writes past the last page of the tensor into unrelated DRAM |
| `num_contiguous_pages` returns 1 unless the accessor's page size equals the interleaved aligned page size (`tensor_accessor.h:568-573`) | `run_pages` | pass the **aligned** page size (`output.buffer_aligned_page_size()`) as `page_bytes`. 1088, 2048 and 4096 are multiples of the 64 B DRAM alignment (`core_config.h:42`), so aligned = actual | `run_pages` stays 1: one write per page (correct but the rejected per-page regime) |
| Host and kernel must agree on the bank count | the bank axis | host `num_dram_banks = device.dram_grid_size().x * .y`, passed as a CT arg; the kernel `ASSERT`s it equals `NUM_DRAM_BANKS` (which `contiguous_page_stride()` uses, `tensor_accessor.h:562`) | the host groups pages by the wrong bank: runs are not contiguous and some pages are never written |
| `NOC_MAX_BURST_SIZE` (16384 B on Blackhole) | packet size | none needed: `noc_async_write` splits a larger call into burst packets (`dataflow_api.h:876`). `zero_source_bytes <= 16384` keeps one packet per call | n/a (correctness holds at any `zero_source_bytes`) |
| L1 zero fill: NoC loopback read, destination 16 B aligned (`noc.h:827-829`) | `cb_zero_source` base | CB base addresses are aligned. `zero_source_bytes` is a multiple of 64 | an unaligned destination ASSERTs under `--dev` |
| DRAM write alignment 64 B | chunk start and size | the run start is page-aligned and every chunk is `zero_source_bytes` (a multiple of 64) except the tail, `run_bytes mod zero_source_bytes`, also a multiple of 64 | misaligned DRAM writes |

No accumulator, DEST or fixed-width index cap applies. 32-bit offsets hold: the largest bank-local
extent is p1's `32831 * 1088 = 35719328` B.

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 bank-run writes from a per-core zero source**: each writer core zeroes `cb_zero_source` once by L1 loopback, then writes each of its blocks (one bank, `block_slots` consecutive slots = one contiguous DRAM region) as `ceil(run_bytes / zero_source_bytes)` NoC writes from that one source | **built** | every call: every shape, every TARGET dtype (the kernel handles bytes only. `page_bytes` is the only dtype input) | `block_banks (1) x block_slots (= core_slots)` | **minimum**: `out` crosses DRAM once (every page written once), no DRAM read, no cross-core traffic. Added core-local traffic: one loopback fill of `zero_source_bytes` per writer core per call | per block: one run-start address (`get_noc_addr`) and one partial-burst tail. Per call and not per block: one zero fill + L1 barrier and one final write barrier per core. At `block_slots = core_slots` each core pays one address setup and at most one partial packet |
| R2 per-page writes: one NoC write per page from a one-page zero source (includes `Noc::async_write_zeros` overload 2, which stays inside one page, `noc_zero_dram.inl:24`) | **rejected**: R1 supersedes it. Same bytes, but `tensor_pages` transactions (262648 at p1, 1088 B each) where R1 issues about `sum ceil(run_bytes/16384)` (about 17.6 K at p1). A dead end: R1 replaces its per-page loop rather than building on it | always | 1 page | minimum bytes, 15x (bf8) to 4x (fp32) more transactions than R1 | n/a |
| R3 compute-engine zeros: a compute kernel fills DEST, packs tiles into an output CB, and a writer drains them | **rejected**: R1 supersedes it. Same DRAM bytes, plus unpack/math/pack work per page and an output CB pipeline. For bfloat8_b the packer computes shared exponents (risk R5) | always | `k` tiles | minimum DRAM. Adds per-page L1 write and read through the CB | n/a |
| R4 hardware zero region as the direct write source (`MEM_ZEROS_BASE`, 512 B, `dev_mem_map.h:88`), no scratch CB | **rejected**: R1 supersedes it. Each write is capped at 512 B: 32x the packets of R1's 16 KiB writes on long runs. Saving one fill of at most 16 KiB L1 does not pay for that | always | 512 B chunks | minimum bytes, 32x transactions | n/a |
| R5 zero source read from a DRAM zero buffer | **rejected**: spec rule 4 forbids DRAM reads, and it adds one DRAM read per byte written | always | n/a | 2x DRAM traffic | n/a |
| R6 broadcast zero source: one injector core fills zeros and multicasts them to the writer cores | **rejected**: R1's core-local loopback fill supersedes it. Each core makes its own zeros from its own `MEM_ZEROS` region in at most 32 local reads, while a multicast moves `zero_source_bytes` per receiver over the NoC and adds a semaphore round trip | always | n/a | adds cross-core traffic | n/a |

R1 is the only regime. It covers the whole domain: every shape of TARGET (rank 4, tile-aligned)
and every dtype, because every page is the same zero bytes and only `page_bytes` differs. No other
regime applies there. The op has no dependent axis, so there is no "split the dependent axis +
combine" alternative. Its only shared operand is the zero bytes, which R1 already generates on each
core (R6 covers the broadcast). No selection predicate exists, so no regime-pinned tests are needed.

### Traffic ranking

Every candidate split writes each output page once to DRAM, reads nothing and moves nothing between
cores. **All tie at the minimum** in bytes per tier. So the ranking is decided by transfer shape
(size and count of NoC writes) and fan-out, which the blocking model keeps as a tie-breaker:

| Rank | Candidate split | DRAM bytes | Cross-core | Transfer shape | Fan-out |
|------|-----------------|-----------|------------|----------------|---------|
| 1 (chosen) | **bank-affine**: each core owns one bank and one contiguous slot range in it (`banks_per_core = 1`) | 1 write of `out` | 0 | one contiguous run per core, issued in 16 KiB writes. p2: 6 x 16 KiB per core. p3: 3 x 16 KiB | 1 bank per core. `bank_group_size` writers per bank |
| 2 | slot-range across all banks: contiguous page range per core (the `split_work_to_cores` shape), coalesced per bank (`banks_per_core = num_dram_banks`) | 1 write | 0 | `num_dram_banks` runs per core, each `1/num_dram_banks` as long. p2: 8 x 12 KiB. p3: 8 x 4-8 KiB. More, smaller writes and 8 partial tails per core | every core to every bank. Needs `bank_stagger` to avoid all cores starting on bank 0 |
| 3 | page-range split, one write per page | 1 write | 0 | `tensor_pages` writes of one page (rejected regime R2) | every core to every bank |

Rank 2 is the same kernel with a different RT-arg assignment (`banks_per_core`). It is lamp L4,
not a regime, because it changes no structure.

Stall-shadow check: the scheme has two waits per core. (1) The L1 zero-fill barrier: every DRAM
write depends on it, so the only work that can run in its shadow is address setup (accessor
construction, `first_page`, run length). The schedule does that before the barrier. (2) The final
write barrier: nothing remains to run. No stage computes on one core while others wait. There is no
designated core.

### Block schedule

Per writer core (one writer lane, BRISC on NoC1), per call:

```cpp
make_zero_source();                                    // once per core
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    write_bank_run_block(block_idx);                   // one contiguous DRAM region
}
drain_writes();                                        // once per core
```

| Operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------|------------------------|--------------------|-----------------------------------|
| `make_zero_source` | `zero_source_bytes` of `cb_zero_source` | n/a (it creates the resident zero source) | one L1 loopback zero call and **one** `write_zeros_l1_barrier` per core per call. Address setup for the first block is issued before the barrier (stall shadow) |
| `write_bank_run_block` | `block_banks (1) x block_slots` pages = `run_bytes` contiguous bytes at `get_noc_addr(first_page)`, `first_page = core_slot_begin * num_dram_banks + bank`, `bank = (first_bank + block_idx) mod num_dram_banks` | `cb_zero_source` (read-only, never re-zeroed) | one run-address computation per block. `writes_per_block` NoC writes issued back to back with **no** barrier between them. No CB handshake |
| `drain_writes` | all writes of the core | n/a | **one** `noc_async_write_barrier` per core per call (BB-03: no transaction in flight at kernel return) |

No unit-at-a-time loop in the main schedule: a block's writes walk bytes in `zero_source_bytes`
steps inside `write_bank_run_block`, never page by page.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 **grid-synchronization**: `max_writer_cores = grid_cores` | the op is DRAM-write-bound. About 16 writers per bank may only add NoC contention at the DRAM endpoints, and p4 leaves 1 page per core, so per-core start cost is the whole kernel | `max_writer_cores = num_dram_banks * g` for `g` in {1, 2, 4, 8}. Judge on unprofiled trace-replay host time, not profiled kernel time (lesson RB-053: cores ending at different times skew profiled durations) |
| L2 core-to-bank map: round-robin `c mod num_dram_banks` | a bank's group is scattered over the grid, so each write crosses up to the whole chip. On Blackhole the banks sit in two NoC columns (x = 0 and x = 9, `blackhole_140_arch.yaml:11-21`), and the NoC1 west-then-north route from a core west of x = 9 to a middle-column bank wraps around the torus | DRAM-adjacent groups: bank `b`'s cores are the writer cores nearest its NoC1 endpoint (RT args only, still one rectangle of writer cores) |
| L3 one writer lane per core (BRISC/NoC1) | per-lane demand is low (about 2.6 GB/s at p1 on 130 cores), so issue is not the bound. But routes to the middle DRAM column are shorter on NoC0 for cores west of it | a second lane on NCRISC/NoC0 with its own `cb_zero_source_noc0` (no cross-RISC handshake): either split each core's run in half, or pick per core the NoC with the shorter route to its bank. The second option needs two kernel groups, so check BB-17 core-range counts |
| L4 `banks_per_core = 1` | the catalog's `read_coalescing` found bank affinity not universally better than plain coalescing (0.91-1.28x) | `banks_per_core = num_dram_banks`, each core starting at bank `core mod num_dram_banks` (`bank_stagger`) |
| L5 write issue path: generic `noc_async_write` (multi-packet capable, non-posted) | every call is at most one burst and goes to the same bank endpoint, so per-call command-buffer programming repeats work | one-packet stateful writes (`Noc::set_async_write_state` / `async_write_with_state`, `noc.h:542-630`) or the `max_page_size = NOC_MAX_BURST_SIZE` one-packet template. Posted writes only with a BB-03 argument that they have landed before return |
| L6 `zero_source_bytes = min(16384, max run_bytes)` | larger calls do not reduce packets, but they do reduce API calls | `zero_source_bytes` in {32768, 65536} on p1 (L1 cost stays at most 64 KiB) |

Overlap lamp: does not apply. There is no producer/consumer pipeline. The only serialization is
the one-time fill (at most 32 loopback reads of 512 B) before the first write, and address setup
already sits in its shadow.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| H1 validate | n/a | Python, first line of the entry point | argument checks raise `ValueError`, then SUPPORTED raises `UnsupportedAxisValue`, then EXCLUSIONS (Key Risks, row "validation order") |
| H2 allocate | n/a | `ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, device, memory_config)` | new storage every call, on every device of the mesh. Never cached, never reused |
| H3 describe | n/a | `ttnn.ProgramDescriptor`, cached on the host by key | a cache hit patches only the output buffer address (common RT arg). See Work Distribution |
| H4 dispatch | n/a | `ttnn.generic_op([output], descriptor)` | one program. A `ProgramDescriptor` (not a `MeshProgramDescriptor`) runs the same program SPMD on every device of the mesh |
| D1 make zeros | raw bytes in L1 (`cb_zero_source`) | `Noc::async_write_zeros` overload 1 (NoC loopback read from the core's own `MEM_ZEROS_BASE`) + `write_zeros_l1_barrier` | the bytes are undefined until the barrier returns (`noc.h:825-826`). The barrier precedes the first DRAM write (risk R2) |
| D2 write runs | TILE pages as raw bytes, `page_bytes` each, all zero | `noc_async_write` from `cb_zero_source` to `accessor.get_noc_addr(first_page) + written` | the format never matters: an all-zero page is +0.0 in float32 and bfloat16, and +0.0 in bfloat8_b (zero shared exponents, zero sign and magnitude bytes) |
| D3 drain | n/a | `noc_async_write_barrier` | BB-03 |

No reader kernel, no compute kernel, no inter-core communication. The placement axis
(`memory_layout`) is not a TARGET axis: TARGET is DRAM interleaved only. A DRAM-sharded output
would be a knob-turn: the shard replaces the bank as the contiguous unit and the kernel writes one
shard's run per block. It is not designed further.

Refinements toward TARGET: `bfloat16` and `bfloat8_b` are SUPPORTED-list changes. The kernel and
host split have no dtype branch. Only `page_bytes` and the derived `zero_source_bytes` change.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: one bank's run of `block_slots` consecutive slots (Phase 0: one block per core = the core's whole assignment) |
| Grid | `device.compute_with_storage_grid_size()`. Writer cores = the first `num_writer_cores = min(max_writer_cores, tensor_pages)` cores in row-major order: one rectangle (full grid) or two (full rows + one partial row) (BB-17) |
| Per-core work | bank `core_bank(c)`, slots `[core_slot_begin(c), core_slot_begin(c) + core_slots(c))` (balanced: sizes differ by at most 1 within a bank group) |
| Remainder | alignment-aware: `tensor_pages = output.buffer_num_pages()` (= `d_0 * d_1 * ceil(d_2/32) * ceil(d_3/32)`), `bank_pages(b) = ceil((tensor_pages - b) / num_dram_banks)`. The partial last slot row is handled by the per-bank page count, and the kernel clips each run at `tensor_pages` |
| Mesh | the same descriptor on every device (SPMD). Grid and bank count are read from `device` (a MeshDevice reports its first device, and production meshes are four identical Blackhole chips) |
| Kernel groups | one writer kernel (BRISC, NoC1, `WriterConfigDescriptor`) and one CB on the writer-core range set. Idle grid cores (only p4-sized shapes, `tensor_pages < grid_cores`) run nothing |
| Host cache (BB-07) | the Python entry point keeps a dict from `(tuple(shape), dtype, layout, memory_config, device identity)` to the built `ProgramDescriptor` and its derived plan. A hit patches only the buffer-address common RT arg and rebuilds nothing. Everything that shapes the program (shape, dtype, grid, bank count) is in the key or derived from it. The cache holds no tensor |
| Plan report (BB-08 / BB-21) | the op module exposes a host function that returns the derived plan for a call (`num_writer_cores`, per-core `(core, bank, slot_begin, slots)`, `zero_source_bytes`, writer RISC/NoC) without dispatching, and a profiler-build zone per operation: `ZEROS-FILL` (D1 including its barrier), `ZEROS-WRITE` (D2 issue), `ZEROS-DRAIN` (D3) |

`ttnn.split_work_to_cores` is not used: it hands each core a contiguous range of page ids (the
rank-2 split of the Traffic ranking), not one bank's slots. The bank-group split is the integer
arithmetic of the derived-quantities table.

Single regime: no selection predicate, no regime-pinned tests.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_zero_source` | 0 | `zero_source_bytes` | 1 | live set = the whole zero source, read by every in-flight write. Spans no block axis: it streams over `bank` and `bank_slot` (each write re-reads it). Bounded by `ZERO_SOURCE_MAX_BYTES` | output dtype (a dataflow-only byte scratch that never passes DEST; any format works, see Key Risks) | writer (L1 zero fill) | writer (NoC write source) | whole writer kernel |

`cb_zero_source` is a same-kernel scratch: it is never reserved, pushed, waited or popped. A fresh CB
has `read_ptr == write_ptr` (`noc.h:858-866`), and the writer uses its write pointer as both the
fill destination and the write source. It has one owning thread (the writer RISC). CB sync rule:
0 pushes = 0 waits.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `make_zero_source` | `zero_source_bytes` | no (raw API; helpers considered below) | none | `cb_zero_source`, 1 page, at its write pointer | holds `zero_source_bytes` zero bytes after the L1 barrier. Read-only for the rest of the kernel |
| 2 | `write_bank_run_block` | `1 bank x block_slots` pages = `run_bytes` | no (raw API; helpers considered below) | `cb_zero_source`, 1 page, zeroed, read at its write pointer, not popped | none (DRAM `out` pages `first_page + k * num_dram_banks`, `k < run_pages`) | unchanged |
| 3 | `drain_writes` | all writes of the core | no | none | none | unchanged |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `make_zero_source` | raw_api | `Noc::async_write_zeros(dst, size_bytes)` (overload 1, local L1) | `tt_metal/hw/inc/api/dataflow/noc.h:849` (impl `tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl:13-44`) | `dst` = `cb_zero_source` handle, `size_bytes = zero_source_bytes` | none | `cb_zero_source` | `size_bytes` = `zero_source_bytes` |
| `make_zero_source` | raw_api | `Noc::write_zeros_l1_barrier()` | `noc.h:904` (impl `noc_zero_l1.inl:46`) | none | n/a | n/a | none |
| `write_bank_run_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:554` | `page_id = first_page`, `offset = written` (or add `written` to the returned address) | n/a | n/a | none |
| `write_bank_run_block` | raw_api | `TensorAccessor::num_contiguous_pages(page_id, end_page_id)` | `tensor_accessor.h:568` | `first_page`, `tensor_pages` | n/a | n/a | clips `block_slots` to the tensor end |
| `write_bank_run_block` | raw_api | `noc_async_write(src_l1, dst_noc_addr, size)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:876` | `src_l1` = `cb_zero_source` write pointer, `size = min(zero_source_bytes, run_bytes - written)` | `cb_zero_source` | n/a (DRAM) | `size` <= `zero_source_bytes`. Trip count `writes_per_block` |
| `drain_writes` | raw_api | `noc_async_write_barrier()` | `dataflow_api.h:1951` | none | n/a | n/a | none |

There is no compute phase, so no compute helper applies and `compute_kernel_hw_startup` is not
called (there is no compute kernel). Helpers considered and rejected:

| Raw entry | Candidate | File:Line of the mismatch | Concrete reason it cannot express the planned shape |
|-----------|-----------|---------------------------|-----------------------------------------------------|
| `make_zero_source` | `dataflow_kernel_lib::zero_tile(dfb)` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:53-57` | zeroes exactly `dfb.get_tile_size()` bytes and runs a barrier per call. The zero source is `zero_source_bytes` (16384, not a multiple of the 1088 B bfloat8_b tile), so it needs several calls with one barrier each, or cannot cover the size at all |
| `make_zero_source` | `dataflow_kernel_lib::prepare_zero_tile(dfb)` | `l1_helpers.hpp:67-71` | adds `reserve_back`/`push_back` credit for a consumer that does not exist (same-kernel scratch). It is one tile per call, as above |
| `make_zero_source` | `dataflow_kernel_lib::fill_l1_range<4>(addr, n, 0)` | `l1_helpers.hpp:101` (4 B store loop) | one RISC store per 4 B: 4096 stores for 16 KiB, against 32 NoC loopback reads of 512 B for overload 1 |
| `write_bank_run_block` | `Noc::async_write_zeros(accessor, size, {page_id}, scratch)` (overload 2, DRAM pages from a pre-zeroed scratch) | `tt_metal/hw/inc/internal/noc_zero_dram.inl:24` (`ASSERT(offset_bytes + size_bytes <= aligned_page_size)`). The gap is acknowledged at `noc_zero_dram.inl:10-11` | one call stays inside one page, so a bank run of `run_pages` pages becomes `run_pages` calls and `run_pages` writes (regime R2). What would close the gap: a page-range overload that writes a bank run from the scratch in burst-sized chunks. That is exactly `write_bank_run_block` |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Pages left unwritten (R1 of `risks.md`) | a fresh test buffer often holds zeros already, and the model's buffers hold stale data | every page belongs to exactly one block by construction (bank groups partition the banks, slot ranges partition `bank_pages(b)`). The acceptance test and golden loose cases poison the destination range first (dirty destination) |
| Zero source read before its fill lands (R2) | the loopback fill is asynchronous | `write_zeros_l1_barrier` sits between `make_zero_source` and the first `noc_async_write`, once per core |
| Writes in flight at kernel return (BB-03) | up to about 137 non-posted writes per core at p1 | one `noc_async_write_barrier` at kernel end. Posted writes are only a lamp (L5) and need the same guarantee |
| Run spills into another core's pages or past the tensor | a bank run's length depends on the partial last slot row | `run_pages` is clipped through `num_contiguous_pages(first_page, tensor_pages)`. Chunk size is clipped to `run_bytes - written` |
| Host/kernel bank-count mismatch | the host split assumes `num_dram_banks`, while the accessor strides by `NUM_DRAM_BANKS` | host value passed as CT arg, kernel `ASSERT`s equality (fires under `--dev`). A mismatch would also fail the dirty-destination test |
| `page_bytes` not the aligned page size | `num_contiguous_pages` silently returns 1, so the op gets correct but slow per-page writes | host passes `output.buffer_aligned_page_size()` |
| bfloat8_b `-0.0` or a mis-sized page (R5) | a datum byte 0x80 reads as -0.0, and a 1088 B page differs from the others | the op writes raw zero bytes (exponents and data all 0), never through the packer. Sizes come from `buffer_aligned_page_size()` |
| CB format for a non-tile-sized scratch | `cb_zero_source` holds `zero_source_bytes`, which for bfloat8_b is not a multiple of its 1088 B tile | the format is irrelevant to a byte scratch. If CB validation rejects the page size for `Bfp8_b`, give the CB any non-block format (e.g. `float32`) |
| Program-cache key (BB-07, R6) | p2 and p3 differ only in `d_0`. p1 and p4 differ in dtype | the host descriptor cache key holds shape and dtype. The device-op hash covers CT args and core ranges, and per-core RT args are re-applied from the descriptor on a hit. The acceptance test runs p3 -> p2 -> p3 and checks values and cache-entry counts |
| Output reused across calls (spec rule 2) | caching tensors would give right values | the cache holds descriptors only. The output is allocated in every call. The acceptance test checks that two calls give two addresses |
| Mesh: only one device written (R9) | tests run on one device | `ProgramDescriptor` is SPMD over every device of the mesh, and the output is a mesh buffer (same address everywhere). No per-device branch exists |
| Validation order | the spec requires `ValueError`/`RuntimeError` for a non-TILE layout, a non-DRAM-interleaved memory config, a bad shape or a bad device, while SUPPORTED refusals are `NotImplementedError` subclasses (`_op_contract.py:23-31`) | `validate()` first checks arguments (`ValueError`: 4 positive ints with `d_2`, `d_3` multiples of 32; `layout == TILE_LAYOUT`; DRAM + INTERLEAVED; `device` is an open `MeshDevice`/device), then builds `({"shape", "dtype", "layout"},)`, runs `tag_dtype`/`tag_layout`, and checks SUPPORTED, then EXCLUSIONS (empty). No size, shape or rank axis, no extra tagger |
| Delegation or host-made values (R7, BB-01) | an existing fill/creation op or a host upload gives right values | the entry point dispatches only its own `generic_op` with its own writer kernel. Nothing is uploaded |
| Per-call overhead on p4 (R11, BB-17/18) | 128 pages, called 240 times per FLUX pass | writer cores form at most 2 rectangles. The kernel text is a few dozen lines. Lamp L1 measures fewer cores |

### Structural impossibilities

None to add: INVALID is correctly empty (every TARGET dtype is storable in tiles).

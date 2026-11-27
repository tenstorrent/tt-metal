# L1 Ledger: g6_zeros_like

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md`, Blocking Model.
Block axes: `page` (folded from `d_0`, `d_1`, `h_tiles`, `w_tiles`; block = the core's whole range, extent `num_pages_this_core`). Regime: `page_split_local_zero` (the only built regime).

## Inventory

Before any sizing: the op has one phase that needs storage (a zero source for NoC writes). No input is read, so there is no input CB. The output is written straight from the zero source to DRAM, so there is no output CB. No compute kernel, so no intermediate. One buffer remains.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_zero_page` | `zero_scratch_pages = 1` (constant: one zero page is the source of every page write of the core) | 1 page (`page_size` B), resident for the whole kernel | `{page: streams → every one of the core's num_pages_this_core writes reads the same 1 page; d_0, d_1, h_tiles, w_tiles: folded into page → streams}` | output dtype: `Float32` (4096 B page) / `Bfp8_b` (1088 B page). Never passes through DEST (dataflow-only payload: NoC loopback in, NoC write out), so audit 2's fp32-under-16-bit-DEST finding does not apply; there is no DEST in this op | writer (zero fill, `zero_tile`) | writer (NoC write source, overload 2) | whole writer kernel: filled once, read by every page write, never popped | none: it is the only CB of the op. Alternatives considered: (1) pack into the destination — the destination is DRAM pages; the zero source must be L1 (NoC write source); (2) write from the firmware `MEM_ZEROS` region directly, using no CB — rejected in `op_design.md` (`sub_page_writes_from_mem_zeros`: 512 B region, 8× sub-page write commands); (3) alias — nothing to alias with |

## Symbol table

| Symbol | Bound | Established by |
|--------|-------|----------------|
| `page_size` | 4096 B (float32 TILE tile) or 1088 B (bfloat8_b TILE tile: 1024 B mantissas + 64 B exponents); ≤ 4096 B for every TARGET dtype | `validate()`: TILE layout, dtype ∈ TARGET; value read from `output_tensor.buffer_aligned_page_size()` |
| `zero_scratch_pages` | 1 | design constant of `page_split_local_zero` |
| `NOC_MAX_BURST_SIZE` | 16384 B (BH: 256 words · 64 B) | `noc_parameters.h:289-290`; scratch must cover `min(page_size, 16384)` = `page_size` |
| `num_pages_this_core` | does not appear in any capacity expression | the block extent is free of L1: the live set does not span `page` |

## Footprint

Total per core = `zero_scratch_pages · page_size` = `page_size` ≤ 4096 B (float32) / 1088 B (bfloat8_b), against the model's 1347776 B per core (BB-05): 0.3 %. No term scales with `num_pages_this_core`, `NUM_CORES_CAP` or `P`; the only variable is the dtype. No L1 budget predicate, safety fraction or blocking search exists, and none is needed: no production shape (and no TARGET shape) can exceed the bound, so no host refusal is required.

## Data-movement budget

Split: `page_split_local_zero` (folded page index over `min(P, grid_cores)` cores, one local zero page per core). Production: float32 `(1,12,128,128)` → 192 pages of 4096 B on 110 cores; bfloat8_b `(5,1,32,256)` → 40 pages of 1088 B on 40 cores.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 0 | not an operand: only its metadata (shape, dtype, layout, placement) is used; its buffer is not bound to the program | 0 |
| output (new tensor or `optional_tensor`) | 1 (write) | every page written exactly once from the core-resident zero page; nothing is read back | 0 |
| zero page (core-local, not a tensor) | 0 | made in each core's L1 by NoC loopback from the firmware zero region | 0 (core-local: `page_size` per participating core) |

Total per tier:

| Tier | float32 `(1,12,128,128)` | bfloat8_b `(5,1,32,256)` |
|------|--------------------------|--------------------------|
| DRAM read | 0 B | 0 B |
| DRAM write | 786432 B (192 × 4096) | 43520 B (40 × 1088) |
| Cross-core NoC | 0 B | 0 B |
| Core-local (loopback fill) | 450560 B (110 × 4096) | 43520 B (40 × 1088) |

> Cheapest-traffic split considered: folded page split with a core-local zero page — 0 B read, one write of the output, 0 B cross-core; it ties at DRAM and cross-core with `bank_run_coalesced` (same bytes, fewer write commands; core-local fill `min(r·page, 16 KiB)` per core). Implemented: the folded page split (`page_split_local_zero`), which **is** the cheapest at the DRAM and cross-core tiers. Its core-local term (`C · page_size`) is minimized by fewer cores, which is the grid-synchronization lamp, measured separately. `bank_run_coalesced` is deferred because it moves no fewer bytes on any tier and at the built split removes at most one write command per core (float32) and none (bfloat8_b); the kernel structure (constant scratch, write loop, one barrier) keeps it reachable.

# L1 Ledger: plus_one

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Design: `op_design.md` (same directory). Target: `quasar` (4 MiB L1 per cluster), ttsim.

The program has **one** L1 buffer per core: a scratchpad, no DataflowBuffer. Block axes of the Blocking Model: `row` (extent `block_units` per thread) and `col` (extent `seg_elems` per unit).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `scratch_units` (ScratchpadSpec; a "page" = one unit slot of `slot_stride_bytes`) | `WORKER_THREADS × SCRATCH_DEPTH × block_units` slots in `WORKER_THREADS` regions of `region_stride_bytes = align_up(SCRATCH_DEPTH·block_units·slot_stride_bytes, CACHE_LINE_BYTES)`, plus `CACHE_LINE_BYTES` slack: `size_per_node = WORKER_THREADS·region_stride_bytes + CACHE_LINE_BYTES` bytes | equal to capacity minus alignment: during `increment_block` every thread can hold its whole block at once (all `WORKER_THREADS` regions live simultaneously, `block_units` slots each). Gap capacity − live set = the per-region round-up to a 64-B line + 64 B base slack (mechanism: L2 line exclusivity, op_design Mechanism caps) + each slot's round-up from `n·4` to `slot_stride_bytes` (L1 alignment) | `{row: spans → block_units per thread × WORKER_THREADS threads, col: spans → seg_elems per slot}` | raw 32-bit words, no `data_format_metadata` (uint32 is not a buffer format on Quasar; DM-only use needs none) | `worker` (NoC reads; thread `t` writes region `t` only) | `worker` (in-place increment, then NoC writes; same thread, same region) | the whole program; each region reused block after block (Phase 0: one block per thread) | **No other buffer exists to share with.** It cannot be removed: the data lives in DRAM (or, later, in another core's L1 bank), which a DM hart cannot address with loads/stores, so the words must land in this core's L1 to be incremented; the read target and the write source are the **same** slot (the increment is in place — no separate output buffer, Rule 3 "transform in place"). Regions are not shared between threads because their lifetimes overlap (threads run concurrently) and a shared 64-B line would let one thread's invalidate discard another's dirty data. |

**Verifier note (depth knob, Phase 0 verification).** The live set above holds at `SCRATCH_DEPTH = 1`, the only value the op runs. At `SCRATCH_DEPTH > 1` the kernel alternates a thread's blocks between `SCRATCH_DEPTH` sub-regions, but its schedule stays serial (`load_block` → barrier → `increment_block` → `store_block` → barrier before the next block's first read), so only one block per thread is ever live: capacity would exceed the live set by `(SCRATCH_DEPTH − 1)·block_units` slots per thread. The knob is kept (it is the design's overlap lamp, and `test_block_path` exercises depth 2), but turning it buys L1, not overlap, until the kernel issues block k+1's reads before retiring block k. No INPUTS shape gives a thread more than one block (largest share 64 KiB per call ≪ `max_block_bytes` ≈ 1 MiB), so this is recorded, not queued (verification_report.md).

## Symbol table

| Symbol | Bound | Predicate / source that establishes it |
|--------|-------|----------------------------------------|
| `WORKER_THREADS` | 4 (≤ 6 user DM harts) | host constant; thread plan |
| `SCRATCH_DEPTH` | 1 | host constant |
| `CACHE_LINE_BYTES` | 64 | host constant = `L2_CACHE_LINE_SIZE` (`tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/overlay_addresses.h:44`) |
| `seg_elems` | `1 ≤ seg_elems ≤ min(W, floor_to(max_block_bytes/4, seg_granule_elems))` | clamp in the host split (op_design Mechanism caps) |
| `slot_stride_bytes` | `align_up(seg_elems·4, ttnn.get_l1_alignment()) ≤ max_block_bytes` | derived from `seg_elems` |
| `block_units` | `1 ≤ block_units ≤ floor(max_block_bytes / slot_stride_bytes)` and `≤ ceil(max_core_units / WORKER_THREADS)` | clamp in the host split |
| `max_block_bytes` | `floor_to((budget − CACHE_LINE_BYTES) / (WORKER_THREADS·SCRATCH_DEPTH), CACHE_LINE_BYTES)` | derived from `budget` |
| `budget` | `ttnn.get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES`, `L1_RESERVE_BYTES = 0`; slightly under 4 MiB on this target (the query's value, never a literal) | device query |
| `W`, `rows` | unbounded op dimensions — **do not appear in any capacity expression**; they enter only through `seg_elems ≤ W` and the unit count, which sets the number of blocks, not their size | — |

## Total per-core footprint

`L1 = WORKER_THREADS · align_up(SCRATCH_DEPTH · block_units · align_up(seg_elems·4, l1_align), 64) + 64`

- scales linearly with `WORKER_THREADS`, `SCRATCH_DEPTH`, `block_units` (row extent) and `seg_elems` (col extent);
- independent of the tensor size by construction: `block_units·slot_stride_bytes ≤ max_block_bytes` makes `L1 ≤ budget` for every shape;
- Phase 0 worked values (this target): `(1,)`–`(100,)`: 4 × align_up(≤400, 64) + 64 ≤ 1.7 KiB; `(4096,)`: 4 × 1024 + 64 = 4,160 B on each of 4 cores. Later `(512, 8)`: 4 × align_up(32·32, 64) + 64 = 4,160 B; `(1,1,2,8192)`: 4,160 B.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (also the result) | 2 (read once, written once) | in-place read–modify–write: each element is resident in exactly one thread's scratch block from its read to its write, so it is never re-read; padding bytes are never moved | none (no combine, no multicast, no shared operand) |

Totals per tier, for a tensor of `rows·W` elements:

| Tier | Bytes |
|------|-------|
| DRAM | `2 · rows·W·4` (the minimum for an in-place op) |
| NoC (cross-core) | `2 · rows·W·4` DRAM↔core transfers only; 0 core↔core |
| Core-local L1 | `rows·W·4` landed by the NoC, read and written once by the hart (through the L2 cache), read once by the NoC |

Transfers: `2 · rows · segs_per_row` (one read and one write per unit); `segs_per_row = 1` whenever `rows ≥ workers_wanted`.

> Cheapest-traffic split considered: `row` (whole-row units) — 0 extra bytes on any tier versus every alternative; it moves the fewest transfers. Implemented: `row` first, adding `col` segments only when `rows < workers_wanted` (same bytes, `2·rows·(segs_per_row − 1)` extra transfers, the only parallelism a single-row tensor has). The implemented split is also the cheapest-traffic split: every candidate moves the same DRAM bytes, and segments are introduced only where the row split cannot occupy the wanted workers.

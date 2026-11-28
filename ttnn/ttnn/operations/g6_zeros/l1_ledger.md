# L1 Ledger: g6_zeros

Blocking axes (from `op_design.md` → Blocking Model): `bank` (extent `B`), `bank_row` (per-core
block = `run_pages`), `burst` (`burst_pages` pages per NoC write), `mesh` (devices). The page-index
axes `d_0, d_1, tile_rows, tile_cols` are folded into `bank × bank_row` and carry no separate
buffer extent.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_zero_source` | 1 page of `zero_source_bytes = burst_pages · aps` bytes | 1 page (the whole source of one burst; every burst reads the same bytes) | `{bank: streams → 1 per core, bank_row: streams → re-read per burst, burst: spans → burst_pages·aps bytes, mesh: streams → per device}` | output `dtype` label only; dataflow-only payload, never unpacked, never through DEST (audit 2 n/a: no compute, `fp32_dest_acc_en` irrelevant) | writer (local zero-fill) | writer (NoC write source) | whole writer kernel | Only buffer of the op — nothing to share with. Not removable: the alternative source, the 512 B hardware zero region `MEM_ZEROS_BASE` (`dev_mem_map.h:88`), would cap each write at 512 B (32× the transaction count of a 16 KiB burst); 16 KiB of L1 buys that factor. No double buffer: the content never changes, so no producer/consumer overlap exists to buy. |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `aps` | `output.buffer_aligned_page_size()` | 1088 ≤ aps ≤ 4096 (bfp8 / bf16 / fp32 tiles) | TARGET dtypes, TILE layout |
| `MAX_WRITE_BURST_BYTES` | host constant, 16384 (Blackhole `NOC_MAX_BURST_SIZE`) | constant | — |
| `burst_pages` | `max(1, floor(MAX_WRITE_BURST_BYTES / aps))` | 4 ≤ burst_pages ≤ 15 | `aps` bound above |
| `zero_source_bytes` | `burst_pages · aps` | ≤ 16384 | definition of `burst_pages` (and `aps ≤ 16384`) |
| `run_pages`, `B`, `cores_per_bank`, `N` | per-core run / bank count / cores per bank / output pages | unbounded `N` (up to 262 648 in INPUTS) — **appears in no capacity expression** | — |

## Total per-core footprint

`L1_per_core = zero_source_bytes = burst_pages · aps ≤ 16 384 B` — scales only with the
`burst_pages` knob (and `dtype` through `aps`); independent of `N`, `run_pages`, `cores_per_bank`,
`B` and the mesh size. Bound against the model budget (BB-05): `16 384 ≤ 1 347 776`, for every
shape — no shape can OOM on L1 and none is refused.

## Data-movement budget (chosen split R1, per device)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| output | 1 (write of every page, `N · aps` bytes) | zeros are made in each core's own L1 and never staged in DRAM; each page belongs to exactly one core's run and is written once | 0 |
| (inputs) | 0 | the op has no input and reads no DRAM | 0 |

Per tier, per device: DRAM writes `N · aps` (p2 12 582 912 B, p3 6 291 456 B, p1 285 761 024 B,
p4 262 144 B); DRAM reads 0; cross-core NoC 0; core-local L1 `zero_source_bytes` loopback zero-fill
per participating core (≤ 16 KiB · cores used). NoC write transactions `Σ_cores ceil(run_pages / burst_pages)`
(p2 ≈ 768, p1 ≈ 17 600 at 128 cores) instead of `N` for a per-page split.

> Cheapest-traffic split considered: R1 bank × bank-row with coalesced runs — 0 delta in bytes per
> tier against every other legal split (all move exactly the minimum: one output write, no reads, no
> cross-core traffic); it is cheapest on transaction count (`N / burst_pages` vs `N`). Implemented:
> R1. It is the cheapest split, so nothing is deferred on traffic grounds.

## Block-size default: departure from "full grid" (Refinement 2, measured)

The Blocking Model's default `cores_per_bank = floor(G/B)` (104 cores, row-major placement) is
**replaced** by `cores_per_bank = 1` (8 cores) at `placement = "bank_near"` (each bank's writer is
the NoC1-optimal worker for that bank, `get_optimal_dram_bank_to_logical_worker_assignment`).
Concurrent writers into one bank contend, and row-major cores send their NoC1 write streams over
shared links; one bank-adjacent writer per bank runs every case at its fastest
(Blackhole p150, device kernel ns, Phase-0 default → new default):
p1 1 092 860 → 559 495, p2 47 114 → 25 492, p3 24 271 → 13 139, p4 1 807 → 1 439.
The per-core footprint is unchanged (`zero_source_bytes ≤ 16 KiB`); per-core run length grows to
`ceil(N/B)` pages, which appears in no capacity expression. Both knobs stay live
(`make_plan(..., cores_per_bank=, placement=)`), and `test_g6_zeros_perf.py` sweeps them.

# L1 Ledger: g6_zeros

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes, from `op_design.md`
(Blocking Model): `bank` (`block_extent[bank]` = 1) and `bank_row` (`block_extent[bank_row]` =
`core_bank_rows`). The logical axes `d_0`, `d_1`, `tensor_ht_tiles`, `tensor_wt_tiles` are folded
into `page`, which is factored into those two. No L1 buffer is indexed by a logical axis.

## Inventory

The op has one L1 buffer. Nothing is read from DRAM, nothing is computed, and nothing passes
between kernels: the only bytes a core needs are the constant zero source of its own writes.

Considered and not created:

- No input CB: the op has no input.
- No output staging CB: the writes are sourced straight from the zero source. A per-page staging
  pipeline would be a phase boundary expressed as storage for bytes that never change.
- No compute intermediates: there is no compute kernel.
- No second zero source per RISC: there is one writer kernel.

## Table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_zero_source` | 1 page of `zero_chunk_bytes` = `min(NOC_MAX_BURST_SIZE, max_c(core_bank_rows_c) * page_bytes)` bytes. Constant in the block extents: a mechanism-capped transfer quantum, not a block extent. | 1 page (all `Z` bytes are the source of every full-size transfer) | `{bank: streams -> one run at a time, re-read for every transfer; bank_row: streams -> re-read once per Z bytes of the run, never spans core_bank_rows}` | raw bytes, declared `bfloat16`: dataflow-only payload, never unpacked or packed, so the DEST-width audit does not apply | writer (NoC loopback zeroing) | writer (NoC write source) | whole kernel: zeroed once, then read-only | Shares with nothing: it is the only buffer. No disjoint-lifetime partner exists to alias with. |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `NOC_MAX_BURST_SIZE` | 16384 B on Blackhole (`noc_parameters.h:287-290`) | architecture constant; kernel `static_assert(zero_chunk_bytes <= NOC_MAX_BURST_SIZE)` |
| `page_bytes` | in {4096, 2048, 1088} (float32, bfloat16, bfloat8_b TILE) | `dtype` in TARGET; host asserts `buffer_page_size() == buffer_aligned_page_size()` |
| `core_bank_rows` | `<= ceil(rows_in_bank(b) / W_b)`; unbounded in L1 terms, and needs no bound because no buffer spans it | Work Distribution formula |
| `zero_chunk_bytes` (`Z`) | `64 <= Z <= 16384`, `Z % 64 == 0` | `min(...)` on the host plus the `static_assert` |
| `W`, `B` | `8 <= W <= 110`, `B = 8` here | host clamp; `static_assert(B == NUM_DRAM_BANKS)` |

## Total per-core footprint

`L1_op = Z <= NOC_MAX_BURST_SIZE = 16384 B` on every core. It does not scale with any block extent,
`P`, the core count or the shape. Only the transfer-quantum knob moves it, and that knob is capped.
Against the model's budget of 1347776 B per core (BB-05), it uses 1.2% at the largest `Z`. No shape
in the golden INPUTS (or any shape) can exceed it, so no host-side size refusal exists.

Per shape: p1, p2, p3: 16384 B; p4: 4096 B.

## Audits

| Audit | Result |
|---|---|
| 1. Capacity vs live set | capacity = live set = 1 page; over: none; under: no buffer spans a block axis |
| 2. Page format vs DEST width | not applicable: no compute kernel, no DEST; the payload is dataflow-only |
| 3. Disjoint lifetimes without justification | none: one buffer |
| 4. Bounds and closed form | every symbol above is bounded; total is closed-form (`Z`) |

## Data-movement budget

Chosen split: bank-affine (core `c` writes one contiguous run of bank `c mod B`), full worker grid.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `out` | 1 (write of every page, once) | each page belongs to exactly one core's bank run, and each run is written exactly once from the core's own L1 zero source; no clearing pass, no second write | none |
| zero source (`cb_zero_source`) | 0 | made in each core's L1 from the hardware zero region by NoC loopback (`Z` bytes once per core); never read from or written to DRAM | none: each core makes its own; broadcasting it would add `(W - 1) * Z` bytes to replace a core-local operation |

Totals per tier, per call per device:

| Case | DRAM written | DRAM read | Cross-core | Core-local L1 (zeroing + source reads by the NoC engine) |
|---|---|---|---|---|
| p1 | 285761024 B | 0 | 0 | `110 * 16384` B zeroed + 285761024 B sourced |
| p2 | 12582912 B | 0 | 0 | `110 * 16384` B zeroed + 12582912 B sourced |
| p3 | 6291456 B | 0 | 0 | `110 * 16384` B zeroed + 6291456 B sourced |
| p4 | 262144 B | 0 | 0 | `110 * 4096` B zeroed + 262144 B sourced |

> Cheapest-traffic split considered: every candidate split (bank-affine, bank-major contiguous,
> page-order contiguous) moves exactly the minimum: `out` written once, 0 B read, 0 B
> cross-core, delta 0 B on every tier. Implemented: bank-affine, which is the cheapest.
> It wins the tie on fixed cost (1 run per core vs up to `B`) and on bank concurrency (a fixed set
> of writers per bank). The bank-major alternative is a perf lamp in `op_design.md` (equal bytes per
> core); the kernel's bank-major range interface keeps it a host-only change.

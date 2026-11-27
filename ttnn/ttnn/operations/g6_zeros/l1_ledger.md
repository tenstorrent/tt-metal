# L1 Ledger: g6_zeros

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes, from the Blocking Model in
`op_design.md`: `bank` (extent `block_banks = 1`), `bank_slot` (extent `block_slots`), `device`. The
logical axes `d_0`, `d_1`, `d_2/32`, `d_3/32` are folded into the page index, which `bank` and
`bank_slot` decompose.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_zero_source` | 1 page of `zero_source_bytes` B, where `zero_source_bytes = min(ZERO_SOURCE_MAX_BYTES, max_c run_bytes(c))` | 1 page (`zero_source_bytes` B). Every write issued so far reads it until the final barrier, so all of it is live for the whole kernel. Capacity = live set | `{bank: streams → 0 (one bank's run at a time; the source does not grow with banks), bank_slot: streams → window zero_source_bytes (each chunk of a run re-reads the same bytes), device: streams → per-device copy (SPMD)}` | `float32` for every output dtype (`ZERO_SOURCE_CB_FORMAT`): a dataflow-only byte scratch that never passes DEST, so a fixed non-block format avoids a `Bfp8_b` CB whose 16384 B page is not a multiple of its 1088 B tile. There is no compute kernel and `fp32_dest_acc_en` does not apply (op_design Key Risks) | writer (L1 loopback zero fill) | writer (NoC write source) | whole writer kernel: filled once, read by every write, never re-zeroed | none: it is the op's only CB. The pack-into-destination pattern does not apply (the destination is DRAM, and the source is what makes the zeros). No other buffer exists to alias. The hardware zero region `MEM_ZEROS_BASE` (512 B) could replace it as the direct source but caps writes at 512 B (regime R4, rejected) |

Over/under audit: capacity equals the live set. The buffer spans no block axis, so its capacity
correctly does not scale with `block_slots`, which can grow to the whole per-core assignment at no
L1 cost.

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `ZERO_SOURCE_MAX_BYTES` | 16384 | one host constant (= `NOC_MAX_BURST_SIZE` on Blackhole, `noc_parameters.h:287-290`) |
| `zero_source_bytes` | `<= 16384`, multiple of 64 | `min(ZERO_SOURCE_MAX_BYTES, max run_bytes)`. `run_bytes` is a multiple of `page_bytes`, and every `page_bytes` is a multiple of 64 |
| `page_bytes` | in {1088, 2048, 4096} | `output.buffer_aligned_page_size()` for the TARGET dtypes (bfloat8_b, bfloat16, float32) |
| `run_bytes(c)` | `<= bank_pages(b) * page_bytes`, unbounded above (p1: 2232576 B per core) | does not enter any capacity expression. Only the trip count `ceil(run_bytes / zero_source_bytes)` depends on it |
| `num_dram_banks` | device constant (8 on this Blackhole) | `device.dram_grid_size()`, checked against the kernel's `NUM_DRAM_BANKS` |

## Total per-core footprint

`L1(core) = zero_source_bytes` for a writer core (0 for an idle core), at most 16384 B. Nothing
scales with `block_slots`, `banks_per_core`, `max_writer_cores` or the shape. Lamp L3 (a second
writer lane) doubles it to `2 * zero_source_bytes`, and lamp L6 raises it to at most 65536 B. Both
stay far below the 1347776 B the model leaves to the op (BB-05).

## Data-movement budget

Chosen split: bank-affine bank runs (regime R1), per device.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `out` | 1 (write only) | every page belongs to exactly one block and is written once from the core-local zero source. Nothing is read: the zeros are made in L1 by a loopback read of the core's own `MEM_ZEROS` region | 0 |
| zero source (L1, not a tensor) | 0 | core-local: one loopback fill of `zero_source_bytes` per writer core per call | 0 |

Bytes per tier, per device and per call:

| Case | DRAM write | DRAM read | Cross-core NoC | Core-local fill (all writer cores, 130-core illustration) |
|------|-----------:|----------:|---------------:|-----------------------------------------------------------|
| p1 (32831,1,32,256) bfloat8_b | 285761024 | 0 | 0 | 130 x 16384 = 2129920 |
| p2 (16,12,128,128) float32 | 12582912 | 0 | 0 | 130 x 16384 = 2129920 |
| p3 (8,12,128,128) float32 | 6291456 | 0 | 0 | 130 x 16384 = 2129920 |
| p4 (1,4,256,128) bfloat16 | 262144 | 0 | 0 | 128 x 2048 = 262144 |

> Cheapest-traffic split considered: every candidate split (bank-affine, slot-range across all
> banks, page-range) ties at the minimum: `out` written once, 0 B read, 0 B cross-core. Delta: 0 B
> on every tier. Implemented: bank-affine bank runs, which is the cheapest. The tie is broken by
> transfer shape: one contiguous run per core, issued in 16 KiB writes. The slot-range split stays
> reachable as an RT-arg value of the same kernel (`banks_per_core`, lamp L4).

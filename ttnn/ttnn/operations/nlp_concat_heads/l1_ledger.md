# L1 ledger: nlp_concat_heads

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Target: quasar (4 MiB L1 per cluster, 32 clusters), regime `tile_relay` (Phase 0). Thread plan (R, C, W) = (1, 0, 1).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_relay_tiles` | `RING_DEPTH_TILES × max(R, W)` = 8 | `RING_DEPTH_TILES` pages per DM thread endpoint (the STRIDED ring's sub-ring of each of the `max(R, W)` harts on the wider side, `DataflowBuffer.md:255`), so `RING_DEPTH_TILES × max(R, W)` in all: the reads in flight plus the writes not yet sent; capacity equals live set, because depth *is* the in-flight count | `{out_tile: streams → window RING_DEPTH_TILES; batch: streams (folded into out_tile); token_row: streams (folded); head: streams (folded); head_dim_tile: streams (folded)}` | the input dtype (bfloat16 or float32), page = one tile. No DEST is involved: a dataflow-only payload, and float32 pages are the output ABI (R1), so the page-format-vs-DEST audit does not apply | reader | writer | whole program | Nothing to share with: it is the only buffer. It cannot be removed: a NoC read lands in L1 and a NoC write leaves from L1, so a DRAM → DRAM move needs one staging ring. |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `page_bytes` | 2048 (bfloat16) or 4096 (float32) | `input.buffer_page_size()` for a 32×32 tile of a TARGET dtype; host asserts it equals the output's page size |
| `RING_DEPTH_TILES` | 8 in Phase 0; a multiple of 4; `RING_DEPTH_TILES × page_bytes ≤ 1,048,560` (DFB ring cap) | host constant; host asserts the multiple-of-4 rule (implicit-sync batch divisibility) and the cap |
| `R`, `W` | 1 each in Phase 0; `R + W ≤ 6` | `THREAD_PLAN` host constant, target page *Thread plan* |
| L1 budget | `ttnn.get_max_worker_l1_unreserved_size()` (value on quasar not established here; below 4,194,304 B) | run-time query; the host asserts the footprint below fits it. No reserve knob: the footprint is shape-independent |

## Total per-core footprint

`RING_DEPTH_TILES × max(R, W) × page_bytes` = 16 KiB (bfloat16) / 32 KiB (float32) in Phase 0.

Scales with `RING_DEPTH_TILES` and the thread plan only; constant in every tensor dimension (B, H, S, D) and in `block_tiles`. No budget predicate, safety fraction or blocking search exists, and none is needed: under 1% of a cluster's L1 at float32.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` `[B, H, S, D]` | 1 (read) | each input tile is the source of exactly one output tile and is read by exactly one core; nothing is re-read | none |
| output `[B, 1, S, H·D]` | 1 (write) | each output page is written once, by the core that owns it | none |

Per tier, for `T = B · ceil(S/32) · H · (D/32)` tiles of `page_bytes`:

| Tier | Bytes |
|------|-------|
| DRAM read | `T × page_bytes` |
| DRAM write | `T × page_bytes` |
| Cross-core NoC | 0 |
| Core-local L1 | `2 × T × page_bytes` (each tile written into the ring once by its read and read once by its write) |

(For an L1 input or output the same counts apply to the L1 banks in place of DRAM.)

> Cheapest-traffic split considered: flat output pages over all clusters (`tile_relay`) — no delta: every candidate split (input images, token rows) moves the same `2 × T × page_bytes` with no cross-core traffic, and the op has no dependent axis to combine. Implemented: flat output pages, `split_work_to_cores(..., row_wise=True)`. The implemented split is the cheapest; the alternatives were rejected on occupancy (`B·H` and `B·St` units), not on bytes.

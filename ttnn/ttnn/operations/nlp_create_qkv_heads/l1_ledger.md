# L1 Ledger: nlp_create_qkv_heads

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Regime: `tile_relocate` (built).
Target for the worked numbers: `quasar` (ttsim), budget `ttnn.get_max_worker_l1_unreserved_size()` = 3,963,328 B.

Block axes (from `op_design.md` Blocking Model): `batch`, `token_row`, `head`, `head_dim_tile`, folded into `page`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `ring_entries = align_up(RING_DEPTH_TILES, 4·max(R, W))` = 8 (Phase 0: `RING_DEPTH_TILES = 8`, R = W = 1) | `ring_entries` pages: the reads in flight ahead of the writer (one txn-id slice landing while another drains) — capacity equals live set by design; the excess over the 1 page a single transfer needs is the overlap mechanism (double buffering over txn-id slices) | `{page: streams → ring_entries, batch: streams, token_row: streams, head: streams, head_dim_tile: streams}` — spans no axis: every page passes through once and is never revisited | input dtype (bfloat16 → 2048 B pages; float32 → 4096 B). A dataflow-only payload that never passes through DEST, so a Float32 page is the tensor's own bytes (audit 2's mechanism reason), required by R1 | reader | writer | whole program | Only buffer in the program — nothing to share with. Inventory check: no phase boundary exists (the reader's landing slot is the writer's source; pack-into-destination and in-place are already the case: the page is written to DRAM straight from the slot it landed in). A writer-side staging buffer, a compute copy, or per-output rings were considered and are not needed. |

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `RING_DEPTH_TILES` | 8 (host constant; lamp range 4–32) | constant |
| `R`, `W` (reader / writer harts) | `1 ≤ R, W`; `R + W ≤ 6`; `R | W or W | R` | host `validate_thread_plan` (target page *Thread plan*; `DataflowBuffer.md:253-254`) |
| `page_bytes` | 2048 (bf16) or 4096 (fp32) | the SUPPORTED dtype gate (TARGET dtypes: bfloat16, float32) |
| `ring_entries` | ≤ `align_up(32, 4·6)` = 48 at the lamp range's top | from the two rows above |
| budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` (= 3,963,328 B on quasar, `l1_reserve_bytes = 0`) | run-time query — never evaluated as a predicate: the footprint is bounded by constants far below it |

No op dimension (`S`, `W`, `B`, `num_pages`, heads) appears in any capacity.

## Total per-core footprint

`ring_entries · page_bytes` — scales with the buffer-depth knob `RING_DEPTH_TILES` (and the thread plan through the alignment), and with dtype; independent of every block extent and every op dimension.

Phase 0 (bf16): 8 · 2048 = 16,384 B (0.41 % of the budget). float32: 32,768 B. Lamp maximum (48 entries, fp32): 196,608 B.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input` | 1 (read) | each page belongs to exactly one cluster's range and is read once into the ring; nothing is re-read because nothing is used twice | 0 |
| `q` | 1 (write) | each output page is written once, straight from the ring slot its source page landed in | 0 |
| `k` | 1 (write) | same | 0 |
| `v` | 1 (write) | same | 0 |

Per tier, per call: DRAM read = `num_pages · page_bytes`, DRAM write = `num_pages · page_bytes` (q + k + v together hold exactly the input's pages); cross-core = 0; core-local = each page lands in L1 once and leaves once. Traced vision call `[1,1,128,3072]` bf16: 786,432 B read + 786,432 B written; traced text call `[1,1,128,6144]`: 1,572,864 B + 1,572,864 B.

> Cheapest-traffic split considered: the folded input-page range — the minimum (each tensor crosses DRAM once, 0 cross-core bytes). Implemented: the folded input-page range. Every other candidate split (output-order range, (row, head) units, tile rows) moves the identical bytes; none is cheaper. The implemented split is the cheapest.

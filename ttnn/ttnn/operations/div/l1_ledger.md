# L1 Ledger: div

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Worked numbers are for **Quasar** (target page `.claude/references/targets/quasar.md`), thread plan `THREAD_PLAN = (4, 4, 2)` → C = 4, `BLOCK_TILES = 4`, `BUFFER_DEPTH = 2`. Block axes (from the Blocking Model): `flat_tile` (extent `block_tiles`); `leading`, `tile_row`, `tile_col` are folded into `flat_tile`; `intra_tile` is the fixed 32×32 unit.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_dividend_tiles` (`in_a`) | `BUFFER_DEPTH * block_tiles * C` = 32 | `block_tiles * C` (each NEO holds one block it computes on) + up to `block_tiles * C` landing = `BUFFER_DEPTH * block_tiles * C` | {flat_tile: spans → block_tiles (× C NEO slices × BUFFER_DEPTH); leading: streams (folded); tile_row: streams (folded); tile_col: streams (folded); intra_tile: spans → 1 tile per page} | a.dtype = Float16_b (input ABI: a dataflow payload as stored in the bf16 tensor; widened only in DEST) | reader | compute | whole program, every block | none: live concurrently with `cb_divisor_tiles` and `cb_quotient_tiles` (pipelined across blocks: reader fills block k+1 while compute packs block k) — aliasing forbidden by concurrent lifetimes; capacity > one block is the double-buffer mechanism |
| `cb_divisor_tiles` (`in_b`, tensor regime only) | `BUFFER_DEPTH * block_tiles * C` = 32 | as above | {flat_tile: spans → block_tiles (× C × BUFFER_DEPTH); leading: streams (folded); tile_row: streams (folded); tile_col: streams (folded); intra_tile: spans → 1 tile per page} | b.dtype = Float16_b (input ABI) | reader | compute | whole program; absent in `eltwise_number` (the number is an RT arg, 0 L1 bytes) | none: concurrent with the other two rings; it cannot fold into `cb_dividend_tiles` because both operands of one tile must be resident at once (two DEST loads per tile) |
| `cb_quotient_tiles` (`out`) | `BUFFER_DEPTH * block_tiles * C` = 32 | as above | {flat_tile: spans → block_tiles (× C × BUFFER_DEPTH); leading: streams (folded); tile_row: streams (folded); tile_col: streams (folded); intra_tile: spans → 1 tile per page} | output dtype: Float16_b or Float32 (output ABI, R10: the result rounded once into the requested dtype). Float32 under fp32 DEST is width-matched; Float16_b under fp32 DEST is the requested output format, not a truncated intermediate (nothing reads it back) | compute | writer | whole program | none: concurrent with the input rings (pipelined); packing in place into `cb_dividend_tiles` is rejected — the reader fills that ring concurrently (a second producer is forbidden), the writer would become a second consumer, and the output page may be twice the input page (fp32) |

No other buffers: no scaler (the number is the RT arg `divisor_bits`), no intermediate (divide and floor/trunc are fused in DEST and packed straight into the destination — reuse pattern 1), no constant ring.

## Symbol table

| Symbol | Meaning | Bound | Predicate / source |
|--------|---------|-------|--------------------|
| `C` | compute threads (NEOs) per cluster | {1, 2, 4}; Phase 0 = 4 | `THREAD_PLAN`, host check `C ∈ {1,2,4}`, `C % R == 0`, `C % W == 0`, `R + W ≤ 6` |
| `block_tiles` | tiles one NEO takes per chain call | 1 ≤ `block_tiles` ≤ 32 (ring cap at depth 2, C 4, 4096 B page); Phase 0 = 4 | host constant `BLOCK_TILES`; host assert `BUFFER_DEPTH*BLOCK_TILES*C*page ≤ 1,048,560` |
| `BUFFER_DEPTH` | blocks per NEO in flight per ring | ≥ 1; Phase 0 = 2 | host constant |
| `page_a`, `page_b` | input page bytes | 2048 (bf16 tile) in Phase 0; 4096 for float32 operands later | `tensor.buffer_page_size()` |
| `page_out` | output page bytes | 2048 (bf16) or 4096 (fp32) | `out.buffer_page_size()` |
| `L1_budget` | per-cluster bytes the op may spend | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` (`L1_RESERVE_BYTES` = 0); value on Quasar not recorded in the profile, < 4,194,304 | run-time query; no predicate needed in Phase 0 (footprint ≤ 256 KiB, see below) |

No whole-op dimension appears in any capacity: tensor size enters only the RT args (`num_tiles`, `start_id`), never a ring size.

## Footprint

`L1_div = BUFFER_DEPTH · block_tiles · C · (page_a + [tensor regime] page_b + page_out)`

- Every term scales linearly with `block_tiles`, `BUFFER_DEPTH` and `C`; none with the tensor.
- Quasar, Phase 0: tensor regime, bf16 out: 2·4·4·(2048+2048+2048) = 196,608 B (192 KiB); fp32 out: 2·4·4·(2048+2048+4096) = 262,144 B (256 KiB); number regime: 128 KiB / 192 KiB.
- Headroom: ≤ 6.25 % of 4 MiB; the ring cap (1 MiB per ring), not L1, bounds `block_tiles` (≤ 32).

## Data-movement budget

T = `tensor_tiles` = ∏(padded leading dims)·⌈H/32⌉·⌈W/32⌉ of the output (= of a in Phase 0).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| a | 1 (T · page_a bytes) | each tile read once by the reader hart that serves its NEO; nothing is re-read (no second pass, no overlap between blocks) | none |
| b (tensor regime) | 1 (T · page_b) | same-shape: page i of b is used by output page i only | none |
| b (number regime) | 0 | the number travels as a 4-byte RT arg to each engaged cluster | none |
| out | 1 (T · page_out) | written once from the out ring | none |

Totals per call: DRAM/L1-bank tier T·(page_a + page_b + page_out) (tensor) or T·(page_a + page_out) (number); cross-core tier 0; core-local: each page passes L1 once in and once out.
(L1-resident operands, `memory_config = L1`, cross the same NoC path from their L1 banks; the count is identical.)

> Cheapest-traffic split considered: `flat_tile` split in whole rounds of C across the grid — the minimum (each input once, output once, no cross-core traffic). Implemented: `flat_tile` split in whole rounds of C across the grid. The implemented split **is** the cheapest; every alternative split (by `tile_row`, `leading`, `tile_col`) moves the same bytes with worse balance, and no dependent-axis split exists.

## Implementation status (Phase 0)

Implemented as inventoried: three rings `in_a`, `in_b` (tensor regime only), `out`, each
`BUFFER_DEPTH * BLOCK_TILES * C` entries (host constants in `div_program_artifacts.py`), host-asserted
against the 1,048,560 B ring cap. No buffer added, merged or deleted; the data-movement budget above is
the implemented one (a, b, out each cross once; the number is the RT arg `divisor_bits`).

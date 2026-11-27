# L1 Ledger: g6_concat

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from
`op_design.md` → Blocking Model. The one blocking axis is `out_page`, with extent `block_pages`.
The tensor axes `lead`, `cat`, `mid`, `h_tiles` and `w_tiles` are folded into `out_page`.
`rows_in_tile` is sub-page.

**Inventory before solve.** There are two CBs, and the second exists only in `row_gather` on
Blackhole bf16. Patterns considered:
- **Pack into the destination.** Every page, gathered pieces included, is assembled directly in
  the slot the writer stores from. No input-side CB exists.
- **Transform in place.** Not applicable; nothing is transformed.
- **Alias disjoint lifetimes.** `cb_gather_staging` cannot alias `cb_output_pages`: both are live
  in the same block, staging being read while the output slots are written.
- **Fold into DEST.** Not applicable: no compute kernel, and a DEST round trip would convert
  values.

No L1 budget solve, safety fraction or blocking search exists. `block_pages` is
`min(core range, BLOCK_BYTES_CAP / page_bytes)`, and its reason is stated in the Blocking Model.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_output_pages` | `OUTPUT_PAGES_DEPTH · block_pages` = `2 · block_pages` | `2 · block_pages`: the reader holds block `i+1` while the writer holds block `i` | `{out_page: spans → block_pages × depth; lead, cat, mid, h_tiles, w_tiles: folded into out_page (no separate extent); rows_in_tile: spans → whole page (page_bytes per page; trimmed pages use the first v rows of the same slot)}` | input = output dtype (`Float16_b` Phase 0). Raw bytes: no DEST involved, so audit 2 does not apply (dataflow-only payload). | reader | writer | whole program | **Not shared.** It is the only reader→writer CB, and live for the whole program. Capacity exceeds one block by exactly one block, which is the double buffer (depth knob, overlap). |
| `cb_gather_staging` (`row_gather` ∧ `rb mod A_dram_read ≠ 0` only) | `block_pages + 1` | `block_pages` page-sized mirror slots, of which only the bytes of misaligned pieces are written. The +1 page holds the ≤ 32 B `shift` that runs past the last slot (alignment mechanism, see Mechanism caps). | `{out_page: spans → block_pages (one mirror slot per output page of the block); lead, cat, mid, h_tiles, w_tiles: folded into out_page; rows_in_tile: spans → the destination rows of the misaligned pieces, at dst_off + shift; input k: streams (pieces of every input land at disjoint destination rows of one slot)}` | dtype bytes (`Float16_b`); dataflow-only | reader | reader (private scratch: fetch writes it, place reads it) | one block: from `gather_fetch_block` to the end of `gather_place_block` | **Cannot share with `cb_output_pages`**: concurrent lifetime, since place reads staging while writing the output slots of the same block. Not allocated in `page_copy`, for fp32 (`rb = 64`), on Wormhole (`A_dram_read = 32`) or for all-L1 inputs (`A = 16`), because every piece is direct there. |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `page_bytes` | bf16 2048, fp32 4096, bf8_b 1088 (`output.buffer_page_size()`) | dtype ∈ TARGET; TILE layout |
| `BLOCK_BYTES_CAP` | 131072 (host constant) | — |
| `block_pages` | `1 ≤ block_pages ≤ block_pages_cap = max(1, BLOCK_BYTES_CAP // page_bytes)` (bf16 64, fp32 32, bf8_b 120); also `≤ n1` (pages of the most-loaded core) | `block_pages = min(n1, block_pages_cap)` |
| `OUTPUT_PAGES_DEPTH` | 2 (host constant) | — |
| `rb`, `A_dram_read` | `rb` ∈ {32 (bf16), 64 (fp32)}; `A_dram_read` = 64 Blackhole / 32 Wormhole (`noc_parameters.h:379`) | staging allocated iff `rb mod A_dram_read ≠ 0` |
| `out_l1_per_bank` | `ceil(P_out / num_l1_banks) · page_bytes` when the output is in L1, else 0 | `output_buffer == "l1"` (TARGET refinement) |
| `K` | `1 ≤ K ≤ K_MAX = 64` | host refusal above `K_MAX` (runtime-arg capacity) |
| `L1_OP_BUDGET` | 1347776 B per core (model environment, BB-05) | host `RuntimeError` if the total below exceeds it |

## Total per-core footprint

```
L1_op = OUTPUT_PAGES_DEPTH · block_pages · page_bytes                       # scales with block_pages, depth
      + [row_gather ∧ rb mod A_dram_read ≠ 0] · (block_pages + 1) · page_bytes  # scales with block_pages
      + out_l1_per_bank                                                     # scales with P_out (L1 output only)
```

| Case | `block_pages` | L1_op |
|------|---------------|-------|
| bf16 `page_copy`, cap reached (p2–p4) | 64 | 2·64·2048 = 262,144 B |
| fp32 `page_copy`, cap reached (p1) | 32 | 2·32·4096 = 262,144 B |
| bf16 `row_gather`, Qwen 79 (`n1 = 18` on the 110-core 11×10 Blackhole grid, measured plan) | 18 | 2·18·2048 + 19·2048 = 112,640 B |
| bf16 `row_gather`, worst case (cap) | 64 | 2·64·2048 + 65·2048 = 395,264 B |
| bf8_b `page_copy` with L1 output (Panoptic 23, refinement) | ≤ 120 | ≤ 2·120·1088 + ceil(P_out/banks)·1088 ≈ 261,120 + ~25 KB |

Every case stays well under `L1_OP_BUDGET = 1,347,776 B`. The cap leaves room for L1-resident
inputs and outputs of the model, which sit in the same L1.

## Data-movement budget

The chosen split is output-page ranges. Crossings are counted at the most expensive tier each
tensor touches: DRAM in Phase 0.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| each input `x_k` (`page_copy`) | 1 | Each input page is read once by the one core that owns its output page. A partially valid tile row with `v ≤ TRIM_ROWS_MAX` moves only its `v` valid rows. | none |
| each input `x_k` (`row_gather`) | 1, logical rows only | Each output tile row's run table reads disjoint row ranges. No padding row moves, and a straddling input tile is read as two disjoint ranges. | none. Core-local L1→L1: the misaligned pieces cross once more inside the core (bf16 on Blackhole, odd row shift). |
| output | 1 | Each output page is written once by its owner. The last tile row of an image is trimmed to `v` rows when `v ≤ TRIM_ROWS_MAX`. | none |

Totals per tier for the cases that set the bar (bytes):

| Case | DRAM read | DRAM write | Core-local L1→L1 | NoC cross-core |
|------|-----------|------------|------------------|----------------|
| p2 Qwen 17 (16 × (1,6,4096,256) bf16, dim 0) | 201,326,592 | 201,326,592 | 0 | 0 |
| p3 Qwen 19 (16 × (1,4096,1536) bf16, dim 0) | 201,326,592 | 201,326,592 | 0 | 0 |
| p4 Qwen 23 (8 × (1,4096,1536) bf16, dim 0) | 100,663,296 | 100,663,296 | 0 | 0 |
| p1 Qwen 18 (fp32, refinement) | 402,653,184 | 402,653,184 | 0 | 0 |
| Qwen 79 (16 × (1,1,1,62080) bf16, dim 2, `row_gather`) | 1,986,560 (whole-tile reads would be 63,569,920) | 1,986,560 (`v = 16`, one 1024 B write per page) | 993,280 (8 of 16 inputs have odd shift) | 0 |
| Qwen 160 (3 × (1,1,1,16384) + (1,1,1,12928) bf16, dim −1, `page_copy`, `v = 1`) | 124,160 (untrimmed: 3,973,120) | 124,160 (untrimmed: 3,973,120) | 0 | 0 |

> Cheapest-traffic split considered: output-page ranges. It is at the minimum on every tier:
> 1× per input and 1× for the output, with no cross-core traffic. Implemented: output-page ranges,
> which **is** the cheapest-traffic split. The two alternatives move the same DRAM bytes and are
> recorded in the Blocking Model's traffic ranking: input-page ranges (ties `page_copy`, adds
> partial writes in `row_gather`) and per-input core groups (unbalanced). No dependent-axis split
> and no multicast exist to consider.

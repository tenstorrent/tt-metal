# L1 Ledger: g6_reshape

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md`
(Blocking Model). Named block axes: `granule` (flattened `out_tile_col × out_image × row slab`),
`row_in_granule`, `face_col`, `line_elem`.

## Inventory

Two buffers. The minimum: staging cannot be removed (a line cannot be read into its output position on
Blackhole — DRAM reads need src ≡ dst mod 64 — so it lands at source parity and is written from there), and
the writer needs one private region for the pre-computed write plan (stall-shadow) and, in
`face_row_assemble` only, for the assembled lines (an element segment cannot be moved by the NoC: its
source and destination differ mod 16). Plan and assembly share one allocation (same owner, same lifetime).
There is no compute kernel, so no DEST, no intermediate tile CB, no scaler.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_staged_lines` (index 0) | `staging_depth` pages × `staging_page_bytes = align_up(64 + block_granules · L_g · M · 64, 64)` | every source line of one block, at source parity: all land before the push and all are read by the writes until `noc_async_writes_flushed` → the whole page is live at once. Peak = distinct source lines of the block × ≤ 64 B (bf16 32 B line + ≤ 32 B parity pad; fp32 64 B line, parity 0) | `{granule: spans → block_granules, row_in_granule: spans → min(R_g, H_out), face_col: spans → min(2, ceil(W_out/16)), line_elem: spans → whole line inside its 64 B slot}`; `out_tile_col`, `out_image` are flattened into `granule` | raw bytes, tagged with the input dtype; never unpacked — a dataflow-only payload that never passes through DEST (audit 2) | reader | writer | one block: reserve → reads → read barrier → push → wait → writes flushed → pop | **Not shared** with `cb_writer_scratch`: concurrent lifetimes (the writer fills its plan while this page is being filled and reads both during the store) and different producers. Capacity − live set: (a) +64 B so a run can align to 64 inside a page whose base is only 16-aligned; (b) `M` is the fan-in **bound** (assemble regime), the dedupe usually stages fewer lines; (c) the second page exists only when `staging_depth = 2` (more than one block per core) — the overlap mechanism |
| `cb_writer_scratch` (index 1) | 1 page × `writer_scratch_bytes = block_granules · L_g · (16 + A · (line_bytes + 12 · M))` | write plan (one 16 B entry per write run, ≤ one per output line) computed before the wait and read during the store; assemble regime adds copy segments (12 B each, ≤ `M` per line) and the assembly slots (`line_bytes` per output line) — all live together from `plan_block_writes` to the end of `store_output_lines` | `{granule: spans → block_granules, row_in_granule: spans → min(R_g, H_out), face_col: spans → min(2, ceil(W_out/16)), line_elem: spans → line_bytes per assembly slot (assemble only); streams → nothing (plan entries are per write run)}`; `out_tile_col`, `out_image` flattened into `granule` | raw bytes (plan words and assembled lines); no DEST involvement | writer | writer (private: `get_write_ptr` only, no reserve/push/wait/pop) | one block: plan → wait → assemble → store | Plan and assembly regions **share** this allocation (same owner, same lifetime). Cannot alias `cb_staged_lines` (concurrent, see above). Capacity − live set: write runs merge, so the plan usually holds fewer than one entry per line — the bound is kept so the capacity is closed-form |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `es` | element size | 2 (bf16, Phase 0); 4 (fp32 / int32 / uint32, refinements) | dtype |
| `line_bytes` | `16 · es` (one face row) | 32 or 64 | dtype |
| `rows_per_word` | `max(1, 64 / line_bytes)` | 1 or 2 | dtype |
| `R_g` | rows per granule | `rows_per_word ≤ R_g ≤ 16`, power of two | `R_g = rows_per_word`; raised to `min(16, next_pow2(ceil(16/W_out)))` only in `face_row_assemble` with `W_out < 16` |
| `L_g` | output lines per granule, bound: `min(R_g, H_out) · min(2, ceil(W_out/16))` | ≤ 4 when `W_out ≥ 16` (then `R_g ≤ 2`); ≤ 16 when `W_out < 16` (one face col) | definition of the granule |
| `M` | source lines per output line, bound | 1 (`face_row_direct`); 3 (`face_row_assemble`, `W_in ≥ 16`); `1 + ceil(15/W_in)` ≤ 16 (`face_row_assemble`, `W_in < 16`) | regime predicate `P_direct`; a 16-element window crosses ≤ 2 line boundaries when lines are ≥ 16 apart in one of every two gaps |
| `A` | 1 in `face_row_assemble`, 0 in `face_row_direct` | — | `P_direct` |
| `n1` | max granules per core | `ceil(total_granules / num_cores)` | `split_work_to_cores` |
| `block_granules` | granules per pipeline pass | `1 ≤ block_granules ≤ n1`; `= n1` when `staging_page_bytes(n1) + writer_scratch_bytes(n1) ≤ BLOCK_L1_CAP`, else the largest value with `2 · staging_page_bytes + writer_scratch_bytes ≤ BLOCK_L1_CAP` | `BLOCK_L1_CAP` clamp; one granule needs ≤ 21 KB, so ≥ 1 always fits |
| `staging_depth` | staging pages | 1 or 2 | 1 iff `ceil(n1 / block_granules) == 1` |
| `BLOCK_L1_CAP` | CB bytes per core the op may use | 1 048 576 | below the 1 347 776 B model budget (BB-05); never binds on golden INPUTS (max footprint 5 KB) — it exists so that unbounded N cannot overflow L1 |

## Total per-core footprint

```
L1_per_core = staging_depth · align_up(64 + block_granules · L_g · M · 64, 64)
            + block_granules · L_g · (16 + A · (line_bytes + 12 · M))
            ≤ BLOCK_L1_CAP
```

- Scales with `block_granules` (both terms), `L_g` (rows per granule × face columns), `M` (staging only,
  plus the copy-segment term in the assemble regime), `staging_depth` (staging only).
- Nothing scales with N, `W`, `H` or a tile count directly: N reaches L1 only through `block_granules ≤ n1`,
  and the `BLOCK_L1_CAP` clamp bounds that — a larger tensor adds blocks per core, never bytes per block.

Per golden shape (block = whole assignment, `staging_depth = 1`; 110-core grid):

| Shape | Regime | `R_g` | `n1` | `L_g` | `M` | staging B | scratch B | total B |
|-------|--------|-------|------|-------|-----|-----------|-----------|---------|
| `(1,16,6,256)→(1,1,16,1536)` | direct | 2 | 4 | 4 | 1 | 1088 | 256 | 1344 |
| `(16,1,64)→(1,1,16,64)` | direct | 2 | 1 | 4 | 1 | 320 | 64 | 384 |
| `(16,1,1536)→(1,1,16,1536)` | direct | 2 | 4 | 4 | 1 | 1088 | 256 | 1344 |
| `(1,1,16,4608)→(16,1,4608)` | direct | 2 | 21 | 2 | 1 | 2752 | 672 | 3424 |
| `(1,16,64)→(16,1,64)` | direct | 2 | 1 | 2 | 1 | 192 | 32 | 224 |
| `(16,1,5120)→(1,1,16,5120)` | direct | 2 | 12 | 4 | 1 | 3136 | 768 | 3904 |
| `(2,1,5120)→(1,1,2,5120)` | direct | 2 | 2 | 4 | 1 | 576 | 128 | 704 |
| `(1,8,6,256)→(1,1,8,1536)` | direct | 2 | 2 | 4 | 1 | 576 | 128 | 704 |
| `(8,1,64)→(1,1,8,64)` | direct | 2 | 1 | 4 | 1 | 320 | 64 | 384 |
| `(8,1,1536)→(1,1,8,1536)` | direct | 2 | 2 | 4 | 1 | 576 | 128 | 704 |
| `(1,1,8,4608)→(8,1,4608)` | direct | 2 | 11 | 2 | 1 | 1472 | 352 | 1824 |
| `(1,8,64)→(8,1,64)` | direct | 2 | 1 | 2 | 1 | 192 | 32 | 224 |
| `(8,1,5120)→(1,1,8,5120)` | direct | 2 | 6 | 4 | 1 | 1600 | 384 | 1984 |
| `(1,1,6,256)→(1,1,1536)` | direct | 2 | 1 | 2 | 1 | 192 | 32 | 224 |
| `(1,185)→(1,1,185,1)` bf16 (regime pin) | assemble | 16 | 1 | 16 | 3 | 3136 | 1344 | 4480 |
| `(1,512)→(1,1,512,1)` bf16 (regime pin) | assemble | 16 | 1 | 16 | 3 | 3136 | 1344 | 4480 |
| FLUX fp32 `(1,W)→(1,1,W,1)` (refinement) | assemble | 16 | 1 | 16 | 3 | 3136 | 1856 | 4992 |

## Data-movement budget

Chosen split: output-keyed granules, column-major `u = (tc·B_out + b)·S + s`, contiguous per core.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 (`face_row_direct`: each logical input line is read once, by the core that owns the output line it becomes); `face_row_assemble`: ≤ 2 for an input line whose output lines straddle a core boundary, exactly 1 on the FLUX geometries (`R_g = 16` keeps each input line on one core) | ownership by output granules over a bijection; read runs contain only needed lines; input padding is never read as a source | 0 |
| output | 1 (each valid output line written once; output tile padding rows never written) | single owner per line (single-writer 64 B words) | 0 |

Per golden shape (bf16 = 2 B; DRAM read and write bytes are equal because every production width is a
multiple of 16; transactions from a host model of the walk on the 110-core grid):

| Shape | Regime | Cores | Lines | DRAM read B | DRAM write B | Read runs | Write runs | Max transactions / core |
|-------|--------|-------|-------|-------------|--------------|-----------|------------|-------------------------|
| `(1,16,6,256)→(1,1,16,1536)` | direct | 110 | 1536 | 49 152 | 49 152 | 1536 | 1536 | 32 |
| `(16,1,64)→(1,1,16,64)` | direct | 16 | 64 | 2 048 | 2 048 | 64 | 64 | 8 |
| `(16,1,1536)→(1,1,16,1536)` | direct | 110 | 1536 | 49 152 | 49 152 | 1536 | 1536 | 32 |
| `(1,1,16,4608)→(16,1,4608)` | direct | 110 | 4608 | 147 456 | 147 456 | 492 | 4608 | 48 |
| `(1,16,64)→(16,1,64)` | direct | 32 | 64 | 2 048 | 2 048 | 64 | 64 | 4 |
| `(16,1,5120)→(1,1,16,5120)` | direct | 110 | 5120 | 163 840 | 163 840 | 5120 | 5120 | 96 |
| `(2,1,5120)→(1,1,2,5120)` | direct | 110 | 640 | 20 480 | 20 480 | 640 | 640 | 16 |
| `(1,8,6,256)→(1,1,8,1536)` | direct | 110 | 768 | 24 576 | 24 576 | 768 | 768 | 16 |
| `(8,1,64)→(1,1,8,64)` | direct | 8 | 32 | 1 024 | 1 024 | 32 | 32 | 8 |
| `(8,1,1536)→(1,1,8,1536)` | direct | 110 | 768 | 24 576 | 24 576 | 768 | 768 | 16 |
| `(1,1,8,4608)→(8,1,4608)` | direct | 110 | 2304 | 73 728 | 73 728 | 466 | 2304 | 28 |
| `(1,8,64)→(8,1,64)` | direct | 16 | 32 | 1 024 | 1 024 | 32 | 32 | 4 |
| `(8,1,5120)→(1,1,8,5120)` | direct | 110 | 2560 | 81 920 | 81 920 | 2560 | 2560 | 48 |
| `(1,1,6,256)→(1,1,1536)` | direct | 48 | 96 | 3 072 | 3 072 | 96 | 96 | 4 |
| `(1,185)→(1,1,185,1)` bf16 pin | assemble | 12 | 185 | 384 | 5 920 | 12 | 12 | 2 |
| `(1,512)→(1,1,512,1)` bf16 pin | assemble | 32 | 512 | 1 024 | 16 384 | 32 | 32 | 2 |

The assemble pins write more bytes than values (one valid element per 32 B output line, `W_out = 1`):
the line is the write quantum, and those lines merge into one write per face (512 B), which is fewer
transactions than one 2 B write per element.

**Totals per tier (chosen split):** DRAM = N·es read + N·es written on every production shape (the
minimum); cross-core NoC = 0; core-local = 0 in `face_row_direct`, N·es through RISC-V loads/stores in
`face_row_assemble`.

> Cheapest-traffic split considered: output-keyed granules in column-major order — DRAM bytes at the
> line-granular minimum (N·es each way), zero cross-core bytes, and the fewest read transactions of the
> equal-byte candidates (row-major order: 4608 vs 492 reads on `(1,1,16,4608)→(16,1,4608)`).
> Implemented: the same split. It is the cheapest; the regimes that would cut transactions further at the
> same bytes are the deferred `tile_page_copy` (identity tilings; no golden shape) and `rm_bank_run`
> (ROW_MAJOR refinement), both reachable on this walk/placer/writer structure.

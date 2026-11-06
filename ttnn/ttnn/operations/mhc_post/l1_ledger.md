# L1 Ledger: mhc_post

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes are taken from `op_design.md` → Blocking Model → Axes: `row`, `c`, `i`, `j`, `coef`, `k_w`, `stat_col`, `hc_q`. Every row accounts for all eight. `spans` means the live set scales with that axis's extent. `streams` means the buffer is reused as the axis advances. DEST is fp32 (`fp32_dest_acc_en = True` always), so every intermediate page is Float32. bf16 pages appear only where the value is *defined* as bf16 (X' / F / y in bf16 streams, and the bf16 X' that the projection and y must read).

## Phase 0 inventory (plain, built)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_coef_src` | `n_src` (constant 1–3: 2 for separate, 1 for packed, +1 for a separate P in collapse) | `n_src` | {row: streams, c: streams, i: streams, j: streams, coef: spans → n_src source tiles, k_w: streams, stat_col: streams, hc_q: streams} | Float32 (coefficients are fp32 by contract) | reader | reader (private scratch) | load_coefficients of one row | Not aliased with `cb_coef_cols`: the expansion reads the source while writing the column tiles, so the lifetimes overlap. Not packed into the destination: a source tile holds all coefficients interleaved by column, while each destination tile holds one coefficient in column 0. A 4–12 KiB constant. |
| `cb_coef_cols` | `n_coef_cols · COEF_DEPTH` = 20 · 1 (plain/stats), 24 · 1 (collapse) | `n_coef_cols` | {row: streams, c: streams (read by every block of the row), i: spans → n (comb rows), j: spans → n, coef: spans → n_coef_cols, k_w: streams, stat_col: streams, hc_q: streams} | Float32 (must stay fp32: unpack-to-dest broadcast is exact only for Float32, `bcast.h:46-53`; **implemented: tagged `UnpackToDestFp32`** — `unary_bcast` keys its unpack-to-dest branch on the CB's unpack *dst* format, which is TF32 when untagged; measured 8e-4 rel error untagged, `test_mhc_post_debug.py::post_path`) | reader | compute | one row (all its blocks) | Cannot share with the F/X CBs: it is concurrent with every block of the row. Capacity = live set (depth 1). Depth 2 is the `coef_depth` lamp. Only column 0 of each page is meaningful; a compact layout is impossible because `UnaryBcast<Col>` consumes whole tile pages. |
| `cb_f_tiles` | `block_c · IO_DEPTH` | `block_c` (one block, resident across the block's n output rows) | {row: streams, c: spans → block_c, i: streams, j: streams (re-read for each j inside the block), coef: streams, k_w: streams, stat_col: streams, hc_q: streams} | F dtype (Float32 in Phase 0; Float16_b for bf16 F, exact into fp32 DEST through srcA) | reader | compute | per block | Capacity − live set = one block, for double buffering (reader fills b+1 during b). Not aliased with `cb_x_tiles`: they are concurrent operands of the same chain. |
| `cb_x_tiles` | `n · block_c · IO_DEPTH` | `n · block_c` | {row: streams, c: spans → block_c, i: spans → n, j: streams (re-read per j), coef: streams, k_w: streams, stat_col: streams, hc_q: streams} | X dtype (Float32 in Phase 0) | reader | compute | per block | Double-buffering gap, as for `cb_f_tiles`. Not transformed in place into the output: every X_i tile feeds all n outputs j, so it must stay intact until the block's last output is packed. |
| `cb_out_tiles` | `n · block_c · OUT_DEPTH` (implemented `OUT_DEPTH = 2`; design value 1) | ≤ `n · block_c` per block window (compute pushes per tile; the writer writes each tile as it lands, then one barrier and one nominal-window pop per block; a ragged tail is padded to `n · block_c` pages by compute) | {row: streams, c: spans → block_c, i: streams, j: spans → n, coef: streams, k_w: streams, stat_col: streams, hc_q: streams} | X dtype (= X'.dtype; Float32 in Phase 0) | compute | writer | per block | Capacity = 2 windows so compute packs block b+1 while the writer barriers block b (measured: `OUT_DEPTH` 1 → 2 = −8 % at T=160/C=5120 and −10 % at T=4096/C=1792; flat on compute-bound shapes). Not packed into `cb_x_tiles`: X_i is still live (see above). |

### Phase 0 symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `n` | = 4 | `validate()`: n = post window width = 4 (DeepSeek hc_mult) |
| `n_src` | ∈ {1, 2, 3} | Number of distinct coefficient source pages per row: packed → 1; separate → 2; collapse with separate P → 3 |
| `n_coef_cols` | ∈ {20, 24} | n + n² (+ n in collapse) |
| `block_c` | 1 ≤ `block_c` ≤ `block_c_cap` | `choose_block_c`: `block_c_cap = ⌊(L1_CB_BUDGET − fixed_bytes)/per_c_bytes⌋` |
| `IO_DEPTH`, `COEF_DEPTH` | 2, 1 | host constants |
| `L1_CB_BUDGET` | 900 KiB | host constant (single source; WH and BH both have ≥ 1.4 MiB L1) |
| tile(F), tile(X) | 4096 (fp32) / 2048 (bf16) | dtype |

### Phase 0 footprint

```
fixed_bytes(plain)  = n_src·4096 + n_coef_cols·4096                       = 88 KiB (separate)  /  84 KiB (packed)
per_c_bytes(plain)  = IO_DEPTH·tile(F) + IO_DEPTH·n·tile(X) + OUT_DEPTH·n·tile(X)  = 72 KiB (fp32/fp32)  /  36 KiB (bf16/bf16)   [OUT_DEPTH = 2]
footprint(plain)    = fixed_bytes + block_c · per_c_bytes
block_c_cap         = ⌊(900 KiB − 88 KiB) / 72 KiB⌋ = 11 (fp32)   ;  ⌊(900 − 88)/36⌋ = 22 (bf16)
```

Scaling: `cb_f_tiles`, `cb_x_tiles` and `cb_out_tiles` scale with `block_c` (and, for the first two, with `IO_DEPTH`). `cb_coef_cols` scales with `COEF_DEPTH` and the mode. `cb_coef_src` is constant. At the focus shape (T=160, C=5120, fp32, `block_c = 2`): 88 + 2·72 = **232 KiB**.

## Fused inventory (stats / collapse, deferred regimes; designed now so the L1 is already accounted)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_xprime_mm` | `n · block_c` | `n · block_c` (both the projection and y read the whole block) | {row: streams, c: spans → block_c, i: spans → n (as y's input stream), j: spans → n (as X'_j), coef: streams, k_w: spans → n·block_c (matmul K window), stat_col: streams, hc_q: streams} | Float16_b (the bf16 X' as written; must be the same bits) | compute | compute | per block | Not shared with `cb_out_tiles`: that one is consumed by the writer, and two consumers on one CB are illegal. Repacking from DEST costs one pack, not a DRAM read. |
| `cb_w_slice` | `n · core_c_max` | `n · core_c_max` (resident whole kernel) | {row: streams (resident across rows: the residency that avoids re-reading W per row), c: spans → core_c_max, i: streams, j: spans → n, coef: streams, k_w: spans → n·core_c_max, stat_col: spans → 1 tile (32 cols), hc_q: streams} | Float32 (W is fp32 with 9-bit mantissas; bf16 would round it) | reader | compute | whole kernel | Bounded by `W_RESIDENT_BUDGET` through `choose_split`. Resident by design: streaming it per row is the `w-stream` rejected regime. |
| `cb_proj_block` | 1 | 1 | {row: streams, c: streams, i: streams, j: streams, coef: streams, k_w: streams (contracted), stat_col: spans → 1 tile, hc_q: streams} | Float32 | compute | compute | per block | Not folded into `cb_proj_run` by matmul accumulation: the matmul's DEST window cannot persist across the interleaved mix chains. Fold candidate: matmul packer-L1-acc into `cb_proj_run` (ss-accumulation lamp family). |
| `cb_proj_run` | 2 | 1 (ping-pong: pop 1 / push 1) | {row: streams, c: streams (accumulates over blocks), i: streams, j: streams, coef: streams, k_w: streams, stat_col: spans → 1 tile, hc_q: streams} | Float32 | compute | compute | per row | Capacity 2 gives real CB credit between successive in-place updates (Rule 3 race). Could alias `cb_partial` only if `cb_partial` were compute-consumed; it is writer-consumed. |
| `cb_ss_run` | 2 | 1 (ping-pong per output tile) | {row: streams, c: streams, i: streams, j: streams, coef: streams, k_w: streams, stat_col: streams, hc_q: streams; elementwise 32×32 running sum} | Float32 (exact ss) | compute | compute | per row | Same credit reason as `cb_proj_run`. Concurrent with `cb_proj_run` (both live all row), so they cannot alias. |
| `cb_ss_row` | 1 | 1 | {all axes: streams; per-row tile, col 0 valid} | Float32 | compute | writer | per row | Could share an allocation with `cb_partial` (both 1 page, same lifetime), but they are two values the writer merges, so both are live at once. |
| `cb_partial` | 1 | 1 | {row: streams, stat_col: spans → 1 tile; others: streams} | Float32 | compute | writer | per row (until shipped) | See `cb_ss_row`. |
| `cb_zero` | 1 | 1 (constant) | {all axes: streams} | Float32 | compute | compute | whole kernel | A constant seed tile. Alternative: seed with `FillScalar` straight into DEST on each row's first block, which would delete this CB. The implementer may take that and drop the row. |
| `cb_gather` | `COMBINE_FANIN · COMBINE_DEPTH` = 32 | ≤ fan-in per rooted row in flight | {row: spans → COMBINE_DEPTH rooted rows, c: spans → fan-in partials (one per column-group core), others: streams} | Float32 | reader (push after semaphore; data written by remote writers) | compute | per rooted row | The root's second-level slots reuse this CB: root fan-in ⌈Q/16⌉ ≤ 9 < 16, and leader/root roles for one row are sequential on one core. Not shrunk to a pull ring: a push with fixed slots gives order-independent arrival in one round. |
| `cb_stats_row` | 1 | 1 | {row: streams, stat_col: spans → 1 tile; others: streams} | Float32 | compute | writer | per rooted row | Not shared with `cb_partial`: on a root they can be live together (its own partial and the combined row). |
| `cb_hc_lanes` | 3 (2 quantity tiles + 1 constants lane tile) | 3 | {row: streams, hc_q: spans → 25 quantities (2 tiles) + constants; others: streams} | Float32 | writer | compute | per rooted row (constants: whole kernel) | Not shared with `cb_hc_out`: the hc program runs in DEST, but the writer's inverse permute of row r can overlap the permute of row r+1. A candidate to alias when `COMBINE_DEPTH = 1`. |
| `cb_hc_out` | 2 | 2 | {row: streams, hc_q: spans → 24 outputs; others: streams} | Float32 | compute | writer | per rooted row | See above. |
| `cb_hc_row` | 1 | 1 | {row: streams, hc_q: spans → 24 cols; others: streams} | Float32 | writer | writer (private scratch) | per rooted row | Could alias `cb_coef_src` (both reader/writer scratch). It cannot: different kernels own them, and they are live concurrently. |
| `cb_y_tiles` | `block_c` | 1–2 (per-tile push) | {row: streams, c: spans → block_c, i: streams (contracted), others: streams} | Float16_b (y is bf16 by contract) | compute | writer | per block | Transfer-quantum gap, as for `cb_out_tiles`. |

### Refinement 1 — stats inventory as built (departures from the table above)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_partial` | 2 (design 1) | 1 | as designed | Float32 | compute | writer | per row (until shipped) | Depth 2 so compute can finish row r+1's partial while the writer waits for the leader's grant of row r. |
| `cb_lead_out` (new) | 1 | 1 | {row: streams, stat_col: spans → 1 tile; others: streams} | Float32 | compute | writer | per led row | The leader's group sum, shipped to the root (two-level only). Not shared with `cb_stats_row`: on a core that both leads and roots, both are live and the writer drains them in a different order. |
| `cb_comb_acc` (new) | 2 | 1 (ping-pong) | {row: streams, c: streams (accumulates over gather slots), stat_col: spans → 1 tile; others: streams} | Float32 (UnpackToDestFp32) | compute | compute | per combine | Running sum of the combine (seed g0, then `acc + g[s]` in slot order in one streaming chain). Separate from `cb_proj_run`: a leader's combine of row r−1 runs between its row-r blocks, while `cb_proj_run` holds row r's running projection. |
| `cb_root_gather` (new; design reused `cb_gather`) | `num_groups · COMBINE_DEPTH` (two-level only) | `num_groups` per rooted row in flight | {row: spans → COMBINE_DEPTH rooted rows, c: spans → num_groups leader tiles; others: streams} | Float32 (UnpackToDestFp32) | reader (push after semaphore) | compute | per rooted row | Separate from `cb_gather`: grants run LAG rows ahead of collects, so a root's leader-level and root-level slot sets are live at once and their ring positions advance at different rates. |
| `cb_proj_masks` (new) | `PROJ_EXACT_PASSES` = 16 (HiFi3/HiFi4 only) | 16 (resident whole kernel) | {all axes: streams; constant} | Float16_b | reader (NoC-zeroed, 64 lanes/tile written) | compute | whole kernel | Column masks {j, j+16} for the exact projection. Needed because the FPU rounds every 16-term face partial sum to ~tf32 (measured; see changelog), so the projection runs 16 passes with one product per face. |
| `cb_xprime_masked` (new) | `n · block_c` (HiFi3/HiFi4 only) | `n · extent` | {c: spans → block_c, j: spans → n, k_w: spans → n·block_c; others: streams} | Float16_b | compute | compute (matmul in0) | per pass | One masked X' block, re-made 16 times per block. It cannot be packed into `cb_xprime_mm` in place, because the unmasked block feeds all 16 passes. |
| `cb_zero` | **dropped** | — | — | — | — | — | — | `cb_ss_run` is seeded by a `FillScalar` chain into DEST at each row start (ledger fold option taken). `cb_proj_run` needs no seed: the row's first block packs straight into it. |
| `cb_hc_*`, `cb_y_tiles` | not built (collapse, Refinement 2) | — | — | — | — | — | — | Reserved in `fused_fixed_bytes` / `fused_per_c_bytes` so the split is already mode-invariant. |

Sizing (host, `mhc_post_program_descriptor.py`): `fused_fixed_bytes` = the collapse inventory above (n_src 3, 24 coefficient columns, the resident W slice, the combine CBs, the hc CBs) plus the 16 mask tiles. `fused_per_c_bytes` = `plain_per_c_bytes` + `cb_xprime_mm` + `cb_xprime_masked` + y. Both are charged at every fidelity and in both fused modes, so `(R, Q, block_c)` depends only on shape, dtypes and coef_source. Built focus split (bf16, packed, T=160, C=5120): R=2, Q=54, 3 rows × 3 c-tiles per core, `block_c = 2`.

Semaphores (fused): `2·COMBINE_DEPTH` arrival counters (leader / root, one per slot set, cumulative), 1 leader-credit counter, and `num_groups` root-credit counters. That is 14 at Q=130, within the 16-id budget; the host asserts it.

### Fused symbol table (additional)

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `core_c_max` | `n·core_c_max·4096 ≤ W_RESIDENT_BUDGET = 384 KiB` → `core_c_max ≤ 24` | `choose_split` constraint (fused modes) |
| `COMBINE_FANIN` | 16 | host constant; Q > 16 → two-level tree |
| `COMBINE_DEPTH` | 2 | host constant |

### Fused footprint (sized against the collapse inventory, for both fused modes, so the split is mode-invariant)

```
fixed_bytes(fused) = (n_src + n_coef_cols)·4096                      # 3 + 24 tiles  = 108 KiB
                   + n·core_c_max·4096                               # W slice        ≤ 384 KiB
                   + (1 + 2 + 2 + 1 + 1 + 1)·4096                    # proj_block, proj_run, ss_run, ss_row, partial, zero = 32 KiB
                   + COMBINE_FANIN·COMBINE_DEPTH·4096                # gather         = 128 KiB
                   + (1 + 3 + 2 + 1)·4096                            # stats_row, hc_lanes, hc_out, hc_row = 28 KiB
per_c_bytes(fused, bf16) = IO_DEPTH·2048 + IO_DEPTH·n·2048 + n·2048 + n·2048 + 2048   # F, X, out, xprime_mm, y = 38 KiB
footprint(fused)   = fixed_bytes(fused) + block_c·per_c_bytes(fused)
```

At the focus shape (bf16, R=2, Q=54, `core_c_max = 3`, `block_c = 1`): 108 + 48 + 32 + 128 + 28 + 38 = **382 KiB**. Worst case (`core_c_max = 24`, `block_c_cap = ⌊(900 − 680)/38⌋ = 5`): 680 + 5·38 = 870 KiB ≤ 900 KiB.

## Data-movement budget

### Phase 0 focus (plain, fp32/fp32, separate; T=160, C=5120; R=5, Q=20)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| F (3.28 MB) | 1 | Each (row, c) tile is owned by exactly one core. | 0 |
| X (13.1 MB) | 1 | Same: all n stream tiles of column c are read by the owner of c. | 0 |
| X' (13.1 MB) | 1 (write) | Written once by the owner. | 0 |
| post (20 KB of tiles) | Q = 20 (0.41 MB) | Reuse-shared along c: every column group of the row reads it (`coef-mcast` deferred). | 0 |
| comb (20 KB of tiles) | Q = 20 (0.41 MB) | Same as post. | 0 |

Totals: DRAM 30.3 MB (minimum 29.5 MB; +2.7 %); NoC cross-core 0; core-local re-reads: F re-read n× and coefficient tiles re-read per block from L1 (`Row` / `Col` mappings), no DRAM cost.

> Cheapest-traffic split considered: `R = 5, Q = 1` (coefficients read once): −0.78 MB DRAM (−2.6 %), but it uses only 5 cores, so the busiest core moves 20× more (5.9 MB vs 296 KB). Implemented: `(R, Q) = (5, 20)` from `choose_split`. The cheapest-traffic split is a point of the built `(R, Q)` family, not a missing algorithm. The traffic it saves is recovered without losing occupancy by the `coef-mcast` deferred regime (positive reason: ≤ 3 % of bytes at every INPUTS shape, and it layers on unchanged).

### Fused focus (collapse, bf16, packed; T=160, C=5120; R=2, Q=54)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| F (1.64 MB) | 1 | owner-only | 0 |
| X (6.55 MB) | 1 | owner-only | 0 |
| X' (6.55 MB) | 1 (write) | The projection / ss / y read it from L1 (`cb_xprime_mm`, DEST), never from DRAM. | 0 |
| y (1.64 MB) | 1 (write) | From the core's own X' block. | 0 |
| mixes (= post = comb = P; 20 KB) | Q = 54 (1.1 MB) | Reuse-shared along c. Packed means one page serves all three roles. | 0 |
| W (2.62 MB) | R = 2 (5.24 MB) | Reuse-shared along rows. Resident per core (read once per core, not per row). | 0 |
| partial stats tiles | — | — | 270 × 4 KiB = 1.1 MB core → leader; 5 rows × 4 × 4 KiB = 80 KB leader → root |
| stats (20 KB), hc (20 KB) | 1 (write) | written by the row's root | 0 |

Totals: DRAM ≈ 22.8 MB (minimum ≈ 19.1 MB with W and the mixes once); NoC ≈ 1.2 MB in 4 KiB unicasts.

> Cheapest-traffic split considered: `R = 1, Q = 80` (W once). −2.6 MB of W, +0.5 MB of mixes ⇒ −2.1 MB DRAM (−9 %); NoC partials 400 × 4 KiB = 1.6 MB (+0.4 MB). Implemented: `(R, Q) = (2, 54)`, because `choose_split` minimizes the busiest core's bytes (227 KB vs 232 KB). Within the built family this is a choice of objective, and it is the `choose_split objective` lamp in `op_design.md`. The W re-read itself is removed without changing the split by the `w-mcast` deferred regime (positive reason: the stats scheme must exist first, and the mcast layers on it unchanged).

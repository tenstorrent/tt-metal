# Operation Design: mhc_post

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (streaming eltwise mix, plus a cross-core row reduction and a per-token SFPU program in the fused modes) |
| Goal | Write the n = 4 new residual streams `X'_j = post_j·F + Σ_i comb[i][j]·X_i` in one device program. Optionally, from the same pass, also produce the next hc step's stats row (`X'@W` plus the sum of squares of the unrounded mix), its coefficients `hc` (DeepSeek-V4 Sinkhorn), and its collapsed input `y = Σ_i P_i·X'_i`. |
| Math | `X'[t, j·C + c] = post[t, post_col+j]·F[t,c] + Σ_i comb[t, comb_col+i·n+j]·X[t, i·C+c]` (comb applied **transposed**), computed in fp32 and rounded once to X.dtype. `stats[t,k] = Σ_m X'[t,m]·W[m,k]` for k ≠ 24; `stats[t,24] = Σ_m mix[t,m]²`. `hc[t] = deepseek_hc(stats[t])`. `y[t,c] = Σ_i P[t,i]·X'[t, i·C+c]`. |
| References | `eval/golden_tests/mhc_post/helpers.py` (`pytorch_mhc_post`, `deepseek_hc`, `pytorch_mhc_post_collapse`, all gates); `eval/golden_tests/mhc_post/feature_spec.py`; `models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py` (`TtMHCWrap.hc_post`, the composite being replaced); `models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py`; `models/demos/deepseek_v3_d_p/reference/mhc/deepseek_sinkhorn_kernel.py` (hc step order of operations: softmax with row-max, `+eps`, column step first, then `iters-1` row/column pairs); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; perf catalog `ttnn/ttnn/operations/examples/master.md`. |
| Companion artifact | `ttnn/ttnn/operations/mhc_post/l1_ledger.md`: the per-CB ledger and the data-movement budget. |

Notation used throughout: `n = 4` (`hc_mult`, taken from `post` / `comb` geometry and validated to equal 4), `NG = n(n+2) = 24`, `SS_COL = NG = 24`. One tile is 32×32: 4096 B in fp32 and 2048 B in bf16.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (F) | Tensor | yes | (…, T, C), TILE, DRAM interleaved, C % 32 == 0 | — | buffer address RT |
| `residual` (X) | Tensor | yes | (…, T, n·C), same leading dims as F | — | buffer address RT |
| `post` | Tensor | yes | float32 TILE (…, T, Wp). The n values sit at columns `post_col..post_col+n-1`, all inside one 32-wide tile column. | — | address RT |
| `comb` | Tensor | yes | float32 TILE (…, T, Wc). The n² values sit at `comb_col..comb_col+n²-1`, all inside one 32-wide tile column. | — | address RT |
| `compute_kernel_config` | ComputeConfigDescriptor | no | `fp32_dest_acc_en` must be True; any `math_fidelity` | `default_compute_kernel_config()` = HiFi4, fp32 DEST, approx off | math_fidelity / approx feed the ComputeConfigDescriptor (program-level) |
| `post_col` | int | no | `post_col % 32 + n ≤ 32`, `post_col + n ≤ Wp` | 0 | RT (reader) |
| `comb_col` | int | no | `comb_col % 32 + n² ≤ 32`, `comb_col + n² ≤ Wc` | 0 | RT (reader) |
| `stats_weight` (W) | Tensor | no | float32 TILE (1.., n·C, 32) | None | address RT; its presence selects mode `stats` |
| `collapse_pre` (P) | Tensor | no | float32 TILE (…, T, ≥n), pre in columns 0..n-1; requires W | None | address RT; its presence selects mode `collapse` |
| `hc_scale` | list[float] (3) | collapse only | finite | — | RT (compute, fp32 bits) |
| `hc_base` | list[float] (NG) | collapse only | finite | — | RT (writer builds the hc constants block from these, fp32 bits) |
| `hc_norm_width` | float | no | > 0 | n·C | RT (compute: fp32 bits of `1/hc_norm_width`, computed in double on the host) |
| `hc_norm_eps` | float | no | ≥ 0 | 1e-6 | RT (compute) |
| `hc_eps` | float | no | ≥ 0 | 1e-6 | RT (compute) |
| `hc_sinkhorn_iters` | int | no | ≥ 1 | 20 | RT (compute loop bound) |
| `output_tensor` / `stats_output` / `hc_output` / `y_output` | Tensor | no | must equal what the op would allocate in shape, dtype, layout and memory config, and must not alias an input | None | address RT |
| `max_grid` | CoreCoord | no | 1 ≤ x, y; clamped to `device.compute_with_storage_grid_size()` | whole grid | host. Sets G, the core-grid knob source. |

Index canonicalization: the op has no `dim` argument. Leading dims fold into token rows (see Work Distribution).

Mode is derived from the kwargs: `plain` = neither W nor P; `stats` = W only; `collapse` = W and P. P without W raises `ValueError` (not a support refusal). In collapse mode, a missing `hc_scale` / `hc_base` or one of the wrong length raises `ValueError`. `coef_source = "packed"` iff `post` and `comb` are the same device buffer (`buffer_address()` equal).

## Tensors

### Input

| Property | F | X | post / comb | W | P |
|----------|---|---|-------------|---|---|
| Shape | (…, T, C) | (…, T, n·C) | (…, T, Wp) / (…, T, Wc) | (1.., n·C, 32) | (…, T, Wpre ≥ n) |
| Dtype | fp32 / bf16 (`sublayer_dtype`) | fp32 / bf16 (`dtype`) | fp32 | fp32 | fp32 |
| Layout | TILE | TILE | TILE | TILE | TILE |
| Memory | DRAM interleaved | DRAM interleaved | DRAM interleaved | DRAM interleaved | DRAM interleaved |

### Output

| Property | X' | stats | hc | y |
|----------|----|-------|----|---|
| Shape | = X.shape | (…, T, 32) | (…, T, NG) | (…, T, C) |
| Dtype | = X.dtype | fp32 | fp32 | bf16 |
| Layout | TILE | TILE | TILE | TILE |
| Memory | DRAM interleaved | DRAM interleaved | DRAM interleaved | DRAM interleaved |
| Mode | all | stats, collapse | collapse | collapse |

Return value: plain → `X'`; stats → `(X', stats)`; collapse → `(X', stats, hc, y)`.

## Registry contract (op file)

| Item | Value |
|------|-------|
| `INPUT_TAGGERS` | Exactly one, `"alignment"`. It receives `(F_shape, X_shape, post_shape, comb_shape)` and returns `"tile_aligned"` if `X_shape[-2] % 32 == 0`, else `"h_non_aligned"`. |
| Phase 0 `SUPPORTED` (written by the implementer) | `dtype [float32]`, `sublayer_dtype [float32]`, `layout [TILE]`, `fp32_dest_acc_en [True]`, `mode [plain]`, `coef_source [separate]`, `alignment [tile_aligned, h_non_aligned]` |
| `validate()` order | First line of `mhc_post`. (1) Structural errors (`ValueError`): P without W, hc scalars missing or the wrong length in collapse, n ≠ 4, a column window crossing a tile column, mismatched leading dims, C % 32 ≠ 0, a bad caller buffer. (2) Per-axis `UnsupportedAxisValue` in TARGET order: dtype, sublayer_dtype, layout, fp32_dest_acc_en (False → refuse), mode, coef_source, alignment. (3) `ExcludedCell` for EXCLUSIONS. INVALID is never checked. |
| `PROPERTIES` | `multi_core: declared True` (Phase 0 spreads both independent axes across the grid). |

Column offsets work from Phase 0 for `coef_source = separate`. The reader reads each coefficient at `col % 32` of tile column `col // 32` (see Dataflow), so the engram-add form `post_col = 12` is a plain-mode `separate` call and needs no refinement.

## Blocking Model

### Axes

The op's index space is token-tile-rows (all leading dims plus T, folded) × column tiles of C × input stream i × output stream j. The fused modes add the 32 stats columns and the hc quantities. Characters are derived from the math above.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `row` (token-tile row; leading dims + T, per-image `ceil(T/32)`) | **independent**: every output is token-row-local (mix, matmul row, ss, hc lanes, y). | `block_rows` | 1 | host constant `BLOCK_ROWS = 1` | Balanced contiguous split into `R` row groups; Phase 0 spreads rows across the grid. | knob-turn |
| `c` (column tile of C, `tensor_c_tiles = C/32`) | **independent** for X' and y: tile `(r, j, c)` needs only column c of F and of every X_i. **Dependent** for the stats row: the projection and ss sum over all n·C columns. | `block_c` | `choose_block_c(...)` (Work Distribution): the coarsest value ≤ the L1 cap that still leaves ≥ `MIN_PIPELINE_BLOCKS` blocks in the core's pipeline | host `choose_block_c` → CT arg `block_c` | Balanced contiguous split into `Q` column groups. The `(R, Q)` pair is chosen per call by `choose_split`. | plain: knob-turn. Stats: scheme-change (cross-core combine, designed below). |
| `i` (input stream, n=4) | **dependent**: `X'_j` contracts over i. | `block_streams_in` | n (whole) | constant `n` from tensor geometry | Never split: the contraction stays inside one core. | none (n=4 fixed) |
| `j` (output stream, n=4) | **independent**, but X_i is **reuse-shared** along j: every j reads every X_i. | `block_streams_out` | n (whole) | constant `n` | Never split: splitting j across cores re-reads X n times from DRAM (`rejected` row). | none |
| `coef` (the n + n² (+n) per-token coefficients) | **reuse-shared** along `c`: they do not vary with the column, so every column group re-reads them. | `n_coef_cols` | n + n² = 20 (plain/stats); n + n² + n = 24 (collapse) | derived from mode | Read once per (core, row). | multicast across a row group: `deferred` row |
| `k_w` (W's rows = n·C, projection contraction) | **dependent** (projection) and **reuse-shared** along `row`: W does not vary with the token, so every row group re-reads its W slice. | resident slice `n × core_c_tiles` (stats/collapse) | not built in Phase 0 | derived from `core_c_tiles` | Each core holds the W rows of its own columns. Split ≡ the `c` split. | W multicast across row groups: `deferred` row |
| `stat_col` (32 stats columns) | **independent** (one W column each); the whole width is one tile. | — (1 tile, whole) | 1 tile | fixed by the stats contract (…, T, 32) | Not split: one tile is the atom. | none |
| `hc_q` (25 hc quantities per token) | **dependent**: softmax and Sinkhorn mix quantities. | — (whole, 2 lane tiles) | whole | fixed by NG + 1 | Not split: runs on the row's root core, lane-parallel over the 32 tokens. | none |

Why `block_rows = 1`: the 20–24 coefficient column tiles differ per token row. A block spanning `b` rows would hold `b×` the coefficient tiles and still run `b` separate chains, because the chain's broadcast operand changes per row. Every fixed cost a block amortizes (chain init, CB handshakes, transfer batching) is already amortized over `n × block_c` output tiles by the `c` extent. The row axis is therefore the outer block walk, not a block extent. Rows are still spread across cores (`R`). This is a decision, not a collapsed knob: raising it buys nothing until a row-batched coefficient layout exists.

### Intermediate stages re-tabled (the scheme's own data)

| Stage data | Axes | Character | Assignment | Why |
|-----------|------|-----------|-----------|-----|
| Per-core partial stats tile (proj + ss in col 24), one per (core, row) | row, stat_col | row independent; partials of one row are summed (dependent over column groups) | Each row's combine is rooted on a **different** core of its row group (round-robin over leaders), so combines and hc steps run on as many cores as there are rows in flight. | A fixed root would serialize all rows' combines and hc on one core while the rest of the group idles. |
| Combined stats row | row, stat_col | independent across rows | On the row's root core. | One tile per row. Nothing to spread inside a row. |
| hc lanes (25 quantities × 32 tokens) | row, hc_q | rows independent, hc_q dependent | On the row's root core. 32 tokens = the lanes of one SFPU program. | hc_q cannot be split (Sinkhorn couples it). Rows already go to distinct roots. |
| y tile (r, c) | row, c | independent | Same core as the X' tile (r, c). | Needs only that core's X'_{i,c}. |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_f_tiles` | `io_depth` | 2 blocks | Reader fills block b+1 while compute mixes block b. |
| `cb_x_tiles` | `io_depth` | 2 blocks | Same as `cb_f_tiles`. |
| `cb_out_tiles` | `out_depth` | 1 block (`n·block_c` pages), pushed per tile | The writer drains tile by tile while compute packs, so a single block of capacity already overlaps pack with write. |
| `cb_coef_cols` | `coef_depth` | 1 row | Holds the row's broadcast operands for all its blocks. Depth 2 would hide the row-boundary expansion (perf lamp). |
| `cb_coef_src` | — | 1 row (reader-private scratch) | Nothing to overlap: the reader consumes it immediately. |
| fused only: `cb_xprime_mm` | 1 block | — | Compute-internal hand-off from the mix to the projection and y. |
| fused only: `cb_gather` | `combine_depth` | 2 rows in flight | Senders can ship row r+1's partial before the leader has consumed row r's. |

All depths are host constants (single source) feeding both the CB capacity and the kernels' CT args.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DEST capacity, fp32 half-sync: `DEST_AUTO_LIMIT = 4` (`ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:103`) | Live DEST slots per chain iteration: acc, coef, data (+ ss in the fused modes) = 3 (4) | Mix chain fixed at slots D0..D2 (D3 for ss). Each iteration is one (j, c) tile, so slot use is independent of `block_c`. | `static_assert` in every chain element. Holding all of F, X_0..3 and 4 accumulators (≥ 9 slots) is illegal, which is why outputs are produced one (j, c) at a time. |
| `UnaryBcast<Col>` reads column 0 only (`chain.hpp:324-325`; fp32 path = unpack-to-dest, `tt_metal/hw/inc/api/compute/bcast.h:46-53,86-95`) | The coefficient's position inside its tile | The reader writes every coefficient into column 0 of its own tile in `cb_coef_cols`. | Broadcasting from `post_col ≠ 0` without the expansion broadcasts the wrong coefficient. |
| Coefficient tile column: n (n²) values must lie within one 32-wide tile column | `post_col`, `comb_col` | `validate()` raises `ValueError` if `col % 32 + width > 32` | The reader would read half the values from the wrong tile. |
| Upfront-wait chain operands must be resident | `block_c` (F, X resident across the n output rows of a block) | `block_c ≤ block_c_cap(L1)` | CB deadlock: wait count > capacity. |
| W residency (fused modes) | `core_c_tiles` (W slice = `n·core_c_tiles` fp32 tiles) | `choose_split` constraint `n·core_c_tiles·4096 ≤ W_RESIDENT_BUDGET` | Out of L1, or W streamed per row (a re-read; `rejected` row). |
| Combine fan-in (fused) | Partials per leader | `combine_fanin` (`COMBINE_FANIN = 16`) → two-level tree when Q > 16 | Gather buffer `Q·4 KiB·depth` grows with Q (520 KiB at Q=130); catalog `tensix_all_reduce`: a flat root's gather buffer limits L1. |
| hc lane layout | 32 tokens per lane block | Fixed: one token-tile row per hc program | — |
| CB ring wrap (`ttnn-cb-memory-fundamentals.md`) | Ragged last c-block | Reader/compute/writer push and pop the **nominal** `block_c`-sized counts; only valid tiles are read, computed and written. | Pointer misalignment → silent corruption on the next block. |

### Regimes

Named boundary: **DRAM**. Minimum = every input byte read once, every output byte written once. For coefficients, minimum = each coefficient tile read once in total.

| Regime | Status | Predicate | Block | Data movement (vs. minimum) | What a bigger block buys |
|--------|--------|-----------|-------|-----------------------------|--------------------------|
| **plain-2d** | **built (Phase 0)** | `mode == plain` | `1 row × block_c c-tiles × n streams` (inputs F: block_c, X: n·block_c; outputs n·block_c) | F, X read once; X' written once. Coefficient source tiles are read once per (row, column group): `Q × rows × n_src` tiles instead of `rows × n_src`. At the focus shape (Q=20) that is +1.4 % of bytes. | Per block: one CB handshake per operand (wait/pop of `(1+n)·block_c` pages and one `n·block_c` window on the writer), one chain call (its element inits repeat per tile because unpack modes alternate; see lamp), and one NoC barrier per operand batch. Larger `block_c` pays these fewer times, until the overlap lamp. |
| **stats-2d-tree** | deferred: every stats cell needs `dtype = sublayer_dtype = bfloat16` (fp32 is INVALID for stats), and Phase 0 supports fp32 only. It layers on plain-2d unchanged: same block, same split family, same mix chain. | `mode == stats` | plain block + resident W window `n·block_c` tiles per block | + W read once per row group (`R × n·C·32·4 B`). `choose_split` prices it and picks R=2 at the focus shape, so W crosses DRAM twice. + Cross-core: one 4 KiB partial per (core, row) to its leader, one per (leader, row) to the root. | Adds per block: one matmul call (`K = n·block_c`) and one SFPU partial add. Larger `block_c` amortizes the matmul's init and the partial add. |
| **collapse-2d-tree** | deferred: needs stats-2d-tree plus bf16. Layered on stats-2d-tree: adds y in the mix pass and the hc step on the root. | `mode == collapse` | stats block + y `block_c` tiles | + P source tile once per (row, column group); none extra when packed (same tile as the coefficients). y written once; hc once per row. | The y chain per block reuses the resident `cb_xprime_mm` block. |
| **coef-mcast** | deferred: the coefficient re-read is Q × 4–12 KiB per row (≤ 3 % of bytes at every INPUTS shape). It would layer on any regime above: one core per row group reads the source tile(s) and multicasts them (`ttnn.Mcast`, `mcast_args.hpp:309`) to the group's `cb_coef_src`. | `Q > 1` | unchanged | Coefficient reads `Q·rows·n_src` → `rows·n_src` tiles. Adds one mcast per row. | — |
| **w-mcast** | deferred: `choose_split` already prices W per row group (focus shape R=2: one extra W read, 2.6 MB, ≈ 12 % of the call's bytes; the mcast saves at most that, and adds an R-way multicast). The successor layers on stats-2d-tree: a W injector per column group multicasts the group's W slice once to the R cores sharing those columns (catalog `shared_input_reuse`: 1.71×; `mcast_topology`). | `mode ≠ plain and R > 1` | unchanged | W reads `R·|W|` → `|W|`. Adds R-way mcast of the W slice. | — |
| **stream-split** (split j or i across cores) | rejected: superseded by plain-2d. Splitting j re-reads every X_i per stream group (`n×` X traffic); splitting i needs a cross-core combine for every output tile. A dead end: no good scheme passes through it. | — | — | X read up to n× (j split), or every X' tile combined across cores (i split). | — |
| **fpu-mix** (FPU `mul_tiles_bcast` / binary with bcast) | rejected: the FPU reads fp32 operands through tf32 by truncation, a −7e-4 relative bias per call that fails the signed-bias gate (`helpers.py SIGNED_BIAS_FLOOR`). Superseded by the SFPU mix in DEST. A split-precision FPU form for bf16 streams is a perf lamp, not a regime. | — | — | same | — |
| **dram-reread-stats** (a second pass reading X' back for projection / ss) | rejected: forbidden by contract ("MUST NOT re-read X' from DRAM"), and it adds one full X' read. Superseded by stats-2d-tree (projection from the X' block already in L1). | — | — | + one X' read | — |
| **w-stream** (W streamed per row instead of resident) | rejected: a re-read of W once per row per core (`rows ×` W). Superseded by resident W bounded via `choose_split`. | — | — | `rows_per_core ×` W | — |

Selection predicate: the regime is exactly `mode` (from kwargs). The `(R, Q, block_c)` extents come from `choose_split` / `choose_block_c`, which are host-pure functions of (T, C, dtypes, mode, G). Regime-pinned tests: the acceptance test pins plain-2d at shapes where `choose_split` yields R > 1 with Q > 1, R = 1 (T = 1 decode), and Q = 1 (narrow C, many rows), plus `max_grid = (3, 2)`.

### Traffic ranking

Per-call DRAM crossings (count, per tensor) for each candidate split. The focus shape is T=160 (5 rows), C=5120 (160 c-tiles), G=110. Byte figures are for bf16 collapse (the perf focus); the plain fp32 Phase 0 figures are in `l1_ledger.md`.

| Candidate split | F / X / X' / y | coefficients (+P) | W (fused) | Cross-core | Verdict |
|-----------------|----------------|-------------------|-----------|-----------|---------|
| rows only (Q=1, R ≤ rows) | 1× each | 1× | R× (R = 5 → 13 MB) | none | Fills only 5 cores at T=160 (the rule's case "only 5 token-tile rows"). Kept as a point in the `(R, Q)` family: `choose_split` takes it for narrow C with many rows. |
| columns only (R=1, Q ≤ G) | 1× | Q× (80 × 20 KB = 1.6 MB) | **1×** (2.6 MB) | 1 partial/core/row → tree | Near-optimal for the fused modes at T=160: R=1, Q=80 gives a per-core max of 232 KB, vs 276 KB for R=5, Q=22. |
| 2D rows × columns (R, Q) | 1× | Q× | R× | per row group | **Chosen.** Plain focus fp32: R=5, Q=20. Fused focus bf16: R=2, Q=54 (per-core max 227 KB). W crosses DRAM twice (5.2 MB), coefficients 54× (1.1 MB). |
| split j (output streams) | X **n×** | — | — | none | rejected (dead end) |
| split i (input streams, dependent) | 1× | — | — | every output tile combined across cores (4 KiB × n per tile) | rejected: combine traffic = the whole X' |
| dependent `c` split + combine (fused) | — | — | — | partial tile per (core, row) | This IS the columns split in fused mode. The combine moves `Q·rows·4 KiB` (80 × 5 × 4 KiB = 1.6 MB on-chip) to save `(R−1)·|W|` DRAM bytes. Built in stats-2d-tree. |

Chosen: the 2D `(R, Q)` family, with extents picked per call by minimizing the critical-path core's DRAM bytes (`choose_split`). That objective counts the reuse-shared re-reads (coefficients × Q, W × R) and the per-core work at once. Independent axes are preferred when bytes are equal (the plain mode has no combine at any (R, Q)). Occupancy is not the objective: the minimum-bytes split already uses most of G, and the tie-break picks the fewest cores among equal-cost splits.

Operand-reuse check over the chosen split, run once per operand:

| Operand | Varies along `row`? | Varies along `c`? | Consequence |
|---------|--------------------|-------------------|-------------|
| F | yes | yes | read once |
| X | yes | yes | read once |
| post / comb / P | yes | **no** | reuse-shared by the column split → read Q×; `coef-mcast` deferred row |
| W | **no** | yes | reuse-shared by the row split → read R×; priced into `choose_split`; `w-mcast` deferred row |

Stall-shadow check:

| Waiting stage | Waits for | Work independent of the wait, scheduled into the window |
|---------------|-----------|---------------------------------------------------------|
| compute at a row boundary | the reader's coefficient expansion for row r+1 | The reader issues row r+1's coefficient-source read **and** its first F/X block reads before blocking on `cb_coef_cols` space, so DRAM latency hides under row r's last blocks. Only the 32 × n_coef_cols-word expansion remains exposed (lamp `coef_depth`). |
| (fused) root's combine of row r | partials from Q cores | The root's own mix of rows r+1 … r+L. The root combines row r only after its mix of row `r + combine_lag` (`COMBINE_LAG = 1`). This is legal: the mix of later rows does not depend on row r's stats. It changes no FP order, so no precision gate applies. |
| (fused) hc of row r | the combined stats row | y of row r does **not** wait: it is computed in the mix pass from the bf16 X' already in L1, before the partial is shipped. Nothing else depends on hc. |
| writer, sender side | leader credit for the gather slot | Writing X' / y tiles of the next blocks (the writer services them in program order). |
| last row's combine + hc | everything | Nothing remains; this is the tail. It is bounded by one combine (≤ 16 + ⌈Q/16⌉ SFPU adds) plus one hc program, and runs on a root that is distinct per row. |

### Block schedule

Logical schedule per core. Reader, compute and writer each realize their part asynchronously.

```cpp
for (uint32_t row_idx = 0; row_idx < num_rows_this_core; ++row_idx) {
    load_coefficients(row_idx);                 // reader
    for (uint32_t block_idx = 0; block_idx < num_c_blocks_this_core; ++block_idx) {
        load_block(row_idx, block_idx);         // reader
        mix_block(row_idx, block_idx);          // compute
        project_block(row_idx, block_idx);      // compute, stats/collapse only
        collapse_block(row_idx, block_idx);     // compute, collapse only
        store_block(row_idx, block_idx);        // writer
    }
    finish_row_partial(row_idx);                // compute + writer, stats/collapse only
    release_coefficients(row_idx);              // compute pops cb_coef_cols
    combine_row(rooted_row(row_idx - COMBINE_LAG)); // leader/root, stats/collapse only
    hc_row(rooted_row(row_idx - COMBINE_LAG));      // root, collapse only
}
drain_combines();                               // the last COMBINE_LAG rooted rows
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_coefficients` | `n_src` (1–3) fp32 source tiles → `n_coef_cols` column tiles | `cb_coef_cols` for the whole row | 1 DRAM read batch + 1 barrier per (core, row). Expansion = `32 × n_coef_cols` word moves on the reader RISC. 1 push of `n_coef_cols` pages. |
| `load_block` | F `block_c` + X `n·block_c` tiles | — | 1 reserve + 1 push per operand per block. Reads issued back-to-back, 1 barrier per block. |
| `mix_block` | `IterationShape::grid(n, block_c)`: one (j, c) output tile per iteration | F, X block (caller-managed wait before, pop after) and `cb_coef_cols` (whole row) | `compute_kernel_hw_startup` once per kernel. One chain call per block. The chain re-inits its elements per tile (bcast-unpack / copy / SFPU alternate). F/X wait + pop once per block. Output reserve/push per tile. |
| `project_block` (fused) | matmul `1 × (n·block_c) × 1` tiles | W slice (whole core) and the `cb_xprime_mm` block | 1 matmul_block call per block (`num_k_blocks = 1`) + 1 SFPU add into the row partial. |
| `collapse_block` (collapse) | `IterationShape::tiles(block_c)` | `cb_xprime_mm` block, P column tiles (in `cb_coef_cols`) | 1 chain call per block. `cb_xprime_mm` popped once after both consumers. |
| `store_block` | X' `n·block_c` tiles (+ y `block_c`) | — | Writes issued per tile as they are pushed, 1 barrier per block. |
| `finish_row_partial` (fused) | 1 ss tile → `sfpu_reduce` row → column 0; writer moves it into col 24 of the partial and ships 4 KiB | — | Once per (core, row): 1 SFPU reduce, 32 word moves, 1 NoC write + 1 semaphore increment. |
| `combine_row` (fused) | ≤ `COMBINE_FANIN` (leader) or ⌈Q/16⌉ (root) partial tiles | gather slots | Once per rooted row: 1 semaphore wait, one SFPU add per partial **in index order**, 1 pack. |
| `hc_row` (collapse) | 2 lane tiles (25 quantities × 32 tokens) | — | Once per rooted row: 2 permutes on the writer (stats → lanes, lanes → hc row), 1 custom SFPU program, 2 DRAM writes (stats, hc). |
| `release_coefficients` | `cb_coef_cols` | — | 1 pop of `n_coef_cols` per row. |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap** (`MIN_PIPELINE_BLOCKS = 4`) | Default deliberately takes a block **smaller** than the L1 cap whenever the core's work would otherwise be < 4 blocks (plain focus: 1 row × 8 c-tiles → `block_c = 2`), so read/compute/write overlap. Per-block fixed cost is paid 4× instead of once (catalog `compute_block_size`: ~1.6 µs/pass). | `MIN_PIPELINE_BLOCKS ∈ {1, 2, 4, 8}` at T=160/C=5120 and T=640/C=7168. |
| **Grid synchronization** (fused) | R=2, Q=54 at the focus shape gives 3 rows × 3 c-tiles per core and a 54-way (two-level) combine per row. Tree latency and the hc tail may dominate a ~50 µs call. | Q capped at 16 (single-level combine) + `w-mcast`, vs the default. |
| **`choose_split` objective** | It minimizes the busiest core's DRAM bytes, which favors occupancy. When the whole grid is DRAM-saturated, aggregate bytes set the time: at the fused focus shape, R=1/Q=80 moves 2.1 MB (9 %) less in aggregate than the chosen R=2/Q=54 (`l1_ledger.md`). | Objective `max(aggregate_bytes, K·busiest_core_bytes)` for `K ∈ {16, 32, G}`, vs the default. Keep the fused modes mode-invariant. |
| **SFPU-bound mix** | Per output tile: 5 bcast unpacks + 5 copies + 5 SFPU tile ops (each ~32 vector ops). With bf16 streams the DRAM time per tile halves; the mix is likely compute-bound. | A custom fused SFPU block op that loads each F/X_i vector once and emits all 4 outputs with full-sync DEST (8 fp32 slots), or a split-precision FPU form for **bf16** streams: coefficient = bf16 hi + mid + lo (exact), bf16 × bf16 products exact into fp32 DEST. It must pass the signed-bias gate. |
| **Chain re-init per tile** | Alternating `UnaryBcast` / `CopyTile` / `MulBinary` / `Addcmul` re-initialize the unpacker/SFPU each tile. | Loop the c axis inside each element (per (j, i), all `block_c` tiles), keeping one coefficient tile in a DEST slot across c. |
| **ss accumulation** (fused) | An exact SFPU running tile ping-pongs through `cb_ss_run` once per output tile (catalog `eltwise_l1_vs_dest_accumulate`: L1 RMW is 5× packer-L1-acc and 10× DEST-resident). | Packer L1 accumulation (`OutputSpec L1Accumulation::Enabled`, `chain.hpp:297`), after a bit-exactness probe against the SFPU sum (the SUMSQ gate is 1e-6). |
| **Combine add on SFPU** | Catalog `tensix_all_reduce_compute`: FPU `add_tiles` beats serial SFPU add 6.75× at 16 contributors. Here it is precision-forbidden (fp32 partials through tf32), so this is not an alternative. Recorded for the perf pass so it is not "discovered". | — (the alternative is fan-in: `COMBINE_FANIN ∈ {8, 16, 32}`) |
| **coef_depth** | Row-boundary stall of one expansion (~640 word moves) per row on multi-row cores. | `coef_depth = 2` (+ `n_coef_cols × 4 KiB`). |
| **io_depth** | Catalog `trid_double_issue`: depth 3–4 pays on a single core; a DRAM-saturated grid was not measured. | `io_depth = 3`. |
| **hc lane layout** | The robust layout uses 2 SFPU vectors per quantity (half wasted). | Dense layout, 1 vector per quantity, after a probe pins the SFPU datum ↔ lane mapping (`ckernel_sfpu_binary_bcast.h:30` says 4 rows × 8 cols per load). |

Catalog entries that informed the knobs: `width_split` (spread the column axis when rows under-fill: the rule's T=160 case); `double_buffer` (≥ 4–8 tiles per transfer batch, depth 2; a 2-c-tile block = 10 input tiles); `noc_placement` (enumerate cores row-wise; reader on NoC0, writer on NoC1); `read_coalescing` (does not apply: interleaved pages are not contiguous); `shared_input_reuse` / `mcast_topology` (the deferred mcast rows); `tensix_all_reduce` (tree reduction beats a flat root under grid contention on BH, and a flat root's gather buffer limits L1 → `COMBINE_FANIN`); `tensix_all_reduce_compute` (FPU add is faster but precision-forbidden here); `eltwise_l1_vs_dest_accumulate` (ss accumulation lamp); `sfpu_tile_scope` (~24 ns per SFPU vector op; motivates the lane layout for hc instead of broadcast tiles with 32× redundancy).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Coefficient source → L1 | fp32 tile(s), page `row · width_tiles + col // 32` per tensor | reader `TensorAccessor` reads of 1 (packed) / 2 (separate) / 3 (separate + P in collapse) pages into `cb_coef_src` | `post_col`, `comb_col` (and P col 0) are RT args. Packed: one page serves post, comb and P. |
| Coefficient expansion | fp32, `cb_coef_cols` tile q: column 0 = coefficient q for the 32 tokens | reader RISC word moves: for token row t: `src[t][col%32]` → `dst_q` words `t·16` (t<16) / `512+(t−16)·16` (t≥16). Order q: `post_0..3, comb_{i,j}` at `n + i·n + j`, `P_0..3` at `n + n² + i`. | Only column 0 of faces 0/2 is read by the Col broadcast. Other words are don't-care. |
| F / X → L1 | F dtype / X dtype tiles | reader: per block, `block_c` F pages `row·Ct + c0..`, then for i in 0..n−1, `block_c` X pages `row·n·Ct + i·Ct + c0..` | X block layout in CB: `[i][c]`. |
| Mix | DEST fp32 | compute: per (j, c): `D1 ← bcast_col(post_j)`, `D2 ← F_c`, `D0 = D1·D2` (SFPU), then for i: `D1 ← bcast_col(comb_ij)`, `D2 ← X_{i,c}`, `D0 = D1·D2 + D0` (SFPU addcmul, value 1.0), pack D0 → `cb_out_tiles` | fp32 operands enter DEST exactly: fp32 CBs use unpack-to-dest (`unpack_to_dest_mode = UnpackToDestFp32` for every fp32 CB that `CopyTile` reads; `UnaryBcast` selects it by format). bf16 operands are exact through srcA. |
| X' → DRAM | X dtype | writer: out tile (j, c) → page `row·n·Ct + j·Ct + c0 + c` | Writes on NoC1. |
| (fused) X' bf16 for compute | bf16 | compute packs D0 a second time into `cb_xprime_mm` (same rounding → same bits as written) | Single producer / consumer: compute pack → compute unpack. |
| (fused) projection | in0 bf16 X', in1 fp32 W → fp32 partial in DEST | `matmul_block` per block, `K = n·block_c`; `cb_w_slice` resident in block-major order `[block][j][c]` | Fidelity = caller's `math_fidelity`. The only FPU product. |
| (fused) ss | fp32 | SFPU `ss += acc²` per output tile (exact), `sfpu_reduce<SUM, Float32, REDUCE_ROW>` per row → column 0 | Uses the **unrounded** D0. |
| (fused) partial → leader | fp32 4 KiB | writer: insert ss (col 0 of `cb_ss_row`) into col 24 of `cb_partial` (32 word moves; col 24 of the projection is exactly 0 because W col 24 is 0), then NoC unicast write to `cb_gather` slot `(row_slot, q_local)` of the leader + `noc_semaphore_inc(leader.sem_gather)` | Push model, fixed slot per sender. Order-independent. |
| (fused) leader → root | fp32 4 KiB | the leader's combined tile is shipped the same way to root slot `leader_idx` | Single level when Q ≤ `COMBINE_FANIN` (root = leader of everyone). |
| (fused) gather → compute | fp32 | leader/root reader waits `sem_gather ≥ expected` (then resets), then `cb_push_back(cb_gather, fan-in)`; compute sums slots 0..fan-in−1 **in index order** with SFPU `AddBinary`; credit back to senders via `sem_credit` increments | Credit semaphores on senders bound slot reuse to `combine_depth` rows. |
| (collapse) stats → hc lanes | fp32 | root writer: `cb_stats_row` → DRAM stats page `row`; permute cols 0..24 into `cb_hc_lanes` (2 tiles) | Robust lane layout: quantity k → tile k/16, face (k%16)/4, face rows `4·(k%4)..+3`, all 16 cols. Token t → rel row t/16, col t%16 (rel rows 2–3 duplicate 0–1). Every quantity block is a whole number of SFPU loads, so the program is lane-consistent regardless of the intra-load datum mapping. |
| (collapse) hc program | fp32 DEST | custom SFPU block op (rsqrt, sigmoid, exp, reciprocal: exact variants; softmax with row-max; Sinkhorn column step, then `iters−1` row/column pairs), in place over the 2 lane tiles. The writer builds a constants lane block (b_k replicated per quantity) once per kernel from the RT args. | Lane-wise fp32 only; no FPU operand. |
| (collapse) hc lanes → DRAM | fp32 | root writer: inverse permute 24 quantities → `cb_hc_row` tile (row t, col k; cols 24..31 = 0) → DRAM hc page `row` | — |
| (collapse) y → DRAM | bf16 | writer: page `row·Ct + c0 + c` | — |

Physical sharding: TARGET has no `memory_layout` axis; every tensor is DRAM interleaved. The logical shards of the `(R, Q)` split are block shards of the (row, c) plane. A future `BLOCK_SHARDED` placement in which each core's shard equals its `(core_rows × core_c_tiles)` assignment would be a knob-turn for plain (a CB backed on the shard via `ttnn.cb_descriptor_from_sharded_tensor`, no NoC read). For stats/collapse the combine is unchanged.

Inter-Tensix contract for the deferred fused regimes (kept reachable by Phase 0): every core already knows `(g_r, g_c)`, its row group's Q cores (consecutive in row-wise enumeration), and the row-group's row range. Phase 0 passes these as RT args even in plain mode, so the fused regimes add semaphores and slots without changing the work assignment.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `1 row × block_c c-tiles × n streams` |
| Grid | `G = gx·gy` cores with `(gx, gy) = min(max_grid, device.compute_with_storage_grid_size())` per axis. Cores enumerated **row-wise**: index k → logical `(k % gx, k // gx)`. |
| Split | `(R, Q) = choose_split(tensor_row_tiles, tensor_c_tiles, mode, dtypes, n_src, G)`. Core `k < R·Q` gets row group `g_r = k // Q` and column group `g_c = k % Q`. Cores `≥ R·Q` are not in the program. |
| Per-core work | rows `[⌊g_r·Rt/R⌋, ⌊(g_r+1)·Rt/R⌋)`, c-tiles `[⌊g_c·Ct/Q⌋, ⌊(g_c+1)·Ct/Q⌋)` (balanced contiguous: counts differ by ≤ 1) |
| Remainder | `tensor_row_tiles Rt = Π(shape[:-2]) · ⌈T/32⌉` (per image, ceil). `tensor_c_tiles Ct = C/32` (exact by contract). Within a core, `num_c_blocks_this_core = ⌈core_c_tiles / block_c⌉`, `last_block_extent = core_c_tiles − (num_c_blocks_this_core − 1)·block_c`. The ragged block keeps nominal CB counts. |

`ttnn.split_work_to_cores` is not used: it is a 1-D splitter, and this split is 2-D with a row-group structure the fused modes' combine needs.

**`choose_split`** (host, pure; the single source of R, Q):

```
Rt_max(R) = ceil(Rt / R);  Ct_max(Q) = ceil(Ct / Q)
unit_bytes   = tile(F) + 2·n·tile(X)            [+ tile_bf16 (y) in collapse]
row_bytes    = n_src_tiles · 4096               (coefficient source tiles per row; + P when separate)
w_bytes_per_c = n · 4096                        (stats / collapse only, else 0)
cost(R, Q)   = Rt_max·(Ct_max·unit_bytes + row_bytes) + Ct_max·w_bytes_per_c
minimize cost over 1 ≤ R ≤ min(Rt, G), 1 ≤ Q ≤ min(Ct, G // R)
  subject to (fused) n·Ct_max·4096 ≤ W_RESIDENT_BUDGET
tie-break: smaller R·Q, then smaller Q
```

Mode-invariance requirement: `test_fused_modes_agree` needs stats-mode and collapse-mode stats rows to be **bit-identical**, and the summation order depends on `(R, Q, block_c)`. So `choose_split` and `choose_block_c` take `fused = mode ≠ plain` and must ignore every collapse-only term (y bytes, P bytes, collapse-only CBs). Both fused modes size against the **collapse** inventory. X' is elementwise and identical across all modes and all splits by construction (same chain, same per-element op order), which is what `test_max_grid` relies on.

**`choose_block_c`** (host, pure; the single source of `block_c`):

```
block_c_cap   = floor((L1_CB_BUDGET − fixed_bytes(mode)) / per_c_bytes(mode, dtypes))   # ledger expression
core_c_max    = Ct_max(Q); core_rows_max = Rt_max(R)
nb            = max(ceil(core_c_max / block_c_cap), min(core_c_max, ceil(MIN_PIPELINE_BLOCKS / core_rows_max)))
block_c       = ceil(core_c_max / nb)
```

Constants (one host module, single source): `L1_CB_BUDGET = 900 KiB`, `MIN_PIPELINE_BLOCKS = 4`, `W_RESIDENT_BUDGET = 384 KiB`, `COMBINE_FANIN = 16`, `COMBINE_DEPTH = 2`, `COMBINE_LAG = 1`, `IO_DEPTH = 2`, `COEF_DEPTH = 1`, `BLOCK_ROWS = 1`.

Worked values (Phase 0, plain fp32 separate, G = 110):

| Shape (T, C) | Rt, Ct | (R, Q) | cores | core_rows × core_c | block_c | blocks/core |
|--------------|--------|--------|-------|--------------------|---------|-------------|
| 160, 5120 | 5, 160 | (5, 20) | 100 | 1 × 8 | 2 | 4 |
| 640, 7168 | 20, 224 | (10, 11) | 110 | 2 × 21 | 11 | 4 |
| 1, 7168 (decode) | 1, 224 | (1, 75) | 75 | 1 × 3 | 1 | 3 |
| 4096, 1792 | 128, 56 | (26, 4) | 104 | 5 × 14 | 14 | 5 |
| 150, 5120 (ragged) | 5, 160 | (5, 20) | 100 | 1 × 8 | 2 | 4 |
| fused bf16, 160, 5120 (packed; stats cost used for both fused modes) | 5, 160 | (2, 54) | 108 | 3 × 3 | 1 | 3 |

These values were computed by running the function as specified, with `L1_CB_BUDGET` giving `block_c_cap = 14` for plain fp32. The implementer reproduces the function, not these numbers.

## Circular Buffers

Indices are any distinct values. Every CB is described by its semantic name.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_coef_src` | 0 | 4096 | `n_src` (1–3) | Reader-private landing tiles for the row's coefficient source pages. Spans nothing; constant. | Float32 | reader | reader (private scratch, no push/pop) | per row |
| `cb_coef_cols` | 1 | 4096 | `n_coef_cols · COEF_DEPTH` | Spans the coefficient axis (20 / 24) for one row; streams over row. | Float32 | reader | compute | whole row |
| `cb_f_tiles` | 2 | tile(F) | `block_c · IO_DEPTH` | Spans `c` (block_c), streams over row/blocks. | F dtype | reader | compute | per block |
| `cb_x_tiles` | 3 | tile(X) | `n · block_c · IO_DEPTH` | Spans `c` × i. | X dtype | reader | compute | per block |
| `cb_out_tiles` | 4 | tile(X) | `n · block_c` | Spans `c` × j (the block's output window), pushed per tile. | X dtype | compute | writer | per block |
| `cb_xprime_mm` (fused) | 5 | 2048 | `n · block_c` | Spans `c` × j. Consumed by the projection and y in the same compute thread. | Float16_b | compute | compute | per block |
| `cb_w_slice` (fused) | 6 | 4096 | `n · core_c_max` (≤ W_RESIDENT_BUDGET / 4096) | Spans the core's whole c range (resident), block-major `[block][j][c]`. | Float32 | reader | compute | whole kernel |
| `cb_proj_block` (fused) | 7 | 4096 | 1 | The matmul output for one block. | Float32 | compute | compute | per block |
| `cb_proj_run` (fused) | 8 | 4096 | 2 | Running row partial, ping-pong (pop 1 / push 1 per block). | Float32 | compute | compute | per row |
| `cb_ss_run` (fused) | 9 | 4096 | 2 | Running elementwise ss, ping-pong per output tile. | Float32 | compute | compute | per row |
| `cb_ss_row` (fused) | 10 | 4096 | 1 | Row-reduced ss (col 0) for the writer. | Float32 | compute | writer | per row |
| `cb_partial` (fused) | 11 | 4096 | 1 | The core's final row partial (projection) for the writer to patch and ship. | Float32 | compute | writer | per row |
| `cb_gather` (fused) | 12 | 4096 | `COMBINE_FANIN · COMBINE_DEPTH` (leader); root slots reuse the same CB (root fan-in ⌈Q/16⌉ ≤ 9 < 16) | Fixed slot per sender. Written by remote writers; pushed by the local reader after the semaphore. | Float32 | reader (pushes after semaphore) | compute | per rooted row |
| `cb_stats_row` (fused) | 13 | 4096 | 1 | Combined stats tile on the root. | Float32 | compute | writer | per rooted row |
| `cb_hc_lanes` (collapse) | 14 | 4096 | 2 (+1 constants lane tile, filled once) | 25 quantities × 32 tokens + constants. | Float32 | writer | compute | per rooted row |
| `cb_hc_out` (collapse) | 15 | 4096 | 2 | hc result lanes. | Float32 | compute | writer | per rooted row |
| `cb_hc_row` (collapse) | 16 | 4096 | 1 | Writer-private landing tile for the inverse permutation. | Float32 | writer | writer (private scratch) | per rooted row |
| `cb_y_tiles` (collapse) | 17 | 2048 | `block_c` | Spans `c`, pushed per tile. | Float16_b | compute | writer | per block |
| `cb_zero` (fused) | 18 | 4096 | 1 | A zero tile used to seed `cb_ss_run` / `cb_proj_run` at row start (via a `FillScalar` chain, `eltwise/generators/fill.hpp:20`). | Float32 | compute | compute | whole kernel |

Phase 0 builds only `cb_coef_src`, `cb_coef_cols`, `cb_f_tiles`, `cb_x_tiles`, `cb_out_tiles`. The ledger shows the fused inventory in full so the deferred regimes' L1 is already accounted for.

`ComputeConfigDescriptor.unpack_to_dest_mode` must be `UnpackToDestFp32` for every Float32 CB read by `CopyTile` (`cb_f_tiles` and `cb_x_tiles` when fp32, `cb_proj_*`, `cb_ss_run`, `cb_gather`, `cb_hc_*`) and `Default` for bf16 CBs and for `cb_w_slice` (matmul operand). Without it, `copy_tile` routes fp32 through srcA (tf32), which is the bias the gate exists to catch.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_coefficients` | `n_src` → `n_coef_cols` tiles | no (dataflow) | DRAM post/comb/(P) pages | `cb_coef_src` (scratch), `cb_coef_cols` (`n_coef_cols`) | `cb_coef_cols` pushed `n_coef_cols`, resident for the row |
| 2 | `load_block` | F `block_c`, X `n·block_c` | no (dataflow) | DRAM | `cb_f_tiles` (`block_c`), `cb_x_tiles` (`n·block_c`) | pushed nominal counts |
| 3 | `mix_block` | `grid(n, block_c)` | yes: `eltwise_chain` | `cb_coef_cols` (`n_coef_cols`, waited per row, not popped), `cb_f_tiles` (`block_c`, waited upfront by caller), `cb_x_tiles` (`n·block_c`, waited upfront by caller) | `cb_out_tiles` (`n·block_c`, per tile) [+ `cb_xprime_mm`, `cb_ss_run` in fused] | caller pops `cb_f_tiles` `block_c`, `cb_x_tiles` `n·block_c` after the chain |
| 4 | `store_block` | `n·block_c` (+ `block_c` y) | no (dataflow) | `cb_out_tiles`, `cb_y_tiles` | DRAM | popped per tile |
| 5 | `release_coefficients` | `n_coef_cols` | no | `cb_coef_cols` | — | popped `n_coef_cols` |
| 6 | `project_block` (fused) | `1 × n·block_c × 1` | yes: `matmul_block` | `cb_xprime_mm` (NoWaitNoPop; caller waits), `cb_w_slice` (resident, base offset `block_idx·n·block_c`) | `cb_proj_block` (1) → `AddBinary` with `cb_proj_run` → `cb_proj_run` | `cb_xprime_mm` popped by the caller after op 7 |
| 7 | `collapse_block` (collapse) | `tiles(block_c)` | yes: `eltwise_chain` | `cb_xprime_mm` (`[j][c]`, Offset base `i·block_c`), `cb_coef_cols` P tiles | `cb_y_tiles` (per tile) | — |
| 8 | `finish_row_partial` (fused) | 1 tile | yes: `sfpu_reduce` (raw API, see mapping) inside a thin block op | `cb_ss_run` (1), `cb_proj_run` (1) | `cb_ss_row` (1), `cb_partial` (1) | writer patches col 24, ships, pops |
| 9 | `combine_row` (fused) | fan-in tiles | yes: `eltwise_chain` (`CopyTile` + `AddBinary`, Scalar-pinned per slot, index order) | `cb_gather` (fan-in) | `cb_stats_row` (root) or `cb_partial` (leader) | gather popped fan-in |
| 10 | `hc_row` (collapse) | 2 lane tiles | **no**: custom SFPU block op | `cb_hc_lanes` (2 + constants) | `cb_hc_out` (2) | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all compute | helper prerequisite | `compute_kernel_hw_startup` | caller-owned per `ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp:26-30` / `matmul_block_helpers.hpp:100-106` | Once at kernel top. In the fused modes use the matmul form `compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out)` (`matmul_block_helpers.hpp:100-106`); the chain elements re-init themselves. | — | — | — |
| `mix_block` | helper | `eltwise_chain` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:619-620` | `IterationShape::grid(n, block_c)` (`chain.hpp:160`) | — | — | `block_c` (Wt of the shape) |
| `mix_block` | helper element | `UnaryBcast<BroadcastDim::Col, input(cb_coef_cols, None, None, Col, …, Offset), Dst::D1>` | `ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp:26-27` | base `q_post = 0` (post_j at Col index j), or `q_comb + i·n` (comb_ij at Col index j) | `cb_coef_cols` | DEST D1 | — |
| `mix_block` | helper element | `CopyTile<input(cb_f_tiles, None, None, Row), Dst::D2>`; `CopyTile<input(cb_x_tiles, None, None, Row, …, Offset), Dst::D2>{x, i·block_c}` | `chain.hpp:553` | Row mapping = indexed by c (`chain.hpp:247-249`) | `cb_f_tiles`, `cb_x_tiles` | D2 | `block_c` (Offset base) |
| `mix_block` | helper element | `MulBinary<Dst::D1, Dst::D2, Dst::D0>` | `ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp:27-28` | fp32 SFPU multiply | — | D0 | — |
| `mix_block` | helper element | `Addcmul<DataFormat::Float32, D0, D1, D2, D0>{0x3f800000}` | `ttnn/cpp/ttnn/kernel_lib/eltwise/ternary/ternary.hpp:26`; numerics `tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h:49-60` (value·in1 exact for 1.0, then one fused MAD) | 4 instances (i = 0..3) | — | D0 | — |
| `mix_block` | helper element | `PackTile<output(cb_out_tiles, PerTile, PerTile)>` | `chain.hpp:587` | — | D0 | `cb_out_tiles` | — |
| `mix_block` (fused ext.) | helper elements | `PackTile<cb_xprime_mm>`; `CopyTile<input(cb_ss_run, PerTile, PerTile, Scalar), D3>`; `Addcmul<Float32, D3, D0, D0, D3>{1.0}`; `PackTile<cb_ss_run, D3>` | as above | ss += acc² from the unrounded D0. If the chain rejects two `PackTile` terminals, realize as a thin custom block op with the same per-tile contract. | `cb_ss_run` | `cb_xprime_mm`, `cb_ss_run` | — |
| `project_block` | helper | `matmul_block` + `MatmulBlockShape::of(1, 1, 1, 1, n·block_c, 1)` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:339-363`, `:160` | in0 policy `NoWaitNoPop` (`:77`), `In1BaseOffsetFn` → `block_idx·n·block_c`, `num_k_blocks = 1` (pass `out` as `interm`, `:248`) | `cb_xprime_mm`, `cb_w_slice` | `cb_proj_block` | `in0_block_k = n·block_c` |
| `project_block` accumulate, `combine_row` | helper | `eltwise_chain` with `CopyTile` ×2 + `AddBinary<D0, D1, D0>` | `basic.hpp:15-20` | fp32 SFPU add, fixed order | `cb_proj_block`/`cb_proj_run`; `cb_gather` | `cb_proj_run`; `cb_stats_row`/`cb_partial` | fan-in |
| `collapse_block` | helper | `eltwise_chain(tiles(block_c), UnaryBcast<Col, P_0>→D1, CopyTile<xprime, Offset 0>→D2, MulBinary, [UnaryBcast<Col,P_i>, CopyTile<xprime, Offset i·block_c>, Addcmul]×3, PackTile<cb_y_tiles>)` | as above | — | `cb_xprime_mm`, `cb_coef_cols` | `cb_y_tiles` | `block_c` |
| `finish_row_partial` | raw_api | `sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>` + `sfpu_reduce_init` | `tt_metal/hw/inc/api/compute/compute_kernel_api.h:797`, `:847`; result in column 0 of each row (WH `ckernel_sfpu_reduce.h:730`, BH `:484`) | `ct_dim = rt_dim = 1` | `cb_ss_run` | `cb_ss_row` | — |
| `hc_row` | raw_api | custom SFPI kernel using the exact `exp`, `reciprocal`, `rsqrt`, `sigmoid` SFPU primitives over the lane blocks | (new) | RT: `a_pre, a_post, a_res, 1/width, norm_eps, hc_eps, iters` (fp32 bits / int) | `cb_hc_lanes` | `cb_hc_out` | — |
| `load_*`, `store_*`, permutes | dataflow | `TensorAccessor` reads/writes; local L1 word moves | `tech_reports/tensor_accessor/tensor_accessor.md` | — | — | — | — |
| combine transport | dataflow | `noc_async_write` + `noc_semaphore_inc` / `noc_semaphore_wait`; semaphores via `ttnn.SemaphoreDescriptor` (`ttnn-python-utility-bindings.md:212`); peer coords via `device.worker_core_from_logical_core` (`:224`) | — | — | `cb_partial` | remote `cb_gather` | `COMBINE_FANIN`, `COMBINE_DEPTH` |

Helpers considered and rejected (the raw_api rows):

| Raw entry | Helper considered | File:line of the mismatch | Reason |
|-----------|-------------------|---------------------------|--------|
| `sfpu_reduce` (exact row sum of ss) | `reduce()` (`reduce_helpers_compute.hpp`) | The FPU reduce feeds the tile through srcA/srcB (19-bit; `tt_metal/hw/inc/api/compute/bcast.h:46` "SrcB is only 19bits wide"). | Truncating fp32 squares to tf32 is ~1e-3 relative error, against a gate of `SUMSQ_REL = 1e-6` (`helpers.py`). No kernel_lib helper wraps the SFPU reduce. Gap: a `SfpuReduce` chain element would close it. |
| `hc_row` custom SFPU | `eltwise_chain` + `Exp`/`Recip`/`Rsqrt`/`Sigmoid` (`eltwise/unary/math.hpp:22,34,38`, `activations.hpp:24`) | Chain elements operate on whole DEST tiles (`chain.hpp:619`). Softmax and Sinkhorn sums run **across quantities**, which in any tile layout sit at different rows (row/column sums of M). | Whole-tile helpers cannot add quantity-block k to quantity-block k' without a data permutation between every Sinkhorn half-step (40 writer round-trips per row). Broadcast-tile layouts waste 32× SFPU work (`sfpu_tile_scope`: cost ∝ vector count). Gap: a lane-block SFPU helper ("apply f over quantity blocks of a tile") would close it. |
| `finish_row_partial` col-24 patch, coefficient expansion, lane permutes | `UnaryBcast` / FPU transpose (`transpose_wh_tile`) | FPU transpose / bcast go through srcA/B (tf32). `transpose_dest<true>` (`tt_metal/hw/inc/api/compute/transpose_dest.h:51-52`) is exact, but it produces token-major rows, not the quantity-block lanes the hc program needs. | Pure data movement of ≤ 1024 words per row is cheapest and exact on a dataflow RISC. |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| mix: post_j·F | SFPU `MulBinary` on DEST | `cb_coef_cols[post_j]`: Col0 (rows 0..31 of column 0) → broadcast to All by `UnaryBcast<Col>` | `cb_f_tiles`: All | Col (on the coefficient, via unpack-to-dest) |
| mix: + comb_ij·X_i | SFPU `Addcmul` | `cb_coef_cols[n + i·n + j]`: Col0 → All | `cb_x_tiles[i·block_c + c]`: All | Col |
| y: P_i·X'_i | SFPU `MulBinary` / `Addcmul` | `cb_coef_cols[n + n² + i]`: Col0 → All | `cb_xprime_mm[i·block_c + c]`: All | Col |
| combine | SFPU `AddBinary` | `cb_gather[slot]`: All | running: All | None |

The coefficient column tiles' valid region is column 0 only, because the reader writes nothing else. Every consumer of them must be a `Col` broadcast.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| tf32 truncation bias | Any FPU touch of fp32 F/X/coefficients (including `copy_tile` without unpack-to-dest) gives a −7e-4 bias and fails the signed-bias and depth-chain gates. | SFPU-only mix. `unpack_to_dest_mode = UnpackToDestFp32` on every fp32 CB read by `CopyTile`. `UnaryBcast` auto-selects unpack-to-dest for Float32. The only FPU product is the projection. |
| comb orientation | `X'_j` uses `comb[i][j]` (transposed); `test_cyclic_comb` fails outright if wrong. | Coefficient tile index `n + i·n + j` with `comb[i][j]` at source column `comb_col + i·n + j`. The chain's Col index is j, the Offset base is `n + i·n`. |
| Coefficient column offsets | Engram add reads post at column 12 of a 32-wide row. | Reader RT args `post_col`, `comb_col`; tile column `col // 32`, in-tile column `col % 32`. Validated not to cross a tile column. |
| Ragged T | Padded token rows hold garbage. | Every stage is token-row-local (mix, matmul row, ss row, hc lane). Rows are never mixed, so padding cannot contaminate real rows. Tile counts use `⌈T/32⌉` per image. |
| Stats bits must match across stats/collapse modes | The summation order depends on `(R, Q, block_c)`. | `choose_split` / `choose_block_c` are mode-invariant across the fused modes (they size against the collapse inventory and ignore collapse-only bytes). |
| Determinism of the combine | Arrival order varies between calls. | Fixed slot per sender, a semaphore count (not order), and summation in slot-index order at both tree levels. |
| bf16 pack rounding (bf16 refinement) | If the packer truncates fp32 → bf16, X' and y are biased (bias floor 1e-5). | Before claiming bf16: probe the packer's fp32→bf16 rounding on device. If it is not round-to-nearest-even, round in SFPU before the pack. Same for `cb_xprime_mm` (must equal the written X' bits). |
| Caller output buffers / program cache | The result must land in the caller's buffer, and runtime hc scalars must not be frozen. | Buffer addresses and every hc scalar / column offset are **runtime args**. Shapes, dtypes, mode, block_c, R, Q are compile-time / descriptor structure. |
| max_grid | Must confine every core and keep X' bits identical. | G derives from max_grid (single source). X' is per-element identical for any split. |
| hc lane datum mapping | An SFPU load covers 4 dest rows × 8 cols (`ckernel_sfpu_binary_bcast.h:30`), and the even/odd column interleave is documented inconsistently. | The robust lane layout gives each quantity whole 4-row × 16-col blocks (two loads), so lane-wise ops are mapping-independent. A probe test writes k into each block and checks an SFPU identity program round-trips. |
| Sinkhorn order of operations | The gate is 1e-5 vs float64 DeepSeek order. | Softmax with row max, `+hc_eps`; **column** normalization first; then `iters−1` × (row, column). Divisions as exact reciprocal × multiply (2 roundings, ≈1e-7). `approx = Exact` on every transcendental, regardless of the caller's `math_approx_mode`. |
| Sum of squares source | Must use the **unrounded** fp32 mix, not bf16 X'. | ss is computed from D0 before / independent of the bf16 pack. |
| FPU dot-product accumulation (found on device, Refinement 1) | The matmul rounds every 16-term face partial sum to ~tf32 at every fidelity. Exact operands therefore do not give an exact projection: plain `matmul_block` measures rel 3e-4 at HiFi4 vs the 1e-5 gate. | HiFi3/HiFi4: `PROJ_EXACT_PASSES` = 16 column-masked passes, one product per face, accumulated exactly on the SFPU (`cb_proj_masks`, `cb_xprime_masked`). HiFi2/LoFi: a single matmul. |
| One dispatch | The fused modes might tempt a second program. | Every stage is in the one program. Host work is validation, split selection, output allocation and argument packing only. |

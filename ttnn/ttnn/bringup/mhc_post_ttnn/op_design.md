# Operation Design: mhc_post

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (per-token weighted stream mix — eltwise with per-row coefficient broadcast) |
| Goal | Fold the sublayer output F back into the n residual streams and mix the streams with comb^T, in ONE device program (one `ttnn.generic_op` dispatch), DRAM-bound at every (T, C) of the perf sweep. |
| Math | For every token t and output stream j: `X'[t, j*C:(j+1)*C] = post[t, j] * F[t, :] + Σ_i comb[t, i*n + j] * X[t, i*C:(i+1)*C]` (comb applied TRANSPOSED) |
| Mode | Derivative (new fused op; replaces the 81-op composite `hc_post`) |
| References | `models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py` (`MHCWrap.hc_post`, torch ground truth); `models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py` (composite being replaced — same math, same packing); `eval/golden_tests/mhc_post/{feature_spec.py,helpers.py}`; `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp`; `ttnn/cpp/ttnn/kernel_lib/eltwise/ternary/ternary.hpp`; `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp`; `ttnn/ttnn/operations/examples/master.md` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (F) | `ttnn.Tensor` | yes | `(..., T, C)`, rank 2–4, TILE, DRAM interleaved, `C % 32 == 0` | — | buffer address RT |
| `residual` (X) | `ttnn.Tensor` | yes | `(..., T, n*C)`, leading dims == F's | — | buffer address RT |
| `post` | `ttnn.Tensor` | yes | `(..., T, n)`, float32 TILE, `1 <= n <= 5` | — | buffer address RT |
| `comb` | `ttnn.Tensor` | yes | `(..., T, n*n)`, float32 TILE | — | buffer address RT |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` (kw-only) | no | `fp32_dest_acc_en` must be True; `math_fidelity`, `math_approx_mode` honoured | `None` → `default_compute_kernel_config()` = HiFi4, fp32 DEST, approx off | compute config |

`n` = `post.shape[-1]` (CT arg to all three kernels). There are no other user parameters.

Public entry points (exact): `mhc_post(input_tensor, residual, post, comb, *, compute_kernel_config=None) -> ttnn.Tensor` and `default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor`, importable as `from ttnn.bringup.mhc_post_ttnn import mhc_post, default_compute_kernel_config`. The package must also export `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS` (the golden harness imports them).

### Registry contract (names exactly as in `feature_spec.py` TARGET)

| Axis | Read from | TARGET | Phase 0 SUPPORTED |
|------|-----------|--------|-------------------|
| `dtype` | `residual.dtype` (X and X') | float32, bfloat16 | float32 |
| `sublayer_dtype` | `input_tensor.dtype` (F) | float32, bfloat16 | float32 |
| `layout` | `residual.layout` | TILE | TILE |
| `fp32_dest_acc_en` | resolved `compute_kernel_config.fp32_dest_acc_en` | True | True |
| `alignment` | tagger, see below | tile_aligned, h_non_aligned | tile_aligned, h_non_aligned |

- `INPUT_TAGGERS = {"alignment": tag_alignment}`; `tag_alignment(inputs, axes)` receives `(F_shape, X_shape, post_shape, comb_shape)` and returns `"tile_aligned"` iff `F_shape[-2] % 32 == 0`, else `"h_non_aligned"`.
- `EXCLUSIONS = []` in Phase 0.
- `validate()` is the entry point's first line: build the axes dict (config resolved through `default_compute_kernel_config()` when None), apply the tagger, raise `UnsupportedAxisValue` per axis outside SUPPORTED, with the axis name in the message (this refuses `fp32_dest_acc_en=False`; the acceptance test matches on "fp32_dest_acc_en"), then `ExcludedCell` per EXCLUSIONS. It does NOT check INVALID.
- Shape-contract checks (after the registry gate) raise `ValueError`: rank 2–4 for all four; leading dims (`shape[:-1]`) identical across F, X, post, comb; `C % 32 == 0`; `X.shape[-1] == n*C`; `comb.shape[-1] == n*n`; `1 <= n <= 5` (mechanism cap, below); post / comb float32 TILE; all four DRAM interleaved, TILE.
- Structural impossibilities (candidate INVALID entries): none.

## Tensors

### Input

| Property | F (`input_tensor`) | X (`residual`) | post | comb |
|----------|--------------------|----------------|------|------|
| Shape | `(..., T, C)` | `(..., T, n*C)` | `(..., T, n)` | `(..., T, n*n)` |
| Dtype | float32 (Phase 0); bfloat16 (refinement) | float32 (Phase 0); bfloat16 (refinement) | float32 (fixed) | float32 (fixed) |
| Layout | TILE | TILE | TILE | TILE |
| Memory | DRAM interleaved | DRAM interleaved | DRAM interleaved | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `residual.shape` (`(..., T, n*C)`), same packing: stream j in columns `[j*C, (j+1)*C)` |
| Dtype | `residual.dtype` |
| Layout | TILE |
| Memory | DRAM interleaved (allocated host-side with `ttnn.allocate_tensor_on_device`; not an op dispatch) |

## Blocking Model

Every tensor is tiled; all index arithmetic is in tiles. Shared symbols (host-derived once, passed as CT/RT args):

| Symbol | Definition | Scope |
|--------|------------|-------|
| `tensor_token_tiles` | `prod(F.shape[:-2]) * ceil(F.shape[-2] / 32)` — per-image tile padding (rank 2: empty product = 1) | tensor |
| `tensor_col_tiles` | `C / 32` (Ct) — tiles per stream per token-tile row | tensor |
| `total_units` | `tensor_token_tiles * tensor_col_tiles` — one unit = (token-tile row `r`, column tile `c`): 1 F tile + n X tiles in, n X' tiles out | tensor |
| `core_units` | this core's contiguous share of the flattened (r-major, c-minor) unit index | core |
| segment | the maximal run of this core's units inside one token-tile row: `(r, c0, seg_col_tiles)` | core |

Tile page indices (all four inputs share the same token-tile row index `r` because their leading dims match):
`F: r*Ct + c` · `X_i: r*(n*Ct) + i*Ct + c` · `X'_j: r*(n*Ct) + j*Ct + c` · `post: r*ceil(n/32)` · `comb: r*ceil(n*n/32)`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| token-tile row `r` (all leading dims folded in, per-image padded) | **independent** — every output row depends only on the same token row of F, X, post, comb (no cross-token term) | `block_token_tiles` | 1 — a block never spans two token rows: the per-token coefficient set (n + n² expanded tiles, 80 KB fp32 at n=4) is the working set that grows with this extent, and it buys no fixed-cost amortization that `block_col_tiles` does not already buy at zero coefficient cost | host constant `BLOCK_TOKEN_TILES = 1`; realized by cutting core ranges at row boundaries (segments) | spread: flattened (r, c) units split contiguously over the full grid, so a core owns 1–few token rows | knob-turn (only together with a 2-D rectangle core assignment; see regimes) |
| hidden column tile `c` | **independent** for F / X / X' (elementwise across columns); **reuse-shared** for post / comb (the coefficients do not vary along C, so every core on the same token row needs the same coefficient set) | `block_col_tiles` (B) | `min(block_col_tiles_fit, max_segment_col_tiles)` — the coarsest that fits L1 (closed form in `l1_ledger.md`: 12 at fp32/fp32, 25 at bf16/bf16, n=4, since Refinement 2's half-packed coefficient set) and never more than the longest segment any core owns | host: computed once from (n, F dtype, X dtype, depths, `L1_BUDGET_BYTES`) → CT arg to reader, compute, writer and every CB size | spread: part of the same flattened split (a core's range crosses at most `ceil(core_units / Ct) + 1` rows) | knob-turn |
| output stream `j` | **independent** — X'_j depends on all inputs of its column but not on X'_{j'} | `n` (whole) | n | `post.shape[-1]` → CT arg | NOT assigned across cores: every X'_j of a unit needs the same F tile and all n X tiles; splitting j would re-read F and X once per core-part (n× the X traffic) — rejected in regimes | none (dead end) |
| input stream `i` (contraction) | **dependent** — X'_j sums over i (`Σ_i comb_ij X_i`) | `n` (whole) | n | `post.shape[-1]` → CT arg | NOT assigned across cores: the whole contraction is n tiles of one unit, resident in the block; a cross-core split adds partial-sum traffic and saves no DRAM crossing — rejected in regimes | scheme-change (rejected) |
| coefficient index `k` (stage data: expanded coefficient tiles, `k ∈ [0, n + n²)`) | **independent** — each expanded tile depends only on one raw column | `n + n²` (whole set per token row) | n + n² | derived from `n` | per core: each core expands the set for the rows it owns — the expansion is a per-core local step, not a designated-core stage, so no core idles on another's expansion | knob-turn (move expansion to the other DM RISC — perf lamp) |

Knob summary in the planner's one-row-per-axis form (no blanks):

| Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|---|---|---|---|---|---|
| token-tile row `r` | independent | 1 | flattened contiguous split over full grid | coef set: `coef_depth = 2` rows | the extent that costs coefficient L1; parallelism comes from the flattened split |
| column tile `c` | independent / reuse-shared (coefs) | `block_col_tiles` = coarsest fit | flattened contiguous split over full grid | `depth_in = 2`, `depth_out = 2` blocks | carries the per-block fixed-cost amortization; filling the grid comes first, then coarsest block |
| output stream `j` | independent | n (whole) | not split — decision | inside the block | splitting re-reads F and X per core (n× X DRAM traffic) |
| input stream `i` | dependent | n (whole) | not split — decision | inside the block | contraction of n resident tiles; a combine buys nothing |
| coefficient `k` (stage data) | independent | n + n² (whole row set) | per core, own rows only | `coef_depth = 2` | built where it is consumed; no designated core |

Every knob is a parameter: `block_col_tiles`, `n`, `depth_in`, `depth_out`, `coef_depth`, the grid (`device.compute_with_storage_grid_size()`), and `L1_BUDGET_BYTES` are defined once on the host; CB sizes, loop trip counts and kernel args derive from them. No kernel loop bound is a literal, and no loop strides over an element count.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_sublayer_tiles` | `depth_in` | 2 blocks | reader fetches block k+1 while compute mixes block k |
| `cb_residual_tiles` | `depth_in` (same knob) | 2 blocks | same; F and X of one block are pushed together, so they share one depth |
| `cb_output_tiles` | `depth_out` | 2 blocks | compute packs block k+1 while writer drains block k |
| `cb_coef_bcast` | `coef_depth` | 2 token-row sets | reader expands row r+1's coefficients while compute still consumes row r's (row boundaries fall mid-range on most cores) |
| `cb_coef_raw` | none (1 set) | 1 set (post + comb raw tiles) | reader-private scratch; consumed by the expansion immediately after it lands |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| comb row in one raw tile (expansion reads raw column `k = i*n + j` of ONE tile) | `n*n <= 32` → `n <= 5` | `validate` raises `ValueError` for n > 5 | the expansion reads columns past the tile → wrong coefficients |
| DEST slots live at once per output window (Refinement 2): P = ceil((n+1)/2) coefficient tiles + K·(n+1) data tiles | `P + K·(n+1) <= DEST_AUTO_LIMIT` = 8 (fp32 DEST, SyncFull — host `DST_FULL_SYNC`); K = 1 at n = 3, 4; K > 1 at n <= 2; n = 5 does not fit and runs the grouped path (terms loaded in groups, accumulated in the first data slot) | `static_assert`s in the compute kernel derive K / the regime from `DEST_AUTO_LIMIT`; the chain asserts every slot `< DEST_AUTO_LIMIT` | slot overflow corrupts DEST |
| fp32 CB → DEST through srcA truncates to tf32 (19-bit) | every fp32 CB read by `copy_tile` (`cb_sublayer_tiles` if F fp32, `cb_residual_tiles` if X fp32, `cb_coef_bcast` always) | host sets `unpack_to_dest_mode[cb] = UnpackToDestFp32` for exactly those CBs, derived from the CB's data format (bf16 CBs stay `Default`; bf16 → fp32 through srcA is exact) | every product biased toward zero by ~2⁻¹¹; fails the signed-bias gate (−7e-4 per call) and the 122-wrap regression |
| `C % 32 == 0` (stream boundaries on tile boundaries) | column tiles per stream | `validate` raises `ValueError` | a tile straddles two streams; stream offsets `i*Ct` are wrong |
| L1 residency | `block_col_tiles <= block_col_tiles_fit` | closed form in `l1_ledger.md`; host asserts `block_col_tiles_fit >= 1` | CB allocation fails / overlaps the kernel region |
| CB ring wrap (pointers reset only on an exact `fifo_limit` hit) | push/pop quantum of the three streaming CBs | every push/pop is the nominal block (`B`, `n*B`, `n*B`); a ragged last block narrows only the NoC transfers and the compute walk | ring drifts; producer/consumer pages misalign → wrong data or hang |
| CopyTile / PackTile with `TileAddressing::Offset` | lifecycle of every CB the compute chain reads or writes | caller-managed `(None, None)` wait/pop and reserve/push (`chain.inl:77-88`); the compute kernel waits/pops/reserves/pushes once per block | static_assert failure at compile time |

### Regimes

Minimum (named boundary: DRAM): each F and X element crosses DRAM once, each X' element once; post/comb are < 0.7% of bytes.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `flat_stream` — flattened (r, c) units split contiguously over the full grid; per core, segments per token row, blocks of `block_col_tiles` columns | **built** | all valid inputs (every INPUTS shape, T=1 decode through T=4096, C=32 through 7168) | `1 × block_col_tiles` columns × (1 F + n X) in, n X' out | F, X read once; X' written once (minimum). post/comb raw tiles (2 × 4 KB) read once per (core, token row) pair: ≤ `tensor_token_tiles + num_cores − 1` pairs (≈1.06 MB at T=640 vs 165 MB total, fp32) — above minimum only by that per-core coefficient re-read | fixed per-block cost: one reserve/push per streaming CB (3 on reader, 2 on compute, 1 on writer), one reader NoC read barrier, one writer NoC write barrier, and pipeline fill/drain per block. Coefficient load + expansion is per segment, not per block. Intended frequencies in the Block schedule. |
| `height_split` — token rows only | **rejected** — superseded by `flat_stream` | — | `rows × Ct` | identical bytes to `flat_stream` | at T=640 only 20 rows → 20 of 110 cores busy; `flat_stream` moves the same bytes on the full grid |
| `width_split` — column ranges only, every core walks all token rows | **rejected** — superseded by `flat_stream` | — | `1 × cols` per row | coefficient raw re-read per (core, row) = cores × rows (110 × 20 × 8 KB ≈ 17.6 MB at T=640, ~10% of traffic) and every core expands every row's coefficients | nothing it buys that `flat_stream` lacks |
| `stream_split` — output streams j across cores | **rejected** — dead end | — | `unit × 1 stream` | F read n×, every X_i read n× from DRAM (each core-part needs all inputs of its column) | no good scheme passes through it |
| `contraction_split_combine` — input streams i across cores + cross-core partial-sum combine | **rejected** — superseded by `flat_stream` | — | `unit × 1 input stream` | DRAM unchanged (minimum already), adds n partial X' tiles per contributor per unit over the NoC + semaphore rendezvous | no DRAM crossing to save: the contraction is 4 resident tiles; the dependent-axis split is neither a parallelism need (independent axes over-fill the grid at every sweep shape) nor a residency need (the reduced extent is already resident) |
| `coef_mcast` — one core per token row loads + expands the coefficients and multicasts the n + n² expanded tiles to the row's other cores | **rejected** — superseded by `flat_stream`'s per-core raw read | — | coefficient set | saves ≤ `(cores_per_row − 1) × 8 KB` of DRAM per row (< 1% of traffic); adds an 80 KB/row (fp32) NoC payload per receiver and a semaphore handshake — moves ~10× more bytes than it saves | — |
| `sharded_inputs` — F / X / X' resident in L1 shards, CBs backed on the shards | **deferred** — the placement axis is not in TARGET (contract: DRAM interleaved in and out) | memory_config sharded | the shard (one block per shard) | zero-copy input, no DRAM crossing on read | reachable as a knob-turn: every data axis is independent, so any height/width/block shard is self-contained; CBs become `cb_descriptor_from_sharded_tensor`, the reader keeps only the coefficient path |

Single built regime: no selection predicate, no regime-pinned tests required.

### Traffic ranking

Per candidate split, DRAM crossings per tensor and cross-core traffic (no ns):

| Rank | Split | F | X | X' | post/comb | Cross-core | Occupancy at T=640 |
|------|-------|---|---|----|-----------|------------|--------------------|
| 1 | `flat_stream` (flattened r-major units, contiguous per core) | 1 | 1 | 1 | ≤ (rows + cores − 1) × 8 KB | none | full grid |
| 1 (tie on bytes) | `height_split` | 1 | 1 | 1 | rows × 8 KB | none | 20 cores |
| 3 | `width_split` | 1 | 1 | 1 | cores × rows × 8 KB | none | full grid |
| 4 | `contraction_split_combine` | 1 | 1 | 1 | as flat | n partial tiles per contributor per unit | full grid |
| 5 | `stream_split` | n | n | 1 | as flat | none | full grid |

Chosen: `flat_stream` — cheapest traffic (tied with `height_split` on DRAM bytes up to < 1%), and among the cheapest it is the one that fills the grid. Operand-reuse check against the chosen split: F, X vary along both split axes (no reuse); post/comb do not vary along the column split → reuse-shared by construction; the broadcast alternative is the `coef_mcast` row (rejected on bytes). No independent axis is left unassigned: r and c are both spread by the flattening.

Stall-shadow check: the reader blocks on the NoC read barrier of each block. The coefficient expansion of a new token row depends only on the (small, already-landed) raw coefficient tiles, not on the block data in flight, so it is scheduled INSIDE that window: at a segment start the reader (1) reads the raw post/comb tiles and barriers (8 KB), (2) issues the segment's first block reads without a barrier, (3) expands the coefficients into `cb_coef_bcast` and pushes it, (4) barriers and pushes the data block. No floating-point order changes (the expansion is a copy). Compute waits on data; nothing it could do is independent of that data. The writer waits on compute; nothing independent.

### Block schedule

```cpp
for (uint32_t seg_idx = 0; seg_idx < num_segments_this_core; ++seg_idx) {        // one token-tile row each
    load_coefficients(seg_idx);                                                  // raw post/comb -> n + n^2 expanded tiles
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_segment; ++block_idx) {
        load_block(block_idx);                                                   // F[B], X[n][B]
        mix_block(block_idx);                                                    // X'[n][B]
        store_block(block_idx);                                                  // X'[n][B] -> DRAM
    }
    release_coefficients(seg_idx);
}
```

Segments and blocks are derived identically in all three kernels from the same RT args (`start_unit`, `num_units`) and CT args (`Ct`, `block_col_tiles`, `n`) — one shared derivation (a header-level inline function) so the three kernels cannot disagree. `num_blocks_this_segment = ceil(seg_col_tiles / block_col_tiles)`; the last block's `block_valid_col_tiles` may be smaller (ragged runtime extent, same operation).

| Operation | Kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|--------|-------------|--------------------|-----------------------------------|
| `load_coefficients` | reader | 1 token row: raw `ceil(n/32) + ceil(n²/32)` tiles → `n + n²` expanded fp32 tiles | previous row's expanded set (depth 2) | once per segment: one raw-tile read + barrier, one reserve/push of `n + n²` pages on `cb_coef_bcast`; the expansion is scheduled in the first block's read-barrier shadow (above) |
| `load_block` | reader | `block_col_tiles` F tiles + `n × block_col_tiles` X tiles (valid: `block_valid_col_tiles` columns) | nothing | once per block: one reserve per CB, `(n+1)·block_valid_col_tiles` async page reads, ONE `noc_async_read_barrier`, one push per CB (nominal counts) |
| `mix_block` | compute | for each output stream j (n chain calls), `block_valid_col_tiles` output tiles; per output tile: `X'_j = post_j·F + Σ_i comb_ij·X_i` in DEST, fp32 SFPU | F and X block (waited once, popped after all j); expanded coefficient set (waited once per segment, popped at segment end) | per block: one wait on `cb_sublayer_tiles` (B) and `cb_residual_tiles` (n·B), one reserve of `cb_output_tiles` (n·B), n chain calls, one pop/push each. `compute_kernel_hw_startup` once per kernel. Per-element inits (copy / SFPU mul / SFPU addcmul) are re-emitted per tile by the chain because the elements alternate engines inside one DEST window — intended and accepted (perf lamp). |
| `store_block` | writer | `n × block_col_tiles` X' tiles (valid: `n × block_valid_col_tiles`) | nothing | once per block: one wait, `n·block_valid_col_tiles` async page writes, ONE `noc_async_write_barrier`, one pop (nominal count) |
| `release_coefficients` | compute | the segment's coefficient set | — | once per segment: pop `n + n²` pages |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap** — `block_col_tiles` at its L1 fit (11 fp32 / 23 bf16) | at T=640, C=7168 a core owns ~41 units → ~4 blocks; the first block's fill (11 × 5 × 4 KB = 220 KB) and the last block's drain are not overlapped, a large fraction of a 4-block pipeline | `block_col_tiles ∈ {2, 4}` with `depth_in = depth_out ∈ {2, 3}`; the catalog's `double_buffer` entry saturates at ~4–8 reads per barrier, and one column already carries n+1 = 5 reads |
| **Compute-bound SFPU mix** (bf16 streams, the perf focus) | per output tile: 2(n+1) `copy_tile` unpacks + 1 SFPU mul + n SFPU `addcmul`; at bf16 the DRAM time halves while SFPU time does not (`compute_fusion` catalog entry: SFPU mul is ~0.58× the FPU) | (a) keep the n+1 coefficient tiles of one output stream resident in DEST across the column walk (SyncFull fp32 DEST = 8 slots) to drop n+1 coefficient copies per tile; (b) for bf16 X only: FPU multiply with the fp32 coefficient split into exactly representable bf16 parts (hi + mid + lo), accumulating in fp32 DEST — must pass the signed-bias gate |
| **Per-tile element re-init** in the chain | copy_init / mul_binary_tile_init / addcmul_tile_init re-emitted per tile | uniform-element realization (all n+1 terms as `Addcmul` onto an accumulator seeded by the first product) or a thin raw-LLK block op with inits hoisted where the engine state allows |
| **Reader issue / expansion on NCRISC** | the reader issues (n+1) page reads per column AND performs ~(n+n²)·1024 L1 stores per segment | move the expansion to BRISC (writer; then the writer is `cb_coef_bcast`'s producer), or halve the stores: expand faces 0/2 and duplicate into faces 1/3 by a self-aimed local NoC copy (`local_copy_helpers_dataflow.hpp:95`) (`split_reader` catalog entry) |
| **NoC placement** | `split_work_to_cores` default order gives column-localized DRAM traffic | `row_wise=True` (catalog `noc_placement`) — Phase 0 uses `row_wise=True`; measure the default against it |
| **Raw coefficient read size** | a raw coefficient tile is 4 KB but only columns `< n` / `< n²` (faces 0 and 2) are used | read faces 0 and 2 only (2 × 1 KB) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → `cb_coef_raw` | fp32 tiles (post, comb) | reader (NCRISC, NoC0) `TensorAccessor` page reads, one barrier | once per segment |
| `cb_coef_raw` → `cb_coef_bcast` | fp32 tiles, column-broadcast | reader L1 stores (`fill_l1_range<4>`, `l1_helpers.hpp:90`) | expanded tile k: every element (row ρ, col γ) = raw(ρ, k). Face layout: element (ρ, γ) lives in face `(ρ ≥ 16)·2 + (γ ≥ 16)` at offset `(ρ mod 16)·16 + (γ mod 16)` (fp32: 4 B each, 1 KB per face). Order: k = j for post_j (j < n), k = n + i·n + j for comb_ij, read from raw comb column `i·n + j` (TRANSPOSED application: output j pulls input i with comb[i][j]). |
| DRAM → `cb_sublayer_tiles`, `cb_residual_tiles` | F dtype / X dtype tiles | reader `TensorAccessor` page reads, one barrier per block | X_i tile (i, c) lands at CB slot `i·B + c` of the block; F tile c at slot `c` |
| CBs → DEST | fp32 in DEST | `copy_tile` (unpack-to-dest fp32 for fp32 CBs; srcA for bf16 CBs, exact) | never through the FPU math path |
| DEST mix | fp32 | SFPU `mul_binary_tile` (first term) and `addcmul_tile` with value = 1.0f (`0x3F800000`; the scalar multiply by 1.0 is exact, then one fused MAD) | unbiased fp32 arithmetic; fixed evaluation order → bitwise-deterministic |
| DEST → `cb_output_tiles` | X dtype | `pack_tile` | X'_j tile c at slot `j·B + c` of the block |
| `cb_output_tiles` → DRAM | X dtype tiles | writer (BRISC, NoC1) `TensorAccessor` page writes, one barrier per block | page `r·n·Ct + j·Ct + c` |

- No inter-Tensix communication in any built regime. The rejected `coef_mcast` would use `mcast_pipe` (`SenderPipe`/`ReceiverPipe`) along each token row's cores; not built.
- Placement axis: TARGET has no `memory_layout` axis — inputs and output are DRAM interleaved by contract. Were placement added, every shard flavor would be a **knob-turn** (all data axes are independent; no combine exists to build): the data CBs are backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy), never re-read over the NoC; the coefficient path is unchanged.
- Non-tile-aligned T (`h_non_aligned`): the op is strictly row-local (no cross-row operation anywhere — the column broadcast is along columns within one row, the SFPU is elementwise), so padded token rows can never contaminate real rows. No masking is needed; padded rows of X' carry whatever the padded inputs produce and are dropped by the tensor's logical shape.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one (token-tile row r, column tile c): 1 F tile + n X tiles in, n X' tiles out; the block is `block_col_tiles` consecutive units within one token row |
| Grid | `device.compute_with_storage_grid_size()` at call time (a parameter, never a constant) |
| Split | `ttnn.split_work_to_cores(grid, total_units, row_wise=True)` → `(num_cores, all_cores, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2)`; cores enumerated in the same order; `start_unit` is the running sum |
| Per-core work | RT args `start_unit`, `num_units` (same pair to reader, compute, writer). Segments: `r = u / Ct`, `c = u % Ct`, `seg_col_tiles = min(Ct − c, remaining)` |
| Remainder | balanced ±1 unit by `split_work_to_cores`; ragged last block per segment handled by `block_valid_col_tiles` (runtime extent); `tensor_token_tiles` uses `ceil(T/32)` per image, never `floor(prod(lead)·T/32)` |
| Idle cores | cores outside `all_cores` get no kernels (`total_units < grid` only for tiny shapes, e.g. `(32, 128)` → 1 unit) |
| Examples (n=4, grid 110) | T=640, C=7168: 4480 units → 40–41 per core, ≤ 2 rows per core. T=640, C=1792: 1120 → 10–11. T=4096, C=1792: 7168 → 65–66, ≤ 3 rows. T=1, C=7168: 224 → 2–3 per core |

Single regime: no selection function.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_sublayer_tiles` | 0 | F tile bytes (`F.buffer_page_size()`) | `depth_in × block_col_tiles` | spans c (B); streams r, j (resident across the j walk), i (absent) | F dtype (Float32 Phase 0) | reader | compute | per block |
| `cb_residual_tiles` | 1 | X tile bytes | `depth_in × n × block_col_tiles` | spans c (B) and i (n); streams r, j | X dtype (Float32 Phase 0) | reader | compute | per block |
| `cb_coef_raw` | 2 | 4096 (fp32 tile) | `ceil(n/32) + ceil(n²/32)` (= 2) | spans nothing of the block: one raw tile per coefficient tensor for one token row | Float32 | reader | reader | per segment (read → expand → pop) |
| `cb_coef_bcast` | 3 | 4096 (fp32 tile) | `coef_depth × (n + n²)` | spans k (n + n²) for one token row; streams c, j, i (indexed, not streamed); r: one row per set | Float32 | reader | compute | per segment |
| `cb_output_tiles` | 16 | X' tile bytes | `depth_out × n × block_col_tiles` | spans c (B) and j (n); streams r, i | X dtype (Float32 Phase 0) | compute | writer | per block |

Unpack-to-dest: `UnpackToDestFp32` on every CB above whose format is Float32 and that compute reads with `copy_tile` (`cb_sublayer_tiles`, `cb_residual_tiles` in Phase 0; `cb_coef_bcast` always); `Default` otherwise. Derived on the host from each CB's data format — one rule, no per-dtype literal.

CB sync: `cb_sublayer_tiles` push B / wait B / pop B; `cb_residual_tiles` push n·B / wait n·B / pop n·B; `cb_output_tiles` reserve n·B / push n·B, writer wait n·B / pop n·B; `cb_coef_bcast` push n+n² / wait n+n² / pop n+n² per segment; `cb_coef_raw` reader push/wait/pop 2 per segment. Every push count equals the matching wait count, and every quantum divides its capacity exactly.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_coefficients` | one token row | `fill_l1_range<4>` for the stores; TensorAccessor for the reads | DRAM post/comb → `cb_coef_raw` (2, reserved/pushed, then waited by reader) | `cb_coef_bcast` (n + n²) | `cb_coef_raw` popped; `cb_coef_bcast` holds this row's set until `release_coefficients` |
| 2 | `load_block` | B columns × (1 + n) | TensorAccessor | DRAM F, X | `cb_sublayer_tiles` (B), `cb_residual_tiles` (n·B) | pushed nominal B / n·B |
| 3 | `mix_block` | n streams × `block_valid_col_tiles` | `eltwise_chain` with `CopyTile`, `MulBinary`, `Addcmul`, `PackTile` | `cb_sublayer_tiles` (B, waited once), `cb_residual_tiles` (n·B, waited once), `cb_coef_bcast` (n+n², waited once per segment, indexed, not popped) | `cb_output_tiles` (n·B, reserved once) | F/X popped after the last j; output pushed n·B |
| 4 | `store_block` | n streams × `block_valid_col_tiles` | TensorAccessor | `cb_output_tiles` (n·B) | DRAM X' | popped n·B |
| 5 | `release_coefficients` | one token row | — | `cb_coef_bcast` | — | popped n + n² |

`mix_block`, per output stream j, one `eltwise_chain(IterationShape::tiles(block_valid_col_tiles), …)` whose per-tile body is, in order (caller-managed `(None, None)` lifecycles, `TileAddressing::Offset` bases):

| Step | Element | Reads | Writes |
|------|---------|-------|--------|
| a | `CopyTile<input(cb_sublayer_tiles, None, None, Block, Enabled, Direct), D1>` | F tile c | D1 |
| b | `CopyTile<input(cb_coef_bcast, None, None, Scalar, Enabled, Offset), D2>{j}` | post_j expanded | D2 |
| c | `MulBinary<D1, D2, D0>` | D1, D2 | D0 = post_j·F |
| d (×n, i = 0..n−1) | `CopyTile<input(cb_residual_tiles, None, None, Block, Enabled, Offset), D1>{i·B}`; `CopyTile<input(cb_coef_bcast, None, None, Scalar, Enabled, Offset), D2>{n + i·n + j}`; `Addcmul<DataFormat::Float32, D0, D1, D2, D0>{0x3F800000}` | X_i tile c, comb_ij expanded | D0 += comb_ij·X_i |
| e | `PackTile<output(cb_output_tiles, None, None, Enabled, Offset), D0>{j·B}` | D0 | out slot j·B + c |

The n repetitions of step d are generated at compile time from the CT arg `n` (index-sequence pack expansion). The implementer may realize `mix_block` as this chain, a thin wrapper, or a raw-LLK block op with the same per-tile arithmetic order, the same fp32 SFPU operations and no FPU math on fp32 operands.

**As built after Refinement 2** (the per-tile chain above is the Phase 0 realization). Per output stream j, the block's columns are walked in DEST windows, each one `eltwise_chain(IterationShape::one_tile(), …)` iteration in SyncFull DEST:
1. `CopyTile` × P: stream j's half-packed coefficient tiles (`cb_coef_bcast` tile j·P + p) → D0 .. D(P−1).
2. `CopyTile` × (n+1) per window column: F[c] and X_0..X_{n−1}[c] → the column's data slots. Copies are grouped by source CB. Consecutive windows alternate coef-first / data-first, so a window starts on the CB its predecessor ended on, and that first copy's (unconditional) reconfig is disabled.
3. `WeightedSum` (custom DEST-only chain element, raw SFPI): `out = Σ_t d_t · c_t`, with the same arithmetic as the old chain (`d_0·c_0`, then fused MADs in term order). It writes over the column's F slot, keeps the accumulator in LREGs, and reads each coefficient vector once for the two data faces of the same rows.
4. `PackTile` → out slot j·B + c. Pack reconfig is disabled because the output CB is the only pack target.

The coefficient expansion writes each term into half a tile: faces 0/2 for even t, faces 1/3 for odd t (`mhc_post_common.hpp`). It no longer duplicates faces over the NoC.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | raw_api (mandatory boot) | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | `(cb_residual_tiles, cb_coef_bcast, cb_output_tiles)`; first statement of `MAIN()` | — | — | none |
| `mix_block` | helper | `eltwise_chain` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:533` | `IterationShape::tiles(block_valid_col_tiles)` (`chain.hpp:121`); owner `Chain` | see elements | see elements | `block_valid_col_tiles` (≤ `block_col_tiles`) is the shape |
| `mix_block` | helper | `CopyTile` | `chain.hpp:482`; impl `eltwise/core/chain.inl:874` | `input(cb, WaitPolicy::None, PopPolicy::None, Block|Scalar, Enabled, Offset)` (`chain.hpp:356`); legal per `chain.inl:77` | `cb_sublayer_tiles`, `cb_residual_tiles`, `cb_coef_bcast` | DEST D1/D2 | base offsets `i·B`, `j`, `n+i·n+j` derived from `block_col_tiles`, `n` |
| `mix_block` | helper | `MulBinary` (SFPU `mul_binary_tile`) | `eltwise/binary/sfpu/basic.hpp:28`; impl `eltwise/binary/sfpu/sfpu.inl:55`; LLK `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h:67` | `<D1, D2, D0>` | DEST | DEST | none |
| `mix_block` | helper | `Addcmul` (SFPU `addcmul_tile`) | `eltwise/ternary/ternary.hpp:26`; impl `eltwise/ternary/ternary.inl:38`; LLK `tt_metal/hw/inc/api/compute/eltwise_unary/addcmul.h:41`, `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h:51` (`a_prod * a_in2 + a_in0`, fp32 store when fp32 DEST) | `<DataFormat::Float32, D0, D1, D2, D0>{0x3F800000}` | DEST | DEST | none |
| `mix_block` | helper | `PackTile` | `chain.hpp:503`; walk semantics `chain.inl:1036` | `output(cb_output_tiles, ReservePolicy::None, PushPolicy::None, Enabled, Offset)` (`chain.hpp:383`); legal per `chain.inl:84` | DEST D0 | `cb_output_tiles` | base `j·B` |
| `load_coefficients` | helper | `fill_l1_range<4>` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:90` | `(face_row_addr, 64 /*16 fp32*/, raw_value_bits)` per (tile, row ρ, column half) | `cb_coef_raw` | `cb_coef_bcast` | none (n + n² from `n`) |
| `load_coefficients`, `load_block`, `store_block` | raw_api | `TensorAccessor` page read / write + one `noc_async_read_barrier` / `noc_async_write_barrier` per block | `tech_reports/tensor_accessor/tensor_accessor.md` | CT: `TensorAccessorArgs(F/X/post/comb/out)` at the end of the CT arg list | DRAM | CBs / DRAM | trip counts from `block_valid_col_tiles`, `n` |

Helpers considered and rejected:

| Candidate | File:Line | Why it cannot be used here |
|-----------|-----------|----------------------------|
| `BinaryFpu<Mul, …, input(cb, BroadcastDim::Col)>` (FPU column broadcast of a coefficient column) | `chain.hpp:305` (`BroadcastDim::Col`), `chain.hpp:491` (`BinaryFpu`) | FPU operands enter srcA/srcB as tf32; the fp32 coefficients (and fp32 X / F) are truncated → systematic −2⁻¹¹ product bias, forbidden by the precision contract |
| `UnaryBcast<BroadcastDim::Col, …>` (build the column-broadcast coefficient tile in compute) | `eltwise/broadcast/bcast.hpp:25` | the broadcast datacopy passes the coefficient through srcB (tf32) → the expanded coefficient is truncated; the reader-side L1 expansion is lossless |
| `AddBinary` + `MulBinary` pairs for the i terms | `eltwise/binary/sfpu/basic.hpp:20,28` | legal but two SFPU passes and two roundings per term; `Addcmul` is one fused MAD per term |
| raw-LLK compute | — | not needed: the chain expresses every step (the raw_api rows above are boot + dataflow, which have no compute helper) |

`reduce_*`, `tilize` / `untilize`, `matmul_block`, `mcast_pipe`: not applicable (no reduction over a tile axis, TILE in/out, no matmul, no cross-core traffic).

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| mix: post·F | SFPU `MulBinary` in DEST | `cb_sublayer_tiles` → D1: All | `cb_coef_bcast[j]` → D2: All (column broadcast already materialized by the reader) | None (in compute) |
| mix: comb·X | SFPU `Addcmul` in DEST | `cb_residual_tiles[i·B + c]` → D1: All | `cb_coef_bcast[n + i·n + j]` → D2: All | None (in compute) |
| expansion | reader L1 stores | `cb_coef_raw`: Col k (one column of a raw tile) | → `cb_coef_bcast[k]`: All | Col → All (reader) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| tf32 truncation of fp32 operands | `copy_tile` of an fp32 CB without `UnpackToDestFp32` goes through srcA (19-bit) — silently shrinks every term by ~2⁻¹¹; PCC still passes, the signed-bias gate and 122-wrap regression fail | host sets `UnpackToDestFp32` per fp32 CB read by compute (rule in Circular Buffers); no FPU math anywhere |
| comb orientation | `X'_j = Σ_i comb[i][j] X_i` — comb applied transposed; using raw column `j·n + i` passes PCC-like tests on symmetric combs but fails the golden `cyclic` pinning | expanded index `n + i·n + j` is built from raw comb column `i·n + j`; acceptance test includes a cyclic (non-symmetric) comb |
| face layout in the expansion | a tile is 4 faces of 16×16; row ρ ≥ 16 lives in faces 2/3 | expansion addresses via the face formula in Dataflow Strategy; single-tile and multi-row tests pin it |
| three kernels disagreeing on segments/blocks | a different segment boundary in any kernel is a CB count mismatch → hang | one shared inline derivation from (`start_unit`, `num_units`, `Ct`, `block_col_tiles`) used by all three kernels |
| ragged last block | pushing `block_valid_col_tiles` instead of the nominal B breaks the ring-wrap invariant | nominal push/pop counts; only NoC transfers and the chain shape use the valid extent; X/out CB slots use stride B (`i·B + c`, `j·B + c`) so the valid tiles sit at fixed positions |
| `Addcmul` value encoding | `value` is uint32 bits of a float; passing integer 1 multiplies by a denormal | pass `0x3F800000` (1.0f) |
| bf16 X' output rounding (refinement) | packing fp32 DEST to bf16 must round-to-nearest-even, else a systematic bias fails the 1e-5 gate | refinement must verify the packer's fp32→bf16 rounding on device; if it truncates, round in DEST with SFPU RNE before pack |
| SFPU MAD rounding on Wormhole | the golden emulation assumes unbiased fp32 SFPU arithmetic; verified target is Blackhole | first on-device run checks the signed bias on the fp32 cell; a failure on another arch is an arch-specific finding |
| non-aligned T | padded rows exist in every input tile row `r = last` | row-local math: no masking; never introduce a cross-row op (e.g. a transpose) into this op |
| n from `post.shape[-1]` | `comb` last dim must equal n², X last dim n·C | host validation (Parameters) before program build |
| determinism | the requirement is bitwise-identical repeated outputs | static schedule, fixed per-tile arithmetic order, no atomics / races; acceptance test checks equality |

# Operation Design: rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (row reduction + broadcast eltwise), one native dispatch |
| Goal | `y = x / sqrt(mean(x², dim=-1) + ε) · γ` over the last dim, any rank ≥ 2, TILE and ROW_MAJOR, tile-aligned or not, as exactly one `ttnn.generic_op` dispatch per call. |
| Math | `rinv[r] = rsqrt( (1/W) · Σ_{w<W} x[r,w]² + ε )`; `y[r,w] = (x[r,w] · γ[w]) · rinv[r]` (γ ≡ 1 when absent) |
| Mode | Derivative (generic_op + ProgramDescriptor; kernel_lib helpers) |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/precision_convention.md`, `eval/golden_tests/rms_norm/feature_spec.py`, `eval/golden_tests/rms_norm/helpers.py`, catalog `ttnn/ttnn/operations/examples/master.md` (entries cited per knob below), `ttnn/cpp/ttnn/kernel_lib/*` (cited in API Mapping) |

### Files the implementer creates

| File | Content |
|------|---------|
| `ttnn/ttnn/operations/rms_norm/rms_norm.py` | INPUT_TAGGERS (existing `tag_alignment`, `tag_rank`; add `tag_gamma_dtype`, `tag_gamma_layout` returning `"none"` when gamma is None), SUPPORTED, EXCLUSIONS, `default_compute_kernel_config()`, `validate()`, `rms_norm()` |
| `ttnn/ttnn/operations/rms_norm/rms_norm_program_descriptor.py` | `select_blocking(...)` (the single source of every knob, §Work Distribution), CB / kernel / semaphore / Mcast descriptors |
| `ttnn/ttnn/operations/rms_norm/kernels/rms_norm_{reader,compute,writer}.cpp` | the three kernels |

### Phase 0 scope (the SUPPORTED rectangle — implementer writes the block)

| Axis (TARGET name) | Phase 0 | Refinement candidates (TARGET − Phase 0) | Unlock class |
|---|---|---|---|
| `dtype` | float32, bfloat16 | bfloat8_b | knob-turn (CB formats; INVALID already removes bf8b×RM and bf8b×non-aligned) |
| `fp32_dest_acc_en` | True | False (bf16 only) | knob-turn: intermediate CB page format Float32 → Float16_b (ledger audit 2) |
| `layout` | TILE, ROW_MAJOR | — | — |
| `alignment` | tile_aligned, w_non_aligned, h_non_aligned | — | — |
| `rank` | 2, 3, 4 | — | — |
| `gamma_mode` | gamma, no_gamma | — | — |
| `gamma_dtype` | float32, bfloat16, `"none"` | bfloat8_b | knob-turn (cb_gamma format) |
| `gamma_layout` | ROW_MAJOR, TILE, `"none"` | — | — |
| `memory_layout` | INTERLEAVED | HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED | placement of already-built regimes (see Dataflow Strategy → sharded placements) |

EXCLUSIONS (permanent per precision convention): `{dtype: float32, fp32_dest_acc_en: False}`.
`"none"` is always legal for `gamma_dtype` / `gamma_layout`; `validate()` never refuses it.

### Compute-config contract

| Item | Decision |
|------|----------|
| Default factory | `default_compute_kernel_config()` returns `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)` — the Phase 0 corner. **The current stub returns `fp32_dest_acc_en=False`; Phase 0 must change it to `True`**, otherwise `rms_norm(x)` resolves to an unsupported cell (and float32 to an EXCLUDED one). When the `bf16 + fp32_dest_acc_en=False` refinement lands, the default may move back to match `ttnn.rms_norm`; the factory is the only place it changes. |
| None resolution | `cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()` — nowhere else inlines the default |
| Gating | `validate()` reads `cfg.fp32_dest_acc_en`; `False` → refused (`UnsupportedAxisValue`, Phase 0); `float32 + False` → `ExcludedCell`. `math_fidelity` / `math_approx_mode` never gated |
| Use | compute KernelDescriptor `config=cfg` as-is (honors fidelity / approx). `Rsqrt<Approx::Exact>` regardless of approx mode is NOT allowed — map `math_approx_mode` → `Rsqrt<Approx>` template arg via a CT define so the caller's choice is honored |

### Validation order (`validate()` / entry point)

| # | Check | Raises |
|---|-------|--------|
| 1 | `len(input.shape) < 2` | `ValueError` |
| 2 | gamma present and `gamma.shape[-1] != input.shape[-1]`, or gamma volume ≠ W | `ValueError` |
| 3 | SUPPORTED per axis (axes built exactly like the harness: dtype, layout, fp32_dest_acc_en from cfg, gamma_mode/dtype/layout with `"none"` sentinel, memory_layout, then INPUT_TAGGERS) | `UnsupportedAxisValue` |
| 4 | EXCLUSIONS | `ExcludedCell` |
| 5 | `memory_config` (output placement) must be None or interleaved in Phase 0 | `UnsupportedAxisValue` |

No host-side tensor transformation of any kind (no to_layout / tilize / pad / slice / reshape of device data). Gamma is consumed in whatever layout it arrives (RM `(1,1,1,W)` stick, or TILE).

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor | yes | rank ≥ 2; Phase 0: fp32/bf16, TILE/RM, DRAM interleaved | — | — |
| gamma | ttnn.Tensor | no | last dim == W; shape (1,1,1,W); fp32/bf16; RM or TILE | None | CT flag `has_gamma`, `gamma_is_rm` |
| epsilon | float | no | > 0 (any float accepted; ε ≤ 0 is caller's numerics) | 1e-6 | RT (fp32 bits) to compute |
| compute_kernel_config | ComputeConfigDescriptor | no | fp32_dest_acc_en=True in Phase 0 | `default_compute_kernel_config()` | kernel config |
| memory_config | ttnn.MemoryConfig | no | None or interleaved (Phase 0) | None → DRAM interleaved | host |
| (derived) inv_w | float | — | 1/W, logical W (not padded) | — | RT (fp32 bits) to compute |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[..., H, W]`, rank 2–4, any H, W ≥ 1 |
| Dtype | float32, bfloat16 |
| Layout | TILE or ROW_MAJOR |
| Memory | interleaved (DRAM in tests; L1-interleaved is the same code path) |

### Gamma

| Property | Requirement |
|----------|-------------|
| Shape | (1,1,1,W) |
| Dtype | float32, bfloat16 (may differ from input) |
| Layout | ROW_MAJOR (one stick of W elements) or TILE (one tile-row, row 0 valid) |
| Memory | interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input (logical and padded) |
| Dtype | = input dtype |
| Layout | = input layout |
| Memory | `memory_config` or DRAM interleaved; allocated on host (`ttnn.allocate_tensor_on_device`) before the dispatch |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Everything below this section is its realization.

### Axes

Shape-derived extents (host, alignment-aware, per image):

| Symbol | TILE | ROW_MAJOR |
|--------|------|-----------|
| `tensor_rows_tiles` | `prod(shape[:-2]) · ceil(H/32)` (each image tile-padded independently) | `ceil(prod(shape[:-1]) / 32)` — sticks are contiguous, a tile-row of 32 sticks may span images (rows are independent, so this is legal and wastes nothing) |
| `tensor_w_tiles` | `ceil(W/32)` | `ceil(W/32)` |
| `valid_rows_in_last_tile_row` | per image: `H − 32·(ceil(H/32)−1)` (only matters for padding, never masked — rows are independent) | `prod(shape[:-1]) − 32·(tensor_rows_tiles−1)` (reader/writer skip absent sticks) |
| `w_valid_in_last_tile` | `W − 32·(tensor_w_tiles−1)` | same |

Leading dims (batch, channel, H-tiles) fold into one axis **R**: the op treats them identically and they are contiguous in tile-row order.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| **R** — tile-rows (leading dims × ceil(H/32)) | **independent** — every output row depends only on its own input row | `block_rows` | coarsest that fits: `min(line_rows_tiles_max, ⌊(L1_CB_BUDGET_BYTES − fixed_bytes)/per_row_bytes⌋)` (≥ 1 by construction of `w_split`) | `select_blocking()` → CT arg `block_rows` | contiguous tile-row ranges, balanced over `num_lines` lines (`⌈⌉/⌊⌋` groups), lines laid row-major over the full grid | knob-turn |
| **W** — tiles along last dim | **dependent** — the Σx² that every element's result needs spans the whole axis | `block_w_tiles ≡ core_w_tiles` (a block always spans the core's whole W-slice: residency of the slice is what removes the second DRAM read; cutting W *within* a core is the rejected STREAM_TWO_PASS regime) | `core_w_tiles = ⌈tensor_w_tiles / w_split⌉` | `select_blocking()` → CT arg `core_w_tiles` (nominal, identical on every core); RT `core_w_valid_tiles`, `core_w_start_tiles` | `w_split` consecutive cores form a **line** that shares one R-range; `w_split = 1` → LOCAL, `> 1` → W_SPLIT | scheme-change — **built in Phase 0** (W_SPLIT) |
| **γ along W** (operand) | **reuse-shared** — γ does not vary along R, the primary split | — (γ slice = `core_w_tiles`, resident for the whole kernel) | read once per core (row-0 lanes only) | follows `core_w_tiles` | each core reads its own W-slice | GAMMA_MCAST (deferred row) |
| **within-tile** (32×32 lanes) | rows independent, cols dependent (collapsed by the SFPU within-tile reduce) | fixed by hardware tile | 32×32 | — | — | — |

Stage outputs re-tagged (every stage's data has axes too):

| Stage output | Axes | Character | Block | Who computes it | Why |
|---|---|---|---|---|---|
| `sumsq_raw` (Σ_W x² per lane, raw 32×32 per tile-row) | R | independent | `block_rows` tiles | every core, on its own (R-range × W-slice) | spread by construction |
| `partial` (column-0 collapsed sumsq of one slice) | R × line_pos | line_pos dependent | `block_rows` tiles per core | every core | spread |
| `rinv` (combined, finalized, broadcast to full tile) | R | independent | `block_rows` tiles | **LOCAL**: the core itself. **W_SPLIT**: the line root (communication role: gather + mcast) **plus** the combine compute (`w_split` tile-adds + one finalize per tile-row) | Spreading the combine across the line (reduce-scatter) is the deferred COMBINE_SCATTER row: the root's extra compute is ~`w_split/2 + 4` tile-ops per tile-row vs. each core's `3·core_w_tiles` tile-ops of its own work, and with `min_core_w_tiles = 8` the root extra stays ≤ the line's per-core work for `w_split ≤ 40`; beyond that the grid-sync lamp applies |
| `y` | R × W | independent | `block_rows × core_w_tiles` | every core | spread |

Every axis is assigned; nothing is single-core: Phase 0 spreads R (and, when R under-fills the grid or a row does not fit, W) across the full grid.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x` (TILE) | `x_block_depth` | 2 blocks | reader prefetches block b+1 while compute runs normalize(b) — and, in W_SPLIT, **during the combine round trip** (the stall shadow, see below) |
| `cb_x` (RM) | (fixed) | 1 block | producer is compute (tilize); compute is sequential, nothing to overlap — the RM double buffer lives in `cb_x_rm` |
| `cb_x_rm` (RM) | `x_rm_depth` | 2 tile-rows | reader fills tile-row i+1 while compute tilizes tile-row i |
| `cb_y` (TILE) | `y_depth` × `y_chunk_tiles` | 2 × 8 tiles | compute packs chunk c+1 while writer drains chunk c (catalog `double_buffer`: 4–8 tiles per barrier, gains flat beyond) |
| `cb_out_rm` (RM) | `out_rm_depth` | 2 tile-rows | compute untilizes tile-row i+1 while writer writes sticks of tile-row i |
| reader transfer quantum | `read_chunk_tiles` | 8 tiles per barrier+push (TILE) | NoC reads in flight per barrier (catalog `double_buffer`, `trid_double_issue`); also the granularity at which the cumulative sumsq can start |
| all other CBs | — | 1 block | sequential compute→compute hand-offs or once-per-block messages; nothing to overlap |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `tilize` / `untilize` block width is a **template parameter** (`tilize_helpers.hpp:187-197`, `untilize_helpers.hpp:145-154`) | `core_w_tiles` must be compile-time and identical on all cores | `core_w_tiles` is the **nominal** slice width (CT); a core whose valid slice is narrower (last core of a line) runs the nominal width over pre-zeroed tail tiles | per-core CT widths would need per-core CB layouts; nominal-with-zeros keeps one layout and makes pushes ring-closing |
| CB ring pointer resets only on an exact `fifo_limit` hit | every push/pop quantum on `cb_x`, `cb_x_rm`, `cb_y`, `cb_out_rm` | pushes/pops are **nominal**: `cb_x` per block `block_rows·core_w_tiles` (chunks never straddle a block boundary); `cb_y` writer pops one page at a time after each chunk's barrier; ragged last block only ever occurs as a core's final block | pointer overshoot → silent data corruption on the next wrap |
| Remote-written CBs (`cb_gathered`, `cb_rinv`) are addressed by the sender at **its own** CB address | all CBs allocated on **all** participating cores with identical descriptors; `cb_gathered` capacity = exactly one block (`w_split·block_rows`), `cb_rinv` = exactly one block (`block_rows`) so the landing address is the constant CB base | senders write a slot the receiver has already moved past / into a different core's layout |
| `cb_gathered` fan-in `w_split · block_rows` Float32 tiles | `w_split ≤ grid cores`; term counted in the L1 predicate | `select_blocking()` includes `w_split·block_rows·4096` in `per_row_bytes` | CB allocation OOM at large k |
| Single-core residency of one W-slice | `core_w_tiles` | `w_split ≥ w_split_fit` (smallest k whose `block_rows = 1` footprint fits) | CB OOM; if even `w_split = grid cores` does not fit (W ≳ grid·~300 tiles, far beyond TARGET), `validate()`-time `UnsupportedAxisValue("row too wide")` from the host before dispatch |
| DEST capacity (fp32 → `DEST_AUTO_LIMIT = 4`, `dest_helpers.hpp:103`) | none on block extents: every compute op keeps **one** live accumulator (`D0`) per output tile; chains clamp their internal `block_size` themselves (`chain.hpp:435-439`) | — | — |
| `AccumulateViaAdd` streaming / partial restrictions (`reduce_helpers_compute.hpp:141-158`) | collapse/combine inputs must be tile-aligned along the reduced dim | inputs to both reduces are already-summed whole tiles (padding was zeroed before the square), so the aligned path is always taken; `ReducePartialScaler::none()` | — |
| NoC read alignment (DRAM 64 B on BH) | γ TILE row-0 reads; RM stick offsets | γ TILE: read face0-row0 at offset 0 and face1-row0 at offset `256·es` (both 64 B-aligned for bf16/fp32); RM: W-slice offset = `core_w_start_tiles·32·es` (multiple of 64 B) | misaligned DRAM reads return shifted data |

### Regimes

Named boundary for the minimum: **DRAM** — x crosses once, y crosses once, γ crosses once (its W elements).

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **LOCAL** — R split across the grid, whole row resident per core | **built** | `select_blocking().w_split == 1`, i.e. `footprint(block_rows=1, core_w_tiles=tensor_w_tiles, k=1) ≤ L1_CB_BUDGET_BYTES` **and** `k_occ == 1` (`tensor_rows_tiles ≥ grid_cores` or `tensor_w_tiles < 2·min_core_w_tiles`) | `block_rows × tensor_w_tiles` | x 1×, y 1× (minimum). γ: `num_lines` × (W·es_γ) — one row-0 read per core, ≤ 1/32 of one tile-row of x per core; above minimum by `(num_lines−1)·W·es_γ` (GAMMA_MCAST removes it) | per block: 5–6 compute phase switches (chain/reduce/tilize/untilize init + format reconfig), the cb_x block handshake, pipeline fill/drain of the cumulative sumsq. Intended frequency: once per block |
| **W_SPLIT** — lines of `w_split` cores share an R-range; each core holds its W-slice resident; Σx² partials gathered to the line root, rinv multicast back | **built** | `select_blocking().w_split > 1` (row does not fit one core, or R under-fills the grid and W ≥ `2·min_core_w_tiles`) | `block_rows × core_w_tiles` | x 1×, y 1×. γ: `num_lines·W·es_γ` (fewer lines than LOCAL → less). Cross-core per block per line: `(w_split−1)·block_rows` Float32 tiles unicast to root + one `block_rows`-tile multicast to `w_split−1` receivers + 2 semaphore signals | everything LOCAL lists **plus** one gather + mcast rendezvous per block (the dominant fixed cost: ~a NoC round trip) — larger `block_rows` pays it fewer times |
| **STREAM_TWO_PASS** — one core per row-range, W cut into chunks, pass 1 accumulates Σx² across chunks (cross-call `Accumulate`), pass 2 re-reads x from DRAM to normalize | **rejected** — superseded by W_SPLIT | — | `block_rows × w_chunk` | x crosses DRAM **2×** | — (dead end: W_SPLIT holds any row up to `grid_cores·core_w_tiles_max` resident with x read once and fills the grid while doing it; no good scheme passes through a second DRAM read) |
| **GAMMA_MCAST** — one injector per grid reads γ once and multicasts each slice to the cores that share it | **deferred** — γ traffic is `num_lines·W·es_γ`, at most `(num_lines/tensor_rows_tiles)·(es_γ/es_x)/32 ≤ 1/16` of x's DRAM bytes (row-0-only reads; ≤ 1/32 when γ dtype = x dtype), while it adds one rendezvous per core | as LOCAL / W_SPLIT | unchanged | γ 1× (minimum) | — |
| **COMBINE_SCATTER** — the `block_rows` tile-rows of a block's combine are spread over the line's cores (reduce-scatter + all-gather) instead of all on the root | **deferred** — the root's combine is `w_split/2` pairwise adds + 1 finalize per tile-row, ≤ the per-core main work for `w_split ≤ 40` with `min_core_w_tiles = 8`; a scatter needs ≥ 2 rendezvous per block, each costing more than the work it moves. Reachable: combine is already its own named op on its own CBs; only the gather slot map and the mcast sender set change | W_SPLIT ∧ `block_rows ≥ w_split` | same | same bytes, root hotspot removed | — |
| **TREE_COMBINE** — partials reduced up a log₂(k) tree instead of k→1 | **deferred** — a tree adds `log₂(w_split)` serial rendezvous per block; with the Phase 0 `min_core_w_tiles` the root fan-in is ≤ `tensor_w_tiles/8` tiles, so k→1 is one rendezvous against ≥ 3. Reachable: same partial/gather CBs, only sender targets change | W_SPLIT ∧ large `w_split` | same | same total bytes, lower root fan-in | — |
| **HEIGHT_SHARDED placement of LOCAL** | **built (Refinement 2)** — this is placement, not a new algorithm: shard = one line's R-range × full W, `cb_x` backed on the shard (`cb_descriptor_from_sharded_tensor`, zero-copy), block = whole shard (sub-chunked only if `footprint` fails), output written into its own sharded CB | input HEIGHT_SHARDED | whole shard | x 0× DRAM (already in L1) | as LOCAL |
| **WIDTH / BLOCK_SHARDED placement of W_SPLIT** | **built (Refinement 2)** — same as above; the line = one shard-grid row, `w_split` = shard-grid width, `core_w_tiles` = shard width; the combine already exists, so it is a knob-turn plus placement | input WIDTH/BLOCK_SHARDED | whole shard | x 0× DRAM; combine as W_SPLIT | as W_SPLIT |

Selection function and regime-pinned tests: see Work Distribution.

### Traffic ranking

Counted per tensor and tier (DRAM most expensive, then cross-core, then core-local). `R = tensor_rows_tiles`, `Wt = tensor_w_tiles`, `C` = grid cores, `T` = tile bytes.

| Rank | Candidate split | x DRAM | y DRAM | γ DRAM | Cross-core | Notes |
|------|-----------------|--------|--------|--------|------------|-------|
| 1 | **R split, row resident** (LOCAL) | 1× | 1× | `min(R,C)` × W·es_γ | 0 | cheapest when one row fits and R fills the grid |
| 1′ | **R × W split, slice resident, combine** (W_SPLIT) | 1× | 1× | `num_lines` × W·es_γ (≤ LOCAL's) | `R·(k−1)` Float32 tiles in + `R` tiles mcast to `k−1` | the only split that keeps x at 1× when a row does not fit, and the only one that fills the grid when R < C. Combine bytes per core per tile-row: 4 KB out (partial) + 4 KB in (rinv) vs `core_w_tiles·T_in` of x read and written |
| 2 | R split with W streamed through one core (STREAM_TWO_PASS) | **2×** | 1× | as LOCAL | 0 | rejected |
| 3 | any split re-reading γ per block | 1× | 1× | `R·W·es_γ` | 0 | not built: γ is held resident per core (reuse-shared → read once per core) |

Choice: LOCAL and W_SPLIT are both minimum-x splits; the selector picks LOCAL whenever it fits **and** fills the grid (no combine), W_SPLIT otherwise. Occupancy is used only to choose *between two minimum-traffic splits*, never to override the traffic ranking.

Operand-reuse check (every operand × the chosen split):

| Operand | Varies along R (primary split)? | Varies along W (W_SPLIT)? | Consequence |
|---------|----------------------------------|---------------------------|-------------|
| x | yes | yes | no reuse |
| γ | **no** → reuse-shared by construction | yes | read once per core and kept resident across all its blocks; cross-core broadcast = GAMMA_MCAST (deferred) |
| rinv (stage output) | yes | **no** → every core of a line needs the same tile | computed once on the root and **multicast** to the line (built) |

Stall-shadow check (W_SPLIT): after pushing `partial(b)` a non-root core blocks until `rinv(b)` arrives; the root blocks until all partials arrive. Work independent of that wait:

| Independent work | Scheduled into the stall? | How |
|------------------|---------------------------|-----|
| reading x(b+1) from DRAM | **yes (built)** | `x_block_depth = 2`: the reader never waits on the combine |
| draining y(b−1) to DRAM | **yes (built)** | writer is independent of compute's wait |
| loading γ | **yes (built)** | writer loads γ before its first partial; compute tilizes RM γ lazily before the first normalize |
| `x·γ` (commutes with `·rinv`: `y = (x·γ)·rinv`) | **yes, gated (Refinement 3, `Blocking.xg_prescale`)** — W_SPLIT ∧ TILE ∧ resident γ, and only when the one-block `cb_xg` (`br·cw` intermediate tiles) fits beside the chosen block without shrinking `block_rows`. Non-roots run it right after collapse, the root right after finalize (overlapping its own multicast; combine stays first). After the rendezvous only `y = xg·rinv` (one FPU mul) remains; FP order and DEST precision are the fused chain's, so y is bit-identical. Measured: −0.4 to −0.8 µs on the WIDTH decode cases, −1.0 µs on interleaved decode, −2.5 to −5 µs on the BLOCK 640-row / 7168×1024 shards. Elsewhere (LOCAL, RM, streamed γ, or no L1 room) the fused chain stays | `prescale_block` → `cb_xg` |
| sumsq(b+1) before normalize(b) | **no (perf lamp L6)** — legal by independence, but it breaks the causal chain that makes the gather/mcast landing slots safe without a ready handshake (see Dataflow Strategy); needs an extra ready semaphore | — |

FP-order note: the fused order `(x·γ)·rinv` is used in **both** regimes, so LOCAL and W_SPLIT produce bit-identical y for the same rinv.

### Block schedule

```cpp
// prologue (once per kernel)
load_gamma();                      // writer: γ slice (row-0 lanes) -> cb_gamma_rm (RM γ) / cb_gamma (TILE γ)
// compute: tilize_gamma() once (RM γ only), immediately before the first normalize_block
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_x_block(block_idx);           // reader -> cb_x (TILE) | cb_x_rm (RM)
    tilize_x_block(block_idx);         // RM only: cb_x_rm -> cb_x
    sumsq_block(block_idx);            // Σ_W x² per tile-row, raw 32x32, x stays resident
    collapse_block(block_idx);         // within-tile row collapse -> cb_rinv_col (LOCAL) | cb_partial (W_SPLIT)
    gather_partials_block(block_idx);  // W_SPLIT: every core's writer -> root's cb_gathered
    combine_block(block_idx);          // W_SPLIT root: Σ over the line's partials -> cb_rinv_col
    finalize_rinv_block(block_idx);    // LOCAL / root: bcast col, ·(1/W), +ε, rsqrt -> cb_rinv | cb_rinv_bcast
    broadcast_rinv_block(block_idx);   // W_SPLIT: root writer mcast -> every line core's cb_rinv
    prescale_block(block_idx);         // xg_prescale only (W_SPLIT TILE γ): x·γ -> cb_xg before the rinv wait
                                       // (non-root after collapse, root after finalize); pops the x block
    normalize_block(block_idx);        // y = (x·γ)·rinv -> cb_y (TILE) | in place in cb_x (RM); pops the x block
                                       // (xg_prescale: y = xg·rinv)
    untilize_y_block(block_idx);       // RM only: cb_x (now y) -> cb_out_rm
    store_y_block(block_idx);          // writer -> DRAM
}
```

| Block operation | Kernel | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------------|--------|-------------|--------------------|-------------------------------|
| `load_gamma` | writer | `core_w_tiles` γ tiles (row-0 lanes) | — | once per kernel; pre-zero cb_gamma / cb_gamma_rm once |
| `tilize_gamma` (RM γ) | compute | 1 tile-row × `core_w_tiles` (1 valid row) | cb_gamma for the whole kernel | once per kernel (tilize init+uninit once) |
| `load_x_block` | reader | `block_rows × core_w_tiles` (TILE: pushed in `read_chunk_tiles` chunks; RM: one tile-row of sticks per push) | — | one NoC barrier per `read_chunk_tiles` (TILE) / per tile-row (RM); pre-zero of never-written L1 once per kernel; W-pad lane zeroing once per tile-row (TILE, partial-W core only) |
| `tilize_x_block` (RM) | compute | `block_rows` tile-rows × `core_w_tiles` | — | tilize init/uninit once per block |
| `sumsq_block` | compute | grid(`block_rows`, `core_w_tiles`), DEST accumulation per tile-row | cb_x block (waited cumulatively, **not popped**) | chain init/reconfig once per block; one DEST acquire/pack per tile-row |
| `collapse_block` | compute | `block_rows` × 1 tiles | cb_x block | reduce light init + SFPU reduce macro once per block |
| `gather_partials_block` (W_SPLIT) | writer (all) → reader (root) | `block_rows` Float32 tiles per core | cb_x block (both ends) | one write burst + one barrier + one semaphore inc per core per block; one semaphore wait per root per block |
| `combine_block` (W_SPLIT root) | compute | `block_rows × w_split` tiles → `block_rows` | cb_x block | reduce init once per block |
| `finalize_rinv_block` | compute | `block_rows` tiles | cb_x block | chain init (bcast + 3 SFPU inits) once per block |
| `broadcast_rinv_block` (W_SPLIT) | writer (root) → reader (all) | `block_rows` Float32 tiles | cb_x block | one multicast + one data-ready signal per line per block |
| `normalize_block` | compute | grid(`block_rows`, `core_w_tiles`) | cb_gamma (whole kernel), cb_rinv (this block) | chain init once per block; pops x block at end |
| `untilize_y_block` (RM) | compute | `block_rows` tile-rows × `core_w_tiles`, read from `cb_x` (holding y after the in-place normalize) | — | untilize init/uninit once per block |
| `store_y_block` | writer | `block_rows × core_w_valid_tiles` tiles (TILE, in `y_chunk_tiles` chunks) / valid sticks (RM) | — | one barrier per chunk (TILE) / per tile-row (RM) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 overlap (`block_rows`) | LOCAL prefill puts only 2–3 tile-rows on a core, so the coarsest `block_rows` makes **one** block per core: normalize+store of the last block cannot overlap any read | `block_rows = ⌈line_rows_tiles/2⌉` (two blocks ping-pong in `x_block_depth = 2`) |
| L2 grid synchronization (`min_core_w_tiles = 8`, `k_occ`) | Decode-like shapes (1 tile-row, wide W) spread to `⌊Wt/8⌋` cores; per-core work then shrinks to ~8 tiles and the per-block gather/mcast rendezvous may dominate (catalog `tensix_all_reduce`, `width_split`) | `min_core_w_tiles ∈ {4, 16}`; cap `w_split` at 32 |
| L3 `x_block_depth = 2` | The x block term is doubled, which forces W_SPLIT for wide prefill (e.g. 8192×7168 bf16 → `w_split = 2`) where LOCAL with depth 1 would fit | `x_block_depth = 1` (LOCAL, cumulative-sumsq overlap only) vs depth 2 + W_SPLIT |
| L4 gather payload | Partials are column-0-valid, yet whole 4 KB Float32 tiles are gathered (`w_split·block_rows` tiles into one root) | **built (Refinement 3, `GATHER_COL_FACES`)**: faces 0 + 2 only; −450 ns on the 28/32-core WIDTH decode grids, −100 ns on 8/9-core lines. A compact 128 B column was not taken (strided column → 16-element NoC writes) |
| L5 Mcast handshake | `McastConfig(handshake=True)` adds a receiver-ready round trip per block; the causal chain partial(b+1) ⇒ cb_rinv(b) popped already guarantees the landing slot is free | **built (Refinement 3, `RINV_MCAST_HANDSHAKE = False`)**: −10 to −110 ns |
| L11 finalize scope (Refinement 3) | The finalize chain broadcast column 0 first and then ran `*(1/W)`, `+ε`, `rsqrt` over all 32 vectors of a tile (stock elements hardcode `VectorMode::RC`): ~1.4 µs per tile-row on the W_SPLIT root, on the rendezvous critical path | **built**: one fused SFPU pass (`RinvColumn`) over the 8 column-0 vectors (`VectorMode::C` + even-parity stride, catalog `sfpu_tile_scope`), in place on `cb_rinv_col`, then `UnaryBcast<Col>`; ~0.6 µs. Shared by LOCAL |
| L12 γ load placement (Refinement 3) | The writer loaded γ (zero-fill of the whole chunk + 2 row-0 reads per tile + barrier, 1.3–1.9 µs) **before** sending its partial | **built**: tail-only zero fill; W_SPLIT issues the reads at kernel start (inside the writer's wait for the first partial) and commits after the partial send |
| L6 sumsq lookahead | In W_SPLIT compute idles during the combine round trip | issue `sumsq_block(b+1)` + `collapse_block(b+1)` before `normalize_block(b)` (needs a gather-ready semaphore, see stall-shadow table) |
| L7 fused normalize via DEST reuse | catalog `compute_fusion`: FPU fed from DEST measured 0.82–0.94× vs CB-fed in its setting | split into `x·γ → cb_xg` (Float32 full block) and `cb_xg·rinv` (only if L1 allows; costs one block of Float32) |
| L8 L1 budget constant | `L1_CB_BUDGET_BYTES = 1 MiB` is conservative on Blackhole (≈1.4 MB usable) and decides LOCAL vs W_SPLIT | 1.25 MiB |
| L9 bank stagger | all cores start their walk on the same DRAM bank (RM: 32·k sticks; TILE: when `tensor_w_tiles` is a multiple of the bank count) — catalog `bank_stagger`: 1–7 % | rotate each core's start column/stick by `core_idx % num_banks` |
| L10 `read_chunk_tiles = 8` | catalog `double_buffer` flat 4→64; larger chunk delays the cumulative sumsq start | 4, 16 |

Catalog entries that informed knobs: `double_buffer` (read/write chunk 8, depth 2), `noc_placement` (row-wise core order; reader NoC0 / writer NoC1), `compute_block_size` (block-wide helper calls, one init per block), `compute_fusion` (+ε/rsqrt on SFPU in DEST; multiplies on FPU), `row_reduce_accumulate` / `reduce_block` / `reduce_accumulate` (AccumulateViaAdd, fp32 DEST accumulation, no L1 round trip of the running sum), `eltwise_l1_vs_dest_accumulate` (DEST accumulation of the squares), `width_split` (W split when R is small), `mcast_topology` / `tensix_all_reduce` (1-D line, root reduce + mcast back for 1-tile payloads; no rings), `bank_stagger` (L9).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x DRAM → cb_x (TILE) | input dtype tiles | reader, `TensorAccessor` tile pages `t·Wt + core_w_start_tiles + c`, `read_chunk_tiles` reads per barrier, push per chunk | owner of global tile column `Wt−1` zero-fills lanes `[w_valid_in_last_tile, 32)` of that tile (all 32 rows, face layout) after the barrier and before the push. Tail tiles `c ≥ core_w_valid_tiles` are never read: pre-zeroed once at kernel start in every block slot of cb_x and never touched again (pushes are nominal) |
| x DRAM → cb_x_rm (RM) | input dtype sticks, stride `core_w_tiles·32·es` | reader, one NoC read per stick of `core_w_valid_elems·es` bytes at byte offset `core_w_start_tiles·32·es`; push `core_w_tiles` pages per tile-row (tilize symmetric contract) | cb_x_rm is zero-filled once at kernel start; data writes only ever cover `[0, valid_bytes)` of each stick slot, so W padding and tail tiles stay 0 forever. Absent sticks of the last tile-row (H/stick tail) are not read |
| cb_x_rm → cb_x (RM) | input dtype tiles | compute `tilize` | pure rearrangement |
| γ → cb_gamma | γ dtype tiles, row 0 valid | **TILE γ**: writer reads only face0-row0 and face1-row0 (16 elems each) of each of its `core_w_valid_tiles` tiles. **RM γ**: writer reads the slice of the single stick into `cb_gamma_rm` (one row page), compute tilizes it (1 valid row) into `cb_gamma` | Row-broadcast reads only row 0; rows 1–31 never matter. cb_gamma / cb_gamma_rm zero-filled once before load |
| cb_x → (DEST) → cb_sumsq_raw | Float32 | compute fused square+accumulate | padding lanes are 0 → contribute 0; no mask needed |
| cb_sumsq_raw → cb_rinv_col / cb_partial | Float32, column 0 valid | compute AccumulateViaAdd + SFPU within-tile collapse | — |
| cb_partial (all) → root cb_gathered | Float32 | **W_SPLIT gather**: writer waits `block_rows` partial tiles, unicasts tile r to root address `cb_gathered_base + (r·w_split + line_pos)·4096`, `noc_async_write_barrier()`, then `noc_semaphore_inc(root.sem_partials, 1)`, then pops. Root reader: `cb_reserve_back(cb_gathered, w_split·block_rows)`, `noc_semaphore_wait(sem_partials, w_split)`, reset to 0, push | landing slot is the fixed CB base (capacity = one block). Safe without a ready signal: a core sends partial(b+1) only after its compute consumed rinv(b), which the root produced only after its compute popped gathered(b) |
| root cb_rinv_bcast → every line core's cb_rinv | Float32 full tiles | **W_SPLIT mcast**: `ttnn.Mcast(device, McastConfig(noc=<the root writer's NoC, NOC_1 under WriterConfigDescriptor>), receivers=participating cores (row-major), receiver_group_size=w_split, McastFixedSenderConfig(sender_index=0))`; root writer `sender.send(cb_rinv_bcast read ptr, cb_rinv base, block_rows·4096)`; every core's reader (root included) `cb_reserve_back(cb_rinv)`, `receiver.receive(block_idx)`, `cb_push_back(cb_rinv)`. Channel attached to reader and writer kernels | groups that are not a single rectangle (w_split ∤ grid_x) are handled by the helper (irregular → multiple multicasts). Root's own copy comes through the same send (in-group sender; verify loopback/local-copy semantics in `mcast_args.hpp:18-33`) |
| cb_x, cb_gamma, cb_rinv → cb_y (TILE) / → cb_x in place (RM) | output dtype tiles | compute fused normalize chain | RM: cb_x has compute as its only producer and consumer, so the normalize pops tile i and packs y_i into the slot it just freed (capacity = exactly one block); `untilize` then consumes those pushes. TILE: cb_x is reader-produced, so in place would add a second producer → separate `cb_y` window |
| cb_y → DRAM (TILE) | output tiles | writer, `y_chunk_tiles` per barrier; writes only `c < core_w_valid_tiles`; dead tail tiles popped unwritten | pops one page at a time after the barrier (ring-safe ragged chunks) |
| cb_x (y) → cb_out_rm → DRAM (RM) | output sticks | compute `untilize`; writer writes `core_w_valid_elems·es` bytes per present stick at byte offset `core_w_start_tiles·32·es` | absent sticks skipped |

Semaphores (W_SPLIT only), allocated on all participating cores: `sem_partials` (initial 0; root counts arrivals), plus the Mcast channel's own semaphores (`mcast.attach(..., first_semaphore_id = 1)`).

### Sharded placements (TARGET `memory_layout`, Phase 0 = INTERLEAVED)

| Value | Cuts which axis | Character | Cost class | Realization |
|-------|-----------------|-----------|------------|-------------|
| HEIGHT_SHARDED | R | independent | knob-turn (placement) | LOCAL; cb_x backed on the shard (zero-copy), line = shard core, block = whole shard; output sharded likewise |
| WIDTH_SHARDED | W | dependent | knob-turn (combine already built) | W_SPLIT with the line = the shard grid, `core_w_tiles` = shard width; cb_x backed on the shard |
| BLOCK_SHARDED | R and W | R independent, W dependent | knob-turn (combine already built) | W_SPLIT, one line per shard-grid row |

A physical shard is consumed in place through a CB backed on the sharded buffer, never re-read over the NoC.

Built (Refinement 2): `shard_placement()` turns the shard spec into lines (HEIGHT: one per core; WIDTH: the whole grid in
shard order; BLOCK: one per shard-grid row, per column for COL_MAJOR, with the Mcast core order to match). An
all-padding shard-grid row of a ragged grid is dropped. `select_blocking()`'s sharded branch derives `w_split`,
`core_w_tiles` and `block_rows` (the whole shard when it fits the CB region below the shards), plus the W-chunk
`w_chunk_tiles`. TILE: `cb_x` / `cb_y` are backed on the shards and the reader only publishes the block. Sumsq is
`Strided` over the slice's fully valid tiles, so a ragged shard's tail tiles are never read. The partial-W tile is
squared into a second per-row slot and masked by the collapse reduce (`only_partial()`), so the caller's pad lanes are
never written. RM: a core-local re-stride (L1→L1) in and out. A HEIGHT row too wide to hold γ and the RM staging next
to its two shards streams γ (and chunks the RM tilize / untilize) in `w_chunk_tiles`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_rows` tile-rows × `core_w_tiles` tiles of one line's R-range and one core's W-slice |
| Grid | `device.compute_with_storage_grid_size()` (`grid_x × grid_y`, `grid_cores`); host knob `max_cores` (Phase 0 = all) — the grid is a parameter, never a literal |
| Per-core work | line `i` owns tile-rows `[row_start_i, row_start_i + line_rows_tiles_i)`; core `line_pos` of that line owns tiles `[line_pos·core_w_tiles, …+core_w_valid_tiles)` |
| Remainder | rows: `line_rows_tiles_i = ⌊R/num_lines⌋ + (i < R mod num_lines)`; blocks: `num_blocks_this_core = ⌈line_rows_tiles_i / block_rows⌉`, `last_block_rows` ragged (always the core's final block). W: `core_w_valid_tiles = min(core_w_tiles, Wt − line_pos·core_w_tiles)` ≥ 1 by construction |

### `select_blocking()` — the single source of every knob (host, `rms_norm_program_descriptor.py`)

Constants (one definition each): `L1_CB_BUDGET_BYTES = 1 << 20`, `min_core_w_tiles = 8`, `x_block_depth = 2`, `x_rm_depth = 2`, `y_chunk_tiles = 8`, `y_depth = 2`, `out_rm_depth = 2`, `read_chunk_tiles = 8`.

```
R  = tensor_rows_tiles; Wt = tensor_w_tiles; C = grid_cores
footprint(br, cw, k)  = per_row_bytes(cw, k) * br + fixed_bytes(cw)        # exact terms: l1_ledger.md
k_fit = smallest k in [1, C] with footprint(1, ceil(Wt/k), k) <= L1_CB_BUDGET_BYTES   (none -> refuse "row too wide")
k_occ = 1 if R >= C else max(1, min(C // R, Wt // min_core_w_tiles))
k     = max(k_fit, k_occ)
core_w_tiles = ceil(Wt / k);  w_split = ceil(Wt / core_w_tiles)             # every core of a line has >= 1 valid tile
num_lines    = min(R, C // w_split)
line_rows_tiles_max = ceil(R / num_lines)
block_rows   = min(line_rows_tiles_max, (L1_CB_BUDGET_BYTES - fixed_bytes(core_w_tiles)) // per_row_bytes(core_w_tiles, w_split))
regime       = LOCAL if w_split == 1 else W_SPLIT
cores        = first num_lines * w_split cores of the grid in row-major order (ttnn.grid_to_cores(n, gx, gy, row_wise=True));
               line i = cores[i*w_split : (i+1)*w_split], root = cores[i*w_split]
```

Every dependent quantity (CB page counts, loop trip counts, RT args, Mcast group size) is computed from these values — never restated as a literal. Program cache key must include `(regime, core_w_tiles, block_rows, w_split, layout, dtypes, has_gamma, gamma_is_rm)`.

Worked examples (BH 110 cores, bf16 TILE, γ RM bf16): `(1,1,32,64)` → LOCAL, 1 core; `(2,4,128,512)` R=32, Wt=16 → `k_occ = min(3, 2) = 2` → W_SPLIT k=2, 32 lines; `(1,1,32,8192)` R=1, Wt=256 → k=32, cw=8; `(1,1,32,32768)` Wt=1024 → k=110→cw=10→w_split=103; `(8192,7168)` R=256 → `k_fit = 2` (x depth 2) → W_SPLIT k=2, 55 lines, block_rows=1.

### Regime-pinned tests (required; the predicate is grid-dependent)

| Regime | Pinning shape | Why it pins on any grid ≥ 2 cores |
|--------|---------------|-----------------------------------|
| LOCAL | `(4, 8, 32, 256)` | Wt = 8 < 2·min_core_w_tiles → k_occ = 1; a row fits → k_fit = 1 |
| W_SPLIT (occupancy) | `(1, 1, 32, 4096)` | R = 1 < C and Wt = 128 ≥ 16 → k_occ ≥ 2 |
| W_SPLIT (residency) | `(1, 1, 32, 16384)` | Wt = 512: one bf16 row at depth 2 is 2 MiB > budget → k_fit > 1 |
| W_SPLIT, ragged last slice + partial W tile | `(1, 1, 32, 4095)`, `(1, 1, 32, 4064)` (Wt = 127, prime) | last core of the line has a narrower valid slice; 4095 also exercises the W-pad zeroing on the last core only |

## Circular Buffers

Notation: `br = block_rows`, `cw = core_w_tiles`, `k = w_split`, `T_in` = input-dtype tile bytes, `T_out = T_in`, `T_f = 4096` (Float32 tile), `T_g` = γ-dtype tile bytes, `es` = element bytes. Full ledger with axis accounting and sharing: `l1_ledger.md`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_rm` (RM only) | 0 | `T_in` | `x_rm_depth · cw` | streams R one tile-row at a time; spans W-slice | input dtype | reader | compute | per tile-row |
| `cb_x` | 1 | `T_in` | TILE: `x_block_depth · br · cw`; RM: `br · cw` (exactly one block — required by the in-place normalize) | spans the block (x is resident across sumsq → normalize; RM: then holds y until untilize) | input dtype (= output dtype) | TILE: reader / RM: compute (tilize, and the in-place normalize) | compute | per block |
| `cb_gamma_rm` (RM γ only) | 2 | `cw · 32 · es_γ` (one row page) | 1 | one stick slice | γ dtype | writer | compute | prologue |
| `cb_gamma` (γ only) | 3 | `T_g` | `cw` | spans W-slice, constant over R | γ dtype | TILE γ: writer / RM γ: compute (tilize) | compute | whole kernel |
| `cb_sumsq_raw` | 4 | `T_f` | `br` | one raw tile per tile-row of the block | Float32 | compute | compute | per block (sumsq → collapse) |
| `cb_partial` (W_SPLIT) | 5 | `T_f` | `br` | one collapsed tile per tile-row | Float32 | compute | writer | per block (collapse → gather) |
| `cb_gathered` (W_SPLIT; used on roots, allocated on all) | 6 | `T_f` | `k · br` | the line's partials, slot `r·k + line_pos`; capacity = exactly one block (fixed landing address) | Float32 | reader (root; data lands via remote NoC writes) | compute (root) | per block |
| `cb_rinv_col` (LOCAL; W_SPLIT root) | 7 | `T_f` | `br` | column-0 Σx² per tile-row | Float32 | compute | compute | per block (collapse/combine → finalize) |
| `cb_rinv_bcast` (W_SPLIT root; allocated on all) | 8 | `T_f` | `br` | finalized rinv, mcast source | Float32 | compute | writer | per block |
| `cb_rinv` | 9 | `T_f` | `br` | full-tile rinv per tile-row; capacity = exactly one block (fixed mcast landing address) | Float32 | LOCAL: compute / W_SPLIT: reader (mcast receive) | compute | per block (finalize/receive → normalize) |
| `cb_y` (TILE only) | 10 | `T_out` | `y_depth · y_chunk_tiles` | streams W (window); spans nothing | output dtype | compute | writer | per chunk |
| `cb_out_rm` (RM only) | 11 | `T_out` | `out_rm_depth · cw` | streams R one tile-row at a time | output dtype | compute (untilize) | writer | per tile-row |

All CBs of a program variant are allocated on **all** participating cores with identical descriptors (uniform L1 addresses — required by the remote writes into `cb_gathered` and `cb_rinv`).

CB sync (push = wait, per CB per block): `cb_x` reader (TILE) / tilize (RM) pushes `br·cw` = sumsq waits `br·cw` (cumulatively); TILE: normalize pops `br·cw` AtEnd; `cb_x_rm` push `cw` per tile-row = tilize waits `cw` per tile-row; `cb_sumsq_raw` push 1 per tile-row (`br`) = collapse waits `br`; `cb_partial` push `br` = writer waits `br`; `cb_gathered` push `k·br` = combine waits `k·br`; `cb_rinv_col` push `br` = finalize waits `br`; `cb_rinv_bcast` push `br` = writer waits `br`; `cb_rinv` push `br` = normalize waits `br` (Upfront) and pops `br` (AtEnd); `cb_y` (TILE) push 1 per tile = writer waits ≤ `y_chunk_tiles`, pops 1 per page; RM in-place `cb_x`: normalize pops 1 / pushes 1 per tile (`br·cw` per block) = untilize waits `cw` per tile-row; `cb_out_rm` push `cw` per tile-row = writer waits `cw`; `cb_gamma` push `cw` once = normalize waits `cw` once (before the first block), popped once at kernel end; `cb_gamma_rm` push 1 = tilize waits 1.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `tilize_gamma` (RM γ) | 1 × `cw` (1 valid row) | yes | `cb_gamma_rm`, 1 page, popped | `cb_gamma`, `cw` | cb_gamma resident until kernel end |
| 1 | `tilize_x_block` (RM) | `br` × `cw` | yes | `cb_x_rm`, `cw` per tile-row, popped | `cb_x`, `br·cw` | — |
| 2 | `sumsq_block` | grid(`br`, `cw`) | yes (eltwise_chain) | `cb_x`, waited cumulatively `br·cw`, **not popped** | `cb_sumsq_raw`, `br` (1 per tile-row) | cb_x resident |
| 3 | `collapse_block` | `br` × 1 | yes (reduce) | `cb_sumsq_raw`, `br`, popped | LOCAL `cb_rinv_col` / W_SPLIT `cb_partial`, `br` | — |
| 4 | `combine_block` (W_SPLIT root) | `br` × `k` | yes (reduce) | `cb_gathered`, `k·br`, popped | `cb_rinv_col`, `br` | — |
| 5 | `finalize_rinv_block` (LOCAL / root) | `br` tiles | yes (eltwise_chain) | `cb_rinv_col`, `br`, popped | LOCAL `cb_rinv` / root `cb_rinv_bcast`, `br` | — |
| 6 | `normalize_block` | grid(`br`, `cw`) | yes (eltwise_chain) | `cb_x` (TILE: `br·cw`, no wait, **pop AtEnd**; RM: per-tile wait/pop, in place), `cb_gamma` (`cw`, caller-waited once, no pop), `cb_rinv` (`br`, Upfront wait, AtEnd pop) | TILE: `cb_y`, 1 per tile; RM: `cb_x`, 1 per tile (in place) | TILE: cb_x slot freed for block b+2; RM: cb_x holds y |
| 7 | `untilize_y_block` (RM) | `br` × `cw` | yes | `cb_x` (holding y), `cw` per tile-row, popped | `cb_out_rm`, `cw` per tile-row | cb_x empty, ready for the next tilize |
| — | `load_gamma`, `load_x_block`, `gather_partials_block`, `broadcast_rinv_block`, `store_y_block` | see Block schedule | dataflow (raw NoC + Mcast helper) | — | — | — |

Compute boot: exactly one `compute_kernel_hw_startup(...)` as the first statement of the compute kernel (RM: `(cb_x_rm, cb_x)`; TILE: `(cb_x, cb_x, cb_sumsq_raw)`); every helper keeps its data-format reconfig enabled (defaults) because consecutive phases change formats (input dtype ↔ Float32 ↔ γ dtype ↔ output dtype). Never re-call hw_startup mid-kernel (`reduce_helpers_compute.hpp:31-35`).

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `tilize_gamma` | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:187-197` | `<core_w_tiles, cb_gamma_rm, cb_gamma>(1, /*total_input_pages=*/1)` — asymmetric row-page mode (`tilize_helpers.hpp:130-136`, `tilize_helpers_dataflow.hpp:20-24`: fewer than 32 row pages is a supported contract) | cb_gamma_rm | cb_gamma | `core_w_tiles` |
| `tilize_x_block` | helper | `compute_kernel_lib::tilize` | `tilize_helpers.hpp:187-197` | `<core_w_tiles, cb_x_rm, cb_x>(rows_this_block)`; `Fp32Mode::Fast` (TF32 in DEST is irrelevant: every consumer is FPU, `tilize_helpers.hpp:103-110`) | cb_x_rm | cb_x | `core_w_tiles` (CT), `block_rows` (num_blocks) |
| `sumsq_block` | helper | `compute_kernel_lib::eltwise_chain` with `BinaryFpu<BinaryFpuOp::Mul, A, B, Dst::D0, DestAccumulation::PerRow>` + `PackTile` | `eltwise/api/chain.hpp:532-533`, `BinaryFpu` `:485-494`, `DestAccumulation` `:283-287`, same-CB dedup `eltwise/core/chain.inl:1100,1135`, Cumulative legality `chain.inl:35` | shape `IterationShape::grid(rows_this_block, core_w_tiles)`; A = B = `input(cb_x, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block)`; `PackTile<output(cb_sumsq_raw, ReservePolicy::PerOuter, PushPolicy::PerOuter, DataFormatReconfig::Enabled, TileAddressing::Direct, DestAccumulation::PerRow)>` → one raw Σx² tile per tile-row | cb_x | cb_sumsq_raw | grid H = `block_rows`, grid W = `core_w_tiles` |
| `collapse_block` | helper | `compute_kernel_lib::reduce` | `reduce_helpers_compute.hpp:620-641`; AccumulateViaAdd `:133-158`; scaler not read on the aligned path `reduce_helpers_compute.inl:320-336` | `<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sumsq_raw, cb_x /*scaler slot: never waited or read on this path; must only differ from in/out (inl:919-920)*/, out, ReduceInputPolicy::BulkWaitBulkPop, INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::AccumulateViaAdd, ReduceWithinTile::Collapse>(ReduceInputBlockShape::of(rows_this_block, 1))`, no post-op; out = `cb_rinv_col` (LOCAL) / `cb_partial` (W_SPLIT) | cb_sumsq_raw | cb_rinv_col / cb_partial | `block_rows` |
| `combine_block` | helper | `compute_kernel_lib::reduce` | `reduce_helpers_compute.hpp:620-641`; `ReduceWithinTile::Skip` `:162-195`; combine idiom `:197-236` | `<SUM, REDUCE_ROW, cb_gathered, cb_x /*unused scaler slot*/, cb_rinv_col, BulkWaitBulkPop, INPUT_AND_OUTPUT, Fast, AccumulateViaAdd, ReduceWithinTile::Skip>(ReduceInputBlockShape::of(rows_this_block, w_split))` | cb_gathered | cb_rinv_col | `block_rows`, `w_split` |
| `finalize_rinv_block` | helper (2 chains) + one raw-SFPU chain element | `eltwise_chain` (1) `CopyTile<cb_rinv_col>` → `RinvColumn{inv_w, eps}` → `PackTile<cb_rinv_col>` (in place), (2) `UnaryBcast<Col, cb_rinv_col>` → `PackTile<out>` | `rms_norm_compute.cpp` (`RinvColumn`: `UnaryOp` CRTP element calling `_llk_math_eltwise_unary_sfpu_params_(…, VectorMode::C)` with a fused `v·(1/W)+ε → _calculate_sqrt_body_<APPROX, RECIPROCAL, fast>` body and an even-parity DEST stride) | Refinement 3: the stock `MulUnary` / `AddUnary` / `Rsqrt` hardcode `VectorMode::RC` (`binop_with_scalar.h`, `rsqrt.h:44-52`), i.e. 3 × 32 vectors on a tile whose only meaningful lanes are column 0; the scoped element runs 8. Order is still `(Σ·(1/W)) + ε` then rsqrt; out = `cb_rinv` (LOCAL) / `cb_rinv_bcast` (root) | cb_rinv_col | cb_rinv / cb_rinv_bcast | `block_rows` |
| `normalize_block` (γ) | helper | `eltwise_chain` with `BinaryFpu<Mul, x, γ(Row bcast)>` + `DestReuseBinary<Mul, rinv, DEST_TO_SRCA>` + `PackTile` | `chain.hpp:485-494`, `:499-501` (`DestReuseType` `:450-453`), `BroadcastDim::Row` `:305-310`, `InputTileMapping` `:232-237`, (None, AtEnd) legality `chain.inl:31` | `IterationShape::grid(rows_this_block, core_w_tiles)`; A = TILE `input(cb_x, WaitPolicy::None, PopPolicy::AtEnd, Block)` / RM `input(cb_x, WaitPolicy::PerTile, PopPolicy::PerTile)` (in place); B = `input(cb_gamma, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row)`; then `DestReuseBinary<Mul, input(cb_rinv, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col), DestReuseType::DEST_TO_SRCA, Dst::D0>`; `PackTile<output(TILE: cb_y / RM: cb_x, ReservePolicy::PerTile, PushPolicy::PerTile)>`. If the chain's pack-collision check rejects input == output for this element mix, record the gap and fall back to an RM `cb_y` of `br·cw` pages (re-run the ledger) | cb_x, cb_gamma, cb_rinv | cb_y (TILE) / cb_x (RM) | `block_rows`, `core_w_tiles` |
| `normalize_block` (no γ) | helper | `eltwise_chain` with `BinaryFpu<Mul, x, rinv>` + `PackTile` | `chain.hpp:485-494` | A = as above; B = `input(cb_rinv, BroadcastDim::None, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col)` (rinv is already a full tile); output as the γ variant | cb_x, cb_rinv | cb_y (TILE) / cb_x (RM) | `block_rows`, `core_w_tiles` |
| `untilize_y_block` | helper | `compute_kernel_lib::untilize` | `untilize_helpers.hpp:145-154` | `<core_w_tiles, cb_x, cb_out_rm>(rows_this_block)` | cb_x (holding y) | cb_out_rm | `core_w_tiles` (CT), `block_rows` |
| `load_x_block` (TILE) | raw_api | `TensorAccessor` + `noc_async_read_tile` / `noc_async_read_barrier`; W-pad zeroing via `fill_l1_range`; one-time tail pre-zero via `noc.async_write_zeros` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:46-55, 70-90`; `tech_reports/tensor_accessor/tensor_accessor.md` | chunked reads, push per chunk | DRAM | cb_x | `read_chunk_tiles`, `block_rows`, `core_w_tiles` |
| `load_x_block` (RM) | raw_api | `TensorAccessor::get_noc_addr(stick, byte_offset)` + `noc_async_read` | — | one read of `core_w_valid_elems·es` bytes per present stick at stride `core_w_tiles·32·es`; push `core_w_tiles` pages per tile-row | DRAM | cb_x_rm | `core_w_tiles` |
| `store_y_block` (RM) | raw_api | `noc_async_write` per present stick | — | `core_w_valid_elems·es` bytes from stride `core_w_tiles·32·es` | cb_out_rm | DRAM | `core_w_tiles` |
| `load_gamma`, `store_y_block` (TILE) | raw_api | `TensorAccessor` + `noc_async_read` / `noc_async_write_tile` | — | γ TILE: two 16-element reads per tile (row 0 of faces 0, 1) | DRAM / cb_y | cb_gamma(_rm) / DRAM | `core_w_tiles`, `y_chunk_tiles` |
| `gather_partials_block` | raw_api | `noc_async_write` + `noc_async_write_barrier` + `noc_semaphore_inc` / `noc_semaphore_wait` / `noc_semaphore_set` | `METALIUM_GUIDE.md` (semaphores) | per tile r: dst `get_noc_addr(root_x, root_y, cb_gathered_base + (r·w_split + line_pos)·4096)` | cb_partial | cb_gathered | `block_rows`, `w_split` |
| `broadcast_rinv_block` | helper | `ttnn.Mcast` host + `McastArgs<…>().sender(noc).send(src, dst, bytes)` / `.receiver(noc).receive(round)` | host `ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast.hpp:86-129`; device `mcast/kernel/mcast_args.hpp:15-53`; Python binding `ttnn/cpp/ttnn-nanobind/mcast_host.cpp:38-128` | `Mcast(device, McastConfig(noc=<writer NoC>), participating_cores, w_split, McastFixedSenderConfig(sender_index=0), McastCoreOrder.RowMajor)`; `attach(descriptor, "rinv_mcast", [reader, writer], first_semaphore_id=1)` | cb_rinv_bcast | cb_rinv | `block_rows` (payload) |

**Helpers considered and rejected (raw_api rows):**

| Raw entry | Candidate helper | File:line of the mismatch | Reason |
|-----------|------------------|---------------------------|--------|
| `load_x_block` (RM) | `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.inl:92-93` | derives the L1 stick stride and the pushed page count from `row_bytes` (`padded_row_bytes = round_up(row_bytes, tile_row_bytes)`). The last core of a W_SPLIT line has `core_w_valid_tiles < core_w_tiles`, so the helper would stride and push its *valid* width while `tilize` (CT width) and the CB ring need the *nominal* width. Closing the gap: a `stride_bytes` / `push_pages` override on the helper |
| `store_y_block` (RM) | `dataflow_kernel_lib::write_sticks_after_untilize` | `tilize_helpers_dataflow.inl:208-209` | same derivation on the write side |
| `load_x_block` / `store_y_block` (TILE), `load_gamma` | — | — | no kernel_lib dataflow helper reads/writes tile ranges of an interleaved tensor; these are the standard TensorAccessor loops |
| `gather_partials_block` | `ttnn.Mcast` (`mcast/host/mcast.hpp:121-129`) | `mcast.hpp:119-120` ("Partitions the ordered receiver cores into equal consecutive groups" — one sender → many receivers) | the gather is many → one (fan-in); the Mcast channel models only fan-out. Closing the gap: a gather/collect channel in `kernel_lib/mcast` or `kernel_lib/ccl` |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| sumsq_block | BinaryFpu Mul (+DEST acc PerRow) | `cb_x`: All (W-pad lanes zeroed) | `cb_x` (same tile): All | None |
| collapse_block | reduce SUM REDUCE_ROW (AccumulateViaAdd, Collapse) | `cb_sumsq_raw`: All | — | out: Col0 |
| combine_block | reduce SUM REDUCE_ROW (AccumulateViaAdd, Skip) | `cb_gathered`: Col0 (REDUCE_ROW outputs) | — | out: Col0 |
| finalize_rinv_block | UnaryBcast | `cb_rinv_col`: Col0 | — | Col → out All |
| normalize_block (γ) | BinaryFpu Mul | `cb_x`: All | `cb_gamma`: Row0 (1-D `[W]` γ) | Row |
| normalize_block (γ) | DestReuseBinary Mul | DEST: All | `cb_rinv`: All (full broadcast tile) | None |
| normalize_block (no γ) | BinaryFpu Mul | `cb_x`: All | `cb_rinv`: All | None |
| untilize_y_block (RM) | untilize | `cb_x` (y): All | — | — |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Padding leaks into Σx² | golden pad-poison cases fill TILE padding with 1000.0 (→ 10⁶ after squaring); RM pad bytes in L1 are uninitialized | padding is made **zero in the data** before the square: TILE — zero lanes `[w_valid_in_last_tile, 32)` of the last W tile after every read; RM — cb_x_rm zero-filled once, stick reads never touch pad bytes; nominal tail tiles pre-zeroed once. If an RM read of a non-16-byte-multiple stick is observed to stomp pad bytes, re-zero per tile-row instead (same code as the TILE path) |
| Divisor uses padded W | `mean` must be over valid elements only | `inv_w = 1/W` with logical W, applied once in finalize (never derived from tile geometry — do **not** use `PoolType::AVG`) |
| H padding / absent sticks | padded rows (TILE) or stale sticks (RM tail) hold garbage, possibly non-finite | every op is row-independent (FPU mul, per-row DEST accumulation, row collapse, Col/Row broadcasts), so garbage rows only produce garbage rows, which are never read back (TILE pad rows are not part of the logical output; RM writer skips absent sticks) |
| W_SPLIT landing-slot races | `cb_gathered` and `cb_rinv` are written by other cores | capacity exactly one block (constant landing address); causal chain partial(b+1) ⇒ normalize(b) consumed rinv(b) ⇒ root popped gathered(b). Do **not** reorder compute (sumsq lookahead) without adding a ready semaphore (lamp L6). Data write `noc_async_write_barrier()` **before** the semaphore increment |
| RM in-place normalize on `cb_x` | two phases (normalize, untilize) touch the same bytes; an in-place phase with `None` policies would move no credit and untilize could read pre-normalize data (`l1-footprint-discipline.md` Rule 3.2) | the in-place normalize uses **real** per-tile credit (wait/pop 1, reserve/push 1) and `cb_x` capacity is exactly one block, so pack lands in the slot unpack just freed and untilize waits on normalize's pushes. `cb_x` RM has compute as its only producer and consumer |
| Uniform CB addresses | remote writes target the sender's own CB address | identical CB descriptors (same set, same sizes, same order) on every participating core, including root-only CBs |
| `compute_kernel_hw_startup` repeated | multi-phase kernel with 5+ format changes | call once; rely on helper reconfig (all defaults enabled). Mixed γ dtype (fp32 γ with bf16 x, or vice versa) is a srcA/srcB format pair the chain reconfigures per operand |
| Scaler slot for AccumulateViaAdd | template requires a scaler CB id that differs from input/output (`reduce_helpers_compute.inl:919-920`) though the aligned path never waits on or reads it | pass `cb_x` as the scaler slot id; never push a scaler tile; no reduce-scaler CB exists in this op (so no bf16 scaler packing applies) |
| fp32 precision | float32 input must not be downcast to 16 bit anywhere | every intermediate CB is Float32 (fp32 DEST); FPU operands are TF32 by hardware (not a downcast to bf16); rsqrt exact unless the caller set approx mode |
| Root CBs exist on non-roots | uniform layout forces `cb_gathered` (`k·br` tiles) onto every core | counted in the L1 predicate; it bounds `block_rows` at large `w_split` |
| Program cache | CT args (`core_w_tiles`, `block_rows`, regime, flags) vary with shape | cache key includes every CT arg and the regime; RT args (addresses, ranges, ε, 1/W) via runtime-arg override |
| Grid-dependent regime | W_SPLIT triggers by occupancy on some grids only | regime-pinned test shapes chosen to pin on any grid (Work Distribution) |
| Output is TILE with non-aligned shape | writer must write every tile of the padded output including padded rows | TILE writer writes all `core_w_valid_tiles` tiles of every tile-row in its range (padding rows included) |

### Structural impossibilities

None beyond those already in `feature_spec.py` INVALID.

# Operation Design: rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (row-wise reduction + broadcast eltwise, fused; one device program per invocation) |
| Goal | Normalize every row of the input along its last dimension by its root-mean-square, optionally scaled by a per-column weight `gamma`, in one native dispatch; multi-core from Phase 0 with the reduced axis splittable across cores. |
| Math | `output[..., r, w] = input[..., r, w] * rsqrt( (1/W) * Σ_w input[..., r, w]^2 + epsilon ) * gamma[w]` (`gamma` term omitted when absent) |
| Mode | Derivative (composes existing kernel-lib helpers; cross-core combine via `mcast_pipe`) |
| References | `eval/golden_tests/rms_norm/feature_spec.py` (TARGET / INPUTS / INVALID / LOOSE_CASES — authoritative); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `.claude/references/precision_convention.md`; `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`; `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp`; `ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp`; `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `untilize_helpers.hpp`, `tilize_helpers_dataflow.hpp`; `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`, `host/mcast_host.hpp`; `ttnn/ttnn/operations/examples/master.md` (catalog); `l1_ledger.md` (beside this file) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2; `shape[-1] % 32 == 0`, `shape[-2] % 32 == 0`; dtype ∈ {bfloat16, float32}; layout ∈ {TILE, ROW_MAJOR}; memory ∈ {INTERLEAVED (DRAM/L1), WIDTH_SHARDED (L1, TILE only)} | — | host |
| `gamma` | `Optional[ttnn.Tensor]` (keyword-only) | no | `shape[-1] == input.shape[-1]`, all other dims 1; layout ∈ {ROW_MAJOR `(1,1,1,W)`, TILE `(1,1,1,W)` padded to one tile-row}; dtype ∈ {bfloat16, float32}; interleaved | `None` (→ `gamma_mode="no_gamma"`, `gamma_dtype=gamma_layout="none"`) | host |
| `epsilon` | `float` (keyword-only) | no | finite, ≥ 0 | `1e-6` | RT (fp32 bit pattern `eps_bits`) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` (keyword-only) | no | any `math_fidelity`, any `math_approx_mode`; Phase 0 requires `fp32_dest_acc_en == True` (`False` is a refinement; `float32 + False` is natively rejected forever) | `None` → `default_compute_kernel_config()` (HiFi4, fp32_dest_acc_en=True, math_approx_mode=False) — the single exported factory | host → `config=compute_kernel_config` on the compute `KernelDescriptor` |
| `memory_config` | `Optional[ttnn.MemoryConfig]` (keyword-only) | no | INTERLEAVED input: any interleaved config; WIDTH_SHARDED input: must equal the input's memory config (output inherits the shard spec) | `None` → input's memory config when sharded, else `DRAM_MEMORY_CONFIG` | host |

Index axes: none (the reduction dim is always `-1` by definition of the op; no `dim` parameter).

The golden harness (`eval/golden_tests/rms_norm/helpers.py::run_rms_norm`) passes `memory_config=input.memory_config()` for sharded cells — the entry point MUST accept it. `axes.classify_call` also accepts a `program_config=None` keyword for signature compatibility; it is accepted and ignored.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(..., H, W)`, rank 2/3/4 (all leading dims fold into tile-rows: `tensor_row_tiles = prod(shape[:-1]) / 32`, exact because `H % 32 == 0`); `tensor_w_tiles = W / 32` |
| Dtype | bfloat16, float32 |
| Layout | TILE or ROW_MAJOR (native; no host-side layout transform) |
| Memory | INTERLEAVED (DRAM or L1) or WIDTH_SHARDED (L1; TILE layout only; shard = `[prod(shape[:-1]), shard_w]`, `shard_w % 32 == 0`; core→slice map = `ttnn.corerange_to_cores(shard_spec.grid, None, row_wise=True)` order) |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | identical to input |
| Layout | identical to input |
| Memory | `memory_config` (see Parameters); WIDTH_SHARDED output has the input's shard spec |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Everything below this section is a realization of the decisions here.

### Axes

The op's tile-index space is honestly 2-D: all leading dims and `H` fold into one **row** axis (RMSNorm treats every row identically and rows are contiguous in both layouts), and `W` is the **w** axis. A block is a `block_rows × block_w_tiles` rectangle of tiles that one core drives through the whole phase sequence in one pass. `gamma` is a 1-D operand along `w`.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `row` — tile-rows, `tensor_row_tiles = prod(shape[:-1])/32` | **independent** — every row's rstd depends only on that row | `block_rows` (B) | `min(core_row_tiles, block_rows_max_l1)` — the coarsest block that fits L1 within the core's assignment (Host derivation §H3); it is 1 only where the closed-form L1 bound forces it (wide W) | host `derive_blocking()` → RT arg `block_rows` (+ `last_block_rows`, `num_blocks_this_core`) | tile-rows split contiguously over `num_row_groups` groups of cores (each group = one W-split rectangle); Phase 0 already spreads rows across the grid | knob-turn (B, group count) |
| `w` — tile-columns, `tensor_w_tiles = W/32`; **reduced** | **dependent** — Σ_w x² spans the whole row | `block_w_tiles` ≡ `core_w_tiles` (Wc) — a block always spans the core's **entire** W slice (never sub-chunked: any W sub-chunk would force a second read of x for the normalize pass — the rejected regime R5) | `core_w_tiles = q + (c < r)` for W-split index `c`, `q = Wt // Cw`, `r = Wt % Cw`; `core_w_tiles_max = q + (r > 0)` | host → CT arg `core_w_tiles` per core group (≤ 2 distinct values: tilize/untilize need it compile-time), RT arg `w_tile_start` | W slice `[w_tile_start, w_tile_start + core_w_tiles)` per core; the `num_w_splits` (Cw) cores of a group combine partials via root-gather + mcast (built in Phase 0) | knob-turn on Cw (the combine exists) |
| `w_split` — the Cw per-row partial sums a group produces (the cross-core face of `w`) | **dependent** — one rstd per row needs all Cw partials | `num_partials` (Np) — gather extent = Np slots per row (Np = the group's ACTIVE W slices; `num_w_splits` = Cw ≥ Np is only the multicast rectangle's core count — a non-rectangular shard grid has Cw > Np, and summing Cw slots reads never-written L1) | Host derivation §H2: `max(residency floor, occupancy term)` shaped into an `a × b` rectangle | host → RT args `num_w_splits`, `w_split_index`, `is_root`, `num_partials_expected` | root = first core of the group rectangle; **communication role only** — the root's combine work is Cw tile-adds + one SFPU pass per row (µs-scale), which is stated, not hidden; all other stages are spread over the whole group | built |
| `gamma.w` — the weight's only axis | **reuse-shared** — constant along `row`, so every row-group re-reads the same slice | resident slice extent = `core_w_tiles` | whole per-core slice resident for the kernel lifetime | same `core_w_tiles` | each core reads its own slice once (stepping stone); broadcast along the grid column is regime R4 | scheme-change (R4, deferred) |
| RM stick sub-axis (32 sticks per tile-row) | not an op axis — a layout quantum of `row` | folded into `block_rows` (32·B sticks) | — | — | — | — |

No axis is dropped: a **W-sub-block extent of 1 tile is never used** (the block always spans the core's slice), and `block_rows = 1` appears only as the output of the L1 bound, with its reason recorded in §H3.

Intermediate results and their axes (re-running the table on stage outputs): `sumsq_partial` (B tiles: spans `row`, collapsed `w` → same `row` assignment as its block); `collapsed partial` (B tiles, column-0 valid); `gather` (B × Cw tiles: spans `row` and `w_split`, lives on the root only in use, allocated uniformly); `rstd` (B tiles, column-0 valid, identical on every core of the group after the mcast); `normed` (B × Wc, spans both block axes). Every one of them inherits the block's core assignment; none introduces a new split.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_tiles` (TILE interleaved) | `depth_x` (blocks) | 2 | reader prefetches block k+1 while compute runs block k (catalog `double_buffer`) |
| `cb_x_tiles` (RM builds) | fixed 1 block | 1 | producer is compute (tilize) — double-buffering a compute→compute CB buys nothing; overlap comes from `cb_x_sticks` |
| `cb_x_sticks` (RM input) | `depth_x_sticks_rows` (tile-rows of 32 sticks) | `2 · block_rows` (two whole blocks of sticks) | full DRAM-read/compute overlap for the RM path |
| `cb_output_tiles` (TILE interleaved) | `depth_out` (blocks) | 2 | writer drains block k while compute produces k+1 |
| `cb_output_tiles` (RM builds) | fixed 1 block | 1 | compute→compute (untilize input); must hold the whole block (sequential helpers) |
| `cb_out_sticks` (RM output) | `depth_out_sticks_rows` (tile-rows) | 2 | writer overlaps the untilize |
| `cb_gather`, `cb_rstd` | rounds in flight | **1 (fixed — mechanism cap, not a knob)** | see Mechanism caps: remote writers rely on slot addresses that never move |
| `cb_sumsq_partial`, `cb_partial_collapsed`, `cb_rstd_handoff`, `cb_normed`, `cb_gamma_tiles`, `cb_scaler` | none | 1 block / 1 slice / 1 tile | sequential-helper intermediates or constants |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Remote NoC writes into `cb_gather` (peers → root) and the rstd mcast into `cb_rstd` | capacity of `cb_gather` = exactly `block_rows · num_partials` pages and of `cb_rstd` = exactly `block_rows` pages — **one round** | allocate exactly one round; every round's `cb_reserve_back` returns the CB base again, so slot `r·Cw + c` is at a fixed L1 address identical on all cores (uniform CB descriptors over the program's core range) | a 2-round ring makes the write pointer alternate between halves; senders compute the slot from the base → half the rounds land in the wrong half → silently wrong rstd |
| Ragged last block | `last_block_rows ≤ block_rows` | all cores of a group use the same runtime `rows` for the last block (they own the same row range); gather/rstd pushes and waits use `rows·Cw` / `rows`; slot addresses are still base-relative | mismatched push/wait counts → hang |
| Group rectangle for `Mcast2D` | `num_w_splits = a · b`, `a ≤ grid_x`, `b ≤ grid_y`, `a·b ≤ tensor_w_tiles` | §H2 shapes Cw into `a = min(Cw_req, grid_x)`, `b = ceil(Cw_req / a)`; asserted `a·b ≤ Wt` (provably holds while `core_w_tiles_max ≥ 2`; asserted anyway) | mcast bounding box would include cores not in the program → corrupts a stranger's L1 |
| WIDTH_SHARDED grid that is not a rectangle (`auto_shard_config` lays 64 shards on a 13-wide grid as 4 rows + 12) | program core range = **bounding box** of `shard_spec.grid`; non-member cores are **passive participants** with `core_w_tiles = 0` | passive cores skip every tensor-CB op and every partial send, but run the writer's mcast `receive()` (they are inside the rect and must ack); `num_partials_expected = active_cores − 1` | mcasting into a core that is not running the program is UB |
| `tilize` / `untilize` `block_width_tiles` is a template parameter | `core_w_tiles` per core | one kernel descriptor per distinct `core_w_tiles` value (≤ 2 groups per program, like `split_work_to_cores`); CB descriptors stay uniform (sized with `core_w_tiles_max`) | a uniform template width on a ragged core pushes more tile pages than the writer pops → hang |
| `DEST_AUTO_LIMIT = 4` (fp32 dest, half sync) — `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:328-342` | none on the block: chains use `Dst::D0` only; the reduce uses DST[0]; matmul-style subblocking does not apply | — (helpers honor it) | — |
| RM stick chunk page = `core_w_tiles · 32 · elem_size` bytes | must be L1-aligned (16 B) | always true (≥ 64 B and a multiple of 64) | — |
| `1/W` and `epsilon` as fp32 bit patterns in RT args | `W ≤ 2^24` exactly representable count | assert on host | — |
| Sharded per-core residency | `block_rows` ≤ `block_rows_max_l1` with the shard buffers already resident | blocks walk the resident shard with `TileOffset::Strided` (`StridedTileRange{block_idx · block_rows · shard_w_tiles, shard_w_tiles}`, chain.hpp:277-282) over `rows × core_w_tiles` valid tiles | CB OOM at program creation |
| Ragged (padded) last WIDTH shard | `(num_partials − 1)·shard_w_tiles < tensor_w_tiles ≤ num_partials·shard_w_tiles` | the last shard-grid core gets `core_w_tiles = tensor_w_tiles − i·shard_w_tiles` (its own compute build); the strided walk never touches the padded columns, the padded output columns are never written | padding is summed into Σx² → a pure per-row scale error (PCC stays high) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 `row_split`** — every core owns whole rows of its slice; `num_w_splits = 1`, no cross-core traffic | **built** | `memory_layout == INTERLEAVED` **and** `derive_w_splits(...) == 1` (§H2) | `block_rows × core_w_tiles` with `core_w_tiles = tensor_w_tiles` | **minimum** against the DRAM boundary: x crosses once (read), output once (write); gamma crosses once **per active core** (`active_cores × Wt·T_g` bytes total) — the one term above the minimum, carried into R4 | per block: `compute_kernel_hw_startup` 0× (once per kernel), 5 chain/reduce inits + format reconfigs (A, B, C, D, E) 1× each, 1 x-block push/wait pair, 1 out-block drain; intended frequency: once per block; bigger B also lets the reader issue `B·Wc` reads per barrier |
| **R2 `w_split_root_combine`** — each row-group of `Cw` cores splits W; per-row Σx² partials gathered to a root, summed, finalized (·1/W, +ε, rsqrt), rstd multicast back | **built** (same kernels as R1; CT `NUM_W_SPLITS>1` enables the collective) | `memory_layout == INTERLEAVED` **and** `derive_w_splits(...) > 1` — reached both by the **residency floor** (`Wt > core_w_tiles_max_l1`, e.g. bf16 W ≥ ~3 K) and by the **occupancy term** (`tensor_row_tiles < num_cores`) | `block_rows × core_w_tiles`, `core_w_tiles = ceil(Wt / Cw)` | x once, out once (minimum); gamma once per active core; **added:** per block per group `(Cw−1)·B·4 KiB` unicast into the root + `B·4 KiB` mcast to `Cw−1` receivers + `Cw−1` semaphore incs + 2 handshake semaphores — for W=7168 prefill (Cw=3) ≈ 16 KiB per 448 KiB row (3.5 %) | R1's fixed costs **plus** one gather round trip + one mcast round + the root's Cw-tile combine per block; bigger B amortizes the collective over more rows (at B=1 the collective is per tile-row) |
| **R3 `width_sharded_resident`** — WIDTH_SHARDED input: the shard **is** the core assignment; x and out are zero-copy CBs on the shard buffers; combine as R2 with `Cw = |bounding box of shard grid|` | **built** (same compute/writer; reader only publishes the shard and loads gamma) | `memory_layout == WIDTH_SHARDED` (TILE layout only; RM+sharded is INVALID) | `block_rows × core_w_tiles`, `core_w_tiles = shard_w/32` (from the shard spec, per core), `block_rows = min(tensor_row_tiles, block_rows_max_l1)` — one block (the whole shard) unless the intermediates do not fit | x and out cross **no** tier (already in the core's L1); gamma once per core; cross-core as R2 | R2's list; with `block_rows = tensor_row_tiles` there is exactly one round of everything per core |
| **R4 `gamma_column_broadcast`** — one injector core per W-slice reads gamma and `Mcast1D(PerColumn)`s it to the other row-groups' cores holding the same slice | **deferred** — positive reason: gamma's extra crossings are `(num_row_groups − 1)·Wt·T_g` — ~14 % of the input bytes at 8192×7168 prefill and **zero** for every decode/`Rt < grid` shape (one row-group); R1–R3 already cover every shape it would serve. Reachable: gamma lands in the same `cb_gamma_tiles` on every core; R4 only changes who writes it (a second `mcast_pipe` family on `base_sem_id` after the rstd family) | unchanged | gamma crosses DRAM **once** per W-slice instead of once per core; adds one `Wc·T_g` mcast per grid column | unchanged |
| **R5 `w_stream_two_pass`** — one core streams W chunks: pass 1 accumulates Σx², pass 2 re-reads x to normalize | **rejected** — superseded by R2: it doubles x's DRAM crossings (+100 % of the dominant tensor) to save a ≤ 4 % combine; it is a dead end (no better regime is built on top of it) | — | — | x crosses DRAM **twice** | — |
| **R6 `all_gather_redundant_combine`** — every core of a group receives all Cw partials and computes rstd redundantly (no mcast-back hop) | **rejected** — superseded by R2: cross-core bytes scale `Cw·(Cw−1)·B·4 KiB` vs R2's `Cw·B·4 KiB` (Cw× more; 4 MiB vs 128 KiB at Cw=32), and the catalog (`tensix_all_reduce`) measures root/tree reduce fastest for tiny payloads on 1-D groups | — | — | Cw× R2's cross-core bytes | — |

Regime-pinned tests are required: R2 triggers on grid size (occupancy term) and dtype (residency floor), R3's passive-core path triggers only on grids where the shard grid is not a rectangle. See Work Distribution.

### Traffic ranking

Candidate splits, ranked by aggregate bytes per tier for the chosen block (qualitative, no timings):

| Rank | Split | DRAM crossings | Cross-core | Notes |
|------|-------|----------------|------------|-------|
| 1 | rows only (R1) | x 1×, out 1×, gamma `active_cores`× (each core a `1/Cw` slice) | none | applies only when the whole row fits (`Wt ≤ core_w_tiles_max_l1`) **and** rows fill the grid; otherwise it is either impossible or leaves the grid idle |
| 2 | rows × W with root combine (R2/R3) | x 1×, out 1× (0× when sharded), gamma `active_cores`× slices | `Cw·B·4 KiB` per block per group + handshakes — always ≤ ~4 % of the x bytes for realistic W | **the primary split**; R1 is its `Cw = 1` point, so the two are one parameterized scheme |
| 3 | rows × W with all-gather combine (R6) | as 2 | `Cw²·B·4 KiB` | rejected |
| 4 | W chunks streamed twice (R5) | x **2×** | none | rejected |
| — | gamma broadcast (R4) | gamma 1× per slice | `Wc·T_g` per column | orthogonal to the primary split; deferred |

Operand-reuse check for the chosen split: **x** varies along both `row` and `w` → no cross-core reuse. **gamma** does not vary along `row` → reuse-shared across row-groups → R4 (deferred, stepping stone built). The rstd tile is produced once per group and multicast — no re-computation.

Stall-shadow check: the only stage that waits on a peer is `broadcast_rstd_block` (non-root cores wait ~one gather + one mcast hop per block; the root waits for `Cw−1` arrivals). Work independent of the wait: block k+1's `load_x_block` (already overlapped — separate reader kernel, `depth_x = 2`) and block k+1's `sumsq_block`/`collapse_block` (independent rows, no FP-order change). Phase 0 does **not** skew the compute loop to fill the wait: it would require 2-round rings for `cb_gather`/`cb_rstd` (breaking the one-round address-stability cap) and the wait is per block on a path that is already short (decode has exactly one block per core; prefill-wide has ~7 blocks per core with a ~µs wait each against ~1 ms of work). Recorded as the `software_pipelined_blocks` perf lamp.

### Block schedule

Logical schedule (reader, compute, writer are asynchronous kernels; adjacent blocks pipeline through the depth-2 CBs):

```cpp
prepare_scaler();            // once per core: 1 bf16 tile = 1.0 (SUM), pool-type-aware overload
load_gamma_slice();          // once per core (gamma present): Wc tiles resident for the whole kernel
                             //   TILE gamma: reader reads tiles [w_tile_start, +Wc) of the padded tile-row
                             //   RM gamma:   reader reads the W-slice of the single stick (+ zero-fills 31 rows),
                             //               compute tilize_gamma_slice() -> cb_gamma_tiles
publish_x_shard();           // R3 only, once: cb_wait_front(cb_x_tiles, Rt*Wc) — the shard is already resident

for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    const uint32_t rows = (block_idx + 1 < num_blocks_this_core) ? block_rows : last_block_rows;
    load_x_block(block_idx, rows);        // reader: rows*Wc tiles -> cb_x_tiles, ONE push per block
                                          //   RM: 32*rows stick-chunks -> cb_x_sticks; compute tilize_x_block()
    sumsq_block(rows);                    // A: Σ_w x*x accumulated in DEST per tile-row -> rows fp32 tiles
    collapse_block(rows);                 // B: within-tile row-sum -> rows column-0-valid tiles
    exchange_partials_block(rows);        // Cw>1: writer unicasts rows tiles into root's cb_gather slot; root gathers
    combine_block(rows);                  // root (or Cw==1): Σ over Cw slots (no re-collapse) -> *1/W, +eps, rsqrt
    broadcast_rstd_block(rows);           // Cw>1: root mcasts rows rstd tiles -> cb_rstd on every group core (loopback)
    normalize_block(rows);                // D: x ⊙ bcast_col(rstd) -> cb_normed (gamma) | cb_output_tiles (no gamma)
    scale_block(rows);                    // E (gamma only): ⊙ bcast_row(gamma_slice) -> cb_output_tiles
    store_block(block_idx, rows);         // writer: rows*Wc tiles -> DRAM; RM: untilize_x_block() + stick-chunks; R3: none
}
```

Per operation — block shape, what stays resident, intended fixed-cost frequency:

| Operation | Acts on | Resident across it | Inits / barriers / collectives — intended frequency |
|-----------|---------|--------------------|------------------------------------------------------|
| `load_x_block` | `rows × Wc` tiles (RM: `32·rows` stick-chunks of `Wc·32·e` bytes) | — | all `rows·Wc` reads issued, **one** `noc_async_read_barrier`, **one** `cb_push_back(rows·Wc)` per block |
| `tilize_x_block` (RM) | `rows` tile-rows of `Wc` tiles | — | one `tilize<Wc>(rows, 32·rows)` call per block (init+uninit once per block) |
| `sumsq_block` | `rows × Wc`, DEST accumulates per tile-row | x stays in `cb_x_tiles` (Upfront wait, no pop) | one chain call per block: one init, one reconfig, `rows` packs |
| `collapse_block` | `rows × 1` | — | one `reduce` call per block |
| `exchange_partials_block` | `rows` tiles per sender | `cb_gather` slots (root) | per block: `Cw−1` unicast writes (one per sender, `rows` tiles each, strided by Cw slots), `Cw−1` semaphore incs, root: one wait + one reset |
| `combine_block` | `rows × Cw` slots | — | one `reduce` call per block (AccumulateViaAdd, Skip) with the post-op (1/W, ε, rsqrt) fused into the same DEST window |
| `broadcast_rstd_block` | `rows` tiles | `cb_rstd` (one round) | one `SenderPipe::send` (pre-handshake + data + flag) per block; receivers one `receive()` |
| `normalize_block` | `rows × Wc` | rstd (`Upfront`, popped at end), x popped at end | one chain call per block |
| `scale_block` | `rows × Wc` | gamma slice (never popped) | one chain call per block |
| `untilize_x_block` (RM) | `rows` tile-rows | — | one `untilize<Wc>(rows)` per block |
| `store_block` | `rows × Wc` tiles | — | writes batched, **one** `noc_async_write_barrier` per block (per tile-row for RM sticks) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **grid-synchronization** (`W_TILES_PER_CORE_TARGET = 16` sets Cw when rows under-fill the grid) | for 1-tile-row decode the combine cost grows with Cw (root ingress `Cw·4 KiB`, `Cw−1` incs) while per-core DRAM bytes shrink as `1/Cw`; the balance point is shape- and arch-dependent | targets 8 and 32; also full-grid `Cw = min(Wt, grid)` |
| **overlap** (`block_rows` = coarsest that fits) | at `Rt ≥ cores` with narrow W, one block = the whole per-core assignment → no reader/compute overlap within a core (single block) | `block_rows = ceil(core_row_tiles / 2)` (two blocks, depth 2) |
| `collapse_algorithm` (`ReduceTile` for the 1-tile-per-row collapse) | catalog `row_reduce_accumulate`: for 1–2 tiles the single FPU reduce is fastest, but the SFPU `AccumulateViaAdd` collapse is more accurate in bf16 and shares a DEST window with the finalize | `ReduceAlgorithm::AccumulateViaAdd` for `collapse_block` |
| `partial_payload` (send the whole 4 KiB collapsed tile) | only column 0 (faces 0 and 2, 2 × 1 KiB fp32) carries the sum | send faces 0 and 2 only (two 1 KiB writes per sender), halving gather/mcast bytes |
| `sfpu_scope` (`rsqrt_tile` at `VectorMode::RC`) | only column 0 of the rstd tile is read downstream (`BroadcastDim::Col`); catalog `sfpu_tile_scope` measures ~2–4× on the finalize | col-0 stride (`c_skip`) body for mul/add/rsqrt |
| `format_reconfig` (chains/reduce reconfig formats at every phase) | only a subset of boundaries actually change format (x bf16 → fp32 partials → fp32 rstd → out bf16); catalog `compute_block_size` second lever up to 1.19× | `DataFormatReconfig::Disabled` / `ReduceDataFormatReconfigMode::NONE` on boundaries where the host proves both formats equal |
| `normed_roundtrip` (`cb_normed` L1 round trip between D and E) | costs `B·Wc·4 KiB` L1 (lowers `core_w_tiles_max_l1`, raising Cw); catalog `compute_fusion` says the FPU L1 round trip is faster than DEST reuse, but the L1 saving could pay back through fewer W-splits | fuse D+E in DEST: `BinaryFpu<Mul, x, rstd Col>` → `CopyTile<gamma_rep, D1>` → SFPU binary mul (needs a row-replicated gamma tile via `unary_bcast<Row>` once) — **gated on the precision baseline** (the DEST→Src path truncates) |
| `reader_noc_placement` (`row_wise=True` group layout) | catalog `noc_placement`: column lines are ~2.9× slower on WH for interleaved reads | groups laid out along x (default here) vs y |
| `software_pipelined_blocks` (compute loop not skewed) | non-root cores idle during the gather+mcast round trip each block | skew: `sumsq/collapse(k+1)` before `normalize(k)`; needs 2-round rings in `cb_gather`/`cb_rstd` (parity slots) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| gamma slice → core (once) | TILE gamma: `Wc` tiles (gamma dtype); RM gamma: one stick chunk of `Wc·32·e_g` bytes → 32-row RM page set → tilized | reader `TensorAccessor` reads; RM: `read_sticks_for_tilize<cb_gamma_sticks, ROW>(acc, 1, chunk_bytes, 0, w_byte_offset)` then `fill_l1_range` zero of rows 1..31; compute `tilize<Wc, cb_gamma_sticks, cb_gamma_tiles>(1, 1)` | only row 0 is ever read (`BroadcastDim::Row`); the zero-fill keeps rows 1..31 defined. `cb_gamma_sticks` aliases `cb_normed`'s allocation (disjoint lifetimes) |
| x block → core (R1/R2) | TILE: `rows·Wc` tiles, tile id `= row_tile · Wt + w_tile`; RM: `32·rows` sticks, each the core's byte range `[w_byte_offset, +chunk_bytes)` | reader `TensorAccessor`; TILE: `noc_async_read` per tile, one barrier per block; RM: `read_sticks_for_tilize<cb_x_sticks, ROW>(acc, 32·rows, chunk_bytes, start_stick, w_byte_offset)` → compute `tilize<Wc, cb_x_sticks, cb_x_tiles>(rows, 32·rows)` | one push of `rows·Wc` per block (TILE); RM pushes 1 page per stick, tilize waits 32 per tile-row |
| x shard (R3) | resident TILE shard | `ttnn.cb_descriptor_from_sharded_tensor(CB_X_TILES, input)` (zero-copy); reader does `cb_reserve_back/cb_push_back(Rt·Wc)` once to publish; **never** re-read over the NoC | blocks index the resident CB with `TileOffset::Set` base `block_idx·B·Wc` |
| compute phases A→E | fp32 intermediates (`cb_sumsq_partial`, `cb_partial_collapsed`, `cb_gather`, `cb_rstd_handoff`, `cb_rstd`, `cb_normed` are `Float32` under `fp32_dest_acc_en=True`); `cb_x_tiles`/`cb_gamma_tiles` keep their tensor dtype (pure relayout, no accumulation) | kernel-lib helpers (see API Mapping) | `compute_kernel_hw_startup(cb_x_tiles, cb_scaler, cb_output_tiles)` exactly once at kernel start; helpers reconfigure formats per phase |
| partial → root (R2/R3) | `rows` fp32 tiles per sender | writer: `noc_async_write(get_read_ptr(cb_partial_collapsed) + r·4 KiB, get_noc_addr(root_x, root_y, gather_base + (r·Cw + w_split_index)·4 KiB), 4 KiB)` for `r < rows`, `noc_async_write_barrier`, `noc_semaphore_inc(root SEM_GATHER, 1)`; root: `cb_reserve_back(cb_gather, rows·Cw)`, copies its own `rows` tiles into slot `(r·Cw + root_index)` (self-aimed local copy), `Semaphore<>::wait(num_partials_expected)`, `set(0)`, `cb_push_back(cb_gather, rows·Cw)` | Tensix-to-Tensix contract: destination address = the root's `cb_gather` base (identical on all cores; one-round capacity), slot order row-major `[row][w_split_index]` = the `ReduceInputBlockShape::of(rows, Cw)` BulkWaitBulkPop order |
| rstd → group (R2/R3) | `rows` fp32 column-0-valid tiles | writer on root: `McastArgs<CT,RT>::sender(noc).send(get_read_ptr(cb_rstd_handoff), get_write_ptr(cb_rstd), rows·4 KiB)` (sender inside rect ⇒ loopback delivers the root's own copy — `mcast_pipe.inl:91-99`); receivers: `receiver(noc).receive()` then `cb_push_back(cb_rstd, rows)`; root also `cb_push_back(cb_rstd, rows)` after `send` returns | host wiring: one `ttnn.Mcast2D(device, group_rect, root, McastConfig(sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED]))` per group; identical CT args across groups ⇒ one writer binary |
| output block → DRAM (R1/R2) | TILE: `rows·Wc` tiles; RM: `32·rows` stick chunks | writer `TensorAccessor` `noc_async_write`, one barrier per block; RM: compute `untilize<Wc, cb_output_tiles, cb_out_sticks>(rows)` then writer `write_sticks_after_untilize<cb_out_sticks>(acc, 32·rows, chunk_bytes, start_stick, w_byte_offset)` | — |
| output shard (R3) | resident TILE shard | `ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output)`; compute packs with `TileOffset::Set` base; writer does nothing for output | — |

Placement axis (`memory_layout`) against this scheme: **INTERLEAVED** = R1/R2 (host derives Cw). **WIDTH_SHARDED** cuts the *dependent* axis, so its cost is the combine — which R2 already builds; adding the physical shard is *placement* (zero-copy CBs, Cw read off the shard spec), a knob-turn on the built scheme. **HEIGHT_SHARDED** (not in TARGET) would cut the independent axis: `Cw = 1`, x/out zero-copy — a placement-only knob-turn. **BLOCK_SHARDED** (not in TARGET) = R3 with `num_row_groups = shard grid y`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_rows × core_w_tiles` tiles (RM: `32·block_rows` stick chunks); a core runs `num_blocks_this_core` blocks over its contiguous tile-row range and fixed W slice |
| Grid | R1/R2: `num_row_groups × num_w_splits` cores laid out as `floor(grid_x / a) × floor(grid_y / b)` rectangles of `a × b` cores each (group `g` at rect origin `((g mod floor(grid_x/a))·a, (g div floor(grid_x/a))·b)`); `num_row_groups = min(floor(grid_x/a)·floor(grid_y/b), tensor_row_tiles)`; **R3:** the bounding box of `shard_spec.grid` (one group) |
| Per-core work | rows: `tensor_row_tiles` split over `num_row_groups` groups with `ttnn.split_work_to_cores`-style two-group ceil/floor (`rows_per_group_g1 = ceil`, `g2 = floor`); W: `core_w_tiles = q + (w_split_index < r)`, `w_tile_start = w_split_index·q + min(w_split_index, r)` (`q = Wt // Cw`, `r = Wt % Cw`); R3: `core_w_tiles = shard_w/32` for shard-grid members in `corerange_to_cores(grid, None, row_wise=True)` order, 0 for passive bbox cores |
| Remainder | `num_blocks_this_core = ceil(core_row_tiles / block_rows)`, `last_block_rows = core_row_tiles − (num_blocks_this_core − 1)·block_rows`; tile counts always `ceil`-derived per image: `tensor_row_tiles = prod(shape[:-1]) / 32` is exact only because `H % 32 == 0` is enforced — keep the formula as `prod(shape[:-2]) · ceil(H/32)` so the alignment refinement does not have to find it |

Regime selection is `derive_blocking()` (§H); it is exact and host-checkable, and the acceptance test pins one shape per regime (R1: `(2,4,128,512)`; R2 by occupancy: `(1,1,32,4096)`; R2 by residency: `(1,1,64,12288)`; R3: `(1,1,32,2048)` on 8 cores). The golden LOOSE_CASES `_WIDE` (`W = 16384, 32768, 12288`) and `_SHARDED` cells are the regime-pinned cells for the golden suite.

### H. Host derivation (single source of truth for every knob)

All constants below live once in the op's descriptor module; every CB size, loop bound and kernel arg is computed from them.

```
H0  constants (knobs):  W_TILES_PER_CORE_TARGET = 16     # grid-sync lamp
                        DEPTH_X = 2, DEPTH_OUT = 2       # TILE interleaved buffer depths
                        DEPTH_X_STICKS_BLOCKS = 2, DEPTH_OUT_STICKS_ROWS = 2   # RM depths
                        L1_MARGIN_BYTES = 64 KiB
    L1_CB_BUDGET = ttnn.get_max_worker_l1_unreserved_size() - L1_MARGIN_BYTES
                   - (R3: input_shard_bytes + output_shard_bytes)          # shards are already resident
    T_acc = tile bytes(fp32_dest_acc_en ? Float32 : Float16_b) = 4096 | 2048   # every accumulated intermediate follows the DEST width
    T_in = tile bytes(input dtype); T_out = T_in; T_g = tile bytes(gamma dtype); T_sc = 2048

H1  per-block bytes  per_block(Wc, Cw) = Wc·(x_term + HG·T_acc + out_term) + T_acc·(Cw + 2 + 2·[Cw>1])
        x_term   = DEPTH_X·T_in (TILE interleaved) | (DEPTH_X_STICKS_BLOCKS + 1)·T_in (RM) | 0 (R3)
        out_term = DEPTH_OUT·T_out (TILE)          | T_out (RM; + fixed DEPTH_OUT_STICKS_ROWS·Wc·T_out) | 0 (R3)
    fixed(Wc)   = HG·Wc·T_g + T_sc  (+ RM: DEPTH_OUT_STICKS_ROWS·Wc·T_out) (+ RM gamma: Wc·T_g — the stick block aliased on cb_normed, sized max(B·Wc·T_acc, Wc·T_g))
    block_rows_max_l1(Wc, Cw) = floor((L1_CB_BUDGET - fixed(Wc)) / per_block(Wc, Cw))

H2  num_w_splits (interleaved only; R3 takes Cw from the shard grid bbox):
        core_w_tiles_max_l1 = max { Wc ≤ Wt : block_rows_max_l1(Wc, ceil(Wt/Wc)) ≥ 1 }   (descending scan)
        Cw_res = ceil(Wt / core_w_tiles_max_l1)                       # residency floor
        Cw_occ = ceil(Wt / W_TILES_PER_CORE_TARGET) if Rt < num_cores else 1   # occupancy term
        Cw_req = clamp(max(Cw_res, Cw_occ), 1, Wt)
        a = min(Cw_req, grid_x); b = ceil(Cw_req / a); Cw = a·b;  assert a·b ≤ Wt and b ≤ grid_y
H3  block_rows = min(core_row_tiles, block_rows_max_l1(core_w_tiles_max, Cw));  assert block_rows ≥ 1
        (block_rows == 1 is therefore always the L1 bound's answer, never a default)
```

## Circular Buffers

Formats follow the DEST width (`fp32_dest_acc_en=True` ⇒ `Float32`, `False` ⇒ `Float16_b` for every accumulated intermediate — host rule `acc_dtype_for()` in the program descriptor, surfaced to the writer as the `ACC_TILE_BYTES` payload stride); `cb_x_tiles`/`cb_gamma_tiles`/`cb_output_tiles` carry tensor dtypes (relayout only — no accumulation crosses them). "Producer/Consumer" are per compile-time build; a `/` separates the two builds (`NUM_W_SPLITS == 1` / `> 1`), never two owners in one build.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_tiles` | 0 | `T_in` | TILE interleaved: `DEPTH_X · block_rows · core_w_tiles_max`; RM: `block_rows · core_w_tiles_max`; R3: `Rt · core_w_tiles` (zero-copy over the shard) | live set spans `row` (B) and `w` (Wc) — x must stay resident from `sumsq_block` to `normalize_block`; streams over blocks | input dtype | reader / compute (RM: tilize) / reader-published (R3) | compute | per block (R3: whole kernel) |
| `cb_x_sticks` (RM only) | 1 | `align_up(core_w_tiles_max·32·e_in, 16)` | `32 · DEPTH_X_STICKS_BLOCKS · block_rows` | streams over `w` at stick granularity; spans `row` for 2 blocks (the RM overlap knob) | input dtype | reader | compute | per block |
| `cb_scaler` | 2 | 2048 | 1 | constant (SUM scaler 1.0) — spans nothing | Float16_b | reader (once) | compute | whole kernel |
| `cb_sumsq_partial` | 3 | 4096 | `block_rows` | spans `row`; `w` collapsed | Float32 | compute (A) | compute (B) | per block |
| `cb_partial_collapsed` (Cw>1 only) | 4 | 4096 | `block_rows` | spans `row`; handoff to the writer for the gather send | Float32 | compute (B) | writer | per block |
| `cb_gather` | 5 | 4096 | `block_rows · num_partials` — **exactly one round** | spans `row` and `w_split` (one slot per W-split core); allocated on every core, filled on the root (Cw=1: B's direct output) | Float32 | compute (B) / writer (after `Cw−1` arrivals + own copy) | compute (C) | per block |
| `cb_rstd_handoff` (Cw>1 only) | 6 | 4096 | `block_rows` | spans `row`; mcast source on the root | Float32 | compute (C) | writer | per block |
| `cb_rstd` | 7 | 4096 | `block_rows` — **exactly one round** | spans `row`; mcast landing (identical address on all cores) | Float32 | compute (C) / writer (mcast receive or loopback) | compute (D) | per block |
| `cb_gamma_tiles` (gamma only) | 8 | `T_g` | `core_w_tiles_max` | spans `w`; resident for the whole kernel; streams over nothing | gamma dtype | reader (TILE gamma) / compute (RM gamma, tilize) | compute (E) | whole kernel |
| `cb_gamma_sticks` (RM gamma only) | 9 | `align_up(core_w_tiles_max·32·e_g, 16)` | 32 (tilize reads 32 rows; 1 pushed, 31 zero-filled) | spans `w`; **aliases** `cb_normed`'s allocation (same `CBDescriptor`, second format descriptor) — lifetimes disjoint (start-up vs. per block) | gamma dtype | reader | compute (tilize) | start-up only |
| `cb_normed` (gamma only) | 10 | 4096 | `block_rows · core_w_tiles_max` | spans `row` and `w` — sequential helpers D→E, must hold the full block | Float32 | compute (D) | compute (E) | per block |
| `cb_output_tiles` | 11 | `T_out` | TILE interleaved: `DEPTH_OUT · block_rows · core_w_tiles_max`; RM: `block_rows · core_w_tiles_max`; R3: `Rt · core_w_tiles` (zero-copy over the output shard) | spans `row` and `w`; streams over blocks | output dtype | compute (D or E) | writer / compute (RM: untilize) / none (R3 — the tensor itself) | per block |
| `cb_out_sticks` (RM only) | 12 | `T_out` (tile pages — untilize output contract) | `DEPTH_OUT_STICKS_ROWS · core_w_tiles_max` | streams over `row` one tile-row at a time; spans `w` | output dtype | compute (untilize) | writer | per block |

CB sync (push = wait, per block, `rows` = runtime extent): `cb_x_tiles` `rows·Wc` (TILE reader push / A Upfront wait; D pops AtEnd) — RM: tilize pushes `Wc` per tile-row, A waits `rows·Wc`; `cb_x_sticks` 1 per stick / 32 per tile-row; `cb_sumsq_partial` `rows`/`rows`; `cb_partial_collapsed` `rows`/`rows`; `cb_gather` `rows·Cw`/`rows·Cw`; `cb_rstd_handoff` `rows`/`rows`; `cb_rstd` `rows`/`rows`; `cb_gamma_tiles` `Wc` once / Upfront `Wc` per block (never popped); `cb_normed` `rows·Wc`/`rows·Wc`; `cb_output_tiles` `rows·Wc`/`rows·Wc`; `cb_out_sticks` `Wc` per tile-row both sides; `cb_scaler` 1/1 (never popped).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `prepare_scaler` | 1 tile | yes (`calculate_and_prepare_reduce_scaler`) | — | `cb_scaler`, 1 | resident, never popped |
| 1 | `load_gamma_slice` | `core_w_tiles` tiles | TILE gamma: raw accessor reads; RM gamma: `read_sticks_for_tilize` + `tilize` | gamma DRAM pages | `cb_gamma_tiles`, Wc (via `cb_gamma_sticks`, 32 pages, for RM) | resident for the kernel |
| 2 | `publish_x_shard` (R3) | `Rt × Wc` | raw (`cb_reserve_back`/`cb_push_back`, reader; `cb_wait_front`, compute) | shard buffer | `cb_x_tiles`, `Rt·Wc` | resident for the kernel |
| 3 | `load_x_block` | `rows × Wc` tiles (RM: `32·rows` chunks) | TILE: raw accessor reads (batched, one barrier); RM: `read_sticks_for_tilize` | x DRAM pages | `cb_x_tiles`, `rows·Wc` (RM: `cb_x_sticks`, `32·rows`) | — |
| 4 | `tilize_x_block` (RM) | `rows` tile-rows × Wc | yes (`tilize`) | `cb_x_sticks`, 32 per tile-row, popped | `cb_x_tiles`, `rows·Wc` | — |
| 5 | `sumsq_block` | `rows × Wc` | yes (`sum_of_squares`; R3: `eltwise_chain` with `TileOffset::Set`) | `cb_x_tiles`, `rows·Wc`, Upfront wait, **not popped** | `cb_sumsq_partial`, `rows` (PerOuter push) | x still resident |
| 6 | `collapse_block` | `rows × 1` | yes (`reduce`, ReduceTile, SUM, REDUCE_ROW) | `cb_sumsq_partial`, `rows`, BulkWaitBulkPop | `cb_gather` (Cw=1) / `cb_partial_collapsed` (Cw>1), `rows` | — |
| 7 | `exchange_partials_block` (Cw>1) | `rows` tiles per sender | raw dataflow (`noc_async_write`, `noc_semaphore_inc`, `Semaphore<>::wait/set`) — see "Helpers considered" | `cb_partial_collapsed`, `rows` (writer waits, sends, pops) | root: `cb_gather`, `rows·Cw` | root's slots filled, semaphore reset |
| 8 | `combine_block` (root or Cw=1) | `rows × Cw` | yes (`reduce`, AccumulateViaAdd, Skip, post-op) | `cb_gather`, `rows·Cw`, BulkWaitBulkPop | `cb_rstd` (Cw=1) / `cb_rstd_handoff` (Cw>1), `rows` | — |
| 9 | `broadcast_rstd_block` (Cw>1) | `rows` tiles | yes (`mcast_pipe` `SenderPipe::send` / `ReceiverPipe::receive`) | root: `cb_rstd_handoff`, `rows` (writer waits, sends, pops) | `cb_rstd`, `rows` on every group core (writer pushes after receive / after send-loopback) | — |
| 10 | `normalize_block` | `rows × Wc` | yes (`mul`, `BroadcastDim::Col`) | `cb_x_tiles` (None wait, AtEnd pop `rows·Wc`), `cb_rstd` (Upfront `rows`, AtEnd pop) | gamma: `cb_normed`, `rows·Wc`; no gamma: `cb_output_tiles`, `rows·Wc` | x and rstd released |
| 11 | `scale_block` (gamma) | `rows × Wc` | yes (`mul`, `BroadcastDim::Row`) | `cb_normed` streaming (PerTile/PerTile), `cb_gamma_tiles` (Upfront `Wc`, never popped) | `cb_output_tiles`, `rows·Wc` | gamma resident |
| 12 | `untilize_x_block` (RM) | `rows` tile-rows × Wc | yes (`untilize`) | `cb_output_tiles`, `rows·Wc` | `cb_out_sticks`, `Wc` per tile-row | — |
| 13 | `store_block` | `rows × Wc` tiles (RM: `32·rows` chunks) | TILE: raw accessor writes (one barrier per block); RM: `write_sticks_after_untilize`; R3: nothing | `cb_output_tiles` / `cb_out_sticks` | output DRAM pages | — |

## API Mapping

All line numbers verified against the working tree.

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| kernel boot (compute) | helper | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:54` | `(cb_x_tiles, cb_scaler, cb_output_tiles)` — once, first statement of `MAIN` | — | — | — |
| `prepare_scaler` | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:97-99` | pool-type-aware overload; `reduce_factor` default (SUM ⇒ 1.0) | — | `cb_scaler` | — |
| `load_gamma_slice` (RM gamma) | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_sticks, TilizeGranularity::ROW>(acc, 1, chunk_bytes, 0, w_byte_offset)` + `fill_l1_range<e_g>(row1_addr, 31·chunk_bytes, 0)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:446-452`; `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:453-454` | `chunk_bytes = core_w_tiles·32·e_g`, `w_byte_offset = w_tile_start·32·e_g` | gamma stick page 0 | `cb_gamma_sticks` | `core_w_tiles`, `w_tile_start` |
| `load_gamma_slice` (RM gamma, compute) | helper | `compute_kernel_lib::tilize<core_w_tiles, cb_gamma_sticks, cb_gamma_tiles>(1, /*total_input_pages=*/1)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:187-197` | `block_width_tiles = core_w_tiles` (CT), asymmetric mode | `cb_gamma_sticks` | `cb_gamma_tiles` | `core_w_tiles` |
| `load_gamma_slice` (TILE gamma) | raw_api | `noc_async_read(acc.get_noc_addr(w_tile_start + i), get_write_ptr(cb_gamma_tiles) + i·T_g, T_g)` for `i < Wc`, `noc_async_read_barrier`, `cb_push_back(cb_gamma_tiles, Wc)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:552,1750,208` | — | gamma tile pages `[w_tile_start, +Wc)` | `cb_gamma_tiles` | `core_w_tiles`, `w_tile_start` |
| `load_x_block` (TILE) | raw_api | `TensorAccessor` + `noc_async_read` per tile (`tile_id = (row_tile_start + r)·Wt + w_tile_start + c`), one `noc_async_read_barrier`, one `cb_push_back(cb_x_tiles, rows·Wc)` | `tech_reports/tensor_accessor/tensor_accessor.md`; `dataflow_api.h:552,1750,208` | — | x tile pages | `cb_x_tiles` | `block_rows`, `core_w_tiles` |
| `load_x_block` (RM) | helper | `read_sticks_for_tilize<cb_x_sticks, TilizeGranularity::ROW>(acc, 32·rows, chunk_bytes, 32·row_tile_start_of_block, w_byte_offset)` | `tilize_helpers_dataflow.hpp:446-452` | ROW granularity ⇒ 1 page per stick | x stick pages | `cb_x_sticks` | `block_rows`, `core_w_tiles` |
| `tilize_x_block` | helper | `compute_kernel_lib::tilize<core_w_tiles, cb_x_sticks, cb_x_tiles>(rows, 32·rows)` | `tilize_helpers.hpp:187-197` | `fp32_mode = Fast` (default; FPU consumers truncate to tf32 anyway) | `cb_x_sticks` | `cb_x_tiles` | `core_w_tiles` (CT), `rows` (RT) |
| `sumsq_block` | helper | `compute_kernel_lib::sum_of_squares<input(cb_x_tiles, WaitPolicy::Upfront, PopPolicy::None, OperandKind::Block), row_output(cb_sumsq_partial)>(IterationShape::grid(rows, core_w_tiles))` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:96-97` (decl), `convenience.inl:37-48` (body: `DestAccumulation::PerRow`, PerOuter push); `chain.hpp:356-362` (`input`), `:248` (`OperandKind`) | R3: `eltwise_chain(grid(rows, Wc), BinaryFpu<Mul, in(Set), in(Set), D0, DestAccumulation::PerRow>{base, base}, PackTile<output(cb_sumsq_partial, PerOuter, PerOuter, …, DestAccumulation::PerRow)>{})` with `base = block_idx·B·Wc` (`chain.hpp:277,513,531`) | `cb_x_tiles` | `cb_sumsq_partial` | `rows`, `core_w_tiles` (IterationShape) |
| `collapse_block` | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sumsq_partial, cb_scaler, CB_COLLAPSE_OUT, ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(rows, 1))` with `CB_COLLAPSE_OUT = (NUM_W_SPLITS == 1) ? cb_gather : cb_partial_collapsed` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:632` (decl), `:112` (policy), `:290` (shape) | `algorithm = Auto` (ReduceTile) — the `collapse_algorithm` lamp swaps to `AccumulateViaAdd` | `cb_sumsq_partial` | `cb_gather` / `cb_partial_collapsed` | `rows` |
| `exchange_partials_block` | raw_api | sender: `noc_async_write(src, get_noc_addr(root_x, root_y, gather_base + (r·Cw + w_split_index)·4096), 4096)` ×`rows`, `noc_async_write_barrier()`, `noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SEM_GATHER)), 1)`; root: `cb_reserve_back(cb_gather, rows·Cw)`, own-slot local copy (`noc_async_read(get_noc_addr(my_x, my_y, src), dst, rows·4096)` + barrier), `Semaphore<> gather(SEM_GATHER); gather.wait(num_partials_expected); gather.set(0);` `cb_push_back(cb_gather, rows·Cw)` | `dataflow_api.h:828,1780,2264,1501,404,208`; `tt_metal/hw/inc/api/dataflow/noc_semaphore.h:248,282` | root coords as virtual coords via `device.worker_core_from_logical_core` (host) | `cb_partial_collapsed` | `cb_gather` | `rows`, `num_w_splits` |
| — Helpers considered and rejected for `exchange_partials_block` | | `mcast_pipe` `SenderPipe`/`ReceiverPipe` (`mcast_pipe.hpp:139,205`) — one active sender per round to a rectangle (`mcast_pipe.hpp:33`), i.e. a broadcast; the gather is a many-to-one unicast with a counting semaphore, the inverse pattern. `Mcast2D` rotating-sender all-gather (`mcast_host.hpp:134-142`) would serialize `Cw` mcast rounds each with a handshake — it is regime R6's traffic with R6's rejection. The catalog reference realization is `ttnn/ttnn/operations/examples/tensix_all_reduce/program_descriptor_with_inline_kernels.py:372-384` (root gather: `noc_async_write` into `gather_addr + my_index·payload_bytes`, `noc_async_write_barrier`, semaphore inc). | | | | | |
| `combine_block` | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_gather, cb_scaler, CB_RSTD_OUT, ReduceInputPolicy::BulkWaitBulkPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::AccumulateViaAdd, ReduceWithinTile::Skip>(ReduceInputBlockShape::of(rows, num_partials), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, post_op)` with `CB_RSTD_OUT = (NUM_W_SPLITS == 1) ? cb_rstd : cb_rstd_handoff`; `post_op = [](uint32_t dst){ binop_with_scalar_tile_init(); mul_unary_tile(dst, inv_w_bits); add_unary_tile(dst, eps_bits); rsqrt_tile_init(); rsqrt_tile(dst); }` | `reduce_helpers_compute.hpp:632`, `:156` (algorithm), `:191` (Skip), `reduce_helpers_compute.inl:576-583` (Skip semantics), `:1350-1356` (post-op must arm SFPU under Skip); `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:156,60,35`; `tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h:19,38` | Skip: inputs are already column-0-valid (`collapse_block` output) so the within-tile collapse is skipped; `rsqrt_tile<false,false,DST_ACCUM_MODE>` honours `APPROX` = `math_approx_mode` | `cb_gather` | `cb_rstd` / `cb_rstd_handoff` | `rows`, `num_w_splits` |
| `broadcast_rstd_block` | helper | root writer: `constexpr auto mc = McastArgs<CT_BASE, RT_BASE>(); auto sender = mc.sender(noc); sender.send(get_read_ptr(cb_rstd_handoff), get_write_ptr(cb_rstd), rows·4096);` then `cb_push_back(cb_rstd, rows)`; receivers: `mc.receiver(noc).receive(); cb_push_back(cb_rstd, rows)` | `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp:242-289` (`McastArgs`), `:154-155` (`send`), `:218` (`receive`); loopback: `mcast_pipe.inl:91-99` | host: `ttnn.Mcast2D(device, group_rect, root, ttnn.McastConfig(sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED]))` per group (`ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp:134-142`; bindings per `.claude/references/ttnn-python-utility-bindings.md`) — CT args identical across groups, RT args per core | `cb_rstd_handoff` | `cb_rstd` (all group cores) | `rows` |
| `normalize_block` | helper | `compute_kernel_lib::mul<input(cb_x_tiles, WaitPolicy::None, PopPolicy::AtEnd, OperandKind::Block), input(cb_rstd, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Col), output(CB_NORM_OUT)>(IterationShape::grid(rows, core_w_tiles))` with `CB_NORM_OUT = HAS_GAMMA ? cb_normed : cb_output_tiles`; R3: `TileOffset::Set` bases on x and out | `convenience.hpp:52-53`; `chain.hpp:314` (`BroadcastDim`), `:370-377` (`input` with broadcast), `:391-400` (`output`); legality `chain.inl:71-90` | REDUCE_ROW result is column-shaped ⇒ `BroadcastDim::Col` (chain.hpp:309-313) | `cb_x_tiles`, `cb_rstd` | `cb_normed` / `cb_output_tiles` | `rows`, `core_w_tiles` |
| `scale_block` | helper | `compute_kernel_lib::mul<input(cb_normed), input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, OperandKind::Row), output(cb_output_tiles)>(IterationShape::grid(rows, core_w_tiles))`; R3: `TileOffset::Set` base on out | `convenience.hpp:52-53`; `chain.hpp:356` (default streaming input = Scalar kind, PerTile/PerTile) | gamma is `[1, W]` ⇒ `BroadcastDim::Row`, `OperandKind::Row` (indexed by column) | `cb_normed`, `cb_gamma_tiles` | `cb_output_tiles` | `rows`, `core_w_tiles` |
| `untilize_x_block` | helper | `compute_kernel_lib::untilize<core_w_tiles, cb_output_tiles, cb_out_sticks>(rows)` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:346-355` | `block_width_tiles = core_w_tiles` (CT) | `cb_output_tiles` | `cb_out_sticks` | `core_w_tiles`, `rows` |
| `store_block` (TILE) | raw_api | `noc_async_write(get_read_ptr(cb_output_tiles) + i·T_out, acc.get_noc_addr(tile_id), T_out)` batched, one `noc_async_write_barrier`, `cb_pop_front(cb_output_tiles, rows·Wc)` | `dataflow_api.h:828,1780,259` | — | `cb_output_tiles` | output tile pages | `block_rows`, `core_w_tiles` |
| `store_block` (RM) | helper | `dataflow_kernel_lib::write_sticks_after_untilize<cb_out_sticks>(acc, 32·rows, chunk_bytes, 32·row_tile_start_of_block, w_byte_offset)` | `tilize_helpers_dataflow.hpp:488-494` | — | `cb_out_sticks` | output stick pages | `block_rows`, `core_w_tiles` |
| host CBs on shards (R3) | helper | `ttnn.cb_descriptor_from_sharded_tensor(CB_X_TILES, input)`, `(CB_OUTPUT_TILES, output)` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:518` | — | — | — | — |
| host work split | helper | `ttnn.split_work_to_cores(...)` (two-group ceil/floor) for rows over groups; `ttnn.corerange_to_cores(grid, None, True)` for shard order; `ttnn.get_max_worker_l1_unreserved_size()` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:469,503`; `ttnn/cpp/ttnn-nanobind/tensor.cpp:623`; `ttnn/ttnn/device.py:20` | — | — | — | all of §H |

Helpers considered and rejected for the two raw dataflow entries `load_x_block`/`store_block` (TILE): `read_sticks_for_tilize`/`write_sticks_after_untilize` are stick-indexed (`tilize_helpers_dataflow.hpp:429-435`: "Accessor for the source tensor (stick-indexed)"); tiled pages are addressed by tile id, for which the kernel-lib exposes no block-read helper — `TensorAccessor` + batched `noc_async_read` is the documented pattern (`.claude/references/ttnn-cb-memory-fundamentals.md` → "TensorAccessor Pattern"). This is a block operation (all `rows·Wc` reads, one barrier, one push), not a unit-at-a-time loop in the main schedule.

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `sumsq_block` | `x * x` (same CB both operands, DEST-accumulated per tile-row) | `cb_x_tiles` — All | `cb_x_tiles` — All | None |
| `combine_block` post-op | SFPU scalar mul / add / rsqrt on DEST | `cb_gather` slots — Col0 (REDUCE_ROW output) | — (scalars `inv_w_bits`, `eps_bits`) | — |
| `normalize_block` | `x * rstd` | `cb_x_tiles` — All | `cb_rstd` — **Col0** (REDUCE_ROW out) | **Col** (`BroadcastDim::Col`) |
| `scale_block` | `normed * gamma` | `cb_normed` — All | `cb_gamma_tiles` — **Row0** (1-D `[W]` operand in a padded tile-row / tilized stick) | **Row** (`BroadcastDim::Row`) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Gather slot addresses drift across rounds | peers write into the root's `cb_gather` by base + slot offset; a multi-round ring moves the base | `cb_gather` and `cb_rstd` capacity = exactly one round (Mechanism caps); ragged last block keeps base-relative slots |
| Two consumers on one CB (silent UB) | tempting shortcuts: in-place gamma multiply on `cb_output_tiles` (writer also reads it), or the root reading rstd from the mcast source CB | dedicated `cb_partial_collapsed` / `cb_rstd_handoff` handoff CBs; the root gets its own rstd back through the mcast **loopback** (`mcast_pipe.inl:91-99`), so `cb_rstd` has one producer (writer) and one consumer (compute) |
| Semaphore reset race on the root | a fast peer's inc for round k+1 could be lost if it landed before the root's reset | it cannot: a peer sends round k+1 only after consuming rstd_k, which the root multicasts **after** `wait` + `set(0)`; state the ordering in the writer |
| `ReduceWithinTile::Skip` + post-op SFPU init | under Skip the reduce never arms the SFPU; `mul_unary_tile` without `binop_with_scalar_tile_init` computes garbage (`reduce_helpers_compute.inl:1350-1356`) | post-op begins with `binop_with_scalar_tile_init()` and calls `rsqrt_tile_init()` before `rsqrt_tile` |
| `Skip` is only legal on `AccumulateViaAdd` + `SUM` | `Auto` resolves to ReduceTile and fails to compile with Skip (`reduce_helpers_compute.inl:909-918`) | `combine_block` pins `ReduceAlgorithm::AccumulateViaAdd` explicitly |
| 1/W must be the **full** W, not the core's slice | each core's partial covers `core_w_tiles·32` elements; the mean divides by W | `inv_w_bits = bits(1.0f / W)` from the tensor width, RT arg; the scaler CB holds 1.0 (SUM), never 1/W |
| `sum_of_squares` output is NOT collapsed | its tile holds 32×32 element-wise partial sums (`convenience.hpp:85-95`) | `collapse_block` (REDUCE_ROW, Collapse) runs before any cross-core sum; only then is Skip valid |
| Eltwise operand kinds vs. policies | `OperandKind::Block` forbids PerTile/PerTile; Row/Col need Upfront-or-None wait (`chain.inl:71-90`) | policies fixed per operand in API Mapping; streaming operands use the default (`Scalar` kind + PerTile/PerTile) |
| Passive cores in a sharded bounding box | non-member cores of a non-rectangular shard grid must not touch tensor CBs but must ack the mcast | RT `core_w_tiles = 0` branch skips loads/compute/sends; writer still runs `receive()`; `num_partials_expected = active − 1` |
| `tilize`/`untilize` width is compile-time | ragged W-splits give two `core_w_tiles` values | one kernel descriptor per distinct width (≤ 2 groups); CBs sized uniformly with `core_w_tiles_max` |
| RM gamma tilize reads 32 rows from a 1-row push | `tilize<Wc>(1, 1)` reads 32 stick-pages; only page 0 is written | `cb_gamma_sticks` allocated with 32 pages; rows 1..31 zero-filled by the reader before the push; downstream reads row 0 only |
| fp32 input precision through the FPU | every FPU op (`mul`, `add_tiles`, reduce) unpacks to tf32 (10-bit mantissa) ⇒ ~5e-4 relative error on fp32 outputs; the fast tilize truncates the same way | within fp32 tolerance (PCC 0.999 / rel-RMS 0.02); accumulation itself is fp32 in DEST; `ReduceFp32Mode::Accurate` remains available for the collapse if a stricter baseline is ever set |
| `compute_kernel_hw_startup` re-boot advice differs between helper docs | `chain.hpp:34-40` suggests one boot per pack-CB stage; `reduce_helpers_compute.hpp:31-35` forbids re-calling it | call it **once**; rely on the helpers' per-phase format reconfig (`DataFormatReconfig::Enabled`, `ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT`) — the catalog's multi-phase kernel does exactly this (`examples/compute_block_size/program_descriptor_with_inline_kernels.py:127`) |
| Regime reachability is device-dependent | R2-by-occupancy depends on `num_cores`; R3's passive-core path only on grids where the shard grid is not a rectangle | regime-pinned tests (Work Distribution); `derive_blocking()` is pure and unit-testable on the host |
| Output `memory_config` for sharded input | the harness requests the input's shard spec for the output | validate `memory_config == input.memory_config()` when sharded; allocate the output with it before building the descriptor |

## Support contract (Phase 0 rectangle — informative for the implementer's `SUPPORTED`)

| Axis (name as in `feature_spec.TARGET`) | Phase 0 | Refinement candidates (`TARGET − SUPPORTED`) |
|---|---|---|
| `dtype` | bfloat16, float32, bfloat8_b (Refinement 1) | — |
| `fp32_dest_acc_en` | True, False (Refinement 1: page formats and `DEST_AUTO_LIMIT` follow the config; `{float32, False}` stays an EXCLUSION forever) | — |
| `layout` | TILE_LAYOUT, ROW_MAJOR_LAYOUT | — |
| `rank` (tagger `tag_rank`) | 2, 3, 4 | — |
| `gamma_mode` | gamma, no_gamma | — |
| `gamma_dtype` | bfloat16, float32, bfloat8_b (Refinement 1), "none" | — |
| `gamma_layout` | TILE_LAYOUT, ROW_MAJOR_LAYOUT, "none" | — |
| `memory_layout` | INTERLEAVED, WIDTH_SHARDED | — |

`validate()` order: (1) `ValueError` for rank < 2, non-tile-aligned last two dims, gamma last-dim mismatch, gamma rank/shape other than `(1,…,1,W)` — the message text MUST contain the substring `rank` for the rank error and `gamma` for every gamma-shape error (the acceptance test matches on them via the `expect_error` fixture); (2) `UnsupportedAxisValue` / `ExcludedCell` (`ttnn/ttnn/operations/_op_contract.py`) for anything outside the rectangle; `fp32_dest_acc_en` is read from the caller's config (resolved through `default_compute_kernel_config()` when `None`) and gated; `math_fidelity` / `math_approx_mode` are passed through unchanged. Gamma format axes are derived from the gamma tensor (`"none"` sentinel when absent) exactly as `eval/golden_tests/rms_norm/axes.py::classify_call` does; `INPUT_TAGGERS = {"rank": tag_rank}` per the feature spec's docstring.

### Structural impossibilities

None beyond those already in `feature_spec.INVALID` (`bfloat8_b + ROW_MAJOR` on either tensor, the `"none"` sentinel coupling, `ROW_MAJOR + WIDTH_SHARDED`).

## Catalog pointers

| Knob / decision | Catalog entry (`ttnn/ttnn/operations/examples/master.md`) | What it informed |
|---|---|---|
| W-split when rows under-fill the grid | `width_split` | occupancy term in §H2 (`Cw_occ`) |
| `depth_x = depth_out = 2`, batched reads with one barrier per block | `double_buffer`, `tile_reorder` | buffer-depth knobs; `load_x_block`/`store_block` shape |
| `sum_of_squares` (DEST-accumulated `x·x`) + one finalize reduce instead of a wide reduce | `row_reduce_accumulate`, `reduce_accumulate`, `eltwise_l1_vs_dest_accumulate` | phases A/B; `collapse_algorithm` lamp |
| root-gather + mcast for the cross-core partial sum | `tensix_all_reduce` (root/tree reduce fastest for tiny payloads on 1-D groups; pull/push and L1 notes), `tensix_all_reduce_compute` | R2's combine topology; R6 rejected |
| `mcast_pipe` for the rstd broadcast | `shared_input_reuse`, `mcast_topology` | `broadcast_rstd_block`; R4 (gamma `Mcast1D(PerColumn)`) |
| L1 round trip between D and E instead of DEST reuse | `compute_fusion` | `cb_normed`; `normed_roundtrip` lamp |
| coarse `block_rows`, format-reconfig elision | `compute_block_size` | §H3 default; `format_reconfig` lamp |
| group layout along x | `noc_placement` | Work Distribution grid; `reader_noc_placement` lamp |
| SFPU scoping of the rsqrt finalize | `sfpu_tile_scope` | `sfpu_scope` lamp |

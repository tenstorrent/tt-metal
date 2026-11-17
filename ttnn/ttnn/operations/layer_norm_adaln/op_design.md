# Operation Design: layer_norm_adaln

## Overview

| Field | Value |
|-------|-------|
| Classification | fused compute (row reduction + cross-core statistics combine + broadcast eltwise) |
| Goal | Per-row LayerNorm over the chip-local last dim of a `(1,1,H,W)` bf16 TILE DRAM-interleaved tensor, with optional per-call adaLN modulation `out = norm(x)·weight + bias`. One `ttnn.generic_op` dispatch per call. |
| Math | `mean_i = Σ_j x_ij / W`; `var_i = Σ_j (x_ij − mean_i)² / W`; `y_ij = (x_ij − mean_i)·rsqrt(var_i + ε)`; affine: `out_ij = y_ij·w_j + b_j`, else `out = y` |
| Numerics | Pivot-shifted two-pass (exact bf16 pivot `p_i ≈ mean_i`, deviations `x − p` formed exactly by the FPU) + Chan's parallel combine of per-slice `(mean, M2)` in fp32 SFPU. Never `E[x²] − E[x]²`. |
| References | `eval/golden_tests/layer_norm_adaln/{feature_spec.py,helpers.py}` (golden, read-only); `.claude/references/{blocking-model.md,l1-footprint-discipline.md,precision_convention.md,ttnn-cb-memory-fundamentals.md}`; `ttnn/cpp/ttnn/kernel_lib/{reduce_helpers_compute.hpp,reduce_helpers_dataflow.hpp,eltwise/api/chain.hpp,eltwise/broadcast/bcast.hpp,mcast/host/mcast.hpp,mcast/kernel/mcast_args.hpp}`; generic-op template `.claude/references/generic_op_template/`. Perf catalog `ttnn/ttnn/operations/examples/master.md` is **absent on this branch** — knobs below are set from first principles. |
| Companion | `l1_ledger.md` (CB ledger + data-movement budget), same directory |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | see Tensors | — | buffer addr RT |
| `weight` | `ttnn.Tensor \| None` | no (with `bias`) | `(1,1,1,W)` bf16 TILE | `None` | buffer addr RT (new tensor every call) |
| `bias` | `ttnn.Tensor \| None` | no (with `weight`) | `(1,1,1,W)` bf16 TILE | `None` | buffer addr RT |
| `epsilon` (kw-only) | float | no | `> 0` (not gated) | `1e-6` | RT (fp32 bit pattern, compute) |
| `compute_kernel_config` (kw-only) | `ttnn.ComputeConfigDescriptor \| None` | no | any `math_fidelity` / `math_approx_mode`; `fp32_dest_acc_en ∈ {True, False}` | `default_compute_kernel_config()` | build config |
| `HAS_AFFINE` | derived bool | — | weight and bias both present | — | CT (reader, compute) |

`default_compute_kernel_config()` is an **exported factory** (fresh object per call) returning `ttnn.ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)`; the entry point resolves `None` through it (`precision_convention.md`). The op honors the caller's `math_fidelity` and `fp32_dest_acc_en`; it never gates on fidelity. The op file exports `layer_norm_adaln`, `default_compute_kernel_config`, `INPUT_TAGGERS` (`alignment` → `"tile_aligned" | "w_non_aligned" | "h_non_aligned"`, `rank` → `len(shape)`), `SUPPORTED`, `EXCLUSIONS`, and `select_blocking` (below). Axis names are exactly feature_spec's TARGET keys: `dtype, fp32_dest_acc_en, layout, alignment, rank, affine, affine_dtype, affine_layout, memory_layout`.

### Input validation (`validate()`; raise `ValueError`)

| Check | Rule |
|-------|------|
| input dtype / layout / memory | `bfloat16`, `TILE_LAYOUT`, `memory_config().memory_layout == INTERLEAVED` and buffer type DRAM |
| rank / alignment | `len(shape) == 4`; `H % 32 == 0`; `W % 32 == 0` (shape `(N,C,H,W)`; the design treats `N·C` as additional rows, see Blocking Model) |
| affine pairing | `(weight is None) != (bias is None)` → error |
| affine tensors | shape `(1,1,1,W)` (same W as input), `bfloat16`, `TILE_LAYOUT`, on device, DRAM interleaved |
| precision | `fp32_dest_acc_en=False` **allowed** (16-bit DEST variant of every CB, ledger column `acc page`); fidelity never gated |
| fit | `select_blocking(...)` finds a feasible regime (else `ValueError`, see regime `wslice_reread`) |

The entry point performs **no** `to_layout` / `tilize` / `pad` / `slice` / composite ttnn op; it validates, allocates the output with `ttnn.allocate_tensor_on_device(shape, bfloat16, TILE_LAYOUT, device, DRAM_MEMORY_CONFIG)`, builds one `ProgramDescriptor`, and calls `ttnn.generic_op` once. Nothing derived from `weight` / `bias` is cached host-side (their addresses are runtime args, rebuilt every call).

### Structural impossibilities

None beyond feature_spec's `INVALID` (the affine/sentinel coupling).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(N, C, H, W)`, rank 4, `H % 32 == 0`, `W % 32 == 0` (golden: `N = C = 1`) |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | DRAM interleaved |
| weight / bias | `(1,1,1,W)` bf16 TILE DRAM interleaved; physical page = one 32×32 tile whose **row 0** holds the data |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | DRAM interleaved |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Index space is tiles: `tensor_h_tiles = N·C·ceil(H/32)` (per-image ceil; equals `N·C·H/32` for the supported aligned shapes), `tensor_w_tiles = ceil(W/32)`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `N`, `C` (batch) | independent — rows of different images never interact | folded into the row axis (`tensor_h_tiles` counts `N·C·ceil(H/32)`) | whole (folded) | `tensor_h_tiles` host derivation | via row axis | knob-turn (already folded) |
| row (`tensor_h_tiles`) | **independent** — each row's stats span only its own W | `block_height` (tile rows per block) | `core_h_tiles` (whole per-core assignment: one block per core) unless the L1 predicate clamps it | `select_blocking()` → CT arg | split over `num_row_groups` groups, balanced (`ttnn.split_work_to_cores` semantics over tile rows; groups differ by ≤ 1 row) | knob-turn |
| W (`tensor_w_tiles`) | **dependent** — mean/var span the whole row | `block_width` (= `core_w_tiles` = `tensor_w_tiles / num_w_slices`) | whole slice — **must** be resident for the two-pass (a smaller value forces a DRAM re-read; regime `wslice_reread`) | `select_blocking()` → CT arg | split over `num_w_slices` = S consecutive cores of a group (one physical grid row), combined by an all-gather of per-slice stats | knob-turn (combine is built; S is a parameter) |
| intra-tile rows / cols (32×32) | n/a — hardware tile; never split | tile | 32×32 | — | — | — |
| weight / bias W | **reuse-shared** — identical for every row group (does not vary along the row split) | `block_width` (same slice as x) | whole slice, fetched **once per core per call**, resident across all blocks | `select_blocking()` | each core fetches its slice; only row 0 of each tile is fetched (see Dataflow) | deferred mcast (regime `affine_mcast`) |
| per-row stats (intermediate: `r`, `M2` per row per slice) | rows independent; slices dependent (combined) | `block_height` × `num_w_slices` | per block | derived | **every core of a group computes the combine redundantly for its own frame** — there is no designated combiner core, so no stage runs on one core while the group waits | — |
| normalized output (intermediate→output) | independent (same as x) | `block_height` × `block_width` | per block | derived | same core as x | — |

**Phase 0 is multi-core**: rows over up to `grid_y·grid_x/S` groups and W over S cores per group, picked per shape by `select_blocking`. Every knob (`num_w_slices`, `num_row_groups`, `block_height`, `block_width`, all depths, the grid) is a parameter defined once in `select_blocking` / the program-descriptor constants; kernels receive them as CT/RT args, and every CB size and trip count is derived from them.

#### Regime/knob selection — `select_blocking(tensor_h_tiles, tensor_w_tiles, grid_x, grid_y, *, has_affine=True, acc_tile_bytes=4096, l1_budget_bytes=DEFAULT_L1_BUDGET_BYTES) -> Blocking`

Pure host function, exported from the op package (the acceptance test calls it with explicit grids). `Blocking` fields: `num_w_slices`, `num_row_groups`, `core_h_tiles` (max over groups), `block_height`, `block_width`, `num_blocks_per_core` (max). Host constants (single source each): `MIN_SLICE_TILES = 8`, `GATHER_TILE_CAP = 48`, `L1_BUDGET_BYTES` = the `l1_budget_bytes` argument: the entry point passes the device query (L1 per core minus the allocator's reserved base); the default `DEFAULT_L1_BUDGET_BYTES = 1 MiB` keeps the function pure for tests and is ≤ the BH value, `INPUT_BLOCK_DEPTH = 1`, `OUTPUT_ROW_DEPTH = 2`, `AFFINE_STREAM_DEPTH = 2`.

```
candidates S ∈ { s : s | grid_x, s | tensor_w_tiles, (s == 1 or tensor_w_tiles/s >= MIN_SLICE_TILES) }
for each S:
    block_width   = tensor_w_tiles / S
    groups_max    = (grid_x / S) * grid_y
    num_row_groups= min(tensor_h_tiles, groups_max)
    core_h_tiles  = ceil(tensor_h_tiles / num_row_groups)
    block_height  = core_h_tiles
    feasible iff footprint(block_height, block_width, S) <= L1_BUDGET_BYTES      (l1_ledger.md total)
             and (S == 1 or 2*S*block_height <= GATHER_TILE_CAP)
    cost          = core_h_tiles * block_width          # tiles per busiest core
choose feasible S with min cost; tie -> smaller S.
fallback (no feasible single-block S): S = 1, block_height = largest bh with footprint(bh, tensor_w_tiles, 1) <= budget
          (num_blocks_per_core = ceil(core_h_tiles / block_height));
          if bh < 1: smallest S whose bh=1 is feasible (cap included); none -> ValueError (regime wslice_reread).
```

Resulting selections on the 12×10 Blackhole grid (pinned by the acceptance test):

| Shape | tensor_h_tiles × tensor_w_tiles | S | groups | block (h × w) | cores | Regime |
|-------|------|---|--------|---------------|-------|--------|
| (1,1,32,64) | 1 × 2 | 1 | 1 | 1 × 2 | 1 | `rows_only` |
| (1,1,64,128) | 2 × 4 | 1 | 2 | 1 × 4 | 2 | `rows_only` |
| (1,1,128,512) | 4 × 16 | 2 | 4 | 1 × 8 | 8 | `rows_x_wslices` |
| (1,1,256,1024) | 8 × 32 | 4 | 8 | 1 × 8 | 32 | `rows_x_wslices` |
| (1,1,32,3072) | 1 × 96 | 12 | 1 | 1 × 8 | 12 | `rows_x_wslices` |
| (1,1,256,3072) | 8 × 96 | 12 | 8 | 1 × 8 | 96 | `rows_x_wslices` |
| (1,1,2048,3072) | 64 × 96 | 3 | 40 | 2 × 32 | 120 | `rows_x_wslices` |
| (1,1,2304,3072) | 72 × 96 | 3 | 40 | 2 × 32 | 120 | `rows_x_wslices` |

(2048: S=6 also costs 64 and fits the cap; tie → S=3. S=12 costs 56 but needs `2·12·7 = 168 > 48` gathered tiles → infeasible.)

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `INPUT_BLOCK_DEPTH` (blocks) | 1 | whole block resident (required by the two-pass); depth 2 would let the reader prefetch block b+1 on multi-block cores (none in INPUTS) |
| `cb_output_tiles` | `OUTPUT_ROW_DEPTH` (tile rows of `block_width`) | 2 | compute packs row r+1 while the writer drains row r |
| `cb_weight_rows`, `cb_bias_rows` | `AFFINE_STREAM_DEPTH` (tiles) | 2 | reader fetches tile t+1 while compute broadcasts tile t |
| `cb_gathered_stats` | 1 block of S payloads | 1 | the exchange is once per block; no cross-block overlap in Phase 0 |
| all others | 1 block | 1 | per-block intermediates between sequential compute phases (same thread → no pipelining to buy) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Hardware mcast rectangle for the stats all-gather | `num_w_slices` must divide `grid_x` (a group is a 1×S run inside one physical row) | `select_blocking` candidate filter | a group straddles grid rows → the mcast rectangle covers foreign cores and overwrites their `cb_gathered_stats` region → silent corruption |
| Chan combine assumes equal slice counts | `num_w_slices` must divide `tensor_w_tiles` | candidate filter | unequal `n_slice` weights → wrong mean/var |
| Gathered-stats L1 | `2·S·block_height ≤ GATHER_TILE_CAP` | candidate infeasible (or block_height reduced in the fallback) | gathered CB grows as S·bh → L1 overflow / allocator failure |
| Group lockstep (same-address remote writes) | all S cores of a group process **identical** `(num_blocks, block_height)` sequences | rows are assigned per *group*, never per core | CB write pointers diverge → peer writes land in the wrong slot |
| L1 accumulation in `PackTile` | one output tile per chain call (`chain.inl:1133-1136`: `(OneUpfront, OneAtEnd)` or caller-managed) | `dev_sq_block` invokes the chain once per tile row (over `block_width` tiles) | static_assert / wrong accumulation |
| DEST slots for the fp32 combine | 4 simultaneously live fp32 tiles (μ/M2 accumulators + 2 temporaries) ≤ `DEST_AUTO_LIMIT` (4 at fp32 half-sync) | the combine walks one tile row at a time inside the block op | DEST overflow (static_assert on `Dst` slot) |
| Reduce scaler format | scaler = **1.0** (exact in bf16); `1/n_slice` applied as fp32 `MulUnary` in the reduce post-op | fixed design | a bf16 `1/3072` is off by ~0.4% → biased mean/var |
| Pivot exactness | `cb_pivot` must be **Float16_b** so the value in L1 is exactly what srcB sees and what the combine copies to DEST | fixed format | an fp32 pivot is truncated to tf32 in srcB but read exactly by the combine → frame mismatch → large-offset error |
| DRAM read alignment for the affine row-0 fetch | fetch granularity = `max(32 B, dram_alignment)` per face (64 B on BH = face rows 0–1), at tile offsets 0 and 512 | reader constant from `ttnn.get_dram_alignment()` | misaligned DRAM read → NoC error / wrong bytes |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `rows_only` | **built** | `select_blocking(...).num_w_slices == 1` | `block_height × tensor_w_tiles` | Boundary DRAM: x read once, out written once (= minimum). w/b: each of `num_row_groups` cores fetches row 0 of all `tensor_w_tiles` tiles (2 × 64 B × 2 faces per tile) — above the once-per-call minimum by `num_row_groups×`, ≤ 4% of x+out bytes. No NoC combine. | per block: the 8 block ops' inits/reconfigs (reduce, chains, bcast), one scaler wait, the per-phase CB handshakes. Intended frequency: once per block = once per core in every INPUTS shape. |
| `rows_x_wslices` | **built** | `num_w_slices > 1` | `block_height × (tensor_w_tiles / S)` | DRAM same as `rows_only` (x, out minimum; w/b row-0 per core: `num_row_groups·S` cores × `block_width` tiles). Adds NoC: per group per block, S rotating mcasts of `2·block_height` acc tiles to S−1 receivers. | everything in `rows_only` **plus** the per-block exchange rendezvous (S handshaked rotating rounds), amortized over `block_height` tile rows; intended once per block. |
| `rows_only` multi-block (fallback inside `rows_only`) | **built** (same code; `num_blocks_per_core > 1`) | `rows_only` and `footprint(core_h_tiles, …) > L1_BUDGET_BYTES` | `block_height < core_h_tiles` | x/out still minimum (block resident, read once) | fewer per-block inits; with `INPUT_BLOCK_DEPTH=2` overlap of next-block read (lamp) |
| `wslice_reread` | **deferred** — unreachable for every TARGET/INPUTS shape (W ≤ 3072 → a full-row slice is ≤ 96 tiles = 192 KB; raising S reaches residency first) | no S makes a 1-row block fit | `1 × chunk` of a slice | x read 2× (stats, normalize) | n/a. Reachable: the reduce helper's `Accumulate` path chunks the mean/dev passes over `block_width` sub-chunks with the same CB set; the entry raises `ValueError` until built. |
| `three_pass_streaming` (mean pass, var pass, normalize pass, each re-reading x) | **rejected** — dead end: 3× x DRAM reads; superseded everywhere by the resident block | — | `1 × chunk` | x read 3× | — |
| `raw_moments_single_pass` (`E[x²]−E[x]²`) | **rejected** — violates the stability rule; the pivot-shift two-pass has identical traffic | — | — | minimum | — |
| `two_round_combine` (exchange Σx, then exchange Σ(x−μ)²) | **rejected** — same bytes as the one-round all-gather, one more rendezvous, and the variance pass is serialized behind the first exchange | — | — | +1 rendezvous per block | — |
| `gather_reply_combine` (designated combiner per group gathers, computes, replies) | **rejected** — reply payload equals the all-gather's, adds a second rendezvous, and idles S−1 cores while one combines | — | — | (S−1)·2bh tiles in + reply | — |
| `affine_mcast` (one core per slice column fetches w/b, mcasts to the `num_row_groups` cores sharing the slice) | **deferred** — the built row-0 fetch already cuts w/b DRAM bytes 32× (≈ 1 MB at 2048 rows vs 25 MB x+out), so the multicast would save < 4% of DRAM bytes; it needs a non-rectangular (strided) receiver set | — | `block_width` tiles of w/b | w/b once per call | Reachable: `cb_weight_rows`/`cb_bias_rows` are reader-produced streams; swapping the producer from a DRAM read to an mcast receive changes no compute/CB contract. |

### Traffic ranking

Per call, per tier (bytes for (1,1,2048,3072); `x` = 12.6 MB, `out` = 12.6 MB):

| Candidate split | DRAM crossings | Cross-core (NoC) | Occupancy (separate question) |
|-----------------|----------------|------------------|-------------------------------|
| rows only, resident row (`rows_only`) | x 1×, out 1×, w/b row-0 × 64 cores ≈ 1.6 MB | 0 | 64 cores, 96 tiles each |
| rows × W-slices, all-gather combine (`rows_x_wslices`, S=3) | x 1×, out 1×, w/b row-0 × 40 groups ≈ 1.0 MB | 40 groups × 3 mcasts × 2·2 tiles × 4 KB ≈ 1.9 MB injected, ≈ 3.8 MB delivered to 2 receivers each (on-chip, row-local) | 120 cores, 64 tiles each |
| rows × W-slices, two-round combine | same | same bytes, 2 rendezvous | same |
| three-pass streaming | x 3×, out 1× (+25 MB) | 0 | any |
| W-split for residency (re-read avoided) | — needed only if a row slice does not fit; not the case for W ≤ 3072 | | |

Ranking (DRAM first): `rows_x_wslices` ≈ `rows_only` < `three_pass_streaming`. The two built regimes tie on DRAM (x, out at the minimum; w/b row-0 bytes differ by < 3%); `rows_only` has zero NoC traffic, `rows_x_wslices` adds a small row-local all-gather. With traffic comparable, the tiebreaker is the per-core work term (the busiest core's tile count), which `select_blocking` minimizes — S>1 is chosen only where it lowers that term (H=256: 96 → 8 tiles/core; H=2048/2304: 96 → 64). Cheapest-traffic split: `rows_x_wslices` at the selected S (DRAM) — implemented.

**Stall-shadow check.** The writer's all-gather is the only wait on another core. (1) The variance pass is lifted *before* the exchange: each slice accumulates `Σ(x−p_k)²` about its **own** bf16 pivot `p_k`, which is legal because a change of pivot is an exact algebraic identity (`M2` about the slice mean is pivot-invariant; Chan's combine merges slices with different means). This is what makes the combine one round. (2) The weight/bias Row-broadcast (`prep_affine_block`, independent of all stats) is scheduled in the exchange wait on the first block. (3) Nothing in `normalize_block` is independent of the combined stats. This reorder changes floating-point association (fp32 Chan combine vs a single global sum); its precision is gated by the large-offset and per-slice-offset acceptance tests.

### Block schedule

Per core (compute view; reader/writer realize their parts asynchronously):

```cpp
prepare_reduce_scaler_once();                         // reader: 1.0 scaler tile (kernel scope)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);            // reader: block_height × block_width x tiles → cb_input_tiles (push per tile row)
    mean_block(block_idx);            // compute: r = Σx / n_slice per row        → cb_local_mean (fp32)
    pivot_block(block_idx);           // compute: p = bf16(r)                    → cb_pivot (bf16)
    dev_sq_block(block_idx);          // compute: s2 = Σ (x − p)² per row        → cb_dev_sq_acc (fp32)
    publish_stats_block(block_idx);   // compute: (r, M2 = s2 − n(r−p)²)         → cb_local_stats
    exchange_block(block_idx);        // writer: rotating all-gather over S cores → cb_gathered_stats
    if (block_idx == 0 && HAS_AFFINE)
        prep_affine_block();          // compute: Row-bcast w, b slices (in the exchange wait) → cb_weight_full, cb_bias_full
    combine_block(block_idx);         // compute: μ, M2, m = μ − p, rstd          → cb_row_stats
    bcast_stats_block(block_idx);     // compute: Col-bcast m, rstd               → cb_shift_full, cb_rstd_full
    normalize_block(block_idx);       // compute: ((x − p) − m)·rstd [·w + b]     → cb_output_tiles
    store_block(block_idx);           // writer: cb_output_tiles → DRAM, per tile row
}
```

| Block operation | Block shape | Resident across it | Intended init / sync frequency |
|-----------------|-------------|--------------------|--------------------------------|
| `load_block` | `block_height × block_width` x tiles (+ first block: `block_width` w and b tiles, row 0 only) | — | one NoC read barrier per tile row; push per tile row |
| `mean_block` | `block_height × block_width` → `block_height` col tiles | `cb_input_tiles` (not popped) | one reduce init per block; post-op `MulUnary(1/n_slice)` per row |
| `pivot_block` | `block_height` tiles | `cb_local_mean` (not popped) | one copy/pack init per block |
| `dev_sq_block` | `block_height × block_width` → `block_height` acc tiles, then in-place row-reduce to col tiles | `cb_input_tiles`, `cb_pivot`, `cb_local_mean` | chain init once per block (Caller-owned hoist where legal), one L1-acc window per tile row; one reduce (Accurate fp32) per block |
| `publish_stats_block` | `block_height` rows → `2·block_height` tiles | `cb_pivot` | one SFPU chain setup per block |
| `exchange_block` | `2·block_height` tiles × S slots | — | S handshaked rotating mcast rounds per block (round index `block_idx·S + j`); one push of `2·S·block_height` |
| `prep_affine_block` | `block_width` w tiles + `block_width` b tiles | — | once per **call** per core |
| `combine_block` | S × `2·block_height` tiles → `2·block_height` tiles | `cb_pivot` | one SFPU setup per block; walks rows inside (DEST cap) |
| `bcast_stats_block` | `2·block_height` col tiles → `2·block_height` full tiles | — | one bcast init per block |
| `normalize_block` | `block_height × block_width` → same | `cb_shift_full`, `cb_rstd_full` (Col mapping), `cb_weight_full`, `cb_bias_full` (Row mapping) | one chain init per block; pops x, pivot, shift, rstd at block end |
| `store_block` | `block_height × block_width` | — | one write barrier per tile row |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (input) | Whole-block residency + a mean reduce over the block can serialize read against compute, worst when `block_height = 1` (H=256: each core's only row must fully land before `mean_block` finishes) | reader push quantum = tile row (default) vs `MIN_SLICE_TILES`-tile sub-row chunks with the reduce's per-tile/cumulative waits |
| Grid synchronization | S=12 at H=256 leaves 8 tiles/core while each core does 12 handshaked mcast rounds | `select_blocking` with `GATHER_TILE_CAP`/`MIN_SLICE_TILES` raised (S=6: 16 tiles/core, 6 rounds) |
| Grid synchronization (tie-break) | 2048/2304: S=3 vs S=6 tie on tile count; S=6 halves per-core w/b fetch and doubles rounds | S=6 |
| Stats transport format | each exchanged stat is a full acc tile with only column 0 meaningful (4 KB carries 128 B) | compact 128 B column records packed/unpacked by the writer |
| Normalize chain length | 5 FPU ops per output tile (Sub-bcast, 2 DestReuse for stats, 2 DestReuse for affine) | fold `shift` into an SFPU add on DEST (fewer srcA round trips) |
| Multi-block pipelining (fallback only) | `INPUT_BLOCK_DEPTH=1` cannot prefetch, and block b+1's stats could run in block b's exchange wait | `INPUT_BLOCK_DEPTH=2` with stats(b+1) before combine(b) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x DRAM → `cb_input_tiles` | bf16 tiles | reader, `TensorAccessor` page reads; tile id `(row_start + r)·tensor_w_tiles + w_start + c` | push per tile row (`block_width` pages) |
| w/b DRAM → `cb_weight_rows` / `cb_bias_rows` | bf16 tile pages, row 0 valid | reader, **row-0 fetch**: per tile two reads of `max(32, dram_alignment)` bytes at page offsets 0 and 512 (face 0 / face 1 row 0) into the same offsets of the CB page; rest of the page is don't-care | first block only; Row bcast reads only row 0. Full-page read is a correct (slower) fallback |
| scaler | bf16 tile = 1.0 | reader `calculate_and_prepare_reduce_scaler<cb_reduce_scaler, SUM, REDUCE_ROW>` | once per kernel |
| compute phases | DEST fp32 (when `fp32_dest_acc_en`) | helpers, see API Mapping | intermediates in acc format (Float32 / Float16_b) except `cb_pivot` (bf16 by design) |
| `cb_local_stats` → peers' `cb_gathered_stats` | acc tiles | writer, `ttnn.Mcast` with `McastRotatingSenderConfig` (groups of S consecutive row-major cores = 1×S rectangle), `handshake=True`. Round `j` of block b: core j of the group sends its `2·block_height` tiles to slot j (byte offset `j·2·block_height·acc_tile_bytes` from the reserved write pointer, identical on all members by lockstep); the sender copies its own payload into its own slot locally | S==1: no channel (`attach_absent_mcast`); the writer copies `cb_local_stats` → `cb_gathered_stats` with a local L1 write. All members `reserve_back(2·S·bh)` before the first round and `push_back(2·S·bh)` after the last |
| `cb_output_tiles` → DRAM | bf16 tiles | writer, `TensorAccessor` page writes | per tile row |

**TARGET placement axis.** TARGET's `memory_layout` is `[INTERLEAVED]` only. Recorded for later: HEIGHT_SHARDED = the row split (knob-turn: input/output CBs backed on the shard via `ttnn.cb_descriptor_from_sharded_tensor`, block = whole shard); WIDTH_SHARDED / BLOCK_SHARDED cut the **dependent** W axis — a knob-turn here only because the all-gather combine is built (`num_w_slices` = shard columns; groups must then follow the shard grid, not the 1×S row rule).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_height` tile rows × `block_width` W tiles (one W slice) |
| Grid | `device.compute_with_storage_grid_size()` → `(grid_x, grid_y)`; runtime parameter, never a literal (12×10 on BH) |
| Core layout | cores used = `num_row_groups · S`, taken row-major from (0,0); core linear id `L = y·grid_x + x`; group `g = L // S`, slice `s = L % S` (S | grid_x ⇒ a group never straddles a grid row) |
| Per-core work | rows: group g gets a contiguous tile-row range from a balanced split of `tensor_h_tiles` over `num_row_groups` (first `tensor_h_tiles % num_row_groups` groups get `ceil`, rest `floor`); W: tiles `[s·block_width, (s+1)·block_width)` |
| Blocks per core | `num_blocks_this_core = ceil(group_rows / block_height)`; last block extent `group_rows − (num_blocks−1)·block_height` passed as a runtime extent (same block op); CB push/pop counts stay at the block's runtime extent identically on reader/compute/writer |
| Remainder | tile geometry is per-image ceil: `tensor_h_tiles = N·C·ceil(H/32)`, `tensor_w_tiles = ceil(W/32)` (aligned-only today; formulas stay ceil) |
| Per-core RT args | x / out / w / b buffer addresses, `row_start`, `group_rows`, `w_start`, slice index `s`, epsilon bits; mcast args appended by `Mcast.attach` |
| CT args | `block_height`, `block_width`, `num_w_slices`, `tensor_w_tiles`, `HAS_AFFINE`, `n_slice = block_width·32`, `W = tensor_w_tiles·32` (inverse constants as fp32 bits), TensorAccessorArgs (placeholders when w/b absent) |
| Regime selection | `select_blocking` (above) — exact, host-checkable. **Regime-pinned tests required**: the acceptance test asserts `select_blocking` output for explicit grids (12×10) and runs shapes in both `rows_only` and `rows_x_wslices` |

## Circular Buffers

`T16` = bf16 tile (2048 B). `TA` = acc tile: Float32 (4096 B) when `fp32_dest_acc_en`, else Float16_b (2048 B). `S` = `num_w_slices`, `bh` = `block_height`, `bw` = `block_width`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_input_tiles` | 0 | T16 | `INPUT_BLOCK_DEPTH·bh·bw` | spans rows and W (two-pass residency) | bf16 | reader | compute | load → normalize (per block) |
| `cb_reduce_scaler` | 1 | T16 | 1 | constant | bf16 | reader | compute | kernel |
| `cb_local_mean` | 2 | TA | `bh` | spans rows; W reduced | acc | compute | compute | mean → publish |
| `cb_pivot` | 3 | T16 | `bh` | spans rows | **bf16** (exact pivot) | compute | compute | pivot → normalize |
| `cb_dev_sq_acc` | 4 | TA | `bh` | spans rows; W accumulated in L1, then reduced in place | acc | compute | compute | dev_sq → publish |
| `cb_local_stats` | 5 | TA | `2·bh` | spans rows (r, M2) | acc | compute | writer | publish → exchange |
| `cb_gathered_stats` | 6 | TA | `2·S·bh` | spans rows × slices | acc | writer | compute | exchange → combine |
| `cb_row_stats` | 7 | TA | `2·bh` | spans rows (m, rstd col tiles) | acc | compute | compute | combine → bcast |
| `cb_shift_full` | 8 | TA | `bh` | spans rows (m broadcast across cols) | acc | compute | compute | bcast → normalize |
| `cb_rstd_full` | 9 | TA | `bh` | spans rows | acc | compute | compute | bcast → normalize |
| `cb_weight_rows` | 10 | T16 | `AFFINE_STREAM_DEPTH` | streams over W | bf16 | reader | compute | first block (affine only) |
| `cb_bias_rows` | 11 | T16 | `AFFINE_STREAM_DEPTH` | streams over W | bf16 | reader | compute | first block (affine only) |
| `cb_weight_full` | 12 | T16 | `bw` | spans W (reused by every row) | bf16 (exact copy of bf16 data) | compute | compute | call (affine only) |
| `cb_bias_full` | 13 | T16 | `bw` | spans W | bf16 | compute | compute | call (affine only) |
| `cb_output_tiles` | 14 | T16 | `OUTPUT_ROW_DEPTH·bw` | streams over rows, spans W per row | bf16 | compute | writer | normalize → store |

Page ordering conventions: `cb_local_stats` / each gathered slot = `[r_0 … r_{bh−1}, M2_0 … M2_{bh−1}]`; `cb_row_stats` = `[m_0 … m_{bh−1}, rstd_0 … rstd_{bh−1}]`. `unpack_to_dest_mode = UnpackToDestFp32` for `cb_local_mean`, `cb_dev_sq_acc`, `cb_gathered_stats`, `cb_row_stats` when `fp32_dest_acc_en` (they are read by `CopyTile` into SFPU math and must not be truncated to tf32); default for all others (FPU srcA/srcB consumers).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `mean_block` | bh × bw | `reduce` | `cb_input_tiles` (bh·bw, wait cumulative per row, **no pop**); `cb_reduce_scaler` (1, never popped) | `cb_local_mean` (bh) | x resident |
| 2 | `pivot_block` | bh | `eltwise_chain` CopyTile→PackTile | `cb_local_mean` (bh, wait upfront, no pop) | `cb_pivot` (bh) | r resident |
| 3 | `dev_sq_block` | bh × bw → bh | `eltwise_chain` (BinaryFpu Sub Col-bcast → Square → PackTile L1-acc), then `reduce` in place | `cb_input_tiles` (no pop), `cb_pivot` (Col mapping, no pop) | `cb_dev_sq_acc` (bh acc tiles → transformed in place to bh col tiles: reduce pops 1 / pushes 1 with real credit) | x, p resident |
| 4 | `publish_stats_block` | bh | `eltwise_chain` (CopyTile×3 + SFPU Sub/Square/MulUnary/Sub) | `cb_local_mean` (pop bh), `cb_dev_sq_acc` (pop bh), `cb_pivot` (no pop) | `cb_local_stats` (2·bh) | — |
| 5 | `exchange_block` | S × 2·bh | dataflow `Mcast` rotating | `cb_local_stats` (writer waits 2·bh, pops 2·bh after its send round) | `cb_gathered_stats` (writer reserves/pushes 2·S·bh) | — |
| 6 | `prep_affine_block` | bw (×2) | `unary_bcast<Row>` | `cb_weight_rows`, `cb_bias_rows` (stream, pop per tile) | `cb_weight_full`, `cb_bias_full` (bw each, never popped until kernel end) | full w/b resident for all blocks |
| 7 | `combine_block` | S·2·bh → 2·bh | `eltwise_chain` SFPU (custom block op walking rows) | `cb_gathered_stats` (2·S·bh, wait upfront, pop at end), `cb_pivot` (no pop) | `cb_row_stats` (2·bh) | — |
| 8 | `bcast_stats_block` | 2·bh | `unary_bcast<Col>` | `cb_row_stats` (pop 2·bh) | `cb_shift_full` (bh), `cb_rstd_full` (bh) | — |
| 9 | `normalize_block` | bh × bw | `eltwise_chain` (BinaryFpu Sub Col-bcast → DestReuse Sub → DestReuse Mul [→ DestReuse Mul → DestReuse Add] → PackTile) | `cb_input_tiles` (pop bh·bw at end), `cb_pivot` (pop bh), `cb_shift_full`, `cb_rstd_full` (Col mapping, pop bh), `cb_weight_full`, `cb_bias_full` (Row mapping, no pop) | `cb_output_tiles` (push per tile row) | block's CBs drained |

Math of ops 4 and 7 (all fp32 in DEST, `n = n_slice = bw·32`):

| Op | Per row (vectorized over the 32 rows of a col tile) |
|----|------|
| `publish_stats_block` | `M2_k = s2_k − n·(r_k − p_k)²`; emit `r_k`, `M2_k` |
| `combine_block` (core k, frame p_k) | `μ = (1/S)·Σ_j r_j`; `M2 = Σ_j M2_j + n·Σ_j (r_j − μ)²`; `m = μ − p_k`; `rstd = rsqrt(M2·(1/W) + ε)`; emit `m`, `rstd` |
| `normalize_block` | `out = ((x − p_k) − m)·rstd` [`·w + b`]. `x − p_k` is exact (both bf16, FPU, fp32 DEST); the following DestReuse ops see O(std) magnitudes, so srcA tf32 rounding is harmless |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| scaler | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:95-103` (id-only `:113-119`) | `<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>` (pool-type-aware; SUM ⇒ 1.0) | — | `cb_reduce_scaler` | — |
| `mean_block` | helper | `compute_kernel_lib::reduce` | `reduce_helpers_compute.hpp:420-441`; policies `:116`; shape `:147-156`; post-op contract `:319-323` | `<SUM, REDUCE_ROW, cb_input_tiles, cb_reduce_scaler, cb_local_mean, ReduceInputPolicy::WaitUpfrontNoPop>`, `ReduceInputBlockShape::of(bh, bw)`, post-op `mul_unary_tile(dst, bits(1/n_slice))` | `cb_input_tiles`, `cb_reduce_scaler` | `cb_local_mean` | `ReduceInputBlockShape` rows=`block_height`, cols=`block_width` |
| `pivot_block` | helper | `eltwise_chain` + `CopyTile` / `PackTile` | `eltwise/api/chain.hpp:619-620`, `:553-557`, `:587-591` | `IterationShape::tiles(bh)`; input `WaitPolicy::Upfront, PopPolicy::None`; output pack to bf16 | `cb_local_mean` | `cb_pivot` | `tiles(block_height)` |
| `dev_sq_block` (accumulate) | helper | `eltwise_chain` + `BinaryFpu<Sub>` (srcB `input(cb_pivot, BroadcastDim::Col, …)`) + `Square` + `PackTile<output(cb_dev_sq_acc, OneUpfront, OneAtEnd, …, L1Accumulation::Enabled)>` | `chain.hpp:562-573`, `:411-418`, `:297-301`, `:442-450`; `Square` `eltwise/unary/misc.hpp:33`; L1-acc legality `eltwise/core/chain.inl:1133-1136` | one call per tile row: `IterationShape::tiles(bw)`, x input `TileAddressing::Offset` (base = row·bw, Upfront/None), pivot `TileAddressing::Offset` (base = row) | `cb_input_tiles`, `cb_pivot` | `cb_dev_sq_acc` | `tiles(block_width)`; row walk = `block_height` |
| `dev_sq_block` (row reduce) | helper | `compute_kernel_lib::reduce` | `reduce_helpers_compute.hpp:420-441`; `ReduceFp32Mode` `reduce_helpers_common.hpp:18` | `<SUM, REDUCE_ROW, cb_dev_sq_acc, cb_reduce_scaler, cb_dev_sq_acc, WaitAndPopPerTile, INPUT_AND_OUTPUT, ReduceFp32Mode::Accurate>`, `ReduceInputBlockShape::col(bh)`… i.e. `of(bh, 1)` | `cb_dev_sq_acc` | `cb_dev_sq_acc` (in place) | rows=`block_height` |
| `publish_stats_block` | helper | `eltwise_chain` + `CopyTile` (D0..D2) + `SubBinary`, `Square`, `MulUnary`, `SubBinary` + `PackTile` | `chain.hpp:553-557`; `eltwise/binary/sfpu/basic.hpp:20-31`; `eltwise/unary/scalar.hpp:31-32` (`MulUnary`, param = fp32 bits; `scalar.inl:44-56`) | `tiles(bh)` per stat | `cb_local_mean`, `cb_dev_sq_acc`, `cb_pivot` | `cb_local_stats` | `tiles(block_height)` |
| `exchange_block` (host) | helper | `ttnn.Mcast(device, McastConfig(noc, handshake=True), receivers, S, McastRotatingSenderConfig())`, `.attach(descriptor, "stats", [writer], first_sem_id)` | `kernel_lib/mcast/host/mcast.hpp:96-140` (rotating `:114-119`); Python binding `ttnn/cpp/ttnn-nanobind/mcast_host.cpp:48-49,123-135` | `receiver_group_size = S`, RowMajor | — | — | `S` |
| `exchange_block` (device) | helper | `McastArgs<CT_OFF, RT_OFF>`; `sender(noc).send(src, dst, bytes)`, `receiver(noc).receive(round)`, `should_send(round)` | `kernel_lib/mcast/kernel/mcast_args.hpp:19-51,178-181` | round = `block_idx·S + j` | `cb_local_stats` | `cb_gathered_stats` | payload = `2·block_height·TA` |
| `prep_affine_block` | helper | `compute_kernel_lib::unary_bcast<BroadcastDim::Row, …>` | `eltwise/broadcast/bcast.hpp:26-34` | `IterationShape::tiles(bw)` | `cb_weight_rows` / `cb_bias_rows` | `cb_weight_full` / `cb_bias_full` | `tiles(block_width)` |
| `combine_block` | helper (composed) | `eltwise_chain` with `CopyTile` (Upfront/None + `TileAddressing::Offset` into gathered slots), `AddBinary`, `SubBinary`, `MulBinary`, `Square`, `MulUnary`, `AddUnary`, `Rsqrt<Approx::Exact>` | `chain.hpp:553-557`; `binary/sfpu/basic.hpp:20-31`; `unary/scalar.hpp:25-32`; `unary/math.hpp:37-38` | rows walked inside; S peers walked inside | `cb_gathered_stats`, `cb_pivot` | `cb_row_stats` | `block_height`, `S` |
| `bcast_stats_block` | helper | `unary_bcast<BroadcastDim::Col, …>` | `bcast.hpp:26-34` | `tiles(bh)` ×2 | `cb_row_stats` | `cb_shift_full`, `cb_rstd_full` | `tiles(block_height)` |
| `normalize_block` | helper | `eltwise_chain` + `BinaryFpu<Sub, input(cb_input_tiles, Upfront, AtEnd, Block), input(cb_pivot, BroadcastDim::Col, Upfront, AtEnd, Col)>` + `DestReuseBinary<Sub, input(cb_shift_full,…,Col), DEST_TO_SRCA>` + `DestReuseBinary<Mul, input(cb_rstd_full,…,Col), DEST_TO_SRCA>` [+ `DestReuseBinary<Mul, input(cb_weight_full, Upfront, None, Row), DEST_TO_SRCA>` + `DestReuseBinary<Add, input(cb_bias_full,…,Row), DEST_TO_SRCA>`] + `PackTile<output(cb_output_tiles, PerOuter, PerOuter)>` | `chain.hpp:562-583`, `:257-262`, `:518-522`, `:587-591`; DEST_TO_SRCA orientation `core/chain.inl:1510-1513` (CB→srcB, DEST→srcA ⇒ `DEST − CB`) | `IterationShape::grid(bh, bw)` | listed | `cb_output_tiles` | `grid(block_height, block_width)` |
| startup | raw_api | `compute_kernel_hw_startup(cb_input_tiles, cb_reduce_scaler, cb_local_mean)` | required by `reduce_helpers_compute.hpp:306-308`, `chain.hpp:24-40` | first statement of `kernel_main` | — | — | — |

`raw_api` entries: only `compute_kernel_hw_startup` (a caller-owned prerequisite of every helper, not a phase). **Helpers considered and rejected** for the affine pre-broadcast: reader-side row-0 replication of w/b (dataflow RISC-V word copies, ~2 KB stores per tile × 2·bw tiles per core) — rejected in favor of `unary_bcast<Row>` (`bcast.hpp:26-34`) because the copy loop costs more cycles than one compute bcast tile op, and `DestReuseBinary` (`chain.hpp:579-583`) carries no broadcast field, so the operand must be a full tile. If any planned block shape (e.g. `TileAddressing::Offset` with L1 accumulation, Col-mapped `DestReuseBinary`) is rejected by a chain static_assert, the implementer builds the missing block operation (thin custom block helper / raw LLK inside the same named op) and records the gap — never a per-tile loop in the main schedule.

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `dev_sq_block` | Sub | `cb_input_tiles`: All | `cb_pivot`: Col0 (REDUCE_ROW output packed through DEST) | Col |
| `normalize_block` | Sub | `cb_input_tiles`: All | `cb_pivot`: Col0 | Col |
| `normalize_block` | DestReuse Sub | DEST: All | `cb_shift_full`: All (Col-broadcast by `bcast_stats_block`) | None |
| `normalize_block` | DestReuse Mul | DEST: All | `cb_rstd_full`: All | None |
| `normalize_block` | DestReuse Mul / Add | DEST: All | `cb_weight_full` / `cb_bias_full`: All (Row-broadcast by `prep_affine_block`) | None |
| `prep_affine_block` | unary bcast | `cb_weight_rows` / `cb_bias_rows`: Row0 | — | Row |
| `bcast_stats_block` | unary bcast | `cb_row_stats`: Col0 | — | Col |
| `publish_stats_block`, `combine_block` | SFPU binary | Col0 tiles (all operands Col0) | Col0 | None (only column 0 is consumed downstream) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Large common offset (mean ≫ std) | FPU srcB holds tf32 (10-bit mantissa); a plain `x − mean` with an fp32 mean loses `|mean|·2⁻¹¹`, i.e. ~5% of std at mean/std = 100 | bf16 pivot `p` (exact in srcA/srcB) so `x − p` is exact; residual mean `m = μ − p` and `M2` carried in fp32 SFPU; acceptance test with offset 50 and with per-slice offsets |
| tf32 truncation of fp32 stat tiles read by `CopyTile` | without `UnpackToDestFp32` the SFPU combine sees truncated `r_j` (~100 magnitude) | set `unpack_to_dest_mode` for `cb_local_mean`, `cb_dev_sq_acc`, `cb_gathered_stats`, `cb_row_stats` when fp32 DEST |
| In-place transforms without credit | `dev_sq_block` reduces `cb_dev_sq_acc` into itself; a None/None phase would race (`l1-footprint-discipline.md` Rule 3.2) | the reduce uses WaitAndPopPerTile → real pop/reserve/push credit on the same CB |
| Group lockstep | remote writes use the sender's own write-pointer address | rows assigned per group; every member runs identical block sequences; all reserve `2·S·bh` before round 0 |
| Mcast rectangle hits foreign cores | groups must be 1×S rectangles | `S | grid_x`, row-major layout (Mechanism caps) |
| Scaler precision | bf16 cannot hold `1/W` | SUM scaler 1.0 + fp32 `MulUnary` post-op |
| Regime only reachable on some grids | S candidates depend on `grid_x` | `select_blocking` is pure and grid-parameterized; tests pin it on 12×10 and run both regimes on device |
| Per-call weight/bias | adaLN tensors change every call | addresses are RT args; pre-broadcast is per call in-kernel; no host-side caching; acceptance test calls twice with different w/b |
| `fp32_dest_acc_en=False` | acc CBs become Float16_b; combine runs at bf16 DEST precision | allowed and tested at `DEFAULT_TOLERANCES[(bf16, False)]`; no fp32 page under 16-bit DEST (ledger audit 2) |
| Affine row-0 fetch alignment | BH DRAM alignment is 64 B | fetch 64 B (face rows 0–1) at offsets 0 and 512 |
| No-affine path | w/b CBs and accessors absent | `HAS_AFFINE` CT flag; `ttnn.TensorAccessorArgs()` placeholders; w/b CBs not allocated |

## Verifier amendments (verification pass)

| Change | Where | Why |
|--------|-------|-----|
| `fp32_dest_acc_en=False`: per-slice payload is `[p, d, M2]` (`stats_per_row = 3`), with `d = Σ(x−p)/n` from an extra `dev_sum` accumulate + row reduce (`cb_dev_sum_acc`, `cb_dev_sum`); combine uses `δ_j = (p_j − p_own) + d_j`. fp32 DEST keeps `[r, M2]`. | `layer_norm_adaln.stats_per_row`, compute `dev_sum_pass` CT flag | Under 16-bit DEST the first-pass mean `r` is bf16-accumulated (error ≈ \|mean\|·2⁻⁸·√n), so `m = μ − p` was off by ~1 std on large-offset inputs (measured rel-RMS 0.19–0.998 at mean 50 / std 1) — a violation of the prompt's MUST rule. Carrying the slice mean pivot-relative keeps every rounded value O(std). |
| Affine row-0 stream: `AFFINE_STREAM_CHUNK` tiles per NoC barrier, CB = `AFFINE_STREAM_DEPTH` chunks | reader, `affine_stream_pages` | One barrier per tile serialized 2·bw DRAM round trips into the first block's critical path. |
| Writer: `noc_async_writes_flushed` per tile row, one `noc_async_write_barrier` at kernel end | writer | Only L1-source reuse needs guarding before `cb_pop_front`. |
| `GATHER_TILE_CAP` check uses `stats_per_row·S·bh` | `select_blocking` | Cap is in tiles; payload height is mode-dependent. No INPUTS selection changes. |

## Perf 1 amendments (perf tournament round 1)

| Change | Where | Why (measured, see `changelog.md` Perf 1) |
|---|---|---|
| Per-row stats chain: for each tile row r, `wait_front((r+1)·bw)` → mean(r) → pivot(r) → dev_sq(r) [→ dev_sum(r)], instead of all-rows mean, then pivot, then dev_sq | compute | row r's stats overlap the reader's load of row r+1; bit-identical. The per-row mean is raw `reduce_init/reduce_tile/reduce_uninit` (the reduce helper has no tile-base offset into an un-popped CB) |
| `publish_stats_block` / `combine_block` are custom SFPI routines (`kernels/layer_norm_adaln_stats_sfpu.hpp`) on faces 0/2 only; one-pass pivot-relative Chan fold (fp32: `d_j = r_j − p_own`, `M2 = Σ(M2_j + n d_j²) − n S m²`); payload packs two stats per tile on the idle faces 1/3: fp32 `[r | M2]` (1 tile/row), 16-bit `[p | d]`, `[M2 | −]` (2 tiles/row) | compute, host (`stats_payload_tiles` replaces `stats_per_row`) | the S-peer combine dominated small-H shapes (21 µs of 40 µs at 256×3072, S=12) |
| normalize chain `.block_size(DEST_AUTO_LIMIT)` | compute | one init per op per DEST window instead of per tile |
| w / b: the top core of each grid column (writer, NoC1, kernel start) fetches row 0 of the column's slice (every core of grid column x owns slice x % S) with one 576 B read per tile and multicasts it down the column; data-ready flag, no handshake. `cb_weight_rows`/`cb_bias_rows` alias `cb_weight_full`/`cb_bias_full` (bcast runs in place); the reader no longer touches w / b; the `AFFINE_STREAM_*` streams are gone | writer, reader, host | 120 cores re-reading the same w / b pages added ~4.4 MB of DRAM traffic to a DRAM-bound op and starved `prep_affine_block`. Supersedes the deferred `affine_mcast` regime above (receiver set is a grid column, a rectangle) |
| `select_blocking` cost = `core_h·bw + has_affine·bw + (C_EXCHANGE + C_COMBINE·stats_payload_tiles)·S·core_h` (C_EXCHANGE = 2.0, C_COMBINE = 0.75; fitted on a forced-S sweep); `LN_ADALN_FORCE_SLICES` experiment hook | host | tiles-only cost picked S=12 where the S-way exchange + combine dominate; 32×3072 and 256×3072 now pick S=6 |
| L1 budget = `ttnn._ttnn.reports.get_device_info(device).cb_limit` (`l1_cb_budget_bytes`) | host | `get_max_worker_l1_unreserved_size()` measures from the KERNEL_CONFIG base; CBs start ~70 KB higher, so large multi-block blockings overflowed L1 |
| Permanent `MaybeDeviceZoneScope` stage zones (`r_*`, `c_*`, `w_*`), compiled in only with the profiler and `LN_ADALN_PERF_ZONES=1` | all kernels | per-stage breakdown; never remove |

Updated selections on 12×10: (1,1,32,3072) and (1,1,256,3072) now S = 6 (was 12); every other row of the selection table is unchanged.

## Perf 2 amendments (perf tournament round 2)

| Change | Where | Why (measured, see `changelog.md` Perf 2) |
|---|---|---|
| `normalize_block` is one raw loop per DEST window: FPU `sub_tiles_bcast<COL>` (x − p), then dest-reuse (DEST→srcA) binaries whose srcB operand is **hardware-broadcast** straight from the compact operands — m / rstd Col from `cb_row_stats`, w / b Row from the row-0 tiles in `cb_weight_rows` / `cb_bias_rows`. `prep_affine_block`, `bcast_stats_block`, `cb_shift_full`, `cb_rstd_full`, `cb_weight_full`, `cb_bias_full` are deleted | compute, host | `DestReuseBinary` has no broadcast field, so the broadcasts had to be materialized as full tiles (`bcast_stats_block` sat on the critical path between combine and normalize). Bit-identical |
| `cb_output_tiles` is pushed per DEST window (not per tile row); the writer waits cumulatively per tile and writes each tile as it lands (flush + pop per row) | compute, writer | the writer idled until a full row (32 tiles) was normalized; writes now stream alongside normalize. Bit-identical |
| Per-core DRAM bank stagger: core L visits its row's `bw` tiles from column `rot = (L % num_dram_banks) % bw` and wraps — reader, normalize and writer all use that order (RT args: reader[4], writer[4] (affine args move to [5..]), compute[2]) | all kernels, host | every row starts on bank 0 (96 and 32 are multiples of BH's 8 banks), so all 120 cores hit bank 0 first and walked the banks in lockstep. Bit-identical |
| `footprint_bytes` no longer counts `cb_shift_full` / `cb_rstd_full` (2·bh acc tiles) | host | they no longer exist; selections for every INPUTS shape are unchanged (only L1-limited 23040-row shapes pick a taller block) |

## Perf 2 amendments (4th tournament: second round of the current series)
- **Stats chain (stats_chain).**
  - The reader pushes each tile row in chunks of `chunk = min(bw, max(STATS_CHUNK=4, ceil(bw/15)))` tiles, one NoC read trid per chunk.
  - Pivot p = bf16(mean of the first chunk). It no longer waits on the full-row mean.
  - One merged pass computes Σ(x−p) and Σ(x−p)² per row in both DEST modes. It replaces mean → pivot → dev_sq [→ dev_sum].
  - Publish (fp32): r = p + d, M2 = s2 − Σ·d (`cf_publish_fp32_dsum`). The payload format is unchanged.
  - This fixes the 16-bit DEST large-offset failure (offset 1000: PCC 0.46–0.70 → ≥ 0.99997). The cause was the bf16-accumulated full-row pivot.
- **Normalize (fp32 DEST, normalize_tile_cost PM2).**
  - D = c′ (SFPU LREG preload), then D += x·rstd: one FPU ELWMUL accumulating into DEST, srcB = rstd Col-bcast.
  - Then the affine ·w + b as before.
  - 16-bit DEST keeps sub/sub/mul, because c′ does not survive a 16-bit DEST.
- **Affine column multicast** is linked: every data multicast but the last keeps the path reserved. `w_affine` 14.4 → 6.9 µs on the focus.
- **Per-core NoC swap (noc_swap).**
  - The `NOC_SWAP_EAST_COLS = 2` easternmost grid columns run reader on NoC1 and writer on NoC0. These are separate reader/writer `KernelDescriptor`s per NoC set.
  - Affine rectangles are ordered per injector NoC.
  - It is disabled on the rotating `ttnn.Mcast` path, which is inexpressible there: one NoC per channel.
  - It is also disabled by a measured carve-out: affine + 16-bit DEST with `core_h_tiles ≥ 4`.

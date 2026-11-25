# Operation Design: groupnorm_sc_N_1_HW_C_b

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (two-pass statistics → cross-core combine → per-channel affine normalize) |
| Goal | GroupNorm over an `(N, 1, HW, C)` activation with channels last. One device-program dispatch per call; multi-core from Phase 0; arbitrary group width `Cg = C/G`, including `Cg % 32 != 0`. Group masks are generated on device. |
| Math | `mean[n,g] = Σ x[n,0,hw,c] / (HW·Cg)` and `var[n,g] = Σ x² / (HW·Cg) − mean²`, each summed over `hw ∈ [0,HW)` and `c ∈ group g`. `rstd = 1/sqrt(var + eps)`. `y[n,0,hw,c] = x·a[n,c] − b'[n,c]`, where `a = rstd[n,g(c)]·γ[c]` and `b' = mean[n,g(c)]·a[n,c] − β[c]`. |
| References | `eval/golden_tests/groupnorm_sc_N_1_HW_C_b/{feature_spec.py,helpers.py}` (TARGET, INPUTS, reference `pytorch_groupnorm_sc_N_1_HW_C`, TOLERANCES); `eval/op_template.py` (registry model); `.claude/references/{blocking-model.md,l1-footprint-discipline.md,ttnn-cb-memory-fundamentals.md,generic_op_template/}`; `ttnn/cpp/ttnn/kernel_lib/` helpers (cited per row in API Mapping); catalog `ttnn/ttnn/operations/examples/master.md` (entries cited in Perf lamps); `l1_ledger.md` (this directory). |

The entry point `groupnorm_sc_N_1_HW_C_b(input_tensor, num_groups, *, gamma=None, beta=None, eps=1e-5, in_place=False)` builds one `ttnn.ProgramDescriptor` and makes one `ttnn.generic_op([input, (gamma), (beta), output], desc)` call. It chains no other TTNN op and makes no host round trip. It must not call `ttnn.group_norm`.

### Registry contract (`groupnorm_sc_N_1_HW_C_b.py`)

| Item | Decision |
|------|----------|
| `INPUT_TAGGERS` | Copy verbatim from the task statement: `_placement`, `_ceil`, `tag_alignment`, `tag_memory_layout`, `tag_shard_fit`. |
| Axis names | Exactly the TARGET keys in `feature_spec.py`: `dtype`, `layout`, `alignment`, `affine`, `affine_dtype`, `affine_layout`, `memory_layout`, `shard_fit`, `in_place`. The op has no index (`dim`) axis. |
| `validate()` | The first line of the entry point. Order: (1) the structural `ValueError`s listed under Parameters; (2) derive the axes from the call: `memory_layout` from `input.memory_config().memory_layout`, `shard_fit` from `input.shard_spec()` (via `tag_shard_fit` geometry), `alignment` from the shape, `affine` / `affine_dtype` / `affine_layout` from gamma/beta, using the `"none"` sentinels when absent; (3) the SUPPORTED / EXCLUSIONS check, which raises `NotImplementedError` (`UnsupportedAxisValue` / `ExcludedCell`). |
| Phase 0 `SUPPORTED` (implementer writes it) | dtype `[bf16]`; layout `[TILE, RM]`; alignment `[tile_aligned]`; affine `[gamma_beta, gamma_only, no_affine]`; affine_dtype `[bf16, "none"]`; affine_layout `[RM, "none"]`; memory_layout `[INTERLEAVED]`; shard_fit `["none"]`; in_place `[False]`. |
| Shape-class gate inside validate | If no column-slice count `gx` satisfies the feasibility predicate (Work Distribution), raise `NotImplementedError`. This is the `cross_column_group_combine` regime row. No INPUTS shape hits it. |
| `TUNING_OVERRIDES` | A module-level dict exported from the package (`ttnn.operations.groupnorm_sc_N_1_HW_C_b.TUNING_OVERRIDES`), mutated in place, never rebound: `{"max_resident_chunks": None, "chunk_tiles_target": None, "max_cores": None}`. `None` means use the derived default. It exists so regime-pinned tests can force every regime on any grid. Every override must flow into the program descriptor (CB sizes, RT args). It must not be cached on shape alone. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | Rank 4 with `shape[1] == 1`, else `ValueError`. bf16. TILE or ROW_MAJOR. DRAM interleaved in Phase 0. | — | buffer address → RT |
| `num_groups` (G) | int | yes | `1 ≤ G`, `C % G == 0` (else `ValueError`) | — | host only. Derives `Cg`, the group map, `inv_n`. |
| `gamma` | `ttnn.Tensor` or None | no | Shape `(1,1,1,C)` (else `ValueError`). DRAM. bf16 RM in Phase 0. | None | presence → CT `HAS_GAMMA`; address → RT |
| `beta` | `ttnn.Tensor` or None | no | same as gamma | None | presence → CT `HAS_BETA`; address → RT |
| `eps` | float | no | `> 0` | `1e-5` | RT (fp32 bits, root compute) |
| `in_place` | bool | no | `True` only with a BLOCK_SHARDED input (`ValueError` on interleaved) | False | host (R2) |
| `inv_n` (derived) | float | — | `1/(HW·Cg)` | — | RT (fp32 bits, root compute) |

The output dtype always equals the input dtype. There is no `memory_config`, `core_grid`, mask, `num_out_blocks` or `compute_kernel_config` parameter.

Each structural `ValueError` message contains the keyword below. The acceptance test matches it case-insensitively through the `expect_error` fixture.

| Check | Message must contain |
|-------|----------------------|
| rank ≠ 4 | `rank` |
| `shape[1] ≠ 1` | `shape[1]` |
| `C % G ≠ 0` | `num_groups` |
| gamma not `(1,1,1,C)` | `gamma` |
| beta not `(1,1,1,C)` | `beta` |
| `in_place=True` on a non-sharded input | `in_place` |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(N, 1, HW, C)`. Phase 0 requires `HW % 32 == 0` and `C % 32 == 0` (`tile_aligned`). |
| Dtype | bfloat16 (Phase 0) |
| Layout | TILE or ROW_MAJOR |
| Memory | DRAM interleaved (Phase 0). L1 BLOCK_SHARDED with ROW_MAJOR orientation comes in R1–R3. |

### Output

| Property | Value |
|----------|-------|
| Shape | same as input |
| Dtype | same as input |
| Layout | Interleaved input gives TILE, whatever the input layout. A sharded input keeps its own layout. |
| Memory | Interleaved input gives DRAM interleaved. A sharded input gives L1 BLOCK_SHARDED with the input's shard spec. With `in_place`, the output occupies the input's buffer address. |

## Blocking Model

The first design section; everything below is its realization. Semantics: `.claude/references/blocking-model.md`.

**Symbols** (all derived on the host from the shape and one knob each):

- `Ht = ceil(HW/32)` per image and `Wt = ceil(C/32)`.
- `Cg = C/G`.
- `span_tiles = lcm(Cg,32)/32`. This is the smallest whole-tile channel run that contains only whole groups.
- `gps = lcm(Cg,32)/Cg` groups per span, and `S = Wt/span_tiles` spans per row. In Phase 0, `lcm(Cg,32) | C`.
- `gx` is the number of column slices per image; it divides `S`.
- `Wt_core = (S/gx)·span_tiles` and `G_core = (S/gx)·gps`.
- `U = N·gx` units, and `P` cores (the grid, capped by `max_cores`).
- `cpi` cores per unit; `rows_core ∈ {q, q+1}` where `q = Ht // cpi`.
- `h = chunk_ht`, `T = h·Wt_core` tiles per chunk, `B = ceil(rows_core/h)` chunks per block, `K` resident chunks, and `R = ceil(cpi/32)` gather tile-rows.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `n` (image) | **independent**: each image's groups normalize alone. | `images_per_unit` | 1. A unit is one image's column slice. The extent is 1 because a unit spans exactly one image's statistics. Spreading the other axes provides the occupancy. | fixed by the unit definition (host `_plan_units`) | Units `u = n·gx + xs`. Spread over cores together with `xs`. | — (already split) |
| `xs` (column span slice, `S` spans per row) | **independent**: a span holds only whole groups, so slices share no statistic. | `gx` (slices per image) → `Wt_core` | Chosen by the host: minimize max per-core tiles, subject to `G_core ≤ 32` and the L1 feasibility predicate. Tie-break: TILE takes the larger `gx` (smaller fan-in, smaller channel vectors); RM takes the smaller `gx` (wider stick segments). | host `_choose_gx()` → CT `WT_CORE` | One unit per `(n, xs)`. | knob-turn |
| `c∈g` (channels inside one group / span) | **dependent**: the group statistic spans `Cg` channels. | `span_tiles` (fixed by `Cg`) | Whole span on one core. A span is never split. | derived from `Cg` | Not split across cores. | scheme-change (`cross_column_group_combine`, deferred) |
| `ht` (tile rows of one image) | **dependent** for the statistics (they span all HW rows); **independent** for the normalize. | `cpi` (row-split ways), `chunk_ht` (streaming quantum), `K` (resident chunks) | `cpi = min(floor(P/U), Ht)` when `U < P`, else 1. `chunk_ht = clamp(floor(CHUNK_TILES_TARGET/Wt_core), 1, rows_core_max)` with `CHUNK_TILES_TARGET = 32`. `K = min(B, K_max)`. | host constants `CHUNK_TILES_TARGET`; `K_max` closed form (`l1_ledger.md`); `TUNING_OVERRIDES` | Balanced `cpi`-way split of each unit's `Ht` (the first `Ht % cpi` cores take `q+1`). Partials combine on the unit's root. | knob-turn (combine already built) |
| `g` (group lane — data of the combine stage) | **independent** across lanes, but all `G_core ≤ 32` lanes fit in one tile row. | `G_core` | One tile (all lanes at once) | derived | On the unit's root only. The finalize is one matmul plus one SFPU chain over 2 tiles, so splitting a single-tile op gains nothing. The root's role is communication. The non-root cores' wait on this stage is the combine round trip, recorded in the grid-sync lamp. | — |
| `i` (sender slot — data of the gather stage) | **dependent**: the root sums all senders. | `R = ceil(cpi/32)` | All `cpi` partial rows in one gather buffer, one matmul over R tile-rows | derived from `cpi` | Root only (see `g`) | tree combine = perf lamp |
| γ/β (channel vectors) | **reuse-shared**: they do not vary along `ht`, the axis split across a unit's `cpi` cores. | `Wt_core` (one slice per core) | Each core reads its own `Wt_core·64 B` slice from DRAM. | derived | every core | `affine_mcast` (deferred) |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_stream` (TILE) | `D_IN` | 2 chunks | Lets the DRAM read of chunk j+1 overlap compute on chunk j. Catalog `double_buffer`: CB = 2·block, with many reads per barrier. |
| `cb_input_stream` (RM) | — | 1 chunk | tilize→reduce is a sequential producer/consumer pair inside compute, so depth buys nothing. The DRAM-side overlap lives in `cb_input_rm`. |
| `cb_input_rm` | `D_RM` | 2 chunks | Overlaps the stick reads with tilize. |
| `cb_input_resident` / `_keep` | `K` | `min(B, K_max)` chunks | Residency: K chunks are read from DRAM once instead of twice. |
| `cb_output_tiles` | `D_OUT` | 2 chunks | Lets the DRAM write of chunk j overlap the normalize of chunk j+1. |
| `cb_x_squared` | — | 1 chunk (aliased inside `cb_output_tiles`' bytes) | Sequential helper intermediate (square → reduce): it must hold one full reduce block. |
| all handoff/stat CBs | — | exactly one block's quantum | The quantum equals the capacity, so every block starts at the CB base. Remote writers (gather, mcast) depend on that invariant. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Group lanes in one tile row (collapse/expand matmul N/K = 1 tile; gather row = 32 lanes) | `G_core ≤ 32` | `_choose_gx` admits only `gx` with `(S/gx)·gps ≤ 32` | Groups past lane 31 alias other lanes or vanish, which gives wrong statistics silently. |
| L1 capacity of the per-channel vectors and fixed CBs | `Wt_core` | `_choose_gx` admits only `gx` with `F_fixed(Wt_core, T, R) ≤ L1_budget` (`l1_ledger.md`). If none qualifies, raise `NotImplementedError`. | CB allocation overflow (an error, not wrong results). |
| Gather slot addressing | `cpi` | `R = ceil(cpi/32)` tile pairs. Slot `i` goes to tile pair `i/32`, row `i%32`. Rows ≥ `cpi` are zero-filled by the root. | Stale rows are summed into the statistics. |
| DEST capacity (fp32, half-sync: `DEST_AUTO_LIMIT = 4`) | The finalize chain uses 3 slots (D0–D2). The reduce/matmul/chain helpers clamp their own DEST blocking. | static 3 ≤ 4 | The compile-time `static_assert` in chain.hpp `Dst` fails. |
| Matmul helper CB contract (`in0`, `in1`, `out` distinct) | collapse / expand / sum-senders matmuls | The outputs go to `cb_local_partials` / `cb_chan_work` / `cb_chan_scale` / `cb_group_sums`, never to an input CB. | FIFO corruption (matmul_block_helpers.hpp:242-243). |
| Ring-wrap (quanta must sum to capacity) | All chunked CBs | Every producer pushes the **nominal** chunk count `T` (ragged tail included). Consumers pop `T` and touch only the valid `rows_valid·Wt_core` tiles. | A wrap straddle corrupts the next block silently. |
| `UnpackToDestFp32` accumulator CBs | `cb_sum_acc`, `cb_sumsq_acc` | Only `copy_tile` (the reduce helper's reload) may read them. The last chunk's reduce writes to a non-U2D CB instead. | Reading them through SrcA/SrcB is invalid (reduce_helpers_compute.inl:325-326). |
| Tilize block width is a template parameter | `Wt_core` | `WT_CORE` is a CT arg | — (compile error) |
| DRAM read alignment (RM sticks, RM affine) | segment offset `wt0·64 B` | Slices start on tile (64 B) boundaries, so every segment is 64 B aligned. | Misaligned DRAM read. |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `dram_resident` | **built** | Interleaved input and `B_max ≤ K_max(Wt_core, T, R)`: every core's whole block fits L1. | One core's `(n, rows [r0, r0+rows_core), Wt_core cols)` | **Minimum.** The input crosses DRAM once, the output once, and γ/β once per core (`Wt_core·64 B` each). The cross-core traffic is 256 B per core gathered plus one 8 KB stats multicast per unit. | Per block, a fixed cost paid **once**: 4 matmul calls, 3 W-tile chains, one gather round trip and one mcast. Per chunk: one Accumulate reload of `2·Wt_core` fp32 tiles plus one read barrier. A bigger chunk (`CHUNK_TILES_TARGET`) amortizes the per-chunk costs. |
| `dram_partial_resident` | **built** (same kernels, `K < B` as an RT arg) | Interleaved input and `B_max > K_max` | same | The input crosses DRAM `1 + (B−K)/B` times; the output once. The minimum is structurally unreachable here: every core already holds a slice and the input exceeds the aggregate free L1 (`P·K_max·T·2 KB`). | More resident chunks (larger `K`) remove re-reads one-for-one. The same per-block and per-chunk amortization applies as in `dram_resident`. |
| `block_sharded_shard_resident` | **deferred** to R1 (even, out-of-place), R2 (in_place), R3 (ragged). Reason: the Phase 0 SUPPORTED rectangle is interleaved, and the queue orders R1–R3 first. | `memory_layout == BLOCK_SHARDED` | The shard: rows `[y·sh, (y+1)·sh)` × channels `[x·sw, (x+1)·sw)`. Default **one block = the whole resident shard**. | Zero DRAM traffic for the input and output; cross-core traffic is the same gather and mcast along each grid column. | The shard fixes the block. Chunking stays the same knob, but the "resident" chunks are the shard itself (`borrowed_from`), so `K = B` for TILE. |
| `cross_column_group_combine` | **deferred** to R3 (groups straddling core columns). Reason: no Phase 0 / INPUTS interleaved shape needs a span wider than the L1-feasible `Wt_core`. | Sharded `sw % Cg ≠ 0`, or no feasible `gx` | A partial span per core | The gather fan-in becomes 2-D (column neighbours plus row neighbours); the bytes per core are unchanged. | — |
| `two_pass_centered_variance` | **deferred**. Reason: the exact fp32 E[x²]−mean² (U2D accumulators, one TF32 rounding per partial) meets TOLERANCES and PERF_TOLERANCE on the zero-mean golden inputs. The centered pass is a precision upgrade for large-mean data. | Precision mode flag | same block | Free in DRAM traffic for resident chunks. Non-resident chunks are re-read once more. One more gather/mcast round. | — |
| `affine_mcast` (γ/β broadcast) | **deferred**. Reason: the γ/β bytes per core are `2·Wt_core·64 B`, at most 0.2% of a core's input `T·B·2 KB` for every INPUTS shape. | — | — | Removes `(cpi−1)/cpi` of the γ/β DRAM reads. | — |
| `three_pass_streaming` (mean pass, var pass, normalize pass) | **rejected**. Superseded by the single Σx/Σx² pass, which removes one full input crossing. It is a dead end: no better scheme builds on it. | — | — | 3 input crossings | — |
| `whole_image_per_core` (no row split, `cpi = 1` forced) | **rejected** as a standalone regime. For `U < P` it idles `P−U` cores and makes every core's block un-resident (2 crossings). Its useful domain (`U ≥ P`) is the `cpi = 1` case of the two built rows. | — | — | 2 crossings | — |

**Single-regime-pair note.** The two built rows are one algorithm. The **regime-selection function** is `K = min(B, K_max)` per core, with `K_max = floor((L1_budget − F_fixed)/(T·2 KB))` (closed form in `l1_ledger.md`). The stream CB is allocated iff `B_max > K_alloc`. The host may override `K` with `TUNING_OVERRIDES["max_resident_chunks"]` (clamped to `[0, K_max]`).

### Traffic ranking

| Candidate split | DRAM bytes (input / output) | Cross-core bytes | Fan-out | Verdict |
|-----------------|----------------------------|------------------|---------|---------|
| `n × xs × ht`, row split plus combine, tail residency (**chosen**) | `X·(1 + Σ(B−K)/ΣB)` / `X` | `cpi·256 B` gather plus `8 KB` mcast to `cpi` cores per unit | gather `cpi:1`, mcast `1:cpi` | Cheapest. Splitting the dependent `ht` axis fills the grid *and* shrinks the per-core slice so it can stay resident; the combine costs bytes that are negligible next to one avoided re-read. |
| `n × xs` only (`cpi = 1`) | `X·2` whenever a slice exceeds L1 (every perf case) / `X` | 0 | — | Rejected: more DRAM bytes and `P − U` idle cores. It is kept only as the `U ≥ P` case. |
| Split `c∈g` across cores (span split) | same as chosen | 2-D gather | larger | No byte saving. Deferred (`cross_column_group_combine`). |
| `gx` variants inside the chosen split | identical DRAM bytes | Larger `gx` → fewer cores per unit → smaller gather | — | Occupancy and the `Wt_core` L1 decide this (Work Distribution), not traffic. |

`X` is the input bytes. Traffic ranks the split first; occupancy only orders `gx` inside it.

**Stall-shadow check.**

- **Reader during the combine.** The writer drives the combine (gather → root → mcast); compute waits on `cb_group_stats`. The reader does not depend on the statistics, so in that window it:
  1. produces the γ/β row tiles for this block;
  2. prefetches the first `D_IN` pass-2 re-read chunks (`K < B`).
- **Compute during the combine (RM input).** Compute tilizes the first re-read chunk into `cb_input_stream` before it blocks on `cb_group_stats`.
- **Pass-2 order.** Pass 2 processes the **resident chunks first** (zero DRAM latency), while the re-reads stream in behind them.
- **No algebraic reorder of normalize around the combine is possible.** Every output element depends on the group statistics. The algebraic fold `y = x·a − b'` (`a = rstd·γ`, `b' = mean·a − β`) moves all per-channel math out of the pass-2 hot loop, leaving two FPU ops per tile. This changes the floating-point order relative to the reference `(x−mean)·rstd·γ+β`. The difference is bounded by one fp32 rounding per channel and is covered by TOLERANCES.

### Block schedule

Logical schedule per core (reader, compute and writer are separate asynchronous kernels):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // >1 only when cpi == 1
    load_block(block_idx);                  // reader: pass-1 chunks, affine rows, pass-2 re-reads
    accumulate_moments(block_idx);          // pass 1 over B chunks → per-channel Σx, Σx²
    collapse_to_group_partials(block_idx);  // per-channel → per-group local partials (1 tile row, 2 moments)
    combine_group_stats(block_idx);         // gather to unit root → sum senders → finalize → mcast [rstd, mean]
    expand_stats_to_channels(block_idx);    // per-group → per-channel a = rstd·γ, b' = mean·a − β
    normalize_block(block_idx);             // pass 2: resident chunks, then re-read chunks: y = x·a − b'
    store_block(block_idx);                 // writer: output tiles to DRAM
}
```

| Block operation | Block shape it acts on | Resident across it | Intended frequency of inits / multicast / barriers |
|-----------------|------------------------|--------------------|---------------------------------------------------|
| `load_block` | Pass 1: B chunks of `h × Wt_core` tiles. Pass 2: `B−K` chunks. γ/β: `Wt_core` tiles each. | The group map, scaler and ones tiles (generated once per kernel) | One read barrier per chunk (T reads in flight; RM: `32·h` stick segments). Affine: one barrier per vector per block. |
| `accumulate_moments` | Per chunk: `rows_valid × Wt_core` tiles. Accumulates over the B chunks. | The `K` resident chunks (in `cb_input_resident_keep`); the per-channel accumulators | Per chunk: 1 square chain + 2 reduce calls (one Accumulate reload each, except chunk 0). RM: +1 tilize per chunk. |
| `collapse_to_group_partials` | `1 × Wt_core` per-channel row × the `Wt_core × 1` map → 1 tile, for each moment | map | Once per block: 2 matmul calls. |
| `combine_group_stats` | `cpi` partial rows (2 moments × 32 lanes) → 2 replicated stats tiles | — | Once per block: one gather (4 × 64 B writes + 1 semaphore inc per sender), one root matmul + one finalize chain, one mcast of 8 KB. |
| `expand_stats_to_channels` | 2 stats tiles × the `1 × Wt_core` transposed map → `Wt_core` tiles; 3 chains over `Wt_core` tiles | map; `cb_chan_scale` (a) persists into pass 2 | Once per block: 2 matmul calls + up to 3 chains. |
| `normalize_block` | Per chunk: `rows_valid × Wt_core` tiles. Resident chunks first, then streamed ones. | `a` (`cb_chan_scale`) and `b'` (`cb_chan_work` / `cb_chan_shift`) for all of pass 2 | One chain per chunk (one init per chunk). RM re-reads: +1 tilize per streamed chunk. |
| `store_block` | Per chunk: `rows_valid × Wt_core` tiles | — | One write flush per chunk; one write barrier at the end of the block. |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap**: chunk size `CHUNK_TILES_TARGET = 32` | 32 is the catalog plateau (`double_buffer`; `compute_block_size`: a whole-block call beats a per-tile-row loop by 1.65×). The reduce reload of `2·Wt_core` fp32 tiles per chunk argues for larger chunks; the double buffer argues for smaller ones. | 16, 64, 128 |
| **Grid-sync**: `cpi` up to `P` cores per unit | Small blocks (the sharded unet cases, the small DRAM shapes) leave a few tiles per core, so the gather/mcast round trip dominates (catalog `tensix_all_reduce`: about 1 µs per root reduce plus mcast round on a line). | Cap `cpi` (for example `cpi ≤ 16`); a two-level tree gather (`tensix_all_reduce` tree variant). |
| Stream depth `D_IN = 2` | DRAM latency on BH with 110 readers in flight | `D_IN = 3, 4` |
| Math fidelity `HiFi3` | HiFi3 is the lowest fidelity that is exact for TF32 operands in SrcB (the matmul in0 statistics, `a` in pass 2) multiplied by 1.0 or bf16. It also avoids the HiFi4 + fp32-DEST issue (matmul_block_helpers.hpp:263-264, #38306). Pass 2 FPU time scales with the fidelity phases. | `HiFi2` (drops SrcB low mantissa bits of `a` ≈ 2⁻⁸ relative; must still meet PERF_TOLERANCE) |
| `cb_x_squared` page Float32 | It doubles the bytes of the x² intermediate relative to bf16. A bf16 x² frees L1 for about one more resident chunk per two. | `Float16_b` x² (precision vs residency) |
| `cpi` uniform per unit | When `P % U ≠ 0`, up to `U−1` cores idle (mochi: 104/110). | Variable `cpi` per unit (`floor` / `ceil` mix) |
| `gx` tie-break | The larger `gx` for TILE (smaller gather, smaller channel vectors) is a structural guess. | Measure the other tied `gx`. |
| Pass-2 fusion (`BinaryFpu<Mul>` + `DestReuseBinary<Sub>` in one chain) | Catalog `compute_fusion`: FPU DEST-reuse fusion measured 0.82–1.02×. | Two unfused passes |
| Reader affine generation by RISC-V word copies | Fine in the stall shadow on DRAM. It sits on the critical path only if the shadow is short. | Local NoC copies from staging |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Reduce scaler | bf16 tile | reader: `prepare_reduce_scaler<cb_reduce_scaler, SUM, REDUCE_COL>(1.0f)` | Pool-type-aware overload. Once per kernel. |
| Group map M | bf16, `Wt_core` tiles | reader: `zero_tile` + 32 halfword stores of 1.0 per tile at `(r, l)`, where `l = ((wt·32 + r) / Cg)` relative to the slice start (`r` = channel row, `l` = group lane, face-layout offset) | Block-invariant: every slice starts on a span boundary, so one map serves every block. Pushed once and never popped. Channels outside the slice (R6 padding) get no 1 entry, which masks them. |
| Ones tiles (root) | bf16, R tiles | reader: `fill_l1_range` with `0x3F803F80` | Pushed once per block on roots. The bytes persist, so later blocks re-push without rewriting. |
| Input, TILE, pass 1 | bf16 tiles | reader: TensorAccessor `noc_async_read` per tile (id `(n·Ht + ht)·Wt + wt0 + wt`) into `cb_input_stream` (chunks `0..B−K−1`) or `cb_input_resident` + `_keep` (chunks `B−K..B−1`, reserved and pushed on **both** alias views) | One barrier per chunk. The nominal T is pushed even for the ragged last chunk. |
| Input, RM, pass 1 | bf16 sticks | reader: one segment read per stick (`Wt_core·64 B` at byte offset `wt0·64` of page `n·HW + row`) into `cb_input_rm`, 32·h sticks per chunk | compute: `tilize<WT_CORE>` into `cb_input_stream`, or into `cb_input_resident` with a manual reserve/push of `_keep` (both views pushed in lockstep). |
| Pass-1 compute | bf16 → fp32 DEST | square chain → `cb_x_squared` (Float32); `reduce<SUM, REDUCE_COL, ReduceTile>` with Accumulate on x and on x² | Accumulators in `UnpackToDestFp32` CBs give an exact fp32 reload. The last chunk's reduce writes to `cb_chan_scale` / `cb_chan_work` (FPU-readable). |
| Collapse | fp32 → fp32 | `matmul_block` (in0 per-channel row, in1 map) ×2 → `cb_local_partials` | Row 0 of each output tile = 32 group lanes. |
| Gather (Tensix → root) | fp32 rows | writer: 4 × 64 B `noc_async_write` (row 0 of faces 0 and 1 of the 2 partial tiles) into the root's `cb_gathered_partials` at slot `i`, then `noc_async_writes_flushed`, then `noc_semaphore_inc(root_gather_sem, 1)` | The root writer zero-fills the rows ≥ `cpi`, waits with `noc_semaphore_wait_min(sem, cpi)`, resets the semaphore and pushes 2R tiles. The CB base is identical on every core, and the quantum equals the capacity, so the remote slot address is `base + offset`. |
| Root sum and finalize | fp32 | compute (root): `matmul_block(ones[1×R], gathered[R×2])` → `cb_group_sums` (replicated rows); SFPU chain → `cb_stats_out` `[rstd, mean]`, replicated | `var = max(S2·inv_n − mean², 0)`. |
| Stats mcast (root → unit) | fp32, 2 tiles (8 KB) | writer: `ttnn.Mcast` channel (`McastFixedSenderConfig(sender_index=0)`, `group_size = cpi`) with `sender.send(src, cb_group_stats base, 8192)`; receivers `receiver.receive()` | The root's own copy goes through a local L1 copy (or the helper's loopback if its group includes the sender). Each writer then pushes 2 tiles into `cb_group_stats`. With `cpi = 1` the copy is local only. |
| Expand | fp32 | `matmul_block<transpose=true>(stats tile, map)` → `Wt_core` per-channel tiles, rows replicated; chains for `a` and `b'` | Rows are replicated because the stats tiles are. |
| γ/β rows | bf16, `Wt_core` tiles | reader: one DRAM read of the stick slice into `cb_affine_staging`; per tile, `zero_tile` + 16 word copies into row 0 of faces 0/1 | Used with `BroadcastDim::Row` only, so row 0 suffices. A TILE affine (later) is a direct tile read. |
| Pass-2 compute | bf16 x, fp32 a, b' → bf16 | chain: `BinaryFpu<Mul>(x, a)` → `DestReuseBinary<Sub>(b', DEST_TO_SRCA)` → `PackTile(cb_output_tiles)` | Resident chunks first, then streamed chunks. A nominal T is pushed per chunk. |
| Output | bf16 tiles | writer: TensorAccessor `noc_async_write` per valid tile, same tile ids as the input | The writer follows the chunk order `[B−K .. B−1, 0 .. B−K−1]`. |

### TARGET placement values as regime rows

| TARGET value | Logical shard it realizes | Cost class (from this op's axis characters) | Realization |
|--------------|---------------------------|---------------------------------------------|-------------|
| `memory_layout=INTERLEAVED` | — | — | Phase 0 (`dram_resident` / `dram_partial_resident`). |
| `BLOCK_SHARDED`, `shard_fit=even`, `sw % Cg == 0` (R1) | Grid columns = `xs` slices (group-aligned); grid rows = the `ht` split (dependent). | **Knob-turn.** The cut along `ht` crosses the dependent axis, but the Phase 0 combine already exists. The unit becomes one grid column per image, with `fan_in = gy / N` per image (or `gy` for N = 1). The map is relative to the slice start, so a slice that is not tile-aligned (e.g. 120 channels, Cg = 30) is masked by the map. | TILE: `cb_input_resident` and `cb_input_resident_keep` are `borrowed_from` the input shard (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy, `K = B`); `cb_output_tiles` is borrowed from the output shard. RM: `cb_input_rm` is borrowed from the shard (all sticks present, no reader NoC traffic); compute tilizes into resident tiles (`K = min(B, K_max)`, re-tilizing streamed chunks from the shard in pass 2) and untilizes (`untilize_helpers.hpp:163`) into the output-shard-backed CB. No `TensorAccessor` reads of the shard. |
| `in_place=True` (R2) | same shard | **Knob-turn** (aliasing) | The output-shard CB is the input-shard CB's bytes (same buffer). Pass 2 writes chunk j only after chunk j has been read (resident tiles first; streamed chunks re-tilize before their rows are overwritten). RM 16384×640 and 16384×960 on 8×8 must fit: the shard is 2048×120 RM = 480 KB, and the tiles are resident if `K_max ≥ B`, otherwise re-tilized. |
| `shard_fit=ragged` (R3) | Padded last row/column; shards straddling images; groups straddling columns | Padding rows/columns: **knob-turn** (the ragged-chunk mechanism plus the map channel bound). Straddled images: **scheme-change** (a core contributes to two units, giving two combine rounds with different roots). Straddled groups: **scheme-change** (`cross_column_group_combine`). | Deferred rows above. |

The other refinements map onto this structure as follows. **R4** fp32: the CB formats follow the dtype and `fp32_dest_acc_en` stays on; fp32 x and γ/β pages. **R5** `hw_non_aligned`: the chunk containing an image's last tile-row uses `ReducePartialScaler::with_partial()` (reduce_helpers_compute.hpp:350) on both the x and x² reduces, `inv_n` uses the true HW, and RM reads skip the pad sticks. **R6** `c_non_aligned`: the map's channel bound masks the pad channels, `S = ceil`, and the last slice is partial. **R7** bf8b: TILE input and affine pages in Bfp8_b, with `cb_x_squared` and all statistics staying Float32.

### Structural impossibilities

None beyond `feature_spec.INVALID`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One **block** = one core's `(image n, column slice xs, row range)` |
| Grid | The `device.compute_with_storage_grid_size()` cores, taken row-major and truncated to `P = min(grid, TUNING_OVERRIDES["max_cores"] or ∞)` |
| Per-core work | **`U < P`:** `cpi = min(floor(P/U), Ht)`. Core `c < U·cpi` gets unit `u = c // cpi` and slot `i = c % cpi`; root = slot 0. Rows: `q = Ht // cpi`, `rem = Ht % cpi`, `start = i·q + min(i, rem)`, `count = q + (i < rem)`. One block per core. **`U ≥ P`:** `cpi = 1`, and `ttnn.split_work_to_cores(grid, U, row_wise=True)` assigns `units_this_core` whole-image slices (`rows = Ht`); every core is its own root with `fan_in = 1`. |
| Remainder | Rows are tile counts per image, `Ht = ceil(HW/32)`, never `floor(N·HW/32)`. The unit row split hands out `Ht % cpi` extra rows. The last chunk of a block has `rows_valid = rows_core − (B−1)·h`, but pushes and pops the nominal `T`. |

**Choice of `gx`** (`_choose_gx`; `CHUNK_TILES_TARGET` and the L1 budget come from `l1_ledger.md`):

1. For every divisor `gx` of `S`, compute `Wt_core`, `G_core`, `U`, `cpi`, `h`, `T`, `R`.
2. Keep `gx` only if `G_core ≤ 32` and `F_fixed(Wt_core, T, R) ≤ L1_budget`.
3. Minimize `max_core_tiles`: that is `ceil(Ht/cpi)·Wt_core` when `U < P`, and `ceil(U/P)·Ht·Wt_core` when `U ≥ P`.
4. Ties: more cores used, then the larger `gx` for TILE and the smaller `gx` for RM.
5. If no `gx` survives, raise `NotImplementedError`.

**Invariant.** `cpi > 1` implies exactly one block per core, so a multi-round combine is never needed in Phase 0. With `cpi = 1`, consecutive blocks on a core reuse the gather semaphore sequentially; the root writer resets it after each wait.

**Mcast groups.** A unit's cores are contiguous in row-major order. The host builds one `ttnn.Mcast` channel over all used cores with `group_size = cpi` and sender index 0. Irregular groups wrap across grid rows; the helper falls back to multiple multicasts or a unicast chain. There is one semaphore for the gather, plus those that `Mcast.attach` adds.

**Regime-pinned tests (required).**

| Test | Pins |
|------|------|
| `max_resident_chunks = 0` | pure stream |
| `max_resident_chunks = 1` | partial residency |
| default | full residency |
| `max_cores = 2` with `U > 2` | multi-unit per core, `fan_in = 1` |
| `(1,1,8192,32)`, G = 1 | `cpi = P > 32`, so `R ≥ 2` on both WH and BH |

The RM variants of the residency pins are also required.

## Circular Buffers

`W = Wt_core` and `T = chunk_ht·W` (tiles); 2 KB = one bf16 tile page and 4 KB = one Float32 tile page. The full ledger, with axis accounting and sharing, is in `l1_ledger.md`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_rm` | 0 | 2 KB (tile-sized stick block) | `D_RM·T` (RM only) | Spans one chunk of `ht` rows × the `W` slice; streams over the chunks. | Float16_b | reader | compute | pass 1 + pass 2 (RM) |
| `cb_input_stream` | 1 | 2 KB | `D_IN·T` (TILE) / `T` (RM); allocated iff `B_max > K_alloc` | streams chunks | Float16_b | reader (TILE) / compute (RM tilize) | compute | pass 1 + pass 2 |
| `cb_input_resident` | 2 | 2 KB | `K_alloc·T` (alias group A) | Spans the resident tail; pass-1 view | Float16_b | reader (TILE) / compute (RM) | compute | pass 1 |
| `cb_input_resident_keep` | 3 | 2 KB | `K_alloc·T` (alias group A, same bytes) | pass-2 view of the same chunks | Float16_b | reader (TILE) / compute (RM) | compute | pass 1 push → pass 2 pop |
| `cb_x_squared` | 4 | 4 KB | `T` (alias group B: total `4 KB·T`) | Sequential helper intermediate (one full reduce block) | Float32 | compute | compute | pass 1 |
| `cb_output_tiles` | 5 | 2 KB | `D_OUT·T = 2T` (alias group B) | Double-buffered output chunk | Float16_b | compute | writer | pass 2 |
| `cb_reduce_scaler` | 6 | 2 KB | 1 | Constant | Float16_b | reader | compute | whole kernel |
| `cb_sum_acc` | 7 | 4 KB | `W` | Running Σx per channel; `UnpackToDestFp32` | Float32 | compute | compute | pass 1 (chunks 0..B−2) |
| `cb_sumsq_acc` | 8 | 4 KB | `W` | Running Σx²; `UnpackToDestFp32` | Float32 | compute | compute | pass 1 |
| `cb_chan_scale` | 9 | 4 KB | `W` | Σx per channel (last chunk) → **a = rstd·γ** per channel | Float32 | compute | compute | pass 1 end → collapse; expand → pass 2 end |
| `cb_chan_work` | 10 | 4 KB | `W` (alias group C) | Σx² → rstd_ch → mean_ch → **b'** (when β) | Float32 | compute | compute | pass 1 end → pass 2 end |
| `cb_chan_shift` | 11 | 4 KB | `W` (alias group C) | `m = mean·a`, then (no β) b' for pass 2 | Float32 | compute | compute | expand → (no β: pass 2 end) |
| `cb_group_map` | 12 | 2 KB | `W` | Constant map | Float16_b | reader | compute | whole kernel |
| `cb_gamma_rows` | 13 | 2 KB | `W` (iff γ) | γ row-0 tiles | Float16_b | reader | compute | load → expand |
| `cb_beta_rows` | 14 | 2 KB | `W` (iff β) | β row-0 tiles | Float16_b | reader | compute | load → expand |
| `cb_local_partials` | 15 | 4 KB | 2 (alias group D) | `[ΣxG, Σx²G]`, one block quantum | Float32 | compute | writer | collapse → gather |
| `cb_gathered_partials` | 16 | 4 KB | `2R` | `cpi` sender rows, one block quantum | Float32 | writer | compute | combine (root) |
| `cb_ones_tiles` | 17 | 2 KB | `R` | matmul in0 for the sender sum | Float16_b | reader | compute | combine (root) |
| `cb_group_sums` | 18 | 4 KB | 2 (alias group D) | `[S1, S2]` replicated | Float32 | compute | compute | combine (root) |
| `cb_stats_out` | 19 | 4 KB | 2 | `[rstd, mean]` replicated | Float32 | compute | writer | combine (root) |
| `cb_group_stats` | 20 | 4 KB | 2 | mcast landing zone, one block quantum | Float32 | writer | compute | combine → expand |
| `cb_affine_staging` | 21 | `align_up((HAS_GAMMA+HAS_BETA)·W·64, 64)` B | 1 (iff any affine) | RM stick slices of γ/β | raw bytes | reader | reader | load (affine) |

**Compute config:**

- `fp32_dest_acc_en = True`
- `math_fidelity = HiFi3`
- `math_approx_mode = False`
- `dst_full_sync_en = False`
- `unpack_to_dest_mode[cb_sum_acc] = unpack_to_dest_mode[cb_sumsq_acc] = UnpackToDestFp32`; all other CBs Default.

**Alias groups.** There are four, each a single `CBDescriptor` with several `CBFormatDescriptor`s:

- **A** = {2, 3}: lockstep (every chunk is pushed on both views; both views pop K·T per block).
- **B** = {4, 5}: disjoint lifetimes. `total_size = 4 KB·T`. Compute waits for the drain of `cb_output_tiles`, via `cb_reserve_back(cb_output_tiles, 2T)`, before the first x² push of block `b > 0`.
- **C** = {10, 11}: lockstep in-place hand-off. Both are at their base for every hand-off, because every quantum is W in capacity W.
- **D** = {15, 18}: disjoint lifetimes, ordered by local CB credit on the root. The writer pops `cb_local_partials` before it waits on the gather semaphore, the gathered push follows, and compute packs `cb_group_sums` only after waiting on `cb_gathered_partials`. Non-root cores never touch 18.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | boot (compute) | — | `compute_kernel_hw_startup<SrcOrder::Reverse>` (the matmul contract) | — | — | Every later helper call keeps its default INPUT_AND_OUTPUT reconfig. |
| 1 | `load_block` (reader) | chunks of T; `W` affine tiles | `prepare_reduce_scaler`, `zero_tile`, `fill_l1_range`; raw NoC reads | DRAM | `cb_reduce_scaler` (1), `cb_group_map` (W), `cb_ones_tiles` (R, root), `cb_input_*`, `cb_gamma_rows` / `cb_beta_rows` (W) | The map and scaler are never popped. |
| 2a | `accumulate_moments`: tilize (RM) | `h` blocks of `W` | `tilize<WT_CORE>` | `cb_input_rm` (T, pop) | `cb_input_stream` or `cb_input_resident` (T); compute also reserves/pushes T on `cb_input_resident_keep` | — |
| 2b | `accumulate_moments`: square | `rows_valid × W` | `eltwise_chain(grid(rows_valid, W), BinaryFpu<Mul, x Block (None/None), x Block (None/None)>, PackTile<cb_x_squared>)` | x chunk (T waited by the caller, not popped) | `cb_x_squared` (`rows_valid·W`, then the caller pushes the `T − rows_valid·W` pad) | — |
| 2c | `accumulate_moments`: Σx | `rows_valid × W` → `W` | `reduce<SUM, REDUCE_COL, NoWaitNoPop, ReduceAlgorithm::ReduceTile>` + `Accumulate(cb_sum_acc, j)` | x chunk (front); the caller pops the nominal T after 2d | j < B−1: `cb_sum_acc` (W); j = B−1: `cb_chan_scale` (W) | `cb_chan_scale` holds Σx |
| 2d | `accumulate_moments`: Σx² | same | same reduce, `Accumulate(cb_sumsq_acc, j)` | `cb_x_squared` (front, NoWaitNoPop; the caller pops T) | j < B−1: `cb_sumsq_acc`; j = B−1: `cb_chan_work` | `cb_chan_work` holds Σx² |
| 3 | `collapse_to_group_partials` | `[1×W]·[W×1]` per moment | `matmul_block<false>`, in1 policy NoWaitNoPop | `cb_chan_scale` (W, popped), then `cb_chan_work` (W, popped); `cb_group_map` (W, kept) | `cb_local_partials` (1 + 1) | Both chan CBs are empty and at base. |
| 4a | `combine_group_stats`: gather (writer) | 2 rows of 32 fp32 | raw NoC | `cb_local_partials` (2, popped) | root: `cb_gathered_partials` (2R) | — |
| 4b | `combine_group_stats`: sum senders (root compute) | `[1×R]·[R×2]` | `matmul_block<false>` | `cb_ones_tiles` (R), `cb_gathered_partials` (2R) | `cb_group_sums` (2) | Replicated rows |
| 4c | `combine_group_stats`: finalize (root compute) | 2 tiles | `eltwise_chain`: `CopyTile`, `MulUnary(inv_n)`, `MulBinary`, `SubBinary`, `Relu`, `AddUnary(eps)`, `Rsqrt`, `PackTile` ×2 | `cb_group_sums` (2, upfront, popped) | `cb_stats_out` (`[rstd, mean]`) | — |
| 4d | `combine_group_stats`: mcast (writer) | 8 KB | `Mcast` sender/receiver | `cb_stats_out` (2, root) | `cb_group_stats` (2, every unit core) | — |
| 5a | `expand_stats_to_channels`: rstd | `[1×1]·[1×W]ᵀ` | `matmul_block<true>` | `cb_group_stats` tile 0 (popped); map | γ: `cb_chan_work` (W); no γ: `cb_chan_scale` (W) | — |
| 5b | `expand`: a = rstd_ch·γ (iff γ) | W tiles | `eltwise_chain(tiles(W), BinaryFpu<Mul, cb_chan_work, cb_gamma_rows bcast Row>, PackTile<cb_chan_scale>)` | `cb_chan_work` (W, popped), `cb_gamma_rows` (W, popped) | `cb_chan_scale` (W) | **a resident through pass 2** |
| 5c | `expand`: mean | `[1×1]·[1×W]ᵀ` | `matmul_block<true>` | `cb_group_stats` tile 1 (popped); map | `cb_chan_work` (W) | — |
| 5d | `expand`: m = mean_ch·a | W tiles | `eltwise_chain(tiles(W), BinaryFpu<Mul, cb_chan_work, cb_chan_scale Block (Upfront, no pop)>, PackTile<cb_chan_shift>)` | `cb_chan_work` (W, popped), `cb_chan_scale` (kept) | `cb_chan_shift` (W): same bytes as `cb_chan_work`, in place, lockstep | No β: **b' = m in `cb_chan_shift`** |
| 5e | `expand`: b' = m − β (iff β) | W tiles | `eltwise_chain(tiles(W), BinaryFpu<Sub, cb_chan_shift, cb_beta_rows bcast Row>, PackTile<cb_chan_work>)` | `cb_chan_shift` (W, popped), `cb_beta_rows` (W, popped) | `cb_chan_work` (W): same bytes, in place, lockstep | **b' resident in `cb_chan_work`** |
| 6 | `normalize_block` | `rows_valid × W` per chunk | `eltwise_chain(grid(rows_valid, W), BinaryFpu<Mul, x Block (None/None), cb_chan_scale Row (Upfront, None)>, DestReuseBinary<Sub, b' Row (Upfront, None), DEST_TO_SRCA>, PackTile<cb_output_tiles>)` | x: `cb_input_resident_keep` (chunks B−K..B−1), then `cb_input_stream` (T each; the caller waits/pops T) | `cb_output_tiles` (`rows_valid·W` + T-pad push) | At block end, compute pops W from `cb_chan_scale` and from the b' CB. |
| 7 | `store_block` (writer) | `rows_valid × W` per chunk | raw NoC + TensorAccessor | `cb_output_tiles` (T, popped) | DRAM | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | helper (contract) | `compute_kernel_hw_startup<SrcOrder::Reverse>` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:100-106` | `(cb_chan_scale, cb_group_map, cb_local_partials)` | — | — | — |
| scaler | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_COL>(1.0f)` | `reduce_helpers_dataflow.hpp:121` (id-only); `:70` (DFB form) | scaler 1.0 | — | `cb_reduce_scaler` | — |
| map / affine / ones init | helper | `zero_tile(dfb)`; `fill_l1_range(addr, bytes, 0x3F803F80)` | `l1_helpers.hpp:53`; `l1_helpers.hpp:101` | — | — | `cb_group_map`, `cb_gamma_rows`, `cb_beta_rows`, `cb_ones_tiles` | — |
| map one-hot stores, affine row-0 copies | raw_api | halfword / word stores to L1 (`addr_to_l1_ptr`) | `l1_helpers.hpp:30` | 32 stores per map tile; 16 words per affine tile | `cb_affine_staging` | map / affine CBs | — |
| tilize (RM) | helper | `compute_kernel_lib::tilize<WT_CORE, cb_input_rm, out>` | `tilize_helpers.hpp:206` | `num_blocks = h` | `cb_input_rm` | `cb_input_stream` / `cb_input_resident` | `WT_CORE` (block width), `h` |
| square | helper | `eltwise_chain` + `BinaryFpu<BinaryFpuOp::Mul, …>` + `PackTile` | `eltwise/api/chain.hpp:620`; `:568`; `:588` | `IterationShape::grid(rows_valid, WT_CORE)`; input `WaitPolicy::None` / `PopPolicy::None`, `InputTileMapping::Block` (`:233`, `:257`) | x chunk | `cb_x_squared` | `rows_valid`, `WT_CORE` |
| Σx, Σx² | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, …, ReduceInputPolicy::NoWaitNoPop, …, ReduceAlgorithm::ReduceTile>` | `reduce_helpers_compute.hpp:673`; policy enum `:121`; algorithm `:165`; `ReduceInputBlockShape::of` `:300`; `Accumulate` `:427` | `ReduceInputBlockShape::of(rows_valid, WT_CORE)`, `Accumulate(acc_dfb, j)` | x / `cb_x_squared` | acc or chan CB | `rows_valid`, `WT_CORE` (= `chunk_ht`·`W` block) |
| collapse, expand, sum senders | helper | `compute_kernel_lib::matmul_block<transpose, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor, InitMode::Short, WaitAndPopPerKBlock, NoWaitNoPop (map) / WaitAndPopPerKBlock (gather)>` | `matmul_block_helpers.hpp:358`; `MatmulBlockShape::of` `:160` | collapse `of(1,1,1,1,W,1)`; expand `of(1, ·, 1, ·, 1, 1)` with block 1×W (subblocking left to the helper); sum senders `of(1,1,1,2,R,1)`; `interm = out` | see Realization | see Realization | Block = output tiles (`W` or 2) |
| a, m, b' chains | helper | `eltwise_chain` + `BinaryFpu<Mul / Sub, …, input(cb, BroadcastDim::Row)>` | `chain.hpp:568`, `input(cb, BroadcastDim, …)` overload `chain.hpp:410-418` | `IterationShape::tiles(W)` | chan / affine CBs | chan CBs | `W` |
| finalize | helper | `eltwise_chain(CopyTile, MulUnary, MulBinary, SubBinary, Relu, AddUnary, Rsqrt, PackTile)` | `chain.hpp:554`; `eltwise/unary/scalar.hpp:32,26`; `eltwise/binary/sfpu/basic.hpp:28,26`; `eltwise/unary/activations.hpp:22`; `eltwise/unary/math.hpp:38` | 2 tiles, DEST slots D0–D2 | `cb_group_sums` | `cb_stats_out` | — |
| normalize | helper | `eltwise_chain(BinaryFpu<Mul>, DestReuseBinary<Sub, …, DestReuseType::DEST_TO_SRCA>, PackTile)` | `chain.hpp:568`, `:580`, `:519-521` | `grid(rows_valid, WT_CORE)`; a and b' with `InputTileMapping::Row`, `WaitPolicy::Upfront`, `PopPolicy::None` | x chunk, `cb_chan_scale`, b' | `cb_output_tiles` | `rows_valid`, `WT_CORE` |
| stats mcast | helper | `dataflow_kernel_lib::McastArgs<…>()` `.sender(noc).send()` / `.receiver(noc).receive()`; host `ttnn.Mcast`, `McastConfig`, `McastFixedSenderConfig`, `.attach` | `mcast/kernel/mcast_args.hpp:13-48`; `mcast/host/mcast.hpp:96,108,131,149` | `group_size = cpi`, `sender_index = 0`, 8192 B | `cb_stats_out` | `cb_group_stats` | `cpi` |
| gather | raw_api | `noc_async_write`, `noc_async_writes_flushed`, `noc_semaphore_inc`, `noc_semaphore_wait_min`, `noc_semaphore_set` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:876, 1980, 2437, 2143, 2172` | 4 × 64 B per sender | `cb_local_partials` | `cb_gathered_partials` (root) | `cpi`, `R` |
| DRAM reads / writes | raw_api | `TensorAccessor` + `noc_async_read` / `noc_async_write`, barriers | `dataflow_api.h:547, 876, 1921, 1951`; `tech_reports/tensor_accessor/tensor_accessor.md` | per tile (TILE) / per stick segment (RM) | DRAM | input CBs / DRAM | `T`, `D_IN` |

**Helpers considered and rejected for the `raw_api` rows:**

| raw_api row | Candidate helper | File:line of the mismatch | Concrete reason |
|-------------|------------------|---------------------------|-----------------|
| map one-hot stores | `generate_index_tile` | `index_tile_dataflow.hpp:74` | It writes an index ramp (`store_index_tile`, `:26`), not a 0/1 one-hot with a channel→group mapping. |
| map one-hot stores | `fill_tile_column` | `tile_column_dataflow.hpp:70` | It sets **every** row of a column. The map needs a 1 only at rows `r` whose channel belongs to lane `l` (a partial column when `Cg < 32`, or across tiles when `Cg > 32`). |
| affine row-0 copies | `fill_tile_column` | `tile_column_dataflow.hpp:70` | It writes all 32 rows. Only row 0 is needed (`BroadcastDim::Row`), so it does 32× the stores. |
| gather | `McastArgs` sender/receiver | `mcast/kernel/mcast_args.hpp:13-27` | The helper is one-to-many: one sender feeds a receiver group. The gather is many-to-one into distinct slots of a single buffer. |
| DRAM reads / writes | `tilize_helpers_dataflow.hpp` | — | Compute tilizes; the reader moves raw sticks and tiles. There is no dataflow helper for interleaved column-slice reads. |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| 2b square | Mul | `x` chunk: All | `x` chunk: All | None |
| 5b a | Mul | `cb_chan_work` (rstd_ch): All (rows replicated) | `cb_gamma_rows`: Row0 | Row |
| 5d m | Mul | `cb_chan_work` (mean_ch): All | `cb_chan_scale` (a): All | None |
| 5e b' | Sub | `cb_chan_shift` (m): All | `cb_beta_rows`: Row0 | Row |
| 6 normalize Mul | Mul | `x`: All | `cb_chan_scale` (a): All (Row tile-mapping, one tile per column) | None |
| 6 normalize Sub | DestReuse Sub (DEST→SrcA) | DEST (x·a): All | b': All | None (DestReuse has no broadcast, which is why b' must be full-row-replicated) |
| 3 collapse (matmul) | in0 × in1 | Σ rows: Row0 meaningful (REDUCE_COL out; other rows only reach other output rows) | map: All | — |
| 4b sum senders (matmul) | ones × gathered | ones: All | gathered: rows `< cpi` (others zero-filled) | — (the output is replicated) |
| 5a/5c expand (matmul, transpose B) | stats × mapᵀ | stats: All (replicated) | map: All | — (the output is replicated) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Matmul boot mixed with reduce/eltwise/tilize helpers | `matmul_block` requires `compute_kernel_hw_startup<SrcOrder::Reverse>` once per kernel (matmul_block_helpers.hpp:100-106). Every other helper here assumes default source order. | Boot with Reverse exactly once. Leave every helper's reconfig at its default INPUT_AND_OUTPUT so each call re-sets the SrcA/SrcB formats. Verify early with the 1-tile acceptance case, then the partial-channel cases. |
| Accumulator precision across chunks | Reloading an fp32 accumulator through SrcA truncates it to TF32 on **every** chunk. The bias grows like `B·2⁻¹²` (mochi: B ≈ 130, which fails PERF_TOLERANCE). | `cb_sum_acc` / `cb_sumsq_acc` are `UnpackToDestFp32`, so the reload is an exact `copy_tile`. The **last** chunk's reduce writes to `cb_chan_scale` / `cb_chan_work` (Default unpack) so the matmul can read them through SrcB. Never point the matmul at an acc CB. |
| Ragged chunks vs ring wrap | The last chunk of a block has fewer valid rows. A ragged push would straddle the ring and corrupt the next block or the pass-2 re-reads. | Every producer pushes the nominal `T`: the reader without transferring the pad, compute with a pad `reserve`/`push` after the chain. Every consumer pops `T` and passes `rows_valid` to the helper shapes. |
| Alias-view discipline (groups A, C) | Two FIFOs over the same bytes have no mutual protection. | A: both views are reserved and pushed by the one producer for every resident chunk; pass 1 pops only `cb_input_resident`, pass 2 pops only `_keep`. C: each hand-off is a full W-tile cycle starting from base. 5d/5e write tile `i` over tile `i` inside one chain (per-tile unpack→math→pack). |
| Group B alias (x² vs output) across blocks | With `cpi = 1` and several blocks per core, block `b+1`'s x² could overwrite output tiles the writer has not drained yet. | Before the first x² push of block `b > 0`, compute calls `cb_reserve_back(cb_output_tiles, 2T)` as a drain barrier. |
| Gather / mcast remote addressing | Remote writers target `CB base + offset`, so the CB's write pointer must be at base. | The handoff CB capacities equal one block's quantum (`cb_gathered_partials` 2R, `cb_group_stats` 2). Only `cpi = 1` cores run several blocks, and those are sequential. |
| Gather rows not owned by a sender | Rows ≥ `cpi` in the last gather tile pair would be summed. | The root writer zero-fills those rows before its semaphore wait (no sender writes them). |
| `UnpackToDestFp32` misuse | A U2D CB read through SrcA/SrcB is invalid. | Only the reduce reload (`copy_tile`) touches `cb_sum_acc` / `cb_sumsq_acc`. |
| Map vs transposed map | The collapse uses M (channel rows × group lanes); the expand uses Mᵀ via `transpose=true`. Getting the orientation wrong gives silently permuted groups. | Pinned by the partial-channel acceptance cases (Cg = 10, 20, 24, 36, 56). With Cg = 32 the map is the identity and hides the bug. |
| `DestReuseType` direction | `DEST_TO_SRCA` gives `DEST − CB` (chain.hpp:519-521); swapping it gives `b' − x·a`. | The acceptance test with β ≠ 0 pins it. |
| Garbage rows in per-channel tiles | REDUCE_COL leaves rows 1..31 unspecified. | Matmul output row `r` depends only on input row `r`. Only row 0 of the collapse output is gathered, and the expand uses the replicated stats, not those rows. |
| Regimes reachable only on some grids | `cpi`, `R`, `K` and `U ≥ P` depend on P and the L1 budget. | `TUNING_OVERRIDES` (`max_cores`, `max_resident_chunks`, `chunk_tiles_target`) pin every regime in the acceptance test on any device. |
| L1 budget source | `ttnn.get_max_worker_l1_unreserved_size()` ignores L1 buffers the caller allocated. **Verifier:** it is also measured from the KERNEL_CONFIG base, so it overstates the CB region by the 70 688 B kernel-config ring (6 DRAM perf cases OOMed). | Use `l1_cb_budget(device)` in the program descriptor: the L1 memory view's per-bank size (measured from the allocator base), capped at the lowest live L1 buffer. That cap covers an R1 shard automatically. |
| Variance clamp | Rounding in `E[x²] − mean²` can go slightly negative when a group is constant. | `Relu` (max with 0) before `+eps`. |

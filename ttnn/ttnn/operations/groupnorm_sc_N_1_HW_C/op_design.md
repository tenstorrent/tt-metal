# Operation Design: groupnorm_sc_N_1_HW_C

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (two reductions + elementwise apply, one device program) |
| Goal | GroupNorm over an `(N, 1, HW, C)` channel-last tensor with per-group statistics over `HW × C/G`, centered two-pass variance, optional per-channel affine. Groups may straddle 32-channel tile boundaries (`(C/G) % 32 != 0`) from Phase 0. One native dispatch. |
| Math | `mean[n,g] = Σ_{h<HW, c∈g} x[n,0,h,c] / (HW·Cg)`; `var[n,g] = Σ_{h,c∈g} (x[n,0,h,c] − mean[n,g])² / (HW·Cg)`; `rstd = rsqrt(var + eps)`; `y[n,0,h,c] = (x[n,0,h,c] − mean[n,g(c)]) · rstd[n,g(c)] · gamma[c] + beta[c]`, with `Cg = C/G`, `g(c) = floor(c / Cg)` |
| Mode | Hybrid (helper-library block operations + two op-specific block operations: membership build, stats all-gather) |
| References | `.claude/skills/groupnorm-partial-channels/SKILL.md` (per-group iteration, two mask sites, L1 bound by G); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `matmul_block_helpers.hpp`, `eltwise/api/chain.hpp`, `eltwise/api/convenience.hpp`, `tilize_helpers.hpp`, `untilize_helpers.hpp`, `tilize_helpers_dataflow.hpp`, `reduce_helpers_dataflow.hpp`, `l1_helpers.hpp`; `tt_metal/hw/inc/api/dataflow/noc_semaphore.h`, `dataflow_api.h`; catalog `ttnn/ttnn/operations/examples/master.md` entries `reduce_block`, `row_reduce_accumulate`, `double_buffer`, `width_split`, `noc_placement`, `compute_block_size`, `compute_fusion`, `tensix_all_reduce`, `shared_input_reuse`; `eval/golden_tests/groupnorm_sc_N_1_HW_C/feature_spec.py` (TARGET / INPUTS / INVALID, authoritative) |

### Registry surface (consumes `feature_spec.py`)

| Item | Decision |
|------|----------|
| Signature | `groupnorm_sc_N_1_HW_C(input_tensor, num_groups, *, gamma=None, beta=None, eps=1e-5, in_place=False)` — exactly as required; no mask parameters |
| `INPUT_TAGGERS` | `alignment` (HW/C tile alignment, exactly the tagger given in the requirements), `affine`, `affine_dtype`, `affine_layout` (return `"none"` when the weight is absent), `groups_alignment` (`(C//G) % 32 == 0 → "group_aligned"` else `"group_straddling"`), `memory_layout` (from `input_tensor.memory_config().memory_layout`), `in_place` |
| Phase 0 `SUPPORTED` | `dtype [bf16]`, `layout [TILE, ROW_MAJOR]`, `memory_layout [INTERLEAVED]`, `in_place [False]`, `alignment ["tile_aligned"]`, `groups_alignment ["group_aligned", "group_straddling"]` (**both — partial channels are not gated**), `affine [gamma_beta, gamma_only, no_affine]`, `affine_dtype [bf16, "none"]`, `affine_layout [ROW_MAJOR, "none"]`. The implementer broadens to what the kernel actually passes (e.g. `float32`, `TILE` affine) |
| `EXCLUSIONS` | none at Phase 0. **No `(C/G) % 32` gate anywhere** |
| `INVALID` | not declared in the op file (lives in `feature_spec.py`) |
| Argument validation (`ValueError`) | rank ≠ 4 (message contains `4D`); `shape[1] ≠ 1` (message contains `dim[1]`); `C % num_groups ≠ 0` or `num_groups < 1` (message contains `num_groups`); gamma/beta shape ≠ `(1,1,1,C)` (message contains `gamma` / `beta`). Runs **after** the registry `validate()` per the template; the acceptance test matches these substrings via the repo's `expect_error` fixture |
| Tunable configuration | module `ttnn.operations.groupnorm_sc_N_1_HW_C.config` holds every host knob (table below). The program-descriptor builder reads them by attribute on the module object at call time (`config.NAME`, never `from config import NAME`) so tests and perf sweeps can monkeypatch them |
| Structural impossibilities noticed | none beyond `feature_spec.INVALID` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `num_groups` (G) | int | yes | `1 ≤ G ≤ C`, `C % G == 0`, `ceil(G/32) ≤ config.MAX_GROUP_TILES` (mechanism cap, see below) | — | RT (`G`, `Cg = C/G`, `num_group_tiles`) |
| `eps` | float | no | > 0 | 1e-5 | RT (fp32 bits) |
| `gamma`, `beta` | tensor / None | no | `(1,1,1,C)`, same dtype+layout as each other | None | CT `has_gamma`, `has_beta` + TensorAccessorArgs (placeholder when absent) |
| `in_place` | bool | no | Phase 0: `False` only | False | host |
| `inv_count` | float | derived | `1 / (HW·Cg)` | — | RT (fp32 bits) — the reduce_mean `n_reduced = HW·Cg` is passed as u32 |
| `config.L1_CB_BUDGET_BYTES` | int | knob | total per-core CB budget | 1_000_000 | host |
| `config.CHUNK_TILES_TARGET` | int | knob | ≥ 1 | 16 | host → CT `chunk_hw_tiles` |
| `config.MAX_CORE_C_TILES` | int | knob | ≥ 1 | 16 | host → split search |
| `config.MAX_GROUP_TILES` | int | knob | ≥ 1 | 4 (G ≤ 128) | host → cap |
| `config.STREAM_DEPTH` | int | knob | ≥ 2 | 2 | host → CB depths |
| `config.SPLIT_ORDER` | str | knob | `"hw_first"` / `"c_first"` | `"hw_first"` | host → split search tiebreak |
| `config.FORCE_STREAMING` | bool | test knob | — | False | host → regime selector override |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(N, 1, HW, C)`; any `N ≥ 1`, `HW ≥ 1`, `C ≥ 1` (Phase 0 SUPPORTED narrowed to `HW % 32 == 0 and C % 32 == 0`; Refinement 2 lifts both — `hw_non_aligned` / `c_non_aligned` are SUPPORTED) |
| Dtype | Phase 0: bf16; **Refinement 4: fp32, bf8b (TILE-only)** — `TB = ttnn.tile_size(dtype)`, intermediates stay fp32 |
| Layout | TILE or ROW_MAJOR |
| Memory | Phase 0: DRAM interleaved (TARGET adds L1 BLOCK_SHARDED) |
| gamma / beta | `(1,1,1,C)`, Phase 0 bf16 ROW_MAJOR; **Refinement 4: fp32 / bf8b and TILE layout** — the reader's `fill_affine_rows` lane-gathers channel `c` from byte `c·elem` of the stick (ROW_MAJOR) or from row 0 of tile page `c/32` (TILE; a run is split at the 16-lane face rows), decoding bf8b tiles lane by lane into a bf16 rows CB — one body for every channel geometry (identity, `c_valid`-clipped, periodic `c_period`); only row 0 of a rows tile is zeroed/valid |

### Output

| Property | Value |
|----------|-------|
| Shape | `(N, 1, HW, C)` (same as input) |
| Dtype | same as input |
| Layout | same as input |
| Memory | same memory config as input (interleaved → DRAM interleaved; sharded → same shard spec; `in_place=True` → the input buffer itself) |

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

### Derived geometry (single source of truth: host, computed once per call)

| Symbol | Definition |
|--------|------------|
| `Cg` | `C // G` channels per group |
| `HWt` | `ceil(HW / 32)` tile rows **per image** (never `floor(N·HW/32)`) |
| `Ct` | `ceil(C / 32)` channel tiles |
| `Ng` | `num_group_tiles = ceil(G / 32)` — group-slot tiles; group `g` lives at lane `g % 32` of slot tile `g // 32` |
| `hw_splits`, `c_splits` | the work-split (Work Distribution); `num_active = hw_splits · c_splits` cores |
| `core_hw_tiles`, `core_c_tiles` | this core's tile-row / tile-column extent (per image): balanced split with remainder, `core_c_tiles ≤ MAX_CORE_C_TILES` by construction of the split |
| `gather_tiles` | `ceil(num_active / 32)` — rows of the stats all-gather |
| `resident` | `core_hw_tiles · core_c_tiles · tile_bytes(dtype) ≤ L1_CB_BUDGET_BYTES − fixed_footprint` (fixed_footprint is the closed form in `l1_ledger.md`) and not `FORCE_STREAMING` |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `N` (image) | **independent** — statistics are per `(n, g)`, nothing crosses images | `batch_block = 1` image per round | 1 (images are sequential rounds; every core participates in every round) | host loop bound `N` → RT | not split across cores in Phase 0: every core owns its `(hw, c)` rectangle of **every** image; images are an outer loop. Reason: the `(HWt, Ct)` split already fills the grid for every INPUTS shape with `N > 1` except sub-µs tiny ones; a per-image sub-grid would need one mcast rectangle + semaphore pair per sub-grid | knob-turn (`batch_groups` sub-rectangles; regime row `batch_subgrid`) |
| `HW` (spatial, tile rows) | **dependent** — mean/var reduce over it | `block_hw_tiles` | resident regime: `core_hw_tiles` (one block = the whole assignment); streaming regime: `chunk_hw_tiles` | host → CT | split across `hw_splits` core rows; the cross-core combine is the stats all-gather (built) | the split is Phase 0; deeper `hw_splits` is a knob-turn |
| `C` tiles (channel tiles) | **independent for the per-channel column sums**; groups couple lanes *within and across* tiles, but that coupling is moved into the group-aggregation matmul (tiny, on rows), so the tile axis is assignable | `block_c_tiles` | `core_c_tiles` (whole per-core column range, capped at `MAX_CORE_C_TILES` by the split) | host → CT | split across `c_splits` core columns | knob-turn (`MAX_CORE_C_TILES`, `c_splits`) |
| `G` (groups) | **independent** — but not a *split* axis: groups do not align to tiles, so the op iterates channel tiles and reaches groups through the membership matrix (skill: "iterate by group, not by tile" is realized as "aggregate channel sums by group") | `num_group_tiles = Ng` (group-slot tiles) | `ceil(G/32)`, capped by `MAX_GROUP_TILES` | host → RT | every active core computes **all** `G` group stats redundantly (a tiny reduce over ≤ 5 gathered tiles); this is a deliberate compute-everywhere choice, not a communication role on one core | scheme-change for `G > 32·MAX_GROUP_TILES` (regime row `sparse_membership`) |
| `Cg` (channels within a group) | **dependent** — reduced into the group stat. Realized as `Σ_c M[c,g]·colsum[c]` (membership matmul), never as a per-tile REDUCE_SCALAR | none beyond `block_c_tiles` (the aggregation runs over the block's columns in one matmul) | — | — | inherits the `C` tiles split; the cross-core part is folded into the same all-gather as `HW` | — |
| stage: per-channel column sums (`colsum_rows`) | **independent along C tiles**, per core | `block_c_tiles` rows (row-0-valid tiles) | `core_c_tiles` | same knob | stays on the producing core; only `Ng` aggregated rows leave the core | — |
| stage: per-core partial group rows → gathered rows | **dependent across cores** (the sum over `hw_splits` rows and over straddled column ranges) | `gather_tiles × Ng` | `ceil(num_active/32) × Ng` | host → RT | all-gather to every active core (no root core, no idle waiting on a single reducer) | grid-sync perf lamp (tree) |
| stage: expanded per-channel rows (`mean_rows`, `scale_rows`, `shift_full`) | **independent along C tiles** | `block_c_tiles` | `core_c_tiles` | same knob | per core, for its columns only | — |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` (streaming regime) | `STREAM_DEPTH` | 2 blocks | reader/compute overlap over DRAM (catalog `double_buffer`: 4–8 tiles in flight, `2 × block`) |
| `cb_input_tiles` (resident regime) | — | 1 block = whole assignment | residency: input crosses DRAM once for all three passes |
| `cb_input_sticks` (RM) | `STREAM_DEPTH` | 2 tile-rows | reader/tilize overlap |
| `cb_output_tiles` | `STREAM_DEPTH` | 2 chunks | compute/writer overlap on the output DRAM write |
| `cb_output_sticks` (RM) | `STREAM_DEPTH` | 2 tile-rows | untilize/writer overlap |
| `cb_gather` | fixed 2 rounds | 2 × `Ng·gather_tiles` | the two combine rounds per image land in disjoint halves — this is what makes the all-gather ack-free (see Dataflow Strategy) |
| every other CB | 1 (single block / single constant) | — | sequential compute-thread phases cannot pipeline against themselves |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Membership matmul `in1` is dense `block_c_tiles × Ng` fp32 tiles | `Ng ≤ MAX_GROUP_TILES` (G ≤ 128 at Phase 0) | host raises `NotImplementedError` (scalar refinement gate; not a registry axis) | membership CB grows as `core_c_tiles·Ng·4 KB` and starves the input budget |
| Per-core stat set scales with `core_c_tiles` | `core_c_tiles ≤ MAX_CORE_C_TILES` and `fixed_footprint ≤ L1_CB_BUDGET_BYTES` | split search raises `c_splits` until both hold (`c_splits ≥ ceil(Ct / max_core_c_tiles_effective)`) | L1 OOM at wide C (C = 4096 → 128 tiles) |
| Gather row index is a tile row | `num_active ≤ 32 · gather_tiles`, `gather_tiles = ceil(num_active/32)` | derived, never fixed | a core's partial row lands outside the gather buffer |
| Monotone counter semaphore | `N · num_active < 2³²` | trivially satisfied | wrap → deadlock |
| `reduce_mean` count `n_reduced = HW·Cg` | `< 2³²`, and `inv_count` exact enough in fp32 | trivially satisfied for INPUTS | — |
| Partial-row lanes | group `g` ↔ lane `g % 32` of slot `g // 32` (host, kernel and membership build all use this one rule) | — | wrong lanes summed |
| Chain `block_size` vs DEST (fp32 DEST, half-sync → `DEST_AUTO_LIMIT = 4`, `dest_helpers.hpp:22-26`) | chain runtime-clamps; matmul subblocks are `1×1` | helper-owned | — |
| `matmul_block` precision rule (`matmul_block_helpers.hpp:253-258`) | both matmul inputs **fp32** (`colsum_rows`, `membership`) so `HiFi4 + fp32_dest_acc_en` is the documented-correct combination; never bf16 inputs under HiFi4 | design decision (membership is fp32) | silent K-accumulator corruption on WH B0 (#38306) |
| DRAM/L1 alignment of RM stick slices | column-chunk byte offsets are multiples of 64 B (bf16) / 128 B (fp32); lengths may be unaligned | by construction (`c0 = 32·tile`) | NoC alignment fault |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `resident_2d` | **built** | `memory_layout == INTERLEAVED` and `resident` (whole per-core assignment fits: `core_hw_tiles·core_c_tiles·tile_bytes ≤ budget − fixed_footprint`) | `block_hw_tiles = core_hw_tiles`, `block_c_tiles = core_c_tiles`, one block per image held across all three passes | **minimum at the DRAM boundary**: input crosses DRAM once, output once, gamma/beta slices once per core. Cross-core: per image 2 rounds × (`Ng` × two 64 B multicasts + `num_active` semaphore increments) per core — structurally required by the `HW`/`Cg` split | one block = one reader fill, one `reduce` init, one membership matmul and one expansion pass per image; the three passes re-read L1, not DRAM |
| `streaming_2d` | **built** (fallback) | `INTERLEAVED` and not `resident` | `block_hw_tiles = chunk_hw_tiles`, `block_c_tiles = core_c_tiles`, `num_blocks_this_core = ceil(core_hw_tiles / chunk_hw_tiles)` per pass | input crosses DRAM **2×** with `two_pass` (**Refinement 5**, `config.STREAMING_TWO_PASS`, compiled iff `!hw_mask`): pass A reads each chunk once and takes BOTH statistics from the resident chunk — S = Σx and U = Σ(x − s)² against a per-channel shift s = the column means of the chunk-0 tile-row 0 (the one matmul body × a 1/32 row tile), S/U alternating on the one accumulate-reduce in a 2K `cb_colsum_rows` ring — and after round 0 one two-DEST-slot chain forms Σ(x − m)² = U + 2(s − m)(S − (n/2)(m + s)) per channel (every square is of a centered value; never E[x²] − mean²), then pass B applies; **3×** (one per pass) on the `hw_mask` path. Measured 4V → 3V = 1.28–1.33× on the VAE cells, both at ~380 GB/s (DRAM-bound) | fewer `Accumulate` reloads and CB handshakes per pass; larger DRAM transactions in flight |
| `block_sharded_resident` | **built (Refinement 1)** — a **placement** of the resident regime: TILE shards back `cb_input_tiles` / `cb_output_tiles` directly (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy; `in_place=True` = a second buffer index on the input shard's region — pass 3 reads each input tile for the last time in the block the pack lands in); `(hw_splits, c_splits, K, Hs)` are read off the shard spec (no split search). RM shards are stick pages whose stick (`shard_w·elem` B, e.g. 80 B for the model's 40-channel shard) is not the tilize's `K·64 B` stick, so the reader stages each tile-row's valid sticks L1 → L1 into `cb_input_sticks` (pad lanes / pad sticks zeroed) and compute tilizes image n's rows once into the resident tiled CB; the writer copies the valid bytes of each untilized stick into the output shard. **Refinement 3 (`rm_direct`, `config.RM_SHARD_DIRECT_VIEW`)**: where the view is exact, the RM shard is instead consumed IN PLACE as the row-major block of width `lcm(shard_w, 32) = 32·K'` (`m = lcm/shard_w` sticks per view row — a `[2048, 40]` bf16 shard is byte-identical to a `[512, 160]` block, `K' = 5`): the shard buffer itself backs `cb_input_sticks` re-paged as tile pages (the tilize reads it zero-copy) and the output shard backs `cb_output_sticks` (the untilize packs straight into it); the reader pushes credits only and the writer stores nothing. View lane `j` is channel `c0 + j % shard_w` (`c_period`), which only the membership build and the affine-row fill see. Exactness gate (`_rm_direct_view`): stick page = `shard_w·elem` B, `K' ≤ MAX_CORE_C_TILES`, `shard_rows` and `HW` multiples of `32·m` (no view row partly outside its image); plus the K'-proportional CBs must fit below the shard (`min(L1_CB_BUDGET_BYTES, lowest shard address − L1 allocator base − margin)`), else the staged path above is the fallback. Chain block size is `b = min(K, DEST_AUTO_LIMIT)` (blocks never cross a K-row). Measured (8×8 model grid, in_place, µs): RM 16384×320 181.5 → 70.5, 4096×640 68.2 → 47.4, 16384×640 214.6 → 111.4, 1024×1280 38.7 → 33.8, 4096×1920 123.6 → 121.6; 16384×960 `[2048,120]` FAIL (L1 clash) → 245 (staged, `Q = 4`). **No `HW % shard_height` predicate was needed**: each kernel intersects the core's stick range with every image (`kernels/groupnorm_sc_N_1_HW_C_geometry.hpp`), so N > 1 shards — even ones straddling two images mid tile-row — work; a core with no sticks in image n unicasts a zero row to the root (uniform row count) and drains the broadcast stats. Partial tile-rows (RM shard heights not multiples of 32) are masked in pass 2 through the **expansion matmul**: the writer copies the landed group-mean tiles with the rows outside the image zeroed (`cb_masked_mean`), and `(32×Ng) @ (Ng×1)` yields `mean_full[r][c] = mask[r]·mean_c` — the pass-2 chain stays one instantiation (the design's `cb_hw_mask` chain element cost +1.1–10 KB of --dev code and overran the kernel-config ring) | `memory_layout == BLOCK_SHARDED` | one block = the shard (image n's sub-block at `row_off·K`) | input crosses DRAM **0×**, output 0× (L1 → L1); same cross-core stats traffic (+ one 128 B zero row per core per missed image) | measured (8×8 model grid, in_place): RM `(1,1,16384,320)` `[2048,40]` 181 µs vs 337 µs reference; RM `(1,1,4096,640)` `[512,80]` 66 vs 105; TILE `(1,1,1024,1280)` `[128,160]` 31.5 vs 42.5; RM `(1,1,1024,1280)` 39 vs 44.5 |
| `batch_subgrid` | **deferred** — spreading `N` across per-image sub-rectangles only helps shapes with `HWt·Ct < num_cores` (e.g. `(8,1,64,64)`), all of them µs-scale where dispatch overhead dominates; reachable because the mcast rectangle, semaphore ids and row ranges are per-core RT args (kernel unchanged, host emits one rectangle + semaphore pair per sub-grid) | `N > 1 and HWt·Ct·k ≤ num_cores` | as `resident_2d` per sub-grid | same as `resident_2d` | — |
| `sparse_membership` | **deferred** — `G > 32·MAX_GROUP_TILES` is outside INPUTS/TARGET shapes (max G = 32) and outside SD/SDXL; the dense `block_c_tiles × Ng` membership would grow with `G`. Reachable: replace the dense in1 with ≤ 2 slot tiles per channel tile and a per-tile K=1 matmul accumulating into slot rows | `Ng > MAX_GROUP_TILES` | as `resident_2d` | same | — |
| `exact_sfpu_stats` | **deferred** — precision refinement for fp32 inputs at extreme `|mean|/σ` (Key Risks: every FPU operand is tf32-rounded → mean error ≈ `2·2⁻¹¹·|mean|`). Reachable: stat CBs configured `UnpackToDestFp32`, gather reduce with `ReduceFp32Mode::Accurate`, group aggregation via SFPU masked sums; hot loops unchanged. Not built because bf16 inputs (Phase 0) quantize the data itself far coarser (`2⁻⁸·|mean|`) than this error | `dtype == float32 and` a large-mean heuristic | same blocks | same | — |
| `group_split` (work unit = `(n, g)`, no combine) | **rejected** — superseded by `resident_2d`/`streaming_2d`: with straddling groups a boundary tile column is owned by two cores → duplicate DRAM reads (~1.3× at `Cg = 10`) **and conflicting writes of the same output tile page** (lane-partial DRAM writes are unaligned 24 B fragments) | — | — | above minimum on reads, incorrect on writes | — |
| `cluster_split` (work unit = lcm-aligned group cluster, no combine) | **rejected** — parallelism collapses to `C / lcm(Cg, 32)` clusters (2 clusters for `C = 320, Cg = 10`; 1 for `C = 200, Cg = 25`); superseded by the 2-D split + all-gather | — | — | minimum reads, but ≤ 2 cores busy | — |
| `one_pass_moments` (`E[x²] − mean²`) | **rejected** — forbidden by the correctness contract (cancellation at large `|mean|`); superseded by the centered second pass | — | — | would save one input pass in `streaming_2d` | — |
| `two_pass_shifted` (Refinement 5) | **built** — the streaming saving `one_pass_moments` promised, without its cancellation: the squares are taken against a per-channel shift within O(σ/√32) of the data (the first tile-row's column means) while the chunk is in L1, and the exact algebraic combine with the group mean adds only products of O(σ) mean differences; the `mean = 10σ` cells measure identical to the fully centered pass 2 | `streaming_2d and not hw_mask` | as `streaming_2d` | input `2·V` | — |

**Regime-selection function (host, exact):**
```
if memory_layout == BLOCK_SHARDED:  regime = "block_sharded_resident"   # Refinement 1: K, Hs, grid from the shard spec; Q shrinks only if fixed (+ RM tiled copy) misses the budget
                                                                       # hw_mask = some tile-row partly outside its image (RM stick heights, N>1 straddles, HW%32 on TILE shards) -> masked-mean expansion compiled in
hw_mask = HW % 32 != 0                                                 # interleaved (Refinement 2): the core owning the image's last tile-row gets row_hi = HW - 32*(HWt-1)
fixed = fixed_footprint(core_c_tiles, Ng, gather_tiles, chunk_hw_tiles, layout, has_gamma, has_beta)   # l1_ledger.md closed form
resident = (not config.FORCE_STREAMING) and core_hw_tiles*core_c_tiles*tile_bytes <= config.L1_CB_BUDGET_BYTES - fixed
regime = "resident_2d" if resident else "streaming_2d"
two_pass = config.STREAMING_TWO_PASS and not resident and not hw_mask   # Refinement 5: pass A (both statistics from one read) + pass B (apply)
```
The two built regimes share every kernel; they differ in `input_resident` (CT flag: pass-1 input policy, pass-2/3 chain wait/pop policies, reader fill count) and in `cb_input_tiles` capacity. **Regime-pinned tests are required**: the acceptance test forces `streaming_2d` through `config.FORCE_STREAMING`.

### Traffic ranking

Candidates, ranked by bytes crossing each tier (structural; no time estimates). Input volume `V = N·HW·C·2 B`.

| Split | DRAM crossings | Cross-core | Grid fill | Verdict |
|-------|----------------|------------|-----------|---------|
| 2-D tile split `(hw_splits × c_splits)` + stats all-gather, **resident** | input `1·V`, output `1·V`, gamma/beta `hw_splits·2C·2 B` (negligible) | per image: `num_active × Ng × 128 B` multicast payload + `num_active²` semaphore increments (64 cores: 8 KB payload, 4096 atomics) | `min(HWt·Ct, num_cores)` | **chosen** |
| same, streaming | input `2·V` (`two_pass`, Refinement 5) / `3·V` (`hw_mask` three-pass) | same | same | built as fallback only when residency fails |
| hw-only split (`c_splits = 1`) / c-only split | same as the 2-D split (each is a corner of it) | same | worse when one axis is short (`(1,1,32,4096)`: `HWt = 1`) | subsumed |
| `(n, g)` group split, no combine | input ≈ `1.3·V` per pass (boundary tiles read twice), output conflicting | none | `N·G` | rejected (row above) |
| lcm cluster split, no combine | `1·V` | none | ≤ 2 cores | rejected |

The dependent-axis split (`HW`, plus the group straddle across `C` tiles) is taken **because** it is also the residency mechanism: cutting `HW` across cores is what makes the per-core slice fit L1 (`(1,1,16384,320)` on 64 cores → 80 tiles = 160 KB per core), turning three DRAM reads into one. The combine it needs is ~8 KB of payload per image against ~10 MB of input.

**Operand-reuse check** (every operand vs the chosen split): `x` varies along both split axes → no re-read. `gamma`/`beta` do **not** vary along `HW` → re-read by `hw_splits` cores (`≤ 64 × C × 2 B` — e.g. 41 KB for C = 320 vs 10 MB input); a multicast is not warranted (catalog `shared_input_reuse` wins when the shared operand is a large fraction of traffic). `membership` is built locally from RT args, not read.

**Stall-shadow check**: the only stage that waits on a peer is the combine (writer waits for `num_active` increments). Nothing compute-side is independent of the group mean at that point (pass 2 needs it), and the reader in the resident regime has already finished; in the streaming regime the reader **prefetches the first pass-2 block during the combine** (the reader does not depend on the stats) — the input CB depth of 2 blocks is what allows it. Idle otherwise, stated.

### Block schedule

Logical schedule per core (reader, compute, writer realize their parts asynchronously). `image` is the outer round; `pass` loops are the three algorithmic passes; `block` is the L1 unit (resident: 1 block per image; streaming: `num_blocks_this_core` per pass).

```cpp
build_membership_block();                        // writer, once per program: Ng×block_c_tiles fp32 0/1 tiles
zero_unused_gather_rows();                       // writer, once: rows ≥ num_active of both gather halves
for (uint32_t image = 0; image < N; ++image) {
    // ---- pass 1: per-channel column sums → group mean ----
    for (block_idx …) { load_block(pass=1, block_idx); colsum_block(block_idx); }     // reduce<SUM,REDUCE_COL,AccumulateViaAdd> (Accumulate across blocks when >1)
    group_partial_rows_block(/*from*/ cb_colsum_rows);                               // matmul_block: (1×block_c_tiles) @ (block_c_tiles×Ng)
    allgather_round_block(round=0);                                                    // writer: mcast 2×64 B per slot tile, count semaphore; compute: reduce_mean over gather → cb_group_mean
    // ---- pass 2: centered squares → group variance → rstd ----
    expand_rows_block(cb_group_mean → cb_mean_rows);                                   // per T: matmul_block<transpose>: (1×Ng) @ (Ng×1)
    for (block_idx …) { load_block(pass=2, block_idx);                               // streaming only; resident re-reads L1
        for (chunk …) { centered_sq_chunk(chunk); colsum_accumulate_chunk(chunk); } } // chain Sub(bcast Row)→Square→pack fp32; reduce Accumulate::at/at_last
    group_partial_rows_block(cb_colsum_rows);
    allgather_round_block(round=1);                                                    // → cb_group_rstd holds var
    finalize_rstd_block();                                                             // in-place chain: +eps, rsqrt on Ng tiles
    // ---- pass 3: apply ----
    expand_rows_block(cb_group_rstd → cb_scale_rows); scale_rows_block();              // scale = rstd_row ⊙ gamma_row (in place; identity when no gamma)
    expand_rows_block(cb_group_mean → cb_mean_rows); shift_full_block();               // shift_full[T] = bcast_rows(beta_row − mean_row ⊙ scale_row); full 32-row fp32 tile per T
    for (block_idx …) { load_block(pass=3, block_idx);
        for (chunk …) { apply_chunk(chunk); [untilize_chunk(chunk);] store_chunk(chunk); } }
    release_block();                                                                   // resident: pop the whole input block; pop group stat CBs
}
```

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------------|------------------------|--------------------|-----------------------------------|
| `build_membership_block` | `block_c_tiles × Ng` fp32 tiles | whole program | once per program (writer, before any input is needed by compute) |
| `load_block(pass, block_idx)` | `block_hw_tiles × block_c_tiles` tiles (TILE: pages; RM: `32·block_hw_tiles` stick slices of `block_c_tiles·64 B`, then `tilize`) | resident: the block for all 3 passes; streaming: `STREAM_DEPTH` blocks in flight | resident: once per image; streaming: once per pass per block |
| `colsum_block` | `block_hw_tiles × block_c_tiles` → `block_c_tiles` row-0 tiles | input block | one `reduce` call per block (one init); `Accumulate` reload only when `num_blocks_this_core > 1` |
| `group_partial_rows_block` | `(1 × block_c_tiles) @ (block_c_tiles × Ng)` | membership | one `matmul_block` call per pass per image |
| `allgather_round_block` | `Ng` row-0 tiles out, `gather_tiles × Ng` tiles in | gather half `round` | 2 rounds per image; one `reduce_mean` call per round; one semaphore wait per round |
| `finalize_rstd_block` | `Ng` tiles in place | — | once per image |
| `expand_rows_block` | `block_c_tiles` calls of `(1×Ng)@(Ng×1)` | group stat rows, membership | 3 per image (mean for pass 2, rstd and mean for pass 3) |
| `scale_rows_block`, `shift_full_block` | `block_c_tiles` row tiles (shift also expanded to full tiles) | — | once per image |
| `centered_sq_chunk` | `chunk_hw_tiles × block_c_tiles` (`IterationShape::grid`) | `cb_mean_rows` (B operand, `InputTileMapping::Row`) | one chain call per chunk; init once per chunk (chain-owned) |
| `colsum_accumulate_chunk` | same chunk → `block_c_tiles` rows | `cb_colsum_accum` | one `reduce` call per chunk with `Accumulate::at/at_last` |
| `apply_chunk` | `chunk_hw_tiles × block_c_tiles` | `cb_scale_rows`, `cb_shift_full` | one fused chain call per chunk |
| `untilize_chunk` (RM only) | `block_c_tiles` wide × `chunk_hw_tiles` tile-rows | — | one `untilize` call per chunk |
| `store_chunk` | chunk tiles / sticks | — | writer, per chunk; barriers batched per chunk |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap (resident pass 1)** | `colsum_block` is a single `reduce` with `WaitUpfrontNoPop`: compute waits for the whole assignment before adding, so the DRAM fill and pass-1 adds do not overlap (pass-1 compute is ~1 add per tile, small) | chunked pass 1 with cumulative `cb_wait_front` and a caller-managed accumulate (needs `Accumulate` without pops — today asserted `BulkWaitBulkPop` only), or the streaming regime's block loop with `STREAM_DEPTH` ≥ 3 |
| **Grid synchronization (combine)** | flat all-gather: `num_active²` semaphore atomics per round (4096 at 64 cores); catalog `tensix_all_reduce` measured tree/leader reductions 1.45–1.6× under grid-filling contention | row-leader tree: each core row gathers to a leader, leaders gather to a root, root multicasts `Ng` tiles (`mcast_pipe`, one sender per round) |
| **Split order** (`SPLIT_ORDER = hw_first`) | maximizing `hw_splits` first minimizes per-core column count (stat set, per-column fixed costs) and keeps RM stick slices wide, but leaves each core few tile rows per column (`(1,1,16384,320)`: 8 rows × 10 cols) | `c_first` / balanced `(hw_splits, c_splits)` (catalog `width_split`: column-range splits win for short tensors) |
| **Chunk size** (`CHUNK_TILES_TARGET = 16`) | 64 KB fp32 scratch per chunk; catalog `compute_block_size` shows diminishing returns past 4–8 tile-rows and `double_buffer` shows 4–8 transactions in flight suffice | 8 and 32 |
| **Fused apply chain** (`Sub`/`Mul(bcast)` + `DestReuseBinary` on full tiles) | catalog `compute_fusion`: DEST-reuse into an FPU consumer measured 0.82× vs an L1 round trip; here it saves one scratch pass and an extra `shift` expansion | two bcast chains through an fp32 scratch CB (`x·scale_row → scratch; scratch + shift_row → out`) |
| **`MAX_CORE_C_TILES = 16`** | bounds the per-core stat set (~16 KB per column); a smaller cap raises `c_splits` and shrinks the fixed footprint, freeing input budget for residency | 8 |
| **Membership build on the writer** | ~`block_c_tiles·Ng` tile zero-fills + `32·block_c_tiles` word stores at program start; overlaps the reader's pass-1 fill on the resident path | hoist to a one-time host-side constant tensor is **not** allowed (no external mask tensor); alternative is the reader building it between input blocks |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input → `cb_input_tiles` (TILE) | bf16 tiles | reader `TensorAccessor` page reads, `block_hw_tiles × block_c_tiles` tile pages per block (row-major within the block), NoC0 | resident: reads the whole assignment once per image; streaming: per pass per block |
| Input → `cb_input_sticks` → `cb_input_tiles` (RM) | bf16 sticks → tiles | reader `dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks>(accessor, 32, block_c_tiles·64 B, start_stick, c0·2 B)` per tile-row (`tilize_helpers_dataflow.hpp:87-93`); compute `tilize<block_c_tiles, cb_input_sticks, cb_input_tiles>(block_hw_tiles)` (`tilize_helpers.hpp:187-197`) | producer of `cb_input_tiles` is **compute** in the RM leg |
| gamma/beta → `cb_gamma_rows`, `cb_beta_rows` | affine dtype, row-0-valid tiles | reader: zero the tile (`l1_helpers.hpp:50-54` `zero_tile`), copy `32·elem` bytes of the weight slice into face-0 row 0 and face-1 row 0 (TILE-layout weights: read tile `T` directly) | once per program (held, never popped until program end) |
| membership → `cb_membership` | fp32 0/1 tiles, `T`-major, `Ng` per `T` | **writer**, once: for each `T`, for each slot `k`: zero tile (`fill_l1_range`, `l1_helpers.hpp:89`), then for each lane `c` with channel `32·(t0+T)+c < C`: `g = channel / Cg`; if `g // 32 == k` store `1.0f` at `[row c][col g % 32]` (face-major address). Runs a running group counter instead of a division per element | `M_T[c][g'] = 1 ⇔ channel 32·T+c ∈ group 32k+g'`. Padded channels (`≥ C`) get all-zero rows — that is the trailing-C mask for free |
| `cb_colsum_rows` → `cb_partial_rows` | fp32 row-0 tiles | compute `matmul_block` (1×K)@(K×Ng) | the reduce-side "mask" site: one matmul replaces `Σ_g mask_g ⊙ ·` |
| `cb_partial_rows` → peers' `cb_gather` | 2 × 64 B per slot tile | **writer**: for slot `k`: `noc_async_write_multicast_loopback_src(src = partial_tile_k + face·1024, dst = gather_half_base + k·gather_tiles·4096 + tile(r)·4096 + ((r/16)·2 + face)·1024 + (r%16)·64, 64 B, rect)` for `face ∈ {0,1}`, `r = core_linear_idx`; `noc_async_write_barrier()`; then `Semaphore<>::up(noc, x_j, y_j, 1)` for every active core `j` (`noc_semaphore.h:137`) on `sem_round[round]` | `dataflow_api.h:1700`. Landing address is identical on all cores because every core allocates the CBs identically (catalog `tensix_all_reduce` :713-720). Receivers wait `sem_round[round].wait_min((image+1)·num_active)` (`noc_semaphore.h:264`) — monotone, never reset, safe with concurrent senders |
| `cb_gather` → compute | fp32 tiles, row `r` = core `r`'s partial | writer `cb_reserve_back(cb_gather, Ng·gather_tiles)` → wait count → `cb_push_back(cb_gather, Ng·gather_tiles)`; compute `reduce_mean<REDUCE_COL,…,AccumulateViaAdd>(of(gather_tiles, Ng), HW·Cg)` → `cb_group_mean` / `cb_group_rstd` | **Ack-free reuse argument**: the two rounds of an image land in the two halves. A core sends round-0 rows of image `i+1` only after receiving *all* round-1 rows of image `i`, which every core sends only after consuming its round-0 gather of image `i`; symmetrically for round 1. Hence no sender can overwrite a half before every receiver has popped it. Rows `≥ num_active` are zeroed once by the local writer (disjoint from any remote write) |
| Expanded rows | fp32 row-0 tiles; `cb_shift_full` full 32-row fp32 tiles via `unary_bcast<BroadcastDim::Row>` | compute | per `T` |
| Pass-2 / pass-3 chunks | fp32 scratch (`cb_fp32_scratch`), output dtype in `cb_output_tiles` | compute chains | chunk = `chunk_hw_tiles × block_c_tiles` |
| `cb_output_tiles` → DRAM (TILE) | output dtype tiles | writer `TensorAccessor` page writes, NoC1 | double-buffered |
| `cb_output_tiles` → `cb_output_sticks` → DRAM (RM) | tiles → sticks | compute `untilize<block_c_tiles, cb_output_tiles, cb_output_sticks>(chunk_hw_tiles)` (`untilize_helpers.hpp:145-154`); writer `write_sticks_after_untilize<cb_output_sticks>(accessor, 32, block_c_tiles·64 B, start_stick, c0·2 B)` (`tilize_helpers_dataflow.hpp:129-135`) | — |
| Deferred: `block_sharded_resident` | shard in L1 | `cb_input_tiles = ttnn.cb_descriptor_from_sharded_tensor(input)`; output CB aliases the output (or input, `in_place`) shard; RM shard → one resident `tilize` into a tiled scratch of shard size | placement of the built resident regime — a knob-turn for `in_place=False`; `in_place=True` is also a knob-turn (pass 3 consumes the input block last, so packing the result over it is legal) |

**Placement axis classification (TARGET `memory_layout`)**: `INTERLEAVED` → built; `BLOCK_SHARDED` → the same 2-D logical shard the work-split already defines, cost = a CB/placement change (knob-turn) because the cross-core combine is already built; `in_place` → knob-turn on top of it.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block `block_hw_tiles × block_c_tiles` of one image; a core owns `core_hw_tiles × core_c_tiles` tiles of **every** image |
| Grid | physical rectangle `rect_x × rect_y` of the compute grid with `rect_x·rect_y ≥ num_active` (`rect_x = min(num_active, grid_x)`, `rect_y = ceil(num_active / rect_x)`); active core `k < num_active` sits at `(k % rect_x, k / rect_x)` with `hw_idx = k / c_splits`, `c_idx = k % c_splits`; cores `k ≥ num_active` receive `is_active = 0` and exit (their CBs exist so multicast landings are harmless) |
| Split search (host) | `max_core_c = min(MAX_CORE_C_TILES, largest c with fixed_footprint(c) ≤ budget)`; `c_min = ceil(Ct / max_core_c)`; over `c_splits ∈ [c_min, min(Ct, num_cores)]`: `hw_splits = min(HWt, num_cores // c_splits)`; objective: minimize `ceil(HWt/hw_splits)·ceil(Ct/c_splits)` (max per-core tiles); tiebreak `SPLIT_ORDER = hw_first` → larger `hw_splits`, then fewer cores |
| Per-core work | tile rows `[r0, r1)` of each image with `r0 = hw_idx·(HWt // hw_splits) + min(hw_idx, HWt % hw_splits)`, `core_hw_tiles = HWt // hw_splits + (hw_idx < HWt % hw_splits)`; tile columns likewise from `Ct`, `c_splits`, `c_idx`. RT args per core: `r0, core_hw_tiles, t0, core_c_tiles, core_linear_idx, is_active` + the shared geometry |
| Remainder | alignment-aware: `HWt = ceil(HW/32)` per image, `Ct = ceil(C/32)`; the balanced split gives the first `HWt % hw_splits` rows one extra tile; the last tile of a ragged axis keeps nominal CB push/wait counts (only the valid rows/lanes are transferred/computed) |
| Regime selection | see Blocking Model; regime-pinned test via `config.FORCE_STREAMING` |
| Examples (64-core grid) | `(1,1,16384,320)` → `Ct=10`, `HWt=512`: `c_splits=1`, `hw_splits=64` → 8×10 tiles/core = 160 KB → `resident_2d`. `(1,1,64,4096)` → `Ct=128`, `HWt=2`: `c_min=8`, `c_splits=32`, `hw_splits=2` → 1×4 tiles/core. `(1,1,32,32)` → 1 core. `(2,1,1024,640)` → `Ct=20`, `HWt=32`: `c_splits=2`, `hw_splits=32` → 1×10 tiles/core per image, 2 image rounds |
| NoC placement | reader on NoC0 (`ReaderConfigDescriptor`), writer + combine on NoC1 (`WriterConfigDescriptor`); active cores enumerated row-wise (catalog `noc_placement`) |

## Circular Buffers

Page size `TB = tile_bytes(input dtype)` (2048 for bf16), `T4 = 4096` (fp32 tile), `TA = tile_bytes(affine dtype)`. `K = block_c_tiles`, `H = block_hw_tiles`, `Q = chunk_hw_tiles`, `D = STREAM_DEPTH`, `GT = gather_tiles`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | `TB` | resident: `core_hw_tiles·K`; streaming: `D·Q·K` | spans `C` tiles; resident spans `HW`, streaming streams over it | input dtype | reader (TILE) / compute (RM tilize) | compute | resident: image round (all 3 passes); streaming: per block |
| `cb_input_sticks` (RM only) | 1 | `TB` (tile-sized pages, 32 sticks of `K·64 B` = `K` pages) | `D·K` | one tile-row of stick slices, double-buffered; streams over `HW` | input dtype | reader | compute (tilize) | per tile-row |
| `cb_scaler` | 2 | 2048 | 1 | template-required scaler CB for `reduce`; never read on the `AccumulateViaAdd` path (constant) | bf16 | reader (once, `prepare_reduce_scaler<SUM,REDUCE_COL>(1.0f)`) | compute | program |
| `cb_membership` | 3 | `T4` | `K·Ng` | spans `C` tiles × group slots; constant over `HW` and images | fp32 | writer | compute (matmul in1, `NoWaitNoPop`) | program |
| `cb_gamma_rows` (when gamma) | 4 | `TA` | `K` | one row-0 tile per owned column | affine dtype | reader | compute | program |
| `cb_beta_rows` (when beta) | 5 | `TA` | `K` | same | affine dtype | reader | compute | program |
| `cb_colsum_accum` | 6 | `T4` | `K` | raw partial-sum tiles for cross-chunk `Accumulate`; spans `C` tiles | fp32 | compute | compute | within a pass |
| `cb_colsum_rows` | 7 | `T4` | `K` | finalized per-channel sums (pass 1) / centered-square sums (pass 2); spans `C` tiles | fp32 | compute (reduce) | compute (matmul in0) | pass 1 then pass 2 (sequential reuse) |
| `cb_partial_rows` | 8 | `T4` | `Ng` | this core's per-group partials (row 0) | fp32 | compute (matmul) | writer (multicast source) | per round |
| `cb_gather` | 9 | `T4` | `2·Ng·GT` | two rounds × `Ng` slots × `GT` row-tiles; landing buffer for remote writes | fp32 | writer (push after count) | compute (reduce_mean) | per image (half per round) |
| `cb_group_mean` | 10 | `T4` | `Ng` | group means, row 0 lane `g%32` | fp32 | compute (reduce_mean) | compute (expansion in0, `WaitAndRetainOnLastBlock`) | image round |
| `cb_group_rstd` | 11 | `T4` | `Ng` | var from `reduce_mean`, transformed in place to `rsqrt(var+eps)` | fp32 | compute | compute | image round |
| `cb_mean_rows` | 12 | `T4` | `K` | expanded per-channel mean rows (pass-2 B operand); re-expanded in pass 3 as the operand of `shift` | fp32 | compute (matmul) | compute (chain B, `InputTileMapping::Row`) | pass 2; pass 3 setup |
| `cb_scale_rows` | 13 | `T4` | `K` | expanded rstd rows, multiplied in place by gamma → `scale` | fp32 | compute | compute (pass-3 chain B) | pass 3 |
| `cb_shift_full` | 14 | `T4` | `K` | full 32-row fp32 tiles `beta − mean⊙scale` (DestReuse operand, non-broadcast) | fp32 | compute (`unary_bcast<Row>`) | compute (pass-3 chain) | pass 3 |
| `cb_fp32_scratch` | 15 | `T4` | `Q·K` | one chunk of centered squares (pass 2); the pass-3 fused chain does not use it (kept for the two-chain perf-lamp alternative) | fp32 | compute (chain) | compute (reduce) | per chunk |
| `cb_output_tiles` | 16 | `TB` | `D·Q·K` | one chunk of output tiles, double-buffered | output dtype | compute | writer (TILE) / compute untilize (RM) | per chunk |
| `cb_output_sticks` (RM only) | 17 | `TB` | `D·K` | one tile-row of output sticks, double-buffered | output dtype | compute (untilize) | writer | per tile-row |
| `cb_masked_mean` (`hw_mask` programs — Refinement 1 for ragged RM shards, Refinement 2 for `HW % 32 != 0` in any placement; **supersedes** the planned `cb_hw_mask`) | 21 | `T4` | `2·Ng` | the landed group-mean tiles with the rows outside the image zeroed, one set per masked head / tail tile-row; the expansion matmul turns them into `mean_full[r][c] = mask[r]·mean_c`, so the pass-2 chain subtracts a full tile and squares — no mask multiply element (+1.1–10 KB of --dev code, past the kernel-config ring) | fp32 | writer | compute | pass 2 |

Semaphores: `sem_round[0]`, `sem_round[1]` (`ttnn.SemaphoreDescriptor`, initial 0, over the whole rectangle).

Compute config: `ttnn.ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)` (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:639-654`).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `build_membership_block` | `K·Ng` tiles | dataflow fill (`fill_l1_range`) | RT args `C, Cg, t0, K, Ng` | `cb_membership`, `K·Ng` | pushed once; compute waits once, never pops |
| 2 | `load_block` (TILE) | `H×K` tiles | raw `TensorAccessor` reads (dataflow, no compute helper applies) | DRAM | `cb_input_tiles`, `H·K` | resident: pages stay until `release_block` |
| 2' | `load_block` (RM) | `32·H` sticks × `K·64 B` | `read_sticks_for_tilize` + `tilize` | `cb_input_sticks`, `K` per tile-row | `cb_input_tiles`, `H·K` | as above |
| 3 | `colsum_block` | `H×K → K` rows | `reduce` | `cb_input_tiles` (`H·K`; resident: `WaitUpfrontNoPop`; streaming: `BulkWaitBulkPop`) | `cb_colsum_rows`, `K` (via `cb_colsum_accum` when `Accumulate`) | — |
| 4 | `group_partial_rows_block` | `(1×K)@(K×Ng)` | `matmul_block` | `cb_colsum_rows` (`K`, popped), `cb_membership` (`K·Ng`, `NoWaitNoPop`) | `cb_partial_rows`, `Ng` | — |
| 5 | `allgather_round_block` | `Ng` rows out, `GT×Ng` in | dataflow raw NoC + `Semaphore<>`; compute `reduce_mean` | `cb_partial_rows` (`Ng`, popped by writer), `cb_gather` half (`Ng·GT`) | `cb_group_mean` (round 0) / `cb_group_rstd` (round 1), `Ng` | gather half popped |
| 6 | `finalize_rstd_block` | `Ng` tiles | `eltwise_chain` in place | `cb_group_rstd` (`Ng`) | `cb_group_rstd` (`Ng`) | — |
| 7 | `expand_rows_block` | `K` × `(1×Ng)@(Ng×1)ᵀ` | `matmul_block<transpose=true>` | `cb_group_mean`/`cb_group_rstd` (`Ng`, retained), `cb_membership` at base `T·Ng` | `cb_mean_rows` or `cb_scale_rows`, `K` | group stat CB still fronted |
| 8 | `scale_rows_block` | `K` row tiles | `mul` (in place) | `cb_scale_rows` (`K`), `cb_gamma_rows` (`K`, held) | `cb_scale_rows` (`K`) | skipped when no gamma |
| 9 | `shift_full_block` | `K` row tiles → `K` full tiles | `eltwise_chain` (Mul, DestReuse Sub/Negative) + `unary_bcast<Row>` | `cb_mean_rows` (`K`, popped), `cb_scale_rows` (`K`, held), `cb_beta_rows` (`K`, held) | `cb_shift_full`, `K` | — |
| 10 | `centered_sq_chunk` | `grid(Q, K)` | `eltwise_chain` | `cb_input_tiles` (resident: `None/None` + `Offset`; streaming: `PerBlockSize`), `cb_mean_rows` (`K`, `Upfront`/no pop, `Row` mapping, `BroadcastDim::Row`) | `cb_fp32_scratch`, `Q·K` | — |
| 11 | `colsum_accumulate_chunk` | `Q×K → K` rows | `reduce` with `Accumulate` | `cb_fp32_scratch` (`Q·K`, `BulkWaitBulkPop`), `cb_colsum_accum` | `cb_colsum_rows` on the last chunk | — |
| 12 | `apply_chunk` | `grid(Q, K)` | `eltwise_chain` | `cb_input_tiles` (as 10; pass-3 pops on the streaming path), `cb_scale_rows` (`Row` mapping, `BroadcastDim::Row`), `cb_shift_full` (`Row` mapping, DestReuse operand) | `cb_output_tiles`, `Q·K` | — |
| 13 | `untilize_chunk` (RM) | `K` wide × `Q` rows | `untilize` | `cb_output_tiles` (`Q·K`) | `cb_output_sticks`, `K` per tile-row | — |
| 14 | `store_chunk` | `Q·K` tiles / `32·Q` stick slices | raw `TensorAccessor` writes / `write_sticks_after_untilize` | `cb_output_tiles` or `cb_output_sticks` | DRAM | — |
| 15 | `release_block` | — | CB ops | resident: `cb_pop_front(cb_input_tiles, core_hw_tiles·K)`; pop `cb_group_mean`, `cb_group_rstd`, `cb_scale_rows`, `cb_shift_full` | — | ring closes exactly once per image |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `colsum_block` | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, cb_input_tiles, cb_scaler, cb_colsum_rows, policy, INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::AccumulateViaAdd>(ReduceInputBlockShape::of(H, K), contiguous(), acc)` | `reduce_helpers_compute.hpp:616-637`; algorithm `:156` and its contract `:118-155`; policies `:112`; `Accumulate::at/at_last` `:419-426`; `ReduceInputBlockShape::of` `:290-301` | resident: `policy = WaitUpfrontNoPop`, `acc = NoAccumulation{}`; streaming: `policy = BulkWaitBulkPop`, `acc = Accumulate::at(cb_colsum_accum, block_idx)` / `at_last(...)` (static_assert: Accumulate ⇒ BulkWaitBulkPop, `.inl:850-853`) | `cb_input_tiles` | `cb_colsum_rows` (+ `cb_colsum_accum`) | `H`, `K` (the `ReduceInputBlockShape`) |
| `colsum_accumulate_chunk` | helper | same `reduce` with `BulkWaitBulkPop`, `Accumulate::at(cb_colsum_accum, chunk_idx)` / `Accumulate::at_last(...)` on the last chunk | as above | `ReduceInputBlockShape::of(Q, K)` | `cb_fp32_scratch` | `cb_colsum_rows` | `Q`, `K` |
| `group_partial_rows_block` | helper | `compute_kernel_lib::matmul_block<false, false, LastBlockTarget::Out, OutputCBLayout::TileRowMajor, InitMode::Short, InputPolicy::WaitAndPopPerKBlock, InputPolicy::NoWaitNoPop>(in0 = cb_colsum_rows, in1 = cb_membership, out = cb_partial_rows, interm = cb_partial_rows, MatmulBlockShape::of(1, Ng, 1, 1, K, 1))` | `matmul_block_helpers.hpp:339-371`; `MatmulBlockShape::of` `:160-168`; `InputPolicy` `:77`; CB-distinctness and interm placeholder `:242-250, :306-314`; precision rule `:253-258` | `in0_num_subblocks=1, in1_num_subblocks=Ng, out_subblock_h=1, out_subblock_w=1, in0_block_k=K, num_k_blocks=1` — each output subblock is one DEST tile, so `Ng` is not DEST-capped | `cb_colsum_rows`, `cb_membership` | `cb_partial_rows` | `K` (in0_block_k), `Ng` (in1_num_subblocks) |
| `expand_rows_block` | helper | `matmul_block<true /*transpose in1*/, false, Out, TileRowMajor, Short, InputPolicy::WaitAndRetainOnLastBlock, InputPolicy::NoWaitNoPop, …, In1BaseOffsetFn>(in0 = cb_group_mean or cb_group_rstd, in1 = cb_membership, out = cb_mean_rows or cb_scale_rows, interm = out, MatmulBlockShape::of(1, 1, 1, 1, Ng, 1), …, in1_base_offset_fn = {T·Ng})` called for `T = 0..K-1`; pop in0 manually after the loop | as above; `In1BaseOffsetFn` contract `:223-227`; `transpose` `:277` | `K'=Ng, N=1, M=1` per call; `transpose=true` makes `out[0][c] = Σ_g stat[0][g]·M_T[c][g]` | `cb_group_*`, `cb_membership` | `cb_mean_rows` / `cb_scale_rows` | `Ng` (K'), `K` (call count) |
| `allgather_round_block` (compute side) | helper | `compute_kernel_lib::reduce_mean<ReduceDim::REDUCE_COL, cb_gather, cb_scaler, cb_group_mean /*or cb_group_rstd*/, ReduceInputPolicy::BulkWaitBulkPop, INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::AccumulateViaAdd>(ReduceInputBlockShape::of(GT, Ng), HW·Cg)` | `reduce_helpers_compute.hpp:675-691`; `n_reduced` semantics `:647-652` | rows of the gather tiles = cores; result row 0 lane `g%32` = mean / var of group `g` | `cb_gather` (half) | `cb_group_mean` / `cb_group_rstd` | `GT`, `Ng` |
| `allgather_round_block` (writer side) | raw_api | `noc_async_write_multicast_loopback_src(src_l1, get_noc_multicast_addr(rect…, dst_l1), 64, num_dests)` ×2 per slot tile; `noc_async_write_barrier()`; `Semaphore<>::up(noc, x_j, y_j, 1)` per active core; receiver `Semaphore<>::wait_min((image+1)·num_active)`; `cb_reserve_back/cb_push_back(cb_gather, Ng·GT)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1700`; `noc_semaphore.h:137` (`up` remote), `:264-271` (`wait_min`) | **Helpers considered and rejected**: `dataflow_kernel_lib::SenderPipe/ReceiverPipe` (`mcast_pipe.hpp:132-231`) — precondition "one active sender per round" (`mcast_pipe.hpp:33`) and `ReceiverPipe::receive` waits on a single sender's flag/counter (`NUM_SENDERS` only sizes rotating-sender coordinates, `:242-262`); here `num_active` cores send concurrently into every core. The catalog's proven multi-sender pattern is raw write + monotone counter (`tensix_all_reduce/program_descriptor_with_inline_kernels.py:784-801`) | `cb_partial_rows` | `cb_gather` | `Ng`, `GT`, `num_active` |
| `finalize_rstd_block` | helper | `compute_kernel_lib::eltwise_chain(IterationShape::tiles(Ng), CopyTile<input(cb_group_rstd, PerTile, PerTile)>, AddUnary<>{eps}, Rsqrt<>, PackTile<output(cb_group_rstd, PerTile, PerTile)>)` | `eltwise/api/chain.hpp:530-531` (chain), `:480-481` (CopyTile), `:501-502` (PackTile); `eltwise/unary/scalar.hpp:25-26` (AddUnary); `eltwise/unary/math.hpp:37-38` (Rsqrt); in-place lifecycle rule `tests/eltwise/chain/lifecycle/inplace_chain.cpp:8-15` | in place: PerTile pop + PerTile reserve/push | `cb_group_rstd` | `cb_group_rstd` | `Ng` |
| `scale_rows_block` | helper | `compute_kernel_lib::mul<input(cb_scale_rows, PerTile, PerTile), input(cb_gamma_rows, Upfront, None, Row…), output(cb_scale_rows, PerTile, PerTile)>(IterationShape::tiles(K))` (`BroadcastDim::None`; both tiles row-0-valid) | `eltwise/api/convenience.hpp:50-51` | skipped when `!has_gamma` | `cb_scale_rows`, `cb_gamma_rows` | `cb_scale_rows` | `K` |
| `shift_full_block` | helper | per `T`: `eltwise_chain(one_tile(), BinaryFpu<Mul, input(cb_mean_rows…), input(cb_scale_rows…)>, [has_beta ? DestReuseBinary<Sub, input(cb_beta_rows…), DEST_TO_SRCB> : Negative<>], PackTile<output(cb_mean_rows)>)` then `unary_bcast<BroadcastDim::Row, input(cb_mean_rows…), output(cb_shift_full…)>(one_tile())` | `chain.hpp:483-492` (BinaryFpu), `:497-499` (DestReuseBinary), `:448-451` (DestReuseType); `eltwise/unary/misc.hpp:15-44` (Negative); `eltwise/broadcast/bcast.hpp:27-28` (unary_bcast) | computes `shift = beta − mean⊙scale` (or `−mean⊙scale`) in fp32 DEST, then replicates row 0 down 32 rows | `cb_mean_rows`, `cb_scale_rows`, `cb_beta_rows` | `cb_shift_full` | `K` |
| `centered_sq_chunk` | helper | `eltwise_chain(IterationShape::grid(Q, K), BinaryFpu<Sub, input(cb_input_tiles, wait, pop, Block, …, addressing), bcast_input(cb_mean_rows, Upfront, None, InputTileMapping::Row, BroadcastDim::Row)>, Square<>, PackTile<output(cb_fp32_scratch, PerBlockSize, PerBlockSize)>)` | `chain.hpp:121-148` (IterationShape), `:230-235` (InputTileMapping::Row = "indexed by column"), `:259-264` (TileAddressing::Offset for the resident block), `:303-308` (BroadcastDim::Row = srcB row 0 replicated); `eltwise/unary/misc.hpp` (Square) | resident: `wait=None, pop=None, addressing=Offset{chunk_idx·Q·K}`; streaming: `PerBlockSize/PerBlockSize` | `cb_input_tiles`, `cb_mean_rows` | `cb_fp32_scratch` | `Q`, `K`, `block_size` (chain-clamped) |
| `apply_chunk` | helper | `eltwise_chain(IterationShape::grid(Q, K), BinaryFpu<Mul, input(cb_input_tiles…), bcast_input(cb_scale_rows, Upfront, None, Row, BroadcastDim::Row)>, DestReuseBinary<Add, input(cb_shift_full, Upfront, None, InputTileMapping::Row), DEST_TO_SRCA>, PackTile<output(cb_output_tiles, PerBlockSize, PerBlockSize)>)` | as above; `chain.hpp:497-499` — `DestReuseBinary` takes a plain `InputSpec` (no broadcast), which is why `cb_shift_full` is pre-expanded to full tiles | `y = x·scale + (beta − mean·scale)`; pack format = output dtype (chain-owned reconfig) | `cb_input_tiles`, `cb_scale_rows`, `cb_shift_full` | `cb_output_tiles` | `Q`, `K` |
| `load_block` (RM) | helper | reader `dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks>(accessor, 32 /*rows*/, K·64, image·HW + 32·row, 64·t0)`; compute `compute_kernel_lib::tilize<K, cb_input_sticks, cb_input_tiles>(H)` | `tilize_helpers_dataflow.hpp:87-93`; `tilize_helpers.hpp:187-197` | `block_width_tiles = K` is the tilize block knob | `cb_input_sticks` | `cb_input_tiles` | `K`, `H` |
| `untilize_chunk` | helper | `compute_kernel_lib::untilize<K, cb_output_tiles, cb_output_sticks>(Q)` | `untilize_helpers.hpp:145-154` | — | `cb_output_tiles` | `cb_output_sticks` | `K`, `Q` |
| `store_chunk` (RM) | helper | `dataflow_kernel_lib::write_sticks_after_untilize<cb_output_sticks>(accessor, 32, K·64, image·HW + 32·row, 64·t0)` | `tilize_helpers_dataflow.hpp:129-135` | — | `cb_output_sticks` | DRAM | `K` |
| `load_block` / `store_chunk` (TILE) | raw_api | `TensorAccessor::get_noc_addr(page_id)` + `noc_async_read` / `noc_async_write`, page id `= (image·HWt + row)·Ct + col` | `tech_reports/tensor_accessor/tensor_accessor.md`; `dataflow_api.h` | **Helpers considered and rejected**: `read_sticks_for_tilize` / `write_sticks_after_untilize` are stick-indexed RM helpers (`tilize_helpers_dataflow.hpp:60-135`) — the TILE leg moves tile pages, which no dataflow helper wraps. Batch reads per block, one barrier per block (catalog `double_buffer`, `tile_reorder`) | — | `cb_input_tiles` / DRAM | `H`, `K`, `Q` |
| `build_membership_block` | raw_api | writer: `dataflow_kernel_lib::fill_l1_range<4>(tile_addr, 4096, 0)` then `32` scalar `float` stores at `tile_addr + face(c,g)·1024 + (c%16)·64 + (g%16)·4` where `face = (c/16)·2 + (g/16)` | `l1_helpers.hpp:89` (fill_l1_range), `:50-54` (zero_tile) | **Helpers considered and rejected**: `prepare_reduce_mask<cb, REDUCE_ROW>(valid)` (`reduce_helpers_dataflow.hpp:73-74`) writes a *prefix* row-0 mask (1s in the first `valid` lanes) — a group's lanes are an arbitrary `[lo, hi)` sub-range and the membership matrix needs one column per group; `generate_mask_w` (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:223`) also prefix-only. No helper builds a 0/1 matrix from `(C, G, T)` | RT args | `cb_membership` | `K`, `Ng` |
| `cb_scaler` fill | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_COL>(1.0f)` | `reduce_helpers_dataflow.hpp:58-60` | pool-type-aware overload as required | — | `cb_scaler` | — |
| boot | helper | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_colsum_rows, cb_membership, cb_partial_rows)` once at the top of the compute kernel; every helper call keeps its default `INPUT_AND_OUTPUT` reconfig so each phase re-arms its own formats | `matmul_block_helpers.hpp:100-107, 236-239`; `reduce_helpers_compute.hpp:31-35` | Reverse is mandatory for the matmul; reduce/chain re-config per call | — | — | — |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| pass 2 `centered_sq_chunk` | `Sub` | `cb_input_tiles` — All (2D `[H,W]`) | `cb_mean_rows[T]` — Row0 (matmul output of a row-0-valid in0) | `BroadcastDim::Row` (B row 0 down all rows), B indexed by column `T` (`InputTileMapping::Row`) |
| pass 3 `apply_chunk` | `Mul` | `cb_input_tiles` — All | `cb_scale_rows[T]` — Row0 | `BroadcastDim::Row`, B by column |
| pass 3 `apply_chunk` | `Add` (DestReuse) | DEST (All) | `cb_shift_full[T]` — All (pre-expanded by `unary_bcast<Row>`) | `None` |
| `scale_rows_block` | `Mul` | `cb_scale_rows[T]` — Row0 | `cb_gamma_rows[T]` — Row0 | `None` (elementwise; rows 1-31 are 0·0) |
| `shift_full_block` | `Mul`, `Sub` | `cb_mean_rows[T]` — Row0 | `cb_scale_rows[T]`, `cb_beta_rows[T]` — Row0 | `None` |
| `unary_bcast` in `shift_full_block` | copy | `cb_mean_rows[T]` (holding shift) — Row0 | — | `BroadcastDim::Row` → output All |
| pass-2 masked segments (`hw_mask`) | `Sub` (plain, no broadcast) | `cb_input_tiles` — All | `cb_mean_rows` expanded from `cb_masked_mean` — All (full masked-mean tile) | `None` (the row mask is inside the operand, not a chain element) |

## Reduce Direction Verification

| Logical Dim | Tile ReduceDim | Output Valid Region | BroadcastDim | ReduceInputBlockShape | EltwiseShape |
|-------------|----------------|--------------------|--------------|-----------------------|--------------|
| `HW` (tile rows) per channel — `colsum_block` / `colsum_accumulate_chunk` | `REDUCE_COL` | Row0 (`reduce.h:50`: each column's sum at row 0) | consumed as `BroadcastDim::Row` operands after expansion | `of(H or Q, K)` → `K` outputs | `grid(Q, K)` for the producing chain |
| cores (gather rows) per group lane — `allgather_round_block` | `REDUCE_COL` (`reduce_mean`) | Row0 | — (consumed as matmul in0) | `of(GT, Ng)` → `Ng` outputs | — |
| `Cg` lanes within/across tiles | **not a tile reduce** — the membership matmul `(1×K)@(K×Ng)` | Row0 | — | — | — |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Per-tile reduction across a straddling tile mixes groups | `C/G ∈ {10,20,30,…}` puts 2–4 groups in one tile; `reduce<REDUCE_SCALAR>` rejects partial scalers | The op never reduces over lanes with a tile reduce. Pass 1/2 reduce **rows only** (`REDUCE_COL`, per channel); the group aggregation is `colsum_rows @ membership`, where the membership matrix is the skill's per-(group, tile) mask rows stacked as columns. The expansion site is the transposed matmul. Both mask sites, one CB, built from RT args (`C, G, t0, K`) |
| Silent NaN from padded lanes | `0 × NaN = NaN` in the aggregation matmul if padded channel lanes / uninitialized L1 hold NaN bit patterns | TILE inputs are zero-padded by ttnn; the RM stick CB is zero-filled once at program start and only valid bytes are ever written; membership rows for channels `≥ C` are zero |
| Centered variance contract | one-pass `E[x²] − mean²` is forbidden | Pass 2 computes `Σ (x − mean_row)²` with the mean from pass 1 (`cb_mean_rows`, expanded per channel). The reduce forms only the centered second moment |
| tf32 operand rounding on the FPU (WH/BH SrcA/SrcB are 19-bit) | fp32 stat tiles are rounded to tf32 when consumed by `add_tiles`, `matmul`, `*_bcast_rows`: mean relative error ≈ `2⁻¹¹` per rounding (≈ 2 roundings) → per-group output offset ≈ `2·2⁻¹¹·|mean|/σ` and variance error `(2·2⁻¹¹·|mean|/σ)²` (second order) | Acceptable for bf16 inputs (their own quantization is `2⁻⁸·|mean|`); documented; `exact_sfpu_stats` regime row is the fp32-input refinement. Never store stat CBs as bf16 (would be `2⁻⁸`). Test includes a `mean = 10σ` case |
| `matmul_block` fidelity rule | `HiFi4 + fp32_dest_acc_en` with **bf16** inputs corrupts the K-accumulator on WH B0 (#38306) | both matmul inputs are fp32 CBs (`cb_colsum_rows`, `cb_membership`); compute config `HiFi4`, `fp32_dest_acc_en=True` |
| Ack-free gather reuse | remote cores write into this core's `cb_gather` before the local compute popped it | two halves (one per round) + the causality argument in Dataflow Strategy; rows `≥ num_active` zeroed once, disjoint from any remote write; `wait_min` on a monotone counter, target `(image+1)·num_active`, never reset |
| Landing address must be identical on all cores | remote writers compute `dst` from their **own** CB address | every core in the rectangle (active or idle) allocates every CB identically; `cb_gather` capacity is exactly `2·Ng·GT` pages and is pushed/popped `Ng·GT` per round, so its write pointer is deterministic (`base` for round 0, `base + half` for round 1) |
| Sender inside its own multicast rectangle | plain `noc_async_write_multicast` excludes the source | use `noc_async_write_multicast_loopback_src` (`dataflow_api.h:1700`) so the core's own row lands too; barrier before the semaphore increments |
| Idle cores in the rectangle | `rect_x·rect_y > num_active` for non-rectangular core counts | idle cores get `is_active=0`, exit immediately, never increment semaphores; they still own the CB memory so multicast landings are safe |
| Ring-wrap of `cb_input_tiles` in the resident regime | capacity `= core_hw_tiles·K` pages, pushed once and popped once per image: pointer returns exactly to base each image | pop the **whole** block in `release_block`; `Accumulate` is not used on the resident pass 1 (`WaitUpfrontNoPop`) |
| `WaitAndRetainOnLastBlock` group-stat in0 | the expansion matmul is called `K` times on the same `Ng`-tile in0 | the policy retains in0; the caller pops `cb_group_*` in `release_block` after pass 3 (mean is needed for `shift`) |
| Chain broadcast limit | only the first `BinaryFpu` may broadcast; `DestReuseBinary` takes a plain `InputSpec` (`chain.hpp:497-499`) | `shift` is pre-expanded to full tiles (`cb_shift_full`); alternative (two bcast chains) is a perf lamp |
| `hw_non_aligned` (**built, Refinement 2**) | padded rows are `(0 − mean)² ≠ 0` in pass 2 | the ragged last tile-row runs as its own pass-2 segment whose mean operand is expanded from the writer's masked group-mean tiles (`cb_masked_mean`) — `(0 − 0)² = 0` on the zero pad rows; pass 1 is unaffected (TILE padding is zero, the RM reader zeroes pad sticks), counts use the true `HW`; in the streaming regime the reader chunks pass 2 in the same `[head][body][tail]` order (`pass2_segments`) |
| `c_non_aligned` (**built, Refinement 2**) | RM stick slices end mid-tile; TILE padding lanes | per-core `(c0, c_valid = min(K·32, C − c0))` (Refinement 1's valid-lane pair): membership rows / gamma-beta lanes `≥ c_valid` are zero (stats exact even with garbage pad lanes), expansion yields 0 output lanes; RM reader reads `c_valid·elem` B per stick into a once-zeroed stick CB, writer writes `c_valid·elem` B per stick; an odd `c_valid` copies whole words (the half-word beyond it is zero scratch) |
| `compute_kernel_hw_startup` order | matmul needs `SrcOrder::Reverse`; reduce/chain assume default | boot once with `Reverse` on the matmul CBs; all helpers keep `INPUT_AND_OUTPUT` reconfig so each call re-arms formats |
| Regime only exercised on some grids | `resident` depends on `num_cores` and `L1_CB_BUDGET_BYTES` | `config.FORCE_STREAMING` test pin; `Ng = 2` pinned by a `G = 64` case |
| RM read fragmentation under a C-split | stick slices of `K·64 B` | `SPLIT_ORDER = hw_first` keeps `K` as large as the cap allows; RM `K·64 B ≥ 640 B` for SD shapes (`c_splits = 1`) |

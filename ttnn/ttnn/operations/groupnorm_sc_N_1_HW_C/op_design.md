# Operation Design: groupnorm_sc_N_1_HW_C

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (reduction + cross-core combine + elementwise normalize/affine) |
| Goal | GroupNorm over a channel-last `(N, 1, H*W, C)` tensor: per (image, group) statistics over `HW × C/G`, then per-element normalize and optional per-channel affine. Groups may straddle 32-channel tile boundaries (`(C/G) % 32 != 0`) from Phase 0; the tile-aligned case is the same code path. One device program per call. |
| Math | `y[n,0,h,c] = (x[n,0,h,c] − mean[n,g(c)]) · rsqrt(var[n,g(c)] + eps) · gamma[c] + beta[c]`, `g(c) = c // (C/G)`, `mean/var` over the `HW · C/G` elements of group `g` in image `n` (biased variance, as `torch.nn.functional.group_norm`). |
| Mode | Derivative (torch reference: `torch.nn.functional.group_norm` on the `(N, C, HW)` permutation — see `eval/golden_tests/groupnorm_sc_N_1_HW_C/helpers.py`) |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/skills/groupnorm-partial-channels/SKILL.md`, `.claude/skills/partial-scaler-reduce/SKILL.md`, `ttnn/cpp/ttnn/kernel_lib/{reduce_helpers_compute,reduce_helpers_dataflow,matmul_block_helpers,tilize_helpers,tilize_helpers_dataflow,mcast_pipe}.hpp`, `ttnn/cpp/ttnn/kernel_lib/eltwise/api/{chain,convenience}.hpp`, `ttnn/ttnn/operations/examples/master.md` (entries cited inline), `eval/golden_tests/groupnorm_sc_N_1_HW_C/feature_spec.py`, `eval/op_template.py` |

### The one idea everything else follows from

The group mask of the `/groupnorm-partial-channels` skill is realised as a **0/1 membership matrix per channel tile**, `E_T[g, c] = 1 ⇔ channel T·32+c belongs to group g` (a `32×32` tile per `(T, k)` with `k` indexing 32-group slabs). Every group of a tile is one row of `E_T`, so:

* **mask site 1 (reduce side)** — `gsum[g] = Σ_T Σ_c colsum_T[c] · E_T[g, c]` is a matmul `[S_T] × [E_Tᵀ]` with `K = channel tiles`; the padded lanes of a ragged last channel tile are columns of zeros.
* **mask site 2 (expand side)** — `mean_T[c] = Σ_g mean[g] · E_T[g, c]` is a matmul `[mean_g] × [E_T]`.

Both are per-tile matmuls whose operands are `row-valid` tiles; the per-group statistics live in **lane form** (`lane g of row 0`), so the whole per-image statistic is `2·ceil(G/32)` tiles regardless of `HW`, `C`, or how many tiles a group straddles. That compactness is what makes the cross-core combine cheap and what keeps the L1 working set independent of `HW` and `C`. `E_T` is generated in-kernel by the reader from `(C, G, T)` runtime args — no host mask tensor.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 4, `shape[1] == 1`, on device, interleaved | — | — |
| `num_groups` | `int` | yes | `1 ≤ G ≤ C`, `C % G == 0` | — | RT (`G`, `Cg = C/G`) |
| `gamma` | `ttnn.Tensor` or `None` | no | `(1, 1, 1, C)`, any TARGET `affine_dtype/affine_layout` | `None` | CT flag `has_gamma`; RT address |
| `beta` | `ttnn.Tensor` or `None` | no | `(1, 1, 1, C)`, same dtype/layout as gamma | `None` | CT flag `has_beta`; RT address |
| `eps` | `float` | no | `> 0` | `1e-5` | RT (fp32 bits) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` or `None` | no | per `.claude/references/precision_convention.md`: `fp32 input + fp32_dest_acc_en=False` refused | `default_compute_kernel_config()` → `HiFi4`, `fp32_dest_acc_en=True`, `math_approx_mode=False` | host |
| `cols_per_group` (knob) | int | derived | `1 ≤ v ≤ min(Ct_core, DEST_AUTO_LIMIT)` | `min(Ct_core, DEST_AUTO_LIMIT)` | CT |
| `chunk_tiles_target` (knob) | int | derived | `≥ 1` | `32` (→ `chunk_rows = clamp(32 / cols_per_group, 1, Ht_core)`) | CT |
| `x_depth` (knob) | int | derived | `≥ 1` | `2` (streaming regime only) | host |
| `membership_depth` (knob) | int | derived | `≥ 1` | `1` | host |
| `out_depth_tiles` (knob) | int | derived | `≥ 1` | `2 · cols_per_group` | host |
| `min_tiles_per_core` (knob) | int | derived | `≥ 1` | `1` | host |
| `max_cores` (knob) | int | derived | `1 … grid size` | full `device.compute_with_storage_grid_size()` | host |
| `l1_budget_bytes` (knob) | int | derived | — | `min(1_000_000, worker L1 − 200 KiB)` | host |
| `reduce_algorithm` (knob) | enum | derived | `Auto`, `AccumulateViaAdd` | `Auto` | CT |
| `set_l1_budget_bytes_override(v)` / `set_max_cores_override(v)` | module-level functions | required by the acceptance test | `None` clears the override | `None` | host — the **regime-pin contract**: they override the two host knobs above so the acceptance test can force `streaming_2d` (`budget = 0`) and `single_core_per_image` (`max_cores ≤ N`) without touching the public signature |

Index-axis convention: this op has no `dim` parameter. The channel axis is always `-1` and the spatial axis always `-2` of the 4-D input; `validate()` canonicalises nothing beyond rank/`dim[1]` checks.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(N, 1, HW, C)`; padded TILE grid per image is `Ht = ceil(HW/32)` × `Ct = ceil(C/32)` tiles, page id `n·Ht·Ct + r·Ct + t` |
| Dtype | TARGET: `bfloat16`, `float32`, `bfloat8_b` (SUPPORTED narrows) |
| Layout | TARGET: TILE or ROW_MAJOR (RM is tilized in-kernel; the rest of the pipeline is layout-agnostic) |
| Memory | DRAM or L1 **interleaved** (TARGET has no `memory_layout` axis; sharded placement is a knob-turn — see Dataflow Strategy) |
| gamma / beta | `(1, 1, 1, C)`; RM = one stick of `C` elements; TILE = `Ct` tiles whose row 0 holds the values; dtype independent of the input dtype |

### Output

| Property | Value |
|----------|-------|
| Shape | `(N, 1, HW, C)` (same logical shape, same per-image tile padding) |
| Dtype | `== input_tensor.dtype` |
| Layout | **TILE_LAYOUT always** (both input layouts) |
| Memory | DRAM interleaved (`ttnn.DRAM_MEMORY_CONFIG`) |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Notation: `Ht`, `Ct` per image; `Kg = ceil(G/32)`; `P_n` cores per image (`Pr × Pc` logical grid); `Ht_core`, `Ct_core` this core's rows/cols; `n_g = Cg · HW` (real HW).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `n` (image) | independent — statistics never cross images | `images_per_core` | `1` when `N < num_cores`, else `ceil(N/num_cores)` (looped sequentially) | host `assign_images()` → RT `image_begin, image_count` | one **rectangle** of `P_n` cores per image (`P_n = 1` when `N ≥ num_cores`) | knob-turn (rectangle shape) |
| `hw` (spatial tile-rows, `Ht`) | **dependent** for the statistics (reduced), independent for the apply | `chunk_rows` (rows per block); `Ht_core` (rows per core) | `chunk_rows = clamp(chunk_tiles_target / cols_per_group, 1, Ht_core)`; `Ht_core = Ht / Pr` (ceil/floor balanced) | host → CT `chunk_rows`, RT `row_begin, Ht_core` | `Pr = min(P_n, Ht)` cores cut it; partial `(sum, sumsq)` combined at the image root | built (the combine is Phase 0) |
| `ct` (channel tiles, `Ct`) | **dependent within a group** (a group's channels span 1..k tiles and are reduced), independent across groups | `cols_per_group` (tiles per block); `Ct_core` | `cols_per_group = min(Ct_core, DEST_AUTO_LIMIT)`; `Ct_core = Ct / Pc` | host → CT `cols_per_group`, RT `col_begin, Ct_core` | `Pc = min(P_n // Pr, Ct)` cores cut it; the per-group partial is already summed over this core's tiles, so the cut needs no neighbour exchange — the same root combine covers it | built |
| `g` (groups, `G`) | independent — one statistic per group; carried as **lanes** of `Kg` tiles | `Kg` | whole extent (`Kg` tiles, never cut across cores) | host → RT `G`, `Kg` | not assigned across cores: a group split would co-own output tiles and re-read every straddled tile up to `⌈32/Cg⌉+1` times (regime `group_split`, rejected) | — |
| `lane` (32 channels inside a tile) | dependent within a group — mixed groups inside one tile | fixed 32 (hardware tile) | whole tile; the membership matrix `E_T` selects lanes | reader generates `E_T` from RT `C, G` | never cut (tile is atomic) | — |
| `c_affine` (gamma/beta channel) | **reuse-shared by construction of the split** — does not vary along `hw`, so every one of the `Pr` cores in a tile-column re-reads its `Ct_core` slices | `cols_per_group` (rows read per block) | `cols_per_group` row-0 tiles per block | same as `ct` | every core reads its own slice (`≤ 128 B` per tile for bf16/fp32 row-0 reads; whole 1088 B page for bf8b) | broadcast regime `mcast_affine` rejected (payload too small — see Regimes) |

Every knob has a single host source (`create_program_descriptor` computes it once and passes it as CT/RT args; kernels never restate a literal). Loop bounds in every kernel are `num_blocks_this_core`-style parameters derived from these.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_pass1` / `cb_x_pass2` (aliased) | `x_depth` (streaming regime) | `2` chunks | reader fills chunk `i+1` while compute reduces / applies chunk `i` (`examples/double_buffer`: 1.74× → 2.78× with batching + depth 2) |
| `cb_x_rm` (RM input only) | `x_rm_depth` | `2` tile-rows of sticks | reader/tilize overlap |
| `cb_membership` | `membership_depth` | `1` block | `2` lets the reader generate `E` for group `k+1` while compute aggregates `k` (lamp) |
| `cb_out` | `out_depth_tiles` | `2 · cols_per_group` tiles | writer drains while compute packs the next tiles |
| all statistic CBs | — | `1` block (live set = capacity) | intermediates between sequential compute helpers must hold the full block (`ttnn-cb-memory-fundamentals.md` → Intermediate CB Sizing) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `reduce<SUM, REDUCE_COL>` ReduceTile datapath, `BulkWaitBulkPop`: bulk = `rows × chunk`, `chunk = DEST_AUTO_LIMIT` columns (`reduce_helpers_compute.hpp:98-101`) | `cols_per_group` | `cols_per_group ≤ DEST_AUTO_LIMIT` (8 with `fp32_dest_acc_en`, `dest_helpers.hpp:103`) | the helper splits columns into DEST chunks and waits `rows × chunk` at a time — the reader's tile order would no longer match the bulk it waits for → hang or wrong columns |
| `matmul_block` subblock width `out_subblock_w = Kg` (`matmul_block_helpers.hpp:139-170`) | `Kg` per output subblock | `in1_num_subblocks = ceil(Kg / DEST_AUTO_LIMIT)`, `out_subblock_w = min(Kg, DEST_AUTO_LIMIT)`; Phase 0 has `Kg ≤ 8` for every INPUTS shape | DEST overflow (silent corruption) |
| gather tile rows: 32 records per tile | `P_used` cores per image | `ceil(P_used/32)` gather tiles per statistic; rows `≥ P_used` zero-filled by the root before the ready signal | a record lands outside the tile / uninitialised rows enter the sum |
| `matmul_block` precision contract (`matmul_block_helpers.hpp:258-264`): bf16 inputs + HiFi4 + fp32 DEST corrupt the K-accumulator on WH B0 | membership / statistic page formats | **all matmul operands are `Float32` pages** (`E`, `colsum`, `stats_g_full`) → "fp32 inputs: HiFi4 + fp32_dest_acc_en" is the sanctioned cell | silent wrong sums |
| `fp32_dest_acc_en` on ⇒ `DEST_AUTO_LIMIT = 8` | every DEST-resident extent above | all block extents derived from `DEST_AUTO_LIMIT`, never a literal `8`/`16` | DEST overflow |
| `read_sticks_for_tilize` pads the L1 stride with stale data for non-aligned widths (`tilize_helpers_dataflow.hpp:49-52`) | `c_non_aligned` RM input (refinement) | zero-fill the pad lanes of `cb_x_rm` before the first push (`0·NaN = NaN` otherwise) | NaN statistics |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `resident_2d` | **built** | `resident = block_bytes + fixed_bytes ≤ l1_budget_bytes`, `block_bytes = Ht_core · Ct_core · x_page_bytes`, `fixed_bytes` = the L1 ledger total minus the x ring | pass 1 in blocks of `chunk_rows × cols_per_group` streamed through the aliased ring **without eviction**; pass 2 reads the whole resident `Ht_core × Ct_core` block from L1 | **minimum** against DRAM: `x` crosses once, `y` once; gamma/beta `Pr ×` a `≤128 B/tile` read; cross-core: `P_used` records of `2·Kg·128 B` + one multicast of `2·Kg` tiles per image | the fixed costs are the per-chunk reduce init/uninit + reconfig (`compute_block_size`: ~1.6 µs per extra pass), one NoC barrier per chunk, and the per-column-group aggregation K-block; a larger `chunk_rows` pays them fewer times |
| `streaming_2d` | **built** | `!resident` | `chunk_rows × cols_per_group` tiles, `x_depth`-deep ring; pass 2 re-streams the same blocks in the same order | `x` crosses DRAM **twice** (pass 1 statistics, pass 2 apply): `+1 × input bytes` above minimum; `y` once; cross-core identical to `resident_2d` | same fixed costs as above plus the second stream's per-chunk barrier |
| `single_core_per_image` | **built** (a parameterisation of the two rows above: `P_n = 1`) | `N ≥ num_cores` (or `tiles_per_image < 2·min_tiles_per_core`) | one core owns the whole `Ht × Ct` image; regime selection between resident/streaming applies unchanged | as the enclosing row; cross-core traffic degenerates to the root's local copy (`mcast_pipe` self-only box → local copy, `mcast_pipe.hpp:184`) | as above; additionally amortises the combine handshake over `images_per_core` images |
| `group_split` (cores own whole groups, cut along `g`) | **rejected** — superseded by the 2-D split + membership matmul | — | a core owns `HW × Cg` channels of several groups | every channel tile straddled by `k` groups is read by `k` cores (`⌈32/Cg⌉+1 ≈ 4×` for `Cg = 10`), and an output tile shared by several groups cannot be written by several cores without read-modify-write | — |
| `mcast_affine` (broadcast gamma/beta once per tile-column) | **rejected** — payload is `≤ 128 B` per tile per core; `examples/shared_input_reuse` and `examples/mcast_topology` show multicast pays only for multi-MB shared streams and loses when the per-chunk handshake dominates | — | — | would remove `(Pr−1) · Ct_core · 128 B` of L1-interleaved/DRAM reads per image | — |
| `all_gather_combine` (every core multicasts its record; every core reduces `P_used` records) | **deferred** — the root's combine work is `2·Kg·ceil(P_used/32)` tile-reduces (≈ µs), so nothing measured motivates the `P_used` concurrent multicasts; reachable because the record format and landing layout are identical — only the sender rectangle changes | — | same | replaces `P_used` unicasts + 1 multicast with `P_used` multicasts | — |
| `shifted_two_pass_variance` (`var = E[(x−m)²]`) | **deferred** — needs a third read of `x` (or a resident block); `E[x²] − m²` in fp32 DEST meets the golden tolerances on the test distribution (`randn`); revisit only if a precision baseline on offset-heavy activations fails | — | — | `+1 × input bytes` in the streaming regime; free in the resident regime | — |
| `sharded_input` (HEIGHT / WIDTH / BLOCK sharded x) | **deferred** — not in TARGET (no `memory_layout` axis); a **knob-turn**: the shard grid is `Pr × Pc`, the shard is the core's block, consumed zero-copy through `ttnn.cb_descriptor_from_sharded_tensor` as `cb_x_pass1/2`; the combine is unchanged | `input_tensor.memory_config().is_sharded()` | whole shard | minimum (no DRAM read at all) | — |

Regime selection is host-side and exact: `resident` is computed from the ledger expression below; `P_n` from `assign_images()`. **Regime-pinned tests are required** (the resident predicate depends on the grid size: a shape resident on 64 cores may stream on 8) — the acceptance test parametrises `l1_budget_bytes` down to force `streaming_2d` on a shape that is resident by default, and pins `single_core_per_image` with `N ≥ num_cores` (batch 8 on an 8-core cap) — see `test_regime_streaming_forced` / `test_regime_single_core_per_image` in the acceptance test.

### Traffic ranking

All candidate splits move the input **once** (resident) or **twice** (streaming) and the output once — the input varies along both `hw` and `ct`, so no split re-reads `x`. What differs is (a) the combine, (b) reuse-shared reads, (c) occupancy.

| Candidate split | DRAM crossings (`x`, `y`) | Cross-core traffic per image | Reuse-shared reads | Fills the grid? |
|---|---|---|---|---|
| `hw` only (`Pc = 1`) | 1–2, 1 | `P_used · 2·Kg·128 B` + 1 mcast of `2·Kg` tiles | gamma/beta `Pr × Ct × ≤128 B` | only when `N·Ht ≥ num_cores` (fails for `HW = 32…128` wide-C shapes) |
| `ct` only (`Pr = 1`) | 1–2, 1 | identical (the record is per-group, not per-tile) | none | only when `N·Ct ≥ num_cores` (fails for narrow-C shapes) |
| **2-D `hw × ct` (chosen)** | 1–2, 1 | identical | gamma/beta `Pr × Ct_core × ≤128 B` (smaller than `hw`-only) | yes: `Pr·Pc ≤ P_n ≤ min(cores, Ht·Ct)` |
| `g` (group) split | `≈ (⌈32/Cg⌉+1)×` input | none | none | no (`N·G ≤ 32` for SD shapes) and output tiles co-owned → rejected |
| no split (`P_n = 1`) | 1–2, 1 | none | none | only for `N ≥ num_cores` (used there) |

The dependent-axis combine costs `P_used · 2·Kg · 128 B` (≤ 16 KiB per image at 64 cores) against the bytes residency saves (one whole read of `x`, MBs) — the residency term, not the combine, decides the ranking, and the 2-D split is the only one that reaches residency on every INPUTS shape at ≥ 56 cores. Chosen: **2-D `hw × ct` split with a per-group lane-form combine at an image root**, `resident_2d` when the predicate holds.

**Operand-reuse check** (per operand × chosen split): `x` varies along both cut axes → no reuse. `gamma`, `beta` do not vary along `hw` → reuse-shared by construction; broadcast rejected on payload size (row above). `E_T` is generated, not read. **Stall-shadow check**: the only stage that waits on a peer is the combine (all cores wait for the root's totals). Work independent of the totals scheduled into that window: the reader prefetches pass-2's first `x_depth` chunks (streaming) and the first column group's `E` tiles and gamma/beta rows; the root's writer zero-fills the gather tiles and issues the ready signal during pass 1; every core's `E`-tile generation and gamma/beta row fetches for pass 1 are done between DRAM read issues. The apply itself depends on the totals — nothing else can be lifted around the wait, and the loop nest is not reordered.

### Block schedule

Logical schedule per core (reader, compute, writer realise their parts asynchronously; the two passes of adjacent images are not pipelined — an image's combine is a barrier). Comments give block shape, residency and intended fixed-cost frequency.

```cpp
for (uint32_t img = 0; img < images_this_core; ++img) {                 // 1 unless N >= num_cores
  // ---------- pass 1: statistics ----------
  prepare_constants(img);        // reader: scaler tile (once per kernel); root writer: zero-fill gather tiles
                                 //         + gather-ready signal (once per image)
  for (uint32_t cg = 0; cg < num_col_groups; ++cg) {                    // Ct_core / cols_per_group blocks
    for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {                  // Ht_core / chunk_rows blocks
      load_x_block(cg, rc);        // reader: chunk_rows x cols_per_group tiles (TILE: DRAM->cb_x; RM: sticks->cb_x_rm)
      tilize_block(cg, rc);        // compute, RM only: cb_x_rm -> cb_x (aliased ring)
      square_block(cg, rc);        // compute: x^2 for the chunk -> cb_xsq (chunk-sized, fp32 pages)
      colsum_block(cg, rc);        // compute: REDUCE_COL SUM of x and of x^2 over chunk_rows, Accumulate across rc
                                   //          -> cb_colsum = [S_T.. | Q_T..] (2 x cols_per_group row-0 tiles)
    }
    aggregate_groups_block(cg);    // compute: [S;Q](2 x K) x E^T(K x Kg) K-block cg of ONE matmul_block call,
                                   //          spill/reload across cg -> cb_partial (2 x Kg tiles, lane form)
  }                                // reduce init/uninit once per row chunk; matmul init once per column group
  // ---------- combine (dependent axes hw, ct across the P_used cores of the image) ----------
  send_partial_record(img);        // writer: after gather-ready, unicast row 0 of the 2*Kg partial tiles into
                                   //         row p of the root's gather tiles; semaphore inc. Once per image.
  root_reduce_records(img);        // root compute: REDUCE_COL SUM over ceil(P_used/32) gather tiles per stat
                                   //               -> cb_totals_src; root writer multicasts to the rectangle.
  finalize_stats(img);             // every core: mean_g = sum/n_g; var = max(sumsq/n_g - mean^2, 0);
                                   //             rstd = rsqrt(var+eps); broadcast row 0 -> full tiles
                                   //             -> cb_stats_g_full (2*Kg tiles, resident through pass 2)
  // ---------- pass 2: apply ----------
  for (uint32_t cg = 0; cg < num_col_groups; ++cg) {
    build_affine_block(cg);        // compute: per T in group: [mean;rstd]_T = stats_g_full x E_T (K = Kg);
                                   //          a_T = rstd_T * gamma_T ; b_T = beta_T - mean_T * a_T
                                   //          -> cb_a_full, cb_b_full (cols_per_group full tiles each)
    for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
      load_x_block(cg, rc);        // streaming: DRAM -> cb_x_pass2 (same order as pass 1); resident: credit only
      apply_block(cg, rc);         // compute: y = x * a_T + b_T over chunk_rows x cols_per_group, one DEST
                                   //          window per tile, no intermediate CB -> cb_out
      store_block(cg, rc);         // writer: cb_out -> DRAM tiles of this core's block, one barrier per chunk
    }
  }
}
```

Residency across operations: `cb_x_pass2` holds the whole per-core block across both passes in `resident_2d`; `cb_colsum` persists across the row chunks of one column group; `cb_partial`/`cb_stats_g_full` persist from the end of pass 1 through pass 2; `cb_a_full`/`cb_b_full` persist across the row chunks of one column group.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap — `chunk_tiles_target = 32` | 32 tiles per chunk (64 KiB bf16) delays the first reduce by a full chunk's DRAM latency and puts 128 KiB of `x^2` pages in L1; `examples/double_buffer` saturates at 4–8 tiles per barrier, `examples/compute_block_size` charges ~1.6 µs per extra helper pass | `chunk_tiles_target ∈ {8, 16, 64}` with `x_depth = 2`; `x_depth = 3` at 16 |
| Grid synchronisation — `min_tiles_per_core = 1` | for tiny images (`Ht·Ct < 16`) a `P_n`-core group pays the gather + multicast handshake for a few tiles of work each | `min_tiles_per_core ∈ {4, 8}` (fewer, fatter cores per image) |
| Reduce datapath — `reduce_algorithm = Auto` (ReduceTile) | `examples/reduce_block`: `AccumulateViaAdd` wins on COL from ≥ 8 reduced tiles (3.4× at 32) and is more accurate for bf16; it also removes the `DEST_AUTO_LIMIT` cap on `cols_per_group` | `ReduceAlgorithm::AccumulateViaAdd` (still `BulkWaitBulkPop` + `Accumulate`, `reduce_helpers_compute.hpp:129-155`) |
| `cb_xsq` page format `Float32` | ledger rule (page follows DEST width) doubles the bytes through the packer/unpacker for a value the FPU re-reads as tf32 anyway | `Float16_b` pages for bf16/bf8b inputs (accept one extra rounding of `x²`) |
| Apply realisation — full `a_T`/`b_T` tiles + `DestReuseBinary` | `examples/compute_fusion`: the DEST→src reuse premium is per tile (≈0.46–2.3 µs per 8–32 tiles) | seed DEST with `unary_bcast<ROW>(b_T)` and accumulate `mul_tiles_bcast_rows(x, a_T)` (`acc_to_dest=1`) from **row-0** `a_T`/`b_T` tiles (halves `cb_a_full+cb_b_full`, skips two bcasts per T) |
| K-accumulation of the aggregation — software spill/reload | `packer_l1_acc` avoids the reload unpack per column group | `matmul_block<…, packer_l1_acc=true, LastBlockTarget::Interm, OutputCBLayout::TileRowMajor>` |
| `membership_depth = 1` | the reader generates `E` for group `k+1` only after compute pops group `k` | `membership_depth = 2` |
| Math fidelity `HiFi4` (precision-convention default) | the apply's FPU multiplies on bf16 need ≤ HiFi2 for exact products | `HiFi2` for pass 2 (`compute_kernel_config` is honoured, never overridden) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `x` DRAM → `cb_x_pass1` (TILE input) | input dtype tiles, `chunk_rows × cols_per_group` per push, tile order **column-group-major, row-major within the chunk** | reader: `TensorAccessor` page reads, all tiles of a chunk in flight, one `noc_async_read_barrier`, one `cb_push_back(chunk_tiles)` to **both** aliased ids (`cb_x_pass1` and `cb_x_pass2`) in `resident_2d`, to `cb_x_pass1` only in `streaming_2d` | page id `n·Ht·Ct + r·Ct + t`, `r` per image (`Ht = ceil(HW/32)`); ragged last chunk pushes the **nominal** count and reads only its valid tiles |
| `x` DRAM → `cb_x_rm` → `cb_x_pass1` (RM input) | 32 sticks × `cols_per_group·32·es` bytes per tile-row block | reader: `read_sticks_for_tilize<cb_x_rm>(acc, 32·chunk_rows, cols_per_group·32·es, start_page = n·HW + r·32, byte_offset = col_begin·32·es)`; compute: `tilize<cols_per_group, cb_x_rm, cb_x_pass1>(chunk_rows)` then credits `cb_x_pass2` (resident) | the same aliased ring; the tilize is the CB producer instead of the reader |
| `E_T` generation | `Float32` 0/1 tiles, `cols_per_group × Kg` per column group | reader writes `1.0f` at `[g − 32k][c]` (pass 1: the **transpose**, `[c][g − 32k]`) for `c < C − 32T`, `g = (32T + c) // Cg`; the CB pages are zeroed once at kernel start and only the previous block's ones are cleared per refill (`≤ 2·32` stores per tile) | tile byte layout: 4 faces of 16×16, face `(r ≥ 16)·2 + (c ≥ 16)`, offset `face·1024 + (r%16)·64 + (c%16)·4` bytes |
| gamma / beta → `cb_gamma_row`, `cb_beta_row` | row-0-valid tiles in the affine dtype, `cols_per_group` per column group | RM: read `32·es_g` bytes of the `(1,1,1,C)` stick at byte `T·32·es_g` into row 0 (two 16-lane face chunks); TILE bf16/fp32: read the two row-0 chunks of tile `T`; TILE bf8b: read the whole 1088 B page | rows 1..31 of the CB page are zero-filled once at kernel start (never written afterwards) |
| scaler → `cb_scaler` | one `Float16_b` tile | `calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_COL>()` once per kernel | `hw_non_aligned` refinement: `calculate_and_prepare_partial_reduce_scalers<…, REDUCE_COL, HW % 32>()` → 2 tiles, `with_partial()` on the chunk that holds the image's last tile-row |
| partial record → root | `2·Kg` rows of `32 × 4 B` | writer: after `ReceiverPipe::receive_signal()` (gather-ready), `noc_async_write` of row 0 of each partial tile (2 chunks of 64 B: face 0 and face 1 row 0) into row `p` of gather tile `p / 32` on the root (`p % 16 < 16` → faces 0/1, else 2/3), `noc_async_write_barrier`, `noc_semaphore_inc(root SEM_GATHER, 1)` | the root sends to itself through the same path; the landing address is the root's `cb_gather` base (identical allocation on every core in the rectangle) |
| totals → every core | `2·Kg` `Float32` tiles (`sum_g`, `sumsq_g` in lane form) | root writer: `noc_semaphore_wait(SEM_GATHER, P_used)`, `cb_push_back(cb_gather)`; after compute pushes `cb_totals_src`: `SenderPipe::send(src, dst = cb_totals_recv base, 2·Kg·4096)` to the image rectangle (sender in rect, `PRE_HANDSHAKE = false`); every core's writer: `ReceiverPipe::receive()` then `cb_push_back(cb_totals_recv, 2·Kg)` | host: one `ttnn.Mcast2D(device, rect, root, McastConfig(sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED], handshake=False))` per image rectangle; `P_n = 1` → degenerate self box → local copy |
| `y` `cb_out` → DRAM | output dtype tiles (TILE layout) | writer: `TensorAccessor` page writes, one barrier per chunk (`out_depth_tiles` in flight) | page id as for `x`; NoC1 for writes, NoC0 for reads (`examples/noc_placement`: 4–5× between the two) |

Sharded placement (not in TARGET): a HEIGHT/WIDTH/BLOCK shard is the `Pr × Pc` block already defined here; it would be consumed through `ttnn.cb_descriptor_from_sharded_tensor` as `cb_x_pass1/2` with no NoC re-read, and the combine is unchanged — a knob-turn on placement, recorded as the `sharded_input` regime row.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a `chunk_rows × cols_per_group` tile block of one image; a core owns `num_col_groups × num_row_chunks` of them per image |
| Grid | `device.compute_with_storage_grid_size()` capped by `max_cores`; `row_wise=True` ordering of cores (`examples/noc_placement`) |
| Image rectangles | `N ≥ num_cores`: `P_n = 1`, core `c` owns images `c, c + num_cores, …`. Otherwise: `N ≤ Gy` → rectangles of `Gx × (Gy // N)`; `N > Gy` → `k = ceil(N / Gy)` images per grid row, rectangles of `(Gx // k) × 1`. Active sub-rectangle: `P_target = min(rect_area, max(1, Ht·Ct // min_tiles_per_core))`; `P_target ≥ rect_w` → the first `P_target // rect_w` full rows, else `(P_target, 1)`. `P_n` = its area |
| In-image split | `Pr = min(P_n, Ht)`, `Pc = min(P_n // Pr, Ct)`, `P_used = Pr·Pc`; core `p < P_used` (row-major in the active rectangle) → `(i, j) = (p // Pc, p % Pc)`; rows `[i·Ht // Pr, (i+1)·Ht // Pr)`, cols `[j·Ct // Pc, (j+1)·Ct // Pc)`; cores `p ≥ P_used` exit immediately (no reads, no record, no receive) |
| Root | `p = 0` of the active rectangle; `SEM_GATHER` target `P_used` |
| Per-core work | `Ht_core × Ct_core` tiles, `num_col_groups = ceil(Ct_core / cols_per_group)`, `num_row_chunks = ceil(Ht_core / chunk_rows)`; ragged last group/chunk keeps nominal push/pop counts and narrows only the work |
| Remainder | tile counts per image with `ceil`: `Ht = ceil(HW/32)`, `Ct = ceil(C/32)`; per-image tile-row indexing, never `floor(N·HW/32)` |
| Regime selection | host: `resident` per the ledger predicate; `P_n`, `P_used`, rectangles as above; all three passed as RT args. Regime-pinned tests required (see Regimes) |

## Circular Buffers

`x_page` = input tile bytes (2048 bf16 / 4096 fp32 / 1088 bf8b); `F32 = 4096`; `g_page` = affine tile bytes; `chunk = chunk_rows · cols_per_group`; `cols = cols_per_group`; `P_max = max(P_used)` over images.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_pass1` | 0 | `x_page` | `resident ? Ht_core·Ct_core : x_depth·chunk` | spans `hw`,`ct` block extents (`chunk`) × depth; in `resident_2d` spans the whole per-core block (predicate-guarded) | input dtype | reader (TILE) / compute-tilize (RM) | compute (`square_block`, `colsum_block`) | pass 1 |
| `cb_x_pass2` | 1 | `x_page` | same as `cb_x_pass1` — **aliased on the same L1 region** (one `CBDescriptor`, two `CBFormatDescriptor`s) | independent credit counter over the same bytes: resident → whole block waited once; streaming → same ring re-filled in pass 2 | input dtype | same producer as `cb_x_pass1` | compute (`apply_block`) | pass 2 |
| `cb_x_rm` | 2 | `x_page` (TILE granularity: 32 sticks = `cols` tile pages) | `x_rm_depth · cols` | RM only; one tile-row of sticks per block, depth 2 | input dtype | reader | compute (`tilize_block`) | pass 1 (+ pass 2 in streaming) |
| `cb_xsq` | 3 | `F32` | `chunk` | sequential helpers (`square` then `reduce`) → full block | `Float32` (lamp: `Float16_b`) | compute | compute | pass 1 |
| `cb_scaler` | 4 | 2048 | `1` (+1 for the `hw_non_aligned` partial pair) | constant | `Float16_b` | reader | compute | whole kernel |
| `cb_colsum` | 5 | `F32` | `2·cols` | `[S_T..; Q_T..]` for one column group = the matmul's `M=2 × K=cols` in0 block; doubles as the `Accumulate` accumulator across row chunks (pop `cols`, push `cols` per stat per chunk) | `Float32` | compute | compute | one column group of pass 1 |
| `cb_membership` | 6 | `F32` | `membership_depth · cols · Kg` | `E_Tᵀ` (pass 1) / `E_T` (pass 2) for one column group | `Float32` | reader | compute | per column group |
| `cb_agg_interm` | 7 | `F32` | `2·Kg` | `matmul_block` spill/reload partial across column groups | `Float32` | compute | compute | pass 1 |
| `cb_partial` | 8 | `F32` | `2·Kg` | `[gsum; gsq]` lane-form partial of this core | `Float32` | compute | writer | end of pass 1 → record sent |
| `cb_gather` | 9 | `F32` | `2·Kg · ceil(P_max/32)` | one row per contributing core per statistic tile; rows `≥ P_used` zero | `Float32` | writer (root; push after `SEM_GATHER == P_used`) | compute (root) | combine |
| `cb_totals_src` | 10 | `F32` | `2·Kg` | root's reduced totals = multicast source | `Float32` | compute (root) | writer (root) | combine |
| `cb_totals_recv` | 11 | `F32` | `2·Kg` | multicast landing (identical address on every core) | `Float32` | writer (push after `receive()`) | compute | combine → `finalize_stats` |
| `cb_stats_g_full` | 12 | `F32` | `2·Kg` | `[mean_g_full; rstd_g_full]` — the `M=2 × K=Kg` in0 of every pass-2 expansion matmul | `Float32` | compute | compute | pass 2 |
| `cb_gamma_row` | 13 | `g_page` | `cols` | row-0-valid gamma tiles of one column group (absent → not allocated) | affine dtype | reader | compute | per column group of pass 2 |
| `cb_beta_row` | 14 | `g_page` | `cols` | as gamma | affine dtype | reader | compute | per column group of pass 2 |
| `cb_stats_T` | 15 | `F32` | `2` | `[mean_T_full; rstd_T_full]` for one channel tile (transient) | `Float32` | compute | compute | per T |
| `cb_beta_full` | 16 | `F32` | `1` | `beta_T` broadcast to all rows (transient; absent → not allocated) | `Float32` | compute | compute | per T |
| `cb_a_full` | 17 | `F32` | `cols` | `a_T = rstd_T·gamma_T` full tiles for the column group | `Float32` | compute | compute | per column group of pass 2 |
| `cb_b_full` | 18 | `F32` | `cols` | `b_T = beta_T − mean_T·a_T` full tiles | `Float32` | compute | compute | per column group of pass 2 |
| `cb_out` | 19 | output tile bytes | `out_depth_tiles` | streaming window; pushes per tile so any capacity closes the ring | output dtype | compute | writer | pass 2 |
| `cb_stats_row` | 20 | `F32` | `2·Kg` — **aliased on `cb_totals_recv`'s region** (second `CBFormatDescriptor`) | `[mean_g; rstd_g]` in row-0 lane form, transient between the finalize chain and the `unary_bcast` | `Float32` | compute | compute | finalize |

Full ledger with capacity-vs-live-set, axis accounting and sharing decisions: `l1_ledger.md` beside this file.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `prepare_constants` | — | `calculate_and_prepare_reduce_scaler` (reader); zero-fill + `send_signal` (root writer) | — | `cb_scaler` (1) | scaler resident for the kernel; gather tiles zero, ready flag set |
| 2 | `load_x_block` | `chunk_rows × cols_per_group` | raw dataflow (`TensorAccessor` reads) / `read_sticks_for_tilize` (RM) | DRAM | `cb_x_pass1` (+`cb_x_pass2` credit in resident) `chunk`; RM: `cb_x_rm` | ring advanced by one chunk |
| 3 | `tilize_block` (RM only) | `chunk_rows × cols_per_group` | `tilize<cols_per_group, cb_x_rm, cb_x_pass1>(chunk_rows)` | `cb_x_rm` (`cols` per tile-row, popped) | `cb_x_pass1` (`chunk`) | compute credits `cb_x_pass2` by `chunk` in resident |
| 4 | `square_block` | `chunk` tiles | `square<input(cb_x_pass1, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block), output(cb_xsq)>(IterationShape::tiles(chunk))` | `cb_x_pass1` (`chunk`, **not popped**) | `cb_xsq` (`chunk`) | x chunk still fronted for op 5 |
| 5 | `colsum_block` | `ReduceInputBlockShape::of(chunk_rows, cols_per_group)` ×2 | `reduce<SUM, REDUCE_COL, cb_x_pass1, cb_scaler, cb_colsum, BulkWaitBulkPop>(shape, contiguous(), Accumulate::at(cb_colsum, rc))` then the same on `cb_xsq` | `cb_x_pass1` (`chunk`, popped), `cb_xsq` (`chunk`, popped), `cb_colsum` (reloaded `cols` per stat when `rc > 0`) | `cb_colsum` (`2·cols`: `[S..; Q..]`) | after the last chunk `cb_colsum` holds the column group's colsums |
| 6 | `aggregate_groups_block` | `M=2, K=cols_per_group, N=Kg`; K-blocks = column groups | `matmul_block<false, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor, InitMode::ShortAfterPreKBlock>(cb_colsum, cb_membership, cb_partial, cb_agg_interm, MatmulBlockShape::of(2, 1, 1, Kg, cols_per_group, num_col_groups))` with `PreKBlockFn` = ops 2–5 for column group `cg` | `cb_colsum` (`2·cols`), `cb_membership` (`cols·Kg`), both popped per K-block | `cb_partial` (`2·Kg`) after the last K-block; `cb_agg_interm` between | `cb_partial` fronted for the writer |
| 7 | `send_partial_record` | `2·Kg` rows | raw dataflow (`noc_async_write` ×2 per tile, barrier, `noc_semaphore_inc`) after `ReceiverPipe::receive_signal()` | `cb_partial` (`2·Kg`, popped) | root `cb_gather` region | — |
| 8 | `root_reduce_records` | `ReduceInputBlockShape::of(ceil(P_used/32), 2·Kg)` | `reduce<SUM, REDUCE_COL, cb_gather, cb_scaler, cb_totals_src, BulkWaitBulkPop>` (rows = gather tiles, cols = statistic tiles) | `cb_gather` (popped) | `cb_totals_src` (`2·Kg`) | writer multicasts, then pops |
| 9 | `finalize_stats` | `Kg` lane-form tiles | `eltwise_chain(IterationShape::tiles(Kg), CopyTile<input(cb_totals_recv, Upfront, None, Block), D0>, MulUnary<D0>{1/n_g}, CopyTile<input(cb_totals_recv, …, Offset Kg), D1>, MulUnary<D1>{1/n_g}, MulBinary<D0,D0,D2>, SubBinary<D1,D2,D1>, Relu<D1>, AddUnary<D1>{eps}, Rsqrt<Approx::Exact, Legacy::Off, D1>, PackTile<output(cb_stats_row), D0>, PackTile<output(cb_stats_row, …Offset Kg), D1>)` then `unary_bcast<BroadcastDim::Row, input(cb_stats_row), output(cb_stats_g_full)>(tiles(2·Kg))` | `cb_totals_recv` (`2·Kg`) | `cb_stats_g_full` (`2·Kg`, full tiles) | resident through pass 2 (`cb_stats_row` is a 2·Kg-page transient aliasable with `cb_totals_recv` — see ledger) |
| 10 | `build_affine_block` | per T: `M=2, K=Kg, N=1`; then 1-tile chains | `matmul_block<…, InputPolicy::WaitAndRetainOnLastBlock /*in0*/>(cb_stats_g_full, cb_membership, cb_stats_T, cb_stats_T, MatmulBlockShape::of(2,1,1,1,Kg,1))`; `mul<input(cb_stats_T, Upfront, AtEnd, Offset 1), input(cb_gamma_row, BroadcastDim::Row), output(cb_a_full)>(one_tile())` (no gamma: matmul packs straight into `cb_a_full` via `copy`); `unary_bcast<Row, cb_beta_row → cb_beta_full>`; `eltwise_chain(one_tile(), BinaryFpu<Mul, input(cb_stats_T…), input(cb_a_full, …Offset T)>, DestReuseBinary<Sub, input(cb_beta_full), DEST_TO_SRCB>, PackTile<output(cb_b_full)>)` (no beta: `Negative<>` instead of the DestReuse) | `cb_stats_g_full` (retained), `cb_membership` (`Kg`, popped), `cb_gamma_row`/`cb_beta_row` (popped per T) | `cb_a_full`, `cb_b_full` (`cols` each) | fronted for the column group |
| 11 | `apply_block` | `IterationShape::grid(chunk_rows, cols_per_group)` | `eltwise_chain(grid, BinaryFpu<Mul, input(cb_x_pass2, <x policy>, Block), input(cb_a_full, Upfront, None, InputTileMapping::Row)>, DestReuseBinary<Add, input(cb_b_full, Upfront, None, InputTileMapping::Row), DEST_TO_SRCA>, PackTile<output(cb_out, PerTile, PerTile)>)`; streaming x policy `PerBlockSize/PerBlockSize` (`block_size(chunk, FullBlock)`), resident x policy `None/None` with `TileAddressing::Strided{base = block offset, row_stride = cols_per_group}` and caller-managed wait/pop | `cb_x_pass2` (`chunk`), `cb_a_full`, `cb_b_full` (retained) | `cb_out` (`chunk`, pushed per tile) | resident: `cb_x_pass2` popped once after the last chunk of the image |
| 12 | `store_block` | `chunk` tiles | raw dataflow (`TensorAccessor` writes, one barrier per chunk) | `cb_out` (popped per tile after the barrier) | DRAM | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | helper | `compute_kernel_hw_startup(cb_x_pass1, cb_scaler, cb_out)` once, first statement of `MAIN()` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:59-60` | `SrcOrder::Regular`; every later helper call runs its own full data-format reconfig (`INPUT_AND_OUTPUT`), which is why one boot serves reduce, matmul and eltwise | — | — | — |
| `prepare_constants` | helper | `calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_COL>()` | `reduce_helpers_dataflow.hpp:97-99` | pool-type-aware overload; `Float16_b` deduced from the CB | — | `cb_scaler` | — |
| `prepare_constants` (hw refinement) | helper | `calculate_and_prepare_partial_reduce_scalers<cb_scaler, SUM, REDUCE_COL, HW % 32>()` | `reduce_helpers_dataflow.hpp:147-153` | `partial_positions = HW % 32` | — | `cb_scaler` (2) | — |
| `prepare_constants` (root) | helper | `SenderPipe::send_signal(VALID)` | `mcast_pipe.hpp:160` | `McastArgs<CT_BASE, RT_BASE>` from `ttnn.Mcast2D.compile_time_args()/runtime_args(core)` | — | — | — |
| `load_x_block` (TILE) | raw_api | `TensorAccessor::get_noc_addr(page)` + `noc_async_read` per tile, `noc_async_read_barrier` per chunk | `tech_reports/tensor_accessor/tensor_accessor.md`; `api/dataflow/dataflow_api.h` | page = `n·Ht·Ct + r·Ct + t` | DRAM | `cb_x_pass1` (+`cb_x_pass2`) | `chunk_rows`, `cols_per_group`, `x_depth` |
| `load_x_block` (RM) | helper | `read_sticks_for_tilize<cb_x_rm, TilizeGranularity::TILE>(acc, 32·chunk_rows, cols_per_group·32·es, n·HW + 32·r, col_begin·32·es)` | `tilize_helpers_dataflow.hpp:87-93` | `row_bytes` = the column group's width; `byte_offset_within_page` = column-group offset | DRAM | `cb_x_rm` | `chunk_rows`, `cols_per_group` |
| `tilize_block` | helper | `tilize<cols_per_group, cb_x_rm, cb_x_pass1>(chunk_rows)` | `tilize_helpers.hpp:187-197` | `block_width_tiles = cols_per_group`; `Fp32Mode::Fast` (FPU consumers downstream) | `cb_x_rm` | `cb_x_pass1` | `cols_per_group`, `chunk_rows` |
| `square_block` | helper | `square<input(cb_x_pass1, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block), output(cb_xsq)>(IterationShape::tiles(chunk))` | `eltwise/api/convenience.hpp:60-61`; policies `chain.hpp:206-235, 354-361` | Upfront wait keeps the chunk fronted for op 5 | `cb_x_pass1` | `cb_xsq` | `chunk` |
| `colsum_block` | helper | `reduce<PoolType::SUM, ReduceDim::REDUCE_COL, cb_x_pass1, cb_scaler, cb_colsum, ReduceInputPolicy::BulkWaitBulkPop, INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, reduce_algorithm>(ReduceInputBlockShape::of(chunk_rows, cols_per_group), contiguous(), Accumulate::at(cb_colsum, rc))` ×2 (`cb_xsq` second) | `reduce_helpers_compute.hpp:616-637` (signature), `:290-301` (shape), `:399-440` (`Accumulate`), `reduce_helpers_compute.inl:688-721` (accumulator reload = wait `cols`, `copy_tile`, pop `cols`) | `input_policy`, `algorithm` are knobs; `Accumulate::at(cb, rc)` with `rc = 0` skips the reload | `cb_x_pass1` / `cb_xsq`, `cb_scaler`, `cb_colsum` (acc) | `cb_colsum` | `chunk_rows`, `cols_per_group`, `reduce_algorithm` |
| `colsum_block` (hw refinement) | helper | same, `ReducePartialScaler::with_partial()` on the chunk containing the image's last tile-row | `reduce_helpers_compute.hpp:340-352, :385-389` | partial applies to the last reduce-dim tile of **that call** only | | | |
| `aggregate_groups_block` | helper | `matmul_block<false, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor, matmul_config::InitMode::ShortAfterPreKBlock, InputPolicy::WaitAndPopPerKBlock, InputPolicy::WaitAndPopPerKBlock, NoPostCompute, ColsumPreKBlockFn>(cb_colsum, cb_membership, cb_partial, cb_agg_interm, MatmulBlockShape::of(2, 1, 1, Kg, cols_per_group, num_col_groups))` | `matmul_block_helpers.hpp:339-371` (signature), `:139-170` (shape), `:92-98, 182-196` (`ShortAfterPreKBlock` + PreKBlockFn restore contract), `:241-252` (distinct CBs; interm own region), `:258-264` (fp32 operands → HiFi4 + fp32 DEST) | `PreKBlockFn(block, num_k_blocks, is_last)` runs ops 2–5 for column group `block`; in0 block = `[S_0..S_{K-1}, Q_0..Q_{K-1}]` (row-major `M×K`), in1 block = `[E^T_{T0,k..}, E^T_{T1,k..}, …]` (row-major `K×N`) | `cb_colsum`, `cb_membership` | `cb_partial` (interm `cb_agg_interm`) | `cols_per_group` (K per block), `num_col_groups`, `Kg` |
| `send_partial_record` | raw_api | `ReceiverPipe::receive_signal()`; `noc_async_write(src_row_chunk, root_noc_addr, 64)` ×2 per tile; `noc_async_write_barrier()`; `noc_semaphore_inc(root_sem_addr, 1)` | `mcast_pipe.hpp:222`; `api/dataflow/dataflow_api.h`; pattern: `examples/tensix_all_reduce` `reduce_root_mcast` | row `p`, faces `(p ≥ 16)·2 + {0,1}`, byte offset `face·1024 + (p % 16)·64` | `cb_partial` | root `cb_gather` | — |
| `root_reduce_records` | helper | `reduce<SUM, REDUCE_COL, cb_gather, cb_scaler, cb_totals_src, BulkWaitBulkPop>(ReduceInputBlockShape::of(ceil(P_used/32), 2·Kg))` | `reduce_helpers_compute.hpp:616-637` | rows = gather tiles per statistic, cols = `2·Kg` statistic tiles (row-major in `cb_gather`) | `cb_gather`, `cb_scaler` | `cb_totals_src` | `P_used` |
| totals multicast | helper | `SenderPipe::send(get_read_ptr(cb_totals_src), cb_totals_recv_base, 2·Kg·4096)`; receivers `ReceiverPipe::receive()` | `mcast_pipe.hpp:154-155, :218`; host `ttnn.Mcast2D` / `ttnn.McastConfig` (`ttnn-nanobind/mcast_host.cpp:44-46, 162`; `ttnn-python-utility-bindings.md` → Cross-Core) | `PRE_HANDSHAKE=false`, `DataReadySignal::Flag`, fixed sender = root inside the rect; degenerate `P_n = 1` → local copy (`mcast_pipe.hpp:184`) | `cb_totals_src` | `cb_totals_recv` | — |
| `finalize_stats` | helper | `eltwise_chain(IterationShape::tiles(Kg), …)` as in Block Operation Realization #9; `unary_bcast<BroadcastDim::Row, input(cb_stats_row, Upfront, AtEnd, Block), output(cb_stats_g_full)>(tiles(2·Kg))` | `chain.hpp:530-531`; `CopyTile :480-481`; `PackTile :501-502`; `MulUnary`/`AddUnary` `eltwise/unary/scalar.hpp:25-35`; `MulBinary`/`SubBinary` `eltwise/binary/sfpu/basic.hpp:21-28`; `Relu` `eltwise/unary/activations.hpp:21-22`; `Rsqrt` `eltwise/unary/math.hpp:37-38`; `unary_bcast` `eltwise/broadcast/bcast.hpp:27-28` | `1/n_g` and `eps` are fp32 bits RT args (`n_g = Cg·HW`, host-known — never summed from the mask) | `cb_totals_recv` | `cb_stats_g_full` | `Kg` |
| `build_affine_block` — expansion | helper | `matmul_block<false, false, LastBlockTarget::Out, SubblockMajor, InitMode::Short, InputPolicy::WaitAndRetainOnLastBlock, InputPolicy::WaitAndPopPerKBlock>(cb_stats_g_full, cb_membership, cb_stats_T, cb_stats_T, MatmulBlockShape::of(2, 1, 1, 1, Kg, 1))` per T | `matmul_block_helpers.hpp:64-77` (`WaitAndRetainOnLastBlock` keeps in0 resident), `:247-250` (interm placeholder = out when `num_k_blocks == 1`) | in0 `[mean_g_full(Kg); rstd_g_full(Kg)]` retained across all T; in1 = `E_T` (`Kg` tiles) | `cb_stats_g_full`, `cb_membership` | `cb_stats_T` (2 tiles: `mean_T_full`, `rstd_T_full`) | `Kg` |
| `build_affine_block` — `a_T` | helper | `mul<input(cb_stats_T, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block, Enabled, TileAddressing::Offset){1}, input(cb_gamma_row, BroadcastDim::Row, PerTile, PerTile), output(cb_a_full)>(IterationShape::one_tile())`; no gamma: `copy<input(cb_stats_T… Offset 1), output(cb_a_full)>(one_tile())` | `convenience.hpp:50-51, 86-87`; `BroadcastDim::Row` `chain.hpp:291-308`; `TileAddressing::Offset` `:241-259` | srcB row-0 gamma broadcast down the tile (srcB is the only FPU broadcast operand, `chain.hpp:325-337`) | `cb_stats_T[1]`, `cb_gamma_row` | `cb_a_full` | — |
| `build_affine_block` — `b_T` | helper | `unary_bcast<Row, input(cb_beta_row), output(cb_beta_full)>(one_tile())`; `eltwise_chain(one_tile(), BinaryFpu<BinaryFpuOp::Mul, input(cb_stats_T, Upfront, AtEnd, Block), input(cb_a_full, Upfront, None, Block, Enabled, Offset){T}>, DestReuseBinary<BinaryFpuOp::Sub, input(cb_beta_full), DestReuseType::DEST_TO_SRCB>, PackTile<output(cb_b_full)>)`; no beta: `Negative<>` replaces the `DestReuseBinary` | `BinaryFpu chain.hpp:483-492`; `DestReuseBinary :497-499` (`DEST_TO_SRCB`: CB → srcA, DEST → srcB ⇒ `beta − mean·a`); `Negative` `eltwise/unary/misc.hpp:18-19` | — | `cb_stats_T[0]`, `cb_a_full[T]`, `cb_beta_full` | `cb_b_full` | — |
| `apply_block` | helper | `eltwise_chain(IterationShape::grid(chunk_rows, cols_per_group)[.block_size(chunk, BlockTailSync::FullBlock)], BinaryFpu<Mul, input(cb_x_pass2, …), input(cb_a_full, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row)>, DestReuseBinary<Add, input(cb_b_full, Upfront, None, InputTileMapping::Row), DestReuseType::DEST_TO_SRCA>, PackTile<output(cb_out, ReservePolicy::PerTile, PushPolicy::PerTile)>)` | `chain.hpp:99-148` (grid shape), `:217-235` (`InputTileMapping::Row` = indexed by column, re-read per row), `:259-264` (`Strided` for the resident block) | streaming: x `PerBlockSize/PerBlockSize`; resident: x `None/None` + `StridedTileRange{block_base, cols_per_group}` with caller `cb_wait_front(cb_x_pass2, Ht_core·Ct_core)` once and `cb_pop_front` once per image | `cb_x_pass2`, `cb_a_full`, `cb_b_full` | `cb_out` | `chunk_rows`, `cols_per_group` |
| `store_block` | raw_api | `TensorAccessor::get_noc_addr(page)` + `noc_async_write` per tile, one `noc_async_write_barrier` per chunk, `cb_pop_front` | `tech_reports/tensor_accessor/tensor_accessor.md` | — | `cb_out` | DRAM | `out_depth_tiles` |

**Helpers considered and rejected (raw_api entries):**

| Raw entry | Candidate helper | Mismatch (file:line) |
|---|---|---|
| `load_x_block` (TILE), `store_block` | `read_sticks_for_tilize` / `write_sticks_after_untilize` (`tilize_helpers_dataflow.hpp:87-93, 129-135`) | stick-indexed accessors for RM ↔ tile conversion; the TILE path moves whole tile pages of a 2-D block — no dataflow helper wraps a strided tile-block read/write; `TensorAccessor` is the sanctioned primitive |
| `send_partial_record` | `SenderPipe::send` (`mcast_pipe.hpp:154-155`) | `send` multicasts one contiguous block to a rectangle from **one** sender per round (`mcast_pipe.hpp:33`); the gather is `P_used` concurrent unicasts of `2·Kg` non-contiguous 64 B face-row chunks landing at per-sender offsets — the `examples/tensix_all_reduce` `reduce_root_mcast` gather shape, which uses raw `noc_async_write` + `noc_semaphore_inc` |
| `E_T` generation | `prepare_reduce_mask<cb, REDUCE_ROW>(valid)` (`reduce_helpers_dataflow.hpp:73-74`) | fills `1.0` in the **first** `valid` lanes of row 0 (`reduce_helpers_dataflow.inl:88-132`); `E_T` needs ones at `[g][c]` for arbitrary lane ranges of up to 32 rows — a different pattern, written directly with the face-layout formula above |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `square_block` | `square` (x·x) | `cb_x_pass1` All | same | None |
| `aggregate_groups_block` | matmul `[S;Q] × Eᵀ` | `cb_colsum` Row0 (REDUCE_COL output) — rows 1..31 are don't-care because row `r` of the product depends only on row `r` of in0 | `cb_membership` All (0/1) | — (matmul) |
| `finalize_stats` | SFPU chain on lane form | `cb_totals_recv` Row0 | — | — |
| `finalize_stats` | `unary_bcast<Row>` | `cb_stats_row` Row0 → `cb_stats_g_full` All | — | Row |
| `build_affine_block` | matmul `[mean;rstd]_full × E_T` | `cb_stats_g_full` All (rows equal) | `cb_membership` All | — (matmul) |
| `build_affine_block` | `mul` (`a_T`) | `cb_stats_T[1]` All | `cb_gamma_row` Row0 | **Row** (srcB) |
| `build_affine_block` | `unary_bcast<Row>` (`beta_full`) | `cb_beta_row` Row0 → All | — | Row |
| `build_affine_block` | `Mul` then `DestReuse Sub` (`b_T`) | `cb_stats_T[0]` All × `cb_a_full[T]` All; then `cb_beta_full` All − DEST | — | None |
| `apply_block` | `Mul` (`x·a_T`) | `cb_x_pass2` All | `cb_a_full[T]` All | None |
| `apply_block` | `DestReuse Add` (`+ b_T`) | DEST All | `cb_b_full[T]` All | None |
| `root_reduce_records` | `REDUCE_COL` over gather tiles | `cb_gather` rows `< P_used` valid, rest zero | `cb_scaler` Row0 (helper-owned) | — |

Every operand that is broadcast is a **row-0-valid** tile broadcast with `BroadcastDim::Row` / `unary_bcast<ROW>`; every non-broadcast binary operand is a full tile. No column-broadcast or scalar-broadcast is used.

## Registry model (SUPPORTED / EXCLUSIONS suggestion — the implementer owns the final block)

| Axis (`feature_spec.TARGET` name) | Phase 0 suggestion | Why / how it grows |
|---|---|---|
| `dtype` | `[bfloat16]` (+ `float32`, `bfloat8_b` if the implementer verifies them — the design is dtype-agnostic: page formats come from the tensors, all statistics are `Float32`) | precision convention: `float32` + `fp32_dest_acc_en=False` → `EXCLUSIONS` |
| `layout` | `[TILE_LAYOUT, ROW_MAJOR_LAYOUT]` | RM adds ops 2(RM)/3 only |
| `alignment` | `["tile_aligned"]` | `hw_non_aligned`: partial COL scaler pair on the chunk with the image's last tile-row (`Mechanism caps`, `API Mapping`); `c_non_aligned`: already handled by `E_T`'s zero columns for TILE input; RM needs the pad-lane zero fill |
| `groups_alignment` | `["group_aligned", "group_straddling"]` — **both from Phase 0**; there is no gate on `(C/G) % 32` | same code path (`E_T` is dense-diagonal-blocks when aligned) |
| `affine` | `["gamma_beta", "gamma_only", "no_affine"]` | CT flags `has_gamma`, `has_beta` |
| `affine_dtype` | `[bfloat16, float32, "none"]` (+ `bfloat8_b` with TILE) | `"none"` is always legal and must never be refused |
| `affine_layout` | `[ROW_MAJOR_LAYOUT, TILE_LAYOUT, "none"]` | both are row-0 tile fetches |

`INPUT_TAGGERS`: `alignment` exactly as given in the requirements (C wins when both are off), plus `groups_alignment`, `affine`, `affine_dtype`, `affine_layout` mirroring `eval/golden_tests/groupnorm_sc_N_1_HW_C/axes.py`. `validate()` runs first in the entry point (SUPPORTED per axis → EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell`), then argument validation raises `ValueError` for rank ≠ 4, `shape[1] ≠ 1`, `C % num_groups ≠ 0`, gamma/beta shape ≠ `(1,1,1,C)`. **Message contract** (the acceptance test matches these substrings via the repo's `expect_error` fixture): the rank error mentions `rank`; the `shape[1]` error mentions `dim`; the divisibility error mentions `num_groups`; the affine-shape error mentions `gamma` (use `gamma`/`beta` naming per tensor — the test exercises gamma). The op exports `default_compute_kernel_config()`.

**Structural impossibilities spotted beyond `feature_spec.INVALID`:** none — `bfloat8_b` gamma with `ROW_MAJOR` is already listed there.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Per-tile reduction instead of per-group | a tile holds lanes of several groups; a scalar per tile is meaningless | statistics are **per-channel column sums** first (group-agnostic), grouped afterwards by the membership matmul — the reduce never mixes groups |
| `E_T` rows 1..31 / padded lanes | garbage in unused rows of a lane-form tile is harmless only if finite; `0 · NaN = NaN` | `E_T` pages zeroed once at start; `cb_gamma_row/beta_row` rows 1..31 zeroed once; gather pad rows zeroed before the ready signal; TILE inputs from `ttnn.from_torch` are zero-padded (documented assumption; RM pad lanes zero-filled in the `c_non_aligned` refinement) |
| Gather record arrives before the root zero-fills | a fast 1-tile core can finish pass 1 in ~2 µs | root zero-fills then `send_signal`; senders `receive_signal` before writing; the second Flag event (totals) cannot overwrite the first because every receiver consumed it before sending its record |
| `cb_colsum` used as both `Accumulate` accumulator and matmul in0 | the reload pops `cols` S tiles then pushes `cols` new S tiles, then the same for Q — order `[S..; Q..]` is restored after both reduces | capacity exactly `2·cols`; the two reduces per chunk always run S then Q; documented in the ledger |
| `matmul_block` after a reduce in the same K-loop | the reduce dirties unpack/math state | `InitMode::ShortAfterPreKBlock` — the helper restores after `PreKBlockFn` (`matmul_block_helpers.hpp:92-98`) |
| bf16 + HiFi4 + fp32 DEST K-accumulation corruption (WH B0, #38306) | the aggregation is a K-accumulating matmul | all matmul operands are `Float32` pages (`E`, `colsum`, `stats_g_full`) |
| `E[x²] − mean²` cancellation | large `|mean|/σ` loses bits | fp32 DEST accumulation, `Relu` before `+eps`, `shifted_two_pass_variance` deferred row |
| Regime depends on grid size | `resident` predicate and `P_n` change with the device | host-side selection from one predicate; regime-pinned tests force `streaming_2d` via `l1_budget_bytes` and `single_core_per_image` via `max_cores` |
| Ragged chunks / column groups and the CB ring | a ragged `push_back` count breaks the ring-wrap invariant | nominal quanta everywhere; only the NoC transfer / compute extent narrows; capacities are multiples of the quantum |
| Aliased `cb_x_pass1/2` | two credit counters over one region | streaming: both rings hold `x_depth·chunk` and are used in disjoint passes; resident: both hold the block, pushed with identical quanta from base — pointers stay in lockstep (`ttnn-cb-memory-fundamentals.md` → Aliased Buffers) |
| Idle cores inside an image rectangle | `P_used < P_n` when `Pr·Pc` does not tile the rectangle | idle cores exit; `SEM_GATHER` target is `P_used`; the multicast writes into their (allocated, unused) landing CB harmlessly |
| `n_g` derived in-kernel | summing the mask wastes compute and breaks on padded lanes | `1/n_g` fp32 bits RT arg, `n_g = (C/G)·HW` (real HW) |
| Output dtype | contract: output dtype == input dtype | packer converts fp32 DEST to the output page format; `cb_out` format is the output tensor's |

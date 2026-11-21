# Operation Design: softmax

The specification (`eval/specs/softmax/spec.py`) governs where this document and it disagree.

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (a reduction followed by a normalisation, fused in one program) |
| Goal | Numerically stable softmax along one dimension of a tile-layout tensor, as **one** `ttnn.generic_op` dispatch of a Metal 2.0 `ttnn.ProgramSpec` on Quasar. Phase 0 serves YOLOv8's calls: bfloat16, rank 4, `dim=-2`, DRAM or L1, every alignment. |
| Math | `m[c] = max_h x[h,c]`; `s[c] = Σ_h exp(x[h,c] − m[c])`; `out[h,c] = exp(x[h,c] − m[c]) · (1 / s[c])`, for every column `c` of every image; the padding rows of the last tile-row enter neither `m` nor `s` |
| References | `eval/specs/softmax/spec.py` (authoritative); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/` (`toy_row_passes` = this op's dealing and host shape, `toy_thread_plan/kernels/reader_mixed.cpp` = both ring protocols in one reader); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; catalog `ttnn/ttnn/operations/examples/master.md` entries `width_split`, `compute_block_size`, `compute_fusion` |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim (`libttsim.so`, slow dispatch, no clock) | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (probe on ttsim: `compute_with_storage_grid_size()` = 8-4) | target page + `tests/ttnn/unit_tests/operations/softmax/probes/probe_001.py` |
| L1 per core | 4,194,304 B per cluster, shared by its 4 NEOs and its DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` = 3,963,328 − 0 = **3,963,328 B** on this target | probe on ttsim (same probe) |
| Ring cap | one DataflowBuffer ring ≤ 1,048,560 B (1 MiB − 16 B) on the compute side; 32 rings per cluster, ≤ 24 active per NEO. No run-time query exists: the host constant `DFB_RING_MAX_BYTES` holds it | target page, *Cluster model* |
| Host program | `ttnn.ProgramSpec` (`ttnn.experimental.ProgramSpec` / `KernelSpec` / `DataflowBufferSpec` / `WorkUnitSpec`, one `ttnn.generic_op`) — `.claude/references/quasar_op_examples/` (copy the shape of `toy_row_passes/toy_row_passes_program_artifacts.py`) | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1, 2, 4}); 6 user DM cores, R + W ≤ 6; on STRIDED rings R and W each divide C; capacity counted per NEO | target page, *Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` and any program-factory device op (Metal 1, refused at kernel creation). bfloat8_b / bfloat4_b (no path on the part). `ReduceFp32Mode::Accurate` (static_assert, `reduce_helpers_compute.inl:79,92`) and the helper's SFPU reduce path (`reduce_helpers_common.hpp:50`). Not used: the reduce helper's indexed input policies (`WaitUpfrontNoPop`, `BulkWaitBulkPop`, `NoWaitNoPop`); `eltwise/binary/sfpu/minmax.hpp` (`BinaryMax` chain element); `UnpackToDest`. `fpu_math_fidelity` is always set explicitly; `pack_init(<ring>)` at every output-ring change | target page |

Every worked number below is for **Quasar, 32 clusters, 3,963,328 B budget**. Every predicate the op evaluates queries the device (`compute_with_storage_grid_size()`, `get_max_worker_l1_unreserved_size()`, `buffer_page_size()`), except the ring cap, which has no query and is one named host constant.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | Phase 0: bfloat16, TILE, rank 4, DRAM or L1 interleaved | — | — |
| `dim` | int | no | canonicalises to −1 or −2 (`dim if dim < 0 else dim − ndim`, R2) before any support check; anything else raises (X1); Phase 0 supports −2 only | −1 (signature default; refused in Phase 0 because it is −1) | host only |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` or tt-metal's device config (`WormholeComputeKernelConfig`) | no, keyword-only | any `math_fidelity`, any `math_approx_mode` (R5); `fp32_dest_acc_en=True` in Phase 0 | `None` → `default_compute_kernel_config()` = `ComputeConfigDescriptor(math_fidelity=HiFi4, math_approx_mode=False, fp32_dest_acc_en=True)` (R4: defined once, exported) | resolved on host |

Compute-config resolution (R4, R5, R10), in the entry point, in this order: `None` → `default_compute_kernel_config()`; otherwise read `fp32_dest_acc_en`, `math_fidelity`, `math_approx_mode` **as attributes** (never by type) and build one `ttnn.ComputeConfigDescriptor` from them (extra fields such as `packer_l1_acc`, `dst_full_sync_en`, `throttle_level` are ignored). `validate()` gates only on `fp32_dest_acc_en` (an axis); fidelity and approximation are never gated. The descriptor's three fields then become the compute `KernelSpec`'s `ttnn.experimental.ComputeHardwareConfig(fpu_math_fidelity=<math_fidelity>, sfpu_precision_mode=Approximate if math_approx_mode else Precise, enable_32_bit_dest=<fp32_dest_acc_en>)` (`ttnn/cpp/ttnn-nanobind/program_spec.cpp:524-552`), with `fpu_math_fidelity` always set explicitly, and `math_approx_mode` is also passed as the compile-time argument `exp_approx` that the `exp_tile_init<approx>` / `exp_tile<approx>` calls take, so the approximation reaches the exponential whatever `sfpu_precision_mode` maps to.

`validate()` order: canonicalise `dim` (raise `ValueError` for X1), derive axes (`INPUT_TAGGERS = AXES` imported from the spec; `memory_config` / `sharding` from `eval.memory.memory_axes(input_tensor)`, R9; `dtype`, `layout`, `fp32_dest_acc_en`, `dim`), gate on `SUPPORTED` (= PHASE0) with `NotImplementedError`, then the R3 refusal (float32 with `fp32_dest_acc_en=False`), then the resident-regime predicate (below) with `NotImplementedError`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 4 `[N, C, H, W]` in Phase 0 (ranks 2, 3 are a refinement: the kernels already see only `NC = prod(shape[:-2])`, `H`, `W`) |
| Dtype | bfloat16 (Phase 0) |
| Layout | TILE (each `[H, W]` image tile-padded independently) |
| Memory | interleaved, DRAM or L1; read through its `TensorAccessor` either way |

### Output

| Property | Value |
|----------|-------|
| Shape | input shape (logical and padded) |
| Dtype | input dtype |
| Layout | TILE |
| Memory | interleaved, in the input's buffer (R8): `ttnn.allocate_tensor_on_device(input_tensor.spec, device)` — the spec carries the input's memory config. Padding rows of the output hold 0 (masked), padding columns hold finite values |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Tile index space of one call: `(image nc ∈ [0, NC), tile-row ht ∈ [0, Ht), tile-column wt ∈ [0, Wt))` with `NC = prod(shape[:-2])`, `Ht = padded_shape[-2] / 32 = ceil(H / 32)`, `Wt = padded_shape[-1] / 32 = ceil(W / 32)`. Phase 0 reduces along `ht` (`dim = −2`).

**Unit of work: a tile-column** `u = nc · Wt + wt` (the `Ht` tiles `(nc, 0..Ht−1, wt)`), `num_cols = NC · Wt`. Images and tile-columns are both independent and are treated identically, so they fold into one column-unit index; a block may straddle an image boundary (every tile is its own interleaved page, so nothing requires contiguity).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| image `nc` (N·C folded) | independent — no result spans images | `block_cols` (extent over the folded column-unit axis `nc·Wt + wt`) | as for `wt` | host `select_blocking()` → compile-time arg `block_cols` | folded with `wt`: whole rounds of C column-units per cluster, one column-unit per NEO per round | knob-turn |
| tile-column `wt` | independent — softmax along H never mixes columns | `block_cols` | the whole per-NEO share when it fits (`= rounds_this_cluster`, max over clusters); else the largest that fits with `in_depth = 2`, else with `in_depth = 1`; clamped by the stat-ring cap (≤ 63 at C = 4). YOLO `[1,4,16,8400]`: 9 | host `select_blocking()` → CT arg `block_cols`; per-cluster `rounds_this_cluster` RT arg; last block `last_block_cols` derived in-kernel | split across clusters (whole rounds, `ttnn.split_work_to_cores(grid, total_rounds, row_wise=True)`) and across the 4 NEOs of a cluster (NEO t owns rounds' slot t) | knob-turn |
| tile-row `ht` | **dependent** — `m` and `s` span the whole column | `block_rows` | `Ht` (whole column resident; a block never cuts H in the built regime) | host: `Ht` from `padded_shape`, CT arg `Ht`; the resident predicate bounds it | not assigned across cores: every column is computed whole by one NEO (the H-split is the `deferred` regime `h_split_combine`) | scheme-change (`h_split_combine`, or `column_streaming` for a column too tall to stay resident) |
| per-column stats (`m`, `1/s`: one tile each per column) — data the scheme creates | independent along the column-unit axis, like their column | rides `block_cols` | `block_cols` tiles per NEO | derived from `block_cols` | the NEO that owns the column computes and consumes them — no designated collector, no single-core stage | — |
| output tiles | independent, same as input | rides `block_cols`, `block_rows` | streamed per tile | — | written by the writer thread that drains the owning NEO | — |

The `Ht` padding rows (`h_valid_last = H − (Ht − 1)·32 < 32`) are a partial *tile*, handled by an additive mask (below), not a partial block.

### Thread plan (the fourth knob)

| Field | Value |
|-------|-------|
| Bound | **compute.** Per input tile (2 KB in, 2 KB out, read once): two exponentials, two row-broadcast subtracts, one row-broadcast multiply, an SFPU max and an SFPU add (the max fold and the sum fold), plus per column two `sfpu_reduce` collapses and one reciprocal — several SFPU passes per tile against one read and one write. |
| Plan | `THREAD_PLAN = (R, C, W) = (2, 4, 2)` — one host constant. C = 4 because the op is compute-bound; R = 2, W = 2 because the resident regime reads every tile once and writes it once. R + W = 4 ≤ 6; 2 | 4. |
| Unit dealt to compute threads | one tile-column (`Ht` ring entries), larger than an entry, so every NEO takes the same number of entries: round `r` of a cluster gives NEO `t` column `col_start + r·C + t` when that exists, else NEO 0's column of the round (`col_start + r·C`), recomputed and stored again with identical bytes by the writer. The host deals **whole rounds** of C columns to clusters (`total_rounds = ceil(num_cols / C)` split over the grid), so at most C − 1 columns are recomputed per op, in the last round of the last cluster. |
| Engagement | `num_clusters = min(total_rounds, grid)`; a shape with fewer than C columns engages one cluster, never pads several. |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `in_depth` (host `select_blocking()`, CT arg) | 1 when the NEO's whole share is one block (every Phase 0 INPUTS shape); 2 when the share needs several blocks (`IN_DEPTH_MULTI = 2`), 1 if 2 does not fit | with several blocks, the read of block `b+1` lands while block `b` computes; with one block there is nothing to overlap (op `column_max_block` waits tile by tile, so compute starts on the first tile anyway) |
| `cb_output_tiles` | `OUT_DEPTH` (host constant) × `out_chunk_tiles` | 2 × `min(block_cols·Ht, OUT_CHUNK_MAX_TILES = 8)` | writer drains one chunk while compute packs the next; 8 = the largest DEST section (fp32 DEST, full sync) |
| `cb_col_max`, `cb_col_recip` | none (live set = the block's columns) | 1 block | — |
| `cb_h_mask` | none (constant) | 1 entry per NEO | — |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DFB ring size ≤ `DFB_RING_MAX_BYTES` = 1,048,560 B, capacity counted per NEO × C | `in_depth · block_cols · Ht · tile_in · C`; `block_cols · tile_f32 · C` (each stat ring); `OUT_DEPTH · out_chunk_tiles · tile_in · C` | `select_blocking()` takes the largest `block_cols` satisfying all three; `Ht · tile_in · C ≤ DFB_RING_MAX_BYTES` is the resident predicate (Ht ≤ 127 for bf16 at C = 4; note 128 tiles = 1,048,576 B overshoots by 16 B, as does every power-of-two ring of 1 MiB) | program build refuses the ring |
| Stat-ring tile size (Float32, 4,096 B) at C = 4 | `block_cols ≤ 63` | min with the share | as above |
| DEST capacity: `DEST_AUTO_LIMIT` = 4 fp32 tiles (half sync) | simultaneously-live DEST tiles per section: `column_max_block` 3 (acc, work, mask); `column_expsum_block` 4 (acc, work, max, mask); `normalize_block` 3 + ≥ 1 output (max, recip, mask on the last tile-row, work) | the realisation sizes each section to `DEST_AUTO_LIMIT` (`dest_helpers.hpp`), never a literal; not a block bound | slot overflow corrupts a live accumulator |
| SFPU row-broadcast and `add_binary_tile` / `binary_max_tile` need fp32 DEST (`sfpu_binary_bcast.h`, "Both source registers must contain FP32 data") | the `fp32_dest_acc_en` axis | Phase 0 supports `True` only | wrong results under 16-bit DEST |
| Indexed access into a ring must not straddle its end | block windows in `cb_input_tiles` | ring capacity = `in_depth` whole nominal blocks; only a NEO's **last** block may be short, and nothing follows it | — |
| `split_work_to_cores` gives clusters rounds differing by at most one | `block_cols` per program vs per cluster | ring sized for the program's `block_cols`; a cluster with fewer rounds runs a short (last) block | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (boundary: DRAM / L1 of the input's buffer) | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `column_resident` | **built** | `resident_fits := Ht · tile_in · C ≤ DFB_RING_MAX_BYTES` and `footprint(block_cols = 1, in_depth = 1) ≤ budget` (always true for INPUTS: max Ht = 64) | `block_cols` column-units × `block_rows = Ht` tile-rows | **minimum**: each input tile crosses DRAM once, each output tile once (+ at most C − 1 recomputed columns' writes per op, identical bytes); the mask is built in L1 (0 DRAM); stats never leave L1 | per block, once instead of per column: 3 `pack_init` re-aims (`cb_col_max` → `cb_col_recip` → `cb_output_tiles`), the unpack format reconfig bf16 ↔ Float32 at each op boundary, the SFPU inits of each op (max/reduce, exp/add/reduce/recip, bcast/exp/mul), and the input ring handshake of a block. Measured shape of this effect: catalog `compute_block_size` (1.65× for whole-block vs per-row passes) |
| `column_streaming` | **deferred** — no shape the Phase 0 rectangle admits from INPUTS (nor any `dim=-2` TARGET shape: max Ht = 64 ≤ 127) fails `resident_fits`; the first shape class that does is the `dim=-1` refinement's wide rows (`Wt = 256`: 512 KB per row per NEO, 2 MB at C = 4). Phase 0 refuses `¬resident_fits` with `NotImplementedError` | `¬resident_fits` | `block_cols = 1` × a window of `stream_rows` tile-rows | above minimum: the input crosses DRAM 3× (max pass, exp-sum pass, normalize pass) or 2× with an online max/sum fold in one pass; same 1× output | the same fixed costs per window. Reachable: the three block ops are unchanged; the reader re-streams the column per op instead of once, and ops `column_max_block` / `column_expsum_block` already walk the column front to back |
| `h_split_combine` | **deferred** — it serves only shapes whose `num_cols < clusters · C` with large `Ht`; in INPUTS only `[1,1,2048,256]` (8 columns × 64 tiles, 2 of 32 clusters engaged), which `column_resident` computes correctly. It is also the **cheapest-traffic** answer to a column too tall for one cluster (1× DRAM instead of `column_streaming`'s 2–3×), so it is built with that refinement, not before | `num_cols < grid · C` and `Ht ≥ 2·S` for S slices | per cluster: `block_cols` columns × `Ht/S` tile-rows | DRAM: minimum (1× in, 1× out). Added cross-core: per column per slice one `(m_i, s_i)` pair (2 Float32 tiles, 8 KB) gathered to the column's leader and one `(m, 1/s)` pair multicast back to S − 1 slices; two semaphore rendezvous per block | rendezvous per block instead of per column. Reachable: the per-slice stats already exist as tiles in `cb_col_max` / `cb_col_recip` between `column_expsum_block` and `normalize_block`; the combine (`m = max m_i`, `s = Σ s_i · exp(m_i − m)`) inserts there through `ttnn.Mcast` + `McastArgs` without touching the three ops |
| `three_pass_reread` | **rejected** — superseded by `column_resident` everywhere it applies (a dead end: streaming a column that fits in L1 three times from DRAM) | — | — | 3× input DRAM crossings vs 1× | — |
| `per_tile_dealing` | **rejected** — a NEO needs its whole column for `m` and `s`; dealing ring entries (tiles) round-robin splits a column across NEOs, which is the H-split with a combine inside the cluster | — | — | — | — |

Regime selection (host, exact): `column_resident` iff `resident_fits`; otherwise `NotImplementedError("column taller than one ring: column_streaming is a later refinement")`. Within `column_resident`, `select_blocking()` (Work Distribution) picks `block_cols` / `in_depth`; the single-block and multi-block paths are one regime with different knob values, and both are pinned by tests (Work Distribution, *Tests*).

### Traffic ranking

| Candidate split | Input DRAM crossings | Output DRAM crossings | Cross-core traffic | Occupancy on YOLO `[1,4,16,8400]` (1,052 columns) | Rank |
|-----------------|----------------------|------------------------|--------------------|------------------------|------|
| column-units (`nc`, `wt` folded) across clusters and NEOs, column resident | 1× | 1× | none | 32 clusters × 4 NEOs, 8–9 columns each | **1 — chosen** |
| images only (`nc`) | 1× | 1× | none | 4 units → 1 cluster | 2 (same bytes, strands the grid; subsumed by row 1) |
| H split across clusters + combine (`h_split_combine`) | 1× | 1× | 2 stat tiles per slice per column in, 2 out, + 2 rendezvous per block | same as row 1 for YOLO (Ht = 1: nothing to split) | 3 (more bytes than row 1; wins only on occupancy for tall, narrow shapes — `deferred`) |
| column split, column streamed (`column_streaming`) | 2–3× | 1× | none | as row 1 | 4 (`deferred`; only for columns that cannot be resident) |
| column split, column re-read 3× although resident (`three_pass_reread`) | 3× | 1× | none | as row 1 | `rejected` |

Operand-reuse check: the only operand is the input; it varies along every split axis. The constant mask tile does not vary, and is built per NEO in L1 from nothing (no DRAM, no multicast needed). Stall-shadow check: no stage waits on another core. Within a NEO, `column_max_block` waits on the reader tile by tile (cumulative wait), so the block's compute starts with its first tile; with several blocks, `in_depth = 2` overlaps the next block's read. `normalize_block` needs the column's final stats by definition (no algebraic licence to start it earlier); the writer drains `cb_output_tiles` while compute packs.

### Block schedule

```cpp
build_h_mask();                                   // once per kernel, per NEO; only when h_valid_last < 32
for (uint32_t block_idx = 0; block_idx < num_blocks_this_neo; ++block_idx) {
    load_block(block_idx);            // reader
    column_max_block(block_idx);      // compute
    column_expsum_block(block_idx);   // compute
    normalize_block(block_idx);       // compute
    store_block(block_idx);           // writer
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `build_h_mask` | 1 tile (bf16): rows `< h_valid_last` = 0.0, rows `≥ h_valid_last` = `NEG_BIG` (bf16 `0xFF7F`, the lowest finite bfloat16), all 32 columns | pushed once per NEO, held by compute for the whole kernel, popped at kernel end | once per kernel per NEO; explicit-sync ring; local L1 stores only (no NoC transfer), `flush_l2_cache_range` before `push_back` |
| `load_block` | `block_cols × Ht` tiles per NEO, column-major in the ring (column k's tile ht at entry `k·Ht + ht`) | — | one tagged read per tile, implicit sync, no barrier; reader thread p serves NEOs p, p+R in strict rotation, one entry each |
| `column_max_block` | `block_cols` columns × `Ht` tiles → `block_cols` stat tiles | input block stays (not popped) | per column: fold the `Ht` tiles elementwise with the SFPU max into one DEST tile (the last tile-row loaded as `x + mask` when masked), then one `sfpu_reduce<MAX, Float32, REDUCE_COL>` collapses the 32 rows into row 0; pack to `cb_col_max`. Waits on `cb_input_tiles` cumulatively (tile by tile). Inits: `pack_init(cb_col_max)` + copy/max/reduce inits once per block |
| `column_expsum_block` | `block_cols` columns × `Ht` tiles → `block_cols` stat tiles | input block and `cb_col_max` stay | per column: load `m` (row 0) once; per tile `x` (+ mask on the last tile-row), `x − bcast_row(m)`, `exp`, SFPU-add into the accumulator; then `sfpu_reduce<SUM, Float32, REDUCE_COL>` and `recip_tile` on row 0; pack to `cb_col_recip`. Inits: `pack_init(cb_col_recip)` once per block; SFPU inits once per block where the LLK keeps state between alternating ops, otherwise as the LLK requires (see Risks) |
| `normalize_block` | `block_cols` columns × `Ht` tiles → `block_cols · Ht` output tiles | — (pops the input block, `cb_col_max`, `cb_col_recip` at block end) | per DEST section: load `m` and `1/s` once, then as many output tiles as the section holds: `x` (+ mask on the last tile-row), `− bcast_row(m)`, `exp`, `· bcast_row(1/s)`, pack to `cb_output_tiles` (one reserve/push per tile). Inits: `pack_init(cb_output_tiles)` once per block |
| `store_block` | `block_cols × Ht` tiles per NEO | — | one tagged write per tile, implicit sync, no barrier; writer thread w serves NEOs w, w+W |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM / L1 interleaved input → `cb_input_tiles` | bf16 tiles (2,048 B) | reader, `TensorAccessor(tensor::input)`, `noc.async_read<NocOptions::TXN_ID>(acc, ring, {.page_id = (nc·Ht + ht)·Wt + wt}, {})`, implicit sync | page id per tile; `nc = u / Wt`, `wt = u % Wt` for column-unit `u` |
| reader → `cb_h_mask` | bf16, 1 tile per NEO | explicit sync: `reserve_back(1)`, `dataflow_kernel_lib::fill_l1_range<2>` over the padded face rows, `flush_l2_cache_range`, `push_back(1)`; ring listed in the reader's `disable_dfb_implicit_sync_for` | no NoC transfer, so no transaction id is spent on it; the implicit-sync reads of the same kernel carry their ring tag (`reader_mixed.cpp`) |
| `cb_input_tiles` → DEST | bf16 → fp32 DEST | default unpack (`copy_tile`; never `UnpackToDest` for bf16 on ttsim) | indexed access `k·Ht + ht` within the block window (unpack_A honours the ring stride) |
| DEST ↔ `cb_col_max` / `cb_col_recip` | Float32 tiles (4,096 B) | `pack_tile` after `pack_init(<ring>)`; reload with `copy_tile`, default unpack | reload passes SrcA (tf32): `m` is reloaded identically by both readers of it, so its truncation cancels in `exp(x − m) / Σ exp(x − m)`; `1/s` loses ≤ 2⁻¹¹ relative |
| DEST → `cb_output_tiles` | fp32 → bf16 tiles | `pack_tile` after `pack_init(cb_output_tiles)` | — |
| `cb_output_tiles` → DRAM / L1 output | bf16 tiles | writer, `noc.async_write<NocOptions::TXN_ID>(ring, acc, {}, {.page_id = ...})`, implicit sync | the writer's last statement: `noc_async_write_set_trid(0, noc.get_noc_id())`; the reader's: `noc_async_read_set_trid(0, noc.get_noc_id())` |

TARGET axes as later work (each `TARGET − PHASE0` value):

| TARGET value | Class | What it needs on this scheme |
|--------------|-------|------------------------------|
| `memory_config = L1` | built | the same kernels; output allocated in L1 (R8) |
| `sharding`: only `interleaved` in TARGET | — | no physical-shard regime is targeted. Were one added, the column split above is the logical shard (a width-or-block shard of whole columns is a knob-turn; a height shard cuts the dependent axis and is `h_split_combine`), consumed in place through a ring backed on the shard, never re-read through a `TensorAccessor` |
| `rank` 2, 3 | knob-turn | `NC = prod(shape[:-2])` already; widen `SUPPORTED` |
| `dim = −1` | knob-turn of the reduce direction (compile-time flag; same loop nest, transposed roles) + `column_streaming` for wide rows | unit = tile-row of `Wt` tiles; `sfpu_reduce<…, REDUCE_ROW>` (supported for SUM and MAX, `compute_kernel_api.h:796-801`), `sfpu_{sub,mul}_bcast_col`, a column-padding mask on the last tile-column. Rows of `Wt ≤ 127` stay resident; wider rows need `column_streaming` or a W-split combine |
| `dtype = float32` | knob-turn of page formats | input/output rings Float32 (4 KB tiles: resident cap Ht ≤ 63 at C = 4); reload precision of the input through SrcA is tf32 — check against (0.999, 0.01) before claiming it; R3 refusal of `fp32_dest_acc_en=False` |
| `fp32_dest_acc_en = False` | scheme detail | the SFPU row-broadcast and SFPU binaries require fp32 DEST: either keep the kernel on fp32 DEST (precision ≥ requested) or move the broadcasts to the FPU |
| `layout = ROW_MAJOR` | new block ops (`tilize_block` in, `untilize_block` out) on the same split | `tilize_helpers.hpp` / `untilize_helpers.hpp`; the stick→tile reader changes |
| `dtype = bfloat8_b` | — | no path on Quasar (`unbuildable on quasar`); refused |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a tile-column (`Ht` tiles); dealt one per NEO per round; a block = `block_cols` consecutive rounds of one NEO |
| Grid | `device.compute_with_storage_grid_size()` (8 × 4 on Quasar); `split_work_to_cores(grid, total_rounds, row_wise=True)` |
| Per-core work | `rounds_this_cluster` rounds = `num_cols_this_cluster = min(rounds·C, num_cols − col_start)` real columns; every NEO of the cluster runs `rounds_this_cluster` columns; `num_blocks_this_neo = ceil(rounds_this_cluster / block_cols)`, `last_block_cols = rounds_this_cluster − (num_blocks − 1)·block_cols` |
| Remainder | `Ht = ceil(H/32)`, `Wt = ceil(W/32)` per image (from `padded_shape`, never `floor(NC·H/32)`); `total_rounds = ceil(num_cols / C)`; padding columns only in the last round of the last cluster (recomputed real columns, re-stored with identical bytes); `h_valid_last = H − (Ht−1)·32`, mask iff `< 32`; padded W columns of the last tile-column need no mask (columns are independent; they compute a finite softmax of the padding and are never read) |
| Bound | compute (see Thread plan) |
| Thread plan | `THREAD_PLAN = (2, 4, 2)` host constant; counts passed as compile-time args `reader_threads`, `compute_threads`, `writer_threads`; kernels size loops from them and `get_my_thread_id()` |
| Unit dealt to compute threads | the tile-column (larger than an entry): equal entries per NEO, short last round padded by recomputing NEO 0's column of that round; whole rounds dealt to clusters |
| Tests the plan needs | (1) every NEO ran: call the op twice in one device session on the same shape with two different inputs, on a shape whose columns cover all 4 NEOs of a cluster (e.g. `[1,4,16,400]`: 13 clusters × 4 NEOs × 1 column), and check every column of the second result (a NEO that did not run leaves the first call's values in its columns); plus assert the program spec's compute `KernelSpec.num_threads == 4`, reader 2, writer 2. (2) one device session: the op after another program (e.g. a `ttnn.add`), and the op followed by `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` and a host read-back — both must complete |

`select_blocking()` (host, exact; worked values for Quasar):

```
C = THREAD_PLAN[1]; tile_in = input.buffer_page_size(); tile_f32 = 4096
budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES
max_rounds = largest rounds per cluster from split_work_to_cores
footprint(bc, d) = C * (d*bc*Ht*tile_in + 2*bc*tile_f32 + OUT_DEPTH*min(bc*Ht, OUT_CHUNK_MAX_TILES)*tile_in + has_mask*tile_in)
fits(bc, d) = d*bc*Ht*tile_in*C <= DFB_RING_MAX_BYTES and bc*tile_f32*C <= DFB_RING_MAX_BYTES
              and footprint(bc, d) <= budget
cap = min(max_rounds, BLOCK_COLS_CAP)            # BLOCK_COLS_CAP: module-level test override, default unbounded
if max_rounds <= BLOCK_COLS_CAP and fits(max_rounds, 1):
    block_cols, in_depth = max_rounds, 1                                   # whole share resident: one block
elif any bc in [1, cap] with fits(bc, IN_DEPTH_MULTI):
    block_cols, in_depth = largest such bc, IN_DEPTH_MULTI                 # several blocks, double-buffered
elif fits(1, 1):
    block_cols, in_depth = largest bc in [1, cap] with fits(bc, 1), 1      # several blocks, single-buffered
else:
    raise NotImplementedError                                              # not resident_fits: column_streaming deferred
```

| Shape (Quasar, 32 clusters) | Ht / Wt / NC | num_cols | total_rounds → clusters × rounds | block_cols / in_depth / blocks | mask | footprint per cluster |
|---|---|---|---|---|---|---|
| `[1,4,16,8400]` (YOLO, DRAM) | 1 / 263 / 4 | 1,052 | 263 → 7 × 9 + 25 × 8 | 9 / 1 / 1 | yes (16 rows) | 507,904 B |
| `[1,4,16,400]` (YOLO, L1) | 1 / 13 / 4 | 52 | 13 → 13 × 1 | 1 / 1 / 1 | yes | 65,536 B |
| `[1,1,2048,256]` | 64 / 8 / 1 | 8 | 2 → 2 × 1 | 1 / 1 / 1 | no | 688,128 B |
| `[4,8,32,256]` | 1 / 8 / 32 | 256 | 64 → 32 × 2 | 2 / 1 / 1 | no | 114,688 B |

Required regime-pinned tests (beyond the acceptance test): (a) multi-block with a ragged last block: monkeypatch `BLOCK_COLS_CAP = 2` on `[1,4,16,8400]` (9 rounds → blocks 2,2,2,2,1, `in_depth = 2`); (b) the tall resident column `[1,1,2048,256]` (in INPUTS); (c) the refusal: a shape with `Ht = 128` (e.g. `[1,1,4096,32]`) raises `NotImplementedError` before any dispatch; (d) the H-mask under approximation: the loose case `yolo_dfl_l1_lofi` (LoFi, `math_approx_mode=True`, H = 16) and `large_magnitude_bfloat16` (scale 10, the always-on max subtraction, R1), with non-monotonic data along H so the max fold is exercised.

## Circular Buffers

Ring capacities are per NEO × C. All rings STRIDED (the default access pattern).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_input_tiles` | 0 | `tile_in` (2,048 B) | `in_depth · block_cols · Ht · C` | spans `block_cols`, spans `block_rows = Ht` (the column is resident across three ops); `in_depth` whole blocks | bf16 (input dtype) | reader | compute | reader: implicit (one tagged read per entry); compute: `wait_front` cumulative, `pop_front(block_cols·Ht)` at block end | load → end of `normalize_block` of the block |
| `cb_h_mask` | 1 | `tile_in` | `1 · C` (allocated only when `h_valid_last < 32`) | constant, no block axis | bf16 | reader | compute | reader: explicit (listed in `disable_dfb_implicit_sync_for`; reserve / local fill / L2 flush / push; no NoC transfer); compute: `wait_front(1)` once, `pop_front(1)` at kernel end | whole kernel |
| `cb_col_max` | 2 | 4,096 B | `block_cols · C` | spans `block_cols`, streams `Ht` (one tile per column) | Float32 (fp32 DEST) | compute | compute | compute-owned: push per column in `column_max_block`; read by index `k` in `column_expsum_block` and `normalize_block`; `pop_front(block_cols_this)` at block end | `column_max_block` → end of `normalize_block` |
| `cb_col_recip` | 3 | 4,096 B | `block_cols · C` | spans `block_cols`, streams `Ht` | Float32 | compute | compute | compute-owned: push per column in `column_expsum_block`; read by index in `normalize_block`; pop at block end | `column_expsum_block` → end of `normalize_block` |
| `cb_output_tiles` | 4 | `tile_in` | `OUT_DEPTH · out_chunk_tiles · C`, `out_chunk_tiles = min(block_cols·Ht, OUT_CHUNK_MAX_TILES)` | streaming window: streams `block_cols` and `Ht` | bf16 (output dtype) | compute | writer | compute: reserve/push per tile; writer: implicit (one tagged write per entry) | `normalize_block` → `store_block` |

Push = wait for every ring: `cb_input_tiles` — reader pushes `block_cols_this·Ht` per block per NEO (implicitly, one per read), compute waits cumulatively up to the same and pops the same; `cb_h_mask` — 1 / 1; `cb_col_max`, `cb_col_recip` — `block_cols_this` pushed, waited and popped per block; `cb_output_tiles` — `block_cols_this·Ht` per block, one per tile on both sides. Every ring has exactly one producer kernel and one consumer kernel.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `build_h_mask` | 1 tile | helper (`fill_l1_range`) | — | `cb_h_mask`, 1 | held by compute until kernel end |
| 1 | `load_block` | `block_cols × Ht` | raw (NoC reads, as `toy_row_passes/kernels/reader.cpp`) | tensor `input` | `cb_input_tiles`, `block_cols·Ht` | — |
| 2 | `column_max_block` | `block_cols × Ht` → `block_cols` | raw SFPU (rejections in API Mapping) | `cb_input_tiles`, `block_cols·Ht`, waited cumulatively, not popped; `cb_h_mask`, 1, held | `cb_col_max`, `block_cols` | input block resident |
| 3 | `column_expsum_block` | `block_cols × Ht` → `block_cols` | raw SFPU | `cb_input_tiles` (resident), `cb_col_max` (resident), `cb_h_mask` | `cb_col_recip`, `block_cols` | input, max resident |
| 4 | `normalize_block` | `block_cols × Ht` → `block_cols·Ht` | raw SFPU | `cb_input_tiles`, `cb_col_max`, `cb_col_recip` (all popped at block end), `cb_h_mask` | `cb_output_tiles`, per tile | input / stats rings empty for the next block |
| 5 | `store_block` | `block_cols × Ht` | raw (NoC writes, as `toy_row_passes/kernels/writer.cpp`) | `cb_output_tiles` | tensor `output` | — |

The compute kernel calls `compute_kernel_hw_startup(cb_input_tiles, cb_col_max, cb_output_tiles)` once, as its first statement, and never again; every later output-ring change goes through `pack_init(<ring>)`, every input-format change through the unpack reconfig.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `build_h_mask` | helper | `dataflow_kernel_lib::fill_l1_range<2>` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:101` | zero the tile, then `NEG_BIG` (`0xFF7F`) over each face's padded rows (face rows `r ≥ h_valid_last` in faces 0/1 for `r < 16`, faces 2/3 for `r ≥ 16`; a face row is 16 bf16 = 32 B contiguous) | — | `cb_h_mask` | none |
| `build_h_mask` | raw_api | `flush_l2_cache_range` before `push_back` | pattern at `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl:277-282` | `(write_addr, tile_bytes)` under `ARCH_QUASAR && COMPILE_FOR_DM` | — | `cb_h_mask` | none |
| `load_block` / `store_block` | raw_api | `noc.async_read<NocOptions::TXN_ID>` / `noc.async_write<NocOptions::TXN_ID>`, `TensorAccessor`, `noc_async_{read,write}_set_trid(0, …)` at kernel end | `.claude/references/quasar_op_examples/toy_row_passes/kernels/reader.cpp`, `writer.cpp` | page id `(nc·Ht + ht)·Wt + wt` | tensor | ring | `block_cols`, `Ht` (loop bounds) |
| all compute ops | raw_api | `copy_init`, `copy_tile` | `tt_metal/hw/inc/api/compute/tile_move_copy.h:47`, `:126` | default unpack; index `k·Ht + ht` | `cb_input_tiles`, `cb_h_mask`, stat rings | DEST | — |
| `column_max_block` | raw_api | `binary_max_tile_init`, `binary_max_tile(idst0, idst1, odst)` | `tt_metal/hw/inc/api/compute/binary_max_min.h:145`, `:126` | elementwise fold of the column's tiles | DEST | DEST | `Ht` (fold length) |
| `column_max_block`, `column_expsum_block` | raw_api | `sfpu_reduce_init<PoolType, DataFormat::Float32>`, `sfpu_reduce<PoolType::MAX / SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(idst)` | `tt_metal/hw/inc/api/compute/compute_kernel_api.h:846`, `:796` | one tile, result in row 0 | DEST | DEST | none |
| masked load, `column_expsum_block` fold | raw_api | `add_binary_tile_init`, `add_binary_tile(idst0, idst1, odst)` | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h:272`, `:103` | `x + mask`; `acc += e` | DEST | DEST | — |
| `column_expsum_block`, `normalize_block` | raw_api | `sfpu_bcast_row_init`, `sfpu_sub_bcast_row`, `sfpu_mul_bcast_row` | `tt_metal/hw/inc/api/compute/sfpu_binary_bcast.h:302`, `:338`, `:364` | `x − m[0,:]`, `e · (1/s)[0,:]` | DEST | DEST | — |
| `column_expsum_block`, `normalize_block` | raw_api | `exp_tile_init<exp_approx>`, `exp_tile<exp_approx>` | `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h:40`, `:85` | `exp_approx` = CT arg from `math_approx_mode` | DEST | DEST | — |
| `column_expsum_block` | raw_api | `recip_tile_init`, `recip_tile` | `tt_metal/hw/inc/api/compute/eltwise_unary/recip.h:32`, `:68` | on the summed tile | DEST | DEST | — |
| all compute ops | raw_api | `pack_init(ring)`, `pack_tile(idst, ring)` | `tt_metal/hw/inc/api/compute/pack.h:32`, `:87` | `pack_init` at every output-ring change (3 per block) | DEST | stat rings, `cb_output_tiles` | — |

**Helpers considered and rejected** (one entry covers every `raw_api` compute row above):

| Candidate | File:line of the mismatch | Why it cannot express the planned block operation |
|-----------|---------------------------|---------------------------------------------------|
| `compute_kernel_lib::reduce<MAX/SUM, REDUCE_COL, …>` for `column_max_block` / `column_expsum_block` | `reduce_helpers_compute.hpp:103` (`WaitAndPopPerTile` pops) | Its indexed input policies (`WaitUpfrontNoPop`, `BulkWaitBulkPop`, `NoWaitNoPop`) are not used under C = 4; the remaining policy, `WaitAndPopPerTile`, pops the resident column, which `column_expsum_block` and `normalize_block` re-read, and so forces `three_pass_reread` (rejected regime). |
| the same helper's exact float path | `reduce_helpers_common.hpp:50` (`static_assert(!requires_sfpu, "SFPU reduce path is not supported on Quasar")`), `reduce_helpers_compute.inl:79,92` | The only path left is the FPU `reduce_tile` matmul-with-scaler: it accumulates through tf32 (~1.8e-4 per tile) and multiplies a bfloat16 source on the matrix unit. It also needs an extra scaler ring. |
| `eltwise_chain` / convenience `sub`/`mul` with `BroadcastDim::Row`, `Exp<>`, `DestAccumulation` for `column_expsum_block` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:303-312` (DEST accumulation is an FPU-binary scope only); `:354-357` (broadcast only on an FPU srcB) | "exp, then accumulate" cannot stay in DEST through the chain: accumulation exists only for FPU binaries, so every `exp` tile would be packed to an extra Float32 ring and reduced afterwards — a buffer the inventory rule forbids when DEST can hold the fold. |
| `eltwise_chain` for `normalize_block` (`BinaryFpu<Sub, x, m bcast Row>` → `Exp<>` → multiply by `1/s`) | `chain.hpp:579-583` (`DestReuseBinary` takes an `InputSpec`, no `BroadcastDim`) | The second binary in the chain must read DEST, and `DestReuseBinary` cannot broadcast row 0 of `1/s`; splitting into two chains needs an intermediate ring and an FPU multiply at HiFi4. |
| `BinaryMax` chain element for the max fold | — | Not used; the float max is the raw `binary_max_tile`. |
| `eltwise_chain` as the shell of a fused section | `chain.hpp:15-17` (the chain owns the dst-sync window) | A chain cannot share one DEST section with the raw `sfpu_reduce` / row-broadcast calls the same section needs. |
| `dataflow_kernel_lib::prepare_reduce_mask` for `build_h_mask` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl:264-275` | It writes a multiplicative 1/0 mask in column 0 for `mul_tiles_bcast_cols`, through an untagged `noc.async_write_zeros` (a NoC transfer without a transaction id in a kernel that has an explicit ring); the max fold needs an additive `NEG_BIG` over all 32 columns of the padded rows. `fill_l1_range` (a helper) does it with no NoC transfer. |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| masked load (last tile-row) | SFPU add | `cb_input_tiles`: All | `cb_h_mask`: All | None |
| `column_max_block` fold | SFPU max | DEST acc: All | `cb_input_tiles` tile: All | None |
| `column_max_block` collapse | `sfpu_reduce` MAX, REDUCE_COL | DEST acc: All → Row0 | — | — |
| `column_expsum_block` subtract | SFPU sub | `cb_input_tiles`: All | `cb_col_max`: Row0 | Row |
| `column_expsum_block` fold | SFPU add | DEST acc: All | DEST `exp` tile: All | None |
| `column_expsum_block` collapse | `sfpu_reduce` SUM, REDUCE_COL, then `recip` | DEST acc: All → Row0 | — | — |
| `normalize_block` subtract | SFPU sub | `cb_input_tiles`: All | `cb_col_max`: Row0 | Row |
| `normalize_block` scale | SFPU mul | DEST `exp` tile: All | `cb_col_recip`: Row0 | Row |

## Reduce Direction Verification

| Logical Dim | Tile ReduceDim | Output Valid Region | BroadcastDim | ReduceInputBlockShape | EltwiseShape |
|-------------|----------------|--------------------|--------------|-----------------------|--------------|
| −2 (Phase 0) | `REDUCE_COL` (`sfpu_reduce`, one tile, after the elementwise fold of `Ht` tiles) | Row0 | Row (`sfpu_*_bcast_row`) | n/a (raw; the fold walks `Ht` tiles of one column) | n/a |
| −1 (refinement) | `REDUCE_ROW` (`sfpu_reduce`, after a fold of `Wt` tiles) | Col0 | Col (`sfpu_*_bcast_col`) | n/a | n/a |

### Structural impossibilities

None beyond the specification's `INVALID`.

## Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (single block per NEO) | Every Phase 0 shape runs one block per NEO at `in_depth = 1`: the block's reads and its compute overlap only through `column_max_block`'s cumulative wait, and `column_expsum_block` waits for the whole column | `block_cols = ceil(rounds/2)` with `in_depth = 2` (two blocks per NEO) on `[1,4,16,8400]`; structural count: blocks per NEO |
| SFPU multiply in `normalize_block` | catalog `compute_fusion`: a plain multiply on the SFPU measured 0.58× of the FPU's | the FPU `mul` with `BroadcastDim::Row` from a packed `exp` ring, **only** after confirming ttsim accepts it at HiFi4 on Float32 sources |
| `exp` recomputed in `normalize_block` | two exponentials per element; the second could be read back | store `exp(x − m)` as Float32 in a compute-owned ring of `block_cols·Ht` per NEO (halves the cap on Ht to 63 at C = 4; tf32 reload) |
| DEST sync mode | half sync gives 4 fp32 slots, so `normalize_block` reloads `m` and `1/s` every 1–2 output tiles | `double_buffer_dest=False` (full sync, 8 slots: up to 5 outputs per reload) |
| Ht = 1 columns (the YOLO shapes) | the stats make two L1 round trips per column for one output tile | a single-section path for `Ht ≤ DEST_AUTO_LIMIT − 2`: stats stay in DEST, `cb_col_max` / `cb_col_recip` unused |
| Thread plan | chosen from the bound, not measured (ttsim here has no timing) | on `ttsim-perf`: `(1,1,1)`, `(2,4,2)` (built), `(4,4,2)` — one constant |
| Grid synchronization | — | not applicable: no collective, barrier or multicast in the built regime |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| H padding entering `m` or `s` | both YOLO shapes have 16 logical rows in a 32-row tile, the reduced dimension | additive `NEG_BIG` mask on every column's last tile-row in all three compute ops; `exp(NEG_BIG − m)` must be 0 in both approximation modes — test (d) |
| `NEG_BIG + x` overflow | `0xFF7F + x` for very negative `x` can round to `−inf` in fp32 | harmless for the max; `exp` of `−inf` must yield 0 (the default `InputClamping::ClampToNegative` of `exp_tile`, `exp.h:85`): verified by test (d) on scale-10 data |
| SFPU state across alternating ops | `column_expsum_block` alternates bcast-sub, `exp`, `add_binary` per tile; `exp_tile_init` loads programmable registers that another init may overwrite | re-issue an op's init wherever the LLK does not keep it across the alternation; measure, then hoist; correctness first |
| Pack target | pack target is set at `pack_init` on this part | `pack_init(<ring>)` before the first pack into each of `cb_col_max`, `cb_col_recip`, `cb_output_tiles` in every block |
| Indexed reads of a ring under C = 4 | `copy_tile` (unpack_A) honours the stride | no `reduce_tile` anywhere; stats collapsed with `sfpu_reduce` in DEST |
| Max fold on monotonic test data | a fold bug that returns `op(first, last)` passes monotonic data | test with non-monotonic H (normal data does it; add one hand-built column) |
| tf32 reload of stats | Float32 stat pages unpack through SrcA | `m` truncates identically for both of its readers (cancels); `1/s` ≤ 2⁻¹¹ relative; do not switch the stat rings to `UnpackToDest` |
| Tag register | implicit-sync kernels leave a ring's transaction id in the hart's tag register | reader and writer end with `noc_async_{read,write}_set_trid(0, …)`; the session tests in Work Distribution |
| Mask ring protocol | the reader writes the mask itself | list `cb_h_mask` in the reader's `disable_dfb_implicit_sync_for`; reserve/push explicitly; no NoC transfer touches it; `flush_l2_cache_range` before `push_back` |
| Padding-round duplicate writes | NEO t recomputes NEO 0's column; two writer threads may write the same page | identical bytes; no ordering needed (as `toy_row_passes`) |
| Fidelity | set on every compute kernel | no FPU multiply in the design; `fpu_math_fidelity` still set explicitly from the resolved config |
| Ring cap at 1 MiB − 16 B | 128 bf16 tiles × 4 NEOs = exactly 1 MiB, 16 B over | `select_blocking()` checks every ring against `DFB_RING_MAX_BYTES`; resident cap Ht ≤ 127 |
| `fp32_dest_acc_en` | SFPU row-broadcast requires fp32 DEST | Phase 0 `SUPPORTED` has `True` only; the refinement note under Dataflow Strategy |

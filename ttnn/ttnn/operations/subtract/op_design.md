# Operation Design: subtract

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (binary eltwise, FPU) |
| Goal | `out = a - b` elementwise with torch broadcasting, as one `ttnn.generic_op` dispatch of one Metal 2.0 `ttnn.ProgramSpec` on Quasar. Phase 0: two bfloat16 TILE operands of one shape, both in DRAM or both in L1, interleaved, rank 2 to 4, any alignment, bfloat16 output. |
| Math | `out[n, h, w] = a[n, h, w] - b[n, h, w]` (Phase 0, `broadcast = none`); in general `out[i] = a[bcast_a(i)] - b[bcast_b(i)]` over the broadcast shape |
| References | `eval/specs/subtract/spec.py` (the specification; governs); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/toy_thread_plan/` (the same program shape with `mul`: host split, rings, implicit-sync reader/writer, `compute.cpp`); `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp`; `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` (Part A); perf catalog `ttnn/ttnn/operations/examples/master.md` entries `width_split`, `double_buffer`, `compute_block_size` |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (probe on this backend: `compute_with_storage_grid_size()` = 8-4). Predicates query `device.compute_with_storage_grid_size()` | target page + probe 2026-11-21 |
| L1 per core | 4,194,304 B (4 MiB) per cluster, shared by its 4 NEOs and 8 DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` = 3,963,328 B − 0 on this target (probe 2026-11-21). No predicate of Phase 0 needs it: the footprint is closed-form and constant in the op's shape (`l1_ledger.md`) | probe |
| Host program | `ttnn.ProgramSpec` (Metal 2.0: `KernelSpec`, `DataflowBufferSpec`, `TensorParameter`, `WorkUnitSpec`, `ProgramRunArgs`, one `ttnn.generic_op`) — `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/toy_thread_plan_program_artifacts.py`) | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1, 2, 4}), 6 user DM cores (R + W ≤ 6), on STRIDED rings R | C and W | C; ring capacity counted per NEO; 32 DFBs per cluster, ≤ 24 active per NEO, ≤ 1 MiB − 16 B per ring | target page *Cluster model* / *Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` and the C++ device operation with a program factory (Metal 1 kernels refused at creation, `kernel.hpp` TT_FATAL); `bfloat8_b` / `bfloat4_b` (no path on the part); `UnpackToDest` of a bfloat16 page into 32-bit DEST (so bf16 rings keep the default `UnpackToSrc`) | target page *Not available* |

All worked numbers below are for this target (32 clusters, bfloat16 tile = 2,048 B).

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor_a` | `ttnn.Tensor` | yes | on device; Phase 0: bfloat16, TILE, DRAM or L1 interleaved, rank 2–4 | — | tensor binding `minuend` |
| `input_tensor_b` | `ttnn.Tensor` | yes | same dtype, layout and placement as `a`; Phase 0: same shape as `a` | — | tensor binding `subtrahend` |
| `dtype` (keyword-only; axis `output_dtype`) | `ttnn.DataType` or `None` | no | Phase 0: `None`, `ttnn.bfloat16` | `None` → operands' dtype | sets the output tensor's dtype and the output ring's format |
| `compute_kernel_config` (keyword-only) | `ttnn.ComputeConfigDescriptor` or `None` | no | Phase 0: `fp32_dest_acc_en=True`; any `math_fidelity`, any `math_approx_mode` (rule R4) | `None` → `default_compute_kernel_config()` = `ComputeConfigDescriptor(math_fidelity=HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False)` | translated to `ComputeHardwareConfig` (table below) |

### Compute-config translation (host, one function)

`ComputeConfigDescriptor` is a Metal 1 type; a Quasar compute `KernelSpec` takes `ComputeHardwareConfig`. The entry point resolves `None` through `default_compute_kernel_config()` (rule R3: the only place the default lives) and maps field by field:

| `ComputeConfigDescriptor` field | `ComputeHardwareConfig` field | Note |
|---|---|---|
| `math_fidelity` | `fpu_math_fidelity` | always set explicitly; never gated (R4) |
| `fp32_dest_acc_en` | `enable_32_bit_dest` | the `fp32_dest_acc_en` axis |
| `math_approx_mode` | `sfpu_precision_mode` = `Precision.Approximate` if true else `Precision.Precise` | no SFPU in Phase 0; carried for later paths; never gated (R4) |
| `dst_full_sync_en` | `double_buffer_dest = not dst_full_sync_en` | also feeds the `sync_tiles` clamp (Mechanism caps) |

### Validation (Python, before any device work)

| Order | Check | Raises |
|---|---|---|
| 1 | Shapes broadcast (`torch.broadcast_shapes` rule; X1: differ where neither is 1) | `ValueError` |
| 2 | Resolve config through `default_compute_kernel_config()`; derive axes: `dtype`, `layout`, `fp32_dest_acc_en`, `output_dtype` (= `dtype` argument), `memory_config` / `sharding` via `eval.memory.memory_axes` of **each** operand (R10; operands in different buffers are refused by name), `alignment` / `rank` / `broadcast` via `INPUT_TAGGERS = AXES` imported from the specification | — |
| 3 | Every axis in `SUPPORTED` (= `PHASE0`), no `EXCLUSIONS` match. EXCLUSIONS carry R1 `{dtype: float32, fp32_dest_acc_en: False}` and R2 `{output_dtype: float32, fp32_dest_acc_en: False}` so they stay refused when those values enter SUPPORTED | `NotImplementedError` (R8) |
| — | `math_fidelity`, `math_approx_mode` are never read by `validate()` (R4) | — |

No index axes; nothing to canonicalize.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `a`: rank 2–4; `b`: Phase 0 identical to `a` (broadcast regimes deferred, Regimes) |
| Dtype | Phase 0: bfloat16 both |
| Layout | Phase 0: TILE (32×32; non-aligned H / W carry tile padding, computed on like data, GUIDANCE) |
| Memory | DRAM or L1, interleaved; both operands in the same buffer |

### Output

| Property | Value |
|----------|-------|
| Shape | `torch.broadcast_shapes(a.shape, b.shape)` (Phase 0: `a.shape`) |
| Dtype | `dtype` if given, else the operands' dtype (Phase 0: bfloat16) — the FP32 DEST result rounded once by the packer (R6) |
| Layout | TILE (same as `a`) |
| Memory | `a`'s buffer (DRAM → DRAM, L1 → L1), interleaved (R9): `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(out_shape, out_dtype, TILE, a.memory_config()), device)` |

## Blocking Model

### Axes

The output's index space in TILE layout is `(N, Ht, Wt)` where `N = prod(shape[:-2])` (1 at rank 2), `Ht = ceil(H/32)` and `Wt = ceil(W/32)` per image. With `broadcast = none` the op treats all three identically and the tile pages of `a`, `b` and `out` are numbered identically and contiguously (`page = (n·Ht + ht)·Wt + wt`), so the space flattens honestly to one axis, the **flat tile index** `tile_id ∈ [0, tensor_tiles)`, `tensor_tiles = N · ceil(H/32) · ceil(W/32)` (`== out.buffer_num_pages()`, which the host uses as the source and asserts against the formula).

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `N` (leading dims, folded) | independent — each output element reads only its own `a`, `b` element | folded into `tile_id` (no own knob: contiguous and treated identically by the op) | whole extent within the flat range | — (derived) | via `tile_id` split | broadcast over leading dims re-introduces it as an index axis for `b` (scheme-change, deferred row) |
| `Ht` (tile-rows per image) | independent — same reason | folded into `tile_id` | whole extent within the flat range | — (derived) | via `tile_id` split | `over_h` / `over_hw` / `first_operand` broadcast need it as an index axis (deferred rows) |
| `Wt` (tile-columns) | independent — same reason | folded into `tile_id` | whole extent within the flat range | — (derived) | via `tile_id` split | `over_w` / `over_hw` broadcast need it (deferred rows) |
| `tile_id` (flat) — per-core block | independent | `core_tiles` (this cluster's contiguous range = **the block**: one helper call, one init, one pipeline pass) | the whole per-core assignment (`ceil` / `floor` of `tensor_tiles / num_cores`) | host `split_work_to_cores` → runtime arg `num_tiles` per core | contiguous ranges over the full grid, `row_wise=True` | grid size is a runtime query; `min_tiles_per_core` knob (Perf lamps) |
| `tile_id` within a NEO — handshake quantum | independent | `sync_tiles` (tiles per DEST section and per ring wait/pop/reserve/push inside the one call) | 4 (= DEST capacity at 32-bit DEST, half sync; clamped on host, Mechanism caps) | host constant `SYNC_TILES = 4` → clamped value → compute CT arg `sync_tiles`; ring depth derives from it | — (within a NEO) | knob-turn |

Neither extent is 1: the block is the whole per-core range, and `sync_tiles` is the DEST-section width. No axis is dependent and none is reuse-shared in Phase 0 (operand-reuse check: with `broadcast = none` both `a` and `b` vary along the split axis `tile_id`, so neither is re-read by any core; `b` becomes reuse-shared exactly in the broadcast regimes, where it does not vary along the broadcast axes).

| Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|---|---|---|---|---|---|
| `N` | independent | whole, folded into `tile_id` | spread via `tile_id` | via ring depth | identical treatment, contiguous pages |
| `Ht` | independent | whole, folded into `tile_id` | spread via `tile_id` | via ring depth | same |
| `Wt` | independent | whole, folded into `tile_id` | spread via `tile_id` | via ring depth | same; splitting by tile-rows instead would strand the YOLO shapes (`Ht = 1`) on one cluster (catalog `width_split`) |
| `tile_id` per cluster | independent | `core_tiles` = whole assignment | `split_work_to_cores(grid, tensor_tiles, row_wise=True)`, all 32 clusters when `tensor_tiles ≥ 32` | `ENTRIES_PER_NEO = 2 · sync_tiles` per ring | fill the grid first, then the coarsest block = the whole range |
| `tile_id` per NEO | independent | `neo_tiles = core_tiles / C (+1 for t < core_tiles % C)`, one call; quantum `sync_tiles` | STRIDED rings deal tile `i` to NEO `i mod C` | same | the unit dealt is the ring entry, so no padding |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_minuend_tiles` | `ENTRIES_PER_NEO` (= `2 · sync_tiles`), × C | 8 entries per NEO × 1 = 8 | the reader fills quantum k+1 while compute consumes quantum k (double buffering) |
| `cb_subtrahend_tiles` | same knob | 8 | same |
| `cb_difference_tiles` | same knob | 8 | compute packs quantum k+1 while the writer drains quantum k |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DEST capacity per section (`DEST_AUTO_LIMIT`, `dest_helpers.hpp:87-103`: 4 tiles at 32-bit half sync, 8 at 32-bit full sync / 16-bit half, 16 at 16-bit full) | `sync_tiles` | host: `sync_tiles = min(SYNC_TILES, dest_tiles(fp32_dest_acc_en, dst_full_sync_en))`, the same table as `get_dest_limit()`; the chain also clamps at run time (`chain.hpp:504-507`) | the chain's own clamp would make compute wait fewer tiles than the host sized for — harmless for correctness under implicit sync, but the host and kernel would disagree on the quantum the ring was sized for; clamping on the host keeps one source |
| Ring size ≤ 1,048,560 B per DFB (target page) | `ENTRIES_PER_NEO · C · page_bytes` | 8 × 4 × 4,096 B (float32 worst case) = 128 KiB ≪ cap | refused at program build |
| STRIDED dealing: R | C and W | C; C ∈ {1, 2, 4}; R + W ≤ 6 | thread plan | host `validate_thread_plan` raises on an illegal plan constant | — |
| Ring index ids ≤ 32 per cluster, ≤ 24 active per NEO | ring count | 3 rings | — |

### Regimes

The domain is `TARGET`. Phase 0 builds one regime; every other row is the record of an algorithm not built.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R0 `tile_stream_same_shape`** — flat tile split across the grid; per cluster one `sub` call over its contiguous range; implicit-sync reader/writer | **built** (Phase 0) | `layout == TILE and broadcast == "none"` (with Phase 0's dtype bf16; other dtypes reuse the regime, rows R3–R5) | `core_tiles` per cluster (whole range), handshake quantum `sync_tiles` | **minimum**: against DRAM (or the operands' L1 bank for an L1 cell), `a` crosses once, `b` once, `out` once: `3 · tensor_tiles · 2,048 B`. No cross-core traffic. No other regime applies in this domain: no axis is dependent, no operand is shared, so nothing can be saved by reuse or combine | the whole range as one block pays the chain's per-element init and format reconfig once per kernel instead of per tile or per row (catalog `compute_block_size`); `sync_tiles` pays the tile-regs acquire/commit/release and ring wait/pop/reserve/push once per 4 tiles instead of per tile |
| R1 `bcast_b_index_mapped` — `b` read per output tile by its broadcast index (`b_page(n, ht, wt)` with size-1 dims pinned to 0), FPU in-tile broadcast `BroadcastDim::Col` (over_w: `b` valid in column 0), `Row` (over_h: row 0), `Scalar` (over_hw), `None` (over_leading: index mapping only) | deferred — positive reason: `PHASE0["broadcast"] == ["none"]`; it is the next refinement's regime | `layout == TILE and broadcast in {over_w, over_h, over_hw, over_leading}` | per cluster: contiguous output range; the reader walks `(n, ht, wt)` of each output tile | `a` and `out` minimum; `b` crosses up to `core_tiles` tile reads per cluster (same bytes as a full-size `b`, never more than R0's) | as R0 |
| R1b `bcast_b_resident` — successor of R1: the cluster reads the `b` tiles its range needs **once** into a resident window and indexes it (`InputTileMapping::Row` / `Col` / `Scalar` with `Upfront` wait, `None` pop, on a `grid(Ht_c, Wt)` shape) | deferred — built on top of R1 once R1 exists; reachable because R0's reader already computes page ids from a start/count and the compute call already takes an `IterationShape` | R1's predicate and `b_tiles_needed_by_core · page_bytes ≤` budget share | `b` window spans the core's distinct `b` tiles | `b` crosses DRAM once per cluster for the tiles it needs (over_h of `[1,32,4096]`: ≤ 128 tiles per cluster instead of one per output tile) | as R0, plus `b` fetched once per block instead of per tile |
| R1c `bcast_b_mcast` — read the shared `b` once on an injector and multicast to all clusters (`ttnn.Mcast` + `McastArgs`) | rejected — superseded by R1b: the broadcast operand is small (one tile for over_hw, one tile-row per image for over_h) and each cluster needs only part of it; a multicast's rendezvous and payload to every receiver exceed one small DRAM read per cluster | — | — | adds cross-core payload to all 32 clusters | — |
| R2 `bcast_a` (`first_operand`) — `a` is the expanded operand. The FPU broadcasts only srcB, so compute `-(b - bcast(a))`: one chain `BinaryFpu<Sub, b, input(a, BroadcastDim::…)>` → `Negative` SFPU element → `PackTile` (negation is exact) | deferred — positive reason: `broadcast = first_operand` is not in PHASE0; R5 requires it later | `layout == TILE and broadcast == "first_operand"` | as R1 with the roles swapped | as R1 / R1b with `a` the reused operand | as R0 |
| R3 `fp32_operands` — float32 operands: FPU source registers hold 19 bits, so the 0-ulp float32 grade needs the SFPU subtract: both rings `UnpackToDest`, `binary_sfpu<SubBinary-type>` (or raw `sub_binary_tile`), chain `DataFormatReconfig::Disabled` | deferred — positive reason: `dtype = float32` not in PHASE0. Same R0 data movement and split; only the compute call and ring formats change | `dtype == float32` (R1 refuses `fp32_dest_acc_en = False`) | as R0 | minimum (pages are 4,096 B) | as R0 |
| R4 `fp32_output` / `acc16` — `output_dtype = float32` (output ring Float32, packed from 32-bit DEST; R2 refuses acc False), and `fp32_dest_acc_en = False` for bf16 (16-bit DEST; `sync_tiles` clamp rises to 8) | deferred — positive reason: values not in PHASE0; knob-turns on R0 (ring format, config translation, clamp table) | `output_dtype == float32` / `fp32_dest_acc_en == False` | as R0 | minimum | as R0 |
| R5 `row_major_flat` — ROW_MAJOR operands with `broadcast = none`: elementwise order is irrelevant, so the reader packs consecutive elements of the flat row-major buffer into tile-sized ring entries (explicit-sync ring assembled from stick reads, tagged transfers), the same `sub` call runs on them as on tiles, the writer scatters back to sticks — no tilize/untilize | deferred — positive reason: `layout = ROW_MAJOR` not in PHASE0 | `layout == ROW_MAJOR and broadcast == "none"` | flat element chunks of 1,024 elements, per-core contiguous | minimum (no re-layout pass) | as R0 plus fewer stick transfers per entry |
| R5b `row_major_tilize` — ROW_MAJOR with broadcast: reader gathers 32 sticks per tile-row, compute `tilize` → `sub` (R1 indexing) → `untilize` in one program | deferred — positive reason: needs R1 first and ROW_MAJOR in SUPPORTED | `layout == ROW_MAJOR and broadcast != "none"` | one tile-row × `Wt` per block | minimum DRAM; adds two in-core re-layout passes | tilize/untilize init amortized per tile-row block |
| R6 `split_on_host_into_two_ops` (e.g. expand `b` with a copy op, then subtract) | rejected — violates one dispatch per invocation and moves an extra output-sized tensor (GUIDANCE: expanding costs a copy of the output's size); R1/R1b supersede it | — | — | +2 crossings of an output-sized tensor | — |

Selection (Phase 0 has one regime): `validate()` admits only R0's cells; later the entry point selects by `(layout, broadcast)` exactly as the predicates above, and each regime gets regime-pinned tests.

### Traffic ranking

| Candidate split | DRAM crossings | Cross-core traffic | Occupancy on the YOLO shapes (`Ht = 1`) | Verdict |
|---|---|---|---|---|
| Flat `tile_id`, contiguous ranges (chosen) | `a`, `b`, `out` once each (minimum) | none | 263 / 1,050 / 13 tiles over 32 / 32 / 13 clusters | **chosen**: minimum bytes, finest balance |
| By tile-rows (`N·Ht`) | minimum | none | 1 tile-row → 1 cluster busy, 31 idle | same bytes; loses on occupancy (tiebreaker) |
| By tile-columns (`Wt`) per row | minimum | none | good for wide-short, poor for tall-narrow | same bytes; strictly less general than flat |
| 2D blocks (`Ht × Wt` tiles per cluster) | minimum; page reads become strided, not contiguous | none | good | same bytes, more complex index walk; no reuse to gain since no operand is shared |
| Split a dependent axis + combine | n/a | n/a | n/a | the op has no dependent axis: no combine exists to rank |

All splits move the same bytes because no operand is shared and no footprint overlaps (`blocking-model.md` §4: the extent is invisible to the split ranking here), so the extent is chosen by the extent ranking: the coarsest block (whole range) and the DEST-width quantum.

**Stall-shadow check.** The only waits are compute on the input rings and the writer on the output ring; all work a cluster has depends on the data being waited for, so nothing can be lifted into the stall. Overlap comes from ring depth (`ENTRIES_PER_NEO = 2 · sync_tiles`) and from implicit sync letting the reader issue the next read without a barrier. No reorder; no floating-point order change.

### Block schedule

Per cluster, per NEO `t` of `C` (the reader and writer deal tiles to NEOs through the STRIDED rings):

```cpp
// one block = this cluster's contiguous range [start_id, start_id + core_tiles)
load_block(start_id, core_tiles);          // reader: tagged implicit-sync reads of a[i], b[i] into the two input rings
subtract_block(neo_tiles, sync_tiles);     // compute: ONE sub call over this NEO's tiles
store_block(start_id, core_tiles);         // writer: tagged implicit-sync writes of out[i]
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `load_block` | `core_tiles` tiles of `a` and of `b`, reader thread `p` of `R` issuing tiles `p, p+R, …`, `a` then `b` per tile | nothing (streaming through rings) | accessor construction once per kernel; one tagged read per tile per operand; **no barrier** in the loop (DM0 ISR posts credits); tag handed back once at kernel end (`noc_async_read_set_trid(0, …)`) |
| `subtract_block` | `neo_tiles` tiles as `IterationShape::tiles(neo_tiles).block_size(sync_tiles)` | nothing | `compute_kernel_hw_startup` once; chain init + format reconfig **once per kernel**; tile-regs acquire/commit/release and ring wait/pop/reserve/push once per `sync_tiles` tiles |
| `store_block` | `core_tiles` tiles of `out`, writer thread `w` of `W` writing tiles `w, w+W, …` | nothing | one tagged write per tile; no barrier in the loop; tag handed back once at kernel end (`noc_async_write_set_trid(0, …)`) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization | small tensors (13 tiles for `[1,2,400]`, 2 tiles for `[1,1,32,64]`) put 1 tile on each of many clusters; per-cluster launch and kernel setup may dominate | `min_tiles_per_core` host knob (Phase 0 = 1): engage `min(grid, ceil(tensor_tiles / min_tiles_per_core))` clusters; measure 1 vs 4 vs 8 |
| Thread plan | the op is bound by data movement, so (1, 1, 1) is the target page's plan; with operands in L1 (shorter NoC path than DRAM) the reader may become issue-bound | (2, 2, 2) and (4, 4, 2) — one host constant; on `ttsim-perf`, model cycles per call |
| Handshake quantum | `sync_tiles = 4` may cost nothing and gain nothing | `SYNC_TILES` ∈ {1, 4, 8 with full-sync DEST} |
| Format reconfig | all-bf16 chain: the one reconfig per call is cheap but useless | `DataFormatReconfig::Disabled` on the three specs (catalog `compute_block_size`, second lever) — only while all three formats are equal |

Overlap lamp: does not apply — the block is consumed in a streaming quantum of `sync_tiles`, never held whole, so a large block never serializes movement against compute.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `a`, `b` in DRAM or L1 (interleaved tile pages) → `cb_minuend_tiles`, `cb_subtrahend_tiles` | bfloat16 tiles, 2,048 B | reader: `noc.async_read<NocOptions::TXN_ID>(accessor, ring, {.page_id = start_id + i}, {})`, `TensorAccessor(tensor::minuend)` / `(tensor::subtrahend)` | the accessor handles L1 and DRAM identically (GUIDANCE, Placement); implicit sync, no reserve/barrier/push; tag reset at end |
| rings → DEST | bf16 → srcA/srcB (`UnpackToSrc`, default) → 32-bit DEST | `sub` helper (`BinaryFpu<Sub>` + `PackTile`) | bf16 is exact in the 19-bit source registers; difference formed in FP32 DEST |
| DEST → `cb_difference_tiles` | bf16 (output dtype), rounded once by the packer | `PackTile` inside the same call | R6: rounded once |
| `cb_difference_tiles` → `out` (same buffer type as `a`) | bf16 tiles | writer: `noc.async_write<NocOptions::TXN_ID>(ring, accessor, {}, {.page_id = start_id + i})` | implicit sync; tag reset at end |

No inter-cluster communication in any built or deferred regime except the rejected R1c. Sharded placements are not in TARGET (`sharding = ["interleaved"]`); were they added, the cut is along `tile_id`, an independent axis, so every shard flavor would be a knob-turn consumed in place through a ring backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor` analogue), never re-read over the NoC.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a tile (`tile_id`); a cluster's block is its contiguous range |
| Grid | `device.compute_with_storage_grid_size()` (8 × 4 here); `ttnn.split_work_to_cores(grid, tensor_tiles, row_wise=True)` (catalog `noc_placement`: row-wise spreads along the DRAM-facing axis) |
| Per-core work | group 1: `ceil(tensor_tiles / num_cores)`, group 2: `floor(...)`; runtime args `num_tiles` (= `core_tiles`) and `start_id` per cluster, identical for reader and writer; compute takes `num_tiles` and derives `neo_tiles = num_tiles / C + (t < num_tiles % C)` from `get_my_thread_id()` |
| Remainder | `tensor_tiles = prod(shape[:-2]) · ceil(H/32) · ceil(W/32)` (per-image `ceil`, never `floor(N·H/32)`); `split_work_to_cores` gives the ragged groups; clusters with 0 tiles are not in the work unit's `target_nodes` |
| Worked (this target) | `[1,2,8400]` → 263 tiles: 7 clusters × 9 + 25 × 8. `[1,2,33600]` → 1,050: 26 × 33 + 6 × 32. `[1,2,400]` → 13: 13 × 1. `[2,512,1024]` → 1,024: 32 × 32. `[1,1,17,50]` → 2: 2 × 1 |
| Bound | **data movement**: per tile, 2 × 2,048 B in and 2,048 B out against one FPU subtract (no SFPU, one pass) — the target page's "one FPU op per tile over tensors in DRAM" case |
| Thread plan | **(R, C, W) = (1, 1, 1)**, host constant `THREAD_PLAN = (1, 1, 1)` in the program-artifacts module, passed as compile-time args `reader_threads`, `compute_threads`, `writer_threads` to all three kernels, and `num_threads` on each `KernelSpec`. Kernels are written for any legal plan (strided reader/writer loops, per-NEO count in compute), so changing the plan is editing one constant |
| Unit dealt to compute threads | the ring entry itself (one tile) — no padded rounds; no whole-round constraint on the split; a cluster with fewer tiles than C simply leaves later NEOs fewer (or zero) tiles |
| Tests the plan needs | (1) the program artifacts built with a non-default plan (`(2, 2, 2)` and `(4, 4, 2)`) give correct results on a shape whose per-cluster count is not a multiple of C (e.g. `[1,2,8400]`), and the built `ProgramSpec`'s `num_threads` equal the plan — under STRIDED rings a NEO that did not run leaves its tiles unproduced, so the writer cannot complete; (2) one device session: another program (e.g. `ttnn.add` or a prior `subtract`), then `subtract`, then `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)`, checking both results |

## Circular Buffers

Dataflow buffers (`DataflowBufferSpec`), STRIDED (default access pattern). `ENTRIES_PER_NEO = 2 · sync_tiles`; `C` from `THREAD_PLAN`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_minuend_tiles` | 0 | `a.buffer_page_size()` (2,048 B bf16) | `ENTRIES_PER_NEO · C` = 8 | streams over `tile_id`; live set spans `sync_tiles` per NEO; ×2 for double buffering | `a.dtype` (bfloat16) | reader | compute | reader: implicit (tagged read per entry); compute: wait/pop via the helper | whole kernel |
| `cb_subtrahend_tiles` | 1 | `b.buffer_page_size()` | `ENTRIES_PER_NEO · C` = 8 | same | `b.dtype` (bfloat16) | reader | compute | same | whole kernel |
| `cb_difference_tiles` | 2 | `out.buffer_page_size()` | `ENTRIES_PER_NEO · C` = 8 | same | output dtype (bfloat16) | compute | writer | compute: reserve/push via the helper; writer: implicit (tagged write per entry) | whole kernel |

No ring is listed in any `disable_dfb_implicit_sync_for`. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `core_tiles` tiles × 2 operands | dataflow API (no helper needed) | `a`, `b` tensor pages | `cb_minuend_tiles`, `cb_subtrahend_tiles`: one entry per tile, credited by the DM0 ISR | — |
| 2 | `subtract_block` | `tiles(neo_tiles).block_size(sync_tiles)` | yes: `compute_kernel_lib::sub` | `cb_minuend_tiles`, `cb_subtrahend_tiles`: waited and popped `sync_tiles` (tail: valid remainder) at a time | `cb_difference_tiles`: reserved and pushed `sync_tiles` (tail: valid remainder) at a time | all three empty at kernel end |
| 3 | `store_block` | `core_tiles` tiles | dataflow API | `cb_difference_tiles` | `out` tensor pages | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (boot) | raw_api (mandatory caller init) | `compute_kernel_hw_startup(dfb::cb_minuend_tiles, dfb::cb_subtrahend_tiles, dfb::cb_difference_tiles)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:27-29` (caller owns it); usage `quasar_op_examples/toy_thread_plan/kernels/compute.cpp:33` | the three ring ids | — | — | — |
| `subtract_block` | helper | `compute_kernel_lib::sub<AInput, BInput, Output>(IterationShape, a, b, out)` → `eltwise_chain(BinaryFpu<BinaryFpuOp::Sub, …>, PackTile<…>)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:54-61`; `convenience.inl:16-22`; Sub dispatch `eltwise/core/chain.inl:1397-1398, 1457-1458` | `AInput = input(dfb::cb_minuend_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block)`; `BInput = input(dfb::cb_subtrahend_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block)` (no broadcast); `Output = output(dfb::cb_difference_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)`; shape `IterationShape::tiles(neo_tiles).block_size(sync_tiles)` (default `BlockTailSync::ValidTiles`) | `cb_minuend_tiles`, `cb_subtrahend_tiles` | `cb_difference_tiles` | `neo_tiles` (block: the whole per-NEO range), `sync_tiles` (quantum, CT arg). Legality: Block mapping with PerBlockSize/PerBlockSize `chain.inl:29-62`; output PerBlockSize pair `chain.inl:40-50`; `IterationShape` `chain.hpp:146-174` |
| `load_block` | raw_api (dataflow) | `Noc::async_read<NocOptions::TXN_ID>(accessor, ring, {.page_id}, {})`; `noc_async_read_set_trid(0, noc.get_noc_id())` at end | pattern `quasar_op_examples/toy_thread_plan/kernels/reader_implicit.cpp:26-35` | `TensorAccessor(tensor::minuend)`, `(tensor::subtrahend)`; stride `R` from `get_my_thread_id()` | tensors | `cb_minuend_tiles`, `cb_subtrahend_tiles` | `num_tiles`, `start_id` (RT) |
| `store_block` | raw_api (dataflow) | `Noc::async_write<NocOptions::TXN_ID>(ring, accessor, {}, {.page_id})`; `noc_async_write_set_trid(0, noc.get_noc_id())` at end | pattern `quasar_op_examples/toy_thread_plan/kernels/writer_implicit.cpp:24-30` | `TensorAccessor(tensor::difference)`; stride `W` | `cb_difference_tiles` | tensor | `num_tiles`, `start_id` (RT) |

Helpers considered and rejected for the raw entries:

| Entry | Candidate | Mismatch | Reason |
|---|---|---|---|
| `load_block` / `store_block` | `dfb_helpers_dataflow.hpp` / `local_copy_helpers_dataflow.hpp` | — | no kernel-library dataflow helper wraps the Quasar implicit-sync tagged read/write of an interleaved accessor; the target page names the raw tagged call as the protocol (*Ring protocols*), and tt-metal's Quasar examples use it directly |
| boot | none | `convenience.hpp:27-29` | the helper contract leaves engine init to the caller |

Host side (Python, `ttnn.experimental`): `DataflowBufferSpec(unique_id, entry_size, num_entries, data_format_metadata)`, `TensorParameter`, `KernelSpec(num_threads, dfb_bindings=[producer_of/consumer_of], tensor_bindings, compile_time_args, runtime_arg_schema, hw_config)`, `create_reader_datamovement_config()` / `create_writer_datamovement_config()`, `ComputeHardwareConfig(fpu_math_fidelity, sfpu_precision_mode, enable_32_bit_dest, double_buffer_dest)` (`ttnn/cpp/ttnn-nanobind/program_spec.cpp:482-560`), `WorkUnitSpec(target_nodes=core_set)`, `ProgramRunArgs`, `ttnn.generic_op([a, b, out], spec, run_args, tensor_indices)` — all as in `toy_thread_plan_program_artifacts.py`.

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `subtract_block` (R0, built) | `BinaryFpu<Sub>` | `cb_minuend_tiles`: All | `cb_subtrahend_tiles`: All | None |
| R1 over_w (deferred) | `BinaryFpu<Sub>` | `cb_minuend_tiles`: All | `cb_subtrahend_tiles`: Col0 (`b` W = 1) | Col |
| R1 over_h (deferred) | `BinaryFpu<Sub>` | All | Row0 (`b` H = 1) | Row |
| R1 over_hw (deferred) | `BinaryFpu<Sub>` | All | element (0,0) | Scalar |
| R2 first_operand (deferred) | `BinaryFpu<Sub>(b, bcast a)` then negate | `cb_subtrahend_tiles`: All (as srcA) | `cb_minuend_tiles`: Col0 / Row0 / (0,0) (as srcB) | Col / Row / Scalar |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Transaction tag | implicit-sync kernels leave a ring's id in the hart's tag register | last statement of reader `noc_async_read_set_trid(0, noc.get_noc_id())`, of writer `noc_async_write_set_trid(0, noc.get_noc_id())`; the session test of Work Distribution |
| `PerBlockSize` under a thread plan not yet exercised on Quasar (the toy verified `PerTile` and `Upfront/AtEnd`) | Phase 0 runs C = 1, but the plan lamp turns C up | the non-default-plan test of Work Distribution covers it; `sync_tiles = 1` is the value-only fallback with no code change |
| Host and kernel disagree on `sync_tiles` | the chain clamps to `DEST_AUTO_LIMIT` silently | host clamps with the same table from `fp32_dest_acc_en` / `dst_full_sync_en` and passes the clamped value; ring depth derives from it |
| Tile count from shape with `floor` | `[1,2,8400]` has H = 2: `floor(1·2/32) = 0` tiles | `tensor_tiles` from `out.buffer_num_pages()`, asserted equal to the per-image `ceil` formula |
| Tiny tensors on a 32-cluster grid | `[1,1,32,64]` has 2 tiles: 30 clusters must not be launched with 0 tiles | `target_nodes` = only the clusters `split_work_to_cores` gave work |
| Output placement | R9: L1 operand → L1 output | output spec takes `a.memory_config()` |
| `compute_kernel_config` type | the signature takes a Metal 1 `ComputeConfigDescriptor`; Quasar needs `ComputeHardwareConfig` | one translation function (Parameters) after `default_compute_kernel_config()` |

### Structural impossibilities

None to add: the specification's `INVALID = []` is right for this rectangle.

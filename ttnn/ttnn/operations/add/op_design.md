# Operation Design: add

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (one FPU binary op per tile; data-movement bound, see Work Distribution) |
| Goal | `out = a + b` elementwise, with torch broadcasting in TARGET. Phase 0 (`PHASE0` of the specification): two bfloat16 TILE operands of one shape, interleaved in DRAM or L1, `fp32_dest_acc_en=True`, no `dtype=`, every alignment, ranks 2–4. One `ttnn.generic_op` dispatch per call, on every cluster of the grid. |
| Math | `out[i] = a[i] + b[i]` (Phase 0: `a.shape == b.shape`); TARGET: `out[idx] = a[bcast_a(idx)] + b[bcast_b(idx)]` |
| References | Specification `eval/specs/add/spec.py` (authoritative); target page `.claude/references/targets/quasar.md`; backend page `.claude/references/backends/ttsim.md`; library gaps `.claude/references/quasar-library-gaps.md`; host-program reference `.claude/references/quasar_op_examples/` (`toy_thread_plan/` is the structural twin of this op: a two-input eltwise FPU op with implicit-sync reader/writer); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md`; `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; perf catalog `ttnn/ttnn/operations/examples/master.md` (entries `double_buffer`, `noc_placement`, `width_split`, `tile_reorder`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (probe on this backend: `compute_with_storage_grid_size()` = 8-4). Op code queries `device.compute_with_storage_grid_size()` | target page; probe 2026-11-21 |
| L1 per core | 4,194,304 B per cluster, shared by its 4 NEOs and 8 DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size()` = 3,963,328 B on this target (probe 2026-11-21). No reserve knob: the op has no budget solve (footprint `3 × RING_DEPTH_BLOCKS × block_tiles × C × tile_bytes` = 49,152 B in Phase 0 and ≤ 3,145,728 B over the bounded knob ranges, asserted against the query; see `l1_ledger.md`) | probe |
| Host program | `ttnn.ProgramSpec` (Metal 2.0: `ttnn.experimental.KernelSpec` / `DataflowBufferSpec` / `WorkUnitSpec` / `ProgramRunArgs`, dispatched with `ttnn.generic_op`) — reference `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1, 2, 4}); 6 user DM cores (R + W ≤ 6); on STRIDED rings R | C and W | C; ring capacity counted per NEO; 32 DFBs per cluster, ≤ 24 active per NEO; one ring ≤ 1,048,560 B compute-side | target page, *Cluster model* / *Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` and the legacy C++ device op (Metal 1 kernels refused at creation). `bfloat8_b` / `bfloat4_b` (no path on this part: refuse in `validate()`, queue no refinement). From the library-gaps reference: `ReduceFp32Mode::Accurate`; `ReduceAlgorithm::AccumulateViaAdd`; MAX+`REDUCE_ROW` `Accumulate`; SFPU post-op inside `reduce` with >1 output tile; reduce indexed access under a thread plan; `matmul_block_helpers.hpp` and `eltwise/binary/sfpu/minmax.hpp` (do not compile); `UnpackToDest` of a **bfloat16** page into 32-bit DEST (ttsim aborts) — so Phase 0 keeps the default unpack mode; eltwise chain operand reconfig with `UnpackToDest`; uint16/uint32 ring formats; `DEVICE_PRINT` from compute (prints nothing); an untagged NoC transfer after an implicit-sync program (hang / `tile counter overflowed capacity`) — so every DM kernel here is implicit-sync and hands its tag back; compute `KernelSpec` must set `fpu_math_fidelity` explicitly (ttsim). Eltwise helpers keep the previous pack target when the output ring changes — not hit here (one output ring), recorded so a fused refinement calls `pack_init`. | target page; `quasar-library-gaps.md` |

Every worked number below is for `quasar` (32 clusters, bf16 tiles of 2,048 B). Every predicate the op evaluates queries the device.

## Parameters

| Name | Type | Required | Valid Range (Phase 0) | Default | CT/RT |
|------|------|----------|-----------------------|---------|-------|
| `input_tensor_a` | `ttnn.Tensor` | yes | bfloat16, TILE, interleaved DRAM or L1, rank 2–4 | — | tensor binding `a` |
| `input_tensor_b` | `ttnn.Tensor \| float` | yes | a tensor of `a`'s shape, dtype, layout; interleaved DRAM or L1. A Python number is refused (`broadcast="number"`, deferred) | — | tensor binding `b` |
| `dtype` (kw-only) | `ttnn.DataType \| None` | no | `None` only (`output_dtype` axis) | `None` → operands' dtype | host |
| `compute_kernel_config` (kw-only) | `ttnn.ComputeConfigDescriptor \| None` | no | `fp32_dest_acc_en=True`; `math_fidelity`, `math_approx_mode` not gated (R3) | `default_compute_kernel_config()` = `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)` (R2: defined once, exported) | compute `ComputeHardwareConfig` |
| `THREAD_PLAN` (host constant) | `(R, C, W)` | — | C ∈ {1,2,4}, R+W ≤ 6, R\|C, W\|C | `(1, 1, 1)` | CT `reader_threads`, `compute_threads`, `writer_threads` on all three kernels |
| `BLOCK_TILES` (host constant) | int | — | 1 ≤ · ≤ `DEST_AUTO_LIMIT` | `4` | CT `block_tiles` (compute), after the clamp in Blocking Model |
| `RING_DEPTH_BLOCKS` (host constant) | int | — | ≥ 2 | `2` | host only: ring `num_entries` |
| `MIN_TILES_PER_CORE` (host constant) | int | — | ≥ 1 | `1` | host only: work split |
| `num_tiles` | uint32 | — | this cluster's tile count | — | RT (reader, compute, writer) |
| `start_id` | uint32 | — | this cluster's first page id | — | RT (reader, writer) |

The private program builder takes `threads=`, `block_tiles=` and `ring_depth=` overrides (defaulting to the constants above) so tests and the perf pass can turn each knob without editing code; the public `add` signature equals the specification's stub exactly.

### Compute config → `ComputeHardwareConfig`

| `ComputeConfigDescriptor` field | `ttnn.experimental.ComputeHardwareConfig` field |
|---|---|
| `math_fidelity` | `fpu_math_fidelity` (always set explicitly — ttsim gap) |
| `fp32_dest_acc_en` | `enable_32_bit_dest` |
| `math_approx_mode` | `sfpu_precision_mode` = `Precision.Approximate` if set else `Precision.Precise` |
| `dst_full_sync_en` | `double_buffer_dest = not dst_full_sync_en` |

### Validation and dispatch (host, in this order)

| # | Step | Raises |
|---|------|--------|
| 1 | Resolve `config = compute_kernel_config or default_compute_kernel_config()` | — |
| 2 | Shape of b: `()` if a Python number, else `tuple(b.shape)`. If b is a tensor, `torch.broadcast_shapes(a.shape, b.shape)`; failure → X1 | `ValueError` |
| 3 | Axes: `dtype` = a.dtype (a tensor b of another dtype → refuse), `layout` = a.layout (b's must match), `fp32_dest_acc_en` = `bool(config.fp32_dest_acc_en)`, `output_dtype` = `dtype` arg, `memory_config`/`sharding` = `eval.memory.memory_axes(t)` for **each** tensor operand (each must be in SUPPORTED: a sharded operand is refused by name, R10), then `INPUT_TAGGERS = AXES` (imported from the specification, never redefined) on `(shape_a, shape_b)` for `alignment`, `rank`, `broadcast` | `NotImplementedError` (typed `UnsupportedAxisValue` / `ExcludedCell`) |
| 4 | R1: `dtype=float32` with `fp32_dest_acc_en=False` refused (unreachable in Phase 0, kept as the rule) | `NotImplementedError` |
| 5 | SUPPORTED (= PHASE0) and EXCLUSIONS (empty) | `NotImplementedError` |
| 6 | `out = ttnn.allocate_tensor_on_device(a.spec, a.device())` (same shape, dtype, layout, memory config → R9: DRAM operands give DRAM output, L1 give L1) | — |
| 7 | Build the program (below), one `ttnn.generic_op([a, b, out], spec, run_args, tensor_indices)` | — |

No index axes (no `dim`).

## Tensors

### Input

| Property | Requirement (Phase 0) |
|----------|-------------|
| Shape | rank 2–4; `a.shape == b.shape` (`broadcast="none"`); any H, W (tile-padded per image) |
| Dtype | bfloat16, both operands |
| Layout | TILE |
| Memory | interleaved, DRAM or L1 (per operand; each read through its own `TensorAccessor`) |

### Output

| Property | Value |
|----------|-------|
| Shape | `torch.broadcast_shapes(a.shape, b.shape)` (= `a.shape` in Phase 0) |
| Dtype | `dtype` if given, else the operands' (bfloat16 in Phase 0) — the fp32 DEST sum rounded once by the packer (R8) |
| Layout | TILE (same as a) |
| Memory | a's buffer type, interleaved (R9) |

## Blocking Model

### Axes

Phase 0 has no broadcast, so a, b and out have identical tile-padded page sequences: page `p` of out is page `p` of a plus page `p` of b. Every logical axis is independent and treated identically and is contiguous in page order, so the leading, tile-row and tile-column axes fold honestly into one flat page axis `tiles` (blocking-model §1). The block is a run of `block_tiles` consecutive pages.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `leading` (dims before H: N·C, B, or 1) | independent — each output element reads one element of each operand at its own index | folded into `block_tiles` (flat) | folded | `BLOCK_TILES` | folded into the flat page split | knob-turn while no broadcast; the broadcast regime (`over_leading`) unfolds it (scheme-change, regime `bcast_resident_b`) |
| `tile_rows` (`ceil(H/32)` per image) | independent — same reason | folded into `block_tiles` | folded | `BLOCK_TILES` | folded into the flat page split | unfolded by `over_h` broadcast (b reuse-shared along it) |
| `tile_cols` (`ceil(W/32)`) | independent — same reason | folded into `block_tiles` | folded | `BLOCK_TILES` | folded into the flat page split | unfolded by `over_w` broadcast (b reuse-shared along it) |
| `tiles` (flat page index `p`, `num_tiles = out.buffer_num_pages()`) | independent — no cross-page dependency | `block_tiles` | 4 (bf16, fp32 DEST half-sync: = one DEST section) | host `BLOCK_TILES`, clamped once to `block_tiles = max(1, min(BLOCK_TILES, ceil(max_core_tiles / C)))`, passed as compute CT arg; ring capacities derive from it | contiguous page ranges over the **full grid** (`split_work_to_cores(grid, num_tiles, row_wise=True)`), each cluster's range dealt round-robin over its C NEOs | knob-turn |
| intra-tile elements (32 × 32) | independent | the tile (hardware unit) | 1 tile | — | never split | none |

**Why `block_tiles` = 4 and not the whole per-core assignment** (the coarse default yields here to the overlap lamp, stated): (1) bytes moved are constant in the block extent — no operand is shared and no footprints overlap — so the split ranking says nothing about the extent; (2) with one helper call per NEO over its whole assignment (Block schedule), init/reconfig is paid once per kernel, so the only per-block fixed cost left is the ring handshake plus one DEST-section turnaround, which the block amortizes; (3) the op is data-movement bound and the per-cluster assignments in `INPUTS` are small (Work Distribution: 1–33 tiles; the YOLOv8 shapes 1–9 except `[1,2,33600]` at 32–33), and pipeline fill + drain costs about two blocks, so a block of the whole assignment serializes read → add → write (up to ~2× on a stream whose read and write cost about the same); (4) the timed simulator measured the twin op (`toy_thread_plan`, multiply) at blocks of 1, 4 and 8 within 2% of each other at the DRAM floor (README table). 4 is the largest block that is one DEST section in every DEST mode (`dest_helpers.hpp:89-100`: 4 at fp32 half-sync, the Phase 0 mode), so one block = one acquire/commit/wait/release. A larger L1 block with several DEST sections per block is the perf lamp's alternative.

**Intermediate data.** The only intermediate is the sum in DEST, one block at a time; it has the `tiles` axis only and is never stored, so there is no stage to assign. No core collects or coordinates: every cluster runs the same three kernels on its own range.

**Stall-shadow check.** No stage waits on a peer core or on a long-latency fetch other than its own stream: compute waits on the reader's next block and the writer on compute's. The depth-2 rings already overlap the next block's reads with this block's add and the previous block's writes. Addition commutes, so the reader's a/b read order is free (it alternates a, b per page) — no reorder is needed. Nothing else is independent of the wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_a_tiles` | `RING_DEPTH_BLOCKS` | 2 blocks per NEO | reader fills block k+1 while compute adds block k; with implicit sync the reader keeps up to `2 × block_tiles` = 8 reads of a in flight with no barrier |
| `cb_input_b_tiles` | `RING_DEPTH_BLOCKS` | 2 blocks per NEO | same for b (16 reads in flight per cluster in total) |
| `cb_output_tiles` | `RING_DEPTH_BLOCKS` | 2 blocks per NEO | compute packs block k while the writer drains block k−1 |

One knob for all three rings (they stream in lockstep); a refinement may split it.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Eltwise chain DEST section (`chain.inl:1660-1676` `chain_max_block`, runtime clamp `chain.inl:3339-3350`) | `block_tiles` ≤ `DEST_AUTO_LIMIT` (4 at fp32 half-sync; lane width 1 for `BinaryFpu`+`PackTile`) | Host constant 4 is legal in every DEST mode; compute kernel `static_assert(block_tiles <= DEST_AUTO_LIMIT)` | The chain silently shrinks its wait/pop/reserve/push quantum below `block_tiles` while the ring is sized `depth × block_tiles`; when the shrunk quantum does not divide the capacity, a block window straddles the ring's end and the chain reads the wrong tiles — wrong results, no error |
| Ring-wrap invariant (blocking-model §2) | ring capacity per NEO vs. block quantum | capacity = `RING_DEPTH_BLOCKS × block_tiles` per NEO by construction; every block but each NEO's last is exactly `block_tiles` (`BlockTailSync::ValidTiles`, `chain.hpp:119-122`) | a short block in the middle of a walk misaligns every later window |
| DFB ring size (target page) | `RING_DEPTH_BLOCKS × block_tiles × C × tile_bytes` ≤ 1,048,560 B | host assert (Phase 0: 16,384 B) | program build failure |
| Active DFBs per NEO (≤ 24) | number of rings | 3 rings | — |
| Thread plan (target page) | C ∈ {1,2,4}, R+W ≤ 6, R\|C, W\|C | host `validate_thread_plan` (as `toy_thread_plan_program_artifacts.py`) | a thread's ring turn lands on the wrong NEO: hang |
| Tile page | the transfer unit is one whole tile page (`buffer_page_size()`, 2,048 B bf16) | pages are never split | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `stream_flat` | **built** | `broadcast == "none"` and TILE and interleaved (DRAM/L1) and dtype bf16 (Phase 0 rectangle) | `block_tiles` consecutive pages of the cluster's range, per NEO | **minimum**: a and b each cross their buffer tier (DRAM, or the L1 banks over the NoC) once, out once, page for page. Measured against tile-padded pages: the padding of a non-aligned shape (e.g. `[1,2,8400]` = 263 tiles with 2 valid rows of 32) is part of the TILE layout the caller chose and cannot be skipped by the op | one ring handshake per side and one DEST section turnaround per block (init/reconfig once per kernel regardless); capped by the DEST section (Mechanism caps) and by the overlap lamp |
| `stream_flat_fp32_sfpu` | deferred — PHASE0 admits bfloat16 only; float32 (0 ulp required) is a TARGET value | dtype float32 (and `fp32_dest_acc_en`, R1) | same | minimum (pages of 4,096 B) | same. The compute element changes: FPU source registers do not hold a full float32, so the exact add is the SFPU `binary_sfpu<AddBinary>` (`eltwise/binary/sfpu/basic.hpp:20`, `add_binary_tile` `eltwise_binary_sfpu.h:103`, exact in fp32 on the simulator per the gaps file) with 32-bit pages unpacked to DEST. Reader, writer, split and rings unchanged — reachable from Phase 0 by a predicate on dtype |
| `bcast_resident_b` | deferred — PHASE0 excludes broadcast (`over_w`, `over_h`, `over_hw`, `over_leading`) | `broadcast != "none"` | output pages walked as `grid(rows, tile_cols)` within the cluster's range; b's slice for that range held resident in its ring (`WaitPolicy::Upfront`, `PopPolicy::None/AtEnd`, mapping `Row`/`Col`) | a once, out once; b at most once per cluster that needs it (≤ engaged clusters × one b slice, always ≤ out's page count). Minimum for b is once overall; the excess is ≤ 32 × one row/column/tile of b | as `stream_flat`, plus b's slice is fetched once per range instead of once per block. FPU broadcast is srcB-only (`chain.hpp:354-357`), so the host swaps the operands when a is the smaller one (addition commutes; R4). Intra-tile broadcast: `over_w` → `BroadcastDim::Col`, `over_h` → `Row`, `over_hw` → `Scalar`, `over_leading` → none (b page = out page mod b's page count). Scheme-change: the flat page walk unfolds into (leading, tile_rows, tile_cols); the host split and reader/compute/writer skeleton stay |
| `bcast_reread_b_per_tile` | rejected — superseded by `bcast_resident_b` | `broadcast != "none"` | — | b re-read per output page: b crosses its tier `num_tiles / |b|` times (e.g. `over_h` on `[2,4,128,512]`: 4× b) | dead end: no good scheme passes through it |
| `number_scalar` | deferred — PHASE0 excludes a number operand | `broadcast == "number"` | as `stream_flat`, no b ring | a once, out once, no b traffic | the number (as float32 bits, R7) is a compute runtime arg to `unary<AddUnary>` (`eltwise/unary/scalar.hpp:26`); availability of that SFPU struct on Quasar to be confirmed at the refinement; fallback: reader fills one scalar tile once, `BroadcastDim::Scalar` |
| `row_major_tilize` | deferred — PHASE0 is TILE only | `layout == ROW_MAJOR` | 32-stick tile-rows: reader reads sticks, compute tilizes (`tilize_helpers.hpp`), adds, untilizes (`untilize_helpers.hpp`) | minimum (each stick read once, written once) | adds a tilize and untilize per block; no shared operand. A no-broadcast fast path that adds raw contiguous 1,024-element runs as pseudo-tiles (elementwise math does not see the face order) is to be costed at that refinement |
| `sharded_in_place` | deferred — not in TARGET (`sharding: ["interleaved"]`; the trace's height-sharded call is not served, DECISIONS 2026-11-21; refused by R10) | any `*_SHARDED` operand | the whole resident shard per cluster, synchronized per `block_tiles` | no NoC traffic: rings borrowed from the shard (`DataflowBufferSpec(borrowed_from=...)`), reader and writer drop out | every axis is independent, so height, width and block sharding are all knob-turns on the compute kernel |
| `split_dependent_axis_combine` | rejected — no dependent axis exists (each output element depends on one element per operand); nothing to combine | — | — | — | — |
| `mcast_shared_b` | rejected — superseded by `bcast_resident_b`'s per-cluster read | `broadcast != "none"` | — | clusters need different b slices except for `number`/`over_hw`, where the shared payload is one tile (≤ 32 × 2,048 B in total); a multicast saves ≤ 31 tile reads and adds a semaphore round trip | — |

Phase 0 has one built regime: no selection function is needed. The first refinement that builds a second one adds the selection on the predicates above and regime-pinned tests.

### Traffic ranking

Every split moves the same bytes for a pointwise op without broadcast (each page of a, b, out crosses its tier exactly once, no cross-cluster traffic), so the ranking is decided by the tiebreakers, occupancy and balance.

| Rank | Candidate split | a / b / out crossings | Cross-core | Occupancy and balance | Verdict |
|------|-----------------|------------------------|------------|-----------------------|---------|
| 1 | Flat page ranges over the full grid, tile granularity (`row_wise=True`) | 1 / 1 / 1 | none | `min(num_tiles, 32)` clusters; imbalance ≤ 1 tile (≤ 1 round of C tiles) | **chosen** |
| 2 | Flat page ranges, block granularity (`block_tiles` units) | 1 / 1 / 1 | none | imbalance up to `block_tiles − 1` tiles: `[1,2,8400]` (263 tiles) → 66 blocks over 32 clusters = 8 or 12 tiles per cluster instead of 8 or 9 | rejected |
| 3 | Tile-row ranges (`leading × tile_rows`) | 1 / 1 / 1 | none | strands wide-short tensors (catalog `width_split`): `[1,2,8400]` and `[1,2,33600]` have one tile-row → one cluster | rejected |
| 4 | Tile-column ranges | 1 / 1 / 1 | none | strands tall-narrow tensors: `[1,1,1600,128]` has 4 tile-columns → 4 clusters; ranges become strided in page order | rejected |
| 5 | Split a dependent axis + combine | — | — | no dependent axis | n/a |

### Block schedule

Per cluster, per NEO `t` of C (its tiles are the cluster's pages `t, t+C, …`; `neo_tiles = num_tiles / C + (t < num_tiles % C)`; `num_blocks_this_neo = ceil(neo_tiles / block_tiles)`):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_neo; ++block_idx) {
    load_block(block_idx);   // reader: this block's pages of a and of b into their rings
    add_block(block_idx);    // compute: a + b for the block in one DEST section, packed to the output ring
    store_block(block_idx);  // writer: this block's output pages to out
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_extent ≤ block_tiles` pages of a and of b (last block of a NEO may be short) | nothing (streamed) | one tagged implicit-sync read per page (`noc.async_read<NocOptions::TXN_ID>`), no barrier, no reserve/push (DM0's interrupt handler posts credits); tag handed back once at kernel end; no multicast, no semaphore |
| `add_block` | `block_extent` tiles, one DEST section | nothing between blocks | `compute_kernel_hw_startup` once per kernel; the chain's unpack/math/pack init and reconfig **once per kernel** (one `add<>` call covers all of the NEO's blocks); wait/pop of a and b and reserve/push of out **once per block**; DEST acquire/commit/wait/release **once per block** |
| `store_block` | `block_extent` output pages | nothing | one tagged implicit-sync write per page, no barrier, no wait/pop; tag handed back once at kernel end |

No cross-cluster barrier, multicast or semaphore exists in the op.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (block extent) | `block_tiles` = 4 favours overlap over amortization; at 33 tiles per cluster a larger block may win back handshake cost | `block_tiles` ∈ {1, 2, 4} in the one-call form, and an L1 block of 8 / 16 / whole assignment in the per-block-call form (`WaitPolicy::Upfront` + `PopPolicy::AtEnd`, DEST sections inside, as `toy_thread_plan/kernels/compute.cpp`), on `[1,2,33600]` and `[1,2,8400]`, model cycles on `ttsim-perf` |
| Ring depth | implicit sync keeps `free entries` reads in flight; 8 per input may not cover DRAM latency (catalog `double_buffer`, `trid_double_issue`: depth 3–4) | `RING_DEPTH_BLOCKS` ∈ {2, 3, 4} |
| Thread plan | the plan is (1,1,1) from the bound; two input streams per output could make one reader hart issue-bound | (2,2,2) and (4,4,2) — the twin op measured 84,562 / 84,542 / 85,938 cycles (no gain) |
| Grid synchronization (occupancy) | small calls (`[1,2,400]`: 13 tiles → 13 clusters of 1 tile; `[1,1,400,128]`: 52 tiles → 1–2 per cluster) pay a per-cluster launch for one or two tiles | `MIN_TILES_PER_CORE` ∈ {1, 2, 4, 8} (engage `ceil(num_tiles / MIN_TILES_PER_CORE)` clusters) |
| Placement | `row_wise=True` spreads a split across the DRAM-facing axis (catalog `noc_placement`, measured on Wormhole) | `row_wise=False` |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| a, b in DRAM or L1 (interleaved) | bf16 tile pages, 2,048 B | `TensorAccessor(tensor::a)`, `TensorAccessor(tensor::b)` | page id `start_id + i`, same id for a, b, out |
| → `cb_input_a_tiles`, `cb_input_b_tiles` | bf16 tiles | reader: `noc.async_read<NocOptions::TXN_ID>(acc, ring, {.page_id}, {})`, implicit sync | reader thread r reads the cluster's pages r, r+R, …; per page: a then b |
| compute | srcA ← a, srcB ← b, FPU add into fp32 DEST | `compute_kernel_lib::add<…>` (eltwise chain), default unpack mode | packer rounds fp32 DEST to bf16 once |
| → `cb_output_tiles` | bf16 tiles (output dtype) | chain `PackTile`, reserve/push per block | |
| → out (a's buffer type) | bf16 tile pages | writer: `noc.async_write<NocOptions::TXN_ID>(ring, acc_out, {}, {.page_id})`, implicit sync | writer thread w writes pages w, w+W, …; last statement `noc_async_write_set_trid(0, noc.get_noc_id())` |

No Tensix-to-Tensix communication in any built or deferred-and-reachable regime (`mcast_shared_b` is rejected).

### TARGET axes as unlocks

| TARGET value outside PHASE0 | Unlock class | What changes |
|---|---|---|
| `dtype=float32` | scheme-change in compute only (regime `stream_flat_fp32_sfpu`) | SFPU add, 4,096 B pages; split and DM unchanged |
| `dtype=bfloat8_b` | none — unbuildable on quasar | refused; no refinement queued |
| `fp32_dest_acc_en=False` | knob-turn | `enable_32_bit_dest=False`; `DEST_AUTO_LIMIT` becomes 8 (`BLOCK_TILES` may follow) |
| `layout=ROW_MAJOR` | scheme-change (`row_major_tilize`) | tilize/untilize phases |
| `broadcast` ≠ none | scheme-change (`bcast_resident_b`, `number_scalar`) | 2-D walk, resident b |
| `output_dtype` float32 / bfloat16 | knob-turn | `cb_output_tiles` format and the output tensor spec; the packer converts from DEST |
| `memory_config=L1` | built (Phase 0) | output allocated in L1 from a's spec |
| `sharding` other than interleaved | not in TARGET (`sharded_in_place` deferred) | refused by R10 |

### Structural impossibilities

None beyond the specification's `INVALID` (`bfloat8_b` × ROW_MAJOR).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a tile page (the ring entry); a cluster's contiguous page range is cut into blocks of `block_tiles` per NEO |
| Grid | `device.compute_with_storage_grid_size()` (8 × 4 here) |
| Per-core work | `ttnn.split_work_to_cores(grid, ceil(num_tiles / C), row_wise=True)` over rounds of C pages, converted back to pages (`num_tiles` and `start_id` RT args per cluster; the last cluster's last round may be short — tiles dealt round-robin to NEOs may stop anywhere). At C = 1 this is the tile split. Only clusters with work are in the `WorkUnitSpec`'s `target_nodes`. Worked (quasar, C = 1): `[1,2,8400]` 263 tiles → 7 × 9 + 25 × 8; `[1,2,33600]` 1,050 → 26 × 33 + 6 × 32; `[1,1,1600,128]` 200 → 8 × 7 + 24 × 6; `[1,1,400,128]` 52 → 20 × 2 + 12 × 1; `[1,2,400]` 13 → 13 × 1 |
| Remainder | `num_tiles = out.buffer_num_pages()` = `prod(leading) × ceil(H/32) × ceil(W/32)` (tile-padded per image; never `volume / 1024`: `[1,2,8400]` is 263 pages, not 16) |
| Bound | **data movement**: per output tile the op reads 2 × 2,048 B and writes 2,048 B and does one FPU add of one tile — no SFPU work, one pass |
| Thread plan | **(1, 1, 1)**, host constant `THREAD_PLAN` in the program-artifacts module; counts passed as CT args `reader_threads`, `compute_threads`, `writer_threads`, and every kernel sizes its loops from them and `get_my_thread_id()` (reader: pages `r, r+R, …`; writer: `w, w+W, …`; compute: `neo_tiles` above). Reason: target page *Choosing the plan* — a data-movement bound op with one FPU op per tile sits on the DRAM floor with one thread of each kind under implicit sync; the twin op measured no gain from more threads |
| Unit dealt to compute threads | the ring entry itself (one tile): no padded rounds; ring capacity counted per NEO (`× C`) |
| Tests the plan needs | (1) the op at plans (1,1,1), (2,2,2), (4,4,2) through the builder's `threads=` override, on a shape where every NEO of every engaged cluster gets tiles (e.g. `[1,3,320,320]`, 300 tiles), result correct — loops sized from the plan's counts make a missing thread a missing page or a hang; (2) one device session: another program first (e.g. `ttnn.to_layout` of an input, then `add` twice), then `ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)` after `add`, results checked; (3) `block_tiles` ∈ {1, 2, 4} on a shape with ragged per-NEO tails; (4) structural count: clusters engaged = `min(num_tiles, grid)` read off the program's core set |

## Circular Buffers

DataflowBuffers (rings), STRIDED (default), bound by `unique_id`; "Index" is the binding order only.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_input_a_tiles` | 0 | `a.buffer_page_size()` (2,048 B) | `RING_DEPTH_BLOCKS × block_tiles × C` (Phase 0: 8) | spans `tiles` (`block_tiles`) × depth; streams over leading / tile_rows / tile_cols (folded) | a.dtype (bf16) | reader | compute | reader: implicit (tagged read per entry); compute: wait/pop per block (chain `PerBlockSize`) | whole program |
| `cb_input_b_tiles` | 1 | `b.buffer_page_size()` (2,048 B) | `RING_DEPTH_BLOCKS × block_tiles × C` (8) | as above | b.dtype (bf16) | reader | compute | as above | whole program |
| `cb_output_tiles` | 2 | `out.buffer_page_size()` (2,048 B) | `RING_DEPTH_BLOCKS × block_tiles × C` (8) | as above | out.dtype (bf16) | compute | writer | compute: reserve/push per block (chain `PerBlockSize`); writer: implicit (tagged write per entry) | whole program |

Push count = wait count: reader posts one credit per page (ISR), compute waits `block_extent` per block, Σ block_extent = `neo_tiles` = pages the reader delivers to that NEO's slice; compute pushes `block_extent` per block, writer drains one per page, Σ = `neo_tiles`. No ring is opted out of implicit sync (`disable_dfb_implicit_sync_for` is empty).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_extent` pages × {a, b} | raw NoC (no dataflow helper covers it) | tensors a, b | `cb_input_a_tiles` +`block_extent`, `cb_input_b_tiles` +`block_extent` (credits posted per entry) | — |
| 2 | `add_block` | `block_extent` tiles, 1-D | yes: `compute_kernel_lib::add`, one call per NEO over `IterationShape::tiles(neo_tiles).block_size(block_tiles)` | `cb_input_a_tiles`, `cb_input_b_tiles`: wait `block_extent`, read by local index (`InputTileMapping::Block`), pop `block_extent` | `cb_output_tiles`: reserve `block_extent`, push `block_extent` | input rings freed per block |
| 3 | `store_block` | `block_extent` pages | raw NoC | `cb_output_tiles` (entry acked per page by the ISR) | tensor out | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (startup) | raw_api | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | `(dfb::cb_input_a_tiles, dfb::cb_input_b_tiles, dfb::cb_output_tiles)`, first statement of the compute kernel | — | — | none |
| `add_block` | helper | `compute_kernel_lib::add<AInput, BInput, Output>(IterationShape, a, b, out)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:45-52`, body `convenience.inl:8-14` (`BinaryFpu<Add>` + `PackTile`) | `AInput = input(dfb::cb_input_a_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block)`; `BInput = input(dfb::cb_input_b_tiles, …same…)` (`chain.hpp:385-391`; the default mapping is `Scalar`, so `Block` must be explicit); `Output = output(dfb::cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)` (`chain.hpp:442`); shape `IterationShape::tiles(neo_tiles).block_size(block_tiles)` (`chain.hpp:146-172`, tail `BlockTailSync::ValidTiles`). Policy legality: `chain.inl:29-37, 54-60` | `cb_input_a_tiles`, `cb_input_b_tiles` | `cb_output_tiles` | `block_tiles` (the `block_size` argument; DEST sectioning inside is the chain's) |
| `load_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(accessor, ring, {.page_id}, {})`; at end `noc_async_read_set_trid(0, noc.get_noc_id())` | `tt_metal/hw/inc/api/dataflow/noc.h:252`; `dataflow_api.h:2558` | accessors `tensor::a`, `tensor::b`; rings `dfb::cb_input_a_tiles`, `dfb::cb_input_b_tiles` | — | both input rings | none (per page) |
| `store_block` | raw_api | `Noc::async_write<NocOptions::TXN_ID>(ring, accessor, {}, {.page_id})`; at end `noc_async_write_set_trid(0, noc.get_noc_id())` | `noc.h:411`; `dataflow_api.h:2579` | accessor `tensor::out`; ring `dfb::cb_output_tiles` | `cb_output_tiles` | — | none |

**Helpers considered and rejected (raw_api rows):**

| Row | Candidate | File:Line | Why it cannot express the shape |
|-----|-----------|-----------|---------------------------------|
| startup | — | — | not a helper concern: the chain requires the caller to own it (`convenience.hpp:27-29`) |
| `load_block` / `store_block` | `dataflow_kernel_lib` DFB helpers | `ttnn/cpp/ttnn/kernel_lib/dfb_helpers_dataflow.hpp:14-18` | declares only `get_tile_r_dim` / `get_tile_c_dim`; no tensor-page ↔ ring transfer |
| `load_block` / `store_block` | local copy helpers | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:9` | L1 → L1 self-aimed copies only; not DRAM/L1-bank pages through a `TensorAccessor` |
| `load_block` / `store_block` | tile-column writer | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp` (header doc) | writes one column of a tile in place; not a page transfer |

## Broadcast Verification

| Phase | Op | CB_A (`cb_input_a_tiles`) Valid Region | CB_B (`cb_input_b_tiles`) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| Phase 0 `add_block` | FPU add | All (2-D `[H, W]` tile) | All (same shape) | `None` |
| deferred `over_w` | FPU add | All | Col0 (b is `[…, H, 1]`) | `Col` |
| deferred `over_h` | FPU add | All | Row0 (b is `[…, 1, W]`) | `Row` |
| deferred `over_hw` | FPU add | All | element [0,0] | `Scalar` |
| deferred `over_leading` | FPU add | All | All (b image reused across leading dims) | `None` (page remap only) |

Padding of non-aligned tiles is computed on and never compared (GUIDANCE: only logical elements are checked).

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Tag register left on a ring id | implicit-sync reader/writer leave a ring's transaction id; a following `ttnn.to_layout(out, ROW_MAJOR)` stops the simulator (`tile counter overflowed capacity`) | last statement of the reader `noc_async_read_set_trid(0, noc.get_noc_id())`, of the writer `noc_async_write_set_trid(0, …)`; session test (Work Distribution, test 2) |
| Explicit sync with untagged transfers | hangs after any implicit-sync program in the session | all rings implicit; no reserve/push/wait/pop in DM kernels; `disable_dfb_implicit_sync_for` empty |
| Default `InputTileMapping::Scalar` | `input(cb)` defaults to `Scalar` (`chain.hpp:385-391`): with `PerBlockSize` every step would read the block's first tile — wrong results, no error | pass `InputTileMapping::Block` explicitly on both inputs |
| Block quantum vs ring wrap | chain clamps `block_size` silently (`chain.inl:3343-3350`) | `static_assert(block_tiles <= DEST_AUTO_LIMIT)`; capacity = `depth × block_tiles` |
| Page count from volume | `[1,2,8400]`, `[1,2,400]` have H = 2: `volume/1024` undercounts 16× | `num_tiles = out.buffer_num_pages()`; tile size from `buffer_page_size()` |
| Math fidelity on ttsim | a compute kernel without explicit `fpu_math_fidelity` aborted the simulator (gaps file) | always set `fpu_math_fidelity=config.math_fidelity`; `toy_thread_plan` runs at HiFi4 on this base |
| bf16 unpacked to 32-bit DEST | `UnpackToDest` of a bf16 page with 32-bit DEST aborts ttsim | leave `unpack_modes` empty (default unpack through srcA/srcB) |
| Rounding | bf16 graded at 1 ulp (`NUMERIC_RESOLVED`) | bf16 operands are exact in srcA/srcB, the sum is formed in fp32 DEST and rounded once at pack |
| L1 operands | input pages live in other clusters' L1 banks; output must be L1 too (R9) | same `TensorAccessor` path; output from `allocate_tensor_on_device(a.spec, …)`; rings are separate allocations (49,152 B) |
| Tiny calls | 13 tiles → 13 clusters; clusters without work must not be launched with zero counts | `target_nodes` = clusters that got work from the split |
| Number operand | `b` may be a float: touching `.shape` / `.dtype` raises `AttributeError` before the refusal | `validate()` maps a number to shape `()` → `broadcast="number"` → refused before any tensor attribute of b is read |
| One dispatch | a host-side `ttnn.add` / `ttnn.to_layout` would be a second dispatch | exactly one `ttnn.generic_op` per call |

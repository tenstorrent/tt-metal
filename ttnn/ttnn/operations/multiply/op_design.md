# Operation Design: multiply

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (eltwise binary, FPU), bound by data movement |
| Goal | `out = a * b` with torch broadcasting, as ONE `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec` (reader / compute / writer) on every cluster of the grid. Phase 0 = PHASE0 of the spec: bf16 operands in TILE layout, DRAM or L1 interleaved, `broadcast` in {none, over_h}, any rank 2–4, any alignment. |
| Math | `out[i0,i1,h,w] = a[i0·sa0, i1·sa1, h·sah, w·saw] * b[i0·sb0, i1·sb1, h·sbh, w·sbw]` (`s·· = 0` on an axis where that operand has size 1, else 1), rounded once into the output dtype |
| References | spec `eval/specs/multiply/spec.py` (governs); target `.claude/references/targets/quasar.md`; backend `.claude/references/backends/ttsim.md`; `.claude/references/quasar-library-gaps.md`; worked example of this exact eltwise on Quasar: `.claude/references/quasar_op_examples/toy_thread_plan/` (host + kernels); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` (Part A, C4, C8); helper `ttnn/cpp/ttnn/kernel_lib/eltwise/api/{chain,convenience}.hpp`; catalog `ttnn/ttnn/operations/examples/master.md` (`width_split`, `double_buffer`, `compute_block_size`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (device query on this run: `GRID 8-4`) | target page; `device.compute_with_storage_grid_size()` |
| L1 per core | 4,194,304 B per cluster, shared by its 4 NEOs and DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` = 3,963,328 − 0 = 3,963,328 B on this target (queried on ttsim this run, `l1_reserve_bytes = 0`) | probe `tests/ttnn/unit_tests/operations/multiply/probes/probe_001.py` |
| Host program | `ttnn.ProgramSpec` (Metal 2.0) via `ttnn.generic_op` — `.claude/references/quasar_op_examples/` (README first; `toy_thread_plan/` is this op's skeleton) | target page |
| Threads per core | thread plan (R, C, W): C ∈ {1,2,4} NEOs, R + W ≤ 6 user DM cores, R and W each divide C (STRIDED rings); ring ≤ 1,048,560 B; ≤ 24 active DFBs per NEO; 16 semaphores; DM binary ≤ 48 KiB, compute ≤ 24 KiB | target page *Cluster model / Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` and the legacy C++ device op (Metal 1 kernels refused, `kernel.hpp` TT_FATAL); `bfloat8_b` / `bfloat4_b` (no path on this family); the whole library-gaps list, of which these touch this op: `UnpackToDest` of a bf16 page into 32-bit DEST aborts ttsim (so the bf16 path keeps the default unpack into SrcA/SrcB — the FPU path planned here); an untagged NoC transfer after an implicit-sync program hangs (every transfer is tagged, the tag is handed back at kernel end); `fpu_math_fidelity` must be set on every compute kernel on ttsim; compute-kernel `DEVICE_PRINT` prints nothing; `matmul_block`, `ReduceFp32Mode::Accurate`, `AccumulateViaAdd`, `minmax.hpp` uint32 (not used here). Header limits: the FPU broadcasts srcB only (`chain.hpp:354-357`), and on Quasar `bcast_row_idx` must be 0 (`bcast.h:276-281`). | target page, `quasar-library-gaps.md`, headers cited |

All worked numbers below are for this target (32 clusters, 3,963,328 B budget, bf16 tile = 2,048 B). Every predicate the op evaluates queries the device.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor_a` | `ttnn.Tensor` | yes | Phase 0: bf16, TILE, DRAM or L1 interleaved, rank 2–4 | — | tensor binding |
| `input_tensor_b` | `ttnn.Tensor` | yes | same dtype / layout / placement as a; shape broadcastable with a (else X1: `ValueError`) | — | tensor binding |
| `dtype` (kw-only) | `ttnn.DataType` | no | Phase 0: `None` or `ttnn.bfloat16` | `None` → operands' dtype | sets output ring format (host) |
| `compute_kernel_config` (kw-only) | `ttnn.ComputeConfigDescriptor` | no | Phase 0: `fp32_dest_acc_en=True`; `math_fidelity` / `math_approx_mode` any value (R4) | `None` → `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False` (R3, the only place the default is written) | compute `ComputeHardwareConfig(fpu_math_fidelity=cfg.math_fidelity, enable_32_bit_dest=cfg.fp32_dest_acc_en)` |

Notes for `validate()` (the implementer writes it): axes are derived exactly as the harness does — `INPUT_TAGGERS = AXES` imported from the spec; `memory_config` / `sharding` from `eval.memory.memory_axes` of each operand (R10; both operands must give the same pair, else `NotImplementedError`); `output_dtype` = the `dtype` argument; `fp32_dest_acc_en` from the resolved config. R1/R2 refusals (`fp32_dest_acc_en=False` with float32 operands or `dtype=float32`) are checked even though Phase 0 refuses those cells anyway by SUPPORTED. The X1 broadcast check (`torch.broadcast_shapes` semantics) runs before `validate()` and raises `ValueError`. No index axes (no `dim`) — nothing to canonicalize.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | a, b of rank 2–4, broadcastable; Phase 0 broadcast class (spec `tag_broadcast`) ∈ {`none`, `over_h`} — `over_h` includes a b that also has size 1 / lower rank in leading dims (the tagger only looks at H/W) |
| Dtype | bfloat16 (both) |
| Layout | TILE (both) |
| Memory | interleaved, both in DRAM or both in L1 (R7); read through `TensorAccessor` either way |

### Output

| Property | Value |
|----------|-------|
| Shape | `torch.broadcast_shapes(a.shape, b.shape)` — may have a higher rank than a (e.g. a `[32,64]` × b `[1,1,32,64]` → `[1,1,32,64]`) |
| Dtype | `dtype` if given, else a's dtype (Phase 0: bfloat16) |
| Layout | TILE (a's layout) |
| Memory | interleaved, buffer type of a (R9): `ttnn.allocate_tensor_on_device(TensorSpec(out_shape, out_dtype, TILE, MemoryConfig(INTERLEAVED, a.memory_config().buffer_type)), device)` |

## Blocking Model

Index space: the output's tile grid, viewed 3-D as `[N_out, Ht, Wt]` with `N_out` = product of the output's leading dims (1 for rank 2), `Ht = ceil(H_out/32)`, `Wt = ceil(W_out/32)`, each image tile-padded on its own (TILE layout pads every image independently). Tile page of `(n, r, c)` in a tensor X of shape `[Nx, Htx, Wtx]` = `img_X(n)·Htx·Wtx + (Htx==1 ? 0 : r)·Wtx + (Wtx==1 ? 0 : c)`, with `img_X(n) = d0·sX0 + d1·sX1` for `n = d0·o1 + d1` (rank padded to 4; `sX0 = X.d0==1 ? 0 : X.d1`, `sX1 = X.d1==1 ? 0 : 1`). The host computes the strides once; Phase 0 passes them for b (identity for a).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `n` (output image = folded leading dims) | independent — no output element depends on another image; **reuse-shared for b** only under leading-dim broadcast of b (b's image does not vary with n) | `block_extent[n]` | 1 image per block — a block never spans images because b's held window is per image (`row_bcast_strips`); in `flat_stream` the n axis is folded into the 1-D tile run, no extent of its own | structural (block walker cuts at image boundaries) | split across cores as part of the unit index (`flat_stream`: tile index; `row_bcast_strips`: unit `(n, g, c)`) | spreading is already done; reuse across n for b = `leading_reuse` row (scheme-change: mcast) |
| `h` (tile rows) | independent for a/out; **reuse-shared for b** in `over_h` (b has one tile row; it does not vary with h) | `block_rows` = rows of the block's row group | `flat_stream`: folded (1-D run). `row_bcast_strips`: `rows(g) = floor((g+1)·Ht/G) − floor(g·Ht/G)` — the whole H extent of the image when `G = 1` | derived on host from `row_groups` G; recomputed identically in each kernel by the shared block walker | NOT split across cores unless the (n, c) units under-fill the grid: `row_groups G = clamp(ceil(num_cores / (N_out·Wt)), 1, Ht)` (`flat_stream`: G = Ht, i.e. fully split, nothing to reuse) | G is a knob: raising it trades b re-reads for occupancy (knob-turn) |
| `w` (tile columns) | independent; b varies along it (not reuse-shared in `over_h`) | `block_cols` | `flat_stream`: folded. `row_bcast_strips`: `min(Wt, max_units_per_core, block_cols_cap)`, `block_cols_cap = floor(min(DFB_RING_MAX_BYTES, budget − 3·stream_depth·tile_bytes) / (b_window_depth · tile_bytes))` = 255 on this target | ONE host derivation → CT arg to reader, compute, writer and the b-ring capacity | the primary core split in `row_bcast_strips` (cut the axis b varies along, so cores read disjoint b tiles — catalog `width_split`) | knob-turn |
| `flat` (the `flat_stream` 1-D tile run: n·h·w in page order) | independent — a, b, out walk identical page ids | `core_tiles` (whole per-core assignment = one block) | `num_tiles_this_core` from `ttnn.split_work_to_cores(grid, N_out·Ht·Wt, row_wise=True)` | host split → RT arg | contiguous runs over all clusters | — |
| in-tile row / col (32×32) | independent — not a block axis; FPU processes whole tiles; padding computed and discarded | — | — | — | — | — |

The one-dispatch rule: both regimes are one `ProgramSpec`, chosen on host; the regime is a CT arg of the three kernels.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_in_a_tiles` | `stream_depth` (entries per NEO) | 4 | 4 reads in flight per ring; implicit sync splits it into 2 txn-id halves of 2 (DFB.md C4: smallest n ≥ 2 dividing 4), so the reader issues one half while DM0 credits the other |
| `cb_in_b_tiles` (`flat_stream`) | `stream_depth` | 4 | as a |
| `cb_in_b_window` (`row_bcast_strips`) | `b_window_depth` (windows of `block_cols`) | 2 | the reader loads block k+1's b window while compute holds block k's; ≥ 2 is also a **liveness** requirement (Mechanism caps) |
| `cb_out_tiles` | `stream_depth` | 4 | compute packs while the writer drains the other half |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DFB ring size (target page: ≤ 1,048,560 B per ring) | `block_cols` | `block_cols_cap = floor(1,048,560 / (b_window_depth·tile_bytes))` (255 bf16 tiles at depth 2), also ≤ budget term | program build refused |
| Implicit-sync credit batching (DFB.md C4: credits post once a txn-id batch of `num_entries / num_txn_ids` completes) | b window size per block | b ring is **explicit sync** with **nominal** windows: reserve/push/wait/pop exactly `block_cols` entries per block, only `cols ≤ block_cols` of them are filled/used | an implicit-sync b ring with variable windows leaves a short window uncredited until the next block's reads → compute waits on b while the reader waits on a-ring space: hang |
| Indexed ring access does not wrap (`llk_unpack_AB_api.h:99-102`: `cursor_entry_idx + tile_index·stride`, no modulo; gaps file: "do not let a window straddle the end of a ring") | b window placement | ring capacity = `b_window_depth · block_cols` and every window is exactly `block_cols` → windows tile the ring and never straddle | silently wrong b tiles for the straddling block |
| `b_window_depth ≥ 2` (liveness) | b ring depth | host asserts `b_window_depth >= 2` | with 1 window the reader blocks reserving block k+1's window before issuing a reads that complete block k's last implicit-sync a batch → deadlock |
| FPU broadcast is srcB-only (`chain.hpp:354-357`) | which operand may be expanded inside a tile | host puts the expanded operand on srcB (Phase 0: b is always the expanded one); a call where both operands need an in-tile broadcast is refused | wrong result / no expressible chain |
| Quasar ROW bcast reads row 0 only (`bcast.h:276-281`, `bcast_row_idx == 0`) | b's valid row | b's valid row is its tile row 0 (b has H = 1) — always true here | — |
| Thread plan rules (R, W divide C; R + W ≤ 6) | plan | `THREAD_PLAN = (1,1,1)`; the host asserts C = 1 for `row_bcast_strips` (its unit dealt is a block, see Work Distribution) | a C > 1 plan would deal b-window entries round-robin across NEOs: wrong b tiles |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (boundary: DRAM-or-L1 interleaved tensor ↔ cluster) | What a bigger block buys |
|--------|--------|-----------|-------|-----------------------------------------------|--------------------------|
| `flat_stream` | **built** | `tag_broadcast == "none"` (shapes equal after left-padding with 1s) | the core's whole run: `core_tiles` tiles, 1-D | **minimum**: a, b, out each cross once; page ids identical for all three | one helper call per core (one init + reconfig per kernel), one ring handshake per tile is fixed by implicit sync; at the DRAM floor a bigger compute block buys nothing measurable (README: blocks of 4/8 = same cycles) |
| `row_bcast_strips` | **built** | `tag_broadcast == "over_h"` (b: H = 1, W equal; b leading dims equal to out's or 1) | `block_rows × cols`, `cols ≤ block_cols`, inside one image and one row group | a and out cross once; b crosses `G` times (`G = 1` whenever `N_out·Wt ≥ num_cores`, i.e. every INPUTS over_h entry except `[128,512]×[1,512]` (G = 2: +16 tile reads)); under leading-dim broadcast of b, b crosses `G · N_out/N_b` times (Phase 0 has no INPUTS entry of that kind) | each block pays: b window load + one explicit barrier + one helper call (bcast init + reconfig). A taller block (`block_rows`, i.e. smaller G) cuts b re-reads; a wider block amortizes the per-block init over more tiles |
| `h_split_shared_b` (flat tile split for over_h, b re-read per tile row) | **rejected** — superseded by `row_bcast_strips` | `over_h` | 1 tile row × cols | b crosses `Ht` times per image (e.g. `[2,4,128,512]×[2,4,1,512]`: 512 b tile reads instead of 128 → total traffic 1,536 vs 1,152 tiles, +33%) | — (a dead end: no better regime is built on it) |
| `col_bcast_strips` (over_w) | **deferred** — not in PHASE0; positive reason: the spec's first phase covers only the traced YOLOv8 calls (over_h) and equal shapes; reachable as a knob-turn | `over_w` | `rows × block_cols` with roles transposed: units `(n, h, wgroup)`, b window = the block's column of `rows` tiles | minimum (b one tile per row, read once per block) | same as `row_bcast_strips` — same walker, b mapping `Col`, `BroadcastDim::Col`, window length = rows |
| `scalar_bcast` (over_hw) | **deferred** — same positive reason; knob-turn | `over_hw` | core's run inside one image; b window = 1 tile | b crosses once per (core, image) run ≤ `N_out + num_cores` tiles | b `Scalar` mapping + `BroadcastDim::Scalar`, window 1 |
| `leading_reuse` (over_leading; and leading-broadcast b inside over_h) | **deferred** — positive reason: built `flat_stream`/`row_bcast_strips` already reach every such shape with the per-operand image strides (b re-read `N_out/N_b` times, a stepping stone); the reuse successor is a multicast of b's image to the cores sharing it, a scheme-change with no INPUTS entry where b is large relative to a | `over_leading` | as `flat_stream` with b page = `img_b(n)`-mapped | b crosses `N_out/N_b` times (e.g. `[4,8,32,256]×[1,1,32,256]`: 32×) | the mcast successor replaces only `load_b_window` (Dataflow Strategy) |
| `operand_swap` (first_operand, one operand expanded) | **deferred** — not in PHASE0; knob-turn | `first_operand` and only a is expanded | as the regime the swapped call lands in | as that regime | host swaps a ↔ b (FPU multiply is commutative, exact); R5 |
| `dual_in_tile_bcast` (both operands expanded inside a tile, e.g. `[32,1]×[1,64]`) | **deferred** — positive reason: no INPUTS entry reaches it, and the FPU broadcasts srcB only; needs a compute scheme-change (unary-broadcast one operand into DEST, then a DEST-reuse multiply with the other) | a and b both need in-tile broadcast | — | — | — |
| `fp32_sfpu` (float32 operands) | **deferred** — positive reason: dtype float32 is outside PHASE0; SrcA/SrcB are 19 bits, so 0-ulp float32 needs the SFPU path (`binary_sfpu<MulBinary>` on fp32 `UnpackToDest` pages, or `sfpu_mul_bcast_row`); same block walker | `dtype == float32` | same blocks | same traffic (×2 bytes) | — |
| `row_major` | **deferred** — positive reason: ROW_MAJOR is outside PHASE0; needs an in-kernel tilize/untilize stage per tile-row (`tilize_helpers.hpp`/`untilize_helpers.hpp`) or a stick path | `layout == ROW_MAJOR` | tile-row blocks | sticks instead of tiles | — |

**Regime-selection function** (host, exact): `regime = "flat_stream" if tag_broadcast(shapes) == "none" else "row_bcast_strips" if tag_broadcast(shapes) == "over_h" else refuse` (Phase 0 `validate()` already refuses every other value). Regime-pinned tests: the acceptance test runs `((32,17),(1,17))` (`row_bcast_strips`) and equal-shape entries (`flat_stream`); the implementer adds one test per regime that asserts which regime the host chose (read off the program artifacts), on a multi-block shape for `row_bcast_strips` (`[2,4,128,512]×[2,4,1,512]`, 4×4 blocks) and on a G > 1 shape (`[128,512]×[1,512]`).

### Traffic ranking

Counted in tile transfers across the DRAM/L1-tensor ↔ cluster boundary (cross-core traffic: none in any built regime).

| Candidate split (over_h) | a | b | out | Cross-core | Rank |
|--------------------------|---|---|-----|-----------|------|
| cut (n, w) across cores, keep h whole per core (`row_bcast_strips`, G = 1) | 1× | 1× | 1× | 0 | 1 (= minimum) — **chosen** |
| same with G row groups when `N_out·Wt < num_cores` | 1× | G× (≤ `ceil(num_cores/(N_out·Wt))`, so G·|b| < num_cores + |b| tiles) | 1× | 0 | 2 — chosen only when occupancy needs it |
| cut h across cores (flat page split) | 1× | Ht× | 1× | 0 | 3 — rejected |
| read b once per G>1 group and multicast to the G cores sharing it | 1× | 1× | 1× | (G−1)·|b_slice| mcast | deferred (the saving is < num_cores tiles per call) |
| dependent-axis split | — | — | — | — | n/a: multiply has no dependent axis |

`flat_stream`: every split moves the minimum (a, b, out once); the flat page split is chosen for balance (±1 tile per core) and because it is the proven toy pattern.

### Block schedule

```cpp
// flat_stream — one block per core
load_stream(core_tiles);            // reader: a[i], b[i] per tile, implicit sync
multiply_block(core_tiles);         // compute: one mul helper call over the run
store_stream(core_tiles);           // writer: out[i] per tile, implicit sync

// row_bcast_strips
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // blocks from the shared walker
    load_b_window(block_idx);       // reader: cols b tiles into a block_cols window, explicit sync, one barrier
    load_a_block(block_idx);        // reader: block_rows×cols a tiles row-major, implicit sync
    multiply_bcast_block(block_idx);// compute: wait window, one mul helper call grid(block_rows, cols), pop window
    store_block(block_idx);         // writer: block_rows×cols out tiles row-major, implicit sync
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_stream` | `core_tiles` tiles, page ids `start .. start+core_tiles` | nothing | one tagged read per tile per operand; no barriers; one tag hand-back at kernel end |
| `multiply_block` | 1-D, `core_tiles` | nothing (a, b, out stream through 4-entry rings) | `compute_kernel_hw_startup` once; helper init + reconfig once per kernel (one call) |
| `store_stream` | `core_tiles` | — | one tagged write per tile; tag hand-back at kernel end |
| `load_b_window` | `cols` tiles of one b image row (`block_cols` nominal) | the window stays resident for the whole block | one `reserve_back(block_cols)`, `cols` user-trid reads, one barrier, one `push_back(block_cols)` per block |
| `load_a_block` | `block_rows × cols`, row-major | — | one tagged read per tile, no barrier |
| `multiply_bcast_block` | `grid(block_rows, cols)`; a streamed per tile (front tile), b indexed by column in the window | b window | one helper call per block (bcast init + reconfig per block); one `wait_front`/`pop_front(block_cols)` on the b ring per block |
| `store_block` | `block_rows × cols`, row-major | — | one tagged write per tile; tag hand-back at kernel end |

`num_blocks_this_core` is small by construction: a core's unit run `[u0, u0+cnt)` over units `(n, g, c)` (c fastest) splits only at (n, g) boundaries and every `block_cols` columns, so ≤ `ceil(cnt / block_cols) + (number of (n,g) boundaries crossed)`; with Phase 0's `block_cols ≥ max_units_per_core` it is 1 per core on every INPUTS over_h entry except shapes whose image row is shorter than a core's run.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Thread plan (1,1,1) | Chosen from the bound (data movement). `toy_thread_plan` — this very multiply, bf16, 3,200 tiles on 32 clusters — measured 84,562 / 84,542 / 85,938 model cycles at (1,1,1) / (2,2,2) / (4,4,2): on the floor already | (2,2,2) and (4,4,2) on `flat_stream` on a timed backend (ttsim-perf); the kernels of `flat_stream` deal per tile and run under any legal plan by editing `THREAD_PLAN` |
| `stream_depth = 4` | README: depth/blocks of 4–8 did not move the floor at (1,1,1); a 2-entry ring is what the toy measured | 2 and 8 entries |
| Overlap (`row_bcast_strips`): compute waits for the whole b window before the block's first tile | with one block per core the first output waits for `cols` b reads | `block_cols` halved (two blocks per core) vs the default |
| Grid synchronization / occupancy (`row_bcast_strips`, G) | G = 1 leaves cores idle when `N_out·Wt < num_cores` and `Ht = 1` (e.g. `[1,4,400]×[1,1,400]`: 13 tiles → 13 clusters, nothing more to split) and G > 1 re-reads b | G one step lower (fewer cores, b read fewer times) on `[128,512]×[1,512]` |
| Per-block helper init (`row_bcast_strips` with many small blocks: small Wt, many images) | the chain re-inits bcast + reconfig per call | hoist with `eltwise_chain<InitReconfigOwner::Caller>` (`chain.hpp:179-191`) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| a tensor → `cb_in_a_tiles` | bf16 tiles, 2,048 B pages | reader, `noc.async_read<NocOptions::TXN_ID>(acc_a, ring, {.page_id}, {})`, **implicit sync** | DRAM and L1 interleaved read the same way (accessor); no sharded path in TARGET |
| b tensor → `cb_in_b_tiles` (`flat_stream`) | bf16 tiles | reader, implicit sync, same page id as a | interleaved with a's read per tile (toy `reader_implicit.cpp:27-30`) |
| b tensor → `cb_in_b_window` (`row_bcast_strips`) | bf16 tiles | reader, **explicit sync**: `reserve_back(block_cols)`, `cols` reads with `NocOptions::TXN_ID` + `{.trid = 0}` at `offset_bytes = j·tile_bytes`, `async_read_barrier<TXN_ID>({.trid=0})`, `push_back(block_cols)` | ring listed in the reader's `disable_dfb_implicit_sync_for` (host); pattern `reader_mixed.cpp:30-35` |
| rings → DEST → `cb_out_tiles` | SrcA/SrcB bf16 → DEST fp32 (`fp32_dest_acc_en=True`) → packed bf16 | compute: FPU `mul` (HiFi4 at default) | the product of two bf16 values is exact in fp32 DEST; one rounding at pack |
| `cb_out_tiles` → out tensor | bf16 tiles | writer, `noc.async_write<NocOptions::TXN_ID>(ring, acc_out, {}, {.page_id})`, implicit sync | out page id = output tile `(n, r, c)` |
| kernel end | — | reader: `noc_async_read_set_trid(0, noc.get_noc_id())`; writer: `noc_async_write_set_trid(0, ...)` as the last statement | the reader's last transfer is a ring-tagged a read in both regimes, so it must hand the tag back |

TARGET's placement axis has only `interleaved` (DRAM or L1): no physical shard exists to consume in place. The `leading_reuse` successor contract (deferred): a designated cluster per b image reads `load_b_window` once and multicasts it (`ttnn.Mcast`, `McastArgs` sender/receiver) into the same `cb_in_b_window` on the receivers; the receivers' compute is unchanged because the window protocol (explicit, nominal `block_cols`) is already the receive side's.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | `flat_stream`: one output tile (page id). `row_bcast_strips`: one column strip `(n, g, c)` = `rows(g)` tiles of column c in row group g of image n; unit index `u = (n·G + g)·Wt + c` |
| Grid | `device.compute_with_storage_grid_size()` (8×4 here); `ttnn.split_work_to_cores(grid, num_units, row_wise=True)` → per-cluster `(start_unit, num_units)` as RT args (toy `toy_thread_plan_program_artifacts.py:50`) |
| Per-core work | `flat_stream`: `ceil` / `floor` of `N_out·Ht·Wt / num_cores`. `row_bcast_strips`: `G = clamp(ceil(num_cores / (N_out·Wt)), 1, Ht)`, units `N_out·G·Wt`; worked: `[1,4,8400]×[1,1,8400]` Wt = 263, G = 1 → 7 clusters × 9 + 25 × 8 units, one 1×9 / 1×8 block each; `[1,4,33600]` Wt = 1050 → 26 × 33 + 6 × 32; `[1,4,400]` → 13 clusters × 1; `[2,4,128,512]×[2,4,1,512]` → 32 × 4 units, one 4×4 block each; `[128,512]×[1,512]` → G = 2, 32 × 1 unit (2×1 block) |
| Remainder | `Ht = ceil(H/32)`, `Wt = ceil(W/32)` per image; `N_out` from the output shape; group rows `floor((g+1)·Ht/G) − floor(g·Ht/G)`; the last core of a split takes the short count via `split_work_to_cores`' second group — the kernels take the count, never a stride over elements |
| Bound | **data movement**: one FPU multiply per output tile against three 2 KB tile transfers per tile (two reads, one write); measured for this exact op in `toy_thread_plan` (plans within 1.6% of each other at the DRAM floor) |
| Thread plan | **(1, 1, 1)**, host constant `THREAD_PLAN = (1, 1, 1)` passed as CT args `reader_threads`, `compute_threads`, `writer_threads`; `flat_stream` kernels deal per tile (reader stride R, compute `my_tiles = n/C + (t < n%C)`, writer stride W — toy `compute.cpp:29`), `row_bcast_strips` kernels require C = R = W = 1 (host asserts) |
| Unit dealt to compute threads | `flat_stream`: the ring entry (one tile) — any plan works, no padding. `row_bcast_strips`: a whole block (b window + its a tiles); at C = 1 this is trivial; a C > 1 plan would need blocks dealt in padded rounds of C and whole rounds per cluster (`toy_row_passes`) — not built (lamp: bound is data movement) |
| Tests the plan needs | (1) the host program carries `num_threads` = `THREAD_PLAN` on each kernel and the per-kernel work counts sum to the output tile count (a missing thread = missing work); (2) in ONE device session: another program (e.g. `ttnn.experimental` op or `ttnn.to_layout`) then `multiply`, and `multiply` then `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` — for both regimes (the `row_bcast_strips` reader mixes explicit and implicit sync) |

## Circular Buffers

Dataflow buffers (DFB rings), all STRIDED, page = one tile of the ring's format; capacities counted per NEO × C (C = 1).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_in_a_tiles` | 0 | a tile (2,048 B bf16) | `stream_depth · C` = 4 | streams over n, h, w (and the flat run); spans nothing | bf16 | reader | compute | reader: implicit; compute: helper `PerTile` wait/pop | whole kernel |
| `cb_in_b_tiles` (`flat_stream`) | 1 | b tile | `stream_depth · C` = 4 | streams over the flat run | bf16 | reader | compute | reader: implicit; compute: helper `PerTile` | whole kernel |
| `cb_in_b_window` (`row_bcast_strips`; same slot 1 — one regime per program) | 1 | b tile | `b_window_depth · block_cols · C` = 2·block_cols | spans w (`block_cols`), streams over h, n | bf16 | reader | compute | reader: **explicit** (reserve/push `block_cols`, user trid 0, listed in `disable_dfb_implicit_sync_for`); compute: caller `wait_front` / `pop_front(block_cols)` around the helper call | one window per block |
| `cb_out_tiles` | 2 | out tile (2,048 B bf16) | `stream_depth · C` = 4 | streams | out dtype (bf16 in Phase 0) | compute | writer | compute: helper `PerTile` reserve/push; writer: implicit | whole kernel |

Full ledger with axis accounting and the traffic budget: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_stream` | `core_tiles` | no (dataflow) | — | `cb_in_a_tiles`, `cb_in_b_tiles` (1 per tile each) | — |
| 2 | `multiply_block` | `tiles(core_tiles)` | yes `mul` | `cb_in_a_tiles` 1/tile popped; `cb_in_b_tiles` 1/tile popped | `cb_out_tiles` 1/tile | rings empty at end |
| 3 | `store_stream` | `core_tiles` | no | `cb_out_tiles` | — | — |
| 4 | `load_b_window` | `cols` (nominal `block_cols`) | no | — | `cb_in_b_window` `block_cols` pushed | window resident |
| 5 | `load_a_block` | `block_rows × cols` | no | — | `cb_in_a_tiles` 1/tile | — |
| 6 | `multiply_bcast_block` | `grid(block_rows, cols)` | yes `mul` | `cb_in_a_tiles` 1/tile popped; `cb_in_b_window` waited `block_cols`, read at index c, popped `block_cols` after the call | `cb_out_tiles` 1/tile | window released |
| 7 | `store_block` | `block_rows × cols` | no | `cb_out_tiles` | — | — |

Block walker: one header in the op's kernel directory, included by all three kernels, turns `(start_unit, num_units, N_out, G, Ht, Wt, block_cols)` into the block sequence `(n, row_begin, block_rows, col_begin, cols)`; it is the single source of the decomposition (reader, compute and writer must agree on every block).

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (boot) | helper prereq | `compute_kernel_hw_startup(in_a, in_b, out)` | toy `compute.cpp:34` | the three ring ids, first statement | — | — | — |
| `multiply_block` | helper | `compute_kernel_lib::mul` → `eltwise_chain(BinaryFpu<Mul>, PackTile)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:63-70`, `convenience.inl:24-30` | `mul<input(dfb::in_a, PerTile, PerTile, Scalar), input(dfb::in_b, PerTile, PerTile, Scalar), output(dfb::out)>(IterationShape::tiles(my_tiles), a, b, out)` — exactly toy `compute.cpp:37-40`; Scalar+PerTile is the legal stream form (`chain.inl:69-70`; Block mapping refuses PerTile, `chain.inl:58-61`) | `cb_in_a_tiles`, `cb_in_b_tiles` | `cb_out_tiles` | `my_tiles` (= `core_tiles` / C) |
| `multiply_bcast_block` | helper | same `mul`, srcB `BroadcastInputSpec` | `chain.hpp:408-436` (`input(id, BroadcastDim, Wait, Pop, Mapping)`), `chain.hpp:247` (Row mapping = index by column), `chain.hpp:330-335` (`BroadcastDim::Row`: b row 0 to every output row), `chain.inl:62-64` (Row legal with `None/None`), `chain.inl:1391-1420,1463-1469` (bcast init / `mul_tiles_bcast<ROW>`) | `mul<input(dfb::in_a, PerTile, PerTile, Scalar), input(dfb::in_b, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row), output(dfb::out)>(IterationShape::grid(block_rows, cols), a, b, out)`, bracketed by `b.wait_front(block_cols)` / `b.pop_front(block_cols)` (real unpacks happen between them, so TEN-4746, DFB.md C8, does not apply) | `cb_in_a_tiles`, `cb_in_b_window` | `cb_out_tiles` | `block_rows`, `cols` (≤ `block_cols`) |
| `load_stream`, `load_a_block` | raw_api (dataflow) | `noc.async_read<NocOptions::TXN_ID>(accessor, ring, {.page_id}, {})` | toy `reader_implicit.cpp:28-29`, tag hand-back `:36` | one call per tile | — | `cb_in_a_tiles` (`cb_in_b_tiles`) | — |
| `load_b_window` | raw_api (dataflow) | `reserve_back` / `noc.async_read<TXN_ID>(acc_b, ring, tile_bytes, {.page_id}, {.offset_bytes = j·tile_bytes}, {.trid = 0})` / `async_read_barrier<TXN_ID>({.trid=0})` / `push_back` | toy `reader_mixed.cpp:30-35`; host opt-out `toy_thread_plan_program_artifacts.py:68` | per block | — | `cb_in_b_window` | `block_cols`, `cols` |
| `store_stream`, `store_block` | raw_api (dataflow) | `noc.async_write<NocOptions::TXN_ID>(ring, acc_out, {}, {.page_id})` | toy `writer_implicit.cpp:26`, hand-back `:30` | one call per tile | `cb_out_tiles` | — | — |
| host | — | `ttnn.experimental.{ProgramSpec, KernelSpec, DataflowBufferSpec, TensorParameter, WorkUnitSpec, ProgramRunArgs}`, `ttnn.generic_op` | toy `toy_thread_plan_program_artifacts.py`, `toy_thread_plan.py:24-25` | compute `hw_config=E.ComputeHardwareConfig(fpu_math_fidelity=cfg.math_fidelity, enable_32_bit_dest=cfg.fp32_dest_acc_en)` (`ttnn/cpp/ttnn-nanobind/program_spec.cpp:516-545`) — fidelity always set (gaps: ttsim) | — | — | `THREAD_PLAN`, `stream_depth`, `b_window_depth`, `block_cols`, `G` |

Helpers considered and rejected for the dataflow `raw_api` rows: none applies — the kernel library's dataflow side offers `mcast` (not needed: no cross-core traffic) and `reduce` scaler prep (no reduce); the Quasar DM NoC API is the documented mechanism (DataflowBuffer.md Part A).

## Broadcast Verification

| Phase | Op | CB_A (`cb_in_a_tiles`) Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-------------------|---------------|
| `multiply_block` (`none`) | FPU mul | All | All (`cb_in_b_tiles`) | None |
| `multiply_bcast_block` (`over_h`) | FPU mul | All | Row0 (`cb_in_b_window`: b has H = 1, its tile's row 0 is the data, rows 1–31 padding) | Row (`BroadcastDim::Row` = `BroadcastType::ROW`: srcB row 0 copied to all 32 rows) |
| deferred `over_w` | FPU mul | All | Col0 | Col |
| deferred `over_hw` | FPU mul | All | [0,0] | Scalar |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| 0-ulp grading of bf16 | NUMERIC requires 0 ulp: the product is exact in fp32 DEST at HiFi4, so the only rounding is the packer's fp32 → bf16; a truncating packer would be off by 1 ulp on about half the elements | keep `fp32_dest_acc_en=True` + HiFi4 default (spec CALL fixes both); run the acceptance test first; if pack rounding is the cause, that is a finding about the target to report, not something to tune around in-op |
| Row broadcast on Quasar not exercised by the pipeline's examples | `toy_*` only use `BroadcastDim::None`; the Quasar LLK path (`llk_unpack_AB<ROW>`, `_llk_math_eltwise_binary_broadcast_init_`) exists | acceptance cell `((32,17),(1,17))` hits it on the smallest shape; add a test with distinct b values per column and non-constant a rows |
| Mixed protocol in one reader (`row_bcast_strips`) | explicit b ring + implicit a ring on one hart; an untagged transfer reuses a ring's tag | every b read carries `{.trid = 0}`; the kernel ends on an a ring read, so it hands the tag back; session-order tests from Work Distribution |
| Short blocks vs ring windows | a core's run can end mid-image row (cols < block_cols); variable windows would straddle the ring end and break batch alignment | nominal `block_cols` reserve/push/wait/pop, only `cols` filled/used (Mechanism caps) |
| `over_h` includes b with leading-dim broadcast | the spec's tagger only compares H and W | reader maps b's image through `(sb0, sb1)` strides from day 1; the host computes them from the 4-D padded shapes |
| Output rank above a's | `none` with a lower-rank a (`[32,64]` × `[1,1,32,64]`) | output shape = `torch.broadcast_shapes`; page count equal, so `flat_stream` is unchanged |
| `N_out·Ht·Wt` small (e.g. 1 tile) | fewer units than clusters | `split_work_to_cores` engages only the clusters with work; the `WorkUnitSpec` targets that core set (toy) |
| L1-placed operands | DFB rings and L1 tensors share each cluster's L1 | rings are small (≤ 24 KB `flat_stream`; ≤ 16 KB + 2·block_cols tiles otherwise; the callers' L1 tensors are a few hundred tiles); `l1_reserve_bytes` is the knob if a refinement needs room |

## Structural impossibilities

None to add: `INVALID = []` holds; bfloat8_b / bfloat4_b are not in TARGET.

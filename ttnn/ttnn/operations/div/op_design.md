# Operation Design: div

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (elementwise binary, SFPU) |
| Goal | `torch.div(a, b, rounding_mode=...)` on Quasar as one `ttnn.ProgramSpec` dispatch: the quotient, its floor or its trunc, computed and rounded in DEST in one fused pass per tile, multi-core and multi-NEO from Phase 0. |
| Math | `out[i] = a[i] / b[i]` (b a tensor) or `a[i] / b` (b a Python number); with `rounding_mode="floor"` `floor(a/b)`, with `"trunc"` `trunc(a/b)`; output dtype = `dtype` or `a.dtype` |
| References | Specification `eval/specs/div/spec.py` (authoritative; RULES R1–R14, REJECTIONS X1–X2, NUMERIC); target `.claude/references/targets/quasar.md`; backend `.claude/references/backends/ttsim.md`; host-program pattern `.claude/references/quasar_op_examples/toy_thread_plan/` (the structural template for this op: binary eltwise, STRIDED rings, implicit sync, plan (R,C,W)); `blocking-model.md`; `l1-footprint-discipline.md`; perf catalog `ttnn/ttnn/operations/examples/master.md` (entries `double_buffer`, `compute_fusion`, `compute_block_size`, `noc_placement`, `sfpu_tile_scope`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (worked numbers); op queries `device.compute_with_storage_grid_size()` | target page |
| L1 per core | 4,194,304 B (4 MiB) per cluster, shared by its 4 NEOs and DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` (`L1_RESERVE_BYTES` = 0 in Phase 0); value on Quasar: not recorded in the profile, < 4 MiB; the Phase 0 footprint (≤ 256 KiB, `l1_ledger.md`) is far below it, so no predicate is evaluated in Phase 0 | target page (query); value `unknown` |
| Host program | `ttnn.ProgramSpec` (Metal 2.0: `KernelSpec`, `DataflowBufferSpec`, `WorkUnitSpec`, `ttnn.generic_op`) — reference `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/`) | target page |
| Threads per core | 4 NEOs (C ∈ {1,2,4}); 6 user DM harts (R + W ≤ 6); on STRIDED rings R \| C and W \| C; ≤ 24 active DFBs per NEO; ring ≤ 1,048,560 B; compute binary ≤ 24 KiB, DM binary ≤ 48 KiB | target page, *Cluster model* / *Thread plan* |
| Unavailable here | (1) `ttnn.ProgramDescriptor` / Metal 1 kernels / C++ program-factory device op — refused at kernel creation (target page). (2) `bfloat8_b` / `bfloat4_b` — no path on Quasar (target page). (3) `DivUnary` / `RdivUnary` scalar SFPU ops — Quasar `calculate_binop_with_scalar` static_asserts Mul only (`tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_binop_with_unary.h:21`). (4) `UnpackToDest` for a bfloat16 page into 32-bit DEST: not used; bfloat16 input rings use the default unpack mode. (5) `fpu_math_fidelity` is set on every compute kernel. (6) `Identity`, `CopyDest`, `rsub_binary_tile`, comparison binary SFPU ops are `#ifndef ARCH_QUASAR` (not used). | target page; headers cited |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor_a` | `ttnn.Tensor` | yes | PHASE0: bfloat16, TILE, interleaved DRAM/L1, rank 2–4 | — | tensor binding `a` |
| `input_tensor_b` | `ttnn.Tensor` \| `int` \| `float` | yes | tensor: same dtype/layout as a, shape equal to a after left-padding with 1s (broadcast `none`); number: any `int`/`float` (R7, R8) | — | tensor binding `b`, or RT arg `divisor_bits` (float32 bit pattern) |
| `rounding_mode` | `str \| None` | kw-only | `None`, `"floor"`, `"trunc"`; anything else → `ValueError` (X2) | `None` | CT `rounding_mode` ∈ {0 none, 1 floor, 2 trunc} |
| `dtype` | `ttnn.DataType \| None` | kw-only | `None`, `bfloat16`, `float32` | `None` (= a.dtype) | output tensor dtype; out ring `data_format_metadata` |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor \| None` | kw-only | PHASE0: `fp32_dest_acc_en=True`; any `math_fidelity`, any `math_approx_mode` (R4) | `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True, math_approx_mode=False (R3) | compute `KernelSpec.hw_config` (`fpu_math_fidelity`, `enable_32_bit_dest`, `sfpu_precision_mode`) |

Exported from the op package: `div`, `default_compute_kernel_config`, `SUPPORTED`, `EXCLUSIONS`, `INPUT_TAGGERS` (= the specification's `AXES`, imported, never redefined), `validate`.

### Validation order (host)

| # | Check | Raises |
|---|-------|--------|
| 1 | `rounding_mode not in (None, "floor", "trunc")` (X2) | `ValueError` |
| 2 | b is a tensor and the shapes do not broadcast (`torch.broadcast_shapes` fails) (X1) | `ValueError` |
| 3 | R1 / R2 as explicit rules, independent of SUPPORTED (they must hold in every phase): `dtype float32 operands and not fp32_dest_acc_en`, `output dtype float32 and not fp32_dest_acc_en` | `UnsupportedAxisValue` (NotImplementedError) |
| 4 | b tensor with dtype or layout ≠ a's (mixed dtypes are not in TARGET) | `UnsupportedAxisValue` |
| 5 | Axes built like the harness: `dtype`=a.dtype, `layout`=a.layout, `fp32_dest_acc_en`, `rounding_mode`, `output_dtype`=`dtype` arg, `memory_config`/`sharding` from `eval.memory.memory_axes` of **each** tensor operand (R14), then every `INPUT_TAGGERS` function on `(a.shape, b.shape or ())`; each against SUPPORTED, then EXCLUSIONS (R12) | `UnsupportedAxisValue` / `ExcludedCell` |

A Python `int` (bool included) and `float` are both the number divisor: `float(b)` (R8). The validator reads neither `math_fidelity` nor `math_approx_mode` (R4).

## Tensors

### Input

| Property | `input_tensor_a` | `input_tensor_b` (tensor) |
|----------|------------------|---------------------------|
| Shape | rank 2–4, any H/W (tile-padded per image by the TILE layout) | equal to a's shape after left-padding the shorter with 1s (broadcast `none`) |
| Dtype | bfloat16 (PHASE0) | = a.dtype |
| Layout | TILE | TILE |
| Memory | interleaved, DRAM or L1 | interleaved, DRAM or L1 (read through its own accessor) |

### Output

| Property | Value |
|----------|-------|
| Shape | `torch.broadcast_shapes(a.shape, b.shape)` (`a.shape` for a number); may have b's higher rank when b has leading 1s |
| Dtype | `dtype` if given, else a.dtype (R10) |
| Layout | TILE (a's layout) |
| Memory | a's buffer type, interleaved (R13): `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(out_shape, out_dtype, ttnn.TILE_LAYOUT, memory_config=<a's interleaved config>), device)` |

## Blocking Model

The first design section; everything below is its realization. Semantics: `.claude/references/blocking-model.md`.

### Axes

Every output element depends on exactly one element of each operand, so **no axis is dependent**. In Phase 0 (broadcast `none` / `number`) a, b and the output have identical tile index spaces, and the TILE layout stores the pages of `(leading, tile_row, tile_col)` contiguously in that order; the op treats the three identically, so they **fold into one flat tile axis** (blocking-model §1), which is the axis Phase 0 blocks and splits.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `leading` (N = ∏ dims before H) | independent — no element reads another index | folded into `block_tiles` of `flat_tile` | folded | — (`flat_tile`) | through `flat_tile` | broadcast regimes decompose page id → `(n, ht, wt)` in the reader only: knob-turn |
| `tile_row` (Ht = ⌈H/32⌉ per image) | independent — same reason | folded into `block_tiles` | folded | — (`flat_tile`) | through `flat_tile` | as above |
| `tile_col` (Wt = ⌈W/32⌉) | independent — same reason | folded into `block_tiles` | folded | — (`flat_tile`) | through `flat_tile` | as above |
| `flat_tile` (page index 0 … `tensor_tiles`−1, `tensor_tiles = N·Ht·Wt` = `out.buffer_num_pages()`) | independent | `block_tiles` = tiles one NEO takes from each ring per chain call | 4 | host constant `BLOCK_TILES` → CT arg `block_tiles` | whole rounds of C tiles over the **full grid** (`split_work_to_cores(grid, ⌈tensor_tiles/C⌉, row_wise=True)`); inside a cluster tile `i` → NEO `i mod C`, reader hart `i mod R`, writer hart `i mod W` | knob-turn (raise `BLOCK_TILES`, change plan) |
| `intra_tile` (32×32 elements) | independent | whole tile (the hardware unit; SFPU walks all 4 faces) | 1 tile | hardware | never split | SFPU row-scoping on padded tile-rows is a perf lamp, not an unlock |
| divisor operand | tensor `none`: varies along every axis (not reuse-shared); `number`: **reuse-shared over every axis** (one value) | number: none (4 bytes) | RT arg | host `struct.pack('<f')` bits → RT arg `divisor_bits` | every cluster receives the same RT value | — |

Operand-reuse check against the chosen split (`flat_tile` across clusters): a varies along it (no reuse); b-tensor `none` varies along it (no reuse); b-number does not vary → reuse-shared by construction → delivered as an RT argument (zero DRAM bytes), not as a tensor (regime `number_as_tensor` rejected). The future broadcast cells make b reuse-shared along the broadcast axes; their regime rows below carry that.

Intermediates: the scheme has no stage that produces stored data — the quotient and its rounding live only in DEST between unpack and pack — so there is no intermediate to re-block. No cluster has a coordinating role.

Stall-shadow check: the only waits are compute on the reader (DRAM latency) and the writer on compute. The work independent of a tile's arrival is the next tile's, which the depth-2 rings already overlap; div has no algebraic reorder that would lift work around a wait. Nothing else to schedule into the stall.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_dividend_tiles` | `BUFFER_DEPTH` (blocks per NEO in flight) | 2 | reader fills block k+1 while NEO computes block k |
| `cb_divisor_tiles` | `BUFFER_DEPTH` | 2 | same, for b |
| `cb_quotient_tiles` | `BUFFER_DEPTH` | 2 | NEO packs block k+1 while writers drain block k |

All three rings: `num_entries = BUFFER_DEPTH × block_tiles × C` (capacity per NEO × C, target page *Thread plan*). One host constant `BUFFER_DEPTH`; every ring derives from it.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DFB ring size ≤ 1,048,560 B | `BUFFER_DEPTH·block_tiles·C·page_bytes` per ring; worst page 4096 B (float32 out) → `block_tiles ≤ 32` at depth 2, C 4 | host asserts `BUFFER_DEPTH*BLOCK_TILES*C*out_page_bytes <= 1048560` at program build | program build rejects the spec |
| DEST capacity per section (`DEST_AUTO_LIMIT`, `dest_helpers.hpp:89-103`: 4 tiles fp32 half-sync) | DEST sub-block of the chain: lane width 2 (D0 a/quotient, D1 b) → 2 tiles per section | the chain clamps `block_size·lane_width ≤ DEST_AUTO_LIMIT` itself (`chain.hpp:504-509`) — use `BlockTailSync::ValidTiles` (default), **never** `FullBlock` | — |
| Thread plan legality | R, C, W | `C ∈ {1,2,4}`, `R+W ≤ 6`, `C % R == 0`, `C % W == 0`, checked on host (as `toy_thread_plan.validate_thread_plan`) | — |
| `divisor_bits` is a float32 transport | the number's value | convert with float32 semantics that saturate (`numpy.float32(b)` → `.view(uint32)`): |b| > FLT_MAX becomes ±inf, not an `OverflowError`; the spec allows the number held as float32 (NUMERIC freedoms) | `struct.pack('<f', 1e40)` raises |
| Compute binary ≤ 24 KiB | number of chain variants compiled into one binary | variants selected by CT args (`rounding_mode`, `divisor_is_number`) with `if constexpr`, so each binary holds one chain | link failure |

### Regimes

Selection predicate (host, exact; evaluated after validation):

```
if isinstance(input_tensor_b, (int, float)):            -> eltwise_number
elif tag_broadcast((a.shape, b.shape), {}) == "none":    -> eltwise_same_shape
else:                                                     -> refused by validate() in Phase 0 (broadcast not in SUPPORTED)
```

Minimum (named boundary DRAM/L1-bank ↔ cluster L1): each tensor input crosses once, the output crosses once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `eltwise_same_shape` | **built** | b is a tensor and `tag_broadcast == "none"` | `block_tiles` tiles of `flat_tile` per NEO per chain call (ragged last block: `last_block_extent`) | minimum: a once, b once, out once; no cross-core traffic | per block: one upfront wait/pop on 2 input rings and one reserve/push on the out ring per NEO, chain entry reconfig fold; per DEST section (2 tiles): copy/SFPU re-inits (div, floor/trunc inits are not uniform, so the chain re-emits them per section — a block ≥ 2 tiles amortizes them over 2 tiles) |
| `eltwise_number` | **built** | `isinstance(b, (int, float))` | same | minimum: a once, out once; the divisor is 4 RT bytes per cluster | same as above, with `FillBitcast` into D1 per DEST section instead of the b copy |
| `bcast_b_leading` (`over_leading`) | deferred — outside PHASE0's `broadcast` axis; reachable: the reader already holds the page id, a broadcast stride decomposition `p → (n, ht, wt) → b page` is a reader-only change | b tensor, `tag_broadcast == "over_leading"` | same | stepping stone: b re-read per output tile (b crosses N_out/N_b times). Successor `bcast_b_resident` | same |
| `bcast_b_inner` (`over_h`, `over_w`, `over_hw`) | deferred — outside PHASE0; reachable: reader maps out tile → b tile `(n_b, ht or 0, wt or 0)`, compute replaces `CopyTile<b, D1>` by a row/col/scalar broadcast load into D1 (chain's unary-broadcast element over `llk_math_unary_broadcast.h`; its Quasar build is to be verified at that refinement) — same chain, one element swapped | `tag_broadcast in {"over_h","over_w","over_hw"}` | same | stepping stone: b tile re-read per output tile | same |
| `bcast_a_first_operand` | deferred — outside PHASE0; reachable: the same mapping applied to a's page id and a's D0 load | `tag_broadcast == "first_operand"` | same | as `bcast_b_inner` on a | same |
| `bcast_b_resident` | deferred — successor of the three broadcast rows; positive reason: no PHASE0 cell has a reused b. Mechanism: split the output so each cluster's range spans the broadcast axes and cuts the axes b varies along (e.g. `(4,8,32,256)/(1,1,32,256)`: cluster owns b tiles {j} for all n), read those b tiles once into a resident ring (`ALL` access pattern, one entry seen by every NEO) | broadcast cells | same | delta: b crosses once in total instead of once per output tile (for `(4,8,32,256)/(1,1,32,256)`: 8 b tiles instead of 256) | same |
| `bcast_b_mcast` | deferred — only pays where several clusters need the same b tiles after `bcast_b_resident`'s split; positive reason: that split already makes each b tile's readers disjoint for every broadcast entry of INPUTS | broadcast cells | same | delta: (readers−1)·\|b_shared\| fewer DRAM crossings, + multicast payload | — |
| `materialize_broadcast` | rejected — a second dispatch (one-dispatch rule) and an output-sized copy; superseded by the broadcast rows above | — | — | +1 output-sized write and read | — |
| `div_then_round_separately` | rejected — a second program or a stored quotient; superseded by the fused chain (`Floor`/`Trunc` act on the quotient in DEST; GUIDANCE "Rounding modes") | — | — | +2 output-sized crossings | — |
| `number_as_tensor` | rejected — a full or one-tile tensor read where an RT arg suffices; superseded by `eltwise_number` | — | — | +\|a\| (full) or +1 tile per cluster | — |
| `split_dependent_axis_combine` | rejected — no dependent axis exists (every output element reads one element of each operand), so there is nothing to combine | — | — | — | — |

### Traffic ranking

| Candidate split | DRAM crossings | Cross-core | Balance | Verdict |
|-----------------|----------------|------------|---------|---------|
| `flat_tile` in whole rounds of C, contiguous per cluster (**chosen**) | a 1×, b 1×, out 1× (minimum) | none | ±C tiles per cluster | cheapest; finest balance; contiguous page ranges |
| `tile_row` only / `leading` only | same bytes | none | coarse: `(1,2,8400)` has 1 tile-row → 1 cluster | rejected: equal bytes, strands the YOLO shapes on one cluster (catalog `width_split`) |
| `tile_col` only | same bytes | none | as `flat_tile` for Ht=1, worse otherwise | dominated by `flat_tile` |
| dependent-axis split + combine | — | — | — | not applicable: no dependent axis |

All candidate splits move equal bytes; `flat_tile` wins on balance. Clusters are laid out with `row_wise=True` (catalog `noc_placement`).

### Block schedule

```cpp
// per cluster, per NEO t (the NEO's share: tiles t, t+C, ... of the cluster's range)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_neo; ++block_idx) {   // num_blocks_this_neo = ceil(neo_tiles / block_tiles)
    load_block(block_idx);           // reader harts: a tiles (+ b tiles) of this block into the input rings
    divide_round_block(block_idx);   // NEO: quotient [-> floor | trunc] in DEST, packed into the out ring
    store_block(block_idx);          // writer harts: out tiles of this block to the output tensor
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_tiles` (last: `last_block_extent`) tiles per operand per NEO; reader hart p serves NEO p (R = C) | nothing | one tagged `noc.async_read<TXN_ID>` per tile per operand ring, no reserve/barrier/push (implicit sync); tag handed back (`noc_async_read_set_trid(0, ...)`) once at kernel end |
| `divide_round_block` | same | nothing persists across blocks (the number lives in an RT arg and is re-filled per DEST section) | `compute_kernel_hw_startup` once per kernel; one `eltwise_chain` call per block: upfront wait + end pop per input ring, upfront reserve + end push on the out ring, element reconfig fold once per call; per DEST section (2 tiles): unpack/copy, fill (number), `div_binary_tile`, `floor_tile`/`trunc_tile`, pack, with the SFPU inits the chain re-emits per section |
| `store_block` | same; writer hart w serves NEOs w, w+W (W = 2) | nothing | one tagged `noc.async_write<TXN_ID>` per tile; tag handed back once at kernel end |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (block size) | per-NEO shares over INPUTS are 1–9 tiles; `block_tiles = 4` gives 1–3 blocks per NEO, so a larger block (whole share) would serialize read → compute → write; a smaller one pays more handshakes | `BLOCK_TILES ∈ {2, 4, 8}` |
| Grid synchronization / occupancy | tiny tensors (`(1,1,32,64)`: 2 tiles → 1 cluster, 2 NEOs) are launch-dominated; the whole-round split already engages only ⌈tiles/C⌉ clusters | per-cluster round size 1 vs. 2 rounds minimum |
| Thread plan | bound reasoned as compute (SFPU div); unmeasured on this backend (no timing) | (1,1,1), (1,4,1), (2,4,2), (4,4,2) — on `ttsim-perf` via `eval.model_cycles`; for the number path (one input) (2,4,2) |
| SFPU work on tile padding | `(1,2,8400)`-class shapes (the traced YOLO calls) have H=2 in a 32-row tile: 30/32 of every tile's SFPU div work is padding | scope the SFPU to the valid rows of the last tile-row (`VectorMode::R`/`ITERATIONS`, catalog `sfpu_tile_scope`) — needs raw calls; measure only if compute-bound is confirmed |
| Data-format reconfig | inputs bfloat16, output possibly float32: reconfig is needed on the pack side; when all formats agree (bf16→bf16) the chain elides it already | — (recorded for completeness; catalog `compute_block_size` second lever) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM/L1 → `cb_dividend_tiles` | bfloat16 tiles (2048 B) | reader, R = 4 harts, `TensorAccessor(tensor::a)`, `noc.async_read<NocOptions::TXN_ID>(acc_a, dfb_a, {.page_id = start_id + i}, {})`, implicit sync | hart p reads cluster tiles p, p+R, … |
| DRAM/L1 → `cb_divisor_tiles` | bfloat16 tiles | same reader, `tensor::b`; ring and binding exist only when `divisor_is_number == 0` | page id identical to a's (same-shape regime) |
| number → DEST D1 | float32 bits | RT arg `divisor_bits` → `FillBitcast<Dst::D1>{bits}` | no ring, no tensor |
| ring → DEST | bf16 → fp32 DEST via srcA datacopy (default unpack mode) | `CopyTile` | **not** `UnpackToDest` |
| DEST compute | fp32 | `DivBinary<D0,D1,D0>` (SFPU: `a · reciprocal(b)` with 2 Newton steps, 0/0→NaN, x/0→±inf, x/x→1), then `Floor<D0>` / `Trunc<D0>` per CT `rounding_mode` | rounding acts on the fp32 quotient before it is stored (R5, R6) |
| DEST → `cb_quotient_tiles` | output dtype (bf16 2048 B or fp32 4096 B) | `PackTile<out>`; one rounding into the output dtype (R10) | pack reconfig from the out ring's format |
| `cb_quotient_tiles` → DRAM/L1 | output dtype | writer, W = 2 harts, `noc.async_write<NocOptions::TXN_ID>(dfb_out, acc_out, {}, {.page_id = start_id + i})`, implicit sync | output allocated in a's buffer (R13) |

All three rings are STRIDED (default), filled/drained by NoC transfers → **implicit sync** on both DM kernels; no `disable_dfb_implicit_sync_for`. Each DM kernel's last statement hands the tag back (`noc_async_read_set_trid(0, noc.get_noc_id())` / `noc_async_write_set_trid(0, noc.get_noc_id())`).

Deferred broadcast regimes keep this topology: only the reader's page-id mapping for the reused operand and the compute element that loads it change; `bcast_b_resident` adds an `ALL`-pattern ring; `bcast_b_mcast` would use `ttnn.Mcast` + `McastArgs` with a fixed sender per b-tile group.

### TARGET − PHASE0 (refinement classes)

| Axis value | Class | What changes |
|------------|-------|--------------|
| `broadcast`: `over_leading`, `over_h`, `over_w`, `over_hw`, `first_operand` | knob-turn on structure (reader mapping + one chain element) → regime rows above | reader page mapping; `CopyTile` → broadcast load; output shape from broadcast |
| `dtype`: float32 operands | knob-turn | input rings Float32; load fp32 operands exactly: `UnpackToDest` (allowed for 32-bit pages) — the srcA path would truncate to tf32 |
| `fp32_dest_acc_en`: False (bf16 operands and bf16 output only; R1/R2 refuse the rest) | knob-turn | `enable_32_bit_dest=False`; `div_binary_tile` then rounds RNE in software itself (`ckernel_sfpu_binary.h`, `round_div_to_bf16`); DEST section 4 tiles for lane width 2 |
| `layout`: ROW_MAJOR | scheme-change (deferred, not a regime of the tiled scheme): sticks → tiles in-kernel (tilize/untilize helpers on Quasar to be verified) or a stick-stream eltwise path | reader/writer and compute front/back |
| `sharding`: only `interleaved` in TARGET | — | — |
| `memory_config` L1 | in PHASE0 | none: accessor-based reads are placement-agnostic |

### Structural impossibilities

None found beyond the specification's (`INVALID = []`).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one tile of `flat_tile`, dealt to NEOs round-robin; clusters get whole rounds of C tiles |
| Grid | `device.compute_with_storage_grid_size()` (Quasar 8×4 = 32) |
| Per-core work | `num_rounds = ceil(tensor_tiles / C)`; `ttnn.split_work_to_cores(grid, num_rounds, row_wise=True)` → rounds per cluster; cluster tiles = `min(rounds·C, tensor_tiles − start_id)` (only the last cluster is short); RT args `num_tiles`, `start_id` per cluster; per NEO `neo_tiles = n/C + (t < n % C)` |
| Remainder | `tensor_tiles = out.buffer_num_pages()` = ∏(padded leading dims) · ⌈H/32⌉ · ⌈W/32⌉ (per image, ceil); tile-aligned or not, the same formula; the ragged last block per NEO is `last_block_extent = neo_tiles − (num_blocks_this_neo−1)·block_tiles` |
| Bound | **compute**: per tile the SFPU runs a reciprocal with two Newton steps, a multiply, two predicated special cases, optionally floor/trunc (≈10 SFPU ops) and, for a number, a fill — against 2 (or 1) 2 KiB reads and one 2–4 KiB write |
| Thread plan | `THREAD_PLAN = (4, 4, 2)` (R, C, W), one host constant, counts passed as CT args `reader_threads`, `compute_threads`, `writer_threads`; R = 4 because the tensor path reads two inputs per output tile; same constant for the number path (R = 4 harmless; (2,4,2) is a lamp) |
| Unit dealt to compute threads | the ring entry itself (one tile): tiles may stop anywhere, no padding; the host still deals whole rounds of C tiles to clusters so no cluster has idle NEOs except the last |
| Tests the plan needs | (1) every NEO ran: a probe CT flag (e.g. `tag_threads`) under which NEO t writes a NEO-identifying variant (e.g. odd NEOs multiply instead of divide), checked per tile against `local_index mod C` (as `test_each_neo_computes_its_own_tiles`); (2) one device session: div after another program (e.g. two different div calls, then a `toy`-style different plan), and div followed by `ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)` (tt-metal program, untagged transfers) |

Worked numbers (Quasar, 32 clusters, C = 4):

| Shape (a) | tensor_tiles | rounds | clusters engaged | tiles / cluster | tiles / NEO | blocks / NEO (block 4) |
|-----------|--------------|--------|------------------|-----------------|-------------|------------------------|
| (1,2,33600) | 1050 | 263 | 32 | 32 or 36 (last 30) | 8–9 | 2–3 |
| (1,2,8400) | 263 | 66 | 32 | 8 or 12 (last 7) | 2–3 | 1 |
| (1,2,400) | 13 | 4 | 4 | 4,4,4,1 | 0–1 | 0–1 |
| (2,512,1024) | 1024 | 256 | 32 | 32 | 8 | 2 |
| (1,1,32,8192) | 256 | 64 | 32 | 8 | 2 | 1 |
| (1,1,32,64) | 2 | 1 | 1 | 2 | 0–1 | 0–1 |

A NEO with 0 tiles runs its chain zero times (no wait, no pop); reader/writer harts loop zero times.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|----------|
| `cb_dividend_tiles` | DFB `in_a` | `a.buffer_page_size()` (2048) | `BUFFER_DEPTH·block_tiles·C` = 32 | spans `flat_tile` (block_tiles per NEO), double-buffered; streams `leading`/`tile_row`/`tile_col` (folded) | a.dtype (Float16_b) | reader | compute | reader: implicit (tagged reads); compute: chain wait/pop | whole program |
| `cb_divisor_tiles` | DFB `in_b` (tensor path only) | `b.buffer_page_size()` (2048) | `BUFFER_DEPTH·block_tiles·C` = 32 | same as above | b.dtype (Float16_b) | reader | compute | same | whole program |
| `cb_quotient_tiles` | DFB `out` | `out.buffer_page_size()` (2048 bf16 / 4096 fp32) | `BUFFER_DEPTH·block_tiles·C` = 32 | same as above | output dtype (Float16_b / Float32) | compute | writer | compute: chain reserve/push; writer: implicit (tagged writes) | whole program |

No scaler, no intermediate, no constant ring: the number is an RT arg; the quotient and its rounding are fused in DEST and packed straight into the destination ring (reuse pattern 1, "pack into the destination").

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_tiles` tiles / operand / NEO | dataflow API (tagged NoC read; no helper covers a ring fill under implicit sync) | tensors a (and b) via `TensorAccessor` | `cb_dividend_tiles`, `cb_divisor_tiles`: `block_tiles` per NEO credited by DM0 ISR | — |
| 2 | `divide_round_block` | `block_tiles` tiles (chain DEST sub-blocks of 2) | yes: `eltwise_chain` | `cb_dividend_tiles` (+ `cb_divisor_tiles`): wait `n` upfront, pop `n` at end | `cb_quotient_tiles`: reserve `n` upfront, push `n` at end | inputs freed, `n` quotient tiles published |
| 3 | `store_block` | `block_tiles` tiles / NEO, served by 2 writer harts | dataflow API (tagged NoC write) | `cb_quotient_tiles` | output tensor via `TensorAccessor` | out entries acked by DM0 ISR |

Chain per variant (`n = this block's extent`; `IterationShape::tiles(n).block_size(block_tiles)`; inputs `input(dfb::X, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block)`, output `output(dfb::out, ReservePolicy::Upfront, PushPolicy::AtEnd)` — the form `toy_thread_plan/kernels/compute.cpp` runs under plans (1,1,1)…(4,4,2) with ragged blocks):

| Variant (CT) | Chain elements |
|--------------|----------------|
| tensor, none | `CopyTile<in_a, D0>{a}`, `CopyTile<in_b, D1>{b}`, `DivBinary<D0,D1,D0>{}`, `PackTile<out, D0>{out}` |
| tensor, floor / trunc | as above + `Floor<D0>{}` / `Trunc<D0>{}` before `PackTile` |
| number, none | `CopyTile<in_a, D0>{a}`, `FillBitcast<D1>{divisor_bits}`, `DivBinary<D0,D1,D0>{}`, `PackTile<out, D0>{out}` |
| number, floor / trunc | as above + `Floor<D0>{}` / `Trunc<D0>{}` before `PackTile` |

The quotient for a number goes through the same `div_binary_tile` as for a tensor, so a number divides exactly like a one-element tensor holding it (R7), including x/x = 1 and the zero cases.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | helper prerequisite | `compute_kernel_hw_startup(icb0, icb1, ocb)` / `(icb0, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60`, `:113` | `(dfb::in_a, dfb::in_b, dfb::out)`; number path `(dfb::in_a, dfb::out)` | — | — | — |
| `divide_round_block` | helper | `eltwise_chain` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:619-620` | `IterationShape::tiles(n).block_size(block_tiles)` (`chain.hpp:146-175`) | in_a, in_b | out | `block_tiles` (CB window), `n` (ragged) |
| ″ | helper element | `CopyTile<input(...), Dst>` | `chain.hpp:553-557` | `input(dfb::in_a, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block)`, D0; in_b → D1 | in_a / in_b | DEST | — |
| ″ | helper element | `FillBitcast<Dst::D1>` | `ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp:24`, `fill.inl:37` (→ `fill_tile_bitcast`, `tt_metal/hw/inc/api/compute/eltwise_unary/fill.h:81`) | ctor `{divisor_bits}` (RT arg) | — | DEST D1 | — |
| ″ | helper element | `DivBinary<D0, D1, D0>` | `ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp:30`, `sfpu.inl:64` (→ `div_binary_tile`, `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h:43`; Quasar body `tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`, `BinaryOp::DIV`) | — | DEST | DEST D0 | — |
| ″ | helper element | `Floor<D0>` / `Trunc<D0>` | `ttnn/cpp/ttnn/kernel_lib/eltwise/unary/rounding.hpp:11,15`, `rounding.inl:14,26` (→ `floor_tile`/`trunc_tile`, `tt_metal/hw/inc/api/compute/eltwise_unary/rounding.h:53,72`; Quasar `ckernel_sfpu_rounding_ops.h`) | — | DEST | DEST D0 | — |
| ″ | helper element | `PackTile<output(...), D0>` | `chain.hpp:587-591` | `output(dfb::out, ReservePolicy::Upfront, PushPolicy::AtEnd)` | DEST | out | — |
| `load_block` | raw_api (dataflow) | `noc.async_read<NocOptions::TXN_ID>` + `noc_async_read_set_trid(0, ...)` | pattern `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader_implicit.cpp:26-35` | `{.page_id = start_id + i}`, i = hart id, step R | tensor a, b | in_a, in_b | — |
| `store_block` | raw_api (dataflow) | `noc.async_write<NocOptions::TXN_ID>` + `noc_async_write_set_trid(0, ...)` | pattern `toy_thread_plan/kernels/writer_implicit.cpp:23-29` | `{.page_id = start_id + i}`, step W | out | tensor out | — |

Helpers considered and rejected:

| Candidate | Where | Why not |
|-----------|-------|---------|
| `binary_sfpu<DivBinary<>, ...>` convenience | `eltwise/api/convenience.hpp:106`, body `convenience.inl:60-64` | covers only tensor/none (`CopyTile a, CopyTile b, Op, PackTile`); it cannot add `Floor`/`Trunc` or replace the b load with a fill. One `eltwise_chain` form serves all six variants; for tensor/none the implementer may use either (identical chain) |
| `DivUnary` (scalar div) | `eltwise/unary/scalar.hpp:33` | Quasar `calculate_binop_with_scalar` supports Mul only (`ckernel_sfpu_binop_with_unary.h:21` static_assert) |
| `MulUnary` with a host reciprocal | `eltwise/unary/scalar.hpp:30` | numerically not "like a one-element tensor" (R7): a·fl(1/b) differs from the device's `div_binary_tile` path (x/x = 1, zero handling, Newton reciprocal); kept off for one quotient path |
| dataflow helpers for the reader/writer | `kernel_lib/dfb_helpers_dataflow.hpp`, `tile_column_dataflow.hpp` | the target page's implicit-sync protocol is one tagged NoC call per entry with no reserve/push; no helper in `kernel_lib` issues tagged implicit-sync transfers |

Host side (Python, `ttnn.experimental` = `E`): `E.DataflowBufferSpec(unique_id, entry_size, num_entries, data_format_metadata)`, `E.KernelSpec(... num_threads=..., dfb_bindings=[E.producer_of / E.consumer_of], tensor_bindings, compile_time_args, runtime_arg_schema)`, `E.ComputeHardwareConfig(fpu_math_fidelity=<config.math_fidelity>, enable_32_bit_dest=<config.fp32_dest_acc_en>, sfpu_precision_mode=<from math_approx_mode>)` (`ttnn/cpp/ttnn-nanobind/program_spec.cpp:522-555`; fidelity always set), `E.WorkUnitSpec(target_nodes=core_set)`, `E.ProgramSpec`, `E.ProgramRunArgs`, `ttnn.generic_op([a, (b,) out], spec, run_args, tensor_indices)` — exactly one dispatch per call. Pattern: `toy_thread_plan/toy_thread_plan_program_artifacts.py`.

## Broadcast Verification

| Phase | Op | CB_A (`cb_dividend_tiles`) Valid Region | CB_B (`cb_divisor_tiles` / D1 fill) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `divide_round_block`, tensor/none | `DivBinary` (DEST–DEST SFPU) | All | All (same shape, page i ↔ page i) | None |
| `divide_round_block`, number | `DivBinary` | All | All (D1 filled with the number on every element) | None (scalar realized by the fill) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| bfloat16 output is graded at **0 ulp** (`NUMERIC_RESOLVED[('bfloat16','default')]`) | correct only if the fp32 quotient is narrowed to bf16 round-to-nearest-even at pack. A quotient of two bf16 values is never closer than ~2⁻¹⁷ relative to a bf16 midpoint (an odd 9-bit × odd ≤8-bit product cannot have ≤8 bits), and the fp32 SFPU quotient is within a few 2⁻²³ — so RNE narrowing gives exactly the correctly rounded result; truncation would miss ~half the elements | first acceptance run: a bf16/bf16/no-rounding cell. If 1-ulp misses appear, narrow in DEST before pack with `Typecast<Float32, Float16_b, D0>` (`eltwise/unary/misc.inl:86-90`) only after checking it rounds RNE on Quasar, else a raw SFPU RNE step (`float32_to_bf16_rne`, used by `ckernel_sfpu_binary.h` itself) |
| `UnpackToDest` on bf16 pages | not used | input rings use the default unpack mode; do not set unpack-to-dest modes in `ComputeHardwareConfig` |
| Ring tag | an implicit-sync kernel leaves a ring's transaction id in the hart's tag register | last statement of reader `noc_async_read_set_trid(0, noc.get_noc_id())`, of writer `noc_async_write_set_trid(0, noc.get_noc_id())`; test (2) above |
| `fpu_math_fidelity` | set on every compute kernel | always pass `config.math_fidelity` into the compute `ComputeHardwareConfig` |
| Padding of non-aligned tiles divides 0/0 | NaN/inf in padding (e.g. `(1,1,17,64)`, `(1,2,8400)`) | allowed (GUIDANCE "Padding"); only logical elements are compared; nothing to mask |
| b of lower rank with broadcast `none` (`(1,1,32,64)` vs `(32,64)` padded equal) | output shape has the larger rank | output shape = `torch.broadcast_shapes`; page ids match because the extra leading dims are 1 |
| Number divisor as `int` (traced `div(a, 2)`) | runner passes floats; R8 | `float(b)` for `int`/`float`; bits via float32 conversion that saturates (mechanism cap) |
| `num_tiles` RT per cluster vs. NEOs with 0 tiles | small tensors leave NEOs idle; a chain call with `n = 0` must not wait | compute loops `num_blocks_this_neo = ceil(neo_tiles / block_tiles)` times; 0 blocks → no call |
| R1 / R2 must hold in every phase | PHASE0 refuses `fp32_dest_acc_en=False` via SUPPORTED, a later phase adding False must still refuse float32 cases | explicit rule checks in `validate()` (step 3), not just SUPPORTED |
| Program-cache / recompile on ttsim | CT args that vary per shape recompile kernels | CT args are only plan counts, `block_tiles`, `rounding_mode`, `divisor_is_number`; shapes, ranges and the number are RT args; `BLOCK_TILES` is not clamped per shape |

# Operation Design: add_

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (eltwise binary, in place, fused activation) |
| Goal | `a <- a + b` (and `a <- max(a + b, 0)` with RELU) written into `input_tensor_a`'s own buffer and returned as that tensor; one native device program per call; no tensor allocated. Phase 0: height-sharded L1 operands consumed and written in place through rings borrowed from the shards, with zero NoC and zero DRAM traffic. |
| Math | `a[i] = act(round_dtype(a[i] + b[i]))`, `act ∈ {identity, relu}`; element i depends on element i of each operand only |
| References | Specification `eval/specs/add_/spec.py` (governs); target `.claude/references/targets/quasar.md`; backend `.claude/references/backends/ttsim.md`; host-program examples `.claude/references/quasar_op_examples/` (README, `toy_thread_plan`); ring semantics `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` (Part A, C2, C6, C8); borrowed ring `tt_metal/api/tt-metalium/workload/specification/dataflow_buffer_spec.hpp:101-112`; sharded ring constructor `ttnn/cpp/ttnn-nanobind/program_specs.cpp:26-84`; catalog `ttnn/ttnn/operations/examples/master.md` (`compute_block_size`, `compute_fusion`, `double_buffer`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (probe on this backend: `compute_with_storage_grid_size()` = 8-4). Predicates query `device.compute_with_storage_grid_size()`. In the sharded regime the grid in use is the **caller's shard grid**, read from the tensor. | target page; probe 2026-11-21 |
| L1 per core | 4,194,304 B | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` = 3,963,328 B − reserve on this target (probe 2026-11-21). **No Phase 0 predicate uses it**: the op allocates 0 B of L1 (every ring is borrowed from a caller's shard, see `l1_ledger.md`). | probe |
| Host program | `ttnn.ProgramSpec` (`ttnn.experimental.*`) dispatched by `ttnn.generic_op(io_tensors, spec, run_args, tensor_args)`. Reference: `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/toy_thread_plan_program_artifacts.py`) | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1, 2, 4}); 6 user data-movement harts (R + W ≤ 6); on STRIDED rings R | C and W | C; ring capacity counted per compute thread | target page, *Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ program factory (refused at kernel creation). `bfloat8_b`, `bfloat4_b` (no path on Quasar; not in TARGET anyway). `UnpackToDest`: not used; every ring here uses the default unpack mode. Compute kernels must set `fpu_math_fidelity` explicitly on ttsim. | target page |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor_a` | `ttnn.Tensor` | yes | Phase 0: bfloat16, TILE, rank 4, W % 32 == 0, HEIGHT_SHARDED in L1 | — | backs two rings (`borrowed_from="a"`) |
| `input_tensor_b` | `ttnn.Tensor` | yes | same shape, dtype, layout and memory config (same shard spec) as `a` | — | backs one ring (`borrowed_from="b"`) |
| `activations` (kw-only) | `None` / list | no | `None`, `[]`, `[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]`, `[ttnn.UnaryOpType.RELU]`, read with the spec's `canonical_activations` → axis `activation` ∈ {None, "relu"}; anything else is `"other"` and refused by name. Phase 0 supports `"relu"` only. | `None` | CT arg `pack_relu` (0/1) on compute |
| `compute_kernel_config` (kw-only) | `ttnn.ComputeConfigDescriptor` | no | Phase 0: `fp32_dest_acc_en=True`; `math_fidelity`, `math_approx_mode` accepted at any value, never gated (R3) | `None` → exported `default_compute_kernel_config()` (R2): HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False` | compute `ComputeHardwareConfig(fpu_math_fidelity=cfg.math_fidelity, enable_32_bit_dest=cfg.fp32_dest_acc_en, sfpu_precision_mode=Approximate if cfg.math_approx_mode else Precise)`; fidelity always set explicitly |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `a.shape == b.shape` (`broadcast` axis = `"none"`). A `b` that does not broadcast to `a` is malformed (X1: `ValueError`). One that broadcasts but differs is axis value `over_*`, refused by name (`NotImplementedError`). |
| Dtype | Phase 0 bfloat16. `b.dtype != a.dtype` is refused (R7). |
| Layout | TILE (`b.layout != a.layout` refused, R7) |
| Memory | Phase 0: both `memory_axes` = `{L1, height}` (from `eval.memory.memory_axes`, R7), **and identical shard specs** (grid, shard shape, orientation), because `b`'s ring must sit on the same cores as `a`'s. A different shard spec raises `NotImplementedError`. |

### Output

| Property | Value |
|----------|-------|
| Shape | `a.shape` |
| Dtype | `a.dtype` |
| Layout | `a.layout` |
| Memory | `a`'s own buffer, shard spec and cores (R4). The op returns the **`input_tensor_a` object itself**. `ttnn.generic_op([b, a], ...)` is called with `a` last and its return value (a handle of the same buffer) is discarded. |

### Entry-point contract (host, before any dispatch)

| Step | Decision |
|------|----------|
| 1 | `activation = canonical_activations(activations, {})` (imported from the specification). |
| 2 | `cfg = compute_kernel_config or default_compute_kernel_config()`. This is the only place the default lives (R2). |
| 3 | X1: if `b.shape` does not broadcast to `a.shape`, raise `ValueError`. |
| 4 | Axes: `dtype`, `layout`, `fp32_dest_acc_en = cfg.fp32_dest_acc_en`, `activation`. `memory_config`/`sharding` come from `eval.memory.memory_axes(a)` and must equal those of `b`, else `NotImplementedError`. `alignment`/`rank`/`broadcast` come from `INPUT_TAGGERS = AXES` on `(a.shape, b.shape)`. |
| 5 | Gate on `SUPPORTED` (= `PHASE0`) and `EXCLUSIONS` (R1's `{float32, fp32_dest_acc_en=False}` is an EXCLUSIONS entry once float32 is supported), raising `NotImplementedError` (R9). Never read `math_fidelity`/`math_approx_mode` (R3). |
| 6 | Sharded regime checks: `a.memory_config().shard_spec` is not None. `b`'s shard spec equals `a`'s. `shard_shape[1] == tensor_col_tiles * 32` (height sharding holds the full padded width). |

Structural impossibilities: none to add to the specification's INVALID.

## Blocking Model

The op's tile index space is 2-D: `[tensor_row_tiles, tensor_col_tiles]`, where `tensor_row_tiles = prod(shape[:-2]) * ceil(H / 32)` (per image: every leading slab is tile-padded independently) and `tensor_col_tiles = ceil(W / 32)`. The leading dimensions fold into the row axis because the op treats them identically and they are contiguous in the tiled physical layout. **Every axis is independent**: no result spans an axis (no reduction, scan or combine), and both operands vary along every axis. The operand-reuse check (below) finds no reuse-shared operand under the chosen split.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `rows` (leading dims × tile rows, `tensor_row_tiles`) | independent: `a[i]` depends on `a[i]`, `b[i]` only | `block_row_tiles` | `shard_row_tiles` (the whole resident shard, `shard_shape[0] / 32`). A core's actual extent is `core_row_tiles_k = clamp(tensor_row_tiles − k·shard_row_tiles, 0, shard_row_tiles)`. | host: derived once from `a.memory_config().shard_spec.shape[0]`; per-core `core_tiles` is an RT arg | the caller's height-shard map: core k (k-th core of `corerange_to_cores(shard_grid, None, row_wise = orientation == ROW_MAJOR)`) owns rows `[k·shard_row_tiles, …)`. Cores whose shard holds no tile get no kernels. | knob-turn (interleaved placement re-deals rows with `split_work_to_cores`) |
| `cols` (`tensor_col_tiles`) | independent: same reason | `block_col_tiles` | `tensor_col_tiles` (a height shard holds the full padded width) | host: `shard_shape[1] / 32`, asserted equal to `ceil(W/32)` | not split across cores: the caller's height shard holds every column. Width sharding is not in TARGET. | knob-turn |
| `core tile walk` (the core's block flattened row-major = ring order, dealt round-robin over C compute threads) | independent: any tile→thread deal is legal | `neo_block_tiles` (tiles one helper call takes per compute thread; the IterationShape `block_size`) | `shard_tiles / C`, the full per-thread ring capacity, so each compute thread does its whole share in **one** helper call. Coarsest possible (catalog `compute_block_size`: one call over the whole block beats per-row passes). | host constant derived from `shard_tiles` and `compute_threads`; passed as compute CT arg `neo_block_tiles` | thread t of C takes tiles t, t+C, … (STRIDED ring) | knob-turn (lower it for overlap; see lamps) |

Intermediate data: none. The scheme has one compute stage (add, with the clamp applied in the packer), which packs straight into its destination. No stage collects or coordinates on one core.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_a_shard` | none: pinned by the shard (`num_entries = shard_tiles`, the borrowed storage) | `shard_tiles` | the whole operand is already resident. The ring must cover the whole shard because it cannot wrap (see caps). |
| `cb_b_shard` | none: pinned by the shard | `shard_tiles` | same |
| `cb_result_shard` | none: pinned by the shard | `shard_tiles` | same. The pack never waits for space. |
| (deferred interleaved regime) `cb_a_tiles`, `cb_b_tiles`, `cb_result_tiles` | `ring_depth_blocks` | 2 | double buffering of NoC reads/writes against compute (catalog `double_buffer`) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| STRIDED ring: capacity per compute thread = `num_entries / C` (DataflowBuffer.md C2; ring entry k belongs to thread k mod C) | compute-thread count vs `shard_tiles` | `compute_threads = max{c ∈ (PLAN_C, PLAN_C/2, …, 1) : shard_tiles % c == 0 and c ≤ max_k core_tiles_k}` | the host rejects the ring, or the tile→thread map stops matching the shard layout and tiles are added to the wrong partners |
| Borrowed ring starts at the shard base with no offset (`dataflow_buffer_spec.hpp:101-112`) and wraps after `num_entries` | entries pushed per core | `num_entries = shard_tiles` (the default of `dfb_spec_from_sharded_tensor`); `core_tiles_k ≤ shard_tiles` by construction | a ring shorter than the shard wraps back to tile 0: wrong tiles read and overwritten |
| Ring extent. The target page states 1,048,560 B per ring on the compute side. The code at this base stores the extent in uint32 L1 units (`tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h:38-44`) and rejects only rings larger than the unreserved L1 (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:1600-1665`, `:2520`). | `shard_tiles · tile_bytes` | none in `validate()`: the host validator is authoritative. Phase 0's largest shard is 802,816 B (ResNet layer 1, 2 cores), under both figures. The op test file runs one shard above 1,048,560 B (Key Risks). | if the page's figure is the real limit, a larger shard is mis-addressed. The `staged_shard` regime row is the fallback. |
| Credit counts are `uint16` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-246`) | per-thread share `≤ neo_block_tiles` | bounded by `shard_tiles ≤ unreserved_L1 / tile_bytes` (≤ 1,935 bf16 tiles on this target) ≪ 65,535 | count truncation, which hangs at kernel-exit drain |
| Thread plan: C ∈ {1, 2, 4}, R + W ≤ 6, R \| C, W \| C | R, C, W | R = W = 1 divides every C | the host refuses the plan |
| Height shard geometry: `shard_shape[0] % 32 == 0`, `shard_shape[1] == tensor_col_tiles·32` | `block_row_tiles`, `block_col_tiles` | assert in validate (step 6) | rows mis-assigned to cores |

DEST: one tile per DST window, no accumulators held live, so no DEST cap enters.

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `sharded_resident` | **built** | both operands `{memory_config: L1, sharding: height}` with identical shard spec, and the cell in SUPPORTED | per core: `[core_row_tiles_k × tensor_col_tiles]` = `core_tiles_k` tiles, dealt round-robin to C compute threads; per thread one block of ≤ `neo_block_tiles` tiles | **minimum at the L1 boundary**: each operand's bytes are unpacked from L1 once, the result packed into L1 once. **0 B over the NoC, 0 B to DRAM.** | the per-call fixed cost of `eltwise_chain`: the FPU binary init + pack init + packer-ReLU config + one data-format reconfig, and the ring handshakes, paid **once per compute thread per call**. At Phase 0 there is one call per thread, so the cost is paid C times per core per op. |
| `interleaved_streaming` | **deferred**: TARGET's `{DRAM, interleaved}` and `{L1, interleaved}` are outside PHASE0 by the specification's decision (no traced call has them). | both operands `{memory_config ∈ {DRAM, L1}, sharding: interleaved}` | per core: `core_tiles` from `ttnn.split_work_to_cores(grid, num_tiles)`; per thread one block of `neo_block_tiles` per helper call; ring depth `ring_depth_blocks · neo_block_tiles · C` | minimum at the DRAM boundary: `a` read once, `b` read once, `a` written once (page i is written only by the core that read it, so in-place is safe) | NoC transactions in flight and ring handshakes. Reachable: the compute kernel is unchanged (same helper call on normally allocated rings). Only the reader (implicit-sync tagged `noc.async_read` per tile from two TensorAccessors) and the writer (tagged `noc.async_write` to `a`'s accessor) replace the credit-only kernels, and both hand the tag back (`noc_async_*_set_trid(0, …)`). The plan there is (1, 1, 1) for DRAM (bound by data movement). |
| `respread_narrow_shard_grid` | **deferred**: it moves `a`, `b` and the result over the NoC (3 × tensor bytes) to parallelize one FPU add per tile, while the resident regime moves 0 B. The traced calls' grids (2 cores) and the suite's grids (≤ 32 cores) are the callers' own placements. | sharded operands whose shard grid engages fewer clusters than the device has | interleaved-style per-core blocks over the whole grid | +2 tensor reads and +1 tensor write over the NoC above minimum | as `interleaved_streaming`. Reachable: a `TensorAccessor` addresses sharded tensors by page id, so it is the interleaved reader/writer run on sharded operands. |
| `staged_shard` | **deferred**: only reachable if a borrowed ring is refused for its extent (caps row 3). No Phase 0 cell or traced call comes near it (largest 802,816 B). | `shard_tiles · tile_bytes` above the device's accepted ring extent | `neo_block_tiles` blocks staged into allocated rings | +2 local L1→L1 reads and +1 write per tile (self-aimed NoC) | handshakes per staged block. Reachable: same compute kernel; reader/writer use a local accessor on the shard. |
| `temporary_output_then_copy` | **rejected**: superseded by `sharded_resident`, which packs into `a`'s own shard. It needs a tensor-sized temporary (the spec's GUIDANCE: no room in L1) and either a second dispatch (forbidden) or a second pass. | — | — | +1 tensor write + 1 tensor read + 1 tensor write | — |
| dependent-axis split + cross-core combine | **n/a**: the op has no dependent axis | — | — | — | — |

Regime selection (host, after validation): `sharding == "height"` selects `sharded_resident`. Phase 0 admits no other placement, so a single program builder exists. Regime-pinned tests: covered by the acceptance cells (all `sharded_resident`). The C-clamp sub-case (`compute_threads = 2` on a 2-tile shard) is pinned by the acceptance entry `(1, 1, 32, 64)`.

### Traffic ranking

| Candidate split | DRAM crossings | Cross-core (NoC) | Core-local L1 | Rank |
|-----------------|----------------|------------------|---------------|------|
| by rows, along the caller's height shards, consumed in place through borrowed rings (`sharded_resident`) | 0 | 0 | a: 1 read + 1 write; b: 1 read | **1 (chosen; cheapest possible)** |
| by rows, shards staged through allocated rings (`staged_shard`) | 0 | 0 remote; 3 × tensor self-aimed transfers | + 3 × tensor copy | 3 |
| re-spread over idle clusters (`respread_narrow_shard_grid`) | 0 | 3 × tensor bytes for moved tiles | as above | 2–3 (depends on fraction moved) |
| interleaved operands (`interleaved_streaming`, different placement) | a: 2 (read + write), b: 1 | 0 extra | ring staging | minimum for that placement |
| split a dependent axis + combine | n/a: no dependent axis | — | — | — |

Operand-reuse check (chosen split = rows): `a` varies along rows, and so does `b`. Neither is reuse-shared, so no broadcast row exists.
Stall-shadow check: the only waits are compute on the reader's credits (posted at kernel start, C calls per ring, no transfer) and the writer on compute's pushes (the writer has nothing else to do). No independent work exists to move into either window.

### Block schedule

```cpp
// per engaged core k (core_tiles_k > 0); per compute thread t < compute_threads
for (uint32_t block_idx = 0; block_idx < num_blocks_this_thread; ++block_idx) {   // Phase 0: num_blocks_this_thread = 1
    credit_shard_block(block_idx);     // reader: credit a's and b's resident tiles (no bytes move)
    add_relu_block(block_idx);         // compute: a + b on the FPU, ReLU + round in the packer, into a's own tiles
    release_result_block(block_idx);   // writer: acknowledge the packed tiles (no bytes move)
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `credit_shard_block` | thread t's share `thread_tiles_t = core_tiles_k / C + (t < core_tiles_k % C)` tiles of each input ring | both shards (caller-owned) | one `reserve_back(thread_tiles_t)` + `push_back(thread_tiles_t)` per input ring per compute thread (each call credits one thread's tile counter, then rotates), i.e. 2·C calls per core. No NoC transfer, barrier or tag. |
| `add_relu_block` | `IterationShape::tiles(thread_tiles_t).block_size(neo_block_tiles)` | both shards; the result overwrites `a`'s tile k right after tile k is unpacked | `compute_kernel_hw_startup` once per kernel. Chain init (FPU add init, pack init, packer-ReLU config, one format reconfig) once per helper call, i.e. once per thread. Wait/pop/reserve/push once per block (`PerBlockSize`). One DST acquire/release per tile (helper-internal). |
| `release_result_block` | `thread_tiles_t` tiles of `cb_result_shard` | `a`'s shard, now holding the result | one `wait_front(thread_tiles_t)` + `pop_front(thread_tiles_t)` per compute thread, i.e. C calls per core |

Calls for a thread with `thread_tiles_t == 0` (only when `core_tiles_k < C`) are **skipped** in all three kernels. Shares are non-increasing in t, so only trailing calls are skipped and the per-call tile-counter rotation stays aligned.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Thread plan (1, 4, 1) | compute-bound reasoning assumes the NEO pipeline, not L1 bandwidth for 4 NEOs × (2 unpacks + 1 pack) per tile, is the limit | (1, 1, 1) and (1, 2, 1) on `ttsim-perf` (one constant: `THREAD_PLAN`) |
| Overlap: `neo_block_tiles` = whole share | one block per thread means its wait covers the whole share before the first unpack, and the writer acknowledges only at the end. Credits post immediately, so the expected loss is ~0. | `neo_block_tiles = 8` and `= 1` (`PerBlockSize` walk unchanged) |
| Packer ReLU (`PackRelu::Zero`) | relies on the packer's clamp on 32-bit DEST at this base. The chain's SFPU path is the alternative. | `eltwise_chain(BinaryFpu<Add>, Relu<>, PackTile)`: one SFPU pass per tile, measured against the packer clamp (catalog `compute_fusion`: never spend the SFPU on what another unit does) |
| Grid synchronization | does not apply: no collective, multicast or barrier; occupancy is the caller's shard grid | — |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `a`, `b` shards in each core's L1 | bfloat16 tiles, ring order = shard tile order (row-major) | `ttnn.experimental.dfb_spec_from_sharded_tensor("cb_a_shard", a, borrowed_from="a")`, likewise `cb_b_shard` on `b` and `cb_result_shard` on `a`: zero copy | **No NoC re-read through a `TensorAccessor`.** The kernels bind no tensor accessor; the tensor parameters are used through `borrowed_from` only (`tt_metal/impl/metal2_host_api/program_spec/validation/resource/tensor.cpp:93-105`). |
| reader → compute | — (credits only) | explicit sync: `reserve_back`/`push_back` per compute thread's share, rings listed in `disable_dfb_implicit_sync_for` | no NoC transfer, so no transaction tag is set and none needs handing back |
| compute | unpack bf16 to srcA/srcB → FPU add into DEST (fp32 when `fp32_dest_acc_en`) → packer ReLU (if `pack_relu`) → pack rounds once to bf16 | `compute_kernel_lib::add<…>` | the value stored is the sum rounded once, then clamped (DEFINITION, R6); the clamp commutes with the rounding |
| compute → writer | — (credits only) | explicit sync: `wait_front`/`pop_front` per compute thread's share | the packed tiles are already in place in `a`'s shard |
| result | `a`'s buffer | `generic_op` returns a handle; the op returns `input_tensor_a` | `b` is never written (R5): its ring has no producer that writes bytes |

TARGET placement values as regimes:

| `memory_config` × `sharding` | Logical shard it realizes | Cost class | Status |
|------------------------------|---------------------------|-----------|--------|
| L1 × height | split of the independent `rows` axis, pinned by the caller | built (Phase 0) | `sharded_resident` |
| L1 × interleaved | `rows`×`cols` tiles dealt by `split_work_to_cores` | knob-turn on the compute side; new reader/writer (tagged NoC transfers) | `interleaved_streaming`, deferred |
| DRAM × interleaved | same | same | `interleaved_streaming`, deferred |
| DRAM × height | INVALID (spec) | — | — |

Other TARGET axes (refinements, none changes the scheme):

| Axis value | Change |
|------------|--------|
| `activation = None` | knob-turn: CT arg `pack_relu = 0` (`PackRelu::Disabled`) |
| `fp32_dest_acc_en = False` (bf16) | knob-turn: `enable_32_bit_dest=False`. Rounding still happens once, at DEST. |
| `dtype = float32` | ring formats follow the tensor. The float32 numeric entry is 0 ulp: the refinement checks the FPU add against it, and where it is not exact switches the one helper call to `binary_sfpu<AddBinary, …>` (`ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp:20`; the raw `add_binary_tile` is exact in float32 on the simulator). That path needs `UnpackToDest` for 32-bit pages and `DataFormatReconfig::Disabled`. EXCLUSIONS gains `{float32, fp32_dest_acc_en=False}` (R1). |
| `alignment = w_non_aligned`, `rank ∈ {2, 3}` | knob-turn: the gate only. Tile padding is computed on and written back (GUIDANCE: only logical elements are compared); the tile-row formula is already per image with `ceil`. |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one tile (the ring entry). A core's block is all `core_tiles_k` tiles of its shard. |
| Grid | the caller's shard grid: `cores = ttnn.corerange_to_cores(shard_spec.grid, None, row_wise = shard_spec.orientation == ROW_MAJOR)`. The work unit's `target_nodes` = the cores with `core_tiles_k > 0` (R8: shards that hold nothing get no kernels). |
| Per-core work | `core_tiles_k = clamp(tensor_row_tiles − k·shard_row_tiles, 0, shard_row_tiles) · tensor_col_tiles` (RT arg `core_tiles` on all three kernels) |
| Remainder | partly filled last shard (R8): smaller `core_tiles_k`, same ring capacity. `tensor_row_tiles = prod(shape[:-2]) · ceil(H/32)`, `tensor_col_tiles = ceil(W/32)`: per image, `ceil`, never `floor`. |
| Bound | **compute**: the sharded regime moves 0 B over the NoC and DRAM, so the per-tile cost (2 unpacks, 1 FPU add, 1 pack with clamp) on the NEO pipeline is the only work. |
| Thread plan | `THREAD_PLAN = (1, 4, 1)` (one named host constant). C = 4 splits the compute-bound tile walk over all 4 NEOs. R = W = 1 because the data-movement threads move no bytes, only C credit calls per ring; more harts buy nothing. Effective `compute_threads` = the clamp in Mechanism caps (4 on every Phase 0 shard except a 2-tile one, where it is 2). The reader, compute and writer take `reader_threads`, `compute_threads`, `writer_threads` as CT args and size their loops from them and `get_my_thread_id()`. |
| Unit dealt to compute threads | the ring entry itself (one tile), round-robin: thread t gets tiles t, t+C, …. The deal may stop anywhere: no padded round, no whole-round split. |
| Tests the plan needs | (1) **every thread ran**: a test-only keyword of the internal program builder (not of `add_`) makes odd compute threads subtract instead of add. The test asserts tile k equals relu(a+b) for k mod C even and relu(a−b) for k mod C odd, on a shard with ≥ 2·C tiles and C = 4. (2) **one device session**: `add_` once after another program (e.g. a first `add_` call) and once before a tt-metal program (`ttnn.to_layout(ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)`), both in one session. (3) **R8**: operands built as ResNet-50's test builds them: `[1,1,784,512]` in 2 shards of 416 rows, `[1,1,196,1024]` in 2 shards of 128 rows, and a 32-core grid with 32-row shards where most shards are empty. Each checks values and R4/R5. (4) The ring-extent risk: one shard above 1,048,560 B (`[1,1,3136,256]` bf16 on one core, 1,605,632 B), run when both operands fit the device's L1. |

Worked Phase 0 numbers (quasar, 8×4 grid):

| Call | Cores engaged | `shard_tiles` | `compute_threads` | Per-thread share (`neo_block_tiles`) |
|------|---------------|---------------|-------------------|--------------------------------------|
| suite `[1,1,32,64]` / `[1,1,17,64]` | 1 | 2 | 2 | 1 (1) |
| suite `[1,1,784,512]` (25 tile rows dealt to 25 cores) | 25 | 16 | 4 | 4 (4) |
| suite `[1,1,3136,256]` | 14 | 56 | 4 | 14 (14) |
| traced layer 1 `[1,1,3136,256]`, 2 cores | 2 | 392 (802,816 B) | 4 | 98 (98) |
| traced layer 2 `[1,1,784,512]`, shards of 416 rows | 2 (208 + 192 tiles) | 208 | 4 | 52 / 48 (52) |
| traced layer 3 `[1,1,196,1024]`, shards of 128 rows | 2 (128 + 96 tiles) | 128 | 4 | 32 / 24 (32) |
| suite `[1,1,32,8192]` | 1 | 256 | 4 | 64 (64) |

## Circular Buffers

Dataflow buffers (Metal 2.0); "Index" is the ring's `unique_id`. All three are **borrowed**: they own no L1.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_a_shard` | `"cb_a_shard"` (`borrowed_from="a"`) | `a`'s tile page | `shard_tiles` (= `neo_block_tiles · compute_threads`) | spans `rows` (`block_row_tiles`), `cols` (`block_col_tiles`) and the tile walk: the shard is the block | `a.dtype` (bf16) | reader (credit only) | compute | reader side explicit (`disable_dfb_implicit_sync_for`); compute side explicit (always) | whole program |
| `cb_b_shard` | `"cb_b_shard"` (`borrowed_from="b"`) | `b`'s tile page | `shard_tiles` | as above | `b.dtype` (bf16) | reader (credit only) | compute | as above | whole program |
| `cb_result_shard` | `"cb_result_shard"` (`borrowed_from="a"`) | `a`'s tile page | `shard_tiles` | as above. Same bytes as `cb_a_shard`: in-place transform with real credit on both rings. | `a.dtype` (bf16): the caller's tensor (R4); the spec's single rounding at store | compute | writer (credit only) | compute explicit; writer side explicit (`disable_dfb_implicit_sync_for`) | whole program |

Sync counts per ring per core: producer pushes Σₜ `thread_tiles_t` = `core_tiles_k`, and the consumer waits/pops the same, split per thread identically (thread t: `thread_tiles_t` in one call at `neo_block_tiles ≥ thread_tiles_t`).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `credit_shard_block` | `thread_tiles_t` per compute thread | raw DFB credit API (no helper credits a resident ring) | — | `cb_a_shard`, `cb_b_shard`: `thread_tiles_t` pushed to thread t's counter | inputs credited, no byte written |
| 2 | `add_relu_block` | `IterationShape::tiles(thread_tiles_t).block_size(neo_block_tiles)` | `compute_kernel_lib::add` | `cb_a_shard`, `cb_b_shard`: wait/pop `thread_tiles_t` (`PerBlockSize`, `InputTileMapping::Block`) | `cb_result_shard`: reserve/push `thread_tiles_t` (`PerBlockSize`), `PackRelu::Zero` iff `pack_relu` | inputs popped; `a`'s tiles hold the result |
| 3 | `release_result_block` | `thread_tiles_t` | raw DFB credit API | `cb_result_shard`: wait/pop `thread_tiles_t` | — | every counter balanced (kernel-exit drain passes) |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| host: borrowed rings | host | `ttnn.experimental.dfb_spec_from_sharded_tensor(unique_id, tensor, num_entries=0, borrowed_from=…)` | `ttnn/cpp/ttnn-nanobind/program_specs.cpp:26-84`; validation `tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp:40-95` | `num_entries=0` → pages per shard | — | — | none (pinned by the shard) |
| host: dispatch | host | `ttnn.generic_op([b, a], spec, run_args, {"b": 0, "a": 1})` | `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:34-48,70-80` | `a` last (in-place output) | — | — | — |
| host: thread plan | host | `ttnn.experimental.KernelSpec(num_threads=…)`, STRIDED bindings `producer_of`/`consumer_of` | `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py:106-145` | R=1, C=`compute_threads`, W=1 | — | — | `THREAD_PLAN` |
| host: explicit rings on DM kernels | host | `DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=[…])` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:434-445`; usage `toy_thread_plan_program_artifacts.py:63-69` | reader: `["cb_a_shard","cb_b_shard"]`; writer: `["cb_result_shard"]` | — | — | — |
| `credit_shard_block` | raw_api | `DataflowBuffer::reserve_back` / `push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-244`; Quasar DM path `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:362-415` (one call credits the current thread's counter, then rotates: `:222-225`) | `thread_tiles_t` | — | `cb_a_shard`, `cb_b_shard` | — |
| `add_relu_block` (boot) | helper prerequisite | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | `(cb_a_shard, cb_b_shard, cb_result_shard)` | — | — | — |
| `add_relu_block` | helper | `compute_kernel_lib::add<input(a, PerBlockSize, PerBlockSize, Block), input(b, PerBlockSize, PerBlockSize, Block), output(result, PerBlockSize, PerBlockSize, Enabled, Direct, DestAccumulation::Disabled, L1Accumulation::Disabled, pack_relu ? PackRelu::Zero : PackRelu::Disabled)>(shape, a, b, out)` | decl `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:45-52`; impl `convenience.inl:9-14`; policies `eltwise/api/chain.hpp:233-242`; `IterationShape` `chain.hpp:146-170`; `output(…, PackRelu)` `chain.hpp:449-460`; `PackRelu` `chain.hpp:315`; packer ReLU emitted `eltwise/core/chain.inl:1209-1214,3026-3027` → `pack_relu_config` `tt_metal/hw/inc/api/compute/pack.h:201` → `tt_metal/hw/ckernels/quasar/metal/llk_api/llk_pack_common_api.h:198-200` | `IterationShape::tiles(thread_tiles_t).block_size(neo_block_tiles)` | `cb_a_shard`, `cb_b_shard` | `cb_result_shard` | `neo_block_tiles` (`block_size`) |
| `release_result_block` | raw_api | `DataflowBuffer::wait_front` / `pop_front` | `dataflow_buffer.h:245-246`; Quasar DM path `dataflow_buffer.inl:419-470` | `thread_tiles_t` | `cb_result_shard` | — | — |

Helpers considered and rejected for the two raw_api rows:

| Raw entry | Candidate helper | Mismatch (file:line) | Reason |
|-----------|------------------|----------------------|--------|
| `credit_shard_block`, `release_result_block` | `ttnn/cpp/ttnn/kernel_lib/dfb_helpers_dataflow.hpp` | `:14-18` declares only `get_tile_r_dim` / `get_tile_c_dim` | it has no credit or ring-lifecycle function |
| same | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp` | `:9` "Local L1 -> L1 copy helpers: self-aimed NoC reads" | it moves bytes, which is exactly the traffic the borrowed ring removes. These operations transfer nothing. |
| same | reader helpers `reduce_helpers_dataflow.hpp`, `tilize_helpers_dataflow.hpp` | — | scaler / tilize producers, not credit of resident data |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `add_relu_block` | FPU add (`BinaryFpuOp::Add`) | `cb_a_shard`: All | `cb_b_shard`: All | `BroadcastDim::None` |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| In-place hazard: pack overwrites a tile not yet read | `cb_a_shard` and `cb_result_shard` are the same bytes | thread t's pack of its i-th tile targets exactly the address its unpack of tile i read (both rings STRIDED over the same base and stride). Pack i follows math i follows unpack i. Unpack of i+1 reads a different tile. No other thread touches those addresses. Real credit on both rings (Rule 3.2). |
| Credit rotation misaligned with compute threads | a DM-side STRIDED call credits one thread's counter and rotates (`dataflow_buffer.inl:222-225`). A skipped middle call shifts every later share. | shares are non-increasing in t, and only trailing zero shares are skipped. Reader and writer use the same share formula as compute. |
| `compute_threads` not dividing `shard_tiles` | e.g. a 2-tile shard (acceptance entry `(1,1,32,64)`) | the clamp in Mechanism caps; the acceptance test pins C = 2 |
| Empty and partly filled shards (R8) | the model's grids hold more rows than the tensor | `core_tiles_k` from the clamp formula; cores with 0 tiles are left out of `target_nodes`; the op test file reproduces the model's grids (Work Distribution, test 3) |
| `b`'s shard on different cores than `a`'s | the borrowed `b` ring must be on the same nodes | identical shard spec required in validate (step 6), else `NotImplementedError` |
| Ring extent: target page (1,048,560 B) vs code (uint32 units) | a shard over the page's figure may be mis-addressed if the page is right | Phase 0 shards ≤ 802,816 B. Work Distribution test 4 settles it; `staged_shard` is the fallback row. |
| Packer ReLU on ttsim | an unmodelled packer clamp would leave negatives (the `all_negative` regression case shows it as non-zero output) | the acceptance and regression cases cover it. If the clamp is not applied, the one-element change is the SFPU `Relu<>` in the chain (perf-lamp alternative); it is not a scheme change. |
| `fpu_math_fidelity` | set on every compute kernel | always map `cfg.math_fidelity` into `ComputeHardwareConfig` |
| Returning a new tensor object | R4 accepts a handle, but the strongest form is the object | return `input_tensor_a` itself |
| DM kernels' transaction tag | — | these kernels issue no NoC transfer and never set a tag. The one-session tests (Work Distribution, test 2) confirm it. |
| A single thread could pass every value test | the golden suite passes with C = 1 | Work Distribution test 1 tags output by thread |

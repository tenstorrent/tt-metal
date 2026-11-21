# Operation Design: sigmoid

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (element-wise unary, SFPU transcendental) |
| Goal | `torch.sigmoid` on a Quasar tensor in one `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec`. Every cluster of the grid works on a contiguous run of tiles under the thread plan (2 readers, 4 NEOs, 2 writers). The SFPU body is accurate to fp32, so a bfloat16 result stays within the ULP bound of the specification's numeric entry. |
| Math | `output[i] = 1 / (1 + exp(-input[i]))` for every element `i` (tile padding is computed like data and never compared) |
| Specification | `eval/specs/sigmoid/spec.py` (governs where this design disagrees) |
| References | `.claude/references/targets/quasar.md`, `.claude/references/backends/ttsim.md`, `.claude/references/quasar_op_examples/toy_thread_plan/` (host split, thread plan, implicit-sync reader/writer: the code to follow), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp`, `tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h`, `eval/op_template.py`, `eval/memory.py` |
| Catalog entries used | `compute_fusion` (keep the whole expression in DEST in one SFPU pass: no intermediate buffer), `compute_block_size` (one helper call per NEO share, not per tile: the init and reconfig are paid once), `double_buffer` (two sync blocks of depth per ring). |

### Files the implementer writes

| File | Role |
|------|------|
| `ttnn/ttnn/operations/sigmoid/__init__.py` | re-exports `sigmoid, SUPPORTED, EXCLUSIONS, INPUT_TAGGERS, validate, default_compute_kernel_config` |
| `ttnn/ttnn/operations/sigmoid/sigmoid.py` | entry point, `default_compute_kernel_config()`, `INPUT_TAGGERS = AXES` (imported from the spec), `SUPPORTED` (= `PHASE0`), `EXCLUSIONS = []`, `validate()` |
| `ttnn/ttnn/operations/sigmoid/sigmoid_program_artifacts.py` | the `ProgramSpec`, the knob constants, the split |
| `ttnn/ttnn/operations/sigmoid/kernels/{reader,compute,writer}.cpp` | three kernels |
| `tests/ttnn/unit_tests/operations/sigmoid/test_sigmoid_threads.py` | the plan tests (Work Distribution, last row). `test_sigmoid.py` is the harness's and is not edited |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters, fixed on this part. Queried with `device.compute_with_storage_grid_size()` (the probe returned `8-4`) | target page; `tests/ttnn/unit_tests/operations/sigmoid/probes/probe_001.py` |
| L1 per core | 4,194,304 B per cluster, shared by its 4 NEOs and 8 DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` = 3,963,328 B − 0 on this target. Phase 0 needs no predicate (the footprint is fixed, see `l1_ledger.md`) | probe_001 |
| Alignment | L1 16 B, DRAM 64 B (`ttnn.get_l1_alignment()` / `ttnn.get_dram_alignment()`) | target page, probe_001 |
| Host program | `ttnn.ProgramSpec` (Metal 2.0, `ttnn.experimental.*`), launched by `ttnn.generic_op`. Reference: `.claude/references/quasar_op_examples/` (README first, then `toy_thread_plan/toy_thread_plan_program_artifacts.py`) | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1, 2, 4}) and 6 user DM cores (R + W ≤ 6). On STRIDED rings R \| C and W \| C. `KernelSpec(num_threads=...)` sets each count | target page, *Thread plan* |
| Buffers | DataflowBuffer (DFB) rings: 32 per cluster, ≤ 24 active per NEO, each ≤ 1,048,560 B on the compute side, exactly one producer kernel and one consumer kernel | target page, *Cluster model* |
| Unavailable here | **Metal 1 `ttnn.ProgramDescriptor` and C++ `ProgramFactoryConcept`**: refused at kernel creation. **`bfloat8_b`/`bfloat4_b`**: there is no path for them. **`add_unary_tile` / `sub_unary_tile` / `div_unary_tile` / `rsub_unary_tile`** (and with them the chain's `AddUnary`/`SubUnary`/`DivUnary`/`RsubUnary`): compiled out on Quasar (`tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:34-58,76-152`), so a "+1" cannot be a scalar chain element. **`UnpackToDest` of a bfloat16 page into a 32-bit DEST**: not used; the input keeps the default unpack mode. **The eltwise chain's operand reconfig under `UnpackToDest`**: not used. **`pack_init` re-aim between output rings**: not needed, because there is one output ring | target page, helper headers |

Every worked number below is for `quasar` (32 clusters, bfloat16 tile = 2,048 B). Every run-time predicate and size uses the query, never the number.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | Phase 0: bfloat16, TILE, DRAM or L1 interleaved, rank 2–4, any alignment | — | tensor binding `input` |
| `compute_kernel_config` (keyword-only) | `ttnn.ComputeConfigDescriptor` or `None` | no | Phase 0: `fp32_dest_acc_en=True`. `math_fidelity` and `math_approx_mode` take any value and are never gated (R3) | `None` → `default_compute_kernel_config()` = `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)` (R2: defined once, exported) | maps onto the compute `KernelSpec.hw_config` (below) |

**Compute config → `ComputeHardwareConfig`** (`ttnn/cpp/ttnn-nanobind/program_spec.cpp:519-556`). The spec's GUIDANCE "pass the config as-is" is a `ProgramDescriptor` idiom. Under `ProgramSpec` the same fields are carried field for field:

| `ComputeConfigDescriptor` field | `ttnn.experimental.ComputeHardwareConfig` argument |
|---|---|
| `math_fidelity` | `fpu_math_fidelity` (always set explicitly) |
| `fp32_dest_acc_en` | `enable_32_bit_dest` |
| `dst_full_sync_en` | `double_buffer_dest = not dst_full_sync_en` (tt-metal's own mapping, `compute_kernel_config.cpp`); `block_tiles` follows the DEST capacity of both fields (added by the verifier: the first build dropped the field) |
| `math_approx_mode` | `sfpu_precision_mode = ttnn.Precision.Approximate if math_approx_mode else ttnn.Precision.Precise` (the sigmoid body below takes the accurate path in either case; the field is passed through, not gated) |

**`validate()` order** (R5, R7): build the axes from `eval.memory.memory_axes(input_tensor)` (`memory_config`, `sharding`), `INPUT_TAGGERS` (`alignment`, `rank` over `(tuple(input_tensor.shape),)`), `dtype`, `layout` and the resolved config's `fp32_dest_acc_en`. Then raise `NotImplementedError` (via `_op_contract`) for any axis value outside `SUPPORTED` or any `EXCLUSIONS` match. R1 (float32 with `fp32_dest_acc_en=False` refused) holds automatically in Phase 0, because float32 is outside `SUPPORTED`. When float32 joins, R1 becomes a permanent `EXCLUSIONS` entry `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}`. There are no index axes, so nothing is canonicalized. `validate()` reads neither `math_fidelity` nor `math_approx_mode`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 2, 3 or 4. Any H, W (padding computed) |
| Dtype | bfloat16 (Phase 0) |
| Layout | TILE |
| Memory | interleaved, DRAM or L1 (read through `TensorAccessor` in both cases) |

### Output

| Property | Value |
|----------|-------|
| Shape | = input shape (same padded shape) |
| Dtype | = input dtype |
| Layout | = input layout |
| Memory | `ttnn.allocate_tensor_on_device(input_tensor.spec, device)`: the spec carries the input's memory config, so the output is DRAM for a DRAM input and L1 for an L1 input, interleaved (R6) |

## Blocking Model

### Axes

The tile index space of a TILE tensor is `(batch = prod(leading dims), h_tiles = ceil(H/32), w_tiles = ceil(W/32))`, and each image is tile-padded on its own. The interleaved buffer stores those tiles as pages `0 .. num_tiles-1` in exactly that row-major order. The op is element-wise, so every axis is independent. The three axes fold into one contiguous **page** axis, because the op treats them identically and they are contiguous in page order. The page axis is the axis that is blocked and split.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (product of leading dims, 1 for rank 2) | independent: each output element reads only its own input element | folded into `page` | folded | `num_tiles = input_tensor.buffer_num_pages()` (host) | via `page` | knob-turn |
| `h_tiles` = `ceil(H/32)` per image | independent (same reason) | folded into `page` | folded | as above | via `page` | knob-turn |
| `w_tiles` = `ceil(W/32)` | independent (same reason) | folded into `page` | folded | as above | via `page` | knob-turn |
| `page` (flat tile index, `num_tiles = batch * ceil(H/32) * ceil(W/32)`) | independent | `core_tiles`: the core's contiguous run of pages = **its block** (one block per core, `num_blocks_this_core = 1`) | the whole per-core assignment: `rounds_this_core * C` pages, with only the core holding the final round shorter | host split (`split_rounds`, Work Distribution) → RT args `num_tiles`, `start_id` per core | all clusters of `device.compute_with_storage_grid_size()`, contiguous runs, dealt in whole rounds of C pages | knob-turn (grid size, round size) |
| `neo` (a core's run dealt over its C NEOs) | independent | `neo_tiles` = `core_tiles / C + (t < core_tiles % C)` for NEO `t` | C = 4 (the plan) | `THREAD_PLAN` host constant → CT arg `compute_threads` | NEO `t` owns core-local pages `t, t+C, ...` (STRIDED rings deal round-robin) | knob-turn (`THREAD_PLAN`) |
| `sync_block` (pages per DEST window and per ring handshake, inside a NEO's share) | independent | `block_tiles` | 4 at `fp32_dest_acc_en=True` (DEST holds 4 fp32 tiles in half sync), 8 at 16-bit DEST; doubled under `dst_full_sync_en=True`: the coarsest window DEST holds | host `DEST_TILES[(fp32_dest_acc_en, dst_full_sync_en)]` → CT arg `block_tiles` (the chain also clamps, see Mechanism caps) | — (inside a NEO) | knob-turn |
| element within a tile (32 × 32) | independent | whole tile | 1 tile (the SFPU walks all 32 vector rows; no axis can be scoped because the whole tile is live data or padding) | — | — | — |

`core_tiles` is not split further: a block is the whole assignment, run as one pipeline pass (one `compute_kernel_hw_startup`, one chain call per NEO, one SFPU init per NEO). The rings stream it, so L1 does not grow with `core_tiles` (`l1_ledger.md`). Splitting it into several blocks would only repeat the init.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `BUFFER_DEPTH` (host constant; `entries_per_neo = BUFFER_DEPTH * block_tiles`) | 2 | the reader fills sync block k+1 of a NEO while the NEO computes block k |
| `cb_output_tiles` | `BUFFER_DEPTH` (same constant) | 2 | the NEO packs block k+1 while the writer drains block k |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| eltwise chain block walk (`chain.inl:3339-3351`) | `block_tiles` ≤ `chain_max_block_v` (DEST capacity / lane width: 4 at fp32 half-sync, 8 at 16-bit) | the chain lowers `block_size` to `chain_max_block` itself. The host derives `block_tiles` from the same DEST table, so the clamp never fires at Phase 0 values. Keep `BlockTailSync::ValidTiles` (the default), because `FullBlock` asserts on an over-size block | n/a (clamped) |
| DFB ring capacity (target page: ≤ 1,048,560 B compute side) | `BUFFER_DEPTH * block_tiles * C * page_size` per ring | host assert `num_entries * page_size <= 1_048_560` | program build fails |
| Thread plan legality (target page) | (R, C, W) | host check `C in {1,2,4}`, `R+W <= 6`, `C % R == 0`, `C % W == 0` (as `validate_thread_plan` in `toy_thread_plan_program_artifacts.py`) | launch failure |
| Quasar exp range reduction `_sfpu_round_to_nearest_int32_` (`ckernel_sfpu_exp.h:32-37`, "valid for \|z\| < 2^22") | magnitude of the exp argument | the sigmoid element clamps `x` to `[-88, 88]` before `exp(-x)`. sigmoid is saturated in fp32 beyond that (`1 - e^-88` rounds to 1, and `e^-88` sits in the spec's tiny-reference band) | — |
| `approx_recip` (`ckernel_sfpu_recip.h:27`: returns ±0 for x ≥ 2^126) | `1 + exp(-x)` for x < −87.3 | none needed: the result 0 is inside the spec's band `ref_abs ≤ 2.35e-38 → out_abs ≤ 2.35e-38` | — |
| compute kernel binary ≤ 24,576 B (target page) | code size of the unrolled SFPU body | the ITERATIONS loop of the sigmoid element is not force-unrolled beyond what the build fits (Key Risks) | link failure |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R-stream**: interleaved TILE, contiguous page runs per cluster in whole rounds of C, plan (2,4,2), fp32-accurate SFPI sigmoid in one fused chain | **built** (Phase 0) | `layout == TILE and sharding == "interleaved"` (DRAM or L1; any rank, alignment) | `core_tiles` per core (whole assignment). Inside it, NEO share `neo_tiles` in sync blocks of `block_tiles` | **Minimum** against DRAM (or the L1 banks for an L1 tensor): each input page crosses once, each output page once, with no cross-core traffic. Padding pages of non-aligned shapes are the tensor's own pages and are moved once like data | `core_tiles` = whole assignment: one `compute_kernel_hw_startup`, one SFPU init (programs `vConstFloatPrgm0`) and one chain setup per NEO per program. `block_tiles` = DEST capacity: one `tile_regs_acquire/commit/wait/release` and one ring wait/pop/reserve/push per 4 tiles instead of per tile. Domain: every Phase 0 cell, so no other regime applies there |
| **R-shard**: the same scheme with each ring backed on the core's own shard (zero-copy, no NoC read) for HEIGHT/WIDTH/BLOCK sharded inputs | **deferred**: `TARGET["sharding"]` is `["interleaved"]` only, so no cell needs it | `sharding != "interleaved"` | the whole resident shard (one block) | below the interleaved minimum: input and output never cross the NoC | knob-turn when TARGET grows. Every shard flavor cuts only independent axes, so none needs a combine. R-stream's kernels keep it reachable: compute is unchanged, and reader/writer become no-ops over a shard-backed ring |
| **R-rm-stream**: ROW_MAJOR input streamed as layout-blind 1,024-element entries (sigmoid is element-wise, so unpack and pack applied as inverse permutations leave values in place). The reader assembles entries from sticks (explicit sync, tagged transfers) | **deferred**: `ROW_MAJOR_LAYOUT` is in TARGET but not in PHASE0. It needs a new reader/writer pair and the R-stream compute unchanged | `layout == ROW_MAJOR` | `core_tiles` worth of sticks per core | minimum (each stick once each way) | as R-stream. Reachable because compute only sees entries |
| **R-tilize**: ROW_MAJOR via tilize → sigmoid → untilize inside the program | **rejected**: superseded by R-rm-stream (layout-blind), which reaches the same minimum with one compute phase instead of three and no tile intermediate buffers. This is a dead end, built instead of the right regime rather than under it | — | — | minimum, plus two extra compute phases and two intermediate rings | — |
| **R-dependent-split** (split a dependent axis + combine) | **rejected**: not applicable. The op has no dependent axis | — | — | — | — |
| **R-mcast** (broadcast a shared operand) | **rejected**: not applicable. The only operand varies along the split axis (operand-reuse check below) | — | — | — | — |
| **R-lut**: Quasar `sigmoid_tile` (LUT exp + LUT reciprocal, `ckernel_sfpu_sigmoid.h:29-48`) instead of the fp32-accurate body | **rejected for Phase 0**: the LUT reciprocal is "~7-bit mantissa" (`ckernel_sfpu_recip.h:20`), and on top of the LUT exp it cannot be shown to hold the numeric entry's 1-ULP bound on `[0,1]` inputs (the `uniform` regression case). R-stream's accurate body supersedes it for every graded cell | — | as R-stream | as R-stream | — (a precision choice; recorded as a perf lamp for a future approximate mode) |

Selection: one built regime, so there is no run-time predicate. Every other cell is refused by `validate()`.

### Traffic ranking

| Candidate split | DRAM / L1-bank crossings | Cross-core traffic | Transfer shape | Rank |
|---|---|---|---|---|
| flat `page` axis in contiguous runs (chosen) | input 1×, output 1× | none | whole 2 KiB pages, consecutive page ids → interleaved banks rotate, both DRAM channels busy | 1 |
| `h_tiles` / `w_tiles` / `batch` alone | input 1×, output 1× | none | same pages, but a 2-D or batch cut under-fills the grid on shapes with one image or one tile-row (e.g. `(1, 80, 400)`: 3 tile-rows) | 2 (same bytes, worse balance) |
| any dependent-axis split | n/a (no dependent axis) | — | — | — |

Every split moves the minimum. The flat page split wins on balance (any `num_tiles` deals to the grid within one round). **Operand-reuse check:** the one operand (`input_tensor`) varies along `page`, so no core re-reads another's bytes, and no broadcast row exists. **Stall-shadow check:** the only waits are a NEO waiting on its reader and a writer waiting on its NEO. The work independent of a wait is the next sync block, and the ring depth of 2 already schedules it into the wait. Nothing else exists to lift.

### Block schedule

```cpp
// per cluster; each kernel realizes its part, all asynchronous through the rings
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core /* = 1 */; ++block_idx) {
    load_block(block_idx);      // reader threads p < R: pages start_id + p, p+R, ... < core_tiles → cb_input_tiles
    sigmoid_block(block_idx);   // NEO t < C: its neo_tiles pages, in sync blocks of block_tiles, one fused chain
    store_block(block_idx);     // writer threads w < W: cb_output_tiles → pages start_id + w, w+W, ...
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `load_block` | `core_tiles` pages, dealt round-robin over R reader threads | nothing (streams through `cb_input_tiles`) | one tagged `noc.async_read<TXN_ID>` per page (implicit sync: no reserve/barrier/push). One `noc_async_read_set_trid(0, noc)` per thread at kernel end |
| `sigmoid_block` | per NEO: `neo_tiles` pages as `IterationShape::tiles(neo_tiles).block_size(block_tiles)` | nothing (in-DEST fusion: copy → sigmoid → pack, no intermediate) | `compute_kernel_hw_startup` once per kernel. Copy init, SFPU init (`vConstFloatPrgm0 = 2.0`) and data-format reconfig once per chain call (hoisted by the chain, which is uniform). DEST acquire/commit/wait/release and ring wait/pop/reserve/push once per `block_tiles` |
| `store_block` | `core_tiles` pages, round-robin over W writer threads | nothing | one tagged `noc.async_write<TXN_ID>` per page. One `noc_async_write_set_trid(0, noc)` per thread at kernel end |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Thread plan (2,4,2) | The bound is argued from SFPU instruction count, not measured. | (1,4,1) and (1,1,1) on `ttsim-perf` (model cycles), `(1, 80, 8400)`. The plan is one constant |
| Whole-round split | Dealing rounds of C pages engages fewer clusters on tiny tensors (`(1, 80, 400)`: 39 pages → 10 clusters) | Split by pages over all 32 clusters (`split_work_to_cores(grid, num_tiles)`). Equal NEO load, more readers |
| `block_tiles` = DEST capacity | Per-tile sync is simpler | `block_tiles = 1` (PerTile policies) |
| Overlap: `BUFFER_DEPTH` = 2 | With 2 readers feeding 4 NEOs, one block of slack may starve a NEO | `BUFFER_DEPTH = 3` |
| Accurate SFPI body vs LUT `sigmoid_tile` | the LUT path is ~4 SFPU instructions per row vs ~35 | Only for a future approximate mode: measure the LUT path's ULP against the numeric entry before it is ever selected. Phase 0 cells must stay accurate |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM or L1 banks → `cb_input_tiles` | bfloat16 tile, 2,048 B page | reader thread `p` of R: `noc.async_read<NocOptions::TXN_ID>(acc_input, dfb_input, {.page_id = start_id + i}, {})` for `i = p, p+R, ... < num_tiles`. Implicit sync | the k-th read of thread p lands in NEO `(p + k·R) mod C`'s slice, which is the NEO that computes core-local page `p + k·R` |
| `cb_input_tiles` → DEST | bf16 in srcA → fp32 DEST (default unpack mode, `CopyTile`) | `eltwise_chain` `CopyTile<input(cb_input_tiles, PerBlockSize, PerBlockSize)>` | not `UnpackToDest` |
| DEST → DEST | fp32 | custom chain element `SigmoidAccurate<Dst::D0>` (API Mapping) | clamp to `[-88, 88]`, then `_sfpu_sigmoid_<DST_ACCUM_MODE>` |
| DEST → `cb_output_tiles` | fp32 → bfloat16 packed | `PackTile<output(cb_output_tiles, PerBlockSize, PerBlockSize)>` | |
| `cb_output_tiles` → DRAM or L1 banks | bfloat16 tile | writer thread `w` of W: `noc.async_write<NocOptions::TXN_ID>(dfb_output, acc_output, {}, {.page_id = start_id + i})`, `i = w, w+W, ...`. Implicit sync | |

**TARGET placement axes as regimes:**

| TARGET value | Regime | Cost class |
|---|---|---|
| `memory_config: DRAM` | R-stream | built |
| `memory_config: L1` | R-stream (same accessor path; output in L1 by spec) | built |
| `sharding: interleaved` | R-stream | built |
| `*_SHARDED` (not in TARGET) | R-shard | knob-turn when added (only independent axes are cut) |
| `layout: ROW_MAJOR` | R-rm-stream | new reader/writer, same compute |
| `dtype: float32` | R-stream with 4,096 B pages | knob-turn (page size from the tensor; ring bytes double) |
| `fp32_dest_acc_en: False` | R-stream, the body follows `DST_ACCUM_MODE` (`_sfpu_sigmoid_<false>`: `exp_21f` + one Newton step), `block_tiles` = 8 | knob-turn plus a precision check against the numeric entry |

Structural impossibilities: none to add (`INVALID = []` stands).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a **round** of C consecutive pages (C = the plan's compute threads). The block of a core is its run of whole rounds |
| Grid | `device.compute_with_storage_grid_size()` (8 × 4 on quasar) |
| Per-core work | `num_rounds = ceil(num_tiles / C)`. `ttnn.split_work_to_cores(grid, num_rounds, row_wise=True)` gives each core `rounds_this_core` and a start round (walk group 1 then group 2, as `split_tiles` in `toy_thread_plan_program_artifacts.py`). `start_id = start_round * C`, `num_tiles(core) = min(rounds_this_core * C, num_tiles - start_id)`. RT args `num_tiles` and `start_id` go to reader and writer, `num_tiles` to compute |
| Remainder | the round count rounds up, so only the core holding the last round gets `num_tiles % C` fewer pages. `num_tiles = input_tensor.buffer_num_pages()` already equals `batch * ceil(H/32) * ceil(W/32)` (per-image tile padding), and no `//` or `floor` appears in any count. Worked: `(1, 80, 8400)` → 3 × 263 = 789 pages → 198 rounds → 6 clusters × 7 rounds and 26 × 6 → 28 or 24 pages per cluster, the last cluster 21. `(1, 80, 400)` → 3 × 13 = 39 pages → 10 rounds → 10 clusters, 4 pages each, the last 3 |
| Bound | **compute**. Per tile the accurate body runs ~35 SFPU instructions on each of 32 vector rows (Cody–Waite reduction, degree-6 polynomial, exponent guards, LUT reciprocal + 2 Newton steps, clamp) against 2 KiB read + 2 KiB written |
| Thread plan | **(R, C, W) = (2, 4, 2)**, host constant `THREAD_PLAN = (2, 4, 2)` in `sigmoid_program_artifacts.py`. CT args `reader_threads`, `compute_threads`, `writer_threads` go to all three kernels, and loops are sized from them and `get_my_thread_id()`, never from `get_num_threads()`. Compute-bound, one input and one output → C = 4, R = W = 2 |
| Unit dealt to compute threads | the ring entry itself (one tile), dealt round-robin by the STRIDED rings: NEO `t` computes core-local pages `t, t+C, ...`, `neo_tiles = core_tiles / C + (t < core_tiles % C ? 1 : 0)`. No padding round is needed, because the dealing simply stops. Whole rounds are still dealt to cores, so every engaged cluster has ≥ C pages except the last one |
| Tests the plan needs | in `test_sigmoid_threads.py`. (1) **Every NEO ran:** a test-only CT arg `probe_tag_neo` (default 0, set only through a keyword of the program-artifacts function, never through the public signature) makes odd NEOs compute `sigmoid(-x)`. The test checks that core-local page `i` holds `sigmoid(-x)` iff `i mod C` is odd, on `(1, 3, 320, 320)` (300 pages). (2) **Plans:** the op correct at (1,1,1), (1,4,1), (2,4,2), (4,4,2) through the same keyword, on 1, 3 (< C), 39 and 789 pages. (3) **One device session:** in one test, `ttnn.add` (a tt-metal program) → `sigmoid` → `sigmoid` → `ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)` → compare. This shows the tag hand-back on both ends |

## Circular Buffers

DFB rings, STRIDED (the default access pattern), named by `unique_id` and in kernels as `dfb::<name>`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|----------|
| `cb_input_tiles` | 0 | `input_tensor.buffer_page_size()` (2,048 B bf16) | `BUFFER_DEPTH * block_tiles * C` = 2·4·4 = 32 | spans `sync_block` (×`block_tiles`) and `neo` (×C, capacity is per NEO). Streams over `page` / `core_tiles` | bfloat16 (tensor dtype) | reader | compute | reader: **implicit** (tagged read per entry, no reserve/push). Compute: chain wait/pop per block | whole program |
| `cb_output_tiles` | 1 | `output.buffer_page_size()` (2,048 B) | `BUFFER_DEPTH * block_tiles * C` = 32 | same as above | bfloat16 (output dtype) | compute | writer | compute: chain reserve/push per block. Writer: **implicit** (tagged write per entry, no wait/pop) | whole program |

Neither ring is listed in `disable_dfb_implicit_sync_for` (use `E.create_reader_datamovement_config()` / `E.create_writer_datamovement_config()` unchanged). Producer push count = consumer wait count: the reader posts one credit per page into NEO `t`'s slice, `neo_tiles` in all. NEO `t` waits and pops `neo_tiles` in total (blocks of `block_tiles`, last block `neo_tiles % block_tiles`, `ValidTiles`). It reserves and pushes the same `neo_tiles` on `cb_output_tiles`, and the writer drains exactly those.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `core_tiles` pages over R threads | no (dataflow; `TensorAccessor` + tagged `noc.async_read`) | `input_tensor` pages `start_id .. start_id+core_tiles-1` | `cb_input_tiles`, 1 entry per page | all `core_tiles` entries credited to their NEO slices |
| 2 | `sigmoid_block` | `IterationShape::tiles(neo_tiles).block_size(block_tiles)` per NEO | yes: `eltwise_chain(CopyTile, SigmoidAccurate, PackTile)` | `cb_input_tiles`, `block_tiles` per wait/pop, `neo_tiles` total | `cb_output_tiles`, `block_tiles` per reserve/push | input slice empty, output slice holds `neo_tiles` until drained |
| 3 | `store_block` | `core_tiles` pages over W threads | no (dataflow; tagged `noc.async_write`) | `cb_output_tiles` | output pages `start_id ..` | empty |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api (required by the chain's contract) | `compute_kernel_hw_startup(dfb::cb_input_tiles, dfb::cb_output_tiles)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:22-38` (caller owns it) | — | `cb_input_tiles` | `cb_output_tiles` | — |
| `sigmoid_block` | helper | `compute_kernel_lib::eltwise_chain(shape, CopyTile<input(dfb::cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize)>{dfb_in}, SigmoidAccurate<Dst::D0>{}, PackTile<output(dfb::cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{dfb_out})` | `chain.hpp:620` (entry), `:146-173` (`IterationShape::tiles(n).block_size(b)`), `:233-242` (policies), `:385-460` (`input`/`output`), `chain.inl:3339-3351` (block clamp) | `shape = IterationShape::tiles(neo_tiles).block_size(block_tiles)`, `InitReconfigOwner::Chain` (default) | `cb_input_tiles` | `cb_output_tiles` | `block_tiles` (CT arg); `neo_tiles` (from RT `num_tiles`, CT `compute_threads`) |
| sigmoid element | helper framework + raw SFPI body | `SigmoidAccurate<Slot>`, an op-local struct deriving `compute_kernel_lib::UnaryOp<SigmoidAccurate<Slot>, Slot>` (`chain.inl:634-648`). `init()` = `MATH(SFPU_UNARY_INIT_FN(sigmoid, sfpu::sigmoid_init, (false)))`, which programs `vConstFloatPrgm0 = 2.0` for the Newton reciprocal (`ckernel_sfpu_sigmoid.h:106-113`, `ckernel_sfpu_recip.h:56-61`). `exec_impl(slot_offset)` = `MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, <op-local functor>, (DST_ACCUM_MODE, 8), to_u32(Slot) + slot_offset, VectorMode::RC))` (`llk_math_eltwise_unary_sfpu_macros.h:35-38`; the functor must sit in `namespace ckernel::sfpu`). The functor, in each of its 8 iterations: `x = sfpi::clamp(dst_reg[0], -88.0f, 88.0f); dst_reg[0] = _sfpu_sigmoid_<DST_ACCUM_MODE>(x); dst_reg++` | `ckernel_sfpu_sigmoid.h:79-104` (`_sfpu_sigmoid_`: `_sfpu_exp_accurate_<true>` = `_sfpu_exp_fp32_accurate_`, `ckernel_sfpu_exp.h:228-284,304-307`; `_sfpu_reciprocal_<2>`, `ckernel_sfpu_recip.h:22-53`); `sfpi::clamp` as in `ckernel_sfpu_clamped_silu_glu.h:54` | `Slot = Dst::D0`, `lane_width` 1 | DEST | DEST | — |
| `load_block` | raw_api (dataflow, as the reference) | `noc.async_read<NocOptions::TXN_ID>(acc, dfb, {.page_id}, {})`; `noc_async_read_set_trid(0, noc.get_noc_id())` last | `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader_implicit.cpp` | `TensorAccessor(tensor::input)` | — | `cb_input_tiles` | `reader_threads` |
| `store_block` | raw_api (dataflow, as the reference) | `noc.async_write<NocOptions::TXN_ID>(dfb, acc, {}, {.page_id})`; `noc_async_write_set_trid(0, noc.get_noc_id())` last | `.claude/references/quasar_op_examples/toy_thread_plan/kernels/writer_implicit.cpp` | `TensorAccessor(tensor::output)` | `cb_output_tiles` | — | `writer_threads` |

**Helpers considered and rejected (for the `SigmoidAccurate` element, whose body is raw SFPI):**

| Candidate | File:Line of the mismatch | Concrete reason |
|---|---|---|
| chain `Sigmoid<Slot>` / `unary<Sigmoid>` | `eltwise/unary/activations.inl:31-34` → `tt_metal/hw/inc/api/compute/compute_kernel_api.h:163-166` (`#ifdef ARCH_QUASAR` → `calculate_sigmoid`) → `ckernel_sfpu_sigmoid.h:29-48` | on Quasar this is `SFPNONLINEAR EXP_MODE` followed by `SFPNONLINEAR RECIP_MODE`, both LUT approximations. The LUT reciprocal alone is "~7-bit mantissa" (`ckernel_sfpu_recip.h:20`), and the spec's numeric entry (`NUMERIC_RESOLVED`, "replace" gate in `eval/metrics.py:113-122`) requires ≤ 1 bf16 ULP on `[0, 1]` inputs (`uniform` regression) and ≤ 2 ULP elsewhere |
| chain `Negative → Exp<Approx::Exact> → AddUnary(1.0) → Recip` (all-helper composition) | `binop_with_scalar.h:34-58` (`add_unary_tile` under `#ifndef ARCH_QUASAR`) | there is no scalar add on Quasar, so the "+1" has no chain element. Building it from an SFPU binary add with a constant ones tile adds a third ring, an unpack per tile and four SFPU passes with their inits in place of one. That is the same math as `_sfpu_sigmoid_` (accurate exp + Newton reciprocal) at four times the DEST traffic |
| `RdivUnary` / `MulUnary` | `eltwise/unary/scalar.inl:53-56,82-88` | neither forms `1 + e` |

The custom element is a **chain element**, not a raw kernel. The chain still owns the DEST window, the ring lifecycle, the format reconfig and the init hoist, so the kernel is one `eltwise_chain` call.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| ULP gate, not PCC | bfloat16 cells are graded by `NUMERIC_RESOLVED` (≤ 2 ULP every element, ≤ 1 ULP when every input is in `[0, 1]`). A PCC-passing approximate sigmoid fails | fp32-accurate body (`_sfpu_sigmoid_<true>`) into fp32 DEST. Its error is far below 1 bf16 ULP before the pack, so the pack's rounding (even if it truncates) stays < 1 ULP. Run the three `NUMERIC["regression"]` cases early |
| `vConstFloatPrgm0` must be 2.0 | `_sfpu_reciprocal_<2>` reads it (`ckernel_sfpu_recip.h:34`) | `SigmoidAccurate::init` = `sigmoid_init<false>` (programs it). No other SFPU op in the kernel overwrites it |
| Large \|x\| through the exp range reduction | `_sfpu_round_to_nearest_int32_` is valid only for \|z\| < 2^22 | clamp `x` to `[-88, 88]` inside the element (Mechanism caps) |
| Compute binary over 24 KiB | the accurate exp + Newton body, unrolled 8× | if the link fails, the functor's ITERATIONS loop uses `#pragma GCC unroll 1` or 2 (same result) |
| Tag register | implicit-sync kernels leave a ring's transaction id in the hart's tag register | last statement of every reader thread `noc_async_read_set_trid(0, noc.get_noc_id())`, of every writer thread `noc_async_write_set_trid(0, noc.get_noc_id())`. Session test (Work Distribution) |
| Thread plan silently degraded | the golden suite passes with one NEO too | loop bounds from the plan's CT counts. `probe_tag_neo` test |
| Tiny tensors | `(32, 64)` is 2 pages: 1 round, 1 cluster, NEOs 0–1 busy, 2–3 with `neo_tiles = 0` | a NEO, reader or writer with 0 pages must fall straight through: the chain with `tiles(0)` does nothing, and the reader/writer loops start at `p ≥ num_tiles`. Covered by the 1- and 3-page tests |

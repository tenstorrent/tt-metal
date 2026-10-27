# Quasar Uplift Report: `ttnn.topk`

**Op root:** `ttnn/cpp/ttnn/operations/reduction/topk/` (namespace `ttnn::prim`, unchanged)
**Date:** 2026-10-21
**Procedure:** [quasar_porting.md](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L24-L53), with `ai/audit/quasar_audit.md` and `ai/audit/cb_dfb_quasar_audit_helper.md` read from branch `akertesz/op-porting-recipe` at `4bd4bf42bfe`.
**Tested on:** a Wormhole n150 board and the craq-sim Quasar simulator. No Blackhole board was available.

This report is for review only. Leave it uncommitted and delete it before merge.

## Status

| Program factory | Reached on Quasar | Verdict |
|---|---|---|
| `TopKSingleCoreProgramFactory` (Metal 2.0 `ProgramSpec`) | Yes, for every call | Uplifted in place, and runs on craq-sim. The results can be wrong when the input has negative values (LLK bug, deferred item 1). Quasar accepts only BFLOAT16 values with UINT16 indices (deferred item 2). |
| `TopKMultiCoreProgramFactory` (`ProgramDescriptor`) | No: Quasar always selects the single-core factory | **RED**: not Metal 2.0. Not edited (deferred item 5). |
| `TopkRoutePrepDeviceOperation`, `TopkRouteFinishDeviceOperation` (legacy `CreateKernel` / `CreateCircularBuffer`) | No: the composite routes to them only on Blackhole ([topk.cpp:364-366](topk.cpp#L364-L366)) | Out of scope. They would be RED if Quasar ever needed them. |

**Overall status: RED**, for two reasons:

1. **An LLK bug on the op's path.** The Quasar top-k SFPU sort compares its float values with the integer compare of `SFPSWAP`, so it puts two negative values in the wrong order. The result is wrong values and indices, with no hang and no LLK assert. This affects `largest=False` on data with negative values, and `largest=True` on rows that have fewer than k values that are not negative. The -inf padding counts as a negative value here. In an experiment on craq-sim, a one-value change in the LLK fixed all 9 `test_topk` and `test_topk_padded_width_sentinel_ties` failures that this bug causes. Deferred item 1 has the evidence and the suggested fix.
2. **The multi-core factory is not Metal 2.0.** Quasar runs every input on the single-core factory. That is correct, but slower for wide rows.

The single-core uplift itself is complete. With inputs that the LLK bug does not affect, every case that was run on craq-sim passes, with LLK asserts on and off.

## Test results

The build (`./build_metal.sh -e --enable-fake-kernels-target`) passed. The kernels are compiled just in time when a test runs, and every run below started after the last edit of the kernel sources, so all results are for the final code.

### Wormhole n150 (parity)

| Test file | Result |
|---|---|
| `tests/ttnn/unit_tests/operations/reduce/test_topk.py` | 308 passed, 43 skipped, 76 xfailed, 0 failed |
| `tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_stable_adversarial.py` | 30 passed |

### Quasar (craq-sim)

Unless a row says otherwise, each `test_topk` case is BFLOAT16 with `N=1`, `C=1`, `H=32`, `dim=3`, `largest=True` and `sorted=True`, with inputs from `torch.randn` ([test_topk.py:40](../../../../../../tests/ttnn/unit_tests/operations/reduce/test_topk.py#L40)). The rows marked "script" used short scripts that are not in the repo; the row gives their input.

Cases that pass:

| Case | LLK asserts | Notes |
|---|---|---|
| `test_topk`, W=64, k=2 | on and off | smallest case |
| `test_topk`, W=64, k=4 | on and off | |
| `test_topk`, W=8192, k=6 | on and off | |
| `test_topk`, W=8192, k=50 | on and off | k > 32, so the insertion steps run with 2 output tiles |
| `test_topk`, W=10000, W=16384 and W=18992, k=32 | on and off | W=10000 and W=18992 are not tile aligned, so the last input tile is padded with -inf |
| `test_topk`, W=64128, k=32 | on | 2004 input tiles, indices up to 64127; about 4 minutes on craq-sim |
| script: W=65536, k=32, 32 planted columns per row | on | 0 of 1024 indices wrong, including the bit patterns 0x7FFF, 0x8000, 0x8001 and 0xFFFF |
| script: H=2080, W=96, k=32, all values > 0 | on | 65 tile rows on 32 cores, so some cores process more than one row; 0 rows wrong |
| script: H=32, W=3100, k=3000, all values > 0 | on | 94 output tiles; correct |
| script: H=32, W=64, k=4, all values > 0, `largest=False` | on | correct |

Cases that fail:

| Case | Result | Cause |
|---|---|---|
| `test_topk`, `largest=False`, W=64, k=4 | PCC -0.427209 | deferred item 1 |
| `test_topk`, `largest=False`, W=8192, k=50 | PCC -0.907640 | deferred item 1 |
| `test_topk`, H=2080, W=96, k=32 | PCC 0.999541 | deferred item 1 |
| `test_topk`, N=5, C=9, H=1024, W=96, k=32 | PCC 0.999912 | deferred item 1 |
| `test_topk`, N=8, C=16, H=18, W=20, k=18 | -inf in the output | deferred item 1; W=20 is padded to 64 with -inf |
| `test_topk_padded_width_sentinel_ties`, the 4 bf16 cases | wrong values; the `largest=True`, W=40 case also returns indices into the padding | deferred item 1 |
| script: H=32, W=64, k=4, all values < 0, `largest=True` and `largest=False` | wrong values | deferred item 1 |
| `test_topk`, H=64, W=64, `dim=2`, k=32 | `DFB 'in0' has data format 'UInt16' which is not supported on architecture quasar` | `ttnn.transpose` of the UINT16 index output (deferred item 4); the top-k op itself ran |

Not run on craq-sim: the complete `-k BFLOAT16_B` subset of `test_topk` (the simulator is too slow for all of it, so the cases above were selected to cover the code paths), and the FLOAT32 and BFLOAT8_B cases, which Quasar validation rejects on purpose.

## Why the uplift is not a no-op

Before these changes, top-k could not run on Quasar at all:

1. Host validation rejects the index DFBs. Quasar has no `UInt16` or `UInt32` DFB format ([tt_backend_api_types.cpp:97-122](../../../../../../tt_metal/common/tt_backend_api_types.cpp#L97-L122), checked in [dfb.cpp:104-112](../../../../../../tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp#L104-L112)).
2. The reader does not compile for Quasar DM. The shared index helper calls `ckernel::load_blocking`, which only the WH/BH `ckernel.h` defines, and Quasar DM builds include the Quasar `ckernel.h` ([qa_hal.cpp:332-334](../../../../../../tt_metal/llrt/hal/tt-2xx/quasar/qa_hal.cpp#L332-L334)).
3. The compute kernel hangs in its first insertion step. On Quasar, `~DataflowBuffer()` waits until the buffer is empty ([dataflow_buffer.inl:481-542](../../../../../../tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl#L481-L542)). The kernel built short-lived `DataflowBuffer` objects for its compute self-loop buffers while those buffers still held tiles, and no other kernel can pop a self-loop buffer.
4. On Quasar, every `pack_tile` to a buffer other than `output_val` writes into the `output_val` ring. On Quasar, `pack_reconfig_data_format` changes only the pack format; the packer destination comes from the last `pack_init` ([llk_pack_tile_api.h:26-34](../../../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_pack_tile_api.h#L26-L34)), and `compute_kernel_hw_startup` programs `output_val` ([compute_kernel_hw_startup.h:82-90](../../../../../../tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h#L82-L90)).
5. Inputs that are eligible for the multi-core factory selected the `ProgramDescriptor` factory. Quasar firmware sets up only DFB interfaces, not circular buffers ([trisc.cc:248-251](../../../../../../tt_metal/hw/firmware/src/tt-2xx/trisc.cc#L248-L251)).
6. With `Int16` index DFBs, the JIT build of the compute kernel fails with "All input data-formats must have the same exponent format". `check_consistent_format_across_buffers` requires one exponent family for all buffers, and it skips `UInt16`, but not `Int16` ([data_format.cpp:68-76](../../../../../../tt_metal/jit_build/data_format.cpp#L68-L76)). So the `Int16` index buffers clashed with the `Float16_b` value buffers.

The audit found items 1 to 5. The first craq-sim runs found item 6, and a `-Werror` compile error in the new code of the shared helper.

## Files changed

| File | Change |
|---|---|
| [device/kernels/compute/topk.cpp](device/kernels/compute/topk.cpp#L181-L193) | One `DataflowBuffer` per buffer, built once in `kernel_main` and passed to the helpers by reference. Quasar-only `pack_init` after each packer retarget. Quasar-only `dummy_unpack` / `dummy_pack` for the TEN-4746 pairs. |
| [device/topk_single_core_program_factory.cpp](device/topk_single_core_program_factory.cpp#L49-L54) | Quasar-only: the four index DFBs use `Int16` instead of `UInt16`. The reader's `uint16_output` flag now reads the index tensor dtype ([lines 256-258](device/topk_single_core_program_factory.cpp#L256-L258)); the value is the same on WH/BH. |
| [device/topk_device_operation.cpp](device/topk_device_operation.cpp#L52-L56) | Quasar-only: never select the multi-core factory, in `select_program_factory` ([lines 121-122](device/topk_device_operation.cpp#L121-L122)) and in the matching branch of `validate_on_program_cache_miss` ([lines 352-361](device/topk_device_operation.cpp#L352-L361)). Reject FLOAT32 and BFLOAT8_B input and 32-bit indices with clear messages ([lines 316-332](device/topk_device_operation.cpp#L316-L332)). |
| [../../../kernel_lib/index_tile_dataflow.hpp](../../../kernel_lib/index_tile_dataflow.hpp#L54-L75) (shared, outside the op) | Quasar-only: write the index tile with RISC stores under `scoped_write_lock`, instead of the NoC replication that needs `ckernel::load_blocking`. Include `ckernel.h` on WH/BH only ([lines 12-14](../../../kernel_lib/index_tile_dataflow.hpp#L12-L14)). The WH/BH code is unchanged. |
| [tt_metal/jit_build/data_format.cpp](../../../../../../tt_metal/jit_build/data_format.cpp#L71-L76) (shared host code, outside the op) | Skip `Int16` in the exponent-family check, as the check already does for `UInt16`, `UInt32`, `Int32` and `UInt8`. |

Namespace and directory are unchanged. Nothing suggested a move, a rename, or a copy into `experimental/quasar/`.

**About the shared helper.** The recipe sanctions only `_metal2` forks as writes outside the op directory. The change to `index_tile_dataflow.hpp` is a different kind of write: an `ARCH_QUASAR` build-skew guard in a shared `kernel_lib` header, the same kind of fix as the `reduce_helpers_compute.inl` row of [§8.1](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L164-L178). The header is also used by `sort`, `moe`, `sampling`, and the multi-core top-k reader. Their behavior changes only on Quasar, where the header did not compile before. Reviewers can move the `index_tile_dataflow.hpp` change into a separate PR.

**About the JIT format check.** `data_format.cpp` is host code that all architectures share, so the change has no `ARCH_QUASAR` guard. It does not change WH/BH, because no WH/BH program in the tree has an `Int16` buffer: ttnn has no INT16 data type, and outside top-k the only `Int16` buffers are in Quasar-only LLK tests ([test_untilize_tilize.cpp:1314-1324](../../../../../../tests/tt_metal/tt_metal/llk/test_untilize_tilize.cpp#L1314-L1324), [test_sfpu_compute.cpp:3684-3711](../../../../../../tests/tt_metal/tt_metal/llk/test_sfpu_compute.cpp#L3684-L3711)). The Quasar typecast work (#47852) deferred the same exemption, together with an `fp32` to `Int16` pack rule, until its narrow-integer data path is ready. The comment at [test_sfpu_compute.cpp:3694-3695](../../../../../../tests/tt_metal/tt_metal/llk/test_sfpu_compute.cpp#L3694-L3695) describes that deferral. Top-k needs only the exemption. If this change lands, the typecast owner should confirm it, and that comment needs an update. Reviewers can also move this change into a separate PR.

## Quasar-uplift audit (`quasar_audit.md`)

### Check 1: device-side CB / DFB (single-core factory)

Scope: [reader_create_index_tensor.cpp](device/kernels/dataflow/reader_create_index_tensor.cpp#L1-L71), [writer_binary_interleaved.cpp](device/kernels/dataflow/writer_binary_interleaved.cpp#L1-L56), [compute/topk.cpp](device/kernels/compute/topk.cpp#L1-L512), and the include [index_tile_dataflow.hpp](../../../kernel_lib/index_tile_dataflow.hpp#L1-L165).

Step 4 scans found no `get_local_cb_interface` field access (GATE), no `read_tile_value` / `get_tile_address`, no `get_pointer_to_cb_data`, no `fifo_page_size`, no legacy CB or NoC free functions, and no semaphores. The only `get_write_ptr()` is in the helper's WH/BH branch.

| Buffer | Class | Kernels | 1xx | 2xx | Notes |
|---|---|---|---|---|---|
| `input` | 1 | reader to compute | Portable (`already_DFB`) | Portable | explicit sync |
| `index` | 1 | reader (shared helper) to compute | Portable | Portable | `Int16` on Quasar; the helper writes under `scoped_write_lock` there |
| `transposed_val`, `transposed_ind` | 1, compute self-loop | compute | Portable | Portable | compute self-loops are legal on Gen2; some pushes are of unwritten slots, which need `dummy_pack` on Quasar |
| `result_prep_val`, `result_prep_ind` | 1, compute self-loop | compute | Portable | Portable | a step pushes `incr` slots after packing 1 or 2 tiles, and the row end pushes `Kt` unwritten slots, to keep the ring aligned; no pointer surgery |
| `output_val`, `output_ind` | 1 | compute to writer | Portable | Portable | explicit sync |

There are no DM self-loops, no `evil_*` calls, and no borrowed DFBs. Rollup: **GREEN**. GATE hits: none. Blocked on runtime: none.

### Check 2: non-zero-initialized semaphores

The single-core factory has no semaphores. The multi-core factory's two semaphores start at `INVALID`, which is 0, so check 2 would not block the multi-core factory either.

## §7 and §8 items

### Applied

| Item | Where | Fix |
|---|---|---|
| §7 `finish()` on every DFB | compute kernel | Since #57646 the drain runs in `~DataflowBuffer()`, so every `DataflowBuffer` object drains when it goes out of scope, not only at kernel end. The kernel now has one object per buffer, alive for the whole kernel ([topk.cpp:181-193](device/kernels/compute/topk.cpp#L181-L193)). The insertion step selects its wait and pop buffers with a branch instead of a local object ([topk.cpp:360-369](device/kernels/compute/topk.cpp#L360-L369), [topk.cpp:449-456](device/kernels/compute/topk.cpp#L449-L456)). See the balance table below. |
| §7 TEN-4746, unpack side | `cb_wait_pop_front` on `transposed_*`: BARE | `dummy_unpack` ([topk.cpp:118-127](device/kernels/compute/topk.cpp#L118-L127)) |
| §7 TEN-4746, pack side | `cb_reserve_push_back` on `result_prep_*`: BARE ([topk.cpp:136-145](device/kernels/compute/topk.cpp#L136-L145)); `transposed_*` reserve/push in the first sort with `Kt > 1` and in the growing phase, where the bottom tile goes to `result_prep_*`: GUARDED-PATH ([topk.cpp:478-491](device/kernels/compute/topk.cpp#L478-L491)) | `dummy_pack`, the no-write `PACR` from #55426. Per the §7 caveat, `dummy_unpack` is not used for these pack-side pairs. |
| §7 re-`pack_init` after a retargeting `pack_reconfig_data_format` (#58488) | `transpose_and_pack`, `pack_results`, input transpose loop ([topk.cpp:35-40](device/kernels/compute/topk.cpp#L35-L40), [topk.cpp:80-91](device/kernels/compute/topk.cpp#L80-L91), [topk.cpp:295-305](device/kernels/compute/topk.cpp#L295-L305)) | `#ifdef ARCH_QUASAR pack_init(dfb); #endif` |
| §7 no `uint16` / `uint32` device format | single-core factory | `Int16` for 16-bit indices. 32-bit indices are rejected, because they also need a 32-bit transpose. |
| §8.1 build skew | shared helper | `ckernel::load_blocking` does not exist on Quasar. The first Quasar compile also needed `lock.template get_ptr<volatile T>()` (`-Werror=missing-template-keyword`). |
| §8.1 build skew (JIT host check) | `data_format.cpp` | `Int16` exemption in the exponent-family check (item 6 of "Why the uplift is not a no-op"). |
| §6 local NoC self-read on the emulator | shared helper | The Quasar path stores with the RISC instead of reading its own SRAM through the NoC. |

### Considered, not needed

- `compute_kernel_hw_startup` exactly once: already at the start of `kernel_main`.
- Unpack-side re-init when the DFB changes: already present (`copy_init` / `transpose_init` before each switch).
- `-inf` / NaN arithmetic: the kernel has no `exp` or other arithmetic on values; the sort only compares them with `SFPSWAP`. The -inf padding is a negative value, so deferred item 1 affects it.
- Multicast rectangles and NoC directions: the single-core factory has no multicast.
- DM self-loops, sync-free DFBs, `Scratchpad`, `LocalTensorAccessor`: none.
- `disable_dfb_implicit_sync_*`: not set. The reader and writer use explicit sync through the plain `Noc` overloads; implicit sync applies only to the `NocOptions::TXN_ID` overloads.
- `get_entry_size()` instead of `fifo_page_size`: the kernels already use `get_entry_size()`.
- 64-bit pointer casts: none in the DM kernels. The new helper path uses `CoreLocalMem`.
- Gen2 hardware configs (`gen2_hardware_configs.md`): zero sites. The DM configs come from `ttnn::create_{reader,writer}_datamovement_config()`, and `ComputeHardwareConfig` is one generation-agnostic struct since #55007. No `#52269` marker is needed: the only configuration Quasar accepts has no `unpack_modes` entries.
- `opt_level`: unchanged (compute O3 explicit, DM default O2). This is a base-port item and is correct.
- Stable sort: existing validation already rejects `stable=true` on Quasar, where the LLK `static_assert`s it off.
- Reactive §8 rows (MOP timeout, DPRINT effects, multicast clamps, tilize faults): none of these symptoms occurred in the craq-sim runs.

## DFB balance for the drain at destruction

Per row of work (one `core_loop` iteration). Validation requires a reduced width of at least 64, so `Wt` is at least 2. `Kt` is `output_tiles`.

| DFB | Pushes | Pops | Count left at the end of the row |
|---|---|---|---|
| `input`, `index` | reader: `Wt` | compute: 2 in the first pass, then 1 per pass, `Wt` in total | 0 |
| `transposed_*` | per pass: the input tiles (2 in the first pass, 1 after), plus 1 per insertion step | per pass: 1 per insertion step (2 in the first sort), plus 1 cleanup pop | 0 (also 0 at the end of every pass) |
| `result_prep_*` | first sort: `Kt`; each later step pushes as many as it pops; row end: `Kt` more | final transpose: `2 * Kt` | 0 |
| `output_*` | compute: `Kt` | writer: `Kt` | 0 |

Each DFB is empty when its `kernel_main`-scope object is destroyed. No craq-sim run hung, including the runs with 2 and 2004 input tiles per row, the run with 94 output tiles, and the run with more than one row per core. The shared helper still builds its own `DataflowBuffer` per index tile, so on Quasar the reader waits for compute to pop each index tile before it continues. That wait always finishes, because compute pops every index tile right after its value tile (deferred item 6).

## Deferred and follow-up items

1. **LLK team: the Quasar top-k sort compares float values with the integer compare.**
   - All 25 min/max `SFPSWAP` calls in [ckernel_sfpu_topk.h](../../../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_topk.h#L252-L275) pass `imm12 = 0`, and the values are always floats (loaded as `FP16B` or `FP32`, [ckernel_sfpu_topk.h:143](../../../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_topk.h#L143-L143)). On Quasar, bit 0 of `imm12` selects the compare: 0 is a two's-complement int32 compare, and 1 is an fp32 compare ([ckernel_sfpu_reduce.h:63-66](../../../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h#L63-L66), and also [ckernel_sfpu_max_pool_indices.h:30](../../../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_max_pool_indices.h#L30-L30)). For floats, the int32 compare gives the correct result when at least one of the two values is not negative, and the opposite result when both are negative. [ckernel_sfpu_binary_max_min.h:55-59](../../../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h#L55-L59) describes the same effect, and that kernel adds a correction swap.
   - Symptom: wrong values and indices. No hang, and no LLK assert with `TT_METAL_LLK_ASSERTS=1`.
   - Smallest repro: `test_topk[sub_core_grids=None-largest=False-sorted=True-N=1-C=1-H=32-W=64-dim=3-k=4-BFLOAT16_B]` on craq-sim (PCC -0.427209).
   - The Quasar LLK test cannot find this bug: its inputs are in [0, 1) ([test_topk_quasar.py:287](../../../../../../tt_metal/tt-llk/tests/python_tests/quasar/test_topk_quasar.py#L287-L287)), or 0, 1, 2, ... in the unique-value variant ([test_topk_quasar.py:130-141](../../../../../../tt_metal/tt-llk/tests/python_tests/quasar/test_topk_quasar.py#L130-L141)).
   - Evidence 1: with all values > 0 the results are correct, and with all values < 0 they are wrong, for both `largest=True` and `largest=False` (see the craq-sim tables). With all values < 0 and `largest=True`, a row that should give -0.0302, -0.0396, -0.0410, -0.0439 gives -1.0, -0.9609, -0.9414, -0.9375, which are the 4 most negative values. With `largest=False`, the same row gives the 4 values closest to zero instead of the 4 most negative values.
   - Evidence 2: a CPU model of the int32 compare reproduces the craq-sim failures. The model orders each row by a key: the raw fp32 bits as an unsigned number for a value that is not negative, and those bits minus 2^32 for a negative value. For the 4 failing `test_topk` cases with finite results, the model gives the same PCC as craq-sim to 6 digits (-0.427209, -0.907640, 0.999541, 0.999912). For each `largest=True` case that passed on craq-sim, the model gives no wrong rows. The model shows that the H=2080, W=96 case has 2 wrong rows: the 2 rows with fewer than 32 values that are not negative.
   - Experiment: with `imm12 = 1` in the 25 calls (a scratch edit, not part of this change), the 9 failures in the craq-sim table that this bug causes (5 `test_topk` cases and the 4 `test_topk_padded_width_sentinel_ties` cases) all pass with LLK asserts on, and the W=64, k=2 case still passes.
   - Suggested fix: pass `imm12 = 1` when the values are floats, as `ckernel_sfpu_reduce.h` does. The file is Quasar-only, so the fix cannot change WH/BH.
2. **LLK team: 32-bit transpose on Quasar (tt-llk#1559).** `transpose_init` and `transpose_tile` assert on Quasar when the unpack destination format is `Float32` or `Int32` ([transpose.h:75-78](../../../../../../tt_metal/hw/inc/api/compute/transpose.h#L75-L78), [transpose.h:129-132](../../../../../../tt_metal/hw/inc/api/compute/transpose.h#L129-L132)). Until the Quasar compute API supports a 32-bit transpose, Quasar rejects FLOAT32 input and 32-bit indices: a padded reduced dimension wider than 65536, or a 32-bit `indices_tensor` or preallocated indices tensor. The Quasar top-k SFPU code has `FP32` and `INT32` load/store modes for a 32-bit DEST, but the Quasar LLK test covers only `Float16_b` values with K = 32 in a 16-bit DEST ([test_topk_quasar.py:335-382](../../../../../../tt_metal/tt-llk/tests/python_tests/quasar/test_topk_quasar.py#L335-L382)).
3. **Verified, no action: indices from 32768 to 65535.** On Quasar these indices travel in an `Int16` buffer, and Quasar hardware stores signed integers as sign-magnitude ([ckernel_instr_params.h:400-401](../../../../../../tt_metal/tt-llk/tt_llk_quasar/common/inc/ckernel_instr_params.h#L400-L401)). Index 32768 is the bit pattern `0x8000`, which is negative zero in sign-magnitude, so a step that normalizes negative zero would turn it into index 0. The craq-sim check with W=65536 returned all 1024 planted indices correctly, including 32768, and the `W=64128` case passed. If a later change in the index path breaks this, Quasar has an unsigned 16-bit format (hardware code 130, [tensix_types.h:256-259](../../../../../../tt_metal/hw/inc/internal/tt-2xx/quasar/tensix_types.h#L256-L259)) that the host does not accept for DFBs today.
4. **Owner of `ttnn.transpose`: UINT16 tensors on Quasar.** For `dim` other than the last dimension, the composite transposes the index output back with `ttnn.transpose` ([topk.cpp:161-164](topk.cpp#L161-L164)). On Quasar that transpose fails host validation: `DFB 'in0' has data format 'UInt16' which is not supported on architecture quasar`. The fix belongs in the transpose op's Quasar uplift, not in top-k.
5. **Op owner: Metal 2.0 port of `TopKMultiCoreProgramFactory`.** Its host side is `ProgramDescriptor` with `CBDescriptor`s. Its kernels use positional `get_arg_val` / `get_compile_time_arg_val`, take DFB ids from compile-time args, and use `get_write_ptr()` of the remote final DFB as a NoC address ([writer_local_topk.cpp:14-53](device/kernels/dataflow/writer_local_topk.cpp#L14-L53)). After the Metal 2.0 port, the multi-core factory needs its own Quasar uplift (multicast rectangles, semaphores, small-grid checks). Until then Quasar runs every input on the single-core factory, which is correct but slower for wide rows. Remove `multicore_factory_supported_on_arch` when the multi-core factory runs on Quasar.
6. **Perf on Quasar (separate PR).** The helper builds its own `DataflowBuffer`, so on Quasar the reader waits for each index tile to be popped before it continues. The Quasar path also uses 1024 RISC stores per index tile instead of the NoC replication. Passing the caller's `DataflowBuffer` into `generate_index_tile` would remove the wait, but the new signature would touch every op that shares the helper. The same wait is a latent deadlock for `moe` on Quasar: its reader generates index tiles `j` and `j+1` before it pushes the matching input pair ([moe reader_create_index_tensor.cpp:53-63](../moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L53-L63)), so the first helper call waits for a pop that compute cannot do yet. It is not reachable today, because Quasar rejects moe's `UInt16` index DFB; passing the caller's `DataflowBuffer` fixes it too.
7. **When the fix for item 1 lands, run again on craq-sim:** the failing cases from the craq-sim table, including `test_topk_padded_width_sentinel_ties` ([test_topk.py:223-239](../../../../../../tests/ttnn/unit_tests/operations/reduce/test_topk.py#L223-L239)), which also checks how the sort breaks ties between equal -inf values. In the experiment of item 1 the failing cases passed, but the experiment ran only one case of the passing table again (W=64, k=2). K > 32 and the `end_phase = 5` sort network that the single-core kernel uses are verified with values > 0 (k=50, k=3000).
8. **Recipe feedback.** (a) [quasar_porting.md §7](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L116-L117) says that the framework calls `finish()` "at kernel end". Since #57646 the drain runs in `~DataflowBuffer()`, so a `DataflowBuffer` built inside a helper or a loop body drains when the helper or loop body exits. On a self-loop buffer that still holds tiles, that drain never completes. A check for helper-scoped and loop-scoped `DataflowBuffer` objects belongs in the audit. (b) `gen2_hardware_configs.md` still describes the `ComputeGen1Config` / `ComputeGen2Config` variants that #55007 removed. (c) [§8.1](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L164-L178) could add two rows: the `Int16` exponent-family error from `data_format.cpp`, and `-Werror=missing-template-keyword` on `lock.get_ptr<T>()` in a template. (d) [§9](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L244-L246) says that the user runs all builds and tests. On this machine the agent could run them, and the runs found item 6 of "Why the uplift is not a no-op", the `-Werror` error, and deferred item 1, which the static audit did not find.

## WH/BH parity claim

- **Host.** Every new branch in the op tests `arch == tt::ARCH::QUASAR` at runtime, so on WH/BH the DFB formats, the factory choice, and the validation are the same as before. The reader's `uint16_output` value is also the same on WH/BH: `datatype_to_dataformat_converter` maps only `UINT16` to `UInt16`, and the `Int32` to `UInt32` remap never produces `UInt16`. The `data_format.cpp` change affects only programs with an `Int16` buffer, and no WH/BH program in the tree has one (see "About the JIT format check").
- **Compute kernel.** Every added instruction is inside `#ifdef ARCH_QUASAR`. The only change outside those guards is where the `DataflowBuffer` objects are built. On WH/BH, `DataflowBuffer` has no destructor, and its constructor only stores the id and a reference to the CB interface, plus a debug-only NoC event ([tt-1xx/dataflow_buffer.inl:22-34](../../../../../../tt_metal/hw/inc/internal/tt-1xx/dataflow_buffer.inl#L22-L34)). So the same CB calls run in the same order on the same CB ids.
- **Shared helper.** Preprocessor guards only. WH/BH compiles the same code as before.
- **Measured.** On Wormhole n150, both top-k test files pass with the final code (see "Test results"). Blackhole was not run; for Blackhole the claim rests on the three points above.

## Commands

The changes are uncommitted in the working tree of `fplavec/quasar_topk`. That branch carries 8 porting-guide commits on top of `main`, so move the changes to a branch off `main` before committing.

Build (from the repository root):

```bash
./build_metal.sh -e --enable-fake-kernels-target
```

WH and BH parity (run on each):

```bash
source python_env/bin/activate && pytest tests/ttnn/unit_tests/operations/reduce/test_topk.py
source python_env/bin/activate && pytest tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_stable_adversarial.py
```

The shared helper's WH/BH code is unchanged, so `sort`, `moe`, and `sampling` need no WH/BH rerun for the `index_tile_dataflow.hpp` change.

Quasar on craq-sim. Set `METAL` to the repository root and `QSR_SIM` to the folder that holds `libttsim.so`, in the same shell as the test command:

```bash
source "$METAL/python_env/bin/activate"
export TT_METAL_HOME="$METAL" TT_METAL_SIMULATOR="$QSR_SIM/libttsim.so" TT_METAL_SIMULATOR_HOME="$QSR_SIM"
export ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1
export LD_LIBRARY_PATH="$METAL/build/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$METAL/ttnn:$METAL"
export TT_METAL_LLK_ASSERTS=1   # run once with this line and once without it
cd "$METAL" && timeout --signal=INT --kill-after=20s 10m pytest \
  "tests/ttnn/unit_tests/operations/reduce/test_topk.py::test_topk[sub_core_grids=None-largest=True-sorted=True-N=1-C=1-H=32-W=64-dim=3-k=2-BFLOAT16_B]"
```

To select other cases, list the node ids with `pytest tests/ttnn/unit_tests/operations/reduce/test_topk.py --collect-only -q`. The pytest `-k` grammar does not accept `=`. The simulator is slow: W=64128 takes about 4 minutes, so pass `--timeout 2400` for the wide cases.

Symptoms to look for on Quasar:

| Symptom | Likely cause |
|---|---|
| wrong values only when the input has negative values (or -inf padding) | deferred item 1 |
| hang at waypoint `AAW` | a DFB that is not empty when its `DataflowBuffer` is destroyed |
| TILE_COUNTERS fault `0x00010000`, or the TEN-4746 LLK assert | a wait/pop or reserve/push pair with no real unpack or pack between them |
| LLK assert "pack_tile pack_output DFB differs from the one llk_pack_init programmed" | a missing `pack_init` |
| reader hang in `generate_index_tile` | the `scoped_write_lock` path in the shared helper |
| JIT error "All input data-formats must have the same exponent format" | an integer format that `check_consistent_format_across_buffers` does not skip |
| `DFB ... has data format 'UInt16' which is not supported` in `ttnn::prim::transpose` | deferred item 4 |

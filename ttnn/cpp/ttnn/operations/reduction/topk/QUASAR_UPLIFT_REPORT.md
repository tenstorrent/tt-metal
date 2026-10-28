# Quasar Uplift Report: `ttnn.topk`

**Op root:** `ttnn/cpp/ttnn/operations/reduction/topk/` (namespace `ttnn::prim`, unchanged)
**Date:** 2026-10-21, updated 2026-10-27 after a rebase onto `main` at `e63758cb687`, which includes the LLK fix #62653.
**Procedure:** [quasar_porting.md](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L24-L53), with `ai/audit/quasar_audit.md` and `ai/audit/cb_dfb_quasar_audit_helper.md` read from branch `akertesz/op-porting-recipe` at `4bd4bf42bfe`.
**Tested on:** a Wormhole n150 board and the craq-sim Quasar simulator. No Blackhole board was available.

This report is for review only. Delete it before merge.

## Status

| Program factory | Reached on Quasar | Verdict |
|---|---|---|
| `TopKSingleCoreProgramFactory` (Metal 2.0 `ProgramSpec`) | Yes, for every call | **GREEN**: uplifted in place. Every case run on craq-sim passes, with LLK asserts on and off. Quasar accepts only BFLOAT16 values with UINT16 indices (deferred item 1). |
| `TopKMultiCoreProgramFactory` (`ProgramDescriptor`) | No: Quasar always selects the single-core factory | **RED**: not Metal 2.0. Not edited (deferred item 3). |
| `TopkRoutePrepDeviceOperation`, `TopkRouteFinishDeviceOperation` (legacy `CreateKernel` / `CreateCircularBuffer`) | No: the composite routes to them only on Blackhole ([topk.cpp:364-366](topk.cpp#L364-L366)) | Out of scope. They would be RED if Quasar ever needed them. |

**Overall status: RED**, because the multi-core factory is not Metal 2.0. Quasar runs every input on the single-core factory. That is correct, but slower for wide rows.

The LLK bug that made the Quasar sort put two negative floating-point values in the wrong order (#62124) is fixed by #62653, and the cases that it broke now pass (see "Test results").

## Test results

All results below are from the rebased tree, after `./build_metal.sh -e --enable-fake-kernels-target` passed. Kernels are compiled just in time, and each craq-sim batch used its own empty kernel cache, so every run compiled the current kernel code.

### Wormhole n150 (parity)

| Test file | Result |
|---|---|
| `tests/ttnn/unit_tests/operations/reduce/test_topk.py` and `tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_stable_adversarial.py` | 338 passed, 43 skipped, 76 xfailed, 0 failed |

### Quasar (craq-sim)

Unless a row says otherwise, each `test_topk` case is BFLOAT16 with `N=1`, `C=1`, `H=32`, `dim=3`, `largest=True` and `sorted=True`, with inputs from `torch.randn` ([test_topk.py:40](../../../../../../tests/ttnn/unit_tests/operations/reduce/test_topk.py#L40)), so they hold both negative and positive values. The rows marked "script" used short scripts that are not in the repo; the row gives their input.

| Case | LLK asserts | Result |
|---|---|---|
| `test_topk`, W=64, k=2 | on and off | pass |
| `test_topk`, W=8192, k=50 | on and off | pass; k > 32, so the insertion steps run with 2 output tiles |
| `test_topk`, W=18992, k=32 | on | pass; W is not tile aligned, so the last input tile is padded with -inf |
| `test_topk`, W=64128, k=32 | on | pass; 2004 input tiles, indices up to 64127 |
| `test_topk`, `largest=False`, W=64, k=4 | on and off | pass |
| `test_topk`, `largest=False`, W=8192, k=50 | on and off | pass |
| `test_topk`, H=2080, W=96, k=32 | on and off | pass; 65 tile rows on 32 cores, so some cores process more than one row |
| `test_topk`, N=5, C=9, H=1024, W=96, k=32 | on | pass |
| `test_topk`, N=8, C=16, H=18, W=20, k=18 | on and off | pass; W=20 is padded to 64 with -inf |
| `test_topk_padded_width_sentinel_ties`, the 4 bf16 cases | on (2 of them also off) | pass |
| script: H=32, W=3100, k=3000, mixed-sign values | on | pass; 94 output tiles |
| script: H=2080, W=96, k=32, mixed-sign values | on | pass |
| script: H=32, W=64, k=4, all values < 0, `largest=True` and `largest=False` | on | pass |
| script: W=65536, k=32, 32 planted columns per row | on | pass: 0 of 1024 indices wrong, including the bit patterns 0x7FFF, 0x8000, 0x8001 and 0xFFFF |
| `test_topk`, H=64, W=64, `dim=2`, k=32 (run before the rebase) | on | fail: `DFB 'in0' has data format 'UInt16' which is not supported on architecture quasar`, raised by `ttnn.transpose` of the UINT16 index output (deferred item 2); the top-k op itself ran |

Not run on craq-sim: the complete `-k BFLOAT16_B` subset of `test_topk` (the simulator is too slow for all of it, so the cases above were selected to cover the code paths), and the FLOAT32 and BFLOAT8_B cases, which Quasar validation rejects on purpose. NaN inputs were not run on Quasar. The comment at [ckernel_sfpu_topk.h:38-42](../../../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_topk.h#L38-L42) says that the fp32 compare never swaps when either operand is NaN, so NaN ordering on Quasar can differ from WH/BH, where `test_topk_stable_adversarial.py` checks NaN payloads.

## Why the uplift is not a no-op

Before these changes, top-k could not run on Quasar at all:

1. Host validation rejects the index DFBs. Quasar has no `UInt16` or `UInt32` DFB format ([tt_backend_api_types.cpp:97-122](../../../../../../tt_metal/common/tt_backend_api_types.cpp#L97-L122), checked in [dfb.cpp:104-112](../../../../../../tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp#L104-L112)).
2. The reader does not compile for Quasar DM. The shared index helper includes `ckernel.h`, which on Quasar is the LLK header that Quasar DM builds include ([qa_hal.cpp:329-331](../../../../../../tt_metal/llrt/hal/tt-2xx/quasar/qa_hal.cpp#L329-L331)). That header stops a data-movement build with an `#error` ([ckernel_trisc_id.h:16-21](../../../../../../tt_metal/tt-llk/tt_llk_quasar/common/inc/ckernel_trisc_id.h#L16-L21)), and it has no `ckernel::load_blocking`, which the helper's NoC replication calls.
3. The compute kernel hangs in its first insertion step. On Quasar, `~DataflowBuffer()` waits until the buffer is empty ([dataflow_buffer.inl:482-536](../../../../../../tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl#L482-L536)). The kernel built short-lived `DataflowBuffer` objects for its compute self-loop buffers while those buffers still held tiles, and no other kernel can pop a self-loop buffer.
4. On Quasar, every `pack_tile` to a buffer other than `output_val` writes into the `output_val` ring. On Quasar, `pack_reconfig_data_format` changes only the pack format; the packer destination comes from the last `pack_init` ([llk_pack_tile_api.h:26-34](../../../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_pack_tile_api.h#L26-L34)), and `compute_kernel_hw_startup` programs `output_val` ([compute_kernel_hw_startup.h:82-90](../../../../../../tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h#L82-L90)).
5. Inputs that are eligible for the multi-core factory selected the `ProgramDescriptor` factory. Quasar firmware sets up only DFB interfaces, not circular buffers ([trisc.cc:243-247](../../../../../../tt_metal/hw/firmware/src/tt-2xx/trisc.cc#L243-L247)).
6. With `Int16` index DFBs, the JIT build of the compute kernel fails with "All input data-formats must have the same exponent format". `check_consistent_format_across_buffers` requires one exponent family for all buffers, and it skipped `UInt16`, but not `Int16` ([data_format.cpp:68-76](../../../../../../tt_metal/jit_build/data_format.cpp#L68-L76)). So the `Int16` index buffers clashed with the `Float16_b` value buffers.

The audit found items 1 to 5. The first craq-sim runs found item 6, a `-Werror` compile error in the new code of the shared helper, and the LLK bug #62124.

## Files changed

| File | Change |
|---|---|
| [device/kernels/compute/topk.cpp](device/kernels/compute/topk.cpp#L181-L193) | One `DataflowBuffer` per buffer, built once in `kernel_main` and passed to the helpers by reference. Quasar-only `pack_init` after each packer retarget. Quasar-only `dummy_unpack` / `dummy_pack` for the TEN-4746 pairs. |
| [device/kernels/dataflow/reader_create_index_tensor.cpp](device/kernels/dataflow/reader_create_index_tensor.cpp#L55-L60) | Pass the kernel's `dfb_index` object to `generate_index_tile`, instead of the DFB id. The id form builds its own `DataflowBuffer`, which on Quasar drains at every return, and it is marked `[[deprecated]]` on Quasar ([id_only_helpers.hpp:14-21](../../../kernel_lib/id_only_helpers.hpp#L14-L21)). On WH/BH the id form only forwards to the object form, so the same code runs. |
| [device/topk_single_core_program_factory.cpp](device/topk_single_core_program_factory.cpp#L49-L54) | Quasar-only: the four index DFBs use `Int16` instead of `UInt16`. The reader's `uint16_output` flag now reads the index tensor dtype ([lines 256-258](device/topk_single_core_program_factory.cpp#L256-L258)); the value is the same on WH/BH. |
| [device/topk_device_operation.cpp](device/topk_device_operation.cpp#L52-L56) | Quasar-only: never select the multi-core factory, in `select_program_factory` ([lines 121-122](device/topk_device_operation.cpp#L121-L122)) and in the matching branch of `validate_on_program_cache_miss` ([lines 354-363](device/topk_device_operation.cpp#L354-L363)). Reject FLOAT32 and BFLOAT8_B input and 32-bit indices with clear messages ([lines 316-334](device/topk_device_operation.cpp#L316-L334)). |
| [../../../kernel_lib/index_tile_dataflow.hpp](../../../kernel_lib/index_tile_dataflow.hpp#L59-L81) (shared, outside the op) | Quasar-only: write the index tile with RISC stores under `scoped_write_lock`, instead of the NoC replication. Include `ckernel.h` on WH/BH only ([lines 12-16](../../../kernel_lib/index_tile_dataflow.hpp#L12-L16)). The WH/BH code is unchanged. |
| [tt_metal/jit_build/data_format.cpp](../../../../../../tt_metal/jit_build/data_format.cpp#L71-L76) (shared host code, outside the op) | Skip `Int16` in the exponent-family check, as the check already does for `UInt16`, `UInt32`, `Int32` and `UInt8`. |

Namespace and directory are unchanged. Nothing suggested a move, a rename, or a copy into `experimental/quasar/`.

**About the shared helper.** The recipe sanctions only `_metal2` forks as writes outside the op directory. The change to `index_tile_dataflow.hpp` is a different kind of write: an `ARCH_QUASAR` build-skew guard in a shared `kernel_lib` header, the same kind of fix as the `reduce_helpers_compute.inl` row of [§8.1](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L164-L178). The header is also used by `sort`, `moe`, `sampling`, and the multi-core top-k reader. Their behavior changes only on Quasar, where the header did not compile before. Reviewers can move the `index_tile_dataflow.hpp` change into a separate PR.

**About the JIT format check.** `data_format.cpp` is host code that all architectures share, so the change has no `ARCH_QUASAR` guard. It does not change WH/BH, because no WH/BH program in the tree has an `Int16` buffer: ttnn has no INT16 data type, and outside top-k the only `Int16` buffers are in Quasar-only LLK tests ([test_untilize_tilize.cpp:1334-1344](../../../../../../tests/tt_metal/tt_metal/llk/test_untilize_tilize.cpp#L1334-L1344), [test_sfpu_compute.cpp:4140-4167](../../../../../../tests/tt_metal/tt_metal/llk/test_sfpu_compute.cpp#L4140-L4167)). The Quasar typecast work (#47852) deferred the same exemption, together with an `fp32` to `Int16` pack rule, until its narrow-integer data path is ready. The comment at [test_sfpu_compute.cpp:4150-4151](../../../../../../tests/tt_metal/tt_metal/llk/test_sfpu_compute.cpp#L4150-L4151) describes that deferral. Top-k needs only the exemption. If this change lands, the typecast owner should confirm it, and that comment needs an update. Reviewers can also move this change into a separate PR.

## Quasar-uplift audit (`quasar_audit.md`)

### Check 1: device-side CB / DFB (single-core factory)

Scope: [reader_create_index_tensor.cpp](device/kernels/dataflow/reader_create_index_tensor.cpp#L1-L71), [writer_binary_interleaved.cpp](device/kernels/dataflow/writer_binary_interleaved.cpp#L1-L56), [compute/topk.cpp](device/kernels/compute/topk.cpp#L1-L512), and the include [index_tile_dataflow.hpp](../../../kernel_lib/index_tile_dataflow.hpp#L1-L177).

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
| §7 `finish()` on every DFB | compute kernel, reader | Since #57646 the drain runs in `~DataflowBuffer()`, so every `DataflowBuffer` object drains when it goes out of scope, not only at kernel end. The compute kernel now has one object per buffer, alive for the whole kernel ([topk.cpp:181-193](device/kernels/compute/topk.cpp#L181-L193)). The insertion step selects its wait and pop buffers with a branch instead of a local object ([topk.cpp:360-369](device/kernels/compute/topk.cpp#L360-L369), [topk.cpp:449-456](device/kernels/compute/topk.cpp#L449-L456)). The reader passes its kernel-scope `dfb_index` to the index helper. See the balance table below. |
| §7 TEN-4746, unpack side | `cb_wait_pop_front` on `transposed_*`: BARE | `dummy_unpack` ([topk.cpp:118-127](device/kernels/compute/topk.cpp#L118-L127)) |
| §7 TEN-4746, pack side | `cb_reserve_push_back` on `result_prep_*`: BARE ([topk.cpp:136-145](device/kernels/compute/topk.cpp#L136-L145)); `transposed_*` reserve/push in the first sort with `Kt > 1` and in the growing phase, where the bottom tile goes to `result_prep_*`: GUARDED-PATH ([topk.cpp:478-491](device/kernels/compute/topk.cpp#L478-L491)) | `dummy_pack`, the no-write `PACR` from #55426. Per the §7 caveat, `dummy_unpack` is not used for these pack-side pairs. |
| §7 re-`pack_init` after a retargeting `pack_reconfig_data_format` (#58488) | `transpose_and_pack`, `pack_results`, input transpose loop ([topk.cpp:35-40](device/kernels/compute/topk.cpp#L35-L40), [topk.cpp:80-91](device/kernels/compute/topk.cpp#L80-L91), [topk.cpp:295-305](device/kernels/compute/topk.cpp#L295-L305)) | `#ifdef ARCH_QUASAR pack_init(dfb); #endif` |
| §7 no `uint16` / `uint32` device format | single-core factory | `Int16` for 16-bit indices. 32-bit indices are rejected (deferred item 1). |
| §8.1 build skew | shared helper | Guard the `ckernel.h` include (item 2 of "Why the uplift is not a no-op"). The first Quasar compile also needed `lock.template get_ptr<volatile T>()` (`-Werror=missing-template-keyword`). |
| §8.1 build skew (JIT host check) | `data_format.cpp` | `Int16` exemption in the exponent-family check (item 6 of "Why the uplift is not a no-op"). |
| §6 local NoC self-read on the emulator | shared helper | The Quasar path stores with the RISC instead of reading its own SRAM through the NoC. |

### Considered, not needed

- `compute_kernel_hw_startup` exactly once: already at the start of `kernel_main`.
- Unpack-side re-init when the DFB changes: already present (`copy_init` / `transpose_init` before each switch).
- `-inf` / NaN arithmetic: the kernel has no `exp` or other arithmetic on values; the sort only compares them with `SFPSWAP`. NaN ordering is not tested on Quasar (see "Test results").
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

Each DFB is empty when its `kernel_main`-scope object is destroyed. No craq-sim run hung, including the runs with 2 and 2004 input tiles per row, the run with 94 output tiles, and the runs with more than one row per core.

## Deferred and follow-up items

1. **LLK team: Quasar unpacker reconfiguration for 32-bit unpack-to-dest buffers (#63025).** The uplift adds a Quasar-only validation that rejects FLOAT32 input and 32-bit indices with a clear message ([topk_device_operation.cpp:316-334](device/topk_device_operation.cpp#L316-L334)); without it these inputs fail later, on the device. FLOAT32 input always uses 32-bit indices ([topk_utils.cpp:343-346](device/topk_utils.cpp#L343-L346)), so both need the same support.
   - The original blocker, no 32-bit transpose on Quasar (tt-llk#1559), is gone: since #62042, `transpose_init` and `transpose_tile` transpose a 32-bit tile on Quasar through unpack-to-dest ([transpose.h:72-90](../../../../../../tt_metal/hw/inc/api/compute/transpose.h#L72-L90)).
   - The remaining blocker is `reconfig_data_format_srca`, which the compute kernel calls before each `transpose_init` to switch the unpacker between the value and the index buffers. On Quasar, `_llk_unpack_reconfig_data_format_src_` checks the format pair with `unpack_to_dest` fixed to `false` ([llk_unpack_common.h:167-169](../../../../../../tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_unpack_common.h#L167-L169)), and `Float32` to `Float32` is legal only for unpack-to-dest ([cunpack_common.h:64-68](../../../../../../tt_metal/tt-llk/tt_llk_quasar/common/inc/cunpack_common.h#L64-L68)).
   - Evidence on craq-sim, with a scratch build that removed the rejection, used `Int32` index DFBs on Quasar, and set unpack-to-dest on the 32-bit value and index DFBs: the FLOAT32 case W=64, k=2 stopped at an LLK assert (`ebreak`) whose only candidates are this format check in unpack and a TEN-4746 check in pack. With the `reconfig_data_format_srca` calls also removed, the simulator reported `UNPACKER_0 ILLEGAL_FORMAT_CONVERSION: Int32(8) -> Float32(0) to Dest`, so the reconfiguration between the value and the index buffers is required.
   - After the LLK fix, the op-side changes are: on Quasar, `Int32` instead of `UInt32` for the 32-bit index DFBs (Quasar has no `UInt32` DFB format), unpack-to-dest for the 32-bit value and index DFBs, and removal of the two rejections. The Quasar top-k SFPU `FP32` and `INT32` load/store modes have not run on Quasar: the Quasar LLK test covers only `Float16_b` values in a 16-bit DEST ([test_topk_quasar.py:366-405](../../../../../../tt_metal/tt-llk/tests/python_tests/quasar/test_topk_quasar.py#L366-L405)).
2. **Owner of `ttnn.transpose`: UINT16 tensors on Quasar.** For `dim` other than the last dimension, the composite transposes the index output back with `ttnn.transpose` ([topk.cpp:161-164](topk.cpp#L161-L164)). On Quasar that transpose fails host validation: `DFB 'in0' has data format 'UInt16' which is not supported on architecture quasar`. The fix belongs in the transpose op's Quasar uplift, not in top-k.
3. **Op owner: Metal 2.0 port of `TopKMultiCoreProgramFactory`.** Its host side is `ProgramDescriptor` with `CBDescriptor`s. Its kernels use positional `get_arg_val` / `get_compile_time_arg_val`, take DFB ids from compile-time args, and use `get_write_ptr()` of the remote final DFB as a NoC address ([writer_local_topk.cpp:14-53](device/kernels/dataflow/writer_local_topk.cpp#L14-L53)). After the Metal 2.0 port, the multi-core factory needs its own Quasar uplift (multicast rectangles, semaphores, small-grid checks). Until then Quasar runs every input on the single-core factory, which is correct but slower for wide rows. Remove `multicore_factory_supported_on_arch` when the multi-core factory runs on Quasar.
4. **Perf on Quasar (separate PR).** The Quasar path of the index helper uses 1024 RISC stores per index tile instead of the NoC replication.
5. **Owner of `moe`: the id form of `generate_index_tile`.** The `moe` reader still passes the DFB id, so on Quasar it would wait for each index tile to be popped before it continues. Its reader generates index tiles `j` and `j+1` before it pushes the matching input pair ([moe reader_create_index_tensor.cpp:53-63](../moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L53-L63)), so the first helper call would wait for a pop that compute cannot do yet. It is not reachable today, because Quasar rejects moe's `UInt16` index DFB. Passing the kernel's `DataflowBuffer`, as the top-k reader now does, fixes it.
6. **Index values from 32768 to 65535: verified, no action.** On Quasar these indices travel in an `Int16` buffer, and Quasar hardware stores signed integers as sign-magnitude ([ckernel_instr_params.h:400-401](../../../../../../tt_metal/tt-llk/tt_llk_quasar/common/inc/ckernel_instr_params.h#L400-L401)). Index 32768 is the bit pattern `0x8000`, which is negative zero in sign-magnitude, so a step that normalizes negative zero would turn it into index 0. The W=65536 script and the `W=64128` case check this (see "Test results"). If a later change in the index path breaks this, Quasar has an unsigned 16-bit format (hardware code 130, [tensix_types.h:256-259](../../../../../../tt_metal/hw/inc/internal/tt-2xx/quasar/tensix_types.h#L256-L259)) that the host does not accept for DFBs today.
7. **Recipe feedback.** (a) [quasar_porting.md §7](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L116-L117) says that the framework calls `finish()` "at kernel end". Since #57646 the drain runs in `~DataflowBuffer()`, so a `DataflowBuffer` built inside a helper or a loop body drains when the helper or loop body exits. On a self-loop buffer that still holds tiles, that drain never completes. A check for helper-scoped and loop-scoped `DataflowBuffer` objects belongs in the audit. (b) `gen2_hardware_configs.md` still describes the `ComputeGen1Config` / `ComputeGen2Config` variants that #55007 removed. (c) [§8.1](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L164-L178) could add two rows: the `Int16` exponent-family error from `data_format.cpp`, and `-Werror=missing-template-keyword` on `lock.get_ptr<T>()` in a template. (d) [§9](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md#L244-L246) says that the user runs all builds and tests. On this machine the agent could run them, and the runs found item 6 of "Why the uplift is not a no-op", the `-Werror` error, and the LLK bug #62124, which the static audit did not find. (e) The Quasar LLK tests used only inputs >= 0 for a sort; signed inputs belong in any LLK test of a compare.

## WH/BH parity claim

- **Host.** Every new branch in the op tests `arch == tt::ARCH::QUASAR` at runtime, so on WH/BH the DFB formats, the factory choice, and the validation are the same as before. The reader's `uint16_output` value is also the same on WH/BH: `datatype_to_dataformat_converter` maps only `UINT16` to `UInt16`, and the `Int32` to `UInt32` remap never produces `UInt16`. The `data_format.cpp` change affects only programs with an `Int16` buffer, and no WH/BH program in the tree has one (see "About the JIT format check").
- **Compute kernel.** Every added instruction is inside `#ifdef ARCH_QUASAR`. The only change outside those guards is where the `DataflowBuffer` objects are built. On WH/BH, `DataflowBuffer` has no destructor, and its constructor only stores the id and a reference to the CB interface, plus a debug-only NoC event ([tt-1xx/dataflow_buffer.inl:22-34](../../../../../../tt_metal/hw/inc/internal/tt-1xx/dataflow_buffer.inl#L22-L34)). So the same CB calls run in the same order on the same CB ids.
- **Reader.** On WH/BH the id form of `generate_index_tile` builds a `DataflowBuffer` and calls the object form, so passing the existing object runs the same code.
- **Shared helper.** Preprocessor guards only. WH/BH compiles the same code as before.
- **Measured.** On Wormhole n150, both top-k test files pass with the final code (see "Test results"). Blackhole was not run; for Blackhole the claim rests on the points above.

## Commands

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
  "tests/ttnn/unit_tests/operations/reduce/test_topk.py::test_topk[sub_core_grids=None-largest=False-sorted=True-N=1-C=1-H=32-W=64-dim=3-k=4-BFLOAT16_B]"
```

To select other cases, list the node ids with `pytest tests/ttnn/unit_tests/operations/reduce/test_topk.py --collect-only -q`. The pytest `-k` grammar does not accept `=`. The simulator is slow: W=64128 takes about 4 minutes, so pass `--timeout 2400` for the wide cases.

Symptoms to look for on Quasar:

| Symptom | Likely cause |
|---|---|
| wrong values only when the input has negative values (or -inf padding) | an LLK without #62653 (#62124) |
| hang at waypoint `AAW` | a DFB that is not empty when its `DataflowBuffer` is destroyed |
| TILE_COUNTERS fault `0x00010000`, or the TEN-4746 LLK assert | a wait/pop or reserve/push pair with no real unpack or pack between them |
| LLK assert "pack_tile pack_output DFB differs from the one llk_pack_init programmed" | a missing `pack_init` |
| reader hang in `generate_index_tile` | the `scoped_write_lock` path in the shared helper |
| JIT error "All input data-formats must have the same exponent format" | an integer format that `check_consistent_format_across_buffers` does not skip |
| `DFB ... has data format 'UInt16' which is not supported` in `ttnn::prim::transpose` | deferred item 2 |

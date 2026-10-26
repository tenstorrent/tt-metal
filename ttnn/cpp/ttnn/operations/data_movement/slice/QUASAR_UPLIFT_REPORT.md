# Quasar uplift report: `ttnn.slice`

**Status: GREEN.**

The slice op was already on Metal 2.0 for Wormhole (WH) and Blackhole (BH). On Quasar, two of its five program
factories could not build a program, and slices of UINT16 tensors failed in the factories that use a dataflow
buffer (DFB). The same format rule also rejects UINT32. After this change, all five factories run on the craq-sim Quasar simulator. On WH, every test in
`test_slice.py` has the same result before and after the change. BH was not tested: this machine has no BH board.
The command for that run is in [Follow-up items](#follow-up-items).

The op stays in place, in this directory and in its existing namespaces. Nothing was copied into or taken from
`experimental/quasar/`.

Guide followed: [quasar_porting.md](../../../../../../docs/source/ttnn/ttnn/ai/quasar_porting.md).

## Starting point

- All five factories return `ProgramArtifacts` from `create_program_artifacts`.
- Every kernel that a factory binds uses the device 2.0 API (`dfb::`, `args::`, `tensor::`, `scratch::`,
  `DataflowBuffer`, `Noc`, `TensorAccessor`).
- The op has no compute kernels, no semaphores, and no multicast.

## Quasar failures before the change

Quasar baseline on the unmodified code (59 cases from `test_slice.py`): 42 passed, 13 failed, 4 skipped.

| Error | Failing cases | Cause |
|---|---|---|
| `DataflowBuffer 'ta_tensor' is self-looped by data-movement kernel 'ta_reader'` | 1, tensor-args factory | The reader both fills and drains the DFB that stages the start and end index tensors. Gen2 rejects a DFB that one data-movement kernel binds as both producer and consumer ([dfb.cpp:590-603](../../../../../../tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp#L590-L603)). |
| `DataflowBuffer 'sharded_in' is self-looped by data-movement kernel 'sharded_reader'` | 7, height-sharded row-major factory | Same rule. The reader bound both ends of two borrowed DFBs, one over the input shard and one over the output shard. |
| `DFB 'stride_in' has data format 'UInt16' which is not supported on architecture quasar` | 5, strided factory, UINT16 | Quasar accepts neither `UInt16` nor `UInt32` as a DFB format ([tt_backend_api_types.cpp:97-123](../../../../../../tt_metal/common/tt_backend_api_types.cpp#L97-L123)). |

## Files changed

| File | Change |
|---|---|
| [slice_program_factory_tile_tensor_args.cpp:61-68](device/slice_program_factory_tile_tensor_args.cpp#L61-L68) | `TA_TENSOR` is now a reader-private `ScratchpadSpec` of the same size (one tile) instead of a self-looped DFB. The reader binds it as a scratchpad under the same accessor name ([lines 141-144](device/slice_program_factory_tile_tensor_args.cpp#L141-L144)). |
| [reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:43-67](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L43-L67) | Reads each index tensor into the scratchpad with the same NoC read as before, then copies the indices out while holding `scoped_lock`. |
| [slice_program_factory_rm_sharded.cpp:293-322](device/slice_program_factory_rm_sharded.cpp#L293-L322) | The two borrowed DFBs are removed. The reader binds `INPUT` and `OUTPUT` as tensor bindings. The `num_sticks_unpadded` compile-time argument is removed, because only the removed DFB reservation used it. |
| [slice_reader_unary_unpad_dims_rm_sharded.cpp:34-81](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L34-L81) | Gets both shards through `LocalTensorAccessor`. The input shard base address is still the remote read address on each source core. The output shard is the NoC destination at a running byte offset. The `reserve_back` that had no matching `push_back` is gone with the DFB. |
| [reader_multicore_slice_4d.cpp:160-177](device/kernels/dataflow/reader_multicore_slice_4d.cpp#L160-L177), [reader_multicore_slice_nd.cpp:134-154](device/kernels/dataflow/reader_multicore_slice_nd.cpp#L134-L154) | The data-movement core rewrites a row inside the DFB entry when the width is sliced. It now does this while holding `scoped_write_lock(1)` and writes through the lock's pointer. |
| [writer_unary_interleaved_start_id.cpp:25-30](device/kernels/dataflow/writer_unary_interleaved_start_id.cpp#L25-L30) | The `OUT_SHARDED` branch now pops what it waits for, as the shared Metal 2.0 writer already does. |
| [slice_device_operation.cpp:95-111](device/slice_device_operation.cpp#L95-L111), [slice_device_operation.hpp:27-28](device/slice_device_operation.hpp#L27-L28) | New `get_slice_dfb_data_format()`: the dtype's own format, except on Quasar, where `UInt16` becomes `RawUInt16` and `UInt32` becomes `RawUInt32`. |
| [slice_program_factory_rm.cpp:340-341](device/slice_program_factory_rm.cpp#L340-L341), [slice_program_factory_rm_stride.cpp:60-61](device/slice_program_factory_rm_stride.cpp#L60-L61), [slice_program_factory_tile.cpp:43-44](device/slice_program_factory_tile.cpp#L43-L44), [slice_program_factory_tile_tensor_args.cpp:49-50](device/slice_program_factory_tile_tensor_args.cpp#L49-L50) | DFB format comes from `get_slice_dfb_data_format()`. |
| [slice_metal2_names.hpp:34-52](device/slice_metal2_names.hpp#L34-L52) | `SHARDED_IN` and `SHARDED_OUT` are removed. `TA_TENSOR` is now a `ScratchpadSpecName`. |

Nothing outside this directory changed. `mesh_partition` builds programs from these factories and refreshes only
`INPUT` and `OUTPUT` on a cache hit, so it needs no change.

## Parity with Wormhole and Blackhole

No change alters what WH or BH execute:

- **Index staging scratchpad (tensor-args).** The kernel reads the same bytes from the same index tensors and
  copies out the same words. Only the SRAM location of the staging region differs. On WH/BH, `scoped_lock` adds
  no device work: its cache operations exist only on Quasar, and its debug event exists only when the NoC
  debugging profiler is built in.
- **Shard access through `LocalTensorAccessor` (height-sharded).** On WH/BH a borrowed DFB is a circular buffer
  placed at the tensor's buffer address, and `get_write_ptr()` of a new circular buffer returns that address. The
  tensor binding hands the kernel the same buffer address, so the kernel issues the same NoC reads, with the same
  addresses and sizes. The removed `reserve_back` reserved space in a buffer that nothing else used.
- **Write lock in the 4D/ND readers.** On WH/BH the lock adds no device work, and its pointer is `fifo_wr_ptr`,
  the same byte address that `get_write_ptr()` returns on a data-movement core.
- **`OUT_SHARDED` pop.** No slice factory defines `OUT_SHARDED`, so no compiled slice kernel changes on any
  architecture.
- **Format remap.** It applies only when `arch == tt::ARCH::QUASAR`.

Evidence on WH (n150), full `tests/ttnn/unit_tests/operations/data_movement/test_slice.py`, 511 tests:

| Run | Passed | Skipped | Xfailed | Failed |
|---|---|---|---|---|
| Before the change | 472 | 38 | 1 | 0 |
| After the change | 472 | 38 | 1 | 0 |

A per-test comparison of the two JUnit result files shows 0 tests with a different outcome.

A second WH run with the watcher on (`TT_METAL_WATCHER=1`) covered the 31 cases that use the changed kernels: all
height-sharded cases, the first two tensor-args cases, the 4D strided cases for BFLOAT16 and UINT16, and the
row-major 5D and 7D strided cases. Result: 29 passed, 2 skipped (the test's own skip condition), no watcher error.

## Quasar results after the change

craq-sim, the same 59 cases: 55 passed, 4 skipped, 0 failed. All 13 baseline failures pass. The 4 skips are
the test's own conditions (`Cannot untilize 5D tensor` twice, `Cannot slice the last dimension to 1 in row major
layout` twice), the same as in the baseline. With `TT_METAL_LLK_ASSERTS=1` the result is the same.

The cases cover all five factories: tensor-args (`test_slice_tensor_args_device_path`, first case),
height-sharded (`test_slice_rm_sharded_shard_grid_order`, `test_slice_rm_sharded_input_and_output_grids_differ`),
strided 4D and ND (`test_stride_slice_two_dim`, `test_stride_slice_three_dim`, `test_slice_5d`,
`test_slice_7d_strided`), and the plain row-major and tile factories (`test_slice_negative_ends` with
`ends=-2`, `test_slice_within_tile_vs_across_tiles`, `test_slice_7d`). The cases were picked to fit the
simulator's 8x4 worker grid. The sharded tests that use an 8x7 grid did not run on Quasar.

An extra check, not part of this change, ran `UINT32` slices through all five factories and `UINT16` slices
through the plain row-major and tile factories, and compared each result bit for bit with torch. All passed.

## Guide items

Applied:

- **DM self-loop DFBs (guide §6).** `TA_TENSOR` became a `Scratchpad` and the two sharded DFBs became
  `LocalTensorAccessor` bindings, as in the mainline tilize uplift (commit `aa97958452b`).
- **`finish()` balance (§7).** Every DFB was checked per kernel and between kernels, on every path, including idle
  cores and `#ifdef` branches. The two imbalances found were the `OUT_SHARDED` branch of the tile writer (fixed)
  and the `reserve_back` with no `push_back` in the sharded reader (removed with its DFB).
- **No UINT16/UINT32 DFB format on Quasar (§7).** The guide allows a factory to remap a DFB to the raw format of
  the same size when the DFB's bits are only copied, never computed on. Every slice DFB only carries bytes from
  one data-movement kernel to another, and its entry size is the same under the raw format. argmax
  ([argmax_multi_core_program_factory.cpp:211-217](../../reduction/argmax/device/argmax_multi_core_program_factory.cpp#L211-L217))
  and reshape_view
  ([reshape_tiled_program_factory.cpp:306-312](../reshape_view/device/reshape_tiled_program_factory.cpp#L306-L312))
  do the same on Quasar. **Reviewer decision:** if you read §7 as "flag it, do not edit the op" for slice, remove
  `get_slice_dfb_data_format()`; slices of UINT16 and UINT32 tensors then fail on Quasar at program build.
- **Data-movement core cache (§8.3, §12).** On Quasar the data-movement core reaches SRAM through a cache.
  `DataflowBuffer::get_write_ptr()` returns an uncached address
  ([dataflow_buffer.h:392-398](../../../../../../tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L392-L398)),
  and `push_back` writes the tile counter without a fence, so nothing orders earlier stores to the entry ahead of
  it ([dataflow_buffer.inl:377-404](../../../../../../tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl#L377-L404)).
  The 4D/ND readers now hold `scoped_write_lock` while they rewrite a row, and the tensor-args reader holds
  `scoped_lock` while it reads the scratchpad. No test failed without these locks: craq-sim does not show the
  missing ordering. They follow the DFB API's rule for local access to entries, and the note in
  [local_l1_copy.hpp:31-34](../../kernel_helper_functions/local_l1_copy.hpp#L31-L34) that such stores need a
  fence before `push_back`.

Considered and not needed:

- **`fifo_page_size` (§5, §8.3).** No slice kernel reads it. Entry sizes come from `get_entry_size()` or from
  arguments.
- **Implicit sync (§7).** No factory disables it. The kernels pass a DFB to the NoC only without
  `NocOptions::TXN_ID`, so they use explicit sync and nothing counts an entry twice.
- **Compute-kernel rules (§7, §8.2, §8.4): init order, bare `wait_front`/`pop_front` pairs, `pack_init`,
  `-inf`.** The op has no compute kernels.
- **Multicast rectangles, non-zero semaphores, borrowed DFBs with an offset.** The op has none.
- **Build skew (§8.1).** All slice kernels compiled for Quasar without change. In this tree (main at
  `8d8b1ab0df9`), `data_movement/common/kernels/common.hpp` already guards its `ckernel.h` include on Quasar.
  Kernels pass the uncached DFB pointers to NoC calls, and the NoC layer maps them back to cached addresses
  ([noc.h:131-140](../../../../../../tt_metal/hw/inc/api/dataflow/noc.h#L131-L140)).
- **`opt_level` (§4).** No factory sets it, and this change does not touch it.

## Follow-up items

1. **Same-core NoC reads on the physical Quasar emulator (§6).** The sharded reader reads input rows over the
   NoC, also when those rows are on the reading core itself. craq-sim runs this correctly. The guide says the
   physical emulator can hang on, or drop, a NoC read whose source and destination are the same core. If that
   happens, the remedy that other mainline kernels use is an `#ifdef ARCH_QUASAR` direct copy with
   `local_l1_copy()` when the source core is the reading core
   ([local_l1_copy.hpp:21-58](../../kernel_helper_functions/local_l1_copy.hpp#L21-L58)). It is not applied here,
   because the symptom has not appeared (the guide applies such fixes only after the symptom appears). The same
   applies to the rare NoC path of `tt_memmove` in the interleaved row-major reader.
2. **Unused kernels.** `strided_slice_reader_rm_interleaved_nd.cpp` and `strided_slice_writer_rm_interleaved.cpp`
   are not bound by any factory and still use positional arguments and fixed DFB ids. Delete them in a separate
   cleanup.
3. **UINT32 in another op.** `interleaved_to_sharded` declares its DFBs with the dtype's own format
   ([interleaved_to_sharded_program_factory.cpp:53-54](../sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp#L53-L54)),
   so on Quasar it rejects UINT32. A flow that reshards a UINT32 tensor before slicing it fails in that op, not in
   slice.
4. **Index tensor read size.** The tensor-args reader reads one input tile of bytes from page 0 of each index
   tensor, which is much smaller than a tile. It reads past that page but never writes past it. This is the
   existing behavior and was not changed, because changing it would change the transfer.
5. **BH parity run.** Run on a BH machine and compare with the same run on main:
   `source python_env/bin/activate && pytest tests/ttnn/unit_tests/operations/data_movement/test_slice.py`
6. **Larger grids on Quasar.** The 8x7-grid sharded tests need a Quasar target with a larger worker grid than
   craq-sim's 8x4.

## How the Quasar runs were made

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD
export TT_METAL_SIMULATOR=/localdev/fplavec/temp/qsr-sim/libttsim.so
export TT_METAL_SIMULATOR_HOME=/localdev/fplavec/temp/qsr-sim
export ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH PYTHONPATH=$PWD/ttnn:$PWD
pytest --timeout 600 <node ids>   # optionally with TT_METAL_LLK_ASSERTS=1
```

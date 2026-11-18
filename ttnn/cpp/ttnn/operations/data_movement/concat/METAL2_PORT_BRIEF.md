# Handoff — concat `ConcatS2SMultiProgramFactory`, Metal 2.0 port

> **Used as this port's `METAL2_PORT_BRIEF.md`** (2026-11-12). The audit (`METAL2_PREPORT_AUDIT.md`) issued no brief because it closed RED; Edwin waived the RED for `ConcatS2SMultiProgramFactory` only and supplied this handoff as the porter-facing input. Target concept: `ProgramSpecFactoryConcept` (audit, TTNN factory analysis). Decision below resolved by Edwin: **per-node** runtime arrays.

*For a porting Claude. Run the standard recipes (audit → port → post-port) on `data_movement/concat` as usual. This brief covers the one place where this factory departs from them. Everything below was read on `main` at `0de8c7f` (Nov 8, 2026). Nothing was compiled; the two unproven seams are listed at the end.*

## The task

Port **`ConcatS2SMultiProgramFactory`** only. Leave the rest of the op alone:
- `ConcatProgramFactory`, `ConcatS2SRMProgramFactory` and `ConcatS2STiledProgramFactory` are already on Metal 2.0.
- `ConcatBlockShardedProgramFactory` is out of scope; its port waits on an open design question.
- `ConcatS2IProgramFactory` is dead code (its reader kernel was deleted in 2024; the factory is unreachable).
- `ConcatTiledUnalignedProgramFactory` and `concat/codegen/` handle interleaved inputs only, so they're unrelated.

One `program_factory_t` may mix ported and legacy factories; concat already does.

## What the factory does

Inputs and output are height- or width-sharded in L1 on the same core grid. Each core builds its output shard from its own input shards, so **nothing crosses cores**. One kernel source, `reader_s2s_tensor_concat.cpp`, runs twice per core (reader and writer configs). The host splits the work between the two instances with the CTAs `block_start`/`block_count` and per-RISC RTAs.

Every input, and the output, is a **borrowed CB** (backed by the tensor's buffer) used only for its address. The kernel calls `get_read_ptr()`/`get_write_ptr()` and never touches the FIFO. It copies with a NoC read to its own coordinates (`my_x/my_y`), using the NoC as a DMA engine, through the stateful `set_async_read_state`/`async_read_with_state` pair when `page_size <= NOC_MAX_BURST_SIZE`. The input CB ids are `0..N-1`, selected by the loop counter (`DataflowBuffer input_dfb(input_id)`).

## The departure: LocalTensorAccessor, not borrowed DFBs

The recipe's basic port turns borrowed CBs into borrowed DFBs and leaves the conversion to `LocalTensorAccessor` (LTA) for the post-port pass (`post_port/style/sync_free_dfbs.md`). **That route is closed here.** The inputs are a variable-count set selected by a runtime index, and Metal 2.0 deliberately has no variadic DFB bindings. The sibling `ConcatS2SRMProgramFactory` uses borrowed DFBs only because it has exactly two inputs; don't copy that part of it.

So do the LTA conversion during the port. These buffers trivially meet `sync_free_dfbs.md`'s criterion: no credit calls anywhere, and a single kernel source binds them.

**Host.** Declare no `DataflowBufferSpec`s. Declare one `TensorParameter` per input, plus the output. Both KernelSpecs bind every input and the output as `TensorBinding`s, and each declares a `KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "inputs", .members = …}` over the input accessor names.
- Model the sequence on `concat_program_factory.cpp:227-254` (same op, same sequence).
- Model the two-KernelSpecs-on-one-source shape, with a `WorkUnitSpec`, on `concat_s2s_rm_program_factory.cpp`.
- Two kernels binding the same `TensorParameter` is legal.

**Kernel.** The core of it:

```cpp
const auto inputs = make_local_tensor_accessors<uint32_t>(tensor::inputs);  // std::array; runtime-indexable
const LocalTensorAccessor<uint32_t> output(tensor::output);
// ... per input i, per page:
noc.set_async_read_state<NocOptions::DEFAULT, page_size>(inputs[i], page_size, {.offset_bytes = read_off});
noc.async_read_with_state<NocOptions::DEFAULT, page_size>(
    inputs[i], output, page_size, {.offset_bytes = read_off}, {.offset_bytes = write_off});
```

- An LTA on the NoC side of a transfer is a loopback to this core's L1 (`tt_metal/hw/inc/internal/local_mem/noc_traits.h`), so it replaces `UnicastEndpoint{my_x, my_y, .addr}` exactly. Precedent: `clone/device/kernels/read_kernel_sharded.cpp:25`.
- Keep byte offsets where the kernel keeps addresses today. Keep the `async_read` branch for `page_size > NOC_MAX_BURST_SIZE`.
- `LocalTensorAccessor<T>` is templated only on element type. That's why the array needs no type erasure.
- Routing guarantees L1 inputs and output, so LTA's DRAM guard can't fire.
- Don't reintroduce `my_x/my_y` endpoints or raw base addresses. Those getters are being removed from the device API.

## Arguments

- **CTAs:** drop `output_dfb_id` and `num_input_tensors` (the sequence carries its length, `inputs.size()`). Make the rest named CTAs. Keep `block_start`/`block_count` compile-time and separate per RISC; the kernel's comment (`reader_s2s_tensor_concat.cpp:24-25`) records a measured cost for making them runtime.
- **RTAs: use runtime array arguments, not varargs.** Each input has 5 values, which the kernel reads by input index. Declare them as five runtime arrays of length N in `runtime_arg_schema.runtime_array_args`: `input_num_pages_per_stick`, `input_num_sticks`, `input_write_offset`, `input_read_offset`, `input_block_stride`. The recipe's "RTA varargs" guidance predates array arguments (#63805), so this overrides it for this factory.
  - Set them per node (`runtime_array_arg_values[name][node] = …`), as the legacy factory does, even though every node gets the same values. The reader and writer instances get different values. **Ask Edwin before switching to common arrays** (decision below).
  - In the kernel, `const auto read_offset = get_arg(args::input_read_offset);` and then `read_offset[input_id]` replace the `get_arg_val(arg_idx++)` walk.
  - Model on `tests/tt_metal/tt_metal/api/metal2_host_api/integration_tests/kernel_args_loopbacks.cpp`: kernel reads at lines 677-684, declaration at 713, values at 769.
- **Decision for Edwin: per-node or common runtime arrays?** Ask him before porting this part.
  - *For common:* every node gets identical values, and `program_run_args.hpp` says a common argument dispatches more efficiently in that case. In Metal 2.0 the reader and writer are separate KernelSpecs with separate common args, so the two instances' different values can't collide.
  - *For per-node:* the legacy factory has a comment (`concat_s2s_multi_program_factory.cpp:274-278`) insisting on per-core args, because making them common caused "precision failures in sharded concat". It came with the May 2026 ProgramDescriptor migration (#43535); before that the code already used per-core args. The root cause was never established, so it may have been specific to ProgramDescriptor, but nobody has shown that.
  - If he picks common, the existing tests are the check; they caught the failure last time. Default to per-node if he has no preference.
- Keep the factory's `num_input_tensors <= 16` check, and reword its message, which blames CB ids. Raising the ceiling is a separate change; see the comment at `concat_device_operation.cpp:385-404`.

## Leave as is

`select_program_factory`, validation, and `compute_program_hash`. The hash folds in the factory index and an input-aliasing signature. Aliasing (`concat([x, x])`) does reach this factory, and `test_concat_cache_aliasing.py` covers it.

## Verify

- **Confirm the tests reach this factory.** Put a temporary `TT_THROW` in it, note which tests fail, then remove it. Candidates:
  - `tests/ttnn/unit_tests/operations/data_movement/`: `test_concat.py` (`test_sharded_concat`, `test_sharded_concat_with_groups`, `test_concat_sharded_pad`), `test_concat_cache_aliasing.py`, `test_concat_program_cache.py`
  - nightly: `test_concat.py::test_sharded_tiled_concat_int`, `test_concat_memory_configs_and_layouts.py`

  S2SMulti takes L1 height- or width-sharded inputs with sharded L1 output, except 2-input width concats (S2SRM/S2STiled) and anything ND-sharded.
- Run the same set before and after the port and compare.

## Unproven seams: stop and report rather than improvise

- No kernel in the tree passes an LTA to `set_async_read_state`/`async_read_with_state`. Both are generic over the endpoint traits, and the address an LTA resolves to is the one the kernel builds by hand today, so it should compile. If it doesn't, report it. Don't fall back to `my_x/my_y` endpoints.
- `make_local_tensor_accessors` has one host-API test and no ttnn caller yet.
- This is the first ttnn use of runtime array arguments. TTNN passes the whole `ProgramRunArgs` to `SetProgramRunArgs` (`mesh_device_operation_adapter.hpp`), so the values should get through. If they don't, report it rather than falling back to varargs.

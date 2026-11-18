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

---

> **Used as the `ConcatBlockShardedProgramFactory` port brief** (2026-11-18). As with S2SMulti, the audit closed RED with no brief; Edwin waived it for this factory and supplied this handoff. It was written against PR #62362 before it merged; the type landed as `ShardView::ResolvedShard`, still obtained with `.shard(shard_id, noc)`. One correction found in the port: `shards_per_row` cannot come from the distribution spec page shapes (see `METAL2_PORT_PLAN.md`, Deferred / Flagged).

# Handoff — concat `ConcatBlockShardedProgramFactory`, Metal 2.0 port

*For a porting Claude. Run the standard recipes (audit → port → post-port) on `data_movement/concat` as usual; this brief covers where this factory departs from them. It's the companion to the S2SMulti handoff, and the host-side shape is the same (tensor sequence, array arguments, LTAs instead of borrowed DFBs). Everything below was read on `main` at `0de8c7f` and in PR #62362 at its Oct 30 revision. Nothing was compiled; the unproven seams are listed at the end.*

## Wait for PR #62362 before starting

This port depends on Almeet's PR #62362 ("Address remote tensor shards by shard id instead of NoC coordinate tables"), which adds `ShardView::Shard`. It is not merged yet. Review has suggested renaming it (`ResolvedShard`, with `.resolve(...)` in place of `.shard(...)`), so use whatever name actually lands.

**Don't port with plain `ShardView` in the meantime.** Plain `ShardView` looks the shard up (shard → bank → core coordinates → address) on every NoC read, and this kernel issues one read per row. Today's kernel does no per-row lookup: the source core comes from its runtime args, and each row only adds a stride. So plain `ShardView` would put new work in the innermost loop, and small copies (row-major tensors, narrow shards) would feel it. That's a performance regression, which the Metal 2.0 migration contract doesn't allow. `ShardView::Shard` does the lookup once per copy and leaves a 32-bit add per row, about what the kernel does today. (#62362's own open perf item concerns plain `ShardView`, not `Shard`.)

**Don't use `TransferAccess::bank_base` with host coordinates either.** It's an internal backdoor slated for removal, and it leaves no record for op-to-op hazard tracking.

## The task

Port **`ConcatBlockShardedProgramFactory`** only. Leave the rest of the op alone. `ConcatProgramFactory`, `ConcatS2SRMProgramFactory` and `ConcatS2STiledProgramFactory` are already ported, `ConcatS2SMultiProgramFactory` has its own handoff, `ConcatS2IProgramFactory` is dead code, and `codegen/` is interleaved-only.

## What the factory does

Inputs and output are block-sharded on one rectangular grid, and each core builds one output shard. Block sharding splits both dimensions, so whichever one the concat runs along is split across cores. The output's shard boundaries then don't line up with the inputs', and most copies read **another core's** input shard.

- The host (`concat_block_sharded_program_factory.cpp`, "Pass 1" from line 258) builds each core's list of copies. Each copy is a rectangle of `num_rows` rows of `copy_size` bytes, with independent source and destination strides.
- Each copy goes into the runtime args as 9 words: `src_noc_x`, `src_noc_y`, `src_dfb_id`, `src_l1_offset`, `src_stride`, `dst_offset`, `dst_stride`, `copy_size`, `num_rows`. `src_dfb_id` is the input index, because input i is borrowed CB i.
- The kernel `reader_writer_block_sharded_concat.cpp` runs on both DM RISCs (reader and writer configs), and the host splits each core's copies between them. For each copy it takes `DataflowBuffer(src_dfb_id).get_read_ptr()`, which is **its own** address for that input, and reads that address on the remote core. That only works because every input sits at the same L1 address on every core.
- All the CBs (inputs 0..N-1, output 16) are borrowed and used only for their addresses. Nothing touches the FIFO.

## The port

**Host:**
- Declare no `DataflowBufferSpec`s. Declare one `TensorParameter` per input, plus the output.
- Both KernelSpecs bind every input and the output, and each declares `KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "inputs", …}`. Model this on `concat_program_factory.cpp:227-254`, and the two-instances-of-one-source shape on `concat_s2s_rm_program_factory.cpp`.
- **Replace the physical-core table with shard ids.** The factory already computes each source shard's tensor-space indices: `(sh, src_c)` for width concat at line 306, `(src_r, sw)` for height concat at line 369. Set `shard_id = shard_row × shards_per_row + shard_col`, where `shards_per_row` comes from **that input's own** distribution spec: `div_up(tensor_shape_in_pages()[-1], shard_shape_in_pages()[-1])`. Reach the spec through the TensorSpec, as `narrow.cpp:207` does.
  - Legacy `BLOCK_SHARDED` uses `GRID_2D` placement, and I checked that this id lands on the same core as the factory's `shard_core(r, c)` in both orientations. A temporary development assert comparing the two is cheap reassurance; remove it before merge.
- **Add the one core-order check this port still needs.** Each core writes its *local* output shard through an LTA, so the factory's assumption that core `(start_x+gx, start_y+gy)` holds output shard `(sh, sw)` now matters. `TT_FATAL` that `output_spec.cores()[sh × out_shards_per_row + sw]` is that logical core. #62362 adds checks of this kind to its own factories.

**Kernel:**

```cpp
const LocalTensorAccessor<uint32_t> out(tensor::output);
constexpr uint32_t N = std::tuple_size_v<decltype(tensor::inputs)>;
for (uint32_t c = 0; c < num_copies; c++) {
    concat_reader::with_input(input_index[c], std::make_integer_sequence<uint32_t, N>(), [&](auto I) {
        const auto in = TensorAccessor(std::get<decltype(I)::value>(tensor::inputs));
        const auto src = ShardView(in).shard(shard_id[c], noc.get_noc_id());  // looked up once per copy
        for (uint32_t row = 0; row < num_rows[c]; row++) {
            noc.async_read(src, out, copy_size[c],
                           {.offset_bytes = src_offset[c] + row * src_stride[c]},
                           {.offset_bytes = dst_offset[c] + row * dst_stride[c]});
        }
    });
}
noc.async_read_barrier();
```

- `with_input` (`concat_reader_common.hpp:21`, already in this op) turns the runtime input index into a compile-time one by comparing it against 0..N-1 and running one compiled copy of the body. The index stays a runtime value, and every core runs the same binary. The ported interleaved reader uses it already. Build the accessor inside the lambda, as that reader does: on Quasar the kernel stack is small (see #59906).
- A `Shard` is valid only on the NoC it was looked up on. The reader and writer run on different NoCs, so always pass this instance's `noc.get_noc_id()`. Watcher asserts a mismatch.
- `Shard` can only be the NoC (remote) side of a transfer. The local side is the output LTA.

## Arguments

- **CTAs:** `output_dfb_id` goes away.
- **RTAs:** use one named runtime arg, `num_copies`, plus eight per-node **runtime array arguments**, one element per copy: `input_index`, `shard_id`, `src_offset`, `src_stride`, `dst_offset`, `dst_stride`, `copy_size`, `num_rows`. That's 8 words per copy instead of 9; the core coordinates disappear and the shard id carries the core.
  - Copy counts vary per core, but an array's length is fixed per kernel. Declare each array with that RISC's maximum count across cores, and pad shorter lists; the kernel reads only `num_copies` entries. `reshard_reader` pads the same way.
  - An array's declared size must be greater than 0. The writer can have zero copies on every core (when every core has a single copy), so declare at least 1.
  - Update the runtime-arg budget `TT_FATAL`s (from line 395) to 8 words per copy. The per-RISC limit rises from 28 copies to 31.
- **Keep the 16-input cap** (`concat_device_operation.cpp:232` and `:449`). Its CB-id rationale is gone, but raising it is a separate change.

## Leave as is

`select_program_factory`, validation, and `compute_program_hash`.

## Verify

- **Confirm the tests reach this factory.** Put a temporary `TT_THROW` in it, note which tests fail, then remove it. The main file is `tests/ttnn/nightly/unit_tests/operations/data_movement/test_concat_block_sharded.py`; `test_concat.py`'s sharded tests may also land here. Check that both shard orientations (ROW_MAJOR and COL_MAJOR), both concat directions, and a row-major dtype are covered. If an orientation is missing, add a case, because the shard-id mapping is where orientation matters.
- Run the same set before and after the port, once with Watcher enabled.
- **Measure performance.** Compare kernel time with the device profiler before and after, on block-sharded shapes that include small copies (row-major, narrow shards), in both width and height concat. Per-copy overhead is new here (the shard lookup and the `with_input` compares); per-row cost should match today's.

## Unproven seams: stop and report rather than improvise

- The final name and signature of `ShardView::Shard` may change before #62362 merges.
- No kernel yet combines `with_input` with `Shard`, or uses runtime array arguments with padding. Both should work; if either doesn't, report it rather than falling back to plain `ShardView`, `bank_base`, or `my_x/my_y`-style raw endpoints.
- Binary size: `with_input` compiles one copy of the inner loop per input (at most 16 here). If the kernel binary gets close to its size limit, report it.

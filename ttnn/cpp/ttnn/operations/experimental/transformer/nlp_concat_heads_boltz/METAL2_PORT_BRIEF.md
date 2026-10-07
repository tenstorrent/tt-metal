# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_boltz`

> Audit cleared all gates. One gate was cleared by user waiver; see below. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user waiver: readiness-sheet row is stale, see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section; hash from the `Port_Recipe` checkout)*

**TTNN gate waiver (record in the port report).** The live readiness sheet (fetched 2026-10-06) still describes this op as it was before PR #57409:

- `Concept` = `legacy device-op`
- factory `NLPConcatHeadsBoltzProgramFactory`, at a now-deleted `.hpp`
- `Is able to port?` = `yes (with PD step)`

The code is already on a direct `create_descriptor`, so the audit flagged the sheet as broken. On 2026-10-06 the user waived that gate, since the problem is only the out-of-date sheet. The sheet refresh is still owed to the readiness-sheet owner.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`, in the **direct-descriptor** shape. `create_descriptor` is a static member of `NLPConcatHeadsBoltzDeviceOperation` itself (`device/nlp_concat_heads_boltz_device_operation.hpp:27-28`, body `device/nlp_concat_heads_boltz_program_factory.cpp:18-219`), with no `program_factory_t`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. There is no `override_runtime_arguments`; the framework's binding refresh covers the cache hit. Per-core args derive from the padded shape and shard spec, which the hash covers.
- **Gate-cleared, confirmed absent:** a non-clearing `TensorParameter relaxation` (cell is `none`) and `get_dynamic_runtime_args` (absent). The op also has no custom hash and no pybound `create_descriptor`. The only nanobind binding is the user function (`nlp_concat_heads_boltz_nanobind.cpp:18-27`), so no pybind line needs deleting.

## Construct — to do

**Introduce a factory struct (forced; `ttnn_factory.md` §3 "Give a direct-descriptor op a conventional program factory").** The `DirectDescriptorFactory` shim (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170`, gated by `HasDirectDescriptor`, `ttnn/api/ttnn/operation_concepts.hpp:158`) only recognizes `create_descriptor`. In `device/nlp_concat_heads_boltz_device_operation.hpp`:

1. Nest `struct NLPConcatHeadsBoltzProgramFactory` (the pre-#57409 name) with `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);`.
2. Add `using program_factory_t = std::variant<NLPConcatHeadsBoltzProgramFactory>;`.
3. Remove the device-op-level `create_descriptor` and its comment (`:23-28`).

Keep the body in `device/nlp_concat_heads_boltz_program_factory.cpp` (already in `experimental/transformer/sources.cmake:25`). Record this under Handoff points. *(Check first that no `program_factory_t` has appeared since the audit.)*

**ProgramSpec contents, by path** (`in_sharded = a.is_sharded()`, `:29`).

*Interleaved path* (`:106-131,185-213`): two DM `KernelSpec`s over `all_cores` (from `split_work_to_cores`), and one DFB.

- **reader**: `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_boltz.cpp`, reader config.
  - Named CTAs: `in0_h_tiles`, `in0_w_tiles`, `in0_c`, `in0_HtWt` (`:107-112`). The `TensorAccessorArgs` append (`:113`) disappears into the binding.
  - Per-node RTAs: `num_blocks`, `in0_h_dim`, `in0_tensor_tile_id` (`:194-201`). RTA 0 (`in0_buffer`) becomes the binding.
  - The node order is `grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major=false)` (`:186`), with group-1/group-2 block counts (`:189`). Keep it.
- **writer**: bind the existing fork `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` (see Watch for). Writer config, no defines.
  - Named RTAs `num_pages` = `num_blocks_per_core * per_tensor_tiles`, `start_id` = `num_blocks_written * per_tensor_tiles` (`:207-208`).
  - RTA 0 (`out_buffer`) becomes `tensor::dst`. CTA 0 (the CB index, `:114`) becomes `dfb::out`; the accessor append (`:115`) disappears.

*Sharded path* (`:83-105,164-184`): two `KernelSpec`s with the **same source**, `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_boltz_sharded.cpp`, one in reader config and one in writer config, both over `all_cores` = the input shard grid.

- Shared named CTAs: `in0_h_tiles`, `head_dim_size_bytes`, `out_row_size_bytes`, `block_size` (`:87-90`). CTAs 0/1 (the CB indices, `:85-86`) become `dfb::in0` / `dfb::out0`.
- Per-node RTAs (`:169-184`, node order `corerange_to_cores(all_cores, nullopt, row_major)`):
  - reader instance: `nheads` = `nheads_first_risc`, `start_read_offset_bytes` = 0, `start_write_offset_bytes` = 0
  - writer instance: `nheads` = `nheads_second_risc`, plus the two offsets computed at `:181-182`

  The values are identical on every node.

**Tensor bindings** (per binding):

- `input`, interleaved: **Case 1** (via `TensorAccessor`). Today it is a `Buffer*` RTA (`:197`), and the reader builds `TensorAccessor(in0_args, in0_tensor_addr)` (reader `:26,30`). Express it as a `TensorParameter` / `TensorBinding`, and have the kernel build `TensorAccessor(tensor::<name>)`. The address RTA (reader `:17`) and the `TensorAccessorArgs<4>` plumbing both go.
- `output`, interleaved: **Case 1**, through the fork's `tensor::dst`. The `Buffer*` RTA (`:206`) and the accessor CTAs (`:115`) go.
- `input`, sharded: **clean** (borrowed-memory DFB). CB 0 `.buffer = in0_buffer` (`:147`) becomes `DataflowBufferSpec::borrowed_from` `input`.
- `output`, sharded with a sharded output: **clean** (borrowed-memory DFB). CB 16 `.buffer = out_buffer` (`:160`) becomes `borrowed_from` `output`.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none. Both accessors are 2-arg.

**CB endpoints:**

- **CB 0, interleaved** (`:139-148`; `2 × per_tensor_tiles` tiles of `tile_size(dtype)`, page = one tile): plain 1:1. Bind the reader as **PRODUCER** (`reserve_back`/`push_back`) and the writer fork as **CONSUMER** (`dfb::out`). No flag.
- **CB 0, sharded** (borrowed from `input`, `per_tensor_tiles` = shard tiles): **assign 1P+1C**. Bind the reader-config instance as **PRODUCER** and the writer-config instance as **CONSUMER**. No flag.
- **CB 16, sharded input + sharded output** (borrowed from `output`, `:150-162`): **assign 1P+1C** the same way.
  - *Why 1P+1C, and the open question:* both instances call `reserve_back(block_size)` on both DFBs (sharded `:34-35`; dead: nothing is ever pushed). The multi-binding flag can't express two producers with no consumer (the validator needs ≥1 consumer per node). The Gen1 `DataflowBuffer::reserve_back` doesn't role-check (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:175`), so the CONSUMER-bound instance runs the same kernel unchanged. **Keep the kernel verbatim**, including both `reserve_back` calls and the commented `push_back`. (The sibling `nlp_concat_heads` port, #54782, stripped them instead. The audit left that choice as an open question to the user; unless they decide otherwise, follow this brief, and record the choice in the port report.)
- **CB 16 is a conditional DFB.** Legacy allocates it iff `out_sharded` (`:150`). It is **dead** under interleaved input + a sharded *preallocated* output (the interleaved kernels never touch index 16), and **live** under sharded input + sharded output. Create its `DataflowBufferSpec` iff `in_sharded && out_sharded`. Do **not** drop it, and don't create it on the interleaved path, where a bindingless DFB is rejected by the validator. Record this in the report as new host-side structure.

## Watch for

- **CB endpoints (multi-binding):** none. The only multi-toucher CBs are the sharded pair above (two touchers each → 1P+1C). There is no hidden second writer, no semaphores, and no third toucher.
- **Cross-op / shared kernels:** the interleaved writer is borrowed from `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`. A `_metal2` fork already exists at `…/writer_unary_interleaved_start_id_metal2.cpp`: bind it, don't re-fork, and don't touch the legacy file.
  - Fork interface: `dfb::out` (CONSUMER), `tensor::dst`, `args::num_pages`, `args::start_id`. Leave `OUT_SHARDED` and `BACKWARDS` undefined, as today.
  - The sibling `nlp_concat_heads_program_factory.cpp:177-189` binds it the same way.
  - Other binding ops still on the legacy file include `data_movement/reshape_on_device`, `data_movement/slice` (tile), `eltwise/unary_backward` (+ `gelu_bw`), `examples/example`, `experimental/matmul/attn_matmul`, `matmul` (multicore); see #52228. **That is a sunset list, not authorization to convert the kernel in place.**
  - The two op-owned kernels have no other binder (outside `experimental/quasar/`, which you must not use as a reference), so convert them in place.
- **RTA varargs:** none. Name every RTA and CTA (listed under Construct).
- **Kernels are already Device 2.0**, so this is a binding-layer swap:
  - `CircularBuffer` → `DataflowBuffer` from the `dfb::` token (interleaved reader `:32`; sharded `:31-32`)
  - drop the `api/dataflow/circular_buffer.h` include (interleaved reader `:9`; sharded `:8`)
  - `get_tile_size(cb_id)` → `dfb.get_tile_size()` (interleaved reader `:29`; sharded `:29`; whitelist rule 7)
  - Keep `my_x[noc_id]` / `my_y[noc_id]` and the `UnicastEndpoint` loopback read (sharded `:37-52`) verbatim.
- **`reserve_back` argument narrows on the DFB swap.** `CircularBuffer::reserve_back(int32_t)` becomes `DataflowBuffer::reserve_back(uint16_t)`. The sharded CTA `block_size` (`:90`) can exceed 65535 for large S. The calls are dead, so behavior doesn't change, but note it in the port report.
- **Sharded input + interleaved output is a legacy bug the port can't carry identically.** Validation accepts it (`nlp_concat_heads_boltz_device_operation.cpp:47-50`). Legacy then runs the sharded kernel against an unallocated CB 16 (hang or stray writes). After the port, `dfb::out0` has no binding there, so it fails at JIT. **Don't add a `TT_FATAL` or otherwise fix it in the port** (that is an ops-team change; the sibling has one at `nlp_concat_heads_device_operation.cpp:52-56`). Record it prominently in the port report as the one config whose failure mode changes. This was left as an open question to the user in the audit; if they decide otherwise, follow them.
- **Preserve the interleaved reader's pointer walk exactly.** It takes `get_write_ptr()` once per block, before the first `reserve_back(1)` (`:40`), and walks it linearly across `in0_c · in0_w_tiles` single-tile rounds (`:43-61`). It relies on CB 0 being exactly two blocks.
- **Leave the known anomalies alone** (sharded over-read via `in0_h_tiles` = S·S/32, unused `single_tile_size_bytes`, stale comments). They are routed to the ops team in the audit; carry them into the port report's findings and don't fix them.
- **Verification.**
  - `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads_boltz.py` covers the interleaved path only, at shapes `(2,32,64)` and `(4,64,64)`. It uses the device grid via `split_work_to_cores`, so it **runs on this workspace's 8×8 Wormhole**. Its 3-iteration loop moves the input address and asserts exactly one new cache entry, which is the cache-hit check for the new tensor bindings.
  - The sharded path has no test, and its docstring says a sharded output can't be constructed for realistic S. Verify that path by JIT-compiling both sharded `KernelSpec`s (e.g. a sharded-in/sharded-out call at the smallest shape that builds, if one exists). Otherwise, say plainly in the report that it is unverified at runtime.

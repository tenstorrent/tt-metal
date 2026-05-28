# Port Report — `sharded_to_interleaved`

## Summary

First TTNN op ported to Metal 2.0 via `ProgramSpecFactoryConcept`. Single factory; one in-family writer kernel pair (TILE + ROW_MAJOR); one broadly-shared reader; one broadly-shared compute kernel (used only when input/output dtypes differ).

**Touched files**:

- `device/sharded_to_interleaved_program_factory.hpp` — declares `create_program_spec` returning `ttnn::device_operation::ProgramArtifacts`.
- `device/sharded_to_interleaved_program_factory.cpp` — rewritten end-to-end. ~370 lines.
- `eltwise/unary/.../reader_unary_sharded_metal2.cpp` — fork (NEW).
- `data_movement/sharded/.../writer_unary_sharded_blocks_interleaved_start_id_metal2.cpp` — fork (NEW).
- `data_movement/sharded/.../writer_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp` — fork (NEW).
- `ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp` — fork (NEW).
- `METAL2_PORT_PLAN.md` — alongside this file.
- `METAL2_PREPORT_AUDIT.md` — produced during audit (precondition).

`device/sharded_to_interleaved_device_operation.cpp` and `device/sharded_to_interleaved_device_operation.hpp` are unchanged — the framework adapter detects `ProgramSpecFactoryConcept` from the `program_factory_t` variant and dispatches via `create_program_spec` automatically.

## Handoff points

- **None observed during this port.** The framework integration surface (`ProgramSpecFactoryConcept`, `ProgramArtifacts`, `MakeProgramFromSpec`) was complete and self-contained for this op's shape. The four legacy plumbing primitives (`buffer()->address()` RTAs, magic CB indices, `TensorAccessorArgs` chains, BufferBinding RTAs) all had clean Metal 2.0 replacements (`TensorBinding`, `DFBBinding`, named CTAs / RTAs).

If the build/test surfaces handoff-class issues, this section will be updated.

## Successes

- **`borrowed_from` for the sharded input DFB** ([migration guide — DataflowBufferSpec → Borrowed-memory DFBs]). The legacy code had `cb.buffer = src_buffer` (Dynamic Circular Buffer pattern). Metal 2.0's `DataflowBufferSpec::borrowed_from = INPUT` is a 1-line replacement that ties the DFB's backing L1 address to the input tensor's buffer via the `TensorBinding` mechanism — the framework patches the address on cache hit. No `dfb_run_params` entry needed for borrowed-memory DFBs; the binding picks up the buffer through the tensor args.
- **Conditional DFB / kernel binding for data-format conversion** ([migration guide — Principle 2 → Optional resources], [patterns catalog — Conditional / optional DFB bindings]). The legacy code conditionally created a second CB and compute kernel when `convert_df`. In Metal 2.0 the same conditional logic lives on the host: `out_dfb_opt` is `std::nullopt` when the formats match; the writer's `DFBBinding::dfb_spec_name` switches between `SRC_DFB` and `OUT_DFB`; the compute `KernelSpec` is pushed only when needed. The kernel sources don't change between the two paths because the writer's `local_accessor_name` is `"out"` either way.
- **`DFBAccessor` implicit conversion to `uint32_t`** ([migration guide — DFB → Direct use in LLK compute APIs]). The forked compute kernel passes `dfb::src` / `dfb::dst` directly to legacy LLK APIs (`unary_op_init_common`, `cb_wait_front`, `pack_tile`). No `.id` extraction or temporary `DataflowBuffer` wrappers needed on WH/BH. Saved a lot of mechanical bookkeeping.
- **`TensorBinding` absorbed the `Buffer*` BufferBinding slot AND the `TensorAccessorArgs` CTAs.** The legacy writer kernels each had a `Buffer*` RTA (slot 0) plus several `TensorAccessor` CTAs (`TensorAccessorArgs(*dst_buffer)`). After port, the writer's `tensor_bindings = {{OUTPUT, "out"}}` replaces both, and kernel-side `TensorAccessor(ta::out)` constructs in one line. The buffer-address-as-RTA anti-pattern is gone, and the legacy writer's slot-0 dst-address read is also gone.

## Friction

### Gaps

- **No reference port exists in TTNN.** A `grep -rln "create_program_spec" ttnn/cpp` returned zero hits during planning. This was the first port; I had no worked example to crib from. The migration guide's "Loopback (one node)" and "Multi-Core" examples have full `ProgramSpec` / `ProgramRunParams` skeletons, which were sufficient, but a real TTNN op port (with `compute_output_specs`, framework adapter integration, and existing-test compatibility constraints) would have shortened the planning time. The recipe's [Workflow at a glance] mentions an "Optional reference port" supplied by the invoker; that absence didn't block the port, but a future doc note pointing at this op once it lands would help the next porter.
- **The legacy ROW_MAJOR writer had a dead RTA slot.** Slot 1 (`num_units_per_row`) was pushed by the factory but never read by the kernel (the kernel reads from index 0 then jumps to index 2). I caught this only when assembling the `named_runtime_args` map and the indices didn't line up. The fix was to drop the dead slot in the port; a comment in the factory's per-core `named_runtime_args.push_back` block records the legacy→port slot mapping. Not a Metal 2.0 issue — just a legacy artifact the port surfaced. The recipe's [Legacy inventory] step caught the RTA list shape but not "this slot is unread by the kernel"; suggest a future addition under "Flags": match each legacy RTA against its kernel-side `get_arg_val` site, and call out any RTAs the kernel doesn't read.

### Confusion

- **`TensorArg::tensor` field expects `MeshTensor`, but op factories receive `ttnn::Tensor`.** The migration guide's examples show `.tensor = input_tensor` directly, but the `TensorArg` field is `std::reference_wrapper<const MeshTensor>` and `ttnn::Tensor` is a *wrapper around* `MeshTensor`, not the same type. The build fails with `no viable overloaded '='` against the `vector<TensorArg>` operator=, which doesn't immediately point at the type mismatch. The bridge is `Tensor::mesh_tensor() const&` (in `ttnn/api/ttnn/tensor/tensor.hpp:240`), and the framework adapter (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:618`) is the proof-by-example — it builds its `io_mesh_tensors` list via `std::cref(t.mesh_tensor())`. The factory must do the same. Suggestion for the migration guide: in the `TensorParameter` section's Metal 2.0 example, replace `.tensor = input_tensor` with `.tensor = std::cref(input_tensor.mesh_tensor())` when the input is a `ttnn::Tensor` (vs. a `MeshTensor` already, as in the standalone Metal 2.0 tests under `tests/tt_metal/`).

- **`tile_format_metadata`'s "operational rule for porters."** The migration guide is explicit: "copy this field from the legacy CB's `format_descriptors[i].tile`." The legacy `CBFormatDescriptor::tile` field was *not set* in this factory (defaults to nullopt), so the port leaves `tile_format_metadata` unset. The guide notes this is observably identical to setting `Tile()`. Confusing on first read because the prose builds up the field's importance before noting the default case is a no-op. Suggestion: lead with "if your legacy CB didn't set `.tile`, leave this unset and move on; otherwise…"

## Open items for downstream

### Cross-op kernel touches (forks)

| Kernel path (legacy → fork) | Owning op family | Remaining unmigrated consumers |
|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` → `..._metal2.cpp` | `eltwise/unary` | `eltwise/unary` (its own consumers); `data_movement/sharded/interleaved_to_sharded` (audit-GREEN, port pending); `data_movement/{tilize,untilize,untilize_with_unpadding}` sharded paths (some audit-GREEN, port pending) |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` → `..._metal2.cpp` | `data_movement/sharded` (in-family) | none — this op was the only consumer per inventory |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` → `..._metal2.cpp` | `data_movement/sharded` (in-family) | none — this op was the only consumer per inventory |
| `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` → `eltwise_copy_metal2.cpp` | shared kernel pool | broadly-shared (used by many ops; audit didn't enumerate all) |

The two in-family writer kernels could be deleted once this port verifies green. The broadly-shared reader and compute kernels' legacy copies stay in place until their other consumers also port to Metal 2.0.

### Per-op carry-over

- **`interleaved_to_sharded`** (next port): can reuse `reader_unary_sharded_metal2.cpp` directly. Its writer is in-family (`writer_unary_sharded.cpp`) — needs its own fork.
- **`tilize` and `untilize`** (later ports): sharded paths can reuse `reader_unary_sharded_metal2.cpp` and `writer_unary_sharded.cpp` → `_metal2` fork. Interleaved paths share `writer_unary_interleaved_start_id*.cpp` kernels which the audit flagged but did not fork (out of scope for this port).
- **`eltwise_copy_metal2.cpp`** can be the template for the (much larger) `tilize.cpp` / `untilize.cpp` compute kernel forks the later ports need.

### Doc-evolution suggestions

- The recipe's [Legacy Inventory → Kernels table] template asks for "RTAs (positional values)" — add a column "Kernel-side reads at indices N" so dead RTAs surface during inventory, not later.
- The migration guide's example uses `tensor.tensor_spec()` for `TensorParameter::spec`. For an op with a preallocated-output path (this op has one), the right spec is the *output tensor's* spec, not the input's, and that's the tensor passed through `tensor_return_value` to the factory. Worth a one-liner note in the [TensorParameter] section.

### Test coverage notes

Verification (build + tests) is the user's responsibility per the workspace's auto-memory. The port targets the same correctness surface the legacy factory served, so existing pytests under `tests/ttnn/unit_tests/operations/` should be applicable directly.

# Pre-port audit: `ttnn/cpp/ttnn/operations/reduction/generic/`

The op directory hosts two TTNN device-operations sharing kernels and the broader op family:

- **`ttnn::prim::ReduceDeviceOperation`** (`reduce_op_device_operation.hpp`)
  - `ReduceMultiCoreHProgramFactory` (`reduce_op_multi_core_h_program_factory.cpp`)
  - `ReduceMultiCoreWProgramFactory` (`reduce_op_multi_core_w_program_factory.cpp`)
  - `ReduceSingleCoreHwProgramFactory` (`reduce_op_single_core_hw_program_factory.cpp`)
- **`ttnn::prim::WelfordReduceDeviceOperation`** (`welford_reduce_device_operation.hpp`)
  - `WelfordReduceProgramFactory` (`welford_reduce_program_factory.cpp`)

The two device-operations are bundled in this audit per the audit doc's [multi-device-op rule](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/port_op_to_metal2_audit.md#feasibility-audit): both live in the same directory, share kernel sources, and were intentionally co-located in the existing codebase.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**GREEN** — all four program factories are on the `ProgramDescriptor` API, all kernels use `TensorAccessor` + Device 2.0 wrappers consistently, and no UNSUPPORTED Appendix A signal fires. Handoff to the recipe doc is appropriate after explicit user go-ahead.

A few small Device 2.0 holdovers and one cross-op writer fork are documented below but do not gate the port. The width-sharded H factory uses a borrowed-memory CB (legacy `CBDescriptor::buffer` set on the input + output), which is now LANDED in Metal 2.0 and translates via `DataflowBufferSpec::borrowed_from`.

## Porting prerequisites

### ProgramDescriptor API: **GREEN**

All four factories populate a `tt::tt_metal::ProgramDescriptor`, register `KernelDescriptor` / `CBDescriptor` / etc. via the descriptor-API surface, and return the descriptor from a `create_descriptor()` method on the factory struct. None of them use the imperative `host_api.hpp` builder API (`CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs`).

`reduce_op_device_operation.hpp` declares the three Reduce factories as a `std::variant`-backed `program_factory_t`; `welford_reduce_device_operation.hpp` does the same for Welford.

### Device 2.0 DM: **YELLOW (isolated holdovers)**

Kernels broadly use the Device 2.0 wrappers (`Noc`, `CircularBuffer`, `TensorAccessor`, `noc.async_read(...)` etc.). A small number of CB-index-keyed free functions remain — each is the standard 1-line mechanical replacement.

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | 26 | `get_tile_size(cb_id_in0)` | `cb_in0` (`CircularBuffer`) |
| `device/kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | 37 | `get_tile_size(cb_id_in0)` | `cb_in0` (`CircularBuffer`) |
| `device/kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | 34 | `get_tile_size(cb_id_in0)` | `cb_in0` (`CircularBuffer`) |
| `device/kernels/dataflow/writer_welford_hw.cpp` | 55 | `get_tile_size(cb_partial)` | `cb_partial_obj` (`CircularBuffer`) |
| `device/kernels/dataflow/writer_welford_hw.cpp` | 56 | `get_tile_size(cb_out)` | `cb_out_obj` (`CircularBuffer`) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | 19 | `get_local_cb_interface(cb_id_out).fifo_page_size` | `cb` (`CircularBuffer`) — cross-op kernel |

**Recommendation:** fold the four in-tree holdovers into the Metal 2.0 port as port-time cleanup (default per the audit doc). The `writer_unary_interleaved_start_id.cpp` site is in a cross-op shared kernel that will be addressed via the fork-with-`_metal2`-suffix path; the holdover is addressed in the forked copy. The `get_local_cb_interface(cb_id).fifo_page_size` call is more invasive than the others because there is no direct DFB-wrapper equivalent of `fifo_page_size` on the public API surface; if the cross-op fork happens, the simplest swap is `cb.get_page_size()` (or equivalent on `DataflowBuffer`) — to be confirmed when the kernel is forked.

### TensorAccessor usage: **GREEN**

Three dataflow kernels are in-scope tensor-touching kernels; all use `TensorAccessor(args, addr)` with `TensorAccessorArgs<N>()` plumbing — the standard pre-Metal 2.0 idiom that translates cleanly via `TensorParameter` + `TensorBinding`:

- `reader_unary_reduce_universal_start_id.cpp` — reads input
- `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` — reads input
- `writer_welford_hw.cpp` — writes output
- The cross-op `writer_unary_interleaved_start_id.cpp` — writes output

The width-sharded H-reduce reader (`reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp`) does **not** use `TensorAccessor`. **Causal-link gate fires here:** the sharded reader reads tensor data via a borrowed-memory CB (`cb_in1` is the legacy dynamic CB built on top of the input's L1 buffer; see Step 0.2's Dynamic CircularBuffer entry below). This is the intended pattern — the borrowed-memory DFB *is* the tensor access — and the port handles it via `DataflowBufferSpec::borrowed_from`. The lack of `TensorAccessor` here is not a Check 3 RED.

Compute kernels do not directly access tensor data (they only consume from / produce to CBs) — out of Check 3 scope.

## Feature compatibility check

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `experimental::GlobalCircularBuffer`, no `CBDescriptor::global_circular_buffer` field anywhere in the family. |
| Dynamic CircularBuffer (CB on borrowed memory) | GREEN | LANDED in Appendix A. Two sites in `reduce_op_multi_core_h_program_factory.cpp` (width-sharded path); see detail below. |
| CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` field set on any `CBDescriptor`, no `set_address_offset` call, no `cb_descriptor_from_sharded_tensor` helper usage. |
| Aliased Circular Buffers | N/A | Every `format_descriptors` initializer in the four factories is single-element. No multi-key `data_format_spec` or multi-index `set_page_size` chain. |
| GlobalSemaphore | N/A | No semaphores at all in any factory — no `SemaphoreDescriptor`, no `experimental::CreateGlobalSemaphore`, no `GlobalSemaphore` parameters. |
| Non-zero semaphore initial value | N/A | No semaphores used by the op family. |
| `ArgConfig::Runtime*` tensor-accessor flavors | GREEN | All `TensorAccessorArgs` constructions are the single-argument form `TensorAccessorArgs(*buffer)`. No `ArgConfig::Runtime` token anywhere in the family. |
| `UpdateCircularBuffer*` | N/A | None of the four factories sit on the legacy override-runtime-arguments path that would house per-execution CB-size updates. |

### Dynamic CircularBuffer (CB built on borrowed Buffer memory): **GREEN** (LANDED)

**Signal:** `CBDescriptor::buffer` field set to a non-null `Buffer*` (the descriptor-API form).

**Sites:**

- `reduce_op_multi_core_h_program_factory.cpp:111` — `src1_cb_index` (input shard CB): `.buffer = a.buffer()` (the input tensor's L1 shard memory). Used by the width-sharded reader.
- `reduce_op_multi_core_h_program_factory.cpp:148` — `output_cb_index` (output shard CB): `.buffer = output.buffer()` (the output tensor's L1 shard memory). Used by the width-sharded writer.

Both fire only when `use_width_sharding == true` (input + output both width-sharded). The other three factories never set `.buffer` on a `CBDescriptor`.

**Expected resolution:** translate each borrowed-memory CB to a `DataflowBufferSpec` with `borrowed_from = <tensor_parameter_name>` referencing the `TensorParameter` whose buffer backs the DFB. The DFB's L1 address resolves at runtime from the corresponding `TensorArg` (no separate `dfb_run_params` entry needed). Confirmed LANDED in the audit doc's Appendix A.

## Path forward

No RED entries to gate the port. Items the recipe step will need to handle:

- **Borrowed-memory DFBs** on the H factory's width-sharded path. Translates 1:1 via `borrowed_from`; the recipe doc and migration guide both cover this pattern.
- **Cross-op writers.** Two cross-op kernels are referenced:
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — used by W, single-core HW, and the interleaved branch of H + the Welford W/H factories.
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` — used by the width-sharded branch of H.

  Both are shared across many unmigrated ops. Per the [Caution: Modifying a shared dataflow kernel](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel) entry, the supported path during the bulk-port window is **fork with `_metal2` suffix**. The writer used by `writer_welford_hw.cpp` is in this op's own directory and is therefore freely modifiable.
- **Device 2.0 holdovers.** The six `get_tile_size` / `get_local_cb_interface` sites listed above are mechanical port-time cleanups. The five in-tree sites swap to wrapper-method form during the port; the cross-op site lands on the forked `_metal2` copy.
- **Multi-variant factory.** The Welford factory has three branches (W / H / HW) inside a single `create_descriptor`; the Metal 2.0 port keeps the same shape via the [Multi-variant factory](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-multi-variant-factories) pattern.
- **Multi-`KernelDescriptor` work split.** Both Reduce H and Reduce W (and the W and H branches of Welford) create a second compute `KernelDescriptor` for `core_group_2` with a different per-group CTA value when the work split has two groups. Per the [Anti-pattern: Demoting per-group CTA to RTA](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) entry, the port preserves the per-group CTA multiplicity — two `KernelSpec`s of the same source, two `WorkUnitSpec`s, sharing the input/output DFBs as multi-bindings.
- **Conditional DFB bindings.** The Reduce-negate path adds `cb_acc` (c_4) and `cb_ineg` (c_5) only when `operation_attributes.negate == true`. The Welford W path adds `cb_scaled` (c_20) only when `do_scale == true`. The Welford HW path adds `cb_partial` (c_21) and `cb_combined` (c_22) unconditionally for that variant. Per the [Conditional / optional DFB bindings](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings) pattern, the port binds the DFBs unconditionally on the host and gates the kernel-side uses with `if constexpr` on a CTA — **not** with `#ifdef`.
- **Same DFB layout produced by Welford HW writer + consumed by Welford HW compute (cb_combined, cb_partial).** Two cross-kernel DFBs that flow from compute → writer (cb_partial) and writer → compute (cb_combined). Straightforward producer/consumer multi-binding.
- **Selectable compute kernel source string.** Both Reduce (`reduce[_w|_h|_hw]_neg`?`.cpp`) and Welford (`welford_reduce_[w|h|hw].cpp`) compute their compute-kernel path via string concatenation on the variant. The port preserves this by computing the `KernelSpec::source` path inline.
- **Post-mul scaler CTA on Reduce.** `post_mul_scaler_bits` is the 4th positional CTA on every Reduce compute `KernelDescriptor`. Named in the port as `post_mul_scaler_bits` (or similar) — only consumed via `if defined(REDUCE_POST_MUL)`. Same `#ifdef`-vs-`if constexpr` consideration as conditional DFB bindings; the right shape is converting `REDUCE_POST_MUL` (currently a kernel `define`) to a named CTA. **This is a kernel modification beyond the sanctioned arg-retrieval / DFB-wrapper / TensorAccessor swaps.** Worth flagging up to the user as a planning-time question — does the port replace the `#ifdef REDUCE_POST_MUL` blocks with `if constexpr (use_post_mul)`, or leave them as-is on the kernel side?

## Questions for the user

1. **`REDUCE_POST_MUL` kernel define → CTA conversion?** The Reduce kernels (`reduce.cpp`, `reduce_h_neg.cpp`, `reduce_w_neg.cpp`, `reduce_hw_neg.cpp`) gate the post-multiplication scaling with `#ifdef REDUCE_POST_MUL` blocks. The legacy factory sets the define when `post_mul_scaler != 1.0f`. Strict scope-boundary reading says "the only sanctioned kernel-side changes are arg-retrieval, DFB-wrapper construction, and TensorAccessor token" — which would leave the `#ifdef REDUCE_POST_MUL` blocks (and their corresponding `defines` map on the `KernelSpec`) intact. That's the conservative answer. Confirm? (Note: the existing kernel uses `#ifdef` to gate logic, *not* to gate optional CB usage — so the anti-pattern entry against `#ifdef`-gated DFB references doesn't apply directly. This is a legacy `#ifdef` that's load-bearing for the post-mul code path; preserving it is sanctioned.)

2. **Cross-op writer fork path.** Both `writer_unary_interleaved_start_id.cpp` (eltwise/unary) and `writer_unary_sharded.cpp` (data_movement/sharded) are used by many unmigrated ops. The supported path is to fork each into a `_metal2`-suffixed copy in the same directory and update the ported factories to reference the new file. Confirm that's the chosen path? (The alternative — in-place modification — requires every sibling consumer op to co-migrate in the same PR, which is far beyond this port's scope.)

3. **Welford HW writer (`writer_welford_hw.cpp`) is op-local.** No fork needed — it's modified in place. Confirm? (Cross-op kernels are explicitly out-of-op-directory; this writer lives in `device/kernels/dataflow/` of the reduction/generic directory, so it's in-scope.)

4. **Single combined PR for both device-operations.** The two device-operations (`ReduceDeviceOperation` and `WelfordReduceDeviceOperation`) share the directory + a couple of kernels + the cross-op writers. The audit treats them as one bundled body of work. Confirm? (Splitting them in two would either re-fork the cross-op writers twice or coordinate between two PRs.)

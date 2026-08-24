# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/reshape_view`

> Audit cleared all gates for **both** factories. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A)

**Recipe docs:** `355760227dd 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources` *(carry this line into the port report's Provenance section)*

Both factories of the one `DeviceOperation` port together to `ProgramSpecFactoryConcept`. The tiled factory additionally carries an op-owned tensor. (An earlier RED on the RM kernel's Device 2.0 compliance has been cleared by a kernel migration — the whole op is now GREEN, so there is no factory-subset scoping.)

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`):

- **`ReshapeViewRMProgramFactory`**
  - **Current concept:** `descriptor` (`create_descriptor` → `ProgramDescriptor`).
  - **Op-owned tensors:** none.
  - **Target concept:** `ProgramSpecFactoryConcept`.
- **`ReshapeViewTiledProgramFactory`**
  - **Current concept:** `WorkloadDescriptor` (secretly SPMD — one `ProgramDescriptor` built once and replicated across `tensor_coords` ranges, `device/reshape_tiled_program_factory.cpp:471-488`; collapses to single-program).
  - **Op-owned tensors:** **yes** — the host-computed input→output page-mapping tensor, parked on `workload_descriptor.buffers` (`device/reshape_tiled_program_factory.cpp:459-461`). Carried natively by the target concept.
  - **Target concept:** `ProgramSpecFactoryConcept`, with op-owned tensors.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none` `TensorParameter relaxation` (it is `none` for both) · `get_dynamic_runtime_args` (absent). **Present but non-gating, leave as-is:** a custom `compute_program_hash` shared by both factories (`device/reshape_device_operation.cpp:48-63`) — do not touch it. No `override_runtime_arguments`, no pybound `create_descriptor`.

## Construct — to do

**Tensor bindings** (per binding — all Case 1, mechanical):

- RM `src` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(src_args, src_addr)` (`device/device/rm_reshape_interleaved.cpp:87`). The `Buffer*` binding-form delivery (`reshape_rm_program_factory.cpp:259`) and its `TensorAccessorArgs` plumbing disappear.
- RM `dst` — **Case 1** → `TensorParameter`; kernel builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(dst_args, dst_addr)` (`rm_reshape_interleaved.cpp:88`).
- Tiled `input` — **Case 1** → `TensorParameter`; reader builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(input_args, input_addr)` (`reader_reshape_tiled.cpp:36`).
- Tiled `mapping` (**op-owned tensor**) — **Case 1** → bind the op-owned mapping tensor as a `TensorParameter` carried by the workload; reader builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(map_args, map_addr)` (`reader_reshape_tiled.cpp:37`). Keep the workload-scoped lifetime (the mapping is fully determined by the hashed input/output shapes; `recreate_mapping_tensor` stays ignored, excluded from the hash).
- Tiled `output` — **Case 1** → `TensorParameter`; writer builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(output_args, output_base_addr)` (`writer_reshape_tiled.cpp:30`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor passes a 3rd argument.

**CB endpoints:**
- RM `c_0` (src0), `c_1` (src1) — **self-loop** (each touched by one kernel instance as scratch; bind that kernel PRODUCER + CONSUMER). Always allocated.
- RM `c_2` (src2), `c_3` (src3) — **self-loop**, and make the DFB spec **conditional on `can_use_dual_kernel`** (these CBs and the second kernel instance exist only in that config — `reshape_rm_program_factory.cpp:187-217`).
- Tiled `c_0` (mapping) — legal 1:1: reader PRODUCER, writer CONSUMER.
- Tiled `c_1` (input) — legal 1:1: reader PRODUCER, writer CONSUMER.
- Tiled `c_2` (output / working scratch) — **self-loop**: touched only by the writer (`writer_reshape_tiled.cpp:38-39,84`); bind the writer PRODUCER + CONSUMER.

## Watch for

- **CB endpoints (multi-binding):** none in either factory.
- **RM dual-instance work-split with disjoint CBs:** the RM factory pushes the same `kernel_source` into a reader-config and a writer-config KernelDescriptor over one `total_cores` (`reshape_rm_program_factory.cpp:175-217`). It looks like the canonical dual-instance work-split, but the two instances use **disjoint** CB sets (0/1 vs 2/3, via the CTA swap at 207-209) — so there is no shared CB to assign 1P+1C; each CB self-loops. Preserve the CB→instance CTA mapping in the port.
- **RM idle-core handling:** the factory creates the kernel on **all** cores in `total_cores`, marking spare cores idle with a trailing `nop=1` RTA and `0u` buffer slots (`reshape_rm_program_factory.cpp:225-227`); the kernel returns early on `nop==1` (`rm_reshape_interleaved.cpp:83-85`) **before** building any `TensorAccessor`. With typed `TensorParameter` bindings the framework still delivers the base to those cores, but the accessor is never constructed there, so it is harmless — confirm the binding model accepts an idle core that early-returns (and keep the `nop` RTA as a named arg).
- **Cross-op / shared kernels:** all three kernels `#include ttnn/operations/data_movement/common/kernels/common.hpp` and call `enhanced_noc_async_read` / `enhanced_noc_async_write` / `tt_memmove` — all already Device 2.0 native (`Noc`-first). No donor-side change, no fork. Each factory instantiates **only its own** kernels — no borrowed kernel files, no `_metal2` fork to create or reuse. (Ignore the quasar copy under `experimental/quasar/reshape_view/` — not a source or precedent.)
- **RTA varargs:** none — every RTA is a fixed, nameable field. Prefer named RTAs throughout (RM: `src_addr`,`dst_addr`,`source_read_size_bytes`,`read_start_page`,`read_end_page`,`write_start_page`,`write_start_offset`,`nop`; tiled reader: `input_addr`,`map_addr`,`start_output_page_idx`,`end_output_page_idx`; tiled writer: `output_base_addr`,`start_output_page`,`end_output_page`).
- **Shared host code:** `reshape_device_operation.cpp` (custom hash, validate, `compute_output_specs`) is common to both factories — leave the custom hash intact and change only what the two ProgramSpec wirings require.

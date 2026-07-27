# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — single factory, `MorehSumBackwardOperation::create_descriptor` returning `ProgramDescriptor` (`device/moreh_sum_backward_program_factory.cpp:66`).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` / runtime-args update · pybind `create_descriptor` · smuggled pointer / migration-risky pybind (`Is safe to port? = yes`). All `no` on the fresh sheet.

**Dispatch note:** `tensor_args_t` always carries the mandatory `const Tensor& output_grad`, so the MetalV2 factory adapter's MeshDevice-from-tensor lookup always succeeds — no tensorless-dispatch concern.

## Construct — to do

**Tensor bindings** (per binding):

- `output_grad` — **Case 1** (via `TensorAccessor`) → express as a `TensorParameter` / `TensorBinding`. Kernel builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(output_grad_args, output_grad_addr)` (`device/kernels/reader_moreh_sum_backward.cpp:84`). Remove the `Buffer*` RTA (`program_factory.cpp:252`) and the `TensorAccessorArgs<1>` CTA plumbing (`program_factory.cpp:176`, `reader...cpp:36`).
- `input_grad` (the output tensor) — **Case 1** (via `TensorAccessor`) → bind as a `TensorParameter` / `TensorBinding`. Kernel builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(input_grad_args, input_grad_addr)` (`device/kernels/writer_moreh_sum_backward.cpp:23`). Remove the `Buffer*` RTA (`program_factory.cpp:261`) and the `TensorAccessorArgs<0>` CTA (`program_factory.cpp:187`, `writer...cpp:12`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none (all sites are 2-arg).

**CB endpoints:** all legal — bind each CB as a `DataflowBufferSpec` with its natural single producer + single consumer; no self-loop, no 1P+1C reassignment, no multi-binding flag, no dead-CB drop.
- `c_0` (input): PRODUCER = reader, CONSUMER = compute.
- `c_1` (zero tile): PRODUCER = reader (`fill_cb_with_value`), CONSUMER = compute.
- `c_16` (output): PRODUCER = compute, CONSUMER = writer.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** the op owns all three kernels; no borrowed kernel files. Kernels `#include` the shared moreh helper library `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (`ArgFetcher`, `fill_cb_with_value`) and `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`. These are already on Metal-2.0 idioms (`DataflowBuffer` by value); if a Metal-2.0 rewrite of these shared headers is needed, it is a single rewrite shared with the moreh family and other co-borrowers — coordinate rather than fork.
- **RTA varargs:** `reader_moreh_sum_backward.cpp:44-57` reads three **variable-count** RTA blocks (`output_grad_dim`, `input_grad_dim`, `need_bcast_dim`, each length `input_grad_rank`) in counted loops via `arg_fetcher.get_next_arg_val<uint32_t>()`. Port these via the kernel-side vararg mechanism (per kernel-side whitelist rule 4) — do **not** try to name each element. Name the fixed leading reader scalars (`output_grad_addr`, `num_output_tiles`, `start_id`) and the three writer scalars normally.

## Context

The op's kernels are already written against the Metal-2.0 kernel-side API (`DataflowBuffer`, `Noc`, `TensorAccessor`, `api/dataflow/*` headers), so the bulk of the port is the host-side factory rewrite — `ProgramDescriptor` + `CBDescriptor` + `KernelDescriptor` → the `MetalV2FactoryConcept` spec — plus rebinding the two tensor addresses from `Buffer*` RTAs to typed `TensorParameter`/`TensorBinding`s and dropping the now-redundant `TensorAccessorArgs` CTAs.

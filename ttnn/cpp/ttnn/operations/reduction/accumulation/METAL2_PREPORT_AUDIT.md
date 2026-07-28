# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/accumulation/`

> Feasibility audit for porting the **accumulation** op (the `AccumulationDeviceOperation` shared by cumsum / cumprod / ema) to Metal 2.0. Performed against `main`'s `ProgramSpecFactoryConcept`. Overall: **GREEN**.

## Status summary

| Field | Value |
|---|---|
| Op directory | `ttnn/cpp/ttnn/operations/reduction/accumulation/` |
| Overall | **GREEN** |
| DOps / Factories | `AccumulationDeviceOperation` → single `AccumulationProgramFactory` (`program_factory_t = std::variant<AccumulationProgramFactory>`) |
| Prereq — ProgramDescriptor | **Yes** — `AccumulationProgramFactory::create_descriptor` populates a `tt::tt_metal::ProgramDescriptor` (`accumulation_program_factory.cpp:42,53`); no imperative `host_api.hpp` builder calls |
| Prereq — Device 2.0 (own + donor) | **Yes** — dataflow kernels use `TensorAccessor` / `TensorAccessorArgs<0>()` (not legacy `InterleavedAddrGen`); compute kernel uses kernel-side `CircularBuffer` wrappers. No D1.0 holdovers gating the swaps. |
| Prereq — Cross-op escapes | **Ok** — all 3 kernel `.cpp` are accumulation-owned; no out-of-dir kernel sources |
| Feature — Variadic-CTA / UNSUPPORTED | **Ok** — no UNSUPPORTED Appendix-A feature in use (no per-execution CB-size updates, no CTA varargs) |
| TTNN — Port Type | `ProgramSpecFactoryConcept` (single-program, no op-owned resources, strict tensors) |
| TTNN — infra++ | none — no op-owned tensors; no `GlobalSemaphore`; no semaphores at all |

## Result

**GREEN → brief issued.** Every gate clears: ProgramDescriptor API, Device 2.0 kernels, no UNSUPPORTED feature, and the op fits `main`'s single implemented Metal 2.0 factory concept (single-program, no op-owned device resources). Port may begin on explicit user go-ahead.

## Gate detail

- **ProgramDescriptor:** GREEN. `create_descriptor` returns one `ProgramDescriptor` (`desc.cbs`, `desc.kernels`). No `MeshWorkload` / `WorkloadDescriptor` wrapper.
- **Device 2.0:** GREEN. Reader/writer already on `TensorAccessor` (`accumulation_reader.cpp:16,36`, `accumulation_writer.cpp:13,30`); compute on kernel-side CB wrappers (`accumulation_compute.cpp:26-28`).
- **Feature compatibility:** GREEN. No UNSUPPORTED feature.
- **Factory concept:** GREEN — `ProgramSpecFactoryConcept`. Single-program (one `ProgramDescriptor` per dispatch), no factory-allocated device resources (no `CreateBuffer` / `MeshTensor` / `GlobalSemaphore` in the factory body).

## Port-work summary (mirrors brief)

1. **Tensor bindings — Case 1 (re-express).** Both tensor accessors are the standard interleaved page-access pattern. `input` (reader) and `output` (writer) re-express from `TensorAccessorArgs<0>()` + a buffer-address RTA (`input_base_addr = get_arg_val<uint32_t>(0)`, `output_base_addr = get_arg_val<uint32_t>(0)`) to `TensorParameter` / `TensorBinding` → kernel-side `TensorAccessor(ta::input)` / `TensorAccessor(ta::output)`. No Case-2 bridge needed.
2. **Magic CB indices → DFB bindings.** `CB_IN` / `CB_OUT` / `CB_ACC` become `DataflowBufferSpec`s with `DFBBinding`s; kernel-side `CircularBuffer cb_*_obj(CB_*)` → `DataflowBuffer dfb_*(dfb::*)`.
3. **Positional / remaining RTAs → named.** `num_rows_per_core`, `tiles_per_row`, `input_tile_offset`, `start_id`, `low_rank_offset`, `high_rank_offset`, `flip`, etc. → named `get_arg(args::<name>)`.
4. **Delete custom `compute_program_hash`.** `AccumulationDeviceOperation::compute_program_hash` exists (`accumulation_device_operation.hpp:64`, `.cpp:105`) → delete, revert to default (sanctioned exception; see TTNN integration doc).

## Heads-ups (mirrors brief)

- **Work-split multiplicity — preserve it.** The factory emits a compute `KernelDescriptor` per core group (`compute_desc_1` for `core_group_1` at `:190`; `cd2` for `core_group_2` at `:202-206`) from `split_work_to_cores`. Port to **one `KernelSpec` per legacy compute `KernelDescriptor`** with per-group CTAs reproduced — do **not** collapse to a single KernelSpec by demoting a per-group CTA to an RTA. (Reader/writer are single-instance across `all_cores`.)
- **`CB_ACC` is compute-internal** — produced and consumed within the compute kernel (`reserve_back`/`push_back`/`wait_front`/`pop_front` on `cb_acc_obj`, `:37-94`). Its DFB has the compute KernelSpec as both PRODUCER and CONSUMER (self-loop binding).
- **Pybind:** check the cumsum / cumprod / ema nanobind files for a pybound `create_descriptor` hook on the factory; if present, deletion is forced (build-breaking once `create_descriptor` → `create_program_spec`). Record under Handoff points.

## Team-only

- TA convertibility: both bindings convertible (Case 1). No exotic access patterns.
- Out-of-dir coupling: none.
- Relaxation candidates: not assessed at audit; keep strict during port, capture any in the report.

## TTNN ProgramFactory

### Concept
`ProgramSpecFactoryConcept` — single-program, no op-owned resources.

### Fit
- Single vs multi-program: single — one `ProgramDescriptor` per dispatch; no `MeshWorkload`.
- Op-owned device resources: none — the factory allocates no scratch tensors/semaphores.
- Tensor-arg matching: strict (default).
- Legacy-to-Metal-2.0 shape: 1:1 with legacy single-program shape.

### Custom compute_program_hash
Present at `accumulation_device_operation.hpp:64` / `.cpp:105` → port deletes it.

### Stop signals
None.

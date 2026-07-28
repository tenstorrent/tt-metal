# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/reduction/accumulation/`

> Audit cleared all gates (GREEN). Your actionable input; full record in METAL2_PREPORT_AUDIT.md.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓ · Factory concept available ✓

## Plan — factory concept

Implement `ProgramSpecFactoryConcept` — single-program, no op-owned resources, strict tensors (→ TTNN integration doc). The legacy `AccumulationProgramFactory::create_descriptor` becomes `create_program_spec` returning `ProgramArtifacts{spec, run_params}`. 1:1 with the legacy single-program shape.

## Construct — to do

**Tensor bindings** (per binding, both Case 1 — re-express, no bridge):
- `input` → `TensorParameter` / `TensorBinding` on the reader; kernel builds `TensorAccessor(ta::input)` (replaces `TensorAccessorArgs<0>()` + `input_base_addr` RTA).
- `output` → `TensorParameter` / `TensorBinding` on the writer; kernel builds `TensorAccessor(ta::output)` (replaces `TensorAccessorArgs<0>()` + `output_base_addr` RTA).

**DFBs:** `CB_IN` / `CB_OUT` / `CB_ACC` → `DataflowBufferSpec` + `DFBBinding`s. `CB_ACC` is compute-internal — bind it on the compute KernelSpec as both PRODUCER and CONSUMER (self-loop binding).

**Work-split multiplicity — preserve it:** the legacy factory emits a compute `KernelDescriptor` per core group (`compute_desc_1` → `core_group_1`; `cd2` → `core_group_2`). Port to **one compute `KernelSpec` per legacy compute `KernelDescriptor`**, per-group CTAs reproduced. Do NOT collapse to one KernelSpec by demoting a per-group CTA to an RTA (anti-pattern). Reader/writer are single-instance over all cores.

**Named args:** convert remaining positional RTAs/CTAs to named (`get_arg(args::<name>)`): `num_rows_per_core`, `tiles_per_row`, `input_tile_offset`, `start_id`, `low_rank_offset`, `high_rank_offset`, `flip`, and the compute CTAs.

**Custom hash:** delete the custom `AccumulationDeviceOperation::compute_program_hash` (`accumulation_device_operation.hpp:64` / `.cpp:105`) → default (sanctioned exception).

## Watch for

- **Pybind:** check cumsum / cumprod / ema nanobind files for a pybound `create_descriptor` hook on the factory; if present, delete it (build-breaking once `create_descriptor` → `create_program_spec`) and record under Handoff points.
- **Three consumer ops** share `AccumulationDeviceOperation` (cumsum, cumprod, ema) — the port affects all three; tests live under `tests/ttnn/unit_tests/operations/reduce/` (`test_cumsum.py`, `test_cumprod.py`, `test_ema.py`).
- Notable constructs: `CB_ACC` self-loop (above). No aliased CBs, no borrowed-memory DFBs, no semaphores.

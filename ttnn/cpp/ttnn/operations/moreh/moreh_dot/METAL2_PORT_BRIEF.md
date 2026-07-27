# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_dot`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — single-core, single factory (`MorehDotOperation::create_descriptor` in `device/moreh_dot_program_factory.cpp`).
- **Op-owned tensors:** none (a `descriptor` op cannot carry them).
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind (`Is safe to port? = yes`, `Smuggled pointer = no`). All `no` on the readiness sheet, cross-checked against the code.

## Construct — to do

**Tensor bindings** (per binding — all **Case 1**, via `TensorAccessor`):

- `input_a` (`src0_buffer`) — express as `TensorParameter` / `TensorBinding`; reader builds `TensorAccessor(tensor::name)` in place of `TensorAccessor(src0_args, src0_addr)`. The `Buffer*` at reader RTA index 0 and the `TensorAccessorArgs(src0_buffer).append_to(...)` CTA plumbing (`moreh_dot_program_factory.cpp:118,127`) both disappear.
- `input_b` (`src1_buffer`) — same, reader RTA index 1 / CTA `TensorAccessorArgs(src1_buffer)` (`:119,127`).
- `output` (`dst_buffer`) — same, writer RTA index 0 / CTA `TensorAccessorArgs(dst_buffer)` (`:131,139`); writer builds `TensorAccessor(tensor::name)` in place of `TensorAccessor(dst_args, dst_addr)`.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no site passes an explicit page size.

**CB endpoints:**

- Self-loop `c_24` (`CBIndex::c_24`) and `c_25` (`CBIndex::c_25`) — each is touched **only** by the compute kernel (produced and consumed there), so bind the compute kernel as **both** PRODUCER and CONSUMER (legal on Gen1 for a compute self-loop).
- `c_0`, `c_1`, `c_2`, `c_16` are plain 1P+1C (reader/compute-produce, compute/writer-consume) — bind one PRODUCER + one CONSUMER, no special action.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader on any CB.
- **Cross-op / shared kernels:** reader calls `kernel_lib/reduce_helpers_dataflow` (`calculate_and_prepare_reduce_scaler`), compute calls `kernel_lib/reduce_helpers_compute` (`compute_kernel_lib::reduce`). Both take CB indices as `uint32_t` template NTTPs — the `dfb::name` constexpr cast covers this, so the named tokens pass cleanly. These helpers are `kernel_lib`-team owned; do **not** rewrite them as part of this op's port.
- **RTA varargs:** none — name each RTA directly (reader: `src0_addr`, `src1_addr`, `num_tiles`, `start_id`, `mask_h`, `mask_w`; writer: `dst_addr`, `num_tiles`, `start_id`; compute: `per_core_block_cnt`). Note the compute kernel's second legacy RTA (`1u`, index 1) is dead — drop it, don't name it (see the audit's Misc anomalies).

# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward`

> Audit cleared all statically-auditable gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Precondition before you start:** the readiness sheet could not be fetched in the audit session. Confirm `moreh/moreh_norm_backward` reads `Is able to port? == yes` / `Is safe to port? == yes` on the "Operations analysis" sheet (main session) first. The code cross-check clears every shape conjunct and shows no smuggled pointer, so it is expected to agree.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (code cross-check; sheet confirmation pending) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (`create_descriptor` returns `ProgramDescriptor`, `device/moreh_norm_backward_program_factory.cpp:58`).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · migration-risky pybind — all `no`/absent in the code.
- **Single factory, single DeviceOperation.** No config/sharding branches; the only structural split is the `core_group_1` / `core_group_2` work-split (two compute `KernelDescriptor`s over disjoint node sets — ordinary 1:1 per node, not a two-toucher case).

## Construct — to do

**Tensor bindings** (per binding) — all **Case 1** (fed into a `TensorAccessor`):

- `input` — express as `TensorParameter`/`TensorBinding`; reader builds `TensorAccessor(tensor::input)` in place of `TensorAccessor(input_args, input_addr)` (reader `:86`). Drop the `Buffer*` RTA (`program_factory.cpp:255`) and the `TensorAccessorArgs(*input.buffer())` CTA (`:178`).
- `output` — same shape; reader `:89`; RTA `:256`; CTA `:179`.
- `output_grad` — same shape; reader `:92`; RTA `:257`; CTA `:180`.
- `input_grad` — writer builds `TensorAccessor(tensor::input_grad)` in place of `TensorAccessor(input_grad_args, input_grad_addr)` (writer `:25`). Drop the `Buffer*` RTA (`:268`) and the `TensorAccessorArgs(*input_grad.buffer())` CTA (`:191`).

All four are the mechanical, low-risk Case-1 conversion — no raw-pointer arithmetic to preserve.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none (every `TensorAccessor` is already 2-arg).

**CB endpoints:**

- **Self-loop** `c_24`–`c_31` — the 8 compute-only intermediates (xpow, logx, exp_lxmd, correct_xpow, tmp4, tmp5, recip_ypow, sign). Single toucher (compute fills and drains each); bind the compute kernel as both PRODUCER and CONSUMER. Legal on Gen1 for compute.
- **Plain 1:1, no action** — `c_0` (input), `c_1` (output), `c_2` (output_grad), `c_3` (decimal): reader-produced, compute-consumed. `c_16` (input_grad/dx): compute-produced, writer-consumed.
- No multi-binding flag, no dead-CB drop.

## Watch for

- **RTA varargs (reader):** three consecutive count-bounded blocks — `output_grad_dim` (reader `:52-55`), `input_grad_dim` (`:57-60`), `need_bcast_dim` (`:62-65`), each `for i < input_grad_rank`. `input_grad_rank` is a CTA but the loop count varies per instantiation → port these as **RTA varargs** (kernel-side vararg mechanism), not named args. Name the six leading reader scalars (`input_addr`, `output_addr`, `output_grad_addr`, `decimal`, `num_output_tiles`, `start_id`, `:43-50`) and the writer/compute RTAs (all fixed distinct fields).
- **Cross-op / shared kernels:** `moreh_common.hpp` (dataflow + compute, under `ttnn/cpp/ttnn/kernel/`) is shared across the moreh family and already `DataflowBuffer`-based. No borrowed kernel *files* (all three kernels are op-owned). Any Metal 2.0 token rewrite of the shared header must land as one unit across its co-borrowers — but this op consumes only `fill_cb_with_value` (dataflow) and the compute helper family, all of which take `DataflowBuffer` objects, so no kernel-side signature change is forced here.

# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (single factory, `MorehAbsPowOperation (single-descriptor)`).
- **Op-owned tensors:** none — carried by the target concept, nothing to declare.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · smuggled pointer / migration-risky pybind — all `no` on the readiness sheet and in code.
- **Dispatch safety:** `tensor_args_t.input` is a required `const Tensor&` (non-optional). The op is never dispatched with empty `tensor_args`, so the MetalV2 factory adapter can always source the MeshDevice from the input tensor — no tensorless-dispatch block.

## Construct — to do

**Tensor bindings** (per binding):

- `input` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`. Drop the reader RTA arg 0 (`input.buffer()`) and the `TensorAccessorArgs(*input.buffer())` CT-args plumbing (`program_factory.cpp:183`); the reader's `input_addr` + `TensorAccessorArgs<0>()` (`reader_moreh_abs_pow.cpp:12,26-27`) both disappear.
- `output` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`. Drop the writer RTA arg 0 (`output.buffer()`) and the `TensorAccessorArgs(*output.buffer())` CT-args plumbing (`program_factory.cpp:194`); the writer's `output_addr` + `TensorAccessorArgs<0>()` (`writer_moreh_abs_pow.cpp:14,23-24`) both disappear.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — both accessors are 2-arg; nothing to drop.

**CB endpoints:**
- Self-loop `c_24`, `c_25`, `c_26`, `c_27` (compute-only intermediates `xabs`/`xpow`/`logx`/`exp_lxmd`) — bind the compute kernel as **both** PRODUCER and CONSUMER.
- Assign **1P+1C** on `c_0` (reader→compute), `c_1` (reader→compute), `c_2` (reader→compute), `c_16` (compute→writer). `c_3` (mask_w) is also **1P+1C** (reader→compute) but only exercised under `do_mask_w` (`origin_w % 32 != 0`); bind it 1P+1C — it is **not** a dead CB (live under the mask config; keep the allocation).

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader, no dual-instance work-split. Straightforward 1P+1C / self-loop op.
- **Cross-op / shared kernels:** reader includes `ttnn/kernel/dataflow/moreh_common.hpp`; compute includes `ttnn/kernel/compute/moreh_common.hpp` (shared `moreh`-family pool). The helpers this op calls already take `DataflowBuffer`, so no donor rewrite is forced — but if you do touch these shared headers, they are a family-wide port unit, not an in-op edit.
- **RTA varargs:** none — every kernel reads a fixed set of named args via a sequential `i++` counter; name each (reader: `input_addr`, `input_is_dram`, `decimal`, `num_rows_per_core`, `Wt`, `tile_offset`, `origin_w`; writer: `output_addr`, `output_is_dram`, `num_rows_per_core`, `Wt`, `tile_offset`; compute: `num_rows_per_core`, `Wt`, `origin_w`, `p`, `p_is_negative`). Note `input_is_dram`/`output_is_dram` are dead in the kernels today (see the audit's Misc anomalies) — port them as-is; do not spend the port cleaning them up.

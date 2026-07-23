# Metal 2.0 Port Brief — `data_movement/moe_routing_remap`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `597581e6151 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (single `SingleCore` factory, `create_descriptor` returns a `ProgramDescriptor`).
- **Per-coord dispatch:** `create_descriptor` takes a `std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate` and bakes a per-coordinate scalar (`device_weights_count_offset = mesh_coordinate[axis] * non_zero_per_device`) into each program — one program per coordinate. Preserve this per-coord dispatch when wiring the `MetalV2FactoryConcept`; do not collapse it to a single coordinate-independent program. The offset is a plain scalar (see Construct), not an address.
- **Op-owned tensors:** none — the target concept carries no op-owned tensors.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All `no` on this op.

## Construct — to do

**Tensor bindings** (per binding):

- `routing_weights` (input, CB `c_0`) — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; the reader builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(routing_weights_args, get_arg_val<uint32_t>(0))`. The `Buffer*` runtime-arg entry and the `TensorAccessorArgs<5>` plumbing both disappear.
- `local_weights` (output, CB `c_2`) — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; the writer builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(local_weights_args, get_arg_val<uint32_t>(0))`. Note the writer's raw pointer arithmetic operates on **CB L1 addresses** (`get_read_ptr` results), not on the tensor base — leave that arithmetic unchanged; only the accessor construction changes.

**Non-tensor runtime arg to keep:** `device_weights_count_offset` (reader RTA index 1) is a per-coord expert-count skip counter, not a device address. Port it as a named runtime arg (e.g. `device_weights_count_offset`); it stays baked per coordinate.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — both accessors already pass only `(args, base_addr)`.

**CB endpoints:**
- Self-loop CB `c_2` (`local_weights_dfb`) — touched only by the writer; bind the writer as both PRODUCER and CONSUMER.
- CB `c_0` (`routing_weights_dfb`) and CB `c_1` (`local_weights_idxs_dfb`) — ordinary 1:1 FIFOs (reader PRODUCER → writer CONSUMER); no special action.

Single config (single-core, fixed CB shapes), so these dispositions do not vary.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader.
- **Cross-op / shared kernels:** the op owns both kernels. It calls into the in-family shared header `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` (`tt_memmove`, `fill_with_val`, `ByteSizeAddressType`) — if you touch that header's rewrite, port it as one unit across all its consumers. The unused `#include` of `ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp` in both kernels can be dropped (no symbol from it is used); dropping it is optional and outside the strict port, but it removes the op's only cross-family coupling.
- **RTA varargs:** none — all runtime args are fixed, nameable fields; no arg-indexed loop.

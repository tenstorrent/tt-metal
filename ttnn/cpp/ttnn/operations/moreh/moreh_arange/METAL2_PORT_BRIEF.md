# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_arange`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (`create_descriptor` returns `ProgramDescriptor`; one factory, two config-selected kernel variants — tile vs row-major, chosen by `untilize_out`).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · migration-risky pybind (`Is safe to port? = yes`). All `no` on this op.

## Construct — to do

**Tensor bindings** (per binding):

- `output` — **Case 1** (via `TensorAccessor`) → express as a `TensorParameter` / `TensorBinding`. The kernel builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(dst_args, dst_addr)`. Both the CTA `TensorAccessorArgs(*output.buffer())` (`device/moreh_arange_program_factory.cpp:64`) and the RTA output-address argument (delivered today as the `Buffer*` at `program_factory.cpp:90`, consumed as `dst_addr = get_arg_val<uint32_t>(0)`) disappear. Applies to **both** kernel variants (`writer_moreh_arange.cpp`, `writer_moreh_arange_rm.cpp`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none (both kernels pass a 2-arg `TensorAccessor`).

**CB endpoints:** self-loop `c_16` (`tt::CBIndex::c_16`) — bind the single writer kernel as **both** PRODUCER and CONSUMER. It is a one-toucher scratch CB (`reserve_back` + `get_write_ptr` only; no `push_back`/`pop_front`). Same disposition under both `untilize_out` configs. *(Port-time cleanup, kernel-side whitelist rule 7: the tile kernel's `get_tile_size(cb_out)` at `writer_moreh_arange.cpp:24` moves onto the DFB metadata accessor.)*

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none — both kernels are op-owned, file-path-instantiated; includes are `api/*` (tt_metal HAL/LLK) only.
- **RTA varargs:** none — fixed distinct scalar fields, name each (tile kernel: `dst_addr`, `tile_offset`, `num_tiles`, `start`, `step`; RM kernel adds `element_size`). Note the tile kernel does not read the `element_size` arg the factory pushes — drop it from the tile-path arg set when naming.

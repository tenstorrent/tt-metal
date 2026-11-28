# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/quasar/grid_sample`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.
>
> **Scope note (user instruction):** this port targets the **quasar copy** only. **Do not add or change anything under the main tree** (`pool/`, including `pool/generic` and `pool/device/kernels`). The borrowed compute kernel is handled with a grid_sample-local copy; see *Construct*.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorParameter Case 2 ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `9df4aa1e011 2026-11-27 docs(metal_2.0): fix-ups from an adversarial review of the trial-port fixes` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor`, for both `GridSampleBilinearProgramFactory` and `GridSampleNearestProgramFactory` (`device/grid_sample_device_operation.hpp:23-31`)
- **Op-owned tensors:** none
- **Target concept:** `ProgramSpecFactoryConcept`, for both factories
- **Gate-cleared, confirmed absent:** a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args`. (Also absent, though they wouldn't have blocked: custom hash, `override_runtime_arguments`, pybound `create_descriptor`.)

## Construct — to do

**Tensor bindings** (per binding). Every legacy address is a `Buffer*` runtime arg, so the RTA and its `TensorAccessorArgs` compile-time plumbing both disappear:

- `input` — **Case 1** in every factory and config → `TensorParameter`; kernels build `TensorAccessor(tensor::input)`. Sites: `reader_grid_sample_interleaved_start_id.cpp:19,41,45`; `reader_grid_sample_sharded.cpp:64,89-90`; `writer_grid_sample_nearest_sharded.cpp:168,171,178-179`.
- `grid` — **Case 1** when the grid is interleaved (bilinear reader `reader_grid_sample_interleaved_start_id.cpp:20,42,44`; nearest writers `writer_grid_sample_nearest_sharded.cpp:172,181-182`). **Clean** when the grid is sharded: the CB is backed by the grid buffer (`bilinear:110`, `nearest:91`), so use `DataflowBufferSpec::borrowed_from`.
- `output` — **Case 1** for bilinear with an interleaved grid (writer `writer_grid_sample_interleaved.cpp:11,19,21`). **Clean** otherwise: the CB is backed by the output buffer (`bilinear:215` when sharded; `nearest:134` always) → `borrowed_from`.

**Borrowed compute kernel → local copy (user decision):** copy `pool/generic/device/kernels/compute/compute_pool_2d.cpp` byte for byte into `experimental/quasar/grid_sample/device/kernels/compute/compute_pool_2d.cpp`. Convert the copy (named args, `dfb::` tokens, optional bindings; see *Watch for*) and point the bilinear factory's compute `KernelSpec::source` at it (legacy site `bilinear:353`). Don't edit the main-tree original or add a comment to it, and don't create `compute_pool_2d_metal2.cpp` in `pool/generic`. Only this op binds the local copy, so convert it in place. Keep the kernel's behavior exactly the same: copy first, then convert, with no logic changes.

**TensorParameter relaxation:** `none`

**TensorAccessor 3rd arg:** none

**CB endpoints:**

- **Self-loop:** bilinear grid CB (interleaved; also sharded without split reader) · bilinear output CB (sharded: compute produces, nothing consumes) · nearest `grid0` and `grid1` (interleaved grid: each writer uses its own).
- **Assign 1P+1C** (two copies of one kernel touch the CB, neither calls FIFO functions; the labels have no effect on Gen1): bilinear grid CB (sharded + split reader: reader0/reader1) · nearest `grid0` (sharded grid) · nearest `fill` (all configs) · nearest `output` (all configs).
- **Legal 1:1:** bilinear input0/scalar0 (reader0 → compute), input1/scalar1 (reader1 → compute; sharded + split only), output (interleaved: compute → writer).
- **Conditional DFB specs:** bilinear input1/scalar1 exist only when sharded with split reader (`bilinear:157,185`); nearest `grid1` exists only when the grid is interleaved (`nearest:96`). These conditions already exist in the legacy factory; carry them over.
- No multi-binding flag. No dead CB.

## Watch for

- **CB endpoints (multi-binding):** none. Every co-touched CB has exactly two touchers that never call FIFO functions → 1P+1C, not the flag.
- **Cross-op / shared kernels:** the only shared kernel is `compute_pool_2d.cpp`, handled by the local copy above. **Don't bind `experimental/quasar/pool_generic/device/kernels/compute/compute_pool_2d.cpp`** even though it is already converted: it was reworked for Quasar pool and doesn't fit grid_sample (split-reader path removed; packs to a scratch buffer that needs a reader kernel to drain it; requires equal-width channel blocks; compile-time arg 16 replaced). The donor headers `pool/device/kernels/pool_kernels_common.hpp` and `experimental_device_api.hpp` are include-only and need no changes (token-friendly signatures), so leave them as they are. Other factories still on the main-tree original (information only, not yours to touch): `pool/generic/device/pool_multi_core_program_factory.cpp:939`, `pool/grid_sample/...`, `pool/rotate/...`, `experimental/quasar/rotate/device/rotate_bilinear_program_factory.cpp:294`.
- **Optional bindings in the compute kernel (local copy):** `compute_pool_2d.cpp:104-110` builds `DataflowBuffer` objects for `in_cb_id_1`, `in_scalar_cb_id_1`, `pre_tilize_cb_id`, and `fast_tilize_cb_id`. grid_sample passes `DUMMY_CB_ID` (32) for any of these it doesn't allocate (always the last two, and the `_1` pair unless sharded with split reader). Its Metal 2.0 version needs these to be optional bindings (the "Conditional / optional resource bindings" pattern in `port_patterns.md`). It also reads runtime arg 0 only when compile-time arg 3 is 0 (`:129`); grid_sample never hits that path.
- **Indexed-collection args:** none. Give every argument a name.
- **Lent kernels — joint port with quasar rotate:** rotate's bilinear factory binds `writer_grid_sample_interleaved.cpp` and includes `grid_sample_reader_common.hpp`; it will also bind the local `compute_pool_2d.cpp` copy. Convert them once, for both ops.
- **Two compute KernelSpecs (bilinear, interleaved):** `bilinear:367-375` makes one compute kernel per core group, differing only in compile-time arg 3 (`total_interpolations`). Keep two KernelSpecs. Don't move that value into a runtime arg (the "demoting per-group CTA to RTA" anti-pattern).
- **Nearest writer's grid accessor is unused when the grid is sharded:** `writer_grid_sample_nearest_sharded.cpp:181-182` always builds `grid_tensor_accessor` (base 0). In the sharded config the host fills its args slot with the input's args (`nearest:165-167`), and it is only read under `if constexpr (!is_sharded)` (`:227-231`). Decide how `tensor::grid` binds in the sharded config before rewriting this line, rather than swapping it as-is.
- **`constexpr` metadata lookups keep the token form (rule 7):** `get_dataformat(scalar_cb_index)` in `if constexpr` (`device/kernels/grid_sample_reader_common.hpp:33`, template parameter `scalar_cb_index`), and the donor `zero_out_tiles<cb_id>` (`pool/device/kernels/pool_kernels_common.hpp:73-75`, `constexpr get_tile_size(cb_id)`). Pass the `dfb::` token as the template parameter. Don't switch to member getters or demote to `const`.
- **Kernels already use the newer kernel-side objects:** every dataflow kernel and the compute kernel already use `DataflowBuffer` / `Noc` / `TensorAccessor`. The port is a binding-layer change (positional args → named args, buffer index → `dfb::` token, address arg → `tensor::` token), not a rewrite of how the kernels work.
- **Donor header `pool/device/kernels/experimental_device_api.hpp:11`** includes `api/dataflow/circular_buffer.h` and aliases `CB = CircularBuffer`. Your kernels pick it up through includes but never use it. It's a donor file outside your scope: leave it alone, and don't copy the include into ported kernels.
- **Don't copy other quasar ports as templates:** other `*_metal2.cpp` files under `experimental/quasar/` (e.g. `conv2d/.../conv_unpack_tilize_probe_metal2.cpp`) were quick ports that don't follow the current conventions. Don't take binding names or idioms from them.

# Metal 2.0 Port Report — `data_movement/fold`

## Outcome

**PORTED** — both factories of the `Fold` device operation converted to `MetalV2FactoryConcept`:
- `MultiCore` (height-sharded row-major path)
- `MultiCoreDRAMFold` (interleaved DRAM path; both runtime sub-programs — tiled and row-major)

Verification on `wormhole_b0` (`SAFE_PYTEST_RESULT: PASS` for both; no hangs, no device resets) — **146 passed, 0 failed** total:
- `tests/ttnn/unit_tests/operations/conv/data_movement/test_fold_op.py` — **126 passed, 180 skipped, 0 failed**. The 180 skips are test-defined parameter filters (the test file was not modified, so the skip pattern is identical to pre-port).
- `tests/ttnn/nightly/unit_tests/operations/conv/data_movement/test_fold_op.py` — **20 passed, 0 failed** (`test_fold_sharded` + `test_fold_sharded_tile_layout`).

Passing coverage spans all three program paths (sharded MultiCore, including tile-layout sharded input; DRAM tiled; DRAM row-major) across `bfloat8_b` / `bfloat16`, both L1-aligned and unaligned sticks, and multiple stride/padding/shape combinations.

## Provenance

- **Recipe docs (this port):** `b5b801a923d 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `b5b801a923d 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for both factories, as the audit decided. No deviation.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (the op was already on the default reflection-based hash).
- Pybind entry points removed: **none** (`fold_nanobind.cpp` binds via `ttnn::bind_function<"fold">`; no `create_descriptor` pybind existed).
- Header change: both factory structs' `create_descriptor` → `create_program_artifacts` (return type `ttnn::device_operation::ProgramArtifacts`); `#include <tt-metalium/program_descriptors.hpp>` → `#include "ttnn/metal_v2_artifacts.hpp"` in `fold_device_op.hpp`. The device-op class body (`select_program_factory`, `validate_*`, `compute_output_specs`, `create_output_tensors`, `ttnn::prim::fold`) was left untouched.

### Open items
- None affecting the factory layer. No relaxation candidates (the op has no custom hash and strict tensor matching is correct here).

## Handoff points

- **Cross-op compute-kernel fork (untilize).** The legacy DRAM-tiled sub-program file-path-instantiated `untilize/device/kernels/compute/untilize.cpp`, owned by the **untilize** op (still on the legacy API). Metal-2.0-ifying it in place (named args + DFB bindings) would break untilize's own use, so it was **forked** into `fold/device/kernels/compute/untilize.cpp`. This is a thin wrapper over the shared `compute_kernel_lib::untilize` helper (`ttnn/cpp/ttnn/kernel_lib/`, unchanged). **Sunset:** when the untilize op is ported to Metal 2.0, both it and this fold fork should adopt one shared Metal 2.0 untilize compute kernel and the fold-local copy can be deleted. Owning team: data_movement / untilize.

## Successes

- **Two-toucher DFB → 1P+1C (not multi-binding).** The audit brief and [patterns catalog — Two-toucher DFB → assign 1P+1C] correctly steered the sharded factory's `writer_cb2s_row_major.cpp` dual-instance work-split (one source, Writer- + Reader-config over one grid, both raw-touching the two borrowed CBs) to a plain 1P+1C endpoint assignment. The multi-binding advanced option was never needed. Bindings: `fold_multi_core_program_factory.cpp` (SRC0/DST0, one PRODUCER + one CONSUMER each).
- **`borrowed_from` satisfies the TensorParameter binding requirement.** The sharded input/output are borrowed-memory DFBs with **no** kernel-side `TensorAccessor`. Cross-checking `interleaved_to_sharded_program_factory.cpp` (its sharded-output branch binds only the DFB, not a `TensorBinding`, yet declares the `TensorParameter`) confirmed a borrowed DFB does not need a separate `TensorBinding` — the validator accepts it. Avoided adding spurious tensor bindings the kernels don't use.
- **`CoreLocalMem` preserved the `use<AddrSelector::WRITE_PTR>` source semantics.** See Friction below — the catalog's "the wrapper drops, a bare DFB is pointer-sourced" guidance would have silently changed the source pointer; `CoreLocalMem<uint32_t>(dfb.get_write_ptr())` kept it exact.
- **Conditional-binding pattern fit the `c_1` scratch anomaly cleanly.** The row-major writer's unconditional `get_write_ptr()` on a config-only-allocated CB (flagged in the audit) resolved exactly per [patterns catalog — Conditional / optional DFB bindings]: a `FOLD_RM_NOT_L1_ALIGNED` define gates both the binding and the kernel's `dfb::in1` alias + uses, so the touch matches the allocation in both configs.

## Friction

### Gaps
- **`use<AddrSelector::WRITE_PTR>` has no `DataflowBuffer` equivalent, and the whitelist's "wrapper drops" guidance is unsafe for a WRITE_PTR *source*.** [Kernel-side whitelist rule 1] says the `use<AddrSelector>` / `CircularBufferView` wrappers "drop, because a bare `DataflowBuffer` used as a NoC source/destination is already pointer-sourced," giving a `use<READ_PTR>` example. But `noc_traits_t<DataflowBuffer>::src_addr` returns `get_read_ptr()`, so dropping a `use<WRITE_PTR>(cb)` **source** silently switches the source from the write to the read pointer — a semantic change. Two fold writers (`writer_cb2dram_for_tiled_input.cpp`, `writer_cb2dram_for_rm_input.cpp`) source data from the *write* pointer. Resolved faithfully with `noc.async_write(CoreLocalMem<uint32_t>(cb.get_write_ptr()), dst, size, {.offset_bytes=...}, ...)`, which reproduces `CircularBufferView<WRITE_PTR>::src_addr` exactly. **Suggested doc fix:** the whitelist should note that dropping the wrapper is only equivalent for the pointer the bare-DFB `noc_traits` defaults to (read-ptr as source, write-ptr as destination); a *non-default* selector (WRITE_PTR-as-source or READ_PTR-as-destination) must be reproduced via `CoreLocalMem<uint32_t>(dfb.get_write_ptr()/get_read_ptr())`, not by dropping.

### Confusion
- **The recipe-recommended reference port (accumulation, `akertesz/porting-experiment-accumulation-jun10`) is stale against current headers.** It uses `create_program_spec` (now `create_program_artifacts`), the `ProducerOf`/`ConsumerOf` factories (recipe now prefers full `DFBBinding{...}` designated-initializers), `TensorArgument{std::cref(t)}` (recipe now says pass the `MeshTensor` directly), `ComputeHardwareConfig{.math_fidelity=…, .fp32_dest_acc_en=…, .dst_full_sync_en=…, .math_approx_mode=…, .unpack_to_dest_mode=…}` (the current `ComputeGen1Config` renamed every one of those fields: `fpu_math_fidelity`, `enable_32_bit_dest`, `double_buffer_dest` *(inverted)*, `sfpu_precision_mode`, `unpack_modes`), and the `ta::` tensor-accessor namespace (now `tensor::`). Following it literally would not compile. I followed the recipe + current headers throughout and cross-checked shapes against a *current* MetalV2 op (`interleaved_to_sharded`). **Suggested fix:** either refresh the accumulation branch or point the recipe at a current in-tree MetalV2 op as the shape reference.
- **"A `KernelRunArgs` must be specified for ALL kernels" vs. "may be omitted if no RTAs."** `program_run_args.hpp` says every kernel needs a `KernelRunArgs`; the recipe's Construct step says it "may be omitted entirely" for a no-RTA kernel. I provided empty `KernelRunArgs{.kernel = …}` entries for the no-RTA sharded kernels and the no-RTA tiled compute kernels; tests pass, so empty entries are accepted. Minor — a one-line reconciliation in the recipe would remove the doubt.

## Open items for downstream

- **Cross-op kernel fork:** `fold/device/kernels/compute/untilize.cpp` is a fork of `untilize/device/kernels/compute/untilize.cpp` (path taken: **fork**, `_metal2`-style, but kept the original filename inside fold's own dir since it is not co-resident with the legacy copy). Remaining unmigrated consumer: the **untilize** op (and any other op instantiating untilize's compute kernel). When untilize ports, converge on one shared Metal 2.0 untilize compute kernel and delete this fork.
- **Pre-existing anomaly (unchanged, ops team):** `element_size` is a read-but-unused compile-time arg in `writer_cb2dram_for_tiled_input.cpp` (legacy CTA[7], preserved as the named arg `element_size`). Not touched by the port; flagged for the ops team if they want to drop it.
- **`bfp_pack_precision_mode` / FP32 `unpack_modes`:** the tiled compute sets `unpack_modes = {{SRC0, UnpackToSrc}}` only when the input is `Float32` (`enable_32_bit_dest` true), mirroring the legacy default. The confirmed test set exercises `bfloat8_b`/`bfloat16` only, so the FP32 branch is correct-by-construction but not covered by `test_fold_op.py`. A future FP32 fold test would exercise it.

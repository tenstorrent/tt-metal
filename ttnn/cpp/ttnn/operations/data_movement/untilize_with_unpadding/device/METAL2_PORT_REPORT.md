# Metal 2.0 Port Report — untilize_with_unpadding (single-core factory)

Status: **PORTED** (single-core factory only; not built — this worktree has no build dir).

## TTNN ProgramFactory
- **Concept realized**: `MetalV2FactoryConcept` (`ProgramSpecFactoryConcept`). `UntilizeWithUnpaddingSingleCoreProgramFactory::create_descriptor` → `create_program_spec`, returning `ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`.
- **Device-op-class edits**: none required.
  - `select_program_factory` returns the factory by value; the variant routing in `mesh_device_operation_adapter` dispatches on the concept, so the only change to route to the new path is the factory itself swapping `create_descriptor` → `create_program_spec`. No edit to `untilize_with_unpadding_device_operation.cpp` / `.hpp`.
  - Custom `compute_program_hash`: none (default reflection hash) — nothing to delete.
- **Pybind entry points removed**: none. Grepped the op's pybind/nanobind surface for `create_descriptor` / `create_program` — no exposure of this factory's entry point.
- **Open items (concept fit)**: the device-op's other 5 factories stay on the legacy `ProgramDescriptorFactoryConcept`. The `program_factory_t` variant is mixed-concept and the framework dispatches per-factory, so the op keeps building/running. Those are the remaining work for a future pass.

## Handoff points
- **No worked examples on this branch.** PORT_INSTRUCTIONS.md described worked `rand` / `matmul` Metal 2.0 ports on branch `dgomez/rand-metal2`, but this worktree is on `worktree-agent-...` off a commit where NO in-tree factory uses `create_program_spec` / `device_operation::ProgramArtifacts` (grep returned zero). The Metal 2.0 *framework* headers (`metal2_host_api/*`, `ttnn/metal2_artifacts.hpp`, adapter `ProgramSpecMeshWorkloadFactoryAdapter`, device-side `dataflow_buffer.h` / `tensor_accessor.h` / `kernel_args.h`) ARE all present and complete. The port was written directly against those headers + the recipe/patterns docs. Flagging the missing exemplars so the invoker can confirm the branch is the intended one before building.

## Successes
- **Caution: Modifying a shared dataflow kernel** (patterns catalog) fired correctly. The reader (`reader_unary_interleaved_start_id.cpp`, ~12 consumers) and compute (`untilize/.../compute/untilize.cpp`, ~9 consumers) are heavily shared; both were FORKED to `_m2` copies inside this op's `device/kernels/` tree rather than edited in place, so no sibling op breaks. The writer (`writer_unary_unpad_dims_split_rows.cpp`, this op's dir, single consumer) was ported in place.
- **Pass DFB handles directly** pattern: the compute kernel feeds `dfb::in`/`dfb::out` straight into `compute_kernel_hw_startup(...)` and `compute_kernel_lib::untilize<per_core_block_tile_cnt, dfb::in, dfb::out, ...>(...)` (template-parameter position), relying on the constexpr `DFBAccessor::operator uint32_t()`. No `.id` extraction.

## Friction
- **Gap — fork location for cross-op kernels.** The patterns catalog says fork "alongside the original" (`*_metal2.cpp` next to the legacy file), but the recipe's hard scope fence is "stay within `ttnn/cpp/ttnn/operations/<op>/`". For cross-op shared kernels these conflict (the original lives in another op's dir). Resolved by placing the forks inside THIS op's `device/kernels/` tree with an `_m2` suffix and pointing the factory at them. The docs should state explicitly where a cross-op fork lands when the porter may not write outside the op dir.
- **Confusion — writer CTA slot 1.** The legacy host emitted `unpadded_stick_size` as writer CTA slot 1, but the kernel reads only CTA slot 0 (`FLOAT32_DTYPE`) and `TensorAccessorArgs<2>()`. The slot-1 value is dead. Dropping it (no named CTA) is behavior-preserving; documented in the plan's Dropped Plumbing.

## Open items for downstream
- **Fake-CB self-loops**: none. Both DFBs (IN, OUT) have real producer/consumer pairs (reader→compute, compute→writer).
- **Compute-kernel pointer escape valve**: none. The compute kernel needs no raw L1 base address; the writer's only base-pointer read is `cb_out0.get_read_ptr()` on its own DFB (`CoreLocalMem` over the DFB's L1) — framework-managed, refreshed per execution, no smuggled pointer.
- **Cross-op kernel forks (sunset checklist):**
  - `reader_unary_interleaved_start_id.cpp` — FORKED to `untilize_with_unpadding/device/kernels/dataflow/reader_unary_interleaved_start_id_m2.cpp`. Legacy original (`eltwise/unary/device/kernels/dataflow/`) unchanged; ~11 other consumer op dirs remain unmigrated. Delete the fork only when this op no longer needs it AND the shared original is itself Metal-2.0-ported.
  - `untilize/device/kernels/compute/untilize.cpp` — FORKED to `untilize_with_unpadding/device/kernels/compute/untilize_m2.cpp`. Legacy original unchanged; ~8 other consumer op dirs remain unmigrated.
- **Sibling factories (carry-over):** the 5 remaining factories in this device-op are candidates for the same treatment in a follow-up pass; several share these same reader/compute kernels, so the forks created here can be reused.
- **Test coverage note:** not built or run (no build dir per instructions). Verification (gtests + pytests at `tests/ttnn/unit_tests/operations/data_movement/` and under the simulator) is required before merge.

---

# Metal 2.0 Port Report — untilize_with_unpadding (multi-core interleaved factory)

Status: **PORTED** (multi-core interleaved factory — the default-selected non-sharded multicore
path; not built — this worktree has no build dir).

## TTNN ProgramFactory
- **Concept realized**: `MetalV2FactoryConcept`. `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::create_descriptor` → `create_program_spec`, returning `ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`. Header updated (`tt-metalium/program_descriptors.hpp` include → `ttnn/metal2_artifacts.hpp`; return type `ProgramDescriptor` → `ProgramArtifacts`).
- **Device-op-class edits**: none. `select_program_factory` returns this factory by value (unchanged); the adapter dispatches per-factory on the concept. No custom `compute_program_hash` to delete.
- **Pybind entry points removed**: none — grepped `untilize_with_unpadding_nanobind.{cpp,hpp}`; no `create_descriptor`/`create_program` exposure for any factory.

## Kernels (multi-core)
- **reader** — REUSES `reader_unary_interleaved_start_id_m2.cpp` (the cross-op `_m2` fork already created for single-core). The multicore reader emits `{src, num_tiles_per_core, tile_start_id}` → the m2 reader's `num_pages`/`start_id` named RTAs + `ta::src`. No new fork.
- **compute** — REUSES `untilize_m2.cpp` (the cross-op `_m2` fork already created for single-core). Multicore CTAs `{nblocks_per_core, num_tiles_per_row}` map onto the m2 compute's `per_core_block_cnt`/`per_core_block_tile_cnt`; cb ids become `dfb::in`/`dfb::out`. No new fork.
- **writer** — `writer_unary_stick_layout_split_rows_multicore.cpp` ported **in place** (op-local, single consumer = this factory). dst addr → `ta::dst`; cb 16 → `dfb::out`; FLOAT32_DTYPE/unpadded_X_size → named CTAs; padded_X_size/start_stick_id/n_block_reps → named RTAs; per-core block-rep 5-tuples → runtime varargs (`get_vararg`).

## Successes
- **Multi-group work split → multi-KernelSpec / multi-WorkUnitSpec** (matmul-multicore exemplar shape) applied cleanly: full + cliff compute are two KernelSpecs in two WUs (`uwu_full`, `uwu_cliff`); reader + writer are members of BOTH WUs (so their derived node set = all_cores). The per-group block count stays a CTA — no CTA→RTA demotion.
- **Cross-op fork reuse**: the single-core port's `reader_*_m2.cpp` and `untilize_m2.cpp` forks were directly reusable by the multicore factory (same kernel sources, same named bindings). No additional cross-op kernels touched.
- **Varargs** modeled the writer's variable-length per-core block-rep tuples faithfully (the kernel already bounds its read with the `n_block_reps` named RTA), reusing the `get_vararg` mechanism proven by slice's m2 reader.

## Friction / Open items for downstream
- **BLOCKER-adjacent — deprecated API is the only fit.** The writer needs a DIFFERENT number of runtime varargs per core. The non-deprecated scalar `KernelAdvancedOptions::num_runtime_varargs` only supports a UNIFORM count across all of a kernel's nodes; the per-node-varying case requires `num_runtime_varargs_per_node`, which is marked `[[deprecated]]` ("will be removed once existing uses are refactored"). This port is (per grep) the FIRST in-tree use of that field. It compiles clean because the repo sets `-Wno-deprecated-declarations`. Surfacing precisely so the API owners can decide: either (a) keep a supported per-node-varying-vararg mechanism, or (b) provide guidance to pad every core to the max count via the scalar (inert padding — the kernel reads only `n_block_reps*5` varargs). I chose the deprecated per-node field over padding because padding adds dispatch-buffer bloat and obscures the real per-core count. File: `untilize_with_unpadding_multi_core_interleaved_program_factory.cpp` (the `writer.advanced_options.num_runtime_varargs_per_node[...]` line).
- **Vararg use (report-required)**: writer runtime varargs retained (genuinely variable-length, loop-indexed) — this is the sanctioned vararg case, not a positional-RTA carry-over.
- **Cross-op kernel forks (sunset checklist)**: unchanged from single-core — `reader_unary_interleaved_start_id_m2.cpp` and `untilize_m2.cpp` remain forks; now used by BOTH the single-core and multi-core m2 factories. Delete only when this op no longer needs them AND the shared originals are themselves m2-ported.
- **Compute-kernel pointer escape valve**: none. Writer's only base-pointer read is `cb_out0.get_read_ptr()` on its own DFB (framework-managed). No smuggled addresses.
- **Remaining factories**: 4 of 6 (sharded / col-interleaved / block-interleaved / nd-sharded) stay on the legacy concept; the op keeps building/running (mixed-concept variant, per-factory dispatch).
- **Test coverage**: not built or run (no build dir). gtests + pytests + simulator verification required before merge.

# Port Report — reshard (NdReshardCopyPagesFactory)

## TTNN ProgramFactory

- **Concept realized**: `MetalV2FactoryConcept` (`ProgramSpecFactoryConcept`). The factory now exposes
  `create_program_spec(const ReshardParams&, const ReshardInputs&, Tensor&) -> ttnn::device_operation::ProgramArtifacts`,
  replacing the legacy `create_descriptor` (`ProgramDescriptor`).
- **Custom `compute_program_hash` deletion**: none — `ReshardDeviceOperation` already uses the default reflection-based hash.
- **Pybind entry points removed**: none — `reshard_nanobind.cpp` exposes only the user-facing op, no `create_descriptor`/`create_program_descriptor` hook.
- **Device-op-class edits**: none. `select_program_factory` already returns `NdReshardCopyPagesFactory{}`; the
  `program_factory_t` variant entry is unchanged. The framework's `ProgramSpecMeshWorkloadFactoryAdapter` routes the
  factory automatically once it satisfies `ProgramSpecFactoryConcept` (no longer `ProgramDescriptorFactoryConcept`).
- **Scope**: only the `NdReshardCopyPagesFactory` variant was ported. The other 7 variants in `program_factory_t`
  (SameWidth<t/f>, SameHeight<t/f>, Generic, CopyLocalShard<t/f>) remain on the legacy concept; the op continues to
  build and dispatch per-factory.

## Handoff points

none. No `sem::`/`ta::` out-of-op call site, no kernel-lib gap, no removed pybind surface, no framework gap bit during the port.

## Successes

- **Local-DFB rule (recipe §"Local-DFB rule" / matmul exemplar)** steered the WorkUnitSpec design directly: the reader
  (PRODUCER of `cb_in0`) and writer (CONSUMER) share one `WorkUnitSpec` over the full grid, satisfying the producer+consumer
  same-node invariant. No self-loop needed — this is a real reader→writer FIFO, not a fake CB.
- **Kernel-side whitelist rule 3** collapsed the legacy 6-line `TensorAccessorArgs<0,0>()` + offset-chaining + base-addr-RTA
  dance in each kernel to a single `TensorAccessor(ta::input)` / `TensorAccessor(ta::output)` line. The legacy CRTA carrying
  the buffer base address (`emplace_common_runtime_args({input_buffer})`) dropped entirely, replaced by the `TensorBinding`.

## Friction

- **Gap (minor)**: the worked-example kernels named in `/tmp/PORT_INSTRUCTIONS.md`
  (`writer_bmm_8bank_interleaved_start_id_m2.cpp`, `bmm_m2.cpp`, `reader_bmm_8bank_output_tiles_partitioned.cpp`) do not
  exist on this branch, and there are zero existing `TensorAccessor(ta::...)` / `dfb::` usages anywhere in the tree — no op
  has actually landed a `create_program_spec` port yet. The `matmul`/`rand` "exemplars" are still `create_descriptor`
  (ProgramDescriptor) factories. The port shape was reconstructed from the framework headers
  (`program_spec.hpp`, `kernel_spec.hpp`, `program_run_args.hpp`, `dataflow_buffer.h`, `kernel_args.h`) and the recipe — so
  the kernel-side `ta::`/`dfb::`/`args::` generated-token usage is **unvalidated by build** (the worktree has no build dir,
  per instructions). First op to actually compile this concept should treat the generated-header behavior as shakedown.

## Open items for downstream

- **Remaining reshard factories (7)**: SameWidth/SameHeight/Generic/CopyLocalShard. Several read **L1 shard-spec geometry**
  (`shard_spec().grid()`, block/width-sharded core-count math, `std::gcd` page sizing — see
  `nd_reshard_program_factory_copy_local.cpp:50-80`) absent from this simplest DRAM→DRAM page-copy variant. Those will exercise
  shard-spec handling and possibly the templated `local_is_input/local_is_output` variants as multi-variant factories.
- **Cross-op kernel touches**: none. Both ported kernels
  (`nd_reshard_copy_pages_{reader,writer}.cpp`) live in this op's dir and are used only by this factory
  (`grep -rl` confirms), so they were ported **in place**, not forked.
- **Fake-CB self-loops / compute-pointer escape valves**: none used.
- **Build/test not run** per instructions (no build dir; do-not-build). Verification is limited to a static
  anti-pattern sweep (no `create_descriptor`/`CircularBuffer`/`CBDescriptor`/`TensorAccessorArgs`/`buffer()->address()`/
  positional `get_*_arg_val` survive in the four touched files). A real build + the reshard pytests/gtests are required
  before merge.

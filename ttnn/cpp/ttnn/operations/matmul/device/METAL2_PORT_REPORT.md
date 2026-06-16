# Metal 2.0 Port Report — MatmulMultiCoreReuseOptimizedProgramFactory

**Status:** PORTED
**Factory:** `MatmulMultiCoreReuseOptimizedProgramFactory` (`create_descriptor` → `create_program_spec`)
**Concept:** `MetalV2FactoryConcept` (returns `ttnn::device_operation::ProgramArtifacts`)

## Kernels

| Kernel | Disposition | Reason |
|---|---|---|
| `dataflow/reader_bmm_tile_layout_in0.cpp` | Ported in place | Exclusive to this factory (grep confirmed). |
| `dataflow/reader_writer_bmm_tile_layout_in1.cpp` | Ported in place | Exclusive to this factory (grep confirmed). |
| `compute/bmm_large_block_zm_fused_bias_activation.cpp` | **Forked** → `..._m2.cpp` | Shared by 6 matmul factories (mcast_1d, mcast_2d, two DRAM-sharded, sparse, this one) + a deepseek FFN kernel. |

## Structural decisions

- **Borrowed-memory DFBs from io tensors (the flagged risk): EXPRESSIBLE.**
  The legacy sharded CBs set `cb_desc.tensor = &in0_buffer / &in1_buffer / &output`.
  Modeled as `DataflowBufferSpec::borrowed_from = <io tensor's TensorParameter>`, gated on
  `in0/in1/output_is_sharded`. The framework explicitly supports a `TensorParameter` referenced
  *only* via `borrowed_from` with no kernel `TensorBinding` (program_spec.cpp:490-508 registers it
  as "used (no kernel user)"). The L1_SMALL blocker does NOT apply: matmul sharded io tensors are
  `BufferType::L1`, which is what the borrowed-DFB validator requires (program_spec.cpp:1264-1269).
  On each sharded path the corresponding kernel-side `TensorAccessor` construction already lives
  inside the existing `#ifndef IN*_SHARDED` block, so the factory binds the tensor to the kernel
  only on the non-sharded (NoC) path; on the sharded path only the borrow is used.
- **Aliased out/intermed0 (shared CB).** When `interm0_data_format == output_data_format` and not
  (untilize_out && in1_num_subblocks>1), legacy uses ONE CBDescriptor with two format_descriptors
  (c_4 + c_5). Ported as two DFBs (`out`, `intermed0`) with mutual `advanced_options.alias_with`.
  Alias legality holds: same total size (equal tile size when formats match), same node coverage
  (alias rule 3 is node-set, not kernel-identity — confirmed program_spec.cpp:1351-1360, so it is
  fine that `out` is bound to compute+reader_writer while `intermed0` is bound to compute only), and
  consistent `borrowed_from` (both borrow `out` when output is sharded; neither otherwise).
- **Work-split multiplicity preserved.** Two compute `KernelSpec`s of the forked source (one per
  core group, differing only in the per-group output-block count CTA, kernel-side name `batch`),
  each in its own `WorkUnitSpec`. reader + reader_writer are shared across both WUs; per the
  Local-DFB rule each WU contains reader + reader_writer + its compute so every DFB endpoint pair
  shares the WU. No CTA was demoted to RTA.
- **Conditional DFBs/bindings (all known at factory-construction time):**
  - `in0_transposed` (c_10): bound + a self-loop on compute only when `in0_transpose_tile`. The
    compute kernel references `cb_in0_transposed` from a file-scope ternary, so the kernel-side
    handle is `#ifdef IN0_TRANSPOSE_TILE_CB`-gated (Conditional-DFB pattern) and the factory emits
    that define alongside the binding.
  - `in0_intermediate` (c_8) / `in1_intermediate` (c_9): Blackhole `INTERMEDIATE_CB_READ` workaround;
    bound as real self-loops (reserve/push then wait/pop) on their reading kernel, gated on the
    arch + odd-tile-size condition.
- **Dropped plumbing:** `TensorAccessorArgs(...).append_to(cta)` + kernel `TensorAccessorArgs<N>()`
  + buffer-address RTAs → `TensorBinding` / `ta::`. Named-CB CTAs
  (`get_named_compile_time_arg_val("cb_*")`) → `dfb::`. All positional CTAs/RTAs → named
  `get_arg(args::*)`. Writer's `use<CircularBuffer::AddrSelector::READ_PTR>(cb_out)` → bare `cb_out`.

## Device-op-class / pybind edits (the two sanctioned exceptions)

- **Custom `compute_program_hash` KEPT** (matmul_device_operation.cpp:2107) — per instruction it is
  shared across the matmul factory variant and must not be deleted by a single-factory port.
- **Pybind hook removed** (matmul_nanobind.cpp): deleted the
  `MatmulMultiCoreReuseOptimizedProgramFactory` `create_descriptor` + `default_core_range`
  `nb::class_`/`def_static` block (the legacy `create_descriptor` symbol no longer exists).
  *User-visible surface change:* any Python caller of
  `MatmulMultiCoreReuseOptimizedProgramFactory.create_descriptor` / `.default_core_range` breaks.

## Naming note (kernel-side ↔ legacy CTA slot)

The reader/reader_writer kernels' RTA slot 2 (kernel-side `batch`) carries
`num_output_blocks_per_core`; the compute kernel's CTA slot 13 (kernel-side `batch`) also carries
the per-group output-block count. Both faithfully reproduce the legacy positional values (the legacy
kernels' loop variable is literally named `batch` although it counts output blocks); names were kept
matching the kernel variables, not re-derived to a "truer" meaning, to keep the port mechanical.

## Open items / friction

- **Untested at runtime in this worktree** (no build dir). Borrowed-DFB-from-io-tensor + aliased
  borrowed DFBs are an advanced, lightly-exercised path; the sharded + shared-out/intermed0 +
  borrowed combination is the first matmul use and should be validated on hardware (interleaved and
  each sharded layout, transpose_a/b on/off, the two-core-group split, Blackhole odd-tile path).
- **Not a blocker, flagged for owner:** the legacy factory's Blackhole `INTERMEDIATE_CB_READ`
  helper CBs are genuine scratch FIFOs; modeled as self-loops here (correct), but they are
  candidates for the forthcoming Metal 2.0 kernel-scratchpad resource.

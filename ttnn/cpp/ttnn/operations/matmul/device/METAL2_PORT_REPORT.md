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

---

# Metal 2.0 Port Report — MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory

**Status:** PORTED
**Factory:** `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` (`create_descriptor` → `create_program_spec`)
**Concept:** `MetalV2FactoryConcept` (returns `ttnn::device_operation::ProgramArtifacts`)

## Why this is NOT metal-blocked (correcting a prior pass)

A prior pass marked this factory METAL-BLOCKED for "raw buffer-address-through-RTA on io tensors
(remote storage cores / DRAM bank)". That was wrong. The factory has **zero semaphores** and all of
its CBs are **same-node local FIFOs** (no mcast / no `RemoteDataflowBufferSpec`), so its DFB topology
ports like the MatmulMultiCore / MatmulMultiCoreReuseOptimized exemplars. The raw addresses live in
**data-movement (RISCV) kernels**, so they are **Case 2 — portable** via
`TensorAccessor::get_bank_base_address()`: bind the tensor through the typed channel and pull the base
kernel-side, keeping the explicit NoC / DRAM-bank arithmetic unchanged. No compute (TRISC) kernel needs
a raw base, so no Case-2-in-compute block applies.

## Kernels

| Kernel | Disposition | Reason |
|---|---|---|
| `dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp` | Ported in place | Exclusive to this factory (grep confirmed). |
| `dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp` | Ported in place | Exclusive to this factory (grep confirmed). |
| `compute/bmm_large_block_zm_fused_bias_activation.cpp` | **Reused existing fork** `..._m2.cpp` | Shared by 6 matmul factories + a deepseek FFN kernel; the Metal 2.0 fork already created for MatmulMultiCoreReuseOptimized covers this factory's CTA layout (same positional order) + named CTAs (`bias_ntiles`, `last_subblock_w_valid`, `activation_*`). |

## Structural decisions

- **Case-2 base-address bridge (the flagged "blocker" — actually portable).**
  - in0 reader: `input_shard_l1_addr` (legacy RTA fed by the in0 MeshTensor) →
    `TensorAccessor(ta::a).get_bank_base_address()` (L1-sharded shard base on the remote storage core).
  - in1 writer: `in1_tensor_addr` → `ta::b` DRAM-bank base; `in3_tensor_addr` (bias, FUSE_BIAS) →
    `ta::bias` DRAM-bank base; `output_shard_l1_addr` → `ta::out` L1 shard base on the remote output
    storage core. The explicit `UnicastEndpoint` NoC walks and `AllocatorBank<DRAM>` reads are
    UNCHANGED — only the base now comes from the accessor.
- **Local DFBs only.** c_0 in0, c_1 in1, c_3 bias (conditional), c_4 out, c_5 intermed0 are same-node
  local FIFOs. When `interm0_data_format == output_data_format` the legacy uses ONE CBDescriptor with
  two format_descriptors (c_4 + c_5); ported as two DFBs with mutual `advanced_options.alias_with`
  (alias-size equality holds: equal tile size when formats match, and `out_shard_tiles == per_core_M *
  per_core_N` for this batch-sharded layout). Otherwise two independent DFBs (legacy distinct sizes).
- **Borrowed CBs c_2 / c_6 DROP.** Legacy c_2 (sharded in0 on input storage cores, `tensor=&in0`) and
  c_6 (output reshard on output storage cores, `tensor=&out`) are bound by NO kernel — they were the
  pre-Metal-2.0 way to reserve the io tensors' L1 on the storage cores. In Metal 2.0 the io tensors'
  own buffer reservations cover that, and the kernels read/write those regions by raw NoC using the
  Case-2 base addresses. So c_2/c_6 are subsumed by the `a` / `out` TensorParameters — no DFB needed.
- **Single WorkUnitSpec.** No work-split CTA variation (per-core CTAs identical; only per-core RTAs
  differ: bank_id, vc, storage-core NoC coords). One WorkUnitSpec hosts in0_reader + in1_writer +
  compute on `all_cores_in_rect_grid` (workers + idle storage/gap cores). Idle cores supply
  `worker_core_type`/`is_worker_core == 0` plus dummy values for the remaining named RTAs and return
  early (named-RTA schema requires every name on every target node; the early return makes the dummies
  harmless — behavior-preserving).
- **Explicit Gen1 DM config.** Legacy pins in0 reader to RISCV_1 + preferred_noc_for_dram_write and
  in1 writer to RISCV_0 + preferred_noc_for_dram_read; reproduced via
  `DataMovementHardwareConfig::Gen1Config` (role = UNSPECIFIED), not the READER/WRITER role hint.
- **Dropped plumbing:** buffer-address RTAs → `TensorBinding` + `get_bank_base_address`. Named-CB
  CTAs (`get_named_compile_time_arg_val("cb_*")`) → `dfb::`. All positional CT/RT args → named
  `get_arg(args::*)`. Writer's `use<CircularBuffer::AddrSelector::READ_PTR>(cb_out)` → bare `cb_out`.

## Shared-fork edit (gated, does not affect other factories)

The shared compute fork `bmm_large_block_zm_fused_bias_activation_m2.cpp` had one surviving positional
RTA read `get_arg_val<uint32_t>(0)` under `#ifdef MATMUL_DRAM_SHARDED` (a worker-core gate). Converted
to `get_arg(args::is_worker_core)` per the kernel-side whitelist. This line is preprocessed OUT for the
reuse-optimized factory (it does not define `MATMUL_DRAM_SHARDED`), so that prior port is unaffected;
only this factory (which defines `MATMUL_DRAM_SHARDED`) compiles and supplies the `is_worker_core` RTA.

## Device-op-class / pybind edits

- **Custom `compute_program_hash` KEPT** (shared across the matmul factory variant — must not be
  deleted by a single-factory port).
- **Pybind hook removed** (matmul_nanobind.cpp): deleted the
  `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` `create_descriptor` `def_static`; kept the
  bare `nb::class_` registration (the ttnn Python package imports the symbol).
  *User-visible surface change:* any Python caller of
  `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory.create_descriptor` breaks.

## Open items / friction

- **Untested at runtime in this worktree** (no build dir; not built per instructions). Should be
  validated on hardware: FUSE_BIAS on/off, fused activation (RELU vs SFPU) on/off, the aliased vs
  separate out/intermed0 paths (packer_l1_acc / fp32 combinations), and the idle-core dummy-RTA path.
- **Friction:** the named-RTA schema's "every name on every node" rule forces idle (non-worker)
  cores to carry dummy values for args they never read (the kernel returns early). Faithful and safe,
  but slightly noisier than the legacy variable-length positional RTA lists. A future "optional per
  node" RTA, or routing the worker/idle flag through a per-WorkUnitSpec split, would be cleaner.

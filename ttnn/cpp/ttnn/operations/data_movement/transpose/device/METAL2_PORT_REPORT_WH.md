# Metal 2.0 Port Report — transpose (WH interleaved factory)

## Summary

- **Status:** PORTED (one factory of eight; second after CN).
- **Factory chosen:** `TransposeWHProgramFactory` — the WH (height/width) interleaved factory, the
  common/default WH path. The two *sharded* WH factories (`TransposeWHShardedProgramFactory`,
  `TransposeWHShardedRMProgramFactory`) were NOT ported this pass.
- **Concept:** legacy `ProgramDescriptorFactoryConcept` (`create_descriptor` → `ProgramDescriptor`)
  → Metal 2.0 `ProgramSpecFactoryConcept` (`create_program_spec` → `ProgramArtifacts`).

## Scope note — one factory, two layout sub-paths

`TransposeWHProgramFactory` is a *single* factory / single program that selects its kernel **sources**
at runtime on `row_major` (input layout TILE vs ROW_MAJOR). Per the recipe's atomic-unit rule (a
factory and ALL the kernel entry points it can bind convert together), both sub-paths were ported in
one pass:

| role | TILE (tiled) source | ROW_MAJOR (RM) source |
|---|---|---|
| reader | `reader_unary_transpose_wh_interleaved_start_id.cpp` (in place) | `reader_unary_transpose_wh_interleaved_start_id_rm.cpp` (in place) |
| writer | `writer_unary_interleaved_start_id_m2.cpp` (FORK) | `writer_unary_transpose_wh_interleaved_start_id_rm.cpp` (in place) |
| compute | `transpose_wh_m2.cpp` (FORK) | `transpose_wh_rm_m2.cpp` (FORK) |

## Kernels — ported in place vs forked

`grep -rl <kernel> ttnn/cpp/ttnn/operations` results drove the fork decision:

- **In place (op-local, only this factory uses them):**
  - `reader_unary_transpose_wh_interleaved_start_id.cpp`
  - `reader_unary_transpose_wh_interleaved_start_id_rm.cpp`
  - `writer_unary_transpose_wh_interleaved_start_id_rm.cpp`
- **Forked to `*_m2.cpp` (shared with unmigrated ops):**
  - `writer_unary_interleaved_start_id.cpp` (lives in `eltwise/unary/`, used by ~30 ops) →
    `writer_unary_interleaved_start_id_m2.cpp` placed in this op's `kernels/dataflow/`.
  - `transpose_wh.cpp` (compute; shared with nlp_create_qkv_heads*, split_qkv, permute_tiled) →
    `transpose_wh_m2.cpp`.
  - `transpose_wh_rm.cpp` (compute; shared with the unmigrated `transpose_wh_sharded_rm` factory) →
    `transpose_wh_rm_m2.cpp`.

  All forks must stay in sync with their legacy originals until the last unmigrated consumer ports.

## Spec shape (ported)

- **DataflowBufferSpecs:** `src0` (legacy c_0; entry=src0 tile size, num_entries = `2` tiled / `wt*2`
  RM), `out` (legacy c_16; num_entries = `2` tiled / `ht*2` RM). RM adds `tilize` (legacy c_24;
  num_entries = `ht*wt`).
- **KernelSpecs:** `reader` (PRODUCER src0; tensor `input`), `writer` (CONSUMER out; tensor
  `output`), `compute`. RM compute binds src0 (CONSUMER), `tilize` (self-loop PRODUCER+CONSUMER), out
  (PRODUCER); tiled compute binds src0 (CONSUMER) + out (PRODUCER).
- **TensorParameters:** `input`, `output`.
- **WorkUnitSpec:** one, `{reader, writer, compute}` on `total_cores` (full grid) — all three share
  the src0/tilize/out DFBs so they must share one WorkUnitSpec. No-op cores get count = 0.
- **ComputeHardwareConfig:** `fp32_dest_acc_en` = Float32/Int32/UInt32 (verbatim from legacy);
  `unpack_to_dest_mode` = UnpackToDestFp32 for `src0` (+ `tilize` on RM) when src dtype is Float32.
- **Compute defines:** `DST_ACCUM_MODE=1` emitted to the RM compute kernel when input dtype is
  UINT32/INT32 (verbatim from legacy `compute_defines`).

## Dropped plumbing

- **Buffer-address RTAs:** reader `input_tensor.buffer()` (legacy reader RTA slot 0) and writer
  `output_tensor.buffer()` (writer RTA slot 0) → replaced by `TensorBinding`s (`input`, `output`).
  Kernel-side `TensorAccessor(args, addr)` → `TensorAccessor(ta::input)` / `(ta::output)`.
- **`TensorAccessorArgs` plumbing:** host `TensorAccessorArgs(buffer, RuntimeTensorShape).append_to(
  cta, common_rta)` and kernel `TensorAccessorArgs<N>()` removed end-to-end.
- **Magic CB indices in CTAs:** tiled writer CTA slot 0 (`output_cb_index`) and RM writer CTA slot 0
  (`cb_out0`) → `dfb::out` binding. Compute kernels' `tt::CBIndex::c_0/c_16/c_24` raw constants →
  `dfb::src0` / `dfb::out` / `dfb::tilize`.
- **Positional CTAs/RTAs → named:** all reader/writer/compute CTAs and RTAs renamed (see kernels).

## Faithful-mapping note (compute count → per-core RTA preserved)

The compute kernel's single per-core count (`NHtWt` tiled / `num_hw_blocks_per_core` RM) varies per
core (work split), so it stays a per-core RTA (NOT a CRTA). The tiled reader's `Ht/Wt/HtWt` are
uniform but the legacy emitted them as per-core RTAs; the port preserves them as per-core RTAs (no
CRTA promotion) to stay maximally behavior-preserving for this multi-path factory. (CN promoted its
uniform args to CRTA; here they were left as RTAs because the tiled reader interleaves them with the
genuinely per-core `start_id`/`start_ht`/`start_wt` in one slot block.)

## Kernel edits (whitelist only)

All six kernels were already on Device 2.0 APIs (`Noc`, `CircularBuffer`, `TensorAccessor`). Edits:
1. Added `#include "experimental/kernel_args.h"`.
2. `CircularBuffer cb(c_NN)` → `DataflowBuffer cb(dfb::name)`; the RM compute helper templates'
   `CircularBuffer&` param → `DataflowBuffer&`.
3. `TensorAccessor(args, addr)` → `TensorAccessor(ta::input)` / `(ta::output)`; removed
   `TensorAccessorArgs<N>()` and buffer-address `get_arg_val` reads.
4. All positional `get_compile_time_arg_val` / `get_arg_val` → `get_arg(args::<name>)`.
5. `dfb::name` passed directly to LLKs (`transpose_wh_init`, `transpose_wh_tile`, `pack_tile`,
   `unary_op_init_common`, `get_local_cb_interface`) and to kernel-lib `compute_kernel_lib::tilize<>`
   (template-arg position) via the implicit `DFBAccessor → uint32_t` conversion. No `.id` extraction.
6. The RM compute kernel's `#ifdef SHARDED` blocks are preserved verbatim and stay #ifdef-gated; the
   SHARDED branch's `tt::CBIndex::c_24/c_25/c_27` references are never name-looked-up in this build
   (SHARDED is never defined by this factory). The non-SHARDED CB indices became DFB handles.
   Comments preserved verbatim.

## Open items / friction for downstream

1. **Legacy CB c_25 ("im2", `TODO REMOVE`) is dropped.** The RM path's legacy factory allocated a
   second intermediate CB at index 25 (`num_im2_tiles = ht`), but **no kernel on the interleaved RM
   path reads or writes it** (the non-SHARDED `transpose_wh_rm` compute uses only c_0/c_24/c_16). A
   Metal 2.0 DFB requires >=1 producer and >=1 consumer binding, so a never-touched buffer cannot be
   expressed. It was omitted. Net effect: slightly less L1 reserved on the RM path; numerically
   inert. (It is already flagged `// TODO REMOVE` in the legacy source.) **Confirm on hardware that
   c_25 is genuinely dead on the interleaved RM path** (it IS used on the sharded RM path, which is a
   separate, unported factory).
2. **`RuntimeTensorShape` accessor config → program-cache key (same as CN).** The legacy factory
   built both TensorAccessors with `tensor_accessor::ArgConfig::RuntimeTensorShape` (shape supplied
   as a runtime arg so one cached program serves varying shapes). The Metal 2.0 `TensorParameter`
   binding bakes the shape into the accessor's compile-time args by default → keys the program cache
   on shape and rebuilds per new shape. For interleaved tensors this is numerically inert but may
   regress cache-hit rate vs the legacy shape-agnostic program. If that matters, set
   `dynamic_tensor_shape = true` on the `input`/`output` TensorParameters. Left at default to keep
   the port minimal/behavior-preserving for the common case.
3. **No-op cores in the WorkUnitSpec (same as CN).** All three kernels launch on the full grid;
   no-op cores receive count = 0. Preserved by targeting the WorkUnitSpec at `total_cores` and
   emitting RTAs for every core. A WorkUnitSpec scoped to `all_cores` would be a behavior-equivalent
   simplification — left as-is to preserve exact legacy placement.

## Device-op-class edits

None. `TransposeDeviceOperation` has no custom `compute_program_hash` (nothing to delete); the
`program_factory_t` variant already lists `TransposeWHProgramFactory` and `select_program_factory`
already routes WH to it. No pybind `create_descriptor` hook for this factory existed.

## Build / test

Not built, not run (worktree has no build dir, per instructions).

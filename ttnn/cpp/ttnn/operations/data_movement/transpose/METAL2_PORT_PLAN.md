# Port Plan — transpose (data_movement)

Port plan for `ttnn/cpp/ttnn/operations/data_movement/transpose`, from the TTNN
`descriptor` (`ProgramDescriptorFactoryConcept`) API to Metal 2.0
(`MetalV2FactoryConcept`).

**Scope:** the clean 6-factory subset from `METAL2_PORT_BRIEF.md`:
`TransposeCNProgramFactory`, `TransposeHCRMProgramFactory`,
`TransposeHCTiledInterleavedProgramFactory`, `TransposeHCTiledProgramFactory`,
`TransposeWHProgramFactory` (tiled + row-major), `TransposeWHShardedProgramFactory`.
The two gated factories (`TransposeHCShardedProgramFactory`,
`TransposeWHShardedRMProgramFactory`) stay on the legacy `create_descriptor` path and
are **not** touched.

Written during the inventory and planning steps; committed alongside the port.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — all 8 factories define
  `create_descriptor(...) -> ProgramDescriptor`.
- Device operation: `ttnn::prim::TransposeDeviceOperation` (single device op, 8-factory
  `program_factory_t` variant). Framework hooks: `select_program_factory`,
  `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`.
- Custom `compute_program_hash`: **none** (already the default reflection hash — audit
  confirmed). No device-op-class edit forced.
- `get_dynamic_runtime_args`: declared on the shared device op
  (`transpose_hc_sharded_program_factory.cpp:432`); returns `{}` for the 6 clean
  factories, services only the 2 gated ones. **Left in place** (coupling heads-up).
- Pybind: nanobind binds only the `transpose` free function; no `create_descriptor`
  pybind hook. No pybind removal forced.

### Kernels (per factory)

**CN** (`transpose_cn_program_factory.cpp`), work-split by pages, 1 group pair:
- reader `reader_unary_transpose_cn_interleaved_start_id.cpp` — CTA {cb0, page_size,
  read_size} + `TensorAccessorArgs(src, RuntimeTensorShape)`; RTA {src_addr, N, C, HtWt,
  batch_step, channel_step, num_pages, start_id, hw, n}; define `CN_RM` when row-major.
- writer `writer_unary_transpose_cn_interleaved_start_id.cpp` — CTA {cb0, page_size,
  write_size} + `TensorAccessorArgs(dst, RuntimeTensorShape)`; RTA {dst_addr, num_pages,
  start_id}; define `CN_RM`.
- CB `c_0` (single-tile / stick, 2 entries).

**HC-RM** (`transpose_hc_rm_program_factory.cpp`), work-split by sticks:
- reader `reader_unary_transpose_hc_interleaved_partitioned_rm.cpp` — named CTA read
  N,H,C,stick_size(read),aligned_page(unused) + `TensorAccessorArgs(src,
  RuntimeTensorShape)`; RTA {src_addr, num_sticks_per_core_read, num_read_per_barrier,
  start_id, curr_c, curr_h, curr_n}.
- writer `writer_unary_transpose_hc_interleaved_start_id_rm.cpp` — CTA {cb0, stick_size,
  aligned_page(unused)} + `TensorAccessorArgs(dst, RuntimeTensorShape)`; RTA {dst_addr,
  num_sticks_per_core_read, num_read_per_barrier, start_id}.
- CB `c_0` (stick).

**HC-Tiled-Interleaved** (`transpose_hc_tiled_interleaved_program_factory.cpp`), 2 parallel work-splits:
- reader `reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` — named CTAs
  {num_writes, padding_val_packed, needs_padding, swap_hw, H, W, accumulated_outer_dims,
  tile_height, tile_width} + `TensorAccessorArgs(src, RuntimeTensorShape)`; RTA
  {src_addr, num_tiles, start_id}. Touches `c_1` padding buffer when `needs_padding`.
- writer `writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` — positional
  CTA {element_size, cb0, C, H, W, TILE_H, TILE_W, FACE_H, FACE_W, needs_padding} +
  `TensorAccessorArgs(dst, RuntimeTensorShape)`; RTA {dst_addr, start, end,
  padded_start, padded_end}. Touches `c_1` when `needs_padding`.
- CBs `c_0` (2 tiles), `c_1` (padding, only when `C % tile_h != 0`).

**HC-Tiled** (`transpose_hc_tiled_program_factory.cpp`), work-split by tiles:
- reader `reader_unary_transpose_hc_interleaved_partitioned.cpp` — CTA
  {SUBTILE_LINE_BYTES, FLOAT32_DTYPE, ALIGNMENT} + `TensorAccessorArgs(src)`; RTA (14
  values). Touches scratch `c_1` (via `dfb_scratch.get_write_ptr()`) only when
  MISALIGNED (`ALIGNMENT > SUBTILE_LINE_BYTES`).
- writer **DONOR** `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` — CTA {cb0}
  + `TensorAccessorArgs(dst)`; RTA {dst_addr, num_pages, start_id}.
- CBs `c_0` (2 tiles), `c_1` (scratch = alignment bytes, only when misaligned).

**WH** (`transpose_wh_program_factory.cpp`), one factory, tiled **and** row-major paths:
- tiled: reader `reader_unary_transpose_wh_interleaved_start_id.cpp`; compute
  `transpose_wh.cpp`; writer **DONOR** `writer_unary_interleaved_start_id.cpp`. CBs c_0,
  c_16.
- row-major: reader `reader_unary_transpose_wh_interleaved_start_id_rm.cpp`; compute
  `transpose_wh_rm.cpp` (**shared top-level entry point with the gated WH-Sharded-RM
  factory** via `#ifdef SHARDED`); writer
  `writer_unary_transpose_wh_interleaved_start_id_rm.cpp`. CBs c_0, c_16, c_24 (tilize
  self-loop), **c_25 dead — dropped**.
- compute config Style B: `fp32_dest_acc_en` (Float32/Int32/UInt32);
  `unpack_to_dest_mode[c_0]=UnpackToDestFp32` when Float32 (+ `[c_24]` on RM path);
  define `DST_ACCUM_MODE=1` when row-major and Int32/UInt32.

**WH-Sharded** (`transpose_wh_sharded_program_factory.cpp`), sharded grid, no-op tail cores:
- reader **DONOR** `eltwise/unary/.../reader_unary_sharded.cpp` — CTA {cb0}; RTA
  {num_tiles}. Producer of borrowed `c_0`.
- writer **DONOR** `data_movement/sharded/.../writer_unary_sharded.cpp` — CTA {cb16};
  RTA {num_units}. Consumer of borrowed `c_16`.
- compute `transpose_wh_sharded.cpp` — CTA {cb0, cb16}; RTA {NHtWt, HtWt, N, Ht, Wt}.
- CBs `c_0` (borrowed from input), `c_16` (borrowed from output).
- compute config Style B: `fp32_dest_acc_en` (Float32);
  `unpack_to_dest_mode[c_0]=UnpackToDestFp32` when Float32.

### Semaphores
none (no semaphores anywhere in the subset).

### Tensor accessors (host sites)
Every interleaved factory delivers tensor bases through the `Buffer*`-binding RTA form
(`emplace_runtime_args(core, {tensor.buffer(), ...})`) + `TensorAccessorArgs(...)` in the
CTA list. WH-Sharded uses borrowed CBs (`.buffer =`). All Case 1 (via `TensorAccessor`)
except WH-Sharded (borrowed-DFB, clean). No `->address()` fold anywhere.

### Work split
- CN / HC-Tiled / WH: `split_work_to_cores(grid, num_tensor_tiles|pages)` → group_1/group_2.
- HC-RM: `split_work_to_cores(grid, NCH)`.
- HC-Tiled-Interleaved: two `split_work_to_cores` (unpadded + padded tile counts).
- WH (RM): `split_work_to_cores(grid, NC)`.
- WH-Sharded: shard grid + no-op tail (`grid_to_cores_with_noop`).

### Cross-op kernels (out of the op directory)
1. `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — donor
   writer for HC-Tiled (consumes c_0) and WH-tiled (consumes c_16). ~42 host files.
2. `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` — donor reader for
   WH-Sharded. ~17 host files.
3. `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` — donor
   writer for WH-Sharded. ~15 host files.

Orchestration constraint forbids editing files outside the op directory, so each donor is
**forked into the transpose op's own kernels/dataflow directory** (rather than the
recipe's default "alongside the original"). Legacy copies stay untouched for other
consumers.

### Flags
- `transpose_wh_rm.cpp` is a **shared top-level entry point** between the in-scope WH
  factory (RM path, no `SHARDED`) and the gated WH-Sharded-RM factory (`#define SHARDED`).
  Porting it in place would break the gated factory → **fork** it into the op dir
  (`transpose_wh_rm_transpose_m2.cpp`), strip the `SHARDED` branch, point the WH factory
  at the fork. Legacy `transpose_wh_rm.cpp` stays for the gated factory.
- Unreferenced dead kernel `reader_unary_transpose_wh_interleaved.cpp` — not audited, not
  touched.
- Dead CB `c_25` (WH-RM) dropped.

## TTNN ProgramFactory
- **Concept (inherited from audit):** `MetalV2FactoryConcept` (all 6).
- **Custom `compute_program_hash`:** none — already default. No deletion.
- **Implementation notes:** each ported factory's `.hpp` changes its method from
  `create_descriptor(...) -> ProgramDescriptor` to
  `create_program_artifacts(...) -> ttnn::device_operation::ProgramArtifacts`, signature
  `(const TransposeParams&, const TransposeInputs&, Tensor& output_tensor)`. The device
  operation `.hpp`/`.cpp` need no change (variant unchanged; per-factory concept dispatch;
  no custom hash; `get_dynamic_runtime_args` retained for the 2 gated factories).

## Planned Spec Shape

Default 1:1 with legacy. Per factory (all names prefixed to avoid unity-build
anonymous-namespace collisions across the 6 factory `.cpp`s):

- **CN:** KernelSpecs reader+writer; DFB `c_0` (CN_SRC0); TensorParameters input,output;
  1 WorkUnitSpec (total_cores). Preserved multiplicity: none — single KernelDescriptor
  per source (work-split via per-core RTAs, not per-group CTAs).
- **HC-RM:** reader+writer; DFB `c_0`; input,output; 1 WU.
- **HC-Tiled-Interleaved:** reader+writer; DFB `c_0` + conditional `c_1` (padding, when
  needs_padding); input,output; 1 WU.
- **HC-Tiled:** reader + donor-writer; DFB `c_0` + conditional `c_1` (scratch self-loop,
  when misaligned); input,output; 1 WU.
- **WH tiled:** reader+compute+writer(donor); DFB `c_0`, `c_16`; input,output; 1 WU.
- **WH row-major:** reader+compute+writer; DFB `c_0`, `c_16`, `c_24` (tilize self-loop);
  input,output; 1 WU. Drop dead `c_25`.
- **WH-Sharded:** reader(donor)+compute+writer(donor); DFB `c_0` (borrowed_from input),
  `c_16` (borrowed_from output); input,output; 1 WU (shard grid + no-op tail cores).

## Preserved Multiplicity
none — no factory uses multiple `KernelDescriptor`s of one source for a per-group CTA
work split. Every factory emits **one** KernelDescriptor per kernel source and splits work
purely through per-core runtime args. So each Metal 2.0 KernelSpec is 1:1 with a legacy
KernelDescriptor, work-split preserved via `runtime_arg_values` per node.

## Dropped Plumbing

Per factory, legacy plumbing that evaporates:

- **Buffer-address RTA slot 0** (`emplace_runtime_args(core, {tensor.buffer(), ...})`) →
  `TensorParameter` + `TensorBinding` (input, output). Every reader/writer.
- **Magic CB-index CTA** (`{src0_cb_index}`, `{output_cb_index}`, hardcoded
  `tt::CBIndex::c_0` / `c_16` / `c_1` / `c_24` in kernels) → `DFBBinding` /
  `dfb::name`.
- **`TensorAccessorArgs(...).append_to(...)`** (host) + `TensorAccessorArgs<N>()`
  (kernel) → the binding mechanism end-to-end; the `RuntimeTensorShape` common-runtime
  arg it emitted drops (see Deferred/Flagged for the relaxation note).
- **Vestigial `aligned_page_size` CTA** in WH-RM reader/writer (emitted but never read
  by the kernel; TensorAccessor 3rd-arg-style page size) → dropped.
- **Positional CTAs** (HC-Tiled-Interleaved writer, HC-Tiled reader, CN, HC-RM,
  WH readers) → named CTAs.

No semaphore-ID RTAs (no semaphores). No page-size 3rd TensorAccessor argument anywhere.

## Applied Patterns
- **Self-loop DFB binding** — HC-Tiled `c_1` scratch (reader P+C, conditional on
  misaligned); WH-RM `c_24` tilize (compute P+C).
- **Conditional / optional DFB bindings** — HC-Tiled `c_1` scratch (define
  `TRANSPOSE_HC_SCRATCH` when misaligned); HC-Tiled-Interleaved `c_1` padding (define
  `NEEDS_PADDING` when needs_padding). Host binds conditionally, kernel `#ifdef`-gates the
  construction and uses.
- **Dead-CB drop** — WH-RM `c_25` (im2), factory allocation removed.
- **Multi-variant factory** — WH factory branches tiled vs row-major inside
  `create_program_artifacts`.
- **Pass DFB handles directly to LLKs** — compute kernels pass `dfb::in`/`dfb::out`/
  `dfb::tilize` to `transpose_init`, `transpose_tile`, `pack_tile`,
  `compute_kernel_hw_startup`, `unary_op_init_common`, `compute_kernel_lib::tilize`,
  `pack_untilize_*`.
- **Borrowed-memory DFBs** — WH-Sharded `c_0`/`c_16` (`borrowed_from` input/output
  TensorParameters).
- **Modifying a shared dataflow kernel (fork)** — 3 external donors + `transpose_wh_rm.cpp`
  forked into the op dir.
- **Unity-build hygiene** — all spec-name constants prefixed per factory.

## Deferred / Flagged
- **`ArgConfig::RuntimeTensorShape` vs audit "relaxation = none".** CN / HC-RM /
  HC-Tiled-Interleaved / WH interleaved host code appends
  `TensorAccessorArgs(buffer, RuntimeTensorShape)`. The migration guide's TensorParameter
  pre-flight flags `RuntimeTensorShape` as a candidate for
  `advanced_options.dynamic_tensor_shape = true`. The audit recorded relaxation = none,
  and the recipe/ttnn_factory bias is **strict** (forgetting to relax is merely slower,
  still correct; relaxing wrongly is silent wrong output; and TTNN's program-cache key
  already folds the tensor spec, so a shape change already forces a fresh program — the
  legacy runtime-shape reuse was never exploited at the TTNN cache layer). Decision:
  **strict TensorParameters, no relaxation**, matching the audit. Recorded as a downstream
  relaxation candidate in the report, not applied in the port.
- **`transpose_wh_rm.cpp` shared with the gated factory** — resolved by fork (see Flags).

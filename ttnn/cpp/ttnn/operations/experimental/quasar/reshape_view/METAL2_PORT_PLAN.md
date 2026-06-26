# Metal 2.0 Port Plan — `experimental/quasar/reshape_view`

Audit: GREEN (see `METAL2_PREPORT_AUDIT.md`). Recipe: `port_op_to_metal2_recipe.md` @ `origin/akertesz/metal2-documentation`.
**Porting unit this pass: `ReshapeViewRMProgramFactory`** (the RM/interleaved factory). The tiled factory (`ReshapeViewTiledProgramFactory`) stays on `ProgramDescriptorFactoryConcept` this pass — the `program_factory_t` variant dispatches per-factory, so the op builds and runs with one factory on Metal 2.0 and one on legacy. Tiled factory enumerated as remaining work.

## Legacy Inventory (RM factory)
- **Legacy factory shape:** `ProgramDescriptorFactoryConcept` — `ReshapeViewRMProgramFactory::create_descriptor()` returns `tt::tt_metal::ProgramDescriptor`.
- **Custom `compute_program_hash`:** YES — `ReshapeViewDeviceOperation::compute_program_hash` (`reshape_device_operation.hpp:43`, defined `reshape_device_operation.cpp`). **Port deletes it** (device-op-class edit #1).
- **Kernels:** one source `device/device/rm_reshape_interleaved.cpp`, instantiated as **1 or 2** `KernelDescriptor`s:
  - reader (`ReaderConfigDescriptor`) — always present, binds CBs 0/1.
  - writer (`WriterConfigDescriptor`) — present only when `can_use_dual_kernel` (src/dst page sizes evenly divide); binds CBs 2/3. Same source, different CB pair → **preserved multiplicity**.
- **CBs:** `src0`(idx0,`cb_size0`=`source_read_size_bytes`,×2 total),`src1`(idx1,`cb_size1`); + `src2`(idx2),`src3`(idx3) when dual. `data_format` = input dtype. No tile metadata (row-major). No GlobalCB, no aliasing, no borrowed-mem.
- **Semaphores:** none.
- **Tensor accessors:** `input` (src) and `output` (dst). Host appends `TensorAccessorArgs(*src_buffer)`/`(*dst_buffer)` to CTAs; per-core RTA passes `src_buffer`/`dst_buffer` as `Buffer*` (BufferBinding form). Kernel builds `TensorAccessor(src_args, src_addr)` / `(dst_args, dst_addr)` → **Case 1** (via TensorAccessor) for both.
- **Work split:** bespoke — `responsibility = ceil(input[-2]/num_cores)` rounded up until `responsibility*src_page % dst_page == 0`; iterate `corerange_to_cores(total_cores)`, assign `[read_start,read_end)` per core, track `done`/idle. Not the standard `split_work_to_cores`. Cores: `sub_core_grid` or full `compute_with_storage_grid_size`.
- **Cross-op kernels:** none (kernel is op-owned).
- **Flags:** kernel is Device-2.0 clean (ported this session). Idle cores get a `nop=1` RTA tail and `0u` buffer slots.

## Planned Spec Shape (RM factory → MetalV2FactoryConcept)
- **Concept:** `MetalV2FactoryConcept` — `create_program_artifacts()` returns `ttnn::device_operation::ProgramArtifacts`.
- **KernelSpecs:** `READER` always; `WRITER` when dual — both source `rm_reshape_interleaved.cpp`. Preserved multiplicity (1 source → up to 2 KernelSpecs).
- **DataflowBufferSpecs:** `SRC0,SRC1` always; `SRC2,SRC3` when dual. `entry_size`=page size, `num_entries`=2 (src0/src2) or 1 (src1/src3) to mirror legacy `total_size/page_size`. `data_format_metadata`=input cb_data_format. No tile metadata.
- **TensorParameters:** `INPUT`, `OUTPUT` (both Case 1).
- **WorkUnitSpecs:** `wu` = {READER(+WRITER when dual)} on `total_cores`. (Idle cores remain in the node set; the kernel's `nop` RTA short-circuits them — preserved as a runtime arg, not a work-unit change.)
- **Op-owned tensors:** none.

## Preserved Multiplicity
```
Legacy KernelDescriptors [reader, writer] of rm_reshape_interleaved.cpp
  → KernelSpecs [READER, WRITER] of same source
  → in WorkUnitSpec [wu]
  → reader binds dfb in0→SRC0, in1→SRC1 ; writer binds in0→SRC2, in1→SRC3
```
Single-kernel (non-dual) case: only READER + SRC0/SRC1.

## Dropped Plumbing
- **Buffer-address RTAs** (`src_buffer`/`dst_buffer` Buffer* in slots 0,1) → `TensorBinding` (INPUT, OUTPUT), Case 1. Kernel: `TensorAccessor(ta::input)`/`(ta::output)`.
- **`TensorAccessorArgs` plumbing** (host `append_to` + kernel `TensorAccessorArgs<6>()`/`next_compile_time_args_offset()`) → binding mechanism.
- **Magic CB indices in CTAs** (slots 2,3 = `src0/src1` idx) → `DFBBinding` (`in0`/`in1`).
- **Positional CTAs** (slots 0,1,4,5 = `src_aligned_64`, `src_aligned_16`, `source_page_size_bytes`, `dest_page_size_bytes`) → named CTAs.
- Per-core RTAs (`source_read_size_bytes`, `read_start_page`, `read_end_page`, `write_start_page`, `write_start_offset`, `nop`) → named per-node RTAs (`runtime_arg_schema` + `KernelRunArgs`). `write_start_offset` is always 0 (legacy already dropped it from semantics) — keep as named RTA to preserve kernel arg shape (kernel reads arg 6).

## Applied Patterns
- [Preserve work-split multiplicity](metal2_port_patterns) — 2 KernelSpecs same source (dual-kernel path).
- DM RoleHint: READER → `DataMovementRoleHint::READER`, WRITER → `::WRITER`.

## Deferred / Flagged
- **Tiled factory** (`ReshapeViewTiledProgramFactory`, kernels `reader_reshape_tiled.cpp`/`writer_reshape_tiled.cpp`) — remaining work, next pass. Stays on legacy concept; op still builds/runs (per-factory dispatch).
- Custom `compute_program_hash` deletion is op-wide (affects both factories' caching) — deleting now is correct (default hash covers both); noted as device-op edit forced by this pass.

## VERDICT (this pass): STOP / CAPITULATION
Both factories blocked on DM single-ended producer scratch CBs (under active design). See METAL2_PORT_REPORT.md. No code changed.

# Port Plan — repeat/codegen (`RepeatCodegenProgramFactory`)

Port plan for `ttnn/cpp/ttnn/operations/data_movement/repeat/codegen`, ported from the
`ProgramDescriptor` API (`create_descriptor`) to Metal 2.0 (`create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

*Observation step. Source of truth: `repeat_codegen_program_factory.cpp` @ `3a0efe4bd07` (origin/main).*

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `static ProgramDescriptor create_descriptor(...)`
  (`repeat_codegen_program_factory.hpp:39-42`), living in a **`program_factory_t` variant**
  (`repeat_codegen_device_operation.hpp:20`, single alternative). Not the direct-descriptor shape;
  ttnn_factory exception 3 does not apply.
- Variants: **single factory, three internal branches** selected at `repeat_codegen_program_factory.cpp:91-92`:
  `TILE` (`!is_row_major`, lines 99-165), `RM last-dim` (`is_last_dim_rm`, 167-220), `RM higher-dim`
  (fallthrough, 222-274). The atomic unit is the whole factory + all five kernel entry points it can bind.
- Custom `compute_program_hash`: **none** — default reflection-based hash over `RepeatCodegenParams`
  (`rep_dim, num_repeats, lower_pages, rep_dim_pages, total_out_pages, stick_size, output_mem_config`).
  No `attribute_values` / `to_hash` backdoor. Nothing to leave alone; nothing touched.
- `override_runtime_arguments`: none → target is the **base** concept (inherited from the brief).
- Pybound `create_descriptor`: none (`repeat_nanobind.cpp` binds `repeat`, `repeat_force_native`,
  `repeat_force_codegen` free functions only). No pybind edit forced.

### Kernels

Positional CTAs are listed in **host emission order** (`repeat_codegen_program_factory.cpp`).
`TAA(x)` = `TensorAccessorArgs(*x).append_to(...)`. `n`/`start` = per-core page count / first output page.

#### Branch: TILE (`!is_row_major`, lines 99-165)

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `data_movement/common/kernels/codegen/reader_tile_interleaved_unified.cpp` (**borrowed**) | `split.all_cores` | `[TAA(src_buffer)]` at index 0 | `seq_id=1 (SEQ_REPEAT)`, `cb_id=0`, `batch=kReadBatch=4`, `src_page_pitch=0` | `[src_buffer, n, start, num_repeats, lower_pages, rep_dim_pages]` (kernel reads as `ArgsRepeat` struct overlay via `get_arg_addr(0)`) | none | none | unset → **O2** | `ReaderConfigDescriptor{}` → RISCV_1 / NOC_0 / DM_DEDICATED_NOC |
| writer | `data_movement/common/kernels/codegen/writer_interleaved.cpp` (**borrowed**) | `split.all_cores` | `[0 (cb_out), page_size (REQUESTED_WRITE_SIZE = dst aligned page), TAA(dst_buffer), kWriteBatch=4 (BATCH)]` | none | `[dst_buffer, num_tiles=n, start_id=start]` | none | none | unset → **O2** | `WriterConfigDescriptor{}` → RISCV_0 / NOC_1 / DM_DEDICATED_NOC |

#### Branch: RM last-dim (`is_last_dim_rm`, lines 167-220)

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `repeat/codegen/kernels/reader_repeat_last_dim_rm.cpp` (own) | `split.all_cores` | `[stick_size=in_stick_size, in_read_size=in_aligned, out_l1_stride=out_aligned, TAA(src_buffer), cb_id=0, NUM_REPEATS=num_repeats, BATCH=kReadBatch]` | none | `[src_buffer, num_pages=n, start_page=start]` | none | none | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer | `repeat/codegen/kernels/writer_repeat_rm.cpp` (own) | `split.all_cores` | `[cb_out=0, xfer_size=out_aligned, l1_stride=out_aligned, TAA(dst_buffer), BATCH=kWriteBatch]` | none | `[dst_buffer, num_pages=n, start_id=start]` | none | none | unset → **O2** | `WriterConfigDescriptor{}` |

#### Branch: RM higher-dim (fallthrough, lines 222-274)

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `repeat/codegen/kernels/reader_repeat_higherdim_rm.cpp` (own) | `split.all_cores` | `[xfer_size=aligned, l1_stride=aligned, TAA(src_buffer), cb_id=0, NUM_REPEATS, LOWER_PAGES, REP_DIM_PAGES, BATCH=kReadBatch]` | none | `[src_buffer, num_out_pages=n, out_start_page=start]` | none | none | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer | `repeat/codegen/kernels/writer_repeat_rm.cpp` (own, **shared with the last-dim branch of this same factory** — converts together, no fork needed) | `split.all_cores` | `[cb_out=0, xfer_size=aligned, l1_stride=aligned, TAA(dst_buffer), BATCH=kWriteBatch]` | none | `[dst_buffer, num_pages=n, start_id=start]` | none | none | unset → **O2** | `WriterConfigDescriptor{}` |

### CBs

One plain `CBDescriptor` per branch; no `GlobalCircularBuffer`, no `buffer` (borrowed memory), no `address_offset`, no `tile`.

| branch | index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|---|
| TILE | 0 | `kRepeatCbDepth(8) * dst_buffer->aligned_page_size()` | `split.all_cores` | `datatype_to_dataformat_converter(input.dtype())` | `dst_buffer->aligned_page_size()` | unset |
| RM last-dim | 0 | `8 * dst_buffer->aligned_page_size()` (`out_aligned`) | `split.all_cores` | same | `out_aligned` | unset |
| RM higher-dim | 0 | `8 * src_buffer->aligned_page_size()` (`aligned_page_size`; == dst aligned on this branch) | `split.all_cores` | same | `aligned_page_size` | unset |

Kernel-touch census per `(CB0, branch)` (re-derived from the kernels, agrees with the brief):
- TILE: `reader_tile_interleaved_unified.cpp::read_pages` FIFO-produces (`reserve_back`/`push_back`) → locked PRODUCER; `writer_interleaved.cpp` FIFO-consumes (`wait_front`/`pop_front`) → locked CONSUMER. 2 touchers, 1P+1C.
- RM last-dim: `reader_repeat_last_dim_rm.cpp` FIFO-produces and additionally peeks `cb_in.get_write_ptr()` (line 60) + uses `cb_in` as the L1→L1 `noc.async_read(self_ep, cb_in, …)` destination (84-89) — same kernel, one toucher; `writer_repeat_rm.cpp` FIFO-consumes. 2 touchers, 1P+1C.
- RM higher-dim: `reader_repeat_higherdim_rm.cpp` FIFO-produces; `writer_repeat_rm.cpp` FIFO-consumes. 2 touchers, 1P+1C.

### Semaphores
none

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) | kernel site |
|---|---|---|---|
| `repeat_codegen_program_factory.cpp:115` `TAA(*src_buffer)` | `tensor_args.input` | reader RTA[0] = `src_buffer` (`:152`) | `reader_tile_interleaved_unified.cpp:157,169` — `TensorAccessor(src_args, base->src_addr, source_page_size)` (3-arg, Class 2) |
| `:136` `TAA(*dst_buffer)` | `tensor_return_value` | writer RTA[0] = `dst_buffer` (`:158`) | `writer_interleaved.cpp:21,28` — `TensorAccessor(dst_args, dst_addr, destination_page_size)` (3-arg, Class 2) |
| `:184` `TAA(*src_buffer)` | input | reader RTA[0] (`:212`) | `reader_repeat_last_dim_rm.cpp:42,47` — 2-arg |
| `:198` `TAA(*dst_buffer)` | output | writer RTA[0] (`:213`) | `writer_repeat_rm.cpp:33,36` — 2-arg |
| `:237` `TAA(*src_buffer)` | input | reader RTA[0] (`:266`) | `reader_repeat_higherdim_rm.cpp:26,33` — 2-arg |
| `:253` `TAA(*dst_buffer)` | output | writer RTA[0] (`:267`) | `writer_repeat_rm.cpp:33,36` — 2-arg |

All Case 1 (every access goes through `noc.async_read/async_write(accessor, …, {.page_id})`). No offset folding: the
`start` page index travels as a separate scalar and is applied as `page_id` kernel-side.

### Work split
- Driver: `split_work_to_cores(device->compute_with_storage_grid_size(), operation_attributes.total_out_pages, /*row_wise=*/false)`
  (`repeat_codegen_program_factory.cpp:46-66`), then `corerange_to_cores(all_cores, num_cores, /*row_wise=*/false)`.
- num_cores: derived per call.
- core_group_1: `core_group_1`, count_per_core: `work_per_core_1`
- core_group_2: `core_group_2`, count_per_core: `work_per_core_2`
- Per-core `n` = `work_for_core(split, core)`; `start` accumulates over `cores_in_order`. **Keep `row_wise=false`**
  (the comment at `:49-55` explains the perf reason — column-major enumeration to match the generator).
- Both kernels are pushed once over `split.all_cores`; the per-group count reaches the kernel as an **RTA** (`n`),
  not a CTA — there is no per-group `KernelDescriptor` multiplicity to preserve.

### Shared kernels

Census: `grep -rl reader_tile_interleaved_unified ttnn/cpp/ttnn/operations/` and the same for `writer_interleaved.cpp`,
each hit disambiguated by the bound *path*:

| kernel source | shape | other binders (by full path) | `_metal2` fork beside it? | rung |
|---|---|---|---|---|
| `ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/reader_tile_interleaved_unified.cpp` | **borrowed** (data_movement/common shared codegen pool) | `repeat_interleave/codegen/repeat_interleave_codegen_program_factory.cpp:38→113` (`seq_id = SEQ_REPEAT_INTERLEAVE`) | **no** (`ls` of the dir: `reader_tile_interleaved_unified.cpp`, `sequencers.h`, `writer_interleaved.cpp`) | **2 — create `reader_tile_interleaved_unified_metal2.cpp`** beside it + pointer comment in the original |
| `ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/writer_interleaved.cpp` | **borrowed** | `repeat_interleave/codegen/repeat_interleave_codegen_program_factory.cpp:40→124,188` | **no** | **2 — create `writer_interleaved_metal2.cpp`** + pointer comment |
| `common/kernels/codegen/sequencers.h` | included by the borrowed reader; pure scalar index arithmetic, no resource handles | — | n/a | **untouched**; the fork keeps `#include "sequencers.h"` (same directory) |
| `repeat/codegen/kernels/{reader_repeat_last_dim_rm,reader_repeat_higherdim_rm,writer_repeat_rm}.cpp` | own, **not lent** (tree-wide grep: bound by this factory only) | — | n/a | convert in place |

Discarded grep hits (not consumers): `untilize/codegen/kernels/reader_tile_interleaved_unified.cpp` is a same-named
*private copy* at a different path; `transformer/sdpa/*`, `experimental/slice_write/*`, `reduction/sampling/*` match the
substring `writer_interleaved.cpp` inside other filenames; the two `METAL2_*.md` files and `sequencers.h` mention the
file in prose.

Fork vocabulary (fixed by this first fork; `repeat_interleave/codegen` will inherit it):
- reader fork: `tensor::src`, `dfb::in`, named CTAs `seq_id`, `batch`, `src_page_pitch`; named RTAs `num_pages`,
  `start_id`, `num_repeats`, `lower_pages`, `rep_dim_pages`.
- writer fork: `tensor::dst`, `dfb::out`, named CTAs `requested_write_size`, `batch`; named RTAs `num_tiles`, `start_id`.

### Flags
- Unreferenced kernel files in the op directory: none (all three under `kernels/` are bound).
- Descriptor types used: `CBDescriptor`/`CBFormatDescriptor`, `KernelDescriptor` with `ReaderConfigDescriptor` /
  `WriterConfigDescriptor`, `TensorAccessorArgs` — all inside the audit's Appendix A scope. No `SemaphoreDescriptor`,
  no `ComputeConfigDescriptor`, no `WorkloadDescriptor`, no `GlobalCircularBuffer`.
- The borrowed reader is a **multi-sequencer** kernel (10 `SEQ_*` ids). Only `SEQ_REPEAT` (this op) and
  `SEQ_REPEAT_INTERLEAVE` (`repeat_interleave/codegen`) have binders in the tree. Its `SEQ_SLICE` / `SEQ_PERMUTE`
  branches read genuine varargs tails; `SEQ_PAD` reads a CB index (`cb_pad`) and `SEQ_CONCAT` a second buffer address
  (`src_addr_1`) out of the RTA block — all dead for every current binder. See Planned Spec Shape / Deferred for the
  fork decision.
- `src_page_pitch` (TILE reader named CTA, always `0` from both binders) feeds **two** things in the legacy kernel:
  the accessor's 3rd argument (dropped in the fork) **and** `source_read_size` (the transfer size, still live). It is
  therefore *not* a page-size-3rd-argument-only CTA and stays.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — `create_program_artifacts` returning
  `ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`; no `override_runtime_arguments`; no op-owned tensors.
- **Custom `compute_program_hash`**: none — default reflection hash; nothing to leave intact.
- **Implementation notes**:
  - The device-op keeps `program_factory_t = std::variant<RepeatCodegenProgramFactory>` and `select_program_factory`
    unchanged. The only device-op-class touch is **none**: `repeat_codegen_device_operation.{hpp,cpp}` are not edited.
    `repeat_codegen_program_factory.hpp` swaps the method signature and the `program_descriptors.hpp` include for
    `ttnn/metal_v2_artifacts.hpp`; `kRepeatCbDepth`, `RepeatCodegenParams`, `RepeatCodegenInputs` stay (the header is
    also included by `repeat_codegen_supported.cpp` for `kRepeatCbDepth`).
  - `RepeatCodegenParams` already carries every value the kernels consume; no attribute-struct change.
  - Spec names are declared **function-locally** (unity build: this `.cpp` shares a TU with
    `repeat_interleave_codegen_program_factory.cpp`, whose anonymous namespace already prefixes its constants to avoid
    colliding with this file's `kReadBatch`/`kWriteBatch`/`kSeqRepeat`).

## Planned Spec Shape

Default 1:1 with legacy, one `ProgramSpec` per branch (the three branches stay mutually exclusive, selected by the
same `is_row_major` / `is_last_dim_rm` tests at the same place).

- **KernelSpecs**: exactly two per branch (`READER{"reader"}`, `WRITER{"writer"}`), one per legacy `KernelDescriptor`.
  - `hw_config`: reader → `ttnn::create_reader_datamovement_config(device->arch())` (legacy `ReaderConfigDescriptor{}` =
    RISCV_1/NOC_0/dedicated), writer → `ttnn::create_writer_datamovement_config(device->arch())` (RISCV_0/NOC_1/dedicated).
    Both match the role defaults value-for-value; no custom Gen1 config.
  - `compiler_options.opt_level`: left at the Metal 2.0 default `O2` on both — legacy DM `opt_level` was unset (→ O2).
    No compute kernels, so rule 2 (explicit O3) has no subject.
  - `defines`: none. No conditionally bound resources.
- **DataflowBufferSpecs**: one per branch, `PAGES{"pages"}` (legacy `buffer_index 0`): `entry_size` = the branch's
  aligned page (TILE: `dst_buffer->aligned_page_size()`; RM last-dim: `out_aligned`; RM higher-dim:
  `aligned_page_size`), `num_entries = kRepeatCbDepth`, `data_format_metadata = datatype_to_dataformat_converter(input.dtype())`
  (carried over from the legacy `CBFormatDescriptor.data_format`; harmless on a DM-only DFB). No `tile_format_metadata`
  (legacy set none), no `borrowed_from`, no `advanced_options`.
  Endpoints: reader `DFBBinding{PAGES, "in", PRODUCER}`, writer `DFBBinding{PAGES, "out", CONSUMER}` — plain 1P+1C on
  every node in every branch. No self-loop, no multi-binding flag.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT{"input"}` (`input.tensor_spec()`) and `OUTPUT{"output"}` (`output.tensor_spec()`), strict
  matching (brief: relaxation `none`). Reader binds `TensorBinding{INPUT, "src"}`; writer binds `TensorBinding{OUTPUT, "dst"}`.
  `run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}}` — the same `Tensor` objects the
  factory received, no copies.
- **WorkUnitSpecs**: one per branch, `{READER, WRITER}` over `split.all_cores`.
- **Op-owned tensors**: none.
- **KernelRunArgs**: one per kernel; the legacy `for (core : cores_in_order)` loop is kept as-is and transposed with
  `AddRuntimeArgsForNode`. No CRTAs (none in legacy; `num_repeats`/`lower_pages`/`rep_dim_pages` are the same on every
  node and are noted as a *later* RTA→CRTA cleanup candidate, not converted here).

## Preserved Multiplicity

none — no work-split multiplicity in legacy. Each branch pushes exactly one reader and one writer `KernelDescriptor`
over `split.all_cores`; the per-core page count is an RTA.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:115` + reader RTA[0] `:152` (`src_buffer`) | `TAA(*src_buffer)` positional CTA at 0 + `Buffer*` address RTA | `TensorParameter INPUT` + `TensorBinding{INPUT,"src"}`; fork: `TensorAccessor(tensor::src)` |
| factory `:136` + writer RTA[0] `:158` (`dst_buffer`) | `TAA(*dst_buffer)` after 2 positional CTAs + `Buffer*` RTA | `TensorParameter OUTPUT` + `TensorBinding{OUTPUT,"dst"}`; fork: `TensorAccessor(tensor::dst)` |
| factory `:184`,`:237` + reader RTA[0] `:212`,`:266` | `TAA(*src_buffer)` + `Buffer*` RTA | same INPUT binding; `reader_repeat_*_rm.cpp`: `TensorAccessor(tensor::src)` |
| factory `:198`,`:253` + writer RTA[0] `:213`,`:267` | `TAA(*dst_buffer)` + `Buffer*` RTA | same OUTPUT binding; `writer_repeat_rm.cpp`: `TensorAccessor(tensor::dst)` |
| factory `:125` `{"cb_id", 0}` (TILE reader named CTA) | CB index by name | `DFBBinding{PAGES,"in",PRODUCER}`; fork: `DataflowBuffer dfb(dfb::in)` |
| factory `:135` `writer_ct_args[0] = 0` (cb_out) | magic CB index | `DFBBinding{PAGES,"out",CONSUMER}`; fork: `DataflowBuffer dfb(dfb::out)` |
| factory `:185`,`:238` `reader_ct_args.push_back(0)` (cb_id) | magic CB index after TAA | `DFBBinding{PAGES,"in",PRODUCER}`; `dfb::in` |
| factory `:197`,`:252` `writer_ct_args = {0, …}` (cb_out) | magic CB index | `DFBBinding{PAGES,"out",CONSUMER}`; `dfb::out` |
| `reader_tile_interleaved_unified.cpp:157` `TensorAccessorArgs<0>()`, `:169` 3rd arg `source_page_size` | positional accessor args + Class-2 page-size 3rd arg | `TensorAccessor(tensor::src)`; page size from `s.get_aligned_page_size()` where still needed for the transfer size |
| `reader_tile_interleaved_unified.cpp:292` (dead `SEQ_CONCAT` twin) | 3-arg accessor on `src_addr_1` | branch not carried into the fork (see Deferred) |
| `writer_interleaved.cpp:21-22` `TensorAccessorArgs<2>()` + `get_compile_time_arg_val(dst_args.next_compile_time_args_offset())`, `:28` 3rd arg | positional accessor args, offset chaining, Class-2 3rd arg | `TensorAccessor(tensor::dst)`; `BATCH = get_arg(args::batch)`; `destination_page_size = d.get_aligned_page_size()` (still needed for the min() clamp) |
| `reader_repeat_last_dim_rm.cpp:42-45`, `reader_repeat_higherdim_rm.cpp:26-31`, `writer_repeat_rm.cpp:33-34` | `TensorAccessorArgs<N>()` + `next_compile_time_args_offset() + k` chains | `TensorAccessor(tensor::src|dst)`; every trailing CTA becomes `get_arg(args::<name>)` |
| `reader_tile_interleaved_unified.cpp:160,179` `reinterpret_cast<const ArgsBase/ArgsRepeat*>(get_arg_addr(0))` | struct overlay over the RTA block | per-field named RTAs `num_pages`, `start_id`, `num_repeats`, `lower_pages`, `rep_dim_pages` (fixed field set → named, not varargs) |
| all positional `get_arg_val<uint32_t>(0..2)` in the 4 other kernels | positional RTAs | named RTAs (names = the kernel's own local variable names, see table below) |
| all positional `get_compile_time_arg_val(k)` | positional CTAs | named CTAs (table below) |

Named-argument assignment (kernel-side identifiers are **unchanged**; the arg name is the lowercase spelling of the
variable it is assigned to — the TILE reader's legacy `batch` named CTA already set this precedent for `BATCH`):

| kernel | named CTAs | named RTAs |
|---|---|---|
| `reader_tile_interleaved_unified_metal2.cpp` | `seq_id`, `batch`, `src_page_pitch` | `num_pages`, `start_id`, `num_repeats`, `lower_pages`, `rep_dim_pages` |
| `writer_interleaved_metal2.cpp` | `requested_write_size`, `batch` | `num_tiles`, `start_id` |
| `reader_repeat_last_dim_rm.cpp` | `stick_size`, `in_read_size`, `out_l1_stride`, `num_repeats`, `batch` | `num_pages`, `start_page` |
| `reader_repeat_higherdim_rm.cpp` | `xfer_size`, `l1_stride`, `num_repeats`, `lower_pages`, `rep_dim_pages`, `batch` | `num_out_pages`, `out_start_page` |
| `writer_repeat_rm.cpp` | `xfer_size`, `l1_stride`, `batch` | `num_pages`, `start_id` |

All CTA reads that drive `if constexpr` (`NUM_REPEATS`, `stick_size`, `REP_DIM_PAGES`, `LOWER_PAGES`, `BATCH`) stay
`constexpr` — `get_arg(CtaVal<T>)` is `constexpr`.

Kernel-side CB → DFB (whitelist §A/§B/§C):
- `CircularBuffer cb(x)` → `DataflowBuffer dfb(dfb::in|out)`; `reserve_back/push_back/wait_front/pop_front` 1:1.
- `get_local_cb_interface(cb).fifo_page_size << cb_addr_shift` (`reader_tile…:164`, `writer_interleaved.cpp:39`) →
  `dfb.get_entry_size()` — confirmed **bytes** (`internal/tt-1xx/dataflow_buffer.inl:42-48` applies `<< cb_addr_shift`).
- `cb_in.get_write_ptr()` (`reader_repeat_last_dim_rm.cpp:60`) → `dfb_in.get_write_ptr()` — confirmed the same absolute
  L1 byte address on WH/BH (`dataflow_buffer.h`: `get_write_ptr_impl() + L1_UNCACHED_OFFSET`, offset 0 off-Quasar;
  `.inl:118-124` returns `fifo_wr_ptr`, same field `CircularBuffer::get_write_ptr` returns). The L1→L1 self-copy and
  `CoreLocalMem` RISC copies keep working unchanged.
- Transfers stay as-is: `noc.async_read(s, dfb, …, {.page_id}, {.offset_bytes})`, `noc.async_read(self_ep, dfb, …)`,
  `noc.async_write(dfb, d, …)` — `noc_traits_t<DataflowBuffer>` accepts the same `offset_bytes` argument shapes.
- Includes: `api/dataflow/circular_buffer.h` → `api/dataflow/dataflow_buffer.h`; add `experimental/kernel_args.h`.
  Nothing else.

Kept deliberately (not plumbing):
- `src_page_pitch` named CTA (`=0`) — still consulted for `source_read_size` in the fork; both binders pass 0.
- `requested_write_size` CTA — feeds the `min()` clamp in the writer, not the accessor.
- `TT_FATAL(src_buffer/dst_buffer != nullptr)` (`:88-89`) — `src_buffer`/`dst_buffer` are still used for
  `aligned_page_size()`, so both guards stay verbatim (TT_FATAL census: 2 → 2).

## Applied Patterns

- [Caution: Porting a shared kernel](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel),
  rung 2 (create the fork) twice — `common/kernels/codegen/reader_tile_interleaved_unified_metal2.cpp` and
  `common/kernels/codegen/writer_interleaved_metal2.cpp`, each beside its original with the pointer comment added to
  the original. Bindings named for the kernel's role vocabulary (`tensor::src`/`tensor::dst`, `dfb::in`/`dfb::out`,
  `seq_id`/`batch`/`src_page_pitch`), not this op's locals, so `repeat_interleave/codegen` can bind the same forks.
- [Two-toucher DFB → assign 1P+1C](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)
  — trivially: the census yields one locked producer + one locked consumer per branch; the pattern's procedure was run,
  the result is a plain 1P+1C.
- [Unity-build hygiene](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
  — spec names declared function-locally inside `create_program_artifacts`.
- [Multi-variant factories](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)
  — the three-way branch stays inside `create_program_artifacts`, one spec per branch.
- Pass DFB handles directly / `.id` extraction: not needed — no LLK or kernel-lib call takes a CB id in these kernels.

## Deferred / Flagged

- **Fork keeps only the sequencers that have Metal 2.0 binders.** `reader_tile_interleaved_unified_metal2.cpp` carries
  the transport loop plus the `SEQ_REPEAT` and `SEQ_REPEAT_INTERLEAVE` dispatch (both read the same five `ArgsRepeat`
  fields, so one named-RTA schema serves both) and `static_assert`s on `SEQ_ID`. The `IDENTITY` / `SLICE` / `PERMUTE` /
  `TRANSPOSE_WH` / `PAD` / `CONCAT` branches are **not** carried: Metal 2.0 `args::` / `dfb::` names are generated from
  the binding factory's schema and `if constexpr` does not suppress name lookup on the discarded branch, so every
  branch's argument set would have to be declared by every binder (and `PAD`'s `cb_pad` RTA and `CONCAT`'s `src_addr_1`
  RTA are a DFB binding and a second tensor binding, respectively, not scalars). A future binder of another sequencer
  adds its branch under a `#ifdef` selected by `compiler_options.defines` (the conditional-binding pattern). The legacy
  original keeps all ten branches; nothing in the tree loses coverage.
- **`src_page_pitch` stays** in the fork's named-CTA set (rationale in Dropped Plumbing / Flags). The 3rd-argument site
  is dropped; the override still selects the transfer size.
- **RTA→CRTA candidates**, not converted (dispatch-semantics change): TILE reader `num_repeats`, `lower_pages`,
  `rep_dim_pages` are identical on every node.
- **Not a blocker, recorded**: the reader fork's `SEQ_REPEAT_INTERLEAVE` branch is untested by this port (no binder yet);
  it is a verbatim carry of the legacy branch body over the same named fields.
- No new audit-gate findings: no GlobalCB, no semaphores, no compute kernels, no offset base pointers, no Case-2 bindings.

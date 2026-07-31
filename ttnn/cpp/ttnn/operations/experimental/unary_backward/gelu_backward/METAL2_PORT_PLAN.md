# Port Plan — `experimental/unary_backward/gelu_backward`

Port plan for `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward`, ported from
`ProgramDescriptor` (`ProgramDescriptorFactoryConcept`) to Metal 2.0 (`ProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `GeluBackwardProgramFactory::create_descriptor()`
  returning `tt::tt_metal::ProgramDescriptor`, at `device/gelu_backward_program_factory.cpp:17`
  (declared `device/gelu_backward_program_factory.hpp:13`).
- Variants: single. `program_factory_t = std::variant<GeluBackwardProgramFactory>`
  (`device/gelu_backward_device_operation.hpp:25`).
- Custom `compute_program_hash`: none — already the default reflection-based hash
  (`device/gelu_backward_device_operation.cpp` defines only `validate_on_program_cache_miss`,
  `compute_output_specs`, `create_output_tensors`). Matches the brief's gate-cleared list.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see
`METAL2_PORT_BRIEF.md` → TTNN factory analysis. Carried forward in the TTNN ProgramFactory section
below.)*

### Kernels

All three `KernelDescriptor`s are `SourceType::FILE_PATH` over `all_cores`, and none sets `defines`
or `opt_level` (`grep -n opt_level device/gelu_backward_program_factory.cpp` → no hits, so every
level below is the *resolved* one: `O2` for the two DM descriptors, **`O3`** for the
`ComputeConfigDescriptor` — `tt_metal/impl/program/program.cpp:456` resolves an absent
`KernelDescriptor::opt_level` to `O3` for compute).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs (per core) | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` (**borrowed**) | `all_cores` | `{0}` (`:84`) then `TensorAccessorArgs(*src0_buffer)` (`:85`) then `TensorAccessorArgs(*src1_buffer)` (`:86`) | none | `{src0_buffer, src1_buffer, num_tiles_per_core, num_tiles_written, 0u, 0u, num_cores_y}` (`:151-152`) | none | none | `O2` (resolved) | `ReaderConfigDescriptor{}` (`:95`) → `ReaderDataMovementConfig` = `RISCV_1 / NOC_0 / DM_DEDICATED_NOC` |
| writer | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**borrowed**) | `all_cores` | `{output_cb_index}` = `{CBIndex::c_2}` (`:98`) then `TensorAccessorArgs(*dst_buffer)` (`:99`) | none | `{dst_buffer, num_tiles_per_core, num_tiles_written}` (`:156`) | none | none | `O2` (resolved) | `WriterConfigDescriptor{}` (`:108`) → `WriterDataMovementConfig` = `RISCV_0 / NOC_1 / DM_DEDICATED_NOC` |
| compute | **runtime-selected** (`:118-127`): `device/kernels/compute/eltwise_bw_gelu_approx_tanh.cpp` when `args.approximate == "tanh"`, else `device/kernels/compute/eltwise_bw_gelu_poly.cpp` (**op-owned**) | `all_cores` | none | none | `{num_tiles_per_core}` (`:154`) | none | none | **`O3`** (resolved) | `ComputeConfigDescriptor{.math_fidelity = HiFi4, .fp32_dest_acc_en = <computed>, .unpack_to_dest_mode = <computed>}` (`:133-137`) |

Resolved compute config, field by field (everything not listed is left at the
`ComputeConfigDescriptor` default — `tt_metal/api/tt-metalium/program_descriptors.hpp:99-108`):

| legacy field | legacy value |
|---|---|
| `math_fidelity` | `MathFidelity::HiFi4` (explicit; also the default) |
| `fp32_dest_acc_en` | `(dst_cb_data_format == Float32 \|\| Int32 \|\| UInt32)` (`:111-112`) |
| `dst_full_sync_en` | `false` (default) |
| `unpack_to_dest_mode` | `vector(NUM_CIRCULAR_BUFFERS, Default)` with `[c_0] = [c_1] = UnpackToDestFp32` (`:114-116`) |
| `bfp8_pack_precise` | `false` (default) |
| `math_approx_mode` | `false` (default) |

### CBs

None is a `GlobalCircularBuffer`; each has a single-element `format_descriptors` (no aliasing) and
leaves `tile` unset.

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (`:49-57`) | `2 * src0_single_tile_size` | `all_cores` | `datatype_to_dataformat_converter(grad_output.dtype())` | `src0_single_tile_size` | not set |
| `c_1` (`:59-67`) | `2 * src1_single_tile_size` | `all_cores` | `datatype_to_dataformat_converter(input.dtype())` | `src1_single_tile_size` | not set |
| `c_2` (`:69-77`) | `2 * dst_single_tile_size` | `all_cores` | `datatype_to_dataformat_converter(output.dtype())` | `dst_single_tile_size` | not set |

Kernel-touch census (re-derived from the four kernel bodies, not transcribed from the brief):

| CB | touchers | roles | disposition |
|---|---|---|---|
| `c_0` | reader (`reserve_back`/`push_back`), compute (`wait_front`/`pop_front`) | 1 locked producer + 1 locked consumer | plain 1P+1C |
| `c_1` | reader (`reserve_back`/`push_back`), compute (`wait_front`/`pop_front`) | 1 locked producer + 1 locked consumer | plain 1P+1C |
| `c_2` | compute (`reserve_back`/`push_back`), writer (`wait_front`/`pop_front`) | 1 locked producer + 1 locked consumer | plain 1P+1C |

No self-loop, no multi-binding flag, no dead CB. Agrees with the brief. The compute-side
`copy_tile(cb, …)` / `pack_tile(0, cb)` calls are peeks on bindings compute already holds, and there
is no `get_write_ptr()` / `fifo_wr_ptr` raw write or cursor mutation anywhere in the four kernels.

### Semaphores

none — the legacy factory declares no `SemaphoreDescriptor` (`desc.semaphores` is never touched).

### Tensor accessors

All three are **Case 1** (accessed through `TensorAccessor`; no raw base-pointer arithmetic
anywhere), and all three constructions are two-argument (no page-size 3rd argument to drop).

| host site (file:line) | originating Tensor | RTA slot (host) | kernel construction |
|---|---|---|---|
| `gelu_backward_program_factory.cpp:85` | `grad_output` (input) | reader slot 0 (`:152`) | `reader_binary_interleaved_start_id.cpp:46` `TensorAccessor(src0_args, src0_addr)` |
| `gelu_backward_program_factory.cpp:86` | `input` (input) | reader slot 1 (`:152`) | `reader_binary_interleaved_start_id.cpp:53` `TensorAccessor(src1_args, src1_addr)` |
| `gelu_backward_program_factory.cpp:99` | `output` (return value) | writer slot 0 (`:156`) | `writer_unary_interleaved_start_id.cpp:31` `TensorAccessor(dst_args, dst_addr)` |

Pre-migration relaxation check (migration guide — `TensorParameter`): `grep -rn 'ArgConfig::Runtime'`
over the op directory and both borrowed dataflow kernels returns **zero hits**, so no
`dynamic_tensor_shape` / `match_padded_shape_only` relaxation is required despite the `eltwise`
family heads-up. Matches the brief ("TensorParameter relaxation: none"). Strict matching is kept.

### Work split

- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_tiles)` (`:39-40`), where
  `num_tiles = input.physical_volume() / TILE_HW` (`:34`) and
  `compute_with_storage_grid_size = device->compute_with_storage_grid_size()` (`:37`).
- `num_cores`, `all_cores`, `core_group_1`, `core_group_2`, `num_tiles_per_core_group_1`,
  `num_tiles_per_core_group_2`.
- The groups feed **only** the per-core RTA loop (`:140-159`) — `num_tiles_per_core` and the running
  `num_tiles_written`. No CTA and no kernel descriptor varies per group: all three
  `KernelDescriptor`s are declared once over `all_cores`.

### Shared kernels

Census run as `grep -rl <kernel-filename> ttnn/cpp/ttnn/operations/`, then each hit checked for a
factory that actually binds the path (same-named private copies and prose mentions discarded). A
locational `ls` of each original's directory shows **no `_metal2` sibling** for either file, so both
land on **rung 2 — create the fork beside the original**.

| kernel source | kind | `_metal2` fork exists? | rung | other binders (sunset list — *not* authorization to convert in place) |
|---|---|---|---|---|
| `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | borrowed | no | 2 (create) | `eltwise/unary_backward/gelu_bw`, `eltwise/unary_backward/tanh_bw`, `tests/ttnn/unit_tests/gtests/test_generic_op.cpp` |
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | borrowed | no | 2 (create) | very broadly shared — ~34 non-quasar factories (tilize, tilize_with_val_padding, reduction/generic, reduction/prod, transpose, slice, concat, copy, permute, reshape_on_device, bcast, typecast, embedding, examples, attn_matmul, nlp_concat_heads, kv_cache, `gelu_bw`, `tanh_bw`, …); full list in `METAL2_PREPORT_AUDIT.md` → Heads-ups |

Fork binding vocabulary is taken from **the kernel's own** vocabulary, not this factory's locals
(brief instruction — these names become the interface every later consumer inherits):
- reader fork: `dfb::in0` / `dfb::in1`, `tensor::src0` / `tensor::src1`,
  `args::block_or_width_sharded`, `args::num_tiles`, `args::start_id`, `args::block_height`,
  `args::block_width`, `args::num_cores_y`.
- writer fork: `dfb::out`, `tensor::dst`, `args::num_pages`, `args::start_id`.

The two **compute** kernels are op-owned with no other consumer (`eltwise/unary_backward/gelu_bw`
binds its own same-named private copies under its own `device/kernels/compute/`), so they convert
**in place** — no fork.

### Runtime kernel-source selection

The compute `KernelDescriptor` selects its source at runtime on **one** axis — `args.approximate`
(`:119`): `"tanh"` → `eltwise_bw_gelu_approx_tanh.cpp`, anything else → `eltwise_bw_gelu_poly.cpp`.
There is no second axis (no broadcast type, no `is_sfpu` fork, no row-major path — the device op
rejects non-TILE and sharded inputs outright). Both sources are bound by the same single
`KernelSpec` shape (identical bindings, identical CTAs, identical RTA schema), so both convert in
this change; the factory is not buildable with only one converted.

### Flags

none — every kernel file under the op directory is referenced (`eltwise_bw_gelu_approx_tanh.cpp` and
`eltwise_bw_gelu_poly.cpp` are the two selectable compute sources), and no descriptor type outside
the audit's scan appears.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — `create_program_artifacts`
  returning `ttnn::device_operation::ProgramArtifacts`. No disagreement with the audit's choice.
- **Custom `compute_program_hash`**: none — nothing to delete.
- **Implementation notes**:
  - `op_owned_tensors` stays defaulted-empty (audit: op-owned tensors none).
  - No pybind change is forced: `gelu_backward_nanobind.cpp` binds only the user-facing
    `ttnn.experimental.gelu_bw` entry point, not `create_descriptor`.
  - Two now-dead `#include <tt-metalium/program_descriptors.hpp>` lines
    (`device/gelu_backward_program_factory.hpp:8`, `device/gelu_backward_device_operation.hpp:11`)
    drop with the legacy CB/descriptor API, per kernel-side whitelist rule 1's both-sides sweep.

## Planned Spec Shape

1:1 with legacy throughout — three `KernelSpec`s, three `DataflowBufferSpec`s, three
`TensorParameter`s, one `WorkUnitSpec`, no semaphores.

- **KernelSpecs** (3, one per legacy `KernelDescriptor`):
  - `READER{"reader"}` → the reader `_metal2` fork.
    `dfb_bindings`: `GRAD_OUTPUT_DFB` PRODUCER as `"in0"`, `INPUT_DFB` PRODUCER as `"in1"`.
    `tensor_bindings`: `GRAD_OUTPUT` as `"src0"`, `INPUT` as `"src1"`.
    `compile_time_args`: `{{"block_or_width_sharded", 0}}`.
    `runtime_arg_schema.runtime_arg_names`: `{"num_tiles", "start_id", "block_height",
    "block_width", "num_cores_y"}`.
    `hw_config`: `create_reader_datamovement_config(device->arch())`.
    `compiler_options.opt_level`: left at Metal 2.0's `O2` default (= the resolved legacy level).
  - `WRITER{"writer"}` → the writer `_metal2` fork.
    `dfb_bindings`: `GRAD_IN_DFB` CONSUMER as `"out"`.
    `tensor_bindings`: `OUTPUT` as `"dst"`.
    `compile_time_args`: **none** (the legacy CTA 0 was the CB index; it becomes the DFB binding).
    `runtime_arg_schema.runtime_arg_names`: `{"num_pages", "start_id"}`.
    `hw_config`: `create_writer_datamovement_config(device->arch())`.
    `compiler_options.opt_level`: left at `O2`.
  - `COMPUTE{"compute"}` → the runtime-selected compute source.
    `dfb_bindings`: `GRAD_OUTPUT_DFB` CONSUMER as `"grad_out"`, `INPUT_DFB` CONSUMER as `"input"`,
    `GRAD_IN_DFB` PRODUCER as `"grad_in"`.
    `tensor_bindings`: none.
    `compile_time_args`: none.
    `runtime_arg_schema.runtime_arg_names`: `{"num_tiles"}`.
    `hw_config`: a hand-built `ComputeGen1Config` (Style B — see *Hardware configuration* below).
    `compiler_options.opt_level`: **explicit `KernelBuildOptLevel::O3`** (legacy `ComputeConfig`
    defaults to `O3`, Metal 2.0's type-agnostic `CompilerOptions` to `O2`).
- **DataflowBufferSpecs** (3, one per legacy `CBDescriptor`; `num_entries = 2` each, matching
  `total_size / page_size`; `tile_format_metadata` left unset because the legacy `tile` field was
  unset; no `borrowed_from`, no `alias_with`, no `allow_instance_multi_binding`):
  - `GRAD_OUTPUT_DFB{"grad_output"}` — `entry_size = src0_single_tile_size`,
    `data_format_metadata = src0_cb_data_format`.
  - `INPUT_DFB{"input"}` — `entry_size = src1_single_tile_size`,
    `data_format_metadata = src1_cb_data_format`.
  - `GRAD_IN_DFB{"grad_in"}` — `entry_size = dst_single_tile_size`,
    `data_format_metadata = dst_cb_data_format`.
- **SemaphoreSpecs**: none — legacy declares no semaphore.
- **TensorParameters** (3, one per distinct legacy `TensorAccessor` originating tensor, each from
  `<mesh_tensor>.tensor_spec()`, strict matching): `GRAD_OUTPUT{"grad_output"}`, `INPUT{"input"}`,
  `OUTPUT{"output"}`.
- **WorkUnitSpecs**: one — `{.name = "main", .kernels = {READER, WRITER, COMPUTE},
  .target_nodes = all_cores}`. All three legacy descriptors shared one `core_ranges`, so there is a
  single (kernels, nodes) pairing.
- **Op-owned tensors**: none.

### Hardware configuration

DM kernels resolve to the reader / writer defaults exactly, so both take the arch-agnostic TTNN
helper from `ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp`
(which reproduces the Gen1 triple byte-for-byte and supplies the Gen2 branch for free):

| kernel | legacy resolved `(processor, noc, noc_mode)` | Metal 2.0 |
|---|---|---|
| reader | `RISCV_1 / NOC_0 / DM_DEDICATED_NOC` (reader default) | `create_reader_datamovement_config(arch)` |
| writer | `RISCV_0 / NOC_1 / DM_DEDICATED_NOC` (writer default) | `create_writer_datamovement_config(arch)` |

The compute config is **Style B**: the legacy factory sets a Metal `ComputeConfigDescriptor` with
literal / computed field values and never resolves a TTNN `ComputeKernelConfig`
(no `init_device_compute_kernel_config` / `get_compute_kernel_config_args` anywhere in the op). So
the port builds a `ComputeGen1Config` directly and copies each field the op set; fields the op left
at their Metal defaults are left unset, because `ComputeGen1Config`'s defaults coincide with
`ComputeConfigDescriptor`'s. **Not** routed through `to_compute_hardware_config`, whose defaults are
the high-performance ones.

| legacy field | legacy value | Metal 2.0 field | ported value |
|---|---|---|---|
| `math_fidelity` | `HiFi4` | `fpu_math_fidelity` | `MathFidelity::HiFi4` (set explicitly, mirroring the op) |
| `math_approx_mode` | `false` (default) | `sfpu_precision_mode` | `Precision::Precise` — the `ComputeGen1Config` default; left unset |
| `fp32_dest_acc_en` | `(dst_cb_data_format ∈ {Float32, Int32, UInt32})` | `enable_32_bit_dest` | same expression, 1:1 |
| `dst_full_sync_en` | `false` (default) | `double_buffer_dest` | `!false = true` — the default; left unset |
| `bfp8_pack_precise` | `false` (default) | `bfp_pack_precision_mode` | `Precision::Approximate` — the default; left unset |
| `unpack_to_dest_mode` | `[c_0] = [c_1] = UnpackToDestFp32`, rest `Default` | `unpack_modes` | see below |

**`unpack_modes` — reindexed by DFB name, and gated on the DFB's data format.** The legacy vector's
`UnpackToDestFp32` entries sit at CB ids `c_0` / `c_1`, which reindex to `GRAD_OUTPUT_DFB` /
`INPUT_DFB`. The value translation is `UnpackToDestFp32 → UnpackMode::UnpackToDest` and
`Default → UnpackToSrc` (expressed by omitting the entry) — but a literal *unconditional*
transcription would **not** be behavior-preserving, and would be rejected outright by the Gen1
validator on the only dtype this op actually ships (BFLOAT16). Two facts drive the gate:

1. **Legacy ignores the setting unless the CB's format is `Float32`.**
   `tt_metal/jit_build/data_format.cpp:213-214` consults `unpack_to_dest_mode[i]` only inside
   `if (src_format == DataFormat::Float32 && …)`; for every other format the entry is inert and the
   CB takes the same `unpack_dst_format` as `Default` would give.
2. **The Metal 2.0 Gen1 validator rejects `UnpackToDest` on a ≤16-bit consumed DFB**
   (`tt_metal/impl/metal2_host_api/program_spec.cpp:1032-1039`, "bypassing the SrcA/B path (with no
   precision benefit) is not permitted because it leads to worse performance").

So the behavior-identical *and* legal translation emits the entry only where the legacy entry was
live:

```cpp
if (src0_cb_data_format == DataFormat::Float32) unpack_modes[GRAD_OUTPUT_DFB] = UnpackMode::UnpackToDest;
if (src1_cb_data_format == DataFormat::Float32) unpack_modes[INPUT_DFB]       = UnpackMode::UnpackToDest;
```

This also satisfies the validator's *required-entry* rule
(`program_spec.cpp:1051-1072`: a consumed `Float32` DFB under `enable_32_bit_dest = true` must carry
an explicit entry) in every reachable combination, because the device op forces
`output_dtype == input.dtype()` (`gelu_backward_device_operation.cpp:29-33`) and therefore
`INPUT_DFB` is `Float32` exactly when `enable_32_bit_dest` is true. `GRAD_IN_DFB` is only
*produced* by compute, so no entry is required or emitted for it.

One residual combination is a **behavior change, recorded in the report, not worked around**:
`grad_output` `Float32` while `input` / `output` is a ≤16-bit dtype makes `GRAD_OUTPUT_DFB`
`Float32` with `enable_32_bit_dest = false`, and the validator rejects that
(`program_spec.cpp:1024-1031`) where legacy silently accepted a genuinely misconfigured unpack
(32-bit datum into a 16-bit Dest). No test or documented dtype reaches it — the op documents
BFLOAT16 only and every test is BFLOAT16 — and inventing a config to get past it would change the
op's behavior, so the faithful mapping stands.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. `split_work_to_cores` produces two core *groups*, but
they feed only the per-core RTA loop; every `KernelDescriptor` is declared once over `all_cores`
with no per-group CTA, so there is nothing to preserve as a second `KernelSpec` and no
`WorkUnitSpec` split.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:85` + reader RTA slot 0 (`:152`) | `TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args)` + `src0_buffer` in the RTA list | `TensorParameter GRAD_OUTPUT` + `TensorBinding{GRAD_OUTPUT, "src0"}` on READER |
| factory `:86` + reader RTA slot 1 (`:152`) | `TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args)` + `src1_buffer` in the RTA list | `TensorParameter INPUT` + `TensorBinding{INPUT, "src1"}` on READER |
| factory `:99` + writer RTA slot 0 (`:156`) | `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` + `dst_buffer` in the RTA list | `TensorParameter OUTPUT` + `TensorBinding{OUTPUT, "dst"}` on WRITER |
| factory `:79-81` | `grad_output.buffer()` / `input.buffer()` / `output.buffer()` locals, existing only to feed the above | gone — the factory works from `MeshTensor` references |
| factory `:98` writer CTA slot 0 | `output_cb_index` (`CBIndex::c_2`) as a magic CB index | `DFBBinding{GRAD_IN_DFB, "out", CONSUMER}` on WRITER |
| reader kernel `:25-26` | `constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;` / `cb_id_in1 = tt::CBIndex::c_1;` (hardcoded magic indices) | `dfb::in0` / `dfb::in1` |
| reader kernel `:29-35` | `TensorAccessorArgs<1>()` + `TensorAccessorArgs<src0_args.next_compile_time_args_offset()>()` chain | gone — layout metadata rides the binding |
| reader kernel `:17-18` | `src0_addr` / `src1_addr` = `get_arg_val<uint32_t>(0/1)` | gone — the binding auto-injects the base address |
| reader kernel `:27` | `get_compile_time_arg_val(0)` (positional CTA) | `get_arg(args::block_or_width_sharded)` (named CTA) |
| reader kernel `:19-23` | `get_arg_val<uint32_t>(2..6)` (positional RTAs) | `get_arg(args::num_tiles / start_id / block_height / block_width / num_cores_y)` |
| reader kernel `:45,52` | `get_tile_size(cb_id_in0)` / `get_tile_size(cb_id_in1)` (cb-id free helper) | `dfb0.get_tile_size()` / `dfb1.get_tile_size()` (whitelist §A) |
| writer kernel `:15` | `constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);` (CB index via positional CTA) | `dfb::out` |
| writer kernel `:16` | `constexpr auto dst_args = TensorAccessorArgs<1>();` | gone |
| writer kernel `:11` | `dst_addr = get_arg_val<uint32_t>(0)` | gone — binding auto-injects it |
| writer kernel `:12-13` | `get_arg_val<uint32_t>(1/2)` (positional RTAs) | `get_arg(args::num_pages)` / `get_arg(args::start_id)` |
| writer kernel `:19` | `get_local_cb_interface(cb_id_out).fifo_page_size` | `dfb.get_entry_size()` (whitelist §B) |
| compute kernels `:26-28` / `:24-26` | `constexpr auto cb_grad_out = tt::CBIndex::c_0;` etc. (hardcoded magic indices) | `dfb::grad_out` / `dfb::input` / `dfb::grad_in` |
| compute kernels `:30-32` / `:28-30` | `CircularBuffer` objects + `#include "api/dataflow/circular_buffer.h"` (`:18` / `:19`) | `DataflowBuffer` objects + `#include "api/dataflow/dataflow_buffer.h"` |
| compute kernels `:24` / `:22` | `num_tiles = get_arg_val<uint32_t>(0)` (positional RTA) | `get_arg(args::num_tiles)` |

Nothing else in the legacy CTA / RTA lists survives: the reader keeps five named RTAs
(`num_tiles`, `start_id`, `block_height`, `block_width`, `num_cores_y`) and one named CTA, the
writer two named RTAs, the compute one named RTA. **`block_height` / `block_width` / `num_cores_y`
are kept even though this factory passes `0u, 0u` for two of them and the branch consuming all three
is compile-time dead here** (`block_or_width_sharded` CTA is hardcoded `0`): the kernel is shared, it
reads them unconditionally at the top of `kernel_main`, and dropping them is out of scope.

There are no semaphore-ID RTAs, no page-size 3rd-argument CTAs/RTAs, and no Case 2 (raw base
pointer) bindings anywhere.

## Applied Patterns

- [Caution: Porting a shared kernel](../shared/port_patterns.md#caution-porting-a-shared-kernel) —
  **rung 2** for both borrowed dataflow kernels: create
  `reader_binary_interleaved_start_id_metal2.cpp` and `writer_unary_interleaved_start_id_metal2.cpp`
  beside their originals, leave each original untouched apart from the pointer comment, and name the
  fork bindings from the kernel's own vocabulary.
- [Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers](../shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
  — the compute kernels' `unary_op_init_common`, `copy_tile`, `pack_tile` call sites take `dfb::…`
  handles directly via the `DFBBindingToken → uint32_t` conversion; no `.id`, no temp wrappers.
- [Pattern: Conditional / optional DFB bindings](../shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  — applied only in the *inherited* sense: both forks keep their pre-existing `IN0_SHARDED` /
  `IN1_SHARDED` / `OUT_SHARDED` / `BACKWARDS` `#ifdef`s, and every `tensor::` / `dfb::` reference the
  gates would remove already sits inside the matching `#ifdef`, so a future consumer that defines one
  of those and omits the corresponding binding still compiles. This factory defines none of them and
  binds everything unconditionally.
- [Anti-pattern: Demoting per-group CTA to RTA](../shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  — not applicable (no per-group CTA existed), noted so the reader knows it was checked.

## Deferred / Flagged

- **New finding (planning):** the recipe's `unpack_modes` translation table is stated
  unconditionally (`UnpackToDestFp32 → UnpackToDest`), but legacy only honours the entry for a
  `Float32`-formatted CB, and the Gen1 validator *rejects* the unconditional transcription for a
  ≤16-bit consumed DFB. Resolved as the format-gated emission documented under *Hardware
  configuration*; carried to the port report as a doc gap plus one residual, untested dtype
  combination that now hard-errors instead of silently misconfiguring.
- Nothing else — no structural issue the audit missed, no feature gate outside the audit's
  Appendix A, no construct requiring a legacy workaround.

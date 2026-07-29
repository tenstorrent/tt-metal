# Port Plan — `ttnn/cpp/ttnn/operations/reduction/accumulation/ema`

Port plan for `EmaDeviceOperation` / `EmaProgramFactory`, ported from the legacy `ProgramDescriptor`
API to Metal 2.0. Written during the inventory and planning steps; committed alongside the port for
review.

**Inputs:** `METAL2_PORT_BRIEF.md` (audit GREEN, brief issued), `METAL2_PREPORT_AUDIT.md`.

## Legacy Inventory

> Every `file:line` in this section refers to the **pre-port** revision of the file (the parent of the
> commit this plan lands in). The links are clickable but resolve against the ported file, so read the
> quoted legacy content as authoritative, not the line the link lands on.

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — `EmaProgramFactory::create_descriptor` returns
  `tt::tt_metal::ProgramDescriptor` ([ema_device_operation.hpp:24-27](device/ema_device_operation.hpp#L24-L27),
  body [ema_program_factory.cpp:21-196](device/ema_program_factory.cpp#L21-L196)).
- Variants: single. `program_factory_t = std::variant<EmaProgramFactory>`
  ([ema_device_operation.hpp:30](device/ema_device_operation.hpp#L30)).
- Custom `compute_program_hash`: none — already the default reflection-based hash. Nothing to delete.
- No `get_dynamic_runtime_args`, no `override_runtime_arguments`, no pybind `create_descriptor`
  (verified by grep over the op directory; the nanobind file exposes only `ttnn::ema`,
  [ema_nanobind.cpp:70-80](ema_nanobind.cpp#L70-L80)).

**No configuration branch.** One core-range set, one kernel triple, three CBs. No sharded/interleaved
fork, no split reader, no multicast, no runtime kernel-source selection. Input shape and requested grid
change CTA/RTA *values* only ([ema_program_factory.cpp:30-64](device/ema_program_factory.cpp#L30-L64)),
never program structure. So there is exactly one instantiation shape to inventory and classify.

### Kernels

Three `KernelDescriptor`s, each with a distinct `kernel_source`, all over the single `all_cores` range.
CTA slot positions are read from the host factory's emission order.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `ema/kernels/dataflow/ema_reader.cpp` ([:143](device/ema_program_factory.cpp#L143)) | `all_cores` ([:145](device/ema_program_factory.cpp#L145)) | `[0]` = `total_tiles_per_core`; `[1…]` = `TensorAccessorArgs(input)` ([:124-125](device/ema_program_factory.cpp#L124-L125)) | none | per-node, 2 slots: `[0]` = `input` (`MeshTensor` binding overload), `[1]` = `src_start_tile` ([:184](device/ema_program_factory.cpp#L184)) | none | none | `DataMovementConfigDescriptor{.processor = RISCV_0, .noc = preferred_noc_for_dram_read(arch)}`, `noc_mode` defaulted to `DM_DEDICATED_NOC` ([:147-150](device/ema_program_factory.cpp#L147-L150)) |
| writer | `ema/kernels/dataflow/ema_writer.cpp` ([:153](device/ema_program_factory.cpp#L153)) | `all_cores` ([:155](device/ema_program_factory.cpp#L155)) | `[0]` = `total_tiles_per_core`; `[1…]` = `TensorAccessorArgs(output)` ([:127-128](device/ema_program_factory.cpp#L127-L128)) | none | per-node, 2 slots: `[0]` = `output` (`MeshTensor` binding overload), `[1]` = `dst_start_tile` ([:185](device/ema_program_factory.cpp#L185)) | none | none | `DataMovementConfigDescriptor{.processor = RISCV_1, .noc = preferred_noc_for_dram_write(arch)}`, `noc_mode` defaulted to `DM_DEDICATED_NOC` ([:157-160](device/ema_program_factory.cpp#L157-L160)) |
| compute | `ema/kernels/compute/ema_compute.cpp` ([:166](device/ema_program_factory.cpp#L166)) | `all_cores` ([:168](device/ema_program_factory.cpp#L168)) | `[0]` = `total_batch_channel_tiles_per_core`, `[1]` = `tiles_per_channel`, `[2]` = `alpha_bits`, `[3]` = `beta_bits` ([:130-135](device/ema_program_factory.cpp#L130-L135)) | none | none | none | none | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` from `get_compute_kernel_config_args` ([:162-175](device/ema_program_factory.cpp#L162-L175)); `unpack_to_dest_mode` and `bfp8_pack_precise` left at their defaults |

Resolved DM triples (the *values*, which is what the port must reproduce): `preferred_noc_for_dram_read`
returns `NOC_0` and `preferred_noc_for_dram_write` returns `NOC_1` on every arch
([kernel_types.hpp:134-146](../../../../../../../tt_metal/api/tt-metalium/kernel_types.hpp#L134-L146)), so:

| kernel | processor | noc | noc_mode | matches a Metal 2.0 default? |
|---|---|---|---|---|
| reader | `RISCV_0` | `NOC_0` | `DM_DEDICATED_NOC` | **no** — reader default is `RISCV_1`/`NOC_0`, writer default is `RISCV_0`/`NOC_1` |
| writer | `RISCV_1` | `NOC_1` | `DM_DEDICATED_NOC` | **no** — same reason, mirrored |

This op deliberately swaps the conventional RISC assignment (reader on `RISCV_0`, writer on `RISCV_1`)
while keeping the conventional NOC assignment. Both triples are therefore **custom**, and the port must
build raw `DataMovementGen1Config`s rather than reach for `create_reader_datamovement_config` /
`create_writer_datamovement_config`. The RISCs are distinct and, under `DM_DEDICATED_NOC`, the NOCs are
distinct, so the Metal 2.0 spec validator's Gen1 node invariants are satisfied.

### CBs

Three `CBDescriptor`s, each a single-element `format_descriptors` (no aliasing), each over `all_cores`.
None sets `.tile`, `.buffer`, `.tensor`, `.address_offset`, or `.global_circular_buffer`.

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (`src_cb_index`) | `src_tile_size * ema_buffer_depth` (= `* 2`) ([:88](device/ema_program_factory.cpp#L88), CB at [:92-100](device/ema_program_factory.cpp#L92-L100)) | `all_cores` | `src_data_format` = `datatype_to_dataformat_converter(input.dtype())` | `src_tile_size` | unset |
| `c_1` (`dst_cb_index`) | `dst_tile_size * ema_buffer_depth` ([:89](device/ema_program_factory.cpp#L89), CB at [:102-110](device/ema_program_factory.cpp#L102-L110)) | `all_cores` | `dst_data_format` | `dst_tile_size` | unset |
| `c_2` (`prev_cb_index`) | `src_tile_size` (one tile) ([:90](device/ema_program_factory.cpp#L90), CB at [:112-120](device/ema_program_factory.cpp#L112-L120)) | `all_cores` | `src_data_format` | `src_tile_size` | unset |

### Semaphores

none — `desc.semaphores` is never populated, and a case-insensitive grep for `semaphore` over the whole
op directory returns zero hits.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| [ema_program_factory.cpp:125](device/ema_program_factory.cpp#L125) (`TensorAccessorArgs(input).append_to`) | `input` (`tensor_args.input.mesh_tensor()`) | reader RTA slot 0 ([:184](device/ema_program_factory.cpp#L184)), read at [ema_reader.cpp:21](kernels/dataflow/ema_reader.cpp#L21), consumed at [ema_reader.cpp:34](kernels/dataflow/ema_reader.cpp#L34) |
| [ema_program_factory.cpp:128](device/ema_program_factory.cpp#L128) (`TensorAccessorArgs(output).append_to`) | `output` (`tensor_return_value.mesh_tensor()`) | writer RTA slot 0 ([:185](device/ema_program_factory.cpp#L185)), read at [ema_writer.cpp:21](kernels/dataflow/ema_writer.cpp#L21), consumed at [ema_writer.cpp:34](kernels/dataflow/ema_writer.cpp#L34) |

Both are **Case 1** bindings (every access goes through the `TensorAccessor`). No kernel does raw address
arithmetic, so no `get_bank_base_address` bridge is needed anywhere in this op. Both accessor
constructions are two-argument — no page-size third argument to drop.

### Work split

- Driver: `get_max_cores_divisible_by_tiles_per_core_tiles(total_batch_channel_tiles, num_cores_available, /*request_even=*/false)`
  ([ema_program_factory.cpp:48-49](device/ema_program_factory.cpp#L48-L49)).
- num_cores: returned by that call; nodes are `grid_to_cores(num_cores, grid_size.x, grid_size.y, false)`
  ([:52](device/ema_program_factory.cpp#L52)).
- core_group_1: `all_cores`, count_per_core: `total_batch_channel_tiles_per_core` (hence
  `total_tiles_per_core = total_batch_channel_tiles_per_core * tiles_per_channel`, [:64](device/ema_program_factory.cpp#L64)).
- core_group_2: **none.** This driver picks a core count that divides the tile count *exactly*, so there
  is a single group with a uniform per-core count — not the `split_work_to_cores` two-group shape. Every
  node gets identical CTAs; only the `*_start_tile` RTA varies per node ([:180-189](device/ema_program_factory.cpp#L180-L189)).

### Shared kernels

none. This op owns all three kernel files and is their only binder — `grep -rl <filename> ttnn/cpp/ttnn/operations/`
returns exactly one code hit each, `device/ema_program_factory.cpp` (the other two hits are the
`METAL2_*.md` artifacts in this directory). A directory listing of
`kernels/dataflow/` and `kernels/compute/` shows no `_metal2` sibling beside any of them, and
`find ttnn/cpp/ttnn/operations/reduction -name '*_metal2*'` is empty. So the three kernels convert
**in place**: no fork to reuse, no fork to create, no pointer comment to leave, no remaining consumer to
coordinate with.

The one out-of-directory kernel include is `../../../device/kernels/accumulation_common.hpp` — the
in-family constants header shared with `cumsum` / `cumprod`. The EMA kernels use exactly one constant
from it (`ONE_TILE`) and call no function; it declares no CB id, semaphore, or accessor in any
signature, so no named Metal 2.0 handle has to bridge into it. It is **not modified** by this port.

### Flags

- The kernels live at `ema/kernels/`, i.e. at the op root, **not** under `ema/device/kernels/` as the
  sibling `accumulation` op does. Only a place where a path guess goes wrong.
- No unreferenced kernel files in the directory: all three files are referenced by the factory.
- No descriptor type outside the audit's scan appears in this factory.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept` (spelled `ProgramSpecFactoryConcept` in
  code today — [operation_concepts.hpp:119-121](../../../../../../api/ttnn/operation_concepts.hpp#L119-L121)),
  without op-owned tensors.
- **Custom `compute_program_hash`**: none — nothing to delete.
- **Implementation notes**:
  - The factory method becomes
    `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&)`,
    replacing `create_descriptor` in [ema_device_operation.hpp](device/ema_device_operation.hpp). The
    `#include <tt-metalium/program_descriptors.hpp>` in that header goes away with it.
  - Op-owned tensors: none. The output is allocated through the ordinary TTNN path
    (`create_output_tensors` → `create_device_tensor`, [ema_device_operation.cpp:91-97](device/ema_device_operation.cpp#L91-L97)).
  - No device-op-class edit beyond the factory declaration itself: no custom hash to delete, no pybind
    entry point vanishing, no pybind-hook-only factory parameter.
  - Tensor-arg matching stays **strict**. No relaxation is applied (the audit found none required).

## Planned Spec Shape

1:1 with legacy throughout — one instantiation shape, no config branch.

- **KernelSpecs** (3, one per legacy `KernelDescriptor`, one instance each):
  - `READER{"reader"}` — source `ema/kernels/dataflow/ema_reader.cpp`, custom `DataMovementGen1Config`,
    1 named CTA, 1 named RTA, 1 DFB binding (SRC PRODUCER), 1 tensor binding (INPUT).
  - `WRITER{"writer"}` — source `ema/kernels/dataflow/ema_writer.cpp`, custom `DataMovementGen1Config`,
    1 named CTA, 1 named RTA, 1 DFB binding (DST CONSUMER), 1 tensor binding (OUTPUT).
  - `COMPUTE{"compute"}` — source `ema/kernels/compute/ema_compute.cpp`, `ComputeGen1Config` from
    `to_compute_hardware_config`, 4 named CTAs, no RTAs (so no `KernelRunArgs` entry), 4 DFB bindings
    (SRC CONSUMER, DST PRODUCER, PREV PRODUCER + PREV CONSUMER — the self-loop).
- **DataflowBufferSpecs** (3, one per legacy `CBDescriptor`; no aliasing, no borrowed memory, no
  `tile_format_metadata` since no legacy CB set `.tile`):

  | unique_id | accessor_name(s) | entry_size | num_entries | data_format_metadata | legacy CB |
  |---|---|---|---|---|---|
  | `SRC{"src"}` | reader `"src"`, compute `"src"` | `src_tile_size` | `ema_buffer_depth` (2) | `src_data_format` | `c_0` |
  | `DST{"dst"}` | compute `"dst"`, writer `"dst"` | `dst_tile_size` | `ema_buffer_depth` (2) | `dst_data_format` | `c_1` |
  | `PREV{"prev"}` | compute `"trp"` (both endpoints) | `src_tile_size` | 1 | `src_data_format` | `c_2` |

  `num_entries` is `total_size / page_size` in each case, so the L1 footprint is byte-identical to legacy.
- **SemaphoreSpecs**: none — the op uses no semaphores.
- **TensorParameters** (2): `INPUT{"input"}` (`input.tensor_spec()`), `OUTPUT{"output"}`
  (`output.tensor_spec()`). One `TensorBinding` each, on the reader and writer respectively.
- **WorkUnitSpecs** (1): `{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}`.
  All three kernels run on every node, which is what the legacy `core_ranges = all_cores` on all three
  descriptors says, and it satisfies the local-DFB invariant (a DFB's producer and consumer share
  identical work-unit membership).
- **Op-owned tensors**: none.

### Resource naming

The host and the kernels each already have a word for every resource, and for two resources those words
differ. The Metal 2.0 binding model has two independent name slots — the program-scope `unique_id` and
the per-kernel `accessor_name` — so each side keeps the vocabulary it already uses and nothing is
renamed:

| resource | host word (legacy) | kernel word (legacy) | `unique_id` | `accessor_name` |
|---|---|---|---|---|
| `c_0` | `src_cb_index` | `src_cb_idx` / `dfb_src` | `SRC{"src"}` | `"src"` |
| `c_1` | `dst_cb_index` | `dst_cb_idx` / `dfb_dst` | `DST{"dst"}` | `"dst"` |
| `c_2` | `prev_cb_index` | `trp_cb_idx` / `dfb_trp` | `PREV{"prev"}` | `"trp"` |
| input tensor | `input` | `src_accessor` | `INPUT{"input"}` | `"src"` |
| output tensor | `output` | `dst_accessor` | `OUTPUT{"output"}` | `"dst"` |

For `c_2` this matters: the audit records the host name `prev_cb_index` as misleading (the buffer stages
one tile through SRAM for a second transpose; it does not hold the previous EMA output, which lives in an
SFPU register) and the kernel name `trp` as the accurate one. Keeping `unique_id = "prev"` and
`accessor_name = "trp"` leaves the mismatch exactly where the audit found it — visible for the ops team,
not silently renamed by the port, and with the accurate word kept in the kernel code.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. The work-split driver
(`get_max_cores_divisible_by_tiles_per_core_tiles`, [ema_program_factory.cpp:48-49](device/ema_program_factory.cpp#L48-L49))
picks a core count that divides the tile count exactly, so there is a single core group with a uniform
per-node count and no per-group CTA to preserve. Each legacy `KernelDescriptor` maps to exactly one
`KernelSpec`, all three in the single `WorkUnitSpec`.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| [ema_program_factory.cpp:125](device/ema_program_factory.cpp#L125) | `TensorAccessorArgs(input).append_to(reader_compile_args)` | `TensorBinding{INPUT, "src"}` on `READER` + `TensorParameter{INPUT}` |
| [ema_reader.cpp:17](kernels/dataflow/ema_reader.cpp#L17) | `constexpr auto src_args = TensorAccessorArgs<1>()` | folded into `TensorAccessor(tensor::src)` |
| [ema_program_factory.cpp:184](device/ema_program_factory.cpp#L184) RTA slot 0 | `emplace_runtime_args(core, {input, …})` — `MeshTensor` binding overload | `TensorArgument` in `ProgramRunArgs::tensor_args` |
| [ema_reader.cpp:21](kernels/dataflow/ema_reader.cpp#L21) | `const uint32_t src_base_addr = get_arg_val<uint32_t>(0)` | gone — the binding auto-injects the per-enqueue base address |
| [ema_program_factory.cpp:128](device/ema_program_factory.cpp#L128) | `TensorAccessorArgs(output).append_to(writer_compile_args)` | `TensorBinding{OUTPUT, "dst"}` on `WRITER` + `TensorParameter{OUTPUT}` |
| [ema_writer.cpp:17](kernels/dataflow/ema_writer.cpp#L17) | `constexpr auto dst_args = TensorAccessorArgs<1>()` | folded into `TensorAccessor(tensor::dst)` |
| [ema_program_factory.cpp:185](device/ema_program_factory.cpp#L185) RTA slot 0 | `emplace_runtime_args(core, {output, …})` | `TensorArgument` in `ProgramRunArgs::tensor_args` |
| [ema_writer.cpp:21](kernels/dataflow/ema_writer.cpp#L21) | `const uint32_t dst_base_addr = get_arg_val<uint32_t>(0)` | gone |
| [ema_program_factory.cpp:78-80](device/ema_program_factory.cpp#L78-L80) | `constexpr auto src_cb_index = tt::CBIndex::c_0;` (and `c_1`, `c_2`) — magic CB indices | `DFBSpecName` + `DFBBinding`; no CB index appears on the host |
| [ema_reader.cpp:26](kernels/dataflow/ema_reader.cpp#L26), [ema_writer.cpp:26](kernels/dataflow/ema_writer.cpp#L26), [ema_compute.cpp:78-80](kernels/compute/ema_compute.cpp#L78-L80) | kernel-side `constexpr auto *_cb_idx = tt::CBIndex::c_N` | `dfb::src` / `dfb::dst` / `dfb::trp` |
| [ema_program_factory.cpp:124](device/ema_program_factory.cpp#L124) CTA slot 0 | positional `{total_tiles_per_core}` | named CTA `{"total_tiles_per_core", total_tiles_per_core}` |
| [ema_program_factory.cpp:127](device/ema_program_factory.cpp#L127) CTA slot 0 | positional `{total_tiles_per_core}` | named CTA `{"total_tiles_per_core", total_tiles_per_core}` |
| [ema_program_factory.cpp:130-135](device/ema_program_factory.cpp#L130-L135) CTA slots 0-3 | positional `{total_batch_channel_tiles_per_core, tiles_per_channel, alpha_bits, beta_bits}` | named CTAs `{"total_batches_per_core", …}`, `{"tiles_per_channel", …}`, `{"alpha_bits", …}`, `{"beta_bits", …}` (see the naming note below) |
| [ema_reader.cpp:16](kernels/dataflow/ema_reader.cpp#L16), [ema_writer.cpp:16](kernels/dataflow/ema_writer.cpp#L16), [ema_compute.cpp:71-74](kernels/compute/ema_compute.cpp#L71-L74) | `get_compile_time_arg_val(N)` | `get_arg(args::<name>)` |
| [ema_reader.cpp:22](kernels/dataflow/ema_reader.cpp#L22) RTA slot 1 | `get_arg_val<uint32_t>(1)` | named RTA `get_arg(args::src_start_tile)` |
| [ema_writer.cpp:22](kernels/dataflow/ema_writer.cpp#L22) RTA slot 1 | `get_arg_val<uint32_t>(1)` | named RTA `get_arg(args::dst_start_tile)` |
| [ema_reader.cpp:30](kernels/dataflow/ema_reader.cpp#L30), [ema_writer.cpp:30](kernels/dataflow/ema_writer.cpp#L30) | `get_tile_size(src_cb_idx)` / `get_tile_size(dst_cb_idx)` — CB-id free function | `dfb_src.get_tile_size()` / `dfb_dst.get_tile_size()` (whitelist rule 7) |

Not present in this op, so nothing to drop: page-size third-argument CTAs/RTAs (both accessors are
two-argument), semaphore-ID RTAs (no semaphores), `tensor.buffer()->address()` (never called).

**Named-CTA naming for compute slot 0.** A named CTA forces one name shared by host and kernel, and this
slot is the one place the two sides already disagree: the kernel reads it as `total_batches_per_core`
([ema_compute.cpp:71](kernels/compute/ema_compute.cpp#L71)) while the host passes
`total_batch_channel_tiles_per_core` ([ema_program_factory.cpp:131](device/ema_program_factory.cpp#L131)),
which is `num_batches * num_channel_tiles` split across nodes — so the kernel-side name is off by the
channel-tile factor (audit anomaly 2). The port uses the **kernel's** name, `total_batches_per_core`,
per the recipe's rule that an argument name matches the variable it is assigned to. That renames nothing
on either side and leaves the mismatch visible on one host line for the ops team, rather than the port
quietly resolving a naming question that is theirs.

## Applied Patterns

- [Self-loop DFB binding](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  `PREV` on the compute `KernelSpec`, bound both `PRODUCER` and `CONSUMER` under one shared
  `accessor_name` (`"trp"`), so the kernel keeps a single `DataflowBuffer` object driving both FIFO ends.
  Re-derived census (not transcribed from the brief): on a node, `c_2` has exactly **one** toucher — the
  compute kernel — which drives `reserve_back`/`push_back`
  ([ema_compute.cpp:109](kernels/compute/ema_compute.cpp#L109), [:113](kernels/compute/ema_compute.cpp#L113))
  *and* `wait_front`/`pop_front` ([:116](kernels/compute/ema_compute.cpp#L116), [:120](kernels/compute/ema_compute.cpp#L120))
  to round-trip a tile through SRAM for a second transpose. One toucher → self-loop; there is no second
  kernel to assign a role to. Agrees with the brief.
- [Pass DFB handles directly to LLKs](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  `compute_kernel_hw_startup(dfb::src, dfb::dst)` ([ema_compute.cpp:93](kernels/compute/ema_compute.cpp#L93)),
  `transpose_init(dfb::src)` ([:95](kernels/compute/ema_compute.cpp#L95)),
  `transpose_tile(dfb::src | dfb::trp, …)` ([:104](kernels/compute/ema_compute.cpp#L104), [:118](kernels/compute/ema_compute.cpp#L118)),
  `pack_tile(…, dfb::trp | dfb::dst)` ([:111](kernels/compute/ema_compute.cpp#L111), [:124](kernels/compute/ema_compute.cpp#L124)).
  These compute LLKs take a `uint32_t` CB id and have no `DataflowBuffer` method equivalent; the
  `DFBAccessor → uint32_t` implicit conversion is the sanctioned bridge, and the call shapes are
  otherwise unchanged.
- **Custom DM hardware configs**, per the recipe's
  [Hardware configuration](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#data-movement-kernels)
  section: this op's resolved triples match neither the reader nor the writer default (see the Kernels
  table above), so both DM kernels get a raw `DataMovementGen1Config` with every field copied verbatim,
  including the `preferred_noc_for_dram_{read,write}(device->arch())` calls that produce the NOC values.
  Reaching for the arch-agnostic helpers here would silently swap this op's RISC assignment.
- **Compute hardware config via the TTNN helper (Style A)**: the legacy factory resolves a TTNN
  `ComputeKernelConfig` and feeds `get_compute_kernel_config_args`
  ([ema_program_factory.cpp:162-163](device/ema_program_factory.cpp#L162-L163)), so the port uses
  `ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config)`, which
  applies the same four knobs with the two representation changes (`math_approx_mode` bool → `Precision`,
  `dst_full_sync_en` → inverted `double_buffer_dest`). The two Metal-only fields stay at their defaults
  because legacy left both at theirs: `unpack_to_dest_mode` was empty and `bfp8_pack_precise` was
  `false`.
- **No `unpack_modes` entry is required.** `enable_32_bit_dest` is `true` by default for this op
  (`init_device_compute_kernel_config(..., /*default_fp32_acc=*/true)`, [ema.cpp:20-25](ema.cpp#L20-L25)),
  but the newly-required-entry rule is Float32-only, and this op validates both input and output to
  `DataType::BFLOAT16` ([ema_device_operation.cpp:20-21](device/ema_device_operation.cpp#L20-L21),
  [:40-42](device/ema_device_operation.cpp#L40-L42)), so every DFB carries `Float16_b` and no consumed
  DFB is Float32. Mirroring legacy means leaving `unpack_modes` empty (legacy's empty
  `unpack_to_dest_mode` == `UnpackToSrc` everywhere).

Not applied, and why: no conditional/optional bindings (nothing in this factory is conditional), no
aliased DFBs (every legacy `format_descriptors` has one element), no same-FIFO aliasing (no kernel
aliases one CB index to a second name), no multi-binding flag (no CB has ≥3 touchers or two kernels
locked to the same FIFO role), no dead-CB drop (all three CBs are touched), no multi-variant branch
(single instantiation shape), no varargs (every CTA/RTA is a distinct field read once).

## Deferred / Flagged

- **New findings during planning:** none that change the port. Planning confirmed every item the brief
  carried, and the two dispositions the porter is expected to re-derive (the `c_2` self-loop census and
  the endpoint assignment for `c_0` / `c_1`) came out matching the brief.
- The one planning-time judgement the brief did not pre-decide is the **compute CTA slot 0 name** (see
  the Dropped Plumbing note). Recorded here because it is the only place the port had to choose between
  two pre-existing names rather than carry one across.
- The audit's remaining misc anomalies (host name `prev_cb_index`, the docstring's `output_0 = input_0`
  initial condition, the unenforced interleaved-only documentation, `alpha` validated for NaN only, the
  unused `CB_IN`/`CB_OUT`/`CB_ACC` constants pulled in from `accumulation_common.hpp`) are **not touched**
  by this port and are carried forward to `METAL2_PORT_REPORT.md` for the ops team.

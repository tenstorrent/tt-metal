# Port Plan — `moreh_nll_loss_unreduced_backward`

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward`, ported from
`ProgramDescriptor` (`ProgramDescriptorFactoryConcept`) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Line references below are to the **pre-port** revision (`git merge-base origin/main HEAD` =
`f38fbebd760`, op unchanged since `047fecfec7f`).

## Legacy Inventory

### Legacy factory shape

- Concept: **`ProgramDescriptorFactoryConcept`** — `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`.
- Variants: **single** factory (`Factory`, `program_factory_t = std::variant<Factory>`,
  `..._device_operation.hpp:35-42`). The factory methods live **in the `program_factory_t` variant**,
  not directly on the device-operation struct — so the direct-descriptor exception does **not** apply
  and the device-operation class stays untouched apart from the method-swap in the `Factory` struct.
- **Three rank-dispatched configs of that one factory**, not three factories.
  `Factory::create_descriptor` (`..._program_factory.cpp:453`) branches on
  `input_grad.logical_shape().rank()` into three free functions in the same file:
  `..._impl_2d` (`:46`), `..._impl_3d` (`:182`), `..._impl_4d` (`:316`). One port converts all three.
- **No compute kernel.** Every program is reader + writer only; the readers compute `input_grad`
  themselves with `CoreLocalMem` scalar writes into the output CB.
- Custom `compute_program_hash`: **none** — default reflection-based hash. No backdoor
  `attribute_values` / `to_hash` either. Nothing to preserve; nothing to touch.
- `override_runtime_arguments`: **absent** → target concept is the **base** `ProgramSpecFactoryConcept`.
- `get_dynamic_runtime_args`: absent. Op-owned tensors: none. Semaphores: none.
- Second config axis, orthogonal to rank: **`WEIGHT`** (`weight_tensor` is `std::optional`).
  Dtypes are pinned by validation — `target` INT32, `output_grad` / `weight` / `input_grad` BFLOAT16.

### Kernels

Two `KernelDescriptor`s per config (one reader, one writer). `core_ranges` is `all_cores` for both,
in all three configs. `source_type = FILE_PATH`. `opt_level`: **not set anywhere in the op**
(`grep -n opt_level` over the op directory returns zero hits) — both kernels are DM, so the resolved
legacy level is `O2`.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs (per node) | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (2d, `:122`) | `device/kernels/reader_moreh_nll_loss_unreduced_backward_2d.cpp` | `all_cores` | 3 × `TensorAccessorArgs` block, appended in order `target`, `output_grad`, `weight` (`:97-99`) — 6 words total, `(2,4096),(2,2048),(0,0)` | none | `target_buf`, `output_grad_buf`, `weight_buf`, `ignore_index`, `units_per_core`, `tile_offset`, `Nt`, `channel_size`, `Ct` (`:157-169`) | none | `WEIGHT=1` if `weight_has_value`; `FP32_DEST_ACC_EN=1` if `fp32_dest_acc_en` (`:107-113`) | unset → **O2** | `ReaderConfigDescriptor{}` (`:128`) |
| reader (3d, `:256`) | `..._3d.cpp` | `all_cores` | same 3 blocks (`:231-233`) | none | …same first six…, `channel_size`, `Ct`, `Wt` (`:291-303`) | none | same | unset → **O2** | `ReaderConfigDescriptor{}` |
| reader (4d, `:393`) | `..._4d.cpp` | `all_cores` | same 3 blocks (`:368-370`) | none | …same first six…, `num_inner_tile`, `channel_size`, `Ct` (`:428-440`) | none | same | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer (all 3 configs) | `device/kernels/writer_moreh_nll_loss_unreduced_backward.cpp` | `all_cores` | 1 × `TensorAccessorArgs(input_grad)` block (`:102` / `:236` / `:373`) | none | `input_grad_buf`, `units_per_core`, `tile_offset` | none | **none** — `writer_defines` is declared and moved in **empty** in all three configs | unset → **O2** | `WriterConfigDescriptor{}` |

`ReaderConfigDescriptor` / `WriterConfigDescriptor` are empty structs
(`tt_metal/api/tt-metalium/program_descriptors.hpp:92-93`), so the resolved DM triples are exactly the
reader / writer defaults: reader `(RISCV_1, NOC_0, DM_DEDICATED_NOC)`, writer `(RISCV_0, NOC_1,
DM_DEDICATED_NOC)`. No custom triple anywhere; `noc_mode` is default on both.

### CBs

All built by the local `push_cb` helper (`:23-42`), which skips creation entirely when `num_tiles == 0`
and sets `total_size = num_tiles * tile_size(data_format)`, `page_size = tile_size(data_format)`,
`core_ranges = all_cores`. **`CBFormatDescriptor::tile` is never set** → `tile_format_metadata` stays
`nullopt` in the port. No `.buffer`, no `.global_circular_buffer`, no `address_offset`, single-element
`format_descriptors` throughout (so: no borrowed memory, no GlobalCB, no aliasing).

`data_format = datatype_to_dataformat_converter(input_grad.dtype())`.

| index | role | total_size | core_ranges | data_format | page_size | tile (if set) | configs |
|---|---|---|---|---|---|---|---|
| `c_0` | target | `1 × tile_size(Int32)` | `all_cores` | `Int32` | `tile_size(Int32)` | — | all |
| `c_1` | output_grad | 2d: `Nt × ts`; 3d/4d: `1 × ts` | `all_cores` | `data_format` | `tile_size(data_format)` | — | all |
| `c_2` | weight | `Ct × ts` when `weight_has_value`, else **not created** (`push_cb` early-returns on 0) | `all_cores` | `data_format` | `tile_size(data_format)` | — | `WEIGHT` only |
| `c_7` | weight scratch | `1 × ts`, inside `if (weight_has_value)` (`:90` / `:226` / `:363`) | `all_cores` | `data_format` | `tile_size(data_format)` | — | `WEIGHT` only |
| `c_8` | output_grad scratch | `1 × ts`, unconditional but **2d only** (`:93`) | `all_cores` | `data_format` | `tile_size(data_format)` | — | **2d only** |
| `c_16` | input_grad | `1 × ts` | `all_cores` | `data_format` | `tile_size(data_format)` | — | all |

Per-config `Ct` / `Nt`: 2d `Ct = div_up(channel_size, TILE_WIDTH)`, `Nt = div_up(N, TILE_WIDTH)`;
3d `Ct = channel_size / TILE_HEIGHT`; 4d `Ct = div_up(channel_size, TILE_WIDTH)`.

**Kernel-touch census (re-derived from the kernel bodies, not transcribed from the brief).**
Each node runs exactly two kernels — one reader, one writer. Neither is instantiated twice, so there
is no dual-instance work-split anywhere in this op.

| CB | distinct touchers on a node | tags | disposition |
|---|---|---|---|
| `c_0` | reader only — `read_tile` (donor `reserve_back`+`push_back`), `wait_front`/`pop_front`, `get_read_ptr` peek | 1 toucher, locked both roles | **self-loop** |
| `c_1` | reader only — 2d: `read_line` produces `Nt`, `wait_front(Nt)` + peek, never popped. 3d/4d: `read_tile` per iteration, `wait_front`/`pop_front` + peek | 1 toucher | **self-loop** |
| `c_2` | reader only — `read_line` produces `Ct`, `wait_front(Ct)` + peek, never popped | 1 toucher | **self-loop** |
| `c_7` | reader only, inside the donor's `read_line` — NoC-read destination + `get_write_ptr()`, **no FIFO ops at all** | 1 toucher, role-free | **self-loop** |
| `c_8` | reader only, same sync-free shape | 1 toucher, role-free | **self-loop** |
| `c_16` | reader `reserve_back` + raw `get_write_ptr()` write + `push_back`; writer `wait_front` + `noc.async_write` + `pop_front` | 2 touchers, one locked to each role | **1P + 1C** |

Two consumers legitimately never pop (`c_1` on 2d, `c_2` everywhere) — the whole row / line is held
for the loop. A held single-toucher CB is still a self-loop; the missing `pop` is not a missing endpoint.
**No multi-binding, and no dead CB** — the census agrees with the brief in every row, so nothing is
noted as a disagreement.

### Semaphores

**none** — the op uses no semaphores of any kind.

### Tensor accessors

Four distinct originating tensors, ten construction sites (3 readers × 3 + writer × 1), every one
two-argument (**no `TensorAccessor` 3rd-argument / page-size site anywhere**).

| host site (file:line) | device site | originating Tensor | RTA slot (host) |
|---|---|---|---|
| `..._program_factory.cpp:97` / `:231` / `:368` | `_2d.cpp:42`, `_3d.cpp:41`, `_4d.cpp:41` | `target` (input) | reader RTA 0 (`Buffer*`) |
| `..._program_factory.cpp:98` / `:232` / `:369` | `_2d.cpp:59`, `_3d.cpp:58`, `_4d.cpp:42` | `output_grad` (input) | reader RTA 1 (`Buffer*`) |
| `..._program_factory.cpp:99` / `:233` / `:370` | `_2d.cpp:50`, `_3d.cpp:49`, `_4d.cpp:50` (all inside `#if defined(WEIGHT)`) | `weight` (optional input) | reader RTA 2 (`Buffer*`, or **`nullptr`** when absent) |
| `..._program_factory.cpp:102` / `:236` / `:373` | `writer...cpp:20` | `input_grad` (output / `tensor_return_value`) | writer RTA 0 (`Buffer*`) |

All four are **Case 1** (consumed through `TensorAccessor`). **No Case 2**: every raw typed pointer in
the readers (`CoreLocalMem<volatile uint16_t> input_grad_l1_ptr(dfb_input_grad_obj.get_write_ptr())`
and siblings) is a pointer into CB/L1 obtained from a **DFB method**, never a tensor base from an RTA.
No kernel does address arithmetic on a tensor base, so no binding needs the `get_bank_base_address`
bridge. No `->address()` appears anywhere in the op, so there is no offset-base-pointer fold either.

### Work split

Identical driver in all three configs; only `units_to_divide` differs.

- Driver: `split_work_to_cores(grid, units_to_divide)` where `grid = device->compute_with_storage_grid_size()`
- `units_to_divide`:
  - 2d / 3d: `input_grad.physical_volume() / TILE_HEIGHT / TILE_WIDTH`
  - 4d: `input_grad.physical_volume() / H / W * Ht * Wt`
- `(num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2)`
- Node walk: `for (i = 0, tile_offset = 0; i < num_cores; i++) { CoreCoord core = {i / core_h, i % core_h}; … tile_offset += units_per_core; }`, `core_h = grid.y`
- Per-node `units_per_core` selected by `core_group_1.contains(core)` / `core_group_2.contains(core)`,
  else `TT_THROW("Core not in specified core ranges")`.

**Per-node work rides an RTA (`units_per_core`), not a per-group CTA** — there are no CTAs at all —
so the *demoting-per-group-CTA* anti-pattern has no purchase here and there is no same-source
`KernelDescriptor` pair to preserve.

### Shared kernels

**none.** Census run per kernel (`grep -rl <filename> ttnn/cpp/ttnn/operations/`), hits disambiguated:

| kernel | binders repo-wide | `_metal2` sibling? |
|---|---|---|
| `reader_..._2d.cpp` | this op's factory only | no |
| `reader_..._3d.cpp` | this op's factory only | no |
| `reader_..._4d.cpp` | this op's factory only | no |
| `writer_....cpp` | this op's factory only (all three configs, but they all convert together) | no |

The only other hits were the two `METAL2_*.md` audit artifacts — documentation, not binders. The op
borrows no kernel *file* and lends none; no fork is created and none is reused. Sharing is
header-level only: the donors `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
(`read_tile` `:666`, `read_line` `:739`, `get_tilized_idx` `:618`) and
`tt_metal/hw/inc/api/numeric/bfloat16.h` (`bf16_to_fp32`, `fp32_to_bf16_truncate`). Both take
`DataflowBuffer` **by value** / plain scalars and need no donor-side change.

### Flags

- **No unreferenced kernel file** in the op directory — all four are bound.
- **No descriptor type outside the audit's scan**: the factory uses only `KernelDescriptor`,
  `CBDescriptor` / `CBFormatDescriptor`, `ReaderConfigDescriptor`, `WriterConfigDescriptor`.
- **The whole compute-kernel-config path is vestigial** (audit *Misc anomalies*): the op has no
  compute kernel, `ComputeConfigDescriptor` appears nowhere, and four of the five values
  `get_compute_kernel_config_args` returns (`math_fidelity`, `math_approx_mode`, `packer_l1_acc`,
  `dst_full_sync_en`) are unused. The port therefore builds **no** `ComputeHardwareConfig` and needs
  none of the compute-config translation table. The `compute_kernel_config` **attribute** stays
  exactly as it is (its removal is an ops-team API change, explicitly out of scope for this port).
- **`FP32_DEST_ACC_EN` is read after all — the brief's instruction to drop it is wrong, and the port
  keeps it.** The brief and audit say the define "no kernel reads (zero hits under `device/kernels/`)".
  That grep is scoped to the op's own kernel directory and misses the donor: the readers
  `#include "ttnn/kernel/dataflow/moreh_common.hpp"`, which branches on the macro at
  `moreh_common.hpp:22` (`#if defined(FP32_DEST_ACC_EN)` → selects the `FP32_DEST_ACC_FTYPE` typedef
  and two `fp32_dest_acc_cast` overloads). The pre-port build log confirms `-DFP32_DEST_ACC_EN=1`
  reaching the reader compile. This op's kernels happen not to *use* those symbols, so the emitted
  binary is very likely unchanged either way — but "very likely" is not the porting invariant, and
  carrying the define forward verbatim costs one line and makes the port a true syntax swap.
  Recorded in the port report under Friction → Gaps.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — the **base** concept.
  `override_runtime_arguments` is absent, so there is nothing to translate and the
  `Translating override_runtime_arguments` step is skipped entirely.
- **Custom `compute_program_hash`**: **none** — default reflection-based hash. Nothing to leave intact.
- **Implementation notes**:
  - The swap is a **method swap inside the existing `Factory` struct**: `create_descriptor` →
    `create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`. The
    device-operation class itself (`validate_inputs`, `validate_on_program_cache_miss`,
    `compute_output_specs`, `create_output_tensors`, `operation_attributes_t`, `tensor_args_t`) is
    untouched. `program_factory_t = std::variant<Factory>` already exists, so **exception 3
    (direct-descriptor conversion) does not fire** and the struct is *not* renamed to
    `MorehNllLossUnreducedBackwardProgramFactory` — renaming an existing factory struct is not port work.
  - `create_descriptor` must actually *disappear*, not merely gain a sibling:
    `ProgramSpecFactoryConcept` is defined with `!ProgramDescriptorFactoryConcept<T>`, and that
    concept is satisfied by the mere presence of `&T::create_descriptor`
    (`ttnn/api/ttnn/operation_concepts.hpp:72-74, 119-121`).
  - **No pybind change**: `..._nanobind.cpp` never bound `create_descriptor`, so exceptions 1 and 2
    don't fire either. This port carries **no** user-visible API change.
  - Header edits forced by the swap: `Factory`'s method declaration, `#include "ttnn/metal_v2_artifacts.hpp"`
    added, `#include <tt-metalium/program_descriptors.hpp>` dropped (it was there only for
    `ProgramDescriptor`).
  - `MeshTensor` extraction: the three impl helpers take `const MeshTensor&` and read shapes/dtype off
    `tensor_spec()`. The three substitutions are provably identity-preserving —
    `Tensor::logical_shape()`/`padded_shape()` forward to `tensor_spec()`
    (`ttnn/core/tensor/tensor.cpp:455-461`), `Tensor::dtype()` to
    `tensor_spec().tensor_layout().get_data_type()` (`:463`), and
    `Tensor::physical_volume()` is literally `padded_shape().volume()` (`:438`) — so the work-split and
    DFB-sizing arithmetic is unchanged. `MeshTensor::device()` returns the same `MeshDevice` that
    `ttnn::Tensor::device()` did (`ttnn/api/ttnn/tensor/tensor.hpp:245`), so `compute_with_storage_grid_size()`
    and `arch()` resolve identically.

## Planned Spec Shape

Structure is 1:1 with legacy, per config. The rank dispatch stays host-side in
`create_program_artifacts`, exactly where `create_descriptor`'s was
([Pattern: Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)).

### Variant: 2d

- **KernelSpecs** (2): `reader` ← `reader_..._2d.cpp`; `writer` ← `writer_....cpp`
- **DataflowBufferSpecs** (6, two of them conditional): `target`, `output_grad`, `input_grad`,
  `output_grad_scratch`, plus `weight` and `weight_scratch` **only when `weight_has_value`**
- **SemaphoreSpecs**: none
- **TensorParameters** (4, one conditional): `target`, `output_grad`, `input_grad`, plus `weight`
  **only when `weight_has_value`**
- **WorkUnitSpecs** (1): `main` = {`reader`, `writer`} over `all_cores`
- **Op-owned tensors**: none

### Variant: 3d

As 2d, minus `output_grad_scratch` → 5 DFBs (3 unconditional + 2 under `WEIGHT`).

### Variant: 4d

Identical to 3d in shape; only the RTA set and the work-split arithmetic differ.

### Endpoint bindings (all three variants)

| DFB | reader binding(s) | writer binding | net per node |
|---|---|---|---|
| `target` | PRODUCER + CONSUMER (`"target"`) | — | self-loop |
| `output_grad` | PRODUCER + CONSUMER (`"output_grad"`) | — | self-loop |
| `weight` *(WEIGHT only)* | PRODUCER + CONSUMER (`"weight"`) | — | self-loop |
| `weight_scratch` *(WEIGHT only)* | PRODUCER + CONSUMER (`"weight_scratch"`) | — | self-loop |
| `output_grad_scratch` *(2d only)* | PRODUCER + CONSUMER (`"output_grad_scratch"`) | — | self-loop |
| `input_grad` | PRODUCER (`"input_grad"`) | CONSUMER (`"input_grad"`) | 1P + 1C |

Both endpoints of a self-loop share one `accessor_name`, so the kernel keeps exactly one
`DataflowBuffer` object per DFB — no second handle, no `alias_with`, no multi-binding flag anywhere.

### `hw_config` / `compiler_options`

| KernelSpec | legacy resolved `(processor, noc, noc_mode)` | Metal 2.0 `hw_config` | `opt_level` |
|---|---|---|---|
| reader | `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` = reader default | `ttnn::create_reader_datamovement_config(arch)` | legacy `O2` = Metal 2.0 default → **leave unset** |
| writer | `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` = writer default | `ttnn::create_writer_datamovement_config(arch)` | legacy `O2` = Metal 2.0 default → **leave unset** |

No compute kernel ⇒ no `ComputeHardwareConfig`, no `unpack_modes`, no `bfp_pack_precision_mode`,
and the compute `O3` rule does not apply to any spec in this op.

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** Each config emits exactly one reader
`KernelDescriptor` and one writer `KernelDescriptor`, both over the same `all_cores`; the per-node
work difference between `core_group_1` and `core_group_2` travels as the `units_per_core` **RTA**, not
as a per-group CTA. So there is no same-source `KernelSpec` pair to preserve and no second
`WorkUnitSpec`.

## Dropped Plumbing

Per config (line numbers from the 2d impl; 3d/4d are the same shape at `:231-233`/`:236` and
`:368-370`/`:373`).

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `..._program_factory.cpp:97` | `TensorAccessorArgs(*target.buffer()).append_to(reader_compile_time_args)` | `TensorParameter{target}` + reader `TensorBinding{"target"}` |
| `..._program_factory.cpp:98` | `TensorAccessorArgs(*output_grad.buffer()).append_to(...)` | `TensorParameter{output_grad}` + reader `TensorBinding{"output_grad"}` |
| `..._program_factory.cpp:99` | `TensorAccessorArgs(weight.has_value() ? weight.value().buffer() : nullptr).append_to(...)` — the **`nullptr` placeholder block** that emitted two words purely to keep the kernel's offset chain aligned across configs | **conditional** `TensorParameter{weight}` + **conditional** reader `TensorBinding{"weight"}`. The placeholder is *not* carried forward — under Metal 2.0 the framework builds accessor args from the bindings, so there is no chain left to align. |
| `..._program_factory.cpp:102` | `TensorAccessorArgs(*input_grad.buffer()).append_to(writer_compile_time_args)` | `TensorParameter{input_grad}` + writer `TensorBinding{"input_grad"}` |
| reader RTA slot 0 (`:160`) | `target_buf` (`Buffer*`) | `TensorBinding` (address auto-injected) |
| reader RTA slot 1 (`:161`) | `output_grad_buf` (`Buffer*`) | `TensorBinding` |
| reader RTA slot 2 (`:162`) | `weight_buf` (`Buffer*` **or `nullptr`**) | conditional `TensorBinding` |
| writer RTA slot 0 (`:171`) | `input_grad_buf` (`Buffer*`) | `TensorBinding` |
| `_2d.cpp:38-40`, `_3d.cpp:37-39`, `_4d.cpp:37-39` | `TensorAccessorArgs<0>()` → `next_compile_time_args_offset()` chain (3 links) | gone — `TensorAccessor(tensor::name)` |
| `writer....cpp:18` | `constexpr auto input_grad_args = TensorAccessorArgs<0>()` | gone |
| `_2d.cpp:13-15`, `_3d.cpp:13-15`, `_4d.cpp:13-15`, `writer....cpp:12` | `get_arg_val<uint32_t>(i++)` address reads | gone with the RTA slots |
| `_2d.cpp:23-30`, `_3d.cpp:23-29`, `_4d.cpp:23-29`, `writer....cpp:16` | `constexpr uint32_t cb_target = tt::CBIndex::c_0;` and the 5 siblings (magic CB indices) | `dfb::target` &c. via `DFBBinding` |
| `..._program_factory.cpp:23-42` | the `push_cb` helper + every `CBDescriptor` / `CBFormatDescriptor` | `DataflowBufferSpec` |
| reader positional RTAs 3-8 (`:163-168`) | positional `get_arg_val<uint32_t>(i++)` reads | **named** RTAs: `ignore_index`, `num_tiles_per_core`, `start_id`, then per rank `Nt`,`C`,`Ct` (2d) / `C`,`Ct`,`Wt` (3d) / `num_inner_tile`,`C`,`Ct` (4d) |
| writer positional RTAs 1-2 (`:171`) | positional reads | **named** RTAs: `num_tiles_per_core`, `start_id` |
| `_2d.cpp:34,36`; `_3d.cpp:33,35`; `_4d.cpp:33,35` | six dead `get_dataformat(cb_*)` locals (`weight_data_format`, `output_grad_data_format` — assigned once, never read) | **deleted** (invoker-confirmed; audit Question 1). Not modernised to `dfb.get_dataformat()`: two of the six query `c_2`, which is not allocated in the non-`WEIGHT` config and whose `dfb::weight` token therefore does not exist, and both sit *outside* the `#if defined(WEIGHT)` guard. Provably dead ⇒ deletion is behaviour-preserving. |

**Nothing else is dropped.** There are no CTAs beyond the accessor blocks (zero
`get_compile_time_arg_val` calls in any kernel), no semaphore-ID RTAs, and no page-size third-argument
CTA/RTA anywhere.

**Deliberately *not* dropped, though it looks droppable:** the `FP32_DEST_ACC_EN` define — see
*Flags* above.

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  `target`, `output_grad`, `weight` (single-ended — reader-local FIFO with no distinct consumer) and
  `weight_scratch`, `output_grad_scratch` (genuinely sync-free — no FIFO ops at all). Five DM
  self-loops, legal on Gen1; Quasar-uplift's concern, not a Gen1 blocker.
- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  the shared-`accessor_name` mechanism the five self-loops borrow.
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings):
  `weight` + `weight_scratch` DFBs and the `weight` `TensorParameter` / `TensorBinding`, all gated on
  `weight_has_value` with the matching `WEIGHT` define. **No gate promotion is needed** — the legacy
  kernels already `#if defined(WEIGHT)`-guard every construction site, and the host already emits
  `WEIGHT` on exactly that condition, so the define and the bindings share one condition as-is.
- [Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories):
  the rank dispatch inside `create_program_artifacts`, one helper per rank.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  not needed for a handle, but the same spirit applies to the donor calls — `read_tile` / `read_line`
  take `DataflowBuffer` **by value**, so the readers build the named object from the token and pass it
  straight through, unchanged.
- [Unity-build hygiene for anonymous-namespace symbols](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols):
  `ttnn_op_moreh` is a unity-build target (`ttnn/CMakeLists.txt`, `cmake/project_options.cmake:20`)
  and this port introduces eight new named constants (`READER`, `WRITER`, six DFB names) whose names
  are exactly the kind a sibling moreh port would also pick. Declared as `const` at the **op's own
  namespace scope** rather than in the file's anonymous namespace: `const` at namespace scope already
  has internal linkage, and the enclosing namespace is unique per op, so a future sibling port cannot
  collide however it names its own constants. (The alternative the catalog suggests —
  factory-name-prefixing eight constants inside the anonymous namespace — buys the same guarantee
  more noisily.) The legacy `push_cb` helper, which *did* sit in the anonymous namespace and *does*
  collide by name with `moreh_nll_loss_step2`'s, is deleted by the port.

Explicitly **not** applied: `alias_with` (no multi-element `format_descriptors`), same-FIFO aliasing
(no `uint32_t` CB-index alias in any kernel), varargs (every RTA is a distinct field read once), the
multi-binding flag (no DFB has ≥3 touchers or two same-role touchers), borrowed-memory DFBs (no
`CBDescriptor::buffer`), op-owned tensors (none), and the shared-kernel fork rungs (nothing is shared).

## Deferred / Flagged

- **New finding, acted on:** the brief's "`FP32_DEST_ACC_EN` is read by nothing" is false — the donor
  `moreh_common.hpp:22` reads it. The port keeps the define. See *Flags*; reported under Friction → Gaps.
- **New finding, not acted on:** four of the six reader RTAs and both of the writer's non-address RTAs
  carry the **same value on every node** (`ignore_index`, `C`, `Ct`, `Nt`/`Wt`/`num_inner_tile`), so
  they are really CRTAs. Left as RTAs — RTA→CRTA changes dispatch semantics and is a separate
  cleanup, not port work. Routed to the port report under *Open items for downstream*.
- No structural issue uncovered that the audit missed; nothing here is a stop signal.

# Port Plan — `moreh_getitem`

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_getitem`, ported from `ProgramDescriptor`
(`ProgramDescriptorFactoryConcept`) to Metal 2.0 (`ProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

Both factories are ported in this change, which means **all three program shapes** the audit
identified (RM · Tilized-W · Tilized-noW) and **all six kernels**. Nothing is left for a later pass.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — two `static ProgramDescriptor create_descriptor(...)`
  (`moreh_getitem_device_operation.hpp:34,41`)
- Variants: two factories, **three program shapes** —
  - `MorehGetItemRmFactory` (`moreh_getitem_rm_factory.cpp:25`) → shape **RM**
  - `MorehGetItemTilizedFactory` (`moreh_getitem_tilized_factory.cpp:26`), which branches internally on
    `is_w_index_exist` (`:87`, `else` path from `:354`) → shapes **Tilized-W** and **Tilized-noW**
- Custom `compute_program_hash`: **none** — already the default reflection-based hash. No device-op
  hash edit in this port.
- Selection: `select_program_factory` on input layout (`moreh_getitem_device_operation.cpp:69-77`);
  the internal tilized branch is on `index_dims` + input rank, not on an attribute.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN
factory analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section
below.)*

Shared facts across all three shapes:

- **No compute kernels.** The op is pure data movement: reader → `c_0` → writer.
- **No semaphores** anywhere in the op.
- **No `opt_level`** set on any `KernelDescriptor` (`grep -n opt_level` over the op directory returns
  only the two `METAL2_*.md` artifacts). Both kernels per shape are DM, so the resolved legacy level is
  `O2`, which is also Metal 2.0's `CompilerOptions` default → **nothing to set** on any `KernelSpec`.
- **Work split**, identically shaped in all three:
  `split_work_to_cores_wt_core_range(core_range, num_units)` over the full compute grid
  (`CoreRange({0,0},{grid.x-1, grid.y-1})`). The per-group unit count rides an **RTA**
  (`num_units_per_core`), not a per-group CTA — so there is **no** `KernelDescriptor` multiplicity to
  preserve (see [Preserved Multiplicity](#preserved-multiplicity)).
- **Every buffer base arrives as a `Buffer*` entry in `emplace_runtime_args`** (the framework's
  `BufferBinding` form), never as `->address()`. Undefined index slots are passed as `nullptr`, which
  `emplace_runtime_args_impl` lowers to a literal `0u` with no binding.

### Variant: RM (`moreh_getitem_rm_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `moreh_getitem_kernels/reader_moreh_getitem.cpp` | `all_cores` (`:154`) | 6 × `TensorAccessorArgs` blocks — `input` then `index_info[0..4]`, all five slots unconditionally (`:144-148`) | none | 36 per core (`:187-236`); 6 are `Buffer*`, 30 scalar | none | none (`reader_defines` declared empty, `:141`) | unset → `O2` | `ReaderConfigDescriptor{}` (`:157`) |
| writer | `moreh_getitem_kernels/writer_moreh_getitem.cpp` | `all_cores` (`:166`) | 1 × `TensorAccessorArgs` block — `output` (`:159-160`) | none | 4 per core (`:238-250`); 1 `Buffer*`, 3 scalar | none | none (empty, `:142`) | unset → `O2` | `WriterConfigDescriptor{}` (`:169`) |

Reader RTA order (host `:187-236` ↔ kernel `reader_moreh_getitem.cpp:13-57`, positionally 1:1):
`input.buffer()`, `index_info[0..4].buffer` · `input_stick_idx_stride_{n,c,d,h}` ·
`input_size_{n,c,d,h,w}` · `index{0..4}_is_defined` · `index{0..4}_stick_size` · `index_size`,
`index_start_dim`, `index_end_dim` · `output_size_{n,c,d,h,w}` · `start_id`, `num_sticks`
(host `num_units_per_core`), `stick_size` (host `input_unit_size`).

Writer RTA order: `output.buffer()` · `output_stick_size` · `start_id`, `num_sticks`.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (`:99-109`) | `rounded_input_page_size` = `round_up_to_mul32(input_5d_shape[-1] * elt)` | `all_cores` | input dtype | same as total | not set |
| `c_1 + dim`, per **defined** dim ∈ [0,5) (`:111-127`) | `round_up_to_mul32(index_info[dim].unit_size)` | `all_cores` | index dtype | same as total | not set |
| `c_16` (`:129-138`) | `rounded_input_page_size` | `all_cores` | output dtype | same as total | not set |

Every CB is `total_size == page_size` → `num_entries = 1`, `entry_size = page_size`. Holds for all
three shapes.

#### Semaphores

none — the op declares none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `rm_factory.cpp:145` | `input` | reader slot 0 |
| `rm_factory.cpp:75,146-148` | `index_tensors[i]` → `index_info[dim]`, dim 0..4 | reader slots 1..5 (`nullptr` when undefined) |
| `rm_factory.cpp:160` | `output` | writer slot 0 |

Kernel-side: `reader_moreh_getitem.cpp:66-83` (6 accessors, **3-arg** form), `writer_moreh_getitem.cpp:24-27`
(1 accessor, 3-arg form). All **Case 1** — every access is through the accessor; no raw base pointers.

#### Work split

- Driver: `split_work_to_cores_wt_core_range(core_range, num_units)` (`:89-90`)
- `num_units` = `output.physical_volume() / output_shape[-1]` (one unit = one output stick)
- `num_cores`, `all_cores`, `core_group_1`, `core_group_2`, `num_units_per_core_group_{1,2}`
- Per-core RTA loop `:183-253`, node order `{(i / core_h), (i % core_h)}` (offsets are 0)

### Variant: Tilized-W (`moreh_getitem_tilized_factory.cpp:87-353`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `moreh_getitem_tilized_kernels/reader_moreh_getitem_tilize_w.cpp` | `all_cores` (`:200`) | 6 × `TensorAccessorArgs` — `input` + `index_info[0..4]` (`:189-193`) | none | 46 per core (`:259-318`); 6 `Buffer*`, 40 scalar | none | `ROW_MAJOR_INDEX=1` **or** `TILIZE_INDEX=1` (`:183-187`) | unset → `O2` | `ReaderConfigDescriptor{}` |
| writer | `moreh_getitem_tilized_kernels/writer_moreh_getitem_tilize_w.cpp` | `all_cores` (`:213`) | 1 × `TensorAccessorArgs` — `output` (`:205-206`) | none | 16 per core (`:320-344`); 1 `Buffer*`, 15 scalar | none | none (empty) | unset → `O2` | `WriterConfigDescriptor{}` |

Reader RTA order ↔ `reader_moreh_getitem_tilize_w.cpp:15-69`: buffers ·
`input_stick_idx_stride_{n,c,d,h,w}` · `input_size_{c,d,h}_without_padding` ·
`input_num_stick_width` · `input_noc_id_stride_{n,c,d,h}` · `input_size_{n,c,d,h,w}` ·
`index{0..4}_is_defined` · `index{0..4}_stick_size` · `index_size` ·
`output_size_{n,c,d,h,w}`, `output_num_stick_width` · `start_id`, `num_sticks`, `element_size`,
`num_elements_per_alignment`, `num_alignment_width`.

Writer RTA order ↔ `writer_moreh_getitem_tilize_w.cpp:16-35`: `output.buffer()` ·
`output_size_{c,d,h,w}_without_padding` · `output_noc_id_stride_{n,c,d,h}` ·
`output_num_stick_width` · `start_id`, `num_sticks`, `stick_size`, `element_size`,
`num_elements_per_alignment`, `num_alignment_width`.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `c_0` (`:126-136`) | `round_up_to_mul32(input.element_size())` | `all_cores` | input dtype | same | not set |
| `c_1 + dim`, per defined dim ∈ [0,5) (`:138-154`) | `1024 * 4` (one INT32 tile) | `all_cores` | index dtype | same | not set |
| `c_16` (`:156-166`) | `round_up_to_mul32(output.element_size())` | `all_cores` | output dtype | same | not set |
| `c_17` (`:168-177`) | `round_up_to_mul32(output.element_size())` | `all_cores` | output dtype | same | not set |

#### Semaphores / Tensor accessors / Work split

- Semaphores: none.
- Tensor accessors: `input` (`:190`), `index_info[0..4]` (`:97,191-193`), `output` (`:206`) — all
  **Case 1**, all **2-arg** kernel-side (`reader_..._tilize_w.cpp:85-91`,
  `writer_..._tilize_w.cpp:40-41`): no page-size third argument to drop here.
- Work split: `split_work_to_cores_wt_core_range` (`:115-117`) with
  `num_units = ∏ output_5d_without_padding[0..3] × ceil(output_w / num_elements_per_alignment)`;
  per-core loop `:255-347`.

### Variant: Tilized-noW (`moreh_getitem_tilized_factory.cpp:356-597`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `moreh_getitem_tilized_kernels/reader_moreh_getitem_tilize.cpp` | `all_cores` (`:450`) | 6 × `TensorAccessorArgs` (`:439-443`) | none | 45 per core (`:507-565`); 6 `Buffer*`, 39 scalar | none | `ROW_MAJOR_INDEX=1` **or** `TILIZE_INDEX=1` (`:433-437`) | unset → `O2` | `ReaderConfigDescriptor{}` |
| writer | `moreh_getitem_tilized_kernels/writer_moreh_getitem_tilize.cpp` | `all_cores` (`:463`) | 1 × `TensorAccessorArgs` (`:455-456`) | none | 14 per core (`:567-589`); 1 `Buffer*`, 13 scalar | none | none (empty) | unset → `O2` | `WriterConfigDescriptor{}` |

Reader RTA order ↔ `reader_moreh_getitem_tilize.cpp:15-68`: as Tilized-W but with
`input_noc_id_stride_{n,c,d,h}` **before** `input_num_stick_width` (the two are transposed relative to
the W reader — a real per-shape difference, not a typo), and the tail is
`start_id`, `num_sticks`, `stick_size`, `element_size` (no alignment pair).

Writer RTA order ↔ `writer_moreh_getitem_tilize.cpp:14-31`: `output.buffer()` ·
`output_size_{c,d,h,w}_without_padding` · `output_noc_id_stride_{n,c,d,h}` ·
`output_num_stick_width` · `start_id`, `num_sticks`, `stick_size`, `element_size`.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `c_0` (`:388-398`) | `round_up_to_mul32(16 * input.element_size())` | `all_cores` | input dtype | same | not set |
| `c_1 + dim`, per defined dim ∈ [0,5) (`:400-416`) | `1024 * 4` | `all_cores` | index dtype | same | not set |
| `c_16` (`:418-427`) | `rounded_input_page_size` | `all_cores` | output dtype | same | not set |

#### Semaphores / Tensor accessors / Work split

- Semaphores: none.
- Tensor accessors: `input` (`:440`), `index_info[0..4]` (`:364,441-443`), `output` (`:456`) — all
  **Case 1**, all **2-arg** kernel-side. Nothing to drop.
- Work split: `split_work_to_cores_wt_core_range` (`:378-379`),
  `num_units = ∏ output_5d_without_padding[0..3] × ceil(output_w / 16)`; loop `:503-592`.

### Shared kernels

**none.** Census run per the catalog's procedure — `grep -rl <filename> ttnn/cpp/ttnn/operations/` for
each of the six kernel `.cpp` files and for `moreh_getitem_tilized_kernels/common.hpp` returns **no hit
outside `moreh_getitem/`**, so none is *lent*. No `kernel_source` points outside the op directory, so
none is *borrowed*. The three shapes bind six **distinct** sources, so there is no *intra-op* sharing
either — and both factories convert in this same change regardless. No `_metal2` sibling exists in the
op directory (`find … -name '*_metal2*'` empty), so no rung-1 reuse and no rung-2 fork is created.

### Flags

- **`index_cbs[5]` is dead in all three readers** (`reader_moreh_getitem.cpp:93-99`,
  `reader_moreh_getitem_tilize.cpp:100-106`, `reader_moreh_getitem_tilize_w.cpp:101-107`). In the two
  tilized readers it is never read; in the RM reader its only use is
  `tt::CBIndex idx_cb = index_cbs[dim];` (`:151`) and `idx_cb` is itself never used. Its element type is
  `tt::CBIndex` and its values are the legacy CB indices, so the CB→DFB transition removes it
  outright — it is not ported (brief: "do not port it").
- **`cb_in5` in the RM and Tilized-noW readers is reachable only through that dead array** — both
  readers' dim loops never select it. Removed with the array.
- **RM `c_5` has zero endpoints** whenever a normalized index dim of 4 is defined
  (`rm_factory.cpp:111-127` allocates it; `reader_moreh_getitem.cpp:146` loops `dim = 3 … 0`). Audit
  Question 3 — see [Deferred / Flagged](#deferred--flagged) for the decision and its treatment.
- No unreferenced kernel files in the op directory; all six are bound.
- `IndexInfo::args` (`rm_factory.cpp:18`, `tilized_factory.cpp:17`) exists only to carry
  `TensorAccessorArgs` into the CTA list; it disappears with the binding model. `IndexInfo::buffer`
  (a `Buffer*`) is replaced by the originating `Tensor` the `TensorParameter` / `TensorArgument` need.

## TTNN ProgramFactory

*Filled in during the planning step. The concept itself was chosen in the audit; this section carries
it forward.*

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — plain, for **both** factories.
  Confirmed against `ttnn/api/ttnn/operation_concepts.hpp:118-121`: satisfied by declaring
  `create_program_artifacts` and nothing else (no `override_runtime_arguments`, no
  `create_descriptor`, no `create_workload_descriptor`).
- **Custom `compute_program_hash`**: none — nothing to delete.
- **Implementation notes**:
  - Both factory declarations in `moreh_getitem_device_operation.hpp:33-45` change from
    `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` to
    `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`; the header gains
    `ttnn/metal_v2_artifacts.hpp` and drops the now-unused `<tt-metalium/program_descriptors.hpp>`.
    This is the declaration of the ported factory itself, not a device-op-class edit.
  - **No pybind change**: `moreh_getitem_nanobind.cpp:18` binds only `ttnn::moreh_getitem`; no
    `create_descriptor` is exposed, so exception 2 does not apply.
  - No op-owned tensors → `ProgramArtifacts::op_owned_tensors` left defaulted.
  - The factory bodies keep their `ttnn::Tensor` locals for the 5-D shape arithmetic and reach
    `.mesh_tensor()` only at the `TensorArgument` sites. Extracting `MeshTensor` at factory entry is
    the documented preference, but here it would mean rewriting shape/stride math the port is not
    otherwise touching; the same shape is used by the landed `moreh_mean` port. Recorded in the report.

## Planned Spec Shape

Default 1:1 with legacy, minus the dead `c_16` (all shapes) and the endpoint-less RM dim-4 index DFB.
Resource names below are the `unique_id`s; the kernel-side accessor names in parentheses are chosen to
match the **existing** kernel locals so the kernel diff stays at the API swap.

### Variant: RM

- **KernelSpecs** (2): `reader` (source unchanged), `writer` (source unchanged).
- **DataflowBufferSpecs** (1 + #defined dims in [0,4)):
  - `in0` — legacy `c_0`; `entry_size = rounded_input_page_size`, `num_entries = 1`,
    `data_format_metadata =` input dtype. Reader accessor `in0`, writer accessor `out`.
  - `in1` … `in4` — legacy `c_1 + dim` for each **defined** dim ∈ [0,4);
    `entry_size = round_up_to_mul32(index_info[dim].unit_size)`, `num_entries = 1`,
    `data_format_metadata =` index dtype. Reader accessors `in1` … `in4`.
  - **No spec for legacy `c_16`** (dead — drop) and **none for the dim-4 index** (`c_5`, zero
    endpoints — see Deferred / Flagged).
- **SemaphoreSpecs**: none.
- **TensorParameters** (2 + #defined dims): `input`, `output`, and `index0` … `index4` for each defined
  dim (all five slots are eligible; the reader binds whichever exist).
- **WorkUnitSpecs** (1): `{reader, writer}` on `all_cores`.
- **Op-owned tensors**: none.

### Variant: Tilized-W

- **KernelSpecs** (2): `reader`, `writer`.
- **DataflowBufferSpecs** (2 + #defined dims in [0,5)):
  - `in0` — legacy `c_0`; `entry_size = round_up_to_mul32(input.element_size())`, `num_entries = 1`.
    Reader accessor `in0`, writer accessor `out0`.
  - `in1` … `in5` — legacy `c_1 + dim` per defined dim ∈ [0,5); `entry_size = 4096`, `num_entries = 1`.
  - `out1` — legacy `c_17`; `entry_size = round_up_to_mul32(output.element_size())`, `num_entries = 1`.
    Writer accessor `out1`; **self-loop** (see Applied Patterns).
  - **No spec for legacy `c_16`** (dead — drop).
- **SemaphoreSpecs**: none.
- **TensorParameters** (2 + #defined dims): `input`, `output`, `index0` … `index4`.
- **WorkUnitSpecs** (1): `{reader, writer}` on `all_cores`.

### Variant: Tilized-noW

- **KernelSpecs** (2): `reader`, `writer`.
- **DataflowBufferSpecs** (1 + #defined dims in [0,4)):
  - `in0` — legacy `c_0`; `entry_size = round_up_to_mul32(16 * input.element_size())`,
    `num_entries = 1`. Reader accessor `in0`, writer accessor `out`.
  - `in1` … `in4` — legacy `c_1 + dim` per defined dim ∈ [0,4); `entry_size = 4096`, `num_entries = 1`.
    (A defined dim 4 is impossible in this shape — it routes to Tilized-W by construction,
    `tilized_factory.cpp:74-79` — so the loop stops at dim 3, matching the reader's touch set.)
  - **No spec for legacy `c_16`** (dead — drop).
- **SemaphoreSpecs**: none.
- **TensorParameters** (2 + #defined dims): `input`, `output`, `index0` … `index4`.
- **WorkUnitSpecs** (1): `{reader, writer}` on `all_cores`.

### DFB endpoint assignment (census re-derived, not transcribed)

Counted per node, per shape, from a code reference in a kernel (a reference removed by `#ifdef` is not
a toucher; one under a plain `if` is). One reader instance and one writer instance per node in every
shape; no compute kernels; no semaphores, so no hidden co-fill is possible.

| Shape | DFB | Touchers | Disposition | Agrees with brief |
|---|---|---|---|---|
| RM | `in0` | reader FIFO-produces (`reader:211,214`), writer FIFO-consumes (`writer:34,37`) | 1P + 1C | yes |
| RM | `in1`–`in4` | reader only — full FIFO cycle in one kernel (`reserve_back:161` … `push_back:184` … `wait_front:189` … `pop_front:190`) | **self-loop** (1 toucher) | yes |
| Tilized-W | `in0` | reader produces (`:331,371`), writer consumes (`:83,101`) | 1P + 1C | yes |
| Tilized-W | `in1`–`in5` | reader only — locked producer (`reserve_back` + `get_write_ptr`, no `push_back`) | **self-loop** (1 toucher) | yes |
| Tilized-W | `out1` | writer only — role-free: `get_read_ptr` (`:50`), raw stores (`:91,98`), NoC source (`:104`); no FIFO ops | **self-loop** (1 toucher) | yes |
| Tilized-noW | `in0` | reader produces (`:276,295`), writer consumes (`:43,64`) | 1P + 1C | yes |
| Tilized-noW | `in1`–`in4` | reader only — locked producer | **self-loop** (1 toucher) | yes |

No DFB in any shape has ≥3 distinct touchers or two kernels locked to the same FIFO role, so
`advanced_options.allow_instance_multi_binding` is **not set anywhere**. No DFB is both self-looped and
multi-bound.

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** Each shape emits exactly one reader and one writer
`KernelDescriptor` over a single `all_cores` range; the per-group difference
(`num_units_per_core_group_1` vs `_2`) is already an **RTA**, so there is no per-group CTA to preserve
and no second `KernelSpec` of the same source. One `WorkUnitSpec` per shape covers `all_cores`.

## Dropped Plumbing

Per shape. "slot" numbers are 1-based positions in the legacy `emplace_runtime_args` list.

### RM

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `rm_factory.cpp:191` (reader RTA slot 1) | `input.buffer()` | `TensorBinding{input, "s0"}` on reader |
| `rm_factory.cpp:192-196` (reader RTA slots 2-6) | `index_info[0..4].buffer` (`nullptr` → literal `0u`) | `TensorBinding{index0..index4, "index0".."index4"}` on reader, **declared only for defined slots** |
| `rm_factory.cpp:242` (writer RTA slot 1) | `output.buffer()` | `TensorBinding{output, "s0"}` on writer |
| `rm_factory.cpp:145` + `reader:66` | `TensorAccessorArgs(input.buffer()).append_to(cta)` + `TensorAccessorArgs<0>()` | binding mechanism (host codegen) |
| `rm_factory.cpp:146-148` + `reader:67-71` | 5 × `dim.args.append_to(cta)` + `next_compile_time_args_offset()` chain | binding mechanism |
| `rm_factory.cpp:160` + `writer:24` | `TensorAccessorArgs(output.buffer()).append_to(cta)` + `TensorAccessorArgs<0>()` | binding mechanism |
| `reader:75` | 3rd ctor arg `stick_size` | dropped — `TensorAccessor(tensor::s0)`; token supplies the aligned page size |
| `reader:79-83` | 3rd ctor arg `index{0..4}_stick_size` | dropped — `TensorAccessor(tensor::index{0..4})` |
| `writer:27` | 3rd ctor arg `output_stick_size` | dropped — `TensorAccessor(tensor::s0)` |
| `reader:59-64` | `constexpr auto cb_in0..cb_in5 = tt::CBIndex::c_*` | `dfb::in0` … `dfb::in4` (DFB bindings); `cb_in5` gone with the dead array |
| `writer:22` | `constexpr uint32_t cb_id_out = tt::CBIndex::c_0` | `dfb::out` (CONSUMER binding on `in0`) |
| `reader:93-99,151` | `tt::CBIndex index_cbs[5]` + `idx_cb` dead store | removed (dead; CB-index vocabulary) |
| `rm_factory.cpp:129-138` | `c_16` `CBDescriptor` (dead CB) | **no spec** — dropped |
| `rm_factory.cpp:111-127`, dim 4 | `c_5` `CBDescriptor` (zero endpoints) | **no spec** — dropped |
| all reader/writer RTAs | positional `get_arg_val<uint32_t>(i++)` | named `get_arg(args::<name>)`, names taken from the kernel locals |

The `stick_size` / `index{0..4}_stick_size` / `output_stick_size` **RTAs themselves stay** — the kernels
also use them as NoC transfer sizes (`reader:163,212`, `writer:35`). Only the accessor's third
constructor argument goes away.

### Tilized-W

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `tilized_factory.cpp:263` | `input.buffer()` (reader RTA slot 1) | `TensorBinding{input, "s0"}` |
| `tilized_factory.cpp:264-268` | `index_info[0..4].buffer` (slots 2-6) | `TensorBinding{index0..index4, …}`, defined slots only |
| `tilized_factory.cpp:324` | `output.buffer()` (writer RTA slot 1) | `TensorBinding{output, "s0"}` |
| `tilized_factory.cpp:190-193` + `reader_…_w:78-83` | `TensorAccessorArgs` × 6 + offset chain | binding mechanism |
| `tilized_factory.cpp:205-206` + `writer_…_w:40` | `TensorAccessorArgs` × 1 | binding mechanism |
| `reader_…_w:71-76` | `cb_in0..cb_in5` CB-index constants | `dfb::in0` … `dfb::in5` |
| `writer_…_w:37-38` | `cb_id_out0 = c_0`, `cb_id_out1 = c_17` | `dfb::out0` (on `in0`), `dfb::out1` (on `out1`) |
| `reader_…_w:101-107` | dead `index_cbs[5]` | removed |
| `tilized_factory.cpp:156-166` | `c_16` `CBDescriptor` (dead CB) | **no spec** — dropped |
| all RTAs | positional `i++` reads | named `get_arg(args::…)` |

No third-argument accessor sites in this shape (all 2-arg already).

### Tilized-noW

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `tilized_factory.cpp:511` | `input.buffer()` (reader RTA slot 1) | `TensorBinding{input, "s0"}` |
| `tilized_factory.cpp:512-516` | `index_info[0..4].buffer` (slots 2-6) | `TensorBinding{index0..index4, …}`, defined slots only |
| `tilized_factory.cpp:571` | `output.buffer()` (writer RTA slot 1) | `TensorBinding{output, "s0"}` |
| `tilized_factory.cpp:440-443` + `reader_…_tilize:77-82` | `TensorAccessorArgs` × 6 + offset chain | binding mechanism |
| `tilized_factory.cpp:455-456` + `writer_…_tilize:35` | `TensorAccessorArgs` × 1 | binding mechanism |
| `reader_…_tilize:70-75` | `cb_in0..cb_in5` CB-index constants | `dfb::in0` … `dfb::in4` (`cb_in5` gone with the dead array) |
| `writer_…_tilize:33` | `cb_id_out = c_0` | `dfb::out` (CONSUMER on `in0`) |
| `reader_…_tilize:100-106` | dead `index_cbs[5]` | removed |
| `tilized_factory.cpp:418-427` | `c_16` `CBDescriptor` (dead CB) | **no spec** — dropped |
| all RTAs | positional `i++` reads | named `get_arg(args::…)` |

## Applied Patterns

- **Conditional / optional DFB bindings** (patterns catalog) — applied to the **optional index
  tensors and their DFBs**, in all three shapes. This is the port's central design item: the legacy op
  passes all five `index_info[N].buffer` slots and lets `nullptr` lower to a literal `0u` with no
  binding, then constructs all five `TensorAccessor`s unconditionally. Metal 2.0 has no
  "declared but absent" binding, so per the catalog the host **omits** the binding for an undefined
  slot, emits a matching `HAS_INDEX{0..4}` define via `KernelSpec::compiler_options.defines`, and the
  kernel `#ifdef`-gates the accessor construction, the `DataflowBuffer` construction, and every
  `if (dim == N)` block that references either token. The `index{0..4}_is_defined` **RTAs and the
  runtime `if (index_is_defined[dim])` guard stay exactly as they are** — the preprocessor gate sits
  *inside* that guard and removes only the token references, so no kernel logic changes. The set of
  defined slots is a function of `index_dims` + input rank, both of which the default program hash
  covers, so the gating cannot go stale across a cache hit.
- **Self-loop DFB binding** (sync-free / single-ended CB → self-loop): RM `in1`–`in4`, Tilized-W
  `in1`–`in5` and `out1`, Tilized-noW `in1`–`in4`. Each has exactly **one** toucher (see the census
  table), so the touching kernel is bound both PRODUCER and CONSUMER under one accessor name. All are
  DM self-loops — legal on Gen1, and Quasar-uplift's concern rather than a blocker here.
- **Dead-CB drop**: `c_16` in all three shapes (audit Q2, confirmed by the user), plus the RM dim-4
  index CB `c_5` (audit Q3 — see below). No spec is built; nothing else changes.
- **Multi-variant factory**: `MorehGetItemTilizedFactory::create_program_artifacts` keeps its internal
  `is_w_index_exist` branch and returns a different `ProgramArtifacts` from each side.
- **Unity-build hygiene**: all resource-name constants are declared **function-local** in each factory
  body (the two factory `.cpp` files share a unity-build translation unit, and both would otherwise
  declare `IN0`, `READER`, … in a merged anonymous namespace).
- **Pass DFB handles directly to LLKs / kernel-lib helpers**: not exercised — this op has no compute
  kernels and calls no helper that takes a CB id.

## Deferred / Flagged

- **Audit Q1 (RM `TensorAccessor` third argument) — decided by the invoker: drop the argument at all 7
  RM sites and set **no** relaxation** (`TensorParameter::relaxations` left default-empty,
  `dynamic_tensor_shape` not set). Verified mechanically while planning: the legacy override passes the
  *unaligned* logical stick size, the binding token instead supplies the host-emitted
  `buffer->aligned_page_size()` (`tt_metal/impl/buffers/tensor_accessor_args.cpp:179-185`), and the
  interleaved accessor **realigns whatever it is given** —
  `InterleavedAddrGen::aligned_page_size = align_power_of_2(page_size, allocator_alignment)`
  (`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:289-290`). `align(align(p,A),A) = align(p,A)`,
  so the addressing is byte-identical either way. The RM tests cover a width whose row bytes are not
  32-aligned (`[10, 5, 7, 70]` bfloat16 → 140 B), which exercises exactly that equivalence.
- **Audit Q2 (`c_16` dead-CB drop) — confirmed by the invoker.** Applied in all three shapes; each drop
  site is recorded in the report.
- **Audit Q3 (rank-4 ROW_MAJOR input with an index on its last dimension) — out of scope per the
  invoker.** The `dim != 4` guard (`moreh_getitem_device_operation.cpp:47-51`) is **not touched** and
  **no dim-4 handling is added** to the RM path. What the port does have to decide is what happens to
  the endpoint-less `c_5` allocation in that configuration, because a bindingless DFB is rejected by the
  Metal 2.0 validator where legacy merely wasted a page: the RM index-DFB loop therefore covers dims
  0..3 — the reader's actual touch set — so no spec is built for dim 4. That is the ordinary dead-CB
  disposition and is zero-functional-change (the CB was never touched, and the index tensor is still
  silently ignored exactly as it is today). The underlying defect is recorded in the report as a
  pre-existing finding, not fixed here.
- **New finding during planning — the RM reader's `DataflowBuffer*` is portable as-is.** The brief
  expected `DataflowBuffer* index_dfb_obj` (`reader_moreh_getitem.cpp:158`, dereferenced at `:184,189,190`)
  to have no binding-token analogue and the FIFO calls to have to move into the per-`dim` branches.
  They do not: `DataflowBuffer` is a plain non-template class
  (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:61`) whose Metal 2.0 constructor merely takes a
  `DFBBindingToken`, so a pointer to a locally-constructed object stays valid C++ and the three
  dereference sites are untouched. Only the objects' construction changes (`dfb::in1` … `dfb::in4`).
  The pointer is assigned only inside `#ifdef`-gated `dim == N` branches, and it is reached only under
  the matching `index_is_defined[dim]` runtime guard, so it cannot be dereferenced null.
- **`index4` in the Tilized-noW reader is unreachable, and kept.** `is_w_index_exist` routes any defined
  dim-4 index to the W shape, so `HAS_INDEX4` is never emitted for this shape and the gated accessor
  compiles away. Kept for structural uniformity with the other two readers (legacy constructs it too);
  noted rather than deleted so the port stays a syntax swap.

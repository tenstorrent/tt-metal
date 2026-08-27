# Port Plan — `data_movement/concat`

Port plan for `ttnn/cpp/ttnn/operations/data_movement/concat`, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

> ### Scope — inherited from `METAL2_PORT_BRIEF.md`
>
> The op is **RED at op level**. This plan covers the brief's clean three-factory subset:
>
> | Factory | This pass |
> |---|---|
> | `ConcatProgramFactory` (default: all interleaved + ND-sharded fallback) | **ported** |
> | `ConcatS2SRMProgramFactory` | **ported** |
> | `ConcatS2STiledProgramFactory` | **ported** |
> | `ConcatS2SMultiProgramFactory` | ⛔ gated — untouched |
> | `ConcatBlockShardedProgramFactory` | ⛔ gated — untouched |
> | `ConcatS2IProgramFactory` | ⛔ gated (dead code) — untouched |
>
> The atomic unit of a port is one ProgramFactory, so the `program_factory_t` variant ends up holding a
> mix of `descriptor` and `MetalV2` factories. The brief records this as confirmed-supported by the TTNN
> framework owner.

---

# `ConcatProgramFactory`

## Legacy Inventory

### Legacy factory shape

- **Concept**: `ProgramDescriptorFactoryConcept` — `static tt::tt_metal::ProgramDescriptor create_descriptor(const ConcatParams&, const ConcatInputs&, Tensor&)` at `device/concat_program_factory.hpp:14`.
- **Where the factory methods live**: in a `program_factory_t` variant (`device/concat_device_operation.hpp:29-35`, 6 alternatives). **Not** the direct-descriptor shape, so `ttnn_factory.md` exception 3 does not apply.
- **Variants**: single factory, but four config axes selected inside one `create_descriptor` body:
  | axis | source | effect |
  |---|---|---|
  | `rm_layout` (`output.layout() == ROW_MAJOR`) | `:31` | swaps **both** reader and writer kernel sources; changes page-size math; adds a writer CTA |
  | `WIDTH_CONCAT` (`rm_layout && dim == rank-1`) | `:208-210` | reader `defines` only |
  | `sub_core_grids` present & output not sharded | `:55-84` | replaces `split_work_to_cores` with a sub-grid work split |
  | CB depth 1 vs 2 | `:122-125` | L1-budget fallback |
- **Custom `compute_program_hash`**: **none.** `grep -rn 'compute_program_hash|attribute_values|to_hash|override_runtime_arguments|get_dynamic_runtime_args'` over the whole op directory returns zero hits. Default reflection-based hash; nothing to leave alone.

### Kernels

Two `KernelDescriptor`s. `N` = `num_input_tensors` (up to 47, `concat_device_operation.cpp:285`).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs (per core) | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `rm_layout ? device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp : device/kernels/dataflow/reader_concat_interleaved_start_id.cpp` (`:221-224`) | `all_cores` | `[0]=src0_cb_index(0)`, `[1]=N`, `[2..2+N-1]=page_size_per_tensor[N]`, then `N`× `TensorAccessorArgs(input_i)` blocks (`:200-205`) | none | `[0]=num_pages_per_core`, `[1]=curr_tensor`, `[2]=curr_tensor_id`, `[3..3+N-1]=Buffer*` per input, `[3+N..3+2N-1]=num_pages_per_block[N]`, `[3+2N..3+3N-1]=page_id_per_tensor[N]` (`:270-281`) | none | `WIDTH_CONCAT=1` iff `rm_layout && dim==rank-1` (`:207-210`) | **unset → resolved `O2`** (DM) | `ReaderConfigDescriptor{}` (`:229`) |
| writer | `rm_layout ? ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp : ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (`:232-235`) — **both outside the op directory: borrowed, see [Shared kernels](#shared-kernels)** | `all_cores` | RM: `[0]=src0_cb_index`, `[1]=dst_buffer->page_size()`; TILE: `[0]=src0_cb_index`; then `TensorAccessorArgs(dst)` (`:212-218`) | none | RM: `{Buffer*, output page_size, num_pages_per_core, num_pages_written}`; TILE: `{Buffer*, num_pages_per_core, num_pages_written}` (`:283-291`) | none | none | **unset → resolved `O2`** (DM) | `WriterConfigDescriptor{}` (`:239`) |

`grep -n opt_level` over the op directory returns **zero lines** — no `KernelDescriptor::opt_level` is set anywhere.
Both descriptors are DM, so `std::nullopt` resolves to `O2`, which is also Metal 2.0's `CompilerOptions` default: **nothing to carry**. (Rule 2 — the explicit-`O3`-on-compute rule — does not apply; this factory builds no compute kernel.)

**RM writer CTA slot 1 is dead.** `:214` bakes `dst_buffer->page_size()` into the RM writer's CTA slot 1, but the legacy RM donor never reads it — it takes `stick_size` from RTA slot 1 and its accessor args start at `TensorAccessorArgs<2>()`. The slot exists only to fill that offset. It disappears with the accessor-args plumbing; not a behaviour change.

### CBs

One `CBDescriptor` (`:126-134`).

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 (`src0_cb_index`) | `num_input_pages * single_page_size` | `all_cores` | `cb_data_format` = `datatype_to_dataformat_converter(output.dtype())` | `single_page_size` | **unset** (`nullopt`) |

- `single_page_size` = `tt::align(output.element_size() * output.padded_shape()[-1], common_align_len)` (RM) or `tt::tile_size(cb_data_format)` (TILE) — `:37-43`.
- `num_input_pages` = 2, falling back to 1 when `2 * single_page_size > l1_budget` (`:122-125`).
- **No `.buffer`** → not borrowed memory. Plain L1 staging.
- **No GlobalCircularBuffer** anywhere in this factory (no `.global_circular_buffer`, no `global_cb` parameter, no `remote_cb_config`).

### Semaphores

**none** — the factory declares no `SemaphoreDescriptor`. (Op-wide: concat declares no semaphores at all.)

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `concat_program_factory.cpp:204` — `TensorAccessorArgs(*input_tensors[i].buffer()).append_to(reader_compile_time_args)`, `i ∈ [0, N)` | `tensor_args.input_tensors[i]` | reader RTA slots `3 .. 3+N-1`, delivered as `Buffer*` (`:276`) |
| `concat_program_factory.cpp:218` — `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` | `tensor_return_value` (output) | writer RTA slot `0`, delivered as `Buffer*` (`:285`, `:290`) |

Kernel side: both readers build them through `make_tensor_accessor_args_tuple<num_tensors, page_size_base_idx + num_tensors>()` + `make_tensor_accessor_tuple(args, src_addr_base_idx=3)` (reader TILE `:23-36`, reader RM `:23-35`); both donor writers construct a single two-argument `TensorAccessor(dst_args, dst_addr)`.

**Deliveries are `Buffer*`-form, not `->address()`.** The whole op contains **zero** `->address()` sites — so this is routine port work, not the silent-wrong hazard. **No offset-folded base pointer anywhere**, consistent with the audit's Offset-base-pointers gate.

**3rd-argument accessors: none.** Every accessor in play is two-argument.

### Work split

Two mutually-exclusive drivers.

- **Default**: `split_work_to_cores(compute_with_storage_grid_size, num_output_pages, /*row_major=*/false)` (`:97`)
  - `num_cores`, `all_cores`, `core_group_1`, `core_group_2`, `num_tiles_per_core_group_1`, `num_tiles_per_core_group_2`
  - core enumeration: `grid_to_cores(num_cores, num_cores_x, num_cores_y, false)` (`:243`)
- **`sub_core_grids` path** (`:55-84`, when set and output not sharded): `ncores` reduced to the largest divisor of `num_output_pages`; `all_cores = num_cores_to_corerangeset_in_subcoregrids(...)`; `core_group_1 = all_cores`, `core_group_2 = {}`, `count_per_group_1 = num_output_pages / ncores`, `count_per_group_2 = 0`
  - core enumeration: `cores_list = corerange_to_cores(sub_core_grids, ncores, false)` (`:74`)

**Crucially, the group split never reaches the kernels as a CTA.** Both branches feed a single per-core RTA (`num_pages_per_core`, `:247-248` → `:272`) on **one** reader and **one** writer `KernelDescriptor` over `all_cores`. There is no per-core-group `KernelDescriptor` multiplicity to preserve, and nothing to demote.

### Shared kernels

| kernel path | shape | census result | `_metal2` fork beside it? | rung |
|---|---|---|---|---|
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | **borrowed** | **1** other legacy binder: `data_movement/copy/device/copy_same_memory_config_program_factory.cpp:37` | **yes** — `..._metal2.cpp`, already bound by `embedding/device/embeddings_rm_program_factory.cpp:329` and `embedding/device/embeddings_tilized_indices_program_factory.cpp:231` | **1 — reuse** |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **borrowed** | **23** other legacy binders (24 factories bind the path, one of them concat); authoritative list + sunset plan at issue #52228 | **yes** — `..._metal2.cpp` beside the original under `eltwise/unary/` | **1 — reuse** |
| `device/kernels/dataflow/reader_concat_interleaved_start_id.cpp` | own | `grep -rl` → only `concat_program_factory.cpp` (+ the two `METAL2_*.md` docs) | n/a | not shared — convert in place |
| `device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp` | own | same | n/a | not shared — convert in place |

**Census disambiguation actually mattered on the RM donor.** The raw filename grep returned five extra factories; four are false positives on inspection —
`embeddings_rm_program_factory.cpp:329` and `embeddings_tilized_indices_program_factory.cpp:231` bind the **`_metal2` fork** (they are the fork's existing consumers, which is what makes it read-only to this port);
`data_movement/slice/device/slice_program_factory_rm.cpp:366` binds a different file, `slice_writer_unary_stick_layout_interleaved_start_id.cpp` (substring match);
and `experimental/quasar/slice/...` is out of bounds. Only `copy_same_memory_config_program_factory.cpp:37` is a real remaining legacy consumer.

**Fork binding vocabulary — this is now the constraint, not a choice:**

| fork | `dfb::` | `tensor::` | named args | `#ifdef`s it gates on | CTAs |
|---|---|---|---|---|---|
| RM: `.../kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp` | `out0` | `dst` | `stick_size`, `num_sticks`, `start_id` | `BACKWARDS` | none |
| TILE: `.../eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` | `out` | `dst` | `num_pages`, `start_id` | `OUT_SHARDED`, `BACKWARDS` | none |

Fit re-checked in both directions against the legacy donors and both are **total** — every value concat passes has a home, and neither fork reads anything concat lacks:

| legacy writer RTA/CTA (concat) | legacy donor slot | RM fork | TILE fork |
|---|---|---|---|
| `Buffer* dst_buffer` | RTA 0 (`dst_addr`) | → `tensor::dst` binding | → `tensor::dst` binding |
| `output.buffer()->page_size()` | RTA 1 (`stick_size`, RM only) | `stick_size` | *(absent; fork uses `dfb.get_entry_size()`, matching the legacy donor's `get_local_cb_interface(cb_id_out).fifo_page_size`)* |
| `num_pages_per_core` | RTA 2 (RM) / 1 (TILE) | `num_sticks` | `num_pages` |
| `num_pages_written` | RTA 3 (RM) / 2 (TILE) | `start_id` | `start_id` |
| `src0_cb_index` | CTA 0 | → `dfb::out0` binding | → `dfb::out` binding |
| RM `dst_buffer->page_size()` | CTA 1 — **never read** | (drops) | n/a |

Concat sets **no** `defines` on the writer, so `BACKWARDS` / `OUT_SHARDED` stay undefined in both forks — the forward, non-sharded path, byte-for-byte the legacy behaviour.

**⚠ Two forks exist for the TILE donor.** `copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` is a functionally identical second fork living in a consumer's directory, and it names its accessor `tensor::output` rather than `tensor::dst`. The canonical one — the sibling of the original, which the locational rung-1 test finds — is the `eltwise/unary/` copy, and that is the one bound here. Both are flagged for consolidation in the canonical fork's header and under issue #52228.

**No writes land in either peer directory:** rung 1 means bind and adopt; the pointer comments already exist in both legacy originals, and a fork with existing consumers is read-only.

### Flags

- **`experimental/quasar/` stayed out of bounds.** Broad greps for both donor filenames surfaced quasar hits (`quasar/slice`, `quasar/matmul`, `quasar/tilize`, …); none were read, counted for the rung-1 test, or used for naming. The rung-1 check was run **locationally** (`ls` the original's directory for a `_metal2` sibling), which is what kept them out.
- **Unreferenced kernel files in the op directory**: `writer_s2i_width.cpp` is present but the factory that would bind it (`ConcatS2IProgramFactory`) binds `reader_s2i_width.cpp`, which **does not exist anywhere in the repository** — the audit's dead-code finding. Both are outside this pass's scope and were not audited here.
- **No descriptor type outside the audit's scan** appears in this factory: `KernelDescriptor`, `CBDescriptor`, `ReaderConfigDescriptor`, `WriterConfigDescriptor` only.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — the base concept. No `override_runtime_arguments` exists anywhere in the op, so the custom concept does not apply and [Translating `override_runtime_arguments`](#) is skipped entirely.
- **Custom `compute_program_hash`**: **none** — default reflection-based hash.
- **Implementation notes**:
  - `create_descriptor` → `create_program_artifacts` on the same `ConcatProgramFactory` struct; the header's include of `<tt-metalium/program_descriptors.hpp>` is replaced by the Metal 2.0 artifact header.
  - **No pybind line to delete.** `concat_nanobind.cpp` binds only the public `ttnn::concat` free function; `create_descriptor` is not pybound. No user-visible API surface change.
  - The device-operation class is untouched: `select_program_factory` keeps dispatching to all six factories, and the `program_factory_t` variant keeps all six alternatives.
  - The factory body moves onto `MeshTensor` per the migration guide. `MeshTensor` has no `buffer()`; the two `Buffer*` queries the factory genuinely needs route through `mesh_buffer()`: `page_size()` directly, and `get_reference_buffer()->alignment()` for the alignment. `ttnn::Tensor::buffer()` **is** `mesh_buffer().get_reference_buffer()` (`ttnn/core/tensor/tensor.cpp:469` → `storage.cpp:156`), so the translation is exact, not approximate.

## Planned Spec Shape

Default is 1:1 with legacy.

- **KernelSpecs** (2 — one per legacy `KernelDescriptor`):
  - `READER` — source selected by `rm_layout`, exactly as legacy. Layout-dependent: source, `defines`, `compile_time_varargs`, and the slot-0 RTA name (the two kernels' own locals are `num_tiles` / `num_pages`).
  - `WRITER` — the `_metal2` **fork** selected by `rm_layout`.
- **DataflowBufferSpecs** (1): `SRC0_DFB`, `entry_size = single_page_size`, `num_entries = num_input_pages` (the 2→1 L1 fallback carried over — it is behaviour, not plumbing), `data_format_metadata = cb_data_format`, `tile_format_metadata` left unset (legacy `.tile` was unset). Not borrowed, not aliased.
- **SemaphoreSpecs**: none.
- **TensorParameters** (`N+1`): `input_0 … input_{N-1}` from `input_tensors[i].tensor_spec()`, `output` from the output's `tensor_spec()`. All `relaxations` default (strict) — the brief records relaxation as `none`.
- **WorkUnitSpecs** (1): `{READER, WRITER}` over `all_cores`. Both kernels ran on `all_cores` in legacy and the DFB is local to the pair, so the local-DFB invariant (producer and consumer share identical WorkUnitSpec membership) holds by construction.
- **Op-owned tensors**: none.

### DFB endpoint census — re-derived, not transcribed

Per node, for `SRC0_DFB`:

| toucher | evidence | tag |
|---|---|---|
| reader | `dfb_in.reserve_back(...)` / `push_back(...)` — TILE reader `:51,60`, RM reader `:51,90` | **locked producer** |
| donor writer | `wait_front(1)` / `pop_front(1)` — RM fork `:39,43`, TILE fork `:52,55` | **locked consumer** |

Two touchers, one locked to each role → **plain 1:1**. No self-loop, no 1P+1C assignment question, no multi-binding flag. Holds across all four config axes: `rm_layout` swaps *which* donor writer binds, but both are locked consumers; `WIDTH_CONCAT`, `sub_core_grids` and CB depth change no touch. **My census agrees with the brief.**

`get_write_ptr()` (TILE reader `:52`, RM reader `:52`) is a public peek on the producer side, not a second toucher.

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** The per-group counts (`num_tiles_per_core_group_1` / `_2`) are resolved host-side into a per-core RTA (`concat_program_factory.cpp:247-248`), not into per-group CTAs on duplicated `KernelDescriptor`s. One reader `KernelSpec` and one writer `KernelSpec` over one `WorkUnitSpec` is the faithful 1:1 translation; there is nothing to promote to CTAs and nothing to demote to RTAs.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `concat_program_factory.cpp:276` (reader RTA slots `3..3+N-1`) | `reader_kernel_args.push_back(src_buffers[j])` — `Buffer*` per input | `N` × `TensorBinding{input_i, "in<i>"}` on `READER`, plus one `TensorBindingSequence{"inputs", {in0…}}`; `N` × `TensorParameter` + `TensorArgument` |
| `concat_program_factory.cpp:285, :290` (writer RTA slot `0`) | `dst_buffer` — `Buffer*` | `TensorBinding{output, "dst"}` on `WRITER` (name fixed by both forks) |
| `concat_program_factory.cpp:200` CTA slot 0 | `src0_cb_index` (magic CB index `0`) | `DFBBinding{SRC0_DFB, "in", PRODUCER}` on `READER` |
| `concat_program_factory.cpp:214, :216` writer CTA slot 0 | `src0_cb_index` | `DFBBinding{SRC0_DFB, "out0"/"out", CONSUMER}` on `WRITER` |
| `concat_program_factory.cpp:200` CTA slot 1 | `num_input_tensors` | **gone** — the sequence carries its own length; kernel reads `std::tuple_size_v<decltype(tensor::inputs)>` (`advanced_options.hpp:138-139`) |
| `concat_program_factory.cpp:203-205` | `N` × `TensorAccessorArgs(input_i).append_to(reader_cta)` | binding mechanism end-to-end; kernel-side `make_tensor_accessor_args_tuple<...>()` + `make_tensor_accessor_tuple(args, 3)` collapse to `make_tensor_accessors(tensor::inputs)` |
| `concat_program_factory.cpp:218` | `TensorAccessorArgs(*dst_buffer).append_to(writer_cta)` | same; the forks already do `TensorAccessor(tensor::dst)`, and their `TensorAccessorArgs<1>()`/`<2>()` chains are already gone |
| `concat_program_factory.cpp:201-202` CTA slots `2..2+N-1` | `page_size_per_tensor[N]` | **RM reader**: `advanced_options.compile_time_varargs` (read at a runtime index — see below). **TILE reader**: dropped entirely; that kernel never reads it (it takes its size from the DFB) and the block existed only to offset the accessor args |
| `concat_program_factory.cpp:214` RM writer CTA slot 1 | `dst_buffer->page_size()` | **gone** — dead slot; existed only to offset the legacy donor's `TensorAccessorArgs<2>()` |
| reader RTA slots `0,1,2` | positional `get_arg_val<uint32_t>(0..2)` | named RTAs (below) |
| reader RTA slots `3+N .. 3+3N-1` | `arg_ptr[...]` raw walk off `get_arg_addr(3)` | runtime varargs (below) |
| writer RTAs | positional | the forks' named args: RM `stick_size`/`num_sticks`/`start_id`, TILE `num_pages`/`start_id` |

No semaphore-ID RTAs (no semaphores). No page-size 3rd-argument CTAs/RTAs (no 3-argument accessors). No `->address()` sites.

### Reader arguments — named vs vararg, decided per the caution

| legacy reader slot | contents | kernel access | disposition |
|---|---|---|---|
| RTA `0` | `num_pages_per_core` | `get_arg_val<uint32_t>(0)`, read once into `num_tiles` (TILE) / `num_pages` (RM) | **named** — distinct field |
| RTA `1` | `curr_tensor` | read once into `start_tensor` | **named** |
| RTA `2` | `curr_tensor_id` | read once into `start_tensor_id` | **named** |
| RTA `3+N .. 3+2N-1` | `num_pages_per_block[N]` | `arg_ptr[num_tensors + i]` inside `for (i < num_tensors)` — TILE `:40-43`, RM `:39-42` | **runtime vararg** — indexed-collection element, count not a source literal |
| RTA `3+2N .. 3+3N-1` | `page_id_per_tensor[N]` | `arg_ptr[2*num_tensors + i]`, same loop | **runtime vararg** |
| CTA `2 .. 2+N-1` (RM only) | `page_size_per_tensor[N]` | `kernel_compile_time_args[page_size_base_idx + curr_tensor]` at RM `:57` and `:70`, where `curr_tensor` is a **runtime** value | **compile-time vararg** — data-selected index |

`num_runtime_varargs = 2 * N`, laid out as `[0, N)` = `num_pages_per_block`, `[N, 2N)` = `page_id_per_tensor` — the same order the legacy `arg_ptr` walk used, so the kernel's two offsets become `i` and `num_tensors + i`.

Slots 0-2 deliberately stay named rather than being swept into the vararg block: they are distinct fixed fields, and their legacy position ahead of a variable-count block is a legacy-buffer artifact that the named/vararg section split erases. Conversely the two `N`-element blocks and the RM page-size lookup are genuine indexed-collection reads with no per-element identity to name.

**Named CTAs: zero on both readers.** Every legacy positional CTA becomes a binding, a `tuple_size`, a vararg, or nothing — so `compile_time_args` is empty on both KernelSpecs. That is the expected shape here, not an omission.

## Applied Patterns

- **[Caution: Porting a shared kernel](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)** — rung **1** (reuse an existing `_metal2` fork) on **both** borrowed donor writers. No fork created, no peer directory written to, no fork edited.
- **[Caution: Avoid varargs](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)** — three genuine vararg blocks retained (2 runtime + 1 compile-time), each an indexed-collection element; slots 0-2 named against the `arg_index`-style pull. Reported in `METAL2_PORT_REPORT.md`.
- **`KernelAdvancedOptions::TensorBindingSequence`** (`advanced_options.hpp:147-151`) — the compile-time-variadic tensor-binding mechanism, for the reader's `N` inputs accessed positionally through `make_abstract_tensor_accessor_wrappers`. Its documented purpose is exactly this shape. Not a catalog entry yet; candidate for one.
- Not applied, deliberately: no self-loop, no 1P+1C assignment, no `alias_with`, no `allow_instance_multi_binding`, no conditional DFB binding (`WIDTH_CONCAT` gates neither a binding nor a resource — it only changes the read loop, so it stays an ordinary `define`).

## Deferred / Flagged

- **`get_tile_size(cb_id_in)` at `reader_concat_interleaved_start_id.cpp:28` is declared `const`, not `constexpr`** — so per [CB→DFB whitelist §A](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md) it takes the **member getter** (`dfb_in.get_tile_size()`), not the token form. That forces a small reorder: the DFB is constructed at `:45` but the tile size is read at `:28`. The read moves down to just after the DFB construction; nothing else about the line changes. The brief flagged this one to confirm rather than swap blind.
- Recipe/doc friction found during inventory and planning is recorded in `METAL2_PORT_REPORT.md`; none of it blocked planning.
- **New findings the audit missed: none.** Every construct in this factory maps onto an inventory row above.

---

# `ConcatS2SRMProgramFactory`

## Legacy Inventory

### Legacy factory shape

- **Concept**: `ProgramDescriptorFactoryConcept` — `static ProgramDescriptor create_descriptor(...)` (`concat_s2s_rm_program_factory.hpp:14`). In the `program_factory_t` variant; not the direct-descriptor shape.
- **Variants**: single factory, one config axis — whether `num_output_rows_per_core_last > 0` (`concat_s2s_rm_program_factory.cpp:190`), which splits the grid into a first/last core group.
- **Custom `compute_program_hash`**: none (op-wide).

### Kernels

**One** kernel source, pushed into **two or four** `KernelDescriptor`s. Selected only when `dim == 3`, both inputs height/width-sharded (not block, not ND), exactly 2 inputs, ROW_MAJOR, widths divisible by `groups` (`concat_device_operation.cpp:47-52`).

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader (×1 or ×2) | `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors.cpp` (`:170-172`) | `all_cores`, or `first_cores` / `last_cores` | `compile_time_args_0` / `_0_last` (14 elements, `:104-149`) | **none** | none | unset → `O2` (DM) | `ReaderConfigDescriptor{}` |
| writer (×1 or ×2) | **the same source** (`:180-182`) | same as its paired reader | `compile_time_args_1` / `_1_last` (`:119-164`) | **none** | none | unset → `O2` (DM) | `WriterConfigDescriptor{}` |

The 14 positional CTAs, and where each goes:

| slot | legacy value | kernel local (`reader_height_sharded_width_concat_two_tensors.cpp`) | disposition |
|---|---|---|---|
| 0 | `cb_dst_id` (16) | `output_dfb_id` (`:14`) | → `DFBBinding` |
| 1 | `input_0_stick_size` | `input_stick_size_0` **`constexpr`** (`:16`) | named CTA |
| 2 | `input_1_stick_size` | `input_stick_size_1` **`constexpr`** (`:17`) | named CTA |
| 3 | `input_0_stride` | `input_stride_0` **`constexpr`** (`:18`) | named CTA |
| 4 | `input_1_stride` | `input_stride_1` **`constexpr`** (`:19`) | named CTA |
| 5 | `num_output_rows_per_core * N` | `num_output_pages` `const` (`:21`) — **never used** | named CTA (dead local preserved) |
| 6 | per-group | `page_start` `const` (`:22`) | named CTA |
| 7 | per-group | `page_end` `const` (`:23`) | named CTA |
| 8 | per-group | `output_stick_offset` `const` (`:24`) | named CTA |
| 9 | per-group | `input_start_0` `const` (`:25`) | named CTA |
| 10 | per-group | `input_start_1` `const` (`:26`) | named CTA |
| 11 | `groups` | `groups` **`const`** (`:28`) | named CTA |
| 12 | `cb_ids[0]` (0) | `input_dfb_0_id` (`:29`) | → `DFBBinding` |
| 13 | `cb_ids[1]` (1) | `input_dfb_1_id` (`:30`) | → `DFBBinding` |

**`groups` is `const`, not `constexpr`, and four `constexpr` initialisers depend on it** (`:32-35`). This is legal today because `get_compile_time_arg_val(11)` is a constant expression, so an integral `const` initialised from it is itself usable in constant expressions. `get_arg(args::groups)` on a `CtaVal` is `constexpr` too, so the ported line preserves that property exactly — the declaration keeps its `const` and lines 32-35 keep compiling. **The `constexpr`-vs-`const` distinction that decides token-form versus member-getter is a DFB-*metadata* rule; it does not apply to argument reads.** Each declaration's cv-qualifier is carried across verbatim regardless.

`grep -n opt_level` → zero lines. Both descriptors DM ⇒ resolved `O2` ⇒ Metal 2.0's default. Nothing to carry.

### CBs

Three `CBDescriptor`s, **all with `.buffer` set — all borrowed memory**.

| index | total_size | core_ranges | data_format | page_size | tile | `.buffer` |
|---|---|---|---|---|---|---|
| 0 | `num_input_units * input_page_size` | `all_cores` | `dfb_data_format` | `round_up_to_mul32(shard.shape[1] * element_size)` | unset | `input_tensors[0]` (`:69`) |
| 1 | same, for input 1 | `all_cores` | `dfb_data_format` | same | unset | `input_tensors[1]` (`:69`) |
| 16 | `num_output_units * output_page_size` | `all_cores` | `dfb_data_format` | `round_up_to_mul32(out shard.shape[1] * element_size)` | unset | `output` (`:86`) |

No GlobalCircularBuffer. `total_size == page_size * num_units` in every case, so `entry_size` / `num_entries` reproduce the legacy sizes exactly.

### Semaphores / Tensor accessors / Work split

- **Semaphores**: none.
- **Tensor accessors**: **none** — no kernel constructs a `TensorAccessor`. Tensor data is reached entirely through the borrowed buffers' read/write pointers.
- **Work split**: not a `split_work_to_cores` call. `all_cores` comes from `input_tensors[0].shard_spec().value().grid` (`:50`); when `num_output_rows % num_output_rows_per_core > 0` the core list is split into all-but-last / last (`:192-195`), and the per-RISC halves come from `tt::div_up(num_output_rows_per_core, 2)`.

### Shared kernels

**none.** `grep -rn reader_height_sharded_width_concat_two_tensors.cpp ttnn/cpp/ttnn/operations/` returns exactly one binder — this factory. No other concat factory and no peer op binds it, so it converts in place with no fork.

### Flags

- `num_output_pages` (CTA slot 5) is read into a local the kernel never uses. Preserved as a named CTA — dropping it would be a scope-exceeding cleanup, and it costs one word.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: no pybind surface, no device-op-class edit. `MeshTensor` covers everything this factory reads (`shard_spec()`, `element_size()`, `padded_shape()`, `dtype()`, `device()`); the legacy `.buffer()` calls disappear with the borrows rather than needing a translation.

## Planned Spec Shape

- **KernelSpecs**: 2 (`reader`, `writer`) in the common case; **4** (`reader_first`/`writer_first`, `reader_last`/`writer_last`) when the last-core group is non-empty. All four are the same source with per-group `compile_time_args`.
- **DataflowBufferSpecs**: 3, each `borrowed_from` its `TensorParameter` — `input_0`, `input_1`, `output`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: 3. **None is bound on a `KernelSpec`** — they exist because the DFBs borrow from them, which the validator counts as a use in its own right.
- **WorkUnitSpecs**: 1 over `all_cores`, or 2 over the disjoint `first_cores` / `last_cores`.
- **Op-owned tensors**: none.

### DFB endpoint census — re-derived, not transcribed

The kernel contains **no** `reserve_back`, `push_back`, `wait_front` or `pop_front` anywhere. Every touch is a raw cursor peek: `output_dfb.get_write_ptr()` (kernel `:42`), `input_dfb_0.get_read_ptr()` (`:45`), `input_dfb_1.get_read_ptr()` (`:70`).

Per node, the one source runs **twice** — a `ReaderConfigDescriptor` instance and a `WriterConfigDescriptor` instance over the *same* `core_ranges` (`:166-188`) — so each of the three DFBs has exactly **2 touchers, both role-free**. Two role-free touchers is exactly enough to fill the validator's ≥1-producer / ≥1-consumer requirement: **assign 1P+1C**, one instance producer and the other consumer, on all three. The labels drive FIFO machinery this kernel never invokes, so on Gen1 they are cosmetic and the kernel body is untouched by the choice. **Not** a self-loop (that is the one-toucher resolution) and **not** the multi-binding flag (that needs ≥3 touchers or two kernels locked to the same role). **My census agrees with the brief.**

**The two shapes stack, and the disposition does not flip.** Across the groups the node sets are *disjoint*, so each node sees one pair; within each node there are still two touchers. Census per node is 2 in both configs.

## Preserved Multiplicity

```
Legacy KernelDescriptors [reader_first, writer_first] of reader_height_sharded_width_concat_two_tensors.cpp
  → KernelSpecs [reader_first, writer_first] of same source
  → in WorkUnitSpecs [main_first]  (target_nodes = first_cores)
  → sharing DFBs: input_0 (P/C), input_1 (P/C), output (P/C)

Legacy KernelDescriptors [reader_last, writer_last] of the same source
  → KernelSpecs [reader_last, writer_last]
  → in WorkUnitSpecs [main_last]  (target_nodes = last_cores)
  → sharing DFBs: input_0 (P/C), input_1 (P/C), output (P/C)
```

The per-group values (`num_output_pages`, `page_start`, `page_end`, `output_stick_offset`, `input_start_0/1`) stay **compile-time** args on their own `KernelSpec`s. Moving them to `runtime_arg_names` would be the demoting-per-group-CTA anti-pattern — it costs the kernel's compile-time loop unrolling on `page_start`/`page_end` for no benefit.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `:105, :120, :136, :151` CTA slot 0 | `cb_dst_id` = 16 (magic index) | `DFBBinding{output, "output", …}` |
| `:117-118` etc. CTA slots 12-13 | `cb_ids[0]`, `cb_ids[1]` = 0, 1 | `DFBBinding{input_0, "input_0", …}`, `DFBBinding{input_1, "input_1", …}` |
| `:69`, `:86` | `CBDescriptor::buffer = <tensor>.buffer()` | `DataflowBufferSpec::borrowed_from = <TensorParameter>` |
| `:52-53, :71` | the `cb_ids` vector that carried indices into the CTA lists | gone with the indices |
| CTA slots 1-11 | positional | named CTAs, one per kernel-side local |

No buffer-address RTAs (there are no RTAs at all). No `TensorAccessorArgs` plumbing. No page-size 3rd-argument CTAs. No semaphore-ID RTAs.

## Applied Patterns

- **[Two-toucher DFB → assign 1P+1C (dual-instance work-split)](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)** — on all three DFBs. This is the entry's canonical shape: one source, Reader-config + Writer-config, one grid, sync-free raw touches.
- **Disjoint-node work-split** (the [demoting-CTA anti-pattern's](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) correct port) — two `WorkUnitSpec`s over disjoint node sets, each with its own same-source pair and its own CTAs.
- Borrowed-memory DFBs via `borrowed_from` (all three).
- Not applied: no self-loop, no `alias_with`, no multi-binding flag, no conditional binding, no varargs.

## Deferred / Flagged

- **New findings: none.** Every construct maps onto an inventory row above.

---

# `ConcatS2STiledProgramFactory`

## Legacy Inventory

### Legacy factory shape

- **Concept**: `ProgramDescriptorFactoryConcept` — `static ProgramDescriptor create_descriptor(...)` (`concat_s2s_tiled_program_factory.hpp:14`). In the `program_factory_t` variant.
- **Variants**: single factory. Two `defines` axes (`BF8`, `USE_SINGLE_PACKET_READ`) that change kernel arithmetic, not the resource set.
- **Custom `compute_program_hash`**: none.

### Kernels

Three `KernelDescriptor`s, **one shared 14-element positional CTA list handed to all three** (`:190-205`, `:221`, `:231`, `:240`), all over `all_cores`.

| unique_id | source | CTAs read by *this* kernel | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors_tiled.cpp` | slots 0-4 (DFB indices), 7-12 | none | `BF8`, `USE_SINGLE_PACKET_READ` (`:207-213`) | unset → `O2` (DM) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_height_sharded_width_concat_two_tensors_tiled.cpp` | slots 5-6 (DFB indices), 7-12 | none | **none** | unset → `O2` (DM) | `WriterConfigDescriptor{}` |
| compute | `device/kernels/compute/height_sharded_width_concat_two_tensors.cpp` | slots 0-6 (DFB indices), 7-13 | none | **none** | **unset → resolved `O3`** (`ComputeConfigDescriptor`) | `ComputeConfigDescriptor{HiFi4, fp32_dest_acc_en=computed, math_approx_mode=false}` (`:241-246`) |

**`opt_level` is the one that needed action.** `grep -n opt_level` over the legacy factory printed **nothing**, and the reflex reading is "nothing to do." It isn't: a genuinely absent `KernelDescriptor::opt_level` resolves at lowering the way legacy did — `O2` for a DM descriptor but **`O3` for a `ComputeConfigDescriptor`** — while Metal 2.0's single type-agnostic `CompilerOptions` defaults to `O2` for both. So the compute `KernelSpec` gets an **explicit `KernelBuildOptLevel::O3`**; the two DM specs need nothing. This is an absent line with nothing on either side of a diff to compare, which is why it is checked mechanically rather than by reading: post-port, `grep -nE 'opt_level'` over the three ported factories prints exactly one line, and the port builds exactly one compute `KernelSpec`. One line, one spec — paired.

The 14 shared CTAs:

| slot | legacy value | reader local | writer local | compute local | disposition |
|---|---|---|---|---|---|
| 0 | `0` | `input0_dfb_id` | `input0_cb_id` (**unused**) | `input0_cb_id` | → `DFBBinding` |
| 1 | `1` | `input1_dfb_id` | `input1_cb_id` (**unused**) | `input1_cb_id` | → `DFBBinding` |
| 2 | `cb_input0_transpose_id` | `input0_transpose_dfb_id` | (**unused**) | `input0_transpose_cb_id` | → `DFBBinding` |
| 3 | `cb_input1_transpose_id` | `input1_transpose_dfb_id` | (**unused**) | `input1_transpose_cb_id` | → `DFBBinding` |
| 4 | `cb_concat_id` | `concat_dfb_id` | (**unused**) | `concat_cb_id` | → `DFBBinding` |
| 5 | `cb_output_transpose_id` | (**unused**) | `output_transpose_dfb_id` | `output_transpose_cb_id` | → `DFBBinding` |
| 6 | `cb_output_id` | (**unused**) | `output_dfb_id` | `output_dfb_id` (**unused — dead local**) | → `DFBBinding` |
| 7-10 | per-shard tile counts | used | 7,8,10 used; 9 unused | used | named CTAs |
| 11 | `tile_size` | used | used | declared, **unused** | named CTA |
| 12 | `groups` | used | used | declared, **unused** | named CTA |
| 13 | `batch_size` | not declared | not declared | `MAX_BATCH_SIZE`, used | named CTA (**compute only**) |

Named arguments let each kernel declare only what it reads, so the shared positional list becomes six named CTAs common to all three plus `max_batch_size` on compute alone. The DFB-index slots become bindings on the kernels that actually touch each buffer.

**Seven dead DFB-index locals across the three kernels** (marked *unused* above) become **forced deletions**, not discretionary cleanup: a kernel that does not bind a DFB gets no `dfb::<name>` token, so a surviving `DataflowBuffer x(dfb::y)` or `constexpr uint32_t x = dfb::y` would not compile. The dead *scalar* CTAs (writer slot 9, compute slots 11-12) have no such forcing, so they are preserved as named args and unused locals exactly as legacy had them.

### CBs

Seven `CBDescriptor`s, all over `all_cores`, all with `tile` unset. `tile_size` = `tt::tile_size(data_format)` where `data_format` is input[0]'s (both inputs asserted equal, `:86`).

| index | name | total_size | data_format | page_size | `.buffer` |
|---|---|---|---|---|---|
| 0 | input0 | `total_num_tiles(in0) * tile_size` | in0 dtype | `tile_size(in0 dtype)` | `input_tensors[0]` (`:102`) |
| 1 | input1 | `total_num_tiles(in1) * tile_size` | in1 dtype | `tile_size(in1 dtype)` | `input_tensors[1]` (`:102`) |
| 2 | output | `total_num_output_tiles * tile_size` | output dtype | `tile_size(output dtype)` | `output` (`:116`) |
| 3 | input0_transpose | `in0_total_tiles_width * dfb_tile_size` | `dfb_data_format` | `dfb_tile_size` | — |
| 4 | input1_transpose | `in1_total_tiles_width * dfb_tile_size` | `dfb_data_format` | `dfb_tile_size` | — |
| 5 | concat | `out_total_tiles_width * dfb_tile_size` | `dfb_data_format` | `dfb_tile_size` | — |
| 6 | output_transpose | `out_total_tiles_width * tile_size` | `data_format` | `tile_size` | — |

Index 2 is the one where `total_size` and `page_size` are computed from *different* formats — total from the input's `tile_size`, page from the output's. `entry_size * num_entries` must reproduce the legacy `total_size` exactly, so this needed checking rather than assuming: `compute_output_specs` builds the output `TensorLayout` from `ref_in_tensor.dtype()` (`concat_device_operation.cpp:178-179`), so the output dtype **always** equals input[0]'s and the two tile sizes are the same number. `entry_size = tile_size(output dtype)`, `num_entries = total_num_output_tiles`.

`dfb_data_format` / `dfb_tile_size` (legacy `cb_data_format` / `cb_tile_size`) are bf16 when the inputs are `BFLOAT8_B`, else the input format (`:119-125`).

### Semaphores / Tensor accessors / Work split

- **Semaphores**: none.
- **Tensor accessors**: **none** in any of the three kernels. All tensor access is through the three borrowed buffers.
- **Work split**: none — `all_cores` is `input_tensors[0].shard_spec().value().grid` (`:59`) and every kernel covers it. No per-group split, no multiplicity.

### Shared kernels

**none.** All three sources have exactly one binder each (this factory), confirmed by filename census across `ttnn/cpp/ttnn/operations/`. No fork, no peer-directory write.

### Flags

- The `TODO` at `:174-176` (skip the transpose when both widths divide evenly by `groups`) is carried across verbatim — it is a note about kernel logic, not plumbing.
- Compute slots 11-12 (`tile_size`, `groups`) and writer slot 9 (`input1_num_tiles_height`) are read into locals those kernels never use. Preserved.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: no pybind surface, no device-op-class edit. The four `TT_FATAL` blocks at the top of the factory and the two inside `get_num_tiles_per_shard` are carried across unchanged (message text included).

## Planned Spec Shape

- **KernelSpecs**: 3 — `reader`, `writer`, `compute`.
- **DataflowBufferSpecs**: 7 — three `borrowed_from` (`input_0`, `input_1`, `output`), four plain L1 scratch.
- **SemaphoreSpecs**: none.
- **TensorParameters**: 3, none bound on a `KernelSpec` (borrow-only, as in the RM factory).
- **WorkUnitSpecs**: 1 — `{reader, writer, compute}` over `all_cores`.
- **Op-owned tensors**: none.

### DFB endpoint census — re-derived, not transcribed

Per node, from the kernel bodies:

| DFB | reader | writer | compute | touchers | disposition |
|---|---|---|---|---|---|
| input0 | `push_back` (reader `:49`) → **locked P** | — | `wait_front`/`pop_front` via `transpose()` (compute `:13,30`) → **locked C** | 2 | plain 1:1 |
| input1 | `push_back` (reader `:50`) → **locked P** | — | same → **locked C** | 2 | plain 1:1 |
| input0_transpose | `wait_front`/`pop_front` (reader `:56,94`) → **locked C** | — | `reserve_back`/`push_back` → **locked P** | 2 | plain 1:1 |
| input1_transpose | `wait_front`/`pop_front` (reader `:98,137`) → **locked C** | — | `reserve_back`/`push_back` → **locked P** | 2 | plain 1:1 |
| concat | `reserve_back`/`push_back` (reader `:53,140`) → **locked P** | — | `wait_front`/`pop_front` → **locked C** | 2 | plain 1:1 |
| output_transpose | — | `wait_front`/`pop_front` (writer `:39,56`) → **locked C** | `reserve_back`/`push_back` → **locked P** | 2 | plain 1:1 |
| **output** | — | `reserve_back`/`push_back` (writer `:38,57`) → **locked P**; `get_write_ptr()` (`:35`) is a peek, not a second toucher | **not a toucher** | **1** | **self-loop** |

**The output buffer's census turned on reading the compute kernel's body, not its declarations.** Compute constructs `DataflowBuffer output_dfb(output_dfb_id)` at `height_sharded_width_concat_two_tensors.cpp:57` (pre-port numbering) and **never uses it** — a dead local. Binding output on compute to satisfy that construction would manufacture a two-toucher where the code has one, and would turn a correct self-loop into a wrong 1P+1C. So: the writer is bound as **both** PRODUCER and CONSUMER of `output` (one accessor name, two bindings), and compute gets no output binding. On Gen1 the buffer lowers to a plain circular buffer that one RISC both fills and drains, so this is behaviour-identical to the legacy CB.

Consequently the dead local **must be deleted** — without a binding there is no `dfb::output` token in compute's generated header, so the line cannot compile. The brief suggested leaving it for the ops team; that is not available, and the deletion is zero-functional-change because the local was dead. Recorded in the port report.

**My census agrees with the brief on all seven.** No DFB reaches ≥3 touchers or has two kernels locked to the same role, so the multi-binding flag is set nowhere.

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** One `KernelDescriptor` per role, all over one `core_ranges`.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `:191-197` CTA slots 0-6 | seven magic buffer indices (`0`, `1`, `cb_input0_transpose_id`, `cb_input1_transpose_id`, `cb_concat_id`, `cb_output_transpose_id`, `cb_output_id`) | seven `DFBBinding`s, distributed to the kernels that touch each buffer |
| `:102`, `:116` | `CBDescriptor::buffer = <tensor>.buffer()` (×3) | `DataflowBufferSpec::borrowed_from` (×3) |
| `:190-205` | one positional list shared by all three kernels | per-kernel named `compile_time_args`; slot 13 goes to compute only |
| compute `:57` (pre-port) | `DataflowBuffer output_dfb(output_dfb_id)` — dead local | deleted (forced: no binding ⇒ no token) |
| reader `:19-20`, writer `:15-19` (pre-port) | dead DFB-index locals | deleted (same forcing) |

No RTAs at all. No `TensorAccessorArgs` plumbing, no accessors, no page-size 3rd arguments, no semaphore-ID RTAs, no `->address()` sites.

## Applied Patterns

- **[Sync-free and single-ended CBs → self-loop DFB](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)** — the `output` buffer. The single-ended shape with a real endpoint: the writer is a genuine FIFO producer, and it is the sole toucher, so it self-loops. A **DM** self-loop, legal on Gen1; Gen2 rejects DM self-loops, which is Quasar-uplift's concern and needs no tracking here (the declarative binding makes it trivially greppable post-port).
- **[Pass DFB handles directly to LLKs](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)** — `dfb::name` flows straight into `compute_kernel_hw_startup`, `transpose_init`, `transpose_tile`, `pack_tile`, `reconfig_data_format_srca`, `pack_reconfig_data_format`, and into the kernel's own local `transpose<>` helper's `uint32_t` parameters. No `.id`, no temp wrapper.
- Borrowed-memory DFBs via `borrowed_from` (three).
- Not applied: no `alias_with`, no multi-binding flag, no conditional binding, no varargs, no 1P+1C assignment (every two-toucher here is a genuine locked-role pair).

## Deferred / Flagged

- **`unpack_modes` needed a newly-required entry the legacy config did not have.** The validator requires an explicit unpack mode for every Float32 DFB a compute kernel consumes while `enable_32_bit_dest` is on; the legacy `ComputeConfigDescriptor` left `unpack_to_dest_mode` empty, i.e. `Default` on every buffer, which translates to `UnpackMode::UnpackToSrc`. Compute consumes `input0`, `input1` and `concat`; when `data_format == Float32` all three carry that format (an `is_bf8` input cannot also be Float32), so all three get an explicit `UnpackToSrc`. The value is *derived* from the legacy vector, not guessed — reversing it to `UnpackToDest` would flip the precision/perf tradeoff with no compile or test signal. Entries are emitted only in the Float32 case; Int32/UInt32 also set `enable_32_bit_dest` but the required-entry rule is deliberately Float32-only for now, so no entries are added there.
- **The compute config is Style B** (a Metal `ComputeConfigDescriptor` set directly, no TTNN `ComputeKernelConfig` resolved), so the Gen1 config is built by hand and **not** routed through `to_compute_hardware_config` — that helper's defaults are the high-performance ones and any field not explicitly copied would flip. `ComputeGen1Config`'s defaults were checked field by field against the legacy descriptor's and they coincide: `math_approx_mode=false` → `sfpu_precision_mode=Precise`, unset `dst_full_sync_en=false` → `double_buffer_dest=true`, unset `bfp8_pack_precise=false` → `bfp_pack_precision_mode=Approximate`. Only `fpu_math_fidelity` (explicit in legacy) and `enable_32_bit_dest` are set.
- **New findings: none.**

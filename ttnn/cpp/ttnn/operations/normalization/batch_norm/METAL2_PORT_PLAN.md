# Port Plan — `normalization/batch_norm` (`BatchNormOperation` + `RunningStatistics`)

Port plan for the two device-operations in `ttnn/cpp/ttnn/operations/normalization/batch_norm/`, ported
from `ProgramDescriptorFactoryConcept` (`create_descriptor`) to Metal 2.0 `ProgramSpecFactoryConcept`
(`create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

**Audit inputs:** `METAL2_PREPORT_AUDIT.md` (GREEN, both device-ops) and `METAL2_PORT_BRIEF.md`.

**Two independent porting units, one PR** (as the invoker requested). Each unit is
*one factory + the 4 kernel entry points it can bind* (reader, writer, and **both** runtime-selected
compute sources). 8 kernel files total. They share no kernels and no factories; nothing couples them
structurally. Sequenced below as Unit 1 (BatchNorm) then Unit 2 (RunningStatistics).

---

## Legacy Inventory

### Legacy factory shape

- Concept: **`ProgramDescriptorFactoryConcept`** — both factories implement
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`
  (`batch_norm_device_operation.hpp:39`, `running_statistics_device_operation.hpp:36`).
- Variants: **single** per device-op — `program_factory_t` is a one-alternative `std::variant` in both
  (`batch_norm_device_operation.hpp:45`, `running_statistics_device_operation.hpp:42`).
  *Not* a multi-variant factory: the runtime kernel-source selection (below) is a per-source
  `KernelSpec` choice inside one spec, not a per-variant spec.
- Custom `compute_program_hash`: **none** in either device-op — already the default reflection-based
  hash (`grep compute_program_hash` over the directory: zero hits). Nothing for the port to delete.
- **`to_hash()` backdoor — present on `BatchNormOperation::operation_attributes_t` only**
  (`batch_norm_device_operation.hpp:22`, `batch_norm_device_operation.cpp:121-123`). This is a
  *different* mechanism from `compute_program_hash`; the audit analysed it as harmless (it narrows only
  the attributes half of the key; `tensor_args`, hence `TensorSpec`, is hashed separately). It is
  device-op-class code and **off-limits — the port does not touch it.**

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN
factory analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section
below.)*

---

### Variant: `BatchNormOperation` / `BatchNormFactory`

Factory body: `batch_norm_program_factory.cpp:140-416`; per-core RTA loop in the file-local helper
`populate_runtime_arguments` (`:25-134`).

#### Kernels

`all_device_cores` below is `CoreRangeSet(CoreRange({0,0}, {grid.x-1, grid.y-1}))`
(`batch_norm_program_factory.cpp:188`) — i.e. **every** device core, not the working set.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_batch_norm.cpp` | `all_device_cores` | `[0]=input_tensor_cb(c_0)`, `[1]=eps_cb(c_4)`, `[2..]=TensorAccessorArgs(input)`, `[next]=any_float32` (`:303-308`) | none | 11 per core, **9 read** (`:87-99`) | none | none | **`O2`** (field unset → DM default) | `ReaderConfigDescriptor{}` (`:337`) |
| writer | `kernels/dataflow/writer_batch_norm.cpp` | `all_device_cores` | `[0]=weight_has_value`, `[1]=bias_has_value`, `[2]=batch_mean_cb(c_1)`, `[3]=writer_output_cb(c_2 \| c_9)`, `[4]=batch_var_cb(c_3)`, `[5]=weight_cb(c_5)`, `[6]=bias_cb(c_6)`, then `TensorAccessorArgs` ×5 (batch_mean, output, batch_var, weight, bias), then `batch_stat_is_fp32`, `param_is_fp32` (`:310-328`) | none | 14 per core, **12 read** (`:109-124`) | none | none | **`O2`** (unset → DM default) | `WriterConfigDescriptor{}` (`:346`) |
| compute | `kernels/compute/batch_norm_sfpu_kernel.cpp` **or** `kernels/compute/batch_norm_kernel.cpp` — selected at runtime on `(fp32_dest_acc_en \|\| any_float32)` (`:388-390`) | `all_device_cores` | 15 values (`:370-385`): `weight_has_value`, `bias_has_value`, `input_cb`, `batch_mean_cb`, `output_cb`, `batch_var_cb`, `eps_cb`, `den_cb`, `weight_cb`, `temp_1_cb`, `bias_cb`, `writer_output_cb`, `needs_output_typecast`, `tc_in_fmt(Float32)`, `tc_out_fmt` | none | 3 per core (`:129-130`) | none | none | **`O3`** (field unset → `ComputeConfigDescriptor` default; **Metal 2.0 defaults to `O2`, so this must be stated explicitly**) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, math_approx_mode}` (`:394-400`) |

`grep -n opt_level` over the directory returns **zero hits** — confirming both resolved levels above.

#### CBs

10 descriptors, one conditional. No `CBFormatDescriptor::tile` is set anywhere → every
`tile_format_metadata` stays `nullopt`. `num_tiles_per_cb = 2` throughout (`:191-192`;
`b_num_tiles_per_cb` is a redundant alias of it — audit "Misc anomalies", cosmetic, left alone).

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` input (`:196-204`) | `a_tile × 2` | all | `a_data_format` (input dtype) | `a_tile` | — |
| `c_1` batch_mean (`:206-214`) | `b_tile × 2` | all | `b_data_format` | `b_tile` | — |
| `c_2` output_0 (`:216-224`) | `(typecast ? interm : c)_tile × 2` | all | `typecast ? interm : c` | same | — |
| `c_9` writer_out (`:227-239`) **conditional: `needs_output_typecast`** | `c_tile × 2` | all | `c_data_format` | `c_tile` | — |
| `c_3` batch_var (`:240-249`) | `d_tile × 2` | all | `d_data_format` | `d_tile` | — |
| `c_4` eps (`:250-259`) | `interm_tile × 2` | all | `interm_data_format` | `interm_tile` | — |
| `c_5` weight (`:260-269`) | `e_tile × 2` | all | `e_data_format` (`Float16_b` when absent) | `e_tile` | — |
| `c_6` bias (`:270-279`) | `f_tile × 2` | all | `f_data_format` (`Float16_b` when absent) | `f_tile` | — |
| `c_7` den (`:282-291`) | `interm_tile × 2` | all | `interm_data_format` | `interm_tile` | — |
| `c_8` temp_1 (`:292-301`) | `interm_tile × 2` | all | `interm_data_format` | `interm_tile` | — |

No `GlobalCircularBuffer` (zero hits), no `address_offset`, no `set_globally_allocated_address`,
no multi-element `format_descriptors` (so no aliased CBs).

#### Semaphores

none — neither factory declares a `SemaphoreDescriptor`.

#### Tensor accessors

6 sites, all the **2-arg** `TensorAccessor(args, addr)` form (no page-size 3rd argument anywhere).
All six addresses arrive as a `Buffer*` pushed into `emplace_runtime_args` — never `->address()`, so
there is no host-side offset fold.

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `batch_norm_program_factory.cpp:307` (CTAs) + `:90` (addr) | `tensor_args.input` | reader slot 1 |
| `:319` + `:111` | `tensor_args.batch_mean` | writer slot 0 |
| `:320` + `:115` | `tensor_return_value` (output) | writer slot 4 |
| `:321` + `:112` | `tensor_args.batch_var` | writer slot 1 |
| `:322-323` + `:101-104` | `tensor_args.weight` *(optional; literal `0u` when absent)* | writer slot 2 |
| `:324` + `:105-108` | `tensor_args.bias` *(optional; literal `0u` when absent)* | writer slot 3 |

Kernel-side accessor construction: `reader_batch_norm.cpp:38`; `writer_batch_norm.cpp:53, 57, 61, 65, 69`.

#### Work split

- Driver: `tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, /*row_major=*/true)`
  (`batch_norm_program_factory.cpp:57`)
- num_cores: the returned `num_cores` is **discarded** (`_unused_num_cores`); the loop walks
  `num_cores_total = grid.x * grid.y` cores from `grid_to_cores(...)` (`:59-64`)
- core_group_1: `num_tiles_per_core_group_1` tiles per core
- core_group_2: `num_tiles_per_core_group_2` tiles per core
- cores in **neither** group: an **all-zero RTA vector** is emitted (`:72-79`) rather than narrowing
  `core_ranges`. Kernel placement is all device cores.

**The split result drives per-core RTA values only** — both core groups get *identical* CTAs and there
is a single `KernelDescriptor` per kernel. No per-group CTA multiplicity.

#### Shared kernels

**none.** All four kernel sources live in this directory and `grep -rl <filename> ttnn/cpp/ttnn/operations/`
returns hits only inside `normalization/batch_norm/`. No `_metal2` fork is needed and no peer-directory
write is required. Coupling is limited to three donor **headers** (out-of-scope, cross cleanly):
`eltwise/binary_ng/.../fill_tile_utils.hpp` (raw `uint32_t l1_write_ptr`),
`ttnn/kernel/dataflow/cb_fill_helpers.hpp` (`uint32_t cb_id`),
`ttnn/kernel/compute/dest_format_helpers.hpp` (`uint32_t` cb ids).

#### Runtime kernel-source selection

Single axis: `(fp32_dest_acc_en || any_float32) ? "sfpu_kernel" : "kernel"` (`:388-390`). Both sources
convert with the factory — 4 entry points in this unit. No further fan-out (no layout axis, no
broadcast axis). The two sources differ in CTA arity (SFPU reads CTAs 0–14; non-SFPU reads 0–10) and in
the presence of the typecast stage, so they get **separate `KernelSpec`s** with different
`compile_time_args`, `defines`, and DFB bindings.

DFB producer/consumer roles are **identical across the two sources** for every DFB they share — the
non-SFPU source simply never touches `writer_out` (`c_9`) nor consumes `out` (`c_2`), and it cannot:
`needs_output_typecast ⇒ interm == Float32 ⇒ any_float32 ⇒ the SFPU source is selected`.

---

### Variant: `RunningStatistics` / `RunningStatisticsProgramFactory`

Factory body: `running_statistics_program_factory.cpp:137-466`; per-core RTA loop in
`populate_runtime_arguments` (`:25-131`).

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_running_statistics.cpp` | `all_device_cores` (`:185`) | `[0]=batch_mean_cb(c_0)`, `[1]=momentum_cb(c_5)`, `[2]=one_cb(c_6)`, `[3..]=TensorAccessorArgs(batch_mean)`, `[next]=any_float32` (`:346-352`) | none | 11 per core, **9 read** (`:85-97`) | none | none | **`O2`** (unset) | `ReaderConfigDescriptor{}` (`:379`) |
| writer | `kernels/dataflow/writer_running_statistics.cpp` | `all_device_cores` | `[0]=running_mean_has_value`, `[1]=running_var_has_value`, `[2]=batch_var_cb(c_1)`, `[3]=output_cb(c_2)`, `[4]=old_running_mean_cb(c_3)`, `[5]=old_running_var_cb(c_4)`, `[6]=writer_updated_m_cb(c_7 \| c_12)`, `[7]=writer_updated_v_cb(c_8 \| c_13)`, then `TensorAccessorArgs` ×4 (batch_var, output, running_mean, running_var), then `old_stat_is_fp32` (`:354-370`) | none | 13 per core, **11 read** (`:107-121`) | none | none | **`O2`** (unset) | `WriterConfigDescriptor{}` (`:388`) |
| compute | `kernels/compute/running_statistics_sfpu_kernel.cpp` **or** `kernels/compute/running_statistics_kernel.cpp` — runtime-selected on `(fp32_dest_acc_en \|\| any_float32)` (`:438-440`) | `all_device_cores` | 19 values (`:416-435`): `running_mean_has_value`, `running_var_has_value`, `batch_mean_cb`, `batch_var_cb`, `output_cb`, `old_running_mean_cb`, `old_running_var_cb`, `updated_m_cb`, `updated_v_cb`, `momentum_cb`, `one_cb`, `tmp1_cb`, `tmp2_cb`, `tmp3_cb`, `writer_updated_m_cb`, `writer_updated_v_cb`, `stat_format_needs_typecast`, `tc_in_fmt(Float32)`, `tc_out_fmt` | none | 3 per core, **1 read** (`:126-127`) | none | none | **`O3`** (unset → `ComputeConfigDescriptor` default) | `ComputeConfigDescriptor{...}` (`:444-450`) |

#### CBs

14 descriptors, two conditional. No `tile` field set anywhere. `num_tiles_per_cb = 2` throughout.

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` batch_mean (`:193-201`) | `a_tile × 2` | all | `a_data_format` | `a_tile` | — |
| `c_1` batch_var (`:203-211`) | `b_tile × 2` | all | `b_data_format` | `b_tile` | — |
| `c_2` output (`:213-221`) | `c_tile × 2` | all | `c_data_format` | `c_tile` | — |
| `c_3` old_running_mean (`:223-231`) | `d_tile × 2` | all | `d_data_format` (`Float16_b` when absent) | `d_tile` | — |
| `c_4` old_running_var (`:233-241`) | `e_tile × 2` | all | `e_data_format` (`Float16_b` when absent) | `e_tile` | — |
| `c_5` momentum (`:243-251`) | `interm_tile × 2` | all | `interm_data_format` | `interm_tile` | — |
| `c_6` one (`:253-261`) | `interm_tile × 2` | all | `interm_data_format` | `interm_tile` | — |
| `c_7` updated_m (`:263-271`) | `(mean_typecast ? interm : d)_tile × 2` | all | `mean_typecast ? interm : d` | same | — |
| `c_8` updated_v (`:273-281`) | `(var_typecast ? interm : e)_tile × 2` | all | `var_typecast ? interm : e` | same | — |
| `c_12` writer_updated_m (`:286-295`) **conditional: `needs_mean_typecast`** | `d_tile × 2` | all | `d_data_format` | `d_tile` | — |
| `c_13` writer_updated_v (`:299-308`) **conditional: `needs_var_typecast`** | `e_tile × 2` | all | `e_data_format` | `e_tile` | — |
| `c_9` tmp1 (`:313-322`) | `interm_tile × 2` | all | `interm_data_format` | `interm_tile` | — |
| `c_10` tmp2 (`:324-333`) | `interm_tile × 2` | all | `interm_data_format` | `interm_tile` | — |
| `c_11` tmp3 (`:335-344`) | `interm_tile × 2` | all | `interm_data_format` | `interm_tile` | — |

`needs_mean_typecast = running_mean_has_value && stat_format_needs_typecast`;
`needs_var_typecast = running_var_has_value && stat_format_needs_typecast` (`:176-179`).
**They are independently keyed** — one may typecast while the other does not (only possible when
exactly one stat is absent, since the device-op guarantees at least one is present:
`running_statistics_device_operation.cpp:42-44`).

#### Semaphores

none.

#### Tensor accessors

5 sites, all 2-arg form, all `Buffer*`-delivered.

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `running_statistics_program_factory.cpp:351` + `:88` | `tensor_args.batch_mean` | reader slot 1 |
| `:364` + `:109` | `tensor_args.batch_var` | writer slot 0 |
| `:365` + `:112` | `tensor_return_value` (output) | writer slot 3 |
| `:366-367` + `:99-102` | `tensor_args.running_mean` *(optional; in-place read+write)* | writer slot 1 |
| `:368-369` + `:103-106` | `tensor_args.running_var` *(optional; in-place read+write)* | writer slot 2 |

Kernel-side: `reader_running_statistics.cpp:39`; `writer_running_statistics.cpp:52, 55, 58, 61`.
`running_mean` / `running_var` are read (`:87-92`, `:116-121`) **and written back to the same pages**
(`:103-108`, `:132-137`) through the same accessor — one `TensorParameter` each covers both directions.

#### Work split

Identical shape to BatchNorm: `split_work_to_cores(grid, num_output_tiles, row_major=true)`
(`:57`), result used only for per-core RTA values, all-zero RTA padding for cores outside both groups
(`:71-78`), kernels placed on all device cores.

#### Shared kernels

**none** — same verification as BatchNorm.

#### Runtime kernel-source selection

Single axis, same predicate (`:438-440`). SFPU source reads CTAs 0–18; non-SFPU reads 0–13.

---

### Flags

Things the inventory noticed; none is a stop signal.

1. **Dead trailing RTAs on all four dataflow kernels.** Each factory emits `cHt` and `cWt` in the last
   two RTA slots, and **no kernel reads them**:
   - BatchNorm reader: 11 emitted (`batch_norm_program_factory.cpp:87-99`), slots 0–8 read
     (`reader_batch_norm.cpp:15-23`) → slots **9, 10 dead**.
   - BatchNorm writer: 14 emitted (`:109-124`), slots 0–11 read (`writer_batch_norm.cpp:14-25`) →
     slots **12, 13 dead**.
   - RS reader: 11 emitted (`running_statistics_program_factory.cpp:85-97`), slots 0–8 read
     (`reader_running_statistics.cpp:16-24`) → slots **9, 10 dead**.
   - RS writer: 13 emitted (`:107-121`), slots 0–10 read (`writer_running_statistics.cpp:14-24`) →
     slots **11, 12 dead**.

   The port **drops them** (no named arg is declared). Zero functional change — the kernel never read
   the bytes. This is what makes the brief's named-RTA arities exact (reader 9 read → 8 named after the
   address slot goes; writer 12 read → 7 named after 5 address slots go).

2. **Dead trailing CTAs on the non-SFPU compute sources.** `batch_norm_program_factory.cpp:370-385`
   emits 15 CTAs but `batch_norm_kernel.cpp` reads only 0–10; RS emits 19 (`:416-435`) but
   `running_statistics_kernel.cpp` reads only 0–13. Harmless today (the unread values are consistent).
   The Metal 2.0 per-source `KernelSpec` shape drops them naturally — the drop is intentional, not lost
   plumbing.

3. **`packer_l1_acc` is resolved and dropped on the floor** (`batch_norm_program_factory.cpp:349`,
   `running_statistics_program_factory.cpp:391`; defaulted to `true` at `batch_norm_utils.cpp:28`).
   `ComputeConfigDescriptor` has no field for it and neither does `ComputeGen1Config`, so there is
   nothing to carry over and nothing to restore. Pre-existing gap — **the port must not "fix" it.**

4. **Single-entry CBs are double-buffered.** `eps` / `momentum` / `one` hold one tile for the kernel's
   whole lifetime but are allocated `num_entries = 2`. Reproduced verbatim (changing it is an L1
   footprint change — out of scope).

5. **Unreferenced kernel files:** none. All eight kernels are instantiated by their device-op's factory.

6. **Unity-build hazard — already neutralised by the repo's own idiom, no prefixes needed.**
   `TT_ENABLE_UNITY_BUILD(ttnn_op_normalization)`
   (`ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt:7`) merges both factory `.cpp`s into one TU,
   which would merge their anonymous namespaces. But `cmake/unity.cmake:9` sets
   `UNITY_BUILD_UNIQUE_ID "CMAKE_UNIQUE_NAMESPACE"`, and **both factory files already wrap their
   file-local code in `namespace { namespace CMAKE_UNIQUE_NAMESPACE { … } }`** — CMake defines that
   macro to a per-original-source unique identifier, so each file gets its own nested namespace. The
   port keeps that existing wrapper and declares its spec-name constants inside it, referenced from the
   factory body via `using namespace CMAKE_UNIQUE_NAMESPACE;`. See [Applied Patterns](#applied-patterns).

7. **No descriptor type outside the audit's scan.** The factories use only `KernelDescriptor`,
   `CBDescriptor`, `CBFormatDescriptor`, `ReaderConfigDescriptor`, `WriterConfigDescriptor`,
   `ComputeConfigDescriptor` — every one maps onto an audit Appendix A entry.

---

## TTNN ProgramFactory

*Filled in during the planning step. The concept itself was chosen in the audit; this section carries it forward.*

- **Concept (inherited from audit)**: **`ProgramSpecFactoryConcept`** — for **both** factories.
- **Custom `compute_program_hash`**: **none** — already the default reflection-based hash in both
  device-ops. Nothing to delete.
- **Implementation notes**:
  - The only device-op **header** change is each factory's method signature:
    `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` →
    `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`, plus swapping
    `#include <tt-metalium/program_descriptors.hpp>` for `#include "ttnn/metal_v2_artifacts.hpp"`.
    Nothing else in either device-op class is touched — in particular **not**
    `BatchNormOperation::operation_attributes_t::to_hash()`.
  - **No pybind change**: `batch_norm_nanobind.cpp` binds only the user-facing `ttnn::batch_norm`
    (`grep create_descriptor|nb::class_`: zero hits), so sanctioned exceptions 2 and 3 do not apply.
  - **Op-owned tensors: none** — neither factory allocates a device tensor beyond its io.
    `ProgramArtifacts::op_owned_tensors` stays defaulted.
  - **Tensor-arg matching: strict** (no relaxations). Confirmed independently: zero
    `ArgConfig::Runtime*` uses in the directory and no `TensorAccessor` third argument. All shape
    information travels as ordinary scalar RTAs (`HtWt`, `n_stride`, `c_stride`, `N`, `C`), so the
    accessor configuration does not vary with shape.
  - `MeshTensor` is extracted once at factory entry (`tensor.mesh_tensor()`) and used throughout;
    the file-local `extract_shape_dims` helper is retyped to `const MeshTensor&`.

---

## Planned Spec Shape

Default is 1:1 with legacy. Both units are single-work-unit, single-instance-per-kernel specs.

### Variant: `BatchNormOperation` — `ProgramSpec{.name = "batch_norm"}`

- **KernelSpecs — 3** (one per legacy `KernelDescriptor`; **no** work-split multiplicity):
  - `BN_READER` ← `reader_batch_norm.cpp`
  - `BN_WRITER` ← `writer_batch_norm.cpp`
  - `BN_COMPUTE` ← the runtime-selected compute source. **One** `KernelSpec`, whose `source`,
    `compile_time_args`, `defines` and `dfb_bindings` are built for whichever source is selected —
    exactly mirroring the legacy single `KernelDescriptor` with a runtime-chosen `kernel_source`.
- **DataflowBufferSpecs — 10** (9 unconditional + 1 conditional), one per legacy `CBDescriptor`.
  No aliased CBs (no legacy multi-element `format_descriptors`), so **no `alias_with`** anywhere.
  Names, and the legacy CB each carries:

  | `DFBSpecName` | legacy CB | `entry_size` | `num_entries` | `data_format_metadata` | conditional? |
  |---|---|---|---|---|---|
  | `input` | `c_0` | `a_tile` | 2 | `a_data_format` | — |
  | `batch_mean` | `c_1` | `b_tile` | 2 | `b_data_format` | — |
  | `out` | `c_2` | `(typecast ? interm : c)_tile` | 2 | `typecast ? interm : c` | — |
  | `batch_var` | `c_3` | `d_tile` | 2 | `d_data_format` | — |
  | `eps` | `c_4` | `interm_tile` | 2 | `interm_data_format` | — |
  | `weight` | `c_5` | `e_tile` | 2 | `e_data_format` | — |
  | `bias` | `c_6` | `f_tile` | 2 | `f_data_format` | — |
  | `den` | `c_7` | `interm_tile` | 2 | `interm_data_format` | — |
  | `temp_1` | `c_8` | `interm_tile` | 2 | `interm_data_format` | — |
  | `writer_out` | `c_9` | `c_tile` | 2 | `c_data_format` | **yes** — `needs_output_typecast` |

  `tile_format_metadata` is left `nullopt` on all ten (no legacy `tile` field was set).
- **SemaphoreSpecs — none** (legacy declares none).
- **TensorParameters — 6** (4 unconditional + 2 conditional), one per distinct originating tensor:
  `input`, `batch_mean`, `batch_var`, `output`, and `weight` / `bias` **only when present**.
- **WorkUnitSpecs — 1**: `{.name = "batch_norm", .kernels = {BN_READER, BN_WRITER, BN_COMPUTE},
  .target_nodes = all_device_cores}` — preserving the legacy all-device-cores placement verbatim.
- **Op-owned tensors — none.**

#### DFB endpoint census — re-derived from the kernel bodies

Producer/consumer read off the FIFO calls, not the kernel names. **The writer produces four of these**
(it reads batch_mean / batch_var / weight / bias from DRAM on the compute kernel's behalf).

| DFB | reader | writer | compute (both sources) | disposition |
|---|---|---|---|---|
| `input` | **P** (`reader_batch_norm.cpp:66,69`) | — | **C** (`batch_norm_kernel.cpp:73,90` as `dfb_other`) | 1P+1C |
| `eps` | **P** (`:46,54`) | — | **C** (`batch_norm_kernel.cpp:170,205`) | 1P+1C |
| `batch_mean` | — | **P** (`writer_batch_norm.cpp:85,93`) | **C** (`batch_norm_kernel.cpp:63,128` as `dfb_bcast`) | 1P+1C |
| `batch_var` | — | **P** (`:96,105`) | **C** (`batch_norm_kernel.cpp:47,60`) | 1P+1C |
| `weight` | — | **P** (`:108,116`, gated) | **C** (`batch_norm_kernel.cpp:66,131`, gated) | 1P+1C — **both bind in all configs** |
| `bias` | — | **P** (`:120,128`, gated) | **C** (`batch_norm_kernel.cpp:69,134`, gated) | 1P+1C — **same** |
| `den` | — | — | **P+C** (`batch_norm_kernel.cpp:46,61` / `:64,129`) | **self-loop** (1 toucher) |
| `temp_1` | — | — | **P+C** (as `dfb_affine_or_out` / `dfb_scaled_output` / `dfb_tmp_1`) | **self-loop** (1 toucher) |
| `out` | — | **C** (`writer_batch_norm.cpp:133,137` as `dfb_dst`) *only when `!needs_output_typecast`* | **P** always; **+C** when `needs_output_typecast` (`batch_norm_sfpu_kernel.cpp:164,183`) | **no typecast:** 1P+1C · **typecast:** **self-loop** |
| `writer_out` | — | **C** (`dfb_dst`) | **P** (`batch_norm_sfpu_kernel.cpp:166,184`) | 1P+1C *(only exists when typecast)* |

**Census agrees with the brief on all ten.** No DFB reaches ≥3 distinct touchers and none has two
kernels locked to the same FIFO role, so **`allow_instance_multi_binding` is not set anywhere**, and no
DFB is both self-looped and multi-bound. No dead CB (every DFB has ≥1 real or role-free toucher on each
endpoint).

Two points worth stating because they look like exceptions and are not:
- `weight` / `bias` are bound 1P+1C **even when their tensor is absent**. Both kernels reference the
  handles *outside* their `if constexpr` guards (`writer_batch_norm.cpp:49,64`;
  `batch_norm_sfpu_kernel.cpp:49-50`), so the binding is required as well as faithful — and legacy
  allocates the CBs unconditionally (`:260-279`), so unconditional DFBs reproduce the legacy L1
  footprint exactly.
- The `dfb_affine_or_out` / `dfb_scaled_output` selection inside the compute kernels is a **runtime**
  choice among already-bound DFBs (`batch_norm_sfpu_kernel.cpp:42-43`). It stays a `uint32_t` local; the
  `DFBAccessor → uint32_t` conversion makes the assignment legal. Not restructured.

### Variant: `RunningStatistics` — `ProgramSpec{.name = "running_statistics"}`

- **KernelSpecs — 3**: `RS_READER`, `RS_WRITER`, `RS_COMPUTE` (source runtime-selected as above).
- **DataflowBufferSpecs — 14** (12 unconditional + 2 conditional). No `alias_with`.

  | `DFBSpecName` | legacy CB | `entry_size` | `num_entries` | `data_format_metadata` | conditional? |
  |---|---|---|---|---|---|
  | `batch_mean` | `c_0` | `a_tile` | 2 | `a_data_format` | — |
  | `batch_var` | `c_1` | `b_tile` | 2 | `b_data_format` | — |
  | `output` | `c_2` | `c_tile` | 2 | `c_data_format` | — |
  | `old_running_mean` | `c_3` | `d_tile` | 2 | `d_data_format` | — |
  | `old_running_var` | `c_4` | `e_tile` | 2 | `e_data_format` | — |
  | `momentum` | `c_5` | `interm_tile` | 2 | `interm_data_format` | — |
  | `one` | `c_6` | `interm_tile` | 2 | `interm_data_format` | — |
  | `updated_mean` | `c_7` | `(mean_tc ? interm : d)_tile` | 2 | `mean_tc ? interm : d` | — |
  | `updated_var` | `c_8` | `(var_tc ? interm : e)_tile` | 2 | `var_tc ? interm : e` | — |
  | `tmp1` | `c_9` | `interm_tile` | 2 | `interm_data_format` | — |
  | `tmp2` | `c_10` | `interm_tile` | 2 | `interm_data_format` | — |
  | `tmp3` | `c_11` | `interm_tile` | 2 | `interm_data_format` | — |
  | `writer_updated_mean` | `c_12` | `d_tile` | 2 | `d_data_format` | **yes** — `needs_mean_typecast` |
  | `writer_updated_var` | `c_13` | `e_tile` | 2 | `e_data_format` | **yes** — `needs_var_typecast` |

- **SemaphoreSpecs — none.**
- **TensorParameters — 5** (3 unconditional + 2 conditional): `batch_mean`, `batch_var`, `output`,
  and `running_mean` / `running_var` **only when present**.
- **WorkUnitSpecs — 1**: `{"running_statistics", {RS_READER, RS_WRITER, RS_COMPUTE}, all_device_cores}`.
- **Op-owned tensors — none.**

#### DFB endpoint census — re-derived

| DFB | reader | writer | compute | disposition |
|---|---|---|---|---|
| `batch_mean` | **P** (`reader_running_statistics.cpp:73,76`) | — | **C** (`running_statistics_kernel.cpp:50,75`) | 1P+1C |
| `momentum` | **P** (`:59,67`) | — | **C** (`running_statistics_kernel.cpp:41,80`) | 1P+1C |
| `one` | **P** (`:56` → `fill_cb_with_value`, which does `reserve_back`/`push_back`) | — | **C** (`running_statistics_kernel.cpp:40,79`) | 1P+1C |
| `batch_var` | — | **P** (`writer_running_statistics.cpp:79,82`) | **C** (`running_statistics_kernel.cpp:51,76`) | 1P+1C |
| `output` | — | **C** (`:144,148` as `dfb_dst`) | **P** (`running_statistics_sfpu_kernel.cpp:99,295`) | 1P+1C |
| `old_running_mean` | — | **P** (`:86,99`, gated) | **C** (`running_statistics_sfpu_kernel.cpp:142,160`, gated) | 1P+1C — **both bind in all configs** |
| `old_running_var` | — | **P** (`:115,128`, gated) | **C** (`:240,257`, gated) | 1P+1C — **same** |
| `tmp1` | — | — | **P+C** (`:103,119` / `:141,161`) | **self-loop** |
| `tmp2` | — | — | **P+C** (`:122,138` / `:164,197`) | **self-loop** |
| `tmp3` | — | — | **P+C** (`:143,158` / `:165,196`) | **self-loop** |
| `updated_mean` | — | **C** (`dfb_new_mean`, `:102,110`) *only when `!needs_mean_typecast`* | **P** (`:166,187`); **+C** when `needs_mean_typecast` (via `maybe_typecast_stat`, `:20,39`) | **no mean typecast:** 1P+1C · **mean typecast:** **self-loop** |
| `updated_var` | — | **C** (`dfb_new_var`, `:131,139`) *only when `!needs_var_typecast`* | **P** (`:263,280`); **+C** when `needs_var_typecast` | same, keyed on `needs_var_typecast` |
| `writer_updated_mean` | — | **C** (`dfb_new_mean`) | **P** (`maybe_typecast_stat` `dst_obj`, `:22,40`) | 1P+1C *(only when mean typecast)* |
| `writer_updated_var` | — | **C** (`dfb_new_var`) | **P** | 1P+1C *(only when var typecast)* |

**Census agrees with the brief on all fourteen.** No multi-binding flag, no dead CB, no stacking.

One case that deserves the explicit note because it *looks* like a dead CB and is not: when
`running_mean` is absent, `updated_mean` (`c_7`) is bound compute-PRODUCER + writer-CONSUMER but
**neither kernel performs a FIFO op on it** — both bodies are inside a skipped
`if constexpr (old_running_mean_has_value)`. That is two role-free touchers → an ordinary 1P+1C
assignment (the labels are cosmetic, exactly as for BatchNorm `weight` / `bias`). Legacy allocates the
CB regardless (`:263-271`), so the L1 footprint is unchanged.

---

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** Both factories call `split_work_to_cores` but use the
result **only** to compute per-core RTA values: the two core groups receive *identical* CTAs and a
single `KernelDescriptor` per kernel over `all_device_cores`
(`batch_norm_program_factory.cpp:64-133`, `running_statistics_program_factory.cpp:63-130`).

So there is nothing to preserve and **one `WorkUnitSpec` per factory** over all device cores. This is
the *inverse* of the [demoting-per-group-CTA anti-pattern](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta):
no per-group dimension exists, so inventing a two-group split (or narrowing the work unit to the
working cores) would be a *behavior change* — kernel placement — not a faithful port.

Consequence the construction step must honour: **`SetProgramRunArgs` requires every named RTA on every
node the kernel runs on**, and the kernels run on every device core. The legacy all-zero padding for
cores outside both work groups is therefore preserved verbatim, as named-RTA values of `0`
(`batch_norm_program_factory.cpp:72-79`, `running_statistics_program_factory.cpp:71-78`). The kernels
already handle it — `batch_norm_kernel.cpp:145-147` and `batch_norm_sfpu_kernel.cpp:205-207` return
early on `num_tiles == 0`.

---

## Dropped Plumbing

Every legacy CTA/RTA that does **not** survive the port. The enumeration is the gate against
builder-pattern carry-over: anything not listed here would be translated by reflex.

### Buffer-address RTAs → `TensorBinding` (11 slots)

All eleven arrive in the `Buffer*` form (the framework's pointer-patching interim mechanism), not
`->address()`. The typed binding supersedes it and the RTA slot disappears from each kernel.

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `batch_norm_program_factory.cpp:90` — reader RTA slot 1 | `input_tensor.buffer()` | `TensorParameter BN_INPUT` + `TensorBinding` on `BN_READER` (`tensor::input`) |
| `:111` — writer RTA slot 0 | `batch_mean_tensor.buffer()` | `BN_BATCH_MEAN` + binding on `BN_WRITER` (`tensor::batch_mean`) |
| `:112` — writer RTA slot 1 | `batch_var_tensor.buffer()` | `BN_BATCH_VAR` (`tensor::batch_var`) |
| `:101-104` — writer RTA slot 2 | `weight_tensor->buffer()`, else literal `0u` | `BN_WEIGHT` (`tensor::weight`) — **conditional** binding |
| `:105-108` — writer RTA slot 3 | `bias_tensor->buffer()`, else literal `0u` | `BN_BIAS` (`tensor::bias`) — **conditional** binding |
| `:115` — writer RTA slot 4 | `c.buffer()` | `BN_OUTPUT` (`tensor::output`) |
| `running_statistics_program_factory.cpp:88` — reader RTA slot 1 | `batch_mean_tensor.buffer()` | `RS_BATCH_MEAN` (`tensor::batch_mean`) |
| `:109` — writer RTA slot 0 | `batch_var_tensor.buffer()` | `RS_BATCH_VAR` (`tensor::batch_var`) |
| `:99-102` — writer RTA slot 1 | `running_mean_tensor->buffer()`, else `0u` | `RS_RUNNING_MEAN` (`tensor::running_mean`) — **conditional** |
| `:103-106` — writer RTA slot 2 | `running_var_tensor->buffer()`, else `0u` | `RS_RUNNING_VAR` (`tensor::running_var`) — **conditional** |
| `:112` — writer RTA slot 3 | `c.buffer()` | `RS_OUTPUT` (`tensor::output`) |

All eleven are **Case 1** (every address feeds a `TensorAccessor`): no raw base-pointer arithmetic
anywhere, so the `get_bank_base_address` bridge is **not** used and the compute-kernel Case-2 block
cannot arise (the compute kernels construct no `TensorAccessor` at all).

### `TensorAccessorArgs` plumbing → the binding mechanism (11 host sites, 11 kernel chains)

| legacy host site | legacy kernel-side chain | Metal 2.0 replacement |
|---|---|---|
| `batch_norm_program_factory.cpp:307` | `reader_batch_norm.cpp:27` `TensorAccessorArgs<2>()` | `TensorAccessor(tensor::input)` |
| `:319-324` (5 calls) | `writer_batch_norm.cpp:37-41` — `TensorAccessorArgs<7>()` then 4× `next_compile_time_args_offset()` | `TensorAccessor(tensor::batch_mean \| output \| batch_var \| weight \| bias)` |
| `running_statistics_program_factory.cpp:351` | `reader_running_statistics.cpp:29` `TensorAccessorArgs<3>()` | `TensorAccessor(tensor::batch_mean)` |
| `:364-369` (4 calls) | `writer_running_statistics.cpp:36-39` — `TensorAccessorArgs<8>()` then 3× chained | `TensorAccessor(tensor::batch_var \| output \| running_mean \| running_var)` |

**Page-size 3rd-argument CTAs/RTAs: none** — every accessor is the 2-arg form, so no third-argument
value is emitted and there is nothing to drop here. (Audit gate: GREEN — N/A.)

### Magic CB indices in CTAs → `DFBBinding` (24 CTA slots)

| legacy location | legacy CTA slots | Metal 2.0 replacement |
|---|---|---|
| `batch_norm_program_factory.cpp:303-306` (reader) | `[0]` `input_tensor_cb`, `[1]` `eps_cb` | `DFBBinding{input, "src", PRODUCER}`, `DFBBinding{eps, "eps", PRODUCER}` |
| `:313-317` (writer) | `[2]` `batch_mean_cb`, `[3]` `writer_output_cb`, `[4]` `batch_var_cb`, `[5]` `weight_cb`, `[6]` `bias_cb` | `DFBBinding`s on `BN_WRITER`: `batch_mean`/`batch_var`/`weight`/`bias` as PRODUCER, and `out` **or** `writer_out` as CONSUMER |
| `:373-382` (compute) | `[2]`–`[11]` — `input_cb`, `batch_mean_cb`, `output_cb`, `batch_var_cb`, `eps_cb`, `den_cb`, `weight_cb`, `temp_1_cb`, `bias_cb`, `writer_output_cb` | `DFBBinding`s on `BN_COMPUTE` per the census table above |
| `running_statistics_program_factory.cpp:347-349` (reader) | `[0]` `batch_mean_cb`, `[1]` `momentum_cb`, `[2]` `one_cb` | three PRODUCER `DFBBinding`s |
| `:357-362` (writer) | `[2]`–`[7]` — `batch_var_cb`, `output_cb`, `old_running_mean_cb`, `old_running_var_cb`, `writer_updated_m_cb`, `writer_updated_v_cb` | `DFBBinding`s on `RS_WRITER` per the census table |
| `:419-432` (compute) | `[2]`–`[15]` — the twelve unconditional CBs plus `writer_updated_m_cb`, `writer_updated_v_cb` | `DFBBinding`s on `RS_COMPUTE` per the census table |

Kernel side, these become `dfb::<name>` handles; the `constexpr auto dfb_id_* = get_compile_time_arg_val(N)`
lines go away and their **role comments are relocated** to the `DataflowBuffer` construction (whitelist rule 8).

### CTA gates promoted to `compiler_options.defines`

These CTAs exist today *only* to drive an `if constexpr` that guards a reference to a conditionally-bound
resource. They must move to the preprocessor (an `if constexpr` still name-looks-up the discarded branch).

| legacy location | legacy CTA | Metal 2.0 replacement | emitted on |
|---|---|---|---|
| `batch_norm_program_factory.cpp:311` | writer CTA `[0]` `weight_has_value` | `#define WEIGHT_HAS_VALUE` — gates `TensorAccessor(tensor::weight)` construction (`writer_batch_norm.cpp:64-65`) **and its uses** (`:107-117`) | `BN_WRITER` only |
| `:312` | writer CTA `[1]` `bias_has_value` | `#define BIAS_HAS_VALUE` — gates `writer_batch_norm.cpp:68-69` and `:119-129` | `BN_WRITER` only |
| `:383` | compute CTA `[12]` `needs_output_typecast` | `#define NEEDS_OUTPUT_TYPECAST` — gates the `dfb_output_final` handle alias **and** the `needs_output_typecast` template argument | `BN_COMPUTE` only (see [Deferred / Flagged](#deferred--flagged) item 1 for why the writer needs no gate) |
| `running_statistics_program_factory.cpp:355` | writer CTA `[0]` `running_mean_has_value` | `#define OLD_RUNNING_MEAN_HAS_VALUE` — gates `writer_running_statistics.cpp:57-58` and `:84-111` | `RS_WRITER` only |
| `:356` | writer CTA `[1]` `running_var_has_value` | `#define OLD_RUNNING_VAR_HAS_VALUE` — gates `:60-61` and `:113-140` | `RS_WRITER` only |
| `:433` | compute CTA `[16]` `stat_format_needs_typecast` | `#define NEEDS_MEAN_TYPECAST` / `#define NEEDS_VAR_TYPECAST` — the pair is computed **host-side** (`running_mean_has_value && stat_format_needs_typecast`, etc.) rather than re-derived in the kernel from two CTAs | `RS_COMPUTE` only (see [Deferred / Flagged](#deferred--flagged) item 1) |

The compute kernels keep `weight_has_value` / `bias_has_value` (BatchNorm) and
`old_running_mean_has_value` / `old_running_var_has_value` (RS) as ordinary **named CTAs**: they never
reference a `tensor::` token, and the DFBs they gate are bound *unconditionally*, so their guards need
no preprocessor promotion. (In the BatchNorm compute kernels these are *runtime* `if`s anyway —
`batch_norm_sfpu_kernel.cpp:82,117,140` — which is precisely why weight/bias must be bound
unconditionally there. Left alone.)

### Dead RTA slots → dropped (8 slots)

BatchNorm reader `[9],[10]`; BatchNorm writer `[12],[13]`; RS reader `[9],[10]`; RS writer `[11],[12]` —
the `cHt` / `cWt` values no kernel reads. See [Flags](#flags) item 1. No named arg is declared.

### Dead compute CTAs on the non-SFPU sources → dropped per-source

BatchNorm compute `[11]`–`[14]` and RS compute `[14]`–`[18]` are emitted today but unread by
`batch_norm_kernel.cpp` / `running_statistics_kernel.cpp`. The per-source `KernelSpec` omits them.
See [Flags](#flags) item 2.

### Positional CTAs → named CTAs

Every surviving scalar CTA gets a name. Complete post-port lists:

| KernelSpec | named `compile_time_args` |
|---|---|
| `BN_READER` | `fill_eps_fp32` (from `any_float32`, `:308`) |
| `BN_WRITER` | `batch_stat_is_fp32` (`:325`), `param_is_fp32` (`:328`) |
| `BN_COMPUTE` (non-SFPU) | `weight_has_value`, `bias_has_value` |
| `BN_COMPUTE` (SFPU) | `weight_has_value`, `bias_has_value`, `tc_in_fmt` (`:384`), `tc_out_fmt` (`:385`) |
| `RS_READER` | `fill_momentum_fp32` (from `any_float32`, `:352`) |
| `RS_WRITER` | `old_stat_is_fp32` (`:370`) |
| `RS_COMPUTE` (non-SFPU) | `old_running_mean_has_value`, `old_running_var_has_value` |
| `RS_COMPUTE` (SFPU) | + `tc_in_fmt` (`:434`), `tc_out_fmt` (`:435`) |

### Named RTA schemas (all named — no varargs anywhere)

Every runtime arg is read exactly once as a distinct field at a literal index — no counted loops, no
`arg_index++` runs, no data-selected indices. So **every** RTA is a named arg and
`advanced_options.num_runtime_varargs` stays 0.

| KernelSpec | `runtime_arg_names` | count |
|---|---|---|
| `BN_READER` | `eps`, `start_tile_id`, `num_tiles`, `HtWt`, `n_stride`, `c_stride`, `N`, `C` | 8 |
| `BN_WRITER` | `start_tile_id`, `num_tiles`, `HtWt`, `n_stride`, `c_stride`, `N`, `C` | 7 |
| `BN_COMPUTE` | `num_tiles`, `tile_freq`, `tile_start` | 3 |
| `RS_READER` | `momentum`, `start_tile_id`, `num_tiles`, `HtWt`, `n_stride`, `c_stride`, `N`, `C` | 8 |
| `RS_WRITER` | `start_tile_id`, `num_tiles`, `HtWt`, `n_stride`, `c_stride`, `N`, `C` | 7 |
| `RS_COMPUTE` | `num_tiles` | 1 |

`RS_COMPUTE` reads only RTA slot 0 today (`running_statistics_kernel.cpp:12`,
`running_statistics_sfpu_kernel.cpp:45`) although the factory emits three (`:126`); slots 1 and 2
(`freq`, `counter`) are dead and are dropped with the other dead RTAs.

**No `common_runtime_arg_names` anywhere** — legacy declares no CRTAs. Several RTAs *do* hold the same
value on every working node (`HtWt`, `n_stride`, `c_stride`, `N`, `C`) and would be more efficient as
CRTAs, but converting them changes dispatch semantics: **not port work**, noted for a later pass.

### Semaphore-ID RTAs

**none** — the op declares no semaphores at all.

---

## Hardware configuration and compiler options

The two silent-regression surfaces. Both factories are identical in shape here.

- **Compute config is Style A** — both resolve a TTNN `DeviceComputeKernelConfig` through
  `batch_norm::utils::resolve_compute_kernel_config` (`batch_norm_utils.cpp:14-38`) and destructure it
  with `get_compute_kernel_config_args` (`batch_norm_program_factory.cpp:349`,
  `running_statistics_program_factory.cpp:391`). Port translation:
  `ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config)`,
  which carries `math_fidelity → fpu_math_fidelity`, `math_approx_mode → sfpu_precision_mode`,
  `fp32_dest_acc_en → enable_32_bit_dest`, and `dst_full_sync_en → double_buffer_dest` (**inverted**).
  The op's defaults are non-standard — `default_fp32_acc = true` and, on Wormhole,
  `default_fp32_acc_math_fidelity = HiFi3` (hardware bug #38306) — so `enable_32_bit_dest = true` is the
  **common** path and `unpack_modes` is load-bearing on the default configuration.
- **`bfp_pack_precision_mode`**: legacy sets no `bfp8_pack_precise`, so it stays at its default
  (`Precision::Approximate`) — no action.
- **`unpack_modes` — a faithful re-key of the legacy `vector<UnpackToDestMode>`**, keyed by
  `DFBSpecName`, value `UnpackMode::UnpackToDest`, emitted **only under `fp32_dest_acc_en`** exactly as
  legacy (`batch_norm_program_factory.cpp:352-368`, `running_statistics_program_factory.cpp:394-411`).
  Legacy `Default` → `UnpackToSrc`, expressed by **omitting** the entry.
  - **BatchNorm:** `input`, `batch_mean`, `batch_var`, `eps`, `den`, `weight`, `temp_1`, `bias` —
    **plus `out`** when `needs_output_typecast`. `writer_out` gets **no** entry.
  - **RS:** `batch_mean`, `batch_var`, `output`, `old_running_mean`, `old_running_var`, `updated_mean`,
    `updated_var`, `momentum`, `one`, `tmp1`, `tmp2`, `tmp3`. `writer_updated_mean` /
    `writer_updated_var` get **no** entry.

  Both sets are legal and complete as-is; verified against the validator
  (`tt_metal/impl/metal2_host_api/program_spec.cpp:990-1073`) rather than assumed:
  - Every entry sits under `enable_32_bit_dest == true`, which the validator accepts unconditionally
    (`:1010-1012`). So the "≤16-bit format + `UnpackToDest` is rejected on Gen1" rule **cannot** fire,
    including on the `fp32_dest_acc_en && !any_float32` path where `den`/`temp_1`/`eps` are `Float16_b`.
  - The newly-required-explicit-entry rule (consumed **Float32** DFB under `enable_32_bit_dest`,
    `:1043-1073`) is already satisfied: the legacy list covers every DFB the compute kernel *consumes*
    in every configuration. `writer_out` / `writer_updated_*` are **producer-only** for compute, and
    RS `output` is producer-only too — the validator explicitly tolerates a producer-side entry as
    inert (`:1005-1007`), so keeping RS `output` in the list is faithful, not a mistake.
  - The entries are emitted on the **same condition as their DFB's binding**, so no entry can name a
    DFB the kernel does not bind (`:993-998`).

  **The risk here is a copy-paste between the two factories, or a dropped key — both silent.** Each set
  is written out longhand at its own factory and diffed against the legacy list line by line.
- **DM configs are plain defaults** — `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` on all
  four dataflow kernels, no custom `(processor, noc, noc_mode)` triple and no `DM_DYNAMIC_NOC`. Port to
  `ttnn::create_reader_datamovement_config(device->arch())` /
  `ttnn::create_writer_datamovement_config(device->arch())`. Role name matches resolved values here, so
  reader→reader and writer→writer.
- **`opt_level`** — `grep -n opt_level` over the directory: **zero hits**. Resolved legacy levels are
  `O2` for the four DM kernels (Metal 2.0's `CompilerOptions` default — **no action**) and **`O3` for
  the compute kernels** (the `ComputeConfigDescriptor` default, which Metal 2.0 does *not* share).
  So **both** compute `KernelSpec`s state
  `.compiler_options = {.defines = …, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3}` explicitly.
- **Gen2 out of scope** — only the Gen1 alternative is built; no `if (arch == QUASAR)` branch is added.
  The two TTNN helpers already supply the Gen2 branch for the default DM cases.

---

## Applied Patterns

- **[Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)**
  — compute bound both PRODUCER and CONSUMER (one shared `accessor_name`) on: BatchNorm `den`, `temp_1`,
  and `out` *in the typecast configuration*; RS `tmp1`, `tmp2`, `tmp3`, and `updated_mean` /
  `updated_var` *in their respective typecast configurations*. Each is a genuine one-toucher.
- **[Two-toucher DFB → assign 1P+1C](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)**
  — the endpoint-assignment procedure was run per `(DFB, config)` for all 24 DFBs. Notably the
  *role-free* two-toucher cases (BatchNorm `weight`/`bias` with the tensor absent; RS
  `old_running_*`/`updated_*` with the stat absent) resolve to a cosmetic 1P+1C, **not** the
  multi-binding flag.
- **[Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)**
  — three conditional DFBs (`writer_out`, `writer_updated_mean`, `writer_updated_var`) and four
  conditional tensors (`weight`, `bias`, `running_mean`, `running_var`), each host-bound conditionally
  with a matching `compiler_options.defines` entry and an `#ifdef`-gated kernel-side alias / accessor
  construction. The four tensor cases are the **mandatory**-gate flavour (nothing to bind when absent).
- **[Same-FIFO aliasing, path-dependent variant](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)**
  — the writer-facing handles resolve to *different* DFBs per compile-time path. **One** `#ifdef`-gated
  `constexpr` handle alias each; **not** a second `DFBBinding` and **not** `alias_with`:

  ```cpp
  #ifdef NEEDS_OUTPUT_TYPECAST
  constexpr auto dfb_output_final = dfb::writer_out;
  #else
  constexpr auto dfb_output_final = dfb::out;
  #endif
  ```

  Needed on the **compute** sources only — the writers get there without a gate; see
  [Deferred / Flagged](#deferred--flagged) item 1.
- **[Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)**
  — `dfb::one` into `fill_cb_with_value(uint32_t cb_id, …)` (`reader_running_statistics.cpp:56`), and
  `dfb::name` into the `dest_format_helpers.hpp` helpers (`pack_tile_with_dt`,
  `copy_tile_to_dst_init_short_with_dt`, `copy_tile_init_with_dt`) and the LLKs (`add_tiles`,
  `pack_tile`, `binary_op_init_common`, `pack_reconfig_data_format`, …). No `.id` extraction, no
  temporary `DataflowBuffer` wrappers.
- **[Unity-build hygiene for anonymous-namespace symbols](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)**
  — the concern applies (`ttnn_op_normalization` is a unity-build target, and both factories declare
  same-named constants), but the catalog's **prefixing** remedy is *not* what this port uses: the repo
  already solves it structurally via `UNITY_BUILD_UNIQUE_ID "CMAKE_UNIQUE_NAMESPACE"`
  (`cmake/unity.cmake:9`) and both files already have the
  `namespace { namespace CMAKE_UNIQUE_NAMESPACE { … } }` wrapper. The port declares its constants
  inside that existing wrapper and pulls them in with `using namespace CMAKE_UNIQUE_NAMESPACE;`, so
  `READER` / `INPUT_DFB` / … can be spelled identically in both files with no collision and no prefix
  noise. See [Flags](#flags) item 6.
- **Multi-variant factories: NOT applied.** Each device-op has a single factory and a single spec shape;
  the runtime compute-source selection is a `KernelSpec::source` choice, not a per-variant spec branch.

---

## Deferred / Flagged

1. **Why only the *compute* sources need the `#ifdef` alias — the writers are path-dependent too, but
   the binding model absorbs it.** The writers read the same host-selected value through their own CTAs
   (`writer_batch_norm.cpp:33`, CTA `[3]` `dfb_id_dst` = `writer_output_cb`;
   `writer_running_statistics.cpp:34-35`, CTAs `[6]`/`[7]` = `writer_updated_m_cb`/`_v_cb`), so on first
   read they look like three more alias sites. They are not, and the distinction is the *accessor name*:

   - Each writer references its path-dependent buffer through exactly **one** handle. So one
     `DFBBinding` with a **fixed `accessor_name`** and a `dfb_spec_name` chosen host-side
     (`needs_output_typecast ? WRITER_OUT_DFB : OUT_DFB`, accessor `"dst"`) emits the same `dfb::dst`
     token on both paths. The kernel writes `DataflowBuffer dfb_dst(dfb::dst);` unconditionally — no
     define, no `#ifdef`, no second binding. This is the direct Metal 2.0 analogue of the legacy single
     CTA slot carrying whichever CB index.
   - The **compute** kernels cannot do that, because they *already* bind `out` (resp. `updated_mean` /
     `updated_var`) under its own accessor name. Adding a second name for the same DFB on the same
     kernel is rejected by the validator, so on the non-typecast path there is no way to route
     `dfb_output_final` through a second binding — hence the `#ifdef`-gated handle alias the brief
     prescribes, which is a *kernel-side* alias of an already-bound token.

   Net: the brief's table is correct as written. Recorded here (and in the report under Successes)
   because the near-miss is easy to make in the other direction — emitting the define on the writers
   too and adding a redundant `#ifdef` there.

2. **New finding — `RS_COMPUTE` has two dead RTA slots, which the brief's arity table does not flag.**
   The brief lists "RS ... compute 1", which is correct for the *named* count, but the factory emits
   three RTAs per core (`running_statistics_program_factory.cpp:126`) and both RS compute sources read
   only slot 0. The `freq` / `counter` values are dead on this factory (they are live on BatchNorm's
   compute, which reads all three). Dropped with the other dead RTAs; no functional change.

3. **No structural issue the audit missed.** Nothing in either factory requires a legacy workaround, no
   feature gate fired late, and no construct falls outside the kernel-side whitelist. In particular:
   no `GlobalCircularBuffer`, no `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`, no cursor
   surgery (`evil_set_*` not needed — no `LocalCBInterface` field writes exist), no host-computed
   base-pointer offset, no Case 2 binding, and no out-of-op kernel edit required.

4. **Out of scope, routed to the report** (not changed here): `packer_l1_acc` dropped on the floor;
   `b_num_tiles_per_cb` redundant alias; single-entry CBs allocated with `num_entries = 2`; the
   node-invariant RTAs that would be more efficient as CRTAs; the `to_hash` inertness fragility.

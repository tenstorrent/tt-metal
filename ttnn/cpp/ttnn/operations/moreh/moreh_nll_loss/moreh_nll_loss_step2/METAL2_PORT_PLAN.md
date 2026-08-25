# Port Plan — `moreh_nll_loss_step2`

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2`, ported from
`ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Inputs consumed: `METAL2_PORT_BRIEF.md` (actionable), `METAL2_PREPORT_AUDIT.md` (detail), and the
sibling `moreh_nll_loss_step1` port on branch `anasuya/metal2_port_moreh_nll_loss` (read for spec
*shape* only — its `c_7` disposition is deliberately **not** reused; see
[Deferred / Flagged](#deferred--flagged)).

## Legacy Inventory

### Legacy factory shape

- Concept: **`ProgramDescriptorFactoryConcept`** — `Factory::create_descriptor` returns
  `tt::tt_metal::ProgramDescriptor` (`device/moreh_nll_loss_step2_device_operation.hpp:35`; body at
  `device/moreh_nll_loss_step2_program_factory.cpp:701-732`).
- Variants: **single** — `using program_factory_t = std::variant<Factory>`
  (`..._device_operation.hpp:41`). The factory methods live **inside** a `program_factory_t` variant, so
  this is *not* the direct-descriptor shape and `ttnn_factory.md` exception 3 does **not** apply.
- **Three internal rank paths, not three factories.** `create_descriptor` is a thin dispatcher over
  three file-local builders, each producing its own `ProgramDescriptor`:

  | Path | Builder | Reader | Writer | Selected when |
  |---|---|---|---|---|
  | 2d | `moreh_nll_loss_step2_impl_2d` (`:45`) | `reader_..._2d.cpp` | `writer_..._2d.cpp` | `rank == 2` |
  | 3d | `moreh_nll_loss_step2_impl_3d` (`:258`) | `reader_..._3d.cpp` | `writer_..._3d.cpp` | `rank == 3` |
  | 4d | `moreh_nll_loss_step2_impl_4d` (`:471`) | `reader_..._4d.cpp` | `writer_..._4d.cpp` | `rank >= 4` |

- Custom `compute_program_hash`: **none** — default reflection-based hash. No backdoor
  `attribute_values` / `to_hash` either. Nothing to leave alone, nothing to touch.
- No `override_runtime_arguments`; no `get_dynamic_runtime_args`; no pybound `create_descriptor`
  (`moreh_nll_loss_nanobind.cpp` binds only the user-facing `ttnn::moreh_nll_loss`).

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN
factory analysis section. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### The config space

**Twelve configs, all reachable:** 3 rank paths × `weight` present/absent × `divisor` present/absent.
`weight` and `divisor` are `std::optional<Tensor>`; `divisor` is supplied only for `reduction == MEAN`
(`moreh_nll_loss.cpp:44-53`), absent for `SUM` and `NONE`. The inventory below is organised per rank
path, with per-config qualifiers on the rows that vary.

### Kernels

`opt_level` is **unset on every `KernelDescriptor`** in all three paths (`grep -n opt_level` over the
factory returns nothing), so the resolved levels below are the API defaults: `O2` for the
reader/writer descriptors, **`O3`** for the `ComputeConfigDescriptor` ones.

#### Variant: 2d

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs (per core) | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `kernels/reader_moreh_nll_loss_step2_2d.cpp` | `all_cores` | 4 × `TensorAccessorArgs` blocks — input, target, weight-or-`nullptr`, divisor-or-`nullptr` (`:107-110`) | none | 10: `input_buf`, `target_buf`, `weight_buf`, `divisor_buf`, `ignore_index`, `units_per_core`, `tile_offset`, `origin_N`, `origin_C`, `input.element_size()` (`:210-223`) | none | `WEIGHT`?, `DIVISOR`?, `FP32_DEST_ACC_EN`? (`:119-131`) | O2 (unset) | `ReaderConfigDescriptor{}` |
| writer | `kernels/writer_moreh_nll_loss_step2_2d.cpp` | `all_cores` | 1 × `TensorAccessorArgs` — output (`:113`) | none | 4: `output_buf`, `units_per_core`, `tile_offset`, `origin_N` (`:225-232`) | none | none (`writer_defines` is built but never populated) | O2 (unset) | `WriterConfigDescriptor{}` |
| compute_1 | `kernels/moreh_nll_loss_step2_kernel.cpp` | `core_group_1` | `{units_per_core_group_1}` (`:163`) | none | 1: `{units_per_core}` — **dead** (`:235-243`) | none | `WEIGHT`?, `DIVISOR`?, `FP32_DEST_ACC_EN`? | **O3** (unset) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, math_approx_mode}` (`:165-171`) |
| compute_2 | same source | `core_group_2` (only if non-empty) | `{units_per_core_group_2}` (`:179`) | none | same, **dead** | none | same | **O3** (unset) | same (`:181-187`) |

#### Variant: 3d

Identical shape; the deltas are:

| unique_id | source | RTAs (per core) |
|---|---|---|
| reader | `kernels/reader_moreh_nll_loss_step2_3d.cpp` | 11: `…`, `origin_N`, `origin_C`, `origin_W`, `input.element_size()` (`:421-435`) |
| writer | `kernels/writer_moreh_nll_loss_step2_3d.cpp` | 5: `output_buf`, `units_per_core`, `tile_offset`, `origin_W`, `output.element_size()` (`:437-445`) |
| compute_1 / compute_2 | same compute source | `{units_per_core}` — **dead** (`:448-456`) |

CTAs, defines, configs and `opt_level` as in 2d (`:317-324`, `:330-342`, `:376-398`).

#### Variant: 4d

| unique_id | source | RTAs (per core) |
|---|---|---|
| reader | `kernels/reader_moreh_nll_loss_step2_4d.cpp` | 13: `input_buf`, `target_buf`, `weight_buf`, `divisor_buf`, `ignore_index`, `units_per_core`, `tile_offset`, `origin_N`, `origin_C`, `Wt`, `num_inner_tile`, `weight_num_tile`, `input.element_size()` (`:651-667`) |
| writer | `kernels/writer_moreh_nll_loss_step2_4d.cpp` | 3: `output_buf`, `units_per_core`, `tile_offset` (`:669-675`) |
| compute_1 / compute_2 | same compute source | `{units_per_core}` — **dead** (`:678-686`) |

CTAs, defines, configs and `opt_level` as in 2d (`:547-554`, `:560-572`, `:606-628`).

**Compute-kernel multiplicity is a work split, not a variant.** The one compute source is instantiated
**twice per path** — `compute_desc_1` over `core_group_1`, `compute_desc_2` over `core_group_2` — with
different per-group CTA values. See [Preserved Multiplicity](#preserved-multiplicity).

### CBs

Every CB is created through the file-local `push_cb` helper (`:22-41`), which **returns early on
`num_tiles == 0`** — that early return is what makes the optional CBs conditional. Each descriptor is
single-format: `total_size = num_tiles * tile_size(data_format)`, `page_size = tile_size(data_format)`,
`core_ranges = all_cores`. **No `format_descriptors[i].tile` is ever set**, so
`tile_format_metadata` stays `nullopt` on every DFB. No `CBDescriptor::buffer`, no
`set_globally_allocated_address`, no `global_circular_buffer` anywhere — no borrowed-memory CBs, no
GlobalCircularBuffer, no aliased CBs (every `format_descriptors` list has exactly one element).

`data_format` = `datatype_to_dataformat_converter(input.dtype())`, which `validate_inputs` pins to
`BFLOAT16` → `Float16_b` (`..._device_operation.cpp:23`).
`fp32_acc_format` = `fp32_dest_acc_en ? Float32 : data_format`.

| index | role | num_tiles (2d / 3d) | num_tiles (4d) | data_format | core_ranges | 2d/3d site | 4d site |
|---|---|---|---|---|---|---|---|
| `c_0` | input | 1 | 1 | `data_format` | `all_cores` | `:85`, `:296` | `:521` |
| `c_1` | target | 1 | 1 | `Int32` | `all_cores` | `:86`, `:297` | `:522` |
| `c_2` | weight | `weight ? 1 : 0` | `weight ? weight_num_tile : 0` | `data_format` | `all_cores` | `:87`, `:298` | `:523-528` |
| `c_3` | divisor | `divisor ? 1 : 0` | `divisor ? 1 : 0` | `data_format` | `all_cores` | `:88`, `:299` | `:529` |
| `c_7` | weight scratch | `weight ? 1 : 0` — **dead** | `weight ? 1 : 0` | `data_format` | `all_cores` | `:102`, `:313` | `:543` |
| `c_16` | output | 1 | 1 | `data_format` | `all_cores` | `:96`, `:307` | `:537` |
| `c_24` | tmp_weight | 1 (unconditional) | 1 (unconditional) | `fp32_acc_format` | `all_cores` | `:89-90`, `:300-301` | `:530-531` |
| `c_25` | tmp_input | 1 | 1 | `fp32_acc_format` | `all_cores` | `:91-92`, `:302-303` | `:532-533` |
| `c_26` | tmp1 | 1 | 1 | `fp32_acc_format` | `all_cores` | `:93`, `:304` | `:534` |
| `c_27` | divisor_recip | 1 | 1 | `fp32_acc_format` | `all_cores` | `:94`, `:305` | `:535` |
| `c_28` | tmp3 | 1 | 1 | `fp32_acc_format` | `all_cores` | `:95`, `:306` | `:536` |

`c_27`'s factory comment reads `// tmp2` but the compute kernel names it `cb_divisor_recip` and stores
`1/divisor` (`compute:23-24`). The kernel's name is the accurate one; the DFB is named for the kernel's
use. (Recorded as an anomaly by the audit; the port does not "fix" the stale comment, it just does not
propagate it into the new name.)

### Semaphores

**None** — zero `SemaphoreDescriptor`s, zero `semaphore` references anywhere in the op.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) | kernel construction |
|---|---|---|---|
| `:107` / `:318` / `:548` | `input` | reader **0** (`:213`, `:424`, `:654`) | `TensorAccessor(input_args, input_addr)` — 2d `:47`, 3d `:50`, 4d `:51` |
| `:108` / `:319` / `:549` | `target` | reader **1** (`:214`, `:425`, `:655`) | `TensorAccessor(target_args, target_addr)` — 2d `:48`, 3d `:51`, 4d `:52` |
| `:109` / `:320` / `:550` | `weight` (optional) | reader **2** (`:215`, `:426`, `:656`) | `TensorAccessor(weight_args, weight_addr)` — 2d `:49`, 3d `:52`, 4d `:53`; built **outside** the `WEIGHT` guard |
| `:110` / `:321` / `:551` | `divisor` (optional) | reader **3** (`:216`, `:427`, `:657`) | `TensorAccessor(divisor_args, divisor_addr)` — 2d `:54`, 3d `:57`, 4d `:58`; built **inside** the `DIVISOR` guard |
| `:113` / `:324` / `:554` | `output` | writer **0** (`:228`, `:440`, `:672`) | `TensorAccessor(output_args, output_addr)` — 2d `:19`, 3d `:24`, 4d `:21` |

All five arrive as `Buffer*` objects through
`emplace_runtime_args(core, initializer_list<variant<uint32_t, Buffer*>>)` — never `->address()`. The
framework auto-registers each as a `BufferBinding` and re-patches it on cache hits, so the op is
already correct on cache hits today; the typed `TensorBinding` supersedes that interim mechanism. All
five are **Case 1** (accessed only through the `TensorAccessor`), so no `get_bank_base_address` bridge
is needed anywhere — and, critically, the **compute kernel constructs no `TensorAccessor` and reads no
runtime args at all**, so there is no Case-2-in-a-compute-kernel blocker.

The absent-optional plumbing is three coordinated pieces, and the middle one is load-bearing:
`nullptr` `Buffer*` → framework emits `0u` with no binding; `TensorAccessorArgs(nullptr).append_to(...)`
still appends a **placeholder args block**, which is what keeps the *following* accessors'
`next_compile_time_args_offset()` chain aligned (2d `:42-45`); and the `WEIGHT` / `DIVISOR` defines.
`weight` and `divisor` sit at positions 3 and 4 of a four-accessor chain, so a dropped placeholder
would shift every downstream offset. **The whole chain disappears in the port** — see
[Dropped Plumbing](#dropped-plumbing).

### Work split

All three paths: `split_work_to_cores(grid, units_to_divide)` with
`grid = device->compute_with_storage_grid_size()`, yielding
`(num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2)`.
`all_cores == core_group_1 ∪ core_group_2`, and the two groups are **disjoint**.

| path | `units_to_divide` | site |
|---|---|---|
| 2d | `N / TILE_HEIGHT` (padded `input_shape[0]`) | `:60`, `:72-73` |
| 3d | `origin_N * div_up(origin_W, FACE_WIDTH)` | `:281`, `:283-284` |
| 4d | `target.physical_volume() / H / W * Ht * Wt` | `:504`, `:506-507` |

`core_group_2` may be empty; `has_core_group_2` gates the second compute descriptor (`:174`, `:385`,
`:615`).

### Shared kernels

**None.** The op owns all seven kernel `.cpp` files, and `grep -rl <filename>
ttnn/cpp/ttnn/operations/` finds no consumer outside this op's own factory — nothing *borrowed*,
nothing *lent*, and the three rank paths bind **disjoint** reader/writer sources so there is no
*intra-op* sharing either. The one source the paths **do** share, `moreh_nll_loss_step2_kernel.cpp`,
is shared *within the single factory being ported*, which converts atomically with it — not a
shared-kernel Caution case. No `_metal2` fork to reuse, none to create, no sunset list.

The two out-of-directory dependencies are **headers, not borrowed kernel files**:
`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (all three readers, 3d/4d writers) and
`ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` (compute). Both donors already take
`DataflowBuffer` **by value** at every symbol this op consumes (`read_tile`, `read_value`, `read_line`,
`copy_tile_init_with_dt`, `pack_tile_with_dt`, `mul_tiles_init_with_dt`), so the kernels keep passing
their named DFB locals and **no donor-side change and no fork is needed**.

### Flags

- **No unreferenced kernel files** — all seven are bound by the factory.
- **No descriptor type outside the audit's scan.** The factory uses only `KernelDescriptor`,
  `CBDescriptor` / `CBFormatDescriptor`, `ReaderConfigDescriptor`, `WriterConfigDescriptor`,
  `ComputeConfigDescriptor`, and `ProgramDescriptor`.
- **Dead code the port removes** (each zero-functional-change, each listed in
  [Dropped Plumbing](#dropped-plumbing)): the dead `c_7` allocations in 2d/3d; four dead CB
  declarations; nine dead `get_dataformat` locals; seven dead RTAs (four from the brief, **three newly
  found** — see [Deferred / Flagged](#deferred--flagged)).
- **Kernels are already part-modernized.** All seven are on Device 2.0 (`DataflowBuffer`, `Noc`,
  `CoreLocalMem`, `TensorAccessor`) with the current `api/dataflow/dataflow_buffer.h` include, not the
  stale `api/dataflow/circular_buffer.h`. This is a binding-layer change, not an idiom rewrite.
- **No varargs anywhere**, on either the RTA or the CTA side. Every reader/writer arg is a distinct
  field read once from a fixed `i++` run (a sequential counter over a *fixed* set is legacy positional
  plumbing, not a collection), and the only CTA read is the compute kernel's single
  `get_compile_time_arg_val(0)`. Everything becomes a named arg.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: **`ProgramSpecFactoryConcept`** — the plain one. The ported-from
  factory has no `override_runtime_arguments`, so the framework refreshes tensor bindings on a cache
  hit and the factory writes exactly one method.
- **Custom `compute_program_hash`**: **none** — default reflection-based hash. Nothing to preserve.
- **Implementation notes**:
  - `Factory::create_descriptor` → `Factory::create_program_artifacts` **inside the existing
    `Factory` struct**. The struct already sits in `program_factory_t`, so no nested-struct
    introduction is needed and `ttnn_factory.md` exception 3 does not apply.
  - The header swaps `#include <tt-metalium/program_descriptors.hpp>` for
    `#include "ttnn/metal_v2_artifacts.hpp"`. That is the *only* device-op-class edit: it is forced by
    the return-type change, not a scope excursion.
  - **No pybind deletion** — nothing pybinds `create_descriptor`, so exception 1 does not fire and the
    port removes no user-visible Python surface.
  - The three rank paths stay three file-local builders, each now returning
    `ttnn::device_operation::ProgramArtifacts` — the
    [multi-variant factory pattern](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)
    applied to a rank dispatch. `create_program_artifacts` stays the thin dispatcher it is today.
  - The three builders live in one translation unit and share DFB / kernel / tensor name constants, so
    the constants are declared **once** at file scope in the anonymous namespace rather than per
    builder — which also sidesteps the
    [unity-build hygiene](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
    duplicate-symbol trap.

## Planned Spec Shape

The three rank paths produce **structurally identical** specs; they differ only in kernel *sources*,
the `c_2` tile count, the RTA name sets, and whether `WEIGHT_SCRATCH` exists. One shape, stated once,
with the per-path deltas called out.

- **KernelSpecs** (3 or 4): `READER`, `WRITER`, `COMPUTE_G1`, and `COMPUTE_G2` **only when
  `core_group_2` is non-empty** — one per legacy `KernelDescriptor`, preserving the compute
  multiplicity.
- **DataflowBufferSpecs** (8–11), one per legacy `CBDescriptor` that survives:
  - **Unconditional (8)**: `INPUT` (`c_0`), `TARGET` (`c_1`), `OUTPUT` (`c_16`), `TMP_WEIGHT` (`c_24`),
    `TMP_INPUT` (`c_25`), `TMP1` (`c_26`), `DIVISOR_RECIP` (`c_27`), `TMP3` (`c_28`).
  - **Conditional (2)**: `WEIGHT` (`c_2`) when `weight.has_value()`; `DIVISOR` (`c_3`) when
    `divisor.has_value()`.
  - **Conditional, 4d only (1)**: `WEIGHT_SCRATCH` (`c_7`) when `weight.has_value()` — **not built at
    all in the 2d and 3d paths**, where the legacy allocation is dead.
  - No `alias_with`, no `borrowed_from`, no `allow_instance_multi_binding`, `tile_format_metadata`
    `nullopt` throughout (the legacy `.tile` field is never set).
- **SemaphoreSpecs**: **none** — no legacy `SemaphoreDescriptor`.
- **TensorParameters** (3–5): `TENSOR_INPUT`, `TENSOR_TARGET`, `TENSOR_OUTPUT` always;
  `TENSOR_WEIGHT` when `weight.has_value()`; `TENSOR_DIVISOR` when `divisor.has_value()`. One per
  distinct originating tensor — the readers' four plus the writers' one.
- **WorkUnitSpecs** (1 or 2):
  - `WU_G1` = `{READER, WRITER, COMPUTE_G1}` over `core_group_1`
  - `WU_G2` = `{READER, WRITER, COMPUTE_G2}` over `core_group_2` — only when non-empty

  `READER` / `WRITER` therefore have an effective node set of `core_group_1 ∪ core_group_2` =
  `all_cores`, matching their legacy `core_ranges`, while each compute spec stays on its own group.
- **Op-owned tensors**: **none** — the legacy `ProgramDescriptor` carries no `buffers` vector and the
  factory allocates no device tensors of its own. `ProgramArtifacts::op_owned_tensors` is left
  defaulted.

### DFB endpoint assignment (re-derived from the kernel-touch census, not transcribed)

Census run per CB **per node**. On any node the population is 1 reader + 1 writer + **1** compute
instance (the two compute specs cover disjoint groups). Verdicts below agree with the brief's §7 on
every CB; the one place I want the reasoning on the record is why three *bindings* on `TMP_INPUT`,
`OUTPUT`, `TMP_WEIGHT` and `DIVISOR` are still an ordinary 1:1 — see the note under the table.

| DFB | Config | Touchers per node | Assignment |
|---|---|---|---|
| `INPUT` | all 12 | 1 — reader only (`read_value` reserves/pushes; body waits / peeks / pops) | **self-loop** on `READER` (PRODUCER + CONSUMER, one accessor name) |
| `TARGET` | all 12 | 1 — reader only (`read_tile` reserves/pushes; body waits / peeks / pops) | **self-loop** on `READER` |
| `WEIGHT` | weight present | 1 — reader only | **self-loop** on `READER` |
| `WEIGHT_SCRATCH` | weight present, **4d only** | 1, role-free — reader only, entirely sync-free (`read_line` uses it as an `async_read` destination + `get_write_ptr`; no FIFO ops) | **self-loop** on `READER` (labels cosmetic on Gen1) |
| `DIVISOR` | divisor present | 2 — reader FIFO-produces (`read_tile`), compute FIFO-consumes | **1P+1C**: `READER` PRODUCER, both compute specs CONSUMER |
| `TMP_INPUT` | all 12 | 2 — reader FIFO-produces, compute FIFO-consumes | **1P+1C**: `READER` PRODUCER, both compute specs CONSUMER |
| `TMP_WEIGHT` | weight present | 2 — reader FIFO-produces, compute FIFO-consumes | **1P+1C**: `READER` PRODUCER, both compute specs CONSUMER |
| `TMP_WEIGHT` | weight **absent** | 1, role-free — reader does not bind it at all; compute constructs its DFB (`compute:18`) and names it in `compute_kernel_hw_startup` (`compute:34`), both unconditional | **self-loop** on each compute spec |
| `OUTPUT` | all 12 | 2 — compute FIFO-produces, writer FIFO-consumes | **1P+1C**: both compute specs PRODUCER, `WRITER` CONSUMER |
| `TMP1` | all 12 | 1 — compute only, both roles under `WEIGHT` or `DIVISOR`; role-free (constructed at `compute:22`, no FIFO op) in the 3 neither-present configs | **self-loop** on each compute spec |
| `DIVISOR_RECIP` | all 12 | 1 — compute only, both roles under `DIVISOR`; role-free (`compute:24`) otherwise | **self-loop** on each compute spec |
| `TMP3` | all 12 | 1 — compute only, both roles under `WEIGHT && DIVISOR`; role-free (`compute:26`) otherwise | **self-loop** on each compute spec |

**No DFB in this op takes `allow_instance_multi_binding`.** Max per-node census is 2, and every such
pair is one locked producer plus one locked consumer. The three multi-toucher faces were each checked
and none applies: no hidden co-filler (the op has **no semaphores at all**, so the coordinating
mechanism cannot exist); no CB read from two co-resident kernels; and the two compute descriptors are
one source over **disjoint** node sets — the *disjoint-node* work split, explicitly **not** the
same-grid two-toucher shape.

> **Three `KernelSpec`s bind `TMP_INPUT` / `OUTPUT` / `TMP_WEIGHT` / `DIVISOR`, and that is legal.**
> Counting *bindings* gives three (reader-or-writer + both compute specs) and would bait the
> multi-binding flag. The invariant is **per node**: `dataflow_buffer_spec.hpp` states that multiple
> `KernelSpec`s may share one endpoint role provided they have non-overlapping node coverage, the same
> kernel kind, and identical binding-site parameters. `COMPUTE_G1` and `COMPUTE_G2` satisfy all three
> (disjoint groups, both compute, both default STRIDED / `num_threads = 1`), so each node still sees
> exactly one producer and one consumer. The framework validates the non-overlap.

> **Why the role-free self-loops are not dead CBs.** `TMP_WEIGHT` (no-weight), `TMP1`,
> `DIVISOR_RECIP` and `TMP3` have configs with *no FIFO op at all* — but the compute kernel constructs
> a `DataflowBuffer` for each **unconditionally** (`compute:18`, `:22`, `:24`, `:26`), and
> `TMP_WEIGHT` is additionally named in `compute_kernel_hw_startup` (`compute:34`). Each therefore
> needs its `dfb::` token to resolve in **every** config or the kernel does not compile, so the spec
> is built unconditionally and self-looped where no producer/consumer pair exists. Whether
> `compute_kernel_hw_startup(dfb::tmp_weight, …)` is itself an *endpoint binding* is
> [audit Question 1](METAL2_PREPORT_AUDIT.md#questions-for-the-user), open with the framework team and
> explicitly not a blocker: it changes only the *label* (1 role-free toucher → self-loop vs. 0
> touchers → conditional DFB), never the instruction that the spec must exist in all 12 configs. The
> plan takes the side that is correct under both readings.

## Preserved Multiplicity

```
Legacy KernelDescriptors [compute_desc_1, compute_desc_2] of source
  device/kernels/moreh_nll_loss_step2_kernel.cpp
  → KernelSpecs [COMPUTE_G1, COMPUTE_G2] of the same source, differing ONLY in the
    per-group CTA `per_core_tile_cnt` (units_per_core_group_1 / units_per_core_group_2)
  → in WorkUnitSpecs [WU_G1 over core_group_1, WU_G2 over core_group_2]  (disjoint)
  → sharing DFBs, endpoint role each KernelSpec binds:
       TMP_INPUT      CONSUMER (both)          OUTPUT     PRODUCER (both)
       TMP_WEIGHT     CONSUMER (both) when weight present;
                      PRODUCER + CONSUMER (both, self-loop) when absent
       DIVISOR        CONSUMER (both) when divisor present
       TMP1, DIVISOR_RECIP, TMP3   PRODUCER + CONSUMER (both, self-loop)
```

`COMPUTE_G2` and `WU_G2` are built **only when `core_group_2` is non-empty**, mirroring the legacy
`has_core_group_2` gate.

**This is the op the [demoting-per-group-CTA anti-pattern](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
was written about, and it baits the trap unusually well.** The factory *already* populates a per-core
compute RTA carrying exactly `units_per_core` (`:235-243`, `:448-456`, `:678-686`) — and the compute
kernel never reads it. Collapsing to one `KernelSpec` fed by that RTA would look like a simplification
and would cost compile-time loop unrolling on `per_core_tile_cnt` (`compute:11`, loop bound at
`compute:54`) — a measurable kernel-perf regression the port is not entitled to make. The plan keeps
two `KernelSpec`s and **deletes** the dead RTA rather than adopting it.

## Dropped Plumbing

### Buffer-address RTAs → `TensorBinding`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 0 — `:213`, `:424`, `:654` | `input_buf` (`Buffer*`) | `TensorParameter TENSOR_INPUT` + `TensorBinding` on `READER` (`tensor::input`) |
| reader RTA slot 1 — `:214`, `:425`, `:655` | `target_buf` | `TENSOR_TARGET` + `TensorBinding` (`tensor::target`) |
| reader RTA slot 2 — `:215`, `:426`, `:656` | `weight_buf` (or `nullptr`) | `TENSOR_WEIGHT` + **conditional** `TensorBinding` (`tensor::weight`) |
| reader RTA slot 3 — `:216`, `:427`, `:657` | `divisor_buf` (or `nullptr`) | `TENSOR_DIVISOR` + **conditional** `TensorBinding` (`tensor::divisor`) |
| writer RTA slot 0 — `:228`, `:440`, `:672` | `output_buf` | `TENSOR_OUTPUT` + `TensorBinding` (`tensor::output`) |

All five are **Case 1** — no `get_bank_base_address` bridge anywhere.

### `TensorAccessorArgs` plumbing → the binding mechanism

| legacy site | legacy form | Metal 2.0 replacement |
|---|---|---|
| `:107-110`, `:318-321`, `:548-551` | four `TensorAccessorArgs(...).append_to(reader_compile_time_args)` calls, including the two `nullptr` placeholder blocks | gone — `TensorBinding` carries the layout metadata |
| `:113`, `:324`, `:554` | `TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args)` | gone |
| readers 2d `:42-45`, 3d `:43-46`, 4d `:46-49` | `TensorAccessorArgs<0>()` + three chained `next_compile_time_args_offset()` blocks | gone — `TensorAccessor(tensor::name)` |
| writers 2d `:17`, 3d `:22`, 4d `:19` | `constexpr auto output_args = TensorAccessorArgs<0>();` | gone |

The whole placeholder / offset-chain apparatus disappears together, which is why the load-bearing
`nullptr` placeholder needs no replacement: there are no positional CTA offsets left to keep aligned.

**TensorAccessor 3rd arg**: none — all 15 accessor constructions in the op are the 2-arg form. Nothing
to drop, no `dynamic_tensor_shape` to set.
**TensorParameter relaxation**: `none` — strict `TensorSpec` matching throughout.

### Magic CB indices → `DFBBinding`

Every `constexpr uint32_t cb_* = tt::CBIndex::c_N;` in every kernel becomes a `dfb::<name>` handle.
No CB index was ever carried by a CTA or an RTA in this op (the readers' and writers' CTA lists are
`TensorAccessorArgs` blocks and nothing else), so there is **no dead CTA slot to remove** alongside —
the indices were hardcoded kernel-side.

| kernel | legacy constants | Metal 2.0 |
|---|---|---|
| reader 2d `:25-33` / 3d `:26-34` / 4d `:27-37` | `cb_input`, `cb_target`, `cb_weight`, `cb_divisor`, `cb_tmp_weight`, `cb_tmp_input`, `cb_output` (+ 4d `cb_weight_scratch`) | `dfb::input`, `dfb::target`, `dfb::weight`, `dfb::divisor`, `dfb::tmp_weight`, `dfb::tmp_input` (+ `dfb::weight_scratch`); `cb_output` **deleted** — see below |
| writer 2d `:15` / 3d `:20` / 4d `:17` | `cb_output` | `dfb::output` |
| compute `:13-29` | `cb_weight`, `cb_divisor`, `cb_tmp_weight`, `cb_tmp_input`, `cb_tmp1`, `cb_divisor_recip`, `cb_tmp3`, `cb_output` | `dfb::divisor`, `dfb::tmp_weight`, `dfb::tmp_input`, `dfb::tmp1`, `dfb::divisor_recip`, `dfb::tmp3`, `dfb::output`; `cb_weight` **deleted** — see below |

### Positional CTAs → named CTAs

| legacy location | legacy form | Metal 2.0 |
|---|---|---|
| `:163`, `:179`, `:374`, `:390`, `:604`, `:620` | `compile_time_args = {units_per_core_group_N}`; kernel reads `get_compile_time_arg_val(0)` (`compute:11`) | `compile_time_args = {{"per_core_tile_cnt", units_per_core_group_N}}`; kernel reads `get_arg(args::per_core_tile_cnt)` |

Reader and writer CTA lists held **only** `TensorAccessorArgs` blocks, so after those drop the readers
and writers have **no CTAs at all**.

### Positional RTAs → named RTAs

Names are taken straight off the kernels' own declarations.

| kernel | named RTAs after the port |
|---|---|
| reader 2d | `ignore_index`, `num_tiles_per_core`, `start_id`, `N`, `C` |
| reader 3d | `ignore_index`, `num_tiles_per_core`, `start_id`, `C`, `W`, `element_size` |
| reader 4d | `ignore_index`, `num_tiles_per_core`, `start_id`, `C`, `num_inner_tile`, `weight_num_tile` |
| writer 2d | `num_tiles_per_core`, `start_id` |
| writer 3d | `num_tiles_per_core`, `start_id`, `W`, `element_size` |
| writer 4d | `num_tiles_per_core`, `start_id` |
| compute | **none** — the kernel reads no runtime args |

### Dead plumbing deleted (zero-functional-change)

| what | legacy location | why it goes |
|---|---|---|
| `c_7` allocation, 2d path | `:102` | **Zero endpoints in every 2d config** — the 2d reader never names `c_7` (constants at `:25-33`) and reads weight via `read_value`, which takes no scratch. A bindingless DFB is rejected by the validator, and a dead CB has no behavior, so the drop changes L1 footprint and nothing else. |
| `c_7` allocation, 3d path | `:313` | Same — 3d reader constants at `:26-34`, `read_value` again. `impl_4d`'s `:543` allocation is **kept**: the 4d reader genuinely uses it (`:32`, `:73-74`). |
| dead `cb_output` declaration | readers 2d `:33`, 3d `:34`, 4d `:37` | Never used. Converting it mechanically would add a **reader binding** on `OUTPUT`, taking the per-node census 2 → 3 and baiting a spurious multi-binding flag. |
| dead `cb_weight` declaration | compute `:13` | Never used. Converting it would add a **compute binding** on `WEIGHT`, turning a clean self-loop into a spurious 1P+1C. |
| nine dead `get_dataformat` locals | readers 2d `:36`/`:38`/`:40`, 3d `:37`/`:39`/`:41`, 4d `:40`/`:42`/`:44` | `input_data_format`, `weight_data_format`, `divisor_data_format` — all computed, none used. Deleted rather than converted to DFB getters (whitelist rule 7 would apply if any were live; all are `const`, so they would take the member-getter form). `divisor_data_format` additionally reads `c_3` **unconditionally**, so converting it would be a third no-divisor compile failure. |
| dead per-core compute RTA vector | `:235-243`, `:448-456`, `:678-686` | The compute kernel reads **no** runtime args (zero `get_arg_val`). Every dispatch wrote per-core args nothing read. Deleted, **not** adopted — adopting it is the demoting-per-group-CTA anti-pattern. |
| dead writer RTA `origin_N` | `:231` (2d) | The 2d writer reads only indices 0-2 (`:11-13`). |
| dead reader RTA `element_size` | `:222` (2d), `:666` (4d) | Read into a local and never used (2d `:23`, 4d `:25`). The **3d** reader's `element_size` **is** used (`:89`) and is kept. |
| dead reader RTA `N` | `:431` (3d), `:661` (4d) | **New finding, not in the brief.** Read at 3d `:21` / 4d `:20`, never used. The 2d reader's `N` **is** used (`:73`) and is kept. |
| dead reader RTA `Wt` | `:663` (4d) | **New finding, not in the brief.** Read at 4d `:22`, never used. |

## Applied Patterns

- **[Multi-variant factories](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)**
  — rank (2d / 3d / 4d) selection inside `create_program_artifacts`, dispatching to three file-local
  builders. Each builds its own `ProgramSpec` + `ProgramRunArgs`; name constants are shared at file
  scope.
- **[Self-loop DFB binding](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)**
  / **[Sync-free and single-ended CBs → self-loop DFB](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)**
  — `INPUT`, `TARGET`, `WEIGHT`, `WEIGHT_SCRATCH` on `READER`; `TMP1`, `DIVISOR_RECIP`, `TMP3` on each
  compute spec; `TMP_WEIGHT` on each compute spec in the no-weight configs. All one-toucher
  resolutions; `WEIGHT_SCRATCH` is the pure sync-free case (no FIFO ops at all).
- **[Conditional / optional DFB bindings](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)**
  — `WEIGHT` and `DIVISOR` DFBs, `TENSOR_WEIGHT` and `TENSOR_DIVISOR` tensor parameters, and
  `WEIGHT_SCRATCH` (4d), each bound only on its own condition. The `WEIGHT` / `DIVISOR` defines the
  legacy factory already emits are exactly the matching preprocessor gates, so **no CTA gate needs
  promoting to a define** — the kernels are already `#ifdef`-structured. Two kernel-side declarations
  sit on the wrong side of an existing guard and move inside it (compute's `c_3` DFB; the readers'
  `weight` accessor); see [Deferred / Flagged](#deferred--flagged).
- **[Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)**
  — the compute kernel's `compute_kernel_hw_startup`, `copy_tile`, `mul_tiles`,
  `mul_tiles_bcast_scalar`, `mul_bcast_scalar_init` and `reconfig_data_format` all take `uint32_t`
  CB ids and have **no** `DataflowBuffer` overload anywhere in `tt_metal/hw/inc/api/compute/` — the
  compute LLK surface is index-based by design. `dfb::name` is passed directly and the implicit
  conversion fires; no `.id` / `.get_id()` extraction, no temporary wrapper. The donors' `*_with_dt`
  helpers already take `DataflowBuffer`, so they keep taking the named local.
- **[Unity-build hygiene for anonymous-namespace symbols](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)**
  — name constants and the `make_dfb` / `bind_self_loop` helpers are declared once in this file's
  anonymous namespace, not duplicated per rank builder.
- **Explicitly NOT applied**: `allow_instance_multi_binding` (no DFB needs it — max per-node census 2);
  `alias_with` (no legacy aliased CBs); `borrowed_from` (no borrowed-memory CBs); varargs (every
  argument is a nameable distinct field); `CustomProgramSpecFactoryConcept` (no
  `override_runtime_arguments` to translate).

### Hardware configuration and compiler options

- **DM kernels.** Both resolved triples are the API defaults — `ReaderConfigDescriptor{}` → reader
  default (`RISCV_1` / `NOC_0` / `DM_DEDICATED_NOC`), `WriterConfigDescriptor{}` → writer default
  (`RISCV_0` / `NOC_1` / `DM_DEDICATED_NOC`). Ported with the arch-agnostic TTNN helpers
  `ttnn::create_reader_datamovement_config(device->arch())` and
  `ttnn::create_writer_datamovement_config(device->arch())`. No custom `noc_mode`, so no paired
  per-node setting to carry on both kernels.
- **Compute kernels — Style A.** The op resolves a TTNN `ComputeKernelConfig` via
  `get_compute_kernel_config_args(device->arch(), compute_kernel_config)` (`:75`, `:286`, `:509`), so
  the port translates the resolved config with
  `ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config)`.
  **Checked for a dropped field**, per the recipe: the factory resolves five values and sets
  **all four** that have a Metal 2.0 counterpart onto its `ComputeConfigDescriptor` —
  `math_fidelity` → `fpu_math_fidelity`, `math_approx_mode` → `sfpu_precision_mode`,
  `fp32_dest_acc_en` → `enable_32_bit_dest`, `dst_full_sync_en` → `double_buffer_dest` (inverted).
  `packer_l1_acc` is destructured and never used (`:75`), and has no Metal 2.0 counterpart, so it
  needs no action. **Nothing is dropped**, so the helper's output needs no hand-patched field.
  `bfp_pack_precision_mode` is left at its default — the legacy `ComputeConfigDescriptor` never sets
  `bfp8_pack_precise`, and the defaults coincide.
- **`unpack_modes` — the one field that must be set by hand.** Legacy passes
  `vector<UnpackToDestMode>(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default)`, i.e. `Default` for every
  CB, which maps to `UnpackMode::UnpackToSrc` and would normally be expressed by *omitting* every
  entry. But Metal 2.0's validator **requires an explicit entry** for every `Float32` DFB a compute
  kernel *consumes* when `enable_32_bit_dest = true`. When `fp32_dest_acc_en` is set, the five
  intermediate DFBs are `Float32` (`fp32_acc_format`) and the compute kernel consumes all five, so
  under that flag the port adds five explicit `UnpackMode::UnpackToSrc` entries — value **derived from
  the legacy vector** (`Default` → `UnpackToSrc`), not guessed:
  `TMP_WEIGHT`, `TMP_INPUT`, `TMP1`, `DIVISOR_RECIP`, `TMP3`.
  All five are bound in **every** config, so the entries need no per-config gating and the validator's
  "entry names a DFB the kernel doesn't bind" rule cannot fire. `DIVISOR` (`data_format`, pinned to
  `Float16_b` by `validate_inputs`) and `TARGET` (`Int32`, deliberately deferred — issue #49936) get no
  entry. When `fp32_dest_acc_en` is false nothing is `Float32` and the table stays empty, exactly as
  legacy behaved.
- **`opt_level`.** `grep -n opt_level` over the legacy factory returns **nothing**, so no kernel set an
  explicit level. DM kernels need nothing (legacy `O2` = Metal 2.0 `O2`). The compute kernels **do**:
  legacy `ComputeConfigDescriptor` resolves to **`O3`** while `KernelSpec::compiler_options` defaults
  to `O2`, so each compute `KernelSpec` sets `.opt_level = KernelBuildOptLevel::O3` explicitly.
  Per `KernelSpec`, not per role — that is **two** specs per rank path, six in total, and both are
  built through the same shared lambda so the level cannot be set on one and missed on the other.

## Deferred / Flagged

New findings from the inventory and planning steps:

1. **Three dead reader RTAs the audit and brief both missed** — the 3d reader's `N` (`:431`, read at
   `reader_..._3d.cpp:21`), the 4d reader's `N` (`:661`, read at `reader_..._4d.cpp:20`), and the 4d
   reader's `Wt` (`:663`, read at `reader_..._4d.cpp:22`). Each is read into a local and never
   referenced; confirmed by word-boundary grep over each reader (one occurrence each — the declaration
   itself). The brief listed **four** dead RTAs and the invoker's decision was to drop those rather
   than name them. These three are the identical class — provably unread, zero behavior — so the port
   applies the same decision and drops them too, bringing the total to **seven**. Flagged here and in
   the port report as an extension of the invoker's decision beyond its literal four, so a reviewer
   can see it was deliberate and can reverse it cheaply (naming them instead is a one-line change per
   arg).
2. **Two kernel-side declarations sit outside the guard their uses sit inside**, and both become hard
   compile failures in the port rather than the latent no-ops they are today. Both are decided the
   same way — move the declaration inside the existing guard:
   - **compute `:15`** — `DataflowBuffer dfb_divisor_obj(cb_divisor);` outside `#if defined(DIVISOR)`
     while every use is inside (`:37`, `:40`, `:41`, `:46`). `c_3` is not allocated when divisor is
     absent, so `dfb::divisor` will not exist and the six no-divisor configs would not compile. The
     three readers already do this correctly (2d `:56`, 3d `:59`, 4d `:60`); only the compute kernel is
     inconsistent, and only it is changed.
   - **readers 2d `:49`, 3d `:52`, 4d `:53`** — `TensorAccessor(weight_args, weight_addr)` outside
     `#if defined(WEIGHT)` while every use is inside. `tensor::weight` will not exist in the six
     no-weight configs. Note this one is **forced, not chosen**: there is no weight tensor to bind when
     absent, so an always-bind alternative does not exist.
3. **No structural issue the audit missed.** No GlobalCircularBuffer, no `get_cb_tiles_acked_ptr` /
   `get_cb_tiles_received_ptr`, no `AddrSelector` / `CircularBufferView` wrapper, no cursor surgery
   (`evil_set_*` / raw `LocalCBInterface` field writes), no host-computed `base + offset` folded into
   an address arg, no `ArgConfig::Runtime*` in any kernel, no descriptor type outside the audit's
   Appendix A scope. Nothing here trips a stop signal.
4. **Not reusing `step1`'s `c_7` answer** — deliberately, per the brief's warning. `step1` has the
   same defect shape (a scratch CB allocated on `weight_has_value` alone while only one reader variant
   uses it) but there both variants live behind one host flag inside **one** program, so its `c_7` had
   to become a *conditional* DFB. Here the rank paths are three separate builders producing three
   separate specs, so in the 2d and 3d builders the allocation is dead **unconditionally** → a straight
   drop with no conditional to write, and in the 4d builder it is a live conditional self-loop. Same
   defect, three different dispositions.
5. **Pre-port baseline recorded** before any edit, with the Metal 2.0 legality checks forced on and
   proven live: `53 passed, 44 skipped, 106 deselected` over the invoker-confirmed test set
   (`tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_nll_loss.py` and
   `test_moreh_nll_loss_unreduced.py`, `-k "not backward"`). The 44 skips are the `bfloat8_b`
   parametrisations, which the tests skip themselves. This is the no-regression bar.

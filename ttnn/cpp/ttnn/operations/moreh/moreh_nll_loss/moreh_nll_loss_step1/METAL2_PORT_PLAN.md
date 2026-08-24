# Port Plan — `moreh_nll_loss_step1`

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1`, ported from the
`ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

All `file:line` references below are pre-port (base `9d839c902f5`). Paths are relative to the op
directory unless stated otherwise; `..._program_factory.cpp` is
`device/moreh_nll_loss_step1_program_factory.cpp`.

**The three configs** (from the audit; several rows below are config-scoped):

| Config | Meaning |
|---|---|
| **A** | small algorithm, no weight (`use_large_algorithm == false`, `weight_has_value == false`) |
| **B** | small algorithm, with weight |
| **C** | large algorithm, with weight |

`use_large_algorithm` implies `weight_has_value`, so *large without weight* is unreachable.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `Factory::create_descriptor` returns
  `tt::tt_metal::ProgramDescriptor` (`device/moreh_nll_loss_step1_device_operation.hpp:34`; body at
  `..._program_factory.cpp:17-226`).
- Factory methods live in a **nested `Factory` struct** with
  `using program_factory_t = std::variant<Factory>` (`..._device_operation.hpp:40`). This is *not* the
  direct-descriptor shape, so [`ttnn_factory.md` exception 3] does not apply — the port is a method
  swap inside the existing struct.
- Variants: single.
- Custom `compute_program_hash`: **none** — default reflection-based hash. No backdoor
  `attribute_values` / `to_hash` anywhere under `moreh_nll_loss/`.
- `override_runtime_arguments`: **absent** → base concept, nothing to translate.

### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_nll_loss_step1.cpp` (small) **or** `..._large.cpp` (large) — runtime source selection at `:158-162` | `all_cores` | `{weight_has_value}` then `TensorAccessorArgs(target)` block then `TensorAccessorArgs(weight-or-nullptr)` block (`:141-143`) | none | 9 per core (`:203-215`): `target_buf`, `weight_buf`, `ignore_index`, `num_units_per_core`, `tile_offset`, `channel_size`, `weight_num_tile`, `element_size`, `target.element_size()` | none | `WEIGHT=1` if `weight_has_value` (`:151-153`); `FP32_DEST_ACC_EN=1` if `fp32_dest_acc_en` (`:155-157`) | **O2** (resolved; field absent — `grep -n opt_level` on the factory returns nothing) | `ReaderConfigDescriptor{}` (`:173`) |
| writer | `device/kernels/writer_moreh_nll_loss_step1.cpp` | `all_cores` | `TensorAccessorArgs(output)` block only (`:145-146`) | none | 3 per core (`:217`): `output_buf`, `num_units_per_core`, `tile_offset` | none | none (`writer_defines` declared at `:149`, never populated) | **O2** (resolved; field absent) | `WriterConfigDescriptor{}` (`:181`) |

*Runtime kernel-source selection*: the reader's `kernel_source` is chosen at `:158-162` from
`use_large_algorithm`. Both sources are entry points of the same `KernelDescriptor`, so **both convert
together** with the factory — that is the true size of this port (1 factory + 3 kernel entry points).
There is no third axis: the writer is fixed, and the `WEIGHT` / `FP32_DEST_ACC_EN` defines select
*within* a source, not between sources.

### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) | allocated when |
|---|---|---|---|---|---|---|
| `c_0` target (`:75-83`) | `target_tile_size` | `all_cores` | `tt::DataFormat::Int32` (literal) | `target_tile_size` | unset | always |
| `c_1` weight (`:91-99`) | `weight_cb_tiles * data_tile_size` | `all_cores` | `data_format` | `data_tile_size` | unset | `weight_cb_tiles > 0`, where `weight_cb_tiles = use_large_algorithm ? 1u : weight_num_tile` (`:89-90`) |
| `c_24` intermed (`:104-112`) | `intermed_tile_size` | `all_cores` | `intermed_data_format` | `intermed_tile_size` | unset | always |
| `c_16` output (`:115-123`) | `data_tile_size` | `all_cores` | `data_format` | `data_tile_size` | unset | always |
| `c_7` weight scratch (`:129-137`) | `data_tile_size` | `all_cores` | `data_format` | `data_tile_size` | unset | `weight_has_value` (`:125`) |

No `CBDescriptor` sets `.global_circular_buffer`, `.address_offset`, or `.buffer`. No
GlobalCircularBuffer, no borrowed memory, no aliasing (every `format_descriptors` list has exactly one
element).

Note `c_0`: the **size** derives from the target tensor's real dtype
(`target_tile_size = tt::tile_size(target_data_format)`, `:56`) while the **declared format** is a
hardcoded `Int32`. Preserved exactly as-is.

### Semaphores

none — `grep -rni semaphore` over the op directory returns zero hits.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `..._program_factory.cpp:142` (`TensorAccessorArgs(*target.buffer())`) | `tensor_args.target_tensor` | reader idx 0 (`:206`, `Buffer*`) |
| `..._program_factory.cpp:143` (`TensorAccessorArgs(weight-or-nullptr)`) | `tensor_args.weight_tensor` (optional) | reader idx 1 (`:207`, `Buffer*` or `nullptr`) |
| `..._program_factory.cpp:146` (`TensorAccessorArgs(*output.buffer())`) | `tensor_return_value` | writer idx 0 (`:217`, `Buffer*`) |

All three are **Case 1** (fed to a `TensorAccessor`; no raw base-pointer arithmetic). No accessor in
this op passes a 3rd (page-size) constructor argument. No `TensorParameter` relaxation.

### Work split
- Driver: `split_work_to_cores(grid, units_to_divide)` (`:45-46`), where
  `units_to_divide = target.physical_volume() / H / W * (Ht * Wt)` (`:39`).
- `num_cores`, `all_cores`, `core_group_1`, `core_group_2`, `units_per_core_group_1`,
  `units_per_core_group_2` — all bound; only `num_cores` / `all_cores` / the two groups / the two
  counts are used, and the per-group counts feed **RTAs**, not CTAs.

### Shared kernels

**none.** All three `kernel_source` paths (`:158-165`) point inside this op's own `device/kernels/`, and
`grep -rl <kernel-filename> ttnn/cpp/ttnn/operations/` returns only this factory for each (plus the
family `CMakeLists.txt` glob, which is not a consumer). No `_metal2` fork exists beside any of them and
none is needed. Nothing to sunset, nothing to coordinate.

The single out-of-directory dependency is a **header**, not a borrowed kernel file:
`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (`read_tile`, `read_value`, `read_line`,
`get_tilized_idx`, `union Scalar`). Every consumed signature already takes `DataflowBuffer` **by value**,
so no donor-side change and no fork.

### Flags
- No unreferenced kernel files in the directory — all three are bound by the factory.
- Every descriptor the factory uses maps onto an audit Appendix A entry; no unknown descriptor type.
- `writer_defines` (`:149`) is declared and never populated — a no-op that vanishes with the port.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: `Factory::create_descriptor` becomes `Factory::create_program_artifacts`
  in the existing nested struct; the device-op header swaps the declaration and the `ProgramDescriptor`
  include for `ttnn/metal_v2_artifacts.hpp`. No pybind change — `moreh_nll_loss_nanobind.cpp` binds
  only the user-facing `ttnn::moreh_nll_loss`, never a factory entry point.

## Planned Spec Shape

- **KernelSpecs (2)**:
  - `READER` — source runtime-selected (small / large); DFB + tensor bindings, 1 named CTA, 5 named RTAs.
  - `WRITER` — 1 DFB binding, 1 tensor binding, no CTAs, 2 named RTAs.

  One reader `KernelSpec`, not two: the two sources are alternatives *for one program*, not a work
  split. Only one is ever compiled per program.
- **DataflowBufferSpecs (up to 4)** — one per surviving legacy `CBDescriptor`:

  | DFB name | from | entry_size | num_entries | data_format_metadata | declared when |
  |---|---|---|---|---|---|
  | `target` | `c_0` | `target_tile_size` | 1 | `Int32` | always |
  | `weight` | `c_1` | `data_tile_size` | `weight_dfb_tiles` | `data_format` | `weight_dfb_tiles > 0` (legacy guard verbatim) |
  | `output` | `c_16` | `data_tile_size` | 1 | `data_format` | always |
  | `weight_scratch` | `c_7` | `data_tile_size` | 1 | `data_format` | **`weight_has_value && !use_large_algorithm`** (tightened — see Applied Patterns) |

  `c_24` gets **no spec** — dead-CB drop (see Dropped Plumbing). No `tile_format_metadata` on any DFB
  (no legacy `CBFormatDescriptor` set `.tile`). No `borrowed_from`, no `alias_with`, no
  `allow_instance_multi_binding` anywhere.
- **SemaphoreSpecs**: none.
- **TensorParameters (2 or 3)**: `target` (always), `weight` (iff `weight_has_value`), `output` (always).
- **WorkUnitSpecs (1)**: `{READER, WRITER}` over `all_cores`. Both legacy `KernelDescriptor`s carry
  `core_ranges = all_cores`, so one work unit reproduces placement exactly.
- **Op-owned tensors**: none — the legacy factory allocates no device tensor beyond the op's io.

### DFB endpoint census — re-derived, not transcribed

Per node, per config. An endpoint is any kernel that touches the buffer (FIFO produce, FIFO consume, or
raw-pointer peek).

| DFB | Config | Distinct touchers | Roles observed | Disposition |
|---|---|---|---|---|
| `target` | A, B, C | 1 — reader | produces via donor `read_tile` (`moreh_common.hpp:678`,`:690`); consumes in body (`wait_front` small `:70` / large `:63`, `get_read_ptr` `:73`/`:66`, `pop_front` `:97`/`:100`) | **self-loop** — reader PRODUCER **and** CONSUMER |
| `weight` | A | not allocated | — | no DFB |
| `weight` | B | 1 — small reader | produces via donor `read_line` (`:750`,`:806`); consumes in body (`wait_front` `:60`, `get_read_ptr` `:61`) | **self-loop** |
| `weight` | C | 1 — large reader | produces via donor `read_value` (`:706`,`:717`); consumes in body (`wait_front` `:81`, `get_read_ptr` `:82`, `pop_front` `:86`) | **self-loop** |
| `weight_scratch` | A | not allocated | — | no DFB |
| `weight_scratch` | B | 1 — small reader, **role-free** | donor `read_line` uses it as an `async_read` destination and reads `get_write_ptr()` (`:782-784`, `:797`); **no FIFO ops at all** | **self-loop** (label cosmetic on Gen1) |
| `weight_scratch` | **C** | **0** — large reader never names it (its buffer set is target / weight / output only) | — | **conditional DFB** — tighten the guard; **not** a drop, B needs it |
| `output` | A, B, C | 2 — reader + writer | reader locked PRODUCER (`reserve_back` small `:69` / large `:62`, `push_back` `:95`/`:98`); writer locked CONSUMER (`wait_front` `:29`, `pop_front` `:32`) | **plain 1P + 1C** — no flag |
| `c_24` intermed | A, B, C | **0** in every config | — | **dead-CB drop** |

**My census agrees with the brief in every particular.** Two points worth stating explicitly:

- `output`'s maximum census is 2, and that pair is exactly one locked producer + one locked consumer, so
  it is a plain 1:1. The reader's `get_write_ptr()` on it (small `:72`, large `:65`) is a public peek on
  its own PRODUCER binding, not a third endpoint.
- **No DFB in this op needs `allow_instance_multi_binding`.** No buffer reaches 3 distinct touchers and
  no two kernels are locked to the same FIFO role. The hidden-co-filler shape cannot arise here: it needs
  semaphore-gated coordination and this op has no semaphores.

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** `split_work_to_cores` yields two core groups, but the
per-group difference (`units_per_core_group_1` / `_2`) is carried by a **runtime** arg
(`num_units_per_core`), not by a per-group CTA, in both the legacy factory and the port. No legacy
`KernelDescriptor` is instantiated twice from one source, so there is nothing to preserve and no
CTA→RTA demotion is being made (the value was already an RTA).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `..._program_factory.cpp:206` — reader RTA idx 0 | `target_buf` (`Buffer*`) | `TensorBinding{TENSOR_TARGET, "target"}` + `TensorParameter` |
| `..._program_factory.cpp:207` — reader RTA idx 1 | `weight_buf` (`Buffer*` or `nullptr`) | `TensorBinding{TENSOR_WEIGHT, "weight"}` + `TensorParameter`, both conditional on `weight_has_value` |
| `..._program_factory.cpp:217` — writer RTA idx 0 | `output_buf` (`Buffer*`) | `TensorBinding{TENSOR_OUTPUT, "output"}` + `TensorParameter` |
| `..._program_factory.cpp:142` | `TensorAccessorArgs(*target.buffer()).append_to(reader_cta)` | binding mechanism end-to-end; kernel `TensorAccessor(tensor::target)` |
| `..._program_factory.cpp:143` | `TensorAccessorArgs(weight-or-nullptr).append_to(reader_cta)` — the **placeholder block** that pinned the kernel's CTA offsets | gone: there are no CTA offsets to pin. `TensorAccessor(tensor::weight)` exists only under `WEIGHT` |
| `..._program_factory.cpp:146` | `TensorAccessorArgs(*output.buffer()).append_to(writer_cta)` | kernel `TensorAccessor(tensor::output)` |
| `reader_...step1.cpp:31-32`, `..._large.cpp:31-32` | `TensorAccessorArgs<1>()` / `TensorAccessorArgs<target_args.next_compile_time_args_offset()>()` | dropped — layout metadata is host-packed |
| `writer_...step1.cpp:17` | `TensorAccessorArgs<0>()` | dropped |
| `..._program_factory.cpp:213` — reader RTA idx 7 | `element_size` (host local at `:201`) | **dropped entirely** — dead in both readers (never read after `reader_...step1.cpp:20` / `..._large.cpp:20`). Per the invoker's decision: drop rather than name. The host local at `:201` goes with it. |
| `..._program_factory.cpp:214` — reader RTA idx 8 | `target.element_size()` | **dropped entirely** — dead in both readers (`:21` in each). Same decision. |
| `..._program_factory.cpp:141` — reader positional CTA 0 | `static_cast<uint32_t>(weight_has_value)` | named CTA `{"weight_has_value", …}` — **kept**, see below |
| reader RTAs idx 2-6 | positional | named: `ignore_index`, `num_units_per_core`, `start_id`, `C`, `weight_num_tile` |
| writer RTAs idx 1-2 | positional | named: `num_units_per_core`, `start_id` |
| `..._program_factory.cpp:104-112` | `CBDescriptor` for `c_24` | **no DFB spec** — dead in every config; a bindingless DFB is rejected by the validator. **Its `cb_usage` term at `:67-68` stays** (see Applied Patterns). |

**No magic CB indices to drop from CTAs.** Neither kernel ever received a CB index through an argument —
all four legacy CB handles were literal `tt::CBIndex::c_N` initializers in the kernel source
(`reader_...step1.cpp:23-26`, `..._large.cpp:23-26`, `writer_...step1.cpp:15`). They are replaced by
`dfb::` tokens; nothing leaves the CTA list on their account.

**No semaphore-ID RTAs** — the op has no semaphores.

### The one CTA that is kept, and why

`weight_has_value` (positional CTA 0) is **declared and never used** by either reader
(`reader_...step1.cpp:30`, `..._large.cpp:30`) — the weight paths are selected by `#if defined(WEIGHT)`
instead. Pre-port its *slot* was still load-bearing as positional padding that pinned
`TensorAccessorArgs<1>` at its offset; post-port that role is gone with the accessor plumbing, so it
becomes a purely dead named CTA.

**Decision: carry it across as a named CTA and keep the kernel-side read.** The port's default is 1:1
translation, and deleting it is a dead-code cleanup rather than port work — the same class of change the
scope discipline routes to the report rather than the diff. Keeping it also keeps the two halves of the
one host boolean visibly paired (`weight_has_value` CTA + `WEIGHT` define), which is what the brief's
"don't fix half of it" warning protects. Flagged in the report for the ops team, alongside the audit's
own note on it.

*(The two dead **RTAs** go the other way only because the invoker made that call explicitly; this CTA was
not part of that instruction, and I did not extend the deletion to it on my own.)*

## Applied Patterns

- **[Sync-free and single-ended CBs → self-loop DFB]** — `target` (all configs), `weight` (B, C), and
  `weight_scratch` (B) each have exactly one toucher, so the reader is bound both PRODUCER and CONSUMER
  with a single shared `accessor_name`. `weight_scratch` is additionally *entirely* sync-free (no FIFO
  op ever runs on it), so its labels are cosmetic on Gen1. All three are **DM self-loops**, legal on
  Gen1 and Quasar-uplift debt only.
- **[Conditional / optional DFB bindings]** — three coordinated conditionals, all keyed off host
  booleans already in scope:
  - `weight` DFB spec + reader binding, on the legacy guard `weight_dfb_tiles > 0`.
  - `weight` `TensorParameter` + reader `TensorBinding`, on `weight_has_value`.
  - `weight_scratch` DFB spec + reader binding, on **`weight_has_value && !use_large_algorithm`** — the
    guard tightened from the legacy `weight_has_value` alone. This is **new structure**: legacy gated
    the *use* by which kernel file it compiled, so there is no existing host conditional to translate.
    Without the tightening, config C allocates a buffer no kernel binds and the spec validator rejects
    the program.

  The kernel side needs **no new `#ifdef`s**: both readers already gate every weight reference under
  `#if defined(WEIGHT)`, which the factory already emits. The one subtlety is that `WEIGHT` and the
  `weight_scratch` binding **diverge in config C** (define on, binding off) — safe only because the large
  reader source never names `weight_scratch`. That coupling gets a comment at the binding site.
- **[Pass DFB handles directly to kernel-lib helpers]** — `read_tile` / `read_value` / `read_line` take
  `DataflowBuffer` by value and the kernels already pass named DFB locals, so constructing those locals
  from the tokens (`DataflowBuffer dfb_target_obj(dfb::target);`) leaves every call site unchanged. No
  `.id` extraction, no donor change, no fork.
- **Kernel-side whitelist rule 7 (DFB metadata via the object)** — two sites, both declared `const`
  (not `constexpr`), so both take the **member-getter** form:
  - `writer_...step1.cpp:25` `get_tile_size(cb_output)` → `dfb_out.get_tile_size()`.
  - `..._large.cpp:37` `get_tile_size(cb_weight)` → `dfb_weight_obj.get_tile_size()`. This one requires
    moving the `DataflowBuffer dfb_weight_obj` construction (currently `:54`) above the metadata read,
    since the getter needs the object. One object for the buffer, per the same-FIFO rule.

  The donor's own internal `get_tile_size(cb.get_id())` calls (`moreh_common.hpp:683`, `:709`, `:753`)
  are in a shared header outside the op directory and are **left alone**.
- **Not applied, recorded so the absence is deliberate**: no aliased DFBs, no same-FIFO aliasing, no
  borrowed-memory DFB, no multi-variant factory branch, no multiplicity/`WorkUnitSpec` split, no
  `allow_instance_multi_binding`, no varargs (RTA, CRTA, or CTA), no `override_runtime_arguments`, no
  unity-build symbol collision (this op has a single factory `.cpp`).

## Hardware configuration and compiler options

- **reader** — legacy `ReaderConfigDescriptor{}` resolves to `(RISCV_1, NOC_0, DM_DEDICATED_NOC)`, the
  reader default → `ttnn::create_reader_datamovement_config(device->arch())`.
- **writer** — legacy `WriterConfigDescriptor{}` resolves to `(RISCV_0, NOC_1, DM_DEDICATED_NOC)`, the
  writer default → `ttnn::create_writer_datamovement_config(device->arch())`.
- No custom NOC / processor / `noc_mode` anywhere; no `DM_DYNAMIC_NOC`.
- **No compute kernel exists in this op**, so there is no `ComputeHardwareConfig`, no `unpack_modes`
  question, and no FP32 `enable_32_bit_dest` interaction to reproduce.
- **`opt_level`**: `grep -n opt_level` on the legacy factory returns **nothing**, so both kernels
  resolve to the DM default `O2`, which is exactly Metal 2.0's `CompilerOptions` default. Neither
  `KernelSpec` sets it. The compute-kernel `O3` rule has a **denominator of zero** here — the factory
  builds no compute `KernelSpec` — so it does not apply rather than being skipped.
- **Gen2**: not populated anywhere; the two TTNN helpers supply the Gen2 branch for the default case.

## Deferred / Flagged

- **New finding — test coverage.** No test in the repository reaches **config C** (the large
  algorithm). `use_large_algorithm` requires `cb_usage >= available_L1` (~1 MiB on WH), which needs
  `weight_num_tile * data_tile_size` on that order, i.e. a channel size around 16k. The largest `C` in
  `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_nll_loss.py` is 300. So
  `reader_moreh_nll_loss_step1_large.cpp` — one of the two kernel entry points this port converts — is
  **not exercised by the confirmed test set**. Verification adds a targeted ad-hoc run for it; recorded
  in the report as a standing coverage gap.
- **Also reached only when `reduction == "mean"`.** `moreh_nll_loss.cpp:29-42` calls
  `prim::moreh_nll_loss_step1` on the mean branch only, so the `sum` / `none` parametrizations of the
  op's tests do not touch this factory at all. Relevant to reading the test result, not to the port.
- No structural issue the audit missed. Nothing here changes the port's shape.

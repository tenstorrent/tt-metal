# Port Plan — `matmul` / `MatmulMultiCoreProgramFactory`

Port plan for `ttnn/cpp/ttnn/operations/matmul`, factory `MatmulMultiCoreProgramFactory`, ported from
`ProgramDescriptorFactoryConcept` (`create_descriptor`) to Metal 2.0 `ProgramSpecFactoryConcept`.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope: ONE factory.** The op directory holds two DeviceOperations and eight ProgramFactories; only
`MatmulMultiCoreProgramFactory` was audited and only it is ported. The other seven stay on their
legacy concepts; the `program_factory_t` variant is valid with factories on mixed concepts and the
framework dispatches per factory, so the op keeps building and running.

---

## Legacy Inventory

### Legacy factory shape

- **Concept:** `ProgramDescriptorFactoryConcept` — `create_descriptor` returning a
  `tt::tt_metal::ProgramDescriptor`, declared at `device/factory/matmul_multicore_program_factory.hpp:14`,
  defined at `device/factory/matmul_multicore_program_factory.cpp:27`.
- **Where the factory methods live:** in a proper `program_factory_t` variant
  (`device/matmul_device_operation.hpp:24-31`). **Not** the direct-descriptor shape, so
  `ttnn_factory.md` exception 3 does not apply — this is a method swap inside the existing struct.
- **Variants:** single. The factory has one code path; the only branch is whether `core_group_2` is
  non-empty.
- **Custom `compute_program_hash`:** **none framework-visible** — the op uses the default reflection
  hash. A deliberately-renamed helper `compute_descriptor_program_hash` sits at
  `device/matmul_device_operation.hpp:50` with a comment explaining it is *intentionally* not named
  `compute_program_hash`, plus a pybind that exposes it under that name
  (`matmul_nanobind.cpp:1233-1237`). **Recorded so it is left alone.** Not a custom hash; not the
  port's to touch.
- **`override_runtime_arguments`:** absent from this factory. Target concept is therefore the base
  `ProgramSpecFactoryConcept`, and no override is added.

### Kernels

Four `KernelDescriptor`s over three sources (the fourth is the conditional second compute instance).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp` (:141-142) | `all_cores` | `{last_ktile_w, last_ktile_h}` then `TensorAccessorArgs(a).append_to(...)` then `TensorAccessorArgs(b).append_to(...)` (:136-138) | `cb_in0`=`c_0`, `cb_in1`=`c_1` (:146) | per-node, 12 slots: `{a, b, Mt, Kt, Nt, MtKt, KtNt, B, bcast_batch, num_tiles_written, num_output_tiles_per_core, MtNt}` (:174-187) | none | none | field absent → resolves **O2** | `ReaderConfigDescriptor{}` (:147) |
| writer | `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (:154-155) | `all_cores` | `TensorAccessorArgs(output).append_to(...)` only (:150-151) | `cb_out`=`c_16` (:159) | per-node, 3 slots: `{output, num_output_tiles_per_core, num_tiles_written}` (:188) | none | none | field absent → resolves **O2** | `WriterConfigDescriptor{}` (:160) |
| compute (group 1) | `device/kernels/compute/bmm.cpp` (:208) | `core_group_1` | `{1, 1, Kt, num_output_tiles_per_core_group_1}` (:205) | `cb_in0`=`c_0`, `cb_in1`=`c_1`, `cb_out`=`c_16` (:212-213) | none | none | `mm_kernel_defines` — stagger + throttle, from `add_stagger_defines_if_needed` / `throttle_mm_perf` (:196-200, :214) | field absent → resolves **O3** (`ComputeConfigDescriptor`) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` (:215-219) |
| compute (group 2) *(conditional: `!core_group_2.ranges().empty()`)* | same source (:226) | `core_group_2` | `{1, 1, Kt, num_output_tiles_per_core_group_2}` (:223) | same three (:230-231) | none | none | same (:232) | field absent → resolves **O3** | same (:233-237) |

`grep -n opt_level` on the factory returns **nothing**, so both DM descriptors resolve to legacy
`O2` and both compute descriptors resolve to legacy **`O3`**.

### CBs

Three `CBDescriptor`s, each single-element `format_descriptors` (no aliasing), no `.tile` set, no
`address_offset`, no `.buffer`, no GlobalCircularBuffer.

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` = 0 (in0) | `2 * in0_single_tile_size` (:103-111) | `all_cores` | `in0_data_format` | `in0_single_tile_size` | not set |
| `c_1` = 1 (in1) | `2 * in1_single_tile_size` (:112-120) | `all_cores` | `in1_data_format` | `in1_single_tile_size` | not set |
| `c_16` = 16 (out) | `2 * output_single_tile_size` (:123-131) | `all_cores` | `output_data_format` | `output_single_tile_size` | not set |

**Endpoint census re-derived from the kernel bodies** (not transcribed from the brief) — per-node,
across all three kernels:

| CB | FIFO producer (locked) | FIFO consumer (locked) | distinct touchers | disposition |
|---|---|---|---|---|
| `c_0` | reader `dfb_in0.reserve_back` / `push_back` (reader :65,:80) | compute `in0_dfb.wait_front` / `pop_front` (bmm :46,:51) | 2, one locked to each role | plain 1P+1C, no flag |
| `c_1` | reader `dfb_in1.reserve_back` / `push_back` (reader :84,:87) | compute `in1_dfb.wait_front` / `pop_front` (bmm :47,:52) | 2, one locked to each role | plain 1P+1C, no flag |
| `c_16` | compute `out_dfb.reserve_back` / `push_back` (bmm :57,:63) | writer `dfb_out.wait_front` / `pop_front` (writer :40,:43) | 2, one locked to each role | plain 1P+1C, no flag |

Census agrees with the brief. The reader's `dfb_in0.get_write_ptr()` at reader :71 and :77 (feeding
`pad_last_ktile` / `pad_last_transposed_ktile`) is the CB's own FIFO producer peeking its own buffer
— a public peek that a PRODUCER binding already covers, **not** a third toucher. Nothing to
self-loop, nothing to assign, no `allow_instance_multi_binding`, no dead CB, no conditional DFB.

### Semaphores

**none** — this factory creates no semaphores.

### Tensor accessors

| host site | kernel site | originating Tensor | RTA slot (host) |
|---|---|---|---|
| `TensorAccessorArgs(a).append_to(reader_compile_time_args)` (:137) | `TensorAccessor(src0_args, src0_addr)` (reader :57) | input 0 (`a`) | reader slot 0 |
| `TensorAccessorArgs(b).append_to(reader_compile_time_args)` (:138) | `TensorAccessor(src1_args, src1_addr)` (reader :58) | input 1 (`b`) | reader slot 1 |
| `TensorAccessorArgs(output).append_to(writer_compile_time_args)` (:151) | `TensorAccessor(dst_args, dst_addr)` (writer :31) | output 0 | writer slot 0 |

All three are **Case 1** (base address consumed only by a `TensorAccessor`; no raw-pointer
arithmetic). All three are 2-argument constructions — **no page-size third argument anywhere**, so
nothing to drop on that account. No `TensorParameter` relaxation: no kernel uses
`ArgConfig::Runtime*`.

The addresses arrive in the tensor-object form (`reader_desc.emplace_runtime_args(core, {a, b, …})`),
not `->address()` — the descriptor framework auto-registers and patches them. Correct today, and
still real port work: each becomes a `TensorParameter` + `TensorBinding`.

### Work split

- Driver: `tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total)` (:80-87)
- `num_cores`, `all_cores`, `core_group_1`, `core_group_2`,
  `num_output_tiles_per_core_group_1`, `num_output_tiles_per_core_group_2`
- `all_cores == core_group_1 ∪ core_group_2` (the documented contract: group 1 is the
  greater-work set and equals `all_cores` when the work divides evenly, in which case group 2 is
  empty). This is what lets the reader's and writer's derived node sets come out as `all_cores`
  from two `WorkUnitSpec`s.
- `compute_with_storage_grid_size` comes from `pc.allowed_worker_cores.value().bounding_box().grid_size()`,
  auto-populated with a `log_warning` when the program config did not carry it (:65-76). Carried
  verbatim.

### Shared kernels

**The brief says "NONE — convert all three kernels in place". That is correct for the census the
audit ran, and wrong for this port.** The audit's census command is scoped to
`grep -rl <kernel-filename> ttnn/cpp/ttnn/operations/`, so it structurally cannot see a binder
outside the operations tree. A repo-wide census finds two:

| kernel | binders outside the op | rung |
|---|---|---|
| `device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp` | `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:623` (`TestGenericOpMatmul`, via `generic_op` + legacy `ProgramDescriptor`) | **2 — create the fork** |
| `device/kernels/compute/bmm.cpp` | `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:658` and `:668` (same test, two core groups); plus `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:1433` (`TestCrossOpCompilation` reads the source *text* and compiles a fused kernel from it) | **2 — create the fork** |
| `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | none — sole binder is this factory | **convert in place** |

**Why in-place conversion breaks those binders.** The Metal 2.0 generated headers
(`kernel_bindings_generated.h`, carrying `dfb::` / `tensor::` / `sem::`, and
`kernel_args_generated.h`, carrying `args::`) are emitted and auto-included **only** when
`JitBuildSettings::is_metal2_kernel()` is true. That flag defaults to false and is set true only on
the `ProgramSpec` path (`tt_metal/impl/metal2_host_api/program_spec.cpp:3125`); a kernel created
through the legacy descriptor API never gets them (`tt_metal/jit_build/genfiles.cpp:126-129`,
`:294-296`, `:550-559`). So a converted kernel source cannot JIT-compile for a legacy binder, and
`TestGenericOpMatmul` would fail at runtime rather than at build time.

Rung 2 taken per the shared-kernel Caution: the fork lands **beside the original**, in matmul's own
directory (the *lent* case), the factory binds the fork, and the original keeps serving the legacy
binder with a pointer comment added. `file(GLOB_RECURSE kernels device/kernels/*.cpp …)` in
`CMakeLists.txt` already covers the directory, so no build-system change is needed for the new files.

**Fork binding vocabulary** (the interface every later consumer inherits and cannot rename):
`dfb::in0`, `dfb::in1`, `dfb::out` — taken from the kernels' own named-CTA keys `cb_in0` / `cb_in1` /
`cb_out` with the `cb_` prefix dropped, which is also what the brief specifies. Tensors:
`tensor::in0`, `tensor::in1` in the reader (brief), `tensor::output` in the writer.

The writer's brief-noted 24-to-1 filename decoy is real and was re-checked: 23 of the 24 binders of
`writer_unary_interleaved_start_id.cpp` bind a *different same-named copy* (22 in
`eltwise/unary/…`, one in `data_movement/slice/…`), and `ls` of
`device/kernels/dataflow/` shows no `_metal2` sibling of matmul's copy, so the two existing
`_metal2` forks of that filename (`eltwise/unary/…`, `copy/typecast/…`) are siblings of other copies
and must not be bound.

### Flags

- **`in0_last_ktile_h` is hardcoded to 0** (:135), so the reader's
  `if constexpr (in0_last_ktile_h > 0) { … pad_last_transposed_ktile … }` block
  (reader :74-79) is unreachable from this factory. Live for sibling factories' transposed paths.
  Carried across unchanged; flagged for the op owner in the audit's Misc anomalies, not for the port.
- **Dead preprocessor branches in matmul's private writer copy** — `OUT_SHARDED` (writer :24) and
  `BACKWARDS` (writer :33) are never defined by this factory. Carried across unchanged.
- **`create_descriptor`'s fourth parameter `core_range_set` is accepted and ignored**, spelled
  `/*core_range_set*/` (:31). Dropped by the port (`ttnn_factory.md` exception 2).
- **No unreferenced kernel file** in the op's own `device/kernels/` that this factory owns; the other
  kernel sources there belong to the seven un-audited factories.
- **No descriptor type outside the audit's scan** appears in this factory.

---

## TTNN ProgramFactory

- **Concept (inherited from audit):** `ProgramSpecFactoryConcept` (base). The ported-from factory has
  no `override_runtime_arguments`, so the framework refreshes tensor bindings on cache hit and the
  port writes one method, `create_program_artifacts`. **No override is added.**
- **Custom `compute_program_hash`:** none — default reflection hash. The renamed
  `compute_descriptor_program_hash` helper (`device/matmul_device_operation.hpp:50`) and its pybind
  name stay exactly as they are.
- **Implementation notes:**
  - Two sanctioned device-op-class edits are forced (`ttnn_factory.md` exceptions 1 and 2):
    delete the whole `nb::class_<ttnn::prim::MatmulMultiCoreProgramFactory>` block at
    `matmul_nanobind.cpp:1260-1274` (its only member is the vanishing `create_descriptor`), and drop
    the ignored `core_range_set` parameter. The separate `nb::class_<MatmulDeviceOperation>` block at
    `matmul_nanobind.cpp:1222-1237` is left untouched. Exception 3 does not apply.
  - Spec-name constants are declared as **function-local** typed `const`s rather than
    anonymous-namespace file-scope ones. `TT_ENABLE_UNITY_BUILD(ttnn_op_matmul)` is on and all six
    matmul factory `.cpp`s are in that unity target, so file-scope `READER` / `IN0` constants here
    would collide with the sibling ports planned for the same directory. Function-local `const`s are
    the collision-free form of the same typed-constants pattern.

---

## Planned Spec Shape

Default 1:1 with legacy.

- **KernelSpecs (4)** — one per legacy `KernelDescriptor`:
  - `READER` `"reader"` → `…/reader_bmm_8bank_output_tiles_partitioned_metal2.cpp`
  - `WRITER` `"writer"` → `…/writer_unary_interleaved_start_id.cpp` (converted in place)
  - `COMPUTE_G1` `"compute_g1"` → `…/bmm_metal2.cpp`
  - `COMPUTE_G2` `"compute_g2"` → same source, conditional on `core_group_2` non-empty
- **DataflowBufferSpecs (3)** — one per legacy `CBDescriptor`; no aliasing, no borrowed memory, no
  advanced options:
  - `IN0` `"in0"`: `entry_size = in0_single_tile_size`, `num_entries = num_input_tiles` (2),
    `data_format_metadata = in0_data_format`
  - `IN1` `"in1"`: `entry_size = in1_single_tile_size`, `num_entries = num_input_tiles` (2),
    `data_format_metadata = in1_data_format`
  - `OUT` `"out"`: `entry_size = output_single_tile_size`, `num_entries = num_output_tiles` (2),
    `data_format_metadata = output_data_format`
  - `tile_format_metadata` left `nullopt` on all three — the legacy `format_descriptors[i].tile` was
    unset. `num_entries * entry_size` reproduces each legacy `total_size` exactly.
- **SemaphoreSpecs:** none — legacy has no `SemaphoreDescriptor`.
- **TensorParameters (3)** — one per distinct originating tensor, `spec = <tensor>.tensor_spec()`,
  `relaxations` left default (strict): `IN0` `"in0"` ← `a`, `IN1` `"in1"` ← `b`,
  `OUTPUT` `"output"` ← `tensor_return_value[0]`.
- **WorkUnitSpecs (1 or 2)**:
  - `{READER, WRITER, COMPUTE_G1}` over `core_group_1`
  - `{READER, WRITER, COMPUTE_G2}` over `core_group_2` — only when non-empty
  - Derived placement: reader and writer land on `core_group_1 ∪ core_group_2 == all_cores`, matching
    their legacy `core_ranges`; each compute lands on its own group, matching its legacy
    `core_ranges`.
- **Op-owned tensors:** none.

### Endpoint bindings

| KernelSpec | DFB bindings | Tensor bindings |
|---|---|---|
| `READER` | `IN0` PRODUCER `"in0"`, `IN1` PRODUCER `"in1"` | `IN0`→`"in0"`, `IN1`→`"in1"` |
| `WRITER` | `OUT` CONSUMER `"out"` | `OUTPUT`→`"output"` |
| `COMPUTE_G1` / `COMPUTE_G2` | `IN0` CONSUMER `"in0"`, `IN1` CONSUMER `"in1"`, `OUT` PRODUCER `"out"` | none — a compute kernel cannot bind a `TensorAccessor` |

Per-node census over the whole footprint: every node in `all_cores` runs exactly one reader, exactly
one writer and exactly one compute instance, so each of the three DFBs sees exactly 1 PRODUCER + 1
CONSUMER per node. The two compute `KernelSpec`s share each role legally because their node sets are
disjoint; they agree on `access_pattern` (default STRIDED), `num_threads` (1) and kernel kind
(compute), which is what the role-uniformity check requires.

---

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `compute_desc_1` over `core_group_1` (CTA `Nt = num_output_tiles_per_core_group_1`), `compute_desc_2` over `core_group_2` (CTA `Nt = num_output_tiles_per_core_group_2`) — both of `bmm.cpp` | `COMPUTE_G1`, `COMPUTE_G2` — both of `bmm_metal2.cpp`, differing only in the `Nt` CTA | `core_group_1`, `core_group_2` — **disjoint** node sets | `IN0` (CONSUMER on both), `IN1` (CONSUMER on both), `OUT` (PRODUCER on both) |

Each node sees exactly one compute instance, so these are ordinary single-role bindings — **not**
`allow_instance_multi_binding`, and not the same-grid two-toucher case. The per-group tile count
stays a **CTA**; demoting it to an RTA to collapse the two specs into one is the documented
anti-pattern and would cost the compile-time unrolling of the `Nt` loop in `bmm.cpp`.

The reader and writer keep one `KernelSpec` each, as in legacy (one descriptor over `all_cores`).

---

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory :137 + reader :30 (`TensorAccessorArgs<2>()`) + reader :15 (RTA slot 0 `src0_addr`) | `TensorAccessorArgs(a).append_to(cta)` + address in RTA slot 0 | `TensorParameter IN0` + `TensorBinding` on `READER`; kernel `TensorAccessor(tensor::in0)` |
| factory :138 + reader :31 (`TensorAccessorArgs<src0_args.next_compile_time_args_offset()>()`) + reader :16 (RTA slot 1 `src1_addr`) | chained `TensorAccessorArgs` + address in RTA slot 1 | `TensorParameter IN1` + `TensorBinding` on `READER`; kernel `TensorAccessor(tensor::in1)` |
| factory :151 + writer :16 (`TensorAccessorArgs<0>()`) + writer :11 (RTA slot 0 `dst_addr`) | `TensorAccessorArgs(output).append_to(cta)` + address in RTA slot 0 | `TensorParameter OUTPUT` + `TensorBinding` on `WRITER`; kernel `TensorAccessor(tensor::output)` |
| factory :146 (`{"cb_in0", c_0}`, `{"cb_in1", c_1}`) + reader :37-38 | named CTA carrying a magic CB index | `DFBBinding` `IN0`/`IN1` PRODUCER on `READER` |
| factory :159 (`{"cb_out", c_16}`) + writer :15 | named CTA carrying a magic CB index | `DFBBinding` `OUT` CONSUMER on `WRITER` |
| factory :212-213 / :230-231 (`cb_in0`, `cb_in1`, `cb_out`) + bmm :27-29 | named CTAs carrying magic CB indices | `DFBBinding` `IN0`/`IN1` CONSUMER + `OUT` PRODUCER on both compute specs |
| factory :121 (`uint32_t output_cb_index = tt::CBIndex::c_16;`) | host-side magic CB index local | gone — no CB index exists in the spec |
| factory :136 (positional `{last_ktile_w, last_ktile_h}`) + reader :28-29 | positional CTAs | named CTAs `in0_last_ktile_w`, `in0_last_ktile_h` |
| factory :205 / :223 (positional `{1, 1, Kt, per_core}`) + bmm :22-25 | positional CTAs | named CTAs `batch`, `Mt`, `Kt`, `Nt` |
| factory :174-187, reader :17-26 (slots 2-11) | positional RTAs | named RTAs `Mt`, `Kt`, `Nt`, `MtKt`, `KtNt`, `batch`, `bcast_B`, `output_tile_start_id`, `num_output_tiles`, `MtNt` |
| factory :188, writer :12-13 (slots 1-2) | positional RTAs | named RTAs `num_pages`, `start_id` |
| writer :19 (`get_local_cb_interface(dfb_id_out).fifo_page_size`) | CB-interface field read by id | `dfb_out.get_entry_size()` — a member getter; the legacy value is `const uint32_t`, not `constexpr` (whitelist §B) |
| reader :70, :76 (`get_dataformat(dfb_id_in0)`) | free helper on a CB id | `get_dataformat(dfb::in0)` — **stays** a free function with the binding token, because the legacy declaration is `constexpr` and no member getter can yield a constant expression (whitelist §A) |
| `#include <tt-metalium/tensor_accessor_args.hpp>` (factory :13) | host-side accessor-args plumbing | gone with the bindings |

**Page-size third argument:** none. All three accessor constructions are 2-argument, so there is no
third-argument CTA/RTA to drop.

**Semaphore-ID RTAs:** none — no semaphores.

---

## Applied Patterns

- **[Demoting per-group CTA to RTA — avoided]**: the work-split multiplicity is preserved as two
  compute `KernelSpec`s of one source in two `WorkUnitSpec`s over disjoint node sets, both binding
  the same three DFBs with the same roles. The per-group `Nt` stays a CTA.
- **[Porting a shared kernel — rung 2]**: `_metal2` forks created beside the originals for the
  reader and `bmm.cpp` (the *lent* case; the binder is `test_generic_op.cpp`), with the pointer
  comment added to each original. The writer converts in place — sole binder.
- **[Pass DFB handles directly to LLKs]**: `compute_kernel_hw_startup`, `matmul_init`,
  `matmul_tiles`, `pack_tile` in `bmm_metal2.cpp` take `dfb::in0` / `dfb::in1` / `dfb::out`
  directly, via `DFBBindingToken::operator uint32_t()`. No `.id` extraction, no temporary
  `DataflowBuffer`.
- **[Removing pybound legacy factory entry points]**: the
  `nb::class_<ttnn::prim::MatmulMultiCoreProgramFactory>` block is deleted; recorded under Handoff
  points as a user-visible API surface change.
- **[Unity-build hygiene]**: spec-name constants are function-local, not anonymous-namespace
  file-scope, because six matmul factories share one unity target and sibling ports are queued.
- **Crossing the boundary (`pad_tile.hpp`)**: the reader's calls to
  `pad_last_ktile<DataFormat, uint32_t>(uint32_t)` and
  `pad_last_transposed_ktile<…>` in `ttnn/cpp/ttnn/operations/kernel_helper_functions/pad_tile.hpp`
  take a raw `uint32_t` L1 address and a `DataFormat` NTTP. Neither needs a `sem::` or `tensor::`
  handle, so the boundary assumption holds and the donor needs no change. The address comes from
  `dfb_in0.get_write_ptr()` (a public peek) and the `DataFormat` from the `constexpr`
  `get_dataformat(dfb::in0)` token form.

Not applied, and why: self-loop DFB (no one-toucher CB), two-toucher 1P+1C assignment (every CB
already has one kernel locked to each role), aliased DFBs (every `format_descriptors` is
single-element), same-FIFO aliasing (no kernel-side or host-side CB-index alias), conditional /
optional bindings (no config-dependent resource), multi-variant factory (single code path),
varargs (every argument is a distinct field read once at a constant index).

---

## Deferred / Flagged

- **New finding — the shared-kernel census was incomplete in the brief.** Two of the three kernels
  have binders outside `ttnn/cpp/ttnn/operations/`, which the audit's census command cannot reach.
  See [Shared kernels](#shared-kernels) above. The invoker was consulted and confirmed rung 2. The
  audit recipe's census command is a doc-improvement candidate (recorded in the port report).
- **RTA→CRTA candidates, deliberately not converted.** Eight of the reader's ten named RTAs are
  node-invariant (`Mt`, `Kt`, `Nt`, `MtKt`, `KtNt`, `batch`, `bcast_B`, `MtNt`) and would dispatch
  more efficiently as common runtime args. RTA→CRTA changes dispatch semantics, so it is a separate
  later pass, not port work. Recorded in the port report.
- **No structural issue the audit missed beyond the census scope** — every feature gate, endpoint
  disposition and hardware-config finding in the brief was re-checked against the code and agrees.
- **Verification is build-only in this workspace.** The container has no attached device
  (`/dev/tenstorrent` absent, `lspci` shows no card, `tt-smi -ls` reports none), so the invoker runs
  the confirmed test set. Recorded in the port report with the exact commands.

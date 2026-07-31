# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/concat`

One DeviceOperation in this directory, with six program factories (all declared in the
`program_factory_t` variant at `concat_device_operation.hpp:29-35`):

- **`ttnn::prim::ConcatDeviceOperation`**
  - `ConcatProgramFactory` (`device/concat_program_factory.cpp`) — interleaved inputs, and the fall-through default
  - `ConcatS2IProgramFactory` (`device/concat_s2i_program_factory.cpp`) — sharded in / interleaved out
  - `ConcatS2SMultiProgramFactory` (`device/concat_s2s_multi_program_factory.cpp`) — sharded→sharded, dim 2 or 3, N inputs
  - `ConcatS2SRMProgramFactory` (`device/concat_s2s_rm_program_factory.cpp`) — sharded→sharded, row-major, **exactly 2** inputs, dim 3
  - `ConcatS2STiledProgramFactory` (`device/concat_s2s_tiled_program_factory.cpp`) — sharded→sharded, tiled, **exactly 2** inputs, dim 3
  - `ConcatBlockShardedProgramFactory` (`device/concat_block_sharded_program_factory.cpp`) — block-sharded, ≤16 inputs

Kernel files in the directory: 9, **all referenced** by a factory — no unreferenced/dead kernel files.
One kernel is referenced but **missing from the repo** (see *Misc anomalies* item 1).

There is no `experimental/quasar` copy of concat, so there is no pre-port copy to be misled by.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `74e2d788513 2026-07-31 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/concat` |
| **Overall** | **RED** at op level; subset `ConcatS2SRMProgramFactory` + `ConcatS2STiledProgramFactory` is clear |
| **DOps / Factories** | `ConcatDeviceOperation` → `ConcatProgramFactory`, `ConcatS2IProgramFactory`, `ConcatS2SMultiProgramFactory`, `ConcatS2SRMProgramFactory`, `ConcatS2STiledProgramFactory`, `ConcatBlockShardedProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 11 existing kernels (9 own + 2 donors) are structurally Device 2.0; no holdovers found |
| *Prereqs* — Cross-op escapes | Ok — no `#include` leaves `tt_metal/hw/inc/api/**` in any kernel. Two *file-path* borrows, confined to `ConcatProgramFactory` |
| *Feature Support* — overall | **RED** |
| *Feature Support* — Variadic-CTA | **Unsupported → RED** on `ConcatProgramFactory` (`device/concat_program_factory.cpp:200-205`; kernels `device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp:23-24,57,70` and `.../reader_concat_interleaved_start_id.cpp:23-24`) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — `yes` on all six factory rows; cross-check clean |
| *TTNN Readiness* — Concept (current) | `descriptor` (all six rows; each factory defines `create_descriptor`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — concept is `descriptor` |
| *TTNN Readiness* — Is safe to port? | Yes (all six rows) |
| *TTNN Readiness* — Custom hash | No — no `compute_program_hash` anywhere in the op |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | No — method absent |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `concat_nanobind.cpp` binds no device-op/factory internals |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | **none** — no `->address()` expression exists anywhere in the op |
| *Port work* — Tensor bindings (per binding) | subset: all **clean** (borrowed-memory DFB). Blocked factories: `ConcatProgramFactory` Case 1 ×(N+1), `ConcatS2IProgramFactory` Case 1 ×1 + clean ×N |
| *Port work* — TensorParameter relaxation | none (sheet: `none` on all six rows; no custom hash to reconcile) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor in the op passes a 3rd argument |
| *Port work* — CB endpoints | subset: `1P+1C` ×3 (`ConcatS2SRMProgramFactory`), `legal 1:1` ×6 + `self-loop` ×1 (`ConcatS2STiledProgramFactory`). No multi-binding, no dead CB |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves at port time
(a 1P+1C role assignment on the dual-instance factory, one self-loop on `ConcatS2STiledProgramFactory`).
Nothing in that subject blocks a Gen1 port.

## Result

**RED at op level; subset `ConcatS2SRMProgramFactory` + `ConcatS2STiledProgramFactory` is clear.**

Concat's device-op takes a **runtime-varying number of input tensors**
(`ConcatInputs::input_tensors` is a `std::vector<Tensor>`, `device/concat_device_operation_types.hpp:19-21`),
and four of the six factories build their per-kernel argument or resource set as a function of that
count. In Metal 2.0 a kernel can only *read* a compile-time value or a DFB through a name fixed when the
kernel source was written, so those four factories cannot be ported today. (Note the constraint is
kernel-side, not host-side — the host's `CompileTimeArgs` container is already variable-length; see
*Expected resolution* below.) The blockers, by owner:

1. **Variable-count compile-time arguments (CTA varargs) — Appendix A, `UNSUPPORTED`** →
   *wait-for-feature* (a CTA-vararg mechanism is on the host API roadmap).
   Affects **`ConcatProgramFactory`** — the interleaved factory, and also the fall-through default
   (`device/concat_device_operation.cpp:69`), so it is the highest-traffic path in the op.

2. **Kernels access DFBs by a runtime-computed index** → *recipe maintainer* (this construct has **no
   Appendix A entry** — see *Recipe notes* item 1) **and** the Metal 2.0 framework side.
   Affects **`ConcatS2SMultiProgramFactory`**, **`ConcatBlockShardedProgramFactory`**,
   **`ConcatS2IProgramFactory`**. Each declares an input-count-dependent number of DFBs, and its kernels
   use a runtime-computed index to access them. In Metal 2.0 each DFB a kernel uses must be referenced by
   a predefined name, so there is no way to access one by index at runtime.

3. **`ConcatS2IProgramFactory` binds a kernel source that does not exist in the repo** → *ops team*.
   Independent of Metal 2.0: the factory cannot build a program at all today. See *Misc anomalies* item 1.

**The path forward is narrow and concrete.** Every other gate is green: the op is fully on the
`ProgramDescriptor` API with a clean readiness row on every factory, every kernel it uses is already
Device 2.0, no host-folded offset base pointers exist, and no `TensorAccessor` passes a page-size
override. Nothing here needs an op refactor except (3), which is a pre-existing breakage rather than a
porting cost.

Blockers (1) and (2) share a root cause — the runtime-varying input count — and both walls are on the
kernel-read side, not the host side: `KernelSpec::compile_time_args` is a `Table` and `dfb_bindings` is a
`Group`, so a factory can already build `N` of either. **They are distinct as #45388 is currently
scoped, but plausibly closer than that suggests**, so treat them as one dependency to confirm rather than
two to file:

- (1) needs a variable-length compile-time argument list plus a way to read one by index — which is
  exactly #45388.
- (2) needs a kernel to reach a DFB by index. The parts are largely present: `DataflowBuffer` has a
  plain runtime constructor `DataflowBuffer(uint16_t logical_dfb_id)`
  (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:75`), `DFBBindingToken` is constructible from a
  `uint16_t` and converts back to `uint32_t` (`:46-58`), and the framework already assigns each DFB an id
  and **emits it as an implicit CTA** — `tt_metal/jit_build/genfiles.cpp:158` says so directly ("DFB
  tokens and semaphore ids are emitted as constexpr variables, i.e. as implicit CTAs"), one named
  `constexpr DFBBindingToken` per binding (`:194-198`). What is missing is only that those `N` tokens are
  emitted as `N` separate *names* rather than as an indexable collection — the same category of thing
  (1) makes variable-length.

So the honest position is: (2) *may* fall out of #45388, or may need a small follow-on; it is not
obviously an independent feature. That is a judgement from the headers, not a scoping statement — only
#45388 or its owner settles it (*Questions for the user* item 3). The practical consequence: confirm
scope before assuming either that one feature clears all four factories or that three of them need a
separately-filed capability.

**The clean subset is real and worth porting on its own.** `ConcatS2SRMProgramFactory` and
`ConcatS2STiledProgramFactory` are selected only when `input_tensors.size() == 2`
(`device/concat_device_operation.cpp:46-59`), so their argument and resource sets are structurally
fixed: fixed 14-entry CTA lists, no runtime args at all, exactly 3 tensor bindings, and a fixed DFB
set. Both are on the sheet's `llama` model list. A mixed-concept `program_factory_t` — two
alternatives on `ProgramSpecFactoryConcept`, four still on `ProgramDescriptorFactoryConcept` — is
supported by the framework: `AllFactoriesValid` requires each variant alternative to satisfy exactly
*one* concept, not the same one (`ttnn/api/ttnn/operation_concepts.hpp:174-189`), and the adapter
dispatches per-alternative through `std::visit` (`ttnn/api/ttnn/device_operation.hpp:227-243`).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** All six factory rows read `yes`.
  Cross-check against the code is clean on every cheaply-checkable column:
  - `Concept = descriptor` — each factory defines `static ProgramDescriptor create_descriptor(...)`
    (`device/concat_program_factory.hpp:14`, `device/concat_s2i_program_factory.hpp:14`,
    `device/concat_s2s_multi_program_factory.hpp:14`, `device/concat_s2s_rm_program_factory.hpp:14`,
    `device/concat_s2s_tiled_program_factory.hpp:14`, `device/concat_block_sharded_program_factory.hpp:13`).
    No mesh-workload return, no `create()`/`override_runtime_arguments()` pair.
  - `Custom hash = no` — no `compute_program_hash` (nor `attribute_values` / `to_hash` backdoor) in the op.
  - `Runtime-args update (get_dynamic_runtime_args) = no` — hook absent from `ConcatDeviceOperation`
    (`device/concat_device_operation.hpp:37-49` lists its full static surface).
  - `Override runtime args method? = no` — absent.
  - `Pybind descriptor = no` — `concat_nanobind.cpp` contains no `nb::class_` of the device-op and no
    `create_descriptor` binding.
  - `Secretly SPMD Workload?` — not applicable (concept is `descriptor`, and the column is blank).
  - **Factory-set match** — the sheet's six rows map one-to-one onto the six alternatives of
    `program_factory_t`. No phantom row, no missing row.
  - **Cross-column invariants** — `Op-owned tensors?` blank on a `descriptor` concept (consistent);
    `get_dynamic_runtime_args = no` (so the `legacy device-op` invariant cannot be violated).
  - `Is safe to port? = yes` and `Smuggled pointer = no` are the sheet owner's calls and were not
    re-derived. Worth noting they are *consistent* with the independent finding below that the op
    contains no `->address()` expression at all — every buffer reaches a kernel either as a
    framework-patched `Buffer*` runtime arg or as a borrowed-memory CB.

- **Device 2.0 (every kernel used):** **GREEN.** No violations table — there are no violations.
  All 11 kernel files the op's factories instantiate are structurally Device 2.0: `Noc`,
  `DataflowBuffer` / `CircularBuffer` wrapper objects, `CoreLocalMem`, `UnicastEndpoint`, and
  wrapper-method `get_read_ptr()` / `get_write_ptr()` calls. No `noc_async_read` / `noc_async_write`
  free functions, no `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, no
  `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`,
  no `get_noc_addr_from_bank_id`, no raw semaphore addresses (the op uses no semaphores at all).

  | Kernel file | Owner | Notes |
  |---|---|---|
  | `device/kernels/dataflow/reader_concat_interleaved_start_id.cpp` | concat | `get_tile_size(cb_id_in)` @ `:28` — **sanctioned** free function (kept by Device 2.0 itself; the migration guide's own migrated example at `docs/.../device_api_migration_guide.md:630` uses it). Not a holdover. |
  | `device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp` | concat | clean |
  | `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors.cpp` | concat | clean (`UnicastEndpoint` + `{.noc_x, .noc_y, .addr}`, the Device 2.0 form) |
  | `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors_tiled.cpp` | concat | clean |
  | `device/kernels/dataflow/writer_height_sharded_width_concat_two_tensors_tiled.cpp` | concat | clean |
  | `device/kernels/compute/height_sharded_width_concat_two_tensors.cpp` | concat | clean (compute kernel; `DataflowBuffer` + LLK `transpose`) |
  | `device/kernels/dataflow/reader_s2s_tensor_concat.cpp` | concat | clean |
  | `device/kernels/dataflow/reader_writer_block_sharded_concat.cpp` | concat | clean |
  | `device/kernels/dataflow/writer_s2i_width.cpp` | concat | clean |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | shared pool (`ttnn/cpp/ttnn/kernel/`) | clean — uses the Device 2.0 `CircularBuffer` wrapper @ `:23`. (CB→DFB is Metal 2.0 port work, not a Device 2.0 gap.) |
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | cross-family donor — `eltwise/unary` | clean — `get_local_cb_interface(cb_id_out).fifo_page_size` @ `:19` is a **sanctioned** free function. |

  **One kernel could not be checked because its file does not exist**:
  `ConcatS2IProgramFactory` names
  `device/kernels/dataflow/reader_s2i_width.cpp` (`device/concat_s2i_program_factory.cpp:54-55`) and
  that file is absent from the working tree *and* from a freshly-fetched `origin/main`. Since the
  factory cannot build at all, this is not scored as a Device 2.0 gap — it is *Misc anomalies* item 1.

- **Feature compatibility:** one entry fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `experimental::CreateGlobalCircularBuffer`, no `global_circular_buffer.hpp` include, no `CBDescriptor::global_circular_buffer` field set, no `remote_index` / `remote_cb_*` idiom, no 4-arg `experimental::CreateCircularBuffer(..., global_cb)`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` assignment, no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. The op's borrowed-memory CBs are declared via the descriptor-API `.buffer` field with the offset left at its default 0 (e.g. `device/concat_s2s_rm_program_factory.cpp:69,86`). The only textual hit is a *comment* mentioning the API (`device/kernels/dataflow/reader_writer_block_sharded_concat.cpp:21`). |
  | GlobalSemaphore | N/A | The op uses no semaphores of any kind — a case-insensitive search for `semaphore` across the whole op directory returns nothing. |
  | Variable-count compile-time arguments (CTA varargs) | **RED** | Fires on `ConcatProgramFactory`. Detail below. |

#### Variable-count compile-time arguments (CTA varargs) — UNSUPPORTED, RED

**Affected factory:** `ConcatProgramFactory` only. The other five factories emit a fixed-length CTA
list and read every CTA at a constexpr offset, so this entry does **not** fire on them.

**Op-level signal (a prompt, not a verdict).** `ConcatDeviceOperation::tensor_args_t` is
`ConcatInputs`, whose sole member is `std::vector<Tensor> input_tensors`
(`device/concat_device_operation_types.hpp:19-21`) — a variable-count input container. Per the entry's
own guidance this only sends you to read the kernel. Two kernel-level signals then fire.

**Signal 1 — host emits a CTA list whose length is the input count.**
`device/concat_program_factory.cpp:200-205`:

```cpp
KernelDescriptor::CompileTimeArgs reader_compile_time_args = {src0_cb_index, num_input_tensors};
reader_compile_time_args.insert(
    reader_compile_time_args.end(), page_size_per_tensor.cbegin(), page_size_per_tensor.cend());
for (uint32_t i = 0; i < num_input_tensors; ++i) {
    TensorAccessorArgs(*input_tensors[i].buffer()).append_to(reader_compile_time_args);
}
```

Length is `2 + 3N` where `N = input_tensors.size()` — the op's own comment states the same arithmetic
(`device/concat_device_operation.cpp:229-232`). `N` ranges from 2 up to 47 for the interleaved path
(`device/concat_device_operation.cpp:285`), decided per invocation.

**Signal 2 — the kernel reads a compile-time arg at a runtime-varying index (the decider).**
`device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp:70` (and identically at `:57`
inside the `WIDTH_CONCAT` branch):

```cpp
auto page_size = kernel_compile_time_args[page_size_base_idx + curr_tensor];
```

`curr_tensor` is a **runtime** value: initialized from runtime arg 1 (`:18`, `:47`) and advanced inside
the read loop (`:66`, `:83`). So the kernel selects a compile-time arg with an index it does not know
until run time.

**Read the form of that line carefully — it is not the `get_compile_time_arg_val` macro, and it could
not be.** The macro expands to `get_ct_arg<arg_idx>()`
(`tt_metal/hw/inc/api/compile_time_args.h:64`), which makes the index a **template parameter**
(`:25-29`), so a runtime index there would not compile. What the kernel indexes instead is
`kernel_compile_time_args` — the underlying `constexpr std::array<uint32_t, N>` object
(`tt_metal/hw/inc/api/compile_time_args.h:23`). `constexpr` on that *object* means it is initialized
during compilation and is usable in constant expressions; it places no restriction on how the object is
subscripted. A runtime subscript of it is an ordinary runtime load, and the result here is assigned to a
plain `auto` (not `constexpr`), so the legacy code is valid and compiles today. The same kernel uses the
macro for `num_tensors` at `:21` and reaches past it to the raw array at `:70` — the author needed a
runtime-selected lookup, which the macro cannot express.

**Why it still blocks the port.** The obstacle is not a C++ rule, it is that Metal 2.0 removes the
array. Named compile-time args are emitted as individual named constants, so there is no
`kernel_compile_time_args` object left for a runtime index to subscript, and no way to declare a
name set whose size is the op's runtime input count. (This form is *not* the shape Appendix A's
recognition text describes — see *Recipe notes* item 5.)

**Signal 2b — a kernel template instantiated over a variable count derived from a CTA.** Both readers,
at `:23-24`:

```cpp
constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
constexpr auto tensor_accessor_args =
    make_tensor_accessor_args_tuple<num_tensors, page_size_base_idx + num_tensors>();
```

`make_tensor_accessor_args_tuple<NUM_TENSORS, CTA_OFFSET>`
(`tt_metal/hw/inc/api/tensor/tensor_accessor_args.h:237-242`) expands to `NUM_TENSORS`
`TensorAccessorArgs<offset>` instantiations, each at a CTA offset walked out from the previous one.
The *number* of compile-time args consumed is a function of a CTA value. This signal also fires on
`device/kernels/dataflow/reader_concat_interleaved_start_id.cpp:23-24`, which otherwise takes its page
size from `get_tile_size(cb_id_in)` and so does not trip signal 2.

**False-positive guards checked and not applicable.** This is not the "variable-count input list whose
per-input data rides RTAs or runtime CB-streaming" case: the per-input page sizes and the per-input
`TensorAccessorArgs` both ride *compile-time* args, so the compile-time arg count is not fixed. It is
also not a fixed-count input list — `N` genuinely varies per call.

**Expected resolution — tracked as issue #45388.** Confirmed against the API rather than inferred from
Appendix A's prose: `KernelAdvancedOptions` carries a **Compile time varargs** section whose field is
deliberately absent (`tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp:61-66`):

```cpp
//--------------------------------
// Compile time varargs
//--------------------------------
// TODO: This is currently unimplemented.
//       However, certain variadic kernels require this workaround.
//       (#45388 tracks the implementation of this feature.)
```

The port becomes possible when that lands. Do not demote the per-tensor CTAs to RTAs and do not
hand-unroll the loop.

**Where the gap actually sits — kernel side, not host side.** Worth stating precisely, because two
neighbouring capabilities *are* implemented and it is easy to conclude from them that this one is too:

- **Runtime and common-runtime varargs are fully implemented** — `num_runtime_varargs` and
  `num_common_runtime_varargs` on the same struct (`advanced_options.hpp:73,78`), with
  `ProgramRunArgs::runtime_varargs` / `common_runtime_varargs` carrying the values (`:149-159`) and
  kernel-side `get_vararg()`. This op's variable-count *runtime* arg blocks are therefore routine port
  work, not gates (see *RTA varargs for the blocked factories*).
- **The host-side compile-time-arg container is already variable-length** —
  `using CompileTimeArgs = Table<std::string, uint32_t>`
  (`tt_metal/api/tt-metalium/experimental/metal2_host_api/kernel_spec.hpp:192`), a name→value table
  sized when the factory runs. Nothing prevents a factory from putting `N` entries in it.

What is missing is the *kernel's* ability to read an `N`-sized set of them, and the reason shows up in
the accessor types (`tt_metal/hw/inc/experimental/kernel_args.h:53-66`):

```cpp
template <typename T> struct RtaArg  { uint32_t byte_offset; };
template <typename T> struct CrtaArg { uint32_t byte_offset; };
template <typename T> struct CtaVal  { T value; };
```

A runtime arg is a **byte offset** into a dispatch buffer, so varargs can sit past the named section and
be reached by index (`kernel_args.h:51-52`). A compile-time arg is a **value** baked into the compiled
kernel, reachable only through a name the JIT emits into a generated header and the kernel source
references literally — there is no addressable compile-time section, hence nothing to index
positionally. So the blocker is not that the host cannot *produce* `N` compile-time values; it is that
the kernel cannot *read* an `N`-sized set of them.

#### Kernels access DFBs by a runtime-computed index — not an Appendix A entry (flagged)

**Affected factories:** `ConcatS2SMultiProgramFactory`, `ConcatBlockShardedProgramFactory`,
`ConcatS2IProgramFactory`.

Each of these declares **N input DFBs** at `buffer_index` `0 .. N-1` — one per input tensor, borrowed
from that tensor's buffer — with `N` varying per invocation:

- `device/concat_s2s_multi_program_factory.cpp:83-103` (capped at 16, `:43`)
- `device/concat_block_sharded_program_factory.cpp:143-155` (capped at 16 by the device-op's validate,
  `device/concat_device_operation.cpp:156-160`)
- `device/concat_s2i_program_factory.cpp:32-47`

and each kernel then reaches an input DFB through a value it computes **at runtime**:

| Kernel | Site | How the DFB index arrives |
|---|---|---|
| `device/kernels/dataflow/reader_s2s_tensor_concat.cpp` | `:30` — `DataflowBuffer input_dfb(input_id);` | `input_id` is the counter of a loop bounded by CTA `num_input_tensors` (`:24`) |
| `device/kernels/dataflow/reader_writer_block_sharded_concat.cpp` | `:36`, `:44` — `DataflowBuffer src_dfb(src_dfb_id);` | `src_dfb_id` is read from a **runtime arg** (one per transfer descriptor) |
| `device/kernels/dataflow/writer_s2i_width.cpp` | `:28-29` — `DataflowBuffer input_shard_dfb(input_shard_dfb_id);` | `input_shard_dfb_id` is read from a **runtime arg** |

In Metal 2.0 each DFB a kernel uses must be referenced by a predefined name, so there is no way to
access one by index at runtime. `DataflowBufferSpec` carries a string `unique_id` and no numeric index
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp:79-81`); the kernel
reaches a DFB only through the `constexpr` token generated from its `DFBBinding` — `constexpr
DFBAccessor dfb::my_dfb`, constructed as `DataflowBuffer dfb(dfb::my_dfb)`, per
`shared/migration_guide.md` ("Construction from local accessor name. You construct your DFB from its
local accessor, not a magic-number `cb_id`"). That doc also lists magic-number CB indexing among the
patterns that "should **not** appear in a faithful Metal 2.0 port."

**The varying *count* is not itself the problem — locate the wall on the kernel side.**
`DataflowBufferSpec::unique_id` is a `std::string` and `KernelSpec::dfb_bindings` is a `Group`, so a
factory *can* generate `N` DFB specs and `N` bindings at run time. What has no expression is the kernel
source: it would have to name each of those `N` accessors literally, and it selects among them with a
value it computes at runtime. So a variable-*count* binding feature alone would not clear these three
factories — the kernels need a way to reach a DFB *by index*.

**How far off that is, is genuinely unclear — do not read this as a large missing feature.** The parts
are largely present: `DataflowBuffer` has a plain runtime constructor
`DataflowBuffer(uint16_t logical_dfb_id)` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:75`),
`DFBBindingToken` is constructible from a `uint16_t` and converts back to `uint32_t` (`:46-58`), and the
framework already assigns each DFB an id and **emits it as an implicit CTA** — `constexpr
DFBBindingToken <name>{id};`, one per binding (`tt_metal/jit_build/genfiles.cpp:158`, `:194-198`). So
today's wall is narrow: those `N` ids exist as `N` separate *names* rather than as an indexable
collection, and the kernel has no way to *obtain* the id for a DFB it did not name. (Note the type would
accept a runtime id — the gap is getting the right value, not passing it.) Whether closing that is inside
#45388's scope or a small follow-on is not answerable from the headers; see *Questions for the user*
item 3.

**This is not one of Appendix A's four entries.** I am reporting it as a blocker rather than waving it
through on the "not listed ⇒ supported" rule, because the construct has no expression in the target
API at all and classifying it as portable would send a porter into a wall. Routing: the **recipe
maintainer** (Appendix A needs an entry — see *Recipe notes* item 1) and the **Metal 2.0 framework
side** (the capability itself). It shares a root cause with the CTA-vararg gate, so one
variable-count-binding feature would likely clear both.

- **CB endpoints (GATE-free):** run for the clean subset (both factories' kernels are structurally
  Device 2.0, so the recognition signals are valid). No CB in the subset needs the multi-binding
  advanced option, and none is dead. Full census in *Port-work summary*; the blocked factories'
  censuses are in *Team-only* for planning.

- **Offset base pointers:** **GREEN — nothing to split out.** The op contains **no `->address()`
  expression of any kind** (verified by searching the whole op directory for `address()`), so there is
  no host-folded `base + offset` to classify. Every tensor reaches a kernel one of two ways:
  - as a `Buffer*` pushed into `KernelDescriptor::RTArgList` / `emplace_runtime_args`, which the
    framework auto-registers as a `BufferBinding` and delivers as a clean base
    (`tt_metal/api/tt-metalium/program_descriptors.hpp:173-186`); or
  - as a borrowed-memory CB via `CBDescriptor::buffer`, with `address_offset` left at its default 0.

  Type 3 (`address_offset`) is absent (see the Feature table). Type 4 (`ttnn::narrow` /
  `MeshBuffer::create(..., parent_base + offset)`) is absent. Concat appears in **neither** of the
  triage doc's Type-1/Type-2 tables (`analyses/2026-07-19_offset_base_pointers.md`), which agrees with
  the scan — this is a "no fold, op not in the tables" outcome, i.e. clean rather than
  merely-unlisted.

  **Do not mistake these for folds** — they are kernel-side offsets applied to a base the *kernel*
  obtained from a DFB, not host-folded pointers, and they port unchanged:
  `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors.cpp:42,45,70`
  (`output_dfb.get_write_ptr()` / `input_dfb_0.get_read_ptr() + input_start_0`),
  `device/kernels/dataflow/reader_s2s_tensor_concat.cpp:21,32`,
  `device/kernels/dataflow/reader_writer_block_sharded_concat.cpp:28,45`.

- **TensorAccessor 3rd argument:** **GREEN — no site exists.** Only three `TensorAccessor`
  constructions appear across every kernel the op uses, and all three pass exactly two arguments:
  - `device/kernels/dataflow/writer_s2i_width.cpp:24` — `TensorAccessor(dst_args, dst_addr)`
  - `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp:20` — `TensorAccessor(dst0_args, dst_addr)`
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:31` — `TensorAccessor(dst_args, dst_addr)`

  The reader accessors in `ConcatProgramFactory`'s kernels are built through
  `make_tensor_accessor_tuple`, which also constructs each accessor with two arguments —
  `TensorAccessor(std::get<Indexes>(args), get_arg_val<uint32_t>(start + Indexes))`
  (`tt_metal/hw/inc/api/tensor/tensor_accessor.h:626-637`) — so no page-size override rides that path
  either. Concat does not appear in `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, consistent
  with having no 3rd-arg site to classify.

## Port-work summary  *(mirrors the brief — scoped to the clean subset)*

### `ConcatS2SRMProgramFactory`

- **Tensor bindings** (3): `input_0`, `input_1`, `output` — **all clean**. Each is a borrowed-memory
  DFB (`CBDescriptor::buffer` set at `device/concat_s2s_rm_program_factory.cpp:69` for the two inputs
  and `:86` for the output); the kernel reads/writes them by raw pointer off the DFB, so the DFB *is*
  the tensor access. The causal-link gate applies — neither Case 1 nor Case 2. Port via
  `DataflowBufferSpec::borrowed_from` on three `TensorParameter`s.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none (no accessor in this factory).
- **CB endpoints** — one config axis only (the remainder-core split at
  `device/concat_s2s_rm_program_factory.cpp:190-200`, which changes which CTA set a node gets but not
  its toucher count). Every node carries **two instances of one kernel source**
  (`reader_height_sharded_width_concat_two_tensors.cpp` pushed into both a `ReaderConfigDescriptor` and
  a `WriterConfigDescriptor` KernelDescriptor over the same core range, `:166-188`) — the
  dual-instance work-split. The kernel contains **no FIFO operations at all**, so both touchers of every
  CB are role-free:

  | CB | buffer_index | Touchers on a node | Disposition |
  |---|---|---|---|
  | input 0 (borrowed) | 0 | 2 × raw read `input_dfb_0.get_read_ptr()` @ `:45` | **1P+1C** — bind one instance PRODUCER, the other CONSUMER |
  | input 1 (borrowed) | 1 | 2 × raw read `input_dfb_1.get_read_ptr()` @ `:70` | **1P+1C** |
  | output (borrowed) | 16 | 2 × raw write `output_dfb.get_write_ptr()` @ `:42` | **1P+1C** |

  No multi-binding flag anywhere: the census is exactly two role-free touchers per CB, which fits
  1P+1C. No third kernel touches any of them, and no CB is dead.

### `ConcatS2STiledProgramFactory`

- **Tensor bindings** (3): `input_0`, `input_1`, `output` — **all clean**, same borrowed-memory DFB
  shape (`device/concat_s2s_tiled_program_factory.cpp:102` for the inputs, `:116` for the output).
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** — three distinct kernels (reader, writer, compute), all over `all_cores`. Config
  axes are the `BF8` and `USE_SINGLE_PACKET_READ` defines (`:207-213`); neither changes any census.

  | CB | buffer_index | Producer | Consumer | Disposition |
  |---|---|---|---|---|
  | input 0 (borrowed) | 0 | reader `input0_dfb.push_back` @ reader `:53` | compute `wait_front`/`pop_front` @ compute `:68,71` | **legal 1:1** |
  | input 1 (borrowed) | 1 | reader `input1_dfb.push_back` @ reader `:54` | compute `wait_front`/`pop_front` @ compute `:75,78` | **legal 1:1** |
  | output (borrowed) | 2 | writer `reserve_back`/`push_back`/`get_write_ptr` @ writer `:40,43,61` | *none* | **self-loop** — 1 toucher; bind the writer PRODUCER **and** CONSUMER |
  | input0_transpose | 3 | compute `reserve_back`/`push_back` | reader `wait_front`/`pop_front`/`get_read_ptr` @ reader `:49,59,101` | **legal 1:1** |
  | input1_transpose | 4 | compute `reserve_back`/`push_back` | reader `wait_front`/`pop_front`/`get_read_ptr` @ reader `:50,103,145` | **legal 1:1** |
  | concat | 5 | reader `reserve_back`/`push_back`/`get_write_ptr` @ reader `:51,57,147` | compute `wait_front`/`pop_front` @ compute `:85,88` | **legal 1:1** |
  | output_transpose | 6 | compute `reserve_back`/`push_back` | writer `wait_front`/`pop_front`/`get_read_ptr` @ writer `:44,46,60` | **legal 1:1** |

  The compute kernel's FIFO operations on CBs 3/4/5/6 all happen inside its `transpose` helper
  (`device/kernels/compute/height_sharded_width_concat_two_tensors.cpp:12-31`), which takes the DFBs by
  reference — read that helper, not just `kernel_main`, when tracing the census.

  **Hidden-second-writer hunt: negative.** No kernel raw-writes a CB it does not FIFO-produce, and
  there are no semaphores to coordinate such a co-fill (the op declares none), so face (a) cannot
  apply. Every raw `get_read_ptr()`/`get_write_ptr()` above is a peek by the kernel that already holds
  the matching FIFO role, which is one toucher, not two.

  **One care point on the self-loop CB.** The compute kernel *constructs* a handle for the output CB
  (`compute:57`, `DataflowBuffer output_dfb(output_dfb_id)`) and then never touches it. It is therefore
  **not** a toucher and does not turn the self-loop into a 1P+1C — but in Metal 2.0 the unused
  construction cannot survive as-is, since `dfb::` handles only exist where the host declared a
  binding. See *Heads-ups* for what the porter does with it.

## Heads-ups  *(mirrors the brief — scoped to the clean subset)*

- **CB endpoints (multi-binding shapes to watch):** **none.** No CB in either subset factory reaches
  three touchers or doubles a FIFO role. The `ConcatS2SRMProgramFactory` CBs *look* like the
  multi-binding trap — one kernel source, two instances, both touching all three CBs — but every touch
  is sync-free, so the census fits 1P+1C and the flag stays off.
- **Unused DFB handle in the compute kernel:** `device/kernels/compute/height_sharded_width_concat_two_tensors.cpp:57`
  constructs `output_dfb` and never uses it. Metal 2.0 generates `dfb::` handles only for declared
  bindings, so the porter must either drop the dead construction or declare a binding for it. Dropping
  it is behavior-neutral — the object has no side effects — and is the smaller change; declaring a
  binding purely to keep the line would add a spurious third toucher to a CB whose census is otherwise
  a clean self-loop. Same situation, less visibly, for the reader/writer kernels: the three tiled
  kernels share one CTA list (`device/concat_s2s_tiled_program_factory.cpp:190-205`), so each reads
  only part of it — the reader never uses CTA 5/6 and the writer never uses CTA 0-4. In Metal 2.0 each
  `KernelSpec` declares only the named args its own kernel reads, so those unused entries simply do
  not carry over.
- **Cross-op / shared kernels:** **none for the subset.** All four kernel sources the two subset
  factories bind are concat-owned and live in this directory, and a filename census across
  `ttnn/cpp/ttnn/operations/` finds **no other op binding any of them** — so nothing is *lent* either,
  and no `_metal2` fork question arises. One intra-op note:
  `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors.cpp` is bound **twice by the
  same factory** (`ConcatS2SRMProgramFactory`, as both its reader and its writer instance) — that is
  the dual-instance work-split, not a shared-kernel situation, and both instances convert in the same
  change.
- **RTA varargs:** **none for the subset.** Neither `ConcatS2SRMProgramFactory` nor
  `ConcatS2STiledProgramFactory` sets a single runtime arg — everything is compile-time
  (`device/concat_s2s_rm_program_factory.cpp:104-164`, `device/concat_s2s_tiled_program_factory.cpp:190-205`).
  All 14 CTAs in each are read once as distinct fields, so they all become **named** compile-time args.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean** on function-call escape; a file-path borrow exists but is confined to
`ConcatProgramFactory` (a blocked factory).

- **Function-call escape: none.** Every kernel the op uses `#include`s only `api/**` headers from
  `tt_metal/hw/inc` (LLK / HAL / firmware — donor class 1, no concern) plus `<stdint.h>` / `<cstdint>`.
  No kernel includes another op's header, no `kernel_lib`, no
  `operations/kernel_helper_functions/`. There is therefore no per-call shape analysis to run and no
  summary table to fill: no `uint32_t sem_id`, no `TensorAccessorArgs<N>` hand-off, no NTTP CTA offset,
  no old-style addr-gen donor, no `CircularBuffer&` parameter anywhere.

- **Borrowed kernel files (file-path instantiation)** — two, both bound by `ConcatProgramFactory` only:

  | Kernel file | Owner | Also bound by | `_metal2` fork beside it? |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` (bound at `device/concat_program_factory.cpp:234`) | shared pool `ttnn/cpp/ttnn/kernel/` (donor class 3) | 3 other ops — `operations/embedding/device/embeddings_rm_program_factory.cpp`, `operations/embedding/device/embeddings_tilized_indices_program_factory.cpp`, `operations/data_movement/copy/device/copy_same_memory_config_program_factory.cpp` | **No** — directory listing shows only `writer_unary_stick_layout_interleaved_blocks.cpp` and `writer_unary_stick_layout_interleaved_start_id.cpp`. A future port of this factory creates the first fork. |
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (bound at `device/concat_program_factory.cpp:235`) | cross-family donor — `eltwise/unary` (donor class 6) | **28 other factories** (reduction/generic ×4, prod ×2, embedding, kv_cache, examples ×2, tilize_with_val_padding ×2, slice, permute, reshape_on_device, copy, transpose, attn_matmul, nlp_concat_heads, nlp_concat_heads_boltz, and more) | **No** — no `*_metal2*` file in that directory. |

  Both lists are **sunset and coordination lists, not authorization** to convert either file in place.
  Neither borrow is relevant to the clean subset — the subset owns all its kernels.

  *Classification note:* `ttnn/cpp/ttnn/kernel/` (singular) is the audit's donor class 3 for
  *function-call* escape ("treat as shared-lib class"), but `shared/port_patterns.md`'s shared-kernel
  exclusion list names only `ttnn/cpp/ttnn/kernel_lib/`. It is therefore unclear whether a *file-path*
  borrow from `ttnn/cpp/ttnn/kernel/` is fork-eligible or out of porter scope. Recorded in *Recipe
  notes* item 3; it does not affect this audit's verdict.

### CB endpoint censuses for the blocked factories  *(planning only — re-derive at re-audit)*

Recorded because they were cheap to establish while reading the kernels, and because they show the
same pattern the recipe predicts for concat. They are **not** porter input — these factories are
blocked, and their code may change before a re-audit.

| Factory | CB | Touchers on a node | Disposition |
|---|---|---|---|
| `ConcatProgramFactory` | `src0` (index 0, local, `device/concat_program_factory.cpp:126-134`) | reader FIFO-produces (`reserve_back`/`push_back`/`get_write_ptr`); writer FIFO-consumes (`wait_front`/`pop_front`) | **legal 1:1** |
| `ConcatS2SMultiProgramFactory` | inputs 0..N-1 (borrowed), output 16 (borrowed) | 2 instances of one source (`reader_s2s_tensor_concat.cpp`, reader-config + writer-config over one core range, `:146-172`), all touches sync-free | **1P+1C** on every CB |
| `ConcatBlockShardedProgramFactory` | inputs 0..N-1 (borrowed), output 16 (borrowed) | 2 instances of one source (`reader_writer_block_sharded_concat.cpp`, `:322-338`), all touches sync-free. Note the input reads target *remote* cores' instances via `src_dfb.get_read_ptr()` + `{.noc_x, .noc_y}` from runtime args | **1P+1C** on every CB |
| `ConcatS2IProgramFactory` | inputs 0..N-1 (borrowed) | writer FIFO-consumes (`wait_front`/`pop_front`, `writer_s2i_width.cpp:30,43`); the producing side lives in the **missing** reader kernel | **cannot be censused** — reader source absent |
| `ConcatS2IProgramFactory` | *(no output CB — the output is written through a `TensorAccessor`)* | — | — |

### RTA varargs for the blocked factories  *(planning only)*

All four blocked factories carry a genuine variable-count RTA block (recognition shape (a) — a counted
loop advancing `arg_index++` inside the loop body). Metal 2.0 supports RTA varargs, so this never
gates; it is recorded so a re-audit does not have to re-derive it:

- `device/kernels/dataflow/reader_s2s_tensor_concat.cpp:23-28` — 4 args per input, loop bounded by CTA
  `num_input_tensors`. Vararg (a) — CTA-bounded still counts.
- `device/kernels/dataflow/reader_writer_block_sharded_concat.cpp:33-42` — 9 args per transfer, loop
  bounded by the runtime `num_transfers`. Vararg (a). `num_transfers` itself (`:31`) is read once
  before the loop as a distinct field → **name it**, do not let it ride the varargs.
- `device/kernels/dataflow/writer_s2i_width.cpp:27-28` — 1 arg per input, loop bounded by CTA
  `num_tensors`. Vararg (a). The five leading scalars (`:16-20`) are distinct fields → name them.
- `device/kernels/dataflow/reader_concat_interleaved_start_id.cpp:39-43` and
  `reader_concat_stick_layout_interleaved_start_id.cpp:38-42` — a `get_arg_addr` raw pointer walk over
  two `N`-length RTA blocks, indexed by the loop variable. Vararg (a). The three leading scalars
  (`:16-18`) are distinct fields → name them.

### Relaxation candidates

None. The op has no custom hash to mine, and the sheet lists `TensorParameter relaxation = none` on all
six rows.

### TTNN factory analysis

Sheet-derived facts, each confirmed against the code:

| Fact | Value | Evidence |
|---|---|---|
| Current concept | `descriptor` (all 6 factories) | `create_descriptor` returning `ProgramDescriptor` on each factory (paths in *Gate detail*) |
| Op-owned tensors | none | no `WorkloadDescriptor`, no `buffers` vector; `create_output_tensors` allocates only the single output (`device/concat_device_operation.cpp:182-186`) |
| MeshWorkload need | none — genuinely single-program | sheet `Execution Model = SPMD`; no mesh-workload return anywhere |
| Pybind `create_descriptor` | absent | `concat_nanobind.cpp` — no `nb::class_` of the device-op, no factory binding |
| Other risky pybind | none | `Is safe to port? = yes` on all rows, no `warning` |
| Custom hash | absent | no `compute_program_hash`, no `attribute_values` / `to_hash` backdoor |
| `get_dynamic_runtime_args` | absent | not on `ConcatDeviceOperation` (`device/concat_device_operation.hpp:37-49`) |
| `override_runtime_arguments` | absent | — |
| **Target concept** | **`ProgramSpecFactoryConcept`**, no op-owned tensors | matches the sheet's `Porting Target` column on all six rows |

The four gate conjuncts (custom hash, pybind `create_descriptor`, `get_dynamic_runtime_args`,
`override_runtime_arguments`) are all `no`, and the concept is `descriptor` — which is why the TTNN gate
clears despite the op-level RED. Mixed-concept variants are supported, so porting only the clean subset
does not require the other four factories to move (`ttnn/api/ttnn/operation_concepts.hpp:174-189`,
`ttnn/api/ttnn/device_operation.hpp:227-243`).

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

1. **`ConcatS2IProgramFactory` binds a kernel source that does not exist.**
   `device/concat_s2i_program_factory.cpp:54-55` sets
   `reader_desc.kernel_source = ".../kernels/dataflow/reader_s2i_width.cpp"`, but that file is absent
   from the working tree and from a freshly-fetched `origin/main` (`git ls-tree -r origin/main` lists
   only `writer_s2i_width.cpp` under that directory). Git history shows the file existed and was
   removed around `a51949d7b2a` ("#6418: optimize height sharded width concat") while the factory kept
   its reference. Any dispatch reaching this factory would fail at program creation. It appears to be
   unreachable in practice, which is presumably why nothing has caught it:
   `select_program_factory` picks it only for sharded inputs with a non-sharded output
   (`device/concat_device_operation.cpp:32-34`), and `concat_impl` never produces that combination —
   the sharded-input path always hands `ttnn::prim::concat` a *sharded* output config
   (`device/concat_device_operation.cpp:364`, `:390`) or unshards and recurses (`:401-406`) — and a
   search across `ttnn/` and `tests/` finds no other caller of `ttnn::prim::concat`. Recommend the ops
   team either restore the reader or delete the factory (and its sheet row).

2. **Sharded-batching limit disagrees with the factory's own cap.**
   `calculate_max_tensors_per_concat` returns `max(2, (256 - 4) / 4 * 0.9)` = **56** for
   height/width-sharded inputs (`device/concat_device_operation.cpp:270-280`), so `concat_impl` only
   splits into batches above 56 inputs (`:317`). But `ConcatS2SMultiProgramFactory` hard-asserts
   `num_input_tensors <= 16` (`device/concat_s2s_multi_program_factory.cpp:43`), because CB index 16 is
   the output. A sharded→sharded concat on dim 2 or 3 with 17..56 inputs therefore reaches a `TT_FATAL`
   instead of being batched. The block-sharded path avoids this because both sides agree on 16
   (`device/concat_device_operation.cpp:261-266` and `:318-323`).

3. **Dead compile-time arg in the row-major interleaved writer.**
   `device/concat_program_factory.cpp:214` puts `dst_buffer->page_size()` at CTA index 1 of
   `writer_unary_stick_layout_interleaved_start_id.cpp`, but that kernel reads CTA 0 and then starts its
   accessor args at index 2 (`:17-18`) — it takes the stick size from **runtime** arg 1 instead
   (`:13`, fed at `device/concat_program_factory.cpp:286`). The CTA is never read; it only shifts the
   accessor-args offset. The tiled branch passes one CTA and matches `TensorAccessorArgs<1>()`, so the
   two branches are consistent with the kernels — the RM entry is simply redundant.

4. **Unused DFB handle in the tiled compute kernel.**
   `device/kernels/compute/height_sharded_width_concat_two_tensors.cpp:57` constructs
   `DataflowBuffer output_dfb(output_dfb_id)` and never uses it. Harmless today; it becomes a port
   decision (see *Heads-ups*).

5. **`const` where `constexpr` was intended in the row-major sharded reader.**
   `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors.cpp:21-28` reads seven
   compile-time args into `const uint32_t` locals while its neighbours use `constexpr`; `groups`
   (`:28`) is then used to initialize `constexpr` values at `:32-35`. That works because a `const`
   object of integral type initialized with a constant expression is itself usable in constant
   expressions, and `get_compile_time_arg_val` expands to a `constexpr` call — so the initializer
   qualifies. Cosmetic, but the inconsistency invites a reader to assume those seven values are runtime,
   and a later edit that gives one of them a genuinely runtime initializer would break the `constexpr`
   lines below with an error pointing at the wrong place.

6. **Implicit two-input assumption is unguarded inside the subset factories.**
   `ConcatS2SRMProgramFactory` and `ConcatS2STiledProgramFactory` index `input_tensors[1]` and
   `num_tiles_for_each_input_shard[1]` unconditionally
   (`device/concat_s2s_rm_program_factory.cpp:93`, `device/concat_s2s_tiled_program_factory.cpp:30-34,139`)
   while also looping their CB creation over `num_input_tensors`. Correct today because
   `select_program_factory` only picks them when `input_tensors.size() == 2`
   (`device/concat_device_operation.cpp:46`), but neither factory asserts it. Worth a `TT_FATAL` — and
   worth knowing when reading the port brief, since the exactly-two property is what makes these two
   factories portable.

## Per-DeviceOperation attribution

Only one DeviceOperation shares this directory, so no bundling is needed. Findings do differ **per
factory**, and are attributed inline throughout. Summary:

| Factory | Verdict | Blocker |
|---|---|---|
| `ConcatProgramFactory` | **RED** | CTA varargs (Appendix A) → wait-for-feature |
| `ConcatS2IProgramFactory` | **RED** ×2 | kernel accesses DFBs by runtime index; **and** its reader kernel file does not exist → ops team |
| `ConcatS2SMultiProgramFactory` | **RED** | kernel accesses DFBs by runtime index |
| `ConcatBlockShardedProgramFactory` | **RED** | kernel accesses DFBs by runtime index |
| `ConcatS2SRMProgramFactory` | **GREEN** | — |
| `ConcatS2STiledProgramFactory` | **GREEN** | — |

## Questions for the user

1. **Is the subset port wanted at all?** The clean subset is two of six factories, both restricted to
   the exactly-2-input sharded→sharded dim-3 path (`device/concat_device_operation.cpp:46-59`). They
   are on the sheet's `llama` list, so they are not a backwater — but they leave concat's default and
   interleaved path (`ConcatProgramFactory`) on the legacy API indefinitely, and the op will carry a
   mixed-concept `program_factory_t` until the variable-count feature lands. A brief is issued per the
   recipe's config-scoped-GATE rule; whether to spend the port is your call, not the audit's.

2. **Should `ConcatS2IProgramFactory` be deleted rather than fixed?** It has been referencing a
   deleted kernel for some time and looks unreachable through `concat_impl` (*Misc anomalies* item 1).
   If the ops team deletes it, the op drops to five factories and the readiness sheet needs its row
   removed — worth raising with the sheet owner at the same time.

3. **Does the DFB-by-runtime-index blocker already have a home — specifically, is it inside #45388?**
   The compile-time-arg half of concat's problem is tracked: `advanced_options.hpp:61-66` names **#45388**
   for the unimplemented compile-time-vararg feature. The DFB half has no Appendix A entry and I found no
   equivalent placeholder in the API (*Recipe notes* item 1), so I could not tell whether #45388's scope
   covers "a variadic number of tensor arguments" end-to-end — its own comment lists exactly that as a
   motivating use case (`advanced_options.hpp:52-54`), which suggests it might — or whether reaching a
   DFB by index needs its own work item. Worth confirming with whoever owns #45388, because the two are
   distinct as that issue is scoped — a compile-time vararg mechanism lets a kernel read `N` *values* by
   index, which is what `ConcatProgramFactory` needs, and says nothing on its face about reaching a *DFB*
   by index — yet plausibly close in implementation, since the DFB ids are **already** emitted as
   implicit CTAs, one named constant per binding (`tt_metal/jit_build/genfiles.cpp:158`, `:194-198`), and
   the missing step is emitting them as an indexable collection instead. My reading is that (2) may fall
   out of (1) or need a small follow-on rather than a separate feature, but that is a judgement from the
   headers, not a scoping claim. If #45388 covers both, all four blocked factories unblock together; if
   not, the DFB side needs filing or three factories stay blocked after it lands.

## Recipe notes

1. **Appendix A has no entry for a kernel reaching a *resource* by a runtime-computed index, only for
   variable-count compile-time args.** The audit tells me the Appendix A list is authoritative and that
   "a construct not listed there is supported ... don't gate on it." Applied literally, that would have
   made me pass `ConcatS2SMultiProgramFactory`, `ConcatBlockShardedProgramFactory`, and
   `ConcatS2IProgramFactory` as portable, when their kernels construct a DFB from a runtime-computed
   index (`DataflowBuffer input_dfb(input_id)`, `DataflowBuffer src_dfb(src_dfb_id)`). In Metal 2.0 each
   DFB a kernel uses must be referenced by a predefined name, so there is nothing for that construct to
   become. I gated it and routed it here rather than passing it. Two suggestions: (a) add an Appendix A
   entry — proposed name *"Resource access by runtime index (DFB/tensor/semaphore)"*, with the
   recognition signal "a `DataflowBuffer` / `TensorAccessor` / `Semaphore` constructed from a value that
   is not a constant expression (a loop counter, or a value unpacked from a runtime arg)"; (b) if the
   intent is that CTA varargs is *meant* to cover this whole family, say so in that entry's Status
   field, because its current recognition signals are strictly about compile-time *arguments* and
   mention no resource bindings at all. Note item 5 below proposes a separate correction to those same
   signals, for a different reason — the two edits touch the same bullet and are best made together.

   **Name the entry for the runtime index, not for the varying count.** Worth stating because I got this
   wrong first: the count is *not* the blocker on the host side — `DataflowBufferSpec::unique_id` is a
   `std::string` and `KernelSpec::dfb_bindings` is a `Group`, so a factory can already build `N` specs and
   `N` bindings at run time. An entry named "variable-count bindings" would therefore describe something
   that already works, and would mislead an auditor into checking the wrong side of the API. The wall is
   only that the kernel source must name each accessor literally and cannot select among them at runtime.

2. **"RED ⇒ no brief" and "config-scoped GATE ⇒ brief for the clean subset" contradict each other, and
   this op sits exactly on the seam.** The GATE role says "A *config-scoped* GATE — e.g.
   GlobalCircularBuffer confined to one factory — still issues a brief for the clean subset," while
   *Output: the two documents* says `METAL2_PORT_BRIEF.md` is "emitted only on a fully GREEN audit
   (every gate cleared) ... Never on RED," and the brief's own template opens with "Audit cleared all
   gates." I followed the GATE-role rule and issued a brief scoped to the two clean factories, with a
   header saying so — but the brief template has no wording for a partial scope, so I had to invent it.
   Suggest the template carry an explicit subset variant of its opening block and its "Gates cleared"
   line.

3. **`ttnn/cpp/ttnn/kernel/` (singular) is classified for function-call escape but not for file-path
   borrowing.** The audit's inventory phase makes it donor class 3 ("a second shared-kernel pool. Treat
   as shared-lib class"), but `shared/port_patterns.md`'s *Excluded from "shared kernel"* clause names
   only `ttnn/cpp/ttnn/kernel_lib/` and `tt_metal/hw/inc/api/...`, and its rung procedure is scoped to
   "kernels owned by *ops*, under `ttnn/cpp/ttnn/operations/`." So for
   `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` — bound by
   `ConcatProgramFactory` and three other ops — I could not tell whether a porter forks it (rung 2) or
   treats it as out of scope like kernel-lib. I reported it as a fork-eligible borrow with the
   ambiguity flagged. Worth one line in either doc.

4. **The CTA-varargs entry's own "Examples in the wild" names this op, which made the finding easy —
   but the entry's op-level signal really is insufficient here, in an instructive way.** Concat's
   `std::vector<Tensor>` fires the op-level cue for all six factories, yet only one of them actually
   trips the kernel-level decider; two are outright clean; three fail for a *different* reason (note 1).
   The entry's guidance to go read the kernel is exactly right, and the outcome here is a good
   ground-truth example of why: a single op can hold all three answers at once. Might be worth citing
   concat that way rather than as a flat example of the feature being in use.

5. **The CTA-varargs entry's kernel-level recognition signal describes a construct that cannot
   compile, and therefore misses the construct that actually blocks the port.** The entry says the
   decider is "`get_compile_time_arg_val(i)` inside a loop where `i` depends on a count value." That
   cannot exist: the macro expands to `get_ct_arg<arg_idx>()`
   (`tt_metal/hw/inc/api/compile_time_args.h:64`), so the index is a template parameter (`:25-29`) and
   a runtime `i` is a compile error. An auditor grepping for the literal macro-with-runtime-index shape
   finds nothing and passes the op.

   The shape that *does* compile, and that concat actually uses, is a **runtime subscript of the
   `kernel_compile_time_args` array object** — `constexpr std::array<uint32_t, N>` at
   `tt_metal/hw/inc/api/compile_time_args.h:23`. `constexpr` on the object constrains its
   initialization, not its subscripting, so `kernel_compile_time_args[<runtime expr>]` is legal and the
   result flows into a non-`constexpr` variable
   (`device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp:70`). This is the form
   to key on. Suggested replacement for the recognition bullet:

   > The kernel subscripts the compile-time-arg array with a runtime-varying index —
   > `kernel_compile_time_args[<expr>]` where `<expr>` is not a constant expression (typically a loop
   > counter or a value unpacked from a runtime arg), with the result stored in a non-`constexpr`
   > variable. Note this cannot appear as `get_compile_time_arg_val(i)`: that macro takes its index as a
   > template parameter, so a runtime index there is a compile error. A kernel needing a
   > runtime-selected compile-time value must reach past the macro to the array, and that reach is the
   > signal. (Also fires: a kernel template instantiated over a count that came from a compile-time
   > arg, e.g. `make_tensor_accessor_args_tuple<num_tensors, ...>()`.)

   Worth grepping the tree for `kernel_compile_time_args[` when the entry is updated — there are hits
   outside concat (e.g. `tt_metal/programming_examples/pad_multi_core/kernels/`,
   `tt_metal/hw/inc/experimental/udm/accessor/mesh_tensor_accessor.h`), and some may be constant-index
   uses that the new wording should not over-fire on.

6. **The CTA-varargs entry's Status field could cite its tracking issue, and its one-line rationale
   points at the wrong side of the API.** Status currently reads "Metal 2.0's `compile_time_args` schema
   requires fixed-shape declaration at factory-construction time; there is no kernel-side equivalent of
   the legacy positional-CTA loop yet. A CTA-vararg feature is on the host API roadmap." The second
   clause is exactly right. The first is not: the host-side container is already variable-length —
   `using CompileTimeArgs = Table<std::string, uint32_t>`
   (`tt_metal/api/tt-metalium/experimental/metal2_host_api/kernel_spec.hpp:192`) — so a factory *can*
   declare `N` compile-time args today. The whole constraint is the kernel-read side, for the
   `byte_offset`-versus-`value` reason spelled out in this report's *Expected resolution* block.

   This matters beyond wording: an auditor who believes the limit is host-side may go looking for a
   fixed-size array in `KernelSpec`, not find one, and conclude the entry is stale — which the appendix's
   own maintenance rule tells them not to act on, but which puts them in the awkward position of
   reporting a gate they think has lifted. Pointing at the kernel side removes the ambiguity, because
   the placeholder is right there in the API and unambiguous.

   Two concrete suggestions: (a) reword the first clause to locate the gap kernel-side; (b) add the
   tracking issue — `advanced_options.hpp:61-66` names **#45388**. An issue number turns "on the
   roadmap" into something a reader of a RED audit can actually go and check the status of, which is
   the difference between a blocker they can schedule around and one they can only wait on.

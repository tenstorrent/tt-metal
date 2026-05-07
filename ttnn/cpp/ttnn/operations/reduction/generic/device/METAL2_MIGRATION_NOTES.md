## Metal 2.0 Migration Guide — Shortcomings Discovered During the Reduction Factory Ports

Targets the version of `metal2_migration_guide.md` on `origin/akertesz/metal2-documentation`
(commit-resolved at port time). Originally scoped to the multi-core W reduction port; the H,
HW (single-core), and Welford (W/H/HW) factory ports landed in the same series and surfaced
new shortcomings — added as sections 15–20 below. The reference port for comparison is
`origin/bbradel-41067_sum_quasar`, which targeted the *predecessor* DFB API
(`tt_metal::experimental::dfb::*`), not Metal 2.0 — it remains useful for
contrasting cache-hit override patterns and for surfacing which guide gaps fall
out of moving from a "free-function + KernelHandle" world to a
"ProgramSpec + named bindings" world.

The list below is structured "what bit me, and why the guide didn't help".

---

### 1. `TensorAccessorArgs` was incompatible with `ProgramSpec` (resolved by Metal 2.0 TensorBindings)

The legacy reader/writer use the modern `TensorAccessor` device-side accessor,
which is configured by appending positional CTAs via
`TensorAccessorArgs(*buffer).append_to(reader_compile_time_args)`. Metal 2.0
`KernelSpec::compile_time_arg_bindings` is **named-only**; positional CTAs are
hard-coded to empty inside `MakeProgramFromSpec`. The named CTAs themselves are
emitted as `experimental::CtaVal<uint32_t>` constants in
`kernel_args_generated.h`, never reaching the legacy positional CTA slot table
that `TensorAccessorArgs<CTA_OFFSET>` reads via `get_compile_time_arg_val(...)`.

Concrete consequences (pre-resolution):
- The canonical
  `constexpr auto args = TensorAccessorArgs<CTA_OFFSET>(); auto a = TensorAccessor(args, addr);`
  pattern doesn't work in Metal 2.0 kernels.
- An interim local helper
  (`tt_metal/hw/inc/api/tensor/interleaved_tensor_accessor_helpers.h`) was
  introduced for the non-sharded interleaved case: the host declared `is_dram`
  + `aligned_page_size` as named CTAs and the kernel constructed
  `TensorAccessor<InterleavedDSpec<IsDram>>` directly. It worked, but it was a
  per-port shim and didn't extend to sharded buffers.

**Resolution** — `akertesz/tensor-binding-support` adds first-class TensorParameter
bindings to Metal 2.0:

- `ProgramSpec::tensor_parameters` declares each tensor a program operates on
  (`{unique_id, TensorSpec}`).
- `KernelSpec::tensor_bindings` binds a tensor parameter to a per-kernel
  accessor name (`{tensor_parameter_name, accessor_name}`).
- `ProgramRunParams::tensor_args` supplies the actual MeshTensors at execution
  time (`{tensor_parameter_name, std::cref(mesh_tensor)}`).
- The kernel constructs the accessor via `TensorAccessor accessor(ta::name)`;
  the codegen-emitted token in `kernel_bindings_generated.h` carries the right
  CTA layout, and the buffer base address comes from the binding's CRTA slot
  populated automatically by `SetProgramRunParameters` from the TensorArg.

The reduction factory ports use this directly. The interim helper was deleted.
`is_dram` and `aligned_page_size` are no longer named CTAs in the kernel — they
fall out of the bound TensorSpec.

Sharded support: the binding mechanism is sharded-capable in principle (the
TensorAccessorArgs bridge handles all distribution kinds), but the ttnn factory
ports still `TT_FATAL` on sharded inputs because the sharded reader / writer
plumbing wasn't part of this migration. Re-enabling sharded reductions under
Metal 2.0 just requires lifting that fail-fast and supplying the matching
sharded reader/writer; no further framework work is required.

The guide should:
- Document the TensorParameter / TensorBinding / TensorArg trio as the
  canonical way to expose tensor accessors in Metal 2.0.
- Show the kernel-side `TensorAccessor accessor(ta::name)` pattern explicitly.
- Mention that this replaces the pre-binding interim of constructing
  `TensorAccessor` from named-CTA `is_dram` + `aligned_page_size` (the
  reduction factories' first-pass workaround, since deleted).

### 2. Self-referential DFB bindings (producer == consumer)

Negate-mode reduce keeps an accumulator and a negated-input scratch buffer
that the compute kernel both produces *and* consumes. The validation in
`program_spec.cpp` asserts "exactly one producer" and "exactly one consumer"
per DFB, which is fine — but it also forbids a kernel having two `DFBBinding`s
with the same `local_accessor_name`. The result is that you must declare the
*same* DFB twice on the same kernel under two different accessor names
(e.g. `acc_w` for PRODUCER, `acc_r` for CONSUMER):

```cpp
BindDFB(compute, ACC_DFB, "acc_w", DFBEndpointType::PRODUCER);
BindDFB(compute, ACC_DFB, "acc_r", DFBEndpointType::CONSUMER);
```

…and on the device side the kernel constructs *two* `DataflowBuffer` objects
that resolve to the same underlying CB id on Gen1:

```cpp
experimental::DataflowBuffer cb_acc_writer(dfb::acc_w);
experimental::DataflowBuffer cb_acc_reader(dfb::acc_r);
```

The migration guide silently presents producer and consumer as always being
distinct kernels. The "Local DFB invariant" troubleshooting bullet reinforces
that impression. Recommended additions:
- An "intra-kernel DFB" example (or at least a sentence) showing the
  PRODUCER/CONSUMER binding can resolve to the same kernel.
- Confirmation that on Gen1 the two `DFBAccessor` ids alias the same CB
  (today's de facto behavior, but undocumented).

### 3. Per-WorkUnit compile-time arguments are not expressible

`split_work_to_cores` produces two core groups with different per-core work
counts. The legacy factory creates *two distinct compute KernelDescriptors*,
one per group, each with its own per-group CTA. Metal 2.0 has *one* `KernelSpec`
per kernel, so there is no host-side mechanism to give group 1 a CTA value
of `X1` and group 2 a CTA value of `X2`.

`WorkUnitSpec` solves placement, but the guide's "kernel in multiple
WorkUnitSpecs" example does not address per-WU CTA values. The only Metal 2.0
escape hatch today is to demote per-group CTAs to per-node RTAs, at the cost
of losing compile-time loop unrolling on the demoted dimension.

This pattern recurred across every reduction factory port:

| Factory | Work split axis | Per-group CTA → RTA |
|---|---|---|
| W (multi-core)    | along `H` rows           | `Ht` → runtime |
| H (multi-core)    | along `W` columns        | `Wt` → runtime |
| HW (single-core)  | none                     | n/a (single core, single CTA set) |
| Welford W         | along `NC * Ht`          | `NCHt` → runtime |
| Welford H         | along `NC * Wt`          | `NCWt` → runtime |
| Welford HW        | along `NC / batch_size`  | `NC_per_core` → runtime |

The general rule fell out the same way each time: the dimension whose value
varies between core groups is the one demoted to RTA; everything else stays
compile-time.

A subtle consequence: when two factories share a compute kernel source
(`reduce.cpp` is shared by W and HW non-negate paths) and *disagree* on which
CTA was demoted, they can't actually share the source. The W factory promotes
`Ht` to runtime, but H promotes `Wt` to runtime — and the kernel signature
can't satisfy both. We added a separate `reduce_h.cpp` for H non-negate. See
section 20.

The guide should:
- Add a short section on "split_work_to_cores → Metal 2.0".
- Describe the demote-to-RTA pattern (or document a future feature for
  per-WU CTA bindings).

### 4. Kernel-library helpers still take raw CB ids

The host-side guide promotes a clean DFB-by-name story (`dfb::input`,
`DataflowBuffer cb(dfb::input)`). But many existing kernel-library helpers
still take raw `uint32_t` CB ids:

- `compute_kernel_hw_startup(cb_in, cb_scaler, cb_out)`
- `compute_kernel_lib::reduce<...>(cb_in, cb_scaler, cb_out, ...)`
- `dataflow_kernel_lib::prepare_reduce_scaler<cb_id, ...>(scaler_f)`
- LLK helpers (`reduce_init`, `reduce_tile`, `pack_tile`, `copy_tile`, …)

The Gen1-only workaround used in this port is to forward `dfb::name.id` (a
`uint16_t` widened to `uint32_t`) into those helpers:

```cpp
constexpr uint32_t cb_input = dfb::input.id;
compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);
```

This relies on the Gen1 invariant that `DFBAccessor::id == underlying CB id`.
The guide doesn't say this is supported, and the device-side header comment
on `DFBAccessor::id` is a "currently backed by a compile-time ID" note that
explicitly anticipates the field's semantics changing.

The guide should:
- Either commit to "`DFBAccessor::id` is the CB id on Gen1, forever" or
  provide a migration story for the kernel libraries that still want a CB id.

### 5. No first-class TTNN integration story

`ttnn::device_operation` exposes two factory concepts:
`ProgramFactoryConcept` (`create` + `override_runtime_arguments`, returning a
`CachedProgram<shared_variables_t>`), and
`ProgramDescriptorFactoryConcept` (`create_descriptor` returning
`tt::tt_metal::ProgramDescriptor`). There is **no** adapter for a factory
that returns a `ProgramSpec` (or a `Program` built from one).

This port chose to wrap `MakeProgramFromSpec` inside a `ProgramFactoryConcept`
factory — i.e. the create/override pattern owns a `tt::tt_metal::Program`
returned by `MakeProgramFromSpec`, and the override path calls
`SetProgramRunParameters` instead of `SetRuntimeArgs`/`GetRuntimeArgs`. That
works (the `MeshWorkloadFactoryAdapter` happily wraps any
`ProgramFactoryConcept` for mesh dispatch), but it wasn't obvious from the
guide; the natural reading is that `MakeProgramFromSpec` belongs in
`create_descriptor`-style code.

The guide should:
- Add a TTNN integration section: how to wire a `ProgramSpec`-based factory
  into `ttnn::device_operation::ProgramFactoryConcept`.
- Document the recommended `shared_variables_t` shape: cache the work-split
  snapshot (cores, groups, per-group sizes) so `override_runtime_arguments`
  can rebuild a `ProgramRunParams` cheaply on each cache hit.
- Show the `SetProgramRunParameters` re-call pattern explicitly. The
  reference Quasar branch uses `GetRuntimeArgs(...)[core.x][core.y][0] = addr`
  on cache hit; the Metal 2.0 equivalent (rebuild + apply RunParams) should
  be the canonical example.

### 6. Cache-hit semantics of `SetProgramRunParameters` are under-specified

The "Note" on `program.hpp::SetProgramRunParameters` says the function is
intended for one-shot setup; the in-place power-user
`ProgramRunParamsView` API is the documented "high-performance inner loop"
choice — and is "not yet implemented". In practice, TTNN ops *must* re-set
RTAs on every cache hit because tensor buffer addresses change between calls.

The guide should answer:
- Is calling `SetProgramRunParameters` repeatedly on a cached `Program`
  supported and performant, or is it expected to be replaced by the view API?
- If the view API is the future, what's the migration path for ops written
  today against `SetProgramRunParameters`?
- What completeness rules apply on the *second* `SetProgramRunParameters`
  call? The header says "every (kernel, node) pair that requires runtime
  arguments". This port re-emits *all* RTAs every call to be safe; whether a
  partial update is supported is unclear.

### 7. Schema-vs-spec mismatch errors are footgun-prone

Today a missing or extra named RTA produces an error inside
`SetProgramRunParameters`. With named CTAs/RTAs, kernel code references
`args::name` at compile time via `kernel_args_generated.h`. If the host adds a
new named RTA but the kernel doesn't reference it (or vice versa) the error
isn't surfaced until enqueue.

The guide should:
- State explicitly that `kernel_args_generated.h` is regenerated from
  `KernelSpec` on each build, and that the host is the source of truth.
- Add a Troubleshooting bullet for "I added a named RTA but
  `SetProgramRunParameters` complains" / "kernel code references
  `args::foo` but it doesn't exist".

### 8. `Defines` API mismatch with the legacy world

`KernelDescriptor::defines` is a `std::vector<std::pair<std::string,
std::string>>` initialized from a `std::map`. `KernelSpec::CompilerOptions::Defines`
is the *same* vector-of-pair shape — but the typical legacy code constructs a
`std::map` first (because `reduce_op_utils::get_defines(...)` returns a map),
then converts. Worth a short note in the guide that `Defines` is a vector,
plus a one-line helper pattern for converting from the
`std::map<std::string, std::string>` returned by existing utility code.

### 9. The scaler-vs-post_mul story for MAX/MIN is invisible to migrators

This is *not* a Metal 2.0 issue, but the guide is the natural place to remind
op authors of in-tree conventions that affect their kernel argument schema.
For MAX/MIN with non-unity scalar, the high-level `reduce()` already swaps
`scaler→1.0` and stashes the user value in `post_mul_scaler`. A migrating
factory does **not** need to re-apply that swap; the original W factory
forwards `operation_attributes.scaler` straight through. (An earlier draft of
this port doubled the swap, which was harmless only because the value happens
to already be 1.0; it would have been a real bug had the upstream contract
changed.)

The guide could simply add a "domain-specific gotchas" pointer for
op-migrators to check the high-level wrapper before re-doing host-side math.

### 10. Upstream bug: `MakeGen1ComputeConfig` undersizes `unpack_to_dest_mode` (compile-time blocker)

`tt_metal/impl/metal2_host_api/program_spec.cpp::MakeGen1ComputeConfig` builds the
`ComputeConfig::unpack_to_dest_mode` vector sized to `dfb_name_to_id.size()`
(the number of DFBs in the spec). The downstream JIT framework
(`tt_metal/jit_build/data_format.cpp::get_unpack_dst_formats`) then validates
`unpack_to_dest_mode.size() >= buf_formats.size()`, where `buf_formats.size()`
equals `NUM_CIRCULAR_BUFFERS = 64` on host. Any compute kernel built from a
`ProgramSpec` with fewer than 64 DFBs (i.e. all real-world cases on Gen1) hits:

```
Failed to generate binaries for <kernel> TT_FATAL @ ... data_format.cpp:172:
  unpack_to_dest_mode.size() >= buf_formats.size()
info: unpack_to_dest_mode vector must have 64 elements
```

This first surfaced when the migrated reduce_w compute kernel went to JIT-compile
with 5 DFBs (negate path). There is no user-side workaround:
`ComputeConfiguration::unpack_to_dest_mode` is a `std::vector<std::pair<DFBSpecName, UnpackToDestMode>>`,
and the size of the lowered vector is forced to `dfb_name_to_id.size()` regardless
of what the user supplies.

This port patches `MakeGen1ComputeConfig` to size to
`max(NUM_CIRCULAR_BUFFERS, dfb_name_to_id.size())`, matching the
"host-side allocations sized for maximum CB count across all architectures"
contract documented in `data_format.cpp`.

**Confirmed identical bug in `MakeQuasarComputeConfig`.** When the migrated
W reduction was first run on a Quasar emulator
(`unit_tests_ttnn --gtest_filter=SumTensorLastDimTests/SumTensorLastDimFixture.SumTensorCorrectly/*`)
it hit the same TT_FATAL at the same `data_format.cpp:172` site, with the same
"vector must have 64 elements" message. The Quasar variant has been patched in
the same way (mirrored fix and comment in `program_spec.cpp::MakeQuasarComputeConfig`).

### 11. Upstream bug: TRISC prolog for Metal 2.0 compute kernels does not pre-include `api/compute/common.h`

`tt_metal/jit_build/genfiles.cpp::build_trisc_prolog` injects this prolog
ahead of every TRISC `chlkc_*.cpp` translation unit for Metal 2.0 kernels:

```cpp
#define TRISC_UNPACK
#include "kernel_bindings_generated.h"
#include "kernel_args_generated.h"
#include "defines_generated.h"
// ...then the user kernel.cpp
```

The auto-generated `kernel_args_generated.h` unconditionally emits

```cpp
FORCE_INLINE uint32_t get_vararg(uint32_t idx) { return get_arg_val<uint32_t>(N + idx); }
FORCE_INLINE uint32_t get_common_vararg(uint32_t idx) { return get_common_arg_val<uint32_t>(M + idx); }
```

and `experimental/kernel_args.h` defines
`experimental::get_arg(RtaArg<T>)` in terms of `get_arg_addr(...)`. All three
referenced names — `get_arg_val<T>`, `get_common_arg_val<T>`, and
`get_arg_addr` — are non-dependent inside their definitions, so they must be
in scope at parse time. They live in `api/compute/common.h` for compute
kernels and are only included transitively via the user's compute headers,
which come *after* `kernel_args_generated.h`. Result:

```
kernel_args_generated.h: error: 'get_arg_val' was not declared in this scope
kernel_args_generated.h: error: 'get_common_arg_val' was not declared in this scope
experimental/kernel_args.h: error: 'get_arg_addr' was not declared in this scope,
                                   and no declarations were found by argument-dependent lookup
note: 'uint32_t get_arg_addr(int)' declared here, later in the translation unit
```

DM kernels avoid this only by accident: `brisck.cc` / `ncrisck.cc` already
pre-include `api/dataflow/dataflow_api.h` *before* `kernel_includes.hpp`, so
the equivalent dataflow declarations are already visible by the time the
metal2 generated headers are processed. There is no analogous pre-include on
the TRISC entry point (`trisck.cc -> chlkc_list.h -> chlkc_*.cpp`), and
`chlkc_*.cpp` is generated by `genfiles.cpp` itself.

This port patches `build_trisc_prolog` to inject
`#include "api/compute/common.h"` ahead of `kernel_bindings_generated.h` /
`kernel_args_generated.h` for Metal 2.0 TRISC builds.

The same root cause would also bite if anyone tried to build a Metal 2.0
ERISC kernel: `erisck.cc` does pre-include `api/dataflow/dataflow_api.h`
on Wormhole, but the prolog is shared and there's no conditional emission
based on processor class. The fix should either:
  - generate `get_vararg` / `get_common_vararg` only when the kernel actually
    declares varargs (which is the lowest-risk fix),
  - or formalize "the TRISC prolog must pre-include `api/compute/common.h`
    for compute Metal 2.0 kernels" in the generator.

### 12. Helper-library buffer-type abstraction (and what's still arch-specific)

The first port of the reduce kernels to Metal 2.0 left two kinds of code in the kernels:

1. **Compute helper** (`compute_kernel_lib::reduce`) and **dataflow helper**
   (`dataflow_kernel_lib::prepare_reduce_scaler`) took raw `uint32_t` CB ids and
   internally constructed `experimental::CircularBuffer` to do reserve/push/wait/pop.
2. **Kernel sources** extracted `dfb::*.id` to feed the helpers, and forwarded the
   same id to the LLK calls (`reduce_init`, `reduce_tile`, `pack_tile`, ...).

That worked on Gen1 only because the Gen1 invariant "DFB id == underlying CB id" makes
the CB-typed wrapper indirectly drive the right hardware. On Quasar the same code would
miss DFB hardware semantics.

We refactored to remove the Gen1-only assumption from the helpers themselves:

- New `kernel_lib::BufferRef<T>` adapter (`ttnn/cpp/ttnn/kernel_lib/buffer_helpers.hpp`)
  normalizes any of `uint32_t` / `experimental::CircularBuffer&` /
  `experimental::DataflowBuffer&` to a uniform `id() / wait_front / pop_front /
  reserve_back / push_back` interface. Specializations dispatch at compile time.
- `compute_kernel_lib::reduce()` is now templated on three buffer types and uses
  `BufferRef` internally. Kernel code passes `experimental::DataflowBuffer` objects
  directly (no more `dfb::*.id` extraction); legacy callers that pass `uint32_t`
  continue to work via the `BufferRef<uint32_t>` specialization.
- `dataflow_kernel_lib::prepare_reduce_scaler` keeps its constexpr `cb_id` template
  parameter (used for compile-time format / tile-dim queries), but its runtime sync
  side now goes through `experimental::DataflowBuffer` instead of
  `experimental::CircularBuffer`. The DataflowBuffer wrapper is arch-agnostic: same
  Gen1 behavior, plus correct Gen2 DFB driving.

What this **does** unblock for Quasar:

- The compute kernels (`reduce.cpp`, `reduce_w_neg.cpp`, `reduce_h.cpp`,
  `reduce_h_neg.cpp`, `reduce_hw_neg.cpp`, plus the welford trio) compile
  and execute the reduce flow without any `#ifdef ARCH_QUASAR` branches.
- The host factory now sets both `gen1_data_movement_config` and
  `gen2_data_movement_config` on the reader/writer KernelSpecs so the same
  ProgramSpec lowers on both archs.
- The reader/writer data path goes through `TensorAccessor` (constructed from
  Metal 2.0 TensorBindings) and `experimental::Noc::async_read/async_write`,
  both of which have arch-agnostic noc_traits specializations
  (`noc_traits_t<TensorAccessor<DSpec>>` in `experimental/tensor.h`,
  `noc_traits_t<DataflowBuffer>` in `experimental/dataflow_buffer.h`). No
  `InterleavedAddrGenFast` references remain in the reduction kernels.

What this **does not** yet unblock:

- Sharded reductions: the framework's TensorBinding mechanism handles sharded
  TensorAccessors fine, but the ttnn factory ports here `TT_FATAL` on sharded
  inputs because the sharded reader / writer kernels haven't been migrated.
  Lifting the fail-fast plus porting the sharded reader/writer is a separate
  task; no further framework work is needed.
- ~~`MakeQuasarComputeConfig` in `tt_metal/impl/metal2_host_api/program_spec.cpp`
  still sizes `unpack_modes` to `dfb_name_to_id.size()` (mirror of the Gen1 bug
  patched as #10).~~ **Patched** when first hit on a Quasar emulator run; see #10
  for details.

The guide should:
- Have a "buffer-type abstraction" section showing the `BufferRef`-style pattern
  (or whatever the framework owners settle on) so future helpers don't replicate
  the CB-only assumption.
- State explicitly that helpers must not directly construct
  `experimental::CircularBuffer` if they want Gen2 coverage, and that
  `experimental::DataflowBuffer` is the portable choice.

### 13. `kernel_lib` helpers depend on Gen1-only `DataFormat` enumerators and Gen1-only `_with_dt` LLKs

Two unrelated arch-coupled assumptions are baked into the merged kernel helper
library (`ttnn/cpp/ttnn/kernel_lib/`):

1. **`cb_helpers_compute.inl::get_full_tile_size_impl`** is a `constexpr`
   switch on `DataFormat` enumerators. Several of those enumerators
   (`Lf8`, `UInt32`, `Bfp8`, `Bfp8_b`, `Bfp4`, `Bfp4_b`, `Bfp2`, `Bfp2_b`)
   only exist on Gen1 (`tt-1xx/.../tensix_types.h`). Quasar
   (`tt-2xx/quasar/tensix_types.h`) declares a different set —
   `Tf32`, `Fp8R`, `Fp8P`, `MxFp8R/P`, `MxFp6R/P`, `MxFp4`, `MxInt8/4/2`,
   `Int16/4`, etc. Compilation of the helper for Quasar fails with eight
   `'X' is not a member of 'DataFormat'` errors.

2. **`reduce_helpers_compute.inl`** defines four matmul-related helpers
   (`reduce_with_matmul_init`, `reduce_with_matmul_init_with_dt`,
   `reduce_matmul_tiles`, `reduce_init_short_with_dt`) whose bodies call into a
   Gen1-shaped LLK API:
   - `llk_math_matmul_init<MathFidelity, MM_THROTTLE>(in0_cb, in1_cb, transpose)`
     — Quasar's overload is `llk_math_matmul_init<MathFidelity>(ct_dim, rt_dim)`,
     a different parameter shape with no MM_THROTTLE and no CB-id args.
   - `llk_math_matmul<MathFidelity, MM_THROTTLE>(idst)` — does not exist on
     Quasar (closest name is `llk_math_matmul_tile`).
   - `llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(...)` and the math-
     side counterpart used in the `_with_dt` accumulator-reload path — neither
     exists on Quasar.
   Even though `_with_dt` and the matmul wrappers are only reachable from the
   accumulation-reload + REDUCE_ROW SUM/AVG paths (which our
   `NoAccumulation` test never exercises), GCC's `-Wtemplate-body` checks
   non-dependent names eagerly, and the non-template helpers are parsed
   unconditionally. Compilation fails on Quasar with
   `'llk_math_matmul_init' candidate expects 2 arguments, 3 provided`,
   `'llk_math_matmul' was not declared in this scope`, and
   `'llk_unpack_reconfig_data_format_srca' was not declared in this scope`.

Pragmatic fix applied here:
- `#ifndef ARCH_QUASAR` guards around the offending DataFormat case labels and
  the matmul-helper / `_with_dt`-helper bodies.
- `reduce_uses_matmul<>()` (in `reduce_helpers_common.hpp`) returns `false` on
  Quasar, so the matmul-based reduce specialization is disabled there and the
  helpers' Quasar-stubbed bodies are never reached at runtime. Both reader
  (scaler tile fill) and compute consult the same predicate, so they stay in
  sync — the reader fills row-0 (reduce-LLK layout) on Quasar, matching the
  compute path.
- Quasar paths in the `_with_dt` and matmul wrappers fall back to
  `ASSERT(false)` so accidental future use surfaces clearly rather than
  silently misbehaving.
- The Quasar tile-size formulas for `Tf32 / Fp8R/P / MxFp* / MxInt*` are not
  added — none of the W-reduction tests exercise them — but should be added
  under `#ifdef ARCH_QUASAR` when needed.

Functional consequence on Quasar: SUM/AVG along W goes through the regular
`reduce_tile` LLK rather than `matmul_tiles`. Correct, but loses the Gen1 perf
optimization. Re-enabling matmul-based reduce on Quasar requires bridging the
Quasar matmul LLK shape and is tracked separately.

The user's stated preference was templating over `#ifdef`, but this case is
fundamentally non-portable: the enumerators *literally do not exist* on the
other arch, so no template / SFINAE wrapper can hide the unresolved name.
The idiomatic option is `#ifdef`-gating, exactly as
`tt_metal/hw/inc/experimental/circular_buffer.h` already does for its
`reserve_back` / `push_back` LLK selection.

The guide should:
- Inventory which `kernel_lib` helpers call out to LLKs whose names differ by
  arch, and which rely on `DataFormat` enumerators that aren't shared.
- State whether helper authors are expected to keep Gen2-equivalent
  implementations behind `#ifdef ARCH_QUASAR` (and what those equivalents are).

### 14. `program_id` / `unique_id` collision rules are not stated

`ProgramSpec::program_id`, `WorkUnitSpec::unique_id`, `KernelSpec::unique_id`,
`DFBSpecName`, `SemaphoreSpecName` are all strings. Validation rejects
duplicate `unique_id`s within a spec, but the guide doesn't say:

- Whether `program_id` must be unique across a `MeshWorkload` (the comment in
  `program_spec.hpp` implies yes).
- What characters are legal (the validator rejects non-C++ identifiers for
  `local_accessor_name` but appears to accept arbitrary strings for the
  `unique_id` fields — which then can't be used as identifiers if you wanted
  to).
- Whether host-side `constexpr const char*` constants must match string
  contents or just identifier pattern.

A short "naming and uniqueness" subsection would help.

### 15. Unity-build collisions in anonymous namespaces (cross-cpp)

The reduction CMake target uses Unity builds: multiple `.cpp` files are
concatenated into a single `unity_*.cxx` translation unit. C++ rules: each
translation unit has *one* unnamed namespace, so all `namespace { … }` blocks
inside that TU merge into the same scope. As soon as two factory cpps each
declare an anonymous-namespace symbol with the same name, the unity TU rejects
the duplicate.

This bit me three times in the same series:

1. **DFB id strings** (`INPUT_DFB`, `SCALER_DFB`, …) and helper functions
   (`MakeDFB`, `BindDFB`, `DefinesFromMap`) were duplicated verbatim across the
   W and HW factories — same name, same definition, different cpp files.
   Compile error: "redefinition of 'MakeDFB'".

2. **Kernel-spec / work-unit constants** (`READER_KERNEL`, `WRITER_KERNEL`,
   `COMPUTE_KERNEL`, `WORK_UNIT`) — same name across factories but *different*
   string values. Same redefinition error, with the added hazard that whichever
   `.cpp` was concatenated first won the symbol and the others silently used the
   wrong string at runtime if the redefinition error were suppressed.

3. **Per-factory structs and helper functions** (`WorkDistribution`,
   `ComputeWorkDistribution`) — different field names per factory (W uses
   `num_rows_per_core_group_*`, H uses `num_cols_per_core_group_*`), so the
   first definition won and the *second* factory's accesses were rejected as
   "no member named 'num_rows_per_core_group_1' in 'WorkDistribution'".

Patterns that worked:

- Hoist truly identical declarations into a shared header
  (`reduce_metal2_factory_helpers.hpp`) with `inline` (functions) /
  `inline constexpr` (constants) linkage. This is where DFB id strings,
  `MakeDFB`, `BindDFB`, and `DefinesFromMap` now live.
- Prefix per-factory anonymous-namespace constants with the factory name
  (`W_READER_KERNEL`, `H_READER_KERNEL`, `HW_READER_KERNEL`,
  `WELFORD_READER_KERNEL`).
- Rename per-factory structs / helper functions with the same prefix
  (`WWorkDistribution`, `HWorkDistribution`).
- Function overloads with a unique first-parameter type are fine: the
  factories' `BuildRunParams(const ReduceMultiCoreWSharedVariables&, …)` /
  `BuildRunParams(const ReduceMultiCoreHSharedVariables&, …)` /
  `BuildRunParams(const ReduceSingleCoreHwSharedVariables&, …)` /
  `BuildRunParams(const WelfordReduceSharedVariables&, …)` coexist as overloads
  in the merged anon namespace.

The guide should:
- Add a "Unity build hygiene" troubleshooting bullet describing this class of
  failure and the prefix / shared-header patterns.
- Note that `metal2_migration_guide.md`-style code that uses generic anonymous-
  namespace constant names will silently break the second time it's adopted
  inside a unity-build target.

### 16. DFB capacity must be a multiple of 2 on Quasar (implicit-sync DFB scheduling)

Quasar's implicit-sync DFB scheduling allocates 2 transaction IDs per side and
asserts `capacity % num_txn_ids == 0`. So *every* DFB needs `num_entries ≥ 2`
and even. On Gen1 a 1-tile DFB happens to work for "produce-once, consume-once"
buffers (e.g., the scaler tile) but the same spec lowered to Quasar fails the
runtime assertion.

Pragmatic patterns used by the reduction factories:

- The simple cases (`INPUT`, `SCALER`, `OUTPUT`, single-tile scratch) just
  hardcode `kNumInputEntries = 2`, etc. Cheap on L1.
- The non-trivial case is the H factory's negate path. The compute kernel
  pushes `ntiles` tiles per inner-loop iteration where `ntiles ∈ {chunk_size,
  Wt_per_core % chunk_size, Wt_per_core when Wt_per_core < chunk_size}`. The
  CB / DFB capacity has to be a common multiple of every push size that
  actually occurs across all core groups, *and* a multiple of 2.
  `ComputeNegateScratchEntries` in
  `reduce_op_multi_core_h_program_factory.cpp` walks both core groups, takes
  the LCM of the realized push sizes, then rounds up to a multiple of 2 if
  needed. The same logic appeared in the original Gen1 factory (modulo the
  Quasar even-multiple constraint), so this isn't pure metal2 cost — but the
  even-multiple round-up is.

The guide should:
- Document the even-multiple-of-2 capacity rule, attribute it to Quasar
  implicit-sync, and recommend `kNum*Entries = 2` as the safe default.
- For variable-push-size scratch DFBs, describe the LCM-of-push-sizes pattern.

### 17. Always-bind-then-gate-with-`if constexpr` for optional DFBs

When a DFB is only used on some kernel paths (negate-only ACC/INEG; W-Welford
`cb_scaled` only when `do_scale=true`; HW-Welford `partial`/`combined` only
present in the HW variant), the host has a binary choice:

a. Bind the DFB unconditionally on the host. The kernel uses
   `if constexpr (cond) { … cb.wait_front(…); … }` to gate runtime use; the
   wrapper construction (`experimental::DataflowBuffer cb_x_obj(dfb::x);`)
   sits in the outer scope.
b. Bind the DFB only when the path is taken. The kernel needs an `#ifdef`
   to skip the wrapper declaration, because `dfb::x` simply isn't generated
   when the host doesn't bind it.

Option (a) — what the reduction factories chose — costs ~1 tile of L1 per
core when the optional path isn't taken (`cb_scaled` for non-`do_scale`
W-Welford). Option (b) is more invasive on the kernel side and bifurcates the
source.

A subtlety with option (a): `if constexpr (false) { … cb_x_obj.wait_front(…) … }`
discards the entire branch at parse time, so even though the wrapper exists,
the compiler never resolves names inside the discarded block — so it doesn't
matter whether the LLK calls inside reference the unused buffer. The wrapper
*declaration* itself, though, is parsed and *does* require `dfb::x` to exist.
That's the binding constraint.

The guide should:
- State that "the wrapper construction is what requires the host-side
  binding to exist", so authors know option (a) and option (b) are real
  alternatives with different cost.
- If the framework grows truly-optional bindings (`OptionalDFBBinding` or
  similar), point to that instead.

### 18. Recent API change: `MakeProgramFromSpec` now requires a `MeshDevice&`

Mid-port, `tt_metal/api/tt-metalium/experimental/metal2_host_api/program.hpp`
changed signatures from

```cpp
Program MakeProgramFromSpec(const ProgramSpec& spec, bool skip_validation = false);
```

to

```cpp
Program MakeProgramFromSpec(
    const distributed::MeshDevice& mesh_device,
    const ProgramSpec& spec,
    bool skip_validation = false);
```

Any in-flight metal2 ports with the old signature get a compile error
("too few arguments to function call, expected at least 2"). Standard fix:
pass `*tensor_args.device()` from the factory's `create()` — TTNN's
`Tensor::device()` returns `distributed::MeshDevice*`, so

```cpp
Program program = m2::MakeProgramFromSpec(*a.device(), spec);
```

is the universal patch.

The guide should:
- Surface this signature on the very first `MakeProgramFromSpec` example
  (the legacy 1-arg signature is dead).
- Add a Troubleshooting bullet: "too few arguments to function call,
  expected at least 2, have 1".

### 19. Multi-variant factories: one `create()`, multiple ProgramSpecs

The W-port doc treats `create()` as building a single ProgramSpec. The Welford
factory builds three: one per `reduce_dim` (W, H, HW). Each variant has its
own kernels (sequential reader vs column-partitioned reader; generic writer
vs Welford-specific writer; one of three compute kernels), its own DFB set
(W needs `var` + `scaled`; HW needs `partial` + `combined`), and its own RTA
schema (`NCHt` for W, `NCWt` for H, `NC_per_core` for HW).

Patterns that worked:

- Branch on the variant inside `create()` to choose kernel paths and CTA
  bindings. There's no factory-class indirection — the variant is a
  configuration, not a class hierarchy.
- Capture the variant in `shared_variables_t` (the Welford
  `WelfordReduceSharedVariables` carries `reduce_dim`) so
  `override_runtime_arguments` can rebuild the right per-core RTA shape on
  cache hit. Without this, the override path would have to re-derive the
  variant from `operation_attributes`, which is also fine but adds a layer.
- Per-variant DFB ids declared as factory-local constants, not in the shared
  helpers header — they're only meaningful to one factory.
- Per-variant kernel-side DFB declarations: the kernel for variant X expects
  `dfb::partial` to exist; the host for variant X is the only path that binds
  it. As long as the host's variant selection matches the kernel source's
  hardcoded DFB references, this works without `#ifdef`s in the kernel.

The guide should:
- Add a short "multi-variant factory" example (Welford is a clean reference).
- Describe how `shared_variables_t` interacts with variant selection: the
  variant is captured at miss-time and reused at cache-hit time, so
  `override_runtime_arguments` doesn't need to re-derive it.

### 20. Shared compute kernels disagree when factories disagree on which CTA gets demoted

This is a corollary of section 3. `reduce.cpp` is the unified non-negate
compute kernel for the reduction primitive; the Gen1 version took
`(Ht, Wt, NC)` as compile-time args. Metal 2.0 forces a choice for any
multi-core factory: which of those becomes the per-node RTA?

- The W factory promotes `Ht` to RTA (work split varies `Ht` per core).
- The H factory promotes `Wt` to RTA (work split varies `Wt` per core).
- The HW factory has a single core, so all three stay compile-time, but it
  *uses* the W-style kernel and just supplies `Ht` as a constant runtime arg.

W and HW can share `reduce.cpp` (`Ht` runtime, `Wt`/`NC` compile-time, HW
passes the constant `Ht` runtime). H cannot reuse the same source — it would
need `Wt` runtime and `Ht` compile-time, which the W-style signature can't
express. The H factory therefore uses a dedicated `reduce_h.cpp` with the
swapped runtime/compile-time classification.

The general rule: if two factories share a kernel source, they must agree on
which CTAs are demoted to RTAs. When they don't agree, accept the second
kernel source — it's cheaper than threading both axes as runtime in a single
"super-kernel", which loses compile-time loop unrolling on whichever axis a
given factory could otherwise hold constant.

The guide should:
- Note this as the natural consequence of demote-to-RTA: when factories with
  different work-split axes share a non-negate kernel, separate sources are
  the right answer.

---

### Summary table

| # | Shortcoming | Severity |
|---|-------------|---------|
| 1 | ~~`TensorAccessorArgs` incompatible with `ProgramSpec` (positional CTAs)~~ — **resolved** by the Metal 2.0 TensorBinding mechanism (`akertesz/tensor-binding-support`) | Was high; now n/a for non-sharded |
| 2 | Self-referential DFB bindings undocumented | Medium — easy to discover but not documented |
| 3 | No way to express per-WorkUnit CTAs (split_work_to_cores) | Medium — forces RTA demotion |
| 4 | Kernel-library helpers still take raw CB ids | Medium — relies on Gen1 invariant |
| 5 | No documented TTNN `ProgramFactoryConcept` integration | High — every TTNN port hits this |
| 6 | Cache-hit `SetProgramRunParameters` semantics under-specified | Medium |
| 7 | Schema-vs-kernel mismatches errors only at enqueue time | Low — quality-of-life |
| 8 | `Defines` API expects vector, legacy code returns map | Low — one-line helper |
| 9 | Domain-specific scaler/post_mul convention easy to misapply | Low — op-specific |
| 10 | **Upstream bug**: `MakeGen1ComputeConfig` undersizes `unpack_to_dest_mode` | **Critical — blocks all compute kernels** |
| 11 | **Upstream bug**: TRISC prolog doesn't pre-include `api/compute/common.h` for Metal 2.0 compute kernels | **Critical — blocks all compute kernels** |
| 12 | Helper libraries hard-coded to Gen1 CBs; needed buffer-type abstraction for Quasar | **High — blocks Quasar port of any helper-using kernel** |
| 13 | `kernel_lib` helpers reference Gen1-only `DataFormat` enumerators and Gen1-only `_with_dt` LLKs in non-template / eagerly-checked template bodies | **High — blocks compilation of any `compute_kernel_lib::reduce`-using kernel on Quasar** |
| 14 | `unique_id` / `program_id` naming/uniqueness rules unstated | Low |
| 15 | Unity-build collisions in anonymous namespaces (cross-cpp helpers, constants, structs) | Medium — hits any second factory in the same target |
| 16 | DFB capacity must be a multiple of 2 on Quasar (implicit-sync) | Medium — silent runtime assertion |
| 17 | Optional / conditional DFBs require always-bind + `if constexpr` gating | Low — pattern, not a defect |
| 18 | `MakeProgramFromSpec` API change: now requires `MeshDevice&` (#43589) | Medium — breaks every in-flight port |
| 19 | Multi-variant factories (one `create()`, multiple ProgramSpecs) undocumented | Low — pattern not in guide |
| 20 | Shared compute kernels disagree when factories disagree on which CTA gets demoted | Low — corollary of #3 |

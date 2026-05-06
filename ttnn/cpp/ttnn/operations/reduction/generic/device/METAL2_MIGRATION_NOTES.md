## Metal 2.0 Migration Guide — Shortcomings Discovered During the Reduction W Factory Port

Targets the version of `metal2_migration_guide.md` on `origin/akertesz/metal2-documentation`
(commit-resolved at port time). The reference port for comparison is
`origin/bbradel-41067_sum_quasar`, which targeted the *predecessor* DFB API
(`tt_metal::experimental::dfb::*`), not Metal 2.0 — it remains useful for
contrasting cache-hit override patterns and for surfacing which guide gaps fall
out of moving from a "free-function + KernelHandle" world to a
"ProgramSpec + named bindings" world.

The list below is structured "what bit me, and why the guide didn't help".

---

### 1. `TensorAccessorArgs` is silently incompatible with `ProgramSpec`

The legacy reader/writer use the modern `TensorAccessor` device-side accessor,
which is configured by appending positional CTAs via
`TensorAccessorArgs(*buffer).append_to(reader_compile_time_args)`. Metal 2.0
`KernelSpec::compile_time_arg_bindings` is **named-only**; positional CTAs are
hard-coded to empty inside `MakeProgramFromSpec`.

Concrete consequences:
- Sharded tensor support cannot be ported as-is. The Metal 2.0 reader/writer
  for `MULTI_CORE_W` was forced to fall back to `InterleavedAddrGenFast` and
  the host factory now `TT_FATAL`s on non-interleaved memory configs.
- Any kernel that uses `TensorAccessor` (the recommended accessor going
  forward) currently has to be downgraded to `InterleavedAddrGenFast` for the
  Metal 2.0 path.

The guide should:
- Call this out explicitly in [Troubleshooting](#troubleshooting) and in the
  CTA section of the kernel-args migration.
- Either provide a shim (named CTA bindings emitted by
  `TensorAccessorArgs::append_named_to(...)`) or an officially recommended
  workaround.

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
one per group, each with its own `Ht` CTA. Metal 2.0 has *one* `KernelSpec`
per kernel, so there is no host-side mechanism to give group 1 a CTA value
of `Ht1` and group 2 a CTA value of `Ht2`.

`WorkUnitSpec` solves placement, but the guide's "kernel in multiple
WorkUnitSpecs" example does not address per-WU CTA values. The only Metal 2.0
escape hatch today is to demote per-group CTAs to per-node RTAs (which is what
this port does — `Ht` is now a named RTA), at the cost of losing compile-time
loop unrolling.

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

- The compute kernels (`reduce_metal2.cpp`, `reduce_w_neg_metal2.cpp`) compile
  and execute the reduce flow without any `#ifdef ARCH_QUASAR` branches.
- The host factory now sets both `gen1_data_movement_config` and
  `gen2_data_movement_config` on the reader/writer KernelSpecs so the same
  ProgramSpec lowers on both archs.

What this **does not** yet unblock for Quasar:

- The reader/writer **data path** still uses Gen1-only primitives:
  `InterleavedAddrGenFast<...>` and `noc_async_read_tile / noc_async_write_tile`.
  Porting requires either (a) the Metal 2.0 framework supporting positional CTAs
  so `TensorAccessor` works (then `experimental::Noc::async_read(tensor_accessor,
  dfb, ...)` lights up via the existing `noc_traits_t<DataflowBuffer>` specialization),
  or (b) an upstream `noc_traits_t<InterleavedAddrGenFast<...>>` specialization so
  the arch-agnostic `Noc` API can drive the older interleaved address gen. Neither
  is in place today; tracked under shortcoming #1.
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

---

### Summary table

| # | Shortcoming | Severity |
|---|-------------|---------|
| 1 | `TensorAccessorArgs` incompatible with `ProgramSpec` (positional CTAs) | High — blocks sharded migration |
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

# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/normalization/batch_norm`

> Audit cleared all gates for **both** factories in this directory. This is your actionable input; the
> full record is in `METAL2_PREPORT_AUDIT.md`.
>
> **Scope: both factories, one PR** (invoker's explicit scoping).
> - `BatchNormOperation::BatchNormFactory` + its 2 dataflow and 2 compute kernels
> - `RunningStatistics::RunningStatisticsProgramFactory` + its 2 dataflow and 2 compute kernels
>
> The two device-operations share **no** factories and **no** kernels, so neither constrains the
> other's conversion. Port them **one at a time, in the order below**, completing each before starting
> the next: each is a self-contained sub-port, so if the pass has to stop after the first, that is a
> complete deliverable with the second cleanly enumerated for a fresh instance.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ ·
TensorAccessor 3rd arg ✓ *(all five, for both factories)*

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

## Recommended order and sizing

| Order | Unit | Files to convert |
|---|---|---|
| **1** | `RunningStatisticsProgramFactory` | `running_statistics_program_factory.cpp`, `reader_running_statistics.cpp`, `writer_running_statistics.cpp`, `running_statistics_kernel.cpp`, `running_statistics_sfpu_kernel.cpp` (+ header retarget) |
| **2** | `BatchNormFactory` | `batch_norm_program_factory.cpp`, `reader_batch_norm.cpp`, `writer_batch_norm.cpp`, `batch_norm_kernel.cpp`, `batch_norm_sfpu_kernel.cpp` (+ header retarget) |

`RunningStatistics` first because it is the larger and more intricate of the two (12–14 DFBs vs 9–10,
three self-loops vs two, and the in-place input writes) — doing it while context is freshest is the
better trade. Nothing forces this order; reverse it if you prefer to warm up on the smaller factory.

Each factory's conversion is **atomic**: the factory and *both* of its runtime-selected compute
sources flip together, so expect no green build within a unit until its last file is converted. Ten
files total across the PR, single-axis selection, no shared kernels — this is well inside one primary
session's budget.

---

# Common to both factories

## TTNN factory analysis

- **Current concept:** `descriptor` — `create_descriptor` returning a `tt::tt_metal::ProgramDescriptor`
  (`batch_norm_device_operation.hpp:39`, `running_statistics_device_operation.hpp:36`). Each
  `program_factory_t` has a single alternative (`:45`, `:42`).
- **Op Classification / Execution Model:** `PD Op (pointer-patching)` / `SPMD`.
- **Op-owned tensors:** none (either factory).
- **Target concept:** `ProgramSpecFactoryConcept` — `create_program_artifacts` returning
  `ttnn::device_operation::ProgramArtifacts`; leave `op_owned_tensors` defaulted. Matches the sheet's
  `Porting Target` column.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom
  `compute_program_hash` · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind
  descriptor (`nb::class_` of the device op) · smuggled pointer · other migration-risky pybind. All
  `no` on both factories.

### Device-op-class edits — exactly one per factory

In each device-op **header**, retarget that factory's declaration and swap the include:

```cpp
-#include <tt-metalium/program_descriptors.hpp>
+#include "ttnn/metal_v2_artifacts.hpp"
...
-    static tt::tt_metal::ProgramDescriptor create_descriptor(
+    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
```

Nothing else in either device-op class changes. **No pybind removal** — `batch_norm_nanobind.cpp`
exposes only the user-facing `ttnn::batch_norm`, no factory entry point.

> **Do NOT delete `BatchNormOperation::operation_attributes_t::to_hash()`**
> (declared `batch_norm_device_operation.hpp:22`, defined `batch_norm_device_operation.cpp:121`).
> The sheet lists it under `Backdoor custom hash` and `Formerly custom hashed? = yes (to_hash, still
> present)`, so it will look like debt to clear — it is not yours. The sanctioned custom-hash deletion
> is scoped to a `compute_program_hash` override; the `Custom hash (compute_program_hash)` cell is
> `no` and this op has none. `to_hash()` is the **ttsl attribute-hash protocol**, customising how
> `operation_attributes_t` hashes *within* the framework's default reflection hash rather than
> replacing the cache key — tensor args (hence `TensorSpec`) are still folded in, so the
> `UpdateTensorArgs`-on-cache-hit hazard does not apply. Deleting it would change the cache key.
> `RunningStatistics` has no such member at all. Leave both exactly as they are, and note in the
> report that you were told to.

## Tensor bindings — 11 total, all Case 1

Every address feeds a `TensorAccessor`, so all eleven express as `TensorParameter` / `TensorBinding`
and the kernel builds `TensorAccessor(tensor::<name>)`. **No `get_bank_base_address` bridge is needed
anywhere in this op** — no kernel does raw address arithmetic.

Legacy delivers all of them via the **`Buffer*` binding form** (the factory pushes the `Buffer*`
object, not `->address()`), which the framework auto-registers and patches on cache hits — so this is
*not* the silent-wrong stale-address hazard, just routine port work. Every host-side
`TensorAccessorArgs(...).append_to(...)` call and every kernel-side `TensorAccessorArgs<N>()` /
`next_compile_time_args_offset()` chain disappears.

**TensorParameter relaxation: none** (both factories, per the sheet). No `ArgConfig::Runtime*` in this
op or its donors — keep strict `TensorSpec` matching; do not set `dynamic_tensor_shape` or
`match_padded_shape_only`.

**TensorAccessor 3rd arg: none** — all 11 sites are already 2-arg. Nothing to drop.

## Optional tensors — the DFB and the tensor answers are OPPOSITE

This is the highest-risk item in the whole PR and it recurs in **both** factories: `weight`/`bias` in
`BatchNormFactory`, `running_mean`/`running_var` in `RunningStatisticsProgramFactory`. Each flag gates
both a DFB and a tensor, and the correct treatment differs.

**DFBs → bind UNCONDITIONALLY.** In both factories the legacy host allocates the corresponding CBs on
*every* config, outside any conditional (`batch_norm_program_factory.cpp:260-279`;
`running_statistics_program_factory.cpp:222-241,262-281`), and the kernels reference those buffers in
compiled code regardless of the flag — the writers construct their `DataflowBuffer` objects outside
the `if constexpr` (`writer_batch_norm.cpp:49-50,64,68`; `writer_running_statistics.cpp:46-49,57,60`),
the `batch_norm` compute kernels gate on a plain **runtime** `if` (`batch_norm_kernel.cpp:65,93`) so
both branches compile, and `running_statistics_sfpu_kernel.cpp:72-75` constructs all of them
unconditionally. Therefore:

- Declare the `DataflowBufferSpec`s unconditionally and bind PRODUCER on the writer / CONSUMER on
  compute unconditionally. Census is a plain **1P+1C**.
- **Do not** treat an absent optional as a dead CB and drop it. That would (a) shrink the op's L1
  footprint relative to legacy — a functional change the port may not make — and (b) fail to compile,
  since `dfb::<name>` must exist for the unconditional `DataflowBuffer` construction.
- **Do not** reach for the conditional-DFB `#ifdef` pattern here. The usual caution ("don't bind
  unconditionally to dodge `if constexpr`") targets ops whose legacy host allocated the CB
  *conditionally*; these do not, so unconditional binding **is** the faithful translation.

**TensorParameters → bind CONDITIONALLY, with `#ifdef` gating.** There is no tensor to supply as a
`TensorArgument` when the optional is absent, so the parameter cannot be declared — the
mandatory-`#ifdef` case from *Pattern: Conditional / optional DFB bindings*, which applies verbatim to
`tensor::`. Legacy fakes it today by building accessor args from a null `Buffer*`
(`batch_norm_program_factory.cpp:322,324`; `running_statistics_program_factory.cpp:366,368`) and
constructing an accessor over address `0` unconditionally (`writer_batch_norm.cpp:65,69`;
`writer_running_statistics.cpp:58,61`) while gating only the *uses*. The port removes that idiom:

- Host: declare the `TensorParameter` and its `TensorBinding` only when the optional has a value; emit
  a matching define via `KernelSpec::compiler_options.defines` (e.g. `WEIGHT_HAS_VALUE`,
  `BIAS_HAS_VALUE`, `RUNNING_MEAN_HAS_VALUE`, `RUNNING_VAR_HAS_VALUE`).
- Kernel: `#ifdef`-gate the `TensorAccessor(tensor::<name>)` construction **and** every expression
  referencing it — moving the construction *inside* the gate, which is a change from today's
  unconditional construction. The existing `if constexpr (<flag>)` blocks in the writers become
  `#ifdef` blocks (a CTA gate promoted to a preprocessor gate).
- The `*_has_value` CTAs still feed the **compute** kernels' conditionals, so keep them as named CTAs
  there.

## Kernel code tracks CB ids in runtime `uint32_t` variables — leave it

`batch_norm_kernel.cpp:31-32` and `batch_norm_sfpu_kernel.cpp:42-43` compute `dfb_affine_or_out` /
`dfb_scaled_output` with a **runtime** ternary over the `*_has_value` flags and then construct
`DataflowBuffer` from the result; the SFPU kernels additionally thread a mutable `last_srca_dfb` cb id
through function returns and parameters (`batch_norm_sfpu_kernel.cpp:37,61-62,228`;
`running_statistics_sfpu_kernel.cpp:17,83`), and `batchnorm_bcast_tiles(...)` passes all thirteen CB
ids as plain `uint32_t` parameters.

All of this keeps working: `DFBAccessor::operator uint32_t()` is `constexpr`, and `DataflowBuffer` has
a `uint16_t` constructor for the runtime-selected case. **Do not** try to eliminate every `uint32_t`
cb id in favour of `dfb::` tokens — the runtime-selected ones cannot be, and rewriting a helper's
signature or the `last_srca_dfb` threading is kernel-logic surgery, out of scope.

## Named arguments — name everything, no varargs

Every RTA in all eight kernels is read once at a source-literal index, as a distinct field. **No
varargs anywhere in this op.** All CTAs become named — except that **CB-index CTAs become DFB
bindings, never named args**.

**Heads-up — both factories emit more RTAs than their kernels read** (audit anomaly A1): the trailing
`cHt`/`cWt` are dead in all four dataflow kernels, and `freq`/`counter` are dead in *both*
RunningStatistics compute kernels (they read only `num_tiles`). A named schema makes the surplus
explicit, so dropping the dead names is the natural reading of the port — but note it in the report,
and keep the idle-core zero-fill counts consistent with whatever schema you declare.

## Hardware config and `opt_level` — same shape in both factories

**DM kernels** — both factories use bare `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` with
no custom fields and no `noc_mode` override, i.e. the conventional defaults:

```cpp
.hw_config = ttnn::create_reader_datamovement_config(device->arch()),   // reader
.hw_config = ttnn::create_writer_datamovement_config(device->arch()),   // writer
```

**Compute kernels — Style A** (the op resolves a TTNN `ComputeKernelConfig`): both factories call
`get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config)`
(`batch_norm_program_factory.cpp:349`, `running_statistics_program_factory.cpp:391`) over a config
resolved by `batch_norm::utils::resolve_compute_kernel_config`. Translate with
`ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config)`, then
set the two Metal-only fields by hand on the returned Gen1 alternative:

- `bfp_pack_precision_mode` — legacy leaves `bfp8_pack_precise` at its default in both ⇒ **do nothing**
  (defaults coincide).
- **`unpack_modes`** — needed in both, per-factory lists below.

**`opt_level`** — `grep -n opt_level` over both factories returns **nothing**: the field is absent from
every `KernelDescriptor`. Absent on a `ComputeConfigDescriptor` still resolves to **`O3`**, so **each**
compute `KernelSpec` needs it stated explicitly:

```cpp
.compiler_options = {.defines = compute_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
```

The four DM kernels need nothing — their legacy `O2` is Metal 2.0's default.

## Work split and placement — one WorkUnitSpec each, no multiplicity

Both factories call
`split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major)`
(`batch_norm_program_factory.cpp:57`, `running_statistics_program_factory.cpp:57`), **but all three
kernels are placed on `all_device_cores`** and the per-group counts are delivered purely as **runtime
args** — no per-group CTAs, no second `KernelDescriptor` per core group.

So each factory gets **one `WorkUnitSpec`** over `all_device_cores` holding its three kernels, and
**one compute `KernelSpec`**. No preserved multiplicity; the demoting-per-group-CTA anti-pattern is not
in play (nothing was ever a per-group CTA).

Cores outside both work groups get **zero-filled** RTAs today
(`batch_norm_program_factory.cpp:73-79`, `running_statistics_program_factory.cpp:72-78`), and the
compute kernels early-return on `num_tiles == 0`. Metal 2.0 requires every named RTA to be set on every
node the kernel runs on, so **keep that zero-fill** — it is exactly what satisfies the requirement. Use
`AddRuntimeArgsForNode` inside the existing per-core loop rather than inverting it to name-first.

## Unity-build hygiene — matters more because both factories land in one PR

Both factory `.cpp` files compile into the same translation unit and both already wrap their helpers in
`namespace { namespace CMAKE_UNIQUE_NAMESPACE { … } }`. They will want overlapping spec-name constants
(`INPUT`, `OUTPUT`, `READER`, `WRITER`, …). Declare every `DFBSpecName` / `KernelSpecName` /
`TensorParamName` **function-local inside `create_program_artifacts`**, as the `moreh_mean` reference
port did — do **not** add anonymous-namespace constants, and do not hoist a shared header for them.

---

# Unit 1 — `RunningStatistics::RunningStatisticsProgramFactory`

## Tensor bindings (5, all Case 1)

| Binding | Declared from | Bound on | Legacy delivery |
|---|---|---|---|
| `batch_mean` | `tensor_args.batch_mean` | reader | `Buffer*`, reader RTA slot 1 |
| `batch_var` | `tensor_args.batch_var` | writer | `Buffer*`, writer RTA slot 0 |
| `running_mean` †‡ | `tensor_args.running_mean` | writer | `Buffer*`, writer RTA slot 1 — literal `0u` when absent |
| `running_var` †‡ | `tensor_args.running_var` | writer | `Buffer*`, writer RTA slot 2 — literal `0u` when absent |
| `output` | `tensor_return_value` | writer | `Buffer*`, writer RTA slot 3 |

† optional — conditional binding, per the common section above.

‡ **read *and* written in place.** `writer_running_statistics.cpp:87-92` reads the old statistic and
`:103-110` writes the updated one back to the *same* tensor (likewise `:116-121` / `:132-139` for the
variance). One `TensorParameter` with one `TensorBinding` serves both directions — the accessor is used
as both an `async_read` source and an `async_write` destination, which needs no special handling. Do
not be tempted to declare two parameters for one tensor.

Host-side plumbing that disappears: `running_statistics_program_factory.cpp:351,364,365,366,368`;
kernel-side: `reader_running_statistics.cpp:29`, `writer_running_statistics.cpp:36-39`.

## CB → DFB endpoints — 12 DFBs (up to 14 with typecast)

| DFB (legacy CB) | Role | Disposition |
|---|---|---|
| `batch_mean` (`c_0`) | reader → compute | 1P+1C |
| `batch_var` (`c_1`) | **writer** → compute | 1P+1C |
| `out0` (`c_2`) | compute → writer | 1P+1C |
| `old_running_mean` (`c_3`) | writer → compute | 1P+1C, **bind unconditionally** |
| `old_running_var` (`c_4`) | writer → compute | 1P+1C, **bind unconditionally** |
| `momentum` (`c_5`) | reader → compute | 1P+1C |
| `one` (`c_6`) | reader → compute | 1P+1C — produced via `fill_cb_with_value` |
| `updated_m` (`c_7`) | compute → writer *or* compute | **config-flip** |
| `updated_v` (`c_8`) | compute → writer *or* compute | **config-flip** |
| `tmp1` (`c_9`) | compute ↔ compute | **self-loop** |
| `tmp2` (`c_10`) | compute ↔ compute | **self-loop** |
| `tmp3` (`c_11`) | compute ↔ compute | **self-loop** |
| `writer_updated_m` (`c_12`) | compute → writer | 1P+1C — **declare only when** `needs_mean_typecast` |
| `writer_updated_v` (`c_13`) | compute → writer | 1P+1C — **declare only when** `needs_var_typecast` |

The **writer** kernel is a producer on `c_1`, `c_3` and `c_4` (it reads tensor memory into them) as
well as the consumer of `c_2`/`c_7`/`c_8`/`c_12`/`c_13`. Bind by what the kernel body does, not by the
kernel's name.

**Config-flip on `updated_m` / `updated_v`.** Gated by `needs_mean_typecast` / `needs_var_typecast`
(`running_statistics_program_factory.cpp:178-179`):

- **typecast off** — `writer_updated_*_cb == updated_*_cb`: compute packs, writer drains → **1P+1C**.
- **typecast on** — compute packs FP32 into `c_7`/`c_8`, then **compute itself** re-reads it to
  typecast into `c_12`/`c_13` (`running_statistics_sfpu_kernel.cpp:18-41`, via `maybe_typecast_stat`),
  and the writer drains `c_12`/`c_13` → `c_7`/`c_8` become **compute self-loops**.

`one` (`c_6`) is produced by `fill_cb_with_value(dfb_id_one, one_u)`
(`reader_running_statistics.cpp:56`), which does `reserve_back` / fill / `push_back` internally — so
the reader is a genuine locked producer. Port the call to `fill_cb_with_value(dfb::one, one_u)`; the
constexpr `DFBAccessor → uint32_t` conversion handles it. Note the reader never constructs a
`DataflowBuffer` for this CB, and does not need to.

## Runtime-selected compute source

`running_statistics_program_factory.cpp:438` picks with `fmt::format`:

```
running_statistics_{sfpu_kernel | kernel}.cpp   ← (fp32_dest_acc_en || any_float32) ? sfpu : plain
```

Both sources bind the same spec, so **both convert with the factory**. One axis only.

**The two sources read different CTA counts:** `running_statistics_kernel.cpp` reads CTAs 0–13;
`running_statistics_sfpu_kernel.cpp` reads 0–18 (it additionally reads `writer_updated_m`,
`writer_updated_v`, `stat_needs_typecast`, `tc_in_fmt`, `tc_out_fmt`). Declare the **superset** of
named CTAs and let the plain kernel ignore the five it does not read — do **not** prune them, and do
not split into two `KernelSpec`s on that basis.

## `unpack_modes`

Legacy builds a computed `std::vector<UnpackToDestMode>` at
`running_statistics_program_factory.cpp:394-411`: all `Default`, except when `fp32_dest_acc_en` these
**twelve** CBs become `UnpackToDestFp32` —
`batch_mean`(`c_0`), `batch_var`(`c_1`), `out0`(`c_2`), `old_running_mean`(`c_3`),
`old_running_var`(`c_4`), `updated_m`(`c_7`), `updated_v`(`c_8`), `momentum`(`c_5`), `one`(`c_6`),
`tmp1`(`c_9`), `tmp2`(`c_10`), `tmp3`(`c_11`).

Port as a `Table<DFBSpecName, UnpackMode>` keyed by **DFB name**, translating `UnpackToDestFp32` →
`UnpackMode::UnpackToDest` and `Default` → `UnpackMode::UnpackToSrc` (normally expressed by *omitting*
the entry). Reversing the mapping flips the precision/perf tradeoff with no compile or test signal.

Two validator rules interact:
- An explicit entry is **required** for every `Float32` DFB the compute kernel *consumes* when
  `enable_32_bit_dest = true`. With `fp32_dest_acc_en` set, the interm-format DFBs
  (`momentum`, `one`, `tmp1`, `tmp2`, `tmp3`, and `updated_*` on the typecast path) are `Float32`, so
  those entries are forced. Derive each value from the legacy vector; do not guess.
- An entry naming a DFB the kernel does **not** bind is **rejected**. All twelve are bound
  unconditionally here (`old_running_*` per the unconditional-DFB rule), so all twelve entries are
  unconditional — but `c_12`/`c_13` are never in the legacy list and must not be added.

## Named arguments

- reader (`reader_running_statistics.cpp:16-24`) — the `src_addr` slot disappears (TensorBinding),
  leaving **8**: `momentum`, `start_tile_id`, `num_tiles`, `HtWt`, `n_stride`, `c_stride`, `N`, `C`.
- writer (`writer_running_statistics.cpp:14-24`) — the four address slots (0–3) disappear, leaving
  **7**: `start_tile_id`, `num_tiles`, `HtWt`, `n_stride`, `c_stride`, `N`, `C`.
- compute (`running_statistics_kernel.cpp:12`) — **1**: `num_tiles`. The factory emits three
  (`num_tiles_per_core`, `freq`, `counter`) but neither compute source reads slots 1–2 (anomaly A1).

Named CTAs are the non-CB scalars only: `old_running_mean_has_value`, `old_running_var_has_value`,
`fill_momentum_fp32`, `old_stat_is_fp32`, `stat_needs_typecast`, `tc_in_fmt`, `tc_out_fmt`. CTAs
carrying CB indices (reader 0–2, writer 2–7, compute 2–13) all become `dfb::` bindings.

---

# Unit 2 — `BatchNormOperation::BatchNormFactory`

## Tensor bindings (6, all Case 1)

| Binding | Declared from | Bound on | Legacy delivery |
|---|---|---|---|
| `input` | `tensor_args.input` | reader | `Buffer*`, reader RTA slot 1 |
| `batch_mean` | `tensor_args.batch_mean` | writer | `Buffer*`, writer RTA slot 0 |
| `batch_var` | `tensor_args.batch_var` | writer | `Buffer*`, writer RTA slot 1 |
| `weight` † | `tensor_args.weight` | writer | `Buffer*`, writer RTA slot 2 — literal `0u` when absent |
| `bias` † | `tensor_args.bias` | writer | `Buffer*`, writer RTA slot 3 — literal `0u` when absent |
| `output` | `tensor_return_value` | writer | `Buffer*`, writer RTA slot 4 |

† optional — conditional binding, per the common section above.

Host-side plumbing that disappears: `batch_norm_program_factory.cpp:307,319,320,321,322,324`;
kernel-side: `reader_batch_norm.cpp:27`, `writer_batch_norm.cpp:37-41`.

## CB → DFB endpoints — 9 DFBs (10 with typecast)

| DFB (legacy CB) | Role | Disposition |
|---|---|---|
| `input` (`c_0`) | reader → compute | 1P+1C |
| `batch_mean` (`c_1`) | **writer** → compute | 1P+1C |
| `output_0` (`c_2`) | compute → writer *or* compute | **config-flip** |
| `batch_var` (`c_3`) | **writer** → compute | 1P+1C |
| `eps` (`c_4`) | reader → compute | 1P+1C |
| `weight` (`c_5`) | writer → compute | 1P+1C, **bind unconditionally** |
| `bias` (`c_6`) | writer → compute | 1P+1C, **bind unconditionally** |
| `den` (`c_7`) | compute ↔ compute | **self-loop** |
| `temp_1` (`c_8`) | compute ↔ compute | **self-loop** |
| `writer_output` (`c_9`) | compute → writer | 1P+1C — **declare only when** `needs_output_typecast` |

Again the writer is a producer on four DFBs (`batch_mean`, `batch_var`, `weight`, `bias`) as well as
the consumer of `output_0`/`writer_output`.

**Config-flip on `output_0` (`c_2`).** Gated by `needs_output_typecast`
(`batch_norm_program_factory.cpp:181`):

- **typecast off** — `writer_output_cb == output_tensor_cb`: compute packs, writer drains → **1P+1C**.
- **typecast on** — compute packs FP32 into `c_2`, then **compute itself** re-reads `c_2` to typecast
  into `c_9` (`batch_norm_sfpu_kernel.cpp:163-185`), writer drains `c_9` → `c_2` becomes a **compute
  self-loop**.

`temp_1` (`c_8`) is the self-loop that is easiest to mis-slot: it is reached only through the runtime
aliases `dfb_affine_or_out` / `dfb_scaled_output` (`batch_norm_kernel.cpp:31-32`), and when neither
weight nor bias is present those aliases resolve to `output_0` so `temp_1` is untouched at runtime —
but still referenced in compiled code (`dfb_tmp_1_obj` is constructed unconditionally at `:40`). Bind
it as a compute self-loop in every config.

## Runtime-selected compute source

`batch_norm_program_factory.cpp:388` picks with `fmt::format`:

```
batch_norm_{sfpu_kernel | kernel}.cpp   ← (fp32_dest_acc_en || any_float32) ? sfpu : plain
```

**The two sources read different CTA counts:** `batch_norm_kernel.cpp` reads CTAs 0–10;
`batch_norm_sfpu_kernel.cpp` reads 0–14 (adding `writer_output_cb`, `needs_output_typecast`,
`tc_in_fmt`, `tc_out_fmt`). Declare the **superset**; do not prune, do not split.

## `unpack_modes`

Legacy builds the vector at `batch_norm_program_factory.cpp:352-368`: all `Default`, except when
`fp32_dest_acc_en` these **eight** CBs become `UnpackToDestFp32` —
`input`(`c_0`), `batch_mean`(`c_1`), `batch_var`(`c_3`), `eps`(`c_4`), `den`(`c_7`), `weight`(`c_5`),
`temp_1`(`c_8`), `bias`(`c_6`) — **plus** `output_0`(`c_2`) **only when** `needs_output_typecast`
(`:365-367`).

Same two validator rules as Unit 1. The forced entries under `fp32_dest_acc_en` are the interm-format
DFBs (`eps`, `den`, `temp_1`, and `output_0` on the typecast path). **`output_0`'s entry must be gated
on `needs_output_typecast`**, exactly as legacy gates it — an ungated entry is still legal here (the
DFB is always bound) but would flip that DFB's unpack mode in the non-typecast config, a silent
precision/perf change.

## Named arguments

- reader (`reader_batch_norm.cpp:15-23`) — the `src_addr` slot disappears, leaving **8**: `eps`,
  `start_tile_id`, `num_tiles`, `HtWt`, `n_stride`, `c_stride`, `N`, `C`.
- writer (`writer_batch_norm.cpp:14-25`) — the five address slots (0–4) disappear, leaving **7**:
  `start_tile_id`, `num_tiles`, `HtWt`, `n_stride`, `c_stride`, `N`, `C`.
- compute (`batch_norm_kernel.cpp:139-141`) — **3**: `num_tiles`, `tile_freq`, `tile_start` (all three
  are read here, unlike RunningStatistics).

Named CTAs are the non-CB scalars only: `weight_has_value`, `bias_has_value`, `fill_eps_fp32`,
`batch_stat_is_fp32`, `param_is_fp32`, `needs_output_typecast`, `tc_in_fmt`, `tc_out_fmt`. CTAs
carrying CB indices (reader 0–1, writer 2–6, compute 2–11) all become `dfb::` bindings.

---

# Watch for

- **CB endpoints (multi-binding):** none, in either factory. Every CB has at most one producing and one
  consuming kernel per node; the hidden-second-writer hunt came back empty across all eight kernels
  (the op has **no semaphores**, so there is no semaphore-gated co-fill shape). If you find yourself
  reaching for `allow_instance_multi_binding`, recount — the answer here is always 1P+1C or a
  self-loop, and never both on one DFB.
- **Three config-flipping DFBs** (`c_2` in unit 2; `c_7`/`c_8` in unit 1) flip 1P+1C ↔ self-loop with
  the typecast path. Do not fix one config's shape and assume the other follows. The typecast path is
  reachable **only** through the SFPU compute source.
- **Cross-op / shared kernels:** none — no kernel here is borrowed, lent, or shared between the two
  device-ops, and no `_metal2` fork exists or is needed. Three donor *headers* are called into, all
  with crossing-friendly signatures: pass `dfb::name` straight into `fill_cb_with_value` and the
  `dest_format_helpers.hpp` helpers (`uint32_t cb_id` shape), and keep the `dfb.get_write_ptr()` peeks
  that feed `fill_tile_utils.hpp` **as-is** (raw-L1-address shape; whitelist-sanctioned, and the
  recipe directs transfer/peek idioms to stay).
- **RTA varargs:** none — name every runtime arg.
- **Perf comparison is worth extra attention on this op.** The sheet flags
  `Pointer patching perf issue? = suspect perf regression (+ fixed latent bug)`, which attaches to the
  exact `Buffer*`-patching mechanism this port removes. Gathering before/after numbers is *beyond* the
  recipe's no-behavior-change verification, so treat it as optional unless the invoker asks — but if
  you notice anything perf-relevant while testing, it is unusually worth recording in the report.
- **Out of scope, do not touch:** `to_hash()` (see the boxed note); the device-op classes beyond the
  one entry-point retarget each; the duplicated `extract_shape_dims` / `populate_runtime_arguments`
  between the two factories (audit anomaly A4 — the single-PR scoping puts both copies in one diff,
  which will be tempting; leave them); the `reserve_back`-less `push_back` on RunningStatistics' `c_2`
  (A3 — a legacy FIFO defect, carry it forward unchanged); and anomalies A1–A6 generally. Route any
  further findings to `METAL2_PORT_REPORT.md`.

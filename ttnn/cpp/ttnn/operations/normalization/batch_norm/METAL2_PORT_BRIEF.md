# Metal 2.0 Port Brief — `normalization/batch_norm` (both device-operations)

> Audit cleared all gates for **both** device-operations in this directory. This is your actionable input;
> the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓
— for `BatchNormOperation` **and** `RunningStatistics`.

**Recipe docs:** `a38e7b405db 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

**Audited at:** branch `virdhatchani/BN_Porting`, HEAD `a38e7b405db`, merge-base with `main` `f6e166da2c1`.

## Scope — two independent factories, one requested PR

The directory holds **two** device-operations with **one factory each**, sharing no kernels and no factories:

| Device-op | Factory | Kernel entry points to convert |
|---|---|---|
| `BatchNormOperation` | `BatchNormFactory` (`device/batch_norm_program_factory.cpp:140`) | `reader_batch_norm.cpp`, `writer_batch_norm.cpp`, **both** `compute/batch_norm_kernel.cpp` and `compute/batch_norm_sfpu_kernel.cpp` |
| `RunningStatistics` | `RunningStatisticsProgramFactory` (`device/running_statistics_program_factory.cpp:137`) | `reader_running_statistics.cpp`, `writer_running_statistics.cpp`, **both** `compute/running_statistics_kernel.cpp` and `compute/running_statistics_sfpu_kernel.cpp` |

**The invoker has asked for both in the same PR.** They are structurally independent, so port them as two
sequential units (each factory + its 4 kernels is one atomic unit) and land them together. Nothing forces
co-porting; the reason it is the right call is the test set — `test_batch_norm_program_cache.py` drives both
in a single `ttnn.batch_norm(training=True)` call and cannot isolate either.

**Each factory runtime-selects its compute source** (`fmt::format` on
`(fp32_dest_acc_en || any_float32) ? "sfpu_kernel" : "kernel"` — `batch_norm_program_factory.cpp:388-390`,
`running_statistics_program_factory.cpp:438-440`). Both selected sources convert together with their
factory; there is no partial-path build. **8 kernel files total.** Expect no green build until the last one
flips — that is the expected shape, not a stop signal.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor` (both) — `create_descriptor` returning `tt::tt_metal::ProgramDescriptor`
- **Op-owned tensors:** none (both) — leave `ProgramArtifacts::op_owned_tensors` defaulted
- **Target concept:** `ProgramSpecFactoryConcept` (both)
- **Device-op-class edits forced: NONE.** No custom `compute_program_hash` to delete, no pybound
  `create_descriptor` to remove, no pybind-hook-only factory parameter to unwind. The only device-op-header
  change is the two factory signatures (`ProgramDescriptor create_descriptor` →
  `ttnn::device_operation::ProgramArtifacts create_program_artifacts`) plus swapping
  `<tt-metalium/program_descriptors.hpp>` for `"ttnn/metal_v2_artifacts.hpp"` in both device-op headers.
- **Gate-cleared, confirmed absent:** custom hash · `get_dynamic_runtime_args` ·
  `override_runtime_arguments` · pybind `create_descriptor` — all `no` on both rows.

> **⚠ One device-op member you must NOT touch.** `BatchNormOperation::operation_attributes_t::to_hash()`
> (`device/batch_norm_device_operation.cpp:121-123`) is a *backdoor* custom hash — a different mechanism from
> `compute_program_hash`, and **not** covered by the recipe's custom-hash deletion rule. Do not delete it,
> do not patch it. The audit verified it is harmless: it narrows only the *attributes* half of the cache key
> (collapsing `input_dtype`/`dtype` into `get_dtype()`), while `tensor_args` — hence `TensorSpec` — is
> hashed and canonicalized separately, so the `UpdateTensorArgs` legality failure mode does not apply. It is
> device-op-class code, so it is off-limits. `RunningStatistics` has no such member.

## Construct — to do

### Tensor bindings — 11, all Case 1

All eleven feed a `TensorAccessor`, so all are the mechanical case: declare a `TensorParameter` from
`<tensor>.tensor_spec()`, bind it on the kernel that walks it, and collapse the kernel to
`TensorAccessor(tensor::<name>)`. **No Case 2 anywhere** — no kernel does raw base-pointer arithmetic, so you
will not need the `get_bank_base_address` bridge, and the compute kernels construct no `TensorAccessor` at
all (no compute-kernel Case-2 block to worry about).

All eleven arrive today in the **`Buffer*` form** — the factory pushes the `Buffer*` object into
`emplace_runtime_args`, not `->address()`. That is the framework's pointer-patching interim hack; the typed
binding supersedes it, and the address RTA slot disappears entirely from each kernel.

**`BatchNormOperation`** — 6 bindings:

| `TensorParameter` | Origin | Bind on | Kernel accessor site to collapse |
|---|---|---|---|
| input | `tensor_args.input` | reader | `reader_batch_norm.cpp:38` |
| batch_mean | `tensor_args.batch_mean` | writer | `writer_batch_norm.cpp:53` |
| batch_var | `tensor_args.batch_var` | writer | `writer_batch_norm.cpp:61` |
| weight **(conditional)** | `tensor_args.weight` | writer | `writer_batch_norm.cpp:65` |
| bias **(conditional)** | `tensor_args.bias` | writer | `writer_batch_norm.cpp:69` |
| output | `tensor_return_value` | writer | `writer_batch_norm.cpp:57` |

**`RunningStatistics`** — 5 bindings:

| `TensorParameter` | Origin | Bind on | Kernel accessor site to collapse |
|---|---|---|---|
| batch_mean | `tensor_args.batch_mean` | reader | `reader_running_statistics.cpp:39` |
| batch_var | `tensor_args.batch_var` | writer | `writer_running_statistics.cpp:52` |
| running_mean **(conditional, in-place RW)** | `tensor_args.running_mean` | writer | `writer_running_statistics.cpp:58` |
| running_var **(conditional, in-place RW)** | `tensor_args.running_var` | writer | `writer_running_statistics.cpp:61` |
| output | `tensor_return_value` | writer | `writer_running_statistics.cpp:55` |

Two notes on RS so nothing reads as a mistake: the **writer** kernel *reads* `batch_var` and the old stats
(it is a reader-writer, which is also why it carries five accessors), and `running_mean` / `running_var` are
**read and written through the same accessor** — old value in at `writer_running_statistics.cpp:87`, updated
value back to the same pages at `:103`. One `TensorParameter` each covers both directions. Do not confuse
the in-place stat with the op's `tensor_return_value`: that is a separate created tensor whose value the
caller discards (`batch_norm.cpp:131`).

### Conditional tensor bindings — 4, each needs a `#define` promotion

All four optional tensors are constructed **unconditionally** in the kernel today and gated only at their
*uses*. Metal 2.0 emits `tensor::<name>` only where the host binds it, so the **construction** must move
behind an `#ifdef` — mandatory for optional tensors (there is nothing to bind when absent).

| Binding | Absent when | Unconditional construction to gate | Legacy gate |
|---|---|---|---|
| BatchNorm `weight` | `!weight.has_value()` | `writer_batch_norm.cpp:64-65` | CTA 0 `weight_has_value` |
| BatchNorm `bias` | `!bias.has_value()` | `writer_batch_norm.cpp:68-69` | CTA 1 `bias_has_value` |
| RS `running_mean` | `!running_mean.has_value()` | `writer_running_statistics.cpp:57-58` | CTA 0 `old_running_mean_has_value` |
| RS `running_var` | `!running_var.has_value()` | `writer_running_statistics.cpp:60-61` | CTA 1 `old_running_var_has_value` |

Each is a **promote-a-CTA-gate-to-a-define**: emit the condition via
`KernelSpec::compiler_options.defines` and `#ifdef`-gate both the accessor construction and its uses.
**Promotion is needed only on the two writer kernels** — the compute kernels gate on the same conditions but
never reference a `tensor::` token, so their copies stay ordinary named CTAs.

RS guarantees at least one stat is present (`running_statistics_device_operation.cpp:42-44`, mirrored by a
`static_assert` in both RS compute kernels), so you will never face both absent.

### Conditional DFB specs — 3, path-dependent handle aliases

Three CBs exist only in the typecast configuration, and a second kernel-side name resolves to either the
conditional CB or the unconditional staging CB. That is same-FIFO aliasing, path-dependent variant: **one
`#ifdef`-gated `constexpr` alias, not a second binding, and not `alias_with`.**

| Conditional DFB | Exists when | Aliased kernel name | Resolves to |
|---|---|---|---|
| BatchNorm `writer_out` (`c_9`) | `needs_output_typecast` (`batch_norm_program_factory.cpp:227-239`) | `dfb_output_final`, CTA 11 (`batch_norm_sfpu_kernel.cpp:219`) | `writer_out` when typecast, else `out` (`c_2`) |
| RS `writer_updated_mean` (`c_12`) | `needs_mean_typecast` (`running_statistics_program_factory.cpp:285-297`) | `dfb_writer_updated_mean`, CTA 14 (`running_statistics_sfpu_kernel.cpp:64`) | `writer_updated_mean` when typecast, else `updated_mean` (`c_7`) |
| RS `writer_updated_var` (`c_13`) | `needs_var_typecast` (`running_statistics_program_factory.cpp:298-310`) | `dfb_writer_updated_var`, CTA 15 (`running_statistics_sfpu_kernel.cpp:65`) | `writer_updated_var` when typecast, else `updated_var` (`c_8`) |

Emit `NEEDS_OUTPUT_TYPECAST` / `NEEDS_MEAN_TYPECAST` / `NEEDS_VAR_TYPECAST` as host defines — compute the RS
pair **host-side** (`running_mean_has_value && stat_format_needs_typecast`) rather than re-deriving it in the
kernel from two CTAs, so one define gates one alias:

```cpp
#ifdef NEEDS_OUTPUT_TYPECAST
constexpr auto dfb_output_final = dfb::writer_out;
#else
constexpr auto dfb_output_final = dfb::out;
#endif
```

Convenient: `needs_*_typecast` implies `interm_data_format == Float32` implies `any_float32` implies the
**SFPU** source is selected. So the conditional DFBs only ever exist on the SFPU path, and the non-SFPU
compute kernels never read those CTAs (verified: `batch_norm_kernel.cpp` reads CTAs 0–10 only;
`running_statistics_kernel.cpp` reads CTAs 0–13 only).

**Every other DFB is bound unconditionally — that is the faithful choice, not a shortcut.** Legacy allocates
the `weight` / `bias` / `old_running_mean` / `old_running_var` CBs unconditionally even when the tensor is
absent (`batch_norm_program_factory.cpp:260-279`, `running_statistics_program_factory.cpp:222-241`), so
unconditional DFBs reproduce the legacy L1 footprint exactly. The kernels also reference those handles
*outside* their `if constexpr` guards (e.g. `writer_batch_norm.cpp:49` and `:64` call
`dfb_weight.get_entry_size()` unconditionally), so unconditional binding is required as well as faithful.

### CB endpoints

Re-derive these from the kernel-touch census yourself; they are recorded here so you know what to expect,
not to be transcribed. **No multi-binding flag anywhere, and no dead CB** — every out-of-window CB is a
one-toucher self-loop or a two-toucher 1P+1C.

- **Self-loop** (compute produces *and* consumes; one toucher): BatchNorm `den` (`c_7`), `temp_1` (`c_8`);
  RS `tmp1` (`c_9`), `tmp2` (`c_10`), `tmp3` (`c_11`).
- **Plain 1P+1C** — everything else. Note the *producer* is not always the reader: the **writer** kernel
  produces BatchNorm `batch_mean` / `batch_var` / `weight` / `bias` and RS `batch_var` /
  `old_running_mean` / `old_running_var`, because it reads those tensors from DRAM on the compute kernel's
  behalf. Map roles from the kernel bodies, not the kernel names.
- **⚠ Four CBs flip disposition with config** — classify per instantiation:

  | CB | Without typecast | With typecast |
  |---|---|---|
  | BatchNorm `out` (`c_2`) | compute P → writer C = **1P+1C** | compute P + compute C = **self-loop** (writer consumes `c_9` instead) |
  | RS `updated_m` (`c_7`) | compute P → writer C = **1P+1C** | compute P + compute C (in `maybe_typecast_stat`) = **self-loop** (writer consumes `c_12`) |
  | RS `updated_v` (`c_8`) | compute P → writer C = **1P+1C** | same, keyed on `needs_var_typecast` (writer consumes `c_13`) |
  | RS `updated_m`/`_v` are independently keyed | — | `needs_mean_typecast` and `needs_var_typecast` are separate booleans; one may typecast while the other does not |

  A single disposition applied across configs mis-binds the other one.
- `weight` / `bias` / `old_running_mean` / `old_running_var` are **1P+1C in all configs**, including when
  their tensor is absent: both writer and compute bind them (needed for the handle references above) and the
  FIFO ops are simply gated off. Two role-free-in-that-config touchers → 1P+1C, no flag.

### Hardware configuration — the silent-regression surface

Both factories are Style A (they resolve a TTNN `DeviceComputeKernelConfig`), so use
`ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config)`.

- **This op's defaults are non-standard.** `resolve_compute_kernel_config` (`batch_norm_utils.cpp:14-38`)
  sets `default_fp32_acc = true` and, on Wormhole, `default_fp32_acc_math_fidelity = HiFi3` (a documented
  workaround for hardware bug #38306). So `enable_32_bit_dest = true` is the **common** path here, not a
  corner case — which makes `unpack_modes` load-bearing on the default configuration.
- **`unpack_modes` — re-key, do not re-derive.** Legacy builds a `vector<UnpackToDestMode>` indexed by CB id
  and sets `UnpackToDestFp32` for a fixed list, only under `if (fp32_dest_acc_en)`
  (`batch_norm_program_factory.cpp:352-368`, `running_statistics_program_factory.cpp:394-411`). Port each
  entry to a `Table<DFBSpecName, UnpackMode>` keyed by DFB name, value `UnpackMode::UnpackToDest`. The exact
  legacy sets:
  - **BatchNorm:** `input`, `batch_mean`, `batch_var`, `eps`, `den`, `weight`, `temp_1`, `bias` — **plus**
    `out` (`c_2`) when `needs_output_typecast`. `writer_out` (`c_9`) gets **no** entry.
  - **RS:** `batch_mean`, `batch_var`, `output`, `old_running_mean`, `old_running_var`, `updated_m`,
    `updated_v`, `momentum`, `one`, `tmp1`, `tmp2`, `tmp3`. `writer_updated_m` / `writer_updated_v` get
    **no** entry.
  The audit traced both sets against the validator and they are **legal and complete as-is** — no entry to
  add, none to drop. Two facts that let you stop worrying about the two rules that look threatening here:
  every entry sits under `enable_32_bit_dest == true`, which the validator accepts unconditionally (so the
  "≤16-bit format + `UnpackToDest` is rejected on Gen1" rule cannot fire, even on the
  `fp32_dest_acc_en && !any_float32` path where `den`/`temp_1`/`eps` are `Float16_b`); and the legacy list
  already covers every DFB the compute kernel *consumes*, so the newly-required-explicit-entry rule for
  consumed Float32 DFBs is already satisfied. The two omitted CBs are producer-only for compute.
  **The risk here is a copy-paste between the two factories, or a dropped key — both silent.**
- **DM configs are plain defaults** — `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` on all four
  dataflow kernels. Use `ttnn::create_reader_datamovement_config(device->arch())` /
  `create_writer_datamovement_config(device->arch())`. No custom triple, no `DM_DYNAMIC_NOC`.
- **`opt_level`: set `O3` explicitly on all four compute `KernelSpec`s.** Neither factory sets `opt_level`
  anywhere (grep: zero hits), so the legacy resolved level is `O3` for compute (the `ComputeConfigDescriptor`
  default) and `O2` for DM. Metal 2.0 defaults to `O2` for both, so the compute specs silently drop a level
  unless you state it: `.compiler_options = {.defines = …, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3}`.
  The four DM specs need nothing.

### Runtime args — all named, per node, including the idle ones

Every runtime arg is read exactly once as a distinct field at a literal index — no counted loops, no
`arg_index++` runs, no data-selected indices. **All RTAs become named args; no varargs anywhere.**
Arities after the address slots drop out: BatchNorm reader 9 → 8 named, writer 12 → 7 named, compute 3;
RS reader 9 → 8 named, writer 11 → 7 named, compute 1.

**Both factories place their kernels on all device cores** and hand the cores outside both work groups an
all-zero RTA vector (`batch_norm_program_factory.cpp:72-79`, `running_statistics_program_factory.cpp:71-78`)
rather than narrowing `core_ranges`. Preserve that verbatim: one `WorkUnitSpec` over all device cores per
factory, and a named-RTA value for **every** node (`SetProgramRunArgs` requires completeness), with `0` on
the idle nodes. Do **not** narrow the work unit to the working cores — that would change kernel placement.
The kernels already handle the zero case (`batch_norm_sfpu_kernel.cpp:205-207` returns early on
`num_tiles == 0`).

The legacy loop is node-first, so keep it and let `AddRuntimeArgsForNode` transpose into the name-first
table; do not re-architect the loop.

### No work-split multiplicity

Both factories call `split_work_to_cores` but use the result **only** for per-core RTA values — the two core
groups get identical CTAs and a single `KernelDescriptor` per kernel over `all_device_cores`. So there is no
preserved-multiplicity case and no per-group `KernelSpec` / `WorkUnitSpec` split. One work unit per factory.
(This is the *inverse* of the demoting-per-group-CTA anti-pattern: there is nothing to preserve — do not
invent a split.)

## Watch for

- **CB endpoints (multi-binding):** none. No CB reaches ≥3 touchers or doubles a FIFO role; the
  hidden-second-writer hunt found nothing (the op declares **no semaphores at all**, so there is no
  semaphore-gated raw co-fill to miss). If you find yourself reaching for
  `allow_instance_multi_binding`, recount — and never stack it with a self-loop.
- **Cross-op / shared kernels:** **none — no `_metal2` fork needed.** All eight kernel sources live in this
  directory and are bound only by this directory's factories (verified per-file). Coupling is limited to
  three donor *headers*, all of which cross cleanly with no bridge work:
  - `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp` — takes raw
    `uint32_t l1_write_ptr`; nothing to bridge. Keep passing `dfb.get_write_ptr()`.
  - `ttnn/cpp/ttnn/kernel/dataflow/cb_fill_helpers.hpp` — `fill_cb_with_value(uint32_t cb_id, …)`, called at
    `reader_running_statistics.cpp:56`. Pass `dfb::one` directly; the constexpr `DFBAccessor → uint32_t`
    conversion handles it. Do **not** extract `.id` or build a temporary wrapper.
  - `ttnn/cpp/ttnn/kernel/compute/dest_format_helpers.hpp` — `pack_tile_with_dt`,
    `copy_tile_to_dst_init_short_with_dt`, `copy_tile_init_with_dt`; `uint32_t` cb ids, same conversion.
  None of these is yours to edit. If you find you need to, stop and record a Handoff point.
- **The compute kernels select DFB handles at runtime — keep them `uint32_t`-valued.** Both BatchNorm
  compute sources compute `dfb_affine_or_out` / `dfb_scaled_output` from `weight_has_value` /
  `bias_has_value` at runtime (`batch_norm_sfpu_kernel.cpp:42-43`) and construct a `DataflowBuffer` from the
  result. Assign `dfb::name` into a `uint32_t` local exactly as-is — the implicit conversion makes the
  reassignment legal. Do not try to make these `constexpr`, and do not restructure the selection.
- **`weight_has_value` / `bias_has_value` are *runtime* `if`s in the BatchNorm compute kernels**, not
  `if constexpr` (`batch_norm_sfpu_kernel.cpp:82, 117, 140`). That is why the weight/bias DFBs must be bound
  unconditionally on compute. Leave the runtime-`if` structure alone; it is not yours to tighten.
- **RTA varargs:** none — name every runtime arg. Report it if you end up with any `get_vararg`.
- **Comments are load-bearing here and easy to lose.** Several carry real hazard information you must
  preserve — the `HAZARD:` block explaining why `batch_mean`/`batch_var` must be popped unconditionally
  (`running_statistics_kernel.cpp:46-48`), the numerical-stability note on the two-pass variance
  (`batch_norm.cpp:110-113`), and the `last_srca_*` reconfiguration commentary
  (`batch_norm_sfpu_kernel.cpp:16-20`). The `#ifdef` blocks you are adding will sit right on top of some of
  them. Keep them; relocate rather than delete when a CB-id line goes away — the role comments on the CB
  index constants belong on the `DFBBinding` or the `DataflowBuffer` construction.
- **Do not "fix" the two anomalies the audit flagged.** `packer_l1_acc` is resolved and then dropped on the
  floor (`batch_norm_program_factory.cpp:349`) — Metal 2.0 has no field for it either, so there is nothing to
  carry over; leave it. And the 4–5 trailing CTAs the non-SFPU compute kernels never read are pre-existing;
  the per-source `KernelSpec` shape drops them naturally, so just note it in the report rather than treating
  it as a find.
- **Test baseline** — confirm with the invoker before relying on it (the audit's Question 2). The coverage is
  under the **`fused/`** slug, not `normalization/`:
  - `tests/ttnn/unit_tests/operations/fused/test_batch_norm.py` — primary functional coverage, both
    device-ops, both dtypes, all weight/bias/mean/var combinations.
  - `tests/ttnn/unit_tests/operations/fused/test_batch_norm_program_cache.py` — **run this one first after
    the build goes green.** It pins program-cache keying *and* the running-statistics in-place side effect
    across cache hits — precisely the surface Metal 2.0's `UpdateTensorArgs` cache-hit path touches.
  - Excluded: the `sweep_framework` sweep (unimportable today) and
    `tests/tt_eager/.../fallback_ops/test_batch_norm2d.py` (a torch fallback op, never calls
    `ttnn.batch_norm`). No C++ gtests, no nightly variants.
  - While converting kernel-by-kernel, `-k`-exclude the paths whose compute source you have not converted
    yet: a selected-but-unconverted kernel hits a positional `get_compile_time_arg_val(0)` that
    `static_assert`s at JIT and can take down the whole pytest session (exit 139) rather than failing
    cleanly.

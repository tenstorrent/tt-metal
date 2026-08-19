# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ *(N/A — no accessor passes one)*

**Recipe docs:** `9440205cf62 2026-08-19 docs(metal_2.0): have the porter prove the legality checks are running` *(carry this line into the port report's Provenance section)*

**Scale:** one device operation, two factories (200 and 264 lines), five kernels, no semaphores, no
borrowed memory, no op-owned tensors, no custom hash. This is a small port — both factories are
comfortably one pass. See the shared-writer note under *Watch for*: they should convert **together**.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both factories port to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `create_descriptor()` returning `ProgramDescriptor`, on both
  factories (`device/moreh_linear_backward_device_operation.hpp:33,40`).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (base concept — the op has no
  `override_runtime_arguments`, so the framework refreshes tensor bindings on a cache hit and you
  write one method per factory).
- **Device-op-class edits forced: none.** All three sanctioned exceptions were checked and none
  applies — no pybound `create_descriptor` (the nanobind file binds only the user-facing
  `ttnn::moreh_linear_backward`), no pybind-hook-only factory parameter, and the op already has a
  `program_factory_t` variant with both factories as nested structs
  (`device/moreh_linear_backward_device_operation.hpp:46`), so it is **not** the direct-descriptor
  shape. Your only header change is the two signatures:
  `ProgramDescriptor create_descriptor(...)` → `ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`,
  plus swapping `#include <tt-metalium/program_descriptors.hpp>` for `#include "ttnn/metal_v2_artifacts.hpp"`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none`
  `TensorParameter relaxation` · `get_dynamic_runtime_args`. A custom hash, an
  `override_runtime_arguments`, and a pybound `create_descriptor` are **not** in that list because none
  of them gates — and all three happen to be absent here too, so there is nothing to leave alone.

## Construct — to do

**Tensor bindings** (two per factory, identical in both):

- `output_grad` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`;
  reader uses `TensorAccessor(tensor::<name>)`. The legacy delivery is the `Buffer*`-binding form —
  `reader_desc.emplace_runtime_args(core, {output_grad_buf, …})` — so the address does **not** appear as
  `->address()`; grep for `emplace_runtime_args` / a bare `Buffer*`, not for `->address()`, when you
  self-audit this.
- `bias_grad` — **Case 1**, same shape on the output side (`writer_desc.emplace_runtime_args(core, {bias_grad_buf, …})`).
- `bias` is **not** a binding — it is read only on the host, for the output spec. Do not declare a
  `TensorParameter` for it.

Both `TensorAccessorArgs(*…buffer()).get_compile_time_args()` calls
(`..._single_core_program_factory.cpp:128-131`, `..._multi_core_program_factory.cpp:132-135`) disappear
entirely, along with the kernel-side `TensorAccessorArgs<0>()` lines and the `src_addr` / `src0_addr` /
`dst_addr` reads. Note both readers and the writer have **no other CTAs** — their `compile_time_args`
is *only* the accessor plumbing, so after the port their `compile_time_args` is empty.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor in the op passes one, so there is nothing to drop.

**CB endpoints** (classified per `(CB, factory)`; the two factories differ only on `c_2`):

- **Legal 1:1, both factories** — `c_0` (reader→compute), `c_1` (scaler: reader→compute), `c_16`
  (compute→writer). Bind one PRODUCER + one CONSUMER; nothing special.
- **`c_2` (mask_h_w)** — legal 1:1 (reader→compute) in **both** factories, but in
  `SingleCoreProgramFactory` the CB is **allocated conditionally**: `in2_t = (do_mask_h || do_mask_w) ? 2 : 0`
  and the `CBDescriptor` sits behind `if (in2_t > 0)` (`..._single_core_program_factory.cpp:43,86-96`).
  → make that factory's `DataflowBufferSpec` **conditional** on the same host-time condition, and apply
  the *Conditional / optional DFB bindings* pattern in `port_patterns.md`.
  The multi-core factory allocates it unconditionally (`..._multi_core_program_factory.cpp:64`) — keep
  that unconditional. Detail in *Watch for*, because the kernel side needs care.
- **Compute self-loop, both factories** — `c_24` (intermed0) and `c_25` (intermed1/accumulator). Each is
  touched by the **compute kernel only**: `c_24` is `reserve_back`/`push_back`-produced and then consumed
  as the `reduce<…, cb_intermed0, …>` input; `c_25` is written as the `reduce` output and read back via
  `Accumulate::at(cb_intermed1, …)`. Bind compute as **both PRODUCER and CONSUMER**. These are the
  legitimate accumulator/staging case — supported on Gen2 as well as Gen1, so no Quasar debt.
- **No dead CB. No multi-binding — do not set `allow_instance_multi_binding` anywhere.** The census was
  run per node for all twelve `CBDescriptor`s and no CB has ≥3 touchers or two kernels locked to the
  same FIFO role. Neither factory uses the dual-instance work-split shape.
- One interaction not to conflate: the conditional DFB is `c_2`; the `unpack_modes` entry below is on
  `c_25`. Different DFBs, so the "gate the `unpack_modes` entry on the binding's condition" rule does
  not bite here.

**`unpack_modes` — the two factories need DIFFERENT values. This is the highest-risk item in the port.**

Both factories build `std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, Default)`
and then diverge:

| Factory | Legacy | Metal 2.0 | Why |
|---|---|---|---|
| `MultiCoreProgramFactory` | `unpack_to_dest_mode[CBIndex::c_25] = UnpackToDestFp32` when `fp32_dest_acc_en` (`:159-163`) | `{INTERMED1_DFB, UnpackMode::UnpackToDest}` when `fp32_dest_acc_en` | faithful translation of an explicit legacy setting |
| `SingleCoreProgramFactory` | **never modified** — every entry stays `Default` (`:160`) | `{INTERMED1_DFB, UnpackMode::UnpackToSrc}` when `fp32_dest_acc_en` | legacy `Default` ⇒ `UnpackToSrc`; Metal 2.0 **requires the entry explicitly** here |

The required-entry rule fires because `c_25`'s format is `fp32_dest_acc_en ? Float32 : cb_data_format`
(`..._single_core_program_factory.cpp:64,116-122`) and `enable_32_bit_dest` comes from the same
`fp32_dest_acc_en` — so under fp32 the compute kernel consumes a Float32 DFB with a 32-bit dest, which
legacy defaulted silently and the Metal 2.0 validator will not. Gate both entries on
`fp32_dest_acc_en`, exactly as the multi-core factory already gates its legacy assignment.

**Do not copy one factory's value into the other.** `UnpackToDest` and `UnpackToSrc` are a
precision/performance tradeoff with no compile signal, and the *wrong-valued* entry is silent even
though the *missing* one is loud. `c_24` needs no entry in either factory — its format is
`cb_data_format`, never Float32.

Follow the recipe's `std::get<ComputeGen1Config>(compute_hw).unpack_modes = …` form. The
`unpack_modes(cfg)` common-field accessor exists, but converting to it is the job of the
`gen2_hardware_configs` post-port pass — leave it for that pass rather than pre-empting it.

**Compute config — Style A, and no field is dropped.** Both factories resolve a TTNN
`DeviceComputeKernelConfig` (built in `ttnn::prim::moreh_bias_add_backward`,
`device/moreh_linear_backward_device_operation.cpp:79`) and destructure it via
`get_compute_kernel_config_args`. Use
`ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config)`. I checked the
resolved-vs-set comparison for you: the factories set `math_fidelity`, `fp32_dest_acc_en`,
`dst_full_sync_en`, `unpack_to_dest_mode` and `math_approx_mode` — every resolved field that has a
Metal 2.0 counterpart. `packer_l1_acc` is resolved but has no counterpart (no action).
`bfp8_pack_precise` is never set, so `bfp_pack_precision_mode` stays at its matching default. **No
dropped-field correction needed.** Mind the two non-1:1 transforms the helper handles for you
(`math_approx_mode` bool→`Precision`, and the `dst_full_sync_en` → `double_buffer_dest` **inversion**) —
do not re-apply them by hand on top of the helper's output.

**DM configs — both are the role defaults.** `ReaderConfigDescriptor{}` and `WriterConfigDescriptor{}`
in both factories, so the resolved triples are exactly the reader and writer defaults and the roles
match the names. Use `ttnn::create_reader_datamovement_config(device->arch())` and
`ttnn::create_writer_datamovement_config(device->arch())`. No custom triple anywhere, no `noc_mode`
override, so the paired-per-node `DM_DYNAMIC_NOC` concern does not arise.

**Compute `opt_level` — set it explicitly on every compute `KernelSpec`.** `grep -n opt_level` over the
whole op returns **zero hits**, so all three compute descriptors (single-core, multi-core G1, multi-core
G2) resolve to the legacy `ComputeConfigDescriptor` default **O3**. Metal 2.0's `CompilerOptions`
defaults to O2, so each compute `KernelSpec` needs
`.compiler_options = {.defines = …, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3}`. The DM kernels
need nothing (legacy O2 = Metal 2.0 O2). Remember this is *per `KernelSpec`*, so the multi-core factory
needs it on **both** compute specs.

**Preserved multiplicity — `MultiCoreProgramFactory`.** Two compute `KernelDescriptor`s of the *same*
source over **disjoint** core groups (`compute_desc_1` on `core_group_1` at `:170`, `compute_desc_2` on
`core_group_2` at `:187`, both guarded by `has_core_group_2`). Port as **two `KernelSpec`s of the same
source in two `WorkUnitSpec`s**, reader and writer belonging to both work units (their `core_ranges` is
`all_cores`) — structurally identical to the merged `moreh_mean` H factory. Each node sees exactly one
compute instance, so their shared-DFB bindings are ordinary single-role bindings; this is **not** the
`allow_instance_multi_binding` case and **not** the two-toucher 1P+1C case.

There is a wrinkle, and it is a trap — see *Watch for*: **that per-group CTA is dead.**

**Runtime args.** Both factories set RTAs node-first in a core loop. Use `AddRuntimeArgsForNode` and
keep the loop as it is — do not invert it to name-first. The single-core factory's three
`emplace_runtime_args` calls are on one core (`{0,0}`); `MakeRuntimeArgsForSingleNode` fits there.

## Watch for

- **CB endpoints (multi-binding):** none. The hidden-second-writer and multi-reader faces were hunted
  and did not fire — no kernel takes a raw pointer into a CB another kernel writes, and there are no
  semaphore-gated co-fills (the op has no semaphores at all). You should not need
  `allow_instance_multi_binding`; if you find yourself reaching for it, re-run the census and say so in
  the report.

- **Cross-op / shared kernels:** `device/kernels/writer_moreh_bias_backward.cpp` is bound by **both**
  factories — an *intra-op* shared kernel (`..._single_core_program_factory.cpp:143`,
  `..._multi_core_program_factory.cpp:147`). No `_metal2` fork exists beside it. Nothing outside this op
  binds it (or any of the other four kernels) — the lent/borrowed census came back empty, so there is
  **no sunset list and no cross-op coordination cost**. Because the factories share it, converting one
  alone breaks the other: **port both factories in the same change.** That keeps you at rung 0 (no
  fork). If you must split, the shared-kernel rung is a `writer_moreh_bias_backward_metal2.cpp` fork
  beside the original — but for an op this size that is unnecessary cost.

- **RTA varargs:** **none needed — and this op is shaped exactly like the trap.** Every kernel walks its
  runtime args with `ArgFetcher` (`moreh_common.hpp:44` / `:128`), whose `get_next_arg_val<T>()` is a
  running `get_arg_val<T>(arg_idx++)`. That is a **fixed run of distinct fields read once each at the top
  of the kernel** — the *non-signal* case. **Name every one of them**; do not translate
  `get_next_arg_val` into `get_vararg`. No kernel has a variable-count loop, a data-selected index, or a
  sentinel-terminated scan, and none reads any compile-time arg, so there is no CTA-vararg case either.
  `ArgFetcher` itself disappears from the ported kernels; the donor header is untouched.

- **The multi-core compute CTA is dead — preserve it anyway, and name it.**
  `compute_desc_{1,2}.compile_time_args = {num_cols_per_core_group_N}` (`:171`, `:188`), but
  `moreh_bias_backward_multi_core_h.cpp` reads **no** compile-time argument at all (zero
  `get_compile_time_arg_val` hits across all five kernels). The value the kernel uses arrives as RTA
  slot 2 (`Wt_per_core`, `:14`), fed from the same `num_cols_per_core`. So the two compute descriptors
  differ *only* in an unread CTA. **Do not "simplify" to one `KernelSpec`** — that collapses the
  per-group multiplicity, and dropping the CTA is an owner decision, not port work. Reproduce it as a
  named CTA with each group's value. A dead *named* CTA compiles fine and this is the sanctioned shape:
  the merged `moreh_mean` NC factory does exactly this
  (`../moreh_mean/device/moreh_mean_nc_program_factory.cpp:218` declares
  `{{"units_per_core", units_per_core}}`, and `../moreh_mean/device/kernels/moreh_mean_nc.cpp` never
  reads it). Since no kernel variable exists to infer a name from, `units_per_core` is the precedent —
  and it matches the host-side variable. Record the dead CTA in the port report.

- **`get_tile_size(cb_id)` → the member getter, not the token form.** Three sites:
  `reader_..._hw.cpp:43`, `reader_..._h.cpp:39`, `writer_...cpp:24`. All three are declared
  `const auto`, **not** `constexpr`, so kernel-side whitelist rule 7 wants
  `dfb_in0.get_tile_size()` / `dfb_out.get_tile_size()` — the `get_tile_size(dfb::name)` token form is
  reserved for a value the legacy kernel declared `constexpr`, and none here is. The merged
  `moreh_mean` reader/writer made exactly this swap. (These are *sanctioned* Device 2.0 free functions,
  so they were not a Device 2.0 gate — but the port does move them onto the object.)

- **The conditional `c_2` needs a compile-time gate promoted from a runtime one, in the single-core
  kernels.** Today the mask CB's *use* is gated at runtime (`if (do_mask_h || do_mask_w)` in
  `reader_..._hw.cpp:34` and `..._single_core_hw.cpp:33`, driven by RTAs), while its *allocation* is
  gated at host time. In Metal 2.0 the token `dfb::mask_h_w` will not exist at all in a no-mask build,
  so every reference must be behind a preprocessor gate: emit a define from the host on the same
  condition as the binding, and `#ifdef` the kernel-side construction and uses. Two specific traps:
  - `moreh_bias_backward_single_core_hw.cpp:21-22` constructs `DataflowBuffer dfb_mask_h_w_obj(cb_mask_h_w)`
    **unconditionally**, outside the runtime `if` — that construction, and the `wait_front` at `:34`,
    both need gating.
  - The `if (do_mask)` block references `cb_mask_h_w` in two sub-blocks (`:59-64`, `:69-74`); those
    references need gating too, even though the block is unreachable when neither mask applies.

    Keep the existing runtime `if`s nested inside the `#ifdef`s — they are redundant when compiled in
    (the two conditions coincide), and removing them would be kernel-logic surgery you are not entitled
    to. The multi-core kernels need none of this: their `c_2` is unconditional.

- **A misleading name you will have to bake in.** The multi-core factory passes
  `num_tiles = batch_num * Ht` (`:40`) as reader RTA slot 1, and the reader unpacks it into a local
  named `batch_num` (`reader_..._h.cpp:13`); the multi-core compute kernel does the same (`:12`). The
  value is correct — a column spans `batch_num * Ht` tiles — but the name understates it, and naming
  this argument is now unavoidable. Follow the recipe (name it after the kernel-side variable, so
  `batch_num`) **and add a short comment at the `compile_time_args`/schema site saying what the value
  actually is**, which is how the merged `moreh_mean` H factory handled the same wart on its
  `units_per_core`/`Wt` pair. Do not rename the kernel local — that is an ops-team cleanup, noted in the
  audit's Misc anomalies.

- **Tests are nightly-only, and they do cover the risky paths.** The op's only coverage is
  `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_linear.py` — there is no non-nightly test
  file for it, so the usual "skip nightly unless you suspect a regression" guidance does not apply here.
  Confirm the set with your invoker as the recipe requires; the relevant cases are
  `test_moreh_linear_backward` (parameterized `fp32_dest_acc_en ∈ {False, True}` × `requires_bias_grad`
  × shapes that include both scalar `[1,1]` bias → **SingleCore** and 1-D bias → **MultiCore**),
  `test_moreh_bias_backward_fp32` (MultiCore, fp32 only), and
  `test_moreh_linear_backward_enable_cache` (the program-cache hit path, both factories).
  Consequence worth knowing: **both factories' fp32 paths are exercised**, so a *missing* `unpack_modes`
  entry will fail loudly — but a *wrong-valued* one still will not. Also note the file exercises
  `moreh_matmul` and `moreh_sum` through the same composite entry point, so a failure there is not
  necessarily yours.

- **Legality checks:** nine `skip_validation` sites across
  `tt_metal/impl/metal2_host_api/program_run_args.cpp` and `program_spec.cpp` as of this audit — force
  them all and prove both `METAL2_CHECKS_FORCED` markers appear before trusting any green. The spec-side
  choke point is `BuildProgramFromSpec` (`program_spec.cpp:2845`); on this op's base concept the
  cache-hit path runs through `UpdateTensorArgs` (`program_run_args.cpp:797`), which does carry the
  parameter in this tree. Never commit any of it.

# Metal 2.0 Port Report — `moreh_group_norm_backward`

> Read with `moreh_layer_norm_backward`'s report. The two ops were ported as one change: this op owns
> no compute kernel, so every atomic unit it belongs to includes a layer-norm factory. That report
> carries the shared compute kernels' internal analysis and the cross-cutting friction entries; this
> one carries what is specific to this op.

## Outcome

**`PORTED`** — both factories converted, and the confirmed test set passes with the same result as the
pre-port baseline.

- `MorehGroupNormBackwardGammaBetaGradFactory` → `ProgramSpecFactoryConcept` ✓
- `MorehGroupNormBackwardInputGradFactory` → `ProgramSpecFactoryConcept` ✓ (both selectable reader
  sources and both selectable borrowed compute sources converted together)

Nothing is left for a later pass on this op.

## Provenance

- **Recipe docs (this port):** `82c2b5569eb 2026-08-13 docs(metal_2.0): stop gating on a pybound create_descriptor, and drop a stale reference port`
- **Audit docs (inherited):** `82c2b5569eb 2026-08-13 docs(metal_2.0): stop gating on a pybound create_descriptor, and drop a stale reference port`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` on both factories, as the audit chose. Neither ported-from factory had an
`override_runtime_arguments`, so no cache-hit override was written; the framework refreshes tensor
bindings on its own.

### Device-op-class edits

- **Pybind entry points removed:** none. `moreh_group_norm_backward_nanobind.cpp` exposes only the
  user-facing op.
- **Custom `compute_program_hash`:** none on either device-operation (default reflection hash, no
  `attribute_values` / `to_hash` backdoor). Nothing to leave intact.
- The only device-op-class edits are the two forced signature changes,
  `create_descriptor` → `create_program_artifacts`, in
  `device/gamma_beta_grad/moreh_group_norm_backward_gamma_beta_grad_device_operation.hpp:39` and
  `device/input_grad/moreh_group_norm_backward_input_grad_device_operation.hpp:32`, plus swapping the
  now-unused `<tt-metalium/program_descriptors.hpp>` include for `ttnn/metal_v2_artifacts.hpp`. Both
  `*_device_operation.cpp` files are byte-identical to their pre-port revision.

### Open items

- **Relaxation candidates:** none identified. Every `TensorParameter` stays strict.
- **RTA → CRTA candidates.** `num_inner_tiles`, `num_channels`, `num_groups`, `origin_h`, `origin_w`
  (both readers), `num_inner_tiles` and `batch` (both writers) carry the same value on every node and
  are really common runtime args. Not converted — RTA→CRTA changes dispatch semantics, which is
  outside a port.
- **Name-first `runtime_arg_values` restructure.** Both factories still build run-args from the legacy
  node-first core loop, bridged by `AddRuntimeArgsForNode`. A separate cleanup.

## Handoff points

**None.** No capitulation, no boundary-rule assumption violation, no kernel-lib gap, no framework gap,
no removed pybind surface.

The bundled in-place conversion of the three borrowed compute kernels was **not** a boundary
violation: it is [shared-kernel rung 3](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel),
taken under the invoker's explicit assignment of both ops to one branch and PR (on record 2026-08-13,
restated in the request that opened this port). The census was re-run before touching the files and
confirms the assigned set is the complete set — exactly two consumers each, this op and its owner —
so no `_metal2` fork was created and none is needed.

## Successes

- **[Caution: Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)
  — "a consumer list is a sunset and coordination list, not authorization."** This op borrows *all
  three* of its compute kernels, so rung 3 was the difference between a 5-file port and a 13-file one.
  The entry's insistence on confirming the assignment against the census (rather than reading the
  brief's consumer table as a licence) is exactly the check that made in-place conversion safe here —
  and it is also what established that no fork and no pointer comment were needed, since both binders
  land in the same change.
- **The same entry's "name the bindings for the kernel, not for your op."** Three names had to be
  conceded to the kernel's vocabulary: `c_4` is *one* in this factory but `cb_scaler` in the kernel
  (→ `scaler`); `c_5` in input_grad is *inner_size(==n)* here but `cb_n_recip_n` in the kernel
  (→ `n_recip_n`); and the compute CTA at slot 3 is `num_inner_tiles` here but `NCHt` in the kernel.
  Naming those for this factory's locals would have produced an interface the sibling op could not
  reuse — and it is the sibling that owns the file.
- **[Compute kernels — Style A vs Style B](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compute-kernels).**
  This op sets an all-default `ComputeConfigDescriptor{}` while its sibling resolves a TTNN
  `ComputeKernelConfig`. Building `ComputeGen1Config{}` directly (rather than routing through
  `ttnn::to_compute_hardware_config`, which would have been the obvious move after doing the sibling
  factory) is what keeps this op's fidelity, approx-mode and dest-buffering where they were. Verified
  field by field: legacy `ComputeConfigDescriptor{}` defaults are HiFi4 / fp32_dest_acc_en=false /
  dst_full_sync_en=false / bfp8_pack_precise=false / math_approx_mode=false, and
  `ComputeGen1Config{}` defaults are HiFi4 / Precise / Approximate / enable_32_bit_dest=false /
  double_buffer_dest=true — the same five values under the renamed, partly-inverted vocabulary.
- **[Caution: Avoid varargs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  — trap (1) fired here and the caution caught it.** Every one of this op's five dataflow kernels walks
  its runtime args with a running `int i{0}; get_arg_val<uint32_t>(i++)`, and two readers assign CB ids
  with a second running counter. The mechanical translation of that shape is `get_vararg(i++)`, which
  compiles and runs. The caution's "`arg_index++` is not a vararg signal — these are distinct fields
  read once each, in a fixed block, before any loop" is precisely why all 23 of them became named args
  and both counters disappeared. This port declares **zero** varargs.

## Friction

The cross-cutting entries — the same-FIFO-aliasing object rule, the optional-output copy footgun, and
the `fill_cb_with_value` hits in the `[Cc][Bb]_` sweep — are written up once in
`moreh_layer_norm_backward`'s report and apply here identically. Specific to this op:

### Gaps

- **Nothing tells the porter how to reconcile *two* factories' defines for one shared kernel.** The
  conditional-binding pattern is written from a single factory's point of view: bind conditionally,
  emit the matching define. With a shared kernel it becomes a *consistency* requirement across
  binders — this op emits `DO_MASK_W` (its `do_mask_w` can be true) and the sibling never does (its
  `is_groupnorm` is compile-time false), so the same `#ifdef DO_MASK_W` branch is live on one side and
  dead on the other. Getting that wrong in the tidying direction — deleting the branch because *your*
  op never takes it — silently breaks the other binder, and nothing in the port would catch it: the
  build is green and the tests are green on the side you looked at. A sentence in the shared-kernel
  Caution ("a branch that is dead for your factory may be live for another binder; the file's
  `#ifdef` set is the union across binders, not your factory's subset") would close it.
- **The reader's mask predicate is a *runtime* value, which the pattern's examples do not cover.**
  This op's readers compute `do_mask_h` / `do_mask_w` from `origin_h` / `origin_w`, which arrive as
  **RTAs**, so the legacy gate is a runtime `if`, not a `constexpr` one. The host computes the same
  predicate at spec-construction time, so promoting it to a define is faithful — but the pattern's
  worked examples are all CTA-gated, and it took a second look to be sure that a *runtime*-gated use
  of a *host-conditionally-bound* buffer is the same case. It is (the host is the source of truth for
  whether the buffer exists), and the port keeps the runtime `mask_h` / `mask_w` value computation
  while gating only the buffer's construction and use. Worth one line in the pattern.

### Confusion

- **`ComputeConfigDescriptor{}` reads as "nothing to carry over."** An all-default legacy config looks
  like the case where you can safely reach for whichever Metal 2.0 constructor is convenient. It is
  the opposite: it is the case where *every* field is load-bearing, because there is no explicit value
  anywhere to compare against afterwards. The recipe does say this; the confusion is that the signal
  (`ComputeConfigDescriptor{}`) looks like an absence of information rather than a specification.

## Open items for downstream

- **Shared kernel touches — three kernels borrowed from `moreh_layer_norm_backward`, rung 3
  (converted in place, in the owner's directory), both binders converted.**

  | kernel path | rung taken | remaining unmigrated consumers |
  |---|---|---|
  | `.../moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` | **in-place** | **none** |
  | `.../moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp` | **in-place** | **none** |
  | `.../moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp` | **in-place** | **none** |

  No `_metal2` fork was created and none is needed — no consumer is left behind, so there is no sunset
  checklist. The binding vocabulary those files now expose is listed in the owner op's report; a future
  binder inherits it rather than choosing its own.

- **Findings — pre-existing behavior shipped forward unchanged.** None of these was fixed; each is a
  candidate for a separate PR by the op owners.
  1. **Both factories set `ComputeConfigDescriptor{}` — all defaults — while the op carries a
     `compute_kernel_config` operation attribute that reaches the factory and is never read.**
     (`device/gamma_beta_grad/..._factory.cpp` and `device/input_grad/..._factory.cpp`; the attribute
     is declared at `..._device_operation.hpp:24` and `:17` respectively.) A user-supplied
     `math_fidelity` / `fp32_dest_acc_en` / `dst_full_sync_en` on this op is silently ignored, and the
     sibling op *does* honor its equivalent. This is the most substantive finding in this op: it is
     either a deliberate pin that deserves a comment, or a dropped wire-up. Ported as-is.
  2. **Cache-key-invariant values riding as per-core RTAs** — `num_groups`, `num_channels`, `origin_h`,
     `origin_w`, `num_inner_tiles`, `batch`. They are part of the op's cache key, so they cannot differ
     between dispatches sharing a cached program; they are CRTAs in all but name. See the RTA→CRTA note
     above.
  3. **The literal-`0u` absent-optional sentinel.** Where the sibling op passed a null `Buffer*` for an
     absent optional output, this op pushed a literal `0u` into the runtime-arg list
     (`gamma_beta_grad` writer slots 0–1, `input_grad` reader slot 4). Both disappear with the
     `TensorBinding` conversion, so this is now moot — recorded because the two ops disagreeing on the
     sentinel is a sign the pattern was copied rather than designed.

- **Test coverage note.** This op exercises **both** algorithm paths — `Large
  moreh_group_norm_backward_input_grad` is selected 10 times and `Small` 21 times across the confirmed
  set — so all four of its kernel sources and both borrowed compute kernels are covered here. That is
  also why the sibling op's large *compute* path is covered at all: `moreh_layer_norm_backward` never
  selects its own large algorithm in any test. The gap that remains is on that side (its large
  *reader*), and is written up in its report.

- **Per-op carry-over.** `moreh_group_norm` (forward) in the neighbouring directory shares this op's
  reader shapes — the `mean_rstd_tile_idx` / `tilized_..._idx_in_tile` addressing, the mask-CB
  generation, the running `get_arg_val(i++)` block — so the same conversion decisions apply almost
  line for line.

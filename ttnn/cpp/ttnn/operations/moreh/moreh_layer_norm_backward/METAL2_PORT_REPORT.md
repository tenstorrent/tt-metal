# Metal 2.0 Port Report — `moreh_layer_norm_backward`

> Read with `moreh_group_norm_backward`'s report. The two ops were ported as one change: three of this
> op's compute kernels are bound by both, so the atomic units span both ops.

## Outcome

**`PORTED`** — both factories converted, all 13 kernel sources in the bundled set converted, and the
confirmed test set passes with the same result as the pre-port baseline.

- `MorehLayerNormBackwardGammaBetaGradFactory` → `ProgramSpecFactoryConcept` ✓
- `MorehLayerNormBackwardInputGradFactory` → `ProgramSpecFactoryConcept` ✓ (both selectable reader
  sources and both selectable compute sources converted together)

Nothing is left for a later pass on this op.

## Provenance

- **Recipe docs (this port):** `82c2b5569eb 2026-08-13 docs(metal_2.0): stop gating on a pybound create_descriptor, and drop a stale reference port`
- **Audit docs (inherited):** `82c2b5569eb 2026-08-13 docs(metal_2.0): stop gating on a pybound create_descriptor, and drop a stale reference port`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` on both factories, as the audit chose. Neither ported-from factory had an
`override_runtime_arguments`, so nothing routed to the custom concept and no cache-hit override was
written; the framework refreshes tensor bindings on its own.

### Device-op-class edits

- **Pybind entry points removed:** none. `moreh_layer_norm_backward_nanobind.cpp` exposes only the
  user-facing op — `create_descriptor` was never pybound, so the port forced no pybind change.
- **Custom `compute_program_hash`:** none on either device-operation (default reflection hash, no
  `attribute_values` / `to_hash` backdoor). Nothing to leave intact.
- The only device-op-class edits are the two forced signature changes,
  `create_descriptor` → `create_program_artifacts`, in
  `device/moreh_layer_norm_backward_gamma_beta_grad_device_operation.hpp:34` and
  `device/moreh_layer_norm_backward_input_grad_device_operation.hpp:33`, plus swapping the now-unused
  `<tt-metalium/program_descriptors.hpp>` include for `ttnn/metal_v2_artifacts.hpp`. Both
  `*_device_operation.cpp` files are byte-identical to their pre-port revision.

### Open items

- **Relaxation candidates:** none identified. Every `TensorParameter` stays strict.
- **RTA → CRTA candidates.** Several runtime args carry the same value on every node and are really
  common runtime args: `num_outer`, `num_inner`, `mask_h`, `normalized_dims`, `mean_rstd_height`,
  `mean_rstd_width` on the gamma_beta_grad reader; `num_inner`, `n`, `recip_n`, `mask_h`, `mask_w`,
  `normalized_dims`, `mean_rstd_height`, `mean_rstd_width` on the input_grad reader; `Wt` on the
  input_grad writer. Not converted here — RTA→CRTA changes dispatch semantics, which is outside a
  port. Worth a follow-up pass for dispatch efficiency.
- **Name-first `runtime_arg_values` restructure.** Both factories still build run-args from the legacy
  node-first core loop, bridged by `AddRuntimeArgsForNode`. Inverting the loop nesting is a separate
  cleanup; doing it inside the port would have added transposition risk to an already-large rewrite.

## Handoff points

**None.** No capitulation, no boundary-rule assumption violation, no kernel-lib gap, no framework gap,
no removed pybind surface.

Specifically, the two things that would have blocked this port did not occur:
- No `sem::` or `tensor::` handle is required at any out-of-op call site. The only cross-boundary
  calls are to `ttnn/cpp/ttnn/kernel/` helpers (`fill_cb_with_value`, `generate_mask_h`,
  `generate_mask_h_w`, `get_tilized_idx`) and `compute_kernel_lib::reduce`, all of which take either a
  `DataflowBuffer` object or a `uint32_t` buffer id. `dfb::name` bridges the latter through its
  implicit conversion, including in **non-type template parameter** position
  (`compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::ydyadd, dfb::scaler, dfb::dgamma>`).
- No `Case 2` (raw base pointer) binding anywhere, so the compute-kernel `TensorAccessor` block never
  came into play. Every one of the 11 accessor sites is `Case 1` and 2-argument.

## Successes

- **[Pattern: Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  — the parse-time-ternary warning fired exactly as written.**
  The pre-port kernel had
  `constexpr auto cb_out_init = gamma_grad_has_value ? cb_dgamma : cb_dbeta;`; the gate that replaced it
  is at `device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp:73-79`. Both operands resolve at
  parse time regardless of the constant condition, so gating only the *uses* would have failed name
  lookup on `dfb::dgamma` in an absent-gamma build. The pattern says to gate the selection itself,
  which is what the port does — and this really is the first thing that would have broken.
- **The same entry's "promote a CTA gate to a define, and feed it to every kernel that names the
  resource" paragraph.** The legacy factories emit **no** mask define at all, and the shared compute
  kernel derives `do_mask_h` / `do_mask_w` itself from `origin_H` / `origin_W` /
  `is_lastdim_layernorm` / `is_groupnorm`. Reading only the reader's CTA list would have produced a
  define on the reader and a compile failure on the compute kernel.
- **[Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options)
  — "answering *did it set one?* is the part that goes wrong."** `grep -n opt_level` returns nothing in
  either factory, which reads as "nothing to carry over" but actually means the compute kernels were at
  the legacy `ComputeConfig` default of `O3` while Metal 2.0 would silently give them `O2`. All four
  compute `KernelSpec`s (two factories × two core groups) set `O3` explicitly.
- **[Anti-pattern: Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta).**
  Both factories keep two compute `KernelSpec`s over two `WorkUnitSpec`s with the per-group count as a
  CTA. The catalog's note that the disjoint-node split is *not* the same-grid two-toucher case settled
  the endpoint question without reaching for `allow_instance_multi_binding` — which appears nowhere in
  this port.

## Friction

### Gaps

- **[Pattern: Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
  covers the handle but not the object.** The entry says "alias the *handle*, keep *one* object" and
  shows `constexpr auto cb_x = dfb::cb_in;`. But the legacy kernels here construct a **second
  `DataflowBuffer` object** per working name — `DataflowBuffer dfb_xmm_obj(cb_xmm);` where `cb_xmm`
  is `cb_tmp2` — and those object names are load-bearing documentation (`dfb_ndymdysum_obj` says what
  the tile *is* at that point, which `dfb_tmp2_obj` does not). Following the entry literally leaves the
  porter choosing between two bad options: keep the duplicate objects (explicitly forbidden) or delete
  the working names (loses the documentation rule 8 asks you to preserve).
  The resolution used here is a **reference alias beside the handle alias**:
  ```cpp
  constexpr auto dfb_ndymdysum = dfb::tmp2;      // handle alias, for LLK call sites
  auto& dfb_ndymdysum_obj = dfb_tmp2_obj;        // reference alias, for FIFO calls — still ONE object
  ```
  See `device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp:397-399` and its siblings.
  Worth one code block in the catalog entry; it took a while to be sure this was the sanctioned shape
  rather than an improvisation.
- **Nothing warns that binding an optional output through a *converting* reference makes a copy.**
  Both gamma_beta_grad factories held their optional outputs as
  `const std::optional<const Tensor>& gamma_grad = output_tensor.at(0);`. The element is
  `std::optional<Tensor>`, so that line materializes a temporary `std::optional<const Tensor>` — and
  therefore a `Tensor` copy — whose lifetime is extended to the end of scope. Under
  `ProgramDescriptor` that was harmless (only `.buffer()` was read). Under Metal 2.0 it is exactly the
  ["matched back by `MeshTensor` identity, so a copy fails"](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md#extracting-the-tensor)
  footgun, and it is invisible at the call site — the declaration *looks* like a plain reference bind.
  The port binds `std::optional<Tensor>&` directly to the vector element instead. The
  "Extracting the tensor" section covers `ttnn::Tensor` → `MeshTensor`; a sentence about
  *optional-of-const* conversions on the way in would close the gap.
- **The `[Cc][Bb]_` sweep has a third innocent category the recipe doesn't name.** The
  [anti-pattern self-audit](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#anti-pattern-self-audit)
  says to expect **zero** hits, excluding only `cbegin` / `cbrt`. This port's op directories return 14
  hits, all of them `fill_cb_with_value` — a kernel-lib helper in
  `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` that five of the ported kernels call and that is
  outside the porter's writeable surface. Kernel-lib symbol names deserve a mention alongside
  `cbegin`/`cbrt`, otherwise every moreh port re-adjudicates the same 14 lines.

### Confusion

- **Whether the `gamma_grad_has_value` / `beta_grad_has_value` / `gamma_has_value` CTAs should survive
  as *named CTAs* alongside the new defines.** [Whitelist rule 4](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#kernel-side-whitelist)
  says every compile-time arg becomes named, and [rule 6](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#kernel-side-whitelist)
  says a binding-selecting condition moves to a define. These read as independent instructions, and
  emitting both would have been a defensible literal reading. The port drops the CTA and keeps only
  the define, on the grounds that a value that exists *only* to gate a binding has no other reader —
  and every `if (gamma_grad_has_value)` block in these kernels touches a conditional resource. Saying
  "when a CTA's only role is gating a conditional binding, it becomes a define and the CTA is dropped"
  in rule 6 would remove the ambiguity.
- **`ComputeConfigDescriptor{}` vs. the TTNN helper.** The two sibling ops look interchangeable —
  same kernels, same shapes — but layer-norm resolves a TTNN `ComputeKernelConfig` (Style A) while
  group-norm sets an all-default Metal `ComputeConfigDescriptor{}` (Style B). Routing both through
  `ttnn::to_compute_hardware_config` would have compiled, passed every test, and silently flipped
  group-norm's fidelity, approx-mode and dest-buffering. The
  [Compute kernels](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compute-kernels)
  section does state this ("the two config structs default *opposite* ways"), and the briefs flagged
  it per-op — but only because the auditor checked each factory separately. A port that read one
  factory and assumed the other matched would have shipped the regression. No doc change needed; this
  is a note that the per-factory check is the load-bearing step, not the per-op one.

## Open items for downstream

- **Shared kernel touches — rung 3 (converted in place), three kernels, both binders converted.**

  | kernel path | rung taken | remaining unmigrated consumers |
  |---|---|---|
  | `device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` | **in-place**; bundled set = this op's + `moreh_group_norm_backward`'s gamma_beta_grad factories | **none** |
  | `device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp` | **in-place**; bundled set = this op's + `moreh_group_norm_backward`'s input_grad factories | **none** |
  | `device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp` | **in-place**; same bundled set | **none** |

  No `_metal2` fork was created and none is needed: `grep -rl <filename> ttnn/cpp/ttnn/operations/`
  returns exactly two consumers for each file, both converted in this change. There is no sunset
  checklist to leave behind. **Any future op that wants to bind one of these three must adopt its
  binding vocabulary** — `dfb::dy` / `x` / `mean` / `rstd` / `scaler` / `mask_h` / `mask_w` /
  `dgamma` / `dbeta` / `y` / `ydy` / `dyadd` / `ydyadd` / `xmm` / `dycopy` for gamma_beta_grad;
  `dfb::dy` / `x` / `mean` / `rstd` / `scaler` / `n_recip_n` / `gamma` / `mask_h_w` / `dx` / `dycopy` /
  `y` / `dysum` / `ydysum` / `tmp1..3` (+ `recip_nrstd` on the small path) for input_grad; named CTAs
  `num_cols_per_core` or `num_rows_per_core`, `origin_H`, `origin_W`, `NCHt`, `Wt`,
  `is_lastdim_layernorm`, `is_groupnorm`; and defines `GAMMA_GRAD_HAS_VALUE`, `BETA_GRAD_HAS_VALUE`,
  `GAMMA_HAS_VALUE`, `DO_MASK_H`, `DO_MASK_W`.

- **Findings — pre-existing behavior shipped forward unchanged.** None of these was fixed; each is a
  candidate for a separate PR by the op owners.
  1. **`device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp:492-494`** — the kernel
     ends with a second `wait_front(2)` on `mask_h_w` where the large-algorithm kernel has
     `pop_front(2)` at the same point. Waiting for entries that are already available is a no-op, so
     there is no observable effect, but it is almost certainly a typo for `pop_front`. Preserved
     verbatim, with a comment at the site saying so.
  2. **`device/moreh_layer_norm_backward_input_grad_program_factory.cpp:406-409`** — compile-time arg slot
     3 is named `Wt` by the compute kernels but carries `num_inner`, which is the inner-dimension tile
     count and only equals a width-tile count in the last-dim case. The name is now pinned into the
     shared kernels' interface, so renaming it is a coordinated change across both ops. Carried over
     as-is with a comment at the emission site.
  3. **Cache-key-invariant values riding as per-core RTAs** — `normalized_dims`, `mean_rstd_height`,
     `mean_rstd_width` (both factories' readers). They are part of the op's cache key, so they cannot
     differ between dispatches that share a cached program; they are CRTAs in all but name. See the
     RTA→CRTA note under TTNN ProgramFactory → Open items.
  4. **`device/moreh_layer_norm_backward_gamma_beta_grad_program_factory.cpp:63`** — `mean_rstd_shape`
     is computed and never used. Pre-existing; left in place (removing it is not port work).
  5. **The gamma_beta_grad factory allocated `c_16`/`c_17` unconditionally** (`out0_t = out1_t = 1`)
     even when the matching optional output was absent, leaving a buffer with zero touchers in that
     configuration. The port binds them conditionally, so the buffer is simply not declared there —
     zero functional change (a bindingless DFB is rejected by the validator regardless), and the
     sibling op already allocated 0 tiles in that case.

- **Test coverage note — the large-algorithm reader on this op is never exercised.** Across the whole
  confirmed test set, `Large moreh_layer_norm_backward_input_grad algorithm is selected` appears
  **zero** times (`Small` appears 26 times); every shape in the suite fits L1. The large *compute*
  kernels are covered, but only through `moreh_group_norm_backward`, which does select its large path
  (10 times). So `device/kernels/reader_moreh_layer_norm_backward_input_grad_large.cpp` is converted
  but untested, and the large compute kernels are tested only under `is_groupnorm == true`. A shape
  large enough to trip `dfb_usage >= available_L1` would close both gaps and is worth adding — the port
  did not add one, since new tests are outside its scope.

- **Per-op carry-over.** The `moreh_layer_norm` *forward* op in the neighbouring directory has the same
  shape (mask CBs gated on `do_mask_h`/`do_mask_w`, `read_mean_rstd`-style helpers, an
  `is_lastdim_layernorm` CTA) and already emits `DO_MASK_H` / `DO_MASK_W` defines under the legacy API,
  so its conditional-binding work is largely pre-done. It looks like the cheapest next port in this
  family.

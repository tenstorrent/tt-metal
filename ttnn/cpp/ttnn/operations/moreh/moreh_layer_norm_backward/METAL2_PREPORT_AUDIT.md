# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward`

Two independent DeviceOperations share this directory. They do **not** share kernels with each other,
but both own kernel sources that `moreh_group_norm_backward` binds (see Out-of-directory coupling).

- **`MorehLayerNormBackwardGammaBetaGradOperation`**
  - `MorehLayerNormBackwardGammaBetaGradFactory` (`device/moreh_layer_norm_backward_gamma_beta_grad_program_factory.cpp`)
- **`MorehLayerNormBackwardInputGradOperation`**
  - `MorehLayerNormBackwardInputGradFactory` (`device/moreh_layer_norm_backward_input_grad_program_factory.cpp`)

Audited together (one combined report) because they sit in one op directory, are dispatched from one
user-facing entry point (`moreh_layer_norm_backward.cpp`), and are being ported in one change. Findings
are attributed per DeviceOperation wherever they differ.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `82c2b5569eb 2026-08-13 docs(metal_2.0): stop gating on a pybound create_descriptor, and drop a stale reference port`

**Companion audit:** `moreh_group_norm_backward` was audited in the same session and is being ported in
the same PR. The two are coupled through three shared compute kernels; read that report's
Out-of-directory coupling section alongside this one.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehLayerNormBackwardGammaBetaGradOperation` → `MorehLayerNormBackwardGammaBetaGradFactory`; `MorehLayerNormBackwardInputGradOperation` → `MorehLayerNormBackwardInputGradFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 8 own kernels + 3 donor headers clean |
| *Prereqs* — Cross-op escapes | Ok — all donor signatures take `DataflowBuffer` / templated `AddrGen` |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factory rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — not a `WorkloadDescriptor` op |
| *TTNN Readiness* — Custom hash | **No** — default reflection hash (sheet `no` · confirmed by grep) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** (sheet `no` · confirmed by grep) |
| *TTNN Readiness* — `override_runtime_arguments` | **No** (sheet `no` · confirmed by grep) |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `moreh_layer_norm_backward_nanobind.cpp` exposes only the user-facing op |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (both factories) |
| *Port work* — Offset base pointer | none — every address arg is a bare `Buffer*`, no host-folded offset |
| *Port work* — Tensor bindings (per binding) | 5 (gamma_beta_grad) + 6 (input_grad), **all Case 1** |
| *TTNN Readiness* — TensorParameter relaxation | `none` (no `ArgConfig::Runtime*` anywhere) |
| *Port work* — TensorAccessor 3rd arg | none — every site is the 2-arg form |
| *Port work* — CB endpoints | legal 1:1 + **13 self-loops** + **conditional bindings**; no unconditional dead CB |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves to a **self-loop**
(compute-only intermediate, one toucher). No CB in either factory needs the multi-binding advanced
option, and no CB is unconditionally dead. Two CBs are *config-scoped* dead — see
[Conditional bindings](#conditional-bindings-the-central-port-work).

## Result

**GREEN → briefs issued for both factories.**

Both factories clear every gate. The port is mechanical in shape but non-trivial in one dimension: this
op's kernels are riddled with **compile-time-conditional CB usage** (optional gamma / gamma_grad /
beta_grad, optional mask tiles, and a small/large algorithm fork that changes the intermediate CB map).
Metal 2.0 turns each of those into a conditional binding plus a preprocessor gate, and there are enough
of them that the `#ifdef` discipline is the main correctness surface of this port.

**The port is coupled to `moreh_group_norm_backward`** and must be planned jointly — see
[Out-of-directory coupling](#out-of-directory-coupling).

## Consequence: the atomic unit is not one factory

*Added after the invoker confirmed the bundled port (Question 1).*

The recipe's default atomic unit is **one ProgramFactory plus the kernel entry points it binds**, and a
multi-factory op is normally ported one factory at a time. **That default does not hold here.** Because
the three shared compute kernels are converted *in place* rather than forked, each one flips to Metal
2.0 exactly once — and at that moment **every** factory binding it must already speak Metal 2.0
bindings, or the build breaks. The unit therefore grows to *one shared compute kernel + all of its
binders*.

Across the two ops that yields **two** atomic units, not four:

| Unit | Factories that must convert together | Kernel sources in the unit |
|---|---|---|
| **A — gamma_beta_grad** | `MorehLayerNormBackwardGammaBetaGradFactory` **+** `MorehGroupNormBackwardGammaBetaGradFactory` | 5 — the shared gbg compute kernel, LN reader + writer, GN reader + writer |
| **B — input_grad** | `MorehLayerNormBackwardInputGradFactory` **+** `MorehGroupNormBackwardInputGradFactory` | 8 — the shared ig **small** + **large** compute kernels, LN reader small + large + writer, GN reader small + large + writer |

Unit A and Unit B are independent of each other (they share no kernel), so they can be done in either
order, in separate sessions, and each is separately buildable and testable. Within a unit there is no
smaller buildable increment — expect a long stretch with no green build, which is the expected shape of
an atomic multi-source conversion, not a stop signal.

Unit B is the larger of the two by a clear margin (8 sources, two algorithm paths, the same-FIFO
aliasing that differs between small and large). Size is **not** grounds for capitulation; if it overruns
a session's budget, hand the whole unit to a fresh primary instance to continue from
`METAL2_PORT_PLAN.md` — never leave a half-converted unit, which does not build.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The invoker supplied the readiness-sheet
  rows for both factories: `Concept = descriptor`, `Is safe to port = yes`, `Is able to port? = yes`.

  **Cross-check: full sheet rows supplied by the invoker (2026-08-13), every checkable column
  verified against the code. Zero conflicts.** Both factory rows are identical on every column below.

  | Sheet column | Sheet value | Code evidence | Verdict |
  |---|---|---|---|
  | `Concept` | `descriptor` | both factories define `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` — `..._gamma_beta_grad_device_operation.hpp:34`, `..._input_grad_device_operation.hpp:33` | ✓ agrees |
  | `Custom hash (compute_program_hash)` | `no` | no `compute_program_hash` anywhere in the op | ✓ agrees |
  | `Backdoor custom hash (attribute_values / to_hash)` | `no` | no `attribute_values` / `to_hash` override | ✓ agrees |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | no such hook on either device-op | ✓ agrees — **gate conjunct clears** |
  | `Override runtime args method? (PD only)` | `no` | no `override_runtime_arguments` on either factory | ✓ agrees — fixes the target concept as the **base** `ProgramSpecFactoryConcept` |
  | `Pybind descriptor (nb::class_ of device op)` | `no` | `moreh_layer_norm_backward_nanobind.cpp` exposes only the user-facing op | ✓ agrees — no pybind cleanup forced |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` | zero `->address()` / `.address()` sites; every address rides as a `Buffer*` | ✓ agrees |
  | `TensorParameter relaxation` | `none` | no `ArgConfig::Runtime*` in factory or kernels | ✓ agrees — **gate conjunct clears** |
  | `Op-owned tensors?` | *(blank)* | no `WorkloadDescriptor`, no op-allocated device tensors | ✓ consistent |
  | `Secretly SPMD Workload?` | *(blank)* | N/A — only meaningful when `Concept == WorkloadDescriptor` | ✓ consistent |
  | `Known op issues` | *(blank)* | — | nothing to route |
  | `Is able to port?` | **`yes`** | — | **the gate — clears** |
  | `Is safe to port?` | `yes` | *(deliberately not verified — expert-judgment axis)* | recorded |
  | `Porting Target` | `ProgramSpecFactoryConcept` | — | ✓ matches the concept derived from `Override runtime args method? = no` |

  - **Factory-set match:** the sheet lists exactly 2 rows for this op; the code has exactly 2
    factories, one per DeviceOperation. No phantom row, no missing row. ✓
  - **Cross-column invariants:** `get_dynamic_runtime_args = no` (legal on any concept);
    `Op-owned tensors?` blank on a `descriptor` concept (the only legal value — the `descriptor` form
    cannot carry them). ✓ internally consistent.
  - **Two informational columns worth carrying forward.** `Op Classification = "PD Op
    (pointer-patching)"` and `Execution Model = SPMD` corroborate the code independently: the
    factories deliver every tensor address through the `Buffer*`-binding form (which the framework
    patches on program-cache hits), and each builds a single program stamped across the mesh. The
    pointer-patching classification is precisely the shape a Metal 2.0 `TensorBinding` supersedes, so
    it is a statement of what this port *buys*, not a problem. `Pointer patching perf issue? = OK`
    says the sheet owner sees no perf concern in the current form either.
  - **One minor sheet imprecision, not a conflict.** `Factory definition path` and `Declared in` both
    name the `*_device_operation.hpp`, whereas each factory is *defined* in its
    `*_program_factory.cpp`. The rows still identify their factory unambiguously via
    `Device operation` + `Factory (variant)`, and the named header does declare the factory struct,
    so this is a wrong-ish pointer rather than a broken row. Not escalated; noted for the sheet owner.

- **Device 2.0 (every kernel used):** **GREEN.** All 8 kernel sources this op binds are structurally
  Device 2.0 — `DataflowBuffer`, `Noc`, `CoreLocalMem`, `Semaphore`-free. A scan for Device 1.0 idioms
  (`InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`, bare `noc_async_read/write`,
  `noc_semaphore_*`, `get_semaphore(`, `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front`)
  returns **zero hits** across both the op's kernels and the three donor headers.

  No **GATE** idioms: no `get_local_cb_interface`, no `get_cb_tiles_acked_ptr` /
  `get_cb_tiles_received_ptr`, no `read_tile_value` / `get_tile_address`, no `get_pointer_to_cb_data`.

  The only CB-index free functions present are `get_tile_size(cb_id)`, which the Device 2.0 migration
  guide **explicitly sanctions** and which therefore does not knock the op out of Green. *Breadcrumb
  for the porter:* the Metal 2.0 port moves these onto the object (`dfb.get_tile_size()`), per
  kernel-side whitelist rule 7. Sites:

  | File | Lines |
  |---|---|
  | `device/kernels/reader_moreh_layer_norm_backward_gamma_beta_grad.cpp` | 27, 158, 159 |
  | `device/kernels/reader_moreh_layer_norm_backward_input_grad_small.cpp` | 27, 167, 168, 169 |
  | `device/kernels/reader_moreh_layer_norm_backward_input_grad_large.cpp` | 27, 168, 169, 170 |
  | `device/kernels/writer_moreh_layer_norm_backward_gamma_beta_grad.cpp` | 35, 36 |
  | `device/kernels/writer_moreh_layer_norm_backward_input_grad.cpp` | 28 |

- **Feature compatibility:** every Appendix A entry scanned against both factories and all 8 kernels.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer`, no `.global_circular_buffer` field, no `remote_cb_*` / `remote_index` idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | Field never set; no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress` |
  | GlobalSemaphore | N/A | No semaphores of any kind in this op |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Every `get_compile_time_arg_val` uses a **literal** index. The `std::vector<std::optional<Tensor>>` on `tensor_return_value_t` is a fixed-count (2) *return*, not a variable-count input list, so the op-level cue does not fire; the kernel-level decider is clean |

- **CB endpoints (GATE-free):** see [CB endpoint census](#cb-endpoint-census). Every CB is either a
  legal 1:1 cross-kernel FIFO or a compute-only **self-loop**. Nothing here blocks a Gen1 port.

- **Offset base pointers:** **GREEN.** Every address argument in both factories is passed as a bare
  `Buffer*` (`output_grad.buffer()`, `input.buffer()`, `mean.buffer()`, `rstd.buffer()`,
  `gamma.value().buffer()`, `input_grad.buffer()`, `gamma_grad.value().buffer()`,
  `beta_grad.value().buffer()`) — the `Buffer*`-binding form. A repo-wide grep for `->address()` /
  `.address()` inside this op returns **only two comment lines**
  (`..._input_grad_program_factory.cpp:287`, `..._gamma_beta_grad_program_factory.cpp:248`), both
  explaining *why* the factory passes `Buffer*` rather than a raw address. No host arithmetic is folded
  into any address, so there is no Type 1 or Type 2 fold. Type 3 (`address_offset`) is absent; Type 4
  (`narrow`) does not apply.

  Cross-referenced against the dated triage `analyses/2026-07-19_offset_base_pointers.md`: this op is
  **not** in its tables, and the scan above independently confirms clean — the "no fold, not in tables"
  outcome. Not waved through on absence from the table.

- **TensorAccessor 3rd argument:** **GREEN.** All 11 `TensorAccessor(...)` construction sites across the
  op's 8 kernels use the **2-argument** form `TensorAccessor(args, addr)`. No site passes a page size,
  so the subject does not fire and the taxonomy is not engaged. Cross-referenced against
  `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`: this op is absent from the table, consistent
  with the scan.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — **all Case 1** (address flows into a `TensorAccessor`; the kernel
  does all access through the accessor). Both factories deliver the address via the `Buffer*`-binding
  form, which the framework already patches on cache hits, so none of these is the silent-wrong hazard —
  they are routine port work.

  | Factory | Binding | Kernels | Case |
  |---|---|---|---|
  | gamma_beta_grad | `output_grad` | reader | 1 |
  | gamma_beta_grad | `input` | reader | 1 |
  | gamma_beta_grad | `mean` | reader | 1 |
  | gamma_beta_grad | `rstd` | reader | 1 |
  | gamma_beta_grad | `gamma_grad` (optional) | writer | 1 — **conditional** |
  | gamma_beta_grad | `beta_grad` (optional) | writer | 1 — **conditional** |
  | input_grad | `output_grad` | reader (small \| large) | 1 |
  | input_grad | `input` | reader | 1 |
  | input_grad | `mean` | reader | 1 |
  | input_grad | `rstd` | reader | 1 |
  | input_grad | `gamma` (optional) | reader | 1 — **conditional** |
  | input_grad | `input_grad` | writer | 1 |

- **TensorParameter relaxation:** `none`. No `ArgConfig::Runtime*` in either factory or any kernel.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** 13 self-loops total (6 in gamma_beta_grad, 7 in input_grad-small / 6 in
  input_grad-large); all remaining CBs are legal 1:1. No multi-binding flag anywhere. No unconditional
  dead CB; two config-scoped ones handled by conditional binding.

## CB endpoint census

Counted **per CB, per node, per configuration**, over the distinct kernels that touch it. Every toucher
here is *locked* to a FIFO role (real `reserve_back`/`push_back` or `wait_front`/`pop_front`) — there
are no raw-pointer-only touchers in this op, so no role-free relabelling is in play and no CB is a
1P+1C assignment case.

### `MorehLayerNormBackwardGammaBetaGradFactory`

Kernels per node: `reader`, `writer`, `compute` (one of two same-source `KernelSpec`s, per core group).

| CB | Meaning | Producer | Consumer | Touchers | Disposition |
|---|---|---|---|---|---|
| `c_0` | output_grad (dy) | reader | compute | 2 | legal 1:1 |
| `c_1` | input (x) | reader | compute | 2 | legal 1:1 |
| `c_2` | mean | reader (`read_mean_rstd`) | compute | 2 | legal 1:1 |
| `c_3` | rstd | reader (`read_mean_rstd`) | compute | 2 | legal 1:1 |
| `c_4` | scaler | reader (`fill_cb_with_value`) | compute | 2 | legal 1:1 |
| `c_5` | mask_h | reader (`generate_mask_h`) | compute | 2 | legal 1:1 — **conditional on `do_mask_h`** (host allocates 0 tiles otherwise) |
| `c_6` | mask_w | — | — | **0** | **Not allocated by this factory.** The shared compute kernel constructs `DataflowBuffer dfb_mask_w_obj(cb_mask_w)` unconditionally at `moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp:33` but only *uses* it under `do_mask_w`, which is compile-time false here (`is_groupnorm == false`). No DFB to declare; the kernel reference must be `#ifdef`-gated |
| `c_16` | gamma_grad (dgamma) | compute | writer | 2 when `gamma_grad_has_value`, else **0** | legal 1:1 when present; **conditional binding** |
| `c_17` | beta_grad (dbeta) | compute | writer | 2 when `beta_grad_has_value`, else **0** | legal 1:1 when present; **conditional binding** |
| `c_24` | y | compute | compute | 1 | **self-loop** |
| `c_25` | y·dy | compute | compute | 1 | **self-loop** |
| `c_26` | Add[dy] | compute | compute | 1 | **self-loop** |
| `c_27` | Add[y·dy] | compute | compute | 1 | **self-loop** |
| `c_28` | x − mean | compute | compute | 1 | **self-loop** |
| `c_29` | dycopy | compute | compute | 1 | **self-loop** |

### `MorehLayerNormBackwardInputGradFactory`

Kernels per node: `reader` (**small** or **large** source), `writer`, `compute` (**small** or **large**
source, two same-source `KernelSpec`s per core group). Classified per source path, since the
intermediate CB map differs between them.

| CB | Meaning | Producer | Consumer | Touchers | Disposition |
|---|---|---|---|---|---|
| `c_0` | output_grad (dy) | reader | compute | 2 | legal 1:1 |
| `c_1` | input (x) | reader | compute | 2 | legal 1:1 |
| `c_2` | mean | reader | compute | 2 | legal 1:1 |
| `c_3` | rstd | reader | compute | 2 | legal 1:1 |
| `c_4` | scaler | reader | compute | 2 | legal 1:1 |
| `c_5` | n_recip_n (2 entries) | reader (two `fill_cb_with_value`) | compute | 2 | legal 1:1 |
| `c_6` | gamma | reader | compute | 2 when `gamma_has_value`, else **0** | legal 1:1 when present; **conditional binding** |
| `c_7` | mask_h_w (2 entries) | reader (`generate_mask_h_w`) | compute | 2 when `do_mask_h \|\| do_mask_w`, else **0** | legal 1:1 when present; **conditional binding**. ⚠ the **small** compute kernel `wait_front`s it twice and **never `pop_front`s** — deliberate fill-once/read-many reuse, *not* an unbalanced FIFO to "fix" |
| `c_16` | input_grad (dx) | compute | writer | 2 | legal 1:1 |
| `c_24` | dycopy | compute | compute | 1 | **self-loop** |
| `c_25` | y | compute | compute | 1 | **self-loop** |
| `c_26` | Sum[dy] | compute (via `compute_kernel_lib::reduce`) | compute | 1 | **self-loop** |
| `c_27` | Sum[y·dy] | compute (via `compute_kernel_lib::reduce`) | compute | 1 | **self-loop** |
| `c_28` | small: recip_nrstd · large: tmp1 | compute | compute | 1 | **self-loop** |
| `c_29` | small: tmp1 · large: tmp2 | compute | compute | 1 | **self-loop** |
| `c_30` | small: tmp2 · large: tmp3 | compute | compute | 1 | **self-loop** |
| `c_31` | small: tmp3 | compute | compute | 1 (small only) | **self-loop**. Large sets `im7_t = 0` so the CB is not allocated, and the large kernel declares no `c_31` — consistent, not a dead CB |

**No dead CB, no multi-binding, no 1P+1C assignment case anywhere in this op.**

## Conditional bindings: the central port work

This is the finding the porter most needs. Six CBs and three tensors are bound only on some
compile-time path, and in **every** case the current kernel constructs the object (or names the id)
*unconditionally* while guarding only its use. That works today because a CB index is just a number.
It does **not** work in Metal 2.0: `dfb::<name>` / `tensor::<name>` are generated per-binding, so an
unbound resource has no token and the unconditional reference fails name lookup at parse time.

Each of these needs the conditional-binding treatment — host binds conditionally, host emits a matching
`KernelSpec::compiler_options.defines` entry, kernel `#ifdef`-gates both the alias and every expression
naming it:

| Resource | Condition | Kernel sites that reference it unconditionally today |
|---|---|---|
| `c_5` mask_h DFB | `do_mask_h` | `reader_..._gamma_beta_grad.cpp:146`; `..._gamma_beta_grad_kernel.cpp:31` |
| `c_6` mask_w DFB | never true in this op | `..._gamma_beta_grad_kernel.cpp:33` |
| `c_16` dgamma DFB | `gamma_grad_has_value` | `..._gamma_beta_grad_kernel.cpp:37`, **:73** (ternary); `writer_..._gamma_beta_grad.cpp:33` |
| `c_17` dbeta DFB | `beta_grad_has_value` | `..._gamma_beta_grad_kernel.cpp:40`, **:73** (ternary); `writer_..._gamma_beta_grad.cpp:34` |
| `c_6` gamma DFB | `gamma_has_value` | `reader_..._input_grad_{small,large}.cpp:~166`; `..._input_grad_{small,large}_kernel.cpp:37` |
| `c_7` mask_h_w DFB | `do_mask_h \|\| do_mask_w` | `..._input_grad_{small,large}_kernel.cpp:39` |
| `tensor::gamma_grad` | `gamma_grad_has_value` | `writer_..._gamma_beta_grad.cpp:25` |
| `tensor::beta_grad` | `beta_grad_has_value` | `writer_..._gamma_beta_grad.cpp:26` |
| `tensor::gamma` | `gamma_has_value` | `reader_..._input_grad_{small,large}.cpp:~139` |

Three of these deserve individual attention:

1. **The `cb_out_init` ternary** (`..._gamma_beta_grad_kernel.cpp:73`):
   ```cpp
   constexpr auto cb_out_init = gamma_grad_has_value ? cb_dgamma : cb_dbeta;
   ```
   Both operands resolve at parse time regardless of which branch the constant condition selects, so
   `#ifdef`-gating only the *uses* is not enough — the ternary itself must be gated. This is the
   path-dependent variant the patterns catalog calls out; it is the single most likely compile break in
   this port.

2. **The optional output tensors are the mandatory-`#ifdef` case.** When `gamma_grad` is absent the
   factory today passes `nullptr` to `TensorAccessorArgs` and `Buffer*`-null as the RTA, and the writer
   constructs an accessor over it that it never uses. Post-port there is *nothing to bind* — no
   `TensorParameter`, no token — so an always-bind fallback does not exist. Gate it.

3. **The gate condition must be promoted from a CTA to a define.** Today `do_mask_h` / `do_mask_w`
   inside the compute kernel are `constexpr` values *computed from other CTAs*
   (`(origin_H % TILE_H) != 0 && (is_lastdim_layernorm || is_groupnorm)`), not host-supplied flags. The
   host must compute the same predicate and emit a matching define. I verified the host and kernel
   predicates agree in every configuration of this op:
   - gamma_beta_grad: host `do_mask_h = (origin_H % TILE_HEIGHT) != 0 && is_lastdim_layer_norm`;
     kernel `(origin_H % TILE_H) != 0 && (is_lastdim_layernorm || is_groupnorm)` with
     `is_groupnorm == false`. Equivalent. ✓
   - gamma_beta_grad `do_mask_w`: kernel-side `… && is_groupnorm` ⇒ always false here. ✓
   - input_grad: host `do_mask_h = (origin_H % TILE_HEIGHT) != 0 && !is_lastdim_layer_norm`,
     `do_mask_w = (origin_W % TILE_WIDTH) != 0`; kernel identical. ✓

   **Emission target matters:** the promoted define must reach *every* kernel that references the
   conditionally-bound name — for mask_h that is both the reader and the compute kernel, and the legacy
   factory sends `FP32_DEST_ACC_EN` to the reader but sends no mask define at all today.

## Same-FIFO aliasing in the input_grad compute kernels

Both input_grad compute kernels alias one CB under several names via block-scope `constexpr`:

| Kernel | Aliases |
|---|---|
| `..._input_grad_small_kernel.cpp` | `cb_xmm = cb_tmp2` (L113), `cb_dyadd = cb_tmp1` (L256), `cb_ydy = cb_tmp2` (L298), `cb_ydyadd = cb_tmp3` (L300), `cb_ndy = cb_tmp1` (L368), `cb_ndymdysum = cb_tmp2` (L385), `cb_yydysum = cb_tmp3` (L409) |
| `..._input_grad_large_kernel.cpp` | `cb_dyadd = cb_tmp1` (L87), `cb_ydyadd = cb_tmp2` (L89), `cb_xmm = cb_tmp3` (L94), `cb_ydy = cb_tmp3` (L268), `cb_recip_nrstd = cb_tmp3` (L335), `cb_ndy = cb_tmp1` (L443), `cb_ndymdysum = cb_tmp2` (L462), `cb_xmm = cb_tmp1` (L486), `cb_yydysum = cb_tmp1` (L548), `cb_tmp4 = cb_y` (L572) |

This is **Same-FIFO aliasing**, not `alias_with`: one buffer, several kernel-side names, shared FIFO
pointers. One `DataflowBufferSpec`, one `DFBBinding`, and the second name becomes a handle alias
(`constexpr auto cb_xmm = dfb::tmp2;`) with a **single** `DataflowBuffer` object per buffer. Modelling
any of these with `advanced_options.alias_with` would be a correctness bug.

Note `cb_recip_nrstd` is a **real distinct CB (`c_28`) in the small kernel** but an **alias of `cb_tmp3`
in the large kernel** — the name→buffer mapping genuinely differs per source path, so the DFB map must
be derived per path rather than once for the factory.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No hidden second writer, no multi-reader, no
  dual-instance work-split — every CB has at most two touchers and each is locked to one FIFO role.
- **Cross-op / shared kernels:** three of this op's own compute kernels are *lent* to
  `moreh_group_norm_backward`. See below.
- **RTA varargs:** none. Every kernel reads its runtime args as a block of distinct fields at the top of
  `kernel_main`, at literal indices; no loop-indexed reads, no data-selected indices, no sentinels. All
  RTAs become **named** args.

## Out-of-directory coupling

### Function-call escape — `✓ clean`

Every `#include` outside this op's directory resolves to a donor whose signatures are already Device 2.0
native, so the Metal 2.0 named handles cross without a bridge.

| Op kernel(s) | Donor file | Class | Shapes used | Status |
|---|---|---|---|---|
| all 5 dataflow kernels | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | shared-kernel pool (singular `kernel/`) | `fill_cb_with_value(DataflowBuffer, …)`, `generate_mask_h(DataflowBuffer, …)`, `generate_mask_h_w(DataflowBuffer, …)`, `read_tile/read_value/read_line(DataflowBuffer, AddrGen, …)`, `get_tilized_idx(...)` | ✓ |
| all 3 compute kernels | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | shared-kernel pool | `*_init_with_dt(DataflowBuffer…)`, `pack_tile_with_dt(uint32_t, DataflowBuffer)`, `copy_tile_to_cb(DataflowBuffer, DataflowBuffer, …)` and peers | ✓ |
| all 3 compute kernels | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | official kernel-lib | `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_in, cb_scaler, cb_out>(…)` — CB ids as **non-type template parameters** | ✓ — `DFBAccessor::operator uint32_t()` is `constexpr`, so `dfb::name` is valid in template-argument position |
| all | `tt_metal/hw/inc/api/…` (`noc.h`, `dataflow_buffer.h`, `core_local_mem.h`, `noc_traits.h`, `dataflow_api.h`) | LLK / HAL | — | ✓ no concern |

No donor takes `CircularBuffer&`, a `uint32_t sem_id`, a `TensorAccessorArgs<N>`, an NTTP CTA offset, or
an old-style addr-gen. There is no Shape-4 (pre-Device-2.0) donor, so the donor-side Device 2.0 gate has
nothing to flag.

### Borrowed kernel files (file-path instantiation)

This op **borrows nothing** — it owns all 8 kernel sources it binds.

### Lent kernel files — the coupling that matters

Three kernel sources in this op's directory are bound by `moreh_group_norm_backward`'s factories:

| Kernel file (owned here) | Also bound by | `_metal2` fork exists? |
|---|---|---|
| `device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` | `moreh_group_norm_backward` gamma_beta_grad factory | **No** |
| `device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp` | `moreh_group_norm_backward` input_grad factory | **No** |
| `device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp` | `moreh_group_norm_backward` input_grad factory | **No** |

**Census result:** `grep -rl <filename> ttnn/cpp/ttnn/operations/` returns **exactly two consumers each**
— this op and `moreh_group_norm_backward` — with no build-file, private-copy, or comment false
positives, and no `_metal2` sibling anywhere. **These two ops are the complete consumer set.**

That makes the shared-kernel disposition unusually clean, *given the invoker's stated plan to port both
ops in one branch and PR*: the bundled-port rung applies, and the three kernels convert in place rather
than being forked. That authorization comes from the invoker's assignment, not from this consumer list —
the list is recorded so the assignment can be checked against it, and it happens to match exactly.

**If the plan changes and only one op is ported**, the disposition flips to "create the fork beside the
original" for all three files, and the porter must re-read the shared-kernel Caution before touching
them.

**Binding-name guidance (either way):** name the bindings for the *kernel's* vocabulary, not for either
op's factory locals. The kernels already have the words — `dfb::dy`, `dfb::x`, `dfb::mean`, `dfb::rstd`,
`dfb::scaler`, `dfb::mask_h`, `dfb::mask_w`, `dfb::dgamma`, `dfb::dbeta`, `dfb::gamma`,
`dfb::mask_h_w`, `dfb::dx`, `dfb::dycopy`, `dfb::y`, `dfb::tmp1..3`. Note the two factories disagree on
one name: layer-norm's `c_4` is *scaler*, group-norm's `c_4` is *one*. The kernel calls it `cb_scaler`,
so **`scaler` wins**.

## Team-only

- **Relaxation candidates:** none observed. No custom hash exists to mine, and no kernel reads a
  dimension in a way that would tolerate a relaxed `TensorSpec` match.
- **TTNN factory analysis:** both factories are plain `descriptor` ops with no op-owned tensors, no
  `MeshWorkload`, no pybound `create_descriptor`, no custom hash, and no `override_runtime_arguments`.
  Target concept `ProgramSpecFactoryConcept` for both. The only device-op-class edit the port forces is
  the `create_descriptor` → `create_program_artifacts` signature change in the two `.hpp` files — not a
  sanctioned-exception case, just the factory's own declaration.
- **Test coverage** (for the porter's baseline confirmation, not a gate): nightly pytests only —
  `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py`, which carries
  `test_moreh_layer_norm_backward`, `test_moreh_layer_norm_backward_with_gamma_or_beta`,
  `test_moreh_layer_norm_backward_compute_kernel_options`, `test_moreh_layer_norm_backward_callback`
  (exercises the program-cache hit path), and two `_rejects_*` validation tests. No C++ gtests.
  `test_moreh_layer_norm_backward_with_gamma_or_beta` is the one that reaches the
  gamma-absent / beta-absent configurations, i.e. the conditional-binding paths — it is the
  load-bearing test for this port.

## Misc anomalies  *(team-only, non-gating)*

- **CTA slot 3 of both input_grad compute kernels is named `Wt` but is not a tile width.** The kernels
  read `constexpr uint32_t Wt = get_compile_time_arg_val(3)`
  (`..._input_grad_small_kernel.cpp:17`, `..._input_grad_large_kernel.cpp:17`) while this op's factory
  passes `num_inner` (`..._input_grad_program_factory.cpp:244`) and `moreh_group_norm_backward` passes
  `num_inner_tiles`. The name is misleading in both binders. The port names CTAs, so it will have to
  pick one — recommend naming it for what the kernel does with it, and noting the divergence rather
  than silently renaming the kernel local.
- **`normalized_dims` is passed as an RTA but is a program-cache key attribute.** Both factories send
  `operation_attributes.normalized_dims` per core as a runtime arg
  (`..._gamma_beta_grad_program_factory.cpp:276`, `..._input_grad_program_factory.cpp:322`) even though
  it cannot vary across a cached program. Same for `mean_rstd_height` / `mean_rstd_width`. These are
  RTA-shaped but effectively immutable; a later cleanup pass could demote them to CTAs or at least
  CRTAs. **Not port work** — an RTA→CRTA change alters dispatch semantics and is out of scope.
- **`log_info` on every factory invocation.** Both input_grad factories log
  `"Large/Small … algorithm is selected."` at `log_info` level (`..._input_grad_program_factory.cpp:131`
  and `:136`) on every cache miss, using `tt::LogTest` as the log module from production code.
  Cosmetic; ops-team call.

## Per-DeviceOperation attribution

| Field | GammaBetaGrad | InputGrad |
|---|---|---|
| Overall | GREEN | GREEN |
| Current / target concept | `descriptor` → `ProgramSpecFactoryConcept` | `descriptor` → `ProgramSpecFactoryConcept` |
| Kernel sources bound | 3 (reader, writer, compute) | **5** (reader ×2, writer, compute ×2) — runtime-selected |
| Tensor bindings | 6 (2 conditional) | 6 (1 conditional) |
| CBs | 14 declared, 6 self-loop | 17 (small) / 16 (large), 7 / 6 self-loop |
| Conditional DFBs | `mask_h`, `dgamma`, `dbeta` (+ `mask_w` never bound) | `gamma`, `mask_h_w` |
| Lends kernels to group_norm_backward | 1 (compute) | 2 (compute small + large) |

## Questions for the user

1. ~~**Bundled-port authorization.**~~ **RESOLVED (invoker, 2026-08-13):** bundled port confirmed —
   both ops in one branch and PR, the three shared compute kernels converted **in place**, no forks.
   The census matches the assigned set exactly (two consumers, both assigned). See
   [Consequence: the atomic unit is not one factory](#consequence-the-atomic-unit-is-not-one-factory).

2. ~~**Readiness-sheet columns.**~~ **RESOLVED (invoker, 2026-08-13):** full rows supplied. Every
   checkable column cross-checked against the code with **zero conflicts** — see the cross-check table
   under [Gate detail](#gate-detail). The gate now rests on a real cross-check rather than a
   code-only derivation. Two non-blocking observations were raised for the sheet owner: the
   `Factory definition path` / `Declared in` columns both point at the `*_device_operation.hpp`
   rather than the factory `.cpp`, and the sheet has no column that can express this op's kernel
   sharing.

## Recipe notes

- The audit's **Red outcome scoping rule** and the finding-role table are written for a single op. This
  session audited two ops that share kernels, where one op's *lent-kernel* finding is the other's
  *borrowed-kernel* finding and neither report is complete alone. The recipe's guidance on bundling
  ("multiple device-operations in one op directory") covers the intra-directory case but not this
  cross-directory one; a sentence on cross-op kernel coupling — "audit them together, cross-reference
  both reports" — would have saved a judgement call.
- **`get_tile_size(cb_id)` is sanctioned by the Device 2.0 gate but must be rewritten by the Metal 2.0
  port** (whitelist rule 7). The audit recipe's Green bullet says exactly this, including the breadcrumb —
  it worked as written. Noting it only because the two rules pointing opposite directions at the same
  call is the kind of thing a hurried auditor flags as a violation.
- The **conditional-binding** surface of this op is much larger than the recipe's examples suggest (nine
  resources across five kernels, including a parse-time ternary and three optional tensors). The
  patterns-catalog entry covers each shape correctly, but a porter meeting them one at a time may not
  realise they form the bulk of the port. Consider a note that ops with optional inputs/outputs should
  expect the conditional-binding pattern to dominate.

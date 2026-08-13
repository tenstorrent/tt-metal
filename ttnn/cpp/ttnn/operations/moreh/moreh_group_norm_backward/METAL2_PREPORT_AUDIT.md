# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward`

Two independent DeviceOperations share this directory, one per `device/` subdirectory. Neither owns a
compute kernel: **all three compute kernels this op runs are borrowed from
`moreh_layer_norm_backward`.**

- **`MorehGroupNormBackwardGammaBetaGradOperation`**
  - `MorehGroupNormBackwardGammaBetaGradFactory` (`device/gamma_beta_grad/moreh_group_norm_backward_gamma_beta_grad_factory.cpp`)
- **`MorehGroupNormBackwardInputGradOperation`**
  - `MorehGroupNormBackwardInputGradFactory` (`device/input_grad/moreh_group_norm_backward_input_grad_factory.cpp`)

Audited together (one combined report) — one op directory, one user-facing entry point
(`moreh_group_norm_backward.cpp`), one change. Findings are attributed per DeviceOperation wherever
they differ.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `82c2b5569eb 2026-08-13 docs(metal_2.0): stop gating on a pybound create_descriptor, and drop a stale reference port`

**Companion audit:** `moreh_layer_norm_backward` was audited in the same session and is being ported in
the same PR. It **owns** the three compute kernels this op binds. Its report carries the compute
kernels' internal analysis (self-loop intermediates, same-FIFO aliasing, the `cb_out_init` ternary);
read the two together.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehGroupNormBackwardGammaBetaGradOperation` → `MorehGroupNormBackwardGammaBetaGradFactory`; `MorehGroupNormBackwardInputGradOperation` → `MorehGroupNormBackwardInputGradFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — 5 own dataflow kernels + 3 borrowed compute kernels + 3 donor headers, all clean |
| *Prereqs* — Cross-op escapes | Ok — all donor signatures take `DataflowBuffer` / templated `AddrGen` |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factory rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — not a `WorkloadDescriptor` op |
| *TTNN Readiness* — Custom hash | **No** — default reflection hash (sheet `no` · confirmed by grep) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** (sheet `no` · confirmed by grep) |
| *TTNN Readiness* — `override_runtime_arguments` | **No** (sheet `no` · confirmed by grep) |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `moreh_group_norm_backward_nanobind.cpp` exposes only the user-facing op |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (both factories) |
| *Port work* — Offset base pointer | none — every address arg is a bare `Buffer*` or a literal `0u` |
| *Port work* — Tensor bindings (per binding) | 6 (gamma_beta_grad) + 6 (input_grad), **all Case 1** |
| *TTNN Readiness* — TensorParameter relaxation | `none` (no `ArgConfig::Runtime*` anywhere) |
| *Port work* — TensorAccessor 3rd arg | none — every site is the 2-arg form |
| *Port work* — CB endpoints | legal 1:1 + **13 self-loops** + **conditional bindings**; no dead CB |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves to a **self-loop**
(compute-only intermediate, one toucher). No CB needs the multi-binding advanced option, and no CB is
dead in any configuration.

## Result

**GREEN → briefs issued for both factories.**

Both factories clear every gate. The distinguishing feature of this op is that it is a **borrower**: it
owns five dataflow kernels and no compute kernels, running `moreh_layer_norm_backward`'s three compute
kernels with its own CB layout and its own CTA values. That makes the port inseparable from
`moreh_layer_norm_backward`'s — see [Out-of-directory coupling](#out-of-directory-coupling).

Beyond that the shape matches its sibling: heavy compile-time-conditional CB usage (optional gamma /
gamma_grad / beta_grad, optional mask tiles, small/large algorithm fork), which Metal 2.0 turns into
conditional bindings plus preprocessor gates.

## Consequence: the atomic unit is not one factory

*Added after the invoker confirmed the bundled port (Question 1).*

The recipe's default atomic unit is **one ProgramFactory plus the kernel entry points it binds**, and a
multi-factory op is normally ported one factory at a time. **That default does not hold here.** Because
the three borrowed compute kernels are converted *in place* rather than forked, each one flips to Metal
2.0 exactly once — and at that moment **every** factory binding it must already speak Metal 2.0
bindings, or the build breaks. The unit therefore grows to *one shared compute kernel + all of its
binders*, and every one of this op's units spans both ops.

Across the two ops that yields **two** atomic units, not four:

| Unit | Factories that must convert together | Kernel sources in the unit |
|---|---|---|
| **A — gamma_beta_grad** | `MorehGroupNormBackwardGammaBetaGradFactory` **+** `MorehLayerNormBackwardGammaBetaGradFactory` | 5 — the shared gbg compute kernel (owned by LN), GN reader + writer, LN reader + writer |
| **B — input_grad** | `MorehGroupNormBackwardInputGradFactory` **+** `MorehLayerNormBackwardInputGradFactory` | 8 — the shared ig **small** + **large** compute kernels (owned by LN), GN reader small + large + writer, LN reader small + large + writer |

Unit A and Unit B are independent of each other (they share no kernel), so they can be done in either
order, in separate sessions, and each is separately buildable and testable. Within a unit there is no
smaller buildable increment — expect a long stretch with no green build, which is the expected shape of
an atomic multi-source conversion, not a stop signal.

**This op cannot be ported on its own at all** under the confirmed disposition: it owns no compute
kernel, so every unit it participates in necessarily includes a `moreh_layer_norm_backward` factory.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The invoker supplied the readiness-sheet
  rows for both factories: `Concept = descriptor`, `Is safe to port = yes`, `Is able to port? = yes`.

  **Cross-check: full sheet rows supplied by the invoker (2026-08-13), every checkable column
  verified against the code. Zero conflicts.** Both factory rows are identical on every column below.

  | Sheet column | Sheet value | Code evidence | Verdict |
  |---|---|---|---|
  | `Concept` | `descriptor` | both factories define `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` — `gamma_beta_grad/..._device_operation.hpp:39`, `input_grad/..._device_operation.hpp:32` | ✓ agrees |
  | `Custom hash (compute_program_hash)` | `no` | no `compute_program_hash` anywhere in the op | ✓ agrees |
  | `Backdoor custom hash (attribute_values / to_hash)` | `no` | no `attribute_values` / `to_hash` override | ✓ agrees |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | no such hook on either device-op | ✓ agrees — **gate conjunct clears** |
  | `Override runtime args method? (PD only)` | `no` | no `override_runtime_arguments` on either factory | ✓ agrees — fixes the target concept as the **base** `ProgramSpecFactoryConcept` |
  | `Pybind descriptor (nb::class_ of device op)` | `no` | `moreh_group_norm_backward_nanobind.cpp` exposes only the user-facing op | ✓ agrees — no pybind cleanup forced |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` | zero `->address()` / `.address()` sites; every address rides as a `Buffer*`, absent optionals as a literal `0u` | ✓ agrees |
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
    name the `*_device_operation.hpp`, whereas each factory is *defined* in its `*_factory.cpp`. The
    rows still identify their factory unambiguously via `Device operation` + `Factory (variant)`, and
    the named header does declare the factory struct, so this is a wrong-ish pointer rather than a
    broken row. Not escalated; noted for the sheet owner.
  - **The sheet is silent on the kernel sharing.** It has no column that would express "this factory
    binds a compute kernel owned by another op," so the borrow that dominates this port
    ([Out-of-directory coupling](#out-of-directory-coupling)) is invisible in the readiness data. Not
    a defect — just a reminder that a `yes` here is a TTNN-shape verdict, not a statement that the
    port is independent.

- **Device 2.0 (every kernel used):** **GREEN**, and this gate is location-independent, so it covers the
  three **borrowed** compute kernels as well as the five this op owns. A scan for Device 1.0 idioms
  (`InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`, bare `noc_async_read/write`,
  `noc_semaphore_*`, `get_semaphore(`, `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front`)
  returns **zero hits** across all eight kernels and the three donor headers.

  No **GATE** idioms: no `get_local_cb_interface`, no `get_cb_tiles_acked_ptr` /
  `get_cb_tiles_received_ptr`, no `read_tile_value` / `get_tile_address`, no `get_pointer_to_cb_data`.

  The only CB-index free functions present are `get_tile_size(cb_id)`, which the Device 2.0 migration
  guide **explicitly sanctions**. *Breadcrumb:* the Metal 2.0 port moves these onto the object
  (whitelist rule 7). Sites in this op's own kernels:

  | File | Lines |
  |---|---|
  | `device/gamma_beta_grad/kernels/dataflow/reader_...gamma_beta_grad.cpp` | 99, 100, 101, 102 |
  | `device/gamma_beta_grad/kernels/dataflow/writer_...gamma_beta_grad.cpp` | 33, 37 |
  | `device/input_grad/kernels/dataflow/reader_...input_grad_small.cpp` | 78, 88, 108, 109, 110, 111 |
  | `device/input_grad/kernels/dataflow/reader_...input_grad_large.cpp` | 78, 88, 108, 109, 110, 111 |
  | `device/input_grad/kernels/dataflow/writer_...input_grad.cpp` | 30 |

- **Feature compatibility:** every Appendix A entry scanned against both factories and all eight kernels.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer`, no `.global_circular_buffer` field, no `remote_cb_*` / `remote_index` idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | Field never set; no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress` |
  | GlobalSemaphore | N/A | No semaphores of any kind in this op |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Every `get_compile_time_arg_val` uses a **literal** index. The `std::vector<std::optional<Tensor>>` on gamma_beta_grad's `tensor_return_value_t` is a fixed-count (2) *return*, not a variable-count input list |

- **CB endpoints (GATE-free):** see [CB endpoint census](#cb-endpoint-census). Every CB is either a
  legal 1:1 cross-kernel FIFO or a compute-only **self-loop**. Nothing here blocks a Gen1 port.

- **Offset base pointers:** **GREEN.** Every address argument in both factories is passed as a bare
  `Buffer*` (`output_grad.buffer()`, `input.buffer()`, `mean.buffer()`, `rstd.buffer()`,
  `gamma.value().buffer()`, `input_grad.buffer()`, `gamma_grad.value().buffer()`,
  `beta_grad.value().buffer()`), or as a literal `0u` where the optional is absent
  (`..._gamma_beta_grad_factory.cpp:262,267`; `..._input_grad_factory.cpp:272`). A grep for
  `->address()` / `.address()` inside this op returns **zero hits**, so no host arithmetic is folded
  into any address anywhere: no Type 1, no Type 2. Type 3 (`address_offset`) is absent; Type 4
  (`narrow`) does not apply.

  Cross-referenced against the dated triage `analyses/2026-07-19_offset_base_pointers.md`: this op is
  **not** in its tables, and the scan above independently confirms clean — the "no fold, not in tables"
  outcome. Not waved through on absence from the table.

  *Note for the porter, not a gate:* the literal-`0u` slot is a small asymmetry with the sibling op,
  which passes a null `Buffer*` instead. Both mean "absent"; both become "no binding" post-port.

- **TensorAccessor 3rd argument:** **GREEN.** All 11 `TensorAccessor(...)` construction sites across the
  op's five dataflow kernels use the **2-argument** form `TensorAccessor(args, addr)`. No site passes a
  page size. Cross-referenced against `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`: this op
  is absent from the table, consistent with the scan.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — **all Case 1** (address flows into a `TensorAccessor`; the kernel
  does all access through the accessor).

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
  input_grad-large); all remaining CBs are legal 1:1. No multi-binding flag anywhere. No dead CB.

## CB endpoint census

Counted **per CB, per node, per configuration**. Every toucher is *locked* to a FIFO role — there are no
raw-pointer-only touchers, so no role-free relabelling is in play and no CB is a 1P+1C assignment case.
(The `get_write_ptr()` calls in this op's readers sit **between** a `reserve_back` and a `push_back` on
the same kernel's own producer binding — ordinary FIFO usage, a public peek, not a separate endpoint.)

### `MorehGroupNormBackwardGammaBetaGradFactory`

Kernels per node: `reader`, `writer`, `compute` (borrowed
`moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp`, two same-source `KernelSpec`s per core group).

| CB | Meaning | Producer | Consumer | Touchers | Disposition |
|---|---|---|---|---|---|
| `c_0` | output_grad (dy) | reader | compute | 2 | legal 1:1 |
| `c_1` | input (x) | reader | compute | 2 | legal 1:1 |
| `c_2` | mean | reader | compute | 2 | legal 1:1 |
| `c_3` | rstd | reader | compute | 2 | legal 1:1 |
| `c_4` | **one** (the reduce scaler; kernel calls it `cb_scaler`) | reader (`fill_cb_with_value`) | compute | 2 | legal 1:1 |
| `c_5` | mask_h | reader (`generate_mask_h`) | compute | 2 | legal 1:1 — **conditional on `do_mask_h`** |
| `c_6` | mask_w | reader (`generate_mask_w`) | compute | 2 | legal 1:1 — **conditional on `do_mask_w`**. This is the CB the sibling op never allocates |
| `c_16` | gamma_grad (dgamma) | compute | writer | 2 | legal 1:1 — **conditional on `gamma_grad_has_value`** (host already allocates 0 tiles otherwise) |
| `c_17` | beta_grad (dbeta) | compute | writer | 2 | legal 1:1 — **conditional on `beta_grad_has_value`** |
| `c_24` | y | compute | compute | 1 | **self-loop** |
| `c_25` | y·dy | compute | compute | 1 | **self-loop** |
| `c_26` | Add[dy] | compute | compute | 1 | **self-loop** |
| `c_27` | Add[y·dy] | compute | compute | 1 | **self-loop** |
| `c_28` | x − mean | compute | compute | 1 | **self-loop** |
| `c_29` | dycopy | compute | compute | 1 | **self-loop** |

### `MorehGroupNormBackwardInputGradFactory`

Kernels per node: `reader` (**small** or **large**, owned here), `writer` (owned here), `compute`
(**small** or **large**, borrowed from `moreh_layer_norm_backward`, two same-source `KernelSpec`s per
core group). Classified per source path.

| CB | Meaning | Producer | Consumer | Touchers | Disposition |
|---|---|---|---|---|---|
| `c_0` | output_grad (dy) | reader | compute | 2 | legal 1:1 |
| `c_1` | input (x) | reader | compute | 2 | legal 1:1 |
| `c_2` | mean | reader | compute | 2 | legal 1:1 |
| `c_3` | rstd | reader | compute | 2 | legal 1:1 |
| `c_4` | one / scaler | reader | compute | 2 | legal 1:1 |
| `c_5` | n_recip_n (2 entries) | reader (two `fill_cb_with_value`) | compute | 2 | legal 1:1 |
| `c_6` | gamma | reader | compute | 2 when `gamma_has_value`, else **0** | legal 1:1 when present; **conditional binding** |
| `c_7` | mask_h_w (2 entries) | reader (`generate_mask_h_w`) | compute | 2 when `do_mask_h \|\| do_mask_w`, else **0** | legal 1:1 when present; **conditional binding**. ⚠ the **small** compute kernel `wait_front`s it and **never `pop_front`s** — deliberate fill-once/read-many reuse, *not* an unbalanced FIFO to "fix" |
| `c_16` | input_grad (dx) | compute | writer | 2 | legal 1:1 |
| `c_24`–`c_31` | intermediates (`dycopy`, `y`, `Sum[dy]`, `Sum[y·dy]`, `tmp1..3`, and `recip_nrstd` in small) | compute | compute | 1 each | **self-loop** — 8 in small, 7 in large (`c_31` not allocated when `im7_t = 0`) |

**No dead CB, no multi-binding, no 1P+1C assignment case anywhere in this op.**

## Conditional bindings

Same shape as the sibling op: resources bound only on some compile-time path, referenced
*unconditionally* by the kernel today. Each needs conditional host binding + a matching
`compiler_options.defines` flag + kernel-side `#ifdef` gating of the alias and every expression naming
it.

| Resource | Condition | Kernel sites that reference it unconditionally today |
|---|---|---|
| `c_5` mask_h DFB | `do_mask_h` | reader gbg (guarded block, but the id is declared unconditionally); `moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp:31` |
| `c_6` mask_w DFB | `do_mask_w` | reader gbg; `moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp:33` |
| `c_16` dgamma DFB | `gamma_grad_has_value` | `..._gamma_beta_grad_kernel.cpp:37`, **:73** (ternary); writer gbg |
| `c_17` dbeta DFB | `beta_grad_has_value` | `..._gamma_beta_grad_kernel.cpp:40`, **:73** (ternary); writer gbg |
| `c_6` gamma DFB | `gamma_has_value` | reader ig small+large; `..._input_grad_{small,large}_kernel.cpp:37` |
| `c_7` mask_h_w DFB | `do_mask_h \|\| do_mask_w` | reader ig small+large; `..._input_grad_{small,large}_kernel.cpp:39` |
| `tensor::gamma_grad` | `gamma_grad_has_value` | writer gbg:34 |
| `tensor::beta_grad` | `beta_grad_has_value` | writer gbg:38 |
| `tensor::gamma` | `gamma_has_value` | reader ig small+large:95 |

Two points specific to this op:

1. **The `cb_out_init` ternary in the borrowed compute kernel**
   (`moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp:73`) resolves both operands at parse time.
   Because *this* op allocates `c_16` / `c_17` conditionally (unlike its sibling, which allocates both
   unconditionally), the group-norm side is where the absent-`dgamma` configuration is reached most
   directly. The gate must be a `#ifdef`, and **both** binders must emit consistent defines for the
   shared kernel.

2. **The host-side gate conditions already exist as host booleans here.** `do_mask_h`, `do_mask_w`,
   `gamma_grad_has_value`, `beta_grad_has_value`, `gamma_has_value` are all computed in the factory, so
   the promotion to `compiler_options.defines` is a direct lift. I verified the host predicates match
   the borrowed kernels' CTA-derived ones in every configuration:
   - gamma_beta_grad: host `do_mask_h = (origin_h % TILE_HEIGHT) != 0`; kernel
     `(origin_H % TILE_H) != 0 && (is_lastdim_layernorm || is_groupnorm)` with `is_groupnorm == true`.
     Equivalent. ✓ Same for `do_mask_w`. ✓
   - input_grad: host `do_mask_h = (origin_h % TILE_HEIGHT) != 0`,
     `do_mask_w = (origin_w % TILE_WIDTH) != 0`; kernel `do_mask_h = (origin_H % TILE_H) != 0 &&
     !is_lastdim_layernorm` (true here) and `do_mask_w = (origin_W % TILE_W) != 0`. Equivalent. ✓

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No hidden second writer, no multi-reader, no
  dual-instance work-split.
- **Cross-op / shared kernels:** this op **borrows all three of its compute kernels** from
  `moreh_layer_norm_backward`. See below.
- **RTA varargs:** none — but this op is the one where the *trap* shape appears. Every dataflow kernel
  walks its runtime args with a running `get_arg_val<uint32_t>(i++)` counter at the top of `kernel_main`
  (`reader_...gamma_beta_grad.cpp:13-25`, `writer_...gamma_beta_grad.cpp:13-19`,
  `reader_...input_grad_{small,large}.cpp:13-26`, `writer_...input_grad.cpp:13-16`). A running
  `arg_index++` is **not** a vararg signal: these are distinct fields read once each, in a fixed block,
  before any loop. **All named.** The counter itself disappears.

## Out-of-directory coupling

### Function-call escape — `✓ clean`

Identical donor set to the sibling op; all signatures are Device 2.0 native.

| Op kernel(s) | Donor file | Class | Shapes used | Status |
|---|---|---|---|---|
| all 5 dataflow kernels | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | shared-kernel pool (singular `kernel/`) | `fill_cb_with_value(DataflowBuffer, …)`, `generate_mask_h/_w/_h_w(DataflowBuffer, …)`, `read_tile/read_value/read_line(DataflowBuffer, AddrGen, …)`, `get_tilized_idx(...)` | ✓ |
| 3 borrowed compute kernels | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | shared-kernel pool | `*_init_with_dt(DataflowBuffer…)`, `pack_tile_with_dt(uint32_t, DataflowBuffer)` and peers | ✓ |
| 3 borrowed compute kernels | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | official kernel-lib | `compute_kernel_lib::reduce<…, cb_in, cb_scaler, cb_out>(…)` — CB ids as **non-type template parameters** | ✓ — the `DFBAccessor → uint32_t` conversion is `constexpr`, valid in template-argument position |
| all | `tt_metal/hw/inc/api/…` | LLK / HAL | — | ✓ no concern |

No donor takes `CircularBuffer&`, a `uint32_t sem_id`, a `TensorAccessorArgs<N>`, an NTTP CTA offset, or
an old-style addr-gen. No Shape-4 donor, so the donor-side Device 2.0 gate has nothing to flag.

### Borrowed kernel files (file-path instantiation) — the coupling that matters

This op owns **five** kernel sources (all dataflow) and borrows **three** (all compute):

| Borrowed kernel file | Owning op | Bound by this op's | Also bound by | `_metal2` fork exists? |
|---|---|---|---|---|
| `moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` | `moreh_layer_norm_backward` | gamma_beta_grad factory | its owner | **No** |
| `moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp` | `moreh_layer_norm_backward` | input_grad factory | its owner | **No** |
| `moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp` | `moreh_layer_norm_backward` | input_grad factory | its owner | **No** |

**Census result:** `grep -rl <filename> ttnn/cpp/ttnn/operations/` returns **exactly two consumers each**
— this op and `moreh_layer_norm_backward` — with no build-file, private-copy, or comment false
positives, and no `_metal2` sibling anywhere. **These two ops are the complete consumer set.**

Given the invoker's stated plan to port **both ops in one branch and PR**, the bundled-port rung applies
and the three kernels convert **in place**, in their owner's directory, rather than being forked. That
authorization comes from the invoker's assignment, not from this consumer list — the list is recorded so
the assignment can be checked against it, and it matches exactly.

**If the plan changes and only this op is ported**, the disposition flips: create a `_metal2` fork
**beside each original**, in `moreh_layer_norm_backward`'s directory (the sanctioned write outside your
own op), add the pointer comment to each original, and point this op's `KernelSpec::source` at the
forks.

**The CB layouts are already compatible**, which is why the sharing works and why the bundled conversion
is tractable — I verified index-for-index:

| Slot | layer_norm_backward | group_norm_backward | Agreement |
|---|---|---|---|
| gbg `c_4` | scaler | **one** | same role (reduce scaler), **different local name** |
| gbg `c_6` | *not allocated* | mask_w | kernel expects mask_w; LN's path is compile-time dead |
| gbg `c_16`/`c_17` | allocated unconditionally | allocated conditionally | same role; conditional binding unifies them |
| ig `c_5` | n_recip_n | inner_size (`n`) | same role, **different local name** |
| ig `c_6`, `c_7`, `c_16`, `c_24`–`c_31` | identical | identical | ✓ |

**Binding-name guidance:** name the bindings for the *kernel's* vocabulary, not this op's factory
locals — whatever the first fork/conversion uses becomes the interface both ops inherit. The kernels say
`cb_scaler` and `cb_n_recip_n`, so **`scaler`** and **`n_recip_n`** win over this op's `one` /
`inner_size`.

## Team-only

- **Relaxation candidates:** none observed. No custom hash exists to mine.
- **TTNN factory analysis:** both factories are plain `descriptor` ops with no op-owned tensors, no
  `MeshWorkload`, no pybound `create_descriptor`, no custom hash, and no `override_runtime_arguments`.
  Target concept `ProgramSpecFactoryConcept` for both. The only device-op-class edit the port forces is
  the `create_descriptor` → `create_program_artifacts` signature change in the two `.hpp` files.
- **Test coverage** (for the porter's baseline confirmation, not a gate): nightly pytests only —
  `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_group_norm.py`, carrying
  `test_moreh_group_norm_backward`, `test_moreh_group_norm_backward_callback` (exercises the
  program-cache hit path) and `test_moreh_group_norm_backward_rejects_invalid_mean_volume`
  (parameterised over `are_required_outputs`, so it reaches the optional-output configurations). No C++
  gtests.

## Misc anomalies  *(team-only, non-gating)*

- **Compute configs are default-constructed, dropping the op's compute-kernel-config plumbing.** Both
  factories set `compute_desc.config = ComputeConfigDescriptor{}`
  (`..._gamma_beta_grad_factory.cpp:210,220`; `..._input_grad_factory.cpp:228,238`) — no
  `math_fidelity`, no `fp32_dest_acc_en`, no `dst_full_sync_en`, no `math_approx_mode` — even though
  `MorehGroupNormBackwardGammaBetaGradOperation` carries a `compute_kernel_config` attribute. The
  sibling op resolves the same attribute through `init_device_compute_kernel_config` and passes all four
  fields. Whether that divergence is deliberate is an ops-team question; **the port reproduces it
  exactly** (build a `ComputeGen1Config` with defaults — Style B — and do *not* reroute through the TTNN
  helper, whose defaults differ).
- **`num_groups` is passed as a per-core RTA but is a program-cache key attribute.** Both factories
  send `operation_attributes.num_groups` per core (`..._gamma_beta_grad_factory.cpp:253`,
  `..._input_grad_factory.cpp:278`), along with `num_channels`, `origin_h`, `origin_w`,
  `num_inner_tiles` — all invariant across a cached program. RTA-shaped but effectively immutable; a
  later cleanup could demote them. **Not port work** (RTA→CRTA changes dispatch semantics).
- **The writer's absent-optional sentinel is a literal `0u`, not a null `Buffer*`**
  (`..._gamma_beta_grad_factory.cpp:262,267`; `..._input_grad_factory.cpp:272`). The sibling op passes a
  null `Buffer*` for the same meaning. Harmless today; both collapse to "no binding" post-port. Worth an
  ops-team note only as an inconsistency between two closely-related ops.
- **`log_info` on every factory invocation.** `..._input_grad_factory.cpp:116` and `:121` log
  `"Large/Small … algorithm is selected."` at `log_info` from production code using `LogTest` as the
  module. Cosmetic; mirrors the sibling op.

## Per-DeviceOperation attribution

| Field | GammaBetaGrad | InputGrad |
|---|---|---|
| Overall | GREEN | GREEN |
| Current / target concept | `descriptor` → `ProgramSpecFactoryConcept` | `descriptor` → `ProgramSpecFactoryConcept` |
| Kernel sources bound | 3 — 2 owned + **1 borrowed** | **5** — 3 owned + **2 borrowed**; reader/compute runtime-selected |
| Tensor bindings | 6 (2 conditional) | 6 (1 conditional) |
| CBs | 15 declared, 6 self-loop | 17 (small) / 16 (large), 8 / 7 self-loop |
| Conditional DFBs | `mask_h`, `mask_w`, `dgamma`, `dbeta` | `gamma`, `mask_h_w` |
| Compute config | `ComputeConfigDescriptor{}` (Style B, all defaults) | `ComputeConfigDescriptor{}` (Style B, all defaults) |

## Questions for the user

1. ~~**Bundled-port authorization.**~~ **RESOLVED (invoker, 2026-08-13):** bundled port confirmed —
   both ops in one branch and PR, the three borrowed compute kernels converted **in place** in
   `moreh_layer_norm_backward`'s directory, no forks. The census matches the assigned set exactly
   (two consumers, both assigned). See
   [Consequence: the atomic unit is not one factory](#consequence-the-atomic-unit-is-not-one-factory).

2. ~~**Readiness-sheet columns.**~~ **RESOLVED (invoker, 2026-08-13):** full rows supplied. Every
   checkable column cross-checked against the code with **zero conflicts** — see the cross-check table
   under [Gate detail](#gate-detail). The gate now rests on a real cross-check rather than a code-only
   derivation. Two non-blocking observations were raised for the sheet owner: the
   `Factory definition path` / `Declared in` columns both point at the `*_device_operation.hpp` rather
   than the factory `.cpp`, and the sheet has no column that can express this op's borrowing of three
   compute kernels from `moreh_layer_norm_backward`.

3. **Default-constructed compute config (informational).** Flagged under Misc anomalies: this op's
   compute kernels take an all-default `ComputeConfigDescriptor{}` while its sibling threads a resolved
   `ComputeKernelConfig` through. The port will reproduce the current behaviour exactly, but the ops
   team may want to know.

## Recipe notes

- The recipe's shared-kernel vocabulary is *borrowed* / *lent* / *intra-op*, and this pair is a fourth
  shape: **two ops being ported together where one lends and the other borrows.** The bundled-port rung
  covers it, but only if you notice that the invoker's "port both in one PR" *is* the assignment the
  rung requires. A worked sentence — "when the invoker assigns two ops that share a kernel, that is the
  bundled-port assignment; confirm the census matches the assigned set" — would have removed the
  judgement call.
- **Two reports, one coupling.** The lent/borrowed finding is symmetric, so each report is incomplete
  alone. The audit doc has no convention for cross-referencing a companion audit; I added a
  "Companion audit" line to both headers. Worth standardising.
- The audit's **Device 2.0 gate** is location-independent and therefore covers borrowed kernels — that
  is stated clearly and it worked. But the **CB endpoint census** is where borrowing actually bites: the
  census has to be run against a compute kernel that lives in another op's tree and whose CB semantics
  are set by *this* op's factory. The recipe's "follow kernel references, not directory boundaries" rule
  covers it in principle; an explicit note that the borrowed kernel's endpoint roles must be re-derived
  per binder would help.

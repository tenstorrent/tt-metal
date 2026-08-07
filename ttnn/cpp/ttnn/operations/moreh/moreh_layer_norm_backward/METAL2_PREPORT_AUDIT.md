# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward`

Two **independent** device-operations live under this op directory. They share **no** kernels and **no**
factories, but they are sequenced by a single user-facing entry point (`ttnn::moreh_layer_norm_backward`,
`moreh_layer_norm_backward.cpp:65` and `:72`), so they are bundled into one report at directory scope.
**Per-DeviceOperation attribution is retained throughout.**

- **`MorehLayerNormBackwardInputGradOperation`** — `device/moreh_layer_norm_backward_input_grad_device_operation.hpp:*`
  - `MorehLayerNormBackwardInputGradFactory` (`device/moreh_layer_norm_backward_input_grad_program_factory.cpp:17`),
    sole member of `program_factory_t`
  - kernels (5 files; **4 selectable sources** — see Runtime kernel-source selection):
    `kernels/writer_moreh_layer_norm_backward_input_grad.cpp`, **one of**
    `kernels/reader_moreh_layer_norm_backward_input_grad_{small,large}.cpp`, and **one of**
    `kernels/moreh_layer_norm_backward_input_grad_{small,large}_kernel.cpp`
- **`MorehLayerNormBackwardGammaBetaGradOperation`** — `device/moreh_layer_norm_backward_gamma_beta_grad_device_operation.hpp:*`
  - `MorehLayerNormBackwardGammaBetaGradFactory`
    (`device/moreh_layer_norm_backward_gamma_beta_grad_program_factory.cpp:*`), sole member of `program_factory_t`
  - kernels (3, no runtime selection): `kernels/reader_moreh_layer_norm_backward_gamma_beta_grad.cpp`,
    `kernels/writer_moreh_layer_norm_backward_gamma_beta_grad.cpp`,
    `kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp`

**Unreferenced kernel files:** none — all eight kernels are instantiated by one of the two factories.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `a38e7b405db 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

**Audited at:** branch `virdhatchani/BN_Porting`, HEAD `a38e7b405db`, merge-base with `main` `f6e166da2c1` (2026-08-03).

> ### Readiness-sheet provenance
>
> The claude.ai Google Drive connector is **not authorized in this session**, so the *"Operations analysis"*
> sheet could not be fetched programmatically. **The user supplied both rows directly**; they are transcribed
> verbatim in the *TTNN factory concept* gate section. Both read **`Is able to port? = yes`**. The supplied
> extract carries six columns (`Op`, `Device operation`, `Factory (variant)`, `Concept`, `Is safe to port`,
> `Is able to port?`); the `Is able to port?` conjuncts that were **not** supplied (`Custom hash`, both
> runtime-args columns, `Pybind descriptor`) were verified directly against the code instead — all absent.
> See the cross-check table.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward` |
| **Overall** | **GREEN** — both device-operations clear every gate; brief issued for both |
| **DOps / Factories** | `…InputGradOperation` → `MorehLayerNormBackwardInputGradFactory` · `…GammaBetaGradOperation` → `MorehLayerNormBackwardGammaBetaGradFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 8 kernels and all 3 donor headers are Device 2.0 |
| *Prereqs* — Cross-op escapes | Ok — 3 donor headers, all ✓ (`DataflowBuffer` by value / ref, `uint32_t` cb ids incl. NTTP position) |
| *Feature Support* — overall | **GREEN** — all Appendix A entries N/A |
| *Feature Support* — Variadic-CTA | Ok — no variable-count tensor container; every CTA read is at a literal or `constexpr` offset |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both), with a proper named factory inside `program_factory_t` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — neither is a `WorkloadDescriptor` op |
| *TTNN Readiness* — Is safe to port? | Yes (both) — sheet-owner judgment, not re-derived |
| *TTNN Readiness* — Custom hash | No (both) — confirmed: no `compute_program_hash`, no `to_hash`, no `attribute_values` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — confirmed absent |
| *TTNN Readiness* — `override_runtime_arguments` | No — confirmed absent |
| *TTNN Readiness* — Pybind `create_descriptor` | No — confirmed: `moreh_layer_norm_backward_nanobind.cpp` binds only the user-facing op |
| *TTNN Readiness* — Op-owned tensors | No — neither factory allocates a device tensor beyond its io |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (both), no op-owned tensors |
| *Port work* — Offset base pointer | none — every address arg is a clean `Buffer*`; no `->address()` expression in the directory |
| *Port work* — Tensor bindings | 11 total (6 InputGrad + 5 GammaBetaGrad), **all Case 1**; 3 conditional |
| *Port work* — TensorParameter relaxation | none — no `ArgConfig::Runtime*` anywhere |
| *Port work* — TensorAccessor 3rd arg | none — every `TensorAccessor` is the 2-arg form |
| *Port work* — CB endpoints | 1P+1C / self-loop only; **no multi-binding flag, no dead CB** — but see the `c_6` finding below |
| **⚠ Port work — newly-required `unpack_modes`** | **legacy sets none, and Metal 2.0 requires entries here**: 6 (GammaBetaGrad), 8/7 (InputGrad small/large) |
| **⚠ Heads-up — unallocated CB referenced by a kernel** | GammaBetaGrad compute references `c_6` (`cb_mask_w`) that the factory **never allocates**; safe today only because `is_groupnorm` is hardwired `false`. **Decision D1: leave as-is — `#ifdef`-gate the dead path** |
| **PR scope** | **Decision D3: one PR for both device-ops**, GammaBetaGrad first, InputGrad second |
| **Test gate** | **Decision D2: all six backward tests** in `test_moreh_layer_norm.py`; `_large` path covered by a **local-only, uncommitted** test (D2b) |

## Result

**GREEN → brief issued for both device-operations.** No gate fired for either. Both factories target
`ProgramSpecFactoryConcept`, and **no device-op-class edit is forced** — both already carry a proper
`program_factory_t` with a named factory struct, so only the factory method signature and one include change.

This is a substantially larger port than its line count suggests. Three things drive that, and all three are
recorded in detail below because each is a place a port goes quietly wrong:

1. **InputGrad runtime-selects two of its three kernel roles** (reader *and* compute) on an **L1-capacity
   computation**, so the atomic unit is the factory plus **4 selectable sources** plus the writer — and the
   DFB *set itself* differs between the two algorithms.
2. **`unpack_modes` entries must be added where legacy had none.** Both factories set `fp32_dest_acc_en` and
   give their intermediates `Float32` format, which trips Metal 2.0's require-an-explicit-entry rule. This is
   the one item in this port that is both mandatory and silent if the value is wrong.
3. **The InputGrad compute kernels alias three tmp DFBs under seven or eight different names**, each alias
   constructing its own `DataflowBuffer` object — which the port must consolidate.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** for both rows. Sheet cells as supplied:

  | `Op` | `Device operation` | `Factory (variant)` | `Concept` | `Is safe to port` | `Is able to port?` |
  |---|---|---|---|---|---|
  | `moreh/moreh_layer_norm_backward` | `MorehLayerNormBackwardGammaBetaGradOperation` | `MorehLayerNormBackwardGammaBetaGradFactory` | `descriptor` | `yes` | **`yes`** |
  | `moreh/moreh_layer_norm_backward` | `MorehLayerNormBackwardInputGradOperation` | `MorehLayerNormBackwardInputGradFactory` | `descriptor` | `yes` | **`yes`** |

  Cross-check against the code:

  | Column | Sheet | Code evidence | Match |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` → `tt::tt_metal::ProgramDescriptor` on both factory structs | ✓ |
  | `Factory (variant)` | named factory | both device-ops declare `using program_factory_t = std::variant<…Factory>;` — **not** the direct-descriptor shape | ✓ |
  | `Custom hash` | *(not supplied)* | zero hits for `compute_program_hash`, `to_hash`, `attribute_values` → `no` | ✓ (verified directly) |
  | `get_dynamic_runtime_args` | *(not supplied)* | zero hits → `no` | ✓ (verified directly) |
  | `override_runtime_arguments` | *(not supplied)* | zero hits → `no` | ✓ (verified directly) |
  | `Pybind descriptor` | *(not supplied)* | `moreh_layer_norm_backward_nanobind.cpp`: no `nb::class_` of a device op, no `create_descriptor` → `no` | ✓ (verified directly) |
  | `Secretly SPMD Workload?` | *(not supplied)* | N/A — no `WorkloadDescriptor` / `create_workload_descriptor` anywhere | ✓ |
  | Factory-set match | 2 rows | 2 DeviceOperations × 1 factory each; no phantom or missing row | ✓ |

  `Is safe to port` was **not** re-derived (expert-judgment axis). Cross-column invariants hold.

  Worth noting against the sibling `moreh_clip_grad_norm` audit: these two device-ops **do** carry a
  `program_factory_t`, so they do *not* hit the direct-descriptor problem (the framework has
  `HasDirectDescriptor` for a bare `create_descriptor` but no spec-factory equivalent —
  `ttnn/api/ttnn/operation_concepts.hpp:120, 139, 207`). No unsanctioned device-op-class edit is forced here.

- **Device 2.0 (every kernel used):** **GREEN.** All eight kernels use `Noc` for transfers,
  `DataflowBuffer` objects for every buffer, and `TensorAccessor` for every tensor walk. The
  CB→`DataflowBuffer` object swap is already done; only the id source changes in a Metal 2.0 port.

  Scans returning **zero** hits across `device/kernels/`: `InterleavedAddrGen`, `ShardedAddrGen`,
  `InterleavedPow2AddrGen`, bare `noc_async_read(` / `noc_async_write(`, `get_noc_addr(`, `noc_semaphore*`,
  `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, `get_local_cb_interface`,
  `get_dataformat(`, `get_pointer_to_cb_data`, `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`,
  `CircularBuffer`.

  Two shapes that a scan flags but which are **not** violations:
  - `get_tile_size(cb_id)` — 14 sites across 5 kernels. **Explicitly sanctioned** by the Device 2.0 gate.
    (A Metal 2.0 port does move these onto the object per kernel-side whitelist rule 7 — port work, not a
    prerequisite.)
  - `dfb.get_write_ptr()` at `reader_…_gamma_beta_grad.cpp:32`,
    `reader_…_input_grad_small.cpp:32`, `reader_…_input_grad_large.cpp:32` — the **member** form on a
    `DataflowBuffer`, i.e. the sanctioned Device 2.0 public cursor peek, not the free function.

  Donor headers were checked (the gate is location-independent) — all three are Device 2.0-clean with no
  legacy addr-gen idiom.

- **Feature compatibility:** all four Appendix A entries **N/A**.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | zero hits for `GlobalCircularBuffer`, `global_circular_buffer`, `remote_cb*`, `remote_index`; no `CBDescriptor.global_circular_buffer` set |
  | CBDescriptor `address_offset` (non-zero) | N/A | zero hits for `address_offset`, `set_globally_allocated_address`, `UpdateDynamicCircularBufferAddress`; no borrowed-memory CBs |
  | GlobalSemaphore | N/A | zero hits; neither factory declares **any** semaphore (no `SemaphoreDescriptor`) |
  | Variable-count compile-time arguments (CTA varargs) | N/A | neither `tensor_args_t` carries a variable-count container — both are fixed named-tensor structs with `std::optional` members. Every kernel `get_compile_time_arg_val` index is a literal; no runtime-varying CTA index anywhere |

- **CB endpoints (GATE-free):** every CB in both factories resolves to **1P+1C** or a **self-loop**; none
  needs the multi-binding advanced option, and none is dead. Per-CB detail in *Port-work summary*. The
  hidden-second-writer hunt (face (a)) was run across all eight kernels: **no** semaphore-gated raw co-fill
  exists (neither factory declares a semaphore), and no CB is touched by more than two distinct kernel
  instances on a node. The dual-instance work-split shape (face (c)) does not occur — each factory
  instantiates each kernel source once per core group, over disjoint node sets.

  One finding in this subject is not a normal endpoint disposition and is called out separately below: a CB
  index the **kernel references but the factory never allocates**.

- **Offset base pointers:** **GREEN** for both. There is **no `->address()` expression anywhere in the
  directory** — the only two textual matches are *comments* explaining why `Buffer*` is used instead
  (`…input_grad_program_factory.cpp:287`, `…gamma_beta_grad_program_factory.cpp:248`). Every address argument
  is a `Buffer*` pushed into `KernelDescriptor::runtime_args`, with `nullptr` for an absent optional. No host
  arithmetic could fold an offset in. Types 3 and 4 do not appear.

- **TensorAccessor 3rd argument:** **GREEN — N/A** for both. Every `TensorAccessor` construction is the 2-arg
  form; no site passes an explicit page size.

## Port-work summary  *(mirrors the brief)*

### ⚠ Newly-required `unpack_modes` entries — legacy has none, Metal 2.0 requires them

**This is the most important item in the port, and the only one that is both mandatory and silent if the
value is wrong.**

Both factories construct `ComputeConfigDescriptor` **without** an `unpack_to_dest_mode` field
(`…input_grad_program_factory.cpp:249-254`, `…gamma_beta_grad_program_factory.cpp:208-213`), so every CB
silently defaults to `UnpackToDestMode::Default`. Both also set `fp32_dest_acc_en` from the resolved compute
config, and — the part that matters — give their **intermediate** CBs `Float32` format when it is set:

```cpp
auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;
```
(`…input_grad_program_factory.cpp:120`, `…gamma_beta_grad_program_factory.cpp:105`.)

`fp32_dest_acc_en` maps to `enable_32_bit_dest = true`, and Metal 2.0's validator **requires** an explicit
`unpack_modes` entry for every **consumed Float32** DFB in that configuration
(`tt_metal/impl/metal2_host_api/program_spec.cpp:1051-1073`) where legacy silently defaulted. So the port
must **add** entries that do not exist in the legacy source, with value **`UnpackMode::UnpackToSrc`** —
the translation of the legacy `Default`. Choosing `UnpackToDest` instead would flip the precision/perf
tradeoff with no compile or test signal.

Consumed-Float32 DFBs, derived from each compute kernel's `wait_front` set:

| Factory / source | DFBs needing an explicit `UnpackToSrc` entry | Count |
|---|---|---|
| **GammaBetaGrad** | `y` (c_24), `ydy` (c_25), `dyadd` (c_26), `ydyadd` (c_27), `xmm` (c_28), `dycopy` (c_29) | 6 |
| **InputGrad — small** | `dycopy` (c_24), `y` (c_25), `dysum` (c_26), `ydysum` (c_27), `recip_nrstd` (c_28), `tmp1` (c_29), `tmp2` (c_30), `tmp3` (c_31) | 8 |
| **InputGrad — large** | `dycopy` (c_24), `y` (c_25), `dysum` (c_26), `ydysum` (c_27), `tmp1` (c_28), `tmp2` (c_29), `tmp3` (c_30) | 7 |

The **input** CBs (`c_0`–`c_7`) carry `cb_data_format` (the io dtype, bf16 in the tested configurations), so
they are not Float32 and need no entry. The **output** CBs (`c_16`, and `c_17` on GammaBetaGrad) are
producer-only for the compute kernel, so the requirement does not reach them either
(`program_spec.cpp:1053-1055` skips non-consumer bindings).

Two notes that keep this from being harder than it is:
- The requirement is **conditional on `fp32_dest_acc_en`**, because that is exactly the condition under which
  the formats become Float32. When it is false the intermediates are `cb_data_format` and
  `enable_32_bit_dest` is false, so no entry is required.
- `UnpackToSrc` is **always accepted** by the validator (`program_spec.cpp:999-1000`), so adding the entries
  unconditionally is legal too. Conditional is cleaner and self-documenting; either is correct.

#### `unpack_modes` porting policy  *(team guidance — record in both docs)*

The agreed pattern is **explicit per-DFB listing, never an auto-fill sweep.**

**Do**

- **Explicitly hand-list** `UnpackToSrc` for every consumed Float32 intermediate when `fp32_dest_acc_en` is
  true.
- Use **per-DFB calls** (e.g. `unpack_via_src(compute_config, DFB_NAME)`) — one explicit line per DFB.
- **Gate entries on `fp32_dest_acc_en`** (the same condition that makes the intermediates Float32).
- Treat validator / compiler errors for missing `unpack_modes` as **expected** — add the named DFB; do not
  work around them with a helper.

**Do not**

- **Do not** use a blanket auto-fill helper (e.g. a `fill_default_unpack_modes` that walks CONSUMER bindings
  and sets defaults). That hides missing entries and defeats the Metal 2.0 legality check. No such helper
  exists in the tree to copy; this is a prohibition on inventing one.
- **Do not** guess `UnpackToDest` — legacy had no field, so the effective default was `Default` →
  `UnpackToSrc`.

**Counts** (the explicit set to hand-list; full per-CB detail in the table above):

| Factory / path | DFBs needing explicit `UnpackToSrc` when `fp32_dest_acc_en` |
|---|---|
| GammaBetaGrad | **6**: `y`, `ydy`, `dyadd`, `ydyadd`, `xmm`, `dycopy` |
| InputGrad — small | **8**: `dycopy`, `y`, `dysum`, `ydysum`, `recip_nrstd`, `tmp1`, `tmp2`, `tmp3` |
| InputGrad — large | **7**: `dycopy`, `y`, `dysum`, `ydysum`, `tmp1`, `tmp2`, `tmp3` |

The reference shape is `unpack_via_src` / `gen1_compute_config` in
`ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_distributed_metal2_helpers.hpp`
(`:58`, `:73`). **That header belongs to a peer op and must not be `#include`d** — it is not a shared-pool
header, and including it would create the cross-op coupling the scope boundary forbids. Replicate the pattern
locally in this op's factory files.

**Reference:** Team thread — explicit listing over layernorm_distributed-style auto-fill; Frank's revised
explicit approach is the agreed pattern.

### The `c_6` finding — a kernel-referenced CB the factory never allocates

**GammaBetaGrad only. Port work, plus a latent-bug question for the ops team.**

The GammaBetaGrad compute kernel declares and *uses* a mask-w buffer:

```cpp
constexpr auto cb_mask_w = tt::CBIndex::c_6;
DataflowBuffer dfb_mask_w_obj(cb_mask_w);   // …_gamma_beta_grad_kernel.cpp:32-33
…
if (do_mask_w) { dfb_mask_w_obj.wait_front(onetile); }          // :81-83
…      copy_tile(cb_mask_w, 0, dst1);                          // :116, :191
if (do_mask_w) { dfb_mask_w_obj.pop_front(onetile); }           // :341-343
```

**The factory never allocates `c_6`.** Its CB list is `c_0`–`c_5`, `c_16`, `c_17`, `c_24`–`c_29`
(`…gamma_beta_grad_program_factory.cpp:125-138`) — there is no `c_6` push at any condition.

It is harmless *today* only because of a hardwired constant. In the kernel,
`do_mask_w = (origin_W % TILE_W) != 0 && is_groupnorm` (`…_kernel.cpp:68`), and the factory passes
`is_groupnorm` as a CTA whose value is a hardcoded `const bool is_groupnorm = false`
(`…gamma_beta_grad_program_factory.cpp:50`). So `do_mask_w` is a `constexpr bool` that is always `false`, the
path is dead, and the missing CB is never touched.

**A second, related divergence sits alongside it.** The factory and the kernel compute `do_mask_h`
*differently*:

| | expression |
|---|---|
| factory (`…gamma_beta_grad_program_factory.cpp:55`) | `(origin_H % TILE_HEIGHT) != 0 && is_lastdim_layer_norm` |
| kernel (`…_gamma_beta_grad_kernel.cpp:64`) | `(origin_H % TILE_H) != 0 && (is_lastdim_layernorm \|\| is_groupnorm)` |

These agree only because `is_groupnorm` is false. The factory allocates `c_5` on *its* `do_mask_h`
(`in5_t = do_mask_h ? 1 : 0`, `:92`), so if `is_groupnorm` ever became true with
`is_lastdim_layer_norm == false`, the kernel would `wait_front` on an unallocated `c_5` as well.

**What the port does:** `#ifdef`-gate the `cb_mask_w` declaration and every expression referencing it, with
the host emitting no such define (there is no DFB to bind). That is the standard
conditional-binding treatment and it is **zero-functional-change**, because the gated path is already dead.
Without it the kernel does not compile: `dfb::mask_w` will not exist, and `constexpr bool do_mask_w = …` does
not stop name lookup at file scope. The `do_mask_h` divergence needs the same treatment on `c_5` — gate on
the **factory's** condition, which is the one that decides whether the DFB exists.

**What the port must not do:** delete the mask-w code path outright. That is kernel-logic surgery, off the
whitelist, and it would discard the groupnorm scaffolding the kernel was written to support.

**Invoker decision (D1): leave as-is** — `#ifdef`-gate the dead path, do not allocate `c_6`, do not enable
groupnorm, preserve current behavior. See *Decisions from the invoker* below.

The residual observation still routes to the **ops team** as an open item, *not* port work: the kernel is
written for a groupnorm mode the factory hardwires off, and the two sides' mask conditions have already
drifted apart. The port will freeze that shape into `#ifdef`s, so the divergence becomes harder to notice
afterwards — which is the reason to record it now.

### Runtime kernel-source selection — the true size of the port

**InputGrad selects two of its three kernel roles at runtime**, on an **L1-capacity computation**:

```cpp
const uint32_t cb_usage = …;                              // …input_grad_program_factory.cpp:123-125
const bool use_large_algorithm = cb_usage >= available_L1;  // :128
```

- reader → `reader_…_input_grad_large.cpp` or `…_small.cpp` (`:205-209`)
- compute → `moreh_layer_norm_backward_input_grad_large_kernel.cpp` or `…_small_kernel.cpp` (`:230-234`)

So the atomic unit for InputGrad is **factory + writer + 4 selectable sources = 5 kernel files**, all
converting together. GammaBetaGrad has no selection: 3 files.

**The algorithm choice also changes the DFB set and the name→DFB mapping**, which is unusual and easy to
miss:

| | small | large |
|---|---|---|
| `im0_t` (dycopy, c_24) / `im1_t` (y, c_25) | `num_inner` entries each | **1** entry each (`:132-133`) |
| `im7_t` (c_31) | 1 entry | **0 → CB not allocated** (`:134`, and `push_cb` skips zero-size CBs at `:141-145`) |
| tmp1 / tmp2 / tmp3 | c_29 / c_30 / c_31 | c_28 / c_29 / c_30 |
| `recip_nrstd` | its own CB, c_28 | **aliased onto tmp3** (`…large_kernel.cpp:335`) |

So the small path declares 17 CBs and the large path 16, with three of them bound to different roles. Each
selected source therefore needs its own `KernelSpec` **and** the `DataflowBufferSpec` set must branch on
`use_large_algorithm` — which the legacy code already does, so the port keeps that branch rather than
inventing one.

### Same-FIFO aliasing at scale — InputGrad compute kernels

The InputGrad compute kernels give the three tmp DFBs a **semantic name per use-phase**, via
`constexpr auto` aliases — and construct a **separate `DataflowBuffer` object for each alias**:

**small kernel** (tmp1 = c_29, tmp2 = c_30, tmp3 = c_31):

| DFB | names it wears | alias sites |
|---|---|---|
| c_29 (tmp1) | `tmp1`, `dyadd`, `ndy` | `:58`, `:256-257`, `:368-369` |
| c_30 (tmp2) | `xmm`, `ydy`, `ndymdysum` | `:113-114`, `:298-299`, `:385-386` |
| c_31 (tmp3) | `ydyadd`, `yydysum` | `:300-301`, `:409-410` |

**large kernel** (tmp1 = c_28, tmp2 = c_29, tmp3 = c_30):

| DFB | names it wears | alias sites |
|---|---|---|
| c_28 (tmp1) | `dyadd`, `ndy`, `xmm`, `yydysum` | `:87-88`, `:443-444`, `:486-487`, `:548-549` |
| c_29 (tmp2) | `ydyadd`, `ndymdysum` | `:89-90`, `:462-463` |
| c_30 (tmp3) | `xmm`, `ydy`, `recip_nrstd` | `:94-95`, `:268-269`, `:335-336` |
| c_25 (y) | also `tmp4` | `:572-573` |

This is **[Same-FIFO aliasing]**, not `alias_with`: one DFB, several kernel-side names, shared FIFO pointers.
Modelling it with `advanced_options.alias_with` would be a bug — it would create independent FIFOs at one
address and lose the pointer coherence the kernel relies on.

The port keeps **one `DataflowBufferSpec` and one `DFBBinding` per real DFB**, and expresses each extra name
as a `constexpr auto` handle alias. It must also **consolidate the objects**: constructing several
`DataflowBuffer`s from the same handle compiles and runs but breaks the object↔DFB identity that device-side
debug tooling depends on. One object per DFB, aliases on the handle.

Note `cb_tmp4 = cb_y` in the large kernel aliases an **input-side intermediate** (c_25), not a tmp — worth
not overlooking when mapping names to DFBs.

### Preserved multiplicity — both factories

Both factories emit **two compute `KernelDescriptor`s** over the two work-split core groups, differing only
in the per-group row/column-count CTA:

| Factory | CTA that varies | descriptors |
|---|---|---|
| InputGrad | `num_rows_per_core_group_{1,2}` | `:236-254`, `:256-277` |
| GammaBetaGrad | `num_cols_per_core_group_{1,2}` | `:193-213`, `:215-238` |

This maps 1:1 to **two compute `KernelSpec`s of the same source in two `WorkUnitSpec`s** — the canonical
preserved-multiplicity case. Reader and writer sit on `all_cores`, so they belong to **both** work units and
their derived node set is the union:

```
wu_g1: {READER, WRITER, COMPUTE_G1}  target_nodes = core_group_1
wu_g2: {READER, WRITER, COMPUTE_G2}  target_nodes = core_group_2
```

Do **not** collapse to one compute `KernelSpec` by demoting the per-group count to an RTA — that is the
documented anti-pattern and it costs compile-time loop unrolling. The second group is conditional
(`has_core_group_2`), exactly as in legacy.

Both factories cover **only** the cores the work split assigned — the per-core loop `TT_THROW`s on a core
outside both groups (`…input_grad_program_factory.cpp:305`, `…gamma_beta_grad_program_factory.cpp:263`), so
unlike some ops there is no idle-core RTA padding to reproduce.

### Tensor bindings — 11, all Case 1, 3 conditional

Every tensor address feeds a `TensorAccessor`, so all eleven are **Case 1**. No Case 2 anywhere — no kernel
does raw base-pointer arithmetic, and the compute kernels construct no `TensorAccessor`, so the blocked
compute-kernel Case-2 path cannot arise. All eleven arrive in the **`Buffer*` form** (correct-on-cache-hit
today, superseded by the typed binding); the factories comment explicitly on why
(`…input_grad_program_factory.cpp:287-289`, `…gamma_beta_grad_program_factory.cpp:248-252`).

**InputGrad** — 6:

| `TensorParameter` | Origin | Bind on | Accessor site | Conditional |
|---|---|---|---|---|
| output_grad | `tensor_args.output_grad` | reader | reader small/large | — |
| input | `tensor_args.input` | reader | reader small/large | — |
| mean | `tensor_args.mean` | reader | reader small/large | — |
| rstd | `tensor_args.rstd` | reader | reader small/large | — |
| gamma | `tensor_args.gamma` | reader | reader small/large | **yes** (`gamma_has_value`) |
| input_grad | `tensor_return_value` | writer | `writer_…_input_grad.cpp` | — |

**GammaBetaGrad** — 5:

| `TensorParameter` | Origin | Bind on | Accessor site | Conditional |
|---|---|---|---|---|
| output_grad | `tensor_args.output_grad` | reader | `reader_…_gamma_beta_grad.cpp` | — |
| input | `tensor_args.input` | reader | " | — |
| mean | `tensor_args.mean` | reader | " | — |
| rstd | `tensor_args.rstd` | reader | " | — |
| gamma_grad | `tensor_return_value[0]` | writer | `writer_…_gamma_beta_grad.cpp` | **yes** |
| beta_grad | `tensor_return_value[1]` | writer | " | **yes** |

GammaBetaGrad's `tensor_return_value_t` is `std::vector<std::optional<Tensor>>` — a **fixed-length-2 vector
of optionals**, not a variable-count list, so it is ordinary conditional-binding work, not a variadic case.
The composite guarantees at least one is present (`moreh_layer_norm_backward.cpp:28-30`).

### Conditional DFB specs and bindings

| Factory | Conditional DFB | Condition | Factory site |
|---|---|---|---|
| InputGrad | `gamma` (c_6) | `gamma_has_value` | `in6_t`, `:102` |
| InputGrad | `mask_h_w` (c_7) | `do_mask_h \|\| do_mask_w` | `in7_t`, `:103` |
| InputGrad | `tmp3` (c_31) | `!use_large_algorithm` | `im7_t`, `:116, :134` |
| GammaBetaGrad | `mask_h` (c_5) | factory's `do_mask_h` | `in5_t`, `:92` |
| GammaBetaGrad | `mask_w` (c_6) | **never allocated** — see the `c_6` finding | — |

The `push_cb` lambda skips zero-size CBs (`…input_grad_program_factory.cpp:141-145`,
`…gamma_beta_grad_program_factory.cpp:109-113`), which is what makes these conditional rather than
zero-sized. Each needs a host `defines` entry plus `#ifdef`-gated kernel-side alias and uses. `c_31` is a
special case: because it exists only on the small path and each source gets its own `KernelSpec`, the natural
treatment is *per-source bindings* rather than an `#ifdef` — the large kernel simply never names it.

### CB endpoints — per factory

**GammaBetaGrad** (14 allocated CBs):

| CB | Role | Disposition |
|---|---|---|
| `c_0` dy, `c_1` x, `c_2` mean, `c_3` rstd, `c_4` scaler | reader P → compute C | 1P+1C ×5 |
| `c_5` mask_h *(conditional)* | reader P → compute C | 1P+1C |
| `c_16` dgamma, `c_17` dbeta | compute P → writer C | 1P+1C ×2 |
| `c_24` y, `c_25` ydy, `c_26` dyadd, `c_27` ydyadd, `c_28` xmm, `c_29` dycopy | compute produces + consumes | **self-loop** ×6 |

**InputGrad** (17 small / 16 large):

| CB | Role | Disposition |
|---|---|---|
| `c_0` dy, `c_1` x, `c_2` mean, `c_3` rstd, `c_4` scaler, `c_5` n_recip_n | reader P → compute C | 1P+1C ×6 |
| `c_6` gamma *(conditional)*, `c_7` mask_h_w *(conditional)* | reader P → compute C | 1P+1C ×2 |
| `c_16` dx | compute P → writer C | 1P+1C |
| `c_24` dycopy, `c_25` y, `c_26` dysum, `c_27` ydysum | compute produces + consumes | **self-loop** ×4 |
| `c_28` recip_nrstd *(small)* / tmp1 *(large)* | compute produces + consumes | **self-loop** |
| `c_29`, `c_30` | compute produces + consumes | **self-loop** ×2 |
| `c_31` *(small only)* | compute produces + consumes | **self-loop** |

No dead CB, no ≥3-toucher, no FIFO-role doubling anywhere → **no multi-binding flag in either factory**. The
self-loop count is high because both compute kernels do all their intermediate algebra internally; that is
the ordinary compute-intermediate shape, legal on Gen1 for compute kernels.

### Hardware configuration and `opt_level`

- **Compute config is Style A in both factories** — each resolves a TTNN config and destructures it with
  `get_compute_kernel_config_args` (`…input_grad_program_factory.cpp:91-92`,
  `…gamma_beta_grad_program_factory.cpp:81-82`), then sets four fields on `ComputeConfigDescriptor`:
  `math_fidelity`, `fp32_dest_acc_en`, `dst_full_sync_en`, `math_approx_mode`. So
  `ttnn::to_compute_hardware_config(device->arch(), config)` is the right translation and carries exactly
  those four (minding the `math_approx_mode` bool→`Precision` mapping and the
  `dst_full_sync_en` → `double_buffer_dest` **inversion**).
  - **InputGrad re-resolves the config inside the factory**
    (`…input_grad_program_factory.cpp:33-34`: `init_device_compute_kernel_config(arch,
    operation_attributes.compute_kernel_config)`) rather than using the attribute directly. Translate the
    *re-resolved* value, not the raw attribute.
  - `bfp_pack_precision_mode` is not set by either factory → leave at its default (defaults coincide).
  - `unpack_modes` is the exception — see the dedicated section above.
- **DM configs are plain defaults** — `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` on all
  dataflow kernels in both factories. Use the arch-agnostic TTNN reader/writer helpers.
- **`opt_level`: not set anywhere** in the directory. Resolved: `O2` for the DM kernels (no action) and
  **`O3` for the compute kernels** — Metal 2.0 defaults to `O2`. Every compute `KernelSpec` needs an explicit
  `compiler_options.opt_level = KernelBuildOptLevel::O3`, and there are **two per factory** (one per core
  group), so four sites in the change.

### Runtime args

All named; **no varargs anywhere**. Every kernel reads each arg once as a distinct field at a literal index.

| Kernel | Legacy RTA count | after the address slots drop |
|---|---|---|
| InputGrad reader | 15 (5 `Buffer*` + 10 scalars) | 10 named |
| InputGrad writer | 4 (1 `Buffer*` + 3) | 3 named |
| InputGrad compute | 0 (all CTAs) | — |
| GammaBetaGrad reader | 12 (4 `Buffer*` + 8) | 8 named |
| GammaBetaGrad writer | 4 (2 `Buffer*` + 2) | 2 named |
| GammaBetaGrad compute | 0 (all CTAs) | — |

Both legacy loops are node-first, so keep them and let `AddRuntimeArgsForNode` transpose; do not
re-architect into name-first form as part of this port.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** **no borrowed or lent kernel *files*** — all eight sources live in this
  directory and are bound only by these two factories (verified per file with
  `grep -rl <filename> ttnn/cpp/ttnn/operations/`). No `_metal2` fork needed. Coupling is function-call
  escape only, via three donor headers, all ✓:
  - `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (shared pool) — `DataflowBuffer`-taking helpers, native
    Device 2.0 shape.
  - `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` (shared pool) — same; also the
    `*_init_with_dt(DataflowBuffer)` family the compute kernels lean on heavily.
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` (official kernel_lib) — `compute_kernel_lib::reduce`
    (`:392`) takes cb ids as **non-type template parameters**. `dfb::name` is valid there (the
    `DFBAccessor → uint32_t` conversion is `constexpr`), so it crosses without a shim.
- **`read_mean_rstd` is an in-file helper, not a donor** — defined identically at line 12 of all three
  reader kernels, taking a `uint32_t cb_id` and constructing a `DataflowBuffer` internally. Because it is in
  the porter's own files, either treatment works: pass `dfb::mean` / `dfb::rstd` through the implicit
  conversion (minimal diff), or change the parameter to `DataflowBuffer`. The minimal-diff choice is
  preferred for this port.
- **RTA varargs:** none.
- **The GammaBetaGrad compute kernel picks its pack target with a file-scope ternary** —
  `constexpr auto cb_out_init = gamma_grad_has_value ? cb_dgamma : cb_dbeta;`
  (`…_gamma_beta_grad_kernel.cpp:73`). Both `c_16` and `c_17` are allocated **unconditionally**
  (`out0_t = out1_t = 1`, `…gamma_beta_grad_program_factory.cpp:94-95, 131-132`), so both DFBs always exist
  and the ternary needs **no** `#ifdef` — only the *tensor* bindings for gamma_grad/beta_grad are
  conditional. Worth stating because a ternary over two DFB names is normally the shape that does need
  gating.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** Three donor headers, all ✓; no ⚠ / ✗ / ⭐ entries; no borrowed kernel files.
No donor gates either factory, and none needs donor-side work. Per-call detail is in the Heads-ups section.

One Quasar-facing note (not Gen1, not porter work): the compute kernels depend heavily on the
`*_with_dt` (data-format-reconfiguring) helpers in `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`, which
take `DataflowBuffer` objects and read tile/format metadata off them. Those are already on the object rather
than on cb-id arrays, which is why this op's Device 2.0 story is unusually clean — worth the kernel-lib
owners knowing that a Metal 2.0 caller reaches them with DFB-derived handles.

### Relaxation candidates

None. No `ArgConfig::Runtime*` anywhere, no `TensorAccessor` third argument. Keep strict matching. The
shape-dependent quantities (`origin_H`, `origin_W`, `num_inner`, `num_outer`, `mean_rstd_height/width`,
`normalized_dims`) all travel as CTAs or ordinary scalar RTAs, and the tensors are tile-layout, so the
accessor configuration does not vary with shape — a shape change correctly invalidates the cache entry.

### TTNN factory analysis

- **Op-owned tensors:** none in either factory. Both take all buffers from `tensor_args` /
  `tensor_return_value`.
- **MeshWorkload need:** none — both are plain `descriptor` single-program ops.
- **Pybind surface:** `moreh_layer_norm_backward_nanobind.cpp` binds only the user-facing operation. None of
  the three sanctioned device-op-class edits applies, and — unlike `moreh_clip_grad_norm` — the
  fourth (direct-descriptor → `program_factory_t`) does not either, since both device-ops already declare
  `program_factory_t`. **No device-op-class edit is forced beyond the factory method signature and the
  `program_descriptors.hpp` → `metal_v2_artifacts.hpp` include swap.**
- **Custom hash:** none, in any form.

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **`packer_l1_acc` is resolved and dropped on the floor in both factories**
  (`…input_grad_program_factory.cpp:91`, `…gamma_beta_grad_program_factory.cpp:81`) — destructured from
  `get_compute_kernel_config_args` and never used. `ComputeConfigDescriptor` has no such field, and neither
  does Metal 2.0's `ComputeGen1Config`, so nothing is lost in the port and nothing can be restored by it.
  A pre-existing gap from the op's ProgramDescriptor migration.
- **`is_groupnorm` is hardwired `false` in both factories** (`…input_grad_program_factory.cpp:50`,
  `…gamma_beta_grad_program_factory.cpp:50`) while all three compute kernels carry live `is_groupnorm`
  branches driven by a CTA. Substantial dead-but-compiled code, and — per the `c_6` finding — the reason a
  CB the kernel expects is never allocated. Either the groupnorm path should be reachable or the scaffolding
  should be retired; today it is neither.
- **`log_info(tt::LogTest, …)` on the algorithm-selection hot path** —
  `…input_grad_program_factory.cpp:131` and `:136` log "Large/Small … algorithm is selected" on **every
  cache miss**, at `LogTest` severity from production code. Noise, and the wrong log category.
  **⚠ Do not clean this up in this port, or ahead of it:** per decision D2b the *large* line is the porter's
  only signal that the `_large` algorithm was actually exercised, since no committed test reaches that path.
  If it is ever tidied, keep an equivalent signal — otherwise the next porter of this op loses the ability to
  confirm large-path coverage.
- **`read_mean_rstd` is triplicated** — byte-identical definitions at line 12 of all three reader kernels
  (~100 lines each). A natural candidate for the shared moreh kernel helpers; the port must **not** hoist it
  (that would be an out-of-scope refactor touching a shared pool).
- **InputGrad re-resolves the compute-kernel config the caller already resolved.** The composite calls
  `init_device_compute_kernel_config` (`moreh_layer_norm_backward.cpp:24-25`) and the factory calls it again
  on the stored attribute (`…input_grad_program_factory.cpp:33-34`). GammaBetaGrad does not. Harmless if
  idempotent, but the asymmetry between the two factories is a trap for anyone changing the defaults.
- **`im5_t` / `im6_t` (c_29 / c_30 on InputGrad) are declared with no explanatory comment**
  (`…input_grad_program_factory.cpp:114-115`) while every other CB has one. Their roles (tmp2 / tmp3, or
  tmp1 / tmp2 depending on algorithm) are only discoverable from the kernels.

## Per-DeviceOperation attribution

| Field | `InputGrad` | `GammaBetaGrad` |
|---|---|---|
| Overall | GREEN | GREEN |
| Factory | `MorehLayerNormBackwardInputGradFactory` | `MorehLayerNormBackwardGammaBetaGradFactory` |
| Kernel entry points to convert | **5** (writer + 2 readers + 2 computes) | 3 |
| Runtime source selection | **yes** — reader *and* compute, on L1 capacity | no |
| CBs → DFBs | 17 (small) / 16 (large); 3 conditional | 14; 1 conditional (+ the unallocated `c_6`) |
| Tensor bindings | 6 (1 conditional: gamma) | 5 (2 conditional: gamma_grad, beta_grad) |
| Semaphores | none | none |
| Self-loop DFBs | 8 (small) / 7 (large) | 6 |
| Multi-binding flag | not needed | not needed |
| Dead CBs | none | none allocated-but-dead; **one referenced-but-unallocated (`c_6`)** |
| Preserved multiplicity | 2 compute specs + 2 work units | 2 compute specs + 2 work units |
| New `unpack_modes` entries | 8 (small) / 7 (large) | 6 |
| `opt_level` | explicit `O3` on both compute specs | explicit `O3` on both compute specs |
| Forced device-op edit | none beyond the signature | none beyond the signature |

## Decisions from the invoker  *(questions raised by this audit, now resolved)*

All three questions this audit raised have been answered by the invoker. Recorded here so the team doc and the
porter brief agree, and so a later reader can see what was decided rather than what was asked.

### D1 — GammaBetaGrad `c_6` / groupnorm: **leave as-is**

**Decision:** preserve current behavior; zero functional change.

- The port **`#ifdef`-gates** the dead `cb_mask_w` (`c_6`) path in the GammaBetaGrad compute kernel.
- The factory does **not** allocate `c_6`, and groupnorm is **not** enabled.
- The `do_mask_h` divergence between factory and kernel stays as it is; the gate is taken from the
  **factory's** condition (the one that decides whether the DFB exists).

So the port freezes the current dead-path shape into `#ifdef`s, which is exactly the faithful,
zero-functional-change translation. The underlying groupnorm question (should the mode be reachable, or the
scaffolding retired?) is **not** resolved by this decision and remains an open item for the op owner —
recorded in Misc anomalies, not port work.

### D2 — Test baseline: **all `moreh_layer_norm_backward` tests, not a subset**

**Decision:** run every backward-relevant test; backward tests are the **gate**. Forward tests in the same
file are fine to run for regression but are not the gate.

Minimum set, all in `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py`:

| Test | Line | Why it matters for this port |
|---|---|---|
| `test_moreh_layer_norm_backward_compute_kernel_options` | `:608` | **Run first during the port** — sweeps compute-kernel options, so it covers `fp32_dest_acc_en` both ways and therefore the newly-required `unpack_modes` entries |
| `test_moreh_layer_norm_backward_with_gamma_or_beta` | `:542` | Conditional gamma_grad / beta_grad bindings |
| `test_moreh_layer_norm_backward_callback` | `:680` | Program-cache hit path / `UpdateTensorArgs` |
| `test_moreh_layer_norm_backward` | `:508` | Primary functional coverage |
| `test_moreh_layer_norm_backward_rejects_invalid_mean_volume` | `:695` | Validation |
| `test_moreh_layer_norm_backward_rejects_same_volume_wrong_mean_shape` | `:729` | Validation |

**"Any other test in the repo that exercises `moreh_layer_norm_backward`" — swept, and there are none.**
Verified after the decision:

- `grep -rln 'layer_norm_backward\|layernorm_backward' tests/ models/` → exactly two hits:
  `test_moreh_layer_norm.py` (the file above) and `tests/sweep_framework/Allops.txt`, which is a **plain
  name list**, not an executable test.
- **No C++ gtests** — zero hits under `tests/ttnn/unit_tests/gtests/` or `tests/tt_metal/`.
- **No sweep covers this op.** The three `layer_norm`-named sweeps
  (`sweeps/fused/layer_norm_traces.py`, `sweeps/model_traced/layer_norm_model_traced.py`,
  `sweeps/normalization/generality/layernorm.py`) exercise the **non-moreh** `ttnn.layer_norm`; none matches
  `layer_norm_backward`. *(Independently, sweep-framework modules are currently unimportable — the `tests/`
  packaging breaks `tests.ttnn.*` imports — so a sweep could not serve as a regression gate regardless.)*
- No non-nightly variant of the file exists.

So the six tests above **are** the complete backward gate, and the file's forward tests are the available
regression margin.

### D2b — `_large` algorithm coverage: **local-only verification, not committed**

**Decision:** the existing parametrizations likely reach the **small** path only. Cover `_large` locally.

Procedure for the porter:

1. Create a **local-only** test file with a deliberately large shape that forces `use_large_algorithm`
   (`cb_usage >= available_L1`, `…input_grad_program_factory.cpp:128`).
2. Run it and confirm the log line
   **`"Large moreh_layer_norm_backward_input_grad algorithm is selected."`**
   (emitted at `…input_grad_program_factory.cpp:131`).
3. **Do not commit or push** that file.
4. Record the result in `METAL2_PORT_REPORT.md` under test-coverage notes.

**Consequence for the Misc anomalies list:** the two `log_info(tt::LogTest, …)` calls at
`…input_grad_program_factory.cpp:131` and `:136` are flagged below as noise at the wrong severity — but this
decision makes the *large* one the porter's **verification signal**. It must survive the port untouched.
Cleaning it up is now doubly out of scope, and any future cleanup should keep an equivalent signal (or the
next porter loses the only way to confirm the large path ran).

### D3 — PR scope: **one PR, both device-operations, GammaBetaGrad first**

**Decision:** land `GammaBetaGrad` and `InputGrad` together in a single PR, implemented in this order:

1. **GammaBetaGrad** — 3 kernels, no runtime source selection. The smaller, simpler half.
2. **InputGrad** — 5 kernels, the small/large branch and the algorithm-dependent DFB set.

The ordering matters beyond convenience: GammaBetaGrad exercises the two mechanisms InputGrad then needs at
larger scale — the newly-required `unpack_modes` entries and conditional DFB/tensor bindings — on a factory
small enough to debug. Getting it green first de-risks the harder half.

## Recipe notes

1. **The require-an-explicit-`unpack_modes`-entry rule deserves a worked "legacy set nothing" case.** The
   recipe's `unpack_modes` guidance is framed around *translating* a legacy `unpack_to_dest_mode` vector
   (reindexing, value mapping, the `Default`→`UnpackToSrc` / `UnpackToDestFp32`→`UnpackToDest` fork). This op
   is the other shape: legacy sets the field **not at all**, yet Metal 2.0 requires entries because
   `fp32_dest_acc_en` is on and the intermediates are `Float32`. A porter following the "copy the legacy
   values" instruction literally would emit an empty table and hit a validator failure whose message names a
   DFB but not the reason. One paragraph — *"when the legacy factory sets no `unpack_to_dest_mode` at all but
   does set `fp32_dest_acc_en`, you must still add an explicit `UnpackToSrc` entry for every consumed
   Float32 DFB; the absent legacy field means `Default`, which is `UnpackToSrc`"* — would close it.

2. **Nothing covers a CB the kernel references but the factory never allocates.** The CB-endpoints subject
   has a *dead CB* case (allocated, zero touchers) and its careful "distrust a `(0,0)` result" guidance, but
   not the mirror image: a `buffer_index` a kernel names and FIFO-touches under a permanently-false
   compile-time gate, with no `CBDescriptor` behind it. It is invisible to a CB-first census (there is no CB
   to enumerate) and only shows up if you enumerate from the *kernel* side. On Gen1 legacy it is harmless;
   in Metal 2.0 it is a hard compile failure. Suggested addition to the CB-endpoints subject: after the
   per-CB census, sweep each kernel's CB-index references and confirm every one has an allocating
   `CBDescriptor` in at least one config — flagging any that do not as conditional-binding work plus an
   ops-team question.

3. **The audit's `override_runtime_arguments` gate conjunct looks stale.** `CustomProgramSpecFactoryConcept`
   — a spec factory whose `override_runtime_arguments` returns a `ProgramRunArgs` applied via
   `UpdateProgramRunArgs` — is on `main` at `ttnn/api/ttnn/operation_concepts.hpp:132`, so routing that column
   to "not supported yet on the Metal 2.0 side" will mis-gate the next op that trips it. Did not affect this
   audit (both rows clean). *(Third report carrying this note; repeated so this document stands alone.)*

4. **A "runtime source selection changes the DFB set" case would be worth naming.** The recipe covers
   runtime kernel-source selection well (enumerate every selectable source; they convert together; map DFB
   producer/consumer roles per source path). What it does not say is that the **`DataflowBufferSpec` set
   itself** may differ per selected source — here the small path allocates `c_31` and a dedicated
   `recip_nrstd`, and the large path allocates neither, aliasing `recip_nrstd` onto a tmp instead. A porter
   who builds one DFB list and then branches only the `KernelSpec`s would either over-allocate L1 on the
   large path or bind a DFB the large kernel never names. One sentence in the inventory step —
   *"record whether the CB set, not just the kernel sources, varies with the selection"* — would catch it.

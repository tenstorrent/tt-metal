# Metal 2.0 Port Brief — `moreh_layer_norm_backward` (both device-operations)

> Audit cleared all gates for **both** device-operations in this directory. This is your actionable input;
> the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓
— for `MorehLayerNormBackwardInputGradOperation` **and** `MorehLayerNormBackwardGammaBetaGradOperation`.

**Recipe docs:** `a38e7b405db 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

**Audited at:** branch `virdhatchani/BN_Porting`, HEAD `a38e7b405db`, merge-base with `main` `f6e166da2c1`.

## Scope — two independent factories, 8 kernel files

| Device-op | Factory | Kernel entry points to convert |
|---|---|---|
| `…GammaBetaGradOperation` | `MorehLayerNormBackwardGammaBetaGradFactory` | `reader_…_gamma_beta_grad.cpp`, `writer_…_gamma_beta_grad.cpp`, `moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` — **3, no runtime selection** |
| `…InputGradOperation` | `MorehLayerNormBackwardInputGradFactory` | `writer_…_input_grad.cpp`, **both** `reader_…_input_grad_{small,large}.cpp`, **both** `moreh_layer_norm_backward_input_grad_{small,large}_kernel.cpp` — **5, two roles runtime-selected** |

### PR scope and order — decided by the invoker

**One PR for both device-operations**, implemented in this order:

1. **GammaBetaGrad first** — 3 kernels, no runtime source selection. The smaller, simpler half.
2. **InputGrad second** — 5 kernels, the small/large branch and the algorithm-dependent DFB set.

The order is not just convenience: GammaBetaGrad exercises the two mechanisms InputGrad then needs at larger
scale — the newly-required `unpack_modes` entries (item 1) and conditional DFB/tensor bindings (items 6–7) —
on a factory small enough to debug. Get it green before starting InputGrad.

The two factories are structurally independent, so a stalled InputGrad does not invalidate a finished
GammaBetaGrad; if you have to stop, stop cleanly between them and report the split.

**No device-op-class edit is forced.** Both device-ops already declare
`using program_factory_t = std::variant<…Factory>;` with a named factory struct, so the only header changes
are the factory method signature (`ProgramDescriptor create_descriptor` →
`ttnn::device_operation::ProgramArtifacts create_program_artifacts`) and swapping
`<tt-metalium/program_descriptors.hpp>` for `"ttnn/metal_v2_artifacts.hpp"`. No custom hash to delete, no
pybind cleanup, no pybind-hook-only parameter.

## TTNN factory analysis

- **Current concept:** `descriptor` (both), named factory inside `program_factory_t`
- **Op-owned tensors:** none (both) — leave `ProgramArtifacts::op_owned_tensors` defaulted
- **Target concept:** `ProgramSpecFactoryConcept` (both)
- **Gate-cleared, confirmed absent:** custom hash (in any form — no `compute_program_hash`, no `to_hash`, no
  `attribute_values`) · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor`

## Construct — to do

### ⚠ 1. Add `unpack_modes` entries that legacy does not have — the one mandatory-and-silent item

Both factories build `ComputeConfigDescriptor` **without** an `unpack_to_dest_mode` field
(`…input_grad_program_factory.cpp:249-254`, `…gamma_beta_grad_program_factory.cpp:208-213`), so every CB
defaults to `Default`. Both also set `fp32_dest_acc_en` **and** give their intermediates `Float32` format:

```cpp
auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;
```
(`…input_grad_program_factory.cpp:120`, `…gamma_beta_grad_program_factory.cpp:105`.)

`fp32_dest_acc_en` → `enable_32_bit_dest = true`, and Metal 2.0 **requires** an explicit `unpack_modes` entry
for every **consumed Float32** DFB in that configuration
(`tt_metal/impl/metal2_host_api/program_spec.cpp:1051-1073`). So you must **add** entries the legacy source
never had, all with value **`UnpackMode::UnpackToSrc`** — the translation of the absent legacy field
(`Default`). Picking `UnpackToDest` instead flips the precision/perf tradeoff with **no compile or test
signal**.

| Factory / source | DFBs needing an explicit `UnpackToSrc` entry | Count |
|---|---|---|
| **GammaBetaGrad** | `y` (c_24), `ydy` (c_25), `dyadd` (c_26), `ydyadd` (c_27), `xmm` (c_28), `dycopy` (c_29) | 6 |
| **InputGrad — small** | `dycopy` (c_24), `y` (c_25), `dysum` (c_26), `ydysum` (c_27), `recip_nrstd` (c_28), `tmp1` (c_29), `tmp2` (c_30), `tmp3` (c_31) | 8 |
| **InputGrad — large** | `dycopy` (c_24), `y` (c_25), `dysum` (c_26), `ydysum` (c_27), `tmp1` (c_28), `tmp2` (c_29), `tmp3` (c_30) | 7 |

Inputs (`c_0`–`c_7`) carry the io dtype, not Float32 → no entry. Outputs (`c_16`, and `c_17` on
GammaBetaGrad) are producer-only for compute → the requirement does not reach them
(`program_spec.cpp:1053-1055`).

Gate the entries on `fp32_dest_acc_en` — that is exactly when the formats become Float32. (`UnpackToSrc` is
always accepted, `program_spec.cpp:999-1000`, so unconditional entries would also be legal; conditional is
cleaner.)

#### `unpack_modes` porting policy — team guidance, follow it exactly

**Explicit per-DFB listing, never an auto-fill sweep.**

**Do**

- **Hand-list `UnpackToSrc` explicitly** for every consumed Float32 intermediate when `fp32_dest_acc_en` is
  true — one line per DFB, so the set is visible in the diff and reviewable against the counts below.
- Use a **per-DFB call** — one explicit line each, in the shape of `unpack_via_src(compute_config, DFB_NAME)`.
- **Gate the entries on `fp32_dest_acc_en`** — the same condition that makes the intermediates Float32, and
  the same condition that binds them.
- Treat a validator or compile error naming a missing `unpack_modes` DFB as **expected feedback**: add that
  named DFB explicitly. Do **not** work around it with a helper.

**Do not**

- **Do not use a blanket auto-fill helper** — e.g. a `fill_default_unpack_modes` that walks the kernel's
  CONSUMER bindings and sets defaults. It hides missing entries and defeats the Metal 2.0 legality check,
  which exists precisely to force the choice to be stated. (No such helper exists in the tree to copy; this is
  a prohibition on inventing one.)
- **Do not guess `UnpackToDest`.** Legacy had no field at all, so the effective default was `Default` →
  **`UnpackToSrc`**. `UnpackToDest` would flip the precision/perf tradeoff silently.

**Counts** — the explicit set to hand-list, per factory and path (full per-CB detail in item 1's table above):

| Factory / path | DFBs needing explicit `UnpackToSrc` when `fp32_dest_acc_en` |
|---|---|
| GammaBetaGrad | **6**: `y`, `ydy`, `dyadd`, `ydyadd`, `xmm`, `dycopy` |
| InputGrad — small | **8**: `dycopy`, `y`, `dysum`, `ydysum`, `recip_nrstd`, `tmp1`, `tmp2`, `tmp3` |
| InputGrad — large | **7**: `dycopy`, `y`, `dysum`, `ydysum`, `tmp1`, `tmp2`, `tmp3` |

**The reference implementation, and the boundary caveat.** The shape to replicate lives at
`ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_distributed_metal2_helpers.hpp`:

```cpp
inline void unpack_via_src(m2::ComputeGen1Config& compute_config, const m2::DFBSpecName& dfb) {   // :58
    compute_config.unpack_modes.emplace(dfb, tt::tt_metal::UnpackMode::UnpackToSrc);
}
inline m2::ComputeGen1Config& gen1_compute_config(m2::ComputeHardwareConfig& config) { … }         // :73
```

`gen1_compute_config` is worth copying too: it resolves the Gen1 alternative behind a `TT_FATAL` instead of
letting `std::get` raise `std::bad_variant_access` on a non-Gen1 device.

**That header belongs to a peer op, so do not `#include` it.** It sits inside
`normalization/layernorm_distributed/device/`, which is outside this op's directory — including it would
create exactly the cross-op coupling the scope boundary forbids, and it is not a shared-pool header. Take it
as a **pattern to reproduce locally** in this op's own factory files (both factories can share one small
file-local helper), and cite the *idea*, not the path, in any comment you leave.

**Reference:** Team thread — explicit listing over layernorm_distributed-style auto-fill; Frank's revised
explicit approach is the agreed pattern.

### ⚠ 2. GammaBetaGrad references a CB the factory never allocates — `#ifdef` it out

> **Invoker decision (D1): leave as-is.** `#ifdef`-gate the dead `cb_mask_w` (`c_6`) path in the compute
> kernel. Do **not** allocate `c_6`, and do **not** enable groupnorm in the factory. Preserve current
> behavior — zero functional change. This is settled; do not revisit it mid-port.

The GammaBetaGrad compute kernel declares and FIFO-touches a mask-w buffer:

```cpp
constexpr auto cb_mask_w = tt::CBIndex::c_6;
DataflowBuffer dfb_mask_w_obj(cb_mask_w);                    // …_gamma_beta_grad_kernel.cpp:32-33
if (do_mask_w) { dfb_mask_w_obj.wait_front(onetile); }        // :81-83
       copy_tile(cb_mask_w, 0, dst1);                         // :116, :191
if (do_mask_w) { dfb_mask_w_obj.pop_front(onetile); }         // :341-343
```

**The factory never allocates `c_6`** — its CB list is `c_0`–`c_5`, `c_16`, `c_17`, `c_24`–`c_29`
(`…gamma_beta_grad_program_factory.cpp:125-138`), with no `c_6` push under any condition. The path is dead
only because `do_mask_w = (origin_W % TILE_W) != 0 && is_groupnorm` (`…_kernel.cpp:68`) and the factory
hardwires `const bool is_groupnorm = false` (`…gamma_beta_grad_program_factory.cpp:50`).

**Without a gate the kernel will not compile** — `dfb::mask_w` will not exist, and a `constexpr bool
do_mask_w = false` does not stop name lookup at file scope. So: emit no define, and `#ifdef`-gate the
declaration **and every expression referencing it**. Zero functional change (the path is already dead).

**Do not delete the mask-w path outright** — that is kernel-logic surgery, off-whitelist, and it discards
groupnorm scaffolding the kernel was written for.

**A related trap on `c_5`.** The factory and kernel compute `do_mask_h` from *different* expressions:

| | expression |
|---|---|
| factory (`…gamma_beta_grad_program_factory.cpp:55`) | `(origin_H % TILE_HEIGHT) != 0 && is_lastdim_layer_norm` |
| kernel (`…_gamma_beta_grad_kernel.cpp:64`) | `(origin_H % TILE_H) != 0 && (is_lastdim_layernorm \|\| is_groupnorm)` |

They agree only because `is_groupnorm` is false. `c_5` **is** conditionally allocated
(`in5_t = do_mask_h ? 1 : 0`, `:92`), so gate it on the **factory's** condition — the one that decides
whether the DFB exists — not on a re-derivation of the kernel's.

### 3. Runtime source selection — InputGrad, and the DFB set moves with it

InputGrad selects **two** of its three kernel roles at runtime, on an L1-capacity computation:

```cpp
const uint32_t cb_usage = …;                                 // …input_grad_program_factory.cpp:123-125
const bool use_large_algorithm = cb_usage >= available_L1;     // :128
```
reader → `_large` / `_small` (`:205-209`), compute → `_large_kernel` / `_small_kernel` (`:230-234`).

**The algorithm also changes the DFB set and the name→DFB mapping** — do not build one DFB list and branch
only the `KernelSpec`s:

| | small | large |
|---|---|---|
| `dycopy` (c_24) / `y` (c_25) entries | `num_inner` each | **1** each (`:132-133`) |
| `c_31` | 1 entry | **0 → not allocated** (`:134`; `push_cb` skips zero-size CBs, `:141-145`) |
| tmp1 / tmp2 / tmp3 | c_29 / c_30 / c_31 | c_28 / c_29 / c_30 |
| `recip_nrstd` | its own CB, c_28 | **aliased onto tmp3** (`…large_kernel.cpp:335`) |

So the legacy branch on `use_large_algorithm` stays, and it drives the `DataflowBufferSpec` list as well as
the kernel sources. `c_31` needs no `#ifdef` — it simply is not named by the large kernel, and each source
gets its own `KernelSpec` bindings.

Expect a long stretch with no green build: the factory and all four selectable sources flip together.

### 4. Same-FIFO aliasing at scale — InputGrad compute kernels

Both InputGrad compute kernels give the three tmp DFBs a semantic name per use-phase via `constexpr auto`
aliases, **and construct a separate `DataflowBuffer` object for each alias**:

**small** (tmp1 = c_29, tmp2 = c_30, tmp3 = c_31):

| DFB | names | alias sites |
|---|---|---|
| c_29 | `tmp1`, `dyadd`, `ndy` | `:58`, `:256-257`, `:368-369` |
| c_30 | `xmm`, `ydy`, `ndymdysum` | `:113-114`, `:298-299`, `:385-386` |
| c_31 | `ydyadd`, `yydysum` | `:300-301`, `:409-410` |

**large** (tmp1 = c_28, tmp2 = c_29, tmp3 = c_30):

| DFB | names | alias sites |
|---|---|---|
| c_28 | `dyadd`, `ndy`, `xmm`, `yydysum` | `:87-88`, `:443-444`, `:486-487`, `:548-549` |
| c_29 | `ydyadd`, `ndymdysum` | `:89-90`, `:462-463` |
| c_30 | `xmm`, `ydy`, `recip_nrstd` | `:94-95`, `:268-269`, `:335-336` |
| c_25 (`y`) | also `tmp4` | `:572-573` |

This is **same-FIFO aliasing** — one DFB, several names, shared FIFO pointers. **Do not model it with
`advanced_options.alias_with`**: that would create independent FIFOs at one address and silently lose the
pointer coherence the kernel depends on.

Keep **one `DataflowBufferSpec` and one `DFBBinding` per real DFB**, express each extra name as a
`constexpr auto` handle alias, and **consolidate the objects** — one `DataflowBuffer` per DFB. Constructing
several from the same handle compiles and runs but breaks the object↔DFB identity device-side debug tooling
relies on.

Watch `cb_tmp4 = cb_y` in the large kernel: it aliases the **y** intermediate (c_25), not a tmp.

### 5. Preserved multiplicity — both factories

Each factory emits **two compute `KernelDescriptor`s** over the work-split core groups, differing only in the
per-group count CTA:

| Factory | varying CTA | sites |
|---|---|---|
| InputGrad | `num_rows_per_core_group_{1,2}` | `:236-254`, `:256-277` |
| GammaBetaGrad | `num_cols_per_core_group_{1,2}` | `:193-213`, `:215-238` |

→ **two compute `KernelSpec`s of the same source in two `WorkUnitSpec`s.** Reader and writer are on
`all_cores`, so they belong to **both** work units and their derived node set is the union:

```
wu_g1: {READER, WRITER, COMPUTE_G1}  target_nodes = core_group_1
wu_g2: {READER, WRITER, COMPUTE_G2}  target_nodes = core_group_2
```

Keep the second group conditional on `has_core_group_2`, as legacy does. **Do not** collapse to one compute
spec by demoting the per-group count to an RTA — documented anti-pattern, and it costs compile-time loop
unrolling.

Both factories cover only the assigned cores (the per-core loop `TT_THROW`s otherwise —
`…input_grad_program_factory.cpp:305`, `…gamma_beta_grad_program_factory.cpp:263`), so there is **no**
idle-core RTA padding to reproduce.

### 6. Tensor bindings — 11, all Case 1, 3 conditional

All feed a `TensorAccessor` → mechanical Case 1. **No Case 2 anywhere**, and the compute kernels construct no
`TensorAccessor`, so the blocked compute-kernel Case-2 path cannot arise. All arrive as `Buffer*` RTAs today
(the factories comment on why — `…input_grad_program_factory.cpp:287-289`,
`…gamma_beta_grad_program_factory.cpp:248-252`); the address slot disappears from every kernel.

**InputGrad** — 6: `output_grad`, `input`, `mean`, `rstd`, **`gamma` (conditional)** on the reader;
`input_grad` (`tensor_return_value`) on the writer.

**GammaBetaGrad** — 5: `output_grad`, `input`, `mean`, `rstd` on the reader;
**`gamma_grad` and `beta_grad` (both conditional)** on the writer.

`GammaBetaGrad::tensor_return_value_t` is `std::vector<std::optional<Tensor>>` — a **fixed-length-2** vector
of optionals, so this is ordinary conditional-binding work, not a variadic case. The composite guarantees at
least one is present (`moreh_layer_norm_backward.cpp:28-30`).

Each conditional binding needs a host `defines` entry and `#ifdef`-gating of the accessor construction and
its uses — mandatory for optional tensors, since there is nothing to bind when absent.

### 7. Conditional DFB specs

| Factory | Conditional DFB | Condition | Site |
|---|---|---|---|
| InputGrad | `gamma` (c_6) | `gamma_has_value` | `in6_t`, `:102` |
| InputGrad | `mask_h_w` (c_7) | `do_mask_h \|\| do_mask_w` | `in7_t`, `:103` |
| InputGrad | `tmp3` (c_31) | `!use_large_algorithm` | `im7_t`, `:116, :134` — handled by per-source bindings, no `#ifdef` |
| GammaBetaGrad | `mask_h` (c_5) | factory's `do_mask_h` | `in5_t`, `:92` |
| GammaBetaGrad | `mask_w` (c_6) | **never allocated** — see item 2 | — |

### 8. CB endpoints

Re-derive the census yourself; recorded for what to expect. **No multi-binding flag in either factory, and no
dead CB.**

- **1P+1C:** all input CBs (reader P → compute C) and all output CBs (compute P → writer C).
- **Self-loop** (compute produces *and* consumes; one toucher): GammaBetaGrad `c_24`–`c_29` (6);
  InputGrad `c_24`–`c_31` small (8) / `c_24`–`c_30` large (7).

The high self-loop count is just the compute-intermediate shape — legal on Gen1 for compute kernels.

### 9. Hardware configuration and `opt_level`

- **Style A both factories** — each destructures `get_compute_kernel_config_args`
  (`…input_grad_program_factory.cpp:91-92`, `…gamma_beta_grad_program_factory.cpp:81-82`) and sets four
  fields: `math_fidelity`, `fp32_dest_acc_en`, `dst_full_sync_en`, `math_approx_mode`. Use
  `ttnn::to_compute_hardware_config(device->arch(), config)` — minding the `math_approx_mode` bool→`Precision`
  mapping and the `dst_full_sync_en` → `double_buffer_dest` **inversion**. `bfp_pack_precision_mode` is unset
  → leave at its default.
  - **InputGrad re-resolves the config inside the factory** (`:33-34`,
    `init_device_compute_kernel_config(arch, operation_attributes.compute_kernel_config)`). Translate the
    **re-resolved** value, not the raw attribute. GammaBetaGrad does not do this — don't "harmonize" them.
- **DM configs are plain defaults** — use `ttnn::create_reader_datamovement_config(device->arch())` /
  `create_writer_datamovement_config(device->arch())`.
- **`opt_level`: explicit `O3` on every compute `KernelSpec` — four sites** (two core groups × two factories).
  Nothing in the directory sets `opt_level`, so legacy resolves to `O3` for compute and `O2` for DM; Metal 2.0
  defaults to `O2` for both.

### 10. Runtime args — all named, no varargs

Every kernel reads each arg once as a distinct field at a literal index. After the address slots drop:
InputGrad reader 10 named / writer 3; GammaBetaGrad reader 8 / writer 2. Both compute kernels take their
work counts as **CTAs**, not RTAs, so they need no `KernelRunArgs` entry.

Both legacy loops are node-first — keep them and let `AddRuntimeArgsForNode` transpose into the name-first
table. Do not re-architect into name-first form as part of this port.

## Watch for

- **CB endpoints (multi-binding):** none. Neither factory declares a semaphore, so there is no
  semaphore-gated raw co-fill to miss.
- **Cross-op / shared kernels:** **none — no `_metal2` fork needed.** All eight sources live in this
  directory, bound only by these two factories. Coupling is function-call escape via three donor headers, all
  crossing cleanly:
  - `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` and `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` —
    `DataflowBuffer`-taking helpers (native shape), including the `*_init_with_dt(DataflowBuffer)` family the
    compute kernels lean on heavily. Pass the objects you already construct.
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` — `compute_kernel_lib::reduce` (`:392`) takes cb ids
    as **non-type template parameters**. `dfb::name` works there (the conversion is `constexpr`).

  None is yours to edit. If you need to, stop and record a Handoff point.
- **`read_mean_rstd` is an in-file helper, not a donor** — defined identically at line 12 of all three
  reader kernels, taking `uint32_t cb_id` and building a `DataflowBuffer` internally. Simplest treatment:
  pass `dfb::mean` / `dfb::rstd` through the implicit conversion, leaving the signature alone. **Do not
  hoist the triplicated definition** into the shared pool — out-of-scope refactor.
- **`get_tile_size(cb_id)` → `dfb.get_tile_size()`** — 14 sites across 5 kernels. Sanctioned Device 2.0 free
  functions (which is why the gate passed), but the cb id is gone in Metal 2.0, so rule 7 applies: query the
  object; don't extract `.id`.
- **The GammaBetaGrad pack-target ternary needs NO gate.**
  `constexpr auto cb_out_init = gamma_grad_has_value ? cb_dgamma : cb_dbeta;` (`…_gamma_beta_grad_kernel.cpp:73`)
  looks like the shape that requires `#ifdef`-ing, but **both `c_16` and `c_17` are allocated
  unconditionally** (`out0_t = out1_t = 1`, `…gamma_beta_grad_program_factory.cpp:94-95, 131-132`), so both
  DFBs always exist. Only the *tensor* bindings are conditional.
- **Do not "fix" the pre-existing anomalies.** `packer_l1_acc` is destructured and dropped in both factories
  (Metal 2.0 has no field for it either — nothing to carry); `is_groupnorm` is hardwired `false` while all
  three compute kernels carry live `is_groupnorm` branches (settled by decision D1 — leave it);
  and the two `log_info(tt::LogTest, …)` calls on the algorithm-selection path
  (`…input_grad_program_factory.cpp:131, :136`) fire on every cache miss at the wrong severity — but the large
  one is now **your `_large`-path verification signal**, so leaving it is not merely permitted but required
  (see *Test gate* below). Route all of these to the report, none to the diff.
- **Comments are load-bearing and dense here.** The CB declaration blocks carry the algebra each buffer holds
  (`// Sum[dy]`, `// Sum[y * dy]`, `// rstd / n`, `// x - mean`, `// copy output_grad(==dycopy)`) in both the
  factories and the kernels, and the `// comes from the reader` annotations mark cross-kernel handoffs. When a
  CB-index line goes away, **relocate** the role comment onto the `DFBBinding` or the `DataflowBuffer`
  construction. The alias blocks in the InputGrad compute kernels are the highest-risk spot — you are
  consolidating objects there, and the per-phase name is often the only documentation of what the buffer holds
  at that point.
- While converting kernel-by-kernel, `-k`-exclude paths whose source you have not converted yet: a
  selected-but-unconverted kernel hits a positional `get_compile_time_arg_val(0)` that `static_assert`s at
  JIT and can take down the whole pytest session (exit 139) rather than failing cleanly.

## Test gate — decided by the invoker, no confirmation needed

**Run every backward-relevant test, not a subset. The backward tests are the gate.** Forward tests in the
same file are fine to run for regression but are not the gate.

Coverage is **nightly-only**, under the `moreh/` slug, in a file shared with the forward op:
`tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py`.

Run in this order:

| # | Test | Line | Why |
|---|---|---|---|
| 1 | `test_moreh_layer_norm_backward_compute_kernel_options` | `:608` | **Run first.** Sweeps compute-kernel options → covers `fp32_dest_acc_en` both ways, hence the `unpack_modes` work of item 1 |
| 2 | `test_moreh_layer_norm_backward_with_gamma_or_beta` | `:542` | Conditional gamma_grad / beta_grad bindings (items 6–7) |
| 3 | `test_moreh_layer_norm_backward_callback` | `:680` | Program-cache hit path / `UpdateTensorArgs` |
| 4 | `test_moreh_layer_norm_backward` | `:508` | Primary functional coverage |
| 5 | `test_moreh_layer_norm_backward_rejects_invalid_mean_volume` | `:695` | Validation |
| 6 | `test_moreh_layer_norm_backward_rejects_same_volume_wrong_mean_shape` | `:729` | Validation |

**There is nothing else in the repo to run.** Swept and confirmed: `grep -rln
'layer_norm_backward\|layernorm_backward' tests/ models/` returns only this file plus
`tests/sweep_framework/Allops.txt` (a plain name list, not a test); there are **no** C++ gtests; and the
three `layer_norm`-named sweeps exercise the **non-moreh** `ttnn.layer_norm`, not this op. No non-nightly
variant exists. So you do not need to go hunting — items 1–6 plus the file's forward tests are the whole
available surface.

### `_large` algorithm coverage — local-only, do not commit

The committed parametrizations almost certainly reach the **small** path only, and both selected sources must
convert together, so `_large` is the specific uncovered risk on this port. Cover it locally:

1. Write a **local-only** test with a deliberately large shape that forces `use_large_algorithm`
   (`cb_usage >= available_L1`, `…input_grad_program_factory.cpp:128`).
2. Run it and confirm the log line
   **`"Large moreh_layer_norm_backward_input_grad algorithm is selected."`**
   (emitted at `…input_grad_program_factory.cpp:131`).
3. **Do not commit or push that file.** It is a verification scaffold, not a deliverable — keep it out of the
   PR entirely.
4. Record the outcome in `METAL2_PORT_REPORT.md` under test-coverage notes: the shape you used, whether the
   large path was confirmed selected, and whether the ported `_large` kernels passed.

> **Corollary — do not tidy the algorithm-selection `log_info` calls.** The audit flags the two
> `log_info(tt::LogTest, …)` lines (`…input_grad_program_factory.cpp:131`, `:136`) as noise at the wrong
> severity. **Leave them exactly as they are:** the large one is your only signal that the `_large` path was
> actually exercised, because no committed test reaches it. Route the observation to the report, not to the
> diff.

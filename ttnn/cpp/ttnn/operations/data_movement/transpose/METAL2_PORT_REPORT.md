# Port Report — transpose (data_movement)

## Outcome

`PORTED` — the six clean factories (`TransposeCNProgramFactory`, `TransposeHCRMProgramFactory`,
`TransposeHCTiledInterleavedProgramFactory`, `TransposeHCTiledProgramFactory`,
`TransposeWHProgramFactory` [tiled **and** row-major], `TransposeWHShardedProgramFactory`) converted to
`ProgramSpecFactoryConcept`, built, and verified on device at exact parity with the pre-port baseline.
The two gated factories (`TransposeHCShardedProgramFactory`, `TransposeWHShardedRMProgramFactory`) are
untouched on the legacy descriptor path, as the audit scoped.

## Verification

Wormhole n150. Build via `./build_metal.sh --build-tests`; the Metal 2.0 legality checks were forced on
in all 9 `skip_validation` sites and **proven live in the binary under test** — both markers present
(`program_spec.cpp:2847`, `program_run_args.cpp:502`), firing 1656 times across the post-port run. The
forcing scaffolding was reverted before commit and is absent from the diff.

| | passed | failed | skipped |
|---|---|---|---|
| **Baseline** (pre-port, before any kernel edit) | 1075 | **0** | 75 |
| **Post-port** | 1075 | **0** | 75 |

Confirmed test set (agreed with the invoker before it was relied on) — transpose's own coverage plus
every peer op affected by a shared kernel this port touched or reused:
`misc/test_transpose.py`, `base_functionality/test_reshape_transpose.py`,
`sweep_tests/tt_dnn/test_transpose.py`, `sweep_tests/tt_dnn/test_permute.py`,
`misc/test_create_qkv_heads.py`, `misc/test_nlp_create_qkv_heads{,_boltz,_vit}.py`,
`operations/transformers/test_transformer.py`.
The peer-op files are in the set deliberately: `transpose_wh.cpp` and `transpose_wh_sharded.cpp` are
lent to the qkv-heads ops, and permute owns two of the forks this port reuses.

**Factory routing verified in the built library**, not inferred from the source: all six ported
factories resolve to `ProgramSpecMeshWorkloadFactoryAdapter<...>` in
`nm -C build_Release/lib/_ttnncpp.so`, and both gated factories resolve to neither adapter template.

### Anti-pattern self-audit

Each sweep reported as *hits / files scanned*; the op directory holds **47** `.cpp`/`.hpp` files and the
diff-scoped sweeps saw **27**.

| check | result |
|---|---|
| Buffer address / `Buffer*` in run-args | **0** / 47 |
| `CBDescriptor` / `CircularBuffer` / `.cbs` / `ProgramDescriptor` in ported factories | **0** / 47 |
| `TensorAccessorArgs` in ported kernels | **0** (2 survivors, both unbound legacy files — see Open items) |
| Positional `compile_time_args` | **0** in ported factories (2 hits, both in the untouched gated factory) |
| `get_vararg` / `compile_time_varargs` | **0** / 47 |
| `.id` extraction at LLK call sites | **0** / 47 |
| `allow_instance_multi_binding` | **0** / 47 — re-derived from the kernel-touch census; every DFB is 1P+1C or a single-toucher self-loop |
| `TT_FATAL` / `TT_ASSERT` / `TT_THROW` census vs `$BASE` | **no delta** — every guard accounted for |
| Ephemeral `.md` cited from code | **0** / 27 |
| Forced-legality scaffolding in diff | **0** — no `tt_metal/impl` file, no marker string |
| `opt_level` per compute `KernelSpec` | 2 compute specs exist (`transpose_wh`, `transpose_wh_sharded`); **both** set `O3` explicitly; the 4 DM-only factories correctly set none |

**`cb`-name sweep — 8 hits, all attributed, none a leftover.** Six are `accessor_name` strings (and two
comments naming them) that are **owned by reused `_metal2` forks** and therefore not renameable from
this op: `cb_in0` / `cb_pad` (permute's padding-aware reader) and `cb_in` / `cb_out` (permute's
`transpose_wh_metal2.cpp`). The remaining hits are in `transpose.cpp`, op-level host code this port did
not touch. The one fork this port **authored** (`transpose_wh_sharded_metal2.cpp`) uses clean
`dfb::in` / `dfb::out`. See the Friction entry on this sweep's tension with the fork-vocabulary rule.

### Risk called out in the plan, and how it resolved

The plan flagged the WH-Sharded borrowed input buffer as the port's most likely failure point: legacy
allocated it on the shard grid (`all_cores`) while all three kernels run on the full grid
(`total_cores`), and a Metal 2.0 `DataflowBufferSpec` has no `core_ranges` — placement is derived from
the bindings, so the borrowed buffer is placed across `total_cores`. It resolved cleanly with no spec
change: the validator accepted it and the sharded tests pass at baseline parity. Recorded because the
reasoning is non-obvious and the next porter of a borrowed-DFB-on-a-subgrid op will hit the same
question.

## Provenance

- **Recipe docs (this port):** `c1c4d4eceb0 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources`
  *(the recipe docs live only on the doc branch `vsureshTT/metal2-data_movement/transpose`; they are
  not present on `origin/main`, so the command in the recipe prints nothing from a main-based checkout.
  The line above is that branch's tip for the `metal_2.0/` doc tree.)*
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

**Port base:** `origin/main` @ `f6b36f3b1be` (2026-08-25). Baseline established before the first kernel
edit: **1075 passed / 0 failed / 75 skipped** over the confirmed test set (see Verification below).

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept` on all six ported factories — matching the audit's
  decision, re-confirmed against current `main` (none of the six carries an `override_runtime_arguments`,
  so the custom concept does not apply and the recipe's translation step was skipped).
  Verified in the built library rather than by reading: every one of the six resolves to
  `ProgramSpecMeshWorkloadFactoryAdapter<...>` in `nm -C build_Release/lib/_ttnncpp.so`, and both gated
  factories resolve to neither adapter template (they keep the descriptor shim).
- **Custom `compute_program_hash`:** none — the op uses the default reflection hash, and it was not
  touched. No `attribute_values` / `to_hash` backdoor either.
- **Pybind entry points removed:** none. The nanobind module exposes only the `transpose` free
  function, never `create_descriptor`, so none of the three device-op-class exceptions applied and
  `transpose_device_operation.{hpp,cpp}`, `transpose.cpp` and `transpose_nanobind.cpp` are byte-identical
  to `main`.
- **`program_factory_t` variant:** left as-is. It now holds a **mix** — six factories on the Metal 2.0
  concept and two on the legacy descriptor concept — which `std::visit` dispatches per factory. This
  worked with no framework friction and is the thing that made a subset port shippable.
- **Open items:** none blocking. The two gated factories re-audit separately (see Handoff point 1).

## Handoff points

### 1. Audit is stale on the gate rationale for the two blocked factories — and their target concept has changed

*(→ readiness-sheet owner / TTNN PD-migration team)*

The audit (2026-08-04) blocks `TransposeHCShardedProgramFactory` and `TransposeWHShardedRMProgramFactory`
on two conjuncts, the first being `Runtime-args update == yes` via the device-op `get_dynamic_runtime_args`
hook "declared at `transpose_hc_sharded_program_factory.cpp:432`".

**That hook no longer exists in transpose.** Commit `383674438e5` ("Port transpose off
`get_dynamic_runtime_args` onto `override_runtime_arguments`", #52566) removed it; `grep -rn
get_dynamic_runtime_args ttnn/cpp/ttnn/operations/data_movement/transpose/` returns nothing on current main.

Two consequences:

- **For this port: none.** `override_runtime_arguments` is declared only on those same two factories
  (`transpose_hc_sharded_program_factory.hpp:19`, `transpose_wh_sharded_rm_program_factory.hpp:19`), so the
  six in-scope factories are unaffected and remain on the base `ProgramSpecFactoryConcept`.
- **For the gated two, when their remaining gate clears:** they now carry an `override_runtime_arguments`
  and therefore route to **`CustomProgramSpecFactoryConcept`**, not the `MetalV2FactoryConcept` (base) that
  the audit recorded as their target. Whoever re-audits them needs the recipe's
  "Translating `override_runtime_arguments`" step, which the current brief does not mention.

The second conjunct (`Is safe to port? = no`, the readiness-sheet owner's correctness call) is unchanged
and remains the live blocker. Not a porter-resolvable item; recorded so the sheet row can be refreshed.

## Successes

- **The shared-kernel Caution fired exactly as designed, on a case the brief had not listed.**
  `transpose_wh_rm.cpp` looks transpose-private by path, and the instinct on a "convert the op's own
  kernels in place" pass is to convert it. The recipe's *intra-op* shared-kernel bullet
  (§Read this first, "Shared top-level entry point") is what prompted the factory×kernel overlap check
  that found the gated factory compiling the same source with `SHARDED`. Cost of the check: one script.
  Cost of skipping it: a JIT-time `'args' has not been declared` on the sharded-RM path, discovered
  after the port looked green. Full write-up as Handoff point 4.

- **The `opt_level` section's "grep it, don't eyeball it" instruction caught a real silent regression.**
  Both compute-bearing factories (`transpose_wh`, `transpose_wh_sharded`) build their compute kernel
  from a `ComputeConfigDescriptor` that never mentions `opt_level` — so nothing in the legacy source
  reads as a setting, and nothing in the ported spec reads as missing. The resolved legacy level is
  **O3**; a `KernelSpec` that stays silent gets **O2** (`kernel_spec.hpp:116`). Both now set
  `KernelBuildOptLevel::O3` explicitly. This is a pure perf delta with no build, validator or test
  signal — the section's framing of it as "an absent line, not a wrong value" is what made it findable.

- **Forcing and *proving* the legality checks was worth the extra step.** The markers confirmed live
  in both translation units (`program_spec.cpp:2847`, `program_run_args.cpp:502`) and fired ~960 times
  across the baseline run, so every green result in this port was measured with validation on rather
  than assumed to be.

- **The header-first rule beat precedent-hunting.** `kernel_spec.hpp` settled in one read that
  `DFBBinding::accessor_name` is *per binding*, not per DFB — which is what makes it legal for the
  reused forks to keep their own token names while this op's own kernels use clean ones. Inferring that
  from another port's code would have been guesswork.

## Friction

### Gaps

- **The unity build makes anonymous-namespace spec constants collide across an op's factories, and the
  recipe does not warn about it.** The natural way to write spec names — `const DFBSpecName IN0{"in0"};`
  in the factory's anonymous namespace, as the migration guide's examples suggest — compiles fine per
  file and then fails the moment a second factory in the same op does the same thing, because the op's
  factory `.cpp`s are concatenated into one translation unit:
  `error: redefinition of 'IN0'`. Every one of this op's six factories wanted the same handful of names
  (`IN0`, `INPUT`, `OUTPUT`, `READER`, `WRITER`). Resolved by declaring the constants **function-locally**
  inside `create_program_artifacts`, which is unambiguous and needs no per-factory prefixes. Worth one
  line in the Construct section, since it hits every multi-factory op and the error arrives at the first
  build after the *second* factory converts — far from the code that caused it.

- **No guidance on what to do with a legacy CTA/RTA the kernel never reads.** This op had four
  (the CN reader's `C`, the HC-RM reader's `N` and both kernels' `aligned_page_size`), all of them
  values the host computes and ships to a slot the kernel's `TensorAccessorArgs<N>` boundary sits past.
  They are not "dropped plumbing" (no Metal 2.0 primitive replaces them) and not a dead CB (the recipe's
  one worked example of removing something inert). The choice matters because a named arg the kernel
  never references leaves a host/kernel arg-set mismatch, which the Build section elsewhere tells you to
  "reconcile". Resolved by emitting exactly what each kernel reads and reporting the rest as findings —
  the same reasoning the dead-CB disposition uses (an unread value has no behavior). A sentence
  extending the dead-CB rule to dead scalars would settle it.

### Confusion

- **"Reuse the existing `_metal2` fork" leaves the legacy original unreferenced, and the recipe does not
  say whether that is the porter's problem.** After repointing the HC-Tiled-Interleaved reader at
  permute's fork, `reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` is bound by nothing in
  the repo. Deleting it is outside a syntax-swap port's remit and risks a binding by a string this sweep
  did not find; leaving it ships dead code. Left in place and recorded under Open items. The
  shared-kernel Caution covers "which rung to take" thoroughly but is silent on the leftover.

- **The `cb`-name sweep and the shared-fork vocabulary rule pull in opposite directions, and neither
  cites the other.** The self-audit says expect **zero** `[Cc][Bb]_` hits across the op directory and
  calls every hit "a real leftover". But three of the forks this port is told to reuse own token names
  like `dfb::cb_in0` / `dfb::cb_pad` / `dfb::cb_in`, and conforming to them is mandatory — so a
  correctly-executed port cannot reach zero. The sweep result for this op is documented under
  Verification with each surviving hit attributed. Suggested fix: have the `cb`-sweep item say that hits
  which are *accessor names owned by a reused fork* are expected, and must be attributed rather than
  renamed.

## Open items for downstream

### Leftover from a rung-1 fork reuse

`device/kernels/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` (the legacy
original) is, after this port, **bound by nothing in the repo**: transpose's HC-Tiled-Interleaved
factory was its last consumer and now binds permute's `_metal2` fork instead. It is a deletion
candidate for the ops team. Not deleted here — removing a kernel source is outside a syntax-swap port,
and a binding by some string this sweep did not match would fail only at JIT.

The same directory already contains one pre-existing unreferenced kernel,
`device/kernels/dataflow/reader_unary_transpose_wh_interleaved.cpp` (the audit flagged it; bound by no
factory before this port either). Both could go in one cleanup.

### Findings — bugs / dead plumbing found while reading, deliberately NOT fixed

Per the port's scope discipline these are preserved byte-for-behavior and reported rather than repaired.

1. **Dead runtime arg — CN reader.** `kernels/dataflow/reader_unary_transpose_cn_interleaved_start_id.cpp`
   reads `C` (legacy RTA slot 2) and never uses it; the host computes and ships it for every core
   (`transpose_cn_program_factory.cpp`, reader `emplace_runtime_args`). Preserved as a named RTA `C`.
   Fix would be: drop the arg on both sides.

2. **Dead compile-time args — HC-RM.** Three CTA slots are emitted by the host and never read by the
   kernels, because each kernel's `TensorAccessorArgs<N>` boundary skips past them:
   - reader slot 0 (`N`) and slot 4 (`src0_buffer->aligned_page_size()`) — the kernel's accessor args
     start at `TensorAccessorArgs<5>()` and it reads only slots 1–3 (`H`, `C`, `stick_size`).
   - writer slot 2 (`dst_buffer->aligned_page_size()`) — accessor args start at `TensorAccessorArgs<3>()`
     and it reads only slots 0–1.

   Preserved as named CTAs. Fix would be: drop all three on both sides.

3. **Dead CB `c_25` (im2) — WH row-major path.** Allocated at `transpose_wh_program_factory.cpp` in the
   row-major branch, carrying an existing `// TODO REMOVE`. Referenced by no kernel this factory binds:
   `transpose_wh_rm.cpp` uses `c_25` as `cb_tilize` only under `#ifdef SHARDED`, a define set solely by
   the *gated* `TransposeWHShardedRMProgramFactory`. **This one the port does act on** — a dead CB has no
   behavior and a bindingless DFB is rejected by the validator, so the allocation is dropped (the
   recipe's dead-CB disposition), not merely reported. Recorded here because it is a real L1 saving the
   ops team may want to mirror into the gated factory's own path when that one ports.

### Shared kernel touches

Established by `grep -rl <filename> ttnn/cpp/ttnn/operations/` plus a factory-level overlap check
between the six ported factories and the two gated ones. The `experimental/quasar/` tree was excluded
from consideration and not read.

**Rung 1 — reused an existing `_metal2` fork (no new file):**

| fork reused | owner / other consumers | vocabulary this port had to conform to |
|---|---|---|
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` | eltwise/unary; ~26 host files bind the legacy original | `dfb::out`, `tensor::dst`, `args::num_pages`, `args::start_id` |
| `eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp` | eltwise/unary; ~11 host files bind the legacy original | `dfb::in`, `args::num_tiles_per_core` |
| `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp` | data_movement/sharded; ~11 host files bind the legacy original | `dfb::out`, `args::num_units` |
| `transpose/device/kernels/compute/transpose_wh_metal2.cpp` | created by the **permute** port; legacy original still bound by `nlp_create_qkv_heads{,_boltz,_vit}` and `split_query_key_value_and_split_heads` | `dfb::cb_in`, `dfb::cb_out`, `args::NHtWt` |
| `transpose/device/kernels/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware_metal2.cpp` | created by the **permute** port | `dfb::cb_in0`, `dfb::cb_pad`, `tensor::input`, `NEEDS_PADDING` define |

**Rung 2 — created the fork (pointer comment landed in the legacy original):**

| new fork | legacy original's remaining consumers |
|---|---|
| `transpose/device/kernels/compute/transpose_wh_sharded_metal2.cpp` | `create_qkv_heads`, `create_qkv_heads_from_separate_tensors`, `split_query_key_value_and_split_heads_sharded` — all still on the legacy host API |
| `transpose/device/kernels/compute/transpose_wh_rm_metal2.cpp` | **`TransposeWHShardedRMProgramFactory` — this op's own gated factory** (see the finding below) |

**In-place conversions** (verified transpose-only, bound by no peer op and by no gated factory):
the CN reader/writer, the HC-RM reader/writer, the HC-Tiled reader, the HC-Tiled-Interleaved writer,
and the three WH interleaved dataflow kernels.

### 4. The brief's shared-kernel list missed an intra-op collision — `transpose_wh_rm.cpp`

*(→ audit / brief authors; recipe's shared-kernel section)*

The brief's "Cross-op / shared kernels" heads-up lists three **borrowed** kernels (the eltwise/unary and
sharded donors) and nothing else. It does not flag `device/kernels/compute/transpose_wh_rm.cpp`, which is
bound by **two factories on opposite sides of this port's scope line**:

- `TransposeWHProgramFactory` (in scope) — compiles it **without** `SHARDED`
- `TransposeWHShardedRMProgramFactory` (**gated**, stays on the legacy descriptor path) —
  compiles it **with** `SHARDED=1` (`transpose_wh_sharded_rm_program_factory.cpp:222,236`)

Converting it in place would have compiled Metal 2.0 `dfb::` / `args::` tokens into a kernel that a
legacy `CreateKernel` path still builds, which emits no generated headers — the exact
`'args' has not been declared` breakage the shared-kernel Caution exists to prevent, and it would have
surfaced only at JIT time on the sharded-RM tests. Handled with rung 2: `transpose_wh_rm_metal2.cpp`
carries the non-sharded path only, the legacy copy keeps the sharded path and gains a pointer comment.

**Why the brief missed it:** its shared-kernel scan looks for kernels *outside* the op directory
(borrowed) and for peer ops binding kernels *inside* it (lent). This one is neither — it is shared
between two factories of the *same* op, separated only by a `#define`. The recipe does name this
"intra-op" case, but the brief has no field for it, so a porter working from the brief alone would not
see it. **Suggested audit change:** have the audit emit a per-op factory×kernel matrix and flag any
kernel bound by both an in-scope and an out-of-scope factory.

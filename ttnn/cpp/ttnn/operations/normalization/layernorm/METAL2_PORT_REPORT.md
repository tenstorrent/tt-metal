# Metal 2.0 Port Report — layernorm (`LayerNormDeviceOperation`)

**Outcome: grounded stop (successful capitulation).** Scope-tight inventory + plan completed
for the multi-core (interleaved) factory; **no code changed.** The factory's atomic port unit
(factory body + all 10 runtime-selectable kernel sources, coupled through one shared
`named_compile_time_args` table) exceeds a faithful one-pass port, and no shippable sub-factory
subset exists. This is the recipe's
[explicitly-sanctioned grounded stop](port_op_to_metal2_recipe.md#when-the-discipline-doesnt-fit)
for an over-large single-factory unit — same success tier as a finished port.

Build: not run (no code changed). Tests: not run.

See `METAL2_PORT_PLAN.md` for the full multi-core inventory, construction blueprint, and the
stop rationale ([§Grounded stop](METAL2_PORT_PLAN.md)).

---

## Successful failure (the grounded stop)

- **Op / factory:** `ttnn/cpp/ttnn/operations/normalization/layernorm/`, both
  `LayerNormMultiCoreProgramFactory` (`device/layernorm_op_multi_core.cpp`) and
  `LayerNormShardedProgramFactory` (deferred, larger).
- **Why a one-pass faithful port is not achievable:**
  - The multi-core factory selects its kernel **source file** at runtime (reader 4-way
    `.cpp:484-497`, writer 2-way `:633-637`, compute 4-way `:541-549`) → **10 selectable
    sources, ~3,200 kernel lines**. Per the recipe's atomic-unit rule, all flip together; the
    coupling is concrete: **one `named_compile_time_args` table** (`cb_named_args`, `.cpp:448-481`,
    24 CB entries + 4 alias-flag scalars) is shared verbatim across all three
    `KernelDescriptor`s, several entries conditionally remapped.
  - **No shippable subset:** `create_program_spec` is invoked by the framework for every
    attribute combination, and the tests parametrize `use_welford=[True,False]` plus exercise
    large-tensor / row-major / fused-pre-add / gamma+beta. A factory converting only the
    default path would mis-dispatch or fail on the welford/large/rm paths the tests drive. The
    "port the common path only" sub-target the recipe warns against does not build here.
  - The sharded factory (~2,000 host lines + ~13 mcast kernels + 3 semaphores + mcast topology)
    is larger still.
- **What the off-rules change would have been:** none off-rules. The *mechanism* is fully
  covered by the catalog (Multi-variant factories, Conditional bindings, Aliased DFBs,
  Same-FIFO aliasing, DFB-handle-direct-to-LLK). The blocker is purely one-pass
  size/faithfulness budget vs. the ~30-min-per-stuck-point / single-pass posture the recipe
  sets — exactly the sanctioned grounded-stop trigger.
- **Prior art:** a prior clean-room agent reached the same conclusion on this op (recorded on
  branch `akertesz/porting-experiment-accumulation-jun10`: *"layernorm — grounded stop … the
  op's runtime kernel-source selection forces atomic conversion of factory + all variants; too
  large for a one-pass port"*). This pass independently re-derived it and added the full
  multi-core inventory + construction blueprint so the eventual multi-pass port has a running
  start.

## TTNN ProgramFactory

### Concept realized
Not realized (grounded stop). Inherited target confirmed correct: `ProgramSpecFactoryConcept`
(single-program, no op-owned device resources, strict tensor matching). No disagreement with
the audit's choice.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (none exists; the `compute_program_hash`
  static at `layernorm_nanobind.cpp:253` is a Python test hook on the framework default, not an
  override — confirms audit).
- Pybind entry points removed: **none** (no code changed). **When ported,** the port *will*
  force dropping the `core_range_set` parameter and deleting both `create_descriptor` pybind
  hooks (`layernorm_nanobind.cpp:322` multi-core, `:363` sharded) per ttnn-factory **exception
  3** — see Handoff points.

### Open items
- Relaxation candidate (do **not** apply during port): `validate_on_program_cache_miss`
  enforces strict `padded_shape` matching between input/residual and gamma/beta padded-width
  equality. A future `match_padded_shape_only`-style relaxation might widen cache equivalence —
  unverified; flagged by the audit, not a port-time call.

## Handoff points

1. **Pybind surface to remove when ported (API surface: removed entry point).** Both factories
   expose a `create_descriptor(... , core_range_set)` pybind hook returning a
   `ProgramDescriptor` (`ttnn/cpp/ttnn/operations/normalization/layernorm/layernorm_nanobind.cpp:322`
   multi-core, `:363` sharded; the `core_range_set` kwarg is declared at `:333` / `:374`). The
   Metal 2.0 `create_program_spec` signature cannot carry `core_range_set`, and the
   `ProgramDescriptor` return is exactly what the port eliminates → both hooks must be deleted
   and the production default (`default_core_range(device)`) inlined. *User-visible:* downstream
   Python (tests/notebooks/tooling) calling `...create_descriptor(...)` must be updated. Owner:
   layernorm op owner. Not done here (no code changed) — flagged for the porting PR.

2. **Device 2.0 `get_tile_size(cb_id)` holdovers (do NOT absorb into the port).** The audit
   tables ~13 `get_tile_size(cb_id)` call sites across the layernorm dataflow kernels (e.g.
   `reader_unary_interleaved_ln.cpp:108,114,118,122`; `writer_unary_interleaved_start_id_blocked.cpp:25`).
   These are a Device 2.0 cleanup (`cb_obj.get_tile_size()` member form), routed to the
   Device 2.0 track. During a Metal 2.0 port they cross the boundary fine via
   `dfb::name → uint32_t` implicit conversion, so they don't block — but the kernel-side
   whitelist forbids the port doing the member-form cleanup. Owner: Device 2.0 track. (See
   Friction: the two docs disagree on whether `get_tile_size` has a member form.)

3. **Dead reader RTA (op owner, not port).** `reader_unary_interleaved_ln.cpp:33` documents
   arg[4] `packed_one_value` as legacy/unused; the host still computes and passes it
   (`layernorm_op_multi_core.cpp:553-554,591`). Owner: layernorm op owner.

## Successes

- **The atomic-unit / runtime-source-selection guidance fired exactly as intended.** The
  recipe's [Legacy inventory — "Runtime kernel-source selection"](port_op_to_metal2_recipe.md#legacy-inventory)
  bullet ("the factory and **all** of its selectable sources convert together — there is no
  'port the common path only' sub-target that builds … size the effort against that") is
  precisely the lens that made the multi-core factory's true size visible and justified the
  grounded stop. Applied to `layernorm_op_multi_core.cpp:484-549`. This is a doc section worth
  keeping verbatim — it converted a tempting "just port the default path" trap into a correct
  stop.
- **The grounded-stop off-ramp framing is well-calibrated.** Knowing a clean stop is "same
  success tier as a finished port" made it possible to spend the budget on a *complete*
  inventory + construction blueprint (the high-value artifact) rather than burning it forcing
  partial code that wouldn't build. Catalog cross-refs (Aliased DFBs vs Same-FIFO aliasing
  distinction) resolved the `cb_x_welford` classification cleanly before any code was at stake.

## Friction

### Gaps
- **No guidance on a factory with a *shared* `named_compile_time_args` table across multiple
  `KernelDescriptor`s.** The recipe's spec-shape default ("one KernelSpec per KernelDescriptor,
  per-KernelSpec bindings") tells you bindings are per-kernel in Metal 2.0 — but the legacy
  layernorm factory hands *the same 24-entry `cb_named_args` table to all three kernels*
  (`layernorm_op_multi_core.cpp:625,641,652`), with entries conditionally remapped per path.
  The migration is "dissolve the shared table into per-KernelSpec `dfb_bindings`," but no doc
  section names this shared-CB-table → per-kernel-bindings dissolution as a recognizable
  pattern. It is the single biggest structural transform in this op and the main driver of the
  atomic coupling. **Suggested:** a patterns-catalog entry "Dissolving a shared named-CTA CB
  table into per-KernelSpec bindings."
- **`get_tile_size(cb_id)` member-form ambiguity (carried from the audit's own recipe-notes).**
  The audit notes the Device 2.0 migration guide keeps `get_tile_size(cb_id)` verbatim in
  *migrated* example code (no `cb.get_tile_size()` member shown), while the prereq check lists
  it as a holdover with a member-form replacement. A porter can't tell from the docs whether
  these sites are "leave as free function (crosses fine via `dfb::`)" or "route to Device 2.0
  track for member-form." This bit at inventory time. **Suggested:** reconcile the two docs and
  state explicitly that during a *Metal 2.0* port the free function is acceptable (it takes
  `dfb::name` by implicit conversion) and only the Device 2.0 track does member-form.

### Confusion
- **Audit per-binding bullets vs. per-factory classification.** The audit's flat per-binding
  bullet list classifies input/residual/output as *both* "Case 1 (interleaved)" and "clean
  borrowed-memory DFB (sharded)" for the *same* `TensorParameter` name. The audit's own
  recipe-notes already flag this; confirming from the porter's seat: the Per-DeviceOperation
  attribution table is what disambiguates, and a porter should read that table first, not the
  bullets. Minor — the audit handled it well; noting that the friction is real for the next
  porter who reads top-to-bottom.
- **"Default to porting factories one at a time, autonomously" vs. an over-large *single*
  factory.** The recipe strongly frames multi-factory ops as "port one factory, ship it, the
  rest is remaining work" — which reads as *every* op decomposes into shippable single-factory
  units. layernorm is the counterexample: even one factory's unit is too large for one pass.
  The recipe *does* cover this (the size-grounded-stop sentence in Legacy inventory), but the
  two messages sit far apart and the optimistic "one factory is shippable" framing dominates.
  **Suggested:** add a forward-reference from the atomic-unit note to the size-grounded-stop
  sentence, so a porter sizing a big factory sees the escape hatch in the same breath as the
  "you don't have to do the whole op" encouragement.

## Open items for downstream

- **Cross-op kernel touches:** none. All 10 multi-core sources + 2 shared headers are
  layernorm-owned (in-family); donor headers (`kernel_lib/`, `kernel/`, `kernel_util/`) are
  out-of-scope shared-lib/framework and Device-2.0-clean.
- **The eventual multi-pass port should reuse `METAL2_PORT_PLAN.md`'s blueprint:** the
  multi-core inventory (kernels/CBs/tensors/work-split), the Dropped-Plumbing table, the
  per-branch source-selection map, and the welford-fp32 aliasing classification are all
  construction-ready. A reasonable multi-pass split: (a) non-welford tile-input path
  (`layernorm.cpp` + `layernorm_large_tensor.cpp` + default reader/writer); (b) welford paths
  (`layernorm_welford.cpp` + `layernorm_large_tensor_welford.cpp` + welford reader, aliased
  DFBs); (c) row-major paths (`TILIZE_IN`/`UNTILIZE_OUT`, rm writer, `rm_gb` reader) — though
  note all three sub-passes must land before the factory can flip, since they share one
  `create_program_spec`. The split is for *reviewer/effort* tractability, not separate shipping.
- **Sharded factory** is a separate, larger port (semaphores, mcast, pre/post-allgather, reshard).
- **Test coverage note:** `tests/ttnn/unit_tests/operations/fused/test_layer_norm.py` (15 tests,
  `use_welford` parametrized) is the multi-core gate; sharded/distributed tests
  (`test_layer_norm_sharded.py`, `test_distributed_layernorm*.py`) gate the sharded factory.

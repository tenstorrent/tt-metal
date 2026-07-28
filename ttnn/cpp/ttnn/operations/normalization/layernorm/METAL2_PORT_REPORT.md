# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/normalization/layernorm/`

> Friction record + handoff log for the layernorm Metal 2.0 port. Opened at the
> start of the port per the recipe ("Open `METAL2_PORT_REPORT.md` at the start").
> Captured in the moment; polished at the end.

## Outcome

**Grounded stop on full-port scope; thorough plan + structural findings delivered.**
Per the recipe's [§When the discipline doesn't fit] / "a grounded stop is a complete,
valued deliverable" posture: I did the legacy inventory and spec planning in full
(see `METAL2_PORT_PLAN.md`), verified the framework precondition and the exact factory
wiring against the on-branch headers, and surfaced the structural findings below. I
did **not** land converted factory/kernel code, for two compounding reasons that I
judged make a one-pass faithful port infeasible (detail in Friction):

1. **Volume + atomicity.** Two factories (~844 + ~600 + ~1900 helper lines) and 26
   kernels, each factory selecting its kernel *source string at runtime*. Because the
   spec's DFB/tensor bindings must match the selected source, a factory and **all**
   kernel variants it can select must be converted **atomically** — there is no
   "port the common path only" sub-target that builds. The minimal atomic unit for
   the MultiCore factory alone is the factory + 5 reader/writer variants + 4 compute
   variants (~hundreds of conversion sites).
2. **A structural pattern the recipe/catalog does not yet cover** — kernel-side
   CB-aliasing-by-name (Finding F1 below). I judged this worth stopping on rather than
   improvising, per the operating posture.

On reason 2: after surfacing F1 I checked the on-branch spec validator
(`tt_metal/impl/metal2_host_api/program_spec.cpp:310-370`) and confirmed F1's
case-(a) resolution **is validator-supported** — two different accessor names binding
the same DFB is legal (endpoints accumulate per `dfb_spec_name`, not per
accessor_name). So F1 is a **documentation gap, not a hard blocker**: the pattern
exists, it's just undocumented and I'd be inventing it. The real blocker is reason 1
(volume + atomicity), which is not improvable-around in one pass. A grounded stop here,
with F1 + the other findings surfaced, is higher-value to the recipe maintainers than
a rushed full conversion of ~13 kernels + 2 factories that bakes in unreviewed answers
to F1/C2 across hundreds of sites.

## TTNN ProgramFactory

### Concept realized
Not realized in code (see Outcome). Target confirmed correct: `ProgramSpecFactoryConcept`
/ `MaximizeCacheReuse`. Verified the adapter calls
`ProgramSpecFactory::create_program_spec(attrs, tensor_args, tensor_return_value)`
and consumes a `ttnn::device_operation::ProgramArtifacts{spec, run_params}`
(`mesh_device_operation_adapter.hpp:728`), matches `TensorArgument`s by `MeshTensor`
pointer identity (`resolve_bindings`, `:686`), and refreshes via `UpdateTensorArgs`
on cache hit (`:761`). The audit's concept choice holds; no disagreement to surface.

### Device-op-class edits (would be forced by the port)
- Custom `compute_program_hash` deleted: **none** — `LayerNormDeviceOperation` has no
  override (audit-confirmed; the `layernorm_nanobind.cpp:253` static is a test hook
  calling the framework default).
- Pybind entry points to remove: **two** — `create_descriptor` hooks at
  `layernorm_nanobind.cpp:322` (MultiCore) and `:363` (Sharded). The port replaces
  `create_descriptor` with `create_program_spec`, so these reference vanished symbols.
  Sanctioned pybind-removal exception applies. **Handoff: user-visible surface change**
  — Python callers of these test hooks must be retargeted separately. (Recorded under
  Handoff points.)

### Open items
- Tensor-arg relaxation candidate (forward-looking, not for this port): the audit's
  Team-only note flags `validate_on_program_cache_miss` enforcing padded-shape matching
  between input/residual and gamma/beta padded-width equality; a future
  `match_padded_shape_only` relaxation *might* be safe for some bindings. Unverified;
  default strict is correct for the port.

## Handoff points

- **Removed pybind surface (would-be).** `ttnn/cpp/ttnn/operations/normalization/layernorm/layernorm_nanobind.cpp:322` and `:363` expose
  `create_descriptor` for `LayerNormMultiCoreProgramFactory` and
  `LayerNormShardedProgramFactory` respectively (Python test/introspection hooks that
  return a `ProgramDescriptor`). A Metal 2.0 port deletes `create_descriptor` →
  these must be removed (build-breaking otherwise). Tagged "API surface: removed entry
  point." Downstream Python consumers of `*.create_descriptor(...)` need to be found
  and updated by their owners.
- **Out-of-op `get_tile_size(cb_id)` Device-2.0 holdovers** (audit table) are *not* a
  port handoff — they're already routed to the Device 2.0 track. Noting here only so
  the next porter doesn't re-discover them: the port must **not** absorb them
  (kernel-side whitelist), and they don't block the Metal 2.0 conversion (the in-scope
  `dfb::name → uint32_t` implicit conversion makes `get_tile_size(dfb::name)` compile
  if the call is reached, though the cleaner member-form fix belongs to that track).

## Successes

- **Patterns catalog "Aliased DFBs" + audit heads-up fired correctly.** The audit
  pre-identified the welford-fp32 aliased CBs (c_0/c_18/c_19/c_23 → c_29/c_30/c_31) and
  pointed at `advanced_options.alias_with`. Reading the legacy CB construction
  (`layernorm_op_multi_core.cpp:688-693,709-714,734-739,810-815`) confirmed the
  multi-`format_descriptor` shape the catalog's recognition signal describes exactly.
  Without the heads-up I would have mis-modeled these as independent DFBs.
- **`test_pack_relu.cpp` as a de-facto worked example.** With no TTNN reference port,
  `tests/tt_metal/tt_metal/test_pack_relu.cpp` was the single most useful artifact for
  pinning down construction idioms — `DFBSpecName`/`KernelSpecName` typed constants,
  `ProducerOf`/`ConsumerOf`, the `ProgramRunArgs::KernelRunArgs` brace-init shape, the
  per-node `runtime_arg_values` table. The recipe could point porters at it explicitly.
- **The headers are genuinely self-documenting**, as the migration guide promises.
  `kernel_spec.hpp` / `dataflow_buffer_spec.hpp` / `compute_hardware_config.hpp`
  resolved every field-name and semantics question without guesswork (e.g. the
  `unpack_to_dest_mode` FP32-consumer requirement is stated inline at
  `compute_hardware_config.hpp:55-66`).

## Friction

### Gaps

- **F1 (highest value): kernel-side CB-aliasing-by-name has no documented Metal 2.0
  pattern.** Layernorm's compute kernels resolve one logical CB *name* to a different
  *physical CB* depending on a compile-time path, via `#ifdef` + named-CTA chains:
  - `layernorm.cpp:96-105`: `cb_x` = `cb_xmm` (RMSNORM+fused) | distinct `c_23`
    (non-RMS fused) | `cb_in` (non-fused). The kernel then does
    `CircularBuffer cb_x_obj(cb_x)` and uses `cb_x` as both consumer (read input) and
    producer (write post-add).
  - `layernorm.cpp:66`: `cb_xmm = cb_in` under RMSNORM.
  - `layernorm.cpp:119`: `cb_im_or_out = (do_gamma|do_beta) ? cb_fusion : cb_out`.
  - `layernorm_welford.cpp:67-74,80-82`: same `cb_x` chain plus `cb_x_welford` which is
    *either* a real alias index (c_29) *or* equal to `cb_x`/`cb_in`.
  - `layernorm.cpp:290`: `uint32_t cb_outg = do_beta ? cb_fusion : cb_out;` — a
    **value-based (not `#ifdef`) ternary** over two CB names, where `do_beta` is a CTA.
    This compiles in Metal 2.0 only when *both* `dfb::cb_fusion` and `dfb::cb_out` are
    bound on the relevant path (they are, whenever gamma/beta present), but the porter
    must reason that out per-path — the migration guide's Principle-2 warning about
    file-scope ternaries referencing possibly-unbound `dfb::` names makes this a
    parse-time hazard that needs case-by-case verification across every such site.
  These aliasing/ternary sites are entangled with the `#ifdef FUSE_PRE_ADD / RMSNORM /
  TILIZE_IN` paths throughout the compute kernels, so a faithful conversion is not
  mechanical — it requires per-path reasoning about which DFB each name resolves to and
  whether every referenced `dfb::` handle is bound on that path. This is the dominant
  reason the conversion volume is high-risk, not merely high-count.

  In the legacy model this is trivial: a named CTA carries a `uint32_t` CB index, and
  the kernel does integer aliasing. In Metal 2.0, CB identity comes only from a
  `DFBBinding → dfb::name`; there is no "named CTA that is a CB index." Two distinct
  sub-cases, neither documented:
    (a) **same-FIFO aliasing** (`cb_x` == `cb_in` on the non-fused path): the two names
        must refer to the *same DFB with shared read/write pointers* — NOT the
        `advanced_options.alias_with` feature (that gives two distinct unique_ids
        sharing backing L1, with independent FIFO pointers). The plausible Metal 2.0
        expression is a *second `DFBBinding` to the same `dfb_spec_name`* with a
        different `accessor_name` ("cb_x" → INPUT), yielding `dfb::cb_x` and
        `dfb::cb_in` as two handles for one FIFO (cf. the Self-loop pattern, which
        binds the same DFB twice on one kernel). **Unverified** that two
        non-self-loop accessor names to one DFB compile/validate, and the per-path
        *which-DFB-does-cb_x-bind-to* selection has to move host-side as conditional
        bindings + kernel `#ifdef` — i.e. the Conditional-binding pattern applied to
        accessor→DFB *mapping*, not to presence/absence. The catalog's
        Conditional-binding entry only covers presence/absence.
    (b) **real alias index** (`cb_x_welford` == c_29): this IS `alias_with`, already
        covered.
  **Verified:** case (a) is supported by the spec validator —
  `program_spec.cpp:310-370` accumulates DFB endpoints per `dfb_spec_name` and tracks
  bindings per `accessor_name`, so two accessor names ("cb_in", "cb_x") binding one DFB
  ("INPUT") is legal; the only same-name reuse restriction is the self-loop-pair case
  (`:333-342`). So F1(a) is a **documentation gap**, not a structural blocker: the
  pattern works, it's just undocumented and the porter has to derive it (and derive
  that the per-path accessor→DFB selection moves host-side as conditional bindings +
  kernel `#ifdef`). The catalog's Conditional-binding entry covers presence/absence,
  not accessor→DFB *remapping*; an explicit "CB-aliasing-by-name" entry would close it.

- **F2: `get_named_compile_time_arg_val("cb_*")` retirement is implied, not stated.**
  Every layernorm kernel pulls CB indices via `get_named_compile_time_arg_val("cb_in")`
  etc. (legacy `KernelDescriptor::named_compile_time_args`). The migration guide
  documents positional CTA → named CTA and CB-index → `dfb::name`, but does not
  explicitly say that an existing *named* CTA carrying a CB index also converts to
  `dfb::name` (not to `get_arg(args::cb_in)`). It's inferable from Principle 2, but a
  one-line note ("legacy named CTAs that carry CB indices become DFB bindings, not
  named args") would remove ambiguity for the bulk wave — many recent ops use the
  named-CTA half.

- **F3: the factory's non-standard 4th `create_descriptor` param has no recipe note.**
  Both legacy factories take `const std::optional<CoreRangeSet>& core_range_set`
  (`layernorm_device_operation.hpp:24,36`), which the fixed `create_program_spec`
  signature cannot carry. Production always uses the default; only the pybind test
  hooks pass it. The resolution (drop the param, inline the default, delete the pybind
  hooks) is clear once seen, but the recipe's device-op-edit section only anticipates
  custom-hash + pybind-`create_program_descriptor`; an extra factory parameter that
  exists *only* for a pybind hook is a third, undocumented forced edit shape.

### Confusion

- **C1: `create_program_spec` vs `create_program_artifacts` naming inconsistency.**
  The on-branch concept (`operation_concepts.hpp:90`) and adapter
  (`mesh_device_operation_adapter.hpp:728`) use **`create_program_spec`**. The
  migration guide (`metal2_migration_guide.md:869`) and the patterns catalog's
  Multi-variant example (`metal2_port_patterns.md:194`) use **`create_program_artifacts`**.
  The recipe and TTNN-integration doc use `create_program_spec`. A porter following
  the catalog's Multi-variant example verbatim would write a method the concept does
  not recognize (silent — `ProgramSpecFactoryConcept` just wouldn't match, surfacing as
  the `AllFactoriesValid` static_assert the recipe lists as a build failure mode).
  Reconcile the docs to the on-branch name `create_program_spec`.

- **C2: recip `TensorParameter` with no `TensorBinding` — validator tension.**
  The welford reciprocal LUT (c_25) is a borrowed-memory DFB
  (`borrowed_from = RECIP_PARAM`). Per the migration guide a borrowed DFB names a
  `TensorParameter`; but the spec validator (`tensor_parameter.hpp` /
  migration-guide "every TensorParameter needs ≥1 TensorBinding") requires every
  `TensorParameter` to be bound by some kernel. The recip tensor is consumed *only*
  via the DFB, never via a kernel `ta::` accessor — so it has a `borrowed_from`
  reference but no `TensorBinding`. **Unverified** whether `borrowed_from` counts as
  "bound" for the validator's ≥1-binding rule, or whether borrowed-backing
  `TensorParameter`s are exempt. The docs don't address the borrowed-but-not-`ta`-bound
  case. (Same question applies to all the sharded factory's borrowed CBs:
  input/residual/stats/output are borrowed *and* some are also `ta`-accessed, but the
  recip/stats backing tensors may be borrowed-only.)

## Open items for downstream

- **Sharded factory + `sharded_layernorm_factory_helpers.cpp` (~1900 lines)** is the
  larger half and was inventoried only at summary level (plan §Variant: Sharded).
  Notable for the next pass: 3 program-scope semaphores → `SemaphoreSpec` +
  `SemaphoreBinding` on the mcast sender/receiver kernels; many borrowed-memory CBs;
  the `writer_unary_sharded_ln.cpp:38` runtime-known-count `segment_args` counted-loop
  read (RTA-vararg-shaped; supported, prefer named unless the runtime-varying index is
  genuinely needed).
- **Cross-op kernels: none** — all kernel `.cpp` are layernorm-owned; no fork/in-place
  decision. (Shared *headers* in `kernel_lib/`/`kernel/`/`kernel_util/` are out of
  scope and Device-2.0-clean.)
- **Dead reader RTA** `packed_one_value` (`reader_unary_interleaved_ln.cpp:33`,
  host `layernorm_op_multi_core.cpp:553-554,591`) — documented unused; op-owner cleanup,
  not port scope.
- **Test coverage**: primary correctness test is the pytest
  `tests/ttnn/unit_tests/operations/fused/test_layer_norm.py` (no dedicated C++ gtest).
  Not run — no converted code to test. The build/test step was therefore not exercised;
  a future completed port must run it (and the sibling sharded/distributed tests under
  `tests/ttnn/unit_tests/operations/fused/`).

## Recipe-improvement summary (for doc maintainers)

1. Add a patterns-catalog entry for **F1** (kernel-side same-FIFO CB-aliasing-by-name) —
   the highest-value gap; layernorm is unlikely to be the only op doing this.
2. Reconcile **C1** (`create_program_spec` vs `create_program_artifacts`) across
   migration guide + patterns catalog to the on-branch name.
3. Add the **F2** one-liner (named-CTA-carrying-a-CB-index → `dfb::name`).
4. Document **C2** (borrowed `TensorParameter` and the ≥1-`TensorBinding` validator rule).
5. Note **F3** (a factory param existing only for a pybind hook → drop + delete hook)
   as a third forced device-op-edit shape.
6. Add a recipe note that **runtime kernel-source selection forces atomic conversion**
   of a factory and all its selectable kernel variants (no partial-path sub-target),
   and flag such ops as large during the audit so porters can plan accordingly.

---

## Running notes (chronological; reorganized into sections at the end)

- **[scale]** This op is far larger than the single-factory ports the recipe's examples
  assume. `LayerNormDeviceOperation` has TWO program factories
  (`LayerNormMultiCoreProgramFactory` ~844 lines, `LayerNormShardedProgramFactory`
  ~600 lines + `sharded_layernorm_factory_helpers.cpp` ~1900 lines) under one
  device-op `program_factory_t` variant, plus 26 kernel source files. The recipe's
  "the program factory body is the port" framing is accurate but the *volume* here
  is exceptional. Noting up front because it bears on how much can land in one pass.

- **[framework precondition OK]** Metal 2.0 framework headers are present on this
  branch: `tt_metal/api/tt-metalium/experimental/metal2_host_api/*.hpp`,
  `ttnn/api/ttnn/metal2_artifacts.hpp` (`ProgramArtifacts`),
  `ttnn/api/ttnn/operation_concepts.hpp` (`ProgramSpecFactoryConcept`). The concept
  keys on `&T::create_program_spec` (recipe/migration-guide also call it
  `create_program_artifacts` in places — see Friction/Confusion).

- **[no reference port — confirmed]** Grep confirms NO ttnn op defines
  `create_program_spec` today; layernorm is genuinely first. The only worked examples
  of the host API in-tree are low-level tt_metal gtests under
  `tests/tt_metal/tt_metal/` (e.g. `test_pack_relu.cpp`) — these are Gen2/Quasar
  fixture tests using `MakeProgramFromSpec(*mesh_device, spec)` directly, NOT the
  TTNN `ProgramSpecFactoryConcept` adapter path. Useful for spec-construction
  idioms; silent on the TTNN factory wiring (what `create_program_spec` returns and
  how the adapter consumes it). See Friction.

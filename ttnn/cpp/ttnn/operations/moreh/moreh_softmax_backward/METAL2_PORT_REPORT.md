# Metal 2.0 Port Report — `moreh_softmax_backward`

## Outcome

**`PORTED`** — all five factories (`WSmall`, `WLarge`, `HSmall`, `HLarge`, `CLarge`) converted in
one change, together with the 13 kernel entry points they bind. None deferred.

Verification: `./build_metal.sh --build-tests` green, and the invoker-confirmed test set —
`tests/ttnn/nightly/unit_tests/operations/moreh/{test_moreh_softmax, test_moreh_softmin,
test_moreh_logsoftmax, test_moreh_logsoftmax_ulp}.py`, run in full — gives **439 passed, 96
skipped, 0 failed** (7m49s, Wormhole). The 96 skips are a test-authored `bfloat8_b` dtype guard
(`test_moreh_softmax.py:42-43` and peers), not a path the port disabled. The program-cache-hot
path is covered and passing (`test_{softmax,softmin,logsoftmax}_backward_callback`), so the
cache-hit `UpdateTensorArgs` tensor rebinding works against the new `TensorParameter`s.

One JIT diagnostic appears in the log and is **not** from this port: a deprecation warning for
`sfpi::int32_to_float` inside `ckernel_sfpu_signbit.h`, raised while compiling the *forward* op's
`moreh_softmax/device/kernels/moreh_softmax_c_large.cpp`. Pre-existing, in an LLK header, on a
file this change does not touch.

## Provenance

- **Recipe docs (this port):** `bace43c8fb5 2026-08-12 docs(metal_2.0): stop the port from deleting the op's custom program hash`
- **Audit docs (inherited):** `bace43c8fb5 2026-08-12 docs(metal_2.0): stop the port from deleting the op's custom program hash`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, as the audit chose. No re-decision, nothing surfaced to the invoker.
All five factories now expose `create_program_artifacts` (declared by the existing
`DEFINE_SOFTMAX_BACKWARD_FACTORY` macro, `device/moreh_softmax_backward_device_operation.hpp:50-63`);
the `program_factory_t` variant is unchanged, so the whole op moved together and no factory was
left on the legacy concept.

### Device-op-class edits

- **Pybind entry points removed:** none. `moreh_softmax_backward_nanobind.cpp` never bound
  `create_descriptor` — it exposes only the three public functions and two value enums — so the
  vanishing factory entry point took no pybind line with it.
- **Custom `compute_program_hash`:** none — the op uses the framework default. Nothing to leave
  intact, and no `UpdateTensorArgs` legality surprises on the program-cache-hot invocations.
- The device-operation `.cpp` is byte-identical; only the header's factory-declaration macro
  changed (plus the `program_descriptors.hpp` → `metal_v2_artifacts.hpp` include swap it forces).

### Open items

- **Relaxation candidates:** none. Every `TensorParameter` is left strict, matching the audit.
- **RTA → CRTA candidates (noted, deliberately not converted).** Several runtime args carry the
  same value on every node: reader/writer `Wt` and `Ht`, the H readers' `scaler` and `mask_h`, the
  W readers' `mask_w`, and the C path's `outer_stride` / `inner_size` / `dim_size`. Only `N` and
  `tile_offset` genuinely vary per node. These are `common_runtime_arg_names` in the natural
  Metal 2.0 shape, and the dispatch saving is real across a full compute grid — but the recipe is
  explicit that RTA→CRTA changes dispatch semantics and is a separate pass, so this port left them
  as per-node RTAs. Good first candidate for that follow-up.
- No capability gaps: single-program, no op-owned tensors, no op-owned `GlobalSemaphore`s.

## Handoff points

**None.** No capitulation, no boundary-rule assumption violation, no kernel-lib gap, no framework
gap, no removed pybind surface. Every donor call site bridged with the conversions the brief
predicted (`DataflowBuffer`-by-value via the implicit `DFBBindingToken` constructor; `uint32_t`
NTTPs via the `constexpr operator uint32_t()`), and no file outside the op directory was touched.

## Successes

- **[Anti-pattern: Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  fired correctly, and it was a real near-miss.** Each factory emits the compute kernel twice from
  one source, differing only in the leading CTA — and with five structurally identical factories
  the pull toward "one `KernelSpec`, move `num_tiles_per_core` to an RTA, done" was strong; it
  would have removed five `KernelSpec`s and five `WorkUnitSpec`s from the diff. The entry's flat
  statement that the premise ("Metal 2.0 supports only one `KernelSpec` per source") is *false*,
  plus its worked two-`WorkUnitSpec` block, is what stopped it. Realized at
  `device/softmax_backward_w_small/softmax_backward_w_small.cpp:121` (`make_compute_spec`) and the
  paired `core_group_1` / `core_group_2` work units at `:242-262`, and identically in the other
  four factories.
- **[Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options)
  — "answering *did it set one?* is the part that goes wrong".** `grep -n opt_level` over the op
  returns zero hits, which reads as "nothing to carry over". The section's explicit statement that
  an absent `KernelDescriptor::opt_level` still resolves to **`O3`** on a `ComputeConfigDescriptor`
  is the only reason all ten compute `KernelSpec`s (five factories × two core groups) carry
  `.compiler_options.opt_level = KernelBuildOptLevel::O3` instead of silently dropping to Metal
  2.0's `O2` default. Nothing in the build or the tests would have caught that.
- **[Unity-build hygiene for anonymous-namespace symbols](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols).**
  `TT_ENABLE_UNITY_BUILD(ttnn_op_moreh)` is on (`ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:7`)
  and all five factories need the same spec-name constants and builders. Reading the pattern
  *before* writing meant the shared header
  (`device/moreh_softmax_backward_metal2_common.hpp`) existed from the first draft and the
  duplicate-symbol failure never happened.
- **[Two-toucher → assign 1P+1C](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)
  — "Re-derive, don't transcribe".** The re-derived census agreed with the brief exactly (self-loop
  `c_24`/`c_25`/`c_26`, plus `c_27` in the two `*Large` W/H factories; everything else plain
  1P+1C). The value was not catching a wrong brief — it was that walking the compute kernels
  toucher-by-toucher is what surfaced the same-FIFO aliasing the audit had not named (see Friction).
- **The brief's dead-RTA call** — "don't invent a name for an arg no kernel reads". The W factories
  pushed `std::bit_cast<uint32_t>(scaler)` at reader RTA index 5 that no W reader ever reads; the
  reflex under a named-arg conversion is to name every slot. Confirmed against
  `reader_moreh_softmax_backward_w.cpp` (reads index 6, never 5) and dropped, along with the local
  `float scaler` that fed it. The H factories' `scaler` **is** live and is named and kept.

## Friction

### Gaps

- **The recipe's named reference port is stale against the current headers on five points, and
  two of them are idioms the recipe now forbids.** [§Inputs the invoker should have
  supplied](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#inputs-the-invoker-should-have-supplied)
  points at accumulation on `origin/akertesz/porting-experiment-accumulation-jun10` as "a good
  shape reference when no closer op exists". Reading
  `ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_program_factory.cpp` there:
  1. the method is `create_program_spec`, not `create_program_artifacts`;
  2. `hw_config = DataMovementHardwareConfig{.role = RoleHint::READER}` — that struct is now a
     `std::variant<DataMovementGen1Config, DataMovementGen2Config>` with no `role` field;
  3. `ComputeHardwareConfig{.math_fidelity=…, .fp32_dest_acc_en=…, .dst_full_sync_en=…,
     .math_approx_mode=…, .unpack_to_dest_mode=…}` — now a variant over `ComputeGen1Config` /
     `ComputeGen2Config`, every one of those five fields renamed and two of them transformed;
  4. `runtime_arg_values.push_back({core, args})` — **node-first**; the field is now
     `Table<std::string, Table<NodeCoord, uint32_t>>`, name-first;
  5. it uses `ProducerOf(...)` / `ConsumerOf(...)` and `TensorArgument{std::cref(tensor)}`, both of
     which the current recipe explicitly tells the porter *not* to write.

  Items 1–4 fail loudly at compile time, so they cost time rather than correctness. Item 5 is the
  harmful one: it compiles, and a porter who reaches the reference before reaching the recipe's
  operating-principle paragraph will copy it. Suggest either refreshing that branch or repointing
  the bullet at something on `main` — `ttnn/cpp/ttnn/operations/copy/typecast/device/
  typecast_program_factory.cpp` is current on every one of these five points and has the same
  reader/writer/compute + work-split shape.

- **Recipe and header disagree on how to set a common compute field.**
  [Compute kernels](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compute-kernels)
  says to reach into the variant: "Style A: via `std::get<ComputeGen1Config>(compute_hw).<field> =
  …`". `compute_hardware_config.hpp:212-213` says the opposite, in as many words: "For common
  fields, prefer this syntax over e.g. `std::get<ComputeGen1Config>(config).field`, which throws if
  the wrong architecture is targeted", and supplies free accessors (`unpack_modes(config)`,
  `enable_32_bit_dest(config)`, …). I followed the header (`unpack_modes(compute_hw_config) = …`,
  e.g. `softmax_backward_w_small.cpp:108`), per the recipe's own "go to the headers first" rule —
  but the two should agree, since the recipe's spelling is the one that breaks on a Quasar build.

- **The `unpack_modes` guidance is written for the wrong default case.** The section leads with
  "Fields the op left at their Metal defaults need no action" and frames `unpack_modes` as
  reindex-and-translate work. This op's legacy vector is
  `vector<UnpackToDestMode>(NUM_CIRCULAR_BUFFERS, Default)` — *entirely* default — which reads as
  "nothing to carry over", and the reindex/translate items 1 and 2 genuinely have nothing to do.
  Item 3 (the newly-required explicit entry) is the one that applies, and it applies to **seven or
  eight DFBs per factory**. It is easy to stop reading after items 1–2. Suggest hoisting a sentence
  to the top of the `unpack_modes` item: *an all-`Default` legacy vector is not "no action" — under
  `fp32_dest_acc_en` every consumed Float32 DFB still needs an explicit `UnpackToSrc` entry.*
  Realized as `MakeUnpackModes` in `device/moreh_softmax_backward_metal2_common.hpp`.

### Confusion

- **Same-FIFO aliasing: the catalog's prescription doesn't reach this op's shape.**
  [Pattern: Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
  says "alias the *handle*, keep *one* object", and its worked example is a `constexpr` alias of the
  token (`constexpr auto cb_x = dfb::cb_in;`). That covers the case where the second name is used
  as a **CB id**. In these compute kernels the second name is used as a **`DataflowBuffer`
  object**: `c_24` is `cb_ydy` / `cb_exp` / `cb_inter0` and `c_25` is `cb_sum` / `cb_inter1`, and
  the legacy code constructs a *separate* `DataflowBuffer` for each name, then passes the object by
  value to donor helpers (`exp_tile_to_cb`, `mask_tile_to_cb`, `add_tiles_to_cb`). A `constexpr`
  token alias does not serve those call sites, and the pattern's explicit "Don't construct two
  `DataflowBuffer` objects from the same `DFBAccessor`" rules out the mechanical translation.

  I resolved it with a **reference alias** — `auto& dfb_exp_obj = dfb_ydy_obj;` — which keeps
  exactly one `DataflowBuffer` per DFB (so the object↔DFB identity the pattern protects is intact)
  while preserving the kernel's own role vocabulary at every call site, so the diff stays a rename.
  Where the alias is used as an NTTP instead, it is a `constexpr` token alias as the pattern
  prescribes (`constexpr auto dfb_inter0 = dfb::ydy;`,
  `device/kernels/moreh_softmax_backward_w.cpp:41`). Both forms appear in the same `#ifdef LOG`
  block, which is itself a decent illustration. Worth adding the object-valued case to the pattern:
  *when the second name names an object rather than an id, alias the object by reference — one
  `DataflowBuffer`, two names.*

  Secondary note: **the audit didn't flag this at all.** Its endpoint census counted `c_24`/`c_25`/
  `c_26` correctly as single-toucher self-loops, but a census counts *kernels*, so duplicate
  objects inside one kernel are invisible to it. Not a gate gap — the port handled it under an
  existing pattern — but a line in the audit's CB-endpoints output ("N kernel-side names per
  index") would have pointed the porter at it before construction rather than during.

- **The migration guide's "Local DFB invariant" bullet reads as forbidding the standard per-core-
  group split.** [Troubleshooting](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#troubleshooting)
  says: "A local DFB's producer and consumer kernels must share *identical* `WorkUnitSpec`
  membership." In this port the reader (producer of `y`) belongs to **both** work units while
  `compute_g1` (consumer of `y`) belongs to **one** — so read literally, the invariant is violated
  by the very shape the *Demoting per-group CTA to RTA* entry's Correct-port block prescribes. The
  real rule is per-node, and both the declaring header
  (`dataflow_buffer_spec.hpp:41-50` — "This is a per-node rule, not a per-spec one … you MAY bind
  more than one KernelSpec to a producer (or consumer) endpoint") and the validator
  (`program_spec.cpp`) say so. Cost ~15 minutes of re-reading before the header settled it.
  Suggest rewording the bullet to *identical node coverage* rather than *identical `WorkUnitSpec`
  membership*.

- **Minor — `Table` has no `push_back`, and the recipe says so, but the run-args helper is the
  place you feel it.** The recipe's `Table`s-are-maps note is accurate and I hit no compile error,
  but the thing that actually made the node-first legacy loop translate cleanly was
  `AddRuntimeArgsForNode`, which is documented in
  [Construct](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#construct-paired-spec--run-args)
  one bullet *above* the `Table` note. Reading them in the other order would have saved a pass.
  Not a defect — just noting the ordering, since every factory in this op needed the helper.

## Open items for downstream

- **Shared kernel touches — rung 3 (converted in place), sunset list empty.**
  - `device/kernels/writer_moreh_softmax_w.cpp` — bound by `MorehSoftmaxBackwardWSmallFactory` and
    `MorehSoftmaxBackwardWLargeFactory`.
  - `device/kernels/writer_moreh_softmax_h.cpp` — bound by `MorehSoftmaxBackwardHSmallFactory` and
    `MorehSoftmaxBackwardHLargeFactory`.

  Both were converted **in place**, not forked, because the invoker explicitly assigned the bundled
  all-five port: the bundled set is the complete consumer set, so no binder is left on the legacy
  API. **Remaining unmigrated consumer op directories: none.** No `_metal2` fork was created and no
  pointer comment was added — neither is needed when nothing is left behind. The identically-named
  files under `moreh_softmax/device/kernels/` are the *forward* op's separate private copies and
  are untouched and unaffected; they are not consumers of this directory.

- **Two unreferenced kernel files left on the legacy API, by design.**
  `device/kernels/writer_moreh_softmax_backward_h.cpp` and
  `device/kernels/writer_moreh_softmax_backward_w.cpp` are bound by no factory in the repo, were
  outside the audit's scope, and were not ported. They are now the only files in this op directory
  still carrying `TensorAccessorArgs<0>()` and `tt::CBIndex::c_16`, so a post-port grep of the
  directory for legacy CB idioms hits them and *should* — they are the audit's deletion candidates,
  not port leftovers. Whoever deletes them closes that out.

- **Op-owner findings inherited from the audit, untouched by this diff** (each is real, none is
  port work):
  - Hardcoded 512 KiB L1 budget in the small-path availability heuristics
    (`device/moreh_softmax_backward_device_operation.cpp:11`, used at `:31` and `:52`) — WH and BH
    both have ~1.5 MiB, so the fast small path is under-selected.
  - The same heuristic sizes every buffer at the **data-format** tile size (`:19`, `:40`) while the
    factories allocate the three intermediates at the **intermediate** format size — ~2× low under
    `fp32_dest_acc_en`, so it can admit a configuration that does not fit. The two errors push in
    opposite directions.
  - Tautological guard `TT_FATAL(dim >= 0 && dim < rank, …)` (`:87`) — `dim` is `uint32_t`.
  - `log_info(tt::LogTest, "…tensor algorithm selected")` on the production path at every cache
    miss in all five factories — wrong channel for shipped op code. Preserved verbatim.
  - Unused kernel locals preserved as-is: `uint32_t l1_write_addr_in;` in all four H/W readers,
    `constexpr uint32_t onetile = 1;` in the two W/H writers (they loop on `blk`), and
    `constexpr int dst0 = 0;` in `moreh_softmax_backward_c_large.cpp`.

- **Test coverage note.** This op has **no C++ gtest coverage** — every test that exercises it is a
  nightly pytest (`test_moreh_softmax.py`, `test_moreh_softmin.py`, `test_moreh_logsoftmax.py`;
  `test_moreh_logsoftmax_ulp.py` covers the forward op only). The recipe's recommended
  "gtests first, they fail fast" step therefore has nothing to run for this op, and the first
  signal a porter gets is a multi-minute pytest session. Ops in this family are likely the same.

- **Per-op carry-over.** `moreh_softmax` (the forward op) is this op's structural twin — same five
  parallelization strategies, same buffer roles, same per-core-group compute pair, and its own
  private copies of the two writer kernels. The DFB naming and the `MakeDFB` / `MakeComputeDefines`
  / `MakeUnpackModes` shape in
  `device/moreh_softmax_backward_metal2_common.hpp` should transfer almost verbatim, and the two
  ops' shared-header layout would be worth keeping symmetric.

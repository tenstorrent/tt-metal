# Metal 2.0 Port Report — `moreh_getitem`

## Outcome

**PORTED** — both factories, all three program shapes (RM · Tilized-W · Tilized-noW) and all six
kernels converted in one change. Nothing left for a later pass. Tests: see
[Verification](#verification) below.

## Provenance

- **Recipe docs (this port):** `b72c35b810e 2026-08-04 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `b72c35b810e 2026-08-04 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## Verification

| | result |
|---|---|
| Build | `./build_metal.sh --build-tests` → **SUCCESS** (warnings-as-errors ON; no warning from any file this port touched) |
| Pre-port baseline | `pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_getitem.py` → **338 passed, 0 failed** (78.9 s) |
| Post-port | same command → **338 passed, 0 failed** (165 s cold kernel cache; **84 s** on an immediate warm re-run — the rebuild invalidates the JIT cache, so the first run pays every kernel compile) |
| Post-port, after the rule-8 comment relocation | same command → **338 passed, 0 failed** (96.9 s; kernel-source edit re-JITs) |

Test set confirmed with the invoker before use. The one other `*getitem*` hit,
`tests/ttnn/unit_tests/base_functionality/test_getitem.py`, exercises `__getitem__` / slice rather than
this op and is not part of the baseline. There are no C++ gtests and no sweeps for this op.

Coverage note: the 338 cases exercise all three program shapes, both index layouts
(`ROW_MAJOR_INDEX` / `TILIZE_INDEX`), index counts 1–5, both supported dtypes, and — via the two
`*_callback` tests — the program-cache-hit path, which is the path Q1's decision bears on.

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` (plain) for **both** `MorehGetItemRmFactory` and
`MorehGetItemTilizedFactory`, exactly as the audit decided. No re-decision, nothing surfaced.
Confirmed against `ttnn/api/ttnn/operation_concepts.hpp:118-121`: each factory declares only
`create_program_artifacts`, so the `AllFactoriesValid` check sees exactly one concept per variant.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — the op never had one.
- **Pybind entry points removed:** none. `moreh_getitem_nanobind.cpp:18` binds only
  `ttnn::moreh_getitem`; no legacy factory entry point was exposed.
- The only edit to `moreh_getitem_device_operation.hpp` is the two factory declarations themselves
  (`create_descriptor` → `create_program_artifacts`) plus the matching include swap
  (`<tt-metalium/program_descriptors.hpp>` → `ttnn/metal_v2_artifacts.hpp`). Nothing in
  `moreh_getitem_device_operation.cpp` — `validate_inputs`, `select_program_factory`,
  `compute_output_specs`, `create_output_tensors` — was touched.

### Open items

- **Relaxation candidates:** none applied. `TensorParameter::relaxations` is left default-empty on all
  seven parameters, per the invoker's Q1 decision. See [Q1](#q1--tensoraccessor-third-argument-verified-value-identical)
  for the mechanical verification that this is value-identical to the legacy override.
- **Deliberate style deviation:** the factory bodies keep their `ttnn::Tensor` locals for the 5-D shape
  and stride arithmetic and reach `.mesh_tensor()` only at the `TensorArgument` sites, rather than
  extracting `MeshTensor` at factory entry as the TTNN integration doc prefers. Extracting would have
  meant rewriting shape math the port is otherwise not touching, which trades a documented preference
  against the scope-discipline rule; the landed `moreh_mean` port has the same shape. Flagged so a
  reviewer can overrule cheaply.

## Handoff points

**None.** Specifically, none of the categories fired:

- No capitulation — every construct in the op had a supported Metal 2.0 form.
- No boundary-rule assumption violation — no call site needed `sem::name` or `tensor::name` outside the
  op directory (the op has no semaphores at all, and every accessor is consumed in its own kernel).
- No kernel-lib gap — unusually for this family, the op depends on no kernel-lib helper and no
  `moreh_common.hpp`; the complete include set across all six kernels is `api/*` plus the op's own
  `common.hpp`.
- No framework gap bit during the port. In particular the audit's "biggest design item" (below) turned
  out to be a solved pattern, not a missing capability.
- No pybind surface removed.
- No shared-kernel touch: the census (`grep -rl <filename> ttnn/cpp/ttnn/operations/` for each of the
  six kernels and for `common.hpp`) returns no consumer outside `moreh_getitem/`, no `kernel_source`
  points outside the op directory, and the three shapes bind six distinct sources — so no `_metal2`
  fork was reused or created, and no file outside the op directory was written.

## Successes

- **Patterns catalog — [Conditional / optional DFB bindings]: answered the audit's flagged unknown
  outright.** Both the brief and the audit called the optional/absent index tensors "the port's biggest
  design item… the item most likely to need a framework conversation; surface it early rather than
  inventing a shape." No conversation was needed: the catalog entry states the `tensor::` case
  explicitly ("Optional **tensors** are the case where the `#ifdef` gate is not merely preferred but
  **mandatory**: an absent tensor often has nothing to bind even in principle"), which is exactly this
  op — five slots of which only some exist per call. The prescription (omit the binding, emit a matching
  define, `#ifdef` the token references) transplanted with no adaptation to all three readers:
  `reader_moreh_getitem.cpp:60-74,116-130,150-179`, `reader_moreh_getitem_tilize.cpp:68-84,120-134,154-243`,
  `reader_moreh_getitem_tilize_w.cpp:69-85,123-138,169-296`. The audit's caution ("do not invent a shape for it
  under time pressure") plus the catalog's coverage is what kept this from becoming a design detour.
- **Recipe — "re-derive each endpoint disposition from the census, don't transcribe."** Following it
  produced agreement with the brief on all seven dispositions, so the re-derivation looks redundant in
  hindsight — except that it is what forced the one decision the brief could not make: RM `c_5` (the
  dim-4 index CB) is bindingless in a reachable configuration, and a census-driven "build a spec only
  for what a kernel touches" loop resolves it without touching the guard the invoker put out of scope
  (`moreh_getitem_rm_factory.cpp:155-190`). A transcribed disposition list would have left the
  allocation in place and the validator would have rejected that program at runtime.
- **Recipe — "`Table`s are maps, not vectors."** `KernelSpec::CompilerOptions::Defines` is
  `Table<std::string, std::string>`, and the legacy code it replaces is a
  `KernelDescriptor::Defines` **vector** built with `emplace_back` (`moreh_getitem_tilized_factory.cpp`
  legacy `:183-187`). The mechanical translation is `emplace_back` → `emplace_back`, which does not
  compile; the warning meant the conditional define table was built with `emplace` from the first draft
  (`moreh_getitem_tilized_factory.cpp:185-208`, `moreh_getitem_rm_factory.cpp:164`).
- **Whitelist rule 8 — "a comment on a line you're *right* to delete: relocate it."** The RM kernels
  carried a comment explaining why the accessor took an explicit page-size third argument ("…overrides
  TensorAccessorArgs::AlignedPageSize, which may be stale on program cache hits"). The argument is gone
  by decision, so keeping the comment verbatim would assert something now false — but the *question* it
  raises is one the next reader of these kernels will ask again, and the answer is not obvious from the
  code. Rule 8 is what stopped the reasoning from disappearing with the line: it is restated, forward
  and self-contained, at the surviving construction site
  (`reader_moreh_getitem.cpp:51-53`, `writer_moreh_getitem.cpp:19-21`).
- **Recipe — "no ephemeral doc cited from code," and the checklist grep for it.** The natural comment to
  write above each `#ifdef HAS_INDEX<n>` block is a pointer at the plan's Applied Patterns section. The
  rule (and the reason: those files are deleted before the merge) redirected that into self-contained
  comments that state *why* the gate exists at each site. The checklist grep over the diff returns zero
  `.md` citations.

## Friction

### Gaps

- **The docs name a `TensorParameter` field that does not exist.** The TTNN integration doc
  ("The relaxation infrastructure exists (`TensorParameter::advanced_options`, holding
  `dynamic_tensor_shape` / `match_padded_shape_only`)" — `ttnn_factory.md:89`) and the migration guide
  in two places (`migration_guide.md:368` and its step 1 at `:436`) point at an `advanced_options`
  member on `TensorParameter`. There isn't one. The field is
  `TensorSpecRelaxations relaxations;`
  (`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp:45`), and the two flags do
  live on that struct (`tensor_spec_relaxations.hpp:41-61`) — so only the owning member's name is wrong,
  which is the kind of error that reads as correct right up to the compile. `DataflowBufferSpec` and
  `KernelSpec` *do* have `advanced_options`, which is probably where it came from. This port sets no
  relaxation, so nothing broke here; an op that needs one hits a compile error and goes hunting.
  Suggested fix: `TensorParameter::advanced_options` → `TensorParameter::relaxations` at those three
  sites. (The invoker's Q1 answer used the correct name, which is how the mismatch surfaced.)
- **The recipe's TensorParameter planning bullet reads as unconditional.** *Plan the spec* says
  "**TensorParameters**: one per distinct legacy `TensorAccessor` originating tensor," with the
  conditional-resource discussion living entirely under *DFB* bindings; the catalog's conditional entry
  is reachable only from the DFB side of the recipe. For an op whose *parameter set* varies per
  instantiation — five optional index tensors here — the planning step gives no hint that the answer
  exists. One cross-reference on that bullet ("…and see [Conditional / optional DFB bindings] when a
  tensor is optional — the parameter and its binding are both omitted") would have taken this port's
  headline unknown off the table at planning time rather than at pattern-catalog reading time.
- **No guidance for a dead CB that is *config-scoped* and entangled with an open question.** The
  dead-CB disposition is written for a CB that is dead in every configuration ("a dead CB has no
  behavior, so the drop is zero-functional-change"). RM `c_5` is dead only in the configurations that
  define a normalized index dim of 4, and those are exactly the configurations the audit's Question 3
  flagged as a possible pre-existing bug — so "drop it" and "don't touch the question" appear to
  conflict. They don't (the drop is what the validator forces, and it changes no numerics), but the
  reasoning was mine to construct. A sentence in the CB-endpoints disposition — *a CB that is
  endpoint-less only in some configurations gets the same drop, scoped to those configurations; that is
  not a resolution of any correctness question attached to them* — would cover it.
- **The recipe's canonical doc path does not resolve on a normal working branch.** The invocation names
  `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md`, but the whole
  `metal_2.0/` doc tree lives only on the doc branch (here: the port branch the audit ran on), and the
  checkout I was handed was on an unrelated feature branch where that path is absent. Older copies of
  the same files were reachable elsewhere on the filesystem and **were not identical** — the copy I
  found first predated, among other things, the `experimental/quasar/` out-of-bounds rule and the whole
  `Compiler options` / `opt_level` section. Diffing against the branch version was what surfaced that.
  Suggested fix: have the recipe's *Before you begin* say that the doc tree is branch-local, and tell
  the porter to read it with `git show <branch>:<path>` (or to confirm the provenance hash from the
  brief matches `git log -1 -- docs/…/metal_2.0/` in their own checkout) **before** reading anything —
  it is the one instruction that cannot be recovered from reading the wrong copy.

### Confusion

- **Brief vs recipe on where argument names come from.** The brief says the host arg lists "line up
  positionally with the kernel reads — use them as the naming source"; the recipe's kernel-side rule 4
  says "pick names that match the variables they were going to be assigned to," i.e. the kernel locals.
  The two disagree wherever the legacy host and kernel names differ for the same slot — here
  `num_units_per_core` vs `num_sticks`, `input_unit_size` vs `stick_size`, `output_unit_size` vs
  `output_stick_size`. I followed the recipe (kernel-side names), on the grounds that the name appears
  in the kernel at every use and only once on the host. Worth one clarifying clause in whichever doc
  generates the brief's wording.
- **The brief's `DataflowBuffer*` warning is wrong, and it argues for a restructure the recipe would
  otherwise forbid.** The brief states the RM reader's `DataflowBuffer* index_dfb_obj`
  (legacy `reader_moreh_getitem.cpp:158`, dereferenced at `:184,189,190`) has "no binding-token
  analogue at all" and that "the `push_back`/`wait_front`/`pop_front` at those three lines will need to
  move inside the per-`dim` branches." Neither holds: `DataflowBuffer` is a plain non-template class
  (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:61`) and its Metal 2.0 constructor just takes a
  `DFBBindingToken`, so a pointer to a locally-constructed object is ordinary C++ and all three
  dereference sites port unchanged (`reader_moreh_getitem.cpp:149-189`). Acting on the brief would have
  duplicated three FIFO calls into four branches — a kernel-logic change, for no reason. The general
  lesson for the audit's heads-up: the question to ask is whether the kernel-side type is a template,
  not whether the *token* is static.

## Open items for downstream

- **Pre-existing defect, not fixed (audit Q3, out of scope per the invoker).** The ROW_MAJOR guard tests
  the **user-space** dimension — `TT_FATAL(dim != 4, …)` at
  `ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_device_operation.cpp:47-51` — while
  the factory and kernel work in **5-D-normalized** dimensions (`dim = index_dims[i] + (5 - rank)`,
  `moreh_getitem_rm_factory.cpp:70`). A rank-4 ROW_MAJOR input with `index_dims = {3}` therefore passes
  the guard and normalizes to dim 4, which the RM reader's `for (dim = 3; dim >= 0; dim--)` loop never
  visits: the index tensor is **silently ignored** and the op returns a wrong answer rather than
  erroring. Repro shape: input `[10, 5, 7, 70]` ROW_MAJOR, `index_dims = [3]`. The tilized factory
  computes the same predicate correctly in normalized space (`moreh_getitem_tilized_factory.cpp:77-79`)
  and routes to a reader that handles dim 4, which is what makes the RM path look like an oversight.
  This port neither fixed nor worsened it; the only related change is that the endpoint-less index DFB
  that configuration used to allocate is no longer built.
- **No test covers that configuration**, which is why the defect is invisible in CI: the RM cases in
  `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_getitem.py:26-33` pair rank 4 with
  `index_dim` 2 and rank 5 with `index_dim` 3, never rank 4 with `index_dim` 3. Whoever picks up the
  defect should add that case.
- **RTA → CRTA candidates, deliberately not converted** (the recipe says note, don't do: it changes
  dispatch semantics). Most of this op's runtime args hold the **same value on every node** — only
  `start_id` and `num_sticks` vary per core. Concretely: 28 of the RM reader's 30 named RTAs, 35 of the
  Tilized-W reader's 40, 37 of the Tilized-noW reader's 39, and all but two of every writer's. Moving
  them to `common_runtime_arg_values` would cut the per-dispatch runtime-arg payload by roughly an order
  of magnitude on an op that runs across the whole grid. This is the largest single follow-up the port
  surfaced.
- **Name-first RTA restructure.** The per-core loops still build node-first values and go through
  `AddRuntimeArgsForNode` (`moreh_getitem_rm_factory.cpp:288-353`, `moreh_getitem_tilized_factory.cpp:392-476,730-806`),
  which the recipe sanctions for the port and flags as a worthwhile separate cleanup. It pairs naturally
  with the CRTA item above — most of the transposed rows disappear entirely if they become CRTAs.
- **The `index{0..4}_is_defined` RTAs are now redundant.** The set of defined index dimensions is fixed
  at program-build time and is already expressed as the `HAS_INDEX<n>` defines the port introduced, so
  the five runtime flags and the `if (index_is_defined[dim])` guard they feed could both be replaced by
  the preprocessor. Not done here: it is kernel-logic surgery, and keeping the guard is what makes the
  port a syntax swap. A follow-up would save five RTAs per reader and one branch per dimension per
  stick.
- **More JIT kernel variants than legacy, by construction.** The `HAS_INDEX<n>` defines are part of the
  kernel build key, so two calls that differ only in *which* index dimensions they supply now compile
  separate binaries where legacy shared one (legacy passed the same source a `nullptr` address instead).
  Nothing is wrong — this is the conditional-binding pattern working, and it buys back the L1 an
  unconditional binding would waste — but it does mean first-dispatch JIT cost scales with the number of
  distinct index-dimension combinations an application uses. Measured on the test suite: 165 s cold vs
  **84 s** warm, against a 79 s pre-port baseline, i.e. steady-state runtime is unchanged (the ~5 s gap
  is within run-to-run noise on this bench) and the whole difference is one-time compilation. Worth knowing for any sibling indexing/
  gather op ported the same way (`moreh_getitem` is likely the family's template here).
- **Carry-over for sibling ops:** the optional-index-tensor shape (max-rank slot array on the host,
  `is_defined` flags on RTAs, all accessors constructed unconditionally in the kernel) is a
  family pattern, not a `moreh_getitem` quirk. Any sibling gather/scatter/index op ported next can lift
  the host loop and the `#ifdef` gating from `moreh_getitem_rm_factory.cpp:150-192` verbatim.
- **Still-open items from the audit's Misc anomalies, untouched by this port** (they are not port work
  and remain true): the index CB page size hardcoded as `1024 * 4` on the host against
  `#define INDEX_TILE_SIZE (4096)` in `moreh_getitem_tilized_kernels/common.hpp:11` with nothing tying
  the two together; tile geometry hardcoded in that same header; the unexplained
  `num_elements_per_alignment == 8` special case; and `output_dim_offset` computed from the *input* rank
  in `moreh_getitem_tilized_factory.cpp:67`. One anomaly on that list *was* resolved by the port: the
  dead `index_cbs[5]` array (and the dead `idx_cb` store) is gone from all three readers, since its
  element type was the legacy CB-index vocabulary.

## Appendix — decisions the invoker settled

### Q1 — `TensorAccessor` third argument: verified value-identical

Decision applied as given: the third constructor argument is dropped at all 7 RM sites
(`reader_moreh_getitem.cpp` legacy `:75,79-83`, `writer_moreh_getitem.cpp` legacy `:27`) and **no**
relaxation is declared. While planning I confirmed the drop cannot change addressing, which is worth
recording because the audit could not close it: the legacy override passed the *unaligned* logical stick
size, the binding token instead supplies the host-emitted `buffer->aligned_page_size()`
(`tt_metal/impl/buffers/tensor_accessor_args.cpp:179-185`), and the interleaved accessor **realigns
whatever it is handed** —
`InterleavedAddrGen::aligned_page_size = align_power_of_2(page_size, allocator_alignment)`
(`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:289-290`). Since
`align(align(p, A), A) == align(p, A)`, both spellings resolve to the same page stride. The suite covers
the case where the two inputs genuinely differ (`[10, 5, 7, 70]` bfloat16 → 140 B logical row, 160 B
aligned), and the two `*_callback` tests cover the program-cache-hit path the kernel comment worried
about.

### Q2 — `c_16` dead-CB drop: applied in all three shapes

Legacy allocation sites dropped, with no replacement `DataflowBufferSpec`:

| shape | legacy site | legacy name |
|---|---|---|
| RM | `moreh_getitem_rm_factory.cpp:129-138` | `out_cb_index` |
| Tilized-W | `moreh_getitem_tilized_factory.cpp:156-166` | `out_cb0_index` |
| Tilized-noW | `moreh_getitem_tilized_factory.cpp:418-427` | `out_cb_index` |

No kernel referenced index 16; every writer drains `c_0` (now `dfb::out` / `dfb::out0`) instead. Saves
one page of L1 per core in each shape.

### Q3 — rank-4 ROW_MAJOR with a last-dimension index: recorded, not fixed

Out of scope per the invoker: the guard is untouched and no dim-4 handling was added to the RM path. The
defect is written up under [Open items for downstream](#open-items-for-downstream). The one thing the
port had to settle — what becomes of the endpoint-less `c_5` allocation in that configuration — is the
ordinary dead-CB drop, scoped to the configurations that reach it: the RM index-DFB loop covers dims 0–3,
the reader's actual touch set (`moreh_getitem_rm_factory.cpp:166-190`). Zero functional change (the CB
was never touched, and the index tensor is still ignored exactly as before), and it keeps that
configuration from being rejected by the spec validator for a bindingless DFB.

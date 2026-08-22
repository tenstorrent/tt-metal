# Port Report — `experimental/transformer/nlp_concat_heads`

## Outcome

**`PORTED`** — the op's single factory (`NLPConcatHeadsProgramFactory`) is on
`ProgramSpecFactoryConcept`, with **both** of its internal configs (INTERLEAVED and SHARDED)
converted. No factories left for a later pass. Test set unchanged pre/post port: **219 passed,
2 skipped** (the 2 skips are the `grid_size0` = (12,8) parametrizations, skipped pre-port too on this
device's core grid).

One item needs an ops-team ruling before merge — see Handoff points #1. It does not affect any tested
configuration.

## Provenance

- **Recipe docs (this port):** `bcf38615192 2026-08-03 docs(metal_2.0): add the op-porting recipe set`
- **Audit docs (inherited):** `bcf38615192 2026-08-03 docs(metal_2.0): add the op-porting recipe set`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, plain (no op-owned tensors) — exactly the audit's decision, no revision.
`create_descriptor` → `create_program_artifacts`; the legacy parameter list already matched the
concept's `(attributes, tensor_args, tensor_return_value)` shape (`tensor_args_t = Tensor`,
`tensor_return_value_t = Tensor`), so no parameter unwinding was needed.

### Device-op-class edits

- Custom `compute_program_hash` deleted: **none** — the op was already on the default reflection-based
  hash.
- Pybind entry points removed: **none** — `nlp_concat_heads_nanobind.cpp` binds only the user-facing
  `ttnn.experimental.nlp_concat_heads`, never `create_descriptor`. No file outside the factory
  (`.hpp` + `.cpp`) was touched in the op directory.

### Open items

- **Relaxation candidates:** none applied and none obviously available. The op is *not* shape-agnostic
  — the reader bakes `in0_h_tiles` / `in0_w_tiles` / `in0_c` / `in0_HtWt` into CTAs and the sharded
  kernel bakes byte strides — so a `dynamic_tensor_shape` relaxation would be unsound here. Strict
  matching is the right default and is what the port ships.
- No capability missing from the concept was needed.

## Handoff points

1. **Ops team (op owners) — the `in_sharded && !out_sharded` hole, and the one construction decision
   it forced.** *This is the audit's own "Misc anomaly 1" reaching the porter exactly as the brief
   predicted, and it is the single item in this port that needs a decision from outside.*

   - **The pre-existing defect.** `validate_on_program_cache_miss`
     (`device/nlp_concat_heads_device_operation.cpp:48-51`) only forbids a `HEIGHT_SHARDED` output when
     the input is sharded, so an **INTERLEAVED** output on a sharded input passes validation and
     `compute_output_specs` (`:73-88`) builds a well-formed spec for it. The legacy factory allocated
     `cb_out0` (index 16) only when `out_sharded` (`nlp_concat_heads_program_factory.cpp:153` pre-port)
     while the SHARDED kernel constructs and writes through it on every path
     (`...sharded.cpp:33,36,44` pre-port). In that combination legacy ran a kernel against a circular
     buffer that was never created — silent undefined behaviour. No test covers it:
     `test_sharded_concat_heads` parametrizes only `[True, True]` and `[False, False]`
     (`tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py:1342`), and
     `test_nlp_concat_heads.py` is interleaved-only.
   - **What the port did, and did not, do.** It invented no semantic: no `TT_FATAL` was added, no
     validation changed, no device-op code touched. The one thing it could not avoid choosing was
     whether the `OUT` `DataflowBufferSpec` and its bindings are emitted conditionally on
     `out_sharded` (mirroring legacy) or unconditionally inside the SHARDED branch. It emits them
     **unconditionally**, because the kernel names `dfb::out` on every path and a Metal 2.0 kernel
     cannot reference a binding the spec does not declare. The conditional-binding pattern was not
     available: `#ifdef`-gating the kernel's output path would have required inventing an answer to
     "what does this kernel write when there is no output buffer" — which *is* the ops-team question.
   - **The observable consequence, stated plainly.** For the mixed config, `borrowed_from = OUTPUT`
     against a non-L1 output now trips the framework's own borrowed-DFB invariant
     (`tt_metal/impl/metal2_host_api/program_spec.cpp:1528-1533`, "TensorSpec is not L1-resident"), so
     a sharded input with an INTERLEAVED-DRAM output **fails loudly at program build** where legacy
     silently corrupted memory. That is the framework check firing, not a check the port added — but it
     is a behaviour change for a reachable, untested configuration and should be ratified, not
     inherited by accident.
   - **What we need.** A ruling: is the mixed config meant to be rejected outright (the audit's own
     suggested fix — a `TT_FATAL` requiring a sharded output whenever the input is sharded), or is a
     genuine interleaved-output code path wanted? Either way the fix belongs in a separate PR against
     the device-op class. If "reject outright" is the answer, this port already produces that outcome
     for the DRAM case and the explicit `TT_FATAL` is a clarity improvement rather than a functional
     one.

2. **Metal 2.0 / runtime team — the multi-binding disposition the audit's classification table
   produces for a two-toucher DFB is not constructible.** Full detail under Friction → Gaps #1. In
   short: `advanced_options.allow_instance_multi_binding` relaxes the per-node census's *upper* bound
   from "exactly one" to "at least one" per role
   (`tt_metal/impl/metal2_host_api/program_spec.cpp:1352-1362`) but keeps the requirement that every
   node host **≥1 PRODUCER and ≥1 CONSUMER**. A DFB with exactly two touchers both locked to PRODUCER
   — this op's SHARDED case — therefore has no legal binding assignment that uses the flag, and the
   only spec that validates is plain 1P+1C. Worth either a note in the classification table or a
   validator message that says so.

3. **Ops / Metal-2.0-track — the misplaced typecast fork of the shared writer.** A converted Metal 2.0
   fork of `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` already existed at
   `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
   (from `cbde3d44ff3`), i.e. in typecast's own tree rather than beside its donor. The locational rung-1
   check therefore reported "no fork" and this port created the correctly-placed one at
   `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`.
   The donor pool is now served by **two** divergent Metal 2.0 entry points. The new fork deliberately
   adopts the typecast fork's binding vocabulary verbatim (`dfb::out`, `tensor::output`,
   `args::num_pages`, `args::start_id`) so the two are drop-in interchangeable and consolidating them
   is a pure delete-and-repoint. Relocating/merging the typecast copy is not this port's edit to make.
   (The same misplacement was corrected for `transpose_wh` in `5ecda11bb71`.)

No kernel-lib gaps, no boundary-rule assumption violations (no call site required `sem::` or
`tensor::` to cross out of the op), no framework gaps beyond #2, no removed pybind surface.

## Successes

- **[Two-toucher DFB → assign 1P+1C](port_patterns.md) fired correctly, and the recipe's "re-derive,
  don't transcribe" instruction is what caught it.** The brief and audit both prescribed the
  multi-binding flag on `(cb_src0, SHARDED)` and `(cb_out0, SHARDED)`. Following that literally would
  have produced a spec that fails validation at the first sharded test. The catalog's step 3
  ("Treat a brief's endpoint disposition as 'this DFB needs endpoint attention', then run the census
  yourself and follow *it*") plus the "guard against stacking" note pointed straight at the 1P+1C
  assignment, which is what the code ships
  (`device/nlp_concat_heads_program_factory.cpp:135-171`) and what passes.

- **The scope-discipline section stopped a tempting cleanup.** The factory carries a provably-dead
  `row_major` variable and a dead `grid_to_cores` result, the sharded kernel a dead
  `single_tile_size_bytes` and a commented-out `push_back`, and the factory a stale `// 142` comment.
  Every one of them is a one-line delete. All were left in place (the dead `row_major` disappeared only
  as a mechanical consequence of the branch restructure — see Open items) and routed here instead.
  §Scope discipline's four compounding reasons — attribution loss in particular — are what made that
  feel like the right call rather than laziness.

- **"Go to the headers first" was strictly faster than hunting precedents.** Two of this port's
  load-bearing facts came straight from source, not from any ported op: the borrowed-DFB L1-residency
  invariant (`dataflow_buffer_spec.hpp:116-130` and its enforcement in `program_spec.cpp:1505-1550`),
  and the per-node census's behaviour under the multi-binding flag (`program_spec.cpp:1250-1389`).
  The second one is what turned Friction #1 from a debugging session into a planning-step decision.

- **The `opt_level` trap was pre-empted by the recipe's "grep, don't read `config`" instruction.**
  `grep -n opt_level` on the legacy factory returns nothing, and all three kernels are DM, so the
  resolved legacy level is `O2` — which is also Metal 2.0's `CompilerOptions` default. Nothing to set.
  Had any kernel been a compute kernel, reading `ComputeConfigDescriptor` (which has no `opt_level`
  field at all) would have suggested the same "nothing to do" and silently dropped a level.

## Friction

### Gaps

1. **The multi-binding disposition and the validator disagree, and no doc says so.** The audit's
   [CB endpoints](metal2_audit.md) classification table and the catalog's endpoint-assignment procedure
   both route "≥2 kernels locked to the same FIFO role" to
   `advanced_options.allow_instance_multi_binding = true`. This op's SHARDED config is exactly that
   census: two same-source instances, both calling `reserve_back` on both borrowed DFBs
   (`...sharded.cpp:35-36`), with no consumer anywhere. But the flag cannot express it. Reading
   `tt_metal/impl/metal2_host_api/program_spec.cpp:1352-1362`: under the flag the per-node census's
   check becomes `num_producers >= 1 && num_consumers >= 1` — the *upper* bound relaxes, the
   *lower* bound does not. So a node with two producers and zero consumers is rejected with or without
   the flag. With only two touchers available there is no third kernel to be the consumer, and the
   self-loop escape is closed twice over (the recipe forbids stacking self-loop with multi-binding, and
   the validator's self-loop rule requires the producer and consumer *kernel sets* to be equal, which
   `{A,B}` vs `{A}` violates). **The only spec that validates is 1P+1C** — which is the two-toucher
   pattern's own answer, and it needs no flag.

   What would have helped: one sentence in the classification table saying that the flag raises the
   per-role ceiling but never removes the "≥1 of each role per node" floor, so a **two**-toucher census
   always resolves to 1P+1C regardless of role locks, and the flag only becomes reachable at ≥3
   touchers. That also reconciles the table with the recipe's own self-audit item ("A two-toucher
   work-split … is a **1P+1C assignment, not a flag**"), which is already correct — the table is the
   part that over-triggers. The audit's *own* Recipe note 1 arrived at the same discomfort from the
   other direction (asking whether a provably-non-blocking `reserve_back` should lock a role at all);
   the constructibility argument settles it without needing that judgment.

2. **`get_tile_size(cb_id)` → `dfb.get_tile_size()` is available on DM kernels, but the whitelist
   doesn't say so and the header suggests otherwise.** Whitelist §A maps the free helper to the member
   getter, and the getter is gated on `DFB_DESCRIPTORS_DEFINED`, whose definition
   (`dataflow_buffer.h:28-31`) is `__has_include("chlkc_descriptors.h")` — plus the surrounding
   comment says "PACK TRISC uses `pack_*`; UNPACK/MATH TRISC **and DM** use `unpack_*`", which reads as
   reassurance but not a guarantee that a DM build gets the header at all. Both of this op's readers
   need the call (`reader_tm_tile_layout_nlp_concat_heads.cpp:30`,
   `...sharded.cpp:34`), and it does work. Confirming that cost a detour through
   already-ported DM kernels (`data_movement/clone`, `data_movement/sort`) to establish by example what
   the whitelist could have stated in a clause. A note in §A — "the metadata getters are available in
   DM builds as well as compute ones" — would remove the detour.

3. **No stated convention for the *host-side* DFB `unique_id` when the legacy name was a CB index.**
   The recipe is precise about kernel-side accessor names (name for the kernel's role, and for a shared
   kernel that vocabulary is inherited), but the `DFBSpecName` / `TensorParamName` / `KernelSpecName`
   values are free choices with no guidance. This matters more than it looks for a *shared* kernel:
   the fork's accessor name (`dfb::out`) is fixed by the pool, while the DFB it maps to is this op's
   `cb_src0`. So `DFBBinding{.dfb_spec_name = SRC0_DFB, .accessor_name = "out", …}` is correct and
   reads like a mistake, and there is nothing in the docs to point a reviewer at. A sentence noting
   that the two names are deliberately independent — spec-side names describe the *program's* buffers,
   accessor names describe the *kernel's* view — would help.

### Confusion

1. **"The atomic unit is one ProgramFactory" is clear; what it means for an intra-factory branch is
   not.** This op has one factory whose `create_descriptor` branches into two configs sharing no
   kernel, no CB layout, and no binding classification — effectively two ports in one function. The
   recipe's per-factory framing (and its "port one factory now, the rest later" lever) offers no
   sub-target here: both branches must convert together because they are one function. That is the
   right answer, but it was reached by elimination rather than by reading it anywhere. The audit's own
   Recipe note 3 raises the same vocabulary gap from the audit side; it applies verbatim to the port
   recipe's [atomic-unit note](metal2_port.md). One clause — "a factory that branches internally
   converts as a whole; the per-factory lever does not subdivide it" — would close both.

2. **Near-miss on the borrowed-DFB `TensorParameter`, avoided by the validator's own source.** In the
   SHARDED config no kernel builds a `TensorAccessor` at all, so neither `INPUT` nor `OUTPUT` carries a
   `TensorBinding` — only a `borrowed_from`. The migration guide's rule ("every `TensorParameter` needs
   ≥1 `TensorBinding` across the program's kernels", `migration_guide.md` §TensorParameter) reads as
   though that spec must be rejected. It isn't: `program_spec.cpp:533-546` registers a `borrowed_from`
   as a parameter *use*, so a borrow alone satisfies the requirement. The guide's own borrowed-memory
   paragraph sits in the DFB section and doesn't mention the interaction. Two sentences apart in the
   same document would have saved reading the validator.

## Open items for downstream

### Shared kernel touches

| kernel path | rung taken | notes |
|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **rung 2 — created the fork** at `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` (beside the original). Pointer comment **landed** in the legacy original (lines 5-9); nothing else in it changed. No `CMakeLists.txt` edit (kernels are JIT'd from the source tree; the per-family install glob already covers the directory). | Rung 1 re-verified at port time by `ls` of the donor's directory: no `*_metal2*` sibling existed. Binding vocabulary: `dfb::out`, `tensor::output`, `args::num_pages`, `args::start_id` — taken verbatim from the misplaced typecast fork so the two are interchangeable (Handoff #3). |

**Remaining unmigrated consumers of the legacy `writer_unary_interleaved_start_id.cpp`** — the sunset
checklist; the legacy copy can be deleted once this list is empty. 36 non-quasar files bind the
filename (`grep -rl`, quasar excluded), spanning:
`data_movement/{concat, copy, tilize, tilize_with_val_padding, transpose, slice, permute, reshape_on_device, bcast}`,
`reduction/{generic, prod}`, `matmul`, `embedding`, `kv_cache`,
`eltwise/unary_backward/{gelu_bw, tanh_bw}`, `experimental/matmul/attn_matmul`,
`experimental/unary_backward/gelu_backward`, `experimental/transformer/nlp_concat_heads_boltz`,
`examples/example`, plus two `tt_metal/programming_examples` and two generic-op tests.
`nlp_concat_heads` is now **off** this list.

The op's other two kernels
(`device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp`,
`device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`) are private to this op —
a repo-wide grep finds no other binder — so both were converted in place, no fork.

### Findings routed here rather than fixed (all pre-existing; each is a separate-PR candidate)

1. **Dead `row_major` / dead `grid_to_cores` in the factory.** Pre-port, `row_major` was assigned only
   inside the `if (in_sharded)` block (`:59`) and read only by
   `grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major)` (`:167`), whose result `cores` was
   consumed only in the *interleaved* branch (`:191`) — where `row_major` is always `false`. So the
   shard orientation it read had **no effect on any code path**. The port's branch restructure makes
   `grid_to_cores` local to the interleaved branch with an explicit `/*row_wise=*/false`, and
   `row_major` therefore disappears. This is the one legacy line the port removes rather than
   translates; it is zero-functional-change by the argument above, but flagging it explicitly since
   "dead" is a claim a reviewer should check rather than take on trust.
2. **Hardcoded `row_wise=true` against a possibly COL_MAJOR shard.** The SHARDED RTA loop iterates
   `corerange_to_cores(all_cores, num_cores, /*row_wise=*/true)` (carried over verbatim) while the
   shard may be `COL_MAJOR` — the sharded test uses `ShardOrientation.COL_MAJOR`. Harmless **only**
   because every core in that branch receives byte-identical RTAs, so iteration order cannot matter.
   The invariant is undocumented and breaks the moment those args become per-core.
3. **The SHARDED RTAs are really CRTAs.** Every core gets the same three values, so
   `common_runtime_arg_values` would be strictly better for dispatch efficiency. Not converted here —
   RTA→CRTA changes dispatch semantics and the recipe explicitly defers it to a later cleanup pass.
   This is a clean, self-contained follow-up.
4. **Dead local in the sharded kernel.** `single_tile_size_bytes` (`...sharded.cpp:34` post-port) is
   computed and never used — the loops stride by `head_dim_size_bytes` and `out_row_size_bytes`.
   Translated mechanically (`get_tile_size(cb_id)` → `dfb_in0.get_tile_size()`) rather than deleted.
5. **Vestigial FIFO calls in the sharded kernel.** `dfb_in0.reserve_back(block_size)` (its own comment
   says `// Redundant`) and `dfb_out0.reserve_back(block_size)` with the matching `push_back`
   commented out (`:63`), on buffers sized so the reservation is unconditionally satisfiable. Left
   exactly as found. If the ops team drops both `reserve_back` calls, the census becomes two role-free
   touchers — still 1P+1C, so it would not change this port's bindings, only remove the tension the
   audit's Recipe note 1 describes.
6. **Stale comment.** `nlp_concat_heads_program_factory.cpp` — `per_tensor_tiles = … ; // 142` is a
   leftover Falcon-7B-specific value, not a general invariant (and `per_tensor_tiles` is recomputed for
   the sharded case anyway). Carried over verbatim.
7. **One stale comment deliberately dropped.** The sharded kernel's `// interleaved accessor args`
   sat above the CTA block and described nothing — there is no accessor in that kernel, in either the
   legacy or ported form. It labelled the two `get_compile_time_arg_val` CB-index reads the port
   removes, so keeping it would have left an actively misleading line above named CTAs. This is the
   only comment the port deletes; flagged because comment preservation is otherwise strict.

### Test coverage notes

- **The confirmed test set was discovered, not invoker-supplied — please sanity-check it.**
  Broad sweep (`find tests -iname '*nlp_concat_heads*' -name '*.py'` plus a `grep -rl` across all test
  trees), filtered to *this* op:
  - `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads.py` — INTERLEAVED
    coverage (dtype × in/out buffer-type sweep, plus `test_nlp_concat_heads_with_program_cache`).
  - `tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py::test_sharded_concat_heads` —
    the **only** SHARDED coverage.
  - Excluded as different ops: `nlp_concat_heads_decode`, `nlp_concat_heads_boltz`.
  - No C++ gtest exists (`grep` for `nlp_concat_heads` / `NLPConcatHeads` across `tests/ttnn/**/*.cpp`
    returns nothing), so there was no gtest layer to run first.
  - Indirect model-level users not run as part of the port: `tests/ttnn/distributed/test_multidevice_TG.py`,
    `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py`, and the two
    `tests/sweep_framework/sweeps/model_traced/nlp_concat_heads*_model_traced.py` sweeps.
- **The SHARDED config rests on a single test case.** With `grid_size0` = (12,8) skipped on this
  device, SHARDED coverage post-port is exactly one parametrization
  (`test_sharded_concat_heads[DataType.BFLOAT8_B-sharded-in0_shape1-grid_size1]`, a 1×4 grid). It
  passes, and it is the case that exercises both borrowed DFBs, the two same-source instances, and the
  1P+1C assignment — but a config this structurally distinct deserves more than one case, and a
  wider-grid run on a machine with a 12×8 core grid would be worth doing before merge.
- **The `in_sharded && !out_sharded` combination has no test at all**, pre- or post-port. Whichever way
  Handoff #1 is ruled, the resolving PR should add one.

# Metal 2.0 Port Report — `experimental/ssm/repeat_and_interleave_eltwise_mul`

## Outcome

**`PORTED`** — the op's only factory, `RepeatAndInterleaveEltwiseMulProgramFactory`, is converted
from `ProgramDescriptorFactoryConcept` (`create_descriptor`) to `ProgramSpecFactoryConcept`
(`create_program_artifacts`), together with all three kernel entry points it binds. No factory
is left behind: the device-op's `program_factory_t` variant holds this one factory only.

**Not yet verified on hardware.** The invoker retained the build and the test run, so this port
has **not** been compiled and has **not** been run on an N150. Everything below the build line —
compile-time correctness of the spec construction, spec-validator acceptance, and numerics across
the three configurations — is unverified by me. What *was* verified is recorded under
*Verification performed*; the commands the invoker needs are at the end.

---

## Provenance

```
git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/
```

prints nothing in this checkout — the op's clone carries no `metal_2.0` doc tree, so the version
cannot be pinned from here. Pinned instead from the doc-branch checkout at
`/localdev/edwinlee/Port_Recipe` (branch `akertesz/op-porting-recipe`), where the same command
gives:

- **Recipe docs (this port):** `32720e020e2 2026-08-05 docs(metal_2.0): clear two stale references`
- **Audit docs (inherited):** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

The port ran against `/localdev/edwinlee/metal2_port.md`, verified byte-identical (`diff`) to
`ai/port/metal2_port.md` at `32720e020e2`, plus the four shared docs
(`port_patterns.md`, `migration_guide.md`, `ttnn_factory.md`, `cb_dfb_api_whitelist.md`) read
directly from that checkout.

> **Friction — the audit's pinned doc hash no longer resolves on the doc branch.** `4386dc456a1`
> still exists as a loose object but is **not an ancestor of the doc branch HEAD**
> (`git merge-base --is-ancestor 4386dc456a1 HEAD` fails); the branch was rewritten between the
> audit and the port, and the commit's content now lives at `b0dcb05f3b2` under the same subject
> line. A reader reconstructing the audit's guidance from the hash alone will get *"unknown
> revision"* on a fresh clone. See *Friction → Gaps* for the suggested fix.

---

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, as the audit chose. No re-decision, no disagreement.

`RepeatAndInterleaveEltwiseMulProgramFactory::create_program_artifacts(const RepeatMulParams&,
const RepeatMulInputs&, Tensor&)` returns a `ttnn::device_operation::ProgramArtifacts` carrying
`spec` + `run_params`. `op_owned_tensors` is left defaulted — the factory allocates nothing
beyond the op's io (no `CBDescriptor` set `.buffer`; there is no config or index-table tensor).

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — the op never had one. (The readiness sheet's
  `Formerly custom hashed? = yes` refers to a hook removed before the audit; nothing in the tree
  today.) The device-operation class files are **byte-identical** pre- and post-port.
- **Pybind entry points removed:** none. `..._nanobind.cpp:23-32` binds the plain host function
  through `ttnn::bind_function<"repeat_and_interleave_eltwise_mul", "ttnn.experimental.">`; no
  `nb::class_` of the device op and no `create_descriptor` exposure, so the vanishing factory
  entry point has no Python surface. The nanobind files are untouched.
- **Factory parameter dropped for a pybind hook:** none.

The five files the port touches are the two factory files and the three kernels. Nothing else in
the op directory changed.

### Open items

- **Relaxation candidates:** none identified. All three `TensorParameter`s are strict. No kernel
  uses `ArgConfig::Runtime*` (`RuntimeTensorShape` / `RuntimeShardShape` / `RuntimeBankCoords`),
  so the pre-migration check does not fire, and the audit's sheet row records `TensorParameter
  relaxation = none`. There is no custom hash to mine for a candidate.

  That said, the kernels *look* like they would tolerate `dynamic_tensor_shape` on `src0` /
  `src1` / `dst`: none of the three bakes in a shape — every dimension the kernels use arrives
  as an RTA (`in1_num_blocks_h`, `in1_num_blocks_w`, `in0_num_blocks_w`, `out_num_blocks_h`,
  `out_total_blocks_w`). Widening cache equivalence across batch sizes would be worth measuring.
  Deliberately **not** applied: the port is not the place to self-decide a relaxation, and the
  op's admissible widths are pinned by `TT_FATAL`s anyway
  (`..._device_operation.cpp:73-76`), so the cache-equivalence win is limited to the `ashape[2]`
  (batch) axis.
- **Capabilities not yet on this concept that the op would benefit from:** none. The op is
  genuinely single-program SPMD with no op-owned resources — the cleanest possible fit.
- **Concept-fit friction:** none. `create_descriptor` → `create_program_artifacts` was a
  signature-for-signature swap; the framework adapter needed no help.

---

## Handoff points

**No capitulation, no boundary-rule violation, no kernel-lib gap, no framework gap, no removed
pybind surface.** The port stayed entirely inside the op directory, and nothing required a
`sem::` or `tensor::` handle at an out-of-op call site (the op uses no semaphores at all, and both
`TensorAccessor` constructions are inside the op's own kernels).

Two items are for the **ops team**, not the framework, and both were raised by the audit before
the port. Neither is a port blocker; both are recorded here so the port's shape is traceable to
them.

### 1. `c_25` / `in1_transposed` is popped twice per push — port ships the multi-binding option

**Owner:** ops team (ssm). **Not fixed here** — it is a behavior change, out of port scope.

Per in1-block iteration the compute kernel pushes one tile
(`device/kernels/ssm_eltwise_mul.cpp`, the `reserve_back` / `push_back` pair in the
`REPEAT_INTERLEAVE_IN1` branch) and **two** kernels pop it: compute itself
(`ssm_eltwise_mul.cpp`, the `dfb_in1_transposed.pop_front` at the bottom of the in1-block loop)
and the reader (`reader_ssm_eltwise_mul.cpp`, `wait_front` before the face loops and `pop_front`
after them). Two acks per push means the producer's free-space check can never block, so the
buffer's two-entry double buffer provides no protection against compute overwriting a tile the
reader is still reading. It is masked today by an implicit handshake — compute cannot reach the
next push until the reader has produced the `in1_bcast_row` rows compute waits on.

**Consequence for the port:** the census is 1 producer + **2 locked consumers** on one node,
which no relabelling reduces to 1P+1C, so `IN1_TRANSPOSED`'s
`advanced_options.allow_instance_multi_binding` is set to `true` under Configs A and B
(`..._program_factory.cpp`, the `IN1_TRANSPOSED` `DataflowBufferSpec`). The flag is a hard error
on Gen2, so it books Quasar debt that the redundant pop is the sole cause of. **If the ops team
removes the compute-side pop, `IN1_TRANSPOSED` collapses to a plain 1:1, the flag comes off, and
the buffer regains real backpressure** — three wins from a one-line change, sequenced on their
own branch.

### 2. Config C's four zero-endpoint DFBs — bound, not dropped (the audit's open question)

**Owner:** ops team (ssm), for the optional cleaner end state. **Resolved for the port.**

The audit's *Questions* item 1 asked the user to confirm the disposition of `c_24`, `c_25`,
`c_26`, `c_27` under Config C, where the factory allocates all seven CBs but the kernels' every
*access* to those four is compiled out. The invoker's go-ahead did not answer it, so the port
took the audit's own recommendation — **bind them rather than drop them** — for the reason the
audit gave: dropping shrinks Config C's L1 footprint (a behavior change) and forces `#ifdef`
guards onto four kernel-side `DataflowBuffer` constructions plus the `pack_reconfig_data_format`
metadata reference, which is kernel surgery the port would not otherwise do. Binding keeps
footprint and runtime behavior byte-identical to legacy and needs **zero** kernel edits.

**The cleaner end state remains available to the ops team**, exactly as the audit framed it: make
the *host-side* DFB allocations config-conditional, and gate the matching kernel-side
constructions. That would drop four buffers' worth of L1 in Config C. It is a functional change
and belongs on the ops team's track, not in a port diff.

---

## Successes

- **Patterns catalog, *Two-toucher DFB → assign 1P+1C* — the endpoint-assignment procedure caught
  a brief over-read, in the direction the recipe predicted.** The recipe insists
  the porter *re-derive* endpoint dispositions from the census rather than transcribe them, and
  that paid off here. The brief's recommendation for Config C was "self-loop each from the kernel
  that constructs its wrapper." Re-running the census showed that `in1_transposed` and
  `in1_bcast_row` are named by **two** kernels under Config C (reader *and* compute), and the
  procedure is explicit that a self-loop is a *one-toucher* resolution — with two candidates you
  assign 1P+1C. So those two get one role each (`..._program_factory.cpp`, the reader's
  `IN1_TRANSPOSED` CONSUMER / `IN1_BCAST_ROW` PRODUCER bindings against compute's
  `IN1_TRANSPOSED` PRODUCER / `IN1_BCAST_ROW` CONSUMER), while only `in0_transposed` and
  `out_transposed` — compute-only in every config — self-loop. Following the brief literally
  would have produced two self-loops that the "guard against stacking" rule flags as the tell of
  a mis-slotted 1:1, *and* would have left `in1_bcast_row` with two producer endpoints and no
  consumer under Config C. The re-derivation instruction is load-bearing; keep it.
- **The self-popping-producer multi-binding trigger was correctly pre-flagged, and the *"Watch
  for"* entry made it cheap to confirm.** The brief told me the extra `c_25` endpoint was a
  producer-side `pop_front`, not a hidden raw writer, and named both sites. Confirming that took
  one read of each kernel instead of a hunt, and it stopped me reaching for 1P+1C by reflex — the
  prose framing "most CBs with two touchers are 1P+1C" points the wrong way for this shape, and
  the audit's own *Recipe notes* already suggest adding a fourth "self-popping producer" face.
  Seconded.
- **The `Compiler options` section caught the one setting nothing else would have.** `grep -n
  opt_level` on the legacy factory returns **zero hits**, which reads as "nothing to carry." The
  recipe's rule 2 is exactly about that trap: an absent `KernelDescriptor::opt_level` resolves to
  `O3` on a `ComputeConfigDescriptor` but Metal 2.0's `CompilerOptions` defaults to `O2`, so the
  compute `KernelSpec` needs an explicit `KernelBuildOptLevel::O3`
  (`..._program_factory.cpp`, the compute `KernelSpec`'s `compiler_options`). Nothing in the
  build or the tests would have flagged the silent level drop. The section's "answering *did it
  set one?* is the part that goes wrong" framing is accurate and earned its place.
- **The Style A / Style B fork in `Hardware configuration` prevented a silent precision flip.**
  This op sets a Metal `ComputeConfigDescriptor` directly (`math_fidelity` from
  `operation_attributes`, `fp32_dest_acc_en = false`, `math_approx_mode = false`) with no TTNN
  `ComputeKernelConfig` anywhere. Routing it through `to_compute_hardware_config` to save typing
  — the obvious shortcut, and the one the section explicitly forbids — would have taken the TTNN
  helper's high-performance defaults, flipping `sfpu_precision_mode` from the legacy `Precise`
  to `Approximate`. Building `ComputeGen1Config` field by field
  (`..._program_factory.cpp`, `compute_hw_config`) preserves all six resolved values.
- **The generated-doc citation ban.** I had drafted a code comment pointing at this report for the
  `c_25` double-pop rationale — precisely the "see the report's Open items" form the recipe names
  as the most common offender. The rule caught it; the comment now states the reason inline at the
  `IN1_TRANSPOSED` spec instead, and the diff cites no `.md` at all
  (`git diff --name-only | grep -E '\.(cpp|hpp|h)$' | xargs grep -nE '\.md'` → zero hits).

---

## Friction

### Gaps

- **The dead-CB rule still has no shape for a *config-scoped* zero-toucher whose index is still
  named — the audit predicted this, and it is the port's single largest judgment call.**
  The recipe's construction step says a brief-flagged dead CB (zero endpoints) gets **no** spec:
  "drop the allocation and any dead CTA carrying its index." But that guidance is written for an
  op-wide dead CB. Here four CBs are genuinely live in two configs and untouched in a third, and
  in that third a kernel still *names* them — via an unconditional `DataflowBuffer` construction
  and, for `c_24`, via `pack_reconfig_data_format`. Metal 2.0's actual constraint is by **token**,
  not by access: a `dfb::name` must exist for every DFB a kernel names, whether or not it touches
  it. So the literal dead-CB rule and the "every named DFB needs a token" requirement point at
  opposite actions, and the cheap answer — bind it with cosmetic roles — appears in neither.
  **Suggested fix:** give the census an explicit **named-but-untouched** category whose resolution
  is "bind it; assign roles by the *naming* count (1 namer → self-loop, 2 namers → 1P+1C)", and
  state plainly that a format-metadata call (`pack_reconfig_data_format`,
  `reconfig_data_format_srca`, `binary_op_init_common`) obliges a binding even though it touches
  no memory. That last sentence is the one that decides the whole question, and no doc currently
  contains it. (The audit's *Recipe notes* raise both halves; this port is the confirming case.)
- **The recipe's DFB-metadata rule (whitelist rule 7) does not say whether the member getters are
  available on the data-movement path.** Both DM kernels here read `get_tile_size(cb_id)`
  (`reader:46-47`, `writer:23`), and rule 7 says to rewrite those as `dfb.get_tile_size()`. But
  `DataflowBuffer::get_tile_size()` is gated on `DFB_DESCRIPTORS_DEFINED`, and the whitelist's
  own note — "PACK TRISC uses `pack_*` arrays; UNPACK/MATH/DM use `unpack_*`" — only implies DM
  availability. I had to establish it from the headers: `DFB_DESCRIPTORS_DEFINED`
  (`api/dataflow/dataflow_buffer.h:28-31`) and the legacy free helper's `DATA_FORMATS_DEFINED`
  (`api/dataflow/dataflow_api.h:8-11`) are keyed on `__has_include("chlkc_descriptors.h")` — the
  *same* condition — so the member getter is available in exactly the builds where the free
  helper was. **Suggested fix:** one line in the whitelist's metadata table stating that
  equivalence. It converts a 15-minute header dig into a lookup, and every DM kernel that reads a
  tile size hits it.
- **No local `clang-format`, and the recipe's verification step does not mention formatting.** The
  repo's `.pre-commit-config.yaml` runs `clang-format`, but no `clang-format` binary exists in
  this environment (`which clang-format` → nothing; no `python_env`), so the port's formatting is
  hand-matched to the file's existing style (120 columns, 4-space indent — verified: no line
  exceeds 120). **The invoker should expect one `pre-commit run --files ...` pass to adjust
  wrapping**, particularly in the designated-initializer blocks, which clang-format re-flows
  aggressively. Worth a line in the recipe's Verification section alongside the build: a port that
  cannot run the formatter should say so rather than leave the reviewer to discover it.

### Confusion

- **`DFBAccessor` vs `DFBBindingToken` — the recipe's name for the DFB handle type does not exist
  in the tree.** The recipe and the patterns catalog both attribute the `dfb::name → uint32_t`
  bridge to `DFBAccessor::operator uint32_t()`, and the anti-pattern entry warns against
  `.id`-extraction from a `DFBAccessor`. The type is actually **`DFBBindingToken`**
  (`api/dataflow/dataflow_buffer.h:46-58`); `DFBAccessor` appears nowhere in the repo. The
  behaviour is exactly as documented — the `constexpr operator uint32_t()` is there and every LLK
  call site in this port relies on it — so nothing broke, but a porter who greps the recipe's type
  name to confirm the conversion exists finds nothing and has to reconstruct the mechanism from
  the generated-header emitter. Two other stale spellings in the same family: the recipe says
  `TensorParameter::advanced_options` holds `dynamic_tensor_shape` / `match_padded_shape_only`,
  but the field is `TensorParameter::relaxations` of type `TensorSpecRelaxations`
  (`tensor_parameter.hpp:45`), and the migration guide's `KernelSpec::DFBEndpointType::PRODUCER`
  spelling works only via the namespace-level `DFBEndpointType` alias. All three are one-word
  fixes.
- **Whether the port should adopt the `TT_KERNEL` entry point was unclear, and the answer is
  "no" — but only the JIT source says so.** `experimental/kernel_args.h` (the one header the
  recipe tells you to add) opens with a `TT_KERNEL` macro documented as "marks the named-arg entry
  point; the JIT generates `kernel_main()` from its signature." That reads like the Metal 2.0 way
  to write a kernel entry, and it is not mentioned anywhere in the recipe, whose examples all keep
  `void kernel_main()`. I resolved it from `tt_metal/jit_build/genfiles.cpp:331-335`: a source with
  no `TT_KERNEL` marker is "a legacy / hand-written `kernel_main()` — fully backward compatible,"
  and the only in-tree users are `tests/tt_metal/.../tt_kernel_named_args_*`. So plain
  `kernel_main()` is correct for a port, and all three kernels keep it. **Suggested fix:** one
  sentence in the kernel-side whitelist saying `TT_KERNEL` is a separate, newer entry-point
  mechanism that ports do **not** adopt — otherwise every porter who reads the header they were
  told to add will stop on it.
- **The two-level `hw_config` variant conversion is a live footgun the recipe writes as if flat.**
  `KernelSpec::hw_config` is `variant<DataMovementHardwareConfig, ComputeHardwareConfig>`, and
  each of those is *itself* a variant over a Gen1 and a Gen2 config. The recipe's examples show
  `.hw_config = DataMovementGen1Config{...}` / a bare `ComputeGen1Config`, i.e. assigning the
  innermost type straight into the outer variant — two conversions deep. The DM helpers
  (`create_reader_datamovement_config`) return the *middle* type, so they are safe, but the
  compute Style-B example is not obviously so. I sidestepped it by declaring
  `const ComputeHardwareConfig compute_hw_config = ComputeGen1Config{...}` and assigning that
  (`..._program_factory.cpp`) — one conversion at each step, and the shape a couple of already-ported
  ops in the tree independently landed on. **Suggested fix:** show the middle-type intermediate in
  the recipe's compute example rather than the bare inner config.

---

## Open items for downstream

- **Shared kernel touches: none.** All three kernel sources are op-owned and bound by no other op
  or test (`grep -rl <filename> ttnn/cpp/ttnn/operations/` returns only this op's factory for each).
  No `_metal2` fork existed beside any of them and the port created none, so there is no pointer
  comment, no sunset checklist, and no remaining unmigrated consumer. The sibling ssm ops
  (`prefix_scan`, `ssm_1d_sum_reduce`) carry their own private kernel files.
- **RTAs that are really CRTAs — a follow-up cleanup, deliberately not done here.** Six of the
  eleven named runtime args hold the **same value on every node**: reader `in1_num_blocks_h`,
  `in1_num_blocks_w`, `in0_num_blocks_w`; writer `out_num_blocks_h`, `out_total_blocks_w`;
  compute `in1_num_blocks_h`. Promoting them to `common_runtime_arg_names` would cut per-node
  dispatch traffic by roughly half on a 64-node grid. Not done: RTA→CRTA changes dispatch
  semantics, and the recipe is explicit that it is a separate pass. The kernel side needs **no**
  change at all when it happens — `get_arg(args::name)` is identical for both kinds — so this is a
  host-only, low-risk follow-up.
- **The `runtime_arg_values` loop is still node-first, bridged by `AddRuntimeArgsForNode`.** The
  legacy per-core loop is preserved verbatim (same iteration order, same group selection) and the
  helper transposes into the name-first table. A name-first restructure is the recommended end
  state but buys nothing for the port and adds transposition risk; leaving it is the recipe's own
  advice. Whoever does the CRTA pass above should do this at the same time.
- **Two known-inert reader oddities carried forward untouched** (audit *Misc anomalies*, ops-team
  owned):
  - `reader_ssm_eltwise_mul.cpp` — the hardcoded `5120` in the third/fourth-face page-id
    computation, where the structurally identical first/second-face loop above it derives the same
    stride from the `in0_num_blocks_w` RTA. Inert today because that code compiles only under
    `#ifndef REPEAT_IN0`, which forces `in0_num_blocks_w == 5120`; it mis-addresses the moment
    `HIDDEN_SIZE` changes or a new width is admitted. Suggested fix: use `in0_num_blocks_w`.
  - Reader RTA `in0_num_blocks_w` is **dead in Configs A and C** (read only inside the
    `#ifndef REPEAT_IN0` branch) yet supplied unconditionally on every node. It survives the port
    as a named RTA rather than a positional one, which at least makes the deadness legible.
- **Perf must be measured, not assumed.** The readiness sheet flags this op `Pointer patching perf
  issue? = suspect perf regression (+ fixed latent bug)` and classifies it `PD Op
  (pointer-patching)` — the mechanism Metal 2.0's typed `TensorBinding` supersedes, so the port
  *plausibly is* the fix. **I could not measure it: no build and no device run in this
  environment.** The three legacy `Buffer*` RTAs are now `TensorBinding`s whose base addresses
  ride implicit CRTAs patched by the framework on cache hit, which is the change that should move
  the number. A before/after on `test_ssm_eltwise_mul_with_program_cache` (which drives all three
  configs twice, cache-cold then cache-hot) is the cheapest signal; the mamba model perf test is
  the meaningful one.
- **Doc-evolution suggestion (broader than a Gap entry): pin recipe provenance by content, not by
  commit.** The audit's `4386dc456a1` no longer resolves on the doc branch after a rewrite (see
  *Provenance*), which defeats the point of recording it. Recording the doc *subject line* plus a
  content hash of the specific files read (e.g. `git hash-object ai/port/metal2_port.md`) would
  survive a rebase. Cheap change to the Provenance section of both the audit and the port
  templates.
- **Test-coverage note the verification step surfaced but did not act on.** The op's unit test
  parametrizes memory configs across `{DRAM, L1}³` and dtypes `{bfloat16, bfloat8_b}` × three
  `(in0_W, in1_W)` pairs × two batches = 288 cases, which is thorough on the *tensor* axis. What
  it does not vary is the **grid**, and the port's work split is grid-driven
  (`split_work_to_cores` over `bshape[-1] / TILE_WIDTH` blocks — 160 for Configs A/B, 5120 for
  Config C). Neither count divides evenly across an N150's storage grid, so the **two**-core-group
  path is the one that always runs, and the single-group case (`core_group_2` empty, either from
  even divisibility or from fewer blocks than cores) is never exercised. So the per-group RTA
  split *is* covered, but incidentally rather than by design, and its complement is not covered at
  all. Not a port regression — the legacy code had exactly the same coverage — but if the group
  selection ever breaks, no test names it.

---

## Verification performed

What I actually ran, so the invoker knows where the unverified line falls.

**Not run:** `./build_metal.sh --build-tests` (no build tree in this checkout, and the invoker
retained the build), and every test. **Consequence:** no compile, no spec-validator run, no
numerics, no perf number.

**Ran and passed** (all over the op directory, working tree):

| Check | Result |
|---|---|
| `buffer()->address()` survivals | zero |
| `TensorAccessorArgs` survivals | zero |
| `CircularBuffer` / `CBDescriptor` / `CBFormat` / `CBIndex` survivals | zero |
| Leftover CB names — `grep -rnE '[Cc][Bb]_\|_[Cc][Bb]\b\|\b[Cc][Bb]\b\|\bCB[A-Z]'` | zero |
| `.id` extraction on a `dfb::` handle | zero |
| Positional args — `get_compile_time_arg_val` / `get_arg_val` / `get_common_arg_val` / `get_vararg` | zero (every CTA and RTA is named; no varargs) |
| `TT_FATAL` / `TT_ASSERT` / `TT_THROW` per-file counts, pre vs post | identical (factory 1, device-op 20) |
| Ephemeral-doc citations in changed `.cpp` / `.hpp` | zero |
| Lines over 120 columns | zero |
| Changed-file set | the 5 in-scope files only; device-op class, types header, nanobind, top-level `.cpp`/`.hpp` untouched |
| `allow_instance_multi_binding` sites | one (`IN1_TRANSPOSED`), census-justified, config-gated, not stacked with a self-loop |
| LLK operand types at all 11 compute call sites | all plain `uint32_t` (checked in `tt_metal/hw/inc/api/compute/`) — the `DFBBindingToken` conversion covers every one, no shim needed |
| `TensorAccessor(TensorBindingToken)` constructor + deduction guide | present (`api/tensor/tensor_accessor.h:97,416,545`) |
| `noc_traits_t<DataflowBuffer>` (DFB as NoC source and destination) | present (`api/dataflow/dataflow_buffer.h:366-404`) — the reader's and writer's transfers keep their existing shape |
| `DataflowBuffer::get_tile_size()` availability on the DM path | confirmed via the shared `__has_include("chlkc_descriptors.h")` gate |
| Every `hw_config` value vs the legacy resolved config | reader/writer = the reader/writer defaults (`RISCV_1`/`NOC_0` and `RISCV_0`/`NOC_1`, both `DM_DEDICATED_NOC`) via the arch-agnostic TTNN helpers; compute = all six `ComputeConfigDescriptor` fields carried, table in `METAL2_PORT_PLAN.md` |
| Every `KernelSpec` `opt_level` vs legacy resolved | reader/writer `O2` (Metal 2.0 default = legacy DM default), compute explicit `O3` (legacy `ComputeConfigDescriptor` default) |
| DFB endpoint census, re-derived per `(DFB, config)` | matches the audit except the Config-C refinement noted above; every DFB has ≥1 PRODUCER and ≥1 CONSUMER in all three configs |
| Gen1 DM node invariants | reader and writer on distinct RISCs, distinct NOCs, agreeing `noc_mode` |
| **Kernel bodies are a pure syntax swap** | proven mechanically — see below |

**The kernel-side "no logic changed" claim is not an eyeball assertion.** For each of the three
kernels I reverse-substituted the Metal 2.0 names back to their legacy spellings (`dfb::in0` →
`cb_id_in0`, `dfb_in1_bcast_row` → `cb_in1_bcast_row_buf`, and so on) and diffed the result against
the pre-port file. In all three the *only* surviving hunks are the ones the whitelist mandates:
the `circular_buffer.h` → `dataflow_buffer.h` include swap, the added
`experimental/kernel_args.h`, the deleted CB-index CTA constants, the positional →
`get_arg(args::…)` RTA reads, the collapsed `TensorAccessorArgs` construction, the
`CircularBuffer` → `DataflowBuffer` wrapper type, and `get_tile_size(cb_id)` →
`dfb.get_tile_size()`. Every LLK call site, every NoC transfer, every loop bound, every `#ifdef`
boundary and every comment is byte-identical — including the two known-inert oddities that were
deliberately left alone. One forced reordering: the writer's `tile_bytes` initialization moves two
lines down, because it now reads the metadata off the `DataflowBuffer` object and so must follow
its construction.

---

## Rebase onto main

The port was written against `587a4f30937` and later rebased onto `8f340f92af1` (~11 days of
main). **One conflict, in `device/kernels/ssm_eltwise_mul.cpp`, three hunks — all the same
shape**, plus a silent-drift audit of everything the port depends on.

### The conflict

Upstream `52670925503` *"[LLK] Fix #22943: Eltwise binary + broadcast init cleanup — migrate all
call sites (#50745)"* renamed three LLK init calls **in this very kernel**, and the port had
changed the *operands* of the same three lines from CB-index constants to `dfb::` handles. Same
lines, orthogonal edits:

| line | upstream renamed | port changed | resolution |
|---|---|---|---|
| both `#ifdef` branches | `binary_op_init_common` → `compute_kernel_hw_startup` | operands → `dfb::in0_transposed, dfb::in1_bcast_row, dfb::out` / `dfb::in0, dfb::in1, dfb::out` | new name + `dfb::` operands |
| `#ifndef REPEAT_INTERLEAVE_IN1` | `mul_tiles_init` → `mul_init` | operands → `dfb::in0, dfb::in1` | new name + `dfb::` operands |
| bcast-row inner loop | `mul_bcast_rows_init_short` → `mul_bcast_rows_init` | operands → `dfb::in0_transposed, dfb::in1_bcast_row` | new name + `dfb::` operands |

Git widened each hunk to swallow neighbouring `reconfig_data_format_srca` /
`pack_reconfig_data_format` / `mul_tiles*` lines that upstream did **not** touch; those took the
port's side unchanged.

Three things were checked before accepting the renames rather than assuming they were cosmetic:

- **The new signatures still take plain `uint32_t`**, so `DFBBindingToken`'s implicit conversion
  still carries `dfb::name` into them with no shim. `compute_kernel_hw_startup` is a template, but
  only over a *defaulted* `SrcOrder` non-type parameter — nothing is deduced from the arguments, so
  the conversion is unaffected.
- **`mul_init(a, b)` is behavior-identical to the old `mul_tiles_init(a, b)`.** `mul_init` exposes a
  new third parameter `acc_to_dest`, defaulted to `true`, and the deprecated `mul_tiles_init` shim
  forwards with exactly `true` — and the field is documented unused on WH/BH. No silent numerics
  change. (Worth checking: a flipped default here would have been invisible.)
- **`compute_kernel_hw_startup` is reachable from this kernel's existing includes** — via
  `api/compute/common.h`, which all three of `bcast.h` / `eltwise_binary.h` / `transpose.h` pull in.
  It does *not* come through `circular_buffer.h`, so the port's swap of that include for
  `dataflow_buffer.h` does not break it.

Taking the old names was not an option regardless: they are now `[[deprecated]]` shims (removal
after September 15th, 2026) and some builds compile with `-Werror` on deprecations.

### Verification of the resolution

The same mechanical proof used for the original port: reverse-substitute the Metal 2.0 names in the
resolved file back to their legacy spellings and diff against **main's** version of the file. The
three LLK init lines then show **zero** difference — upstream's renames are fully absorbed — and the
only remaining hunks are the port's own mechanical swaps (include, deleted CB-index CTAs, named arg
reads, `DataflowBuffer` type). No conflict marker survives anywhere in the file.

### Silent-drift audit (no textual conflict, but could still break)

Eleven days of main moved several things the port sits on. Each was diffed across
`587a4f30937..8f340f92af1` and checked against what the port actually uses:

| Moved | Impact on this port |
|---|---|
| **Tensor headers left `experimental/`** — `tt-metalium/experimental/tensor/{mesh_tensor,spec/tensor_spec,tensor_types}.hpp` → `tt-metalium/tensor/…` | **None.** The factory includes none of these directly; it reaches `MeshTensor`, `TensorSpec` and `datatype_to_dataformat_converter` transitively (`ttnn/tensor/tensor.hpp` → `tt-metalium/tensor/mesh_tensor.hpp` → `tensor_types.hpp`). Chain re-verified; the op directory contains zero `experimental/tensor` include paths. |
| `ComputeGen1Config` | **None** — struct unchanged; the file only gained additive free-function accessors (`fpu_math_fidelity(cfg)`, `unpack_modes(cfg)`, …). The port's designated initializer is unaffected, and its field order still matches the declaration order. |
| `KernelAdvancedOptions` gained a deprecated `compile_time_varargs` | **None** — appended at the end of the struct; the port sets no advanced kernel options. |
| `KernelSpec` | **None** — the only change is a comment about `Scratchpad<T>`. Every field the port uses, and their declaration order, are unchanged. |
| `DFBAdvancedOptions::allow_instance_multi_binding` | **Unchanged** — the one advanced option the port sets is intact. |
| Kernel-side `DataflowBuffer`: `scoped_lock()` replaced by `scoped_write_lock()` / `scoped_read_lock()` | **None** — the kernels never used `scoped_lock`. `DFBBindingToken`, the token constructor, the FIFO methods, `get_read_ptr`, `get_tile_size()` and `noc_traits_t<DataflowBuffer>` are all untouched. |
| `metal_v2_artifacts.hpp`, `program_run_args.hpp`, `tensor_parameter.hpp` | **None** — include-path updates only for the tensor move above; `ProgramArtifacts`, `AddRuntimeArgsForNode`, `TensorArgument` and `TensorParameter` are unchanged. |
| TTNN DM/compute config helpers | **Unchanged** in this range. |

No other file in the op directory was touched by upstream in this range — `52670925503` changed
only the compute kernel's three init lines.

**Still unbuilt.** The drift audit is a read of the headers, not a compile. Everything under
*Verification performed* above was re-run after the resolution and still passes.

## Tests to run (N150)

The confirmed set, found by sweeping every test tree for the op name and filtering to this op.
There are **no C++ gtests** and **no sweeps** for it (`tests/sweep_framework/Allops.txt` lists the
name but carries no sweep module), so pytest is the whole story.

**Primary — the op's own unit test.** Covers all three kernel-source configurations and asserts
the three-entry program cache:

```bash
pytest tests/ttnn/nightly/unit_tests/operations/ssm/test_ssm_repeat_and_interleave_eltwise_mul.py -v
```

Cheap first pass — the program-cache test alone drives Configs A, B and C twice each (cache-cold
then cache-hot), so it is the fastest way to catch a spec-validator or tensor-binding error:

```bash
pytest tests/ttnn/nightly/unit_tests/operations/ssm/test_ssm_repeat_and_interleave_eltwise_mul.py::test_ssm_eltwise_mul_with_program_cache -v
```

Per-configuration slices, if something fails and needs isolating. The configuration axis is the
`(in0_W, in1_W)` parametrization, so `-k` on the width pair selects it (`32*5120` renders as
`163840` in the test id). The Config-C expression carries an exclusion because a batch-32 Config-B
id contains the `32-163840` substring too:

```bash
T=tests/ttnn/nightly/unit_tests/operations/ssm/test_ssm_repeat_and_interleave_eltwise_mul.py
# Config A — REPEAT_IN0 + REPEAT_INTERLEAVE_IN1   (in0_W=32,     in1_W=5120)
pytest $T -v -k "32-5120"
# Config B — REPEAT_INTERLEAVE_IN1 only           (in0_W=163840, in1_W=5120)
pytest $T -v -k "163840-5120"
# Config C — REPEAT_IN0 only                      (in0_W=32,     in1_W=163840)
pytest $T -v -k "32-163840 and not 163840-5120"
```

(If the ids don't line up as expected, `pytest $T --collect-only -q` shows them; the whole-file run
above covers all three regardless.)

**Downstream consumer — mamba.** The only model that calls the op
(`models/demos/wormhole/mamba/tt/mamba_ssm.py`, four call sites, exercising Configs A and C):

```bash
pytest models/demos/wormhole/mamba/tests/test_mamba_ssm.py -v
pytest models/demos/wormhole/mamba/tests/test_mamba_block.py -v
pytest models/demos/wormhole/mamba/tests/test_residual_block.py -v
pytest models/demos/wormhole/mamba/tests/test_mamba_model.py -v
```

**Perf, for the pointer-patching question** (`Pointer patching perf issue? = suspect perf
regression`) — run on `origin/main` and on this branch and compare:

```bash
pytest models/demos/wormhole/mamba/tests/test_mamba_perf.py -v
```

**Sibling ssm ops — a no-touch sanity check.** They share no kernel and no factory with this op,
so they should be unaffected; running them confirms the port did not disturb the family's build:

```bash
pytest tests/ttnn/nightly/unit_tests/operations/ssm/ -v
```

All of these passed before the port and must pass after — the port asserts **no** behavior change.
If a previously-passing case now fails, the likely causes in order are: a spec-validator rejection
at `MakeProgramFromSpec` (read the message — DFB endpoint counts and the `IN1_TRANSPOSED`
multi-binding flag are the interesting spots), a `TensorParameter` / `TensorArgument` mismatch, or
an RTA name/value mis-wiring in the per-core loop.

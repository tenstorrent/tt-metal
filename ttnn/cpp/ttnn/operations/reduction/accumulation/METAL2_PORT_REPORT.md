# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/reduction/accumulation`

## Outcome

**`PORTED`** — both factories in the directory converted, nothing left for a later pass:

- `AccumulationDeviceOperation::AccumulationProgramFactory` (backs `cumsum` / `cumprod`)
- `EmaDeviceOperation::EmaProgramFactory` (backs `ema`)

All six kernel sources and the shared kernel header converted with them. Verification: clean build,
and the confirmed test set is green — 156 passed / 25 skipped across
`test_cumsum.py`, `test_cumprod.py`, `test_ema.py`, `test_reduction_examples.py`, plus 24 passed in
`test_reduction_ops.py -k test_accumulation`. The skips are pre-existing shape/dim exclusions inside
`test_cumprod.py`, not port-induced.

## Provenance

- **Recipe docs (this port):** `dc266b472bd 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`
- **Audit docs (inherited):** `dc266b472bd 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`

## TTNN ProgramFactory

### Concept realized

`MetalV2FactoryConcept` on both factories, as the audit decided — each now exposes a single
`static ttnn::device_operation::ProgramArtifacts create_program_artifacts(attributes, tensor_args,
tensor_return_value)` and returns a `ProgramSpec` + `ProgramRunArgs` pair with no `op_owned_tensors`.
No deviation from the audit's choice, so nothing needed re-deciding with the invoker.

One naming note for readers: on this branch the concept is spelled `ProgramSpecFactoryConcept` in
code ([`operation_concepts.hpp:119`](../../../../../api/ttnn/operation_concepts.hpp#L119)), not
`MetalV2FactoryConcept`. See Friction gap 3.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — neither device operation defined one; both were
  already on the default reflection-based hash.
- **Pybind entry points removed:** none. The three nanobind files
  (`cumsum/cumsum_nanobind.cpp`, `cumprod/cumprod_nanobind.cpp`, `ema/ema_nanobind.cpp`) bind only
  the public op functions; no factory entry point was exposed. A repo-wide grep confirmed no external
  caller of either `create_descriptor`, so its disappearance is not a user-visible surface change.
- **Factory parameter dropped for a pybind hook:** none — both legacy signatures were already the
  standard three-argument shape.

The only device-op-class edits were the two mechanical ones the concept switch forces, on each
header: the factory method declaration
([accumulation_device_operation.hpp:41-44](device/accumulation_device_operation.hpp#L41-L44),
[ema_device_operation.hpp:24-27](ema/device/ema_device_operation.hpp#L24-L27)) and the include swap
from `<tt-metalium/program_descriptors.hpp>` to `"ttnn/metal_v2_artifacts.hpp"`. Plus the deletion of
`AccumulationProgramFactory::AccumulationCB` — the magic-CB-index enum the `DFBBinding`s replace —
and with it the now-unused `hostdevcommon/kernel_structs.h` and `<type_traits>` includes. Nothing
else in either class was touched: no `validate_on_program_cache_miss`, `compute_output_specs`, or
`create_output_tensors` change.

### Open items

- **Relaxation candidates (not applied).** Both dataflow kernel pairs are written shape-agnostically
  — they address tiles purely by arithmetic on runtime args, baking in no dimension — so a
  `dynamic_tensor_shape` relaxation would plausibly widen cache equivalence. Deliberately **not**
  applied: strict matching is the default, and a grep for `ArgConfig::Runtime*` across the op's
  kernels returns zero hits, so there is no legacy relaxation to mirror. This is a judgment call for
  the op owner, not the porter.
- **RTA→CRTA candidates.** Several of the accumulation reader/writer runtime args carry the *same*
  value on every node — `tiles_per_row`, `input_tile_offset`, and `flip` are computed once per
  dispatch and broadcast identically. As common runtime args they would dispatch more cheaply. Not
  converted here (RTA→CRTA changes dispatch semantics, so it is a separate cleanup, per the recipe's
  `runtime_arg_values` guidance).
- **Node-first RTA loops retained.** Both factories keep their legacy per-core loops and bridge to
  the name-first table via `AddRuntimeArgsForNode`, rather than being restructured name-first. That
  restructure is a worthwhile separate cleanup; doing it inside the port would have added
  transposition-error risk to an already-large host rewrite.

## Handoff points

### 1. `KernelSpec::CompilerOptions::opt_level` silently downgrades every ported compute kernel from O3 to O2

**Owner: Metal 2.0 host API.** Tagged "silent perf regression in the spec→program lowering."

Legacy `KernelDescriptor::opt_level` is a `std::optional<KernelBuildOptLevel>` defaulting to
`nullopt`, and the descriptor→program path resolves that **per kernel kind**: data-movement kernels
get `O2` ([program.cpp:439](../../../../../../tt_metal/impl/program/program.cpp#L439)), compute
kernels get `O3` ([program.cpp:455](../../../../../../tt_metal/impl/program/program.cpp#L455)).

Metal 2.0's `CompilerOptions::opt_level` is instead a plain field with a hard `O2` default for both
kinds ([kernel_spec.hpp:116](../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/kernel_spec.hpp#L116)),
and the lowering assigns it straight through to `ComputeConfig::opt_level`
([program_spec.cpp:2721](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L2721)).

**Consequence:** any op ported to Metal 2.0 whose compute `KernelSpec` does not set `opt_level`
explicitly loses `-O3` on its compute kernels. There is no build error, no validator complaint, and
no test signal — exactly the silent perf shift the recipe's hardware-configuration section warns
about, except this field lives in `compiler_options`, not `hw_config`, so nothing directs a porter to
diff it. This port sets `KernelBuildOptLevel::O3` explicitly on all three compute `KernelSpec`s
([accumulation_program_factory.cpp:240-242](device/accumulation_program_factory.cpp#L240-L242),
[ema_program_factory.cpp:176-178](ema/device/ema_program_factory.cpp#L176-L178)) and verified the
result from the built kernel ELFs' DWARF producer strings: compute at `-O3`, data movement at `-O2`,
matching legacy on every kernel.

**Suggested fix:** make the field `std::optional<OptLevel>` so the lowering can apply the same
per-kind default legacy did, or default it by kernel kind. Either removes the trap for every
subsequent port. Until then, the recipe should list `compiler_options.opt_level` alongside the
`hw_config` fields in its before/after diff instructions and in the anti-pattern checklist.

### 2. Boundary-rule assumption violations

**None.** No call site in any of the six kernels needed a `sem::` or `tensor::` handle passed to an
out-of-op callee. Every `#include` in every kernel resolves either inside the op directory or under
`tt_metal/hw/inc/api/`, and the one shared in-directory helper (`get_tile_id`) takes plain scalars.
The `dfb::name → uint32_t` implicit conversion covered every LLK call site (`unary_op_init_common`,
`reconfig_data_format`, `pack_reconfig_data_format`, `copy_tile_to_dst_init_short`, `copy_tile`,
`pack_tile`, `compute_kernel_hw_startup`, `transpose_init`, `transpose_tile`) with no wrapping.

### 3. Kernel-lib gaps

**None.** The op includes nothing from `ttnn/cpp/ttnn/kernel_lib/`.

### 4. Framework gaps that bit during the port

**None.** No audit-flagged UNSUPPORTED feature was reached. No `GlobalCircularBuffer`, no
`get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`, no cursor surgery, no borrowed-memory or
aliased DFB, no conditional binding, no Case 2 tensor binding, no op-owned tensor.

### 5. Removed pybind surface

**None** — see *Device-op-class edits* above.

## Successes

- **[Two-toucher DFB → assign 1P+1C](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split),
  specifically its *Constraint — distinguish from the disjoint-node work-split* clause.** The
  accumulation factory has **two** compute `KernelSpec`s that both bind `ACCUM_SRC` as CONSUMER and
  `ACCUM_DST` as PRODUCER
  ([accumulation_program_factory.cpp:243-264](device/accumulation_program_factory.cpp#L243-L264)),
  which reads at a glance like a two-producer / two-consumer DFB needing the multi-binding flag. The
  entry's insistence on counting **per node** rather than per spec, plus the disjoint-node carve-out,
  made it immediately clear each node hosts exactly one compute instance and each DFB is an ordinary
  1:1. The `dataflow_buffer_spec.hpp` INVARIANT block confirmed the same thing at the field. I did
  not reach for `allow_instance_multi_binding` anywhere, and the spec validator accepted the shape
  first try.

- **[Anti-pattern: Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta),
  and the brief's matching *Watch for* item.** Two legacy compute descriptors that differ **only** in
  `core_ranges` are a standing invitation to collapse them into one `KernelSpec` over the union — the
  audit itself records the split as pointless (Misc anomaly 5). Both the pattern entry and the brief
  say to keep the legacy shape anyway, and that is what the port does
  ([accumulation_program_factory.cpp:298-333](device/accumulation_program_factory.cpp#L298-L333)).
  This is the clearest case in the port of the docs stopping a "harmless" scope creep.

- **"Go to the headers first; they are ground truth" — it changed the outcome, not just the wording.**
  The recipe's `unpack_modes` summary says a `UnpackToDest` on a `≤16-bit` format is "rejected on
  Gen1 as a pure perf loss." Legacy accumulation sets `UnpackToDestFp32` on its `SRC` CB whenever the
  input format is not `Float16_b` — which includes `bfloat8_b`, a ≤16-bit format
  ([accumulation_program_factory.cpp:126-129, pre-port](device/accumulation_program_factory.cpp)). Read
  from the recipe alone, that looks like a faithful port being rejected by the validator, i.e.
  grounds to stop. Reading the validator itself
  ([program_spec.cpp:1011-1013](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L1011-L1013))
  shows the `≤16-bit` rule is only reached when `enable_32_bit_dest` is **false**, and this kernel sets
  it true — so the check short-circuits and the faithful port is legal. The header/source was
  definitive where the paraphrase was not.

- **Scope discipline on the dead `start_id` argument.** Reader/writer RTA slot 4 is a dead *value*
  (audit anomaly 1) and is genuinely tempting to drop while rewriting the whole arg list. The brief's
  explicit "**Port it as-is**" carried it through unchanged as a named RTA
  ([accumulation_reader.cpp:19](device/kernels/dataflow/accumulation_reader.cpp#L19)), keeping the diff
  a pure syntax swap.

## Friction

### Gaps

1. **The brief's `constexpr get_tile_size()` claim is wrong, and it is the one claim that forced a
   kernel-line change beyond a rename.** The brief states that `DataflowBuffer::get_tile_size()`
   being `constexpr` "matters for the two EMA sites where the result is bound to a
   `constexpr uint32_t`." It does not help: `get_tile_size()` is a `constexpr` *member*, but
   `DataflowBuffer`'s constructor is declared out-of-line and is not `constexpr`
   ([dataflow_buffer.h:72-75](../../../../../../tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L72-L75)),
   so a call on a non-`constexpr` object is not a constant expression and
   `constexpr uint32_t x = dfb.get_tile_size();` does not compile. Both EMA sites moved to
   `const uint32_t` ([ema_reader.cpp:30](ema/kernels/dataflow/ema_reader.cpp#L30),
   [ema_writer.cpp:30](ema/kernels/dataflow/ema_writer.cpp#L30)). Behaviour-neutral — the value is a
   NoC transfer byte count, and the getter still folds at `-O2` — but a porter who trusts the brief
   will hit a compile error and may suspect their own binding is wrong.
   **Right answer:** whitelist rule 7 should note that the DFB metadata getters cannot initialize a
   `constexpr`, so a legacy `constexpr` CB-metadata local becomes `const`. Cheap to state, and it
   turns a confusing error into an expected one.

2. **`compiler_options.opt_level` is absent from the recipe's diff-the-config discipline.** The
   Hardware configuration section is thorough about `hw_config` and its anti-pattern checklist item
   says to diff "the compute knobs," but `opt_level` lives on `compiler_options`, so neither the
   section nor the checklist reaches it — while the Metal 2.0 default differs from legacy for compute
   kernels. Full detail and the suggested framework fix are in Handoff point 1. This is the port's
   most consequential finding: it is silent, it affects perf not correctness, and it will hit every
   op ported before the default is fixed or the recipe is amended.

3. **Concept name drift, docs vs code.** The recipe, the TTNN integration doc, and the brief all name
   the target concept `MetalV2FactoryConcept`. The code calls it `ProgramSpecFactoryConcept`, with a
   `CustomProgramSpecFactoryConcept` sibling for factories that also define a spec-runtime-args
   override ([operation_concepts.hpp:119-133](../../../../../api/ttnn/operation_concepts.hpp#L119-L133));
   `metal_v2_artifacts.hpp`'s own comment refers to the code spelling too. Harmless for the port
   itself — the factory satisfies the concept structurally, by declaring `create_program_artifacts` —
   but the doc name is not greppable, which costs a porter a few minutes confirming they are looking
   at the right thing, and it leaves them unsure whether `CustomProgramSpecFactoryConcept` is
   something they should be considering. The docs should use the code spelling and say one line about
   the `Custom` variant.

4. **The no-regression baseline instruction has a hole: kernels are JIT-compiled at *test* time.**
   The recipe's verification step treats "tests passing pre-conversion" as the baseline but never
   says when to capture it. I started the pre-port test run and then began editing, reasoning that
   the tests were already linked against the built `.so` — true for host code, false for kernels,
   which are compiled from source on the first dispatch. The first test in the run picked up my
   half-edited kernel and the baseline was lost. Recoverable (the port is green, and a failure could
   have been bisected with `git stash`), but a wasted build-and-test cycle.
   **Right answer:** the recipe should say explicitly that the pre-port test run must *complete*
   before the first kernel edit, because kernel sources are read at dispatch time, not at build time.

5. **`AddRuntimeArgsForNode` takes a `std::initializer_list`, so a shared per-node arg list cannot be
   factored out cleanly.** The accumulation reader and writer take an identical 7-argument list per
   node, but an `initializer_list` parameter means the only options are to write the braced list
   twice or hoist it into a named `initializer_list` local with lifetime care. The port writes it
   twice ([accumulation_program_factory.cpp:290-311](device/accumulation_program_factory.cpp#L290-L311)),
   which matches how the legacy factory also duplicated the list, but a `std::span` overload (or
   accepting any range of `pair<string, uint32_t>`) would let ports share it. Minor.

### Confusion

1. **"The port adds exactly two headers" does not hold for a kernel already on `DataflowBuffer`.**
   The kernel-side whitelist states the port adds `experimental/kernel_args.h` and
   `api/dataflow/dataflow_buffer.h`, with "the now-unused `CircularBuffer` include drops with rule
   1's sweep." All six kernels here were *already* on `DataflowBuffer` and already included that
   header (the audit's own Recipe note 1 flags this shape), so the port added exactly **one** header
   and dropped none. Not a problem — I convinced myself quickly — but the phrasing reads as a
   checklist a porter might try to satisfy literally, and a kernel that is already ahead of the
   Device 2.0 `CircularBuffer` wrapper is going to become the common case, not the exception. Worth
   one clause: "…or one, if the kernel is already on `DataflowBuffer`."

2. **`TT_KERNEL` exists and the recipe never mentions it.** `experimental/kernel_args.h` documents a
   `TT_KERNEL` marker on the entry point, "from which the JIT generates `kernel_main()`"
   ([kernel_args.h:44-47](../../../../../../tt_metal/hw/inc/experimental/kernel_args.h#L44-L47)), and
   there is a whole signature parser behind it
   ([kernel_signature_parser.hpp](../../../../../../tt_metal/jit_build/kernel_signature_parser.hpp)).
   The recipe's and migration guide's examples all use a plain `void kernel_main()` with
   `get_arg(args::name)`, which is what this port does. Reading the parser confirms the marker is
   optional and a hand-written `kernel_main()` is "fully backward compatible", so the choice is safe
   either way — but a porter who opens the header they were told to include finds a mechanism the
   recipe is silent about, and cannot tell whether the plain form is the intended endpoint or legacy
   tolerance. One sentence in whitelist rule 4 would settle it.

## Open items for downstream

- **Shared kernel touches: none.** No `_metal2` fork was reused, created, or needed; no kernel source
  was modified outside this op directory. The one intra-directory sharing point,
  [device/kernels/accumulation_common.hpp](device/kernels/accumulation_common.hpp) (included by all
  six kernels across **both** device operations), was edited in place — legitimately, because both of
  its consuming factories convert in this same change, so no consumer is left behind on the legacy
  API. There is no sunset list and no coordination signal for a future porter. Repo-wide greps for
  the six kernel filenames find no consumer outside this directory.
- **Audit Misc anomalies 1-8 remain open for the ops team.** The port acted on none of them, as
  instructed. The two most worth picking up: anomaly 1 (the dead `start_id` value threaded through
  reader and writer on every core) and anomaly 5 (the redundant core-group split of the compute
  kernel — now visible as two near-identical `KernelSpec`s plus a second `WorkUnitSpec`, so the
  redundancy costs slightly more boilerplate under Metal 2.0 than it did under
  `ProgramDescriptor`).
- **EMA `c_2` carries two names, and the port kept both.** The host calls it `prev`, the kernel calls
  it `trp` (audit anomaly 8: it is a transpose round-trip scratchpad, not a "previous output" store).
  Rather than pick a side inside a syntax-only port, the `DFBSpecName` keeps the host's word and the
  `accessor_name` keeps the kernel's, so the binding reads
  `{.dfb_spec_name = EMA_PREV, .accessor_name = "trp"}`
  ([ema_program_factory.cpp:191-200](ema/device/ema_program_factory.cpp#L191-L200)). A follow-up
  should settle on `trp`, which is what the code actually does.
- **`packer_l1_acc` is dropped from the EMA compute path, with no behaviour change.** Legacy
  destructured it out of `get_compute_kernel_config_args` and never used it; the port replaces that
  call with `to_compute_hardware_config`, which documents `packer_l1_acc` as an op-side concern it
  does not translate. Same resulting configuration either way — noting it so a reviewer diffing the
  two configs does not read the disappearance as a lost setting.
- **Test coverage note.** This op has **no** C++ gtest and no sweep; its entire coverage is the five
  pytest files confirmed with the invoker. For an op whose numerics vary across five dtypes and an
  integer/float kernel split, a gtest would give the next porter a much faster inner loop than a
  85-second pytest run. Not acted on.
- **Doc-evolution suggestion (beyond the Gap entries).** Three of this port's five gaps
  (`opt_level`, `constexpr get_tile_size`, the JIT-timing of kernel compilation) are all the same
  shape: a fact about the *build/lowering* path that a porter cannot discover from the spec-shaped
  parts of the recipe. A short "what the lowering does to your spec" section — where the values
  actually land, what defaults differ from legacy, and when each artifact is compiled — would catch
  this class of finding earlier than the per-field prose does.

# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/moreh/moreh_norm`

> Opened at the start of the port; friction captured as it happened, polished at the end.

## Outcome

**`PORTED`** — all three factories of `MorehNormOperation` (`ProgramFactoryWOther`,
`ProgramFactoryHOther`, `ProgramFactoryNCOther`) converted to `ProgramSpecFactoryConcept`, together
with all nine live kernels under `device/ord_other/`. Nothing left for a later pass.

- `./build_metal.sh --build-tests`: **SUCCESS**, no warnings mentioning the op.
- Confirmed test set (`tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_norm.py`,
  forward tests only, per the invoker's sign-off): **no regression — identical to the pre-port
  baseline.**

  | | pre-port (baseline) | post-port |
  |---|---|---|
  | | `362 passed, 1194 skipped, 776 deselected` | `362 passed, 1194 skipped, 776 deselected` |

  A pre-port baseline was captured on the unmodified tree *before* any edit, so the comparison is a
  real before/after rather than a bare green run. Beyond the counts, the two runs' executed test-ID
  sets were diffed and are byte-identical (180 unique ids, zero difference) — so the skip set did not
  shift and no test silently stopped running. `test_moreh_norm_callback` is in the passing set, which
  exercises the program-cache **hit** path (`UpdateTensorArgs` refreshing the two `TensorBinding`s
  without re-running the factory).

  One coverage caveat the green run does *not* cover, stated up front because it would otherwise be
  invisible: the FP32 path of these factories is not exercised by this suite at all, so the
  `unpack_modes` table is unvalidated by test. Details and a cheap fix in *Open items for downstream* 5.

## Provenance

- **Recipe docs (this port):** `04c24de2a33 2026-07-29 docs(metal_2.0): follow main's MetalV2FactoryConcept -> ProgramSpecFactoryConcept rename`
- **Audit docs (inherited):** `9bba65ffd6b 2026-07-29 docs(metal_2.0): follow main's MetalV2FactoryConcept -> ProgramSpecFactoryConcept rename`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, as the audit chose, for all three factories. No re-decision, nothing
surfaced back to the invoker on the concept. Each factory returns a single `ProgramArtifacts`
(`spec` + `run_params`); `op_owned_tensors` left defaulted-empty — the legacy factories allocated no
device tensor beyond the op's io.

The variant is homogeneous: all three converted in the same change, so `program_factory_t`
(`device/moreh_norm_device_operation.hpp:56`) carries no mixed-concept transition state.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — the op never defined one, so it was already on
  the default reflection-based hash. (The `UpdateTensorArgs` `TensorSpec`-legality signature of a
  surviving custom hash — failures on the *second* dispatch, cache hot — would have shown up in
  `test_moreh_norm_callback`, which exercises the cache-hit path; it passes.)
- **Pybind entry points removed:** none. `moreh_norm_nanobind.cpp:38-49` binds only the user-facing
  `ttnn::moreh_norm` free function via `ttnn::bind_function<"moreh_norm">` — no `create_descriptor`
  exposure, no factory internals, no pybind-hook-only factory parameter. The pybind layer is
  untouched, so there is **no user-visible API surface change** from this port.
- **The one forced header edit** (`device/moreh_norm_device_operation.hpp`): the three nested
  factories' `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` declarations became
  `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`, and
  `#include <tt-metalium/program_descriptors.hpp>` became `#include "ttnn/metal_v2_artifacts.hpp"`.
  Mandatory, not discretionary: `ProgramSpecFactoryConcept` is
  `requires { &T::create_program_artifacts; } && … && !ProgramDescriptorFactoryConcept<T>`
  (`ttnn/api/ttnn/operation_concepts.hpp:119-121`), so leaving the old declaration in place would make
  each factory satisfy two concepts and trip the `AllFactoriesValid` `static_assert`. See Friction
  gap 3 — this edit is not in `ttnn_factory.md`'s list of sanctioned device-op-class exceptions.
- **Nothing else in the device-op class changed.** `validate_inputs`, `select_program_factory`,
  `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`,
  `operation_attributes_t`, `tensor_args_t` and the dead
  `get_floored_p_and_decimal_and_p_is_negative` are all byte-identical.

### Open items

- **`TensorParameter` relaxation: none, and none is warranted.** Strict `TensorSpec` matching kept on
  both parameters in all three factories. Two independent confirmations: the op has no custom hash, so
  no relaxation can be active; and the pre-migration `grep -r 'ArgConfig::Runtime'` over the nine live
  kernels returns **zero hits**, so no `dynamic_tensor_shape` / `match_padded_shape_only` opt-in is
  implied. Also nothing in these kernels would obviously *tolerate* one — both accessors are used for
  ordinary per-page addressing driven by RTAs, so a shape relaxation would have no cache-equivalence
  payoff worth the risk. No relaxation candidate to hand downstream.
- **Concept fit was clean.** Single-program, no op-owned tensors, no op-owned `GlobalSemaphore`s, no
  per-coord variation. No capability the op wants that this concept lacks, with one ergonomic
  exception noted under *Open items for downstream* (a `KernelSpec` cannot join two `WorkUnitSpec`s
  with per-WU runtime args, which is why this op carries two byte-identical compute specs).

## Handoff points

**None.** Recorded explicitly rather than omitted, because each category was checked:

- **No capitulation.** Every construct in all three factories mapped onto a documented Metal 2.0
  primitive. Nothing needed a workaround, and no rule had to be stretched.
- **Boundary-rule assumption held.** No call site outside the op directory requires a `sem::` or
  `tensor::` handle. The four donor headers the kernels reach into all cross cleanly by design:
  `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (`fill_cb_with_value`, `generate_mask_w`,
  `generate_mask_h`) and `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` (`*_with_dt` helpers) take
  `DataflowBuffer` **by value**, so the kernel's local object passes straight in;
  `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` takes `uint32_t` CB-id **NTTPs**, which
  `DFBAccessor`'s `constexpr operator uint32_t()` satisfies in template-argument position. Zero donor
  edits.
- **No kernel-lib gap, no framework gap.** No audit-time UNSUPPORTED entry bit during the port; every
  Appendix A feature was `N/A` at audit time and stayed that way.
- **No removed pybind surface** (see *TTNN ProgramFactory* above).
- **No shared-kernel fork.** All nine live kernels are op-exclusive; see *Open items for downstream*.

## Successes

**The audit brief's dead-kernel-tree banner fired exactly as intended, and it mattered.**
Two of the nine live kernels — `moreh_norm_w_kernel.cpp` and `moreh_norm_h_kernel.cpp` — share a
basename with an unreferenced sibling under `device/moreh_norm_{w,h}/kernels/`, and because the
Device 2.0 sweeps migrated the dead copies as though live, they read as current code. I edited by full
path throughout and closed with a grep over the op tree: the only surviving `get_arg_val<uint32_t>` and
`tt::CBIndex` hits are in the three dead files, which is the confirmation that no edit landed in the
wrong tree. Without the banner, an editor fuzzy-open on `moreh_norm_w_kernel` is a coin flip, and the
resulting diff compiles, installs, and changes nothing. (Brief: *"there are two kernel trees, and one
is dead"*; the audit's Recipe note 1 proposes promoting this from "mention it" to "call it out
prominently" — this port is evidence for that.)

**"Re-derive the endpoint disposition, don't transcribe" earned its cost.**
Recipe [§Read this first](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#read-this-first)
plus the catalog's endpoint-assignment procedure. I re-ran the kernel-touch census on all 19
`(factory, CB)` pairs; it agreed with the brief in every case — 11 plain 1:1, 8 self-loop, no dead CB,
no flag. Agreement is the useful outcome, not a wasted step: it converted "the brief says self-loop"
into "I know `cal` does real FIFO work in *both* directions" — post-port
`moreh_norm_w_kernel.cpp:87,103` (`reserve_back`), `:98,125` (`push_back`), `:102` (`wait_front`),
`:124` (`pop_front`), plus the `reduce<>` input at `:129`. That is what distinguishes a genuine
accumulator self-loop from a sync-free or single-ended CB, and therefore what tells you the kernel
body needs no change at all.

**The disjoint-node clause in [Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) handed me the `WorkUnitSpec` shape.**
The intuitive translation of the legacy placement — reader/writer on `all_cores`, compute on its group
— is a single work unit over `all_cores`, and it is wrong: DFB placement is *derived* from work-unit
membership, so the `input` and `output` DFBs would have a producer but no consumer on group-2 nodes and
the per-node census (`tt_metal/impl/metal2_host_api/program_spec.cpp:1355-1390`) would reject the spec.
The catalog's `wu_g1` / `wu_g2` example, with reader and writer listed in **both** work units, is the
correct shape and I took it directly (`..._w_other.cpp:345-359`). This is the entry that saved the
most debugging.

**"`Table`s are maps, not vectors" prevented a compile error I would otherwise have written.**
The legacy factories build a `std::map<std::string,std::string>` of compute defines and convert. The
recipe's explicit "no `push_back`, no iterator-pair constructor" note meant I skipped the intermediate
map entirely and built `KernelSpec::CompilerOptions::Defines` with `operator[]` under the same
conditional structure (`..._w_other.cpp:228-238`) — smaller than the legacy code and correct first try.

## Friction

### Gaps

**1. The `unpack_modes` required-entry rule has two independent triggers in this op; the recipe reads
as though it has one.** The recipe's *"A newly-required explicit entry"* bullet says the rule fires
"when a compute kernel **consumes a Float32 DFB** with `enable_32_bit_dest = true`". For an op like
this one, whose intermediates become `Float32` precisely when `fp32_dest_acc_en` is set, the natural
reading is "so add entries for the intermediates" — and that is a **latent config-dependent bug**,
because `cb_data_format` is *also* `Float32` whenever the input tensor's dtype is float32, which makes
`input`, `one` and `mask_*` require entries too, on a completely unrelated condition. A porter who
enumerated only the intermediates ships a spec that validates for bfloat16 input and `TT_FATAL`s for
float32 input — and, in this op, a bf16-only test suite would not catch it (see *Open items* 5).
**Suggested fix:** state the rule as *derive the entry set from the census of DFBs the compute kernel
consumes, not from which DFBs you expect to be FP32* — the DFB-consumption set is knowable statically,
the format set is not. What I did: an `UnpackToSrc` entry for **every consumed DFB**, unconditionally
(`..._w_other.cpp:245-260`). That is exactly equivalent to the legacy all-`Default` vector
(`BuildUnpackToDestModeVector`, `program_spec.cpp:2673-2693`, lowers `UnpackToSrc` back to
`UnpackToDestMode::Default`), always accepted (`program_spec.cpp:999-1001`), and immune to both
triggers.

**2. Multiple same-source `KernelSpec`s each self-looping the same DFB is legal, but only the
validator source says so.** This op needs it 8 times over (three compute-private accumulators × two
work-split compute specs). [Pattern: Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)
describes the self-loop purely as a one-kernel construct, and the sibling entry's "hard gate" is
phrased in terms of *distinct kernels touching the CB on a node* — which is satisfied here, but says
nothing about two `KernelSpec`s of one source each self-looping on disjoint node sets. Because
self-loop is explicitly *mutually exclusive* with multi-binding, and multi-binding is described as
unsafe and Gen2-forbidden, the docs made this look like it might be the forbidden stacking. The answer
is an unambiguous yes and it is written down — just in the framework, at
`tt_metal/impl/metal2_host_api/program_spec.cpp:1425-1444`: *"the producer set must equal the consumer
set as sets of `KernelSpec*` … This permits the natural pattern of multiple same-source KernelSpecs
each self-looping the DFB on their disjoint node ranges"*. I found it by reading the validator, which
cost ~20 minutes. **Suggested fix:** one sentence in the self-loop pattern entry. Work-split ×
compute-private accumulator is a very common combination, so the next porter will hit this too.

**3. The forced `create_descriptor` → `create_program_artifacts` declaration change is not on
`ttnn_factory.md`'s list of sanctioned device-op-class edits.** That document enumerates exactly three
exceptions to "the device-operation class is off-limits" (custom hash, pybind lines, pybind-hook-only
parameter). For an op whose factories are **nested structs inside the device-op class** — as all three
of `moreh_norm`'s are (`device/moreh_norm_device_operation.hpp:33-52`) — editing the header is
unavoidable, and `ProgramSpecFactoryConcept`'s `!ProgramDescriptorFactoryConcept<T>` conjunct makes
leaving the old declaration a hard build break, not a style choice. It is obviously in scope; it just
isn't written down, so a porter following the scope discipline strictly pauses to check. **Suggested
fix:** name it as the zeroth, always-applicable edit in *Device-operation-class edits the port forces*,
distinguishing "the factory's own declaration" from "the rest of the class."

### Confusion

**4. `experimental/kernel_args.h` advertises a `TT_KERNEL` entry-point macro the recipe never
mentions.** The recipe says to go to the headers first, and the header a porter is told to add opens
with *"TT_KERNEL: marks the named-arg entry point; the JIT generates `kernel_main()` from its
signature"* — backed by a real `kernel_signature_parser` in `tt_metal/jit_build/`. Every recipe and
migration-guide example, though, keeps plain `void kernel_main()` with `get_arg(args::name)`, and the
only in-tree users of `TT_KERNEL` are `tests/tt_metal/`. I followed the recipe. **Suggested fix:** a
line in the kernel-side whitelist saying which entry-point form a port uses today, and that
`TT_KERNEL` is not part of the port surface yet — otherwise every porter who reads the header
(as instructed) has to resolve the same ambiguity.

**5. "Extract `MeshTensor` and work with it throughout" reads as forbidding a correct thing.**
`ttnn_factory.md`'s *Extracting the tensor* says to hold the `MeshTensor` and pass
`const MeshTensor&` to helpers rather than "reaching back through `.mesh_tensor()`". But this op's
preserved `*_is_dram` RTA is computed by a moreh helper whose signature is
`is_dram(const Tensor&)` (`moreh_helper_functions.hpp:19`), so two sites per factory must keep using
the `ttnn::Tensor` (`..._w_other.cpp:390-391`). Minor, and I kept them; the "throughout" wording just
made me re-read the section to be sure I wasn't violating it. **Suggested fix:** a parenthetical that
existing host-side helpers taking `const Tensor&` are fine to keep calling — the guidance is about
what the factory *builds the spec from*, not a ban on the TTNN type.

## Open items for downstream

**1. Shared kernel touches: none.** All nine live kernels (`device/ord_other/moreh_norm_{w,h,nc}/kernels/`)
are op-exclusive and converted in place. Census run per the shared-kernel Caution: no factory of this
op binds another's kernel (each of the three owns its own reader/compute/writer copy);
`grep -rl <basename> ttnn/cpp/ttnn/operations/` for each of the nine returns only this op, the family
`CMakeLists.txt` install glob (not a consumer), and — for the two colliding basenames — the *dead*
sibling that no factory binds; and `ls` of each kernel directory shows no pre-existing `_metal2` fork.
**No fork created, none reused, no in-place edit of anyone else's kernel, no pointer comment owed, no
sunset list to track.** Nothing for the next porter to coordinate here.

**2. RTA → CRTA cleanup: 12 arguments across the three factories.** `Wt`, `Ht`, `origin_w`, `origin_h`,
`outer_stride`, `num_inner_tiles`, `num_reduced_tiles_along_dim`, `input_is_dram` and `output_is_dram`
all carry the **same value on every node** and are therefore really common runtime args. Left as RTAs
deliberately — RTA→CRTA changes dispatch semantics and the recipe routes it to a separate pass. Cheap
dispatch-efficiency win for whoever does that sweep; the only per-node values that genuinely vary are
`tile_offset` and the `num_*_per_core` work count.

**3. The two compute `KernelSpec`s per factory are byte-identical apart from `unique_id`.** Because
the per-group work count was already an RTA in legacy (both compute `KernelDescriptor`s carried
`compile_time_args = {}`), `compute_g1` and `compute_g2` differ in *nothing* but their name — same
source, same defines, same `hw_config`, same RTA schema. They exist solely so each can sit in a
different `WorkUnitSpec` and thereby land on its own core group. That is correct today and not a
defect: placement is derived from work-unit membership, and one `KernelSpec` cannot be in two
`WorkUnitSpec`s with *different* co-kernels. The cost is that the op compiles the same compute binary
twice per program. If the framework ever lets one `KernelSpec` join multiple `WorkUnitSpec`s with
per-work-unit runtime args, all three factories collapse from four `KernelSpec`s to three. Worth
knowing before someone reads the duplication as a porting mistake.

**4. Ops-team items the audit logged — the port preserved every one verbatim.** (Audit *Misc anomalies*.)
   - *Anomaly 1 — three unreferenced kernel files.* `device/moreh_norm_{h,w,other}/kernels/*.cpp`, live
     only in the install glob at `ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:44-46`. Untouched and
     unaudited by this port. Deleting them (files + glob lines) would remove the basename-collision
     hazard permanently — and is now *more* worth doing than before, because those three files are the
     last `get_arg_val` / `tt::CBIndex` users left in the op tree, so a future grep-based sweep of the
     op will keep tripping over them.
   - *Anomaly 2 — six dead `*_is_dram` RTAs.* Preserved: each is now a **named** arg
     (`args::input_is_dram` / `args::output_is_dram`), still computed and passed by the host, still
     read into the same never-used `const bool`. Removing them is one RTA per dataflow kernel × three
     factories, and is a functional change the ops team owns.
   - *Anomaly 3 — `get_floored_p_and_decimal_and_p_is_negative`* is still dead at
     `device/moreh_norm_device_operation.cpp:14-22`; device-op-class code, untouched.
   - *Anomaly 4 — NC's `one` tile is produced and consumed but never read.* Preserved verbatim: the
     reader still fills it (`reader_moreh_norm_nc.cpp:23-24`) and compute still waits/pops it
     (`moreh_norm_nc_kernel.cpp:28, 128`). It is a live 1P+1C DFB in the census, so it was never a
     dead-CB drop candidate. Removing it would delete one DFB, one reader fill and one compute
     wait/pop from the NC factory.
   - *Anomaly 5 — deprecated `tt::CB` enum in the compute kernels: **self-resolved**.* Those
     `tt::CB::c_in0` / `c_out0` / `c_intermed0` constants were exactly the magic CB indices the port
     replaces with `dfb::` handles, so the factory/kernel vocabulary mismatch the audit flagged is gone
     as a side effect of the port, not as a bundled cleanup. Nothing left to do.
   - *Anomaly 6 — three spellings of the CB-id counter idiom: **self-resolved**.* The `constexpr`
     vs. mutable-counter distinction the audit and brief both flagged as needing "different treatment
     within one port" turned out not to: in all nine kernels the counter and its derived ids are simply
     **deleted**, and each `DataflowBuffer` is constructed straight from its `dfb::` token. Nothing
     needed the id to survive as a value, so the constexpr-ness never came into play. Worth recording
     because the brief predicted a fork in the work that did not materialise — a heads-up whose cost
     was zero but which a porter might otherwise try to honour.

**5. Test-coverage gap: the FP32 path of these three factories is not exercised at all.** This one
matters for reading the verification result honestly. Two independent restrictions in the confirmed
test set combine:
   - `ttnn_dtype` is parametrized over `bfloat16` and `bfloat8_b` only, and `bfloat8_b` is
     `pytest.skip`ped by the harness (`test_moreh_norm.py:180-182`, *"bfloat8_b is not supported in the
     kernel"*) — which is most of the 1194 skips. So `cb_data_format` is **always bfloat16**.
   - `fp32_dest_acc_en` is varied only by `test_moreh_norm_compute_kernel_options`, which parametrizes
     `p ∈ {2.0, 2.5, -2.5}` (`test_moreh_norm.py:501`). Every one of those values is routed by the host
     wrapper through `moreh_abs_pow` + `moreh_sum` (`moreh_norm.cpp:29-60`) and **never reaches these
     three factories**. Every test that does reach them leaves `compute_kernel_options=None`, i.e.
     `fp32_dest_acc_en = false`.

   Net effect: `enable_32_bit_dest = true` and `intermed_data_format == Float32` are never hit on the
   ported factories, so the `unpack_modes` table this port added (Friction gap 1) is **never
   validated by these tests**. It is correct by construction — an entry for every consumed DFB, value
   `UnpackToSrc`, which `BuildUnpackToDestModeVector` lowers to the legacy `Default` — and I verified
   the rule against the validator source rather than against a passing test, but the reader of this
   report should not take the green run as evidence for that specific field.
   **Cheap fix, and it closes both holes at once:** give `test_moreh_norm_compute_kernel_options` a
   `p` value the device op actually reaches (`0.0`, `inf`, or `-inf`). Adding `ttnn.float32` to
   `test_moreh_norm`'s `ttnn_dtype` parametrization would additionally cover the `cb_data_format ==
   Float32` trigger. Neither is port work — flagging for whoever owns this op's coverage.

**6. Per-op carry-over for the sibling moreh reductions.** `moreh_norm`'s three factories are the same
reader-fills-`one`-tile → compute-accumulates → writer-drains shape as several other ops in this
family, and two decisions here should transfer verbatim: the `wu_g1`/`wu_g2` work-unit shape for a
`split_work_to_cores` + per-group-compute factory (Successes, entry 3), and the enumerate-every-
consumed-DFB `unpack_modes` rule (Friction gap 1). `moreh_mean` is the closest sibling and is the
contrast case the audit already documented for CB-endpoint config dependence (`#ifdef`-elided reader
access there vs. plain runtime `if` here), so a porter moving between the two should expect the mask CB
to behave differently.

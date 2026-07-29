# Metal 2.0 Port Report — `moreh_mean`

## Outcome

**`PORTED`** — all three factories (`MorehMeanHFactory`, `MorehMeanWFactory`, `MorehMeanNCFactory`)
converted to `ProgramSpecFactoryConcept`, together with all 8 kernel entry points they bind. Build
clean, tests pass (37/37 for the op, 76/76 for the whole test file), program-cache hit path verified.
Nothing left for a later pass.

## Provenance

- **Recipe docs (this port):** `5fcf2963d45 2026-07-29 docs(metal_2.0): follow main's MetalV2FactoryConcept -> ProgramSpecFactoryConcept rename`
- **Audit docs (inherited):** `5fcf2963d45 2026-07-29 docs(metal_2.0): follow main's MetalV2FactoryConcept -> ProgramSpecFactoryConcept rename`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, as the audit chose — for all three factories. Each is still a nested
`struct` on `MorehMeanOperation`; only the entry point changed
(`static tt::tt_metal::ProgramDescriptor create_descriptor(...)` →
`static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`). `program_factory_t`
is unchanged, `op_owned_tensors` is left default-empty, and no factory needed a `CustomProgramSpecFactoryConcept`
runtime-args override. No disagreement with the audit arose.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — the op never had one (default reflection-based
  hash, confirmed at audit and re-confirmed during inventory).
- **Pybind entry points removed:** none — `moreh_mean_nanobind.cpp:19-31` binds only
  `&ttnn::moreh_mean`; `create_descriptor` was never pybound, so no user-visible surface changed.
- The only edit to `moreh_mean_device_operation.hpp` outside the three factory declarations is the
  include swap `<tt-metalium/program_descriptors.hpp>` → `"ttnn/metal_v2_artifacts.hpp"` — forced by
  the entry-point change, not discretionary.

### Open items

- **Relaxation candidates: none applied, and one worth a look.** All six `TensorParameter`s are strict.
  The H/W/NC **dataflow** kernels are written tile-index-agnostically (every access is
  `{.page_id = <computed index>}` off a `TensorAccessor`), which is the shape that usually tolerates
  `match_padded_shape_only`. But the *compute* kernels bake `origin_H` / `origin_W` into a CTA to
  decide masking, so a relaxation that let `logical_shape` vary while `padded_shape` stayed fixed
  would silently reuse a program built for the wrong mask. **Do not relax without changing the mask
  gating first.** Recording it because the dataflow half looks relaxable in isolation and a future
  reader might stop there.
- **`num_reduce_input_tile` is an RTA with the same value on every node** in the NC factory (reader
  and both compute kernels). It is really a CRTA and would dispatch more cheaply as one. Not changed
  here — the recipe is explicit that RTA→CRTA alters dispatch semantics and belongs to a separate
  name-first/CRTA cleanup pass.
- The three factories still build their per-node RTA tables from the legacy node-first loop via
  `AddRuntimeArgsForNode`. A name-first rewrite is the same separate cleanup.

## Handoff points

**None.** No capitulation, no boundary-rule assumption violation, no kernel-lib gap, no framework
gap, no removed pybind surface. Specifically:

- No `sem::` / `tensor::` handle was required at an out-of-op call site (the op has no semaphores, and
  every `TensorAccessor` is consumed inside the op's own kernels).
- All five donor headers took `dfb::name` unchanged — `DataflowBuffer` by value via the implicit
  `DataflowBuffer(DFBAccessor)` converting constructor, or `uint32_t` dfb-id (NTTP **and** runtime
  position) via `DFBAccessor::operator uint32_t()`. **No donor file was edited or forked.**
- No kernel source outside the op directory was touched.

## Successes

- **[Caution: Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)
  — the *intra-op* clause fired correctly and saved a needless fork.**
  `kernels/writer_moreh_mean_unary_interleaved_start_id.cpp` is bound by **two** of this op's
  factories (H at `..._h_program_factory.cpp:168`, W at `..._w_program_factory.cpp:165`). Read
  naively — "shared writer, convert in place breaks the other consumer" — the reflex is rung 2
  (create `..._metal2.cpp`). The entry's rung 3 clause is what makes in-place correct here: the
  invoker explicitly assigned the bundled three-factory port, so **both** consumers convert in the
  same change. Being told to check the *assigned set* against the census, rather than treat any
  consumer list as a bar to in-place conversion, is what avoided a fork that would have needed
  sunsetting immediately.

- **The endpoint-assignment procedure's "re-derive, don't transcribe" instruction paid off on the
  mask CBs.** The brief called `H/c_3` and `W/c_3` self-loops "unmasked config only; plain 1:1 when
  masked". Transcribing that as a *disposition* would have missed the actionable consequence: the
  two configs need **different host bindings**, so the reader's PRODUCER binding must be conditional
  and the compute kernel needs a *second* (PRODUCER) binding only on the unmasked path
  (`..._h_program_factory.cpp:247-253`, `..._w_program_factory.cpp:244-250`). Running the census
  per config myself is what surfaced that. My count agreed with the brief on all 16 CBs — no
  disagreement to report.

- **[Pattern: Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  caught a latent build break before it happened.** Both readers declared the mask CB index at file
  scope **outside** the `#ifdef` that gated its use (`reader_moreh_mean_h.cpp:40` /
  `reader_moreh_mean_w.cpp:23`, pre-port). Left there, `dfb::mask_h` would have entered name lookup
  in the unmasked build where the host binds nothing — the exact `'mask_h' is not a member of 'dfb'`
  failure the entry's *Recognition signal* describes. The declarations moved inside the `#ifdef`
  (they dissolved into the `DataflowBuffer(dfb::mask_h)` construction).

- **[Anti-pattern: Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta),
  plus its *Constraint* paragraph.** Six compute `KernelSpec`s (two per factory) keep their per-group
  CTA. The Constraint paragraph — disjoint node sets are *not* the same-grid two-toucher case — is
  what kept `allow_instance_multi_binding` out of this port entirely: each node hosts exactly one
  compute instance, so every shared DFB stays an ordinary per-node 1:1.

- **The Hardware-configuration section's `unpack_modes` warning caught a real trap in the W factory.**
  See the Friction *Gap* entry below for the part the docs did not cover; the part they did cover —
  legacy `Default` → `UnpackToSrc`, and derive the value rather than guess — is what made the fix
  obviously value-preserving instead of a judgment call
  (`..._w_program_factory.cpp:186-191`).

## Friction

### Gaps

- **The `unpack_modes` "newly-required explicit entry" rule needs the *sibling-factory divergence*
  case spelled out.** The recipe describes the required entry as something you **add** where legacy
  silently defaulted, and tells you to derive its value from the legacy vector. That is exactly what
  the W factory needed — but the surrounding text reads as though a port either has a non-default
  legacy entry to translate *or* nothing to do. This op has **both, in sibling factories over the
  same DFB name**:
  - H set `unpack_to_dest_mode[c_24] = UnpackToDestFp32` → `{{ACCUM_DST, UnpackToDest}}`
    (`..._h_program_factory.cpp:189-195`).
  - W left the vector **entirely `Default`** while still allocating `c_24` at `Float32` under fp32 →
    the validator now *demands* an entry, and the correct value is the **opposite** mode,
    `UnpackToSrc` (`..._w_program_factory.cpp:186-191`).
  - NC never widens `c_24` at all → no entry, and the rule does not fire.

  Three factories, one flag, three different right answers. A porter who found H first and pattern-
  matched W to it would have silently flipped W's precision/perf tradeoff with **no** compile or test
  signal — precisely the failure the section warns about, arriving by a route the section does not
  name. **Suggested doc change:** add a sentence to *Both styles — `unpack_modes`* item 3 along the
  lines of *"derive the value per factory from that factory's own legacy vector — sibling factories
  of the same op frequently disagree, and the required entry's value is not inferable from a sibling."*
  (The audit had already flagged the divergence as anomaly 5, which is what made me check each
  factory separately; without that flag the trap is easy to walk into.)

- **No rule for a genuinely dead CTA.** *Dropped Plumbing* enumerates buffer-address RTAs, magic CB
  indices, `TensorAccessorArgs`, page-size 3rd-arg CTAs, semaphore-ID RTAs, and positional CTAs.
  This op has two CTAs that are none of those and are simply **never read**:
  - `HtWt` in the H reader (`reader_moreh_mean_h.cpp:21` pre-port) — read into a local the body
    never uses.
  - the NC compute kernel's only CTA — `moreh_mean_nc.cpp` reads *no* compile-time arg at all, yet
    `..._nc_program_factory.cpp` emits `units_per_core_group_N` for both per-group descriptors.

  Under legacy these are invisible. Under Metal 2.0 they force a naming decision for a value nothing
  consumes, and the NC one interacts with *Preserved Multiplicity*: it is the **only** thing
  distinguishing the two per-group compute `KernelSpec`s, so dropping it would make them identical
  and invite collapsing the multiplicity the recipe insists on preserving. I kept both verbatim as
  named CTAs. **Suggested doc change:** a short *Dropped Plumbing* entry — *"a CTA the kernel never
  reads is **not** dropped by the port (that is a separate cleanup); name it after the host-side
  value and report it. If it is the per-group CTA of a work-split pair, dropping it would also
  collapse the preserved multiplicity."*

- **Naming guidance is silent on a CTA whose *value* contradicts the kernel's local.** Rule 4 says
  "pick names that match the variables they were going to be assigned to" — which here produces the
  **wrong** name twice (H compute CTA(1) → `Wt`, W compute CTA(0) → `Ht`, both actually carrying
  `units_per_core_group_N`). The brief flagged both sites, so I named the bindings `units_per_core`
  and left the kernel locals alone; the result is the slightly jarring
  `uint32_t Wt = get_arg(args::units_per_core);` (`moreh_mean_h.cpp:19`). That is, I believe, the
  right call — but rule 4 as written points the other way, and only the brief rescued it.
  **Suggested doc change:** amend rule 4 to *"name it after what the host passes; when the kernel's
  local name contradicts the value, follow the host and leave the local alone (renaming locals is
  out of scope)."*

### Confusion

- **"Bind unconditionally to satisfy the validator" vs. "don't bind a conditionally-used DFB
  unconditionally" read as contradictory for ~10 minutes on the mask CBs.** The migration guide's
  *Spec-validator: every DFB needs ≥1 PRODUCER and ≥1 CONSUMER* paragraph says to satisfy the
  constraint by *"declaring the conditional-side `DFBBinding` unconditionally on the host"* — which,
  applied literally to `H/c_3`, means binding the **reader** as PRODUCER even in the unmasked build
  where it never touches the buffer. The recipe's Construct stop-signal list says the opposite:
  binding a conditionally-used DFB unconditionally is a stop signal. Both are reachable and both
  "work".

  What resolved it: the stop signal's stated *reason* is L1 waste, and here there is none — the CB is
  allocated in both configs either way, so the guide's L1 argument does not discriminate. The
  discriminator is the **endpoint-assignment procedure**: count *touchers*, assign the *minimal*
  endpoint set. The reader is not a toucher when unmasked, so the honest census is one toucher →
  self-loop on compute, with the reader's binding conditional. That also keeps the declaration
  truthful about which kernel is the FIFO producer per node — which matters on Gen2, where the labels
  stop being cosmetic. **Suggested doc change:** in the migration guide's producer/consumer paragraph,
  qualify *"declare the conditional-side binding unconditionally"* with *"…when that kernel does touch
  the DFB on the taken path; if the conditional side genuinely has no toucher in some config, run the
  endpoint-assignment census and self-loop the one that remains."*

- **The recipe's test-confirmation checkpoint has no non-interactive fallback.** *Locate and confirm
  the op's tests* requires presenting the found set to the invoker and getting sign-off *before*
  relying on it. This session is non-interactive, so I recorded the set instead of blocking on it
  (see Verification). Worth naming the case, as the audit doc's own *Recipe notes 1* does for the
  readiness sheet: a headless run structurally cannot close a human checkpoint, and "block" and
  "silently proceed" are both wrong answers.

- **Where the tensor-extraction recommendation stops.** The migration guide recommends extracting
  `MeshTensor` once at factory entry and using it *"for the rest of the factory body."* These factory
  bodies query `padded_shape()`, `logical_shape()`, `physical_volume()`, `dtype()` — all present on
  `MeshTensor` — **and** `device()`, which on `ttnn::Tensor` yields the `IDevice*` the existing
  `device->compute_with_storage_grid_size()` / `device->arch()` lines need, whereas
  `MeshTensor::device()` returns a `const MeshDevice&`. Following the recommendation therefore means
  rewriting device handling that the port otherwise does not touch, which scope discipline forbids.
  I kept the `ttnn::Tensor` accessors and used `.mesh_tensor()` only at the two `TensorArgument`
  sites (matching the already-landed `clone` port). **Suggested doc change:** note that the
  extract-once style applies to *tensor* queries, and that a factory reaching `IDevice*` through
  `tensor.device()` should keep doing so rather than restructure device handling mid-port.

## Open items for downstream

- **Shared kernel touches — one, intra-op.**
  - Path: `ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/writer_moreh_mean_unary_interleaved_start_id.cpp`
  - Rung taken: **in-place modification** (rung 3), authorized by the invoker's bundled
    three-factory assignment.
  - Bundled consumer set: `MorehMeanHFactory` + `MorehMeanWFactory` — **both converted in this
    change**, and both now bind it with an identical schema (`dfb::out`, `tensor::dst`, named RTAs
    `num_tiles` / `start_id`).
  - Remaining unmigrated consumers: **none.** No `_metal2` fork exists or was created, so there is
    no sunset list and no pointer comment was added.
  - Cross-op sharing: **none at all.** `grep -rl <filename> ttnn/` over all 8 kernel files returns
    only this op's factories, the `METAL2_*.md` artifacts, and `ttnn/ttnn.egg-info/SOURCES.txt`
    (a packaging manifest, not a consumer).

- **Findings for the op owner (pre-existing; the port preserved all of them verbatim).**
  1. **`fp32_dest_acc_en` is handled three different ways across the three factories** (audit anomaly
     5): H widens `c_24` to `Float32` **and** sets `UnpackToDestFp32`; W widens `c_24` but leaves
     unpack mode default; NC never widens `c_24` at all yet still defines `FP32_DEST_ACC_EN` for its
     kernel. The port cements whichever is correct. Now visible as a three-way divergence in the
     `unpack_modes` blocks (`..._h_program_factory.cpp:189-195`, `..._w_program_factory.cpp:186-191`,
     `..._nc_program_factory.cpp:168-171`), which is a better place to notice it than a
     `vector<UnpackToDestMode>` indexed by magic CB id. **Worth resolving.**
  2. **Two compute-kernel CTAs are misnamed in the kernel** (audit anomaly 2): `moreh_mean_h.cpp:19`
     assigns `units_per_core` into a local named `Wt`; `moreh_mean_w.cpp:18` into a local named `Ht`.
     The bindings are now correctly named; the locals and their comments still describe a reduction
     over tile counts. Renaming the locals is a small, safe, separate PR.
  3. **Two dead CTAs** — `HtWt` in the H reader (read into an unused local) and the NC compute
     kernel's `units_per_core` (never read at all). Both now named CTAs. Dropping the H one is
     trivial; dropping the NC one should be paired with a decision about whether the two per-group
     compute `KernelSpec`s should merge.
  4. **Redundant NC reader RTA** (audit anomaly 3): `input_tile_stride == HtWt * inner_size`
     identically, yet all three are sent (`..._nc_program_factory.cpp` reader RTAs). One is
     derivable on-device.
  5. **`packer_l1_acc` is destructured and never used** in all three factories (audit anomaly 4). It
     is also absent from Metal 2.0's compute config, so nothing was lost — but the unused binding
     remains.
  6. **Dead include** `api/debug/dprint.h` in `reader_moreh_mean_nc.cpp:5` (audit anomaly 6), and the
     dead `divisor` attribute that is hard-rejected yet still hashed and pybound (audit anomaly 1).
  7. **`kernel_lib` sentinel aliases a live buffer** (audit anomaly 7,
     `reduce_helpers_compute.inl:340-343`): the "no accumulator" sentinel is dfb id `0`, which in all
     three `moreh_mean` factories is the live **input** DFB. Harmless today (guarded), but a sentinel
     that names a real buffer is fragile — and in Metal 2.0 it is now *more* jarring, since the id is
     the one thing the binding model otherwise removes from kernel code. **Routes to the `kernel_lib`
     owner, not the ops team.**

- **Test coverage notes.** Coverage for this op is a single nightly pytest file, and it exercises
  only `bfloat16` / `bfloat8_b` (the latter skipped) — i.e. **no `fp32_dest_acc_en` compute-config
  case is covered for the NC factory**, which is the one whose `c_24` handling diverges from its
  siblings. `test_moreh_mean_compute_kernel_options` does sweep compute configs, but only over H, W
  and one NC shape; there is no C++ gtest at all. A porter or op owner touching the fp32 paths should
  not expect the existing tests to catch a regression there.

- **Per-op carry-over.** `moreh_mean_backward` sits in the same directory tree, shares the
  `moreh_common.hpp` / `generate_mm_scaler.hpp` donor headers, and has the same
  reader/writer/compute-pair shape with a `descriptor` factory. The DFB naming and the
  conditional-mask-binding shape used here should transfer almost verbatim. `moreh_mean`'s three
  factories are a reasonable template for the rest of the `moreh` family.

## Verification

- **Build:** `./build_metal.sh --build-tests` — **SUCCESS**, zero errors and zero warnings. The two
  unity chunks carrying this op (`ttnn_op_moreh` `unity_6` = device-op + H + NC, `unity_7` = W) were
  force-rebuilt and recompiled clean. Kernels are JIT-compiled at test time and all 8 compiled
  without error during the run below.
- **Tests: PASS.** `pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_mean.py`
  - `-k "not backward"` (the ported op): **37 passed, 34 skipped, 0 failed** (skips are the
    `bfloat8_b` cases the test file skips itself: *"bfloat8_b is not supported in the kernel"*).
    All three factories exercised, masked and unmasked, single- and multi-tile, keepdim both ways,
    with and without an optional output tensor, across the `compute_kernel_options` sweep.
  - Full file (adds the untouched `moreh_mean_backward` cases): **76 passed, 72 skipped, 0 failed** —
    no collateral damage to the sibling op sharing this directory tree and unity-build chunk.
  - **Program-cache hit path confirmed:** `test_moreh_mean_callback` logs
    `num_program_cache_entries_list=[1, 1]` — the second dispatch hit the cached `Program` and went
    through `UpdateTensorArgs` rather than rebuilding. That is the exact path a surviving custom
    `compute_program_hash` would have broken on the *second* invocation; it is clean, consistent with
    the op never having had one.
- **Test set (discovered, not invoker-confirmed — see Friction/Confusion).** A broad sweep
  (`find tests -iname '*moreh*mean*'`, plus `grep -rl moreh_mean tests/`) finds exactly one
  functional test file across every test tree:
  - `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_mean.py` — the no-regression baseline.
    Relevant cases: `test_moreh_mean_ttnn_dtype` (H / W / NC, single- and multi-tile, masked and
    unmasked, keepdim both ways), `test_moreh_mean_compute_kernel_options` (compute-config sweep,
    incl. `fp32_dest_acc_en`), `test_moreh_mean_optional_output`, and
    `test_moreh_mean_callback` (**program-cache hit path — the case that exercises
    `UpdateTensorArgs`**).
    The same file also holds `moreh_mean_backward` cases; that is a **different** op, untouched here,
    and its results are baseline noise for this port.
  - No C++ gtest matches this op (`grep -rn 'MorehMean' tests/ --include=*.cpp` is empty), and
    `tests/sweep_framework` has no `moreh_mean` sweep (only a mention in `Allops.txt`).
  - `tests/ttnn/profiling/ops_for_profiling.py` references `primary_moreh_mean_*` but is a profiling
    harness, not a correctness test.
- **Anti-pattern self-audit:** all 10 checklist items pass. Greps over the op directory return zero
  hits for `CircularBuffer` / `CBDescriptor` / `CBIndex` / `TensorAccessorArgs` /
  `buffer()->address()` / `get_compile_time_arg_val` / `get_arg_val` / `emplace_runtime_args` /
  positional CTA lists / `.id` extraction / `get_vararg` / `allow_instance_multi_binding` /
  `ProducerOf`-`ConsumerOf` / self-included generated headers, in code. The only residual textual
  matches are two comments that deliberately record what the legacy `unpack_to_dest_mode` vector did,
  which is the silent-precision setting most worth documenting inline.

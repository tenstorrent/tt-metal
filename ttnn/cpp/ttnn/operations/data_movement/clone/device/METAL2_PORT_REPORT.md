# Port Report — `ttnn::operations::data_movement::clone::CloneOperation`

> Port report for the clone op's migration to Metal 2.0. Captures handoff points, successes, friction, and open items for downstream. Pairs with `METAL2_PREPORT_AUDIT.md` and `METAL2_PORT_PLAN.md`.

**Final status:** GREEN. `ttnncpp` and `ttnn` build clean; all 118 clone unit tests pass (24 pre-existing skips). Branch: `akertesz/porting-recipe-clone-test`.

## Handoff points

### API: `TensorAccessor::bank_base_address` is private

Audit Q2 originally proposed option (a): introduce `TensorAccessor(ta::name)` in the sharded kernels and use `.bank_base_address` as the local L1 base. This approach compiled host-side and looked correct from reading `tensor_accessor.h`, but failed at kernel compile time:

```
ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_sharded.cpp:28:78:
error: 'const uint32_t TensorAccessor<...>::bank_base_address' is private within this context
```

The field is `private:` at `tt_metal/hw/inc/api/tensor/tensor_accessor.h:286`. There is no public accessor (no `get_bank_base_address()` method) and no public method that returns the local shard's L1 NoC address directly. `TensorAccessor::get_noc_addr(page_id, offset)` works on the *global* tensor's page layout; for a sharded tensor it computes the bank coordinates from page_id's bank mapping, so the result is the page-0 bank's NoC address, *not* the local core's bank base. `get_shard_noc_addr(shard_id, offset)` has the same shape — shard_id 0 maps to some specific bank (not necessarily the local one).

The legacy buffer-address-RTA pattern (`buffer.address()` injected per-execution) is fundamentally answering a different question than `TensorAccessor`'s page accessors: it's *"what L1 address has the framework assigned to this tensor's shard on the core I'm running on?"* — and the answer is in the framework-injected base address, which TensorAccessor stores privately as `bank_base_address`.

**Workaround taken:** fell back to audit Q2 option (c) — keep buffer-address as a named RTA in the sharded kernels (no TensorAccessor, no TensorBinding on those kernel specs). Audit confirmed this is the documented escape hatch.

**Suggested API change:** add a public accessor on `TensorAccessor` that returns the local shard's L1 NoC address (e.g., `get_local_shard_noc_addr(uint32_t offset = 0) const`). For interleaved tensors, this returns the local bank's address if any. The accessor should be a 1-line wrapper around `bank_base_address` + the noc-addr math, with the understanding that it's only meaningful for sharded-with-one-bank-per-core configurations. Without such an accessor, every porting case that legacy-routed the buffer address through an RTA falls back to the same escape hatch, and the binding mechanism's Principle 2 ("named bindings replace address RTAs") leaves these kernels behind.

- Affected files (sharded kernels using escape hatch): `device/kernels/read_kernel_sharded.cpp`, `device/kernels/read_kernel_rm_sharded.cpp`, `device/kernels/write_kernel_sharded.cpp`, `device/kernels/write_kernel_rm_sharded.cpp`.
- Tagged "API: TensorAccessor.bank_base_address accessor needed."

### Doc: audit doc's Check 1 RED action doesn't account for the "direct jump from `host_api.hpp` to Metal 2.0" path

Audit Q1: the audit doc's Check 1 categorically reds-out ops on imperative `host_api.hpp` builder style. The clone op was on that style; the user explicitly requested the direct jump to Metal 2.0, skipping the `ProgramDescriptor` intermediate stop. The port worked cleanly. The current audit doc text reads:

> Report to the user that `ProgramDescriptor` migration is a **prerequisite to Metal 2.0 porting** — a substantial, standalone body of work with TTNN-infrastructure implications, addressed in its own PR. The audit's deliverable here is the report identifying the prereq; the prereq work itself is a separate session. **Do not attempt it as part of this audit, do not bundle it with anything, do not propose a partial conversion.**

This wording is strict enough that a porting AI following it literally would refuse the port outright. The clone port itself was straightforward in the host-API shape — the legacy imperative calls (`CreateCircularBuffer`, `CreateKernel`, `SetRuntimeArgs`) all have direct Metal 2.0 equivalents, and the test cases provided a tight correctness check. The intermediate `ProgramDescriptor` stop would have added infrastructure churn without separating any concerns the Metal 2.0 port doesn't already separate.

**Suggested doc evolution:** soften Check 1's prescribed action from "refuse, do not propose a partial conversion" to "stop, surface the gap, and ask the user whether to (a) bundle a `ProgramDescriptor` migration, (b) attempt a direct port to Metal 2.0, or (c) defer." Mention that direct ports may be appropriate for small ops without TTNN-infrastructure entanglement. Alternatively: add a new tier `BYPASS-CANDIDATE` for cases where the direct jump is sanctioned by judgment, with criteria for recognizing them (small op, kernel set is op-local, no `override_runtime_arguments` complexity beyond buffer-address updates).

- Tagged "Doc: audit Check 1 prescribed-action wording."

### Doc/framework: `unpack_to_dest_mode` requirement isn't documented in the recipe / migration guide

The Metal 2.0 validator requires an explicit `unpack_to_dest_mode` entry on `ComputeConfiguration` whenever a compute kernel consumes an FP32 DFB with `fp32_dest_acc_en = true`:

```
TT_FATAL: Kernel 'compute_g1' consumes FP32 DFB 'input' with fp32_dest_acc_en=true,
but has no unpack_to_dest_mode entry for it. This configuration requires an explicit
choice between UnpackToDestMode::Default (unpack via SrcA/B — enables binary FPU
ops, precision reduced to ~19 bits) and UnpackToDestMode::UnpackToDestFp32
(unpack direct to Dest — full FP32 precision, SrcA/B access disabled for this DFB).
```

Legacy `ProgramDescriptor`-based ops (and even older `ComputeConfig` users) don't set this field explicitly — they rely on a silent default. Metal 2.0 made the choice explicit, which is correct, but the port recipe doesn't mention the field at all. A porter following the recipe mechanically (legacy `ComputeConfig` → Metal 2.0 `ComputeConfiguration`, 1:1 field translation) misses the new requirement and hits the validator at first execution.

**Suggested doc evolution:** add a brief note to the recipe's [Construct paired spec + run-params](recipe.md#construct-paired-spec--run-params) step under `ComputeConfiguration`:

> When a compute kernel consumes an FP32 DFB with `fp32_dest_acc_en = true`, you must explicitly set `unpack_to_dest_mode` on `ComputeConfiguration` with an entry for that DFB. `UnpackToDestMode::Default` preserves legacy semantics; `UnpackToDestMode::UnpackToDestFp32` selects the FP32-via-Dest pipeline. The validator will error loudly if the entry is missing.

This is the kind of "framework-conformance fix" that's easy to discover on first build but is silent in the legacy → Metal 2.0 diff. Calling it out in the recipe prevents the round trip.

- Affected file in the port: `clone_program_factory.cpp:268` (added `.unpack_to_dest_mode = {{INPUT_DFB, UnpackToDestMode::Default}}`).
- Tagged "Doc: recipe ComputeConfiguration.unpack_to_dest_mode requirement."

## Successes

### Port plan's "Dropped Plumbing" enumeration caught the alias case before construction

The legacy clone factory has `dst_cb_id = src_cb_id` when `!convert_dtype`, creating only one CB. Without the "Dropped Plumbing" listing, the reflex would have been to declare INPUT_DFB + OUTPUT_DFB always (mirroring the "one CB per buffer index" mental model that holds when `convert_dtype = true`). Explicitly enumerating the magic-CB-index → DFBBinding substitution in the plan made the alias visible: writer binds INPUT_DFB as CONSUMER when !convert_dtype, not OUTPUT_DFB. The validator's PRODUCER/CONSUMER balance rule then makes sense (reader produces, writer consumes, balanced). Saved a constructed-but-wrong shape that would have failed at MakeProgramFromSpec validation.

### Migration guide Example 1 (`Single-Core Reader / Writer with One DFB`) was a near-perfect template

The clone op's structure (reader → DFB → writer with optional compute in between) maps directly onto Example 1 plus the compute-in-the-middle elaboration. The example's named-handle pattern (`READER`, `WRITER`, `DFB`, `INPUT`, `OUTPUT` constants) translated 1:1 with prefix changes for unity-build hygiene (`C_READER` etc.). The interleaved-vs-sharded branching pattern from Example 2's per-node RTAs translated to clone's per-core RTAs without friction. This is the kind of "shape lift" the migration guide examples are designed for.

### Patterns catalog's [Preserved Multiplicity](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) discipline applied cleanly to the compute kernels

When `convert_dtype = true` and `split_work_to_cores` yields a non-empty `core_group_2`, the legacy code creates two compute KernelDescriptors with the same source but different `num_units_per_core` CTA values. The patterns catalog's anti-pattern entry against demote-to-RTA was crystal clear: instead, two compute KernelSpecs (`C_COMPUTE_G1`, `C_COMPUTE_G2`) of the same source with different CTA values, each placed in its own WorkUnitSpec, both binding INPUT_DFB and OUTPUT_DFB as multi-binding endpoints. No demotion. Took ~15 lines vs. the wrong-shape's ~3. The `test_clone_dtype_conversion` matrix exercises this with both core groups populated, and all variants pass.

### Recipe's "stop signal" on adding `wait_front`/`pop_front` to balance topology fired pre-emptively (no edit needed)

The clone DFBs are simply pipelined (reader produces → writer consumes), no conditional binding. The pattern in question didn't arise — but reading the stop signal during planning calibrated my expectation that DFB balance is purely a host-side concern. When the !convert_dtype writer-binds-INPUT_DFB construction landed, I was already framing this as "host validator concern" not "is the kernel topology balanced?".

### Reference port branch (W reduce on `akertesz/metal2-documentation`) was the most valuable single artifact

When stuck on the framework conventions (`ProgramSpecFactoryConcept` hpp shape, `m2::` aliasing, `KernelRunParams::NodeNamedRTAs` shape, `std::cref(tensor.mesh_tensor())` for TensorArg), opening the W reduce factory at commit `3c182a50ca6` gave the canonical form. The migration guide's examples are simplified for pedagogy; the W reduce factory is a real, complete port that landed. Treat it as the de facto template alongside the docs. (This was a "shape lift" inside-out: rather than building from the docs, I built from the W reduce factory's pattern and cross-referenced the docs for the underlying rationale.)

## Friction

### Gap: the audit doc's Check 3 YELLOW guidance has no entry for "direct L1 access via buffer-address RTA" (the legacy sharded-clone pattern)

The audit's Check 3 YELLOW description targets "exotic NoC walks; sub-page access; address arithmetic the iterators don't support." The clone sharded kernels don't fit that — they iterate by simple stride. But they also don't use `TensorAccessor`, so the GREEN case doesn't apply. And the RED case ("convert to TensorAccessor in a prior PR") was the audit's default classification, but turned out to be wrong: the access fundamentally needs the local shard's L1 base address, and TensorAccessor doesn't surface that publicly (see Handoff point 1).

This is the most actionable gap in the audit doc. **Suggested addition:** a fourth Check 3 classification case:

> **Yellow — accesses tensor memory via a buffer-address RTA + `get_noc_addr(addr)` direct L1 reads (no per-page iteration).** This pattern occurs for sharded kernels that copy from / to their local shard's L1 region. Today, the recommended path is to keep the buffer-address RTA as an explicit escape hatch — declare a named RTA `<tensor>_buffer_addr`, populate from `tensor.buffer()->address()` per execution, and skip the TensorBinding on this kernel's spec. The TensorParameter may also be dropped if no kernel binds it; otherwise it must be referenced. **Future:** a `TensorAccessor::get_local_shard_noc_addr()` accessor (when added) will subsume this case into a TensorBinding-clean form.

This corrects the audit's default-to-RED guidance for the case and documents the supported workaround.

### Gap: the `unpack_to_dest_mode` requirement on compute kernels

Already detailed under Handoff points. Worth restating as a friction entry — this was the first test failure I hit and the recipe didn't prepare me for it.

### Confusion: "TensorParameter must have ≥1 TensorBinding" invariant is implicit

The Metal 2.0 validator at `program_spec.cpp:422-426` errors if a TensorParameter is declared but not bound to any kernel. The migration guide doesn't mention this; the recipe doesn't either. I discovered it by reading the validator source after my first sharded port attempt confused me ("Why does the TensorAccessor approach also need to drop the TensorParameter? Can't I just declare it and not bind it?").

**Suggested doc evolution:** add to the migration guide's TensorParameter section a one-line statement: *"Every TensorParameter declared in `ProgramSpec::tensor_parameters` must be referenced by at least one TensorBinding on at least one KernelSpec. Otherwise the validator errors."* It's the kind of invariant where stating it removes a whole class of stuck-debugging.

### Confusion: `bank_base_address` field appearance in `tensor_accessor.h`

The header has a `public:` block at line 70, then private members start at line 285 with `private:`. Skimming the file (which I did during planning), the field `const uint32_t bank_base_address` looks public — the private label is far above its declaration in the file but reading-mid-file (which is what I was doing to confirm my approach) I missed the access label. This is a minor file-structure issue, not a doc issue, but it cost me one full build cycle.

This is the kind of low-frequency, high-cost confusion that's hard to engineer out — the header is structured normally for C++. Mentioning in the patterns catalog that `bank_base_address` is private (as part of the API: get_local_shard_noc_addr() handoff entry) would short-circuit future confusion.

### Confusion: the `noc_async_read` (free function) vs. `Noc::async_read` (DataflowBuffer-aware method) inconsistency

The interleaved clone kernels use the `Noc` wrapper with `noc.async_read(s, src_cb, ...)` for TensorAccessor + DataflowBuffer-aware reads. The sharded clone kernels use the free `noc_async_read(local_l1_read_addr, src_cb_write_addr, tile_size)` because the read is from a raw L1 address, not a TensorAccessor page. The Device 2.0 migration guide notes the `Noc` wrapper supersedes the free functions; but the wrapper's methods are typed around TensorAccessor + DataflowBuffer, with no overload taking a raw `(uint64_t noc_addr, uint32_t local_addr, uint32_t size)`.

This isn't a port blocker — the free function still works and is documented as available — but it's a noticeable seam between "Device 2.0 native" and "legacy fallback" in the same kernel set. Worth surfacing as an API gap for future Device 2.0 evolution: the `Noc` wrapper could grow a `Noc::async_read(uint64_t noc_addr, DataflowBuffer&, size, ...)` overload to bring the sharded kernels under the wrapper umbrella.

### Friction: the port plan template's "Cross-references" links use relative paths with ~7 `..` hops

The catalog and recipe sit at `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`. The port plan lives at `ttnn/cpp/ttnn/operations/data_movement/clone/device/`. The relative link is `../../../../../../../docs/source/...` — seven `..` hops. This is technically correct, looks fragile, and is brittle to anyone moving the op directory. Not a port-blocker but worth noting; the doc system might benefit from absolute-from-repo-root link form (`/docs/source/...`) or symbolic anchors.

### Friction novel to this op (vs. the prior reduction test-drives)

- **Direct-jump-from-`host_api.hpp`** scope question (audit Q1) didn't apply to the reduction port (which was on `ProgramDescriptor`).
- **Sharded direct-L1-access** pattern (audit Q2) — reduction has a borrowed-memory DFB path on its H factory's sharded branch, but reduction's sharded path uses borrowed-memory DFBs and TensorAccessor for the input/output read, not raw L1 addressing. Clone's sharded path uses raw L1 — different shape entirely, and the friction it generates (Handoff point 1) is novel.
- **`unpack_to_dest_mode`** wasn't surfaced in the reduction port (where the reduce compute kernels happen to set it via SrcA/B-only fidelity choices, or the input DFBs aren't FP32 in the test matrix). Clone's dtype-conversion test parametrizes over `fp32_dest_acc_en` explicitly, which forced the field's requirement.
- **Branch-multiplicity within a single factory.** Clone has one factory with four kernel-pair branches (× optional compute). Reduction has multiple factories, each with a more linear shape. The clone factory ended up with significant if/else logic inside `create_program_spec`; the per-branch RTA-schema-and-RTA-population assembly was ~80 lines of branching code. The recipe's planning step didn't specifically prepare me for this — the "preserve multiplicity" guidance covers compute kernels, but not the broader "one factory handling multiple structurally distinct branches via runtime configuration." Worth a future doc note: "for ops whose single factory branches widely on operation_attributes, consider helper functions or per-branch lambdas to keep create_program_spec scannable."

## Open items for downstream

- **Pattern catalog candidate — "Sharded shard-address access via buffer-address RTA (escape hatch)."** The pattern is reusable across any op that needs the local L1 address of a sharded tensor without per-page iteration. Likely fits as a `Pattern` (or possibly `Caution`) entry, paired with the planned future TensorAccessor accessor.
- **Sibling-op audit candidate.** Other `data_movement/` ops with similar (interleaved + sharded) × (tilized + RM) variant matrices and the same legacy direct-L1 sharded pattern likely exist. Reshape, slice, untilize are obvious candidates. A targeted audit of `ttnn/cpp/ttnn/operations/data_movement/` for ops still on `host_api.hpp` would identify the cluster.
- **Test coverage observation.** The `test_clone_sharded_*` tests cover the (tilized × RM) × sharded paths. The compute-with-sharded combination (`test_clone_sharded_dtype_conversion`) exercises the compute kernel + sharded reader/writer pairing. The per-core-group multiplicity case (`core_group_2` non-empty + `convert_dtype`) is only exercised on the interleaved path (`test_clone_dtype_conversion`). That's adequate coverage but worth knowing if the per-group-CTA-CTA mapping is a regression risk.
- **Recipe / catalog evolution from the friction list above:**
  - Check 1 prescribed-action softening / new BYPASS-CANDIDATE tier (Friction Q1).
  - Check 3 sharded-direct-L1 case addition (Friction Gap 1).
  - `unpack_to_dest_mode` note on ComputeConfiguration in recipe (Friction Gap 2).
  - "Every TensorParameter must have a TensorBinding" invariant statement in migration guide (Friction Confusion 1).
  - `bank_base_address` privacy mention paired with API: get_local_shard_noc_addr (Friction Confusion 2 / Handoff 1).
  - `Noc` wrapper overload gap for raw NoC reads (Friction Confusion 3).
  - "Wide branching in `create_program_spec`" guidance (Friction novel-to-clone bullet).

## Verification summary

- `cmake --build build_Release --target ttnn -j 8`: clean (no warnings, no errors).
- `pytest tests/ttnn/unit_tests/operations/data_movement/test_clone.py`: **118 passed, 24 pre-existing skips**.

Test functions exercised:
- `test_clone_shape` (interleaved tilized + RM, various shapes)
- `test_clone_memory_config` (interleaved layout / dtype combinations)
- `test_clone_dtype_conversion` (interleaved with compute kernel; per-group CTA multiplicity case)
- `test_clone_callback` (program-cache hit path; UpdateTensorArgs fast path exercised)
- `test_clone_sharded_tilized` (sharded tilized escape-hatch path)
- `test_clone_sharded_row_major` (sharded RM escape-hatch path)
- `test_clone_sharded_dtype_conversion` (sharded + compute kernel)

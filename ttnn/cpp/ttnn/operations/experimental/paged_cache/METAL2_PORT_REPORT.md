# Metal 2.0 Port Report — `experimental/paged_cache`

## Outcome

**`PORTED`** — all eight factories of `experimental/paged_cache` converted to Metal 2.0, across three
DeviceOperations and eleven kernels. Nothing in the op builds a `ProgramDescriptor` any more.

- Four single-device factories on **`CustomProgramSpecFactoryConcept`** (the audit's target).
- Four `*MeshWorkloadFactory` variants on **`MeshWorkloadSpecFactoryConcept`** — **not** the concept the
  brief named, and outside the port recipe's stated coverage. Proceeded on explicit invoker
  authorisation after surfacing the mismatch; see [Handoff points](#handoff-points) #1.

**Tests: 242 passed / 48 skipped / 0 failed, identical to the pre-port baseline**, with the Metal 2.0
legality checks forced on and proven live (231 markers from each of the two translation units). Full
detail under [Verification](#verification).

## Provenance

- **Recipe docs (this port):** `f9451e2a21d 2026-09-09 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `f9451e2a21d 2026-09-09 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## TTNN ProgramFactory

### Concept realized

Two concepts, not one.

- `PagedFillCacheProgramFactory`, `PagedUpdateCacheProgramFactory`,
  `PagedTiledFusedUpdateCacheProgramFactory`, `PagedRowMajorFusedUpdateCacheProgramFactory` →
  **`CustomProgramSpecFactoryConcept`**, as the audit chose. Each has an `override_runtime_arguments`,
  now returning a `ProgramRunArgs`.
- `PagedFillCacheMeshWorkloadFactory`, `PagedUpdateCacheMeshWorkloadFactory`,
  `PagedTiledFusedUpdateCacheMeshWorkloadFactory`, `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` →
  **`MeshWorkloadSpecFactoryConcept`**. The audit named `CustomProgramSpecFactoryConcept` for these
  too; it cannot express what they do. Full evidence in [Handoff points](#handoff-points) #1 and in
  `METAL2_PORT_PLAN.md`.

**Cache-hit tensor-arg completeness (custom concept).** On this concept the framework refreshes
nothing on the porter's behalf, so each override returns a `TensorArgument` for **every**
`TensorParameter` the spec declares, on every dispatch, including the borrow-only parameters that no
kernel binds (they are what re-point the borrowed input-shard buffers). No parameter is deliberately
skipped. The ported-from overrides re-applied every address they had, so this matches their set.

### Device-op-class edits

- **Pybind entry points removed: none.** [`paged_cache_nanobind.cpp`](paged_cache_nanobind.cpp) binds
  only the three user-facing op functions; no `create_descriptor` was ever exposed. **This port makes
  no user-visible API change.**
- **Custom `compute_program_hash`: left intact, untouched, on all three device-ops** —
  [`paged_fill_cache_device_operation.cpp:216-225`](device/fill_cache/paged_fill_cache_device_operation.cpp#L216-L225),
  [`paged_update_cache_device_operation.cpp:314-338`](device/update_cache/paged_update_cache_device_operation.cpp#L314-L338),
  [`paged_fused_update_cache_device_operation.cpp:250-268`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L250-L268).
  Each excludes scalar attributes (`update_idxs`, `batch_offset`, `batch_idx_fallback`, `noop`) that
  the translated overrides re-apply on every cache hit, so the two halves stay paired.
- **One edit outside a factory body, forced by the port**:
  [`paged_fused_update_cache_device_operation.cpp`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp)
  held the four fused factories' `override_runtime_arguments` and the shared `patch_runtime_args`
  helper they used, placed there so the legacy arg-index constants lived in one translation unit.
  Metal 2.0 run args are keyed by name, so those constants no longer exist; the four overrides moved
  into their own factory `.cpp` files and the helper block was deleted. Three now-unused includes went
  with it (`circular_buffer.hpp`, `host_api.hpp`, `program.hpp`). The device-operation's own methods
  (`select_program_factory`, `validate*`, `compute_output_specs`, `compute_program_hash`) are
  byte-identical.

### Open items

- **Relaxation candidates: none observed.** All three hashes feed `tensor_args` wholesale into
  `hash_operation`, so every input's full `TensorSpec` participates; nothing invites a relaxation. The
  port declares none, matching the audit's `TensorParameter relaxation = none`.
- The mesh-workload concept has no port procedure. Four structural decisions a procedure would have
  settled are recorded in `METAL2_PORT_PLAN.md`; see [Handoff points](#handoff-points) #1.

## Handoff points

### 1. The audit named a concept that cannot express four of the eight factories

**Owner: readiness-sheet / audit owners.** The brief records
`Target concept: CustomProgramSpecFactoryConcept` for all eight rows. For the four
`*MeshWorkloadFactory` variants that is wrong, and the failure would have been silent.

Each mesh factory builds a **per-mesh-coordinate** program. `CustomProgramSpecFactoryConcept` has no
channel for a per-coordinate value in either direction:

- **Cache miss** — the adapter calls `create_program_artifacts` once (it takes no coordinate) and
  applies the *same* `ProgramRunArgs` to every range
  ([`mesh_device_operation_adapter.hpp:971-985`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L971-L985)).
  The adapter states the consequence itself at
  [`:1005-1007`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L1005-L1007).
- **Cache hit** — `override_runtime_arguments` runs once per **range**, not per coordinate
  ([`:1017-1025`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L1017-L1025)), and
  with uniform tensor storage the whole mesh collapses to one range
  ([`device_operation.hpp:334-336`](../../../../../../ttnn/api/ttnn/device_operation.hpp#L334-L336)).

**Why this matters beyond a concept label:** had `PagedFillCacheMeshWorkloadFactory` been ported onto
the named concept, every chip would have received the `noop` computed for coordinate `(0,0)` — so a
chip *excluded* from `mesh_coords` would have written to the cache. Wrong numerics, on the
`mesh_coords` path only, with no build or validator error. The brief's assessment that this factory
"is already in the recommended shape" because its exclusion rides on a runtime arg, and its
recommended fix for the other three (build the full program, set an all-cores `has_work = 0`), both
fail for the same reason.

The port instead put all four on `MeshWorkloadSpecFactoryConcept`, which does express it and **is**
implemented in this tree
([`operation_concepts.hpp:118-132`](../../../../../../ttnn/api/ttnn/operation_concepts.hpp#L118-L132),
adapter at
[`mesh_device_operation_adapter.hpp:1035-1130`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L1035-L1130)).

**Two asks.** (a) Correct the sheet's `Porting Target` for these four rows, and add the recognition
rule the audit lacked: *a `descriptor` factory whose `create_descriptor` takes a
`std::optional<MeshCoordinate>` and varies per coordinate is a mesh-workload target, not a
single-program one.* (b) The recipe covers only the two single-program concepts and tells a porter to
stop on a mesh-workload target; a procedure for it would have settled the four decisions this port had
to make unaided (program granularity, omitted vs. noop coordinates, how `tensor_coords` bounds the
emitted ranges, and where the cache-hit override's range maps to a coordinate).

### 2. Empty or fully-disjoint `mesh_coords` now raises on three ops

**Owner: paged_cache op owners.** `MakeMeshWorkloadFromSpecs` requires at least one program
([`program_spec.cpp:3385`](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L3385)), and
the mesh-spec adapter re-asserts it
([`mesh_device_operation_adapter.hpp:1078`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L1078)).

`PagedUpdateCacheMeshWorkloadFactory` and both fused mesh factories preserve the ported-from idiom of
**omitting** an excluded coordinate. When `mesh_coords` excludes *every* coordinate in `tensor_coords`,
they now emit zero programs and the call raises, where the ported-from path silently dispatched
nothing. `paged_fill_cache` is immune: it emits a program for every coordinate and only varies `noop`.

This is a **behaviour delta the port did not fix**, because both available alternatives change more
than they preserve: emitting a dummy program for excluded coordinates would alter the program set on
the normal path too, and the concept has no way to express "no program anywhere." It needs a one-line
ruling: is raising acceptable for an empty `mesh_coords`, or should the op reject that argument
earlier in `validate`? *(Not reachable from the confirmed test set — no test passes an
all-excluding `mesh_coords`; see [Open items](#open-items-for-downstream) #4.)*

### 3. No kernel-lib or LLK gaps

`compute_kernel_lib::untilize` / `tilize` take their handles as `uint32_t` non-type template
parameters, and `dfb::name`'s `constexpr` conversion covers template-argument position, so the
`dfb::` tokens pass straight through with no donor-side change and no fork. `compute_kernel_hw_startup`
likewise takes the tokens directly. No call site required a `sem::` or `tensor::` handle to cross the
op boundary, so the recipe's boundary assumption held.

## Successes

- **[Conditional / optional resource bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-resource-bindings)
  was the single most valuable entry**, and its *Promote a CTA gate to a define* paragraph fired exactly
  as written. Every one of these kernels declares its optional buffers unconditionally at function
  scope and gates only the *use* behind `if constexpr` — which still name-looks-up the discarded
  branch. Without the warning the natural move is to bind unconditionally, which the entry explicitly
  calls out as wrong on two counts. The catalog's "the define must reach **every** kernel that
  references the resource, not just the one the legacy factory happened to send a flag to" is precisely
  the shape here: `use_index_tensor` and `is_paged_cache` are read by the reader *and* the writer in
  all three update paths.
- **The [self-loop hard gate](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
  ("count the distinct kernels first") kept `fill_cache`'s three metadata buffers correct.** Re-derived
  rather than transcribed, per the recipe: the writer alone touches each of `page_table`, `batch_idx`
  and `valid_seq_len`, so each is a one-toucher self-loop, and the brief's disposition matched.
- **The recipe's insistence that `opt_level` is an *absent line*, not a wrong value, caught a real
  miss.** All three compute kernels came off `ComputeConfigDescriptor` with no `opt_level` field, which
  reads as "nothing to carry over" — and resolves to `O3`, against Metal 2.0's `O2` default. Every
  compute `KernelSpec` here carries an explicit `O3` because the section said to check it mechanically
  rather than by eye.
- **The `hw_config` dropped-field check found one.** All three compute factories resolve a full TTNN
  `ComputeKernelConfig` and then copy only `fp32_dest_acc_en` onto the descriptor. Routing them through
  `to_compute_hardware_config` would have silently *started* honouring `math_fidelity`,
  `math_approx_mode` and `dst_full_sync_en` — a behaviour change dressed as a translation. Building
  `ComputeGen1Config` directly with only `enable_32_bit_dest` set reproduces the ported-from config
  exactly, because the two structs' defaults coincide.

## Friction

### Gaps

1. **The audit's CB-endpoint census has no column for CB *aliasing*, and missed three instances.**
   The census answers "how many kernels touch this CB" per `(CB, config)` and lists c_24 and c_25 (and
   the row-major c_5 / c_6) as independent rows. They are not independent: each pair sits on **one**
   `CBDescriptor` with two `format_descriptors`
   ([`paged_update_cache_program_factory.cpp:210-225`](device/update_cache/paged_update_cache_program_factory.cpp) in
   the pre-port revision, and the matching blocks in both fused factories), which is the
   [Aliased DFBs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-aliased-dfbs-legacy-aliased-cbs)
   pattern. Splitting them into independent DFBs, which is what a census read literally invites, would
   have doubled the intermediate SRAM footprint and broken the kernels' assumption that the two indices
   share an address. The port catches it only because the inventory step reads the `CBDescriptor`
   literals. **Suggested fix:** the audit's CB inventory should flag a multi-element
   `format_descriptors` explicitly, the way it flags `GlobalCircularBuffer` and `address_offset`; the
   port recipe already expects the inventory to surface it, so the gap is on the audit side.
2. **The recipe's stop rule and its disagree-with-the-audit rule overlap, and point opposite ways.**
   [§What this procedure covers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md)
   says "a brief naming any other target concept is outside this procedure — stop", while the planning
   step says a disagreement with the audit's concept is to be surfaced to the invoker. When the brief
   names a *supported* concept and the porter finds it *wrong*, both rules apply and they do not agree.
   One clause would settle it: the stop rule keys on the brief's *named* concept, so a porter who
   concludes a different concept is required is in the disagree-and-surface case, not the stop case.
3. **A dead CB-index compile-time arg has no Metal 2.0 representation, and the "keep dead args" rule
   does not carve it out.** The recipe says dead compile-time args are the ops team's and should be
   neither plumbed nor removed. That works for scalars (`max_blocks_per_seq`,
   `log_base_2_of_page_size`, `log2_page_table_stick_size` are all kept). It cannot work for a dead
   **CB index**: rule 2 converts a CB index into a `DFBBinding`, and the row-major fused compute
   kernel's `in1_cb` / `in2_cb` have no binding to make, because that kernel never touches an input
   buffer. Binding one anyway would add a third toucher to a buffer whose census is exactly
   reader-P + writer-C and force the multi-binding flag for something nothing reads. They are dropped;
   the value was never read, so it is zero-functional-change. **Suggested fix:** one sentence in the
   dead-arg guidance saying a dead CB-index CTA is dropped rather than kept, since the construct it
   would convert to does not exist.

### Confusion

4. **"No `_metal2` fork exists beside any of them, so this port creates the first fork for each"** in
   the audit brief reads as an instruction to fork all eleven kernels. It is a rung-1 observation
   (*does a fork already exist?*) stated as a rung-2 conclusion. No fork is warranted here: the
   shared-kernel Caution triggers on a source bound by factories that **will not all convert in the
   same change**, and every binder of every kernel converts in this change — each mesh factory
   delegates its build to its single-device sibling, and no two device-ops share a kernel. Converting
   in place is correct and leaves no legacy binder behind. Forking would have created eleven
   permanent duplicate files for nothing.
5. **`Table` versus `Group` in the conditional-build case.** `KernelSpec::compile_time_args` is a
   `Table` and `dfb_bindings` a `Group`, and both frequently need conditional population. The
   recipe warns that `Table` has no `push_back`, which is right, but the readable shape for a
   conditionally-built spec is to declare *both* collections as named locals up front and
   `emplace` / `push_back` into them before the designated-initializer block — otherwise the ternary
   form the pattern entry shows does not scale past one condition. Worth a line in the catalog: this op
   has up to four independent conditions on one kernel.

## Open items for downstream

1. **Shared kernel touches: none.** All eleven kernels are owned by this op and converted in place; no
   `_metal2` fork was created or reused, and no peer op's directory was written to. There is no sunset
   list and no coordination cost for a later port. (See Friction #4 for why the brief's phrasing
   suggests otherwise.)
2. **The `unused_cores` instances disappear, and it is a net gain.** Both fused factories previously
   placed all three kernels over `all_cores_bb` (the bounding box of the two input shard grids) and
   handed each core in neither grid a one-element `{!has_work}` arg vector. The two-`WorkUnitSpec`
   split leaves those cores outside both work units, so they get no kernels and no buffer reservation.
   **No output value moves** — they early-exit today and are absent afterwards. The op dispatches to
   fewer cores and stops reserving roughly **59–112 KB on each**. The set is empty in the unit tests,
   and holds **20 cores** in the llama3-70b galaxy decode configuration (the audit derives this under
   *Team-only → `unused_cores` in a shipping configuration*). Observable only as a profiler trace with
   fewer cores and an SRAM report with less reserved; worth confirming on a galaxy run.
   The `in0_sequential_mode` semaphore was narrowed from `all_cores_bb` to the union of the two shard
   grids for the same reason: no kernel outside the union can touch it.
3. **`has_work` is now constant 1 on every node**, since every node a work unit covers does work. It is
   kept as a named runtime arg and the kernels' early `return` is untouched, so the diff stays minimal.
   A later pass could drop the argument and the branch together; that is a kernel-logic change, not
   port work.
4. **Mesh-factory test coverage is partial, and the gap is specific.** Of the four factories on
   `MeshWorkloadSpecFactoryConcept`:
   - `PagedFillCacheMeshWorkloadFactory` **is** exercised, by
     `test_paged_fill_cache_mesh_coords` (×2) and `test_paged_fill_cache_batched_mesh_coords` in
     [`test_paged_update_cache.py`](../../../../../../tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py).
     On a single device the mesh is 1×1 and `mesh_coords = {MeshCoordinate(0, 0)}`, so these cover the
     **included**-coordinate path only.
   - `PagedUpdateCacheMeshWorkloadFactory` is covered only by
     [`test_paged_cache_mask.py`](../../../../../../tests/ttnn/unit_tests/operations/transformers/test_paged_cache_mask.py),
     which **skips** on a single-device machine ("Requested more devices (2) than available (1)"). It
     skipped in the pre-port baseline too, so this is not a regression, but it means the factory is
     compile-and-review verified only here.
   - Both **fused** mesh factories have no in-repo caller that passes `mesh_coords` at all, so nothing
     exercises them.

   The **excluded**-coordinate branch is unobservable on one device by construction, which also puts
   the behaviour delta in Handoff #2 out of reach of this run. `test_paged_cache_mask.py` is the
   natural place for a T3K/Galaxy acceptance run, and the fused ops have no equivalent test to
   piggyback on.
5. **The `unpack_modes` branch is unexercised.** All three compute factories resolve
   `fp32_dest_acc_en` from the caller's `compute_kernel_config`, and no test in the confirmed set
   overrides it, so it is `false` throughout: the intermediate format stays `Float16_b`, no consumed
   buffer is `Float32`, and the port's `unpack_modes` block never runs. It only fires when a caller
   passes `fp32_dest_acc_en = true`, and when it does the validator enforces the required-entry rule
   loudly rather than silently — but the first caller to set that flag is the first to exercise this
   code. Worth a targeted case if the flag is ever meant to be used.
6. **Op-team findings noticed while reading, not acted on** (these restate the audit's *Misc anomalies*
   from the porter's side, and the port preserved every one):
   - The `share_cache` sequencing semaphore is allocated unconditionally even when `share_cache` is
     false and no kernel touches it.
   - `paged_fill_cache` accepts a `compute_kernel_config` and discards it
     ([`paged_cache.cpp:80-81`](paged_cache.cpp#L80-L81)); the factory builds no compute kernel.
   - All three compute factories resolve a full `ComputeKernelConfig` and honour only
     `fp32_dest_acc_en`, silently ignoring `math_fidelity`, `math_approx_mode` and `dst_full_sync_en`.
     A caller setting any of those gets no error and no effect.
   - The two fused writers disagree on draining the index and page-table buffers: the tiled one
     `pop_front`s both, the row-major one `wait_front`s and never pops. Both work because each buffer
     is filled and read exactly once per execution. Preserved exactly as found, asymmetry included.
   - The row-major fused compute kernel's two input-buffer compile-time args and its `is_input1`
     runtime arg were dead; they are dropped by the port (Friction #3).

## Verification

### Legality checks were provably live

`skip_validation` was forced to `false` as the first statement of **all nine** functions
`grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp` names, with a one-shot marker in
each of the two translation units. The post-port run logged:

```
231 METAL2_CHECKS_FORCED (program_run_args.cpp:565)
231 METAL2_CHECKS_FORCED (program_spec.cpp:2950)
```

Both files fresh, both gates running, on every one of the 231 program constructions. The scaffolding
has been reverted; `git status` shows no file outside the op directory.

### Tests

Same four files before and after, `TT_METAL_WATCHER=10` on both runs:

| | result |
|---|---|
| **Baseline** (pre-port) | **242 passed, 48 skipped, 0 failed** (815s) |
| **Post-port** | **242 passed, 48 skipped, 0 failed** (581s) |

Identical pass/skip sets, no Watcher assertion and no `0xdeadc0de` in either run. The skips are
environmental (`test_paged_cache_mask.py` needs 2 devices; 1 available) and skipped identically before
and after.

Confirmed test set:
[`nightly/.../test_paged_update_cache.py`](../../../../../../tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py),
[`test_paged_fused_update_cache.py`](../../../../../../tests/ttnn/unit_tests/operations/transformers/test_paged_fused_update_cache.py),
[`test_paged_cache_flexible_geometry.py`](../../../../../../tests/ttnn/unit_tests/operations/transformers/test_paged_cache_flexible_geometry.py),
[`test_paged_cache_mask.py`](../../../../../../tests/ttnn/unit_tests/operations/transformers/test_paged_cache_mask.py).
No C++ gtest references these ops.

The `*_program_cache` variants are in the passing set, so the translated
`override_runtime_arguments` is exercised on real cache hits, not just built.

**The runtime drop from 815s to 581s is not a port claim.** It is a single unrepeated pair of runs on
a shared machine, and nothing here measures op performance; treat it as noise unless a profiler run
says otherwise.

### Anti-pattern self-audit

Denominator for every code sweep: **32** `.cpp` / `.hpp` files under the op directory (the `.md`
artifacts are excluded — they quote the legacy idioms deliberately).

| Check | Result |
|---|---|
| Buffer address in run args (`->address()`, `emplace_runtime_args`, `RTArgList`) | **0** |
| Magic CB indices / positional CTA vectors (`CBIndex::`) | **0** |
| `TensorAccessorArgs` surviving | **0** |
| `cb`-shaped names (`grep -rnE '[Cc][Bb]_\|_[Cc][Bb]\b\|\b[Cc][Bb]\b\|\bCB[A-Z]'`) | **0** |
| `.id` extraction on a `dfb::` handle | **0** |
| `allow_instance_multi_binding` | **0** |
| Varargs (`get_vararg`, `num_runtime_varargs`, …) | **0** |
| Legacy positional arg readers (`get_compile_time_arg_val`, `get_arg_val<`, `get_common_arg_val`) | **0** |
| `CircularBuffer` / `CBDescriptor` / `circular_buffer.h` | **0** |
| Ephemeral `.md` cited from code (22 changed/new code files scanned) | **0** |
| Forced-legality scaffolding in the diff | **0** |
| Files outside the op directory in the diff | **0** |

- **TT_FATAL census** — `diff` of per-file `TT_FATAL|TT_ASSERT|TT_THROW` counts, pre-port vs. working
  tree, over the op directory: **no output**. No guard moved or was lost, including the ones inside
  the translated overrides.
- **`opt_level`** — three source lines, one per compute-kernel construction site, each
  `KernelBuildOptLevel::O3`. The two fused sites sit inside the per-work-unit helper, which runs twice,
  so all **five** compute `KernelSpec`s the op builds carry an explicit `O3`. Legacy set none on a
  `ComputeConfigDescriptor`, which resolves to `O3`; Metal 2.0 would have defaulted to `O2`.
- **`hw_config`** — every DM kernel used a bare `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`
  (empty tag structs, [`program_descriptors.hpp:92-93`](../../../../../../tt_metal/api/tt-metalium/program_descriptors.hpp#L92-L93)),
  so all eight resolve to the stock reader / writer triples and each takes the matching
  `create_reader_datamovement_config` / `create_writer_datamovement_config`. No custom triple anywhere,
  so no exact-replication case and no `noc_mode` pairing to preserve. Compute: `ComputeGen1Config` with
  only `enable_32_bit_dest` set, because `ComputeConfigDescriptor`'s defaults
  (`HiFi4`, `dst_full_sync_en=false`, `bfp8_pack_precise=false`, `math_approx_mode=false`) map
  field-for-field onto `ComputeGen1Config`'s (`HiFi4`, `double_buffer_dest=true`,
  `bfp_pack_precision_mode=Approximate`, `sfpu_precision_mode=Precise`). Routing through the TTNN
  helper instead would have flipped three of them.
- **Conditional bindings** — for each of the eight conditional buffers and six conditional tensors, the
  host binding, the `compiler_options.defines` entry and the kernel-side `#ifdef` share one condition,
  and each define reaches every kernel that names the resource (both the reader and the writer for
  `USE_INDEX_TENSOR` / `IS_PAGED_CACHE`). No binding was made unconditional to dodge the gate.
- **Markdown links** — all 28 unique link targets in the plan and the report resolve from the artifact
  directory, and every cited line range is within its file.

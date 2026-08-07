# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/data_movement/sort`

## Outcome

**PORTED — all 3 factories on `MetalV2FactoryConcept`.**

- `SortProgramFactorySingleRowSingleCore` — ported, tests pass.
- `SortProgramFactoryCrossCoreDataExchange` — ported, tests pass.
- `SortProgramFactorySingleRowMultiCore` — ported, tests pass.

No legacy `ProgramDescriptor` code remains anywhere in the op directory.

**This took two passes.** The first pass landed the first two factories and capitulated on the third:
it was converted and built cleanly, but its programs could not be enqueued, because the shape it
requires (two `WorkUnitSpec`s over disjoint node sets, each binding its own dataflow buffers) hit a
framework out-of-bounds write in dispatch-command assembly, with no spec-level workaround. That
defect was reported from this port, fixed in `tt_metal`, and the second pass completed the factory
against the fix. The diagnosis, the fix that landed, and how the two differ are in
[Handoff points](#handoff-points).

## Provenance

- **Recipe docs (this port):** `9cafc69c9ce 2026-07-27 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

The two hashes differ while the subject line matches, so the audit ran against the same doc change
under a different commit id (a rebase between the audit and the port).

## Verification

Confirmed test set (agreed with the invoker before relying on it):
`tests/ttnn/unit_tests/operations/data_movement/test_sort.py` as the no-regression baseline.

| Run | Result |
|---|---|
| Pass 1, pre-port baseline (109 tests) | 108 passed, 1 failed |
| Pass 1, as shipped (109 tests) | 108 passed, 1 failed |
| **Pass 2, all three factories ported** (111 tests) | **111 passed, 0 failed** |

Pass 1 achieved exact parity with its baseline. Its single failure,
`test_sort_datatypes[shape=[32, 64]-dim=-1-descending=False-torch_value_dtype=torch.uint8-ttnn_value_dtype=DataType.UINT16-ttnn_index_dtype=DataType.UINT32]`,
was pre-existing and unrelated to the port; it has since been fixed on main by the UInt16-in-32-bit-DEST
work (#50215) and now passes. The suite also grew from 109 to 111 tests between the two passes.

The three tests that select the multi-core factory all pass in pass 2, having been the only failures
in the intermediate pass-1 state: `test_sort_long_tensor[shape=[1, 524288]-dim=-1-descending=False]`
and both parameterizations of `test_sort_multi_row_multi_core_no_deadlock`.

Anti-pattern self-audit over the whole op directory, now that nothing is exempt: zero hits for
`buffer()->address()`, `CircularBuffer`, `CBDescriptor`, `CBIndex`, `TensorAccessorArgs`,
`get_compile_time_arg_val`, `get_arg_val<`, `get_common_arg_val`, `get_vararg`, `AddrSelector`,
`ProgramDescriptor`, `KernelDescriptor`, `emplace_runtime_args`, `allow_instance_multi_binding`, and
`.id` extraction on a `dfb::` handle.

**Environment note (not a port finding).** The workspace as handed over could not build or run at
all: `build_Release/` and `python_env/` were both copies of a sibling checkout
(`git_2026_07_23_ops2_0_baseline`), so CMake refused to reconfigure and every test errored in the
JIT firmware build. Both were moved aside (`build_Release.stale-from-baseline-workspace`,
`python_env.stale-from-baseline-workspace`) and regenerated with `./build_metal.sh` and
`./create_venv.sh`. The stale copies are still on disk and can be deleted.

## TTNN ProgramFactory

### Concept realized

`MetalV2FactoryConcept` for all three factories, as the audit decided. In this tree the concept is
spelled `ProgramSpecFactoryConcept`
([operation_concepts.hpp:119](../../../../../../ttnn/api/ttnn/operation_concepts.hpp#L119-L121)); the
docs' `MetalV2FactoryConcept` name does not appear in code. No re-decision was needed.

`SortProgramFactoryCrossCoreDataExchange` shed its `WorkloadDescriptor` entirely. Its SPMD
replication over `tensor_coords.ranges()` is what the adapter now does for free, and its
`shared_ptr<Tensor>` lifetime hack for the physical-core lookup table became a one-line
`release_mesh_tensor()` into `ProgramArtifacts::op_owned_tensors`. The `tensor_coords` parameter
disappeared from the signature.

### Device-op-class edits

- Custom `compute_program_hash` deleted: **none** — the op never had one.
- Pybind entry points removed: **none** — `sort_nanobind.cpp` binds only `ttnn::sort`.

The device-operation class (`sort_device_operation.cpp` / `.hpp`) is untouched.

### Open items

- **Relaxation candidates:** none applied, none obviously warranted. The op's `TensorParameter`s all
  use strict matching and no kernel reads a runtime shape.
- **Op-owned tensors, first production exercise.** The cross-core factory is (per the recipe) among
  the first real users of `op_owned_tensors`. The path worked first try: `release_mesh_tensor()`
  into a `reserve(1)`'d vector, bind against `op_owned.back()`, `std::move` into the artifact. No
  friction with release ergonomics, binding identity, or cache-hit re-patching. Worth recording as a
  positive datapoint for conv2d / halo / pool.

## Handoff points

### 1. Segfault at dispatch assembly for a spec with two WorkUnitSpecs on disjoint node sets

**Team:** Metal 2.0 runtime / dispatch.

`SortProgramFactorySingleRowMultiCore` is the only factory in this op (and, as far as the framework's
own tests go, the only shape anywhere) that declares **two `WorkUnitSpec`s over disjoint node sets**:
a single-node coordinator work unit and a 63-node worker work unit, each binding its own set of
dataflow buffers.

- **Root cause (confirmed against a Debug build).** The per-core-range dataflow-buffer config payload
  is **sized by a count but indexed by a program-global id**:

  - Sizing, [dataflow_buffer.cpp:813-829](../../../../../../tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp#L813-L829):
    for each kernel group, sum `serialized_size()` over the buffers whose range intersects that
    group, then take the **max over groups**. That is a *count* of the buffers present on the
    busiest group.
  - Indexing, [dispatch.cpp:1386-1394](../../../../../../tt_metal/impl/program/dispatch.cpp#L1386-L1394):
    on WH/BH the write offset is `dfb->id * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * 4`, and
    `dfb->id` is assigned program-globally as `dataflow_buffers_.size()` at registration
    ([dataflow_buffer.cpp:886](../../../../../../tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp#L886)).

  When every node carries every buffer these agree. When node sets carry *different* buffer subsets
  they diverge, and any buffer whose global id is ≥ the busiest group's count is serialized past the
  end of the payload. For this factory: 8 buffers (ids 0-7), busiest group holds 6, so
  `dfb_size = 6 * 16 = 96` bytes (the runtime logs exactly `dfb size: 96`), while the worker buffers
  hold ids 2-7 and the id-7 record needs bytes [112, 128). The Debug build names it directly:

  > `TT_ASSERT @ tt_metal/impl/program/dispatch.cpp:1395: dfb_byte_offset + serialized.size() <= payload.size()`

  In Release that assert is compiled out, so `std::copy` corrupts the heap and the failure surfaces
  later as a segfault in the memcpy inside `add_dispatch_write_packed_large`, called from
  `BatchedTransferGenerator::assemble_commands` ← `assemble_device_commands` ←
  `ProgramImpl::generate_dispatch_commands` ← `MeshWorkloadImpl::generate_dispatch_commands` ←
  `EnqueueMeshWorkload`.

- **Reproducer:**
  `pytest "tests/ttnn/unit_tests/operations/data_movement/test_sort.py::test_sort_long_tensor" -k 524288`
  against a tree with the multi-core factory ported. Shape `[1, 524288]` is the smallest input that
  routes to that factory.

- **Why the port did not work around it.** The condition the buggy code needed was
  `max_id < max_per_group_count`, which for `N` buffers with ids `0..N-1` forces
  `max_per_group_count == N`: some kernel group had to cover *every* buffer. The coordinator and
  worker buffers sit on disjoint nodes, so no group did.

  A workaround did exist and the port deliberately did not take it. Binding a buffer is a host-side
  declaration and nothing checks that the kernel body touches it, so having the worker reader bind
  the coordinator's two buffers as a self-loop it never uses would have made the worker group cover
  all 8. That passes the endpoint rules: producers and consumers would both be
  `{coordinator, reader}`, satisfying the self-loop set-equality rule
  ([program_spec.cpp:1436-1445](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L1436-L1445)),
  and both are data-movement kernels, satisfying the same-kind rule. It was rejected because it
  declares endpoints that do not exist and pays SRAM on 63 nodes for buffers they never touch,
  purely to make a framework arithmetic bug line up. The recipe's guidance is to stop at that point
  and report, which is what happened.

  Note the asymmetry, which is why the first pass initially over-claimed impossibility: the reverse
  direction genuinely is blocked. Extending the *coordinator* onto the worker buffers fails, because
  for `input_tensor_output` the producer is compute and the consumer is the writer, so adding the
  coordinator to both sides gives `{compute, coordinator}` against `{writer, coordinator}`, which are
  not equal.

- **Exposure is narrower than "node-dependent buffer sets".** Uneven sets alone are harmless. The
  triggering condition is that **the program declares more distinct buffers than any single kernel
  group uses**, so no group covers the whole id range. Legacy sort satisfies the safe case by
  accident: in TILE it declares 6 CBs (c_0-c_5) on `all_core_set`, the worker group uses all 6, and
  the highest id in play is 5, which fits a 6-slot region even though the coordinator only touches
  c_0/c_1.

- **What pushed this port over the line.** Legacy *shared* c_0 and c_1 between the two roles: the
  coordinator used them as DRAM-to-DRAM staging, the workers used the same indices as sort input.
  That sharing has no Metal 2.0 expression. The coordinator fills and drains its staging buffer on
  its own node, so it holds both roles there, and once any kernel is on both sides of a buffer the
  producer and consumer kernel sets must be identical
  ([program_spec.cpp:1436-1445](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L1436-L1445)).
  Sharing would give `{coordinator, reader}` against `{coordinator, compute}`, which are not equal.
  Dropping the coordinator to a single role does not help either: its node would then have a producer
  and no consumer, which the per-node census rejects
  ([program_spec.cpp:1370-1391](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L1370-L1391)).

  The same-kind-per-role rule
  ([program_spec.cpp:1289-1295](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L1289-L1295))
  is a second obstacle but not the decisive one, because it is skipped under
  `allow_instance_multi_binding` while the set-equality rule above is not.

  Splitting each shared buffer into a coordinator-scoped and a worker-scoped spec takes the program
  from 6 buffers to 8 while the busiest group still used 6, which was exactly the unsafe case for the
  old sizing. The same arithmetic held in ROW_MAJOR: 10 legacy indices become 12 specs against a
  busiest group of 10.

- **Suggested fix, with a working reference implementation already in the tree.** This is a regression
  against the legacy circular-buffer path, which computes the same quantity correctly:
  `finalize_cbs` ([dispatch.cpp:307-320](../../../../../../tt_metal/impl/program/dispatch.cpp#L307-L320))
  tracks a per-kernel-group *mask* of the slot indices in use, takes the position of the highest set
  bit, and sizes the region from that. `finalize_dfbs` replaced that with a per-group *count* while
  leaving the id-based addressing in place. Restoring the max-index basis is close to a copy of
  `finalize_cbs`.

  Note that the Gen2 branch of the same write loop
  ([dispatch.cpp:1383-1391](../../../../../../tt_metal/impl/program/dispatch.cpp#L1383-L1391)) lays
  buffers out sequentially with a running offset, which *is* consistent with a count-based size. The
  count-based sizing looks correct for that layout and was applied to the Gen1 branch too, where the
  offset is `id * slot_size`.

- **Independently, make the bounds check unconditional.** Promote the two `TT_ASSERT`s at
  [dispatch.cpp:1394-1395](../../../../../../tt_metal/impl/program/dispatch.cpp#L1394-L1395), and the
  transfer-overlap asserts in `BatchedTransferGenerator::assemble_commands`, to `TT_FATAL`.
  `TT_ASSERT` compiles to `(void)(condition)` outside a Debug build
  ([assert.hpp:163-168](../../../../../../tt_stl/tt_stl/assert.hpp#L163-L168)), so in the build
  everyone uses, this defect is a silent out-of-bounds `std::copy` followed by a segfault in
  unrelated code. It cost most of the debugging time on this port; a Debug build answered it in one
  run. These checks run once per buffer at program setup, so the cost of leaving them in is nil.

- **What pass 1 did instead.** Reverted `SortProgramFactorySingleRowMultiCore` and its four kernels
  to the legacy `ProgramDescriptor` concept, so the op kept working on every shape while the
  framework was fixed.

- **RESOLVED. The fix landed on main and pass 2 completed the factory against it.** The fix went
  further than this report suggested, and better. Rather than only correcting the sizing basis, it
  introduced a per-buffer `device_slot` distinct from the program-global id, assigned as the lowest
  slot not already taken by a buffer whose node range intersects
  ([dataflow_buffer.cpp:1716-1736](../../../../../../tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp#L1716-L1736)).
  Buffers on disjoint nodes now reuse low slots, so a node's config table is sized by its own buffer
  count rather than by how many buffers exist elsewhere in the program. The region sizing was
  switched to `max_slot_plus_one` to match
  ([dataflow_buffer.cpp:1656-1687](../../../../../../tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp#L1656-L1687)),
  and the write side now indexes by `device_slot`
  ([dispatch.cpp:1386-1394](../../../../../../tt_metal/impl/program/dispatch.cpp#L1386-L1394)).

  That removes the class of failure entirely rather than just this instance: the earlier suggestion
  (size by highest id) would have made the table correct but still as large as the highest id in the
  whole program, so a program with many disjoint buffer sets would have paid for all of them on every
  node. Slot reuse avoids that. Applied to this factory, the coordinator's two buffers and the
  workers' six now both occupy slots starting at 0, and all 111 tests pass.

### 2. The lookup-table buffer's endpoint census has no expressible multi-binding form

**Team:** Metal 2.0 docs / patterns catalog (see also [Friction](#friction)).

The audit and brief both instructed the porter to set
`advanced_options.allow_instance_multi_binding = true` on `CrossCoreDataExchange`'s
`physical_core_lookup_table` buffer (legacy c_10), on the grounds that two kernels drive its producer
cursor. That instruction cannot be carried out: the buffer has **two producer-role touchers and zero
consumers**, so satisfying the "≥1 producer and ≥1 consumer" rule requires one kernel to take both
roles, and the validator then rejects the result:

> DFB 'physical_core_lookup_table' is self-looped (some kernel appears as both producer and
> consumer), but the set of producer KernelSpecs differs from the set of consumer KernelSpecs. When
> a DFB is self-looped, every same-side binding must come from a self-loop participant.

The port resolves it as the catalog's stacking guard prescribes: recount, assign **1P+1C** (reader
PRODUCER, writer CONSUMER), no flag. That is behaviour-identical on Gen1 — the buffer is a
reader-private lookup window, the labels drive no machinery either kernel invokes, and neither
kernel's transfers change.

**The Quasar debt does not disappear, it just becomes untracked.** On Gen2 a CONSUMER-labelled kernel
cannot `push_back`, so the writer's vestigial
[`writer_cross_core_data_exchange.cpp:98`](device/kernels/dataflow/writer_cross_core_data_exchange.cpp#L98)
`push_back` has to go. The ops team already noted (audit, Misc anomalies) that removing it is a
functional change out of port scope; doing so would make the reader a clean self-loop and close this
out.

## Successes

- **[Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  and its "promote a CTA gate to a define" note were exactly right, and load-bearing.** This op's
  whole TILE / ROW_MAJOR split was a single `is_row_major` compile-time arg gating `if constexpr`
  blocks in all 10 kernels. Every one of those blocks touches buffers that only exist in one
  configuration, so a mechanical CTA-to-named-arg translation would have failed to compile on
  `dfb::rm_input` and friends. The pattern told me up front to promote the gate to
  `KernelSpec::compiler_options.defines` and `#ifdef` both the alias and every use. That is the
  single largest structural change in the kernel diff and it was prescribed, not discovered.
  The catch that mattered most:
  [`sort_cross_core_data_exchange.cpp:82`](device/kernels/compute/sort_cross_core_data_exchange.cpp#L69-L74)
  had a plain runtime ternary
  (`compute_kernel_hw_startup(is_row_major ? rm_input_cb_id : input_tensor_cb_id, ...)`), which
  resolves both operands regardless of the condition.

- **The recipe's insistence on re-deriving endpoint dispositions from the census rather than
  transcribing the brief** paid for itself twice: once on the multi-core coordinator buffers (below)
  and once on the lookup table (Handoff 2), where following the brief verbatim would have produced a
  spec the validator rejects.

- **[Anti-pattern: Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  fired correctly as a non-event.** Sort has no per-group CTA specialization at all, so "Preserved
  Multiplicity: none" was the honest answer; the section made me check rather than assume.

## Friction

### Gaps

- **The census-to-disposition table has no entry for "≥2 producer-role touchers and zero
  consumers".** The
  [endpoint-assignment procedure](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)
  says "≥2 kernels locked to the same FIFO role → multi-binding", and the stacking guard says
  self-loop and multi-binding are mutually exclusive and to "recount and assign 1P+1C". When a buffer
  has zero consumers, those two rules are in direct conflict: the multi-binding assignment the first
  rule points at is *only* reachable by self-looping one of the touchers, which the second rule (and
  the validator) forbid. The right answer turns out to be the second rule, but only the validator
  says so. **Suggested fix:** add a row for zero-consumer buffers stating that the consumer role goes
  to a toucher regardless of its FIFO ops, and that the flag is not available in that shape.
  Sites: [sort_program_factory.cpp:800-830](device/sort_program_factory.cpp#L800-L830).

- **The `use<AddrSelector::WRITE_PTR>` wrapper does not simply "drop".** The
  [kernel-side whitelist](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#kernel-side-whitelist)
  says the pointer-selection wrappers drop "because a bare `DataflowBuffer` used as a NoC
  source/destination is already pointer-sourced", and illustrates it with `READ_PTR`. That holds for
  `READ_PTR` only: as a NoC **source** a bare DFB resolves to `get_read_ptr()`
  ([dataflow_buffer.h:387](../../../../../../tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L387)), so
  dropping the wrapper at a `WRITE_PTR` source site silently transmits a different slot. This op has
  two such sites, both in the multi-core coordinator. The port preserves the exact address by feeding
  the whitelist-sanctioned public peek into a `CoreLocalMem<uint32_t>` source, the same idiom the
  cross-core exchange already used for local-to-peer writes.
  Sites: [coordinator_single_row_multi_core.cpp:89](device/kernels/dataflow/coordinator_single_row_multi_core.cpp#L86-L96)
  and [:126](device/kernels/dataflow/coordinator_single_row_multi_core.cpp#L126-L135).
  **Suggested fix:** state the asymmetry explicitly, and name `CoreLocalMem<T>(dfb.get_write_ptr())`
  as the sanctioned replacement for a `WRITE_PTR` source.

- **Rule 7 ("query DFB metadata off the object") has no answer when the kernel does not bind that
  buffer in the active configuration.** The cross-core reader needs the TILE-format value and index
  tile sizes for the peer exchange, which runs in both configurations, but in ROW_MAJOR it binds no
  buffer paged at those sizes: the legacy kernel read them via `get_tile_size(cb_id)` off buffers
  that were *allocated but untouched* in that configuration. Metal 2.0 has no equivalent, since an
  unbound buffer has no `dfb::` handle. The port passes both as named compile-time args
  ([reader_cross_core_data_exchange.cpp:31-32](device/kernels/dataflow/reader_cross_core_data_exchange.cpp#L26-L32)).
  This is a legitimate named scalar, not recreated legacy plumbing, but the recipe should say so:
  **when the metadata's buffer is not bound on the active path, a named CTA is the fallback.**
  This one was a near-miss worth flagging: my first attempt sourced the sizes from the *intermediate*
  buffers, which the legacy factory pages at `index_tensor_tile_size` rather than the value tile
  size, and would have silently changed the peer transfer length for mixed-width dtypes.

- **`unpack_modes` needs configuration-gating that the recipe does not mention.** The legacy
  `unpack_to_dest_mode` vector is sized `NUM_CIRCULAR_BUFFERS` and indexed by CB id, so it happily
  carries entries for buffers the current configuration never allocates. All three factories do this.
  Metal 2.0's validator rejects a key naming a buffer the kernel does not bind, so every entry has to
  be gated on the same condition as its binding. The recipe's
  [Hardware configuration](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compute-kernels)
  section covers reindexing and value translation but not this. Behaviour-neutral (a legacy entry for
  an unallocated CB was inert), but it is a compile-clean, validator-caught trap.
  Sites: [sort_program_factory.cpp:414-425](device/sort_program_factory.cpp#L414-L425),
  [:1150-1163](device/sort_program_factory.cpp#L1150-L1163),
  [:1743-1753](device/sort_program_factory.cpp#L1743-L1753).

- **Two work units on disjoint node sets are untested on hardware.** Every hardware test under
  `tests/tt_metal/tt_metal/api/metal2_host_api/` uses a single `WorkUnitSpec` on a single node
  (`MakeMinimalWorkUnit`). The first op-level spec to use two turned out not to dispatch (Handoff 1).
  A framework hardware test for this shape would have caught it before the port.

### Confusion

- **"Narrow the over-scoped core range" does not translate literally.** The brief asks the porter to
  narrow the multi-core factory's c_2-c_5 from `all_core_set` to the worker range. Metal 2.0 has no
  core range on a `DataflowBufferSpec` at all, so the instruction only makes sense once restated as
  "do not bind the buffer on the coordinator kernel". For c_0/c_1 in the TILE configuration it is not
  a tidy-up but mandatory: one spec bound by the coordinator (a DM kernel, both ends) and by the
  workers (reader producing, *compute* consuming) puts a DM and a compute kernel on the same consumer
  endpoint, which the validator rejects with "All KernelSpecs bound to the same DFB role must be of
  the same kind"
  ([program_spec.cpp:1289-1295](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L1289-L1295)).
  The port splits them into coordinator-scoped and worker-scoped specs, which is the same treatment
  the brief already prescribes for the aliased c_6/c_7. **Suggested fix:** phrase the disposition as
  "drop the binding on the kernel that does not touch it, and split the spec if the remaining
  touchers are on disjoint node sets", and note the same-kind rule as the forcing constraint.

- **The concept name in the docs does not exist in code.** The recipe and the TTNN integration doc
  both say `MetalV2FactoryConcept`; the code has `ProgramSpecFactoryConcept` (plus a
  `CustomProgramSpecFactoryConcept` sibling). Harmless once you find it, but it costs a grep at the
  exact moment you are trying to confirm you are on the right concept.

## Open items for downstream

- **Cross-op kernel touches:** none. All 10 kernel sources and all 3 shared headers are in the op's
  own directory; no other op reads them.

- **`get_core_physical_coordinates` constructs a second `DataflowBuffer` for a buffer the reader
  already holds an object for**
  ([cross_core_data_exchange_common.hpp:146](device/kernels/dataflow/cross_core_data_exchange_common.hpp#L141-L147)).
  The catalog warns against two objects aliasing one FIFO because it breaks the object-to-buffer
  identity device-side debug tooling relies on. This is pre-existing (the legacy helper did the same
  through a CB index) and fixing it means changing the helper's signature to take a
  `DataflowBuffer&`, which is a refactor rather than a syntax swap. Left as-is; worth doing when
  someone is next in this file.

- **Behaviourally-inert attributes still in the hash.** `stable` is asserted false at the `ttnn::sort`
  entry yet is still threaded to the compute kernel as a named arg and left in the hashed
  `SortParams`; `dim` is constrained to the last axis so its two legal values hash distinctly for
  identical behaviour. Both carried over verbatim. Cleaning them up would narrow the program cache
  key.

- **Vestigial compile-time args in the cross-core writer.** `number_of_cores_used` survives as a named
  arg with its "unused - for future improvements" comment. Two others could not survive the port and
  were dropped, both behaviour-neutral: arg 4 `value_tensor_peer_cb_index` (a buffer index for a
  buffer the writer never touches — a buffer index cannot be a named arg in Metal 2.0, and binding it
  would have made that buffer a spurious three-toucher) and arg 10 (a semaphore id read by nothing,
  which the legacy factory passed only to keep positional slots aligned; there are no positional
  slots to align any more).

- **The multi-core factory has thin coverage.** Only three tests select it:
  `test_sort_long_tensor[shape=[1, 524288]-dim=-1-descending=False]` and both
  `test_sort_multi_row_multi_core_no_deadlock` parameterizations. All three pass, and they are the
  entire safety net for a factory with a coordinator, a semaphore-driven schedule, and two work
  units. Nothing here was ever disabled; during pass 1 these three were deselected on the command
  line for a single diagnostic run, which is not a change to any file.

- **Comment terminology.** Comments this port authored use "SRAM" per the workspace convention;
  inherited comments that say "L1" were preserved verbatim rather than reworded, on the grounds that
  rewriting inherited documentation is not port work. A sweep of the op's older comments would
  normalize them.

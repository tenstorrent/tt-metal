# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/reduction/sampling`

## Outcome

**`PORTED`.** The op's only factory, `SamplingProgramFactory`, is on `ProgramSpecFactoryConcept`, and all three
kernels are converted. Nothing is left for a later pass. Verified on a Wormhole n150 (`WORMHOLE_B0`).

Test results on the invoker-confirmed baseline:

| Test | Result |
|---|---|
| `tests/ttnn/unit_tests/operations/reduce/test_sampling.py` | **26 passed** |
| `tests/ttnn/unit_tests/operations/reduce/test_manual_seed.py` | **8 passed** |
| `tests/ttnn/nightly/.../test_reduction_ops.py::test_sampling` | **4 passed** |
| `tests/ttnn/docs_examples/test_reduction_examples.py::test_sampling` | **1 passed** |
| `tests/ttnn/unit_tests/operations/reduce/test_tiebreak_input_adjust.py` | 3 passed, **15 pre-existing failures (unchanged)** |

The 15 `test_tiebreak_input_adjust.py` failures are **not** a port regression. They were confirmed against the
pre-port code by stashing the port, rebuilding, and re-running that file: the pre-port run produces the *same* 15
failures with the *same* test ids and the same 3 passes. Ten of them
(`test_tiebreak_boosts_lowest_global_index_for_greedy_users`) never call `ttnn.sampling` at all — they exercise
`TTSampling._adjust_values_for_tiebreak` (`models/common/sampling/tt_sampling.py:616`), which is built from
`ttnn.max` / `eq` / `lt` / `abs` / `multiply` / `add` / `min`. The failure is that the tie-break boost is not applied
at all (the adjusted row comes back bit-identical to the input), so `argmax` lands on position 0 instead of the
lowest-global-index tied maximum. The five `test_sampling_picks_lowest_global_index_after_adjust` cases do call
`ttnn.sampling`, but they feed it the output of that same broken adjustment, so they fail for the upstream reason.
Routed to the model owners under *Open items* below.

## Provenance

- **Recipe docs (this port):** `ccf3df7c4ab 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `ccf3df7c4ab 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, as the audit chose. No revision, nothing surfaced back to the invoker.
`SamplingProgramFactory::create_descriptor` became `create_program_artifacts` returning
`ttnn::device_operation::ProgramArtifacts`; `op_owned_tensors` stays defaulted-empty because the factory allocates
nothing beyond the framework-allocated output.

One shape worth recording, because it is unusually clean: **`ProgramRunArgs::kernel_run_args` is empty.** All six
legacy runtime args were tensor base addresses, so all six became `TensorBinding`s and no kernel is left with a
runtime or common runtime argument. `ProgramRunArgs` carries only `tensor_args`. That is legal — a `KernelRunArgs`
entry is required only for kernels that *have* runtime args
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/program_run_args.hpp:89-91`) — and the program-cache
callback tests confirm the cache-hit tensor-rebinding path works with nothing else in the run args.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — the op never had one.
- **Pybind entry points removed:** none. `sampling_nanobind.cpp` binds only the user-facing `ttnn::sampling`; it
  never exposed `create_descriptor`, so no pybind line had to go and there is no user-visible API surface change.
- The only edit outside the factory body was dropping the now-unused `#include <tt-metalium/program_descriptors.hpp>`
  from `device/sampling_device_operation.hpp`, part of the "no legacy CB-API reference survives" sweep.
  `ttnn::CoreRangeSet`, the one type that header might have been supplying, comes from `ttnn/types.hpp:47`.

### Open items

- **No relaxation candidates identified.** Strict `TensorSpec` matching held on all six parameters with no friction.
  `k`, `p` and `temp` are `[num_users]` row-major tensors whose shape is baked into the writer's `num_cores` CTA and
  into the `k`/`p`/`temp` buffer sizes, so a shape relaxation there would be wrong, not merely unapplied.
- **A Style-B compute config has no generation-agnostic helper.** See the first Friction/Gap entry below.

## Handoff points

### 1. `generate_bcast_unary_scalar` takes a `CircularBuffer` by value — the one `CircularBuffer` reference the port cannot remove

*Tagged: kernel-lib / shared-kernel-pool API, cross-team decision. This is the ⭐ shape the audit flagged; the port
did not attempt to resolve it.*

- **Donor:** `generate_bcast_unary_scalar(CircularBuffer cb, uint32_t scalar)` at
  `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp:44`.
- **Call site:** `device/kernels/dataflow/writer_interleaved.cpp:108`, now
  `generate_bcast_unary_scalar(CircularBuffer(dfb::temp), temp_packed)`.
- **What happened:** the call site compiles and runs, exactly as the brief predicted — `dfb::temp`'s constexpr
  `operator uint32_t()` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:55`) satisfies the donor's
  `explicit CircularBuffer(uint32_t)` constructor. So this was not an assumption-violation stop.
- **Why it still needs someone:** the kernel-side whitelist's rule 1 says the CircularBuffer-to-DataflowBuffer
  transition is *total* — a post-port grep for `CircularBuffer` across the op directory should return zero hits in
  code. It returns exactly one, this line, and the port cannot remove it without changing a donor signature shared
  by nine other kernels across `normalization/softmax`, `transformer/sdpa` and `data_movement/bcast`. The two rules
  (total CB removal, and don't change a broadly-shared donor) genuinely collide on this one call, and the collision
  will recur in every op that uses this header. The fix — a `DataflowBuffer` overload beside the `CircularBuffer`
  one, or a templated parameter — belongs to whoever owns that shared-kernel pool.
- **Also worth their attention:** the same header's `generate_bcast_col_scalar` and `generate_bcast_row_scalar` have
  the identical signature shape, so a fix should cover all three.

### 2. `generate_mask` fills from `get_read_ptr()` after `reserve_back` — safe today, fragile by construction

*Tagged: `sdpa_decode` kernel owners. Observation, not a blocker; the audit recorded it as Misc anomaly 7 and the
port confirmed it on the Metal 2.0 side.*

`generate_mask` (`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:262-263`)
takes its fill base from `cb_mask.get_read_ptr()` immediately after `cb_mask.reserve_back(...)`, where
`get_write_ptr()` is the conventional pairing. The writer binds `topk_mask` as **PRODUCER only**, so it is worth
recording *why* the read pointer is still valid under Metal 2.0 rather than leaving the next porter to wonder: on
Gen1 both DM RISCs initialise **both** FIFO pointers for every buffer in their slot mask
(`tt_metal/hw/firmware/src/tt-1xx/brisc.cc:503` and `ncrisc.cc:157` instantiate
`setup_local_cb_read_write_interfaces<true, true, …>`), and a buffer's slot mask is
`producer_risc_mask | consumer_risc_mask` (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:1714`). So a
PRODUCER-only binding still yields a valid read pointer, and because nothing has been popped it equals the write
pointer. The pairing breaks for any future consumer that pops.

## Successes

- **[Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb),
  plus the entry's explicit "a DM self-loop is legal on Gen1" note.** Three of this op's buffers are writer-only
  (`output` at `writer_interleaved.cpp:129`/`:223`, `k` at `:72-80`, `p` at `:83-91`) and none presents a
  producer/consumer pair. Without that entry the natural guesses are both wrong: bind only the role the kernel
  visibly uses (the validator then rejects the buffer for having no consumer), or reach for the multi-binding flag.
  The entry's hard gate — *count distinct touchers first; one toucher means self-loop* — landed all three correctly
  the first time, and its Gen1-vs-Gen2 note pre-answered the doubt about self-looping a data-movement kernel. The
  validator agrees: `tt_metal/impl/metal2_host_api/program_spec.cpp:1417-1423` rejects a DM self-loop only on Gen2.

- **[Two-toucher DFB → assign 1P+1C](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)'s
  *Re-derive, don't transcribe* step, and its **Constraint** paragraph.** Re-running the census independently
  reproduced the brief's 18 dispositions exactly, so there was no disagreement to report — but the Constraint
  paragraph did real work: this op instantiates one writer per core over **disjoint single-core** ranges, which
  reads like the dual-instance work-split shape at a glance. The paragraph draws the line explicitly (disjoint node
  sets means each node sees one instance, so each node's buffer is an ordinary 1:1), which is what kept
  `allow_instance_multi_binding` out of this diff.

- **[Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options),
  and specifically its "answering *did it set one?* is the part that goes wrong" warning.** `grep -n opt_level` on
  the legacy factory returns nothing, which reads as "no setting, nothing to carry". The section's insistence that
  an absent `KernelDescriptor::opt_level` still resolves to **O3** on a `ComputeConfigDescriptor` is the only reason
  `device/sampling_program_factory.cpp:311` sets `KernelBuildOptLevel::O3` explicitly. Nothing would have caught
  the omission: the build stays green, and all 39 tests pass either way.

- **[Hardware configuration → Compute kernels](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compute-kernels)'s
  Style A / Style B split.** This op sets a Metal `ComputeConfigDescriptor` directly, so it is Style B. The section's
  "don't reroute these through the TTNN helper to save typing" warning fired correctly: `to_compute_hardware_config`
  was the obvious-looking move, and taking it would have silently flipped `sfpu_precision_mode` and
  `bfp_pack_precision_mode` toward the helper's high-performance defaults on a kernel whose legacy config left them
  at the high-precision ones.

## Friction

### Gaps

- **The Gen1-only compute config narrows an op that deliberately supports more than Gen1, and there is no
  generation-agnostic helper for a Style-B config.** The recipe's
  [Gen2 is out of scope](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#gen2-is-out-of-scope)
  is categorical ("Build only the Gen1 config, and add no `if (arch == QUASAR)` branch of your own"), and the port
  followed it: `device/sampling_program_factory.cpp:298-300` builds a `ComputeGen1Config`. But this op is not
  Gen1-only by construction — its host and kernels carry an architecture-gated 32-bit-index path written for Quasar
  (`use_32bit_index` / `stable_sort`, `:123` and `:130`, with matching validation in
  `device/sampling_device_operation.cpp:29-34`), and the legacy `ComputeConfigDescriptor` was generation-agnostic. So
  the port converts an op whose compute config worked on any architecture into one that will fail spec validation on
  Gen2. The two DM kernels do *not* have this problem, because `ttnn::create_reader_datamovement_config(arch)` /
  `create_writer_datamovement_config(arch)` pick the generation for you. The asymmetry is the gap: the recipe notes
  that "in the *default* DM and compute cases the arch-agnostic helpers already emit a correct Gen2 branch for free",
  but a Style-B compute config is not a default case and has no such helper — `to_compute_hardware_config` requires a
  TTNN `ComputeKernelConfig`, which a Style-B op does not have. A `to_compute_hardware_config(arch,
  ComputeGen1Config)`-shaped overload, or an explicit note that Style-B ops knowingly give up non-Gen1 support, would
  close it. Recorded rather than worked around.

- **Neither the recipe nor the whitelist says whether a DFB's binding roles narrow the per-RISC FIFO-pointer
  initialisation.** This is the fact Handoff point 2 turns on, and it is a correctness question, not a style one: if
  a PRODUCER-only binding left `fifo_rd_ptr` uninitialised, `generate_mask`'s `get_read_ptr()` on the
  PRODUCER-only-bound `topk_mask` would fill at a garbage address — a silent wrong answer or a hang, with nothing in
  the diff to point at. The whitelist's [Access control](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md#access-control-get_ptr-vs-evil_set)
  table says `get_read_ptr` / `get_write_ptr` are public to "any kernel that needs the current FIFO cursor", with no
  mention of the binding role, and the DFB header's endpoint invariant is about instance counts, not pointer
  initialisation. Answering it meant reading the Gen1 firmware. One line in the whitelist — *on Gen1 both DM RISCs
  initialise both cursors for every bound DFB, so either peek is valid on either role* — would save that detour, and
  the peek-versus-role question will recur on any op whose donor peeks the "wrong" pointer.

- **Dropping the `use<AddrSelector::WRITE_PTR>` wrapper silently swaps which cursor a NoC source reads.** The
  whitelist says a bare `DataflowBuffer` used as a NoC source or destination is "already pointer-sourced", so the
  wrapper drops. What it does not say is that the bare form is not symmetric: `noc_traits_t<DataflowBuffer>` resolves
  a **source** to `get_read_ptr()` and a **destination** to `get_write_ptr()`
  (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:383-395`). The line at `writer_interleaved.cpp:223` used
  `use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out)` as a **source**, so the mechanical drop changes the cursor
  it reads. It is safe here only because `output` never runs a FIFO operation, leaving both cursors at the buffer
  base — but that is a per-op argument the porter has to construct, and on a buffer that *had* been pushed or popped
  the same edit would be a silent wrong-address bug. The whitelist's wrapper-drop note should carry the caveat: the
  drop is behaviour-preserving only when the wrapper selected the cursor the bare form would have picked anyway, or
  when the two coincide.

### Confusion

- **"Preserve the multiplicity" and "one KernelSpec per legacy KernelDescriptor" pull opposite ways when the legacy
  descriptors are identical.** This factory pushes one compute `KernelDescriptor` per core, up to 32 of them, and
  they are byte-identical apart from `core_ranges` — `compute_args` holds no per-core value. The
  [Planned Spec Shape](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#planned-spec-shape)
  default ("one per legacy `KernelDescriptor`") and the brief's instruction not to consolidate the per-core instances
  both read as "emit 32 compute specs", while the anti-pattern the rule protects against — losing a per-group CTA —
  cannot apply when there is no per-group CTA to lose. The port emits **one** compute `KernelSpec` over the union of
  the cores (`device/sampling_program_factory.cpp:306-311`) and 32 writer specs, since only the writer carries a
  differing CTA (`core_id`). That is behaviour-identical and is the shape Metal 2.0's derived placement is built for,
  but it took re-reading the anti-pattern's *rationale* to be confident the rule's letter wasn't being broken. A
  sentence in Planned Spec Shape — *legacy descriptors that differ only in `core_ranges` collapse to one KernelSpec,
  since placement is derived; preserve multiplicity only where a CTA differs* — would make this a lookup instead of a
  judgment call.

- **`WorkUnitSpec`s may not overlap in target nodes, which forces the work-unit granularity, and the recipe's worked
  examples do not show that case.** Because the writer is per-node, and work units may not overlap
  (`tt_metal/impl/metal2_host_api/program_spec.cpp:1694-1707`), the only legal shape is **one single-node work unit
  per core** with the reader and the compute kernel as members of all of them
  (`device/sampling_program_factory.cpp:540-547`). Every `WorkUnitSpec` example in the migration guide is either one
  work unit over a grid or a small fixed set (inner/halo), and the accumulation reference port has one per core
  group, so the "N work units, N = number of cores, two kernels in all of them" shape had to be derived from the
  non-overlap rule rather than recognised. The migration guide's *A kernel belonging to multiple work units* example
  is the closest and does state that a kernel's node set is the union of its work units — pointing that example at
  the per-node case would have made it land faster.

## Open items for downstream

- **Shared kernel touches: none.** All three kernel sources belong to this op, no other op or test binds them, and no
  `_metal2` fork existed or was created. No sunset list, no coordination signal for a sibling port.

- **`test_tiebreak_input_adjust.py`: 15 pre-existing failures for the model owners.**
  `TTSampling._adjust_values_for_tiebreak` (`models/common/sampling/tt_sampling.py:616`) applies no boost at all —
  the returned row is bit-identical to its input, at every magnitude in the test's `MAGNITUDES` list and on both the
  restricted sub-core grid and the full grid. Since the method is itself documented as a workaround for
  tenstorrent/tt-metal#33492 (unreliable stable top-k), and this op's `stable_sort` path is enabled on WH/BH, the
  first question for its owners is probably whether the workaround is still needed at all rather than why the boost
  is being lost. Independent of this port in both directions: the method never calls `ttnn.sampling`, and the
  failures reproduce identically on the pre-port build.

- **Dead writer compile-time arg, now named.** `out_stick_size` is supplied to every writer spec
  (`device/sampling_program_factory.cpp:518`) and read by no kernel. Kept, because dropping it is a behaviour-neutral
  cleanup that belongs to the ops team (the audit's Misc anomaly 2), and because keeping it makes the port's argument
  set a faithful translation of the legacy one. Whoever removes it should delete the host entry and the
  `writer_interleaved.cpp:41` comment together.

- **The op's other latent issues stay where the audit left them.** `Wt == 1` (`W == 32`) hangs the compute kernel's
  local-sort loop; the `temp` buffer is sized as a 2-bytes-per-core staging area yet also serves as
  `mul_tiles_bcast_scalar`'s operand; `logWt` goes through `std::log2` on a value validation already guarantees is a
  power of two. All three are unchanged by this port — the `temp` sizing in particular ports across verbatim, since
  `entry_size` is the legacy `page_size` and `num_entries` is 1. They are recorded in `METAL2_PREPORT_AUDIT.md` under
  *Misc anomalies* (items 1, 5, 6) and route to the ops team.

- **Test coverage note.** No C++ gtest exercises this op, and there is no
  `tests/sweep_framework/sweeps/reduction/sampling/`. The Python coverage is good on the axes that matter to a port
  (program-cache callback, `sub_core_grids`, sub-32-user grids, both index dtypes, preallocated output), so this is a
  note rather than a gap the port should have filled.

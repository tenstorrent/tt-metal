# Port Report — reshard (NdReshardCopyPagesFactory)

> NOTE (second pass — remaining factories): this report originally covered only
> `NdReshardCopyPagesFactory`. A follow-up pass ported `NdReshardCopyLocalShardFactory`
> and assessed the three legacy-reshard factories. See **§Per-factory STATUS (all variants)**
> at the bottom for the authoritative per-factory result.

## TTNN ProgramFactory

- **Concept realized**: `MetalV2FactoryConcept` (`ProgramSpecFactoryConcept`). The factory now exposes
  `create_program_spec(const ReshardParams&, const ReshardInputs&, Tensor&) -> ttnn::device_operation::ProgramArtifacts`,
  replacing the legacy `create_descriptor` (`ProgramDescriptor`).
- **Custom `compute_program_hash` deletion**: none — `ReshardDeviceOperation` already uses the default reflection-based hash.
- **Pybind entry points removed**: none — `reshard_nanobind.cpp` exposes only the user-facing op, no `create_descriptor`/`create_program_descriptor` hook.
- **Device-op-class edits**: none. `select_program_factory` already returns `NdReshardCopyPagesFactory{}`; the
  `program_factory_t` variant entry is unchanged. The framework's `ProgramSpecMeshWorkloadFactoryAdapter` routes the
  factory automatically once it satisfies `ProgramSpecFactoryConcept` (no longer `ProgramDescriptorFactoryConcept`).
- **Scope**: only the `NdReshardCopyPagesFactory` variant was ported. The other 7 variants in `program_factory_t`
  (SameWidth<t/f>, SameHeight<t/f>, Generic, CopyLocalShard<t/f>) remain on the legacy concept; the op continues to
  build and dispatch per-factory.

## Handoff points

none. No `sem::`/`ta::` out-of-op call site, no kernel-lib gap, no removed pybind surface, no framework gap bit during the port.

## Successes

- **Local-DFB rule (recipe §"Local-DFB rule" / matmul exemplar)** steered the WorkUnitSpec design directly: the reader
  (PRODUCER of `cb_in0`) and writer (CONSUMER) share one `WorkUnitSpec` over the full grid, satisfying the producer+consumer
  same-node invariant. No self-loop needed — this is a real reader→writer FIFO, not a fake CB.
- **Kernel-side whitelist rule 3** collapsed the legacy 6-line `TensorAccessorArgs<0,0>()` + offset-chaining + base-addr-RTA
  dance in each kernel to a single `TensorAccessor(ta::input)` / `TensorAccessor(ta::output)` line. The legacy CRTA carrying
  the buffer base address (`emplace_common_runtime_args({input_buffer})`) dropped entirely, replaced by the `TensorBinding`.

## Friction

- **Gap (minor)**: the worked-example kernels named in `/tmp/PORT_INSTRUCTIONS.md`
  (`writer_bmm_8bank_interleaved_start_id_m2.cpp`, `bmm_m2.cpp`, `reader_bmm_8bank_output_tiles_partitioned.cpp`) do not
  exist on this branch, and there are zero existing `TensorAccessor(ta::...)` / `dfb::` usages anywhere in the tree — no op
  has actually landed a `create_program_spec` port yet. The `matmul`/`rand` "exemplars" are still `create_descriptor`
  (ProgramDescriptor) factories. The port shape was reconstructed from the framework headers
  (`program_spec.hpp`, `kernel_spec.hpp`, `program_run_args.hpp`, `dataflow_buffer.h`, `kernel_args.h`) and the recipe — so
  the kernel-side `ta::`/`dfb::`/`args::` generated-token usage is **unvalidated by build** (the worktree has no build dir,
  per instructions). First op to actually compile this concept should treat the generated-header behavior as shakedown.

## Open items for downstream

- **Remaining reshard factories (7)**: SameWidth/SameHeight/Generic/CopyLocalShard. Several read **L1 shard-spec geometry**
  (`shard_spec().grid()`, block/width-sharded core-count math, `std::gcd` page sizing — see
  `nd_reshard_program_factory_copy_local.cpp:50-80`) absent from this simplest DRAM→DRAM page-copy variant. Those will exercise
  shard-spec handling and possibly the templated `local_is_input/local_is_output` variants as multi-variant factories.
- **Cross-op kernel touches**: none. Both ported kernels
  (`nd_reshard_copy_pages_{reader,writer}.cpp`) live in this op's dir and are used only by this factory
  (`grep -rl` confirms), so they were ported **in place**, not forked.
- **Fake-CB self-loops / compute-pointer escape valves**: none used.
- **Build/test not run** per instructions (no build dir; do-not-build). Verification is limited to a static
  anti-pattern sweep (no `create_descriptor`/`CircularBuffer`/`CBDescriptor`/`TensorAccessorArgs`/`buffer()->address()`/
  positional `get_*_arg_val` survive in the four touched files). A real build + the reshard pytests/gtests are required
  before merge.

---

## Per-factory STATUS (all variants)

`reshard`'s `program_factory_t` holds 8 factory entries (5 distinct factory types).
This section is the authoritative per-factory result after the two porting passes.

| Factory | STATUS | Notes |
|---|---|---|
| `NdReshardCopyPagesFactory` | **PORTED** (pass 1) | DRAM→DRAM page copy. 1 DFB (reader→writer FIFO), 2 TensorAccessors. |
| `NdReshardCopyLocalShardFactory<true/false>` | **PORTED** (pass 2) | L1↔DRAM / L1→L1 local-shard copy. No DFB; 2 TensorAccessors. |
| `ReshardSameHeightFactory<true/false>` | **PORTED** (pass 3 — Case-2) | Remote base addr via `TensorAccessor(ta::remote).get_bank_base_address()`; local shard = borrowed self-loop DFB; per-segment tail = varargs. |
| `ReshardSameWidthFactory<true/false>` | **PORTED** (pass 3 — Case-2) | Same Case-2 remote-addr bridge; + local scratch self-loop DFB on the unaligned reader path (`#ifdef UNALIGNED`). |
| `ReshardGenericFactory` | **PORTED** (pass 3 — Case-2) | Input base addr via `TensorAccessor(ta::input).get_bank_base_address()`; output = borrowed self-loop DFB; physical-core maps + stride table = varargs. |

### CORRECTION to pass-2 conclusion (pass 3)

Pass 2 marked the three legacy-reshard factories METAL-BLOCKED on the premise that a raw
`buffer->address()` threaded through an RTA into an `AllocatorBank`/`UnicastEndpoint` NOC primitive
has no typed channel. **That premise was wrong for data-movement kernels.** Per the updated recipe's
Case-2 rule, a DM kernel that consumes a tensor's raw base address is **portable**: bind the tensor as a
normal `TensorParameter`/`TensorBinding` and pull the base kernel-side via
`TensorAccessor(ta::name).get_bank_base_address()` (which returns the bank-relative base — identical to
the legacy `buffer->address()` offset, same in every bank), keeping the existing raw `AllocatorBank` +
`bank_id` / `UnicastEndpoint` + physical-core NOC walk UNCHANGED. None of these reshard kernels are
COMPUTE (the one Case-2 carve-out that stays blocked), so all three port.

### PORTED — `ReshardSameHeightFactory<local_is_output>` (pass 3)

- **Concept**: `MetalV2FactoryConcept` — `create_program_spec` returning `ProgramArtifacts`. Both template
  instantiations ported; routing/variant unchanged.
- **Kernels (forked)**: `reshard_same_height_{reader,writer}.cpp` live in the cross-op dir
  `sharded/device/kernels/dataflow/` (used only by this factory but outside the op's own dir), so forked
  to `kernels/reshard_same_height_{reader,writer}_m2.cpp`. Whitelist changes only: positional CTAs/RTAs →
  `get_arg(args::…)`; local shard CB → `dfb::shard_cb`; remote `base_read_addr`/`base_write_addr` RTA →
  `TensorAccessor(ta::remote).get_bank_base_address()` (Case-2 bridge). The variable-length per-segment
  RTA tail (legacy read via `get_arg_addr(5)`) → runtime varargs. All NOC logic, `AllocatorBank`, bank_id,
  offset arithmetic UNCHANGED.
- **Spec shape**: 2 KernelSpecs of one source (READER + WRITER role; CT define `read_from_dram`/`write_to_dram`
  by `local_is_output`), 2 TensorParameters (`local` borrowed, `remote` Case-2), 1 borrowed-memory DFB
  (`shard_cb`, `borrowed_from = local`) self-looped on both kernels, 1 WorkUnitSpec.
- **Dropped plumbing**: remote-buffer base-address RTA (slot 3, a `Buffer*`) → Case-2 TensorBinding; CB-index
  CTA → `dfb::`.

### PORTED — `ReshardSameWidthFactory<local_is_output>` (pass 3)

- Same Case-2 bridge for the remote base address; same borrowed shard self-loop DFB. The **unaligned reader
  path** uses a second local (non-borrowed) scratch DFB (`scratch_cb`) — bound as a self-loop **only** when
  `local_is_output && unaligned`, gated kernel-side by `#ifdef UNALIGNED` (the legacy `if constexpr (unaligned)`
  CTA gate promoted to a preprocessor define so `dfb::scratch_cb` never enters name lookup on the aligned
  path / the writer source). Per-write tail → varargs.
- **Spec shape**: 2 KernelSpecs of one source, 2 TensorParameters, 1–2 DFBs (shard always; scratch on the
  unaligned reader path), 1 WorkUnitSpec.
- The redundant double `get_bank_ids_from_logical_core` call in the legacy work-split loop was collapsed to
  the single live call (functionally identical; bank_id recomputed each iteration as before).

### PORTED — `ReshardGenericFactory` (pass 3)

- **Kernels (forked)**: `reshard_reader.cpp` / `reshard_reader_diff_width.cpp` → `kernels/reshard_reader_m2.cpp`
  / `kernels/reshard_reader_diff_width_m2.cpp`. Input base address (legacy RTA at index `grid.x+grid.y`,
  back-patched to a `Buffer*`) → `TensorAccessor(ta::input).get_bank_base_address()`. The leading
  physical-core-coord maps **and** the host-compressed stride(-of-strides) table ride VERBATIM as a runtime
  vararg block (the kernel still indexes them positionally — `get_arg_val(idx)` → `get_vararg(idx)`); the
  dropped input-addr slot is erased from the block host-side (`strip_addr`). The manual `UnicastEndpoint{}`
  NOC reads by physical core coord stay UNCHANGED.
- **Spec shape**: 2 KernelSpecs of one runtime-selected source (page-size match selects diff-width vs same),
  2 TensorParameters (`input` Case-2, `output` borrowed), 1 borrowed-memory DFB (`shard_cb`,
  `borrowed_from = output`) self-looped on both kernels, 1 WorkUnitSpec.
- **Note**: the legacy `detail::get_runtime_args_for_given_ranges*` helpers (unchanged, out of scope) still
  take an `input_addr` parameter; the real `input_buffer->address()` is passed in and then immediately
  stripped before the args reach varargs — it never crosses to the device. The typed binding supplies the
  base address.

### PORTED — `NdReshardCopyLocalShardFactory<local_is_input>` (pass 2)

- **Concept**: `MetalV2FactoryConcept` — `create_program_spec` returning `ProgramArtifacts`. Both
  template instantiations (`<true>`, `<false>`) ported; both already routed by `select_program_factory`
  and present in `program_factory_t` (no device-op-class edit needed beyond the factory method swap).
- **Kernel**: `device/kernels/nd_reshard_copy_local_shards.cpp` — ported **in place** (lives in this op's
  own kernels dir, used only by this factory; `grep -rl` confirms). Whitelist changes only:
  positional `get_compile_time_arg_val`/`get_common_arg_val`/`get_arg_val` → named `get_arg(args::…)`;
  the two explicit `TensorAccessor(args, bank_base_address)` (base address read from CRTAs) collapse to
  `TensorAccessor(ta::input)` / `TensorAccessor(ta::output)`. All logic, `if constexpr (is_reader)`
  branches, padding math, and `CoreLocalMem` local-shard reads UNCHANGED. (`CoreLocalMem` here derives
  its address from the accessor's own `page.noc_addr()` — it is NOT a smuggled host address, so it stays.)
- **Spec shape**: 2 KernelSpecs (`brisc` role READER, `ncrisc` role WRITER) of one source, identical 7
  scalar CTAs + identical CRTA schema (`num_shards`, `shard_id_stride`); they differ only in DM role and
  the per-core `first_shard_id` RTA. 2 TensorParameters (`input`, `output`). **No DFB** (`dfb_bindings`
  empty is legal) — the program moves data directly local-L1↔remote-bank via the two accessors. 1
  WorkUnitSpec hosting both kernels on the shard-data grid.
- **Dropped plumbing**: the two CRTA remote-bank base addresses (`bank_base_address_src/dst`) → auto-injected
  by the `TensorBinding`s; positional CTA TensorAccessorArgs chains → carried by the bindings.
- **Why this one is portable (unlike the other three)**: the remote side is accessed through a real
  `TensorAccessor` (constructed from an explicit base address in legacy, now from the binding), so the
  base address rides the typed channel. The other three never build a `TensorAccessor` for the remote —
  they hand a raw `buffer->address()` to an `AllocatorBank`/`UnicastEndpoint` NOC primitive, which the
  typed binding cannot supply.

> NOTE (superseded by pass 3): the original pass-2 "METAL-BLOCKED" detail sections for SameHeight /
> SameWidth / Generic that previously followed here have been removed — those three are now **PORTED**
> via the Case-2 `get_bank_base_address` bridge. See **§Per-factory STATUS (all variants)** above for the
> authoritative result and the per-factory pass-3 writeups.

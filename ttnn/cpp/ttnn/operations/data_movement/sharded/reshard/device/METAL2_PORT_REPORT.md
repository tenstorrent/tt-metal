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
| `ReshardSameHeightFactory<true/false>` | **METAL-BLOCKED** | Raw remote `buffer->address()` through RTA → `AllocatorBank` NOC. |
| `ReshardSameWidthFactory<true/false>` | **METAL-BLOCKED** | Same remote-address-through-RTA pattern (+ unaligned scratch CB). |
| `ReshardGenericFactory` | **METAL-BLOCKED** | Raw input `buffer->address()` + manual `UnicastEndpoint` NOC by physical core. |

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

### METAL-BLOCKED — `ReshardSameHeightFactory<local_is_output>`

Raw remote-buffer base address threaded through an RTA into a non-accessor NOC primitive.
- Host: `reshard_program_factory_same_height.cpp:127,134` push `remote_buffer` (a `Buffer*`) as RTA slot 3.
- Kernel `…/sharded/device/kernels/dataflow/reshard_same_height_reader.cpp:21,44` and
  `…/reshard_same_height_writer.cpp:21,44`: `base_read_addr/base_write_addr = get_arg_val<uint32_t>(3)`
  is fed to `noc.async_read/write(bank, …, {.bank_id=…, .addr=read_offset}, …)` over an
  `AllocatorBank<bank_type>` — there is no `TensorAccessor` for the remote tensor, so no `ta::` binding can
  supply the address, and the remote-buffer address is per-dispatch-varying (io tensor), so the
  CRTA escape valve would go stale on a cache hit. The local sharded CB (`cb.buffer = local_buffer`,
  `:79`) IS borrowed-memory-DFB-expressible, but the remote address is the hard wall.
- Off-rules change that would be needed: a kernel-side `TensorAccessor` (or `get_bank_base_address`
  bridge) for the remote tensor, which would require rewriting the kernel's `AllocatorBank` + bank_id +
  hand-computed offset NOC loop — outside the kernel-side whitelist.

### METAL-BLOCKED — `ReshardSameWidthFactory<local_is_output>`

Identical remote-address-through-RTA pattern, plus an unaligned-path scratch CB.
- Host: `reshard_program_factory_same_width.cpp:156` pushes `remote_buffer` as RTA slot 0
  (`std::vector<std::variant<uint32_t, Buffer*>>` back-patch).
- Kernel `…/dataflow/reshard_same_width_reader.cpp:24,50,79` / `reshard_same_width_writer.cpp:24,44`:
  `src_addr/dst_addr = get_arg_val<uint32_t>(0)` → `noc.async_read/write(bank, …, {.bank_id, .addr}, …)`
  over `AllocatorBank<bank_type>`; no remote `TensorAccessor`. Same stale-on-cache-hit wall as same-height.
- Additionally the unaligned path uses a second (non-borrowed) scratch CB (`cb_scratch_index`, host `:104`)
  — that part would self-loop fine, but the remote-address blocker stands.

### METAL-BLOCKED — `ReshardGenericFactory`

Raw input-buffer base address through RTA + fully manual `UnicastEndpoint` NOC by physical core coord.
- Host: `reshard_program_factory_generic.cpp:785,795` back-patch `input_buffer` (a `Buffer*`) into RTA slot
  `grid.x + grid.y`; `:746,754,764,774` pass `input_buffer->address()` into the detail RTA builders;
  `:724-733` precompute physical core x/y coords as leading RTAs.
- Kernel `…/dataflow/reshard_reader.cpp:23,67` and `reshard_reader_diff_width.cpp:23,85`:
  `input_shard_addr = get_arg_val<uint32_t>(arg_index)` and `start_x/start_y = get_arg_val(...)` feed
  `noc.async_read(UnicastEndpoint{}, dst, …, {.noc_x, .noc_y, .addr = input_shard_addr + addr_offset}, …)`.
  No `TensorAccessor` anywhere; the kernel walks a host-compressed stride table by physical core id and
  raw base+offset. The typed binding model cannot express the manual physical-core unicast, and the input
  address is per-dispatch-varying. Output CB (`cb.buffer = output_buffer`, `:692`) is borrowed-memory
  expressible, but the input-address-by-unicast is the wall.
- Off-rules change that would be needed: replace the manual `UnicastEndpoint` + packed physical-core RTAs
  with a `TensorAccessor`-based read, a substantial kernel rewrite far outside the whitelist.

### Common root cause / handoff

All three blocked factories share one shape: a **resident sharded tensor accessed by raw `buffer->address()`
threaded through an RTA into a non-`TensorAccessor` NOC primitive** (`AllocatorBank` + bank_id, or
`UnicastEndpoint` + physical core coord). This is the recipe's named genuine blocker ("a raw
`buffer->address()` threaded through a CTA/RTA to a kernel") and the gaps-doc remote-address blocker. The
local sharded buffers in all three ARE borrowed-memory-DFB-expressible (`cb.buffer = local/output_buffer`);
it is exclusively the *remote* tensor's address that has no typed channel. These unblock once a kernel-side
`TensorAccessor` (or sanctioned bridge) can supply a sharded remote tensor's base address to a bank/unicast
NOC walk — i.e. a remote-sharded `TensorAccessor` read path. No clever workaround was attempted (per recipe
§When the discipline doesn't fit); they remain on the legacy `ProgramDescriptorFactoryConcept` and the op
continues to build and dispatch per-factory.

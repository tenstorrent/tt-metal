# Quasar Uplift Report: `reduction/topk`

Recipe followed: [quasar_porting.md](docs/source/ttnn/ttnn/ai/quasar_porting.md), plus the canonical
documents it extends on branch `akertesz/op-porting-recipe`
(`ai/audit/quasar_audit.md`, `ai/audit/cb_dfb_quasar_audit_helper.md`,
`ai/post_port/semantic/gen2_hardware_configs.md`, `ai/post_port/pass_procedure.md`).

Leave this file uncommitted for review and delete it before merge.

---

## Status: RED

The op is not ready for an in-place Quasar uplift today. Four of the five RED-stop conditions in
§1 of the recipe fire. No source file was changed, so the deliverable is this report plus the work
list below.

**Files changed: none.** `git status` for the op directory shows only this report.

**Parity claim.** The diff is empty, therefore Wormhole and Blackhole behaviour is unchanged by
construction. No test run is needed to establish that, and none was performed here (see
[Commands for the human](#commands-for-the-human)).

---

## Why RED (four independent blockers)

### Blocker 1: the multi-core factory is not on Metal 2.0 yet

`TopKDeviceOperation` has two program factories and only one of them is ported.

| Factory | State |
|---|---|
| `TopKSingleCoreProgramFactory` | Metal 2.0. Returns `ProgramArtifacts` from `create_program_artifacts`, kernels use `dfb::` / `args::` / `tensor::` and `get_entry_size()`. |
| `TopKMultiCoreProgramFactory` | Legacy. Returns `ProgramDescriptor` from `create_descriptor`. |

- [topk_device_operation.hpp:26-37](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.hpp#L26-L37) declares both entry points side by side.
- [topk_multi_core_program_factory.cpp:66-78](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.cpp#L66-L78) is the legacy `create_descriptor`.
- Its kernels are still on positional arguments and raw tensor addresses, for example
  [reader_create_index_local_topk.cpp:16-19](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_local_topk.cpp#L16-L19)
  and [writer_final_topk.cpp:12-13](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_final_topk.cpp#L12-L13).

The recipe is explicit: "If the op is still on the legacy `descriptor` concept, stop and run the
Metal 2.0 port first." The multi-core factory needs its own pre-port audit and its own Metal 2.0
port before it can be uplifted.

This matters for Quasar and not only on paper.
[topk_device_operation.cpp:141-144](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp#L141-L144)
selects the multi-core factory whenever the reduced dimension is a power of two in
[8192, 65535), `k <= 64`, and the memory and core-count check passes. Nothing in that decision
looks at the architecture, so on Quasar the op can still dispatch a legacy `ProgramDescriptor`
program.

Two smaller pieces of good news for whoever picks that port up:

- The two semaphores it creates are zero-initialised. `INVALID` is `0`
  ([common_values.hpp:13](tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp#L13)), so
  [topk_multi_core_program_factory.cpp:333-344](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.cpp#L333-L344)
  passes Quasar-audit check 2. A non-zero initial value would have been a second, independent
  blocker.
- Its kernels are already on the Device 2.0 data-movement API (`Noc`, `DataflowBuffer`), and the
  buffer-usage scan below finds no hard blockers in them.

### Blocker 2: the compute kernel uses buffer credits as pointer arithmetic (needs an owner decision)

This is the deepest blocker and the one that would survive the other three being fixed.

Four of the eight buffers in the single-core factory are filled and drained by the compute kernel
alone. That much is fine: the recipe says a compute self-loop is legal on both hardware
generations. The problem is *how* the kernel moves those buffers along. It pushes and pops credits
that do not match the tiles it actually wrote or read, using the credit counters as a way to step
a read or write cursor around a ring:

| Site | What it does |
|---|---|
| [topk.cpp:100-104](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L100-L104) (`cb_wait_pop_front`) | `wait_front(n)` immediately followed by `pop_front(n)`, with no read of the buffer in between. Called at [topk.cpp:395-396](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L395-L396). |
| [topk.cpp:113-117](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L113-L117) (`cb_reserve_push_back`) | `reserve_back(n)` immediately followed by `push_back(n)`, with nothing written in between. Called at [topk.cpp:401-402](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L401-L402). |
| [topk.cpp:368-378](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L368-L378) | reserves and pushes `incr` entries but writes exactly one tile, so the cursor jumps to the other half of the double buffer. |
| [topk.cpp:36-55](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L36-L55) | waits on `2 * total_tiles`, reads `total_tiles`, then pops in two separate calls of `total_tiles` each. The comment on line 52 states the reason in terms of the ring's internal read pointer and limit. |

On Wormhole and Blackhole this works, because a Metal 2.0 dataflow buffer there has the same
behaviour as the old circular buffer and the credit counters are ordinary memory.

On Quasar the credits are hardware tile counters, and there is a hardware rule the pattern breaks.
A counter wait must be followed by at least one real data-movement operation on the *same* buffer
before the matching retire; otherwise the wait can be satisfied before the tiles or the free space
actually exist. The rule and its check are written into the low-level layer:

- [llk_io_unpack.h:38-41](tt_metal/hw/ckernels/quasar/metal/llk_io/llk_io_unpack.h#L38-L41): "TEN-4746: `llk_pop_tiles` on a dfb with no unpack (UNPACR) since `llk_wait_tiles`".
- [llk_io_pack.h:38-41](tt_metal/hw/ckernels/quasar/metal/llk_io/llk_io_pack.h#L38-L41): the write-side twin.
- [llk_tdma_guard.h:7-19](tt_metal/tt-llk/tt_llk_quasar/common/inc/llk_tdma_guard.h#L7-L19) explains the constraint.
- [dataflow_buffer.inl:140-247](tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl#L140-L247) shows that a compute kernel's `reserve_back` / `push_back` / `wait_front` / `pop_front` on Quasar route straight into those four guarded functions.

So `cb_wait_pop_front` and `cb_reserve_push_back` fail the check with low-level asserts enabled,
and hit a real hardware hazard with them disabled. Note that the guard is only a debug aid; the
hazard it describes exists either way.

The sanctioned way to move a ring cursor without moving data does not exist on Quasar.
`evil_set_read_ptr` and `evil_set_write_ptr` are declared only off Quasar, and the comment beside
them names this exact situation:
[dataflow_buffer.h:338-342](tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L338-L342), "Not
declared on Quasar (redesign Classes 2-5)."

Recipe §7 covers this case by name and tells us not to work around it in the op: "If a sanctioned
Metal 2.0 DFB/kernel API is missing on Quasar (e.g. the ring-rewind `evil_set_*` is
`#ifndef ARCH_QUASAR`, so absent on Gen2), flag it as a missing-feature for the runtime team rather
than hand-rolling an op-level equivalent." Inserting a dummy tile copy purely to satisfy the
hardware rule would change what the math unit does and cannot be validated from here, so it is an
owner decision, not a mechanical guard.

### Blocker 3: Quasar cannot transpose 32-bit tiles (low-level gap, already ticketed)

The whole algorithm is built on `transpose_tile`, at
[topk.cpp:40](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L40) and
[topk.cpp:90](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L90). On
Quasar that call refuses any buffer whose unpacked format is `Float32` or `Int32`:
[transpose.h:126-131](tt_metal/hw/inc/api/compute/transpose.h#L126-L131) and the matching check in
`transpose_init` at
[transpose.h:73-78](tt_metal/hw/inc/api/compute/transpose.h#L73-L78), both tagged `tt-llk#1559`.

That rules out two of the op's three supported input types and one of its two index widths:

| Configuration | Effect on Quasar |
|---|---|
| `FLOAT32` input | Value buffers are `Float32`, so the transpose is refused. |
| 32-bit index output (`UINT32` / `INT32`, or any padded width above 65535) | Index buffers are 32-bit, so the transpose is refused. |

Only `BFLOAT16` input with 16-bit indices survives, and `BFLOAT8_B` is separately excluded by
blocker 4. This is not fixable inside the op; it belongs to the low-level kernel team under
`tt-llk#1559`.

### Blocker 4: two data formats the op selects do not exist on Quasar

The Metal 2.0 spec validator rejects a buffer format that the target architecture does not have:
[program_spec.cpp:1673-1683](tt_metal/impl/metal2_host_api/program_spec.cpp#L1673-L1683), against
the list at
[tt_backend_api_types.cpp:97-122](tt_metal/common/tt_backend_api_types.cpp#L97-L122). Quasar has
`Int16` and `Int32` but not `UInt16` or `UInt32`, and it has no block-float formats at all
(`Bfp8_b` and `Bfp4_b` are replaced by the MXFP family).

The single-core factory selects all three of the missing ones:

- Index buffers get `UInt16` or `UInt32`:
  [topk_single_core_program_factory.cpp:44-47](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_single_core_program_factory.cpp#L44-L47)
  and the buffer declarations at
  [topk_single_core_program_factory.cpp:140-190](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_single_core_program_factory.cpp#L140-L190).
- The input and output-value buffers carry the tensor's own format, which is `Bfp8_b` for a
  `BFLOAT8_B` tensor:
  [topk_single_core_program_factory.cpp:38-47](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_single_core_program_factory.cpp#L38-L47).

The index case is a real in-op fix (see work item A below), because the host `Int16` value is
translated to the right hardware code for Quasar when the kernel is built:
[genfiles.cpp:635-663](tt_metal/jit_build/genfiles.cpp#L635-L663). The `BFLOAT8_B` case is not; it
would need an MXFP path, which is a feature, not a port step. Recipe §7's guidance applies: the
op does have its own format branch, so the branch is the op's to arch-select, but a format the
hardware simply lacks gets flagged rather than papered over.

---

## Quasar-audit checks

### Check 1: device-side buffer redesign

Scans from `cb_dfb_quasar_audit_helper.md` step 4, run over the single-core factory's kernels
(`compute/topk.cpp`, `dataflow/reader_create_index_tensor.cpp`,
`dataflow/writer_binary_interleaved.cpp`, `dataflow/topk_dataflow_common.hpp`):

| Scan | Result |
|---|---|
| `get_local_cb_interface(...).<field>` (hard blocker) | none |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` | none |
| `read_tile_value` / `get_tile_address` | none |
| `get_pointer_to_cb_data` | none |
| `fifo_page_size` / `fifo_num_pages` | none (already `get_entry_size()`) |
| `evil_*`, `fifo_wr_ptr` / `fifo_rd_ptr` writes | none (one mention inside a comment) |
| `get_read_ptr()` / `get_write_ptr()` | one, at [topk_dataflow_common.hpp:36](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/topk_dataflow_common.hpp#L36) |

The single `get_write_ptr()` is a local fill: the reader writes an index tile directly into the
buffer entry through `CoreLocalMem`, which is the sanctioned local-fill form of a linear FIFO, not
pointer surgery. `CoreLocalMem` uses `uintptr_t`
([core_local_mem.h:28](tt_metal/hw/inc/api/core_local_mem.h#L28)), so it is safe against Quasar's
64-bit pointers.

Per-buffer portability, with the two columns the audit asks for (1xx = Wormhole and Blackhole,
2xx = Quasar end state):

| Buffer | Endpoints | Class | 1xx | 2xx | Note |
|---|---|---|---|---|---|
| `input` | reader produces, compute consumes | 1 (linear FIFO) | Portable | Portable | Format needs the arch mapping only when the tensor is `BFLOAT8_B` or `FLOAT32` (blockers 3 and 4). |
| `index` | reader produces, compute consumes | 1 | Portable | Portable | Format needs the arch mapping (work item A). |
| `transposed_val` | compute produces and consumes | 4 (credits decoupled from data) | Portable | **Blocked** | Needs a design decision. Blocker 2. |
| `transposed_ind` | compute produces and consumes | 4 | Portable | **Blocked** | Blocker 2. |
| `result_prep_val` | compute produces and consumes | 4 | Portable | **Blocked** | Blocker 2. Double-buffered ring stepped by credit pushes. |
| `result_prep_ind` | compute produces and consumes | 4 | Portable | **Blocked** | Blocker 2. |
| `output_val` | compute produces, writer consumes | 1 | Portable | Portable | Format mapping only. |
| `output_ind` | compute produces, writer consumes | 1 | Portable | Portable | Format mapping (work item A). |

Rollup: any **Blocked** row makes the op **RED**. Four rows are blocked.

The four blocked buffers would be self-loop candidates on Quasar (a compute kernel packing tiles
that the same kernel later unpacks is a legal Quasar pattern) if the credits matched the data. The
audit helper's own table rules out the self-loop when credits are decoupled from addresses, which
is exactly this case.

### Check 2: non-zero-initialised semaphores

Pass. The single-core factory declares no semaphores at all. The multi-core factory declares two,
both with initial value `INVALID`, which is `0`.

---

## The §7 and §8 items, one by one

Applied: none. The recipe says these are reactive fixes, applied when their symptom fires, and no
build or run happened here.

| Item | Applies to `topk`? |
|---|---|
| §7 implicit sync must stay on | Yes, and it is on. The data-movement configs come from `create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`, which default the opt-out to false ([datamovement_kernel_config.hpp:26-42](ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp#L26-L42)). Nothing to change. |
| §7 `compute_kernel_hw_startup` exactly once | Already correct: one call, first statement of the kernel body ([topk.cpp:141](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L141)). |
| §7 re-init on every buffer-id change | **Not satisfied.** Two distinct gaps; work items B and C below. |
| §7 tilize needs pack config first | Not applicable. The op does not tilize. |
| §7 no wide-tilize chunking | Not applicable. |
| §7 Quasar has `Int32`, no `uint16` / `uint32` | Applies. Blocker 4 and work item A. |
| §7 row-major shard width alignment | Not applicable. All buffers are tiled. |
| §7 no non-zero-init semaphores | Satisfied. |
| §7 do not invent Quasar-only device interfaces | Applies, and is why blocker 2 stops here instead of being worked around. |
| §8.1 build skew | Cannot be assessed. The ahead-of-time kernel-compile target only covers Wormhole and Blackhole ([fake_kernels_target/CMakeLists.txt:53-63](tt_metal/jit_build/fake_kernels_target/CMakeLists.txt#L53-L63)), so Quasar kernel compilation only happens during a simulator run. One certain failure is already known: work item B. |
| §8.2 hangs, including the historical double-count rows | Cannot be assessed without a run. If one of those three rows appears, report it as a regression; do not disable implicit sync. |
| §8.3 stale `fifo_page_size` | Already clean. Both data-movement kernels read `get_entry_size()`. |
| §8.4 unported low-level kernels | Applies. The bitonic top-k itself **is** ported for Quasar ([ckernel_sfpu_topk.h](tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_topk.h), all three stages, stable sort excluded by design). The gaps are `transpose_tile` for 32-bit data (blocker 3) and `copy_tile_to_dst_init_short_with_dt` (work item B). |
| §8.5 wait-then-pop and reserve-then-push traps | Applies, and is blocker 2. |
| §11 NoC and multicast | Not applicable to the single-core path: no multicast, no semaphores, no directional NoC tricks. It **will** apply to the multi-core factory, which multicasts between local cores and one aggregating core. |

---

## Work list for when the blockers clear

Everything here is mechanical and each item can be guarded so that Wormhole and Blackhole keep
their current path exactly. None of it was applied, because landing unverifiable changes for a
path that cannot run yet is the failure mode both recipes warn against.

### A. Map the index buffer format per architecture (host factory)

In [topk_single_core_program_factory.cpp:44-47](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_single_core_program_factory.cpp#L44-L47),
select `Int16` in place of `UInt16` and `Int32` in place of `UInt32` when
`input_tensor.device().arch() == tt::ARCH::QUASAR`. Both are valid Quasar formats and the build
step translates the host `Int16` value to the correct hardware code
([genfiles.cpp:641](tt_metal/jit_build/genfiles.cpp#L641)). The index tensor's own data type stays
`UINT16` / `UINT32`; only the on-core buffer format changes, and the stored bytes are the same
16-bit or 32-bit little-endian integers.

One follow-on: the reader's `uint16_output` compile-time argument at
[topk_single_core_program_factory.cpp:244-245](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_single_core_program_factory.cpp#L244-L245)
is currently derived by comparing the buffer format against `UInt16`. Once the format becomes
architecture-dependent that comparison is wrong, so it should read the `uint16_output` local
already computed at
[topk_single_core_program_factory.cpp:35](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_single_core_program_factory.cpp#L35).
On Wormhole and Blackhole the two are equivalent, so that edit is behaviour-preserving there.

One item to confirm on hardware once it runs: an index value at or above 32768 is a negative
number when read as `Int16`. The sort only ever compares index values when stable sorting is on,
and stable sorting is already refused on Quasar, and an `Int16`-to-`Int16` pack is a straight copy
with no numeric conversion, so my expectation is that the bytes round-trip unchanged. It is worth
a targeted check at a padded width above 32768 rather than an assumption.

### B. Replace `copy_tile_to_dst_init_short_with_dt` on Quasar (compute kernel)

That function is declared only off Quasar
([tile_move_copy.h:76-84](tt_metal/hw/inc/api/compute/tile_move_copy.h#L76-L84)), so the compute
kernel will not compile for Quasar as written. Four call sites:
[topk.cpp:326](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L326),
[topk.cpp:330](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L330),
[topk.cpp:341](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L341),
[topk.cpp:345](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L345).

The replacement is the two public calls the Wormhole and Blackhole version expands to, in the same
order, and both exist on Quasar:

```cpp
reconfig_data_format_srca(old_dfb, new_dfb);
copy_tile_to_dst_init_short(new_dfb);
```

The second call is also what Quasar requires on its own terms: buffer descriptors live in the
operation's init, and the format reconfiguration alone only touches a separate set of registers.
The framework header states this at
[reconfig_data_format.h:136-139](tt_metal/hw/inc/api/compute/reconfig_data_format.h#L136-L139).

**Better home for this fix.** Eight other places in the tree call the same function, in
`reduction/sampling`, `reduction/moe`, the DeepSeek prefill ops, a matmul collective kernel, and
several `tt-train` kernels. Every one of them will hit this wall. The durable fix is a Quasar
branch inside `tile_move_copy.h` so all callers share one mechanism, and that file is outside the
op directory, which the recipe puts off limits for an op port. Routing: compute-API and low-level
kernel owners.

### C. Re-initialise the packer on every output-buffer change (compute kernel)

On Quasar the packer's buffer descriptor holds the output buffer's base address, and
`pack_reconfig_data_format` deliberately does not reprogram it
([llk_pack_common_api.h:50-58](tt_metal/hw/ckernels/quasar/metal/llk_api/llk_pack_common_api.h#L50-L58)).
`pack_tile` only computes a tile index and relies on whatever descriptor is currently programmed
([llk_pack_tile_api.h:90-100](tt_metal/hw/ckernels/quasar/metal/llk_api/llk_pack_tile_api.h#L90-L100)).
So a kernel that switches output buffer with only a format reconfiguration writes into the
previously initialised buffer.

The framework documents the fix: "When the pack output operand changes, call `pack_init(new_cb_id)`
before `pack_tile`"
([reconfig_data_format.h:713-721](tt_metal/hw/inc/api/compute/reconfig_data_format.h#L713-L721),
repeated at [reconfig_data_format.h:750-760](tt_metal/hw/inc/api/compute/reconfig_data_format.h#L750-L760)).
`pack_init` exists on both generations
([pack.h:34-37](tt_metal/hw/inc/api/compute/pack.h#L34-L37)).

The kernel switches output buffer at five places, each currently guarded only by a format
reconfiguration:
[topk.cpp:33](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L33),
[topk.cpp:68](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L68),
[topk.cpp:73](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L73),
[topk.cpp:251](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L251),
[topk.cpp:255](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L255).
`compute_kernel_hw_startup` initialises the packer for `output_val` only, at
[topk.cpp:141](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L141), so
every other output buffer is affected.

This is the kind of gap that is silent rather than loud: nothing asserts, the data simply lands in
the wrong buffer.

### D. Add the Gen2 compute hardware config (host factory)

Per `gen2_hardware_configs.md`, the single-core factory is one "shape 4 compute" site: a
`ComputeGen1Config` written out by hand at
[topk_single_core_program_factory.cpp:379-384](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_single_core_program_factory.cpp#L379-L384).
A compute kernel whose config holds only the Gen1 alternative cannot run on Quasar at all.

The prescribed change: lift the existing initialiser into a local without retyping it, then add

```cpp
m2::ComputeHardwareConfig compute_hw = compute_gen1;
if (arch == tt::ARCH::QUASAR) {
    // TODO(#52269): Quasar unpack_modes are copied from Gen1 and not yet optimized for Quasar.
    compute_hw = m2::ComputeGen2Config{
        .enable_32_bit_dest = ...,   // copied verbatim
        .double_buffer_dest = ...,   // copied verbatim
        .unpack_modes = ...,         // copied verbatim
    };
}
```

`bfp_pack_precision_mode` has no Gen2 equivalent and is not set here anyway.
`enable_2x_src_register` must be left at its default. `ComputeGen2Config` and the shared-field
accessors are present in this checkout
([compute_hardware_config.hpp:129-182](tt_metal/api/tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp#L129-L182)).

The two data-movement configs are "shape 1" and need no work: they already go through the
architecture-agnostic helper.

### E. Refuse the configurations Quasar cannot run (device operation)

Once A to D are in, the reachable Quasar envelope is: `BFLOAT16` input, 16-bit indices, unstable
sort, single-core factory. `FLOAT32` and `BFLOAT8_B` inputs and 32-bit indices are all out, for
the reasons in blockers 3 and 4.

The op already has precedent for exactly this, and the same reasoning, at
[topk_device_operation.cpp:168-178](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp#L168-L178),
where stable sorting is refused on anything other than Wormhole and Blackhole so the caller gets a
usable error instead of a kernel build failure. New checks belong beside it. Without them, a
`BFLOAT8_B` input fails inside the spec validator and a `FLOAT32` input trips a low-level assert
far from the cause.

Separately, `select_program_factory` needs to stop choosing the legacy multi-core factory on
Quasar while blocker 1 stands. Falling back to single-core there is consistent with what that
function already does when the multi-core path is not feasible
([topk_device_operation.cpp:106-144](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp#L106-L144)).

Note that both edits are in the device-operation class rather than a program factory. The
post-port pass procedure keeps passes out of that file, and rightly so for a single narrow fix; the
Quasar uplift recipe scopes the work to "kernels/factories" too. I am flagging these two as
deliberate exceptions to be agreed with the op owner, on the grounds that the op's own established
pattern for an architecture that cannot run a configuration is a check in
`validate_on_program_cache_miss`.

---

## Out of scope, and why

`topk_route_prep` and `topk_route_finish` are two additional device operations in the same
directory. Both are on the oldest style of factory (`CreateProgram`, `CreateCircularBuffer`,
`CreateKernel`, `SetRuntimeArgs`, plus `create` and `override_runtime_arguments`), so both are far
from a Quasar uplift:
[topk_route_prep_program_factory.cpp:139-218](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_route_prep_program_factory.cpp#L139-L218)
and
[topk_route_finish_program_factory.cpp:157-261](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_route_finish_program_factory.cpp#L157-L261).

They are also unreachable on Quasar. Their only caller is the large-`k` route at
[topk.cpp:342-363](ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp#L342-L363), and the predicate
that gates it returns false on any architecture other than Blackhole
([topk.cpp:326-328](ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp#L326-L328)). They are recorded
here for completeness and excluded from the verdict.

I also did not run the full buffer-usage classification over the multi-core kernels. Blocker 1
means they will be rewritten by the Metal 2.0 port before any uplift reads them, so classifying
them now would describe code that is about to change. The cheap hard-blocker scans were run and
came back clean.

---

## Commands for the human

I ran no builds and no tests. These are the commands to confirm the report.

Wormhole and Blackhole baseline (should be unaffected, since the diff is empty):

```bash
./build_metal.sh -e --enable-fake-kernels-target
source python_env/bin/activate && pytest tests/ttnn/unit_tests/operations/reduce/test_topk.py
```

Quasar simulator, smallest single-core `BFLOAT16` case in the suite. Expect it to fail to build the
compute kernel on work item B; that is the first symptom to confirm:

```bash
TT_METAL_SIMULATOR=<path>/libttsim.so TT_SIMULATOR_LOCALHOST=1 \
ARCH_NAME=quasar CHIP_ARCH=quasar \
TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest "tests/ttnn/unit_tests/operations/reduce/test_topk.py::test_topk[None-True-True-N=1-C=1-H=32-W=64-dim=3-k=2-BFLOAT16_B]"
```

Adjust the test id if parametrisation has moved. Per recipe §9, run with low-level asserts on
first, then again with `TT_METAL_LLK_ASSERTS` unset, because several failures only appear one way.
With asserts on, the expected second symptom after B is the TEN-4746 message from blocker 2.

---

## Notes on the recipe

Offered in the spirit the recipes invite, for whoever maintains them.

1. **`quasar_audit.md` check 1 has no entry for credit-only ring stepping on a compute self-loop.**
   The buffer-audit helper does cover it, as Class 4, but the path there runs through "is this a
   compute self-loop?", and the recipe's summary in §6 of `quasar_porting.md` says a compute
   self-loop is "fine on both". That is true of the self-loop and false of what this op does with
   it. A one-line qualifier in §6 ("legal on both, provided credits match the tiles actually
   packed and unpacked") would have saved a wrong turn.

2. **§8.5's fix column understates the wait-then-pop item.** It reads as a timing workaround
   ("intervening IDMA op"). Since the check landed as a named hardware constraint with an assert
   (TEN-4746) and the ring-rewind operations that would otherwise avoid it are absent on Quasar,
   the item is closer to a design blocker than to a one-line insertion, at least when the credit
   operation is moving a cursor rather than accompanying real data.

3. **Two Quasar re-init requirements are documented only in framework headers.** The packer one
   ("call `pack_init(new_cb_id)` before `pack_tile`") and the unpack one ("call the op init again
   for the new operand pair") are both spelled out precisely, in
   `api/compute/reconfig_data_format.h`, and neither is discoverable from §7's one-line summary.
   §7 could point at those two notes by name.

4. **`copy_tile_to_dst_init_short_with_dt` is listed in §8.4 as a low-level gap but has no
   replacement given.** The replacement is two public calls that both exist on Quasar. Nine call
   sites across the tree are waiting on it. Worth naming in the recipe, along with the fact that
   the right place to fix it is the framework header rather than each op.

5. **Format support is stricter than §7 says.** §7 says "Quasar has Int32, not UInt32 (and no
   uint16)". It is worth adding that `Int16` *is* available and is the intended 16-bit integer
   format, that the build step translates the host `Int16` value to the right hardware code, and
   that block-float formats are absent as well. The third point is the one that catches ops out,
   because it changes which tensor data types an op can accept rather than only which buffer
   format it names.

6. **An honest note on what an audit without hardware can and cannot say.** §9 covers the parity
   side well. What it does not cover is that on Quasar the ahead-of-time kernel-compile target does
   not exist, so an auditor cannot even establish that the kernels compile. Everything about the
   Quasar side of a report like this one is read from headers, and the recipe could say so plainly
   so readers calibrate accordingly.

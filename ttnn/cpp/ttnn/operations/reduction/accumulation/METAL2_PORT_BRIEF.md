# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/reduction/accumulation`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `dc266b472bd 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports` *(carry this line into the port report's Provenance section)*

## Scope — two device operations, one port unit

The directory holds two device operations, each with exactly one factory. They share the kernel
header `device/kernels/accumulation_common.hpp`, so they are one port unit:

- **`AccumulationDeviceOperation`** (backs `cumsum` / `cumprod`) → `AccumulationProgramFactory` in
  `device/accumulation_program_factory.cpp`; kernels `device/kernels/dataflow/accumulation_reader.cpp`,
  `device/kernels/compute/accumulation_compute.cpp`,
  `device/kernels/dataflow/accumulation_writer.cpp`.
- **`EmaDeviceOperation`** (backs `ema`) → `EmaProgramFactory` in
  `ema/device/ema_program_factory.cpp`; kernels `ema/kernels/dataflow/ema_reader.cpp`,
  `ema/kernels/compute/ema_compute.cpp`, `ema/kernels/dataflow/ema_writer.cpp`.

All six kernel files and both factories are owned by this op — nothing is borrowed, nothing is lent.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both factories port to
`MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — both factories, vanilla single-program `ProgramDescriptor`.
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept` for both factories.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash ·
  `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported
  replacement) · pybind `create_descriptor` — all gate conjuncts — plus other migration-risky
  pybind, which would have surfaced as a `safe` warning. All `no` on both rows, and confirmed against
  the code.
- **One input, one output per factory.** Each factory binds `tensor_args`' input and
  `tensor_return_value`. The preallocated-output case never reaches the factory as a separate tensor:
  both device operations resolve it in `create_output_tensors`, which returns the caller's tensor when
  present. So do **not** model an optional-output binding.
- **No semaphores.** Neither factory pushes a `SemaphoreDescriptor`, and no kernel performs a
  semaphore operation. There is no `SemaphoreSpec` work in this port.

## Construct — to do

**Tensor bindings** (4 total, per binding):

- `AccumulationProgramFactory` **input** — **Case 1** (via `TensorAccessor`) → express as
  `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::<name>)`.
  Legacy: base at reader RTA index 0 (pushed as the `MeshTensor` itself in
  `emplace_runtime_args`, `accumulation_program_factory.cpp:231-240`), consumed at
  `accumulation_reader.cpp:36`. The `TensorAccessorArgs(input_tensor).append_to(...)` CTA block at
  `accumulation_program_factory.cpp:165-166` and the kernel's `TensorAccessorArgs<0>()` both disappear.
- `AccumulationProgramFactory` **output** — **Case 1** → same treatment. Base at writer RTA index 0
  (`accumulation_program_factory.cpp:242-251`), consumed at `accumulation_writer.cpp:30`; CTA block at
  `accumulation_program_factory.cpp:168-169`.
- `EmaProgramFactory` **input** — **Case 1**. Base at reader RTA index 0
  (`ema_program_factory.cpp:184`), consumed at `ema_reader.cpp:34`; CTA block at
  `ema_program_factory.cpp:124-125`. Note the accessor args start at CTA slot **1**, behind
  `total_tiles_per_core` — hence the kernel's `TensorAccessorArgs<1>()`, which becomes moot once the
  binding replaces it.
- `EmaProgramFactory` **output** — **Case 1**. Base at writer RTA index 0
  (`ema_program_factory.cpp:185`), consumed at `ema_writer.cpp:34`; CTA block at
  `ema_program_factory.cpp:127-128`.

No Case 2 binding exists — no kernel does raw address arithmetic, so you need no
`get_bank_base_address` bridge anywhere in this port. No borrowed-memory CBs either.

**A note on what the legacy code already does right.** These four bases do *not* ride a hand-written
`buffer()->address()` runtime arg. The factories push the `MeshTensor` object into
`emplace_runtime_args`, and the framework registers a `BufferBinding` it patches on cache hits. So
you are replacing a *correct* interim mechanism with the typed binding, not repairing a stale-pointer
bug — the numerics are already right today, and they must stay right.

**TensorParameter relaxation:** none. Neither factory needs one, and neither has a custom hash.

**TensorAccessor 3rd arg:** none — all four `TensorAccessor` constructions already use the
two-argument form, so there is nothing to drop.

**CB endpoints:**

| Factory | CB | Index | Disposition |
|---|---|---|---|
| `AccumulationProgramFactory` | `SRC` / `CB_IN` | `c_0`, 4 tiles | legal 1:1 — reader PRODUCER, compute CONSUMER |
| `AccumulationProgramFactory` | `DST` / `CB_OUT` | `c_1`, 4 tiles | legal 1:1 — compute PRODUCER, writer CONSUMER |
| `AccumulationProgramFactory` | `ACC` / `CB_ACC` | `c_2`, 1 tile | **self-loop** — bind the compute kernel PRODUCER **and** CONSUMER |
| `EmaProgramFactory` | `src_cb` | `c_0`, 2 tiles | legal 1:1 — reader PRODUCER, compute CONSUMER |
| `EmaProgramFactory` | `dst_cb` | `c_1`, 2 tiles | legal 1:1 — compute PRODUCER, writer CONSUMER |
| `EmaProgramFactory` | `prev_cb` (kernel: `trp_cb_idx`) | `c_2`, 1 tile | **self-loop** — bind the compute kernel PRODUCER **and** CONSUMER |

No 1P+1C assignment is needed, no multi-binding advanced option, no dead CB. Dispositions hold for
every configuration of both factories (each factory has a single code path; see the audit's CB
endpoints section for why).

**Both self-loops carry real FIFO traffic — keep both paths live.** Unlike a sync-free scratchpad,
these two CBs are driven by genuine `reserve_back`/`push_back` *and* `wait_front`/`pop_front` from
the one compute kernel:

- `CB_ACC` (accumulation) is how the compute kernel sequences its own unpacker against its own
  packer; the kernel documents the data race it prevents at `accumulation_compute.cpp:41-42`, and the
  1-tile depth is deliberate so every `reserve_back` lands on the same address
  (`accumulation_compute.cpp:56`). Keep the depth at one tile.
- `c_2` (EMA) is a packer→unpacker round trip for the transpose: pack the transposed tile out, read
  it straight back to transpose again (`ema_compute.cpp:109-120`).

So the second binding is not cosmetic here — both directions must remain functional. Both are
compute-kernel self-loops, so neither records Quasar debt.

**`get_tile_size(cb_id)` → the DFB method** (kernel-side whitelist rule 7). Four sites; the
`DataflowBuffer` object is already in scope at each:

| File | Line | Legacy | Replacement |
|---|---|---|---|
| `device/kernels/dataflow/accumulation_reader.cpp` | 33 | `get_tile_size(CB_IN)` | `dfb_in_obj.get_tile_size()` |
| `device/kernels/dataflow/accumulation_writer.cpp` | 27 | `get_tile_size(CB_OUT)` | `dfb_out_obj.get_tile_size()` |
| `ema/kernels/dataflow/ema_reader.cpp` | 30 | `get_tile_size(src_cb_idx)` | `dfb_src.get_tile_size()` |
| `ema/kernels/dataflow/ema_writer.cpp` | 30 | `get_tile_size(dst_cb_idx)` | `dfb_dst.get_tile_size()` |

`DataflowBuffer::get_tile_size()` is `constexpr`, which matters for the two EMA sites where the
result is bound to a `constexpr uint32_t`. In all four kernels the value is used as the NoC transfer
byte count, not as an accessor page size.

**Runtime args — all nameable, none vararg.** Every read is at a literal constant index, so name all
of them. The legacy positions and the names the kernels already give them:

- `accumulation_reader.cpp` / `accumulation_writer.cpp` (identical arg layout): 0 = tensor base
  (→ the binding), 1 `num_rows_per_core`, 2 `tiles_per_row`, 3 `input_tile_offset`, 4 `start_id`,
  5 `low_rank_offset`, 6 `high_rank_offset`, 7 `flip`.
- `accumulation_compute.cpp`: 0 `num_rows`, 1 `tiles_per_row`.
- `ema_reader.cpp`: 0 tensor base (→ the binding), 1 `src_start_tile`.
- `ema_writer.cpp`: 0 tensor base (→ the binding), 1 `dst_start_tile`.

Neither factory uses common runtime args.

## Watch for

- **CB endpoints (multi-binding):** none. No CB in either factory has a hidden second writer or a
  second reader, and the hidden-writer face is structurally impossible here — no kernel in the
  directory calls `get_write_ptr` / `get_read_ptr` / `get_local_cb_interface` / `evil_set_*` at all,
  and there are no semaphores to gate a raw co-fill. Every CB access in this op is a FIFO operation
  on a `DataflowBuffer` object.
- **Two identical compute descriptors split only by core range (accumulation).** The factory pushes
  `accumulation_compute.cpp` into two `KernelDescriptor`s
  (`accumulation_program_factory.cpp:187-217`) that differ **only** in `core_ranges` (`core_group_1`
  vs `core_group_2`); the CTAs, defines, and `ComputeConfigDescriptor` are byte-identical, because
  the per-core work count is a runtime arg, not a per-group compile-time arg. Two consequences:
  - The node sets are **disjoint**, so each node still has exactly one compute instance and the
    `c_0`/`c_1` censuses stay 1P+1C. This is *not* the dual-instance work-split shape — do not go
    looking for co-touched CBs.
  - Merging them into one `KernelSpec` over the union would be behaviour-preserving, but that is a
    change the port does not make. **Keep two `KernelSpec`s**, mirroring the legacy shape. (The
    redundancy is recorded for the ops team in the audit's Misc anomalies.)
- **Cross-op / shared kernels:** no borrowed kernel files, so no fork decision to make — the
  shared-kernel rungs do not apply. No `_metal2` fork exists anywhere under
  `ttnn/cpp/ttnn/operations/reduction/`, and there is no `experimental/quasar` copy of this op, so
  there is nothing in that (out-of-bounds) tree to be tempted by. Other binding ops: **none** — a
  repo-wide grep for the six kernel filenames finds no consumer outside this directory, so there is
  no sunset list.
  - The one intra-directory sharing point is the header `device/kernels/accumulation_common.hpp`,
    included by all six kernels **across both device operations**. The EMA kernels use exactly one
    thing from it (`ONE_TILE`); the three `CB_*` constants are used only by the accumulation kernels
    and are what your `dfb::` bindings replace. Both device operations are inside your port unit, so
    edits there are in scope — just don't leave one device operation's kernels half-converted
    against a changed header. Two constants in that header (`FIRST_TILE`, `WORKING_REG`) are used by
    no kernel; leave them alone, they are recorded as a team-only anomaly.
- **RTA varargs:** none — prefer named RTAs throughout (the full name list is in *Construct* above).
- **Dead arg you will be tempted to clean up.** `start_id` (reader/writer RTA index 4) is a dead
  *value*: the kernels use it only as a loop-counter base whose variable is never referenced in the
  body. It is recorded as a team-only anomaly and routes to the ops team. **Port it as-is** — carry
  the named arg through unchanged; removing it is a functional change and out of scope.

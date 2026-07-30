# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/reduction/manual_seed`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

## Scope

One DeviceOperation, `ManualSeedDeviceOperation`, with four factories — all four clear, so the whole op ports as one unit:

| Factory | Defined at | Shape |
|---|---|---|
| `ManualSeedSingleSeedToAllCoresProgramFactory` | `device/manual_seed_program_factory.cpp:58` | 1 compute kernel · no CBs · no tensors |
| `ManualSeedSingleSeedSingleCoreProgramFactory` | `device/manual_seed_program_factory.cpp:82` | 1 compute kernel · no CBs · no tensors |
| `ManualSeedSingleSeedSetCoresProgramFactory` | `device/manual_seed_program_factory.cpp:108` | reader + compute · 2 CBs · 1 tensor |
| `ManualSeedSetSeedsSetCoresProgramFactory` | `device/manual_seed_program_factory.cpp:176` | reader + compute · 3 CBs · 2 tensors |

Note that the first two factories build a program with **no DFBs and no tensor bindings at all** — a single compute kernel carrying one compile-time arg (the seed). That is a legitimately minimal `ProgramSpec`, not a sign you missed something.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — all four factories define `static ProgramDescriptor create_descriptor(...)` (`device/manual_seed_program_factory.hpp:19,24,29,34`).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` — uniform across all four factories, so the wiring is the same for each. (The readiness sheet's `Porting Target` column independently says the same.)
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) · pybind `create_descriptor` — all gate conjuncts — plus other migration-risky pybind, which surfaces as a `safe` warning that also fails the gate. All `no` on this op; the nanobind file exposes only the top-level `manual_seed` function (`manual_seed_nanobind.cpp:73-81`).

## Construct — to do

**Tensor bindings** (per binding):

- **`SingleSeedToAllCores`, `SingleSeedSingleCore`** — no tensor bindings. `tensor_args` is unused in both (`device/manual_seed_program_factory.cpp:59,83`).
- **`SingleSeedSetCores`** — `user_ids` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::user_ids)`.
  - Legacy delivery to remove: the `MeshTensor` entry in `emplace_runtime_args(core, {user_ids_mesh, core_id})` @ `:157`, the `TensorAccessorArgs(user_ids_mesh).append_to(...)` CTA append @ `:143`, and kernel-side `get_arg_val<uint32_t>(0)` @ `reader_manual_seed_read_user_id.cpp:16` + `TensorAccessorArgs<3>()` @ `:23`. All of it disappears.
- **`SetSeedsSetCores`** — `user_ids` — **Case 1**; `seeds` — **Case 1**. Both the same mechanical shape.
  - Legacy delivery to remove: `emplace_runtime_args(core, {user_ids_mesh, seeds_mesh, core_id})` @ `:235`, both `TensorAccessorArgs(...).append_to(...)` appends @ `:220-221`, and kernel-side `get_arg_val<uint32_t>(0)` / `(1)` @ `reader_manual_seed_read_all_data.cpp:16-17` plus the chained `TensorAccessorArgs<4>()` / `TensorAccessorArgs<...next_compile_time_args_offset()>()` @ `:25-27`.

There is no Case 2 anywhere — no kernel does raw pointer arithmetic on a tensor base, so you need no `get_bank_base_address` bridge.

**Note on the current delivery mechanism:** these bases arrive today as `MeshTensor` references in the RTA list, which the framework auto-registers as buffer bindings and patches on cache hits. So this is **routine port work, not a stale-pointer bug** you are fixing — don't describe it as a correctness fix in the port report.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — all three accessor constructions pass two arguments (`reader_manual_seed_read_user_id.cpp:30`, `reader_manual_seed_read_all_data.cpp:34,37`). Nothing to drop.

**CB endpoints:**

| Factory | CB | Action |
|---|---|---|
| `SingleSeedSetCores` | `c_0` `user_ids_cb_index` | **self-loop** — bind the reader PRODUCER *and* CONSUMER |
| `SingleSeedSetCores` | `c_1` `kernel_communication_cb_index` | legal 1:1 — reader PRODUCER, compute CONSUMER. No action. |
| `SetSeedsSetCores` | `c_0` `user_ids_cb_index` | **self-loop** — bind the reader PRODUCER *and* CONSUMER |
| `SetSeedsSetCores` | `c_1` `seeds_cb_index` | **self-loop** — bind the reader PRODUCER *and* CONSUMER |
| `SetSeedsSetCores` | `c_2` `kernel_communication_cb_index` | legal 1:1 — reader PRODUCER, compute CONSUMER. No action. |

All three self-loops are the same shape: a NoC read landing area the reader fills (`reserve_back` + `get_write_ptr`) and then reads straight back through `CoreLocalMem`, with no handoff to any other kernel. One toucher → self-loop. Legal on Gen1 for DM kernels.

No dead CBs to drop, and no multi-binding flag to set anywhere — the audit ran all three faces of the multi-binding hunt and all came back negative. If a CB here looks like it needs `allow_instance_multi_binding`, re-derive the census before setting it; the expected answer is no.

**DFB spec construction — one thing to check.** The legacy `CBDescriptor`s are built by the `push_tensor_circular_buffer` helper @ `device/manual_seed_program_factory.cpp:39-54`, which sets only `buffer_index`, `data_format`, and `page_size` on the `CBFormatDescriptor` — it **never sets the `.tile` field**. So there is no `tile_format_metadata` to copy per `DataflowBufferSpec` construction. Do set `data_format_metadata`: the reader kernels call `get_tile_size(<dfb_index>)` to size their NoC reads (`reader_manual_seed_read_user_id.cpp:40`, `reader_manual_seed_read_all_data.cpp:48,54`), which resolves through that metadata. Each CB is one entry (`total_size == page_size == tensor_tile_size` @ `:46,51`).

**Kernel-side metadata lookups (whitelist rule 7).** Move these onto the DFB object:

- `get_tile_size(<dfb_index>)` → `<dfb>.get_tile_size()` @ `reader_manual_seed_read_user_id.cpp:40` and `reader_manual_seed_read_all_data.cpp:48,54`. The `DataflowBuffer` is already in scope at all three sites.
- `get_dataformat(<dfb_index>)` @ `reader_manual_seed_read_user_id.cpp:29` and `reader_manual_seed_read_all_data.cpp:33,36` — **these three results are dead**; the `constexpr DataFormat` variables they initialize are never read. The `DataflowBuffer` is also declared *after* them, so moving the call onto the object would require hoisting the object declaration. Applying rule 7 here buys nothing and costs a reorder. Leave these three lines as they are, and note the choice in the port report. (The dead variables themselves are recorded as a team-only anomaly for the ops team; deleting them is not yours.)

## Watch for

- **CB endpoints (multi-binding):** none. All CBs are one-toucher (self-loop) or a clean locked-producer/locked-consumer pair. There is no hidden co-fill to hunt — the compute kernels receive *only* the `kernel_communication` CB index as a CTA (one CTA in `manual_seed_receive_all_data.cpp:16`, two in `manual_seed_single_seed_receive_user_id.cpp:16-17`), so they structurally cannot touch the `user_ids` / `seeds` CBs. The op declares no semaphores, so the semaphore-gated co-fill pattern has nothing to hide behind.
- **Cross-op / shared kernels:** **no cross-op sharing** — the op owns all five kernel files, borrows none, and the filename census confirms no other op binds any of them. No `_metal2` fork exists in either kernel directory, and none is needed for a cross-op reason.

  **One intra-op share to be deliberate about:** `device/kernels/compute/manual_seed_set_seed.cpp` is bound by **two** of this op's factories — `SingleSeedToAllCores` @ `device/manual_seed_program_factory.cpp:68-69` and `SingleSeedSingleCore` @ `:94-95`. Porting all four factories in one change (the expected scope — nothing is blocked) means both binders convert together, so convert the kernel **in place**; no fork. The two `KernelSpec`s differ only in core ranges and both pass the same single `seed` CTA, so one converted kernel serves both. If you end up porting the factories piecemeal, this kernel needs an `_metal2` fork beside the original instead — see `port_patterns.md` → *Caution: Porting a shared kernel*, intra-op shape.
- **RTA varargs:** none — prefer named RTAs throughout. Every RTA is read at a fixed literal index and has an obvious name from the kernel's own local: `reader_manual_seed_read_user_id.cpp:16-17` → `user_ids_tensor_buffer_addr`, `core_id`; `reader_manual_seed_read_all_data.cpp:16-18` → `user_ids_tensor_buffer_addr`, `seeds_tensor_buffer_addr`, `core_id`. After the Case-1 conversions the address args are gone, leaving `core_id` as the only named RTA on each reader. No common runtime args are set.
- **Don't "fix" the tile-sized NoC read.** The readers transfer `get_tile_size(...)` bytes (4096 for UINT32) from `page_id = 0` of what validation guarantees is a rank-1 ROW_MAJOR tensor, so the transfer is larger than the tensor's page. The audit recorded this as a latent issue for the ops team (see the audit's Misc anomalies #2). It is **out of port scope** — keep the size expression semantically identical, translated only per whitelist rule 7. Changing it would be a functional change.

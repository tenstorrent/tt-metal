# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard`

Single device operation, `ttnn::prim::ReshardDeviceOperation` (new-infra device op; `program_factory_t` is a `std::variant` of 8 factory instantiations across 5 factory templates). All eight are on the `descriptor` concept (`create_descriptor()` → `ProgramDescriptor`). Audited together as one porting unit (one device op, shared kernels).

- **`ReshardDeviceOperation`**
  - `ReshardSameWidthFactory<local_is_output=true>` / `<false>` (`reshard_program_factory_same_width.cpp`)
  - `ReshardSameHeightFactory<local_is_output=true>` / `<false>` (`reshard_program_factory_same_height.cpp`)
  - `ReshardGenericFactory` (`reshard_program_factory_generic.cpp`)
  - `NdReshardCopyPagesFactory` (`nd_reshard_program_factory_copy_pages.cpp`)
  - `NdReshardCopyLocalShardFactory<local_is_input=true>` / `<false>` (`nd_reshard_program_factory_copy_local.cpp`)

**Kernels used** (all Device-2.0-idiom):
- Own directory (`reshard/device/kernels/`): `nd_reshard_copy_local_shards.cpp`, `nd_reshard_copy_pages_reader.cpp`, `nd_reshard_copy_pages_writer.cpp`.
- In-family shared pool (`data_movement/sharded/device/kernels/dataflow/`): `reshard_reader.cpp`, `reshard_reader_diff_width.cpp`, `reshard_same_width_reader.cpp`, `reshard_same_width_writer.cpp`, `reshard_same_height_reader.cpp`, `reshard_same_height_writer.cpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `6a16e3bf8d8 2026-07-20 recipe: reframe migration-guide 'not yet available' as surface-maturation`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `ReshardDeviceOperation` → SameWidth(×2), SameHeight(×2), Generic, NdCopyPages, NdCopyLocalShard(×2) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — all 9 referenced kernels are Device-2.0 compliant |
| *Prereqs* — Cross-op escapes | Ok — kernel includes are `api/*` (LLK) only; no function-call escape |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (fixed-count CTAs; no `get_compile_time_arg_val(i)` loop) |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes (all 8 factory rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No (confirmed: no `compute_program_hash` override) |
| *TTNN Readiness* — Runtime-args update | No (confirmed: no `get_dynamic_runtime_args` / `override_runtime_arguments`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (confirmed: `reshard_nanobind.cpp` binds no descriptor internals) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none (no host-folded offset) |
| *Port work* — Tensor bindings (per binding) | Case 1 (ND factories, via `TensorAccessor`) · Case 2 (legacy factories, raw) · clean (borrowed-DFB locals) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (all 4 accessor sites are 2-arg) |
| *Port work* — CB endpoints | multi-binding (legacy borrowed CBs) · legal (NdCopyPages staging CB) · no CB (NdCopyLocalShard) |

**CB endpoints** are dispositions, not gates. Recorded per `(CB, config)` below.

## Result

**GREEN → brief issued.** All five gates clear (Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓). Port brief written to `METAL2_PORT_BRIEF.md`. The port is a mechanical CB→DFB / typed-binding rewrite: the kernels are already on Device-2.0 idioms (`Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem`, `AllocatorBank`, `UnicastEndpoint`), so no prereq work stands between this op and the port.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. All 8 factory rows (readiness sheet lines 68–75) read `Is able to port? = yes`, each with `Concept=descriptor`, `Custom hash=no`, `Runtime-args update=no`, `Pybind descriptor=no`, `Smuggled pointer=no`, `Is safe to port?=yes`. Cross-check clean:
  - `Concept=descriptor` — confirmed: every factory's only method is `create_descriptor(...) → ProgramDescriptor` (the 5 `*_program_factory_*.hpp` headers).
  - `Custom hash=no` — confirmed: no `compute_program_hash` in the op.
  - `Runtime-args update=no` — confirmed: no `get_dynamic_runtime_args` / `override_runtime_arguments`.
  - `Pybind descriptor=no` — confirmed: `reshard_nanobind.cpp` exposes no `create_descriptor` / device-op internals.
  - Cross-column invariants hold (`Op-owned tensors` blank on a `descriptor` row; `Runtime-args update=no`).
- **Device 2.0 (every kernel used):** GREEN. All 9 referenced kernels use Device-2.0 objects throughout — `Noc` (`noc.async_read/write`, barriers), `DataflowBuffer` with the `.get_write_ptr()`/`.get_read_ptr()`/`.reserve_back()`/`.push_back()`/`.wait_front()`/`.pop_front()` **methods**, `TensorAccessor`, `CoreLocalMem`, `AllocatorBank<bank_type>`, `UnicastEndpoint{}`. No raw `noc_async_*`, no `get_noc_addr_from_bank_id`, no `InterleavedAddrGen`/`ShardedAddrGen`, no CB-index free-function holdovers (`get_write_ptr(cb)` / `get_read_ptr(cb)` / `get_tile_size(cb)` / `get_local_cb_interface(cb)`). No donor kernels (see Out-of-directory coupling — all kernel includes are `api/*` LLK/HAL).
- **Feature compatibility:** all Appendix A entries N/A — no signal fires in host code, factories, descriptors, or kernels.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `.global_circular_buffer` field, no `remote_index`/`remote_cb` idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset` / `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor(…, offset, …)` |
  | GlobalSemaphore | N/A | no `GlobalSemaphore` type / `CreateGlobalSemaphore`; op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is fixed (`Tensor input` + `optional<Tensor>`); kernels read CTAs at fixed offsets, no `get_compile_time_arg_val(i)` loop |

- **CB endpoints (GATE-free):** classified per `(CB, config)`:
  - `ReshardGenericFactory` — CB **c_16** (borrowed from `output_buffer`, `reshard_program_factory_generic.cpp:699`): **multi-binding**. The one kernel source (`reshard_reader.cpp` / `reshard_reader_diff_width.cpp`) is instantiated as *both* a Reader-config and a Writer-config kernel on the same core range (`:710–722`), each raw-writing the borrowed CB via `dfb.get_write_ptr()` (`reshard_reader.cpp:30`) → **2 raw writers per node**.
  - `ReshardSameWidthFactory<true>` — CB **c_0** (borrowed from `local_buffer`, `:104`): **multi-binding** (2 raw writers/node — reader source `reshard_same_width_reader.cpp` on both configs, `:37`). CB **c_1** scratch (`buffer=nullptr`, `:113`, config: *unaligned && local_is_output*): **multi-binding** — each of the 2 instances writes and reads the scratch (`reshard_same_width_reader.cpp:40–41`).
  - `ReshardSameWidthFactory<false>` — CB **c_0** (borrowed from `local_buffer`): **multi-binding** (2 raw readers/node — writer source `reshard_same_width_writer.cpp:36`). No scratch CB (scratch is gated on `local_is_output`).
  - `ReshardSameHeightFactory<true>` — CB **c_0** (borrowed from `local_buffer`, `:78`): **multi-binding** (2 raw writers/node — `reshard_same_height_reader.cpp:31`).
  - `ReshardSameHeightFactory<false>` — CB **c_0** (borrowed from `local_buffer`): **multi-binding** (2 raw readers/node — `reshard_same_height_writer.cpp:31`).
  - `NdReshardCopyPagesFactory` — CB **c_0** (scratch staging, `buffer=nullptr`, `:65`): **legal (1, 1)**. Distinct sources: reader (`nd_reshard_copy_pages_reader.cpp`, FIFO producer — `reserve_back`/`push_back`) + writer (`nd_reshard_copy_pages_writer.cpp`, FIFO consumer — `wait_front`/`pop_front`), both on the full grid.
  - `NdReshardCopyLocalShardFactory<*>` — **no CB** (the factory pushes none; the two DM kernels copy directly through `TensorAccessor`, no staging buffer).

  Nothing here blocks the port. The multi-binding shape is the *dominant* pattern here (see Recipe notes 2): every legacy factory splits its work across two co-resident instances of one kernel source (BRISC/Reader + NCRISC/Writer config), so every borrowed CB is touched by two same-kind endpoints on each node.
- **Offset base pointers:** GREEN — no address RTA folds a host-side offset into its base. The 4 `input_buffer->address()` sites in `reshard_program_factory_generic.cpp:746,754,764,774` are passed as the `input_addr` scalar to the `get_runtime_args_for_given_ranges*` helpers, but that scalar is then **overwritten** by the `input_buffer` `Buffer*` binding at RTA index `grid.x+grid.y` (`:781–798`), so the delivered base is a clean `Buffer*` binding and the numeric address is discarded. All device-side offsetting (`input_shard_addr + addr_offset`) is computed kernel-side from *separate* stride args, not host-folded. The same_width/same_height factories deliver the remote base via `Buffer*` binding plus separate offset args (added kernel-side). Not in the offset-base triage tables (`2026-07-19_offset_base_pointers.md`), consistent with a clean scan (no fold, op not in tables → clean).
- **TensorAccessor 3rd argument:** GREEN — no site. All four `TensorAccessor(...)` constructions (in `nd_reshard_copy_local_shards.cpp:44,45`, `nd_reshard_copy_pages_reader.cpp:26`, `nd_reshard_copy_pages_writer.cpp:26`) pass exactly 2 args (`args, base_addr`); the legacy kernels construct no `TensorAccessor` at all. Not in the 3rd-arg triage table (`2026-07-06_tensor_accessor_3rd_arg_triage.md`; only the unrelated CCL `slice_reshard_async` appears there).

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per factory, per binding). Every base is delivered today via a `Buffer*` binding (RTA or common-RTA) or a borrowed-memory CB — **none** is a raw `->address()`-in-RTA value, so none is the fast-path-cache silent-stale hazard. The port replaces each with a typed `TensorParameter`/`TensorBinding`.
  - `ReshardGenericFactory` — **input**: Case 2 (raw; kernel uses `input_shard_addr` directly in `noc.async_read({.noc_x,.noc_y,.addr=input_shard_addr+off})`). **output**: clean (borrowed-DFB c_16, `borrowed_from` output).
  - `ReshardSameWidthFactory<*>` — **remote** (input if `local_is_output`, else output): Case 2 (raw; `AllocatorBank` + `bank_id` + `src_addr/dst_addr` base). **local**: clean (borrowed-DFB c_0). Scratch c_1 is not a tensor binding.
  - `ReshardSameHeightFactory<*>` — **remote**: Case 2 (raw; `AllocatorBank` + `bank_id` + base). **local**: clean (borrowed-DFB c_0).
  - `NdReshardCopyPagesFactory` — **input**: Case 1 (`TensorAccessor(args_src, base)`). **output**: Case 1 (`TensorAccessor(args_dst, base)`). Scratch c_0 is not a tensor binding.
  - `NdReshardCopyLocalShardFactory<*>` — **input**: Case 1. **output**: Case 1 (both fed to `TensorAccessor`).
- **TensorParameter relaxation:** none (sheet `none` on all rows; no custom hash to reconcile).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** multi-binding advanced-option flag on the legacy borrowed CBs (per `(CB, config)` above); `NdReshardCopyPagesFactory` c_0 is legal; `NdReshardCopyLocalShardFactory` has no CB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** all the legacy borrowed CBs are multi-binding via the *visible* face — two co-resident instances of one kernel source (Reader-config + Writer-config splitting work across BRISC/NCRISC). No hidden semaphore-gated second writer was found (the op uses no semaphores). The scratch CB c_1 (SameWidth`<true>`, unaligned) is a per-instance write+read self-loop, doubled by the 2 instances.
- **Cross-op / shared kernels:** the 6 legacy kernels are borrowed (file-path instantiation) from the in-family parent pool `data_movement/sharded/device/kernels/dataflow/`; **no other op** instantiates those shared paths (quasar/reshard has its own copies). Their Metal 2.0 rewrite is a single shared rewrite, but the port-together set is just this op.
- **RTA varargs:** all 6 legacy kernels consume a variable-length RTA block via a loop-indexed read — use the vararg mechanism, don't try to name each:
  - `reshard_reader.cpp:35–80` / `reshard_reader_diff_width.cpp:35–101` — `get_arg_val<uint32_t>(arg_index++)` in nested range/block loops.
  - `reshard_same_width_reader.cpp:30` / `reshard_same_width_writer.cpp:29` and `reshard_same_height_reader.cpp:25` / `reshard_same_height_writer.cpp:25` — `args = (tt_l1_ptr uint32_t*)get_arg_addr(N)` then `args[args_idx++]` in counted loops.
  - The 3 ND kernels read only fixed-index args — not varargs.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Function-call escape:** ✓ clean. Every referenced kernel's `#include`s are `api/*` (tt_metal LLK/HAL — class 1, no concern) plus `<cstdint>`/`<stdint.h>`. No cross-op or shared-pool *function* includes; no donor call-shape analysis needed.
  - **Host-side helper:** `ReshardSameHeightFactory` calls `ttnn::operations::data_movement::detail::compute_width_sharding_reshard_segments` from `sharded/sharded_common.hpp` — in-family, host-side (not a kernel), does not gate.
  - **Borrowed kernel files (file-path instantiation):** reshard owns its 3 ND kernels; it borrows all 6 legacy kernels from the in-family parent pool `data_movement/sharded/device/kernels/dataflow/`. Broadly-shared? No — grep of that exact path shows only this op's factories. (`experimental/quasar/reshard` maintains its own kernel copies under `experimental/quasar/reshard/device/kernels/`, so it is *not* a co-borrower.)
- **TTNN factory analysis:** current concept `descriptor` on all 8 factory rows; no op-owned tensors; no custom hash; no custom `override_runtime_arguments`; no pybind `create_descriptor`; `Is safe to port?=yes` (no smuggled pointer). Target concept `MetalV2FactoryConcept` (no op-owned tensors).

## Misc anomalies  *(team-only, non-gating; route to the ops team — not porter work)*

- **Unreachable code in `is_valid_for_legacy_reshard`** — `reshard_device_operation.cpp:39` is an unconditional `return out_mem_config.buffer_type() == BufferType::L1;`, which strands the entire `if (input_tensor.layout() == Layout::ROW_MAJOR) { … }` block at `:41–50` as dead code. It reads like an early return was inserted above pre-existing ROW_MAJOR handling; if that ROW_MAJOR shard-shape check was meant to run, the selection logic has a latent behavior bug. Not porter work (the port preserves behavior), but the ops team should decide whether `:41–50` should execute.
- **Dead debug `DPRINT`** — `reshard_same_width_reader.cpp:46` (`DPRINT("addr: {}\n", addr);`) left in the unaligned read path; also commented-out `print_bf16_pages` calls at `:52–53,70`. Harmless, but stray debug output in a production kernel.
- **Dead-but-read RTA** — `num_output_pages` is read (`reshard_reader.cpp:24`, `reshard_reader_diff_width.cpp:24`) but never used in either kernel. The factory still computes and passes it (`get_runtime_args_for_given_ranges*` writes it at RTA slot `physical_core_coords.size()+1`). A dead runtime arg the ops team could drop.

## Recipe notes

1. **`Buffer*`-binding form is classified "Case 2" by the detection bullet, but the governing rule is kernel-side usage.** In [TensorParameter analysis], the `Buffer*`-binding-form bullet states "the kernel consumes a raw `uint32_t` base, so it is **Case 2**." Reshard's two ND factories deliver the base via a `Buffer*` binding (`emplace_common_runtime_args({input_buffer})`) *and* feed it straight into `TensorAccessor(args, base)` — which is **Case 1** by the two cases' own definitions (Case 1 = "feeds that base address into a `TensorAccessor` constructor"). I classified by kernel-side usage (Case 1 for the ND factories, Case 2 for the legacy ones). Suggest the bullet say the `Buffer*`-binding form is a *delivery* mechanism orthogonal to the Case 1/2 split (which is decided kernel-side), rather than hard-asserting Case 2 — the current wording would mislead an auditor into miscounting a `TensorAccessor`-fed `Buffer*` binding.
2. **The dominant multi-binding cause here isn't the "hidden second writer" the subject foregrounds.** [CB endpoints] frames multi-binding as uncommon and spends its hunt-guidance on face (a) — a raw co-fill by a *different* kernel, semaphore-gated. In reshard, *every* legacy borrowed CB is multi-binding for a plainer reason: one kernel source is instantiated twice (Reader-config + Writer-config) on the same core range to split work across BRISC/NCRISC, so the CB has two same-kind endpoints per node with no hidden writer and no semaphores. This is neither face (a) (not hidden, not a distinct source) nor cleanly face (b) (it's writers, not readers, in the SameWidth/SameHeight `<true>` cases). A short "two co-resident instances of one source → multi-binding" note would name the case the recipe's two faces don't quite cover, and would reassure an auditor that finding *many* multi-binding CBs (not the "uncommon" the subject implies) is expected for split-across-RISC ops.
3. **Minor:** the readiness sheet lists 8 rows for this op but the `Factory (variant)` column doesn't distinguish the template parameter (e.g. `ReshardSameWidthFactory` appears twice with identical cells for `<true>`/`<false>`). Not a problem for the gate (both clear), but a reader reconciling rows to `std::variant` members can't tell which row is which instantiation from the sheet alone. Non-blocking; noted only because I had to map by count.

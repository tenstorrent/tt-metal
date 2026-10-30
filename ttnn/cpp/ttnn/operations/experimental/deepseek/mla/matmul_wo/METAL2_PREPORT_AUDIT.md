# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo`

- **`MatmulWODeviceOperation`** (`device/matmul_wo_device_operation.hpp:16`) is the directory's only DeviceOperation, and it is its own factory. `create_descriptor` is declared on the device-op itself (`device/matmul_wo_device_operation.hpp:22-25`) and there is no `program_factory_t`. The framework detects this shape as `HasDirectDescriptor` (`ttnn/api/ttnn/operation_concepts.hpp:155-158`) and wraps it in a synthesized `DirectDescriptorFactory` (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170-213`).
  - The single factory is `MatmulWODeviceOperation::create_descriptor`, defined in `device/matmul_wo_program_factory.cpp:82-295`. The readiness sheet's label for this shape is `MatmulWODeviceOperation (single-descriptor)`, but the sheet's row still calls it `MatmulWOProgramFactory` (see the TTNN gate).
- **Kernels.** All five live in `device/kernels/`, are owned by the op, and are each referenced exactly once by the factory. There are no unreferenced kernel files.
  - `dm0.cpp` streams this bank's slice of the bf8 weight from DRAM into CB `c_0`. It runs on the 12 workers aligned to DRAM banks, with `ReaderConfigDescriptor` (RISCV_1, NoC 0).
  - `compute.cpp` runs `matmul_block` on the L1-resident input (borrowed CB `c_1`) against `c_0`, and writes 7-tile partials to `c_2`. Same 12 cores.
  - `dm1.cpp` fans each 7-tile partial out across the 7 collector cores, one tile per collector, each with a raw NoC write plus a remote semaphore increment. Same 12 cores, with `WriterConfigDescriptor` (RISCV_0, NoC 1).
  - `dm1_collector.cpp` waits on the semaphore and then publishes the 12 received partials in `c_3`. Runs on the 7 collector cores.
  - `compute_collector.cpp` sums the 12 partials for each output tile into the output (borrowed CB `c_4`). Runs on the 7 collector cores.
  - `matmul_wo_ring_common.h` holds the compile-time ring constants (the 12-core K split) and is included by all five kernels.
- **Host-side include from outside the directory:** `ttnn/cpp/ttnn/operations/ccl/common/host/moe_utils.hpp`, for `stringify`, which builds the `COLLECTOR_CORE_COORDS` define (`device/matmul_wo_program_factory.cpp:111-112`). It is host-only and creates no kernel coupling.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`. In practice the op runs only on Wormhole; see Misc anomalies.

**Recipe docs:** `` c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form ``

**Readiness sheet:** "Operations analysis", file ID `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`. The launching session fetched it at **2026-10-30 12:21 UTC** (`download_file_content` as `text/csv`, base64-decoded). This audit ran as a subagent and read that git-ignored local copy, `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/analyses/ttnn_op_porting_readiness.csv`. One row matches this op.

**Code basis:** branch `iwrosz/matmul_wo_2.0` at `1a5f6910e7b`, which is `origin/akertesz/op-porting-recipe` (`c41f7f4ef0a`) merged with `origin/main` (`012c19beaae`, 2026-10-30). This was read-only analysis; nothing was built or run.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo` |
| **Overall** | **RED** — no portable subset |
| **DOps / Factories** | `MatmulWODeviceOperation` → its own direct `create_descriptor`. One factory. The sheet's label for it, `MatmulWOProgramFactory`, is stale. |
| *Prereqs* — Device 2.0 (every kernel used) | **No** — RED, routed to the Device 2.0 team. Broad Device 1.0 confined to the NoC/semaphore layer of `dm0.cpp` and `dm1.cpp`: neither builds a `Noc` object, and there are 15 Device 1.0 call sites between them. The CB layer already uses `CircularBuffer` in all five kernels, and `dm1_collector.cpp`, `compute.cpp` and `compute_collector.cpp` are clean. These are not isolated holdovers. |
| *Prereqs* — Cross-op escapes | Ok. No borrowed kernel files, and no donor headers outside `tt_metal/*`. All five kernels are op-owned. (The full coupling inventory was skipped per the Red scoping rule.) |
| *Feature Support* — overall | GREEN — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok — no CTA is read at a variable index. (This is not an Appendix A entry; see Recipe notes.) |
| *TTNN Readiness* — `Is able to port?` (the gate) | The sheet reads `yes` and the cross-check of the primary columns is clean. **The gate is still RED because the sheet is broken (factory-set mismatch).** The row's `Factory (variant)` and `Factory definition path` name `MatmulWOProgramFactory` and `device/matmul_wo_program_factory.hpp`, both deleted by #57521 (`dc788a6a7ff`, 2026-09-29). Routed to the readiness-sheet owner as a two-cell relabel. |
| *TTNN Readiness* — Concept (current) | `descriptor`; sheet and code agree |
| *TTNN Readiness* — Execution Model | `SPMD`. Not a mesh-shaped concept, so `Secretly SPMD Workload?` does not apply (blank, correctly). |
| *TTNN Readiness* — Custom hash | No. There is no `compute_program_hash`, `attribute_values` or `to_hash`, so the default hash covers `operation_attributes_t{layer_id}`. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No. The factory before #57521 had one; the `ProgramDescriptor` migration removed it. |
| *TTNN Readiness* — Pybind `create_descriptor` | No. `matmul_wo_nanobind.cpp:12-30` binds only the free function. |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept`, from the sheet's `Porting Target` cell. (The TTNN porting shape subject was skipped.) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — Offset base pointer | None — GREEN. No offset is folded into a base on the host, and the op is not in the triage tables. |
| *Port work* — Tensor bindings (per binding) | Skipped — whole-op RED, no portable subset; re-audit on unblock |
| *Port work* — TensorAccessor 3rd arg | None. No kernel constructs a `TensorAccessor` at all. |
| *Port work* — CB endpoints | Deferred — re-evaluate after the Device 2.0 migration (skipped per the Red scoping rule) |

## Result

**RED.** The op is blocked on two gates:

- **Device 2.0**, in `dm0.cpp` and `dm1.cpp`. Routed to the **Device 2.0 team**.
- **TTNN factory concept**, because the readiness sheet's factory-set match fails. Routed to the **readiness-sheet owner** (Diego).

The op is RED as a whole and **has no portable subset.** It has one DeviceOperation with one factory, and `dm0.cpp` and `dm1.cpp` are always instantiated.

1. **Device 2.0 (the primary blocker).** Both DRAM-side data-movement kernels do all of their NoC work with Device 1.0 free functions and no `Noc` object. In `dm0.cpp` that is the DRAM weight stream, pipelined with transaction IDs. In `dm1.cpp` it is sending partials out to the collectors and incrementing their semaphores remotely. Fixing this means migrating the NoC layer of two performance-critical kernels on the Device 2.0 track; it is not a port-time change. **Path forward:** the Device 2.0 team migrates both kernels, and then the op is re-audited. Most of these calls have a direct `Noc`, `Semaphore` or `BankView` form. `dm1`'s completion protocol (posted writes tagged with a transaction ID) needs a decision first; see the Device 2.0 section below.
2. **The readiness sheet is out of step with the code (secondary; clears without touching the op).** The op's only row still carries the factory identity from before the `ProgramDescriptor` migration. Every content column that I cross-checked matches the current code and contradicts the code before the migration. So the fix is a two-cell relabel, not a re-evaluation. **Path forward:** set `Factory (variant)` to `MatmulWODeviceOperation (single-descriptor)`, which is the sheet's own label for this shape. Set `Factory definition path` to `ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/matmul_wo_program_factory.cpp`.

The other gate-bearing subjects are already clear: no Appendix A feature is in use, no address argument carries an offset folded in on the host, and no `TensorAccessor` passes a page-size argument. Once both blockers above are cleared, re-audit.

**Red scoping.** The Device 2.0 blocker is cleared on the **op-code side.** It rewrites `dm0.cpp` and `dm1.cpp`, which are exactly what the porter-only subjects describe. For example, if the migration uses Device 2.0's preferred `BankView(TensorAccessor)` form, the shape of the weight tensor's access changes. So the seven purely-informational subjects are **skipped**; they are listed under "Port-work summary". The sheet blocker is cleared elsewhere, but that doesn't change the decision, because the Device 2.0 blocker already forces a re-audit against changed code.

## Gate detail

### TTNN factory concept (`Is able to port?`)

**RED: spreadsheet-broken (factory-set mismatch), routed to the readiness-sheet owner.** `Is able to port?` itself reads `yes`, and every column that can block is clear.

The sheet has one row for this op. Values are quoted verbatim:

| Column | Sheet value | Code evidence | Check |
|---|---|---|---|
| `Device operation` | `MatmulWODeviceOperation` | `device/matmul_wo_device_operation.hpp:16` | ✓ |
| `Factory (variant)` | `MatmulWOProgramFactory` | No such type exists. It was deleted in `dc788a6a7ff` (#57521). | ✗ phantom (renamed) |
| `Concept` | `descriptor` | `create_descriptor(attrs, tensor_args, ret)` returns a `ProgramDescriptor` and takes no `MeshCoordinate` (`matmul_wo_device_operation.hpp:22-25`, `matmul_wo_program_factory.cpp:82-83`). No workload, mesh-workload or MetalV2 entry points exist. | ✓ |
| `Custom hash` / `Backdoor custom hash` | `no` / `no` | No `compute_program_hash`, `attribute_values` or `to_hash` anywhere in the op directory | ✓ |
| `Runtime-args update (get_dynamic_runtime_args)` | `no` | None | ✓ |
| `Override runtime args method?` | `no` | None | ✓ |
| `Pybind descriptor` | `no` | `matmul_wo_nanobind.cpp:12-30` only calls `ttnn::bind_function` on `ttnn::experimental::deepseek::mla::matmul_wo` | ✓ |
| `Smuggled pointer` | `no` | Tensor addresses reach the kernels as `Buffer*` RTAs (`matmul_wo_program_factory.cpp:272-274`), which are patched on a cache hit. They are not raw `->address()` values. | consistent (this column is not part of the cross-check) |
| `Known op issues` | (blank) | — | nothing blocks |
| `TensorParameter relaxation` | `none` | — | clears |
| `Op-owned tensors?` | `no` | `descriptor` concept | ✓ |
| `Secretly SPMD Workload?` / `Why secretly SPMD?` | (blank) / (blank) | Not a mesh-shaped concept | ✓ |
| `Op Classification` · `Execution Model` · `Porting Target` | `PD Op (pointer-patching)` · `SPMD` · `ProgramSpecFactoryConcept` | Derived columns; read, not vetted | — |
| `Is able to port?` | `yes` | Derived | — |
| `Factory definition path` | `…/device/matmul_wo_program_factory.hpp` | The file was deleted in `dc788a6a7ff`. The definition now lives in `device/matmul_wo_program_factory.cpp`. | ✗ stale |
| `Declared in` | `…/device/matmul_wo_device_operation.hpp` | — | ✓ |

The cross-column invariants hold: `get_dynamic_runtime_args` is `no`, and `Op-owned tensors?` is `no` on a `descriptor` row.

**The mismatch.** The factory-set match fails. The row names a factory, `MatmulWOProgramFactory`, declared in `device/matmul_wo_program_factory.hpp`. PR #57521 ("Experimental DeepSeek matmul and MoE gate ops to ProgramDescriptor", `dc788a6a7ff`, 2026-09-29) deleted both when it folded the factory into the device-op. Today the code's only factory is the device-op's own `create_descriptor`. Elsewhere the sheet labels that shape `<DeviceOperation> (single-descriptor)`, with the `.cpp` that holds the definition as the path. For example, the `moreh/moreh_arange` row reads `MorehArangeOperation (single-descriptor)` and `moreh_arange_program_factory.cpp`. Under the recipe this is a phantom (renamed) row, so the sheet counts as broken and the gate is RED.

**Why this is a relabel, not a re-evaluation.** The row's content was refreshed after #57521. Before that PR the factory was `create()` plus `override_runtime_arguments()` on a `CachedProgram` (`git show dc788a6a7ff^:ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/matmul_wo_program_factory.hpp`). That is the `legacy device-op` concept with an override. The row instead reads `descriptor`, `Override … = no`, and `PD Op (pointer-patching)`, all of which match today's code. Only the two identity cells lag. Fixing them needs no change to the op. See Questions 1 and Recipe notes 1.

### Device 2.0 (every kernel used)

**RED, routed to the Device 2.0 team.** The owning family is the op itself (`experimental/deepseek/mla/matmul_wo`). All five kernels belong to the op. None borrows a donor kernel or includes a donor header from outside `tt_metal/*`.

| Kernel | Verdict |
|---|---|
| `dm0.cpp` | **Not compliant.** It builds no `Noc` object, and its 6 NoC call sites (all of its DRAM traffic) use Device 1.0 functions. |
| `dm1.cpp` | **Not compliant.** It builds no `Noc` or `Semaphore` object, and its 9 call sites (all of its NoC and semaphore traffic) use Device 1.0 functions. |
| `dm1_collector.cpp` | Compliant: uses `CircularBuffer` (`:25`) and `Semaphore<>` (`:36`) |
| `compute.cpp` | Compliant: CB sync goes through `CircularBuffer` (`:31-33`). Its LLK calls take CB ids, but that is the compute API, not a Device 2.0 surface. |
| `compute_collector.cpp` | Compliant, same as `compute.cpp` (`:25-26`) |

`get_tile_size(cb_id)` at `dm0.cpp:53-55` and `dm1.cpp:38-40` is sanctioned and not flagged.

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `dm0.cpp` | 90 | `get_noc_addr_from_bank_id<true>(dram_bank_id, 0)`: the bank's NoC base address for the weight stream | None. No `Noc` is built. The weight's `TensorAccessorArgs` exist (`:34`), but no accessor is constructed from them. |
| `dm0.cpp` | 113 | `noc_async_read_one_packet_set_state<true>(dram_noc_addr, w_bytes_per_txn, vchannel)`, with a custom VC | None |
| `dm0.cpp` | 129 | `noc_async_read_set_trid(trid_to_issue)`. The trid rotates through 1..3, one per block. | None |
| `dm0.cpp` | 132 | `noc_async_read_one_packet_with_state_with_trid<false, true>(dram_noc_addr, w_dram_read_offset, slot_addr[…][…], trid_to_issue)`, with `skip_ptr_update=false` and `skip_cmdbuf_chk=true` | None. The destination is a raw slot address, precomputed from `cb_r2c_w.get_write_ptr()` (`:95`). |
| `dm0.cpp` | 142, 156 | `noc_async_read_barrier_with_trid(trid_to_wait)` | None |
| `dm1.cpp` | 60 | `get_noc_addr(x, y, local_collector_base_addr)`: the collector-core destination. The address comes from the local `cb_s2c_in2.get_write_ptr()` (`:57`). | None |
| `dm1.cpp` | 83 | `get_semaphore(reduce_semaphore_id)`: a raw semaphore address | None. No `Semaphore` is built. |
| `dm1.cpp` | 86 | `get_noc_addr(x, y, semaphore_addr)`: the remote semaphore's NoC address | None |
| `dm1.cpp` | 93, 116 | `noc_async_write_set_trid(semaphore_trid, /*noc=*/1)` sets a sticky write trid, first to `4` and finally back to `0` | None |
| `dm1.cpp` | 100 | `noc_async_write_one_packet_set_state</*posted=*/true>(collector_dst_base_addr[i], out_tile_size, /*noc=*/1, vchannel)` | None |
| `dm1.cpp` | 102 | `noc_async_write_one_packet_with_state</*posted=*/true>(src, dst)`. The NoC is implicit `noc_index`, which `WriterConfigDescriptor` makes 1, so it matches `:100`. | None. `src` comes from `cb_c2w_out.get_read_ptr()` (`:97`). |
| `dm1.cpp` | 103 | `noc_semaphore_inc</*posted=*/true>(partial_semaphore_noc_addr[i], 1, /*noc=*/1, vchannel)` | None |
| `dm1.cpp` | 115 | `noc_async_write_flushed_with_trid(semaphore_trid, /*noc=*/1)` | None |

**How big the work is.** This is broad Device 1.0, confined to the NoC/semaphore layer of two kernels. Neither kernel builds a `Noc` object, so this is not the isolated-holdover case: no site can be fixed by swapping in a wrapper method on one line. The CB layer is already on Device 2.0 in all five kernels. The mapping onto the Device 2.0 surface below uses `device_api_migration_guide.md`, `tt_metal/hw/inc/api/dataflow/noc.h` and `tt_metal/hw/inc/api/semaphore.h`.

- **`dm0` maps onto existing APIs.** The reads become `Noc` with `set_async_read_state<NocOptions::TXN_ID | NocOptions::CUSTOM_VC>`, `async_read_with_state<NocOptions::TXN_ID>` and `async_read_barrier<NocOptions::TXN_ID>`. The bank addressing becomes `BankView` over a `TensorAccessor` on the weight; the guide calls this "Addressing a Tensor by Bank" and discourages the `AllocatorBank<AllocatorBankType::DRAM>` fallback. Two points are not byte-identical, and both sit in the op's hot loop, so check performance after the migration:
  - Device 2.0 sets the trid in `set_async_read_state` (a sticky `NOC_PACKET_TAG`, `noc.h:277-279` and `:312-315`), and `async_read_with_state<TXN_ID>` requires the same trid (`noc.h:324-326`). Because `dm0` rotates its trid per block, it would need a fresh set-state call per block.
  - `async_read_with_state<TXN_ID>` calls `noc_async_read_one_packet_with_state_with_trid` with its default template arguments (`noc.h:360`), and the default is `skip_cmdbuf_chk = false` (`dataflow_api.h:2493`). So `dm0`'s `skip_cmdbuf_chk=true` cannot be expressed.
- **`dm1` maps only partly.** The writes become `Noc noc(1)` with `set_async_write_state<NocOptions::POSTED | NocOptions::CUSTOM_VC>` and `async_write_with_state<NocOptions::POSTED>` to a `UnicastEndpoint`. The semaphore increment becomes `Semaphore<> sem(reduce_semaphore_id); sem.up(noc, x, y, 1, vchannel)` (`semaphore.h:197-204`). Three pieces have **no one-to-one Device 2.0 form today**:
  - The **posted** semaphore increment. `Semaphore::up` issues a non-posted `noc_semaphore_inc` (`semaphore_dm_impl.h:253`).
  - The sticky write trid on stateful writes. `set_async_write_state` and `async_write_with_state` both `static_assert` against `TXN_ID` (`noc.h:543`, `:595`).
  - A flush of posted writes scoped to one trid. `async_writes_flushed<POSTED>` `static_assert`s against `TXN_ID` (`noc.h:763`).

  So migrating `dm1` needs either an addition to the Device 2.0 surface or a restructured completion wait, for example a posted-writes flush without a trid. That is a semantic choice for the Device 2.0 team to make with the op owner, not a mechanical rewrite. While deciding, pin down what `dm1.cpp:114-115` is meant to guarantee; the comment says "Ensure write and semaphore have left the core". The trid is set on `write_cmd_buf` (`dataflow_api.h:2542`), which the stateful writes use (`dataflow_api.h:1022-1023`), but the semaphore increments travel the atomic path.
- **Two cross-node addressing idioms in `dm1` must survive the rewrite.** The eventual Metal 2.0 port will also need to re-examine both.
  - The address `dm1` writes partials to on the collector cores is this core's own `cb_s2c_in2.get_write_ptr()` (`dm1.cpp:57-68`). This works only because CB `c_3` is one `CBDescriptor` spanning both the DRAM cores and the collector cores (`matmul_wo_program_factory.cpp:141`), so its L1 address is the same everywhere.
  - The remote semaphore is addressed by id from the DRAM cores (`dm1.cpp:83`), although the host allocates it only on the collector cores (`matmul_wo_program_factory.cpp:161-166`). Device 2.0's `Semaphore<>(id)` computes the same L1 offset from the id (`semaphore.h:58-59`), so this carries over on Gen1.
- **Test coverage for the migration is thin.** The only live test is `test_mla_wo_program_cache`, which runs on Wormhole only (`tests/ttnn/nightly/unit_tests/operations/experimental/test_mla_wo.py:481-519`). The accuracy and performance test, `test_mla_wo`, is skipped unconditionally by #44858 (`test_mla_wo.py:346-348`).

### Feature compatibility

**GREEN: no gate fired.** Every entry is N/A.

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer` call, `global_circular_buffer` field, `remote_index`, or "remote CB" idiom. `make_cb` (`matmul_wo_program_factory.cpp:60-78`) leaves `CBDescriptor::global_circular_buffer` at its `nullptr` default (`program_descriptors.hpp:86`). |
| CBDescriptor `address_offset` (non-zero) | N/A | `make_cb` never sets `address_offset`, which defaults to `0` (`program_descriptors.hpp:85`). The borrowed CBs `c_1` and `c_4` attach `buffer` at its base. The only grep hit is the kernel-side `bank_address_offset` argument of `get_noc_addr_from_bank_id` (`dm0.cpp:90`), which the entry's false-positive guard excludes. |
| GlobalSemaphore | N/A | The op uses only a plain `SemaphoreDescriptor` (`matmul_wo_program_factory.cpp:161-166`). |

### CB endpoints (GATE-free)

**Deferred: re-evaluate after the Device 2.0 migration.** The subject is skipped per the Red scoping rule (whole-op RED, no portable subset).

The re-audit should watch for two things:
- `c_3` is filled across nodes. `dm1` on the 12 DRAM cores writes partials with raw NoC writes into the collector cores' `c_3`, gated by the semaphore, and `dm1_collector` then `push_back`s them. This is the hidden-second-writer shape, but across nodes.
- `dm0` and `dm1` call `get_tile_size()` on CBs they otherwise never touch (`dm0.cpp:53`, `:55`; `dm1.cpp:38-39`). All of these are dead locals.

### Offset base pointers

**GREEN.** No address argument has an offset folded into its base on the host. The op is not in the tables of `2026-07-19_offset_base_pointers.md`, so this is the "no fold, not in the tables → clean" outcome. The only route by which the op's host code passes addresses is as `Buffer*` values; there is no `->address()` expression anywhere in it.

| Address argument | Host source | Kernel use | Fold? |
|---|---|---|---|
| RTA[2] `in_addr` (dm0, dm1, compute) | `Buffer*` `input_buffer` (`matmul_wo_program_factory.cpp:272`), which resolves to the bare `buf->address()` (`tt_metal/impl/program/program_descriptors.cpp:262`, `:268`) | Read and never used (`dm0.cpp:41`, `dm1.cpp:24`, `compute.cpp:22`) | no |
| RTA[3] `w_addr` | `Buffer*` `weight_buffer` (`:273`) | `dm0.cpp:87` computes `w_dram_read_offset = w_layer_offset + w_addr`. The per-layer offset (the `layer_id` CTA times `w_size_per_layer`, `dm0.cpp:84`) is added **in the kernel**, on a bare base. The result is then used raw with `get_noc_addr_from_bank_id` (`:90`, `:132`). Dead in `dm1.cpp:25` and `compute.cpp:23`. | no |
| RTA[4] `out_addr` | `Buffer*` `output_buffer` (`:274`) | Read and never used (`dm0.cpp:43`, `dm1.cpp:26`, `compute.cpp:24`) | no |
| CB `c_1` / `c_4` `buffer` | `input_buffer` / `output_buffer` (`:142-143`), with `address_offset` 0 | Borrowed-memory CBs | no (Type 3 does not apply) |
| Compile-time args | Three `TensorAccessorArgs(*buffer)` (`:147-152`), plus the named CTAs `layer_id`, `num_cores` and `reduce_semaphore_id` (`:154-158`) | — | No addresses |

### TensorAccessor 3rd argument

**N/A.** No kernel constructs a `TensorAccessor`. The kernels only declare `TensorAccessorArgs` (`dm0.cpp:33-35`, `dm1.cpp:16-18`, `dm1_collector.cpp:17-19`). So no accessor passes a page-size argument, and the subject never fires. The op is also not listed in `2026-07-06_tensor_accessor_3rd_arg_triage.md`.

## Port-work summary  *(mirrors the brief)*

**Not produced.** The op is RED as a whole with no portable subset, so no brief is issued. The following subjects were skipped under the Red scoping rule, to be run on re-audit once the blockers clear:

- **TTNN porting shape:** skipped. Whole-op RED, no portable subset; re-audit on unblock. The sheet's `Porting Target` today is `ProgramSpecFactoryConcept`.
- **TensorParameter relaxations:** skipped as a porter subject. The cell reads `none`, which clears, and is recorded under the TTNN gate.
- **TensorParameter analysis:** skipped. Whole-op RED, no portable subset; re-audit on unblock. Today the weight is read with raw arithmetic on a `Buffer*` base. If the Device 2.0 migration uses its preferred `BankView(TensorAccessor)` form, that classification changes, which is why it waits.
- **CB endpoints:** deferred until after the Device 2.0 migration (see Gate detail).
- **RTA varargs:** skipped. Whole-op RED, no portable subset; re-audit on unblock.
- **Out-of-directory coupling:** skipped. Whole-op RED, no portable subset; re-audit on unblock. The Device 2.0 gate's donor check found no out-of-directory kernel or donor header.
- **Incidental anomalies:** no pass was run (skipped under the rule). The items under Misc anomalies were noticed while running the gate-bearing subjects; see Recipe notes 3.

## Heads-ups  *(mirrors the brief)*

**Not produced.** See Port-work summary.

## Team-only

- **Out-of-directory coupling and donor shape:** skipped (Red scoping). For the record, the Device 2.0 gate found that the kernels include only `api/*` headers (tt_metal LLK, HAL and Device 2.0; no concern) and the op's own `matmul_wo_ring_common.h`. Nothing comes from `kernel_lib/`, `kernel/` or `kernel_helper_functions/`, or from an in-family or cross-family donor, and no kernel file is borrowed.
- **Relaxation candidates:** none, because there is no custom hash to mine.
- **TTNN factory analysis:**
  - The concept is `descriptor`, with the direct-descriptor shape on the device-op. The execution model is `SPMD`. There are no op-owned tensors.
  - There is no pybound `create_descriptor` and no other risky pybind; `matmul_wo_nanobind.cpp:12-30` binds only the public free function.
  - There is no custom hash. The default hash covers `operation_attributes_t{layer_id}` (`device/matmul_wo_device_operation_types.hpp:12-14`). `layer_id` is also a named CTA on all five kernels, so each layer gets its own program-cache entry.
  - There is no `get_dynamic_runtime_args` and no `override_runtime_arguments`. On a cache hit, the framework refreshes the `Buffer*` RTAs and `CBDescriptor::buffer` itself (`matmul_wo_program_factory.cpp:137`).
  - The target concept is `ProgramSpecFactoryConcept`.

## Misc anomalies  *(team-only, non-gating; noticed while running the gate-bearing subjects, not a full pass)*

- **The Wormhole 12-bank geometry is hardcoded, and the host does not check it.**
  - **Kernel side:** the K split is a compile-time table for exactly 12 cores (`matmul_wo_ring_common.h:20`, `:40-53`), whose entries sum to `NUM_W_TILES_H = 512` (`:26`). It is indexed by each core's `dram_bank_id` RTA (`dm0.cpp:62`, `dm1.cpp:47`, `compute.cpp:40`, `:75`).
  - **Host side:** the ring is sized from the device (`num_cores = dram_bank2core_coords.size()`, `matmul_wo_program_factory.cpp:85-89`). That count becomes the `num_cores` CTA, which sets how many partials the collectors wait for and sum (`dm1_collector.cpp:37-44`, `compute_collector.cpp:48-61`) and the partial stride (`dm1.cpp:66`).
  - **Validation:** `validate_on_program_cache_miss` checks only `rank >= 2` (`matmul_wo_device_operation.cpp:11-17`). Nothing checks the bank count, the shapes baked into the kernels and CB sizes (K = 512 tiles, N = 28 tiles, M = one tile row), the dtypes, or the memory layouts.
  - **Consequence:** on a device whose DRAM-bank-to-worker map does not have 12 entries, the op would sum only part of K and raise no error.
  - **Tests:** they already treat the op as Wormhole-only. `test_mla_wo_program_cache` skips other architectures with "matmul_wo's DRAM bank map is the Wormhole 12-bank ring" (`test_mla_wo.py:483-484`). `test_mla_wo` is skipped outright by #44858 ("stable PCC assertion failure in test_mla_wo on nightly blackhole P150b", `:346-348`). That is consistent with this anomaly, but the audit did not confirm it is the cause.
  - **Route:** ops team. A host-side `TT_FATAL` on the bank count and the shapes would turn a silent wrong answer into an error.
- **Dead runtime and compile-time arguments.**
  - **Shared RTA list:** `dm0`, `dm1` and `compute` all receive the same five RTAs (`matmul_wo_program_factory.cpp:268-277`).
    - `in_addr` and `out_addr` are read but never used by any of the three (`dm0.cpp:41`, `:43`; `dm1.cpp:24`, `:26`; `compute.cpp:22`, `:24`).
    - `w_addr` is used only by `dm0` (`:87`) and is dead at `dm1.cpp:25` and `compute.cpp:23`.
    - `vchannel` is dead in `compute` (`:21`).
  - **Collector RTA:** `core_id` (`:281-286`) is never read by `dm1_collector`, and `compute_collector` reads it without using it (`:19`).
  - **Unused `TensorAccessorArgs`:** all three tensors' `TensorAccessorArgs` are appended to every kernel's CTAs (`:147-152`), but no kernel builds a `TensorAccessor` from them.
  - **`layer_id`:** it is a CTA on all five kernels, but only `dm0` reads it (`:84`). So every layer compiles all five kernels.
  - **Other unused items:** `K_TILES_PER_CORE_B` (`matmul_wo_ring_common.h:55-68`), and the `get_tile_size()` locals at `dm0.cpp:53`, `:55` and `dm1.cpp:38-39`.
  - **Cost:** the dead `Buffer*` slots are still registered as `BufferBinding`s and re-patched on every cache hit.
  - **Route:** ops team. They are harmless today.

## Questions for the user

1. **Should a stale factory label be a gate or a note?** The readiness row's content matches today's code. Only `Factory (variant)` and `Factory definition path` still name the factory as it was before #57521. I applied the recipe's factory-set-match rule literally: a phantom (renamed) row means the sheet is broken, which makes it a GATE routed to the sheet owner. If you or the sheet owner treat a lag that affects only the identity cells as non-gating, this gate clears with no change to the op. The op stays RED on Device 2.0 either way. The proposed cell values are `MatmulWODeviceOperation (single-descriptor)` and `ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/matmul_wo_program_factory.cpp`.

   *Outside this audit's scope, and not audited:* #57521 also migrated `experimental/deepseek/moe/moe_gate_mm`. Its row still reads `legacy device-op` / `MoEGateMMProgramFactory` / `moe_gate_mm_program_factory.hpp`. But `moe_gate_mm_device_operation.hpp:22` now declares `create_descriptor` directly, and that `.hpp` no longer exists. The sheet owner may want to refresh that PR's rows together.

## Recipe notes

1. **The factory-set match can't tell a stale label from a stale evaluation.** (This is about "Factory-set match" and the Routing list in *TTNN factory concept prerequisite*.)
   - **What happened here:** the per-column cross-check positively shows the row was re-evaluated after the rename. `Concept` and `Override runtime args method?` both changed to match the new code. So the rule's stated purpose, "a staleness check the per-column checks miss", isn't served. The literal rule still REDs the op over two identity cells.
   - **Suggestion:** when every cross-checked content column matches the current code (and contradicts the code from before the rename), let the auditor route the relabel as non-gating. Otherwise, state explicitly that staleness in the identity cells alone does gate.
   - **Missing documentation:** neither the recipe nor `ttnn_op_porting_readiness.md` documents the sheet's `<DeviceOperation> (single-descriptor)` label for `HasDirectDescriptor` device-ops. An auditor needs it to run the factory-set match on that shape; I found it by reading other ops' rows.
2. **The `Feature Support — Variadic-CTA` row is a leftover.** It still appears in the status-summary template, but Appendix A has no Variadic-CTA entry. The *RTA varargs* section also says CTA varargs port onto `KernelAdvancedOptions::compile_time_varargs` and do not gate. I filled the row with `Ok`. Suggest dropping or redefining it.
3. **Incidental anomalies don't fit the Red scoping rule.**
   - **Why skipping doesn't fit:** the subject is "note what you happen to see". It is team-only and routes to the ops team whether or not a port follows, so its value doesn't go stale the way porter detail does.
   - **What happened here:** the gate-bearing subjects (the line-by-line Device 2.0 scan and the offset scan over every address RTA) surfaced the anomalies anyway. Skipping the subject would have thrown away a Wormhole-only, silently-wrong-result risk. So I recorded what I saw and labelled it as no full pass.
   - **Suggestion:** exempt Incidental anomalies from the skip.
4. **The Device 2.0 sizing vocabulary doesn't fit this case.**
   - **Neither bucket fits:** the two buckets (isolated CB-index holdovers, or broad Device 1.0) don't name what I found here. The CB layer is entirely on `CircularBuffer`, while the NoC/semaphore layer hasn't been touched. I called it "broad Device 1.0 confined to the NoC/semaphore layer".
   - **Surface gaps:** the recipe doesn't say what to record when the Device 2.0 surface itself lacks a one-to-one form for a legacy call. Here that means a posted `Semaphore::up`, `TXN_ID` on stateful writes, and a `POSTED` plus `TXN_ID` flush. I recorded these as sizing detail for the Device 2.0 team; a sentence in that section would help.
   - **CB-endpoints precondition:** "defer on a broadly Device-1.0 RED op" is ambiguous when the CB idioms are intact but the NoC idioms are not. That is moot here, because Red scoping skips the subject anyway.
5. **The *TensorParameter analysis* cross-check misfires on dead address RTAs.** The rule reads: "does the same buffer also appear in an RTA address argument? If yes, that buffer is consumed as a raw pointer → Case 2". That assumes the RTA is used. Here the input and output `Buffer*` RTAs are read and never used, while the kernels actually reach those tensors through the borrowed CBs `c_1` and `c_4`. Read literally, the rule would classify both as Case 2. Suggest qualifying "consumed". (The subject was skipped this time; this is noted for the re-audit.)
6. **This audit ran as a subagent.** *TTNN factory concept prerequisite* says "Do not fetch or cross-check in a subagent". This audit ran in a subagent, using the CSV the launching session fetched at 2026-10-30 12:21 UTC. The cross-check's stated reason, having the op's code in hand, still held. If the rule is only about the Drive connector's authorization, it could say so.

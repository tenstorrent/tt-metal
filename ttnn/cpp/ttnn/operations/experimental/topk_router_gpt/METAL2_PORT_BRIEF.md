# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/topk_router_gpt`

> Audit cleared all gates. One gate was cleared by user waiver; see below. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user waiver: readiness-sheet row is stale, see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section; hash from the `Port_Recipe` checkout)*

**TTNN gate waiver (record in the port report).** The live readiness sheet (fetched 2026-10-06) still describes this op as it was before PR #57409:

- `Concept` = `legacy device-op`
- factory `TopkRouterGptProgramFactory`, at a now-deleted `.hpp`
- `Is able to port?` = `yes (with PD step)`

The code is already on a direct `create_descriptor`, so the audit flagged the sheet as broken. On 2026-10-06 the user waived that gate, since the problem is only the out-of-date sheet. The sheet refresh is still owed to the readiness-sheet owner.

**Audited at:** `5ce297c508a`, which includes `8e04962ad72` (#58858, Device 2.0 barrier fix) and `16327cdfae5` (#58472, Blackhole P150 + partial batches). If `device/` has changed since then, stop and ask for a re-audit.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`, in the **direct-descriptor** shape. `create_descriptor` is a static member of `TopkRouterGptDeviceOperation` itself (`device/topk_router_gpt_device_operation.hpp:26-29`, body `device/topk_router_gpt_program_factory.cpp:44-373`), with no `program_factory_t`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. There is no `override_runtime_arguments`, and the header comment at `device_operation.hpp:22-25` explains why. The framework's binding refresh covers the cache hit.
- **Gate-cleared, confirmed absent:** a non-clearing `TensorParameter relaxation` (cell is `none`) and `get_dynamic_runtime_args` (absent). The op also has no custom hash and no pybound `create_descriptor`. The only nanobind binding is the user function (`topk_router_gpt_nanobind.cpp:29-58`), so no pybind line needs deleting.

## Construct — to do

**Introduce a factory struct (forced; `ttnn_factory.md` §3 "Give a direct-descriptor op a conventional program factory").** The `DirectDescriptorFactory` shim (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170`, gated by `HasDirectDescriptor`, `ttnn/api/ttnn/operation_concepts.hpp:158`) only recognizes `create_descriptor`, and has no `create_program_artifacts` counterpart. In `device/topk_router_gpt_device_operation.hpp`:

1. Nest `struct TopkRouterGptProgramFactory` (the pre-#57409 name) with `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);`.
2. Add `using program_factory_t = std::variant<TopkRouterGptProgramFactory>;`.
3. Remove the device-op-level `create_descriptor` and its comment (`:22-29`).

Keep the body in `device/topk_router_gpt_program_factory.cpp`. It is already listed in `topk_router_gpt/sources.cmake:12`, so no build edits are needed. Record this under Handoff points: the op arrived in the direct-descriptor shape. *(Check first that no `program_factory_t` has appeared since the audit.)*

**Keep the host-side geometry exactly as it is** (`program_factory.cpp:50-151`). This covers:

- the DRAM-bank → worker assignment (`get_optimal_dram_bank_to_logical_worker_assignment(RISCV_0_default)`)
- `cores_per_group = num_cores >= 12 ? 3 : 2`, `num_senders`, `required_cores`, and the `TT_FATAL` that checks them
- the NOC1 ring sort
- the collector at ring position `num_senders`
- the k-tile split

None of this is port work. It only changes what it feeds: specs instead of descriptors.

**ProgramSpec contents:** three `KernelSpec`s, all on `all_cores` (every DRAM-aligned core: 12 on WH, 8 on BH P150).

- **dm0** — `device/kernels/dm0.cpp`. DM on RISCV_1 / NOC_0 (`program_factory.cpp:237-240`).
- **dm1** — `device/kernels/dm1.cpp`. DM on RISCV_0 / NOC_1 (`:248-251`).
- **compute** — `device/kernels/compute.cpp`. `HiFi2`, `fp32_dest_acc_en = true`, `dst_full_sync_en = false`, `bfp8_pack_precise = false`, `math_approx_mode = false` (`:259-265`).
- **Named CTAs** (`:216-227`): `num_cores`, `num_groups`, `cores_per_group`, `num_senders`, `collector_physical_x`, `collector_physical_y`, `topk_k`, `k_padded`, `n_tiles`, `tile_size_bf16`. All three kernels get the same set today, so keep it that way. The positional `TensorAccessorArgs` block (`:203-212`) disappears into the bindings.
- **Per-node RTAs**, emplaced for the `required_cores` ring positions (`:319-366`). All three kernels read the same 20-slot block through one fixed `argidx++` run. Name every field after the kernel variable it unpacks into:
  - `dram_bank_id`, `vchannel`, `sem_partial_ready`, `is_sender`, `is_worker`, `is_collector`
  - `num_k_tiles`, `k_tile_offset`, `n_tile_id`, `worker_phys_x`, `worker_phys_y`, `sender_slot`, `worker_gather_slot`
  - `sem_topk_ready`, `aligned_page_size`
  - The five address slots `[2] [3] [4] [17] [18]` leave the RTA block and become tensor bindings (below). `sem_partial_ready` / `sem_topk_ready` become semaphore bindings (below).
  - Several fields are read and unused in some kernels, e.g. `dram_bank_id` and `vchannel` everywhere. That is an ops-team anomaly. **Carry the fields over; don't prune them in the port.**
- **Semaphores:** two `SemaphoreSpec`s on `all_cores`, initial value 0, both bound to **dm1 only**:
  - `sem_partial_ready` (legacy id 0, RTA `[5]`)
  - `sem_topk_ready` (legacy id 1, RTA `[16]`)

  They are created at `program_factory.cpp:269-274`. dm1 builds `Semaphore<> x(<rta>)` at `dm1.cpp:162, 222, 258, 299`; switch those to the `sem::` token. dm0 and compute read the two RTAs but never use them.
- No CRTAs and no defines.

**Tensor bindings** (per binding). Today every address arrives as a `Buffer*` RTA, which the framework patches on a cache hit. All five bindings are **Case 1** (via `TensorAccessor`): express each as a `TensorParameter` / `TensorBinding`, and have the kernel build `TensorAccessor(tensor::<name>)`. The address RTA and its `TensorAccessorArgs<…>` plumbing both go.

- `input` — RTA `[3]` (`program_factory.cpp:345`) → `dm0.cpp:61`. Bind on **dm0 only**.
- `weight` — RTA `[2]` (`:344`) → `dm0.cpp:62`. Bind on **dm0 only**.
- `bias` — RTA `[4]` (`:346`) → `dm0.cpp:99` (worker path). Bind on **dm0 only**.
- `indices_rm` (output 0) — RTA `[17]` (`:359`) → `dm1.cpp:343`. Bind on **dm1 only**.
- `weights_rm` (output 1) — RTA `[18]` (`:360`) → `dm1.cpp:344`. Bind on **dm1 only**.

Today the CTA order (input, weight, bias, indices_rm, weights_rm) differs from the RTA order (weight first). Binding by name makes that moot. Don't recreate the positional order anywhere.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** drop the redundant page-size arg at `dm1.cpp:343` and `dm1.cpp:344`. These are Class 2: interleaved L1 row-major outputs, and the value equals `aligned_page_size` (`k_padded × 2` bytes, constant per program). **Do not** set `dynamic_tensor_shape`.

After the drop, the `aligned_page_size` RTA (`[19]`, host `program_factory.cpp:296-297, 361`) has no consumer in any kernel. Remove it only if the port recipe's rules on orphaned args allow it, and record either outcome in the port report. The stale comment at `dm1.cpp:341-342` goes with the arg.

**CB endpoints** (the same in the WH and BH configs). Every legacy CB is Float16_b. Carry the sizes over exactly (`program_factory.cpp:153-200`). Only `c_2`'s depth varies, as `num_senders`.

| CB → DFB | Legacy cores | Pages × page size | Binding |
|---|---|---|---|
| c_0 `cb_weight` | all | `max_k_tiles` × tile | dm0 **PRODUCER**, compute **CONSUMER** |
| c_1 `cb_input` | all | `max_k_tiles` × tile | dm0 **PRODUCER**, compute **CONSUMER** |
| c_2 `cb_partial_recv` | all | `num_senders` × tile | dm1 **PRODUCER**, compute **CONSUMER** |
| c_3 `cb_local_out` | all | 1 × tile | compute **PRODUCER**, dm1 **CONSUMER** |
| c_4 `cb_bias` | workers | 1 × tile | dm0 **PRODUCER**, compute **CONSUMER** |
| c_5 `cb_index` | workers | 1 × tile | **self-loop: dm1 PRODUCER + CONSUMER**. Do **not** bind compute; it declares `cb_index` (`compute.cpp:84`) but never uses it. |
| c_6 `cb_topk_val` | workers | 1 × tile | compute **PRODUCER**, dm1 **CONSUMER** |
| c_8 `cb_gathered_val` | workers | 4 × tile | dm1 **PRODUCER**, compute **CONSUMER** |
| c_9 `cb_gathered_ind` | workers | 4 × tile | dm1 **PRODUCER**, compute **CONSUMER** |
| c_10 `cb_intermed_val` | collector | 2 × tile | **self-loop: compute** |
| c_11 `cb_intermed_ind` | collector | 1 × tile | **self-loop: compute** |
| c_12 `cb_softmax_mask` | collector | 1 × tile | dm1 **PRODUCER**, compute **CONSUMER** |
| c_13 `cb_softmax_tmp` | collector | 1 × tile | **self-loop: compute** |
| c_14 `cb_reduce_scalar` | collector | 1 × tile | **self-loop: compute** |
| c_15 `cb_bcast_scaler` | collector | 1 × tile | dm1 **PRODUCER**, compute **CONSUMER** |
| c_16 `cb_final_out` | collector | 2 × tile | compute **PRODUCER**, dm1 **CONSUMER** |
| c_19 `cb_dispatch` | collector | 1 page × `2·32·k_padded·2` B (non-tile) | **self-loop: dm1 PRODUCER + CONSUMER** |

There are no dead CBs and no multi-binding flags. On sender nodes, dm1 only *peeks* c_2 (`dm1.cpp:140`), which its PRODUCER binding covers. On non-collector workers, dm1 only peeks c_8/c_9 (`dm1.cpp:237, 246`), which is also covered.

## Watch for

- **CB endpoints (multi-binding):** none. The two cross-core raw writes are **remote**, semaphore-coordinated NoC writes into a slot the receiving core's dm1 owns, not hidden co-resident writers:
  - sender → worker c_2 (`dm1.cpp:151-158`)
  - worker → collector c_8/c_9 (`dm1.cpp:237-253`)

  Do not add a flag for them.
- **Cross-op / shared kernels:** none. All three kernel files are private to this op, and this factory is their only binder, so convert them in place. No `_metal2` fork is needed. Do not borrow anything from `experimental/quasar/`.
- **RTA varargs:** none. The `argidx++` runs (`dm0.cpp:30-50`, `dm1.cpp:88-108`, `compute.cpp:39-59`) are fixed positional plumbing; name every field. Every CTA is a named CTA. The `num_senders` loop in compute (`compute.cpp:156-158`) iterates over CB tiles, not args.
- **⚠ Uniform DFB L1 offsets are load-bearing. Keep one placement for every DFB.** dm1 reads its *own* write pointer and uses it as the NoC destination address on *another* core:
  - c_2: sender → worker (`dm1.cpp:140, 151`)
  - c_8/c_9: worker → collector (`dm1.cpp:237-238, 246-247`)

  The host comment at `program_factory.cpp:85-87` says so too. Legacy keeps the offsets uniform through allocation order. In Metal 2.0, DFB placement is *derived* from the bound kernels (`migration_guide.md`, "DFB placement is derived, not specified"). Every kernel here runs on every core, so a single work unit allocates every DFB on all 12 (WH) or 8 (BH) cores. The layout stays uniform. The cost is extra L1 on the sender and worker cores for the narrower-scoped legacy CBs, which is a footprint change only.

  **Do not** split the kernels into per-role work units to recover legacy's narrower CB placement unless you can show that c_2/c_8/c_9 land at identical L1 offsets on every role set. A broken offset gives wrong data with no assert. *(This is open question 2 in the audit; the user has not ruled on the extra footprint. Default to the single placement and call out the footprint delta in the port report.)*
- **Every named RTA must be set on every node** (`migration_guide.md`). The kernels sit on all `num_cores` DRAM-aligned cores (`program_factory.cpp:56, 234, 245, 256`), but RTAs are emplaced only for the `required_cores` ring positions (`:319`). The two sets are equal on WH (12) and on BH P150 (8).
  - Keep the kernel placement and the run-args node set identical. Do not add new asserts, since the port makes no functional changes.
  - Do **not** invent RTA values for extra cores, and do **not** shrink the placement to `required_cores`: that would change which cores host the DFBs that the uniform-offset trick needs. If the two sets ever differ, stop and raise it.
- **Two device configs, one code path.** `num_senders` decides three things, and all three must keep coming from one host value:
  - the `c_2` depth (`program_factory.cpp:158`)
  - the collector ring position (`:130`)
  - the kernels' wait/push counts (`dm1.cpp:219, 223, 226`; `compute.cpp:153, 156, 160`)
- **Kernels are already Device 2.0.** They use `Noc`, `CircularBuffer`, `Semaphore<>`, `TensorAccessor`, `CoreLocalMem` and `UnicastEndpoint`, so the kernel change is a binding-layer swap, not an idiom rewrite:
  - `CircularBuffer cb_x(cb_x_id)` → `DataflowBuffer` from the `dfb::` token
  - drop the `api/dataflow/circular_buffer.h` include (`dm0.cpp:13`, `dm1.cpp:23`, `compute.cpp:24`)
  - accessor from the `tensor::` token
  - `Semaphore<>` from the `sem::` token
  - named RTAs
- **Compute passes raw CB ids into LLK calls** (`compute_kernel_hw_startup`, `matmul_block`, `pack_tile`, `transpose_tile`, `reduce_*`, `add_reuse_dest_*`, `*_bcast_*`, `copy_tile`). Those `constexpr auto cb_*_id = tt::CBIndex::c_N` constants (`compute.cpp:62-77`) become the `dfb::` tokens, which convert implicitly to `uint32_t`. Leave the LLK call sites alone.
- **Compute names c_3 in `compute_kernel_hw_startup` on every node** (`compute.cpp:104`), but FIFO-produces it only on senders. Compute is still c_3's PRODUCER everywhere, so this needs no extra binding.
- **Preserve the collector's raw slot-0 fill and its late reserve.** On the collector, dm1 `reserve_back(num_groups)`s c_8/c_9 (`dm1.cpp:276-277`) *after* the other workers may already have NoC-written slots 1–3. It then raw-copies its own tile into slot 0 (`:279-292`) and pushes only after the semaphore wait (`:299-304`). Carry this over verbatim; don't "fix" the ordering.
- **Verification.** Run `tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_router_gpt.py` on **both WH and BH P150**. #58472 validated 41 tests on each. The file skips devices with fewer than 8 DRAM-aligned workers (`:42`).
  - `test_topk_router_gpt_program_cache` (`:180`) is the cache-hit check for the new tensor bindings. It moves every allocation between iterations and asserts exactly 1 new cache entry (`:206`).
  - `test_topk_router_gpt_fresh_buffers_and_trace` (`:256-316`) covers batches 1/9/32, DRAM and L1 inputs, fresh buffers, and trace replay.

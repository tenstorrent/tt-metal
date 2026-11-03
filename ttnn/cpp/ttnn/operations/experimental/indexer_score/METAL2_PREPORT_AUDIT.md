# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/indexer_score`

- **`IndexerScoreDeviceOperation`** — the only DeviceOperation in the directory. `program_factory_t = std::variant<IndexerScoreProgramFactory, RingIndexerScoreDsaMeshWorkloadFactory>` (`device/indexer_score_device_operation.hpp:27-28`). `select_program_factory` picks by `attrs.has_fused_ring()` (`device/indexer_score_device_operation.cpp:378-384`).
  - `IndexerScoreProgramFactory` (`device/indexer_score_program_factory.{hpp,cpp}`) is the classic, unfused path behind `ttnn.experimental.indexer_score_dsa` and `ttnn.experimental.indexer_score_msa`. It is imperative `host_api` code: `CreateProgram` / `CreateCircularBuffer` / `CreateKernel` / `SetCommonRuntimeArgs`, plus `create_mesh_workload` + private `create_at` + `override_runtime_arguments`.
  - `RingIndexerScoreDsaMeshWorkloadFactory` (`device/ring_indexer_score_dsa_program_factory.{hpp,cpp}`) is the fused-ring path behind `ttnn.experimental.ring_indexer_score_dsa`. A hand-written `MeshWorkloadFactoryConcept` shell (`.hpp:45-89`) wraps `DescriptorMeshWorkloadAdapter<RingIndexerScoreDsaProgramFactory>`. The inner factory builds one `ProgramDescriptor` per coordinate through `create_workload_descriptor` (`.hpp:37-43`, `.cpp:718-731`) and co-schedules the ring-attention all-gather (AG) inside the same program.
- **Kernels.** Both factories bind the same three op-owned kernels: `device/kernels/reader_indexer_score.cpp`, `writer_indexer_score.cpp`, and `compute_indexer_score.cpp`. The flavour is selected by compile-time args (`fused_ring_enabled`, `block_pool`, `fuse_single`, ...). The ring factory also file-path-instantiates two borrowed AG kernels, two instances each: `experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_{reader,writer}.cpp`. No unreferenced kernel files exist in the op directory.
- The op is **Blackhole-only**: `validate_on_program_cache_miss` fails on anything else (`device/indexer_score_device_operation.cpp:490-491`).

**Scope:** TTNN op, Gen1 (WH/BH) target, within the scope of `audit/metal2_audit.md` (this op validates as BH-only).

**Recipe docs:** ``c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form``

**Readiness sheet:** fresh CSV fetched by the orchestrating session on 2026-11-03 13:06 UTC (28 columns, 477 data rows). This op has two rows, one per factory. Columns are referenced by header name below.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/indexer_score` |
| **Overall** | **RED**. RED at op level, with **no portable subset**: each factory is blocked by a different gate. |
| **DOps / Factories** | `IndexerScoreDeviceOperation` → `IndexerScoreProgramFactory`, `RingIndexerScoreDsaMeshWorkloadFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **No, for the ring factory only.** The residuals sit in the borrowed AG kernels and a shared CCL helper. They are isolated, not broad: GlobalSemaphore-bound legacy primitives (structural, cleared together with GlobalSemaphore support) plus 2 actionable `noc_async_writes_flushed()` holdovers in `ccl/common/kernels/minimal_ccl_common.hpp`. **Classic factory: Yes.** The op's own kernels are Device 2.0 for both factories. |
| *Prereqs* — Cross-op escapes | **issue** (FYI, non-gating; ring factory). ⭐ A `CircularBuffer&` parameter in the borrowed AG reader's prefetch helper. ⚠ Semaphore ids delivered as RTAs to two cross-family donors (`RingSDPAOpReceiver`, `OpSignaler`). ✗ A Shape-2 metadata reader. The factory also assembles its program through another op's `ProgramDescriptor` helper. Classic factory: deferred. |
| *Feature Support* — overall | **RED.** GlobalSemaphore is used by the ring factory only. The classic factory is all N/A. |
| *Feature Support* — Variadic-CTA | Ok. Not an Appendix A entry, and no kernel reads CTAs at a varying index. |
| *TTNN Readiness* — `Is able to port?` (the gate) | Classic: `yes (with PD step)` → RED, legacy. Ring: `yes` → cleared. |
| *TTNN Readiness* — Concept (current) | Classic: `legacy (MeshWorkload)`. Ring: `WorkloadDescriptor`. |
| *TTNN Readiness* — Execution Model | Classic: `SPMD (+ per-device args)` (`Secretly SPMD Workload?` `yes`, `Per-coord RTAs`, cross-checked). Ring: `MPMD` (`Secretly SPMD Workload?` `no`, cross-checked). |
| *TTNN Readiness* — Custom hash | Yes (not a gate; the port leaves it intact): `IndexerScoreDeviceOperation::compute_program_hash` (`device/indexer_score_device_operation.hpp:34`, `.cpp:386-466`). It is shared by both factories. There is no backdoor hash. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | Classic: present, but as part of the legacy-concept signature (`indexer_score_program_factory.hpp:40-44`, `.cpp:433-465`), so it gates via Concept. Ring: Yes (not a gate; the porter translates it): `ring_indexer_score_dsa_program_factory.hpp:82-86`, `.cpp:775-969`. |
| *TTNN Readiness* — Pybind `create_descriptor` | No. `indexer_score_nanobind.cpp` binds only `IndexerScoreProgramConfig` and three `bind_function`s. |
| *TTNN Readiness* — Op-owned tensors | No. The ring `WorkloadDescriptor` fills only `programs` (`ring_indexer_score_dsa_program_factory.cpp:723-730`). |
| *TTNN Readiness* — Target concept | `MeshWorkloadSpecFactoryConcept` on both rows. The classic value is provisional and should be re-read after its PD migration. |
| *Port work* — Offset base pointer | none (both factories) |
| *Port work* — Tensor bindings (per binding) | Ring: all 8 bindings are **Case 1**, delivered as `Buffer*`. Classic: skipped (PD-migration RED; re-audit). |
| *TTNN Readiness* — TensorParameter relaxation | `none` on both rows → clears. |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2) at 5 sites: reader `:515`, `:699`, `:700`, `:701`; writer `:269`. |
| *Port work* — CB endpoints | Ring: 9 legal 1P+1C instances, 5 self-loops (2 compute, 3 DM), and 4 conditional DFBs. No multi-binding and no dead CB. Classic: skipped. |

### Per-factory verdicts

| Gate | `IndexerScoreProgramFactory` (classic) | `RingIndexerScoreDsaMeshWorkloadFactory` (fused ring) |
|---|---|---|
| Device 2.0 | GREEN | **RED**: donor AG / CCL code only (isolated; see Gate detail) |
| Feature compatibility | GREEN (every entry N/A) | **RED**: GlobalSemaphore |
| TTNN factory concept | **RED**: `yes (with PD step)` (legacy) | GREEN: `yes` |
| Offset base pointers | GREEN | GREEN |
| TensorAccessor 3rd argument | GREEN (Class 2 drops) | GREEN (Class 2 drops) |
| Which side the blocker clears on | **op-code side** (a `ProgramDescriptor` migration rewrites the factory) | **elsewhere** (a GlobalSemaphore feature landing; see the note below) |
| Purely-informational subjects | **skipped**; re-audit after the PD migration | **run** (below) |

**Scoping judgment (per the Red-outcome exception).** The ring factory's dominant blocker is an Appendix A feature that clears without touching this op's code. Its Device 2.0 residuals are of two kinds. Most are structural and clear together with that feature. The rest are two flush lines in a donor CCL helper, which do not rewrite any indexer code these subjects describe. I therefore judged the ring factory to be **elsewhere** and ran the seven informational subjects for it. The classic factory's blocker is a PD migration, which is **op-code side**, so its informational subjects are skipped (see *Skipped subjects*).

## Result

**RED → blocked.** RED at op level, with **no portable subset**. Both factories are blocked, independently:

1. **`IndexerScoreProgramFactory` is blocked on the TTNN factory concept.** It is still on the legacy imperative `host_api` with a `MeshWorkload` shape: the sheet reads `legacy (MeshWorkload)`, and `Is able to port?` = `yes (with PD step)`. It needs a `ProgramDescriptor` migration first, under a separate recipe these docs don't cover, owned by TTNN / the PD-migration team. Every other gate already clears for this factory.
2. **`RingIndexerScoreDsaMeshWorkloadFactory` is blocked on Feature compatibility: GlobalSemaphore**, which Metal 2.0 does not support yet. The co-scheduled ring all-gather is driven by caller-supplied `GlobalSemaphore`s: their addresses ride AG kernel RTAs and are re-patched on every cache hit. This waits on the feature. The factory is **also Device 2.0 RED** through its borrowed AG and CCL code. GlobalSemaphore-bound legacy semaphore and NoC-address primitives are retained in the AG kernels; these are structural, and the Device 2.0 track cannot clear them until GlobalSemaphore has a representation. In addition, two `noc_async_writes_flushed()` holdovers remain in the shared CCL fabric helper `minimal_ccl_common.hpp`, and those route to the Device 2.0 track. This factory's TTNN, offset, and 3rd-arg gates clear.

**Path forward.**
- When the classic factory's PD migration lands, it becomes the natural clean-subset candidate (re-audit; it is likely GREEN on every other gate today). Its kernels are shared with the ring factory, so a classic-only port forks them; see *Heads-ups*.
- The ring factory unblocks when GlobalSemaphore support lands in Metal 2.0. It also needs the two CCL-helper flush fixes, and a Metal 2.0 counterpart of the ring-attention all-gather host helper that it uses to assemble its program; see *Team-only → host-side coupling*.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** Classic: **RED**, `yes (with PD step)` (a legacy concept). The op needs a `ProgramDescriptor` port first, under a separate recipe these docs don't cover. Ring: **GREEN**, `yes`, and the primary cross-check is clean. No blocking column fires on either row: `Known op issues` is empty, `Runtime-args update (get_dynamic_runtime_args)` is `no`, `Smuggled pointer` is `no`, and `TensorParameter relaxation` is `none`.

  Primary-column cross-check (sheet value → code evidence):

  | Column | Classic row | Ring row |
  |---|---|---|
  | `Concept` | `legacy (MeshWorkload)` ✓: `create_mesh_workload` + `create_at` + `override_runtime_arguments` (`indexer_score_program_factory.hpp:34-54`) | `WorkloadDescriptor` ✓: the program is built by `create_workload_descriptor()` (`ring_...factory.hpp:38-42`). The variant member is a `MeshWorkloadFactoryConcept` shell (`.hpp:89`) that delegates to `DescriptorMeshWorkloadAdapter` (`.cpp:740`) only to host a custom override. That makes it descriptor-API code, consistent with the sheet's `PD Op (custom)`. See *Recipe notes* #2. |
  | `Custom hash` | `yes` ✓ (`device_operation.hpp:34`) | `yes` ✓ (same hook) |
  | `Backdoor custom hash` | `no` ✓: no `attribute_values` / `to_hash` in the op | `no` ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` ✓: no hook on the device op | `no` ✓ |
  | `Override runtime args method?` | `n/a` ✓ (the column is PD-only; on a legacy op the method is part of the concept signature) | `yes` ✓ (`.hpp:82-86`) |
  | `Pybind descriptor` | `no` ✓ | `no` ✓ |
  | `Secretly SPMD Workload?` / `Why secretly SPMD?` | `yes` / `Per-coord RTAs` ✓. Coordinates differ only in compute/writer CRTA values (`chunk_t`, straddle terms; `.cpp:388-393`). CTAs, CBs, and core ranges are coordinate-independent. | `no` / blank ✓. Per-coordinate **CTAs** differ: AG `kMyTransportRank` and the rank-dependent target counts (AG helper `.cpp:857-862`, `:1036`, `:1091`) and the fabric route args (`:1151-1152`, `:1172-1173`), so the workload is MPMD. |
  | Factory-set match | Two code factories ↔ two sheet rows ✓ (no phantom or missing row) | |
  | Invariants | `get_dynamic_runtime_args` = `no`, and op-owned tensors = `no`: both hold ✓ | |

- **Device 2.0 (every kernel used):** **RED, ring factory only. Scope: isolated, not broad.**
  - **GREEN for everything the classic factory uses.** The op's own reader, writer, and compute kernels are structurally Device 2.0: `Noc`, `CircularBuffer`, `Semaphore<>`, `CoreLocalMem`, `MulticastEndpoint`, and `noc.async_read`/`async_write` on `TensorAccessor` endpoints. Compute does every FIFO op through `CircularBuffer`, and its remaining CB-index arguments go only to compute LLKs and `kernel_lib` templates. The only CB-index free function in these kernels is `get_tile_size(cb_*)` (reader `:41-43`), which is sanctioned. The donor helpers they call are Device 2.0 too:
    - `fill_tile_zeros` uses `noc.async_write_zeros` on a `CircularBuffer` (`sdpa/.../dataflow_common.hpp:25-37`).
    - `fill_neginf_tile` uses `CircularBuffer::get_write_ptr` (`:332-357`).
    - `RingSDPAOpReceiver` uses `Semaphore<>::wait_min` (`fused_op_receiver.hpp:95-109`).
    - `read_metadata_scalar_u32` uses `noc.async_read` on a `TensorAccessor` (`metadata_scalar_read.hpp:34-42`).
    - The `kernel_lib` reduce-scaler, untilize, and reduce helpers are `DataflowBuffer`-based.

    The raw `reinterpret_cast<volatile tt_l1_ptr …*>` accesses on wrapper-obtained pointers (reader `:232-233`, `:320`, `:342-366`; writer `:167-171`) follow the "peek current cursor for local entry data" form that `cb_dfb_api_whitelist.md` sanctions, so they are not flagged (*Recipe notes* #5).
  - **RED in the borrowed AG kernels and the CCL helper they call.** The AG kernels are cross-family donors owned by `experimental/ccl/ring_attention_all_gather_async`, and the helper belongs to `ccl/common`. Both AG kernels are otherwise structurally Device 2.0; they were migrated in #45061, which "leave[s] fabric untouched". `get_local_cb_interface(cb_output_id)` (AG reader `:245-246`) is sanctioned and not flagged.

    | File | Line | Call | Wrapper in scope | Category |
    |---|---|---|---|---|
    | `experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp` | 371, 373, 397 | `noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), …)` | `Noc noc_obj` (`:248`); `Semaphore` has only id / binding-token constructors (`tt_metal/hw/inc/api/semaphore.h:52-58`), so it cannot wrap a GlobalSemaphore address | **structural**: GlobalSemaphore-bound and retained by design (the in-code comment at `:366-368` and #45061's "GlobalSemaphore-bound semaphore ops — structural") |
    | same | 530 | `noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0)` | same | **structural** |
    | `.../kernels/ring_attention_all_gather_writer.cpp` | 270-274 | `safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0)` → fabric `NocUnicastAtomicIncCommandHeader` | the fabric packet header takes a raw NoC address | **structural**: GlobalSemaphore + fabric |
    | `ttnn/cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp` | 338 | `noc_async_writes_flushed()` in `fabric_write_unidir<AddrGenType>` (`:327-339`), reached from the AG writer at `:112` (via `write_bank_owned_slices`), `:393`, and `:582` | none in the helper (it has no `Noc` parameter); the replacement `Noc::async_writes_flushed()` exists and is used by this op's reader at `:170` | **isolated holdover**: actionable on the Device 2.0 track |
    | same | 376 | `noc_async_writes_flushed()` in `scatter_fabric_write_unidir<AddrGenType>` (`:360-377`), reached from the AG writer at `:374` and `:572` | same | **isolated holdover**: actionable |

    The structural rows cannot clear on the Device 2.0 track alone; they clear together with the GlobalSemaphore Appendix A entry. The two `minimal_ccl_common.hpp` rows each need a 1-line change plus a `Noc` in scope. The helper has 33 includers across the CCL tree and was not part of #45061's helper migrations. Routing: the Device 2.0 team. These are gated conservatively; see *Questions* #1.
- **Feature compatibility:** **RED** (GlobalSemaphore, ring factory). Every entry, in order:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | The only hit is `ccl/ccl_op_fusion.hpp` (included by the ring factory), where `GlobalCircularBuffer*` is a default-null parameter of `make_cb_descriptor` (`:148-153`). Neither factory nor the AG helper calls that function. No `CBDescriptor.global_circular_buffer`, `remote_index`, or `CreateGlobalCircularBuffer` appears. A header-presence-only false positive. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset`, `set_address_offset`, 4-arg `UpdateDynamicCircularBufferAddress`, or `cb_descriptor_from_sharded_tensor` appears in the op or the AG helper. Every `CBDescriptor` literal leaves the field at its default (ring `.cpp:263-267`, `:293-299`; AG helper `.cpp:945-1018`). |
  | GlobalSemaphore | **RED** | Ring factory only; unconditional within that factory. Detail below. |

  #### GlobalSemaphore (UNSUPPORTED): `RingIndexerScoreDsaMeshWorkloadFactory`
  - **Signals.**
    - The type is referenced as `std::vector<tt::tt_metal::GlobalSemaphore> ag_semaphore` in `FusedRingConfig` (`device/indexer_score_device_operation_types.hpp:53`), along with `#include <tt-metalium/global_semaphore.hpp>` (`:18`).
    - The public entry point takes `const std::vector<tt::tt_metal::GlobalSemaphore>& ag_multi_device_global_semaphore` (`device/indexer_score_device_operation.hpp:160`, `.cpp:1251`, stored at `:1276`), and the ownership checks run on both miss and hit (`.cpp:250-259`).
    - The factory passes `fused.ag_semaphore` to `ring_attention_all_gather_async_multi_core_with_workers_helper` (`ring_...factory.cpp:664`), whose signature takes `const std::vector<GlobalSemaphore>&` (AG helper `.hpp:294`). The helper writes `semaphore.at(sem_index).address()` into the AG reader and writer RTAs (AG helper `.cpp:1293-1294`, `:1335-1336`).
    - The override re-patches those addresses on every hit (`ring_...factory.cpp:824-833`, `:873-878`).
    - The kernels consume them as raw L1 addresses (AG reader `:153`, `:371-397`, `:530`; AG writer `:157`, `:270-274`).
  - **Code-path scope.** The classic factory never sets `fused_ring`: it is `nullopt` on the classic path, and `select_program_factory` routes it at `.cpp:380-383`. The GlobalSemaphore type is reachable from the classic factory only through that unset optional in the shared attributes struct, so the classic factory is N/A. Inside the ring factory the all-gather is always co-scheduled, so there is no clean sub-path.
  - **Not the "op-owned GlobalSemaphores" stop** in `ttnn_factory.md`'s feasibility gate. The semaphores are caller-owned and arrive through `operation_attributes_t`; the Appendix A entry is what blocks.
  - **Expected resolution.** GlobalSemaphore is not yet supported in Metal 2.0; this factory becomes portable once GlobalSemaphore bindings land on `KernelSpec` (Appendix A Status). Re-audit then.
- **CB endpoints (GATE-free):** For the ring factory, every CB is either legal 1P+1C or a single-toucher **self-loop**. There are no multi-binding CBs and no dead CBs. Four CBs are **conditional DFBs**; the legacy factory already allocates them conditionally on the host. The per-`(CB, config)` table is under *Port-work summary*. For the classic factory the census is **skipped**: its blocker is on the op-code side, so re-audit after the PD migration. The Device 2.0 precondition is met: the census idioms in the kernels, including the AG ones, are intact.
- **Offset base pointers:** **GREEN.** Neither factory has an address RTA with a host-folded offset.
  - Classic: the reader CRTAs `q/k/w.buffer()->address()` (`indexer_score_program_factory.cpp:383-384`, refreshed at `:444-451`) and the writer CRTA `out.buffer()->address()` (`:393`, `:451`) are clean bases.
  - Ring: every tensor address is a bare `Buffer*` binding. That covers the reader `q`, `k`, `w`, and `k_local` (`ring_...factory.cpp:553-556`) and the optional metadata tensors (`:561`, `:571`, `:580`); the writer `out` (`:589`); and the AG input and output plus the `slot_id` / `kv_actual_isl` slots (AG helper `.cpp:1296-1311`, `:1338-1342`).
  - Every offset in the op is a **separate page-index scalar** that the kernel adds to `page_id`:
    - `k_batch_page_offset`, which is `cache_batch_idx·Tt·Dt` (`indexer_score_host_common.hpp:197-204`, used in the reader at `:437`, `:471`, `:616`);
    - `k_local_batch_page_offset` (ring `.cpp:523-524`, used in the reader at `:464`);
    - the AG `input_batch_base` (AG helper `.cpp:1248-1257`, used in the AG reader at `:260` and `:304`).
  - The GlobalSemaphore address RTA is not a tensor base; it belongs to Appendix A.
  - The triage doc (`2026-07-19_offset_base_pointers.md`, a dated prior) lists neither this op nor the ring-attention AG. Outcome: no fold, and the op is not in the tables → clean.
- **TensorAccessor 3rd argument:** **GREEN.** Every site is **Class 2**, so the port drops the argument and no `dynamic_tensor_shape` is needed. All the sites live in the op's own kernels and are shared by both factories. The donor and AG accessors are all 2-arg (`make_tensor_accessor_tuple`, `tt_metal/hw/inc/api/tensor/tensor_accessor.h:781-787`; `read_metadata_scalar_u32`, `metadata_scalar_read.hpp:37`).

  | Site | Tensor | Sharded? | Value passed | True page | Class |
  |---|---|---|---|---|---|
  | reader `:699` | q | interleaved (`validate_static`, `device_operation.cpp:63-69`) | `q_tile_bytes = get_tile_size(cb_q)` (`:41`): the tile of q's own format (2048 B bf16, 1088 B bfp8_b) | same | 2 (a) |
  | reader `:700` | k (classic: the cache; ring: the gathered buffer) | interleaved **or ND-sharded DRAM** | `k_tile_bytes = get_tile_size(cb_k)` | equals the aligned page; 2048 and 1088 are both multiples of 64, so the verbatim sharded stride is correct | 2 (a) |
  | reader `:701` | weights | interleaved bf16 TILE (`:625-633`) | `bf16_tile_bytes` = 2048 | 2048 | 2 (a). On MSA (`synthesize_gate`, classic only) the accessor is built over the q placeholder but never used, because `read_w_group` returns through `fill_w_group_const` (`:376-379`) → 2 (c) |
  | reader `:515` (and the unevaluated `decltype` at `:489`, which names that accessor's type) | k_local (ring only) | interleaved or ND-sharded DRAM; TILE; dtype must equal k's (validated at `.cpp:533-539`) | `k_tile_bytes` | equals the aligned page | 2 (a) |
  | writer `:269` | out | interleaved (q's memory config; `compute_output_specs`, `:747-750`); ROW_MAJOR bf16 | `page_bytes`, one output row: classic `.cpp:318-321` (`T` or `nblocks` × 2 B), ring `.cpp:443-444` (`T` × 2 B) | `buffer->page_size()` (the logical row) | 2 (b): realigned by the interleaved accessor. Unpooled widths (`T % 32 == 0`) already equal the aligned page. |

  This agrees with the triage doc (`2026-07-06_tensor_accessor_3rd_arg_triage.md`, a dated prior), which lists `indexer_score` as Class 2 with the writer as the (b)-type site. My read adds that the reader sites are (a)-type and the MSA weights site is (c)-type.

## Port-work summary (ring factory; recorded for the re-audit, no brief issued)

- **Tensor bindings (per binding).** All 8 are **Case 1**. Each address is delivered today as a `Buffer*` binding, which the descriptor framework patches on cache hits and the custom override re-applies (`ring_...factory.cpp:787-822`), so it is correct on cache hit. There are no Case 2 or clean (borrowed-DFB) bindings, and nothing uses `set_globally_allocated_address`.

  | Binding | Kernel(s) | Legacy delivery | Kernel use |
  |---|---|---|---|
  | `q` | indexer reader | CRTA `Q` (`.cpp:553`) | `TensorAccessor(q_args, q_addr, …)` (reader `:699`) |
  | `k` (gathered buffer) | indexer reader; AG reader and AG writer (as the AG output) | CRTA `K` (`:554`); AG RTA `output_tensor[0].buffer()` (AG helper `.cpp:1299-1301`, `:1338-1340`) | reader `:700`; AG `make_tensor_accessor_tuple` (AG reader `:194`, AG writer `:193`) |
  | `weights` | indexer reader | CRTA `W` (`:555`) | reader `:701` |
  | `k_local` | indexer reader; AG reader (as the AG input) | CRTA `KLocal` (`:556`); AG RTA `input_tensor[0].buffer()` (AG helper `.cpp:1296-1298`) | `FusedRingGate` accessor (reader `:515`); AG reader `:191` |
  | `out` | indexer writer | CRTA `Output` (`:589`) | writer `:269` |
  | `chunk_start_idx_tensor` (optional) | indexer reader; AG reader and AG writer (as `kv_actual_isl`, except when `key_stripe_split > 1`, `.cpp:683-684`) | CRTA `ChunkMetadata`: the buffer, or `0` when absent (`:560-564`); AG RTAs (AG helper `.cpp:1311`, `:1342`) | `read_metadata_scalar_u32` → `TensorAccessor(args, addr)` (reader `:831-832`; AG reader `:222-223`; AG writer `:212-213`) |
  | `cache_batch_idx_tensor` (optional) | indexer reader; AG reader (as `slot_id`) | CRTA `SlotMetadata`, or `0` (`:570-574`); AG RTA (AG helper `.cpp:1306-1310`) | reader `:932-933`; AG reader `:213-214` |
  | `valid_end_tensor` (optional) | indexer reader | CRTA `ValidEnd`, or `0` (`:579-583`) | reader `:877-878` |
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** drop the page-size argument at reader `:515`, `:699`, `:700`, `:701` and writer `:269`. The `decltype` at reader `:489` follows `:515`. All sites are Class 2, so `dynamic_tensor_shape` is not needed.
- **CB endpoints (per `(CB, config)`, per node).** The compute rectangle is `[0, cols_used) × [0, rows_used)`. The AG worker cores sit in the reserved columns from `compute_cols_x` onward (`.cpp:167-168`), so the two node sets are disjoint. Each node set has its own CB instances, even where the numeric indices coincide.

  | CB (index) | Nodes | Config | Touchers | Disposition |
  |---|---|---|---|---|
  | `cb_q` (c_0) | compute rect | always | reader: producer (`read_q_rows` reserve/push per row, `:290-313`); compute: consumer (cumulative `wait_front`, `:654`; `pop_front`, `:729`) | 1P+1C (legal) |
  | `cb_k` (c_1) | compute rect | always | reader: producer (`read_block_or_mcast`, `:185-205`); compute: consumer (`:584`, `:723`) | 1P+1C |
  | `cb_w` (c_2) | compute rect | always | reader: producer (`read_w_group` reserves, then a raw `expand_w_group` and `push_back`, `:330-406`); compute: consumer (`:272`, `:727`) | 1P+1C |
  | `cb_mask` (c_3) | compute rect | always | reader: producer (`build_mask_tiles`, `:249-257`, uses zero fill plus raw fill); compute: consumer (`wait_front`, never popped, `:525`) | 1P+1C |
  | `cb_qk` (c_4) | compute rect | always | compute only, as both producer and consumer (`matmul_phase` `:246-259`; `mul_phase` `:265-289`) | **self-loop (compute)** |
  | `cb_out_strip` (c_5) | compute rect | always | compute: producer (`compute_kernel_lib::untilize`, `:719`); writer: consumer (`:44-61`, `:75-124`) | 1P+1C |
  | `cb_acc_strip` (c_6) | compute rect | always | compute only (reserve/push at `:649`/`:693`; the untilize helper waits and pops) | **self-loop (compute)** |
  | `cb_meta_derived` (c_7) | compute rect | `chunk_start_idx_tensor` present | reader: producer (`:826-898`, which also serves as the NoC landing slot); compute: consumer (`:511-517`) | 1P+1C; **conditional DFB** (`.cpp:290-304`) |
  | `cb_meta_writer` (c_8) | compute rect | `chunk_start_idx_tensor` present | reader: producer (`:901-909`); writer: consumer (`:254-262`) | 1P+1C; **conditional DFB** |
  | `cb_meta_slot` (c_9) | compute rect | `cache_batch_idx_tensor` present | reader only (`reserve_back` plus a landing-slot peek; never pushed; `:930-933`) | **self-loop (DM)**; **conditional DFB** (`.cpp:305-311`) |
  | AG forward data (`c_in0`) | AG forward workers | always | AG reader fwd: producer (prefetch reserve/push; it also uses the write pointer as metadata scratch, AG reader `:207-223`); AG writer fwd: consumer (wait/pop, AG writer `:111-119`, `:359-404`) | 1P+1C |
  | AG backward data (`c_in2`) | AG backward workers | always | AG reader bwd: producer; AG writer bwd: consumer | 1P+1C |
  | AG packet header (`c_in1`), one instance per direction | AG forward / backward workers | always | AG writer only (reserve/push twice, then raw header writes; AG writer `:243-252`) | **self-loop (DM)** |
  | AG metadata (`c_in3`), one instance per direction | AG forward / backward workers | when the AG receives `kv_actual_isl` | AG writer only (landing-slot peek, no FIFO ops; AG writer `:207-213`) | **self-loop (DM)**, role-free; **conditional DFB** (AG helper `.cpp:1001-1019`) |

  The DM self-loops are legal on Gen1. They are recorded only for the later Quasar-uplift audit, because Gen2 rejects a DM self-loop.

## Heads-ups (ring factory; recorded for the re-audit, no brief issued)

- **CB endpoints (multi-binding shapes to watch).** None. The row multicast (q and w) and column multicast (k) let a remote sender `MulticastEndpoint`-write into receivers' slots at the sender's own `get_write_ptr()` (reader `:154-171`). That relies on all DFB instances in the rectangle having identical layout. It adds no local toucher.
- **Cross-op and shared kernels.**
  - *Intra-op:* both factories bind `device/kernels/{reader,writer,compute}_indexer_score.cpp` (classic `.cpp:356-377`, ring `.cpp:471-505`). No other op binds them, and no `_metal2` fork exists. A port of either factory on its own takes rung 2: fork beside the originals in `device/kernels/`. **Sunset list, not authorization to convert in place:** the other factory.
  - *Borrowed:* `ring_attention_all_gather_{reader,writer}.cpp` have no `_metal2` fork; the first port creates one beside the original, in the peer directory (rung 2). **Sunset list, not authorization:** `experimental/ccl/ring_attention_all_gather_async` itself and `transformer/sdpa` `ring_joint_sdpa`.
  - *Host-side coupling:* the program is assembled by another op's `ProgramDescriptor` helper; see *Team-only*.
- **RTA varargs.** No CTA varargs: every CTA read is at a constexpr offset. The genuine **CRTA** vararg blocks are:
  - the shard arrival order, `ring_size` entries, read through the iterator's data-selected `get_shard(...)`: reader `:751-753` at `reader::Count + shard`, writer `:293-295`, compute `:557-559` (host `.cpp:533-536`, `:584`, `:591`, `:595`);
  - the reader's physical multicast axes, `cols_used` X entries then `rows_used` Y entries, read in CTA-bounded loops plus data-selected `x(col)`, `x(diag)`, `y(row)` (reader `:659-692`; host `append_multicast_axes`, `indexer_score_host_common.hpp:62-77`).

  **Name, do not vararg:**
  - the 14 `indexer_common::reader::*` slots and the writer/compute `KvLength` / `ChunkStart` / `Straddle*` slots;
  - the per-core `core_id` (RTA 0);
  - the **9-field fused-ring block**. It sits at `reader::Count + ring_size`, after the shard-order block, and the donor `RingSDPAOpReceiver` reads it once per field through a positional `read_arg(rt_args_idx++)` callback (reader `:916-920`; `fused_op_receiver.hpp:69-93`). The fields are distinct, so they are named; their legacy offset is irrelevant. Two of the nine are semaphore ids; see coupling.

  The AG kernels' own vararg blocks belong to their fork: the per-input descriptor loop over the `num_inputs` CTA, `OpSignaler`'s NoC-coordinate lists read through `get_arg_addr`, and the fabric connection args. Note that typed kernel array arguments landed on 2026-11-03 (#63805; `kernel_spec.hpp:199-236`); see *Recipe notes* #7.
- **Anything else.**
  - **Optional tensors whose presence the kernel infers from a zero address.** The reader tests `slot_addr != 0` (`:927-928`) and `valid_end_addr != 0` (`:873-874`), and absent tensors get placeholder accessor CTAs built over `q.buffer()` (`.cpp:416-439`). Under Metal 2.0 the address rides the binding, so presence has to become a host define with an `#ifdef`-gated `tensor::` alias, following `port_patterns.md` "Pattern: Conditional / optional resource bindings". That is faithful, because presence is hashed (`device_operation.cpp:436-446`). The same applies to the optional DFBs: `cb_meta_*` here, and `cb_scaler` / `cb_pool_scratch` on the classic path.
  - **Zero-filled CB-index slots.** On this path `cb_scaler` and `cb_pool_scratch` get no CB, and their CTA slots carry index 0, which aliases `cb_q` (`.cpp:258`). They are referenced only under `if constexpr (block_pool)`. A Metal 2.0 `dfb::` name cannot exist for them, so the shared kernels need the `#ifdef` form, not `if constexpr` (`migration_guide.md` "Optional resources").
  - **Override translation** (`.cpp:775-969`). The ported-from override always rebinds every resolved buffer. It refreshes scalars (the consumer CRTAs plus the AG per-core fields) **only when a cached `ScalarControls` changed** (`:824-837`, `:968`). That guard lives in mutable shared variables that a Metal 2.0 override does not have, so how to reproduce it is a porter call. It must also return a `TensorArgument` for every io-tensor `TensorParameter` (the `WorkloadDescriptor`-source exception in `ttnn_factory.md`).
  - **Semaphore-id RTAs and cross-node-set signaling.** `fused_sem0` and `fused_sem1` (`.cpp:341-343`) are allocated only on the compute rectangle. The AG reader's `OpSignaler`, running on the AG worker cores, increments them by id; the id arrives as an RTA (`worker_sync_utils.hpp:47`, `:54`; the increments are at `:78` and `:85`, inside `synchronize_workers_and_signal_op`). The indexer reader gets the same ids through CRTAs into `RingSDPAOpReceiver`. Expressing this needs semaphore bindings that span the AG kernels' node set, plus donor-side changes; see coupling.

## Team-only

### Out-of-directory coupling and donor shape (ring factory)

**Op-level roll-up: ⭐ (scheduling friction), plus ⚠ and ✗.**
- ⭐ The borrowed AG reader depends on `CircularBuffer&` prefetch helpers.
- ⚠ Semaphore ids reach two cross-family donor structs as `uint32_t` RTAs.
- ✗ The metadata reader takes `TensorAccessorArgs` plus a raw base (Shape 2).
- The ring factory's **host** code assembles its program through another op's `ProgramDescriptor`-building helper.

There is no Shape-4 (pre-Device-2.0 addr-gen) donor.

| Op kernel | Donor file | Class | Status |
|---|---|---|---|
| indexer reader | `transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp` | cross-family (transformer/sdpa) | ✓ |
| indexer reader | `transformer/sdpa/device/kernels/dataflow/block_cyclic_remap.hpp` | cross-family | ✓ (pure index math) |
| indexer reader | `transformer/sdpa/device/kernels/dataflow/fused_op_receiver.hpp` (also `ring_utils.hpp` and `kernels/ring_id_sequencer.hpp`) | cross-family | ⚠ |
| indexer reader | `transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp` (also `kernels/chunked_q_mapping.hpp`) | cross-family | ✗ (Shape 2) |
| indexer reader | `experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_rank_mapping.hpp` (also `ccl/shared_with_host/snake_ring.hpp`) | cross-family (experimental/ccl) | ✓ (pure) |
| indexer reader | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | shared kernel-lib | ✓ (classic block-pool only) |
| indexer compute | `kernel_lib/untilize_helpers.hpp`, `kernel_lib/reduce_helpers_compute.hpp` | shared kernel-lib | ✓ (`reduce` is classic block-pool only) |
| indexer compute | `tt_metal/hw/inc/api/compute/experimental/indexer_mul_custom.h`, `api/compute/reduce_custom.h` | tt_metal | no concern |
| indexer writer | (no out-of-directory includes beyond `tt_metal` APIs) | — | ✓ |
| AG reader (borrowed) | `ccl/kernel_common/worker_sync_utils.hpp` (`OpSignaler`) | cross-family (ccl) | ⚠ |
| AG reader (borrowed) | `.../ring_attention_all_gather_async/device/kernels/ring_attention_prefetch_utils.hpp` | in-family to the AG | ⭐ |
| AG reader and AG writer (borrowed) | `transformer/sdpa/.../metadata_scalar_read.hpp`; `ring_attention_all_gather_metadata.hpp` | cross-family / in-family | ✗ (Shape 2) / ✓ |
| AG writer (borrowed) | `ccl/common/kernels/minimal_ccl_common.hpp` (`fabric_write_unidir` and `scatter_fabric_write_unidir` over an accessor `AddrGenType`) | cross-family (ccl) | ✓ by signature shape (accessor); its Device 2.0 residual is listed under Gate detail |
| AG writer (borrowed) | `tt_metal/fabric/hw/inc/*`, `ccl/kernel_common/worker_routing_utils.hpp` | tt_metal / ccl utility | no concern |

**Per-call detail.**
- `fill_tile_zeros<N>(const Noc&, uint32_t cb_id, uint32_t tile_id)` and `fill_neginf_tile<N>(uint32_t cb_id, uint32_t tile_id)` (`dataflow_common.hpp:25-37`, `:332-357`) take `uint32_t cb_id` → ✓. `dfb::name` casts implicitly.
- `RingSDPAOpReceiver(bool, uint32_t& rt_args_idx, ReadArg read_arg)` (`fused_op_receiver.hpp:68-93`) reads `bwd_semaphore_id` and `fwd_semaphore_id` (`:78-79`) as plain `uint32_t` from the caller's CRTA callback (indexer reader `:917-920`), then builds `Semaphore<>(id)` (`:99`, `:106`, `:116`). That is the `uint32_t sem_id` shape → **⚠ not workable today**: `sem::name` has no `uint32_t` conversion, and the semaphore-ID RTA is the pattern the binding model removes. It needs a donor-side form that takes binding tokens or `Semaphore&`; for a porter, that is an assumption-violation stop.
- `OpSignaler(uint32_t& rt_args_idx)` (`worker_sync_utils.hpp:42-64`) builds `Semaphore<>(get_arg_val(...))` from RTAs for the worker-sync semaphore and the consumer-signal semaphore (`:47`, `:54`). The consumer-signal semaphore lives on a different node set (the indexer compute rectangle). → **⚠**, same shape.
- `read_metadata_scalar_u32(NocT&, const AccessorArgsT&, uint32_t tensor_addr, uint32_t dst_l1_addr)` (`metadata_scalar_read.hpp:34-42`) is **Shape 2**: `TensorAccessorArgs` plus a raw base → **✗, workable**. Pass `tensor::name.args`, and the base through the `get_bank_base_address` bridge (`TensorBindingToken` exposes `args`; `tt_metal/hw/inc/api/tensor/tensor_binding_token.h:43-47`). Call sites: indexer reader `:831-832`, `:877-878`, `:932-933`; AG reader `:213-214`, `:222-223`; AG writer `:212-213`.
- `prefetch_batch_read_tiles`, `prefetch_batch_read_physically_contiguous_packets`, and `prefetch_batch_read_packets` (`ring_attention_prefetch_utils.hpp:24-64`, `:74-108`, `:119-156`) take `CircularBuffer&` → **⭐ ⚠ flag**. A `_metal2` fork of the AG reader would hold a `DataflowBuffer`. The helper is shared with `ring_attention_neighbor_halo_reader.cpp` and gemma4's `sliding_halo_reader.cpp`. This needs cross-team discussion with the AG owners.

**Borrowed kernel files (file-path instantiation; ring factory only).**

| Kernel file | Owner | Instances | Other binders (sunset list) | `_metal2` fork |
|---|---|---|---|---|
| `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp` | experimental/ccl (`ring_attention_all_gather_async`) | 2 (fwd and bwd), through the helper (AG helper `.cpp:1027-1029`, `:1133-1161`, `:1389-1392`) | `ring_attention_all_gather_async` (its own factory, `.cpp:65`); `transformer/sdpa` `ring_joint_sdpa_program_factory.cpp:3749` | none; the first port creates it beside the original |
| `.../kernels/ring_attention_all_gather_writer.cpp` | same | 2 | same | none |

The intra-op shared kernels (`device/kernels/{reader,writer,compute}_indexer_score.cpp`) are bound by both factories only, and have no fork.

**Host-side helper coupling.** This is not one of the recipe's two escape types; see *Recipe notes* #6.
- The ring factory calls `ttnn::ring_attention_all_gather_async_multi_core_with_workers_helper(ProgramDescriptor& desc, …)` (AG helper `.hpp:280-330`; called at ring `.cpp:652-692`). The helper appends 4 kernels, up to 6 CBs, the worker-sync semaphores, and every AG RTA into the indexer's `ProgramDescriptor`.
- The ring override then patches AG RTAs using the helper's positional-layout constants: `kReaderReadySemaphoreFieldOffset`, `kWriterReadySemaphoreFieldOffset`, `kInputBatchBaseFieldOffset`, `kValidPagesFieldOffset`, `reader_metadata_base()`, and `kReaderMetadata*Offset` (`.cpp:869-897`, `:916-940`).
- The consumer-side RT blocks come from `ttnn::prim::RingSDPAFusedOpSignaler` (`transformer/sdpa/device/ring_fusion.hpp`; `.cpp:316-345`, `:527-529`) and `AllGatherFusedOpSignaler` (`ccl/ccl_op_fusion.hpp`; `.cpp:604-609`).
- A Metal 2.0 port of this factory therefore needs a `ProgramSpec`-emitting counterpart of the AG helper. That helper is owned by experimental/ccl and shared with `ring_attention_all_gather_async` and `ring_joint_sdpa`, and a porter cannot write it from this op's directory. See *Questions* #2.

### Relaxation candidates

None mined. The custom hash folds in `tensor_args` whole (`device_operation.cpp:465`), and `Tensor` hashes `("storage", "tensor_spec")` (`ttnn/api/ttnn/tensor/tensor.hpp:252-253`), so every `TensorSpec` is pinned. The hash-excluded items are all attribute scalars: `chunk_start_idx`, the values of `cache_batch_idx` and `kv_len`, `index_cache_num_layers` / `index_cache_layer_idx`, and GlobalSemaphore identity. The hash is therefore consistent with `none`.

### TTNN factory analysis (`ttnn_factory.md` deliverable)

#### TTNN ProgramFactory

##### Concept
- **Ring:** `MeshWorkloadSpecFactoryConcept`, the sheet's `Porting Target`.
  - Ported-from per-coordinate entry point: `RingIndexerScoreDsaProgramFactory::create_workload_descriptor` (`ring_indexer_score_dsa_program_factory.cpp:718-731`), which calls `build_ring_program_descriptor(…, coord)` (`:87-713`) once per coordinate.
  - Override: `RingIndexerScoreDsaMeshWorkloadFactory::override_runtime_arguments` (`.cpp:775-969`), with its `create_mesh_workload` shell at `.cpp:733-773`.
- **Classic:** `MeshWorkloadSpecFactoryConcept` per the sheet. This is provisional, because the factory must first move to the `ProgramDescriptor` API. Its current per-coordinate entry point is `create_mesh_workload` / `create_at` (`indexer_score_program_factory.cpp:414-431`, `:34-412`), and its override is at `:433-465`.

##### Fit
- **Execution model (ring):** `MPMD`. What varies per coordinate:
  - the AG CTAs `kMyTransportRank`, `num_targets_{forward,backward}`, and the fabric route args (AG helper `.cpp:857-862`, `:1036`, `:1091`, `:1151-1152`, `:1172-1173`), plus the fabric connection RTAs (`:1352-1360`);
  - the indexer reader's fused-ring block (ring index and expected writes; `.cpp:316-317`, `:527-528`), `shard_order` (the arrival waves for this rank; `:195-201`, `:533-536`), and the geometry ranks (`:567-569`);
  - the compute and writer CRTAs `chunk_t` and the straddle terms (`:122-123`, `:590`, `:594`).
- **Execution model (classic):** `SPMD (+ per-device args)`. Only the compute and writer causal CRTAs vary per coordinate (`.cpp:388-393`).
- **Op-owned device resources:** none. No op-owned tensors. The GlobalSemaphores are caller-owned, so they are gated by Appendix A, not by the op-owned-GlobalSemaphore stop.
- **Tensor-arg matching:** strict (`none`).
- **Ported-from shape:** 1:1. There is one `WorkloadDescriptor.programs` entry per coordinate (`.cpp:724-729`), which maps to one `MeshWorkloadArtifacts::programs` entry per coordinate, over the same ranges.

##### Custom compute_program_hash
Present at `device/indexer_score_device_operation.hpp:34` / `.cpp:386-466`. There is no backdoor. It is recorded so the porter knows it is there and leaves it alone.

##### Stop signals
Ring: GlobalSemaphore (Appendix A). Classic: the legacy concept. The overall audit result is **RED**.

## Misc anomalies (team-only, non-gating)

- `ring_indexer_score_dsa_program_factory.cpp:171` hard-codes `ring_size <= 32` ("consumer gate capacity 32"). Validation (`device_operation.cpp:548-552`) and the reader's arrays (`reader_indexer_score.cpp:488`) use `kMaxRingSize` (`kernels/indexer_score_work_split.hpp:17`, which is also 32). The duplicated literal can drift.
- **Borrowed AG (route to the experimental/ccl owners).** The helper appends `num_inputs` page-size CTAs ahead of the accessor blocks (AG helper `.cpp:1066`, `:1120`; commented "TensorAccessorArgs tuples expect one page-size entry per tensor"). Neither AG kernel reads them: both start their accessor tuples at `kNumFixedCompileTimeArgs + num_inputs` (AG reader `:132`, AG writer `:141`). They are dead CTAs that only enter the compile hash, and the comment looks stale.
- **Unallocated CB-index CTA slots alias c_0.** `std::array<uint32_t, num_cb_args> cb_id{}` leaves `cb_scaler` and `cb_pool_scratch` at 0 whenever block-pooling is off: always in the ring factory (`.cpp:258`), and in the classic factory when `block_size == 0` (`.cpp:176`, `:235-238`). They reach the kernels as index 0, which is `cb_q`. That is harmless today, because every use sits behind `if constexpr (block_pool)`, but an unguarded future reference would silently alias `cb_q`.

## Skipped subjects (classic factory `IndexerScoreProgramFactory`)

All seven are skipped because this factory is RED on the op-code side (a `ProgramDescriptor` migration is required): re-audit on unblock. The kernel files are shared with the ring factory, so the ring-scoped census above covers only the `fused_ring_enabled` paths. The classic-only configs are `fuse_single`, `fused_stream_k`, `stream_heads`, `block_pool`, and `synthesize_gate`.
- **TTNN porting shape:** skipped. The sheet's `MeshWorkloadSpecFactoryConcept` is provisional until the PD migration.
- **TensorParameter relaxations:** skipped. The sheet reads `none` (also checked under the TTNN gate).
- **TensorParameter analysis:** skipped. The re-audit should classify the post-migration delivery of `q`, `k`, `weights`, and `out`.
- **CB endpoints:** skipped. Classic-only CBs and configs (`cb_scaler`, `cb_pool_scratch`, `fuse_single`'s 1-tile `cb_qk`) are not censused.
- **Out-of-directory coupling:** skipped. Its donors are a subset of the ring set above.
- **RTA varargs:** skipped.
- **Incidental anomalies:** skipped, except for the shared-code items already listed.

## Questions for the user

1. **The Device 2.0 boundary for borrowed CCL code.** #45061, which you authored, classifies GlobalSemaphore-bound semaphore ops as structural retained primitives and leaves fabric untouched. I gated two things conservatively. The first is the GlobalSemaphore-bound primitives in the AG kernels (AG reader `:371`, `:373`, `:397`, `:530`; AG writer `:270-274`). The second is the two `noc_async_writes_flushed()` calls in `minimal_ccl_common.hpp` (`:338`, `:376`), reached from the AG writer. Should either count as a Device 2.0 violation for gating, or do you consider them outside the Device 2.0 boundary? Neither changes the verdict, because the ring factory is RED on GlobalSemaphore regardless. The answer only decides whether a Device 2.0 work item exists, which would be a 2-site fix in a helper with 33 includers.
2. **Ownership of a Metal 2.0 AG host helper.** The ring factory builds its program through `ring_attention_all_gather_async_multi_core_with_workers_helper(ProgramDescriptor&, …)`. Its port will need a `ProgramSpec`-emitting counterpart, which a porter cannot write from this op's directory. Is that counterpart planned on the CCL side, or should the ring factory's eventual port be scheduled together with `ring_joint_sdpa`, the helper's other fused consumer? This decides who unblocks the ring factory once GlobalSemaphore support lands.

## Recipe notes

1. **Stale status-summary row.** The template's "*Feature Support* — Variadic-CTA" row has no counterpart in Appendix A, which lists only GlobalCircularBuffer, `address_offset`, and GlobalSemaphore. The RTA-varargs subject says CTA varargs port onto `KernelAdvancedOptions::compile_time_varargs`. I filled the row with "Ok (not an Appendix A entry)".
2. **The Concept legibility list misses a hybrid shape.** Under *TTNN factory concept prerequisite → Lightweight cross-check*, the list maps `create_mesh_workload()` to `legacy (MeshWorkload)`. Read literally, that contradicts the sheet's `WorkloadDescriptor` for this op's ring factory. That factory is a `MeshWorkloadFactoryConcept` shell around `DescriptorMeshWorkloadAdapter<F>`, where `F::create_workload_descriptor()` builds the program and the shell exists only to host a custom override. I took the semantic reading (descriptor API → `WorkloadDescriptor`) and did not raise a primary-column conflict. About ten factories in-tree use this shape (`exp_ring_joint_sdpa`, `all_gather_matmul_async`, `matmul_reduce_scatter_async`, `sliding_ring_sdpa`, `ring_attention_all_gather_async`, …), so the list should name it.
3. **The Red-outcome exception is phrased per op.** This op has two factories whose REDs clear on different sides: the classic factory on the op-code side (a PD migration), the ring factory elsewhere (a feature landing). I applied the rule per factory, running the seven informational subjects for the ring factory and skipping them for the classic one. The rule could say explicitly that it applies per factory.
4. **Device 2.0 versus GlobalSemaphore and fabric.** Device 2.0's own migration retains GlobalSemaphore-bound primitives because `Semaphore` has no address-based constructor (`tt_metal/hw/inc/api/semaphore.h:52-58`), and it leaves fabric helpers alone. The sanctioned list ("the list is the whole test") does not cover these. A strict reading therefore REDs Device 2.0 for every GlobalSemaphore-using donor, even though the Device 2.0 track cannot clear it. Consider stating that these residuals are subsumed by the Appendix A GlobalSemaphore entry. Also say whether fabric-adjacent `noc_async_*` free functions in CCL helpers count.
5. **Raw volatile L1 casts.** The Device 2.0 guide lists `CoreLocalMem` as the new form of raw `volatile` L1 access ("Memory Access"). The gate's Red bullets don't mention raw L1 casts, and `cb_dfb_api_whitelist.md` explicitly sanctions `reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr())`. I did not flag such casts. It would help if the gate said so.
6. **Out-of-directory coupling has no slot for host-side helper coupling.** The subject covers kernel `#include`s and file-path kernel instantiation. It doesn't cover a factory that assembles its program through another op's `ProgramDescriptor`-building host helper and patches that helper's positional RTA layout in its override. That shape is a hard dependency for the eventual port.
7. **Two stale code citations.** (a) Appendix A's GlobalSemaphore Status cites `kernel_spec.hpp` listing "GlobalSemaphore bindings" under "coming soon". In this checkout that text is gone (the header was last touched by #63805, 2026-11-03), and no `GlobalSemaphore` appears in `metal2_host_api/`. Per the recipe, I reported the entry as written. (b) #63805 adds typed compile-time, runtime, and common-runtime **array arguments** (`kernel_spec.hpp:199-236`). The RTA-varargs subject and `port_patterns.md` "Caution: Avoid varargs" still describe varargs as the interim mechanism "pending typed array arguments". This op's fixed-per-program CRTA blocks (shard order, multicast axes) may now be better served by arrays.

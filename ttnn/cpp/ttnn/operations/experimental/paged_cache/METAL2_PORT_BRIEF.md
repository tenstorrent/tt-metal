# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/paged_cache`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources` *(carry this line into the port report's Provenance section; pinned from the sibling doc-branch checkout `/localdev/edwinlee/Port_Recipe`, whose recipe copy is byte-identical to the one this audit ran against — the working checkout carries no `metal_2.0` doc tree)*

**Device 2.0 note:** cleared by `47a266001ad` *"[Cleanup] Port Paged Cache to Device 2.0 (#54598)"*. All 11 kernels are structurally Device 2.0 with **zero** CB-index-keyed free functions of any name — so every `dfb::` / `sem::` / `tensor::` token has a wrapper object to bind to, and you need touch no Device 2.0 idiom.

---

## Scope — what you are actually porting

**Three `DeviceOperation`s, eight factories, but only four program bodies.** Each `*MeshWorkloadFactory` delegates its `create_descriptor` *and* its `override_runtime_arguments` to its single-device sibling, so the pair shares one body:

| Program body | Factories it serves | `create_descriptor` | `override_runtime_arguments` |
|---|---|---|---|
| `update_cache` | `PagedUpdateCacheProgramFactory`, `PagedUpdateCacheMeshWorkloadFactory` | `paged_update_cache_program_factory.cpp:89` (mesh wrapper `:443`) | `:457` (mesh delegate `:522`) |
| `fill_cache` | `PagedFillCacheProgramFactory`, `PagedFillCacheMeshWorkloadFactory` | `paged_fill_cache_program_factory.cpp:74` (`build_paged_fill_cache_descriptor`, entry points `:340` / `:348`) | `:361` (mesh delegate `:420`) |
| tiled fused | `PagedTiledFusedUpdateCacheProgramFactory`, `…MeshWorkloadFactory` | `paged_tiled_fused_update_cache_program_factory.cpp:79` (mesh wrapper `:539`) | `paged_fused_update_cache_device_operation.cpp:395` (mesh delegate `:428`) |
| row-major fused | `PagedRowMajorFusedUpdateCacheProgramFactory`, `…MeshWorkloadFactory` | `paged_row_major_fused_update_cache_program_factory.cpp:79` (mesh wrapper `:538`) | `paged_fused_update_cache_device_operation.cpp:409` (mesh delegate `:438`) |

Both fused overrides funnel into one shared `patch_runtime_args` template (`paged_fused_update_cache_device_operation.cpp:54-125`).

**Do not port one member of a factory pair without the other** — they share the body, so they convert together. (This is technically the shared-kernel caution's *intra-op* shape, but it needs no fork: converting the body converts both.)

> **Naming caution:** the four `*MeshWorkloadFactory` types do **not** return a `WorkloadDescriptor`. Each declares `create_descriptor(...) -> ProgramDescriptor` with an extra `mesh_dispatch_coordinate` parameter. The concept is `descriptor`. There is no `create_workload_descriptor` in this directory.

---

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `CustomProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (all 8 factories; `create_descriptor` returning a `ProgramDescriptor`).
- **Op-owned tensors:** none.
- **Target concept:** **`CustomProgramSpecFactoryConcept`** — driven by `Override runtime args method? == yes` on all 8 rows, and agreeing with the sheet's own `Porting Target` cell. Each of the four program bodies' `override_runtime_arguments` must be translated into a `ProgramRunArgs` producer (see the port recipe's *Translating `override_runtime_arguments`*).
- **Custom hash:** present on all three DeviceOperations — `paged_update_cache_device_operation.cpp:313`, `paged_fill_cache_device_operation.cpp:207`, `paged_fused_update_cache_device_operation.cpp:371`. **Leave each exactly as it is.** Not a gate, and not yours to touch.
- **Pybound `create_descriptor`:** none. `paged_cache_nanobind.cpp` binds only the three public entry points via `ttnn::bind_function` (`:48`, `:89`, `:134`) — nothing to delete, so this port makes no user-visible API change.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none` `TensorParameter relaxation` (the sheet reads `none` on all 8 rows) · `get_dynamic_runtime_args` (the deprecated hook). A custom hash, an `override_runtime_arguments`, and a pybound `create_descriptor` are **not** in this list: none of them gate, and this op has the first two.

---

## Construct — to do

### Tensor bindings (15 bindings across 3 DeviceOperations)

**`PagedUpdateCacheDeviceOperation`**

- `cache_tensor` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::cache)`. Legacy: `Buffer*` in reader RTA[0] and writer RTA[0] (`factory:406,426`), re-applied raw at `:504,509`. Kernel sites: `reader_update_cache…:69`, `writer_update_cache…:55`.
- `input_tensor` — **clean** (borrowed-memory DFB). CB `c_1` with `.buffer = in1_buffer` (`factory:200-209`), re-pointed on cache hit at `:518`. → `DataflowBufferSpec::borrowed_from`. The reader only `reserve_back`/`push_back`es it (`reader:60-61`); there is no NoC read to translate.
- `update_idxs_tensor` (optional) — **Case 1**. `Buffer*` → reader RTA[2] (`factory:409`); kernel `reader:74`, read at `:79`.
- `page_table` (optional) — **Case 1**. `Buffer*` → reader RTA[4] (`factory:415`); kernel `reader:95`, read at `:98-103`.

**`PagedFillCacheDeviceOperation`** — all five Case 1:

- `input_tensor` — **Case 1**. reader RTA[0] (`factory:302`) → `reader_fill:29`.
- `cache_tensor` — **Case 1**. writer RTA[0] (`factory:311`) → `writer_fill:145`.
- `page_table` — **Case 1**. writer RTA[1] (`factory:312`) → `writer_fill:146`.
- `batch_idx_tensor` (optional) — **Case 1**, and the slot is overloaded; see *Overloaded RTA slots* below. writer RTA[4] (`factory:315-319`) → `writer_fill:101`.
- `valid_seq_len_tensor` (optional) — **Case 1**, overloaded slot. writer RTA[6] (`factory:323-327`) → `writer_fill:122`.

**`PagedFusedUpdateCacheDeviceOperation`** (identical in the tiled and row-major factories):

- `cache_tensor1` — **Case 1**. reader RTA[2] & writer RTA[1] **on `cores1` only** (tiled `:438`, `:456-467`; RM `:435`, `:453-465`). Kernel `reader:82`, `writer:60` (tiled) / `:67` (RM).
- `cache_tensor2` — **Case 1**, reaching the **same** RTA slots on `cores2` (tiled `:483`, `:501-512`; RM `:481`, `:499-511`). See *Two tensors, one arg slot, selected by core* below.
- `input_tensor1` — **clean** (borrowed DFB `c_1` over `input1_cores`; tiled `:203-212`, RM `:208-217`; re-pointed at `paged_fused_update_cache_device_operation.cpp:73`).
- `input_tensor2` — **clean** (borrowed DFB `c_2` over `input2_cores`; tiled `:213-222`, RM `:218-227`; re-pointed at `:74`).
- `update_idxs_tensor` (optional) — **config-split: Case 1 on DRAM-interleaved · clean on L1-sharded.** It is *both* a `Buffer*` RTA (tiled `:441`, RM `:438`) *and* a borrowed CB `c_3` when the tensor is sharded (`.buffer = index_buffer_ptr`, tiled `:276`, RM `:272`; the pointer is `nullptr` when not sharded, tiled `:117`, RM `:117`). Kernel: DRAM path builds the accessor and reads (`reader:87,92`); L1-sharded path compiles the read out via `if constexpr (index_is_dram)` and reads straight from the borrowed CB (`reader:96,98`). **Both paths must survive** — express the `TensorParameter` and make the DFB's `borrowed_from` conditional on the sharded case, exactly as the legacy factory conditions `.buffer`.
- `page_table` (optional) — **config-split: Case 1 on DRAM · clean on L1-sharded.** Same double shape: `Buffer*` reader RTA[6] (tiled `:447`, RM `:444`) plus borrowed CB `c_4` when sharded (tiled `:289`, RM `:285`). DRAM: `reader:109-116` (tiled) / `:113-120` (RM). L1-sharded: a pointer walk inside the borrowed CB (`reader:118` / `:122`, `writer:91` / `:98`) — preserve that arithmetic verbatim.

**No Case 2 anywhere.** Every *tensor* base is fed to a `TensorAccessor`; nothing does hand-rolled NoC arithmetic on a tensor base, so **no `get_bank_base_address` bridge is needed**. The raw pointer arithmetic that does exist (`page_table_cb_wr_ptr += my_batch_idx * page_table_stick_size`) walks an **L1 CB** pointer, not a tensor base — the borrowed-DFB translation preserves it unchanged.

**None of these 15 bindings is a correctness fix.** Every buffer already arrives as an annotated `Buffer*` (auto-registered `BufferBinding`, patched on cache hits), and every factory's `override_runtime_arguments` re-applies addresses on top of that. This is routine port work.

**TensorParameter relaxation:** none — the only value that reaches a brief.

**TensorAccessor 3rd arg:** none. All 17 `TensorAccessor(...)` constructions in the op are the 2-arg form; there is no page-size override to drop and no `dynamic_tensor_shape` to set.

### CB endpoints

**29 `(CB, config)` instances: 26 plain 1:1, 3 self-loop, nothing else.** No multi-binding advanced option anywhere, no dead-CB drop, no new conditional DFB.

- **Self-loop (bind the one kernel PRODUCER *and* CONSUMER)** — all three in `fill_cache`, all touched only by the writer:
  - `page_table_cb_index` `c_1`, **all configs** — `reserve_back(1)` @`writer_fill_cache_interleaved.cpp:148` + raw write via `get_write_ptr()` @`:149` + repeated `noc.async_read` into that pointer @`:210-216`; never pushed, never popped, no other kernel references the index.
  - `cb_batch_idx_id` `c_2`, **`use_batch_idx_tensor` only** — `reserve_back(1)` @`:102` + raw @`:103-113`.
  - `cb_valid_seq_len_id` `c_3`, **`use_valid_seq_len` only** — `reserve_back(1)` @`:123` + raw @`:124-128`.
- **All 26 others are genuine 1P+1C FIFOs** whose roles the existing ops already fix — no assignment decision, no flag. The full per-`(CB, config)`-per-node census is in `METAL2_PREPORT_AUDIT.md` → *Gate detail → CB endpoints*; read it before writing the DFB specs, because **several CB-index CTA names are counter-intuitive**: in all three writer kernels the CTA named `cache_cb_id` is the *output* CB (`c_16` tiled/update, `c_7` RM), not the cache CB `c_0`. Resolve every index through the factory's arg list, not the kernel's local name.
- The configuration-optional CBs (`cb_index`, `cb_pagetable`, and `fill_cache`'s `cb_batch_idx` / `cb_valid_seq_len`) are **already conditionally allocated host-side** (`if (use_index_tensor)` / `if (is_paged_cache)` / `if (use_batch_idx_tensor)` / `if (use_valid_seq_len)`), so those guards translate directly onto conditional `DataflowBufferSpec`s. Nothing is dead-in-some-configs-live-in-others in a way that needs new structure.

### The aliased two-format intermediate DFB — one DFB, two format descriptors. Do not split it.

In three of the four factories a *single* `CBDescriptor` carries **two** `CBFormatDescriptor`s, so two buffer indices alias one L1 allocation:

- `update_cache`: `c_24` + `c_25` — `paged_update_cache_program_factory.cpp:210-225`
- tiled fused: `c_24` + `c_25` — `paged_tiled_fused_update_cache_program_factory.cpp:223-238`
- row-major fused: `c_5` + `c_6` — `paged_row_major_fused_update_cache_program_factory.cpp:228-243`

**The aliasing is the algorithm.** Compute untilizes a cache block and publishes it through index 0; the writer takes `cb_untilized_cache.get_read_ptr() + cache_tile_offset_B` (index 0), NoC-writes the new row into that L1 region **in place**, then publishes the *same memory* through index 1 via `cb_untilized_cache2.push_back(Wt)` for compute to re-tilize — `writer_update_cache…:125,133`; `writer_paged_fused…:137,145`; `writer_paged_row_major…:139,147`.

Port as **one** `DataflowBufferSpec` with two format descriptors. Splitting it into two independent DFBs compiles, validates, and silently produces wrong numerics.

---

## Watch for

- **CB endpoints (multi-binding):** **none.** All three faces were hunted and came back empty — no hidden second writer, no multi-reader CB, no dual-instance work-split. The only semaphore in the op coordinates `share_cache` ordering between kernels on *different cores*, not a raw CB co-fill.

- **★ Runtime-selected DFB index across two DFBs with disjoint core ranges (both fused factories) — resolve this before writing the fused specs.** Three kernels pick which input CB to touch from a **runtime** arg:
  - `reader_paged_fused_update_cache_interleaved_start_id.cpp:30-35` and `reader_paged_row_major_…:30-35` — `input_cb_id = is_input1 ? input1_cb_id : input2_cb_id`, then `CircularBuffer cb_input(input_cb_id)` at `:67`.
  - `writer_paged_row_major_…:59-62` — same selection for `untilized_input_cb_id`, then `CircularBuffer cb_untilized_input(...)` at `:72`.
  - `compute/paged_fused_update_cache.cpp:21-26,36` — `compute_kernel_hw_startup(in_cb, …)` on the runtime-selected index, then a branch at `:39-55` into **two** compile-time-parameterised `untilize<Wt, in1_cb, …>` / `untilize<Wt, in2_cb, …>` instantiations, both compiled into every node's binary.

  Since a kernel cannot touch a DFB it has not bound, each of these `KernelSpec`s must bind **both** `src1` and `src2`. But `src1`'s `CBDescriptor` covers only `input1_cores` and `src2`'s only `input2_cores`; the two are validated **disjoint** (`paged_fused_update_cache_device_operation.cpp:350-351`) and equal in core count (`:352-356`), while every `KernelSpec` spans `all_cores_bb`. So on any given node exactly one of the two bound DFBs actually exists. Nothing in Appendix A covers this, so it is not a gate — but it is not mechanical either. The audit raises it as an open design question (`METAL2_PREPORT_AUDIT.md` → *Questions* #1): bind both and rely on per-node existence, or split into per-core-set `KernelSpec`s. **Get an answer before you write the fused specs**; it will shape them more than any other single choice.

- **★ Two tensors, one arg slot, selected by core (both fused factories).** Reader RTA[2] and writer RTA[1] carry `cache_tensor1` on `cores1` and `cache_tensor2` on `cores2` — tiled `:438` vs `:483` and `:456-467` vs `:501-512`; RM `:435` vs `:481` and `:453-465` vs `:499-511`. `patch_runtime_args` patches them the same way (`paged_fused_update_cache_device_operation.cpp:100-124`: `patch_core(cores1[i], dst1_addr)` / `patch_core(cores2[i], dst2_addr)`). A `tensor::name` binding is per-`KernelSpec`, so one reader spec would need *two* `TensorParameter`s reaching one arg position by node. Same structural question as the bullet above, on the tensor-binding channel rather than the DFB channel — resolve them together.

- **Overloaded RTA slots — tensor-or-scalar by config. One of the three instances is hard; two are easy.**
  - **`fill_cache` (hard).** Writer RTA[4] is a `Buffer*` when `use_batch_idx_tensor` and the *meaningful* scalar `operation_attributes.batch_idx_fallback` otherwise (`paged_fill_cache_program_factory.cpp:315-319`, patched `:413`); writer RTA[6] is a `Buffer*` or literal `0` (`:323-327`, patched `:415`). In Metal 2.0 the binding channel and the named-scalar channel are different, so RTA[4] must split into a **config-conditional `TensorParameter` plus a named scalar arg**. The kernel already keys off the same CTA (`writer_fill_cache_interleaved.cpp:56`, `:100-116`), so the branch exists — it just has to move onto two channels.
  - **`update_cache` and both fused (easy).** The scalar alternative is a literal `0` the kernel never reads (access is behind `if constexpr`): `paged_update_cache_program_factory.cpp:408-418`; tiled `:440-450`; RM `:437-447`. These collapse to a conditional `TensorParameter` with no scalar counterpart.

- **Per-core runtime-arg count varies within one `KernelSpec` (both fused factories).** Working cores get 8 reader args, 8 writer args (tiled) or 9 (RM, which appends `is_input1` at `:464`/`:510`), and 2 compute args. Every node in `unused_cores = all_cores_bb − all_cores` gets a **single** arg `{!has_work}` (tiled `:524-530`; RM `:523-529`) and early-returns on it (`reader:17-20`, `writer:18-21`, `compute:15-18`). A `runtime_arg_schema` is one schema for the whole `KernelSpec`, so decide up front how the short-arg nodes are expressed (supply the full named set with don't-care values, or narrow the `KernelSpec`'s core range) rather than discovering it at validation time. `unused_cores` is non-empty only when `input1_cores ∪ input2_cores` is not itself a rectangle. Note that these CBs are **not** dead on those nodes — the early return is runtime control flow, and the `buffer_index` is still statically referenced by the kernels bound over that range. Do not drop them.

- **`override_runtime_arguments` is index-addressed, and every mismatch is silent.** Each of the four bodies reaches args by hard-coded positional index and CBs by hard-coded position in `desc.cbs`:
  - `update_cache`: `kReaderKernelIdx=0` / `kWriterKernelIdx=1` (`:478-479`), `kInputCbPos = 1` (`:485`), then `reader_args[0]/[2]/[4]` and `writer_args[0]/[1]/[2]` (`:503-515`), and `UpdateDynamicCircularBufferAddress(program, program.circular_buffers().at(kInputCbPos)->id(), …)` (`:518`).
  - `fill_cache`: `reader_args[0]/[3]`, `writer_args[0]/[1]/[4]/[5]/[6]` (`:406-415`). No CB re-pointing — none of its four CBs is globally allocated (`:417`).
  - both fused: `kReaderCacheAddrArg=2`, `kReaderCacheStartIdArg=3`, `kReaderIndexAddrArg=4`, `kReaderPageTableAddrArg=6`, `kWriterCacheAddrArg=1`, `kWriterCacheStartIdArg=2`, `kWriterTileUpdateOffsetArg=3` (`paged_fused_update_cache_device_operation.cpp:28-37`), plus `kSrc1CbPos=1`, `kSrc2CbPos=2`, `kFirstOptionalCbPosTiled=6`, `kFirstOptionalCbPosRowMajor=5` (`:42-45`), walked with a post-increment at `:75-81`.

  Named args and named DFB bindings **delete** all of these constants — that is the point — but read `kFirstOptionalCbPos*` together with the `if (use_index_tensor)` / `if (is_paged_cache)` push order in each factory before rewriting: the two values differ only because tiled pushes one extra `intermed2` descriptor, and that offset is maintained by hand.

- **Kernels declare CB wrapper objects unconditionally for conditionally-allocated CBs.** `CircularBuffer cb_index(cb_index_id)` / `cb_page_table(page_table_cb_id)` at `reader_update_cache:56-57`, `writer_update_cache:61-62`, `reader_paged_fused:69-70`, `writer_paged_fused:66-67`, `reader_paged_row_major:69-70`, `writer_paged_row_major:73-74`, and `writer_fill_cache:92-93` — constructed even when the guarding CTA is false and the CB was never allocated. Harmless today (every *access* is behind `if constexpr`), but a Metal 2.0 binding is not a no-op the way an unused `CircularBuffer(id)` is. Check whether the binding must be made conditional alongside the `DataflowBufferSpec`.

- **`share_cache` cross-core semaphore chain — ports as an ordinary `SemaphoreSpec`, but keep the trailing barrier.** One plain `SemaphoreDescriptor` per factory (`paged_update_cache_program_factory.cpp:247`; tiled `:260`; RM `:256`). Writer *i* signals reader *i+1*: `Semaphore<>::up(noc, send_core_x, send_core_y, 1)` at `writer_update_cache:164`, `writer_paged_fused:176`, `writer_paged_row_major:178`; awaited at `reader_update_cache:126-128`, `reader_paged_fused:152-154`, `reader_paged_row_major:154-156`. The `send_core_x/y` are **physical** coordinates baked host-side via `worker_core_from_logical_core` (`paged_update_cache_program_factory.cpp:394`; tiled `:422,427`; RM `:419,424`). The `noc.async_atomic_barrier()` that follows each `.up()` (`writer_update_cache:165`, `writer_paged_fused:182`, `writer_paged_row_major:184`) carries a comment documenting a real Watcher NOC-idle race — **keep it**.

- **Two different mesh-filtering idioms; neither is the port's to normalise.** `update_cache` and both fused families return an **empty `ProgramDescriptor`** for a coordinate outside `mesh_coords` (`paged_update_cache_program_factory.cpp:448-453`; tiled `:544-549`; RM `:547-552`) and early-return from the override on the same test (`paged_update_cache_program_factory.cpp:472-475`; `paged_fused_update_cache_device_operation.cpp:129-134`, called at `:402`, `:416`). `fill_cache` instead builds a **full descriptor whose kernels early-exit** on a `noop` RTA (`paged_fill_cache_program_factory.cpp:33-40`, `:348-359`; consumed at `reader_fill_cache_interleaved.cpp:21,23-25` and `writer_fill_cache_interleaved.cpp:77,80-82`) so the cache slot is still populated for that coordinate, and its override re-patches `noop` (`:399,408,414`). Preserve both behaviours as they are.

- **Cross-op / shared kernels: none — but a basename grep will lie to you.** All 11 kernels this op instantiates live in `device/kernels/` here, and only this op's four factories instantiate them. **No `_metal2` fork exists beside any of them and none is needed — no fork to create, no sunset list, no cross-op coordination.** However, `ttnn/cpp/ttnn/operations/kv_cache/device/kernels/` holds three files with *identical basenames* (`dataflow/reader_update_cache_interleaved_start_id.cpp`, `dataflow/writer_update_cache_interleaved_start_id.cpp`, `compute/update_cache.cpp`) that are **separate private copies**, instantiated by `kv_cache`'s own factories via their own paths (`update_cache_multi_core_program_factory.cpp:291,319,374`; `fill_cache_multi_core_program_factory.cpp:197`). Editing them does nothing for this op. Confirm the *path* on every kernel you open.

- **Donor functions: all ✓, no donor-side change.** The three compute kernels call `compute_kernel_lib::untilize<Wt, input_dfb, output_dfb, …>` and `tilize<Wt, input_dfb, output_dfb, …>` from `ttnn/cpp/ttnn/kernel_lib/{untilize,tilize}_helpers.hpp`. CB handles are `uint32_t` **NTTPs**, which `dfb::name`'s constexpr cast covers in template-parameter position, and the donors are already DFB-aware internally (`untilize_helpers.inl:199-200`, `tilize_helpers.inl:149-150` build `DataflowBuffer` from the index). Every other include resolves under `tt_metal`'s `api/*`. No fork, no bridge, no cross-team discussion.

- **RTA / CTA varargs: none — name every argument.** No kernel reads an arg at a loop-variable or data-selected index. The six fused kernels use a **fixed run** of `rt_args_idx++` at the top (8–9 reads each), which is legacy positional plumbing that dissolves into named args, not a vararg block. Every `get_compile_time_arg_val` index is a literal; the only computed CTA offsets are `TensorAccessorArgs<N>()` chained through `constexpr next_compile_time_args_offset()` (`reader_update_cache:48-50`, `writer_fill_cache:84-88`, and the fused equivalents), a fixed set at constexpr offsets. No `get_common_arg_val` anywhere. Reach for neither the RTA vararg mechanism nor `compile_time_varargs`.

- **Dead CTAs you will meet and should leave alone.** Several kernels read compile-time args that nothing uses — `log_base_2_of_page_size` (always host-side `0`), `log2_page_table_stick_size`, `max_blocks_per_seq`, and the RM compute kernel's `in1_cb`/`in2_cb`/`is_input1` (already `[[maybe_unused]]`). They are catalogued as team-only anomalies in `METAL2_PREPORT_AUDIT.md` → *Misc anomalies* and route to the ops team. **The port does not remove them** — carrying a dead named arg through is not the same as a dead CB, and dropping one is a functional change to the arg schema that no longer matches the ops team's source of truth.

# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/transformer/sdpa_decode`

Single device operation, single program factory (one `create_descriptor`), three op-owned kernels:

- **`SdpaDecodeDeviceOperation`** (`ttnn::prim`) — concept `descriptor`, sheet variant *single-descriptor*
  - `create_descriptor` in `device/sdpa_decode_program_factory.cpp`
  - Kernels (all file-path-instantiated from the op's **own** `device/kernels/`, all over the full `core_grid`):
    - reader `device/kernels/dataflow/reader_decode_all.cpp` (`ReaderConfigDescriptor`)
    - writer `device/kernels/dataflow/writer_decode_all.cpp` (`WriterConfigDescriptor`)
    - compute `device/kernels/compute/sdpa_flash_decode.cpp` (`ComputeConfigDescriptor`)
  - Kernel-shared headers: `device/kernels/dataflow/dataflow_common.hpp`, `device/kernels/rt_args_common.hpp`

The four nanobind entry points (`scaled_dot_product_attention_decode`, `paged_...`, `flash_multi_latent_attention_decode`, `paged_flash_...`) all route to this one device operation; paged / MLA / sharded are internal branches of the single `create_descriptor`, not separate factories.

No unreferenced kernel files in the directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `c9ef66ee339 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `SdpaDecodeDeviceOperation` → single `create_descriptor` (sheet: *single-descriptor*) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — own kernels + all donors are Device 2.0 (only sanctioned `get_tile_size(cb)` free functions) |
| *Prereqs* — Cross-op escapes | Ok (workable; no sequence-blockers) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — GlobalCircularBuffer / `address_offset` / GlobalSemaphore | N/A / N/A / N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD | N/A (concept is `descriptor`, not `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | No (sheet `Formerly custom hashed?`=yes; grep confirms none now) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none (no `->address()` fold; buffers bound as clean `Buffer*`) |
| *Port work* — Tensor bindings (per binding) | Case 1 (most) + one Case 2 (Q, sharded non-MLA) + clean borrowed-DFB (sharded configs) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | 2 sites, both **Class 2** (redundant → drop; no `dynamic_tensor_shape`) |
| *Port work* — CB endpoints | 1P+1C + self-loops + one **multi-binding** (`c_16`, tree reduction) + borrowed-DFB (sharded) + one possible dead CB (`c_11`) |

**CB endpoints** are dispositions, not gates. Recorded per `(CB, config)` below; the disposition of `c_0` and `c_16` flips with config.

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓, Offset base pointers ✓, TensorAccessor 3rd argument ✓. `METAL2_PORT_BRIEF.md` is written alongside this file. No portable-subset scoping needed — the op ports whole.

## Gate detail

- **TTNN factory concept (`Is able to port?` = `yes`):** GREEN. Readiness sheet row `transformer/sdpa_decode` / `SdpaDecodeDeviceOperation (single-descriptor)`: `Concept=descriptor`, `Is able to port?=yes`, `TensorParameter relaxation=none`, `Known op issues=` (empty), `Smuggled pointer=no`, `Op-owned tensors=` (empty). Cross-check vs. code all clean:
  - `Concept=descriptor` ✓ — `create_descriptor(...)` returns `tt::tt_metal::ProgramDescriptor` (`sdpa_decode_device_operation.hpp:95`, `sdpa_decode_program_factory.cpp:29`).
  - `Custom hash=no` ✓ — no `compute_program_hash` anywhere in the op (grep NONE). Sheet notes `Formerly custom hashed?=yes` / `Pointer patching perf issue? = OK (old custom hash was complete, and dumb)` — the hash was removed; no residual.
  - `get_dynamic_runtime_args=no` ✓ — grep NONE on the device-op.
  - `override_runtime_arguments=no` ✓ — grep NONE. (Selects the plain target concept.)
  - `Pybind descriptor=no` ✓ — `sdpa_decode_nanobind.cpp` binds only the four user functions via `ttnn::bind_function<>`; no `nb::class_` of the device op, no `create_descriptor` binding.
  - **Factory-set match** ✓ — one sheet row ↔ one `create_descriptor` in the code.
  - `Op Classification = PD Op (pointer-patching)` — the op uses the `Buffer*`-binding (`BufferBinding`) form; see Port-work / TensorParameter analysis. Correct-on-cache-hit, not the silent-wrong hazard.

- **Device 2.0 (every kernel used):** **GREEN.** All data movement uses the Device 2.0 `Noc` object (`noc.async_read`/`async_write`/`*_barrier`/`async_write_multicast`), CBs via `CircularBuffer` wrapper objects, semaphores via `Semaphore<>` objects, DRAM addressing via `TensorAccessor`. No `InterleavedAddrGen`/`ShardedAddrGen`/`get_noc_addr_from_bank_id`, no raw `noc_async_*`, no `noc_semaphore_*`, no `evil_set_*` cursor mutators, no CB-index free-function holdovers — the only CB-index free functions are the **sanctioned** `get_tile_size(cb)` (Green-bullet allowed). Verified for the op's three kernels **and** every donor header the op `#include`s (see Team-only → coupling). No violations to route.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | *(none)* | — | — | — |

- **Feature compatibility:** every Appendix A entry N/A (no `GLOBAL` feature present). Scanned host code, factory, all three kernels, and all donor headers.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer`, `.global_circular_buffer` field, `remote_index`/`remote_cb`, or 4-arg `CreateCircularBuffer`. CBs are plain `CBDescriptor` (some borrowed via `.buffer`). |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set anywhere; `add_cb` never assigns it. Borrowed CBs use base only (`cb.buffer = buffer`). |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` / `CreateGlobalSemaphore`. Three plain `SemaphoreDescriptor`s (reducer / output / k_mcast). |

- **CB endpoints (GATE-free):** classified per `(CB, config)` — full census in Port-work summary. One **multi-binding** (`c_16`), several self-loops, one possible dead CB (`c_11`), four borrowed-memory CBs in sharded configs. Nothing here blocks. Device 2.0 is GREEN, so the census ran against intact Device-2.0 idioms (not deferred).

- **Offset base pointers:** **GREEN.** No address RTA anywhere folds a host-side offset. The factory binds every buffer as a clean `Buffer*` (`q/k/v/cur_pos/page_table/attn_mask/attention_sink` on the reader, `out` on the writer — `sdpa_decode_program_factory.cpp:919-925, 947`); there is **no** `->address()` call in the factory (grep NONE) and hence no `base + offset` fold. The one raw-pointer kernel use (sharded Q, Case 2 below) consumes the **clean base** `q_addr` and does its own kernel-side tile striding — not a host-folded offset. Not in the offset-base-pointers triage tables (`2026-07-19_offset_base_pointers.md`); scan confirms clean, consistent with the doc.

- **TensorAccessor 3rd argument:** **GREEN** — two sites, both **Class 2 (redundant → drop)**. (The triage `2026-07-06_tensor_accessor_3rd_arg_triage.md` lists `sdpa_decode (page-table)` as Class 2; my read finds a *second* 3rd-arg site on **Q** the dated doc did not catch. Both classify Class 2 — see Recipe notes for the discrepancy.)
  - **Q accessor** — `device/kernels/dataflow/dataflow_common.hpp:572`, `TensorAccessor(q_args, q_addr, q_page_size_bytes)`, reached only in the **non-sharded (DRAM interleaved)** branch of `read_q`. `q_page_size_bytes` = CTA #29 = `full_tile.get_tile_size(q_df)` (`sdpa_decode_program_factory.cpp:677`), i.e. the true full 32×32 tile page in Q's format — **correct magnitude** (block-float-safe). Interleaved + correct-magnitude → **Class 2**, clean drop. Do **not** set `dynamic_tensor_shape` (page size is compile-time-pinned, not width-varying).
  - **Page-table accessor** — sdpa in-family donor `../sdpa/device/kernels/dataflow/dataflow_common.hpp:83`, `TensorAccessor(page_table_args, page_table_addr, page_table_stick_size)`, inside `read_page_table_for_batch` (called by sdpa_decode's reader in the non-sharded paged path). 3rd arg is `page_table_buffer->aligned_page_size()` (`sdpa_decode_program_factory.cpp:229`) → literally `== aligned_page_size`, page_table is ROW_MAJOR DRAM (interleaved) → **Class 2**, clean drop. (Drop lands in the shared donor — see shared-kernel heads-up.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding; classification varies per config — the op is one factory with paged/MLA/sharded branches):
  - **q** — **Case 1** (DRAM interleaved: `TensorAccessor(q_args, q_addr, q_page_size_bytes)`, `dataflow_common.hpp:572`) · **Case 2** (HEIGHT_SHARDED, non-MLA: `q_addr` used raw as an L1 NoC address in the core-to-core read, `reader_decode_all.cpp:211`→`read_q` `dataflow_common.hpp:544-559`; bind + bridge via `get_bank_base_address`, keep the raw read) · **clean** (MLA `q_locally_available`: `c_0` borrowed from `q_buffer`, `read_q` just reserve/push, `dataflow_common.hpp:513-521`).
  - **k** — **Case 1** (`TensorAccessor(k_args, k_addr)`, `reader_decode_all.cpp:222`; always DRAM).
  - **v** — **Case 1** (`TensorAccessor(v_args, v_addr)`, `reader_decode_all.cpp:224`; used only when `!reuse_k`. Under MLA `reuse_k`, V data is read from K's L1 and there is no independent V tensor).
  - **cur_pos_tensor** — **Case 1** (`TensorAccessor(pos_args, pos_addr)`, `reader_decode_all.cpp:129`, DRAM) · **clean** (sharded: `c_8` borrowed from `cur_pos_buffer`).
  - **page_table_tensor** — **Case 1** (via `read_page_table_for_batch`'s `TensorAccessor`, DRAM) · **clean** (sharded: `c_9` borrowed from `page_table_buffer`).
  - **attn_mask** — **Case 1** (`TensorAccessor(mask_args, mask_addr)`, `reader_decode_all.cpp:226`; DRAM).
  - **attention_sink** — **Case 1** (`TensorAccessor(attention_sink_args, ...)`, `reader_decode_all.cpp:230`; DRAM).
  - **output** — **Case 1** (`TensorAccessor(out_args, out_addr)`, `writer_decode_all.cpp:246`, DRAM) · **clean** (sharded: `c_20` borrowed from `out_buffer`).
  - Delivery mechanism today is the `Buffer*`-binding (`BufferBinding`) form (`emplace_runtime_args` with `Buffer*`, `sdpa_decode_program_factory.cpp:918-1001`) — correct-on-cache-hit, not the silent-stale hazard; the port replaces it with typed `TensorParameter` bindings regardless.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** drop the redundant page-size arg at (a) `dataflow_common.hpp:572` (Q) and (b) sdpa donor `read_page_table_for_batch` `../sdpa/.../dataflow/dataflow_common.hpp:83` (page-table). Both Class 2; neither sets `dynamic_tensor_shape`. Confirm-not-swap-blind for Q: Metal 2.0 supplies `aligned_page_size` from Q's **DRAM buffer** spec (full tile), which equals the dropped override even when the Q CB `c_0` is a half-tile under `use_half_tile`.
- **CB endpoints** (per `(CB, config)`; all kernels span the full `core_grid`):
  - **1P+1C:** `c_1` (K: reader→compute), `c_2` (V: reader→compute), `c_3` (mask: reader *or* writer→compute, mutually-exclusive by `is_causal`), `c_4` (sink: reader→compute), `c_5` (scale: writer→compute), `c_6` (m_in: writer→compute), `c_7` (l_in: writer→compute), `c_8` (cur_pos-writer: reader→writer), `c_10` (q_rm tilize: reader→compute), `c_12` (zero: writer→compute), `c_13` (sliding mask: writer→compute), `c_14` (block-pad mask: writer→compute), `c_15` (cur_pos-compute: reader→compute), `c_17` (out_m: compute→writer), `c_18` (out_l: compute→writer), `c_20` (out: compute→writer).
  - **self-loop (1 toucher):** `c_9` (page_table: reader produces + raw-reads its own buffer), `c_19` (intermed_out: writer raw read/write only, cross-core via NoC, no FIFO ops; allocated only when `num_cores_per_head>1`), and the compute-only intermediates `c_21 c_22 c_23 c_24 c_25 c_26 c_27 c_28 c_29 c_30 c_31` (each produced+consumed within compute).
  - **multi-binding (advanced option):** **`c_16`** (`cb_out_o` == `cb_out_worker`) — bidirectionally reused by the tree reduction: **writer produces** (receive child O, `writer:325-334`) + **compute consumes** (`writer`-fed reduction, `compute:571`), *and* **compute produces** (own reduced O, `compute:666`) + **writer consumes** (send to parent, `writer:347-389`). On an **intermediate** tree node (exists when `num_cores_per_head ≥ 4`) all four bindings are live → 2 locked producers + 2 locked consumers. `move_block` is FIFO (`compute_common.hpp:772`), so both roles are locked, not relabellable. On root/leaf nodes it collapses to 1P+1C. Set the multi-binding advanced option on `c_16` (covers the intermediate-node config).
  - **borrowed-DFB (`DataflowBufferSpec::borrowed_from`) in sharded configs:** `c_0` (Q, MLA `q_locally_available`), `c_8` (cur_pos sharded), `c_9` (page_table sharded), `c_20` (output sharded). Host binds via `cb.buffer = buffer` today (`add_cb`, `sdpa_decode_program_factory.cpp:533-535`).
  - **config-flip:** `c_0` (q_in) — self-loop (tilize_q: compute tilizes `c_10`→`c_0` then consumes) / 1P+1C (DRAM non-tilize: reader→compute) / borrowed (MLA local).
  - **possible dead CB → confirm, do not drop blind:** `c_11` (col_identity) — see Misc anomalies; disposition is self-loop (writer-only) unless the ops team confirms it can be removed.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding to watch):** `c_16` — the hidden shape is the *bidirectional reuse* (receive-child vs. send-to-parent), not a raw co-fill. A hurried 1P+1C binding will under-bind it on intermediate tree nodes (`num_cores_per_head ≥ 4`). No hidden raw second-writer elsewhere (K-multicast and tree-reduction cross-core writes target each node's *own* CB instance at a same-offset address, so per node they stay 1P+1C / self-loop).
- **Cross-op / shared kernels (function-call escapes; sunset/coordination list, NOT a bundled-port authorization):**
  - In-family donors shared with **sdpa (prefill)**: `../sdpa/device/kernels/compute/compute_common.hpp` and `../sdpa/device/kernels/dataflow/dataflow_common.hpp`. In-family escapes don't gate the Metal 2.0 syntax rewrite (boundary features bridge; donor headers not rewritten). The Q/page-table 3rd-arg drop for page-table lands inside the shared `read_page_table_for_batch`.
  - `ttnn/kernel/dataflow/generate_bcast_scalar.hpp` (`generate_bcast_col_scalar` takes a **`CircularBuffer`** by value — the ⭐ shape) — **a `_metal2` fork already exists beside it** (`generate_bcast_scalar_metal2.hpp`, takes `DataflowBuffer`); bind the fork, don't re-fork. (This call feeds the possibly-dead `c_11` — see anomalies.)
  - `ttnn/cpp/ttnn/kernel_lib/{tilize_helpers,untilize_helpers,reduce_helpers_dataflow,l1_helpers}.hpp` — official shared lib; CB-index / template / `DataflowBuffer` shapes (all ✓). Lib-team-owned.
- **RTA varargs (use the vararg mechanism, don't try to name each):** variable-count physical-core-coordinate arrays read via `get_arg_addr` + runtime index —
  - reader: `all_output_noc_x/y` (`reader_decode_all.cpp:184-186`, count `num_output_cores`).
  - writer: `reduction_group_core_xs/ys` (`writer_decode_all.cpp:108-111`, count `num_cores_per_head`), `all_reducer_noc_x/y` (`188-191`, count `num_reducer_cores`), `all_output_noc_x/y` (`192-194`, count `num_output_cores`).
  - The `children_per_round[MAX_TREE_REDUCTION_ROUNDS]` block (fixed 6, read in a bounded loop — reader/writer/compute) is **nameable**, not a vararg.
- **Reducer semaphore raw poll:** the writer resolves `get_semaphore(reducer_semaphore_id)` and polls the value directly with a custom 4-bit-per-round nibble decode (`writer_decode_all.cpp:32,248-249,275-278`) because `Semaphore<>::wait(threshold)` can't express per-round counters. Not a Device 2.0 idiom (no `noc_semaphore_*`; the semaphore is a real `SemaphoreDescriptor` and `Semaphore<>` handles up/wait/set elsewhere). The port must bind this semaphore via `sem::name` and derive the poll address from the named binding, not just swap the `Semaphore<>` constructions.

## Team-only

- **Out-of-directory coupling & donor shape** — roll-up **⚠ workable** (no ⭐ sequence-blockers; no donor on pre-Device-2.0 idioms → no donor-side Device 2.0 gate). Borrowed *kernel files* (file-path instantiation): **none** — the op instantiates only its own three kernels.

  | Op kernel | Donor file | Class | Shape(s) called | Status |
  |---|---|---|---|---|
  | compute | `../sdpa/.../compute/compute_common.hpp` | in-family | `reduce_c`, `matmul_blocks`, `sub_exp_block*`, `correction_block`, `move_block`, `max_block`, `recip_block_inplace`, `mul/add_block*` — all take `uint32_t cb_id` | ✓ OK |
  | reader/writer | `../sdpa/.../dataflow/dataflow_common.hpp` | in-family | `read_page_table_for_batch(Noc&, uint32_t cb, …, TensorAccessorArgs, uint32_t addr, uint32_t page_size)`; `copy_tile(Noc&, …)`; `virtual_seq_tile_id_to_physical_tile_id(...)` | ✓ (`Noc&`, `uint32_t cb`) / ⚠ workable (`TensorAccessorArgs` Shape 2 → pass `tensor::name.args`) |
  | compute | `ttnn/kernel_lib/{tilize,untilize}_helpers.hpp` | kernel_lib | `compute_kernel_lib::tilize/untilize<cb_id,…>()` (template CB-id) | ✓ OK |
  | writer | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | kernel_lib | `calculate_and_prepare_reduce_scaler<cb_id,…>()` | ✓ OK |
  | writer | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` | kernel_lib | `prepare_zero_tile<dfb_id>()`, `zero_tile(::DataflowBuffer)` | ✓ excellent (already `DataflowBuffer`) |
  | writer | `ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | kernel (singular) pool | `generate_bcast_col_scalar(CircularBuffer, uint32_t)` | ⭐→resolved: `generate_bcast_scalar_metal2.hpp` fork (takes `DataflowBuffer`) already exists — bind it |

- **TTNN factory analysis (sheet-derived facts + evidence):** current concept `descriptor`; target `ProgramSpecFactoryConcept` (`Override runtime args method?=no`); no op-owned tensors; no pybound `create_descriptor`; no custom hash (removed — `Formerly custom hashed?=yes`); no `get_dynamic_runtime_args`; delivery via `Buffer*` `BufferBinding`s (`Op Classification = PD Op (pointer-patching)`, `Smuggled pointer=no`). `Execution Model=SPMD` with concept `descriptor` (single-program `create_descriptor`) — no `WorkloadDescriptor`, so no secretly-SPMD question.
- **Relaxation candidates mined from a custom hash:** none (no custom hash to mine).

## Misc anomalies  *(team-only, non-gating)*

- **`c_11` (`cb_col_identity`) is produced but never consumed in sdpa_decode.** The writer unconditionally fills it via `generate_bcast_col_scalar(CircularBuffer(cb_col_identity), …)` (`writer_decode_all.cpp:215`), but sdpa_decode's compute kernel references `c_11` only at its `constexpr` definition (`sdpa_flash_decode.cpp:85`) and never passes it to a consumer — it reduces via `reduce_c` on `cb_identity_scale_in` (`c_5`) instead. The column-identity buffer is consumed by the **sdpa prefill** op's `matmul_reduce` (`compute_common.hpp:1918`), so this reads as dead code carried over from a matmul-based reduce path. Effect: one wasted CB allocation (`scale_tiles * col_identity_tile_size`, `c_11`) + wasted writer cycles on every launch. Route to the ops team to confirm removable. Per the dead-CB caution it is **not** dropped by the port (it has one toucher, not zero → the porter self-loops it); the removal is a separate ops-team cleanup.

## Recipe notes

- **3rd-arg triage doc staleness (expected per the dated-doc contract).** `2026-07-06_tensor_accessor_3rd_arg_triage.md` attributes sdpa_decode's sole 3rd-arg use to the *page-table* accessor. Current code has **two** 3rd-arg sites: the page-table one (in the shared sdpa donor, still present and matching) **and** a Q-reader one (`dataflow_common.hpp:572`) the doc doesn't mention. Both classify **Class 2**, so the verdict is unchanged; flagging only so the triage-doc owner can add the Q site. No action needed for this audit (I classified both from the two questions, per the doc's "classify it yourself" instruction).
- **CB reused across two dataflows under one index (`c_16`).** The recipe's multi-binding faces (a hidden raw co-fill, b multiple readers, c dual-instance work-split) don't name the shape here: a *single* index used for two opposite-direction FIFOs (receive-child vs. send-parent) in a tree reduction, giving 2 locked producers + 2 locked consumers on intermediate nodes. It resolves cleanly to the multi-binding advanced option, so it's not a gap — but a fourth "bidirectional-reuse" face would have caught it faster than reasoning it out from the census.

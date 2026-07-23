# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/fold`

- **`Fold`** (single DeviceOperation)
  - `MultiCore` (`device/fold_multi_core_program_factory.cpp`) — height-sharded row-major path
  - `MultiCoreDRAMFold` (`device/fold_multi_core_dram_program_factory.cpp`) — interleaved DRAM path; branches at runtime on input layout into two sub-variants:
    - *tiled* (`fold_multi_core_tiled_interleaved`) — TILE input
    - *row-major* (`fold_multi_core_row_major_interleaved`) — ROW_MAJOR input

All five kernel `.cpp` files under `device/kernels/dataflow/` are referenced by a factory; none are dead. The tiled sub-variant additionally file-path-instantiates the untilize op's compute kernel (`untilize/device/kernels/compute/untilize.cpp`).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `b5b801a923d 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/fold` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `Fold` → `MultiCore`, `MultiCoreDRAMFold` (tiled + row-major sub-variants) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — own DM kernels, pool donor helper, and untilize compute donor all Device 2.0 |
| *Prereqs* — Cross-op escapes | Ok (workable: pool `experimental_device_api.hpp` + in-family `common.hpp` + untilize compute donor) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok — all CTAs constexpr-indexed |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factory rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes (both) |
| *TTNN Readiness* — Custom hash | No (confirmed: no `compute_program_hash` override) |
| *TTNN Readiness* — Runtime-args update | No (confirmed: no `get_dynamic_runtime_args` / `override_runtime_arguments`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (confirmed: plain `bind_function<"fold">`, no `nb::class_`) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none (clean bases; offsets ride separate scalar args) |
| *Port work* — Tensor bindings (per binding) | sharded: clean (borrowed-DFB) · DRAM: Case 1 (both bindings) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none present (all `TensorAccessor(args, addr)` — 2-arg) |
| *Port work* — CB endpoints | legal 1:1 · self-loop · 1P+1C (dual-instance work-split) — see detail |

## Result

**GREEN → brief issued.** All five gates clear:

- **TTNN factory concept:** both factory rows read `Is able to port? = yes` on the readiness sheet; cross-check against the code is clean.
- **Device 2.0:** every kernel the op exercises — its own five dataflow kernels, the pool-owned `experimental_device_api.hpp` helper, and the untilize-owned compute donor — is on Device 2.0 idioms (`Noc`, `experimental::CB`, `TensorAccessor`, DFB-aware compute helpers).
- **Feature compatibility:** no Appendix A feature is in use.
- **Offset base pointers:** no host-folded offset reaches a kernel as a base; the DRAM factories pass a clean `Buffer*` base plus separate scalar index/offset args, and the sharded factory uses borrowed-memory CBs (no address RTA at all).
- **TensorAccessor 3rd argument:** no accessor passes an explicit page-size argument.

The port work is routine (Case-1 tensor bindings on the DRAM factories, borrowed-DFB translation on the sharded factory, CB endpoint dispositions). No portable-subset scoping is needed — the whole op clears.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Both rows (`Fold`/`MultiCore`, `Fold`/`MultiCoreDRAMFold`) are `descriptor` concept with `Is able to port? = yes`. Derivation conjuncts all satisfied — `Is safe to port? = yes`, `Custom hash = no`, `Runtime-args update = no`, `Pybind descriptor = no`, `Concept = descriptor`. Cross-check (trust-but-verify) against the code:
  - `Concept = descriptor` — confirmed: both factories expose `create_descriptor()` returning a `ProgramDescriptor` (`fold_device_op.hpp:30-42`).
  - `Custom hash = no` — confirmed: no `compute_program_hash` anywhere in the op directory.
  - `Runtime-args update = no` — confirmed: no `get_dynamic_runtime_args` / `override_runtime_arguments`.
  - `Pybind descriptor = no` — confirmed: `fold_nanobind.cpp:35` binds via `ttnn::bind_function<"fold">`, no `nb::class_` of the device op, no `create_descriptor` binding.
  - Cross-column invariants hold: `Runtime-args update = no` (legal on `descriptor`); `Op-owned tensors = no` (consistent with `descriptor`).

- **Device 2.0 (every kernel used):** **GREEN.** Idiom census below. All five own dataflow kernels use `Noc`, `noc.async_read`/`async_write`, `experimental::CB` with `reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_read_ptr`/`get_write_ptr`, `use<experimental::CB::AddrSelector::WRITE_PTR>`, and `TensorAccessor`. The only free-function CB lookup in use is `get_tile_size(cb_id)` (`reader_dram2cb_tiled.cpp:20`), which is **sanctioned** by the Device 2.0 migration guide — not a holdover. Donor kernels/headers:

  | File | Owner | Device 2.0? | Notes |
  |---|---|---|---|
  | `pool/device/kernels/experimental_device_api.hpp` | pool (cross-family) | Yes | Device 2.0 helper: `Noc`, `UnicastEndpoint`, `CoreLocalMem`, `noc_traits_t`, trid setters. Included by all 5 DM kernels. |
  | `data_movement/common/kernels/common.hpp` | data_movement (in-family) | Yes | `Noc`-based `enhanced_noc_async_*` / `noc_async_*_sharded`. Included by 3 kernels. |
  | `untilize/device/kernels/compute/untilize.cpp` | untilize (in-family) | Yes | Compute kernel; uses `compute_kernel_hw_startup` + `compute_kernel_lib::untilize` (DFB-aware). No DM idioms to migrate. |
  | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | shared kernel library | Yes | Official shared-lib compute helper (DFB terminology throughout). Lib team owns. |

- **Feature compatibility:** every Appendix A entry, in order:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `.global_circular_buffer` field, `remote_index`, or `CreateGlobalCircularBuffer`. The sharded factory's `cb.buffer = src_buffer` / `dst_buffer` (`fold_multi_core_program_factory.cpp:63,79`) is the ordinary **borrowed-memory** pattern (a mechanical port-recipe translation via `DataflowBufferSpec::borrowed_from`), **not** a GCB. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set (all default 0); no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type or `CreateGlobalSemaphore`. The op uses no semaphores at all. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` carries a single `const Tensor&` (`fold_device_op.hpp:24`), not a variable-count container. All kernel `get_compile_time_arg_val(...)` calls use constexpr literal indices; `TensorAccessorArgs<N>()` uses a fixed `N`. |

- **CB endpoints (GATE-free):** classified per `(CB, config)` — all resolve at port time, none block.

  | Factory / config | CB | Census on a node | Disposition |
  |---|---|---|---|
  | `MultiCore` (sharded) | `c_0` src0 (borrowed) | 2 sync-free raw-readers (both same-source instances raw-read `src_cb_obj.get_read_ptr()`) | **1P+1C** — dual-instance work-split |
  | `MultiCore` (sharded) | `c_16` dst0 (borrowed) | 2 sync-free raw-writers (both instances raw-write `dst_cb_obj.get_write_ptr()`) | **1P+1C** — dual-instance work-split |
  | `MultiCoreDRAMFold` tiled | `c_0` src0 | reader `push_back` (locked P) + compute `wait_front`/`pop_front` (locked C) | **legal 1:1** |
  | `MultiCoreDRAMFold` tiled | `c_1` src1 | compute produces (untilize out) + writer `wait_front`/`pop_front` (locked C) | **legal 1:1** |
  | `MultiCoreDRAMFold` RM | `c_0` src0 | reader `reserve_back`/`push_back` (locked P) + writer `wait_front`/`pop_front` (locked C) | **legal 1:1** |
  | `MultiCoreDRAMFold` RM, `!is_l1_aligned` | `c_1` src1 (scratch) | writer only, raw `get_write_ptr` (sole toucher, sync-free) | **self-loop** |
  | `MultiCoreDRAMFold` RM, `is_l1_aligned` | `c_1` src1 | writer references `get_write_ptr` **but factory does not allocate `c_1`** | **watch-for** — see Heads-ups + Misc anomalies |

  The sharded factory is the canonical dual-instance work-split (face (c)): one `writer_cb2s_row_major.cpp` source pushed into a `WriterConfigDescriptor` and a `ReaderConfigDescriptor` over the same `all_cores` (`fold_multi_core_program_factory.cpp:103-122`), differing only by the `is_reader` CTA (index 13) that splits the output columns (`cols_per_core = num_dst_cols / 2`, `core_col_offset`). Both instances touch both CBs by raw pointer with **no FIFO ops** → role-free → assign 1P+1C, **not** the multi-binding flag. No hidden second FIFO writer, no third toucher.

- **Offset base pointers:** **GREEN.** No address RTA folds a host-side offset into a base:
  - `MultiCoreDRAMFold` tiled: RTAs are `{src0_buffer, block_start_id, nblocks_per_core}` and `{dst_buffer, block_start_id, nblocks_per_core, patch_height_offset, output_offset}` (`fold_multi_core_dram_program_factory.cpp:223-225,236-238`). The `Buffer*` is a clean base (BufferBinding); `block_start_id`/`output_offset` are separate scalar block/stick indices, not folded into the base.
  - `MultiCoreDRAMFold` row-major: RTAs are `{src0_buffer, src_idx, src_col_offset}` and `{dst_buffer, dst_idx}` (`fold_multi_core_dram_program_factory.cpp:395-396`). Again a clean `Buffer*` base; `src_idx`/`dst_idx` are page indices consumed as `.page_id` in the kernel (`reader_dram2cb_for_rm_input.cpp:34`, `writer_cb2dram_for_rm_input.cpp:55,58`).
  - `MultiCore` sharded: no address RTA — borrowed CBs deliver the base implicitly. Not applicable.
  - No `narrow` / interior-base view (Type 4). Not in the offset-base-pointer triage doc, and the scan confirms clean.

- **TensorAccessor 3rd argument:** **GREEN.** Every `TensorAccessor` construction is the 2-arg form `TensorAccessor(args, addr)` (`reader_dram2cb_tiled.cpp:24`, `reader_dram2cb_for_rm_input.cpp:21`, `writer_cb2dram_for_tiled_input.cpp:34`, `writer_cb2dram_for_rm_input.cpp:26`). No explicit page-size third argument anywhere; the subject does not fire. Not in the 3rd-arg triage doc, consistent with the scan.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory — classification varies per factory):
  - `MultiCore` (sharded): **input** → clean (borrowed-DFB, `c_0` `cb.buffer = src_buffer`); **output** → clean (borrowed-DFB, `c_16` `cb.buffer = dst_buffer`). Port via `DataflowBufferSpec::borrowed_from`.
  - `MultiCoreDRAMFold` (tiled + row-major): **input** → **Case 1** (`Buffer*` base → `TensorAccessor` in the reader); **output** → **Case 1** (`Buffer*` base → `TensorAccessor` in the writer). Delivered today as `Buffer*` BufferBindings (correct on cache hit); the port expresses both as `TensorParameter` / `TensorBinding`, kernel builds `TensorAccessor(tensor::name)`, and the address-via-RTA + `TensorAccessorArgs` plumbing disappears.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none to drop.
- **CB endpoints:** self-loop `c_1` src1 scratch @ `fold_multi_core_dram_program_factory.cpp:318-331` (RM, `!is_l1_aligned` config) · 1P+1C assign `c_0` and `c_16` in the sharded factory (dual-instance work-split) · all other CBs legal 1:1.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (dual-instance work-split — the shape to handle, not a multi-binding):** the sharded `MultiCore` factory instantiates `writer_cb2s_row_major.cpp` twice over the same cores (Reader/Writer configs). Both instances raw-touch `c_0` and `c_16` — assign 1P+1C on each (bind one instance PRODUCER, the other CONSUMER); do **not** reach for the multi-binding flag. There is no hidden second writer and no semaphore-gated co-fill.
- **CB endpoints (`c_1` referenced in an unallocated config):** in the RM sub-variant, `writer_cb2dram_for_rm_input.cpp:33` calls `cb_in1.get_write_ptr()` **unconditionally**, but the factory allocates `c_1` only when `!is_l1_aligned` (`fold_multi_core_dram_program_factory.cpp:318-331`). In Metal 2.0 a kernel cannot touch a DFB it hasn't bound. The porter must either (a) allocate/bind `c_1` in both configs, or (b) guard the `get_write_ptr()` behind `if constexpr (!is_l1_aligned)` so the touch matches the allocation. Prefer (b) — the value is dead in the aligned config anyway (see Misc anomalies).
- **Cross-op / shared kernels:** the tiled sub-variant file-path-instantiates `untilize/device/kernels/compute/untilize.cpp` (owned by the untilize op) at `fold_multi_core_dram_program_factory.cpp:171`. A Metal 2.0 rewrite of that shared compute kernel is a single change shared with untilize (and any other borrower) — port it as one unit. Low risk: it is a thin wrapper over the DFB-aware `compute_kernel_lib::untilize` helper.
- **RTA varargs:** none — every RTA is read a fixed number of times at a distinct constexpr offset (readers: args 0/1/2; tiled writer: 0-4; RM writer: 0/1). Port as named args.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up:** ⚠ workable (all donor shapes cross cleanly; no ⭐ blockers). No donor is pre-Device-2.0.
  - **Summary table** (op kernel → donor):

    | Op kernel | Donor file | Donor class | Shape / status |
    |---|---|---|---|
    | all 5 dataflow kernels | `pool/device/kernels/experimental_device_api.hpp` | cross-family (pool) | Device 2.0 helper (`Noc`/`CB`/`UnicastEndpoint`) — ✓ excellent |
    | `reader_dram2cb_tiled`, `writer_cb2dram_for_tiled_input`, `writer_cb2dram_for_rm_input` | `data_movement/common/kernels/common.hpp` | in-family shared | `Noc`-based free helpers — ✓ (included; RM writer's `using namespace` pulls it in but calls `noc.async_write` directly) |
    | `untilize.cpp` (compute donor, file-path) | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | official shared kernel library | DFB-aware compute helper — ✓ (lib team owns) |
  - **Borrowed kernel files (file-path instantiation):** `untilize/device/kernels/compute/untilize.cpp` — owned by the untilize op family, broadly shared (untilize op and its callers). Instantiated by fold's tiled sub-variant. Forms a Metal 2.0 **port-together set** with untilize's compute kernel — the shared CB→DFB rewrite must land in one change. (Fold owns all five of its *dataflow* kernels; only the compute kernel is borrowed.)
- **Relaxation candidates:** none — no custom hash to mine.
- **TTNN factory analysis:** both factories `descriptor` concept, single-program, no op-owned tensors, no MeshWorkload, no pybind `create_descriptor`, no custom hash, no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept`. All gate conjuncts confirmed absent against the code (see Gate detail).

## Misc anomalies  *(team-only, non-gating)*

- **Dead `get_write_ptr` in the L1-aligned config.** `writer_cb2dram_for_rm_input.cpp:33` unconditionally computes `intermed_l1_scratch = cb_in1.get_write_ptr()` and casts it to `patch_data`, but `patch_data` is only written/read inside `if constexpr (!is_l1_aligned)` blocks (lines 40-47, 48-55). When `is_l1_aligned`, `c_1` is not allocated by the factory yet the pointer is still fetched (and unused). Harmless today (value never dereferenced when aligned); becomes a real binding mismatch under Metal 2.0 (see Heads-ups). The ops team may want to hoist the `get_write_ptr()` inside the `if constexpr (!is_l1_aligned)` guard. Route: ops team.
- **`compile_time_args[13]` role naming is inverted vs. config.** In the sharded factory (`fold_multi_core_program_factory.cpp:97,113`) the CTA comment labels index 13 `is_reader`; it is set `true` for the kernel attached with `WriterConfigDescriptor` and `false` for the one attached with `ReaderConfigDescriptor`. The kernel behaves correctly (the two instances just need to disagree on the column split), but the naming reads backwards and cost a moment to reconcile during the census. Cosmetic; route: ops team.

## Questions for the user  *(none)*

None — every gate resolved from the code and the readiness sheet without residual ambiguity.

## Recipe notes

- **Single factory row covering two runtime sub-variants.** `MultiCoreDRAMFold::create_descriptor` dispatches at runtime on input layout (TILE vs ROW_MAJOR) into two structurally distinct programs (`fold_multi_core_tiled_interleaved` / `fold_multi_core_row_major_interleaved`) with different kernels and CB topologies, but the readiness sheet carries a single `MultiCoreDRAMFold` row. The audit template's per-factory model mostly absorbs this (I reported per sub-variant inside the one factory), but the CB-endpoints "classify per instantiation" guidance and the sheet's one-row-per-factory rule sit at slightly different granularities here. Worked fine; noting in case the recipe wants to say explicitly how to attribute findings when one factory forks its program by a `tensor_args` property rather than by config/sharding.

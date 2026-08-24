# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/transformer/sdpa_decode`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `c9ef66ee339 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources` *(carry into the port report's Provenance section)*

## TTNN factory analysis

One `DeviceOperation` (`ttnn::prim::SdpaDecodeDeviceOperation`), one `create_descriptor` in `device/sdpa_decode_program_factory.cpp` (paged / MLA / sharded are internal branches, not separate factories). The four nanobind entry points all route here.

- **Current concept:** `descriptor` (`create_descriptor` returns `tt::tt_metal::ProgramDescriptor`).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (`Override runtime args method? = no`; the framework refreshes tensor bindings on cache hit and the factory writes one method).
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none` `TensorParameter relaxation`; `get_dynamic_runtime_args`. Present but **not** gating (leave/translate per recipe): none — no custom hash (removed; `Formerly custom hashed?=yes`), no `override_runtime_arguments`, no pybound `create_descriptor`.
- **Delivery today:** every buffer is bound as a `Buffer*` (`BufferBinding`, the "pointer-patching" form) via `emplace_runtime_args`, not `->address()`. Replace with typed `TensorParameter` bindings (below).

## Construct — to do

**Tensor bindings** (per binding; this op is one factory with paged / MLA / sharded branches, so several bindings differ by config — bind each per its config path):

- **q** — **Case 1** (DRAM interleaved) → `TensorParameter`; kernel builds `TensorAccessor(tensor::q)` (currently `dataflow_common.hpp:572`). **Case 2** (HEIGHT_SHARDED, non-MLA) → bind `tensor::q`, pull base via `get_bank_base_address`, keep the raw core-to-core L1 read unchanged (`read_q` `dataflow_common.hpp:544-559`; `q_addr` is a clean base — do not rewrite the raw walk). **clean** (MLA `q_locally_available`) → `c_0` is borrowed → `DataflowBufferSpec::borrowed_from(tensor::q)`.
- **k** — **Case 1** → `TensorAccessor(tensor::k)` (`reader_decode_all.cpp:222`).
- **v** — **Case 1** → `TensorAccessor(tensor::v)` (`reader_decode_all.cpp:224`; used only when `!reuse_k`).
- **cur_pos_tensor** — **Case 1** (DRAM: `reader_decode_all.cpp:129`) · **clean** borrowed `c_8` (sharded).
- **page_table_tensor** — **Case 1** (DRAM, via `read_page_table_for_batch`) · **clean** borrowed `c_9` (sharded).
- **attn_mask** — **Case 1** → `TensorAccessor(tensor::attn_mask)` (`reader_decode_all.cpp:226`).
- **attention_sink** — **Case 1** → `TensorAccessor(tensor::attention_sink)` (`reader_decode_all.cpp:230`).
- **output** — **Case 1** (DRAM: `writer_decode_all.cpp:246`) · **clean** borrowed `c_20` (sharded).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** drop the redundant page-size arg at **both** sites (both Class 2, **neither** sets `dynamic_tensor_shape`):
- Q: `device/kernels/dataflow/dataflow_common.hpp:572` (`q_page_size_bytes` = full-tile size).
- page-table: shared sdpa donor `../sdpa/device/kernels/dataflow/dataflow_common.hpp:83` inside `read_page_table_for_batch` (`page_table_stick_size` = `aligned_page_size()`). This edit lands in a kernel shared with sdpa prefill — coordinate per the shared-kernel note below.
- Confirm-don't-swap-blind for Q: Metal 2.0's implicit `aligned_page_size` comes from Q's **DRAM buffer** spec (full tile), which equals the dropped override even when the Q CB `c_0` is a half-tile under `use_half_tile`.

**CB endpoints** (per `(CB, config)`; all three kernels span the full `core_grid`):
- **self-loop** (single toucher): `c_9` (page_table, reader) · `c_19` (intermed_out, writer raw + cross-core; only when `num_cores_per_head>1`) · compute intermediates `c_21 c_22 c_23 c_24 c_25 c_26 c_27 c_28 c_29 c_30 c_31`.
- **set the multi-binding advanced option:** **`c_16`** (`cb_out_o`/`cb_out_worker`) — the tree reduction reuses this one index for *both* directions: writer produces + compute consumes (receive child O, `writer:325-334` / `compute:571`) **and** compute produces + writer consumes (send own O to parent, `compute:666` / `writer:347-389`). On intermediate tree nodes (present when `num_cores_per_head ≥ 4`) that is 2 locked producers + 2 locked consumers; `move_block` is FIFO so the roles don't relabel. 1P+1C on root/leaf. Bind for the multi-binding case.
- **borrowed-DFB** (`DataflowBufferSpec::borrowed_from`) in sharded configs: `c_0` (Q, MLA local), `c_8` (cur_pos sharded), `c_9` (page_table sharded), `c_20` (output sharded). Host sets `cb.buffer = buffer` today (`sdpa_decode_program_factory.cpp:533-535`).
- **config-flip:** `c_0` (q_in) — self-loop (tilize_q: compute tilizes `c_10`→`c_0` then consumes) / 1P+1C (DRAM non-tilize: reader→compute) / borrowed (MLA local).
- **1P+1C** (bind one PRODUCER, one CONSUMER): `c_1 c_2 c_3 c_4 c_5 c_6 c_7 c_8 c_10 c_12 c_13 c_14 c_15 c_17 c_18 c_20`.
- **`c_11` (col_identity) — do NOT drop; confirm first.** Produced by the writer (`generate_bcast_col_scalar`, `writer:215`) but no consumer in sdpa_decode (see `METAL2_PREPORT_AUDIT.md` → Misc anomalies). It has one toucher, so bind it as a **self-loop**; its removal is an ops-team question, not a port drop.

## Watch for

- **CB endpoints — multi-binding `c_16`:** the trap is the *bidirectional reuse* (receive-child vs. send-parent under one index), not a raw co-fill. Binding it 1P+1C will under-bind the intermediate-node config. There is **no** hidden raw second-writer: K-multicast (`read_k`, `dataflow_common.hpp:667`) and the tree-reduction cross-core writes (`writer:363-383`) target each node's *own* CB instance at a same-offset L1 address, so per node they stay 1P+1C / self-loop.
- **Cross-op / shared kernels** — **sunset / coordination list, NOT authorization to convert a shared kernel in place.** The op instantiates only its own three kernels (no borrowed *kernel files*), but `#include`s these shared headers:
  - in-family, shared with **sdpa (prefill)**: `../sdpa/device/kernels/compute/compute_common.hpp`, `../sdpa/device/kernels/dataflow/dataflow_common.hpp`. In-family escapes bridge via boundary features; don't rewrite the donor headers. Note the page-table 3rd-arg drop above lands in the shared `read_page_table_for_batch` — verify sdpa prefill's callers tolerate it (it's a pure no-op drop).
  - `ttnn/kernel/dataflow/generate_bcast_scalar.hpp` — `generate_bcast_col_scalar` takes a `CircularBuffer`; a **`_metal2` fork already exists** (`generate_bcast_scalar_metal2.hpp`, takes `DataflowBuffer`) — **bind the fork, don't re-fork** (and see the `c_11` note — this call may be removable).
  - `ttnn/cpp/ttnn/kernel_lib/{tilize_helpers,untilize_helpers,reduce_helpers_dataflow,l1_helpers}.hpp` — lib-team-owned; CB-index / template / `DataflowBuffer` shapes, all bridge cleanly.
- **RTA varargs** — use the vararg mechanism (don't try to name each) for the variable-count physical-core-coordinate arrays read via `get_arg_addr`: reader `all_output_noc_x/y` (`reader_decode_all.cpp:184-186`); writer `reduction_group_core_xs/ys` (`108-111`), `all_reducer_noc_x/y` (`188-191`), `all_output_noc_x/y` (`192-194`). The `children_per_round[6]` block is fixed-count → name it, not a vararg.
- **Reducer semaphore raw poll** — the writer resolves `get_semaphore(reducer_semaphore_id)` and polls the value with a custom 4-bit-per-round nibble decode (`writer_decode_all.cpp:32,248-249,275-278`) because `Semaphore<>::wait(threshold)` can't express per-round counters. Bind this semaphore via `sem::name` and derive the poll address from the named binding — don't only swap the `Semaphore<>` constructions and leave a bare `get_semaphore(id)`.
- **`fp32_dest_acc_en` / opt-level / `use_half_tile` tile geometry** — the compute kernel carries load-bearing `reconfig_*` and half-tile (16×32) vs full-tile handling and one-time SrcA/SrcB geometry reprograms (`sdpa_flash_decode.cpp:224-238`). These are compute-microcode, untouched by the binding-layer port; don't perturb them.

# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/`

Single `DeviceOperation` with a single unified `ProgramFactory`:

- **`ttnn::prim::SdpaDecodeDeviceOperation`**
  - `create_descriptor` (`device/sdpa_decode_program_factory.cpp`, ~1027 LOC) — the one program factory. It internally fans out across many configs (paged / unpaged, MLA, Q-sharded / DRAM, output-sharded, sliding-window, attention-sink, GQA, K-multicast, subcoregrid, block-padding) via runtime branches, not separate factories.

Kernels referenced by the factory (all by `FILE_PATH`):
- `device/kernels/dataflow/reader_decode_all.cpp`
- `device/kernels/dataflow/writer_decode_all.cpp`
- `device/kernels/compute/sdpa_flash_decode.cpp`
- Shared headers: `device/kernels/rt_args_common.hpp`, `device/kernels/dataflow/dataflow_common.hpp` (which `#include`s the sibling `sdpa/device/kernels/dataflow/dataflow_common.hpp`), and the sibling `sdpa/device/kernels/compute/compute_common.hpp`; plus `ttnn/kernel_lib/*` and `ttnn/kernel/dataflow/generate_bcast_scalar.hpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

**Recipe docs:** `776151aeca8 2026-06-24 docs(metal2): clarify SPSC face-(b) producer-as-consumer, aliased-vs-same-FIFO, no-portable-subset`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/` |
| **Overall** | **YELLOW** — open questions on the tree-reduction CB SPSC ceiling and a Device-2.0 raw-semaphore-poll holdover |
| **DOps / Factories** | `SdpaDecodeDeviceOperation` → single unified `create_descriptor` factory |
| *Prereqs* — ProgramDescriptor | **Yes** — fully on `ProgramDescriptor` / `KernelDescriptor` / `CBDescriptor` / `SemaphoreDescriptor` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes-with-holdover (YELLOW)** — one raw-semaphore-poll site in the writer (see Gate detail); everything else is `Noc` / `CircularBuffer` / `Semaphore<>` / `TensorAccessor` / `CoreLocalMem` / `UnicastEndpoint` |
| *Prereqs* — Cross-op escapes | **Ok** — sibling `sdpa` headers + `kernel_lib` donors are all Device-2.0-clean (no addr-gen, no raw NoC/semaphore) |
| *Feature Support* — overall | **GREEN** (one open question, see below) |
| *Feature Support* — Variadic-CTA | **Ok** — all CTAs are literal-indexed; `tensor_args_t` has no variable-count container |
| *TTNN Readiness* — Op-owned tensors | **No** — only `create_device_tensor` for the declared output |
| *TTNN Readiness* — MeshWorkload needed | **No** — single-program; no `create_mesh_workload` / `cached_mesh_workload` |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — nanobind binds only the four user-facing functions; no factory/device-op class binding |
| *TTNN Readiness* — Other risky pybind | **None** |
| *TTNN Readiness* — Custom hash | **Yes → delete** (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | **No** — factory provides no `override_runtime_arguments` |
| *Ops readiness* — Sync-free CBs (address-only) | **Present** — `c_19` (`cb_intermed_out`) tree-reduction scratch (raw-pointer only, writer-only); `c_9` (`cb_id_page_table`) in the sharded path (self-loop workaround) |

**Sync-free CBs** = CBs used purely as an address source (the kernel grabs the base pointer and walks the memory, no FIFO ops).

## Result

**YELLOW → open questions.** No hard GATE fired: the op is on `ProgramDescriptor`, uses no UNSUPPORTED feature, owns no tensors, needs no MeshWorkload, and pybinds nothing risky. Two items need a decision before a brief can issue:

1. **Tree-reduction CB `c_16` (`cb_out_o`/`cb_out_worker`) SPSC ceiling.** On an *intermediate* tree node (a core that both receives from children and sends to a parent — present whenever `num_cores_per_head > 2`), `c_16` is bound by **both** the writer and the compute kernel as **both** producer and consumer. That is a per-node multi-endpoint shape that Metal 2.0's SPSC validator will reject. Config-scoped: configs with `num_cores_per_head <= 2` (root + leaves only, no intermediate nodes) keep `c_16` at a legal `(1 producer, 1 consumer)` per role. See [DFB endpoint legality](#dfb-endpoint-legality-spsc) and Question 1.
2. **Device 2.0 raw-semaphore-poll holdover** in the writer (`reducer_semaphore`) — see Gate detail and Question 2. It is an isolated holdover *with no clean `Semaphore<>` member-form replacement today* (the wrapper exposes no raw-value read for the nibble-encoded counter), so the yellow tier here is a genuine judgment call, not a mechanical 1-liner.

Neither is a permanent blocker. (1) resolves by an op-owner pre-port functional change (or by confirming the protocol is SPSC-safe); (2) resolves either by a small `Semaphore<>` API addition or by confirming the raw poll is sanctioned.

## Gate detail

- **ProgramDescriptor:** **GREEN.** `create_descriptor` populates a `ProgramDescriptor` with `CBDescriptor` (`add_cb` helper), `SemaphoreDescriptor` (×3), and three `KernelDescriptor`s; runtime args via `emplace_runtime_args`. No `CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs`.

- **Device 2.0 (every kernel used):** **YELLOW — one isolated holdover.** All three kernels and all shared/donor headers consistently use Device-2.0 idioms: `Noc noc`, `CircularBuffer` wrappers (member `get_read_ptr()`/`get_write_ptr()`/`reserve_back`/`push_back`/`wait_front`/`pop_front`), `Semaphore<>`, `UnicastEndpoint` / `MulticastEndpoint`, `CoreLocalMem<>`, and `TensorAccessor`. No `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / raw `noc_async_read` / raw `noc_semaphore_*`. `get_tile_size(cb_id)` (reader `dataflow/reader_decode_all.cpp:197-201`, plus `dataflow_common.hpp`) is the **sanctioned** free function — not a holdover. The one holdover:

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/dataflow/writer_decode_all.cpp` | 32 | `uint32_t reducer_semaphore_addr = get_semaphore(reducer_semaphore_id);` | `Semaphore<>` is used elsewhere in the same kernel (`.up()` L387/L482, `.wait()` L426, `.set()`) |
  | `device/kernels/dataflow/writer_decode_all.cpp` | 249-250, 274-281 | `reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr)` then a hand-rolled spin: `uint32_t sem_val = *in0_receiver_semaphore_addr_ptr; uint8_t step_sem = (sem_val >> step_semaphore_shift[round]) & 0x0F;` | — |

  This is the legacy `get_semaphore(sem_id)` + `reinterpret_cast` + raw poll pattern the [Device 2.0 migration guide](../../kernel_apis/data_movement/device_api_migration_guide.md) (Semaphore Operations section) flags as the form `Semaphore<>` replaces. **But** the writer is not doing a simple threshold wait — it reads a **nibble-encoded** per-round counter out of the raw 32-bit semaphore word (`(sem_val >> shift) & 0x0F`). `Semaphore<>` (`tt_metal/hw/inc/api/dataflow/noc_semaphore.h`) exposes only `wait(value)` / `wait_min(value)` and **no raw-value getter**, so there is *no* member-form replacement for this access today. This sits at the holdover/sanctioned boundary the recipe warns about. Routed to the Device 2.0 effort to decide; see Question 2. (Per the recipe, even an isolated holdover blocks the *start* of the port and is fixed on the Device 2.0 track, never folded into the port diff — so a brief, if issued, must flag it as a blocker.)

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer`, `remote_cb`, `remote_index`, `global_circular_buffer` field |
  | Dynamic CircularBuffer (borrowed memory) | **GREEN** | `add_cb(..., buffer)` sets `CBDescriptor.buffer` for `c_0` (Q, when `q_locally_available`), `c_8` (cur_pos, when sharded), `c_9` (page_table, when sharded), `c_20` (output, when sharded). Port uses `borrowed_from`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `add_cb` never sets `address_offset` |
  | Aliased Circular Buffers | N/A | `add_cb` always emits single-element `format_descriptors = {format}` (factory L529); no multi-format CB |
  | GlobalSemaphore | N/A | only `SemaphoreDescriptor` (×3, all `initial_value = 0`); no `GlobalSemaphore` |
  | Non-zero semaphore initial value | N/A | all three semaphores `initial_value = 0` (factory L632/L634/L636) |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | static `TensorAccessorArgs` only; reader overrides only the *page size* (`q_page_size_bytes`), not the accessor config |
  | `UpdateCircularBuffer*` | N/A | no `Update*CircularBuffer*` calls |
  | Variable-count compile-time arguments (CTA varargs) | N/A | all `get_compile_time_arg_val(...)` use literal indices; `tensor_args_t` carries fixed `Tensor` / `optional<Tensor>` members (no `std::vector<Tensor>`) |

  Subject verdict: **GREEN — no gate fired.** (One open question, see below: the reader does pull `output_core_physical_xs/ys` arrays out of L1 via `get_arg_addr` in a `num_output_cores`-length walk — this is *RTA* varargs, not CTA varargs; see Other signals. The writer likewise walks `reduction_group_core_xs/ys`, `all_reducer_noc_*`, `all_output_noc_*`. None are compile-time.)

- **DFB endpoint legality (SPSC):** **YELLOW (config-scoped open question).** Most CBs sit cleanly in the legal window; the tree-reduction handoff CBs are where the count matters. Per-`(CB, role)` census across the three kernels:

  **Clean (1 producer, 1 consumer):**
  - `c_0` cb_q_in — reader produces / compute consumes. (When `q_locally_available`: borrowed-memory DFB, reader `reserve_back`+`push_back` of resident data, compute consumes → synchronized borrowed DFB, clean.)
  - `c_1`/`c_2` cb_k_in/cb_v_in — reader produces / compute consumes.
  - `c_3` cb_mask_in — writer produces (`generate_mask`) / compute consumes. (Reader also produces in the non-causal mask path; causal vs. non-causal are mutually exclusive configs — classify per config, each is (1,1).)
  - `c_4` cb_attention_sink — reader produces / compute consumes.
  - `c_5`/`c_11`/`c_12` cb_identity_scale_in/cb_col_identity/cb_zero_in — writer produces / compute consumes.
  - `c_6`/`c_7` cb_m_in/cb_l_in — writer produces (NoC-reads child M/L, `writer:312`/`299`) / compute consumes (`compute:536`/`571`).
  - `c_8` cb_writer_cur_pos — reader produces / writer consumes. (Borrowed-memory DFB when cur_pos sharded.)
  - `c_10` cb_q_rm — reader produces / compute consumes (tilize path).
  - `c_13` cb_sliding_window_mask_in, `c_14` cb_block_pad_mask — writer produces / compute consumes.
  - `c_15` cb_compute_cur_pos — reader produces / compute consumes. (Note: cur_pos is **split** into `c_8` for writer + `c_15` for compute exactly to avoid a one-CB two-consumer race, per factory comment #44366 — this split is already SPSC-friendly.)
  - `c_17`/`c_18` cb_out_m/cb_out_l — compute produces (`compute:654`/`656`) / writer consumes (`writer:345-346,349-350`).
  - `c_20` cb_out / cb_out_final — compute produces (`move_block`/`untilize`, `compute:641`) / writer (root) consumes (`writer:405-494`). (Borrowed-memory DFB when output sharded; in the sharded-GQA path the root writer also raw-reads peer reducers' `c_20` via NoC into its own `c_20` — same core, raw read+write, see note below.)
  - `c_21`–`c_31` (cb_prev_sum_2, cb_exp_max_diff_2, cb_out_accumulate_im_2, cb_qk_im, cb_out_im, cb_out_accumulate_im, cb_max_1/2, cb_sum_1/2, cb_exp_max_diff) — compute-only intermediates (single kernel, FIFO-self or ping-pong); legal.

  **⛔ Open SPSC question — `c_16` (`cb_out_o` = `cb_out_worker`):** the *same* CB index is used for two opposite handoffs:
  - As **`cb_out_worker`**: compute produces its local O (`compute:652 move_block → cb_out_o`), writer consumes it to send to the parent (`writer:344 cb_out_w(cb_out_worker).wait_front/get_read_ptr/pop_front`).
  - As **`cb_out_o`**: writer produces a *child's* received O (`writer:325 cb_o(cb_out_o).reserve_back/push_back`), compute consumes it to fold into the accumulator (`compute:557 move_block(cb_out_o → cb_out_accumulate_im_2)`).

  On a **leaf** node (no children): only the `cb_out_worker` handoff fires → (1 producer = compute, 1 consumer = writer). Legal. On the **root** (no parent): only the `cb_out_o` handoff fires → (1 producer = writer, 1 consumer = compute). Legal. On an **intermediate** node (has children *and* a parent — exists when `num_cores_per_head > 2`, i.e. tree depth ≥ 2): *both* handoffs fire on the one node, so `c_16` is bound by **two producers** (writer L325 + compute L652) and **two consumers** (writer L344 + compute L557). That is the per-node SPSC ceiling violation. The two roles are temporally disjoint within a head iteration, but SPSC is a *binding-count* rule, not a temporal one — both kernels bind `c_16` as both producer and consumer, which the spec validator rejects regardless of ordering. **Config-scoped:** `RED at the intermediate-tree-node config (num_cores_per_head > 2); the num_cores_per_head ≤ 2 subset is clear.** Resolution, if confirmed a violation, is an **op-owner pre-port functional change** — e.g. split the receive-from-child and send-to-parent buffers into two distinct CB indices (mirroring the cur_pos #44366 split), so each index has one producer and one consumer per node. See Question 1; do not assume — confirm with the user before treating as a hard RED, since the receive/send phases never overlap in time and the protocol may have been validated as safe.

  **Sync-free CB (self-loop workaround, FYI-P):**
  - `c_19` cb_intermed_out — touched **only by the writer**, entirely by raw pointer: writer-as-sender writes child→parent intermediates via `cb_intermed.get_write_ptr()` (`writer:360`), writer-as-receiver reads them back via `cb_intermed.get_read_ptr()` (`writer:294`). No `reserve_back`/`push_back`/`wait_front`/`pop_front` anywhere — coordinated by the `reducer_semaphore`, not FIFO sync. One kernel, both ends, pointer-only → **sync-free**, resolved by the self-loop DFB workaround. (Allocated only when `intermed_output_tiles > 0`, i.e. `num_cores_per_head > 1`; absent for single-core-per-head configs.)
  - `c_9` cb_id_page_table (sharded path) — reader `reserve_back`/`push_back` then reads by raw pointer; no other kernel touches it. Single-ended (producer only, no FIFO consumer) → self-loop workaround. (Borrowed-memory when sharded.)

  **No dead CBs.** Every allocated index is referenced by at least one kernel.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — all reach the kernel as a raw `uint32_t` base address pushed onto an RTA in the factory (`reader_rt_args.push_back(q_buffer->address())` etc., factory L909-915, L936; deliberately *not* `Buffer*` per the #paged-attention comment at L904-907), then fed into `TensorAccessor(args, addr)` on the device side → all **Case 1** (via `TensorAccessor`):
  - `q` — Case 1 in the DRAM path (`read_q` → `TensorAccessor(q_args, q_addr, q_page_size_bytes)`); **clean (borrowed-memory DFB)** in the `q_locally_available` config (data resident in `c_0`, no accessor read). Per-config split.
  - `k`, `v` — Case 1 (`TensorAccessor(k_args, k_addr)`, `TensorAccessor(v_args, v_addr)`).
  - `attn_mask` — Case 1 (`TensorAccessor(mask_args, mask_addr)`).
  - `cur_pos_tensor` — Case 1 in the DRAM path (`TensorAccessor(pos_args, pos_addr)`); **clean (borrowed-memory DFB)** when sharded (read from `c_8`).
  - `page_table_tensor` — Case 1 in the DRAM path (`read_page_table_for_batch(... page_table_args, page_table_addr ...)`); **clean (borrowed-memory DFB)** when sharded (read from `c_9` by pointer).
  - `attention_sink` — Case 1 (`TensorAccessor(attention_sink_args, attention_sink_addr)`).
  - `output` — Case 1 in the writer (`TensorAccessor(out_args, out_addr)`, `write_tiles_to_memory` / `write_partial_tiles_to_memory`); **clean (borrowed-memory DFB)** when output sharded (`c_20`).

  Op-level roll-up: **⚠ port work** (all bindings Case 1; none are Case 2 raw-pointer, so the compute-kernel `TensorBinding` block does **not** apply). Mechanical, low-risk. The factory currently smuggles every base address through an RTA `uint32_t` — the port replaces each with a typed `TensorParameter`/`TensorBinding`, eliminating the address-via-RTA plumbing and the `TensorAccessorArgs(...).append_to(...)` CTA plumbing (factory L684-696, L728).

- **Custom hash:** **delete** custom `compute_program_hash` (`device/sdpa_decode_device_operation.cpp:500-555`) → default (sanctioned exception). See Custom program hash subject.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:** borrowed-memory DFB — `c_0` Q (`q_locally_available`), `c_8` cur_pos (sharded), `c_9` page_table (sharded), `c_20` output (sharded); factory `add_cb(... buffer)` sites L538/L557-564/L566-575/L610-616. Port uses `DataflowBufferSpec::borrowed_from`. No aliased CBs, no dynamic-TA opt-ins, no non-zero sem init.
- **Sync-free CBs (address-only):** `c_19` cb_intermed_out (writer-only, raw `get_write_ptr`@`writer:360` / `get_read_ptr`@`writer:294`); `c_9` cb_id_page_table (reader, single-ended). Port resolves both with the self-loop workaround; non-gating.
- **Dead CBs:** none.
- **Cross-op / shared kernels:** the decode kernels `#include` and call into the **sibling `sdpa`** family (`compute_common.hpp`, `dataflow_common.hpp`) and the shared `kernel_lib` / `kernel` pools — see Team-only. These form a **port-together set** (sdpa + sdpa_decode share `dataflow_common.hpp`/`compute_common.hpp`). All donors are Device-2.0-clean, so they do not gate, but the Metal 2.0 rewrite of the shared headers must land in lockstep across both ops.
- **RTA varargs:** the reader walks `all_output_noc_x/y` (`num_output_cores`-length) via `get_arg_addr` (`reader:184-189`); the writer walks `children_per_round` (fixed 6), `reduction_group_core_xs/ys` (`num_cores_per_head`), `all_reducer_noc_x/y` (`num_reducer_cores`), `all_output_noc_x/y` (`num_output_cores`) (`writer:102-194`); the compute reads `children_per_round` (fixed 6). These are RTA arrays whose lengths are program-fixed per invocation — prefer named RTAs; not gating.
- **TTNN factory analysis (porter-relevant):** no pybind `create_descriptor`, no other risky pybind, no custom `override_runtime_arguments`. Only the custom hash (→ delete).

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean / port-together**. No ⚠/✗/⭐ donor-signature shapes (no donor consumes `InterleavedAddrGen`/`ShardedAddrGen`/raw-sem/`CircularBuffer&` in a gating way). Summary by donor:
  - `sdpa/device/kernels/dataflow/dataflow_common.hpp` — **in-family** (sibling `sdpa`). Functions called: `get_barrier_read_threshold`, `virtual_seq_tile_id_to_physical_tile_id`, `read_page_table_for_batch`, `copy_tile`. Signatures take `Noc`, `TensorAccessor`-typed readers (Shape 1 ✓), CB-index `uint32_t` (✓), and `tt_l1_ptr` pointers — Device-2.0-clean. Port-together with sdpa.
  - `sdpa/device/kernels/compute/compute_common.hpp` — **in-family** (sibling `sdpa`). Compute helpers (`reduce_c`, `sub_exp_block*`, `correction_block`, `add_block_inplace`, `mul_block_bcast*`, `matmul_blocks`, `recip_block_inplace`, `max_block`, `move_block`, `read_tile_value`) take CB indices — compute-side, clean. Port-together with sdpa.
  - `ttnn/kernel_lib/l1_helpers.hpp`, `ttnn/kernel_lib/reduce_helpers_dataflow.hpp`, `ttnn/kernel_lib/tilize_helpers.hpp`, `ttnn/kernel_lib/untilize_helpers.hpp`, `ttnn/kernel/dataflow/generate_bcast_scalar.hpp` — **shared kernel library** (lib team owns); Device-2.0-clean (no addr-gen / raw NoC / raw sem found).
  - **File-path kernel instantiation:** the op owns all three of its instantiated kernel files (`reader_decode_all.cpp`, `writer_decode_all.cpp`, `compute/sdpa_flash_decode.cpp`); it does not file-path-instantiate any kernel `.cpp` it doesn't own. (The *headers* it `#include`s are shared, hence the port-together set above.)
- **Relaxation candidates** (mined from the custom hash before deletion — **FALLIBLE, default strict**): the custom hash separates `logical_shape` + `padded_shape` for Q/K/V into an explicit `qkv_logical_padded_shape_key` and hashes `cur_pos_tensor` `logical_shape` separately, suggesting the op is sensitive to both logical and padded shapes of Q/K/V (so `match_padded_shape_only` is likely *unsafe* here — padded shape genuinely drives CB sizing and CTAs). The `share_cache` 3-state tag (unset/false/true) and the explicit inclusion of `block_size_override` / `num_kv_heads_override` / `cache_position_modulo` confirm these feed compile-time args — they must remain in the key. No obvious safe relaxation candidate; treat the default strict hash as correct.
- **TTNN factory analysis (six questions):**
  1. **Op-owned tensors? No.** `create_output_tensors` calls `create_device_tensor(compute_output_specs(...), q.device())` for the *declared output* only (`device/sdpa_decode_device_operation.cpp:495-498`). No intermediate/scratch device tensors.
  2. **MeshWorkload needed? No.** No `create_mesh_workload` / `create_workload_descriptor` / `cached_mesh_workload_t`. Single-program.
  3. **Pybind `create_descriptor`? No.** `sdpa_decode_nanobind.cpp` binds only the four user-facing functions (`scaled_dot_product_attention_decode`, `paged_scaled_dot_product_attention_decode`, `flash_multi_latent_attention_decode`, `paged_flash_multi_latent_attention_decode`) via `ttnn::bind_function<...>`. No `nb::class_<...ProgramFactory>` / `def_static("create_descriptor", ...)`.
  4. **Other migration-risky pybind? None.** No device-op method or factory/param class exposed to Python.
  5. **Custom hash? Yes** — `compute_program_hash` at `device/sdpa_decode_device_operation.cpp:500-555`. Treatment: delete (see Custom program hash).
  6. **Custom override-runtime-args? No.** The factory defines no `override_runtime_arguments`.

## Custom program hash

`SdpaDecodeDeviceOperation::compute_program_hash` (`device/sdpa_decode_device_operation.cpp:500-555`) overrides the default reflection hash. **PORT WORK — delete it**, reverting to the default TTNN hash (sanctioned device-op-class edit). The custom hash hand-builds sub-keys for Q/K/V logical+padded shapes and the optional `cur_pos_tensor` shape, and encodes `share_cache` as a 3-state tag; it *does* fold the tensors and their shapes into the key, but the default hash (which keys on each tensor's full `TensorSpec`) is correct-by-construction and removes the risk that a hand-rolled key silently omits a `TensorSpec` field. Do **not** patch it; delete it.

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **`reuse_k` CTA** is computed as `reuse_k = (tensor_args.v.has_value() ? 0 : 1)` (factory L642) — i.e. V is reused-from-K only in the MLA/no-explicit-V case. Behaves as documented; noted only because the name reads inverted at first glance.
- The reader declares `is_worker`/`is_output_core` from RTAs (`reader:95-96`) but several downstream branches key off `cur_pos`/`do_k_mcast` instead; confirm no RTA slot is dead. Not gating.

## Questions for the user

1. **Tree-reduction CB `c_16` SPSC ceiling (the central question):** On an intermediate tree-reduction node (`num_cores_per_head > 2`), CB index `c_16` is bound by both the writer and the compute kernel as both producer and consumer — the receive-from-child handoff (`cb_out_o`: writer:325 produces, compute:557 consumes) and the send-to-parent handoff (`cb_out_worker`: compute:652 produces, writer:344 consumes) both fire on the one node. The two handoffs are temporally disjoint within a head iteration. **Is this protocol considered SPSC-safe (and should it be passed through), or does it need an op-owner pre-port split into two distinct CB indices (one per direction), mirroring the cur_pos #44366 split?** If it must be split, the `num_cores_per_head ≤ 2` configs are a clean portable subset.
2. **Writer raw-semaphore poll (`reducer_semaphore`):** `writer_decode_all.cpp:32,249-281` reads a nibble-encoded per-round counter out of the raw semaphore word (`(sem_val >> shift) & 0x0F`) via `get_semaphore(...)` + `reinterpret_cast` + a hand-rolled spin loop. `Semaphore<>` offers `wait`/`wait_min` but **no raw-value getter**, so there is no member-form replacement for a *partial-bitfield* read today. **Is this a Device-2.0 holdover to be cleaned on the D2.0 track (e.g. by adding a raw-read accessor to `Semaphore<>`), or is the raw poll sanctioned given the bitfield encoding?** This gates the *start* of the port (per the Device 2.0 holdover rule) but is not a Metal 2.0 feature gap.

## Recipe notes

- The Device-2.0 holdover rule (Prerequisites Check 2 YELLOW) is framed around the **CB-index-keyed free-function family** (`get_read_ptr(cb_id)` → `cb_obj.get_read_ptr()`). The holdover here is a *different* family: a `get_semaphore(sem_id)` + raw-pointer poll where the `Semaphore<>` wrapper exists *but lacks the needed accessor* (a partial-bitfield read). The recipe's holdover/sanctioned litmus ("member-form replacement exists") returns *no* here, which by the letter would make it **sanctioned** (not a holdover) — yet the migration guide explicitly lists this exact `get_semaphore` + `reinterpret_cast` + poll shape as the legacy form `Semaphore<>` replaces. The two guidances pull in opposite directions for a *non-threshold* semaphore read. A sentence on semaphore-read holdovers (parallel to the CB-index family) would resolve this; I have surfaced it as Question 2 rather than silently picking a tier.
- SPSC face (a) "hidden second writer" is described as a raw co-fill coordinated by *dedicated* semaphores alongside a FIFO producer. The `c_16` case here is a *related but distinct* shape: not a hidden raw co-fill, but the *same CB index reused for two opposite FIFO handoffs* (`cb_out_o` vs `cb_out_worker`) that collide only on intermediate tree nodes. It is fully FIFO-visible (no raw co-fill), so the floor check passes (1P+1C exist), yet the per-node *count* exceeds the ceiling. The recipe's SPSC examples (conv2d/pool/halo) are all sharded-reader shapes; a tree-reduction "same-index two-direction handoff" example would help future auditors of reduction-style ops.

---

## ⚠️ Post-port-attempt correction (2026-06-25) — sync-free CBs are framework-blocked (self-loop is compute-only)

This audit graded the op YELLOW and noted sync-free CBs **`c_19`** (`cb_intermed_out` tree-reduction scratch, **writer-only**) and **`c_9`** (`cb_id_page_table`, "self-loop workaround"). The self-loop prescription does **not** apply: the self-loop workaround is **compute-kernel-only**; `c_19` is owned by the (DM) writer, so it cannot self-loop and cannot pair (no consumer), and Metal 2.0 has no scratch/sync-free DFB. This is the same framework wall that blocks sampling/embedding/create_qkv_decode.

**Corrected status: framework-blocked** on the DM-kernel sync-free CB (in addition to the tree-reduction CB `c_16` SPSC ceiling for `num_cores_per_head > 2`, which remains an op-owner item). Wait-for-feature: sync-free/scratch DFB or DM-kernel self-loop. See [[metal2-port-portability-predictor]].

---

## 🔄 Revision (2026-06-25, supersedes the correction above) — workaround found; NOT framework-blocked

The "framework-blocked / wait-for-feature" verdict above is **overstated**. A workaround exists with **no framework change**: the **cross-kernel DFB bridge**. Only a DM-kernel *self*-loop FATALs; a DM kernel paired *cross-kernel* with a different co-located kernel (DM↔DM or DM↔compute) is fully legal. **Proven in shipped code:** the landed JointSDPA port (PR #48175, 160 passed/0 failed) binds `mask`/`scale`/`col_identity` as PRODUCER on the **writer (DM)** → CONSUMER on **compute** (`joint_sdpa_program_factory.cpp:359-451`); the SPSC validator accepts and runs them.

**This op:** the writer-only sync-free `c_19` (tree-reduction scratch) → bridge to a co-located compute CONSUMER (cross-kernel bridge; high-confidence). **The sync-free CB is NOT a framework blocker.** The *remaining* real constraint is the separate, config-scoped **tree-reduction CB `c_16` SPSC ceiling** for `num_cores_per_head > 2` (an op-owner functional item), plus the raw-semaphore-poll holdover — neither is the sync-free-CB issue.

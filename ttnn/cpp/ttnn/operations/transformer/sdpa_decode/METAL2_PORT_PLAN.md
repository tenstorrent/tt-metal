# Port Plan — sdpa_decode

Port plan for `ttnn/cpp/ttnn/operations/transformer/sdpa_decode`, ported from the
`ProgramDescriptor` direct-descriptor API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Audit: GREEN (`METAL2_PREPORT_AUDIT.md`). Target arch: Wormhole (Gen1). One factory, ported whole.

## Legacy Inventory

### Legacy factory shape
- Concept: **direct-descriptor** — `SdpaDecodeDeviceOperation::create_descriptor(...)` returns
  `tt::tt_metal::ProgramDescriptor`, declared as a static member on the device-op struct with
  **no `program_factory_t`**. → forces [ttnn_factory exception 3](../shared/ttnn_factory.md): wrap
  the factory in a nested `SdpaDecodeProgramFactory` struct + `using program_factory_t = std::variant<...>`.
- Variants: single (paged / MLA / sharded are internal branches of the one factory).
- Custom `compute_program_hash`: **none** (removed historically; grep confirms none). Leave as-is.

### Kernels (all file-path-instantiated, all over the full `core_grid`)
| unique_id | source | config | opt_level (resolved) |
|---|---|---|---|
| reader  | `device/kernels/dataflow/reader_decode_all.cpp` | `ReaderConfigDescriptor{}` (RISCV_1/NOC_0 default) | O2 (DM default) |
| writer  | `device/kernels/dataflow/writer_decode_all.cpp` | `WriterConfigDescriptor{}` (RISCV_0/NOC_1 default) | O2 (DM default) |
| compute | `device/kernels/compute/sdpa_flash_decode.cpp` | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` | **O3** (compute default — must set explicitly) |

Shared kernel headers (own): `device/kernels/dataflow/dataflow_common.hpp`,
`device/kernels/rt_args_common.hpp`. Compute pulls in `pack_untilize`, `tilize_helpers`, `untilize_helpers`.

Reader CTAs: 37 positional scalars (idx 0–36), then 7 `TensorAccessorArgs` in order
q,k,v,mask,pos,page_table,attention_sink. Writer CTAs: 28 scalars then 1 `TensorAccessorArgs` (out).
Compute CTAs: 32 scalars (no TAs — compute binds no tensors).

### CBs (`add_cb`; entry_size = per-tile bytes, num_entries = tile count)
| CB | role | df | tile | borrowed? | disposition |
|---|---|---|---|---|---|
| c_0 | q_in | q_df | q_tile(half/full) | q_buffer if `q_locally_available` | **config-flip**: self-loop (tilize compute), 1P+1C (reader→compute DRAM), borrowed (MLA local) |
| c_1 | k_in | k_df | full | — | 1P+1C reader→compute |
| c_2 | v_in | v_df | full | — | 1P+1C reader→compute |
| c_3 | mask_in | mask_df | half/full | — | 1P+1C: producer = reader (non-causal read) OR writer (causal generate_mask); consumer = compute |
| c_4 | attention_sink | stats_df | half/full | — | 1P+1C reader→compute (only if use_attention_sink) |
| c_5 | scale/identity_scale | scalar_df | half/full | — | 1P+1C writer→compute |
| c_6 | m_in | stats_df | half/full | — | 1P+1C writer→compute |
| c_7 | l_in | stats_df | half/full | — | 1P+1C writer→compute |
| c_8 | writer_cur_pos | cur_pos_df | — | cur_pos_buffer if sharded | 1P+1C reader→writer (only if use_cur_pos_tensor) |
| c_9 | page_table | page_table_df | — | page_table_buffer if sharded | **self-loop** reader (reader fills + raw-reads own buffer) (only if is_paged_attention) |
| c_10 | q_rm | q_df | q_tile | — | 1P+1C reader→compute (tilize path) |
| c_11 | col_identity | scalar_df | full | — | **self-loop writer** (produced by writer generate_bcast_col_scalar; DEAD — no consumer in decode). Do NOT drop; ops-team cleanup. |
| c_12 | zero_in | scalar_df | half/full | — | 1P+1C writer→compute |
| c_13 | sliding_window_mask | mask_df | half/full | — | 1P+1C writer→compute (only if sliding_window_size>0) |
| c_14 | block_pad_mask | mask_df | half/full | — | 1P+1C writer→compute (only if has_block_padding) |
| c_15 | compute_cur_pos | cur_pos_df | — | — | 1P+1C reader→compute (only if use_cur_pos_tensor) |
| c_16 | out_o / out_worker | stats_df | half/full | — | **multi-binding**: writer P+C, compute P+C (2P+2C on intermediate tree nodes). `allow_instance_multi_binding=true`. |
| c_17 | out_m | stats_df | half/full | — | 1P+1C compute→writer |
| c_18 | out_l | stats_df | half/full | — | 1P+1C compute→writer |
| c_19 | intermed_out | stats_df | half/full | — | **self-loop writer** (writer raw cross-core read/write, no FIFO) (only if intermed_output_tiles>0, i.e. num_cores_per_head>1) |
| c_20 | out | out_df | full | out_buffer if is_output_sharded | 1P+1C compute→writer |
| c_21..c_31 | compute intermediates | im_df/stats_df | half/full | — | **self-loop compute** (produced+consumed within compute) |

### Semaphores (all WORKER, core_grid, initial_value 0)
| id | name | touchers |
|---|---|---|
| 0 | reducer | writer: raw poll via `get_semaphore(id)` (4-bit-per-round nibble decode) + `Semaphore<>(id).up(...)` |
| 1 | output | writer: `Semaphore<>(id).wait/up(...)` |
| 2 | k_mcast | reader (dataflow_common read_k): `Semaphore<>(id).set/set_multicast/wait/set` |

### Tensor accessors → TensorParameters
| tensor | host site | kernel accessor use | binding case |
|---|---|---|---|
| q | reader CTA TA #0 | `TensorAccessor(q_args,q_addr,q_page_size)` (dataflow_common:572) DRAM path; raw L1 `q_addr` (sharded) | Case1 (DRAM interleaved) / **Case2** (HEIGHT_SHARDED non-MLA raw L1 read via `get_bank_base_address`) / clean (MLA local → c_0 borrowed) |
| k | reader CTA TA #1 | `TensorAccessor(k_args,k_addr)` | Case1 |
| v | reader CTA TA #2 | `TensorAccessor(v_args,v_addr)` (only `!reuse_k`) | Case1 |
| cur_pos | reader CTA TA #4 | `TensorAccessor(pos_args,pos_addr)` (DRAM) | Case1 / clean (sharded → c_8 borrowed) |
| page_table | reader CTA TA #5 | via donor `read_page_table_for_batch` (DRAM) | Case1 / clean (sharded → c_9 borrowed) |
| attn_mask | reader CTA TA #3 | `TensorAccessor(mask_args,mask_addr)` | Case1 |
| attention_sink | reader CTA TA #6 | `TensorAccessor(attention_sink_args,...)` | Case1 |
| output | writer CTA TA #0 | `TensorAccessor(out_args,out_addr)` | Case1 / clean (sharded → c_20 borrowed) |

Delivery today: every buffer bound as a `Buffer*` (`BufferBinding`) via `emplace_runtime_args` (RTA slot 0..N),
re-patched on the fast cache-hit path. Port replaces with typed `TensorParameter` bindings.

### Work split
- Driver: manual core-group assignment (`core_group` vector of `num_active_cores` active cores; the rest idle).
  Not `split_work_to_cores`. Each of the 3 kernels runs on every active core (reordered so reducers land at
  batch boundaries). Idle cores get zeroed RTAs + early-return.
- `num_active_cores = num_cores_per_head * num_kv_heads * B / num_heads_per_core`.

### Shared kernels
- `../sdpa/device/kernels/dataflow/dataflow_common.hpp` (donor, shared with sdpa **prefill**): provides
  `copy_tile`, `virtual_seq_tile_id_to_physical_tile_id`, `read_page_table_for_batch`, `get_barrier_read_threshold`.
  All take `uint32_t cb` / raw ptrs → **bridged** (`dfb::name`→`uint32_t`), NOT rewritten. Exception:
  `read_page_table_for_batch` constructs `TensorAccessor(args,addr,size)` internally — incompatible with a
  binding token, so its ~6 lines are **inlined into sdpa_decode's own reader** using `TensorAccessor(tensor::page_table)`.
  Donor untouched.
- `../sdpa/device/kernels/compute/compute_common.hpp` (donor, prefill): `reduce_c`, `matmul_blocks`,
  `sub_exp_block*`, `correction_block`, `move_block`, `max_block`, `recip_block_inplace`, `mul/add_block*` — all
  take `uint32_t cb` → bridged, NOT rewritten.
- `ttnn/kernel/dataflow/generate_bcast_scalar.hpp` (`generate_bcast_col_scalar(CircularBuffer,...)`): a `_metal2`
  fork **already exists** (`generate_bcast_scalar_metal2.hpp`, takes `DataflowBuffer&`) → **bind the fork** (rung 1).
- `ttnn/cpp/ttnn/kernel_lib/{tilize_helpers,untilize_helpers,reduce_helpers_dataflow,l1_helpers}.hpp` — lib-owned,
  CB-id/template/`DataflowBuffer` shapes, bridge cleanly. Not forked.

### Flags
- `dataflow_common.hpp` (own) is private to sdpa_decode (only reader+writer include it) → converts with them.
- `rt_args_common.hpp` is pure math (no CB/arg/tensor/sem) → unchanged.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (base; no `override_runtime_arguments`).
- **Custom `compute_program_hash`**: none — leave intact (nothing to do).
- **Implementation notes**: direct-descriptor op → apply exception 3 (nested `SdpaDecodeProgramFactory` +
  `program_factory_t` variant; move `create_output_tensors`/`validate`/`compute_output_specs` stay on the device-op).
  Delete the pybind exposure of `create_descriptor` **if present** — audit says no pybound `create_descriptor`,
  so likely nothing to remove (verify).

## Planned Spec Shape
- **KernelSpecs (3)**: reader, writer, compute — 1:1 with legacy KernelDescriptors. No work-split multiplicity
  (single manual grid, not `split_work_to_cores`), so one KernelSpec each.
- **DataflowBufferSpecs**: one per live CB (c_0..c_20 present-per-config + c_21..c_31). Conditional CBs
  (c_4,c_8,c_9,c_13,c_14,c_15,c_19) declared only when their gate is set. Borrowed via `borrowed_from`
  (c_0 MLA, c_8/c_9/c_20 sharded). c_16 → `allow_instance_multi_binding=true`.
- **SemaphoreSpecs (3)**: reducer, output, k_mcast — `target_nodes = active cores`.
- **TensorParameters (8)**: q,k,v,cur_pos,page_table,attn_mask,attention_sink,output. Conditional ones
  declared only when present. `relaxations = none`.
- **WorkUnitSpecs (1)**: `{reader, writer, compute}` over the **active core set** (see Placement decision below).
- **Op-owned tensors**: none.

### Placement decision (structural, behavior-preserving) — flagged
Legacy creates all 3 kernels over the full `core_grid` and idle cores early-return via the vanishing
address-RTA (`q_addr==0` reader, `out_addr==0` writer, `arg(0)==65` compute). Since those address RTAs
become `TensorBinding`s (auto-injected, non-zero), the idle signal disappears. Resolution: place the single
WorkUnitSpec on the **active core set only** (`core_group`, `num_active_cores` cores) and drop the idle
machinery (host idle-args loop + the 3 kernel idle early-returns). Behavior-preserving: idle cores did nothing;
no cross-core op (K-multicast, tree reduction, output gather) ever targets an idle core (all targets are active
reducer/output/group cores). Aligns with "don't create kernels on unused cores". Documented in the report.

## Preserved Multiplicity
none — no work-split multiplicity in legacy (manual single-instance-per-core grid).

## Dropped Plumbing
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slots 0–6 (q/k/v/pos/page_table/mask/sink addrs, `Buffer*`) | `BufferBinding` | `TensorParameter` + `TensorBinding` |
| reader CTA TA #0–#6 (`TensorAccessorArgs(...).append_to`) | TA plumbing | binding mechanism (`TensorAccessor(tensor::name)`) |
| reader CTA #29 `q_page_size_bytes` (3rd TA arg for Q, `full_tile.get_tile_size(q_df)`) | Class-2 3rd arg | dropped (binding supplies aligned page size); Q DRAM read keeps its own tile-size for the async_read |
| reader CTA #32 `k_mcast_semaphore_id` | id in CTA | `SemaphoreBinding` → `sem::k_mcast` |
| writer RTA slot 0 (out addr, `Buffer*`) | `BufferBinding` | `TensorParameter` + `TensorBinding` |
| writer CTA TA #0 | TA plumbing | `TensorAccessor(tensor::output)` |
| writer CTA #11/#12 reducer/output sem ids | ids in CTA | `SemaphoreBinding` → `sem::reducer`/`sem::output` |
| page-table donor 3rd TA arg (`page_table_stick_size`) | Class-2 3rd arg | dropped from `TensorAccessor(tensor::page_table)`; value kept as named RTA `page_table_page_size` for async_read size + sharded stride |
| all positional scalar CTAs (reader 0–36, writer 0–27, compute 0–31) | positional | named CTAs |
| all positional scalar RTAs | positional | named RTAs (varargs only where noted below) |

Retained varargs (genuine indexed collections, per audit):
- reader: `all_output_noc_x/y` (count `num_output_cores`) — data-indexed by `cur_batch`.
- writer: `reduction_group_core_xs/ys` (count `num_cores_per_head`), `all_reducer_noc_x/y` (count
  `num_reducer_cores`), `all_output_noc_x/y` (count `num_output_cores`) — data-indexed.
- `children_per_round[6]` (reader n/a; writer+compute) — **named** (6 named RTAs; fixed source-literal count,
  per-round distinct field, per audit).

## Applied Patterns
- Self-loop DFB binding: c_9 (reader), c_19 (writer), c_21..c_31 (compute), c_11 (writer, dead-but-kept), and
  c_0 under the tilize/MLA configs.
- Sync-free / single-ended → self-loop: c_9, c_19, c_11.
- Multi-binding advanced option: c_16 (2P+2C tree reduction).
- Borrowed-memory DFB: c_0 (MLA), c_8 (sharded cur_pos), c_9 (sharded page_table), c_20 (sharded out).
- Conditional / optional DFB bindings: c_4, c_8, c_9, c_13, c_14, c_15, c_19 (host-conditional bind +
  `compiler_options.defines` + kernel `#ifdef`).
- Pass DFB handles directly to LLKs / kernel-lib helpers (compute + donor helpers).
- Shared-kernel rung 1 reuse: `generate_bcast_scalar_metal2.hpp`.

## Deferred / Flagged
- `sem::name` is emitted by genfiles as `constexpr std::uint32_t` (a bare id) — it flows freely through
  `Semaphore<>(sem::x)`, `get_semaphore(sem::x)`, and the `KMcastParams.mcast_sem_id` field. (Recipe's
  "sem:: doesn't convert to uint32_t" note is more conservative than genfiles reality.) Report as clarification.
- `c_11` col_identity dead-code (writer produces, no consumer) — kept as self-loop per audit; ops-team cleanup.
- unpack_modes: compute has `enable_32_bit_dest = fp32_dest_acc_en`. For every Float32-formatted DFB the compute
  kernel consumes, an explicit `UnpackMode::UnpackToSrc` entry (legacy default) is required when fp32 is on.
  Candidates: c_0(q_df), c_1(k_df), c_2(v_df), c_3/c_13/c_14(mask_df), c_5/c_12(scalar_df), c_10(q_df). Add
  conditionally on `(fp32_dest_acc_en && df==Float32)`.

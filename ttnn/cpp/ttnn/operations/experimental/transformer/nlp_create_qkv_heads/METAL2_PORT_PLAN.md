# Port Plan — `experimental/transformer/nlp_create_qkv_heads`

Port plan for `nlp_create_qkv_heads`, ported from the legacy `ProgramDescriptor`
(`create_descriptor`) API to Metal 2.0 (`MetalV2FactoryConcept` / `create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

Audit: GREEN (both factories). Brief + full audit alongside this file.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (both factories define `create_descriptor()` returning `ProgramDescriptor`).
- Variants: `NlpCreateHeadsDeviceOperation` → `Interleaved`, `Sharded` (`std::variant<Interleaved, Sharded>`).
  Selected by `select_program_factory`: `Sharded` when `input_tensor.is_sharded()`, else `Interleaved`.
- Custom `compute_program_hash`: **none** — already default reflection-based hash (audit confirmed).

*(Metal 2.0 target concept chosen in audit: `MetalV2FactoryConcept`. Carried forward below.)*

---

### Variant: Interleaved  (`device/nlp_create_qkv_heads_program_factory.cpp:19`)

Runtime-selected axes: `transpose_k_heads` (bool) and `read_from_input_tensor_kv` (bool, = `input_tensor_kv.has_value()`).
- `transpose_k_heads` toggles the presence of a compute kernel (`transpose_wh.cpp`) and re-routes K through cb0/cb16.
- `read_from_input_tensor_kv` toggles a second input tensor (in1) and its accessor.

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp` | all_cores | `q_num_tiles`, `kv_num_tiles`, then `TensorAccessorArgs(in0)`, `TensorAccessorArgs(in1-or-null)` | in0_addr(Buffer\*), in1_addr(Buffer\* or 0), num_blocks, in0_tile_id, in1_tile_id | `TRANSPOSE_K_HEADS`(cond), `READ_FROM_INPUT_TENSOR_KV`(cond) | Reader |
| writer | `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp` | all_cores | `q_out_h_tiles`, `q_out_w_tiles`, `q_out_HtWt`, `num_q_heads`, `num_kv_heads`, then `TensorAccessorArgs(q)`,`(k)`,`(v)` | q_addr,k_addr,v_addr (Buffer\*), num_blocks, q_out_h_dim, q/k/v_out_tensor_tile_id | `TRANSPOSE_K_HEADS`(cond) | Writer |
| compute_g1 | `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` **(cross-op / shared pool)** | core_group_1 | `num_blocks_per_core_group_1 * kv_num_tiles` (NHtWt) | — | — | Compute, `fp32_dest_acc_en = (input dtype == FLOAT32)` |
| compute_g2 | same | core_group_2 (if non-empty) | `num_blocks_per_core_group_2 * kv_num_tiles` | — | — | Compute, same fp32 flag |

Compute kernels present **only** when `transpose_k_heads`. Multiplicity = one compute KernelDescriptor per non-empty core group (work-split).

#### CBs
| index | total_size | core_ranges | data_format | page_size | present when |
|---|---|---|---|---|---|
| 1 (`cb_id_qv`) | 4·tile | all_cores | input df | tile | always (reader→writer FIFO for Q,V; also K when not transpose) |
| 0 (`cb_id_k`/src0) | 4·tile | all_cores | input df | tile | only `transpose_k_heads` (reader→compute) |
| 16 (out) | 4·tile | all_cores | input df | tile | only `transpose_k_heads` (compute→writer) |

No `.tile` set → default 32×32. No `.buffer` (all plain L1). No GlobalCircularBuffer.

#### Semaphores
none.

#### Tensor accessors (host sites)
| host site | originating Tensor | legacy RTA slot |
|---|---|---|
| reader CTA `TensorAccessorArgs(in0)` | input_tensor_q | reader RTA 0 (addr) |
| reader CTA `TensorAccessorArgs(in1)` | input_tensor_kv (optional) | reader RTA 1 (addr / 0 placeholder) |
| writer CTA `TensorAccessorArgs(q)` | output q | writer RTA 0 (addr) |
| writer CTA `TensorAccessorArgs(k)` | output k | writer RTA 1 (addr) |
| writer CTA `TensorAccessorArgs(v)` | output v | writer RTA 2 (addr) |

All **Case 1** (via `TensorAccessor`, addressed by `page_id`).

#### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_blocks)` where `num_blocks = B*1*S/TILE_HEIGHT`.
- `(num_cores, all_cores, core_group_1, core_group_2, per_g1, per_g2)`.
- Reader/writer run on `all_cores` (single KernelSpec each). Compute has per-group multiplicity.

---

### Variant: Sharded  (`device/nlp_create_qkv_heads_program_factory.cpp:430`)

One kernel source `reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp` instantiated **twice**
(reader-config + writer-config) over `q_cores`, differing only in CTAs (K→c_17 vs V→c_18) and per-instance RTAs
(risc0 vs risc1 Q heads; K vs V region offset). No compute kernel. `transpose_k_heads` forbidden (validate).

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | config |
|---|---|---|---|---|
| reader | `reader_..._sharded.cpp` | q_cores | `c_16 (q_out)`, `c_17 (k_out)` | Reader |
| writer | `reader_..._sharded.cpp` (same source) | q_cores | `c_16 (q_out)`, `c_18 (v_out)` | Writer |

RTA layout (slots 0–18 fixed named scalars; base-addr slots 6/15 are `Buffer*` bindings; slots 19+ = per-core NoC coord arrays):
0 head_size · 1 num_q_heads(risc0/risc1) · 2 num_q_heads_per_core · 3 remote_q_head_start_idx · 4 start_q_x · 5 start_q_y ·
**6 q_base (Buffer\*)** · 7 q_region_offset · 8 q_offset(L1 write) · 9 read_kv_heads · 10 num_kv_heads · 11 num_kv_heads_per_core ·
12 remote_kv_head_start_idx · 13 start_kv_x · 14 start_kv_y · **15 kv_base (Buffer\*)** · 16 kv_region_offset · 17 num_kv_tiles ·
18 num_x · 19..18+num_x noc_x[] · 19+num_x.. noc_y[].

#### CBs
| index | total_size | core_ranges | data_format | page_size | `.buffer` |
|---|---|---|---|---|---|
| 16 (`c_16`, q_out) | q_num_tiles·tile | q_cores | input df | tile | `std::get<0>(output).buffer()` (borrowed) |
| 17 (`c_17`, k_out) | k_num_tiles·tile | k_cores | input df | tile | `std::get<1>(output).buffer()` (borrowed) |
| 18 (`c_18`, v_out) | v_num_tiles·tile | v_cores(=k_cores) | input df | tile | `std::get<2>(output).buffer()` (borrowed) |

All three are **borrowed-memory CBs** (`.buffer` set). No GlobalCircularBuffer.

#### Semaphores
none.

#### Tensor accessors
None constructed kernel-side. Inputs used as **raw NoC base + arithmetic** (Case 2): q base (slot 6) and kv base (slot 15).
Outputs are the borrowed CBs (no accessor).

#### Work split
- Not `split_work_to_cores`. Per-core args in `build_sharded_core_args`; `num_cores = max(q_cores, k_cores) = q_cores`
  (validate guarantees `num_q_heads ≥ num_kv_heads` and shared grid, so `k_cores ⊆ q_cores`, `num_cores = q_cores.num_cores()`).
- `read_kv_heads = (i < k_cores.num_cores())` gates KV work to the first `k_cores` cores.

### Cross-op kernels
- `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` — shared **compute** pool (`ttnn/cpp/ttnn/kernel/compute/`), **outside** the op directory.
  Borrowers: this op, `nlp_create_qkv_heads_boltz`, `nlp_create_qkv_heads_vit`, `split_query_key_value_and_split_heads`.
  Only this op is being ported now → consumers cannot co-migrate → **fork** path (see Applied Patterns).

### Flags
- Interleaved reader slot 1 (`in1_tensor_addr`) is a dead placeholder (`0`) on the non-KV path — read into `in1_tensor_addr`
  but only used under `#ifdef READ_FROM_INPUT_TENSOR_KV`. Audit "keep offsets stable." In the port it becomes a conditional
  tensor binding (present only under KV), so the placeholder disappears cleanly.
- Sharded slot 7 (`q_region_offset`) is constant 0; kept as a real scalar the kernel adds. Keep as a named RTA (not dropped).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none — nothing to delete.
- **Pybind `create_descriptor`**: none — `nlp_create_qkv_heads_nanobind.cpp` binds only the top-level op; no factory
  entry point is pybound, so **no pybind removal is forced**.
- **Implementation notes**: Two factories in one `program_factory_t` variant. Ported **independently** — each factory's
  `create_descriptor` becomes `create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`. A mixed
  variant (one factory on Metal 2.0, one still on the descriptor concept) builds and dispatches per-factory, so the two
  can land in either order. Order chosen: **Interleaved first, then Sharded** (Sharded carries a borrowed-DFB node-set risk, see Deferred/Flagged).

## Planned Spec Shape

### Variant: Interleaved
- **KernelSpecs**: `reader` (1, on all_cores), `writer` (1, on all_cores), and — only when `transpose_k_heads` —
  `compute_g1` (core_group_1) + `compute_g2` (core_group_2, if non-empty). Preserve compute multiplicity (one KernelSpec per group).
- **DataflowBufferSpecs**: `QV_DFB` (cb1, always), plus — only when `transpose_k_heads` — `K_IN_DFB` (cb0, reader→compute)
  and `K_OUT_DFB` (cb16, compute→writer). `entry_size = single_tile_size`, `num_entries = 4`, `data_format_metadata = input df`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `IN0` (input_q), `IN1` (input_kv, conditional), `Q_OUT`, `K_OUT`, `V_OUT`. All Case 1.
- **WorkUnitSpecs**: one per core group. `wu_g1 {reader, writer, [compute_g1]}` on core_group_1; `wu_g2 {reader, writer, [compute_g2]}`
  on core_group_2 (when present). Reader/writer are single KernelSpecs shared across both WUs.

### Variant: Sharded
- **KernelSpecs**: `reader` and `writer` — **same source**, two KernelSpecs (dual-instance work-split), both on q_cores.
- **DataflowBufferSpecs**: `Q_OUT_DFB` (c_16, borrowed_from Q_OUT), `K_OUT_DFB` (c_17, borrowed_from K_OUT), `V_OUT_DFB` (c_18, borrowed_from V_OUT).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `IN_Q` (input_q, Case 2), `IN_KV` (kv tensor if present else q tensor — fused path, Case 2),
  `Q_OUT`, `K_OUT`, `V_OUT` (back the borrowed DFBs; no kernel-side accessor).
- **WorkUnitSpecs**: one WU `{reader, writer}` on q_cores.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| Interleaved compute_g1 (grp1) + compute_g2 (grp2), source `transpose_wh.cpp` (fork) | `compute_g1`, `compute_g2` | `wu_g1`, `wu_g2` (disjoint node sets) | each binds `K_IN_DFB` CONSUMER + `K_OUT_DFB` PRODUCER over its own group (disjoint → one role each, no flag) |
| Sharded reader-config + writer-config, source `reader_..._sharded.cpp` | `reader`, `writer` (same grid q_cores) | one WU (both) | `Q_OUT_DFB`(c_16): reader PRODUCER + writer CONSUMER (**1P+1C dual-instance**, cosmetic on Gen1, no flag). `K_OUT_DFB`(c_17): reader self-loop. `V_OUT_DFB`(c_18): writer self-loop. |

## Dropped Plumbing

### Interleaved
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA 0 | `in0_buffer` (Buffer\* addr) → `get_arg_val<uint32_t>(0)` → `TensorAccessor(in0_args, addr)` | `TensorBinding(IN0)`; kernel `TensorAccessor(tensor::in0)` |
| reader RTA 1 | `in1_buffer`/0 placeholder → addr | conditional `TensorBinding(IN1)` under `READ_FROM_INPUT_TENSOR_KV`; placeholder drops |
| reader CTA `TensorAccessorArgs(in0)`, `(in1)` | accessor-args CTAs | binding mechanism (dropped) |
| writer RTA 0/1/2 | q/k/v `Buffer*` addrs | `TensorBinding(Q_OUT/K_OUT/V_OUT)` |
| writer CTA `TensorAccessorArgs(q/k/v)` | accessor-args CTAs | binding mechanism (dropped) |
| reader positional CTAs 0,1 | `q_num_tiles`, `kv_num_tiles` | named CTAs `{q_num_tiles, kv_num_tiles}` |
| writer positional CTAs 0..4 | `q_out_h_tiles`,`q_out_w_tiles`,`q_out_HtWt`,`q_out_c`,`kv_out_c` | named CTAs |
| compute positional CTA 0 | `NHtWt` | named CTA `{NHtWt}` |
| reader/writer positional RTAs | num_blocks, tile ids, dims | named RTAs |
| Magic CB indices `cb_id_qv=1`, `cb_id_k=0/1`, `cb_id=16` (kernel literals) | hardcoded CB ids | `dfb::qv`, `dfb::k_in`, `dfb::k_out` |
| `get_tile_size(cb_id)` (kernel) | free fn by cb id | `dfb.get_tile_size()` (member) |

No page-size 3rd-arg CTAs (all `TensorAccessor` 2-arg). No semaphore-ID RTAs.

### Sharded
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader/writer RTA slot 6 | q input `Buffer*` base | `TensorBinding(IN_Q)`; kernel `TensorAccessor(tensor::in_q).get_bank_base_address()` (Case 2) |
| reader/writer RTA slot 15 | kv input `Buffer*` base (fused: q) | `TensorBinding(IN_KV)`; `TensorAccessor(tensor::in_kv).get_bank_base_address()` (Case 2) |
| reader/writer CTA 0,1 | `c_16`, `c_17`/`c_18` (magic CB idx) | `dfb::q_out`, `dfb::kv_out` bindings |
| reader/writer RTA slots 0–5,7–14,16–18 | positional RTAs | named RTAs |
| reader/writer RTA slots 19+ | `in0_mcast_noc_x[]`/`in0_mcast_noc_y[]` (indexed collections) | RTA **varargs** (`get_vararg`) — see Applied Patterns |
| `get_write_ptr()` on borrowed CBs | CB by id | `DataflowBuffer(dfb::name).get_write_ptr()` |

## Applied Patterns

- **[Two-toucher DFB → assign 1P+1C]**: Sharded `Q_OUT_DFB` (c_16) — reader-config PRODUCER + writer-config CONSUMER
  (both raw `get_write_ptr`, disjoint Q-head offsets; sync-free, cosmetic label). **No** multi-binding flag.
- **[Sync-free / single-ended CB → self-loop]**: Sharded `K_OUT_DFB` (c_17) — reader self-loop (PRODUCER+CONSUMER);
  `V_OUT_DFB` (c_18) — writer self-loop.
- **[Borrowed DFB]** (`DataflowBufferSpec::borrowed_from`): Sharded c_16/c_17/c_18 back the Q/K/V output buffers.
- **[Case 2 raw-pointer binding]** (`get_bank_base_address`): Sharded input Q and KV bases, DM kernel, raw walk unchanged.
- **[Conditional / optional DFB + tensor bindings]**: Interleaved `IN1` tensor binding + `K_IN_DFB`/`K_OUT_DFB` gated
  by `READ_FROM_INPUT_TENSOR_KV` / `TRANSPOSE_K_HEADS` `#define`s.
- **[Multi-variant / runtime-selected sources]**: Interleaved compute path presence.
- **[Modifying a shared dataflow kernel — FORK]**: `transpose_wh.cpp` forked into the op directory as
  `device/kernels/compute/transpose_wh_metal2.cpp` (consumers can't co-migrate). Original untouched; keeps all writes in-scope.
- **[Caution: Avoid varargs unless necessary]**: Sharded NoC-coord arrays are genuine indexed collections
  (count = num_cores_x/num_cores_y, runtime-indexed by `q_x`/`kv_x`) → legitimately retained as varargs.
- **[Preserved multiplicity — per-group compute CTA]**: Interleaved `compute_g1`/`compute_g2` keep per-group CTAs.

## Deferred / Flagged

- **RISK — Sharded borrowed-DFB node set vs. binding kernel node set.** In GQA (`num_q_heads > num_kv_heads`), `q_cores ⊃ k_cores`.
  The reader/writer kernels run on `q_cores`, so any DFB they bind (incl. `K_OUT_DFB`/`V_OUT_DFB`, borrowed from the K/V output
  tensors that are sharded on `k_cores` only) derives a node set of `q_cores`. On `q_cores \ k_cores` the borrowed backing tensor
  has no shard. Legacy allocated c_17/c_18 with `core_ranges = k_cores` (narrower than the kernel grid), which Metal 2.0's
  derived-placement model cannot directly express (DFB node set = union of bound kernels' WU nodes). The kernel never *touches*
  those DFBs on `q_cores \ k_cores` (guarded by the runtime `read_kv_heads`), so it *may* be harmless — but the spec validator /
  borrowed-address resolution may reject it. **Plan: build the natural spec, run the sharded GQA tests
  (`test_sharded_nlp_create_qkv_heads_*`, which cover 16q/8kv), and if they fail, capitulate on the Sharded factory with a precise
  framework-gap finding — the Interleaved factory ships independently.**
  **RESOLVED (post-construction): the risk did NOT materialize.** All three output DFBs derive a `q_cores` node set (reader+writer
  both on `q_cores`), so every node hosts the contiguous id set `{q_out=0, k_out=1, v_out=2}` — no interior hole (avoids
  [[metal2-borrowed-dfb-id-hole-bug]]), and the borrowed backing resolves correctly on `q_cores \ k_cores` (the kernel never touches
  k_out/v_out there via the runtime `read_kv_heads` guard). Sharded GQA tests pass at PCC ≥ 0.9999. See the port report's Successes.
- Recipe/triage note: `analyses/2026-07-19_offset_base_pointers.md` over-lists this op (stale post-`86872e0a06a`) — for the triage owner (audit already flagged).

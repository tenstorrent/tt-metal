# Port Plan — rotary_embedding_llama_fused_qk

Port plan for `rotary_embedding_llama_fused_qk`, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (single factory returns `tt::tt_metal::ProgramDescriptor`)
- Variants: one — `RotaryEmbeddingLlamaFusedQKProgramFactory` (no `select_program_factory`; the variant has a single factory, framework auto-selects)
- Custom `compute_program_hash`: none — already default reflection-based hash

This op is **compute-only**: there are no reader/writer dataflow kernels. There is ONE compute KernelDescriptor whose `kernel_source` is selected at host time by `operation_attributes.row_major_QK`:
- `device/kernels/compute/rotary_embedding_llama_sharded.cpp` (tile / non-row-major)
- `device/kernels/compute/rotary_embedding_llama_sharded_row_major.cpp` (row-major)

Both sources are converted; the host picks the `.source` path, the chosen one is the single compute KernelSpec.

### Kernels
| unique_id | source | core_ranges | CTAs (positional → named) | RTAs | config |
|---|---|---|---|---|---|
| compute | `rotary_embedding_llama_sharded.cpp` OR `..._row_major.cpp` (host-selected on `row_major_QK`) | `all_cores_bb` (bounding box of cos/sin grid) | 13 positional CB-idx/dim CTAs → see Dropped Plumbing | `is_q` (per-core: 1 on q_cores, 0 on k_cores) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en}` |

### CBs (10 total: 7 borrowed-memory + 3 interm scratch)
| index | total_size | core_ranges | data_format | page_size | borrowed (`.buffer`) |
|---|---|---|---|---|---|
| c_0  q_in      | num_q_input_tiles · in_tile  | all_cores_bb | input fmt (bf16) | in_tile  | **q_input.buffer()** |
| c_1  k_in      | num_k_input_tiles · in_tile  | all_cores_bb | input fmt (bf16) | in_tile  | **k_input.buffer()** |
| c_2  cos       | num_cos_sin_tiles · cos_tile | all_cores_bb | cos fmt          | cos_tile | **cos.buffer()** |
| c_3  sin       | num_cos_sin_tiles · sin_tile | all_cores_bb | sin fmt          | sin_tile | **sin.buffer()** |
| c_4  trans_mat | 1 · trans_tile               | all_cores_bb | trans fmt        | trans_tile | **trans_mat.buffer()** |
| c_24 rotated_in_interm | head_dim_t · in_tile | all_cores_bb | input fmt | in_tile | — (scratch) |
| c_25 cos_interm        | head_dim_t · in_tile | all_cores_bb | cos fmt   | cos_tile | — (scratch) |
| c_26 sin_interm        | head_dim_t · in_tile | all_cores_bb | sin fmt   | sin_tile | — (scratch) |
| c_16 q_out     | num_q_output_tiles · out_tile | all_cores_bb | output fmt | out_tile | **q_output.buffer()** |
| c_17 k_out     | num_k_output_tiles · out_tile | all_cores_bb | output fmt | out_tile | **k_output.buffer()** |

### Semaphores
none

### Tensor accessors
none. The compute kernel never builds a `TensorAccessor` and never reads a base address — every CB is accessed only by its CB id (FIFO ops / LLK reads). The 7 backing tensors are declared as `TensorParameter`s **only** to resolve the borrowed DFB addresses via `borrowed_from`; they are NOT bound as `TensorBinding`s.

### Work split
n/a — single compute KernelDescriptor over `all_cores_bb`. Per-core RTA `is_q` (1 for cores in `q_cores`, 0 for cores in `k_cores`); cores in `all_cores_bb` outside both `q_cores`/`k_cores` (`unused_cores`) get no runtime args in legacy. No `split_work_to_cores` core-group CTA multiplicity → no preserved multiplicity.

### Cross-op kernels
none — both compute kernels live in this op's directory.

### Flags
- The compute kernel selects `in_cb`/`out_cb`/`Ht` at RUNTIME from `is_q` (a per-core RTA). The two candidate CBs (q_in vs k_in, q_out vs k_out) are bound on the SAME compute kernel; the kernel uses `uint32_t in_cb = is_q ? q_in : k_in;` runtime-dynamic CB selection (whitelist rule for runtime-dynamic CB selection).
- `cos_cb`, `sin_cb`, `trans_mat_cb` are read only through LLK calls (`mul_tiles_bcast` / `mul_tiles` / `matmul_tiles`) — no FIFO ops in the active code path → **fake CBs** (address sources).
- `q_in`/`k_in` use `reserve_back/push_back/wait_front/pop_front` and `q_out`/`k_out` use `reserve_back/push_back` — the FIFO ops are present but the data is resident (borrowed), so the producer is a white-lie; they are still **fake CBs** (one-ended from the validator's view: only the compute kernel touches them).
- `rotated_in_interm`, `cos_interm`, `sin_interm` are **real** intra-kernel FIFOs (reserve_back/push_back then wait_front/pop_front all within compute) — genuine self-loops.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`
- **Custom `compute_program_hash`**: none
- **Implementation notes**: `create_descriptor` → `create_program_spec` returning `ProgramArtifacts`. The factory selects the compute `.source` on `row_major_QK` (single KernelSpec, not multi-variant fan-out — both sources have identical bindings/CTAs).

## Planned Spec Shape
- **KernelSpecs**: one — `compute` (source chosen on `row_major_QK`), `ComputeHardwareConfig{math_fidelity, fp32_dest_acc_en}`, per-core RTA `is_q`.
- **DataflowBufferSpecs**: 10 — 7 with `borrowed_from` (Q_IN, K_IN, COS, SIN, TRANS_MAT, Q_OUT, K_OUT) + 3 scratch (ROTATED_INTERM, COS_INTERM, SIN_INTERM).
- **SemaphoreSpecs**: none.
- **TensorParameters**: 7 — Q_IN, K_IN, COS, SIN, TRANS_MAT, Q_OUT, K_OUT (each backs exactly one borrowed DFB; none bound as a TensorBinding).
- **WorkUnitSpecs**: one — `{compute}` on `all_cores_bb`.

### DFB endpoint bindings — ALL ON THE SINGLE COMPUTE KERNEL (self-loops)
Because the op is compute-only and all data is resident, every DFB is one-ended from the validator's view; each is bound as a **self-loop** (PRODUCER + CONSUMER) on the compute kernel, and each gets `advanced_options.dfb_self_loop_connectivities[DFB] = DFBSelfLoopConnectivity::INTRA`.
- Fake CBs (borrowed, white-lie self-loops): Q_IN, K_IN, COS, SIN, TRANS_MAT, Q_OUT, K_OUT.
- Real self-loops (genuine intra-kernel FIFO): ROTATED_INTERM, COS_INTERM, SIN_INTERM.

## Preserved Multiplicity
none — no work-split core-group CTA multiplicity in legacy. (The per-core q/k distinction is an RTA in legacy and stays an RTA — it was never a CTA, so this is not a CTA→RTA demotion.)

## Dropped Plumbing
| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory CTA slot 0 | `q_input_cb_index` (c_0) | `DFBBinding(Q_IN, …)` → kernel `dfb::q_in` |
| factory CTA slot 1 | `q_output_cb_index` (c_16) | `DFBBinding(Q_OUT, …)` → kernel `dfb::q_out` |
| factory CTA slot 2 | `q_n_heads_t` | named CTA `q_Ht` |
| factory CTA slot 3 | `k_input_cb_index` (c_1) | `DFBBinding(K_IN, …)` → kernel `dfb::k_in` |
| factory CTA slot 4 | `k_output_cb_index` (c_17) | `DFBBinding(K_OUT, …)` → kernel `dfb::k_out` |
| factory CTA slot 5 | `k_n_heads_t` | named CTA `k_Ht` |
| factory CTA slot 6 | `head_dim_t` | named CTA `Wt` |
| factory CTA slot 7 | `cos_cb_index` (c_2) | `DFBBinding(COS, …)` → kernel `dfb::cos` |
| factory CTA slot 8 | `sin_cb_index` (c_3) | `DFBBinding(SIN, …)` → kernel `dfb::sin` |
| factory CTA slot 9 | `trans_mat_cb_index` (c_4) | `DFBBinding(TRANS_MAT, …)` → kernel `dfb::trans_mat` |
| factory CTA slot 10 | `rotated_input_interm_cb_index` (c_24) | `DFBBinding(ROTATED_INTERM, …)` → kernel `dfb::rotated_in_interm` |
| factory CTA slot 11 | `cos_interm_cb_index` (c_25) | `DFBBinding(COS_INTERM, …)` → kernel `dfb::cos_interm` |
| factory CTA slot 12 | `sin_interm_cb_index` (c_26) | `DFBBinding(SIN_INTERM, …)` → kernel `dfb::sin_interm` |
| compute RTA slot 0 | `get_arg_val<uint32_t>(0)` (is_q) | named RTA `get_arg(args::is_q)` |

Note: of the 13 CTAs, 10 were magic CB indices (→ DFB bindings) and 3 were dimensions (`q_Ht`, `k_Ht`, `Wt` → named CTAs). All CTAs become named/DFB-bound.

## Applied Patterns
- [Fake CB → self-loop DFB](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-fake-cb--self-loop-dfb-interim-workaround): Q_IN, K_IN, COS, SIN, TRANS_MAT, Q_OUT, K_OUT (borrowed, address-source; PRODUCER+CONSUMER on compute).
- [Self-loop DFB binding](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-self-loop-dfb-binding): ROTATED_INTERM, COS_INTERM, SIN_INTERM (real intra-kernel FIFO; PRODUCER+CONSUMER on compute). All self-loops also set `dfb_self_loop_connectivities[DFB] = INTRA`.
- Borrowed-memory DFB: `borrowed_from = <TensorParameter>` for all 7 backing tensors.
- Runtime-dynamic CB selection: kernel-side `uint32_t in_cb = is_q ? (uint32_t)dfb::q_in : (uint32_t)dfb::k_in;` (and out_cb, Ht). No `.id` extraction — uses the `operator uint32_t` implicit conversion.

## Deferred / Flagged
- The 7 borrowed/fake-CB self-loops are interim validator-satisfying devices (tensor-local-view kind for the 5 read-only inputs; the 2 outputs are write-only address sinks). Flagged for the eventual local-`TensorAccessor` / scratchpad migration — see PORT_REPORT "Open items."
- `row_major[2048-1-1-64-64]` is a PRE-EXISTING Blackhole failure unrelated to this port (flagged by the orchestrator). Not a regression.

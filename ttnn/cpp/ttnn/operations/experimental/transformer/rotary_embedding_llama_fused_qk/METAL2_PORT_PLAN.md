# Port Plan — `experimental/transformer/rotary_embedding_llama_fused_qk`

Port plan for `rotary_embedding_llama_fused_qk`, ported from the `ProgramDescriptor`
(`ProgramDescriptorFactoryConcept`) API to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — single `create_descriptor(...)` returning a `ProgramDescriptor`
  (`rotary_embedding_llama_fused_qk_program_factory.cpp:18`, `.hpp:18`).
- Variants: single (`program_factory_t = std::variant<RotaryEmbeddingLlamaFusedQKProgramFactory>`).
- Custom `compute_program_hash`: none — default reflection-based hash (audit confirmed, grep clean).
- Single `KernelDescriptor` (a compute kernel), whose **source file** is runtime-selected by
  `operation_attributes.row_major_QK`: tiled `.../compute/rotary_embedding_llama_sharded.cpp`
  vs row-major `.../compute/rotary_embedding_llama_sharded_row_major.cpp`
  (`program_factory.cpp:231-240`). Same CB layout and descriptor shape for both → one porting unit,
  one `KernelSpec` whose `source` is the existing runtime selection.

*(Target concept `MetalV2FactoryConcept`, chosen in the audit — see brief's TTNN factory analysis.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional 0..12) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|
| compute | tiled `rotary_embedding_llama_sharded.cpp` OR row-major `..._row_major.cpp` (runtime `row_major_QK`) | `all_cores_bb` (bounding box of `cos_sin_shard_spec->grid`) | `0 q_in_cb`, `1 q_out_cb`, `2 q_n_heads_t`, `3 k_in_cb`, `4 k_out_cb`, `5 k_n_heads_t`, `6 head_dim_t`, `7 cos_cb`, `8 sin_cb`, `9 trans_mat_cb`, `10 rotated_input_interm_cb`, `11 cos_interm_cb`, `12 sin_interm_cb` | `is_q` (1 on q-cores, 0 on k-cores; per-core, `program_factory.cpp:252-262`) | none | none | `ComputeConfigDescriptor{.math_fidelity=<resolved>, .fp32_dest_acc_en=<resolved>}` (all other fields left at descriptor defaults) |

CTA classification: 10 of the 13 positional CTAs are **CB indices** (slots 0,1,3,4,7,8,9,10,11,12)
→ become DFB bindings. 3 are **scalar dims** (slot 2 `q_n_heads_t`→kernel `q_Ht`, slot 5
`k_n_heads_t`→`k_Ht`, slot 6 `head_dim_t`→`Wt`) → become named CTAs.

### CBs
All ten allocated over `all_cores_bb`. `tile` field never set (default 32×32). `page_size` = the
format's single-tile size; `total_size` = num_tiles × single-tile size. All tensors are bfloat16
(validated), so every single-tile size == 2048 B; `entry_size = page_size`, `num_entries = total_size/page_size`.

| index | role | total_size | page_size / data_format | num_entries (→ DFB) | backing |
|---|---|---|---|---|---|
| c_0  | q_input   | `num_q_input_tiles·input_tile` | `input_single_tile_size` / input fmt | `num_q_input_tiles = q_n_heads_t·head_dim_t` | borrowed `q_src_buffer` (`:103`) |
| c_1  | k_input   | `num_k_input_tiles·input_tile` | `input_single_tile_size` / input fmt | `num_k_input_tiles = k_n_heads_t·head_dim_t` | borrowed `k_src_buffer` (`:115`) |
| c_2  | cos       | `num_cos_sin_tiles·cos_tile`   | `cos_single_tile_size` / cos fmt     | `num_cos_sin_tiles = head_dim_t·1` | borrowed `cos_buffer` (`:127`) |
| c_3  | sin       | `num_cos_sin_tiles·sin_tile`   | `sin_single_tile_size` / sin fmt     | `num_cos_sin_tiles` | borrowed `sin_buffer` (`:139`) |
| c_4  | trans_mat | `1·trans_mat_tile`             | `trans_mat_single_tile_size` / trans fmt | `num_trans_mat_tiles = 1` | borrowed `trans_mat_buffer` (`:153`) |
| c_24 | rotated_input_interm | `head_dim_t·input_tile` | `input_single_tile_size` / input fmt | `num_interm_tiles = head_dim_t` | local |
| c_25 | cos_interm | `head_dim_t·input_tile` | `cos_single_tile_size` / cos fmt | `num_interm_tiles` | local |
| c_26 | sin_interm | `head_dim_t·input_tile` | `sin_single_tile_size` / sin fmt | `num_interm_tiles` | local |
| c_16 | q_output  | `num_q_output_tiles·output_tile` | `output_single_tile_size` / output fmt | `num_q_output_tiles = num_q_input_tiles` | borrowed `q_dst_buffer` (`:199`) |
| c_17 | k_output  | `num_k_output_tiles·output_tile` | `output_single_tile_size` / output fmt | `num_k_output_tiles = num_k_input_tiles` | borrowed `k_dst_buffer` (`:210`) |

### Semaphores
none — the op declares no semaphores.

### Tensor accessors
none — no `TensorAccessor` anywhere; no address RTAs. Sharded tensors reach the kernel through
`CBDescriptor::buffer` borrowed-memory bindings, not accessors.

### Work split
- Driver: not `split_work_to_cores`. Core sets come straight from shard specs:
  `q_cores = q_shard_spec->grid`, `k_cores = k_shard_spec->grid`,
  `all_cores = cos_sin_shard_spec->grid`, `all_cores_bb = all_cores.bounding_box()`,
  `unused_cores = all_cores_bb.subtract(all_cores)`.
- Kernel + CBs placed on `all_cores_bb`. `is_q` RTA set to 1 on `q_cores`, 0 on `k_cores`.
  (`batch_per_core` is hardcoded to 1, `program_factory.cpp:80`, matched by the `seq_len==1`
  decode-mode validation — a documented current limitation, not port work.)

### Cross-op kernels
none — the op owns both kernel sources under its own `device/kernels/compute/`, instantiated by
file path. Includes are LLK-only (`api/compute/*`, `api/dataflow/circular_buffer.h`).

### Flags
- Tiled kernel declares three **unused** `CircularBuffer` objects (`cos_cb_obj`/`sin_cb_obj`/
  `trans_mat_cb_obj`, `rotary_embedding_llama_sharded.cpp:61-63`) — dead locals; the row-major
  variant does not. Whitelist rule 1 touches those lines (type rename), so they convert to
  `DataflowBuffer` in place and stay dead (faithful, zero functional change).
- Both kernels carry a commented-out `has_work` early-return (lines 24-28), a pre-existing TRISC2
  code-size workaround — left verbatim, not revived.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none — nothing to delete.
- **Factory selection**: single-variant `program_factory_t`; the framework adapter supplies the
  default `select_program_factory` (`mesh_device_operation_adapter.hpp:201-214` — only *required*
  when the variant has >1 alternative). No device-op-class edit needed.
- **Implementation notes**:
  - Factory method changes from `create_descriptor` → `create_program_artifacts`
    (returns `ttnn::device_operation::ProgramArtifacts`); signature `(const Params&, const Inputs&,
    Result&)` is preserved. No pybind `create_descriptor` exists (nanobind uses `ttnn::bind_function`),
    so no pybind removal.
  - Runtime source selection stays a runtime `.source = row_major_QK ? ... : ...` on the single `KernelSpec`.

## Planned Spec Shape

- **KernelSpecs**: one (`compute`). Source path runtime-selected by `row_major_QK`.
- **DataflowBufferSpecs**: ten, 1:1 with the legacy CBs. Seven borrowed (`borrowed_from = <TensorParameter>`):
  q_input, k_input, cos, sin, trans_mat, q_output, k_output. Three local: rotated_input_interm,
  cos_interm, sin_interm. Each carries `data_format_metadata` (required — all bound to a compute kernel);
  `tile_format_metadata` left default (legacy `.tile` unset). No `alias_with`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: seven (one per distinct tensor): q_input, k_input, cos, sin, trans_mat,
  q_output, k_output. Each paired with a `TensorArgument` referencing the same `MeshTensor`.
- **WorkUnitSpecs**: one — `{kernels = {compute}, target_nodes = all_cores_bb}`.
- **Op-owned tensors**: none.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. There is exactly one `KernelDescriptor` (a compute
kernel). Its two selectable **source files** collapse to one `KernelSpec` (same CB layout / bindings /
CTAs) with a runtime-selected `source` path, per the brief's shape note.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| compute CTA slots 0,1,3,4 (`q_in_cb`,`q_out_cb`,`k_in_cb`,`k_out_cb`) | CB-index CTAs | `DFBBinding`s (self-loop) → `dfb::q_in_cb` / `dfb::q_out_cb` / `dfb::k_in_cb` / `dfb::k_out_cb` |
| compute CTA slots 7,8,9 (`cos_cb`,`sin_cb`,`trans_mat_cb`) | CB-index CTAs | `DFBBinding`s → `dfb::cos_cb` / `dfb::sin_cb` / `dfb::trans_mat_cb` |
| compute CTA slots 10,11,12 (`rotated_input_interm_cb`,`cos_interm_cb`,`sin_interm_cb`) | CB-index CTAs | `DFBBinding`s → `dfb::rotated_in_interm_cb` / `dfb::cos_interm_cb` / `dfb::sin_interm_cb` |
| compute CTA slots 2,5,6 (`q_n_heads_t`,`k_n_heads_t`,`head_dim_t`) | positional CTAs | named CTAs `args::q_Ht`, `args::k_Ht`, `args::Wt` |
| compute RTA slot 0 (`is_q`) | positional `get_arg_val<uint32_t>(0)` | named RTA `args::is_q` (schema `runtime_arg_names = {"is_q"}`) |
| CBDescriptor `.buffer = <src/dst>_buffer` (7 sites) | borrowed-memory CB | `DataflowBufferSpec::borrowed_from = <TensorParameter>` + `TensorParameter`/`TensorArgument` |

No `tensor.buffer()->address()`, no `TensorAccessorArgs`, no page-size 3rd-arg, no semaphore-ID RTAs
(none exist in this op).

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md): all 10 DFBs are single-toucher (one compute kernel) → bind COMPUTE as both PRODUCER and CONSUMER (shared accessor name).
- [Pass DFB handles directly to LLKs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md): `dfb::name` passed straight into `matmul_tiles` / `mul_tiles_bcast` / `add_tiles` / `pack_tile` / init LLKs.
- Runtime CB selection (in_cb/out_cb chosen by the `is_q` RTA): seed a runtime `uint32_t` from the
  `dfb::` handle via the `DFBAccessor::operator uint32_t()` conversion, then construct the
  `DataflowBuffer` from that runtime id (low-level `DataflowBuffer(uint16_t)` ctor). This preserves
  the legacy kernel's runtime-selection shape (kept deliberately to avoid TRISC2 code-size blow-up
  from duplicating the loop body). Noted in the report — not an `.id` extraction or temp-wrapper.

## Deferred / Flagged

- **Compute hw_config is a hybrid Style A/B** (see report): the op resolves a TTNN `ComputeKernelConfig`
  (`get_compute_kernel_config_args`) but only threads `math_fidelity` + `fp32_dest_acc_en` into the Metal
  `ComputeConfigDescriptor`, leaving `math_approx_mode` / `dst_full_sync_en` at the **descriptor
  defaults** (both `false`). Porting via `to_compute_hardware_config(arch, resolved_config)` would import
  those two from the *resolved* config (resolved `math_approx_mode = true`) and silently flip
  `sfpu_precision_mode` Precise→Approximate. Faithful port = **Style B**: build `ComputeGen1Config`
  directly, set only `fpu_math_fidelity` + `enable_32_bit_dest`; the remaining `ComputeGen1Config`
  defaults (`sfpu_precision_mode=Precise`, `double_buffer_dest=true`, `bfp_pack_precision_mode=Approximate`)
  exactly reproduce the legacy descriptor defaults.
- **`unused_cores` RTA completeness**: legacy left bounding-box cores outside q∪k at the zero-default
  (=> k branch); Metal 2.0 requires an RTA on every target node, so `is_q=0` is set explicitly there.

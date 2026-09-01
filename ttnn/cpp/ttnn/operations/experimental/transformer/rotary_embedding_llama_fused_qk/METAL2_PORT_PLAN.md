# Port Plan — rotary_embedding_llama_fused_qk

Port plan for `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk`, ported from the legacy `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `RotaryEmbeddingLlamaFusedQKProgramFactory::create_descriptor` at `device/rotary_embedding_llama_fused_qk_program_factory.cpp:18`, returning `tt::tt_metal::ProgramDescriptor`. Factory methods live in a proper `program_factory_t` variant (`std::variant<RotaryEmbeddingLlamaFusedQKProgramFactory>`, `device/rotary_embedding_llama_fused_qk_device_operation.hpp:20`) — **not** the direct-descriptor shape; no exception-3 restructuring needed.
- Variants: single factory. The one `KernelDescriptor`'s **source is runtime-selected** by `operation_attributes.row_major_QK` (factory:237-242):
  - `false` → `device/kernels/compute/rotary_embedding_llama_sharded.cpp` (tiled QK)
  - `true` → `device/kernels/compute/rotary_embedding_llama_sharded_row_major.cpp` (tile-wrapped row-major QK)
  Both sources convert together with the factory (atomic unit). The CB/CTA/RTA structure is identical in both variants (verified by reading both kernels); only the inner compute loop differs.
- Custom `compute_program_hash`: none — default reflection-based hash (verified: no `compute_program_hash`, no `attribute_values`/`to_hash` backdoor).
- `override_runtime_arguments`: none. `get_dynamic_runtime_args`: none. Pybound `create_descriptor`: none (nanobind binds only the public composite op via `ttnn::bind_function`).

### Kernels
One `KernelDescriptor` (the op has **zero dataflow kernels** — all tensors are HEIGHT_SHARDED and reach the kernel as buffer-backed CBs):

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| compute | runtime-selected (see above) | `work_cores = q_cores ∪ k_cores` (factory:76 — deliberately NOT `all_cores_bb`; see Flags) | 13 slots (factory:220-236): 0=`q_input_cb_index`(c_0), 1=`q_output_cb_index`(c_16), 2=`q_n_heads_t`, 3=`k_input_cb_index`(c_1), 4=`k_output_cb_index`(c_17), 5=`k_n_heads_t`, 6=`head_dim_t`, 7=`cos_cb_index`(c_2), 8=`sin_cb_index`(c_3), 9=`trans_mat_cb_index`(c_4), 10=`rotated_input_interm_cb_index`(c_24), 11=`cos_interm_cb_index`(c_25), 12=`sin_interm_cb_index`(c_26) | none | 1 per core: `is_q` (1 on q cores, 0 on k cores; factory:258-268) | none | none | **O3** (resolved — `KernelDescriptor::opt_level` unset on a `ComputeConfigDescriptor` resolves to O3; `grep -n opt_level` on the factory: zero hits) | `ComputeConfigDescriptor{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en}` — a **subset** of the resolved TTNN compute config; `math_approx_mode` and `dst_full_sync_en` are dropped (descriptor defaults: `math_approx_mode=false`, `dst_full_sync_en=false`); `packer_l1_acc` has no descriptor counterpart |

### CBs
All 10 CBs are placed on `all_cores_bb` (bounding box of the cos/sin grid, factory:69) — see Flags for the deliberate kernel-range/CB-range asymmetry. Legacy `format_descriptors[i].tile` never set anywhere.

| index | total_size | core_ranges | data_format | page_size | tile | buffer (borrowed) |
|---|---|---|---|---|---|---|
| c_0 (q_input) | `num_q_input_tiles * input_single_tile_size` | all_cores_bb | input fmt | input tile size | — | `q_src_buffer` (factory:109) |
| c_1 (k_input) | `num_k_input_tiles * input_single_tile_size` | all_cores_bb | input fmt | input tile size | — | `k_src_buffer` (:121) |
| c_2 (cos) | `num_cos_sin_tiles * cos_single_tile_size` | all_cores_bb | cos fmt | cos tile size | — | `cos_buffer` (:133) |
| c_3 (sin) | `num_cos_sin_tiles * sin_single_tile_size` | all_cores_bb | sin fmt | sin tile size | — | `sin_buffer` (:145) |
| c_4 (trans_mat) | `1 * trans_mat_single_tile_size` | all_cores_bb | trans_mat fmt | trans_mat tile size | — | `trans_mat_buffer` (:159) |
| c_24 (rotated_input_interm) | `num_interm_tiles * input_single_tile_size` | all_cores_bb | input fmt | input tile size | — | plain |
| c_25 (cos_interm) | `num_interm_tiles * input_single_tile_size` | all_cores_bb | cos fmt | **cos tile size** (mixed bases vs total — see Flags) | — | plain |
| c_26 (sin_interm) | `num_interm_tiles * input_single_tile_size` | all_cores_bb | sin fmt | **sin tile size** (same) | — | plain |
| c_16 (q_output) | `num_q_output_tiles * output_single_tile_size` | all_cores_bb | output fmt | output tile size | — | `q_dst_buffer` (:205) |
| c_17 (k_output) | `num_k_output_tiles * output_single_tile_size` | all_cores_bb | output fmt | output tile size | — | `k_dst_buffer` (:216) |

No GlobalCircularBuffer anywhere (`.buffer` is the plain borrowed-memory path; `global_circular_buffer` / `address_offset` never set).

### Semaphores
none

### Tensor accessors
none — the op constructs no `TensorAccessor` (host or device) and passes no address RTAs. All tensor traffic is via `CBDescriptor::buffer` borrowed memory.

### Work split
n/a — no `split_work_to_cores`; parallelization is one batch-row per core over the sharded grids. Per-core variation is solely the `is_q` RTA (1 on `q_cores`, 0 on `k_cores`; both enumerated with `corerange_to_cores(..., row_wise=true)`).

### Shared kernels
none — both kernel sources are op-owned and this factory is their **sole binder** (filename census `grep -rl rotary_embedding_llama_sharded ttnn/cpp/ttnn/operations/` — the same-named files in the sibling `rotary_embedding_llama` op are that op's own private copies at a different path, already Metal 2.0 there). No `_metal2` fork exists or is needed — both sources convert **in place** (ordinary sole-binder case).

### Flags
- **Kernel range ≠ CB range is deliberate — keep it.** The kernel is on `work_cores = q_cores ∪ k_cores`, the CBs on `all_cores_bb`; the comment at factory:71-76 documents a watcher SIGABRT (out-of-bounds `get_arg_val(0)`) if the kernel lands on bounding-box "hole" cores that receive no RTAs. Metal 2.0 derives DFB placement from kernel bindings, so post-port the DFBs live on `work_cores` — the bounding-box hole cores lose their (never-dereferenced) CB configs and the interm CBs' L1 allocation there. Zero functional impact (holes run nothing); recorded in the port report.
- **Mixed tile-size bases in c_25/c_26** (total sized in `input_single_tile_size`, page in `cos/sin_single_tile_size`): consistent today only because validate forces every tensor to bfloat16 (all tile sizes equal). `DataflowBufferSpec` expresses total as `num_entries * entry_size`, which reproduces today's bytes exactly; the latent mismatch is unexpressible (and unrealizable) — noted in the report.
- **TRISC2 code-size cliff:** both kernels sit within ~4 bytes of the TRISC2 code-size limit with the profiler on (kernels:24-28, factory:255-257 — the `has_work` early-return is commented out for exactly this reason). Port keeps kernel code-shape minimal and sets `opt_level = O3` explicitly (legacy resolved level).
- **Dead locals in the tiled kernel:** `cos_cb_obj`/`sin_cb_obj`/`trans_mat_cb_obj` (sharded.cpp:61-63) are constructed and never used. Kept (renamed per the API rename) for minimal diff; the DFBs they name are bound regardless.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`
- **Custom `compute_program_hash`**: none
- **Implementation notes**: all name constants (`KernelSpecName`/`DFBSpecName`/`TensorParamName`) are declared function-locally inside `create_program_artifacts` to avoid unity-build anonymous-namespace collisions across the transformer family's factory TUs.

## Planned Spec Shape

- **KernelSpecs**: 1 — `compute`, source runtime-selected by `row_major_QK` (one `KernelSpec`, two possible sources; CTAs/RTA/bindings identical for both). `compiler_options.opt_level = O3` explicit. `hw_config = ComputeGen1Config{.fpu_math_fidelity = math_fidelity, .enable_32_bit_dest = fp32_dest_acc_en}` (Style B mirror of the legacy descriptor subset; the two dropped legacy fields land on `ComputeGen1Config` defaults which equal the legacy descriptor defaults: `sfpu_precision_mode = Precise` ⟷ `math_approx_mode = false`, `double_buffer_dest = true` ⟷ `dst_full_sync_en = false`; `bfp_pack_precision_mode` untouched — legacy never set `bfp8_pack_precise`). No `unpack_modes` entries: every DFB format is bfloat16 (validate forces BFLOAT16), so the Float32 required-entry rule never triggers even when `fp32_dest_acc_en = true`.
- **DataflowBufferSpecs**: 10, 1:1 with the legacy CBs (names extend the landed sibling `rotary_embedding_llama` vocabulary for the q/k split):
  - borrowed: `q_input` (c_0, `borrowed_from = q_input` param), `k_input` (c_1), `cos` (c_2), `sin` (c_3), `trans_mat` (c_4), `q_out` (c_16, from `q_output` param), `k_out` (c_17, from `k_output` param)
  - plain: `rotated_interm` (c_24), `cos_interm` (c_25), `sin_interm` (c_26)
  - `entry_size` = legacy `page_size`, `num_entries` = legacy `total_size / page_size` (all exact multiples), `data_format_metadata` = legacy `data_format`, `tile_format_metadata` unset (legacy `.tile` never set). No aliasing, no multi-binding flags, no dead-CB drops, no conditional DFBs.
- **SemaphoreSpecs**: none.
- **TensorParameters**: 7 — `q_input`, `k_input`, `cos`, `sin`, `trans_mat`, `q_output`, `k_output`, each from `<mesh_tensor>.tensor_spec()`. No kernel `TensorBinding`s (a compute kernel cannot bind a `TensorAccessor`); each parameter is referenced via its DFB's `borrowed_from`, which the validator accepts as the required use.
- **WorkUnitSpecs**: 1 — `main`, kernels {compute}, `target_nodes = work_cores` (**not** `all_cores_bb`; preserves the hole-core RTA hazard workaround).
- **Op-owned tensors**: none.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (single `KernelDescriptor`; per-core variation is the `is_q` RTA, which stays an RTA).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory:220-236 CTA slots 0,1,3,4,7,8,9,10,11,12 | magic CB indices (`tt::CBIndex::c_*`) in positional CTAs | `DFBBinding`s; kernel-side `dfb::q_input`, `dfb::q_out`, `dfb::k_input`, `dfb::k_out`, `dfb::cos`, `dfb::sin`, `dfb::trans_mat`, `dfb::rotated_interm`, `dfb::cos_interm`, `dfb::sin_interm` |
| factory:220-236 CTA slots 2, 5, 6 | positional CTAs `q_n_heads_t`, `k_n_heads_t`, `head_dim_t` | named CTAs `q_Ht`, `k_Ht`, `Wt` (names match the kernel's variables) |
| factory:258-268 / kernels:29 RTA slot 0 | positional per-core RTA (`is_q_arg`/`is_k_arg`) read via `get_arg_val<uint32_t>(0)` | named RTA `is_q` (`runtime_arg_schema` + per-node `runtime_arg_values` via `AddRuntimeArgsForNode`) |
| factory:100-217 `CBDescriptor{.buffer = <Buffer*>}` ×7 | borrowed-memory CBs carrying raw `Buffer*` | `TensorParameter` + `DataflowBufferSpec::borrowed_from` + `TensorArgument` (framework patches addresses on cache hit) |

No buffer-address RTAs, no `TensorAccessorArgs` plumbing, no page-size third-arg CTAs, no semaphore-ID RTAs existed.

## Applied Patterns

- **Sync-free and single-ended CBs → self-loop DFB** (patterns catalog): all 10 DFBs self-loop — the single compute kernel is the sole toucher of every CB on every node (census re-derived from both kernel sources; identical in both variants, agrees with the brief). Each gets 2 `DFBBinding`s on the compute `KernelSpec` (PRODUCER + CONSUMER, shared accessor name).
- **Pass DFB handles directly to LLKs**: `dfb::` tokens flow into `matmul_tiles`, `mul_tiles_bcast`, `mul_tiles`, `add_tiles`, `pack_tile`, `*_init`, `compute_kernel_hw_startup` via the constexpr `uint32_t` conversion.
- **q/k runtime mux (brief's one non-mechanical spot)**: the kernel keeps the legacy shape — constexpr locals take the `dfb::` tokens, the `is_q` branch selects into runtime `uint32_t` locals, and the in/out `DataflowBuffer` objects are constructed from the runtime-selected id via the public low-level `DataflowBuffer(uint16_t logical_dfb_id)` constructor (`api/dataflow/dataflow_buffer.h:113`). No compile-time q/k split (would change kernel instantiation shape); no per-object duplication (TRISC2 code-size cliff).

## Deferred / Flagged

- New findings during planning: none beyond the audit's. The row-major kernel variant's in-tree coverage is **TG-only** (`run_test_row_major_rotary_embedding_llama(..., fuse_qk=True)` is invoked by the Galaxy tests `models/demos/llama3_70b_galaxy/tests/unit_tests/test_llama_ops.py:554` and `test_qwen_ops.py:554`, both requiring an (8, 4) mesh) — not runnable on this bench, so it converts together with the factory and is verified by compile + review here, with TG execution evidence deferred to CI / the op owner; recorded in the report.

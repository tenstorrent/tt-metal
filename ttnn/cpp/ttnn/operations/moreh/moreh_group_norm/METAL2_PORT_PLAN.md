# Port Plan — moreh_group_norm

Port plan for `moreh/moreh_group_norm`, ported from the TTNN `descriptor`
(`create_descriptor` → `ProgramDescriptor`) concept to Metal 2.0
(`MetalV2FactoryConcept` → `create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (direct `create_descriptor` on the op struct; `HasDirectDescriptor`).
- Variants: single. One internal `use_large_algorithm` cache-miss branch selects small vs. large reader + compute kernels (same op, same concept — NOT a factory variant).
- Custom `compute_program_hash`: none — already the default reflection-based hash (audit-confirmed).

*(Target concept `MetalV2FactoryConcept`, inherited from the audit — see [TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels (per KernelDescriptor)
| unique_id | source | core_ranges | CTAs (positional) | RTAs (per-core) | defines | config |
|---|---|---|---|---|---|---|
| reader (small/large) | `device/kernels/dataflow/reader_moreh_group_norm_{small,large}.cpp` | all_cores | `gamma_has_value`, `beta_has_value`, then `TensorAccessorArgs(input/gamma/beta)` | input_buf, gamma_buf, beta_buf, scaler, eps, tile_offset, num_rows_per_core, num_inner_tiles, num_channels, origin_h, origin_w, block_size | — | ReaderConfigDescriptor (default) |
| writer | `device/kernels/dataflow/writer_moreh_group_norm.cpp` | all_cores | `mean_has_value`, `rstd_has_value`, then `TensorAccessorArgs(output/mean/rstd)` | output_buf, mean_buf, rstd_buf, tile_offset, num_rows_per_core, num_inner_tiles, num_groups, block_size | — | WriterConfigDescriptor (default) |
| compute_1 | `moreh_layer_norm/device/kernels/moreh_layer_norm_{small,large}_kernel.cpp` (BORROWED) | core_group_1 | num_rows_per_core_group_1, origin_h, origin_w, num_inner_tiles, block_size, gamma_has_value, beta_has_value, mean_has_value, rstd_has_value, is_lastdim_layernorm(=0), is_group_norm(=1) | — | REDUCE_OP=PoolType::AVG, REDUCE_DIM=ReduceDim::REDUCE_SCALAR | ComputeConfigDescriptor (Style A) |
| compute_2 | same source | core_group_2 (if non-empty) | as compute_1 but num_rows_per_core_group_2 | — | same | same |

### CBs (per CBDescriptor) — created only when tile count > 0 (`push_cb_if_nonzero`)
| CBIndex | role | tiles | present when |
|---|---|---|---|
| c_0 | input (x) | num_inner_tiles (small) / block_size (large) | always |
| c_1 | scaler | 1 | always |
| c_2 | eps | 1 | always |
| c_3 | gamma | block_size | gamma present |
| c_4 | beta | block_size | beta present |
| c_5 | mask_h | 1 | do_mask_h (origin_h % 32 != 0) |
| c_6 | mask_w | 1 | do_mask_w (origin_w % 32 != 0) |
| c_16 | output | block_size | always |
| c_17 | mean | 1 | mean required |
| c_18 | rstd | 1 | rstd required |
| c_24 | E[x] | 1 | always |
| c_25 | x-E[x] | num_inner (small) / 2*block_size (large) | always |
| c_26 | (x-E[x])^2 | 1 (small) / 2*block_size (large) | always |
| c_27 | Sum[(x-E[x])^2] | 1 | always |
| c_28 | Var[x] | 1 | always |
| c_29 | 1/sqrt(Var+eps) | 1 | always |
| c_30 | y*gamma+beta | 2*block_size | gamma or beta present |
| c_31 | Sum[x] | 2 | always |

All CBs share `data_format = datatype_to_dataformat_converter(input.dtype())`, `page_size = single_tile_size`.

### Semaphores
None (op declares no semaphores).

### Tensor accessors (all Case 1, via `TensorAccessor`)
- reader: input (RTA slot 0), gamma (slot 1, optional), beta (slot 2, optional).
- writer: output (slot 0), mean (slot 1, optional), rstd (slot 2, optional).
Each buffer is delivered as a `Buffer*` runtime-arg binding today; addresses folded into a `TensorAccessor(args, addr)` (2-arg, no 3rd page-size arg).

### Work split
`split_work_to_cores(grid, num_rows)` where `num_rows = n * num_groups`.
`(num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2)`.

### Cross-op / borrowed kernels
- Compute kernels `moreh_layer_norm_{small,large}_kernel.cpp` live in **`moreh_layer_norm`** (in-family), co-instantiated by moreh_layer_norm's own factory. moreh_layer_norm is being ported IN PARALLEL. Per orchestration constraints, these are **FORKED** into this op's directory (`device/kernels/compute/moreh_group_norm_{small,large}_kernel.cpp`), based on the committed (HEAD) legacy version, and the factory repointed to the forks. The port is fully self-contained.
- Shared headers `ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp` and `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` are shared-pool / lib-team surfaces; used unchanged (they already take `DataflowBuffer` / CB-index template params — Device 2.0 clean).

### Flags
- No unreferenced kernel files. No descriptor type outside the audit scan.
- Anomalies (team-only, NOT port work): `mean_memory_config`/`rstd_memory_config` accepted but unused; small reader double-reserves the input CB. Both preserved verbatim.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none (nothing to delete).
- **Implementation notes**: single-type `program_factory_t = std::variant<ProgramFactory>`; no custom `select_program_factory` needed (framework auto-returns the single factory). Pybind binds the plain `&ttnn::moreh_group_norm` function, not `create_descriptor` — no pybind edit needed.

## Planned Spec Shape
- **KernelSpecs**: reader, writer, compute_g1 (+ compute_g2 iff core_group_2 non-empty). 1:1 with legacy KernelDescriptors. Compute multiplicity preserved (per-group `num_rows_per_core` stays a CTA).
- **DataflowBufferSpecs**: one per present CB (mirror `push_cb_if_nonzero`). No aliasing, no borrowed memory.
- **SemaphoreSpecs**: none.
- **TensorParameters**: input, output (always); gamma, beta, mean, rstd (conditional). Each with a `TensorBinding` on the reader (inputs) / writer (outputs).
- **WorkUnitSpecs**: WU1 {reader, writer, compute_g1} on core_group_1; WU2 {reader, writer, compute_g2} on core_group_2 (iff non-empty). reader/writer are in both WUs (disjoint node coverage — legal shared-endpoint bindings).
- **Op-owned tensors**: none.

## Preserved Multiplicity
```
Legacy compute_desc_1/compute_desc_2 of source moreh_layer_norm_{small,large}_kernel.cpp
  → KernelSpecs [COMPUTE_G1, COMPUTE_G2] of forked moreh_group_norm_{small,large}_kernel.cpp
  → in WorkUnitSpecs [WU1(core_group_1), WU2(core_group_2)]
  → sharing input(c_0 CONSUMER), output(c_16 PRODUCER), + all self-loop intermediates; disjoint node sets, one role each — no flag.
```
reader (PRODUCER c_0..c_6) and writer (CONSUMER c_16..c_18) are single KernelSpecs in both WUs (disjoint node coverage per WU).

## Dropped Plumbing
- **Buffer-address RTAs** → `TensorBinding`:
  - reader slots input_buf/gamma_buf/beta_buf → `tensor::input`/`tensor::gamma`/`tensor::beta`.
  - writer slots output_buf/mean_buf/rstd_buf → `tensor::output`/`tensor::mean`/`tensor::rstd`.
- **`TensorAccessorArgs(...).append_to(cta)` plumbing** (reader `:217-219`, writer `:231-233`) + kernel-side `TensorAccessorArgs<N>()` / `next_compile_time_args_offset()` chains → binding mechanism; kernels build `TensorAccessor(tensor::name)`.
- **Magic CB indices**: none in CTAs (CB indices were kernel-local `cb_id++` counters / `tt::CBIndex::c_N` constants) → replaced by `dfb::name` handles.
- **Page-size 3rd CTA/RTA**: none (all accessors already 2-arg).
- **Semaphore-ID RTAs**: none.
- **Positional CTAs** → named CTAs (compute) or defines (presence flags):
  - reader/writer presence-flag CTAs `gamma_has_value`/`beta_has_value` / `mean_has_value`/`rstd_has_value` → `#define` (drive conditional bindings, see Applied Patterns).
  - compute CTAs → named: `num_rows_per_core`, `origin_H`, `origin_W`, `num_inner`, `block_size`, `is_lastdim_layernorm`, `is_groupnorm`. Presence-flag CTAs `gamma/beta/mean/rstd_has_value` → defines.

## Applied Patterns
- **Conditional / optional DFB bindings**: gamma (c_3), beta (c_4), mask_h (c_5), mask_w (c_6), mean (c_17), rstd (c_18), gamma_beta (c_30). Host conditionally binds + emits matching define (`GAMMA_HAS_VALUE`, `BETA_HAS_VALUE`, `DO_MASK_H`, `DO_MASK_W`, `MEAN_HAS_VALUE`, `RSTD_HAS_VALUE`); kernels `#ifdef`-gate the token alias and every reference. gamma_beta gated on `defined(GAMMA_HAS_VALUE) || defined(BETA_HAS_VALUE)`.
- **Self-loop DFB binding**: compute-only intermediates c_24–c_31 (E[x], x-E[x], (x-E[x])^2, Sum, Var, 1/sqrt, gamma_beta, Sum[x]) — compute bound both PRODUCER and CONSUMER (single accessor name each).
- **Same-FIFO aliasing (kernel-side)**: `cb_tmp = cb_ex` (small compute) / `cb_reuse = cb_xmm` (large compute) — one DFB, aliased handle; preserved as legacy kernel logic (faithful port).
- **Pass DFB handles directly to LLKs / kernel-lib**: `copy_tile`, `add_tiles`, `mul_tiles`, `mask_tile`, `binary_op_init_common`, `compute_kernel_lib::reduce<...>` all take `dfb::name` (implicit `→ uint32_t`).
- **Preserved-multiplicity work split** (compute_g1/g2 over disjoint node sets).

## Deferred / Flagged
- Compute config: Style A (op resolves a TTNN `ComputeKernelConfig`) → `ttnn::to_compute_hardware_config(arch, operation_attributes.compute_kernel_config)`. `unpack_modes` under Float32 + fp32_dest_acc_en: an explicit `UnpackToSrc` entry (legacy default) is added for each compute-consumed Float32 DFB only in that case (validator-required); otherwise omitted (legacy left `unpack_to_dest_mode` default).
- No structural issue uncovered that the audit missed. See report for the conditional-binding surface (7 conditional DFBs across 3 kernel families) which is heavier than a typical port but fully mechanical.

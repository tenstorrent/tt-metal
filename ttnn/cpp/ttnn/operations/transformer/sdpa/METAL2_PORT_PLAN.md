# Port Plan — SDPA (`SDPAOperation`) + JointSDPA (`JointSDPADeviceOperation`)

Port plan for the clean subset of `ttnn/cpp/ttnn/operations/transformer/sdpa`, ported from the
`ProgramDescriptor` (`create_descriptor`) API to Metal 2.0 (`ProgramSpecFactoryConcept`).
Scope is exactly the two factories the audit cleared GREEN: `SDPAProgramFactory` and
`JointSDPAProgramFactory`. The five RED DeviceOperations (Sparse, SparseMSA, RingDistributed,
RingJoint, ExpRingJoint) are **not** touched.

Written during the inventory + planning steps; committed alongside the port for review.

---

## Legacy Inventory

### Legacy factory shape
- **SDPA**: `SDPAOperation::SDPAProgramFactory::create_descriptor` → `ProgramDescriptor`
  (concept `ProgramDescriptorFactoryConcept`). Nested factory struct inside a
  `program_factory_t = std::variant<SDPAProgramFactory>` — **not** direct-descriptor, so
  `ttnn_factory.md` exception 3 does not apply.
- **JointSDPA**: `JointSDPADeviceOperation::JointSDPAProgramFactory::create_descriptor` →
  `ProgramDescriptor`. Same nested-variant shape.
- Variants: single (each op has one factory; no multi-variant `program_factory_t`).
- Custom `compute_program_hash`: **none** for either (grep clean; audit confirms). Default
  reflection hash. Nothing to preserve or touch.

*(Target concept `ProgramSpecFactoryConcept` chosen by the audit; carried forward below.)*

### Kernels

**SDPA** (all on the full compute grid `core_grid`, one KernelSpec each):
| id | source | config | opt_level (resolved) |
|---|---|---|---|
| reader | `kernels/dataflow/reader_interleaved.cpp` | ReaderConfigDescriptor (reader default: RISCV_1/NOC_0/DEDICATED) | O2 (DM default) |
| writer | `kernels/dataflow/writer_interleaved.cpp` | WriterConfigDescriptor (writer default: RISCV_0/NOC_1/DEDICATED) | O2 (DM default) |
| compute | `kernels/compute/sdpa.cpp` | ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode} | **O3** (compute default, must set explicitly) |

**JointSDPA** (all on `core_grid`, one KernelSpec each):
| id | source | config | opt_level |
|---|---|---|---|
| reader | `kernels/dataflow/joint_reader.cpp` | ReaderConfigDescriptor | O2 |
| writer | `kernels/dataflow/joint_writer.cpp` | WriterConfigDescriptor | O2 |
| compute | `kernels/compute/joint_sdpa.cpp` | ComputeConfigDescriptor{…same 4…} | **O3** |

Compute config style = **Style A** (op resolves a TTNN `ComputeKernelConfig` via
`get_compute_kernel_config_args`). Translate with `ttnn::to_compute_hardware_config(arch, compute_kernel_config)`.
The four resolved knobs (math_fidelity, math_approx_mode, fp32_dest_acc_en, dst_full_sync_en) are all
carried onto the legacy `ComputeConfigDescriptor` — no dropped-field hazard. `packer_l1_acc` has no
Metal 2.0 counterpart (no action). No legacy `unpack_to_dest_mode` / `bfp8_pack_precise`.

Defines (all three kernels, both ops): `STATS_GRANULARITY`, `SUB_EXP_GRANULARITY`,
`MUL_BCAST_GRANULARITY`, `DHT_GRANULARITY`, `REDUCE_GRANULARITY`, `EXP_APPROX_MODE`.

### CBs

**SDPA** — allocated sequentially via a running `next_cb_index++` (numeric ids config-dependent;
irrelevant post-port — each becomes a named DFB). Sizes: `entry_size = tile_size(df)`,
`num_entries = <n>_tiles`. Conditionals noted.
| DFB name | entry df | num_entries | condition |
|---|---|---|---|
| q_in | q_df | Sq_chunk_t·DHt·q_buffer_factor | always |
| k_in | k_df | Sk_chunk_t·DHt·2 | always |
| v_in | v_df | Sk_chunk_t·vDHt·2 | always |
| mask_in | lightweight?Float16_b:mask_df | mask_tiles | `needs_mask_cb` |
| cu_window_seqlens | cu_df | 1 | `is_windowed` |
| identity_scale_in | scalar_df | 1 | always |
| col_identity | scalar_df | 1 | always |
| page_table | Int32 | 1 (entry=page_table_stick_size) | `is_chunked` |
| chunk_start_idx_compute | Int32 | 1 (entry=32) | `flexible_chunked` |
| chunk_start_idx_writer | Int32 | 1 (entry=32) | `flexible_chunked` |
| attention_sink | sink_df | attention_sink_tiles | `use_attention_sink` |
| recip_scratch | im_df (Float16_b) | 1 | `use_streaming_compute` |
| qk_im | qk_im_df (Float32 if fp32_dest_acc_en else Float16_b) | qk_tiles | always |
| out_im_A / out_im_B | im_df | out_im_tiles | always |
| max_A / max_B | stats_df (Float16_b) | statistics_tiles | always |
| sum_A / sum_B | sum_df (Float32 if fp32_dest_acc_en else Float16_b) | statistics_tiles | always |
| exp_max_diff | stats_df | statistics_tiles | always |
| out | out_df | out0_t | always |

**JointSDPA** — fixed `tt::CBIndex::c_*` in kernels:
c_0 q_in, c_1 k_in, c_2 v_in, c_3 mask_in (**cond `use_joint_mask`**, Bfp4_b), c_5 identity_scale_in,
c_7 col_identity, c_16 out, c_24 qk_im, c_25 out_im_A, c_26 out_im_B, c_27 max_A, c_28 max_B,
c_29 sum_A, c_30 sum_B, c_31 exp_max_diff. im_df/stats_df = Float16_b (no Float32 DFBs → no unpack_modes).

### Semaphores
- **SDPA**: 3, only when `!is_causal`, on `core_grid`, CoreType WORKER:
  `sender` (initial 0=INVALID), `receiver` (initial 0=INVALID), `valid` (**initial 1=VALID**).
  Bound to the **reader** only. `valid`'s non-zero initial → `SemaphoreAdvancedOptions::initial_value`
  (deprecated field → suppress the deprecation warning with a local `#pragma`).
- **JointSDPA**: none.

### Tensor accessors (→ TensorParameter/TensorBinding)
- **SDPA reader** (`reader_interleaved.cpp:211-216` + inlined page-table): q_in, k_in, v_in, mask (cond),
  page_table (cond chunked), attention_sink (cond), chunk_start_idx (cond flexible).
- **SDPA writer** (`writer_interleaved.cpp:84,114`): out, cu_window_seqlens (cond windowed).
- **JointSDPA reader** (`joint_reader.cpp:56-61`): input_q, input_k, input_v, joint_q, joint_k, joint_v.
- **JointSDPA writer** (`joint_writer.cpp:52-53`): output, joint_output.
- All **Case 1** (via `TensorAccessor`). No Case-2 raw pointers. No `borrowed_from`.

### Work split
- Both ops: **single WorkUnitSpec** over the full `core_grid`; per-core work distributed purely via
  **RTAs** (SDPA: `global_q_start`/`global_q_count`; Joint: `local_batch/nh/q_start/end`). No
  per-group CTA multiplicity → no multi-KernelSpec.

### Shared kernels  *(census: `grep -rln "<exact kernel path>" ttnn/cpp/ttnn/operations`)*
- **SDPA `reader_interleaved.cpp`, `writer_interleaved.cpp`, `compute/sdpa.cpp` are LENT** — also
  bound by `ring_distributed_sdpa_program_factory.cpp` (a RED/blocked factory). No `_metal2` fork
  exists → **rung 2: create the fork** (`reader_interleaved_metal2.cpp`,
  `writer_interleaved_metal2.cpp`, `compute/sdpa_metal2.cpp`) beside the originals, convert the forks,
  point `SDPAProgramFactory`'s `KernelSpec::source` at them, and add a pointer comment to each original.
  Remaining consumer: `RingDistributedSDPADeviceOperation`.
- **JointSDPA `joint_reader.cpp`, `joint_writer.cpp`, `compute/joint_sdpa.cpp` are exclusive** to
  `joint_sdpa_program_factory.cpp` (the `ring_joint_*` substring hits are `ring_joint_reader.cpp` etc.,
  different files) → **convert in place**.
- Shared **helper headers** `kernels/dataflow/dataflow_common.hpp`, `kernels/compute/compute_common.hpp`,
  `kernels/compute/compute_streaming.hpp` are `#include`d by SDPA/Joint **and** the blocked ops. They
  are *shared routines*, not entry points: they take `uint32_t` cb-ids as params and read **no** args
  positionally (grep clean). Leave them untouched; pass `dfb::name` at call sites
  (`DFBAccessor::operator uint32_t()` bridges). They contain `CircularBuffer cb(cb_id)` and one
  page-table `TensorAccessor` — see Deferred/Flagged. `generate_bcast_scalar.hpp` →
  bind the existing `generate_bcast_scalar_metal2.hpp` fork.

### Flags
- Page-table read: `read_page_table_for_batch()` (a template in `dataflow_common.hpp`) is called
  **only** by the SDPA reader (`reader_interleaved.cpp:310`). It takes `TensorAccessorArgs` + raw addr;
  `tensor::` can't cross into the shared header → **inline the ~6-line read** into the reader fork with
  `TensorAccessor(tensor::page_table)`, dropping the redundant 3rd page-size arg (Class 2, no-op).
  The shared header stays byte-identical (its template goes uninstantiated).

---

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (both — no `override_runtime_arguments`).
- **Custom `compute_program_hash`**: none (both) — nothing to preserve.
- **Device-op-class edits forced**: none expected. Nested `program_factory_t` already exists (no
  exception 3); nanobind binds no `create_descriptor` (no exception 1/2). Only the factory method
  changes name/signature `create_descriptor` → `create_program_artifacts` in the `.hpp` + `.cpp`.

## Planned Spec Shape

### SDPA
- **KernelSpecs**: `reader`, `writer`, `compute` (sources = the `_metal2` forks).
- **DataflowBufferSpecs**: one per CB above (conditionals gated on host; kernel `#ifdef`).
- **SemaphoreSpecs**: `sender`, `receiver`, `valid` (only `!is_causal`), bound on reader.
- **TensorParameters**: q_in, k_in, v_in, mask, page_table, attention_sink, chunk_start_idx (reader);
  out, cu_window_seqlens (writer).
- **WorkUnitSpec**: one, `{reader, writer, compute}` over `core_grid`.
- **unpack_modes** (compute, only when `fp32_dest_acc_en`): `{qk_im, sum_A, sum_B} → UnpackToSrc`
  (Float32 DFBs consumed with enable_32_bit_dest; legacy default = UnpackToSrc). Always-bound DFBs.

### JointSDPA
- **KernelSpecs**: `reader`, `writer`, `compute` (in-place converted sources).
- **DataflowBufferSpecs**: c_0..c_31 above; `mask_in` conditional on `use_joint_mask`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: input_q/k/v, joint_q/k/v (reader); output, joint_output (writer).
- **WorkUnitSpec**: one, `{reader, writer, compute}` over `core_grid`.
- **unpack_modes**: none (no Float32 DFBs).

## Preserved Multiplicity
none — no work-split KernelDescriptor multiplicity in either legacy factory (work split via RTAs).

## Dropped Plumbing
- **Buffer-address RTAs → TensorBinding**: SDPA reader RTA[0-6] (q/k/v/mask/page_table/attention_sink/
  chunk_start_idx buffers); SDPA writer RTA[0] (out), RTA[10] (cu_window); Joint reader RTA[0-5]
  (6 buffers); Joint writer RTA[0-1] (out, joint_out). All were `Buffer*`-binding pushes.
- **Magic-number CB indices → DFBBinding**: SDPA reader/writer/compute CB-id CTA blocks
  (`sdpa_interleaved_cb_ids.hpp` `*_compile_time_args()`); Joint kernels' hardcoded `tt::CBIndex::c_*`.
- **`TensorAccessorArgs` plumbing → binding**: all `TensorAccessorArgs<N>()` chains in the kernels and
  the `TensorAccessorArgs(buf).append_to(...)` host emissions.
- **Semaphore-ID CTAs → SemaphoreBinding**: SDPA reader CTAs 29/30/31 (sender/receiver/valid ids).
- **Page-size 3rd-arg**: `dataflow_common.hpp:83` page-table `TensorAccessor` 3rd arg (dropped when
  inlined into the reader fork).
- **Positional CTAs → named CTAs**: every kernel's `get_compile_time_arg_val(N)` block.
- **Dead host-only RTAs dropped**: SDPA reader chain `q_chunk_start`/`q_chunk_count` (kernel `argidx+=2`
  skip) — not emitted / not read.

## Applied Patterns
- **Conditional / optional DFB, tensor & semaphore bindings** (`#ifdef` at preprocessor): SDPA
  `mask_in`, `page_table`, `attention_sink`, `chunk_start_idx_*`, `recip_scratch`, `cu_window_seqlens`,
  and the KV-chain `sender/receiver/valid` semaphores + chain RTAs (gated on `!is_causal`); Joint
  `mask_in` (`use_joint_mask`). **Named RTAs are gated too** (the chain-metadata RTAs) — same `#ifdef`.
- **Self-loop DFB binding**: SDPA compute intermediates (qk_im, out_im_A/B, max_A/B, sum_A/B,
  exp_max_diff) + recip_scratch; reader-only page_table; writer-only cu_window_seqlens. Joint compute
  intermediates (c_24..c_31).
- **Pass DFB handles directly to LLKs / kernel-lib** (`dfb::name → uint32_t`): all
  `dataflow_common.hpp` / `compute_common.hpp` / `compute_streaming.hpp` / `reduce_helpers_dataflow.hpp`
  call sites.
- **Porting a shared kernel (rung 2, create fork)**: the three SDPA kernels.
- **`constexpr` metadata via free-function + token**: `get_tile_size(dfb::name)` where legacy declared
  the value `constexpr`.
- **mask_in producer flips by config** (SDPA): PRODUCER binding = reader when `use_provided_mask`, else
  writer; CONSUMER = compute. 1P+1C either way.

## Deferred / Flagged
- **In-directory shared helper headers retain `CircularBuffer` and one `TensorAccessorArgs`**
  (`dataflow_common.hpp`, `compute_common.hpp`, `compute_streaming.hpp`). These are shared with the
  blocked ops and are boundary-bridged, so the anti-pattern self-audit's "no CircularBuffer in op dir"
  grep will legitimately hit them. Documented; not converted. (Report → Open items.)
- KV-chain forwarding host topology (chain building, mcast eligibility) is pure host RTA computation —
  carried over verbatim; its per-core outputs become named RTAs.

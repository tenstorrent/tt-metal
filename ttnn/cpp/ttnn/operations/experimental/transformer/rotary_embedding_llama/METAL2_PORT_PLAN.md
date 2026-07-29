# Port Plan — rotary_embedding_llama

Port plan for `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama`,
ported from the `ProgramDescriptor` (`descriptor`) concept to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

Porting unit: one `RotaryEmbeddingLlamaDeviceOperation` with **three** `descriptor` factories that
share kernel sources — ported **together** (per the brief). The three factories are the members of
`program_factory_t`; each independently flips from `create_descriptor`→`ProgramDescriptor` to
`create_program_artifacts`→`ProgramArtifacts`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (each factory has `static ProgramDescriptor create_descriptor(...)`).
- Variants: three factories in the device-op `program_factory_t` variant, chosen by `select_program_factory`:
  - `RotaryEmbeddingLlamaMultiCore` — interleaved prefill (reader + writer + compute).
  - `RotaryEmbeddingLlamaMultiCorePrefillSharded` — prefill, sharded cos/sin/trans_mat (reader + writer + compute).
  - `RotaryEmbeddingLlamaMultiCoreSharded` — decode, fully sharded (compute only).
- Custom `compute_program_hash`: **none** — already default reflection-based hash (audit confirmed; grep clean).

*(Target concept `MetalV2FactoryConcept` chosen by the audit; carried forward below.)*

### Kernels

#### Factory 1 — `RotaryEmbeddingLlamaMultiCore` (interleaved prefill)
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `dataflow/reader_rotary_embedding_llama_interleaved_start_id.cpp` | all_cores | [0]input_cb, [1]cos_cb, [2]sin_cb, [3]trans_mat_cb, [4]n_heads, [5]Ht(seq_len_t), [6]Wt(head_dim_t), [7]freq_per_head, [8]cos_Ht, [9]sin_Ht, [10]rotary_Ht, [11+]TensorAccessorArgs(input,cos,sin,trans_mat) | src_addr(Buffer\*), cos_addr, sin_addr, trans_mat_addr, batch_start, batch_end, seq_t_start, seq_t_end | RELOAD_IMPL | ReaderConfigDescriptor{} |
| writer | `dataflow/writer_rotary_embedding_llama_interleaved_start_id.cpp` | all_cores | [0]out_cb(c_16), [1]zero_cb(c_27), [2]n_heads, [3]Wt, [4]Ht, [5]rotary_Ht, [6+]TensorAccessorArgs(dst) | dst_addr(Buffer\*), batch_start, batch_end, seq_t_start, seq_t_end | RELOAD_IMPL (unused) | WriterConfigDescriptor{} |
| compute | `compute/rotary_embedding_llama.cpp` | all_cores | [0]in_cb, [1]cos_cb, [2]sin_cb, [3]trans_mat_cb, [4]rotated_interm(c_24), [5]cos_interm(c_25), [6]sin_interm(c_26), [7]out_cb(c_16), [8]Wt, [9]n_heads, [10]rotary_Ht | batch_start, batch_end, seq_t_start, seq_t_end | RELOAD_IMPL | ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en} |

#### Factory 2 — `RotaryEmbeddingLlamaMultiCorePrefillSharded` (prefill, sharded cos/sin/trans_mat)
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `dataflow/reader_rotary_embedding_llama_prefill_sharded.cpp` | all_cores | [0]input_cb, [1]cos_cb, [2]sin_cb, [3]trans_mat_cb, [4]n_heads, [5]Ht, [6]Wt, [7]freq_per_head, [8]trans_mat_use_global_cb, [9]cos_sin_sharded, [10]cos_Ht, [11]sin_Ht, [12]rotary_Ht, [13+]TensorAccessorArgs(input,cos,sin,trans_mat) | src_addr, cos_addr, sin_addr, trans_mat_addr, batch_start, batch_end, seq_t_start, seq_t_end | RELOAD_IMPL, COS_SIN_SHARDED_RELOAD | ReaderConfigDescriptor{} |
| writer | `dataflow/writer_rotary_embedding_llama_interleaved_start_id.cpp` (**shared with factory 1**) | all_cores | same as factory-1 writer | same as factory-1 writer | RELOAD_IMPL, COS_SIN_SHARDED_RELOAD (unused) | WriterConfigDescriptor{} |
| compute | `compute/rotary_embedding_llama.cpp` (**shared with factory 1**) | all_cores | same as factory-1 compute | same | RELOAD_IMPL, COS_SIN_SHARDED_RELOAD (unused by compute) | ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en} |

#### Factory 3 — `RotaryEmbeddingLlamaMultiCoreSharded` (decode, compute-only)
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| compute | `compute/rotary_embedding_llama_sharded.cpp` | shard grid bbox (all_cores) | [0]in_cb, [1]cos_cb, [2]sin_cb, [3]trans_mat_cb, [4]rotated_interm(c_24), [5]cos_interm(c_25), [6]sin_interm(c_26), [7]out_cb(c_16), [8]Wt(head_dim_t), [9]Ht(n_heads_t) | none | none | ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en} |

### CBs

All CBs use `.data_format` = the dtype's dataformat (all bfloat16 for io/interm; interm cos/sin match cos/sin format), `.page_size` = single tile size, `.tile` unset (standard 32×32). One `CBFormatDescriptor` each (no aliasing).

#### Factory 1
| index | total_size | core_ranges | borrowed? |
|---|---|---|---|
| c_0 input | input_cb_num_tiles × input_tile | all_cores | no |
| c_1 cos | num_cos_sin_tiles × cos_tile | all_cores | no |
| c_2 sin | num_cos_sin_tiles × sin_tile | all_cores | no |
| c_3 trans_mat | 1 × trans_mat_tile | all_cores | no |
| c_24 rotated_interm | head_dim_t × input_tile | all_cores | no |
| c_25 cos_interm | head_dim_t × cos_tile | all_cores | no |
| c_26 sin_interm | head_dim_t × sin_tile | all_cores | no |
| c_16 output | 2·head_dim_t × output_tile | all_cores | no |
| c_27 zero | head_dim_t × output_tile | all_cores | no |

#### Factory 2 (config-dependent — see the merged-CB note in Flags)
Same nine CBs. Config-dependent backing:
- c_1 cos / c_2 sin: `.buffer = cos_buffer/sin_buffer` (borrowed) on the sharded **fast path** (`cos_sin_sharded && !reload`), over `cos_sin_cb_cores` (shard grid if partial, else all_cores); a second same-`buffer_index` descriptor (plain, 1 tile) covers `remaining_cores` when the shard grid is partial. Plain (non-borrowed) on the reload path and the interleaved path.
- c_3 trans_mat: `.buffer = trans_mat_buffer` (borrowed) on the global-CB path (`trans_mat_use_global_cb`), over `tm_cb_cores` (+ remaining plain when partial); plain otherwise.
- All other CBs plain, all_cores.

#### Factory 3 (decode)
| index | total_size | core_ranges | borrowed? |
|---|---|---|---|
| c_0 input | num_input_tiles × input_tile | all_cores | **`.buffer = src_buffer`** |
| c_1 cos | num_cos_sin_tiles × cos_tile | all_cores | **`.buffer = cos_buffer`** |
| c_2 sin | num_cos_sin_tiles × sin_tile | all_cores | **`.buffer = sin_buffer`** |
| c_3 trans_mat | 1 × trans_mat_tile | all_cores | **`.buffer = trans_mat_buffer`** |
| c_24 rotated_interm | head_dim_t × input_tile | all_cores | no |
| c_25 cos_interm | head_dim_t × cos_tile | all_cores | no |
| c_26 sin_interm | head_dim_t × sin_tile | all_cores | no |
| c_16 output | num_output_tiles × output_tile | all_cores | **`.buffer = dst_buffer`** |

### Semaphores
none (all three factories — grep clean).

### Tensor accessors
| factory | host site | originating Tensor | RTA slot (addr) |
|---|---|---|---|
| 1 | reader s0/s1/s2/s3 | input, cos, sin, trans_mat | RTA 0/1/2/3 |
| 1 | writer s | output | RTA 0 |
| 2 | reader s0 (always); s1/s2 (reload or interleaved); s3 (non-global-cb) | input; cos/sin; trans_mat | RTA 0; 1/2; 3 |
| 2 | writer s | output | RTA 0 |
| 3 | none (all borrowed) | — | — |

All Case 1 (`TensorAccessor`); no Case 2 raw-pointer bindings; no 3rd (page-size) argument at any site.

### Work split
Not `split_work_to_cores`. Custom batch×seq parallelization (factories 1 & 2, identical code):
- `num_cores = grid_x*grid_y`; `batch_parallel_factor = min(batch, num_cores)`; `seq_parallel_factor = min(num_cores/batch_parallel_factor, seq_len_t)`.
- `cores = grid_to_cores(num_cores, grid_x, grid_y, row_major=true)`; core_idx = `batch_parallel*seq_parallel_factor + seq_parallel`.
- Cores whose `start_seq>=seq_len_t` or `start_batch>=batch` are **idle** (loops don't execute).
- Factory 3: placement = `shard_spec->grid.bounding_box()`; one program per core, no split loop.

### Shared kernels
Intra-op shared sources (bound by more than one of this op's factories); all convert **in this same change**, so no `_metal2` fork is created (rung 3 — invoker assigned the whole three-factory unit):
- `dataflow/writer_rotary_embedding_llama_interleaved_start_id.cpp` — factories 1 & 2.
- `compute/rotary_embedding_llama.cpp` — factories 1 & 2.

No cross-op (borrowed/lent) sharing: `grep -rl <filename> ttnn/cpp/ttnn/operations/` returns only this op's factories for every kernel (brief confirms; verified). All five kernels owned by this op; all `#include`s resolve to `api/*` (tt_metal LLK/HAL).

### Flags
- **Merged CBs (factory 2)**: legacy emits multiple `CBDescriptor`s sharing one `buffer_index` over disjoint core ranges (shard-grid borrowed + remaining plain) for c_1/c_2 and c_3. Metal 2.0 has no per-node borrowed/plain split for one DFB — see the placement decision under *Planned Spec Shape → Factory 2*.
- **Stale comment** `multi_core_program_factory.cpp:295` (`CoreArgs` comment lists a 5th `active` field that doesn't exist) — cosmetic, out of scope, routed to report.
- **Redundant `matmul_init`** in `compute/rotary_embedding_llama_sharded.cpp:47` (pre-loop + per-iteration) — harmless, out of scope, routed to report.
- Unreferenced kernel files: none.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `MetalV2FactoryConcept` (no op-owned tensors).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: three sibling factories flip independently; each returns `ttnn::device_operation::ProgramArtifacts` from `create_program_artifacts`. The device-op class (`select_program_factory`, `validate_*`, `compute_output_specs`, `create_output_tensors`) is unchanged except that the three factory `.hpp`/`.cpp` method signatures change. No pybind edits (only the plain op function is bound; `create_descriptor` is not pybound). No custom-hash deletion.

## Planned Spec Shape

Default 1:1 with legacy. Named-constant vocabulary (shared across all factories so the shared
writer/compute kernels bind consistently):

- DFB names: `INPUT`(c_0), `COS`(c_1), `SIN`(c_2), `TRANS_MAT`(c_3), `ROTATED_INTERM`(c_24),
  `COS_INTERM`(c_25), `SIN_INTERM`(c_26), `OUT`(c_16), `ZERO`(c_27).
- Kernel-side accessor names: `input`, `cos`, `sin`, `trans_mat`, `rotated_interm`, `cos_interm`,
  `sin_interm`, `out`, `zero`.
- Tensor params: `INPUT`, `COS`, `SIN`, `TRANS_MAT`, `OUTPUT`.
- Kernel names: `READER`, `WRITER`, `COMPUTE`.

### Factory 1 (interleaved) — placement all_cores, one WorkUnitSpec
- KernelSpecs: reader (DM), writer (DM), compute.
- DataflowBufferSpecs: 9, all plain.
- DFB bindings:
  - reader: PRODUCER of INPUT/COS/SIN/TRANS_MAT; TensorBinding input/cos/sin/trans_mat.
  - compute: CONSUMER of INPUT/COS/SIN/TRANS_MAT; PRODUCER of OUT; self-loop (PRODUCER+CONSUMER) of ROTATED_INTERM/COS_INTERM/SIN_INTERM.
  - writer: CONSUMER of OUT; self-loop of ZERO; TensorBinding output.
- TensorParameters: INPUT, COS, SIN, TRANS_MAT (reader bindings), OUTPUT (writer binding).
- WorkUnitSpec: one, {READER, WRITER, COMPUTE} over all_cores.
- Idle cores: kept in placement (all_cores); per-node RTAs zero-filled (batch_start=batch_end=seq_t_start=seq_t_end=0) exactly as legacy, so their loops don't execute. **Exact legacy placement match.**

### Factory 2 (prefill sharded) — placement **all_cores** (faithful), one WorkUnitSpec
- KernelSpecs: reader (prefill_sharded source), writer (shared), compute (shared).
- DataflowBufferSpecs: 9. cos/sin/trans_mat `borrowed_from` set **only on the full-shard fast path** else plain; rest plain.
- DFB bindings: same shape as factory 1 (reader produces INPUT/COS/SIN/TRANS_MAT unconditionally — the prefill_sharded reader always `reserve_back`/`push_back`s them even on the borrowed fast path; compute consumes; writer/interm/zero identical).
- TensorParameters: INPUT (reader binding, always), OUTPUT (writer binding, always), COS/SIN (borrowed_from **or** reader binding, per config), TRANS_MAT (borrowed_from **or** reader binding, per config).
- **Borrow-eligibility decision (load-bearing)**: a borrowed DFB (globally-allocated L1 view of a resident shard) is only expressible in Metal 2.0 when the shard grid covers **all** cores. Metal 2.0 derives DFB placement from the union of its bound kernels' work-unit nodes and offers **no** per-node borrowed/plain split for a single DFB (the legacy merged-CB idiom: borrowed on the shard grid + a plain placeholder on the remaining cores). So we borrow cos/sin/trans_mat **only** when `shard_spec()->grid.num_cores() == num_cores` (full shard); a **partial** shard falls back to the reload / TensorAccessor path, which is layout-agnostic and runs on all_cores exactly as the interleaved path does. Placement therefore stays **all_cores** — the faithful legacy placement (idle cores kept, per-node RTAs zero-filled) — because a borrowed DFB now only ever lands where every node has its own shard to back it.
  - *Faithfulness*: the full-shard fast path is a pure syntax swap (borrowed-over-all_cores == legacy borrowed CB over all_cores, no placeholder split). The **only** observable deviation from legacy is that a *partial*-shard config that legacy would have served from the fast L1 view now takes the (output-identical, slower) reload path. This matches the op's own test contract, whose docstring states an `N`-core shard `-> tests TensorAccessor reload path with fewer shards` while `-1` (all cores) `-> tests globally-allocated CB fast path`. Flagged in the report.

### Factory 3 (decode) — placement shard-grid bbox (all_cores), one WorkUnitSpec
- KernelSpecs: compute only.
- DataflowBufferSpecs: 9. INPUT/COS/SIN/TRANS_MAT/OUT `borrowed_from` the respective tensors; ROTATED_INTERM/COS_INTERM/SIN_INTERM plain.
- DFB bindings: compute self-loops **every** CB (PRODUCER+CONSUMER) — the lone compute kernel is the only toucher of all nine.
- TensorParameters: INPUT/COS/SIN/TRANS_MAT/OUTPUT — each referenced via a DFB `borrowed_from` (no kernel TensorBinding; validator accepts `borrowed_from` as the reference — `program_spec.cpp:533-552`).
- WorkUnitSpec: one, {COMPUTE} over the shard-grid bbox.

## Preserved Multiplicity
none — no work-split multiplicity in legacy. Each factory has a single `KernelDescriptor` per source
(the batch×seq parallelization varies only per-node RTAs, not CTAs), so one `KernelSpec` per source.
No two-`KernelDescriptor`-per-group construct anywhere.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader/writer RTA slots (addr) | `src/cos/sin/trans_mat/dst` `Buffer*` RTAs (`multi_core:337-338`, `prefill_sharded:472-475`) | `TensorBinding` (INPUT/COS/SIN/TRANS_MAT/OUTPUT) auto-injected base address |
| reader/writer CTAs (TensorAccessorArgs) | `TensorAccessorArgs(*buf).append_to(cta)` (`multi_core:228-231,240`; `prefill_sharded:365-368,377`) + kernel `TensorAccessorArgs<N>()` chains | binding mechanism end-to-end; kernel `TensorAccessor(tensor::name)` |
| reader/writer/compute CTAs (CB indices) | positional `cb` index CTAs | `DFBBinding` (dfb::name) |
| all kernels CTAs (scalars) | positional `get_compile_time_arg_val(N)` | named CTAs `get_arg(args::name)` |
| all kernels RTAs (scalars) | positional `get_arg_val<uint32_t>(N)` | named RTAs `get_arg(args::name)` |
| factory-2 reader CTA [8] trans_mat_use_global_cb | positional bool CTA gating `if constexpr` | `#define TRANS_MAT_USE_GLOBAL_CB` (conditional binding — promoted CTA→define) |
| factory-2 reader CTA [9] cos_sin_sharded | positional bool CTA gating `if constexpr` | `#define COS_SIN_SHARDED` (conditional binding — promoted CTA→define) |
| reader/writer `get_tile_size(cb_id)` | free helper on CB id | `dfb.get_entry_size()` member (DM-safe; entry_size == legacy page_size) |

No page-size 3rd-argument CTAs/RTAs (no accessor passes a 3rd arg). No semaphore-ID RTAs (no semaphores).

## Applied Patterns
- [Self-loop DFB binding](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding): ROTATED_INTERM/COS_INTERM/SIN_INTERM on compute (factories 1&2); ZERO on writer (factories 1&2); **every** CB on the lone compute kernel (factory 3).
- [Sync-free / single-ended → self-loop](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb): ZERO (writer fills + reads via base pointer, single toucher).
- Borrowed-memory DFBs (migration guide — DataflowBufferSpec): factory 3 INPUT/COS/SIN/TRANS_MAT/OUT; factory 2 COS/SIN (fast path) and TRANS_MAT (global-cb) per config.
- [Conditional / optional DFB & tensor bindings](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings): factory-2 reader `tensor::cos`/`tensor::sin` (bound iff `!cos_sin_sharded || cos_sin_sharded_reload`) and `tensor::trans_mat` (bound iff `!trans_mat_use_global_cb`), with the two `if constexpr` gates promoted to `#ifdef COS_SIN_SHARDED` / `#ifdef TRANS_MAT_USE_GLOBAL_CB`.
- Multi-factory op (three sibling `create_program_artifacts`, one per `program_factory_t` member).
- hw_config: **Style B** — build `ComputeGen1Config` directly (`fpu_math_fidelity`, `enable_32_bit_dest` only; rest default), mirroring the legacy `ComputeConfigDescriptor{.math_fidelity, .fp32_dest_acc_en}`. Do **not** route through `to_compute_hardware_config`, which would translate the resolved `math_approx_mode=true` (default) into `sfpu_precision_mode=Approximate` — a value the legacy descriptor discarded (used its own default `false` → Precise). DM configs: `create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)` (legacy Reader/WriterConfigDescriptor defaults). No `unpack_modes` (all DFBs bfloat16; the FP32-required rule never fires even when `enable_32_bit_dest` is true).

## Deferred / Flagged
- **Factory-2 borrow restricted to full-shard**: see the load-bearing decision above. Borrow only when the shard grid covers all cores (faithful all_cores placement, exact syntax swap); partial shards take the reload path. The one place the legacy op has no 1:1 Metal 2.0 shape; the deviation is a partial-shard config taking the (output-identical) reload path instead of the fast L1 view, which matches the op's test contract. Flagged for reviewer/patterns-catalog attention.
- **No borrowed DFB is ever placed over a subset of its backing shard** — borrowing is gated on the shard covering all cores, so every placed node has its own shard. (An earlier revision narrowed placement to active cores to borrow over a partial shard; reverted as a non-syntax-swap behavior change + latent runtime-hang risk.)
- No new blocking findings; audit gate set holds.

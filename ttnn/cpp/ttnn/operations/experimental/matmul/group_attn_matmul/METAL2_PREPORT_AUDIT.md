# Metal 2.0 Audit Findings — `experimental/matmul/group_attn_matmul`

- **`GroupAttnMatmulDeviceOperation`**
  - `GroupAttnMatmulProgramFactory` (`device/group_attn_matmul_program_factory.cpp`)

Single DeviceOperation, single ProgramFactory — no bundling. Kernels referenced by the factory (all in-directory, all Device-2.0 object API):
- reader — `device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp`
- writer — `device/kernels/dataflow/writer_transformer_group_attn_matmul.cpp`
- compute — `device/kernels/compute/transformer_group_attn_matmul.cpp`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `b6dadc46ee0 2026-07-21 docs: fix metal_2.0 doc links (READ_ME_FIRST rename + two stragglers)`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `GroupAttnMatmulDeviceOperation` → `GroupAttnMatmulProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 3 kernels on the object API (`Noc`, `CircularBuffer`, `Semaphore<>`, `TensorAccessor`) |
| *Prereqs* — Cross-op escapes | Ok — no out-of-directory includes; no borrowed kernels |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok — all CTAs read at constant indices; fixed 2-tensor `tensor_args_t` |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (no `->address()` fold; buffers ride `Buffer*` bindings) |
| *Port work* — Tensor bindings (per binding) | `input_tensor_a` / `input_tensor_b` / `output` — Case 1 (interleaved) · clean borrowed-DFB (sharded) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — all 3 accessor sites are 2-arg |
| *Port work* — CB endpoints | all legal 1:1, except `c_2` self-loop (IN1_SHARDED only) |

**CB endpoints** are dispositions, not gates. Every CB here is a legal 1P+1C FIFO except `c_2` (the in1 sharded borrowed CB), which is single-toucher → self-loop. Classified per `(CB, config)` below.

## Result

**GREEN → brief issued.** All five gates clear: Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓. `group_attn_matmul` is a single-factory `descriptor`-concept op, already fully migrated to the Device 2.0 object kernel API, with clean tensor delivery (framework `Buffer*` bindings feeding `TensorAccessor`s, plus borrowed-memory CBs for the sharded configs). Target concept `MetalV2FactoryConcept`. No portable-subset scoping needed — the whole op clears.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness sheet row (one factory) reads `Is able to port? == yes`. Derivation all clear: `Concept == descriptor`, `Custom hash == no`, `Runtime-args update == no`, `Pybind descriptor == no`, `Is safe to port? == yes` (`Smuggled pointer == no`). Cross-check against code agrees on every cheaply-checkable column:
  - `Concept` — `GroupAttnMatmulProgramFactory::create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`group_attn_matmul_program_factory.hpp:21`). ✓ `descriptor`.
  - `Custom hash` — no `compute_program_hash` override anywhere in the op (device-op declares only `validate_on_program_cache_miss` / `compute_output_specs` / `create_output_tensors`). The factory-header and factory-body comments *reference* `compute_program_hash()` (`group_attn_matmul_program_factory.hpp:17`, `group_attn_matmul_program_factory.cpp:158`) but no override exists — the op relies on the default device-operation hash (which captures tensor specs / `padded_shape`). Sheet `no` confirmed. (See Misc anomalies — the comment wording is slightly misleading.)
  - `Runtime-args update` — no `get_dynamic_runtime_args` / `override_runtime_arguments`. ✓ `no`.
  - `Pybind descriptor` — `group_attn_matmul_nanobind.cpp` binds only the op function `group_attn_matmul`; no `create_descriptor` binding. ✓ `no`.
  - Cross-column invariants hold: `Op-owned tensors?` blank on a `descriptor` row (consistent — `descriptor` can't carry op-owned tensors); `Runtime-args update == no`.

- **Device 2.0 (every kernel used):** **GREEN.** All three kernels are written against the current Device 2.0 object API — the exact surface prescribed by `device_api_migration_guide.md` (`api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/dataflow/noc_semaphore.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h`):
  - **NoC** via the `Noc` object — `noc.async_read`, `noc.async_write`, `noc.async_write_multicast`, `noc.async_read_barrier`, `noc.async_write_barrier`, `noc.async_writes_flushed`. No raw `noc_async_read/write`.
  - **CBs** via `CircularBuffer` wrappers — `cb_*.reserve_back / push_back / wait_front / pop_front / get_write_ptr / get_read_ptr`. No free-function CB management.
  - **Semaphores** via `Semaphore<>` objects — `.set / .wait / .up / .set_multicast`. No raw sem addresses.
  - **Addr-gen** via `TensorAccessor` — no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen*`.
  - Only CB-index free functions present: `get_tile_size(cb_id)` (**sanctioned** per the Green bullet) and the compute LLK entry points (`matmul_tiles`, `tilize_block`, `pack_untilize_*`, `reconfig_data_format_*`), which are the normal compute-kernel surface, not data-movement holdovers.
  - Grep for legacy DM idioms (`noc_async_read`, `InterleavedAddrGen`, `get_read_ptr(`/`get_write_ptr(` free-fn, `cb_wait_front`, `noc_semaphore_*`, `get_noc_addr_from_bank_id`) across all three kernels: **zero hits.**
  - No donor kernels — every `#include` resolves under `api/*` (tt_metal LLK/HAL/firmware, donor class 1: no concern). No cross-op or in-family kernel includes.

- **Feature compatibility:** all Appendix A entries scanned; none fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateCircularBuffer(..., global_cb)`, no `remote_cb`/`.remote_index`, no `.global_circular_buffer` field on any `CBDescriptor`. CBs are plain `CBDescriptor`s (some with `.buffer` set = ordinary borrowed-memory, a mechanical port step). |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set on any `CBDescriptor`; no `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type / `CreateGlobalSemaphore` / `global_semaphore.hpp`. Semaphores are plain `SemaphoreDescriptor` (IDs 0/1, `group_attn_matmul_program_factory.cpp:142,148`) → `SemaphoreSpec`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed pair (`input_tensor_a`, `input_tensor_b`) + optional preallocated output — no `std::vector<Tensor>`. Kernels read all `get_compile_time_arg_val` at **constant** indices (0..3), plus `TensorAccessorArgs<N>` at constexpr offsets. No runtime-varying CTA index loop. |

- **CB endpoints (GATE-free):** census per `(CB, config)`, per node. All kernels bound on `all_device_cores`, so every node hosts reader + writer + compute. `get_tile_size(cb_id)` reads are metadata, **not** endpoints.

  | CB | Config | Touchers (kernel · role) | Verdict | Disposition |
  |---|---|---|---|---|
  | `c_0` in0 | interleaved | writer (produce) · compute (consume) | 1 locked P + 1 locked C | 1:1 legal |
  | `c_0` in0 | IN0_SHARDED | writer (produce, sync-only) · compute (consume) | 1 locked P + 1 locked C | 1:1 legal; borrowed backing → `borrowed_from(input_tensor_a)` |
  | `c_1` in1 | both | reader (produce) · compute (consume) | 1 locked P + 1 locked C | 1:1 legal (writer only reads its tile size — not a toucher) |
  | `c_2` in1-sharded | IN1_SHARDED only | reader (raw `get_read_ptr`, role-free) | 1 toucher | **self-loop**; borrowed backing → `borrowed_from(input_tensor_b)` |
  | `c_3` intermed0 | both | compute (produce) · writer (consume) | 1 locked P + 1 locked C | 1:1 legal |
  | `c_4` intermed1 | both | writer (produce) · compute (consume) | 1 locked P + 1 locked C | 1:1 legal |
  | `c_5` out | interleaved | compute (produce) · writer (consume) | 1 locked P + 1 locked C | 1:1 legal |
  | `c_5` out | OUT_SHARDED | compute (produce) · writer (consume via `wait_front`) | 1 locked P + 1 locked C | 1:1 legal; borrowed backing → `borrowed_from(output)` |

  No dead CBs, no multi-binding, no hidden second writer. The mcast into `c_1` is issued by the (single) reader source against its own producer binding + a NoC destination — it does not add a second local binding, so `c_1` stays 1P+1C. Nothing here blocks a Gen1 port.

- **Offset base pointers:** **GREEN — no fold.** Grep for `->address()` / `.address()` across the whole op: zero hits. Buffers reach the kernels as `Buffer*` runtime args (`src0_buffer`, `src1_buffer`, `dst_buffer` — the framework `BufferBinding` shape, patched on cache hit), never as a host-folded `base + offset`. All offsets (`in1_start_id`, `num_blocks_written * MtKt`, `num_blocks_written * MtNt`) are passed as **separate scalar** RTAs and added in-kernel. Not in the offset-base triage tables — confirmed clean by scan. No Type 1/2/3/4.

- **TensorAccessor 3rd argument:** **GREEN — none.** All three `TensorAccessor` construction sites are 2-arg (`args, base_addr`):
  - `reader_mcast_transformer_group_attn_matmul.cpp:83` — `TensorAccessor(in1_args, src1_addr)`
  - `writer_transformer_group_attn_matmul.cpp:65` — `TensorAccessor(in0_args, src0_addr)`
  - `writer_transformer_group_attn_matmul.cpp:71` — `TensorAccessor(out_args, dst_addr)`

  No page-size override anywhere; not in the 3rd-arg triage table — confirmed clean by scan.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — three `TensorParameter`s, each with a per-config split:
  - `input_tensor_a` (in0, `src0_buffer`) — **Case 1** via `TensorAccessor` in the writer (interleaved) · **clean** borrowed-DFB read from `c_0` (IN0_SHARDED).
  - `input_tensor_b` (in1, `src1_buffer`) — **Case 1** via `TensorAccessor` in the reader (interleaved) · **clean** borrowed-DFB read from `c_2` (IN1_SHARDED).
  - `output` (`dst_buffer`) — **Case 1** via `TensorAccessor` in the writer (interleaved) · **clean** borrowed-DFB (`c_5`, OUT_SHARDED).
  - All three are delivered today as framework `Buffer*` bindings (correct-on-cache-hit, not the silent-wrong hazard); the port replaces each with a typed `TensorParameter`. No Case 2 (no raw-pointer arithmetic bridge needed).
- **TensorParameter relaxation:** none (sheet `none`; no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_2` (IN1_SHARDED); all other CBs legal 1:1. Sharded backings port via `DataflowBufferSpec::borrowed_from` (`c_0`, `c_2`, `c_5`).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no multi-binding, no hidden second writer. (`c_2` self-loop and the borrowed-memory backings are in Port-work.)
- **Cross-op / shared kernels:** none — all three kernels are op-owned and in-directory; no borrowed kernel files, no port-together set.
- **RTA varargs:** reader consumes a variable-length RTA tail — `in1_mcast_sender_noc_x[num_x]` then `in1_mcast_sender_noc_y[num_y]`, read by pointer via `get_arg_addr(i)` with the cursor advanced by a runtime count (`reader_mcast_transformer_group_attn_matmul.cpp:61-64`; appended host-side at `group_attn_matmul_program_factory.cpp:396-399`). No per-element names exist → port as **RTA varargs** (supported; kernel-side vararg mechanism), don't try to name each. Count is fixed per compiled program (mcast sender grid) but read as a runtime-length block.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.** No kernel `#include` resolves outside the op directory (every include is `api/*` = tt_metal LLK/HAL/firmware, donor class 1). No file-path kernel instantiation of another op's `.cpp` — all three kernels are op-owned. No port-together coupling.
- **Relaxation candidates (mined from a custom hash):** N/A — no custom hash.
- **TTNN factory analysis:** current concept `descriptor`; op-owned tensors none; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept`. Semaphores are program-scope `SemaphoreDescriptor` (IDs 0/1 over `all_device_cores`) → `SemaphoreSpec`.

## Misc anomalies  *(team-only, non-gating)*

- **Misleading `compute_program_hash` comments.** `group_attn_matmul_program_factory.hpp:15-24` and `group_attn_matmul_program_factory.cpp:57-59,155-158` describe shape-dependent CB sizing being "folded into `compute_program_hash()` via `padded_shape`," implying a custom hash. There is **no** `compute_program_hash` override — the op relies on the default device-operation hash (which already keys on tensor specs / `padded_shape`). The behavior is correct; only the comment wording (referring to a hook that doesn't exist) is misleading. Route to the ops team for a comment cleanup; the port does not act on it.
- **Unused `onetile` constant.** `constexpr uint32_t onetile = 1;` appears in the writer (`writer_transformer_group_attn_matmul.cpp:57`) and compute (`transformer_group_attn_matmul.cpp:60`) kernels and is never referenced. Dead constant; harmless.

## Recipe notes

- **Analyses-doc path.** The audit recipe's links to `../../analyses/*` resolve to `metal_2.0/analyses/` (i.e. `ai/`'s sibling), not `metal_2.0/ai/analyses/`. This is correct once computed, but easy to mis-guess on a first read; a one-line note in the recipe ("the analyses/ folder is a sibling of ai/, not under it") would save a step. Minor.
- **Object-NoC namespace.** The recipe's Device 2.0 examples name `experimental::Noc`, but the current migrated surface (per `device_api_migration_guide.md`) is the un-namespaced `Noc` object from `api/dataflow/noc.h`. The recipe's Green bullet keys on the *idiom* (object vs. raw) so this didn't cause a mis-call, but a reader matching the literal `experimental::` token could be briefly thrown. Consider updating the recipe's example token to match the shipped `Noc`.

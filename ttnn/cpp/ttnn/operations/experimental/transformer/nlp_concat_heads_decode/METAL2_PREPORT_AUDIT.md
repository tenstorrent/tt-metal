# Metal 2.0 Audit Findings — `experimental/transformer/nlp_concat_heads_decode`

Single device-operation directory:

- **`NLPConcatHeadsDecodeDeviceOperation`**
  - `NLPConcatHeadsDecodeProgramFactory` (`nlp_concat_heads_decode_program_factory.cpp`) — full-grid path (`on_subcoregrids == false`)
  - `NLPConcatHeadsDecodeSubcoregridsProgramFactory` (`nlp_concat_heads_decode_subcoregrids_program_factory.cpp`) — `on_subcoregrids == true`

Both factories share the same DeviceOperation, the same one output CB, and the same reader/writer kernel structure (distinct kernel *files*, identical shape). Findings are identical across both factories except where noted; audited together as one porting unit.

Referenced kernels (in scope):
- `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode.cpp` (full-grid factory)
- `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp` (subcoregrids factory)

No unreferenced kernel files in the directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `NLPConcatHeadsDecodeDeviceOperation` → `NLPConcatHeadsDecodeProgramFactory`, `NLPConcatHeadsDecodeSubcoregridsProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — both kernels are structurally Device 2.0 (`Noc`, `CircularBuffer` wrapper, `UnicastEndpoint`, `CoreLocalMem`) |
| *Prereqs* — Cross-op escapes | Ok — no out-of-directory `#include`s (only `api/*` LLK/HAL); no borrowed kernel files |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (kernels read CTAs at constexpr offsets only) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factories) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes (both factories) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none (clean base + separate offset arg, already split out) |
| *Port work* — Tensor bindings (per binding) | `input` → Case 2 (raw pointer) · `output` → clean (borrowed-memory DFB) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | N/A — no `TensorAccessor` in either kernel |
| *Port work* — CB endpoints | `c_16` → **1P+1C** (dual-instance work-split), both factories |

**CB endpoints** are dispositions, not gates. The one CB (`c_16`, the output) is a dual-instance work-split (same-source reader+writer instances over the same core range, raw-writing disjoint tile phases) → assign 1P+1C. No flag, no dead CB.

## Result

**GREEN → brief issued.** All five gates clear:

- **Device 2.0** — both kernels are fully Device 2.0 (object-oriented `Noc`/`CircularBuffer`/`UnicastEndpoint`/`CoreLocalMem` idioms; no CB-index free-function holdovers, no Device-1.0 addr-gen).
- **Feature compatibility** — no Appendix A feature in use (all N/A).
- **TTNN factory concept** — both factory rows read `Is able to port? == yes` on the readiness sheet; code cross-check agrees on every cheaply-checkable column.
- **Offset base pointers** — no host-folded offset; the input base is delivered as a clean `Buffer*` and the head offset rides a *separate* RTA, added in the kernel (the already-split-out shape).
- **TensorAccessor 3rd argument** — N/A; the op uses hand-rolled NoC walks, no `TensorAccessor`.

Port work is light: one Case-2 tensor binding (input), one borrowed-memory output DFB, and a 1P+1C assignment on the single output CB. Ports to `MetalV2FactoryConcept`.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** for both factory rows. Sheet `Is able to port? == yes`. Derivation confirmed: `Is safe to port? == yes` AND `Custom hash == no` AND `Runtime-args update == no` (both the `get_dynamic_runtime_args` column and the `PD override_runtime_args` column) AND `Pybind descriptor == no` AND `Concept == descriptor`. Cross-check clean:
  - `Concept == descriptor` — confirmed: `create_descriptor()` returns `ProgramDescriptor` (`nlp_concat_heads_decode_program_factory.cpp:17`, `nlp_concat_heads_decode_subcoregrids_program_factory.cpp:18`).
  - `Custom hash == no` — confirmed: no `compute_program_hash` override on `NLPConcatHeadsDecodeDeviceOperation` (`nlp_concat_heads_decode_device_operation.hpp`/`.cpp`).
  - `Runtime-args update == no` — confirmed: no `get_dynamic_runtime_args` / `override_runtime_arguments` anywhere in the op.
  - `Pybind descriptor == no` — confirmed: `nlp_concat_heads_decode_nanobind.cpp` binds the high-level function `nlp_concat_heads_decode`, not a `create_descriptor` on the device op.
  - Cross-column invariants OK: `Op-owned tensors?` blank on a `descriptor` concept; `Runtime-args update == no` on a `descriptor` concept.
- **Device 2.0 (every kernel used):** **GREEN.** Both kernels use only Device-2.0 idioms:
  - `Noc noc;` + `noc.async_read(src_ep, CoreLocalMem<uint32_t>(...), size, {.noc_x, .noc_y, .addr}, {})` (`reader_tm_tile_layout_nlp_concat_heads_decode.cpp:13,57,67`; `..._subcoregrid.cpp:13,56,65`).
  - `CircularBuffer cb_q_out(cb_id_q_out);` + `cb_q_out.get_write_ptr()` — the wrapper *method*, not the CB-index free function (`reader...:34,49`; `..._subcoregrid:34,48`).
  - `UnicastEndpoint`, `CoreLocalMem<uint32_t>` — Device-2.0 endpoint/memory wrappers.
  - `get_arg_val` / `get_arg_addr` / `get_compile_time_arg_val` are standard host-arg accessors, not CB-index free functions. No `noc_async_read`, no `InterleavedAddrGen`/`ShardedAddrGen`, no raw sem addresses, no `get_write_ptr(cb_id)` free-function holdover. Nothing to route to the Device 2.0 team.
- **Feature compatibility:** all-N/A (clean scan). No entry's recognition signals fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `.global_circular_buffer` field, no `remote_index`/`remote_cb`/`global_cb`; the two CBs are plain `CBDescriptor`s |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset` set on either `CBDescriptor`; no `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | no semaphores of any kind in this op; no `GlobalSemaphore` type or `CreateGlobalSemaphore` |
  | Variable-count compile-time arguments (CTA varargs) | N/A | fixed-count input (single `Tensor` in `tensor_args_t`); kernels read CTAs only at constexpr offsets `get_compile_time_arg_val(0..9)` — no runtime-varying CTA index |

- **CB endpoints (GATE-free):** one CB — `c_16` (`q_output_cb_index`), the op's output, borrowed-memory (`.buffer = output.buffer()`), same in both factories. Census per node: the reader-config instance and the writer-config instance (same kernel source, both over `q_cores`) each raw-write disjoint tile phases via `cb_q_out.get_write_ptr()` (no FIFO ops); nothing on-device consumes it (it is the final output). **Two role-free touchers → 1P+1C** — bind one instance PRODUCER, the other CONSUMER (cosmetic on Gen1). No third toucher; the multi-binding flag is **not** needed. This is the reshard-style dual-instance work-split the recipe warns is commonly mis-slotted as multi-binding. Disposition holds across both configs.
  - `c_16` main factory: `nlp_concat_heads_decode_program_factory.cpp:45-55` (CB), `:93-112` (reader+writer descriptors, same source, `q_cores`, Reader/WriterConfigDescriptor).
  - `c_16` subcoregrids factory: `nlp_concat_heads_decode_subcoregrids_program_factory.cpp:55-65` (CB), `:100-119` (reader+writer descriptors).
- **Offset base pointers:** **GREEN** — no fold. Each factory pushes the head-row byte offset `in_tile_offset_by_batch` as its **own** RTA and the input base as a **separate** `Buffer*` (`in_buffer`), then the kernel adds them: `qkv_read_addr = q_start_addr + in_tile_offset_by_head`. Base and offset are already split — the clean shape, not a `base + offset` fold. Not listed in the offset-base-pointer triage doc (`analyses/2026-07-19_offset_base_pointers.md`); confirmed clean by scan (no-fold, not-in-tables → clean, handed to TensorParameter analysis).
  - Main: offset `nlp_concat_heads_decode_program_factory.cpp:121-129` computed, pushed `:129`; base `Buffer* in_buffer` pushed `:130`. Kernel add `reader_tm_tile_layout_nlp_concat_heads_decode.cpp:45`.
  - Subcoregrids: offset `nlp_concat_heads_decode_subcoregrids_program_factory.cpp:127-136`, pushed `:136`; base pushed `:137`. Kernel add `..._subcoregrid.cpp:44`.
- **TensorAccessor 3rd argument:** **N/A** — neither kernel constructs a `TensorAccessor` (memory access is hand-rolled `noc.async_read` for the input and `cb.get_write_ptr()` for the output). No 3rd-arg site to classify. Not in the 3rd-arg triage doc (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`); consistent.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, both factories):
  - `input` — **Case 2** (raw pointer). Delivered today via the **`Buffer*`-binding form** (`in_buffer` pushed into the RTA list → framework auto-registers a `BufferBinding`, patched on cache hits — the correct-on-cache-hit interim shape, *not* the silent-stale-pointer hazard). The kernel consumes the base raw: `q_start_addr = get_arg_val<uint32_t>(1)`, then `qkv_read_addr = q_start_addr + in_tile_offset_by_head` fed straight to `noc.async_read(.addr = ...)` with explicit `.noc_x/.noc_y` (a hand-rolled gather across input shard cores). **Port work:** express the input as a `TensorParameter` / `TensorBinding`; the kernel pulls the base via the sanctioned `TensorAccessor::get_bank_base_address` bridge and keeps its raw NoC arithmetic unchanged. Do **not** rewrite the raw walk into `TensorAccessor` iteration.
    - Host: `nlp_concat_heads_decode_program_factory.cpp:130`; `nlp_concat_heads_decode_subcoregrids_program_factory.cpp:137`.
    - Kernel: `reader_tm_tile_layout_nlp_concat_heads_decode.cpp:18,45,57,67`; `..._subcoregrid.cpp:16,44,56,65`.
  - `output` — **clean** (borrowed-memory DFB). The output CB `c_16` is created with `.buffer = output.buffer()` and the kernels write it via `cb_q_out.get_write_ptr()`. Causal-link gate → clean; port via `DataflowBufferSpec::borrowed_from`. (Its endpoint legality is the 1P+1C item above.)
    - `nlp_concat_heads_decode_program_factory.cpp:46-55` (`.buffer` at `:54`); `nlp_concat_heads_decode_subcoregrids_program_factory.cpp:56-65` (`.buffer` at `:64`).
  - Op-level roll-up: **⚠ port work** (one Case-2 binding).
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation == none`; no custom hash).
- **TensorAccessor 3rd arg:** none (no accessor).
- **CB endpoints:** `c_16` → **1P+1C** (both configs). All accounted; no self-loop, no multi-binding flag, no dead CB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none require the flag. The single output CB is a dual-instance work-split (visible; no hidden second writer, no semaphore-gated co-fill) → 1P+1C. Watch only that it is not mis-slotted as multi-binding.
- **Cross-op / shared kernels:** none. Both kernels are owned by this op; no file-path borrows, no out-of-directory `#include`s.
- **RTA varargs:** the two NoC-coordinate blocks (`noc_x_coords` / `noc_y_coords`) are **variable-count RTA blocks** — their count is CTA-bounded (`in_num_cores_x`/`in_num_cores_y` in the main factory; `in_num_cores` in the subcoregrids factory) and varies with the input shard grid across instantiations. The kernel reads them as L1 arrays via `get_arg_addr` (`reader_tm_tile_layout_nlp_concat_heads_decode.cpp:31-32`; `..._subcoregrid.cpp:31-32`), indexing `in0_mcast_noc_x[...]` / `in0_mcast_noc_y[...]`. **Port these as varargs** (RTA vararg mechanism), not as individually-named args. The two named scalars (offset at index 0, base at index 1) are ordinary named/binding args — do not let them ride the varargs.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.** No function-call escape (both kernels `#include` only `api/dataflow/*` and `api/core_local_mem.h` — `tt_metal/*` LLK/HAL, "No concern"). No file-path kernel instantiation of donor kernels — both factories `CreateKernel` only the op's own two kernel files. No port-together coupling.
- **Relaxation candidates (mined from a custom hash):** none — no custom hash to mine.
- **TTNN factory analysis:** sheet-derived facts, `file:line`-confirmed — `Concept == descriptor` (both factories); no op-owned tensors; no MeshWorkload; no pybind `create_descriptor`; no other risky pybind; no custom hash; no custom `override_runtime_arguments`. `Porting Classification == Simple Port`, `Model == llama`. Target concept: `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Dead/shadowed local `q_write_addr` in both kernels.** `uint32_t q_write_addr = 0;` is declared once (`reader_tm_tile_layout_nlp_concat_heads_decode.cpp:47`; `..._subcoregrid.cpp:46`) and then immediately shadowed by an inner-loop redeclaration `uint32_t q_write_addr = cb_write_ptr_base + wptr_offset;` (`reader...:53`; `..._subcoregrid:52`). The outer declaration is dead. Harmless; a cleanup for the ops team, not port work. (Do not "fix" this during the port — kernel behavior must not change.)

## Per-DeviceOperation attribution

Single DeviceOperation (`NLPConcatHeadsDecodeDeviceOperation`); both factory rows carry identical gate verdicts (`Is able to port? == yes`) and identical port-work findings. No per-factory divergence to attribute.

## Recipe notes

- **RTA-array-via-`get_arg_addr` vs. the RTA-varargs recognition signals.** The recipe's RTA varargs recognition (a) is framed around `get_arg_val<uint32_t>(arg_index++)` *loops*. This op instead takes the *address* of the RTA region (`get_arg_addr(2)`) and treats a CTA-bounded run of RTAs as an L1 `uint32_t*` array. It's the same "variable-count block of positional RTAs with no per-arg names" case and I classified it as an RTA vararg per the CTA-bounded clause, but the pointer-cast-into-L1-array shape isn't spelled out among the recognition examples — worth adding as an explicit sub-shape so a future auditor doesn't read the `get_arg_val` loop framing narrowly and miss it.

# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk`

- **`RotaryEmbeddingLlamaFusedQKDeviceOperation`** (`device/rotary_embedding_llama_fused_qk_device_operation.hpp:15`)
  - `RotaryEmbeddingLlamaFusedQKProgramFactory` (`device/rotary_embedding_llama_fused_qk_program_factory.cpp:18`, `create_descriptor`)

Single device-operation, single factory. The factory emits **one compute kernel only — the op has no dataflow (reader/writer) kernels at all**: every tensor is HEIGHT_SHARDED (enforced in `validate_on_program_cache_miss`, `device/rotary_embedding_llama_fused_qk_device_operation.cpp:30`) and reaches the kernel as a buffer-backed (borrowed-memory) CB. The kernel source is selected by the `row_major_QK` attribute (`device/rotary_embedding_llama_fused_qk_program_factory.cpp:237-242`):

- `device/kernels/compute/rotary_embedding_llama_sharded.cpp` — tiled QK path
- `device/kernels/compute/rotary_embedding_llama_sharded_row_major.cpp` — row-major (tile-wrapped) QK path

Both kernel files are **op-owned and bound only by this op's factory** (filename census run; the same-named `rotary_embedding_llama_sharded.cpp` in the sibling `rotary_embedding_llama` op is that op's own private copy — different path, and already Metal 2.0 there; see Heads-ups).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** version cannot be pinned — `git log -1 -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` prints nothing (the docs directory is untracked in this checkout, not from a tracked doc-branch).

**Readiness sheet:** fetched fresh 2026-08-27 (Diego's *"Operations analysis"* sheet, via the Google Drive connector).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `RotaryEmbeddingLlamaFusedQKDeviceOperation` → `RotaryEmbeddingLlamaFusedQKProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — both compute kernels use the kernel-side `CircularBuffer` wrapper for all FIFO ops; zero DM idioms (no NoC calls, no addr-gens, no raw sem addresses, no `get_read_ptr`/`get_write_ptr` free functions) |
| *Prereqs* — Cross-op escapes | **Ok** — kernel includes are `tt_metal` API headers only (class 1); no borrowed kernel files |
| *Feature Support* — overall | **GREEN** — all Appendix A entries N/A |
| *Feature Support* — Variadic-CTA | Ok — CTA set is fixed (13 entries, all read at constexpr indices) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (`descriptor` concept; sheet `Execution Model` = `SPMD`, `Secretly SPMD Workload?` blank) |
| *TTNN Readiness* — Custom hash | No (verified: no `compute_program_hash`, no `attribute_values`/`to_hash` backdoor in the device-op) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (verified by grep of the device-op) |
| *TTNN Readiness* — `override_runtime_arguments` | No (verified — the factory defines only `create_descriptor`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (verified: `rotary_embedding_llama_fused_qk_nanobind.cpp:17-45` binds only the public composite op via `ttnn::bind_function`) |
| *TTNN Readiness* — Op-owned tensors | No (sheet cell blank; `create_descriptor` returns a plain `ProgramDescriptor`, no `buffers` vector) |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (matches the sheet's `Porting Target`) |
| *Port work* — Offset base pointer | **none** — the op has zero address RTAs (the only RTA is the `is_q` flag); cleared |
| *Port work* — Tensor bindings (per binding) | **all 7 clean** (borrowed-memory via `CBDescriptor::buffer` → `DataflowBufferSpec::borrowed_from`) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **none — no accessor in the op passes a 3rd arg** (the op constructs no `TensorAccessor` at all; it has no dataflow kernels) |
| *Port work* — CB endpoints | **self-loop, all 10 CBs** (single toucher — the one compute kernel — per node, in both kernel variants) |

**CB endpoints** are dispositions, not gates: here every CB has exactly one toucher (the op's single compute kernel), so every CB self-loops — bind the compute kernel PRODUCER **and** CONSUMER. Legal on Gen1 for compute kernels. Per-`(CB, config)` inventory below; the census is identical in both kernel variants (config-invariant).

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, same directory). All five gates clear: Device 2.0 ✓, Feature compatibility ✓ (all N/A), TTNN factory concept ✓ (`Is able to port?` = `yes`, cross-check clean), Offset base pointers ✓ (no address RTAs exist), TensorAccessor 3rd arg ✓ (no accessors exist). This is an unusually small port surface: one factory, one compute kernel (two source variants), zero dataflow kernels, all tensor traffic via borrowed-memory CBs, a single named per-core RTA.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN — the sheet's `Is able to port?` == `yes`; cross-check clean. Sheet row (one row, matching the single factory in code — factory-set match 1:1):
  - `Concept` = `descriptor` — verified: `create_descriptor` returning `ProgramDescriptor` (`device/rotary_embedding_llama_fused_qk_program_factory.cpp:18`), `program_factory_t = std::variant<RotaryEmbeddingLlamaFusedQKProgramFactory>` (`device/rotary_embedding_llama_fused_qk_device_operation.hpp:20`).
  - `Custom hash (compute_program_hash)` = `no`, `Backdoor custom hash` = `no` — verified by grep.
  - `Runtime-args update (get_dynamic_runtime_args)` = `no` — verified by grep of the device-op.
  - `Override runtime args method?` = `no` — verified; target concept is therefore plain `ProgramSpecFactoryConcept`.
  - `Pybind descriptor` = `no` — verified against `rotary_embedding_llama_fused_qk_nanobind.cpp`.
  - `Smuggled pointer (raw buffer addr in RTA/CRTA)` = `no` — verified: no `->address()` anywhere in the op directory.
  - `Known op issues` = empty; `TensorParameter relaxation` = `none`; `Op Classification` = `PD Op (pointer-patching)`; `Is able to port?` = `yes`. No cross-column invariant violated. No sheet-vs-code disagreement.
- **Device 2.0 (every kernel used):** GREEN. The op uses exactly two kernels, both op-owned compute kernels; there are no donor kernels. Both are structurally Device 2.0:
  - All CB FIFO traffic goes through kernel-side `CircularBuffer` wrapper objects (`reserve_back`/`push_back`/`wait_front`/`pop_front` methods) — e.g. `device/kernels/compute/rotary_embedding_llama_sharded.cpp:59-66, 73-81`; include is `api/dataflow/circular_buffer.h` (both kernels, line 12).
  - Zero data-movement idioms: no `noc_*` calls, no `InterleavedAddrGen`/`ShardedAddrGen` family, no raw semaphore addresses, no `get_read_ptr(cb)`/`get_write_ptr(cb)`/`get_local_cb_interface` free functions anywhere.
  - The only `get_arg_val` is a control flag (`is_q`, both kernels line 29) — not an address.
  - LLK compute free functions taking `uint32_t` CB indices (`matmul_tiles`, `mul_tiles_bcast`, `mul_tiles`, `add_tiles`, `pack_tile`, the `*_init` family, `compute_kernel_hw_startup`) are the standard compute-API surface, not CB-index-keyed DM holdovers — no wrapper-method replacement exists for them, and they are not on the holdover cue's shape. No violations table needed.
- **Feature compatibility:** all Appendix A entries N/A (grep across the full op directory for each entry's recognition signals: `GlobalCircularBuffer`, `global_cb`, `remote_cb`/`remote_index`, `UpdateDynamicCircularBufferAddress`, `address_offset`, `GlobalSemaphore` — zero hits; the `CBDescriptor`s set only the plain `.buffer` field, the regular borrowed-memory path).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no signal fires; `.buffer = <Buffer*>` is the plain borrowed-memory path, `global_circular_buffer` field never set |
  | CBDescriptor `address_offset` (non-zero) | N/A | field never set anywhere in the op |
  | GlobalSemaphore | N/A | the op creates no semaphores at all |

- **CB endpoints (GATE-free):** every CB has **one toucher per node** — the single compute kernel (`KernelDescriptor` at `device/rotary_embedding_llama_fused_qk_program_factory.cpp:244-270`, placed on `work_cores = q_cores ∪ k_cores`). Disposition: **self-loop all 10**, in both configs (`row_major_QK` = false → `rotary_embedding_llama_sharded.cpp`; true → `..._row_major.cpp`; the census is identical). Per-CB inventory:

  | CB | Index | Backing | Kernel touch (both variants) | Census | Disposition |
  |---|---|---|---|---|---|
  | q_input | `c_0` | borrowed (`q_src_buffer`, factory:109) | FIFO produce **and** consume by the same kernel (`reserve_back`/`push_back`/`wait_front`/`pop_front`, sharded.cpp:79-81,115) | 1 toucher, both roles in one kernel | **self-loop** |
  | k_input | `c_1` | borrowed (`k_src_buffer`, factory:121) | same as q_input (runtime-selected via `is_q`) | 1 toucher | **self-loop** |
  | cos | `c_2` | borrowed (`cos_buffer`, factory:133) | LLK index reads only (`mul_tiles_bcast`/`mul_tiles` srcB); no FIFO ops, no raw pointers — role-free | 1 toucher | **self-loop** |
  | sin | `c_3` | borrowed (`sin_buffer`, factory:145) | LLK index reads only — role-free | 1 toucher | **self-loop** |
  | trans_mat | `c_4` | borrowed (`trans_mat_buffer`, factory:159) | LLK index reads only (`matmul_tiles` srcB, tile 0) — role-free | 1 toucher | **self-loop** |
  | rotated_input_interm | `c_24` | plain (factory:164-172) | FIFO produce + consume by the same kernel | 1 toucher | **self-loop** |
  | cos_interm | `c_25` | plain (factory:175-183) | FIFO produce + consume by the same kernel | 1 toucher | **self-loop** |
  | sin_interm | `c_26` | plain (factory:186-194) | FIFO produce + consume by the same kernel | 1 toucher | **self-loop** |
  | q_output | `c_16` | borrowed (`q_dst_buffer`, factory:205) | FIFO produce only (`reserve_back` + `pack_tile` + `push_back`); output stays resident, nothing drains | 1 toucher (locked producer) | **self-loop** |
  | k_output | `c_17` | borrowed (`k_dst_buffer`, factory:216) | same as q_output (runtime-selected via `is_q`) | 1 toucher | **self-loop** |

  No hidden second writers (no `get_write_ptr`/`fifo_wr_ptr` anywhere), no multi-reader shapes, no dual-instance work-split (one `KernelDescriptor`, one source per config), no multi-binding, **no dead CBs** — all 10 indices are statically referenced by both kernel variants via CTAs 0-12. Note the q/k halves are *runtime*-selected per node (the `is_q` RTA), not compile-time-gated, so the kernel statically references all 10 CBs on every work core — no conditional-DFB case arises.
- **Offset base pointers:** GREEN — the op passes **no address RTAs at all**. The kernel's entire runtime-arg surface is one control flag (`is_q`, factory:258-268 / kernels:29); all tensor addresses reach the device exclusively through `CBDescriptor::buffer` (borrowed memory). Nothing to fold an offset into. The op does not appear in the offset-base-pointer triage analysis (`2026-07-19_offset_base_pointers.md`) — consistent with the scan (no fold, op not in the tables → clean).
- **TensorAccessor 3rd argument:** **N/A — no accessor in the op passes a 3rd argument; the op constructs no `TensorAccessor` anywhere** (it has no dataflow kernels; grep for `TensorAccessor` across the op is zero-hit). The op is absent from the 3rd-arg triage table (`2026-07-06_tensor_accessor_3rd_arg_triage.md`) — consistent.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): all **clean** (borrowed-memory DFB — `DataflowBufferSpec::borrowed_from`): `q_input` (c_0), `k_input` (c_1), `cos` (c_2), `sin` (c_3), `trans_mat` (c_4), `q_output` (c_16), `k_output` (c_17). No Case 1, no Case 2 — the op has no address RTAs/CRTAs, no `Buffer*`-binding RTAs, no CTA-baked addresses.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none — no accessors exist.
- **CB endpoints:** self-loop all 10 `(CB, both configs)` — bind the compute kernel PRODUCER + CONSUMER on each. No multi-binding flags, no dead-CB drops, no conditional DFBs.

## Heads-ups  *(mirrors the brief)*

- **q/k runtime mux over static bindings:** the kernel selects between the q and k CB triples at *runtime* from the `is_q` RTA into non-constexpr locals (`uint32_t in_cb = q_in_cb; … if (!is_q) { in_cb = k_in_cb; … }`, both kernels lines 40-47), then constructs `CircularBuffer` objects from the runtime-selected index (sharded.cpp:59-60) and passes the runtime index to LLK calls. Metal 2.0 bindings are static named tokens, so the kernel binds *all* DFBs; the port must re-express the mux — `dfb::name`'s constexpr `uint32_t` cast covers the LLK-call positions (assign the token into the runtime local), but constructing the `DataflowBuffer` *object* from a runtime-selected index (vs. from a token) needs confirming — the fallback is two objects (q and k) with a branch selecting a reference. This is the port's one non-mechanical kernel-side spot.
- **Kernel core range ≠ CB core range — do not "clean this up":** all 10 CBs span `all_cores_bb` (the bounding box of the cos/sin grid, factory:69) while the compute kernel is deliberately placed on `work_cores = q_cores ∪ k_cores` (factory:76) — the comment at factory:71-76 documents that placing the kernel on bounding-box "hole" cores SIGABRTs under watcher (those cores get zero RTAs, so `get_arg_val(0)` reads out of bounds). Keep the KernelSpec on `work_cores`. For the DFB specs, decide the core ranges deliberately (borrowed DFBs plausibly follow their tensor's shard grid); note that legacy configures q-backed CBs on k cores (and vice versa) where the backing shard buffer has no allocation on that core — harmless because never dereferenced there, but worth keeping in mind when choosing spec core ranges.
- **RTA surface:** exactly one named RTA — `is_q` (1 on every q core, 0 on every k core; factory:258-268). No RTA/CRTA varargs, no CTA varargs (all 13 CTAs read at constexpr indices, kernels:34-57; the 10 CB-index CTAs dissolve into `dfb::` tokens, leaving `q_Ht`, `k_Ht`, `Wt` as named CTAs).
- **Landed sibling precedent (vocabulary, not a template):** the sibling op `rotary_embedding_llama` is already **MetalV2 on `main`** (all factories), and its same-named, same-algorithm compute kernel `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama_sharded.cpp` is a landed Metal 2.0 kernel using the vocabulary `dfb::input`/`dfb::cos`/`dfb::sin`/`dfb::trans_mat`/`dfb::rotated_interm`/`dfb::cos_interm`/`dfb::sin_interm` (lines 27-35). This op's kernels are private copies with this factory as their sole binder (census verified), so no `_metal2` fork convention applies — convert in place — but conforming to the sibling's established binding vocabulary (extended for the q/k split, e.g. `dfb::q_input`/`dfb::k_input`) avoids gratuitous family divergence. Until this port lands, the two same-named files sit in different API eras — don't confuse them.
- **Compute kernel opt level:** prior ports have hit a silent perf regression from Metal 2.0's `KernelSpec` defaulting compute kernels to a lower optimization level than the legacy path built with — set it explicitly rather than relying on the default (no build/test signal fires if it's wrong; both kernels note they are already within 4B of the TRISC2 code-size limit with the profiler on, kernels:24, factory:256, so the build flags genuinely matter here).

## Team-only

- **Out-of-directory coupling & donor shape:** ✓ **clean.** No function-call escapes: every `#include` in both kernels resolves to `tt_metal` framework API headers (class 1 — no concern): `api/compute/common.h`, `api/compute/eltwise_binary.h`, `api/compute/bcast.h`, `api/compute/matmul.h`, `api/compute/compute_kernel_hw_startup.h`, `api/dataflow/circular_buffer.h` (both kernels, lines 7-12). No borrowed kernel files: the factory instantiates only the two op-owned sources; the filename census (`grep -rl rotary_embedding_llama_sharded ttnn/cpp/ttnn/operations/`) shows no other factory binding either file — the sibling `rotary_embedding_llama` op binds its own private copy at a different path. No `_metal2` fork exists beside either kernel (locational check of `device/kernels/compute/` — none needed either, sole-binder files convert in place). No summary table or per-call detail needed (all rolls ✓).
- **Relaxation candidates:** none — the op has no custom hash to mine.
- **TTNN factory analysis:** op-owned tensors: none (plain `ProgramDescriptor` return). MeshWorkload need: none. Pybind: only the public composite op is bound (`ttnn::bind_function`, nanobind.cpp:18) — no `create_descriptor` or device-op internals exposed. Custom hash: none. `get_dynamic_runtime_args`: none. `override_runtime_arguments`: none. Target concept: `ProgramSpecFactoryConcept` (no op-owned tensors, no override method). The factory's own header comment (`device/rotary_embedding_llama_fused_qk_program_factory.hpp:14-17`) accurately summarizes the pointer-patching contract the sheet classifies as `PD Op (pointer-patching)`.

## Misc anomalies  *(team-only, non-gating)*

- **Mixed tile-size bases in the interm CB descriptors:** `cos_interm` (factory:175-183) and `sin_interm` (factory:186-194) set `.total_size = num_interm_tiles * input_single_tile_size` but `.page_size = cos_single_tile_size` / `sin_single_tile_size`. Consistent today only because `validate_on_program_cache_miss` forces every tensor to bfloat16 (`device/rotary_embedding_llama_fused_qk_device_operation.cpp:33`), making all tile sizes equal; a latent total/page mismatch if dtypes ever diverge.
- **Dead kernel locals (tiled variant only):** `cos_cb_obj`, `sin_cb_obj`, `trans_mat_cb_obj` are constructed and never used (`device/kernels/compute/rotary_embedding_llama_sharded.cpp:61-63`); the row-major variant correctly omits them (cos/sin/trans_mat are read only via LLK index calls, which need no wrapper object).
- **TRISC2 code-size cliff:** the `has_work` early-return is commented out in both kernels because TRISC2 overflows its code size by 4 bytes with the profiler on (kernels:24-28; factory:255-257 — "need to reduce stack size by 4B"). The op sits at the edge of the TRISC2 binary budget; any kernel-side growth risks tipping it.
- **Bounding-box CB allocation:** all 10 CBs are allocated over `all_cores_bb` (factory:103 and siblings), which can include hole cores that belong to neither q nor k (the very cores the kernel-placement comment at factory:71-76 works around) — L1 is burned on cores that do no work, and q-backed CBs are configured on k cores (and vice versa) at addresses where the backing shard buffer has no per-core allocation (never dereferenced, so harmless today).

## Questions for the user

None.

## Recipe notes

- **Endpoint definition vs. LLK compute reads:** the CB-endpoints census defines an endpoint as FIFO-produce, FIFO-consume, or raw-pointer access (`get_write_ptr`/`get_read_ptr`/`fifo_*_ptr`). This op's cos/sin/trans_mat CBs are touched by none of those — the compute kernel reads them purely through LLK index-taking calls (`matmul_tiles`/`mul_tiles_bcast` srcB operands) with no FIFO sync (data resident from the sharded backing buffer). I counted such LLK index reads as role-free touches (in Metal 2.0 the kernel must bind the DFB to name it at all, so the access is a binding), which the recipe's definition doesn't explicitly enumerate. Suggest adding "LLK compute calls consuming the CB index" to the endpoint definition's access list.
- **Per-node census where the CB core range exceeds every kernel's core range:** the CBs span `all_cores_bb` while the op's only kernel runs on `work_cores` ⊂ bb — on hole nodes every CB has zero touchers, yet this isn't the recipe's "dead CB" case (dead is per `(CB, config)`, and each CB is live wherever any kernel runs) nor its conditional-DFB case (the emptiness varies per *node*, not per config). I classified on the kernel-bearing nodes and surfaced the range mismatch as a porter heads-up; the recipe could say explicitly how a DFB whose core range exceeds all binder kernels' ranges should be treated/expressed.
- **Provenance command prints nothing:** as anticipated by the recipe — this checkout carries the metal_2.0 docs untracked (`??` in git status), so the version line records the fact instead of a hash.

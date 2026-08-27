# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode`

One DeviceOperation, three program factories, three op-owned kernels:

- **`NLPCreateQKVHeadsDecodeDeviceOperation`** (`device/nlp_create_qkv_heads_decode_device_operation.hpp`)
  - `NLPCreateQKVHeadsDecodeInterleavedProgramFactory` (`device/nlp_create_qkv_heads_decode_interleaved_program_factory.cpp`) — interleaved (DRAM or L1) input; selected when the input is not sharded
  - `NLPCreateQKVHeadsDecodeShardedProgramFactory` (`device/nlp_create_qkv_heads_decode_sharded_program_factory.cpp`) — width-sharded input on a full coregrid
  - `NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory` (`device/nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp`) — width-sharded input on subcoregrids

Kernels (each bound by exactly one factory; all three live in `device/kernels/`, none referenced by any other op — census grep over `ttnn/cpp/ttnn/operations/` returns no external binders, and no `_metal2` forks exist beside them):

- `reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` — interleaved factory, instantiated **twice** (Reader phase-1 / Writer phase-2 configs) over the same core range
- `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` — sharded factory, instantiated 2× (overlap_qk_coregrid) or 4× (non-overlap: q pair on q-cores, k pair on k-cores)
- `reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp` — subcoregrid factory, instantiated 2× or 4× (same scheme)

There are **no compute kernels** — the op is pure data movement (a head-shuffle TM).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources` *(the `git log -- docs/.../metal_2.0/` provenance command prints nothing in this checkout — the recipe docs are not on this branch; hash above is the HEAD of `origin/akertesz/op-porting-recipe`, the doc branch this audit was read from)*

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `NLPCreateQKVHeadsDecodeDeviceOperation` → Interleaved, Sharded, ShardedSubcoregrid |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all three op kernels are structurally Device 2.0 (`Noc`, `CircularBuffer` wrappers, `TensorAccessor`, `UnicastEndpoint`, `CoreLocalMem`); the one donor call (`tt_memmove`) is Device 2.0 native |
| *Prereqs* — Cross-op escapes | Ok — one function-call escape (`tt_memmove`, `data_movement/common/kernels/common.hpp`), signature is `(Noc, uint32_t, uint32_t, uint32_t)` — no resource handles, nothing to bridge |
| *Feature Support* — overall | **GREEN** — no Appendix A entry fires (all N/A) |
| *Feature Support* — Variadic-CTA | Ok — all CTAs read at constexpr offsets |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** for all three factory rows (sheet fetched fresh 2026-08-27; cross-check clean) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all three factories) |
| *TTNN Readiness* — Secretly SPMD | N/A (not WorkloadDescriptor) |
| *TTNN Readiness* — Custom hash | No (grep for `compute_program_hash` / `attribute_values` / `to_hash`: no hits) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`nlp_create_qkv_heads_decode_nanobind.cpp` binds only the composite user-facing function) |
| *TTNN Readiness* — Op-owned tensors | No (sheet blank; `descriptor` concept cannot carry them) |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (all three factories) |
| *Port work* — Offset base pointer | none — no host-side fold anywhere; interleaved passes base (`Buffer*`) + a *separate* scalar offset RTA; sharded/subcoregrid compute the offset on-device |
| *Port work* — Tensor bindings (per binding) | input: **Case 1** (Interleaved factory) / **Case 2** (Sharded, Subcoregrid) · batch_offset (optional): **Case 1** (Sharded, Subcoregrid) · q/k/v outputs: **clean** (borrowed-memory CBs, all factories) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (all three rows — clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor in the op passes a 3rd arg (the usual outcome) |
| *Port work* — CB endpoints | 1P+1C (q/k/v outputs, all factories) · self-loop (scratch CBs: interleaved `c_0`/`c_1`, subcoregrid `c_15`/`c_14`) · **multi-binding flag** (sharded `c_15` — two locked producers) · **dead-CB drop** (sharded `c_14`) · conditional DFBs (all scratch/batch-offset CBs are config-gated, host conditionals already exist) |

**CB endpoints** are dispositions, not gates (see `metal2_audit.md`, CB endpoints): every out-of-window CB here has a port-time resolution. Details per `(CB, config)` below.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, same directory). All five gates clear on every factory: Device 2.0 ✓ · Feature compatibility ✓ · TTNN factory concept ✓ (all three rows `Is able to port? == yes`, relaxation `none`) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (no sites). The three factories share one DeviceOperation and port as one unit to `ProgramSpecFactoryConcept`.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Sheet fetched fresh this session (2026-08-27); the op has exactly three rows, one per factory, all `Concept == descriptor`, `Is able to port? == yes`, `TensorParameter relaxation == none`, `Known op issues` empty. Cross-check against the code is clean on every primary column:
  - `Concept` — all three factories define `create_descriptor()` returning `ProgramDescriptor` ✓
  - `Custom hash` == `no` — no `compute_program_hash`, no `attribute_values`/`to_hash` backdoor in the op directory ✓
  - `Runtime-args update (get_dynamic_runtime_args)` == `no` — no such hook on the device-op ✓
  - `Override runtime args method?` == `no` — no `override_runtime_arguments` anywhere ✓
  - `Pybind descriptor` == `no` — the nanobind file binds only the composite `ttnn::experimental::nlp_create_qkv_heads_decode` function ✓
  - `Smuggled pointer` == `no` — no `->address()` flows into any RTA; the factories push `Buffer*` objects (the framework-patched `Buffer*`-binding form), which is the *correct-on-cache-hit* delivery, not the smuggled-address hazard ✓ (matches `Op Classification == "PD Op (pointer-patching)"`)
  - Factory-set match — three sheet rows ↔ three factories in `program_factory_t` (`device/nlp_create_qkv_heads_decode_device_operation.hpp:22-25`), no phantom or missing rows ✓
  - Cross-column invariants — no violations (`get_dynamic_runtime_args == no`; `Op-owned tensors?` blank on `descriptor` rows) ✓
- **Device 2.0 (every kernel used):** GREEN. All three kernels are built on Device 2.0 idioms throughout: `Noc noc` + `noc.async_read(...)` / `async_read_barrier()`, `CircularBuffer` wrapper objects with method calls (`cb_q_out.get_write_ptr()` etc. — wrapper *methods*, not CB-index free functions), `TensorAccessor`, `UnicastEndpoint`, `CoreLocalMem`. No `InterleavedAddrGen*`/`ShardedAddrGen`, no raw `noc_async_*` free functions, no raw semaphore addresses, no CB-index-keyed free-function holdovers. The single donor function called, `tt::data_movement::common::tt_memmove` (`ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:143`), is itself Device 2.0 native (takes a leading `Noc`, uses `noc.async_read/async_write` internally); the kernel calls the current `Noc`-parameter overload, **not** the `[[deprecated]]` Noc-less one (`common.hpp:211`).
- **Feature compatibility:** every Appendix A entry scanned (host, factories, kernels, nanobind); no recognition signal fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no type reference, no `global_circular_buffer` field on any `CBDescriptor`, no `remote_index`/remote-CB idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset` anywhere in the op |
  | GlobalSemaphore | N/A | no semaphores of any kind in this op |

- **CB endpoints (GATE-free):** full per-`(CB, config)` census below. Summary: all q/k/v output CBs are two-toucher dual-instance work-splits → **1P+1C**; the interleaved scratch CBs and subcoregrid batch-offset CBs are one-toucher → **self-loop**; the sharded factory's `c_15` is a genuine two-locked-producer CB → **multi-binding flag**; the sharded factory's `c_14` is a confirmed **dead CB → drop**. Nothing blocks the Gen1 port.
- **Offset base pointers:** GREEN — no address RTA folds a host-side offset into its base.
  - Interleaved factory: the per-core `in_tile_offset_by_batch` is passed as a **separate scalar RTA** (`nlp_create_qkv_heads_decode_interleaved_program_factory.cpp:181-188`) alongside the clean `Buffer*` base; the kernel applies it via the accessor's `.offset_bytes` read parameter. This is the already-split-out GREEN shape.
  - Sharded/subcoregrid factories: the offset is **computed on-device** (from the batch-offset tensor value + core index, `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:60-64`); the host passes only the clean `Buffer*` base.
  - Triage-doc reconcile (`2026-07-19_offset_base_pointers.md`, a dated prior): this op is **not** in its tables — the listed `nlp_create_qkv_heads` / `nlp_create_qkv_heads_boltz` rows are the *sibling prefill ops*, not this one. Scan says clean; "no fold, op not in the tables" → clean.
- **TensorAccessor 3rd argument:** N/A — no accessor in the op passes a 3rd argument, so the subject never fires. Sites checked: `reader_interleaved_...cpp:59` (`TensorAccessor(qkv_args, q_start_addr)`), `reader_tm_...cpp:48` and `..._on_subcoregrids.cpp:48` (`TensorAccessor(index_args, batch_offset_tensor_addr)`) — all two-argument. The op is absent from `2026-07-06_tensor_accessor_3rd_arg_triage.md`, consistent.

## CB endpoint census (per CB, per config, per node)

No kernel in this op uses FIFO sync on the q/k/v output CBs — they are borrowed-memory (output-tensor-backed) CBs written by raw `get_write_ptr() + offset` only. The only FIFO ops in the op are `reserve_back(1)`/`push_back(1)` on the batch-offset scratch CBs.

### Interleaved factory (2 same-source instances — Reader phase-1 + Writer phase-2 — both over `q_cores`; `overlap_qk_coregrid` is always true on this path)

| CB | Backing | Config | Touchers per node | Disposition |
|---|---|---|---|---|
| `c_16` q_out | borrowed `output[0].buffer()` | always | 2 role-free (both instances raw-write) | **1P+1C** |
| `c_17` k_out | borrowed `output[1].buffer()` | always | 2 role-free | **1P+1C** |
| `c_18` v_out | borrowed `output[2].buffer()` | always | 2 role-free | **1P+1C** |
| `c_0` reader scratch | plain | only when `use_aligned_path` (DRAM input **and** `sub_tile_line_bytes < dram_alignment`, factory `:98-100`) | 1 (Reader instance only — CTA[12]=`c_0`) | **self-loop**, **conditional DFB** (host conditional already exists, factory `:111-133`) |
| `c_1` writer scratch | plain | only when `use_aligned_path` | 1 (Writer instance only — CTA[12] overridden to `c_1`, factory `:165`) | **self-loop**, **conditional DFB** |

The scratch CBs are read *and* written by their one owner (accessor reads land in them; `tt_memmove` NoC-reads out of them) — sync-free single-toucher, the canonical self-loop.

### Sharded factory (2 instances on `q_cores` when overlap; 4 instances — q pair on `q_cores`, k pair on disjoint `k_cores` — when non-overlap)

| CB | Backing | Config | Touchers per node | Disposition |
|---|---|---|---|---|
| `c_16` q_out | borrowed `output[0]` | always (lives on `q_cores`) | 2 role-free (q_reader + q_writer; k instances have `PROCESS_QV=0` and live on disjoint cores) | **1P+1C** |
| `c_17` k_out | borrowed `output[1]` | always (lives on `k_cores`) | 2 role-free (overlap: q pair; non-overlap: k pair) | **1P+1C** |
| `c_18` v_out | borrowed `output[2]` | always (lives on q grid) | 2 role-free (q pair) | **1P+1C** |
| `c_15` batch-offset (reader idx) | plain | only when `batch_offset.has_value()` (on `qk_cores`) | **2 locked producers** — *both* co-resident instances get CTA[16]=`c_15` (the writer copy overrides only CTA[9], factory `:200-201`; k copies likewise `:218-220,230-231`), and each does `reserve_back(1)`/`push_back(1)` + raw read-back (kernel `:47-59`) | **multi-binding advanced option** (census can't fit 1P+1C: two kernels locked to the producer role) · **conditional DFB** (host conditional exists, factory `:64-88`) |
| `c_14` batch-offset (writer idx) | plain | allocated when `batch_offset.has_value()` (factory `:79-88`) | **0 in every config** — no kernel CTA ever carries `c_14` in this factory; the kernel reads its CB index only from CTA[16] | **dead-CB drop** (positively confirmed; see anomaly below — the drop is the port's zero-functional-change move, the suspected missing override is the ops team's) |

### Subcoregrid factory (2 or 4 instances, same scheme as sharded)

| CB | Backing | Config | Touchers per node | Disposition |
|---|---|---|---|---|
| `c_16` q_out | borrowed `output[0]` | always | 2 role-free (q pair) | **1P+1C** |
| `c_17` k_out | borrowed `output[1]` | always (on `k_cores`) | 2 role-free | **1P+1C** |
| `c_18` v_out | borrowed `output[2]` | always (on q grid) | 2 role-free | **1P+1C** |
| `c_15` batch-offset reader | plain | only when `batch_offset.has_value()` | 1 locked producer (Reader instances only — writers get CTA[15] overridden to `c_14`, factory `:198,229`) | **self-loop** · **conditional DFB** |
| `c_14` batch-offset writer | plain | only when `batch_offset.has_value()` | 1 locked producer (Writer instances only) | **self-loop** · **conditional DFB** |

**Cross-cutting kernel note for the conditional DFBs:** each kernel constructs its `CircularBuffer` wrappers **unconditionally** and gates *use* behind `if constexpr` (`PROCESS_QV`/`PROCESS_K`/`use_batch_offset`/`USE_ALIGNED_PATH`), and in the sharded/subcoregrid non-overlap configs the untouched CB (e.g. `c_17` for a q-pair instance) **does not even exist on that node**. Under Metal 2.0 an unbound token is a compile error and `if constexpr` does **not** gate `dfb::` name lookup, so per-instance conditional bindings need `KernelSpec` `defines` + `#ifdef` gating. Flagged as porter work in the brief.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, with per-factory split):
  - `input_tensor` — **Interleaved factory: Case 1** (`Buffer*` RTA delivers the base, kernel feeds it to `TensorAccessor(qkv_args, q_start_addr)` and reads through it, `reader_interleaved_...cpp:59,90-95`). **Sharded + Subcoregrid factories: Case 2** (same `Buffer*` delivery, but the kernel walks shard addresses by hand — `qkv_read_addr = q_start_addr + in_tile_offset_by_batch`, remote-core reads via `UnicastEndpoint` with explicit `noc_x/noc_y/addr` — never constructing an accessor over the input): bind as `TensorParameter`, pull the base via the `TensorAccessor::get_bank_base_address` bridge, keep the raw walk unchanged.
  - `batch_offset` (optional tensor; Sharded + Subcoregrid only) — **Case 1** (`TensorAccessor(index_args, batch_offset_tensor_addr)`, kernel `:48`). Delivered today as `Buffer*` when present / literal `0` when absent (factories' `push_batch_offset` lambda), with `use_batch_offset` CTA + `TensorAccessorArgs(nullptr)` on the absent path — the binding (and the kernel's read) must stay conditional.
  - `q/k/v outputs` — **clean**: borrowed-memory CBs (`.buffer = output[i].buffer()`) in every factory → `DataflowBufferSpec::borrowed_from`; legality resolved in the CB census (all 1P+1C).
- **TensorParameter relaxation:** none (all three sheet rows).
- **TensorAccessor 3rd arg:** none — no sites.
- **CB endpoints:** per the census — 1P+1C on all q/k/v output CBs (every factory); self-loop `c_0`/`c_1` (interleaved, conditional on `use_aligned_path`) and `c_15`/`c_14` (subcoregrid, conditional on `batch_offset`); multi-binding flag on sharded `c_15` (conditional on `batch_offset`); dead-CB drop of sharded `c_14` @ `nlp_create_qkv_heads_decode_sharded_program_factory.cpp:79-88`.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shape to watch):** sharded `c_15` — both co-resident instances FIFO-produce one page and read it back (a "both RISCs stage the same scalar" idiom, not a hidden co-fill; there is no consumer). Census: 2 locked producers → the flag. `(c_15, sharded × batch_offset present)` @ kernel `:47-59`, factory `:171-188,200-201`.
- **Cross-op / shared kernels:** none borrowed, none lent — all three kernel files are op-owned, bound only by this op's factories (census grep clean), no `_metal2` forks exist anywhere for them (this port creates none — they aren't shared), no quasar copies found. Intra-op: each kernel is bound by exactly one factory, and all three factories port together in this one port, so no fork rungs apply.
- **RTA varargs:** sharded + subcoregrid kernels read the input-shard NoC coordinate tables through `get_arg_addr` pointers with runtime indexing — `in0_mcast_noc_x/_y` blocks of CTA-bounded, config-varying length (`num_x`/`num_y`, sharded kernel `:42-43`; `in_num_cores`, subcoregrid kernel `:42-43`). Genuine varargs (variable-count block; also runtime-data-selected element reads). The three leading scalars (`q_start_addr`, `batch_offset_tensor_addr`, `index_in_cores`, fixed indices 0–2) are nameable and must **not** ride the varargs. Interleaved kernel: no varargs (two fixed named RTAs).
- **Conditional bindings need define-gating:** `if constexpr` does not gate `dfb::`/`tensor::` name lookup, and several bindings are config- or instance-conditional (scratch DFBs; batch-offset tensor + DFB; `k_out` vs `q_out`/`v_out` per instance in non-overlap mode, where the unused CB *doesn't exist on the node*). The port needs `KernelSpec` `defines` + `#ifdef` per the port recipe's conditional-binding guidance.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Roll-up: ✓ clean.** One function-call escape across the whole op; no borrowed kernel files.
  - Summary table:

    | Op kernel | Donor file | Bucket | Functions used | Shape | Status |
    |---|---|---|---|---|---|
    | `reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | cross-family donor (data_movement common pool) | `tt_memmove<false, true, true, SUBTILE_LINE_BYTES>(noc, dst, src, bytes)` | `(Noc, uint32_t l1_addr, uint32_t l1_addr, uint32_t)` — plain scalars + Device 2.0 `Noc` by value; no CB/semaphore/tensor handles in the signature | ✓ |
    | all three kernels | `tt_metal` framework headers (`api/dataflow/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`, `internal/risc_attribs.h`, `tt-metalium/constants.hpp`) | tt_metal / framework | — | — | ✓ no concern |

  - Per-call detail: `tt_memmove` needs no token bridging — both addresses the op passes are locally computed L1 pointers (output-CB write pointer, scratch-CB pointer), not bound-resource handles. The kernel already uses the current leading-`Noc` overload; the `[[deprecated]]` Noc-less overload (`common.hpp:211`) is not used.
  - **Borrowed kernel files:** none. The op's factories instantiate only the three op-owned kernel files; no other op's factory binds them (grep across `ttnn/cpp/ttnn/operations/` finds no external references to any of the three filenames).
- **Relaxation candidates:** none — there is no custom hash to mine.
- **TTNN factory analysis:** current concept `descriptor` (×3); target **`ProgramSpecFactoryConcept`** (×3; `Override runtime args method? == no` everywhere). No op-owned tensors, no MeshWorkload need, no pybound `create_descriptor` or other risky pybind (the nanobind file binds one composite function with plain tensor/scalar args), no custom hash, no `get_dynamic_runtime_args`, no `override_runtime_arguments`. Outputs are created by the standard `create_output_tensors` from `compute_output_specs`; the `Buffer*`-binding RTA form is used for `input_tensor` and `batch_offset` (framework-patched on cache hits — consistent with the sheet's `PD Op (pointer-patching)` classification and `Pointer patching perf issue? == OK`).

## Misc anomalies  *(team-only, non-gating)*

- **Sharded factory: writer batch-offset CB never wired — `c_14` allocated dead, both RISCs share `c_15`.** `nlp_create_qkv_heads_decode_sharded_program_factory.cpp:79-88` allocates `c_14` (`batch_offset_cb_index_writer`) whenever `batch_offset` is provided, but the writer CTAs copy the reader's and override only phase (`:200-201`) — CTA[16] stays `c_15` for all four instances (k copies too, `:218-220,230-231`). Compare the subcoregrid factory, which *does* override the writer's CB index (`nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp:198,229`) — the sharded override looks simply missing. Consequences today: `c_14` burns L1 for nothing, and both RISCs on a node `reserve_back`/`push_back` the *same* `c_15` instance — an unsynchronized shared-counter push that is benign only because both write identical data to the same first page and nothing ever consumes the CB. Ops-team fix candidate; the port must *not* fix it (it drops dead `c_14` — no behavior — and expresses the shared `c_15` faithfully via the multi-binding flag).
- **Subcoregrid factory sizes the V output CB from Q's shard spec.** `nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp:114-116`: `v_shard_spec = output[0].shard_spec()` (Q) and `v_cores = q_shard_spec.grid`, while `.buffer = output[2].buffer()` (V). Masked today because `num_q_heads` and `num_kv_heads` both pad to 32 (≤32-heads validation), making the Q and V shard shapes equal — but it's a latent mis-size if the padding rule ever diverges. The sharded factory does it correctly from `output[2]` (`..._sharded_program_factory.cpp:114-116`). (Interleaved factory has the cosmetic cousin: `v_cores = q_shard_spec.grid` at `..._interleaved_program_factory.cpp:74`, harmless since the grids are equal by construction.)
- **Possible one-past-the-end read of the NoC-coordinate RTA arrays after the final tile.** All three shard-reading loops advance the core cursor *after* the last tile and immediately re-index the coordinate arrays without a bounds check (`reader_tm_...cpp:182-191` — `qkv_y` can reach `num_y`; `..._on_subcoregrids.cpp:226-231` — `cur_core_idx` can reach `in_num_cores`). The fetched garbage coordinate is never used (no further reads are issued), so it's harmless today, but it reads beyond the argument block.
- **Composite entry point accepts and ignores `optional_output_tensors`.** `nlp_create_qkv_heads_decode.cpp:17` — the parameter is anonymous/unused, yet the nanobind surface exposes `output_tensors` (`nlp_create_qkv_heads_decode_nanobind.cpp:33`), so a user passing preallocated outputs is silently ignored.

## Questions for the user

*(none)*

## Recipe notes

- The sharded `c_15` census — **two locked producers that each stage one page of identical data for their own read-back, with no consumer ever** — is a shape the CB-endpoints table maps cleanly to the multi-binding flag (≥2 locked producers), but it is really *two independent single-toucher scratch uses accidentally sharing one CB index* (the missing writer-CB override above). The classification table handles it; noting it because a porter seeing "multi-binding" may go hunting for a hidden co-fill/consumer that doesn't exist. The brief's Watch-for says this explicitly.

# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf`

- **`RotaryEmbeddingHfDeviceOperation`** (`device/rotary_embedding_hf_device_operation.hpp`)
  - `RotaryEmbeddingHfMultiCore` (`device/rotary_embedding_hf_multi_core_program_factory.cpp`) — prefill; internally two descriptor shapes: **single-tile prefill** (`Wt == 1`, borrows the sibling `rotary_embedding` op's compute kernel) and **multi-tile prefill**.
  - `RotaryEmbeddingHfMultiCoreSharded` (`device/rotary_embedding_hf_sharded_program_factory.cpp`) — decode (height-sharded); internally **single-tile decode** and **multi-tile decode** shapes.

All 9 kernel files in `device/kernels/` are referenced by a factory (no unreferenced/dead kernel files). One out-of-directory kernel is instantiated: `../rotary_embedding/device/kernels/compute/rotary_embedding_single_tile.cpp`.

This audit covers **rotary_embedding_hf only** — not the sibling `rotary_embedding` op (audited separately) and not the `rotary_embedding_llama` variants.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `2cd0286fa17 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `RotaryEmbeddingHfDeviceOperation` → `RotaryEmbeddingHfMultiCore`, `RotaryEmbeddingHfMultiCoreSharded` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 9 own kernels + the 1 borrowed kernel are Device 2.0 (`Noc`, `CircularBuffer` wrappers, `TensorAccessor`); only sanctioned free functions (`get_tile_size(cb_id)`) remain |
| *Prereqs* — Cross-op escapes | Ok — all donor shapes ✓ (`uint32_t cb_id` / kernel-lib); one borrowed kernel file (fork convention applies, see Team-only) |
| *Feature Support* — overall | GREEN — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok — none (all CTAs read at fixed constexpr indices) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factory rows; sheet fetched fresh 2026-08-25) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories; `create_descriptor` returning `ProgramDescriptor` confirmed in code) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (concept is `descriptor`) |
| *TTNN Readiness* — Custom hash | No (grep of op dir: no `compute_program_hash`, no `attribute_values`/`to_hash`) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`rotary_embedding_hf_nanobind.cpp` binds only the public op function) |
| *TTNN Readiness* — Op-owned tensors | No (sheet cell blank; `descriptor` concept cannot carry them) |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (sheet's `Porting Target` column agrees) |
| *Port work* — Offset base pointer | none — no address RTA folds an offset; factories pass whole `Buffer*` objects |
| *Port work* — Tensor bindings (per binding) | MultiCore: `input`/`cos`/`sin`/`output` **Case 1** (with per-config borrowed-DFB overlaps, see detail); Sharded: all four **clean** (borrowed-DFB, no address RTAs at all) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears; quoted verbatim from both rows) |
| *Port work* — TensorAccessor 3rd arg | **drop (Class 2)** — 9 sites, all `get_tile_size(cb)` == exact tile page size; triage doc row `rotary_embedding_hf → 2 — Redundant` confirmed by code |
| *Port work* — CB endpoints | legal 1P+1C or **self-loop** everywhere; **no** multi-binding, **no** dead CB, **no** conditional DFB |

**CB endpoints** are dispositions, not gates (see `metal2_audit.md` — CB endpoints): every out-of-window CB here resolves to a self-loop (single toucher). Full per-`(CB, config)` census in Gate detail.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md` beside this file). All five gate-bearing subjects clear for both factories: Device 2.0 ✓, Feature compatibility ✓ (all Appendix A rows N/A), TTNN factory concept ✓ (`Is able to port? == yes`, cross-check clean), Offset base pointers ✓ (none), TensorAccessor 3rd arg ✓ (all sites Class 2 — mechanical drop). `TensorParameter relaxation == none` clears the relaxation conjunct. Target: `ProgramSpecFactoryConcept`.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Sheet fetched fresh this session (2026-08-25) per `ttnn_op_porting_readiness.md`. Both rows (`RotaryEmbeddingHfMultiCore`, `RotaryEmbeddingHfMultiCoreSharded`): `Is able to port? = yes`, `Concept = descriptor`, `TensorParameter relaxation = none`, `Known op issues` empty, `Custom hash = no`, `Backdoor custom hash = no`, `Runtime-args update (get_dynamic_runtime_args) = no`, `Override runtime args method? = no`, `Pybind descriptor = no`, `Smuggled pointer = no`.
  **Cross-check (all clean):**
  - Concept: both factories define `create_descriptor(...) → ProgramDescriptor` (`rotary_embedding_hf_multi_core_program_factory.cpp:634`, `rotary_embedding_hf_sharded_program_factory.cpp:412`). ✓
  - Custom hash / backdoor: no `compute_program_hash`, `attribute_values`, `to_hash` anywhere in the op dir. ✓
  - `get_dynamic_runtime_args` / `override_runtime_arguments`: absent from the device-op. ✓
  - Pybind: `rotary_embedding_hf_nanobind.cpp` binds only `ttnn::experimental::rotary_embedding_hf`; no `create_descriptor` binding. ✓
  - Smuggled pointer = `no`: consistent — the factories pass `Buffer*` objects via `emplace_runtime_args` (the framework-annotated `BufferBinding` form, patched on cache hits), never a raw `->address()` in an RTA. ✓
  - Factory-set match: `program_factory_t = std::variant<RotaryEmbeddingHfMultiCore, RotaryEmbeddingHfMultiCoreSharded>` (`device/rotary_embedding_hf_device_operation.hpp:19`) ↔ exactly two sheet rows, one-to-one. ✓
  - Cross-column invariants: `descriptor` + blank `Op-owned tensors?` — consistent. ✓
  Informational sheet cells: `Op Classification = PD Op (pointer-patching)`, `Pointer patching perf issue? = OK (old custom hash was complete)`, `Formerly custom hashed? = yes`, `Model = other`, not used by llama.

- **Device 2.0 (every kernel used):** GREEN. All ten kernels are structurally Device 2.0: `Noc` object for NoC ops, `CircularBuffer` wrapper objects for all CB access (`reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_write_ptr()` as **methods**), `TensorAccessor` for tensor addressing, `CoreLocalMem` for L1 targets. No `InterleavedAddrGen*`/`ShardedAddrGen`, no raw `noc_async_read` free functions, no raw semaphore addresses (the op uses no semaphores at all). The only CB-index free functions are the **sanctioned** `get_tile_size(cb_id)` (dataflow + compute kernels) and compute-LLK primitives (`mul_tiles`, `matmul_tiles`, `pack_tile`, `reconfig_data_format`, …), which take CB indices by design and are not data-movement APIs. No holdover table needed — zero violations.
  Kernels audited: `reader_rotary_embedding_hf_interleaved.cpp`, `writer_rotary_embedding_hf_interleaved.cpp`, `reader_rotary_embedding_hf_single_tile_interleaved_start_id.cpp`, `reader_rotary_embedding_hf_single_tile_interleaved_start_id_sharded.cpp`, `reader_rotary_embedding_hf_sharded.cpp`, `reader_rotary_embedding_hf_single_tile_sharded.cpp`, `compute/rotary_embedding_hf.cpp`, `compute/rotary_embedding_hf_sharded.cpp`, `compute/rotary_embedding_hf_single_tile_sharded.cpp`, and the **borrowed** `../rotary_embedding/device/kernels/compute/rotary_embedding_single_tile.cpp` (owning family: `experimental/transformer/rotary_embedding`) — also fully Device 2.0.

- **Feature compatibility:** no entry fires. Grep of the op dir for `GlobalCircularBuffer`, `GlobalSemaphore`, `address_offset`, `remote_index`, `remote_cb`: zero hits. No `CBDescriptor` sets `global_circular_buffer` or `address_offset`; the `.buffer = <tensor buffer>` fields are the plain borrowed-memory pattern (supported, mechanical `borrowed_from` translation), not a feature gate.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | absent |
  | CBDescriptor `address_offset` (non-zero) | N/A | field never set (default 0) |
  | GlobalSemaphore | N/A | absent (op uses no semaphores) |

- **CB endpoints (GATE-free):** every CB is a legal 1P+1C FIFO or a single-toucher **self-loop**; no multi-binding, no dead CB, no conditional DFB. All self-loops are on the **compute** kernel (no DM self-loops, so no Quasar-uplift debt on that axis). Census per `(CB, config)`:

  **`RotaryEmbeddingHfMultiCore` — multi-tile prefill** (interleaved; `in_sharded`/`out_sharded` variants only change borrowing, not endpoints):
  | CB | Touchers | Disposition |
  |---|---|---|
  | `c_0` input | reader FIFO-P, compute FIFO-C | legal 1:1 (borrowed from `input.buffer()` when `in_sharded`, factory `:406`) |
  | `c_1` rotated_input | reader FIFO-P, compute FIFO-C | legal 1:1 |
  | `c_2` cos / `c_3` sin | reader FIFO-P, compute FIFO-C | legal 1:1 |
  | `c_4` scalar | reader FIFO-P, compute FIFO-C (`wait_front` only, never pops — still a locked consumer) | legal 1:1 |
  | `c_24`/`c_25`/`c_26` interm | compute only (produce + consume) | **self-loop** |
  | `c_16` output | compute FIFO-P, writer FIFO-C | legal 1:1 (borrowed from `output.buffer()` when `out_sharded`, factory `:499`; writer then only `wait_front`s under `OUT_SHARDED`) |

  **`RotaryEmbeddingHfMultiCore` — single-tile prefill** (`Wt == 1`; interleaved or `in_sharded`):
  | CB | Touchers | Disposition |
  |---|---|---|
  | `c_0` input | reader FIFO-P (in the `in_sharded` variant the produce is a cursor-advance `reserve_back(num_rows)`/`push_back(num_rows)`, reader `..._start_id_sharded.cpp:95-96`), compute FIFO-C | legal 1:1 (borrowed when `in_sharded`, factory `:102`) |
  | `c_1` trans_mat | reader FIFO-P (raw fill inside reserve/push), compute FIFO-C (`wait_front` only) | legal 1:1 |
  | `c_2` cos / `c_3` sin | reader FIFO-P, compute FIFO-C | legal 1:1 |
  | `c_24`/`c_25`/`c_26` interm | compute only | **self-loop** |
  | `c_16` output | compute FIFO-P, writer FIFO-C | legal 1:1 (borrowed when `out_sharded`, factory `:181`) |

  **`RotaryEmbeddingHfMultiCoreSharded` — single-tile decode:**
  | CB | Touchers | Disposition |
  |---|---|---|
  | `c_0` input (borrowed, `:87`) | compute only (reserve/push/wait/pop) | **self-loop** |
  | `c_1` cos (borrowed, `:99`) / `c_2` sin (borrowed, `:111`) | compute only | **self-loop** |
  | `c_3` trans_mat | reader FIFO-P, compute FIFO-C (`wait_front` only) | legal 1:1 |
  | `c_24`/`c_25`/`c_26` interm | compute only | **self-loop** |
  | `c_16` output (borrowed, `:168`) | compute FIFO-P only (result stays resident; nothing drains) | **self-loop** |

  **`RotaryEmbeddingHfMultiCoreSharded` — multi-tile decode:**
  | CB | Touchers | Disposition |
  |---|---|---|
  | `c_0` input (borrowed, `:278`) / `c_1` cos (`:290`) / `c_2` sin (`:302`) | compute only | **self-loop** |
  | `c_3` scalar | reader FIFO-P (`reader_rotary_embedding_hf_sharded.cpp` — its only job), compute FIFO-C (waits at start, pops at end) | legal 1:1 |
  | `c_24`/`c_25`/`c_26` interm | compute only | **self-loop** |
  | `c_16` output (borrowed, `:360`) | compute FIFO-P only | **self-loop** |

  Dead-CB check: every allocated CB is touched in every config that allocates it (single-tile vs multi-tile shapes allocate *different* CB sets from separate descriptor-builder functions, so there is no CB allocated-but-unused under either shape). No hidden second writer: no semaphores exist in the op, and no kernel raw-writes a CB it doesn't FIFO-produce.

- **Offset base pointers:** GREEN — no address RTA folds a host-side offset into its base. Both factories pass whole `Buffer*` objects (`emplace_runtime_args` at multi_core factory `:298-312` and `:608-618`; the sharded factory passes **no** runtime args at all), and every kernel-side address arg is consumed solely as a `TensorAccessor` base. Not listed in the offset-base-pointer triage doc (`2026-07-19_offset_base_pointers.md`) — reconciliation outcome: *no fold, op not in the tables → clean*.

- **TensorAccessor 3rd argument:** GREEN — every site is **Class 2 (redundant)** → mechanical drop. Nine sites, all of the form `TensorAccessor(args, addr, get_tile_size(cb))` where the CB's data format matches the tensor's dtype, so the value is the exact tile page size (correct magnitude; equals the buffer page size for tile-layout tensors, and is 64-aligned for all reachable formats — bfp8 1088, bf16 2048, fp32 4096 — so it equals the aligned page size verbatim even if an accessor is ever sharded):
  - `device/kernels/dataflow/reader_rotary_embedding_hf_interleaved.cpp:39,42,45` (src/cos/sin)
  - `device/kernels/dataflow/writer_rotary_embedding_hf_interleaved.cpp:23` (dst)
  - `device/kernels/dataflow/reader_rotary_embedding_hf_single_tile_interleaved_start_id.cpp:93,96,99` (src/cos/sin)
  - `device/kernels/dataflow/reader_rotary_embedding_hf_single_tile_interleaved_start_id_sharded.cpp:99,102` (cos/sin)
  Cross-check against the triage doc (`2026-07-06_tensor_accessor_3rd_arg_triage.md:72`): `rotary_embedding_hf → 2 — Redundant` — agrees with the code read.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory):
  - `RotaryEmbeddingHfMultiCore` / multi-tile prefill: `input` **Case 1** (reader RTA arg0 `Buffer*` → `TensorAccessor s0`, reader `:33,39`; when `in_sharded`, the same tensor is *also* the borrowed backing of `c_0` — the port carries both a `TensorParameter` binding and a `borrowed_from` DFB for the same tensor); `cos` **Case 1**; `sin` **Case 1**; `output` **Case 1** (writer; under `out_sharded`/`OUT_SHARDED` the accessor path is compiled out and the tensor is the borrowed backing of `c_16`).
  - `RotaryEmbeddingHfMultiCore` / single-tile prefill: interleaved — `input`/`cos`/`sin`/`output` all **Case 1**; `in_sharded` — `input` **clean** (borrowed-DFB only; the sharded-variant reader takes no src accessor), `cos`/`sin` **Case 1**, `output` Case 1 / borrowed as above.
  - `RotaryEmbeddingHfMultiCoreSharded` (both decode shapes): `input`/`cos`/`sin`/`output` all **clean** — borrowed-memory DFBs, zero address RTAs.
  - All `Buffer*` RTAs are the descriptor-API `BufferBinding` delivery form (patched on cache hits today) — routine port work, not a correctness hazard.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** drop the redundant page-size arg at the 9 sites listed in Gate detail (all Class 2; no `dynamic_tensor_shape` needed).
- **CB endpoints:** self-loop the compute-only CBs — `c_24`/`c_25`/`c_26` in every config; additionally `c_0`/`c_1`/`c_2` (in/cos/sin) and `c_16` (out) in both sharded-decode configs. Everything else binds one PRODUCER + one CONSUMER as the FIFO roles already dictate. No multi-binding flag, no dead-CB drop, no conditional DFB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no CB has more than two touchers on any node in any config.
- **Cross-op / shared kernels:** the single-tile prefill path instantiates `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/rotary_embedding_single_tile.cpp` (multi_core factory `:257-259` and `:273-275`) — a **borrowed** kernel owned by the sibling `rotary_embedding` op, which also binds it (its factory `:399,416`, including a `DECODE_MODE` define-path this op never enables). **No `_metal2` fork exists yet** beside it (`ls` of `../rotary_embedding/device/kernels/compute/`: only `rotary_embedding.cpp`, `rotary_embedding_single_tile.cpp`). Consumers: {`rotary_embedding`, `rotary_embedding_hf`} — a **sunset list, not authorization to convert in place** (see `port_patterns.md` — Caution: Porting a shared kernel). The sibling `rotary_embedding` op is being audited/ported in the same effort — whichever port lands first creates the fork; the other reuses it (rung 1).
- **RTA varargs:** none — every `get_arg_val` in every kernel is a distinct fixed-index scalar read (readers: 0–6 / 0–4; writer: 0–2). CTA varargs: none (all `get_compile_time_arg_val` at fixed constexpr indices). Port everything as named args.
- **Per-core-group duplicate compute KernelDescriptor:** the MultiCore factory instantiates the compute kernel twice over **disjoint** core groups with one differing CTA (`num_rows_per_core`: single-tile CTA[8] at `:271`, multi-tile CTA[9] at `:581`). Each node sees one instance → ordinary 1:1 bindings, *not* a two-toucher case; keep the per-group CTA per the demoting-per-group-CTA-to-RTA anti-pattern in `port_patterns.md`.
- **`OUT_SHARDED` define on the writer** (multi_core factory `:219-221,528-531`): the writer's accessor path is compiled out under it; carry the define conditionally.
- **In-sharded multi-tile prefill self-aliasing read:** when `in_sharded`, `c_0` is borrowed from `input.buffer()` *and* the reader NoC-reads src tiles via `TensorAccessor s0` into that same borrowed region (reader `:89-95`; rotated-half reads at `:70-80` land in the non-borrowed `c_1`). Legacy behavior — port it byte-for-byte; do not "optimize away" the copy.

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean** (no ⚠/✗/⭐ entries).

  | Op kernel | Donor file | Bucket | Status |
  |---|---|---|---|
  | `compute/rotary_embedding_hf_sharded.cpp` | `ttnn/cpp/ttnn/kernel/compute/dest_format_helpers.hpp` | 3 (shared pool, `ttnn/kernel/`) | ✓ — calls `copy_tile_init_with_dt(uint32_t icb, uint32_t transpose = 0)` (`dest_format_helpers.hpp:37`): `uint32_t cb_id` shape, `dfb::name` constexpr cast handles it |
  | borrowed `rotary_embedding_single_tile.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `untilize_helpers.hpp` | 2 (`kernel_lib`) | ✓ — `compute_kernel_lib::tilize/untilize` take CB ids as `uint32_t` NTTPs: `uint32_t cb_id` shape, constexpr cast works in template-parameter position. (Only the `DECODE_MODE` path even calls them, which this op never enables — relevant to the fork's conversion, not to this op's bindings) |
  | all kernels | `tt_metal` `api/*` headers (`dataflow_api.h`, `noc.h`, `circular_buffer.h`, `core_local_mem.h`, `noc_traits.h`, compute `api/*`) | 1 (LLK/HAL) | ✓ no concern |

  **Borrowed kernel files (file-path instantiation):** one — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/rotary_embedding_single_tile.cpp`, owned by in-family sibling `rotary_embedding`; co-instantiated by that op's own factory (`rotary_embedding_program_factory.cpp:399,416`). No `_metal2` fork exists (locational check done). Coordination/sunset set: {`rotary_embedding`, `rotary_embedding_hf`}; both are in the current audit batch, so the two porters should coordinate on who creates the fork and on its binding names.
- **Relaxation candidates:** none — no custom hash exists to mine.
- **TTNN factory analysis:** current concept `descriptor` (both factories); no op-owned tensors; no MeshWorkload need; no pybind `create_descriptor` or other internals-pybind (`rotary_embedding_hf_nanobind.cpp` binds the public function only); no custom hash (`Formerly custom hashed? = yes` on the sheet — the hash was already removed when the op moved to pointer-patching, nothing for the port to preserve); no `get_dynamic_runtime_args`; no `override_runtime_arguments` → target **`ProgramSpecFactoryConcept`**, framework-refreshed bindings.

## Misc anomalies  *(team-only, non-gating)*

- **Dead CTA:** `device/kernels/compute/rotary_embedding_hf_sharded.cpp:25,29` — CTA index 9 (`Ht`, fed `n_heads_t` by the sharded factory `:373`) is read and immediately `(void)Ht;`-discarded. Dead plumbing; the port must still keep arg-list positions consistent until the ops team removes it (or the porter names it and it naturally drops — porter's recipe governs).
- **Inert writer plumbing under `OUT_SHARDED`:** `device/kernels/dataflow/writer_rotary_embedding_hf_interleaved.cpp:15,17,23` — `dst_addr` (RTA 0), `start_id` (RTA 2) and the constructed `TensorAccessor s` are unused when `OUT_SHARDED` is defined (only `wait_front(num_tiles)` runs). Harmless today; noted for the ops team.
- **Self-aliasing NoC read in in-sharded multi-tile prefill** (also in Heads-ups): the reader re-reads resident shard tiles through the NoC into their own storage (borrowed `c_0` over `input.buffer()` + accessor reads of the same tensor). Functionally correct but wasteful; a candidate for an ops-team cleanup independent of the port.

## Recipe notes

- The readiness-sheet fetch doc says the Drive connector "authorizes only in the main interactive session; a spawned subagent hits the OAuth wall." This audit ran as a spawned agent and the fetch **succeeded** — the warning may be stale, or connector-dependent. Worth re-testing before agents keep routing fetches to the main session.
- Minor: the audit template's status-summary row *"Prereqs — Cross-op escapes"* has no owning subject that runs before the informational Out-of-directory coupling subject; on a GREEN audit this is moot, but on a whole-op RED (where that subject is skipped) the row would have no source. Consider deriving it from the Device 2.0 subject's donor findings explicitly.

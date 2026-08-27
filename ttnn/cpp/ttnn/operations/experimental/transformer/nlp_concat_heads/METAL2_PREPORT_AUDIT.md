# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads`

- **`NLPConcatHeadsDeviceOperation`** (`device/nlp_concat_heads_device_operation.hpp`)
  - `NLPConcatHeadsProgramFactory` (`device/nlp_concat_heads_program_factory.cpp`) — the op's single factory (sole member of `program_factory_t`), with two internal config branches: **interleaved** (`!in_sharded`) and **sharded** (`in_sharded`).

Kernels referenced (all in scope):

| Kernel | Owner | Used by config |
|---|---|---|
| `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` | this op (sole consumer, verified) | interleaved (reader) |
| `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` | this op (sole consumer, verified) | sharded (instantiated **twice**: Reader-config + Writer-config instances) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **borrowed** — eltwise/unary family, broadly shared | interleaved (writer) |

No unreferenced kernel files in the op directory. The sibling ops `nlp_concat_heads_boltz` and `nlp_concat_heads_decode` are separate ops with their own similarly-named kernels — they do **not** bind this op's kernel files (verified by path grep).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** version cannot be pinned — `git log -1 -- docs/.../metal_2.0/` prints nothing; the `metal_2.0` docs tree is untracked in this checkout (`??` in `git status`).

**Readiness sheet:** fetched fresh this session (2026-08-27) from the live "Operations analysis" sheet.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads` |
| **Overall** | **GREEN** — brief issued |
| **DOps / Factories** | `NLPConcatHeadsDeviceOperation` → `NLPConcatHeadsProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all three kernels (own reader, own sharded kernel, borrowed eltwise/unary writer) are structurally Device 2.0; only sanctioned free functions present |
| *Prereqs* — Cross-op escapes | Ok — function-call escapes are `tt_metal/hw/inc/api/**` only (bucket 1); one borrowed kernel *file* with an existing `_metal2` fork (see Team-only) |
| *Feature Support* — overall | **GREEN** — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok — none (fixed CTA sets + `TensorAccessorArgs`) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (verbatim `yes`; cross-check clean) |
| *TTNN Readiness* — Concept (current) | `descriptor` (verified: `create_descriptor()` returning `ProgramDescriptor`, `nlp_concat_heads_program_factory.cpp:19`) |
| *TTNN Readiness* — Secretly SPMD | N/A (not a `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | No (sheet `no`; no `compute_program_hash` / `attribute_values` / `to_hash` in device-op — verified) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet `no`; grep of device-op clean — verified) |
| *TTNN Readiness* — `override_runtime_arguments` | No (sheet `no`; grep clean — verified) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (sheet `no`; `nlp_concat_heads_nanobind.cpp` binds only the public op via `bind_function` — verified) |
| *TTNN Readiness* — Op-owned tensors | No (sheet blank; `descriptor` concept can't carry them — consistent) |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (matches the sheet's `Porting Target` column) |
| *Port work* — Offset base pointer | **none** — cleared (no host-folded offsets; sharded byte-offsets already passed as separate scalar args) |
| *Port work* — Tensor bindings (per binding) | input: **Case 1** (interleaved) / **clean** borrowed-DFB (sharded) · output: **Case 1** (interleaved) / **clean** borrowed-DFB (sharded) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (verbatim) — clears |
| *Port work* — TensorAccessor 3rd arg | **none — no accessor passes a 3rd arg** (both sites are 2-arg) |
| *Port work* — CB endpoints | interleaved: **legal 1:1** · sharded: two-toucher borrowed CBs — **1P+1C** is the true census, but vestigial dead `reserve_back` calls strictly lock both touchers as producers (see Gate detail + Questions) |

**CB endpoints** are dispositions, not gates: every out-of-window CB here has a port-time resolution. Detail below.

## Result

**GREEN → brief issued.** All five gates clear: Device 2.0 ✓, Feature compatibility ✓ (all Appendix A entries N/A), TTNN factory concept ✓ (`Is able to port?` = `yes`, cross-check clean, relaxation `none`), Offset base pointers ✓, TensorAccessor 3rd arg ✓ (no sites). One porter-facing open question (sharded-config dead FIFO sync — see *Questions for the user*) which affects the CB-endpoint disposition but does not gate.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Sheet row `experimental/transformer/nlp_concat_heads` / `NLPConcatHeadsDeviceOperation` / `NLPConcatHeadsProgramFactory`: `Is able to port?` = `yes`, `Concept` = `descriptor`, `Op Classification` = `PD Op (pointer-patching)`, `Known op issues` empty, `TensorParameter relaxation` = `none`, `Smuggled pointer` = `no`, `Pointer patching perf issue?` = `OK`. **Cross-check clean on every cheaply-checkable column** (concept, custom hash, backdoor hash, `get_dynamic_runtime_args`, `override_runtime_arguments`, pybind descriptor — all verified against the code and all agree). **Factory-set match:** one factory in code (`program_factory_t = std::variant<NLPConcatHeadsProgramFactory>`), one sheet row — ✓. No cross-column invariant violated.
- **Device 2.0 (every kernel used):** GREEN — no violations; the table below is empty.
  - `reader_tm_tile_layout_nlp_concat_heads.cpp` — `Noc`, `CircularBuffer` wrapper, `TensorAccessor`, `CoreLocalMem`; `get_tile_size(cb_id_in0)` (line 30) is **sanctioned**.
  - `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` — `Noc`, `CircularBuffer` wrappers, `UnicastEndpoint`, `CoreLocalMem`; `get_tile_size(cb_id_in0)` (line 30) is **sanctioned**. The raw `my_x[noc_id]` / `my_y[noc_id]` firmware globals (lines 39–40) are not a Device 2.0 violation — Device 2.0's own surface uses them (`tt_metal/hw/inc/api/dataflow/noc.h:160`, `api/tensor/tensor_accessor.h:224`) and the migration guide's migrated examples do too.
  - `writer_unary_interleaved_start_id.cpp` (borrowed, eltwise/unary) — `Noc`, kernel-side `DataflowBuffer`, `TensorAccessor`; `get_local_cb_interface(cb_id_out).fifo_page_size` (line 27) is **sanctioned**. Already part-modernized (kernel-side `DataflowBuffer`), so the port is a binding-layer change for it — and in fact a `_metal2` fork already exists (below).

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | *(none)* | | | |

- **Feature compatibility:** clean scan — all N/A.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no type/factory/field/`remote_index` signal anywhere in the op dir (grep clean) |
  | CBDescriptor `address_offset` (non-zero) | N/A | both `CBDescriptor` literals (`nlp_concat_heads_program_factory.cpp:142`, `:155`) leave `address_offset` unset (default 0) |
  | GlobalSemaphore | N/A | no signal; the op uses no semaphores at all |

- **CB endpoints (GATE-free):** classified per `(CB, config)`. Reachable intended configs are (interleaved-in, interleaved-out) and (sharded-in, sharded-out); the validation-reachable (sharded-in, interleaved-out) combination appears to be a pre-existing latent bug, not a port target (see *Misc anomalies*).
  - **`cb0` (index 0), interleaved:** regular double-buffered CB (`buffer = nullptr`). Reader FIFO-produces (`reserve_back`/`push_back`, reader kernel lines 47/59; its own `get_write_ptr` peek at line 41 rides the same PRODUCER binding — one toucher). Borrowed writer FIFO-consumes (`wait_front`/`pop_front`, writer lines 48/51, `OUT_SHARDED` undefined). **Legal 1:1** (1 locked producer + 1 locked consumer).
  - **`cb0` (index 0), sharded:** borrowed-memory CB (`.buffer = in0_buffer`, factory line 150). Two touchers — the two same-source instances of the sharded kernel (dual-instance work-split, factory lines 86–109: same `kernel_source`, `ReaderConfigDescriptor` vs `WriterConfigDescriptor`, both over `all_cores`). Each instance raw-peeks (`cb_in0.get_read_ptr()`, kernel line 43) **and** issues `cb_in0.reserve_back(block_size)` (kernel line 35, comment-marked `// Redundant`). Strictly per the census table, `reserve_back` is a FIFO-produce op → **2 locked producers → multi-binding flag**. But the locking calls are **dead sync**: a full-capacity reserve on an empty borrowed CB (block_size == CB capacity == shard tiles) that returns immediately and is never followed by a `push_back`. The true dataflow census is **two role-free touchers → 1P+1C**. See *Questions for the user*: stripping the dead lines (ops-team-scoped 3-line cleanup, off the porter's whitelist) resolves this cleanly; keeping them forces PRODUCER-capable bindings on both instances (an odd 2-producer/0-consumer shape).
  - **`cb16` (index 16), sharded only:** borrowed-memory CB (`.buffer = out_buffer`, factory lines 153–165). Same two touchers: each instance raw-peeks (`cb_out0.get_write_ptr()`, kernel line 44) and issues a dead `cb_out0.reserve_back(block_size)` (kernel line 36; the matching `push_back` is commented out at kernel line 62). Same disposition and same question as `cb0`/sharded.
  - **`cb16`, interleaved:** not allocated (host-side conditional `if (out_sharded)` already exists, factory line 153) — the DFB spec is naturally **conditional** in the port; this is a translation of an *existing* host conditional, not new structure.
- **Offset base pointers:** GREEN — no address RTA folds a host-side offset into its base. Interleaved config passes whole `Buffer*` objects (`in0_buffer` at factory line 201, `out_buffer` at line 210) with no arithmetic. Sharded config passes **no tensor addresses at all**; the byte offsets (`nheads_first_risc * in0_HtWt * single_tile_size` etc., factory lines 185–186) are separate scalar RTAs added kernel-side to CB base pointers (sharded kernel lines 43–44) — the already-split clean pattern. Op absent from the offset-base-pointer triage tables — consistent (*no fold, not in tables → clean*).
- **TensorAccessor 3rd argument:** **N/A — no accessor in the op passes a 3rd argument.** Both construction sites are 2-arg: `TensorAccessor(in0_args, in0_tensor_addr)` (reader kernel line 31) and `TensorAccessor(dst_args, dst_addr)` (borrowed writer line 39). Op absent from the 3rd-arg triage table — consistent.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per config — classification varies per factory branch):
  - `input` — interleaved: **Case 1** — `Buffer*` delivered via reader RTA 0 (factory line 201; `Buffer*`-binding form, framework-patched today), kernel feeds it into `TensorAccessor` (reader kernel lines 17→31). Sharded: **clean** — borrowed-memory CB (`cb0` `.buffer = in0_buffer`) → `DataflowBufferSpec::borrowed_from`.
  - `output` — interleaved: **Case 1** — `Buffer*` delivered via writer RTA 0 (factory line 210), kernel feeds it into `TensorAccessor` (writer lines 19→39). Sharded: **clean** — borrowed-memory CB (`cb16` `.buffer = out_buffer`) → `borrowed_from`.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** `cb0`/interleaved legal 1:1 · `cb0`/sharded + `cb16`/sharded: **1P+1C** (two role-free touchers) *pending the dead-sync question*; strictly-by-census fallback is the multi-binding flag with 2 locked producers · `cb16` conditional DFB (exists only under `out_sharded` — translate the existing host conditional, do not drop).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (sharded config):** both borrowed CBs are two-toucher, dual-instance work-split shapes (face (c)) — no hidden second writer beyond the two visible instances, no third toucher (the op has **no compute kernel** in either config). The only wrinkle is the vestigial `reserve_back` pair (sharded kernel lines 35–36) locking both touchers to the producer role; see Questions.
- **Cross-op / shared kernels:** the interleaved writer is borrowed from eltwise/unary and a **checked-in `_metal2` fork already exists beside it**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` — rung 1: bind it, don't re-fork, adopt its vocabulary (`dfb::out`, `tensor::dst`, `args::num_pages`, `args::start_id`). It fits this op exactly (writer RTAs = dst addr / num_pages / start_id; no `OUT_SHARDED`/`BACKWARDS` defines needed on this op's interleaved path). The fork owns the binding names — the factory conforms; in particular `cb0` must be bound as `dfb::out` on the writer side. A **second duplicate fork** exists at `copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` (names the accessor `tensor::output`) — bind the eltwise/unary one, per the fork's own header note.
- **RTA varargs:** none — every RTA in all three kernels is a fixed, distinct, nameable scalar (no counted loops, no data-selected reads). No CTA varargs either.
- **Op-owned kernels are not shared:** both `reader_tm_tile_layout_nlp_concat_heads*.cpp` files are bound only by this op's factory (verified) — they convert in place as ordinary port work; no fork needed.
- **Dual instantiation detail:** the sharded config instantiates one kernel source twice with the *same* CTA vector (factory lines 87–108: reader copies it, writer takes it by move) and differing RTAs (`{nheads_first_risc, 0, 0}` vs `{nheads_second_risc, read_off, write_off}`). The reader instance's two zero offsets are real named args (`start_read_offset_bytes`, `start_write_offset_bytes`) that happen to be 0.
- **Sharded kernel local-copy idiom:** it does an L1→L1 self-copy via `UnicastEndpoint` addressed with `my_x[noc_id]`/`my_y[noc_id]` (kernel lines 38–44) — ports as-is; not a Device 2.0 issue.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up: ✓ clean** (function-call escapes). Every kernel `#include` resolves to `tt_metal/hw/inc/api/**` (bucket 1 — LLK/HAL/framework; no concern): `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/dataflow/dataflow_buffer.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`. No donor *function* calls → per-call detail omitted.
  - **Borrowed kernel file (file-path instantiation):** `writer_unary_interleaved_start_id.cpp` (owner: eltwise/unary). Broadly shared; `_metal2` fork exists beside it (see Heads-ups). **Sunset list** (legacy-copy binders found by grep, excluding quasar; tracked in issue #52228) — *coordination/sunset data, not a must-port-together bundle and not authorization to convert in place*: data_movement/concat, data_movement/reshape_on_device, data_movement/slice (tile factories), data_movement/tilize (5 factories), data_movement/transpose (wh, hc_tiled), eltwise/unary_backward (gelu_bw, tanh_bw), embedding (fused), examples/example (2), experimental/matmul/attn_matmul, experimental/transformer/nlp_concat_heads (this op), experimental/transformer/nlp_concat_heads_boltz, matmul (multicore), reduction/generic (4 factories).
- **Relaxation candidates:** none — no custom hash to mine.
- **TTNN factory analysis:** current concept `descriptor` (`create_descriptor` at `nlp_concat_heads_program_factory.cpp:19`); no op-owned tensors; no MeshWorkload need; no pybind of internals (`nlp_concat_heads_nanobind.cpp` binds only the public `ttnn::experimental::nlp_concat_heads` function); no custom hash; no `get_dynamic_runtime_args`; no `override_runtime_arguments` → **target `ProgramSpecFactoryConcept`**. Sheet also notes `ProgramFactory used in llama?` = `yes` and `Uses llama kernels?` = `yes` (informational: this op is on a llama model path — regression coverage matters).

## Misc anomalies  *(team-only, non-gating)*

- **Latent broken config — sharded input + interleaved output.** Validation permits it (`nlp_concat_heads_device_operation.cpp:48–51` only forbids HEIGHT_SHARDED output when input is sharded), and `compute_output_specs` would produce an interleaved output — but the factory then takes the sharded kernel branch while creating no `cb16` (`nlp_concat_heads_program_factory.cpp:153` gates it on `out_sharded`), so both kernel instances would touch an unconfigured CB index 16, and nothing would ever write the interleaved output buffer. Pre-existing (legacy behavior identical); routes to the ops team — either forbid the config in validation or implement it.
- **Dead FIFO sync in the sharded kernel** (`reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`): `cb_in0.reserve_back(block_size)` at line 35 is self-annotated `// Redundant`; `cb_out0.reserve_back(block_size)` at line 36 is likewise a full-capacity no-op; the matching `cb_out0.push_back(block_size)` is commented out at line 62. Dead code with port-shaping consequences (see CB endpoints).
- **Stale comments in the factory:** shape-specific literals from a past model (`// 142` at line 36, `// Output shape is: [B, 1, s, 4544]` at line 39) and a `Grayskull Device Setup` banner (line 73). Cosmetic.
- Both reader kernels label their runtime args `// WRITER RUNTIME ARGS` (reader kernel line 16 / sharded kernel line 15). Cosmetic copy-paste.

## Questions for the user

1. **Strip the dead FIFO sync in the sharded kernel pre-port?** `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp:35–36` (`reserve_back` ×2; the paired `push_back` at line 62 is already commented out). These are functionally dead (full-capacity reserve on an empty borrowed CB) but, read strictly, lock **both** kernel instances to the PRODUCER role on **both** sharded CBs — forcing the multi-binding advanced option in a 2-producer/0-consumer shape instead of the natural 1P+1C. A 3-line ops-team-scoped cleanup (off the porter's whitelist) reduces both CBs to plain role-free 1P+1C. Recommend doing it before or alongside the port with explicit approval; otherwise the porter should attempt 1P+1C only if a CONSUMER-bound `reserve_back` is legal, else set the flag.

## Recipe notes

- **CB endpoints census table doesn't anticipate *dead* FIFO ops.** The sharded kernel's vestigial `reserve_back` calls make both touchers "locked producers" by the letter of the rule (`reserve_back`/`push_back` ⇒ locked producer), yielding a 2-producer/**0-consumer** multi-binding — a shape the table's resolutions don't obviously cover (is a DFB with producers but no consumer even expressible?). The dual-instance work-split face (c) asserts co-touches are "sync-free by construction", which is nearly true here but defeated by dead sync the author left in. A guard like *"a FIFO op that is provably a no-op in every config (full-capacity reserve on a borrowed CB, never paired with a push/pop) does not lock a role — but flag it as a question rather than silently relabelling"* would have resolved this without a judgment call.
- **Provenance command prints nothing on an untracked docs tree.** The `metal_2.0` docs directory is untracked here (`?? docs/.../metal_2.0/` in `git status`), so the provenance line can't be pinned; recorded per the fallback instruction.

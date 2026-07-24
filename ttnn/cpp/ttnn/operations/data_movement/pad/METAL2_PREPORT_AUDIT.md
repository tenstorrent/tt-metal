# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/pad`

- **`PadDeviceOperation`** (single device-operation in the directory)
  - `PadRmReaderWriterMultiCoreDefaultProgramFactory` (`pad_rm_reader_writer_multi_core_default_program_factory.cpp`) — concept `descriptor`
  - `PadRmReaderWriterProgramFactory` (`pad_rm_reader_writer_program_factory.cpp`) — concept `WorkloadDescriptor` (secretly SPMD, op-owned tensor)
  - `PadRmShardedWidthOnlyProgramFactory` (`pad_rm_sharded_width_only_program_factory.cpp`) — concept `descriptor`
  - `PadTileCoreProgramFactory` (`pad_tile_program_factory.cpp`) — concept `descriptor`
  - `PadTileMulticoreProgramFactory` (`pad_tile_multicore_program_factory.cpp`) — concept `descriptor`
  - `PadRmShardedHeightOnlyProgramFactory` (`pad_rm_sharded_height_only_program_factory.cpp`) — concept `descriptor` — **GATED**
  - `PadRmReaderWriterMultiCoreProgramFactory` (`pad_rm_reader_writer_multi_core_program_factory.cpp`) — concept `WorkloadDescriptor` — **unreachable / dead** (declared in the `program_factory_t` variant but never returned by `select_program_factory`; see Misc anomalies)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `8f7eb3e47dc 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/pad` |
| **Overall** | **RED (config-scoped)** — 1 of 7 factories gated; a clean subset survives |
| **DOps / Factories** | `PadDeviceOperation` → 7 factories (6 reachable), 1 gated (`PadRmShardedHeightOnly`), 1 dead (`PadRmReaderWriterMultiCore`) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 12 kernels (own + eltwise/unary donor) are Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | Ok — in-family `data_movement/common` helper (Noc/TensorAccessor shapes) + one cross-family donor, both Device 2.0 |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok — every `get_compile_time_arg_val` uses a literal index; no runtime-varying CTA loop |
| *TTNN Readiness* — `Is able to port?` (the gate) | **No for `PadRmShardedHeightOnly`**: `Runtime-args update == yes` **and** `Is safe to port? == no`. **Yes** for the other 6 factories |
| *TTNN Readiness* — Concept (current) | `descriptor` (5) + `WorkloadDescriptor` (2, both secretly SPMD) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | Yes (both `WorkloadDescriptor` factories — single-program descriptor replicated across mesh coords) |
| *TTNN Readiness* — Is safe to port? | **No** for `PadRmShardedHeightOnly` (→ readiness-sheet owner); Yes for the rest |
| *TTNN Readiness* — Custom hash | No (all factories; no `compute_program_hash` override in the directory) |
| *TTNN Readiness* — Runtime-args update | **Yes** for `PadRmShardedHeightOnly` only (`get_dynamic_runtime_args` in `pad_device_operation.cpp:235`, active only for that factory); No for the rest |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind binds only the `ttnn::pad` host overloads) |
| *TTNN Readiness* — Op-owned tensors | **Yes** for the 2 `WorkloadDescriptor` factories (pad-value const tensor parked on `wd.buffers`); No for the `descriptor` factories |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (+ op-owned tensors for the 2 `WorkloadDescriptor` factories) |
| *Port work* — Offset base pointer | **none** — every address arg is a clean `Buffer*` base (no offset fold) |
| *Port work* — Tensor bindings (per binding) | Case 1 (interleaved/tiled factories) · clean/borrowed (width-only sharded) |
| *Port work* — TensorParameter relaxation | none (sheet: `none`; no custom hash) |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2 redundant) — default MC factory only |
| *Port work* — CB endpoints | 1:1 / self-loop / **1 dead-CB drop** (tile-MC `output_cb`) — no multi-binding |

**CB endpoints** are dispositions, not gates. Every out-of-window CB here resolves at port time (self-loop or dead-CB drop); no multi-binding flag is required. Recorded per `(CB, config)` below.

## Result

**RED at op level; subset {`PadRmReaderWriterMultiCoreDefault`, `PadRmReaderWriterProgram`, `PadRmShardedWidthOnly`, `PadTileCore`, `PadTileMulticore`} is clear.**

The single blocker is the **TTNN factory-concept gate on `PadRmShardedHeightOnlyProgramFactory`** — its `Is able to port? == no` because it carries the `get_dynamic_runtime_args` **Runtime-args-update** hook (a shape gate that lifts when the TTNN runtime-args-update infra ships) **and** the readiness-sheet owner marks it **`Is safe to port? == no`** (a correctness call). That factory is the height-sharded RM *optimization* path; it shares no kernels with the rest of the op and is cleanly severable. All five gate-bearing subjects are otherwise GREEN across the whole op, so a subset port of the other factories can proceed now. A `METAL2_PORT_BRIEF.md` is issued for the clean subset.

> **Forward path (reassurance).** Neither leg of this RED is a permanent blocker. The Runtime-args-update leg is *missing-TTNN-infrastructure* — it clears when the runtime-args-update path lands (or when the height-sharded fast-path opt-in in `get_dynamic_runtime_args` is reworked). The `Is safe to port?` leg is a readiness-sheet-owner correctness call to reconcile. The other six factories are portable today.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED (config-scoped).** Per-factory verdicts from the readiness sheet (fetched fresh this run; cross-checked against code):

  | Factory | Concept | Custom hash | Runtime-args update | Pybind desc | Is safe? | **Is able to port?** |
  |---|---|---|---|---|---|---|
  | `PadRmReaderWriterMultiCoreDefault` | descriptor | no | no | no | yes | **yes** |
  | `PadRmReaderWriterProgram` | WorkloadDescriptor | no | no | no | yes | **yes** |
  | `PadRmShardedWidthOnly` | descriptor | no | no | no | yes | **yes** |
  | `PadTileCore` | descriptor | no | no | no | yes | **yes** |
  | `PadTileMulticore` | descriptor | no | no | no | yes | **yes** |
  | `PadRmShardedHeightOnly` | descriptor | no | **yes** | no | **no** | **no** ← gated |
  | `PadRmReaderWriterMultiCore` (dead) | WorkloadDescriptor | no | no | no | yes | yes (but unreachable) |

  **Failing conjuncts for `PadRmShardedHeightOnly`:** (1) `Runtime-args update == yes` — a **shape** failure → **TTNN / ProgramDescriptor-migration team** (lifts when runtime-args-update support lands); (2) `Is safe to port? == no` — a **correctness** failure → **readiness-sheet owner** (Diego) to reconcile.

  **Cross-check (trust-but-verify) — clean:**
  - `Concept`: confirmed by factory methods — 5 factories define `create_descriptor()→ProgramDescriptor` (descriptor), 2 define `create_workload_descriptor()→WorkloadDescriptor`. ✓ matches sheet.
  - `Custom hash`: no `compute_program_hash` override anywhere in the directory. ✓ all `no`.
  - `Runtime-args update`: `get_dynamic_runtime_args` exists at DeviceOperation level (`pad_device_operation.cpp:235`) but early-returns `{}` unless the selected factory is `PadRmShardedHeightOnly` (`pad_device_operation.cpp:243-248`). The sheet's per-factory attribution (`yes` only for height-only) matches the code. ✓
  - `Pybind descriptor`: `pad_nanobind.cpp` binds only the `ttnn::pad` host overloads — no `create_descriptor` / device-op `nb::class_`. ✓ all `no`.
  - `Secretly SPMD` (the 2 `WorkloadDescriptor` factories): each `create_workload_descriptor` builds **one** `ProgramDescriptor` and replicates it across `tensor_coords.ranges()` (`pad_rm_reader_writer_program_factory.cpp:205-211`) — morally single-program. ✓ `yes`.
  - Cross-column invariants: `Op-owned tensors? == yes` occurs only on the `WorkloadDescriptor` rows (`wd.buffers.push_back(...)`, `pad_rm_reader_writer_program_factory.cpp:200`); `Runtime-args update == yes` occurs on a `descriptor` row (allowed). No violation. ✓

- **Device 2.0 (every kernel used):** **GREEN.** All 12 referenced kernels use Device 2.0 idioms (`Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem`, `UnicastEndpoint`); no `InterleavedAddrGen*`/`ShardedAddrGen`, no raw `noc_async_*`, no raw sem addresses. Sanctioned free functions in use are **not** violations: `get_tile_size(cb_id)` (`writer_unary_pad_dims_interleaved.cpp:28`) and `get_local_cb_interface(cb_id).fifo_page_size` (donor `reader_unary_interleaved_start_id.cpp:20`). Kernels covered:

  | Kernel | Used by (factory) | Device 2.0 |
  |---|---|---|
  | `reader_pad_dims_rm_interleaved_v2.cpp` / `writer_..._v2.cpp` | default MC (descriptor) | ✓ |
  | `reader_pad_dims_rm_interleaved.cpp` / `writer_...interleaved.cpp` | single-core RM (WD) + dead MC | ✓ |
  | `reader_pad_dims_rm_sharded_stickwise.cpp` / `writer_...stickwise.cpp` | width-only (descriptor) | ✓ |
  | `reader_pad_tiled.cpp` / `writer_pad_tiled.cpp` | tile-MC (descriptor) | ✓ |
  | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` (donor) | tile-core (descriptor) | ✓ |
  | `writer_unary_pad_dims_interleaved.cpp` | tile-core (descriptor) | ✓ |
  | `reader_pad_dims_rm_sharded.cpp` / `writer_pad_dims_rm_sharded.cpp` | height-only (**gated**) | ✓ |

- **Feature compatibility:** every Appendix A entry N/A. No `GlobalCircularBuffer` / `global_circular_buffer` field / `remote_index` / remote-CB idioms; no `CBDescriptor.address_offset` (all borrowed-memory CBs use `.buffer =` with implicit zero offset); no `GlobalSemaphore` (no semaphores at all); no CTA varargs (all CTA reads at literal indices; `tensor_args_t = PadInputs` carries a single `Tensor`, not a `std::vector<Tensor>` — the `std::vector<Tensor>` in the file is only the perf-model signature at `pad_device_operation.cpp:58`).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no GCB type / field / remote-CB idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | borrowed CBs set `.buffer` with implicit zero offset |
  | GlobalSemaphore | N/A | op uses no semaphores |
  | Variable-count compile-time arguments (CTA varargs) | N/A | all CTA reads at literal indices; single-tensor `tensor_args_t` |

- **CB endpoints (GATE-free):** every CB resolves at port time; no multi-binding flag needed. Per `(CB, config)`:

  | Factory | CB | Touchers | Disposition |
  |---|---|---|---|
  | default MC | `c_0` (data) | reader produce, writer consume | **1:1 legal** |
  | default MC | `c_1` (`cb_pad`) | reader raw-fill + loop-back read | **self-loop** |
  | default MC | `c_2` (`dfb_pad_align`) | reader raw-write/read | **self-loop** — allocated only when `stick_size_padded_front != 0 \|\| unaligned` (`..._default_...cpp:128`) |
  | single-core RM | `c_0` (data) | reader produce, writer consume | **1:1 legal** |
  | width-only | `c_0` (input shard, borrowed) | reader raw-read only | **self-loop** |
  | width-only | `c_16` (output shard, borrowed) | writer locked-producer (`reserve_back`/`push_back`), reader locked-consumer (`wait_front`/`pop_front`) | **1P+1C (plain 1:1)** — writer pre-fills padding, reader overwrites data region |
  | width-only | `c_1` (`pad_val`) | writer raw-fill + read | **self-loop** |
  | tile-core | `c_0` (data) | donor reader produce, writer consume | **1:1 legal** |
  | tile-core | `c_1` (pad buffer) | writer `reserve_back` + raw fill/read | **self-loop** |
  | tile-MC | `c_0` (`input_cb`) | reader produce, writer consume | **1:1 legal** |
  | tile-MC | `c_1` (`output_cb`) | **none** — `output_cb_id` (`writer_pad_tiled.cpp:23`, CTA 1) read into a constexpr, never used; output goes straight to the tensor via `TensorAccessor` | **DEAD-CB DROP** (see Port-work; also drop dead CTA 1) |
  | tile-MC | `c_2` (`pad_val_cb`) | writer `reserve_back`/`push_back` + raw fill/read | **self-loop** |

- **Offset base pointers:** **GREEN.** Every input/output/pad-value address reaches a kernel as a clean `Buffer*` base with **no host-folded offset**. All factories use the `Buffer*`-binding (`BufferBinding`) form and the source comments explicitly state "raw buffer base address (**no offset**)" (`..._default_...cpp:190-194`, `..._program_factory.cpp:136-140`, `pad_tile_program_factory.cpp:119-121`, `pad_tile_multicore_...cpp:214-217`). Where a kernel-side stick/page offset exists it rides a **separate** arg and is applied via the `TensorAccessor` `offset_bytes` field, never folded into the base. Width-only passes empty runtime-arg lists (addresses ride the borrowed CBs). Pad appears in **neither** the offset-base-pointer triage table nor the 3rd-arg triage table — scanned from first principles and clean. (Type 3/`address_offset` = N/A; Type 4/`narrow` = absent.)

- **TensorAccessor 3rd argument:** **GREEN — Class 2 (redundant), drop.** Only the **default MC factory** passes a 3rd arg: `accessor_page_size` on `reader_pad_dims_rm_interleaved_v2.cpp:95` (CTA 21) and `writer_..._v2.cpp:25` (CTA 4). Classification from the two questions:
  - *Sharded or interleaved?* This factory handles both. Interleaved: `input_accessor_page_size = stick_size` (`= W * element_size`), `output = stick_size_padded`. Sharded fallback: `= a.buffer()->aligned_page_size()` / `output.buffer()->aligned_page_size()` (`..._default_...cpp:57-73`).
  - *Correct or wrong magnitude?* Interleaved value equals the logical row-major page (`buffer->page_size()`), and the interleaved accessor realigns `align_power_of_2(page_size, alignment)` → **inert**. Sharded value **is** `aligned_page_size()` — exactly what Metal 2.0 supplies implicitly. Both correct-magnitude → **Class 2 redundant** → drop the arg (Metal 2.0 auto-supplies `aligned_page_size`). No Class 1 (no relaxation/custom hash; page size is fixed per program by the default hash), no Class 3/4/Special. No other clean-subset kernel passes a 3rd arg; the gated height-only kernels use no `TensorAccessor`.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory — all Case 1 or clean; no Case 2, no raw-pointer bridge needed):
  - **default MC** (descriptor): `input` Case 1, `output` Case 1. Delivered via `Buffer*` binding (`..._default_...cpp:221-222`), consumed through `TensorAccessor`.
  - **single-core RM** (WorkloadDescriptor): `input` Case 1, `output` Case 1, **pad-value const (op-owned)** Case 1 (arg 13 `Buffer*` binding, `..._program_factory.cpp:143,144,156`; the const tensor is parked on `wd.buffers`).
  - **width-only** (descriptor): `input` **clean** (borrowed-memory DFB, `cb_input.buffer = input_buffer`), `output` **clean** (borrowed-memory DFB, `cb_output.buffer = output_buffer`) → mechanical `DataflowBufferSpec::borrowed_from`.
  - **tile-core** (descriptor): `input` Case 1, `output` Case 1 (`Buffer*` bindings, `pad_tile_program_factory.cpp:122,126`).
  - **tile-MC** (descriptor): `input` Case 1, `output` Case 1 (`Buffer*` bindings, `pad_tile_multicore_...cpp:222-223`).
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** drop the redundant `accessor_page_size` arg — default MC factory only (`reader_..._v2.cpp:95` CTA 21, `writer_..._v2.cpp:25` CTA 4).
- **CB endpoints:** self-loop the single-toucher scratch/borrowed CBs; assign 1P+1C on width-only `c_16`; drop dead CB `c_1` (`output_cb`) in tile-MC and its dead CTA 1 in `writer_pad_tiled.cpp`. All other CBs are legal 1:1.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no ≥3-toucher CB. The width-only `c_16` is a genuine producer/consumer FIFO handshake (writer pre-fills padding, reader overwrites the data region), not a multi-binding. No flag.
- **Cross-op / shared kernels:**
  - *File-path donor (cross-family):* tile-core instantiates `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`. **Broadly shared** — ~17 ops instantiate it (typecast, prod, copy, untilize, untilize_with_unpadding, nlp_create_qkv_heads_falcon7b, examples, several quasar variants, and pad). Its Metal 2.0 rewrite is a **single, coordinated change** across the whole port-together set.
  - *In-family helper (function-call escape):* `reader/writer_..._v2.cpp` `#include "ttnn/operations/data_movement/common/kernels/common.hpp"` and call `noc_async_read_sharded` / `noc_async_write_sharded`, whose signatures take `Noc` + a templated `TensorAccessor` (Device 2.0 native shapes) → clean; port the `data_movement/common` shared header alongside the family.
- **RTA varargs:** the tiled kernels (`reader_pad_tiled.cpp:22-25`, `writer_pad_tiled.cpp:35-38`) read four **rank-length** per-dim arrays (`input_page_shape` / `output_page_shape` / `input_id_per_dim` / `output_id_per_dim`) as one `get_arg_addr` block bounded by the `num_dims` CTA; the block length varies with tensor rank. Port these as an RTA vararg region rather than naming each element (the three leading scalars — base addr, page count, start offset — are ordinary named args). Not a gate (RTA/CRTA varargs are supported in Metal 2.0).

## Team-only

- **Out-of-directory coupling & donor shape:**
  - *Op-level roll-up:* ✓ clean (no ⚠/✗/⭐ shapes; all donor signatures are Device 2.0 native).
  - *Summary table:*

    | Op kernel | Donor file | Class | Shape / status |
    |---|---|---|---|
    | `reader/writer_..._v2.cpp` | `data_movement/common/kernels/common.hpp` | in-family shared | `noc_async_read_sharded`/`noc_async_write_sharded(Noc, uint32_t l1_addr, TensorAccessor, …)` → ✓ `TensorAccessor` + `Noc` (Device 2.0 native) |
    | `pad_tile_program_factory.cpp` (file-path) | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | cross-family donor (file-path instantiation) | Device 2.0 compliant; broadly-shared (~17 ops) → port-together set |
  - *Borrowed kernel files (file-path instantiation):* only the eltwise/unary donor above (pad owns and instantiates all other kernels). Port-together set: the ~17 ops listed under Heads-ups.
- **Relaxation candidates (FYI-U):** none mined — no custom hash on any factory.
- **TTNN factory analysis (sheet-derived, `file:line` confirmed):** custom hash — absent (all factories); custom `override_runtime_arguments` — absent; pybind `create_descriptor` — absent; op-owned tensors — present on the 2 `WorkloadDescriptor` factories (pad-value const tensor, `..._program_factory.cpp:197-200`); genuine multi-program — none (both WD factories secretly SPMD); Runtime-args-update hook — present but active only for the gated height-only factory.

## Misc anomalies  *(team-only, non-gating)*

- **Dead / unreachable factory:** `PadRmReaderWriterMultiCoreProgramFactory` (`pad_rm_reader_writer_multi_core_program_factory.cpp`) is declared in the `program_factory_t` variant (`pad_device_operation.hpp:34`) and fully defines `create_workload_descriptor` (`...cpp:404`), but `select_program_factory` (`pad_device_operation.cpp:70-112`) never returns it — the row-major sharded fallback and the multicore RM path both route to `PadRmReaderWriterMultiCoreDefaultProgramFactory` instead. The readiness sheet lists it `Is able to port? = yes`, but it is unreachable in practice. Its kernels (`reader/writer_pad_dims_rm_interleaved.cpp`) are still live via `PadRmReaderWriterProgramFactory`, so they are not dead. Recommend the ops team confirm and remove the dead variant.
- **Dead CB + dead CTA in tile-MC:** `output_cb` (`c_1`) is allocated in `pad_tile_multicore_program_factory.cpp:70-78` and its index is threaded to the writer as CTA 1, but `writer_pad_tiled.cpp:23` reads it into an unused `constexpr` — no kernel ever touches `c_1`. Burns `page_size * 2` of L1 per core for nothing. (Reported as a dead-CB drop in Port-work; noted here as the underlying latent waste for the ops team.)
- **Hardcoded 64-byte pad-const assumption:** `reader_pad_dims_rm_interleaved.cpp:52` — `pad_value_const_buffer_nbytes = 64; // assumed to be 64 bytes, fails on BH when > 64. TODO: generalize? (Issue #21978)`. A known Blackhole limitation tracked in #21978; not port work, but the porter should not "fix" it silently.

## Questions for the user  *(omit if none)*

None blocking. (The two `Is able to port? = no` conjuncts on the height-only factory route to their owning teams per Gate detail; no auditor ambiguity remained on any gate.)

## Recipe notes

- **DeviceOperation-level `get_dynamic_runtime_args` vs. per-factory gate.** The Runtime-args-update hook lives on the *DeviceOperation* (`pad_device_operation.cpp:235`) but is functionally scoped to one factory (early-returns `{}` for the rest). The readiness sheet correctly attributes `Runtime-args update = yes` to only the height-only row, and the audit's config-scoped-gate machinery handles this cleanly — but the recipe's cross-check bullet ("grep the factory for `get_dynamic_runtime_args`") reads as *per-factory*, whereas the hook is a *DeviceOperation* method whose effect is per-factory only by an internal branch. A one-line note in the cross-check that this hook may be DeviceOp-level but factory-scoped-by-branch would save the next auditor a moment of "the grep hits the DeviceOp, not the factory."
- **Config-scoped RED still emits a brief — confirmed applied.** The recipe's "config-scoped GATE still issues a brief for the clean subset" rule (Output section) is what governs here; flagging it because the top-level "RED → no brief" phrasing in several places could mislead a hurried reader into skipping the brief on a config-scoped RED. This audit issues `METAL2_PORT_BRIEF.md` for the 5 clean reachable factories.

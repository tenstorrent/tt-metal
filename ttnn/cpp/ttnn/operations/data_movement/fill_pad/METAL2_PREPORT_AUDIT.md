# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/fill_pad`

- **`FillPadDeviceOperation`** (`device/fill_pad_device_operation.hpp`)
  - `FillPadProgramFactory` — DRAM interleaved + DRAM-sharded (`device/fill_pad_program_factory.cpp:20`)
  - `FillPadL1ShardedProgramFactory` — L1 HEIGHT/WIDTH/BLOCK sharded (`device/fill_pad_program_factory.cpp:310`)

Single device operation, two program factories sharing one compute kernel — audited together as one porting unit (one combined report).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `56373090d3d 2026-08-05 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

**Readiness sheet:** *"TTNN Operations analysis"* (`1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`), fetched fresh via the Google Drive connector this run (2026-08-05).

> **Context — this is a re-port after a revert.** The Metal 2.0 port of these factories landed as **#50904** (`83983e08bfa`) and was **reverted as #51605** (`af2c5ce1d0f`) after it broke sanity tests. The revert restored *both* the factories and the kernels to their pre-port state, so the code audited here is the clean, working post-revert baseline — `ProgramDescriptor` host + Device-2.0 kernels — identical to `main`. The revert's root cause is captured as a porter watch-for (see [Heads-ups](#heads-ups)); it is **not** a gate on the current code.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/fill_pad` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `FillPadDeviceOperation` → `FillPadProgramFactory`, `FillPadL1ShardedProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 5 kernels use `Noc` / `DataflowBuffer` / `TensorAccessor` / `UnicastEndpoint`; only sanctioned `get_tile_size(cb)` + `dfb.get_write_ptr()` methods |
| *Prereqs* — Cross-op escapes | Ok — no out-of-directory kernel includes; op owns all its kernels |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — Variadic-CTA | Ok — all CTA reads at constexpr literal indices |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factories) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (not a `WorkloadDescriptor` op) |
| *TTNN Readiness* — Is safe to port? | Yes (both) |
| *TTNN Readiness* — Custom hash | No (both; no `compute_program_hash`, no backdoor `attribute_values`/`to_hash`) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (both) |
| *TTNN Readiness* — `override_runtime_arguments` | No (both) |
| *TTNN Readiness* — Pybind `create_descriptor` | No — nanobind binds only the `fill_implicit_tile_padding` free function |
| *TTNN Readiness* — Op-owned tensors | No (both) |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (both) |
| *Port work* — Offset base pointer | none (clean bases) |
| *Port work* — Tensor bindings (per binding) | `input` (in-place): **Case 1** in `FillPadProgramFactory`, **Case 2** in `FillPadL1ShardedProgramFactory` |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2) — `FillPadProgramFactory` reader + writer |
| *Port work* — CB endpoints | legal — every CB is plain 1:1 in both factories |

**CB endpoints** are dispositions, not gates. Here every CB (`c_0` data-in, `c_1` right-mask, `c_2` bottom-mask, `c_16` data-out) is a genuine 1-producer/1-consumer FIFO on every node in every config — no self-loop, 1P+1C assignment, multi-binding, or dead-CB drop is needed.

## Result

**GREEN → brief issued.** All five gates clear: Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (both factories) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓. Port work is confined to the two mechanical tensor-binding cases (Case 1 in the DRAM factory, Case 2 in the sharded factory) and dropping a redundant `TensorAccessor` 3rd argument. See `METAL2_PORT_BRIEF.md`.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** for both factory rows. The readiness sheet's `Is able to port?` == `yes` for `FillPadProgramFactory` and `FillPadL1ShardedProgramFactory`, and the cross-check is clean:
  - `Concept == descriptor` — both factories define `create_descriptor()` returning a `tt::tt_metal::ProgramDescriptor` (`fill_pad_program_factory.hpp:97`, `:105`). Confirmed.
  - `Custom hash == no` — no `compute_program_hash` override, and no backdoor `attribute_values`/`to_hash`, anywhere in the op. Confirmed by grep.
  - `get_dynamic_runtime_args == no` / `override_runtime_arguments == no` — neither hook present on `FillPadDeviceOperation`. Confirmed by grep.
  - `Pybind descriptor == no` — `fill_pad_nanobind.cpp:37` binds only the `fill_implicit_tile_padding` function; no `nb::class_` of the device op, no `create_descriptor` binding. Confirmed.
  - `Smuggled pointer == no` / `Is safe to port? == yes` — trusted (the readiness-sheet owner's correctness axis). Consistent with the code: both factories deliver the tensor base via the framework's `Buffer*`-binding form (`emplace_runtime_args({tens_buffer, …})`, `fill_pad_program_factory.cpp:293-295`, `:620-623`), which is patched on cache hits — not a raw `->address()` smuggle.
  - **Factory-set match** — the sheet has exactly two rows for this op, one per code factory; the `program_factory_t` variant (`fill_pad_device_operation.hpp:19`) lists exactly those two. One-to-one; no phantom or missing row.
  - **Cross-column invariants** — `Op-owned tensors?` blank on a `descriptor` op (consistent); `get_dynamic_runtime_args == no`. No inconsistency.
- **Device 2.0 (every kernel used):** **GREEN.** All five referenced kernels are structurally Device 2.0 and use no legacy Device-1.0 idioms (no `noc_async_read/write`, no `InterleavedAddrGen`/`ShardedAddrGen`, no raw `CircularBuffer`, no CB-index free-function holdovers). They use `Noc`, `DataflowBuffer`, `TensorAccessor`, and (sharded) `UnicastEndpoint`. The only CB-index free function is `get_tile_size(cb_id)` (sanctioned per the Green bullet). The `get_write_ptr()` calls in `fill_pad_dataflow_common.hpp:41,52` are `dfb.get_write_ptr()` **methods** on `DataflowBuffer` objects — the producer writing into its own reserved slot — not the free-function holdover. No donor/borrowed kernels exist (see [Team-only](#team-only)), so this gate covers only the op's own kernels, all compliant.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | — | — | *(no violations)* | — |

- **Feature compatibility:** every Appendix A entry scanned; all absent → `N/A`. No `GlobalCircularBuffer`, `address_offset`, `GlobalSemaphore`, or CTA-varargs signal fires anywhere in the op (host, factories, kernels).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `remote_cb`/`remote_index`, `.global_circular_buffer` field, or 4-arg `CreateCircularBuffer`. Plain `CBDescriptor`s only. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset`/`set_address_offset`/4-arg `UpdateDynamicCircularBufferAddress`/`cb_descriptor_from_sharded_tensor`. No borrowed-memory CBs at all. |
  | GlobalSemaphore | N/A | Op uses no semaphores of any kind. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a single `Tensor` (`fill_pad_device_operation_types.hpp:16-18`); every kernel reads CTAs at fixed constexpr literal indices — no runtime-varying CTA index, no `std::vector<Tensor>`. |

- **CB endpoints (GATE-free):** **✓ legal** — every CB is a genuine 1-producer/1-consumer FIFO on every node, in both factories and every config. Two touchers per CB, exactly one locked producer + one locked consumer:
  - `c_0` data-in: reader **produces** (`reserve_back`/`push_back`), compute **consumes** (`wait_front`/`pop_front`).
  - `c_1` right-mask (only when `has_right_pad`): **writer produces** the mask tile (`push_right_mask_tile`), compute consumes it persistently. *(Unusual but legal: the writer, not the reader, is this CB's producer.)*
  - `c_2` bottom-mask (only when `has_bottom_pad`): writer produces, compute consumes.
  - `c_16` data-out: compute **produces**, writer **consumes**.

  No hidden second writer (the only raw `get_write_ptr()` sites are the mask producer filling its own reserved slot), no multi-reader, no dual-instance work-split (reader and writer are *distinct* kernel sources, not one source instantiated twice), and no dead CB (`c_1`/`c_2` are simply not allocated when their pad is absent, rather than allocated-and-untouched). Nothing here needs a self-loop, 1P+1C assignment, multi-binding flag, or drop.
- **Offset base pointers:** **GREEN** — no address RTA folds a host-side offset into its base. Neither factory computes `buffer()->address() + <offset>` on the host; both push the bare `Buffer*` (`tens_buffer`) as RTA[0]. The DRAM kernels address by clean `page_id` (`fill_pad_reader.cpp:99,117,134`); the sharded kernels compute `shard_l1_base + <geometry>` **in the kernel** from a clean base RTA (`fill_pad_sharded_reader.cpp:70,90,111`), which is not a host fold. Consistent with the checked-in triage `2026-07-19_offset_base_pointers.md` (a dated prior), where fill_pad appears in **no** Type-1/2/3/4 table.
- **TensorAccessor 3rd argument:** **GREEN — Class 2 (redundant), drop.** Only the DRAM factory constructs `TensorAccessor`; both sites pass a 3rd arg: `fill_pad_reader.cpp:87` and `fill_pad_writer.cpp:81`, each `TensorAccessor(args, buf_addr, tile_bytes)` with `tile_bytes = get_tile_size(cb_id)`. Classifying from the two questions: (1) the accessor serves interleaved / DRAM-sharded tiled tensors; (2) `get_tile_size(cb)` is a **correct-magnitude** value (block-float-safe, equals the true tiled page). Correct magnitude ⇒ Class 2 (interleaved silently realigns; sharded uses it verbatim but it already equals the true page) ⇒ pure no-op drop. Matches `2026-07-06_tensor_accessor_3rd_arg_triage.md` (a dated prior), which lists `fill_pad` as **Class 2 — Redundant** and explicitly notes it was *previously mis-flagged as a bug* — it uses `tt::tile_size`, correct for block-float. The sharded factory constructs no `TensorAccessor`, so it has no 3rd-arg site.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (single in-place `input` binding; output tensor *is* the input, `fill_pad_device_operation.cpp:39-43`):
  - `input` — **Case 1** in `FillPadProgramFactory`: the base (`buf_addr`, RTA[0]) is fed into `TensorAccessor(args, buf_addr, tile_bytes)` in both reader (`fill_pad_reader.cpp:86-87`) and writer (`fill_pad_writer.cpp:80-81`); all access is through the accessor. Port → express as `TensorParameter`/`TensorBinding`; kernels build `TensorAccessor(tensor::name)`; the RTA[0] base and its `TensorAccessorArgs` plumbing (`fill_pad_program_factory.cpp:173,192`) both disappear.
  - `input` — **Case 2** in `FillPadL1ShardedProgramFactory`: the base (`shard_l1_base`, RTA[0]) is used **raw** in hand-rolled NoC self-reads/writes via `UnicastEndpoint` (`fill_pad_sharded_reader.cpp:70,73-80`; `fill_pad_sharded_writer.cpp:92-101`), never through a `TensorAccessor`. Port → bind the tensor as `TensorParameter`, pull the base via the `TensorAccessor::get_bank_base_address` bridge, keep the existing raw address arithmetic unchanged.
  - Both factories deliver the base today via the framework `Buffer*`-binding form (`emplace_runtime_args({tens_buffer, …})`) — patched on cache hits, correct today; the typed binding supersedes it. Compute kernel (`fill_pad_compute.cpp`) is **out of scope** — CB-only, touches no tensor memory.
- **TensorParameter relaxation:** none (sheet `none`; no custom hash).
- **TensorAccessor 3rd arg:** drop the redundant `tile_bytes` 3rd arg at `fill_pad_reader.cpp:87` and `fill_pad_writer.cpp:81` (Class 2 — no `dynamic_tensor_shape`, since it is *not* Class 1).
- **CB endpoints:** all legal (plain 1:1). Note config-dependence — `c_1`/`c_2` exist only when `has_right_pad`/`has_bottom_pad`; declare each DFB only in those configs, matching the current conditional CB allocation (`fill_pad_program_factory.cpp:109,122,484,495`).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no CB needs the multi-binding flag.
- **Cross-op / shared kernels:** none — the op owns all five kernels; no borrowed files, no `_metal2` fork to reuse, no sunset list.
- **RTA varargs:** none — every RTA is a fixed, nameable field.
- **Sharded lock-step counts (prior-revert root cause) — read before touching the sharded factory.** #50904 was reverted (#51605) for a sanity-test failure that, per the porter's own investigation, was a **sharded-factory CB deadlock** on a WIDTH_SHARDED case: the reader/writer moved a number of tiles that did not match what the compute kernel consumed (`num_work` conflated with the per-phase tile count vs. compute's `num_bottom = local_valid_w`). In the current baseline the reader/writer derive their actual tile counts from *shard geometry* (`shard_H_tiles`, `local_valid_w`, `has_right_pad`, `has_bottom_pad_core`), and `num_work` is only a "has any work" guard (used by the writer's early-return at `fill_pad_sharded_writer.cpp:60`, dead in the reader). **The port must preserve the exact right→bottom→corner lock-step tile counts across reader ↔ compute ↔ writer, and must not repurpose `num_work` as a loop bound.** A mismatch re-introduces the deadlock the revert removed.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean — nothing to inventory.** Every kernel `#include` resolves to `api/*` (tt_metal LLK/HAL/firmware — class 1, no concern) or the in-directory `fill_pad_dataflow_common.hpp`. No `ttnn/cpp/ttnn/kernel_lib/`, `kernel/`, `kernel_helper_functions/`, in-family, or cross-family includes. All five kernel `.cpp` sources are file-path-instantiated from within the op's own `device/kernels/` (`fill_pad_program_factory.cpp:213,222,239,537,558,592`); no borrowed kernel files, no `_metal2` fork exists or is needed, no shared-kernel coordination/sunset list.
- **Relaxation candidates (from a custom hash):** none — the op has no custom hash to mine.
- **TTNN factory analysis (sheet-derived facts + `file:line`):**
  - Current concept `descriptor`, execution model SPMD, target `ProgramSpecFactoryConcept` (both factories).
  - Op-owned tensors: no (empty on both rows; consistent with `descriptor`). No `WorkloadDescriptor`, no MeshWorkload.
  - No pybind `create_descriptor`, no other risky pybind, no custom hash, no `get_dynamic_runtime_args`, no `override_runtime_arguments` — all `no`.
  - `Op Classification = PD Op (pointer-patching)`, `Pointer patching perf issue? = OK`, `Formerly custom hashed? = no`.

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **Dead `elem_size` compile-time arg.** Read into a `constexpr` and then never used, in `fill_pad_reader.cpp:64` (CTA[7]), `fill_pad_compute.cpp:94` (CTA[4], shared by both factories), and `fill_pad_sharded_reader.cpp:41` (CTA[2]). The kernels size everything off `tile_bytes = get_tile_size(cb)` instead. Harmless; a cleanup for the ops team.
- **Dead `num_work` runtime arg in the sharded reader.** `fill_pad_sharded_reader.cpp:49` reads RTA[3] but the reader body never references it (its loop bounds come from shard geometry). The sharded *writer* does use it (early-return guard). Harmless — and, per the [Heads-ups](#heads-ups), keeping it inert in the reader is exactly the safe behavior the port must not disturb.

## Recipe notes

- **Pre-ported-then-reverted op — the audit recipe assumes a first-time port.** fill_pad's kernels were already migrated to Device 2.0 (`Noc`/`DataflowBuffer`/`TensorAccessor`) *before* the reverted factory port, so the Device 2.0 gate is trivially GREEN and the kernel-side reads look more modern than the recipe's "Device-2.0 baseline" framing implies. The recipe handled this fine (I audit the current committed state, which equals `main`), but a one-line acknowledgement that an op may arrive at audit *after a reverted port attempt* — and that the auditor should confirm current == `main` rather than assume a pristine legacy starting point — would orient the next auditor. The git-history / revert context proved to be the single most decision-relevant fact for framing this audit, yet the recipe doesn't prompt for it.

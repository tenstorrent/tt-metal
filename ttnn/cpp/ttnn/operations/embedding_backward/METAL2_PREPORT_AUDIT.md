# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/embedding_backward`

Single device operation, single program factory:

- **`ttnn::prim::EmbeddingBackwardDeviceOperation`**
  - `EmbeddingBackwardDeviceOperation (single-descriptor)` — `create_descriptor` in `device/embedding_backward_program_descriptor.cpp`

Kernels referenced by the factory:

- `device/kernels/dataflow/reader_embedding_backward.cpp` (reader, DM)
- `device/kernels/compute/embedding_backward.cpp` (compute)

No unreferenced kernel files in the directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/embedding_backward` |
| **Overall** | **RED** |
| **DOps / Factories** | `EmbeddingBackwardDeviceOperation` → `EmbeddingBackwardDeviceOperation (single-descriptor)` |
| *Prereqs* — Device 2.0 (every kernel used) | **No (RED)** — routed to the Device 2.0 track; **isolated CB-index holdovers only** (2 sites, compute kernel; idioms otherwise Device 2.0) |
| *Prereqs* — Cross-op escapes | Ok — no out-of-directory kernel `#include`s or borrowed kernel files (skipped in full per RED scoping; see below) |
| *Feature Support* — overall | GREEN (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (sheet; cross-check clean) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (cleared) |
| *Port work* — Tensor bindings (per binding) | *(deferred — informational subject skipped per RED scoping)* |
| *Port work* — TensorParameter relaxation | none (sheet) |
| *Port work* — TensorAccessor 3rd arg | none (no 3rd-arg site; cleared) |
| *Port work* — CB endpoints | *(deferred — informational subject skipped per RED scoping)* |

## Result

**RED → blocked on the Device 2.0 prerequisite**, routed to the **Device 2.0 migration team**.

`RED at op level; no portable subset` — the op has a single factory whose single compute kernel carries the blocking holdovers, so no clean factory/config subset can be carved out.

The block is **narrow and cheap**: two CB-index-keyed free-function calls in the compute kernel (`read_tile_value`, `get_tile_address`) that have `CircularBuffer` wrapper-method replacements. Everything else clears — the TTNN factory-concept gate is GREEN (`Is able to port? = yes`, cross-check clean), no unsupported Appendix A feature is used, no offset-base-pointer fold, and no `TensorAccessor` 3rd-arg site. Once the Device 2.0 track lands these ~1-line swaps, the op should re-audit cleanly and (given the other gates already pass) proceed to a port.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness sheet row (`embedding_backward` / `EmbeddingBackwardDeviceOperation` / `single-descriptor`): `Is able to port? = yes`. Derivation all-clear — `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args? = no`, `Pybind descriptor = no`, `Is safe to port? = yes`, `Smuggled pointer = no`. Cross-check against code confirms every cheaply-checkable column:
  - `Concept = descriptor` — `create_descriptor()` returns `ProgramDescriptor` (`device/embedding_backward_program_descriptor.cpp:18`).
  - `Custom hash = no` — no `compute_program_hash` override anywhere in the op directory.
  - `Runtime-args update = no` — no `get_dynamic_runtime_args` / `override_runtime_arguments` in the factory.
  - `Pybind descriptor = no` — `embedding_backward_nanobind.cpp:42` binds the host function `embedding_bw`, not `create_descriptor`.
  - Cross-column invariants hold (no runtime-args-update / op-owned-tensors on a `descriptor` row).

- **Device 2.0 (every kernel used):** **RED — isolated CB-index holdovers.** The reader (`reader_embedding_backward.cpp`) is fully Device 2.0 (`Noc`, `CircularBuffer`, `CoreLocalMem`, `TensorAccessor`; `get_tile_size(cb_id)` is the sanctioned free function). The compute kernel (`compute/embedding_backward.cpp`) is structurally Device 2.0 (uses `CircularBuffer` wrappers for all FIFO ops) but retains two **non-sanctioned CB-index-keyed free-function** calls that have `CircularBuffer` wrapper-method replacements. Neither `read_tile_value` nor `get_tile_address` is on the sanctioned list (only `get_tile_size` and `get_local_cb_interface` are), and the `CircularBuffer` wrapper implements both as standalone methods (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:72` and `:91` — full bodies, not forwarders to the free function), so they are genuine holdovers, not sanctioned survivors.

  | File | Line | Call | Wrapper-method replacement | Wrapper in scope? |
  |---|---|---|---|---|
  | `device/kernels/compute/embedding_backward.cpp` | 40 | `get_tile_address(cb_mask_idx, 0)` | `cb_mask.get_tile_address(0)` | Yes — `CircularBuffer cb_mask(cb_mask_idx)` at line 25 (1-line swap) |
  | `device/kernels/compute/embedding_backward.cpp` | 34 | `read_tile_value(cb_chunk_count_scratch_idx, 0, 0)` | `cb_chunk_count_scratch.read_tile_value(0, 0)` | No — `cb_chunk_count_scratch` is never instantiated as a `CircularBuffer` in this kernel; needs the wrapper declared first, then the method call (2-line swap) |

  **Sizing for the Device 2.0 team:** isolated holdovers, *not* a broad Device 1.0 migration. Both sites are the only remaining pre-Device-2.0 idioms in the op; the surrounding kernel already uses `CircularBuffer` wrappers and the reader is fully migrated. Total fix is ~2–3 lines in one file. Per the kernel-side whitelist this is out of port scope (a Device 2.0 change, even a 1-liner), so it lands on the Device 2.0 track and the op returns for a cheap re-audit.

  Not flagged (correctly Device-2.0 or not in scope for this gate): compute-thread math/pack APIs that take a `cb_id` — `unary_op_init_common(cb_grad_idx, cb_out_idx)` (line 28), `copy_tile(cb_grad_idx, …)` / `copy_tile(cb_out_intermed_idx, …)` (lines 55, 57), `pack_tile(1, cb_out_idx, …)` (line 66) — have no `CircularBuffer` wrapper-method equivalent and are the normal compute API surface, not DM holdovers. Reader `get_tile_size(cb_id)` calls (lines 111–112, 130) are sanctioned.

- **Feature compatibility:** every Appendix A entry scanned; none fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer`, `.global_circular_buffer` field, `remote_index`/`remote_cb` idiom, or 4-arg `CreateCircularBuffer`. CBs are plain `CBDescriptor`s. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set on any `CBDescriptor` (all six CBDescriptors omit it → default 0); no `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | No semaphores of any kind in the op (host or kernel) — no `Semaphore`, `GlobalSemaphore`, or `CreateSemaphore`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed 2-tensor (+ optional preallocated output) struct — no `std::vector<Tensor>`. Both kernels read compile-time args only at constexpr offsets (`get_compile_time_arg_val(0..6)`; `TensorAccessorArgs<7>` / `<…next_offset>`). No runtime-varying CTA index. |

- **Offset base pointers:** **GREEN — no fold.** Scanned every address-carrying RTA. The factory passes tensor bases via the `Buffer*`-binding form — `reader_desc.emplace_runtime_args(core, {grad_tensor_buffer, index_tensor_buffer, out_buffer, embedding_tiles, offset, tiles_this_core})` (`embedding_backward_program_descriptor.cpp:193-194`) — never `->address()` and never `base + offset`. There is no `.address()`/`->address()` call anywhere in the factory. The `offset` RTA is a plain tile-index accumulator (`offset += tiles_this_core`, line 197), consumed on-device as a page index into a `TensorAccessor` (`grad_tile_idx = hidden_offset; … {.page_id = grad_tile_idx + hidden_dim}` — reader lines 153-180), not folded into a device base pointer. No Type 1/2/3/4 site. (Op does not appear in the offset-base-pointer triage prior, consistent with this clean scan.)

- **TensorAccessor 3rd argument:** **GREEN — no site.** All three `TensorAccessor` constructions are 2-arg (`args, base_addr`) with no explicit page-size argument: `TensorAccessor(grad_args, grad_tensor_addr)`, `TensorAccessor(index_args, index_tensor_addr)`, `TensorAccessor(out_args, output_tensor_addr)` (reader lines 118-120). The page-size subject cannot fire.

## Informational subjects — skipped (whole-op RED, no portable subset)

Per the Red-outcome scoping rule (`audit/metal2_audit.md`, Feasibility audit § top): on a whole-op RED with no portable subset, the seven purely-informational subjects are deferred to the re-audit that runs once the Device 2.0 holdovers clear (the compute-kernel idioms will change under that migration, so a census now would be re-done against changed code). Each recorded so the omission is not mistaken for a clean result:

- **TTNN porting shape** — skipped — whole-op RED, no portable subset; re-audit on unblock. *(Target concept is nonetheless straightforward: `descriptor` + no op-owned tensors → `MetalV2FactoryConcept`.)*
- **TensorParameter relaxations** — skipped — whole-op RED, no portable subset; re-audit on unblock. *(Sheet lists `none`.)*
- **TensorParameter analysis** — skipped — whole-op RED, no portable subset; re-audit on unblock.
- **CB endpoints** — skipped — whole-op RED, no portable subset; re-audit on unblock. *(Additionally, this subject's precondition prefers a post–Device-2.0 census, since the compute-kernel idiom changes could alter the toucher set.)*
- **RTA varargs** — skipped — whole-op RED, no portable subset; re-audit on unblock.
- **Out-of-directory coupling** — skipped — whole-op RED, no portable subset; re-audit on unblock. *(Cheap observation retained: both kernels `#include` only `api/…` LLK/HAL headers — `tt_metal/*` class, no concern — and the factory instantiates only the op's own two kernel files by path. No donor coupling that would add a Device 2.0 gate beyond the one above.)*
- **Incidental anomalies** — skipped — whole-op RED, no portable subset; re-audit on unblock.

## Recipe notes

- **Isolated-holdover definition vs. wrapper-not-yet-in-scope.** The Device 2.0 Green/Red bullets define an *isolated holdover* as a CB-index free function "where the corresponding Device-2.0 wrapper object is **already in scope** at the call site *and* a wrapper-method replacement exists." The `read_tile_value(cb_chunk_count_scratch_idx, …)` site (compute line 34) has a wrapper-method replacement but **no wrapper in scope** — the CB is never instantiated as a `CircularBuffer` in that kernel. It is plainly the same class of holdover (non-sanctioned CB-index free function with a wrapper replacement) and the same track (Device 2.0), just a 2-line fix (declare wrapper + call method) rather than 1-line. Reading the "wrapper in scope" clause as *strictly required* would leave this site unclassified; I treated it as an isolated holdover with a note. A one-line clarification that "wrapper-in-scope" describes the cheapest sub-case, not a gating condition, would remove the ambiguity.
- **`get_tile_address` / `read_tile_value` are not named in the Device 2.0 migration guide.** The guide's examples cover `get_write_ptr`/`get_read_ptr` → wrapper migration and explicitly sanction `get_tile_size`, but say nothing about these two compute-thread accessors. I classified them as holdovers because (a) they are not on the sanctioned list, (b) `CircularBuffer` provides standalone (non-forwarding) `#ifdef COMPILE_FOR_TRISC` method equivalents, and (c) they are widely used across not-yet-migrated compute kernels. If the intent is that compute-thread tile accessors are exempt at the Device 2.0 stage (analogous to `copy_tile`/`pack_tile`), that exemption should be stated in the sanctioned list, since the shape otherwise reads as a textbook holdover.

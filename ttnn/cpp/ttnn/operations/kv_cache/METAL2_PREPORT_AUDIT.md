# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/kv_cache/`

One device operation shares this directory, with two program factories:

- **`UpdateKVCacheOperation`** (`device/update_cache_device_operation.{hpp,cpp}`)
  - `UpdateCacheMultiCoreProgramFactory` (`device/update_cache_multi_core_program_factory.cpp`) — decode/update path (`UpdateCacheOpType::UPDATE`)
  - `FillCacheMultiCoreProgramFactory` (`device/fill_cache_multi_core_program_factory.cpp`) — prefill/fill path (`UpdateCacheOpType::FILL`)

`select_program_factory` picks between them on `op_type`. Both factories are on the `ProgramDescriptor` API (`create_descriptor` returning a `ProgramDescriptor`), and each carries an `override_runtime_arguments` cache-hit hook.

Kernels referenced (audited in scope):

- `device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp` (update reader)
- `device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp` (update writer)
- `device/kernels/compute/update_cache.cpp` (update compute; instantiated once per core group)
- `device/kernels/dataflow/reader_fill_cache_interleaved_start_id.cpp` (fill reader)
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (fill writer — **cross-family donor**, file-path instantiated)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `63ca139b420 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/kv_cache/` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `UpdateKVCacheOperation` → `UpdateCacheMultiCoreProgramFactory`, `FillCacheMultiCoreProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — own kernels + donor `writer_unary_interleaved_start_id.cpp` all Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | Ok — one cross-family donor kernel (Device 2.0, `_metal2` fork already exists); compute includes shared `kernel_lib/` |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — GlobalCircularBuffer | N/A — header included but **never used** (dead include; see Misc) |
| *Feature Support* — CBDescriptor `address_offset` (non-zero) | N/A — no CB sets a non-zero offset |
| *Feature Support* — GlobalSemaphore | N/A — no semaphores of any kind |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factory rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD | N/A — concept is `descriptor`, not `WorkloadDescriptor` |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `update_cache_device_operation.cpp:160` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects CustomProgramSpecFactoryConcept): `update_cache_multi_core_program_factory.cpp:436`, `fill_cache_multi_core_program_factory.cpp:262` |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** (both factories) |
| *Port work* — Offset base pointer | none — every address is a clean `->address()` base; offsets ride separate scalar page-index / L1-offset args |
| *Port work* — Tensor bindings (per binding) | `cache` → Case 1 · `input` → Case 1 (interleaved) / clean borrowed-DFB (sharded) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor in the op passes a 3rd argument |
| *Port work* — CB endpoints | legal 1:1 for all CBs **except** the interm0/interm1 aliased CB — see heads-up |

**CB endpoints** are dispositions, not gates. Every CB here is a legal 1-producer/1-consumer FIFO except the single `CBDescriptor` that carries **two** format descriptors (`c_24`/`c_25`, aliasing one L1 region), which is a genuine multi-endpoint CB on one instance and needs a careful census at port time (see Heads-ups). Nothing here blocks the port.

## Result

**GREEN → brief issued.** Both factories clear every gate: Device 2.0 (own + donor kernels), Feature compatibility (no GlobalCircularBuffer / `address_offset` / GlobalSemaphore in use), TTNN factory concept (`Is able to port? == yes`, cross-check clean), Offset base pointers (all clean bases), and TensorAccessor 3rd argument (no site). Port work is ordinary tensor-binding translation plus a careful CB census on one aliased buffer. Target concept: `CustomProgramSpecFactoryConcept` (both factories carry `override_runtime_arguments`).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Both factory rows read `Is able to port? == yes`. Cross-check against the code is clean:
  - `Concept == descriptor` ✓ — both factories define `create_descriptor()` returning a `ProgramDescriptor`.
  - `Custom hash == yes` ✓ — `UpdateKVCacheOperation::compute_program_hash` at `update_cache_device_operation.cpp:160` (hashes `op_type` + the two tensors; deliberately excludes `batch_idx`/`update_idx`/`batch_offset` and `compute_kernel_config`). Not a gate; the port leaves it as-is.
  - `Runtime-args update (get_dynamic_runtime_args) == no` ✓ — no such hook; the only textual matches are comments noting the framework uses *neither* `resolve_bindings` *nor* `get_dynamic_runtime_args` for this op.
  - `Override runtime args method? == yes` ✓ — `override_runtime_arguments` on both factories (`update_cache_multi_core_program_factory.cpp:436`, `fill_cache_multi_core_program_factory.cpp:262`). Not a gate; it selects the target concept.
  - `Pybind descriptor == no` ✓ — `kv_cache_nanobind.cpp` binds only the four high-level user functions; no `create_descriptor` binding.
  - `TensorParameter relaxation == none` ✓ — clears the relaxation conjunct.
  - `Op-owned tensors? == (blank)` — no op-owned tensors, consistent with the `descriptor` concept.
  - Factory-set match: 2 sheet rows ↔ 2 code factories, one-to-one. No phantom / missing rows.
  - Cross-column invariants hold (no `get_dynamic_runtime_args` on this concept; no op-owned tensors on a `descriptor` row).

  (The sheet's `Execution Model == SPMD` is the execution model, not a `WorkloadDescriptor` concept — the `Concept` cell is `descriptor`, so there is no genuine-multi-program gate.)

- **Device 2.0 (every kernel used):** **GREEN.** Every kernel the op instantiates is on Device 2.0 data-movement idioms:
  - `reader_update_cache_interleaved_start_id.cpp` — `Noc`, `CircularBuffer` objects, `noc.async_read(...)`, `TensorAccessor`; `get_tile_size(cb_id)` is a **sanctioned** free function.
  - `writer_update_cache_interleaved_start_id.cpp` — `Noc`, `CircularBuffer` objects (`get_read_ptr()` methods), `TensorAccessor`, `UnicastEndpoint{}`, `CoreLocalMem<uint32_t>`, `noc.get_noc_id()`; `get_tile_size(cb_id)` sanctioned.
  - `compute/update_cache.cpp` — pure compute API (`compute_kernel_hw_startup`, `compute_kernel_lib::untilize/tilize`, `reconfig_data_format_srca`); no legacy data-movement idioms.
  - `reader_fill_cache_interleaved_start_id.cpp` — `Noc`, `CircularBuffer`, `TensorAccessor`; `get_tile_size(cb_id)` sanctioned.
  - Donor `writer_unary_interleaved_start_id.cpp` (eltwise/unary) — `Noc`, `DataflowBuffer` object, `TensorAccessor`; `get_local_cb_interface(cb_id).fifo_page_size` is a **sanctioned** free function. Already Device 2.0.

  No CB-index-keyed data-movement holdovers, no `InterleavedAddrGen`/`ShardedAddrGen`, no raw semaphore addresses.

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | `<tt-metalium/global_circular_buffer.hpp>` is **#included** at `update_cache_device_operation.hpp:13` but the type, `CreateGlobalCircularBuffer`, `.global_circular_buffer` field, `remote_index`/`remote_cb`, and the 4-arg `experimental::CreateCircularBuffer(..., global_cb)` are all **absent**. No signal fires — the include is dead code (see Misc). |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `.address_offset`; no `set_address_offset`. The `UpdateDynamicCircularBufferAddress(program, cb_id, *buffer)` calls (`update_cache_...cpp:491`, `fill_cache_...cpp:300`) are the **3-arg `Buffer&`** form (the false-positive guard) — not the 4-arg offset form. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore`, no `CreateGlobalSemaphore`, no semaphores of any kind in the op or its kernels. |

- **CB endpoints (GATE-free):** legal for every CB except one aliased buffer that needs a careful port-time census. Counted per node, holds across configs unless noted.

  **`UpdateCacheMultiCoreProgramFactory`:**
  | CB | Producer | Consumer | Verdict |
  |---|---|---|---|
  | `c_0` (cache / src0) | reader (`reserve_back`/`push_back`) | compute (`untilize` cache_cb) | legal 1:1 |
  | `c_1` (input / src1) | reader | compute (`untilize` in_cb) | legal 1:1 (interleaved); sharded → borrowed-memory (`.buffer = src_buffer`), reader `reserve/push` + compute consume → still 1:1 |
  | **`c_24`/`c_25`** (interm0/interm1 — **one CBDescriptor, two format descriptors, aliased L1**) | compute produces `c_24` (`untilize` output); writer produces `c_25` (`reserve_back`/`push_back`) | writer consumes `c_24` (`wait_front`/`pop_front`, **plus a raw in-place L1 write** via `cb_untilized_cache.get_read_ptr() + offset`); compute consumes `c_25` (`tilize` input) | **multi-endpoint on one instance** — careful census; see heads-up |
  | `c_26` (interm2 / untilized_input) | compute (`untilize` untilized_in output) | writer (`wait_front`/`get_read_ptr`/`pop_front`) | legal 1:1 |
  | `c_16` (output) | compute (`tilize` out) | writer (`wait_front`/`pop_front`) | legal 1:1 |

  **`FillCacheMultiCoreProgramFactory`:**
  | CB | Producer | Consumer | Verdict |
  |---|---|---|---|
  | `c_0` (src0, reused as output — pass-through) | reader (`reserve_back`/`push_back`) | donor writer (`dfb.wait_front`/`pop_front`) | legal 1:1 (interleaved); sharded → borrowed-memory (`.buffer = src_buffer`), reader `reserve/push` + writer consume → 1:1 |

- **Offset base pointers:** **GREEN.** Every device address delivered to a kernel is a clean `buffer->address()` base:
  - Cache-hit addresses in both `override_runtime_arguments` are `src_buffer->address()` / `dst_buffer->address()` — no `+ offset` fold (`update_cache_...cpp:464-465`, `fill_cache_...cpp:282-283`).
  - `cache_start_id`, `input_start_id`, `batch_start_id`, `start_id` are **tile/page indices** passed as separate scalar RTAs and consumed as `page_id` by the `TensorAccessor`, not folded into a base address.
  - `tile_update_offset`, `batch_read_offset`, `Wbytes` are **L1-local byte offsets** applied to CB read pointers (`cb_untilized_cache.get_read_ptr() + offset`, `cb_untilized_input.get_read_ptr() + batch_read_offset`), never to a tensor base.

  Not in the offset-base-pointer triage tables (`2026-07-19_offset_base_pointers.md`) — consistent with "no fold, op not in tables → clean."

- **TensorAccessor 3rd argument:** **N/A** — no accessor in the op passes a 3rd (page-size) argument. Every `TensorAccessor(...)` construction is 2-arg: `reader_update:38,43`, `writer_update:44`, `reader_fill:29`, donor `writer_unary:39`. (The triage table's `zero_padded_kv_cache` is a *different* op — `experimental/deepseek_prefill/zero_padded_kv_cache` — not this one.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `cache` (in-place output) — **Case 1** (via `TensorAccessor`), both factories, all configs. Reader and writer feed `dst_buffer->address()` into `TensorAccessor(cache_args, cache_addr)` and address by `page_id`.
  - `input` — **Case 1** (via `TensorAccessor`) in the interleaved config (`TensorAccessor(input_args, input_addr)` / `TensorAccessor(src_args, src_addr)`); **clean** (borrowed-memory DFB) in the sharded config, where the CB is `.buffer = src_buffer` and the reader only `reserve_back`/`push_back`s under `INPUT_SHARDED`. Record the per-config split.
  - Delivery today is the `Buffer*`-binding form (`emplace_runtime_args({dst_buffer, src_buffer, ...})`) with the addresses additionally re-applied by `override_runtime_arguments`. The port replaces both with typed `TensorParameter` bindings; the custom override then re-applies only the non-address, hash-excluded scalars (see brief).
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal 1:1 except the interm0/interm1 aliased CB (`c_24`/`c_25`) — see Heads-ups.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints — the aliased interm CB (`c_24`/`c_25`).** `create_descriptor` pushes **one** `CBDescriptor` (`update_cache_multi_core_program_factory.cpp:235-261`, `total_size = num_interm_tiles * interm_single_tile_size`) carrying **two** `CBFormatDescriptor`s: `interm0_cb_index = c_24` and `interm1_cb_index = c_25`. The two indices alias the same L1 region — that is what makes the writer's data flow correct: the writer waits on `c_24`, does a raw in-place L1 write into it (`cb_untilized_cache.get_read_ptr() + offset`, `writer_update:60-68`), signals `c_25` (`reserve_back`/`push_back`), and compute then `tilize`s from `c_25`. On this single CB instance the census is compute (produce `c_24` / consume `c_25`) + writer (consume `c_24` / produce `c_25`), i.e. two producers and two consumers across the two views, plus a raw in-place write. The porter must **preserve the two-format-descriptor structure** and decide the DFB binding carefully — this is very likely a multi-binding-advanced-option CB, but confirm how the two format views share (or don't share) FIFO state before choosing the binding, since that decides producer/consumer assignment. (GATE-free — this does not block the port.) See Recipe notes.
- **CB endpoints — writer raw L1 poke.** The update writer additionally performs an L1→L1 `noc.async_read(UnicastEndpoint{}, CoreLocalMem(...), ...)` copy into `c_24`'s memory (`writer_update:60-68`). This is a poke on a CB the writer already binds (its consumer side of `c_24`) — not a tensor-memory access, so it is **not** a TensorParameter Case 2; it is only relevant to the `c_24`/`c_25` census above.
- **Cross-op / shared kernels:**
  - `FillCacheMultiCoreProgramFactory` file-path-instantiates the cross-family donor `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`. A **`_metal2` fork already exists beside it** — `writer_unary_interleaved_start_id_metal2.cpp` (same directory). The port should **bind the existing fork**, not create a new one and not convert the legacy file in place. The donor's own header + issue **#52228** track the full consumer/sunset list — treat that as a sunset list, not authorization to convert in place.
  - `compute/update_cache.cpp` `#include`s the shared kernel library `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `tilize_helpers.hpp` (function-call escape, shared-lib class — lib team owns; does not gate and needs no fork). It is a compute kernel, so out of scope for TensorParameter analysis.
- **RTA varargs:** none. Every kernel reads its runtime args as a fixed set of distinct fields at constant indices; no variable-count loop indexes `get_arg_val`, and no read is data-selected. Ordinary named-arg port work.
- **Target concept:** both factories → `CustomProgramSpecFactoryConcept` (each has an `override_runtime_arguments`). The override translates to a method returning `ProgramRunArgs`; note that on port the tensor **addresses** move to the typed binding channel (which refreshes on cache hit), so the translated override re-applies only the **non-address, hash-excluded** scalars: `cache_start_id` (per core), `tile_update_offset`, `batch_read_offset`, `Wbytes` (update), and `cache_start_id` (fill). Those derive from `update_idx`/`batch_offset`/`fp32_dest_acc_en`, all excluded from `compute_program_hash`, so they are not stable across cache hits and must be re-applied. `compute_update_cache_dynamic_args` / `compute_fill_cache_start_ids` are the shared source of truth for the work-split and formulas — keep them shared between the create path and the translated override.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up:** ✓ clean (no sequence-blocking donor). One cross-family file-path borrow (Device 2.0, fork exists) and one shared-lib function-call escape.
  - **Summary table:**

    | Op kernel | Donor file | Class | Status |
    |---|---|---|---|
    | (fill factory instantiation) | `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | 6 — cross-family donor (file-path) | ✓ Device 2.0; `_metal2` fork exists — bind it |
    | `compute/update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | 2 — official shared kernel library | ✓ lib team owns |
    | `compute/update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | 2 — official shared kernel library | ✓ lib team owns |

    All other includes in the op's kernels resolve under `tt_metal/*` (LLK/HAL/firmware — class 1, no concern).
  - **Borrowed kernel files (file-path instantiation):** `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — owning family `eltwise/unary`; broadly shared (many ops borrow this writer, see #52228); `_metal2` fork **present** (`writer_unary_interleaved_start_id_metal2.cpp`, same dir; a `_wh.cpp` variant also sits beside it but is unrelated to the port).
- **TTNN factory analysis (sheet-derived, code-confirmed):** current concept `descriptor` (both); custom hash present (`update_cache_device_operation.cpp:160`, non-gating, leave intact); `override_runtime_arguments` present on both factories (non-gating, → CustomProgramSpecFactoryConcept); no pybound `create_descriptor`; no op-owned tensors; no `get_dynamic_runtime_args`; `TensorParameter relaxation == none`. Sheet `Model == other`; `Uses llama kernels? == yes` (fill), `no` (update) — informational.
- **Relaxation candidates:** none noticed. The custom hash intentionally excludes cache-hit-varying attributes (`batch_idx`/`update_idx`/`batch_offset`) and `compute_kernel_config`; that is the designed cache-key, not a relaxation candidate.

## Misc anomalies  *(team-only, non-gating)*

- **Dead include — `<tt-metalium/global_circular_buffer.hpp>`** at `update_cache_device_operation.hpp:13`. GlobalCircularBuffer is never referenced anywhere in the op; the header appears to be a leftover. Harmless, but it trips a first-pass Appendix A scan. Candidate for removal by the ops team (not port work).
- **`<tt-metalium/experimental/program_descriptor_patching.hpp>`** at `update_cache_device_operation.hpp:14` — included but no `patch`/patching symbol is used directly in the op's `.cpp` files. May be pulled in transitively by the override pattern; worth a quick confirm-and-remove if unused. Non-gating.

## Recipe notes

- **Multi-format-descriptor (aliased) CBs aren't covered by the CB endpoints model.** The [CB endpoints](../ai/audit/metal2_audit.md) subject defines "a CB" as "one `CBDescriptor` over a core range → one instance per node" and counts endpoints per instance. It does not address a single `CBDescriptor` carrying **two `CBFormatDescriptor`s that alias one L1 region** (here `c_24`/`c_25`), where two kernels touch *both* views in opposite roles (compute produces `c_24`/consumes `c_25`; writer consumes `c_24`/produces `c_25`) and one kernel raw-writes the shared region in place. The census rules (locked-producer / locked-consumer / role-free) are written for one buffer_index per instance; whether the two aliased views share FIFO state (making this one 2-producer/2-consumer FIFO) or track independently (making it two coupled FIFOs over shared memory) changes the correct binding, and the recipe gives no rule for it. It resolves to PORT WORK either way (GATE-free), but the porter is handed a genuine judgment the auditor could not fully close from the recipe. Worth a dedicated paragraph in the CB endpoints subject (or the port recipe) on how DFB expresses a two-view aliased CB.

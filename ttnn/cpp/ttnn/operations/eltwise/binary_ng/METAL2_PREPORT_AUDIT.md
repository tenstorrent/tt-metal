# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/eltwise/binary_ng`

- **`BinaryNgDeviceOperation`**
  - `ProgramFactory` (`device/binary_ng_program_factory.cpp`) — single factory, `create_descriptor` returns `tt::tt_metal::ProgramDescriptor`. Runtime-selects its reader/writer/compute kernel *source files* from a large menu (see Runtime kernel-source selection below). Helpers in `device/binary_ng_utils.cpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

> **⚠ Provenance note — read first (2026-06-11).** This audit's *original* finding was **RED**, blocked on the Device 2.0 prerequisite: the 7 row-major (rm) dataflow kernels used the legacy raw `noc_async_read`/`noc_async_write` transfer idiom. To unblock a Metal 2.0 **port experiment**, that Device 2.0 migration was performed **locally on this branch** (`akertesz/port-experiment-eltwise-binary-ng`) — it is *not* an upstream landing. The status below reflects the **post-migration** state. The migration + its validation are described in the Device 2.0 gate detail. A real port off this branch inherits that local D2.0 work as a dependency; if it is not yet upstreamed, it must be carried or landed first. The original RED finding and the reasoning behind it are preserved in the gate detail and git history for reproducibility.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/eltwise/binary_ng` |
| **Overall** | **GREEN** *(after local Device 2.0 migration — see provenance note)* · originally RED |
| **DOps / Factories** | `BinaryNgDeviceOperation` → `ProgramFactory` (single) |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** *(after local migration)* — 7 rm kernels moved to `noc.async_read/write` + `CoreLocalMem`; validated on Wormhole (3224 row-major + 94 tile/sharded bcast tests pass). Originally **No**. |
| *Prereqs* — Cross-op escapes | Ok (no cross-family / kernel-lib coupling; only tt_metal HAL + in-op helpers) |
| *Feature Support* — overall | GREEN (no UNSUPPORTED feature in use) |
| *Feature Support* — Variadic-CTA | Ok (fixed-count tensor inputs; no runtime-varying CTA index) |
| *TTNN Readiness* — Port Type | `ProgramSpecFactoryConcept` / caching strategy TBD at port time (see Factory concept) |
| *TTNN Readiness* — TTNN infra++ | delete custom `compute_program_hash`; dynamic-TensorAccessor (`RuntimeTensorShape`) relaxation decision pending |

Port Type → Option map: **1** = `MinimizeCacheHitCost`; **2** = `MaximizeCacheReuse` basic; **3** = `MaximizeCacheReuse` advanced.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`). All gates clear: ProgramDescriptor ✓, Device 2.0 ✓ *(after the local migration described below)*, no UNSUPPORTED feature, factory concept (`ProgramSpecFactoryConcept`) implemented.

**Original finding was RED**, blocked on the Device 2.0 prerequisite: the op's single `ProgramFactory` runtime-selects its kernel source files by input layout, and the **row-major (rm)** dataflow kernels performed their NoC transfers with the legacy free functions `noc_async_read`/`noc_async_write`, which the [Device 2.0 data-movement migration guide](../../../../../../docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md) classifies as legacy. Because the atomic porting unit is *one ProgramFactory + every kernel source it can runtime-select*, the whole factory was blocked until those rm kernels migrated.

**That migration was performed on this branch** (see Device 2.0 gate detail) and validated on a Wormhole, clearing the only failing gate. With the rm kernels on `noc.async_read/write` + `CoreLocalMem`, the entire factory (tile + rm paths) is now Device 2.0 clean and portable in a single pass.

## Gate detail

- **ProgramDescriptor:** **GREEN.** `device/binary_ng_program_factory.cpp:384` `create_descriptor(...)` populates a `ProgramDescriptor` (`desc.cbs.push_back(CBDescriptor{...})`, `KernelDescriptor`, `desc.kernels.push_back(...)`). No imperative `host_api.hpp` builder calls (`CreateProgram`/`CreateKernel`/`CreateCircularBuffer`/`SetRuntimeArgs`) anywhere in the host code.

- **Device 2.0 (every kernel used):** **GREEN** *(after local migration; originally RED)*.

  **Original RED:** all 7 row-major dataflow kernels used raw `noc_async_read`/`noc_async_write` free-function transfers (Device 1.0 idiom per the migration guide). They already constructed `TensorAccessor` objects for addressing (`src.get_noc_addr(row)`) and used the `noc.async_*_barrier()` method form — a mixed picture — but the transfer call itself was legacy. Not the sanctioned isolated-CB-index-holdover carve-out; it was the primary data-movement mechanism using the legacy NoC API.

  **Migration performed on this branch (2026-06-11).** Each `noc_async_read(addr, l1, len)` → `noc.async_read(src_accessor, CoreLocalMem<uint32_t>(l1), len, {.page_id, .offset_bytes}, {})`, and the symmetric `noc.async_write(...)` for the writer. Equivalence is exact: the `TensorAccessor` src-args path computes `get_noc_addr(page_id, offset_bytes)`, and `get_noc_addr(page, offset)` is `… + offset` (raw byte add — verified `tensor_accessor.h:315`), identical to the legacy `get_noc_addr(row) + chunk_offset`. `CoreLocalMem<uint32_t>(l1)` resolves to `l1 + 0` (the guide's blessed raw-L1 wrapper). For the scratch single-element reads the explicit `get_noc_addr()` call (a D2.0 accessor method) was retained where the low-bits alignment arithmetic needs it; only the transfer was swapped. Added includes: `api/tensor/noc_traits.h` + `api/core_local_mem.h` to the rm kernels. Files now clean (`noc.async_read/write` + `CoreLocalMem`):

  | File (`device/kernels_ng/dataflow/`) | Sites migrated |
  |---|---|
  | `reader_interleaved_rm_no_bcast.cpp` | 2 |
  | `reader_interleaved_rm_col_bcast.cpp` | 4 |
  | `reader_interleaved_rm_row_bcast.cpp` | 4 |
  | `reader_interleaved_rm_row_col_mixed_bcast.cpp` | 4 |
  | `reader_interleaved_rm_scalar_bcast.cpp` | 4 |
  | `reader_interleaved_rm_scalar_op.cpp` | 1 |
  | `writer_interleaved_rm_no_bcast.cpp` | 1 (`noc.async_write`) |

  **CB-index holdovers (also fixed on this branch):** the two tile readers' `cb_reserve_back`/`cb_push_back` (inside `#if SRC_SHARDED[_B]`, wrapper in scope) → `cb_src.reserve_back()`/`.push_back()`; and `fill_tile_utils.hpp`'s `get_write_ptr(cb_id)` → `CircularBuffer(cb_id).get_write_ptr()` (added `circular_buffer.h`). All are thin shims over the same free functions (verified in `circular_buffer.h`), so behavior is identical.

  | File | Holdover fixed |
  |---|---|
  | `device/kernels_ng/dataflow/reader_interleaved_col_bcast.cpp` | `cb_reserve_back`/`cb_push_back` → wrapper methods |
  | `device/kernels_ng/dataflow/reader_interleaved_scalar_bcast.cpp` | `cb_reserve_back`/`cb_push_back` → wrapper methods |
  | `device/kernels/dataflow/fill_tile_utils.hpp` | `get_write_ptr(cb_id)` → `CircularBuffer(cb_id).get_write_ptr()` (×20) |

  **Not touched (correctly clean):** compute kernels use the free-function CB API (`cb_reserve_back(cb_out, …)` etc.) — the standard, only mechanism on TRISC cores, *not* a holdover (no `CircularBuffer` wrapper in scope).

  **Validation (Wormhole n300):** `test_binary_bcast.py -k "row_major and not sharded"` → **3224 passed, 0 failed** (281 kernels JIT-built, 0 build failures). `-k "subtile or sharded_bcast_{scalar,w,h}_height"` → **94 passed, 0 failed**. No PCC/allclose mismatches, no hangs.

- **Feature compatibility:** no UNSUPPORTED feature in use. Two LANDED features fire (heads-ups). Full Appendix A scan:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` / `.global_circular_buffer` field |
  | Dynamic CircularBuffer (borrowed memory) | **GREEN** | `CBDescriptor::buffer` set on sharded path (`binary_ng_program_factory.cpp:570, 601, 661`: `.buffer = *_sharded ? *_buffer : nullptr`) → port uses `borrowed_from` |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset` set anywhere |
  | Aliased Circular Buffers | N/A | every `format_descriptors` initializer is single-element (`{{CBFormatDescriptor{…}}}`, lines 565–656) |
  | GlobalSemaphore | N/A | no semaphores of any kind in this op |
  | Non-zero semaphore initial value | N/A | no semaphores |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | **GREEN** | `RuntimeTensorShape` on all 3 tensors (`binary_ng_program_factory.cpp:700` c, `855` a, `858` b) → UNSAFE relaxation opt-in; heads-up below |
  | `UpdateCircularBuffer*` | N/A | no `UpdateCircularBuffer*` calls |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is fixed-count (a + optional b + optional output); no runtime-varying CTA index in kernels |

- **Factory concept:** **GREEN (available; PORT WORK, currently moot under the Device 2.0 RED).** Single factory, no op-owned buffers/semaphores beyond CBs and tensors → Decision 4 (Advanced) does not fire. Concept = `ProgramSpecFactoryConcept` (implemented). Caching strategy to be confirmed at port time per `port_op_to_metal2_ttnn_factory.md`; note the interaction with the custom-hash deletion and the dynamic-shape relaxation (Heads-ups / Team-only below) — the op's current custom hash deliberately omits tensor shape, so the caching-strategy choice is coupled to whether `dynamic_tensor_shape` is adopted.

## Port-work summary  *(for the eventual port, once unblocked)*

- **Factory concept:** `ProgramSpecFactoryConcept` · caching strategy TBD (see Factory concept note).
- **Tensor bindings** (per binding): `a` / `b` / `c` (output) — all **clean**. Each uses `TensorAccessor` end-to-end: host plumbs `TensorAccessorArgs(*buffer, RuntimeTensorShape).append_to(cta, common_rta)` (lines 700/855/858) and the base address flows via RTA into `TensorAccessor(args, addr)` (and the rm 3-arg `TensorAccessor(args, addr, page_size)`) kernel-side. No buffer-address-RTA bypass, no Case 2 bridge. (The rm kernels' raw `noc_async_read` is a Device-2.0 transfer issue, *not* a TensorAccessor-handling case — addressing already goes through the accessor.) Re-express each as `TensorParameter`/`TensorBinding`; the `TensorAccessorArgs` plumbing and base-address RTAs disappear.
  - **b-accessor placeholder:** when `b` is absent (scalar op), the reader binds `*a_buffer` as b's accessor (`binary_ng_program_factory.cpp:858`). The port must reproduce the conditional-binding shape (b bound only when present), not a placeholder.
- **Custom hash:** **delete** custom `compute_program_hash` → default (sanctioned exception). Located `device/binary_ng_device_operation.cpp:487`. It hashes `attributes`, `input_tensor_a.dtype()/.memory_config()`, `input_tensor_b->dtype()/.memory_config()`, and `shard_volumes` — and **omits tensor shape**. See relaxation note in Team-only.
- **Borrowed-memory DFBs:** the `a`/`b`/`c` CBs (`c_0`/`c_1`/`c_2`) borrow the tensor buffer on the sharded path (`.buffer = *_sharded ? *_buffer : nullptr`). Port with `DataflowBufferSpec::borrowed_from` on those, gated on the sharding condition. (Row-major and sharding are mutually exclusive — `binary_ng_program_factory.cpp:667` `TT_FATAL(!has_sharding, "Row-major binary_ng path does not support sharded tensors yet")`.)

## Heads-ups

- **Notable LANDED constructs:**
  - **Borrowed-memory DFB** — `binary_ng_program_factory.cpp:570, 601, 661` (CBs `c_0`/`c_1`/`c_2`, `.buffer` set on sharded path) → `DataflowBufferSpec::borrowed_from`.
  - **Dynamic TensorAccessor (`ArgConfig::RuntimeTensorShape`)** — `binary_ng_program_factory.cpp:700` (output `c`), `855` (input `a`), `858` (input `b`). **All three tensor bindings** use the runtime-shape flavor: the tensor shape is an implicit common runtime arg, so the program is built shape-independent. Metal 2.0 expresses this via `TensorParameterAdvancedOptions::dynamic_tensor_shape = true` (full) or `match_padded_shape_only = true`, both documented **UNSAFE** with per-dispatch-caching implications. **Adopting the relaxation is an explicit user-OK decision, not an automatic port step** — and it is coupled to the custom-hash deletion (see Team-only). If the relaxation is *not* adopted, the default-strict hash will key per shape (correct, but more cache entries / behavior change vs today).
- **Cross-op / shared kernels:** none. No kernel file is instantiated from another op family or a shared pool; no out-of-op function-call escapes. (See Team-only for the full include inventory.)
- **RTA varargs:** none of the porter-actionable kind. The runtime tensor shape is consumed internally by `TensorAccessor`, not via a hand-rolled `get_arg_val(i)` loop in op kernel code.

## Team-only

- **TensorAccessor convertibility:** N/A — no Case-2 bindings; all bindings are clean Case-1-equivalent (already on `TensorAccessor`).
- **Out-of-directory coupling & donor shape:** **✓ clean.** Every out-of-directory `#include` resolves to `tt_metal/*` HAL/LLK headers (`api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h`, `api/alignment.h`, `api/compute/*`) — no concern. The only non-HAL cross-directory includes are *in-op* cross-subdirectory references: `kernels_ng/` kernels include helpers from `kernels/` (`fill_tile_utils.hpp`, `eltwise_utils.hpp`, `eltwise_utils_common.hpp`, `eltwise_utils_sfpu.hpp`). No cross-family donors, no kernel-lib (`ttnn/cpp/ttnn/kernel_lib/`) escapes, no file-path instantiation of another op's kernels. No port-together coupling set.
- **Relaxation candidates** (mined from the custom hash before deletion — **FALLIBLE, candidates to verify; default strict**): the custom `compute_program_hash` (`binary_ng_device_operation.cpp:487`) keys on dtype + memory_config + shard_volumes and **omits `TensorSpec`/shape**. Combined with the `RuntimeTensorShape` accessor usage on all three tensors, this is a coherent signal that the op was designed to reuse one cached program across differing tensor shapes → `TensorParameter::advanced_options.dynamic_tensor_shape = true` on `a`/`b`/`c` is a candidate relaxation. **Caveat:** custom hashes are frequently themselves wrong; verify against the op's actual shape-dependence before adopting. This is the explicit-user-OK decision referenced in the dynamic-TensorAccessor heads-up.

## Runtime kernel-source selection  *(sizing note for the port)*

The single factory selects its reader/writer/compute source files at runtime via `get_kernel_file_path` (`binary_ng_utils.cpp:81`) over two roots — `device/kernels` (tile, older) and `device/kernels_ng` (newer tile + row-major). Selection axes: `SubtileBroadcastType` (none / scalar / row / col / row-col-mixed), `is_sfpu`, `is_where_op`, scalar-vs-tensor `b`, and `inputs_row_major` (`binary_ng_program_factory.cpp:556, 687`). The port's atomic unit is this factory + **every** source it can select (~34 kernel files). The row-major branch (`binary_ng_program_factory.cpp:687–695`, `get_reader_rm_kernel_name_and_defines`, `WriterRmNoBcastNg`) is the part that pulls in the Device-2.0-blocking rm kernels.

## Misc anomalies  *(team-only, non-gating; route to op owner, not the port)*

- **Missing `.cpp` extension** — `binary_ng_utils.cpp:125`: the `ComputeScalar` / `is_where_op` branch returns `"eltwise_where_sfpu_scalar"` (no `.cpp`), unlike its siblings (e.g. line 120 `"eltwise_where_sfpu.cpp"`). `device/kernels/compute/eltwise_where_sfpu_scalar.cpp` does exist, so this would resolve to a nonexistent path for the where + scalar compute kernel. Latent bug, unrelated to the port.
- **`num_tiles_per_cycle = 2` for SFPU multi-tile** — `binary_ng_program_factory.cpp:548–550` carries a comment that the SFPU kernel "should handle 4, but for unknown reason, only 2 works … need further investigation." Pre-existing TODO; not port scope.

## Questions for the user

1. **Device 2.0 broad-vs-isolated boundary on the rm kernels — RESOLVED (moot).** Originally I classified the rm kernels' raw `noc_async_read`/`noc_async_write` as a Device 2.0 **RED** (legacy primary-transfer idiom), *not* the YELLOW isolated-holdover carve-out. That judgment call is now moot: the rm kernels were migrated on this branch (see Device 2.0 gate detail), so the classification no longer gates anything. The underlying recipe gap it exposed still stands — see Recipe notes.

## Recipe notes  *(friction with the audit recipe itself)*

- **Gap: "raw NoC transfer alongside Device 2.0 accessors" has no clear home in the Device 2.0 tier rules.** `port_op_to_metal2_audit.md` Check 2 lists "raw `noc_async_read`" under the **RED** "broadly uses legacy Device 1.0 idioms" criterion, while the **YELLOW** carve-out is scoped *only* to the "CB-index-keyed free-function family … where the corresponding Device-2.0 wrapper-method replacement exists." The binary_ng rm kernels are a third shape the rules don't name: a kernel that is *structurally* on Device 2.0 (uses `TensorAccessor`, `CircularBuffer` wrappers, and `noc.async_*_barrier()` methods) but whose actual NoC *transfer* call is the legacy free function. "Broadly" felt wrong (the kernel is mostly D2.0); "isolated CB-index holdover" is wrong (it's not a CB-index function and it's the primary transfer). I resolved it as RED, but a one-line rule for "legacy `noc_async_read`/`noc_async_write` transfer = RED regardless of surrounding D2.0 idiom" would remove the judgment call. Surfaced as a Question above.
- **`ArgConfig::RuntimeTensorShape` recognition was easy** — Appendix A already names `binary_ng_program_factory.cpp` as an example site, which ground-truthed the match immediately. Helpful; no change needed.

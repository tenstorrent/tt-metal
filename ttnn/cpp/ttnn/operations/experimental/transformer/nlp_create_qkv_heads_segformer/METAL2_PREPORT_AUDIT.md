# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer`

- **`NlpCreateHeadsSegformerDeviceOperation`** (`device/nlp_create_qkv_heads_segformer_device_operation.hpp:15`)
  - *(no named factory)*: the op is in the **direct-descriptor** shape. `create_descriptor` is a static member of the device-op (`device/nlp_create_qkv_heads_segformer_device_operation.hpp:21-27`, body `device/nlp_create_qkv_heads_segformer_program_factory.cpp:19-155`), and there is no `program_factory_t`. It is driven through `MeshDeviceOperationAdapter::DirectDescriptorFactory` (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170`; `HasDirectDescriptor`, `ttnn/api/ttnn/operation_concepts.hpp:158`).
  - Kernels used (both op-owned, both bound only by this op):
    - reader: `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp`
    - writer: `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp`
  - No compute kernel. No unreferenced kernel files.

The op splits `[B, 1, S, hidden]` (TILE; `hidden % 32 == 0`; interleaved output only) into Q heads `[B, hidden/32, S, 32]`, with head_dim hard-coded to 32 (`device_operation.cpp:70-75`). The device-op allocates and returns a 3-tuple `(q, k, v)`, all with the same spec (`device_operation.cpp:76,89-92`). It also accepts preallocated `optional_output_tensors` (empty or exactly 3, `device_operation.cpp:42-52,81-87`). **Only Q is written**; the factory touches only `std::get<0>(output)` (`program_factory.cpp:61-64`). Every caller takes `[0]` (`models/demos/vision/segmentation/segformer/tt/ttnn_segformer_efficient_selfattention.py:99-101,149`; `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_segformer.py:26`). This is a fresh audit. The op's last two commits are PD batch #57409 (`f5093e705ae`, which moved it to the direct descriptor and deleted `nlp_create_qkv_heads_segformer_program_factory.hpp`) and Device 2.0 cleanup #58858 (`8e04962ad72`).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(hash from the `Port_Recipe` checkout)*

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer` |
| **Overall** | **GREEN (user waiver)**. The original finding was RED: the TTNN factory concept gate fails because the readiness sheet is stale (spreadsheet-broken). The user waived it on 2026-10-07. Every other gate is GREEN. |
| **DOps / Factories** | `NlpCreateHeadsSegformerDeviceOperation` → direct `create_descriptor` (no factory struct; the sheet still names `NlpCreateQkvHeadsSegformerProgramFactory`) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok. No function-call escapes outside `tt_metal/`, and no borrowed kernel files. |
| *Feature Support* — overall | GREEN (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok (none) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Waived by the user (2026-10-07).** Original finding: No, **spreadsheet-broken**. The cell reads `yes (with PD step)`, but the row conflicts with the code on `Concept` and names a phantom factory. |
| *TTNN Readiness* — Concept (current) | Code: `descriptor` (direct-descriptor shape). Sheet: `legacy device-op` (stale). |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No (sheet `no`; no `compute_program_hash` / `attribute_values`) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No (sheet `n/a`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No. `nlp_create_qkv_heads_segformer_nanobind.cpp:19-30` binds only the user function. |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (sheet `Porting Target` agrees) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | `input` Case 1 · `q` Case 1 (K/V outputs are never touched by a kernel, so they get no binding) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none: no accessor passes a 3rd arg |
| *Port work* — CB endpoints | legal: CB 1 is a plain 1:1 |

## Result

**GREEN (user waiver) → brief issued.** On 2026-10-07 the user waived the TTNN factory concept gate ("Yes run the audit on them as well. If it's just the stale sheet, waive it"). The original finding is kept below. The sheet refresh is still owed to the readiness-sheet owner.

*Original finding:* **RED → blocked on the TTNN factory concept gate (spreadsheet-broken), routed to the readiness-sheet owner.** The live readiness sheet (fetched 2026-10-07) still lists the op as before PD batch #57409: `Concept` = `legacy device-op`, factory `NlpCreateQkvHeadsSegformerProgramFactory` at the deleted `device/nlp_create_qkv_heads_segformer_program_factory.hpp`, and `Is able to port?` = `yes (with PD step)`. This is the only RED. The blocker clears off-code, and the op has one code path.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **Waived by the user (2026-10-07)**, original finding kept. **RED: spreadsheet-broken**, routed to the readiness-sheet owner (Diego). Two broken-sheet triggers fire:
  1. **`Concept` conflict.** The sheet says `legacy device-op`; the code has a static `create_descriptor` returning `ProgramDescriptor` (`device_operation.hpp:24-27`).
  2. **Phantom factory row.** `NlpCreateQkvHeadsSegformerProgramFactory` no longer exists, and #57409 deleted its definition file (`f5093e705ae`: "…segformer_program_factory.hpp | 35 ------").

  The other primary columns cross-check clean: `Custom hash` `no`, backdoor hash `no`, `get_dynamic_runtime_args` `no`, `Override runtime args method?` `n/a`, `Pybind descriptor` `no`, `Smuggled pointer` `no` (input/q addresses are `Buffer*` bindings), `Op-owned tensors?` `no`. `Known op issues` is empty and the relaxation is `none`. No invariant is violated. **Path forward:** refresh the row to the post-#57409 shape. This is the same pattern as the other #57409 ops.
- **Device 2.0 (every kernel used):** **GREEN.**
  - Reader: `Noc` (`:13`), `CircularBuffer cb_qv(1)` with method calls (`:34,40-44`), and `noc.async_read(s0, CoreLocalMem…)` (`:42`).
  - Writer: `Noc` (`:14`), `CircularBuffer cb_qv(1)` (`:33,49-59`), and `noc.async_write(CoreLocalMem…, sq, …)` / `async_write_barrier` (`:51-58`).
  - The only CB-index free function is `get_tile_size(cb_id_qv)` (reader `:35`, writer `:34`), which is sanctioned. There are no `*AddrGen*` and no raw `noc_async_*` calls.
- **Feature compatibility:** **GREEN.** No gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | The one `CBDescriptor` (`program_factory.cpp:105-113`) has no GCB field. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Not set. |
  | GlobalSemaphore | N/A | No semaphores. |

- **CB endpoints (GATE-free):** **legal.** CB 1 (`program_factory.cpp:103-113`; `2 × per_tensor_tiles` tiles, page = one tile, over `all_cores`) has the reader as a locked producer (`reserve_back`/`push_back`, reader `:40,44`) and the writer as a locked consumer (`wait_front`/`pop_front`, writer `:49,59`; the `get_read_ptr` peek at `:50` is covered). Plain 1:1, single config.
- **Offset base pointers:** **GREEN.** Address RTAs: reader RTA 0 `in0_buffer` and writer RTA 0 `q_buffer`, both bare `Buffer*` (`program_factory.cpp:129,143`). Reader RTA 1 `in1_buffer_addr` is the constant `0` (`:32,130`). It is not a buffer address, and the kernel never uses it (see Misc anomalies). The op is not in the offset triage doc. Outcome: clean.
- **TensorAccessor 3rd argument:** **N/A.** Both accessors are 2-arg (reader `:32`, writer `:31`). The op is not in the triage doc.

## Port-work summary  *(mirrors the brief)*

- **Factory shape (forced, `ttnn_factory.md` §3):** add a nested factory struct `NlpCreateQkvHeadsSegformerProgramFactory` with `create_program_artifacts` and `program_factory_t`, and remove the device-op-level `create_descriptor` (`device_operation.hpp:21-27`).
- **Tensor bindings:**
  - `input`: Case 1. A `Buffer*` RTA (`program_factory.cpp:129`) feeds `TensorAccessor(in0_args, in0_tensor_addr)` (reader `:16,26,32`).
  - `q`: Case 1. A `Buffer*` RTA (`:143`) feeds `TensorAccessor(q_args, q_tensor_addr)` (writer `:17,27,31`).
  - K/V outputs: no kernel touches them, so they get no binding. Leave them allocated and returned exactly as today; that is device-op code, not the factory.
- **TensorParameter relaxation:** none. **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal. CB 1 has the reader as PRODUCER and the writer as CONSUMER.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none. Both kernels are op-owned and bound by no other op, so convert them in place. Their filenames match `nlp_create_qkv_heads`'s kernels, but these are separate private copies; bind by path. `nlp_create_qkv_heads_vit` has its own copies too.
- **RTA varargs:** none. All reads are at fixed indices.
- **Unused reader RTAs.** The reader reads `in1_tensor_addr` (RTA 1, host value `0`) and `in1_tensor_tile_id` (RTA 4, host value `0u`) but never uses them (reader `:17,20`; host `:130,133`). Carry them as named args with the same constant values. Don't bind `in1_tensor_addr` as a tensor (there is no in1 tensor), and don't delete them in the port.
- **CB→DFB.** Both kernels include `api/dataflow/circular_buffer.h` and build `CircularBuffer cb_qv(cb_id_qv)` with `cb_id_qv = 1` (reader `:8,28,34`; writer `:9,29,33`). Apply the whitelisted swap to `DataflowBuffer` from a `dfb::` token. `get_tile_size(cb_id_qv)` moves to the DFB's tile-size accessor (whitelist rule 7).
- **Mutated RTAs.** The writer reassigns `q_out_h_dim` and `q_out_tensor_tile_id` (`:67-76`), and the reader increments `in0_tensor_tile_id` (`:45`). Use non-`const` locals.
- **Node order and tests.** Column-major cores (`program_factory.cpp:116`) over the device's full compute grid. Tests: `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_segformer.py`, including a program-cache test (`:91-107`). They check Q only, so they can run on the 8×8 Wormhole here.

## Team-only

- **Out-of-directory coupling:** ✓ clean. All includes are `tt_metal/hw/inc/api/*` (class 1), and there are no borrowed kernel files.
- **Relaxation candidates:** none.
- **TTNN factory analysis:** `descriptor` (direct). No op-owned tensors, no custom hash, no `get_dynamic_runtime_args`, no `override_runtime_arguments`, no pybound descriptor. Preallocated outputs come in through `tensor_args_t::optional_output_tensors`. Target: `ProgramSpecFactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **K and V outputs are allocated and returned but never written** (`device_operation.cpp:76,89-92` vs `program_factory.cpp:61-64`). Every in-tree caller takes only `[0]`, so the effect is two wasted output allocations per call and uninitialized tensors in the returned tuple. When the caller passes `output_tensors`, the K/V slots are required (`device_operation.cpp:47-51`) but ignored.
- **The nanobind docstring is wrong.** It says "Shuffles [B, 1, S, 2304] fused qkv matrix into 3 heads with shapes [B, 12, S, 64]…" (`nlp_create_qkv_heads_segformer_nanobind.cpp:23`), copied from `_vit`. The op actually produces Q heads of head_dim 32 for any `hidden % 32 == 0`.
- **Unused reader RTAs:** `in1_tensor_addr` / `in1_tensor_tile_id` (reader `:17,20`; host `program_factory.cpp:32,130,133`), leftovers from the `nlp_create_qkv_heads` lineage.
- **`q_out_h_tiles = ashape[2] / TILE_WIDTH`** (`program_factory.cpp:43`) should be `TILE_HEIGHT`. Both are 32, so it is harmless.
- **Unreachable code after `TT_FATAL(false, …)`** in the sharded branch of `compute_output_specs` (`device_operation.cpp:57-66`).

## Recipe notes

- **`Is able to port?` = `yes (with PD step)`** is outside the documented vocabulary (also noted on `nlp_create_qkv_heads_falcon7b`).
- **The unused-arg case isn't covered.** A positional RTA that the kernel reads but never uses (here a dummy `0` sitting where an address would go) is neither the dead-CB case nor a named-arg question the recipe addresses. I defaulted to "carry it unchanged as a named arg" under zero-functional-change. Explicit guidance would help.

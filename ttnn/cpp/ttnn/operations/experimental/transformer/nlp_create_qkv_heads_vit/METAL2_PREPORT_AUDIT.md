# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit`

- **`NlpCreateHeadsVitDeviceOperation`** (`device/nlp_create_qkv_heads_vit_device_operation.hpp:15`)
  - *(no named factory)*: the op is in the **direct-descriptor** shape. `create_descriptor` is a static member of the device-op (`device/nlp_create_qkv_heads_vit_device_operation.hpp:21-27`, body `device/nlp_create_qkv_heads_vit_program_factory.cpp:19-236`), and there is no `program_factory_t`. It is driven through `MeshDeviceOperationAdapter::DirectDescriptorFactory` (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170`; `HasDirectDescriptor`, `ttnn/api/ttnn/operation_concepts.hpp:158`).
  - Kernels instantiated on the live path (both op-owned, both bound only by this op):
    - reader: `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp`
    - writer: `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp`
  - Kernel named only on a **compile-time-dead** path: `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`, under `const bool transpose_k_heads = false;` (`program_factory.cpp:98-121`). The factory never instantiates it today.

The op splits ViT's fused QKV `[B, 1, S, 2304]` (TILE, interleaved output only, `device_operation.cpp:30-34`) into Q, K and V, each `[B, 12, S, 64]`, returned as a `std::vector<Tensor>` of 3. Preallocated outputs come in as `std::optional<std::vector<std::optional<Tensor>>>` (exactly 3, `device_operation.cpp:36-48`). This is a fresh audit. The op's last commits are Device 2.0 cleanup #58858 (`8e04962ad72`), PD batch #57409 (`f5093e705ae`, which moved it to the direct descriptor and deleted `nlp_create_qkv_heads_vit_program_factory.hpp`), and #55611 (an unused-`in1_args` warning fix in the reader).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(hash from the `Port_Recipe` checkout)*

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit` |
| **Overall** | **GREEN (user waiver)**. The original finding was RED: the TTNN factory concept gate fails because the readiness sheet is stale (spreadsheet-broken). The user waived it on 2026-10-07. Every other gate is GREEN. |
| **DOps / Factories** | `NlpCreateHeadsVitDeviceOperation` → direct `create_descriptor` (no factory struct; the sheet still names `NlpCreateQkvHeadsVitProgramFactory`) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes. Reader and writer are Device 2.0. `transpose_wh.cpp` is never instantiated; its Metal 2.0 fork is DFB-native anyway. |
| *Prereqs* — Cross-op escapes | Ok. No function-call escapes outside `tt_metal/`. The only borrowed kernel file is on the dead path, and its `_metal2` fork exists. |
| *Feature Support* — overall | GREEN (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok (none) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Waived by the user (2026-10-07).** Original finding: No, **spreadsheet-broken**. The cell reads `yes (with PD step)`, but the row conflicts with the code on `Concept` and names a phantom factory. |
| *TTNN Readiness* — Concept (current) | Code: `descriptor` (direct-descriptor shape). Sheet: `legacy device-op` (stale). |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No (sheet `n/a`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No. `nlp_create_qkv_heads_vit_nanobind.cpp:19-30` binds only the user function. |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (sheet `Porting Target` agrees) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | `input` Case 1 · `q` Case 1 · `k` Case 1 · `v` Case 1 |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none: no accessor passes a 3rd arg |
| *Port work* — CB endpoints | legal. Live: CB 1 is 1:1. Dead `transpose_k_heads` config: CBs 0, 1 and 16 are each 1:1 and allocated only under that config (an existing host-side conditional). |

## Result

**GREEN (user waiver) → brief issued.** On 2026-10-07 the user waived the TTNN factory concept gate ("Yes run the audit on them as well. If it's just the stale sheet, waive it"). The original finding is kept below. The sheet refresh is still owed to the readiness-sheet owner.

*Original finding:* **RED → blocked on the TTNN factory concept gate (spreadsheet-broken), routed to the readiness-sheet owner.** The live readiness sheet (fetched 2026-10-07) still lists the op as before PD batch #57409: `Concept` = `legacy device-op`, factory `NlpCreateQkvHeadsVitProgramFactory` at the deleted `device/nlp_create_qkv_heads_vit_program_factory.hpp`, and `Is able to port?` = `yes (with PD step)`. This is the only RED, and it clears off-code. The user has decided the dead `transpose_k_heads` branch is carried over faithfully (2026-10-07; see Questions).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **Waived by the user (2026-10-07)**, original finding kept. **RED: spreadsheet-broken**, routed to the readiness-sheet owner (Diego). Two broken-sheet triggers fire:
  1. **`Concept` conflict.** The sheet says `legacy device-op`; the code has a static `create_descriptor` returning `ProgramDescriptor` (`device_operation.hpp:24-27`).
  2. **Phantom factory row.** `NlpCreateQkvHeadsVitProgramFactory` no longer exists; #57409 deleted its definition file (`f5093e705ae`: "…vit_program_factory.hpp | 35 ----").

  The other primary columns cross-check clean: `Custom hash` `no`, backdoor hash `no`, `get_dynamic_runtime_args` `no`, `Override runtime args method?` `n/a`, `Pybind descriptor` `no`, `Smuggled pointer` `no`, `Op-owned tensors?` `no`. `Known op issues` is empty and the relaxation is `none`. No invariant is violated. **Path forward:** refresh the row.
- **Device 2.0 (every kernel used):** **GREEN.**
  - Reader: `Noc` (`:13`), `CircularBuffer cb_qv` / `cb_k` with method calls (`:45-46,53-58,64-76,81-93`), and `noc.async_read(…CoreLocalMem…)`.
  - Writer: `Noc` (`:14`), `CircularBuffer cb_qv` / `cb_k` (`:46-47`, methods at `:66-76,94-104,122-132`), and `noc.async_write` / `async_write_barrier`.
  - The only CB-index free functions are `get_tile_size(cb_id_qv)` / `get_tile_size(cb_id_k)` (reader `:47-48`, writer `:48-49`), which are sanctioned.
  - `transpose_wh.cpp` is not instantiated on any reachable path. It is a `CircularBuffer`-wrapper compute kernel, and its `_metal2` fork is DFB-native.
- **Feature compatibility:** **GREEN.** No gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No GCB field on any `CBDescriptor` (`program_factory.cpp:147-155,164-184`). |
  | CBDescriptor `address_offset` (non-zero) | N/A | Not set. |
  | GlobalSemaphore | N/A | No semaphores. |

- **CB endpoints (GATE-free):** **legal in both configs.**
  - *Live config* (`transpose_k_heads == false`, the only reachable one): CB 1 (`:145-155`; `2 × per_tensor_tiles` = 144 tiles, page = one tile). The reader is the one locked producer: its `cb_qv` and `cb_k` wrappers both resolve to index 1 (`cb_id_k = 1` under `#ifndef TRANSPOSE_K_HEADS`, reader `:31-36`). The writer is the one locked consumer: `cb_qv` and `cb_k` both resolve to index 1 (writer `:36-41`). That makes two distinct touchers, so plain 1:1. The two wrapper objects per kernel are one binding each, not extra endpoints.
  - *Dead config* (`transpose_k_heads == true`, unreachable because the flag is `const false` at `:98`): CB 0 (reader → compute), CB 16 (compute → writer) and CB 1 (reader → writer, Q and V) are each 1:1. CBs 0 and 16 are allocated only in that config (`:161-185`), so the host already makes them conditional. Nothing needs dropping.
- **Offset base pointers:** **GREEN.**
  - Address RTAs: reader RTA 0 `in0_buffer` (`:201`), and writer RTAs 0-2 `q_buffer` / `k_buffer` / `v_buffer` (`:220-222`), all bare `Buffer*`.
  - Reader RTA 1 `in1_buffer_addr` is the constant `0` (`:32,202`). The kernel uses it only under `READ_FROM_INPUT_TENSOR_KV`, which is never defined.
  - The K/V output tile ids are separate scalars (`:211-215`).
  - The op is not in the offset triage doc; the doc's `nlp_create_qkv_heads` fused-QKV fold is a different op. Outcome: clean.
- **TensorAccessor 3rd argument:** **N/A.** All accessors are 2-arg (reader `:39,42`; writer `:42-44`). The op is not in the triage doc.

## Port-work summary  *(mirrors the brief)*

- **Factory shape (forced, `ttnn_factory.md` §3):** add `NlpCreateQkvHeadsVitProgramFactory` with `create_program_artifacts` and `program_factory_t`, and remove the device-op-level `create_descriptor` (`device_operation.hpp:21-27`).
- **Tensor bindings:**
  - `input`: Case 1. A `Buffer*` RTA (`:201`) feeds `TensorAccessor(in0_args, in0_tensor_addr)` (reader `:16,26,39`).
  - `q`, `k`, `v`: Case 1. `Buffer*` RTAs (`:220-222`) feed `TensorAccessor(q/k/v_args, …)` (writer `:17-19,32-34,42-44`).
  - There is no in1 tensor. The empty `TensorAccessorArgs()` placeholder (`:85`) backs only the `#ifdef READ_FROM_INPUT_TENSOR_KV` `in1_args` (reader `:27-29`), which is never compiled. It disappears with the rest of the accessor plumbing, and no binding is created for it.
- **TensorParameter relaxation:** none. **TensorAccessor 3rd arg:** none.
- **CB endpoints:** live CB 1 has the reader as PRODUCER and the writer as CONSUMER. Dead-config CBs 0 and 16 are each 1:1, and are already conditional on `transpose_k_heads` in the host.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding):** none.
- **The dead `transpose_k_heads` branch** (`program_factory.cpp:97-121,157-185,213-215`; kernel `#ifdef TRANSPOSE_K_HEADS` blocks). See Questions. The brief's default is to carry it over faithfully: keep `const bool transpose_k_heads = false;`. Under it, add the two compute `KernelSpec`s (core_group_1 / core_group_2, named CTA `NHtWt` = `num_blocks_per_core_group_N * kv_num_tiles`) bound to `ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp`, the conditional DFBs for CBs 0 and 16, and the `TRANSPOSE_K_HEADS` defines. The fork's interface is `dfb::in`, `dfb::out` and named CTA `NHtWt` (fork `:18-24`). This path cannot be exercised by any test.
- **Two wrappers, one DFB.** On the live path the reader's and the writer's `cb_qv` and `cb_k` both name index 1. In Metal 2.0, both must come from the **same** `dfb::` token (e.g. `dfb::qv`) under `#ifndef TRANSPOSE_K_HEADS`, and from the K-specific tokens under `#ifdef`. Keep the `#ifdef` structure.
- **Cross-op / shared kernels:** the reader and writer are op-private (their filenames match the `nlp_create_qkv_heads` / `_segformer` copies, but they are different files), so convert them in place. Only the dead path borrows `transpose_wh.cpp` (shared compute pool `ttnn/cpp/ttnn/kernel/compute/`). Its `_metal2` fork exists beside it: bind it and don't re-fork. Fork consumers today: `data_movement/permute` (tiled), `data_movement/transpose` (WH), and `experimental/transformer/nlp_create_qkv_heads`.
- **RTA varargs:** none.
- **Unused-in-live-config reader RTAs.** `in1_tensor_addr` (RTA 1, host `0`) and `in1_tensor_tile_id` (RTA 4, host `0u`) are read unconditionally (reader `:17,20`) but used only under `READ_FROM_INPUT_TENSOR_KV`. Carry them as named args with value `0`; they are not a tensor binding.
- **CB→DFB swap** in both kernels (`api/dataflow/circular_buffer.h`, reader `:8`, writer `:9`). `get_tile_size(cb_id_*)` moves to the DFB accessor (whitelist rule 7).
- **Mutated RTAs.** Writer `q_out_h_dim`, `q_out_tensor_tile_id`, `k_out_tensor_tile_id` and `v_out_tensor_tile_id` (`:140-161`), and reader `in0_tensor_tile_id` / `in1_tensor_tile_id`, so use non-`const` locals.
- **Node order and tests.** Column-major cores (`program_factory.cpp:187-188`) over the device's full compute grid. Tests: `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_vit.py` (`:100`, plus a program-cache test at `:104`). They run on the 8×8 Wormhole here.

## Team-only

- **Out-of-directory coupling:** ✓ clean. Kernel includes are all `tt_metal/hw/inc/api/*` (class 1). Borrowed kernel file: `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` (shared compute pool, class 3), on the dead path only. Its legacy copy is still bound by `experimental/transformer/nlp_create_qkv_heads_boltz` and `experimental/transformer/split_query_key_value_and_split_heads` (plus this op's dead path); its `_metal2` fork exists.
- **Relaxation candidates:** none.
- **TTNN factory analysis:** `descriptor` (direct). No op-owned tensors, no custom hash, no `get_dynamic_runtime_args`, no `override_runtime_arguments`, no pybound descriptor. Target: `ProgramSpecFactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Compile-time-dead `transpose_k_heads` path** (`program_factory.cpp:97-121,157-185,213-215`): a hard-coded `const bool … = false` gating a compute kernel, two CBs, defines and an alternate K tile-id formula. It is untestable as-is. If no ViT user needs transposed K, the ops team may want to remove it, or plumb it through as a real attribute the way `nlp_create_qkv_heads` does.
- **`READ_FROM_INPUT_TENSOR_KV` is never defined.** The empty `TensorAccessorArgs()` CTA placeholder (`:85`) and the dummy in1 RTAs (`:32,202,205`) are dead plumbing from the `nlp_create_qkv_heads` lineage. #55611 already silenced an unused-`in1_args` warning that came from this.
- **Hard-coded geometry.** `q_num_tiles_per_tensor = 24`, `num_q_heads = num_kv_heads = 12` and `q_out_w_tiles = 2` (`:38-45`) are fixed to the validated 2304-wide input. That is consistent with validation, but it would silently mis-split any relaxed shape.
- **`compute_output_specs` sharded path** (`device_operation.cpp:53-55`) does `TT_ASSERT(false); return {};`. In a release build an empty spec vector reaches `create_output_tensors`. Validation already rejects non-interleaved output, so this can't be reached from the public path.

## Questions for the user

1. **What should the port do with the dead `transpose_k_heads` branch?** *Answered 2026-10-07: carry it faithfully ("Keep the behaviour unchanged, stick with the default").* The brief defaults to **carrying it over faithfully** (zero functional change). That adds untestable conditional compute/DFB specs that bind `transpose_wh_metal2.cpp`. The alternative is to leave the `if (false)` branch out of the ProgramSpec and record it in the port report as dead code not carried. That changes no behaviour and keeps the diff smaller, but it is a scope call the porter shouldn't make alone. Context: `program_factory.cpp:98` (`const bool transpose_k_heads = false;`).

## Recipe notes

- **`Is able to port?` = `yes (with PD step)`** is outside the documented vocabulary (also noted on `nlp_create_qkv_heads_falcon7b`).
- **The recipe doesn't cover compile-time-dead branches.** It covers configs that are *live* under some instantiation ("classify per instantiation"), but not a branch guarded by a hard-coded `const false`, where no instantiation reaches it. The dead-CB rule ("dead in every config → drop") almost applies to CBs 0 and 16, but the CBs are only *allocated* in that unreachable config, so they are not dead CBs in the recipe's sense. Guidance on whether to carry or drop such branches would close this gap.

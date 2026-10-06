# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors`

- **`CreateQKVHeadsSeparateTensorsDeviceOperation`** (`device/create_qkv_heads_from_separate_tensors_device_operation.hpp:17`) is the only DeviceOperation in the directory.
  - **Direct-descriptor factory.** `create_descriptor` is a static member of the device-op itself, and there is no `program_factory_t` (`device/create_qkv_heads_from_separate_tensors_device_operation.hpp:26-29`; body at `device/create_qkv_heads_from_separate_tensors_program_factory.cpp:16-183`). The framework wraps it in the `DirectDescriptorFactory` shim. Before PR #57409 the device-op declared `using program_factory_t = std::variant<CreateQKVHeadsSeparateTensorsProgramFactory>;` (`git show f5093e705ae^:…/create_qkv_heads_from_separate_tensors_device_operation.hpp:22`).
  - One code path with a single config switch, `transpose_k_heads` (Python default `true`, `create_qkv_heads_from_separate_tensors_nanobind.cpp:34`). It controls two kernels:
    - reader `device/kernels/reader_create_qkv_heads_sharded_separate.cpp`. Owned by the op, `ReaderConfigDescriptor`, always present (`program_factory.cpp:71-82`). Under `transpose_k_heads` it gets the define `TRANSPOSE_K_HEADS=1`.
    - compute `experimental/transformer/split_query_key_value_and_split_heads/device/kernels/compute/transpose_wh_sharded.cpp`. **Borrowed**, in-family. Present only when `transpose_k_heads` is set (`program_factory.cpp:84-101`).

    There is no writer kernel. All five tensors are block-sharded and reached through globally-allocated (borrowed) CBs. The reader performs local-core NoC reads from the input shards into the output shards.

The op splits a `[B, 1, S, Hq]` Q tensor and a fused `[B, 1, S, 2·Hkv]` KV tensor into Q `[B, nq, S, d]`, K `[B, nkv, S, d]` (or `[B, nkv, d, S]` when transposed) and V `[B, nkv, S, d]`. Its Python entry point is `ttnn.experimental.create_qkv_heads_from_separate_tensors` (`create_qkv_heads_from_separate_tensors_nanobind.cpp:21`). The in-tree callers are the Stable Diffusion cross-attention demo and two unit tests.

**Scope:** TTNN op, Gen1 (WH/BH) target. This is within the scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
*(The recipe tree isn't in this checkout (`Metal_Ports`, branch `edwinlee/PD_Metal_Ports`). The hash comes from the `Port_Recipe` checkout (branch `akertesz/op-porting-recipe`). This audit followed `/localdev/edwinlee/metal2_audit.md`, which is a symlink to that checkout's `ai/audit/metal2_audit.md`.)*

**Readiness sheet:** fetched live on 2026-10-06 via the Google Drive connector (`download_file_content`, CSV). One row for this op.

**History since the last relevant change:** this is the first audit of the op. Recent commits on the op:
- `f5093e705ae` #57409 (2026-09-25): PD migration. It deleted `device/create_qkv_heads_from_separate_tensors_program_factory.hpp` (−36 lines) and moved `create_descriptor` onto the device-op.
- `8e04962ad72` #58858 (2026-10-01): Device 2.0 cleanup. It deleted the unused `get_dataformat(cb_inq)` local in the reader.
- `35e83c6252f` #58382: a zero-head-count validation fix.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors` |
| **Overall** | **GREEN (user waiver).** Every code-side gate is clean. The only RED, the stale readiness-sheet row from PD batch #57409, was waived by the user on 2026-10-06 (see Result). |
| **DOps / Factories** | `CreateQKVHeadsSeparateTensorsDeviceOperation` → direct `create_descriptor` (no factory struct). The sheet still lists `CreateQKVHeadsSeparateTensorsProgramFactory`, which was deleted in #57409. |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** The reader uses `Noc`, `CircularBuffer`, `UnicastEndpoint` and `CoreLocalMem`. The borrowed compute kernel uses `CircularBuffer` for all FIFO sync. The only CB-index DM free function is the sanctioned `get_tile_size(cb_id)`. |
| *Prereqs* — Cross-op escapes | Ok. All includes are `tt_metal/hw/inc/*` (class 1). One borrowed kernel file (in-family compute, see Heads-ups). |
| *Feature Support* — overall | GREEN (all N/A) |
| *Feature Support* — Variadic-CTA | Ok. Reader CTAs are fixed at 0–6, and compute uses CTA 0. |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet says `yes (with PD step)`, but the row is **stale**: `Concept` conflicts with the code, and the factory row is a phantom. Treated as **spreadsheet-broken**, which makes this a GATE routed to the readiness-sheet owner. |
| *TTNN Readiness* — Concept (current) | Code: **`descriptor`** (direct-descriptor shape, since #57409). Sheet: `legacy device-op`. |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No. The default hash is used. Sheet agrees (`no` / backdoor `no`). |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No. Sheet agrees. |
| *TTNN Readiness* — `override_runtime_arguments` | No: the code has none. The PD migration replaced it with CB `.buffer` re-pegging (`program_factory.cpp:108-111`). The sheet says `n/a`, which fits its stale `legacy` concept. |
| *TTNN Readiness* — Pybind `create_descriptor` | No. The only binding is the user entry point via `bind_function` (`create_qkv_heads_from_separate_tensors_nanobind.cpp:21`). Sheet agrees. |
| *TTNN Readiness* — Op-owned tensors | No. Sheet agrees. |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept`. The sheet's `Porting Target` column agrees. The port must first introduce a factory struct (see Heads-ups). |
| *Port work* — Offset base pointer | none. The op has **no runtime args at all**, so there are no address RTAs. |
| *Port work* — Tensor bindings (per binding) | All five are **clean** (borrowed-memory DFB reads/writes): `input_tensor` (Q in), `input_tensor_kv`, `output_q`, `output_k`, `output_v`. |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none. The op constructs no `TensorAccessor`. |
| *Port work* — CB endpoints | **self-loop** for every borrowed CB, in both configs. CB 24 is a plain 1:1 and exists only under `transpose_k_heads`; it is already conditional in the legacy factory. |

**CB endpoints** are dispositions, not gates. See Port-work summary for the per-`(CB, config)` table.

## Result

**GREEN by user waiver → brief issued** (`METAL2_PORT_BRIEF.md`). As audited, this was RED on the TTNN factory concept gate (spreadsheet-broken). On 2026-10-06 the user waived that gate ("Waive"): the blocker is only the sheet being out of date, not anything in the op. The sheet refresh is still owed to the readiness-sheet owner as housekeeping; it is no longer a port blocker. The original finding is kept below.

**Original finding (pre-waiver):** RED → blocked on the TTNN factory concept gate (spreadsheet-broken), routed to the **readiness-sheet owner** (Diego, `dgomez@tenstorrent.com`).

The live sheet (fetched 2026-10-06) still describes the op as it was before PR #57409, *[Cleanup] Port More Ops to PD* (`f5093e705ae`, 2026-09-25). That PR deleted the `CreateQKVHeadsSeparateTensorsProgramFactory` struct and its header `device/create_qkv_heads_from_separate_tensors_program_factory.hpp`, and moved `create_descriptor` onto the device-op. The sheet still shows:

- `Concept` = `legacy device-op`
- `Factory (variant)` = `CreateQKVHeadsSeparateTensorsProgramFactory`
- `Factory definition path` = that now-deleted `.hpp`

**This is the only RED, and it is the known stale-sheet pattern from the #57409 PD batch.** The same pattern was seen on `experimental/test/hang_device`, `experimental/transformer/concatenate_heads` and `experimental/topk_router_gpt`. Device 2.0, Feature compatibility, Offset base pointers and TensorAccessor 3rd argument are all clean on the code.

**This RED is cleared outside the op's code.** The sheet owner updates the row; nothing in the op changes. Per the recipe's exception, I ran all the informational subjects. The detail below should survive re-audit unchanged, so once the row is refreshed (`Concept` = `descriptor`, factory row renamed or removed, `Is able to port?` re-derived), the re-audit should be a quick re-read and issue the brief. *(Superseded: the user waived this sheet-only gate on 2026-10-06, and the brief is issued.)*

The op has a single code path, with one config switch that every gate is clean under. Whole-op RED, with no subset distinction (there is nothing to split).

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED (spreadsheet-broken)**, routed to the readiness-sheet owner to reconcile. The sheet row is `Op` = `experimental/transformer/create_qkv_heads_from_separate_tensors`, `Device operation` = `CreateQKVHeadsSeparateTensorsDeviceOperation`, `Factory (variant)` = `CreateQKVHeadsSeparateTensorsProgramFactory`.
  - **Primary-column conflict, `Concept`.** The sheet says `legacy device-op`. The code is `descriptor`: `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` on the device-op (`device/create_qkv_heads_from_separate_tensors_device_operation.hpp:26-29`), with no `create()` and no `override_runtime_arguments`. The device-op's own comment at `:23-25` records the change.
  - **Phantom factory row.** `CreateQKVHeadsSeparateTensorsProgramFactory` no longer exists. `Factory definition path` points at `device/create_qkv_heads_from_separate_tensors_program_factory.hpp`, which #57409 deleted (diffstat: `…_program_factory.hpp | 36 -----`). There is also a **missing row**: the code's only factory is the device-op's direct `create_descriptor`, and no row matches it.
  - `Is able to port?` = `yes (with PD step)`. That PD step has since landed. This is a derived cell, so it is read, not vetted. It is moot here, because the conflicts above already make the row spreadsheet-broken.
  - **The rest of the primary cross-check is clean:**
    - `Custom hash` = `no` and backdoor = `no`. There is no `compute_program_hash`, `attribute_values` or `to_hash` in the op.
    - `Runtime-args update (get_dynamic_runtime_args)` = `no`. There is no hook.
    - `Override runtime args method?` = `n/a`. The code has none.
    - `Pybind descriptor` = `no`. There is no `create_descriptor` binding.
    - `Op-owned tensors?` = `no`.
    - `Smuggled pointer` = `no`. The op emits no runtime args at all.

    No cross-column invariant is violated. `Known op issues` is empty. `Execution Model` = `SPMD` and `Porting Target` = `ProgramSpecFactoryConcept` are consistent with the code.
  - **Path forward:** the sheet owner refreshes the row, or the user waives this sheet-only gate. Re-audit after a refresh is expected to go GREEN, since there are no code-side blockers.

- **Device 2.0 (every kernel used): GREEN.**
  - **Reader** (`device/kernels/reader_create_qkv_heads_sharded_separate.cpp`, owned):
    - `Noc noc` (`:13`)
    - five `CircularBuffer` objects (`:48-52`), with `reserve_back` / `push_back` on the three outputs (`:66,84,93,130,136,155`) and `get_read_ptr` / `get_write_ptr` as object methods (`:57-58,87,94,137`)
    - every transfer is `noc.async_read(src_ep, CoreLocalMem<uint32_t>(dst), size, {.noc_x, .noc_y, .addr}, {})` with `UnicastEndpoint src_ep` (`:59,72-77,103-108,117-122,143-148`), followed by `noc.async_read_barrier()` (`:83,129,154`)
  - **The one CB-index free function** in the reader is `get_tile_size(cb_inq)` (`:41`), which is **sanctioned**.
  - `my_x[noc_id]` / `my_y[noc_id]` (`:55-56`, with `noc_id` from `noc.get_noc_id()`) supply the own-core coordinates for a local loopback read into the `UnicastEndpoint` address struct. This is not a CB-index free function, an addr-gen or a raw semaphore. The same idiom appears in Device 2.0 kernels elsewhere (e.g. `data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_wh_sharded_rm.cpp:36,49`). Not a violation.
  - No raw `noc_async_*`, no `*AddrGen*`, no `get_noc_addr`, no semaphores.
  - #58858 (2026-10-01) removed the last non-compliant line, an unused `get_dataformat(cb_inq)`.
  - **Compute (borrowed, `split_query_key_value_and_split_heads/device/kernels/compute/transpose_wh_sharded.cpp`, owning family `experimental/transformer`):**
    - `CircularBuffer cb_im0_obj(c_24)` with `wait_front` / `pop_front` (`:20,28,34`)
    - `CircularBuffer cb_out1_obj(c_17)` with `reserve_back` / `push_back` (`:21,36,42`)
    - The CB-index calls `compute_kernel_hw_startup` / `transpose_init` / `transpose_tile` / `pack_tile` (`:14-15,31,39`) are compute LLK APIs, outside the data-movement scope of Device 2.0.

- **Feature compatibility: GREEN** (no gate fired).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | There are no `GlobalCircularBuffer`, `global_circular_buffer` or `remote_*` signals. All six `CBDescriptor`s (`program_factory.cpp:114-180`) leave `.global_circular_buffer` unset. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` is never set (default 0). Five CBs are buffer-backed at offset 0 via `.buffer = <tensor>.buffer()`. |
  | GlobalSemaphore | N/A | The op has no semaphores at all. |

- **CB endpoints (GATE-free): self-loop plus one plain 1:1.** See Port-work summary.

- **Offset base pointers: GREEN.** The factory sets no `runtime_args` or `common_runtime_args` on either kernel (`program_factory.cpp:71-101`), and the kernels call no `get_arg_val`. There is no address RTA, so nothing can carry a folded offset. Tensor memory reaches the kernel only through borrowed CBs at offset 0. The per-head and per-sequence-tile offsets the reader adds (`:71,102,116,142`) are kernel-side arithmetic on `get_read_ptr()`, not host folds. The op is not in the `2026-07-19_offset_base_pointers.md` tables, and its scan is clean (the "no fold, not in tables" outcome). *(The tables list `nlp_create_qkv_heads`, which is a different op.)*

- **TensorAccessor 3rd argument: N/A.** The subject never fires: neither kernel constructs a `TensorAccessor`. The op is not in the `2026-07-06_tensor_accessor_3rd_arg_triage.md` table either.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): all **clean** (borrowed-memory DFB). The port expresses each as a `TensorParameter` plus a `DataflowBufferSpec` with `borrowed_from`.

  | Tensor | CB | Factory site | Kernel access |
  |---|---|---|---|
  | `input_tensor` (Q in) | `c_0` | `program_factory.cpp:114-123` | reader `get_read_ptr` (`:57`), local NoC read source |
  | `input_tensor_kv` | `c_1` | `:125-134` | reader `get_read_ptr` (`:87`) |
  | `output_q` | `c_16` | `:137-146` | reader `reserve_back` / `get_write_ptr` / `push_back` (`:58,66,84`) |
  | `output_k` | `c_17` | `:148-157` | reader FIFO-produces when `!transpose_k` (`:30,93-94,130`); compute FIFO-produces when `transpose_k` (`transpose_wh_sharded.cpp:36,42`) |
  | `output_v` | `c_18` | `:159-168` | reader `reserve_back` / `get_write_ptr` / `push_back` (`:136-137,155`) |

- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints**, per `(CB, config)`, one node set (`all_cores` = the Q shard grid):

  | CB | `transpose_k = false` | `transpose_k = true` |
  |---|---|---|
  | `c_0` Q in (borrowed) | reader, raw peek only → **self-loop** (reader) | same |
  | `c_1` KV in (borrowed) | reader, raw peek only → **self-loop** (reader) | same |
  | `c_16` Q out (borrowed) | reader is the locked producer and nothing consumes it → **self-loop** (reader) | same |
  | `c_17` K out (borrowed) | reader is the locked producer (`cb_outk = c_17`, `:30`) → **self-loop** (reader) | compute is the locked producer; the reader does not touch it (`cb_outk = c_24`, `:28`) → **self-loop** (compute) |
  | `c_18` V out (borrowed) | reader is the locked producer → **self-loop** (reader) | same |
  | `c_24` K intermediate (not borrowed) | not allocated (`program_factory.cpp:170`) | reader is the locked producer (`:93,130`), compute is the locked consumer (`transpose_wh_sharded.cpp:28,34`) → **plain 1:1** |

  - No dead CBs.
  - `c_24` is already conditional host-side (`if (transpose_k)`, `program_factory.cpp:170-180`), so its conditional DFB spec is a direct translation, not new structure.
  - **The `c_17` producer flips with config**, from reader to compute.
  - Hidden-2nd-writer hunt: no kernel raw-writes a CB another kernel FIFO-produces, and the op has no semaphores.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. Nothing needs the multi-binding flag in either config.
- **Cross-op / shared kernels:** the compute kernel `experimental/transformer/split_query_key_value_and_split_heads/device/kernels/compute/transpose_wh_sharded.cpp` is **borrowed** (in-family).
  - **No `_metal2` fork exists beside it.** The directory holds only `transpose_wh_sharded.cpp`. Shared-kernel rung 2 applies: this port creates `transpose_wh_sharded_metal2.cpp` in that directory, plus the pointer comment in the original.
  - **Do not bind `data_movement/transpose/device/kernels/compute/transpose_wh_sharded_metal2.cpp`.** It forks a *different* kernel with the same stem: it is RTA-driven (`NHtWt`/`Ht`/`Wt`), works on a bulk `wait_front(NHtWt)`, and takes CB indices from CTAs. It is not a fork of this file.
  - Other binders of this exact file (**sunset list, not authorization to convert in place**):
    - `experimental/transformer/split_query_key_value_and_split_heads` (`split_query_key_value_and_split_heads_sharded_program_factory.cpp:114-116`)
    - `experimental/transformer/create_qkv_heads` (`create_qkv_heads_program_factory.cpp:167-169`)
  - The kernel hardcodes `c_24` / `c_17` (`:14-18`). Name the fork's bindings for the kernel's role (e.g. `dfb::in` / `dfb::out`), not for this op, because two other ops will inherit them.
- **RTA varargs:** none. The op has no RTAs or CRTAs. CTAs are fixed-index (reader `0..6`, compute `0`), so there are no CTA varargs either.
- **Direct-descriptor shape: the port must introduce a factory struct.** The op declares `create_descriptor` on the device-op with no `program_factory_t` (`device/create_qkv_heads_from_separate_tensors_device_operation.hpp:26-29`). Follow `ttnn_factory.md` §"3. Give a direct-descriptor op a conventional program factory":
  - nest a factory struct (e.g. `CreateQKVHeadsSeparateTensorsProgramFactory`, the pre-#57409 name) with `create_program_artifacts`
  - add `using program_factory_t = std::variant<...>;`
  - remove the device-op-level `create_descriptor`

  Keep the body in `device/create_qkv_heads_from_separate_tensors_program_factory.cpp`, and record the change under Handoff points. *(Check again at port time in case TTNN has added a `program_factory_t` since this audit.)*
- **`TRANSPOSE_K_HEADS` selects the reader's K-destination CB** (`reader :27-31`): `c_24` under transpose, `c_17` otherwise. In Metal 2.0, each branch of the existing `#ifdef` names its own `dfb::` token. The reader binds the intermediate DFB under transpose and the K-output DFB otherwise, so its binding set differs by config. Keep the `#ifdef` (the define already exists, `program_factory.cpp:79-81`). It is a syntax swap per branch.
- **`get_tile_size(cb_inq)` is used in a `constexpr`** (`reader :41`), and four more `constexpr`s derive from it (`:63-64,89-91`). The port moves it onto the DFB object (whitelist rule 7). Confirm that the DFB getter is usable in a constant expression before keeping `constexpr`. If it isn't, demote those locals to `const`. Don't restructure the arithmetic.
- **Stale kernel include.** The reader includes `api/dataflow/circular_buffer.h` (`:8`), and so does the compute donor (`:9`, in the fork only). It goes when `CircularBuffer` → `DataflowBuffer`.
- **Raw L1 walks over borrowed DFBs stay as is.** The reader computes source addresses as `get_read_ptr() + seq_tile_offset + head_offset` and walks the output `get_write_ptr()` linearly inside a single bulk `reserve_back` / `push_back` (`:57-84,87-155`). On Gen1 a borrowed DFB lowers to the same circular buffer, so keep the pointer arithmetic and the `my_x`/`my_y` loopback endpoint unchanged.

## Team-only

- **Out-of-directory coupling & donor shape: ✓ clean.**
  - Function-call escapes: every include resolves under `tt_metal/hw/inc/`, which makes them all donor class 1 (LLK/HAL), with no concern:
    - reader: `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h`
    - compute: `api/compute/compute_kernel_hw_startup.h`, `api/compute/transpose.h`, `api/dataflow/circular_buffer.h`
  - No op-owned donor functions are called, so there is no per-call table.

  | Op kernel | Donor file | Class | Status |
  |---|---|---|---|
  | reader | `tt_metal/hw/inc/api/*` | 1 | ✓ |
  | compute (borrowed) | `tt_metal/hw/inc/api/*` | 1 | ✓ |

  - **Borrowed kernel files:** one.
    - `ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/compute/transpose_wh_sharded.cpp`
    - Owner: `experimental/transformer/split_query_key_value_and_split_heads` (in-family).
    - Broadly shared: three binders, namely this op, `split_query_key_value_and_split_heads` (sharded factory) and `create_qkv_heads`.
    - `_metal2` fork: **none**. The `experimental/quasar/transpose/` tree has its own copies; they are out of bounds and not counted.
- **Relaxation candidates:** none (no custom hash to mine).
- **TTNN factory analysis:**
  - op-owned tensors: none
  - MeshWorkload need: none (plain `ProgramDescriptor`)
  - pybind `create_descriptor`: none
  - other risky pybind: none (only `bind_function` of the user API, `create_qkv_heads_from_separate_tensors_nanobind.cpp:21`)
  - custom hash: none
  - `get_dynamic_runtime_args`: none
  - `override_runtime_arguments`: none (the pre-PD one was replaced by CB `.buffer` re-pegging, `program_factory.cpp:108-111`)
  - target concept: **`ProgramSpecFactoryConcept`**

## Misc anomalies  *(team-only, non-gating)*

- **Validation computes `q_shard_ht` with the wrong core count.** `device_operation.cpp:127` divides by `num_w_cores * TILE_HEIGHT`, but the factory divides by `num_h_cores * TILE_HEIGHT` (`program_factory.cpp:44`). The validation value only feeds the `> 0` check (`:131`) and the L1 estimate (`:136,143`). On a non-square grid it can therefore reject a valid shape or under-estimate L1. Route to the ops team.
- **The L1 check counts borrowed CBs as extra L1.** `l1_size >= 2 * (per_core_q_tiles + 2 * per_core_k_tiles) * single_tile_size` (`device_operation.cpp:142-143`). However, five of the six CBs are backed by the tensors' own shards (`program_factory.cpp:122,133,145,156,167`). The only op-allocated L1 is `c_24` (`k_size`, and only under transpose). The check is over-restrictive and doesn't model the real footprint.
- **Reader CTA comments are wrong.** `reader :15-21`: CTA 0 is `q_shard_ht` but is commented "number of Q heads in the group". CTA 1 `q_shard_wt` is commented "number of K heads", CTA 2 `k_shard_ht` "number of V heads", CTA 3 `k_shard_wt` "size of a Q head in bytes", and CTA 6 `tiles_per_head` "size of a K head". The host-side names (`program_factory.cpp:61-69`) are the correct ones.
- **`optional_output_tensors` are not validated.** `create_output_tensors` returns the caller's tensors verbatim (`device_operation.cpp:222-225`). `validate_on_program_cache_miss` never checks them against `compute_output_specs` (shape, shard spec, dtype). This matters because the kernels write straight into their shards through borrowed CBs sized from the *input* geometry (`program_factory.cpp:103-106`).
- **The compute donor declares its CTA non-`constexpr`.** `uint32_t num_tiles = get_compile_time_arg_val(0);` (`transpose_wh_sharded.cpp:12`). This is harmless, but it is inconsistent with the reader.

## Questions for the user

1. **Waive the stale-sheet gate?** The only RED is the #57409 stale row (`Concept` = `legacy device-op`, phantom `CreateQKVHeadsSeparateTensorsProgramFactory`). This is the same shape you waived for `hang_device`, `concatenate_heads` and `topk_router_gpt`. **Answered 2026-10-06: waived.** The audit is "GREEN (user waiver)", the original finding is retained, and the brief is written.

## Recipe notes

- **Stale-sheet RED after a recent PD migration (repeat, 4th #57409 op).** The live sheet on 2026-10-06 still carries the pre-#57409 row: `legacy device-op`, a phantom factory, and `Is able to port?` = `yes (with PD step)`. The `yes (with …)` tag still isn't covered by the routing rules, and I treated it as moot again. A batch refresh of the #57409 rows, or a rule for a provisional brief when the *only* RED is "PD step landed after the sheet was derived", would save a round-trip per op.
- **`my_x[]` / `my_y[]` in Device 2.0 kernels.** The Device 2.0 gate has no explicit ruling on the firmware core-coordinate globals. They appear in the migration guide only inside a *legacy* example (`device_api_migration_guide.md:521`), yet Device 2.0 kernels use them to build `UnicastEndpoint` loopback addresses. I treated them as not a violation, because they aren't a CB-index free function, an addr-gen or a raw semaphore. A one-line ruling in the Green bullet (sanctioned or not) would remove the judgment call.
- **Compute-kernel CB-index LLK calls.** The Device 2.0 gate is framed around data movement. Compute LLK calls taking a CB index (`transpose_tile(cb, …)`, `pack_tile(…, cb)`) are presumably out of its scope, and I treated them that way. Stating it explicitly would help, since a borrowed compute kernel is in the gate's "every kernel used" scope.
- **Provenance in a separate checkout** (repeat): the provenance `git log` prints nothing in `Metal_Ports`. The hash above comes from the sibling `Port_Recipe` checkout.

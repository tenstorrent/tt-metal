# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice`

- **`NlpKVCacheLoadSliceDeviceOperation`** (`device/nlp_kv_cache_load_slice_device_operation.hpp`)
  - Single factory: `NlpKVCacheLoadSliceDeviceOperation::create_descriptor` (a static on the device-op itself, defined in `device/nlp_kv_cache_load_slice_program_factory.cpp:19`). There is no separate factory struct: PD batch #57409 (`f5093e705ae`) deleted `nlp_kv_cache_load_slice_program_factory.hpp` and its `NlpKVCacheLoadSliceProgramFactory` type.
- Kernels bound:
  - reader (own): `device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_shard_optimized.cpp`
  - writer (borrowed): `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp`
- No unreferenced kernel files in the op directory.
- This is a first audit; there was no earlier `METAL2_PREPORT_AUDIT.md`. Op history since the PD migration: only `f5093e705ae [Cleanup] Port More Ops to PD (#57409)`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice` |
| **Overall** | **GREEN (user waiver)**. The audit found RED on the stale readiness-sheet row only, and the user waived it on 2026-10-07. Every other gate is clear |
| **DOps / Factories** | `NlpKVCacheLoadSliceDeviceOperation` → `create_descriptor` (single program) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok (no kernel `#include` escapes; one borrowed kernel file, which already has a `_metal2` fork) |
| *Feature Support* — overall | GREEN (no Appendix A entry fires) |
| *Feature Support* — Variadic-CTA | Ok (none) |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet says `yes (with PD step)`, but the row is **spreadsheet-broken**: `Concept` conflicts with the code, and its factory row is phantom. → GATE, routed to the readiness-sheet owner |
| *TTNN Readiness* — Concept (current) | Code: `descriptor`. Sheet: `legacy device-op` (stale) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No (sheet `no`; code agrees) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet `no`; code agrees) |
| *TTNN Readiness* — `override_runtime_arguments` | No (sheet `n/a`; code has none) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (sheet `no`; `nlp_kv_cache_load_slice_nanobind.cpp` binds only the function) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none (input base delivered clean; tile offset is a separate `start_id` RTA) |
| *Port work* — Tensor bindings (per binding) | `input`: Case 1 · `output`: clean (borrowed-memory DFB backing c_0) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none (no accessor passes a 3rd arg) |
| *Port work* — CB endpoints | legal (c_0 is a plain 1:1) |

## Result

**GREEN (user waiver) → brief issued.** On 2026-10-07 the user waived the stale-sheet RED below ("waive"). The original finding is kept unchanged. The sheet row still needs reconciling by its owner.

*Original finding:* **RED: blocked on the TTNN factory concept gate, because the readiness sheet is broken for this op.** Routed to the **readiness-sheet owner** (Diego, `dgomez@tenstorrent.com`) to reconcile.

The sheet row predates PD-migration batch #57409 (`f5093e705ae`, 2026-09-25). That batch moved this op to `ProgramDescriptor` and deleted the `NlpKVCacheLoadSliceProgramFactory` type the row still names. **Every other gate is clear:** Device 2.0, Feature compatibility, Offset base pointers, and TensorAccessor 3rd argument. Once the row is reconciled, or the user waives it, the op is a clean port to `ProgramSpecFactoryConcept`. RED at op level; the op has a single program, so there is no subset to offer.

The blocker clears in the sheet, not in the op's code. So per the Red-outcome scoping exception, all seven informational subjects were run in full below, and they will hold on re-audit.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED (spreadsheet-broken).** There is exactly one sheet row for this op:
  - `Device operation` = `NlpKVCacheLoadSliceDeviceOperation`, `Factory (variant)` = `NlpKVCacheLoadSliceProgramFactory`, `Concept` = `legacy device-op`, `Op Classification` = `Legacy Op`, `Is able to port?` = `yes (with PD step)`, `TensorParameter relaxation` = `none`, `Known op issues` = *(empty)*, `Custom hash` = `no`, `Backdoor custom hash` = `no`, `Runtime-args update (get_dynamic_runtime_args)` = `no`, `Override runtime args method?` = `n/a`, `Pybind descriptor` = `no`, `Smuggled pointer` = `no`, `Op-owned tensors?` = `no`, `Porting Target` = `ProgramSpecFactoryConcept`, `Factory definition path` = `…/device/nlp_kv_cache_load_slice_program_factory.hpp`.

  Triggers, each grounded in code:
  1. **Primary-column conflict on `Concept`.** The sheet says `legacy device-op`. The code is `descriptor`: `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` at `device/nlp_kv_cache_load_slice_device_operation.hpp:27`, defined at `device/nlp_kv_cache_load_slice_program_factory.cpp:19`. There is no `create()` and no `override_runtime_arguments`.
  2. **Phantom factory row.** `NlpKVCacheLoadSliceProgramFactory` no longer exists. `f5093e705ae` deleted `nlp_kv_cache_load_slice_program_factory.hpp` (36 lines), which is also the row's `Factory definition path`. Conversely, the code's only program source, `NlpKVCacheLoadSliceDeviceOperation::create_descriptor`, has no row.

  Every other primary column cross-checks clean (custom hash, `get_dynamic_runtime_args`, override, pybind, op-owned tensors). No cross-column invariant is violated. The `yes (with PD step)` verdict reads as "portable once the PD migration lands", and that migration has landed, so this is staleness, not a hidden blocker. This is the same pattern as the other ops from batch #57409.
  **Path forward:** the readiness-sheet owner refreshes the row (Concept → `descriptor`, factory → `create_descriptor` on the device op, definition path → `nlp_kv_cache_load_slice_program_factory.cpp`), then a cheap re-audit. Alternatively, the user can waive the gate.
- **Device 2.0 (every kernel used):** **GREEN.**
  - The reader (`reader_unary_unpad_dims_interleaved_start_id_shard_optimized.cpp`) is fully Device 2.0. It uses `Noc noc` (`:19`), `TensorAccessor(src_args, src_addr)` (`:34`), `CircularBuffer cb_in0(cb_id_in0)` (`:36`), `cb_in0.reserve_back` / `get_write_ptr` / `push_back` (`:39,40,66`), `noc.async_read(s0, CoreLocalMem<uint32_t>(…), …)` (`:50`), and `noc.async_read_barrier()` (`:54,65`). Its one CB-index free function, `get_tile_size(cb_id_in0)` (`:33`), is sanctioned. There is no raw `noc_async_*`, no addr-gen, and no raw semaphore.
  - The borrowed writer (`data_movement/sharded/.../writer_unary_sharded.cpp`) is already on `DataflowBuffer` (`dfb_out.wait_front` / `pop_front`, `:30,33`).
- **Feature compatibility:** **GREEN** (no gate fired).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | The `CBDescriptor` at `program_factory.cpp:49-58` sets no `global_circular_buffer` field; no remote-CB idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | `address_offset` is not set (default 0); `.buffer = dst_buffer` is the ordinary borrowed-memory path |
  | GlobalSemaphore | N/A | No semaphores at all |

- **CB endpoints (GATE-free):** **legal.**
  - CB `c_0` (borrowed from the output shard buffer) has two touchers on every node.
    - The reader is a **locked producer**: `reserve_back(num_tiles)`, then raw-pointer fill through `get_write_ptr()`, then `push_back(num_tiles)`.
    - The writer is a **locked consumer**: `wait_front` / `pop_front`.
  - That makes c_0 a plain 1P+1C, with one config only. No hidden second writer and no dead CB.
- **Offset base pointers:** **GREEN.**
  - The only address RTA is reader RTA 0 (`src_addr`). It is delivered as a `Buffer*` (`emplace_runtime_args(core, {src0_buffer, start_id})`, `program_factory.cpp:98`), so it is a clean base with no host-side fold.
  - The slice start is applied as a separate **tile index**, `start_id`. That is `get_tiled_start_offset(a, output_tensor_start)` (`:92`, from `data_movement/slice`), advanced by `num_tiles_shifted_per_core` per core (`:101`), and consumed as `.page_id` on the accessor. This is the tiled-slice shape that the recipe calls unaffected.
  - The op is not listed in `2026-07-19_offset_base_pointers.md`, so this is the "no fold, not in tables" outcome.
- **TensorAccessor 3rd argument:** **N/A.** The op's one accessor, `TensorAccessor(src_args, src_addr)` (reader `:34`), passes no 3rd argument; #41902 removed the former one. The op is not in `2026-07-06_tensor_accessor_3rd_arg_triage.md`. The `tile_size` argument to `noc.async_read` (`:50`) is the read size, not an accessor page-size override.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings:**
  - `input`: **Case 1** (via `TensorAccessor`). Today it arrives as a `Buffer*` RTA (`program_factory.cpp:98`, which the framework patches on cache hit, so it is not a live hazard) plus `TensorAccessorArgs(src0_buffer)` CTAs appended at CTA offset 5 (`:73`). Port: `TensorParameter` / `TensorBinding`, with the kernel building `TensorAccessor(tensor::<name>)`. The `src_addr` RTA and the `TensorAccessorArgs<5>` plumbing both go away.
  - `output`: **clean.** It is the borrowed-memory backing of CB c_0 (`.buffer = dst_buffer`, `:57`); port it via `DataflowBufferSpec::borrowed_from` the output `TensorParameter`.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal. c_0: reader is PRODUCER, writer is CONSUMER.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** the writer `writer_unary_sharded.cpp` is borrowed from `data_movement/sharded`. **A `_metal2` fork already exists beside it**, `writer_unary_sharded_metal2.cpp` (from #51743), with binding vocabulary `dfb::out` (CONSUMER) and `args::num_units` (RTA). That fits this op exactly: the legacy writer gets CB index c_0 as CTA 0 and `num_tiles_per_core` as RTA 0 (`program_factory.cpp:89,99`). Use rung 1: bind the fork and adopt its names.
  - **Sunset list (not authorization to convert in place):** remaining legacy binders of `writer_unary_sharded.cpp` besides this op are `data_movement/sharded_partial/interleaved_to_sharded_partial`, `data_movement/untilize` (`…nd_shard_type_and_shard_spec_identical…` factory), and `experimental/padded_slice` (`padded_slice_rm`).
  - The reader is op-owned and bound by no other factory, so it converts in place.
- **RTA varargs:** none. Reader RTAs are two fixed fields (`src_addr`, `start_id`, reader `:21-22`). Reader CTAs are fixed indices 0–4 plus the accessor args. The writer has one RTA.
- **Other:**
  - The reader includes `api/dataflow/circular_buffer.h` and uses `CircularBuffer cb_in0` (`:9,36`). The Metal 2.0 port swaps it to `DataflowBuffer` via `dfb::` (whitelisted).
  - `get_tile_size(cb_id_in0)` at `:33` is a `constexpr` CB-index lookup that feeds the `constexpr` `barrier_threshold` template (`:43`). Moving it onto the DFB object (whitelist rule 7) must stay usable in a constant expression, or the porter keeps the free form on the token. Confirm which works rather than swapping blind.
  - Per-core `start_id` is structural, because the slice window is an attribute and therefore hashed. The header comment at `device_operation.hpp:23-26` says the bindings are the whole cache-hit refresh, which matches plain `ProgramSpecFactoryConcept`.
  - Runtime verification: the test is `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_kv_cache_load_slice.py`. Check its grid needs against the 8×8 Wormhole here before porting.
  - Do not look at `experimental/quasar/` for precedent.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.**
  - Function-call escapes: none. The reader's includes are all `tt_metal/*` (`dataflow_api.h`, `noc.h`, `circular_buffer.h`, `core_local_mem.h`, `noc_traits.h`). The writer's includes are `dataflow_api.h` and `dataflow_buffer.h`.

    | Op kernel | Donor file | Class | Status |
    |---|---|---|---|
    | reader (own) | `api/dataflow/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | `tt_metal/*` | ✓ |
    | writer (borrowed) | `api/dataflow/dataflow_api.h`, `dataflow_buffer.h` | `tt_metal/*` | ✓ |

  - Borrowed kernel files: `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp`, which is broadly shared and already forked (see Heads-ups for the fork and the sunset list). Fork binders already on Metal 2.0: `interleaved_to_sharded`, `tilize` sharded (×2), `tilize_with_val_padding` sharded, `transpose_wh_sharded`, `untilize` identical-shard, `reduce` h and w.
  - Host-side coupling (not a kernel escape): the factory includes `data_movement/slice/device/slice_device_operation.hpp` for `get_tiled_start_offset` (`program_factory.cpp:11,92`). It is a pure shape → tile-index helper and is unaffected by the port.
- **Relaxation candidates:** none (no custom hash).
- **TTNN factory analysis:**
  - Concept: `descriptor` (`create_descriptor` on the device op).
  - No op-owned tensors.
  - No custom or backdoor hash.
  - No `get_dynamic_runtime_args`, no `override_runtime_arguments`, no pybound `create_descriptor`.
  - Target concept: `ProgramSpecFactoryConcept`, which matches the sheet's `Porting Target`.
  - The sheet's `Uses llama kernels? (primary or shared)` = `yes` is informational. The only shared kernel is the sharded writer.

## Misc anomalies  *(team-only, non-gating)*

- `memory_config` is accepted and ignored: `[[maybe_unused]]` at `device_operation.cpp:121`. Output is always HEIGHT_SHARDED L1 (`compute_output_specs`, `:96-97`).
- `create_output_tensors` returns `preallocated_output` unvalidated (`device_operation.cpp:107-108`). The factory then assumes `output.shard_spec().value()` and a shard grid matching `fused_batch_heads` (`program_factory.cpp:30`). A mismatched preallocated output would fail or mis-address with no clear error.
- The per-core coordinate is `CoreCoord{i % num_cores_x, i / num_cores_x}` with `num_cores_x` taken from only the **first** range of the shard grid (`program_factory.cpp:33-34,96`). This is correct for the row-wise `num_cores_to_corerangeset` grid the op builds itself, but would break for a preallocated output whose shard grid has a narrower first range.

## Questions for the user

1. **Waiver?** The only RED was the stale sheet row from PD batch #57409. **Answered 2026-10-07: "waive".** `METAL2_PORT_BRIEF.md` was issued.

## Recipe notes

- The recipe's `Concept` cross-check describes `create_descriptor()` as a *factory* method. Here, as in other #57409 ops, `create_descriptor` is a static on the DeviceOperation with no factory type at all. So the "factory-set match" check pairs the sheet's factory row against a device-op method. I treated that as one factory (phantom name, real program). It would help if the recipe named this shape explicitly.

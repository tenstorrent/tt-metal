# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice`

> Audit cleared all gates. One gate was cleared by user waiver; see below. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user waiver: readiness-sheet row is stale, see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section; hash from the `Port_Recipe` checkout)*

**TTNN gate waiver (record in the port report).** The live readiness sheet (fetched 2026-10-07) still describes this op as it was before PR #57409:

- `Concept` = `legacy device-op`
- factory `NlpKVCacheLoadSliceProgramFactory`, at a now-deleted `.hpp`
- `Is able to port?` = `yes (with PD step)`

The code is already on a direct `create_descriptor`, so the audit flagged the sheet as broken. On 2026-10-07 the user waived that gate ("waive"), since the problem is only the out-of-date sheet. Every other primary column cross-checked clean. The sheet refresh is still owed to the readiness-sheet owner.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`, in the **direct-descriptor** shape. `create_descriptor` is a static member of `NlpKVCacheLoadSliceDeviceOperation` itself (`device/nlp_kv_cache_load_slice_device_operation.hpp:27-28`, body `device/nlp_kv_cache_load_slice_program_factory.cpp:19-108`), with no `program_factory_t`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. There is no `override_runtime_arguments`; the framework's binding refresh covers the cache hit. The slice window is an operation attribute, so it is hashed, which makes the per-core `start_id` structural (header comment at `device_operation.hpp:23-26`).
- **Gate-cleared, confirmed absent:** a non-clearing `TensorParameter relaxation` (cell is `none`) and `get_dynamic_runtime_args` (absent). The op also has no custom hash and no pybound `create_descriptor`. The only nanobind binding is the user function (`nlp_kv_cache_load_slice_nanobind.cpp:18-31`), so no pybind line needs deleting.

## Construct — to do

**Tensor bindings** (per binding):

- `input` — **Case 1** (via `TensorAccessor`). Today it reaches the reader as a `Buffer*` RTA, `emplace_runtime_args(core, {src0_buffer, start_id})` (`program_factory.cpp:98`), plus `TensorAccessorArgs(src0_buffer).append_to(...)` at CTA offset 5 (`:73`).
  - Port: express it as a `TensorParameter` / `TensorBinding`.
  - In the reader, replace `TensorAccessor(src_args, src_addr)` (reader `:34`) with `TensorAccessor(tensor::<name>)`.
  - Drop the `src_addr` RTA (reader `:21`) and `TensorAccessorArgs<5>()` (`:29`).
  - Name it from the kernel's vocabulary (`src_addr` → `tensor::src`).
- `output` — **clean** (borrowed-memory DFB). CB c_0 is backed by the output shard buffer (`.buffer = dst_buffer`, `program_factory.cpp:57`). Port it as a `DataflowBufferSpec` `borrowed_from` the output `TensorParameter`. There is no kernel-side tensor access.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none. No accessor passes one.

**CB endpoints:** all legal. The one CB, `c_0`, is a plain 1:1:
- PRODUCER: the reader (`reserve_back` → raw fill via `get_write_ptr()` → `push_back`, reader `:39-66`).
- CONSUMER: the writer (`wait_front` / `pop_front`).

**Remaining args → named:**
- Reader RTA `start_id` (reader `:22`).
- Reader CTAs 0–4: `num_tiles`, `num_unpadded_tiles_head_dim`, `num_unpadded_tiles_seqlen_dim`, `num_padded_tiles_seqlen_dim`, `num_readers` (`:24-28`).
- Writer: per the fork's vocabulary below.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** the writer `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` is borrowed. **A `_metal2` fork already exists at `…/writer_unary_sharded_metal2.cpp` — bind it, don't re-fork (rung 1).**
  - The fork's vocabulary is `dfb::out` (bound CONSUMER) and `args::num_units` (RTA). That fits: the legacy writer took the CB index as CTA 0 and `num_tiles_per_core` as RTA 0 (`program_factory.cpp:89,99`). So name the c_0 DFB `out` in the writer's bindings and supply `num_units = num_tiles_per_core`.
  - Leave the legacy original alone; it already carries its pointer comment.
  - Other legacy binders of the original, for the port report: `data_movement/sharded_partial/interleaved_to_sharded_partial`, `data_movement/untilize` (nd-shard identical-spec factory), `experimental/padded_slice` (`padded_slice_rm`). **This is a sunset list, not authorization to convert the kernel in place.**
  - The reader is op-owned and bound by no other factory, so convert it in place.
- **RTA varargs:** none. All reader and writer args are fixed fields, so name each one.
- **DFB swap in the reader:** the reader includes `api/dataflow/circular_buffer.h` and builds `CircularBuffer cb_in0(cb_id_in0)` from `constexpr uint32_t cb_id_in0 = 0` (reader `:9,31,36`). Swap these to a `DataflowBuffer` on `dfb::<name>` (whitelisted).
- **`constexpr` tile size:** `constexpr uint32_t tile_size = get_tile_size(cb_id_in0)` (reader `:33`) feeds the `constexpr` `get_barrier_read_threshold<tile_size, num_readers>()` template (`:43`). Moving the lookup onto the DFB object (whitelist rule 7) must stay usable in a constant expression. Confirm which form compiles (member getter vs. the token-form free function) rather than swapping blind. Do not change the barrier logic.
- **Host-side helper:** `get_tiled_start_offset` comes from `data_movement/slice/device/slice_device_operation.hpp` (`program_factory.cpp:11,92`). It is a pure shape → tile-index helper; keep using it unchanged.
- **Verification:** the test is `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_kv_cache_load_slice.py`. The op needs `B × n_heads` cores, so check the test's grid needs against the 8×8 Wormhole here before starting.
- Do not use anything under `experimental/quasar/` as a precedent.

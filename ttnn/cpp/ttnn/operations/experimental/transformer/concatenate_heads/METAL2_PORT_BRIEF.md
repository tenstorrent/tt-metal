# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/concatenate_heads`

> Audit cleared all gates. One gate was cleared by user waiver; see below. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user waiver: readiness-sheet row is stale, see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section; hash from the `Port_Recipe` checkout)*

**TTNN gate waiver (record in the port report).** The live readiness sheet (fetched 2026-10-01) still describes this op as it was before PR #57409:

- `Concept` = `legacy device-op`
- factory `ConcatenateHeadsProgramFactory`, at a now-deleted `.hpp`
- `Is able to port?` = `yes (with PD step)`

The code is already on a direct `create_descriptor`, so the audit flagged the sheet as broken. On 2026-10-01 the user waived that gate, since the problem is only the out-of-date sheet. The sheet refresh is still owed to the readiness-sheet owner.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`, in the **direct-descriptor** shape. `create_descriptor` is a static member of `ConcatenateHeadsDeviceOperation` itself (`device/concatenate_heads_device_operation.hpp:27-28`, body `device/concatenate_heads_program_factory.cpp:20-143`), with no `program_factory_t`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. There is no `override_runtime_arguments`; the framework's binding refresh covers the cache hit. Per-core tile ids derive from the padded shape, which the hash covers.
- **Gate-cleared, confirmed absent:** a non-clearing `TensorParameter relaxation` (cell is `none`) and `get_dynamic_runtime_args` (absent). The op also has no custom hash and no pybound `create_descriptor`. The only nanobind binding is the user function (`concatenate_heads_nanobind.cpp:28-36`), so no pybind line needs deleting.

## Construct — to do

**Introduce a factory struct (forced; `ttnn_factory.md` §3 "Give a direct-descriptor op a conventional program factory").** The `DirectDescriptorFactory` shim (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170`, gated by `HasDirectDescriptor`, `ttnn/api/ttnn/operation_concepts.hpp:158`) only recognizes `create_descriptor`, and has no `create_program_artifacts` counterpart. In `device/concatenate_heads_device_operation.hpp`:

1. Nest `struct ConcatenateHeadsProgramFactory` (the pre-#57409 name) with `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);`.
2. Add `using program_factory_t = std::variant<ConcatenateHeadsProgramFactory>;`.
3. Remove the device-op-level `create_descriptor` and its comment (`:24-28`).

Keep the body in `device/concatenate_heads_program_factory.cpp`. It is already registered at `experimental/transformer/sources.cmake:10`, so no build edits are needed. Record this under Handoff points: the op arrived in the direct-descriptor shape. *(Check first that no `program_factory_t` has appeared since the audit.)*

**ProgramSpec contents:** two DM `KernelSpec`s and one DFB, all over the same `all_cores` rectangle. The rectangle is `(0,0)`–`(H/32 − 1, B − 1)`, i.e. 12 × B (`concatenate_heads_program_factory.cpp:48-52,70-72`).

- **reader** — `device/kernels/dataflow/reader_tm_tile_layout_concat_heads.cpp`, reader config.
  - Named CTAs: `in0_w_tiles`, `in0_c`, `in0_HtWt` (`:74-79`). The `TensorAccessorArgs` append (`:80`) disappears into the binding.
  - Per-node RTA: `in0_tensor_tile_id` = `core_x * in0_w_tiles + core_y * in0_CHtWt` (`:122`).
- **writer** — `device/kernels/dataflow/writer_tm_tile_layout_concat_heads.cpp`, writer config.
  - Named CTAs: `in0_w_tiles`, `in0_c` (`:81-85`). The `TensorAccessorArgs` append (`:86`) disappears.
  - Per-node RTA: `out_tensor_tile_id` = `(core_x + core_y * num_cores_c) * per_core_tiles` (`:134`).
- No CRTAs, defines or semaphores.

**Tensor bindings** (per binding):

- `input` — **Case 1** (via `TensorAccessor`). Today the host passes it as a `Buffer*` RTA (`concatenate_heads_program_factory.cpp:127`), and the reader builds `TensorAccessor(in0_args, in0_tensor_addr)` (reader `:24,28`). Express it as a `TensorParameter` / `TensorBinding`, and have the kernel build `TensorAccessor(tensor::<name>)`. The address RTA (reader `:16`) and the `TensorAccessorArgs<3>` plumbing both go. RTA 1 `in0_tensor_tile_id` stays, as a named RTA.
- `output` — **Case 1**. Today the host passes a `Buffer*` RTA (`:133`), and the writer builds `TensorAccessor(out_args, out_tensor_addr)` (writer `:25,29`). Same translation: the address RTA (writer `:18`) and `TensorAccessorArgs<2>` go, and `out_tensor_tile_id` stays as a named RTA.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none. Both accessors are 2-arg.

**CB endpoints:** one DFB, all legal. CB 0 (`src0_cb_index`, `concatenate_heads_program_factory.cpp:107-117`) holds `per_core_tiles * 2` tiles of `tile_size(dtype)`, with page = one tile. Bind the reader as **PRODUCER** (`reserve_back`/`push_back`) and the writer as **CONSUMER** (`wait_front`/`pop_front`). It's a plain 1:1; no flag. Keep the 64-tile capacity as is (the zero-functional-change contract), even though only 32 tiles are used. See Watch for.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none. Both kernel files are private to this op, and this factory is their only binder, so convert them in place. No `_metal2` fork is needed. Do not borrow anything from `experimental/quasar/`.
- **RTA varargs:** none. Each kernel has two fixed RTAs, and after the port each has one named RTA (the tile id). CTAs are fixed-index; name them all.
- **Kernels are already Device 2.0.** They use `Noc`, `CircularBuffer`, `TensorAccessor` and `CoreLocalMem`, so the kernel change is a binding-layer swap, not an idiom rewrite:
  - `CircularBuffer cb_x(0)` → `DataflowBuffer` from the `dfb::` token
  - drop the `api/dataflow/circular_buffer.h` include (reader `:8`, writer `:9`)
  - `get_tile_size(cb_id)` → `dfb.get_tile_size()` (reader `:27`, writer `:28`; whitelist rule 7)
  - accessor from `tensor::` token
  - named RTAs
- **Preserve the linear pointer walk exactly.** Both kernels capture `get_write_ptr()` / `get_read_ptr()` once, before the loop (reader `:31`, writer `:32`), and advance it linearly without wrap. The writer's `wait_front` count is cumulative (`:33,36,44`), and it does a single `pop_front` at the end (`:48`). That `pop_front` pops 34 tiles against 32 pushed; this is a known anomaly, routed to the ops team. **Do not fix it in the port**: carry `pop_front(out_num_tiles_read)` over verbatim.
- **Verification.** The direct test is `models/experimental/bert_large_performant/unit_tests/test_bert_large_concatenate_heads.py`. It covers batch 7/8/9, bf16/bf8_b, and DRAM/L1 in and out.
  - It needs a ≥ 12 × 9 compute grid. Note that the test's own skip checks only `x*y ≥ 108`, while validation needs `x ≥ 12 && y ≥ 9`. In practice that means Blackhole; on an 8×8 Wormhole it skips or fails validation.
  - `test_bert_large_concatenate_heads_with_program_cache` is the cache-hit check for the new tensor bindings. It shifts allocations with a dummy tensor between runs and asserts 2 cache entries.
  - `tests/ttnn/unit_tests/operations/transformers/test_concatenate_heads.py` tests a **different** op (`ttnn.transformer.concatenate_heads`). Don't count it.

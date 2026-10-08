# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b`

> Audit cleared all gates. One gate was cleared by user waiver; see below. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user waiver: readiness-sheet row is stale, see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section; hash from the `Port_Recipe` checkout)*

**TTNN gate waiver (record in the port report).** The live readiness sheet (fetched 2026-10-07) still describes this op as it was before PR #57409:

- `Concept` = `legacy device-op`
- factory `NlpCreateQkvHeadsFalcon7BProgramFactory`, at the now-deleted `device/nlp_create_qkv_heads_falcon7b_program_factory.hpp`
- `Is able to port?` = `yes (with PD step)`

The code is already on a direct `create_descriptor`, so the audit flagged the sheet as broken. On 2026-10-07 the user waived that gate ("Okay waive it if it's just the stale sheet"). The sheet refresh is still owed to the readiness-sheet owner.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`, in the **direct-descriptor** shape. `create_descriptor` is a static member of `NlpCreateHeadsFalcon7BDeviceOperation` itself (`device/nlp_create_qkv_heads_falcon7b_device_operation.hpp:27-30`, body `device/nlp_create_qkv_heads_falcon7b_program_factory.cpp:19-155`), with no `program_factory_t`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. There is no `override_runtime_arguments`; the framework's binding refresh covers the cache hit. Per-core args derive from the padded shape and the compute grid (`device_operation.hpp:23-26` says the same).
- **Gate-cleared, confirmed absent:** a non-clearing `TensorParameter relaxation` (cell is `none`) and `get_dynamic_runtime_args` (absent). The op also has no custom hash and no pybound `create_descriptor`. The only nanobind binding is the user function (`nlp_create_qkv_heads_falcon7b_nanobind.cpp:19-28`), so no pybind line needs deleting.

## Construct — to do

**Introduce a factory struct (forced; `ttnn_factory.md` §3 "Give a direct-descriptor op a conventional program factory").** The `DirectDescriptorFactory` shim (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170`, gated by `HasDirectDescriptor`, `ttnn/api/ttnn/operation_concepts.hpp:158`) only recognizes `create_descriptor`. In `device/nlp_create_qkv_heads_falcon7b_device_operation.hpp`:

1. Nest `struct NlpCreateQkvHeadsFalcon7BProgramFactory` (the pre-#57409 name, which the sheet still uses) with `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);`.
2. Add `using program_factory_t = std::variant<NlpCreateQkvHeadsFalcon7BProgramFactory>;`.
3. Remove the device-op-level `create_descriptor` and its comment (`:23-30`).

Keep the body in `device/nlp_create_qkv_heads_falcon7b_program_factory.cpp` (already in `experimental/transformer/sources.cmake:36`). Record this under Handoff points. *(Check first that no `program_factory_t` has appeared since the audit.)*

**ProgramSpec contents.** There is a single code path: two DM `KernelSpec`s over `all_cores` (from `split_work_to_cores(compute_with_storage_grid_size, num_blocks)`, `program_factory.cpp:48-53`) and one DFB.

- **reader**: bind the existing fork `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp` (see Watch for). Reader config, no defines.
  - Named RTAs: `num_pages` = `num_blocks_per_core * per_tensor_tiles`, `start_id` = `num_blocks_written * per_tensor_tiles` (`:129-130`).
  - RTA 0 (`in0_buffer`) becomes `tensor::src`. The `TensorAccessorArgs` CTA append (`:75`) disappears into the binding.
- **writer**: `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads_falcon7b.cpp` (op-owned, convert in place), writer config.
  - Named CTAs: `q_num_tiles`, `kv_num_tiles`, `q_out_h_tiles`, `q_out_w_tiles`, `q_out_c`, `q_out_HtWt` (`program_factory.cpp:76-83`, writer `:26-31`). The three `TensorAccessorArgs` appends (`:84-86`, writer `:32-34`) disappear into the bindings.
  - Per-node named RTAs: `num_blocks`, `q_out_h_dim`, `q_out_tensor_tile_id`, `kv_out_tensor_tile_id` (`program_factory.cpp:142-145`, writer `:20-23`). RTAs 0-2 (`q_buffer`, `k_buffer`, `v_buffer`, `:139-141`) become `tensor::` bindings.
  - Node order: `CoreCoord core = {i / num_cores_y, i % num_cores_y}` for `i < num_cores` (`:118-119`), with group-1/group-2 block counts (`:121-127`) and a running `num_blocks_written` (`:148`). Keep it.

**Tensor bindings** (per binding):

- `input`: **Case 1** (via `TensorAccessor`). Today it is a `Buffer*` RTA (`:130`), and the reader builds `TensorAccessor(src_args, src_addr)`. Bind it as the fork's `tensor::src`.
- `q`, `k`, `v`: **Case 1**. Today they are `Buffer*` RTAs (`:139-141`), and the writer builds `TensorAccessor(q_args, q_tensor_addr)` etc. (writer `:38-40`). Express each as a `TensorParameter` / `TensorBinding`, and have the writer build `TensorAccessor(tensor::<name>)`. The address RTAs (writer `:17-19`) and the accessor-args CTAs both go.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none. All accessors are 2-arg.

**CB endpoints:** all legal.

- **CB 0** (`program_factory.cpp:106-116`; `2 × per_tensor_tiles` = 292 tiles of `tile_size(dtype)`, page = one tile, over `all_cores`): plain 1:1. Bind the reader fork as **PRODUCER** (`dfb::in`) and the writer as **CONSUMER**. The writer's `get_read_ptr()` peek (`:52`) is covered by its consumer binding. No flag.

## Watch for

- **CB endpoints (multi-binding):** none. There is no hidden second writer, no semaphores, and no third toucher.
- **Cross-op / shared kernels:** the reader is borrowed from `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`. A `_metal2` fork already exists at `…/reader_unary_interleaved_start_id_metal2.cpp`: bind it, don't re-fork, and don't touch the legacy file (its pointer comment is already in place).
  - Fork interface: `dfb::in` (PRODUCER), `tensor::src`, `args::num_pages`, `args::start_id`. Leave `BACKWARDS` undefined, as today.
  - Page size moves from `get_local_cb_interface(0).fifo_page_size` to `dfb.get_entry_size()`, which is the same value (one tile).
  - Other legacy binders (a **sunset list, not authorization to convert the kernel in place**): `examples/example` (multi- and single-core), `examples/example_multiple_return`, and `reduction/topk` (`topk_route_prep`).
  - The writer is op-owned and bound by no other op, so convert it in place.
- **RTA varargs:** none. Every RTA and CTA is read at a fixed index, so name them all.
- **Writer CB→DFB.** The writer includes `api/dataflow/circular_buffer.h` (`:9`) and builds `CircularBuffer cb_out0(cb_id_out0)` with `constexpr uint32_t cb_id_out0 = 0` (`:36,42`). Apply the whitelisted swap to `DataflowBuffer` from a `dfb::` token. `get_tile_size(cb_id_out0)` (`:37`) moves to the DFB's tile-size accessor (whitelist rule 7). Name the token for the writer's role (e.g. `dfb::out`). It is the same DFB the reader fork binds as `dfb::in`.
- **Writer RTAs are mutated.** `q_out_h_dim`, `q_out_tensor_tile_id` and `kv_out_tensor_tile_id` are reassigned inside the block loop (`:104-112`). Read them into non-`const` locals.
- **Verification.** `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py:99-114` covers the op, including `test_nlp_create_qkv_heads_falcon7b_with_program_cache`, which exercises the cache-hit binding refresh. The grid is the device's own compute grid, so these tests run on this workspace's 8×8 Wormhole.
- **Don't use `experimental/quasar/**` as a reference**; it holds no valid precedent for this port.

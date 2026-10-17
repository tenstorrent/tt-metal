# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer`

> Audit cleared all gates. One gate was cleared by user waiver; see below. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user waiver: readiness-sheet row is stale, see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section; hash from the `Port_Recipe` checkout)*

**TTNN gate waiver (record in the port report).** The live readiness sheet (fetched 2026-10-07) still describes this op as it was before PR #57409:

- `Concept` = `legacy device-op`
- factory `NlpCreateQkvHeadsSegformerProgramFactory`, at the now-deleted `device/nlp_create_qkv_heads_segformer_program_factory.hpp`
- `Is able to port?` = `yes (with PD step)`

The code is already on a direct `create_descriptor`, so the audit flagged the sheet as broken. On 2026-10-07 the user waived that gate ("Yes run the audit on them as well. If it's just the stale sheet, waive it"). The sheet refresh is still owed to the readiness-sheet owner.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The op ports to `ProgramSpecFactoryConcept`.

- **Current concept:** `descriptor`, in the **direct-descriptor** shape (`device/nlp_create_qkv_heads_segformer_device_operation.hpp:21-27`; body `device/nlp_create_qkv_heads_segformer_program_factory.cpp:19-155`), with no `program_factory_t`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. There is no `override_runtime_arguments`.
- **Gate-cleared, confirmed absent:** a non-clearing relaxation (cell is `none`) and `get_dynamic_runtime_args`. The op also has no custom hash and no pybound `create_descriptor` (`nlp_create_qkv_heads_segformer_nanobind.cpp:19-30` binds only the user function).

## Construct — to do

**Introduce a factory struct (forced; `ttnn_factory.md` §3).** In `device/nlp_create_qkv_heads_segformer_device_operation.hpp`:

1. Nest `struct NlpCreateQkvHeadsSegformerProgramFactory` (the pre-#57409 name) with `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);`.
2. Add `using program_factory_t = std::variant<NlpCreateQkvHeadsSegformerProgramFactory>;`.
3. Remove the device-op-level `create_descriptor` and its comment (`:21-27`).

Keep the body in `device/nlp_create_qkv_heads_segformer_program_factory.cpp`. Record this under Handoff points, and check first that no `program_factory_t` has appeared since the audit.

**ProgramSpec contents.** There is a single code path: two DM `KernelSpec`s over `all_cores` (`split_work_to_cores`, `program_factory.cpp:50-56`) and one DFB.

- **reader**: `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp`, reader config.
  - Named CTA: `q_num_tiles` (`:71-73`). The accessor append (`:74`) disappears.
  - Per-node named RTAs: `in1_tensor_addr` (constant `0`), `num_blocks`, `in0_tensor_tile_id` = `num_blocks_written * per_tensor_tiles`, `in1_tensor_tile_id` (constant `0`) (`:126-134`). RTA 0 (`in0_buffer`) becomes the input binding.
- **writer**: `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp`, writer config.
  - Named CTAs: `q_out_h_tiles`, `q_out_w_tiles`, `q_out_HtWt`, `q_out_c` (`:76-81`). The accessor append (`:82`) disappears.
  - Per-node named RTAs: `num_blocks`, `q_out_h_dim`, `q_out_tensor_tile_id` (`:136-147`). RTA 0 (`q_buffer`) becomes the q binding.
- Node order: `CoreCoord core = {i / num_cores_y, i % num_cores_y}` (`:115-116`) with group-1/group-2 block counts. Keep it.

**Tensor bindings** (per binding):

- `input`: **Case 1**. The `Buffer*` RTA (`:129`) and `TensorAccessorArgs<1>` (reader `:26`) become a `TensorParameter`, and the reader builds `TensorAccessor(tensor::<name>)`.
- `q`: **Case 1**. The `Buffer*` RTA (`:143`) and `TensorAccessorArgs<4>` (writer `:27`) become a `TensorParameter`, and the writer builds `TensorAccessor(tensor::<name>)`.
- `k`, `v` outputs: **no binding**. No kernel touches them. Leave the device-op's allocation and return of all three outputs unchanged.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none.

**CB endpoints:** CB 1 (`:103-113`; `2 × per_tensor_tiles` tiles of `tile_size(dtype)`, page = one tile) is plain 1:1. Bind the reader as **PRODUCER** and the writer as **CONSUMER**. No flag.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none. Both kernels are this op's private copies, so convert them in place. Their filenames match `nlp_create_qkv_heads/device/kernels/dataflow/…` and `nlp_create_qkv_heads_vit/…`, but those are different files; touch only the paths this factory binds. Don't use `experimental/quasar/**` as a reference.
- **RTA varargs:** none.
- **Unused reader RTAs.** `in1_tensor_addr` and `in1_tensor_tile_id` (reader `:17,20`) are read but never used, and the host passes constant `0` (`:32,130,133`). Carry them as plain named args with value `0`. They are not a tensor binding, and the port should neither delete nor "fix" them (the audit lists them as a team-only anomaly).
- **CB→DFB swap.** Both kernels include `api/dataflow/circular_buffer.h` and build `CircularBuffer cb_qv(1)` (reader `:8,34`; writer `:9,33`). Move them to `DataflowBuffer` from the `dfb::` token. `get_tile_size(cb_id_qv)` (reader `:35`, writer `:34`) moves to the DFB's tile-size accessor (whitelist rule 7).
- **Mutated RTAs.** Writer `q_out_h_dim` / `q_out_tensor_tile_id` (`:67-76`) and reader `in0_tensor_tile_id` (`:45`) are modified in place, so use non-`const` locals.
- **Verification.** `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_segformer.py`, including `test_nlp_create_qkv_heads_segformer_with_program_cache` (`:91-107`, which asserts 2 cache entries). The grid is the device's own compute grid, so the tests run on the 8×8 Wormhole here. The tests compare Q only, which matches what the op writes.

# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit`

> Audit cleared all gates. One gate was cleared by user waiver; see below. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user waiver: readiness-sheet row is stale, see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section; hash from the `Port_Recipe` checkout)*

**TTNN gate waiver (record in the port report).** The live readiness sheet (fetched 2026-10-07) still describes this op as it was before PR #57409:

- `Concept` = `legacy device-op`
- factory `NlpCreateQkvHeadsVitProgramFactory`, at the now-deleted `device/nlp_create_qkv_heads_vit_program_factory.hpp`
- `Is able to port?` = `yes (with PD step)`

The code is already on a direct `create_descriptor`, so the audit flagged the sheet as broken. On 2026-10-07 the user waived that gate ("Yes run the audit on them as well. If it's just the stale sheet, waive it"). The sheet refresh is still owed to the readiness-sheet owner.

**Dead `transpose_k_heads` branch: carry it faithfully.** On 2026-10-07 the user decided ("Keep the behaviour unchanged, stick with the default") that the compile-time-dead branch is carried over as-is: zero functional change, with the conditional compute/DFB specs bound to `transpose_wh_metal2.cpp`. Record the decision in the port report, and record the path as untested.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The op ports to `ProgramSpecFactoryConcept`.

- **Current concept:** `descriptor`, in the **direct-descriptor** shape (`device/nlp_create_qkv_heads_vit_device_operation.hpp:21-27`; body `device/nlp_create_qkv_heads_vit_program_factory.cpp:19-236`), with no `program_factory_t`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. There is no `override_runtime_arguments`.
- **Gate-cleared, confirmed absent:** a non-clearing relaxation (cell is `none`) and `get_dynamic_runtime_args`. The op also has no custom hash and no pybound `create_descriptor` (`nlp_create_qkv_heads_vit_nanobind.cpp:19-30` binds only the user function).

## Construct — to do

**Introduce a factory struct (forced; `ttnn_factory.md` §3).** In `device/nlp_create_qkv_heads_vit_device_operation.hpp`:

1. Nest `struct NlpCreateQkvHeadsVitProgramFactory` (the pre-#57409 name) with `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);`.
2. Add `using program_factory_t = std::variant<NlpCreateQkvHeadsVitProgramFactory>;`.
3. Remove the device-op-level `create_descriptor` and its comment (`:21-27`).

Keep the body in `device/nlp_create_qkv_heads_vit_program_factory.cpp`. Record this under Handoff points, and check first that no `program_factory_t` has appeared since the audit.

**ProgramSpec contents: the live path** (`transpose_k_heads == false`). Two DM `KernelSpec`s over `all_cores` (`split_work_to_cores`, `program_factory.cpp:53-58`) and one DFB.

- **reader**: `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp`, reader config, no defines.
  - Named CTAs: `q_num_tiles`, `kv_num_tiles` (`:80-83`). Both accessor appends (`:84` real, `:85` the empty in1 placeholder) disappear.
  - Per-node named RTAs: `in1_tensor_addr` (constant `0`), `num_blocks`, `in0_tensor_tile_id` = `num_blocks_written * per_tensor_tiles`, `in1_tensor_tile_id` (constant `0`) (`:198-206`). RTA 0 (`in0_buffer`) becomes the input binding.
- **writer**: `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp`, writer config, no defines.
  - Named CTAs: `q_out_h_tiles`, `q_out_w_tiles`, `q_out_HtWt`, `q_out_c`, `kv_out_c` (`:86-92`). The three accessor appends (`:93-95`) disappear.
  - Per-node named RTAs: `num_blocks`, `q_out_h_dim`, `q_out_tensor_tile_id`, `k_out_tensor_tile_id`, `v_out_tensor_tile_id` (`:208-228`). RTAs 0-2 become the q/k/v bindings.
- Node order: `CoreCoord core = {i / num_cores_y, i % num_cores_y}` (`:187-188`) with group-1/group-2 block counts. Keep it.

**ProgramSpec contents: the dead path** (carry faithfully, per the user's 2026-10-07 decision). Keep `const bool transpose_k_heads = false;` (`:98`). Under `if (transpose_k_heads)`:

- **Compute**: two compute `KernelSpec`s, one over `core_group_1` and one over `core_group_2` (the latter only if non-empty), bound to `ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp`. Named CTA `NHtWt` = `num_blocks_per_core_group_N * kv_num_tiles` (`:101-118`).
- **DFBs**: CB 0 (reader → compute) and CB 16 (compute → writer) as conditional DFBs, each `2 × per_tensor_tiles` tiles (`:161-185`). Bind them to the fork's `dfb::in` / `dfb::out`.
- **Defines**: add `TRANSPOSE_K_HEADS=1` to the reader and writer defines (`:119-120`).
- **Tile ids**: keep the alternate `k_out_tensor_tile_id` formula (`:213-215`).

**Tensor bindings** (per binding):

- `input`: **Case 1**. The `Buffer*` RTA (`:201`) and `TensorAccessorArgs<2>` (reader `:26`) become a `TensorParameter`; the reader builds `TensorAccessor(tensor::<name>)`.
- `q`, `k`, `v`: **Case 1**. The `Buffer*` RTAs (`:220-222`) and the writer's `TensorAccessorArgs<5>` chain (writer `:32-34`) become three `TensorParameter`s.
- No in1 binding. The reader's `in1_args` / `s1` exist only under the never-defined `READ_FROM_INPUT_TENSOR_KV` (reader `:27-29,41-43`); leave those `#ifdef` blocks verbatim.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none.

**CB endpoints:**

- **CB 1** (`:145-155`; 144 tiles of `tile_size(dtype)`, page = one tile): plain 1:1 on the live path. Bind the reader as **PRODUCER** and the writer as **CONSUMER**. No flag.
- **CBs 0 and 16** (dead path only): 1:1 each. Reader → compute on CB 0; compute → writer on CB 16. Conditional on `transpose_k_heads`, exactly as the legacy host allocation is.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Two wrappers, one DFB.** In both kernels, `cb_qv` and `cb_k` both resolve to index 1 when `TRANSPOSE_K_HEADS` is undefined (reader `:31-36`, writer `:36-41`). Build both `DataflowBuffer`s from the **same** `dfb::` token there (e.g. `dfb::qv`). Under `#ifdef TRANSPOSE_K_HEADS`, `cb_k` takes the K-specific token (reader → `dfb::k_in`-style, writer → `dfb::k_out`-style). Keep the `#ifdef` structure verbatim; only the token names change.
- **Cross-op / shared kernels:** the reader and writer are op-private copies (their filenames match `nlp_create_qkv_heads/…` and `nlp_create_qkv_heads_segformer/…`, but those are different files), so convert them in place. The only shared kernel is `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`, on the dead path. A `_metal2` fork already exists at `ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp`: bind it, don't re-fork, and don't touch the legacy file. Fork interface: `dfb::in`, `dfb::out`, named CTA `NHtWt`. Its other binders, as a **sunset list (not authorization to convert anything in place)**: `data_movement/permute` (tiled), `data_movement/transpose` (WH), and `experimental/transformer/nlp_create_qkv_heads`. Don't use `experimental/quasar/**` as a reference.
- **RTA varargs:** none.
- **Dummy in1 RTAs.** `in1_tensor_addr` / `in1_tensor_tile_id` (reader `:17,20`; host `:32,202,205`) are read unconditionally but used only under `READ_FROM_INPUT_TENSOR_KV`. Carry them as plain named args with value `0`; they are not a tensor binding.
- **CB→DFB swap** in both kernels (`circular_buffer.h` at reader `:8` and writer `:9`). `get_tile_size(cb_id_qv/k)` (reader `:47-48`, writer `:48-49`) moves to the DFB's tile-size accessor (whitelist rule 7).
- **Mutated RTAs.** The writer's four tile-id/h-dim args (`:140-161`) and the reader's `in0_tensor_tile_id` / `in1_tensor_tile_id` are updated in place, so use non-`const` locals.
- **Verification.** `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_vit.py` (`test_nlp_create_qkv_heads_vit_test` at `:100`; program-cache test at `:104`). The grid is the device's own compute grid, so the tests run on the 8×8 Wormhole here. No test can reach the dead transpose path, so record it as untested in the port report.

# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/eltwise/unary`

> The audit cleared all gates. This brief is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.
>
> **One item is decision-required before you start:** the `distribution_key` hash swap below. The relaxation analysis requires it, and `ttnn_factory.md` forbids hash edits. Proceed with it only if the user has sanctioned it (audit *Questions* 1). Otherwise stop at that step and report.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (sheet cells supplied by the user, 2026-09-24) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `edf75ffab9c 2026-09-23 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section. Tree-identical to `origin/akertesz/op-porting-recipe` `4bd4bf42bfe`. Relaxation doc: `analyses/relaxations/eltwise_unary.md`, untracked, blob `653172edfca`.)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor` (`device/unary_device_operation.hpp:44`)
- **Op-owned tensors:** none
- **Target concept:** **`CustomProgramSpecFactoryConcept`**. `Override runtime args method?` = yes: translate `ProgramFactory::override_runtime_arguments` (`device/unary_program_factory.cpp:570-666`) into one returning `ProgramRunArgs`.
- **Custom hash present:** `compute_program_hash` (`device/unary_device_operation.cpp:284-350`) + `operation_attributes_t::to_hash()` (`:113-123`). Leave it as is, except for the decision-required swap below.
- **Gate-cleared, confirmed absent:** a `TensorParameter relaxation` that is neither `none` nor an analysis pointer (the cell is `dynamic (see analysis)`) · `get_dynamic_runtime_args`.

## Construct — to do

**Two data paths, one factory.** The native-L1-sharded path (`has_sharding`) applies only when *both* tensors are 2D-sharded in L1, evenly, on one grid (`common/unary_utils.cpp:36-59`). Every other case, including mixed / ND / DRAM / uneven sharding, takes the accessor path. `SRC_SHARDED` / `DST_SHARDED` / `RM_INTERLEAVED` are compile-time defines, and the key pins them, so they cannot flip on a cache hit.

**Tensor bindings** (per binding):

- `input` (kernel vocabulary `src`):
  - **Case 1** on the accessor path. Express it as a `TensorParameter` / `TensorBinding`; `reader_unary` builds `TensorAccessor(tensor::src)`. RTA0 `src_addr` and the `TensorAccessorArgs(..., RuntimeTensorShape)` CTA/CRTA plumbing (`unary_program_factory.cpp:460-463`) both disappear.
  - **Clean** on the sharded path. `c_0` becomes a DFB `borrowed_from` the input `TensorParameter` (legacy `.buffer = src_buffer`, `:428`).
- `output` (kernel vocabulary `dst`):
  - **Case 1** on the accessor path (`writer_unary` → `TensorAccessor(tensor::dst)`).
  - **Clean** on the sharded path. `c_2` is `borrowed_from` the output `TensorParameter` (`:452`).

**TensorParameter relaxation:** `dynamic (see analysis)` → `analyses/relaxations/eltwise_unary.md` (all 6 validity checks pass on this tree). On **both** `input` and `output`, unconditionally:

```cpp
.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true},
```

Do **not** set `match_page_size` or `match_padded_shape_only`. No shipped non-experimental factory declares relaxations yet, so treat a validation throw as a possible framework gap and report it. Don't loosen the declaration to make it pass.

**⚠ Decision-required, same edit as the declaration (doc §2):** in `compute_program_hash`'s `distribution_key` (`unary_device_operation.cpp:323-336`), swap the Buffer source (`buffer->buffer_distribution_spec()`) for `spec.compute_buffer_sharding_args()` on both slots. The code's `TODO(port)` (`:320-322`) names this swap. Metal 2.0 validation reads the spec's resolution (`tensor_spec_relaxations.cpp:105-109`). Replace the source; don't hash both, which costs +16%. **This edits the device-op class, which `ttnn_factory.md` forbids.** Do it only on the user's explicit sanction, and record it prominently as a device-op edit in the port report. Without that sanction, stop here and report.

**TensorAccessor 3rd arg:** none.

**CB endpoints:**
- Self-loop `c_1` (tmp0): `logit_kernel.cpp` is its sole producer and consumer. Keep its `DataflowBufferSpec` conditional on `ops_chain[0] == LOGIT`, exactly as the legacy host code is (`unary_program_factory.cpp:431-441`).
- `c_0` (reader → compute) and `c_2` (compute → writer): plain 1P+1C in both paths. They are `borrowed_from` the tensors on the sharded path only.
- No dead CBs, no multi-binding.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:**
  - `device/kernels/compute/eltwise_sfpu.cpp` is **lent**. There is no `_metal2` fork beside it, so this port creates `compute/eltwise_sfpu_metal2.cpp` beside the original (rung 2) and adds the pointer comment to the original. The `eltwise/unary/CMakeLists.txt:16` glob installs the fork, so no build edit is needed.
  - Other binders of the legacy copy. This is a **sunset list, not authorization to convert the kernel in place**:
    - `examples/example` (`SingleCore`, `MultiCore`)
    - `examples/example_multiple_return` (`SingleCore`)
    - `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:246`
    - `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:1436`
  - The other 10 bound kernels have no other binder, so convert them in place: `reader_unary.cpp`, `writer_unary.cpp`, and compute `eltwise_identity_kernel`, `where_tss_kernel`, `mac_tss_kernel`, `logit_kernel`, `hardswish_kernel`, `logsigmoid_kernel`, `lgamma_fast_kernel`, `lgamma_kernel`.
  - The four `_metal2` files already in `dataflow/` fork kernels this op doesn't bind. Leave them alone.
  - Name the fork's bindings after the kernel (`dfb::in`, `dfb::out`-style), not after unary's locals.
- **RTA varargs:** none. Name every RTA:
  - reader/writer: `src_addr`/`dst_addr` go away into the binding, plus `num_pages`, `start_id`, and, on the RM accessor path, `chunks_per_row`, `chunk_size`, `last_chunk_size`, `rows_per_tile`, `total_rows`
  - compute: `num_tiles`, `packed_scalar1`, `packed_scalar2`
- **Also needed by the porter:**
  - **Runtime-selected compute source.** `get_compute_kernel_path` (`common/unary_op_utils.cpp:1194-1212`) picks one of 9 compute sources, so all 9 need converting. Seven go through `compute_kernel_lib` with `uint32_t` CB ids: pass `dfb::name` straight in, with no `.id` extraction and no temporary wrappers. The kernels are already on `DataflowBuffer` / kernel_lib, so this is a binding-layer change, not an idiom rewrite.
  - **Translating the override:** keep only the per-core RTA re-writes, done through the shared `enumerate_core_rt_args` (`unary_program_factory.cpp:127-333`):
    - work split and start ids
    - zero-fill of noop cores (a core can flip between noop and active across hits)
    - the RM chunk tail
    - the compute scalars

    The accessor-CRTA rebuild (`:638-653`) and the CB-address `apply_descriptor_runtime_args` (`:655-665`) are what the tensor bindings and `borrowed_from` now refresh.
  - **Compute `opt_level` = `O3`, set explicitly.** Legacy compute resolves to `O3` (`tt_metal/impl/program/program.cpp:485`), while a `KernelSpec` defaults to `O2` (`kernel_spec.hpp:122`). DM kernels stay `O2`.
  - **Unread trailing compute CTA** (`cb_data_format`, `unary_program_factory.cpp:506`). No kernel reads it. Carry it as a named CTA (don't drop it) and note it in the report.
  - **`unpack_to_dest_mode`** is set for both `c_0` and `c_1` under `preserve_fp32_precision` (`:401-404`). Map it to per-DFB unpack modes, and add the tmp0 entry only where the LOGIT-conditional DFB exists.
  - **Don't use `anasuya/metal2_port_unary` (`142243af82e`) as a template.** That earlier, unmerged attempt predates #56858 and the revised relaxation doc.

# QUASAR_UPLIFT_REPORT — ttnn.slice (test_slice.py + test_tensor_getitem.py)

**Status: RED — Not Metal 2.0 on Gen1 yet.**
Every `slice` program factory is still `create_descriptor` → `tt::tt_metal::ProgramDescriptor`;
none implements `create_program_artifacts` → `ProgramArtifacts`, and no kernel uses the
`dfb::` / `args::` / `tensor::` / `scratch::` named-binding model. Per §1 of
`quasar_porting.md` and its RED-stop list ("factory still `create_descriptor`/`ProgramDescriptor`
→ do the Metal 2.0 port first"), the Quasar uplift cannot start. **No source changes were made**
— a RED result is a success of the audit, not a half-finished port.

- Recipe executed: `docs/source/ttnn/ttnn/ai/quasar_porting.md` +
  `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/quasar_audit.md`
  (gate definition cross-checked against `ai/audit/metal2_audit.md`)
- Tests: `models/experimental/llama32_1b_quasar/tests/graph_ops/test_slice.py`,
  `models/experimental/llama32_1b_quasar/tests/graph_ops/test_tensor_getitem.py`
- Op source: `ttnn/cpp/ttnn/operations/data_movement/slice/`

## Op mapping (what the tests actually exercise)

- **`test_slice.py`** → `ttnn.slice` → composite in `slice.cpp` (`ttnn::slice`).
- **`test_tensor_getitem.py`** → `ttnn.Tensor.__getitem__`
  (`ttnn/ttnn/operations/core.py:__getitem__`) which normalizes the python slices and, for
  device tensors, calls **`ttnn.slice(input, starts, ends, steps)`** — i.e. it decomposes
  onto the same op. It would additionally call `ttnn.squeeze` only for integer indices;
  all 3 captured signatures use pure slice objects, so **no squeeze is reached**.
- All 4 captured cases are TILE layout, DRAM-interleaved, `step == 1`, rank 4, with
  tile-aligned begins in the last two dims (`slice.cpp` `check_handled_tile_alignment`),
  so `rm_only == false` → `ttnn::prim::slice` with `use_tensor_args == false` →
  **`SliceTileProgramFactory`** (`device/slice_program_factory_tile.cpp`), per
  `SliceDeviceOperation::select_program_factory` (`device/slice_device_operation.cpp:268`).
- Ops other than slice on the exercised path, owned by **other audits** (not expanded here):
  - `ttnn::experimental::view` (`slice.cpp:385`) — host-side metadata re-wrap after the prim.
  - `ttnn::fill_implicit_tile_padding` — only when a sliced dim needs implicit-pad fill;
    for all 4 captured cases `dim_needs_fill` is false, so it is not reached.
  - `ttnn.squeeze`, `ttnn::copy` / `to_memory_config` / `to_layout` adjustment paths — not
    reached by these signatures.

## Metal 2.0 gate (§1 of quasar_porting.md) — FAILED

Host factories — all five are legacy descriptor form (`create_descriptor` returning
`ProgramDescriptor`, `CBDescriptor` + buffer-address runtime args +
`TensorAccessorArgs(*buffer).append_to(compile_time_args)`):

| Factory | Evidence |
|---|---|
| `SliceTileProgramFactory` (the one these tests hit) | `device/slice_program_factory_tile.cpp:23` |
| `SliceTileTensorArgsProgramFactory` | `device/slice_program_factory_tile_tensor_args.cpp:23` |
| `SliceRmProgramFactory` | `device/slice_program_factory_rm.cpp:311` |
| `SliceRmShardedProgramFactory` | `device/slice_program_factory_rm_sharded.cpp:212` |
| `SliceRmStrideProgramFactory` | `device/slice_program_factory_rm_stride.cpp:22` |

`grep -r create_program_artifacts\|ProgramArtifacts` over the op directory: **zero hits**.

Kernel side (all 12 kernels under `device/kernels/dataflow/`): the op is **Device 2.0
migrated but not Metal 2.0**. Kernels include `api/dataflow/dataflow_api.h` / `noc.h` /
`dataflow_buffer.h`, use `DataflowBuffer` objects, `Noc::async_read/write`, and
`get_entry_size()` — but bind their inputs the ProgramDescriptor way: positional
`get_arg_val<uint32_t>` / `get_common_arg_val`, `get_named_compile_time_arg_val("dfb_id_*")`,
and address-RTA + `TensorAccessorArgs<N>()` (e.g.
`reader_unary_unpad_dims_interleaved_start_id.cpp:13-27`). There are no `dfb::` / `args::` /
`tensor::` tokens anywhere in the op. This is exactly the §1 "not ported yet" state: the
Quasar uplift assumes named bindings from a `KernelSpec`, which do not exist here.

Because the gate fails, the Quasar-uplift audit proper (quasar_audit.md checks + §7–§12)
was not applied as an uplift; the observations below were collected in passing and are left
for the future porter.

## Files changed

**None.** Zero source diff. (This report file is the only artifact; delete before merge.)

## §7–§8 gotchas applied / considered

- **Applied:** none — RED gate, and §7–§8 fixes are reactive (no device run this session).
- **Considered / recorded for the future port** (informational, from the passing scan):
  - **Semaphores** (quasar_audit.md check 2): the op creates **no semaphores at all** —
    clean when the port happens.
  - **DM self-loop / CB redesign** (check 1): each factory uses a single CB as a plain
    reader→writer FIFO (two distinct kernels, e.g. `reader_unary_unpad_dims_...` produces,
    `writer_unary_interleaved_start_id.cpp` consumes) — no self-loop, no sync-free scratch.
  - **`fifo_page_size` staleness** (§5, §8.3): kernels already read sizes via
    `DataflowBuffer::get_entry_size()`; `grep get_local_cb_interface\|fifo_page_size` → zero hits.
  - **Manual L2 flush leftovers** (§8.3, a00dd45/#52769): `grep invalidate/flush_l2_cache` →
    zero hits in this op — the historical slice manual-flush fixes are already gone; nothing
    to clean up.
  - **Int32/uint16 quirk** (§7): the tile factories forward dtype without a format branch;
    the RM paths use `element_size()` arithmetic, not device uint16/uint32 formats.
  - NoC/mcast (§11): no multicast in any slice kernel.

## Deferred / follow-up items

1. **The Metal 2.0 port itself** — the actual blocker. Run `ai/audit/metal2_audit.md`
   (→ `METAL2_PREPORT_AUDIT.md` / `METAL2_PORT_BRIEF.md`) and then `ai/port/metal2_port.md`
   on this op. Porter heads-up from this scan:
   - The RM factories pair `create_descriptor` with a **`get_dynamic_runtime_args` hook**
     re-emitted on every cache hit (`device/slice_program_factory_rm.hpp:40`,
     `slice_program_factory_rm.cpp`, `slice_program_factory_rm_sharded.cpp`) — the
     deprecated-hook situation the metal2 audit flags.
   - The device op has a **custom `compute_program_hash`**
     (`device/slice_device_operation.cpp:302`) — the port must leave it intact.
   - `slice_program_factory_tile.cpp` uses `experimental/program_descriptor_patching.hpp` —
     descriptor-patching has no direct `ProgramArtifacts` analogue; needs the port recipe's
     treatment.
2. Only after that port lands: re-run the Quasar-uplift audit (this recipe) on the ported op.
   Given the observations above (no semaphores, no self-loops, `get_entry_size()` already in
   use, no L2-flush leftovers), the eventual uplift is likely a no-op GREEN.
3. `ttnn::experimental::view` and `ttnn::fill_implicit_tile_padding` (and `ttnn.squeeze` for
   integer-index getitem) belong to other ops' audits.

## WH/BH parity claim (structural — no device run this session)

The working-tree diff for this task is **empty** (no source file touched), so WH/BH
behavior is unchanged by construction. Confirmation commands (user-run, per recipe §9):

```bash
# BH / WH parity (same command on each machine):
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_slice.py
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tensor_getitem.py

# Broader slice regression net on WH/BH:
pytest tests/ttnn/unit_tests/operations/data_movement/test_slice.py

# Quasar (emulator, per the craqsim runbook env) — expected to exercise the
# un-ported ProgramDescriptor path; run after the Metal 2.0 port lands:
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_slice.py
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tensor_getitem.py
```

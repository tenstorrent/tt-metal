# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half`

> Audit cleared all gates. One gate was cleared by user waiver; see below. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user waiver: readiness-sheet row is stale, see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

**TTNN gate waiver (record in the port report).** The live readiness sheet (fetched 2026-10-07) still describes this op as it was before PR #57409 (`f5093e705ae`, 2026-09-25):
- it shows `Concept` = `legacy device-op`
- it has a `RotateHalfProgramFactory` row whose `Factory definition path` (`device/rotate_half_program_factory.hpp`) no longer exists

That is a primary-column conflict plus a phantom factory row, which the audit recipe classes as "spreadsheet-broken". In the code, the op is `descriptor`: `RotateHalfDeviceOperation::create_descriptor` at `device/rotate_half_program_factory.cpp:16`. Every other cross-checked column matches the code, and the sheet's own verdict is `yes (with PD step)`, with the PD step already landed. On 2026-10-07 the user waived this gate ("waive"). The sheet refresh is owed to the readiness-sheet owner and does not block the port.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`. There is no factory struct: `create_descriptor` is a static method on `RotateHalfDeviceOperation` (`device/rotate_half_device_operation.hpp:24`). The port introduces the spec-factory shape the concept expects.
- **Op-owned tensors:** none
- **Target concept:** `ProgramSpecFactoryConcept`
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args` (deprecated hook). A custom hash, an `override_runtime_arguments`, and a pybound `create_descriptor` are **not** in this list: none of them gate, and any may be present on a cleared op. On this op, none of the three is present:
  - default hash
  - no `override_runtime_arguments`
  - `rotate_half_nanobind.cpp:18` binds only the function

## Construct — to do

The program is single-core (`(0,0)`), with three kernels: reader, writer, and the borrowed bcast compute kernel.

**Tensor bindings** (per binding):

- `src` (input) — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`.
  - Today it arrives as the `Buffer*` RTA at `device/rotate_half_program_factory.cpp:109` (reader RTA 0), plus `TensorAccessorArgs` appended to the reader CTAs at `:97`.
  - The reader builds `TensorAccessor(src_args, src_addr)` at `reader_rotate_half_interleaved_start_id.cpp:30`. That becomes `TensorAccessor(tensor::src)`.
- `dst` (output) — **Case 1** → `TensorParameter` / `TensorBinding`.
  - Today it arrives as the `Buffer*` RTA at `:119` (writer RTA 0), plus the CTAs at `:99`.
  - The writer builds `TensorAccessor(dst_args, dst_addr)` at `writer_rotate_half_interleaved_start_id.cpp:28`. That becomes `TensorAccessor(tensor::dst)`.

**TensorParameter relaxation:** `none`

**TensorAccessor 3rd arg:** none — no accessor passes a 3rd arg

**CB endpoints:** all legal, one producer and one consumer each:

| CB | Producer | Consumer |
|---|---|---|
| `c_0` (multiplied half, `:40`) | reader | compute `dfb::in0` |
| `c_1` (scalar, `:53`) | reader, one tile | compute `dfb::in1`, waited once and never popped |
| `c_2` (un-multiplied half, `:66` / `:88`) | reader | writer |
| `c_16` (`:77`) | compute `dfb::out` | writer |

`c_2` carries two host names (`src_no_mul_cb_index`, `output_no_mul_cb_index`). Declare it as **one** `DataflowBufferSpec`.

## Watch for

- **CB endpoints (multi-binding):** none
- **Cross-op / shared kernels:** compute binds `data_movement/bcast/device/kernels/compute/bcast_hw.cpp`, which is borrowed. **A `_metal2` fork already exists at `data_movement/bcast/device/kernels/compute/bcast_hw_metal2.cpp`. Bind it (rung 1); don't re-fork and don't edit it.** Adopt its interface:
  - DFBs: `dfb::in0` CONSUMER (← `c_0`), `dfb::in1` CONSUMER (← `c_1`), `dfb::out` PRODUCER (→ `c_16`)
  - named RTAs: `B`, `Ht`, `Wt` (today `1`, `1`, `num_tiles / 2`; `:130-136`)
  - **Defines differ from what rotate_half passes today.** rotate_half passes `CHAIN_BCAST_OP`, `CHAIN_BCAST_DIM` and `BCAST_SCALAR` (`:125-128`). The fork reads `BCAST_OP`, `BCAST_LLKOP`, `BCAST_DIM` and `BCAST_SCALAR`. Supply `bcast_op_utils::get_defines(BcastOpDim::HW, BcastOpMath::MUL)` (`data_movement/bcast/bcast_types.cpp:9`) plus `BCAST_SCALAR=1`, mirroring `bcast_multi_core_hw_program_factory.cpp:159-163`. `get_defines` returns `mul_tiles_bcast` / `ELWMUL` / `BroadcastType::SCALAR`.
  - Set `opt_level = KernelBuildOptLevel::O3` to keep the legacy compute default, as the bcast port does (`bcast_multi_core_hw_program_factory.cpp:223-224`).
  - **The fork lags its original.** `8cf12808550` (#55584) moved `bcast_hw.cpp` to `kernel_lib` `eltwise_chain` after the fork was cut. So binding the fork swaps rotate_half's compute body for the fork's hand-rolled `bcast_init` / `mul_tiles_bcast` loop. The math is the same (scalar-broadcast multiply, rhs waited once and never popped), but confirm numerics with `tests/tt_eager/python_api_testing/unit_testing/misc/test_rotate_half.py` (both shapes are single-core). Note the drift in the port report; don't sync the fork yourself.
  - Other binding ops — **sunset list, not authorization to convert the kernel in place**: rotate_half is the **last** binder of the legacy `bcast_hw.cpp` (the bcast factory already binds the fork). Record in the port report that the legacy copy has no consumers left after this port. Deleting it is not your job.
- **RTA varargs:** none. Name every arg:
  - reader: `src_addr` → binding, `num_rows`, `half_row_size`, `start_id`
  - writer: `dst_addr` → binding, `num_rows`, `half_row_width`, `start_id`
  - compute: `B`, `Ht`, `Wt`
- **Anything else the porter needs:**
  - The reader writes the `-1.0` bf16 scalar through a raw `uint16_t*` on `cb_in_scalar.get_write_ptr()` (`reader_…:37-42`). That is a producer-side peek on its own DFB; keep it as-is.
  - The scalar reaches the reader as CTA 3 (`:90,96`, read at `reader_…:23` as `constexpr uint16_t`). Keep it a named CTA.
  - CB ids are CTAs 0–2 (reader) and 0–1 (writer); they become `dfb::` tokens.
  - `get_tile_size(cb_id)` at `reader_…:27` / `writer_…:25` moves onto the DFB object (kernel-side whitelist rule 7).
  - Already Device 2.0 (`Noc`, `CircularBuffer`, `CoreLocalMem`), so the kernel change is a binding-layer swap, not an idiom rewrite.
  - Known latent issue, **not yours to fix**: CB `c_1` is declared with the input's data format but a `Float16_b` page size (`:55-63`). Preserve it verbatim; it is recorded for the ops team in the audit's Misc anomalies.
  - Hardware: the tests use small single-core shapes, so they should run on this 8×8 Wormhole.

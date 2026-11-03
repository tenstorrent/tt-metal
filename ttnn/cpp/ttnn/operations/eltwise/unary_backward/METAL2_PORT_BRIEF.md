# Metal 2.0 Port Brief — `eltwise/unary_backward`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `f126189dbb6 2026-11-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

**Scope:** one `DeviceOperation` and one `ProgramFactory`, `UnaryBackwardDeviceOperation` → `UnaryBackwardProgramFactory`. It serves `SIGMOID_BW`, `TANH_BW`, `GELU_BW` and `GELU_TANH_BW` (`device/unary_backward_op_types.hpp:17-22`). The op type only picks the compute kernel (`get_kernel_spec`, `device/unary_backward_op_utils.cpp:13-64`). The audit ran on `main` `a1ce48e3fe8` code, with no conditions.

## Before you start

1. **Your base contains `be782c33d2e` (#63334).** Check with `git merge-base --is-ancestor be782c33d2e HEAD`. That commit makes `compute_program_hash` key each bound tensor's whole `TensorSpec` (`device/unary_backward_device_operation.cpp:247-257`). It also moved the row-major kernels to Device 2.0. If either is missing, **stop and report**.
2. **Re-read `get_kernel_spec` and `UnaryBackwardOpType`.** On 2026-11-03, 43 open PRs added op types to this device op: 41 "Fuse BF16 `<op>`_bw" PRs plus #61521 and #61751. Each adds one compute kernel and one kernel-spec row. Any that has landed is a compute kernel you must port too, the same way as the five below. Whether to wait for them is the user's decision; ask if it is still open.
3. **Readiness sheet.** The `eltwise/unary_backward/gelu_bw` and `.../tanh_bw` rows are phantom rows for deleted device ops. The user replaced them with one `UnaryBackwardDeviceOperation` row (`Is able to port?` = `yes`). If the live sheet shows a different value on that row, read it first.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`. `create_descriptor` is declared at `device/unary_backward_program_factory.hpp:19-20` and defined at `device/unary_backward_program_factory.cpp:26-481`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. Replace `create_descriptor` with `create_program_artifacts` inside the existing struct. There is no `override_runtime_arguments` to translate.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer (the cell is `none (see analysis)`; the doc's verdict `none` is confirmed) · `get_dynamic_runtime_args` (deprecated hook).
- **Custom hash: present; leave it alone** (`device/unary_backward_device_operation.cpp:238-260`; `ttnn_factory.md`, "The cache key: leave the custom hash alone"). It already pins every slot's whole `TensorSpec`, which `.relaxations = {}` requires.
- **No device-operation-class edit is forced.** There is no pybound `create_descriptor` (0 hits in `unary_backward_nanobind.cpp`), no hook-only parameter, and `program_factory_t` already exists (`device/unary_backward_device_operation.hpp:28`).

## Construct — to do

**Tensor bindings.** Declare three `TensorParameter`s from each tensor's `tensor_spec()`: `grad_output`, `input`, and the output (computed or preallocated).

- `grad_output` — **Case 1** (via `TensorAccessor`) in the addressed and RM-blocked modes. Express it as a `TensorParameter` / `TensorBinding`; the kernel uses `TensorAccessor(tensor::name)`, and the `Buffer*` RTA plus its `TensorAccessorArgs` go away.
  - Reader fork: `tensor::src0` (legacy reader RTA 0, `unary_backward_program_factory.cpp:468`).
  - RM reader: legacy RTA 7 `grad_addr` (`reader_unary_backward_row_major.cpp:33`; accessor `:55`).
  - **Clean** in the aliased mode: c_0 is `borrowed_from` it, and the reader gets no tensor binding.
- `input` — **Case 1**. Reader fork: `tensor::src1` (legacy RTA 1, `:468`). RM reader: legacy RTA 12 `input_addr` (`:40`; accessor `:56`). **Clean** when aliased: c_1 is `borrowed_from` it.
- `output` — **Case 1**. Writer fork: `tensor::dst` (legacy writer RTA 0, `:470`). RM writer: legacy RTA 7 `dst_addr` (`writer_unary_backward_row_major.cpp:26`; accessor `:38`). **Clean** when aliased: c_2 is `borrowed_from` it.

There is no Case 2. The RM kernels use each address only to build its accessor. Every access goes through a `TensorAccessor` endpoint with `{.page_id, .offset_bytes}`. The RM kernels are this op's own. You choose their tensor names; the kernels' locals `grad`, `input` and `dst` are the natural ones.

**TensorParameter relaxation:** `none (see analysis)` → `analyses/relaxations/eltwise_unary_backward.md` §2 (blob `ce949868d1f`; all six validity checks pass). Use the same declaration on all three `TensorParameter`s, in every mode:

```cpp
// Exact match. compute_program_hash keys on each bound tensor's whole TensorSpec.
// A cache hit therefore always sees the same specs as the miss that built the program.
.relaxations = {},
```

Do not set `dynamic_tensor_shape`. The RM-blocked path bakes padded-shape geometry into per-core args that a base-concept cache hit never refreshes. Do not set `match_padded_shape_only` either. Do not copy a declaration from `eltwise/unary`.

**TensorAccessor 3rd arg:** none.

**CB endpoints:**

| CB | Exists | Addressed (TILE) | Aliased | RM blocked |
|---|---|---|---|---|
| c_0 (`grad_output`) | always | PRODUCER reader fork (`dfb::in0`), CONSUMER compute | the same, `borrowed_from` `grad_output`, one shard deep | PRODUCER RM reader, CONSUMER compute |
| c_1 (`input`) | always | PRODUCER reader fork (`dfb::in1`), CONSUMER compute | the same, `borrowed_from` `input` | PRODUCER RM reader, CONSUMER compute |
| c_2 (output) | always | PRODUCER compute, CONSUMER writer fork (`dfb::out`) | the same, `borrowed_from` the output | PRODUCER compute, CONSUMER RM writer |
| c_3 (reader scratch) | `rm_blocked` only | — | — | **self-loop**: RM reader as PRODUCER and CONSUMER |
| c_4 (writer scratch) | `rm_blocked` only | — | — | **self-loop**: RM writer as PRODUCER and CONSUMER |

- c_0, c_1 and c_2 are a legal 1P+1C in every mode.
- c_3 and c_4 follow `port_patterns.md`, "Sync-free and single-ended CBs → self-loop DFB". Each is touched by one kernel, through `get_write_ptr()` only (`row_major_pages.hpp:140-143`). Bind both roles under one accessor name. The Gen1 validator accepts a data-movement self-loop.
- Create c_3 and c_4 only when `rm_blocked`, as the legacy factory does (`unary_backward_program_factory.cpp:272-289`). Keep the legacy sizes. Their data format (the output's) is only a label: the kernels use them as raw bytes.
- There is no multi-binding and no dead CB.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels: bind the existing forks; do not re-fork** (`port_patterns.md`, "Caution: Porting a shared kernel", rung 1). The fit check passes: the factory already supplies every named arg, binding and define that both forks read.
  - **Reader:** `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id_metal2.cpp`.
    - Bindings: `dfb::in0` (c_0), `dfb::in1` (c_1), `tensor::src0` (`grad_output`), `tensor::src1` (`input`).
    - Named args: `num_tiles` ← `units`, `start_id` ← `units_done`, `block_height` ← 0, `block_width` ← 0, `num_cores_y` ← `num_cores_y`. Named CTA: `block_or_width_sharded` ← 0 (legacy `:324`, `:468`).
    - It gates on `IN0_SHARDED` / `IN1_SHARDED`.
    - It has no current binder: its only one, the experimental `gelu_backward` op, was deleted by #54799. That still does not make it yours to edit.
  - **Writer:** `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`.
    - Bindings: `dfb::out` (c_2), `tensor::dst`. Named args: `num_pages` ← `units`, `start_id` ← `units_done`.
    - It gates on `OUT_SHARDED`. It also has optional `ENTRIES_PER_PAGE` / `PAGE_BYTES` defines. Do not set them: the default, one page per DFB entry, is what the legacy writer does.
    - 26 factory files already bind it.
  - **Sunset lists** — not authorization to convert either original in place.
    - Legacy reader: this op and `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:437`.
    - Legacy writer: this op; `data_movement/reshape_on_device/device/reshape_tile_program_factory.cpp:108`; `examples/example/device/single_core_program_factory.cpp:83` and `multi_core_program_factory.cpp:81`; `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:446`, `:672`, `:811`, `:894`, `:1013`; six Python generic-op tests (`kernel_lib/chain_test_lib.py:33`, `operations/debug/test_generic_op.py:98`, and four under `operations/eltwise/`: `test_clamped_silu_glu_sfpu.py:121`, `test_exp_approx_disable_sfploadmacro.py:86`, `test_moe_gate_sqrt_softplus_sfpu.py:126`, `test_situ_glu_sfpu.py:117`).
    - Record them in the port report.
  - Both originals already carry the pointer comment. Do not add one.
- **RTA varargs:** none. Name every arg.
  - RM reader: a fixed `arg++` run of 17 (`reader_unary_backward_row_major.cpp:25-45`). The two addresses become tensor bindings. That leaves 15 named args: 7 geometry values and 4 `RowMajorPaging` fields per tensor.
  - RM writer: 12 (`writer_unary_backward_row_major.cpp:18-31`): one tensor binding and 11 named args.
  - Each compute kernel reads 1 RTA, the tile count.
- **The op's own kernels convert in place.** The RM reader, the RM writer, `row_major_pages.hpp`, `operand_reconfig.hpp` and the five compute kernels are bound only by this op.
- **This is a binding-layer change, not an idiom rewrite.** Every kernel is already Device 2.0 with `Noc` and `DataflowBuffer`. Swap `tt::CBIndex` ids for `dfb::` tokens, `TensorAccessorArgs` plus a `Buffer*` address for `tensor::` tokens, and positional args for named args.
- **Compute kernels.** Pass `dfb::` tokens straight to `compute_kernel_hw_startup` and to `ckl::input(…)` / `ckl::output(…)` (`port_patterns.md`, "Pass DFB handles directly to LLKs and kernel-lib helpers"; precedent `eltwise/unary/device/kernels/compute/eltwise_pow_int.cpp:37-40`). Keep one `DataflowBuffer` per DFB for `CopyTile` / `PackTile` (e.g. `eltwise_bw_sigmoid.cpp:28-30`). Never construct two from one token. Each gelu-tanh kernel feeds one `dfb_input` to three `CopyTile`s; keep that single object.
- **Compute config is per DFB in Metal 2.0.**
  - Legacy sets `unpack_to_dest_mode[c_0]` and `[c_1]` to `UnpackToDestFp32` whenever fp32 DEST is on (`unary_backward_program_factory.cpp:362-366`). Translate that one-for-one: an `UnpackToDest` entry for both operand DFBs, by name, exactly when `enable_32_bit_dest` is set.
  - This reproduces the legacy lowered vector, and it satisfies the validator's rule that every consumed Float32 DFB needs an explicit entry when `enable_32_bit_dest` is set.
  - Do not index by CB number. Do not "fix" bf16 operands to `UnpackToSrc` either. Both forms behave the same on Gen1, but one-for-one is the faithful translation.
  - The other inputs come from `:355-357` (`fp32_dest_acc_en`, also forced for `SIGMOID_BW`), `:371-384` (the `MIXED_OPERAND_DATA_FORMATS`, `COPY_DEST_DATA_FORMAT` and `BF16_ROUNDING_MODE` defines) and `:387-397` (kernel path, math fidelity).
- **Aliased mode.** Set `borrowed_from` on c_0, c_1 and c_2, sized to the whole shard (`cb_pages = shard_tiles`, `:230`). Keep the `IN0_SHARDED` / `IN1_SHARDED` / `OUT_SHARDED` defines. Omit the kernel tensor bindings those defines compile out: the reader references no `tensor::` token under them, and the writer none under `OUT_SHARDED` (`port_patterns.md`, "Conditional / optional resource bindings"). Precedent: `data_movement/bcast/device/bcast_multi_core_hw_program_factory.cpp:126`, `:143`, `:173-184`.
- **Cores and per-core args.**
  - In aliased mode the cores follow the shard orientation (`:409-411`). Otherwise they follow the compute-grid column walk (`:413-416`).
  - `units` / `units_done` differ by core group (`:419-474`).
  - The reader's `block_height`, `block_width` and `num_cores_y` are dead while `block_or_width_sharded` is 0. Pass them as today.
- **`CoreLocalMem<uint8_t>(address)` in `row_major_pages.hpp` stays as it is.** It is a raw-address local L1 handle for CB slots and the scratch window, not a tensor binding.
- **Validation runs on every cache hit.** The op has no `validate_on_program_cache_hit`, and the relaxation analysis relies on that. Do not add a lighter hit validator.
- **Out of scope:**
  - Do not edit `ttnn/cpp/ttnn/kernel_lib/`, `tt_metal/` APIs, either fork, the custom hash, or the device-operation class.
  - Do not read or copy anything under `ttnn/cpp/ttnn/operations/experimental/quasar/`.
- **Install:** both forks, the op's own kernels and `kernel_lib` are covered by existing globs. No CMake change is needed.
- **Quasar debt, not port work.** The c_3 and c_4 self-loops are data-movement self-loops, which Gen2 rejects. They are recorded for the Quasar audit.

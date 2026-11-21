# Metal 2.0 Port Brief — `eltwise/unary_backward`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (on the invoker-supplied sheet row) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `0b4911ccc4f6 2026-11-21 Add unary_backward TensorParameter relaxation analysis` *(carry this line into the port report's Provenance section. It is this branch's relaxation-analysis commit; the recipe content itself is upstream `c41f7f4ef0a`, see the audit header)*

**Two decisions are still with the invoker. Ask them before acting; do not decide.**
1. What to do with op types that land while the port is in flight (audit Question 1; see *Anything else*, item 1).
2. Whether to carry over the implicit-sync opt-out on the aliased reader and writer (audit Question 2; see *Anything else*, item 8).

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor`. `UnaryBackwardProgramFactory::create_descriptor(args, tensor_args, output)`, `device/unary_backward_program_factory.cpp:26-27`. It takes no coordinate. It is the only factory (`device/unary_backward_device_operation.hpp:28`).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (the sheet's `Porting Target`). There is no `override_runtime_arguments` to translate. One program serves the mesh. A cache hit refreshes tensor bindings and nothing else. Every other value the factory computes is baked at the miss. That is safe here because the cache key pins everything those values depend on (the audit's relaxation checks confirm it).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args` (deprecated hook). A custom hash is present and a pybound `create_descriptor` is absent; neither gates.
- **Custom hash — leave it alone:** `device/unary_backward_device_operation.cpp:273-300`.
- **Pybind:** `unary_backward_nanobind.cpp` has no `create_descriptor` binding. There is nothing to delete.

## Construct — to do

**Tensor bindings** (per binding). Three `TensorParameter`s: `grad_output`, `input`, and `output` (the `Tensor& output` argument, which is the preallocated tensor when the caller gives one).

- **Address configs** (TILE not aliased, ROW_MAJOR not aliased) — **Case 1**. Every accessor already takes its base from the `Buffer*` RTA: `reader_binary_interleaved_start_id.cpp:46,53`, `writer_unary_interleaved_start_id.cpp:42`, `reader_unary_backward_row_major.cpp:55-56`, `writer_unary_backward_row_major.cpp:38`. Express each as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`. The three `Buffer*` RTA slots and the `TensorAccessorArgs` CTAs (`.cpp:307-308,315,330-331,341`) disappear.
- **Aliased config** (`can_alias_shards`, `.cpp:82-129`) — **clean**. `c_0`, `c_1`, `c_2` are borrowed from the three shards (the legacy `.buffer` at `.cpp:256-266`; in Metal 2.0, `DataflowBufferSpec::borrowed_from`) and no kernel builds an accessor (`IN0_SHARDED`, `IN1_SHARDED`, `OUT_SHARDED`). The legacy factory still pushes the `Buffer*` RTA slots in this config (`.cpp:468,470`), and they are dead there. Bind the tensors only where an accessor is built (port_patterns.md, *Conditional / optional resource bindings*).
- **No Case 2.** No kernel uses a base address raw.

**TensorParameter relaxation:** the sheet cell is `legality (see analysis)`. Doc: `analyses/relaxations/eltwise_unary_backward.md` (blob `f0c9ad2ebc69`). Read it in place; it is committed on this branch as `0b4911ccc4f` and is not on main. Declaration, transcribed — apply it to **every** `TensorParameter` (`grad_output`, `input`, `output`), unconditionally, in every layout, sharding regime and op type:

```cpp
.relaxations = {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true},
```

Not `{}`. Not `match_padded_shape_only` alone. Not `dynamic_tensor_shape`. The audit ran the doc's six validity checks against this tree and all passed. If the op's files change on main before the port starts, re-run them (doc, *Validity checks*); if any fails, stop and report the verdict UNCONFIRMED.

**TensorAccessor 3rd arg:** none. No accessor passes a page size.

**CB endpoints:**
- **Self-loop** `c_3` (reader scratch) and `c_4` (writer scratch), under `rm_blocked` only (ROW_MAJOR and not aliased). Each is touched by exactly one data-movement kernel, through the raw pointer only (`row_major_pages.hpp:140-143`). Bind that kernel PRODUCER and CONSUMER. The legacy factory already allocates them conditionally (`.cpp:272-289`); keep that condition.
- **Legal 1P+1C in every config:** `c_0` and `c_1` (reader produces, compute consumes), `c_2` (compute produces, writer consumes). No dead CB, no multi-binding, no other conditional DFB.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:**
  - **Reader** `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` — **no fork yet; this port creates it** (rung 2 of port_patterns.md, *Caution: Porting a shared kernel*). Add `reader_binary_interleaved_start_id_metal2.cpp` in that same directory, and add the pointer comment at the top of the original; change nothing else in the original. No CMake edit is needed (`eltwise/binary/CMakeLists.txt:16` globs `device/kernels/*.cpp` recursively).
    - Keep the kernel's full interface in the fork. This op supplies `block_or_width_sharded = 0` (CTA, `.cpp:324`) and `block_height = 0`, `block_width = 0` (`.cpp:468`). They are dead for this op and live for the other binder.
    - Take binding names from the kernel's own words. Its locals are `cb_id_in0/1`, `dfb0/1` and `src0_addr/src1_addr`. Names such as `dfb::in0`, `dfb::in1`, `tensor::src0` and `tensor::src1` follow from them. This is a suggestion; the first fork's names become every later consumer's interface.
    - The two `get_tile_size(cb_id)` calls (`:45,52`) are sanctioned in Device 2.0. The whitelist moves them onto the object (`get_entry_size()`, which the existing writer fork already does). The entry size equals the tile size here (`.cpp:244-251`). The value is unchanged.
    - Other binders of the original (**sunset list, not authorization to convert it in place**): `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:437`, a gtest on the legacy generic-op API. No other op factory binds it.
  - **Writer** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — **fork exists** beside it as `writer_unary_interleaved_start_id_metal2.cpp`. Bind it. Do not re-fork it and do not edit it; 26 op factories bind it.
    - Fit was checked. The fork takes `dfb::out`, `tensor::dst` and named args `num_pages`, `start_id`, and it reads the `OUT_SHARDED` define. This factory passes `{grad_input_buffer, units, units_done}` (`.cpp:470`), which are dst, `num_pages`, `start_id`, and sets `OUT_SHARDED` under aliasing (`.cpp:345-347`). The fork's `BACKWARDS` and `ENTRIES_PER_PAGE` / `PAGE_BYTES` knobs are not used here.
    - Other binders of the legacy original (sunset list): `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:894,1013`, `tests/ttnn/unit_tests/operations/eltwise/test_exp_approx_disable_sfploadmacro.py:86`, `tests/ttnn/unit_tests/operations/debug/test_generic_op_host_time.py:68`. No other op factory binds it.
  - **Own kernels:** the 41 compute kernels, the RM reader and writer, and `row_major_pages.hpp` belong to this op and nothing else binds them. Convert them in place.
- **RTA varargs:** none. The RM reader (17 RTAs, `reader_unary_backward_row_major.cpp:25-45`) and RM writer (12 RTAs, `writer_unary_backward_row_major.cpp:18-31`) read fixed-count runs through `arg++`; name every field. The per-tensor paging structs are four fixed fields each. No kernel reads compile-time args at a varying index.
- **Anything else the porter needs:**
  1. **Scope is all 41 compute kernels, and the set is open-ended.** There are 5 hand-registered kernels (`eltwise_bw_sigmoid.cpp`, `_tanh.cpp`, `_gelu_poly.cpp`, `_gelu_tanh.cpp`, `_gelu_tanh_fp32.cpp`) and 36 from `device/ops/*.def`. Count them at the start and after any rebase: `ls ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/ops/*.def | wc -l` (36 at audit time) and the `eltwise_bw_*.cpp` files (41). The kernel paths do not change. `get_kernel_spec` and the `.def` files stay untouched. **Open decision:** until the invoker answers audit Question 1, ask before converting any kernel that was not in the audit-time set.
  2. **Compute-kernel conversion shape.** The 36 `.def` kernels are clones (normalized diff against `eltwise_bw_abs.cpp`: 2 differing lines each). Convert one and apply the same edit to the rest.
     - `compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2)` runs *before* the id constants and takes raw CB numbers (for example `eltwise_bw_abs.cpp:21`). Give it the `dfb::` tokens.
     - The three `constexpr auto dfb_*_id = tt::CBIndex::c_N` lines become the tokens. `kernel_lib/eltwise/api/chain.hpp:48-66` documents the form: `CopyTile<input(dfb::in)>{in}` with `DataflowBuffer in(dfb::in)`. Keep the `DataflowBuffer` objects in spec order; a mismatch trips a watcher assert (`chain.hpp:65-66`).
     - `get_arg_val<uint32_t>(0)` becomes one named RTA. Name it `per_core_tile_cnt` in all 41 so one `runtime_arg_schema` serves every op type. Two kernels call it `num_tiles` today (`eltwise_bw_gelu_tanh.cpp:19`, `eltwise_bw_gelu_tanh_fp32.cpp:21`).
  3. **Per-CB indexing on the host.** `unpack_to_dest_mode` is a vector indexed by CB number (`.cpp:362-366`). It sets `UnpackToDestFp32` on `c_0` and `c_1` only, and only when `fp32_dest_acc_en`. In Metal 2.0 it is keyed per DFB; preserve exactly that.
  4. **Defines the factory supplies.** Compute kernels get `MIXED_OPERAND_DATA_FORMATS` (only when the two operand formats differ), `COPY_DEST_DATA_FORMAT` and `BF16_ROUNDING_MODE` (`.cpp:371-384`), whether or not a kernel uses them. The reader gets `IN0_SHARDED` and `IN1_SHARDED`, and the writer `OUT_SHARDED`, under aliasing (`.cpp:326-328,345-347`). All of them go through `compiler_options.defines`.
  5. **Per-op and per-dtype kernel choice.** `get_kernel_spec` picks the compute kernel per op type. `GELU_TANH_BW` has a second variant, `eltwise_bw_gelu_tanh_fp32.cpp`, chosen when `fp32_dest_acc_en` (`.cpp:387-390`). Use the multi-variant pattern in port_patterns.md.
  6. **Names tests depend on.** 38 nightly `test_backward_*.py` files, 38 `*_bw_device_perf.py` files and `tests/ttnn/unit_tests/operations/eltwise/test_unary_backward_device_op.py` depend on graph-capture strings containing `UnaryBackwardDeviceOperation` and `UnaryBackwardOpType::<NAME>`. Leave `UnaryBackwardParams`, `UnaryBackwardInputs`, the device-operation struct and the op-type names alone.
  7. **Stale contract comment.** `device/unary_backward_op_utils.hpp:19-21` tells new op-type authors the legacy kernel contract (CB indices, positional RTA 0). It goes stale at the port. It sits outside the factory body. Record it under *Open items for downstream* in the port report rather than editing it, unless the invoker says otherwise.
  8. **Open decision: implicit sync.** The earlier port of this op disabled DFB implicit sync on the aliased reader and writer (`create_reader_datamovement_config(can_alias_shards)` and the writer twin, commit `fd1d2dd967b`) at a reviewer's request. The recipe says a Gen1 port does not author Gen2 options. Ask the invoker which applies (audit Question 2). Both are no-ops on WH/BH.
  9. **Do not copy from the earlier port of this op** (PR #64224, branch `anasuya/metal2_port_eltwise_unary_bw`, still open). At its head (`33122d5ea13`) it still declares `.relaxations = {}` on all three slots, and its 36 `.def` kernels are still in legacy form. Ported ops are not templates in any case.
  10. **Tests to run, in small subsets.** `tests/ttnn/unit_tests/operations/eltwise/test_unary_backward_device_op.py` (the device-operation tests), `tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_gelu.py`, `tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_ulp.py`, and a few per-op nightly files for the `.def` kernels (for example `test_backward_abs.py`, `test_backward_log_sigmoid.py`, and `test_backward_softplus.py` as a scalar-parameter op). Leave the full nightly set to CI. The build on disk at audit time (2026-11-20 10:10) is older than this checkout. Rebuild before trusting a device result.
  11. **Post-port passes are separate work.** The two RM scratch DFBs are what `sync_free_dfbs.md` later converts to `Scratchpad`s, and `gen2_hardware_configs.md` adds Gen2 configs. Do neither in the port.

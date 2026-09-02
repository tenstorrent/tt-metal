# QUASAR_UPLIFT_REPORT — ttnn.concat (`data_movement/concat`)

- **Recipe:** `docs/source/ttnn/ttnn/ai/quasar_porting.md` (+ `metal_2.0/ai/audit/quasar_audit.md`, `metal2_audit.md`)
- **Date:** 2026-09-01
- **Driving test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_concat.py`
- **Op directory:** `ttnn/cpp/ttnn/operations/data_movement/concat/`

## Status: RED — Not Metal 2.0 on Gen1 yet

This is the first RED-stop condition in `quasar_porting.md` ("factory still
`create_descriptor`/`ProgramDescriptor`"). The Quasar uplift assumes an already-Metal-2.0 op;
concat is still on the `ProgramDescriptor` API, so the op needs the base Metal 2.0 port
(`ai/port/metal2_port.md`, starting from the `metal2_audit.md` pre-port audit) **before** any
Quasar uplift. Per the recipe, a grounded RED is the deliverable — the uplift was **not**
attempted and no source file was modified.

## What the test exercises (code-path scope)

`test_concat.py` has one captured signature: 21 identical calls to `ttnn.concat` with
**16 input tensors** `[1,1,32,8192]` ×15 + `[1,1,32,5376]`, BFLOAT8_B, TILE layout, DRAM
interleaved inputs, `dim=-1`, output `L1 INTERLEAVED` `[1,1,32,128256]`.

Host routing for this case (`concat.cpp` → `concat_impl` → `ConcatDeviceOperation`):

- Rank 4 → no unsqueeze massage; TILE with **no tile padding on the concat dim**
  (8192 = 256×32, 5376 = 168×32) → `untilize_rm_retilize` predicate false;
  last dim tile-aligned → no transpose fallback; 16 ≤ 47 → no batching split.
- `ConcatDeviceOperation::select_program_factory` (`device/concat_device_operation.cpp:20`):
  inputs not sharded, `can_use_tiled_unaligned_concat` false (no tile padding on dim) →
  **`ConcatProgramFactory`** (`device/concat_program_factory.cpp`), tiled branch.
- Kernels on this path:
  - reader: `device/kernels/dataflow/reader_concat_interleaved_start_id.cpp` (op-owned)
  - writer: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
    (**out-of-op shared kernel**, owned by `eltwise/unary`)

## Gate evidence (§1 of quasar_porting.md)

**Host factories — all 8 are `ProgramDescriptor`, none is `ProgramArtifacts`:**

| Factory | API |
|---|---|
| `concat_program_factory.cpp:17` (the exercised one) | `create_descriptor` → `ProgramDescriptor` |
| `concat_tiled_unaligned_program_factory.cpp:148` | `create_descriptor` |
| `concat_s2i_program_factory.cpp:16` | `create_descriptor` |
| `concat_s2s_rm_program_factory.cpp:37` | `create_descriptor` |
| `concat_s2s_tiled_program_factory.cpp:17` | `create_descriptor` |
| `concat_s2s_multi_program_factory.cpp:31` | `create_descriptor` |
| `concat_block_sharded_program_factory.cpp:31` | `create_descriptor` |

`grep -rn "create_program_artifacts\|ProgramArtifacts" ttnn/cpp/ttnn/operations/data_movement/concat/`
→ zero hits. No `dfb::` / `args::` / `tensor::` / `scratch::` bindings anywhere in the op.

**Kernel side:** the exercised kernels are Device 2.0 (they include `api/dataflow/*`, use
`DataflowBuffer`, `TensorAccessor`) but are **not** Metal 2.0 — positional
`get_arg_val<uint32_t>(i)`, CB indices via `get_compile_time_arg_val`, buffer-address RTAs +
`TensorAccessorArgs<N>` / `make_tensor_accessor_args_tuple` CTAs (e.g.
`reader_concat_interleaved_start_id.cpp:16-24`, `writer_unary_interleaved_start_id.cpp:19-39`).
That is the legacy arg model the Metal 2.0 port replaces; there is nothing for the
`dfb::`/`tensor::` binding tokens to attach to yet.

## Files changed

**None.** No source edit was made (RED-stop before the uplift; a Metal 2.0 port is explicitly
out of scope for this pass).

## §7–§8 gotchas: applied / considered

- **Applied: none** — the gate failed before the Quasar-uplift audit, and §7–§8 fixes are
  reactive (no device run was performed in this session).
- **Considered, recorded as heads-ups for the eventual port** (do not act on them now):
  - `writer_unary_interleaved_start_id.cpp:27` reads
    `get_local_cb_interface(cb_id_out).fifo_page_size` — the exact §5/§8.3 stale-`fifo_page_size`
    hazard on Quasar (fix at port time: `DataflowBuffer::get_entry_size()`). This kernel is
    **out-of-op** (`eltwise/unary`), so per the shared-kernel rule it would need coordination or
    a `_metal2` fork beside the original — never an in-place edit from a concat pass.
  - No semaphores are created anywhere in the op (`quasar_audit.md` check 2: vacuously clean).
  - The exercised path has a single 1-producer/1-consumer CB (`src0_cb_index=0`, reader→writer
    across two DM kernels) — a normal FIFO, not a DM self-loop; no CB-redesign debt visible on
    this path (`quasar_audit.md` check 1 looks clean for `ConcatProgramFactory`).
  - `compute_program_hash` is custom (`concat_device_operation.cpp:78`) because factory selection
    reads live L1 occupancy — the M2 port must carry it verbatim (base-port concern).

## Deferred / follow-up items

1. **Base Metal 2.0 port of `data_movement/concat`** — prerequisite for any Quasar uplift. Run
   `metal2_audit.md` first (note: 8 factories, one device-op → the audit/port is per-factory;
   the llama32_1b_quasar model needs only `ConcatProgramFactory`'s tiled interleaved path).
2. **Shared writer kernel** `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` is used by
   many ops; its M2 conversion (and the `fifo_page_size` → `get_entry_size()` fix) is a
   shared-kernel decision (`_metal2` fork rung), not concat-local.
3. The block-sharded factory hard-caps at 16 input CBs (CB ids 0..15 + out at 16) — worth
   re-examining under DFB naming at port time (not exercised by this test).

## WH/BH parity claim

**Trivially holds: the working tree carries zero changes from this pass** — no file was created
or modified except this report. WH/BH behavior is bit-for-bit the pre-existing `main` path.

## Test commands (user runs; none were run in this session)

- **Target case (WH/BH parity baseline and, later, Quasar):**
  `pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_concat.py`
  (on the emulator, optionally `TTNN_GRAPH_OPS_NO_GOLDEN=1` to drop the bfloat8_b PCC golden)
- **Op suite (BH → WH order per recipe §9):**
  `pytest tests/ttnn/unit_tests/operations/data_movement/test_concat.py`
- After any future kernel edit: `TT_METAL_FORCE_JIT_COMPILE=1`, and purge
  `~/.cache/tt-metal-cache` between baseline and post-port runs.

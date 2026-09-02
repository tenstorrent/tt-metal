# QUASAR_UPLIFT_REPORT — ttnn.embedding (fused-tilized path)

> Per `docs/source/ttnn/ttnn/ai/quasar_porting.md`. Uncommitted, for review; delete before merge.

## Status: RED — Not Metal 2.0 on Gen1 yet (on the exercised code path)

This is the recipe's first RED-stop condition ("factory still `create_descriptor`/`ProgramDescriptor` → do the Metal 2.0 port first"). Per §1 of `quasar_porting.md`, the uplift was **stopped at the gate**; no Metal 2.0 port was performed and no Quasar-uplift audit fixes were applied.

## Op / test mapping

Driving test: `models/experimental/llama32_1b_quasar/tests/graph_ops/test_embedding.py` — 3 captured cases of `ttnn.embedding`, all with UINT32 ROW_MAJOR index tensors and `layout=TILE`.

Host path (`ttnn/cpp/ttnn/operations/embedding/embedding.cpp`):
- Case 00's TILE weights are first converted to ROW_MAJOR via `ttnn::to_layout` (line 30–32).
- All 3 cases satisfy the fused-tilized conditions (indices width % 32 == 0, weight width % 32 == 0, `layout=TILE`) → `fused_tilized = true` (lines 46–59).
- `ttnn::prim::embedding` → `EmbeddingsDeviceOperation::select_program_factory` (`device/embedding_device_operation.cpp:17-26`): indices are ROW_MAJOR (not TILE) and `tilized=true` → **`EmbeddingsFusedProgramFactory`** for **all 3 cases**.
- The surrounding composite also calls `ttnn::reshape`, `ttnn::to_layout`, `ttnn::unsqueeze_to_4D`, and (conditionally) `ttnn::typecast` — separate ops, out of scope here.

## Gate evidence (why RED)

The op is in a **mixed** migration state — two of its three factories are Metal 2.0, but the one this test exercises is not:

| Factory | API era | On this test's path? |
|---|---|---|
| `EmbeddingsFusedProgramFactory` (`device/embeddings_fused_program_factory.cpp:18`) | **Legacy**: `create_descriptor` → `tt::tt_metal::ProgramDescriptor`, `CBDescriptor` + `tt::CBIndex::c_0` (line 133), `KernelDescriptor` with positional compile-time/runtime args | **Yes — all 3 cases** |
| `EmbeddingsRMProgramFactory` (`device/embeddings_rm_program_factory.cpp:21`) | Metal 2.0: `create_program_artifacts` → `ProgramArtifacts`, `dfb::` bindings | No |
| `EmbeddingsTilizedIndicesProgramFactory` (`device/embeddings_tilized_indices_program_factory.cpp:22`) | Metal 2.0: `create_program_artifacts` → `ProgramArtifacts`, `dfb::` bindings | No (only for TILE-layout indices) |

Kernels bound by the fused factory (all in Metal-1 binding form — positional `get_arg_val<uint32_t>(i)`, `get_compile_time_arg_val(i)` CB indices, address-RTA + `TensorAccessorArgs<N>`):
- Reader: `device/kernels/dataflow/embeddings_tilize.cpp` — Device-2.0 objects (`Noc`, `CircularBuffer`, `TensorAccessor`) but Metal-1 arg binding (`get_arg_val` at lines 15–20, `TensorAccessorArgs<10>` at line 33); no `dfb::`/`args::`/`tensor::` tokens.
- Compute: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp` (all 3 test shapes are non-chunked: 2 resp. 64 tiles/block, well under the 1MB chunking threshold at factory lines 104–109; the chunked variant `device/kernels/compute/tilize_chunked.cpp` is only used above it).
- Writer: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — the **legacy** shared writer; its Metal 2.0 fork `writer_unary_interleaved_start_id_metal2.cpp` already exists beside it (see the kernel's header note and issue #52228).

Per `quasar_porting.md` §1, a factory on `create_descriptor`/`ProgramDescriptor` — even with kernels partly on Device-2.0 objects — is "not ported; stop and run the Metal 2.0 port first."

## Files changed

**None.** Zero source changes were made (a RED gate result stops before any edit).

## §7–§8 gotchas: applied / considered

- **Applied: none** — the uplift never started (gated out at §1), and §7–§8 fixes are reactive; no device run was performed in this session.
- **Considered / recorded for the future Metal 2.0 port of the fused path:**
  - §5 / §8.3 `fifo_page_size` hazard: the legacy shared writer reads `get_local_cb_interface(cb_id_out).fifo_page_size` (`writer_unary_interleaved_start_id.cpp:27`) — stale on Quasar. Resolved for free by binding the existing `_metal2` writer fork during the M2 port.
  - §7 uint32 device format: the reader moves UINT32 index pages via NoC only (no compute-format branch on the index dtype in the fused kernels), so per §7 this looks like a "merely forwards a DataType" case — to be re-checked at uplift time.
  - quasar_audit.md check 2 (non-zero-init semaphores): the fused factory creates **no** semaphores — clean.
  - quasar_audit.md check 1 (DM self-loop / CB redesign debt): `cb_in1` in `embeddings_tilize.cpp` (lines 40–91) is a reader-local index scratch CB — reserved, written, and committed by the **same DM kernel** with no other consumer. After the M2 port this is exactly the sync-free/self-loop shape that the uplift must convert to `Scratchpad`/`LocalTensorAccessor` (`sync_free_dfbs.md` / `dm_self_loop_dfbs.md`); flagging it now for the porter.

## Deferred / follow-up items

1. **Metal 2.0 port of `EmbeddingsFusedProgramFactory`** (`ai/port/metal2_port.md`) — the blocker this RED routes to the porting workstream. The op's other two factories are already M2 in-directory, so the port has an in-op reference idiom (including the intra-op `#define`-split pattern already used in `embeddings_common_metal2.hpp` / factory lines around `dfb::` cache-handle comments).
2. That port should bind the existing `writer_unary_interleaved_start_id_metal2.cpp` fork (no new out-of-op write needed).
3. After the port: re-run the Quasar-uplift audit; convert the reader's self-looped index-scratch CB (`cb_in1`) per `dm_self_loop_dfbs.md`.
4. Host-side composite dependencies (`to_layout`, `reshape`, `typecast`, `unsqueeze`) each need their own Quasar status; out of scope for this op directory.

## WH/BH parity claim

Structural: the working-tree diff for this op is **empty** (no source file touched; the only artifact is this report). WH/BH therefore keep the original path by construction — there is no behavior to regress.

## Test commands (user-run; none were run in this session)

BH/WH parity (should pass unchanged — nothing was modified):

```bash
pytest tests/ttnn/unit_tests/operations/data_movement/test_embedding.py
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_embedding.py
```

Quasar (expected to exercise the un-ported legacy fused factory — for baseline/triage only, not a pass expectation until the M2 port lands): the same graph-op test under the Quasar emulator environment:

```bash
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_embedding.py
```

Use `TT_METAL_FORCE_JIT_COMPILE=1` after any kernel change, and purge `~/.cache/tt-metal-cache` between pre/post-port runs.

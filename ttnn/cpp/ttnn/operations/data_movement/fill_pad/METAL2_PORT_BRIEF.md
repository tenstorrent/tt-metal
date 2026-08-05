# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/fill_pad`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (both factories) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `56373090d3d 2026-08-05 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

> **This is a re-port.** The previous Metal 2.0 port (#50904) was reverted (#51605) for a sanity-test failure. The current code is the clean post-revert baseline. The revert's root cause is a sharded-factory CB deadlock — see **Watch for → Sharded lock-step counts** and treat it as the highest-risk part of this port.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both factories port to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — each factory is a `create_descriptor()` returning a `tt::tt_metal::ProgramDescriptor` (`device/fill_pad_program_factory.hpp:97,105`). Execution model SPMD.
- **Op-owned tensors:** none — the target concept needs no op-owned-tensor carry.
- **Target concept:** `ProgramSpecFactoryConcept` (both factories).
- **Op shape:** single `FillPadDeviceOperation`, two factories selected by input (`device/fill_pad_device_operation.cpp:15-24`): `FillPadProgramFactory` for DRAM (interleaved / DRAM-sharded), `FillPadL1ShardedProgramFactory` for L1-sharded. The op is **in-place** — `create_output_tensors` returns the input tensor (`device/fill_pad_device_operation.cpp:39-43`), so reader and writer bind the *same* tensor.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` — all `no` on both factory rows of the readiness sheet, and confirmed in code.

## Construct — to do

**Tensor bindings** — one binding, `input` (in-place; also the output). Classification differs per factory:

- `input` — **Case 1** (via `TensorAccessor`) in **`FillPadProgramFactory`** (DRAM) → express as a `TensorParameter` / `TensorBinding`; the reader and writer build `TensorAccessor(tensor::name)` instead of `TensorAccessor(args, buf_addr, tile_bytes)` (`device/kernels/dataflow/fill_pad_reader.cpp:86-87`, `fill_pad_writer.cpp:80-81`). The RTA[0] `buf_addr` and the host `TensorAccessorArgs(*tens_buffer).append_to(...)` plumbing (`fill_pad_program_factory.cpp:173,192`) both disappear.
- `input` — **Case 2** (raw pointer) in **`FillPadL1ShardedProgramFactory`** (L1-sharded) → bind the tensor as a `TensorParameter`, pull the base via `TensorAccessor::get_bank_base_address`, and **keep the existing raw address arithmetic unchanged** (`shard_l1_base + <geometry>` self-reads/writes through `UnicastEndpoint`, `device/kernels/dataflow/fill_pad_sharded_reader.cpp:70,90,111`; `fill_pad_sharded_writer.cpp:92,111,125,157`). Do **not** rewrite the raw walk into `TensorAccessor` iteration.
- Today both factories smuggle the base through the framework `Buffer*`-binding form (`emplace_runtime_args({tens_buffer, …})`, `fill_pad_program_factory.cpp:293-295,620-623`) — correct-on-cache-hit but superseded by the typed binding. The compute kernel (`device/kernels/compute/fill_pad_compute.cpp`) touches no tensor memory (CB-only) — no binding work there.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** drop the redundant `tile_bytes` 3rd argument at `device/kernels/dataflow/fill_pad_reader.cpp:87` and `device/kernels/dataflow/fill_pad_writer.cpp:81` (Class 2 — pure no-op). Do **not** set `dynamic_tensor_shape`; this is Class 2, not Class 1. The sharded kernels construct no `TensorAccessor`, so nothing to drop there.

**CB endpoints:** all legal — every CB is a plain 1:1 FIFO in both factories. Bindings, per node:

- `c_0` (data-in): **reader** PRODUCER → **compute** CONSUMER.
- `c_1` (right-mask, only when `has_right_pad`): **writer** PRODUCER → **compute** CONSUMER. *(Note: the writer, not the reader, produces this mask.)*
- `c_2` (bottom-mask, only when `has_bottom_pad`): **writer** PRODUCER → **compute** CONSUMER.
- `c_16` (data-out): **compute** PRODUCER → **writer** CONSUMER.

No self-loop, 1P+1C assignment, multi-binding flag, or dead-CB drop is required. Declare `c_1`/`c_2` only in the configs that allocate them (`has_right_pad` / `has_bottom_pad`), matching the current conditional CB allocation (`fill_pad_program_factory.cpp:109,122,484,495`).

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none — the op owns all five kernels (`device/kernels/**`), instantiated by file path from its own directory. No borrowed files, no `_metal2` fork to reuse (create none), no sunset list.
- **RTA varargs:** none — name every RTA. The reader/writer RTAs are `buf_addr` + per-phase `(start, num)` pairs (`fill_pad_reader.cpp:78-84`); the compute RTAs are `num_right`/`num_bottom`/`num_corner` (`fill_pad_compute.cpp:104-106`); the sharded RTAs are `shard_l1_base`/`shard_H_tiles`/`has_bottom_pad_core`/`num_work`/`local_right_col` (`fill_pad_sharded_reader.cpp:46-50`). All fixed, nameable fields.
- **Sharded lock-step counts — the revert root cause; highest risk.** #50904 was reverted after a **sharded-factory CB deadlock** (WIDTH_SHARDED): the reader/writer moved a tile count that did not match the compute kernel's consumed count (`num_work` conflated with a per-phase count vs. compute's `num_bottom = local_valid_w`). In the baseline you're porting, the reader/writer derive their tile counts from *shard geometry* (`shard_H_tiles`, `local_valid_w`, `has_right_pad`, `has_bottom_pad_core`, `fill_pad_program_factory.cpp:628-643`), and `num_work` is only a "has any work" guard (writer early-return `fill_pad_sharded_writer.cpp:60`; inert in the reader). **Preserve the exact right→bottom→corner lock-step counts across reader ↔ compute ↔ writer, and do not repurpose `num_work` as a loop bound.** Re-verify the WIDTH_SHARDED (97,97)-style partial-rightmost-shard case explicitly.
- **Ignore for the port (team-only cleanups, do not fold into the diff):** a dead `elem_size` CTA (`fill_pad_reader.cpp:64`, `fill_pad_compute.cpp:94`, `fill_pad_sharded_reader.cpp:41`) and the inert `num_work` RTA in the sharded reader. These are `METAL2_PREPORT_AUDIT.md` "Misc anomalies," not port work.

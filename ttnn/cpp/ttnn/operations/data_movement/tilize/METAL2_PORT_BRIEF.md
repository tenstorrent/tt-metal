# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/tilize`

> **Config-scoped port.** The op is RED at op level because `TilizeMultiCoreShardedProgramFactory` fails the TTNN factory-concept gate. This brief covers the **clean subset only**: `TilizeMultiCoreDefaultProgramFactory`, `TilizeMultiCoreBlockProgramFactory`, and `TilizeSingleCoreProgramFactory`. **Do NOT port the sharded factory** — leave `select_program_factory`'s sharded branch on the legacy path. The full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared (for the subset):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (Default/Block/SingleCore) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Excluded — blocked:** `TilizeMultiCoreShardedProgramFactory` — `Runtime-args update == yes` (`get_dynamic_runtime_args`, `tilize_device_operation.cpp:270`) **and** `Is safe to port? == no`. Routed to the TTNN/PD-migration team and readiness-sheet owner respectively.

**Recipe docs:** `b6dadc46ee0 2026-07-21 docs: fix metal_2.0 doc links (READ_ME_FIRST rename + two stragglers)` *(carry into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the subset factories port to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (all three subset factories define `create_descriptor()` returning a `ProgramDescriptor`).
- **Op-owned tensors:** none — carried by neither the source nor target concept.
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · migration-risky pybind — all `no` on the subset factories.
- **Note:** `get_dynamic_runtime_args` exists on the device op but early-returns `{}` for these three factories (`tilize_device_operation.cpp:278-281`); it is live only for the excluded sharded factory. It does not affect the subset.

## Construct — to do

**Tensor bindings** (per binding — identical across Default, Block, SingleCore):

- **input** — **Case 1** (via `TensorAccessor`). Today the factory passes the input `Buffer*` in the reader's RTA slot 0 (`Buffer*`-binding form); the reader reconstructs `TensorAccessor(src_tensor_args, src_addr)` and reads through it. Port: express as a `TensorParameter` / `TensorBinding`; the reader builds `TensorAccessor(tensor::name)`. The `Buffer*` RTA slot and the `TensorAccessorArgs` compile-time plumbing both disappear.
- **output** — **Case 1** (via `TensorAccessor`). Same shape: the output `Buffer*` rides the writer's RTA slot 0; the writer (`writer_unary_interleaved_start_id.cpp` for Default/SingleCore, `writer_unary_interleaved_start_id_wh.cpp` for Block) builds a `TensorAccessor` from it. Bind as `TensorParameter` / `TensorBinding`; writer uses `TensorAccessor(tensor::name)`.

No Case 2 (no raw-pointer/bridge access) and no borrowed-memory DFB reads in the subset — those are confined to the excluded sharded factory.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is already the 2-arg form; nothing to drop.

**CB endpoints:**
- **Default / SingleCore:** `c_0` and `c_16` are both legal 1:1 (reader→compute, compute→writer). No action.
- **Block:** `c_0`, `c_16` legal 1:1. **`c_1`** (per-row DRAM-alignment staging buffer) is touched by the **reader only** → **self-loop**: bind the reader as both PRODUCER and CONSUMER. Kernel code is untouched. Applies to all block configs.

## Watch for

- **CB endpoints (multi-binding):** none in the subset — no hidden second writer, no multi-reader CB. (`c_1` is a plain single-toucher self-loop, not a multi-binding.)
- **Cross-op / shared kernels — port the shared kernel as one unit:**
  - `eltwise/unary/…/writer_unary_interleaved_start_id.cpp` (Default/SingleCore writer) is **broadly shared (~28 factories)**. Its CB→DFB / named-token rewrite must land across all co-borrowers in the same change, or the first isolated migrant breaks the rest. Coordinate before touching it.
  - `eltwise/unary/…/writer_unary_interleaved_start_id_wh.cpp` (Block writer) is shared with `tilize_with_val_padding`.
  - `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` (Default/SingleCore compute, shared pool) is shared across `tilize` + `tilize_with_val_padding`.
  - `tilize_with_val_padding/…/reader_unary_pad_multicore_both_dims.cpp` (Block reader) is in-family; it calls `tt_memmove(Noc,…)` from `data_movement/common/kernels/common.hpp` — Device 2.0 native, ports cleanly.
- **RTA varargs:** none — name every RTA (all read at distinct constant indices). Note the dead RTA slots (2, 6, 7 in the Default/SingleCore readers) and the dead CTA slot 0 (`aligned_page_size`/`stick_size`) documented in `METAL2_PREPORT_AUDIT.md` → Misc anomalies; these are **not** yours to fix in the port (leave behavior unchanged).

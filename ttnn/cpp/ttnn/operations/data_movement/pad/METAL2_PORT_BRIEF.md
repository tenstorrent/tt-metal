# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/pad`

> **Config-scoped subset port.** The op is RED at the op level: `PadRmShardedHeightOnlyProgramFactory` is gated by the TTNN factory-concept prerequisite (`Is able to port? == no`: `Runtime-args update == yes` + `Is safe to port? == no`) and is **out of scope** for this port. All five other reachable factories cleared every gate — this brief covers **only** those. The full record is in `METAL2_PREPORT_AUDIT.md`.
>
> **In scope (port these):** `PadRmReaderWriterMultiCoreDefaultProgramFactory`, `PadRmReaderWriterProgramFactory`, `PadRmShardedWidthOnlyProgramFactory`, `PadTileCoreProgramFactory`, `PadTileMulticoreProgramFactory`.
> **Out of scope (leave on the current path):** `PadRmShardedHeightOnlyProgramFactory` (gated), `PadRmReaderWriterMultiCoreProgramFactory` (dead/unreachable — do not port; recommend the ops team removes it).

**Gates cleared (for the subset):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (5 factories) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `8f7eb3e47dc 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The subset spans two current concepts, both targeting `MetalV2FactoryConcept`:

- **Current concepts:**
  - `descriptor` (→ `MetalV2FactoryConcept`, no op-owned tensors): `PadRmReaderWriterMultiCoreDefault`, `PadRmShardedWidthOnly`, `PadTileCore`, `PadTileMulticore`.
  - `WorkloadDescriptor`, **secretly SPMD** (one `ProgramDescriptor` replicated across mesh coords → collapses to single-program): `PadRmReaderWriterProgram`.
- **Op-owned tensors:** present **only** on `PadRmReaderWriterProgramFactory` — a pad-value const tensor built on cache miss and parked on `wd.buffers` (`pad_rm_reader_writer_program_factory.cpp:197-200`; held as a `Tensor`, not a bare `MeshBuffer`, per #44565). Carried natively by `MetalV2FactoryConcept`; wire it as an op-owned tensor binding.
- **Target concept:** `MetalV2FactoryConcept` (with the op-owned pad-value tensor for `PadRmReaderWriterProgram`).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no` on every subset factory.

## Construct — to do

**Tensor bindings** (per binding, per factory — every binding is Case 1 or clean; **no Case 2, no `get_bank_base_address` bridge anywhere**):

- **`PadRmReaderWriterMultiCoreDefault`** — `input` **Case 1**, `output` **Case 1**. Both arrive as `Buffer*` bindings (`..._default_...cpp:221-222`) and are consumed through `TensorAccessor`; express each as a `TensorParameter`/`TensorBinding`, kernel builds `TensorAccessor(tensor::name)`, and the `Buffer*` RTA + its `TensorAccessorArgs` plumbing disappear.
- **`PadRmReaderWriterProgram`** — `input` **Case 1**, `output` **Case 1**, **pad-value const (op-owned)** **Case 1**. The const tensor's address rides arg 13 as a `Buffer*` binding (`..._program_factory.cpp:156`) and is consumed via `TensorAccessor`; bind it as the op-owned tensor and build `TensorAccessor(tensor::name)`.
- **`PadRmShardedWidthOnly`** — `input` **clean**, `output` **clean**. Both are borrowed-memory sharded CBs (`cb_input.buffer = input_buffer` @ `pad_rm_sharded_width_only_program_factory.cpp:75`; `cb_output.buffer = output_buffer` @ `:91`). Translate mechanically via `DataflowBufferSpec::borrowed_from`; no `TensorParameter` address plumbing (the factory already passes empty runtime-arg lists).
- **`PadTileCore`** — `input` **Case 1** (donor reader), `output` **Case 1** (`Buffer*` bindings @ `pad_tile_program_factory.cpp:122,126`).
- **`PadTileMulticore`** — `input` **Case 1**, `output` **Case 1** (`Buffer*` bindings @ `pad_tile_multicore_...cpp:222-223`; idle cores pass `0u`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** drop the redundant `accessor_page_size` arg — **`PadRmReaderWriterMultiCoreDefault` only** (`reader_pad_dims_rm_interleaved_v2.cpp:95`, CTA 21; `writer_pad_dims_rm_interleaved_v2.cpp:25`, CTA 4). Class 2 (redundant/inert): the value equals the logical page on the interleaved path (realigned away) and `aligned_page_size()` on the sharded path — exactly what Metal 2.0 supplies implicitly. No other subset kernel passes a 3rd arg.

**CB endpoints:**

- **self-loop** (single toucher — bind the one kernel PRODUCER *and* CONSUMER): default-MC `c_1` (`cb_pad`) and `c_2` (`dfb_pad_align`, present only when `stick_size_padded_front != 0 || unaligned`); width-only `c_0` (input shard, borrowed) and `c_1` (`pad_val`); tile-core `c_1` (pad buffer); tile-MC `c_2` (`pad_val_cb`).
- **1P+1C assignment** (two touchers): width-only `c_16` (output shard, borrowed) — writer is the locked producer (`reserve_back`/`push_back`), reader the locked consumer (`wait_front`/`pop_front`); a genuine FIFO handshake (writer pre-fills padding, reader overwrites the data region). Bind writer PRODUCER, reader CONSUMER.
- **plain 1:1** (no action beyond normal binding): default-MC `c_0`, single-core-RM `c_0`, tile-core `c_0`, tile-MC `c_0` (`input_cb`).
- **dead-CB drop:** tile-MC `c_1` (`output_cb`) — allocated at `pad_tile_multicore_program_factory.cpp:70-78`, index threaded to the writer as CTA 1, but `writer_pad_tiled.cpp:23` reads it into an unused `constexpr` and no kernel touches it. Drop the allocation **and** the dead CTA. (Confirmed unreferenced across the single tile-MC config; a truly-dead DFB would otherwise be rejected by the spec validator — but the drop here is positively confirmed, not deferred.)

## Watch for

- **CB endpoints (multi-binding):** none. No hidden second writer (no semaphore-gated raw co-fill) and no ≥3-toucher CB anywhere in the subset. Do **not** reach for the multi-binding advanced option; width-only `c_16` is a plain producer/consumer FIFO, not a multi-binding.
- **Cross-op / shared kernels:**
  - `PadTileCore` file-path-instantiates the **cross-family donor** `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`, which ~17 ops share. Its CB→DFB / named-token rewrite is a **single coordinated change** — do not migrate it in isolation, or the co-borrowers break. Coordinate with the port-together set (listed in the audit's Heads-ups).
  - The v2 kernels `#include "ttnn/operations/data_movement/common/kernels/common.hpp"` and call `noc_async_read_sharded` / `noc_async_write_sharded` (signatures take `Noc` + a templated `TensorAccessor` — Device 2.0 native). Port the `data_movement/common` shared header together with the family; the call sites need no signature change.
- **RTA varargs:** in `reader_pad_tiled.cpp` / `writer_pad_tiled.cpp` (tile-MC), the four per-dim arrays (`input_page_shape`, `output_page_shape`, `input_id_per_dim`, `output_id_per_dim`) are one `get_arg_addr` block bounded by the `num_dims` CTA — **rank-length, so vararg**, not per-element-nameable. Use the RTA vararg mechanism for that block; name the three leading scalars (base addr, page count, start offset) normally. (Supported in Metal 2.0 — a heads-up, not a gate.)

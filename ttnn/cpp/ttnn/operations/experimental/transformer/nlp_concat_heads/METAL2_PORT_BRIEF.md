# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — a single `NLPConcatHeadsProgramFactory::create_descriptor` with an internal `if (in_sharded)` branch (interleaved vs. sharded paths, one factory).
- **Op-owned tensors:** none — carried natively by the target concept if any existed; there are none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind (`Is safe to port? = yes`). All `no` on this op.

## Construct — to do

**Tensor bindings** (per binding — classification differs by config; wire both):

- **input tensor**
  - *interleaved path* — **Case 1** (via `TensorAccessor`). Legacy passes `in0_buffer` as a `Buffer*` RTA (`program_factory.cpp:201`) → kernel `in0_tensor_addr = get_arg_val<uint32_t>(0)` → `TensorAccessor(in0_args, in0_tensor_addr)` (`reader_tm_tile_layout_nlp_concat_heads.cpp:31`). Express as a `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`. The `Buffer*` RTA and its `TensorAccessorArgs<4>` plumbing both disappear.
  - *sharded path* — **clean** (borrowed-memory DFB). The input is bound as a borrowed CB (`CBDescriptor.buffer = in0_buffer`, `program_factory.cpp:150`) and read via `cb_in0.get_read_ptr()`. Port via `DataflowBufferSpec::borrowed_from` the input `TensorParameter`; no `TensorAccessor`.
- **output tensor**
  - *interleaved path* — **Case 1** (via `TensorAccessor`). Legacy passes `out_buffer` as a `Buffer*` RTA (`program_factory.cpp:210`) → donor writer `dst_addr = get_arg_val<uint32_t>(0)` → `TensorAccessor(dst_args, dst_addr)` (`writer_unary_interleaved_start_id.cpp:31`). Express as a `TensorParameter` / `TensorBinding`; donor kernel builds `TensorAccessor(tensor::name)`.
  - *sharded path* — **clean** (borrowed-memory DFB). Output bound as a borrowed CB (`CBDescriptor.buffer = out_buffer`, `program_factory.cpp:163`) and written via `cb_out0.get_write_ptr()`. Port via `DataflowBufferSpec::borrowed_from` the output `TensorParameter`.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor passes a 3rd argument; nothing to drop.

**CB endpoints:**

- *Interleaved* `cb_src0` (index 0) — legal **1:1** (reader PRODUCER, donor writer CONSUMER). No action. (Output is interleaved on this path → CB 16 is not allocated.)
- *Sharded* `cb_src0` (index 0, borrowed from `in0_buffer`) — **assign 1P+1C.** Two touchers: the reader-config and writer-config instances of the same sharded kernel, each raw-reading a disjoint head range. Bind one instance PRODUCER, the other CONSUMER (cosmetic on Gen1).
- *Sharded* `cb_out0` (index 16, borrowed from `out_buffer`) — **assign 1P+1C.** Same two instances, each raw-writing a disjoint range.

> Note: the sharded kernel calls `cb_in0.reserve_back` / `cb_out0.reserve_back` with **no** matching `push_back` (one marked `// Redundant`, the other's `push_back` commented out). These do not lock a producer role — the movement is sync-free raw addressing — so the disposition is 1P+1C, **not** the multi-binding flag. Do not reach for the flag here.

## Watch for

- **CB endpoints (multi-binding):** none forced. The sharded path is a **dual-instance work-split** (same `kernel_source` in two `KernelDescriptor`s differing only by `ReaderConfigDescriptor` / `WriterConfigDescriptor` and per-instance offset args, both over `all_cores`) — the standard two-toucher shape that resolves to 1P+1C. There is no hidden second writer (co-fills are visible and sync-free, no coordinating semaphore) and no ≥3-toucher CB.
- **Cross-op / shared kernels:** the interleaved writer is `eltwise/unary/.../writer_unary_interleaved_start_id.cpp`, a **broadly-shared donor** (≈44 ops instantiate it). Its CB→DFB / named-token rewrite is a single shared change — port it as one unit across its co-borrowers, not in isolation for this op.
- **RTA varargs:** none — all RTAs are fixed distinct fields at constant indices; name each (prefer named RTAs over varargs).
- **Sharded output-CB assumption:** the sharded kernel binds `cb_out0` unconditionally, but the factory allocates it only when the output is sharded. Confirm sharded-in ⇒ sharded-out holds before relying on the borrowed output-CB binding (see the audit's Questions section).

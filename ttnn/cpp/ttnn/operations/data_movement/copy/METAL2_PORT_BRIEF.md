# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/copy`

> **Scoped subset brief.** The op is RED at op level (Device 2.0), but the block is confined to the `SameMemoryConfig` factory's row-major kernels. This brief covers the **clean factory subset that clears every gate: `DefaultRowMajor` and `DefaultTilized`.** Do **not** port `SameMemoryConfig` — it is blocked (see `METAL2_PREPORT_AUDIT.md` → Gate detail → Device 2.0). The full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared (for `DefaultRowMajor` + `DefaultTilized`):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `b11662e579e 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both in-scope factories port to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (both factories — `create_descriptor` returning `ProgramDescriptor`).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind (`Is safe to port? == yes`). All `no` on the cleared factories.
- **Do not touch `SameMemoryConfig`** in this port — it remains on the legacy `ProgramDescriptor` path until its three Device-1.0 row-major kernels migrate (Device 2.0 track).

## Construct — to do

**Tensor bindings** (per binding):

- **`DefaultRowMajor`**
  - `input` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter`/`TensorBinding`; reader builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(src_args, src_addr)` (`redistribute_pages_row_major_reader.cpp:37-38`). Drop the `Buffer*` RTA (`copy_default_row_major_program_factory.cpp:172`) and the `TensorAccessorArgs(input.buffer())` CTA plumbing (`:136`).
  - `output` — **Case 1** → writer builds `TensorAccessor(tensor::name)` (`redistribute_pages_row_major_writer.cpp:30-31`). Drop the `Buffer*` RTA (`:173`) and `TensorAccessorArgs(output.buffer())` CTA (`:148`).
- **`DefaultTilized`**
  - `input` — **Case 1** → reader (`reader_unary_interleaved_start_id.cpp`) builds `TensorAccessor(tensor::name)`. Drop the `Buffer*` RTA (`copy_default_tilized_program_factory.cpp:144`) and `TensorAccessorArgs(input.buffer())` CTA (`:105`).
  - `output` — **Case 1** → writer (`writer_unary_interleaved_start_id.cpp`) builds `TensorAccessor(tensor::name)`. Drop the `Buffer*` RTA (`:145`) and `TensorAccessorArgs(output.buffer())` CTA (`:110`).

All four are the `Buffer*` BufferBinding delivery form today (correct-on-cache-hit; not a correctness hazard) — the swap to typed bindings is mechanical.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor passes an explicit page size.

**CB endpoints:**

- **`DefaultRowMajor`** — self-loop `c_0` (one toucher: the reader both stages into and drains its own input scratch CB, `redistribute_pages_row_major_reader.cpp:42,183-185`; bind the reader PRODUCER **and** CONSUMER); `c_1` is a plain reader→writer 1:1 (bind reader PRODUCER, writer CONSUMER). Config-independent.
- **`DefaultTilized`** — all CBs legal 1:1, in both configs:
  - `convert_df == false`: `c_0` = reader PRODUCER + writer CONSUMER (no compute kernel; output index aliases `c_0`).
  - `convert_df == true`: `c_0` = reader PRODUCER + compute CONSUMER; `c_16` = compute PRODUCER + writer CONSUMER.

  No multi-binding, no dead CBs.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no split-reader, no semaphore-gated raw co-fill in either factory. The only non-1:1 disposition is `DefaultRowMajor`'s single-toucher `c_0` self-loop.
- **Cross-op / shared kernels:**
  - `DefaultRowMajor` — owns both kernels; no file-path borrow. Function-call escape to `data_movement/common/kernels/common.hpp` (`tt_memmove(Noc, …)`, Device 2.0 native) bridges cleanly — no fork.
  - `DefaultTilized` — **file-path-instantiates three shared kernels, no `_metal2` fork exists yet — this port creates the first fork of each** (beside the original; do not convert in place):
    - `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
    - `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**very broadly shared** — ~20+ ops)
    - `data_movement/sharded/device/kernels/compute/eltwise_copy.cpp`

    Other binding ops for each are listed in `METAL2_PREPORT_AUDIT.md` → Heads-ups → Cross-op / shared kernels — that is a **sunset list, not authorization to convert the kernel in place**.
- **RTA varargs:** none — prefer named RTAs for every arg (all are fixed, nameable fields).

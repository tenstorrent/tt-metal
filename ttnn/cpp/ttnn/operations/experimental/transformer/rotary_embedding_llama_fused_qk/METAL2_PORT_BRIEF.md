# Metal 2.0 Port Brief — `experimental/transformer/rotary_embedding_llama_fused_qk`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A — no accessor)

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — single `create_descriptor` returning a `ProgramDescriptor`; single program.
- **Op-owned tensors:** none — carried natively by the target concept anyway.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All `no` on this op.

**Shape note.** One factory, one `KernelDescriptor` (a compute kernel), selecting one of two **source files** at runtime by `row_major_QK` (tiled `rotary_embedding_llama_sharded.cpp` / row-major `rotary_embedding_llama_sharded_row_major.cpp`). Same CB layout and descriptor shape for both — port as one KernelSpec whose source path is the existing runtime `row_major_QK` selection. There are **no dataflow kernels**: all tensor I/O is borrowed-memory CBs.

## Construct — to do

**Tensor bindings** (per binding) — all seven are **clean / borrowed-memory DFB**. Each binds through the existing `CBDescriptor::buffer` today; port each to `DataflowBufferSpec::borrowed_from(<tensor::name>)`. No `TensorAccessor`, no `->address()` RTA, no `get_bank_base_address` bridge anywhere — nothing is Case 1 or Case 2.

- `q_input` → CB `c_0` (`buffer = q_src_buffer`) — borrowed
- `k_input` → CB `c_1` (`buffer = k_src_buffer`) — borrowed
- `cos` → CB `c_2` (`buffer = cos_buffer`) — borrowed, read-only
- `sin` → CB `c_3` (`buffer = sin_buffer`) — borrowed, read-only
- `trans_mat` → CB `c_4` (`buffer = trans_mat_buffer`) — borrowed, read-only
- `q_output` → CB `c_16` (`buffer = q_dst_buffer`) — borrowed
- `k_output` → CB `c_17` (`buffer = k_dst_buffer`) — borrowed

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor in the op.

**CB endpoints:** **self-loop every CB** — there is a single kernel, so each CB has exactly one toucher (bind the compute kernel as both PRODUCER and CONSUMER). This covers all ten:
- borrowed tensor CBs: `c_0`, `c_1`, `c_2`, `c_3`, `c_4`, `c_16`, `c_17` (7)
- local interm CBs: `c_24` (rotated_input_interm), `c_25` (cos_interm), `c_26` (sin_interm) (3)

The read-only `cos`/`sin`/`trans_mat` CBs (`c_2`/`c_3`/`c_4`) are consumed only (no `reserve_back`/`push_back` in the kernel) — a consumer-only binding is legal, but the single-toucher self-loop is the safe uniform choice. No dead-CB drop, no multi-binding flag, no 1P+1C assignment anywhere.

**Runtime args:** one named scalar. The kernel reads `is_q = get_arg_val<uint32_t>(0)` (both sources, line 29); the factory sets it per core (`program_factory.cpp:257-262`: `is_q_arg = 1` on q-cores, `is_k_arg = 0` on k-cores). Port as a single named runtime arg (e.g. `is_q`) in the `runtime_arg_schema` / `ProgramRunArgs` — not a vararg.

## Watch for

- **CB endpoints (multi-binding):** none — a single-kernel program cannot have a hidden second writer or a multi-reader; no flag to set.
- **Cross-op / shared kernels:** none — the op owns both kernel sources and instantiates them by file path from its own directory; includes are LLK-only (`api/compute/*`, `api/dataflow/circular_buffer.h`). No port-together coupling.
- **RTA varargs:** none — the sole RTA is the fixed, nameable `is_q` scalar.
- **Minor (not port work):** the tiled kernel declares three unused `CircularBuffer` objects (`cos_cb_obj`/`sin_cb_obj`/`trans_mat_cb_obj`, `rotary_embedding_llama_sharded.cpp:61-63`) — dead locals, leave as-is unless the whitelist otherwise touches those lines. The commented-out `has_work` early return (both kernels, lines 24-28) is a pre-existing TRISC2 code-size workaround — do not revive it as part of the port.

# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `156b384a2cf 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports` *(carry this line into the port report's Provenance section)*

**Porting unit:** one `RotaryEmbeddingLlamaDeviceOperation` with three `descriptor` factories sharing kernels — port together:
- `RotaryEmbeddingLlamaMultiCore` (interleaved prefill) — reader + writer + compute
- `RotaryEmbeddingLlamaMultiCorePrefillSharded` (prefill, sharded cos/sin/trans_mat) — reader + writer + compute
- `RotaryEmbeddingLlamaMultiCoreSharded` (decode) — compute only

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (all three factories)
- **Op-owned tensors:** none
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors)
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind (`Is safe to port? == yes`). All `no` on this op.

## Construct — to do

**Tensor bindings** (per binding — classification varies by factory/config; the four inputs are `input`, `cos`, `sin`, `trans_mat`, plus `output`):

- **`RotaryEmbeddingLlamaMultiCore` (interleaved):** `input`, `cos`, `sin`, `trans_mat`, `output` — all **Case 1** (via `TensorAccessor`). Today the factory pushes raw `Buffer*` into runtime args (`multi_core:337-338`) and appends `TensorAccessorArgs` to CTAs (`:228-231,240`); the reader/writer build `TensorAccessor(args, addr)` from a `get_arg_val` base. Express each as a `TensorParameter`/`TensorBinding`, switch the kernels to `TensorAccessor(tensor::name)`, and delete both the `Buffer*` RTAs and the `TensorAccessorArgs` CTA plumbing.
- **`RotaryEmbeddingLlamaMultiCoreSharded` (decode):** `input`, `cos`, `sin`, `trans_mat`, `output` — all **clean (borrowed-DFB)**. Each binds via `CBDescriptor::buffer` (`sharded:87,99,111,125,171`); express with `DataflowBufferSpec::borrowed_from`. No `TensorAccessor`, no RTAs in this factory.
- **`RotaryEmbeddingLlamaMultiCorePrefillSharded` (hybrid):** `input` → **Case 1**; `output` → **Case 1**; `cos`/`sin` → **clean (borrowed-DFB)** on the sharded fast path (`.buffer`, `:175,186`), **Case 1** on the reload/interleaved path; `trans_mat` → **clean (borrowed-DFB)** on the global-CB path (`.buffer`, `:260`), **Case 1** otherwise. The binding for `cos`/`sin`/`trans_mat` must support **both** shapes selected by the same config flags the factory already computes (`cos_sin_sharded`, `cos_sin_sharded_reload`, `trans_mat_use_global_cb`).

No **Case 2** (raw-pointer) bindings anywhere — do not reach for the `get_bank_base_address` bridge.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor passes a 3rd argument; nothing to drop.

**CB endpoints:**
- Self-loop the interm CBs `c_24` (rotated), `c_25` (cos-interm), `c_26` (sin-interm) — single toucher (compute) in factories 1 & 2.
- Self-loop `c_27` (zero) — single toucher (writer) in factories 1 & 2.
- **Decode factory:** self-loop **every** CB — the lone compute kernel is the only toucher of all eight, producing and consuming each.
- Everything else is legal 1:1 (reader→compute for inputs `c_0`/`c_1`/`c_2`/`c_3`; compute→writer for output `c_16`). No 1P+1C assignments, no multi-binding advanced option, no dead-CB drops.

**Prefill-sharded merged CBs (structural, preserve verbatim):** the prefill-sharded factory emits **multiple `CBDescriptor`s sharing one `buffer_index`** over disjoint core ranges — `c_1`/`c_2` split shard-grid vs remaining cores (`prefill_sharded:167-211`) and `c_3` likewise (`:247-288`). Keep the per-core-range split when expressing these DFBs; do not collapse them to a single range.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader, no ≥3-toucher CB. All non-1:1 CBs are single-toucher self-loops; no flag needed.
- **Cross-op / shared kernels:** none — the op owns all five kernels; every `#include` is `api/*` (tt_metal LLK/HAL). No `_metal2` fork exists or is needed; no sunset list.
- **RTA varargs:** none — every kernel reads a fixed run of named args via a top-of-kernel `argrt++` counter (reader 8, writer 5, compute 4). Name each; do **not** use the vararg mechanism.
- **Shared kernels across factories:** `writer_rotary_embedding_llama_interleaved_start_id.cpp` and `compute/rotary_embedding_llama.cpp` are each used by **both** factory 1 and factory 2 — one port of each kernel serves both. The decode factory uses its own `compute/rotary_embedding_llama_sharded.cpp`.

# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk`

- **`RotaryEmbeddingLlamaFusedQKDeviceOperation`**
  - `RotaryEmbeddingLlamaFusedQKProgramFactory` (`device/rotary_embedding_llama_fused_qk_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk` |
| **Overall** | GREEN |
| **DOps / Factories** | `RotaryEmbeddingLlamaFusedQKDeviceOperation` → `RotaryEmbeddingLlamaFusedQKProgramFactory` (single factory) |
| *Prereqs* — ProgramDescriptor | Yes (factory returns `tt::tt_metal::ProgramDescriptor`) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes (`CircularBuffer`; compute-only, LLK compute APIs) |
| *Prereqs* — Cross-op escapes | Ok (both compute kernels are op-owned; no external kernel files) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (no CTA varargs; one per-core RTA `is_q`) |
| *TTNN Readiness* — Op-owned tensors | No (5 inputs + 2 outputs declared; 3 scratch CBs are program-local, not tensors) |
| *TTNN Readiness* — MeshWorkload needed | No (single-program; ProgramDescriptor) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind binds only the user-facing op entry) |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No (default reflection hash) |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | Present: 7 borrowed-memory CBs — q_in/k_in/cos/sin/trans_mat (read-only address sources) + q_out/k_out (write-only address sinks) (workaround) |

## Result

**GREEN → brief issued.** Single compute-only program on the ProgramDescriptor API with a Device-2.0 kernel; no op-owned device resources. Target concept: `ProgramSpecFactoryConcept`.

## Gate detail

- **ProgramDescriptor:** GREEN — factory populates a `ProgramDescriptor` (`desc.cbs`, `desc.kernels`, `KernelDescriptor`, `CBDescriptor`).
- **Device 2.0:** GREEN — both compute kernels (`rotary_embedding_llama_sharded.cpp`, `rotary_embedding_llama_sharded_row_major.cpp`) use the kernel-side `CircularBuffer` wrapper + LLK compute APIs exclusively; no Device-1.0 idioms.
- **Feature compatibility:** all Appendix A entries N/A except: borrowed-memory CB (LANDED → `borrowed_from`); fake CB (address-only, workaround); compute-kernel self-loop DFB (LANDED → `dfb_self_loop_connectivities = INTRA`). No UNSUPPORTED features in use.

## Port-work summary

- **Tensor bindings:** none in the Case-1/Case-2 sense — the compute kernel builds no `TensorAccessor` and reads no base address. The 7 backing tensors (q_in, k_in, cos, sin, trans_mat, q_out, k_out) are declared as `TensorParameter`s ONLY to resolve their borrowed DFB addresses (`borrowed_from`); none is bound as a `TensorBinding`.
- **Custom hash:** none.

## Heads-ups

- **Compute-only op (no reader/writer).** All work is in a single compute KernelDescriptor; the host selects the `.source` (`row_major_QK` ? row-major : tile). Both sources have identical CTA layout and CB bindings.
- **Notable LANDED constructs:** 7 CBs are **borrowed-memory** built on tensor `.buffer()` (c_0 q_in, c_1 k_in, c_2 cos, c_3 sin, c_4 trans_mat, c_16 q_out, c_17 k_out) → port uses `DataflowBufferSpec::borrowed_from`. 3 CBs (c_24/c_25/c_26 interm) are program-local scratch (no `.buffer`).
- **Fake CBs (address-only / one-ended).** Since the op is compute-only and the data is resident, the 7 borrowed CBs have no separate producer/consumer kernel — only the compute kernel touches them. cos/sin/trans_mat are read purely as LLK operands; q_in/k_in/q_out/k_out carry FIFO ops but against resident memory. From the validator's view all are one-ended → bind each as a self-loop (PRODUCER+CONSUMER) on the compute kernel + `dfb_self_loop_connectivities[DFB]=INTRA`. The 3 interm CBs are real intra-kernel FIFOs and are also self-loops on compute. Recorded as interim workarounds in the port report.
- **Runtime-dynamic CB selection.** The kernel chooses in_cb/out_cb/Ht at runtime from the per-core RTA `is_q` (q vs k). Both candidate CBs are bound on the same compute kernel; the kernel uses `uint32_t in_cb = is_q ? (uint32_t)dfb::q_in : (uint32_t)dfb::k_in;` (whitelist runtime-dynamic CB rule).
- **TTNN factory analysis (porter-relevant):** no pybind `create_descriptor`, no other risky pybind, no custom `override_runtime_arguments`, no `select_program_factory` (single factory).

## Team-only

- **Borrowed/fake-CB self-loops:** the 7 borrowed DFBs are validator-satisfying self-loops, not real FIFOs (5 read-only tensor-local-view inputs, 2 write-only sinks). Candidates for the forthcoming local-`TensorAccessor` / scratchpad migration.
- **Out-of-directory coupling:** none. Both kernels are op-owned; no donor `#include`s outside `api/*`.
- **TTNN factory analysis (6 Qs):** (1) op-owned tensors: No. (2) MeshWorkload: No. (3) pybind create_descriptor: No. (4) other risky pybind: No. (5) custom hash: No. (6) custom override-RTA: No.

# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode`

- **`NLPConcatHeadsDecodeDeviceOperation`**
  - `NLPConcatHeadsDecodeProgramFactory` (`device/nlp_concat_heads_decode_program_factory.cpp`)
  - `NLPConcatHeadsDecodeSubcoregridsProgramFactory` (`device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode` |
| **Overall** | GREEN |
| **DOps / Factories** | `NLPConcatHeadsDecodeDeviceOperation` → `NLPConcatHeadsDecodeProgramFactory`, `NLPConcatHeadsDecodeSubcoregridsProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes (both factories return `tt::tt_metal::ProgramDescriptor`) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes (`Noc`, `CircularBuffer`, `UnicastEndpoint`, `CoreLocalMem`) |
| *Prereqs* — Cross-op escapes | Ok (both kernels are op-owned; no external kernel files) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (no CTA varargs; RTA-array of NoC coords only) |
| *TTNN Readiness* — Op-owned tensors | No (only declared input + output) |
| *TTNN Readiness* — MeshWorkload needed | No (single-program; ProgramDescriptor, not MeshWorkload) |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No (default reflection hash) |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | Present: output CB `c_16` (`(c_16, write)`) — borrowed-memory, write-only address source (workaround) |

## Result

**GREEN → brief issued.** Both factories are single-program on the ProgramDescriptor API with Device-2.0 kernels; no op-owned device resources. Target concept: `ProgramSpecFactoryConcept`.

## Gate detail

- **ProgramDescriptor:** GREEN — both factories populate a `ProgramDescriptor` (`desc.cbs`, `desc.kernels`, `KernelDescriptor`, `CBDescriptor`).
- **Device 2.0:** GREEN — kernels (`reader_tm_tile_layout_nlp_concat_heads_decode.cpp`, `..._subcoregrid.cpp`) use `Noc`, `CircularBuffer`, `UnicastEndpoint`, `CoreLocalMem` exclusively; no Device-1.0 idioms.
- **Feature compatibility:** all Appendix A entries N/A except: borrowed-memory CB (LANDED → `borrowed_from`); fake CB (address-only, workaround). No UNSUPPORTED features in use.

## Port-work summary

- **Tensor bindings** (per binding):
  - `input` — **Case 2 (bridge)**. The input shard L1 base address is currently passed as a raw `Buffer*` RTA (`rt_args.push_back(in_buffer)`, factory line 130/137) and consumed kernel-side as `q_start_addr`; the kernel then performs an **exotic cross-core NoC walk** (explicit `{.noc_x, .noc_y, .addr}` unicast reads at sub-tile / face granularity, `SUBTILE_LINE_BYTES`), which `TensorAccessor` page iteration does not express. Re-express as a `TensorParameter`; recover the base kernel-side via the sanctioned `TensorAccessor::get_bank_base_address()` bridge, leaving the existing NoC arithmetic intact.

    > Per the audit's Case-2 rule, surfacing the verbatim user-facing note: *The use of `TensorAccessor` is an ergonomic choice on Gen1 architectures. It has meaningful performance implications on Gen2 architectures. Ideally, `TensorAccessor` should be updated to support the required iteration pattern; consider filing an issue requesting that support.* (The user explicitly directed this port; proceeding with the bridge.)
  - `output` — borrowed-memory backing for the output DFB (see Heads-ups); not a TensorAccessor read.
- **Custom hash:** none.

## Heads-ups

- **Notable LANDED constructs:** Output CB `c_16` is a **borrowed-memory** CB built on `output.buffer()` (factory line ~54/64) → port uses `DataflowBufferSpec::borrowed_from = OUTPUT`.
- **Fake CBs (address-only):** Output CB `c_16` has a producer (the kernel writes into it via `get_write_ptr()`) but **no on-device consumer** (it *is* the result). It carries no FIFO semantics (no `reserve_back`/`push_back`/`wait_front`). → tensor-local-view fake CB; port satisfies the validator's producer+consumer rule by binding `reader=PRODUCER` / `writer=CONSUMER` of the same DFB (the two kernels run on the same nodes). Recorded as an interim workaround in the port report.
- **RTA varargs:** Both kernels read NoC coordinate arrays via `get_arg_addr(2)` / `get_arg_addr(2+N)` indexed by a runtime loop variable (`in0_mcast_noc_x[qkv_x]`). Genuine vararg case → port uses Metal 2.0 common runtime varargs (`get_common_vararg`), since the arrays are identical across all output cores.
- **TTNN factory analysis (porter-relevant):** no pybind `create_descriptor`, no other risky pybind, no custom `override_runtime_arguments`.

## Team-only

- **TensorAccessor convertibility (`input`, Case 2):** awkward-but-potentially-convertible — a `TensorAccessor` enhancement supporting per-core sub-tile-line gather could let this become Case 1. Candidate for a future TensorAccessor-iteration issue. Default: bridge.
- **Out-of-directory coupling:** none. Both kernels are op-owned; no donor `#include`s outside `tt_metal/*`.
- **TTNN factory analysis (6 Qs):** (1) op-owned tensors: No. (2) MeshWorkload: No. (3) pybind create_descriptor: No. (4) other risky pybind: No. (5) custom hash: No. (6) custom override-RTA: No.

## Per-DeviceOperation attribution

Both factories are structurally identical (borrowed-memory output CB, `Buffer*` input RTA + NoC-coord RTA arrays, two kernel instances from one source distinguished by `PHASES_TO_READ`); findings apply equally to each. Subcoregrid variant differs only in tile/face geometry handling and a flat single-axis core list.

# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/topk`

- **`TopKDeviceOperation`** (`ttnn::prim`)
  - `TopKSingleCoreProgramFactory` (`device/topk_single_core_program_factory.cpp`)
  - `TopKMultiCoreProgramFactory` (`device/topk_multi_core_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/topk` |
| **Overall** | GREEN for single-core; AMBER (deferred) for multi-core |
| **DOps / Factories** | `TopKDeviceOperation` → `TopKSingleCoreProgramFactory`, `TopKMultiCoreProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes (both factories return `tt::tt_metal::ProgramDescriptor`) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes (`Noc`, `CircularBuffer`, `Semaphore`, `UnicastEndpoint`, `TensorAccessor`) |
| *Prereqs* — Cross-op escapes | Ok (all kernels are op-owned, under `topk/device/kernels/`) |
| *Feature Support* — overall | GREEN (single-core); cross-core remote-CB write (multi-core) has no documented pattern |
| *Feature Support* — Variadic-CTA | Ok (no CTA varargs) |
| *TTNN Readiness* — Op-owned tensors | No (input + optional indices + 2 outputs only) |
| *TTNN Readiness* — MeshWorkload needed | No (single-program ProgramDescriptor) |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Custom hash | No (default reflection hash) |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None in single-core. Multi-core: `c_4`/`c_5` written by a remote core via NoC + `get_write_ptr()` base (see grounded stop). |

## Result

**Single-core: GREEN → ported.** Eight normal CBs (`c_0..c_7`), clean Case-1 `TensorAccessor` reads/writes, Device-2.0 kernels. Target concept: `ProgramSpecFactoryConcept` (`create_program_spec`).

**Multi-core: deferred on legacy `create_descriptor`.** Genuine grounded stop (below). The variant runs single-core on Metal 2.0 and multi-core on legacy; `select_program_factory` is unchanged.

## Gate detail

- **ProgramDescriptor:** GREEN — both factories populate a `ProgramDescriptor`.
- **Device 2.0:** GREEN — all kernels use `Noc`, `CircularBuffer`, `Semaphore`, `UnicastEndpoint`, `TensorAccessor`; no Device-1.0 idioms.
- **Feature compatibility (single-core):** all Appendix A entries N/A except the compute kernel's **runtime-dynamic CB selection** (`uint32_t cb0..cb3` reassigned per case) — LANDED via the documented `(uint32_t)dfb::name` + `DataflowBuffer obj(cbX)` pattern. Self-loop staging CBs (`c_2..c_5`) are real producer+consumer loops on the compute kernel.
- **Feature compatibility (multi-core):** UNSUPPORTED — see grounded stop.

## Port-work summary (single-core)

- **Tensor bindings** — all **Case 1** (page-by-page `TensorAccessor`):
  - `input` (reader) — `ta::input`, read by `{.page_id = row * Wt + w}`.
  - `value`, `index` (writer) — `ta::value` / `ta::index`, written by `{.page_id = row * Kt + k}`.
  - `indices` (optional precomputed) — only referenced on the dead `#if not GENERATE_INDICES` reader branch (`GENERATE_INDICES` is hardcoded `"1"`, GH #36329). **Not bound** on the always-on path; kept as a `compiler_options.defines` flag + `#ifdef`-gated `ta::indices`.
- **Custom hash:** none.
- **Pybind `create_descriptor`:** none.

## Grounded stop — multi-core factory (deferred)

The multi-core factory (`reader_create_index_local_topk`, `reader_final_topk`, `writer_local_topk`, `writer_final_topk`, `topk_local`, `topk_final`) does **not** fit the documented Metal 2.0 patterns:

1. **Cross-core remote-CB write with no documented pattern.** `writer_local_topk` (running on local cores) computes the *final* core's CB write pointer locally (`final_values_cb.get_write_ptr()`, CB allocated on `all_cores` for address consistency) and NoC-writes into it at `{.noc_x = noc_final, .noc_y = noc_final, .addr = base + ...}`. The same buffers (`c_4`/`c_5`) are also `reserve_back`/`push_back` PRODUCED by `reader_final_topk` on the final core. This is multiple producers across **disjoint node sets** writing one DFB by raw remote address — the DFB endpoint invariant (one producer / one consumer per node, derived placement) cannot express it, and no pattern in `metal2_port_patterns.md` covers a remote-core CB-address write.
2. **Custom semaphore-multicast flow control.** `reader_final_topk` broadcasts `receiver_sem.set_multicast(...)` to the local-core range and waits on `sender_sem.wait(Wt_final)`; `writer_local_topk` does `sender_sem.up(...)` after each NoC write. While `SemaphoreSpec` exists, the producer/consumer cross-core handshake is coupled to the remote-CB write above.
3. **Allocation-order-sensitive L1 layout.** The factory relies on a specific CB allocation order across two core sets (documented comment) so that `get_write_ptr()` on the final core yields an address the local cores can target. Metal 2.0 DFB placement is derived, not order-pinned.

Per the orchestrator brief, the variant supports mixed concepts; the multi-core remainder is documented as "remaining work" and left on legacy `create_descriptor`.

## Per-DeviceOperation attribution

`select_program_factory` chooses multi-core only for large power-of-two dims with K≤64 and UInt16 indices that pass `verify_multi_core_cost`; otherwise single-core. The ported single-core path covers the default and fallback cases.

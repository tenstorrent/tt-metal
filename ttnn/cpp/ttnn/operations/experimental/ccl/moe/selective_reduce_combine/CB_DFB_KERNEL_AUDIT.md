# CB→DFB Kernel Audit: `experimental/ccl/moe/selective_reduce_combine`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/`

**Scope:** `device/kernels/dataflow/reader.cpp`, `device/kernels/dataflow/writer.cpp`.

## OUT-OF-SCOPE check

This op is under a `moe/` path, so it was screened against the recipe's OUT-OF-SCOPE MOE patterns (`deepseek_moe_gate`, `generalized_moe_gate`, `deepseek_prefill` combine/dispatch/post_combine_reduce — firmware-style `reconfig_cbs_for_mask` / `get_cb_tiles_*_ptr` reinit). It does **not** match: there is **no** `get_local_cb_interface` reconfig, **no** `get_cb_tiles_acked_ptr`/`get_cb_tiles_received_ptr`, and no firmware-style CB reinit. It is a normal **selective reduce/combine dataflow op** (fabric NoC + semaphore token movement), so it is **audited normally** (not marked OUT-OF-SCOPE).

## Overall verdict: GREEN

**Summary:** Pure dataflow reader/writer op. All CBs are canonical Class 1 linear FIFOs / bare-pointer L1 staging for token maps, per-expert token counts, activations, and fabric packet headers. All six Step-4 litmus scans (GATE, silent-wrong, QUASAR-BLOCKED, LTA, ptr-surgery, field-reads) return **zero hits** in scope. Mechanical `CircularBuffer` → `DataflowBuffer` on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `data_cb`, `token_activations_cb` | 1 | `reader.cpp`, `writer.cpp` | Portable | token activation data stream, canonical FIFO / `get_read_ptr()`+`get_write_ptr()` as NoC addresses | Portable | — |
| `dense_token_maps_cb`, `token_counts_cb` | 1 | `reader.cpp`, `writer.cpp` | Portable | routing metadata staging, linear FIFO | Portable | — |
| `packet_header_cb` | 1/6 | `writer.cpp` | Portable | fabric packet-header scratch region (reserved-region L1); mechanical rename (or `ScratchpadSpec` as optional non-gating hardening) | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely on both arches — canonical fabric dataflow with no field surgery, no runtime API dependency, no LTA prerequisite. Packet-header CB could optionally move to `ScratchpadSpec` later, but this is not port-gating.

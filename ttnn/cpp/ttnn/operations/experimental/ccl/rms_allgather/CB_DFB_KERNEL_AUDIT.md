# CB→DFB Kernel Audit: `experimental/ccl/rms_allgather`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/`

**Scope:** `device/kernels/compute/rms_compute.cpp`, `device/kernels/dataflow/rms_sender_reader.cpp`, `rms_receiver_reader.cpp`, `rms_writer.cpp`, `reshard_writer.hpp`.

## Overall verdict: GREEN

**Summary:** Fused RMSNorm + all-gather op: a Welford-style RMS compute kernel plus sender/receiver reader and writer dataflow kernels. All CBs are canonical Class 1 linear FIFOs — normalization pipeline stages (`cb_x`, `cb_xmm`, `cb_var`, `cb_im`, `cb_ex_partial2`, `cb_ex2`, `cb_ex_external2`, `cb_ex_global`, `cb_stats*`), inputs/residual (`cb_in0`, `cb_in1`, `cb_stats`), scalars/eps/gamma (`cb_scaler*`, `cb_eps`, `cb_gamma`), and outputs (`cb_out`, `cb_to_allgather_writer`, `cb_out_resharded`) plus a fabric `reserved_packet_header_cb` and `signaling_cb`. All six Step-4 litmus scans return **zero hits** — notably **no** `get_pointer_to_cb_data` (unlike the layernorm/RMS Welford reciprocal-LUT path) and **no** `read_tile_value`/`get_tile_address`. Mechanical `CircularBuffer` → `DataflowBuffer` on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0`, `cb_in1`, `cb_in`, `cb_stats` | 1 | `rms_compute.cpp`, `rms_*_reader.cpp` | Portable | input / residual / pre-add input / stats tensor, linear FIFO | Portable | — |
| `cb_x`, `cb_xmm`, `cb_im`, `cb_var`, `cb_x2` | 1 | `rms_compute.cpp` | Portable | RMS normalize pipeline intermediates, canonical FIFO | Portable | — |
| `cb_ex_partial2`, `cb_ex2`, `cb_ex_external2`, `cb_ex_global`, `cb_stats_reduced` | 1 | `rms_compute.cpp`, `rms_sender_reader.cpp`, `rms_receiver_reader.cpp` | Portable | partial/global variance reduce + multicast stats, canonical FIFO + `noc_semaphore` | Portable | — |
| `cb_scaler`, `cb_scaler_global`, `post_cb_scaler_global`, `cb_eps`, `eps_cb_id`, `cb_gamma` | 1 | `rms_compute.cpp`, `rms_writer.cpp` | Portable | reduction scalars / epsilon / gamma weights, linear FIFO | Portable | — |
| `cb_out`, `cb_to_allgather_writer`, `cb_out_resharded`, `post_cb_in_4`, `cb_in_2`, `cb_in_4` | 1 | `rms_compute.cpp`, `rms_writer.cpp`, `reshard_writer.hpp` | Portable | normalized output → resharder / all-gather writer, canonical FIFO | Portable | — |
| `reserved_packet_header_cb`, `signaling_cb` | 1/6 | `rms_writer.cpp`, `rms_compute.cpp` | Portable | fabric packet-header scratch + signaling handshake; mechanical rename (optional `ScratchpadSpec` hardening) | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely on both arches — canonical RMSNorm compute + all-gather dataflow FIFOs, no field surgery, no `get_pointer_to_cb_data`/`read_tile_value` dependency, no LTA prerequisite.

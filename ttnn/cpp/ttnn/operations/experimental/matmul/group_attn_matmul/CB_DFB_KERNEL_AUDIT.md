# CB→DFB Kernel Audit: `group_attn_matmul`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/`

**Scope:** `group_attn_matmul_program_factory.cpp` → kernels: `dataflow/reader_mcast_transformer_group_attn_matmul.cpp`, `dataflow/writer_transformer_group_attn_matmul.cpp`, `compute/transformer_group_attn_matmul.cpp`.

## Overall verdict: GREEN

**Summary:** All CBs are canonical Class 1 linear FIFOs via the modern `CircularBuffer` object API. Step-4 litmus scans return **zero** hits across the reader (mcast), writer, and compute kernels — no GATE, no silent-wrong, no ptr surgery, no field reads. Reader uses `noc_semaphore` mcast sync alongside canonical FIFO credits. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0`, `cb_in1` | 1 | `reader_mcast_*.cpp`, `transformer_group_attn_matmul.cpp` | Portable | activation + mcast KV heads, canonical FIFO | Portable | — |
| `cb_in2` | 1 | `reader_mcast_transformer_group_attn_matmul.cpp` | Portable | interleaved/sharded KV heads for one batch, linear FIFO | Portable | — |
| `cb_intermed0`, `cb_intermed1` | 1 | `writer_transformer_group_attn_matmul.cpp`, `transformer_group_attn_matmul.cpp` | Portable | transpose/matmul intermediates | Portable | — |
| `cb_out` (output) | 1 | `writer_transformer_group_attn_matmul.cpp`, `transformer_group_attn_matmul.cpp` | Portable | pack → output, `get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

# CB→DFB Kernel Audit: `nlp_create_qkv_heads_vit`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/`

**Scope:** `nlp_create_qkv_heads_vit` program factory → kernels: `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp`, `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp`. No compute kernel. Shared includes are all standard API headers — no kernel_lib/donor headers.

## Overall verdict: GREEN

**Summary:** Pure dataflow QKV head-split op (ViT variant of the segformer path). Reader fills `cb_qv` (and, per compile-time layout, `cb_k`) with canonical `reserve_back`/`get_write_ptr()`/`push_back`; writer drains with `wait_front`/`get_read_ptr()`/`noc.async_write`/`pop_front`. All CBs are Class 1 linear FIFO on the `CircularBuffer` object API. Zero GATE / silent-wrong / ptr-surgery / runtime-blocker hits. Mechanical `CircularBuffer` → `DataflowBuffer` rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_qv` (Q/V heads) | 1 | `reader_tm_tile_layout_nlp_create_qkv_heads.cpp`, `writer_tm_tile_layout_nlp_create_qkv_heads.cpp` | Portable | linear FIFO; `get_write_ptr()`/`get_read_ptr()` as L1/NoC addr only | Portable | — |
| `cb_k` (K heads) | 1 | `reader_tm_tile_layout_nlp_create_qkv_heads.cpp`, `writer_tm_tile_layout_nlp_create_qkv_heads.cpp` | Portable | K path; `#ifndef TRANSPOSE_K_HEADS` aliases `cb_qv` (id 1), else consumes a transpose CB (id 16) — linear FIFO either way | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

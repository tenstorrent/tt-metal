# CB→DFB Kernel Audit: `reduction` family

**Date:** 2026-07-13
**Op root:** `ttnn/cpp/ttnn/operations/reduction/`

**Scope:** Device kernels for all reduction sub-ops: `accumulation/{cumprod,cumsum,ema}`, `argmax`, `generic` (reduce), `manual_seed`, `moe`, `prod`, `sampling`, `topk`. Host `packer_l1_acc` config destructuring in program factories is **not** a kernel field access — excluded.

## Overall verdict: YELLOW

**Summary:** No hard blockers. Nearly every reduction op is canonical linear-FIFO (Class 1) and mechanically portable. Two exceptions, both non-fatal: `generic` reduce has one mechanical getter swap (`fifo_page_size`), and `manual_seed` uses `read_tile_value()` on a CB → **2xx-blocked** until the Quasar DFB read API lands (1xx clear).

## Per-op rollup

| Sub-op | Rollup | Driver |
|--------|--------|--------|
| `accumulation/cumprod` | GREEN | Canonical FIFO only |
| `accumulation/cumsum` | GREEN | Canonical FIFO only |
| `accumulation/ema` | GREEN | Canonical FIFO only (`packer_l1_acc` is host config) |
| `argmax` | GREEN | Canonical FIFO only |
| `generic` (reduce) | YELLOW | 1 mechanical getter swap (`fifo_page_size`) |
| `manual_seed` | YELLOW | `read_tile_value` → 2xx runtime-blocked |
| `moe` (topk-moe) | GREEN | Canonical FIFO; **not** the OUT-OF-SCOPE moe_gate |
| `prod` | GREEN | Canonical FIFO only |
| `sampling` | GREEN | Canonical FIFO only |
| `topk` | GREEN | Canonical `push_back`/`pop_front` (fifo_* only in a comment) |

## CB portability (notable rows only; unlisted CBs are Class 1 → Portable both arches)

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_rm` | 1 | `generic/.../dataflow/reader_unary_reduce_rm.cpp` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_rm).fifo_page_size` (line 82) → `get_entry_size()` (getter exists today) | Portable | same |
| `kernel_communication_cb` | 1 | `manual_seed/.../compute/manual_seed_receive_all_data.cpp` (27,32), `manual_seed_single_seed_receive_user_id.cpp` (28) | Portable (workaround) | `read_tile_value()` works on 1xx | Blocked (runtime) | needs `read_tile_value` on Quasar DFB (in progress, Runtime team) |

## GATE hits (must be empty to merge)

- `generic/device/kernels/dataflow/reader_unary_reduce_rm.cpp:82` — `get_local_cb_interface(cb_id_rm).fifo_page_size` read — **mechanical**, → `get_entry_size()`. Clears with a trivial swap.

## Blocked on runtime (2xx rollup)

- `read_tile_value` on Quasar DFB → `manual_seed` (`kernel_communication_cb`). In progress with the Runtime team; matches the doc's in-scope QUASAR-BLOCKED list. 1xx path is clear.

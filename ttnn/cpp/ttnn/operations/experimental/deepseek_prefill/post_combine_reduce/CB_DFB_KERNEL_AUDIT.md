# CB→DFB Kernel Audit: `deepseek_prefill/post_combine_reduce`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/`

**Scope:** `device/kernels/{deepseek_moe_post_combine_reduce_reader.cpp, deepseek_moe_post_combine_reduce_writer.cpp, deepseek_moe_post_combine_reduce_compute.cpp}`

## Overall verdict: OUT-OF-SCOPE

**Summary:** Per `CB_DFB_Check.md` [Audit scope → Path exclusions](../../../../../../../CB_DFB_Check.md#audit-scope-device-kernels-only), `deepseek_prefill/post_combine_reduce` is a DeepSeek-prefill MOE-routing / firmware-style op that is **OUT-OF-SCOPE** for CB→DFB port-gating. It is **not** rolled up into GREEN/YELLOW/RED and does **not** gate any other op's port. It must be tracked separately (firmware-style reinit + expert-routing redesign).

## Signals found (recorded, not gated)

| File:line | Signal | Notes |
|-----------|--------|-------|
| `deepseek_moe_post_combine_reduce_compute.cpp:36` | `get_local_cb_interface(...).fifo_rd_ptr` **read** | Base L1 address for a manual `read_tile_value_uint16` helper (Gen1 path); would be a GATE for an in-scope op. |
| `deepseek_moe_post_combine_reduce_compute.cpp:37` | `get_local_cb_interface(...).fifo_page_size` **read** | Per-tile stride for the same manual helper (getter `get_entry_size()` exists, but op is tracked separately). |
| `deepseek_moe_post_combine_reduce_compute.cpp:31,98` | `read_tile_value_uint16` (custom helper) | uint16 element read (`cb_indices_id`) — needs typed Quasar DFB read API. |
| `deepseek_moe_post_combine_reduce_compute.cpp:100,112` | `read_tile_value` | Expert→chip / weight lookups (`cb_dispatch_table_id`, `cb_weights_id`) — QUASAR-BLOCKED. |
| `deepseek_moe_post_combine_reduce_writer.cpp:116` | `read_tile_value_uint16` (comment only) | Explanatory comment — not a code hit; excluded. |

## GATE hits (must be empty to merge)

- N/A — OUT-OF-SCOPE. The `get_local_cb_interface(...).fifo_rd_ptr` / `fifo_page_size` reads (`deepseek_moe_post_combine_reduce_compute.cpp:36,37`) *would* be GATE + NEEDS-FIX for an in-scope op, but this op is tracked under the separate DeepSeek-prefill workstream and does not gate other ports.

## Blocked on runtime (2xx rollup)

- N/A — OUT-OF-SCOPE. `read_tile_value` / typed `read_tile_value_uint16` need the Quasar DFB (typed) read API, tracked separately (expert routing / firmware-style reinit), not under this audit's gate.

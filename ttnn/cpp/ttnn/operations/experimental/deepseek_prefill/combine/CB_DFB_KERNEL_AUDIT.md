# CB→DFB Kernel Audit: `deepseek_prefill/combine`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/`

**Scope:** `device/kernels/dataflow/{reader_combine.cpp, writer_combine.cpp, reader_untilize.cpp, writer_untilize.cpp, zero_init_common.hpp}`, `device/kernels/compute/untilize_combine.cpp`

## Overall verdict: OUT-OF-SCOPE

**Summary:** Per `CB_DFB_Check.md` [Audit scope → Path exclusions](../../../../../../../CB_DFB_Check.md#audit-scope-device-kernels-only), `deepseek_prefill/combine` is a DeepSeek-prefill MOE-routing / firmware-style op that is **OUT-OF-SCOPE** for CB→DFB port-gating. It is **not** rolled up into GREEN/YELLOW/RED and does **not** gate any other op's port. It must be tracked separately (firmware-style reinit + expert-routing redesign).

## Signals found (recorded, not gated)

| File:line | Signal | Notes |
|-----------|--------|-------|
| `compute/untilize_combine.cpp:91` | `read_tile_value` | Real UNPACK-side L1 read of per-expert token counts (`cb_experts_tok_counter_id`); QUASAR-BLOCKED until DFB read API lands — but this op is tracked separately. (`:86` is a comment.) |
| `dataflow/reader_untilize.cpp:202` | `fifo_wr_ptr` / `fifo_limit` (comment only) | Explanatory comment about tt-metal CB fill semantics — not a code hit; excluded. |

## GATE hits (must be empty to merge)

- N/A — OUT-OF-SCOPE. No `get_local_cb_interface(...).<field>` code hits. `read_tile_value` on Quasar is tracked under the separate DeepSeek-prefill workstream.

## Blocked on runtime (2xx rollup)

- N/A — OUT-OF-SCOPE. `read_tile_value` (`untilize_combine.cpp:91`) needs the Quasar DFB read API, but is tracked separately (expert routing / firmware-style reinit), not under this audit's gate.

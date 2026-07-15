# CB→DFB Kernel Audit: `deepseek_prefill/dispatch`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/`

**Scope:** `device/kernels/dataflow/{reader_dispatch.cpp, writer_dispatch.cpp, reader_untilize_dispatch.cpp, writer_untilize_dispatch.cpp, dispatch_plan.hpp}`, `device/kernels/compute/untilize_dispatch.cpp`

## Overall verdict: OUT-OF-SCOPE

**Summary:** Per `CB_DFB_Check.md` [Audit scope → Path exclusions](../../../../../../../CB_DFB_Check.md#audit-scope-device-kernels-only), `deepseek_prefill/dispatch` is a DeepSeek-prefill MOE-routing / firmware-style op that is **OUT-OF-SCOPE** for CB→DFB port-gating. It is **not** rolled up into GREEN/YELLOW/RED and does **not** gate any other op's port. It must be tracked separately (firmware-style reinit + expert-routing redesign).

## Signals found (recorded, not gated)

| File:line | Signal | Notes |
|-----------|--------|-------|
| `compute/untilize_dispatch.cpp:54` | `read_tile_value` | Real UNPACK-side L1 read of a control/signal value (`cb_signal_id`); QUASAR-BLOCKED until DFB read API lands — but this op is tracked separately. |

## GATE hits (must be empty to merge)

- N/A — OUT-OF-SCOPE. No `get_local_cb_interface(...).<field>` code hits. `read_tile_value` on Quasar is tracked under the separate DeepSeek-prefill workstream.

## Blocked on runtime (2xx rollup)

- N/A — OUT-OF-SCOPE. `read_tile_value` (`untilize_dispatch.cpp:54`) needs the Quasar DFB read API, but is tracked separately (expert routing / firmware-style reinit), not under this audit's gate.

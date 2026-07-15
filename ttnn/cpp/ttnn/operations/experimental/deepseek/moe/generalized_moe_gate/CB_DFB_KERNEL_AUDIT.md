# CB→DFB Kernel Audit: `generalized_moe_gate`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/generalized_moe_gate/`

**Scope:** In-scope device kernels under `device/kernels/` → `generalized_moe_gate_kernel.cpp` + its `device/unified_kernels/` include closure (`kernel_op_api.hpp`, `kernel_utils.hpp`, `generalized_moe_gate.hpp`).

## Overall verdict: OUT-OF-SCOPE

**Summary:** This op performs **firmware-style CB reconfiguration** and is explicitly excluded by the recipe (`**/generalized_moe_gate/` path exclusion + SILENT-WRONG special case). `kernel_utils.hpp` defines `reconfig_cbs_for_mask(...)` which, per its own comment, "sets fifo_rd_ptr, fifo_wr_ptr, fifo_size, fifo_limit, fifo_page_size, fifo_num_pages, ..." — i.e. it **rewrites the entire `LocalCBInterface` struct** and **writes the stream registers** via `get_cb_tiles_received_ptr(cb)`/`get_cb_tiles_acked_ptr(cb)`. On Quasar these fields do not exist / have different semantics, so this is **SILENT-WRONG**, not a mechanical getter swap. This needs a firmware-style CB reinit story on Quasar and does **not** gate other op ports tracked by this audit. No per-CB port table is produced.

## Signals (SILENT-WRONG / firmware reconfig)

- `device/unified_kernels/kernel_utils.hpp:127` — `get_local_cb_interface(cb_id).fifo_rd_ptr = byte_address >> cb_addr_shift` (**write**).
- `device/unified_kernels/kernel_utils.hpp:137` — comment: reconfig helper sets `fifo_rd_ptr, fifo_wr_ptr, fifo_size, fifo_limit, fifo_page_size, fifo_num_pages, ...`.
- `device/unified_kernels/kernel_utils.hpp:150` — `FORCE_INLINE void reconfig_cbs_for_mask(uint32_t* cb_config, uint32_t mask, uint32_t start_cb)` — full `LocalCBInterface` rewrite.
- `device/unified_kernels/kernel_utils.hpp:162,165` — `iface.fifo_rd_ptr = fifo_addr` / `iface.fifo_wr_ptr = fifo_addr` (**writes**).
- `device/unified_kernels/kernel_utils.hpp:177,178` — `*get_cb_tiles_received_ptr(cb) = 0` / `*get_cb_tiles_acked_ptr(cb) = 0` (**silent-wrong stream-register writes**).
- `device/unified_kernels/kernel_utils.hpp:213-214` — `reconfig_cbs_for_mask<...>(cb_config, cb_config[256/257], 0/32)` driver calls.
- `device/unified_kernels/kernel_utils.hpp:226,236` — additional `local_cb.fifo_rd_ptr` read/write helpers.

## Port note

Needs a **firmware-style reinit story on Quasar** (mask-driven CB reconfiguration + stream-register reset). Track separately; **does not gate** other in-scope op ports.

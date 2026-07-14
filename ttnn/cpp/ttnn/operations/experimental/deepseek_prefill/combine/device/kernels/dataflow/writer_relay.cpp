// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/dprint.h"
#include "api/debug/device_print.h"
#include "ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/overlap_config.hpp"

// ============================================================================
// writer_relay — relay ("R") core writer kernel.
// ============================================================================
//
// The relay is a dedicated tensix core prepended to each combine row (order
// R-S-U-U-U-U). It is the ONLY kernel on its core (RISCV_1 / compute unused), so it
// may pick whichever NOC it likes — the factory gives it NOC_0, the preferred NOC for
// the eventual write-to-eth. Its L1 holds a deep receive buffer (CB c_relay_buf,
// RELAY_SLOTS slots, each = route_info + one token payload) into which the sender will
// eventually write tokens + routing metadata instead of writing straight to the fabric.
//
// STAGE 1 (this): the relay is IDLE and UNCONNECTED — the buffer is allocated and the
// kernel is created (validating placement + L1 fit), but nothing drives it yet. Later
// stages wire up the sender->relay CB handshake and the relay->eth fabric send.
void kernel_main() {
    // Compile-time args (unused in stage 1, kept so the wiring is visible for later stages):
    //   0: cb_relay_buf id      1: RELAY_SLOTS
    constexpr uint32_t cb_relay_buf = get_compile_time_arg_val(0);
    constexpr uint32_t relay_slots = get_compile_time_arg_val(1);
    (void)cb_relay_buf;
    (void)relay_slots;

    // [debug][cmb-place] Host-computed placement, passed via RT args by the factory. Order matches
    // push_worker_coord_quad: relay_index, then logical(x,y), virt(x,y), phys_noc0(x,y), noc1(x,y).
    uint32_t rt_args_idx = 0;
    const uint32_t relay_index = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_logical_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_logical_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_virt_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_virt_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_phys_noc0_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_phys_noc0_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_noc1_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_noc1_y = get_arg_val<uint32_t>(rt_args_idx++);

    // Mirror [cmb-place sender]: host-computed coords + a device-derived cross-check (my_x/my_y on the
    // running NOC and get_absolute_logical_*). A mismatch means RT-arg misalignment or a coord-system
    // bug. No downstream lines yet — the relay is unconnected in stage 1. Keep this in sync with the
    // relay's topology as later stages add connectivity (sender->relay CB, relay->eth).
    DEVICE_PRINT(
        "[cmb-place relay] idx={} logical=({},{}) virt=({},{}) phys_noc0=({},{}) noc1=({},{}) | "
        "dev_virt=({},{}) dev_logical=({},{}) send_noc={}\n",
        relay_index,
        self_logical_x,
        self_logical_y,
        self_virt_x,
        self_virt_y,
        self_phys_noc0_x,
        self_phys_noc0_y,
        self_noc1_x,
        self_noc1_y,
        (uint32_t)my_x[noc_index],
        (uint32_t)my_y[noc_index],
        (uint32_t)get_absolute_logical_x(),
        (uint32_t)get_absolute_logical_y(),
        (uint32_t)noc_index);
}

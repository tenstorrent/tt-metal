// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rect.hpp"
#include "ring.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp"

namespace linear_fabric = tt::tt_fabric::linear::experimental;

void kernel_main() {
    Noc noc;
    constexpr bool forward = CT_ARG("relay_forward") == 1;
    constexpr uint32_t sources = CT_ARG("relay_sources");
    constexpr uint32_t slot_bytes = CT_ARG("relay_slot_bytes");
    constexpr uint32_t headers = 4;
    const uint32_t slots = arena_base(CT_ARG("cb_relay_slots"));
    const uint32_t link = CT_ARG("relay_link");
    const uint32_t direction = CT_ARG("relay_direction");

    size_t fabric_arg = get_named_compile_time_arg_val("rt_fabric");
    auto fabric = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(fabric_arg);
    fabric.open_finish();
    auto& sender = forward ? fabric.get_forward_connection() : fabric.get_backward_connection();
    volatile PACKET_HEADER_TYPE* packet_headers[headers];
    for (uint32_t i = 0; i < headers; i++) {
        packet_headers[i] = PacketHeaderPool::allocate_header();
    }

    for (uint32_t slot = 0; slot < 2 * sources; slot++) {
        reinterpret_cast<volatile tt_l1_ptr Message*>(slots + slot * slot_bytes)->sequence = 0;
    }
    {
        const uint32_t self_x = RT_ARG("self_x");
        const uint32_t self_y = RT_ARG("self_y");
        volatile tt_l1_ptr uint32_t* arrivals =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RT_ARG("arrival_address"));
        volatile tt_l1_ptr uint32_t* launches =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RT_ARG("launch_address"));
        const uint64_t remote_arrivals = safe_get_noc_addr(self_x, self_y, RT_ARG("arrival_address"), 0);
        for (uint32_t hops = 1; hops <= 2; hops++) {
            linear_fabric::fabric_unicast_noc_unicast_atomic_inc(
                &sender,
                packet_headers[hops - 1],
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_arrivals, 1},
                hops);
        }
        noc_async_writes_flushed();
        *launches += 2;
        noc_semaphore_wait_min(arrivals, *launches);
    }
    uint32_t consumed[sources] = {};
    uint32_t sent = 0;
    uint32_t remaining = sources;
    while (remaining != 0) {
        for (uint32_t source = 0; source < sources; source++) {
            const uint32_t slot = slots + (source * 2 + (consumed[source] & 1)) * slot_bytes;
            const volatile tt_l1_ptr Message* message = reinterpret_cast<volatile tt_l1_ptr Message*>(slot);
            if (message->sequence != consumed[source] + 1) {
                continue;
            }
            if (message->kind == MessageKind::End) {
                remaining--;
            } else {
                if (sent % headers == 0) {
                    noc_async_writes_flushed();
                }
                auto* header = packet_headers[sent++ % headers];
                const uint64_t semaphore =
                    safe_get_noc_addr(message->dst_x, message->dst_y, message->semaphore_address, 0);
                if (message->kind == MessageKind::Data) {
                    linear_fabric::fabric_unicast_noc_fused_unicast_with_atomic_inc(
                        &sender,
                        header,
                        slot + MESSAGE_BYTES,
                        message->bytes,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            safe_get_noc_addr(message->dst_x, message->dst_y, message->dst_address, 0),
                            semaphore,
                            1,
                            false},
                        message->hops);
                } else {
                    linear_fabric::fabric_unicast_noc_unicast_atomic_inc(
                        &sender, header, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{semaphore, 1}, message->hops);
                }
                noc_async_writes_flushed();
            }
            consumed[source]++;
            const uint32_t core = 2 * source + link;
            Semaphore<>(CT_ARG("sem_relay_free") + direction)
                .up(noc,
                    CT_ARG("b_grid_x_start") + core % CT_ARG("b_width"),
                    CT_ARG("b_grid_y_start") + core / CT_ARG("b_width"),
                    1);
        }
    }
    fabric.close();
}

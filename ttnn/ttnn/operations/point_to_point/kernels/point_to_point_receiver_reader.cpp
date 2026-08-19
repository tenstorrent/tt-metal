// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point — receiver reader (NCRISC). Phases 0, 5, 6 and 8.
//
//   0. Fabric-ack the sender ("I am launched and my local semaphore is clean")
//      with the helper's one-shot FabricStreamSender::signal(), which collapses
//      open -> arm_inc -> inc -> close. Without this ready-handshake the sender of
//      invocation k+1 could bump this device's semaphore while it is still inside
//      invocation k, whose phase-8 reset would then erase k+1's "done" -> hang.
//   5. Wait for the sender's "done" inc — the whole payload has landed.
//   6. Read each landed packet back out of the LOCAL intermediate DRAM and
//      de-frame it into cb_output_pages, the exact mirror of the sender's framing.
//   8. Re-arm the semaphore AFTER the wait (cache-reuse rule,
//      ccl_helpers_dataflow.hpp:111-113).
//
// There is deliberately NO fabric ingress kernel: the fabric lands the payload
// directly in this device's DRAM, and the read-back is a plain local
// noc_async_read the op owns (ccl_helpers_dataflow.hpp:109-110 — "the receive
// INGRESS is likewise a local NoC read the op owns; there is no
// FabricStreamReceiver"). Packet de-framing is likewise op-owned (:130-133).
//
// MANDATORY (op_design.md Key Risk #1): the intermediate TensorAccessor is built
// with exactly TWO arguments, identical CT args to the sender's — symmetric
// mesh-buffer addressing is what makes "page k" the same bytes on both chips.
// packet_size is used ONLY as the noc_async_read byte count.

#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

using tt::data_movement::common::round_up;
using tt::data_movement::common::tt_memmove;
using namespace dataflow_kernel_lib::ccl;

void kernel_main() {
    constexpr uint32_t cb_packet_landing = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_pages = get_compile_time_arg_val(1);
    constexpr uint32_t alignment = get_compile_time_arg_val(2);
    constexpr uint32_t page_segments = get_compile_time_arg_val(3);
    constexpr auto intermediate_args = TensorAccessorArgs<4>();

    uint32_t ai = 0;
    const uint32_t intermediate_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t num_pages = get_arg_val<uint32_t>(ai++);
    const uint32_t total_packets = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t packet_size = get_arg_val<uint32_t>(ai++);
    const uint32_t pages_per_packet = get_arg_val<uint32_t>(ai++);
    const uint32_t sem_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t sender_num_hops = get_arg_val<uint32_t>(ai++);

    // Fabric connection arg block is LAST; its leading has_forward flag encodes
    // the route direction back toward the sender.
    size_t conn_arg_idx = ai;
    const bool sender_is_forward = get_arg_val<uint32_t>(conn_arg_idx);
    FabricStreamSender<> ack_sender(conn_arg_idx, sender_is_forward, alignment);

    auto sem_ptr = dataflow_kernel_lib::addr_to_l1_ptr(sem_addr);

    // --- Phase 0: one-shot fabric atomic-inc of the SENDER's semaphore. Both
    // endpoints use logical core (0,0) and the GlobalSemaphore is mesh-wide (same
    // absolute address on every device), so get_noc_addr(sem_addr) computed here
    // names "the same semaphore, on the chip the packet is routed to".
    // signal() is TERMINAL — do not also call open() on this sender.
    ack_sender.signal(sender_num_hops, get_noc_addr(sem_addr));

    // 2-argument ctor: stride == buffer.aligned_page_size() (CT-baked). NOT packet_size.
    const auto intermediate = TensorAccessor(intermediate_args, intermediate_addr);

    const uint32_t aligned_page_size = round_up(page_size, alignment);

    // Reserve-once scratch (0 pushes / 0 waits): the local noc_async_read +
    // barrier completes before the de-framing memmoves touch it.
    cb_reserve_back(cb_packet_landing, 1);
    const uint32_t landing = get_write_ptr(cb_packet_landing);

    // --- Phase 5: wait for the sender's "done". Payload writes and the trailing
    // inc ride the same fabric connection with the same route, so they are
    // delivered in issue order: sem >= 1 implies every payload byte has landed.
    noc_semaphore_wait_min(sem_ptr, 1);

    // --- Phase 6: read back + de-frame (exact mirror of the sender's framing).
    if constexpr (page_segments == 1) {
        // Regime A — de-coalesce: one packet holds up to pages_per_packet pages at
        // aligned_page_size stride.
        for (uint32_t pkt = 0; pkt < total_packets; ++pkt) {
            noc_async_read(intermediate.get_noc_addr(pkt), landing, packet_size);
            noc_async_read_barrier();
            const uint32_t base = pkt * pages_per_packet;
            const uint32_t n = std::min(pages_per_packet, num_pages - base);
            for (uint32_t k = 0; k < n; ++k) {
                cb_reserve_back(cb_output_pages, 1);
                tt_memmove<false, false, false, 0>(
                    get_write_ptr(cb_output_pages), landing + k * aligned_page_size, page_size);
                cb_push_back(cb_output_pages, 1);
            }
        }
    } else {
        // Regime B — reassemble: page_segments packets make one page.
        uint32_t pkt = 0;
        for (uint32_t p = 0; p < num_pages; ++p) {
            cb_reserve_back(cb_output_pages, 1);
            const uint32_t dst = get_write_ptr(cb_output_pages);
            for (uint32_t s = 0; s < page_segments; ++s) {
                noc_async_read(intermediate.get_noc_addr(pkt), landing, packet_size);
                noc_async_read_barrier();
                ++pkt;
                const uint32_t off = s * packet_size;
                const uint32_t bytes = std::min(page_size - off, packet_size);
                tt_memmove<false, false, false, 0>(dst + off, landing, bytes);
            }
            cb_push_back(cb_output_pages, 1);
        }
    }

    // --- Phase 8: re-arm AFTER the wait, for the next program-cache hit.
    noc_semaphore_set(sem_ptr, 0);
}

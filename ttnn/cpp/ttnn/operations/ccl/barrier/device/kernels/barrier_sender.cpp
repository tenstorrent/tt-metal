// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <array>
#include "tt_metal/hw/inc/ethernet/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp"

#define MIN_WAIT 100000

struct addr_sem_pair {
    uint32_t addr;
    uint32_t sem_addr;
};

static constexpr bool DISABLE_CONTEXT_SWITCHING = true;
static constexpr uint8_t NUM_CHANNELS = 8;

FORCE_INLINE
void eth_send_bytes_over_channel_payload_only(
    const uint32_t src_addr,
    const uint32_t dst_addr,
    const uint32_t num_bytes,
    volatile eth_channel_sync_t* channel_sync) {
    channel_sync->bytes_sent = num_bytes;
    channel_sync->receiver_ack = 0;
    internal_::eth_send_packet(0, src_addr >> 4, dst_addr >> 4, num_bytes >> 4);
}

FORCE_INLINE void perform_noc_read(volatile uint32_t* sem_addr) {
    eth_noc_semaphore_wait(sem_addr, 1, MIN_WAIT);
    *sem_addr = 0;
}

FORCE_INLINE void perform_eth_write(
    uint32_t& channels_addrs, volatile eth_channel_sync_t* eth_channel_syncs, const uint32_t page_size) {
    eth_send_bytes_over_channel_payload_only(
        channels_addrs, channels_addrs, page_size + sizeof(eth_channel_sync_t), eth_channel_syncs);
    while (!eth_is_receiver_channel_send_done(0)) {
        asm volatile("");
    }
}

void kernel_main() {
    uint32_t arg_idx = 0;
    const bool is_ring_start = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    erisc::datamover::handshake::deprecated::sender_side_start(handshake_addr);
    uint32_t channels_addrs = get_arg_val<uint32_t>(arg_idx++);
    volatile uint32_t* sem_addr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t host_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t host_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint64_t host_semaphore_addr =
        get_noc_addr(host_noc_x, host_noc_y, get_semaphore(get_arg_val<uint32_t>(arg_idx++)));

    const uint32_t transfer_size = 16;
    volatile eth_channel_sync_t* channels_syncs_addrs =
        reinterpret_cast<volatile eth_channel_sync_t*>(channels_addrs + transfer_size);
    channels_syncs_addrs->bytes_sent = 0;
    channels_syncs_addrs->receiver_ack = 0;
    *sem_addr = 0;
    erisc::datamover::handshake::deprecated::sender_side_finish(handshake_addr);
    noc_semaphore_inc(host_semaphore_addr, 1);

    // Ensure every core has completed their previous tasks
    perform_noc_read(sem_addr);
    perform_eth_write(channels_addrs, channels_syncs_addrs, 16);
    // signal the start
    perform_noc_read(sem_addr);
    perform_eth_write(channels_addrs, channels_syncs_addrs, 16);
}

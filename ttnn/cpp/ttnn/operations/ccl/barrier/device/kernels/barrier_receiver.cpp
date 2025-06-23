// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ethernet/dataflow_api.h"
#include "tt_metal/hw/inc/ethernet/dataflow_api.h"
#include <array>
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp"

#define MIN_WAIT 100000

FORCE_INLINE void perform_rs_loop(uint64_t channel_sem_addr, volatile eth_channel_sync_t* eth_channel_syncs) {
    // In the Ring Start case, we first send the signal to the sender and then we wait on the ethernet
    noc_semaphore_inc(channel_sem_addr, 1);
    uint32_t count = 0;
    const uint32_t page_size = 16;
    constexpr uint32_t bytes_to_wait_for = page_size + sizeof(eth_channel_sync_t);
    eth_wait_for_bytes_on_channel_sync_addr(bytes_to_wait_for, eth_channel_syncs, MIN_WAIT);
    eth_channel_syncs->bytes_sent = 0;
    wait_for_eth_txq_cmd_space(MIN_WAIT);
    send_eth_receiver_channel_done(eth_channel_syncs);
}

FORCE_INLINE void perform_loop(uint64_t channel_sem_addr, volatile eth_channel_sync_t* eth_channel_syncs) {
    uint32_t count = 0;
    const uint32_t page_size = 16;
    constexpr uint32_t bytes_to_wait_for = page_size + sizeof(eth_channel_sync_t);
    eth_wait_for_bytes_on_channel_sync_addr(bytes_to_wait_for, eth_channel_syncs, MIN_WAIT);
    eth_channel_syncs->bytes_sent = 0;
    noc_semaphore_inc(channel_sem_addr, 1);
    count = 0;
    wait_for_eth_txq_cmd_space(MIN_WAIT);
    send_eth_receiver_channel_done(eth_channel_syncs);
}

void kernel_main() {
    uint32_t arg_idx = 0;
    // Get the runtime arguments
    const bool is_ring_start = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint64_t channel_sem_addr = get_noc_addr(eth_noc_x, eth_noc_y, get_arg_val<uint32_t>(arg_idx++));
    volatile uint32_t* start_semaphore = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t channels_addrs = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t host_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t host_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint64_t host_semaphore_addr =
        get_noc_addr(host_noc_x, host_noc_y, get_semaphore(get_arg_val<uint32_t>(arg_idx++)));

    const uint32_t transfer_size = 16;
    // Get the per-channel semaphores
    volatile eth_channel_sync_t* channels_syncs_addrs =
        reinterpret_cast<volatile eth_channel_sync_t*>(channels_addrs + transfer_size);
    channels_syncs_addrs->bytes_sent = 0;
    channels_syncs_addrs->receiver_ack = 0;
    // Semaphore is mapped to sender core
    erisc::datamover::handshake::deprecated::receiver_side_start(handshake_addr);
    erisc::datamover::handshake::deprecated::receiver_side_finish(handshake_addr);

    *start_semaphore = 0;

    noc_semaphore_inc(host_semaphore_addr, 1);
    eth_noc_semaphore_wait(start_semaphore, 1, MIN_WAIT);

    if (is_ring_start) {
        // Ensure everyone has finished the previous OP
        perform_rs_loop(channel_sem_addr, channels_syncs_addrs);
        // Signal the start

        perform_rs_loop(channel_sem_addr, channels_syncs_addrs);
    } else {
        // Propagate the ensurance that everyone is ready to start
        perform_loop(channel_sem_addr, channels_syncs_addrs);

        // Propagate the start signal
        perform_loop(channel_sem_addr, channels_syncs_addrs);
        // You can start once the start signal has passed through
    }
}

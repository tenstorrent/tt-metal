// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt_metal/hw/inc/dataflow_api.h"
#include "debug/assert.h"
#include "debug/dprint.h"
#include <array>
#include "tt_metal/hw/inc/ethernet/dataflow_api.h"


struct addr_sem_pair {
    uint32_t addr;
    uint32_t sem_addr;
};


static constexpr bool DISABLE_CONTEXT_SWITCHING = true;
static constexpr uint8_t NUM_CHANNELS = 8;

FORCE_INLINE void eth_setup_handshake(uint32_t handshake_register_address) {
    eth_send_bytes(handshake_register_address, handshake_register_address, 16);
    eth_wait_for_receiver_done();
}

FORCE_INLINE
void eth_send_bytes_over_channel_payload_only(
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t num_bytes,
    volatile eth_channel_sync_t *channel_sync) {
    channel_sync->bytes_sent = num_bytes;
    channel_sync->receiver_ack = 0;
    internal_::eth_send_packet(
        0, src_addr >> 4, dst_addr >> 4, num_bytes >> 4);
}

FORCE_INLINE void perform_noc_read(
    uint32_t channels_sem_addrs, uint32_t ring_index)
{
    volatile uint32_t *sem_addr = reinterpret_cast<volatile uint32_t*>(get_semaphore(channels_sem_addrs));
    DPRINT << "Sem value is currently" << *sem_addr << ENDL();
    if(ring_index == 0)
    {
        DPRINT << "sender reading noc on device" << ring_index << "at address" << (uint32_t) (sem_addr) << "I am on" << (uint32_t) (my_y[0]) << (uint32_t) (my_x[0]) << ENDL();
    }
    eth_noc_semaphore_wait(sem_addr, 1);
    *sem_addr = 0;
}

FORCE_INLINE void perform_eth_write(
    uint32_t &channels_addrs,
    volatile eth_channel_sync_t* eth_channel_syncs,
    uint32_t page_size)
{
    uint32_t buffer_addr = channels_addrs;
    eth_send_bytes_over_channel_payload_only(buffer_addr, buffer_addr, page_size + sizeof(eth_channel_sync_t), eth_channel_syncs);
    while (!eth_is_receiver_channel_send_done(0)) {
        run_routing();
        asm volatile ("");
    }
}

void kernel_main() {
    uint32_t arg_idx = 0;
    const bool is_ring_start = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ring_index = get_arg_val<uint32_t>(arg_idx++);
    uint32_t channels_addrs = get_arg_val<uint32_t>(arg_idx++);
    uint32_t channels_sem_addrs = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t max_concurrent_samples = 1;
    const uint32_t transfer_size = 16;
    volatile eth_channel_sync_t* channels_syncs_addrs = reinterpret_cast<volatile eth_channel_sync_t*>(channels_addrs + transfer_size);
    channels_syncs_addrs->bytes_sent = 0;
    channels_syncs_addrs->receiver_ack = 0;

    DPRINT << "semaphore raw address on device " << ring_index << "is " << channels_sem_addrs <<ENDL();
    DPRINT << "starting sender " << ring_index << ENDL();
    //for (uint32_t i = 0; i < max_concurrent_samples; i++) {
    //    *(volatile uint32_t*)channels_sem_addrs[i] = 0;
    //}

    // Avoids hang in issue https://github.com/tenstorrent/tt-metal/issues/9963
    //for (uint32_t i = 0; i < 2000000000; i++) {
    //    asm volatile("nop");
    //}
    eth_setup_handshake(handshake_addr);

    //Ensure every core has completed their previous tasks
    DPRINT << "sender handshake is done on device" << ring_index << ENDL();
    perform_noc_read(channels_sem_addrs, ring_index);
    DPRINT << "sender Noc read received on device " << ring_index << ENDL();
    perform_eth_write(channels_addrs, channels_syncs_addrs, 16);
    DPRINT << "Sender wrote to ethernet on device" << ring_index << ENDL();
    //signal the start
    perform_noc_read(channels_sem_addrs,ring_index);
    DPRINT << "sender Noc read received on device " << ring_index << ENDL();
    perform_eth_write(channels_addrs, channels_syncs_addrs, 16);
    DPRINT << "Sender wrote to ethernet on device" << ring_index << ENDL();
    if (is_ring_start) {
        //We will just read the semaphore one last time to make sure we start at the same time as the receiver
        perform_noc_read(channels_sem_addrs,ring_index);
        DPRINT << "sender Noc read received on device " << ring_index << ENDL();
    }
}

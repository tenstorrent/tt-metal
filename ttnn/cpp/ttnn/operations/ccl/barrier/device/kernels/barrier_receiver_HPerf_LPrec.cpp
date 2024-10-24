// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/assert.h"
#include "debug/dprint.h"
#include "ethernet/dataflow_api.h"
#include "tt_metal/hw/inc/ethernet/dataflow_api.h"
#include <array>


struct addr_sem_pair {
    uint32_t addr;
    uint32_t sem_addr;
};

static constexpr uint32_t NUM_CHANNELS = 8;

FORCE_INLINE void send_eth_receiver_channel_done(volatile eth_channel_sync_t *channel_sync) {
    channel_sync->bytes_sent = 0;
    channel_sync->receiver_ack = 0;
    internal_::eth_send_packet(
        0,
        ((uint32_t)(channel_sync)) >> 4,
        ((uint32_t)(channel_sync)) >> 4,
        1);
}

FORCE_INLINE void do_noc_write(
        uint32_t ring_index,
        //uint32_t buffer_addr, uint32_t page_size,
        uint32_t eth_noc_x, uint32_t eth_noc_y, uint32_t sender_sem) {
    volatile uint64_t send_sem_noc_addr = get_noc_addr(eth_noc_x, eth_noc_y, sender_sem);
    //We really only need the semaphore stuff but I am leaving this commented just in case
    //uint64_t send_buffer_noc_addr = get_noc_addr(eth_noc_x, eth_noc_y, buffer_addr);
    //noc_async_write(buffer_addr, send_buffer_noc_addr, page_size);
    noc_semaphore_inc(send_sem_noc_addr, 1);

}

FORCE_INLINE void perform_rs_loop(
    uint32_t channels_addrs,
    uint32_t channels_sem_addrs,
    volatile eth_channel_sync_t* eth_channel_syncs,
    uint32_t page_size,
    uint32_t eth_noc_x,
    uint32_t eth_noc_y,
    uint32_t ring_index
    )
{
    //In the RS case, we first send the signal to the sender and then we wait on the ethernet
    do_noc_write(
        //buffer_addr, page_size,
        ring_index,
        eth_noc_x, eth_noc_y,channels_sem_addrs);
    //Do an eth read
    while(eth_channel_syncs->bytes_sent == 0) {
        asm volatile ("");
        run_routing();
    }
    while (eth_txq_is_busy())
    {
        run_routing();
    }
    send_eth_receiver_channel_done(eth_channel_syncs);
}
FORCE_INLINE void perform_last_noc_write(
    uint32_t channels_addrs,
    uint32_t channels_sem_addrs,
    uint32_t page_size,
    uint32_t eth_noc_x,
    uint32_t eth_noc_y,
    uint32_t ring_index
    )
{
    //Only do the noc write part
    uint32_t sender_sem = channels_sem_addrs;
    uint32_t buffer_addr = channels_addrs;
    do_noc_write(
        //buffer_addr, page_size,
        ring_index,
        eth_noc_x, eth_noc_y,sender_sem);
}

FORCE_INLINE void perform_loop(
    uint32_t channels_addrs,
    uint32_t channels_sem_addrs,
    volatile eth_channel_sync_t* eth_channel_syncs,
    uint32_t page_size,
    uint32_t eth_noc_x,
    uint32_t eth_noc_y,
    uint32_t ring_index
    )
{
    uint32_t buffer_addr = channels_addrs;
    while(eth_channel_syncs->bytes_sent == 0) {
        run_routing();
        asm volatile ("");
    }
    do_noc_write(
        //buffer_addr, page_size,
        ring_index,
        eth_noc_x, eth_noc_y,channels_sem_addrs);
    //noc_async_writes_flushed();
    while (eth_txq_is_busy())
    {
        run_routing();
    }

    send_eth_receiver_channel_done(eth_channel_syncs);


}

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    eth_wait_for_bytes(16);
    eth_receiver_channel_done(0);
}

void kernel_main() {
    uint32_t arg_idx = 0;
    //Get the runtime arguments
    const bool is_ring_start = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ring_index = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_noc_y = get_arg_val<uint32_t>(arg_idx++);
    volatile uint32_t* start_semaphore = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    uint32_t channels_addrs = get_arg_val<uint32_t>(arg_idx++);
    uint32_t channels_sem_addrs = get_arg_val<uint32_t>(arg_idx++);
    uint32_t host_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t host_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t host_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t max_concurrent_samples = 1;
    const uint32_t transfer_size = 16;

    //Get the per-channel semaphores
    volatile eth_channel_sync_t* channels_syncs_addrs = reinterpret_cast<volatile eth_channel_sync_t*>(channels_addrs + transfer_size);
    channels_syncs_addrs->bytes_sent = 0;
    channels_syncs_addrs->receiver_ack = 0;
    //Semaphore is mapped to sender core

    eth_setup_handshake(handshake_addr, false);
    // We reuse the handshake address to send back acks to the sender core.
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_addr)[0] = 1;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_addr)[1] = 1;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_addr)[2] = 0;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_addr)[3] = 0;
    uint32_t eth_channel_sync_ack_addr = handshake_addr;

    *start_semaphore = 0;
    uint64_t host_semaphore_addr = get_noc_addr(host_noc_x, host_noc_y, get_semaphore(host_semaphore));
    noc_semaphore_inc(host_semaphore_addr, 1);
    eth_noc_semaphore_wait(start_semaphore, 1);

    if(is_ring_start)
    {
        //Ensure everyone has finished the previous OP
        perform_rs_loop(channels_addrs, channels_sem_addrs, channels_syncs_addrs, 16, eth_noc_x, eth_noc_y, ring_index);
        //Signal the start

        perform_rs_loop(channels_addrs, channels_sem_addrs, channels_syncs_addrs, 16, eth_noc_x, eth_noc_y, ring_index);

        //Finish off when the signal has come around
        perform_last_noc_write(channels_addrs, channels_sem_addrs, 16, eth_noc_x, eth_noc_y, ring_index);

    }
    else
    {
        //Propagate the ensurance that everyone is ready to start
        perform_loop(channels_addrs, channels_sem_addrs, channels_syncs_addrs, 16, eth_noc_x, eth_noc_y, ring_index);

        //Propagate the start signal
        perform_loop(channels_addrs, channels_sem_addrs, channels_syncs_addrs, 16, eth_noc_x, eth_noc_y, ring_index);
        //You can start once the start signal has passed through
    }


}

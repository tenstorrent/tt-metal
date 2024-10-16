// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ethernet/dataflow_api.h"


namespace erisc {
namespace datamover {

/*
 * Before any payload messages can be exchanged over the link, we must ensure that the other end
 * of the link is ready to start sending/receiving messages. We perform a handshake to ensure that's
 * case. Before handshaking, we make sure to clear any of the channel sync datastructures local
 * to our core.
 *
 * The handshaking process is split into two parts for the sender/master and two parts for the
 * the slave. The handshake is broken into 2 parts so that the master can initiate the handshake
 * as early as possible so the message can be "in flight" over the ethernet link while other EDM
 * initialization is taking place.
 *
 * Important note about handshaking: the sender/master canNOT complete the handshake until all receiver
 * channels are initialized. Otherwise we have a race between channel initialization on the receiver side
 * and real payload data (and signals) using those channels.
 *
 * Note that the master and slave concepts here only apply in the context of handshaking and initialization
 * of the EDM. They do not apply during the main EDM execution loop.
 *
 * The basic protocol for the handshake is to use the reserved space at erisc_info[0] where the master writes
 * and sends payload available information to that channel. The receive must acknowledge that message and upon
 * doing so, considers the handshake complete.
 */
namespace handshake {

static constexpr uint32_t A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH = 1000000000;

/*
 * Initialize base datastructures and values which are common to master and slave EDM cores.
 * The main memory region initialized here is the channel ack region offset 16B from the
 * base handshake address.
 *
 * This memory region serves a special purpose for flow control between EDM cores. This
 * 16B region is initialized to a fixed set of values. This region is used by receiver
 * EDM channels when sending first level acks to its corresponding sender EDM channel.
 *
 * See ChannelBuffer::eth_receiver_channel_ack for more information
 */
FORCE_INLINE void initialize_edm_common_datastructures(std::uint32_t handshake_register_address) {
    reinterpret_cast<volatile tt_l1_ptr uint32_t *>(handshake_register_address)[4] = 1;
    reinterpret_cast<volatile tt_l1_ptr uint32_t *>(handshake_register_address)[5] = 1;
    reinterpret_cast<volatile tt_l1_ptr uint32_t *>(handshake_register_address)[6] = 0x1c0ffee1;
    reinterpret_cast<volatile tt_l1_ptr uint32_t *>(handshake_register_address)[7] = 0x1c0ffee2;

    erisc_info->channels[0].receiver_ack = 0;
    for (uint32_t i = 1; i < eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS; i++) {
        erisc_info->channels[i].bytes_sent = 0;
        erisc_info->channels[i].receiver_ack = 0;
    }
    *(volatile tt_l1_ptr uint32_t *)handshake_register_address = 0;
}

/*
 * As the designated master EDM core, initiate a handshake by sending a packet to reserved
 * memory region.
 */
FORCE_INLINE void sender_side_start(std::uint32_t handshake_register_address) {
    initialize_edm_common_datastructures(handshake_register_address);
    eth_wait_receiver_done(A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH);
    while (eth_txq_reg_read(0, ETH_TXQ_CMD) != 0) {
        asm volatile("nop");
    }
    eth_send_bytes(handshake_register_address, handshake_register_address, 16);
}

/*
 * As the designated master EDM core, wait for the acknowledgement from the slave EDM core
 */
FORCE_INLINE void sender_side_finish(std::uint32_t handshake_register_address) {
    eth_wait_for_receiver_done(A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH);
}

FORCE_INLINE void receiver_side_start(std::uint32_t handshake_register_address) {
    initialize_edm_common_datastructures(handshake_register_address);
}

/*
 * Return: true if slave EDM handshake core is able to complete the handshake with
 * an ack.
 */
FORCE_INLINE bool receiver_side_can_finish() {
    return eth_bytes_are_available_on_channel(0);
}

/*
 * As the designated slave EDM core, send the acknowledgement to the master EDM core.
 * The slave EDM core shall only acknowledge after receiving the initial handshake packet
 * from the master EDM core.
 */
FORCE_INLINE void receiver_side_finish(std::uint32_t handshake_register_address) {
    eth_wait_for_bytes(16, A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH);
    while (eth_txq_reg_read(0, ETH_TXQ_CMD) != 0) {
        asm volatile("nop");
    }
    eth_receiver_channel_done(0);
}
} // namespace handshake

} // namespace datamover
} // namespace erisc

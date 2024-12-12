// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>

#include "dataflow_api.h"
#include "tt_metal/hw/inc/ethernet/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm/edm_handshake.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header_validate.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_transmission.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"

using ttnn::ccl::WorkerXY;

/*

The fabric Erisc Data Mover (EDM) is a component that can be used to build *very* simple linear topology fabrics.
One of these EDMs can be instantiated on each ethernet link. It is built from 3 "channels" (though the definition
of channel here is a little loose since two of the 3 will merge traffic, so this setup could be interpreted as a
two channel setup.). This EDM implements packet based packets only - concepts like sockets are not supported.

## EDM Structure

There are two sender channels and one receiver channel. "Sender" and "receiver" are relative to the Ethernet link,
not the chip. Sender sends over the link and receiver receives from the link.

Each sender channel serves a different purpose:
- Sender channel 0 : Accepts packets from a workers on the local chip
- Sender channel 1: accepts packets from an upstream EDM (i.e. an upstream
  EDM receiver channel on the same chip but different core)

The receiver channel accepts packets from the Ethernet link and can do one (or both) of:
- Write the packet to local chhip if it is the intended destination (unicast or mcast)
- Forward the packet to the next chip in the line if:
  - Unicast and not the target chip
  - Multicast and this chip is in the multicast target range

Sender channels will merge traffic into the remote EDM's receiver channel.

Below is a diagram that shows how EDMs can be connected over an ethernet link. In this case, the two
EDM kernels are run on separate, but connected ethernet link cores.

 ┌───────────────────────┐           ┌───────────────────────┐
 │    Sender Channel 0   │           │    Receiver Channel   │
 │   ┌────────────────┐  │           │   ┌────────────────┐  │
 │   │                ┼──┼───┬───────┼───►                │  │
 │   │                │  │   │       │   │                │  │
 │   └────────────────┘  │   │       │   └────────────────┘  │
 │    Sender Channel 1   │   │       │    Sender Channel 1   │
 │   ┌────────────────┐  │   │       │   ┌────────────────┐  │
 │   │                ┼──┼───┘       │   │                │  │
 │   │                │  │         ┌─┼───┼                │  │
 │   └────────────────┘  │         │ │   └────────────────┘  │
 │    Receiver Channel   │         │ │    Sender Channel 0   │
 │   ┌────────────────┐  │         │ │   ┌────────────────┐  │
 │   │                │  │         │ │   │                │  │
 │   │                ◄──┼─────────┴─┼───┼                │  │
 │   └────────────────┘  │           │   └────────────────┘  │
 │                       │           │                       │
 │                       │           │                       │
 └───────────────────────┘           └───────────────────────┘


## Building a "Fabric"

At present, only linear topologies are supported, and one per ethernet link along that given line.
Below shows the intended connectivity of EDMs across chips in a hypothetical 3-chip fabric. For longer
lines, the pattern would be extended.

           CHIP 0                              CHIP 1                             CHIP 2
     ┌─────────────────┐                ┌─────────────────┐                ┌─────────────────┐
     │                 │                │                 │                │                 │
┌────┴─────┐ ▲   ┌─────┴────┐      ┌────┴─────┐ ▲   ┌─────┴────┐      ┌────┴─────┐ ▲   ┌─────┴────┐
│   EDM    │ │   │   EDM    │      │   EDM    │ │   │   EDM    │      │   EDM    │ │   │   EDM    │
│ ┌──────┐ │ │   │ ┌──────┐ │      │ ┌──────┐ │ │   │ ┌──────┐ │      │ ┌──────┐ │ │   │ ┌──────┐ │
│ │ Rx   ┼─┼─┴───┼─► S1   ┼─┼─┬────┼─► Rx   ┼─┼─┴───┼─► S1   ┼─┼┬─────┼─► Rx   ┼─┼─┘   | | S1   │ │
│ └──────┘ │     │ └──────┘ │ │    │ └──────┘ │     │ └──────┘ ││     │ └──────┘ │     │ └──────┘ │
│ ┌──────┐ │     │ ┌──────┐ │ │    │ ┌──────┐ │     │ ┌──────┐ ││     │ ┌──────┐ │     │ ┌──────┐ │
│ │ S0   ◄─┼──┬──┼─► S0   ┼─┼─┘   ┌┼─┼ S0   ◄─┼──┬──┼─► S0   ┼─┼┘    ┌┼─┼ S0   ◄─┼──┬──┼─► S0   │ │
│ └──────┘ │  │  │ └──────┘ │     ││ └──────┘ │  │  │ └──────┘ │     ││ └──────┘ │  │  │ └──────┘ │
│ ┌──────┐ │  │  │ ┌──────┐ │     ││ ┌──────┐ │  │  │ ┌──────┐ │     ││ ┌──────┐ │  │  │ ┌──────┐ │
│ │ S1   | |  │ ┌┼─┼ Rx   ◄─┼─────┴┼─┼ S1   ◄─┼─┐│ ┌┼─┼ Rx   ◄─┼─────┴┼─┼ S1   ◄─┼─┐│ ┌┼─┼ Rx   │ │
│ └──────┘ │  | |│ └──────┘ │      │ └──────┘ │ └┼─┤│ └──────┘ │      │ └──────┘ │ └┼─┤│ └──────┘ │
└────┬─────┘  │ │└─────┬────┘      └────┬─────┘  │ │└─────┬────┘      └────┬─────┘  │ │└─────┬────┘
     │          ▼      │                │          ▼      │                │          ▼      │
     └─────────────────┘                └─────────────────┘                └─────────────────┘


## Connecting Workers to Channels

As mentioned, only one worker can push to a given EDM sender channel at a time. In order to send to an EDM
sender channel, the worker must establish a connection. The connection protocol is as follows and is started
by the worker (the EDM is a slave in this protocol).

*NOTE*: If multiple workers try to connect to the same EDM sender channel at the same time, the behavior is undefined.
*NOTE*: Additionally, if a worker pushes packets to a channel it isn't connected to, behaviour is undefined.
*NOTE*: Undefined == likely hang

The `WorkerToFabricEdmSender` from `ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/edm_fabric_worker_adapters.hpp`
provides an implementation of the connection protocol. `WorkerToFabricEdmSender` also acts as a wrapper around that
protocol so workers can simply call `open()` to execute the connection protocol without having to manually reimplement
for each kernel.

### Protocol
Worker:
- Read from EDM sender channel buffer_index address
  - Required so that the worker knows where to write its first packet (since the channel may already contain packets from
    a previous connection)
- Write worker core X/Y (NOC 0 based)
- Write worker flow control semaphore L1 address

EDM Sender Channel:
- Check local connection valid semaphore for new established connection
  - When the connection semaphore indicates an active connection, the channel assumes all other relevant fields were
    correctly populated by the worker:
    - Worker core_x (on NOC 0)
    - Worker core_y (on NOC 0)
    - Worker flow control semaphore L1 address


## Tearing Down Connections

Every worker is required to explicitly teardown its connection with the EDM before terminating. To do this, the worker
must simply write a `0` to the EDM sender channel's connection semaphore address. As long as the worker has sent all
of its packets to the EDM before this, then the EDM will guarantee to forward the messages correctly.

At this point, it is safe for another kernel to establish a connection.

## Packet Structure

Workers are responsible for populating packet headers before sending to the EDM. The packet header structure is defined
in `ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp`.

## Channel structure

Each EDM channel is built from one or more buffers. Each buffer is the same size and can hold atmost one packet.
Neighbouring packets occupy nehighouring buffers - with the exception of the last buffer index. The next packet after a write
into the last buffer index will wrap around to the first buffer index. Even if packets do not occupy the full buffer, subsequent
packets will always be written into the next logical buffer. A gap will exist in memory but the EDM will not send that padded data
(unless it is more performant - which is possible in some special cases)

 Example channel with 8 buffers
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
 buf 0   buf 1   buf 2   buf 3   buf 4   buf 5   buf 6   buf 7


Here we have an example of a channel with 4 buffers, filled with some number of packets. Each packet is a different size.
Packets 0, 2, and 3 are smaller than the full buffer size, while packet 1 is the full buffer size.

┌───────────────┬───────────────┬───────────────┬───────────────┐
│H|Payload| / / │H|Payload      │H|Pyld| / / / /│H|Payload  |/ /│
│ |       |/ / /│ |             │ |    |/ / / / │ |         | / │
└───────────────┴───────────────┴───────────────┴───────────────┘
  buf 0           buf 1           buf 2           buf 3


A detail of the channel structure is omitted from the above diagram, namely the EDM <-> EDM flow control region for each buffer.
Each buffer really looks something like this:


             &header->  |----------------| channel_base_address
                        |    header      |
            &payload->  |----------------|
                        |                |
                        |    payload     |
                        |                |
       &channel_sync->  |----------------|
                        |  channel_sync  |  // This is new
                        ------------------

The "channel_sync" is an `eth_channel_sync_t` and is internal to the EDM implementation and is used to indicate packet
transmission state between sender and receiver EDMs.

The protocol for its use is:
1) Sender updates the field indicating new data:
   - set `bytes_sent` to a non-zero value indicating new data
   - clear `receiver_ack` to 0
   - set `src_id` to the sender channel id so the receiver knows who the sender was (and where the ack should go)
2) Sender sends this channel sync to the corresponding location in the receiver channel (either in the same transmission
   as the packet or separately)
3) Receiver sees that `bytes_sent` is non-zero, indicating a new packet. It sends back an acknowledgement (first level):
   - set `receiver_ack` to non-zero
   *NOTE* IMPORTANT: To avoid a race, the receiver must be sure to send its channel_sync_t from a different address it uses
   as for the second level acknowledgement
   3b) When sender receives an ack, it understands it can overwrite its local copy of the packet with new data
4) After receiver properly writes out its packet, it sends a second level acknowledgement, indicating it can receive new
   data into this specific buffer index:
   - clear the bytes_sent and receiver_ack fields and send back the `channel_sync` to the sender



## Sending Packets
Sending a packet is done as follows:

1) Worker waits for flow control semaphore increment from EDM sender channel
  - Indicates there is space at the next buffer index for a packet
2) Worker performs a noc write of its packet to the EDM sender channel at the buffer index

*NOTE*: !!!ALL PACKETS MUST CONTAIN DESTINATION NOC X/Y AS NOC 0 COORDINATES, REGARDLESS OF THE `noc_index` OF THE SENDER!!!

*/

////////////////////////////////////////////////
// Data structures, types, enums, and constants
////////////////////////////////////////////////

enum SenderState : uint8_t {
    SENDER_DONE = 0,

    // we are ready to tell the worker(s) that the buffer is available for writing into
    SENDER_SIGNALING_WORKER,

    // we are waiting for the payload to arrive in L1; we are checking local semaphore for worker
    // completion
    SENDER_WAITING_FOR_WORKER,

    // this state is enterred if the sender was able to send the payload but not the channel sync
    SENDER_SEND_CHANNEL_SYNC,

    // Sender channel is not connected to a worker and is waiting for a new connection
    SENDER_WAIT_WORKER_HANDSHAKE,

    // means we are waiting for ack from receiver that payload was received
    SENDER_WAITING_FOR_ETH,

};

enum ReceiverState : uint8_t {
    RECEIVER_DONE = 0,

    // Receiver is processing the packet, either writing it locally or forwarding to the next EDM
    // (toward next chip), or both
    RECEIVER_SENDING_PAYLOAD,

    // Enter this state after performing writes of the current packet as a sort of soft barrier
    // (for this channel only) so we can make progress on other channels while waiting for the
    // writes to flush
    RECEIVER_WAITING_FOR_WRITE_FLUSH,

    // means we are waitinf for a payload from sender
    RECEIVER_WAITING_FOR_ETH,
};


enum PacketLocalForwardType : uint8_t {
    PACKET_FORWARD_INVALID = 0x0,
    PACKET_FORWARD_LOCAL_ONLY = 0x1,
    PACKET_FORWARD_REMOTE_ONLY = 0x2,
    PACKET_FORWARD_LOCAL_AND_REMOTE = 0x3
};

static constexpr uint32_t SWITCH_INTERVAL = 4000;
static constexpr size_t ETH_BYTES_TO_WORDS_SHIFT = 4;
static constexpr size_t NUM_SENDER_CHANNELS = 2;
static constexpr size_t num_workers_ctor = 1;
static constexpr size_t num_messages_to_move_ctor_value = 1;
// Doesn't REALLY matter but for consistency I picked the next available ID
static constexpr size_t receiver_channel_id = NUM_SENDER_CHANNELS;
static constexpr size_t worker_info_offset_past_connection_semaphore = 32;

/////////////////////////////////////////////
//   SENDER SIDE HELPERS
/////////////////////////////////////////////

FORCE_INLINE void sender_notify_workers_if_buffer_available_sequence(
    tt::fabric::EdmChannelWorkerInterface &local_sender_worker_interface) {
    local_sender_worker_interface.clear_local_semaphore();
    local_sender_worker_interface.increment_worker_semaphore();
}

template <uint8_t SENDER_NUM_BUFFERS, uint8_t RECEIVER_NUM_BUFFERS>
void send_channel_sync(
    tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS> &sender_buffer_channel,
    tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS> &receiver_buffer_channel) {

    DPRINT << "EDMS scs to " << (uint32_t)receiver_buffer_channel.get_current_bytes_sent_address() << "\n";

    eth_send_bytes_over_channel_payload_only_unsafe(
        reinterpret_cast<size_t>(sender_buffer_channel.get_current_bytes_sent_address()),
        reinterpret_cast<size_t>(receiver_buffer_channel.get_current_bytes_sent_address()),
        sizeof(eth_channel_sync_t),
        sizeof(eth_channel_sync_t),
        sizeof(eth_channel_sync_t) >> ETH_BYTES_TO_WORDS_SHIFT);
}

template <uint8_t SENDER_NUM_BUFFERS, uint8_t RECEIVER_NUM_BUFFERS>
tt::fabric::SendStatus send_next_data(
    tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS> &sender_buffer_channel,
    tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS> &receiver_buffer_channel) {

    auto status = tt::fabric::SendStatus::NOT_SENT;

    ASSERT(!eth_txq_is_busy());

    status = tt::fabric::SendStatus::SENT_PAYLOAD_AND_SYNC;
    ASSERT(
        reinterpret_cast<size_t>(sender_buffer_channel.get_current_bytes_sent_address()) ==
        (reinterpret_cast<size_t>(sender_buffer_channel.get_current_buffer_address()) +
         reinterpret_cast<size_t>(sender_buffer_channel.get_current_max_eth_payload_size()) -
         (uint32_t)sizeof(eth_channel_sync_t)));
    *sender_buffer_channel.get_current_bytes_sent_address() = sender_buffer_channel.get_current_max_eth_payload_size();
    *sender_buffer_channel.get_current_bytes_acked_address() = 0;
    *sender_buffer_channel.get_current_src_id_address() = sender_buffer_channel.get_id();
    ASSERT(*sender_buffer_channel.get_current_src_id_address() < 2);

    // TODO: TUNING - experiment with only conditionally breaking the transfer up into multiple packets if we are
    //       a certain threshold less than full packet
    //       we can precompute this value even on host and pass it in so we can get away with a single integer
    //       compare
    //       NOTE: if we always send full packet, then we don't need the second branch below dedicated for
    //             channel sync
    ASSERT(tt::fabric::is_valid(*const_cast<tt::fabric::PacketHeader *>(reinterpret_cast<volatile tt::fabric::PacketHeader *>(sender_buffer_channel.get_current_buffer_address()))));
    const size_t payload_size = sender_buffer_channel.get_current_payload_plus_channel_sync_size();
    eth_send_bytes_over_channel_payload_only_unsafe(
        sender_buffer_channel.get_current_buffer_address(),
        receiver_buffer_channel.get_current_buffer_address(),  // get_remote_eth_buffer_address(),
        payload_size,
        payload_size,
        payload_size >> ETH_BYTES_TO_WORDS_SHIFT);

    bool sent_payload_and_channel_sync_in_one_shot =
        payload_size == sender_buffer_channel.get_channel_buffer_max_size_in_bytes();
    if (!sent_payload_and_channel_sync_in_one_shot) {
        // We weren't able to send the channel_sync_t in one shot with the payload so we need to send a second
        // packet
        // TODO: TUNING - consider busy waiting for a maximum amount of time
        if (!eth_txq_is_busy()) {
            send_channel_sync(sender_buffer_channel, receiver_buffer_channel);
        } else {
            status = tt::fabric::SendStatus::SENT_PAYLOAD_ONLY;
        }
    }

    // Note: We can only advance to the next buffer index if we have fully completed the send (both the payload and sync
    // messages)
    if (status == tt::fabric::SendStatus::SENT_PAYLOAD_AND_SYNC) {
        sender_buffer_channel.advance_buffer_index();
        receiver_buffer_channel.advance_buffer_index();
    }

    return status;
}

template <uint8_t SENDER_NUM_BUFFERS, uint8_t RECEIVER_NUM_BUFFERS>
FORCE_INLINE bool sender_noc_receive_payload_ack_check_sequence(
    tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS> &sender_buffer_channel,
    tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS> &receiver_buffer_channel) {
    return sender_buffer_channel.is_local_semaphore_full();
}

template <uint8_t SENDER_NUM_BUFFERS>
FORCE_INLINE void sender_eth_check_receiver_ack_sequence(
    tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS> &sender_buffer_channel,
    tt::fabric::EdmChannelWorkerInterface &sender_worker_interface) {
    sender_buffer_channel.eth_clear_sender_channel_ack();

    sender_notify_workers_if_buffer_available_sequence(sender_worker_interface);
}

/////////////////////////////////////////////
//   RECEIVER SIDE HELPERS
/////////////////////////////////////////////

template <uint8_t RECEIVER_NUM_BUFFERS>
FORCE_INLINE bool new_unacknowledged_packet_avilable_on_reciever_channel(
    tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS> &local_receiver_channel) {
    return local_receiver_channel.eth_bytes_are_available_on_channel();
}

/*
 * Acting the receiver, we are looking at our receiver channel and acking the sender who sent us the latest packet.
 * Doesn't check to see if indeed a new message is available. It's assumed the caller has handled that separately.
 */
// MUST CHECK !is_eth_txq_busy() before calling
template <size_t NUM_SENDER_CHANNELS, uint8_t SENDER_NUM_BUFFERS, uint8_t RECEIVER_NUM_BUFFERS>
void receiver_send_received_ack(
    std::array<tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS> &remote_sender_channels,
    tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS> &local_receiver_buffer_channel) {
    // Set the acknowledgement bits. We have a different location than the

    const auto src_id = *local_receiver_buffer_channel.get_current_src_id_address();
    ASSERT(src_id < NUM_SENDER_CHANNELS);
    auto &sender_buffer_channel = remote_sender_channels[src_id];
    ASSERT(
        reinterpret_cast<size_t>(sender_buffer_channel.get_current_bytes_sent_address()) ==
        reinterpret_cast<size_t>(sender_buffer_channel.get_current_buffer_address()) +
            reinterpret_cast<size_t>(sender_buffer_channel.get_current_max_eth_payload_size()) -
            sizeof(eth_channel_sync_t));

    const size_t local_ack_channel_sync_src_addr =
        local_receiver_buffer_channel.get_eth_transaction_ack_word_addr() + (src_id * sizeof(eth_channel_sync_t));
    reinterpret_cast<volatile eth_channel_sync_t *>(local_ack_channel_sync_src_addr)->bytes_sent =
        *local_receiver_buffer_channel.get_current_bytes_sent_address();
    reinterpret_cast<volatile eth_channel_sync_t *>(local_ack_channel_sync_src_addr)->receiver_ack = 1;
    reinterpret_cast<volatile eth_channel_sync_t *>(local_ack_channel_sync_src_addr)->src_id =
        *local_receiver_buffer_channel.get_current_src_id_address();

    // Make sure we don't alias the erisc_info eth_channel_sync_t
    ASSERT(
        reinterpret_cast<volatile eth_channel_sync_t *>(local_receiver_buffer_channel.get_current_bytes_sent_address())
            ->bytes_sent != 0);
    ASSERT(
        reinterpret_cast<volatile eth_channel_sync_t *>(local_receiver_buffer_channel.get_current_bytes_sent_address())
            ->receiver_ack == 0);

    DPRINT << "EDMS rsc to " << (uint32_t)sender_buffer_channel.get_current_bytes_sent_address() << "\n";

    ASSERT(!eth_txq_is_busy());
    internal_::eth_send_packet_unsafe(
        0,
        local_ack_channel_sync_src_addr >> 4,
        ((uint32_t)(sender_buffer_channel.get_current_bytes_sent_address())) >> 4,
        1);
}

// MUST CHECK !is_eth_txq_busy() before calling
template <size_t NUM_SENDER_CHANNELS, uint8_t SENDER_NUM_BUFFERS, uint8_t RECEIVER_NUM_BUFFERS>
FORCE_INLINE void receiver_send_completion_ack(
    std::array<tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS> &remote_sender_channels,
    tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS> &local_receiver_buffer_channel) {
    volatile auto local_bytes_sent_addr = local_receiver_buffer_channel.get_current_bytes_sent_address();
    volatile auto local_src_id_ptr = local_receiver_buffer_channel.get_current_src_id_address();

    auto src_sender_channel = *local_src_id_ptr;
    *(local_bytes_sent_addr) = 0;
    *(local_receiver_buffer_channel.get_current_bytes_acked_address()) = 0;
    ASSERT(src_sender_channel < NUM_SENDER_CHANNELS);

    DPRINT << "EDMS rsc to " << (uint32_t)remote_sender_channels[src_sender_channel].get_current_bytes_sent_address() << "\n";

    ASSERT(!eth_txq_is_busy());
    internal_::eth_send_packet_unsafe(
        0,
        (uint32_t)(local_bytes_sent_addr) >> 4,
        (uint32_t)(remote_sender_channels[src_sender_channel].get_current_bytes_sent_address()) >> 4,
        1);

    local_receiver_buffer_channel.advance_buffer_index();
    remote_sender_channels[src_sender_channel].advance_buffer_index();
}


PacketLocalForwardType get_packet_local_forward_type(const volatile tt::fabric::PacketHeader &packet_header) {
    const bool local_chip_is_packet_destination = packet_must_be_consumed_locally(packet_header);
    const bool packet_needs_forwarding = packet_must_be_forwarded_to_next_chip(packet_header);
    PacketLocalForwardType forward_type =
        static_cast<PacketLocalForwardType>(packet_needs_forwarding << 1 | local_chip_is_packet_destination);
    return forward_type;
}

FORCE_INLINE bool can_forward_packet_completely(
    const volatile tt::fabric::PacketHeader &packet_header, tt::fabric::WorkerToFabricEdmSender &downstream_edm_interface) {
    auto forward_status = get_packet_local_forward_type(packet_header);
    bool can_send = true;
    switch (forward_status) {
        case PACKET_FORWARD_INVALID: return false;
        case PACKET_FORWARD_LOCAL_ONLY: return true;

        case PACKET_FORWARD_REMOTE_ONLY:
        case PACKET_FORWARD_LOCAL_AND_REMOTE: return downstream_edm_interface.consumer_has_space();
        default: ASSERT(false); return false;
    };
}

// template <uint8_t NUM_BUFFERS>
tt::fabric::SendStatus receiver_forward_packet(
    volatile tt::fabric::PacketHeader *packet_start, tt::fabric::WorkerToFabricEdmSender &downstream_edm_interface) {
    // Just cache the packet_header - we don't really expect (or care) if contents change during this function.
    tt::fabric::PacketHeader const &packet_header = *const_cast<tt::fabric::PacketHeader *const>(packet_start);
    ASSERT(tt::fabric::is_valid(packet_header));
    auto forward_status = get_packet_local_forward_type(packet_header);

    switch (forward_status) {
        case PACKET_FORWARD_LOCAL_ONLY: {
            execute_chip_unicast_to_local_chip(packet_start);
            return tt::fabric::SendStatus::SENT_PAYLOAD_AND_SYNC;
        } break;

        case PACKET_FORWARD_REMOTE_ONLY: {
            return forward_payload_to_downstream_edm(packet_start, downstream_edm_interface);
        } break;

        case PACKET_FORWARD_LOCAL_AND_REMOTE: {
            ASSERT(packet_header.chip_send_type == tt::fabric::ChipSendType::CHIP_MULTICAST);
            // TODO: make local chip write non-blocking
            execute_chip_unicast_to_local_chip(packet_start);
            return forward_payload_to_downstream_edm(packet_start, downstream_edm_interface);
        } break;

        case PACKET_FORWARD_INVALID:
        default: ASSERT(false); return tt::fabric::SendStatus::ERROR;
    };
}

////////////////////////////////////
////////////////////////////////////
//  Main Control Loop
////////////////////////////////////
////////////////////////////////////
template <uint8_t RECEIVER_NUM_BUFFERS, uint8_t SENDER_NUM_BUFFERS>
bool run_sender_channel_state_machine_step(
    tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS> &local_sender_channel,
    tt::fabric::EdmChannelWorkerInterface &local_sender_channel_worker_interface,
    tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS> &remote_receiver_channel,
    SenderState *const sender_state_out) {
    bool incr_sender_channel_index = true;
    switch (*sender_state_out) {
        case SenderState::SENDER_WAITING_FOR_WORKER: {
            bool able_to_send = local_sender_channel_worker_interface.has_payload() && !eth_txq_is_busy() &&
                                local_sender_channel.eth_is_receiver_channel_send_done();
            if (able_to_send) {
                DPRINT << "EDM send\n";
                auto send_status = send_next_data(local_sender_channel, remote_receiver_channel);
                // TODO: align the enums and state values so I can just do
                // sender_states[sender_channel_index] += send_status :)
                ASSERT(send_status != tt::fabric::SendStatus::ERROR);
                *sender_state_out =
                    send_status == tt::fabric::SendStatus::NOT_SENT            ? SenderState::SENDER_WAITING_FOR_WORKER
                    : send_status == tt::fabric::SendStatus::SENT_PAYLOAD_ONLY ? SenderState::SENDER_SEND_CHANNEL_SYNC
                                                                               : SenderState::SENDER_WAITING_FOR_ETH;
                // Avoid any sort of starvation/bubbles so we only advance if we've sent the packet and channel sync
                // otherwise what can happen is we could start sending another large payload from the other channel
                // and not be able to send the channel sync for the packet we just sent, which overall negatively
                // impact latency
                incr_sender_channel_index = send_status != tt::fabric::SendStatus::SENT_PAYLOAD_ONLY;
            } else {
                if (local_sender_channel_worker_interface.has_worker_teardown_request()) {
                    local_sender_channel_worker_interface.teardown_connection();
                    *sender_state_out = SenderState::SENDER_WAIT_WORKER_HANDSHAKE;
                }
            }
        } break;

        case SenderState::SENDER_WAIT_WORKER_HANDSHAKE:
            if (local_sender_channel_worker_interface.connection_is_live()) {
                bool is_safe_to_receive_next_message = local_sender_channel.eth_is_receiver_channel_send_acked() ||
                                                       local_sender_channel.eth_is_receiver_channel_send_done();
                if (is_safe_to_receive_next_message) {
                    DPRINT << "EDM wkr con ntfy wrkr\n";
                    sender_notify_workers_if_buffer_available_sequence(local_sender_channel_worker_interface);
                    *sender_state_out = SenderState::SENDER_WAITING_FOR_WORKER;
                } else {
                    *sender_state_out = SenderState::SENDER_WAITING_FOR_ETH;
                }
            }
            break;

        case SenderState::SENDER_SEND_CHANNEL_SYNC: {
            bool can_send_channel_sync_without_blocking = !eth_txq_is_busy();
            if (can_send_channel_sync_without_blocking) {
                send_channel_sync(local_sender_channel, remote_receiver_channel);
                local_sender_channel.advance_buffer_index();
                remote_receiver_channel.advance_buffer_index();
                *sender_state_out = SenderState::SENDER_WAITING_FOR_ETH;
            }
        } break;

        case SenderState::SENDER_WAITING_FOR_ETH: {
            bool is_safe_to_receive_next_message = local_sender_channel.eth_is_receiver_channel_send_acked() ||
                                                   local_sender_channel.eth_is_receiver_channel_send_done();
            if (is_safe_to_receive_next_message) {
                // This also notifies workers in the same call
                sender_eth_check_receiver_ack_sequence(local_sender_channel, local_sender_channel_worker_interface);
                *sender_state_out = SenderState::SENDER_WAITING_FOR_WORKER;
            }
        } break;

        default: break;
    };

    return incr_sender_channel_index;
};

template <size_t RECEIVER_NUM_BUFFERS, size_t SENDER_NUM_BUFFERS, size_t NUM_SENDER_CHANNELS>
void run_receiver_channel_state_machine_step(
    tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS> &local_receiver_channel,
    std::array<tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS> &remote_sender_channnels,
    tt::fabric::WorkerToFabricEdmSender &downstream_edm_interface,
    ReceiverState *const receiver_state_out) {
    switch (*receiver_state_out) {
        case ReceiverState::RECEIVER_WAITING_FOR_ETH: {
            bool got_payload = local_receiver_channel.eth_bytes_are_available_on_channel();
            if (got_payload) {
                bool can_ack = !eth_txq_is_busy();
                if (can_ack) {
                    DPRINT << "EDMR got pkt @: " << (uint32_t)reinterpret_cast<volatile uint64_t *>(local_receiver_channel.get_current_packet_header()) << "\n";
                    DPRINT << "EDMR got pkt 0 : " << (uint64_t) reinterpret_cast<volatile uint64_t *>(local_receiver_channel.get_current_packet_header())[0] << "\n";
                    DPRINT << "EDMR got pkt 1: " << (uint64_t) reinterpret_cast<volatile uint64_t *>(local_receiver_channel.get_current_packet_header())[1] << "\n";
                    ASSERT(tt::fabric::is_valid(
                        *const_cast<tt::fabric::PacketHeader *>(local_receiver_channel.get_current_packet_header())));
                    receiver_send_received_ack(remote_sender_channnels, local_receiver_channel);
                    // TODO: PERF Need to add feature to let use perform local noc write and defer the forward to EDM
                    // if we are mcasting to the local chip and neighbours, but the downstream EDM isn't currently able
                    // to accept the packet
                    // ...
                    // but as a starting point we can do the dumb thing and just wait for space downstream
                    // before we do either.
                    *receiver_state_out = ReceiverState::RECEIVER_SENDING_PAYLOAD;
                    // TODO: PERF - SHORT CIRCUIT IF WE CAN TO NESXT STATE TO MINIMIZE LATENCY BUT CURRENTLY
                    //       A LITTLE CODE SIZE BOUND
                }
            }
        } break;

        case ReceiverState::RECEIVER_SENDING_PAYLOAD: {
            auto& packet_header = *local_receiver_channel.get_current_packet_header();
            bool can_send_to_all_local_chip_receivers =
                can_forward_packet_completely(packet_header, downstream_edm_interface);
            if (can_send_to_all_local_chip_receivers) {
                DPRINT << "EDMR writing pkt\n";
                receiver_forward_packet(local_receiver_channel.get_current_packet_header(), downstream_edm_interface);
                *receiver_state_out = ReceiverState::RECEIVER_WAITING_FOR_WRITE_FLUSH;
            }
        } break;

        case ReceiverState::RECEIVER_WAITING_FOR_WRITE_FLUSH: {
            bool writes_flushed = ncrisc_noc_nonposted_writes_sent(noc_index);
            if (writes_flushed) {
                bool can_send_ack_without_blocking = !eth_txq_is_busy();
                if (can_send_ack_without_blocking) {
                    receiver_send_completion_ack(remote_sender_channnels, local_receiver_channel);
                    *receiver_state_out = ReceiverState::RECEIVER_WAITING_FOR_ETH;
                }
            }
        } break;

        default: break;
    };
};


/* Termination signal handling*/
FORCE_INLINE bool got_immediate_termination_signal(volatile tt::fabric::TerminationSignal *termination_signal_ptr) {
    return *termination_signal_ptr == tt::fabric::TerminationSignal::IMMEDIATELY_TERMINATE;
}
FORCE_INLINE bool got_graceful_termination_signal(volatile tt::fabric::TerminationSignal *termination_signal_ptr) {
    return *termination_signal_ptr == tt::fabric::TerminationSignal::GRACEFULLY_TERMINATE;
}
FORCE_INLINE bool got_termination_signal(volatile tt::fabric::TerminationSignal *termination_signal_ptr) {
    return got_immediate_termination_signal(termination_signal_ptr) ||
           got_graceful_termination_signal(termination_signal_ptr);
}

template <size_t RECEIVER_NUM_BUFFERS, size_t SENDER_NUM_BUFFERS, size_t NUM_SENDER_CHANNELS>
bool all_channels_drained(tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS> &local_receiver_channel,
                          std::array<tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS> &local_sender_channels) {
    // Unfortunately have to do this for now instead of only conditionally checking
    // each undrained channel due to code size issues...
    return local_sender_channels[0].all_buffers_drained() && local_sender_channels[1].all_buffers_drained() &&
           local_receiver_channel.all_buffers_drained();
}

/*
 * Main control loop for fabric EDM. Run indefinitely until a termination signal is received
 *
 * Every loop iteration visit a sender channel and the receiver channel. Switch between sender
 * channels every iteration unless it is unsafe/undesirable to do so (e.g. for performance reasons).
 */
template <size_t RECEIVER_NUM_BUFFERS, size_t SENDER_NUM_BUFFERS, size_t NUM_SENDER_CHANNELS>
void run_fabric_edm_main_loop(
    tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS> &local_receiver_channel,
    std::array<tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS> &local_sender_channels,
    std::array<tt::fabric::EdmChannelWorkerInterface, NUM_SENDER_CHANNELS> &local_sender_channel_worker_interfaces,
    tt::fabric::WorkerToFabricEdmSender &downstream_edm_noc_interface,
    std::array<tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS> &remote_sender_channels,
    tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS> &remote_receiver_channel,
    volatile tt::fabric::TerminationSignal *termination_signal_ptr) {
    std::array<SenderState, NUM_SENDER_CHANNELS> sender_states = {
        SenderState::SENDER_WAIT_WORKER_HANDSHAKE, SenderState::SENDER_WAIT_WORKER_HANDSHAKE};
    ReceiverState receiver_state = ReceiverState::RECEIVER_WAITING_FOR_ETH;
    size_t sender_channel_index = 0;
    size_t did_nothing_count = 0;
    *termination_signal_ptr = tt::fabric::TerminationSignal::KEEP_RUNNING;

    while (!got_immediate_termination_signal(termination_signal_ptr)) {
        if (got_graceful_termination_signal(termination_signal_ptr)) {
            DPRINT << "EDM Graceful termination\n";
            bool all_drained = all_channels_drained<RECEIVER_NUM_BUFFERS, SENDER_NUM_BUFFERS, NUM_SENDER_CHANNELS>(
                local_receiver_channel, local_sender_channels);

            if (all_drained) {
                return;
            }
        }

        //     // TODO
        auto &local_sender_channel = local_sender_channels[sender_channel_index];
        auto &local_sender_channel_worker_interface = local_sender_channel_worker_interfaces[sender_channel_index];
        // There are some cases, mainly for performance, where we don't want to switch between sender channels
        // so we interoduce this to provide finer grain control over when we disable the automatic switching
        bool incr_sender_channel_index = run_sender_channel_state_machine_step(
            local_sender_channel,
            local_sender_channel_worker_interface,
            remote_receiver_channel,
            &(sender_states[sender_channel_index]));
        if (incr_sender_channel_index) {
            // TODO: this can probably be optimized
            sender_channel_index = 1 - sender_channel_index;
        }

        run_receiver_channel_state_machine_step<RECEIVER_NUM_BUFFERS, SENDER_NUM_BUFFERS, NUM_SENDER_CHANNELS>(
            local_receiver_channel, remote_sender_channels, downstream_edm_noc_interface, &receiver_state);

        if (did_nothing_count++ > SWITCH_INTERVAL) {
            did_nothing_count = 0;
            run_routing();
        }
    }
    DPRINT << "EDM Terminating\n";
}

void kernel_main() {
    //
    // COMMON CT ARGS (not specific to sender or receiver)
    //
    static constexpr bool is_handshake_sender = get_compile_time_arg_val(0) != 0;
    static constexpr size_t handshake_addr = get_compile_time_arg_val(1);
    *reinterpret_cast<volatile uint32_t*>(handshake_addr) = 0;
    auto eth_transaction_ack_word_addr = handshake_addr + sizeof(eth_channel_sync_t);

    if constexpr (is_handshake_sender) {
        // DPRINT << "EDM Starting handshake as sender\n";
        erisc::datamover::handshake::sender_side_start(handshake_addr);
    } else {
        // DPRINT << "EDM Starting handshake as receiver\n";
        erisc::datamover::handshake::receiver_side_start(handshake_addr);
    }

    // the size of one of the buffers within a sender channel
    // For example if `channel_buffer_size` = 4k, with `SENDER_NUM_BUFFERS` = 2
    // then the total amount of buffering for that
    static constexpr size_t channel_buffer_size = get_compile_time_arg_val(2);

    static constexpr size_t SENDER_NUM_BUFFERS = get_compile_time_arg_val(3);
    static constexpr size_t RECEIVER_NUM_BUFFERS = get_compile_time_arg_val(4);
    static constexpr size_t local_sender_0_channel_address = get_compile_time_arg_val(5);
    static constexpr size_t local_sender_channel_0_connection_info_addr = get_compile_time_arg_val(6);
    static constexpr size_t local_sender_1_channel_address = get_compile_time_arg_val(7);
    static constexpr size_t local_sender_channel_1_connection_info_addr = get_compile_time_arg_val(8);
    static constexpr size_t local_receiver_channel_buffer_address = get_compile_time_arg_val(9);
    static constexpr size_t remote_receiver_channel_buffer_address = get_compile_time_arg_val(10);
    static constexpr size_t remote_sender_0_channel_address = get_compile_time_arg_val(11);
    static constexpr size_t remote_sender_1_channel_address = get_compile_time_arg_val(12);

    // TODO: CONVERT TO SEMAPHORE
    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::fabric::TerminationSignal *>(get_compile_time_arg_val(13));

    static_assert(SENDER_NUM_BUFFERS > 0, "compile time argument [1]: SENDER_NUM_BUFFERS must be > 0");
    static_assert(RECEIVER_NUM_BUFFERS > 0, "compile time argument [2]: RECEIVER_NUM_BUFFERS must be > 0");

    size_t arg_idx = 0;
    ///////////////////////
    // Common runtime args:
    ///////////////////////

    const size_t local_sender_channel_0_connection_semaphore_addr =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));
    const size_t local_sender_channel_1_connection_semaphore_addr =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));

    // unused - can later remove
    const size_t local_sender_channel_0_connection_buffer_index_addr =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));

    const size_t local_sender_channel_1_connection_buffer_index_addr =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));


    // downstream EDM semaphore location
    const bool has_downstream_edm_buffer_connection = get_arg_val<uint32_t>(arg_idx++) != 0;
    const auto downstream_edm_buffer_base_address = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_noc_y = get_arg_val<uint32_t>(arg_idx++);

    // remote address for flow control
    const auto downstream_edm_semaphore_id = get_arg_val<uint32_t>(arg_idx++);  // TODO: Convert to semaphore ID
    const auto downstream_edm_worker_registration_address =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));
    const auto downstream_edm_worker_location_info_address = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_noc_interface_buffer_index_local_addr = get_arg_val<uint32_t>(arg_idx++);

    // Receiver channels local semaphore for managing flow control with the downstream EDM.
    // The downstream EDM should be sending semaphore updates to this address any time it can
    // accept a new message
    const auto edm_forwarding_semaphore_address =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));

    ////////////////////////
    // Sender runtime args
    ////////////////////////
    auto sender0_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t *>(
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++)));
    auto sender1_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t *>(
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++)));
    *sender0_worker_semaphore_ptr = 0;
    *sender1_worker_semaphore_ptr = 0;

    //////////////////////////////
    //////////////////////////////
    //        Object Setup
    //////////////////////////////
    //////////////////////////////

    auto const &local_sender_buffer_addresses =
        std::array<size_t, NUM_SENDER_CHANNELS>{local_sender_0_channel_address, local_sender_1_channel_address};
    auto const &remote_sender_buffer_addresses =
        std::array<size_t, NUM_SENDER_CHANNELS>{remote_sender_0_channel_address, remote_sender_1_channel_address};
    std::array<tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS> remote_sender_channels;
    std::array<tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS> local_sender_channels;
    std::array<tt::fabric::EdmChannelWorkerInterface, NUM_SENDER_CHANNELS> local_sender_channel_worker_interfaces;
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_flow_control_semaphores = {
        reinterpret_cast<size_t>(sender0_worker_semaphore_ptr), reinterpret_cast<size_t>(sender1_worker_semaphore_ptr)};
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_live_semaphore_addresses = {
        local_sender_channel_0_connection_semaphore_addr, local_sender_channel_1_connection_semaphore_addr};
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_info_addresses = {
        local_sender_channel_0_connection_info_addr, local_sender_channel_1_connection_info_addr};
    auto downstream_edm_noc_interface =
        has_downstream_edm_buffer_connection
            ? tt::fabric::WorkerToFabricEdmSender(
                  downstream_edm_noc_x,
                  downstream_edm_noc_y,
                  downstream_edm_buffer_base_address,
                  SENDER_NUM_BUFFERS,
                  downstream_edm_semaphore_id,
                  downstream_edm_worker_registration_address,  // edm_connection_handshake_addr,
                  downstream_edm_worker_location_info_address,
                  channel_buffer_size,
                  local_sender_channel_1_connection_buffer_index_addr, // our downstream is channel 1
                  reinterpret_cast<volatile uint32_t *const>(edm_forwarding_semaphore_address),
                  downstream_noc_interface_buffer_index_local_addr)
            : tt::fabric::WorkerToFabricEdmSender();

    auto local_receiver_channel = tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>(
        local_receiver_channel_buffer_address,
        channel_buffer_size,
        tt::fabric::header_size_bytes,
        eth_transaction_ack_word_addr,  // Assume for receiver channel, this address points to a chunk of memory that
                                        // can fit 2 eth_channel_syncs cfor ack
        receiver_channel_id);
    auto remote_receiver_channel = tt::fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>(
        remote_receiver_channel_buffer_address,
        channel_buffer_size,
        tt::fabric::header_size_bytes,
        eth_transaction_ack_word_addr,  // Assume for receiver channel, this address points to a chunk of memory that
                                        // can fit 2 eth_channel_syncs cfor ack
        receiver_channel_id);

    uint32_t args_offset = 0;

    for (uint8_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        new (&local_sender_channels[i]) tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>(
            local_sender_buffer_addresses[i],
            channel_buffer_size,
            tt::fabric::header_size_bytes,
            0,  // For sender channels there is no eth_transaction_ack_word_addr because they don't send acks
            i);
        new (&remote_sender_channels[i]) tt::fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>(
            remote_sender_buffer_addresses[i],
            channel_buffer_size,
            tt::fabric::header_size_bytes,
            0,  // For sender channels there is no eth_transaction_ack_word_addr because they don't send acks
            i);

        auto connection_live_semaphore_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t *const>(local_sender_connection_live_semaphore_addresses[i]);
        auto connection_worker_info_ptr = reinterpret_cast<volatile tt::fabric::EDMChannelWorkerLocationInfo *>(
            local_sender_connection_info_addresses[i]);
        new (&local_sender_channel_worker_interfaces[i]) tt::fabric::EdmChannelWorkerInterface(
            connection_worker_info_ptr,  // worker_location_info_ptr,
            reinterpret_cast<volatile tt_l1_ptr uint32_t *const>(
                local_sender_flow_control_semaphores[i]),  // local_semaphore_address,
            reinterpret_cast<volatile tt_l1_ptr uint32_t *const>(connection_live_semaphore_ptr));
    }

    if (has_downstream_edm_buffer_connection) {
        downstream_edm_noc_interface.open();
    }

    if constexpr (is_handshake_sender) {
        // DPRINT << "EDM Finishing handshake as sender\n";
        erisc::datamover::handshake::sender_side_finish(handshake_addr);
    } else {
        // DPRINT << "EDM Finishing handshake as receiver\n";
        erisc::datamover::handshake::receiver_side_finish(handshake_addr);
    }
    DPRINT << "EDM Done handshake\n";

    DPRINT << "EDM Core y|x " << (uint32_t)((my_y[0] << 16) | my_x[0]) << "\n";
    // DPRINT << "EDM Connection address0 " << (uint32_t)local_sender_channel_worker_interfaces[0].connection_live_semaphore << "\n";
    // DPRINT << "EDM Connection address1 " << (uint32_t)local_sender_channel_worker_interfaces[1].connection_live_semaphore << "\n";


    //////////////////////////////
    //////////////////////////////
    //        MAIN LOOP
    //////////////////////////////
    //////////////////////////////
    run_fabric_edm_main_loop<RECEIVER_NUM_BUFFERS, SENDER_NUM_BUFFERS, NUM_SENDER_CHANNELS>(
        local_receiver_channel,
        local_sender_channels,
        local_sender_channel_worker_interfaces,
        downstream_edm_noc_interface,
        remote_sender_channels,
        remote_receiver_channel,
        termination_signal_ptr);


    DPRINT << "EDM DONE\n";
    WAYPOINT("DONE");
}

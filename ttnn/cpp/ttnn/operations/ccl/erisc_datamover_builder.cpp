#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn {
namespace ccl {

EriscDatamoverBuilder::ChannelBufferSpec::ChannelBufferSpec(
    bool is_sender,
    uint32_t worker_semaphore_id,
    uint32_t num_eth_messages_to_forward,
    uint32_t channel,
    uint32_t num_buffers,
    std::vector<ccl::WorkerXY> const& worker_coords,
    uint32_t largest_message_size_bytes) :
    worker_coords(worker_coords),
    worker_semaphore_id(worker_semaphore_id),
    num_eth_messages_to_forward(num_eth_messages_to_forward),
    channel(channel),
    largest_message_size_bytes(largest_message_size_bytes),
    is_sender(is_sender) {}

void EriscDatamoverBuilder::push_back_channel_args(std::vector<uint32_t>& args, ChannelBufferSpec const& channel) const {
    args.push_back(this->local_buffer_addresses.at(channel.channel));
    args.push_back(channel.num_eth_messages_to_forward);
    if (channel.largest_message_size_bytes > 0) {
        args.push_back(std::min<uint32_t>(channel.largest_message_size_bytes, this->eth_buffer_size_bytes));
        if (channel.largest_message_size_bytes < this->eth_buffer_size_bytes) {
            log_trace(tt::LogOp, "Trimming buffer size for channel {} to {}", channel.channel, args.back());
        }
    } else {
        args.push_back(this->eth_buffer_size_bytes);
    }
    args.push_back(this->local_semaphore_addresses.at(channel.channel));
    args.push_back(channel.worker_semaphore_id);
    args.push_back(channel.worker_coords.size());
    for (auto const& worker_coord : channel.worker_coords) {
        args.push_back(worker_coord.to_uint32());
    }
}

EriscDatamoverBuilder::EriscDatamoverBuilder(
    uint32_t eth_buffer_size,
    uint32_t handshake_addr,
    std::vector<uint32_t> const& local_semaphore_addresses,
    std::vector<uint32_t> const& local_buffer_addresses,
    ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode,
    ccl::EriscDataMoverTerminationMode termination_mode,
    std::size_t num_buffers_per_channel,
    chip_id_t chip_id) :
    local_semaphore_addresses(local_semaphore_addresses),
    local_buffer_addresses(local_buffer_addresses),
    eth_buffer_size_bytes(eth_buffer_size),
    handshake_addr(handshake_addr),
    num_channel_buffers(local_buffer_addresses.size()),
    buffer_sharing_mode(buffer_sharing_mode),
    num_buffers_per_channel(num_buffers_per_channel),
    termination_mode(termination_mode),
    enable_sender(false),
    enable_receiver(false),
    num_senders(0),
    num_receivers(0),
    chip_id(chip_id) {

    TT_ASSERT(num_buffers_per_channel > 0);
    TT_ASSERT(local_buffer_addresses.size() == local_semaphore_addresses.size());
    active_channels.reserve(num_channel_buffers);
    log_trace(tt::LogOp, "EriscDatamoverBuilder:");
    for (auto const& addr : local_semaphore_addresses) {
        TT_ASSERT(addr > 0);
        TT_ASSERT(addr % 16 == 0);
        log_trace(tt::LogOp, "\tsemaphore_address: {}", addr);
    }
    for (auto const& addr : local_buffer_addresses) {
        TT_ASSERT(addr > 0);
        TT_ASSERT(addr % 16 == 0);
        log_trace(tt::LogOp, "\tbuffer_address: {}", addr);
    }
}

EriscDatamoverBuilder::ChannelBufferInterface EriscDatamoverBuilder::add_sender_channel(
    uint32_t worker_semaphore_id,
    uint32_t num_eth_messages_to_forward,
    std::vector<ccl::WorkerXY> const& worker_coords,
    uint32_t expected_message_size_bytes) {
    this->enable_sender = true;
    this->num_senders++;
    auto channel = active_channels.size();
    active_channels.emplace_back(
        true, worker_semaphore_id, num_eth_messages_to_forward, channel, this->num_buffers_per_channel, worker_coords, expected_message_size_bytes);
    log_trace(tt::LogOp, "Adding sender channel:");
    log_trace(tt::LogOp, "\tworker_semaphore_id: {}", active_channels.back().worker_semaphore_id);
    log_trace(tt::LogOp, "\tnum_eth_messages_to_forward: {}", active_channels.back().num_eth_messages_to_forward);
    log_trace(tt::LogOp, "\tchannel: {}", active_channels.back().channel);
    log_trace(tt::LogOp, "\tis_sender: {}", active_channels.back().is_sender ? 1 : 0);
    log_trace(tt::LogOp, "\tbuffer_address: {}", local_buffer_addresses.at(channel));
    log_trace(tt::LogOp, "\tsemaphore_address: {}", local_semaphore_addresses.at(channel));
    log_trace(tt::LogOp, "\tnum_workers: {}", worker_coords.size());

    return ChannelBufferInterface{channel, local_buffer_addresses.at(channel), local_semaphore_addresses.at(channel)};
}

    // This function is used to set the maximum message size for a given channel. If the maximum
    // message size is < EDM channel buffer size, then the buffer size passed to the EDM for this channel
    // will be trimmed be no larger than the largest message to save on unnecessary eth bandwidth.
void EriscDatamoverBuilder::set_max_message_size_bytes(std::size_t channel, std::size_t max_message_size_bytes) {
    active_channels.at(channel).largest_message_size_bytes = std::max<uint32_t>(active_channels.at(channel).largest_message_size_bytes, max_message_size_bytes);
}

EriscDatamoverBuilder::ChannelBufferInterface EriscDatamoverBuilder::add_receiver_channel(
    uint32_t worker_semaphore_id,
    uint32_t num_eth_messages_to_forward,
    std::vector<ccl::WorkerXY> const& worker_coords,
    uint32_t expected_message_size_bytes) {
    this->enable_receiver = true;
    this->num_receivers++;
    auto channel = active_channels.size();
    active_channels.emplace_back(
        false, worker_semaphore_id, num_eth_messages_to_forward, channel, this->num_buffers_per_channel, worker_coords, expected_message_size_bytes);
    log_trace(tt::LogOp, "Adding receiver channel:");
    log_trace(tt::LogOp, "\tworker_semaphore_id: {}", active_channels.back().worker_semaphore_id);
    log_trace(tt::LogOp, "\tnum_eth_messages_to_forward: {}", active_channels.back().num_eth_messages_to_forward);
    log_trace(tt::LogOp, "\tchannel: {}", active_channels.back().channel);
    log_trace(tt::LogOp, "\tnum_workers: {}", worker_coords.size());
    log_trace(tt::LogOp, "\tis_sender: {}", active_channels.back().is_sender ? 1 : 0);
    return ChannelBufferInterface{channel, local_buffer_addresses.at(channel), local_semaphore_addresses.at(channel)};
}

std::vector<uint32_t> EriscDatamoverBuilder::emit_compile_time_args() const {
    return std::vector<uint32_t>{
        this->buffer_sharing_mode,
        this->termination_mode,
        chip_id
        };
}

std::vector<uint32_t> EriscDatamoverBuilder::emit_runtime_args() const {
    log_info(tt::LogOp, "EriscDatamoverBuilder::emit_runtime_args()");
    std::vector<uint32_t> args;
    uint32_t size = 8 + active_channels.size() * 6;
    for (auto const& channel : active_channels) {
        size += channel.worker_coords.size();
    }
    args.reserve(size);

    // is_handshake_master
    bool is_handshake_master = this->num_senders > 0 && active_channels.at(0).is_sender;
    args.push_back(static_cast<uint32_t>(is_handshake_master));
    // Handshake address
    args.push_back(handshake_addr);

    args.push_back(static_cast<uint32_t>(this->num_senders));
    args.push_back(static_cast<uint32_t>(this->num_receivers));
    args.push_back(static_cast<uint32_t>(this->num_buffers_per_channel));

    bool senders_below_receivers = active_channels.size() == 0 || this->active_channels.front().is_sender;

    // Receiver channel args
    uint32_t receiver_channels_offset = senders_below_receivers ? this->num_senders : 0;
    args.push_back(receiver_channels_offset);
    for (auto const& channel : this->active_channels) {
        if (channel.is_sender) {
            continue;
        }
        push_back_channel_args(args, channel);
    }

    // Sender channel args
    uint32_t sender_channels_offset = senders_below_receivers ? 0 : this->num_receivers;
    args.push_back(sender_channels_offset);
    for (auto const& channel : this->active_channels) {
        if (!channel.is_sender) {
            continue;
        }
        push_back_channel_args(args, channel);
    }

    return args;
}

void EriscDatamoverBuilder::dump_to_log() const {
    log_info(tt::LogOp, "Dumping args to log");
    auto const& rt_args = this->emit_runtime_args();
    log_trace(tt::LogOp, "EDM RT Args:");
    for (auto const& arg : rt_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
};

uint32_t EriscDatamoverBuilder::get_eth_buffer_size_bytes() const {
    return this->eth_buffer_size_bytes;
}

std::vector<EriscDatamoverBuilder::ChannelBufferSpec> const& EriscDatamoverBuilder::get_active_channels() const { return this->active_channels; }

};  // namespace ccl
};  // namespace ttnn

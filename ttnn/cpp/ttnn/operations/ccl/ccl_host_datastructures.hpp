// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eth_l1_address_map.h"
#include "tensor/tensor_impl.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <limits>

namespace ttnn {
namespace ccl {

enum Topology { Ring = 0, Linear = 1, Meash = 2 };

struct EriscDatamoverConfig {
    static constexpr std::size_t total_l1_buffer_space =
        eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    static constexpr std::size_t usable_l1_base_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    static constexpr std::size_t semaphore_size = 32;
    static constexpr std::size_t handshake_location_size = 16;  // ethernet word size
    // The EDM uses this fixed address as a source for a first level ack sent from receiver -> sender
    // side. We have this dedicated source address to avoid a race between first and second level ack
    // where second level ack overwrites the first level ack in L1 before the first one is sent out.
    // The memory contents in L1 will be {1, 1, x, x}. By having this dedicated source memory, we
    // avoid the race
    static constexpr std::size_t edm_receiver_first_level_ack_source_word_size = 16;  // ethernet word size
    static constexpr std::size_t eth_word_size_bytes = 16;

    static uint32_t get_edm_handshake_address() { return usable_l1_base_address; }
    static uint32_t get_semaphores_base_address(std::size_t num_edm_channels) {
        return usable_l1_base_address + (handshake_location_size * 3) + edm_receiver_first_level_ack_source_word_size;
    }
    static uint32_t get_buffers_base_address(std::size_t num_edm_channels) {
        uint32_t base_address =tt::round_up(
            get_semaphores_base_address(num_edm_channels) + num_edm_channels * semaphore_size, eth_word_size_bytes);
        TT_ASSERT(base_address % eth_word_size_bytes == 0);
        return base_address;
    }
    static uint32_t compute_buffer_size(std::size_t num_edm_channels, uint32_t page_size = eth_word_size_bytes) {
        page_size = std::max<uint32_t>(page_size, eth_word_size_bytes);
        TT_ASSERT(num_edm_channels > 0);
        uint32_t buffer_size =tt::round_down(
            (total_l1_buffer_space - get_buffers_base_address(num_edm_channels)) / (num_edm_channels), page_size);
        log_trace(tt::LogOp, "total_l1_buffer_space: {}", total_l1_buffer_space);
        log_trace(
            tt::LogOp, "get_buffers_base_address(num_edm_channels): {}", get_buffers_base_address(num_edm_channels));
        log_trace(
            tt::LogOp, "usable buffer space: {}", total_l1_buffer_space - get_buffers_base_address(num_edm_channels));
        log_trace(tt::LogOp, "num_edm_channels: {}", num_edm_channels);
        log_trace(tt::LogOp, "page_size: {}", page_size);

        log_trace(tt::LogOp, "Buffer size: {}", buffer_size);

        TT_ASSERT(buffer_size > 0 && buffer_size % page_size == 0);
        return buffer_size;
    }
};

struct CCLOpConfig {
   public:
    CCLOpConfig(
        const std::vector<Tensor>& input_tensors, const std::vector<Tensor>& output_tensors, Topology topology) :
        input_tensors(&input_tensors),
        output_tensors(&output_tensors),
        input_sharded(input_tensors.at(0).is_sharded()),
        output_sharded(output_tensors.at(0).is_sharded()),
        page_size(input_tensors.at(0).buffer()->page_size()),
        input_shard_size_bytes(
            input_tensors.at(0).is_sharded() ? static_cast<std::optional<uint32_t>>(
                                                   (input_tensors.at(0).buffer()->page_size() *
                                                    input_tensors.at(0).buffer()->shard_spec().tensor2d_shape[0] *
                                                    input_tensors.at(0).buffer()->shard_spec().tensor2d_shape[1]) /
                                                   input_tensors.at(0).shard_spec()->num_cores())
                                             : std::nullopt),
        output_shard_size_bytes(
            output_tensors.at(0).is_sharded() ? static_cast<std::optional<uint32_t>>(
                                                    (output_tensors.at(0).buffer()->page_size() *
                                                     output_tensors.at(0).buffer()->shard_spec().tensor2d_shape[0] *
                                                     output_tensors.at(0).buffer()->shard_spec().tensor2d_shape[1]) /
                                                    input_tensors.at(0).shard_spec()->num_cores())
                                              : std::nullopt),
        shard_grid_size(output_tensors.at(0).is_sharded() ? input_tensors.at(0).shard_spec()->num_cores() : 0),
        topology(topology) {
        TT_ASSERT(!this->is_input_sharded() || input_shard_size_bytes.has_value());
        TT_ASSERT(!this->is_output_sharded() || output_shard_size_bytes.has_value());
    }

    uint32_t get_input_shard_size_bytes() const {
        TT_ASSERT(input_shard_size_bytes.has_value());
        return input_shard_size_bytes.value();
    }
    uint32_t get_output_shard_size_bytes() const {
        TT_ASSERT(output_shard_size_bytes.has_value());
        return output_shard_size_bytes.value();
    }
    uint32_t get_page_size() const { return this->page_size; }
    Topology get_topology() const { return this->topology; }
    bool is_input_sharded() const { return this->input_sharded; }
    bool is_output_sharded() const { return this->output_sharded; }
    bool get_shard_grid_size() const { return this->shard_grid_size; }
    Tensor const& get_input_tensor(std::size_t i) const { return input_tensors->at(i); }
    Tensor const& get_output_tensor(std::size_t i) const { return output_tensors->at(i); }

   private:
    std::optional<uint32_t> input_shard_size_bytes;
    std::optional<uint32_t> output_shard_size_bytes;
    uint32_t page_size;
    uint32_t shard_grid_size;
    Topology topology;
    bool input_sharded;
    bool output_sharded;

    std::vector<Tensor> const* input_tensors;
    std::vector<Tensor> const* output_tensors;
};

class EriscDatamoverBuilder {
   private:
    struct ChannelBufferSpec {
        ChannelBufferSpec(
            bool is_sender,
            uint32_t worker_semaphore_address,
            uint32_t num_eth_messages_to_forward,
            uint32_t channel,
            std::vector<ccl::WorkerXY> const& worker_coords,
            uint32_t largest_message_size_bytes = 0) :
            worker_coords(worker_coords),
            worker_semaphore_address(worker_semaphore_address),
            num_eth_messages_to_forward(num_eth_messages_to_forward),
            channel(channel),
            largest_message_size_bytes(largest_message_size_bytes),
            is_sender(is_sender) {}

        std::vector<ccl::WorkerXY> const worker_coords;
        uint32_t worker_semaphore_address;
        uint32_t num_eth_messages_to_forward;
        uint32_t channel;
        uint32_t largest_message_size_bytes;
        bool is_sender;
    };

    void push_back_channel_args(std::vector<uint32_t>& args, ChannelBufferSpec const& channel) const {
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
        args.push_back(channel.worker_semaphore_address);
        args.push_back(channel.worker_coords.size());
        for (auto const& worker_coord : channel.worker_coords) {
            args.push_back(worker_coord.to_uint32());
        }
    }

    std::vector<ChannelBufferSpec> active_channels;
    std::vector<uint32_t> const local_semaphore_addresses;
    std::vector<uint32_t> const local_buffer_addresses;
    uint32_t eth_buffer_size_bytes;
    uint32_t handshake_addr;
    uint32_t const num_channel_buffers;
    ccl::EriscDataMoverBufferSharingMode const buffer_sharing_mode;
    ccl::EriscDataMoverTerminationMode const termination_mode;
    uint32_t num_senders;
    uint32_t num_receivers;

    bool enable_sender;
    bool enable_receiver;

   public:
    struct ChannelBufferInterface {
        std::size_t channel;
        uint32_t eth_buffer_l1_address;
        uint32_t eth_semaphore_l1_address;
    };

    EriscDatamoverBuilder(
        uint32_t eth_buffer_size,
        uint32_t handshake_addr,
        std::vector<uint32_t> const& local_semaphore_addresses,
        std::vector<uint32_t> const& local_buffer_addresses,
        ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode,
        ccl::EriscDataMoverTerminationMode termination_mode =
            ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED) :
        local_semaphore_addresses(local_semaphore_addresses),
        local_buffer_addresses(local_buffer_addresses),
        eth_buffer_size_bytes(eth_buffer_size),
        handshake_addr(handshake_addr),
        num_channel_buffers(local_buffer_addresses.size()),
        buffer_sharing_mode(buffer_sharing_mode),
        termination_mode(termination_mode),
        enable_sender(false),
        enable_receiver(false),
        num_senders(0),
        num_receivers(0) {
        TT_ASSERT(local_buffer_addresses.size() == local_semaphore_addresses.size());
        active_channels.reserve(num_channel_buffers);
        TT_ASSERT(eth_buffer_size_bytes < 163000);
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

    [[nodiscard]]
    ChannelBufferInterface add_sender_channel(
        uint32_t worker_semaphore_address,
        uint32_t num_eth_messages_to_forward,
        std::vector<ccl::WorkerXY> const& worker_coords,
        uint32_t expected_message_size_bytes = 0) {
        this->enable_sender = true;
        this->num_senders++;
        auto channel = active_channels.size();
        active_channels.emplace_back(
            true, worker_semaphore_address, num_eth_messages_to_forward, channel, worker_coords, expected_message_size_bytes);
        log_trace(tt::LogOp, "Adding sender channel:");
        log_trace(tt::LogOp, "\tworker_semaphore_address: {}", active_channels.back().worker_semaphore_address);
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
    void set_max_message_size_bytes(std::size_t channel, std::size_t max_message_size_bytes) {
        active_channels.at(channel).largest_message_size_bytes = std::max<uint32_t>(active_channels.at(channel).largest_message_size_bytes, max_message_size_bytes);
    }

    [[nodiscard]]
    ChannelBufferInterface add_receiver_channel(
        uint32_t worker_semaphore_address,
        uint32_t num_eth_messages_to_forward,
        std::vector<ccl::WorkerXY> const& worker_coords,
        uint32_t expected_message_size_bytes = 0) {
        this->enable_receiver = true;
        this->num_receivers++;
        auto channel = active_channels.size();
        active_channels.emplace_back(
            false, worker_semaphore_address, num_eth_messages_to_forward, channel, worker_coords, expected_message_size_bytes);
        log_trace(tt::LogOp, "Adding receiver channel:");
        log_trace(tt::LogOp, "\tworker_semaphore_address: {}", active_channels.back().worker_semaphore_address);
        log_trace(tt::LogOp, "\tnum_eth_messages_to_forward: {}", active_channels.back().num_eth_messages_to_forward);
        log_trace(tt::LogOp, "\tchannel: {}", active_channels.back().channel);
        log_trace(tt::LogOp, "\tnum_workers: {}", worker_coords.size());
        log_trace(tt::LogOp, "\tis_sender: {}", active_channels.back().is_sender ? 1 : 0);
        return ChannelBufferInterface{channel, local_buffer_addresses.at(channel), local_semaphore_addresses.at(channel)};
    }

    [[nodiscard]]
    std::vector<uint32_t> emit_compile_time_args() const {
        return std::vector<uint32_t>{
            static_cast<uint32_t>(this->enable_sender ? 1 : 0),
            static_cast<uint32_t>(this->enable_receiver ? 1 : 0),
            this->num_senders,
            this->num_receivers,
            this->buffer_sharing_mode,
            this->termination_mode};
    }

    [[nodiscard]]
    std::vector<uint32_t> emit_runtime_args() const {
        std::vector<uint32_t> args;
        uint32_t size = 3 + active_channels.size() * 6;
        for (auto const& channel : active_channels) {
            size += channel.worker_coords.size();
        }
        args.reserve(size);

        // Handshake address
        args.push_back(handshake_addr);

        bool senders_below_receivers = active_channels.size() == 0 || this->active_channels.front().is_sender;

        // Sender channel args
        uint32_t sender_channels_offset = senders_below_receivers ? 0 : this->num_receivers;
        args.push_back(sender_channels_offset);
        for (auto const& channel : this->active_channels) {
            if (!channel.is_sender) {
                continue;
            }
            push_back_channel_args(args, channel);
        }

        // Receiver channel args
        uint32_t receiver_channels_offset = senders_below_receivers ? this->num_senders : 0;
        args.push_back(receiver_channels_offset);
        for (auto const& channel : this->active_channels) {
            if (channel.is_sender) {
                continue;
            }
            push_back_channel_args(args, channel);
        }

        return args;
    }

    void dump_to_log() const {
        auto const& rt_args = this->emit_runtime_args();
        log_trace(tt::LogOp, "EDM RT Args:");
        for (auto const& arg : rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
    };

    [[nodiscard]]
    uint32_t get_eth_buffer_size_bytes() const {
        return this->eth_buffer_size_bytes;
    }

    std::vector<ChannelBufferSpec> const& get_active_channels() const { return this->active_channels; }
};

};  // namespace ccl
};  // namespace ttnn

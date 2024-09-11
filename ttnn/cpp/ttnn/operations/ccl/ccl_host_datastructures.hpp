// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eth_l1_address_map.h"
#include "ttnn/cpp/ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
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
    static constexpr std::size_t handshake_padding_multiple = 3;  // ethernet word size
    // The EDM uses this fixed address as a source for a first level ack sent from receiver -> sender
    // side. We have this dedicated source address to avoid a race between first and second level ack
    // where second level ack overwrites the first level ack in L1 before the first one is sent out.
    // The memory contents in L1 will be {1, 1, x, x}. By having this dedicated source memory, we
    // avoid the race
    static constexpr std::size_t edm_receiver_first_level_ack_source_word_size = 16;  // ethernet word size
    static constexpr std::size_t eth_channel_sync_size_bytes = 16;
    static constexpr std::size_t eth_word_size_bytes = 16;
    static constexpr bool enable_merged_payload_and_channel_sync = true;
    static std::size_t get_eth_channel_sync_size_bytes();
    static uint32_t get_edm_handshake_address();
    static std::size_t get_semaphores_region_size(std::size_t num_edm_channels);
    static std::size_t get_semaphores_region_start_offset(std::size_t num_edm_channels);
    static uint32_t get_semaphores_base_address(std::size_t num_edm_channels);
    static uint32_t get_buffers_region_start_offset(std::size_t num_edm_channels);
    static std::size_t get_eth_word_size();
    static uint32_t get_buffers_base_address(std::size_t num_edm_channels);
    static uint32_t compute_buffer_size(std::size_t num_edm_channels, std::size_t num_buffers_per_channel = 1, uint32_t page_size = eth_word_size_bytes);

};

struct CCLOpConfig {
   public:
    CCLOpConfig(
        std::vector<Tensor>& input_tensors, const std::vector<Tensor>& output_tensors, Topology topology);

    uint32_t get_page_size() const;
    Topology get_topology() const;
    bool is_input_sharded() const;
    bool is_output_sharded() const;
    bool get_shard_grid_size() const;
    Tensor const& get_input_tensor(std::size_t i) const;
    Tensor const& get_output_tensor(std::size_t i) const;
    std::map<string, string> emit_worker_defines() const;

   private:
    uint32_t page_size;
    uint32_t shard_grid_size;
    Topology topology;
    bool input_sharded;
    bool output_sharded;
    bool is_row_major;

    std::vector<Tensor> const* input_tensors;
    std::vector<Tensor> const* output_tensors;
};

class EriscDatamoverBuilder {
   private:
    struct ChannelBufferSpec {
        ChannelBufferSpec(
            bool is_sender,
            uint32_t worker_semaphore_id,
            uint32_t num_eth_messages_to_forward,
            uint32_t channel,
            uint32_t num_buffers,
            std::vector<ccl::WorkerXY> const& worker_coords,
            uint32_t largest_message_size_bytes = 0);

        std::vector<ccl::WorkerXY> const worker_coords;
        uint32_t worker_semaphore_id;
        uint32_t num_eth_messages_to_forward;
        uint32_t channel;
        uint32_t largest_message_size_bytes;
        uint32_t num_buffers;
        bool is_sender;
    };

    void push_back_channel_args(std::vector<uint32_t>& args, ChannelBufferSpec const& channel) const;

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
    std::size_t num_buffers_per_channel;
    chip_id_t chip_id;

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
        ccl::EriscDataMoverTerminationMode termination_mode = ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED,
        std::size_t num_buffers_per_channel = 1,
        chip_id_t chip_id = -1);

    [[nodiscard]]
    ChannelBufferInterface add_sender_channel(
        uint32_t worker_semaphore_id,
        uint32_t num_eth_messages_to_forward,
        std::vector<ccl::WorkerXY> const& worker_coords,
        uint32_t expected_message_size_bytes = 0);

    // This function is used to set the maximum message size for a given channel. If the maximum
    // message size is < EDM channel buffer size, then the buffer size passed to the EDM for this channel
    // will be trimmed be no larger than the largest message to save on unnecessary eth bandwidth.
    void set_max_message_size_bytes(std::size_t channel, std::size_t max_message_size_bytes);

    [[nodiscard]]
    ChannelBufferInterface add_receiver_channel(
        uint32_t worker_semaphore_id,
        uint32_t num_eth_messages_to_forward,
        std::vector<ccl::WorkerXY> const& worker_coords,
        uint32_t expected_message_size_bytes = 0);

    [[nodiscard]] std::vector<uint32_t> emit_compile_time_args() const;

    [[nodiscard]] std::vector<uint32_t> emit_runtime_args() const;

    void dump_to_log() const;

    [[nodiscard]] uint32_t get_eth_buffer_size_bytes() const;
    std::vector<ChannelBufferSpec> const& get_active_channels() const;
};

};  // namespace ccl
};  // namespace ttnn

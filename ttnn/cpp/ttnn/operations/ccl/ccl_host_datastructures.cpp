// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

using namespace tt::tt_metal;

namespace ttnn::ccl {

std::size_t EriscDatamoverConfig::get_eth_channel_sync_size_bytes() { return eth_channel_sync_size_bytes; }

uint32_t EriscDatamoverConfig::get_edm_handshake_address() const { return usable_l1_base_address; }

std::size_t EriscDatamoverConfig::get_semaphores_region_size(std::size_t num_edm_channels) {
    return (num_edm_channels * semaphore_size);
}
std::size_t EriscDatamoverConfig::get_semaphores_region_start_offset(std::size_t /*num_edm_channels*/) {
    return handshake_location_size + edm_receiver_first_level_ack_source_word_size;
}
uint32_t EriscDatamoverConfig::get_semaphores_base_address(std::size_t num_edm_channels) const {
    return usable_l1_base_address + get_semaphores_region_start_offset(num_edm_channels);
}
uint32_t EriscDatamoverConfig::get_buffers_region_start_offset(std::size_t num_edm_channels) {
    return get_semaphores_region_start_offset(num_edm_channels) + get_semaphores_region_size(num_edm_channels);
}
std::size_t EriscDatamoverConfig::get_eth_word_size() { return eth_word_size_bytes; }
uint32_t EriscDatamoverConfig::get_buffers_base_address(std::size_t num_edm_channels) const {
    uint32_t base_address =
        tt::round_up(usable_l1_base_address + get_buffers_region_start_offset(num_edm_channels), eth_word_size_bytes);
    TT_ASSERT(base_address % eth_word_size_bytes == 0);
    return base_address;
}
uint32_t EriscDatamoverConfig::compute_buffer_size(
    std::size_t num_edm_channels, std::size_t num_buffers_per_channel, uint32_t page_size) {
    page_size = std::max<uint32_t>(page_size, eth_word_size_bytes);
    TT_ASSERT(num_edm_channels > 0);
    std::size_t channel_sync_bytes_overhead = (enable_merged_payload_and_channel_sync * 16);
    std::size_t total_usable_space = total_l1_buffer_space - get_buffers_region_start_offset(num_edm_channels);
    std::size_t l1_per_buffer_region =
        (total_usable_space / (num_edm_channels * num_buffers_per_channel)) - channel_sync_bytes_overhead;
    uint32_t buffer_size = tt::round_down(l1_per_buffer_region, page_size);
    log_trace(tt::LogOp, "total_l1_buffer_space: {}", total_l1_buffer_space);
    log_trace(tt::LogOp, "get_buffers_base_address(num_edm_channels): {}", get_buffers_base_address(num_edm_channels));
    log_trace(tt::LogOp, "usable buffer space: {}", total_l1_buffer_space - get_buffers_base_address(num_edm_channels));
    log_trace(tt::LogOp, "num_edm_channels: {}", num_edm_channels);
    log_trace(tt::LogOp, "page_size: {}", page_size);

    log_trace(tt::LogOp, "Buffer size: {}", buffer_size);

    TT_ASSERT(page_size == 0 ? buffer_size == 0 : buffer_size % page_size == 0);
    return buffer_size;
}

CCLOpConfig::CCLOpConfig(
    std::vector<Tensor>& input_tensors, const std::vector<Tensor>& output_tensors, Topology topology) :
    topology(topology),
    input_sharded(input_tensors.at(0).is_sharded()),
    output_sharded(output_tensors.at(0).is_sharded()),
    is_row_major(input_tensors.at(0).layout() == Layout::ROW_MAJOR),
    df(tt::tt_metal::datatype_to_dataformat_converter(input_tensors.at(0).dtype())),
    input_tensors(&input_tensors),
    output_tensors(&output_tensors) {
    if (input_tensors.at(0).layout() == Layout::TILE) {
        this->tile = input_tensors.at(0).tensor_spec().tile();
        this->page_size = this->tile.get_tile_size(this->df);
        // this->page_size = input_tensors.at(0).buffer()->page_size();
    } else {
        this->page_size = input_tensors.at(0).buffer()->page_size();
    }
}

uint32_t CCLOpConfig::get_page_size() const { return this->page_size; }

Topology CCLOpConfig::get_topology() const { return this->topology; }

bool CCLOpConfig::is_input_sharded() const { return this->input_sharded; }

bool CCLOpConfig::is_output_sharded() const { return this->output_sharded; }

const Tensor& CCLOpConfig::get_input_tensor(std::size_t i) const { return input_tensors->at(i); }

const Tensor& CCLOpConfig::get_output_tensor(std::size_t i) const { return output_tensors->at(i); }

std::map<std::string, std::string> CCLOpConfig::emit_worker_defines() const {
    std::map<std::string, std::string> worker_defines;
    if (this->is_row_major) {
        worker_defines["ROW_MAJOR_LAYOUT"] = "1";
    } else {
        worker_defines["TILED_LAYOUT"] = "1";
    }
    if (this->input_sharded) {
        TT_ASSERT(
            this->output_sharded,
            "CCL Util functions currently don't  support a mix of input sharded with output interleaved or vice versa");
        worker_defines["SHARDED_MEM_LAYOUT"] = "1";
    } else {
        worker_defines["INTERLEAVED_MEM_LAYOUT"] = "1";
    }

    return worker_defines;
}

}  // namespace ttnn::ccl

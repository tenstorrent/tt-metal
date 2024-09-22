// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include "common/core_coord.h"
#include "eth_l1_address_map.h"
#include "impl/buffers/buffer.hpp"
#include "impl/kernels/data_types.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/circular_buffer_types.hpp"

#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"

// Includes that need to be moved to CCL datastructures header
#include <vector>
#include <algorithm>

using namespace tt::constants;

// Notes on abbreviations:
// cw = clockwise
// ccw = counter-clockwise
// edm = erisc data mover

// How this reduce_scatter op works:
// For each chip, we have a element range of the input tensor shape that will eventually scatter
// out to it. For all other chunks outside that range, the chip will forward the chunk to the next chip.
// While forwarding the data, the chip will also reduce it with the local input tensor chunk corresponding
// with that received chunk. It will forward the partially reduced chunk.
// Reduces along rank

namespace ttnn {

namespace ccl {
namespace reduce_scatter_detail {
struct WorkerTransferInfo {
    WorkerTransferInfo(
        std::vector<uint32_t> pages_per_full_chunk_per_worker, uint32_t num_links, uint32_t num_workers) :
        pages_per_full_chunk_per_worker(pages_per_full_chunk_per_worker),
        num_links(num_links),
        num_workers(num_workers) {}

    uint32_t get_num_pages_per_full_chunk(uint32_t link, uint32_t worker_idx) const {
        return pages_per_full_chunk_per_worker.at(link * num_workers + worker_idx);
    }

    std::vector<uint32_t> pages_per_full_chunk_per_worker;
    uint32_t num_links;
    uint32_t num_workers;
};

static std::size_t decide_number_of_edm_channels(
   ttnn::ccl::CCLOpConfig const& ccl_op_config, std::size_t max_num_workers, bool enable_bidirectional) {
    return std::min<std::size_t>(max_num_workers, enable_bidirectional ? 8 : 4);
}

struct ReduceScatterWorkerArgBuilder {
    ReduceScatterWorkerArgBuilder(
        Device const* device,
        ttnn::ccl::CCLOpConfig const& op_config,
        ttnn::ccl::RingTopology const& topology_config,
        ttnn::ccl::InterleavedTensorWorkerSlice const& worker_input_slice,
        WorkerTransferInfo const& worker_transfer_info,
        uint32_t cb_num_pages_per_packet,
        uint32_t worker_sender_semaphore_id,
        uint32_t worker_receiver_semaphore_id,
        uint32_t num_buffers_per_channel) :
        device(device),
        op_config(op_config),
        topology_config(topology_config),
        worker_input_slice(worker_input_slice),
        worker_transfer_info(worker_transfer_info),
        cb_num_pages_per_packet(cb_num_pages_per_packet),
        worker_sender_semaphore_id(worker_sender_semaphore_id),
        worker_receiver_semaphore_id(worker_receiver_semaphore_id),
        num_buffers_per_channel(num_buffers_per_channel) {
    }

    uint32_t get_total_num_math_pages(uint32_t link, uint32_t worker_idx) const {
        // This algorithm assumes that the worker slices are sized such that they start at the same x offsets for each
        // new row they slice into (as they stride through the tensor)
        std::size_t num_slice_iterations =
            worker_input_slice.compute_num_worker_slice_iterations(worker_transfer_info.num_workers);
        std::size_t worker_slice_num_pages =
            worker_input_slice.worker_slice_shape.x * worker_input_slice.worker_slice_shape.y;
        std::size_t pages_per_full_chunk = worker_transfer_info.get_num_pages_per_full_chunk(link, worker_idx);
        std::size_t num_filler_pages_per_slice = pages_per_full_chunk - (worker_slice_num_pages % pages_per_full_chunk);
        uint32_t total_num_math_pages = (worker_input_slice.get_worker_slice_num_pages() + num_filler_pages_per_slice) *
                                     num_slice_iterations * (topology_config.ring_size - 1);
        return total_num_math_pages;
    }

    std::vector<uint32_t> generate_reduce_op_kernel_ct_args() const {
        log_trace(tt::LogOp, "Reduce Scatter Worker CT Args: None");
        return {};
    }

    std::vector<uint32_t> generate_reduce_op_kernel_rt_args(
        uint32_t link, uint32_t worker_index, uint32_t ring_size) const {
        log_trace(tt::LogOp, "generate_reduce_op_kernel_rt_args");

        uint32_t total_num_math_pages = get_total_num_math_pages(link, worker_index);

        auto const& args = std::vector<uint32_t>{total_num_math_pages, 1, 0};

        std::size_t i = 0;
        log_trace(tt::LogOp, "Reduce Scatter Worker RT Args:");
        log_trace(tt::LogOp, "\tblock_size: {}", args.at(i++));
        log_trace(tt::LogOp, "\ttotal_num_math_pages: {}", args.at(i++));
        log_trace(tt::LogOp, "\tacc_to_dst: {}", args.at(i++));

        return args;
    }

    std::vector<uint32_t> generate_receiver_kernel_ct_args() const {
        auto const& local_input_tensor = this->op_config.get_input_tensor(0);
        auto args = std::vector<uint32_t>{
            static_cast<uint32_t>(this->op_config.is_input_sharded() ? 1 : 0),
            static_cast<uint32_t>(this->op_config.get_input_tensor(0).memory_config().buffer_type == BufferType::DRAM ? 1 : 0),
            static_cast<uint32_t>(this->num_buffers_per_channel)};

        std::size_t i = 0;
        log_trace(tt::LogOp, "Reduce Scatter Receiver Worker CT Args:");
        log_trace(tt::LogOp, "\tis_sharded: {}", args.at(i++));
        log_trace(tt::LogOp, "\tsrc_is_dram: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_buffers_per_channel: {}", args.at(i++));
        TT_ASSERT(args.size() == i, "Missed some args");

        if (local_input_tensor.is_sharded()) {
            auto const& shard_ct_args = ShardedAddrGenArgBuilder::emit_ct_args(local_input_tensor);
            std::copy(shard_ct_args.begin(), shard_ct_args.end(), std::back_inserter(args));
        } else {
            args.push_back(static_cast<uint32_t>(local_input_tensor.memory_config().memory_layout));
        }
        return args;
    }

    std::vector<uint32_t> generate_receiver_kernel_rt_args(
       ttnn::ccl::WorkerXY edm_core,
        uint32_t edm_core_semaphore_address,
        uint32_t edm_core_buffer_address,
        uint32_t link,
        uint32_t worker_index,
        bool is_in_clockwise_direction) const {
        TT_ASSERT(edm_core_semaphore_address > 0);
        TT_ASSERT(edm_core_buffer_address > 0);
        auto const& local_input_tensor = this->op_config.get_input_tensor(0);
        uint32_t starting_ring_index =
            is_in_clockwise_direction ? (this->topology_config.ring_index == 0 ? this->topology_config.ring_size - 1
                                                                               : this->topology_config.ring_index - 1)
                                      : (this->topology_config.ring_index == this->topology_config.ring_size - 1
                                             ? 0
                                             : this->topology_config.ring_index + 1);
        uint32_t total_num_math_pages = get_total_num_math_pages(link, worker_index);
        auto args = std::vector<uint32_t>{
            static_cast<uint32_t>(local_input_tensor.buffer()->address()),
            static_cast<uint32_t>(this->topology_config.ring_size),  // num_transfers
            static_cast<uint32_t>(this->worker_transfer_info.get_num_pages_per_full_chunk(link, worker_index)),
            static_cast<uint32_t>(this->op_config.get_page_size()),
            static_cast<uint32_t>(starting_ring_index),
            static_cast<uint32_t>(this->topology_config.ring_size),
            static_cast<uint32_t>(this->worker_receiver_semaphore_id),
            static_cast<uint32_t>(is_in_clockwise_direction ? 1 : 0),
            static_cast<uint32_t>(this->cb_num_pages_per_packet),
            static_cast<uint32_t>(edm_core.x),
            static_cast<uint32_t>(edm_core.y),
            static_cast<uint32_t>(edm_core_semaphore_address),
            static_cast<uint32_t>(edm_core_buffer_address),

            static_cast<uint32_t>(worker_transfer_info.num_workers),

            static_cast<uint32_t>(this->worker_input_slice.tensor_shape.x),
            static_cast<uint32_t>(this->worker_input_slice.tensor_shape.y),

            static_cast<uint32_t>(this->worker_input_slice.tensor_slice_shape.x),
            static_cast<uint32_t>(this->worker_input_slice.tensor_slice_shape.y),

            static_cast<uint32_t>(this->worker_input_slice.worker_slice_shape.x),
            static_cast<uint32_t>(this->worker_input_slice.worker_slice_shape.y),

            static_cast<uint32_t>(this->worker_input_slice.worker_slice_offset.x),
            static_cast<uint32_t>(this->worker_input_slice.worker_slice_offset.y),

            total_num_math_pages};

        std::size_t i = 0;
        log_trace(tt::LogOp, "Reduce Scatter Receiver Worker RT Args:");
        log_trace(tt::LogOp, "\tsrc_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_transfers: {}", args.at(i++));
        log_trace(tt::LogOp, "\tfull_chunk_num_pages: {}", args.at(i++));
        log_trace(tt::LogOp, "\tpage_size: {}", args.at(i++));
        log_trace(tt::LogOp, "\tmy_ring_idx: {}", args.at(i++));
        log_trace(tt::LogOp, "\tring_size: {}", args.at(i++));
        log_trace(tt::LogOp, "\tsem_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\tis_clockwise_direction: {}", args.at(i++));
        log_trace(tt::LogOp, "\thalf_cb_n_pages: {}", args.at(i++));

        log_trace(tt::LogOp, "\tedm_core_noc0_core_x: {}", args.at(i++));
        log_trace(tt::LogOp, "\tedm_core_noc0_core_y: {}", args.at(i++));
        log_trace(tt::LogOp, "\tedm_core_semaphore_address: {}", args.at(i++));
        log_trace(tt::LogOp, "\tedm_core_buffer_address: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_concurrent_workers: {}", args.at(i++));

        log_trace(tt::LogOp, "\tinput_tensor_shape.x={}", args.at(i++));
        log_trace(tt::LogOp, "\tinput_tensor_shape.y={}", args.at(i++));
        log_trace(tt::LogOp, "\ttensor_slice_shape.x={}", args.at(i++));
        log_trace(tt::LogOp, "\ttensor_slice_shape.y={}", args.at(i++));
        log_trace(tt::LogOp, "\tworker_slice_shape.x={}", args.at(i++));
        log_trace(tt::LogOp, "\tworker_slice_shape.y={}", args.at(i++));
        log_trace(tt::LogOp, "\tworker_slice_offset.x={}", args.at(i++));
        log_trace(tt::LogOp, "\tworker_slice_offset.y={}", args.at(i++));
        log_trace(tt::LogOp, "\ttotal_num_math_pages={}", args.at(i++));

        TT_ASSERT(args.size() == i, "Missed some args");


        if (local_input_tensor.is_sharded()) {
            auto const& shard_rt_args = ShardedAddrGenArgBuilder::emit_rt_args(device, local_input_tensor);
            std::copy(shard_rt_args.begin(), shard_rt_args.end(), std::back_inserter(args));
        }
        return args;
    }

    std::vector<uint32_t> generate_sender_kernel_ct_args() const {
        auto const& local_output_tensor = this->op_config.get_output_tensor(0);
        auto args = std::vector<uint32_t>{
            static_cast<uint32_t>(this->op_config.is_input_sharded() ? 1 : 0),
            static_cast<uint32_t>(this->op_config.get_output_tensor(0).memory_config().buffer_type == BufferType::DRAM ? 1 : 0),
            static_cast<uint32_t>(this->num_buffers_per_channel)};

        std::size_t i = 0;
        log_trace(tt::LogOp, "Reduce Scatter Sender Worker CT Args:");
        log_trace(tt::LogOp, "\tis_sharded: {}", args.at(i++));
        log_trace(tt::LogOp, "\tdst_is_dram: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_buffers_per_channel: {}", args.at(i++));
        TT_ASSERT(args.size() == i, "Missed some args");

        if (local_output_tensor.is_sharded()) {
            auto const& shard_ct_args = ShardedAddrGenArgBuilder::emit_ct_args(local_output_tensor);
            std::copy(shard_ct_args.begin(), shard_ct_args.end(), std::back_inserter(args));
        } else {
            args.push_back(static_cast<uint32_t>(local_output_tensor.memory_config().memory_layout));
        }
        return args;
    }

    std::vector<uint32_t> generate_sender_kernel_rt_args(
        ttnn::ccl::WorkerXY edm_core,
        uint32_t edm_core_semaphore_address,
        uint32_t edm_core_buffer_address,
        uint32_t link,
        uint32_t worker_index,
        bool is_clockwise) const {
        TT_ASSERT(edm_core_semaphore_address > 0);
        TT_ASSERT(edm_core_buffer_address > 0);
        auto const& local_output_tensor = this->op_config.get_output_tensor(0);
        uint32_t total_num_math_pages = get_total_num_math_pages(link, worker_index);
        auto args = std::vector<uint32_t>{
            static_cast<uint32_t>(local_output_tensor.buffer()->address()),
            static_cast<uint32_t>(edm_core_buffer_address),
            static_cast<uint32_t>(edm_core_semaphore_address),
            static_cast<uint32_t>(edm_core.x),
            static_cast<uint32_t>(edm_core.y),
            static_cast<uint32_t>(this->topology_config.ring_size - 1),  // num_transfers),

            static_cast<uint32_t>(this->op_config.get_page_size()),
            static_cast<uint32_t>(this->worker_transfer_info.get_num_pages_per_full_chunk(link, worker_index)),

            static_cast<uint32_t>(this->worker_sender_semaphore_id),
            static_cast<uint32_t>(this->cb_num_pages_per_packet),

            static_cast<uint32_t>(worker_transfer_info.num_workers),

            // For sender side, all worker slice info is the same except for the tensor shape
            // and for sender side specifically, there is only one tensor_slice_shape for the output
            // tensor (as opposed to `ring_size` tensor_slice_shapes for the input tensor), so we can
            // directly use it as the output tensor shape
            static_cast<uint32_t>(this->worker_input_slice.tensor_slice_shape.x),
            static_cast<uint32_t>(this->worker_input_slice.tensor_slice_shape.y),
            static_cast<uint32_t>(this->worker_input_slice.worker_slice_shape.x),
            static_cast<uint32_t>(this->worker_input_slice.worker_slice_shape.y),
            static_cast<uint32_t>(this->worker_input_slice.worker_slice_offset.x),
            static_cast<uint32_t>(this->worker_input_slice.worker_slice_offset.y),

            total_num_math_pages};

        std::size_t i = 0;
        log_trace(tt::LogOp, "Reduce Scatter Sender Worker RT Args:");
        log_trace(tt::LogOp, "\tdst_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\teth_sender_l1_base_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\teth_sender_l1_sem_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\teth_sender_noc_x: {}", args.at(i++));
        log_trace(tt::LogOp, "\teth_sender_noc_y: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_transfers: {}", args.at(i++));
        log_trace(tt::LogOp, "\tpage_size: {}", args.at(i++));
        log_trace(tt::LogOp, "\tfull_chunk_num_pages: {}", args.at(i++));
        log_trace(tt::LogOp, "\twriter_send_sem_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\thalf_cb_n_pages: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_concurrent_workers: {}", args.at(i++));

        log_trace(tt::LogOp, "\toutput_tensor_shape.x: {}", args.at(i++));
        log_trace(tt::LogOp, "\toutput_tensor_shape.y: {}", args.at(i++));
        log_trace(tt::LogOp, "\tworker_slice_shape.x: {}", args.at(i++));
        log_trace(tt::LogOp, "\tworker_slice_shape.y: {}", args.at(i++));
        log_trace(tt::LogOp, "\tworker_slice_offset.x: {}", args.at(i++));
        log_trace(tt::LogOp, "\tworker_slice_offset.y: {}", args.at(i++));

        log_trace(tt::LogOp, "\ttotal_num_math_pages={}", args.at(i++));

        TT_ASSERT(args.size() == i, "Missed some args");

        if (local_output_tensor.is_sharded()) {
            auto const& shard_rt_args = ShardedAddrGenArgBuilder::emit_rt_args(device, local_output_tensor);
            std::copy(shard_rt_args.begin(), shard_rt_args.end(), std::back_inserter(args));
        }
        return args;
    }

    Device const*device;
    ttnn::ccl::RingTopology const topology_config;
    ttnn::ccl::CCLOpConfig const op_config;
    ttnn::ccl::InterleavedTensorWorkerSlice const worker_input_slice;
    WorkerTransferInfo const worker_transfer_info;
    uint32_t cb_num_pages_per_packet;
    uint32_t worker_sender_semaphore_id;
    uint32_t worker_receiver_semaphore_id;
    uint32_t num_buffers_per_channel;

    bool src_is_dram;
    bool dst_is_dram;
};

struct EdmInterfaceAddresses {
    std::unordered_map<int, uint32_t> worker_sender_edm_semaphore_addresses;
    std::unordered_map<int, uint32_t> worker_sender_edm_buffer_addresses;
    std::unordered_map<int, uint32_t> worker_receiver_edm_semaphore_addresses;
    std::unordered_map<int, uint32_t> worker_receiver_edm_buffer_addresses;
};

// Future work: split this up further:
// 1) assign workers to EDM channel (with buffer sharing mode specified too)
// 2) Compute the semaphore and buffer addresses (for each EDM channel and worker)
// For now - the mapping between workers and EDM channels is 1:1
static void add_worker_config_to_edm_builders(
    Device* device,
    RingReduceScatterWrappedTensorSlicer& tensor_slicer,  // TODO: Update to Generic ReduceScatterSlicer when it is implemented
    ccl::CCLOpConfig const& op_config,
    std::vector<CoreCoord> const& worker_cores,
    uint32_t num_channels_per_edm,
    uint32_t num_buffers_per_channel,

    std::vector<ttnn::ccl::EriscDatamoverBuilder>& clockwise_edm_builders,
    std::vector<ttnn::ccl::EriscDatamoverBuilder>& counter_clockwise_edm_builders,

    uint32_t worker_sender_semaphore_id,
    uint32_t worker_receiver_semaphore_id,
    uint32_t link,
    uint32_t ring_size,
    std::function<bool(uint32_t)> is_buffer_in_clockwise_direction_fn,

    EdmInterfaceAddresses& edm_interface_addresses) {
    for (uint32_t c = 0; c < num_channels_per_edm; ++c) {
        uint32_t global_worker_idx = c + num_channels_per_edm * link;
        uint32_t num_workers_per_eth_buffer = 1;

        std::vector<ttnn::ccl::WorkerXY> sender_worker_coords;
        std::vector<ttnn::ccl::WorkerXY> receiver_worker_coords;
        for (uint32_t w = c * num_workers_per_eth_buffer; w < (c + 1) * num_workers_per_eth_buffer; ++w) {
            sender_worker_coords.push_back(ttnn::ccl::WorkerXY(
                device->worker_core_from_logical_core(worker_cores.at(w)).x,
                device->worker_core_from_logical_core(worker_cores.at(w)).y));
            receiver_worker_coords.push_back(ttnn::ccl::WorkerXY(
                device->worker_core_from_logical_core(worker_cores.at(w)).x,
                device->worker_core_from_logical_core(worker_cores.at(w)).y));
        }

        // Get the maximum message size we'd like to use. Not the actual packet size
        uint32_t expected_message_size_bytes = (num_buffers_per_channel == 1) ? tensor_slicer.get_worker_slice_size_bytes(global_worker_idx)
                                                                           : clockwise_edm_builders.at(link).get_eth_buffer_size_bytes();

        bool sender_enabled = true;  // (!is_linear || !is_last_chip_in_chain); // update for linear
        if (sender_enabled) {
            auto& sender_edm_builder = is_buffer_in_clockwise_direction_fn(c) ? clockwise_edm_builders.at(link)
                                                                              : counter_clockwise_edm_builders.at(link);
            log_trace(tt::LogOp, "Adding sender EDM channel");
            ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& sender_channel_buffer_info =
                sender_edm_builder.add_sender_channel(
                    worker_sender_semaphore_id,
                    1,  // cw_edm_channel_num_messages_to_send_per_transfer.at(c) * (ring_size - 1),
                    sender_worker_coords,
                    expected_message_size_bytes);
            edm_interface_addresses.worker_sender_edm_semaphore_addresses.insert(
                {global_worker_idx, sender_channel_buffer_info.eth_semaphore_l1_address});
            edm_interface_addresses.worker_sender_edm_buffer_addresses.insert(
                {global_worker_idx, sender_channel_buffer_info.eth_buffer_l1_address});
        }

        bool receiver_enabled = true;  //(!is_linear || !is_first_chip_in_chain);
        if (receiver_enabled) {
            auto& receiver_edm_builder = is_buffer_in_clockwise_direction_fn(c)
                                             ? counter_clockwise_edm_builders.at(link)
                                             : clockwise_edm_builders.at(link);
            log_trace(tt::LogOp, "Adding receiver EDM channel");
            ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& receiver_channel_buffer_info =
                receiver_edm_builder.add_receiver_channel(
                    worker_receiver_semaphore_id,
                    // Since we are in worker signal EDM termination mode, we don't need to set the actual number of
                    // messages the EDM must forward as it will receive its finish signal from the worker instead
                    1,
                    receiver_worker_coords,
                    expected_message_size_bytes);
            edm_interface_addresses.worker_receiver_edm_semaphore_addresses.insert(
                {global_worker_idx, receiver_channel_buffer_info.eth_semaphore_l1_address});
            edm_interface_addresses.worker_receiver_edm_buffer_addresses.insert(
                {global_worker_idx, receiver_channel_buffer_info.eth_buffer_l1_address});
        }
    }
}

static std::tuple<KernelHandle, KernelHandle, KernelHandle> build_reduce_scatter_worker_ct(
    tt::tt_metal::Program& program,
    ttnn::ccl::CCLOpConfig const& op_config,
    ReduceScatterWorkerArgBuilder const& worker_arg_builder,
    CoreRangeSet const& worker_core_range,
    ttnn::operations::binary::BinaryOpType binary_math_op) {

    auto const& worker_defines = op_config.emit_worker_defines();
    TT_ASSERT(worker_defines.size() > 0);
    for (auto const& [key, value] : worker_defines) {
        log_trace(tt::LogOp, "Worker Define: {} = {}", key, value);
    }
    static std::string const& receiver_kernel_path =
        "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/kernels/worker_interleaved_ring_reduce_scatter_reader.cpp";
    static std::string const& sender_kernel_path =
        "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/kernels/worker_interleaved_ring_reduce_scatter_sender.cpp";
    static std::string const& reduce_kernel_path =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp";

    KernelHandle worker_receiver_kernel_id, worker_sender_kernel_id, worker_reduce_kernel_id;

    worker_receiver_kernel_id = tt::tt_metal::CreateKernel(
        program,
        receiver_kernel_path,
        worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(worker_arg_builder.generate_receiver_kernel_ct_args(), worker_defines));

    worker_sender_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_kernel_path,
        worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(worker_arg_builder.generate_sender_kernel_ct_args(), worker_defines));

    vector<uint32_t> compute_kernel_args = {};
    constexpr bool fp32_dest_acc_en = false;
    constexpr bool math_approx_mode = false;
    std::map<string, string> eltwise_defines = ttnn::operations::binary::utils::get_defines(binary_math_op);
    worker_reduce_kernel_id = tt::tt_metal::CreateKernel(
        program,
        reduce_kernel_path,
        worker_core_range,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = eltwise_defines});

    return {worker_receiver_kernel_id, worker_sender_kernel_id, worker_reduce_kernel_id};
}

static void set_reduce_scatter_worker_rt(
    tt::tt_metal::Program& program,
    Device const* device,
    KernelHandle worker_receiver_kernel_id,
    KernelHandle worker_sender_kernel_id,
    KernelHandle worker_reduce_kernel_id,
    ttnn::ccl::RingTopology const& topology_config,
    ttnn::ccl::CCLOpConfig const& op_config,
    ReduceScatterWorkerArgBuilder const& worker_arg_builder,
    std::vector<ttnn::ccl::EriscDatamoverBuilder>& cw_edm_builders,
    std::vector<ttnn::ccl::EriscDatamoverBuilder>& ccw_edm_builders,
    EdmInterfaceAddresses const& edm_interface_addresses,
    CoreCoord const& worker_core,
    uint32_t num_edm_channels,
    uint32_t link,
    uint32_t ring_size,
    uint32_t worker_index,
    ttnn::operations::binary::BinaryOpType binary_math_op,
    std::function<bool(uint32_t)> is_buffer_in_clockwise_direction_fn) {

    bool is_in_clockwise_direction = is_buffer_in_clockwise_direction_fn(worker_index);
    uint32_t global_worker_index = link * num_edm_channels + worker_index;
    {
        CoreCoord const& receiver_edm = is_in_clockwise_direction ? topology_config.eth_receiver_cores.at(link)
                                                                  : topology_config.eth_sender_cores.at(link);
       ttnn::ccl::WorkerXY receiver_edm_noc_coord =ttnn::ccl::WorkerXY(
            device->ethernet_core_from_logical_core(receiver_edm).x,
            device->ethernet_core_from_logical_core(receiver_edm).y);
        const uint32_t edm_core_semaphore_address =
            is_in_clockwise_direction
                ? edm_interface_addresses.worker_receiver_edm_semaphore_addresses.at(global_worker_index)
                : edm_interface_addresses.worker_sender_edm_semaphore_addresses.at(global_worker_index);
        const uint32_t edm_core_buffer_address =
            is_in_clockwise_direction
                ? edm_interface_addresses.worker_receiver_edm_buffer_addresses.at(global_worker_index)
                : edm_interface_addresses.worker_sender_edm_buffer_addresses.at(global_worker_index);

        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_receiver_kernel_id,
            worker_core,
            worker_arg_builder.generate_receiver_kernel_rt_args(
                receiver_edm_noc_coord,
                edm_core_semaphore_address,
                edm_core_buffer_address,
                link,
                worker_index,
                is_in_clockwise_direction));
    }

    {
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_reduce_kernel_id,
            worker_core,
            worker_arg_builder.generate_reduce_op_kernel_rt_args(link, worker_index, ring_size));
    }

    {
        CoreCoord sender_edm = is_in_clockwise_direction ? topology_config.eth_sender_cores.at(link)
                                                         : topology_config.eth_receiver_cores.at(link);
       ttnn::ccl::WorkerXY const sender_edm_noc_coord =ttnn::ccl::WorkerXY(
            device->ethernet_core_from_logical_core(sender_edm).x,
            device->ethernet_core_from_logical_core(sender_edm).y);
        TT_ASSERT(sender_edm_noc_coord.y == 0 || sender_edm_noc_coord.y == 6);
        const uint32_t edm_core_semaphore_address =
            is_in_clockwise_direction
                ? edm_interface_addresses.worker_sender_edm_semaphore_addresses.at(global_worker_index)
                : edm_interface_addresses.worker_receiver_edm_semaphore_addresses.at(global_worker_index);
        const uint32_t edm_core_buffer_address =
            is_in_clockwise_direction
                ? edm_interface_addresses.worker_sender_edm_buffer_addresses.at(global_worker_index)
                : edm_interface_addresses.worker_receiver_edm_buffer_addresses.at(global_worker_index);

        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_sender_kernel_id,
            worker_core,
            worker_arg_builder.generate_sender_kernel_rt_args(
                sender_edm_noc_coord,
                edm_core_semaphore_address,
                edm_core_buffer_address,
                link,
                worker_index,
                is_in_clockwise_direction));
    }
}

static CoreRangeSet select_worker_cores(
   ttnn::ccl::CCLOpConfig const& op_config, std::size_t num_links, std::size_t num_edm_channels) {
    switch (op_config.get_topology()) {
        case ttnn::ccl::Topology::Linear:
            return CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(num_edm_channels - 1, num_links - 1))});
        case ttnn::ccl::Topology::Ring:
            return CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(num_edm_channels - 1, num_links - 1))});
        default: TT_ASSERT(false, "Unsupported topology"); return CoreRangeSet({});
    };
}

static WorkerTransferInfo compute_num_edm_messages_per_channel(
    ccl::CCLOpConfig const& op_config,
    RingReduceScatterWrappedTensorSlicer& tensor_slicer,  // TODO: Update to Generic ReduceScatterSlicer when it is implemented
    std::vector<ttnn::ccl::EriscDatamoverBuilder> const& cw_per_link_edm_builders,
    std::vector<ttnn::ccl::EriscDatamoverBuilder> const& ccw_per_link_edm_builders,
    std::size_t const num_edm_channels,
    std::size_t const num_links,
    std::size_t const ring_size) {
    uint32_t const page_size_in_bytes = op_config.get_page_size();
    TT_ASSERT(num_edm_channels > 0);
    TT_ASSERT(num_links > 0);
    TT_ASSERT(page_size_in_bytes > 0);
    log_trace(tt::LogOp, "WorkerTransferInfo");
    std::size_t total_num_workers = num_edm_channels * num_links;

    auto get_iter_begin = [num_edm_channels](auto& vec, std::size_t link) -> auto {
        return vec.begin() + (link * num_edm_channels);
    };

    auto get_iter_end = [num_edm_channels, num_links](auto& vec, std::size_t link) -> auto {
        bool last_link = link == num_links - 1;
        TT_ASSERT(
            (!last_link && ((link + 1) * num_edm_channels < vec.size())) ||
            (last_link && ((link + 1) * num_edm_channels == vec.size())));
        return last_link ? vec.end() : vec.begin() + ((link + 1) * num_edm_channels);
    };

    // Pages per EDM channel
    std::size_t total_num_edm_channels = num_links * num_edm_channels;
    log_trace(tt::LogOp, "total_num_edm_channels: {}", total_num_edm_channels);

    std::vector<uint32_t> num_pages_per_full_chunk(total_num_edm_channels * num_links, 0);

    for (std::size_t link = 0; link < num_links; link++) {
        std::size_t edm_channel_size_in_bytes = cw_per_link_edm_builders.at(link).get_eth_buffer_size_bytes();
        std::size_t num_pages_per_edm_buffer = edm_channel_size_in_bytes / page_size_in_bytes;
        log_trace(
            tt::LogOp,
            "link {}, edm_channel_size_in_bytes: {}, page_size_in_bytes: {}, num_pages_per_edm_buffer: {}",
            link,
            edm_channel_size_in_bytes,
            page_size_in_bytes,
            num_pages_per_edm_buffer);

        std::fill(
            get_iter_begin(num_pages_per_full_chunk, link),
            get_iter_end(num_pages_per_full_chunk, link),
            num_pages_per_edm_buffer);
    }

    log_trace(tt::LogOp, "-- num_pages_per_full_chunk:");
    for (std::size_t l = 0; l < num_links; l++) {
        for (std::size_t w = 0; w < num_edm_channels; w++) {
            log_trace(
                tt::LogOp, "\t\t(link={},worker={}): {}", l, w, num_pages_per_full_chunk.at(l * num_edm_channels + w));
        }
    }

    return WorkerTransferInfo(num_pages_per_full_chunk, num_links, num_edm_channels);
}

static uint32_t compute_maximum_worker_slice_in_bytes(
    uint32_t cb_src0_size_pages,
    uint32_t cb_dst0_size_pages,
    uint32_t cb_short_circuit_size_pages,
    std::size_t edm_channel_buffer_size,
    uint32_t page_size) {
    return std::min(cb_short_circuit_size_pages, cb_src0_size_pages + cb_dst0_size_pages) * page_size +
           edm_channel_buffer_size;
}

static bool is_cb_buffering_sufficient_to_avoid_deadlock(
   ttnn::ccl::InterleavedTensorWorkerSlice const& worker_slice,
    uint32_t cb_src0_size_pages,
    uint32_t cb_dst0_size_pages,
    uint32_t cb_short_circuit_size_pages,
    std::size_t edm_channel_buffer_size,
    uint32_t page_size) {
    uint32_t worker_size_pages_rounded_up =
        tt::round_up(worker_slice.worker_slice_shape.x * worker_slice.worker_slice_shape.y, cb_src0_size_pages / 2);
    uint32_t worker_slice_size_bytes = worker_size_pages_rounded_up * page_size;
    uint32_t available_buffering_capacity = compute_maximum_worker_slice_in_bytes(
        cb_src0_size_pages, cb_dst0_size_pages, cb_short_circuit_size_pages, edm_channel_buffer_size, page_size);
    log_trace(tt::LogOp, "worker_slice.worker_slice_shape.x: {}", worker_slice.worker_slice_shape.x);
    log_trace(tt::LogOp, "worker_slice.worker_slice_shape.y: {}", worker_slice.worker_slice_shape.y);
    log_trace(tt::LogOp, "worker_slice_size_bytes: {}", worker_slice_size_bytes);
    log_trace(tt::LogOp, "worker_size_pages_rounded_up: {}", worker_size_pages_rounded_up);
    log_trace(tt::LogOp, "cb_src0_size_pages: {}", cb_src0_size_pages);
    log_trace(tt::LogOp, "cb_dst0_size_pages: {}", cb_dst0_size_pages);
    log_trace(tt::LogOp, "page_size: {}", page_size);
    log_trace(tt::LogOp, "edm_channel_buffer_size: {}", edm_channel_buffer_size);
    log_trace(tt::LogOp, "available_buffering_capacity: {}", available_buffering_capacity);

    return available_buffering_capacity >= worker_slice_size_bytes;
}

static std::tuple<CBHandle, CBHandle, CBHandle, CBHandle> create_worker_circular_buffers(
    Tensor const& input_tensor,
   ttnn::ccl::CCLOpConfig const& op_config,
    CoreRangeSet const& worker_core_range,
    uint32_t worker_pages_per_transfer,
    tt::tt_metal::Program& program) {
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t page_size_bytes = op_config.get_page_size();

    // Input 0 CB
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(worker_pages_per_transfer * page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, page_size_bytes);
    CBHandle cb_src0_workers = CreateCircularBuffer(program, worker_core_range, cb_src0_config);

    // Input 1 CB
    uint32_t src1_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(worker_pages_per_transfer * page_size_bytes, {{src1_cb_index, df}})
            .set_page_size(src1_cb_index, page_size_bytes);
    CBHandle cb_src1_workers = CreateCircularBuffer(program, worker_core_range, cb_src1_config);

    // Dataflow Writer Kernel input CB
    uint32_t cb_dst0_index = tt::CB::c_out0;
    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(worker_pages_per_transfer * page_size_bytes, {{cb_dst0_index, df}})
            .set_page_size(cb_dst0_index, page_size_bytes);
    CBHandle cb_dst0_sender_workers = CreateCircularBuffer(program, worker_core_range, cb_dst0_config);

    // From reader -> writer kernel (I think I need this because sharing the cb_dst0_sender_workers as output
    // of reader kernel (first output) and math kernel (all subsequent outputs) doesn't seem to work because
    // it seems like the math kernels hold some of the CB state in local variables)
    uint32_t cb_short_circuit_index = tt::CB::c_out1;
    tt::tt_metal::CircularBufferConfig cb_short_circuit_config =
        tt::tt_metal::CircularBufferConfig(
            (worker_pages_per_transfer * page_size_bytes) * 2, {{cb_short_circuit_index, df}})
            .set_page_size(cb_short_circuit_index, page_size_bytes);
    CBHandle cb_short_circuit_sender_workers =
        CreateCircularBuffer(program, worker_core_range, cb_short_circuit_config);

    return {cb_src0_workers, cb_src1_workers, cb_dst0_sender_workers, cb_short_circuit_sender_workers};
}

operation::ProgramWithCallbacks reduce_scatter_with_workers(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    ttnn::operations::binary::BinaryOpType reduce_op,
    const uint32_t scatter_split_dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    log_trace(tt::LogOp, "reduce_scatter_with_workers entry");
    TT_ASSERT(
        input_tensor.get_legacy_shape()[scatter_split_dim] ==
            output_tensor.get_legacy_shape()[scatter_split_dim] * ring_size,
        "Input and output tensor shapes must match");
    TT_ASSERT(
        input_tensor.buffer()->num_pages() % ring_size == 0,
        "Reduce scatter current only supports even divisibility of input tensor(s) across ranks");

    /////////////// Constants/Configuration
    /// Constants/Configuration
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    ttnn::ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode =ttnn::ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN;
    auto const& op_config =ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    std::unique_ptr<ttnn::ccl::CclOpTensorConfig> input_tensor_config =
        ttnn::ccl::CclOpTensorConfig::build_all_gather_tensor_config(input_tensor);
    std::unique_ptr<ttnn::ccl::CclOpTensorConfig> output_tensor_config =
        ttnn::ccl::CclOpTensorConfig::build_all_gather_tensor_config(output_tensor);
    // // The input tensor is fractured by ring_size so we divi
    std::size_t input_tensor_n_elems_per_slice = input_tensor.volume() / ring_size;
    uint32_t input_tensor_num_units_per_tensor_slice =
        input_tensor_n_elems_per_slice / (tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT);

    TT_ASSERT(input_tensor_num_units_per_tensor_slice > 0);
    uint32_t max_num_workers = std::min<std::size_t>(user_defined_num_workers.value_or(8), input_tensor_num_units_per_tensor_slice);
    bool enable_bidirectional = true;
    std::size_t num_edm_channels = decide_number_of_edm_channels(op_config, max_num_workers, enable_bidirectional);
    log_trace(tt::LogOp, "num_edm_channels: {}", num_edm_channels);
    auto edm_termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED;

    std::size_t num_buffers_per_channel = 2;
    if (user_defined_num_buffers_per_channel.has_value()) {
        // Override with user defined value
        num_buffers_per_channel = user_defined_num_buffers_per_channel.value();
    }
    auto const& edm_builder = create_erisc_datamover_builder(
        num_edm_channels, op_config.get_page_size(), num_buffers_per_channel, buffer_sharing_mode, edm_termination_mode);
    TT_ASSERT(num_edm_channels > 0);

    Tensor const& local_chip_tensor = input_tensor;
    Tensor const& local_chip_output_tensor = output_tensor;

    std::map<string, string> worker_defines;
    std::vector<ttnn::ccl::EriscDatamoverBuilder> cw_per_link_edm_builders(num_links, edm_builder);
    std::vector<ttnn::ccl::EriscDatamoverBuilder> ccw_per_link_edm_builders(num_links, edm_builder);

    //////////////////
    tt::tt_metal::Program program{};

    const auto& device = local_chip_tensor.device();

    auto const& topology_config =
       ttnn::ccl::RingTopology(device, topology, sender_device_id, receiver_device_id, num_links, ring_size, ring_index);

    CoreRangeSet const& worker_core_range = select_worker_cores(op_config, num_links, num_edm_channels);
    auto const& worker_cores = corerange_to_cores(worker_core_range, std::nullopt, true);

    // Semaphores && CBs
    auto worker_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, worker_core_range, 0);
    auto worker_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, worker_core_range, 0);

    uint32_t cb_num_pages = std::min(input_tensor_num_units_per_tensor_slice,
        (cw_per_link_edm_builders.at(0).get_eth_buffer_size_bytes() / op_config.get_page_size())) * 2;
    uint32_t cb_num_pages_per_packet = cb_num_pages / 2;
    log_trace(tt::LogOp, "cb_num_pages: {}", cb_num_pages);
    auto const& [cb_src0_workers, cb_src1_workers, cb_dst0_sender_workers, cb_short_circuit_sender_workers] =
        create_worker_circular_buffers(local_chip_tensor, op_config, worker_core_range, cb_num_pages, program);

    uint32_t max_worker_slice_in_bytes = compute_maximum_worker_slice_in_bytes(
        cb_num_pages,
        cb_num_pages,
        cb_num_pages,
        cw_per_link_edm_builders.at(0).get_eth_buffer_size_bytes(),
        op_config.get_page_size());
    std::size_t num_workers = worker_cores.size();
    TT_ASSERT(num_workers == num_edm_channels * num_links);
    auto tensor_slicer = ttnn::ccl::RingReduceScatterWrappedTensorSlicer(
        local_chip_tensor,
        local_chip_output_tensor,
        scatter_split_dim,
        ring_index,
        ring_size,
        num_workers,
        max_worker_slice_in_bytes,
        cb_num_pages / 2);

    // Not per buffer because the buffer sharing mode may cause some buffers to share EDM transfers
    WorkerTransferInfo const& worker_transfer_info = compute_num_edm_messages_per_channel(
        op_config,
        tensor_slicer,
        cw_per_link_edm_builders,
        ccw_per_link_edm_builders,
        num_edm_channels,
        num_links,
        ring_size);

    // Configure the EDM builders
    std::function<bool(uint32_t)> is_worker_in_clockwise_direction_fn = [enable_bidirectional, num_edm_channels](uint32_t x) {
                static constexpr bool bidirectional_directions = 2;
                return enable_bidirectional ? (x % bidirectional_directions == 0) : true;
            };
    EdmInterfaceAddresses edm_interface_addresses;
    for (std::size_t link = 0; link < num_links; link++) {
        add_worker_config_to_edm_builders(
            device,
            tensor_slicer,
            op_config,
            worker_cores,
            num_edm_channels,
            num_buffers_per_channel,

            cw_per_link_edm_builders,
            ccw_per_link_edm_builders,

            worker_sender_semaphore_id,
            worker_receiver_semaphore_id,
            link,
            ring_size,
            is_worker_in_clockwise_direction_fn,

            edm_interface_addresses);
    }

    // build worker kernels ct
    auto const& dummy_worker_slice = tensor_slicer.get_worker_slice(0);
    auto worker_arg_builder = ReduceScatterWorkerArgBuilder(
        device,
        op_config,
        topology_config,
        dummy_worker_slice,
        worker_transfer_info,
        cb_num_pages_per_packet,
        worker_sender_semaphore_id,
        worker_receiver_semaphore_id,
        num_buffers_per_channel);
    auto [worker_receiver_kernel_id, worker_sender_kernel_id, worker_reduce_kernel_id] = build_reduce_scatter_worker_ct(
        program,
        op_config,
        worker_arg_builder,
        worker_core_range,
        reduce_op);

    // set worker kernels rt
    tt::tt_metal::ComputeConfig compute_config;
    for (std::size_t link = 0; link < num_links; link++) {
        uint32_t global_worker_index = link * num_edm_channels;
        log_trace(tt::LogOp, "==============================================");
        log_trace(tt::LogOp, "------------------ Link: {} ------------------", link);
        for (std::size_t worker = 0; worker < num_edm_channels; worker++) {
            std::size_t global_worker_index = worker + link * num_edm_channels;
            log_trace(tt::LogOp, "------ Worker: {} (global ID={})", worker, global_worker_index);

            auto const& worker_slice = tensor_slicer.get_worker_slice(global_worker_index);
            auto worker_arg_builder = ReduceScatterWorkerArgBuilder(
                device,
                op_config,
                topology_config,
                worker_slice,
                worker_transfer_info,
                cb_num_pages_per_packet,
                worker_sender_semaphore_id,
                worker_receiver_semaphore_id,
                num_buffers_per_channel);

            log_trace(tt::LogOp, "worker_cores.at(global_worker_index): {}", worker_cores.at(global_worker_index));
            set_reduce_scatter_worker_rt(
                program,
                device,
                worker_receiver_kernel_id,
                worker_sender_kernel_id,
                worker_reduce_kernel_id,
                topology_config,
                op_config,
                worker_arg_builder,
                cw_per_link_edm_builders,
                ccw_per_link_edm_builders,
                edm_interface_addresses,
                worker_cores.at(global_worker_index),
                num_edm_channels,
                link,
                ring_size,
                worker,
                reduce_op,
                is_worker_in_clockwise_direction_fn);

            TT_FATAL(is_cb_buffering_sufficient_to_avoid_deadlock(
                worker_slice,
                cb_num_pages,
                cb_num_pages,
                cb_num_pages,
                cw_per_link_edm_builders.at(0).get_eth_buffer_size_bytes(),
                op_config.get_page_size()), "Error");
        }
    }

    // Generate the EDM kernels
   ttnn::ccl::generate_edm_kernels_for_ring_or_linear_topology(
        program,
        device,
        topology_config,
        cw_per_link_edm_builders,
        ccw_per_link_edm_builders,
        receiver_device_id,
        sender_device_id);

    uint32_t total_num_workers = worker_cores.size();
    auto override_runtime_arguments_callback =
        [topology_config, worker_receiver_kernel_id, worker_sender_kernel_id, worker_cores, total_num_workers, ring_index](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors.at(0);
            const auto& output = output_tensors.at(0);
            auto &worker_receiver_runtime_args_by_core = GetRuntimeArgs(program, worker_receiver_kernel_id);
            auto &worker_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_kernel_id);
            for (uint32_t i = 0; i < worker_cores.size(); ++i) {
                auto core = worker_cores.at(i);
                auto& worker_receiver_runtime_args = worker_receiver_runtime_args_by_core[core.x][core.y];
                worker_receiver_runtime_args.at(0) = input.buffer()->address();

                auto& worker_sender_runtime_args = worker_sender_runtime_args_by_core[core.x][core.y];
                worker_sender_runtime_args.at(0) = output.buffer()->address();
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace reduce_scatter_detail
}  // namespace ccl
}  // namespace ttnn

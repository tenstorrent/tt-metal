// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include "common/core_coord.h"
#include "impl/buffers/buffer.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/circular_buffer_types.hpp"

#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/host/reduce_scatter_worker_builder.hpp"

// Includes that need to be moved to CCL datastructures header
#include <vector>
#include <algorithm>
#include <limits>
#include <ranges>

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




static std::size_t decide_number_of_edm_channels(
   ttnn::ccl::CCLOpConfig const& ccl_op_config, std::size_t max_num_workers, bool enable_bidirectional) {
    bool is_linear_topology = ccl_op_config.get_topology() == ttnn::ccl::Topology::Linear;
    TT_ASSERT(!is_linear_topology || max_num_workers > 1);
    if (is_linear_topology) {
        // Workers must be evenly divided for line reduce scatter
        max_num_workers = tt::round_down(max_num_workers, 2);
    }
    return std::min<std::size_t>(max_num_workers, enable_bidirectional || is_linear_topology ? 8 : 4);
}


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
    std::vector<WorkerAttributes> const& all_worker_attributes,
    std::size_t num_channels_per_edm,
    std::size_t num_buffers_per_channel,

    std::vector<ttnn::ccl::EriscDatamoverBuilder>& clockwise_edm_builders,
    std::vector<ttnn::ccl::EriscDatamoverBuilder>& counter_clockwise_edm_builders,

    ttnn::ccl::RingTopology const& topology_config,
    std::size_t link,

    EdmInterfaceAddresses& edm_interface_addresses) {
    bool is_linear = topology_config.is_linear;
    for (std::size_t c = 0; c < num_channels_per_edm; ++c) {
        std::size_t num_workers_per_eth_buffer = 1;
        auto global_worker_index = get_global_worker_id(link, c, num_channels_per_edm);
        TT_ASSERT(global_worker_index < all_worker_attributes.size());
        WorkerAttributes const& worker_attrs = all_worker_attributes[global_worker_index];

        std::vector<ttnn::ccl::WorkerXY> sender_worker_coords;
        std::vector<ttnn::ccl::WorkerXY> receiver_worker_coords;
        auto const& worker_noc_coords = device->worker_core_from_logical_core(worker_attrs.location_logical);
        sender_worker_coords.push_back(ttnn::ccl::WorkerXY(worker_noc_coords.x, worker_noc_coords.y));
        receiver_worker_coords.push_back(ttnn::ccl::WorkerXY(worker_noc_coords.x, worker_noc_coords.y));

        // Get the maximum message size we'd like to use. Not the actual packet size
        // If linear, then we want to reuse the slicer in both directions
        std::size_t global_worker_idx = c + num_channels_per_edm * link;
        log_trace(tt::LogOp, "get_worker_slice_size_bytes");
        std::size_t worker_tensor_slice_index = !is_linear ? global_worker_idx : (c % (num_channels_per_edm / 2)) + ((num_channels_per_edm / 2) * link);

        bool is_in_clockwise_direction = worker_attrs.direction == Direction::CLOCKWISE;

        // sender kernel enabled
        bool sender_enabled = !is_linear || !topology_config.is_last_device_in_line(is_in_clockwise_direction);
        if (sender_enabled) {
            bool choose_clockwise_edm_builder = is_in_clockwise_direction;
            log_trace(tt::LogOp, "Adding sender EDM channel to {} edm builder", choose_clockwise_edm_builder ? "clockwise" : "counter-clockwise");
            auto& sender_edm_builder = choose_clockwise_edm_builder ? clockwise_edm_builders.at(link)
                                                                              : counter_clockwise_edm_builders.at(link);
            std::size_t expected_message_size_bytes = (num_buffers_per_channel == 1) ? tensor_slicer.get_worker_slice_size_bytes(worker_tensor_slice_index)
                                                                            : sender_edm_builder.get_eth_buffer_size_bytes();
            TT_ASSERT(worker_attrs.send_to_edm_semaphore_id.has_value(), "Internal error");
            ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& sender_channel_buffer_info =
                sender_edm_builder.add_sender_channel(
                    worker_attrs.send_to_edm_semaphore_id.value(),
                    1,
                    sender_worker_coords,
                    expected_message_size_bytes);
            edm_interface_addresses.worker_sender_edm_semaphore_addresses.insert(
                {global_worker_idx, sender_channel_buffer_info.eth_semaphore_l1_address});
            edm_interface_addresses.worker_sender_edm_buffer_addresses.insert(
                {global_worker_idx, sender_channel_buffer_info.eth_buffer_l1_address});
            log_trace(tt::LogOp, "EDM-IF SENDER: Ring Index: {}, sender {}, sem_addr: {}, buf_addr: {}", topology_config.ring_index, global_worker_idx, sender_channel_buffer_info.eth_semaphore_l1_address, sender_channel_buffer_info.eth_buffer_l1_address);
            log_trace(tt::LogOp, "\tAdded");
        }

        // receiver kernel enabled
        bool receiver_enabled = !is_linear || !topology_config.is_first_device_in_line(is_in_clockwise_direction);
        if (receiver_enabled) {
            bool choose_counter_clockwise_edm_builder = is_in_clockwise_direction;
            log_trace(tt::LogOp, "Adding receiver EDM channel to {} edm builder", choose_counter_clockwise_edm_builder ? "counter-clockwise" : "clockwise");
            auto& receiver_edm_builder =
                 is_in_clockwise_direction ? counter_clockwise_edm_builders.at(link) : clockwise_edm_builders.at(link);
            std::size_t expected_message_size_bytes = (num_buffers_per_channel == 1) ? tensor_slicer.get_worker_slice_size_bytes(worker_tensor_slice_index)
                                                                            : receiver_edm_builder.get_eth_buffer_size_bytes();
            TT_ASSERT(worker_attrs.receive_from_edm_semaphore_id.has_value());
            ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& receiver_channel_buffer_info =
                receiver_edm_builder.add_receiver_channel(
                    worker_attrs.receive_from_edm_semaphore_id.value(),
                    // Since we are in worker signal EDM termination mode, we don't need to set the actual number of
                    // messages the EDM must forward as it will receive its finish signal from the worker instead
                    1,
                    receiver_worker_coords,
                    expected_message_size_bytes);

            edm_interface_addresses.worker_receiver_edm_semaphore_addresses.insert(
                {global_worker_idx, receiver_channel_buffer_info.eth_semaphore_l1_address});
            edm_interface_addresses.worker_receiver_edm_buffer_addresses.insert(
                {global_worker_idx, receiver_channel_buffer_info.eth_buffer_l1_address});
            log_trace(tt::LogOp, "EDM-IF RECEIVER: Ring Index: {}, receiver {}, sem_addr: {}, buf_addr: {}", topology_config.ring_index, global_worker_idx, receiver_channel_buffer_info.eth_semaphore_l1_address, receiver_channel_buffer_info.eth_buffer_l1_address);
        }

        TT_ASSERT(receiver_enabled || sender_enabled);
    }
}

static std::tuple<KernelHandle, KernelHandle, KernelHandle, std::optional<KernelHandle>> build_reduce_scatter_worker_ct(
    tt::tt_metal::Program& program,
    ttnn::ccl::RingTopology const& topology_config,
    ttnn::ccl::CCLOpConfig const& op_config,
    ReduceScatterWorkerArgBuilder const& worker_arg_builder,
    CoreRangeSet const& worker_core_range,
    // if line and at the end of the line we split the worker core range
    // because we need to invoke separate kernels
    std::optional<CoreRangeSet> const& split_worker_core_range,
    ttnn::operations::binary::BinaryOpType binary_math_op) {
    log_trace(tt::LogOp, "build_reduce_scatter_worker_ct");

    auto const& worker_defines = op_config.emit_worker_defines();
    TT_ASSERT(worker_defines.size() > 0);
    for (auto const& [key, value] : worker_defines) {
        log_trace(tt::LogOp, "Worker Define: {} = {}", key, value);
    }
    if (split_worker_core_range.has_value()) {
        log_trace(tt::LogOp, "second worker core list:");
        for (const auto &core : corerange_to_cores(split_worker_core_range.value())) {
            log_trace(tt::LogOp, "\tx={},y={}", core.x, core.y);
        }
    }

    static std::string const& receiver_kernel_path = "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/kernels/worker_interleaved_ring_reduce_scatter_reader.cpp";
    static std::string const& forward_sender_kernel_path = "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/kernels/worker_interleaved_ring_reduce_scatter_sender.cpp";
    static std::string const& line_start_sender_kernel_path = "ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send.cpp";
    static std::string const& reduce_kernel_path = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp";

    // Need to be able to split up the workers so that on the end of the lines, some of the cores are for send/receive and
    // others are for CCL send only
    bool is_start_chip_in_line = topology_config.is_linear && (topology_config.ring_index == 0 || topology_config.ring_index == topology_config.ring_size - 1);

    // If we we implementing a line, and are at the end of the line
    bool worker_grid_split_in_half = is_start_chip_in_line;

    KernelHandle worker_receiver_kernel_id, worker_sender_kernel_id, worker_reduce_kernel_id;
    std::optional<KernelHandle> line_start_sender_kernel_id;

    worker_receiver_kernel_id = tt::tt_metal::CreateKernel(
        program,
        receiver_kernel_path,
        worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(worker_arg_builder.generate_receiver_kernel_ct_args(), worker_defines));

    worker_sender_kernel_id = tt::tt_metal::CreateKernel(
        program,
        forward_sender_kernel_path,
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

    if (is_start_chip_in_line) {
        TT_ASSERT(split_worker_core_range.has_value(), "Internal Error. (line) Reduce scatter did not generate a smaller second worker grid to map the line start kernels onto");
        log_trace(tt::LogOp, "Invoking CCL send kernel on split kernel core range");
        for (auto const& core : corerange_to_cores(split_worker_core_range.value())) {
            log_trace(tt::LogOp, "\tcore=(x={},y={})", core.x, core.y);
        }
        line_start_sender_kernel_id = tt::tt_metal::CreateKernel(
            program,
            line_start_sender_kernel_path,
            split_worker_core_range.value(),
            tt::tt_metal::WriterDataMovementConfig(worker_arg_builder.generate_line_start_sender_kernel_ct_args(), worker_defines));
    }

    return {worker_receiver_kernel_id, worker_sender_kernel_id, worker_reduce_kernel_id, line_start_sender_kernel_id};
}

static void set_reduce_scatter_worker_rt(
    tt::tt_metal::Program& program,
    Device const* device,
    KernelHandle worker_receiver_kernel_id,
    KernelHandle worker_sender_kernel_id,
    KernelHandle worker_reduce_kernel_id,
    std::optional<KernelHandle> optional_line_start_ccl_send_kernel,
    ttnn::ccl::RingTopology const& topology_config,
    ttnn::ccl::reduce_scatter_detail::ReduceScatterWorkerArgBuilder const& worker_arg_builder,
    std::vector<ttnn::ccl::EriscDatamoverBuilder>& cw_edm_builders,
    std::vector<ttnn::ccl::EriscDatamoverBuilder>& ccw_edm_builders,
    EdmInterfaceAddresses const& edm_interface_addresses,
    WorkerAttributes &worker_attributes,
    std::size_t num_edm_channels,
    std::size_t edm_num_buffers_per_channel,
    ttnn::operations::binary::BinaryOpType binary_math_op) {
    bool is_in_clockwise_direction = worker_attributes.direction == Direction::CLOCKWISE;
    const std::size_t global_worker_index = get_global_worker_id(worker_attributes, num_edm_channels);

    if (!topology_config.is_first_device_in_line(is_in_clockwise_direction))
    {
        CoreCoord const& receiver_edm = is_in_clockwise_direction
                                            ? topology_config.eth_receiver_cores.at(worker_attributes.link)
                                            : topology_config.eth_sender_cores.at(worker_attributes.link);
        ttnn::ccl::WorkerXY receiver_edm_noc_coord = ttnn::ccl::WorkerXY(
            device->ethernet_core_from_logical_core(receiver_edm).x,
            device->ethernet_core_from_logical_core(receiver_edm).y);
        const uint32_t edm_core_semaphore_address = edm_interface_addresses.worker_receiver_edm_semaphore_addresses.at(global_worker_index);
        const uint32_t edm_core_buffer_address = edm_interface_addresses.worker_receiver_edm_buffer_addresses.at(global_worker_index);

        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_receiver_kernel_id,
            worker_attributes.location_logical,
            worker_arg_builder.generate_receiver_kernel_rt_args(
                receiver_edm_noc_coord, edm_core_semaphore_address, edm_core_buffer_address, worker_attributes));

        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_reduce_kernel_id,
            worker_attributes.location_logical,
            worker_arg_builder.generate_reduce_op_kernel_rt_args(worker_attributes, topology_config.ring_size));
    }

    {
        ttnn::ccl::WorkerXY edm_noc_coord = ttnn::ccl::WorkerXY(0,0);
        uint32_t edm_core_semaphore_address = 0;
        uint32_t edm_core_buffer_address = 0;

        // If we are at the end of a line, then the sender kernel does not forward anything to an EDM
        if (!topology_config.is_last_device_in_line(is_in_clockwise_direction)) {
            CoreCoord sender_edm = is_in_clockwise_direction ? topology_config.eth_sender_cores.at(worker_attributes.link)
                                                            : topology_config.eth_receiver_cores.at(worker_attributes.link);
            edm_noc_coord = ttnn::ccl::WorkerXY(
                device->ethernet_core_from_logical_core(sender_edm).x, device->ethernet_core_from_logical_core(sender_edm).y);
            TT_ASSERT(edm_noc_coord.y == 0 || edm_noc_coord.y == 6);
            edm_core_semaphore_address = edm_interface_addresses.worker_sender_edm_semaphore_addresses.at(global_worker_index);
            edm_core_buffer_address = edm_interface_addresses.worker_sender_edm_buffer_addresses.at(global_worker_index);
        }

        WorkerEdmInterfaceArgs edm_interface = {
            edm_noc_coord.x,
            edm_noc_coord.y,
            edm_core_buffer_address,
            edm_core_semaphore_address,
            edm_num_buffers_per_channel};

        bool use_line_start_kernel = topology_config.is_first_device_in_line(is_in_clockwise_direction);
        if (use_line_start_kernel) {
            log_trace(tt::LogOp, "Setting CCL send RT args");
        }
        auto const rt_args = use_line_start_kernel
                                 ? worker_arg_builder.generate_line_start_sender_kernel_rt_args(
                                       edm_interface, worker_arg_builder.scatter_dim, worker_attributes)
                                 : worker_arg_builder.generate_sender_kernel_rt_args(edm_interface, worker_attributes);
        TT_ASSERT(!use_line_start_kernel || optional_line_start_ccl_send_kernel.has_value());
        auto sender_kernel_id = use_line_start_kernel ? optional_line_start_ccl_send_kernel.value(): worker_sender_kernel_id;

        log_trace(tt::LogOp, "{} rt_args for sender kernel", rt_args.size());
        tt::tt_metal::SetRuntimeArgs(
            program,
            sender_kernel_id,
            worker_attributes.location_logical,
            rt_args);
    }
}

/*
 * Core range sets for line topology
 */
static std::pair<CoreRangeSet, std::optional<CoreRangeSet>> select_worker_cores_for_line_topology(ttnn::ccl::RingTopology const& topology_config, ttnn::ccl::CCLOpConfig const& op_config, std::size_t num_links, std::size_t num_edm_channels) {
    static constexpr std::size_t num_directions_per_line = 2;

    TT_ASSERT(num_edm_channels % 2 == 0, "For line topologies, we expect a multiple of 2 number of channels for the algorithm and worker kernels to work.");
    const std::size_t workers_per_direction = num_edm_channels / num_directions_per_line;
    auto const& lower_half_of_cores =
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(workers_per_direction - 1, num_links - 1)));
    auto const& upper_half_of_cores = CoreRangeSet(
        CoreRange(CoreCoord(workers_per_direction, 0), CoreCoord(num_edm_channels - 1, num_links - 1)));
    if (topology_config.ring_index == 0) {
        log_trace(tt::LogOp, "Start of line, putting CCL send cores in lower half");
        return {upper_half_of_cores, lower_half_of_cores};
    } else if (topology_config.ring_index == topology_config.ring_size - 1) {
        // Flip them for the other end because the send will be for the "second" core range set (conceptually, the other direction)
        // of the line flows in the second half of all workers, for each chip.
        log_trace(tt::LogOp, "End of line, putting CCL send cores in lower half");
        return {lower_half_of_cores, upper_half_of_cores};
    } else {
        log_trace(tt::LogOp, "Middle of line - no CCL kernel");
        return {
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_edm_channels - 1, num_links - 1))),
            std::nullopt};
    }
}

/*
 * Returns 1 or 2 core range sets. Typically returns only one but in the case of a line reduce scatter where we are at the end of the line,
 * then we must split the core range in half (and return 2), one for each direction where half the cores will invoke the ccl::send kernel
 * to implement the start of the line and the others will invoke the typical reduce scatter worker kernels.
 */
static std::pair<CoreRangeSet, std::optional<CoreRangeSet>> select_worker_cores(
    ttnn::ccl::RingTopology const& topology_config, ttnn::ccl::CCLOpConfig const& op_config, std::size_t num_links, std::size_t num_edm_channels) {
    switch (op_config.get_topology()) {
        case ttnn::ccl::Topology::Linear: {
            auto const& core_ranges = select_worker_cores_for_line_topology(topology_config, op_config, num_links, num_edm_channels);
            log_trace(tt::LogOp, "First core range");
            for (const auto &core : corerange_to_cores(core_ranges.first)) {
                log_trace(tt::LogOp, "\tx={},y={}", core.x, core.y);
            }
            if (core_ranges.second.has_value()) {
                log_trace(tt::LogOp, "second worker core list:");
                for (const auto &core : corerange_to_cores(core_ranges.second.value())) {
                    log_trace(tt::LogOp, "\tx={},y={}", core.x, core.y);
                }
            }
            return core_ranges;
        }

        case ttnn::ccl::Topology::Ring:
            return {
                CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_edm_channels - 1, num_links - 1))),
                std::nullopt};

        default: TT_ASSERT(false, "Unsupported topology"); return {CoreRangeSet(), std::nullopt};
    };
}

static WorkerTransferInfo compute_num_edm_messages_per_channel(
    ccl::CCLOpConfig const& op_config,
    RingReduceScatterWrappedTensorSlicer& tensor_slicer,  // TODO: Update to Generic ReduceScatterSlicer when it is implemented
    ttnn::ccl::RingTopology const& topology_config,
    std::vector<ttnn::ccl::EriscDatamoverBuilder> const& cw_per_link_edm_builders,
    std::vector<ttnn::ccl::EriscDatamoverBuilder> const& ccw_per_link_edm_builders,
    std::size_t const num_edm_channels
    ) {
    uint32_t const page_size_in_bytes = op_config.get_page_size();
    TT_ASSERT(num_edm_channels > 0);
    TT_ASSERT(topology_config.num_links > 0);
    TT_ASSERT(page_size_in_bytes > 0);
    log_trace(tt::LogOp, "WorkerTransferInfo");
    const std::size_t num_links = topology_config.num_links;
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

    std::vector<uint32_t> num_pages_per_full_chunk(total_num_edm_channels, 0);

    for (std::size_t link = 0; link < num_links; link++) {
        const auto& an_edm_builder = cw_per_link_edm_builders.size() > 0 ? cw_per_link_edm_builders.at(link) : ccw_per_link_edm_builders.at(link);
        TT_ASSERT(cw_per_link_edm_builders.size() > 0 || topology_config.ring_index == topology_config.ring_size - 1, "Internal logic error");
        std::size_t edm_channel_size_in_bytes = an_edm_builder.get_eth_buffer_size_bytes();
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
    ttnn::ccl::Topology topology,
    uint32_t cb_src0_size_pages,
    uint32_t cb_dst0_size_pages,
    uint32_t cb_short_circuit_size_pages,
    std::size_t edm_channel_buffer_size,
    uint32_t page_size) {
    switch (topology) {
        case ttnn::ccl::Topology::Linear:
            // For linear topology, we only want one slice per worker so we don't
            return std::numeric_limits<uint32_t>::max();

        case ttnn::ccl::Topology::Ring:
            return std::min(cb_short_circuit_size_pages, cb_src0_size_pages + cb_dst0_size_pages) * page_size +
                   edm_channel_buffer_size;

        default: TT_ASSERT(false, "Unsupported topology"); return 0;
    };
}

static bool is_cb_buffering_sufficient_to_avoid_deadlock(
    ttnn::ccl::Topology topology,
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
        topology, cb_src0_size_pages, cb_dst0_size_pages, cb_short_circuit_size_pages, edm_channel_buffer_size, page_size);
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

static std::tuple<
    CBHandle,
    CBHandle,
    CBHandle,
    CBHandle,
    std::optional<CBHandle>,
    std::optional<CBHandle>,
    std::optional<CBHandle>,
    std::optional<CBHandle>>
create_worker_circular_buffers(
    Tensor const& input_tensor,
    ttnn::ccl::CCLOpConfig const& op_config,
    CoreRangeSet const& worker_core_range,
    std::optional<CoreRangeSet> const& second_worker_core_range,
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
    std::optional<CBHandle> cb_src0_workers_2;
    if (second_worker_core_range.has_value()) {
        cb_src0_workers_2 = CreateCircularBuffer(program, second_worker_core_range.value(), cb_src0_config);
    }

    // Input 1 CB
    uint32_t src1_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(worker_pages_per_transfer * page_size_bytes, {{src1_cb_index, df}})
            .set_page_size(src1_cb_index, page_size_bytes);
    CBHandle cb_src1_workers = CreateCircularBuffer(program, worker_core_range, cb_src1_config);
    std::optional<CBHandle> cb_src1_workers_2;
    if (second_worker_core_range.has_value()) {
        cb_src1_workers_2 = CreateCircularBuffer(program, second_worker_core_range.value(), cb_src1_config);
    }

    // Dataflow Writer Kernel input CB
    uint32_t cb_dst0_index = tt::CB::c_out0;
    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(worker_pages_per_transfer * page_size_bytes, {{cb_dst0_index, df}})
            .set_page_size(cb_dst0_index, page_size_bytes);
    CBHandle cb_dst0_sender_workers = CreateCircularBuffer(program, worker_core_range, cb_dst0_config);
    std::optional<CBHandle> cb_dst0_sender_workers_2;
    if (second_worker_core_range.has_value()) {
        cb_dst0_sender_workers_2 = CreateCircularBuffer(program, second_worker_core_range.value(), cb_dst0_config);
    }

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

    std::optional<CBHandle> cb_short_circuit_sender_workers_2;
    if (second_worker_core_range.has_value()) {
        cb_short_circuit_sender_workers_2 =
            CreateCircularBuffer(program, second_worker_core_range.value(), cb_short_circuit_config);
    }

    return {
        cb_src0_workers,
        cb_src1_workers,
        cb_dst0_sender_workers,
        cb_short_circuit_sender_workers,
        cb_src0_workers_2,
        cb_src1_workers_2,
        cb_dst0_sender_workers_2,
        cb_short_circuit_sender_workers_2};
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
    std::size_t input_tensor_num_units_per_tensor_slice =
        input_tensor_n_elems_per_slice / (tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT);

    TT_ASSERT(input_tensor_num_units_per_tensor_slice > 0);
    constexpr bool enable_bidirectional = true;
    constexpr std::size_t default_num_workers = 8;
    uint32_t max_num_workers = std::min<std::size_t>(user_defined_num_workers.value_or(default_num_workers), input_tensor_num_units_per_tensor_slice);
    if (topology == ttnn::ccl::Topology::Linear) {
        max_num_workers = std::max<std::size_t>(max_num_workers, 2);
    }
    auto num_edm_channels_per_link = decide_number_of_edm_channels(op_config, max_num_workers, enable_bidirectional);
    log_trace(tt::LogOp, "num_edm_channels_per_link: {}", num_edm_channels_per_link);
    auto edm_termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED;

    std::size_t num_buffers_per_channel = 2;
    if (user_defined_num_buffers_per_channel.has_value()) {
        // Override with user defined value
        num_buffers_per_channel = user_defined_num_buffers_per_channel.value();
    }
    auto const& edm_builder = create_erisc_datamover_builder(
        num_edm_channels_per_link, op_config.get_page_size(), num_buffers_per_channel, buffer_sharing_mode, edm_termination_mode);
    TT_ASSERT(num_edm_channels_per_link > 0);

    const auto& device = input_tensor.device();
    auto const& topology_config =
       ttnn::ccl::RingTopology(device, topology, sender_device_id, receiver_device_id, num_links, ring_size, ring_index);
    bool is_linear = topology_config.is_linear;
    // For line reduce scatter we have special instantiation behaviour at the ends of the line, namely:
    // The start of the line has no counter-clockwise EDM and the end has no clockwise EDM
    std::size_t num_active_cw_edm_links = (!is_linear || (ring_index != ring_size - 1)) ? num_links : 0;
    std::size_t num_active_ccw_edm_links = (!is_linear || (ring_index != 0)) ? num_links : 0;
    log_trace(tt::LogOp, "ring_index: {}, num_active_cw_edm_links: {}, num_active_ccw_edm_links: {}", ring_index, num_active_cw_edm_links, num_active_ccw_edm_links);
    std::vector<ttnn::ccl::EriscDatamoverBuilder> cw_per_link_edm_builders(num_active_cw_edm_links, edm_builder);
    std::vector<ttnn::ccl::EriscDatamoverBuilder> ccw_per_link_edm_builders(num_active_ccw_edm_links, edm_builder);
    TT_ASSERT(cw_per_link_edm_builders.size() > 0 ||  ccw_per_link_edm_builders.size() > 0, "Internal error. No EDMs were instantiated in reduce scatter.");

    std::function<bool(uint32_t)> is_worker_in_clockwise_direction_fn = [is_linear, enable_bidirectional, num_edm_channels_per_link](std::size_t x) {
                static constexpr std::size_t bidirectional_directions = 2;
                return is_linear ? (x < (num_edm_channels_per_link / bidirectional_directions)):
                    enable_bidirectional ? (x % bidirectional_directions == 0) : true;
            };

    auto const& [worker_core_range, second_worker_core_range] = select_worker_cores(topology_config, op_config, num_links, num_edm_channels_per_link);
    auto const& worker_cores = corerange_to_cores(worker_core_range, std::nullopt, true);
    std::optional<std::vector<CoreCoord>> second_worker_cores_list;
    if (second_worker_core_range.has_value()) {
        second_worker_cores_list = corerange_to_cores(second_worker_core_range.value(), std::nullopt, true);
    }

    //////////////////
    tt::tt_metal::Program program{};

    // Semaphores && CBs
    auto worker_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, worker_core_range, 0);
    auto worker_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, worker_core_range, 0);
    std::optional<uint32_t> worker_receiver_semaphore_id_second_core_range = std::nullopt;
    std::optional<uint32_t> worker_sender_semaphore_id_second_core_range = std::nullopt;
    std::optional<uint32_t> receiver_worker_partial_ready_semaphore_id = std::nullopt;
    std::optional<uint32_t> receiver_worker_partial_ready_semaphore_id_second_core_range = std::nullopt;
    if (second_worker_core_range.has_value()) {
        worker_receiver_semaphore_id_second_core_range = tt::tt_metal::CreateSemaphore(program, second_worker_core_range.value(), 0);
        worker_sender_semaphore_id_second_core_range = tt::tt_metal::CreateSemaphore(program, second_worker_core_range.value(), 0);
    }
    if (topology_config.is_linear) {
        receiver_worker_partial_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, worker_core_range, 0);
        if (second_worker_core_range.has_value()) {
            receiver_worker_partial_ready_semaphore_id_second_core_range = tt::tt_metal::CreateSemaphore(program, second_worker_core_range.value(), 0);
        }
    }

    std::vector<WorkerAttributes> all_worker_attributes = build_worker_attributes(
        topology_config,
        worker_cores,
        second_worker_cores_list,

        worker_sender_semaphore_id,
        worker_receiver_semaphore_id,
        worker_sender_semaphore_id_second_core_range,
        worker_receiver_semaphore_id_second_core_range,

        num_links,
        num_edm_channels_per_link,
        is_worker_in_clockwise_direction_fn);

    const std::size_t edm_buffer_size_bytes = (cw_per_link_edm_builders.size() > 0 ? cw_per_link_edm_builders.at(0) : ccw_per_link_edm_builders.at(0)).get_eth_buffer_size_bytes();
    uint32_t cb_num_pages = std::min(input_tensor_num_units_per_tensor_slice, (edm_buffer_size_bytes / op_config.get_page_size())) * 2;
    uint32_t cb_num_pages_per_packet = cb_num_pages / 2;
    log_trace(tt::LogOp, "cb_num_pages: {}", cb_num_pages);
    auto const& [cb_src0_workers, cb_src1_workers, cb_dst0_sender_workers, cb_short_circuit_sender_workers, optional_cb_src0_workers_2, optional_cb_src1_workers_2, optional_cb_dst0_sender_workers_2, optional_cb_short_circuit_sender_workers_2] =
        create_worker_circular_buffers(
            input_tensor, op_config, worker_core_range, second_worker_core_range, cb_num_pages, program);

    uint32_t max_worker_slice_in_bytes = compute_maximum_worker_slice_in_bytes(
        topology,
        cb_num_pages,
        cb_num_pages,
        cb_num_pages,
        edm_buffer_size_bytes,
        op_config.get_page_size());
    const std::size_t num_workers = all_worker_attributes.size();
    TT_ASSERT(num_workers == num_edm_channels_per_link * num_links);
    // For tensor slicer purposes, if we are working with a linear topology, then half of
    // the workers will be for one direction of the line and the other half will be for
    // the other. Therefore, for each tensor slice, only half of the total workers are available
    // to work on it.
    const std::size_t num_workers_per_slicer = topology_config.is_linear ? num_workers / 2 : num_workers;
    auto tensor_slicer = ttnn::ccl::RingReduceScatterWrappedTensorSlicer(
        input_tensor,
        output_tensor,
        scatter_split_dim,
        ring_index,
        ring_size,
        num_workers_per_slicer,
        max_worker_slice_in_bytes,
        cb_num_pages / 2);

    // Not per buffer because the buffer sharing mode may cause some buffers to share EDM transfers
    WorkerTransferInfo const& worker_transfer_info = compute_num_edm_messages_per_channel(
        op_config,
        tensor_slicer,
        topology_config,
        cw_per_link_edm_builders,
        ccw_per_link_edm_builders,
        num_edm_channels_per_link);

    // Configure the EDM builders
    EdmInterfaceAddresses edm_interface_addresses;
    for (std::size_t link = 0; link < num_links; link++) {
        add_worker_config_to_edm_builders(
            device,
            tensor_slicer,
            all_worker_attributes,
            num_edm_channels_per_link,
            num_buffers_per_channel,

            cw_per_link_edm_builders,
            ccw_per_link_edm_builders,

            topology_config,
            link,

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
        edm_termination_mode, // Can probably remove this once everything is working
        scatter_split_dim,
        cb_num_pages_per_packet,
        receiver_worker_partial_ready_semaphore_id, // This one should go too but not sure how yet
        num_buffers_per_channel);
    auto [worker_receiver_kernel_id, worker_sender_kernel_id, worker_reduce_kernel_id, optional_line_start_ccl_send_kernel] = build_reduce_scatter_worker_ct(
        program,
        topology_config,
        op_config,
        worker_arg_builder,
        worker_core_range,
        second_worker_core_range,
        reduce_op);

    // build the worker kernels

    // set worker kernels rt
    tt::tt_metal::ComputeConfig compute_config;
    for (std::size_t link = 0; link < num_links; link++) {
        log_trace(tt::LogOp, "==============================================");
        log_trace(tt::LogOp, "------------------ Link: {} ------------------", link);
        for (std::size_t worker = 0; worker < num_edm_channels_per_link; worker++) {
            std::size_t global_worker_index = worker + link * num_edm_channels_per_link;

            log_trace(tt::LogOp, "------ Worker: {} (global ID={})", worker, global_worker_index);

            std::size_t worker_tensor_slice_index = get_worker_index_in_slice(topology_config, global_worker_index, worker, num_edm_channels_per_link, link);
            auto const& worker_slice = tensor_slicer.get_worker_slice(worker_tensor_slice_index);
            auto worker_arg_builder = ReduceScatterWorkerArgBuilder(
                device,
                op_config,
                topology_config,
                worker_slice,
                worker_transfer_info,
                edm_termination_mode,
                scatter_split_dim,
                cb_num_pages_per_packet,
                receiver_worker_partial_ready_semaphore_id,
                num_buffers_per_channel);

            set_reduce_scatter_worker_rt(
                program,
                device,
                worker_receiver_kernel_id,
                worker_sender_kernel_id,
                worker_reduce_kernel_id,
                optional_line_start_ccl_send_kernel,
                topology_config,
                worker_arg_builder,
                cw_per_link_edm_builders,
                ccw_per_link_edm_builders,
                edm_interface_addresses,
                all_worker_attributes.at(global_worker_index),
                num_edm_channels_per_link,
                num_buffers_per_channel,
                reduce_op);

            TT_FATAL(is_cb_buffering_sufficient_to_avoid_deadlock(
                    topology,
                    worker_slice,
                    cb_num_pages,
                    cb_num_pages,
                    cb_num_pages,
                    edm_buffer_size_bytes,
                    op_config.get_page_size()), "Internal error: reduce scatter implementation generated a program that will deadlock due to insufficient buffering based on the tensor slice sizes the op chose to use.");
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

    auto override_runtime_arguments_callback =
        [topology_config, worker_receiver_kernel_id, worker_sender_kernel_id, optional_line_start_ccl_send_kernel, worker_cores, second_worker_cores_list](
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
                worker_receiver_runtime_args.at(1) = output.buffer()->address();

                auto& worker_sender_runtime_args = worker_sender_runtime_args_by_core[core.x][core.y];
                worker_sender_runtime_args.at(0) = output.buffer()->address();
            }

            if (second_worker_cores_list.has_value()) {
                TT_FATAL(optional_line_start_ccl_send_kernel.has_value(), "Internal error: line start CCL send kernel was not found but we split the worker grid to place it onto some worker cores");
                auto const &line_start_worker_cores = second_worker_cores_list.value();
                auto &ccl_send_kernel_rt_args_by_core = GetRuntimeArgs(program, optional_line_start_ccl_send_kernel.value());
                for (auto const& core : line_start_worker_cores) {
                    auto& line_start_kernel_rt_args = ccl_send_kernel_rt_args_by_core[core.x][core.y];
                    line_start_kernel_rt_args.at(0) = input.buffer()->address();
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace reduce_scatter_detail
}  // namespace ccl
}  // namespace ttnn

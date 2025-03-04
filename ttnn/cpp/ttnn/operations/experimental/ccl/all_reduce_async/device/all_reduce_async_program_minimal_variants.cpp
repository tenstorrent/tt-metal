// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "all_reduce_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "cpp/ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>
using namespace tt::constants;

namespace ttnn {

using namespace ccl;

CoreRangeSet cores_to_corerangeset(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> core_ranges;
    for (const auto& core : cores) {
        core_ranges.push_back(CoreRange(core));
    }
    return CoreRangeSet(core_ranges);
}

std::optional<std::vector<CoreCoord>> reorder_ethernet_cores(
    IDevice* device, std::optional<IDevice*> remote_device, uint32_t num_links) {
    if (remote_device.has_value()) {
        uint32_t size = device->get_ethernet_sockets(remote_device.value()->id()).size();
        std::vector<CoreCoord> reordered_ethernet_cores;
        reordered_ethernet_cores.reserve(size);
        std::vector<std::pair<CoreCoord, CoreCoord>> ethernet_cores_logical_physical;
        ethernet_cores_logical_physical.reserve(size);

        for (auto core : device->get_ethernet_sockets(remote_device.value()->id())) {
            auto core_physical = device->virtual_core_from_logical_core(core, CoreType::ETH);
            ethernet_cores_logical_physical.emplace_back(core, core_physical);
        }
        std::sort(ethernet_cores_logical_physical.begin(), ethernet_cores_logical_physical.end(), [](auto& a, auto& b) {
            return a.second.x < b.second.x;
        });

        for (auto& core_pair : ethernet_cores_logical_physical) {
            reordered_ethernet_cores.push_back(core_pair.first);
        }
        return reordered_ethernet_cores;
    }

    return std::nullopt;
}

std::vector<CoreCoord> compute_top_row_ethernet_cores(
    IDevice* device,
    std::optional<std::vector<CoreCoord>> forward_ethernet_cores,
    std::optional<std::vector<CoreCoord>> backward_ethernet_cores) {
    std::vector<CoreCoord> ethernet_cores;
    if (forward_ethernet_cores.has_value()) {
        ethernet_cores = forward_ethernet_cores.value();
    } else if (backward_ethernet_cores.has_value()) {
        ethernet_cores = backward_ethernet_cores.value();
    }

    std::vector<CoreCoord> ethernet_cores_virtual;
    for (auto core : ethernet_cores) {
        auto core_virtual = device->virtual_core_from_logical_core(core, CoreType::ETH);
        ethernet_cores_virtual.push_back(core_virtual);
    }

    for (auto& eth_core : ethernet_cores_virtual) {
        eth_core.y = 16;
    }
    return ethernet_cores_virtual;
}

std::pair<CoreRangeSet, std::vector<CoreCoord>> get_optimal_worker_core_placement(
    IDevice* device,
    std::vector<CoreCoord> ethernet_cores_virtual,
    std::optional<CoreRangeSet> reserved_cores,
    uint32_t num_links,
    const std::optional<SubDeviceId>& sub_device_id) {
    std::vector<CoreCoord> sender_worker_cores;
    std::vector<CoreCoord> sender_worker_cores_physical;

    auto available_cores_corerangeset = device->worker_cores(
        HalProgrammableCoreType::TENSIX,
        sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
    auto available_cores = corerange_to_cores(available_cores_corerangeset, std::nullopt, true);
    std::vector<CoreCoord> available_cores_physical;
    for (auto available_core : available_cores) {
        auto available_core_physical = device->physical_worker_core_from_logical_core(available_core);
        available_cores_physical.push_back(available_core_physical);
    }

    // if (device->id() == 4)
    // tt::log_info("available_cores_physical: {}", available_cores_physical);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreCoord right_col_compute_core_physical =
        device->physical_worker_core_from_logical_core(CoreCoord(num_cores_x - 1, 0));
    // Get all logical cores in the worker grid
    std::vector<CoreCoord> compute_cores_logical;
    for (int i = 0; i < num_cores_x; ++i) {
        for (int j = 0; j < num_cores_y; ++j) {
            compute_cores_logical.push_back(CoreCoord(i, j));
        }
    }

    // get reserved core placement
    std::vector<CoreCoord> reserved_cores_physical;
    if (reserved_cores.has_value()) {
        auto reserved_cores_vector = corerange_to_cores(reserved_cores.value(), std::nullopt, true);
        for (auto reserved_core : reserved_cores_vector) {
            auto reserved_core_physical = device->physical_worker_core_from_logical_core(reserved_core);
            reserved_cores_physical.push_back(reserved_core_physical);
        }
    }

    // if (device->id() == 4)
    // tt::log_info("reserved_cores_physical: {}", reserved_cores_physical);

    // get FD core placement
    std::vector<CoreCoord> occupied_cores_in_column_physical;
    auto full_grid_size = device->grid_size();
    uint32_t full_grid_size_x = full_grid_size.x;
    uint32_t full_grid_size_y = full_grid_size.y;
    for (uint32_t grid_y = 0; grid_y < full_grid_size_y; ++grid_y) {
        for (uint32_t grid_x = 0; grid_x < full_grid_size_x; ++grid_x) {
            auto core_physical = CoreCoord(grid_x, grid_y);
            if (core_physical.x > right_col_compute_core_physical.x) {
                occupied_cores_in_column_physical.push_back(core_physical);
            }
        }
    }

    for (uint32_t link = 0; link < num_links; link++) {
        auto core_virtual = ethernet_cores_virtual[link];
        CoreCoord eth_core_physical;
        eth_core_physical.x = core_virtual.x >= 22 ? (core_virtual.x - 16) : (core_virtual.x - 17);
        eth_core_physical.y = (core_virtual.y - 16) * 6;

        // shift down the worker core
        auto worker_core_physical = CoreCoord(eth_core_physical.x, eth_core_physical.y + 1);

        // check the reserved cores
        uint32_t start_core_y = worker_core_physical.y;
        uint32_t end_core_y = num_cores_y;
        for (uint32_t core_y = start_core_y; core_y < end_core_y; ++core_y) {
            auto it = std::find(reserved_cores_physical.begin(), reserved_cores_physical.end(), worker_core_physical);

            auto it_subdevice =
                std::find(available_cores_physical.begin(), available_cores_physical.end(), worker_core_physical);

            worker_core_physical.y = core_y;

            // if (device->id() == 4)
            // tt::log_info("worker_core_physical: {}", worker_core_physical);

            if (it == reserved_cores_physical.end() && it_subdevice != available_cores_physical.end()) {  // not find
                break;
            }
        }
        auto it_subdevice =
            std::find(available_cores_physical.begin(), available_cores_physical.end(), worker_core_physical);
        auto it = std::find(reserved_cores_physical.begin(), reserved_cores_physical.end(), worker_core_physical);
        TT_FATAL(
            it == reserved_cores_physical.end() && it_subdevice != available_cores_physical.end(),
            "iter 1 worker core cannot find a coord to be placed");

        sender_worker_cores_physical.push_back(worker_core_physical);
    }

    // check the occupied the columns by FD
    auto it = std::find(
        occupied_cores_in_column_physical.begin(),
        occupied_cores_in_column_physical.end(),
        sender_worker_cores_physical[num_links - 1]);
    if (it != occupied_cores_in_column_physical.end()) {  // find, shift core left by 1
        sender_worker_cores_physical[num_links - 1].x -= 1;
        // shift down the worker core
        auto worker_core_physical = sender_worker_cores_physical[num_links - 1];
        // check the reserved cores
        uint32_t start_core_y = worker_core_physical.y;
        uint32_t end_core_y = num_cores_y;
        for (uint32_t core_y = start_core_y; core_y < end_core_y; ++core_y) {
            auto it_subdevice =
                std::find(available_cores_physical.begin(), available_cores_physical.end(), worker_core_physical);
            auto it = std::find(reserved_cores_physical.begin(), reserved_cores_physical.end(), worker_core_physical);
            auto it2 = std::find(
                sender_worker_cores_physical.begin(), sender_worker_cores_physical.end(), worker_core_physical);

            worker_core_physical.y = core_y;
            if (it == reserved_cores_physical.end() && it2 == sender_worker_cores_physical.end() &&
                it_subdevice != available_cores_physical.end()) {  // not find
                break;
            }
        }
        auto it_subdevice =
            std::find(available_cores_physical.begin(), available_cores_physical.end(), worker_core_physical);
        auto it = std::find(reserved_cores_physical.begin(), reserved_cores_physical.end(), worker_core_physical);
        auto it2 =
            std::find(sender_worker_cores_physical.begin(), sender_worker_cores_physical.end(), worker_core_physical);
        TT_FATAL(
            it == reserved_cores_physical.end() && it2 == sender_worker_cores_physical.end() &&
                it_subdevice != available_cores_physical.end(),
            "iter 2 worker core cannot find a coord to be placed");

        sender_worker_cores_physical[num_links - 1] = worker_core_physical;
    }

    // Convert to physical worker coordinates to logical.
    for (int i = 0; i < sender_worker_cores_physical.size(); ++i) {
        for (int j = 0; j < compute_cores_logical.size(); ++j) {
            auto core = device->physical_worker_core_from_logical_core(compute_cores_logical[j]);
            if (sender_worker_cores_physical[i] == core) {
                sender_worker_cores.push_back(compute_cores_logical[j]);
            }
        }
    }

    // if (device->id() == 35) {
    //     tt::log_info("dev {} sender_worker_cores: {}", device->id(), sender_worker_cores);
    // }

    // sender_worker_cores.clear();
    // sender_worker_cores.push_back(CoreCoord(7,2));
    // sender_worker_cores.push_back(CoreCoord(7,3));
    // sender_worker_cores.push_back(CoreCoord(7,4));
    // sender_worker_cores.push_back(CoreCoord(7,5));
    //

    // if (device->id() == 4) {
    //     sender_worker_cores.push_back(CoreCoord(0,5));
    //     sender_worker_cores.push_back(CoreCoord(1,5));
    //     sender_worker_cores.push_back(CoreCoord(6,5));
    //     sender_worker_cores.push_back(CoreCoord(7,5));
    // } else {
    //     sender_worker_cores.push_back(CoreCoord(0,3));
    //     sender_worker_cores.push_back(CoreCoord(1,3));
    //     sender_worker_cores.push_back(CoreCoord(6,3));
    //     sender_worker_cores.push_back(CoreCoord(7,3));
    // }

    // if (device->id() == 4)
    // tt::log_info("sender_worker_cores: {}", sender_worker_cores);

    std::set<CoreRange> sender_worker_cores_set;
    for (int i = 0; i < sender_worker_cores.size(); ++i) {
        sender_worker_cores_set.insert(CoreRange(sender_worker_cores[i]));
    }
    CoreRangeSet sender_worker_corerangeset = CoreRangeSet(sender_worker_cores_set);

    return {sender_worker_corerangeset, sender_worker_cores};
}

operation::ProgramWithCallbacks all_reduce_async_minimal_multi_core_with_workers(
    const Tensor& input_tensor,
    const Tensor& all_gather_output_tensor,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<SubDeviceId>& sub_device_id,
    bool enable_persistent_fabric_mode) {
    tt::tt_metal::Program program{};
    const bool enable_async_output_tensor = false;
    TT_FATAL(
        enable_persistent_fabric_mode,
        "only persistent fabric mode is supported for all_gather_async_llama_post_binary_matmul");

    IDevice* device = input_tensor.device();
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    std::optional<std::vector<CoreCoord>> forward_ethernet_cores =
        reorder_ethernet_cores(device, forward_device, num_links);
    std::optional<std::vector<CoreCoord>> backward_ethernet_cores =
        reorder_ethernet_cores(device, backward_device, num_links);

    // std::optional<ttnn::ccl::EdmLineFabricOpInterface> local_fabric_handle =
    //     ttnn::ccl::EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
    //         device,
    //         forward_device.value_or(nullptr),
    //         backward_device.value_or(nullptr),
    //         &program,
    //         enable_persistent_fabric_mode,
    //         num_links,
    //         std::array{forward_ethernet_cores, backward_ethernet_cores});
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> local_fabric_handle =
        ttnn::ccl::EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
            device,
            forward_device.value_or(nullptr),
            backward_device.value_or(nullptr),
            &program,
            enable_persistent_fabric_mode,
            num_links);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    LineTopology line_topology(ring_size, ring_index);
    const size_t num_targets_forward =
        line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::FORWARD);
    const size_t num_targets_backward =
        line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD);
    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec->shape;
    const auto input_tensor_shard_num_pages = input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / TILE_HW;
    const auto num_input_cores = input_tensor_cores.num_cores();
    const auto output_tensor_num_pages = output_tensor.buffer()->num_pages();
    const auto output_tensor_cores = output_tensor.memory_config().shard_spec->grid;
    const auto output_tensor_shard_shape = output_tensor.memory_config().shard_spec->shape;
    const auto output_tensor_shard_num_pages = output_tensor_shard_shape[0] * output_tensor_shard_shape[1] / TILE_HW;
    const auto num_output_cores = output_tensor_cores.num_cores();

    // Get worker cores, assuming 1 worker per link
    std::optional<CoreRangeSet> reserved_cores = output_tensor_cores;
    uint32_t num_workers_per_link = 1;

    TT_FATAL(
        backward_ethernet_cores.has_value() || forward_ethernet_cores.has_value(),
        "at least one dir need to have eth cores");
    TT_FATAL(reserved_cores.has_value(), "reserved core no value");

    // std::vector<CoreCoord> ethernet_cores_virtual =
    //     compute_top_row_ethernet_cores(device, forward_ethernet_cores, backward_ethernet_cores);
    // tt::log_info("dev {} ethernet_cores: {}", device->id(), ethernet_cores_virtual);
    // auto [sender_worker_core_range, sender_worker_cores] =
    //     get_optimal_worker_core_placement(device, ethernet_cores_virtual, reserved_cores, num_links, sub_device_id);

    const auto [sender_worker_core_range, sender_worker_cores] = choose_worker_cores(
        num_links, num_workers_per_link, enable_persistent_fabric_mode, device, sub_device_id, reserved_cores);

    tt::log_debug(tt::LogOp, "input_tensor_num_pages: {}", input_tensor_num_pages);
    tt::log_debug(tt::LogOp, "input_tensor_cores: {}", input_tensor_cores);
    tt::log_debug(tt::LogOp, "input_tensor_shard_shape: {}", input_tensor_shard_shape);
    tt::log_debug(tt::LogOp, "input_tensor_shard_num_pages: {}", input_tensor_shard_num_pages);
    tt::log_debug(tt::LogOp, "output_tensor_cores: {}", output_tensor_cores);
    tt::log_debug(tt::LogOp, "output_tensor_shard_shape: {}", output_tensor_shard_shape);
    tt::log_debug(tt::LogOp, "output_tensor_shard_num_pages: {}", output_tensor_shard_num_pages);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = local_fabric_handle->get_edm_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_base_num_pages = std::lcm(input_tensor_shard_num_pages, output_tensor_shard_num_pages);
    uint32_t cb_num_pages = input_tensor_num_pages;
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    CBHandle cb_src0_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CBIndex::c_3;
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    // Reduction kernel stuff
    auto all_cores = output_tensor_cores.merge(sender_worker_core_range);
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);

    // Create output tensor splits
    std::vector<CoreRangeSet> output_corerangeset_per_link;
    std::vector<uint32_t> num_output_cores_in_link(num_links, 0);
    uint32_t output_cores_per_link = tt::div_up(output_tensor_cores.num_cores(), num_links);
    uint32_t num_assigned_cores = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_cores_this_link = std::min(output_cores_per_link, num_output_cores - num_assigned_cores);
        output_corerangeset_per_link.emplace_back(cores_to_corerangeset(std::vector<CoreCoord>(
            output_cores_vec.begin() + num_assigned_cores,
            output_cores_vec.begin() + num_assigned_cores + num_cores_this_link)));
        num_output_cores_in_link[link] = num_cores_this_link;
        num_assigned_cores += num_cores_this_link;
    }

    // Create output tensor page splits
    std::vector<uint32_t> output_tensor_pages_in_link(num_links, 0);
    uint32_t num_assigned_pages = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_output_pages_per_link = output_tensor_shard_num_pages * num_output_cores_in_link[link];
        uint32_t num_pages_this_link =
            std::min(num_output_pages_per_link, output_tensor_num_pages - num_assigned_pages);
        output_tensor_pages_in_link[link] = num_pages_this_link;
        num_assigned_pages += num_pages_this_link;
    }

    // Create input tensor splits
    std::vector<std::pair<uint32_t, uint32_t>> input_cores_idx_per_link(num_links, {0, 0});
    std::vector<uint32_t> input_tensor_tile_offset_per_link(num_links, 0);
    uint32_t start_core_idx = 0;
    uint32_t num_pages_overflow = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_pages_this_link = output_tensor_pages_in_link[link];

        // Get offset based on previous overflow
        uint32_t input_tensor_tile_offset =
            (input_tensor_shard_num_pages - num_pages_overflow) % input_tensor_shard_num_pages;
        input_tensor_tile_offset_per_link[link] = input_tensor_tile_offset;

        uint32_t end_core_idx = std::min(
            start_core_idx + tt::div_up(num_pages_this_link + input_tensor_tile_offset, input_tensor_shard_num_pages),
            num_input_cores);

        // Num pages allocated based on number of input cores selected for this link
        uint32_t num_pages_allocated =
            (end_core_idx - start_core_idx) * input_tensor_shard_num_pages - input_tensor_tile_offset;

        // Update overflow
        num_pages_overflow = num_pages_allocated - num_pages_this_link;

        // Store core indices
        input_cores_idx_per_link[link] = {start_core_idx, end_core_idx};

        // Set start index based on overflow
        if (num_pages_overflow > 0) {
            start_core_idx = end_core_idx - 1;
        } else {
            start_core_idx = end_core_idx;
        }
    }

    // Create reduction semaphore vector for each link
    std::vector<uint32_t> reduction_semaphore_ids(num_links, 0);
    for (uint32_t link = 0; link < num_links; link++) {
        reduction_semaphore_ids[link] = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    }

    /* reduction cb */
    uint32_t reduction_CB_single_tile_size = output_tensor.get_tensor_spec().tile().get_tile_size(df);
    uint32_t reduction_CB_tiles = output_tensor_shard_num_pages * ring_size;
    uint32_t reduction_CB_size = reduction_CB_tiles * reduction_CB_single_tile_size;

    uint32_t reduction_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig reduction_cb_config =
        tt::tt_metal::CircularBufferConfig(reduction_CB_size, {{reduction_cb_index, df}})
            .set_page_size(reduction_cb_index, reduction_CB_single_tile_size)
            .set_globally_allocated_address(*all_gather_output_tensor.buffer());
    auto cb_reduction = tt::tt_metal::CreateCircularBuffer(program, all_cores, reduction_cb_config);

    /* out cb */
    uint32_t out_CB_single_tile_size = output_tensor.get_tensor_spec().tile().get_tile_size(df);
    uint32_t out_CB_tiles = output_tensor_shard_num_pages;
    uint32_t out_CB_size = out_CB_tiles * out_CB_single_tile_size;

    uint32_t out_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig out_cb_config =
        tt::tt_metal::CircularBufferConfig(out_CB_size, {{out_cb_index, df}})
            .set_page_size(out_cb_index, out_CB_single_tile_size)
            .set_globally_allocated_address(*output_tensor.buffer());  // TODO: Remove once new cb attached for output
    auto cb_out = tt::tt_metal::CreateCircularBuffer(
        program, output_tensor_cores, out_cb_config);  // TODO: This should be the output cores instead

    // Create reduction dataflow kernel
    auto reduction_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    reduction_reader_kernel_config.compile_args = {
        reduction_cb_index,  // reduction_cb_index
        reduction_CB_tiles,  // total_num_reduction_tiles
    };
    auto reduction_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "reduction_receiver.cpp",
        output_tensor_cores,
        reduction_reader_kernel_config);

    // Create reduction dataflow kernel
    auto reduction_kernel_config = tt::tt_metal::ComputeConfig{};
    reduction_kernel_config.compile_args = {
        reduction_cb_index,  // reduction_cb_index
        out_cb_index,        // out_cb_index
    };
    auto reduction_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/"
        "reduction.cpp",
        output_tensor_cores,
        reduction_kernel_config);
    std::vector<uint32_t> reduction_kernel_rt_args = {
        ring_size,                      // num_blocks
        output_tensor_shard_num_pages,  // block_num_tiles
    };
    tt::tt_metal::SetRuntimeArgs(program, reduction_kernel_id, output_tensor_cores, reduction_kernel_rt_args);

    // KERNEL CREATION
    tt::tt_metal::NOC reader_noc = NOC::NOC_1;
    tt::tt_metal::NOC writer_noc = NOC::NOC_0;
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        ring_index,                 // my_chip_id
        src0_cb_index,              // cb0_id
        op_config.get_page_size(),  // tensor0_page_size
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "worker_reader.cpp",
        sender_worker_core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = reader_noc, .compile_args = reader_compile_args});

    // Writer
    auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    std::vector<uint32_t> writer_compile_args = {
        ring_index,                       // my_chip_id
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,      // num_packet_headers_storable
        src0_cb_index,                    // cb0_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
    };
    log_trace(tt::LogOp, "Writer Compile Args:");
    // writer_kernel_config.DataMovementConfig.processor = NOC::RISCV_0;
    // writer_kernel_config.DataMovementConfig.noc = NOC::RISCV_0_default;
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "worker_writer.cpp",
        sender_worker_core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = writer_noc, .compile_args = writer_compile_args});

    // Kernel Runtime Args
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        CoreCoord drain_sync_core = device->worker_core_from_logical_core(core);
        uint32_t worker_num_tiles_to_read = output_tensor_pages_in_link[link];

        uint32_t input_first_core_tile_start_offset = input_tensor_tile_offset_per_link[link];
        uint32_t output_first_core_tile_start_offset = 0;

        std::vector<uint32_t> input_tensor_cores_x;
        std::vector<uint32_t> input_tensor_cores_y;
        std::vector<uint32_t> output_tensor_cores_x;
        std::vector<uint32_t> output_tensor_cores_y;
        for (uint32_t i = input_cores_idx_per_link[link].first; i < input_cores_idx_per_link[link].second; i++) {
            auto this_core = device->worker_core_from_logical_core(input_cores_vec[i]);
            input_tensor_cores_x.push_back(this_core.x);
            input_tensor_cores_y.push_back(this_core.y);
        }
        for (uint32_t i = output_cores_per_link * link;
             i < output_cores_per_link * link + num_output_cores_in_link[link];
             i++) {
            auto this_core = device->worker_core_from_logical_core(output_cores_vec[i]);
            output_tensor_cores_x.push_back(this_core.x);
            output_tensor_cores_y.push_back(this_core.y);

            // if (device->id() == 4) {
            //     tt::log_info("this_core: {}", this_core);
            // }
        }

        std::optional<ttnn::ccl::SenderWorkerAdapterSpec> forward_fabric_connection =
            line_topology.is_first_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD)
                ? std::nullopt
                : std::optional<ttnn::ccl::SenderWorkerAdapterSpec>(local_fabric_handle->uniquely_connect_worker(
                      device, ttnn::ccl::EdmLineFabricOpInterface::FORWARD));
        std::optional<ttnn::ccl::SenderWorkerAdapterSpec> backward_fabric_connection =
            line_topology.is_last_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD)
                ? std::nullopt
                : std::optional<ttnn::ccl::SenderWorkerAdapterSpec>(local_fabric_handle->uniquely_connect_worker(
                      device, ttnn::ccl::EdmLineFabricOpInterface::BACKWARD));

        if (device->id() == 4) {
            tt::log_info("core: {}", core);
            if (forward_fabric_connection.has_value()) {
                tt::log_info(
                    "forward edm: {} {}", forward_fabric_connection->edm_noc_x, forward_fabric_connection->edm_noc_y);

                auto edmcore = CoreCoord(forward_fabric_connection->edm_noc_x, forward_fabric_connection->edm_noc_y);

                // auto it = std::find(edm_coords.begin(), edm_coords.end(), edmcore);
                // if (it == edm_coords.end()) {
                //     TT_THROW("not in edm_coords: {}", edmcore);
                // }
            }
            if (backward_fabric_connection.has_value()) {
                tt::log_info(
                    "backward edm: {} {}",
                    backward_fabric_connection->edm_noc_x,
                    backward_fabric_connection->edm_noc_y);

                auto edmcore = CoreCoord(backward_fabric_connection->edm_noc_x, backward_fabric_connection->edm_noc_y);

                // auto it = std::find(edm_coords.begin(), edm_coords.end(), edmcore);
                // if (it == edm_coords.end()) {
                //     TT_THROW("not in edm_coords: {}", edmcore);
                // }
            }
        }

        // Set reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),    // tensor_address0
            input_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,            // num_tiles_to_read
            input_first_core_tile_start_offset,  // first_core_tile_start_offset
            input_tensor_cores_x.size(),         // num_cores
        };
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_x.begin(), input_tensor_cores_x.end());
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_y.begin(), input_tensor_cores_y.end());
        log_trace(tt::LogOp, "Reader Runtime Args:");
        for (const auto& arg : reader_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        auto output_tensor_bbox = output_corerangeset_per_link[link].bounding_box();
        auto mcast_start_core = device->worker_core_from_logical_core(output_tensor_bbox.start_coord);
        auto mcast_end_core = device->worker_core_from_logical_core(output_tensor_bbox.end_coord);
        auto num_cores_mcast = (output_tensor_bbox.end_coord.x - output_tensor_bbox.start_coord.x + 1) *
                               (output_tensor_bbox.end_coord.y - output_tensor_bbox.start_coord.y + 1);
        // remove self from num_cores_mcast since we are not mcasting to self
        if (output_tensor_bbox.contains(core)) {
            num_cores_mcast -= 1;
        }
        if (writer_noc == NOC::NOC_1) {
            std::swap(mcast_start_core, mcast_end_core);
        }

        if (device->id() == 4) {
            tt::log_info("mcast_start_core: {} mcast_end_core {}", mcast_start_core, mcast_end_core);
        }

        bool wait_output_semaphore = true;   // (link == 0) && !enable_async_output_tensor;
        bool reset_global_semaphore = true;  // (link == 0) && !enable_async_output_tensor;
        uint32_t out_ready_sem_wait_value = ring_size;
        std::vector<uint32_t> writer_rt_args = {
            reduction_cb_index,                   // tensor_address0
            semaphore.address(),                  // out_ready_sem_bank_addr (absolute address)
            output_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,             // num_tiles_to_read
            output_first_core_tile_start_offset,  // first_core_tile_start_offset
            output_tensor_cores_x.size(),         // num_cores
            num_cores_mcast,                      // num_cores_mcast
            wait_output_semaphore,                // wait_output_semaphore
            reset_global_semaphore,               // reset_global_semaphore
            drain_sync_core.x,                    // out_ready_sem_noc0_x
            drain_sync_core.y,                    // out_ready_sem_noc0_y
            out_ready_sem_wait_value,             // out_ready_sem_wait_value
            reduction_semaphore_ids[link],        // reduction_semaphore_id
            mcast_start_core.x,                   // mcast_dest_noc_start_x
            mcast_start_core.y,                   // mcast_dest_noc_start_y
            mcast_end_core.x,                     // mcast_dest_noc_end_x
            mcast_end_core.y,                     // mcast_dest_noc_end_y
            link,                                 // link
        };
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_x.begin(), output_tensor_cores_x.end());
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_y.begin(), output_tensor_cores_y.end());
        log_trace(tt::LogOp, "Writer Runtime Args:");
        for (const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        writer_rt_args.push_back(forward_fabric_connection.has_value());
        if (forward_fabric_connection.has_value()) {
            auto sender_worker_flow_control_semaphore_id = CreateSemaphore(program, {core}, 0);
            auto sender_worker_teardown_semaphore_id = CreateSemaphore(program, {core}, 0);
            auto sender_worker_buffer_index_semaphore_id = CreateSemaphore(program, {core}, 0);
            append_worker_to_fabric_edm_sender_rt_args(
                forward_fabric_connection.value(),
                sender_worker_flow_control_semaphore_id,
                sender_worker_teardown_semaphore_id,
                sender_worker_buffer_index_semaphore_id,
                writer_rt_args);
        }
        writer_rt_args.push_back(backward_fabric_connection.has_value());
        if (backward_fabric_connection.has_value()) {
            auto sender_worker_flow_control_semaphore_id = CreateSemaphore(program, {core}, 0);
            auto sender_worker_teardown_semaphore_id = CreateSemaphore(program, {core}, 0);
            auto sender_worker_buffer_index_semaphore_id = CreateSemaphore(program, {core}, 0);
            append_worker_to_fabric_edm_sender_rt_args(
                backward_fabric_connection.value(),
                sender_worker_flow_control_semaphore_id,
                sender_worker_teardown_semaphore_id,
                sender_worker_buffer_index_semaphore_id,
                writer_rt_args);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);

        std::vector<uint32_t> reduction_reader_rt_args = {
            reduction_semaphore_ids[link],  // reduction_semaphore_id
        };
        tt::tt_metal::SetRuntimeArgs(
            program, reduction_reader_kernel_id, output_corerangeset_per_link[link], reduction_reader_rt_args);

        input_first_core_tile_start_offset =
            (worker_num_tiles_to_read % input_tensor_shard_num_pages) + input_first_core_tile_start_offset;
    }

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id, worker_sender_writer_kernel_id, sender_worker_cores, cb_out, cb_reduction](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];
            const auto& all_gather_output = input_tensors[1];

            auto semaphore = static_cast<const ttnn::AllReduceAsync*>(operation)->semaphore;

            // update senders
            auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
            auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
            for (const auto& core : sender_worker_cores) {
                // reader
                auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                // writer
                auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
                worker_writer_sender_runtime_args[1] = semaphore.address();
            }
            UpdateDynamicCircularBufferAddress(program, cb_out, *output.buffer());
            UpdateDynamicCircularBufferAddress(program, cb_reduction, *all_gather_output.buffer());
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn

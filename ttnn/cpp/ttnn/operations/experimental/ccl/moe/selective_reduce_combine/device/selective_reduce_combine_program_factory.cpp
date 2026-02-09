// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_device_operation.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl::moe {
namespace detail {

std::vector<uint32_t> data_parallel_split(
    uint32_t token_size_bytes, const uint32_t max_packet_size_bytes, const uint32_t num_data_parallel_cores) {
    std::vector<uint32_t> data_parallel_sizes_bytes;

    const uint32_t need_data_parallel_cores =
        std::max(num_data_parallel_cores, token_size_bytes / max_packet_size_bytes);
    data_parallel_sizes_bytes.reserve(need_data_parallel_cores);

    const uint32_t max_segment_size_bytes = token_size_bytes / need_data_parallel_cores;

    for (uint32_t c = 0; c < num_data_parallel_cores; ++c) {
        const uint32_t token_increment = std::min(token_size_bytes, max_segment_size_bytes);
        data_parallel_sizes_bytes.push_back(token_increment);
        token_size_bytes -= token_increment;

        if (token_size_bytes == 0) {
            break;
        }
    }

    return data_parallel_sizes_bytes;
}

auto launch_mux_workers(
    const MeshDevice& mesh_device,
    const CoreRangeSet& mux_core_range_set,
    const tt::tt_fabric::FabricNodeId src_node_id,
    const std::vector<ttnn::MeshCoordinate>& neighbors,
    const uint32_t num_links,
    const uint32_t num_workers,
    Program& program) {
    const auto num_header_only_channels = tt::div_up(num_workers, num_links);
    const auto num_full_size_channels = tt::div_up(num_workers, num_links);
    constexpr auto num_buffers_full_size_channels = 20;    // parameterize?
    constexpr auto num_buffers_header_only_channels = 20;  // parameterize?

    const size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t l1_unreserved_base_address =
        mesh_device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        num_buffers_header_only_channels,
        buffer_size_bytes_full_size_channel,
        l1_unreserved_base_address);

    const auto needed_mux_core_range_set =
        select_from_corerangeset(mux_core_range_set, 0, num_links * (neighbors.size() - 1));
    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        needed_mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    std::vector<std::map<ttnn::MeshCoordinate, CoreCoord>> mux_neigbor_core_maps;
    mux_neigbor_core_maps.reserve(num_links);

    const auto mux_cores = corerange_to_cores(needed_mux_core_range_set);
    auto mux_core_iter = mux_cores.begin();
    for (uint32_t link = 0; link < num_links; ++link) {
        std::map<ttnn::MeshCoordinate, CoreCoord> mux_neigbor_core_map;
        for (const auto& neighbor_coord : neighbors) {
            auto mux_logical_core = *(mux_core_iter++);
            const auto mux_virtual_core = mesh_device.worker_core_from_logical_core(mux_logical_core);

            std::vector<uint32_t> mux_rt_args = {};
            const auto dst_node_id = mesh_device.get_fabric_node_id(neighbor_coord);
            mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                src_node_id, dst_node_id, link, program, {mux_logical_core});

            tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
            mux_neigbor_core_map[neighbor_coord] = mux_virtual_core;
        }
        mux_neigbor_core_maps.push_back(mux_neigbor_core_map);
    }

    return std::make_tuple(mux_kernel_id, mux_kernel_config, mux_neigbor_core_maps);
}

void add_termination_master_rt_args(
    const std::vector<std::map<ttnn::MeshCoordinate, CoreCoord>>& mux_neigbor_core_maps,
    std::vector<uint32_t>& writer_runtime_args) {
    for (const auto& m : mux_neigbor_core_maps) {
        for (const auto& c : m) {
            const auto& mux_virtual_core = c.second;
            writer_runtime_args.push_back(mux_virtual_core.x);
            writer_runtime_args.push_back(mux_virtual_core.y);
        }
    }
}

}  // namespace detail
SelectiveReduceCombineDeviceOperation::UnifiedSelectReduce::cached_mesh_workload_t
SelectiveReduceCombineDeviceOperation::UnifiedSelectReduce::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.dense_input_tensor.device();
    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);

    auto final_barrier_semaphore = operation_attributes.optional_cross_device_semaphore.value_or(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0));

    tt::tt_metal::distributed::Synchronize(
        mesh_device, std::nullopt, {});  // interaction with subdevice needs to be investigated

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_coords.coords(),
            tensor_args,
            tensor_return_value,
            init_barrier_semaphore,
            final_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<SelectiveReduceCombineDeviceOperation::UnifiedSelectReduce::shared_variables_t>
SelectiveReduceCombineDeviceOperation::UnifiedSelectReduce::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const std::vector<ttnn::MeshCoordinate>& all_mesh_coordinates,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore) {
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    Program program{};

    const auto& input_tensor = tensor_args.dense_input_tensor;
    const auto& dense_token_maps_tensor = tensor_args.dense_token_maps_tensor;
    const auto& dense_token_counts_tensor = tensor_args.dense_token_counts_tensor;

    const auto& output_tensor = tensor_return_value;
    const auto batch_size = operation_attributes.batch_size;
    const auto seq_size = operation_attributes.seq_size;
    const auto select_experts_k = operation_attributes.select_experts_k;
    const auto hidden_size = operation_attributes.hidden_size;

    const auto total_tokens = batch_size * seq_size;
    // TODO map number of experts to device
    const auto experts = operation_attributes.experts;

    const auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    // TODO assert (axis.has_value()) in validate
    const auto& axis = operation_attributes.axis;

    const auto fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    const uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    // const auto& metadata_shape = metadata_tensor.tensor_spec().logical_shape();

    const uint32_t num_devices = mesh_view.num_devices();

    // TODO this should eventually be variable per device
    const uint32_t experts_per_device = experts / num_devices;

    const auto input_dtype = input_tensor.dtype();
    const auto& dense_token_maps_tensor_spec = dense_token_maps_tensor.tensor_spec();

    const auto fabric_max_packet_size_bytes = get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t max_packet_size_bytes =
        input_dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes) : fabric_max_packet_size_bytes;

    const uint32_t token_size_bytes = hidden_size * input_tensor.element_size();
    const uint32_t dense_token_maps_page_size_bytes = dense_token_maps_tensor_spec.compute_page_size_bytes();

    const auto l1_alignment = hal::get_l1_alignment();
    const auto aligned_dense_token_maps_page_size_bytes = tt::align(dense_token_maps_page_size_bytes, l1_alignment);

    // in validate, assert that worker_core_range_set.size() == num_token_parallel_cores*num_data_parallel_cores;
    const auto num_token_parallel_cores = operation_attributes.num_token_parallel_cores;
    auto num_data_parallel_cores = operation_attributes.num_data_parallel_cores;
    const auto& worker_core_range_set = operation_attributes.worker_core_range_set;

    // in validate mux_core_range_set.size() == 2(directions) * num_links
    const auto& mux_core_range_set = operation_attributes.mux_core_range_set;

    const auto data_parallel_sizes_bytes =
        detail::data_parallel_split(token_size_bytes, max_packet_size_bytes, num_data_parallel_cores);

    num_data_parallel_cores = data_parallel_sizes_bytes.size();
    const auto num_worker_cores = num_token_parallel_cores * num_data_parallel_cores;

    const auto needed_worker_core_range_set = select_from_corerangeset(worker_core_range_set, 0, num_worker_cores - 1);
    const std::vector<CoreCoord> sender_cores = corerange_to_cores(needed_worker_core_range_set, num_worker_cores);

    // buffer may be padded
    const auto token_segment_buffer_size_bytes = input_tensor.logical_shape()[-1] * input_tensor.element_size();
    const auto expert_token_segment_buffer_block_size_bytes =
        token_segment_buffer_size_bytes * total_tokens / num_token_parallel_cores;
    const auto buffer_size_bytes = expert_token_segment_buffer_block_size_bytes * experts_per_device;

    const auto input_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    // input sharded buffer
    constexpr auto data_cb_id = tt::CBIndex::c_0;
    CircularBufferConfig cb_data_config = CircularBufferConfig(buffer_size_bytes, {{data_cb_id, input_data_format}})
                                              .set_page_size(data_cb_id, buffer_size_bytes)
                                              .set_globally_allocated_address(*input_tensor.buffer());

    // dense_token_maps_tensor page buffer
    constexpr auto dense_token_maps_cb_id = tt::CBIndex::c_1;
    // stash offset and count value for each local expert in uint32_t
    const uint32_t dense_token_maps_tensor_extra_size_bytes = 2 * 4 * experts_per_device;
    const uint32_t aligned_dense_token_maps_buffer_size =
        tt::align(aligned_dense_token_maps_page_size_bytes + dense_token_maps_tensor_extra_size_bytes, l1_alignment);
    const auto dense_token_maps_data_format = datatype_to_dataformat_converter(dense_token_maps_tensor.dtype());
    CircularBufferConfig cb_dense_token_maps_config =
        CircularBufferConfig(
            aligned_dense_token_maps_buffer_size, {{dense_token_maps_cb_id, dense_token_maps_data_format}})
            .set_page_size(dense_token_maps_cb_id, aligned_dense_token_maps_buffer_size);

    // active token counts page buffer
    const auto dram_alignment = hal::get_dram_alignment();
    constexpr auto token_counts_cb_id = tt::CBIndex::c_2;
    const auto token_counts_element_size = dense_token_counts_tensor.element_size();
    const auto token_counts_data_format = datatype_to_dataformat_converter(dense_token_counts_tensor.dtype());
    const uint32_t aligned_token_counts_buffer_size =
        tt::align(token_counts_element_size * experts_per_device, dram_alignment);
    CircularBufferConfig cb_token_counts_config =
        CircularBufferConfig(aligned_token_counts_buffer_size, {{token_counts_cb_id, token_counts_data_format}})
            .set_page_size(token_counts_cb_id, aligned_token_counts_buffer_size);

    // client interface
    constexpr auto num_headers = 3;  // data unicast headers and atomic inc "multicast" headers
    constexpr auto client_interface_cb_id = tt::CBIndex::c_3;
    CircularBufferConfig client_interface_cb_config =
        CircularBufferConfig(num_headers * CLIENT_INTERFACE_SIZE, {{client_interface_cb_id, tt::DataFormat::UInt32}})
            .set_page_size(client_interface_cb_id, CLIENT_INTERFACE_SIZE);

    // create circular buffers
    CreateCircularBuffer(program, needed_worker_core_range_set, cb_data_config);
    CreateCircularBuffer(program, needed_worker_core_range_set, cb_dense_token_maps_config);
    CreateCircularBuffer(program, needed_worker_core_range_set, cb_token_counts_config);
    CreateCircularBuffer(program, needed_worker_core_range_set, client_interface_cb_config);

    // fabric routing info
    std::vector<uint32_t> dest_mesh_id, dest_chip_id, route;
    for (const auto& coord : all_mesh_coordinates) {
        const auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }
    const auto [neighbors, directions] =
        operations::ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, axis);

    // launch mux
    const auto [mux_kernel_id, mux_kernel_config, mux_neigbor_core_maps] = detail::launch_mux_workers(
        *mesh_device, mux_core_range_set, fabric_node_id, neighbors, num_links, num_worker_cores, program);

    // launch reader kernel
    std::unordered_map<std::string, uint32_t> reader_named_ct_args = {
        {"dense_token_maps_cb_id", dense_token_maps_cb_id},
        {"token_counts_cb_id", token_counts_cb_id},
        {"dense_token_maps_page_size_bytes", aligned_dense_token_maps_page_size_bytes},
        {"token_counts_page_size_bytes", aligned_token_counts_buffer_size},
        {"num_local_experts", experts_per_device},
        {"num_token_parallel_cores", num_token_parallel_cores},
        {"global_num_tokens", total_tokens},
        {"select_experts_k", select_experts_k}};

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(dense_token_maps_tensor.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(dense_token_counts_tensor.buffer()).append_to(reader_compile_time_args);

    const DataMovementConfig reader_config{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_1,
        .compile_args = reader_compile_time_args,
        .named_compile_args = reader_named_ct_args};

    KernelHandle ternary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/reader.cpp",
        needed_worker_core_range_set,
        reader_config);

    // launch writer kernel
    const uint32_t flat_mesh_idx = operations::ccl::common::get_linearized_index(mesh_coordinate, mesh_view);

    const auto start_coord =
        mesh_device->worker_core_from_logical_core(needed_worker_core_range_set.bounding_box().start_coord);
    const auto end_coord =
        mesh_device->worker_core_from_logical_core(needed_worker_core_range_set.bounding_box().end_coord);

    const bool use_init_semaphore = !tensor_args.optional_output_tensor.has_value() ||
                                    !operation_attributes.optional_cross_device_semaphore.has_value();
    std::unordered_map<std::string, uint32_t> writer_named_ct_args = {
        {"dense_token_maps_cb_id", dense_token_maps_cb_id},
        {"data_cb_id", data_cb_id},
        {"packet_header_cb_id", client_interface_cb_id},
        {"num_token_parallel_cores", num_token_parallel_cores},
        {"num_data_parallel_cores", num_data_parallel_cores},
        {"use_init_semaphore", use_init_semaphore},
        {"noc_x_start", start_coord.x},
        {"noc_y_start", start_coord.y},
        {"noc_x_end", end_coord.x},
        {"noc_y_end", end_coord.y},
        {"experts", experts},
        {"global_num_tokens", total_tokens},
        {"source_token_segment_buffer_size_bytes", token_segment_buffer_size_bytes},
        {"source_expert_block_size_bytes", expert_token_segment_buffer_block_size_bytes},
        {"token_size_bytes", token_size_bytes},
        {"alignment", l1_alignment},
        {"num_devices", num_devices},
        {"src_chip_id", src_chip_id},
        {"mesh_rows", mesh_view.num_rows()},
        {"mesh_cols", mesh_view.num_cols()},
        {"fabric_max_packet_size_bytes", max_packet_size_bytes},
        {"linearized_mesh_coord", flat_mesh_idx},
        {"topology", static_cast<uint32_t>(topology)},
        {"num_mux_workers", num_links * neighbors.size()}};

    std::vector<uint32_t> writer_compile_time_args;
    ttnn::ccl::fabric_mux_connection_ct_args(
        num_data_parallel_cores * num_token_parallel_cores,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        writer_compile_time_args);
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    using operations::ccl::common::stringify;
    std::map<std::string, std::string> writer_defines = {
        {"DEST_CHIP_ID", stringify(dest_chip_id)},
        {"DEST_MESH_ID", stringify(dest_mesh_id)},
        {"DIRECTIONS", stringify(directions)}};

    if (axis.has_value()) {
        writer_defines["REPLICATE_GROUP_AXIS"] = std::to_string(axis.value());
    }

    const DataMovementConfig writer_config{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
        .compile_args = writer_compile_time_args,
        .defines = writer_defines,
        .named_compile_args = writer_named_ct_args};

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/writer.cpp",
        needed_worker_core_range_set,
        writer_config);

    const auto& termination_master_core = sender_cores[0];
    const auto termination_master_virtual_core = mesh_device->worker_core_from_logical_core(termination_master_core);
    const auto termination_master_semaphore_id = CreateSemaphore(program, {termination_master_core}, 0);

    const uint32_t num_workers_per_link = num_worker_cores / num_links;
    uint32_t link_worker_idx = 0, token_parallel_idx = 0, dest_token_segment_offset_bytes = 0;
    auto core_map_iter = mux_neigbor_core_maps.cbegin();
    auto data_parallel_size_iter = data_parallel_sizes_bytes.cbegin();
    for (const auto& sender_core : sender_cores) {
        std::vector<uint32_t> reader_runtime_args = {
            dense_token_maps_tensor.buffer()->address(),    // dense_token_maps_addr
            dense_token_counts_tensor.buffer()->address(),  // dense_token_counts_addr
            token_parallel_idx,                             // token_parallel_core_id
        };

        const auto source_token_segment_size_bytes = *(data_parallel_size_iter++);
        std::vector<uint32_t> writer_runtime_args = {
            output_tensor.buffer()->address(),  // output_base_addr
            source_token_segment_size_bytes,    // source_token_segment_size_bytes
            dest_token_segment_offset_bytes,    // dest_token_segment_size_bytes
            init_semaphore.address(),           // init_semaphore_addr
            cross_device_semaphore.address()    // global_semaphore_addr
        };

        const bool is_termination_master = (sender_core == termination_master_core);
        for (const auto& neighbor_coordinate : neighbors) {
            const auto& mux_virtual_core = core_map_iter->at(neighbor_coordinate);

            ttnn::ccl::fabric_mux_connection_rt_args(
                true,
                is_termination_master,
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                mux_virtual_core,
                link_worker_idx,
                sender_core,
                mux_kernel_config,
                program,
                termination_master_virtual_core,
                writer_runtime_args,
                termination_master_semaphore_id);
        }

        // termination master is responsible for tearing down all mux workers, needs their coordinates
        if (is_termination_master) {
            detail::add_termination_master_rt_args(mux_neigbor_core_maps, writer_runtime_args);
        }

        SetRuntimeArgs(program, ternary_reader_kernel_id, sender_core, reader_runtime_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, sender_core, writer_runtime_args);

        if (data_parallel_size_iter == data_parallel_sizes_bytes.cend()) {
            data_parallel_size_iter = data_parallel_sizes_bytes.cbegin();
            dest_token_segment_offset_bytes = 0;
            ++token_parallel_idx;
        } else {
            dest_token_segment_offset_bytes += source_token_segment_size_bytes;
        }

        if (++link_worker_idx == num_workers_per_link) {
            link_worker_idx = 0;
            ++core_map_iter;
        }
    }

    return {
        std::move(program),
        {.reader_kernel_id = ternary_reader_kernel_id,
         .writer_kernel_id = unary_writer_kernel_id,
         .cores = sender_cores,
         .init_semaphore = init_semaphore,
         .cross_device_semaphore = cross_device_semaphore}};
}

void SelectiveReduceCombineDeviceOperation::UnifiedSelectReduce::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& coord = range.start_coord();
        TT_FATAL(
            coord == range.end_coord(),
            "Expected single coordinate per program but got range of {} to {}",
            coord,
            range.end_coord());

        const auto& shared_variables = cached_workload.shared_variables.at(range);
        const auto& reader_kernel_id = shared_variables.reader_kernel_id;
        const auto& writer_kernel_id = shared_variables.writer_kernel_id;
        const auto& cores = shared_variables.cores;

        for (const auto& core : cores) {
            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);

            reader_runtime_args.at(0) = tensor_args.dense_token_maps_tensor.buffer()->address();
            reader_runtime_args.at(1) = tensor_args.dense_token_counts_tensor.buffer()->address();

            writer_runtime_args.at(0) = tensor_return_value.buffer()->address();
            writer_runtime_args.at(3) = (uint32_t)shared_variables.init_semaphore.address();

            writer_runtime_args.at(4) = (operation_attributes.optional_cross_device_semaphore.has_value())
                                            ? operation_attributes.optional_cross_device_semaphore->address()
                                            : shared_variables.cross_device_semaphore.address();
        }
    }
}

}  // namespace ttnn::operations::experimental::ccl::moe

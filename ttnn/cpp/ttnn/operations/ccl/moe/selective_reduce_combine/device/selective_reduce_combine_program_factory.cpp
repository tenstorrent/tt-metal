// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "cpp/ttnn/operations/ccl/all_to_all_combine/device/all_to_all_combine_device_operation.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::ccl::moe {
namespace detail{
    
std::vector<uint32_t> data_parallel_split(
    uint32_t token_size_bytes, const uint32_t max_packet_size_bytes, const uint32_t num_data_parallel_cores){
    
    std::vector<uint32_t> data_parallel_sizes_bytes;
    data_parallel_sizes_bytes.reserve(num_data_parallel_cores);
    
    const auto token_increment tt::div_up(token_size_bytes/num_data_parallel_cores, max_packet_size_bytes)
    for(uint32_t c=0, c < num_data_parallel_cores; ++c){
        data_parallel_sizes_bytes.push_back(std::min(token_increment, token_size_bytes));
        token_size_bytes-=token_increment;
    }
    
    return data_parallel_sizes_bytes;
}


auto launch_mux_workers(
    const CoreRangeSet & mux_core_range_set, 
    const tt_fabric::FabricNodeId src_node_id,
    const std::vector<ttnn::MeshCoordinate> & neighbors
    const uint32_t num_links, 
    const uint32_t num_workers){
    
    auto num_full_size_channels = num_workers; // ?  //num_workers_per_direction
    constexpr auto num_header_only_channels = 0;
    const size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        0,
        buffer_size_bytes_full_size_channel,
        l1_unreserved_base_address);
    
    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
    
    std::vector<std::map<ttnn::MeshCoordinate,CoreCoord>> mux_neigbor_core_maps;
    mux_core_map.reserve(num_links);
    
    auto mux_core_iter = mux_core_range_set.ranges().cbegin();
    for (uint32_t link = 0; link < num_links; ++link) {
        std::map<ttnn::MeshCoordinate,CoreCoord> mux_neigbor_core_map;
        for (const auto & neighbor_coord : neighbors){
            auto mux_logical_core = *((mux_core_iter++)->begin());
            const auto mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);

            std::vector<uint32_t> mux_rt_args = {};
            const auto dst_node_id = mesh_device->get_fabric_node_id(neighbor_coord);
            mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                src_node_id, dst_node_id, link, program, {mux_logical_core});
            
            tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
            mux_neigbor_core_map[neighbor_coord] = mux_virtual_core; 
        }
        mux_neigbor_core_maps.push_back(mux_neigbor_core_map);
    }
    
    return std::make_tuple(mux_kernel_id, mux_kernel_config, mux_neigbor_core_maps);

}

    
} // namespace detail
SelectiveReduceCombineDeviceOperation::UnifiedSelectReduce::cached_mesh_workload_t
SelectiveReduceCombineDeviceOperation::UnifiedSelectReduce::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    auto final_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
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
    const GlobalSemaphore& cross_device_semaphore,
    const GlobalSemaphore& IntraDeviceSyncSemaphore) {
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    Program program{};

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& dense_metadata_tensor = tensor_args.dense_metadata_tensor;
    const auto& mapping_tensor = tensor_args.mapping_tensor;
    const auto& output_tensor = tensor_return_value;
    const auto batch_size = operation_attributes.batch_size;
    const auto seq_size = operation_attributes.seq_size;
    const auto selected_experts_k = operation_attributes.select_experts_k;
    
    const auto total_tokens = batch_size*seq_size;
    // TODO map number of experts to device    
    const auto experts =operation_attributes.experts;
    // TODO this should be variable per device
    const uint32_t experts_per_device = experts / num_devices;
    
    const auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;


    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    const auto fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    const uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    //const auto& metadata_shape = metadata_tensor.tensor_spec().logical_shape();

    const uint32_t num_devices = mesh_view.num_devices();
    

    const auto input_dtype = input_tensor.dtype();
    const auto& input_spec = input_tensor.tensor_spec();
    const auto& metadata_spec = metadata_tensor.tensor_spec();

    const bool input_is_dram = input_tensor.buffer()->buffer_type() == BufferType::DRAM;
    
    const auto fabric_max_packet_size_bytes = get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t max_packet_size_bytes =
        input_dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes) : fabric_max_packet_size_bytes;
    
    const auto input_page_size_bytes = input_spec.compute_page_size_bytes();
    
    const uint32_t metadata_page_size_bytes = metadata_spec.compute_page_size_bytes();

    const auto l1_alignment = hal::get_l1_alignment();
    const auto dram_alignment = hal::get_dram_alignment();

    const auto aligned_metadata_page_size_bytes = tt::align(metadata_page_size_bytes, l1_alignment);

    const auto input_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto mapping_data_format = datatype_to_dataformat_converter(mapping_tensor.dtype());
    const auto metadata_data_format = datatype_to_dataformat_converter(metadata_tensor.dtype());
    const auto subdevice_cores = corerange_to_cores(operation_attributes.worker_core_range_set);
    
    // in validate, assert that subdevice_cores.size() == num_token_parallel_cores*num_data_parallel_cores;
    const auto num_token_parallel_cores = operation_attributes.num_token_parallel_cores;
    const auto num_data_parallel_cores = operation_attributes.num_data_parallel_cores;
    const auto & worker_core_range_set= operation_attributes.worker_core_range_set;
    
    // in validate mux_core_range_set.size() == 2(directions) * num_links
    const auto & mux_core_range_set= operation_attributes.mux_core_range_set;
    
    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_token_parallel_cores*num_data_parallel_cores, worker_core_range_set , true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);
    
    const auto data_parallel_sizes_bytes = detail::data_parallel_split(
        token_size_bytes, max_packet_size_bytes, num_data_parallel_cores);
    const auto max_token_segment_size_bytes =
        *std::max_element(data_parallel_sizes_bytes.begin(), data_parallel_sizes_bytes.end())
    const auto aligned_max_token_segment_size_bytes = 
        tt::align(max_token_segment_size_bytes, input_is_dram? dram_alignment:l1_alignment);

    const uint32_t buffering_factor = 2;
    // input sharded buffer
    constexpr auto data_cb_id = tt::CBIndex::c_0;
    CircularBufferConfig cb_data_config =
        CircularBufferConfig(buffering_factor * aligned_max_token_segment_size_bytes, {{data_cb_id, input_data_format}})
            .set_page_size(data_cb_id, aligned_max_token_segment_size_bytes);

    // metadata page buffer
    constexpr auto metadata_cb_id = tt::CBIndex::c_3;
    CircularBufferConfig cb_metadata_config =
        CircularBufferConfig(aligned_metadata_page_size_bytes, {{metadata_cb_id, metadata_data_format}})
            .set_page_size(metadata_cb_id, aligned_metadata_page_size_bytes);

    // client interface
    constexpr auto num_headers = 2;  // data unicast headers and atomic inc "multicast" headers
    constexpr auto client_interface_cb_id = tt::CBIndex::c_4;
    CircularBufferConfig client_interface_cb_config =
        CircularBufferConfig(num_headers * CLIENT_INTERFACE_SIZE, {{client_interface_cb_id, tt::DataFormat::UInt32}})
            .set_page_size(client_interface_cb_id, CLIENT_INTERFACE_SIZE);

    // create circular buffers
    CreateCircularBuffer(program, sender_core_grid, cb_data_config);
    CreateCircularBuffer(program, sender_core_grid, cb_metadata_config);
    CreateCircularBuffer(program, sender_core_grid, client_interface_cb_config);

    const uint32_t flat_mesh_idx = common::get_linearized_index(mesh_coordinate, mesh_view);
    
    std::unordered_map<std::string, uint32_t> reader_named_ct_args = {
        {"dense_data_cb_id", dense_data_cb_id},
        {"dense_metadata_cb", dense_metadata_cb},
        
        {"metadata_entry_si", metadata_entry_si},
        
        {"token_block_height", token_block_height},
        {"token_parallel_core_id", token_parallel_core_id},
        {"data_parallel_core_id", data_parallel_core_id},
        
        {"token_size_bytes", token_size_bytes},
        
        {"num_token_parallel_cores", num_token_parallel_cores},
        {"num_data_parallel_cores", num_data_parallel_cores},
        
        {"source_token_segment_bytes", source_token_segment_bytes},
        {"select_experts_k", select_experts_k},
        {"total_tokens", total_tokens},
    };
    
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(dense_metadata_tensor.buffer()).append_to(reader_compile_time_args);

    const DataMovementConfig reader_config{
        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1, .compile_args = reader_compile_time_args, .named_compile_args reader_named_ct_args};

    KernelHandle ternary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/moe/selective_reduce_combine/device/kernels/dataflow/reader.cpp",
        sender_core_grid,
        reader_config);

    const auto& axis = operation_attributes.axis;

    std::vector<uint32_t> writer_named_ct_args = {

        {"metadata_cb_id", metadata_cb_id},
        {"data_cb_id", data_cb_id},
        
        {"packet_header_cb_id", packet_header_cb_id},
        
        {"metadata_entry_size", metadata_entry_size},
        
        {"token_parallel_core_id", token_parallel_core_id},
        {"data_parallel_core_id", data_parallel_core_id},
        
        {"sync_core_noc_x", sync_core_noc_x},
        {"sync_core_noc_y", sync_core_noc_y},
        {"is_sync_core", is_sync_core},
        
        {"num_token_parallel_cores", num_token_parallel_cores},
        {"num_data_parallel_cores", num_data_parallel_cores},
        {"token_size_bytes", token_size_bytes},
        
        {"token_segment_size_bytes", token_segment_size_bytes},
        {"token_segment_offset_bytes", token_segment_offset_bytes},
        
        {"alignment", alignment},
        
        {"num_devices", num_devices},
        {"src_chip_id", src_chip_id},
        {"mesh_rows", mesh_rows},
        {"mesh_cols", mesh_cols},
        {"fabric_max_packet_size", fabric_max_packet_size},
        {"linearized_mesh_coords", linearized_mesh_coords},
        {"topology", topology},
    };
    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    ccl::fabric_mux_connection_ct_args(
        num_workers_per_direction,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        sender_writer_compile_args);
    
    // fabric routing info
    std::vector<uint32_t> dest_mesh_id, dest_chip_id, route;
    for (const auto& coord : all_mesh_coordinates) {
        const auto fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)fabric_node_id.chip_id);
    }
    const auto [neighbors, directions] = common::get_neighbors(mesh_view, mesh_coordinate, topology, axis);

    std::map<std::string, std::string> writer_defines = {
        {"DEST_CHIP_ID", common::stringify(dest_chip_id)},
        {"DEST_MESH_ID", common::stringify(dest_mesh_id)},
        {"DIRECTIONS", common::stringify(directions)}};

    if (axis.has_value()) {
        writer_defines["REPLICATE_GROUP_AXIS"] = std::to_string(axis.value());
    }

    const DataMovementConfig writer_config{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
        .compile_args = writer_compile_time_args,
        .defines = writer_defines};

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/moe/selective_reduce_combine/device/kernels/dataflow/reader.cpp",
        sender_core_grid,
        writer_config);

    
    const uint32_t num_workers_per_link = num_worker_cores / num_links;
    const auto [mux_kernel_id, mux_kernel_config, mux_neigbor_core_maps] = launch_mux_workers()
    
    workers_per_link=0;
    auto core_map_iter = mux_neigbor_core_maps.cbegin()
    for (const auto& sender_core : sender_cores) {
        std::vector<uint32_t> reader_runtime_args = {
            mapping_tensor.buffer()->address(),
            metadata_tensor.buffer()->address(),
            input_tensor.buffer()->address(),
            0,
        };
        
        
        std::vector<uint32_t> writer_runtime_args = {};
        ...

        for (const auto& neighbor_coordinate : neighbors) {
            const auto mux_virtual_core = (*core_map_iter)[neighbor_coordinate];
            ccl::fabric_mux_connection_rt_args(tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,)
        }
        SetRuntimeArgs(program, ternary_reader_kernel_id, sender_core, reader_runtime_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, sender_core, writer_runtime_args);
        
        if (link_workers++ == num_workers_per_link){
            link_workers=0;
            ++core_map_iter;
        }
    }

    return {
        std::move(program),
        {.ternary_reader_kernel_id = ternary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .cores = sender_cores,
         .init_semaphore = init_semaphore,
         .cross_device_semaphore = cross_device_semaphore}};
}

void SelectiveReduceCombineDeviceOperation::UnifiedSelectReduce::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto & coord = range.start_coord();
        TT_FATAL(coord == range.end_coord(), "Expected single coordinate per program but got range of {} to {}", coord, range.end_coord());

        const auto& shared_variables = cached_workload.shared_variables.at(range);
        const auto& ternary_reader_kernel_id = shared_variables.ternary_reader_kernel_id;
        const auto& unary_writer_kernel_id = shared_variables.unary_writer_kernel_id;
        const auto& cores = shared_variables.cores;

        for (const auto& core : cores) {
            auto& reader_runtime_args = GetRuntimeArgs(program, ternary_reader_kernel_id, core);
            auto& writer_runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);

            reader_runtime_args.at(0) = tensor_args.mapping_tensor.buffer()->address();
            reader_runtime_args.at(1) = tensor_args.metadata_tensor.buffer()->address();
            reader_runtime_args.at(2) = tensor_args.input_tensor.buffer()->address();

            writer_runtime_args.at(0) = tensor_return_value.buffer()->address();
            writer_runtime_args.at(1) = (uint32_t)shared_variables.cross_device_semaphore.address();
            writer_runtime_args.at(2) = (uint32_t)shared_variables.init_semaphore.address();
        }
    }
}

}  // namespace ttnn::operations::ccl

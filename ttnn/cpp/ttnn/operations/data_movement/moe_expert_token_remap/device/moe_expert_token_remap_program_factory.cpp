// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>

#include "moe_expert_token_remap_device_operation.hpp"

namespace ttnn::operations::data_movement {

MoeExpertTokenRemapDeviceOperation::Multicore::cached_mesh_workload_t
MoeExpertTokenRemapDeviceOperation::Multicore::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<MoeExpertTokenRemapDeviceOperation::Multicore::shared_variables_t>
MoeExpertTokenRemapDeviceOperation::Multicore::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& metadata_tensor = tensor_args.metadata_tensor;
    const auto& mapping_tensor = tensor_args.mapping_tensor;
    const auto& topk_tensor = tensor_args.topk_tensor;

    const auto& metadata_tensor_shape = metadata_tensor.logical_shape();
    const auto batch_size = metadata_tensor_shape[1];
    const auto seq_size = metadata_tensor_shape[2];
    const auto selected_experts_k = metadata_tensor_shape[3];
    const auto experts = mapping_tensor.logical_shape()[-2];

    const auto& output_mapping_tensor = tensor_return_value.at(0);
    const auto& output_reduced_tensor = tensor_return_value.at(1);

    const auto experts_per_device = output_mapping_tensor.logical_shape()[-1];

    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    const auto mapping_page_size_bytes = mapping_tensor.tensor_spec().compute_page_size_bytes();
    const auto aligned_mapping_page_size_bytes = tt::align(mapping_page_size_bytes, l1_alignment);

    const auto metadata_page_size_bytes = metadata_tensor.tensor_spec().compute_page_size_bytes();
    const auto aligned_metadata_page_size_bytes = tt::align(metadata_page_size_bytes, l1_alignment);

    const auto topk_page_size_bytes = topk_tensor.tensor_spec().compute_page_size_bytes();
    const auto aligned_topk_page_size_bytes = tt::align(topk_page_size_bytes, l1_alignment);

    const auto output_mapping_page_size_bytes = output_mapping_tensor.tensor_spec().compute_page_size_bytes();
    const auto output_reduced_page_size_bytes = output_reduced_tensor.tensor_spec().compute_page_size_bytes();

    Program program{};

    // todo maybe, subdevice
    auto* mesh_device = topk_tensor.device();
    const auto grid = mesh_device->compute_with_storage_grid_size();
    // CoreCoord grid = {1,1};

    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    using tt::tt_metal::CircularBufferConfig;

    // full mapping buffer
    const auto mapping_data_format = datatype_to_dataformat_converter(mapping_tensor.dtype());
    const auto mapping_tensor_cb_id = tt::CBIndex::c_0;
    CircularBufferConfig cb_mapping_tensor_config =
        CircularBufferConfig(aligned_mapping_page_size_bytes, {{mapping_tensor_cb_id, mapping_data_format}})
            .set_page_size(mapping_tensor_cb_id, aligned_mapping_page_size_bytes);
    CreateCircularBuffer(program, total_cores, cb_mapping_tensor_config);

    // scratch space to store and share indices of per device experts
    const auto local_experts_cb_id = tt::CBIndex::c_1;
    using local_experts_t = uint16_t;
    const auto aligned_local_expert_page_size_bytes =
        tt::align(experts_per_device * sizeof(local_experts_t), l1_alignment);
    const auto local_experts_dataformat =
        datatype_to_dataformat_converter(tt::tt_metal::convert_to_data_type<local_experts_t>());
    CircularBufferConfig cb_local_experts_config =
        CircularBufferConfig(aligned_local_expert_page_size_bytes, {{local_experts_cb_id, local_experts_dataformat}})
            .set_page_size(local_experts_cb_id, aligned_local_expert_page_size_bytes);
    CreateCircularBuffer(program, total_cores, cb_local_experts_config);

    // metadata page buffer
    constexpr uint32_t metadata_buffer_factor = 1;
    const auto metadata_data_format = datatype_to_dataformat_converter(metadata_tensor.dtype());
    const auto metadata_cb_id = tt::CBIndex::c_2;
    CircularBufferConfig cb_metadata_config =
        CircularBufferConfig(
            metadata_buffer_factor * aligned_metadata_page_size_bytes, {{metadata_cb_id, metadata_data_format}})
            .set_page_size(metadata_cb_id, aligned_metadata_page_size_bytes);
    CreateCircularBuffer(program, total_cores, cb_metadata_config);

    // topk page buffer
    constexpr uint32_t topk_buffer_factor = 2;
    const auto topk_data_format = datatype_to_dataformat_converter(topk_tensor.dtype());
    const auto topk_cb_id = tt::CBIndex::c_3;
    CircularBufferConfig cb_topk_config =
        CircularBufferConfig(topk_buffer_factor * aligned_topk_page_size_bytes, {{topk_cb_id, topk_data_format}})
            .set_page_size(topk_cb_id, aligned_topk_page_size_bytes);
    CreateCircularBuffer(program, total_cores, cb_topk_config);

    // output mapping staging buffer
    const auto output_mapping_data_format = datatype_to_dataformat_converter(output_mapping_tensor.dtype());
    const auto output_mapping_cb_id = tt::CBIndex::c_4;
    CircularBufferConfig cb_output_mapping_config =
        CircularBufferConfig(output_mapping_page_size_bytes, {{output_mapping_cb_id, output_mapping_data_format}})
            .set_page_size(output_mapping_cb_id, output_mapping_page_size_bytes);
    CreateCircularBuffer(program, total_cores, cb_output_mapping_config);

    // output reduced staging buffer
    const auto output_reduced_data_format = datatype_to_dataformat_converter(output_reduced_tensor.dtype());
    const auto output_reduced_cb_id = tt::CBIndex::c_5;
    CircularBufferConfig cb_output_reduced_config =
        CircularBufferConfig(output_reduced_page_size_bytes, {{output_reduced_cb_id, output_reduced_data_format}})
            .set_page_size(output_reduced_cb_id, output_reduced_page_size_bytes);
    CreateCircularBuffer(program, total_cores, cb_output_reduced_config);

    const auto& mesh_view = mesh_device->get_view();
    const uint32_t flat_mesh_idx = (mesh_coordinate[0] * mesh_view.num_cols()) + mesh_coordinate[1];

    // slightly abusing this functionality since here we also have a single page if any experts are activated
    constexpr bool local_reduce = true;
    std::vector<uint32_t> reader_ct_args = {
        mapping_tensor_cb_id,
        local_experts_cb_id,
        metadata_cb_id,
        topk_cb_id,
        experts_per_device,
        batch_size,
        seq_size,
        experts,
        flat_mesh_idx,
        topk_page_size_bytes,
        selected_experts_k,
        mapping_page_size_bytes,
        metadata_page_size_bytes,
        local_reduce};
    tt::tt_metal::TensorAccessorArgs(topk_tensor.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(mapping_tensor.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(metadata_tensor.buffer()).append_to(reader_ct_args);

    tt::tt_metal::KernelHandle ternary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    const auto output_datum_size_bytes = tt::datum_size(output_mapping_data_format);
    const auto reduction_size = operation_attributes.reduction_size;
    std::vector<uint32_t> writer_ct_args = {
        local_experts_cb_id,
        metadata_cb_id,
        topk_cb_id,
        output_mapping_cb_id,
        output_reduced_cb_id,
        selected_experts_k,
        experts_per_device,
        output_mapping_page_size_bytes,
        output_datum_size_bytes,
        output_reduced_page_size_bytes,
        reduction_size,
    };
    tt::tt_metal::TensorAccessorArgs(*output_mapping_tensor.buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(*output_reduced_tensor.buffer()).append_to(writer_ct_args);

    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/moe_expert_token_remap/device/kernels/dataflow/"
        "writer_moe_expert_token_remap.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    // split work over metadata pages (batch*seq)
    const auto num_metadata_pages = metadata_tensor.buffer()->num_pages();

    const auto [core_page_increments, all_cores] =
        tt::tt_metal::split_work_to_cores_even_multiples(grid, num_metadata_pages, reduction_size);

    const auto mapping_tensor_addr = mapping_tensor.buffer()->address();
    const auto metadata_tensor_addr = metadata_tensor.buffer()->address();
    const auto topk_tensor_addr = topk_tensor.buffer()->address();
    const auto output_mapping_tensor_addr = output_mapping_tensor.buffer()->address();
    const auto output_reduced_tensor_addr = output_reduced_tensor.buffer()->address();

    uint32_t page_idx_start = 0, page_idx_end = 0;
    constexpr auto num_reader_rt_args = 5, num_writer_rt_args = 5;
    std::vector<CoreCoord> utilized_cores = corerange_to_cores(all_cores, std::nullopt);
    TT_FATAL(utilized_cores.size() == core_page_increments.size(), "Internal error");

    auto cit = utilized_cores.begin();
    for (auto increment : core_page_increments) {
        page_idx_end += increment;
        const std::array<uint32_t, num_reader_rt_args> reader_runtime_args = {
            mapping_tensor_addr, metadata_tensor_addr, topk_tensor_addr, page_idx_start, page_idx_end};
        tt::tt_metal::SetRuntimeArgs(program, ternary_reader_kernel_id, *cit, reader_runtime_args);

        const uint32_t reduction_idx_start = page_idx_start / reduction_size;

        const std::array<uint32_t, num_writer_rt_args> writer_runtime_args = {
            output_mapping_tensor_addr, page_idx_start, page_idx_end, output_reduced_tensor_addr, reduction_idx_start};

        tt::tt_metal::SetRuntimeArgs(program, binary_writer_kernel_id, *cit, writer_runtime_args);

        page_idx_start += increment;
        ++cit;
    }

    return {
        std::move(program),
        {.ternary_reader_kernel_id = ternary_reader_kernel_id,
         .binary_writer_kernel_id = binary_writer_kernel_id,
         .utilized_cores = utilized_cores}};
}

void MoeExpertTokenRemapDeviceOperation::Multicore::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& output_mapping_tensor = tensor_return_value.at(0);
    const auto& output_reduced_tensor = tensor_return_value.at(1);

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& coord = range.start_coord();
        TT_FATAL(
            coord == range.end_coord(),
            "Expected single coordinate per program but got range of {} to {}",
            coord,
            range.end_coord());

        const auto& shared_variables = cached_workload.shared_variables.at(range);
        const auto& ternary_reader_kernel_id = shared_variables.ternary_reader_kernel_id;
        const auto& binary_writer_kernel_id = shared_variables.binary_writer_kernel_id;
        const auto& utilized_cores = shared_variables.utilized_cores;

        for (const auto& c : utilized_cores) {
            auto& reader_runtime_args = GetRuntimeArgs(program, ternary_reader_kernel_id, c);
            auto& writer_runtime_args = GetRuntimeArgs(program, binary_writer_kernel_id, c);

            reader_runtime_args.at(0) = tensor_args.mapping_tensor.buffer()->address();
            reader_runtime_args.at(1) = tensor_args.metadata_tensor.buffer()->address();
            reader_runtime_args.at(2) = tensor_args.topk_tensor.buffer()->address();

            writer_runtime_args.at(0) = output_mapping_tensor.buffer()->address();
            writer_runtime_args.at(3) = output_reduced_tensor.buffer()->address();
        }
    }
};
}  // namespace ttnn::operations::data_movement

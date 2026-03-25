// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>

#include "offset_cumsum_program_factory.hpp"
#include "offset_cumsum_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn::experimental::prim {

namespace {

struct CreatedProgram {
    tt::tt_metal::Program program;
    OffsetCumsumSharedVariables shared_variables;
};

CreatedProgram create_program(const Tensor& input, std::array<Tensor, 2>& tensor_return_value, uint32_t row_idx) {
    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::UINT32);

    CoreCoord core = {0, 0};
    CoreRangeSet core_set = CoreRangeSet(std::vector{CoreRange(core, core)});

    const auto& logical_shape = input.logical_shape();
    uint32_t W = logical_shape[-1];
    uint32_t H = logical_shape[-2];

    auto* src_buffer = input.buffer();
    auto* dst_offsets_buffer = tensor_return_value.at(0).buffer();
    auto* dst_totals_buffer = tensor_return_value.at(1).buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_offsets_is_dram = dst_offsets_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_totals_is_dram = dst_totals_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    uint32_t input_page_size = src_buffer->aligned_page_size();
    uint32_t offsets_page_size = dst_offsets_buffer->aligned_page_size();
    uint32_t totals_page_size = dst_totals_buffer->aligned_page_size();

    uint32_t cb_in0_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_in0_config =
        tt::tt_metal::CircularBufferConfig(input_page_size, {{cb_in0_index, cb_data_format}})
            .set_page_size(cb_in0_index, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_set, cb_in0_config);

    uint32_t cb_out0_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_out0_config =
        tt::tt_metal::CircularBufferConfig(offsets_page_size, {{cb_out0_index, cb_data_format}})
            .set_page_size(cb_out0_index, offsets_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_set, cb_out0_config);

    std::vector<uint32_t> compile_time_args = {
        cb_in0_index,
        cb_out0_index,
        (uint32_t)src_is_dram,
        (uint32_t)dst_offsets_is_dram,
        (uint32_t)dst_totals_is_dram,
        input_page_size,
        offsets_page_size,
        totals_page_size,
        W,
        H,
    };
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_offsets_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_totals_buffer).append_to(compile_time_args);

    std::map<std::string, std::string> kernel_defines;
    tt::tt_metal::KernelHandle kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/offset_cumsum/device/kernels/"
        "reader_offset_cumsum_interleaved.cpp",
        core_set,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args, kernel_defines));

    tt::tt_metal::SetRuntimeArgs(
        program,
        kernel_id,
        core,
        {src_buffer->address(), dst_offsets_buffer->address(), dst_totals_buffer->address(), row_idx});

    return CreatedProgram{
        std::move(program),
        {/* kernel_id = */ kernel_id,
         /* core      = */ core}};
}

}  // namespace

OffsetCumsumProgramFactory::cached_mesh_workload_t OffsetCumsumProgramFactory::create_mesh_workload(
    const OffsetCumsumParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const Tensor& input,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        uint32_t row_idx = coord[operation_attributes.cluster_axis];

        auto result = create_program(input, tensor_return_value, row_idx);
        auto coord_range = ttnn::MeshCoordinateRange(coord);
        mesh_workload.add_program(coord_range, std::move(result.program));
        shared_variables.emplace(coord_range, std::move(result.shared_variables));
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void OffsetCumsumProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const OffsetCumsumParams& operation_attributes,
    const Tensor& input,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto coord = *(coord_range.begin());
        uint32_t row_idx = coord[operation_attributes.cluster_axis];

        auto& shared_vars = cached_workload.shared_variables.at(coord_range);
        auto& runtime_args = GetRuntimeArgs(program, shared_vars.kernel_id, shared_vars.core);
        runtime_args[0] = input.buffer()->address();
        runtime_args[1] = tensor_return_value.at(0).buffer()->address();
        runtime_args[2] = tensor_return_value.at(1).buffer()->address();
        runtime_args[3] = row_idx;
    }
}

}  // namespace ttnn::experimental::prim

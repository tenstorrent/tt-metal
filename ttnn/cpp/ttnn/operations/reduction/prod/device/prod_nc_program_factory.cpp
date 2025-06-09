// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"

namespace tt {
using namespace constants;
namespace operations {

namespace primary {

tt::tt_metal::operation::ProgramWithCallbacks prod_nc_format(
    const tt::tt_metal::Tensor& input, const tt::tt_metal::Tensor& output, int64_t dim) {
    TT_FATAL(dim == 0 || dim == 1, "Dimension ({}) must be either 0 or 1", dim);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = input.device();
    auto program = tt::tt_metal::Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    const auto single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    const auto input_shape = input.get_padded_shape();
    const auto input_shape_without_padding = input.get_logical_shape();

    const auto N = input_shape[0];
    const auto C = input_shape[1];
    const auto Ht = input_shape[2] / TILE_HEIGHT;
    const auto Wt = input_shape[3] / TILE_WIDTH;
    const auto HtWt = Ht * Wt;
    const auto CHtWt = C * Ht * Wt;
    const auto num_reduce_input_tile = input_shape[dim];
    const auto input_tile_offset = (dim == 0) ? (CHtWt) : (HtWt);
    const auto num_output_tiles = output.volume() / TILE_HW;

    log_debug(LogTest, "N {} C {} Ht {} Wt {}", N, C, Ht, Wt);
    log_debug(
        LogTest,
        "dim {} num_reduce_input_tile {} input_tile_offset {}, num_output_tiles {}",
        dim,
        num_reduce_input_tile,
        input_tile_offset,
        num_output_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const uint32_t in0_t = 2;        // input
    const uint32_t in1_t = 1;        // zero
    const uint32_t intermed0_t = 1;  // accumulated sum
    const uint32_t out0_t = 2;       // output
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_output_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    ttnn::operations::CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CBIndex::c_0, in0_t},        // input
            {CBIndex::c_1, in1_t},        // zero
            {CBIndex::c_2, intermed0_t},  // accumulated sum
            {CBIndex::c_3, out0_t},       // output
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer* input_buffer_type = input.buffer();
    bool input_is_dram = input_buffer_type->buffer_type() == tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)input_is_dram, static_cast<uint32_t>(dim)};

    tt_metal::Buffer* output_buffer_type = output.buffer();
    constexpr uint32_t cb_id_out = CBIndex::c_3;
    bool output_is_dram = output_buffer_type->buffer_type() == tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)cb_id_out, (std::uint32_t)output_is_dram};

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/dataflow/reader_prod_nc.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    const auto reader_kernel_id =
        ttnn::operations::CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id =
        ttnn::operations::CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1};
    std::map<string, string> compute_defines;

    const auto compute_kernel_file = "ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/compute/prod_nc.cpp";
    const auto compute_kernel_1_id = ttnn::operations::CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_cols_per_core_group_1, compute_args_group_1}, compute_defines);

    std::optional<tt::tt_metal::KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2};
        compute_kernel_2_id = ttnn::operations::CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input.buffer()->address(),
             num_reduce_input_tile,
             num_tiles_per_core,
             input_tile_offset,
             tile_offset,
             static_cast<uint32_t>(ttnn::operations::is_dram(input)),
             HtWt,
             CHtWt,
             static_cast<uint32_t>(dim)});

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {output.buffer()->address(),
             num_tiles_per_core,
             tile_offset,
             static_cast<uint32_t>(ttnn::operations::is_dram(output))});

        if (core_group_1.contains(core)) {
            SetRuntimeArgs(program, compute_kernel_1_id, core, {num_reduce_input_tile, num_tiles_per_core});
        } else if (core_group_2.contains(core)) {
            TT_FATAL(compute_kernel_2_id.has_value(), "compute_kernel_2_id needs to have a value");
            SetRuntimeArgs(program, compute_kernel_2_id.value(), core, {num_reduce_input_tile, num_tiles_per_core});
        } else {
            TT_THROW("Core not in specified core ranges.");
        }
        tile_offset += num_tiles_per_core;
    }

    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y](
                                                   const void* operation,
                                                   const tt::tt_metal::Program& program,
                                                   const std::vector<tt::tt_metal::Tensor>& input_tensors,
                                                   const std::vector<std::optional<const tt::tt_metal::Tensor>>&,
                                                   const std::vector<tt::tt_metal::Tensor>& output_tensors) {
        const auto* input_buffer = input_tensors.at(0).buffer();
        const auto* output_buffer = input_tensors.at(1).buffer();
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input_buffer->address();
            }

            {
                auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = output_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt

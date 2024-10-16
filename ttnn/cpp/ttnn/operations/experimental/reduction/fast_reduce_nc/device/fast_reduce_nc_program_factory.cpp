// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace ttnn::operations::experimental::reduction::detail {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_and_scale_spatial_dims(const ttnn::SimpleShape &shape,
                                                                                  uint32_t dim) {
    const auto rank = shape.rank();

    TT_FATAL(rank >= 2, "Shape must have at least two dims.");
    uint32_t Wt = shape[-1] / TILE_WIDTH;
    uint32_t Ht = shape[-2] / TILE_HEIGHT;

    uint32_t reduce_dim = shape[dim];
    uint32_t inner_dims_product = 1;
    for (auto i = dim + 1; i < rank - 2; ++i) {
        inner_dims_product *= shape[i];
    }

    uint32_t inner_tile_size = inner_dims_product * Ht * Wt;
    uint32_t reduce_tile_size = reduce_dim * inner_tile_size;

    return {Wt, Ht, inner_tile_size, reduce_tile_size};
}

}  // namespace

operation::ProgramWithCallbacks reduce_nc_factory(const ttnn::Tensor &input,
                                                  const ttnn::Tensor &output,
                                                  int64_t dim,
                                                  const ttnn::DeviceComputeKernelConfig &compute_kernel_config) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto *device = input.device();
    auto program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    const auto single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    const auto cb_1_data_format = datatype_to_dataformat_converter(DataType::BFLOAT16);
    const auto cb_1_tile_size = tt_metal::detail::TileSize(cb_1_data_format);

    const auto input_shape = input.get_padded_shape();
    const auto input_shape_without_padding = input.get_logical_shape();
    const auto [Wt, Ht, inner_tile_size, reduce_tile_size] =
        extract_and_scale_spatial_dims(input_shape, static_cast<uint32_t>(dim));
    const auto num_reduce_input_tile = input_shape[dim];
    const auto num_output_tiles = output.volume() / TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);
    // choose granularity as the largest factor of num_reduce_input_tile that is less than or equal to 8
    uint32_t input_granularity;
    for (input_granularity = 8; input_granularity > 1; --input_granularity) {
        if (num_reduce_input_tile % input_granularity == 0) {
            break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const uint32_t in0_t = input_granularity * 2;  // input
    const uint32_t in1_t = 1;                      // zero
    const uint32_t intermed0_t = 1;                // accumulated sum
    const uint32_t out0_t = 2;                     // output
    const auto [num_cores_to_be_used,
                all_cores,
                core_group_1,
                core_group_2,
                num_cols_per_core_group_1,
                num_cols_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_output_tiles);
    const auto intermed_cb_data_format = (fp32_dest_acc_en) ? tt::DataFormat::Float32 : cb_data_format;
    const auto intermed_cb_single_tile_size = (fp32_dest_acc_en) ? single_tile_size * 2 : single_tile_size;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::CircularBufferConfig cb_scr0_config =
        tt_metal::CircularBufferConfig(in0_t * single_tile_size, {{CB::c_in0, cb_data_format}})
            .set_page_size(CB::c_in0, single_tile_size);
    auto cb_scr0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_scr0_config);

    tt_metal::CircularBufferConfig cb_scr1_config =
        tt_metal::CircularBufferConfig(in1_t * cb_1_tile_size, {{CB::c_in1, cb_1_data_format}})
            .set_page_size(CB::c_in1, cb_1_tile_size);
    auto cb_scr1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_scr1_config);

    tt_metal::CircularBufferConfig cb_intermed0_config =
        tt_metal::CircularBufferConfig(intermed0_t * intermed_cb_single_tile_size,
                                       {{CB::c_intermed0, intermed_cb_data_format}})
            .set_page_size(CB::c_intermed0, intermed_cb_single_tile_size);
    auto cb_intermed0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed0_config);

    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(out0_t * single_tile_size, {{CB::c_out0, cb_data_format}})
            .set_page_size(CB::c_out0, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(input.memory_config().buffer_type == BufferType::DRAM), input_granularity};
    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<uint32_t>(output.memory_config().buffer_type == BufferType::DRAM), input_granularity};
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reader_reduce_nc.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/writer_reduce_nc.cpp";

    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program, reader_kernel_file, all_cores, tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program, writer_kernel_file, all_cores, tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1 = {
        num_cols_per_core_group_1, num_reduce_input_tile, input_granularity};
    std::map<string, string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp";
    const auto compute_kernel_1_id =
        tt_metal::CreateKernel(program,
                               compute_kernel_file,
                               core_group_1,
                               tt_metal::ComputeConfig{.math_fidelity = math_fidelity,
                                                       .fp32_dest_acc_en = fp32_dest_acc_en,
                                                       .math_approx_mode = math_approx_mode,
                                                       .compile_args = compute_args_group_1,
                                                       .defines = compute_defines});

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2 = {
            num_cols_per_core_group_2, num_reduce_input_tile, input_granularity};
        compute_kernel_2_id = tt_metal::CreateKernel(program,
                                                     compute_kernel_file,
                                                     core_group_2,
                                                     tt_metal::ComputeConfig{.math_fidelity = math_fidelity,
                                                                             .fp32_dest_acc_en = fp32_dest_acc_en,
                                                                             .math_approx_mode = math_approx_mode,
                                                                             .compile_args = compute_args_group_2,
                                                                             .defines = compute_defines});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        tt_metal::SetRuntimeArgs(program,
                                 reader_kernel_id,
                                 core,
                                 {input.buffer()->address(),
                                  num_reduce_input_tile,
                                  num_tiles_per_core,
                                  tile_offset,
                                  static_cast<uint32_t>(dim),
                                  reduce_tile_size,
                                  inner_tile_size});

        tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, {output.buffer()->address(), num_tiles_per_core, tile_offset});

        tile_offset += num_tiles_per_core;
    }

    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        const auto *input_buffer = input_tensors.at(0).buffer();
        const auto *output_buffer = output_tensors.at(0).buffer();
        auto &reader_kernel_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
        auto &writer_kernel_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            auto &reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
            reader_kernel_args[0] = input_buffer->address();
            auto &writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
            writer_kernel_args[0] = output_buffer->address();
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::reduction::detail

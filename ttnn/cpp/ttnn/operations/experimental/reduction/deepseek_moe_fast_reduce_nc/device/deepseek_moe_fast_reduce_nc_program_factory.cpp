// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/deepseek_moe_fast_reduce_nc_program_factory.hpp"

namespace ttnn::operations::experimental::reduction::deepseek_moe_fast_reduce_nc::detail {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

DeepseekMoEFastReduceNCProgramFactory::cached_program_t DeepseekMoEFastReduceNCProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // hardcoded constants
    // const uint32_t num_split_tensors = 8;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = tensor_args.input_tensor.device();
    auto program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_rank = input_shape.rank();

    const std::vector<ttnn::Tensor>& output_tensors = tensor_return_value;

    const uint32_t reduction_dim = operation_attributes.reduction_dim;
    // const uint32_t split_dim = operation_attributes.split_dim;

    // const uint32_t input_tensor_Ht = input_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t input_tensor_Wt = input_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t num_tile_elements = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;

    uint32_t inner_dims_product = 1;
    for (uint32_t dim = reduction_dim + 1; dim < input_rank; ++dim) {
        inner_dims_product *= input_shape[dim];
    }

    const uint32_t num_tiles_to_reduce_together = input_shape[reduction_dim];
    const uint32_t inner_num_tiles = inner_dims_product / num_tile_elements;
    const uint32_t reduction_num_tiles = inner_num_tiles * num_tiles_to_reduce_together;

    // Choose granularity as the largest factor of num_reduce_input_tile that is less than or equal to 8.
    // Helps with locality and increases work unit for better performance.
    uint32_t input_granularity;
    for (input_granularity = 8; input_granularity > 1; --input_granularity) {
        if (num_tiles_to_reduce_together % input_granularity == 0) {
            break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t num_output_tiles = input_tensor.physical_volume() / num_tile_elements / num_tiles_to_reduce_together;

    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_x = grid.x;
    // const auto num_cores_y = grid.y;
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_output_tiles, /*row_wise=*/true);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto output_data_format = datatype_to_dataformat_converter(output_tensors.at(0).dtype());
    const auto compute_data_format = datatype_to_dataformat_converter(DataType::BFLOAT16);

    const uint32_t input_page_size = input_tensor.buffer()->page_size();
    const uint32_t output_page_size = output_tensors.at(0).buffer()->page_size();
    const uint32_t compute_page_size = tt::tile_size(compute_data_format);

    const uint32_t input_tensor_buffer_factor = input_granularity * 2;
    const uint32_t compute_buffer_factor = 1;
    const uint32_t output_tensor_buffer_factor = 2;

    uint32_t compute_input_cb_id_0 = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig compute_input_cb_config_0 =
        tt::tt_metal::CircularBufferConfig(
            input_tensor_buffer_factor * input_page_size, {{compute_input_cb_id_0, input_data_format}})
            .set_page_size(compute_input_cb_id_0, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, compute_input_cb_config_0);

    uint32_t compute_input_cb_id_1 = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig compute_input_cb_config_1 =
        tt::tt_metal::CircularBufferConfig(
            compute_buffer_factor * compute_page_size, {{compute_input_cb_id_1, compute_data_format}})
            .set_page_size(compute_input_cb_id_1, compute_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, compute_input_cb_config_1);

    uint32_t compute_output_cb_id = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig compute_output_cb_config =
        tt::tt_metal::CircularBufferConfig(
            output_tensor_buffer_factor * output_page_size, {{compute_output_cb_id, output_data_format}})
            .set_page_size(compute_output_cb_id, output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, compute_output_cb_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_ct_args = {
        input_page_size,
        input_granularity,
        num_cores_to_be_used,
        reduction_dim,
        compute_input_cb_id_0,
        compute_input_cb_id_1};
    TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        output_page_size, num_cores_to_be_used, compute_output_cb_id, input_tensor_Wt};
    TensorAccessorArgs(output_tensors.at(0).buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(output_tensors.at(1).buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(output_tensors.at(2).buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(output_tensors.at(3).buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(output_tensors.at(4).buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(output_tensors.at(5).buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(output_tensors.at(6).buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(output_tensors.at(7).buffer()).append_to(writer_ct_args);

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/"
        "deepseek_moe_fast_reduce_nc_reader.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/"
        "deepseek_moe_fast_reduce_nc_writer.cpp";

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_file, all_cores, tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel_file, all_cores, tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input_tensor.device()->arch(), operation_attributes.compute_kernel_config);

    std::map<std::string, std::string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/"
        "deepseek_moe_fast_reduce_nc_reduce.cpp";

    std::vector<uint32_t> compute_ct_args_group_1 = {
        num_cols_per_core_group_1,
        num_tiles_to_reduce_together,
        input_granularity,
        compute_input_cb_id_0,
        compute_input_cb_id_1,
        compute_output_cb_id,
    };
    tt::tt_metal::CreateKernel(
        program,
        compute_kernel_file,
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_ct_args_group_1,
            .defines = compute_defines});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_ct_args_group_2 = {
            num_cols_per_core_group_2,
            num_tiles_to_reduce_together,
            input_granularity,
            compute_input_cb_id_0,
            compute_input_cb_id_1,
            compute_output_cb_id,
        };
        tt::tt_metal::CreateKernel(
            program,
            compute_kernel_file,
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_ct_args_group_2,
                .defines = compute_defines});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Each core is assigned an output work unit in a row wise round robin
    // fashion. For a given core, the first index is i, and all subsequent
    // indicies are increments of num_cores_to_be_used. The total number of
    // units is num_tiles_per_group times num_cores_to_be_used.
    // For example, with 130 output tiles to be processed and no shards (shard
    // factor is 1) on an 8x8 grid
    // - the increment is 64
    // - the first 2 cores will have num_tiles_per_core 3 and the rest 2
    // - core x=0,y=0 will process output tiles 0, 64, and 128
    // - core x=1,y=0 will process output tiles 1, 65, and 129
    // - core x=2,y=0 will process output tiles 2 and 66
    // - core x=3,y=0 will process output tiles 3 and 67
    // - etc
    // The first tile that needs to be reduced has the same as the output tile.
    // That is the starting point for the reader, which then processes all
    // subsequent tiles to be reduced. The increment for the input indicies is
    // the size of the inner dimensions in tiles (inner_num_tiles). The number
    // of tiles to process is the size of the reduce dimension in tiles
    // (reduction_num_tiles).
    // The shard factor is used to iterate over shards instead of tiles.
    // It is taken into account in the num_cols_per_core_group variables and
    // the tile_offset is incremented by it for the reader to adjust it's
    // reading pattern.
    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }
        uint32_t id_range_length = num_tiles_per_core * num_cores_to_be_used;

        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),
            num_tiles_to_reduce_together,
            id_range_length,
            i,
            reduction_num_tiles,
            inner_num_tiles};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

        std::vector<uint32_t> writer_rt_args = {
            output_tensors.at(0).buffer()->address(),
            output_tensors.at(1).buffer()->address(),
            output_tensors.at(2).buffer()->address(),
            output_tensors.at(3).buffer()->address(),
            output_tensors.at(4).buffer()->address(),
            output_tensors.at(5).buffer()->address(),
            output_tensors.at(6).buffer()->address(),
            output_tensors.at(7).buffer()->address(),
            id_range_length,
            i};
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);
    }

    return cached_program_t{
        std::move(program), {reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_x}};
}

void DeepseekMoEFastReduceNCProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto* input_buffer = tensor_args.input_tensor.buffer();
    const auto* output_buffer = tensor_return_value.at(0).buffer();
    auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    const auto& num_cores_x = cached_program.shared_variables.num_cores_x;

    auto& reader_kernel_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
    auto& writer_kernel_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};
        auto& reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
        reader_kernel_args[0] = input_buffer->address();
        auto& writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
        writer_kernel_args[0] = output_buffer->address();
    }
}

}  // namespace ttnn::operations::experimental::reduction::deepseek_moe_fast_reduce_nc::detail

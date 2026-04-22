// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/deepseek_moe_fast_reduce_nc_fused_program_factory.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

// Matches compile-time arg order in deepseek_moe_fast_reduce_nc_fused_reader.cpp (get_compile_time_arg_val 0..13).
struct DeepseekMoeFastReduceNcFusedReaderCtArgs {
    uint32_t cb_in_act_id{};
    uint32_t cb_scores_id{};
    uint32_t cb_scores_rm_id{};
    uint32_t act_page_size{};
    uint32_t scores_buf_page_size{};
    uint32_t scores_tile_size{};
    uint32_t num_cores{};
    uint32_t input_granularity{};
    uint32_t reduction_dim{};
    uint32_t reduction_dim_size{};
    uint32_t inner_num_tiles{};
    uint32_t reduction_num_tiles{};
    uint32_t num_tokens{};
    uint32_t scores_cb_rm_page_size{};
};

std::vector<uint32_t> to_reader_ct_arg_vector(const DeepseekMoeFastReduceNcFusedReaderCtArgs& ct) {
    return {
        ct.cb_in_act_id,
        ct.cb_scores_id,
        ct.cb_scores_rm_id,
        ct.act_page_size,
        ct.scores_buf_page_size,  // DRAM page size
        ct.scores_tile_size,
        ct.num_cores,
        ct.input_granularity,
        ct.reduction_dim,
        ct.reduction_dim_size,   // experts_k
        ct.inner_num_tiles,      // the number of tiles in the inner dimensions: [reduction_dim+1,...,rank-1]
        ct.reduction_num_tiles,  // the number of tiles in the reduction dimension: inner_num_tiles * reduction_dim_size
        ct.num_tokens,
        ct.scores_cb_rm_page_size,  // cb_scores_rm_id page size: one page = one token row
    };
}

}  // namespace

namespace ttnn::experimental::prim {

DeepseekMoEFastReduceNCFusedProgramFactory::cached_program_t DeepseekMoEFastReduceNCFusedProgramFactory::create(
    const DeepseekMoEFastReduceNCFusedParams& operation_attributes,
    const DeepseekMoEFastReduceNCFusedInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = tensor_args.input_tensor.device();
    auto program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const ttnn::Tensor& scores_tensor = tensor_args.scores_tensor;
    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_rank = input_shape.rank();

    const std::vector<ttnn::Tensor>& output_tensors = tensor_return_value;

    const uint32_t reduction_dim = operation_attributes.reduce_dim;

    const uint32_t num_tile_elements = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;
    const uint32_t input_tensor_Wt = input_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t slice_Wt = input_tensor_Wt / output_tensors.size();

    uint32_t inner_dims_product = 1;
    for (uint32_t dim = reduction_dim + 1; dim < input_rank; ++dim) {
        inner_dims_product *= input_shape[dim];
    }

    const uint32_t reduction_dim_size = input_shape[reduction_dim];  // experts_k
    const uint32_t inner_num_tiles = inner_dims_product / num_tile_elements;
    const uint32_t reduction_num_tiles = inner_num_tiles * reduction_dim_size;

    // scores shape: [tokens, 1, seq, experts_k] (ROW_MAJOR)
    const uint32_t num_tokens = scores_tensor.logical_shape()[0];  // tokens_per_device

    // Choose granularity as the largest factor of num_reduce_input_tile that is less than or equal to 8.
    // Helps with locality and increases work unit for better performance.
    uint32_t input_granularity;
    for (input_granularity = 8; input_granularity > 1; --input_granularity) {
        if (reduction_dim_size % input_granularity == 0) {
            break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t num_output_tiles = input_tensor.physical_volume() / num_tile_elements / reduction_dim_size;

    auto grid = device->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_output_tiles, /*row_wise=*/true);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto scores_data_format = datatype_to_dataformat_converter(scores_tensor.dtype());
    const auto output_data_format = datatype_to_dataformat_converter(output_tensors.at(0).dtype());

    const uint32_t input_page_size = input_tensor.buffer()->page_size();
    const uint32_t output_page_size = output_tensors.at(0).buffer()->page_size();
    const uint32_t scores_page_size = scores_tensor.buffer()->page_size();
    const uint32_t scores_cb_rm_page_size =
        round_up(scores_tensor.buffer()->aligned_page_size(), hal::get_l1_alignment());

    const uint32_t scores_tile_size = tt::tile_size(scores_data_format);

    const uint32_t input_tensor_buffer_factor = input_granularity * 2;
    const uint32_t output_tensor_buffer_factor = 2;

    // CB c_0: activation tiles (double-buffered)
    uint32_t cb_in_act_id = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_in_act_config =
        tt::tt_metal::CircularBufferConfig(
            input_tensor_buffer_factor * input_page_size, {{cb_in_act_id, input_data_format}})
            .set_page_size(cb_in_act_id, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_in_act_config);

    // CB c_1: pre-processed score tiles (one per expert), held resident during compute
    uint32_t cb_scores_id = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_scores_config =
        tt::tt_metal::CircularBufferConfig(reduction_dim_size * scores_tile_size, {{cb_scores_id, scores_data_format}})
            .set_page_size(cb_scores_id, scores_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_scores_config);

    // CB c_2: scratch for reading raw RM scores from DRAM (one page = one token row)
    uint32_t cb_scores_rm_id = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_scores_rm_config =
        tt::tt_metal::CircularBufferConfig(num_tokens * scores_cb_rm_page_size, {{cb_scores_rm_id, scores_data_format}})
            .set_page_size(cb_scores_rm_id, scores_cb_rm_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_scores_rm_config);

    // CB c_16: output tiles (double-buffered)
    uint32_t cb_out_id = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(
            output_tensor_buffer_factor * output_page_size, {{cb_out_id, output_data_format}})
            .set_page_size(cb_out_id, output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Reader CT args: fields 0..13 = DeepseekMoeFastReduceNcFusedReaderCtArgs / reader kernel; then TensorAccessorArgs.
    const DeepseekMoeFastReduceNcFusedReaderCtArgs reader_ct_named{
        .cb_in_act_id = cb_in_act_id,
        .cb_scores_id = cb_scores_id,
        .cb_scores_rm_id = cb_scores_rm_id,
        .act_page_size = input_page_size,
        .scores_buf_page_size = scores_page_size,
        .scores_tile_size = scores_tile_size,
        .num_cores = num_cores,
        .input_granularity = input_granularity,
        .reduction_dim = reduction_dim,
        .reduction_dim_size = reduction_dim_size,
        .inner_num_tiles = inner_num_tiles,
        .reduction_num_tiles = reduction_num_tiles,
        .num_tokens = num_tokens,
        .scores_cb_rm_page_size = scores_cb_rm_page_size,
    };
    std::vector<uint32_t> reader_ct_args = to_reader_ct_arg_vector(reader_ct_named);
    TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(scores_tensor.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        cb_out_id,
        output_page_size,
        num_cores,
        input_tensor_Wt,
        slice_Wt,
        static_cast<uint32_t>(output_tensors.size())};
    for (uint32_t i = 0; i < output_tensors.size(); ++i) {
        writer_ct_args.push_back(output_page_size);
    }
    for (const ttnn::Tensor& output_tensor : output_tensors) {
        TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct_args);
    }

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/kernels/"
        "deepseek_moe_fast_reduce_nc_fused_reader.cpp";
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
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/kernels/"
        "deepseek_moe_fast_reduce_nc_fused_compute.cpp";

    std::vector<uint32_t> compute_ct_args_group_1 = {
        num_cols_per_core_group_1,
        reduction_dim_size,
        input_granularity,
        cb_in_act_id,
        cb_scores_id,
        cb_out_id,
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
            reduction_dim_size,
            input_granularity,
            cb_in_act_id,
            cb_scores_id,
            cb_out_id,
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
    // indices are increments of num_cores. The total number of
    // units is num_tiles_per_group times num_cores.
    // For example, with 130 output tiles to be processed on an 8x8 grid
    // - the increment is 64
    // - the first 2 cores will have num_tiles_per_core 3 and the rest 2
    // - core x=0,y=0 will process output tiles 0, 64, and 128
    // - core x=1,y=0 will process output tiles 1, 65, and 129
    // - core x=2,y=0 will process output tiles 2 and 66
    // - core x=3,y=0 will process output tiles 3 and 67
    // - etc
    // The first tile that needs to be reduced has the same as the output tile.
    // That is the starting point for the reader, which then processes all
    // subsequent tiles to be reduced. The increment for the input indices is
    // the size of the inner dimensions in tiles (inner_num_tiles). The number
    // of tiles to process is the size of the reduce dimension in tiles
    // (reduction_num_tiles).
    TT_FATAL(
        core_group_2.ranges().empty() || num_cols_per_core_group_1 >= num_cols_per_core_group_2,
        "num_cols_per_core_group_1 must be greater than or equal to num_cols_per_core_group_2");

    const auto core_groups = {core_group_1, core_group_2};

    uint32_t start_tiles_read = 0;
    for (const auto& core_group : core_groups) {
        if (core_group.ranges().empty()) {
            continue;
        }

        uint32_t num_tiles_per_core =
            core_group_1.contains(core_group.ranges().at(0)) ? num_cols_per_core_group_1 : num_cols_per_core_group_2;
        uint32_t page_id_range_length = num_tiles_per_core * num_cores;

        for (const auto& core : corerange_to_cores(core_group)) {
            uint32_t start_tiles_to_read = start_tiles_read + page_id_range_length;

            uint32_t start_slice_row_offset = (start_tiles_read / input_tensor_Wt) * slice_Wt;
            uint32_t start_pages_read_in_row = start_tiles_read % input_tensor_Wt;

            std::vector<uint32_t> reader_rt_args = {
                input_tensor.buffer()->address(),
                scores_tensor.buffer()->address(),
                start_tiles_read,
                start_tiles_to_read};
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

            std::vector<uint32_t> writer_rt_args = {
                start_tiles_read, start_tiles_to_read, start_slice_row_offset, start_pages_read_in_row};
            for (const ttnn::Tensor& output_tensor : output_tensors) {
                writer_rt_args.push_back(output_tensor.buffer()->address());
            }
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);

            start_tiles_read++;
        }
    }

    return cached_program_t{
        std::move(program), {reader_kernel_id, writer_kernel_id, corerange_to_cores(all_cores), num_cores}};
}

void DeepseekMoEFastReduceNCFusedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const DeepseekMoEFastReduceNCFusedParams&,
    const DeepseekMoEFastReduceNCFusedInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const ttnn::Tensor& scores_tensor = tensor_args.scores_tensor;
    const std::vector<ttnn::Tensor>& output_tensors = tensor_return_value;

    auto& program = cached_program.program;
    const tt::tt_metal::KernelHandle& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const tt::tt_metal::KernelHandle& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& all_cores = cached_program.shared_variables.all_cores;
    const uint32_t ncores = cached_program.shared_variables.ncores;

    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = all_cores[i];
        auto& reader_rt_args = GetRuntimeArgs(program, reader_kernel_id, core);
        reader_rt_args[0] = input_tensor.buffer()->address();
        reader_rt_args[1] = scores_tensor.buffer()->address();

        auto& writer_rt_args = GetRuntimeArgs(program, writer_kernel_id, core);
        const uint32_t output_tensor_start_idx = 4;
        for (unsigned j = 0; j < output_tensors.size(); ++j) {
            writer_rt_args[output_tensor_start_idx + j] = output_tensors.at(j).buffer()->address();
        }
    }
}

}  // namespace ttnn::experimental::prim

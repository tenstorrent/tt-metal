// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_groupnorm_backward/moreh_groupnorm_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_groupnorm_backward_input_grad_impl(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t num_groups,
    Tensor &input_grad,
    const std::optional<const Tensor> gamma) {
    using namespace tt::constants;
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = output_grad.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto output_grad_shape = output_grad.get_legacy_shape();

    const auto n = output_grad_shape[0];
    const auto c = output_grad_shape[1];
    const auto h = output_grad_shape[2];
    const auto w = output_grad_shape[3];

    const auto origin_output_grad_shape = output_grad_shape.without_padding();

    const auto origin_h = origin_output_grad_shape[2];
    const auto origin_w = origin_output_grad_shape[3];

    const bool is_lastdim_layernorm = false;
    const bool is_groupnorm = true;

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;
    const auto num_rows = n * num_groups;
    const auto num_inner_tiles = (num_channels / num_groups) * Ht * Wt;

    const bool gamma_has_value = gamma.has_value();

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_rows_per_core_group_1,
         num_rows_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_rows);

    log_debug(LogTest, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_debug(LogTest, "num_rows_per_core_group_1: {}", num_rows_per_core_group_1);
    log_debug(LogTest, "num_rows_per_core_group_2: {}", num_rows_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t{1};
    const uint32_t in1_t{1};
    const uint32_t in2_t{1};
    const uint32_t in3_t{1};
    const uint32_t in4_t{1};
    const uint32_t in5_t{2};
    const uint32_t in6_t = gamma_has_value ? 1 : 0;
    const uint32_t in7_t = (do_mask_h || do_mask_w) ? 2 : 0;

    const uint32_t out0_t{1};

    uint32_t im0_t{num_inner_tiles};
    uint32_t im1_t{num_inner_tiles};
    const uint32_t im2_t{1};
    const uint32_t im3_t{1};
    const uint32_t im4_t{1};
    const uint32_t im5_t{1};
    const uint32_t im6_t{1};
    uint32_t im7_t{1};

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(output_grad.get_dtype());
    const auto single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    const auto cb_usage = (in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + in7_t + out0_t + im0_t + im1_t +
                           im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) *
                          single_tile_size;
    const auto available_L1 = device->l1_size_per_core() - L1_UNRESERVED_BASE;
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(LogTest, "Large moreh_layernorm_backward_input_grad algorithm is selected.");
        im0_t = 1;
        im1_t = 1;
        im7_t = 0;
    } else {
        log_info(LogTest, "Small moreh_layernorm_backward_input_grad algorithm is selected.");
    }

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},  // output_grad
            {CB::c_in1, in1_t},  // input
            {CB::c_in2, in2_t},  // mean
            {CB::c_in3, in3_t},  // rstd
            {CB::c_in4, in4_t},  // one
            {CB::c_in5, in5_t},  // inner_size(==n)
            {CB::c_in6, in6_t},
            {CB::c_in7, in7_t},
            {CB::c_out0, out0_t},  // input_grad
            {CB::c_intermed0, im0_t},
            {CB::c_intermed1, im1_t},
            {CB::c_intermed2, im2_t},
            {CB::c_intermed3, im3_t},
            {CB::c_intermed4, im4_t},
            {CB::c_intermed5, im5_t},
            {CB::c_intermed6, im6_t},
            {CB::c_intermed7, im7_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        use_large_algorithm ? "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_groupnorm_backward/input_grad/kernels/dataflow/"
                              "reader_moreh_groupnorm_backward_input_grad_large.cpp"
                            : "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_groupnorm_backward/input_grad/kernels/dataflow/"
                              "reader_moreh_groupnorm_backward_input_grad_small.cpp";

    const std::string writer_kernel_file(
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_groupnorm_backward/input_grad/kernels/dataflow/"
        "writer_moreh_groupnorm_backward_input_grad.cpp");

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";

    const auto compute_kernel_file = use_large_algorithm
                                         ? "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_layernorm_backward/kernels/"
                                           "moreh_layernorm_backward_input_grad_large_kernel.cpp"
                                         : "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_layernorm_backward/kernels/"
                                           "moreh_layernorm_backward_input_grad_small_kernel.cpp";

    const std::vector<uint32_t> compute_args_group_1{
        num_rows_per_core_group_1,
        origin_h,
        origin_w,
        num_inner_tiles,
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(is_lastdim_layernorm),
        static_cast<uint32_t>(is_groupnorm)};

    CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_rows_per_core_group_1, compute_args_group_1}, compute_defines);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_rows_per_core_group_2,
            origin_h,
            origin_w,
            num_inner_tiles,
            static_cast<uint32_t>(gamma_has_value),
            static_cast<uint32_t>(is_lastdim_layernorm),
            static_cast<uint32_t>(is_groupnorm)};

        CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_rows_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto input_addr = input.buffer()->address();
    const auto mean_addr = mean.buffer()->address();
    const auto rstd_addr = rstd.buffer()->address();

    const auto input_grad_addr = input_grad.buffer()->address();

    const auto gamma_addr = gamma_has_value ? gamma.value().buffer()->address() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        const std::vector<uint32_t> reader_runtime_args{
            output_grad_addr,
            static_cast<uint32_t>(is_dram(output_grad)),
            input_addr,
            static_cast<uint32_t>(is_dram(input)),
            mean_addr,
            static_cast<uint32_t>(is_dram(mean)),
            rstd_addr,
            static_cast<uint32_t>(is_dram(rstd)),
            gamma_addr,
            static_cast<uint32_t>(is_dram(gamma)),
            static_cast<uint32_t>(gamma_has_value),
            tile_offset,
            num_rows_per_core,
            num_inner_tiles,
            num_channels,
            num_groups,
            origin_h,
            origin_w,
        };
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            input_grad_addr,
            static_cast<uint32_t>(is_dram(input_grad)),
            tile_offset,
            num_rows_per_core,
            num_inner_tiles,
        };
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        tile_offset += num_rows_per_core * num_inner_tiles;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_args_callback = [reader_kernels_id = reader_kernels_id,
                                           writer_kernels_id = writer_kernels_id,
                                           num_cores_to_be_used = num_cores_to_be_used,
                                           num_cores_y = num_cores_y](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto output_grad_buffer = input_buffers.at(0);
        auto input_buffer = input_buffers.at(1);
        auto mean_buffer = input_buffers.at(2);
        auto rstd_buffer = input_buffers.at(3);
        auto gamma_buffer = input_buffers.at(4);

        auto input_grad_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
                runtime_args[0] = output_grad_buffer->address();
                runtime_args[2] = input_buffer->address();
                runtime_args[4] = mean_buffer->address();
                runtime_args[6] = rstd_buffer->address();
                if (gamma_buffer != nullptr) {
                    runtime_args[8] = gamma_buffer->address();
                }
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernels_id, core);
                runtime_args[0] = input_grad_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt

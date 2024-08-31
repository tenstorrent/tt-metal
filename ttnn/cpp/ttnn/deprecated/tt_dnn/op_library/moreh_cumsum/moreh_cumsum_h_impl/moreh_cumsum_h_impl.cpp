// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_cumsum/moreh_cumsum_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"

namespace tt {
using namespace constants;
namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_cumsum_h_impl(
    const Tensor &input,
    const Tensor &output,
    const bool &flip,
    const ttnn::DeviceComputeKernelConfig &compute_kernel_config) {
    const auto &shape = input.get_legacy_shape();
    const uint32_t W = shape[-1];
    const uint32_t H = shape[-2];
    const uint32_t NC = std::accumulate(shape.begin(), shape.end() - 2, 1, std::multiplies<uint32_t>{});

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;

    const uint32_t origin_H = shape.without_padding()[-2];
    const uint32_t mask_h = origin_H % TILE_HEIGHT != 0 ? origin_H % TILE_HEIGHT : TILE_HEIGHT;

    tt_metal::Program program = tt_metal::CreateProgram();

    const tt::DataFormat src_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    const tt::DataFormat dst_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    tt_metal::Device *device = input.device();

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cols = NC * Wt;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
        tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_cols);

    constexpr CB cb_src = CB::c_in0;
    constexpr CB cb_acc = CB::c_intermed0;
    constexpr CB cb_dst = CB::c_out0;

    constexpr uint32_t cb_src_num_tiles = 2;
    constexpr uint32_t cb_acc_num_tiles = 1;
    constexpr uint32_t cb_dst_num_tiles = 2;

    CreateCircularBuffer(
        program,
        all_cores,
        src_data_format,
        {
            {cb_src, cb_src_num_tiles},
            {cb_acc, cb_acc_num_tiles},
        });
    CreateCircularBuffer(
        program,
        all_cores,
        dst_data_format,
        {
            {cb_dst, cb_dst_num_tiles},
        });

    tt_metal::Buffer *src_buffer = input.buffer();
    const uint32_t src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args{
        src_is_dram,
        Ht,
        Wt,
        static_cast<uint32_t>(flip),
    };
    tt_metal::Buffer *dst_buffer = output.buffer();
    const uint32_t dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args{
        dst_is_dram,
        Ht,
        Wt,
        static_cast<uint32_t>(flip),
    };

    const auto reader_kernel_id = CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_cumsum/moreh_cumsum_h_impl/kernels/reader_moreh_cumsum_h.cpp",
        all_cores,
        reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_cumsum/moreh_cumsum_h_impl/kernels/writer_moreh_cumsum_h.cpp",
        all_cores,
        writer_compile_time_args);

    tt_metal::KernelHandle compute_kernel_1_id, compute_kernel_2_id;

    const std::vector<uint32_t> compute_args_group_1{
        Ht,
        Wt,
        num_cols_per_core_group_1,
        static_cast<uint32_t>(flip),
        mask_h,
    };
    std::map<string, string> compute_defines{};
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    if (is_floating_point(input.get_dtype())) {
        compute_defines["DATA_FLOAT"] = "1";
    }

    log_debug(
        LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);
    log_debug(LogOp, "data format {}", src_data_format);

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_cumsum/moreh_cumsum_h_impl/kernels/moreh_cumsum_h.cpp";
    compute_kernel_1_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_cols_per_core_group_1, compute_args_group_1},
        compute_defines,
        /*math_fidelity=*/math_fidelity,
        /*fp32_dest_acc_en=*/fp32_dest_acc_en,
        /*math_approx_mode=*/math_approx_mode,
        /*preserve_fp32_precision=*/fp32_dest_acc_en);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            Ht,
            Wt,
            num_cols_per_core_group_2,
            static_cast<uint32_t>(flip),
            mask_h,
        };

        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
            compute_defines,
            /*math_fidelity=*/math_fidelity,
            /*fp32_dest_acc_en=*/fp32_dest_acc_en,
            /*math_approx_mode=*/math_approx_mode,
            /*preserve_fp32_precision=*/fp32_dest_acc_en);
    }

    uint32_t cols_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core(i / num_cores_y, i % num_cores_y);

        uint32_t num_cols_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                input.buffer()->address(),
                num_cols_per_core,
                cols_offset,
            });

        tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                num_cols_per_core,
                cols_offset,
            });

        cols_offset += num_cols_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, num_cores, num_cores_y](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        log_debug(LogOp, "{}:{} args_callback ", __func__, __LINE__);
        auto src_dram_buffer = input_buffers.at(0);
        auto dst_dram_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < num_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_dram_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_dram_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt

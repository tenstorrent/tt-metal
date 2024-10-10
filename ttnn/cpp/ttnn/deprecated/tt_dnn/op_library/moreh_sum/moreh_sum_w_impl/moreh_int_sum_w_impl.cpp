// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
// based on reduce_op_multi_core_w.cpp in reduce op

#include <algorithm>

#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_sum_int_w_impl(const Tensor &input, const Tensor &output, const ttnn::DeviceComputeKernelConfig &compute_kernel_config) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Device *device{input.device()};
    tt_metal::Program program{tt_metal::CreateProgram()};

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format{datatype_to_dataformat_converter(output.get_dtype())};
    const auto shape{input.get_padded_shape()};

    const auto [W, H, other_dims_product] = extract_spatial_dims(shape);
    uint32_t Wt{W / TILE_WIDTH};
    uint32_t Ht{H / TILE_HEIGHT};
    uint32_t num_tiles = input.volume() / TILE_HW;
    auto num_rows{other_dims_product * Ht};


    // check mask for w-dim
    const auto input_shape_without_padding {input.get_logical_shape()};
    const auto origin_W {input_shape_without_padding[-1]};
    const bool do_mask_w {(origin_W % TILE_WIDTH) != 0};
    const auto mask_w {do_mask_w ? origin_W % TILE_WIDTH : TILE_WIDTH};

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] = get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);
    log_debug(
        LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    if (!fp32_dest_acc_en) {
        log_warning(LogOp, "fp32_dest_acc_en should be set for integer sum");
        fp32_dest_acc_en = true;
    }
    log_debug(LogOp, "do_mask_w {} mask_w {}", do_mask_w, mask_w);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid{device->compute_with_storage_grid_size()};
    const auto num_cores_y{grid.y};

    const uint32_t in0_t{2};        // input
    const uint32_t in1_t{1};        // mask
    const uint32_t intermed0_t{1};  // accumalated sum
    const uint32_t out0_t{2};       // output
    const auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_rows_per_core_group_1,
         num_rows_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_rows);

    log_debug(LogOp, "num_tiles {}, num_rows {}, num_rows_per_core_group_1 {}, num_rows_per_core_group_2 {}", num_tiles, num_rows, num_rows_per_core_group_1, num_rows_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},              // input
            {CB::c_in1, in1_t},              // mask
            {CB::c_intermed0, intermed0_t},  // accumalated sum
            {CB::c_out0, out0_t},            // output
        });
    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args =
             {static_cast<uint32_t>(is_dram(input))} ;
    std::map<string, string> reader_defines{};
    if (do_mask_w) {
        reader_defines["DO_MASK_W"] = "1";
    }
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(is_dram(output))};
    const auto reader_kernel_file{
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_w_impl/kernels/reader_moreh_int_sum_w.cpp"};
    const auto writer_kernel_file{
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_w_impl/kernels/writer_moreh_int_sum_w.cpp"};
    const auto reader_kernel_id{
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines)};
    const auto writer_kernel_id{CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args)};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{
        num_rows_per_core_group_1,  // num_rows
        Wt,                         // Wt
        origin_W};

    std::map<string, string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    const auto compute_kernel_file{"ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_w_impl/kernels/moreh_int_sum_w.cpp"};
    const auto compute_kernel_1_id = CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_rows_per_core_group_1, compute_args_group_1}, compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    std::optional<KernelHandle> compute_kernel_2_id{std::nullopt};
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_rows_per_core_group_2,  // num_rows
            Wt,                         // Wt
            origin_W};
        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_rows_per_core_group_2, compute_args_group_2},
            compute_defines,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    uint32_t out_dim_divider{Wt};
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core{0};
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        uint32_t num_tensor_tiles_per_core {num_rows_per_core * Wt};
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                input.buffer()->address(),
                num_tensor_tiles_per_core,
                tile_offset,  // tile index of row to start reading from
                mask_w
             });

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                num_tensor_tiles_per_core / out_dim_divider,  // number of tiles to write
                tile_offset / out_dim_divider                 // output tile start index
            });

        tile_offset += num_tensor_tiles_per_core;
    }

     auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, num_cores, num_cores_y](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        log_debug(LogOp, "{}:{} args_callback ", __func__, __LINE__);
        auto src_dram_buffer{input_buffers.at(0)};
        auto dst_dram_buffer{output_buffers.at(0)};

        for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
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

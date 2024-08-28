// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
// based on reduce_op_multi_core_w.cpp in reduce op

#include <algorithm>

#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"

namespace tt {
using namespace constants;
namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_sum_w_impl(const Tensor &a, const Tensor &output, const ttnn::DeviceComputeKernelConfig &compute_kernel_config) {
    tt_metal::ReduceOpMath reduce_op = tt_metal::ReduceOpMath::SUM;
    tt_metal::ReduceOpDim reduce_dim = tt_metal::ReduceOpDim::W;
    float scaler = 1.0f;

    const auto shape = a.get_legacy_shape();
    const auto [W, H, other_dims_product] = extract_spatial_dims(shape);

    uint32_t HW = H * W;
    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;

    // check mask for w-dim
    const auto input_shape_without_padding = shape.without_padding();
    const auto origin_W = input_shape_without_padding[-1];
    const bool do_mask_w = (origin_W % TILE_WIDTH) != 0;
    const auto mask_w = do_mask_w ? origin_W % TILE_WIDTH : TILE_WIDTH;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(a.device()->arch(), compute_kernel_config);
    log_debug(
        LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    tt_metal::Program program = tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    // Scaler datatype is hardcoded bfloat16 due to tile creation in reader
    tt::DataFormat scaler_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt_metal::detail::TileSize(scaler_cb_data_format);
    tt::DataFormat mask_w_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t mask_w_single_tile_size = tt_metal::detail::TileSize(mask_w_cb_data_format);
    tt::DataFormat intermed_cb_data_format = (fp32_dest_acc_en) ? tt::DataFormat::Float32: tt::DataFormat::Float16_b;
    tt::DataFormat intermed1_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t intermed_single_tile_size= tt_metal::detail::TileSize(intermed_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    uint32_t num_tiles = a.volume() / TILE_HW;

    tt_metal::Device *device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_rows = other_dims_product * Ht;

    const CoreRange all_core_range({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        split_work_to_cores(all_core_range, num_rows);

    string compute_kernel_name = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_w_impl/kernels/moreh_sum_w.cpp";

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt_metal::CircularBufferConfig cb_scaler_config =
        tt_metal::CircularBufferConfig(num_input_tiles * scaler_single_tile_size, {{CB::c_in2, scaler_cb_data_format}})
            .set_page_size(CB::c_in2, scaler_single_tile_size);
    auto cb_scaler = tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

    tt_metal::CircularBufferConfig cb_mask_w_config =
        tt_metal::CircularBufferConfig(mask_w_single_tile_size, {{CB::c_in3, mask_w_cb_data_format}})
            .set_page_size(CB::c_in3, mask_w_single_tile_size);
    auto cb_mask_w = tt_metal::CreateCircularBuffer(program, all_cores, cb_mask_w_config);

    tt_metal::CircularBufferConfig cb_intermed0_config =
        tt_metal::CircularBufferConfig(intermed_single_tile_size, {{CB::c_intermed0, intermed_cb_data_format}})
            .set_page_size(CB::c_intermed0, intermed_single_tile_size);
    auto cb_intermed0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed0_config);

    tt_metal::CircularBufferConfig cb_intermed1_config =
        tt_metal::CircularBufferConfig(intermed_single_tile_size, {{CB::c_intermed1, intermed1_cb_data_format}})
            .set_page_size(CB::c_intermed1, intermed_single_tile_size);
    auto cb_intermed1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed1_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    bfloat16 bfloat_scaler_value = bfloat16(scaler);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
    tt_metal::Buffer *src_buffer = a.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram, packed_scaler_value};
    tt_metal::Buffer *dst_buffer = output.buffer();
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};


    std::map<string, string> reader_defines{};
    if (do_mask_w) {
        reader_defines["DO_MASK_W"] = "1";
    }
    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_w_impl/kernels/reader_moreh_sum_w.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_w_impl/kernels/writer_moreh_sum_w.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::map<string, string> reduce_defines = reduce_op_utils::get_defines(reduce_op, reduce_dim);
    if (fp32_dest_acc_en) {
        reduce_defines["FP32_DEST_ACC_EN"] = "1";
    }
    vector<uint32_t> compute_kernel_args_group_1 = {
        num_rows_per_core_group_1,  // Ht
        Wt,                         // Wt
        1,                          // NC
        origin_W,
    };

    // set preserve_fp32_precision to the same value as fp32_dest_acc_en
    bool preserve_fp32_precision = fp32_dest_acc_en;
    auto reduce_compute_kernel_group_1_id = tt_metal::CreateKernel(
        program,
        compute_kernel_name,
        core_group_1,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .preserve_fp32_precision = preserve_fp32_precision, .math_approx_mode = math_approx_mode, .compile_args = compute_kernel_args_group_1, .defines = reduce_defines});

    if (!core_group_2.ranges().empty()) {
        vector<uint32_t> compute_kernel_args_group_2 = {
            num_rows_per_core_group_2,  // Ht
            Wt,                         // Wt
            1,                          // NC
            origin_W,
        };

        auto reduce_compute_kernel_group_2_id = tt_metal::CreateKernel(
            program,
            compute_kernel_name,
            core_group_2,
            tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .preserve_fp32_precision = preserve_fp32_precision, .math_approx_mode = math_approx_mode, .compile_args = compute_kernel_args_group_2, .defines = reduce_defines});
    }

    uint32_t out_dim_divider = Wt;
    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_rows_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;
        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                a.buffer()->address(),
                num_tensor_tiles_per_core,
                num_tiles_read,  // tile index of row to start reading from
                mask_w
            });

        tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                num_tensor_tiles_per_core / out_dim_divider,  // number of tiles to write
                num_tiles_read / out_dim_divider              // output tile start index
            });
        num_tiles_read += num_tensor_tiles_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, num_cores, num_cores_y](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        log_debug(LogOp, "{}:{} args_callback ", __func__, __LINE__);
        auto src_dram_buffer = input_buffers.at(0);
        auto dst_dram_buffer = output_buffers.at(0);

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

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

operation::ProgramWithCallbacks moreh_nll_loss_step1_impl(
    const Tensor &target,
    const std::optional<const Tensor> weight,
    Tensor &output,
    const int32_t ignore_index,
    const bool reduction_mean,
    const uint32_t channel_size,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    auto target_shape = target.get_legacy_shape();
    auto N = target_shape[1];

    const auto target_shape_without_padding = target_shape.without_padding();
    const auto origin_N = target_shape_without_padding[1];

    const bool weight_has_value = weight.has_value();

    auto H = target_shape[-2];
    auto W = target_shape[-1];
    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;

    // copy TILE per core
    uint32_t units_to_divide = target.volume() / H / W * (Ht * Wt);

    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(core_range, units_to_divide);

    auto* device = target.device();
    auto arch = device->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    Program program = Program();

    // create circular buffers
    const auto target_data_format = tt_metal::datatype_to_dataformat_converter(target.get_dtype());
    const auto data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    const auto target_tile_size = tt_metal::detail::TileSize(target_data_format);
    const auto data_tile_size = tt_metal::detail::TileSize(data_format);
    const auto intermed_tile_size = tt_metal::detail::TileSize(intermed_data_format);

    const uint32_t available_L1 = device->l1_size_per_core() - L1_UNRESERVED_BASE;

    uint32_t target_num_tile = 1;
    uint32_t weight_num_tile = weight_has_value ? div_up(channel_size, TILE_WIDTH) : 0;
    uint32_t intermed_num_tile = 1;
    uint32_t output_num_tile = 1;
    uint32_t cb_usage = target_num_tile * target_tile_size + weight_num_tile * data_tile_size +
                        intermed_num_tile * intermed_tile_size + output_num_tile * data_tile_size;

    const bool use_large_algorithm = cb_usage >= available_L1;;

    if (use_large_algorithm) {
        CreateCircularBuffer(
            program,
            all_cores,
            data_format,
            {
                {CB::c_in0, 1, tt::DataFormat::Int32},     // traget
                {CB::c_in1, 1},                            // weight
                {CB::c_intermed0, 1, intermed_data_format},  // tmp_weight
                {CB::c_out0, 1},                           // output
            });
    } else {
        CreateCircularBuffer(
            program,
            all_cores,
            data_format,
            {
                {CB::c_in0, 1, tt::DataFormat::Int32},     // traget
                {CB::c_in1, weight_num_tile},              // weight
                {CB::c_intermed0, 1, intermed_data_format},  // tmp_weight
                {CB::c_out0, 1},                           // output
            });
    }

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(target)),
        static_cast<uint32_t>(is_dram(weight)),
        static_cast<uint32_t>(weight_has_value)};

    const std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(output))};

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    if (weight_has_value) {
        reader_defines["WEIGHT"] = 1;
    }

    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = 1;
    }
    const auto reader_kernel_file =
        use_large_algorithm ? "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_step1/kernels/reader_moreh_nll_loss_step1_large.cpp"
            : "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_step1/kernels/reader_moreh_nll_loss_step1.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_step1/kernels/"
        "writer_moreh_nll_loss_step1.cpp";

    auto reader_kernel_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    auto writer_kernel_id =
        CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    const auto target_addr = target.buffer()->address();
    const auto weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    const auto output_addr = output.buffer()->address();

    // Set Runtime Args
    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
        uint32_t num_units_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_units_per_core = units_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t element_size = weight_has_value ? weight.value().element_size() : 0;
        vector<uint32_t> reader_args = {
            target_addr,
            weight_addr,
            static_cast<uint32_t>(ignore_index),
            num_units_per_core,
            tile_offset,
            origin_N,
            channel_size,
            weight_num_tile,
            element_size,
            target.element_size(),
        };

        vector<uint32_t> writer_args = {output_addr, num_units_per_core, tile_offset};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{num_units_per_core};

        tile_offset += num_units_per_core;
    }

    return {
        .program = std::move(program),
        .override_runtime_arguments_callback =
            create_override_runtime_arguments_callback(reader_kernel_id, writer_kernel_id, num_cores, core_h)};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt

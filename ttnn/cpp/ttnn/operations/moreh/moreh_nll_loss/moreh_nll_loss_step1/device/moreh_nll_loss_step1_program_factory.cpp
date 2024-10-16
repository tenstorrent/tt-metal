// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_step1_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_nll_loss_step1 {

MorehNllLossStep1DeviceOperation::Factory::cached_program_t MorehNllLossStep1DeviceOperation::Factory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& target = tensor_args.target_tensor;
    const std::optional<Tensor>& weight = tensor_args.weight_tensor;
    const Tensor& output = tensor_return_value;
    const std::string reduction = operation_attributes.reduction;
    const uint32_t ignore_index = operation_attributes.ignore_index;
    const uint32_t channel_size = operation_attributes.channel_size;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    auto target_shape = target.get_shape().value;
    auto N = target_shape[1];

    const auto target_shape_without_padding = target_shape.without_padding();
    const auto origin_N = target_shape_without_padding[1];

    const bool weight_has_value = weight.has_value();

    auto H = target_shape[-2];
    auto W = target_shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    // copy TILE per core
    uint32_t units_to_divide = target.volume() / H / W * (Ht * Wt);

    tt::tt_metal::Device* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    Program program = Program();

    // create circular buffers
    const auto target_data_format = tt_metal::datatype_to_dataformat_converter(target.get_dtype());
    const auto data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    const auto target_tile_size = tt_metal::detail::TileSize(target_data_format);
    const auto data_tile_size = tt_metal::detail::TileSize(data_format);
    const auto intermed_tile_size = tt_metal::detail::TileSize(intermed_data_format);

    const uint32_t available_L1 = device->l1_size_per_core() - device->get_base_allocator_addr(HalMemType::L1);

    uint32_t target_num_tile = 1;
    uint32_t weight_num_tile = weight_has_value ? div_up(channel_size, tt::constants::TILE_WIDTH) : 0;
    uint32_t intermed_num_tile = 1;
    uint32_t output_num_tile = 1;
    uint32_t cb_usage = target_num_tile * target_tile_size + weight_num_tile * data_tile_size +
                        intermed_num_tile * intermed_tile_size + output_num_tile * data_tile_size;

    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        tt::operations::primary::CreateCircularBuffer(program,
                                                      all_cores,
                                                      data_format,
                                                      {
                                                          {CB::c_in0, 1, tt::DataFormat::Int32},       // target
                                                          {CB::c_in1, 1},                              // weight
                                                          {CB::c_intermed0, 1, intermed_data_format},  // tmp_weight
                                                          {CB::c_out0, 1},                             // output
                                                      });
    } else {
        tt::operations::primary::CreateCircularBuffer(program,
                                                      all_cores,
                                                      data_format,
                                                      {
                                                          {CB::c_in0, 1, tt::DataFormat::Int32},       // target
                                                          {CB::c_in1, weight_num_tile},                // weight
                                                          {CB::c_intermed0, 1, intermed_data_format},  // tmp_weight
                                                          {CB::c_out0, 1},                             // output
                                                      });
    }

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(target)),
        static_cast<uint32_t>(weight.has_value() ? tt::operations::primary::is_dram(weight.value()) : false),
        static_cast<uint32_t>(weight_has_value)};

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(output))};

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    if (weight_has_value) {
        reader_defines["WEIGHT"] = 1;
    }

    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = 1;
    }
    const auto reader_kernel_file = use_large_algorithm
                                        ? "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1/device/"
                                          "kernels/reader_moreh_nll_loss_step1_large.cpp"
                                        : "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1/device/"
                                          "kernels/reader_moreh_nll_loss_step1.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1/device/kernels/"
        "writer_moreh_nll_loss_step1.cpp";

    auto reader_kernel_id = tt::operations::primary::CreateReadKernel(
        program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    auto writer_kernel_id = tt::operations::primary::CreateWriteKernel(
        program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    const auto target_addr = target.buffer()->address();
    const auto weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    const auto output_addr = output.buffer()->address();

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
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

    return {std::move(program),
            {.unary_reader_kernel_id = reader_kernel_id,
             .unary_writer_kernel_id = writer_kernel_id,
             .num_cores = num_cores,
             .num_cores_y = core_h}};
}

void MorehNllLossStep1DeviceOperation::Factory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    const uint32_t target_addr = tensor_args.target_tensor.buffer()->address();
    const uint32_t weight_addr =
        tensor_args.weight_tensor.has_value() ? tensor_args.weight_tensor.value().buffer()->address() : 0;
    const uint32_t ignore_index = operation_attributes.ignore_index;

    const uint32_t output_addr = tensor_return_value.buffer()->address();

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = target_addr;
            runtime_args[1] = weight_addr;
            runtime_args[2] = ignore_index;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = output_addr;
        }
    }
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_step1

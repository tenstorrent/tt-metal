// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <cmath>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::normalization;

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const tt::tt_metal::Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

template <typename F>
void set_or_update_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    tt::tt_metal::KernelHandle compute_kernel_id,
    CoreCoord compute_with_storage_grid_size,
    const BatchNormOperation::operation_attributes_t& operation_attributes,
    const BatchNormOperation::tensor_args_t& tensor_args,
    BatchNormOperation::tensor_return_value_t& c,
    F handle_args) {
    const auto& [input_tensor, batch_mean_tensor, batch_var_tensor, weight_tensor, bias_tensor, _] = tensor_args;
    const auto eps = operation_attributes.eps;

    const bool weight_has_value = weight_tensor.has_value();
    const bool bias_has_value = bias_tensor.has_value();

    const auto ashape = input_tensor.padded_shape();
    const auto bshape = batch_mean_tensor.padded_shape();

    const auto [aN, aC, aHt, aWt] = extract_shape_dims(input_tensor);
    const auto [bN, bC, bHt, bWt] = extract_shape_dims(batch_mean_tensor);
    const auto [cN, cC, cHt, cWt] = extract_shape_dims(c);

    uint32_t num_output_tiles = c.physical_volume() / c.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    constexpr size_t num_reader_args = 11;
    constexpr size_t num_writer_args = 14;
    constexpr size_t num_kernel_args = 3;
    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            handle_args(program, reader_kernel_id, core, std::array<uint32_t, num_reader_args>{0});
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, num_writer_args>{0});
            handle_args(program, compute_kernel_id, core, std::array<uint32_t, num_kernel_args>{0});
            continue;
        }

        uint32_t cHtWt = cHt * cWt;
        const auto scalar = eps;
        const auto packed_scalar_eps = input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32
                                           ? std::bit_cast<uint32_t>(scalar)
                                           : pack_two_bfloat16_into_uint32({scalar, scalar});

        std::array reader_runtime_args = {
            packed_scalar_eps,
            input_tensor.buffer()->address(),
            start_tile_id,
            num_tiles_per_core,
            cHtWt,
            aHt * aWt * aC * (aN > 1),
            aHt * aWt * (aC > 1),
            cN,
            cC,
            cHt,
            cWt};
        handle_args(program, reader_kernel_id, core, reader_runtime_args);

        const auto weight_addr = weight_has_value ? weight_tensor->buffer()->address() : 0;
        const auto bias_addr = bias_has_value ? bias_tensor->buffer()->address() : 0;
        std::array writer_runtime_args = {
            batch_mean_tensor.buffer()->address(),  //  batch mean
            batch_var_tensor.buffer()->address(),   //  batch var
            weight_addr,                            // weight
            bias_addr,                              // bias
            c.buffer()->address(),                  // output
            start_tile_id,
            num_tiles_per_core,
            cHtWt,
            bHt * bWt * bC * (bN > 1),
            bHt * bWt * (bC > 1),
            cN,
            cC,
            cHt,
            cWt};
        handle_args(program, writer_kernel_id, core, writer_runtime_args);

        auto counter = start_tile_id % cHtWt;
        auto freq = cHtWt;

        std::array compute_runtime_args = {num_tiles_per_core, freq, counter};
        handle_args(program, compute_kernel_id, core, compute_runtime_args);

        start_tile_id += num_tiles_per_core;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::normalization {
BatchNormOperation::BatchNormFactory::cached_program_t BatchNormOperation::BatchNormFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& [input_tensor, batch_mean_tensor, batch_var_tensor, weight_tensor, bias_tensor, _] = tensor_args;

    auto program = CreateProgram();

    auto* device = input_tensor.device();

    const bool weight_has_value = weight_tensor.has_value();
    const bool bias_has_value = bias_tensor.has_value();

    auto a_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    auto b_data_format = datatype_to_dataformat_converter(batch_mean_tensor.dtype());
    auto c_data_format = datatype_to_dataformat_converter(output.dtype());
    auto d_data_format = datatype_to_dataformat_converter(batch_var_tensor.dtype());
    auto e_data_format =
        weight_has_value ? datatype_to_dataformat_converter(weight_tensor->dtype()) : DataFormat::Float16_b;
    auto f_data_format =
        bias_has_value ? datatype_to_dataformat_converter(bias_tensor->dtype()) : DataFormat::Float16_b;

    uint32_t a_single_tile_size = tt_metal::detail::TileSize(a_data_format);
    uint32_t b_single_tile_size = tt_metal::detail::TileSize(b_data_format);
    uint32_t c_single_tile_size = tt_metal::detail::TileSize(c_data_format);
    uint32_t d_single_tile_size = tt_metal::detail::TileSize(d_data_format);
    uint32_t e_single_tile_size = tt_metal::detail::TileSize(e_data_format);
    uint32_t f_single_tile_size = tt_metal::detail::TileSize(f_data_format);

    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    // we parallelize the computation across the output tiles
    constexpr bool row_major = true;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Number of tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    uint32_t b_num_tiles_per_cb = num_tiles_per_cb;

    // Input buffers
    auto [input_tensor_cb, input_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_0, program, all_device_cores, a_single_tile_size, num_tiles_per_cb, a_data_format);  // input
    auto [batch_mean_tensor_cb, batch_mean_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_1,
        program,
        all_device_cores,
        b_single_tile_size,
        b_num_tiles_per_cb,
        b_data_format);  // batch_mean
    auto [output_tensor_cb, output_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_2, program, all_device_cores, c_single_tile_size, num_tiles_per_cb, c_data_format);  // output
    auto [batch_var_tensor_cb, batch_var_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_3,
        program,
        all_device_cores,
        d_single_tile_size,
        b_num_tiles_per_cb,
        d_data_format);  // batch_var
    auto [eps_cb, eps_cb_handle] = create_cb(
        tt::CBIndex::c_4, program, all_device_cores, d_single_tile_size, b_num_tiles_per_cb, d_data_format);  // eps
    auto [weight_tensor_cb, weight_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_5, program, all_device_cores, e_single_tile_size, b_num_tiles_per_cb, e_data_format);  // weight
    auto [bias_tensor_cb, bias_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_6, program, all_device_cores, f_single_tile_size, b_num_tiles_per_cb, f_data_format);  // bias

    // Temporary buffers to store intermediate results
    auto [den_cb, den_cb_handle] = create_cb(
        tt::CBIndex::c_7,
        program,
        all_device_cores,
        a_single_tile_size,
        num_tiles_per_cb,
        a_data_format);  // to store 1/(sqrt(batch_var + eps))
    auto [temp_1_cb, temp_1_cb_handle] =
        create_cb(tt::CBIndex::c_8, program, all_device_cores, a_single_tile_size, num_tiles_per_cb, a_data_format);

    auto a_is_dram = static_cast<uint32_t>(input_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    auto b_is_dram = static_cast<uint32_t>(batch_mean_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    auto c_is_dram = static_cast<uint32_t>(output.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    auto d_is_dram = static_cast<uint32_t>(batch_var_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    const auto e_is_dram = weight_has_value and weight_tensor->buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    const auto f_is_dram = bias_has_value and bias_tensor->buffer()->buffer_type() == tt_metal::BufferType::DRAM;

    std::map<std::string, std::string> dataflow_defines;  // Currently support only for fp32, bf16
    if (input_tensor.dtype() == DataType::FLOAT32) {
        dataflow_defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element<float>";
        dataflow_defines["FILL_WITH_VALUE_FLOAT"] = "fill_with_val<1024, float>";
    } else {
        dataflow_defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element_bfloat16";
        dataflow_defines["FILL_WITH_VALUE"] = "fill_with_val_bfloat16";
    }

    // READER KERNEL
    auto reader_defines = dataflow_defines;
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig({a_is_dram, input_tensor_cb, eps_cb}, std::move(reader_defines)));

    // WRITER KERNEL
    auto writer_defines = dataflow_defines;
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig(
            {
                b_is_dram,
                c_is_dram,
                d_is_dram,
                e_is_dram,
                f_is_dram,
                static_cast<uint32_t>(weight_has_value),
                static_cast<uint32_t>(bias_has_value),
                batch_mean_tensor_cb,
                output_tensor_cb,
                batch_var_tensor_cb,
                weight_tensor_cb,
                bias_tensor_cb,
            },
            std::move(writer_defines)));

    // COMPUTE KERNEL
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        for (const auto cb_index :
             {input_tensor_cb,
              batch_mean_tensor_cb,
              batch_var_tensor_cb,
              eps_cb,
              den_cb,
              weight_tensor_cb,
              temp_1_cb,
              bias_tensor_cb}) {
            unpack_to_dest_mode[cb_index] = UnpackToDestMode::UnpackToDestFp32;
        }
    }

    std::vector<uint32_t> compute_kernel_args = {
        static_cast<uint32_t>(weight_has_value),
        static_cast<uint32_t>(bias_has_value),
        input_tensor_cb,
        batch_mean_tensor_cb,
        output_tensor_cb,
        batch_var_tensor_cb,
        eps_cb,
        den_cb,
        weight_tensor_cb,
        temp_1_cb,
        bias_tensor_cb};
    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        fmt::format(
            "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_{}.cpp",
            fp32_dest_acc_en ? "sfpu_kernel" : "kernel"),
        all_device_cores,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .compile_args = compute_kernel_args});

    auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        set_runtime_args);

    return {
        std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, compute_with_storage_grid_size}};
}

void BatchNormOperation::BatchNormFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto update_args =
        [](tt::tt_metal::Program& program, tt::tt_metal::KernelHandle kernel_id, CoreCoord core, auto&& args) {
            auto& all_args = GetRuntimeArgs(program, kernel_id);
            auto& core_args = all_args.at(core.x).at(core.y);
            std::copy(args.begin(), args.end(), core_args.data());
        };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        cached_program.program,
        cached_program.shared_variables.reader_kernel_id,
        cached_program.shared_variables.writer_kernel_id,
        cached_program.shared_variables.compute_kernel_id,
        cached_program.shared_variables.compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        update_args);
}

}  // namespace ttnn::operations::normalization

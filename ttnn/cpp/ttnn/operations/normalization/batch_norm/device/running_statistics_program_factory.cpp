// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "running_statistics_device_operation.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
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
    bool any_float32,
    const RunningStatistics::operation_attributes_t& operation_attributes,
    const RunningStatistics::tensor_args_t& tensor_args,
    RunningStatistics::tensor_return_value_t& c,
    F handle_args) {
    const auto& [batch_mean_tensor, batch_var_tensor, running_mean_tensor, running_var_tensor] = tensor_args;
    const auto momentum = operation_attributes.momentum;

    const bool running_mean_has_value = running_mean_tensor.has_value();
    const bool running_var_has_value = running_var_tensor.has_value();

    const auto ashape = batch_mean_tensor.padded_shape();
    const auto bshape = batch_var_tensor.padded_shape();

    const auto [aN, aC, aHt, aWt] = extract_shape_dims(batch_mean_tensor);
    const auto [bN, bC, bHt, bWt] = extract_shape_dims(batch_var_tensor);
    const auto [cN, cC, cHt, cWt] = extract_shape_dims(c);

    uint32_t num_output_tiles = c.physical_volume() / c.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    const uint32_t cHtWt = cHt * cWt;
    const auto scalar = momentum;
    const auto packed_scalar_momentum =
        any_float32 ? std::bit_cast<uint32_t>(scalar) : pack_two_bfloat16_into_uint32({scalar, scalar});

    constexpr size_t num_reader_args = 11;
    constexpr size_t num_writer_args = 13;
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
        std::array reader_runtime_args = {
            packed_scalar_momentum,
            batch_mean_tensor.buffer()->address(),
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

        const auto running_mean_addr = running_mean_has_value ? running_mean_tensor->buffer()->address() : 0;
        const auto running_var_addr = running_var_has_value ? running_var_tensor->buffer()->address() : 0;
        std::array writer_runtime_args = {
            batch_var_tensor.buffer()->address(),  //  batch var
            running_mean_addr,                     // old running mean
            running_var_addr,                      // old running var
            c.buffer()->address(),                 // output
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
RunningStatistics::RunningStatisticsProgramFactory::cached_program_t
RunningStatistics::RunningStatisticsProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& [batch_mean_tensor, batch_var_tensor, running_mean_tensor, running_var_tensor] = tensor_args;

    auto program = CreateProgram();

    auto* device = batch_mean_tensor.device();

    const bool running_mean_has_value = running_mean_tensor.has_value();
    const bool running_var_has_value = running_var_tensor.has_value();

    auto a_data_format = datatype_to_dataformat_converter(batch_mean_tensor.dtype());
    auto b_data_format = datatype_to_dataformat_converter(batch_var_tensor.dtype());
    auto c_data_format = datatype_to_dataformat_converter(output.dtype());
    auto d_data_format =
        running_mean_has_value ? datatype_to_dataformat_converter(running_mean_tensor->dtype()) : DataFormat::Float16_b;
    auto e_data_format =
        running_var_has_value ? datatype_to_dataformat_converter(running_var_tensor->dtype()) : DataFormat::Float16_b;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    const bool any_float32 =
        (a_data_format == DataFormat::Float32 || b_data_format == DataFormat::Float32 ||
         c_data_format == DataFormat::Float32 || d_data_format == DataFormat::Float32 ||
         e_data_format == DataFormat::Float32);
    const bool use_fp32_acc = fp32_dest_acc_en || any_float32;
    auto interm_data_format = any_float32 ? DataFormat::Float32 : a_data_format;

    uint32_t a_single_tile_size = tt::tile_size(a_data_format);
    uint32_t b_single_tile_size = tt::tile_size(b_data_format);
    uint32_t c_single_tile_size = tt::tile_size(c_data_format);
    uint32_t d_single_tile_size = tt::tile_size(d_data_format);
    uint32_t e_single_tile_size = tt::tile_size(e_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);

    auto running_stat_data_format =
        running_mean_has_value ? d_data_format : (running_var_has_value ? e_data_format : DataFormat::Float16_b);
    const bool stat_format_needs_typecast =
        (interm_data_format == DataFormat::Float32 && running_stat_data_format != DataFormat::Float32);
    const bool needs_mean_typecast = running_mean_has_value && stat_format_needs_typecast;
    const bool needs_var_typecast = running_var_has_value && stat_format_needs_typecast;

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Number of tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    uint32_t b_num_tiles_per_cb = num_tiles_per_cb;

    // Input buffers
    auto [batch_mean_tensor_cb, batch_mean_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_0,
        program,
        all_device_cores,
        a_single_tile_size,
        num_tiles_per_cb,
        a_data_format);  // batch_mean
    auto [batch_var_tensor_cb, batch_var_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_1,
        program,
        all_device_cores,
        b_single_tile_size,
        b_num_tiles_per_cb,
        b_data_format);  // batch_var
    auto [output_tensor_cb, output_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_2, program, all_device_cores, c_single_tile_size, num_tiles_per_cb, c_data_format);  // output
    auto [old_running_mean_tensor_cb, old_running_mean_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_3,
        program,
        all_device_cores,
        d_single_tile_size,
        b_num_tiles_per_cb,
        d_data_format);  // old running mean
    auto [old_running_var_tensor_cb, old_running_var_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_4,
        program,
        all_device_cores,
        e_single_tile_size,
        b_num_tiles_per_cb,
        e_data_format);  // old running var
    auto [momentum_cb, momentum_cb_handle] = create_cb(
        tt::CBIndex::c_5,
        program,
        all_device_cores,
        interm_single_tile_size,
        b_num_tiles_per_cb,
        interm_data_format);  // momentum
    auto [one_cb, one_cb_handle] = create_cb(
        tt::CBIndex::c_6,
        program,
        all_device_cores,
        interm_single_tile_size,
        b_num_tiles_per_cb,
        interm_data_format);  // to store 1
    auto [updated_m_cb, updated_m_cb_handle] = create_cb(
        tt::CBIndex::c_7,
        program,
        all_device_cores,
        needs_mean_typecast ? interm_single_tile_size : d_single_tile_size,
        b_num_tiles_per_cb,
        needs_mean_typecast ? interm_data_format : d_data_format);  // updated running mean (staging when typecast)
    auto [updated_v_cb, updated_v_cb_handle] = create_cb(
        tt::CBIndex::c_8,
        program,
        all_device_cores,
        needs_var_typecast ? interm_single_tile_size : e_single_tile_size,
        b_num_tiles_per_cb,
        needs_var_typecast ? interm_data_format : e_data_format);  // updated running var (staging when typecast)

    uint32_t writer_updated_m_cb = updated_m_cb;
    uint32_t writer_updated_v_cb = updated_v_cb;
    if (needs_mean_typecast) {
        auto [wm_cb, wm_cb_handle] = create_cb(
            tt::CBIndex::c_12, program, all_device_cores, d_single_tile_size, b_num_tiles_per_cb, d_data_format);
        writer_updated_m_cb = wm_cb;
    }
    if (needs_var_typecast) {
        auto [wv_cb, wv_cb_handle] = create_cb(
            tt::CBIndex::c_13, program, all_device_cores, e_single_tile_size, b_num_tiles_per_cb, e_data_format);
        writer_updated_v_cb = wv_cb;
    }

    // Intermediate buffers required for updation of running stats
    auto [tmp1_cb, tmp1_cb_handle] = create_cb(
        tt::CBIndex::c_9, program, all_device_cores, interm_single_tile_size, b_num_tiles_per_cb, interm_data_format);

    auto [tmp2_cb, tmp2_cb_handle] = create_cb(
        tt::CBIndex::c_10, program, all_device_cores, interm_single_tile_size, b_num_tiles_per_cb, interm_data_format);

    auto [tmp3_cb, tmp3_cb_handle] = create_cb(
        tt::CBIndex::c_11, program, all_device_cores, interm_single_tile_size, b_num_tiles_per_cb, interm_data_format);

    std::vector<uint32_t> reader_compile_time_args = {
        batch_mean_tensor_cb,
        momentum_cb,
        one_cb,
    };
    tt::tt_metal::TensorAccessorArgs(batch_mean_tensor.buffer()).append_to(reader_compile_time_args);
    reader_compile_time_args.push_back(static_cast<uint32_t>(any_float32));

    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<uint32_t>(running_mean_has_value),
        static_cast<uint32_t>(running_var_has_value),
        batch_var_tensor_cb,
        output_tensor_cb,
        old_running_mean_tensor_cb,
        old_running_var_tensor_cb,
        writer_updated_m_cb,
        writer_updated_v_cb,
    };
    tt::tt_metal::TensorAccessorArgs(batch_var_tensor.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(running_mean_tensor ? running_mean_tensor->buffer() : nullptr)
        .append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(running_var_tensor ? running_var_tensor->buffer() : nullptr)
        .append_to(writer_compile_time_args);
    writer_compile_time_args.push_back(static_cast<uint32_t>(running_stat_data_format == DataFormat::Float32));

    // READER KERNEL
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_running_statistics.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // WRITER KERNEL
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_running_statistics.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // COMPUTE KERNEL
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (use_fp32_acc) {
        for (const auto cb_index :
             {batch_mean_tensor_cb,
              batch_var_tensor_cb,
              output_tensor_cb,
              old_running_mean_tensor_cb,
              old_running_var_tensor_cb,
              updated_m_cb,
              updated_v_cb,
              momentum_cb,
              one_cb,
              tmp1_cb,
              tmp2_cb,
              tmp3_cb}) {
            unpack_to_dest_mode[cb_index] = UnpackToDestMode::UnpackToDestFp32;
        }
    }

    auto tc_out_fmt = stat_format_needs_typecast ? static_cast<uint32_t>(running_stat_data_format)
                                                 : static_cast<uint32_t>(DataFormat::Float32);

    std::vector<uint32_t> compute_kernel_args = {
        static_cast<uint32_t>(running_mean_has_value),
        static_cast<uint32_t>(running_var_has_value),
        batch_mean_tensor_cb,
        batch_var_tensor_cb,
        output_tensor_cb,
        old_running_mean_tensor_cb,
        old_running_var_tensor_cb,
        updated_m_cb,
        updated_v_cb,
        momentum_cb,
        one_cb,
        tmp1_cb,
        tmp2_cb,
        tmp3_cb,
        writer_updated_m_cb,
        writer_updated_v_cb,
        static_cast<uint32_t>(stat_format_needs_typecast),
        static_cast<uint32_t>(DataFormat::Float32),
        tc_out_fmt};

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        fmt::format(
            "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/running_statistics_{}.cpp",
            any_float32 ? "sfpu_kernel" : "kernel"),
        all_device_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = use_fp32_acc,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .math_approx_mode = math_approx_mode,
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
        any_float32,
        operation_attributes,
        tensor_args,
        output,
        set_runtime_args);

    return {
        std::move(program),
        {reader_kernel_id, writer_kernel_id, compute_kernel_id, compute_with_storage_grid_size, any_float32}};
}

void RunningStatistics::RunningStatisticsProgramFactory::override_runtime_arguments(
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
        cached_program.shared_variables.any_float32,
        operation_attributes,
        tensor_args,
        output,
        update_args);
}

}  // namespace ttnn::operations::normalization

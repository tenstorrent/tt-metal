// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <cmath>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::normalization;

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

template <typename F>
void set_or_update_runtime_arguments(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    KernelHandle compute_kernel_id,
    CoreCoord compute_with_storage_grid_size,
    const BatchNormOperation::operation_attributes_t& operation_attributes,
    const BatchNormOperation::tensor_args_t& tensor_args,
    BatchNormOperation::tensor_return_value_t& c,
    F handle_args) {
    const auto& [a, b, d, e, f, _] = tensor_args;
    const auto eps = operation_attributes.eps;
    const auto momentum = operation_attributes.momentum;

    const bool weight_has_value = e.has_value();
    const bool bias_has_value = f.has_value();

    const auto ashape = a.padded_shape();
    const auto bshape = b.padded_shape();
    const auto cshape = c.padded_shape();

    const auto [aN, aC, aHt, aWt] = extract_shape_dims(a);
    const auto [bN, bC, bHt, bWt] = extract_shape_dims(b);
    const auto [cN, cC, cHt, cWt] = extract_shape_dims(c);

    uint32_t num_output_tiles = c.volume() / c.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            handle_args(program, reader_kernel_id, core, std::array<uint32_t, 12>{0});
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, 14>{0});
            handle_args(program, compute_kernel_id, core, std::array<uint32_t, 3>{0});
            continue;
        }

        uint32_t cHtWt = cHt * cWt;
        class bfloat16 bfloat_scalar_eps(eps);
        uint32_t packed_scalar_eps = pack_two_bfloat16_into_uint32({bfloat_scalar_eps, bfloat_scalar_eps});
        class bfloat16 bfloat_scalar_momentum(momentum);
        uint32_t packed_scalar_momentum =
            pack_two_bfloat16_into_uint32({bfloat_scalar_momentum, bfloat_scalar_momentum});
        std::array reader_runtime_args = {
            packed_scalar_eps,
            packed_scalar_momentum,
            a.buffer()->address(),
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

        const auto weight_addr = weight_has_value ? e->buffer()->address() : 0;
        const auto bias_addr = bias_has_value ? f->buffer()->address() : 0;
        std::array writer_runtime_args = {
            b.buffer()->address(),  //  batch mean
            d.buffer()->address(),  //  batch var
            weight_addr,            // weight
            bias_addr,              // bias
            c.buffer()->address(),  // output
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

    const auto& [a, b, d, e, f, _] = tensor_args;

    auto program = CreateProgram();

    auto* device = a.device();

    const bool weight_has_value = e.has_value();
    const bool bias_has_value = f.has_value();

    auto a_data_format = datatype_to_dataformat_converter(a.get_dtype());
    auto b_data_format = datatype_to_dataformat_converter(b.get_dtype());
    auto c_data_format = datatype_to_dataformat_converter(output.get_dtype());
    auto d_data_format = datatype_to_dataformat_converter(d.get_dtype());
    auto e_data_format = weight_has_value ? datatype_to_dataformat_converter(e->get_dtype()) : DataFormat::Float16_b;
    auto f_data_format = bias_has_value ? datatype_to_dataformat_converter(f->get_dtype()) : DataFormat::Float16_b;

    uint32_t a_single_tile_size = tt_metal::detail::TileSize(a_data_format);
    uint32_t b_single_tile_size = tt_metal::detail::TileSize(b_data_format);
    uint32_t c_single_tile_size = tt_metal::detail::TileSize(c_data_format);
    uint32_t d_single_tile_size = tt_metal::detail::TileSize(d_data_format);
    uint32_t e_single_tile_size = tt_metal::detail::TileSize(e_data_format);
    uint32_t f_single_tile_size = tt_metal::detail::TileSize(f_data_format);

    uint32_t num_output_tiles = output.volume() / output.tensor_spec().tile().get_tile_hw();

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
    auto [a_cb, a_cb_handle] = create_cb(
        tt::CBIndex::c_0, program, all_device_cores, a_single_tile_size, num_tiles_per_cb, a_data_format);  // input
    auto [b_cb, b_cb_handle] = create_cb(
        tt::CBIndex::c_1,
        program,
        all_device_cores,
        b_single_tile_size,
        b_num_tiles_per_cb,
        b_data_format);  // batch_mean
    auto [c_cb, c_cb_handle] = create_cb(
        tt::CBIndex::c_2, program, all_device_cores, c_single_tile_size, num_tiles_per_cb, c_data_format);  // output
    auto [d_cb, d_cb_handle] = create_cb(
        tt::CBIndex::c_3,
        program,
        all_device_cores,
        d_single_tile_size,
        b_num_tiles_per_cb,
        d_data_format);  // batch_var
    auto [eps_cb, eps_cb_handle] = create_cb(
        tt::CBIndex::c_4, program, all_device_cores, d_single_tile_size, b_num_tiles_per_cb, d_data_format);  // eps
    auto [e_cb, e_cb_handle] = create_cb(
        tt::CBIndex::c_16, program, all_device_cores, e_single_tile_size, b_num_tiles_per_cb, e_data_format);  // weight
    auto [f_cb, f_cb_handle] = create_cb(
        tt::CBIndex::c_18, program, all_device_cores, f_single_tile_size, b_num_tiles_per_cb, f_data_format);  // bias
    auto [momentum_cb, momentum_cb_handle] = create_cb(
        tt::CBIndex::c_24,
        program,
        all_device_cores,
        d_single_tile_size,
        b_num_tiles_per_cb,
        d_data_format);  // momentum

    // Temporary buffers to store intermediate results
    auto [den_cb, den_cb_handle] = create_cb(
        tt::CBIndex::c_5,
        program,
        all_device_cores,
        a_single_tile_size,
        num_tiles_per_cb,
        a_data_format);  // to store 1/(sqrt(batch_var + eps))
    auto [num_cb, num_cb_handle] = create_cb(
        tt::CBIndex::c_6,
        program,
        all_device_cores,
        a_single_tile_size,
        num_tiles_per_cb,
        a_data_format);  // to store input - batch_mean
    auto [temp_1_cb, temp_1_cb_handle] =
        create_cb(tt::CBIndex::c_17, program, all_device_cores, a_single_tile_size, num_tiles_per_cb, a_data_format);

    auto a_is_dram = static_cast<uint32_t>(a.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    auto b_is_dram = static_cast<uint32_t>(b.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    auto c_is_dram = static_cast<uint32_t>(output.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    auto d_is_dram = static_cast<uint32_t>(d.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    const auto e_is_dram = weight_has_value and e->buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    const auto f_is_dram = bias_has_value and f->buffer()->buffer_type() == tt_metal::BufferType::DRAM;

    // READER KERNEL
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig({a_is_dram}));

    // WRITER KERNEL
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig(
            {b_is_dram,
             c_is_dram,
             d_is_dram,
             e_is_dram,
             f_is_dram,
             static_cast<uint32_t>(weight_has_value),
             static_cast<uint32_t>(bias_has_value),
             static_cast<uint32_t>(operation_attributes.training)}));

    // COMPUTE KERNEL
    bool fp32_dest_acc_en = c_data_format == tt::DataFormat::UInt32 || c_data_format == tt::DataFormat::Int32 ||
                            c_data_format == tt::DataFormat::Float32;
    std::vector<uint32_t> compute_kernel_args = {
        static_cast<uint32_t>(weight_has_value),
        static_cast<uint32_t>(bias_has_value),
        static_cast<uint32_t>(operation_attributes.training)};
    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp",
        all_device_cores,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_kernel_args});

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
    auto update_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
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

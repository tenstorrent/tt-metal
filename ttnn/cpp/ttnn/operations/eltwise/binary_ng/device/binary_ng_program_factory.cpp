// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_utils.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::binary_ng;

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const Tensor& x, bool is_5D_output) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    if (is_5D_output) {
        if (shape.rank() == 5) {
            return {shape[-5], shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
        } else {
            return {1, shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
        }
    } else {
        return {1, shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
    }
}

std::tuple<uint32_t, uint32_t> calculate_compute_kernel_args(
    SubtileBroadcastType broadcast_type, uint32_t start_tile_id, uint32_t HtWt, uint32_t Wt) {
    uint32_t start_t = start_tile_id % HtWt;
    uint32_t start_tw = start_t % Wt;

    switch (broadcast_type) {
        case SubtileBroadcastType::NONE:
        case SubtileBroadcastType::ROW_A:
        case SubtileBroadcastType::ROW_B: return {1, 0};
        case SubtileBroadcastType::SCALAR_A:
        case SubtileBroadcastType::SCALAR_B: return {HtWt, start_t};
        case SubtileBroadcastType::COL_A:
        case SubtileBroadcastType::ROW_B_COL_A:
        case SubtileBroadcastType::COL_B:
        case SubtileBroadcastType::ROW_A_COL_B: return {Wt, start_tw};
        default: __builtin_unreachable();  // GCC 12 doesn't compile even though we exhaustively match
    }
}

template <typename F>
void set_or_update_runtime_arguments(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    KernelHandle compute_kernel_id,
    CoreCoord compute_with_storage_grid_size,
    const BinaryNgDeviceOperation::operation_attributes_t& operation_attributes,
    const BinaryNgDeviceOperation::tensor_args_t& tensor_args,
    BinaryNgDeviceOperation::tensor_return_value_t& c,
    F handle_args) {
    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;

    const auto ashape = a.padded_shape();
    const auto bshape = b.has_value() ? b->padded_shape() : SimpleShape{1, 1};
    const auto cshape = c.padded_shape();  // check this shape
    const auto out_rank = c.logical_shape().rank();

    const auto [aD, aN, aC, aHt, aWt] = extract_shape_dims(a, out_rank > 4);
    const auto [bD, bN, bC, bHt, bWt] =
        b.has_value() ? extract_shape_dims(*b, out_rank > 4) : std::tuple{1u, 1u, 1u, 1u, 1u};
    const auto [cD, cN, cC, cHt, cWt] = extract_shape_dims(c, out_rank > 4);

    // std::cout << " aN, aC, aHt, aWt : " << aD << aN << aC << aHt << aWt << std::endl;
    // std::cout << " bN, bC, bHt, bWt : " << bD << bN << bC << bHt << bWt << std::endl;
    // std::cout << " cN, cC, cHt, cWt : " << cD << cN << cC << cHt << cWt << std::endl;

    uint32_t num_output_tiles = c.volume() / c.tensor_spec().tile().get_tile_hw();  // check here

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
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, 13>{0});
            handle_args(program, compute_kernel_id, core, std::array<uint32_t, 3>{0});
            continue;
        }

        uint32_t cHtWt = cHt * cWt;
        std::array reader_runtime_args = {
            a.buffer()->address(),
            start_tile_id,
            num_tiles_per_core,
            cHtWt,
            aHt * aWt * aC * aN * (aD > 1),
            aHt * aWt * aC * (aN > 1),
            aHt * aWt * (aC > 1),
            cD,
            cN,
            cC,
            cHt,
            cWt};
        handle_args(program, reader_kernel_id, core, reader_runtime_args);
        // std::cout << "reader_runtime_args : " << std::endl;
        // std::cout << start_tile_id << " start_tile_id" << std::endl;
        // std::cout << num_tiles_per_core << " num_tiles_per_core" << std::endl;
        // std::cout << cHtWt << " cHtWt " << std::endl;
        // std::cout << aHt * aWt * aC * (aN > 1) << " aHt * aWt * aC * (aN > 1)" << std::endl;
        // std::cout << aHt * aWt * (aC > 1) << " aHt * aWt * (aC > 1)" << std::endl;
        // std::cout << " cN, cC, cHt, cWt : " << cN << cC << cHt << cWt << std::endl;

        if (b.has_value()) {
            std::array writer_runtime_args = {
                b->buffer()->address(),
                c.buffer()->address(),
                start_tile_id,
                num_tiles_per_core,
                cHtWt,
                bHt * bWt * bC * bN * (bD > 1),
                bHt * bWt * bC * (bN > 1),
                bHt * bWt * (bC > 1),
                cD,
                cN,
                cC,
                cHt,
                cWt};
            handle_args(program, writer_kernel_id, core, writer_runtime_args);
            // std::cout << "writer_runtime_args : " << std::endl;
            // std::cout << start_tile_id << " start_tile_id" << std::endl;
            // std::cout << num_tiles_per_core << " num_tiles_per_core" << std::endl;
            // std::cout << cHtWt << " cHtWt " << std::endl;
            // std::cout << bHt * bWt * bC * (bN > 1) << " bHt * bWt * bC * (bN > 1)" << std::endl;
            // std::cout << bHt * bWt * (bC > 1) << " bHt * bWt * (bC > 1)" << std::endl;
            // std::cout << " cN, cC, cHt, cWt : " << cN << cC << cHt << cWt << std::endl;

            auto [freq, counter] =
                calculate_compute_kernel_args(operation_attributes.subtile_broadcast_type, start_tile_id, cHtWt, cWt);

            // std::cout << "calculate_compute_kernel_args : freq " << freq << std::endl;
            // std::cout << "calculate_compute_kernel_args : counter " << counter << std::endl;

            std::array compute_runtime_args = {num_tiles_per_core, freq, counter};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        } else {
            class bfloat16 bfloat_scalar(*operation_attributes.scalar);
            uint32_t packed_scalar = pack_two_bfloat16_into_uint32({bfloat_scalar, bfloat_scalar});
            std::array writer_runtime_args = {
                packed_scalar,
                c.buffer()->address(),
                start_tile_id,
                num_tiles_per_core,
                cHtWt,
                cD,
                cN,
                cC,
                cHt,
                cWt,
                0u,
                0u,
                0u};
            handle_args(program, writer_kernel_id, core, writer_runtime_args);

            std::array compute_runtime_args = {num_tiles_per_core, 0u, 0u};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        }
        std::cout << "start_tile_id pre :  " << start_tile_id << std::endl;
        start_tile_id += num_tiles_per_core;
        std::cout << "start_tile_id + num_tiles_per_core :  " << start_tile_id << std::endl;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::binary_ng {

// Implements c = a op b
BinaryNgDeviceOperation::ProgramFactory::cached_program_t BinaryNgDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t& c) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;

    auto program = CreateProgram();

    auto* device = a.device();

    auto a_data_format = datatype_to_dataformat_converter(a.get_dtype());
    auto b_data_format = b.has_value() ? datatype_to_dataformat_converter(b->get_dtype()) : DataFormat::Float16_b;
    auto c_data_format = datatype_to_dataformat_converter(c.get_dtype());

    uint32_t a_single_tile_size = tt_metal::detail::TileSize(a_data_format);
    uint32_t b_single_tile_size = tt_metal::detail::TileSize(b_data_format);
    uint32_t c_single_tile_size = tt_metal::detail::TileSize(c_data_format);

    uint32_t num_output_tiles = c.volume() / c.tensor_spec().tile().get_tile_hw();  // check here
    std::cout << " How many out tiles " << num_output_tiles << std::endl;
    // we parallelize the computation across the output tiles
    constexpr bool row_major = true;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    Buffer* a_buffer = a.buffer();
    Buffer* b_buffer = nullptr;
    Buffer* c_buffer = c.buffer();

    auto op_type = operation_attributes.binary_op_type;
    OpConfig op_config(op_type);
    auto compute_kernel_defines = op_config.as_defines();

    {
        ttnn::SmallVector<unary::UnaryOpType> lhs_activations = operation_attributes.lhs_activations;
        ttnn::SmallVector<unary::UnaryOpType> rhs_activations = operation_attributes.rhs_activations;
        ttnn::SmallVector<unary::UnaryOpType> post_activations = operation_attributes.post_activations;

        if (op_config.process_lhs.has_value()) {
            lhs_activations.push_back(*op_config.process_lhs);
        }

        if (op_config.process_rhs.has_value()) {
            rhs_activations.push_back(*op_config.process_rhs);
        }

        if (op_config.postprocess.has_value()) {
            post_activations.insert(post_activations.begin(), *op_config.postprocess);
        }

        add_activation_defines(compute_kernel_defines, lhs_activations, "LHS");
        add_activation_defines(compute_kernel_defines, rhs_activations, "RHS");

        if (lhs_activations.empty() and rhs_activations.empty() and post_activations.size() == 1 and
            post_activations[0] == unary::UnaryOpType::RELU) {
            compute_kernel_defines["PACK_RELU"] = "1";
            compute_kernel_defines["PROCESS_POST_ACTIVATIONS(i)"] = "";
            unary::utils::update_macro_defines(unary::UnaryOpType::RELU, compute_kernel_defines);
        } else {
            add_activation_defines(compute_kernel_defines, post_activations, "POST");
        }
    }

    bool op_has_exp =
        op_type == BinaryOpType::LOGADDEXP || op_type == BinaryOpType::LDEXP || op_type == BinaryOpType::LOGADDEXP2;

    // How many tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    auto [a_cb, a_cb_handle] =
        create_cb(tt::CBIndex::c_0, program, all_device_cores, a_single_tile_size, num_tiles_per_cb, a_data_format);

    if (not compute_kernel_defines["PROCESS_LHS_ACTIVATIONS(i)"].empty()) {
        auto a_intermediate_format = op_has_exp ? tt::DataFormat::Float16_b : a_data_format;
        uint32_t a_intermediate_single_tile_size = tt_metal::detail::TileSize(a_intermediate_format);
        auto [a_cb_interim, a_cb_interim_handle] = create_cb(
            tt::CBIndex::c_3, program, all_device_cores, a_intermediate_single_tile_size, 1, a_intermediate_format);
    }

    auto [c_cb, c_cb_handle] =
        create_cb(tt::CBIndex::c_2, program, all_device_cores, c_single_tile_size, num_tiles_per_cb, c_data_format);

    // If b is a scalar, we only need one tile in the CB
    uint32_t b_num_tiles_per_cb = b_buffer != nullptr ? num_tiles_per_cb : 1;
    auto [b_cb, b_cb_handle] =
        create_cb(tt::CBIndex::c_1, program, all_device_cores, b_single_tile_size, b_num_tiles_per_cb, b_data_format);

    if (not compute_kernel_defines["PROCESS_RHS_ACTIVATIONS(i)"].empty()) {
        auto b_intermediate_format = op_has_exp ? tt::DataFormat::Float16_b : b_data_format;
        uint32_t b_intermediate_single_tile_size = tt_metal::detail::TileSize(b_intermediate_format);
        auto [b_cb_interim, b_cb_interim_handle] = create_cb(
            tt::CBIndex::c_4, program, all_device_cores, b_intermediate_single_tile_size, 1, b_intermediate_format);
    }

    auto a_is_dram = static_cast<uint32_t>(a_buffer->buffer_type() == tt_metal::BufferType::DRAM);
    bool b_is_dram = false;
    auto c_is_dram = static_cast<uint32_t>(c_buffer->buffer_type() == tt_metal::BufferType::DRAM);

    auto kernel_config = CMAKE_UNIQUE_NAMESPACE::BinaryNgKernelConfig(operation_attributes.subtile_broadcast_type);

    // READER KERNEL
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.reader_kernel),
        all_device_cores,
        tt_metal::ReaderDataMovementConfig({a_is_dram}));
    std::cout << "reader kernel " << get_kernel_file_path(kernel_config.reader_kernel) << std::endl;
    // WRITER KERNEL
    auto writer_kernel = CMAKE_UNIQUE_NAMESPACE::KernelName::WriterScalar;
    auto compute_kernel = CMAKE_UNIQUE_NAMESPACE::KernelName::ComputeScalar;
    if (b.has_value()) {
        b_buffer = b->buffer();
        b_is_dram = static_cast<uint32_t>(b_buffer->buffer_type() == tt_metal::BufferType::DRAM);
        writer_kernel = kernel_config.writer_kernel;
        compute_kernel = kernel_config.compute_kernel;
    }

    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(writer_kernel),
        all_device_cores,
        tt_metal::WriterDataMovementConfig({b_is_dram, c_is_dram}));
    std::cout << "writer kernel " << get_kernel_file_path(kernel_config.writer_kernel) << std::endl;
    // COMPUTE KERNEL
    bool fp32_dest_acc_en = c_data_format == tt::DataFormat::UInt32 || c_data_format == tt::DataFormat::Int32 ||
                            c_data_format == tt::DataFormat::Float32;

    // Compute kernel needs to know which op it's going to perform
    // This has to be passed as a compile-time argument
    // For now we're just going to do addition
    compute_kernel_defines["BCAST_INPUT"] = kernel_config.bcast_input_str();
    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(compute_kernel),
        all_device_cores,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .defines = compute_kernel_defines});
    std::cout << "compute_kernel " << get_kernel_file_path(compute_kernel) << std::endl;
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
        c,
        set_runtime_args);

    return {
        std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, compute_with_storage_grid_size}};
}

void BinaryNgDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& c) {
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
        c,
        update_args);
}

}  // namespace ttnn::operations::binary_ng

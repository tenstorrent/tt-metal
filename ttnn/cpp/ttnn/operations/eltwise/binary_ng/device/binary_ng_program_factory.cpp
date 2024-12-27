// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_utils.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/cb_utils.hpp"

// template <std::size_t N>
// void print_array(const std::array<uint32_t, N>& arr) {
//     for (const auto& elem : arr) {
//         std::cout << elem << " ";
//     }
//     std::cout << std::endl;
// }
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::binary_ng;

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

std::tuple<uint32_t, uint32_t> calculate_compute_kernel_args(
    SubtileBroadcastType broadcast_type, uint32_t start_tile_id, uint32_t HtWt, uint32_t Wt) {
    uint32_t start_t = start_tile_id % HtWt;
    uint32_t start_tw = start_t % Wt;
    std::cout << "calculate_compute_kernel_args : start tile " << start_t << std::endl;
    std::cout << "calculate_compute_kernel_args : start tilew " << start_tw << std::endl;
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

    const auto ashape = a.get_padded_shape();
    const auto bshape = b.has_value() ? b->get_padded_shape() : SimpleShape{1, 1};
    const auto cshape = c.get_padded_shape();

    const auto [aN, aC, aHt, aWt] = extract_shape_dims(a);
    std::cout << " aN, aC, aHt, aWt : " << aN << aC << aHt << aWt << std::endl;
    const auto [bN, bC, bHt, bWt] = b.has_value() ? extract_shape_dims(*b) : std::tuple{1u, 1u, 1u, 1u};
    std::cout << " bN, bC, bHt, bWt : " << bN << bC << bHt << bWt << std::endl;
    const auto [cN, cC, cHt, cWt] = extract_shape_dims(c);
    std::cout << " cN, cC, cHt, cWt : " << cN << cC << cHt << cWt << std::endl;

    uint32_t num_output_tiles = c.volume() / c.tensor_spec().tile().get_tile_hw();

    std::cout << "num_output_tiles : c.volume() / c.tensor_spec().tile().get_tile_hw(); " << num_output_tiles
              << std::endl;
    std::cout << "num_output_tiles : c.volume()  " << c.volume() << std::endl;
    std::cout << "num_output_tiles : c.tensor_spec().tile().get_tile_hw(); " << c.tensor_spec().tile().get_tile_hw()
              << std::endl;

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
            handle_args(program, reader_kernel_id, core, std::array<uint32_t, 10>{0});
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, 11>{0});
            handle_args(program, compute_kernel_id, core, std::array<uint32_t, 3>{0});
            continue;
        }

        uint32_t cHtWt = cHt * cWt;
        std::cout << "cHtWt : " << cHtWt << std::endl;
        std::array reader_runtime_args = {
            a.buffer()->address(),
            start_tile_id,              // start_tile_id
            num_tiles_per_core,         // num_tiles
            cHtWt,                      // cHtWt ? output HtWt
            aHt * aWt * aC * (aN > 1),  // n-stride
            aHt * aWt * (aC > 1),       // c-stride
            cN,
            cC,
            cHt,
            cWt};
        handle_args(program, reader_kernel_id, core, reader_runtime_args);
        std::cout << "reader_runtime_args : " << std::endl;
        std::cout << start_tile_id << " start_tile_id" << std::endl;
        std::cout << num_tiles_per_core << " num_tiles_per_core" << std::endl;
        std::cout << cHtWt << " cHtWt " << std::endl;
        std::cout << aHt * aWt * aC * (aN > 1) << " aHt * aWt * aC * (aN > 1)" << std::endl;
        std::cout << aHt * aWt * (aC > 1) << " aHt * aWt * (aC > 1)" << std::endl;
        std::cout << " cN, cC, cHt, cWt : " << cN << cC << cHt << cWt << std::endl;

        if (b.has_value()) {
            std::array writer_runtime_args = {
                b->buffer()->address(),
                c.buffer()->address(),
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
            std::cout << "writer_runtime_args : " << std::endl;
            std::cout << start_tile_id << " start_tile_id" << std::endl;
            std::cout << num_tiles_per_core << " num_tiles_per_core" << std::endl;
            std::cout << cHtWt << " cHtWt " << std::endl;
            std::cout << bHt * bWt * bC * (bN > 1) << " bHt * bWt * bC * (bN > 1)" << std::endl;
            std::cout << bHt * bWt * (bC > 1) << " bHt * bWt * (bC > 1)" << std::endl;
            std::cout << " cN, cC, cHt, cWt : " << cN << cC << cHt << cWt << std::endl;

            auto [freq, counter] =
                calculate_compute_kernel_args(operation_attributes.subtile_broadcast_type, start_tile_id, cHtWt, cWt);
            std::cout << "calculate_compute_kernel_args : freq " << freq << std::endl;
            std::cout << "calculate_compute_kernel_args : counter " << counter << std::endl;

            std::array compute_runtime_args = {num_tiles_per_core, freq, counter};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        } else {
            const auto scalar = *operation_attributes.scalar;
            class bfloat16 bfloat_scalar(scalar);
            const auto packed_scalar = a.get_dtype() == DataType::FLOAT32 ? std::bit_cast<uint32_t>(scalar)
                                       : a.get_dtype() == DataType::INT32
                                           ? std::bit_cast<uint32_t>(static_cast<int32_t>(scalar))
                                           : pack_two_bfloat16_into_uint32({bfloat_scalar, bfloat_scalar});
            std::array writer_runtime_args = {
                packed_scalar,
                c.buffer()->address(),
                start_tile_id,
                num_tiles_per_core,
                cHtWt,
                cN,
                cC,
                cHt,
                cWt,
                0u,
                0u};
            handle_args(program, writer_kernel_id, core, writer_runtime_args);

            std::array compute_runtime_args = {num_tiles_per_core, 0u, 0u};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
            std::cout << "compute_runtime_args : " << std::endl;
            std::cout << num_tiles_per_core << " num_tiles_per_core" << std::endl;
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

    tt::DataFormat b_intermediate_format = b_data_format;

    uint32_t a_single_tile_size = tt_metal::detail::TileSize(a_data_format);
    uint32_t b_single_tile_size = tt_metal::detail::TileSize(b_data_format);
    uint32_t c_single_tile_size = tt_metal::detail::TileSize(c_data_format);

    uint32_t num_output_tiles = c.volume() / c.tensor_spec().tile().get_tile_hw();

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
    auto compute_kernel_defines = OpConfig(op_type).as_defines();
    bool op_has_exp =
        op_type == BinaryOpType::LOGADDEXP || op_type == BinaryOpType::LDEXP || op_type == BinaryOpType::LOGADDEXP2;

    // How many tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    std::cout << " How many tiles to store per input CB (double buffer) num_tiles_per_cb " << num_tiles_per_cb
              << std::endl;

    auto [a_cb, a_cb_handle] =
        create_cb(tt::CBIndex::c_0, program, all_device_cores, a_single_tile_size, num_tiles_per_cb, a_data_format);

    if (compute_kernel_defines.find("PREPROCESS_A_INIT") != compute_kernel_defines.end()) {
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

    if (compute_kernel_defines.find("PREPROCESS_B_INIT") != compute_kernel_defines.end()) {
        auto b_intermediate_format = op_has_exp ? tt::DataFormat::Float16_b : b_data_format;
        uint32_t b_intermediate_single_tile_size = tt_metal::detail::TileSize(b_intermediate_format);
        auto [b_cb_interim, b_cb_interim_handle] = create_cb(
            tt::CBIndex::c_4, program, all_device_cores, b_intermediate_single_tile_size, 1, b_intermediate_format);
    }

    auto a_is_dram = static_cast<uint32_t>(a_buffer->buffer_type() == tt_metal::BufferType::DRAM);
    bool b_is_dram = false;
    auto c_is_dram = static_cast<uint32_t>(c_buffer->buffer_type() == tt_metal::BufferType::DRAM);

    auto kernel_config = CMAKE_UNIQUE_NAMESPACE::BinaryNgKernelConfig(operation_attributes.subtile_broadcast_type);

    std::map<std::string, std::string> reader_defines;
    const auto ashape = a.get_logical_shape();
    bool is_scalarA = (ashape.rank() >= 2) && ((ashape[-1] == 1) || (ashape[-2] == 1));
    if (is_scalarA) {
        reader_defines["BF16_SCALARB"] = "1";
    }

    // READER KERNEL
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.reader_kernel),
        all_device_cores,
        tt_metal::ReaderDataMovementConfig({a_is_dram}, reader_defines));

    std::cout << "reader kernel " << get_kernel_file_path(kernel_config.reader_kernel) << std::endl;
    // WRITER KERNEL
    // by default the writer and compute kernel is set for B as scalar value
    auto writer_kernel = CMAKE_UNIQUE_NAMESPACE::KernelName::WriterScalar;
    auto compute_kernel = CMAKE_UNIQUE_NAMESPACE::KernelName::ComputeScalar;
    std::map<std::string, std::string> writer_defines;
    if (b.has_value()) {
        b_buffer = b->buffer();
        b_is_dram = static_cast<uint32_t>(b_buffer->buffer_type() == tt_metal::BufferType::DRAM);
        writer_kernel = kernel_config.writer_kernel;
        compute_kernel = kernel_config.compute_kernel;
        const auto bshape = b->get_logical_shape();
        bool is_scalarB = (bshape.rank() >= 2) && ((bshape[-1] == 1) || (bshape[-2] == 1));
        if (is_scalarB) {
            writer_defines["BF16_SCALARB"] = "1";
        }
    }
    std::cout << "writer kernel " << get_kernel_file_path(writer_kernel) << std::endl;
    if (!b.has_value()) {
        writer_defines["BF16_SCALARB"] = "1";
    }

    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(writer_kernel),
        all_device_cores,
        tt_metal::WriterDataMovementConfig({b_is_dram, c_is_dram}, writer_defines));

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

    std::cout << "compute_kernel  " << get_kernel_file_path(compute_kernel) << std::endl;

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

// Implements c = a op b using binary SFPU
BinaryNgDeviceOperation::SfpuProgramFactory::cached_program_t BinaryNgDeviceOperation::SfpuProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t& c) {
    using namespace tt;
    using namespace tt::tt_metal;
    using ttnn::operations::unary::UnaryWithParam;
    std::cout << " Enter sfpu pgm factory" << std::endl;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;

    // std::vector<UnaryWithParam> fused_activations =
    //     operation_attributes.activations.value_or(std::vector<UnaryWithParam>{});
    std::vector<UnaryWithParam> fused_activations = std::vector<UnaryWithParam>{};

    auto program = CreateProgram();

    auto* device = a.device();

    auto a_data_format = datatype_to_dataformat_converter(a.get_dtype());
    auto b_data_format = b.has_value() ? datatype_to_dataformat_converter(b->get_dtype()) : a_data_format;
    auto c_data_format = datatype_to_dataformat_converter(c.get_dtype());

    uint32_t a_single_tile_size = tt_metal::detail::TileSize(a_data_format);
    uint32_t b_single_tile_size = tt_metal::detail::TileSize(b_data_format);
    uint32_t c_single_tile_size = tt_metal::detail::TileSize(c_data_format);

    uint32_t num_output_tiles = c.volume() / c.tensor_spec().tile().get_tile_hw();

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
    DataType b_dtype = b.has_value() ? b->get_dtype() : a.get_dtype();
    auto compute_kernel_defines = get_defines_fp32(op_type, a.get_dtype(), b_dtype, fused_activations, std::nullopt);
    // no operation_attributes.input_tensor_a_activation yet
    bool op_has_exp =
        op_type == BinaryOpType::LOGADDEXP || op_type == BinaryOpType::LDEXP || op_type == BinaryOpType::LOGADDEXP2;

    // How many tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    std::cout << " How many tiles to store per input CB (double buffer) num_tiles_per_cb " << num_tiles_per_cb
              << std::endl;

    auto [a_cb, a_cb_handle] =
        create_cb(tt::CBIndex::c_0, program, all_device_cores, a_single_tile_size, num_tiles_per_cb, a_data_format);

    if (compute_kernel_defines.find("SFPU_OP_INIT_PRE_IN0_0") != compute_kernel_defines.end()) {
        auto a_intermediate_format = a_data_format;
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

    if (compute_kernel_defines.find("SFPU_OP_INIT_PRE_IN1_0") != compute_kernel_defines.end()) {
        auto b_intermediate_format = b.has_value() ? b_data_format : a_data_format;
        uint32_t b_intermediate_single_tile_size = tt_metal::detail::TileSize(b_intermediate_format);
        auto [b_cb_interim, b_cb_interim_handle] = create_cb(
            tt::CBIndex::c_4, program, all_device_cores, b_intermediate_single_tile_size, 1, b_intermediate_format);
    }

    auto a_is_dram = static_cast<uint32_t>(a_buffer->buffer_type() == tt_metal::BufferType::DRAM);
    bool b_is_dram = false;
    auto c_is_dram = static_cast<uint32_t>(c_buffer->buffer_type() == tt_metal::BufferType::DRAM);

    auto kernel_config = CMAKE_UNIQUE_NAMESPACE::BinaryNgKernelConfig(operation_attributes.subtile_broadcast_type);

    const auto ashape = a.get_logical_shape();
    tt::log_info(tt::LogOp, "******** a logical shape: {}", ashape);
    std::map<std::string, std::string> reader_defines;
    bool is_scalarA = (ashape.rank() >= 2) && ((ashape[-1] == 1) || (ashape[-2] == 1));
    if (is_scalarA && a.get_dtype() == DataType::FLOAT32) {
        reader_defines["F32_SCALARB"] = "1";
    } else if (is_scalarA && a.get_dtype() == DataType::INT32) {
        reader_defines["INT32_SCALARB"] = "1";
    } else if (is_scalarA) {
        reader_defines["BF16_SCALARB"] = "1";
    }
    for (const auto& pair : reader_defines) {
        std::cout << "reader sf" << pair.first << ": " << pair.second << std::endl;
    }

    // READER KERNEL
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        get_sfpu_kernel_file_path(kernel_config.reader_kernel),
        all_device_cores,
        tt_metal::ReaderDataMovementConfig({a_is_dram}, reader_defines));

    std::cout << "reader kernel " << get_sfpu_kernel_file_path(kernel_config.reader_kernel) << std::endl;

    // WRITER KERNEL
    std::map<std::string, std::string> writer_defines;
    auto writer_kernel = CMAKE_UNIQUE_NAMESPACE::KernelName::WriterScalar;
    auto compute_kernel = CMAKE_UNIQUE_NAMESPACE::KernelName::ComputeScalar;
    if (b.has_value()) {
        b_buffer = b->buffer();
        b_is_dram = static_cast<uint32_t>(b_buffer->buffer_type() == tt_metal::BufferType::DRAM);
        writer_kernel = kernel_config.writer_kernel;
        compute_kernel = kernel_config.compute_kernel;

        const auto bshape = b->get_logical_shape();
        tt::log_info(tt::LogOp, "******** b logical shape: {}", bshape);
        bool is_scalarB = (bshape.rank() >= 2) && ((bshape[-1] == 1) || (bshape[-2] == 1));
        if (is_scalarB && b->get_dtype() == DataType::FLOAT32) {
            writer_defines["F32_SCALARB"] = "1";
        } else if (is_scalarB && b->get_dtype() == DataType::INT32) {
            writer_defines["INT32_SCALARB"] = "1";
        } else if (is_scalarB) {
            writer_defines["BF16_SCALARB"] = "1";
        }
        for (const auto& pair : writer_defines) {
            std::cout << "writer_defines sf" << pair.first << ": " << pair.second << std::endl;
        }
    }
    if (!b.has_value() && a.get_dtype() == DataType::FLOAT32) {
        std::cout << " scalar float defines" << std::endl;
        writer_defines["F32_SCALARB"] = "1";
    } else if (!b.has_value() && a.get_dtype() == DataType::INT32) {
        writer_defines["INT32_SCALARB"] = "1";
    } else if (!b.has_value()) {
        writer_defines["BF16_SCALARB"] = "1";
    }
    for (const auto& pair : writer_defines) {
        std::cout << "writer sf" << pair.first << ": " << pair.second << std::endl;
    }
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        get_sfpu_kernel_file_path(writer_kernel),
        all_device_cores,
        tt_metal::WriterDataMovementConfig({b_is_dram, c_is_dram}, writer_defines));

    std::cout << "writer kernel " << get_sfpu_kernel_file_path(writer_kernel) << std::endl;

    // COMPUTE KERNEL
    bool fp32_dest_acc_en = c_data_format == tt::DataFormat::UInt32 || c_data_format == tt::DataFormat::Int32 ||
                            c_data_format == tt::DataFormat::Float32;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t src0interim_cb_index = tt::CBIndex::c_3;
    uint32_t src1interim_cb_index = tt::CBIndex::c_4;

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[src1_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[src0interim_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[src1interim_cb_index] = UnpackToDestMode::UnpackToDestFp32;

    // Compute kernel needs to know which op it's going to perform
    // This has to be passed as a compile-time argument
    // For now we're just going to do addition
    compute_kernel_defines["BCAST_INPUT"] = kernel_config.bcast_input_str();
    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        get_sfpu_kernel_file_path(compute_kernel),
        all_device_cores,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .defines = compute_kernel_defines});

    std::cout << "compute_kernel  " << get_sfpu_kernel_file_path(compute_kernel) << std::endl;

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

void BinaryNgDeviceOperation::SfpuProgramFactory::override_runtime_arguments(
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

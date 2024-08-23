// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/device_operation.hpp"



namespace ttnn::operations::binary {

static const BcastOpMath binary_op_type_to_bcast_op_math(const BinaryOpType binary_op_type) {
    switch (binary_op_type) {
        case BinaryOpType::ADD: return BcastOpMath::ADD;
        case BinaryOpType::SUB: return BcastOpMath::SUB;
        case BinaryOpType::MUL: return BcastOpMath::MUL;
        default: TT_THROW("BinaryOpType cannot be mapped to BcastOpMath");
    }
}

BinaryDeviceOperation::BroadcastWidthMultiCore::cached_program_t BinaryDeviceOperation::BroadcastWidthMultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& output = tensor_return_value;
    auto bcast_math = binary_op_type_to_bcast_op_math(operation_attributes.binary_op_type);

    const auto ashape = a.get_legacy_shape();
    const auto bshape = b.get_legacy_shape();
    uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    uint32_t H = ashape[-2];
    uint32_t W = ashape[-1];
    uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    uint32_t bH = bshape[-2];
    uint32_t bW = bshape[-1];
    uint32_t NC = N * C;
    uint32_t HW = H * W;

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC * Ht * Wt;
    uint32_t num_btensor_tiles = NC * bH * bW / TILE_HW;

    uint32_t bnc1 = (bN * bC == 1) ? 1 : 0;

    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::Device* device = a.device();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    constexpr bool row_major = false;
    auto [num_cores, all_cores, core_group_1, core_group_2, Wt_per_core_group_1, Wt_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, Wt, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    auto src0_buffer = a.buffer();
    auto src1_buffer = b.buffer();
    auto dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;

    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_device_cores, src0_cb_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src1_single_tile_size, {{src1_cb_index, src1_cb_data_format}})
            .set_page_size(src1_cb_index, src1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_device_cores, src1_cb_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_device_cores, output_cb_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

    KernelHandle binary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/reader_bcast_w_interleaved_input_cols_partitioned.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/writer_unary_interleaved_input_cols_batched.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::map<std::string, std::string> bcast_defines = bcast_op_utils::get_defines(BcastOpDim::W, bcast_math);
    auto bcast_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_w.cpp",
        all_device_cores,
        tt_metal::ComputeConfig{.compile_args = {}, .defines = bcast_defines});

    for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t Wt_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            Wt_per_core = Wt_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            Wt_per_core = Wt_per_core_group_2;
        } else {
            tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, std::vector<uint32_t>(16, 0));
            tt_metal::SetRuntimeArgs(program, bcast_kernel_id, core, std::vector<uint32_t>(3, 0));
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, std::vector<uint32_t>(9, 0));
            continue;
        }
        uint32_t num_tensor_tiles_per_core = NC * Ht * Wt_per_core;
        uint32_t Wt_skip = Wt - Wt_per_core;

        tt_metal::SetRuntimeArgs(
            program,
            binary_reader_kernel_id,
            core,
            {
                a.buffer()->address(),      // 0
                0,                          // 1
                0,                          // 2
                num_tensor_tiles_per_core,  // 3
                b.buffer()->address(),      // 4
                0,                          // 5
                0,                          // 6
                num_btensor_tiles,          // 7
                num_tensor_tiles_per_core,  // 8
                NC,                         // 9
                Ht,                         // 10
                Wt_per_core,                // 11
                bnc1,                       // 12
                num_Wtiles_read,            // 13
                Ht * Wt,                    // 14
                Wt_skip,                    // 15
            });

        tt_metal::SetRuntimeArgs(
            program,
            bcast_kernel_id,
            core,
            {
                NC,          // B
                Ht,          // Ht
                Wt_per_core  // Wt
            });

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                0,
                0,
                Ht,
                Wt_per_core,
                num_Wtiles_read,
                Wt_skip,
                NC,
                Ht * Wt,
            });
        num_Wtiles_read += Wt_per_core;
    }

    return {
        std::move(program),
        {binary_reader_kernel_id, unary_writer_kernel_id, bcast_kernel_id, compute_with_storage_grid_size}};
}

void BinaryDeviceOperation::BroadcastWidthMultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    auto& binary_reader_kernel_id = cached_program.shared_variables.binary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& bcast_kernel_id = cached_program.shared_variables.bcast_kernel_id;
    auto& compute_with_storage_grid_size = cached_program.shared_variables.compute_with_storage_grid_size;

    auto& program = cached_program.program;

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;

    auto src_dram_buffer_a = input_tensor_a.buffer();
    auto src_dram_buffer_b = input_tensor_b.buffer();

    auto dst_dram_buffer = output_tensor.buffer();

    const auto ashape = input_tensor_a.get_legacy_shape();
    const auto bshape = input_tensor_b.get_legacy_shape();
    uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    uint32_t H = ashape[-2];
    uint32_t W = ashape[-1];
    uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    uint32_t bH = bshape[-2];
    uint32_t bW = bshape[-1];
    uint32_t NC = N * C;
    uint32_t HW = H * W;

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC * Ht * Wt;
    uint32_t num_btensor_tiles = NC * bH * bW / TILE_HW;

    uint32_t bnc1 = (bN * bC == 1) ? 1 : 0;

    constexpr bool row_major = false;
    auto [num_cores, all_cores, core_group_1, core_group_2, Wt_per_core_group_1, Wt_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, Wt, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    auto& cached_reader_args = GetRuntimeArgs(program, binary_reader_kernel_id);
    auto& cached_eltwise_args = GetRuntimeArgs(program, bcast_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, unary_writer_kernel_id);

    for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t Wt_per_core;

        auto& binary_reader_args = cached_reader_args.at(core.x).at(core.y);
        auto& bcast_kernel_args = cached_eltwise_args.at(core.x).at(core.y);
        auto& unary_writer_args = cached_writer_args.at(core.x).at(core.y);

        if (core_group_1.core_coord_in_core_ranges(core)) {
            Wt_per_core = Wt_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            Wt_per_core = Wt_per_core_group_2;
        } else {
            binary_reader_args[3] = 0;
            binary_reader_args[7] = 0;
            binary_reader_args[8] = 0;
            binary_reader_args[9] = 0;
            binary_reader_args[10] = 0;
            binary_reader_args[11] = 0;
            binary_reader_args[12] = 0;
            binary_reader_args[13] = 0;
            binary_reader_args[14] = 0;
            binary_reader_args[15] = 0;

            bcast_kernel_args[0] = 0;
            bcast_kernel_args[1] = 0;
            bcast_kernel_args[2] = 0;

            unary_writer_args[3] = 0;
            unary_writer_args[4] = 0;
            unary_writer_args[5] = 0;
            unary_writer_args[7] = 0;
            unary_writer_args[8] = 0;
            continue;
        }
        uint32_t num_tensor_tiles_per_core = NC * Ht * Wt_per_core;
        uint32_t Wt_skip = Wt - Wt_per_core;

        binary_reader_args[0] = src_dram_buffer_a->address();
        // binary_reader_args[1] = 0;
        // binary_reader_args[2] = 0;
        binary_reader_args[3] = num_tensor_tiles_per_core;
        binary_reader_args[4] = src_dram_buffer_b->address();
        // binary_reader_args[5] = 0;
        // binary_reader_args[6] = 0;
        binary_reader_args[7] = num_btensor_tiles;
        binary_reader_args[8] = num_tensor_tiles_per_core;
        binary_reader_args[9] = NC;
        binary_reader_args[10] = Ht;
        binary_reader_args[11] = Wt_per_core;
        binary_reader_args[12] = bnc1;
        binary_reader_args[13] = num_Wtiles_read;
        binary_reader_args[14] = Ht * Wt;
        binary_reader_args[15] = Wt_skip;

        bcast_kernel_args[0] = NC;
        bcast_kernel_args[1] = Ht;
        bcast_kernel_args[2] = Wt_per_core;

        unary_writer_args[0] = dst_dram_buffer->address();
        // unary_writer_args[1] = 0;
        // unary_writer_args[2] = 0;
        unary_writer_args[3] = Ht;
        unary_writer_args[4] = Wt_per_core;
        unary_writer_args[5] = num_Wtiles_read;
        unary_writer_args[6] = Wt_skip;
        unary_writer_args[7] = NC;
        unary_writer_args[8] = Ht * Wt;

        num_Wtiles_read += Wt_per_core;
    }
}

}  // namespace ttnn::operations::binary

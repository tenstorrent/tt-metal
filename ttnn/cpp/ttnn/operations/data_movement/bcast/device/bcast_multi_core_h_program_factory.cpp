// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_multi_core_h_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"

namespace ttnn::operations::data_movement::bcast::program {

using namespace tt::tt_metal;
using namespace tt::constants;

BcastMultiCoreHProgramFactory::cached_program_t BcastMultiCoreHProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value) {
    const Tensor& a = tensor_args.input_a;
    const Tensor& b = tensor_args.input_b;
    Tensor& output = tensor_return_value;

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    const uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    const uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    const uint32_t H = ashape[-2];
    const uint32_t W = ashape[-1];
    const uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    const uint32_t bH = bshape[-2];
    const uint32_t bW = bshape[-1];
    const uint32_t NC = N * C;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;

    const uint32_t num_btensor_tiles = NC * bH * bW / TILE_HW;
    const uint32_t bnc1 = (bN * bC == 1) ? 1 : 0;

    Program program = CreateProgram();

    IDevice* device = a.device();

    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat src1_cb_data_format = datatype_to_dataformat_converter(b.dtype());
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    const uint32_t src1_single_tile_size = tt::tile_size(src1_cb_data_format);
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const CoreRange all_device_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    const auto [num_cores, all_cores, core_group_1, core_group_2, Ht_per_core_group_1, Ht_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, Ht);

    Buffer* src0_buffer = a.buffer();
    Buffer* src1_buffer = b.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t src0_cb_index = 0;
    const uint32_t num_input_tiles = 2;

    CircularBufferConfig src0_cb_config =
        CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    CreateCircularBuffer(program, all_device_cores, src0_cb_config);

    const uint32_t src1_cb_index = 1;
    CircularBufferConfig src1_cb_config =
        CircularBufferConfig(num_input_tiles * src1_single_tile_size, {{src1_cb_index, src1_cb_data_format}})
            .set_page_size(src1_cb_index, src1_single_tile_size);
    CreateCircularBuffer(program, all_device_cores, src1_cb_config);

    const uint32_t output_cb_index = tt::CBIndex::c_16;
    const uint32_t num_output_tiles = 2;
    CircularBufferConfig output_cb_config =
        CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    CreateCircularBuffer(program, all_device_cores, output_cb_config);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    const KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/"
        "reader_bcast_h_interleaved_input_rows_partitioned.cpp",
        all_device_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    const KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/"
        "writer_unary_interleaved_input_cols_batched.cpp",
        all_device_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    const std::map<std::string, std::string> bcast_defines =
        bcast_op_utils::get_defines(BcastOpDim::H, operation_attributes.math_op);
    const KernelHandle bcast_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_h.cpp",
        all_device_cores,
        ComputeConfig{.compile_args = {}, .defines = bcast_defines});

    for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_y * num_cores_x; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t Ht_per_core;
        if (core_group_1.contains(core)) {
            Ht_per_core = Ht_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            Ht_per_core = Ht_per_core_group_2;
        } else {
            constexpr std::array<uint32_t, 15> binary_reader_kernel_args{0};
            constexpr std::array<uint32_t, 3> bcast_kernel_args{0};
            constexpr std::array<uint32_t, 9> unary_writer_kernel_args{0};

            SetRuntimeArgs(program, binary_reader_kernel_id, core, binary_reader_kernel_args);
            SetRuntimeArgs(program, bcast_kernel_id, core, bcast_kernel_args);
            SetRuntimeArgs(program, unary_writer_kernel_id, core, unary_writer_kernel_args);
            continue;
        }
        const uint32_t num_tensor_tiles_per_core = NC * Ht_per_core * Wt;

        SetRuntimeArgs(
            program,
            binary_reader_kernel_id,
            core,
            {
                src0_buffer->address(),     // 0
                0,                          // 1
                0,                          // 2
                num_tensor_tiles_per_core,  // 3
                src1_buffer->address(),     // 4
                0,                          // 5
                0,                          // 6
                num_btensor_tiles,          // 7
                num_tensor_tiles_per_core,  // 8
                NC,                         // 9
                Ht_per_core,                // 10
                Wt,                         // 11
                bnc1,                       // 12
                num_Wtiles_read,            // 13
                Ht * Wt,                    // 14
            });

        SetRuntimeArgs(
            program,
            bcast_kernel_id,
            core,
            {
                NC,           // B
                Ht_per_core,  // Ht
                Wt            // Wt
            });

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                0,
                0,
                Ht_per_core,
                Wt,
                num_Wtiles_read,
                0,
                NC,
                Ht * Wt,
            });

        num_Wtiles_read += Ht_per_core * Wt;
    }

    return cached_program_t{
        std::move(program),
        {binary_reader_kernel_id, unary_writer_kernel_id, bcast_kernel_id, compute_with_storage_grid_size}};
}

void BcastMultiCoreHProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    const uint32_t num_cores_x = cached_program.shared_variables.compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = cached_program.shared_variables.compute_with_storage_grid_size.y;

    Buffer* src_dram_buffer_a = tensor_args.input_a.buffer();
    Buffer* src_dram_buffer_b = tensor_args.input_b.buffer();
    Buffer* dst_dram_buffer = tensor_return_value.buffer();

    const auto ashape = tensor_args.input_a.padded_shape();
    const auto bshape = tensor_args.input_b.padded_shape();
    const uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    const uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    const uint32_t H = ashape[-2];
    const uint32_t W = ashape[-1];
    const uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    const uint32_t bH = bshape[-2];
    const uint32_t bW = bshape[-1];
    const uint32_t NC = N * C;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;

    const uint32_t num_btensor_tiles = NC * bH * bW / TILE_HW;
    const uint32_t bnc1 = (bN * bC == 1) ? 1 : 0;

    const auto [num_cores, all_cores, core_group_1, core_group_2, Ht_per_core_group_1, Ht_per_core_group_2] =
        split_work_to_cores(cached_program.shared_variables.compute_with_storage_grid_size, Ht);

    auto& cached_reader_args =
        GetRuntimeArgs(cached_program.program, cached_program.shared_variables.binary_reader_kernel_id);
    auto& cached_eltwise_args = GetRuntimeArgs(cached_program.program, cached_program.shared_variables.bcast_kernel_id);
    auto& cached_writer_args =
        GetRuntimeArgs(cached_program.program, cached_program.shared_variables.unary_writer_kernel_id);

    for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_y * num_cores_x; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t Ht_per_core;

        auto& binary_reader_args = cached_reader_args.at(core.x).at(core.y);
        auto& bcast_kernel_args = cached_eltwise_args.at(core.x).at(core.y);
        auto& unary_writer_args = cached_writer_args.at(core.x).at(core.y);

        if (core_group_1.contains(core)) {
            Ht_per_core = Ht_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            Ht_per_core = Ht_per_core_group_2;
        } else {
            binary_reader_args[3] = 0;
            binary_reader_args[7] = 0;
            binary_reader_args[8] = 0;

            bcast_kernel_args[0] = 0;
            bcast_kernel_args[1] = 0;
            bcast_kernel_args[2] = 0;

            unary_writer_args[3] = 0;
            unary_writer_args[4] = 0;
            unary_writer_args[6] = 0;
            unary_writer_args[7] = 0;
            continue;
        }
        const uint32_t num_tensor_tiles_per_core = NC * Ht_per_core * Wt;

        binary_reader_args[0] = src_dram_buffer_a->address();
        binary_reader_args[3] = num_tensor_tiles_per_core;
        binary_reader_args[4] = src_dram_buffer_b->address();
        binary_reader_args[7] = num_btensor_tiles;
        binary_reader_args[8] = num_tensor_tiles_per_core;
        binary_reader_args[9] = NC;
        binary_reader_args[10] = Ht_per_core;
        binary_reader_args[11] = Wt;
        binary_reader_args[12] = bnc1;
        binary_reader_args[13] = num_Wtiles_read;
        binary_reader_args[14] = Ht * Wt;

        bcast_kernel_args[0] = NC;
        bcast_kernel_args[1] = Ht_per_core;
        bcast_kernel_args[2] = Wt;

        unary_writer_args[0] = dst_dram_buffer->address();
        unary_writer_args[3] = Ht_per_core;
        unary_writer_args[4] = Wt;
        unary_writer_args[5] = num_Wtiles_read;
        unary_writer_args[7] = NC;
        unary_writer_args[8] = Ht * Wt;

        num_Wtiles_read += Ht_per_core * Wt;
    }
}

}  // namespace ttnn::operations::data_movement::bcast::program

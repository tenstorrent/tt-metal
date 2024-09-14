// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn {

namespace operations {

namespace matmul {

operation::ProgramWithCallbacks matmul_multi_core(const Tensor &a, const Tensor &b, Tensor &output, bool bcast_batch) {
    tt_metal::Program program{};

    const tt::tt_metal::Shape& ashape = a.get_legacy_shape(), bshape = b.get_legacy_shape();

    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::Buffer *src1_buffer = b.buffer();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device *device = a.device();
    const tt::tt_metal::Shape& cshape = output.get_legacy_shape();  // C=A*B, N1MK*11KN->N1MN

    auto compute_with_storage_grid_size = DeviceComputeWithStorageGridSize(device);
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t c_batch_size = get_batch_size(cshape);
    auto num_output_tiles_total = c_batch_size * cshape[-2] * cshape[-1] / TILE_HW;
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // C = A*B*...
    // MN = MK*KN
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    uint32_t Kt = ashape[-1] / TILE_WIDTH;
    uint32_t Nt = bshape[-1] / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    uint32_t src0_addr = src0_buffer->address();
    uint32_t src1_addr = src1_buffer->address();
    uint32_t dst_addr = dst_buffer->address();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(num_input_tiles * in0_single_tile_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(num_input_tiles * in1_single_tile_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    auto reader_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_args_group_1 = {
        1,                                 // B
        1,                                 // Mt
        Kt,                                // Kt
        num_output_tiles_per_core_group_1  // Nt
    };  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt
        // for simplicity

    auto eltwise_binary_kernel_group_1_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp",
        core_group_1,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_1});

    if (!core_group_2.ranges().empty()) {
        vector<uint32_t> compute_args_group_2 = {
            1,                                 // B
            1,                                 // Mt
            Kt,                                // Kt
            num_output_tiles_per_core_group_2  // Nt
        };  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set
            // Nt for simplicity

        auto eltwise_binary_kernel_group_2_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp",
            core_group_2,
            tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_2});
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }
        tt_metal::SetRuntimeArgs(
            program,
            reader_id,
            core,
            {src0_addr,
             src1_addr,
             Mt,
             Kt,
             Nt,
             MtKt,
             KtNt,
             B,
             uint32_t(bcast_batch),
             num_tiles_written,
             num_output_tiles_per_core,
             MtNt});
        tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, num_output_tiles_per_core, num_tiles_written});
        num_tiles_written += num_output_tiles_per_core;
    }

    auto override_runtime_args_callback =
        [reader_kernel_id = reader_id, writer_kernel_id = writer_id, num_cores, num_cores_y](
            const Program &program,
            const std::vector<tt_metal::Buffer *> &input_buffers,
            const std::vector<tt_metal::Buffer *> &output_buffers) {
            auto src_dram_buffer_a = input_buffers.at(0);
            auto src_dram_buffer_b = input_buffers.at(1);

            auto dst_dram_buffer = output_buffers.at(0);

            for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
                CoreCoord core = {i / num_cores_y, i % num_cores_y};

                {
                    auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[0] = src_dram_buffer_a->address();
                    runtime_args[1] = src_dram_buffer_b->address();
                }

                {
                    auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = dst_dram_buffer->address();
                }
            }
        };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace matmul

}  // namespace operations

}  // namespace ttnn
